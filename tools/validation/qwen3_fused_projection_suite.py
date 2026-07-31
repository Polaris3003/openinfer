#!/usr/bin/env python3
"""Run and summarize the Qwen3 fused-projection validation matrix.

The suite intentionally keeps raw command logs and JSON reports separate from
the derived decision. A zero process exit code is necessary but not sufficient:
the summarizer also validates matrix completeness, benchmark metadata, numeric
report coverage, and the performance thresholds documented in the project plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODES = {
    "split": ("split", "split"),
    "qkv": ("fused", "split"),
    "gate-up": ("split", "fused"),
    "both": ("fused", "fused"),
}
MODE_ORDER = ("split", "qkv", "gate-up", "both", "both", "gate-up", "qkv", "split")
PROFILES = {
    "prefill": {"prompt_len": 10_000, "output_len": 1, "metric": ("ttft_ms", "p50_ms")},
    "decode": {
        "prompt_len": 1_024,
        "output_len": 256,
        "metric": ("steady_tpot_ms", "p50_ms"),
    },
}
DECODE_SHAPES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_SHAPES = (128, 512, 1_024, 2_048, 4_096, 8_192, 10_000)
SCHEMA = 1
REQUIRED_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
QWEN3_UNIT_PACKAGES = ("openinfer-kernels", "openinfer-qwen3", "openinfer-server")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def split_csv(value: str, *, allowed: set[str] | None = None) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空")
    if allowed is not None:
        unknown = set(values) - allowed
        if unknown:
            raise argparse.ArgumentTypeError(f"不支持的值: {sorted(unknown)}")
    return values


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def lora_fixture_targets(path: pathlib.Path) -> set[str]:
    with path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"{path} 不是有效的 safetensors 文件")
        header_len = int.from_bytes(header_len_bytes, "little")
        header = json.loads(handle.read(header_len))
    metadata = header.get("__metadata__", {})
    raw_targets = metadata.get("target_modules")
    if raw_targets is None:
        raise ValueError(f"{path} 缺少 target_modules metadata")
    targets = json.loads(raw_targets)
    if not isinstance(targets, list) or not all(isinstance(value, str) for value in targets):
        raise ValueError(f"{path} 的 target_modules metadata 非法")
    return set(targets)


def check_lora_fixture(path: pathlib.Path) -> None:
    targets = lora_fixture_targets(path)
    missing = REQUIRED_LORA_TARGETS - targets
    if missing:
        raise ValueError(
            f"{path} 缺少 LoRA targets {sorted(missing)}；请先重新生成五 projection fixture"
        )


def command_text(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def qwen3_unit_test_command() -> list[str]:
    command = ["cargo", "test", "--release"]
    for package in QWEN3_UNIT_PACKAGES:
        command.extend(["-p", package])
    command.append("--lib")
    return command


def probe(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command_text(command), "available": False, "output": None}
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command_text(command),
        "available": True,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def model_hash(model_path: pathlib.Path, method: str) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file()
        and (
            path.suffix == ".safetensors"
            or path.name
            in {
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            }
        )
    )
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in candidates:
        stat = path.stat()
        relative = path.relative_to(model_path).as_posix()
        files.append({"path": relative, "bytes": stat.st_size})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        if method == "full":
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
    return {"method": method, "sha256": digest.hexdigest(), "files": files}


def gpu_snapshot() -> dict[str, Any]:
    return probe(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,"
            "clocks.current.sm,clocks.current.memory,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )


def monitor_gpus(stop: threading.Event, result: dict[str, Any]) -> None:
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,power.draw,clocks.current.sm,clocks.current.memory,"
        "temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    per_gpu: dict[str, dict[str, float]] = {}
    samples = 0
    while not stop.is_set():
        completed = subprocess.run(
            query,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 6:
                    continue
                try:
                    index, memory, power, sm_clock, memory_clock, temperature = fields
                    values = {
                        "peak_memory_mib": float(memory),
                        "peak_power_w": float(power),
                        "min_sm_clock_mhz": float(sm_clock),
                        "max_sm_clock_mhz": float(sm_clock),
                        "min_memory_clock_mhz": float(memory_clock),
                        "max_memory_clock_mhz": float(memory_clock),
                        "peak_temperature_c": float(temperature),
                    }
                except ValueError:
                    continue
                current = per_gpu.setdefault(index, values.copy())
                current["peak_memory_mib"] = max(
                    current["peak_memory_mib"], values["peak_memory_mib"]
                )
                current["peak_power_w"] = max(current["peak_power_w"], values["peak_power_w"])
                current["min_sm_clock_mhz"] = min(
                    current["min_sm_clock_mhz"], values["min_sm_clock_mhz"]
                )
                current["max_sm_clock_mhz"] = max(
                    current["max_sm_clock_mhz"], values["max_sm_clock_mhz"]
                )
                current["min_memory_clock_mhz"] = min(
                    current["min_memory_clock_mhz"], values["min_memory_clock_mhz"]
                )
                current["max_memory_clock_mhz"] = max(
                    current["max_memory_clock_mhz"], values["max_memory_clock_mhz"]
                )
                current["peak_temperature_c"] = max(
                    current["peak_temperature_c"], values["peak_temperature_c"]
                )
            samples += 1
        stop.wait(1.0)
    result.update({"samples": samples, "gpus": per_gpu})


class Suite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = pathlib.Path(args.output_dir).resolve()
        self.logs_dir = self.output_dir / "logs"
        self.raw_dir = self.output_dir / "raw"
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "suite": "qwen3-fused-projection-parity",
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
            "root": str(ROOT),
            "model_path": str(pathlib.Path(args.model_path).resolve()),
            "arguments": vars(args),
            "environment": {},
            "commands": [],
        }

    def initialize(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        model_path = pathlib.Path(self.args.model_path).resolve()
        if not model_path.joinpath("config.json").is_file() and not self.args.dry_run:
            raise SystemExit(f"模型目录缺少 config.json: {model_path}")
        self.manifest["environment"] = {
            "captured_at": utc_now(),
            "platform": platform.platform(),
            "python": sys.version,
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_status_porcelain": git_output("status", "--porcelain=v1"),
            "rustc": probe(["rustc", "--version", "--verbose"]),
            "cargo": probe(["cargo", "--version"]),
            "nvcc": probe(["nvcc", "--version"]),
            "nvidia_smi": probe(["nvidia-smi"]),
            "gpu_before_suite": gpu_snapshot(),
            "cuda_env": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("CUDA", "NCCL", "OPENINFER_CUDA"))
            },
            "model": (
                {"method": "dry-run", "sha256": None, "files": []}
                if self.args.dry_run
                else model_hash(model_path, self.args.model_hash)
            ),
        }
        atomic_json(self.manifest_path, self.manifest)

    def run(
        self,
        tag: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        output_path: pathlib.Path | None = None,
    ) -> bool:
        index = len(self.manifest["commands"]) + 1
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        log_path = self.logs_dir / f"{index:03d}-{safe_tag}.log"
        entry: dict[str, Any] = {
            "index": index,
            "tag": tag,
            "command": command,
            "command_text": command_text(command),
            "cwd": str(ROOT),
            "env": env or {},
            "metadata": metadata or {},
            "log": str(log_path.relative_to(self.output_dir)),
            "output": (
                str(output_path.relative_to(self.output_dir)) if output_path is not None else None
            ),
            "started_at": utc_now(),
            "finished_at": None,
            "duration_s": None,
            "returncode": None,
            "status": "planned" if self.args.dry_run else "running",
        }
        if metadata and metadata.get("kind") == "benchmark":
            entry["gpu_before"] = gpu_snapshot()
        self.manifest["commands"].append(entry)
        atomic_json(self.manifest_path, self.manifest)
        print(f"[{index:03d}] {tag}\n  {entry['command_text']}", flush=True)
        if self.args.dry_run:
            log_path.write_text(entry["command_text"] + "\n")
            entry["finished_at"] = utc_now()
            entry["duration_s"] = 0.0
            atomic_json(self.manifest_path, self.manifest)
            return True

        child_env = os.environ.copy()
        child_env.update(env or {})
        started = time.monotonic()
        monitor_stop: threading.Event | None = None
        monitor_result: dict[str, Any] | None = None
        monitor_thread: threading.Thread | None = None
        if metadata and metadata.get("kind") == "benchmark":
            monitor_stop = threading.Event()
            monitor_result = {}
            monitor_thread = threading.Thread(
                target=monitor_gpus,
                args=(monitor_stop, monitor_result),
                name=f"gpu-monitor-{index}",
                daemon=True,
            )
            monitor_thread.start()
        with log_path.open("w") as log:
            log.write(f"$ {entry['command_text']}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            tail: list[str] = []
            for line in process.stdout:
                log.write(line)
                tail.append(line.rstrip())
                tail = tail[-20:]
            returncode = process.wait()
        if monitor_stop is not None and monitor_thread is not None:
            monitor_stop.set()
            monitor_thread.join(timeout=5.0)
            entry["gpu_during"] = monitor_result
        entry["returncode"] = returncode
        entry["status"] = "passed" if returncode == 0 else "failed"
        entry["finished_at"] = utc_now()
        entry["duration_s"] = round(time.monotonic() - started, 3)
        if metadata and metadata.get("kind") == "benchmark":
            entry["gpu_after"] = gpu_snapshot()
        atomic_json(self.manifest_path, self.manifest)
        if returncode != 0:
            print("\n".join(f"  | {line}" for line in tail), file=sys.stderr)
            print(f"  FAILED，完整日志: {log_path}", file=sys.stderr)
        return returncode == 0

    def correctness(self) -> bool:
        commands = [
            (
                "lora-fixture-targets",
                [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve()),
                    "check-fixture",
                    "--path",
                    str(ROOT / "test_data/qwen3-4b-lora-golden.safetensors"),
                ],
                {"kind": "correctness", "gate": "lora-fixture-targets"},
            ),
            (
                "unit-qwen3",
                qwen3_unit_test_command(),
                {"kind": "correctness", "gate": "qwen3-unit"},
            ),
            (
                "operator-qkv-split",
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    "openinfer-kernels",
                    "split_qkv_is_a_bitwise_copy_for_tp_shapes_and_tails",
                    "--lib",
                    "--",
                    "--nocapture",
                ],
                {"kind": "correctness", "gate": "qkv-split"},
            ),
            (
                "operator-swiglu",
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    "openinfer-kernels",
                    "silu_mul_fused_matches_split_bf16_rounding",
                    "--lib",
                    "--",
                    "--nocapture",
                ],
                {"kind": "correctness", "gate": "swiglu"},
            ),
        ]
        for tag, command, metadata in commands:
            if not self.run(tag, command, metadata=metadata) and self.args.fail_fast:
                return False

        model_path = str(pathlib.Path(self.args.model_path).resolve())
        for test_name in ("hf_golden_gate", "lora_golden_gate"):
            for tp in self.args.tp_sizes:
                for mode in MODES:
                    env = {
                        "OPENINFER_TEST_MODEL_PATH": model_path,
                        "OPENINFER_GOLDEN_TP_SIZE": str(tp),
                        "OPENINFER_QWEN3_PROJECTION_FUSION": mode,
                    }
                    metadata = {
                        "kind": "correctness",
                        "gate": test_name,
                        "tp_size": tp,
                        "mode": mode,
                    }
                    ok = self.run(
                        f"{test_name}-tp{tp}-{mode}",
                        [
                            "cargo",
                            "test",
                            "--release",
                            "-p",
                            "openinfer-qwen3",
                            "--test",
                            test_name,
                            "--",
                            "--nocapture",
                        ],
                        env=env,
                        metadata=metadata,
                    )
                    if not ok and self.args.fail_fast:
                        return False
        return True

    def projection_reports(self) -> bool:
        model_path = str(pathlib.Path(self.args.model_path).resolve())
        shapes = ",".join(str(value) for value in (*DECODE_SHAPES, *PREFILL_SHAPES))
        for tp in self.args.tp_sizes:
            for rank in range(tp):
                output = self.raw_dir / f"projection-tp{tp}-rank{rank}.json"
                metadata = {
                    "kind": "projection-report",
                    "tp_size": tp,
                    "rank": rank,
                    "expected_shapes": list((*DECODE_SHAPES, *PREFILL_SHAPES)),
                }
                ok = self.run(
                    f"projection-report-tp{tp}-rank{rank}",
                    [
                        "cargo",
                        "run",
                        "--release",
                        "-p",
                        "openinfer-qwen3",
                        "--features",
                        "projection-report",
                        "--bin",
                        "qwen3_projection_report",
                        "--",
                        "--model-path",
                        model_path,
                        "--tp-size",
                        str(tp),
                        "--rank",
                        str(rank),
                        "--shapes",
                        shapes,
                        "--warmup",
                        str(self.args.projection_warmup),
                        "--iters",
                        str(self.args.projection_iters),
                        "--out",
                        str(output),
                    ],
                    metadata=metadata,
                    output_path=output,
                )
                if not ok and self.args.fail_fast:
                    return False
        return True

    def topology_reports(self) -> bool:
        model_path = str(pathlib.Path(self.args.model_path).resolve())
        for mode, (qkv, gate_up) in MODES.items():
            for batch in self.args.topology_batches:
                output = self.raw_dir / f"topology-{mode}-n{batch}.json"
                metadata = {
                    "kind": "topology-report",
                    "mode": mode,
                    "batch_size": batch,
                }
                ok = self.run(
                    f"topology-{mode}-n{batch}",
                    [
                        "cargo",
                        "run",
                        "--release",
                        "-p",
                        "openinfer-qwen3",
                        "--features",
                        "kernel-report",
                        "--bin",
                        "qwen3_model_report",
                        "--",
                        "decode",
                        "--batch-size",
                        str(batch),
                        "--kv-len",
                        str(self.args.topology_kv_len),
                        "--format",
                        "json",
                        "--model-path",
                        model_path,
                        "--policy",
                        "tuned",
                        "--qkv-fusion",
                        qkv,
                        "--gate-up-fusion",
                        gate_up,
                        "--iters",
                        str(self.args.topology_iters),
                        "--out",
                        str(output),
                    ],
                    metadata=metadata,
                    output_path=output,
                )
                if not ok and self.args.fail_fast:
                    return False
        return True

    def benchmarks(self) -> bool:
        model_path = str(pathlib.Path(self.args.model_path).resolve())
        for tp in self.args.tp_sizes:
            for profile_name, profile in PROFILES.items():
                for concurrency in self.args.concurrency:
                    seen: dict[str, int] = {mode: 0 for mode in MODES}
                    for mode in MODE_ORDER:
                        seen[mode] += 1
                        repeat = seen[mode]
                        qkv, gate_up = MODES[mode]
                        output = (
                            self.raw_dir
                            / f"bench-tp{tp}-{profile_name}-c{concurrency}-{mode}-r{repeat}.json"
                        )
                        metadata = {
                            "kind": "benchmark",
                            "tp_size": tp,
                            "profile": profile_name,
                            "concurrency": concurrency,
                            "mode": mode,
                            "repeat": repeat,
                            "primary_metric": ".".join(profile["metric"]),
                        }
                        ok = self.run(
                            f"bench-tp{tp}-{profile_name}-c{concurrency}-{mode}-r{repeat}",
                            [
                                "cargo",
                                "run",
                                "--release",
                                "-p",
                                "openinfer-server",
                                "--bin",
                                "bench_serving",
                                "--",
                                "--model-path",
                                model_path,
                                "--tp-size",
                                str(tp),
                                "--qwen3-qkv-fusion",
                                qkv,
                                "--qwen3-gate-up-fusion",
                                gate_up,
                                "--format",
                                "json",
                                "--out",
                                str(output),
                                "--label",
                                f"fused-proj/{profile_name}/tp{tp}/c{concurrency}/{mode}/r{repeat}",
                                "request",
                                "--prompt-len",
                                str(profile["prompt_len"]),
                                "--output-len",
                                str(profile["output_len"]),
                                "--concurrency",
                                str(concurrency),
                                "--warmup",
                                str(self.args.warmup),
                                "--iters",
                                str(self.args.iters),
                                "--seed",
                                str(self.args.seed),
                            ],
                            metadata=metadata,
                            output_path=output,
                        )
                        if not ok and self.args.fail_fast:
                            return False
        return True

    def finish(self, success: bool) -> None:
        self.manifest["finished_at"] = utc_now()
        self.manifest["status"] = (
            "planned"
            if self.args.dry_run
            else ("commands-passed" if success else "command-failed")
        )
        self.manifest["environment"]["gpu_after_suite"] = gpu_snapshot()
        atomic_json(self.manifest_path, self.manifest)


def read_json(path: pathlib.Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def nested_number(value: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = value
    for component in path:
        if not isinstance(current, dict) or component not in current:
            raise ValueError(f"缺少 JSON 字段 {'.'.join(path)}")
        current = current[component]
    if not isinstance(current, (int, float)):
        raise ValueError(f"JSON 字段 {'.'.join(path)} 不是数字")
    return float(current)


def percent_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError(f"baseline 必须 > 0，实际为 {baseline}")
    return (baseline - candidate) / baseline * 100.0


def load_command_output(output_dir: pathlib.Path, entry: dict[str, Any]) -> Any:
    relative = entry.get("output")
    if not relative:
        raise ValueError(f"{entry['tag']} 没有声明 output")
    path = output_dir / relative
    if not path.is_file():
        raise ValueError(f"{entry['tag']} 缺少产物 {path}")
    return read_json(path)


def summarize_correctness(commands: list[dict[str, Any]], tp_sizes: list[int]) -> dict[str, Any]:
    entries = [entry for entry in commands if entry["metadata"].get("kind") == "correctness"]
    required = {
        ("lora-fixture-targets", None, None),
        ("qwen3-unit", None, None),
        ("qkv-split", None, None),
        ("swiglu", None, None),
    }
    required.update(
        (gate, tp, mode)
        for gate in ("hf_golden_gate", "lora_golden_gate")
        for tp in tp_sizes
        for mode in MODES
    )
    actual: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for entry in entries:
        metadata = entry["metadata"]
        key = (metadata["gate"], metadata.get("tp_size"), metadata.get("mode"))
        actual[key] = entry
    missing = sorted(required - actual.keys(), key=str)
    failed = [
        {"gate": key, "returncode": actual[key].get("returncode"), "log": actual[key]["log"]}
        for key in required & actual.keys()
        if actual[key].get("returncode") != 0
    ]
    return {
        "passed": not missing and not failed,
        "required_cells": len(required),
        "observed_cells": len(required & actual.keys()),
        "missing": missing,
        "failed": failed,
    }


def summarize_projection(
    output_dir: pathlib.Path, commands: list[dict[str, Any]], tp_sizes: list[int]
) -> dict[str, Any]:
    entries = [entry for entry in commands if entry["metadata"].get("kind") == "projection-report"]
    expected = {(tp, rank) for tp in tp_sizes for rank in range(tp)}
    actual = {(entry["metadata"]["tp_size"], entry["metadata"]["rank"]): entry for entry in entries}
    missing = sorted(expected - actual.keys())
    failed: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    aggregate_samples: dict[tuple[int, str, str, int, int], list[float]] = {}
    for key in sorted(expected & actual.keys()):
        entry = actual[key]
        if entry.get("returncode") != 0:
            failed.append({"cell": key, "reason": "command_failed", "log": entry["log"]})
            continue
        try:
            report = load_command_output(output_dir, entry)
            observed_shapes = {cell["tokens"] for cell in report["cells"]}
            expected_shapes = set(entry["metadata"]["expected_shapes"])
            finite = all(
                projection["delta"]["nan_or_inf"] == 0
                for cell in report["cells"]
                for layer in cell["layers"]
                for projection in (layer["qkv"], layer["gate_up"], layer["swiglu"])
            )
            if observed_shapes != expected_shapes or not finite:
                raise ValueError(
                    f"shape coverage={sorted(observed_shapes)}, finite={finite}, "
                    f"expected={sorted(expected_shapes)}"
                )
            for cell in report["cells"]:
                phase = "decode" if cell["phase"] == "decode" else "prefill_unified"
                for layer in cell["layers"]:
                    aggregate_samples.setdefault(
                        (key[0], phase, "qkv", key[1], cell["tokens"]), []
                    ).append(float(layer["qkv"]["improvement_pct"]))
                    aggregate_samples.setdefault(
                        (key[0], phase, "gate_up", key[1], cell["tokens"]), []
                    ).append(float(layer["swiglu"]["improvement_pct"]))
            reports.append(
                {
                    "tp_size": key[0],
                    "rank": key[1],
                    "path": entry["output"],
                    "cells": len(report["cells"]),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failed.append({"cell": key, "reason": str(error), "log": entry["log"]})
    aggregate_direction: list[dict[str, Any]] = []
    for tp in tp_sizes:
        for phase in ("decode", "prefill_unified"):
            for projection in ("qkv", "gate_up"):
                shape_rank_means = {
                    f"n{tokens}/rank{rank}": statistics.fmean(samples)
                    for (sample_tp, sample_phase, sample_projection, rank, tokens), samples in (
                        aggregate_samples.items()
                    )
                    if sample_tp == tp
                    and sample_phase == phase
                    and sample_projection == projection
                }
                rank_means = {}
                for rank in range(tp):
                    values = [
                        value
                        for key_name, value in shape_rank_means.items()
                        if key_name.endswith(f"/rank{rank}")
                    ]
                    if values:
                        rank_means[rank] = statistics.fmean(values)
                complete = len(rank_means) == tp
                mean = statistics.fmean(rank_means.values()) if rank_means else None
                aggregate_direction.append(
                    {
                        "tp_size": tp,
                        "phase": phase,
                        "projection": projection,
                        "shape_rank_mean_improvement_pct": shape_rank_means,
                        "rank_mean_improvement_pct": rank_means,
                        "mean_improvement_pct": mean,
                        "direction_pass": bool(
                            complete
                            and mean is not None
                            and mean > 0.0
                            and all(value > 0.0 for value in rank_means.values())
                            and all(value > 0.0 for value in shape_rank_means.values())
                        ),
                    }
                )
    return {
        "passed": not missing and not failed,
        "required_cells": len(expected),
        "missing": missing,
        "failed": failed,
        "reports": reports,
        "aggregate_direction": aggregate_direction,
    }


def summarize_topology(
    output_dir: pathlib.Path, commands: list[dict[str, Any]], batches: list[int]
) -> dict[str, Any]:
    entries = [entry for entry in commands if entry["metadata"].get("kind") == "topology-report"]
    expected = {(mode, batch) for mode in MODES for batch in batches}
    actual = {
        (entry["metadata"]["mode"], entry["metadata"]["batch_size"]): entry for entry in entries
    }
    missing = sorted(expected - actual.keys())
    failed: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for key in sorted(expected & actual.keys()):
        entry = actual[key]
        if entry.get("returncode") != 0:
            failed.append({"cell": key, "reason": "command_failed", "log": entry["log"]})
            continue
        try:
            report = load_command_output(output_dir, entry)
            qkv, gate_up = MODES[key[0]]
            if report["config"]["batch_size"] != key[1]:
                raise ValueError("topology report batch size mismatch")
            if report["config"]["qkv_fusion"] != qkv:
                raise ValueError("topology report QKV mode mismatch")
            if report["config"]["gate_up_fusion"] != gate_up:
                raise ValueError("topology report gate/up mode mismatch")
            if not report.get("schedule"):
                raise ValueError("topology report schedule is empty")
            reports.append({"mode": key[0], "batch_size": key[1], "path": entry["output"]})
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failed.append({"cell": key, "reason": str(error), "log": entry["log"]})
    return {
        "passed": not missing and not failed,
        "required_cells": len(expected),
        "missing": missing,
        "failed": failed,
        "reports": reports,
    }


def benchmark_index(
    output_dir: pathlib.Path, commands: list[dict[str, Any]]
) -> tuple[dict[tuple[int, str, int, str, int], dict[str, Any]], list[dict[str, Any]]]:
    index: dict[tuple[int, str, int, str, int], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for entry in commands:
        metadata = entry["metadata"]
        if metadata.get("kind") != "benchmark":
            continue
        key = (
            metadata["tp_size"],
            metadata["profile"],
            metadata["concurrency"],
            metadata["mode"],
            metadata["repeat"],
        )
        if entry.get("returncode") != 0:
            errors.append({"cell": key, "reason": "command_failed", "log": entry["log"]})
            continue
        try:
            report = load_command_output(output_dir, entry)
            expected_fusion = (
                f"qkv={MODES[key[3]][0]},gate_up={MODES[key[3]][1]}"
            )
            if report["run"]["tp_size"] != key[0]:
                raise ValueError("report TP metadata mismatch")
            if report["run"].get("qwen3_projection_fusion") != expected_fusion:
                raise ValueError(
                    "report fusion metadata mismatch: "
                    f"{report['run'].get('qwen3_projection_fusion')} != {expected_fusion}"
                )
            profile = PROFILES[key[1]]
            if report["workload"]["prompt"]["prompt_tokens"] != profile["prompt_len"]:
                raise ValueError("report prompt length mismatch")
            if report["workload"]["output_len"] != profile["output_len"]:
                raise ValueError("report output length mismatch")
            if report["workload"]["concurrency"] != key[2]:
                raise ValueError("report concurrency mismatch")
            metric = nested_number(report["metrics"], profile["metric"])
            throughput_name = "decode_tok_s" if key[1] == "decode" else "request_tok_s"
            throughput = nested_number(report["metrics"], (throughput_name,))
            gpu_during = entry.get("gpu_during") or {}
            if gpu_during.get("samples", 0) <= 0 or not gpu_during.get("gpus"):
                raise ValueError("benchmark 缺少运行期间 GPU clocks/power/peak-memory 采样")
            index[key] = {
                "metric": metric,
                "throughput": throughput,
                "path": entry["output"],
                "report": report,
                "gpu_during": gpu_during,
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append({"cell": key, "reason": str(error), "log": entry["log"]})
    return index, errors


def performance_decisions(
    index: dict[tuple[int, str, int, str, int], dict[str, Any]],
    tp_sizes: list[int],
    concurrency: list[int],
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    required = {
        (tp, profile, conc, mode, repeat)
        for tp in tp_sizes
        for profile in PROFILES
        for conc in concurrency
        for mode in MODES
        for repeat in (1, 2)
    }
    missing = sorted(required - index.keys())
    decisions: list[dict[str, Any]] = []
    for tp in tp_sizes:
        for profile in PROFILES:
            threshold = 2.0 if profile == "decode" else 3.0
            for projection, mode in (("qkv", "qkv"), ("gate_up", "gate-up")):
                comparisons: list[dict[str, Any]] = []
                for conc in concurrency:
                    for repeat in (1, 2):
                        baseline = index.get((tp, profile, conc, "split", repeat))
                        candidate = index.get((tp, profile, conc, mode, repeat))
                        if baseline is None or candidate is None:
                            continue
                        delta = percent_improvement(baseline["metric"], candidate["metric"])
                        throughput_delta = (
                            (candidate["throughput"] - baseline["throughput"])
                            / baseline["throughput"]
                            * 100.0
                        )
                        comparisons.append(
                            {
                                "concurrency": conc,
                                "repeat": repeat,
                                "baseline": baseline["metric"],
                                "candidate": candidate["metric"],
                                "improvement_pct": delta,
                                "baseline_throughput": baseline["throughput"],
                                "candidate_throughput": candidate["throughput"],
                                "throughput_improvement_pct": throughput_delta,
                            }
                        )
                complete = len(comparisons) == len(concurrency) * 2
                improvements = [row["improvement_pct"] for row in comparisons]
                direction_consistent = complete and all(value > 0.0 for value in improvements)
                no_large_regression = complete and all(value >= -threshold for value in improvements)
                throughput_consistent = complete and all(
                    row["throughput_improvement_pct"] >= -threshold for row in comparisons
                )
                mean_improvement = statistics.fmean(improvements) if improvements else None
                performance_pass = bool(
                    complete
                    and direction_consistent
                    and no_large_regression
                    and throughput_consistent
                    and mean_improvement is not None
                    and mean_improvement >= threshold
                )
                decisions.append(
                    {
                        "projection": projection,
                        "phase": "decode" if profile == "decode" else "prefill_unified",
                        "tp_size": tp,
                        "threshold_pct": threshold,
                        "mean_improvement_pct": mean_improvement,
                        "direction_consistent": direction_consistent,
                        "no_large_regression": no_large_regression,
                        "throughput_consistent": throughput_consistent,
                        "performance_pass": performance_pass,
                        "comparisons": comparisons,
                    }
                )
    return decisions, missing


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 fused projection 验证报告",
        "",
        f"> **TL;DR:** 总体结论为 **{summary['overall_status']}**。只有同时通过正确性、"
        "数值报告完整性和性能门槛的组合才可进入生产白名单。",
        "",
        "## 运行身份",
        "",
        f"- commit：`{summary['environment'].get('git_commit', '')}`",
        f"- model hash：`{summary['environment'].get('model', {}).get('sha256')}` "
        f"（{summary['environment'].get('model', {}).get('method')}）",
        f"- manifest：`manifest.json`",
        "",
        "## 硬门禁",
        "",
        "| 门禁 | 结果 | 详情 |",
        "| --- | --- | --- |",
        f"| 正确性矩阵 | {'PASS' if summary['correctness']['passed'] else 'FAIL'} | "
        f"{summary['correctness']['observed_cells']}/{summary['correctness']['required_cells']} cells |",
        f"| projection 数值报告 | {'PASS' if summary['projection']['passed'] else 'FAIL'} | "
        f"{len(summary['projection']['reports'])}/{summary['projection']['required_cells']} rank reports |",
        f"| decode topology 报告 | {'PASS' if summary['topology']['passed'] else 'FAIL'} | "
        f"{len(summary['topology']['reports'])}/{summary['topology']['required_cells']} cells |",
        f"| E2E 性能矩阵 | {'PASS' if not summary['benchmark']['missing'] and not summary['benchmark']['errors'] else 'INCOMPLETE'} | "
        f"missing={len(summary['benchmark']['missing'])}, errors={len(summary['benchmark']['errors'])} |",
        "",
        "## 独立白名单决策",
        "",
        "| Projection | Phase | TP | E2E 平均改善 | 阈值 | 复测方向一致 | Kernel 同向 | 正确性 | 结论 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["decisions"]:
        mean = row["mean_improvement_pct"]
        mean_text = "N/A" if mean is None else f"{mean:.2f}%"
        lines.append(
            f"| {row['projection']} | {row['phase']} | {row['tp_size']} | {mean_text} | "
            f"{row['threshold_pct']:.1f}% | {'是' if row['direction_consistent'] else '否'} | "
            f"{'PASS' if row['kernel_direction_pass'] else 'FAIL'} | "
            f"{'PASS' if row['correctness_pass'] else 'FAIL'} | **{row['decision']}** |"
        )
    lines.extend(
        [
            "",
            "## 未完成或失败项",
            "",
            "```json",
            json.dumps(
                {
                    "correctness_missing": summary["correctness"]["missing"],
                    "correctness_failed": summary["correctness"]["failed"],
                    "projection_missing": summary["projection"]["missing"],
                    "projection_failed": summary["projection"]["failed"],
                    "topology_missing": summary["topology"]["missing"],
                    "topology_failed": summary["topology"]["failed"],
                    "benchmark_missing": summary["benchmark"]["missing"],
                    "benchmark_errors": summary["benchmark"]["errors"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## 解释规则",
            "",
            "- `ENABLE` 不是“某次更快”，而是对应 projection/phase/TP 的正确性和性能均通过。",
            "- `KEEP_SPLIT` 表示证据否决或证据不完整；它不会自动修改生产 Auto 白名单。",
            "- `both` 模式保留在原始矩阵中用于观察交互，但独立白名单归因使用 QKV-only 和 gate-up-only。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(output_dir: pathlib.Path) -> dict[str, Any]:
    manifest = read_json(output_dir / "manifest.json")
    commands = manifest["commands"]
    arguments = manifest["arguments"]
    tp_sizes = [int(value) for value in arguments["tp_sizes"]]
    concurrency = [int(value) for value in arguments["concurrency"]]
    correctness = summarize_correctness(commands, tp_sizes)
    projection = summarize_projection(output_dir, commands, tp_sizes)
    topology = summarize_topology(
        output_dir, commands, [int(value) for value in arguments["topology_batches"]]
    )
    bench_index, bench_errors = benchmark_index(output_dir, commands)
    decisions, bench_missing = performance_decisions(bench_index, tp_sizes, concurrency)
    for decision in decisions:
        mode = "qkv" if decision["projection"] == "qkv" else "gate-up"
        relevant = [
            entry
            for entry in commands
            if entry["metadata"].get("kind") == "correctness"
            and entry["metadata"].get("tp_size") == decision["tp_size"]
            and entry["metadata"].get("mode") in {"split", mode}
            and entry["metadata"].get("gate") in {"hf_golden_gate", "lora_golden_gate"}
        ]
        decision["correctness_pass"] = (
            len(relevant) == 4 and all(entry.get("returncode") == 0 for entry in relevant)
        )
        decision["projection_report_pass"] = projection["passed"]
        kernel_direction = next(
            (
                row
                for row in projection["aggregate_direction"]
                if row["projection"] == decision["projection"]
                and row["phase"] == decision["phase"]
                and row["tp_size"] == decision["tp_size"]
            ),
            None,
        )
        decision["kernel_direction_pass"] = bool(
            kernel_direction and kernel_direction["direction_pass"]
        )
        decision["kernel_mean_improvement_pct"] = (
            kernel_direction["mean_improvement_pct"] if kernel_direction else None
        )
        decision["decision"] = (
            "ENABLE"
            if decision["performance_pass"]
            and decision["correctness_pass"]
            and decision["projection_report_pass"]
            and decision["kernel_direction_pass"]
            else "KEEP_SPLIT"
        )
    evidence_complete = (
        set(tp_sizes) == {1, 2}
        and set(concurrency) == {1, 8}
        and correctness["passed"]
        and projection["passed"]
        and topology["passed"]
        and not bench_missing
        and not bench_errors
    )
    summary = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "overall_status": "COMPLETE" if evidence_complete else "INCOMPLETE_OR_FAILED",
        "environment": manifest["environment"],
        "correctness": correctness,
        "projection": projection,
        "topology": topology,
        "benchmark": {
            "required_cells_with_repeats": (
                len(tp_sizes) * len(PROFILES) * len(concurrency) * len(MODES) * 2
            ),
            "observed_cells_with_repeats": len(bench_index),
            "missing": bench_missing,
            "errors": bench_errors,
        },
        "decisions": decisions,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "decision-table.json", decisions)
    (output_dir / "report.md").write_text(markdown_report(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3 fused projection 正确性、数值和 32-cell 性能验证套件"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="运行验证并落盘所有原始产物")
    run.add_argument("--model-path", default="models/Qwen3-4B")
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--sections",
        default="correctness,projection,topology,benchmark",
        help="逗号分隔：correctness,projection,topology,benchmark",
    )
    run.add_argument("--tp-sizes", default="1,2")
    run.add_argument("--concurrency", default="1,8")
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--iters", type=int, default=20)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--projection-warmup", type=int, default=2)
    run.add_argument("--projection-iters", type=int, default=5)
    run.add_argument("--topology-batches", default="1,8,32,64")
    run.add_argument("--topology-kv-len", type=int, default=2048)
    run.add_argument("--topology-iters", type=int, default=32)
    run.add_argument("--model-hash", choices=("full", "metadata"), default="full")
    fail_fast = run.add_mutually_exclusive_group()
    fail_fast.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    fail_fast.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    run.set_defaults(fail_fast=True)
    run.add_argument("--dry-run", action="store_true")

    summarize_parser = subparsers.add_parser(
        "summarize", help="校验矩阵完整性并生成 summary.json/report.md"
    )
    summarize_parser.add_argument("--output-dir", required=True)

    fixture = subparsers.add_parser(
        "check-fixture", help="验证 LoRA fixture 是否覆盖五个 projection"
    )
    fixture.add_argument("--path", required=True)
    return parser


def normalize_run_args(args: argparse.Namespace) -> None:
    args.sections = split_csv(
        args.sections, allowed={"correctness", "projection", "topology", "benchmark"}
    )
    args.tp_sizes = [int(value) for value in split_csv(args.tp_sizes)]
    if any(value not in (1, 2) for value in args.tp_sizes):
        raise SystemExit("--tp-sizes 目前只支持 1,2")
    args.concurrency = [int(value) for value in split_csv(args.concurrency)]
    if any(value <= 0 for value in args.concurrency):
        raise SystemExit("--concurrency 必须全部 > 0")
    args.topology_batches = [int(value) for value in split_csv(args.topology_batches)]
    for name in (
        "warmup",
        "iters",
        "projection_warmup",
        "projection_iters",
        "topology_iters",
        "topology_kv_len",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} 必须 > 0")


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "check-fixture":
        try:
            check_lora_fixture(pathlib.Path(args.path).resolve())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(error, file=sys.stderr)
            return 2
        print(f"LoRA fixture targets OK: {sorted(REQUIRED_LORA_TARGETS)}")
        return 0
    if args.command == "summarize":
        summary = summarize(pathlib.Path(args.output_dir).resolve())
        print(
            f"报告已生成：{pathlib.Path(args.output_dir).resolve() / 'report.md'} "
            f"({summary['overall_status']})"
        )
        return 0 if summary["overall_status"] == "COMPLETE" else 2

    normalize_run_args(args)
    suite = Suite(args)
    suite.initialize()
    success = True
    section_methods = {
        "correctness": suite.correctness,
        "projection": suite.projection_reports,
        "topology": suite.topology_reports,
        "benchmark": suite.benchmarks,
    }
    try:
        for section in args.sections:
            section_ok = section_methods[section]()
            success = success and section_ok
            if not section_ok and args.fail_fast:
                break
    finally:
        suite.finish(success)
    if args.dry_run:
        print(f"dry-run manifest: {suite.manifest_path}")
        return 0
    summary = summarize(suite.output_dir)
    print(f"最终报告: {suite.output_dir / 'report.md'} ({summary['overall_status']})")
    return 0 if summary["overall_status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
