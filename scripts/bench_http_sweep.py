#!/usr/bin/env python3
"""Run a reproducible HTTP serving sweep over concurrency and max_tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH = SCRIPT_DIR / "bench_http_serving.py"


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    parsed = [int(item) for item in items]
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def request_hashes(report: dict[str, Any]) -> list[str]:
    return [request["output_hash"] for request in report["requests"] if request["ok"]]


def run_one(
    args: argparse.Namespace,
    prompt_words: int,
    concurrency: int,
    max_tokens: int,
    repeat: int,
) -> dict[str, Any]:
    out = args.out_dir / f"pw{prompt_words}_c{concurrency}_mt{max_tokens}_r{repeat}.json"
    cmd = [
        sys.executable,
        str(BENCH),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--num-requests",
        str(args.num_requests),
        "--concurrency",
        str(concurrency),
        "--warmup",
        str(args.warmup),
        "--prompt-words",
        str(prompt_words),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(args.temperature),
        "--timeout",
        str(args.timeout),
        "--out",
        str(out),
    ]
    if args.no_ignore_eos:
        cmd.append("--no-ignore-eos")
    if args.server_log:
        cmd.extend(["--server-log", str(args.server_log)])
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def build_summary(args: argparse.Namespace, reports: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for report in reports:
        workload = report["workload"]
        key = (workload["prompt_words"], workload["concurrency"], workload["max_tokens"])
        cells.setdefault(key, []).append(report)

    rows = []
    correctness_ok = True
    for (prompt_words, concurrency, max_tokens), cell_reports in sorted(cells.items()):
        baseline = request_hashes(cell_reports[0])
        stable = True
        for report in cell_reports:
            summary = report["summary"]
            hashes = request_hashes(report)
            if summary["failed"] or summary["timeouts"] or hashes != baseline:
                stable = False
        correctness_ok = correctness_ok and stable
        rows.append(
            {
                "concurrency": concurrency,
                "prompt_words": prompt_words,
                "max_tokens": max_tokens,
                "repeats": len(cell_reports),
                "stable_per_request_hashes": stable,
                "combined_output_hashes": [
                    report["summary"]["combined_output_hash"] for report in cell_reports
                ],
                "qps": [report["summary"]["qps"] for report in cell_reports],
                "ttft_avg_ms": [report["metrics"]["ttft"]["avg_ms"] for report in cell_reports],
                "tpot_avg_ms": [report["metrics"]["tpot"]["avg_ms"] for report in cell_reports],
                "itl_avg_ms": [report["metrics"]["itl"]["avg_ms"] for report in cell_reports],
                "failed": [report["summary"]["failed"] for report in cell_reports],
                "timeouts": [report["summary"]["timeouts"] for report in cell_reports],
                "trace_phases_avg_ms": [
                    {
                        phase: stats["avg_ms"]
                        for phase, stats in report["server_trace"]["phases_ms"].items()
                    }
                    for report in cell_reports
                ],
            }
        )

    return {
        "schema_version": 1,
        "kind": "openai_http_completions_sweep",
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "prompt_words": args.prompt_words,
            "temperature": args.temperature,
            "ignore_eos": not args.no_ignore_eos,
            "timeout_s": args.timeout,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
        },
        "correctness_gate": {
            "passed": correctness_ok,
            "rule": "failed=0, timeout=0, and per-request output_hash list stable across repeats for each prompt_words/concurrency/max_tokens cell",
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt-words", type=parse_int_list, default=[16])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-ignore-eos", action="store_true")
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=parse_int_list, default=[16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for prompt_words in args.prompt_words:
        for max_tokens in args.max_tokens:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    reports.append(run_one(args, prompt_words, concurrency, max_tokens, repeat))

    summary = build_summary(args, reports)
    (args.out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["correctness_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
