import contextlib
import importlib.util
import io
import pathlib
import tempfile
import types
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("qwen3_fused_projection_suite.py")
SPEC = importlib.util.spec_from_file_location("qwen3_fused_projection_suite", MODULE_PATH)
SUITE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUITE)


class DecisionTests(unittest.TestCase):
    def benchmark_index(self, qkv_delta, gate_delta):
        index = {}
        for profile in SUITE.PROFILES:
            for concurrency in (1, 8):
                for repeat in (1, 2):
                    baseline = 10.0
                    index[(1, profile, concurrency, "split", repeat)] = {
                        "metric": baseline,
                        "throughput": 100.0,
                    }
                    index[(1, profile, concurrency, "qkv", repeat)] = {
                        "metric": baseline * (1.0 - qkv_delta / 100.0),
                        "throughput": 100.0 * (1.0 + qkv_delta / 100.0),
                    }
                    index[(1, profile, concurrency, "gate-up", repeat)] = {
                        "metric": baseline * (1.0 - gate_delta / 100.0),
                        "throughput": 100.0 * (1.0 + gate_delta / 100.0),
                    }
                    index[(1, profile, concurrency, "both", repeat)] = {
                        "metric": baseline * 0.9,
                        "throughput": 110.0,
                    }
        return index

    def test_thresholds_are_phase_specific(self):
        decisions, missing = SUITE.performance_decisions(
            self.benchmark_index(qkv_delta=2.5, gate_delta=2.5), [1], [1, 8]
        )
        self.assertFalse(missing)
        keyed = {(row["projection"], row["phase"]): row for row in decisions}
        self.assertTrue(keyed[("qkv", "decode")]["performance_pass"])
        self.assertFalse(keyed[("qkv", "prefill_unified")]["performance_pass"])

    def test_repeat_direction_must_be_consistent(self):
        index = self.benchmark_index(qkv_delta=5.0, gate_delta=5.0)
        index[(1, "decode", 8, "qkv", 2)]["metric"] = 10.1
        decisions, _ = SUITE.performance_decisions(index, [1], [1, 8])
        qkv_decode = next(
            row
            for row in decisions
            if row["projection"] == "qkv" and row["phase"] == "decode"
        )
        self.assertFalse(qkv_decode["direction_consistent"])
        self.assertFalse(qkv_decode["performance_pass"])


class FixtureTests(unittest.TestCase):
    def test_missing_targets_fail_closed(self):
        header = {
            "__metadata__": {"target_modules": '["q_proj", "v_proj"]'},
            "dummy": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
        }
        import json

        encoded = json.dumps(header).encode()
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(len(encoded).to_bytes(8, "little"))
            handle.write(encoded)
            handle.flush()
            with self.assertRaises(ValueError):
                SUITE.check_lora_fixture(pathlib.Path(handle.name))


class CommandScopeTests(unittest.TestCase):
    def test_qwen3_unit_gate_does_not_build_unrelated_model_crates(self):
        command = SUITE.qwen3_unit_test_command()
        self.assertEqual(
            command,
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "openinfer-kernels",
                "-p",
                "openinfer-qwen3",
                "-p",
                "openinfer-server",
                "--lib",
            ],
        )
        self.assertNotIn("--workspace", command)

    def test_projection_report_command_enables_its_cli_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                output_dir=directory,
                model_path="models/Qwen3-4B",
                tp_sizes=[1],
                projection_warmup=2,
                projection_iters=5,
                dry_run=True,
            )
            suite = SUITE.Suite(args)
            suite.logs_dir.mkdir()
            suite.raw_dir.mkdir()

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(suite.projection_reports())

            command = suite.manifest["commands"][0]["command"]
            feature_index = command.index("--features")
            self.assertEqual(command[feature_index + 1], "projection-report")

    def test_topology_reports_use_shared_kv_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                output_dir=directory,
                model_path="models/Qwen3-4B",
                topology_batches=[64],
                topology_kv_len=2_048,
                topology_iters=32,
                dry_run=True,
            )
            suite = SUITE.Suite(args)
            suite.logs_dir.mkdir()
            suite.raw_dir.mkdir()

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(suite.topology_reports())

            self.assertEqual(len(suite.manifest["commands"]), len(SUITE.MODES))
            self.assertTrue(
                all(
                    "--shared-kv-pages" in entry["command"]
                    for entry in suite.manifest["commands"]
                )
            )


class EndToEndSummaryTests(unittest.TestCase):
    def test_complete_synthetic_matrix_generates_enable_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = pathlib.Path(directory)
            raw = output_dir / "raw"
            raw.mkdir()
            commands = []

            def add_entry(tag, metadata, output=None):
                entry = {
                    "tag": tag,
                    "metadata": metadata,
                    "returncode": 0,
                    "log": f"logs/{tag}.log",
                    "output": output,
                }
                commands.append(entry)
                return entry

            for gate in ("lora-fixture-targets", "qwen3-unit", "qkv-split", "swiglu"):
                add_entry(gate, {"kind": "correctness", "gate": gate})
            for gate in ("hf_golden_gate", "lora_golden_gate"):
                for tp in (1, 2):
                    for mode in SUITE.MODES:
                        add_entry(
                            f"{gate}-{tp}-{mode}",
                            {
                                "kind": "correctness",
                                "gate": gate,
                                "tp_size": tp,
                                "mode": mode,
                            },
                        )

            for tp in (1, 2):
                for rank in range(tp):
                    name = f"projection-{tp}-{rank}.json"
                    report = {
                        "cells": [
                            {
                                "phase": phase,
                                "tokens": tokens,
                                "layers": [
                                    {
                                        projection: {
                                            "improvement_pct": 10.0,
                                            "delta": {"nan_or_inf": 0},
                                        }
                                        for projection in ("qkv", "gate_up", "swiglu")
                                    }
                                ],
                            }
                            for phase, tokens in (("decode", 1), ("prefill", 128))
                        ]
                    }
                    SUITE.atomic_json(raw / name, report)
                    add_entry(
                        name,
                        {
                            "kind": "projection-report",
                            "tp_size": tp,
                            "rank": rank,
                            "expected_shapes": [1, 128],
                        },
                        f"raw/{name}",
                    )

            for mode, (qkv, gate_up) in SUITE.MODES.items():
                name = f"topology-{mode}.json"
                SUITE.atomic_json(
                    raw / name,
                    {
                        "config": {
                            "batch_size": 1,
                            "qkv_fusion": qkv,
                            "gate_up_fusion": gate_up,
                        },
                        "schedule": [{"op": "synthetic"}],
                    },
                )
                add_entry(
                    name,
                    {"kind": "topology-report", "mode": mode, "batch_size": 1},
                    f"raw/{name}",
                )

            for tp in (1, 2):
                for profile_name, profile in SUITE.PROFILES.items():
                    for concurrency in (1, 8):
                        for mode, (qkv, gate_up) in SUITE.MODES.items():
                            for repeat in (1, 2):
                                improvement = 0.0 if mode == "split" else 5.0
                                metric = 10.0 * (1.0 - improvement / 100.0)
                                throughput = 100.0 * (1.0 + improvement / 100.0)
                                name = (
                                    f"bench-{tp}-{profile_name}-{concurrency}-{mode}-{repeat}.json"
                                )
                                metrics = {
                                    "ttft_ms": {"p50_ms": metric},
                                    "steady_tpot_ms": {"p50_ms": metric},
                                    "request_tok_s": throughput,
                                    "decode_tok_s": throughput,
                                }
                                SUITE.atomic_json(
                                    raw / name,
                                    {
                                        "run": {
                                            "tp_size": tp,
                                            "qwen3_projection_fusion": (
                                                f"qkv={qkv},gate_up={gate_up}"
                                            ),
                                        },
                                        "workload": {
                                            "prompt": {"prompt_tokens": profile["prompt_len"]},
                                            "output_len": profile["output_len"],
                                            "concurrency": concurrency,
                                        },
                                        "metrics": metrics,
                                    },
                                )
                                entry = add_entry(
                                    name,
                                    {
                                        "kind": "benchmark",
                                        "tp_size": tp,
                                        "profile": profile_name,
                                        "concurrency": concurrency,
                                        "mode": mode,
                                        "repeat": repeat,
                                    },
                                    f"raw/{name}",
                                )
                                entry["gpu_during"] = {
                                    "samples": 1,
                                    "gpus": {"0": {"peak_memory_mib": 1.0}},
                                }

            SUITE.atomic_json(
                output_dir / "manifest.json",
                {
                    "arguments": {
                        "tp_sizes": [1, 2],
                        "concurrency": [1, 8],
                        "topology_batches": [1],
                    },
                    "commands": commands,
                    "environment": {
                        "git_commit": "synthetic",
                        "model": {"sha256": "synthetic", "method": "test"},
                    },
                },
            )
            summary = SUITE.summarize(output_dir)
            self.assertEqual(summary["overall_status"], "COMPLETE")
            self.assertTrue(all(row["decision"] == "ENABLE" for row in summary["decisions"]))
            self.assertTrue((output_dir / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
