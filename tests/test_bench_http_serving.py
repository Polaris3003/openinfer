#!/usr/bin/env python3
"""Regression tests for scripts/bench_http_serving.py."""

from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
import urllib.parse
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_serving.py"
SPEC = importlib.util.spec_from_file_location("bench_http_serving", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench_http_serving = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_http_serving
SPEC.loader.exec_module(bench_http_serving)


class DoneOnlyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: object) -> None:
        return


class BenchHttpServingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), DoneOnlyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = urllib.parse.urlparse(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_done_only_stream_fails_when_tokens_requested(self) -> None:
        result = bench_http_serving.request_once(
            index=0,
            request_id="req-empty",
            url=self.url,
            model="fake-model",
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("without streamed text chunks", result.error or "")
        self.assertEqual(result.output_chunks, 0)

    def test_server_trace_log_is_attached_by_vllm_completion_id_prefix(self) -> None:
        result = bench_http_serving.RequestResult(
            index=0,
            request_id="bench-1",
            ok=True,
            status=200,
            error=None,
            timed_out=False,
            start_s=1.0,
            start_wall_s=100.0,
            first_token_s=1.2,
            first_token_wall_s=100.25,
            end_s=1.4,
            end_wall_s=100.4,
            latency_ms=400.0,
            ttft_ms=200.0,
            tpot_ms=20.0,
            itl_ms=[20.0],
            output_chunks=2,
            output_chars=4,
            output_hash="abcd",
            text_prefix="text",
        )
        line = (
            'INFO pegainfer_http_trace {"request_id":"cmpl-bench-1-generated",'
            '"queued_at_unix_s":100.01,"scheduled_at_unix_s":100.03,'
            '"first_token_emit_unix_s":100.20,"prefill_ms":170.0,'
            '"first_decode_ms":28.0}\\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text(line, encoding="utf-8")
            traces = bench_http_serving.load_server_traces(path)
            bench_http_serving.attach_server_traces([result], traces)

        self.assertIsNotNone(result.server_trace)
        assert result.server_trace is not None
        self.assertAlmostEqual(result.server_trace["admission_queue_ms"], 20.0, places=3)
        self.assertAlmostEqual(result.server_trace["stream_flush_ms"], 50.0, places=3)
        self.assertAlmostEqual(result.server_trace["frontend_to_queue_ms"], 10.0, places=3)


if __name__ == "__main__":
    unittest.main()
