# Kimi-K2 TP1 DP8 EP8 performance

> TL;DR: This ledger tracks pegainfer TP1+DP8+EP8 on 8x H20 against the vLLM TP1+DP8+EP8 bs64 target. Every optimization must start from a profile, state the expected gain, show a microbench or isolated measurement, then pass correctness and service-level gates before commit.
>
> Last touched: 2026-05-25

## Target

| Item | Target |
| --- | ---: |
| Hardware | h20-100, 8x NVIDIA H20 |
| Model | `/data/models/Kimi-K2.5` |
| Shape | TP1 DP8 EP8 |
| Workload | prompt_len=1, output_len=128, max_concurrency=64, num_prompts=256 |
| vLLM baseline | output `594.57 tok/s`, TTFT p50/p99 `161.30/303.20ms`, TPOT p50/p99 `107.20/109.20ms`, ITL p50 `108.92ms` |
| Gate | `256/256` success, TPOT p50 `< 107.20ms`, TPOT p99 `< 109.20ms`, output `> 594.57 tok/s` |

Baseline source: h20-100 rerun with explicit bs64 warmup on 2026-05-25:
`/tmp/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`.
The older sweep in `docs/models/kimi-k2/vllm-h20-baseline.md` recorded bs64 TPOT p50/p99
`109.00/109.76ms`; the warmup-after rerun is slightly faster but still the same
100ms-class H20 shape.

## Method

Performance work in this file follows this loop:

1. Profile: record the service JSON/log, in-process JSON, and nsys sqlite/tail report when profiling is needed.
2. Motivation and expected gain: name the bottleneck and estimate the target metric movement.
3. Microbench: isolate the changed stage, or explain why the service/in-process measurement is the smallest meaningful unit.
4. Correctness: keep generated-token hash distributions, mismatch counts, and any relaxed tolerance rationale.
5. Decision: keep, reject, defer, or revert; every kept optimization gets a commit.

For TP1 DP8, correctness checks must include uneven per-rank active rows and empty-rank EP participation, because PPLX collectives still require all ranks to enter each MoE layer in the same order.

## Unified Commands

Build on h20-100:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
/root/.cargo/bin/cargo build --release -p pegainfer-server \
  --features kimi-k2-pplx-ep --bin pegainfer --bin bench_serving
```

In-process bs64:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp1dp8 \
target/release/bench_serving \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi-tp1dp8/tp1dp8_bs64_o128_${COMMIT}.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Service bs64, same client shape as vLLM:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp1dp8 \
target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 8124 --cuda-graph false
```

```bash
source /root/develop/xingming/vllm_test/.venv/bin/activate
vllm bench serve \
  --backend openai \
  --model /data/models/Kimi-K2.5 \
  --tokenizer /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --base-url http://127.0.0.1:8124 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/kimi-tp1dp8-service \
  --result-filename pegainfer_tp1dp8_bs64_${COMMIT}.json
```

nsys profile:

```bash
cd /root/develop/xingming/pegainfer
mkdir -p /tmp/kimi-profile
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp1dp8 \
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --cuda-graph-trace=node --export=sqlite \
  -o /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT} \
  target/release/bench_serving \
    --model-path /data/models/Kimi-K2.5 \
    --cuda-graph false \
    --cuda-profiler-capture \
    --format json \
    --out /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT}.json \
    request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1

uv run --no-project python tools/nsys_tail_stats.py \
  /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT}.sqlite \
  --out /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT}_tail.md
```

## Optimization Log

### O1 - prompt_len=1 admission goes through decode

Status: keep, commit pending.

Profile:

- Code inspection showed TP1 DP8 uses `DpCoordinator`, not the TP8 `KimiK2Scheduler` prompt_len1 batch path.
- Old admission ran each prompt_len=1 request through `synchronized_prefill`, with `decode_batch_size=1`, and padding ranks doing dummy prefill. At bs64 that is 64 synchronized prefill waves.
- Old `MAX_BATCH_PER_DP=4` capped global active requests at 32, so bs64 could not occupy all requested rows.

Motivation and expected gain:

- prompt_len=1 is semantically a decode step at position 0: consume one token, append KV at position 0, produce the first generated token.
- Replace 64 serialized prompt prefill waves with one DP-wide decode admission wave.
- Raise per-DP slots to 8 so TP1 DP8 can hold the full bs64 workload.
- Expected gain: large TTFT reduction and service throughput improvement; TPOT should use rank-local bs8 instead of two bs32 waves.

Change:

- `pegainfer-kimi-k2/src/runner/engine.rs`
  - `MAX_BATCH_PER_DP: 4 -> 8`.
  - Added prompt_len1 admission batching in `DpCoordinator`.
  - For prompt_len1 requests, send `StepCommand::Decode { positions: vec![0], slots, batch_size }` instead of `Prefill`.
  - Empty ranks still run padding decode to preserve PPLX collective order.

Microbench:

- Remote build passed on h20-100.
- Smoke command:

```bash
PEGAINFER_KIMI_PARALLEL=tp1dp8 target/release/bench_serving \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi-tp1dp8/prompt1_decode_admission_bs64_o5_smoke.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

- Smoke result: `64/64` success, `steady_tpot_ms` p50/p95/p99 `35.81/37.07/37.09ms`, first decode step p50 `38.29ms`.

Correctness:

- Smoke generated all 5 tokens for every request without PPLX collective mismatch or slot-state failure.
- bs8/o8 deterministic smoke generated `8/8` full traces with one hash,
  `/tmp/kimi-tp1dp8/prompt1_decode_admission_bs8_o8_correctness.json`.
- Old serial-prefill reference rerun from a detached `8431955` worktree was attempted, but the
  temporary worktree initially lacked the FlashInfer third-party include tree. After fixing that,
  the run was stopped because the current task shifted to rechecking vLLM; no serial-reference
  mismatch claim is made here.

Performance:

- In-process, `/tmp/kimi-tp1dp8/prompt1_decode_admission_bs64_o128_w1_i1.json`:
  `64/64` success, TTFT p50/p99 `73.24/76.42ms`, first decode p50/p99
  `38.09/38.13ms`, steady TPOT p50/p95/p99 `41.26/43.54/44.15ms`.
- Service, same `vllm bench serve` client as vLLM,
  `/tmp/kimi-tp1dp8-service/pegainfer_tp1dp8_bs64_prompt1_decode_admission.json`:
  `256/256` success, output `1325.58 tok/s`, TTFT p50/p99 `72.11/472.45ms`,
  TPOT p50/p95/p99 `47.41/47.56/47.56ms`, ITL p50/p99 `47.70/51.55ms`.
- vLLM warmup-after baseline,
  `/tmp/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`:
  `256/256` success, output `594.57 tok/s`, TTFT p50/p99 `161.30/303.20ms`,
  TPOT p50/p95/p99 `107.20/109.00/109.20ms`, ITL p50/p99 `108.92/116.35ms`.

Decision:

- Keep. O1 moves prompt_len=1 onto the correct decode shape and clears the current H20
  vLLM bs64 TPOT/output gate. Follow-up profiles should focus on lowering pegainfer service
  TPOT from `47ms` toward the H200-reported 30ms-class expectation if that target is confirmed
  on comparable hardware.

## Open Questions

- The H20 vLLM TP1 DP8 EP8 bs64 result remains 100ms-class even after explicit bs64 warmup and
  CUDA graph capture (`PIECEWISE=51`, `FULL=35`) in
  `/tmp/kimi-vllm-dp8-warmup-20260525/server.log`. This conflicts with the remembered 30ms-class
  TPOT, which may have been measured on H200 or with a different all-to-all/backend shape.
- Both vLLM and pegainfer detailed JSON report `max_concurrent_requests=128` while the client
  command uses `--max-concurrency 64`; treat that field as a client-side reporting artifact until
  checked in vLLM bench internals. Throughput and percentile metrics are still computed from the
  completed request traces.
