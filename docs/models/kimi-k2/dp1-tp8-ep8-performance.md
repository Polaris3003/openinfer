# Kimi-K2 DP1 TP8 EP8 Performance

> **TL;DR:** DP1 TP8 EP8 的性能主线从 correctness baseline
> `72c770b` 开始。目标是在 H20 ×8、bs64、decode-heavy 服务口径下超过
> vLLM `0.19.0` 的 bs64 baseline：output `583.9 tok/s`，TPOT median
> `109.00ms`。
>
> **Status:** Project doc opened. No performance optimization is accepted here
> until it has a correctness gate and its own commit.

## Target

| Item | Target / baseline |
| --- | --- |
| Machine | `h20-100`, 8× NVIDIA H20 |
| Model | `/data/models/Kimi-K2.5` |
| Shape | DP1 TP8 EP8 |
| Primary workload | `input_len=1`, `output_len=128`, `ignore-eos`, `bs=64` |
| vLLM baseline | TP1 DP8 EP8, `vllm bench serve`, output `583.9 tok/s`, TPOT median `109.00ms`, TPOT p99 `109.76ms` |
| PegaInfer goal | output tok/s `> 583.9` at bs64, while preserving token correctness |

The comparison target comes from [vllm-h20-baseline.md](vllm-h20-baseline.md).
The correctness ground truth starts from
[pplx-ep-correctness.md](pplx-ep-correctness.md): TP8 PPLX must be token-trace
exact against TP8 NCCL under the same bs64 active-decode schedule.

## Gate Rules

Every kept optimization needs all of these recorded before commit:

| Gate | Requirement |
| --- | --- |
| Profile | Start from an observed profile or benchmark delta. Record the command, output path, and the measured bottleneck or symptom. |
| Motivation / expected gain | State why the change should help and the expected size/direction of the win before implementing it. |
| Microbench | Add or run the smallest probe that isolates the changed subsystem when practical. If no microbench is practical, record why and use the closest lower-level measurement. |
| Correctness | Record the exact command, output file, token hash, and comparison target. For TP8/PPLX changes, compare against the TP8 NCCL baseline unless a stronger reference is documented. |
| Performance | Record bs64 service numbers and the lower-level in-process probe that explains the delta. |
| Scope | State whether the optimization targets frontend/scheduler, CUDA graph, collectives, MLA, MoE, or sampling. |
| Revert line | Record the measurable regression that would make the change revert-worthy. |
| Commit | Commit the code and this doc update together. |

No optimization is accepted on performance numbers alone.

Preferred entry shape:

```text
Profile:
  <command + report path + bottleneck>
Motivation / expected gain:
  <why this change should move bs64, and by roughly how much>
Microbench:
  <isolated probe, or the reason a subsystem-only probe is not available>
Correctness gate:
  <hash / trace / reference path>
Performance gate:
  <bs64 service number + supporting in-process/profile number>
Decision:
  <keep/reject/defer + commit>
```

This is a discipline, not a rigid template. The important part is that future
readers can reconstruct why an optimization was attempted, what it was expected
to buy, and which evidence made it worth keeping.

## Canonical Bs64 Pressure Test

Use this exact service pressure-test shape for bs64 comparisons. This is the
single project-wide pressure command for Kimi-K2 bs64 reports. Do not change
prompt/output length, request count, request rate, concurrency, percentiles,
streaming mode, or `ignore-eos` when reporting numbers against the vLLM bs64
baseline.

Server:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep -- \
  --model-path /data/models/Kimi-K2.5 \
  --port 8124 \
  --cuda-graph true
```

Client:

```bash
cd /root/develop/xingming/pegainfer
COMMIT=$(git rev-parse --short HEAD)
mkdir -p /tmp/kimi-bs64-baseline
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
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/kimi-bs64-baseline \
  --result-filename pegainfer_tp8_pplx_bs64_${COMMIT}.json \
  2>&1 | tee /tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_${COMMIT}.log
```

Required report fields:

| Field | Value |
| --- | --- |
| `--random-input-len` | `1` |
| `--random-output-len` | `128` |
| `--random-range-ratio` | `0` |
| `--num-prompts` | `256` |
| `--max-concurrency` | `64` |
| `--request-rate` | `inf` |
| `--ignore-eos` | enabled |
| `--percentile-metrics` | `ttft,tpot,itl` |
| `--metric-percentiles` | `50,95,99` |

## Correctness Probe

Run this before accepting a performance change, and compare it with the TP8 NCCL
reference from [pplx-ep-correctness.md](pplx-ep-correctness.md):

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_correctness64.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

## Optimization Ledger

| ID | Date | Commit | Area | Change | Correctness gate | bs64 result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 2026-05-25 | `72c770b` | correctness | TP8 PPLX baseline fixed; no performance claim | TP8 NCCL/PPLX 64-token hash `4920f088c2338236` | Not measured | Keep as ground truth |
| B1 | 2026-05-25 | `d639e55` code, `df1cd18` command doc | scheduler / service profile | Canonical bs64 pressure baseline before performance work | No code change after B0; PPLX correctness baseline remains `4920f088c2338236` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json`: output `137.51 tok/s`, TPOT p50/p95/p99 `26.40/28.13/28.46ms`, TTFT p50/p99 `54.76/58.68s`, 256/256 success | Keep as profile baseline; first optimization should address 4-row scheduling/admission before kernel work |
| O1 | 2026-05-25 | this commit | scheduler / decode arena | Raise DP1 TP8 admission to bs64; allocate decode arenas lazily in `1/2/4/8/16/32/64` buckets; preflight arena allocation on all TP ranks before prefill collectives | `/tmp/kimi_pplx_tp8_correctness64_o1_bucket.json`: TP8 PPLX 64-token hash `4920f088c2338236` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.json`: output `145.18 tok/s`, TPOT p50/p95/p99 `195.07/221.08/224.72ms`, TTFT p50/p99 `31.00/35.76s`, 256/256 success | Keep as bs64 enabling baseline; not enough for vLLM target, next profile must attack bs64 kernel/communication cost |
| C1 | 2026-05-25 | this commit | correctness / PPLX MoE | Align TP8 PPLX with TP8 NCCL for active bs64 decode: active MoE rows, TP8-only duplicate-source canonicalization, NCCL-layout local expert compute before PPLX combine | `/tmp/kimi_pplx_tp8_active64_o5_after_review.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 per-index token mismatches; both paths hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | Not a performance optimization; PPLX correctness probe TPOT p50 `110.14ms` vs NCCL `97.53ms`; rerun canonical bs64 pressure after this correctness commit | Keep as the new correctness baseline before further optimization |
| P1 | 2026-05-25 | documentation only | service / scheduler profile | Profile `00b3f1f` after C1 with the canonical bs64 command and an in-process bs64/output128 microbench | No code change after C1; C1 correctness baseline remains the gate | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.json`: output `353.91 tok/s`, TPOT p50/p95/p99 `146.15/172.83/175.10ms`, TTFT p50/p99 `4.58/10.24s`, 256/256 success; in-process warm1 steady TPOT p50 `107.76ms` | Keep as profile baseline; next optimization should target serial first-token prefill without changing token trace |
| O2 | 2026-05-25 | this commit | scheduler / MLA prefill | Replace prompt_len=1 first-token MLA attention with the exact single-token V path; keep microbatch at 1 because seq_len>1 drifted | `/tmp/kimi_pplx_tp8_c1fast_mb1_o5.json` vs `/tmp/kimi_nccl_tp8_active64_o5_final.json`: 0 mismatches; hash counter `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fastmb1_candidate.json`: output `414.28 tok/s`, TPOT p50/p95/p99 `133.36/147.74/149.42ms`, TTFT p50/p99 `2.76/6.90s`, 256/256 success | Keep as an incremental first-token optimization; still below vLLM, next work must make batch>1 prompt_len=1 prefill correct or reduce PPLX TPOT |

### B1 Profile Notes

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json
```

Observed:

- bs64 output throughput is `137.51 tok/s`, far below vLLM bs64 `583.9 tok/s`.
- TPOT p50 is `26.40ms`, much lower than vLLM bs64 TPOT p50 `109.00ms`.
- TTFT p50 is `54.76s`, showing requests are queued in long waves.
- Current TP8 scheduler cap is still `KIMI_RUNNER_MAX_BATCH = 4`, so bs64 service
  pressure effectively runs as repeated 4-row decode waves.

Motivation / expected gain:

Raising the DP1 TP8 admission/arena path beyond 4 rows should attack the main
service-throughput gap directly. If per-token TPOT stayed near the B1 value,
bs64 throughput would have roughly 4x headroom before kernel scaling becomes the
dominant limit. The actual gain must be measured because MLA/MoE kernels,
collectives, scratch size, and graph capture may scale nonlinearly with batch.

Microbench:

B1 is a service profile, not an optimization. The next optimization must add a
lower-level probe for the changed layer, for example an in-process bs sweep or a
decode arena/scheduler trace that confirms active rows > 4 before rerunning the
canonical bs64 pressure command.

### O1 Lazy Bucketed Bs64 Decode Arenas

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json
```

Observed:

- Canonical bs64 service throughput was only `137.51 tok/s`, with TTFT p50
  `54.76s`.
- TPOT p50 was `26.40ms`, which was good for each 4-row wave but did not
  translate into bs64 service throughput.
- Code profile: TP8 scheduler admitted at most `KIMI_RUNNER_MAX_BATCH = 4`,
  and worker startup allocated all decode arenas eagerly up to the worker cap.

Motivation / expected gain:

Raising the scheduler and worker cap to 64 removes the obvious admission limit.
Decode arenas are allocated lazily in power-of-two buckets so canonical bs64
uses one bs64 KV/scratch/graph arena without allocating every size from 1 to 64.
The rank preflight makes allocation failure happen before prefill/decode
collectives, avoiding a partial-rank failure mode. Expected direction: much
lower bs64 TTFT and enough active rows to expose the real bs64 kernel and PPLX
communication cost.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_o1_bucket_micro_bs64.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 0 --iters 1
```

Result:

- Output path: `/tmp/kimi_pplx_tp8_o1_bucket_micro_bs64.json`.
- Workload confirmed `concurrency=64`, `output_len=128`, all `64` traces had
  length `128`.
- In-process wall throughput, computed as `64 * 128 / max_e2e`, was about
  `226.9 tok/s` (`max_e2e=36.108s`).
- Steady TPOT p50/p95/p99 was `178.35/201.96/218.85ms`.

Correctness gate:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_correctness64_o1_bucket.json \
  request --output-len 64 --warmup 0 --iters 1
```

Result: generated-token hash `4920f088c2338236`, matching the TP8 NCCL/PPLX
baseline.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `145.18 tok/s` vs B1 `137.51 tok/s`.
- Peak output throughput: `504.00 tok/s` vs B1 `168.00 tok/s`.
- TTFT p50/p95/p99: `31.00/35.23/35.76s` vs B1 p50/p99
  `54.76/58.68s`.
- TPOT p50/p95/p99: `195.07/221.08/224.72ms`.

Decision:

Keep. O1 preserves token correctness and turns bs64 into real 64-row decode
waves, but the service output throughput is still far below the vLLM `583.9
tok/s` target. The next accepted optimization needs a profile of the bs64
decode step itself, especially PPLX MoE routing/combine, MLA decode, and TP
collectives.

### P1 Post-C1 Bs64 Profile

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.json
```

Observed:

- Canonical bs64 service result at `00b3f1f`: `256/256` success, output
  throughput `353.91 tok/s`, request throughput `2.76 req/s`.
- TTFT p50/p95/p99: `4.58/9.04/10.24s`.
- TPOT p50/p95/p99: `146.15/172.83/175.10ms`.
- ITL p50/p95/p99: `116.62/119.65/122.74ms`.
- Peak output-token bucket from vLLM bench result:
  `max_output_tokens_per_s=640.0`.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_00b3f1f_micro_bs64_o128_warm1.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Result:

- Output path:
  `/tmp/kimi_pplx_tp8_00b3f1f_micro_bs64_o128_warm1.json`.
- All `64` traces have length `128`.
- Hash counter: `32x 82a791616c737442`, `16x 4ae8834e96c7d195`,
  `16x 24b2b3856ac0ea3a`.
- Steady TPOT p50/p95/p99: `107.76/109.06/110.45ms`, equivalent to about
  `593.9 tok/s` for a 64-row decode step.
- End-to-end p50/max: `20.81/20.81s`, equivalent to about `393.6 tok/s`
  over `64 * 128` output tokens.

Motivation / expected gain:

The steady decode step is already in the vLLM target range, while request e2e is
not. `bench_serving` measures `first_decode_step_ms` as the interval from the
first emitted token to the second emitted token; it is not the first kernel
duration. Code inspection shows the Kimi scheduler coalesces 64 requests, then
runs `prefill_request` one slot at a time before entering batched decode. The
observed wall time is consistent with:

```text
64 serial prompt_len=1 first-token forwards + 127 batched decode steps
```

An accepted fix needs to make the first-token path batched while preserving the
C1 TP8 NCCL token trace. Replacing prompt prefill with decode is not sufficient;
see the rejected item below.

### O2 Prompt-Len-1 Single-Row Fast Prefill

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_00b3f1f.json
```

Observed:

- P1 showed canonical bs64 output throughput `353.91 tok/s`, with TTFT p50/p99
  `4.58/10.24s` and TPOT p50/p95/p99 `146.15/172.83/175.10ms`.
- The in-process bs64/output128 probe showed steady TPOT p50 `107.76ms`, so the
  remaining service gap was dominated by first-token work and serving cadence,
  not only steady decode.
- Code inspection confirmed `64` serial `prefill_request` calls before batched
  decode. For `prompt_len=1`, causal MLA attention has exactly one key, so the
  attention output should equal the V slice produced by `kv_b_proj`.

Motivation / expected gain:

Avoid the Q branch, temporary K/V cache assembly, and FlashInfer single-prefill
call for each one-token prompt. The change keeps the original prefill semantics:
embedding and residual all-reduces remain BF16 NCCL, KV is still appended at
position 0, and TP8 prompt MoE remains the NCCL path. Expected gain is lower
TTFT and modest service throughput improvement while preserving the C1 token
trace.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_c1fast_mb1_o5.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

Result:

- TP8 NCCL fast path:
  `/tmp/kimi_nccl_tp8_c1fast_mb1_o5.json`, TTFT p50/p99
  `4.71/6.32s`, e2e p50 `6.92s`, steady TPOT p50 `97.81ms`.
- TP8 PPLX fast path:
  `/tmp/kimi_pplx_tp8_c1fast_mb1_o5.json`, TTFT p50/p99
  `5.27/7.35s`, e2e p50 `8.07s`, steady TPOT p50 `110.45ms`.
- Both files match `/tmp/kimi_nccl_tp8_active64_o5_final.json` exactly:
  0 per-index mismatches and hash counter `32x 7c4c5d83355198fd`,
  `32x 9eecc1ca6fb3409d`.

Correctness gate:

```bash
uv run --no-project python - <<'PY'
import collections, json, subprocess
old=json.loads(subprocess.check_output(
    ['ssh','h20-100','cat','/tmp/kimi_nccl_tp8_active64_o5_final.json']))
new=json.loads(subprocess.check_output(
    ['ssh','h20-100','cat','/tmp/kimi_pplx_tp8_c1fast_mb1_o5.json']))
mis=[i for i,(a,b) in enumerate(zip(
    old['metrics']['generated_token_traces'],
    new['metrics']['generated_token_traces'])) if a['prefix'] != b['prefix']]
print(collections.Counter(t['hash'] for t in new['metrics']['generated_token_traces']))
print('mismatches', len(mis))
PY
```

Observed output: `Counter({'7c4c5d83355198fd': 32, '9eecc1ca6fb3409d': 32})`,
`mismatches 0`.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fastmb1_candidate.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_fastmb1_candidate.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `414.28 tok/s` vs P1 `353.91 tok/s`.
- Peak output throughput: `704.00 tok/s` vs P1 `640.00 tok/s`.
- TTFT p50/p95/p99: `2.76/6.26/6.90s`.
- TPOT p50/p95/p99: `133.36/147.74/149.42ms`.
- ITL p50/p95/p99: `117.15/120.31/126.00ms`.

Decision:

Keep. The change preserves the TP8 NCCL/PPLX correctness baseline and improves
canonical bs64 output throughput by about `17%`. It does not reach vLLM
`583.9 tok/s`; the next optimization should either make prompt_len=1 batch>1
prefill trace-exact, or reduce the PPLX steady TPOT gap.

## Candidate Queue

| Priority | Area | Hypothesis | Correctness risk |
| --- | --- | --- | --- |
| P0 | scheduler / prefill | Implement an exact batched `prompt_len=1` first-token path instead of 64 serial `prefill_request` calls. It must preserve the C1 TP8 NCCL token trace, unlike the rejected decode-substitution probe. | High: first-token KV state affects all following tokens. |
| P0 | PPLX / MoE | O1 shows real bs64 waves but TPOT p50 is `195.07ms`; routed expert dispatch/combine or Marlin route capacity may now dominate. | High: routed expert and combine weights are correctness-sensitive. |
| P0 | CUDA Graph | Reduce bs64 first-step graph capture/replay and metadata overhead after kernel profile identifies host or graph-node cost. | Medium: graph replay must preserve per-row metadata and PPLX participation. |
| P1 | frontend | Measure HTTP/streaming overhead separately from in-process TPOT. | Low for model math, medium for serving semantics. |
| P1 | collectives | Profile TP all-reduce and routed combine tail at bs64. | Medium: BF16/F32 collective boundary is correctness-sensitive. |
| P2 | MLA/MoE | Retune batch-shape kernels only after scheduler and graph bottlenecks are visible. | High: routed expert and MLA cache layout are easy to perturb. |

## Rejected / Deferred

| Date | Idea | Reason |
| --- | --- | --- |
| 2026-05-25 | Use TP1/DP8 correctness as the baseline for this doc | Deferred. TP1/DP8 matched short probes but diverged at 32 tokens, so DP1 TP8 work uses TP8 NCCL/PPLX baseline first. |
| 2026-05-25 | Use the batch decode kernel as the `prompt_len=1` first-token path | Rejected. New TP8 NCCL and PPLX matched each other (`/tmp/kimi_nccl_tp8_single_prefill_batch_o2_o5.json` vs `/tmp/kimi_pplx_tp8_single_prefill_batch_o2_o5.json`: 0 mismatches), but both changed `32/64` per-index traces compared with the C1 TP8 NCCL ground truth `/tmp/kimi_nccl_tp8_active64_o5_final.json`. Hash counter changed from `32x 7c4c5d83355198fd`, `32x 9eecc1ca6fb3409d` to `48x 9eecc1ca6fb3409d`, `16x f45b2f0248e7059d`; this is not correctness-preserving. |
| 2026-05-25 | Run the exact prompt_len=1 fast prefill path with microbatch `2` or larger | Rejected for now. The full-batch probe `/tmp/kimi_nccl_tp8_c1batch_o5.json` produced `42/64` mismatches and hash counter `40x 7c4c5d83355198fd`, `18x f45b2f0248e7059d`, `6x 9eecc1ca6fb3409d`. A block-size-8 A/B still failed (`/tmp/kimi_nccl_tp8_c1batch_block8_o5.json`). The sweep showed bs2 correct in isolation (`/tmp/kimi_nccl_tp8_c1batch_bs2_o5.json`) but bs4+ drifted, and the scheduler microbatch=2 candidate `/tmp/kimi_nccl_tp8_c1micro2_o5.json` still had `37/64` mismatches. The accepted O2 path therefore keeps `KIMI_PROMPT_LEN1_PREFILL_MICROBATCH=1` until seq_len>1 layer parity is proven. |
