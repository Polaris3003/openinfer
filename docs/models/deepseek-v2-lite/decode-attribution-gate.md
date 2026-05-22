# DeepSeek-V2-Lite EP2 Decode Attribution Gate

> **TL;DR:** DeepSeek-V2-Lite now has a narrow EP2 decode attribution report for the same correctness shape as the HF gate: `batch=1`, `prompt="Hello"`, `output_len=16`, host-staged backend, and NCCL backend. The report is CPU-side attribution plus route/transfer counts; it is evidence for the next bottleneck decision, not a throughput or production EP claim.
>
> **Status:** Passing for the covered EP2 `Hello` / 16-token host-staged and NCCL attribution gate.

## Scope

This gate deliberately stays model-specific and shape-specific:

- Model: DeepSeek-V2-Lite.
- Shape: batch `1`, prompt `Hello`, output length `16`.
- Backends: default host-staged EP2 and `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl`.
- Accuracy oracle: the same generated token/text/hash gate used by `hf-accuracy-gate.md`.
- Attribution source: `DeepSeekV2LiteEp2Generator::generate_greedy_with_attribution`.

Out of scope:

- sparse dispatch;
- pegainfer-comm / NVLink backend;
- multi-node or generic EP topology;
- batch > 1 or broader prompts;
- performance improvement or throughput claims.

## Report Shape

`dsv2_lite_ep2_decode_attribution` emits structured JSON:

- `report_type`, `model`, `phase`, `backend`, and fixed-shape `config`;
- nested `accuracy` with generated token ids, generated text, token sha256, and text sha256;
- CPU-side `timing` with total generation, the prefill-produced first output token, `per_output_token_us`, the 15 true decode-token samples for `output_len=16`, and latency stats;
- `by_section`, `by_op`, and `by_call_site` rollups in the same vocabulary family as the Qwen3 model report;
- `coverage` rows that explicitly mark GPU event timing and throughput claims as not covered;
- `ep` counters for host-staged dispatch/combine and NCCL dense exchange/combine plus local/remote route counts.

Host-staged `dispatch_calls` / `combine_calls` count MoE layer invocations in the fixed greedy run. Host-staged `dispatch_elements` / `combine_elements` count selected routed hidden vectors, so the value is route count times hidden size. NCCL `exchange` and `combine` counters count the dense all-reduce calls and elements used by the current naive NCCL gate.

## Commands

Run the accuracy gate first, because attribution is not allowed to weaken the HF / host-staged / NCCL oracle:

```bash
mkdir -p target/accuracy/dsv2-lite-ep2

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --prompt Hello \
  --output-len 16 \
  --out target/accuracy/dsv2-lite-ep2/hf.json

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl.json \
  --out target/accuracy/dsv2-lite-ep2/comparison.json \
  --require-all-exact
```

Then collect attribution for the same two pegainfer backends:

```bash
cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --out target/accuracy/dsv2-lite-ep2/host-staged-attribution.json

PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --out target/accuracy/dsv2-lite-ep2/nccl-attribution.json
```

## Environment Notes

The NCCL path depends on a runtime that supports the selected GPU. On newer GPUs, older NCCL runtimes may fail communicator initialization before the model-level comparison runs, for example with a shared-memory init error like:

```text
ncclMaxSharedMem 82240 exceeds device/fn maxSharedMem 79856
NCCL WARN Cuda failure 1 'invalid argument'
```

Use a newer NCCL runtime through the normal library path if the system runtime fails this way. The project code path should not change just to work around local NCCL installation age.

The HF oracle needs a Python environment that can load DeepSeek-V2-Lite with `trust_remote_code=True`, including the model's `flash_attn` dependency. Keep that environment separate from the Rust runtime claim: it is only the truth-source generator for the comparison JSON.

## Latest Validation

The full gate was last rerun on 2026-05-22 with `models/DeepSeek-V2-Lite`, `prompt="Hello"`, and `output_len=16`:

- HF / host-staged / NCCL comparison: `all_token_text_exact`.
- Token SHA256: `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`.
- Text SHA256: `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.
- Host-staged attribution: `dispatch_calls=416`, `combine_calls=416`, `dispatch_elements=5111808`, `combine_elements=5111808`, `local_route_count=1284`, `remote_route_count=1212`.
- NCCL attribution: `nccl_exchange_calls=416`, `nccl_combine_calls=416`, `nccl_exchange_elements=851968`, `nccl_combine_elements=851968`, `local_route_count=1284`, `remote_route_count=1212`.

## Claim Boundary

This report proves only that the covered DeepSeek-V2-Lite EP2 greedy path still produces the expected token/text hashes and that the current runtime observed the listed CPU-side sections, route counts, and dense collective counts. It does not prove GPU kernel event timing, serving throughput, sparse dispatch readiness, multi-node behavior, or production EP readiness.
