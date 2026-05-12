# DeepSeek V4 Decode Performance

**Created**: 2026-05-12
**Status**: active

## TL;DR

This document consolidates the DeepSeek V4 decode work that moved fixed long decode from the `~108-113ms/token` band to the current small-route mapping branch at about `27.61-27.83ms/token`, with prior clean sub-30 validation at `29.86-29.97ms/token`, earlier shared-expert fused repeats at `28.16-32.22ms/token`, routed W13+SwiGLU-quant validation at `31.18-33.42ms/token`, W13-only validation at `31.99-34.22ms/token`, and shared-quant-only validation at `33.33-34.29ms/token`. The retained changes are grouped MoE pointer caching, rank-worker placement, removal of hot temporary zero-fill, rank-owned decode scratch, caller-owned grouped FP4 workspace, shared W1/W3 activation quantization, W13 grouped FP4 runtime launch, routed fused SwiGLU+W2 activation quantization, shared expert fused W1/W3 quant, shared fused SwiGLU+W2 quant, shared dense FP8 W13, fused MoE mapping clear, small-route MoE mapping, and benchmark/counter instrumentation. Stable sub-`30ms/token` is achieved on the fixed bench; the active MoE goal is now stable sub-`25ms/token` by mirroring mature vLLM/SGLang decode MoE decomposition and only then exploring deeper fusion without bs=1 specialization. Post-small-route experiments rejected so far: direct route-W13 regressed to `33.14-33.36ms/token`; fused expand+act_quant was bitwise and `2.5-3.2x` faster in microbench but repeated runtime TPOT landed at `27.20-28.71ms/token`; GPU-side valid-row skip in W2 SwiGLU+quant regressed to `31.22-31.34ms/token`; host-known per-expert row-tile upper bound kept exactness but landed at `28.46-28.74ms/token`. Exact E2E remains `20/20`, and the fixed bench token hash remains `6346f03343d75a65`.

The retained team lessons are more important than the discarded attempt logs: compare identical token traces, separate NCCL wait from transfer, treat capacity and logical length separately, keep MoE semantic zero on device, and prove allocation cleanup with application-visible CUDA API counters rather than nsys attribution alone.

## Baseline And Result

Use this fixed decode bench for comparable DeepSeek V4 direct-runtime work:

```bash
target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  request \
  --prompt-len 1 \
  --output-len 160 \
  --warmup 2 \
  --iters 3 \
  --seed 42
```

| Milestone | Fixed long decode | Key change |
| --- | ---: | --- |
| AG/RS grouped MoE baseline | `107.92-112.61ms/token` | GPU AG/RS path existed, but repeated pointer-array setup and per-token allocations remained. |
| Grouped MoE pointer cache | `83.37-89.65ms/token` | Cache grouped FP4 expert weight/scale pointer arrays per rank worker. |
| Rank-worker affinity | `72.88-73.60ms/token` | Reduce rank-arrival skew before f32 collectives. |
| Remove hot temporary zero-fill | `63.35-64.51ms/token` | Fully overwritten hot temporaries allocate uninitialized storage. |
| Rank-owned decode scratch | `34.34-35.36ms/token` forced NUMA, `32.75-33.90ms/token` same-code fast band | Move hot intermediate storage to per-rank scratch and remove grouped FP4 C-side growth-cache workspace. |
| Final PR validation | `35.253ms/token` | After review fixes: dynamic NUMA topology, buffer-derived capacity checks, and `_ptsz` counter separation. |
| Shared W1/W3 act quant | `33.330ms`, repeated `34.289ms` | Decode scratch W1 and W3 reuse one TileLang `act_quant_k4096`; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| W13 grouped runtime launch | text `34.22ms`, JSON `31.986ms` | W1 and W3 share one TileLang grouped FP4 launch after shared activation quant; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| Routed fused SwiGLU + W2 act quant | `33.416ms`, repeated `31.180ms` | Mirror vLLM/SGLang's activation+quant fusion after materialized W13 output; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| Shared expert fused quant + dense W13 | `29.764ms`, repeated `31.592ms` | Shared expert scratch path reuses one FP8 act quant for W1/W3, fuses shared SwiGLU+W2 act quant, and uses one dense FP8 W13 launch; token hash stays `6346f03343d75a65`, exact E2E `20/20`. |
| Clean sub-30 repeat after trace cleanup | `29.944ms`, `29.907ms`, `29.896ms` | Same fixed bench after removing per-layer trace syncs; all three measured iterations keep token hash `6346f03343d75a65`. |
| Fused MoE mapping clear | `29.862ms`, `29.969ms`, `29.874ms` | Merge six local-route mapping clear launches into one kernel; exact E2E `20/20`, token hash stays `6346f03343d75a65`. |
| Small-route MoE mapping | first run `27.608ms`, `27.662ms`, `27.826ms`; repeat `27.698ms`, `27.693ms`, `27.644ms` | Decode route mapping uses one small-batch kernel for `route_elems <= 1024`; exact E2E `20/20`, token hash stays `6346f03343d75a65`. |

Final PR validation on 5090:

| Metric | Value |
| --- | ---: |
| steady TPOT avg | `35.253ms` |
| steady TPOT p50 | `34.800ms` |
| steady TPOT p95 | `37.335ms` |
| first decode avg | `33.743ms` |
| generated-token hash | `6346f03343d75a65` |
| exact E2E | `20/20` |

## Retained Design

### Grouped MoE pointer cache

Each persistent rank worker builds a `MoeGroupedPtrCache` once after context binding. The cache stores per-layer GPU arrays for local expert weight pointers and scale pointers for W1/W2/W3 grouped FP4 linears. Decode and prefill MoE paths pass this cache to grouped FP4 local expert execution.

This removed repeated host vector construction and H2D pointer-array copies from every grouped FP4 call. The grouped FP4 kernels did not become materially faster in nsys; the improvement showed up as a shorter MoE reduce-scatter synchronization window, which points to lower rank-arrival skew.

### Rank-worker placement

Rank workers are pinned before CUDA work begins. The final PR path resolves topology dynamically:

1. CUDA driver `cuDeviceGetPCIBusId` maps CUDA ordinal to PCI bus id.
2. `/sys/bus/pci/devices/<pci>/numa_node` maps PCI to NUMA node.
3. `/sys/devices/system/node/node<numa>/cpulist` supplies target CPUs.
4. The target list is intersected with the process's allowed cpuset.
5. Missing topology, empty intersection, or `pthread_setaffinity_np` failure panics.

Do not encode ordinal assumptions such as `GPU0..3 -> NUMA0`. A review caught that earlier draft; it matched 5090 but was still a machine-specific fact in runtime logic. Also avoid CUDA runtime topology calls here: `cudaDeviceGetPCIBusId` loaded an incompatible `libcudart` on 5090, while the CUDA driver API path worked.

5090 final pin evidence:

| GPU ordinal | PCI bus | NUMA | pinned CPU |
| --- | --- | ---: | ---: |
| `0` | `0000:16:00.0` | `0` | `0` |
| `1` | `0000:27:00.0` | `0` | `1` |
| `2` | `0000:38:00.0` | `0` | `2` |
| `3` | `0000:5a:00.0` | `0` | `3` |
| `4` | `0000:98:00.0` | `1` | `36` |
| `5` | `0000:a8:00.0` | `1` | `37` |
| `6` | `0000:c8:00.0` | `1` | `38` |
| `7` | `0000:d8:00.0` | `1` | `39` |

### Rank-owned decode scratch

`RankDecodeScratch` is created once per rank worker and reused by decode commands. The current direct scheduler still sends one token per rank command, but the scratch design is capacity-based and should not assume batch size one in API contracts.

| Area | Scratch owner | Note |
| --- | --- | --- |
| Token upload | `RankDecodeScratch::token_ids` | Replaces per-token `clone_htod(&[token_id])` with H2D copy into rank-owned storage. |
| Entry hidden | `DecodeEntryScratch` | Embedding and HC expand outputs are fully overwritten. |
| HC pre/post | `HcPreNormScratch`, `HcPostScratch` | HC pre-state and layer outputs reuse rank-owned buffers; HC post layer output uses ping-pong slots to avoid adjacent-layer aliasing. |
| Attention | `AttentionProjectionScratch`, `AttentionIndexScratch`, `AttentionAuxScratch`, `AttentionOutputScratch` | Active ratio `0` and ratio `4` decode paths use capacity buffers with logical lengths passed separately. |
| Shared expert | `SharedExpertScratch` | Fixed-shape gate/up/out storage plus caller-owned FP8 activation/scale workspace for shared W1/W3 and W2. |
| MoE AG/RS | `MoeAgRsScratch` | Hidden/token all-gather, route buffers, compact maps, expert intermediates, partial routed output, local reduce-scatter output, routed+shared output. |
| Grouped FP4 workspace | `MoeAgRsScratch::{fp4_act_workspace,fp4_act_scale_workspace}` | Caller-owned workspace avoids the C-side grouped FP4 growth-cache/mutex path. |
| Final logits | `FinalLogitsScratch` | HC head, final norm, local logits, and gathered logits are reusable. |

### Capacity and logical length

Reusable scratch must not use mutable `seq_len` as allocation capacity. The final code exposes buffer-derived `seq_capacity()` helpers for `Bf16HiddenStates`, `F32HiddenStates`, and `HcHiddenStates`. Scratch-backed `*_into` operators check capacity from storage length, then set `seq_len` to the logical length for this decode step.

NCCL calls must use logical prefix slices, not whole-capacity buffers:

- BF16 hidden all-gather sends `hidden_dim * local.seq_len` and receives `hidden_dim * gathered_seq_len`.
- F32 reduce-scatter sends `hidden_dim * global.seq_len` and receives `hidden_dim * local_seq_len`.
- U32 token all-gather and ratio-4 indexer score all-reduce slice to logical prefixes.

### MoE dynamic content

MoE route values remain dynamic. Static storage does not mean static route content:

- route weights and indices change per token/layer.
- compact maps and `expert_indptr` depend on the route.
- local expert counters/cursors need semantic initialization.

Storage is capacity-based. Semantic clears remain inside `deepseek_moe_local_mapping_cuda` for counters/cursors/indptr and mapping sentinels.

## Active MoE Sub-30 Work

### Goal

Drive decode MoE from roughly `17-19ms/token` toward `10-12ms/token`, enough to move overall fixed long decode from roughly `35ms/token` to stable `28-30ms/token`. Optimizations must remain batch-general, keep route/tile scheduling on GPU, and preserve exact E2E `20/20`.

### Attempt: fuse SwiGLU with W2 activation quant

The local expert decode path originally did:

```text
W1 grouped FP4 GEMM -> gate BF16
W3 grouped FP4 GEMM -> up BF16
SwiGLU clamp -> activated BF16
TileLang act_quant(activated) -> FP8 activation + E8M0 scales
W2 grouped FP4 GEMM
```

The attempted branch replaced the decode scratch W2 input path with:

```text
W1 grouped FP4 GEMM -> gate BF16
W3 grouped FP4 GEMM -> up BF16
fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
W2 grouped FP4 GEMM
```

The fused quant keeps the old semantic order by rounding the SwiGLU result to BF16 before FP8 quantization. That matters for exact output: skipping the BF16 intermediate rounding would not be the same operator.

The first implementation used one CTA per `(row, 128-column group)`. That was the wrong GPU shape for this workload: with about `64` routed rows and `16` scale groups, it launched about `1024` CTAs per W2 quant, while the TileLang act_quant shape is `ceil(rows / 32) * 16`, about `32` CTAs. This explains why launch-count fusion did not translate into stable TPOT.

Evidence from the row-per-CTA version:

| Run | Result |
| --- | --- |
| Exact E2E on 5090 | `All 20 DeepSeek V4 exact cases passed` |
| Fixed bench run 1 | steady TPOT avg `31.922ms`, p50 `31.417ms`, p95 `33.812ms`, hash `6346f03343d75a65` |
| Fixed bench run 2 | steady TPOT avg `34.939ms`, p50 `34.388ms`, p95 `37.047ms`, hash `6346f03343d75a65` |
| Fixed bench run 3 | steady TPOT avg `34.572ms`, p50 `34.022ms`, p95 `36.767ms`, hash `6346f03343d75a65` |

Keep/drop decision: do not treat the `31.922ms` run as evidence. The repeated runs show the row-per-CTA version is not a stable win. The useful retained lesson is that a fused kernel must preserve the original row-block quantization shape, not merely reduce kernel launches.

Follow-up repair used a 4-warp row-block kernel and restored exact E2E `20/20`, but the fixed long bench token hash changed from `6346f03343d75a65` to `00fd2d772e4b8886` and TPOT regressed to `35.554ms`. Drop decision: do not retain this path. It changes long decode behavior even though the short exact suite passed.

### Retained: share W1/W3 activation quant

The retained local-expert change is deliberately narrower:

```text
Before:
W1: act_quant(expanded_input) + grouped FP4 GEMM
W3: act_quant(expanded_input) + grouped FP4 GEMM

After:
act_quant(expanded_input) once
W1 grouped FP4 GEMM reuses act/scale
W3 grouped FP4 GEMM reuses act/scale
```

This preserves the original TileLang activation quantization bit path and only removes duplicate work on the identical W1/W3 input. W2, SwiGLU, combine, route planning, and collectives are unchanged.

Validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | steady TPOT avg `33.330ms`, p50 `32.858ms`, p95 `35.274ms`, hash `6346f03343d75a65` |
| fixed bench run 2 | steady TPOT avg `34.289ms`, p50 `33.979ms`, p95 `36.852ms`, hash `6346f03343d75a65` |

Keep decision: retain. The gain is modest and still noisy, but the method is sound: fixed token trace, exact-safe, and it removes one W1/W3 activation quant launch per layer without introducing new quant math. The next MoE step should be W13 grouped GEMM or GPU active tile list, not more W2 quant fusion.

### Retained: W13 grouped FP4 GEMM runtime launch

W13 was first evaluated as a pure operator change before touching runtime. The TileLang generator now emits:

```text
deepseek_tilelang_fp4_grouped_w13_gemm_n2048_k4096
```

It is generated from the existing `N=2048,K=4096` FP4 grouped GEMM, but launches `grid.x=32` blocks:

```text
blockIdx.x 0..15   -> W1 pointer arrays -> gate output
blockIdx.x 16..31  -> W3 pointer arrays -> up output
```

The C++ tool `pegainfer-kernels/tools/deepseek_v4/w13_grouped_fp4_bench.cu` links the generated TileLang object directly and compares:

```text
baseline: grouped_gemm(W1) + grouped_gemm(W3)
candidate: grouped_w13_gemm(W1, W3)
```

Fuzz uses BF16 random input, TileLang `act_quant_k4096`, random FP4 bytes and bounded E8M0-like scale bytes, expert-major `expert_indptr` with empty experts, and bitwise BF16 output comparison for `gate` and `up`.

Verified compile command shape:

```bash
OUT_DIR=$(find target/release/build/pegainfer-kernels-* -maxdepth 1 -type d -name out -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_120 \
  -I/usr/local/cuda/include \
  pegainfer-kernels/tools/deepseek_v4/w13_grouped_fp4_bench.cu \
  "$OUT_DIR/libkernels_cuda.a" \
  -lcudart \
  -o /tmp/w13_grouped_fp4_bench
```

5090 microbench results:

| Rows | Experts | Fuzz | Baseline two GEMMs | W13 one GEMM | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| `64` | `4` | PASS | `0.126931ms` | `0.063485ms` | `1.999x` |
| `64` | `8` | PASS | `0.126988ms` | `0.122769ms` | `1.034x` |
| `64` | `16` | PASS | `0.300209ms` | `0.236817ms` | `1.268x` |
| `96` | `4` | PASS | `0.126970ms` | `0.063513ms` | `1.999x` |
| `96` | `8` | PASS | `0.126975ms` | `0.122862ms` | `1.033x` |
| `96` | `16` | PASS | `0.300129ms` | `0.238110ms` | `1.260x` |
| `160` | `4` | PASS | `0.127066ms` | `0.122937ms` | `1.034x` |
| `160` | `8` | PASS | `0.128706ms` | `0.124902ms` | `1.030x` |
| `160` | `16` | PASS | `0.303603ms` | `0.239735ms` | `1.266x` |
| `256` | `4` | PASS | `0.127242ms` | `0.124970ms` | `1.018x` |
| `256` | `8` | PASS | `0.246201ms` | `0.184480ms` | `1.335x` |
| `256` | `16` | PASS | `0.314086ms` | `0.251675ms` | `1.248x` |

Runtime integration replaces the two W1/W3 grouped GEMM launches after shared activation quant with the W13 launcher:

```text
act_quant(expanded_input) once
W13 grouped FP4 GEMM writes gate and up outputs
W2 path unchanged
```

The first runtime attempt failed exact E2E with `CUDA_ERROR_NOT_SUPPORTED` at decode layer 0. The cause was launch-wrapper setup inside CUDA graph capture: the new W13 kernel had not been used before capture, so its first `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, 98304)` could return `cudaErrorNotSupported`. The W13 wrapper now treats `cudaErrorNotSupported` like the existing `cudaErrorInvalidValue` tolerance and lets the actual launch result decide correctness.

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench text run | steady TPOT avg `34.22ms`, p50 `33.77ms`, p95 `36.53ms`, first decode avg `32.94ms` |
| fixed bench JSON run | steady TPOT avg `31.986ms`, p50 `31.458ms`, p95 `34.052ms`, first decode avg `30.544ms`, hash `6346f03343d75a65` |

Interpretation: W13 is exact at the operator level and in runtime, but the speedup depends heavily on routed-row distribution, local expert count, and run-to-run system noise. It is not automatically a `2x` W1/W3 win; some shapes mainly save launch overhead, while others are dominated by the expanded grid and grouped scheduling. Keep the runtime change because it removes one launch per layer and preserves the fixed token trace, but do not count the `31.986ms` run as stable sub-32 evidence until repeated long benches confirm it.

### Roadmap: mirror vLLM and SGLang decode MoE

The next long-running goal is to systematically absorb the mature vLLM/SGLang decode MoE decomposition and push beyond it only after the reproduced path is stable. The performance target is stable sub-`25ms/token` eventually, with stable sub-`30ms/token` as the first gate. This work remains batch-general and expert-general; do not introduce bs=1 or seq_len=1 special cases.

Reference source positions:

| Runtime | Source | Observed decode MoE shape |
| --- | --- | --- |
| vLLM | `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` | `cutlass_fp4_moe_mm(c1, W13)` writes `gate||up`, then `silu_and_mul_scaled_fp4_experts_quant(c1, ...)`, then `cutlass_fp4_moe_mm(W2)`. MXFP4 uses the same split with `silu_and_mul_mxfp4_experts_quant`. |
| vLLM C++ op registry | `/data/code/workspace-rustllm/vllm/csrc/libtorch_stable/torch_bindings.cpp` | Registers `silu_and_mul_scaled_fp4_experts_quant`, `silu_and_mul_mxfp4_experts_quant`, and grouped FP4/MXFP4 MoE GEMMs as separate ops. |
| SGLang | `/data/code/workspace-rustllm/sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` | `grouped_gemm_nt_f8f8bf16_masked` writes `gateup_output`, then `sglang_per_token_group_quant_8bit(..., fuse_silu_and_mul=True)`, then W2 grouped GEMM. |
| SGLang C++ quant | `/data/code/workspace-rustllm/sglang/sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu` | The `fuse_silu_and_mul` path fuses activation with group quant, including masked expert layout. |

The next reusable lesson is their problem-size representation. vLLM builds `expert_offsets`, `blockscale_offsets`, `problem_sizes1`, and `problem_sizes2` before CUTLASS grouped GEMM. SGLang's masked path passes `masked_m` and `expected_m` into DeepGEMM. Both make the GEMM scheduler aware of per-expert logical M. PegaInfer currently has `expert_indptr`, but the TileLang grouped launch still uses `dim3 grid(out_tiles, ceil(rows / 32), local_experts)` and returns inside the kernel when `blockIdx.y * 32 >= expert_m`. That is correct and GPU-resident, but it still launches empty CTAs for short or empty experts.

The first active-tile design check found a launch-side constraint: a GPU-generated active tile list cannot by itself shrink the next CUDA launch because grid dimensions are chosen on the host. Using a device-side `active_tile_count` would require a D2H count, CUDA dynamic parallelism, or launching the original capacity grid and returning on `tile >= active_count`. The last option preserves correctness but not the desired launch reduction. A better target is the existing `local_count`: decode route mapping already computes the actual number of local routes on GPU, while runtime still carries `num_expanded = routed.seq_len * topk` (`8 * 6 = 48` for MP8 decode) through expand, activation quant, and grouped GEMM. The hard part is exploiting `local_count` without reintroducing route metadata D2H.

Current PegaInfer path after W13:

```text
act_quant(expanded_input)
W13 grouped FP4 GEMM -> gate BF16 + up BF16
deepseek_swiglu_clamp_cuda -> activated BF16
TileLang act_quant_k2048(activated) inside W2 wrapper
W2 grouped FP4 GEMM
```

First reproduction target:

```text
act_quant(expanded_input)
W13 grouped FP4 GEMM -> gate BF16 + up BF16
fused SwiGLU clamp + BF16 semantic rounding + TileLang-compatible act_quant_k2048
W2 grouped FP4 GEMM using the produced FP8 activation and E8M0 scales
```

This mirrors vLLM/SGLang's proven operator split while preserving our exact semantic order. It still writes `gate/up` BF16 because vLLM and SGLang also materialize `gate||up` before activation+quant. The later, higher-risk path is to push activation+quant into the W13 accumulator epilogue and avoid writing `gate/up`; that requires separate microbench and fuzz evidence before runtime integration.

The standalone tool `pegainfer-kernels/tools/deepseek_v4/swiglu_quant_bench.cu` compares:

```text
baseline: deepseek_swiglu_clamp_cuda + TileLang act_quant_k2048
candidate: 4-warp fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
```

The first fused kernel used one CTA to serially process 32 rows. It was exact but too slow: rows `64` measured baseline `0.007064ms` vs fused `0.016471ms`, speedup `0.429x`. That shape was dropped. The retained microbench shape uses one warp per row and one CTA for four rows per 128-column group.

5090 microbench results:

| Rows | Fuzz | Baseline SwiGLU+act_quant | Fused SwiGLU+quant | Speedup |
| ---: | --- | ---: | ---: | ---: |
| `64` | PASS | `0.005612ms` | `0.002789ms` | `2.013x` |
| `96` | PASS | `0.005574ms` | `0.003139ms` | `1.776x` |
| `160` | PASS | `0.006164ms` | `0.004054ms` | `1.520x` |
| `256` | PASS | `0.007793ms` | `0.004073ms` | `1.914x` |

Runtime integration changes only the scratch hot path:

```text
W13 grouped FP4 GEMM -> gate BF16 + up BF16
deepseek_moe_fp4_grouped_w2_swiglu_with_workspace_cuda
  -> fused SwiGLU+act_quant_k2048-compatible FP8 activation
  -> W2 grouped FP4 GEMM
```

The old Rust scratch helper that performed generic grouped W2 activation quantization was removed so the decode path does not accidentally drift back to the split version. The lower C FFI remains for non-scratch compatibility and older callers.

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| `cargo fmt --check` | passed |
| `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON run 1 | steady TPOT avg `33.416ms`, p50 `32.884ms`, p95 `35.510ms`, first decode avg `31.885ms`, hash `6346f03343d75a65` |
| fixed bench JSON run 2 | steady TPOT avg `31.180ms`, p50 `30.675ms`, p95 `33.151ms`, first decode avg `30.020ms`, hash `6346f03343d75a65` |

Keep decision: retain as the vLLM/SGLang reproduction step. It is exact, removes one kernel launch and one BF16 intermediate write for W2 activation input, and the second fixed run matches the faster W13-only band. It is not sufficient for stable sub-`30ms/token`; the next step needs to explain remaining variance and reduce a larger section than activation+quant alone.

### Retained: shared expert fused decode path

The routed expert path was no longer the only split-MoE region. The shared expert decode scratch path still did:

```text
FP8 W1(input) with act_quant_k4096 -> gate BF16
FP8 W3(input) with act_quant_k4096 -> up BF16
SwiGLU clamp -> activated BF16
FP8 W2(activated) with act_quant_k2048 -> shared output
```

The retained shared path now does:

```text
act_quant_k4096(input) once
dense FP8 W13 -> gate BF16 + up BF16
fused SwiGLU clamp + BF16 rounding + FP8/E8M0 quant
FP8 W2 -> shared output
```

Implementation notes:

- `SharedExpertScratch` owns FP8 activation and scale workspaces so the shared scratch path does not use the C-side growth-cache/mutex path.
- `deepseek_fp8_w1_w3_with_workspace_cuda` reuses one activation quant for shared W1/W3 and calls the dense W13 TileLang kernel for the `4096 -> 2048` shared-expert shape.
- `deepseek_fp8_w2_swiglu_with_workspace_cuda` reuses the same fused SwiGLU+quant semantic order as the routed W2 path before calling the dense `4096 x 2048` FP8 W2 GEMM.
- `deepseek_tilelang_fp8_w13_gemm_n2048_k4096` is generated by transforming the existing dense FP8 TileLang GEMM into a two-output W1/W3 launcher. It is shape-specific to the shared expert, not bs=1 or seq_len=1 specific.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 `cargo fmt --check` | passed |
| 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON run 1 | steady TPOT avg `29.764ms`, p50 `29.296ms`, p95 `31.766ms`, first decode avg `28.575ms`, hash `6346f03343d75a65` |
| fixed bench JSON run 2 | steady TPOT avg `31.592ms`, p50 `31.082ms`, p95 `33.699ms`, first decode avg `30.019ms`, hash `6346f03343d75a65` |
| additional fixed repeats | `32.220ms`, `30.061ms`, `28.159ms`, all hash `6346f03343d75a65` |
| clean fixed bench after trace removal | `29.944ms`, `29.907ms`, `29.896ms`, all hash `6346f03343d75a65` |
| current HEAD exact E2E after clean bench | `All 20 DeepSeek V4 exact cases passed` |

Short nsys composition evidence, collected with `--output-len 32 --warmup 1 --iters 1 --seed 42` and used only for kernel composition, not TPOT:

| Kernel family | Evidence |
| --- | --- |
| Shared W13 | `deepseek_tilelang_fp8_w13_gemm_n2048_k4096_kernel` appears with `10,151` instances in the short full-process profile. |
| Old shared W1/W3 split GEMM | `deepseek_tilelang_fp8_gemm_n2048_k4096_kernel` drops to `1,118` residual instances, consistent with prefill/non-scratch residue rather than the decode scratch hot path. |
| Shared W2 activation quant | `deepseek_tilelang_act_quant_k2048_kernel` drops to `1,118` residual instances after fused shared/routed W2 quant. |
| Old SwiGLU clamp | `deepseek_swiglu_clamp_kernel` drops to `1,118` residual instances after decode scratch fusion. |

Keep decision: retain. This is the first run to cross sub-`30ms/token`, and the kernel composition proves the intended launches moved. It still does not satisfy the goal because repeated fixed runs returned `29.764ms`, `31.592ms`, `32.220ms`, `30.061ms`, and `28.159ms`; the current blocker is run-to-run variance and remaining synchronization windows, not exactness or missing vLLM/SGLang decomposition.

### Retained: fused MoE mapping clear

The local route mapping wrapper originally launched six tiny clear kernels per layer/rank/token:

```text
pos_to_token = -1
pos_to_token_topk = -1
token_topk_to_pos = -1
expert_indptr = 0
expert_cursor = 0
local_count = 0
```

These clears are semantic initialization, not removable allocation noise, but they do not need six launches. `deepseek_moe_clear_mapping_kernel` now clears all six buffers in one pass before the count, prefix, and mapping kernels. This keeps route metadata GPU-resident and preserves the same `expert_indptr` / `token_topk_to_pos` semantics.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | `29.862ms`, `29.969ms`, `29.874ms`, all hash `6346f03343d75a65` |
| short nsys kernel summary | old `deepseek_moe_clear_i32_kernel` gone; `deepseek_moe_clear_mapping_kernel` appears once per mapping call |

Keep decision: retain. The fixed bench movement is small, but the structural change removes five launches from every MoE mapping without changing math or adding synchronization.

### Retained: small-route MoE mapping

After fusing clears, decode still paid four mapping launches per layer/rank/token:

```text
clear mapping buffers
count local expert rows
prefix local expert row counts
fill compact maps
```

For MP8 decode, `route_elems = global_batch * topk`; with the fixed single-request bench this is `8 * 6 = 48`. The retained fast path uses one block when `route_elems <= 1024 && local_experts <= 256`, doing clear, count, prefix, and map fill with block-level barriers. Larger routed batches and prefill still use the existing multi-kernel path, so this is a small-problem route-mapping path rather than a bs=1-only branch.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON run 1 | `27.608ms`, `27.662ms`, `27.826ms`, all hash `6346f03343d75a65` |
| fixed bench JSON run 2 | `27.698ms`, `27.693ms`, `27.644ms`, all hash `6346f03343d75a65` |
| short nsys kernel summary | `deepseek_moe_local_mapping_small_kernel` replaces split clear/count/prefix/mapping kernels for the small decode route shape |

Keep decision: retain. This is a real structural win: the same route semantics move from four launches to one launch in decode-sized routed batches, and repeated fixed benches stay in the `27.6-27.8ms/token` band.

### Rejected: route W13 directly from token activations

After small-route mapping, the next tempting idea was to skip `expand_moe_fused_input_into` for W13:

```text
Before:
expand global_hidden -> expanded_input BF16 rows
act_quant(expanded_input)
expert-major W13 grouped FP4 GEMM

Attempt:
act_quant(global_hidden)
route-row W13 grouped FP4 GEMM -> compact gate/up rows
```

This was exact-safe but not performance-safe. The route W13 kernel launched one TileLang W13 tile set per route element, used `route_indices` and `token_topk_to_pos` to pick the local expert and compact output row, and kept W2/reduce unchanged. That removes one BF16 expand launch and quantizes only token rows instead of route rows, but it also destroys the existing expert-major grouped-GEMM shape. For decode-sized routes, the kernel becomes many small route-row GEMM tiles instead of contiguous expert ranges, so tensor-core work is scheduled less favorably than the retained W13 path.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | aggregate steady TPOT avg `33.217ms`, p50 `33.003ms`, p95 `34.584ms`, decode throughput `30.115 tok/s` |
| fixed bench iterations | `33.162ms`, `33.355ms`, `33.135ms`, all hash `6346f03343d75a65` |

Drop decision: do not retain. Correctness and token trace are not enough here; the implementation regresses from the current small-route mapping band of `27.61-27.83ms/token` to `33.14-33.36ms/token`. The reusable lesson is that reducing route-row materialization can lose if it breaks expert-major GEMM locality. Future work should preserve grouped expert ranges or build a real grouped scheduler that consumes per-expert problem sizes, rather than launching one-row route tiles.

### Rejected: fuse expand with W13 activation quant

The next experiment kept the expert-major W13 grouped GEMM shape and only fused the W13 input preparation:

```text
Before:
expand global_hidden -> expanded_input BF16 rows
TileLang act_quant_k4096(expanded_input) -> FP8 activation + E8M0 scales
expert-major W13 grouped FP4 GEMM

Attempt:
expand+act_quant_k4096(global_hidden, pos_to_token) -> FP8 activation + E8M0 scales
expert-major W13 grouped FP4 GEMM
```

This is the safer version of the previous idea because it preserves expert-major compact rows for W13. A temporary C++ microbench compared the baseline `deepseek_moe_expand_to_fused_cuda + deepseek_tilelang_act_quant_k4096` against a fused CUDA kernel using the same BF16, E8M0, and FP8 E4M3 conversion order as the existing exact-safe fused SwiGLU+quant kernel.

5090 microbench results:

| Tokens | Rows | Fuzz | Baseline expand+act_quant | Fused expand+act_quant | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| `8` | `48` | PASS | `0.008197ms` | `0.002575ms` | `3.183x` |
| `16` | `96` | PASS | `0.008199ms` | `0.002642ms` | `3.103x` |
| `32` | `192` | PASS | `0.010240ms` | `0.004074ms` | `2.514x` |

Runtime validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench run 1 | aggregate steady TPOT avg `27.807ms`; iterations `28.080ms`, `28.143ms`, `27.198ms`; all hash `6346f03343d75a65` |
| fixed bench run 2 | aggregate steady TPOT avg `28.565ms`; iterations `28.509ms`, `28.712ms`, `28.474ms`; all hash `6346f03343d75a65` |

Drop decision: do not retain in runtime. The operator microbench is real, but this is too small relative to the full decode step, and repeated fixed benches do not beat the retained small-route mapping band of `27.61-27.83ms/token`. The reusable lesson is methodological: only integrate this kind of local fusion when a profile proves the fused section is still visible at full-runtime scale, or when it is part of a larger fusion that removes a full synchronization/launch cluster.

### Rejected: skip W2 SwiGLU+quant rows after local_count

The route mapping kernel already computes `expert_indptr[local_experts] = local_count` on GPU. The attempted W2 change passed `expert_indptr + local_experts` into `deepseek_swiglu_clamp_act_quant_k2048_kernel` and skipped rows `>= local_count`:

```text
Before:
fused SwiGLU+quant over rows = route_capacity
W2 grouped FP4 GEMM skips empty rows via expert_indptr

Attempt:
fused SwiGLU+quant reads GPU local_count and only computes compact prefix rows
W2 grouped FP4 GEMM unchanged
```

This preserved the no-D2H rule and kept W2 grouped GEMM semantics unchanged. It was still not a win. The likely reason is that the original row-block quant kernel is very regular and small; adding a device-side count read plus row predicate did not reduce a visible full-runtime section and may have made the kernel shape less friendly.

5090 validation:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 release build for `bench_serving` and `deepseek_v4_e2e` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | aggregate steady TPOT avg `31.270ms`, p50 `30.660ms`, p95 `33.575ms` |
| fixed bench iterations | `31.220ms`, `31.342ms`, `31.248ms`, all hash `6346f03343d75a65` |

Drop decision: do not retain. Skipping empty rows inside a tiny regular kernel is not enough, and in this implementation it regressed sharply. Future `local_count` usage needs to remove a larger launch cluster or preserve a fully regular kernel shape; a branch inside W2 quant is the wrong granularity.

### Rejected: shrink grouped GEMM row-tile launch by seq_len

vLLM's CUTLASS path passes logical per-expert `problem_sizes`, while PegaInfer's TileLang grouped FP4 launch uses a host grid of:

```text
grid.x = output tiles
grid.y = ceil(num_expanded / 32)
grid.z = local_experts
```

For fixed MP8 decode, `num_expanded = global_seq_len * topk = 8 * 6 = 48`, but any one expert can receive at most one route per global token, so `expert_m <= global_seq_len = 8`. The attempted change added a `max_expert_rows` argument and launched grouped W13/W2 with:

```text
grid.y = ceil(max_expert_rows / 32)
```

The runtime passed `plan.routed.seq_len` as the upper bound. This is batch-general for top-k routing with unique experts per token; it is not a bs=1 or seq_len=1 special case.

Standalone W13 microbench on the sparse decode-like shape was exact, but it showed the smaller row-tile bound does not move the important cost when `local_experts=32`:

| Shape | Variant | W13 time |
| --- | --- | ---: |
| `rows=48, experts=32, max_expert_rows=8` | optimized row-tile bound | `0.878124ms` |
| `rows=48, experts=32, max_expert_rows=48` | original row-tile bound | `0.882815ms` |
| `rows=8, experts=1, max_expert_rows=8` | active-expert upper-bound shape | `0.063523ms` |
| `rows=16, experts=2, max_expert_rows=8` | active-expert upper-bound shape | `0.063532ms` |
| `rows=24, experts=3, max_expert_rows=8` | active-expert upper-bound shape | `0.124922ms` |

Runtime validation on 5090:

| Check | Result |
| --- | --- |
| local `cargo fmt --check` | passed |
| local `git diff --check` | passed |
| local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` | passed |
| release `deepseek_v4_e2e` | `All 20 DeepSeek V4 exact cases passed` |
| fixed bench JSON | per-iteration steady TPOT avg `28.504ms`, `28.460ms`, `28.735ms`; all hash `6346f03343d75a65` |

Drop decision: do not retain. The exactness proof is useful, but shrinking only `grid.y` does not address the dominant grouped W13/W2 cost; the `local_experts=32` scheduling dimension still launches many expert slots. The next useful prototype must reduce or repack the expert dimension itself, or use a persistent/problem-size-aware grouped scheduler. A host-known row upper bound alone is not enough.

### Diagnostic: route sparsity at fixed token

A temporary hard-coded diagnostic at `start_pos == 80` synchronized the rank stream and copied `expert_indptr` to host once per `(rank, layer)`. This was intentionally not retained because it performs D2H and stream sync in the decode path. The diagnostic ran with:

```bash
target/release/bench_serving \
  --model-path /data/DeepSeek-V4-Flash \
  --format json \
  request \
  --prompt-len 1 \
  --output-len 96 \
  --warmup 1 \
  --iters 1 \
  --seed 42
```

Route stats from `/tmp/dsv4_moe_route_stats.log` covered `43 layers * 8 ranks = 344` rows:

| Metric | Value |
| --- | ---: |
| route capacity per rank/layer | `48` |
| `local_count` min / avg / p50 / p95 / max | `0 / 6.0 / 8 / 16 / 40` |
| nonempty local experts min / avg / p50 / p95 / max | `0 / 0.75 / 1 / 2 / 5` |
| max rows per local expert min / avg / p50 / p95 / max | `0 / 4.35 / 8 / 8 / 8` |

Per-rank averages were also sparse: rank-local `local_count` ranged from `4.19` to `7.81` rows on average, and average nonempty experts ranged from `0.53` to `0.98` out of `32` local experts.

Interpretation: the fixed decode route is extremely sparse at the rank-local expert level. Most rank/layer pairs have zero or one active local expert, and nonempty experts usually have exactly eight rows because the gathered MP8 token batch follows the same token trace across ranks. This confirms that empty expert/empty CTA work is real. It also explains the failed valid-row experiment: skipping rows inside W2 quant is too small and too late. A useful next MoE scheduler must reduce or reshape expert-level grouped GEMM work while preserving expert-major locality; route-row W13 and in-kernel row predicates are the wrong granularity.

The existing W13 grouped FP4 microbench gives an upper-bound sanity check for this direction:

| Shape | W13 one GEMM | Notes |
| --- | ---: | --- |
| `rows=48, experts=32` | `0.399417ms` | Capacity-like shape with many expert slots. The bench distribution is not as sparse as the real route, so treat as a pessimistic capacity proxy. |
| `rows=8, experts=1` | `0.061476ms` | Typical one-active-expert rank/layer shape from the route diagnostic. |
| `rows=16, experts=2` | `0.061866ms` | Approx p95 active-expert count. |
| `rows=24, experts=3` | `0.061477ms` | Upper tail seen in several layers. |

Interpretation: the next plausible MoE win is not another scalar/row predicate inside the existing capacity launch. The useful prototype should make grouped W13/W2 see active expert problem sizes, ideally using compact active pointer/indptr metadata, while keeping W13 expert-major. The hard production constraint remains host launch sizing: a GPU-only active list cannot directly shrink grid dimensions without D2H, CUDA dynamic parallelism, or a fixed small upper-bound launch. The microbench says the direction is worth prototyping; it does not yet solve the runtime launch-sizing problem.

Evidence required for each adoption step:

- vLLM/SGLang source location and whether we copied the decomposition, the kernel shape, or only the validation idea.
- standalone microbench with fuzz against the current PegaInfer baseline.
- exact E2E `20/20`.
- fixed JSON bench with token hash `6346f03343d75a65`.
- repeated TPOT range, not a single fast run.

## Rejected Patterns

These are worth remembering because they looked plausible:

| Attempt | Result | Lesson |
| --- | --- | --- |
| Route W13 directly from token activations | Exact-safe, hash-stable, but regressed to `33.14-33.36ms/token` | Removing BF16 expand and reducing W13 quant rows is not enough if the GEMM shape loses expert-major locality. |
| Fuse expand with W13 activation quant | Bitwise microbench PASS and `2.5-3.2x` faster locally, but runtime repeated at `27.20-28.71ms/token` | Local microbench wins that remove only a tiny section can disappear in full decode; require full-runtime proof before retaining. |
| Skip W2 SwiGLU+quant rows after GPU `local_count` | Exact-safe and hash-stable, but regressed to `31.22-31.34ms/token` | Adding a device-side count read and row predicate inside a tiny regular kernel is the wrong granularity. |
| Shrink grouped GEMM row-tile bound to `seq_len` | Exact-safe and hash-stable, but regressed to `28.46-28.74ms/token` | Empty row tiles are not the dominant grouped FP4 cost; the expert scheduling dimension remains too coarse. |
| Fuse final HC head plus RMSNorm | Exact-safe but regressed TPOT | Saving small launches can lose to worse reduction/kernel shape. |
| Reuse deterministic window top-k across layers | Exact-safe, no stable long-bench win | Launch-count reduction alone is weak evidence. |
| Fuse KV RoPE plus no-PE quant | Exact-safe, regressed short decode | Combining tiny kernels can hurt scheduling/occupancy. |
| Hand-written decode HC mixes kernel | Exact-safe, slower than cuBLAS path | cuBLAS small GEMV remained better on this shape. |
| Isolated final logits scratch | Correct but noisy/regressive in repeated runs | Isolated storage movement near sampling boundary did not address the dominant per-layer allocation/skew structure. |
| Host-sized active-tile count for grouped MoE | Not used | Pulling active counts D2H would reintroduce hot-path synchronization. |

## Profiling And Benchmark Rules

### Token trace first

Always compare generated-token hashes before comparing TPOT. DeepSeek V4 routing and expert balance depend on token sequence. The bench JSON now records per-iteration timing and generated-token trace.

### Repeated fixed bench before claiming a win

The shared W13 branch showed a wide fixed-bench band with the same token hash: `32.220ms`, `30.061ms`, and `28.159ms` across consecutive 5090 repeats. A single sub-`30ms/token` run is therefore only evidence that the code path can enter that band, not that the optimization goal is achieved. Record multiple full JSON runs and prefer ranges over point estimates.

After the repeat series, idle `nvidia-smi` showed all GPUs back at `180MHz` SM / `405MHz` memory with no active throttle reason and no remaining `bench_serving`/`deepseek_v4`/`nsys` process. A follow-up fixed bench with `nvidia-smi --loop-ms 200` sampling produced steady TPOT `28.784ms` with all token hashes still `6346f03343d75a65`. Active-window clock averages were roughly `2622-2699MHz` SM and `13.7-13.8GHz` memory across ranks, with throttle reason always `0x0000000000000000`. That weakens the simple “slow run equals thermal/power throttle” hypothesis. The next diagnostic should add per-rank decode stage timestamps around attention local, collectives, routed MoE, shared expert, and logits to catch rank-arrival skew directly.

A temporary hard-coded trace for `start_pos == 80` synchronized each rank stream between broad decode stages, then logged per-rank totals. The trace build itself perturbs one steady token, so use it only for attribution. In one fixed bench with trace enabled, TPOT stayed in the fast band at `29.630ms` and all token hashes stayed `6346f03343d75a65`.

Trace summary across the 5 traced requests (`2` warmup + `3` measured), aggregated over all layers for a single steady decode token:

| Stage | Median / avg shape | Cross-rank range observed |
| --- | ---: | ---: |
| `attention_local` | avg `16.756ms` | `15.099-17.846ms` |
| `attention_collective_post` | avg `3.741ms` | `2.990-6.225ms` |
| `moe` | avg `15.112ms` | `14.779-15.401ms` |
| `hc_attn_pre` | avg `2.179ms` | `2.063-2.469ms` |
| `hc_ffn_pre` | avg `2.248ms` | `2.158-2.345ms` |
| `ffn_post` | avg `0.533ms` | `0.463-0.604ms` |
| `final_logits` | avg `0.268ms` | `0.241-0.352ms` |

Interpretation: the current shared/routed MoE path is not the largest source of rank skew in this trace. MoE is still a large absolute cost, but its rank range is only about `0.6ms`; the larger variance comes from attention local and attention collective+HC-post windows. The next optimization pass should not blindly keep fusing MoE kernels before explaining attention-local variability.

A narrower per-layer trace at the same hard-coded `start_pos == 80` logged `43 layers * 8 ranks * 5 requests = 1720` rows in `/tmp/dsv4_layer_trace_bench.log`. The fixed bench stayed on the same token trace (`6346f03343d75a65`) and measured around `30.84ms/token`, but this run included per-layer stream synchronizations and is attribution-only.

| Stage | Avg per layer | Approx 43-layer sum | p95 per layer | Max per layer |
| --- | ---: | ---: | ---: | ---: |
| HC attention pre-norm | `0.052ms` | `2.22ms` | `0.067ms` | `0.105ms` |
| Attention local | `0.399ms` | `17.16ms` | `0.489ms` | `0.724ms` |
| Attention all-reduce + HC post | `0.088ms` | `3.80ms` | `0.218ms` | `0.483ms` |
| HC FFN pre-norm | `0.052ms` | `2.24ms` | `0.061ms` | `0.077ms` |
| MoE, including AG/RS + shared expert | `0.354ms` | `15.23ms` | `0.390ms` | `0.428ms` |
| FFN HC post | `0.012ms` | `0.50ms` | `0.021ms` | `0.040ms` |

The largest single layer/request cross-rank ranges were `0.448ms` in layer `1` attention collective, `0.375ms` in layer `19` attention local, and `0.363ms` in layer `19` attention collective. There was no repeated multi-millisecond rank outlier in this per-layer view. That changes the next bet: stable sub-`30ms/token` is less likely to come from only rank-affinity or one MoE scalar cleanup, and more likely from reducing the largest absolute sections, namely attention local (`~17ms`) and MoE (`~15ms`), while keeping launch count and synchronization windows low.

### NCCL wall is wait-inclusive

Nsight Systems NCCL kernel wall time includes rank-arrival waiting. Treat NCCL rows as synchronization-window evidence unless rank-arrival skew and post-arrival tail have been separated. The rank-affinity work was selected because corrected f32 all-reduce grouping showed attention hidden all-reduce dominated by arrival skew, not post-arrival NCCL tail.

### Allocation proof

Full-process nsys attribution was not reliable enough for allocation proof:

- nsys-only `cuMemAllocAsync` attribution did not reconcile with application-visible symbols.
- CUDA event tracing can distort API counts.
- NCCL wall can dominate profile views while reflecting upstream skew.

The retained allocation evidence combines source-level inventory with `tools/cuda_api_counter.c`, an `LD_PRELOAD` counter that covers directly linked runtime/driver symbols and CUDA driver function-table lookup via `cuGetProcAddress`.

| API group | Baseline | Current |
| --- | ---: | ---: |
| `cudaMalloc` calls | `12944` | `136` |
| `cudaFree` calls | `12848` | `32` |
| `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` | noisy nsys-only attribution | `0/0/0` in counter |
| `cudaMallocAsync/cudaFreeAsync/cudaMemsetAsync` | not used | `0/0/0` |
| `cuGetProcAddress` replacements | not covered | `0` |

The counter exports base and `_ptsz` wrappers separately for `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async`. Do not share one stored real function pointer across base and `_ptsz` variants.

## Remote Workflow Notes

Remote test syncs should use touched-file `rsync -azR`. A full repository rsync with delete/excludes stalled for about 10 minutes during this work. A repeated mistake here was running multi-source `rsync` without `-R`, which copied `index.md`, `decode-performance.md`, `core.rs`, `moe.rs`, `state.rs`, `deepseek_quant.cu`, `ffi.rs`, and `swiglu_quant_bench.cu` into the remote repository root as basename files. Clean those accidental root files immediately and resend with `-R` so paths are preserved. Also, `cargo check` does not rebuild already-built release binaries; rebuild `deepseek_v4_e2e` and `bench_serving` before trusting remote validation.

Verified command set for this PR:

```bash
cargo fmt --check
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
cargo check --release -p pegainfer-server --features deepseek-v4
gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl
```

## Validation

Local:

- `cargo fmt --check`
- `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`
- `cargo check --release -p pegainfer-server --features deepseek-v4`
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers
- `git diff --check`
- pre-commit hooks on commit, including clippy

5090:

- `cargo fmt --check`
- `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`
- `cargo check --release -p pegainfer-server --features deepseek-v4`
- release `deepseek_v4_e2e`: `All 20 DeepSeek V4 exact cases passed`
- release fixed bench log `/tmp/dsv4_pr_driver_numa_bench.log`: steady TPOT avg `35.253ms`, p50 `34.800ms`, p95 `37.335ms`, first decode avg `33.743ms`, hash `6346f03343d75a65`
- current clean fixed bench log `/tmp/dsv4_clean_tpot_now.log`: per-iteration steady TPOT avg `29.944ms`, `29.907ms`, `29.896ms`, all hash `6346f03343d75a65`
- current exact E2E log `/tmp/dsv4_e2e_current.log`: `All 20 DeepSeek V4 exact cases passed`
- fused-clear exact E2E log `/tmp/dsv4_clear_fused_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- fused-clear fixed bench log `/tmp/dsv4_clear_fused_bench.log`: per-iteration steady TPOT avg `29.862ms`, `29.969ms`, `29.874ms`, all hash `6346f03343d75a65`
- fused-clear short nsys log `/tmp/dsv4_clear_fused_short_profile.txt`: `deepseek_moe_clear_mapping_kernel` replaces the old repeated clear kernel
- small-route exact E2E log `/tmp/dsv4_small_mapping_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- small-route fixed bench logs `/tmp/dsv4_small_mapping_bench.log` and `/tmp/dsv4_small_mapping_bench_repeat.log`: per-iteration steady TPOT avg `27.608ms`, `27.662ms`, `27.826ms`, then `27.698ms`, `27.693ms`, `27.644ms`; all hash `6346f03343d75a65`
- small-route short nsys log `/tmp/dsv4_small_mapping_short_profile.txt`: `deepseek_moe_local_mapping_small_kernel` is the decode-sized route mapping path
- rejected route-W13 exact E2E log `/tmp/dsv4_route_w13_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected route-W13 fixed bench log `/tmp/dsv4_route_w13_bench.log`: aggregate steady TPOT avg `33.217ms`, per-iteration `33.162ms`, `33.355ms`, `33.135ms`; all hash `6346f03343d75a65`
- rejected expand+act_quant exact E2E log `/tmp/dsv4_expand_act_quant_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected expand+act_quant fixed bench logs `/tmp/dsv4_expand_act_quant_bench.log` and `/tmp/dsv4_expand_act_quant_bench_repeat.log`: per-iteration steady TPOT avg `28.080ms`, `28.143ms`, `27.198ms`, then `28.509ms`, `28.712ms`, `28.474ms`; all hash `6346f03343d75a65`
- rejected W2 valid-row exact E2E log `/tmp/dsv4_valid_rows_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected W2 valid-row fixed bench log `/tmp/dsv4_valid_rows_bench.log`: aggregate steady TPOT avg `31.270ms`, per-iteration `31.220ms`, `31.342ms`, `31.248ms`; all hash `6346f03343d75a65`
- rejected grouped GEMM row-tile upper-bound exact E2E log `/tmp/dsv4_max_expert_rows_e2e.log`: `All 20 DeepSeek V4 exact cases passed`
- rejected grouped GEMM row-tile upper-bound fixed bench log `/tmp/dsv4_max_expert_rows_bench.log`: per-iteration steady TPOT avg `28.504ms`, `28.460ms`, `28.735ms`; all hash `6346f03343d75a65`
- `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
- `nm -D /tmp/cuda_api_counter.so` confirmed base and `_ptsz` wrappers

The benchmark process still prints the existing NCCL communicator abort panic during shutdown after JSON output and scheduler exit. Track that as shutdown cleanup, not decode TPOT evidence.

## Follow-ups

- Fix NCCL communicator shutdown.
- Move DeepSeek V4 off the temporary direct runtime into the scheduler/executor shape used by the rest of the engine.
- Revisit CUDA graph capture after pointer stability is broad enough.
- Keep MoE active-expert/tile-list work separate from allocation scratch; the next MoE win is likely reducing empty CTA/kernel work, not more host allocation cleanup.
