# Qwen3 fused projection A/B on RTX 4090

> **TL;DR:** Issue #746 fused QKV and gate/up paths passed all correctness gates; fusing both together improves TP1 decode TPOT by 2.35% and throughput by 2.30% across all repeats, so production enables that exact case and fails closed to split elsewhere.

## Test boundary

- Hardware: 2 × RTX 4090 24 GiB, SM89; driver 580.105.08; CUDA 12.6.
- Model: Qwen3-4B; TP1 and TP2.
- Source commit measured: `a8ef9286eb329b8106157d2ca36956ad87f48d10`.
- Matrix: split, QKV-only, gate/up-only, and both; prefill and decode; concurrency 1 and 8; two repeats.
- Correctness: HF and LoRA goldens, eager and CUDA Graph, TP1 and TP2. All 21 cells passed; the complete runner finished 104/104 commands.
- Promotion thresholds: prefill 3%, decode 2%; both repeats must improve in the same direction and kernel evidence must agree.

## End-to-end combination result

Positive means lower TPOT/latency or higher throughput than split. The deployable unit is the complete projection topology, not either projection family in isolation.

| TP | Phase | Projection topology | Mean latency change | Mean throughput change | Repeat direction | Decision |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | prefill/unified | both fused | -1.30% | -1.23% | 0/4 positive | split |
| 1 | decode | both fused | +2.35% | +2.30% | 4/4 positive | **fuse both** |
| 2 | prefill/unified | both fused | -0.91% | -0.34% | inconsistent | split |
| 2 | decode | both fused | +0.79% | +1.00% | inconsistent | split |

TP1 decode measurements were:

| Concurrency / repeat | Split TPOT | Fused TPOT | TPOT change | Split output tok/s | Fused output tok/s | Throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 / r1 | 9.820 ms | 9.603 ms | +2.21% | 99.34 | 101.24 | +1.91% |
| c1 / r2 | 9.816 ms | 9.610 ms | +2.10% | 99.46 | 101.29 | +1.84% |
| c8 / r1 | 12.059 ms | 11.760 ms | +2.48% | 647.99 | 665.90 | +2.76% |
| c8 / r2 | 12.068 ms | 11.752 ms | +2.62% | 648.11 | 665.61 | +2.70% |

Kernel reports agree with the TP1 decode result: summed QKV + SwiGLU projection time improved by 5.22% at batch 1 and 7.81% at batch 8.

## Diagnostic isolation

Positive means lower latency than split.

| Projection | Phase | TP | Mean E2E change | Repeat direction | Kernel direction | Decision |
| --- | --- | ---: | ---: | --- | --- | --- |
| QKV | prefill/unified | 1 | -1.02% | inconsistent | regression | keep split |
| gate/up | prefill/unified | 1 | -0.43% | inconsistent | improvement | keep split |
| QKV | decode | 1 | +1.38% | consistent | improvement | below standalone threshold |
| gate/up | decode | 1 | +1.12% | consistent | improvement | below standalone threshold |
| QKV | prefill/unified | 2 | -1.51% | inconsistent | regression | keep split |
| gate/up | prefill/unified | 2 | +0.30% | inconsistent | improvement | keep split |
| QKV | decode | 2 | -1.41% | inconsistent | improvement | keep split |
| gate/up | decode | 2 | +0.27% | inconsistent | regression | keep split |

## Numerical diagnosis and production policy

The split/deinterleave kernel is a BF16 bitwise copy. QKV differences originate in cuBLASLt choosing a different accumulation/reduction order for one larger GEMM than for three row-range GEMMs: TP1 batch 1 was bit-exact; TP1 batch 8 had mean absolute error `3.651e-5` and maximum `0.00195312`. Gate/up output was bit-exact in the projection reports. HF and LoRA goldens, eager and CUDA Graph paths all passed under the repository tolerances.

Production therefore uses one internal, non-configurable whitelist: enable both fused projections only for exact Qwen3-4B geometry, TP1 decode, `NumericPolicy::Tuned`, decode overlap off, and DFlash off. TP2, prefill/unified, batch-invariant policy, overlap, DFlash, and other model geometries automatically retain split GEMMs. This keeps tests and serving on the same default path and prevents unmeasured combinations from being selected.
