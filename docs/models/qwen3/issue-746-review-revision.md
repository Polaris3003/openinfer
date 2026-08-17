# Qwen3 issue #746 reviewer revision plan

> **TL;DR:** #746 收缩为 TP1 decode QKV-only：先用 current-head、order-balanced
> CUDA Event A/B 量化 packed GEMM、`split_qkv` 和 gate/up 各自成本，再决定保留
> copy 还是让 QK-norm/RoPE 与 KV scatter 直接消费 packed QKV；生产默认只对最终
> 实测通过的 GPU SM 与 4B/8B geometry 放行，其他环境 fail closed。
>
> **Last touched:** 2026-08

## Preparation

- **Read**:
  - `docs/index.md` — Qwen3 的模型文档、accuracy、kernel report 和 graph export
    都归入 `models/qwen3/`，本计划需要新增索引项。
  - `docs/models/qwen3/cuda-graph-png.md` — 当前 `507 kernels / 506 edges /
    14-kernel layer` 是历史 split topology 的实测记录；最终 projection topology
    确定后必须刷新当前契约。
  - `docs/models/qwen3/accuracy-gate.md` — HF gate 已覆盖 0.6B–32B，生产
    qualification 不能由 4B/8B 自动外推到所有 Qwen3 geometry。
  - `docs/conventions/bench-regression.md` — 旧 `2%/3%` 是回归调查线，不是
    #746 的硬性优化放行线；本轮以配对重复、置信区间和端到端同向为准。
  - `docs/playbooks/profiling-guide.md` — CUDA Graph node trace 只用于归因；绝对
    性能使用无 profiler 的 CUDA Event 和 HTTP A/B。
  - `pegainfer-qwen3/src/kernel_bench.rs` 与
    `pegainfer-qwen3/src/bin/qwen3_kernel_report.rs` — 已有生产 shape、cuBLASLt
    tuning、CUDA Event 和 cold-L2 基础，应扩展而不是新增通用 benchmark 框架。
  - `pegainfer-qwen3/src/{batch_decode.rs,batch_decode_buffers.rs,batch_decode_dag.rs}`
    — 当前 fused path 先 materialize compact Q/K/V，再由 QK-norm/RoPE 和 paged
    attention 消费。
  - `pegainfer-kernels/src/ops/attention.rs` — decode attention 在 attention 前将
    K/V scatter 到 paged cache；这是消除独立 K/V materialization 的最窄边界。
- **Relevant history**:
  - SM86 历史组件结果仅 TP1 decode QKV 独立 qualified（E2E `+2.32%`）；
    gate/up 为 `+0.61%`，不应与 QKV 绑定为一个生产 topology。
  - SM89 combined path 的 HTTP TPOT/throughput 改善为约 `0.98%–1.80%`，每个
    cell 只有两组保留结果，且没有 current-head projection CUDA Event raw data。
  - 当前五投影 LoRA fixture 只存在于外部验证产物；仓库 fixture 仍是 q/v-only，
    不能长期覆盖 packed QKV 的 Q/K/V logical offsets。
  - vLLM Qwen3 使用一次 `QKVParallelLinear` 后按最后一维创建 Q/K/V views；
    PegaInfer 的 `HiddenStates` 没有通用 stride 语义，因此只能保留 measured copy，
    或在 Qwen3 decode consumer 边界显式支持 packed stride。
- **Plan**:
  1. 将生产 topology 收缩为 `Split | FusedQkv`，gate/up 无条件恢复 split。
  2. 在现有 kernel report 基础上增加 current-head、order-balanced projection A/B。
  3. 先量化 standalone `split_qkv`；根据数据决定保留 copy 或进入 packed-consumer
     分支，不同时实现两套长期路径。
  4. 仅对白名单中的实测 `(SM, geometry)` 选择 `FusedQkv`；其余 fail closed。
  5. 精简低价值 copy tests，固化五投影 LoRA 和必要的 row-offset coverage。
  6. 重跑 correctness、CUDA Event、direct decode、HTTP 和 graph export，最后更新
     PR evidence 与 graph 文档。
- **Risks / open questions**:
  - 最终 QKV-only E2E 收益可能低于现有 combined path；若置信区间不能排除零，
    本 PR 不应改变生产默认。
  - packed consumer 会触及 QK-norm/RoPE 与 KV scatter FFI；只有 copy 数据证明
    materialization 是主要损耗时才承担这部分实现成本。
  - 当前可用 GPU 是 SM89；若没有 current-head SM86 重跑，最终 selector 不得
    因历史数据而放行 SM86。

## Target production contract

### Scope

本 PR 最终只允许改变 Qwen3 decode QKV：

```text
Split:
  q_proj GEMM + k_proj GEMM + v_proj GEMM

FusedQkv:
  packed qkv_proj GEMM + measured unpack/consumer path
```

以下路径保持原实现：

- gate/up：始终 split；
- prefill 与 unified：始终 split；
- TP2 及更高 TP：始终 split；
- DFlash、decode overlap、`Pin`、`PerToken`：始终 split。

不增加用户 flag，不增加跨模型 planner，不把本轮实验规则扩展成通用 runtime
autotuner。

### Path type

生产状态只保留一个 model-local enum：

```rust
enum DecodeProjectionPath {
    Split,
    FusedQkv,
}
```

选择函数保持纯函数，但必须显式接收选择所需事实：TP、numeric policy、overlap、
DFlash、device SM 和 Qwen3 projection geometry。日志打印 resolved path 与
fail-closed reason，不能只打印布尔值。

### Qualification policy

`FusedQkv` 必须同时满足：

1. TP1；
2. `NumericPolicy::Tuned`；
3. decode overlap off；
4. DFlash off；
5. `(SM, hidden_size, q_dim, kv_dim, num_layers)` 与 current-head 实测记录完全
   匹配；
6. 该记录的 CUDA Event 与端到端 gate 均通过。

首轮只计划验证 SM89 上的 Qwen3-4B 与 Qwen3-8B。SM86 只有历史结果，不在新的
allowlist 中；以后增加硬件必须以独立证据扩表。

不采用启动时 topology benchmark，原因是它会把测量噪声变成生产配置、增加双份
临时资源，并使 CUDA Graph topology 在相同部署间可能漂移。若未来确实需要通用
autotune，应作为独立设计任务处理。

## CUDA Event A/B design

### Reuse boundary

扩展现有 `kernel_bench` / `qwen3_kernel_report`，增加窄的 projection topology
case；不恢复之前的大型 Python suite，不让生产 API 暴露 benchmark-only flag。

报告必须使用：

- 与生产相同的 BF16 shape；
- 与 executor 相同的 cuBLASLt tuning；
- cold-L2 sweep 或跨 layer weight rotation；
- 同一 CUDA stream 上的 CUDA Event；
- 同进程 A/B，避免两个 binary 的启动状态差异；
- JSON raw output，不能只输出汇总百分比。

### Cells

每个 Qwen3-4B/8B、batch 1/8 分别测：

| Component | A: split | B: candidate |
| --- | --- | --- |
| QKV aggregate | q + k + v 三次 GEMM | packed GEMM + `split_qkv` |
| QKV GEMM only | q + k + v 三次 GEMM | packed GEMM |
| Materialization | 无 | standalone `split_qkv` |
| MLP projection | gate + up 两次 GEMM | merged gate/up GEMM |
| MLP full boundary | split projection + split SwiGLU | merged projection + fused SwiGLU |

gate/up 数据只用于确认它继续 split，不进入生产 candidate。

### Order balancing

每个 cell 先 warm up，随后交替执行等量的：

```text
ABBA
BAAB
```

至少保留 10 对 block。每一次 Event duration、执行顺序、GPU clocks、temperature、
driver、CUDA toolkit、SM、selected cuBLASLt algorithm 和 git SHA 都写入 raw JSON。

报告以下统计但不替代 raw data：

- count、mean、p50、p95、p99、std；
- 每个 ABBA/BAAB block 的 paired delta；
- paired median improvement；
- paired bootstrap 95% confidence interval；
- standalone copy 占 fused aggregate 的比例。

### Component decision gate

QKV candidate 只有在 batch 1 和 8 同时满足以下条件时继续：

1. paired improvement 的 95% CI 下界大于 0；
2. packed GEMM + materialization 的方向与 packed GEMM-only 方向一致；
3. raw block 没有持续性的顺序偏置；
4. standalone copy 已明确计入，不使用 GEMM-only 数字代替完整路径；
5. 4B/8B 都通过，否则只允许通过的 geometry 进入 qualification table。

不把旧 `2%` 回归调查线机械当作硬阈值；但若收益与同次 paired noise 同量级，
按维护成本 fail closed。

## Copy decision

### Branch A: retain `split_qkv`

仅当完整 QKV aggregate 通过上述 component gate 时保留当前 materialization：

- CUDA kernel 与 checked wrapper 保留；
- production shape + tail shape bitwise-copy test 保留；
- standalone latency 进入 PR 表格；
- selector 仍按 measured SM/geometry gate；
- 不声称 packed GEMM-only 收益等于生产收益。

### Branch B: packed consumer

如果 standalone copy 吃掉主要收益或完整 aggregate 不能稳定通过，则删除独立
`split_qkv`，采用 Qwen3-local packed consumer：

1. packed QKV GEMM 输出 `[Q; K; V, batch]`；
2. Q/K/V LoRA delta 按 `0 / q_dim / (q_dim + kv_dim)` 写入 packed buffer；
3. 新的 packed QK-norm/RoPE op 读取 strided Q/K：
   - Q 完成 norm/RoPE 后写入 compact Q，供 FlashInfer attention 使用；
   - K 完成 norm/RoPE 后写回 packed K region；
4. paged KV scatter 直接读取 packed K/V base pointer，token stride 使用
   `qkv_dim`，不再分配 compact K/V；
5. attention kernel 本身不改，仍从 paged cache 读取 K/V。

实现限制：不为整个 tensor system 增加通用 strided tensor abstraction；只给
Qwen3 decode 的 QK-norm/RoPE 与 KV scatter 增加明确的 packed contract。

buffer 应以 enum 表达合法状态：

```text
Split { q, k, v }
FusedQkv { qkv, q }
```

若 Branch B 的 correctness 或 end-to-end gate 未通过，则删除该 branch 并保持
生产 `Split`，不能退回“收益未知但默认开启”。

## File-level change map

### Always required

- `pegainfer-qwen3/src/projection_fusion.rs`
  - `FusedQkvGateUp` 改为 `FusedQkv`；
  - 加入 measured SM/geometry qualification；
  - selector tests 覆盖一条 allow 和一组 fail-closed 边界。
- `pegainfer-qwen3/src/batch_decode.rs`
  - gate/up 恢复无条件 split；
  - QKV 根据两态 enum 执行；
  - LoRA 顺序与最终 packed/copy branch 对齐。
- `pegainfer-qwen3/src/batch_decode_buffers.rs`
  - 删除 fused gate/up scratch；
  - projection scratch 只表示 Split 或 FusedQkv 合法状态。
- `pegainfer-qwen3/src/executor.rs`
  - gate/up tuning 恢复 split shape；
  - QKV 只为 resolved path tune 对应 shape。
- `pegainfer-qwen3/src/weights.rs`
- `pegainfer-qwen3/src/weights/load.rs`
  - model-load 时解析 device SM 与 geometry，保存最终 path 和 reason。
- `pegainfer-qwen3/src/kernel_bench.rs`、
  `pegainfer-qwen3/src/bin/qwen3_kernel_report.rs`
  - 增加 order-balanced projection topology A/B 与 raw JSON。

### Remove with gate/up scope

- 删除 production fused gate/up 分支、buffer、tuning 和相关 DAG call；
- 删除只用于 gate/up fused production 的测试；
- 已在 base 中存在、且被其他路径使用的通用 SwiGLU operator不删除。

### Conditional for Branch A

- 保留 `split_qkv_cuda`、FFI、checked op、call spec 与 model-report provider；
- copy matrix 缩为一个生产 shape和一个 tail shape；
- 删除 helper self-test；保留 pre-launch shape rejection。

### Conditional for Branch B

- 删除 `split_qkv_cuda` 及其完整 FFI/report/test surface；
- 新增 packed QK-norm/RoPE 与 packed KV scatter 的窄 FFI；
- 增加 packed stride、Q/K 数值、KV cache landing 和 graph replay tests。

### LoRA

- Branch A 在 `split_qkv` 后继续复用既有 grouped Q/K/V delta，因此没有 packed
  LoRA row offset；保留现有 q/v non-zero golden 足以覆盖本 PR 改动的两侧输出，
  不为未改变的 gate/up split 路径扩大 fixture；
- 删除只服务于 fused gate/up 的 row-offset test；
- 只有 Branch B 才增加 packed Q/K/V 三个 row offset 的小型 operator test，并在
  那时按实际 consumer 契约决定是否需要扩展 golden fixture。

## Correctness gates

最终 HEAD 必须通过：

1. format、diff check、locked metadata、Clippy、sm80 CUDA compile；
2. QKV production shape + tail shape operator gate；
3. Qwen3-4B 和 8B HF golden：TP1/TP2、eager/CUDA Graph；
4. 现有 non-zero q/v LoRA golden：TP1/TP2；Branch B 才增加 packed offset gate；
5. eager 与 CUDA Graph greedy output parity；
6. TP2 concurrent decode，且日志确认 resolved `Split`；
7. selector 对未测 SM、未测 geometry、TP2、Pin/PerToken、overlap、DFlash
   全部 fail closed；
8. Branch B 额外验证 packed K/V 写入 paged cache 与 split baseline 一致。

任何 correctness gate 失败都直接保持 `Split`，不调整 tolerance 为优化让路。

## End-to-end performance gates

CUDA Event 通过后才运行端到端：

- model：Qwen3-4B、Qwen3-8B；
- TP：TP1 candidate，TP2 split negative control；
- phase：decode 为决策项，prefill/unified 为 non-target regression；
- execution：eager、CUDA Graph；
- batch/concurrency：1、8；
- HTTP：固定 prompt/output，`ignore_eos=true`，记录成功数、token count、hash；
- direct：记录每步 CUDA Event TPOT；
- repeats：至少两组完整 order-balanced baseline/candidate 交错运行。

生产放行要求：

1. component CUDA Event gate 通过；
2. direct eager 与 graph 在 batch 1/8 均同向；
3. HTTP TPOT 和 output throughput 在 concurrency 1/8 均同向；
4. paired 95% CI 排除零，且收益高于同次运行的测量噪声；
5. prefill/unified、TP2、显存/KV block budget 没有实质回归；
6. 只把实际通过的 `(SM, geometry)` 写入 selector qualification table。

如果最终仍只有约 1% 且无法排除噪声，本 PR保留报告和正确性结论，但生产默认
继续 `Split`。

## Documentation and submission

最终 topology 确定后再更新：

- `docs/models/qwen3/cuda-graph-png.md`
  - `Last touched` 更新为 `2026-08`；
  - `507/506/14` 明确标记为历史 split topology；
  - 在 GPU 上重新导出后记录当前默认图；若没有真实导出则删除当前精确数字，
    不用公式推算物理 node count。
- `docs/index.md`
  - 更新 graph row，避免继续承诺过期 node count；
  - 本计划完成后更新其 TL;DR row。
- PR description
  - 分开列出 current-head component raw data、QKV-only E2E、correctness、
    qualification table 和 fail-closed 范围；
  - gate/up 数据只作为 `KEEP_SPLIT` 证据；
  - 明确 standalone materialization cost；
  - 不把历史 SM86 数据写成 current-head 结果。

历史保持两个逻辑提交，并在最终 rebase 后重新 sign-off；不增加独立 format-fix 或
benchmark-output commit。Raw artifacts 外置并提供 SHA256，稳定 fixture 与可复用
report code进入仓库。

## Stop conditions

出现任一条件就停止扩大实现：

- packed GEMM + copy 在 batch 1 或 8 的 paired CI 不排除零；
- Branch B 仍无法让完整 QKV aggregate 稳定获益；
- QKV-only HTTP/direct 收益与测量噪声同量级；
- 为通用 stride 或 runtime autotune 需要修改跨模型 tensor/runtime contract；
- correctness 只能通过放宽现有 tolerance 才成立。

这些情况的正确结论是生产保持 split，而不是继续增加复杂度追回已经消失的收益。

## Execution Log

### Step 1: 创建 reviewer revision 测试分支

- 从 PR 分支 `feat/qwen3-fused-decode-projections` 的
  `ee4488b0d616094e860e69d0ac4aff03d4928b2a` 创建
  `feat/qwen3-fused-qkv-review-revision`；
- 保留原工作区中未暂存的计划文档、PR comment 和验证脚本，不将其混入分支基线；
- 结果：success。

### Step 2: 收缩 production topology

- `DecodeProjectionPath` 收缩为 `Split | FusedQkv`；
- gate/up 恢复无条件 split，并删除 fused gate/up buffer、DAG call 与 tuning；
- selector 按 TP、numeric policy、overlap、DFlash、device SM 和完整 geometry
  fail closed，日志包含 resolved path 与 reason；
- 当前测试分支只列 SM89 Qwen3-4B/8B 候选 geometry，最终保留项由本轮数据决定；
- 结果：success，format 与 diff check 通过。

### Step 3: 增加 current-head projection A/B

- 扩展 `qwen3_kernel_report projection-ab`，覆盖 4B/8B、batch 1/8；
- 同一进程、同一 stream 和相同 tuned weight 下测量 QKV full、QKV GEMM-only、
  `split_qkv` incremental/standalone、gate/up GEMM-only 与完整 MLP；
- 每个比较交替 ABBA/BAAB，最少 10 blocks，逐次 cold-L2 CUDA Event 原始值；
- JSON 包含 raw order、温度/时钟 telemetry、distribution、paired improvement 和
  deterministic bootstrap 95% CI；
- copy test 已缩为一个生产 shape 与一个 tail，并保留 pre-launch rejection；
- 结果：implementation complete，本机无法执行 GPU 编译和 runtime gate。

### Unexpected

- macOS 本机 `cargo check` 在进入本 PR Rust type-check 前被既有 Linux-only
  `rdma-mummy-sys`（缺少 `endian.h`/`linux/types.h`）和 CUDA GPU detection 阻断；
  这不是代码编译结论。GPU Linux 主机必须运行 release check/Clippy 与新 reporter。

## Next action

在 Linux GPU 主机运行 release check、correctness gates 与 projection A/B。拿到
copy 成本之前不进入 packed-consumer 实现；根据每个 SM89 geometry 的 current-head
数据保留或删除 qualification row。
