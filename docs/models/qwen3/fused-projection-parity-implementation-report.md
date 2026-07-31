# Qwen3 fused projection 候选实现报告（Issue #746）

> **TL;DR:** 本次改动为 Qwen3 增加了彼此独立、构造期固定的 QKV 与 gate/up 融合候选路径，覆盖 decode、prefill、unified、verify、LoRA、CUDA Graph trace 和 TP-aware benchmark；同时补齐真实权重 TP-rank 数值/算法报告与 fail-closed 端到端验证套件。默认 `Auto` 白名单仍为空。代码已通过格式、diff、Cargo metadata、Python 语法/单测和完整 dry-run，但尚未在 Linux CUDA 上完成 type-check、TP1/TP2 正确性、五 projection LoRA fixture 重生成和 64-run 性能复测，当前状态仍是“候选与证据工具完成，生产资格未建立”。
>
> **Last touched:** 2026-07

Issue: [openinfer-project/openinfer#746](https://github.com/openinfer-project/openinfer/issues/746)

## Preparation

- **Read**:
  - `docs/index.md` — 确认报告属于 `models/qwen3/`，并复用现有 Qwen3 accuracy、TP、kernel 与 benchmark 文档作为约束。
  - `docs/models/qwen3/fused-projection-parity-plan.md` — 计划规定两个 projection 必须独立、构造期决策、默认 fail-closed，并且不能用放宽 HF tolerance 代替正确性。
  - 本分支全部未提交 diff — 按配置流、buffer、forward、kernel、测试和 benchmark 六条链核对实际实现，而不是只复述计划。
- **Relevant history**:
  - 历史 fused QKV 曾因 TP shard-local GEMM 数值路径改变而翻转 greedy token。
  - gate/up 历史上也曾因一个 combined GEMM 与两个 row GEMM 的 BF16 reduction 差异导致输出 token 改变。
  - 因此本次实现只建立候选路径；是否默认启用必须由当前 Qwen3-4B、当前 TP shape、当前 GPU/cuBLAS 数据决定。
- **Plan**:
  1. 从用户 CLI 一直追踪到 model 内 resolved policy，说明策略何时固定、哪些模式会拒绝。
  2. 分别解析 QKV、gate/up 在 decode 与 prefill/unified 中的真实数据流、buffer 和 LoRA 写入位置。
  3. 核对 TP tuning、CUDA Graph、trace、HF/LoRA gate 与 benchmark 是否走生产路径。
  4. 将“已实现”“静态验证”“必须在 GPU 完成”拆开报告，并列出实现与原计划的偏差。
- **Risks / open questions**:
  - 当前本机无法完成 Rust type-check 或任何 CUDA 执行，不能把静态审查写成运行通过。
  - 源码行号会随继续开发漂移，报告以符号名和文件为主要定位。

## 1. 执行结论

### 1.1 这次改动解决了什么

改动没有直接把 fused projection 设为默认，而是先搭好一套可以安全实验、独立归因和随时回滚的生产同构路径：

```text
用户选择
  ├─ QKV:    auto | split | fused
  └─ gate/up auto | split | fused
          │
          ▼
模型加载前解析支持矩阵
          │
          ▼
ResolvedProjectionFusion
  ├─ decode_qkv
  ├─ decode_gate_up
  ├─ prefill_unified_qkv
  └─ prefill_unified_gate_up
          │
          ├─ 决定分配哪一种 scratch
          ├─ 决定 tune 哪一种 GEMM shape
          └─ 决定 CUDA Graph 捕获哪一种拓扑
```

它使下面四种实验组可以在同一份生产代码上运行：

| 实验组 | QKV | gate/up | 用途 |
| --- | --- | --- | --- |
| split | 3 个 row GEMM | 2 个 row GEMM | 数值与性能 baseline |
| QKV-only | 1 个 full GEMM + split-copy | 2 个 row GEMM | 单独判断 QKV |
| gate-up-only | 3 个 row GEMM | 1 个 full GEMM | 单独判断 MLP |
| both | fused | fused | 判断组合收益与交互 |

### 1.2 当前不能声称什么

当前不能声称 Issue #746 已经完成生产放行，原因是：

- `Auto` 白名单为空，没有任何 `(projection, phase, TP)` 默认启用。
- 没有 Linux CUDA release type-check 结果。
- 没有 TP1/TP2 HF logits gate 结果。
- committed LoRA fixture 仍是旧的 q/v-only fixture。
- 没有 projection ULP/delta JSON。
- 没有 32-cell 端到端性能数据。
- 没有最终白名单决策。

正确的状态描述是：

> 候选代码和实验入口已经建立；默认路径保持 split；GPU 证据链尚未完成。

## 2. 为什么 QKV 与 gate/up 必须独立

Qwen3-4B 的 rank-local 矩阵尺寸如下：

| projection | TP1 本地 `M × K` | TP2 每 rank `M × K` |
| --- | ---: | ---: |
| Q | `4096 × 2560` | `2048 × 2560` |
| K/V | `1024 × 2560` | `512 × 2560` |
| fused QKV | `6144 × 2560` | `3072 × 2560` |
| gate 或 up | `9728 × 2560` | `4864 × 2560` |
| fused gate/up | `19456 × 2560` | `9728 × 2560` |

融合不是“把完全相同的三个 kernel 拼起来”。它把 cuBLAS 看到的 `M` 改了：

```text
TP1 QKV baseline: M = 4096、1024、1024
TP1 QKV fused:    M = 6144

TP2 QKV baseline: M = 2048、512、512
TP2 QKV fused:    M = 3072
```

cuBLASLt 会根据 `M/N/K`、GPU 架构和 workspace 选择 tile、stage、split-K 与 reduction 方案。`M` 改变后，累加顺序可能改变；BF16 每个元素即使只差 1 ULP，经过 36 层也可能放大到近似 tie 的 logits，并最终改变 greedy token。

QKV 与 gate/up 又是两组不同矩阵，不能用“both 通过/变快”推导两者都安全。因此本次 API 从一开始就保留两个独立控制位，而不是一个总开关。

## 3. 策略层：先解析，再分配，再执行

### 3.1 新增公共配置

文件：`openinfer-qwen3/src/projection_fusion.rs`

新增：

```rust
ProjectionFusionControl {
    Auto,
    Split,
    ForceFused,
}

Qwen3ProjectionFusionOptions {
    qkv,
    gate_up,
}
```

语义：

- `Split`：明确使用 baseline。
- `ForceFused`：用于诊断和 A/B；不支持时必须在构造阶段失败。
- `Auto`：只查询已经有正确性和性能证据的生产白名单。

辅助构造：

- `Qwen3ProjectionFusionOptions::split()`
- `Qwen3ProjectionFusionOptions::force_fused()`

### 3.2 内部 resolved 状态按 phase 拆分

同一文件中新增：

```rust
ResolvedProjectionFusion {
    decode_qkv,
    decode_gate_up,
    prefill_unified_qkv,
    prefill_unified_gate_up,
}
```

这意味着生产白名单未来可以出现部分启用，例如：

```text
TP1 decode QKV: fused
TP1 decode gate/up: split
TP1 prefill QKV: split
TP1 prefill gate/up: fused
```

虽然用户侧 `ForceFused` 会同时强制某个 projection 的 decode 与 prefill/unified，但内部状态没有把两个 phase 焊死，生产资格仍可按 phase 决策。

### 3.3 强制模式的 fail-closed 条件

`validate_force_supported` 只接受：

- geometry 精确等于 Qwen3-4B：
  - hidden `2560`
  - intermediate `9728`
  - Q heads `32`
  - KV heads `8`
  - head dim `128`
- TP world size 为 `1` 或 `2`
- `NumericPolicy::Tuned`
- `DecodeOverlap::Off`
- rank-local Q/KV/intermediate 维度非零

下面情况会在读取 `config.json` 后、加载 safetensor 权重前失败，而不是运行时偷偷 fallback：

- 其他 Qwen3 size
- TP4/TP8
- `NumericPolicy::Pin`
- `NumericPolicy::PerToken`
- Shared-SM decode overlap
- Green Context

### 3.4 `Auto` 为什么仍然是 split

`auto_whitelisted` 当前直接返回 `false`。这是本次最重要的安全控制：

- 新代码存在，不代表已经证明更快。
- force 模式能跑，不代表数值满足 HF gate。
- TP1 通过，不代表 TP2 算法相同。
- decode 变快，不代表长 prefill 的额外 scratch 和 TTFT 可接受。

因此正常用户不传任何新参数时，resolved 的四个字段全部是 `false`。

## 4. 配置如何穿过整个系统

配置链路如下：

```text
openinfer-server Args
  → Qwen3LaunchOptions
  → launch / launch_with_seed
  → scheduler::start_qwen3
  → Qwen3Executor::from_runtime_with_projection_fusion
  → ModelRuntimeConfig
  → Qwen3Model::from_safetensors_with_runtime
  → ResolvedProjectionFusion::resolve
  → Qwen3Model.projection_fusion
```

涉及文件：

- `openinfer-server/src/config.rs`
- `openinfer-server/src/main.rs`
- `openinfer-qwen3/src/lib.rs`
- `openinfer-qwen3/src/scheduler.rs`
- `openinfer-qwen3/src/executor.rs`
- `openinfer-qwen3/src/weights/load.rs`
- `openinfer-qwen3/src/weights.rs`

### 4.1 服务端 CLI

新增：

```text
--qwen3-qkv-fusion auto|split|fused
--qwen3-gate-up-fusion auto|split|fused
```

CLI 的 `fused` 映射为内部 `ForceFused`，名称差异是有意的：

- 对用户表达“我要跑 fused A/B”。
- 对代码表达“这是强制诊断模式，不是生产白名单”。

参数被加入 Qwen3 consumed-args 列表；其他模型不能把这些参数误当作有效配置。

### 4.2 `launch_with_seed`

`openinfer-qwen3/src/lib.rs` 新增 `launch_with_seed`。

原因不是 projection 数学本身，而是 `bench_serving` 为了复用正式 `launch` 的 TP 与 fusion policy，不能丢掉原有 benchmark `--seed`。普通 server 的 `launch` 继续保持历史 seed `42`，benchmark 调用显式 seed 版本。

### 4.3 Dynamo 兼容

`openinfer-dynamo-backend/src/engine.rs` 给新增字段填入默认 options，保证 Dynamo worker 仍保持原 split 行为，没有隐式打开候选路径。

## 5. QKV CUDA split operator

### 5.1 为什么 fused QKV 后还需要一个 kernel

full QKV GEMM 输出是一个连续的 column-major tensor：

```text
[Q rows; K rows; V rows] × tokens
```

后续 QK norm、RoPE、KV append 和 attention 仍需要三个紧凑 buffer：

```text
q: [Q, tokens]
k: [KV, tokens]
v: [KV, tokens]
```

因此候选路径不是单个 GEMM，而是：

```text
full QKV GEMM
  → BF16 split-copy
  → 原 Q/K/V 后处理
```

### 5.2 CUDA kernel 做了什么

文件：`openinfer-kernels/csrc/shared/fused_proj.cu`

`split_qkv_kernel`：

- 遍历 combined tensor 的线性 index。
- 从 `idx / qkv_dim` 得到 token column。
- 从 `idx % qkv_dim` 得到 Q/K/V row。
- 将原始 `__nv_bfloat16` 直接写入对应目标。

它刻意不做：

- BF16 → FP32 → BF16 转换
- head 重排
- QK norm
- RoPE
- scale
- transpose

所以它的 operator contract 是 bitwise copy，不应该引入任何新的数值误差。

### 5.3 checked Rust wrapper

文件：`openinfer-kernels/src/ops/elementwise.rs`

`split_qkv_into` 在 launch 前检查：

- Q、KV、tokens 非零
- K/V hidden dim 相同
- combined dim 等于 `Q + K + V`
- 四个 tensor 的 token count 相同
- Q、KV、tokens 能转换为 i32
- combined element count 不超过 CUDA kernel 的 i32 indexing

launch 后立即检查 `cudaGetLastError` 返回值，避免错误延迟到后面的 attention 或 collective 才暴露。

### 5.4 FFI、导出和 registry

同时修改：

- `openinfer-kernels/src/ffi/shared.rs`
- `openinfer-kernels/src/ops.rs`
- `openinfer-kernels/KERNELS.md`
- `openinfer-core/src/ops/call_spec.rs`

这样 production forward、kernel unit test 和 model trace 使用同一个 checked operator。

## 6. Buffer 设计与显存变化

### 6.1 Decode buffer

文件：`openinfer-qwen3/src/batch_decode_buffers.rs`

原字段被改为：

```text
qkv_out:     Option<HiddenStates>
gate_out:    Option<HiddenStates>
up_out:      Option<HiddenStates>
gate_up_out: Option<HiddenStates>
```

构造规则：

| resolved path | 分配 |
| --- | --- |
| split QKV | 不分配 `qkv_out` |
| fused QKV | 分配 `[Q+2KV, batch]` |
| split gate/up | 分配 `gate_out`、`up_out` |
| fused gate/up | 只分配 `[2I, batch]` 的 `gate_up_out` |

`set_batch_size` 只更新存在的 Option 内 `seq_len`。

### 6.2 Prefill buffer

文件：`openinfer-qwen3/src/prefill.rs`

`PrefillBuffers` 采用相同表示，`set_rows` 同步更新 active buffer 的 logical token rows。

这对 chunked prefill 很重要：物理容量可以保持最大值，但每一步 GEMM 和 kernel 必须看到当前 chunk 的 logical `seq_len`。

### 6.3 显存数字

QKV fused 的 combined scratch 大小：

```text
(Q + 2 × KV) × tokens × 2 bytes
```

| 场景 | TP1 | TP2 每 rank |
| --- | ---: | ---: |
| decode batch 256 | `3.0 MiB` | `1.5 MiB` |
| prefill 10k tokens | `117.2 MiB` | `58.6 MiB` |

gate/up 两种表示的元素数都为：

```text
2 × intermediate × tokens
```

所以 fused gate/up 是替换两个 split buffer，不是额外保留第三份 combined buffer。

### 6.4 默认 split 也有一个资源变化

旧 decode buffer 无论是否使用都分配 `qkv_out`；新实现只在 fused QKV 时分配。

因此默认 `Auto → split` 的计算拓扑没有改变，但资源行为略有改变：

- 少占一份未使用的 decode QKV combined scratch。
- KV budget profile 可能因此得到少量更多 block。

这通常是正向变化，但严格来说不能把默认行为描述为“所有资源数字完全不变”。

## 7. Decode forward 的具体变化

文件：`openinfer-qwen3/src/batch_decode.rs`

### 7.1 QKV baseline

保留原来的三个 row-sliced GEMM：

```text
q = rows [0, Q)
k = rows [Q, Q+KV)
v = rows [Q+KV, Q+2KV)
```

### 7.2 QKV candidate

候选路径：

```text
qkv_out = GEMM(qkv_proj, normed)
q, k, v = split_qkv(qkv_out)
```

完成 split 后才执行原来的 grouped Q/K/V LoRA delta，然后继续：

```text
QK norm → RoPE → KV append → paged decode attention
```

因此 attention、KV layout 和 collective 位置没有变化。

### 7.3 gate/up baseline

保留：

```text
gate = GEMM_rows(gate_up_proj, 0, I)
up   = GEMM_rows(gate_up_proj, I, I)
act  = silu_mul(gate, up)
```

### 7.4 gate/up candidate

候选：

```text
gate_up = GEMM(gate_up_proj, normed)
gate LoRA delta → rows [0, I)
up LoRA delta   → rows [I, 2I)
act = silu_mul_fused(gate_up)
```

随后 down projection、all-reduce 和 residual 顺序保持不变。

### 7.5 理论 launch 数变化

不计 LoRA、attention 和其他算子，每层：

| projection | split | fused candidate | 减少 |
| --- | ---: | ---: | ---: |
| QKV | 3 GEMM | 1 GEMM + 1 split-copy | 1 launch |
| gate/up | 2 GEMM + 1 SwiGLU | 1 GEMM + 1 fused SwiGLU | 1 launch |

两者都启用时，36 层理论上减少 72 次 kernel launch。

这只是机制解释，不是性能结论。更大的 GEMM 可能选到更慢算法，split-copy 也会消耗带宽，所以最终必须看 aggregate 和端到端数据。

### 7.6 一个 TP2、单 token decode 的具体例子

假设当前是 TP2 rank 0，batch size 为 1：

```text
normed hidden: [2560, 1]
```

QKV split baseline：

```text
Q: [2048, 2560] × [2560, 1] → [2048, 1]
K: [ 512, 2560] × [2560, 1] → [ 512, 1]
V: [ 512, 2560] × [2560, 1] → [ 512, 1]
```

QKV candidate：

```text
[3072, 2560] × [2560, 1] → combined [3072, 1]
split-copy:
  rows [0, 2048)       → Q
  rows [2048, 2560)    → K
  rows [2560, 3072)    → V
```

MLP split baseline：

```text
gate: [4864, 2560] × [2560, 1] → [4864, 1]
up:   [4864, 2560] × [2560, 1] → [4864, 1]
```

MLP candidate：

```text
[9728, 2560] × [2560, 1] → [gate 4864 rows; up 4864 rows]
```

rank 1 使用相同 shape、不同权重 shard独立计算。后续 o/down projection 的 TP
all-reduce 没有移动，也没有新增 collective。

候选可能更快，是因为 launch 更少、较大的 `M` 可能让 GPU tile 利用率更高；
也可能更慢，因为 cuBLASLt 对 `M=3072/9728` 选择了不同算法，或 split-copy
抵消了 launch 收益。这正是不能由 TP1 或单个 kernel 结果外推 TP2 的原因。

## 8. Prefill、unified 与 verify

### 8.1 Prefill

文件：`openinfer-qwen3/src/prefill.rs`

`forward_layer_pre_attn`：

- split：原三个 row GEMM。
- fused：full QKV GEMM + checked split-copy。
- 两条路径随后使用同一套 Q/K/V LoRA、QK norm、RoPE 和 paged prefill attention。

`forward_layer_post_attn`：

- split：两个 row GEMM + split SwiGLU。
- fused：full gate/up GEMM + gate/up row-offset LoRA + fused SwiGLU。

### 8.2 Unified mixed step

文件：`openinfer-qwen3/src/unified_forward.rs`

unified path 同时处理 prefill token 和 decode token。它复用 `ProjectionPhase::PrefillUnified`：

- 同一个 `PrefillBuffers` 不携带两套 representation。
- decode rows、prefill rows 和 LoRA token groups 的索引逻辑不变。
- fusion 只改变 dense projection 的分组。

### 8.3 Verify graph

文件：`openinfer-qwen3/src/verify_graph.rs`

DFlash verify 的 fixed `PrefillBuffers` 使用相同 resolved plan。这样：

- capture 前已经决定 topology。
- fixed buffer pointer 在 graph 生命周期内稳定。
- 不会出现正常 prefill fused、verify graph 却仍按 split buffer 捕获的分叉。

## 9. TP 与 cuBLASLt tuning

文件：`openinfer-qwen3/src/executor.rs`

`tune_decode_gemm_algos` 根据 resolved decode topology 选择实际 shape：

### 9.1 QKV

- split：tune `Q` 与 `KV`。
- fused：只 tune `Q + 2KV`。

### 9.2 gate/up

- split：layer samples 同时包含 row offset `0` 和 `I`，tune `I` rows。
- fused：每层 sample 使用 row offset `0`，tune `2I` rows。

### 9.3 为什么 sample 仍按 layer 旋转

36 层权重轮换用于让 tuning 更接近 L2-cold production 行为，避免只反复测一个常驻 cache 的小矩阵而选错算法。

### 9.4 thread-local 约束

模型 profile worker 和长期 serving worker 都会：

- bind 对应 CUDA context。
- 初始化该 worker thread 的 cuBLAS handle。
- 在该 rank 的实际线程上 tune。

TP1 和 TP2 的 local `M` 不同，所以每个 rank 使用自己的 local shape，不用 TP1 结果模拟 TP2。

### 9.5 CUDA Graph 顺序

关键顺序保持：

```text
构造 resolved plan
  → 分配 active buffers
  → worker-thread tune
  → TP ranks 启动期预捕获
  → replay
```

step 内没有：

- 重新解析 fusion policy
- 分配 projection scratch
- tune 新 GEMM shape
- split/fused 热切换

## 10. LoRA 语义

### 10.1 QKV

LoRA adapter 仍按逻辑 projection 存储：

```text
q_proj
k_proj
v_proj
```

即使 base weight 通过一个 combined GEMM 计算，也会先 split 到 q/k/v，再分别添加对应 delta。这样 adapter 不需要知道 combined QKV layout。

### 10.2 gate/up

在 combined `[gate; up]` buffer 中：

```text
gate delta row_offset = 0
up delta row_offset   = I
```

decode 使用现有 fused LoRA delta operator；prefill/unified 使用 range/indexed delta helper。

### 10.3 CUDA Graph

现有 contract 是 LoRA serving 不进入 decode CUDA Graph，因为 adapter pointer 会在请求间变化。本次没有改变这个限制。

### 10.4 Fixture 生成器

`tools/accuracy/dump_qwen3_4b_lora_golden.py` 的 target 从：

```text
q_proj, v_proj
```

扩展为：

```text
q_proj, k_proj, v_proj, gate_proj, up_proj
```

生成器会检查：

- 五个 target 全部被 PEFT 发现。
- 每层每 target 都有 A/B tensor。
- tensor 数量为 `layers × 5 × 2`。

测试读取 fixture 时也检查 metadata 中每个 target 每层都有非零 A/B tensor。

但是 committed safetensors 尚未重生成，所以 gate/up LoRA 的最终证据仍未完成。

## 11. Trace 与报告

### 11.1 Decode DAG

文件：`openinfer-qwen3/src/batch_decode_dag.rs`

新增节点：

- full QKV `gemm`
- `split_qkv`
- full gate/up `gemm`
- fused SwiGLU

同时修正 split SwiGLU 的 call spec：split 与 fused 不再都被记录成 `silu_mul_fused_batch`。

### 11.2 Model report

文件：`openinfer-qwen3/src/bin/qwen3_model_report.rs`

新增：

```text
--qkv-fusion auto|split|fused
--gate-up-fusion auto|split|fused
```

report schema 记录两条 requested topology，默认输出文件名也包含 topology，避免四种实验互相覆盖。

新增 measurement provider：

- split SwiGLU
- fused SwiGLU
- QKV split-copy

限制：

- 当前 model report 是 TP1 trace。
- `NumericPolicy::Tuned` 下，无法忠实复现 startup tuning 的 GEMM 会沿用现有逻辑标为 excluded。
- 因而它能证明 DAG topology 和非 GEMM component，但不能作为 fused GEMM 的最终性能资格。

## 12. Server 与 benchmark

### 12.1 正式 server

真实 server 通过 `Qwen3LaunchOptions` 传入两个独立 control，因此 force A/B 使用的不是测试专用 forward。

### 12.2 `bench_serving`

原 Qwen3 benchmark 构造：

```text
device_ordinals = [0]
```

即使用户传 `--tp-size=2`，Qwen3 也不会真实进入 TP2。

现在改为调用 Qwen3 正式 `launch_with_seed`：

- TP1：使用 `device_ordinal`
- TP2：使用 devices `0..tp_size`
- 同时传入两个 fusion control
- 保留 benchmark sampling seed

### 12.3 Benchmark 产物自描述

`RunInfo` 新增：

- `tp_size`
- `qwen3_projection_fusion`

text 和 JSON 都会记录，例如：

```text
tp_size=2
projection_fusion=qkv=fused,gate_up=split
```

这是 requested mode。真正 resolved 状态仍以模型启动日志为准；当前空白名单下 `auto` 一定解析为 split。

### 12.4 非 Qwen3 防误用

在其他 model type 上显式传非 auto 的 Qwen3 fusion 参数会报错，不会接受一个实际无效的 benchmark 标签。

## 13. 测试改动

### 13.1 Policy 单元测试

`projection_fusion.rs` 覆盖：

- 空白名单下 TP1/TP2 `Auto → split`
- Qwen3-4B TP1/TP2 force 成功
- QKV 与 gate/up 独立解析
- TP>2 拒绝
- 非 4B geometry 拒绝
- Pin/PerToken 拒绝
- SharedSm/GreenCtx 拒绝

### 13.2 QKV split operator

GPU unit test覆盖：

- TP1 local dims，`N=1/8/128`
- TP2 local dims，`N=1/8/128`
- 非 256-thread 整除的 tail shape
- 每个 BF16 element 比较原始 bit pattern

### 13.3 HF golden

`openinfer-qwen3/tests/hf_golden_gate.rs` 支持：

```text
OPENINFER_QWEN3_PROJECTION_FUSION=auto|split|qkv|gate-up|both
```

每个模式通过普通 executor 构造期 options 运行现有：

- sequential bs=1 eager
- deterministic rerun
- batched eager
- prefix-cache replay
- CUDA Graph bucket straddle
- TP2 eager/graph（有两张 GPU 时）

没有修改：

- regret `≤ 0.20 nat`
- mean `≤ 0.06`
- p99 `≤ 0.20`
- HF golden 数据

### 13.4 LoRA golden

LoRA gate使用同一个 fusion 环境变量，并继续覆盖：

- base-only
- LoRA-only
- mixed base/LoRA batch
- TP1/TP2

但在五 projection fixture 重生成前，这部分只能证明旧 q/v adapter 在新拓扑下的兼容性，不能证明 K/gate/up 全部真正参与。

### 13.5 LaunchOptions 调用点

DFlash 测试、TP concurrent 测试和 Dynamo backend 均补上默认 fusion options，保持原行为。

## 14. 正确性不变量核对

| 不变量 | 实现方式 | 当前证据状态 |
| --- | --- | --- |
| 权重顺序 `[Q;K;V]` | split kernel按固定 row range copy | 代码 + 待跑 GPU bitwise test |
| 权重顺序 `[gate;up]` | gate offset 0，up offset I | 代码 + 五 target fixture 待生成 |
| QKV split 不产生舍入 | 直接复制 `__nv_bfloat16` | 代码 + 待跑 GPU test |
| SwiGLU BF16 边界不变 | fused kernel物化 BF16 SiLU 后再乘 up | 既有 test + 待跑 |
| TP 不新增 collective | 只替换 all-reduce 前的 local projection | 静态核对，待 TP gate |
| Graph pointer 稳定 | 构造期分配，capture 前 tune | 静态核对，待 capture/replay |
| LoRA 保持逻辑 projection | Q/K/V split 后写；gate/up row offset 写 | 静态核对，fixture/gate 待完成 |
| unsupported force 不 fallback | `validate_force_supported` 返回错误 | 单测已写，未执行 |
| 默认不启用未验证优化 | 空 `auto_whitelisted` | 代码事实 |

## 15. 本地验证结果

已实际通过：

- `cargo fmt --all --check`
- `git diff --check`
- `cargo metadata --locked --no-deps --format-version 1`
- LoRA Python 生成脚本 AST parse

未通过到目标阶段：

### 15.1 Qwen3 release check

尝试：

```text
OPENINFER_CUDA_SM=120 cargo check --release -p openinfer-qwen3 --lib
```

macOS host 在 `rdma-mummy-sys` build script 因缺少 Linux headers：

```text
endian.h
linux/types.h
```

失败。它没有进入 openinfer-qwen3 Rust type-check。

### 15.2 Core release check

尝试：

```text
OPENINFER_CUDA_SM=120 cargo check --release -p openinfer-core --lib
```

进入 `openinfer-kernels` build script 后因本机无 nvcc 失败，也没有完成 Rust crate type-check。

### 15.3 外部资源

仓库有会创建计费 GPU 实例的 provisioning 脚本，但当前没有用户授权云资源开销，因此未运行。

## 16. 实现与原计划的偏差

### 16.1 Buffer 使用多个 Option，而不是 enum

计划建议：

```text
QkvProjectionScratch::Split | Fused
MlpProjectionScratch::Split | Fused
```

实际实现使用：

```text
qkv_out: Option<_>
gate_out: Option<_>
up_out: Option<_>
gate_up_out: Option<_>
```

构造器当前保证合法组合，forward 用 `.expect(...)` 检查 resolved plan 与 buffer 一致，因此正常路径功能成立。

但类型系统仍允许这些非法状态：

- fused gate/up 时 `gate_up_out=None`
- split gate/up 时 gate 或 up 只分配一个
- split 与 fused scratch 同时存在

建议合入前评审是否改成 enum。enum 能把不变量从运行时约定提升为编译期结构，长期维护成本更低。

### 16.2 `Auto` 没有结构化 fallback reason

计划要求 Auto 未命中时记录一次原因。

当前日志打印：

```text
options=...
resolved=...
```

能看见最终 false，但不能区分：

- 白名单为空
- geometry 不支持
- TP 不支持
- numeric policy 不支持
- overlap 不支持

建议在真正填充白名单前增加 resolution reason enum，并只在 rank 0 启动时打印一次。

### 16.3 Projection 数值诊断 runner 已实现，数据待生成

`qwen3_projection_report` 现在使用真实 rank-local Qwen3-4B 权重和相同
patterned BF16 输入，按 layer、shape、TP rank 输出：

- QKV 三 GEMM vs fused GEMM + split-copy；
- gate/up 两 GEMM vs fused GEMM 的 raw projection delta；
- projection + SwiGLU 完整 aggregate；
- BF16 ULP histogram、exact ratio、mean/p50/p99/max abs delta；
- CUDA event p50/p99/avg、launch count、scratch bytes；
- `N<=32` 实际 selected cuBLASLt algo/tile/stage/split-K/reduction/swizzle
  metadata；large-N 明确标记为 `cublas_gemm_ex`。

小 N tuning 使用与 executor 相同的 all-layer cold-weight rotation。TP2 必须
分别运行 rank 0/rank 1，不能以 local shape 模拟真实 shard weight。

尚缺的是 Linux CUDA 上生成的 JSON 数据，而不是 reporter 代码。

### 16.4 Prefill fused GEMM 没有独立显式 startup tuning 证据

当前新增的显式 topology-aware tuning 位于 decode bucket。prefill 大 N 继续依赖现有 GEMM 路由/缓存行为。

在决定 prefill/unified 白名单前，需要确认：

- 实际 large-N backend 和算法。
- 是否需要与 decode 分开的 prefill tuning。
- 10k prompt 的 scratch 是否压缩 KV admission。

### 16.5 Benchmark 记录 requested，不是完整 resolved plan

force/split 模式下 requested 等于实际。

`auto` 未来可能按 phase/TP 部分命中，单个字符串不足以描述四个 resolved bool。正式启用 Auto entry 前，benchmark JSON 应记录完整 resolved plan。

## 17. 合入前建议门禁

### P0：必须完成

1. Linux CUDA release type-check。
2. QKV split GPU bitwise test。
3. 四 fusion mode × TP1/TP2 HF gate。
4. eager + CUDA Graph bucket-straddle。
5. 重生成五 projection LoRA fixture并跑 TP1/TP2。
6. unified mixed-step 与 verify graph gate。
7. TP1/rank0 与 TP2/rank0+rank1 projection report 无 NaN/Inf，kernel
   aggregate 与 E2E 同方向。
8. 32-cell benchmark，每个关键 cell两次对称交错复测（64 runs）。
9. 保持 `Auto` 白名单为空，直到上述证据完成。

### P1：建议合入前完成

1. 用 enum 替换多个 Option 表示 projection scratch。
2. 增加 Auto resolution reason。
3. benchmark 记录完整 resolved plan，而非只有 requested mode。

## 18. 代码位置索引

| 主题 | 文件 / 符号 |
| --- | --- |
| fusion policy | `openinfer-qwen3/src/projection_fusion.rs` |
| public launch options | `openinfer-qwen3/src/lib.rs::Qwen3LaunchOptions` |
| seed-preserving launch | `openinfer-qwen3/src/lib.rs::launch_with_seed` |
| executor constructor | `openinfer-qwen3/src/executor.rs::from_runtime_with_projection_fusion` |
| model-load resolution | `openinfer-qwen3/src/weights/load.rs::from_safetensors_with_runtime` |
| resolved model state | `openinfer-qwen3/src/weights.rs::Qwen3Model::projection_fusion` |
| QKV CUDA split | `openinfer-kernels/csrc/shared/fused_proj.cu::split_qkv_kernel` |
| checked QKV wrapper | `openinfer-kernels/src/ops/elementwise.rs::split_qkv_into` |
| decode scratch | `openinfer-qwen3/src/batch_decode_buffers.rs::BatchDecodeBuffers` |
| decode forward | `openinfer-qwen3/src/batch_decode.rs::batch_decode_layer` |
| decode DAG/trace | `openinfer-qwen3/src/batch_decode_dag.rs` |
| decode tuning | `openinfer-qwen3/src/executor.rs::tune_decode_gemm_algos` |
| prefill scratch/forward | `openinfer-qwen3/src/prefill.rs::PrefillBuffers` |
| unified forward | `openinfer-qwen3/src/unified_forward.rs` |
| verify fixed buffers | `openinfer-qwen3/src/verify_graph.rs` |
| server CLI | `openinfer-server/src/config.rs::CliProjectionFusion` |
| server wiring | `openinfer-server/src/main.rs::load_engine` |
| benchmark CLI | `openinfer-server/src/bin/bench_serving/cli.rs` |
| benchmark launch | `openinfer-server/src/bin/bench_serving/main.rs` |
| benchmark metadata | `openinfer-server/src/bin/bench_serving/{report,runners,render}.rs` |
| model operator report | `openinfer-qwen3/src/bin/qwen3_model_report.rs` |
| real-weight projection report | `openinfer-qwen3/src/{projection_report.rs,bin/qwen3_projection_report.rs}` |
| cuBLASLt selected algorithm query | `openinfer-kernels/{csrc/shared/linear.cu,src/ops/linear.rs}` |
| validation orchestration + summary | `tools/validation/qwen3_fused_projection_suite.py` |
| validation summary unit tests | `tools/validation/test_qwen3_fused_projection_suite.py` |
| HF matrix selector | `openinfer-qwen3/tests/hf_golden_gate.rs::projection_fusion_options` |
| LoRA matrix selector | `openinfer-qwen3/tests/lora_golden_gate.rs::projection_fusion_options` |
| LoRA fixture generator | `tools/accuracy/dump_qwen3_4b_lora_golden.py` |

## Execution Log

### Step 1 — 重建配置与执行链

- 从 server/bench CLI 追踪到 `Qwen3Model` 内 resolved state。
- 确认策略在 buffer 分配和 CUDA Graph capture 前固定。
- 结果：完成。

### Step 2 — 核对四条 forward 路径

- 核对 decode、prefill、unified、verify 的 QKV 与 gate/up 分支。
- 核对 Q/K/V 与 gate/up LoRA 写入顺序和 row offset。
- 结果：完成静态审查；GPU 执行仍待验证。

### Step 3 — 核对 TP、trace、测试和 benchmark

- 确认 decode tuning 使用 local shape。
- 确认 benchmark Qwen3 TP 不再固定单卡。
- 确认 HF/LoRA gate 使用生产 executor 构造路径。
- 结果：入口完成；实际 TP/GPU 结果尚无。

### Step 4 — 识别计划偏差

- 发现 buffer 实现使用 Option 而非 enum。
- 发现 Auto 没有 fallback reason。
- projection numerical runner、selected cuBLASLt metadata query 和统一验证
  汇总器已经补齐；完整 prefill tuning 证据和 resolved benchmark metadata
  仍未完成。
- 结果：已记录为合入前评审项。

### Step 5 — 补齐证据产出与 fail-closed 汇总

- 新增真实权重 `qwen3_projection_report`，覆盖 TP rank、全 layer、
  decode/prefill shape、数值 delta、ULP、aggregate latency、scratch 和
  selected cuBLASLt metadata。
- HF/LoRA gate 新增 `OPENINFER_GOLDEN_TP_SIZE=1|2`，验证套件的每个
  `(mode, TP)` 都是独立进程；TP2 不足两卡直接失败，指定 TP 的 HF gate
  要求 graph group 已编译，不能以 skip 冒充通过。
- LoRA fixture loader 与 suite preflight 都要求 q/k/v/gate/up 五 target；
  当前 q/v-only committed fixture 按预期 fail closed。
- 新增统一 suite 和 summary unit tests；完整 dry-run 展开 103 个命令，
  保存 raw/log/manifest/summary/decision-table/report 六类证据。Qwen3 unit
  门禁只选择 `openinfer-kernels`、`openinfer-qwen3`、`openinfer-server`，
  避免 `--workspace` 将 GLM/Kimi 的 `moe`/DeepEP 2.30.4 构建依赖带入
  Qwen3 专项验证。
- AutoDL scoped unit gate 暴露 `split_qkv_into` 只从 kernels crate 导出、
  未进入 `openinfer_core::ops` facade 的编译遗漏；补充统一 re-export 后，
  prefill、unified forward 与 topology report 共享同一 operator 入口。
- QKV copy gate 的任意 BF16 payload 包含 NaN；原 slice `PartialEq` 会把
  bit-identical NaN 判为不等。Q/K/V 断言已改为逐元素 `to_bits()`，保留
  全 payload 覆盖并增加 equal-NaN-payload 回归测试。
- `qwen3_projection_report` 的 CLI-only `clap` 现由独立
  `projection-report` feature 管理，binary 声明 required feature，suite
  显式启用；默认 HF/LoRA/lib 构建不再误编译缺少 optional dependency 的
  report target。
- 结果：产出代码完成；本机无 CUDA/nvcc，真实报告仍待 Linux TP2 主机。

## Debrief

- **Outcome**:
  - 形成了从背景、策略、kernel、buffer、forward、TP、LoRA、Graph、trace、benchmark 到验证状态的完整实现报告。
  - 明确区分候选实现与生产资格，没有把未执行的 GPU gate 标记为通过。
- **Pitfalls encountered**:
  - diff 规模较大，仅按文件罗列会掩盖配置、buffer 和 graph 之间的依赖，因此报告改按请求执行链组织。
  - “默认仍 split”不等于资源行为绝对不变；新实现会省掉旧 baseline 未使用的 decode `qkv_out`。
- **Lessons learned**:
  - fused projection 的核心风险不是数学公式，而是 GEMM 分组改变后的 BF16 reduction path。
  - 对 CUDA Graph 路径，policy、buffer variant 和 tuning shape 必须是同一个构造期事实。
  - 实验配置必须进入真实 server/benchmark 路径，否则测到的结果不能作为生产证据。
- **Follow-ups**:
  - 在获授权的 Linux CUDA TP2 环境完成 P0 门禁。
  - 决定是否在合入前完成 enum buffer refactor 和 structured resolution reason。
  - 根据实测数据决定白名单；没有明确收益则继续 split。
