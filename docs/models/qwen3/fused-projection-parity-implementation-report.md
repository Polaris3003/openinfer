# Qwen3 fused projection 候选实现报告（Issue #746）

> **TL;DR:** Qwen3 已具备构造期固定、彼此独立的 QKV 与 gate/up 融合候选及 fail-closed schema-v3 套件；current HEAD `a8ef928` 在 2×RTX 4090（SM89）完成 104/104 命令，数值、LoRA、TP、graph/topology 与真实 HTTP 全绿，但八个独立性能决策全部 `KEEP_SPLIT`。默认 `Auto` 继续为空；本轮五 target fixture 尚未提交到 tracked test data。
>
> **Last touched:** 2026-08

Issue: [pegainfer-project/pegainfer#746](https://github.com/pegainfer-project/pegainfer/issues/746)

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
Qwen3ProjectionFusionPlan
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

当前不能声称 Issue #746 已经完成跨硬件生产放行，原因是：

- `Auto` 白名单为空，没有任何 `(projection, phase, TP)` 默认启用。
- current-head SM89 原始 `raw/log/manifest/summary/decision-table/report` 已取回并
  校验完整，但八个独立项都没有达到既定性能规则。
- runner 隔离生成并实测了五 projection fixture，但 committed fixture 仍是旧
  q/v-only 文件，新 checkout 会在 fixture preflight 失败。
- TP1 decode both 有 `+2.35%` 交互收益，但本轮预注册规则只允许独立归因；
  不能看完结果后修改标准直接放行。
- `ProjectionFusionEnvironment` 不包含 GPU SM、CUDA/cuBLAS 或 selected algo，
  且 SM86 与 SM89 的最优项不同，不能建立跨硬件全局白名单。

正确的状态描述是：

> SM89 current-head 候选路径与证据均完整，但没有独立 fused 行获得生产资格；
> 默认路径保持 split。TP1 decode both 只进入新实验候选，不能成为本轮事后结论。

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

文件：`pegainfer-qwen3/src/projection_fusion.rs`

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
Qwen3 ModelLine / Qwen3Cli
  → Qwen3LaunchOptions
  → launch
  → scheduler::start_qwen3
  → Qwen3Executor::from_runtime_with_projection_fusion
  → ModelRuntimeConfig
  → Qwen3Model::from_safetensors_with_runtime
  → Qwen3ProjectionFusionPlan::resolve
  → Qwen3Model.projection_fusion
```

涉及文件：

- `pegainfer-qwen3/src/model_line.rs`
- `pegainfer-qwen3/src/lib.rs`
- `pegainfer-qwen3/src/scheduler.rs`
- `pegainfer-qwen3/src/executor.rs`
- `pegainfer-qwen3/src/weights/load.rs`
- `pegainfer-qwen3/src/weights.rs`

### 4.1 服务端 CLI

新增：

```text
--qwen3-qkv-fusion auto|split|fused
--qwen3-gate-up-fusion auto|split|fused
```

CLI 的 `fused` 映射为内部 `ForceFused`，名称差异是有意的：

- 对用户表达“我要跑 fused A/B”。
- 对代码表达“这是强制诊断模式，不是生产白名单”。

参数由 Qwen3 `ModelLine` 独占；其他模型不会注册这些参数。上游已删除旧 Dynamo
backend 和 in-process benchmark，因此不再为已删除入口保留 seed 或默认字段 shim。

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

文件：`pegainfer-kernels/csrc/shared/fused_proj.cu`

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

文件：`pegainfer-kernels/src/ops/elementwise.rs`

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

- `pegainfer-kernels/src/ffi/shared.rs`
- `pegainfer-kernels/src/ops.rs`
- `pegainfer-kernels/KERNELS.md`
- `pegainfer-core/src/ops/call_spec.rs`

这样 production forward、kernel unit test 和 model trace 使用同一个 checked operator。

## 6. Buffer 设计与显存变化

### 6.1 Decode buffer

文件：`pegainfer-qwen3/src/batch_decode_buffers.rs`

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

文件：`pegainfer-qwen3/src/prefill.rs`

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

文件：`pegainfer-qwen3/src/batch_decode.rs`

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

文件：`pegainfer-qwen3/src/prefill.rs`

`forward_layer_pre_attn`：

- split：原三个 row GEMM。
- fused：full QKV GEMM + checked split-copy。
- 两条路径随后使用同一套 Q/K/V LoRA、QK norm、RoPE 和 paged prefill attention。

`forward_layer_post_attn`：

- split：两个 row GEMM + split SwiGLU。
- fused：full gate/up GEMM + gate/up row-offset LoRA + fused SwiGLU。

### 8.2 Unified mixed step

文件：`pegainfer-qwen3/src/unified_forward.rs`

unified path 同时处理 prefill token 和 decode token。它复用 `ProjectionPhase::PrefillUnified`：

- 同一个 `PrefillBuffers` 不携带两套 representation。
- decode rows、prefill rows 和 LoRA token groups 的索引逻辑不变。
- fusion 只改变 dense projection 的分组。

### 8.3 Verify graph

文件：`pegainfer-qwen3/src/verify_graph.rs`

DFlash verify 的 fixed `PrefillBuffers` 使用相同 resolved plan。这样：

- capture 前已经决定 topology。
- fixed buffer pointer 在 graph 生命周期内稳定。
- 不会出现正常 prefill fused、verify graph 却仍按 split buffer 捕获的分叉。

## 9. TP 与 cuBLASLt tuning

文件：`pegainfer-qwen3/src/executor.rs`

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

文件：`pegainfer-qwen3/src/batch_decode_dag.rs`

新增节点：

- full QKV `gemm`
- `split_qkv`
- full gate/up `gemm`
- fused SwiGLU

同时修正 split SwiGLU 的 call spec：split 与 fused 不再都被记录成 `silu_mul_fused_batch`。

### 11.2 Model report

文件：`pegainfer-qwen3/src/bin/qwen3_model_report.rs`

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

### 12.2 真实 HTTP benchmark

最新主干已删除绕过 serving stack 的 `bench_serving`，HTTP benchmarking 是唯一保留的服务性能入口。验证 suite 的每个 cell 现在：

1. 用 Qwen3 正式 server 和指定 TP/fusion 参数启动独立进程；
2. 等待 `/v1/models` 就绪；
3. 运行 `scripts/bench_http_serving.py`，采集 TTFT/TPOT/吞吐、失败率和 GPU 状态；
4. 将 requested/resolved plan、server command 和 server log 写入 cell 产物；
5. 终止该 server 后再进入下一个 matched A/B cell。

这避免复活上游明确删除的旧 binary，也确保数字包含 frontend、bridge、scheduler 和真实 executor。
每个 cell 强制 `100%` server-trace coverage；summarizer 还要求 matched A/B 的
实际 input tokens/request 相同，避免把 prompt 生成或 tokenization 漂移误算成收益。

### 12.3 CLI 所有权

两个 fusion flag 属于 `pegainfer-qwen3/src/model_line.rs::Qwen3Cli`，由 Qwen3 自己解析并组装 `Qwen3LaunchOptions`。其他模型不会注册这些参数，因此不存在“接受标签但未执行”的静默路径。

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

`pegainfer-qwen3/tests/hf_golden_gate.rs` 支持：

```text
PEGAINFER_QWEN3_PROJECTION_FUSION=auto|split|qkv|gate-up|both
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

本轮 runner 已隔离生成五 projection fixture，并在 TP1/TP2 × 四种 fusion mode
中全部通过；K/gate/up 的 row offset 已进入真实 LoRA 数值门禁。剩余问题是将该
fixture 提交到 tracked test data，使 fresh checkout 不依赖 override 路径。

### 13.5 LaunchOptions 调用点

DFlash 测试、TP concurrent 测试和 Dynamo backend 均补上默认 fusion options，保持原行为。

## 14. 正确性不变量核对

| 不变量 | 实现方式 | 当前证据状态 |
| --- | --- | --- |
| 权重顺序 `[Q;K;V]` | split kernel按固定 row range copy | current-head SM89 GPU bitwise test 通过 |
| 权重顺序 `[gate;up]` | gate offset 0，up offset I | 五 target LoRA gate 8/8 通过 |
| QKV split 不产生舍入 | 直接复制 `__nv_bfloat16` | current-head GPU test 通过 |
| SwiGLU BF16 边界不变 | fused kernel物化 BF16 SiLU 后再乘 up | current-head GPU test 通过 |
| TP 不新增 collective | 只替换 all-reduce 前的 local projection | TP2 HF/LoRA/projection/HTTP 全部通过 |
| Graph pointer 稳定 | 构造期分配，capture 前 tune | topology 16/16 + HF graph 通过 |
| LoRA 保持逻辑 projection | Q/K/V split 后写；gate/up row offset 写 | 五 target fixture 实跑通过，tracked 文件待提交 |
| unsupported force 不 fallback | `validate_force_supported` 返回错误 | unit + suite resolved-plan 审计通过 |
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
PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-qwen3 --lib
```

macOS host 在 `rdma-mummy-sys` build script 因缺少 Linux headers：

```text
endian.h
linux/types.h
```

失败。它没有进入 pegainfer-qwen3 Rust type-check。

### 15.2 Core release check

尝试：

```text
PEGAINFER_CUDA_SM=120 cargo check --release -p pegainfer-core --lib
```

进入 `pegainfer-kernels` build script 后因本机无 nvcc 失败，也没有完成 Rust crate type-check。

### 15.3 外部资源

仓库有会创建计费 GPU 实例的 provisioning 脚本，但当前没有用户授权云资源开销，因此未运行。

### 15.4 SM86 执行主机结果（Issue 评论证据）

2026-08-03 的 Issue 评论记录双卡 SM86、CUDA 12.6、Qwen3-4B、Tuned、
overlap off 的完整运行：correctness `20/20`、projection `3/3`、topology
`16/16`、E2E `64/64`，command error 为 0。按 suite 决策规则：

| projection | phase | TP | E2E 平均改善 | 结论 |
| --- | --- | ---: | ---: | --- |
| QKV | prefill/unified | 1 | +2.40% | KEEP_SPLIT（低于 3% 且 kernel 不同向） |
| gate/up | prefill/unified | 1 | +1.30% | KEEP_SPLIT |
| QKV | decode | 1 | +2.32% | ENABLE（四次同向且 kernel 同向） |
| gate/up | decode | 1 | +0.61% | KEEP_SPLIT |
| QKV | prefill/unified | 2 | -0.12% | KEEP_SPLIT |
| gate/up | prefill/unified | 2 | -2.13% | KEEP_SPLIT |
| QKV | decode | 2 | +11.69% | KEEP_SPLIT（复测方向严重不一致） |
| gate/up | decode | 2 | -2.96% | KEEP_SPLIT |

这些数字可用于方向判断，但当前工作区没有原始 suite 目录，不能独立重建每个
cell、GPU clock/power/memory 或 selected-algo 细节。

### 15.5 2026-08-12 本地续作验证

- 原分支 Python validation 单测：`7/7` 通过；最新主干 HTTP 迁移后为 `8/8`。
- `cargo fmt --all --check`、`cargo metadata --locked --no-deps`、
  `py_compile`、`git diff --check`：通过。
- fixture preflight：按预期失败，缺 `k_proj/gate_proj/up_proj`。
- `PEGAINFER_CUDA_SM=86 cargo test --release -p pegainfer-qwen3 --lib
  projection_fusion --no-default-features`：macOS 先后被 Linux-only
  `rdma-mummy-sys` headers 与无 nvcc 阻塞，未进入目标 Rust type-check。

### 15.6 2026-08-13 SM89 current-head 全量验证

`a8ef928` 在 2×RTX 4090 上完成 schema-v3 runner：

- 104/104 command passed；21/21 correctness、3/3 projection、16/16 topology、
  64/64 HTTP benchmark 全部完整。
- HF/LoRA 每个 integration gate 都实际运行 1 个测试，无 filtered-out 冒充；
  五 target fixture SHA256 为 `f8b76cb4...`。
- 64 个 HTTP report 均为 failed/timeouts=0、trace/token timing coverage=100%，
  requested/resolved fusion plan 完全一致。
- 八个独立项全部 `KEEP_SPLIT`；完整表和 both 交互分析见
  `fused-projection-release-runbook.md`。

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

### 16.2 `Auto` 结构化 fallback reason（已补齐）

`Qwen3ProjectionFusionPlan` 的四个 phase/projection decision 现在分别记录
`fused` 与结构化 reason，可区分：

- 白名单为空
- geometry 不支持
- TP 不支持
- numeric policy 不支持
- overlap 不支持

另外还区分 explicit split、forced fused 和未来的 Auto whitelist hit。rank 0
启动日志打印完整 plan，benchmark JSON 也保存同一结构。

### 16.3 Projection 数值数据已在 SM89 current HEAD 重跑

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

本轮 schema-v3 artifact 已包含 TP1 rank0、TP2 rank0/rank1 三份 current-head raw
report，projection gate 3/3 通过。SM86 旧报告只保留为历史对照，不再承担当前
代码的证明责任。

### 16.4 Prefill fused GEMM 不进入白名单

当前新增的显式 topology-aware tuning 位于 decode bucket。prefill 大 N 继续依赖现有 GEMM 路由/缓存行为。

SM89 projection report 与 10k prompt HTTP 数据已给出一致的否定结论：TP1/TP2
prefill QKV 不同向或回退，gate/up E2E 也没有达到 3% 且方向要求不满足。因此当前
不继续为 prefill 增加 startup tuning 或扩大实现面；只有新的 kernel/algorithm
方案出现时，才重新建立 prefill 资格实验。

### 16.5 HTTP cell 同时记录 requested 与完整 resolved plan（已补齐）

每个强制 A/B cell 在 JSON 中记录 TP、QKV/gate-up requested mode 和四项
resolved decision。`split` 对应 `explicit_split`，`fused` 只有在 production
resolver 接受后 server 才能就绪，对应 `forced_fused`；不支持的组合在 readiness
之前失败。summarizer 要求该结构与实验 mode 精确一致，缺失或漂移直接使 cell
失败。Auto 资格仍以模型启动日志和 resolver 单测为准，不能由强制 A/B 外推。

## 17. 合入前建议门禁

### P0：必须完成

1. 将本轮已验证的五 projection LoRA fixture 提交到
   `test_data/qwen3-4b-lora-golden.safetensors`。
2. 在不设置 `PEGAINFER_LORA_GOLDEN_PATH` 的 fresh checkout 跑 LoRA gate，证明
   默认 test data 自包含。
3. 保持 `Auto` 白名单为空；本轮八个独立项全部 `KEEP_SPLIT`。

Linux CUDA type-check、QKV bitwise、四 mode × TP1/TP2 HF/LoRA、eager/graph、
unified/verify topology、三 rank projection report 和 64-run HTTP 矩阵均已完成。

### P1：建议合入前完成

1. 用 enum 替换多个 Option 表示 projection scratch：本轮评估后不做；字段只在
   构造器赋值，未发现非法状态入口，缺 Linux type-check 时扩大纯类型重排得不偿失。
2. 增加 Auto resolution reason：完成。
3. benchmark 记录完整 resolved plan：完成。
4. TP1 decode both 若进入生产候选，先建立组合级预注册门禁并增加硬件适用 key；
   不在本轮结果上事后放行。

## 18. 代码位置索引

| 主题 | 文件 / 符号 |
| --- | --- |
| fusion policy | `pegainfer-qwen3/src/projection_fusion.rs` |
| public launch options | `pegainfer-qwen3/src/lib.rs::Qwen3LaunchOptions` |
| executor constructor | `pegainfer-qwen3/src/executor.rs::from_runtime_with_projection_fusion` |
| model-load resolution | `pegainfer-qwen3/src/weights/load.rs::from_safetensors_with_runtime` |
| resolved model state | `pegainfer-qwen3/src/weights.rs::Qwen3Model::projection_fusion` |
| QKV CUDA split | `pegainfer-kernels/csrc/shared/fused_proj.cu::split_qkv_kernel` |
| checked QKV wrapper | `pegainfer-kernels/src/ops/elementwise.rs::split_qkv_into` |
| decode scratch | `pegainfer-qwen3/src/batch_decode_buffers.rs::BatchDecodeBuffers` |
| decode forward | `pegainfer-qwen3/src/batch_decode.rs::batch_decode_layer` |
| decode DAG/trace | `pegainfer-qwen3/src/batch_decode_dag.rs` |
| decode tuning | `pegainfer-qwen3/src/executor.rs::tune_decode_gemm_algos` |
| prefill scratch/forward | `pegainfer-qwen3/src/prefill.rs::PrefillBuffers` |
| unified forward | `pegainfer-qwen3/src/unified_forward.rs` |
| verify fixed buffers | `pegainfer-qwen3/src/verify_graph.rs` |
| server CLI/wiring | `pegainfer-qwen3/src/model_line.rs::{Qwen3Cli,Qwen3Line::launch}` |
| HTTP benchmark client | `scripts/bench_http_serving.py` |
| benchmark cell lifecycle/metadata | `tools/validation/qwen3_fused_projection_suite.py::run_http_benchmark_cell` |
| model operator report | `pegainfer-qwen3/src/bin/qwen3_model_report.rs` |
| real-weight projection report | `pegainfer-qwen3/src/{projection_report.rs,bin/qwen3_projection_report.rs}` |
| cuBLASLt selected algorithm query | `pegainfer-kernels/{csrc/shared/linear.cu,src/ops/linear.rs}` |
| validation orchestration + summary | `tools/validation/qwen3_fused_projection_suite.py` |
| validation summary unit tests | `tools/validation/test_qwen3_fused_projection_suite.py` |
| HF matrix selector | `pegainfer-qwen3/tests/hf_golden_gate.rs::projection_fusion_options` |
| LoRA matrix selector | `pegainfer-qwen3/tests/lora_golden_gate.rs::projection_fusion_options` |
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
- HF/LoRA gate 新增 `PEGAINFER_GOLDEN_TP_SIZE=1|2`，验证套件的每个
  `(mode, TP)` 都是独立进程；TP2 不足两卡直接失败，指定 TP 的 HF gate
  要求 graph group 已编译，不能以 skip 冒充通过。
- LoRA fixture loader 与 suite preflight 都要求 q/k/v/gate/up 五 target；
  当前 q/v-only committed fixture 按预期 fail closed。
- 新增统一 suite 和 summary unit tests；完整 dry-run 展开 103 个命令，
  保存 raw/log/manifest/summary/decision-table/report 六类证据。Qwen3 unit
  门禁只选择 `pegainfer-kernels`、`pegainfer-qwen3`、`pegainfer-server`，
  避免 `--workspace` 将 GLM/Kimi 的 `moe`/DeepEP 2.30.4 构建依赖带入
  Qwen3 专项验证。
- AutoDL scoped unit gate 暴露 `split_qkv_into` 只从 kernels crate 导出、
  未进入 `pegainfer_core::ops` facade 的编译遗漏；补充统一 re-export 后，
  prefill、unified forward 与 topology report 共享同一 operator 入口。
- QKV copy gate 的任意 BF16 payload 包含 NaN；原 slice `PartialEq` 会把
  bit-identical NaN 判为不等。Q/K/V 断言已改为逐元素 `to_bits()`，保留
  全 payload 覆盖并增加 equal-NaN-payload 回归测试。
- `qwen3_projection_report` 的 CLI-only `clap` 现由独立
  `projection-report` feature 管理，binary 声明 required feature，suite
  显式启用；默认 HF/LoRA/lib 构建不再误编译缺少 optional dependency 的
  report target。
- decode topology trace 不再按 `batch × kv_len` 分配 request-local KV；
  suite 显式传 `--shared-kv-pages`，让 synthetic rows 共享一张目标长度的
  有效页表，使 bs64/kv2048 保持真实 operator shape 而无需 128 个物理
  blocks。report config 自描述该模式；默认 standalone report、正确性和
  E2E 性能路径仍使用独立 KV。
- 结果：产出代码完成；本机无 CUDA/nvcc，真实报告仍待 Linux TP2 主机。

### Step 6 — 对齐 SM86 实测并闭环 resolved metadata

- Issue 评论记录双卡 SM86/CUDA 12.6 完整 suite：correctness `20/20`、
  projection `3/3`、topology `16/16`、E2E `64/64`，无 benchmark command error。
- 按 suite 的 2% decode / 3% prefill、四次方向一致、throughput 与 kernel
  同向规则重建结论：只有 TP1 decode QKV 为 `ENABLE`；其他七项均为
  `KEEP_SPLIT`。TP2 QKV decode 存在 `+85.02%/-7.03%` throughput 波动，不能
  用均值放行。
- 将 resolved topology 提升为可序列化 `Qwen3ProjectionFusionPlan`；每个
  decision 带 qualification reason，启动日志和 benchmark JSON 不再只有
  requested mode。
- validation summarizer 强制校验 resolved metadata；原分支 Python 单测 `7/7`、
  format、metadata、语法和 diff 门禁通过。
- suite artifact schema 升至 v2；旧 v1 产物会明确拒绝汇总，避免在缺少
  resolved topology 时复用旧性能结论。
- committed LoRA fixture preflight 仍按预期失败，明确缺少
  `k_proj/gate_proj/up_proj`；这证明实跑 fixture 尚未回填分支。
- scratch enum 经赋值点核对后不改：没有新性能/正确性收益，且会扩大当前无法在
  macOS type-check 的 Rust 改动面。
- 结果：评审元数据闭环完成；Auto 继续空白名单，原始 suite 目录和五投影 fixture
  仍是 PR 自包含证据的缺口。

### Step 7 — 迁移到最新 PegaInfer frontend 边界

- 从远端 `main@489bd55` 建立独立迁移分支，保留原分支和用户工作区不变。
- 核心 projection policy、kernel、buffer、decode/prefill/unified、LoRA 与 TP gate
  均迁入 `pegainfer-*` crate；唯一 facade 偏移是补回 `split_qkv_into` re-export。
- 上游已删除 central server config、Dynamo backend 与 `bench_serving`；不恢复这些
  旧边界。fusion CLI 进入 Qwen3 `ModelLine`，性能 suite 改为真实 HTTP A/B。
- artifact schema 升至 v3，强制拒绝旧主干的 v1/v2 benchmark 产物。
- 本地通过 format、metadata、Python `8/8`、py_compile 和 diff gate；Rust GPU
  编译/执行仍需 Linux CUDA 主机。

### Step 8 — SM89 current-head 证据闭环

- `a8ef928` 在 2×RTX 4090/SM89 完成 104-command schema-v3 runner；所有命令、
  correctness、projection、topology 与 64 个 HTTP cell 全部通过。
- 审计确认测试没有被过滤、fixture 确为五 target、所有 HTTP cell 的 fused
  requested/resolved plan 一致，原始输出和 manifest 引用完整。
- 八个独立资格项全部 `KEEP_SPLIT`；TP1 decode QKV/gate-up 虽然分别稳定改善
  `1.38%/1.12%`，但低于 `2%` 预注册门槛。
- TP1 decode both 平均 `+2.35%`，保留为新的组合候选；不用于事后修改本轮独立
  决策规则。SM86/SM89 的性能资格差异也否定了无硬件 key 的全局 Auto。
- 本轮有效 fixture 只存在于证据归档；tracked q/v-only fixture 是 PR 剩余的唯一
  correctness 自包含缺口。

## Debrief

- **Outcome**:
  - 形成了从背景、策略、kernel、buffer、forward、TP、LoRA、Graph、trace、benchmark 到验证状态的完整实现报告。
  - current-head SM89 原始证据已完整取回；八个独立项全部保持 split。
  - HTTP cell 记录强制 A/B 的实际 resolved plan 与 reason；Auto 部分命中仍由
    production resolver 和启动日志独立证明，避免把 requested mode 冒充执行拓扑。
- **Pitfalls encountered**:
  - diff 规模较大，仅按文件罗列会掩盖配置、buffer 和 graph 之间的依赖，因此报告改按请求执行链组织。
  - “默认仍 split”不等于资源行为绝对不变；新实现会省掉旧 baseline 未使用的 decode `qkv_out`。
- **Lessons learned**:
  - fused projection 的核心风险不是数学公式，而是 GEMM 分组改变后的 BF16 reduction path。
  - 对 CUDA Graph 路径，policy、buffer variant 和 tuning shape 必须是同一个构造期事实。
  - 实验配置必须进入真实 server/benchmark 路径，否则测到的结果不能作为生产证据。
  - 组合协同收益可以成为新候选，但不能在看完数据后改变独立资格规则；SM86/SM89
    结论不同，Auto 必须具备硬件适用边界。
- **Follow-ups**:
  - 提交本轮五 target LoRA fixture，并在无 override 的 fresh checkout 跑默认 LoRA gate。
  - 是否追求 TP1 decode both 的 `2.35%`，作为独立、预注册的组合策略实验决定；
    在此之前 Auto 继续 split。
  - 其他 Qwen3 size、TP>2、sm90/sm120、Pin/PerToken 和 Green Context 仍需独立证据。
