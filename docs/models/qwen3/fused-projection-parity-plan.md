# Qwen3 parity-safe fused projections（Issue #746）

> **TL;DR:** QKV 与 gate/up 候选融合路径已独立接入 Qwen3 decode、prefill/unified、verify、LoRA、trace 和 benchmark 配置，但尚未在 Linux CUDA 上完成编译、TP1/TP2 正确性与 32-cell 性能门禁；因此生产 `Auto` 白名单仍为空、默认行为仍为 split，下一步是 GPU 实测而不是提前放行。
>
> **Last touched:** 2026-07

Issue: [openinfer-project/openinfer#746](https://github.com/openinfer-project/openinfer/issues/746)

## Preparation

- **Read**:
  - `docs/index.md` — Qwen3 的 TP、accuracy、kernel report 和 benchmark 规范是本任务的直接约束。
  - `docs/models/qwen3/tp-design.md`（QKV/MLP 分片与当前状态相关段落）— TP2 每个 rank 的本地 QKV 为 `[3072, 2560]`，本地 gate/up 为 `[9728, 2560]`；TP CUDA Graph 在启动阶段按 rank 同步预捕获。
  - `docs/models/qwen3/accuracy-gate.md` — 正确性基线是 HF BF16 teacher-forced logits 的 regret/mean/p99 门禁，不是跨 batch bit-identical，也不是 exact text。
  - `docs/subsystems/correctness/logits-golden-gate.md` — 不允许用放宽 tolerance、重生成 golden 或 exact-text 偶合掩盖融合带来的系统性漂移。
  - `docs/subsystems/kernels/kernel-op-reports.md`（full-forward/roofline 相关段落）— decode q_proj 只有约 32–36% SoL，gate/up 约 44–52%，但 tuning 必须保持 L2-cold，且 kernel 报告不能替代端到端结论。
  - `docs/conventions/bench-regression.md` — 同卡复测；TPOT p50 2%、TTFT p50 3% 是现有噪声/回归判断线。
  - `docs/playbooks/model-optimization-pipeline.md` — prefill 与 decode 分开归因，一次只判断一个优化变量。
  - `docs/subsystems/scheduler/scheduler.md` — 历史 fused projection 在旧模型形状上有约 6% TPOT 收益，但该数据不能代替当前 Qwen3-4B、TP1/TP2 的验收。
  - `docs/lessons/exact-match-gate-thread-cublas.md` — cuBLAS 状态是 worker-thread local；新增 tuning/诊断必须在实际执行线程、实际 device/rank 上完成。
- **Relevant history**:
  - PR #75 / commit `6a5b826` — TP 引入后，fused QKV 因 shard-local 数值差异导致 greedy token 翻转，被恢复为 3 个 row-sliced GEMM。
  - Issue #174 / PR #175 — Qwen3-4B 为对齐 HF BF16 边界，gate/up 被拆成 2 个 GEMM；历史实例在 `Tell me a story` 第 5 个生成 token 上从 `6941` 翻到了 `879`。
  - Issue #456 — full-forward roofline 报告将 fused QKV、fused gate/up 列为已知性能杠杆，但要求用 kernel 与端到端数据共同证明。
  - commit `010bcd2` — 仓库历史中已有 QKV split/deinterleave CUDA kernel，可作为语义参考；不能直接照搬旧 ABI、旧 shape 或旧数值结论。
- **Plan**:
  1. 建立 split/fused projection 诊断 runner，分别量化 QKV、gate/up 在 TP1/TP2 与 decode/prefill shape 上的投影误差、激活误差、算法配置和 kernel 时间。
  2. 增加 Qwen3-local、构造期固定的融合策略；默认从空白名单开始，强制模式只服务诊断与 A/B，unsupported shape fail closed。
  3. 独立实现 fused QKV（1 GEMM + bitwise copy split）和 fused gate/up（1 GEMM + 现有 fused SwiGLU），接入 decode DAG、prefill、unified 和 fixed-buffer verify path。
  4. 保持 LoRA 的逻辑投影边界：Q/K/V split 后再加 delta；gate/up delta 写入 combined buffer 的两个 row range。
  5. 在每个 TP rank 的 worker thread 上预调 fused GEMM，再做 CUDA Graph 预捕获；禁止 capture 内分配、调参或切换策略。
  6. 扩充 operator、HF golden、LoRA、unified、graph、TP 和 unsupported-mode 覆盖；不改现有 HF tolerance。
  7. 完成 projection 级与端到端 A/B；按明确门槛生成生产白名单，无收益或未覆盖的组合保持 split。
  8. 回填实跑命令、原始报告、决策表和 debrief；同步 kernel registry、Qwen3 文档与索引 TL;DR。
- **Risks / open questions**:
  - fused GEMM 的输出即使逐元素只差 1 ULP，也可能经过 36 层放大并翻转近似 tie 的 greedy token；两个 fusion 必须独立判定。
  - TP1/TP2 的本地 `M` 不同，cuBLASLt 可能选择不同 tile/split-K/reduction；TP1 通过不能外推 TP2。
  - QKV fused path 必然需要一个 combined 临时输出；prefill 10k token 时会形成可见 HBM 增量，必须在性能报告中同时报告 peak scratch。
  - 当前 LoRA golden adapter 只覆盖 `q_proj` 与 `v_proj`，不能证明 K/gate/up row offset 正确，必须扩 fixture。
  - `NumericPolicy::Pin/PerToken` 和 Green Context stream override 改变 GEMM 路由，不属于首轮性能资格范围；错误地沿用 Tuned 结论会形成隐性回归。
  - Issue 中“`weights.rs` 使用 `DeviceMatrix::vstack`”已过时；当前生产 loader 是 `openinfer-qwen3/src/weights/load.rs::StagedWeightLoader::fused_rows`。

## 1. 目标与非目标

### 1.1 目标

在不降低现有正确性标准的前提下，提供两条可独立选择的候选路径：

```text
QKV split baseline:
  q = GEMM_rows(Wqkv, q-range, X)
  k = GEMM_rows(Wqkv, k-range, X)
  v = GEMM_rows(Wqkv, v-range, X)

QKV fused candidate:
  qkv = GEMM(Wqkv, X)
  (q, k, v) = split_copy(qkv)
```

```text
MLP split baseline:
  gate = GEMM_rows(Wgate_up, gate-range, X)
  up   = GEMM_rows(Wgate_up, up-range, X)
  act  = SiLU(gate) * up

MLP fused candidate:
  gate_up = GEMM(Wgate_up, X)
  act     = fused_SiLU_mul(gate_up)
```

最终默认行为由数据决定，而不是预设“融合一定更好”：

- QKV 和 gate/up 分别放行。
- decode 与 prefill/unified 分别放行。
- TP1 与 TP2 分别放行。
- 任意组合没有正确性证据或没有稳定端到端收益，就继续走 split。

### 1.2 首轮生产资格范围

首轮只允许下列 geometry 进入 fused 白名单：

| 模型 | hidden | Q | K/V | intermediate | TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 2560 | 4096 | 1024 | 9728 | 1、2 |

本地 projection shape：

| 路径 | TP1 weight rows × K | TP2 每 rank weight rows × K |
| --- | ---: | ---: |
| Q | `4096 × 2560` | `2048 × 2560` |
| K/V | `1024 × 2560` | `512 × 2560` |
| fused QKV | `6144 × 2560` | `3072 × 2560` |
| gate 或 up | `9728 × 2560` | `4864 × 2560` |
| fused gate/up | `19456 × 2560` | `9728 × 2560` |

其他 Qwen3 size、TP>2、非默认数值策略不从 Qwen3-4B 的结果外推：

- `Auto`：回退到 split，并只记录一次原因。
- `ForceFused`：executor 构造阶段返回带 projection、phase、TP、local dims 的错误。

### 1.3 非目标

- 不改权重文件格式，不重复拼接权重；复用 `fused_rows` 已形成的 rank-local contiguous allocation。
- 不为其他模型建立通用 fused-projection 框架；Qwen3.5/Kimi/GLM 的现有融合仅作为参考。
- 不融合 QK norm、RoPE、attention、down projection 或 collective。
- 不通过改 HF golden、放宽 `MEAN_TOL/P99_TOL/MARGIN_TOL`、改 sampler tie-break 来“修复”融合。
- 不承诺 TP>2、其他 Qwen3 size、Green Context、Pin/PerToken 的 fused 性能。
- 不在同一 executor 中热切换 split/fused；A/B 使用两个独立构造的 executor。

## 2. 必须保持的系统不变量

1. **数学边界不变**：融合只改变 dense projection 的 GEMM 分组，不改变 RMSNorm、RoPE、attention、residual、collective 和 sampling。
2. **权重顺序不变**：
   - `qkv_proj = [Q; K; V]`
   - `gate_up_proj = [gate; up]`
3. **TP 语义不变**：每个 rank 只处理自己的 head/intermediate row shard；不增加 collective，也不改变 all-reduce 位置。
4. **BF16 边界不偷偷改变**：QKV split kernel只复制 BF16 bit pattern；fused SwiGLU 继续保持当前 `bf16(SiLU(gate))` 再乘 up 的物化顺序。
5. **LoRA 逻辑边界不变**：adapter 仍以 q/k/v/gate/up 五个逻辑 projection 为单位加载、分片、缩放和写回。
6. **Graph 稳定性不变**：capture 前完成 buffer 分配、tuning 和 pointer 固化；replay 期间没有 allocation、policy branch 变化或 host-side shape discovery。
7. **TP graph 锁步不变**：所有 rank 完成 fused shape tuning 后，仍按现有“两阶段 capture/instantiate/upload，再 replay”协议预捕获。
8. **错误语义明确**：强制 fused 不得静默回退；自动模式不得在未验证形状上静默启用。
9. **baseline 可复现**：必须保留显式 `Split` 控制，确保 A/B 与回滚不依赖旧 commit 或临时 patch。

## 3. 设计

### 3.1 构造期策略，而非 step-time 分支

新增 Qwen3-local 配置（命名在实现时可微调，但语义固定）：

```text
FusionControl = Auto | Split | ForceFused

Qwen3ProjectionFusionOptions {
    qkv: FusionControl,
    gate_up: FusionControl,
}

ProjectionPhase = Decode | PrefillUnified
```

`qkv` 和 `gate_up` 使用独立控制。`Auto` 在 executor 构造时解析为不可变的：

```text
ResolvedProjectionFusion {
    decode_qkv: bool,
    decode_gate_up: bool,
    prefill_unified_qkv: bool,
    prefill_unified_gate_up: bool,
}
```

理由：

- 运行中不切换，CUDA Graph topology 和 buffer pointer 可保持稳定。
- decode buffer 与 prefill/unified buffer本来分离，可以分别选择 representation。
- 白名单只需按 `(model geometry, TP, phase, projection)` 决策，不引入按 token 数动态切换和双份 scratch。
- QKV 与 gate/up 可单独强制，能完成 `split / qkv-only / gate-up-only / both` 归因。

建议提供 Qwen3 专用诊断参数：

```text
--qwen3-qkv-fusion auto|split|fused
--qwen3-gate-up-fusion auto|split|fused
```

其中 `fused` 对应 `ForceFused`。参数应同时进入真实 server 和 in-process benchmark，避免 benchmark 使用与生产不同的构造路径。

### 3.2 支持矩阵与 fail-closed

首轮 `Auto` 仅在以下条件全部成立时查询生产白名单：

- geometry 精确匹配 Qwen3-4B；
- `tp_size` 为 1 或 2；
- `NumericPolicy::Tuned`；
- `DecodeOverlap::Off`；
- 对应 `(projection, phase, TP)` 已通过本计划全部 gate。

其余情况：

| 控制 | 行为 |
| --- | --- |
| `Auto` | split；启动日志说明未命中白名单的单一原因 |
| `Split` | split；不分配 fused-only scratch，不 tune fused shape |
| `ForceFused` | 构造失败；不允许运行中才发现 unsupported |

LoRA 不改变 fusion resolution：同一个 executor 的 base/LoRA 请求必须走同一 projection topology。LoRA decode 继续 eager，因为现有 contract 已明确 LoRA 不进入 CUDA Graph。

### 3.3 Buffer 所有权与 HBM

不采用“同时保留 split 和 fused 两套输出 buffer，再在 step 中选择”的简单实现，因为它会永久增加 HBM，尤其会放大长 prefill 的 scratch 峰值。

建议使用 phase-local enum：

```text
QkvProjectionScratch =
    Split
    | Fused { qkv_out }

MlpProjectionScratch =
    Split { gate_out, up_out }
    | Fused { gate_up_out }
```

共有的 `q/k/v` 与 `act_out` 始终保留，因为 attention/SwiGLU 后续需要。

- gate/up 两种 representation 都是 `2 * intermediate * tokens` 个 BF16，理论容量相同，不应因 fused 再多占一份。
- QKV fused 额外需要 `(q + 2kv) * tokens` 个 BF16 combined buffer，报告中必须列出 decode max-batch 与 prefill max-token 的额外字节数。
- 以当前 max decode batch 256 和 prefill 10k 为例，QKV combined scratch 是：
  - TP1：decode `3.0 MiB`，prefill `117.2 MiB`；
  - TP2：每 rank decode `1.5 MiB`，prefill `58.6 MiB`。
- gate/up representation 在 10k prefill 时本来就需要 TP1 `371.1 MiB`、TP2 每 rank `185.5 MiB`；enum 的目标是替换 representation，而不是再增加同等大小的一份 buffer。
- `PrefillBuffers::set_rows`、`BatchDecodeBuffers::set_batch_size`、verify graph fixed buffers 必须同步更新 variant 内 logical row count。
- policy 一经构造不得改变，避免 enum variant 与 graph 记录不一致。

### 3.4 Fused QKV 数据路径

1. 用完整 rank-local `qkv_proj` 运行一次 `gemm_into_checked`。
2. 用 Qwen3 所需的轻量 CUDA kernel 将 column-major `[q_dim + 2kv_dim, N]` 拷贝到紧凑的 `q/k/v`。
3. split kernel 只做 BF16 load/store，不做 float convert、重排 head、norm 或 RoPE。
4. 复用历史 `010bcd2` 的索引语义，但重新建立：
   - checked Rust wrapper；
   - CUDA launch error 返回；
   - FFI 声明；
   - kernel registry；
   - TP1/TP2、多 token、非整块尾部测试。
5. split 完成后，沿用现有 Q/K/V LoRA：
   - decode：现有 grouped Q/K/V delta；
   - prefill/unified：现有 range/indexed delta；
   - 然后才进入 QK norm + RoPE。

必须区分两件事：

- `split_copy(qkv)` 与输入 `qkv` 必须 bitwise 相同，这是 operator 硬门禁。
- `fused GEMM` 与 `3× split GEMM` 不要求 bitwise 相同；它们由 HF logits 门禁和 delta 分布判断。

### 3.5 Fused gate/up 数据路径

1. 完整 rank-local `gate_up_proj` 运行一次 `gemm_into_checked` 到 `[2I, N]`。
2. LoRA-disabled：直接调用现有 `silu_mul_fused_batch_into`。
3. LoRA-enabled：
   - gate delta 写 `row_offset=0`；
   - up delta 写 `row_offset=I`；
   - 再调用同一个 fused SwiGLU。
4. decode 首版可复用现有单 projection row-offset kernel，分别加 gate/up 两次；LoRA 本就不进 graph，先保证语义，不为少两个 LoRA launch 扩展通用 grouped ABI。
5. prefill/unified 复用 `apply_lora_projection_delta_{range,indexed}` 的 row offset。

现有 `silu_mul_fused_matches_split_bf16_rounding` 继续作为 elementwise 边界门禁；新增测试不能把 projection GEMM 的数值差异误归因于 SwiGLU。

### 3.6 cuBLASLt tuning 与算法诊断

默认 `Tuned` 路径在每个 rank 的实际 worker thread 上执行：

- split baseline shape：保持现有 q、kv、gate/up-half tuning；
- fused QKV shape：增加 `q + 2kv` 的 layer-rotated、L2-cold tuning；
- fused gate/up shape：增加 `2I` 的 layer-rotated、L2-cold tuning；
- decode 仅调现有 `BATCH_BUCKETS ∩ N<=GEMM_LT_MAX_N`；
- tuning 完成后才进入 TP CUDA Graph startup pre-capture。

诊断报告至少输出：

- `(M, N, K)`、TP rank、phase、projection、split/fused；
- 实际 backend：tuned cuBLASLt 或 GemmEx fallback；
- 若为 cuBLASLt：algo id、tile id、stages、split-K、reduction scheme、workspace；
- numeric policy、CUDA/cuBLAS 版本、GPU、SM、commit；
- split 聚合时间与 fused 聚合时间，而不是只比较单个 GEMM。

若选中算法元数据当前无法从 cache 查询，新增 report-only 查询接口；不要让生产 launch 依赖字符串日志或全局可变诊断状态。

### 3.7 CUDA Graph、unified 与特殊路径

- `batch_decode_dag.rs` 必须为 fused GEMM 和 QKV split 生成独立 call trace 节点，报告中不能把两者伪装成一个 GEMM。
- prefill 与 unified 共用同一个 resolved representation，避免同一 `PrefillBuffers` 同时携带 split/fused 双份 scratch。
- verify piecewise graph 使用同一 representation；ping-pong residual 指针交换规则不变。
- mixed prefill+decode 的 unified step 必须单独有正确性覆盖，不能用“prefill 和 decode 各自通过”代替。
- DFlash/DSpark/EAGLE 自己的 projection weight/layout 不纳入本 issue；但 target model 的 prefill/verify dense path会经过修改后的 `PrefillBuffers`，至少要运行现有 DFlash losslessness gate 或明确记录因缺少 draft weights 未执行。
- `DecodeOverlap != Off` 首轮自动回退 split，强制 fused 报错；原因是 stream override 会禁用 cuBLASLt，未经测量不能沿用普通流结论。
- `Pin/PerToken` 首轮同样回退/报错；后续若要支持，必须单独增加 pin warmup/envelope 和完整 A/B，不允许 GemmEx 静默兜底。

## 4. 诊断先行

在改生产默认前，新增 model-local projection report。报告使用真实 Qwen3-4B 权重、相同输入和相同 stream，按 layer 比较：

### 4.1 QKV

```text
baseline = {q_split, k_split, v_split}
candidate_qkv = GEMM(Wqkv, X)
candidate = split_copy(candidate_qkv)
```

分别对 Q/K/V 输出：

- bit-equal ratio；
- mean absolute delta；
- p50/p99/max absolute delta；
- BF16 ULP distance histogram；
- 每层最大差异与首次非零差异；
- split 三 GEMM 总时间；
- fused GEMM、split-copy、两者合计时间；
- 实际算法元数据。

### 4.2 gate/up

比较两个边界：

1. projection 输出：split gate/up vs combined gate/up 的对应 row；
2. activation 输出：split SiLU×mul vs fused SiLU×mul。

这样可区分“GEMM reduction order 差异”和“elementwise materialization 错误”。

### 4.3 Shape 矩阵

诊断最少覆盖：

| phase | token count `N` |
| --- | --- |
| decode | 1、2、4、8、16、32、64 |
| prefill/unified | 128、512、1024、2048、4096、10000 |

每个 shape 都跑 TP1 和 TP2 rank-local geometry。TP2 至少报告 rank 0 与 rank 1；若两 rank 算法相同，也不能只测一个 rank，因为实际权重数据不同会影响 L2-cold timing 与数值分布。

诊断输出保存为结构化 JSON，并由 report 命令打印可审阅摘要。它是“解释为什么不同”的证据，不是生产正确性 gate。

## 5. 实施步骤与代码位置

### Step 1 — 建立策略与构造期验证

拟修改：

- `openinfer-qwen3/src/lib.rs`
  - 增加两个独立 fusion control；
  - 传入 launch/executor。
- `openinfer-qwen3/src/projection_fusion.rs`（新增）
  - geometry/TP/phase 支持判断；
  - `Auto/Split/ForceFused` resolution；
  - 生产白名单；
  - 明确错误消息和单元测试。
- `openinfer-server/src/main.rs`
  - Qwen3-only CLI 接线。
- `openinfer-server/src/bin/bench_serving/{cli.rs,main.rs}`（以实际文件布局为准）
  - Qwen3 TP size 与 fusion A/B 接线；修正当前 Qwen3 in-process bench 固定 `[0]` 的限制。

完成条件：

- 默认白名单为空时行为与当前 main 完全一致；
- TP3/TP8、其他 geometry、Pin/PerToken、Green Context 的 auto/force 行为有单测；
- 任何 unsupported force 在模型执行前失败。

### Step 2 — 恢复 checked QKV split operator

拟修改：

- `openinfer-kernels/csrc/shared/fused_proj.cu`
- `openinfer-kernels/src/ffi/shared.rs`
- `openinfer-kernels/src/ops/elementwise.rs`，或一个更准确的 QKV split model-local wrapper位置
- `openinfer-kernels/KERNELS.md`

完成条件：

- TP1/TP2 local dims、`N={1,8,128}`、非 256 整除 tail 均 bitwise copy；
- 错误维度在 Rust operator 边界失败；
- CUDA launch error 当场返回，不延迟到后续 attention/collective。

### Step 3 — 接入 decode

拟修改：

- `openinfer-qwen3/src/batch_decode_buffers.rs`
  - 以 resolved variant 分配 QKV/MLP scratch；
  - 更新 batch size、pin shape/report shape。
- `openinfer-qwen3/src/batch_decode_dag.rs`
  - fused QKV GEMM、QKV split、fused gate/up GEMM、fused SwiGLU trace/launch。
- `openinfer-qwen3/src/batch_decode.rs`
  - QKV 与 gate/up 独立分支；
  - LoRA row-offset 接线；
  - 保持 collective/residual 顺序。
- `openinfer-qwen3/src/executor.rs`
  - 每 rank capture 前 tuning；
  - fused shape 与 resolved plan 一致。
- `openinfer-qwen3/src/batch_decode_trace.rs` 及 call-spec 相关文件
  - trace 可区分 split/fused 节点。

完成条件：

- eager TP1/TP2 强制 split、QKV-only、gate-up-only、both 都能跑；
- graph capture/replay 无 step-time allocation；
- graph replay topology 与 resolved plan 一致；
- LoRA 请求仍 eager，base 与 LoRA 混合 batch 不串写。

### Step 4 — 接入 prefill、unified 和 verify graph

拟修改：

- `openinfer-qwen3/src/prefill.rs`
  - `PrefillBuffers` representation；
  - `forward_layer_pre_attn` 与 `forward_layer_post_attn`；
  - `set_rows`。
- `openinfer-qwen3/src/unified_forward.rs`
  - 全 token QKV 与 MLP 路径；
  - decode-row offset、prefill plan 与 LoRA token groups 不变。
- `openinfer-qwen3/src/verify_graph.rs`
  - fixed buffer 与 capture topology 适配。

完成条件：

- 纯 prefill、纯 decode、unified mixed step、verify piecewise graph 都使用构造期选定 topology；
- chunked prefill 多次 `set_rows` 不越界、不保留 stale logical size；
- prefix-cache hit/miss 不改变 fusion plan。

### Step 5 — 扩充 LoRA 证据

拟修改：

- `tools/accuracy/dump_qwen3_4b_lora_golden.py`
  - fixture target 从 `q_proj/v_proj` 扩为 `q/k/v/gate/up`；
  - 每个目标都生成可观测、非零且确定性的 delta。
- `openinfer-qwen3/tests/lora_golden_gate.rs`
  - TP1/TP2；
  - split 与 fused；
  - base-only、LoRA-only、mixed base/LoRA batch；
  - 增加逐 target engagement 检查，避免某个 projection 的 delta 为零却整体通过。
- `openinfer-qwen3/src/lora.rs` / `batch_decode.rs`
  - 只补 row-offset 接线或局部 helper，不改变 adapter 格式。

完成条件：

- 五个逻辑 projection 任意一个 row offset 错误都会令测试失败；
- 不以“整体 LoRA 与 base 有差异”代替逐 target 覆盖；
- TP2 adapter shard rows 与 combined buffer row ranges 一致。

### Step 6 — 正确性矩阵

#### Operator 层

- QKV split bitwise copy。
- fused SwiGLU 与 split SwiGLU 保持现有 BF16 rounding 一致。
- shape validation 与 unsupported force。
- buffer logical rows 更新。

#### HF logits gate

Qwen3-4B 必须完成：

| fusion | TP | eager bs=1 | eager batched | graph bucket straddle |
| --- | ---: | ---: | ---: | ---: |
| split | 1 | 必须 | 必须 | 必须 |
| QKV-only | 1 | 必须 | 必须 | 必须 |
| gate-up-only | 1 | 必须 | 必须 | 必须 |
| both | 1 | 必须 | 必须 | 必须 |
| split | 2 | 必须 | 必须 | 必须 |
| QKV-only | 2 | 必须 | 必须 | 必须 |
| gate-up-only | 2 | 必须 | 必须 | 必须 |
| both | 2 | 必须 | 必须 | 必须 |

规则：

- 继续使用现有 regret `≤0.20 nat`、mean `≤0.06`、p99 `≤0.20`。
- 不重新生成 base HF golden。
- 不要求 split 与 fused bit-identical，也不使用 free-running exact text 作为硬门禁。
- 同一 fusion、同一输入、同一执行形状的重复 eager run 继续要求 deterministic。
- eager 与 graph 用同一 batch composition 时至少都通过相同 HF gate；若出现 graph-only drift，按 pointer/padding/capture bug 处理。

#### 路径层

- 新增或扩展 unified mixed-step teacher-forced 覆盖。
- prefix-cache eager/graph replay 保持通过。
- LoRA 五 projection TP1/TP2 gate。
- 现有 DFlash losslessness gate在具备 draft weights 的环境执行；缺失时在 Execution Log 明确记录，不能写“已覆盖”。
- 其他 Qwen3 size 运行现有 TP1 HF gate或至少证明 `Auto` 选择 split；不得意外命中 4B 白名单。

### Step 7 — Kernel report

扩展：

- `openinfer-qwen3/src/kernel_bench.rs`
- `openinfer-qwen3/kernel_manifests/qwen3.toml`
- `openinfer-qwen3/src/bin/qwen3_kernel_report.rs` 及其 report 数据结构

新增公平的 aggregate case：

```text
QKV split       = q GEMM + k GEMM + v GEMM
QKV fused       = fused GEMM + split-copy

gate/up split   = gate GEMM + up GEMM + split SiLU×mul
gate/up fused   = fused GEMM + fused SiLU×mul
```

要求：

- aggregate 用一对 CUDA events 包住完整候选路径；
- 同时保留 component timing，解释收益来自 GEMM 还是被 split kernel 吃掉；
- tuning 使用与 executor 相同的 cold-weight rotation；
- TP1/TP2、decode/prefill shape 全覆盖；
- 输出 launch count、p50/p99/avg、delta%、HBM scratch；
- Nsight Systems 只用于确认 kernel composition/launch count，不能拿 graph node trace 的膨胀时间作为绝对值。

### Step 8 — 端到端 A/B

每个 TP 在同一 commit、同一 GPU 型号、同一 CUDA/cuBLAS、同一空闲机器上运行四个 mode：

1. split baseline；
2. QKV-only；
3. gate-up-only；
4. both。

最小矩阵是 `4 modes × 2 TP × 2 profiles × 2 concurrency = 32 cells`：

| profile | 请求 shape | concurrency | 主指标 | 次指标 |
| --- | --- | ---: | --- | --- |
| prefill-heavy | `prompt=10000, output=1` | 1、8 | TTFT p50 | TTFT p99、request tok/s、peak HBM |
| decode-heavy | `prompt=1024, output=256` | 1、8 | steady TPOT p50 | TPOT p99、output tok/s |

如果 10k×c8 在目标卡 OOM：

- 不临时调低后仍声称完成原矩阵；
- 记录 OOM 与显存构成；
- 使用同一较小 prompt 对四个 mode 做补充 A/B；
- 原 prefill c8 cell 保持“未完成”，生产 prefill 白名单不得据此放行。

测量规则：

- warmup 后至少 20 个有效样本；
- 每个关键 cell 独立运行两次，顺序交错 `split→candidate→candidate→split`，降低温度/频率漂移；
- 记录 GPU clocks/power、driver、CUDA/cuBLAS、commit、model hash、fusion resolution；
- failed request 必须为 0；
- TP2 使用真实两 rank server/executor，不用把 TP1 kernel shape 除以 2 的模拟结果代替。

### Step 9 — 生产放行与清理

对每个 `(projection, phase, TP)` 独立决策：

#### 正确性硬门槛

- 对应 force mode 的全部 HF/graph/LoRA/path gate 通过；
- 不修改 tolerance；
- 无 unsupported fallback、CUDA error、NaN/Inf、跨请求污染；
- split-copy/operator 边界测试通过。

任一失败：该组合不得进入白名单。

#### 性能门槛

- decode：两次复测方向一致，TPOT p50 相对 split 改善至少 2%，且 c1/c8 任一 cell 不回退超过 2%；
- prefill/unified：两次复测方向一致，TTFT p50 改善至少 3%，且 c1/c8 任一已完成 cell 不回退超过 3%；
- kernel aggregate 必须同方向变快；若 kernel 快而 E2E 不快，不放行；
- throughput 不得与 latency 主指标产生不可解释的反向显著回退；
- HBM 增量必须记录，并且不破坏标准 profile 的 admission。

#### 白名单结果允许是部分启用

例如以下结果是合法的：

```text
TP1 decode:          QKV fused, gate/up split
TP2 decode:          QKV split, gate/up fused
TP1 prefill/unified: both fused
TP2 prefill/unified: both split
```

如果没有任何组合达标：

- 不启用生产 fused；
- 保留结构化诊断结果；
- 删除不再有直接使用者的生产 dead path，而不是长期携带默认关闭的复杂分支；
- issue 以“测量否决”而非虚假性能 claim 收尾。

## 6. 评审重点

请重点 review 以下决策，而不只是文件是否齐全：

1. **首轮是否应只白名单 Qwen3-4B**：本计划认为必须如此，其他 size 没有当前 issue 要求的性能证据。
2. **是否接受构造期静态选择**：它牺牲按 `N` 动态选择的局部最优，换取零双份 scratch、稳定 graph topology 和更小状态空间。
3. **Pin/PerToken/Green Context 是否先回退**：本计划不把默认 Tuned 的结论外推到不同 GEMM backend/stream contract。
4. **端到端是否需要 32 cells**：若只跑 split vs both，无法判断 QKV 与 gate/up 谁真正贡献收益，也无法安全做部分白名单。
5. **LoRA 是否必须重做五 projection fixture**：当前 q/v-only fixture不足以验收 gate/up combined row offset。
6. **放行阈值是否沿用 2%/3%**：低于现有噪声线的“收益”不应换取长期维护成本。

## 7. 计划中的验证命令

以下是执行阶段要落地的命令类别。新 runner/flag 尚未实现，因此本节不伪造最终 CLI；实现后必须先运行 `--help`，再把实际成功命令与报告路径回填到 Execution Log。

已有仓库门禁：

```bash
cargo fmt --all --check
cargo test --release \
  -p openinfer-kernels \
  -p openinfer-qwen3 \
  -p openinfer-server \
  --lib
OPENINFER_TEST_MODEL_PATH=models/Qwen3-4B \
  cargo test --release -p openinfer-qwen3 --test hf_golden_gate -- --nocapture
OPENINFER_TEST_MODEL_PATH=models/Qwen3-4B \
  cargo test --release -p openinfer-qwen3 --test lora_golden_gate -- --nocapture
```

新增后必须有可复现命令覆盖：

- projection numerical report：TP1、TP2；
- kernel aggregate report：split、QKV-only、gate-up-only、both；
- HF golden：四种 fusion mode × TP1/TP2 × eager/graph；
- in-process 或真实 server benchmark：32-cell 矩阵；
- unsupported force：TP>2、其他 geometry、Pin/PerToken、Green Context。

任何未实际运行的命令都不能在 Debrief 中标记为通过。

## 8. 预期交付物

- Qwen3-local fusion policy 与 fail-closed support matrix。
- checked QKV split CUDA operator。
- decode、prefill、unified、verify graph 的独立 QKV/gate-up fused paths。
- TP-aware、LoRA-aware、graph-safe 的 tuning 与 buffer 管理。
- projection numerical JSON 报告。
- TP1/TP2 kernel aggregate JSON 报告。
- 32-cell 端到端 A/B 表与原始产物路径。
- 扩展后的五 projection LoRA fixture/gate。
- 最终生产白名单及每个 entry 的 correctness/performance 证据。
- 对未启用组合的明确否决原因。
- 更新后的 `docs/models/qwen3/accuracy-gate.md`、相关 kernel/perf 文档、`openinfer-kernels/KERNELS.md` 和本任务 Debrief。

## Execution Log

### Step 0 — 计划批准与分支

- 用户已批准本计划。
- 从 `main` 创建 `feat/qwen3-fused-projection-parity`，后续实现不直接落在 `main`。
- 默认生产白名单保持为空，先完成强制路径、正确性和性能证据。

### Step 1 — 策略与入口（已实现，待 Linux/GPU 编译验证）

- 新增 `openinfer-qwen3/src/projection_fusion.rs`：
  - QKV 与 gate/up 独立的 `Auto/Split/ForceFused`；
  - 构造期解析为 decode、prefill/unified 四个不可变布尔值；
  - 强制模式仅接受 Qwen3-4B、TP1/TP2、`NumericPolicy::Tuned`、`DecodeOverlap::Off`；
  - 生产 `Auto` 白名单保持为空。
- 配置已穿过 server、scheduler、executor 与 model loader。
- server 新增 `--qwen3-qkv-fusion`、`--qwen3-gate-up-fusion`。
- in-process `bench_serving` 同样接入两个参数，并修正原 Qwen3 分支将
  device 固定为 `[0]`、无法真正运行 `--tp-size=2` 的问题。
- 新增 `launch_with_seed`，保证 benchmark 改走统一 launch policy 后原有
  `--seed` 仍然生效。

### Step 2 — checked QKV split operator（已实现，待 GPU 单测）

- 在 `openinfer-kernels/csrc/shared/fused_proj.cu` 恢复纯 BF16 load/store 的
  contiguous QKV split kernel。
- 新增 FFI、checked Rust wrapper、call spec 和 kernel registry 条目。
- wrapper 在 launch 前检查非空维度、shape 一致、i32 维度和索引上限，
  launch 后立即返回 CUDA error。
- GPU 单测覆盖 Qwen3-4B TP1/TP2 的 `N=1/8/128`，以及不对齐 256-thread
  block 的 tail shape。

### Step 3/4 — decode、prefill、unified、verify（已实现，待编译与 GPU gate）

- decode 与 prefill buffer 只分配 resolved topology 需要的 representation：
  fused QKV 增加 combined scratch；gate/up 在 split pair 与 combined
  buffer 之间二选一，不保留双份。
- decode DAG 增加 fused QKV GEMM、split-copy、fused gate/up GEMM 和 fused
  SwiGLU 节点；原 split 节点保持不变。
- decode、prefill、unified 与 verify graph fixed buffer 都使用同一构造期
  plan；`set_batch_size`/`set_rows` 同步 Option 内 logical shape。
- decode tuning 按 topology 只 tune 实际会运行的 Q/2KV 或 combined shape，
  并保持 layer-rotated samples。
- LoRA 继续按逻辑 projection 应用：Q/K/V 在 split 后写；gate/up fused
  buffer 分别写 row `0` 与 row `I`。

### Step 5/6 — 正确性入口（部分完成）

- HF golden 与 LoRA golden 新增环境选择：
  `OPENINFER_QWEN3_PROJECTION_FUSION=split|qkv|gate-up|both`。每个模式通过
  正常 executor 构造期配置运行，不在 step 中热切换。
- `tools/accuracy/dump_qwen3_4b_lora_golden.py` 已扩展为生成
  q/k/v/gate/up 五 projection adapter，并校验每层每 target 的 A/B tensor
  数量。
- 尚未重生成 `test_data/qwen3-4b-lora-golden.safetensors`；本机没有
  Qwen3-4B 权重与 CUDA，因此当前 committed fixture 仍只证明 q/v，不足以
  宣称 gate/up LoRA 门禁完成。

### Step 7/8 — 报告与端到端 A/B 入口（代码完成，待 GPU 产物）

- `qwen3_model_report` 新增独立 `--qkv-fusion` / `--gate-up-fusion`，trace
  来自真实 `batch_decode` DAG；split/fused SwiGLU 使用不同 call spec，
  fused QKV 的 split-copy 也有独立 provider。现有 report 在 `Tuned` 下会
  将无法忠实重放 startup tuning 的 GEMM 标为 excluded，因此该 report
  只能证明 topology/非 GEMM component，不能代替端到端性能资格。
- report schema 记录两条 projection topology，默认产物路径包含 topology，
  避免不同模式互相覆盖。
- `bench_serving` 已能以同一启动路径运行四种组合和真实 TP1/TP2，可用于
  端到端 32-cell A/B；JSON/text run metadata 记录 TP size 与两条 requested
  fusion mode，避免产物脱离实验组语义。
- 新增 `qwen3_projection_report`：用真实 rank-local 权重、同一 patterned
  BF16 输入和同一 CUDA stream，按 36 层比较 QKV raw projection、
  gate/up raw projection 与完整 SwiGLU aggregate；覆盖 TP1 与 TP2 两个
  rank、decode/prefill shape，输出 BF16 ULP histogram、abs delta、
  p50/p99/avg、launch 数、scratch bytes 和实际 tuned cuBLASLt algorithm
  metadata。`N<=32` 使用与 executor 一致的 all-layer cold-weight rotation。
- `tools/validation/qwen3_fused_projection_suite.py` 统一编排：
  - 五 projection LoRA fixture fail-closed 预检；
  - Qwen3 scoped unit、operator、HF/LoRA 的四 mode × TP1/TP2 精确矩阵；
  - projection rank reports 与四 mode decode topology reports；
  - 32 个唯一 E2E cell 的两次对称交错复测（实际 64 个 benchmark 进程）；
  - commit/model hash、driver/CUDA、逐 benchmark clocks/power/peak-HBM、
    原始日志、manifest、summary、decision table 与 Markdown 报告。
- 汇总器按 projection/phase/TP 独立执行 2% decode、3% prefill 门槛，同时
  要求重复方向一致、throughput 无同阈值级反向回退、kernel aggregate
  同方向；`both` 只作交互诊断，不用于替代单 projection 归因。
- 当前 committed LoRA fixture 的 metadata 仍只有 q/v；suite 会在任何
  GPU 工作之前明确失败并要求重生成，不会把旧 fixture 误计为 gate/up
  覆盖。
- projection 数值 JSON、TP2 rank-local aggregate 和 32-cell 原始数据仍
  尚未生成；它们依赖 Linux CUDA、两张 GPU、Qwen3-4B 权重和重生成的
  五 projection fixture。
- 新增 `fused-projection-parity-implementation-report.md`，按
  CLI→resolved policy→buffer→forward→TP/LoRA/Graph→gate 的真实执行链
  解析当前 diff，并将 Option-vs-enum、Auto fallback reason、projection
  numerical runner 等实现偏差列为合入前评审项。

### 本地验证记录与环境阻塞

- `cargo fmt --all`：通过。
- `git diff --check`：通过。
- `cargo metadata --no-deps --format-version 1`：通过。
- LoRA fixture 生成脚本 Python AST parse：通过。
- validation suite `--help` 与完整 `--dry-run`：通过；完整矩阵展开为
  103 个命令（20 个正确性/预检、3 个 rank numerical report、16 个
  topology report、64 个 E2E benchmark）。
- validation suite Python 单测：7/7 通过，覆盖 Qwen3 unit package 精确作用域、
  projection report feature、topology shared-KV mode、phase-specific 2%/3%
  threshold、重复方向不一致 fail-closed、旧 q/v-only fixture fail-closed，
  以及完整 synthetic 103-command 证据到最终 Markdown/decision table 的汇总。
- 完整 dry-run 复核仍为 `103 = 20 correctness + 3 projection + 16
  topology + 64 benchmark`，manifest 中没有任何 `--workspace` 参数。
- 当前 fixture 预检：按预期失败，明确缺少 `k_proj/gate_proj/up_proj`；
  这是待补产物，不是已通过 gate。
- `OPENINFER_CUDA_SM=120 cargo check --release -p openinfer-qwen3 --lib`：
  未进入 openinfer crate 编译；macOS host 在 Linux-only
  `rdma-mummy-sys` build script 因缺少 `endian.h`、`linux/types.h` 失败。
- `OPENINFER_CUDA_SM=120 cargo check --release -p openinfer-core --lib`：
  进入 `openinfer-kernels` build script 后因本机无 `nvcc` 失败，未进入
  Rust crate type-check。
- Docker CLI 存在，但 daemon 未运行，不能用仓库 Linux dev container
  继续编译。
- 当前机器无 `nvcc`/CUDA GPU，因此 operator/HF/LoRA/TP/CUDA Graph 与
  性能矩阵均未执行；这些项目保持“待验证”，不会据此填充生产白名单。

### Step 10 — 收敛 Qwen3 验证门禁作用域

- GPU 主机首次执行暴露出 `cargo test --release --workspace --lib` 会编译
  GLM5.2/Kimi-K2，并通过 `openinfer-kernels/moe` 拉入 DeepEP shim，导致
  Qwen3 专项验证额外要求 `OPENINFER_NCCL_ROOT` 指向 NCCL ≥ 2.30.4。
- 将该门禁改为单个 Cargo invocation，只选择 `openinfer-kernels`、
  `openinfer-qwen3`、`openinfer-server` 三个 Qwen3 直接相关 package；
  保留 kernel/model/server 三层 lib tests，不再编译无关模型 feature。
- 汇总键由 `workspace-lib` 改为 `qwen3-unit`，并新增命令作用域单测，
  精确断言 package 集合且禁止 `--workspace` 回归。
- Qwen3 TP2 仍使用其正常 NCCL collective；本修复只移除 DeepEP 2.30.4
  这一无关构建门禁，不绕过 TP 通信或任何 HF/LoRA/operator gate。

### Step 11 — Linux Qwen3 编译面修复

- AutoDL 首次执行 scoped unit gate 后，`openinfer-qwen3` 在 prefill 与
  unified forward 的 `ops::split_qkv_into` 调用处报 `E0425`；后续
  `qwen3_model_report` topology binary 也使用同一路径，尚未执行到。
- 根因是 operator 已从 `openinfer-kernels::ops` 导出，但遗漏了
  `openinfer-core::ops` facade re-export；Qwen3 其他常规 GPU operator
  均通过该 facade 使用。
- 在 core facade 补一个统一 re-export，同时修复 lib 与
  `kernel-report` binary 三个调用面；不复制 wrapper、不改变 kernel ABI
  或运行时行为。
- 本机可完成格式、metadata、Python suite 与 dry-run 检查；Linux CUDA
  type-check 仍以 AutoDL 重跑 scoped unit gate 为准。

### Step 12 — QKV bitwise-copy gate 的 NaN 比较修复

- 编译面修复后，AutoDL 进入
  `split_qkv_is_a_bitwise_copy_for_tp_shapes_and_tails` 并在首个 Q slice
  `assert_eq!` 失败。
- 测试输入从任意 `u16` payload 构造 BF16，覆盖 NaN payload；Rust `bf16`
  的 `PartialEq` 遵守 IEEE 语义，即使 payload 相同也满足 `NaN != NaN`。
  因此原 slice equality 不能证明或否定 bitwise copy。
- 保留任意 payload 覆盖，将 Q/K/V 断言改为逐元素比较 `to_bits()`，错误
  信息带 segment、shape、token 与局部 index；另加相同 NaN payload 的
  CPU 回归测试，固定测试契约。
- 不修改 CUDA split kernel、输入分布或正确性阈值；AutoDL 需重跑 operator
  gate，只有按位断言通过后才能认为 kernel copy 正确。

### Step 13 — Projection report binary 的 feature 边界

- AutoDL 单独编译 HF gate 时，Cargo 同时构建没有 `required-features` 的
  package binary；`qwen3_projection_report` 使用 optional `clap`，但默认
  feature 未启用它，因而报 unresolved import 和缺失 derive attributes。
- 新增最小 `projection-report = ["dep:clap"]` feature，并将该 binary 标记
  为 required feature；HF/LoRA/default lib 构建会跳过它。
- validation suite 的 projection rank 命令显式传
  `--features projection-report`；不复用会额外拉入 CUPTI/trace/report
  dependencies 的 `kernel-report` feature。
- binary 自带 example 同步加入 feature，避免人工复现命令再次失败。

### Step 14 — Topology trace 的 KV 容量去耦

- AutoDL 运行 `batch=64, kv_len=2048` topology cell 时，trace harness 为
  64 个 synthetic request 各分配 2 个 Qwen3 KV blocks，超过当前 pool，
  在收集 DAG 前报 `allocation failed: needed 128 blocks`。
- topology report 只消费 batch/sequence shape 与记录的 operator calls，
  不验证 request-local KV 内容；真实正确性和性能由 HF/LoRA/E2E gate
  负责。
- suite 通过显式 `--shared-kv-pages` 让 trace harness 只分配“一张目标长度
  页表 + padding block”，所有 synthetic rows 共享有效物理页表；仍以真实
  `batch=64, kv_len=2048` 调用 production `batch_decode`，所以 DAG、
  tensor shape 与 launch topology 不缩水。report schema/config 记录该模式，
  默认 standalone model report 继续使用独立 KV。
- 共享仅存在于 `kernel-call-trace` harness；serving executor、HF/LoRA 和
  benchmark 仍使用独立 request KV。新增容量公式回归，确认 2048/1024
  只需 3 个物理 blocks，且不乘 batch。

## Debrief

- **Outcome**:
  - 融合候选与 103-command fail-closed 验证套件已实现；生产白名单仍为空。
  - Qwen3 unit 门禁已收敛到 kernel/model/server 三个相关 package，不再要求
    为无关 GLM/Kimi DeepEP 安装 NCCL ≥ 2.30.4。
- **Pitfalls encountered**:
  - `cargo test --workspace` 不只是“多跑一些测试”；Cargo feature union 会让
    无关 workspace member 激活共享 `openinfer-kernels/moe`，改变构建依赖边界。
  - 本机静态检查不能替代 Linux CUDA TP2 实跑，尤其不能证明 NCCL
    communicator、CUDA Graph capture 或 fused GEMM 数值结果。
  - 新 kernel operator 只在 kernels crate 导出不足以覆盖使用
    `openinfer_core::ops` facade 的模型路径；lib 与 report binary 都必须在
    Linux 编译面验证。
  - “bitwise”测试不能使用浮点 `PartialEq`；任意 bit-pattern fixture 会包含
    NaN，相同 payload 也会按 IEEE 规则比较为不等。
  - optional CLI dependency 与 Cargo binary target 必须成对声明 feature
    boundary；否则看似无关的 integration test 也会在编译 package targets
    时失败。
  - topology/shape trace 不应继承 serving admission 的完整状态成本；可以
    共享无语义 payload，但必须把这种别名严格限制在非正确性、非性能 harness。
- **Lessons learned**:
  - 专项验证的 preflight 必须与被验证产品面的 feature/package closure 一致；
    全 workspace 健康度可以是独立 CI，但不能成为 Qwen3 优化报告的隐藏前置条件。
  - 门禁作用域也需要结构化回归测试，不能只依赖文档约定。
- **Follow-ups**:
  - 在 AutoDL 双卡主机重新运行完整 suite，根据真实报告决定各
    `(projection, phase, TP)` 是否进入白名单。
  - 其他 Qwen3 size、TP>2、Pin/PerToken 和 Green Context 仍需独立证据。
