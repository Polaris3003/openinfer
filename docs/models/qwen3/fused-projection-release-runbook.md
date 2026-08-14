# Qwen3 fused projection GPU 验证与放行 Runbook（Issue #746）

> **TL;DR:** `a8ef928` 已在 2×RTX 4090（SM89）完成 schema-v3 全量验证：104/104 命令、21/21 correctness、3/3 projection、16/16 topology、64/64 HTTP cell 全绿；八个预注册的独立资格项全部 `KEEP_SPLIT`，因此生产 `Auto` 不改。TP1 decode 同时融合 QKV+gate/up 的 TPOT 平均改善 `2.35%`，仅作为下一轮组合策略信号，不能事后改本轮规则直接放行。
>
> **Last touched:** 2026-08

Issue: [pegainfer-project/pegainfer#746](https://github.com/pegainfer-project/pegainfer/issues/746)

## Preparation

- **Read**:
  - `docs/index.md` — 本任务归属 Qwen3 model line，结果应回填模型文档而不是建立独立生命周期目录。
  - `docs/models/qwen3/fused-projection-parity-plan.md` — QKV、gate/up、decode、prefill、TP1、TP2 必须独立判定；正确性不能通过放宽 tolerance 换取。
  - `docs/models/qwen3/fused-projection-parity-implementation-report.md` — 候选实现构造期固定，显式 fused fail closed，生产 `Auto` 从空白名单开始。
  - `docs/models/qwen3/accuracy-gate.md` 与 `docs/subsystems/correctness/logits-golden-gate.md` — HF/LoRA 结论必须看 regret mean/p99 与结构门禁，不能只看 free-running 文本。
  - `docs/conventions/bench-regression.md` — decode `2%`、prefill `3%` 是预先约定的性能资格线。
- **Relevant history**:
  - 旧 SM86 数据只有 TP1 decode QKV 达到资格线；不能外推到 SM89 或其他 cuBLAS 选择。
  - 当前 resolver 不能表达 GPU SM、CUDA/cuBLAS 或 selected-algo 身份，任何全局白名单都会超出证据边界。
- **Plan**:
  1. 固定 commit、模型、fixture、GPU 和工具链身份，校验归档 SHA256 与结构安全性。
  2. 独立核对 104 条命令是否真正执行，排除返回 0 但 test 被过滤、HTTP trace 缺失或 fused 静默 fallback。
  3. 复算八个独立决策及 `both` 交互数据，给出是否修改 fused/Auto 的明确结论。
  4. 记录剩余 PR 闭环项，不把候选实现可合入与默认 fused 混为一个决定。

## 1. 本轮固定身份

| 项目 | 实际值 |
| --- | --- |
| Branch | `feat/qwen3-fused-projection-parity-v2` |
| Commit | `a8ef9286eb329b8106157d2ca36956ad87f48d10` |
| Repository | `/root/openinfer` |
| Model | `/root/autodl-tmp/models/Qwen3-4B` |
| Model full hash | `4b5fcd3463ccfb909a9cf41e5ccabfae1ee4310eb022077434932b0e1f42f0e1` |
| GPU | 2×NVIDIA GeForce RTX 4090，SM89，各 24564 MiB |
| Driver / toolkit | `580.105.08` / CUDA toolkit `12.6.85` |
| Rust | nightly `1.99.0` (`af3d9558`) |
| Generated fixture | `f8b76cb461c30b02d4f3f6d77de8c868ac60d4959192b52a88f6a8d4e0421b4f` |
| Evidence archive | `qwen3-fused-746-jFnKoj.tar.gz` |
| Archive SHA256 | `be9955ae412cee92a3405d119719bec29f6c54ad17cb222b92340d45c1fe277c` |

归档内没有绝对路径、`..`、符号链接、硬链接或设备节点；共 422 个成员、无重复路径。归档中存在 Jupyter 自动生成的 `.ipynb_checkpoints`，但正式 manifest 没有引用这些副本，正式输出路径均存在且非空。

## 2. 已验证的一键执行入口

GPU 主机使用下面的入口完成本轮验证与自动关机：

```bash
cd /root/openinfer
git pull --ff-only
git rev-parse HEAD
bash scripts/run_qwen3_fused_746_gpu.sh --shutdown-after
```

`git rev-parse HEAD` 必须是第 1 节的完整 commit。runner 会依次完成：

1. 拒绝 fixture 之外的 tracked change；未跟踪的旧 artifact 只记录，不作为输入。
2. 备份仓库 fixture；若它仍是 q/v-only，则在隔离 `RUN_ROOT` 生成五 target fixture，不覆盖仓库文件。
3. release 预编译并运行真实 HTTP smoke；等待期间每 15 秒输出 server heartbeat。
4. 生成 104-command dry-run manifest，再执行正式 correctness/projection/topology/HTTP 矩阵。
5. 机械汇总、归档、生成 SHA256；自然成功或失败都先保留证据，再执行 `/usr/bin/shutdown -h now`。

人工 `Ctrl-C`/TERM 会保留当前证据但不关机。只验证 smoke 时可用 `--smoke-only`；需要随后关机时与 `--shutdown-after` 组合。

## 3. 证据完整性

### 3.1 Runner 与结构化结果

| Gate | 结果 |
| --- | ---: |
| `runner-status.txt` | `FULL_PASS` |
| Suite / summarize return code | `0 / 0` |
| Manifest schema / status | `3 / commands-passed` |
| Commands | `104/104`，全部 return code 0 |
| Correctness | `21/21` |
| Projection report | `3/3` |
| Topology report | `16/16` |
| HTTP benchmark | `64/64`，missing=0，errors=0 |

104 条命令的 index 连续、tag 唯一，组成是 21 correctness、3 projection、16 topology、64 benchmark。manifest 引用的所有 log/output 都存在且非空。

### 3.2 不是“返回 0 但没跑测试”

- Qwen3 unit 实际运行 `56 + 17 + 96` 个测试，server CLI 实际运行 4 个测试。
- 两个 CUDA operator gate 各实际运行 1 个目标测试。
- TP1/TP2 × split/QKV/gate-up/both 的 8 个 HF gate 各实际运行 1 个测试。
- 同一矩阵的 8 个 LoRA gate 各实际运行 1 个测试。
- HF 全部日志的最大 mean/p99 分别为 `0.0320/0.1249`；LoRA 为 `0.0330/0.1237`，既有 tolerance 未修改。
- 五 target LoRA fixture 覆盖 `q_proj/k_proj/v_proj/gate_proj/up_proj`，36 层共 360 个 adapter A/B tensor；fixture 中总 tensor 数为 367。

### 3.3 HTTP 证据不是静默 fallback

64 个 HTTP report 全部满足：

- failed=0、timeouts=0；
- server trace coverage=100%；
- token timing coverage=100%；
- prompt/completion token count coverage=100%；
- GPU 运行中采样非空；
- requested split/QKV/gate-up/both 与 resolved plan 完全一致。

TP1/TP2 各有 8 个 split、8 个 QKV-only、8 个 gate-up-only、8 个 both cell，没有 fused 请求静默回退为 split。

## 4. 八个预注册的独立决策

decode 使用 TPOT p50，门槛 `2%`；prefill/unified 使用 TTFT p50，门槛 `3%`。每个决策由 concurrency 1/8 × repeat 1/2 四个 matched A/B 构成。

| Projection | Phase | TP | E2E 平均改善 | 方向一致 | Kernel 同向 | 结论 |
| --- | --- | ---: | ---: | --- | --- | --- |
| QKV | prefill/unified | 1 | `-1.02%` | 否 | 否 | `KEEP_SPLIT` |
| gate/up | prefill/unified | 1 | `-0.43%` | 否 | 是 | `KEEP_SPLIT` |
| QKV | decode | 1 | `+1.38%` | 是 | 是 | `KEEP_SPLIT` |
| gate/up | decode | 1 | `+1.12%` | 是 | 是 | `KEEP_SPLIT` |
| QKV | prefill/unified | 2 | `-1.51%` | 否 | 否 | `KEEP_SPLIT` |
| gate/up | prefill/unified | 2 | `+0.30%` | 否 | 是 | `KEEP_SPLIT` |
| QKV | decode | 2 | `-1.41%` | 否 | 是 | `KEEP_SPLIT` |
| gate/up | decode | 2 | `+0.27%` | 否 | 否 | `KEEP_SPLIT` |

明确结论：

- 数值和执行拓扑都通过；`KEEP_SPLIT` 的原因是性能不足或不稳定，不是 correctness failure。
- TP1 decode 两个单项都稳定变快，但 `1.38%/1.12%` 没达到预先约定的 `2%`，不能为了得到 fused 结论临时降低门槛。
- TP2 既有明显方向翻转，也有单次 `-7.42%` TPOT 回退，不具备放行条件。
- 所有 prefill/unified 组合继续 split；长 prompt 上 QKV kernel 本身也呈负收益。

## 5. `both` 信号如何处理

TP1 decode 的 both 模式相对 split：

| Cell | TPOT 改善 | Output throughput 改善 |
| --- | ---: | ---: |
| c1 repeat 1 | `+2.21%` | `+1.91%` |
| c1 repeat 2 | `+2.10%` | `+1.84%` |
| c8 repeat 1 | `+2.48%` | `+2.76%` |
| c8 repeat 2 | `+2.62%` | `+2.70%` |
| mean | `+2.35%` | — |

这不是坏数据：四个 cell 同向、GPU clocks/power/temperature 正常、正确性和 resolved plan 均通过。但本轮预先定义的是“两个 projection 独立取得资格”，`both` 只观察交互。看完数据后把 `both` 改成资格项，会引入事后选择偏差。

若后续确实要争取这约 `2.35%`，应建立新的组合策略实验：

1. 事先定义 `TP1 + decode + QKV&gate/up` 作为不可拆的候选。
2. 使用新的 `RUN_ROOT` 做更多交错/镜像 repeat，不复用本轮结果作为最终资格数据。
3. 保持 prefill/unified split，只放行 decode combined policy。
4. 先让 resolver 能表达至少 SM89 的适用范围；SM86 与 SM89 的最优项不同，不能使用全局布尔白名单。

## 6. 当前发布决定

本轮做三个分层决定：

1. **候选实现可以继续走 PR**：正确性、LoRA、TP、graph/topology、真实 HTTP 与证据工具链均已闭环。
2. **本轮没有任何独立 fused 行获得生产资格**：八项全部 `KEEP_SPLIT`。
3. **全局 `Auto` 必须保持 split**：当前 resolver 缺少硬件/toolchain 资格边界，且 SM86/SM89 性能结论不同。

显式 `--qwen3-qkv-fusion fused` / `--qwen3-gate-up-fusion fused` 继续作为诊断和受控实验入口；不能宣传为跨硬件默认优化。

## 7. PR 闭环与后续

当前仍有一个 PR 自包含性缺口：仓库 tracked fixture 还是旧 q/v-only 文件；本轮真正通过 8 个 LoRA gate 的五 target fixture 只在归档中。合入前应将 SHA256 为 `f8b76cb4...` 的 fixture 放入 `test_data/qwen3-4b-lora-golden.safetensors` 并提交，然后在不设置 override 环境变量时运行 LoRA gate，证明 fresh checkout 自包含。

`.ipynb_checkpoints` 不影响本轮正式结论，但 runner 后续归档应排除这类自动副本，避免扩大 artifact 和制造审查噪声。

除上述 fixture 自包含 gate 外，本轮不需要重跑完整 104-command suite。若修改 runtime fusion 策略、resolver、projection kernel、fixture 内容或 correctness 路径，则对应证据失效，需要按影响面重跑。

## Execution Log

- 2026-08-12：候选实现与验证套件迁入最新主干，性能门禁改为真实 HTTP serving。
- 2026-08-13：完成可观察的一键 runner、自动关机、tracked-state guard、隔离 fixture 生成及 fixture override。
- 2026-08-13：GPU 主机在 `a8ef928`、2×RTX 4090、CUDA 12.6 上执行完整 runner，生成 `qwen3-fused-746-jFnKoj.tar.gz`。
- 2026-08-14：校验归档 SHA256、tar path/type 安全、fixture header/hash、104 条 manifest command、实际 test count、64 个 HTTP contract/resolved plan 和八项 decision。
- 2026-08-14：额外复算 both 模式，确认 TP1 decode 平均 `+2.35%`，但按预注册独立规则不用于本轮放行。

## Debrief

- **Outcome**：当前 HEAD 的 SM89 证据完整且 correctness 全绿；默认策略明确保持 split，候选实现可进入 PR 收敛。
- **Pitfalls encountered**：tracked fixture 在 pull 后恢复为旧 q/v-only；runner 早期在生成前 assert，现已改为隔离生成。Jupyter 还向结果目录注入 checkpoint 副本，虽未污染 manifest，但应从后续归档排除。
- **Lessons learned**：性能规则必须在看数据前固定；组合协同收益可以成为新候选，但不能用来事后推翻独立资格规则。硬件差异已经实证出现，Auto policy 必须表达适用边界。
- **Follow-ups**：提交本轮五 target fixture并跑 fresh-checkout LoRA gate；是否追求 TP1 decode both 的 `2.35%`，作为独立后续优化决定。
