# 当前代码功能与训练方式严格核查（2026-09-18）

## 1. 核查范围与版本

主工作目录是 `inception-qwen3-4b`。本次核查了参数定义、PEFT 冻结逻辑、完整 loss 调用链、阶段启动/恢复、4B 实验生成器、推理模型加载和计时终止条件，并修复下述问题。

其他目录的角色不同：

|目录|用途|与当前 4B 版本的关系|
|---|---|---|
|`inception`|原始/历史 RecurFT 实现|用于核对原始训练流程|
|`inception-joint`|8B 冻结目标联合训练及历史运行|运行中的源快照独立，未修改|
|`inception-publish-20260918`|此前提交 PR #1 的发布快照|包含冻结目标联合训练和混合推理，不自动包含当前新改动|
|`inception-inference-opt`|推理开发分支|另有 GSM8K/MATH500、可配置验证阈值等实验，未合并到本次 4B 矩阵|
|`inception-qwen3-4b`|本次三组、三 seed 训练与严格同目标推理|后续 4B 实验使用这里的审计版快照|

**不能把这些目录当成完全相同的代码版本。** 本报告对主目录执行了回归与真实模型检查；其他目录仅核对相关训练/推理入口和版本边界，不宣称其所有实验功能都通过本次检查。

推荐的新实验目录：`runs/qwen3_4b_matrix_20260918_v2_audited`。v1 保持原样用于追溯，未启动长训练；v2 包含本次修复。现有 8B 运行、历史结果和旧源快照均未被覆盖。

### 功能总览

|功能|当前实现|
|---|---|
|训练|原始分阶段、冻结目标的 T＋头联合、新增目标 LoRA＋T＋头联合|
|数据与实验管理|MetaMath 训练/验证隔离、三 seed 计划、阶段依赖与预算生成|
|验证与诊断|训练/验证 loss 分项、梯度范围、冻结权重对比、训练曲线分析|
|断点与续训|safetensors/JSON 保存 optimizer、scheduler、RNG；阶段转换与精确恢复分开|
|推理与测速|同一目标的 greedy、T、查找和混合对照；token 数、质量、计时与三 seed 汇总|
|可复核性|源快照、配置/数据 manifests、训练契约、adapter 张量与行为配置核对|

下文核查的是这些 RecurFT 路径，不代表对所附 LLaMA-Factory 框架中所有其他训练算法作全面认证。

## 2. 先把四种参数分清

- **原始基座参数 W₀**：原始 Qwen 权重。这里所有 RecurFT 方案都冻结它们，没有全参数微调。
- **目标 LoRA ΔW**：加在目标模型上的适配器。实际目标模型是 `F(W₀, ΔW)`。冻结 W₀ **不等于** 冻结整个目标模型；ΔW 更新时目标输出仍会变化。
- **递归 T 的 LoRA**：T 从指定目标层的原始权重复制，复制的基础层被冻结，只更新 T 自己的 LoRA。
- **boundary 输出头 H**：LayerNorm + 低秩残差映射，再使用目标的最终 norm 和词表投影。它不是原始 `lm_head` 的全矩阵微调。4B 的 embedding/lm_head 共享原始权重，保持冻结。

模型模块还保留 token conditioner、后续步 residual、首步 residual、verifier probe/adapter 等研究功能。**本次 4B 三组不开启这些附加训练分支**；verifier 参数即使存在，也默认冻结，不能计入正在学习的 T/头参数。

## 3. 训练方式对照：名称相似，含义不同

|方式|起点|更新参数|冻结参数|目标模型是否变化|
|---|---|---|---|---|
|普通 core|原始基座|目标 LoRA + T LoRA|W₀；此时没有 boundary 头|变化|
|普通 multistep|core 检查点|目标 LoRA + T LoRA|W₀|变化|
|boundary warmup / boundary|multistep / warmup 检查点|仅 boundary 头|W₀、目标 LoRA、T|不变|
|旧版从基座联合|原始基座，新 T/头|T LoRA + boundary 头|W₀、目标 LoRA|不变，目标函数等同原始基座|
|检查点后冻结目标联合|已有完成头训练的检查点|T LoRA + boundary 头|W₀、**已经训练过的目标 LoRA**|不变，目标是训练后的目标|
|新版可训练目标联合|4B 原始基座，新 T/头|目标 LoRA + T LoRA + boundary 头|W₀|变化|

**“不联合”指不联合训练 T 与 boundary 头。** core/multistep 本来就同时训练目标 LoRA 与 T。另一个关键区别是：这些配方的目标模型 SFT CE 权重为 **0**，所以“普通训练”在这里指原始 RecurFT 分阶段方案，不能理解成常规 next-token SFT。

“从头联合”必须带限定词：

- `8B frozen joint from base`：从第 1 步训练 T＋头，目标始终冻结。
- `4B full_joint / trainable_target_joint`：从第 1 步开放目标 LoRA＋T＋头。

`recurft_joint_mode: legacy` 是兼容旧 loss 调用的标记，**不是“冻结”的同义词**。是否冻结还取决于 `recurft_stage1_heads_only`、`recurft_recurrent_trainable_only` 和运行期实际 `requires_grad`。v2 用 `training_regime` 把这些信息明确写入计划、运行记录和检查点。

## 4. 4B 的三组与实际模块更新次数

每组 seeds 为 **42、43、44**。A/B 共享同 seed 的普通训练检查点，A 只执行一次。

|组|执行顺序|累计优化器更新|目标 LoRA 更新|T 更新|头更新|
|---|---|---:|---:|---:|---:|
|A `no_joint`|core → multistep → head_warmup → head|60,023|59,039|59,039|984|
|B `post_joint`|继承 A，再冻结目标联合 T＋头|64,023|59,039|63,039|4,984|
|C `full_joint`|从原始基座联合目标 LoRA＋T＋头|64,023|64,023|64,023|64,023|

A 的四段分别为 **49,199 / 9,840 / 222 / 762** 步。B 续训 **4,000** 步，保留 2,000 步中间点。C 额外保存并评测 A 等更新预算的 **60,023** 步检查点。

这些是计划的优化器更新次数。“开放训练”不表示每个 tensor 每步梯度都非零：LoRA 的零初始化、样本 mask 等可能使部分 tensor 当步梯度为零。检查时同时看参数开放范围、实际分组梯度和权重变化。

### 能得出的比较结论

- B 对 A：同一个训练后目标，额外进行 T＋头联合优化后的变化；同时增加了 4,000 步训练预算。
- C 对 B：累计更新数相同的两套训练方案；目标更新次数、头更新次数、阶段学习率和损失历程不同。
- C@60,023 对 A：相同更新预算的方案比较，但模块更新次数仍不同。

**这不是只改变一个“联合开关”的单变量消融。** 相同步数不等于相同 FLOPs、wall time、独立训练样本数或 token 数；不同阶段会重新建立优化器和 sampler，数据也可能重复。不能据此单独归因“联合训练使速度提高 X%”。

## 5. Loss 到底训练了什么

固定 reference 默认是**关闭目标 adapter 后的原始基座**。4B 没有配置另外的 reference 模型。

|4B 阶段|目标对原始基座的约束|一步 T 损失|多步损失|直接 boundary 损失|
|---|---|---|---|---|
|core|hidden MSE×20 + relative MSE×1 + cosine loss×1 + KL×0.2|MSE + 0.05 relative MSE + 0.05 cosine loss；前 200 步预热|关闭|无头|
|multistep|同 core|同 core，权重已全开|四步，外部权重 0.0005；Huber β=0.1 + delta-cosine×0.02；前 1,000 步预热|无头|
|head_warmup/head|权重为 0|仍计算并计入 total，但目标/T 已冻结，是常数项|关闭|数据 CE×1 + 当前目标 KL×0.2|
|B 冻结目标联合|权重为 0|上述一步项整体×0.1|两步，外部权重 0.1；内部 Huber + boundary KL×10；前 200 步预热|关闭|
|C 可训练目标联合|同 core|同 B|同 B|关闭|

因此 B/C 预热后有效 rollout boundary KL 系数是 **0.1 × 10 = 1**；不是直接把 total 加上 10 倍 KL。旧 8B 冻结联合配方的一步 relative/cosine 辅助系数为 0，和当前 4B 的 0.05 不同。

### Teacher 与梯度路径

- 一步 T：`T(h_t)` 预测当前目标的 `h_(t+1)`，标签 detach，输入 `h_t` 保留梯度。C 中这条路径同时更新目标前段 LoRA 和 T。
- rollout：用预测 hidden 接着运行 T。boundary KL 的 teacher **始终取当前目标 logits 并 detach**，不是原始 reference logits。
- 直接 head 训练：用真实目标 hidden 训练 H；当前 4B 将直接 KL teacher 设为当前训练后目标。历史原始 head 配方默认取原始 reference，二者有明确区别。
- `detach_rollout=true` 只截断预测 hidden 的跨步梯度链，不等于冻结 T。普通 multistep 使用 true；B/C 联合使用 false。
- C 中目标前段 LoRA 接收 T/rollout 路径梯度；目标后段 LoRA 主要由最终 hidden/KL 约束更新。不能把“目标组有梯度”说成目标的每一个 LoRA tensor 从第 1 步都有非零梯度。

4B 的 `loss_on_labels_only=true` 用于 token/递归损失的 label mask；最终 hidden 对齐仍使用 attention mask，即包含未 padding 的 prompt 和 response。历史原始配置是 `loss_on_labels_only=false`。

### 曲线解读约束

head-only 阶段的 total 含被冻结的 recurrent 常数损失，不能把 total 全部理解为输出头正在优化的损失。跨阶段、跨 B/C 的损失项与权重也不同。应分别比较同一阶段的 head CE/KL、draft KL、recurrent loss，以及固定验证集指标。旧 +2,000 步训练的局部平台不能代替验证集收敛证明。

## 6. 原始流程与 4B 适配的边界

保留：分阶段训练范围、目标基座约束、四步 multistep、128-token rollout context、0.7 step decay、Huber、delta-cosine、head-only 训练顺序。

4B 的明确调整：

- 36 层，hidden 2560，FFN 9728，head_dim 128；attention Q 宽度为 4096。
- T 层为 33–34（从 0 计数），T rank/alpha=80/160，boundary rank=160。
- T 可训练参数 9,175,040；boundary 824,320；目标 LoRA 16,515,072。
- 全部阶段长度 1024，有效 batch=8（micro=2、累积=4）。历史长度、head 阶段 batch 不完全相同。
- 固定 held-out query split，按实际训练行数计算预算；标签 mask 与直接 head teacher 按本次方案调整。
- 新 runner 阶段间只继承权重、重新初始化 optimizer，步骤从 0 开始；阶段内暂停测速后完整恢复 optimizer/scheduler/RNG。

因此这是**以原始流程为基线的 4B 适配实验**，不是历史 YAML 的逐字复现，也不是旧 8B 结果的直接复现。

## 7. 推理功能与训练方式是两个维度

|推理路径|草稿来源/控制|主要入口|
|---|---|---|
|greedy|训练后目标逐 token argmax|各 benchmark 的配对 baseline|
|T + tail|T 预测 hidden，目标后段读出|`scripts/run_decode.py --route tail`|
|T + boundary|T 预测 hidden，轻量头读出|`scripts/run_decode.py --route boundary`；compact 路径|
|旧 adaptive / 延后启动|margin、冷却、位置等规则控制 T 使用|`benchmark_decode_adaptive.sh`、`benchmark_decode_start.sh`、two-stage 入口|
|lookup-only|在 prompt/已生成文本中查找重复片段，不用 T 生成草稿|`lookup_decode.py`|
|hybrid|优先 lookup，未命中时使用 T|`benchmark_hybrid.sh`、4B 配对 benchmark|
|宽松验证实验|允许阈值小于 1，可能接受非 argmax 草稿|另一个 `inception-inference-opt` 目录，不属于本次 4B 默认矩阵|

**当前 4B 默认不是旧 adaptive 或“后期才启动 T”。** 它使用固定 neural block=3（一个已知目标 token＋最多两个 T 草稿），另测 hybrid 与 lookup-only。查找块可有自己的草稿长度。严格验证阈值为 λ=1。

`benchmark_trained_target.py --checkpoint CKPT` 将 greedy 和投机验证绑定到同一个训练后目标；实际加载 CKPT 的目标 LoRA，再由所有方法共用同一个模型对象。前后检查点比较要求目标 adapter 张量相同，且 rank/alpha 等行为配置相同。目标 LoRA 合并、RMSNorm、attention 优化也用于对应 greedy。

基座以显式本地模型目录加载，协议 manifest 核对其 config；当前没有在每次启动时逐分片校验全部基座权重。adapter 核对加共享目标对象保证本次 greedy/投机使用同一目标，不等于对任意跨机器同名基座的权重身份作保证。

报告 wall ratio、token/s、输出 token 数、token 一致率、数值答案正确率和 cap 命中数。BF16 block forward 与串行 greedy 可能有数值分歧；λ=1 不等于所有输出必然逐 token 相同。Hybrid 的收益包含查找贡献，不能据此声称 T 单独达到相同加速。

## 8. 本次确认并修复的问题

|问题|修复/验证|
|---|---|
|可训练目标模式可关闭 recurrent/rollout/head loss，名义联合却未训练头|配置阶段拒绝不具备完整训练路径的组合；补负向测试|
|4B v1 漏掉原基线 core 的 200 步 recurrent 预热、multistep delta-cosine=0.02|在 v2 明确补回；v1 原样保留|
|旧冻结联合入口可能继承 `trainable_target` 模式|为新建续训阶段选择冻结配方；来源检查点与可训练目标入口保持独立|
|旧入口只拒绝 pickle optimizer，未识别新 safetensors optimizer|拒绝新旧 optimizer 状态跨训练范围继承；只接受模型权重转阶段|
|恢复训练只检查 optimizer 步数，未核对目标函数/数据是否改变|新检查点保存 training contract；严格模式拒绝 seed、预算、loss、scope 或数据身份变化|
|准备后修改 YAML/数据不在旧 source manifest 检查范围|v2 增加 protocol manifest；包括 dry-run 在内都核对代码、计划、配置、数据和模型 config|
|阶段初始化只核对 tensor 布局，未核对 T 层号/LoRA scaling|严格阶段初始化增加层号、rank、alpha 和 head 添加规则检查|
|单目标推理入口能转发 `--bef`/`--aft` 缩写|禁用参数缩写并拒绝前后目标覆盖|
|“相同 adapter 张量”未覆盖 alpha 等配置差异|对比行为相关 adapter 配置；加载后仍逐 tensor 核对|
|greedy 在命中 token cap 后多算一次目标 forward|三个 greedy 实现修正停止条件；用真实调用计数验证 N-token cap 对应 N 次调用（含 prefill）|
|旧 8B 从头冻结联合准备器能接收 4B 路径并套 8B 配置|加入 Qwen3-8B 架构检查，4B 必须走 4B 入口|

修复 token-cap 问题后，之前触及 cap 的测速不能与新计时口径直接合并。历史结果保留原始记录；不会用修复后的逻辑回填旧耗时。

## 9. 可复核的文件与检查

- [分阶段配置生成器](../local_setup/qwen3_4b_recipe.py)
- [模式与恢复契约](../local_setup/training_contract.py)
- [实验计划审计](../local_setup/matrix_audit.py)
- [模型冻结实现](../LLaMA-Factory/src/llamafactory/model/adapter.py)
- [实际 loss 实现](../LLaMA-Factory/src/llamafactory/train/sft/recurft.py)
- [可训练目标模式](../LLaMA-Factory/src/llamafactory/train/sft/trainable_target_joint.py)
- [完整解析配置与 source hashes](../reports/code_audit_20260918/resolved_training_modes.json)
- [核查验证记录](../reports/code_audit_20260918/validation.json)

重新核查准备好的实验（不启动训练）：

```bash
python local_setup/audit_training_modes.py \
  --suite runs/qwen3_4b_matrix_20260918_v2_audited \
  --output runs/resolved_training_modes.json

PYTHON=/path/to/inception/python \
  bash runs/qwen3_4b_matrix_20260918_v2_audited/run_all.sh --gpus 0,1,2 --dry-run
```

本次执行真实 4B 短程训练和保存/测速/恢复链路检查，证明实现与参数范围符合预期；它们不证明收敛、最终准确率或三 seed 加速结果。**9 个完整长训练仍未启动。**
