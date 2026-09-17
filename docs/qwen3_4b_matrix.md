# Qwen3-4B：原始分阶段基线与两种联合训练

这是独立实验目录和独立的可训练目标联合模式。原有冻结目标的联合训练配置、入口和运行中的 8B 实验保持原样。这里只训练 LoRA、递归 T 的 LoRA 和轻量 boundary 输出头，**不更新原始 4B 参数**。boundary 输出头也不是原始词表矩阵。

## 三组 × seeds 42、43、44

|组|训练顺序|目标 LoRA 是否更新|T / boundary 头|
|---|---|---|---|
|A `no_joint`|core → multistep → head_warmup → head|前两阶段更新，后两阶段冻结|前两阶段训练 T；后两阶段只训练 head|
|B `post_joint`|从同 seed 的 A 最终检查点继续联合训练 4,000 步|冻结 **A 训练后的** 目标 LoRA|联合训练 T + head|
|C `full_joint`|从原始 Qwen3-4B 第 1 步联合训练|全程更新目标 LoRA|全程训练 T + head|

A/B 共享同 seed 的完整普通训练结果，不重复跑 A。B 是原有冻结目标联合方法在已训练目标上的应用；C 使用新字段 `recurft_joint_mode: trainable_target`，独立于默认 `legacy` 模式。C 的总更新数与 A + B 续训相同，还保存 A 更新预算对应的检查点并单独测速。更新数相同不代表 FLOPs 或训练时间相同；A 与 B 最终结果的训练预算不同。

## Loss 与梯度

- 更新目标 LoRA 时，原始无 adapter 基座作为固定 reference：hidden MSE ×20、relative MSE ×1、cosine loss ×1、KL ×0.2，限制目标漂移。原始流程的 SFT CE 权重仍为 0。
- T 的一步 hidden 标签来自**当前训练后目标**，标签 `detach`；输入特征保留梯度，让目标与 T 协同更新。
- 这组 4B 实验所有 boundary KL 的 teacher 均为**当前训练后目标的 detached logits**。原来的 `reference` teacher 默认值未改。
- 联合训练：recurrent MSE ×0.1，rollout loss ×0.1，内部 boundary KL ×10，rollout 两步，前 200 次更新渐增 rollout 权重。从第 1 次更新就有 head 梯度。
- core 保留原始 200 步 recurrent 权重预热。普通 multistep 阶段保留原始 delta-cosine=0.02、四步 rollout、128-token context、0.7 decay、detached rollout 和 1,000 步权重预热。
- 每次启动核对预期 trainable groups；前 3 次更新记录分组梯度，保存后核对冻结目标权重。head-only 阶段额外核对 T 未变。
- 训练与验证分别累计 loss 分项，验证不会混入训练曲线。不同阶段的目标项和权重不同，总 loss 不能直接跨阶段比较。joint 每 500 步在固定 held-out MetaMath 上验证，普通阶段每 1,000 步验证，阶段结束也验证。

## 根据 4B 调整

读取并核对本地 config 和 safetensors header，要求与 [Qwen 官方 4B 配置](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) 一致。

|项目|4B 配置/本实验|
|---|---|
|hidden / FFN|2560 / 9728|
|层数|36；T 复制第 33、34 层（从 0 计数）|
|attention|32 Q heads、8 KV heads、head_dim=128；Q 宽度 **4096**，不是 hidden_size|
|词表矩阵|embedding 与 lm_head 共享权重，两者均冻结，运行时校验共享指针|
|目标 LoRA rank|8|
|T rank / alpha|80 / 160；9,175,040 个可训练参数|
|boundary rank|160；824,320 个可训练参数|
|长度 / batch|所有阶段 cutoff=1024；micro batch=2、累积=4，有效 batch=8|
|学习率|core 1e-5；multistep/joint 3e-6；head warmup 1e-4；head 5e-5|

真实 4B 短程检查中，各阶段峰值 allocated 为 20.5–22.6 GiB，最高 reserved 为 23.9 GiB；配置要求至少 30,000 MiB 空闲，留出 CUDA 上下文和分配波动余量。这是长度 1024、batch=2 的检查结果，长时间运行仍记录峰值。

rank 是按此前 r/hidden_size 比例缩放并对齐到 16 的初始容量方案，学习率是待验证的初始值，均不声称已调优。两种模型都有 36 层，所以 T 层号相同有架构依据。

MetaMath 按规范化 query 分组，以固定 salt 的哈希选择 512 个验证 query，并排除训练集中同 query 的全部重复项。三 seed 共用同一个 split。预算按实际剩余训练行数计算：core 1 epoch、multistep 0.2 epoch、head 总计 0.02 epoch；head 预算的 22.5% 用于 warmup。token 截断率及有效训练 token 数应从实际训练数据统计，更新数不是 token 数。

## 推理对照

独立入口：`local_setup/benchmark_trained_target.py --checkpoint CKPT ...`。它只接受一个检查点，将 greedy 与投机路径绑定到同一份训练后目标 adapter。加载后逐 tensor 核对 checkpoint 权重；然后两条路径共用同一个目标模型对象和相同 target 优化配置。`manifest.json` 与 `load_audit.json` 记录路径、权重 SHA256 和检查结果。

每个检查点测试：greedy、T-only、T + n-gram 混合、lookup-only。后两项分别展示混合方法效果及查找贡献。保持 target_match 严格验证，BF16 的块前向可能与串行 greedy 产生数值差异，因此同时报告 token 数、逐 token 一致率、截断率、数值答案正确率。

- 周期监控：held-out MetaMath 中可提取数值答案的固定前 8 题，1 次，256-token 上限。包含符号答案的题仍用于验证 loss，但不用于这个数值正确率。
- 最终端点：GSM8K test 的 0–63 题，2 次，1024-token 上限。C 在预先确定的 A 等预算端点额外测一次。
- 排除加载、编译、预热；包括每题 prefill 与生成。逐题轮换方法顺序。
- wall speedup = 该检查点 greedy 总时间 / 投机总时间。另报 token/s 比，避免输出长度变化被误认为纯吞吐提升。
- `summarize_qwen3_4b_matrix.py` 先按每 seed 自己的 greedy 求比，再报告三个 seed 的均值和样本标准差。重复推理不当作独立 seed。

## 准备与运行

本机保留 `no_joint/seed_42`、其余八组在一台八卡 Linux 主机运行时，使用 [八卡自动化入口与 checkpoint 交接说明](qwen3_4b_eight_gpu.md)。它采用每实验一个 worker；下文原入口采用每 seed 一个 worker。

使用已经安装依赖的 inception Python 环境。准备脚本只生成数据划分、18 份阶段配置、9 个实验计划及源代码快照，不启动长训练：

```bash
python local_setup/prepare_qwen3_4b_matrix.py \
  --model /path/to/Qwen3-4B \
  --train-data /path/to/MetaMathQA-valid.json \
  --test-data /path/to/GSM8K/test_official.jsonl \
  --output /path/to/new_suite

bash /path/to/new_suite/run_all.sh --gpus 0,1,2 --dry-run
bash /path/to/new_suite/run_all.sh --gpus 0,1,2
```

每 GPU 一个 worker，按 seed 执行 A → B → C；GPU 不够时可只写 `--gpus 2`，或 `--seeds 42`。也可分别执行 `run_no_joint.sh`、`run_post_joint.sh`、`run_full_joint.sh`。B 会检查 A 已完成。默认等待 GPU 空闲；`--allow-shared-gpu` 显式允许共享并记录实际 GPU 状态，共享测速需要之后在独占环境复核。

运行前（包括 dry-run）检查 source 和 protocol manifests，覆盖代码、计划、YAML、数据及模型 config；请通过重新准备新的 suite 更新代码，不要修改已准备的源快照。重复启动会跳过完整结果，在阶段内部从 optimizer/scheduler/RNG 完整状态续训；阶段转换只继承模型权重，使用新的 optimizer；层号、rank、alpha 必须兼容。同阶段恢复必须通过 training contract 核对，不能改变 loss、训练范围、seed、预算或数据。到预设测速点会保存、退出训练释放 GPU、测速，再恢复训练。缺失旧 milestone 时拒绝用更晚检查点冒充。

```bash
python /path/to/new_suite/source_snapshot/local_setup/summarize_qwen3_4b_matrix.py \
  --suite /path/to/new_suite
```

短程真实模型检查入口 `local_setup/smoke_qwen3_4b.py` 会在独立输出目录各训练 2 步，验证五个阶段/模式的梯度、1024-token 显存、冻结权重、held-out loss 及同目标推理；它不用于判断收敛或报告速度提升。

## 已有 2,000 步是否够

[8B 完整普通训练后 +2,000 步分析](../reports/full60375_joint2000_plateau/REPORT.md)：最后三个 200-step 窗口 total loss 为 3.98715、3.96448、3.98489，波动范围约 0.57%，已接近训练集局部平台；同期 recurrent MSE 从 4.11757 升至 4.18362。旧运行没有 held-out loss，不能据此认定收敛，也不能直接外推到 4B。

因此 B 默认预留 4,000 步，保留 2,000 和每 500 步检查点。比较固定验证集 loss、draft KL/top-1、接受率及测速趋势；不按 GSM8K test 结果选择停止点，也不自动把 4,000 步当作收敛。

## 本次已准备的运行目录

`runs/qwen3_4b_matrix_20260918_v2_audited`：393,589 条训练样本、512 个 held-out query，另外排除 895 条验证 query 的重复样本。输入使用已有 `MetaMathQA-valid.json`（394,996 条）；原始 395,000 条文件有 4 条空样本，准备脚本会拒绝空样本。

每 seed：core **49,199**、multistep **9,840**、head warmup **222**、head **762**，共 **60,023** 次更新。B 接 **4,000** 步，C 从头 **64,023** 步；2,000 步中间检查点保留。

```bash
export PYTHON=/mnt/llmshared-ssd-hd/wangruitao/conda-envs/inception/bin/python
bash runs/qwen3_4b_matrix_20260918_v2_audited/run_all.sh --gpus 0,1,2 --dry-run
# GPU 空闲后，去掉 --dry-run 启动；共享环境需显式加 --allow-shared-gpu。
```

当前只完成脚本、配置、短程验证；这 9 个完整训练尚未启动。

完整功能、训练模块更新次数、loss 梯度路径与修复记录见 [严格核查报告](code_and_training_audit.md)。v1 原样保留用于追溯，后续使用上述 v2 审计版。
