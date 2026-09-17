# 本机 1 个实验 + 远端 8 卡任务

本机保留 **A `no_joint/seed_42`**。远端一次提交八个独立单卡进程，不使用 DDP；训练配方、三 seed、A/B 同 seed 共享 checkpoint 和配对 greedy 协议保持不变。

## 当前平台直接提交的脚本

与 `Exploration-my/scripts/train/qwem2.5-1.5b/run4_ppo5.sh` 相同形式的入口是：

```bash
bash /mnt/llmshared-ssd-hd/wangruitao/inception-qwen3-4b/scripts/train/qwen3-4b/run8_qwen3_4b.sh
```

这个 `.sh` 已写好 conda 环境、模型/数据和输出目录，显式启动八个后台任务，分别设置 `CUDA_VISIBLE_DEVICES`、写独立日志，最后逐个 `wait` 并报告失败。无需额外的 `.env` 文件。默认 GPU 0–7；平台提供八个可见设备时沿用分配顺序。

该入口针对挂载同一共享盘的八卡自动化任务，本机 A42 的 checkpoint 直接从共享盘读取，不需要 SSH 传输。主日志位于 `inception-qwen3-4b/logs/qwen3_4b_remote8_<时间>_<PID>/`，实验位于 `runs/qwen3_4b_remote8_20260918`。附加 `--dry-run` 可在当前机器检查全部八条启动命令，不启动训练。

下文的配置文件和 SSH 入口用于路径不同或不共享文件系统的其他主机。

## 任务分配

默认远端物理 GPU 编号如下，可通过 `GPUS` 调整顺序：

|GPU|实验|启动条件|
|---|---|---|
|0|B `post_joint/seed_42`|等本机 A42 完成并传入 checkpoint|
|1|C `full_joint/seed_42`|GPU 空闲即可开始|
|2|A `no_joint/seed_43`|GPU 空闲即可开始|
|3|B `post_joint/seed_43`|等同一远端 suite 的 A43 完成|
|4|C `full_joint/seed_43`|GPU 空闲即可开始|
|5|A `no_joint/seed_44`|GPU 空闲即可开始|
|6|B `post_joint/seed_44`|等同一远端 suite 的 A44 完成|
|7|C `full_joint/seed_44`|GPU 空闲即可开始|

八个任务同时提交。没有现成 checkpoint 时，初期是 **五卡训练、三卡等待依赖**，不能保证八卡始终满载。B 继续使用对应 A 的实际 checkpoint，不重跑一份 A 来填满 GPU。每个任务复用自己的卡进行验证和推理测速。

## 环境与输入

- 单台 Linux 主机有八张可见 NVIDIA GPU，每张至少 30,000 MiB 空闲显存。脚本默认等待独占空闲；本机旧任务不会被自动停止。
- 使用已安装本仓库依赖的 Python 3.11 环境。已验证的模型环境是 PyTorch 2.4.1+cu121、Transformers 4.57.1、PEFT 0.17.1；依赖锁文件随代码提供。八进程调度另有 CPU 测试，尚未在真实八卡主机上执行这组长训练。
- 本地 Qwen3-4B 模型目录、`MetaMathQA-valid.json` 和 GSM8K `test_official.jsonl`。模型和数据不在代码包中；启动时不会自动下载。
- 使用持久化共享盘作为 `WORK_ROOT`。训练输出、恢复状态、缓存和临时文件都保存在该目录下。
- 自动传输使用 OpenSSH 的 `ssh` 和 `scp`，不要求 rsync。远端需提前配置到本机的 SSH 密钥和 known_hosts；脚本不会弹密码提示。

## 远端一条启动命令

将代码包解压到远端，或 checkout 含本次自动化入口的提交。复制并填写配置：

```bash
cp automation/qwen3_4b_8gpu.env.example /path/to/job.env
```

至少设置 `PYTHON`、`MODEL`、`TRAIN_DATA`、`TEST_DATA`、`WORK_ROOT`。保持 `LOCAL_EXPERIMENT=no_joint/seed_42`。`GPUS` 默认为 `0,1,2,3,4,5,6,7`，也接受完整 GPU UUID；如果平台设置了 `CUDA_VISIBLE_DEVICES`，所选卡必须位于该分配中。

先做准备与检查，不启动 GPU 任务：

```bash
bash automation/qwen3_4b_8gpu.sh /path/to/job.env --dry-run
```

自动化平台的启动命令：

```bash
bash automation/qwen3_4b_8gpu.sh /path/to/job.env
```

通过 SSH 手动后台启动可用：

```bash
nohup bash automation/qwen3_4b_8gpu.sh /path/to/job.env > /path/to/job.log 2>&1 &
```

入口先校验代码包，再在 `WORK_ROOT/suite` 生成模型专用配置、数据划分及不可变源快照，然后启动八个 worker。重复执行沿用同一 suite 的检查点，不覆盖已完成实验；同一个 `WORK_ROOT` 的并发启动会被锁拒绝。配置、模型或数据若要修改，应使用新的 `WORK_ROOT`，不能更改既有 suite。

## 本机 checkpoint 自动交接

本机已安排的源 suite：

`runs/qwen3_4b_matrix_20260918_v2_audited`

本机 A42 全部完成后，导出器将原子发布：

```text
dependency_exports/no_joint_seed_42/
  READY.json
  checkpoint/
    adapter_model.safetensors
    recurft_recurrent.safetensors
    adapter_config.json
    recurft_config.json
    training_contract.json
    ...
```

### 自动 SSH 拉取

在远端 `job.env` 设置：

```bash
DEPENDENCY_SOURCE=user@local-training-host:/mnt/llmshared-ssd-hd/wangruitao/inception-qwen3-4b/runs/qwen3_4b_matrix_20260918_v2_audited/dependency_exports
```

只需替换 SSH 用户/地址及实际路径。B42 每 30 秒检查一次完成标记，文件就绪后通过 scp 拉取至临时目录，校验通过后才导入。其余七个 worker 不等待这次传输。连接失败或文件尚未就绪会记录原因并重试。

### 共享盘或手动复制

如果两端能访问相同的导出目录，删除 `DEPENDENCY_SOURCE`，设置 `DEPENDENCY_DIR=/shared/dependency_exports`。也可把完整 `no_joint_seed_42` 目录复制到远端 `WORK_ROOT/incoming`；复制时先使用临时目录，完成后重命名，避免提前暴露 READY 标记。

导入时核对 seed、四阶段训练配置、训练/验证数据内容、模型 config、训练实现以及每个文件的 SHA256。路径可因机器不同而改变。只转移模型权重；B 新建优化器，绝不继承 A 的优化器状态。基座权重由两端提供相同 Qwen3-4B 资产，交接指纹不重新哈希全部基座权重。

A42 的原始 loss、推理结果和训练日志仍在本机。远端 suite 中导入的 A42 只用于 B42 初始化；最终九组统计还需要汇总本机 A42 的评测记录。

## 状态、失败与恢复

- `suite/worker_launches/<时间>/status.json`：八个 PID、分配、日志和退出码。
- `suite/<arm>/seed_<seed>/worker.json`：依赖等待、训练/评测、发布或失败状态。
- 同目录的 `status.json`：原训练调度器状态，包括实际训练子进程和 GPU 等待。
- `waiting_for_dependency` 与 `waiting_for_gpu` 都表示尚未进行该阶段的训练；不能当作已训练步数。
- 某个 A 失败，对应 B 会报告依赖失败；其他独立实验继续。协调进程等待所有 worker 退出后，以非零码报告失败。
- 给主协调进程发送 TERM/INT 会停止它管理的 worker 及训练/评测子进程，不触碰其他任务。最近一次完整 optimizer/scheduler/RNG checkpoint 可用于恢复。首次 checkpoint 前中断需检查阶段记录，不能保证无状态的崩溃自动恢复。

直接运行单个本机实验并在结束时发布 checkpoint 的入口为：

```bash
python local_setup/run_qwen3_4b_workers.py \
  --suite /path/to/local_suite --only no_joint/seed_42 --gpus 2 \
  --publish-dependencies /path/to/local_suite/dependency_exports
```

`--dry-run` 会检查所有配置和任务依赖，但不会要求当前主机实际具备八卡，也不会创建训练进程。
