# LayerLoop / RecurFT Executable Code Export

## Joint training and hybrid inference (2026-09-18)

The new [run guide](local_setup/README.md) covers frozen-target T + boundary
joint training, safetensors/JSON checkpoint resume, and prompt/history lookup
with neural T fallback. The [joint recipe](configs/experimental/qwen3_joint_from_base.yaml)
and [hybrid benchmark](local_setup/benchmark_hybrid.sh) use explicit local asset
paths. BF16 block verification has observed differences from serial greedy;
report token equality and answer quality alongside wall time and token/s.
See the [22,000-step retest](reports/joint_hybrid_20260918.md) and
[validation record](reports/joint_hybrid_validation_20260918.json).


完整的修改版 LLaMA-Factory 源码、递归训练实现和可运行入口。不是仅包含几个补丁的补充材料，也不需要重新拉取上游仓库。此包是 2026-09-12 工作区快照，不声称恢复了七月逐字节一致的代码和环境。

## 先明确执行路径

| 入口 route | 实际草稿路径 | 定位 |
| --- | --- | --- |
| `boundary`，默认 | 隐状态 -> 递归 T -> 低秩残差适配 -> final norm + LM head -> token | 真正的递归方法；384 样本受控实验采用此读出 |
| `tail` | 隐状态 -> 递归 T -> 边界后剩余 Transformer 层 -> norm + LM head -> token | 真正的递归方法；较贵的原始读出 |
| `ngram` | 文本历史查找 -> token -> 目标模型验证 | 明确命名的非递归对照，不作为 recurrent 收益 |

两条神经草稿入口均显式设置 `--ngram-draft-mode off` 和严格 target-match，不开放 unchecked commit、宽松接受或自动切到 n-gram 的参数透传。

2026-09-11 审计已确认：旧跨模型图的 24 个加速点全部是 n-gram-only，另 3 格是 greedy fallback，不能用其 1.295x 支持递归加速。真正的 N=384 递归受控实验使用 boundary，实测约 1.040x wall / 1.288x target-call reduction。详见 `provenance/execution_route_audit.json`。本代码包没有改写或重标那些结果。

## 内容

```text
LLaMA-Factory/src/                         完整框架源代码，含模型、LoRA、trainer、loss
LLaMA-Factory/experiments/recurft_math/    解码、rollout、数值审计及其直接依赖
LLaMA-Factory/tests/                      递归、损失、冻结参考与草稿相关单元测试
scripts/run_decode.py                    安全检查 + 显式路由 + 结果审计
scripts/benchmark_decode.sh              集中改参数、检查训练步数、重复测速并生成耗时简报
scripts/reproduce_headline.sh             N=384 / 两遍 / FP16 / K=3 协议入口
scripts/train.py                         四阶段训练入口，生成路径可移植的 YAML
scripts/self_test.py                     离线 CPU 单元与真实小模型端到端测试
scripts/check_assets.py                  checkpoint 文件和 SHA256 检查
scripts/fetch_checkpoint.py              可选：从自己的服务器取回推理 checkpoint
configs/reference/                      历史超参数配置
requirements-linux-cu121.lock.txt        当前服务器实际环境的可移植版本清单
provenance/                              代码来源、审计、历史指标、资产清单
MANIFEST.sha256                          全部交付文件校验和
```

核心文件：模型模块 `src/llamafactory/model/model_utils/recurft.py`；训练损失 `src/llamafactory/train/sft/recurft.py`；参数与加载逻辑 `hparams/finetuning_args.py`、`model/adapter.py`；训练入口 `src/train.py`；推理入口 `experiments/recurft_math/recurft_speculative_generate.py`。

## 环境和离线自测

正式运行目标为 Linux x86_64 / Python 3.11 / NVIDIA CUDA。当前服务器实测环境为 PyTorch 2.4.1+cu121、Transformers 4.57.1、PEFT 0.17.1。依赖清单来自 2026-09-12 的现有环境，不代表七月原始环境快照。

在已有 `otv311` 环境上，不必重装模型依赖。若没有 pytest，用继承现有包的独立测试环境，避免修改原环境：

```bash
source scripts/server_paths.example.sh
"$PYTHON" scripts/verify_package.py
"$PYTHON" -m venv --system-site-packages /tmp/layerloop-test-env
/tmp/layerloop-test-env/bin/python -m pip install -r requirements-test.txt
CUDA_VISIBLE_DEVICES="" /tmp/layerloop-test-env/bin/python scripts/self_test.py
```

自测不下载模型、不访问数据集、不使用 GPU；现场创建一个随机初始化的微型 Llama 和 adapter/checkpoint，实际调用完整解码器跑通 boundary、tail 和 n-gram 对照。随机模型不用于质量或性能结论。

新机器可用 `bash scripts/install.sh` 创建本地 `.venv`。安装会访问 Python 包索引；不会修改已有服务器环境。仓库的 `pyproject.toml` 与实际环境在 PEFT、AV 等版本范围上有冲突，所以安装脚本采用观测版本、`--no-deps` 和独立 venv；运行入口设置 `DISABLE_VERSION_CHECK=1`。这不是通用版本兼容保证。源代码完整性和 CPU 实际执行测试比绕过声明本身更重要。

## 模型与数据不在代码 ZIP 内

需要已有的可信本地基座模型目录、训练过的 checkpoint 和数据文件。包内没有基座权重、训练/测试语料、密钥或 `.env` 文件，也不会自动下载它们。Meta-Llama 权重需要使用者自行取得授权。

对于已存在的 checkpoint-60375：

```bash
"$PYTHON" scripts/check_assets.py "$CHECKPOINT"
```

在自己的另一台机器上，需要取回推理 adapter、T/head 和 tokenizer 时：

```bash
python3 scripts/fetch_checkpoint.py --host 117_jump_vpn --output assets/checkpoint-60375
```

该可选操作约 163 MB，逐文件校验哈希；不包含基座模型和训练恢复状态。继续训练还需要来源 checkpoint 的 `trainer_state.json`，严格恢复优化器还需要对应 optimizer/scheduler/RNG 状态；原历史 `save_only_model` checkpoint 并不提供完整优化器恢复保证。

评测 JSONL 每行必须有 `question` 和 `answer`，不允许空行，选取区间内问题必须唯一。训练数据为 MetaMath 的 `query` / `response` JSON。不要把自造小样本文件当正式测试集。

## 先预检，再运行

```bash
source scripts/server_paths.example.sh
"$PYTHON" scripts/run_decode.py --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --data "$DATA" --output "$PWD/runs/boundary_smoke" --gpu 5 \
  --route boundary --samples 2 --max-new-tokens 64 --dry-run
```

`--dry-run` 只检查文件、数据覆盖并打印命令。确认后移除它才会执行。比较后半网络读出时，使用新的输出目录并把 `--route boundary` 换成 `--route tail`。基座架构必须与 checkpoint 一致；不能用 Llama 的 adapter 跑 Qwen。该轻量入口固定为数学任务提示；其他任务需要显式适配任务协议，不能直接混入主表。

输出含 `result.json`、`run.log`、`inputs.json`、启动/完成/失败 marker 和 GPU 观察日志。已有输出目录绝不覆写。完成 marker 表示样本与执行路径审计通过，不代表质量通过；wall < 1 或质量退化仍会如实保留。

GPU 入口要求空闲显存至少 51200 MiB、util <= 20、无 compute PID（包括未澄清的驱动残留 PID）。使用同用户/设备协作锁防止本包重复启动；不杀任何外部进程。运行中每五秒采样，外部重叠或监控失败标为 provisional。`sampled_exclusive` 仅表示采样时未见重叠，不是连续独占或正式 clean wall 的证明。

## 自定义 checkpoint 测速

在 Linux GPU 节点修改 `scripts/benchmark_decode.sh` 开头的参数，至少填写 `MODEL`、`CHECKPOINT`、`GPU`，并将 `DATA` 指向已有的 GSM8K 测试 JSONL。也可以通过环境变量覆盖参数：

```bash
export MODEL=/path/to/Meta-Llama-3-8B-Instruct
export CHECKPOINT=/path/to/checkpoint-xxxxx
export GPU=GPU-your-allocated-device-uuid
export DATA=/path/to/gsm8k_test.jsonl
bash scripts/benchmark_decode.sh --dry-run
bash scripts/benchmark_decode.sh
# 增加题数、比较单枚草稿，重复两遍：
SAMPLES=128 BLOCK=2 REPEATS=2 bash scripts/benchmark_decode.sh
```

默认参数为 `ROUTE=auto`、`SAMPLES=32`、`MAX_NEW_TOKENS=256`、`BLOCK=3`、`DTYPE=bf16`、`REPEATS=1`。`BLOCK=3` 为一枚目标模型已确定的 token 加最多两枚草稿。`auto` 根据 `boundary_head_rank` 选择 `boundary` 或 `tail`；头部配置存在不代表已经充分训练。core / multistep 权重通常使用 `tail`，完成头部训练后可显式设置 `ROUTE=boundary`。启动前打印 checkpoint 的累计 `global_step`；没有 `trainer_state.json` 的推理权重仍可评测，但无法从该文件确认步数。

默认优先使用项目 `.venv/bin/python`，其次使用当前环境的 Python；可显式设置 `PYTHON`。`OUT` 留空时自动生成实验目录，每次重复写入 `repeatN` 子目录；已有 `OUT` 会被拒绝。结果包括原有 `result.json`、`run.log` 和新增的 `timing_summary.txt`，简报打印速度、输出长度、草稿接受率、输出一致率和降序排列的分项耗时。每遍都比较同一 checkpoint 的 greedy 与投机解码，保持严格 target-match、n-gram off 和已有 GPU 检查。使用默认诊断计时，包含 prefill 和生成，不包含模型加载；该便捷入口不等同于下面的历史 N=384 协议。

`--dry-run` 只做文件和参数预检，不下载资产、不占用 GPU，也不创建输出目录。该脚本尚未在用户集群上完成 GPU 测速。

## 启用缓存优化的测速

新入口 `scripts/benchmark_decode_cache.sh` 默认启用 `CORRECTION_MODE=reuse`，支持已有 core/multistep 的 tail 路线和带头部 checkpoint 的 boundary 路线，不需要为缓存优化重新训练。填写脚本顶部的 `MODEL`、`CHECKPOINT`，确认数据和分配的 GPU；其他参数也可用同名环境变量覆盖。

```bash
export MODEL=/path/to/Qwen3-8B
export CHECKPOINT=/path/to/checkpoint-xxxxx
export DATA=/path/to/gsm8k_test.jsonl
export GPU=0
bash scripts/benchmark_decode_cache.sh --dry-run
bash scripts/benchmark_decode_cache.sh

# 分别测试延迟纠正和原路径，保持模型、样本、长度等条件一致：
CORRECTION_MODE=defer bash scripts/benchmark_decode_cache.sh
CORRECTION_MODE=off bash scripts/benchmark_decode_cache.sh
```

- `reuse`：传递 `--reuse-verify-cache-for-correction`，保留验证阶段的前缀 KV，只运行纠正 token。
- `defer`：仅传递 `--defer-correction-to-next-verify`，保留验证前缀并将纠正 token 留到下一轮验证，避免额外的独立纠正前向。
- `off`：不传递上述两个开关，使用原纠正路径。原 `benchmark_decode.sh` 默认仍为 `off`。

三种模式互斥，默认一次只跑所选模式。`ROUTE=auto` 按头部配置选择 tail/boundary；默认 32 题、block 3、BF16、重复 1 遍。继续使用严格 target-match、n-gram off、相同 checkpoint 的 greedy 基线、原 GPU 检查和同步分项计时，不混入其他运行优化。自动输出目录包含纠正模式；显式 `OUT` 必须为新目录。

`timing_summary.txt` 增加纠正模式、`fast_correction_cache_reuses`（缓存复用次数）和 `deferred_corrective_tokens`（延迟纠正次数）。`inputs.json` 和 `complete.marker.json` 保存模式；原始 `result.json` 的 `summary.args` 保存实际开关。入口会核对开关及每条样本的计数之和；`correction_optimization_observed: false` 表示开关配置正确但本次没有触发相应分支，不代表已经获得优化收益。

可直接使用底层包装器，例如 `python scripts/run_decode.py ... --correction-mode reuse`。新脚本依赖更新后的 `benchmark_decode.sh`、`run_decode.py` 和 `common.py`，同步时应拉取整个提交。缓存路径改变可能影响浮点计算结果；比较速度时继续检查输出一致性和任务质量。12 项新增测试覆盖开关传递、计数检查，以及真实 Bash 到解码命令的 dry-run；未在用户集群验证 GPU 性能或数值等价性。

## N=384 受控协议

```bash
source scripts/server_paths.example.sh
export GPU=5
export OUT="$PWD/runs/headline384_new"
DRY_RUN=1 bash scripts/reproduce_headline.sh
# 检查路径与数据后，以下命令才会执行两遍实验：
bash scripts/reproduce_headline.sh
```

此入口使用 GSM8K 物理行 8..391、384 个样本、最多 256 新 token、prompt cap 1024、K=3、FP16、non-thinking、strict target-match、boundary 和 n-gram off。默认 baseline 是同一个适配后 target 的逐 token greedy，不是未适配基座。当前代码/内核版本与硬件可能使结果不同；旧数值只是参考，不是启动脚本预期必须得到的答案。

## 训练完整链

```bash
# 新输出目录，不会覆盖旧训练。
python3 scripts/train.py --stage core --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --output runs/core --gpu 5 --dry-run
python3 scripts/train.py --stage multistep --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --checkpoint /path/to/checkpoint-49375 \
  --output runs/multistep --gpu 5 --dry-run
python3 scripts/train.py --stage boundary-warmup --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --checkpoint /path/to/checkpoint-59375 \
  --output runs/boundary_warmup --gpu 5 --dry-run
python3 scripts/train.py --stage boundary --model /path/to/base \
  --data /path/to/MetaMathQA-395K.json --checkpoint /path/to/checkpoint-59600 \
  --output runs/boundary --gpu 5 --dry-run
```

移除 `--dry-run` 才启动对应阶段。入口会为每个训练单独生成 dataset_info 与 YAML，不改共享仓库数据注册表。`--steps N` 是新增训练步数，会改变历史训练协议，不能直接称为原实验复现。

历史晚层链：core 49375 步 -> multistep 59375 -> boundary warmup 59600 -> boundary 60375。core 配置来自此前基于保存参数重建的补充材料，非原始 YAML 字节副本；另三个配置直接保留当前工作区历史文件。默认仅适用于该 Llama-3-8B 晚层链，其他模型须重新确定层索引、模板、adapter 与 checkpoint。

## T 与 boundary 联合训练（实验阶段）

`joint` 从已有 boundary checkpoint 继续，冻结目标模型及其 LoRA，开放 recurrent 模块中原本可训练的参数（默认架构为 T LoRA 和 boundary 头）。两步 rollout 使用预测 hidden；`detach_rollout: false` 让第二步损失也能沿状态路径反向更新第一步。当前配方是待验证的实验起点，不代表已经提高接受率或达到某个加速倍率。

配置位于 `configs/experimental/recurft_joint_boundary.yaml`。它叠加在该 checkpoint 对应的 `train.yaml` 上，保留模型结构参数，并检查循环层、T rank、boundary rank 与 checkpoint 元数据是否相符。只支持带 `trainer_state.json`、不含旧 optimizer/scheduler 状态的 checkpoint；历史 `save_only_model: true` 保存格式符合要求。原 checkpoint 保留，新阶段重新建立优化器；`--steps` 表示新增优化器更新次数，预热起点自动设为加载权重的 global_step。

```bash
MODEL=/path/to/Qwen3-8B
DATA=/path/to/MetaMathQA-395K.json
CKPT=/path/to/boundary_run/checkpoint_output/checkpoint-60375
SOURCE_CONFIG=/path/to/boundary_run/train.yaml

python scripts/train.py --stage joint \
  --model "$MODEL" --data "$DATA" --checkpoint "$CKPT" \
  --source-config "$SOURCE_CONFIG" --template qwen3_nothink \
  --output runs/joint_trial --gpu 0 --steps 2000 --dry-run
# 检查输出的 YAML 后，移除 --dry-run 执行。
```

Qwen3 非思考模式使用 `qwen3_nothink`；入口会拒绝 Qwen3 搭配 `llama3`。已使用错误模板训练的权重并不会因为改了模板就恢复，需要在新模板下重新检查目标模型质量与草稿匹配率。不要在联合阶段更换基座模型。

如果现有权重只有 core/multistep、`boundary_head_rank: 0`，先运行头部预热。以下示例适用于当前 29–30 层、pre/post rank 8、T rank 128 的配置；其他结构需先匹配对应配方：

```bash
python scripts/train.py --stage boundary-warmup \
  --model "$MODEL" --data "$DATA" --checkpoint /path/to/core_checkpoint \
  --template qwen3_nothink --output runs/head_warmup --gpu 0 --steps 225 --dry-run
# 完成实际预热后，joint 的 --checkpoint 和 --source-config
# 分别指向此次预热保存的 checkpoint 和 runs/head_warmup/train.yaml。
```

联合阶段的监督来自同一训练文本前缀下、固定的适配后目标模型分布。当前默认不启用数据标签 CE，优先用 KL 训练草稿匹配目标模型；这仍是基于训练文本的 hidden rollout 蒸馏，不等于完整的模型生成轨迹训练，也没有直接优化两枚 token 的联合接受事件。

关键参数：`recurft_stage1_heads_only: false`、`recurft_recurrent_trainable_only: true`、`recurft_multistep_steps: 2`、`recurft_multistep_detach_rollout: false`。`recurft_multistep_loss_weight: 0.1` 会乘到 rollout 内所有损失上，因此 boundary KL 内部权重设为 `10.0`，预热后的有效权重为 `1.0`；基础 rollout hidden 损失权重为 `0.1`。单步 recurrent 权重也保留 `0.1`：当前实现用它作为整个 recurrent/rollout 分支的入口，不能直接设零。两项普通 boundary CE/KL 设零，以免混入真实 hidden 上的旧头部训练目标。

先观察 `recurft_multistep_boundary_teacher_top1_k1`、`..._k2`、`recurft_multistep_boundary_logit_kl_k1`、`..._k2`。这些是训练前缀上的分步指标，不是在线两步连续接受率。每个候选 checkpoint 仍应使用 `ROUTE=boundary` 在独立验证集上测接受率、输出差异、任务质量和 wall speed；最终评测集不要用于反复选择超参数。2000 步只是首轮实验长度，不保证训练充分。

此入口已通过 11 项配置/实际 CLI dry-run 检查，未在集群执行联合训练，也未重新运行完整 ML 梯度测试。原四阶段配置保留，旧 CPU 测试报告不代表这项新实验已经验证。

## 验证范围

本次实际通过 87 项单元/参数化测试，以及 boundary、tail、ngram 三条 CPU 小模型端到端解码；完整框架源码已成功构建 wheel。真实 checkpoint 的 8 个推理文件哈希、GSM8K 的 384 样本输入覆盖、训练入口 dry-run 均已检查。

`provenance/cpu_test_report.json` 记录测试结果、环境版本和对应代码 SHA256，导出器拒绝把报告绑定到修改后的代码。该测试不是全量 8B 训练/评测，也没有重新测量论文 wall。没有在全新机器上重装并验证整套 CUDA 依赖；下载与安装仍取决于网络、wheel 可用性和 NVIDIA 驱动。本包保留上游 Apache-2.0 LICENSE；个人路径与历史审计存在于 provenance，此包是研究代码交付，不是匿名投稿附件。
2026-09-14 评分修复：数学评分现在从官方 GSM8K 原始 `answer` 的 `####` 后提取标准答案，仍兼容已经只保留最终答案的数据。修复前直接使用官方原始 JSONL 会把完整推导当作标准答案，从而错误判零；这个问题不影响已记录的生成 token 或耗时。修复通过 5 项独立评分回归检查（`python tests/test_math_scoring.py`），未重新执行 GPU 实验。上面的历史 CPU 报告仍绑定导出时的代码快照，不代表已验证本次修改后的全部模型路径。


## 自适应投机解码测速

入口是 `scripts/benchmark_decode_adaptive.sh`。填写顶部 `MODEL`、`CHECKPOINT`、`DATA` 和 `GPU`，即可使用已有权重测试；无需先进行联合训练。core/multistep 可设 `ROUTE=tail`，已经训练好 boundary 头的权重可设 `ROUTE=boundary`。

```bash
bash scripts/benchmark_decode_adaptive.sh --dry-run
bash scripts/benchmark_decode_adaptive.sh

# 最多尝试 4 枚草稿，仍按置信度提前停止
BLOCK=5 bash scripts/benchmark_decode_adaptive.sh

# 同样的 checkpoint、数据、BLOCK、GPU、纠正模式下比较固定长度
DRAFT_POLICY=fixed bash scripts/benchmark_decode_adaptive.sh
```

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DRAFT_POLICY` | `adaptive` | `adaptive` 开启门控；`fixed` 关闭门控、提前停止和冷却 |
| `TARGET_SKIP_MARGIN` | `1.0` | 目标 top-1/top-2 原始 logit 差值小于此值，不尝试草稿 |
| `TARGET_SHORT_MARGIN` | `3.0` | 未被跳过、但目标差值仍小于此值，最多尝试 1 枚草稿 |
| `DRAFT_MIN_MARGIN` | `1.0` | 上枚草稿的原始 logit 差值小于此值，停止继续生成草稿；已有草稿仍验证 |
| `COOLDOWN_FAILURES` | `3` | 连续多少个尝试草稿的块零接受后，触发冷却 |
| `COOLDOWN_CYCLES` | `4` | 冷却暂停多少轮，然后恢复尝试 |
| `BLOCK` | `3` | 已确定 token 加草稿的总块上限，`3` 对应最多 2 枚草稿 |
| `CORRECTION_MODE` | `reuse` | 使用已有验证 KV 复用；也可设 `defer` 或 `off` |

三个差值阈值均不是概率，设 `0` 可关闭对应判断；启用短块门控时 `TARGET_SHORT_MARGIN` 必须不小于 `TARGET_SKIP_MARGIN`。冷却的两项同时设 `0` 才关闭。这些是未校准的起始值，需根据实际 checkpoint 的接受长度和耗时调整，不保证加速。

每遍同时测 greedy 基线。`timing_summary.txt` 除速度和任务质量外，还报告无草稿块比例、每个草稿块平均接受草稿数、门控/冷却次数和块长度分布。平均接受草稿数不包含已确定 token；无草稿块比例的分母是已记录验证块，不含直接遇到 EOS 结束的轮次，也包括生成预算只剩一个 token 的块。门控和冷却可能在同一轮发生，两个跳过计数不能相加。全部跳过时仍可保存结果，标记 `recurrent_drafting_observed: false`，不据此声称观察到了 T 的效果。

`inputs.json` 记录入口参数，`result.json` 中的 `summary.args` 记录解码器实际参数，`complete.marker.json` 记录参数核对和聚合统计。全部草稿使用严格 target-match 验证，n-gram 关闭。该脚本接入既有解码路径，无草稿轮次仍有缓存维护开销；需要同时比较耗时、输出长度、一致率和答题准确率。

同步到集群至少需要以下四个文件：`scripts/benchmark_decode_adaptive.sh`、`scripts/benchmark_decode.sh`、`scripts/run_decode.py`、`scripts/common.py`，并使用仓库中支持上述参数的解码器。仅复制新入口脚本到旧版包装器上不能运行。原测速入口默认仍为 fixed，新参数不改变训练。新增 CPU 检查覆盖 Bash 到实际解码参数解析、门控阈值边界、固定长度对照、全部跳过及统计计数；GPU 性能需在集群实测。


## 按生成位置调整草稿上限与长输出实验

自适应入口新增 `SHORT_BLOCK`、`MEDIUM_BLOCK`、`LONG_BLOCK`、`MEDIUM_POSITION`、`LONG_POSITION`。块上限包含 1 枚已确定 token；位置按已生成 token 数计，不含 prompt。三段上限默认都等于 `BLOCK`，位置默认仍是 96/192，因此旧命令的默认行为不变。

- `DRAFT_POLICY=fixed`：每段上限均为 `BLOCK`；若显式设置不同段上限，会报错，避免忽略参数。
- `DRAFT_POLICY=schedule`：仅按位置调整上限，不开置信度门控或冷却。
- `DRAFT_POLICY=adaptive`：在各段上限内继续使用置信度门控和冷却。
- `RUNTIME_MODE=diagnostic`（默认）：同步分项计时，用于定位开销。
- `RUNTIME_MODE=throughput`：开启 `compact_runtime_stats` 和 `production_async_timing`，减少详细诊断和逐组件同步。greedy 和投机的整段 wall 计时仍在起止同步；分项仅是主机提交时间，不用于 GPU 耗时分解。无法据此准确扣除 prefill/T 初始化，因此结果中的 decode-only 派生指标留空。

先填写模型、checkpoint、数据和 GPU。建议用同一开发集依次比较，先保持纠正模式一致：

```bash
export SAMPLES=32 MAX_NEW_TOKENS=2048
export RUNTIME_MODE=throughput CORRECTION_MODE=reuse

# 固定上限：最多 1、2、4、6 枚草稿
for block in 2 3 5 7; do
  DRAFT_POLICY=fixed BLOCK="$block" SHORT_BLOCK="$block" MEDIUM_BLOCK="$block" LONG_BLOCK="$block" \
    bash scripts/benchmark_decode_adaptive.sh
done

# 生成位置 [0,128)、[128,512)、[512,...)：最多 1、2、4 枚草稿
DRAFT_POLICY=schedule BLOCK=5 SHORT_BLOCK=2 MEDIUM_BLOCK=3 LONG_BLOCK=5 \
  MEDIUM_POSITION=128 LONG_POSITION=512 bash scripts/benchmark_decode_adaptive.sh

# 相同分段，再叠加自适应置信度和冷却
DRAFT_POLICY=adaptive BLOCK=5 SHORT_BLOCK=2 MEDIUM_BLOCK=3 LONG_BLOCK=5 \
  MEDIUM_POSITION=128 LONG_POSITION=512 bash scripts/benchmark_decode_adaptive.sh
```

选定长度策略后，再将同一配置的 `CORRECTION_MODE` 从 `reuse` 改成 `defer` 单独对照。每次输出目录都必须不同，脚本默认自动生成。正式报告前应扩大样本并重复测速；不能仅保留速度最好的样本。长输出建议分别设置 512/1024/2048 的生成预算，同时记录实际输出长度和截断情况；提高上限不保证模型实际输出更长，也不保证后段草稿更容易接受。

`timing_summary.txt` 和 `complete.marker.json` 新增 early/mid/late 接受统计，按块起始位置归段，报告进入该段的样本数、草稿数、接受数和每个草稿块的平均接受数。这是接受统计，不是分段 GPU 耗时；尾段仅涵盖实际到达该段的样本，不能直接当作所有样本的后期表现。

可保留 GSM8K 作短输出对照，并使用 [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) 探索更长数学解答。其 JSONL 的题目字段是 `problem`，需映射为本入口要求的 `question`，`answer` 保留最终答案；不要把 `solution` 放入输入。该数据包含符号答案，当前简单字符串判分不能替代符号等价评测，生成长度也需实测。调参在开发集完成，最终测试集只用于固定方案报告。

本轮同步除四个 scripts 文件外，还需更新 `LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py`（修正吞吐模式的派生计时口径）。所有改动仍仅在本地，尚未在 H200 上验证加速。


## 三组对照：greedy、固定长度、固定起始位置后投机

新入口 `scripts/benchmark_decode_start.sh` 自动固定模型、checkpoint、数据区间、草稿上限、纠正模式和计时方式，运行三种方法：

1. `greedy`：目标模型逐 token 生成，每个 repeat 只将一份 greedy 结果纳入统计。
2. `fixed`：从生成开始就允许固定长度投机。
3. `after_start`：生成满 `SPEC_START_POSITION` 枚 token 后，再允许相同长度的投机。

两组投机都关闭置信度门控和失败冷却，使用严格 target-match、n-gram off，并启用延迟 T 初始化及 lazy T 同步。唯一的投机策略差别是起始位置。起始位置前没有草稿，但仍有现有解码器的目标 hidden/KV 维护开销，不能把这一段的成本当成独立 greedy 路径的成本。

```bash
# 首次使用：填写脚本顶部路径，或用环境变量指定
export MODEL=/path/to/Meta-Llama-3-8B-Instruct
export CHECKPOINT=/path/to/checkpoint-xxxxx
export DATA="$PWD/data/gsm8k_test.jsonl"
export GPU=0

# 环境变量方式
SPEC_START_POSITION=128 bash scripts/benchmark_decode_start.sh --dry-run
SPEC_START_POSITION=128 bash scripts/benchmark_decode_start.sh

# 命令行优先于环境变量；START 表示数据行，二者不同
bash scripts/benchmark_decode_start.sh --spec-start-position 256 --start 0 --samples 32

# 同一配置依次比较不同起始位置；每次自动创建不同输出目录
for position in 0 64 128 256 512; do
  bash scripts/benchmark_decode_start.sh --spec-start-position "$position"
done
```

| 环境变量 / 命令行 | 默认值 | 含义 |
| --- | --- | --- |
| `SPEC_START_POSITION` / `--spec-start-position` | 128 | 已生成多少 token 后开启投机，不含 prompt；0 是从头开始的对照 |
| `START` / `--start` | 0 | 数据起始行，0 起算 |
| `SAMPLES` / `--samples` | 32 | 数据区间大小；三组使用相同题目及顺序 |
| `BLOCK` / `--block` | 3 | 总块上限，包含 1 枚已确定 token，故最多 2 枚草稿 |
| `MAX_NEW_TOKENS` / `--max-new-tokens` | 1024 | 生成预算；回答可因 EOS 提前结束 |
| `REPEATS` / `--repeats` | 1 | 重复次数；不同 repeat 交替执行 fixed/after_start |
| `WARMUP_RUNS` / `--warmup-runs` | 1 | 每个解码进程用首题预热 greedy 和投机，不计入结果；0 关闭 |
| `RUNTIME_MODE` / `--runtime-mode` | throughput | 整段同步计时；diagnostic 可开启分项同步 |
| `CORRECTION_MODE` / `--correction-mode` | reuse | 两组均使用同一种纠正模式，可选 off/reuse/defer |

`MODEL/CHECKPOINT/DATA/GPU/ROUTE/DTYPE/OUT` 也分别支持 `--model/--checkpoint/--data/--gpu/--route/--dtype/--output`；命令行优先。脚本不会继承自适应入口的 `DRAFT_POLICY`、分段长度或置信度阈值，避免污染本次单变量对照。

输出结构：

```text
OUT/
  inputs.json                        配置、数据与 checkpoint 文件哈希
  comparison.txt                     三组耗时、吞吐、质量、输出一致率和投机触发情况
  comparison.csv                     三组汇总，可直接用于表格
  comparison.json                    每遍结果及按总耗时/总 token 计算的聚合结果
  repeat1/fixed/result.json           含本遍共用的 greedy 和 fixed 原始结果
  repeat1/after_start/result.json     延后投机原始结果，greedy 取自同遍 fixed
  repeat1/{fixed,after_start}/run.log
  complete.marker.json               所有重复完成后写入；失败则写 failed.marker.json
```

每个解码进程分别加载同一 checkpoint，加载和预热不计入推理耗时，prefill、T 初始化及全部输出生成计入。两组共享同一份 greedy 的 token 输出和时间作为比较基准，避免使用不同基线或只比较投机启用后的片段。重复结果按总 token/总耗时汇总，不简单平均各样本加速比。报告同时保留生成 token 数和吞吐加速比，避免因输出更短误判提速。

`first_draft_position` 在每组 `complete.marker.json` 中按输出 token 的 0 起始下标记录；若 `SPEC_START_POSITION=N`，最早的投机块从位置 N 开始，其首枚 token 已由目标模型确定，所以首枚草稿最早位于 N+1。若回答在阈值之前结束或剩余预算不足，报告会保留该样本，并显示未触发投机，接受率为 null 而非伪造 0 次接受实验。

同步此入口需要两个新文件 `scripts/benchmark_decode_start.sh`、`scripts/benchmark_decode_start.py`，以及同版本 `scripts/run_decode.py`、`scripts/common.py` 和底层 `LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py`（新增可选预热参数；其他入口默认不预热）。无需重新训练 checkpoint。本地测试使用真实参数解析和合成解码结果验证调度、汇总及边界情况，尚未进行集群 GPU 性能测试。


## 逐项测试纠正、KV 和 T 的推理开销

`scripts/benchmark_decode_speed.sh` 使用已有 checkpoint，不修改训练或接受规则；默认 `SPEED_PROFILE=defer`。它调用上面的三组对照入口，每次重新测量共享 greedy 基线。`MODEL/CHECKPOINT/DATA/GPU` 沿用环境变量或 `benchmark_decode_start.sh` 中配置的路径。

| SPEED_PROFILE | 纠正方式 | 原地 KV | 合并 T LoRA | 与上一项的区别 |
| --- | --- | --- | --- | --- |
| reuse | reuse | 关闭 | 关闭 | 当前配置对照；纠正 token 仍有单独完整目标前向 |
| defer | defer | 关闭 | 关闭 | 纠正 token 作为下一轮已确定 token，与草稿一并验证 |
| cache | defer | 开启 | 关闭 | 复用缓存对象并裁掉拒绝后缀，避免显式克隆 T/验证缓存 |
| tmerge | defer | 开启 | 开启 | 推理时将 T 的 LoRA 合并到权重，减少额外算子 |

原地 KV 仍使用 DynamicCache，其内部追加可能分配内存；这不是静态 KV/CUDA Graph 实现。T 合并只修改进程内模型，不写回 checkpoint，也不合并目标模型 LoRA。低精度运算顺序变化可能改变草稿，必须同时检查接受情况及质量。上述优化需要集群实测，不能保证 1.3 倍加速。

```bash
# 已填好 start 脚本路径即可运行；先只改变纠正方式
bash scripts/benchmark_decode_speed.sh --dry-run
bash scripts/benchmark_decode_speed.sh

# 同一批数据逐项比较；每次保存独立报告。检查 fixed 行，勿只选更短的输出。
for profile in reuse defer cache tmerge; do
  SPEED_PROFILE="$profile" bash scripts/benchmark_decode_speed.sh --samples 32 --repeats 1
done

# 再在有效配置上测试每轮 1、2、3 枚草稿；BLOCK 包含已确定的首 token
for block in 2 3 4; do
  SPEED_PROFILE=cache bash scripts/benchmark_decode_speed.sh --block "$block"
done
# 探索集选好配置后，固定参数，在未用于选参的数据区间重复测试。
SPEED_PROFILE=cache bash scripts/benchmark_decode_speed.sh --start 32 --samples 128 --repeats 3
```

`SPEED_PROFILE` 固定该档的纠正/KV/T 合并设置及关闭 hidden hook，避免继承其他实验的开关；命令行仍可覆盖，最终以 `inputs.json` 为准。新报告显示每个草稿轮平均接受数、零接受/全接受比例、目标调用次数、缓存纠正复用数及延迟纠正数。吞吐模式分项仅为主机计时，不用于 GPU 耗时分解；总耗时仍使用同步计时。出现 `provisional_overlap_or_monitor_error` 时检查 `gpu_audit.json`，该状态本身不能区分其他任务占卡与容器 PID/监控问题。

三组对照入口另支持 `INPLACE_DRAFT_CACHE=0/1`、`MERGE_RECURRENT_LORA=0/1`、`ANCHOR_HOOK_HIDDEN_STATES=0/1`，以及对应 `--inplace-draft-cache`、`--merge-recurrent-lora`、`--anchor-hook-hidden-states`（`--no-...` 关闭）。原入口默认仍全部关闭。hidden hook 只保留 T 所需的边界 hidden，可单独试验。此包装器限定原地缓存配合 reuse/defer，避免进入未覆盖的纠正回放组合。

若集群已能运行上述 start 对照，本次同步三个文件即可：`scripts/benchmark_decode_speed.sh`、`scripts/benchmark_decode_start.sh`、`scripts/benchmark_decode_start.py`。底层调用已有推理开关，训练代码未修改；本地验证不包含 GPU 加速或数值一致性验证。


## 批量 boundary、并行草稿与轻量严格验证

四档集群结果复核见 `reports/speed_review_20260916.md`。tmerge 的 fixed 吞吐加速为 1.043 倍；新的三项以它为控制。原四档及原入口默认行为保留，不改训练或 checkpoint。

| SPEED_PROFILE | 在 tmerge 上启用 |
| --- | --- |
| tmerge | 本轮控制 |
| batch | 批量 boundary/LM head 读出 |
| parallel | boundary 读出和 T 下一步使用不同 CUDA stream |
| strict | GPU 连续前缀比较、省略逐 token 诊断、无需 margin 时使用 argmax |
| batch_strict | batch + strict |
| parallel_strict | parallel + strict |

batch 和 parallel 使用不同草稿路径，不能同时开启。batch 要求 boundary 路线及 `token_conditioning_rank=0`；不满足时直接报错，避免静默退回逐步计算。strict 仅支持 compact fixed/schedule、全位置 lambda=1 的神经草稿验证；不兼容宽松接受、类别规则、串行回放、n-gram 或提前流水执行。接受判定仍由完整目标模型给出，拒绝后只提交连续匹配前缀；argmax 的并列处理或批量矩阵形状可能改变草稿，需检查实测输出。

```bash
# MODEL/CHECKPOINT/DATA/GPU 沿用已填写的路径。
SPEED_PROFILE=batch bash scripts/benchmark_decode_speed.sh --dry-run
for profile in tmerge batch parallel strict batch_strict parallel_strict; do
  SPEED_PROFILE="$profile" bash scripts/benchmark_decode_speed.sh --samples 32 --block 3 --repeats 1
done

# 先用未加速的 tmerge 在相同参数下复测，避免把运行间漂移算作优化收益。
# 若想比较 reuse + 原地 KV 的另一基线，显式覆盖纠正方式：
SPEED_PROFILE=batch_strict bash scripts/benchmark_decode_speed.sh --correction-mode reuse

# 选定一档后再测试 BLOCK=2（1 枚草稿）与 BLOCK=3；参数应在探索集选定。
SPEED_PROFILE=batch_strict bash scripts/benchmark_decode_speed.sh --block 2
# 小样本分项诊断：strict 保留 compact stats，但逐组件同步计时。
SPEED_PROFILE=batch_strict bash scripts/benchmark_decode_speed.sh --samples 4 --runtime-mode diagnostic
```

start 入口提供 `BATCHED_DRAFT_BOUNDARY`、`SINGLE_GPU_PARALLEL_DRAFT`、`FAST_STRICT_VERIFICATION` 三个 0/1 环境变量及对应 `--batched-draft-boundary`、`--single-gpu-parallel-draft`、`--fast-strict-verification`；`--no-...` 关闭。speed 档位固定这些变量，CLI 仍可覆盖。报告包含批量块数、并行步数和轻量验证块数；审计实际执行覆盖率，不仅检查命令行。BLOCK=2 不需要额外 T rollout；计时审计同时识别 T 初始化/同步和并行 T，避免将有效实验误报为未使用 T。

本次需同步五个运行文件（保持仓库相对路径）：

- `scripts/benchmark_decode_speed.sh`
- `scripts/benchmark_decode_start.sh`
- `scripts/benchmark_decode_start.py`
- `scripts/common.py`
- `LLaMA-Factory/experiments/recurft_math/recurft_speculative_generate.py`

沿用已同步的 `scripts/run_decode.py` 及其余仓库依赖。可额外复制 `tests/test_strict_decode.py`，在集群模型环境运行：

```bash
python -m unittest discover -s tests -p test_strict_decode.py -v
```

测试执行真实的解码控制流及 PyTorch 张量操作，使用确定性小模型覆盖首步/中途拒绝、全部接受、EOS、生成预算、不同块长、延后起始位置和纠正/KV 分支；GPU 可用时额外运行 CUDA stream 对照。本地 CPU 测试不能证明真实 Llama/H200 数值一致或达到指定加速比。

### 验证前向开销：目标 LoRA 合并对照

`target_merge` = `batch_strict` + 目标模型 LoRA 合并；`target_hook` 在此基础上仅捕获需要的边界 hidden，避免返回所有层 hidden states。这些底层功能已存在，本次接入测速入口；旧档位默认不变。greedy 和投机共用合并后的目标模型，不能把相对未合并 greedy 的收益算作投机加速。BF16 合并可能改变舍入和输出，需同时检查准确率与一致性。尚未实测 H200 收益。

```bash
# 沿用 MODEL/CHECKPOINT/DATA/GPU，同一区间分别重测自身 greedy 基线
for profile in batch_strict target_merge target_hook; do
  SPEED_PROFILE="$profile" bash scripts/benchmark_decode_speed.sh --samples 32 --block 3 --repeats 3
done
# 分项诊断会加入同步开销，不作为最终吞吐结论
SPEED_PROFILE=batch_strict bash scripts/benchmark_decode_speed.sh --samples 4 --runtime-mode diagnostic
```

start 入口支持 `MERGE_TARGET_LORA=1` 或 `--merge-target-lora`（`--no-merge-target-lora` 关闭）；speed 档位固定默认值，CLI 可覆盖。已安装 batch_strict 版本时，只需更新 `scripts/benchmark_decode_speed.sh`、`scripts/benchmark_decode_start.sh`、`scripts/benchmark_decode_start.py`。此增量不修改解码器、训练或权重文件。
