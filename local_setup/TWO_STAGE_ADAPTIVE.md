# 两层自适应投机解码

启动入口：`scripts/run_two_stage_adaptive.sh`；Python 入口：`local_setup/benchmark_two_stage_adaptive.py`。
运行时使用独立的 `adaptive_two_stage_runtime.py`，不修改已有实验正在使用的 decoder。
这是已训练好的 T 和 boundary 的推理策略，不需要重新训练。

## 实际策略

每个问题单独维护历史，问题之间不共享接受率。默认规则如下，所有阈值是探索起点，没有经过速度最优校准。

1. 草稿开始前，读取已有目标 logits 的 top1-top2 **原始 logit 差**。
   - 小于 2：本轮不调用 T，目标模型逐 token 前进。
   - 大于等于 2 且小于 4：最多尝试 1 枚草稿。
   - 大于等于 4：允许更长草稿，但还要通过下面的近期接受率限制。
2. 统计本题最近 8 个实际尝试草稿的循环：`接受草稿总数 / 提议草稿总数`。
   - 无历史或接受率小于 25%：最多尝试 1 枚，保留探索机会。
   - 25% 到 60% 之间：最多尝试 2 枚。
   - 大于等于 60%：最多尝试 4 枚。
   - 本轮预算取两个限制的较小值。跳过的循环不算草稿失败。
3. 连续 2 个尝试循环没有接受任何草稿：接下来的 8 个循环不调用 T，之后重新尝试。不会永久关闭 T。
4. **草稿生成过程中**：每生成一枚草稿，检查该步草稿 logits 的 top1-top2 差；小于 1 时立即停止继续展开，交给目标模型验证。刚生成的低置信度候选仍进入验证，并不直接提交。
5. 全部草稿经过严格目标验证：`target_match`、lambda=1、ngram 关闭；不接受未经验证的 token。

例子：目标 margin=5，最近接受率=70%，本轮允许 4 枚草稿。第一枚 margin=2.1，继续；第二枚 margin=0.6，则在第二枚后停止，验证这两枚，不再计算第三、第四枚。若目标仅接受第一枚，历史新增 `(accepted=1, proposed=2)`，下一轮据更新后的窗口重新判断。

这里的置信度是 **logit 差，不是 softmax 概率**，不能直接照搬其他实现的概率阈值。`--max-drafts 4` 表示最多 4 枚草稿；底层 `max_block_tokens=5` 还包含 1 枚已经由目标选定的 anchor。

## 启动

```bash
cd /mnt/llmshared-ssd-hd/wangruitao/inception-joint
# 先查看配置，不加载模型或启动测评
bash scripts/run_two_stage_adaptive.sh --dry-run

# 默认：联合训练前/后，各自固定与自适应，加同一个 greedy 对照
# 16 道 GSM8K，起始索引 8，重复 2 次，生成上限 256
GPU=2 bash scripts/run_two_stage_adaptive.sh

# 仅调整策略；不改变 checkpoint
GPU=2 SAMPLES=16 bash scripts/run_two_stage_adaptive.sh \
  --adaptive-margin 2 --target-high-margin 4 --draft-margin 1 \
  --max-drafts 4 --acceptance-window 8 \
  --good-acceptance 0.25 --strong-acceptance 0.60 \
  --adaptive-failures 2 --adaptive-cooldown 8

# 较大规模，建议先在独立校准问题上确定阈值再运行
GPU=2 SAMPLES=384 REPEATS=2 START=8 bash scripts/run_two_stage_adaptive.sh

# BF16 单独测；同一轮中 greedy 和全部投机方法使用相同目标 dtype
GPU=2 DTYPE=bf16 SAMPLES=16 bash scripts/run_two_stage_adaptive.sh

# 前 64 枚响应 token 不用 T，之后才允许投机（不含 prompt）
GPU=2 SAMPLES=16 bash scripts/run_two_stage_adaptive.sh --draft-start-token 64
```

`--draft-start-token` 默认为 0，同时作用于 fixed 和 adaptive。设置 64 后，前 64 枚响应 token 逐 token 生成；达到门槛后，fixed 按固定预算尝试，adaptive 仍需通过置信度、历史接受率与冷却判断。首次开启前保存边界 hidden，首次开启时批量建立包含前文的 T 缓存；这部分补算时间包含在 wall time 中。生成提前结束则无需初始化 T。该参数不是按最终答案长度的百分比切分，因为生成开始时尚不知道真实结束位置。

对照可分别设置 `--draft-start-token 0/32/64/128`，保持数据、dtype、生成上限、最大草稿数和其他参数一致。与旧 `block=3` 实验对齐草稿上限时，同时设置 `--max-drafts 2`。新参数的性能还未完成实测；后期接受率更高不自动等于总时间更短。

也可以在一次运行中对每道题配对比较全部开启位置，共用一个 greedy 对照，并轮换方法执行顺序：

```bash
GPU=1 SAMPLES=64 REPEATS=2 bash scripts/run_two_stage_adaptive.sh \
  --draft-start-tokens 0,32,64,128 --max-drafts 2
```

它会生成 `old_fixed_start0`、`old_adaptive_start64` 等 16 种组合；`--draft-start-tokens` 覆盖单个 `--draft-start-token`。上述 shell 默认仍是旧 core 20000 的 checkpoint；最新 core 36000 的自动联合训练与测评由 `local_setup/run_latest_joint_delayed.py` 管理，当前实验路径见 `runs/latest_joint_delayed_experiment.txt`。不要在同一 GPU 上重复启动已经在队列中的测评。

默认 FP32 是为与当前历史测试衔接，也支持 BF16/FP16。严格目标验证并不保证不同形状的浮点 forward 与逐 token forward 数值完全相同；脚本逐题比较完整 token IDs，若不同则保存结果后报错，不将该轮标为成功。BF16 的速度和一致性必须分别实测。

可以通过 `MODEL`、`DATA`、`CHECKPOINT`、`COMPARE_CHECKPOINT`、`OUT`、`PYTHON` 修改路径。Python 入口支持只传一个 checkpoint，例如 `--variants fixed,adaptive`；`old_` 前缀对应 `--compare-checkpoint`。双 checkpoint 模式检查目标 adapter 权重和 tokenizer/template 相同，保证共用 greedy 对照有效。

默认 checkpoint：

- `old_fixed` / `old_adaptive`：`runs/qwen3_joint_core20000_20260915_143933/warmup/checkpoints`，联合训练前。
- `fixed` / `adaptive`：`runs/qwen3_joint_core20000_20260915_143933/joint/checkpoints`，联合训练 2000 步后。

固定对照与自适应使用同一个最大草稿数，默认 4。过去 `block=3` 的固定实验实际只有 2 枚草稿；要采用该上限，请设置 `--max-drafts 2`。新脚本同时使用不同的运行时优化，不应将其绝对耗时与过去不同协议的结果直接比较。

## 计时与输出

输出到 `runs/two_stage_adaptive_时间戳/`，目录已存在则拒绝覆盖。

- `summary.json`：每种方法的总耗时、相对 greedy 加速比、token 一致数、接受率、平均接受草稿数、门控/冷却/中途停止次数。
- `result.json`：逐题 token IDs、逐题耗时、实际配置、草稿和验证统计、提取答案匹配结果。
- `routing_cycles.csv`：逐题逐循环的目标 margin、历史接受率、实际预算、每步草稿 margin、提议/接受长度、提前停止和冷却标记。
- `gpu_telemetry.jsonl`：共享 GPU 上的利用率、显存、其他进程；每 5 秒采样。
- `run_manifest.json` 与源文件快照：命令参数及本次 runner/runtime/policy 的 SHA256。
- `completion.json`：全部测量完成且严格一致性检查通过后才生成。

`speedup = greedy 总秒数 / 投机总秒数`。大于 1 才加速。`mean_accepted_drafts` 的分母是全部循环，`mean_accepted_per_draft_cycle` 的分母是实际尝试草稿的循环；两个指标均不包括 anchor 或纠正 token。底层 `proposed_len` 和 `accepted_len` 包含 anchor，读 CSV 时需减 1。末尾 EOS 可以直接结束而不生成 block record，因此 `recorded_cycles` 可能比总 `cycles` 少。

预热不计入测量；每题轮换方法执行顺序；外层 CUDA 同步计时包含 prefill、实际解码和运行时统计开销，不含模型加载、JSON 序列化与 GPU 采样的显式调用。关闭组件级 CUDA 同步，因此组件计时不是可靠的独立 GPU kernel 耗时。共享 GPU 并发仍会影响总时间，结果只可标为探索测速。

## 减少额外开销

固定和自适应均使用 compact diagnostics、异步组件计时、只捕获边界 hidden 的 hook、去掉全 1 目标 mask、复用验证缓存、合并 LoRA、延迟 T 初始化与批量补齐 T 历史。两者都关闭预先批量计算全部草稿的 boundary 路径，使自适应能够真正停止后续草稿计算。

未使用 T 的循环不复制 T 缓存。为以后恢复 T，仍需保留目标边界 hidden，恢复时仍需补齐 T 历史；目标验证缓存管理也仍有成本。因此跳过投机不等于已经达到原生 greedy 的全部性能，脚本不保证加速。

## 已做的验证

CPU 策略测试：窗口按 token 数加权、冷却后的探索恢复、窗口过期、跨问题重置、最大预算限制。
GPU 功能检查见 `runs/two_stage_adaptive_smoke_20260916/`：Qwen3-8B、两个 checkpoint、FP32、2 道计分问题和 1 道预热、每道最多 64 token。它只验证执行逻辑与 token 一致性，不构成速度或任务准确率结论。
