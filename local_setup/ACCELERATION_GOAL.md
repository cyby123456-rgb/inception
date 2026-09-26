# Qwen3-8B 冻结权重推理优化

工作目录独立于原训练仓库；本轮没有修改训练或 checkpoint。公开代码基线是 31dc971。本目录为尚未提交的推理实验实现，不能当作上游仓库已经包含的功能。

## 达标口径

目标是同一 target 模型、精度、题目和长度上限下，完整生成相对同等级优化 greedy 达到 1.3×。wall 与 token/s 分别报告；不能以减少 target 调用、单个组件提速、缩短输出或使用较慢 greedy 充当达标。共享 GPU 的结果标为探索结果。

开发题目 GSM8K [0,4)，第一轮验证 [48,64)，均为 0-based 行号。16 题重复三次仍是 16 道独立题目。BF16、batch1、最大 prompt 1024、最大生成 1024；默认禁用 thinking，与当前冻结 checkpoint 的测试配置保持一致。

最新 checkpoint：`../inception-joint/runs/qwen3_joint_full60375_20260917_010621/joint/checkpoints`；before 对照为同目录 `initial_full60375`。target adapter 经逐 tensor 比较相同；after 新增联合训练 2000 步。此前 core36000 结果不与此分支混算。

## 实现

- `compact_decode.py`：固定严格 boundary 路线。草稿 ID 留在 GPU，验证后一次回传接受前缀和输出 ID；target/T KV 在每个块边界对应真实已提交前缀，拒绝后回退。target hook 只取必要边界 hidden。EOS 之后的投机后缀可被额外计算，但不能被提交。
- 同文件的 `greedy_decode`：argmax 的 GPU tensor 直接成为下一步输入，CPU 只读 ID 判断停止和保存输出；作为更强 baseline，同时保留 legacy greedy 对照。
- `recurrent_graph.py`：T 使用预分配 KV 和 CUDA Graph。长度分桶 256/512/1024/2048/2304，显式 causal mask 保证读不到未提交的未来位置。拒绝只回退逻辑游标，下一次真实状态同步覆盖对应 KV。
- `draft_cycle_graph.py`：进一步把真实 hidden 同步、递归展开、boundary 和草稿 argmax 合入一次 graph replay。它不能省去完整 target 验证。
- `strict_prefix_kernel.py`：两段 Triton kernel 算完整词表 argmax、连续匹配前缀、EOS 截止和下一枚 target token；保留 first-index tie 与 NaN argmax 语义。与原 PyTorch 决策逐字段校验。
- `lower_right_target.py` / `fused_rmsnorm.py`：此前增加的 target attention 与 RMSNorm 路径。所有匹配 greedy 同样使用 target 优化，不单独削弱 baseline。

CUDA Graph 首次 capture/编译在预热中完成，不计入稳态生成；每道题的 prefill、T 初始化及缓存复制仍计时。部署冷启动成本另外看 warmup 行，不能说这些成本消失了。图 replay 使用稳定地址并复用输入缓冲，遵循 [PyTorch CUDA Graph 官方说明](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)。矩形 attention 需要与 KV 对齐的 lower-right causal mask，不能简单替换成 upper-left `is_causal=True`；见 [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial)。

## 正确性边界

严格 target-match 只保证匹配当次 block forward 的 argmax。BF16 中 block 与单 token 矩阵路径可能有不同舍入，真实全文与 greedy 已发现不一致。因此质量、首次分叉和 token 一致率必须随速度一起报告，不能称为已证明无损。

小模型 oracle 覆盖错误草稿、不同块长、EOS、长度预算和 KV 回退；图专项检查真实小 Qwen 的动态 KV 与静态 KV、跨桶和 reset。真实模型逐题比较 compact、T graph、cycle graph 的输出。测试结果保存在每个 run 内；测试通过不代表真实模型必然与 greedy 一致。

## 执行与复查

主入口为 `benchmark_inference_matrix.py`。每个运行保存完整命令（旁边 `.launcher.json`）、参数/源码及资产哈希（`manifest.json`）、逐题 JSONL、所有 token IDs、GPU 监控和 live summary。运行时会对所用 GPU 取得评测锁，不中断其他训练。源代码在启动或加载时快照到 run 内，复查应以该快照为准。

`--compact --t-graph --gpu-greedy` 打开 T graph 和较强 greedy；`--cycle-graph` 添加整段 T 草稿图；`--fused-decision` 添加严格决策 kernel。`--variants` 显式选择对照，必须包含同配置 greedy。

汇总入口：

```bash
/mnt/llmshared-ssd-hd/wangruitao/conda-envs/inception/bin/python \
  local_setup/analyze_acceleration_goal.py runs/tgraph_pilot_20260917 \
  runs/tgraph_validation_48_64_20260917 \
  runs/tgraph_fair_baseline_validation_20260917 \
  --out runs/acceleration_1p3_goal_20260917
```

汇总中的 CI 按题目配对 bootstrap，保留同题全部完整 repeats。报告明确区分 running 与 completed，开发题目小样本通过不等于目标最终完成。

## 新增对照与当前候选：混合草稿

`lookup_decode.py` 实现的是 **文本复用 + T 的混合路线**，不能称为 T-only 递归方法本身的 1.3×。同场保留三组：

1. T-only：每轮由冻结 T/boundary 生成最多 2 枚草稿；T graph + compact decoder。
2. 文本复用-only：当前上下文最后 2–5 枚 token 如果曾在 prompt 或已生成文本中出现，复制其后最多 8 枚已知 token 作为草稿；没有匹配时退回 target 单 token。它不运行 T。
3. 混合：有匹配时文本复用；没有匹配时由 T 生成草稿。草稿来源写入每个 block 的 `source`，接受数按来源分别统计。

所有草稿仍经过完整 target 严格前缀验证。复用只读当前题目的 prompt 和已经生成的文本，不读参考答案。连续复用时，真实 boundary hidden 暂存，下一次使用 T 前按顺序补齐 T 的真实上下文；不会把错误草稿 KV 当作真实前缀。临近 EOS 或预算耗尽时同样检查截止，不借此截短输出。

开发集 [0,4) ×2 的混合速度约 1.44×，对照 T-only 约 1.22×、复用-only 约 1.30×。它只是开发结果；固定方法的验证是 [64,80) ×3，另在 GPU0 用 [48,64) ×2 复测。最新状态和结果见 `runs/acceleration_1p3_goal_20260917/analysis_report.md`，不能把未完成汇总当最终结果。

运行固定方案及对照：

```bash
GPU=0 START=80 SAMPLES=16 REPEATS=3 \
  bash local_setup/benchmark_hybrid.sh
```

包装器默认采用新的区间 [80,96)，不自动重复开发区间。可通过 MODEL/BEFORE/AFTER/DATA/OUT/PYTHON 覆盖资产和路径。

探索但暂未纳入最佳方案：

- `draft_cycle_graph.py`：合并 T 同步、展开和 boundary，开发集净收益不明显。
- `strict_prefix_kernel.py`：精简严格接受判断，单独收益很小。
- `branched_decode.py`：多候选使接受数增加，但候选验证和复制 KV 的时间抵消收益。它是同题多条路径，不是多题吞吐 batch。
- `shortlist_boundary.py`：仅草稿头使用 GSM8K **train** 7473 题构建的 2048/4096/8192 token 词表，target 始终完整词表。训练 token 覆盖不等于生成时覆盖，也不意味着与完整草稿头一致；开发集没有净提速。该实验没有训练参数更新。

质量重评由 `rescore_numeric.py` 完成。旧规则遇到空 `Final Answer:` 标题和下一行 `\\boxed{...}` 可能误提取；报告保留旧 raw score，同时统一重评所有方法，记录数字提取字段和回退来源。不同浮点路径导致的真实文本差异仍保留，不用评分修正掩盖分叉。

配置审计注意：包装器复用上游神经 decoder 的默认参数，因此 settings 中可能仍有 `ngram_draft_mode=off`；本地 `lookup_decode.py` 的实际路线由 `lookup_only` 和每块 `source` 决定，不能仅凭上游这个未使用字段判为纯神经路线。后续入口额外写入 `local_execution_route`；已完成运行保持原始参数不回写。`lookuponly` 的实际 T 运行计数为零，它明确是单独的对照。
