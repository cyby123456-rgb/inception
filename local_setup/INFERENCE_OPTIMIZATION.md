# 推理侧优化：2026-09-17

工作目录：`/mnt/llmshared-ssd-hd/wangruitao/inception-inference-opt`。
基于 GitHub commit `31dc971` 的独立 worktree；原 `inception`、`inception-joint` 的训练和测速继续运行。没有修改 checkpoint、训练 loss 或已安装依赖。

## 实现内容

1. `benchmark_inference_matrix.py`：同一模型进程轮换 greedy 与联合训练前/后的投机方法；比较 BLOCK=2/3/4、prefill 最后位置读出、下面两项新优化。支持 target LoRA 合并或不合并，每种情况使用自己的同等级 greedy。保留逐题 token、答案、接受前缀、实际路径计数、warmup、资产哈希和 GPU 并发记录。
2. `lower_right_target.py`：在 batch-1、无 padding、连续 KV、full-attention Qwen3 的多 token target forward 中，使用 PyTorch lower-right causal bias。避免把矩形 SDPA 的 upper-left `is_causal=True` 错当成缓存验证所需的 mask。当前仍生成原始 mask，主要改变 attention kernel 路径，没有实现静态 KV 或跨样本 batching。单 token、prefill、T attention 沿用原路径。
3. `fused_rmsnorm.py`：Triton 实现 Qwen3 RMSNorm，融合类型转换、平方、均值、rsqrt 和乘法。保留 Qwen3 的“先把归一化结果转回输入 dtype，再乘权重”的顺序；归约顺序仍可能导致浮点差异。对 target 和 T 都应用；对应 greedy 也使用同样的 target 融合。只支持推理，不支持 backward。
4. `benchmark_target_forward.py`：固定输入和同一份 prefix KV，分别测 query 长度 1/2/3/4、prefix 长度 128/512/1024，排除生成轨迹变化对时间的影响。

这些开关是独立实验入口，不会自动改变公共 `run_decode.py`。上下文安装按单线程离线推理设计，不应直接用于并发服务。

## 当前已经完成的证据

固定输入实验：`runs/target_forward_merged_20260917/{manifest.json,results.jsonl,summary.json}`。
Qwen3-8B，BF16，SDPA，target LoRA merged，三个 prefix 长度，每种 query 长度/方法各 12 次；另有两轮不计入的预热。以下对三个 prefix 等权汇总，因为每个 prefix 重复次数相同：

| 前向类型 | 原路径平均 ms | lower-right + RMSNorm 融合 ms | 前向加速 |
|---|---:|---:|---:|
| query=1 | 27.460 | 23.059 | 1.191× |
| query=2 | 29.471 | 23.957 | 1.230× |
| query=3 | 29.340 | 23.566 | 1.245× |
| query=4 | 30.048 | 24.006 | 1.252× |

这是固定状态 target forward 的结果，**不是整段投机相对 greedy 的加速比**。上下文安装和回退到固定 cache 长度在计时外；T 起草、完整 rollout 等不包含在内。query=1 同样受益，所以不能将 query=3 的 1.245× 直接乘到历史投机加速比。

`pilot_merged_v2_20260917`、`pilot_lower_right_20260917`、`pilot_fused_20260917` 各有 2 道真实题的小试：BLOCK=2 比 BLOCK=3/4 更快；prefill 最后位置 logits 未显示可靠的端到端收益。融合后最好的小试为 after_b2_fused，相对同样融合的 greedy，wall/throughput 都约 1.056×。不同 pilot 分别运行、baseline 有漂移，不能跨 pilot 直接比较绝对秒数后归因。

BF16 数值路径改变会改变输出：lower-right 小试中 before/after 两个 checkpoint 的优化前后完整输出一致均为 0/2；RMSNorm 小试中 greedy 优化前后为 1/2，after_b2 和 after_b3 优化前后均为 0/2。不能把这些新路径称为已验证逐 token 无损。各自仍执行严格的当前 target argmax 前缀匹配。

完整小规模重复对照：`runs/inference_confirm_merged_20260917`，4 道题 × 2 repeat，11 方法；未合并 target 的控制：`runs/inference_unmerged_control_20260917`，2 道题 × 1 repeat。是否完成以各自 `status.json` 为准，结果见 `summary.json`，原始数据见 `results.jsonl`。

4 题 × 2 repeat 对照已经完成：联合训练前最好的组合为 BLOCK=2 + lower-right + fused，wall 1.0149×、吞吐 1.0171×；联合训练后为 BLOCK=3 + lower-right + fused，wall 1.0883×、吞吐 1.0956×。各方法答案正确次数都是 4/8（即每轮 2/4 题），但这不足以证明质量无变化；最好的联合训练后方法与同优化 greedy 完整一致为 4/8。

在同一重复对照中，联合训练后的 BLOCK=2 原路径耗时 49.055 秒，RMSNorm 融合后 39.359 秒；greedy 同时从 51.822 秒降到 41.531 秒，所以相对 wall 加速比从 1.0564× 变成 1.0552×，几乎没有增加。这是绝对耗时改善与投机相对收益必须分开看的直接例子。

未合并 target 的 2 题控制也已完成：联合训练后 BLOCK=2 原路径吞吐加速 1.0199×，融合后约 1.0000×，再加 lower-right 为 0.9968×。这组没有显示相对收益，不能将已合并配置的改善推广到未合并配置。

扩展验证已启动：`runs/inference_validation_32_48_20260917`，本轮未用于选参的题目区间 [32,48)，16 题 × 2 repeat；联合训练前后分别比较原 BLOCK=3 和优化 BLOCK=2/3，所有方法保留匹配 greedy。在 GPU 2 执行，不能提前填入结果。汇总入口 `summarize_inference_optimizations.py` 只把完成运行加入结果表。

所有 GPU 实验均为共享 GPU 2 的探索，不能当独占正式测速。4 题重复 2 次仍是 4 道独立题，不是 8 道独立题。

## 正确性和测试

- GitHub 原有 66 项回归在本机全部通过；新增 prefill 和 lower-right CPU/小模型检查后，70 项全部通过。
- Prefill 检查真实小 Qwen3：只返回最后 logits，但保留完整 prefix KV 和 hidden；随后验证块仍返回各位置 logits。
- Lower-right 检查矩形 mask、未来 token 隔离、真实小 Qwen3 的 KV 前向。GPU 额外校验与 FP32 math attention 的误差和未来 token 隔离，记录在运行目录的 `lower_right_numerical_validation.json`。
- 融合 RMSNorm 有 24 组 GPU 数值检查（BF16/FP16/FP32，宽度 128/4096，多种行数），记录在 `fused_norm_numerical_validation.json`。这只是局部容差检查，不替代真实模型输出一致性。
- 新 loader 根据仓库的 adapter-only 保存规则核对所有 T/边界字段，拒绝漏载训练参数；冻结 base weights 按设计来自官方基座，不要求它们出现在 adapter checkpoint 中。
- 每个实际投机结果断言没有 unchecked/mismatched accepted commits，且 fast strict 覆盖所有 block。

## 重跑示例

使用本机已配置环境；不需要安装 flash-attn 或升级 PyTorch：

```bash
cd /mnt/llmshared-ssd-hd/wangruitao/inception-inference-opt
CUDA_VISIBLE_DEVICES=2 \
/mnt/llmshared-ssd-hd/wangruitao/conda-envs/inception/bin/python \
  local_setup/benchmark_inference_matrix.py \
  --output runs/my_inference_comparison \
  --samples 8 --repeats 3 --max-new-tokens 1024 \
  --merge-target --lower-right --fused-norms \
  --variants greedy,greedy_fused,greedy_lower_fused,before_b2,before_b2_fused,before_b2_lower_fused,after_b2,after_b2_fused,after_b2_lower_fused
```

省略 `--merge-target` 得到未合并控制。可用 `--before/--after/--model/--data/--start` 替换资产和数据区间。每个方法必须包含匹配的 greedy baseline，否则入口拒绝执行。

固定输入验证前向：

```bash
CUDA_VISIBLE_DEVICES=2 \
/mnt/llmshared-ssd-hd/wangruitao/conda-envs/inception/bin/python \
  local_setup/benchmark_target_forward.py \
  --output runs/my_fixed_target_forward --merge-target --iterations 12
```

同一 GPU 上这些入口通过已有 evaluation lock 排队。目录不可覆盖，加载、编译和 warmup 不计入端到端生成统计。原始 JSON 中异步 component timings 仍是 host dispatch 耗时，不能作为 GPU kernel 时间相加。

## 后续决策

优先看完整重复结果中 BLOCK=2、融合及其组合相对同等级 greedy 的吞吐。对存在浮点分叉的路径，同时看输出长度、答案和首次分叉；不能只看 wall 秒数。配置选择完成后应在未参与选参的问题区间与独占 GPU 上确认。

若相对收益依然很小，下一步应针对剩余 T 同步和验证的 CPU 调度/缓存分配做固定形状执行优化，而非继续叠加统计开关。当前没有证据承诺端到端 1.3×。

实现参考：[Triton 官方归一化融合教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)介绍了行级归约与融合的实现方式；本地 RMSNorm 的定义依据安装版 `transformers/models/qwen3/modeling_qwen3.py`，不是把 LayerNorm 公式直接用于 RMSNorm。Lower-right 调度依据本机 PyTorch 2.4.1 的 `torch/nn/attention/bias.py` 源码。
