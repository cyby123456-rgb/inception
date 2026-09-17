"""Compare checkpoint recipes on common questions/repeats and each target's greedy."""
import argparse
import json
from pathlib import Path

import numpy as np

from rescore_numeric import score


METHODS = {
    "greedy_lower_fused": "优化 greedy",
    "after_b3_lower_fused_compact_tgraph": "纯 T 投机",
    "after_b3_lower_fused_compact_tgraph_lookup": "混合：历史匹配 + T",
    "after_b3_lower_fused_compact_tgraph_lookuponly": "仅历史匹配",
}


def read(runs):
    result = {}
    for run in runs:
        assert json.loads((run / "status.json").read_text())["status"] == "completed"
        for line, text in enumerate((run / "results.jsonl").read_text().splitlines(), 1):
            row = json.loads(text)
            if row["phase"] != "measurement":
                continue
            key = (row["sample"], row["repeat"])
            assert key not in result, key
            result[key] = dict(row, source=str(run / "results.jsonl"), line=line)
    return result


def summarize(rows, keys, method):
    vals = [rows[k]["values"][method] for k in keys]
    base = [rows[k]["values"][v["baseline"]] for k, v in zip(keys, vals)]
    seconds = sum(v["wall_seconds"] for v in vals)
    baseline_seconds = sum(v["wall_seconds"] for v in base)
    tokens = sum(len(v["token_ids"]) for v in vals)
    baseline_tokens = sum(len(v["token_ids"]) for v in base)
    by_repeat = {}
    for rep in sorted({k[1] for k in keys}):
        selected = [(k, v) for k, v in zip(keys, vals) if k[1] == rep]
        by_repeat[str(rep + 1)] = {
            "correct": sum(score(v["text"], rows[k]["answer"])["numeric_correct"] for k, v in selected),
            "questions": len(selected),
        }
    sources = {}
    for v in vals:
        for block in v.get("block_records", []):
            src = block.get("source", "neural")
            agg = sources.setdefault(src, {"blocks": 0, "proposed": 0, "accepted": 0})
            agg["blocks"] += 1
            agg["proposed"] += block["proposed_len"] - 1
            agg["accepted"] += block["accepted_len"] - 1
    mat = []
    for sample in sorted({k[0] for k in keys}):
        ix = [i for i, k in enumerate(keys) if k[0] == sample]
        mat.append([sum(vals[i]["wall_seconds"] for i in ix),
                    sum(base[i]["wall_seconds"] for i in ix),
                    sum(len(vals[i]["token_ids"]) for i in ix),
                    sum(len(base[i]["token_ids"]) for i in ix)])
    mat = np.asarray(mat)
    draws = mat[np.random.default_rng(918).integers(0, len(mat), (10000, len(mat)))].sum(1)
    return dict(seconds=seconds, baseline_seconds=baseline_seconds, tokens=tokens,
                baseline_tokens=baseline_tokens, wall_speedup=baseline_seconds / seconds,
                throughput_speedup=(tokens / seconds) / (baseline_tokens / baseline_seconds),
                wall_ci95=np.quantile(draws[:, 1] / draws[:, 0], [.025, .975]).tolist(),
                throughput_ci95=np.quantile(draws[:, 2] * draws[:, 1] / (draws[:, 3] * draws[:, 0]), [.025, .975]).tolist(),
                exact=sum(v["token_ids"] == b["token_ids"] for v, b in zip(vals, base)),
                capped=sum(v["hit_token_cap"] for v in vals), numeric_accuracy_by_repeat=by_repeat,
                sources=sources, target_calls=sum(v["target_calls"] for v in vals),
                unchecked=sum(v.get("accepted_unchecked_tokens", 0) for v in vals),
                mismatched=sum(v.get("accepted_mismatch_tokens", 0) for v in vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--old", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    new, old = read([a.new]), read(a.old)
    keys = sorted(set(new) & set(old))
    assert keys and set(keys) == set(new), "Require old results for every new question/repeat"
    assert all(new[k]["question"] == old[k]["question"] and new[k]["answer"] == old[k]["answer"] for k in keys)
    a.out.mkdir(parents=True, exist_ok=True)
    results = {label: {method: summarize(rows, keys, method) for method in METHODS}
               for label, rows in [("joint_from_base_22000", new), ("staged60375_then_joint2000", old)]}
    provenance = {label: [{"sample": k[0], "repeat": k[1], "source": rows[k]["source"], "line": rows[k]["line"]} for k in keys]
                  for label, rows in [("joint_from_base_22000", new), ("staged60375_then_joint2000", old)]}
    (a.out / "checkpoint_comparison.json").write_text(json.dumps(dict(results=results, provenance=provenance), ensure_ascii=False, indent=2))
    manifest = json.loads((a.new / "manifest.json").read_text())
    lines = ["# 从头联合训练 checkpoint 的最佳推理方案复测", "",
             "使用从第 1 步同时训练 T 和 boundary 的 22,000 步固定快照；计划 49,375 步，训练尚未完成。目标 Qwen3-8B 冻结，目标 LoRA B 保持为零。",
             f"Checkpoint：`{manifest['args']['after']}`。", "",
             "GSM8K 测试集第 48–79 条，共 32 题，每题重复 2 次；BF16、关闭 thinking、prompt 上限 1024、输出上限 1024。各方法轮换顺序，每轮预热排除。整段计时包含 prefill、每题 T 初始化及生成，排除加载、编译和预热。",
             "混合方案优先匹配 prompt/已生成历史，最多提出 8 枚草稿；无匹配时由 T/boundary 提出最多 2 枚。完整目标模型严格验证连续前缀。T 使用静态缓存和 CUDA Graph，目标使用合并 LoRA、融合 RMSNorm 及 lower-right causal attention。greedy 使用同等级目标优化。", "",
             "旧实验为分阶段训练 60,375 步后再联合训练 2,000 步。两组目标权重不同；此表各自除以自己的 greedy。旧实验也仅取相同 32 题的前两轮，不能把差异单独归因于训练方式或训练步数。", "",
             "|Checkpoint|方法|总秒数|输出 token|整段加速比|吞吐加速比|数值答对（两轮分别）|全文一致/64|触及上限/64|",
             "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for label, methods in results.items():
        for method, v in methods.items():
            quality = "、".join(f"{x['correct']}/{x['questions']}" for x in v["numeric_accuracy_by_repeat"].values())
            lines.append(f"|{label}|{METHODS[method]}|{v['seconds']:.3f}|{v['tokens']}|{v['wall_speedup']:.4f}×|{v['throughput_speedup']:.4f}×|{quality}|{v['exact']}/64|{v['capped']}/64|")
    lines += ["", "加速比为总时间比；吞吐另按总 token / 总时间计算。全文一致指与各自 greedy 的 token 序列一致。数值答案使用统一重评，原始评分仍保留。两轮不是 64 道独立题。", "",
              "## 新 checkpoint 的接受情况", "", "|方法|草稿来源|验证块数|提出草稿数|接受草稿数|平均每块接受|", "|---|---|---:|---:|---:|---:|"]
    for method, v in results["joint_from_base_22000"].items():
        for source, s in v["sources"].items():
            lines.append(f"|{METHODS[method]}|{source}|{s['blocks']}|{s['proposed']}|{s['accepted']}|{s['accepted']/s['blocks']:.4f}|")
    hybrid = results["joint_from_base_22000"]["after_b3_lower_fused_compact_tgraph_lookup"]
    lines += ["", f"混合方案按题目聚类 bootstrap 的 95% 区间：整段 {hybrid['wall_ci95']}；吞吐 {hybrid['throughput_ci95']}。这些区间只描述本批题目的抽样不确定性，不能消除 GPU 并发干扰。", "",
              "全部结果来自共享 GPU；不是独占卡最终速度。BF16 块前向与串行前向的舍入可能造成生成分叉，严格草稿匹配不代表已证明与 greedy 逐 token 等价。输出长度变化会影响整段耗时，不删除触及上限的题目。", "",
              f"原始逐题数据：[{a.new / 'results.jsonl'}]({a.new / 'results.jsonl'})；源码/权重/数据哈希：[manifest.json]({a.new / 'manifest.json'})。",
              "逐题行号、训练方案比较和置信区间见 checkpoint_comparison.json；统一重评分、原始评分、各来源接受计数及每题速度见 per_question.csv、aggregate.json。"]
    (a.out / "checkpoint_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
