"""Read-only result aggregation with token divergence and explicit evidence scope."""
import csv
import json
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/inference_optimization_report_20260917'


def divergence(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b)) if len(a)!=len(b) else None


def main():
    OUT.mkdir(exist_ok=True)
    lines=['# 推理优化实测汇总','',f'更新：{datetime.now().isoformat(timespec="seconds")}（服务器本地时间）','',
        '模型：Qwen3-8B；core36000 分支，联合训练前 warmup/checkpoints 与联合训练后 joint/checkpoints；BF16。所有结果是共享 GPU 2 的探索，尚非独占正式结果。',
        '每种投机与使用同样目标优化的 greedy 比较。测试题参与了开发选参；重复次数不增加独立题目数。','',
        '## 固定输入的 target forward','',
        '固定同一份 prefix KV；长度 128/512/1024，每个形状/方法 12 次。下面平均覆盖三个 prefix。排除了生成内容变化，但不包含完整草稿/T 同步/输出流程。',
        '| query tokens | reference ms | lower-right ms | fused RMS ms | combined ms | combined 前向加速 |',
        '|---:|---:|---:|---:|---:|---:|']
    micro=ROOT/'runs/target_forward_merged_20260917/summary.json'
    if micro.exists():
        x=json.loads(micro.read_text())
        for q in [1,2,3,4]:
            vals={}
            for variant in ['reference','lower','fused','lower_fused']:
                v=[r['mean_ms'] for r in x if r['query_tokens']==q and r['variant']==variant]
                vals[variant]=sum(v)/len(v)
            lines.append(f"| {q} | {vals['reference']:.3f} | {vals['lower']:.3f} | {vals['fused']:.3f} | {vals['lower_fused']:.3f} | {vals['reference']/vals['lower_fused']:.4f}× |")
        lines += ['',f'来源：[{micro.name}]({micro})。query=1 同样加速，不能将 query=3 的前向比值写成投机相对 greedy 的加速。','']
    csvrows=[]
    names=['pilot_merged_v2_20260917','pilot_lower_right_20260917','pilot_fused_20260917','inference_confirm_merged_20260917','inference_unmerged_control_20260917','inference_validation_32_48_20260917']
    for name in names:
        run=ROOT/'runs'/name
        if not (run/'status.json').exists():continue
        status=json.loads((run/'status.json').read_text())
        lines += ['## '+name,'',f'状态：`{status.get("status")}`。来源：[{name}]({run})。']
        if status.get('status')!='completed':
            lines.append('运行未完成，本报告暂不把部分题目混入完成结果。');continue
        rows=[json.loads(l) for l in (run/'results.jsonl').read_text().splitlines() if l.strip()]
        rows=[r for r in rows if r['phase']=='measurement']
        summary=json.loads((run/'summary.json').read_text())
        lines += ['',f'独立题目 {len({r["sample"] for r in rows})}；每方法生成次数 {len(rows)}。',
            '| 方法 | 总秒数 | tokens | wall 加速 | 吞吐加速 | 答对次数 | 与同优化 greedy 全等 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for method,s in summary.items():
            if 'seconds' not in s:continue
            lines.append(f"| {method} | {s['seconds']:.3f} | {s['tokens']} | {s['wall_speedup']:.4f}× | {s['throughput_speedup']:.4f}× | {s['answer_correct']}/{s['measurements']} | {s['greedy_exact']}/{s['measurements']} |")
        for row in rows:
            for method,val in row['values'].items():
                baseline=val.get('baseline','greedy_last' if method.endswith('_last') else 'greedy')
                base=row['values'][baseline]
                original=method.removesuffix('_fused').removesuffix('_lower')
                original_val=row['values'].get(original)
                csvrows.append(dict(run=name,sample=row['sample'],repeat=row['repeat'],method=method,
                    baseline=baseline,wall_seconds=val['wall_seconds'],tokens=len(val['token_ids']),
                    answer_correct=val['answer_correct'],first_divergence_vs_greedy=divergence(val['token_ids'],base['token_ids']),
                    unoptimized_method=original if original_val else '',
                    first_divergence_vs_unoptimized=divergence(val['token_ids'],original_val['token_ids']) if original_val else '',
                    target_calls=val['target_calls'],accepted_drafts=val.get('accepted_draft_tokens',0),
                    draft_tokens=val.get('draft_tokens',0),source=str(run/'results.jsonl')))
        eq=summary.get('optimization_equivalence',summary.get('prefill_optimization_equivalence'))
        lines += ['',f'优化前后输出一致统计（不能与 speculative-vs-greedy 混淆）：`{json.dumps(eq,ensure_ascii=False)}`。','']
    if csvrows:
        with (OUT/'per_question_divergence.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines += ['## 解释边界','',
        '- RMSNorm 融合与 lower-right attention 改变浮点执行路径；部分真实输出发生分叉，不能宣称逐 token 无损。',
        '- 固定输入实验的速度改善不能完全解释长文本生成的速度改善，因为生成长度、token 内容和后续接受情况可能变化。',
        '- 本次端到端计时起止 CUDA 同步；组件字段是异步 host 时间，不作为 kernel GPU 占比。',
        '- 原 checkpoint、训练 loss、已有实验未修改。未提供端到端达到 1.3× 的保证。',
        '- 实现、协议与复跑命令见 [INFERENCE_OPTIMIZATION.md](../../local_setup/INFERENCE_OPTIMIZATION.md)。']
    (OUT/'analysis_report.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'analysis_report.md')


if __name__=='__main__':main()
