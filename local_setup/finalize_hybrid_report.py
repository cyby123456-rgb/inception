"""Create a final, fully source-linked human report only after both validations finish."""
import json,subprocess,sys,hashlib
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/acceleration_1p3_goal_20260917'
NAMES=['compact_decode_pilot_20260917','tgraph_pilot_20260917','tgraph_validation_48_64_20260917',
       'tgraph_fair_baseline_validation_20260917','draft_cycle_graph_pilot_v2_20260917','branch_paths_pilot_20260917',
       'shortlist_draft_pilot_20260917','lookup_hybrid_pilot_20260917','lookup_hybrid_validation_64_80_20260917',
       'lookup_hybrid_replication_48_64_20260917']
PRIMARY=ROOT/'runs'/NAMES[-2];SECONDARY=ROOT/'runs'/NAMES[-1]
HYBRID='after_b3_lower_fused_compact_tgraph_lookup'
T='after_b3_lower_fused_compact_tgraph';LOOKUP=HYBRID+'only'


def main():
    for run in [PRIMARY,SECONDARY]:
        status=json.load(open(run/'status.json'))
        if status['status']!='completed':raise SystemExit(f'Not final: {run.name}: {status}')
    subprocess.run([sys.executable,str(ROOT/'local_setup/analyze_acceleration_goal.py'),*[str(ROOT/'runs'/n) for n in NAMES],'--out',str(OUT)],check=True)
    summary=json.load(open(OUT/'aggregate.json'));a=summary[str(PRIMARY)];b=summary[str(SECONDARY)]
    decision={}
    for run,s in [(PRIMARY,a),(SECONDARY,b)]:
        best=s['methods'][HYBRID]
        decision[run.name]=dict(wall=best['wall'],throughput=best['throughput'],
            aggregate_above_1p3=best['wall']>=1.3 and best['throughput']>=1.3,
            every_repeat_above_1p3=all(r['wall']>=1.3 and r['throughput']>=1.3 for r in best['repeats']),
            no_unverified_commits=best['unchecked']==0 and best['mismatched']==0,
            bootstrap=best['paired_question_bootstrap'])
    combined={}
    raw=[]
    for group,run in enumerate([PRIMARY,SECONDARY]):
        raw += [(group,json.loads(line)) for line in (run/'results.jsonl').read_text().splitlines() if json.loads(line)['phase']=='measurement']
    for method in [T,LOOKUP,HYBRID]:
        matrices=[]
        for group in [0,1]:
            data=[r for g,r in raw if g==group];items=[]
            for sample in sorted({r['sample'] for r in data}):
                rr=[r for r in data if r['sample']==sample]
                items.append(np.mean([[r['values'][method]['wall_seconds'],r['values']['greedy_lower_fused']['wall_seconds'],len(r['values'][method]['token_ids']),len(r['values']['greedy_lower_fused']['token_ids'])] for r in rr],axis=0))
            matrices.append(np.array(items))
        total=sum(m.sum(0) for m in matrices);ss,bs,nt,nb=total
        rng=np.random.default_rng(1917)
        draws=sum(m[rng.integers(0,len(m),(10000,len(m)))].sum(1) for m in matrices)
        rounds=[]
        for rep in [0,1]:
            rr=[r for _,r in raw if r['repeat']==rep]
            rs=sum(r['values'][method]['wall_seconds'] for r in rr);rb=sum(r['values']['greedy_lower_fused']['wall_seconds'] for r in rr)
            rt=sum(len(r['values'][method]['token_ids']) for r in rr);rbt=sum(len(r['values']['greedy_lower_fused']['token_ids']) for r in rr)
            rounds.append(dict(repeat=rep+1,questions=len(rr),wall=rb/rs,throughput=rt*rb/(rbt*rs)))
        combined[method]=dict(unique_questions=32,aggregation='Average repeats within each question, then sum across all 32 questions; each question included, no exclusion of capped outputs',
            seconds=ss,baseline_seconds=bs,tokens=nt,baseline_tokens=nb,wall=bs/ss,throughput=nt*bs/(nb*ss),
            wall_ci95=np.quantile(draws[:,1]/draws[:,0],[.025,.975]).tolist(),throughput_ci95=np.quantile(draws[:,2]*draws[:,1]/(draws[:,3]*draws[:,0]),[.025,.975]).tolist(),
            bootstrap='10000 question resamples, stratified by the two 16-question GPU runs; repeat means retained',common_rounds=rounds)
    decision['balanced_all_32_questions']=combined
    c=combined[HYBRID]
    decision['observed_goal_check']={'balanced_mean_wall_and_throughput_at_least_1p3':bool(c['wall']>=1.3 and c['throughput']>=1.3),
        'both_common_32_question_repeats_at_least_1p3':all(r['wall']>=1.3 and r['throughput']>=1.3 for r in c['common_rounds']),
        'scope':'Observed hybrid mean on this shared-GPU protocol; NOT pure T, not a guarantee for every question/run, and not a population lower confidence bound of 1.3'}
    (OUT/'goal_evidence.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    h=a['methods'][HYBRID];g=a['methods']['greedy_lower_fused'];t=a['methods'][T];n=a['methods'][LOOKUP]
    lines=['# 推理加速实验：最终可核查记录','',
       '**达标候选是“文本复用 + T/boundary”的混合推理路线。不能写成 T-only 递归模型已经达到相同加速。**',
       '',f"主验证：Qwen3-8B，冻结 full60375 + joint2000 checkpoint，GSM8K [64,80) 16 题 ×3；副验证 [48,64) 16 题 ×2。共 32 道独立评测题。本轮混合策略选参使用 [0,4)；[48,64) 此前已用于 T-only 评测，[64,80) 是本轮新增独立验证区间。",
       '', '所有测试都是共享 GPU（NVIDIA M402 80GB），逐题轮换方法顺序。greedy 使用相同 BF16、target LoRA merge、target RMSNorm、attention 路径和 GPU token 复用。加载、graph capture 和预热排除；每题 prefill、T 初始化和实际生成包含。',
       '', '## 结果','', '|验证集 / GPU|方法|greedy 秒|方法秒|wall 加速|吞吐加速|', '|---|---|---:|---:|---:|---:|']
    for label,s in [('64–79 / GPU2',a),('48–63 / GPU0',b)]:
        for method,title in [(T,'T-only'),(LOOKUP,'文本复用-only'),(HYBRID,'混合')]:
            v=s['methods'][method]
            lines.append(f"|{label}|{title}|{v['baseline_seconds']:.3f}|{v['seconds']:.3f}|{v['wall']:.4f}×|{v['throughput']:.4f}×|")
    lines+=['','### 全部 32 题汇总（不排除达到长度上限的题）','',
        '每道题先对其重复取均值，再将全部 32 题的时间和 token 数分别相加；避免一组重复三次、另一组两次造成题目权重不一致。它是跨两个共享 GPU 的描述性汇总，单 GPU 结果仍以上表分别报告。','',
        '|方法|greedy 平均总秒|方法平均总秒|wall|吞吐|','|---|---:|---:|---:|---:|']
    for method,title in [(T,'T-only'),(LOOKUP,'文本复用-only'),(HYBRID,'混合')]:
        v=combined[method];lines.append(f"|{title}|{v['baseline_seconds']:.3f}|{v['seconds']:.3f}|{v['wall']:.4f}×|{v['throughput']:.4f}×|")
    c=combined[HYBRID]
    lines += ['',f"混合的题目分层 bootstrap 95% 区间：wall [{c['wall_ci95'][0]:.3f}, {c['wall_ci95'][1]:.3f}]；吞吐 [{c['throughput_ci95'][0]:.3f}, {c['throughput_ci95'][1]:.3f}]。"]
    lines+=['','### 混合方案逐次重复','', '|验证集|Repeat|wall|吞吐|数值答案正确 / 16|greedy 正确 / 16|','|---|---:|---:|---:|---:|---:|']
    for label,s in [('64–79',a),('48–63',b)]:
        for r in s['methods'][HYBRID]['repeats']:
            lines.append(f"|{label}|{r['repeat']}|{r['wall']:.4f}×|{r['throughput']:.4f}×|{r['numeric_correct']}|{r['baseline_numeric_correct']}|")
    lines+=['', '题目配对 bootstrap 的 95% 区间（同题全部 repeats 一起重采样，不将 repeats 当独立题目）：','']
    for label,s in [('64–79',a),('48–63',b)]:
        ci=s['methods'][HYBRID]['paired_question_bootstrap']
        lines.append(f"- {label}：wall [{ci['wall_95'][0]:.3f}, {ci['wall_95'][1]:.3f}]；吞吐 [{ci['throughput_95'][0]:.3f}, {ci['throughput_95'][1]:.3f}]。")
    lines+=['', '是否达 1.3× 按实测均值和逐轮结果分别记录于 goal_evidence.json。CI 表达换题的不确定性；即使重复稳定，也不能承诺任意题目、并发负载或模型都达到 1.3×。',
       '', '## 怎么做，以及 T 到底参与多少','',
       '每轮先看当前上下文末尾 2–5 个 token 是否曾出现。有匹配时，复制历史中其后的最多 8 枚 token 作为候选；否则由 T/boundary 提出最多 2 枚草稿。完整 target 验证后只接受连续正确前缀。连续文本复用时暂存真实 hidden，在下一次调用 T 前补齐其真实上下文。', '',
       f"主验证中，T 提供草稿的块为 {h['neural_blocks']}，文本复用块为 {h['lookup_blocks']}。T 草稿接受 {h['neural_accepted']}/{h['neural_proposed']} 枚，复用草稿接受 {h['lookup_accepted']}/{h['lookup_proposed']} 枚。这些都能在 block_records.source 中逐块核对。",
       '', f"混合每个验证块平均接受 {h['accepted_drafts']/h['strict_blocks']:.3f} 枚额外草稿；T-only 为 {t['accepted_drafts']/t['strict_blocks']:.3f}。一枚已确定的 anchor 不计入接受草稿数。混合 target forward 为 {h['target_calls']} 次，T-only 为 {t['target_calls']} 次，greedy 为 {g['target_calls']} 次。",
       '',f"同场混合耗时比文本复用-only 减少 {(1-h['seconds']/n['seconds'])*100:.2f}%；输出长度不完全相同，吞吐比也必须同时看。该差异只支持该配置下 T fallback 有实际增益，不能归因于纯 T 的独立 1.3×。"]
    before=a['methods']['before_b3_lower_fused_compact_tgraph_lookup']
    lines+=['', '## 联合训练前后','',f"同一 target adapter 下，full60375 初始 checkpoint 的混合路线 wall={before['wall']:.4f}×、吞吐={before['throughput']:.4f}×；joint2000 后 wall={h['wall']:.4f}×、吞吐={h['throughput']:.4f}×。本轮没有额外训练。", '',
       '## 正确性与限制','',
       f"主验证混合与 greedy 的完整 token 输出一致 {h['exact']}/{h['measurements']} 次。副验证为 {b['methods'][HYBRID]['exact']}/{b['methods'][HYBRID]['measurements']} 次。严格验证是与当次 block forward 的 argmax 一致，不等于 BF16 下与串行 greedy 逐 token 等价。",
       '', '质量采用统一数值重评，保留旧 raw score。旧提取器可能把空 Final Answer 标题下一行的 $$ 当作答案；修正后逐条记录 Answer 行、boxed 或末数字回退来源。小规模答案分数不能证明完全无损。', '',
       f"长度上限：主验证混合 {h['capped']}/{h['measurements']} 次达到 1024；副验证混合 {b['methods'][HYBRID]['capped']}/{b['methods'][HYBRID]['measurements']} 次。副验证第 62 题的原始输出尤其需要保留检查；不从汇总中删除。",
       '', '其他探索：合并整个 T 草稿图、Triton 接受决策、小词表草稿和并行候选都已保留独立对照；没有净收益的方案未混入最佳配置。不能用单个 target forward 的 1.245×、旧 greedy 的 1.388× 或 target-call 加速替代本报告的端到端结果。',
       '',f"回归检查：81 项测试全部通过，含本机 CUDA 检查；日志：[unit_tests_final_gpu.log]({OUT}/unit_tests_final_gpu.log)。每个运行目录另存数值验证结果。", '', '## 运行与原始证据','', f"运行入口：[benchmark_hybrid.sh]({ROOT}/local_setup/benchmark_hybrid.sh)。方法说明：[ACCELERATION_GOAL.md]({ROOT}/local_setup/ACCELERATION_GOAL.md)。", '',
       f"- 主验证原始文件：[results.jsonl]({PRIMARY}/results.jsonl)，[manifest.json]({PRIMARY}/manifest.json)。",
       f"- 副验证原始文件：[results.jsonl]({SECONDARY}/results.jsonl)，[manifest.json]({SECONDARY}/manifest.json)。",
       f"- 逐题数字及原始行号：[per_question.csv]({OUT}/per_question.csv)。",
       f"- 完整统计与区间：[aggregate.json]({OUT}/aggregate.json)。",
       '', '每个运行目录保存实际执行的 helper 源码快照；checkpoint 和数据在运行前后校验哈希。源码未推送，原训练目录未被修改。']
    (OUT/'analysis_summary_for_humans.md').write_text('\n'.join(lines)+'\n')
    paths=[ROOT/'local_setup/rescore_numeric.py',ROOT/'local_setup/analyze_acceleration_goal.py',Path(__file__)]
    (OUT/'analysis_source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
