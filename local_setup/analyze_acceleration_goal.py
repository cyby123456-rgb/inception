"""Source-linked, paired wall/throughput reporting. Repeats are not new questions."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from rescore_numeric import score


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);all_stats={};details=[];comparisons=[]
    for run in a.runs:
        if not (run/'results.jsonl').exists():continue
        rows=[(i+1,json.loads(line)) for i,line in enumerate((run/'results.jsonl').read_text().splitlines())]
        rows=[(i,r) for i,r in rows if r['phase']=='measurement']
        if not rows:continue
        status=json.load(open(run/'status.json'));manifest=json.load(open(run/'manifest.json'))
        stats={'status':status,'protocol':manifest['args'],'shared_gpu':manifest['shared_gpu'],'methods':{},'equivalence':{}}
        expected=manifest['args']['samples'];methods=list(rows[0][1]['values'])
        for method in methods:
            per=[]
            for line,r in rows:
                v=r['values'][method];b=r['values'][v['baseline']]
                dt=dict(run=str(run.resolve()),source=str((run/'results.jsonl').resolve()),line=line,sample=r['sample'],repeat=r['repeat'],method=method,
                    baseline=v['baseline'],seconds=v['wall_seconds'],baseline_seconds=b['wall_seconds'],tokens=len(v['token_ids']),baseline_tokens=len(b['token_ids']),
                    wall_speedup=b['wall_seconds']/v['wall_seconds'],throughput_speedup=len(v['token_ids'])*b['wall_seconds']/(len(b['token_ids'])*v['wall_seconds']),
                    answer_correct=v['answer_correct'],baseline_correct=b['answer_correct'],exact=v['token_ids']==b['token_ids'],capped=v['hit_token_cap'],
                    accepted_drafts=v.get('accepted_draft_tokens',0),draft_tokens=v.get('draft_tokens',0),target_calls=v['target_calls'],
                    graph_replays=v.get('t_graph_replays',0),strict_blocks=v.get('fast_strict_blocks',0),
                    neural_blocks=v.get('neural_blocks',sum(z['proposed_len']>1 for z in v.get('block_records',[]))),neural_blocks_saved='neural_blocks' in v,lookup_blocks=v.get('lookup_blocks',0),lookup_proposed=v.get('lookup_proposed',0),lookup_accepted=v.get('lookup_accepted',0),
                    neural_proposed=sum(z['proposed_len']-1 for z in v.get('block_records',[]) if z.get('source','neural')=='neural'),neural_accepted=sum(z['accepted_len']-1 for z in v.get('block_records',[]) if z.get('source','neural')=='neural'),
                    accepted_unchecked=v.get('accepted_unchecked_tokens',0),accepted_mismatched=v.get('accepted_mismatch_tokens',0))
                dt.update(score(v['text'],r['answer']));dt['baseline_numeric_correct']=score(b['text'],r['answer'])['numeric_correct']
                details.append(dt);per.append(dt)
            seconds=sum(x['seconds'] for x in per);bs=sum(x['baseline_seconds'] for x in per)
            tokens=sum(x['tokens'] for x in per);bt=sum(x['baseline_tokens'] for x in per)
            rep=[]
            for j in sorted({x['repeat'] for x in per}):
                rr=[x for x in per if x['repeat']==j];ss=sum(x['seconds'] for x in rr);bb=sum(x['baseline_seconds'] for x in rr)
                tt=sum(x['tokens'] for x in rr);tb=sum(x['baseline_tokens'] for x in rr)
                rep.append(dict(repeat=j+1,questions=len(rr),complete=len(rr)==expected,seconds=ss,baseline_seconds=bb,tokens=tt,baseline_tokens=tb,
                    wall=bb/ss,throughput=tt*bb/(tb*ss),correct=sum(x['answer_correct'] for x in rr),baseline_correct=sum(x['baseline_correct'] for x in rr),numeric_correct=sum(x['numeric_correct'] for x in rr),baseline_numeric_correct=sum(x['baseline_numeric_correct'] for x in rr)))
            # Cluster bootstrap uses complete repeats only; retain every repeat for each sampled question.
            complete={x['repeat']-1 for x in rep if x['complete']};rr=[x for x in per if x['repeat'] in complete]
            ci=None
            if rr:
                ids=sorted({x['sample'] for x in rr})
                mat=np.array([[sum(x[k] for x in rr if x['sample']==i) for k in ['seconds','baseline_seconds','tokens','baseline_tokens']] for i in ids])
                rng=np.random.default_rng(917);draw=mat[rng.integers(0,len(ids),(10000,len(ids)))].sum(1)
                ci={'unit':'question, all complete repeats retained together','draws':10000,'unique_questions':len(ids),
                    'wall_95':np.quantile(draw[:,1]/draw[:,0],[.025,.975]).tolist(),
                    'throughput_95':np.quantile(draw[:,2]*draw[:,1]/(draw[:,3]*draw[:,0]),[.025,.975]).tolist()}
            stats['methods'][method]=dict(measurements=len(per),questions=len({x['sample'] for x in per}),seconds=seconds,baseline_seconds=bs,
                tokens=tokens,baseline_tokens=bt,wall=bs/seconds,throughput=tokens*bs/(bt*seconds),correct=sum(x['answer_correct'] for x in per),
                baseline_correct=sum(x['baseline_correct'] for x in per),numeric_correct=sum(x['numeric_correct'] for x in per),baseline_numeric_correct=sum(x['baseline_numeric_correct'] for x in per),exact=sum(x['exact'] for x in per),capped=sum(x['capped'] for x in per),
                accepted_drafts=sum(x['accepted_drafts'] for x in per),draft_tokens=sum(x['draft_tokens'] for x in per),
                target_calls=sum(x['target_calls'] for x in per),graph_replays=sum(x['graph_replays'] for x in per),
                neural_blocks=sum(x['neural_blocks'] for x in per),lookup_blocks=sum(x['lookup_blocks'] for x in per),lookup_proposed=sum(x['lookup_proposed'] for x in per),lookup_accepted=sum(x['lookup_accepted'] for x in per),neural_proposed=sum(x['neural_proposed'] for x in per),neural_accepted=sum(x['neural_accepted'] for x in per),
                strict_blocks=sum(x['strict_blocks'] for x in per),unchecked=sum(x['accepted_unchecked'] for x in per),mismatched=sum(x['accepted_mismatched'] for x in per),
                repeats=rep,paired_question_bootstrap=ci)
        pairs=[]
        for m in methods:
            if m.endswith('_tgraph') and m.removesuffix('_tgraph') in methods:pairs.append((m.removesuffix('_tgraph'),m))
            if m.endswith('_cycle') and m.removesuffix('_cycle') in methods:pairs.append((m.removesuffix('_cycle'),m))
            if m.endswith('_fdecision') and m.removesuffix('_fdecision') in methods:pairs.append((m.removesuffix('_fdecision'),m))
        if 'greedy_legacy_lower_fused' in methods:pairs.append(('greedy_legacy_lower_fused','greedy_lower_fused'))
        for old,new in pairs:
            count=0
            for line,r in rows:
                x=r['values'][old]['token_ids'];y=r['values'][new]['token_ids'];eq=x==y;count+=eq
                first=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)) if len(x)!=len(y) else None)
                comparisons.append(dict(source=str((run/'results.jsonl').resolve()),line=line,sample=r['sample'],repeat=r['repeat'],old=old,new=new,exact=eq,first_divergence=first))
            stats['equivalence'][old+' -> '+new]=dict(exact=count,total=len(rows))
        all_stats[str(run.resolve())]=stats
    (a.out/'aggregate.json').write_text(json.dumps(all_stats,ensure_ascii=False,indent=2))
    for file,vals in [('per_question.csv',details),('optimization_equivalence.csv',comparisons)]:
        if vals:
            with (a.out/file).open('w') as f:
                w=csv.DictWriter(f,fieldnames=vals[0].keys());w.writeheader();w.writerows(vals)
    lines=['# 1.3× 推理优化实测记录','',
        '整段 wall speedup = 同题 greedy 总秒数 / 投机总秒数；吞吐比另行按 token/秒计算。加载、编译和预热排除；每题 prefill、T 初始化和生成包含。所有结果来自共享 GPU，不是独占发布级结论。',
        '', '严格验证指草稿必须匹配本次完整 target block forward 的 argmax。BF16 下块前向与串行 greedy 可能不同，不能由此宣称逐 token 无损。', '']
    for run,s in all_stats.items():
        lines += ['## '+Path(run).name,'',f"状态：{s['status']['status']}。原始逐题文件：[{run}/results.jsonl]({run}/results.jsonl)。参数和来源哈希：[manifest.json]({run}/manifest.json)。",'',
            '|方法|次数/题目|greedy 秒|方法秒|wall|吞吐|修正数值答对|greedy 修正答对|与 greedy 全文一致|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m,v in s['methods'].items():
            lines.append(f"|{m}|{v['measurements']}/{v['questions']}|{v['baseline_seconds']:.3f}|{v['seconds']:.3f}|{v['wall']:.4f}×|{v['throughput']:.4f}×|{v['numeric_correct']}|{v['baseline_numeric_correct']}|{v['exact']}/{v['measurements']}|")
        lines+=['','质量列为统一数值重评：修复空 Final Answer 标题截取错误；原始 raw answer_correct 仍保存在 JSONL、CSV 和 aggregate.json。数字来源（Answer 行、boxed 或末数字回退）逐条可查。重复运行不扩大独立题目数。逐轮结果、配对题目 bootstrap 区间和执行计数见 aggregate.json；所有关键数字可通过 per_question.csv 的绝对路径、行号和方法字段回溯。','']
        for pair,v in s['equivalence'].items():lines.append(f"- {pair}：{v['exact']}/{v['total']} 次 token 完全一致。")
        lines+=['']
    (a.out/'analysis_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
