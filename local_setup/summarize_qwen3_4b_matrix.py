"""Summarize per-seed trained-target speedups, then mean/sample SD across seeds."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(suite):
    plan = json.loads((suite/'plan.json').read_text())
    rows, pending = [], []
    for exp in plan['experiments']:
        home = suite/exp['arm']/f"seed_{exp['seed']}"
        endpoints = ['final', 'final_A_budget'] if exp['arm']=='full_joint' else ['final']
        for endpoint in endpoints:
            path = home/'evaluations'/endpoint
            aggregate = path/'analysis/aggregate.json'
            if not aggregate.exists():
                pending.append(f"{exp['arm']}/seed_{exp['seed']}/{endpoint}")
                continue
            manifest = json.loads((path/'manifest.json').read_text())
            audit = json.loads((path/'load_audit.json').read_text())['target']
            assert manifest['greedy_target_checkpoint'] == manifest['speculative_target_checkpoint'] == audit['checkpoint']
            assert manifest['target_adapter_sha256'] == audit['adapter_sha256']
            assert audit['exact_after_dtype_cast'] and audit['greedy_and_speculative_share_model_instance']
            stats = json.loads(aggregate.read_text())[str(path.resolve())]
            assert stats['status']['status']=='completed'
            for method, m in stats['methods'].items():
                assert all(r['complete'] for r in m['repeats'])
                rows.append(dict(arm=exp['arm'], seed=exp['seed'], endpoint=endpoint, method=method,
                    wall_speedup=m['wall'], token_throughput_speedup=m['throughput'],
                    greedy_seconds=m['baseline_seconds'], seconds=m['seconds'], tokens=m['tokens'],
                    numeric_accuracy=m['numeric_correct']/m['measurements'],
                    greedy_numeric_accuracy=m['baseline_numeric_correct']/m['measurements'],
                    exact_token_fraction=m['exact']/m['measurements'], capped=m['capped'],
                    target_adapter_sha256=audit['adapter_sha256'], source=str(aggregate)))
    summary = []
    for key in sorted({(r['arm'],r['endpoint'],r['method']) for r in rows}):
        group = [r for r in rows if (r['arm'],r['endpoint'],r['method'])==key]
        item = dict(arm=key[0], endpoint=key[1], method=key[2], seeds=[r['seed'] for r in group],
                    complete=len(group)==len(plan['recipe']['seeds']))
        for metric in ['wall_speedup','token_throughput_speedup','numeric_accuracy','greedy_numeric_accuracy','exact_token_fraction']:
            values = [r[metric] for r in group]
            item[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None)
        summary.append(item)
    return dict(per_seed=rows, across_seeds=summary, pending=pending,
                aggregation='Ratios are paired against each checkpoint\'s own greedy. Mean and sample SD across independent seeds; inference repeats are not additional seeds.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    a = p.parse_args()
    result = summarize(a.suite.resolve())
    (a.suite/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
