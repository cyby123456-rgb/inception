"""Describe joint-training loss trends; training loss alone is not convergence."""
import argparse
import json
from pathlib import Path
import statistics

METRICS = {
    'loss': 'Total loss',
    'recurft_recurrent_loss': 'Recurrent MSE',
    'recurft_multistep_boundary_logit_kl_k1': 'Draft-1 KL',
    'recurft_multistep_boundary_logit_kl_k2': 'Draft-2 KL',
}


def analyze(rows, warmup=200, window=200):
    train = {r['global_step']: r for r in rows if 'loss' in r and r['global_step'] > warmup}
    if not train:
        raise ValueError('No post-warmup training losses')
    end = max(train)
    windows = []
    for start in range(warmup + 1, end + 1, window):
        selected = [r for s, r in sorted(train.items()) if start <= s < start + window]
        if not selected:
            continue
        windows.append(dict(start=start, end=min(start+window-1, end), log_points=len(selected),
            means={k: statistics.mean(r[k] for r in selected if k in r)
                   for k in METRICS if any(k in r for r in selected)}))
    recent = [w for w in windows if w['end']-w['start']+1 == window][-3:]
    trends = {}
    for metric in METRICS:
        vals = [w['means'][metric] for w in recent if metric in w['means']]
        if len(vals) == 3:
            trends[metric] = dict(last_three_window_means=vals,
                relative_range=(max(vals)-min(vals))/abs(statistics.mean(vals)),
                first_to_last_change=(vals[-1]-vals[0])/abs(vals[0]))
    validation = [r for r in rows if 'eval_loss' in r]
    near = len(recent) == 3 and trends.get('loss', {}).get('relative_range', 1) < .01
    return dict(last_step=end, warmup_excluded=warmup, window_steps=window,
        windows=windows, late_trends=trends, heldout_evaluations=len(validation),
        training_total_near_plateau=near,
        conclusion='Training total is near a local plateau; convergence is not established.' if near
                   else 'Training total has not met the descriptive local-plateau threshold.',
        limitations=['The 1% range over three windows is a descriptive heuristic, not a statistical stationarity test.',
                     'Training minibatches vary; held-out KL/top-1 and draft acceptance are needed.',
                     'A budget adequate for an 8B checkpoint does not establish adequacy for a 4B checkpoint.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(s) for s in a.telemetry.read_text().splitlines() if s.strip()]
    result = analyze(rows)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Checkpoint joint-training plateau analysis', '', result['conclusion'], '',
        f"Source: `{a.telemetry.resolve()}`. Exclude the first 200 loss-weight warmup updates.", '',
        '| Steps | Total | Recurrent MSE | Draft-1 KL | Draft-2 KL |', '|---|---:|---:|---:|---:|']
    for w in result['windows']:
        vals = [f"{w['means'][k]:.5f}" if k in w['means'] else '—' for k in METRICS]
        lines.append(f"| {w['start']}–{w['end']} | "+' | '.join(vals)+' |')
    lines += ['', f"Held-out loss evaluations in this telemetry: **{result['heldout_evaluations']}**.", '',
              *['- '+s for s in result['limitations']], '',
              'Keep the 2,000-update checkpoint. For the new model, reserve 4,000 updates and compare fixed held-out evaluations at 1,000/1,500/2,000/2,500/3,000/3,500/4,000. Do not select a stopping point on GSM8K test speed.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    train = [r for r in rows if 'loss' in r and r['global_step'] > 200]
    for ax, (key, label) in zip(axes.flat, METRICS.items()):
        selected = [r for r in train if key in r]
        ax.plot([r['global_step'] for r in selected], [r[key] for r in selected], alpha=.25)
        ws = [w for w in result['windows'] if key in w['means']]
        ax.plot([(w['start']+w['end'])/2 for w in ws], [w['means'][key] for w in ws], marker='o')
        ax.set(title=label, xlabel='Joint optimizer updates')
        ax.grid(alpha=.2)
    fig.suptitle('8B full checkpoint + joint training: raw logs and 200-step means')
    fig.tight_layout()
    fig.savefig(a.output/'loss_plateau.png', dpi=160)
    print(json.dumps({k:v for k,v in result.items() if k!='windows'}, indent=2))


if __name__ == '__main__':
    main()
