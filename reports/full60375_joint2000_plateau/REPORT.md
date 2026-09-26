# Checkpoint joint-training plateau analysis

Training total is near a local plateau; convergence is not established.

Source: `/mnt/llmshared-ssd-hd/wangruitao/inception-joint/runs/qwen3_joint_full60375_20260917_010621/joint/telemetry.jsonl`. Exclude the first 200 loss-weight warmup updates.

| Steps | Total | Recurrent MSE | Draft-1 KL | Draft-2 KL |
|---|---:|---:|---:|---:|
| 201–400 | 4.69907 | 3.84853 | 3.65874 | 4.74159 |
| 401–600 | 4.55499 | 3.92914 | 3.52943 | 4.56468 |
| 601–800 | 4.28341 | 3.99755 | 3.27941 | 4.25389 |
| 801–1000 | 4.16955 | 4.00466 | 3.23754 | 4.06605 |
| 1001–1200 | 4.21164 | 4.06634 | 3.20341 | 4.17011 |
| 1201–1400 | 4.01834 | 4.07506 | 3.05110 | 3.93367 |
| 1401–1600 | 3.98715 | 4.11757 | 3.02753 | 3.88581 |
| 1601–1800 | 3.96448 | 4.13238 | 3.01579 | 3.84812 |
| 1801–2000 | 3.98489 | 4.18362 | 3.01004 | 3.87901 |

Held-out loss evaluations in this telemetry: **0**.

- The 1% range over three windows is a descriptive heuristic, not a statistical stationarity test.
- Training minibatches vary; held-out KL/top-1 and draft acceptance are needed.
- A budget adequate for an 8B checkpoint does not establish adequacy for a 4B checkpoint.

Keep the 2,000-update checkpoint. For the new model, reserve 4,000 updates and compare fixed held-out evaluations at 1,000/1,500/2,000/2,500/3,000/3,500/4,000. Do not select a stopping point on GSM8K test speed.
