# Qwen3-4B validation

- CPU/regression checks: **99 passed**. Includes real tiny Qwen3 backward, detached current-target teacher, independent legacy path, scope rejection, split isolation, inference checkpoint binding, and train/eval metric separation.
- Real 4B: five stages/modes × two optimizer updates, batch 2, sequence length 1024, accumulation 1. Expected target/T/head groups all received nonzero gradients. Original tied vocabulary parameters remained frozen.
- Peak allocated: 20.5–22.6 GiB; peak reserved: 23.9 GiB. Full configuration uses accumulation 4 and a 30,000 MiB free-memory preflight.
- Frozen checks: all 504 target adapter tensors unchanged in head-only and post-joint training; all 61 pre-existing recurrent tensors unchanged in head-only training.
- Trained-target inference: 504 loaded adapter tensors match the checkpoint after dtype conversion; greedy, T-only, hybrid and lookup-only share the same trained target. One measurement question plus separate warmup, 32-token limit.
- End-to-end runner: save at update 1 → dev/fixed-budget evaluation → restore optimizer/RNG → update 2 → dev/final evaluation → completed. Optimizer restored at step 1 and `ignore_data_skip=False`.
- All four scheduler evaluation endpoints completed, analysis and across-seed summary paths executed.
- Prepared 9 experiments / 18 training YAMLs, all dependencies and prescribed milestones checked; source snapshot manifest has 296 verified files.

Short synthetic checks do not establish convergence, speedup or final task accuracy. The first inference fixture needed an additional excluded warmup question; corrected inference passed. The nine full experiments have **not** started.

Machine-readable records: [validation.json](validation.json). Full raw local runs: `runs/qwen3_4b_smoke_v1` and `runs/qwen3_4b_scheduler_smoke`.
