# One local + eight remote Qwen3-4B experiments

- Full regression: **135 passed, 1 skipped**; 12 automation tests included.
- Eight actual CPU worker subprocesses verified scheduling, independent GPU assignment, dependency waits and completion. Training bodies were fixtures, not GPU training.
- Model-only checkpoint export/import preserves weights; identities cover seed, objective, data, model config and training code. Mismatches fail before continuation.
- SSH/scp control flow was tested with mocked transport. No remote machine or login was supplied.
- The direct platform script `scripts/train/qwen3-4b/run8_qwen3_4b.sh` activated the configured conda environment and passed all eight parallel dry-run commands with distinct GPUs; no GPU training was launched.
- The complete 4B model/data preparation and eight-worker dry-run passed, including a repeated invocation.
- Local v2 and remote prepared A42 identities match across different paths. Local A42 is queued on GPU 2; remaining jobs are prepared for the external host.

[Machine-readable record](validation.json) · [Launch guide](../../docs/qwen3_4b_eight_gpu.md).
