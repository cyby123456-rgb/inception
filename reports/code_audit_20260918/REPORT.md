# Code and training audit

Detailed Chinese report: [功能与训练方式核查](../../docs/code_and_training_audit.md).

- 123 tests passed; 1 CUDA test skipped in CPU pytest.
- Five real Qwen3-4B training stages/scopes passed two updates each.
- Frozen target/T tensors remained exactly unchanged where required.
- New protocol and resume contract validated in a complete save/evaluate/resume flow.
- Four corrected-baseline inference endpoints completed; 32 generated greedy tokens used exactly 32 target calls.
- 9 experiments / 18 stages prepared; no full long training started.

[Resolved profiles](resolved_training_modes.json) · [Validation evidence](validation.json).
