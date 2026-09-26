# Joint checkpoint + hybrid inference: 2026-09-18

Checkpoint: original Qwen3-8B with frozen target, T + boundary jointly trained for **22,000 / 49,375** updates. This is an intermediate checkpoint, not completed training.

Protocol: GSM8K test rows 48–79 (32 distinct questions), two repeats, BF16, thinking disabled, batch one, up to 1,024 output tokens. Shared NVIDIA M402 80 GB GPU. Methods rotate order; optimized greedy uses the same target optimizations. Loading, graph compilation and warmup excluded; prefill and per-question T/cache work included.

| Method | Seconds | Output tokens | Wall speedup | Token/s speedup | Exact vs greedy | Numeric correct per repeat |
|---|---:|---:|---:|---:|---:|---|
| Optimized greedy | 312.953 | 17132 | 1.0000x | 1.0000x | 64/64 | 31/32, 31/32 |
| T only | 301.303 | 16430 | 1.0387x | 0.9961x | 26/64 | 31/32, 31/32 |
| Lookup + T hybrid | 242.371 | 16712 | 1.2912x | 1.2596x | 30/64 | 31/32, 31/32 |
| Lookup only | 235.491 | 16712 | 1.3289x | 1.2964x | 30/64 | 31/32, 31/32 |

Hybrid paired-question bootstrap 95% interval: wall [1.2027362499529297, 1.4048338255991328], token/s [1.2153254690940196, 1.302197030494541]. Repeats remain clustered by question. Shared GPU interference is not removed by these intervals.

The hybrid is faster than greedy here, while lookup-only is slightly faster than the hybrid. T fallback has not supplied a net speed benefit on this checkpoint. Output lengths differ, so wall speedup alone does not describe equal-token throughput. All methods stayed below the token cap.

**Correctness limitation:** hybrid output matches greedy token-for-token in only 30/64 measurements. Strict matching to each target block does not establish equivalence to serial BF16 greedy. Numeric answer scores on 32 questions cannot establish lossless behavior. All methods reproduce their own token output across both repeats. No unchecked or intentionally mismatched drafts were accepted.

This retest is separate from the earlier staged 60,375 + joint 2,000 experiment. Those targets have different adapter weights; their speed differences cannot be attributed solely to the training recipe.

See [machine-readable metrics](joint_hybrid_20260918.json) and the [run guide](../local_setup/README.md). Model weights and raw question/output datasets are external assets.
