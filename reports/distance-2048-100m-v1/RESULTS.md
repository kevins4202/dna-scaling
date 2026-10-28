# Frozen CNN distance experiment through 100M

| Continuation bases | 5M AUROC (95% CI) | 10M AUROC (95% CI) | 25M AUROC (95% CI) | 50M AUROC (95% CI) | 100M AUROC (95% CI) |
|---|---|---|---|---|---|
| 0–512 | 0.616 (0.523–0.698) | 0.595 (0.436–0.655) | 0.585 (0.466–0.658) | 0.610 (0.508–0.677) | 0.616 (0.457–0.684) |
| 512–1024 | 0.527 (0.417–0.602) | 0.542 (0.442–0.649) | 0.562 (0.461–0.646) | 0.519 (0.424–0.615) | 0.536 (0.451–0.645) |
| 1024–1536 | 0.684 (0.613–0.763) | 0.704 (0.632–0.782) | 0.656 (0.593–0.771) | 0.667 (0.593–0.763) | 0.660 (0.583–0.764) |
| 1536–2048 | 0.674 (0.539–0.746) | 0.670 (0.511–0.744) | 0.654 (0.496–0.712) | 0.628 (0.472–0.715) | 0.685 (0.608–0.755) |
| Aggregate | 0.684 (0.612–0.745) | 0.687 (0.564–0.756) | 0.676 (0.560–0.751) | 0.647 (0.550–0.740) | 0.685 (0.615–0.759) |

Higher AUROC means easier detection. Aggregate averages logits. Intervals are descriptive and unadjusted.

- Eligibility requires 2304 contiguous ACGT bases; missing/rare categories and limited eukaryotic source groups constrain inference.
- Existing independent reference pool and some CNN test source groups were inspected previously; this is a new distance analysis, not a new untouched biological benchmark.
- Source rows are not guaranteed species independent. Repeated windows are clustered by source.
- Frozen CNN sensitivity gate failed for 5M; chance detection does not establish biological realism.
- Exploratory family extension authorized after smaller-model results; no CNN retraining, no reference changes and no automatic significance/scaling-law claim
