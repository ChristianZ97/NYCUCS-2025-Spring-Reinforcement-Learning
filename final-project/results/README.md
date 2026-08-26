# Historical results

These tables were transcribed from the grading-time final report. The underlying
BDD-OIA data, trained weights, and original logs are unavailable in the
portfolio tree, so the canonical source was not retrained or benchmarked.

- `pipeline_metrics.csv` summarizes the report's end-to-end comparison.
- `clip_reward_metrics.csv` summarizes the domain-specific CLIP classifier.

The report notes that the RLOO run improved scene-description behavior but
suffered catastrophic reward degradation, while the final action stage
collapsed to a single action. The report body calls its `84.12%` action metric
accuracy, whereas an embedded poster labels the same value F1. The CSV records
that unresolved source conflict as `accuracy_or_f1` instead of silently
choosing one interpretation.
