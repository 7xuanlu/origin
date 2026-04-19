# Eval Snapshot Workflow

This project keeps publishable benchmark numbers in a local gitignored file so README metrics can be updated without committing private baseline files.

## Files

- Local metrics (gitignored): `app/eval/baselines/readme_metrics.json`
- Tracked template: `docs/eval/readme_metrics.example.json`
- README updater: `scripts/update-readme-eval.py`

## Update flow

1. Run benchmark(s) locally and record headline metrics.
2. Update `app/eval/baselines/readme_metrics.json`.
3. Regenerate README snapshot:

```bash
python3 scripts/update-readme-eval.py
```

4. Commit the README and script/docs changes (the local metrics JSON stays untracked).

## Notes

- LongMemEval, LoCoMo-Plus, and LifeBench use `Recall@5`, `MRR`, and `NDCG@10` as headline fields.
- Keep `notes` concise, e.g. "pending reproducibility pass" or run metadata.
