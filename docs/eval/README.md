# Eval Snapshot Workflow

This project keeps publishable benchmark numbers in a local gitignored file so README metrics can be updated without committing private baseline files.

## Files

- Local metrics (gitignored): `${EVAL_BASELINES_DIR:-~/.cache/origin-eval}/readme_metrics.json`
- Tracked template: `docs/eval/readme_metrics.example.json`
- README updater: `scripts/update-readme-eval.py`

## Update flow

1. Run benchmark(s) locally and record headline metrics.
2. Update `${EVAL_BASELINES_DIR:-~/.cache/origin-eval}/readme_metrics.json`.
3. Check tracked publishable metrics against their tracked source summaries:

```bash
python3 scripts/update-readme-eval.py --check docs/eval/readme_metrics.example.json
```

4. Regenerate README snapshots:

```bash
python3 scripts/update-readme-eval.py
```

5. Check translated README sync:

```bash
python3 scripts/check-readme-translations.py
```

6. Commit the README and script/docs changes (the local metrics JSON stays untracked).

## Notes

- LongMemEval rows use `Recall@5`, `MRR`, and `NDCG@10` as headline retrieval fields.
- Current README retrieval numbers are retrieval-only, single-run local snapshots unless a reproducibility pass is explicitly documented.
- LME-S 90 retrieval is saved in `docs/eval/results/lme_s_90_bge_base_pool20.summary.json` with raw rows in `docs/eval/results/lme_s_90_bge_base_pool20.jsonl`.
- `scripts/update-readme-eval.py` updates the generated retrieval block in the English README and every root-level localized `README.*.md`.
- Rows with `source_summary` are checked against tracked summary artifacts before they are treated as publishable README metrics.
- `scripts/check-readme-translations.py` discovers root-level localized `README.*.md` files and fails when any does not carry the current English README sync hash.
- Name the retrieval mode once in surrounding prose when all rows use the same mode.
- Keep `notes` in the metrics JSON for maintainer-facing caveats and run metadata; the root README does not render them.

## Answer Quality

End-to-end answer quality is tracked separately because it includes retrieval, answer generation, and judging:

| Benchmark | Mode | Accuracy | Task Avg | Correct | Artifact |
|---|---|---:|---:|---:|---|
| LongMemEval-S (deep, 60 Q) | full stack, CE reranker, single run | 76.7% | 76.7% | 46/60 | `$HOME/.cache/origin-eval-ceiling/lme_fullstack_ceiling_nsemeq_r5_judge_cache.jsonl` |

Run summary: `docs/eval/results/lme_s_fullstack_ce_reranker_best.summary.json`.

## Links

- [wenlan.app](https://wenlan.app) — project home
- [wenlan.app#benchmarks](https://wenlan.app/#benchmarks) — the public benchmark table sourced from this workflow
