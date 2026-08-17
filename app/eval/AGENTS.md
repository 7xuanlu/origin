# Eval fixtures and artifacts

Applies under `app/eval/` in addition to the root instructions.

- Treat cached scenario databases and model-backed baselines as external, reproducible
  artifacts.
- Use smoke limits for wiring checks and the smallest representative subset for method
  checks.
- Do not cite a score without the producing commit, environment/schema stamp, sample
  size, per-run receipt, and the correct pipeline layer.
- Never compare arms produced from different substrates or silently migrate/wipe a
  cache without the explicit owning flag and preflight.

## eval citation discipline

External numbers must come from retained receipts, not terminal recollection or an
average with missing runs. State skipped legs and unavailable gates explicitly.

Commands, cache flags, layouts, bench-specific methodology, and historical receipts
live in `REFERENCE.md`. Rust runner internals live in
`crates/wenlan-core/src/eval/REFERENCE.md`.
