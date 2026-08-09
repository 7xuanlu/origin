# Eval fixtures and artifacts

Applies under `app/eval/` in addition to the root instructions.

- Missing fixtures must fail or report `unchecked`; never silently convert a skipped
  dataset into a passing eval.
- Treat cached scenario databases and model-backed baselines as external, reproducible
  artifacts. Do not keep their only valuable copy inside a disposable worktree.
- Use smoke limits for wiring checks and the smallest representative subset for method
  checks. Run a full eval only when the decision or publication claim requires it.
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
