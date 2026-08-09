# Rust eval runners

Applies under `crates/wenlan-core/src/eval/` in addition to the root and
`crates/wenlan-core/AGENTS.md` instructions.

- Match the existing ephemeral or cached runner shape; do not mix their source-id,
  seeding, or artifact contracts.
- Start with a smoke-sized, category-aware run. Use a full benchmark only when the
  decision needs that statistical power; repeated full runs are not a default gate.
- Refuse experiments whose required substrate is absent, stale, consolidated across
  questions when isolation is required, or otherwise confounded.
- A/B arms must stamp their environment and differ only in the intended treatment.
  Run the matching A/A/no-op control when the analyzer requires a noise floor.
- Retrieval drift goldens detect change, not correctness. Refresh them only after the
  labeled eval is green and the ranking change is intentional.

Detailed runner commands, cache flags, power guidance, methodology, and receipts live
in `REFERENCE.md`. Fixture and artifact handling lives in `app/eval/REFERENCE.md`.
