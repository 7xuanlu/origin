# Retrieval

Applies under `crates/wenlan-core/src/retrieval/` in addition to the root and
`crates/wenlan-core/AGENTS.md` instructions.

- Keep retrieval changes on the existing quick, deep, expanded, or prototype seam;
  do not make an internal core caller silently inherit handler-only reranking.
- Preserve query scope and disclosure filters when a new channel can surface rows.
- Use focused ranking/substrate controls before expensive answer-quality evals.
- Feature-flag defaults, wiring, known hazards, and measurement receipts live in
  `REFERENCE.md`. Read only the entries relevant to the task.
