# Mutation audit — prove the suite can go red

Two tools, split by question:

- **Breadth — "where is my suite blind?"**: `cargo mutants -f <file>` (install:
  `cargo install cargo-mutants`). Auto-enumerates mutants, no hand-writing,
  oracle is cargo test/nextest. Use for the weekly sweep's mutation sample.
- **Depth — "does THIS gate reject THIS breakage?"**: `~/.claude/bin/mutprove.py`
  below. Hand-picked mutations against an ARBITRARY oracle (smoke script, e2e
  binary) — the class cargo-mutants can't drive. Generalizes the pattern of
  `scripts/m3g-gate1-mutation-proof.sh` (same mechanics: unique-anchor assert,
  expect-red, restore); future gate proofs write a mutations.json instead of a
  bespoke script.

Scripted, never conversational (the model round-trip is the cost; measured
19s machine vs ~90s model per cycle).

```bash
# mutations.json: [{"id","desc","file","old","new"}, ...]
TMPDIR=/tmp/claude ~/.claude/bin/attest.sh \
  python3 ~/.claude/bin/mutprove.py mutations.json -- \
  cargo test -p wenlan-core --lib <module>::tests
```

Writing mutations:
- Mutate BEHAVIOR the tests should protect: flip a comparison, drop a filter
  arm, wrong endpoint path, off-by-one a limit. Not syntax the compiler catches.
- `old` must appear exactly once in the file — the runner errors otherwise
  (a no-op edit must never report as a survivor; substring anchors have
  matched wrong indentation before).
- Files must be committed-clean; the runner restores via `git checkout --`.
- Don't pre-declare which test should kill a mutation — the report records
  which tests actually died, which reveals coverage gaps for free.
- The test command can be a smoke script too: mutating `crates/wenlan-cli`
  with `-- bash scripts/smoke-cli.sh` red-proves the smoke itself.

Exit 0 = all killed. Exit 1 = survivors → either add the missing test or
consciously accept and note why. Exit 2 = harness error, fix before trusting.
