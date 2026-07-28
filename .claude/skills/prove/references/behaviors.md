# Behavior trace — "is this the right test?"

The intent anchor is the plan/spec itself (no separate contract artifact):
the plan carries a `## Behaviors` section written BEFORE implementation and
glanced/vetoed by the human at dispatch.

```markdown
## Behaviors
- B1: given a stored memory, `wenlan memories` lists it
- B2: given a dirty file, mutprove refuses to mutate it
```

Rules:
- ≤10 behaviors, each one observable outcome (given/when/then in one line).
- Tests cite what they protect: `// covers: B2` (any comment syntax;
  `covers: B1, B2` for multi-id).
- The checker enforces the mapping both ways — every behavior covered, every
  tag defined: `python3 scripts/check-behavior-trace.py <plan.md> <tests...>`
  Uncovered behavior = untested intent. Dangling tag = test mirroring nothing.

## Blind audit (catches tautological tests the checker can't)

A test can cite B3 and still just assert whatever the implementation does.
Periodically (per risky feature, and in the weekly sweep) spawn a subagent
that gets ONLY the `## Behaviors` section and the test file — never the
implementation — and answers per test:

1. Restated in plain English, what does this test require?
2. Does that match the cited behavior, or does it smell like a mirror of
   implementation internals (asserting exact strings/structures no behavior
   line mentions)?
3. Which behavior would a malicious-but-green implementation break?

Findings are advisory (taste, not teeth) — route real mismatches into fixes,
and recurring ones into the plan template.
