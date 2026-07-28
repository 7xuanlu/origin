# Weekly verification sweep — verify-the-verifier on a schedule

Recurring: `/loop 7d "/prove sweep"` (or run manually alongside
`bash scripts/drift-audit.sh`). Attention only at red — a clean sweep ends
in one line.

1. **Gate red-check** — the self-testing gates prove they can still fail:
   - `python3 scripts/check-behavior-trace.py --self-test`
2. **Mutation sample** — pick the 2-3 modules with the most churn this week
   (`git log --since=1.week --stat`), write 3-5 behavior mutations each,
   run mutprove (references/mutprove.md). Survivors → new tests or a noted
   acceptance.
3. **Blind audit** — for plans merged this week, run the tautology audit
   from references/behaviors.md on their Behaviors sections.
4. **Attest ledger scan** — `tail .claude/attest.jsonl`: any week where
   surfaces shipped but their smoke never ran attested is itself a finding.
5. **Report** — findings to `docs/superpowers/drift-reports/` (gitignored),
   one-line summary in chat. Recurring finding (≥3×) → promote a rule per
   the loop-engineering playbook; a gate that never fired in 4 weeks →
   candidate for retirement.
