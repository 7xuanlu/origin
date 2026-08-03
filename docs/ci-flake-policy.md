# CI flake policy

Use this response policy when a required CI check fails intermittently. A red
job is not automatically a flake, and a green rerun is not automatically a fix.

1. **Name the exact failure from the full job log.** Record the job, command,
   test, assertion or timeout, runner OS, and run URL. Check-run annotations such
   as "exit code 1" are not enough. First separate test failures from setup,
   infrastructure, and CI time-budget failures.
2. **Use at most one rerun for diagnosis.** Rerun only when the first log points
   to timing or runner state. Compare both signatures. A pass strengthens the
   flake hypothesis but does not close the defect; repeated failure points to a
   deterministic problem and should be debugged without more reruns.
3. **Two matching occurrences require an owner and a fix.** Two independent
   occurrences of the same test and symptom, on any branches or runners, are
   enough to open a tracking issue and prioritize deflaking. Prefer event-driven
   waits (`wait_until`, as used in
   `crates/wenlan-server/src/reflection_debounce.rs`) or poll-with-deadline (as
   used in `crates/wenlan-cli/src/commands/service.rs`). Do not use bare sleeps
   or single-shot assertions on asynchronous or process state.
4. **Quarantine must fail closed.** Do not add `#[ignore]` or exclude a test from
   its owning plan unless the same PR first wires that exact test into an
   explicit, visible, non-blocking main-push or scheduled job. Link a tracking
   issue with an owner, removal condition, and deadline. Without that replacement
   execution path, fix the test or leave the required check red; never disable
   or reroute an entire owning CI lane to hide one flaky test. The existing
   [`main-canary.yml`](../.github/workflows/main-canary.yml) runs selected eval
   tests only and is not a general quarantine lane.
5. **Do not revert an innocent trigger.** Cold caches, rotated cache keys, or a
   busy runner can expose an existing race. Revert only when the changed code or
   CI route caused the failure; otherwise fix or quarantine the failing test.
   PR #401 exposed two existing timing races after a toolchain cache rotation;
   PR #403 fixed the tests instead of reverting the trigger.

## Multiple merge-ready PRs

When several PRs are ready together, freeze their dependency order, merge every
predecessor, then update the follower once against the final base. Use stacked
PR bases only for real code dependencies. Let the workflow's
[concurrency group](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency)
cancel superseded heads, and never rerun a stale SHA. Keep
[strict required checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches):
repeated builds are the cost of proving each new merge result, not a reason to
accept an untested combination.

If this personal repository later moves to an eligible GitHub organization,
prefer a [merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)
and add the `merge_group` trigger to every required workflow. GitHub's native
merge queue is not available to personal-owned repositories today.
