// SPDX-License-Identifier: AGPL-3.0-only
/**
 * The retry schedule for the first-run gate query in `src/App.tsx`.
 *
 * This lives in its own module for one reason: the total budget is a contract
 * with the Rust side, and a contract nobody can read is a contract nobody
 * keeps. `bootRetryPolicy.test.ts` pins the arithmetic.
 *
 * ## Why the budget is what it is
 *
 * The daemon binds its port BEFORE it can serve: migrations, embedder load and
 * projection reconcile all happen behind an already-bound socket
 * (`crates/wenlan-server/src/main.rs` binds first, `axum::serve` starts in
 * `crates/wenlan-server/src/main/runtime.rs`, and the mute-port window is
 * described in `crates/wenlan-server/src/main/startup.rs`). So "connection
 * accepted" does not mean "daemon ready", and the gate has to outlast that
 * window rather than treat it as a failure.
 *
 * The app already has an answer for how long that window can be: the Rust
 * health loop in `app/src/lib.rs` waits ~152s for the same daemon
 * (10 attempts, each bounded by the 5s HEALTH_TIMEOUT, with 200ms * 2^i
 * backoff between them — 50_000ms of attempts + 102_200ms of delays). This
 * frontend gate must not give up before the Rust side does, or it drops an
 * already-configured user into the fail-closed SetupWizard while the app's own
 * startup path is still patiently waiting for the very same daemon.
 *
 * So: the budget below is >= RUST_HEALTH_LOOP_BUDGET_MS, on purpose.
 */

/**
 * Per-attempt ceiling, mirroring `HEALTH_TIMEOUT` in `app/src/api.rs`.
 *
 * `get_setup_status` attaches that timeout explicitly, so a bound-but-mute
 * daemon costs one attempt this much and no more.
 */
export const ATTEMPT_TIMEOUT_MS = 5_000;

/**
 * What the Rust health loop in `app/src/lib.rs` tolerates for the same daemon:
 * 10 attempts * 5_000ms, plus 200ms * 2^i backoff for i in 0..=8
 * (200 * 511 = 102_200ms). The frontend budget must be at least this.
 */
export const RUST_HEALTH_LOOP_BUDGET_MS = 10 * ATTEMPT_TIMEOUT_MS + 200 * 511;

/** Retries AFTER the first attempt, i.e. react-query's `retry`. */
export const BOOT_QUERY_RETRY = 20;

/** Exponential backoff, capped so late attempts still poll every 3s. */
export function bootQueryRetryDelay(attemptIndex: number): number {
  return Math.min(1000 * 2 ** attemptIndex, 3000);
}

/**
 * Worst-case wall time before the gate gives up: every attempt burning its
 * full timeout, plus every delay between them.
 */
export function bootQueryBudgetMs(retry: number = BOOT_QUERY_RETRY): number {
  let total = (retry + 1) * ATTEMPT_TIMEOUT_MS;
  for (let attemptIndex = 0; attemptIndex < retry; attemptIndex += 1) {
    total += bootQueryRetryDelay(attemptIndex);
  }
  return total;
}
