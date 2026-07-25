// SPDX-License-Identifier: Apache-2.0
//! Debounced background reflection (T22).
//!
//! When an agent stores many memories in a burst, the per-store detached
//! enrichment spawn (`memory_routes::handle_store_memory`) would otherwise fire
//! N overlapping `run_post_ingest_enrichment` tasks — N classify+extract LLM
//! calls, N page-growth passes — most of which are wasted because a newer write
//! from the same agent lands before the older reflection finishes.
//!
//! [`ReflectionDebouncer`] coalesces those spawns **per agent key**. Each
//! `schedule(key, delay, work)` call cancels the previously-scheduled work for
//! that key, distinguishing two cases so cancellation is always clean:
//!   - **Pre-delay** (work has not started — still inside its debounce `sleep`):
//!     the task is [`tokio::task::JoinHandle::abort`]ed. Nothing has run, so
//!     aborting drops no state.
//!   - **In-flight** (work has started running): the task is NOT aborted.
//!     Instead a shared [`AtomicBool`] is flipped; the running `work` checks it
//!     at clean step boundaries (cooperative cancellation; the actual
//!     checkpoints live in
//!     `wenlan_core::post_ingest::run_post_ingest_enrichment`) and returns early
//!     without half-applying a step. Aborting a started enrichment is exactly
//!     the half-write risk this design avoids.
//!
//! Only the **latest** write in a burst actually enriches. No write is dropped:
//! cancelling the older reflection only defers it — the newer one supersedes it
//! and enriches the same agent's freshly-written rows.
//!
//! Anti-starvation: a never-settling writer that reschedules the same key
//! forever would never reflect. Only **in-flight** cancellations count toward
//! the streak (a pre-delay abort cancelled work that never ran, so it must not
//! push the key toward a force-run). After [`MAX_CANCELLATIONS`] consecutive
//! in-flight cancellations the next `schedule` force-runs: it still stops the
//! prior task first (abort if it never started, cooperative-cancel if it is
//! in-flight) so the prior and the freshly-spawned task never enrich
//! concurrently — exactly the latest write enriches — then resets the streak so
//! the fresh task is not itself immediately starved. Reflection makes progress
//! under sustained load without ever double-enriching.
//!
//! This struct lives in `wenlan-server` (the detached-spawn owner). The
//! cancellation *signal* it produces (`Arc<AtomicBool>`) is passed into
//! `wenlan-core`'s `run_post_ingest_enrichment` as `Option<&AtomicBool>`; core
//! stays framework-agnostic and only sees a plain atomic.
//!
//! No new dependency: built on `tokio::spawn` + `JoinHandle::abort` +
//! `std::sync::atomic::AtomicBool` only (no `tokio-util` / `CancellationToken`).

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;

/// After this many consecutive cancellations of the same key, the next
/// `schedule` lets the in-flight reflection finish instead of cancelling it.
/// Bounds worst-case reflection latency under a writer that never settles.
pub const MAX_CANCELLATIONS: u32 = 5;

/// One in-flight (or pending) reflection for a single key.
struct Slot {
    /// Handle to the delay+work task. Aborted ONLY while the task is still in
    /// its pre-delay `sleep` (see `started`); once work has begun we cancel
    /// cooperatively via `cancel` instead of aborting.
    handle: JoinHandle<()>,
    /// Cooperative-cancel flag the running `work` checks at step boundaries.
    /// Flipping it stops an already-started reflection at a clean boundary.
    cancel: Arc<AtomicBool>,
    /// Set to `true` by the task the instant its debounce delay elapses and it
    /// is about to call `work`. Gates abort: `false` ⇒ safe to abort (nothing
    /// ran), `true` ⇒ must cooperate via `cancel` (never abort mid-work).
    started: Arc<AtomicBool>,
    /// How many times this key's reflection has been cancelled in a row without
    /// completing. Reset to 0 whenever a force-run is allowed through.
    cancellations: u32,
}

/// Per-key debouncer for background reflection spawns.
///
/// Cheap to clone (an `Arc` around the slot map); share one instance across all
/// store handlers via `ServerState`.
#[derive(Clone, Default)]
pub struct ReflectionDebouncer {
    slots: Arc<Mutex<HashMap<String, Slot>>>,
}

impl ReflectionDebouncer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule `work` to run for `key` after `delay`, cancelling/coalescing any
    /// previously-scheduled work for the same key.
    ///
    /// `work` receives an `Arc<AtomicBool>` cancel flag; it MUST check it at
    /// clean boundaries and return early when set (so cancellation never
    /// half-applies state). The latest scheduled work for a key always runs
    /// (subject only to a still-later `schedule` superseding it), so a burst of
    /// K rapid writes for one agent collapses to a single reflection of the
    /// last write rather than K overlapping ones.
    pub fn schedule<F, Fut>(&self, key: &str, delay: Duration, work: F)
    where
        F: FnOnce(Arc<AtomicBool>) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let cancel = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));

        // Cancel any prior work for this key and carry forward its cancellation
        // streak so the anti-starvation ceiling counts across reschedules.
        let prior_count = {
            let mut slots = self.slots.lock().unwrap();
            if let Some(prev) = slots.remove(key) {
                // Did the prior task actually start running its work (in-flight),
                // or is it still inside its pre-delay `sleep`? This gates both
                // how we cancel it and whether it counts toward the streak.
                let prev_started = prev.started.load(Ordering::SeqCst);
                if prev.cancellations >= MAX_CANCELLATIONS {
                    // Anti-starvation force-run: the latest write must win, but we
                    // MUST stop the prior task first — otherwise the prior and the
                    // freshly-spawned task both run `run_post_ingest_enrichment`
                    // concurrently and race on overlapping page writes
                    // (grow_page/update_page). If the prior never started, abort
                    // it (safe — it ran nothing). If it is in-flight, signal
                    // cooperative cancel so it stops at its next clean checkpoint
                    // (never hard-abort a started task — that risks a half-write).
                    // Either way exactly the latest enrichment runs. Reset the
                    // streak so the freshly-scheduled work below is not itself
                    // starved next.
                    if prev_started {
                        prev.cancel.store(true, Ordering::SeqCst);
                    } else {
                        prev.handle.abort();
                    }
                    0
                } else {
                    // Always signal cooperative cancel (covers the in-flight
                    // case). Abort ONLY if work has not started yet — i.e. the
                    // task is still inside its debounce `sleep`. Aborting a
                    // started task could kill it mid-step; the flag handles that
                    // case at a clean boundary instead.
                    prev.cancel.store(true, Ordering::SeqCst);
                    if !prev_started {
                        prev.handle.abort();
                    }
                    // Only count an in-flight cancellation toward the
                    // anti-starvation ceiling. A pre-delay abort cancelled work
                    // that never ran, so it must not push the key toward a
                    // force-run (that would let a bursty-but-quick writer trip the
                    // ceiling without any reflection ever having started).
                    if prev_started {
                        prev.cancellations + 1
                    } else {
                        prev.cancellations
                    }
                }
            } else {
                0
            }
        };

        let cancel_for_task = cancel.clone();
        let started_for_task = started.clone();
        let slots_for_cleanup = self.slots.clone();
        let key_owned = key.to_string();

        let handle = tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            // A newer schedule that arrived during the sleep either aborted us
            // (we never reach here) or set the cancel flag — bail before work.
            if cancel_for_task.load(Ordering::SeqCst) {
                return;
            }
            // Mark started BEFORE running work: from here a concurrent reschedule
            // must cooperate (flag) rather than abort. There is a tiny window
            // between this store and the next `schedule`'s `started.load`; if a
            // reschedule reads `started == false` and aborts right after we set
            // it true, abort still only fires at the next `.await` inside work,
            // where the very first checkpoint reads the (now-set) cancel flag and
            // returns early anyway. Either way no step is half-applied.
            started_for_task.store(true, Ordering::SeqCst);
            work(cancel_for_task.clone()).await;

            // Self-cleanup: drop our own slot if it is still ours (a newer
            // schedule for the same key may have already replaced it).
            let mut slots = slots_for_cleanup.lock().unwrap();
            if let Some(slot) = slots.get(&key_owned) {
                if Arc::ptr_eq(&slot.cancel, &cancel_for_task) {
                    slots.remove(&key_owned);
                }
            }
        });

        let mut slots = self.slots.lock().unwrap();
        slots.insert(
            key.to_string(),
            Slot {
                handle,
                cancel,
                started,
                cancellations: prior_count,
            },
        );
    }

    /// Number of live (pending or running, not yet cleaned-up) keys. Test hook
    /// for the no-unbounded-growth guarantee.
    #[cfg(test)]
    pub fn live_keys(&self) -> usize {
        self.slots.lock().unwrap().len()
    }

    /// Current consecutive-cancellation streak for `key`, or `None` if the key
    /// has no live slot. Test hook for the anti-starvation ceiling reset.
    #[cfg(test)]
    pub fn cancellations_for(&self, key: &str) -> Option<u32> {
        self.slots
            .lock()
            .unwrap()
            .get(key)
            .map(|slot| slot.cancellations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// Settle helper: advance wall-clock so spawned delay+work tasks complete.
    /// Used by time-tolerant tests such as the pre-delay abort cases
    /// whose assertions don't depend on a real-sleep loop finishing inside a
    /// fixed budget. The force-run / ceiling tests use the event-driven
    /// `wait_signal` / `wait_until` helpers below instead.
    async fn settle(ms: u64) {
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }

    /// Block until `notify` is signalled, with a generous timeout so a real hang
    /// (logic regression) fails fast instead of waiting on a wall-clock budget.
    /// This is the event-driven replacement for `settle()` in tests that must be
    /// deterministic under single-threaded CI load: it returns the instant the
    /// awaited work reaches its signal point, not after a fixed real-time span.
    async fn wait_signal(notify: &tokio::sync::Notify) {
        tokio::time::timeout(Duration::from_secs(5), notify.notified())
            .await
            .expect("timed out waiting for signal — work never reached its checkpoint");
    }

    /// Poll `cond` until it holds, yielding between checks, with a 5s timeout.
    /// Used for the self-cleanup assertion, which happens just after the work
    /// body's terminal signal and therefore can't be awaited on a single Notify.
    async fn wait_until(mut cond: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !cond() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("timed out waiting for condition");
    }

    #[tokio::test]
    async fn test_single_schedule_runs_work_once() {
        let deb = ReflectionDebouncer::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        deb.schedule("agentA", Duration::from_millis(30), move |_cancel| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::Relaxed);
            }
        });
        wait_until(|| counter.load(Ordering::Relaxed) == 1).await;
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "work should run exactly once"
        );
    }

    #[tokio::test]
    async fn test_reschedule_cancels_prior_before_delay() {
        let deb = ReflectionDebouncer::new();
        let ran1 = Arc::new(AtomicUsize::new(0));
        let ran2 = Arc::new(AtomicUsize::new(0));

        let r1 = ran1.clone();
        deb.schedule("agentA", Duration::from_millis(200), move |_c| {
            let r1 = r1.clone();
            async move {
                r1.fetch_add(1, Ordering::Relaxed);
            }
        });
        // Reschedule immediately with a much shorter delay — prior work1 is
        // still inside its 200ms sleep, so abort() must prevent it running.
        let r2 = ran2.clone();
        deb.schedule("agentA", Duration::from_millis(30), move |_c| {
            let r2 = r2.clone();
            async move {
                r2.fetch_add(1, Ordering::Relaxed);
            }
        });

        settle(300).await;
        assert_eq!(
            ran1.load(Ordering::Relaxed),
            0,
            "work1 must be cancelled pre-delay"
        );
        assert_eq!(
            ran2.load(Ordering::Relaxed),
            1,
            "work2 must run exactly once"
        );
    }

    #[tokio::test]
    async fn test_reschedule_signals_cancel_to_inflight_work() {
        let deb = ReflectionDebouncer::new();
        let observed_cancel = Arc::new(AtomicUsize::new(0));
        let work2_done = Arc::new(AtomicUsize::new(0));

        // work1 starts quickly, then polls the cancel flag in a loop.
        let oc = observed_cancel.clone();
        deb.schedule("agentA", Duration::from_millis(10), move |cancel| {
            let oc = oc.clone();
            async move {
                for _ in 0..200 {
                    if cancel.load(Ordering::SeqCst) {
                        oc.fetch_add(1, Ordering::Relaxed);
                        return; // bail early — cooperative cancellation
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        });

        // Let work1 start running (past its 10ms delay) before rescheduling, so
        // we exercise the in-flight AtomicBool path, not the pre-delay abort.
        settle(40).await;

        let wd = work2_done.clone();
        deb.schedule("agentA", Duration::from_millis(10), move |_c| {
            let wd = wd.clone();
            async move {
                wd.fetch_add(1, Ordering::Relaxed);
            }
        });

        settle(200).await;
        assert_eq!(
            observed_cancel.load(Ordering::Relaxed),
            1,
            "in-flight work1 must observe cancel and bail early"
        );
        assert_eq!(work2_done.load(Ordering::Relaxed), 1, "work2 must complete");
    }

    #[tokio::test]
    async fn test_concurrent_keys_are_independent() {
        let deb = ReflectionDebouncer::new();
        let a_final = Arc::new(AtomicUsize::new(0));
        let b_ran = Arc::new(AtomicUsize::new(0));

        // Schedule agentB once — it must NOT be disturbed by agentA reschedules.
        let b = b_ran.clone();
        deb.schedule("agentB", Duration::from_millis(40), move |_c| {
            let b = b.clone();
            async move {
                b.fetch_add(1, Ordering::Relaxed);
            }
        });

        // Schedule + reschedule agentA.
        deb.schedule("agentA", Duration::from_millis(200), |_c| async {});
        let af = a_final.clone();
        deb.schedule("agentA", Duration::from_millis(30), move |_c| {
            let af = af.clone();
            async move {
                af.fetch_add(1, Ordering::Relaxed);
            }
        });

        settle(300).await;
        assert_eq!(
            b_ran.load(Ordering::Relaxed),
            1,
            "agentB work must run untouched"
        );
        assert_eq!(
            a_final.load(Ordering::Relaxed),
            1,
            "final agentA work must run"
        );
    }

    /// Event-driven: each scheduled task signals `started` the instant its work
    /// body runs, and we await that signal (not a fixed `settle`) before reading
    /// the streak and scheduling the next, so every cancellation lands on an
    /// in-flight task and counts toward the ceiling regardless of CI load. The
    /// final (force-run winner) task runs a fixed, bounded, cancel-respecting
    /// body and signals `done`; the test awaits that signal instead of a fixed
    /// real-time budget. Only the uncancelled final task completes (intermediates
    /// are cancelled at their first checkpoint, long before the bounded body
    /// finishes), so exactly one reflection completes.
    #[tokio::test]
    async fn test_max_deferral_ceiling_forces_run() {
        use tokio::sync::Notify;

        let deb = ReflectionDebouncer::new();
        let runs = Arc::new(AtomicUsize::new(0));
        // Number of work bodies currently inside their loop. Must drain to 0:
        // the force-run path stops the prior before the new one completes, so no
        // work body is ever left running (the write-write race FIX 1 closes).
        let in_work = Arc::new(AtomicUsize::new(0));
        let peak_streak = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(Notify::new());

        // Schedule MAX_CANCELLATIONS + 2 times. The streak reaches the ceiling
        // after schedule #(MAX_CANCELLATIONS+1); schedule #(MAX_CANCELLATIONS+2)
        // is the force-run that cancels the in-flight prior and runs to
        // completion uncontested.
        for _ in 0..(MAX_CANCELLATIONS + 2) {
            let r = runs.clone();
            let iw = in_work.clone();
            let started = Arc::new(Notify::new());
            let started_w = started.clone();
            let done_w = done.clone();
            deb.schedule("agentA", Duration::from_millis(5), move |cancel| {
                let r = r.clone();
                let iw = iw.clone();
                let started_w = started_w.clone();
                let done_w = done_w.clone();
                async move {
                    iw.fetch_add(1, Ordering::SeqCst);
                    started_w.notify_one();
                    // Fixed, bounded, cancel-respecting work. Intermediates are
                    // cancelled at their first checkpoint (the next schedule sets
                    // the flag before this can finish 50 iterations); only the
                    // final uncancelled task runs the full body and completes.
                    for _ in 0..50 {
                        if cancel.load(Ordering::SeqCst) {
                            iw.fetch_sub(1, Ordering::SeqCst);
                            return;
                        }
                        tokio::task::yield_now().await;
                    }
                    iw.fetch_sub(1, Ordering::SeqCst);
                    r.fetch_add(1, Ordering::SeqCst);
                    done_w.notify_one();
                }
            });
            // Await this work item being in-flight (past its delay) before the
            // next reschedule, so we drive the in-flight cancel/ceiling path.
            wait_signal(&started).await;
            // Track how high the in-flight cancellation streak climbed. With
            // every prior in-flight it must reach exactly MAX_CANCELLATIONS just
            // before the force-run fires.
            if let Some(c) = deb.cancellations_for("agentA") {
                let c = c as usize;
                let mut cur = peak_streak.load(Ordering::SeqCst);
                while c > cur {
                    match peak_streak.compare_exchange(cur, c, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(_) => break,
                        Err(observed) => cur = observed,
                    }
                }
            }
        }

        // The streak must have climbed to exactly the ceiling before the
        // force-run reset it — proving the anti-starvation bound is real, not
        // an off-by-one that fires early or never.
        assert_eq!(
            peak_streak.load(Ordering::SeqCst),
            MAX_CANCELLATIONS as usize,
            "in-flight cancellation streak must reach MAX_CANCELLATIONS before force-run"
        );
        // Force-run resets the streak to 0 so the freshly-spawned task is not
        // itself immediately starved.
        assert_eq!(
            deb.cancellations_for("agentA"),
            Some(0),
            "force-run must reset the cancellation streak to 0"
        );

        // Block until the force-run winner signals completion (timeout-guarded).
        wait_signal(&done).await;
        // Exactly one reflection completes: the final force-run cancelled the
        // prior in-flight task, so the prior and the freshly-spawned task never
        // both finish enrichment (no concurrent double-enrichment / page race).
        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "exactly one reflection must complete after a force-run, got {}",
            runs.load(Ordering::SeqCst)
        );
        // No work body is left running, and the slot is cleaned up. Both happen
        // right after the terminal signal, so poll briefly for them to drain.
        wait_until(|| in_work.load(Ordering::SeqCst) == 0).await;
        assert_eq!(
            in_work.load(Ordering::SeqCst),
            0,
            "no work body may still be in-flight after completion"
        );
        wait_until(|| deb.live_keys() == 0).await;
        assert_eq!(
            deb.live_keys(),
            0,
            "the completed force-run slot must be cleaned up"
        );
    }

    /// FIX 1 focused: a force-run must stop the prior task (cancel if in-flight,
    /// abort if pre-delay) before the latest task runs, so the prior and the new
    /// task never enrich concurrently. Exactly one work body completes, and the
    /// prior observes cancellation.
    ///
    /// Event-driven, not time-budgeted: every "is this task in-flight yet?" and
    /// "did the latest finish?" question is answered by a signal the work body
    /// fires (`started` / `done` `Notify`), never by a fixed real-time `settle()`
    /// racing a real-sleep loop. The CI flake this replaces came from a fixed
    /// 600ms budget that the latest task's ~300ms real-sleep loop could overrun
    /// under single-threaded load. With signals the test blocks until the work
    /// actually reaches each point, independent of scheduler load; a `timeout`
    /// guard (in `wait_signal`) still fails fast if a real hang ever regresses
    /// the logic.
    #[tokio::test]
    async fn test_force_run_cancels_prior_only_latest_completes() {
        use tokio::sync::Notify;

        let deb = ReflectionDebouncer::new();
        let completed = Arc::new(AtomicUsize::new(0));
        let prior_observed_cancel = Arc::new(AtomicUsize::new(0));

        // Prime the streak to the ceiling with in-flight cancels so the NEXT
        // schedule takes the force-run path. The first schedule has no prior to
        // cancel (no bump), so reaching a streak of MAX_CANCELLATIONS needs
        // MAX_CANCELLATIONS + 1 schedules. Each primed task signals `started`
        // the instant its work body runs (i.e. once it is in-flight); we await
        // that signal — not a fixed delay — before the next schedule, so every
        // cancellation is guaranteed to land on an in-flight task and count
        // toward the ceiling regardless of scheduler load.
        for _ in 0..=MAX_CANCELLATIONS {
            let started = Arc::new(Notify::new());
            let started_w = started.clone();
            deb.schedule("agentA", Duration::from_millis(5), move |cancel| {
                let started_w = started_w.clone();
                async move {
                    started_w.notify_one();
                    // Stay in-flight, respecting cancel, until superseded.
                    while !cancel.load(Ordering::SeqCst) {
                        tokio::task::yield_now().await;
                    }
                }
            });
            wait_signal(&started).await;
        }
        assert_eq!(
            deb.cancellations_for("agentA"),
            Some(MAX_CANCELLATIONS),
            "streak must sit at the ceiling so the next schedule force-runs"
        );

        // This in-flight "prior" is the one the upcoming force-run must stop. It
        // signals `started` once in-flight and records whether it observed the
        // cooperative-cancel flag before exiting.
        let prior_started = Arc::new(Notify::new());
        let prior_started_w = prior_started.clone();
        let po = prior_observed_cancel.clone();
        deb.schedule("agentA", Duration::from_millis(5), move |cancel| {
            let prior_started_w = prior_started_w.clone();
            let po = po.clone();
            async move {
                prior_started_w.notify_one();
                loop {
                    if cancel.load(Ordering::SeqCst) {
                        po.fetch_add(1, Ordering::SeqCst);
                        return; // cancelled — must NOT count as completed
                    }
                    tokio::task::yield_now().await;
                }
            }
        });
        // Await the prior actually being in-flight before the force-run lands.
        wait_signal(&prior_started).await;

        // Force-run: streak is at the ceiling, the prior above is in-flight, so
        // schedule() must flip the prior's cancel flag (not abort it) and spawn
        // this latest task, which runs to completion uncontested. The latest does
        // a fixed, bounded amount of cancel-respecting work (proving it is not
        // itself cancelled) and then signals `done`.
        let done = Arc::new(Notify::new());
        let done_w = done.clone();
        let cmp = completed.clone();
        deb.schedule("agentA", Duration::from_millis(5), move |cancel| {
            let done_w = done_w.clone();
            let cmp = cmp.clone();
            async move {
                for _ in 0..50 {
                    if cancel.load(Ordering::SeqCst) {
                        return; // would mean the latest was wrongly cancelled
                    }
                    tokio::task::yield_now().await;
                }
                cmp.fetch_add(1, Ordering::SeqCst);
                done_w.notify_one();
            }
        });

        // Block until the latest signals completion (timeout-guarded so a real
        // hang fails fast instead of waiting on a wall-clock budget).
        wait_signal(&done).await;
        assert_eq!(
            prior_observed_cancel.load(Ordering::SeqCst),
            1,
            "force-run must cooperatively cancel the in-flight prior task"
        );
        assert_eq!(
            completed.load(Ordering::SeqCst),
            1,
            "exactly one (the latest) reflection may complete; the cancelled prior must not"
        );
        // The slot self-cleans once the latest's work body returns; cleanup runs
        // after `done_w.notify_one()`, so poll briefly for it to drain rather
        // than asserting on a fixed budget.
        wait_until(|| deb.cancellations_for("agentA").is_none()).await;
        assert_eq!(
            deb.cancellations_for("agentA"),
            None,
            "after both tasks settle the force-run slot must be cleaned up"
        );
    }

    #[tokio::test]
    async fn test_completed_slot_is_cleaned_up() {
        let deb = ReflectionDebouncer::new();
        deb.schedule("agentA", Duration::from_millis(20), |_c| async {});
        deb.schedule("agentB", Duration::from_millis(20), |_c| async {});
        settle(200).await;
        assert_eq!(
            deb.live_keys(),
            0,
            "completed slots must be removed so the map cannot grow unbounded"
        );
    }
}
