// SPDX-License-Identifier: AGPL-3.0-only
//! On-demand respawn of the wenlan-server sidecar (Diagnostics "Start Wenlan").
//!
//! `setup()` spawns the daemon once. If it dies later — e.g. an ephemeral
//! launchd job killed at logout — the app had no way back, and the Diagnostics
//! "Retry" button only re-probes. This module factors that one-time spawn into
//! [`spawn_daemon_sidecar`] and adds a guarded command that reuses it, so a
//! daemon-down red becomes healable from the UI.
//!
//! It does NOT manage the launchd plist: when launchd owns the daemon we defer
//! to it rather than fighting it with a second process.
//!
//! A sidecar this app spawned is this app's to stop: the child handle is kept
//! and [`stop_sidecar`] *tries* to end it on quit, and on Windows the process
//! is bound to a kill-on-close job object so a hard kill of the app takes it
//! too. Neither clause is categorical — a bind can fail, and so can a stop —
//! so both outcomes are values rather than log lines: [`JobBinding`] on the
//! handle, readable through [`sidecar_job_binding`], and
//! [`SidecarStopOutcome`] returned from every stop and readable through
//! [`last_sidecar_stop`]. Nothing may assume either guarantee held without
//! asking.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri_plugin_shell::process::CommandChild;
use tokio::sync::RwLock;

use crate::state::AppState;

/// Whether `setup()`'s server-plist preflight repair succeeded. Set once at
/// startup and read by the on-demand command, which must not re-run the
/// *mutating* preflight (that would be plist management from a user click).
static STARTUP_PREFLIGHT_OK: AtomicBool = AtomicBool::new(false);

/// Record the startup preflight outcome for the on-demand command to consult.
pub fn set_startup_preflight_ok(ok: bool) {
    STARTUP_PREFLIGHT_OK.store(ok, Ordering::Relaxed);
}

fn startup_preflight_ok() -> bool {
    STARTUP_PREFLIGHT_OK.load(Ordering::Relaxed)
}

/// Whether this app has ever spawned a sidecar without being able to measure
/// whether launchd already owned the daemon.
///
/// A latch, not a current state: the question it answers is "could this
/// machine be running two daemons for a reason nobody saw?", and one such
/// spawn is enough to make the answer yes for the life of the process. It
/// exists because a probe that resolves its own could-not-measure to "not
/// owned" leaves no trace a caller — or a bug report — can find; the log line
/// that used to stand in for this was not a record.
static SPAWNED_ON_UNKNOWN_OWNER: AtomicBool = AtomicBool::new(false);

/// Record that a sidecar was started on an unmeasured owner. Called from
/// [`spawn_for_owner_decision`] *after* the spawn succeeded, never from the
/// probe and never from a decision that has not yet spawned anything.
fn record_spawn_on_unknown_owner(context: &str) {
    SPAWNED_ON_UNKNOWN_OWNER.store(true, Ordering::Relaxed);
    log::warn!(
        "[daemon-start] {context}: launchd ownership could not be measured and the port was \
         silent, so this app started its own sidecar. If a launchd job does own the daemon, two \
         owners are possible; `wire_state().daemon.sidecar_spawned_on_unknown_owner` reports this."
    );
}

/// See [`SPAWNED_ON_UNKNOWN_OWNER`]. Carried onto the diagnostics wire beside
/// `sidecar_job_binding`.
pub fn spawned_on_unknown_owner() -> bool {
    SPAWNED_ON_UNKNOWN_OWNER.load(Ordering::Relaxed)
}

/// Whether an owner decision should latch [`SPAWNED_ON_UNKNOWN_OWNER`].
///
/// Round 3 (defect 3): both owner decisions used to call
/// [`record_spawn_on_unknown_owner`] *before* `spawn_daemon_sidecar`, so a
/// spawn that failed — an ENOENT sidecar, a quarantined bundle — latched the
/// flag forever and `daemon.sidecar_spawned_on_unknown_owner` reported a
/// second daemon that was never started. The latch's documented meaning is
/// "has ever spawned"; a spawn that did not happen cannot make the answer yes.
/// Pure so the ordering rule is testable without an `AppHandle`.
fn records_unknown_owner_spawn(owner_unknown: bool, spawn_succeeded: bool) -> bool {
    owner_unknown && spawn_succeeded
}

#[cfg(test)]
pub(crate) fn reset_spawned_on_unknown_owner_for_test() {
    SPAWNED_ON_UNKNOWN_OWNER.store(false, Ordering::Relaxed);
}

/// How many launchd registrations are in flight: `setup()`'s first-run
/// install and the "Run at Login" toggle each hold a [`LaunchdInstallPending`]
/// while they run `wenlan background on`. Above zero, the on-demand "Start
/// Wenlan" command must not spawn a sidecar that would race the job being
/// registered (first-run gauntlet finding F16).
static LAUNCHD_INSTALLS_PENDING: AtomicUsize = AtomicUsize::new(0);

/// Marks a launchd registration in flight for as long as it is held. The
/// count is released on drop, so a registration that panics or returns early
/// can never leave the "Start Wenlan" button stuck on `launchd_managed`.
pub struct LaunchdInstallPending(());

impl LaunchdInstallPending {
    pub fn begin() -> Self {
        LAUNCHD_INSTALLS_PENDING.fetch_add(1, Ordering::Relaxed);
        Self(())
    }
}

impl Drop for LaunchdInstallPending {
    fn drop(&mut self) {
        LAUNCHD_INSTALLS_PENDING.fetch_sub(1, Ordering::Relaxed);
    }
}

fn launchd_install_pending() -> bool {
    LAUNCHD_INSTALLS_PENDING.load(Ordering::Relaxed) > 0
}

/// Serialises each owner decision with the spawn it leads to. Without it the
/// startup path and the on-demand command could both read "nothing owns the
/// daemon" and each spawn a sidecar; the second spawn kills the first, and a
/// second child that had already seen the first one healthy exits 75, which
/// leaves zero owners.
static OWNER_DECISION: Mutex<()> = Mutex::new(());

/// Whether this app's own sidecar is in the slot: booting or serving.
///
/// Since round 3 the slot also holds a sidecar whose stop was not confirmed
/// (see [`stop_sidecar_inner`]), so this can read `true` for a daemon that has
/// in fact ended. That is a DECISION, not a measurement: "we could not
/// establish that it ended" must not be spent as "it is gone" by the one guard
/// standing between a click and a second daemon. It is bounded — the shell
/// plugin's own wait thread calls [`forget_sidecar`] the moment the child
/// really exits — and the caller
/// ([`decide_daemon_start`]) checks port health first, so the reachable cost is
/// a "Start Wenlan" click reporting `Started` during the window between the
/// process exiting and its `Terminated` event arriving.
fn sidecar_alive() -> bool {
    SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
}

/// Whether the sidecar is a member of the app's kill-on-close job object.
/// That membership is the *whole* of "a hard kill of the app ends the daemon"
/// on Windows: a Task Manager kill, a crash, or `Stop-Process` runs none of
/// the app's own shutdown code, so nothing else can end the child.
///
/// A failed binding used to be a `log::warn!` and nothing more — startup
/// continued and every surface downstream still read as though the daemon
/// died with the app. It is now recorded state ([`sidecar_job_binding`]) and
/// the shutdown path compensates for it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobBinding {
    /// The sidecar pid is in the kill-on-close job.
    Bound,
    /// The bind failed — a restricted token, a job assignment the OS refused,
    /// or a job object that could not be created. A hard kill of the app
    /// leaves this daemon running; only paths that execute app code still end
    /// it, which is why [`stop_sidecar`] now verifies its kill by identity.
    Unbound { reason: String },
    /// No job objects on this platform. macOS and Linux never bound the
    /// sidecar, and never claimed the crash case: their guarantee has always
    /// been the shutdown path alone.
    NotSupported,
}

/// A pid pinned to one process by its start time. Mirrors
/// `DaemonProcessIdentity` in the CLI's `commands/service.rs`, for the same
/// reason: a bare pid can be reused, and killing a reused pid is worse than
/// not killing at all.
#[derive(Clone, Copy, Debug)]
struct SidecarIdentity {
    pid: sysinfo::Pid,
    /// `None` means the process was gone before its identity could be read.
    /// That is a *failed measurement*, and it is never treated as a licence
    /// to kill whatever holds the pid now.
    started_at: Option<u64>,
}

/// Whether the recorded sidecar process is still there. The third state is
/// not decoration: an identity that was never captured can never be told apart
/// from a pid some *other* process now holds, and calling that "gone" is what
/// would let a caller report a daemon ended that is still serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessPresence {
    Running,
    Gone,
    Unknown,
}

/// What a kill attempt did. Returned rather than logged, because
/// [`stop_sidecar`]'s whole contract is that its caller can tell whether the
/// daemon ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KillAttempt {
    /// A kill was issued to the recorded process.
    Issued,
    /// The kill call itself returned failure.
    Failed,
    /// The recorded process was already gone.
    AlreadyGone,
    /// Refused: something holds the pid and this code cannot prove it is the
    /// sidecar (usually because `capture` never got a start time). Killing it
    /// could hit an unrelated process, so nothing is killed and nothing is
    /// claimed. THIS is the case that used to be silent.
    RefusedUnidentified,
}

impl SidecarIdentity {
    fn capture(raw_pid: u32) -> Self {
        let pid = sysinfo::Pid::from_u32(raw_pid);
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        let started_at = system.process(pid).map(sysinfo::Process::start_time);
        if started_at.is_none() {
            log::error!(
                "[daemon-start] could not read a start time for sidecar {pid}; it cannot be \
                 identified later, so the shutdown path will refuse to kill it by pid and will \
                 report its end as unmeasurable"
            );
        }
        Self { pid, started_at }
    }

    /// Whether the process this identity names is running right now.
    fn presence(&self) -> ProcessPresence {
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[self.pid]), true);
        presence_for(
            self.started_at,
            system.process(self.pid).map(sysinfo::Process::start_time),
        )
    }

    /// Kill this exact process if it is still running. A pid whose current
    /// occupant started at a different time is a different process and is left
    /// alone; so is a pid whose occupant cannot be matched at all.
    fn kill_if_still_running(&self) -> KillAttempt {
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[self.pid]), true);
        let current = system.process(self.pid);
        let attempt = match presence_for(self.started_at, current.map(sysinfo::Process::start_time))
        {
            ProcessPresence::Gone => KillAttempt::AlreadyGone,
            ProcessPresence::Unknown => KillAttempt::RefusedUnidentified,
            ProcessPresence::Running => match current {
                Some(process) if process.kill() => KillAttempt::Issued,
                // `Running` came from the same refresh, so `None` here would be
                // a race; either way no kill was issued.
                _ => KillAttempt::Failed,
            },
        };
        log::warn!(
            "[daemon-start] sidecar {} has no reaper outside this process; kill attempt: \
             {attempt:?}",
            self.pid
        );
        attempt
    }
}

/// Whether the process now holding a pid is the recorded one, still running.
/// Pure so the three-way answer can be tested without a process to kill.
///
/// `current: None` is a real measurement in every case — nothing holds the
/// pid, so whatever had it is gone. A pid held by a process whose start time
/// does not match is also `Gone`: ours ended and the pid was reused. Only "a
/// process is there and we never recorded what ours looked like" is unknown.
fn presence_for(recorded: Option<u64>, current: Option<u64>) -> ProcessPresence {
    match (recorded, current) {
        (_, None) => ProcessPresence::Gone,
        (Some(recorded), Some(current)) if recorded == current => ProcessPresence::Running,
        (Some(_), Some(_)) => ProcessPresence::Gone,
        (None, Some(_)) => ProcessPresence::Unknown,
    }
}

/// The sidecar this app spawned, so quitting can stop it. Empty when launchd
/// or the Task Scheduler owns the daemon, or once the sidecar exited. The
/// handle used to be dropped at spawn, so the daemon outlived the app on
/// every exit path (first-run gauntlet finding F14); the shell plugin only
/// kills children spawned through its JS `execute` command.
struct Sidecar {
    generation: u64,
    /// `None` once the handle has been spent. `CommandChild::kill` consumes
    /// the handle, so it is a one-shot: a stop that killed the child and then
    /// could not confirm the process ended has nothing left to kill *with*,
    /// but the record must stay in the slot anyway — the identity below is
    /// what a retry uses, and `sidecar_job_binding` must keep answering for a
    /// daemon that is still up.
    child: Option<CommandChild>,
    identity: SidecarIdentity,
    job_binding: JobBinding,
}

static SIDECAR: Mutex<Option<Sidecar>> = Mutex::new(None);

/// How the sidecar's job binding turned out, or `None` when this app owns no
/// sidecar (launchd or the Task Scheduler holds the daemon, or it has
/// exited). `None` is "nothing of ours to speak for", never a stand-in for
/// `Bound` — a caller that wants to know whether a hard kill of the app would
/// end the daemon must get `Some(JobBinding::Bound)` and nothing else.
pub fn sidecar_job_binding() -> Option<JobBinding> {
    SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|sidecar| sidecar.job_binding.clone())
}

/// Record what `job::bind` reported. Pure, so the mapping is testable without
/// a job object: an error must land on `Unbound` carrying its reason, never
/// on a silent success.
///
/// Compiled on every platform, not behind `cfg(windows)` with its Windows-only
/// call site, so the unit tests that hold this mapping run in every lane.
#[cfg_attr(not(windows), allow(dead_code))]
fn job_binding_for(bind_result: Result<(), String>) -> JobBinding {
    match bind_result {
        Ok(()) => JobBinding::Bound,
        Err(reason) => JobBinding::Unbound { reason },
    }
}

/// Whether the shutdown path must verify its kill by pid. Only a binding that
/// took makes the child handle sufficient; anything else is a daemon nothing
/// else in the system will reap.
///
/// Round 3 (defect 4): this used to match `Unbound` alone, which put
/// [`JobBinding::NotSupported`] on the same side of the branch as `Bound` —
/// i.e. "a job object will clean this up". That is the exact opposite of what
/// `NotSupported` says. `Bound` is the only value that names a reaper outside
/// this process; `Unbound` names a reaper that was asked for and refused, and
/// `NotSupported` names a platform where one was never available. The last two
/// differ only in *why* nothing else will end the daemon, and the shutdown
/// path has to compensate identically for both.
fn needs_pid_kill_fallback(binding: &JobBinding) -> bool {
    !matches!(binding, JobBinding::Bound)
}

/// Whether the stop path must reach for the identity kill, given the binding
/// *and* whether the one-shot child handle is still in hand.
///
/// A retry after a failed stop finds `child: None` — the handle was consumed
/// by the first attempt — so the pid kill is the only route left to the
/// process, whatever the binding says. It stays safe because
/// [`SidecarIdentity::kill_if_still_running`] refuses any pid it cannot match
/// to the recorded start time.
fn needs_pid_kill(binding: &JobBinding, child_handle_available: bool) -> bool {
    !child_handle_available || needs_pid_kill_fallback(binding)
}

/// Counts spawns. The shell plugin's own wait thread reaps a child that exits
/// on its own, and a later spawn can reuse its PID, so a termination event
/// clears the slot only when it names the same spawn.
static SIDECAR_GENERATION: AtomicU64 = AtomicU64::new(0);

/// How long [`stop_sidecar`] waits for the daemon to release its port after
/// the shutdown request before it kills the child: the daemon drains and
/// exits in about a second.
const SIDECAR_STOP_LIMIT: Duration = Duration::from_secs(3);

/// How long [`stop_sidecar`] waits for a killed process to actually leave the
/// process table before it reports `StillRunning`. A kill returns before the
/// OS reaps; without this the honest new outcome would report "still running"
/// on nearly every clean shutdown. Short, because it runs on the quit path.
const SIDECAR_REAP_LIMIT: Duration = Duration::from_millis(600);

/// What [`stop_sidecar`] established about the daemon it was asked to stop.
///
/// Three answers plus "there was nothing of ours to stop", and `Ended` is the
/// only one a caller may treat as the guarantee
/// `docs/cross-platform.md` describes. An identity that could not be captured
/// lands on `CouldNotMeasure`, never on `Ended`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum SidecarStopOutcome {
    /// This app owned no sidecar: a service holds the daemon, or it had
    /// already exited and been reaped. Nothing was stopped and nothing needed
    /// to be.
    NoSidecar,
    /// Measured: the process this app spawned is no longer running.
    Ended,
    /// Measured: it is still running.
    StillRunning { reason: String },
    /// Could not be measured either way.
    CouldNotMeasure { reason: String },
}

/// The outcome of the most recent [`stop_sidecar`], or `None` if it has never
/// run in this process. Kept so the *surviving* callers — the "Run at Login"
/// handover, which continues afterwards — leave a record a user can read,
/// exactly as `sidecar_job_binding` does for the job object.
static LAST_SIDECAR_STOP: Mutex<Option<SidecarStopOutcome>> = Mutex::new(None);

/// See [`LAST_SIDECAR_STOP`]. Carried onto the diagnostics wire.
pub fn last_sidecar_stop() -> Option<SidecarStopOutcome> {
    LAST_SIDECAR_STOP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Remember a freshly spawned sidecar; returns its spawn generation. A child
/// still in the slot is killed first: two sidecars would race for one port,
/// and the first would be left unowned.
fn remember_sidecar(
    child: CommandChild,
    identity: SidecarIdentity,
    job_binding: JobBinding,
) -> u64 {
    let generation = SIDECAR_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    let previous = SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .replace(Sidecar {
            generation,
            child: Some(child),
            identity,
            job_binding,
        });
    if let Some(old) = previous {
        let old_generation = old.generation;
        log::warn!(
            "[daemon-start] sidecar {} (spawn {old_generation}) was still in the slot; killing it",
            old.identity.pid
        );
        // A record left behind by a stop that failed has already spent its
        // handle; the identity is then the only way to reach the process, so
        // the fallback is required regardless of the binding.
        let needs_fallback = needs_pid_kill(&old.job_binding, old.child.is_some());
        match old.child {
            Some(child) => {
                if let Err(e) = child.kill() {
                    log::warn!("[daemon-start] could not kill the earlier sidecar: {e}");
                }
            }
            None => log::warn!(
                "[daemon-start] the earlier sidecar's child handle was already spent by a stop \
                 that did not confirm its end; only the pid kill is left"
            ),
        }
        // An earlier sidecar the job object never took is one nothing else
        // reaps, so confirm the kill landed on that exact process — and say so
        // when it did not, rather than dropping the answer.
        if needs_fallback {
            match old.identity.kill_if_still_running() {
                KillAttempt::Issued | KillAttempt::AlreadyGone => {}
                outcome @ (KillAttempt::Failed | KillAttempt::RefusedUnidentified) => {
                    log::error!(
                        "[daemon-start] the previous sidecar (spawn {old_generation}) may still be \
                         running: {outcome:?}"
                    );
                }
            }
        }
    }
    generation
}

/// Put a sidecar record in the slot with no child handle, as a stop that
/// spent the handle would leave it. Test-only: a real `CommandChild` cannot be
/// manufactured, and the stop path's behaviour around the slot is exactly what
/// defect 1 was about.
#[cfg(test)]
fn install_sidecar_for_test(identity: SidecarIdentity, job_binding: JobBinding) -> u64 {
    let generation = SIDECAR_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    *SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Sidecar {
        generation,
        child: None,
        identity,
        job_binding,
    });
    generation
}

#[cfg(test)]
fn clear_sidecar_for_test() {
    *SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    *LAST_SIDECAR_STOP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

fn forget_sidecar(generation: u64) {
    let mut slot = SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot
        .as_ref()
        .is_some_and(|sidecar| sidecar.generation == generation)
    {
        *slot = None;
    }
}

/// Stop the sidecar this app spawned, if any: ask the daemon to shut down
/// over HTTP (its own clean path, the one the quit flow uses), wait a bounded
/// time for it to release the port, and kill it only if it has not. The kill
/// goes through the child handle, which is a no-op once the child was reaped,
/// so it can never reach a process that reused the PID; a raw signal to the
/// numeric PID could. A daemon owned by launchd or the Task Scheduler is never
/// in the slot and is never touched. Without this a quit left the daemon
/// holding the port, and the next launch adopted the stale one (the upgrade
/// shape of finding F2).
///
/// When the job binding did not take, the child handle is the *only* thing
/// left that ends this daemon — no job object will reap it — so the kill is
/// verified against the recorded [`SidecarIdentity`] and reissued by pid if
/// the process is still there. That identity check is what makes killing by
/// pid safe: a pid whose occupant started at a different time is a different
/// process and is left alone.
///
/// Returns a [`SidecarStopOutcome`] rather than `()`, because every step
/// inside can fail — an identity that was never captured, a
/// `kill_if_still_running` whose bool nobody read, a `child.kill()` that
/// errored — and a `()` makes them indistinguishable. Callers branch on it,
/// and the last one is readable on the diagnostics wire.
pub async fn stop_sidecar() -> SidecarStopOutcome {
    let outcome = stop_sidecar_inner().await;
    let mut last = LAST_SIDECAR_STOP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *last = next_recorded_stop(last.as_ref(), outcome.clone());
    drop(last);
    outcome
}

/// What [`LAST_SIDECAR_STOP`] should hold after a stop reported `next`. Pure so
/// the one rule it encodes is testable.
///
/// Round 3 (defect 1): `stop_sidecar` used to overwrite unconditionally, and
/// the emptied slot guaranteed the *next* call would answer `NoSidecar`. So a
/// daemon that got away was recorded once and then the record was replaced by
/// "this app owned no sidecar" — the failure erased by a reassurance, on the
/// wire the user reads. `NoSidecar` is not a measurement of the daemon; it is
/// the absence of anything to measure, and it may not overwrite a measurement
/// that said the daemon was, or might be, still up. Every other outcome
/// (`Ended`, `StillRunning`, `CouldNotMeasure`) *is* a fresh reading of that
/// same sidecar and does replace the old one.
fn next_recorded_stop(
    previous: Option<&SidecarStopOutcome>,
    next: SidecarStopOutcome,
) -> Option<SidecarStopOutcome> {
    let erases_a_failure = matches!(next, SidecarStopOutcome::NoSidecar)
        && matches!(
            previous,
            Some(
                SidecarStopOutcome::StillRunning { .. }
                    | SidecarStopOutcome::CouldNotMeasure { .. }
            )
        );
    if erases_a_failure {
        previous.cloned()
    } else {
        Some(next)
    }
}

async fn stop_sidecar_inner() -> SidecarStopOutcome {
    // Read the slot, do not empty it. A `take()` before anything is known
    // drops the identity a retry needs, makes `sidecar_job_binding()` answer
    // `None` while a daemon still holds the port, and turns the next
    // `stop_sidecar()` into `NoSidecar`. The record leaves the slot in exactly
    // one place — after `Ended` — and nowhere else.
    //
    // The guard is dropped before the awaits below (repo invariant: never hold
    // a lock across `.await`); `identity` and `job_binding` are cheap copies
    // and the child handle is fetched back under a second, await-free lock
    // only if a kill turns out to be needed.
    let Some((generation, identity, job_binding)) = ({
        let slot = SIDECAR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot.as_ref().map(|sidecar| {
            (
                sidecar.generation,
                sidecar.identity,
                sidecar.job_binding.clone(),
            )
        })
    }) else {
        return SidecarStopOutcome::NoSidecar;
    };
    let pid = identity.pid;
    let client = crate::api::WenlanClient::new();
    if let Ok(http) = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        let _ = http
            .post(crate::lifecycle::shutdown_url_for(&client))
            .send()
            .await;
    }
    // Both paths differ in one thing only: whether a kill had to be issued.
    // Round 3 (defect 2): the port-release path used to return from its own
    // single presence read while only the kill path got the bounded reap poll,
    // so a daemon that closed its listener and spent a few milliseconds
    // flushing was reported `StillRunning`. There is now one measurement site
    // for both, so the two cannot drift apart again.
    let kill_attempt = if crate::lifecycle::wait_for_daemon_to_stop(SIDECAR_STOP_LIMIT).await {
        log::info!("[daemon-start] sidecar {pid} released the port after the shutdown request");
        // A released port is strong evidence and not the question asked. The
        // question is whether the process ended, so it is measured below — a
        // daemon that closed its listener and hung would otherwise be reported
        // as a clean stop.
        None
    } else {
        log::warn!(
            "[daemon-start] sidecar {pid} still holds the port after {SIDECAR_STOP_LIMIT:?}; \
             killing it"
        );
        kill_sidecar_child(generation, &identity, &job_binding)
    };
    let outcome = stop_outcome_after_reap(|| identity.presence(), kill_attempt).await;
    if outcome == SidecarStopOutcome::Ended {
        // The one place the record leaves the slot. Generation-matched, so a
        // sidecar respawned while this stop was in flight is never forgotten
        // by it.
        forget_sidecar(generation);
    } else {
        log::error!(
            "[daemon-start] sidecar {pid} was not confirmed ended ({outcome:?}); keeping its \
             record so a retry has an identity to kill and `sidecar_job_binding` still reports \
             the real binding"
        );
    }
    outcome
}

/// Spend the child handle on `generation`'s sidecar and, when nothing outside
/// this process would reap it, reissue the kill against the recorded identity.
/// Returns the identity kill's result, or `None` when the binding made it
/// unnecessary. Takes and releases the `SIDECAR` lock without awaiting.
fn kill_sidecar_child(
    generation: u64,
    identity: &SidecarIdentity,
    job_binding: &JobBinding,
) -> Option<KillAttempt> {
    let child = {
        let mut slot = SIDECAR
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot.as_mut()
            .filter(|sidecar| sidecar.generation == generation)
            .and_then(|sidecar| sidecar.child.take())
    };
    let pid = identity.pid;
    let had_handle = child.is_some();
    match child {
        Some(child) => match child.kill() {
            Ok(()) => log::info!("[daemon-start] killed sidecar {pid}"),
            Err(e) => log::warn!("[daemon-start] could not kill sidecar {pid}: {e}"),
        },
        // A retry: the first stop already consumed the one-shot handle.
        None => log::warn!(
            "[daemon-start] sidecar {pid} has no child handle left; the pid kill is the only \
             route to it"
        ),
    }
    needs_pid_kill(job_binding, had_handle).then(|| {
        // Safe by construction: the identity refuses any pid whose occupant
        // does not match the recorded start time.
        identity.kill_if_still_running()
    })
}

/// Poll `read_presence` until it stops saying `Running` or
/// [`SIDECAR_REAP_LIMIT`] passes, then map the reading to an outcome.
///
/// A kill — and a clean shutdown — is asynchronous on both platforms: the call
/// returns before the OS removes the process from the table. Reporting the
/// first read calls a dying daemon "still running" more often than not, which
/// is a failed measurement wearing the clothes of a negative one. Generic over
/// the reader so the retry itself is testable without a process to kill.
async fn stop_outcome_after_reap(
    mut read_presence: impl FnMut() -> ProcessPresence,
    kill_attempt: Option<KillAttempt>,
) -> SidecarStopOutcome {
    let mut presence = read_presence();
    let deadline = tokio::time::Instant::now() + SIDECAR_REAP_LIMIT;
    while presence == ProcessPresence::Running && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
        presence = read_presence();
    }
    stop_outcome_for(presence, kill_attempt)
}

/// What is known about the daemon after [`stop_sidecar`] ran. Pure so the
/// mapping from a presence reading to the reported outcome is testable without
/// a process to kill.
///
/// `RefusedUnidentified` is folded into `CouldNotMeasure` even when the
/// presence read says `Unknown` for the same underlying reason: they are the
/// same failure seen twice, and the reason string names it.
///
/// [`KillAttempt`]'s `Failed` and `RefusedUnidentified` differ only in the
/// reason string on the same `StillRunning`, and `Unknown` beside `Failed` is
/// unreachable (`kill_if_still_running` can only answer `Failed` from a
/// `Running` reading, which requires the start time `Unknown` says was never
/// captured). Every caller — quit, SIGTERM, the launchd handover — branches on
/// the [`SidecarStopOutcome`] arm, never on why the kill did not land.
fn stop_outcome_for(
    presence: ProcessPresence,
    kill_attempt: Option<KillAttempt>,
) -> SidecarStopOutcome {
    match presence {
        ProcessPresence::Gone => SidecarStopOutcome::Ended,
        ProcessPresence::Running => SidecarStopOutcome::StillRunning {
            reason: match kill_attempt {
                Some(KillAttempt::Failed) => "the kill by pid failed".to_string(),
                Some(KillAttempt::RefusedUnidentified) => {
                    "the pid could not be identified, so no kill was issued".to_string()
                }
                _ => {
                    "the process is still there after the shutdown request and the kill".to_string()
                }
            },
        },
        ProcessPresence::Unknown => SidecarStopOutcome::CouldNotMeasure {
            reason: "the sidecar's start time was never captured, so the process holding its pid \
                     cannot be identified as ours"
                .to_string(),
        },
    }
}

/// Windows has no parent-death signal. A job object with
/// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` ends every member when its last
/// handle closes, which the OS does when the app process ends for any
/// reason (Task Manager, a crash, `Stop-Process`). The app is the daemon's
/// only owner in sidecar mode, and an orphan kept the port and blocked the
/// next launch's daemon (first-run gauntlet finding F13).
#[cfg(windows)]
mod job {
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    /// One job for the app's lifetime. Its handle is never closed on
    /// purpose: closing it is what kills the members, and the OS closes it
    /// when the app ends. Zero means creating it failed.
    static JOB: OnceLock<usize> = OnceLock::new();

    fn create_job() -> usize {
        // SAFETY: plain Win32 calls with valid arguments; the handle is
        // checked before use and closed on the failure path.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                log::warn!("[daemon-start] CreateJobObjectW failed: {}", GetLastError());
                return 0;
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                log::warn!(
                    "[daemon-start] SetInformationJobObject failed: {}",
                    GetLastError()
                );
                CloseHandle(job);
                return 0;
            }
            job as usize
        }
    }

    /// Put `pid` in the app's kill-on-close job.
    pub fn bind(pid: u32) -> Result<(), String> {
        let job = *JOB.get_or_init(create_job) as HANDLE;
        if job.is_null() {
            return Err("the kill-on-close job object could not be created".into());
        }
        // SAFETY: the process handle is checked and closed here; the job
        // handle stays open by design (see `JOB`).
        unsafe {
            let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
            if process.is_null() {
                return Err(format!("OpenProcess({pid}) failed: {}", GetLastError()));
            }
            let ok = AssignProcessToJobObject(job, process);
            let error = GetLastError();
            CloseHandle(process);
            if ok == 0 {
                return Err(format!("AssignProcessToJobObject({pid}) failed: {error}"));
            }
        }
        Ok(())
    }
}

/// What the guards decided. Pure output of [`decide_daemon_start`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonStartDecision {
    /// The port already answers — never double-spawn.
    AlreadyRunning,
    /// This app's own sidecar is alive (booting or serving). One child is the
    /// most it ever runs: a second spawn kills the first.
    SidecarStarting,
    /// launchd holds a loaded job for the selected data root — it will
    /// (re)start the daemon; don't fight it.
    LaunchdManaged,
    /// The startup plist preflight failed — same skip as `setup()`.
    PreflightFailed,
    /// `setup()` is still registering the launchd job; it decides the owner
    /// when that settles, so don't spawn a rival meanwhile.
    LaunchdInstallPending,
    /// Nothing is serving and it's safe to spawn our own sidecar.
    Spawn,
    /// The same spawn, taken while launchd ownership could not be measured.
    /// The port was measured silent first, so this is the best move available;
    /// it is a separate variant so the caller records the uncertainty rather
    /// than reporting a clean owner decision it never made.
    SpawnOnUnknownOwner,
}

/// What `setup()` does with the sidecar once the first-run LaunchAgent
/// install has settled. Pure output of [`decide_startup_sidecar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupSidecar {
    /// A daemon already answers on the port. Whoever owns it, it is not this
    /// app's to double-start.
    SkipAlreadyServing,
    /// The startup plist preflight failed: no daemon start at all.
    SkipPreflightFailed,
    /// The server LaunchAgent targets the selected data root: launchd owns
    /// the daemon and restarts it; a sidecar would only fight it for the port.
    SkipLaunchdOwns,
    /// No usable LaunchAgent (install skipped or failed): the app runs its
    /// own sidecar, the fallback owner.
    Spawn,
    /// Same spawn, taken without knowing whether launchd owns the daemon —
    /// launchctl could not be read. Distinct from [`Self::Spawn`] because the
    /// caller must record it: the port was measured silent first, so a second
    /// daemon is unlikely, but "unlikely" is not the guarantee `Spawn` carries.
    SpawnOnUnknownOwner,
}

/// Owner decision for the startup path: launchd first, sidecar as fallback.
/// Pure so it is unit-testable without a running app.
///
/// `port_healthy` is measured by the caller before this runs, and it wins over
/// everything — including an unmeasurable launchd. That ordering is what makes
/// [`StartupSidecar::SpawnOnUnknownOwner`] safe enough to take: the only way
/// this app spawns a rival to a launchd job it could not see is if that job's
/// daemon was not answering its port at the moment of the check.
pub fn decide_startup_sidecar(
    preflight_ok: bool,
    port_healthy: bool,
    launchd_owns_daemon: crate::lifecycle::LaunchdOwnership,
) -> StartupSidecar {
    use crate::lifecycle::LaunchdOwnership;
    if port_healthy {
        StartupSidecar::SkipAlreadyServing
    } else if !preflight_ok {
        StartupSidecar::SkipPreflightFailed
    } else {
        match launchd_owns_daemon {
            LaunchdOwnership::Owns => StartupSidecar::SkipLaunchdOwns,
            LaunchdOwnership::DoesNot => StartupSidecar::Spawn,
            LaunchdOwnership::Unknown => StartupSidecar::SpawnOnUnknownOwner,
        }
    }
}

/// Guard order for the on-demand start. Pure so it is unit-testable without a
/// running app. Mirrors `setup()`'s guards, plus a port-health pre-check the
/// button needs that `setup()` does not: `setup()` runs before any daemon
/// could answer, but the button runs when one might already be back up.
///
/// `launchd_managed` is the tri-state, not a bool: an unreadable launchctl
/// used to arrive here already flattened into `false`, so this function could
/// not tell "launchd does not own it" from "nobody could say". It spawns in
/// both cases — the alternative leaves the user with no daemon — but only the
/// first is [`DaemonStartDecision::Spawn`].
pub fn decide_daemon_start(
    port_healthy: bool,
    sidecar_alive: bool,
    launchd_managed: crate::lifecycle::LaunchdOwnership,
    launchd_install_pending: bool,
    preflight_ok: bool,
) -> DaemonStartDecision {
    use crate::lifecycle::LaunchdOwnership;
    if port_healthy {
        DaemonStartDecision::AlreadyRunning
    } else if sidecar_alive {
        DaemonStartDecision::SidecarStarting
    } else if launchd_managed == LaunchdOwnership::Owns {
        DaemonStartDecision::LaunchdManaged
    } else if !preflight_ok {
        DaemonStartDecision::PreflightFailed
    } else if launchd_install_pending {
        DaemonStartDecision::LaunchdInstallPending
    } else if launchd_managed == LaunchdOwnership::Unknown {
        DaemonStartDecision::SpawnOnUnknownOwner
    } else {
        DaemonStartDecision::Spawn
    }
}

/// Result reported to the UI. Discriminated by `status` so the frontend can
/// tell "it's up" from "I started it" from "I couldn't".
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DaemonStartResult {
    Started,
    AlreadyRunning,
    LaunchdManaged,
    Failed { message: String },
}

/// Spawn the wenlan-server sidecar with the app-selected data root and pipe its
/// logs. Factored from `setup()` so the startup path and the on-demand command
/// spawn identically. The child is remembered for [`stop_sidecar`] (and bound
/// to the app's job object on Windows). Returns `Err(msg)` when the sidecar
/// command can't be created or spawned; the caller decides how to surface it.
pub fn spawn_daemon_sidecar(app: &tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_shell::ShellExt;
    let (data_dir_env, data_dir) = crate::identity_paths::sidecar_data_dir_env();
    let command = app
        .shell()
        .sidecar("wenlan-server")
        .map_err(|e| format!("Failed to create wenlan-server sidecar command: {e}"))?;
    let (mut rx, child) = command
        .env(data_dir_env, data_dir.as_os_str())
        .spawn()
        .map_err(|e| format!("Failed to spawn wenlan-server sidecar: {e}"))?;
    let pid = child.pid();
    log::info!(
        "[daemon-start] Spawned wenlan-server daemon (pid {pid}, {}={})",
        data_dir_env,
        data_dir.display()
    );
    // Windows has no parent-death signal, so this binding *is* the "killing
    // the app ends the daemon" guarantee. A failure is recorded on the handle
    // rather than logged and forgotten: startup still continues — refusing to
    // start would leave the user with no daemon at all, a worse outcome than
    // one that needs reaping — but the unbound state travels with the sidecar
    // so `stop_sidecar` can compensate and `sidecar_job_binding` can be asked.
    #[cfg(windows)]
    let job_binding = job_binding_for(job::bind(pid));
    #[cfg(not(windows))]
    let job_binding = JobBinding::NotSupported;
    if let JobBinding::Unbound { reason } = &job_binding {
        log::error!(
            "[daemon-start] sidecar {pid} is NOT bound to the app's job object ({reason}); a hard \
             kill of the app will leave this daemon running"
        );
    }
    let identity = SidecarIdentity::capture(pid);
    let generation = remember_sidecar(child, identity, job_binding);
    tauri::async_runtime::spawn(async move {
        use tauri_plugin_shell::process::CommandEvent;
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    log::info!("[daemon] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Stderr(line) => {
                    log::warn!("[daemon] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Terminated(status) => {
                    log::warn!("[daemon] exited: {:?}", status);
                    forget_sidecar(generation);
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

/// Spawn the sidecar for an owner decision and latch
/// [`SPAWNED_ON_UNKNOWN_OWNER`] only if the spawn actually happened.
///
/// The single site where the two owner decisions turn into a spawn, so the
/// ordering fixed in [`records_unknown_owner_spawn`] cannot be reintroduced by
/// one of them drifting: `result.is_ok()` is not a value that exists before
/// the spawn returns.
fn spawn_for_owner_decision(
    app: &tauri::AppHandle,
    owner_unknown: bool,
    context: &str,
) -> Result<(), String> {
    let result = spawn_daemon_sidecar(app);
    if records_unknown_owner_spawn(owner_unknown, result.is_ok()) {
        record_spawn_on_unknown_owner(context);
    }
    result
}

/// Start the daemon sidecar if — and only if — nothing already serves it.
/// Probes the port first (a daemon that came back on its own must not be
/// double-spawned), then defers to launchd, then honors the startup preflight,
/// and only then spawns. Returns a discriminated result the UI renders inline.
/// `setup()`'s owner decision once the first-run LaunchAgent install has
/// settled: launchd when it holds a loaded job for the selected data root,
/// otherwise this app's sidecar. Takes the install's [`LaunchdInstallPending`]
/// so the flag clears under the same lock that decides and spawns; the
/// on-demand command then sees either "still pending" or the final owner,
/// never the gap between the two.
///
/// The port-health probe is taken here, by the caller, before the lock. It is
/// the measurement the old `launchd_owns_server_daemon` comment *claimed* stood
/// between an unreadable launchctl and a double start, and which this path did
/// not in fact make — `setup()` used to decide the owner from the plist and the
/// preflight alone.
pub async fn settle_startup_owner(
    app: &tauri::AppHandle,
    preflight_ok: bool,
    pending: LaunchdInstallPending,
) {
    let port_healthy = crate::api::WenlanClient::new().health().await.is_ok();
    let _decision = OWNER_DECISION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    drop(pending);
    let launchd_owns_daemon =
        crate::lifecycle::launchd_owns_server_daemon(&crate::lifecycle::SystemLaunchctl);
    match decide_startup_sidecar(preflight_ok, port_healthy, launchd_owns_daemon) {
        StartupSidecar::SkipAlreadyServing => {
            log::info!("[init] a daemon already answers the port; no sidecar");
        }
        StartupSidecar::SkipPreflightFailed => {
            log::warn!("[init] skipping daemon sidecar because server plist preflight failed");
        }
        StartupSidecar::SkipLaunchdOwns => {
            log::info!("[init] launchd owns the daemon after the first-run install; no sidecar");
        }
        decision @ (StartupSidecar::Spawn | StartupSidecar::SpawnOnUnknownOwner) => {
            if let Err(e) = spawn_for_owner_decision(
                app,
                decision == StartupSidecar::SpawnOnUnknownOwner,
                "first-run install settled",
            ) {
                log::error!(
                    "[init] {e}. Run: xattr -cr /Applications/Origin.app or /Applications/Wenlan.app"
                );
            }
        }
    }
}

/// Start the daemon only when nothing owns it. Shared by the on-demand "Start
/// Wenlan" command and the "Run at Login" toggle's failure path. The health
/// probe awaits before the lock; everything the decision reads, and the spawn,
/// sit under it.
pub async fn start_daemon_if_unowned(
    app: &tauri::AppHandle,
    client: &crate::api::WenlanClient,
) -> DaemonStartResult {
    let port_healthy = client.health().await.is_ok();
    start_daemon_under_owner_lock(app, port_healthy)
}

fn start_daemon_under_owner_lock(app: &tauri::AppHandle, port_healthy: bool) -> DaemonStartResult {
    let _decision = OWNER_DECISION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let launchd_managed =
        crate::lifecycle::launchd_owns_server_daemon(&crate::lifecycle::SystemLaunchctl);
    match decide_daemon_start(
        port_healthy,
        sidecar_alive(),
        launchd_managed,
        launchd_install_pending(),
        startup_preflight_ok(),
    ) {
        DaemonStartDecision::AlreadyRunning => DaemonStartResult::AlreadyRunning,
        // Our own child is still booting: to the UI that is a start in
        // progress, not a failure and not a second daemon.
        DaemonStartDecision::SidecarStarting => DaemonStartResult::Started,
        // The job is being registered right now; to the UI that is the
        // same "the system service starts it" outcome.
        DaemonStartDecision::LaunchdManaged | DaemonStartDecision::LaunchdInstallPending => {
            DaemonStartResult::LaunchdManaged
        }
        DaemonStartDecision::PreflightFailed => DaemonStartResult::Failed {
            message: "Wenlan's startup configuration needs repair. Restart the app to fix it."
                .to_string(),
        },
        // Both spawn, and the UI sees the same "started" either way — there is
        // nothing a user can do about an unreadable launchctl from this
        // button. The difference is recorded, not rendered.
        decision @ (DaemonStartDecision::Spawn | DaemonStartDecision::SpawnOnUnknownOwner) => {
            match spawn_for_owner_decision(
                app,
                decision == DaemonStartDecision::SpawnOnUnknownOwner,
                "on-demand start",
            ) {
                Ok(()) => DaemonStartResult::Started,
                Err(e) => DaemonStartResult::Failed { message: e },
            }
        }
    }
}

#[tauri::command]
pub async fn start_daemon_sidecar(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<RwLock<AppState>>>,
) -> Result<DaemonStartResult, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    Ok(start_daemon_if_unowned(&app, &client).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::LaunchdOwnership;

    // Port-health wins over everything: a daemon that came back on its own must
    // never be double-spawned, even if launchd or preflight would say otherwise.
    #[test]
    fn healthy_port_reports_already_running_without_spawning() {
        assert_eq!(
            decide_daemon_start(true, false, LaunchdOwnership::DoesNot, false, true),
            DaemonStartDecision::AlreadyRunning
        );
        assert_eq!(
            decide_daemon_start(true, false, LaunchdOwnership::Owns, false, false),
            DaemonStartDecision::AlreadyRunning
        );
    }

    // launchd-owned but not answering: defer to launchd, do not spawn a rival.
    #[test]
    fn launchd_managed_defers_without_spawning() {
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::Owns, false, true),
            DaemonStartDecision::LaunchdManaged
        );
    }

    // Startup plist repair failed → same skip as setup(); do not spawn.
    #[test]
    fn preflight_failure_skips_spawn() {
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::DoesNot, false, false),
            DaemonStartDecision::PreflightFailed
        );
    }

    // Nothing serving, launchd absent, preflight ok → the one case that spawns.
    #[test]
    fn clear_field_spawns() {
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::DoesNot, false, true),
            DaemonStartDecision::Spawn
        );
    }

    // setup() is still registering the launchd job: a click must not spawn a
    // sidecar that races it for the port (F16). A healthy port still wins.
    #[test]
    fn pending_launchd_install_defers_the_sidecar() {
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::DoesNot, true, true),
            DaemonStartDecision::LaunchdInstallPending
        );
        assert_eq!(
            decide_daemon_start(true, false, LaunchdOwnership::DoesNot, true, true),
            DaemonStartDecision::AlreadyRunning
        );
    }

    // This app's sidecar is already booting: a second spawn would kill it, and
    // a second child that saw the first one healthy exits 75 — zero owners.
    #[test]
    fn a_live_sidecar_is_never_spawned_twice() {
        assert_eq!(
            decide_daemon_start(false, true, LaunchdOwnership::DoesNot, false, true),
            DaemonStartDecision::SidecarStarting
        );
        assert_eq!(
            decide_daemon_start(true, true, LaunchdOwnership::DoesNot, false, true),
            DaemonStartDecision::AlreadyRunning
        );
    }

    /// A2. The probe used to answer `false` for both "launchd does not own it"
    /// and "launchctl could not be read", so this decision could not tell them
    /// apart. Both spawn — the alternative is a user with no daemon — but they
    /// are different decisions, and only one of them is a clean owner call.
    #[test]
    fn an_unmeasured_launchd_owner_spawns_under_its_own_decision() {
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::Unknown, false, true),
            DaemonStartDecision::SpawnOnUnknownOwner
        );
        assert_ne!(
            decide_daemon_start(false, false, LaunchdOwnership::Unknown, false, true),
            decide_daemon_start(false, false, LaunchdOwnership::DoesNot, false, true),
            "'could not measure' must not collapse into 'measured not owned'"
        );
    }

    /// The claim the withdrawn comment made, substantiated where it is
    /// actually true: the caller's port-health measurement runs first, so an
    /// unmeasurable launchctl never reaches the spawn while a daemon is
    /// answering. Every guard ahead of the owner check also still wins, so an
    /// unknown owner cannot conjure a second sidecar past a live one or past a
    /// registration in flight.
    #[test]
    fn a_health_probe_or_a_live_sidecar_beats_an_unmeasured_owner() {
        assert_eq!(
            decide_daemon_start(true, false, LaunchdOwnership::Unknown, false, true),
            DaemonStartDecision::AlreadyRunning
        );
        assert_eq!(
            decide_daemon_start(false, true, LaunchdOwnership::Unknown, false, true),
            DaemonStartDecision::SidecarStarting
        );
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::Unknown, true, true),
            DaemonStartDecision::LaunchdInstallPending
        );
        assert_eq!(
            decide_daemon_start(false, false, LaunchdOwnership::Unknown, false, false),
            DaemonStartDecision::PreflightFailed
        );
    }

    /// The record the reviewer asked for: a log line inside the probe is not
    /// state. `spawned_on_unknown_owner` is, and `wire_state` reads it.
    #[test]
    #[serial_test::serial]
    fn spawning_on_an_unknown_owner_is_recorded_where_it_can_be_read() {
        reset_spawned_on_unknown_owner_for_test();
        assert!(
            !spawned_on_unknown_owner(),
            "the clean case must be a measured negative"
        );
        record_spawn_on_unknown_owner("test");
        assert!(spawned_on_unknown_owner());
        reset_spawned_on_unknown_owner_for_test();
    }

    // Two registrations can overlap (the first-run install and the Run at
    // Login toggle): the flag holds until the last one releases, and releases
    // on drop so an aborted install cannot pin the button on launchd_managed.
    #[test]
    #[serial_test::serial]
    fn pending_flag_counts_overlapping_installs_and_releases_on_drop() {
        assert!(!launchd_install_pending());
        let first = LaunchdInstallPending::begin();
        let second = LaunchdInstallPending::begin();
        drop(first);
        assert!(launchd_install_pending());
        drop(second);
        assert!(!launchd_install_pending());
    }

    // A job binding that failed must survive as state, not evaporate into a
    // log line. `Unbound` carries the reason so the shutdown path and any
    // caller of `sidecar_job_binding` can act on it; only a bind that
    // actually succeeded may read as `Bound`.
    #[test]
    fn a_failed_job_binding_is_recorded_as_unbound_with_its_reason() {
        assert_eq!(job_binding_for(Ok(())), JobBinding::Bound);
        assert_eq!(
            job_binding_for(Err("AssignProcessToJobObject(4321) failed: 5".to_string())),
            JobBinding::Unbound {
                reason: "AssignProcessToJobObject(4321) failed: 5".to_string()
            }
        );
    }

    // The guarantee "a hard kill of the app ends the daemon" is true only for
    // `Bound`. Everything else has to be compensated for on the way out, and
    // the platform that never had job objects is not silently folded into the
    // bound case either.
    //
    // Defect 4: `NotSupported` used to answer `false` here, i.e. it sat on the
    // "a job object will reap this" side of the branch while saying there is
    // no job object at all. The two failing bindings differ only in why
    // nothing outside this process will end the daemon.
    #[test]
    fn only_a_bound_sidecar_needs_no_kill_verification() {
        assert!(!needs_pid_kill_fallback(&JobBinding::Bound));
        assert!(needs_pid_kill_fallback(&JobBinding::Unbound {
            reason: "restricted token".to_string()
        }));
        assert!(
            needs_pid_kill_fallback(&JobBinding::NotSupported),
            "'this platform has no job objects' is not 'the job object will handle it'"
        );
        assert_ne!(JobBinding::NotSupported, JobBinding::Bound);
    }

    /// The one-shot child handle is the other half of the question. A retry
    /// after a stop that spent the handle has only the pid route left, whatever
    /// the binding says — and `kill_if_still_running` is what keeps that safe.
    #[test]
    fn a_spent_child_handle_forces_the_pid_route_whatever_the_binding() {
        assert!(!needs_pid_kill(&JobBinding::Bound, true));
        assert!(
            needs_pid_kill(&JobBinding::Bound, false),
            "a bound sidecar whose handle is gone still has to be reached by pid"
        );
        assert!(needs_pid_kill(&JobBinding::NotSupported, true));
        assert!(needs_pid_kill(&JobBinding::NotSupported, false));
    }

    /// Defect 3. The latch means "this app has ever spawned a sidecar without
    /// being able to see whether launchd already owned the daemon". Both call
    /// sites used to set it *before* `spawn_daemon_sidecar`, so a spawn that
    /// failed — the ENOENT/quarantine case that returns `Failed` to the UI —
    /// still made `daemon.sidecar_spawned_on_unknown_owner` report a second
    /// daemon that was never started, for the life of the process.
    #[test]
    fn a_spawn_that_failed_is_not_a_spawn_on_an_unknown_owner() {
        assert!(records_unknown_owner_spawn(true, true));
        assert!(
            !records_unknown_owner_spawn(true, false),
            "a spawn that failed started nothing, so it cannot make 'two owners are possible' true"
        );
        assert!(!records_unknown_owner_spawn(false, true));
        assert!(!records_unknown_owner_spawn(false, false));
    }

    /// Defect 1, the record half. `stop_sidecar` overwrote `LAST_SIDECAR_STOP`
    /// unconditionally, and the emptied slot guaranteed the next call would
    /// answer `NoSidecar` — so the only record that a daemon got away was
    /// replaced, on the diagnostics wire, by "this app owned no sidecar".
    #[test]
    fn a_no_sidecar_answer_may_not_erase_a_recorded_failure() {
        let still_running = SidecarStopOutcome::StillRunning {
            reason: "held the port".to_string(),
        };
        let unmeasured = SidecarStopOutcome::CouldNotMeasure {
            reason: "no start time".to_string(),
        };

        assert_eq!(
            next_recorded_stop(Some(&still_running), SidecarStopOutcome::NoSidecar),
            Some(still_running.clone()),
            "'nothing of ours to stop' is not a measurement of the daemon that got away"
        );
        assert_eq!(
            next_recorded_stop(Some(&unmeasured), SidecarStopOutcome::NoSidecar),
            Some(unmeasured.clone())
        );

        // Everything else is a fresh reading of the same sidecar and wins.
        assert_eq!(
            next_recorded_stop(Some(&still_running), SidecarStopOutcome::Ended),
            Some(SidecarStopOutcome::Ended),
            "a later stop that measured the end must be able to clear the failure"
        );
        assert_eq!(
            next_recorded_stop(Some(&still_running), unmeasured.clone()),
            Some(unmeasured)
        );
        assert_eq!(
            next_recorded_stop(None, SidecarStopOutcome::NoSidecar),
            Some(SidecarStopOutcome::NoSidecar)
        );
        assert_eq!(
            next_recorded_stop(
                Some(&SidecarStopOutcome::Ended),
                SidecarStopOutcome::NoSidecar
            ),
            Some(SidecarStopOutcome::NoSidecar)
        );
    }

    /// Defect 2, the poll itself. A kill and a clean exit are both
    /// asynchronous: the process is still in the table for a moment after the
    /// port comes free. Reporting the first read turns that moment into
    /// `StillRunning`, which is a failed measurement dressed as a negative one.
    #[tokio::test]
    async fn the_reap_poll_outlasts_a_process_that_is_on_its_way_out() {
        let reads = std::cell::Cell::new(0u32);
        let outcome = stop_outcome_after_reap(
            || {
                let taken = reads.get();
                reads.set(taken + 1);
                if taken == 0 {
                    ProcessPresence::Running
                } else {
                    ProcessPresence::Gone
                }
            },
            None,
        )
        .await;
        assert_eq!(outcome, SidecarStopOutcome::Ended);
        assert!(
            reads.get() > 1,
            "the first read was reported without a retry: {} read(s)",
            reads.get()
        );
    }

    /// A process that never leaves still has to be reported honestly, and the
    /// poll must be bounded — this runs on the quit path.
    #[tokio::test]
    async fn the_reap_poll_gives_up_and_reports_a_process_that_stayed() {
        let started = std::time::Instant::now();
        let outcome =
            stop_outcome_after_reap(|| ProcessPresence::Running, Some(KillAttempt::Failed)).await;
        assert!(matches!(outcome, SidecarStopOutcome::StillRunning { .. }));
        assert!(
            started.elapsed() < SIDECAR_REAP_LIMIT * 4,
            "the poll must be bounded: {:?}",
            started.elapsed()
        );
    }

    /// Point `WenlanClient::new()` at port 0. Every test below drives the real
    /// `stop_sidecar`, which POSTs `/api/shutdown` and then probes the port —
    /// neither may ever reach a real daemon, least of all the developer's on
    /// the default 7878. Port 0 is not an address a listener can hold, so
    /// there is nothing to reach by construction.
    ///
    /// It is also the only choice that reaches the port-release path on this
    /// host. A refused connection to a *closed* loopback port takes ~2s here
    /// (measured), so `wait_for_daemon_to_stop`'s health probe plus its TCP
    /// probe exceed `SIDECAR_STOP_LIMIT` and every stop would fall through to
    /// the kill path. Port 0 fails with `AddrNotAvailable` in well under a
    /// millisecond, so the wait returns `true` and the fast path is exercised.
    fn point_the_client_at_a_port_nothing_can_hold() -> crate::test_env::EnvGuard {
        let guard = crate::test_env::EnvGuard::capture(&["WENLAN_PORT", "ORIGIN_PORT"]);
        std::env::set_var("WENLAN_PORT", "0");
        std::env::remove_var("ORIGIN_PORT");
        guard
    }

    /// A real, live process to stand in for the sidecar. Never the test
    /// process's own pid: the stop path can issue a kill, and a test that can
    /// kill its own runner is not a test.
    fn spawn_a_live_process() -> std::process::Child {
        let mut command = if cfg!(windows) {
            let mut command = std::process::Command::new("ping");
            command.args(["-n", "60", "127.0.0.1"]);
            command
        } else {
            let mut command = std::process::Command::new("sleep");
            command.arg("60");
            command
        };
        command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn a helper process")
    }

    /// The recorded identity of `child`, once the process table has it. A
    /// capture that missed would land on `CouldNotMeasure` and test something
    /// other than what these tests are about, so it is a precondition here
    /// rather than an assertion under test.
    fn identity_of(child: &std::process::Child) -> SidecarIdentity {
        for _ in 0..100 {
            let identity = SidecarIdentity::capture(child.id());
            if identity.started_at.is_some() && identity.presence() == ProcessPresence::Running {
                return identity;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let last = SidecarIdentity::capture(child.id());
        panic!(
            "the helper process (pid {}) never read back as running: started_at={:?}, \
             presence={:?}",
            child.id(),
            last.started_at,
            last.presence()
        );
    }

    /// Defect 1, end to end. `stop_sidecar_inner` began by `take()`ing the slot
    /// before any outcome was known. When it then reported `StillRunning` or
    /// `CouldNotMeasure`: the identity a retry needs was dropped on the floor,
    /// `sidecar_job_binding()` answered `None` while a daemon still held the
    /// port, and the next `stop_sidecar()` answered `NoSidecar` — which then
    /// overwrote the recorded failure.
    #[tokio::test]
    #[serial_test::serial]
    async fn a_stop_that_did_not_end_the_daemon_keeps_the_sidecar_for_a_retry() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
        let _ports = point_the_client_at_a_port_nothing_can_hold();
        clear_sidecar_for_test();

        let mut helper = spawn_a_live_process();
        let binding = JobBinding::Unbound {
            reason: "restricted token".to_string(),
        };
        install_sidecar_for_test(identity_of(&helper), binding.clone());

        let first = stop_sidecar().await;

        assert!(
            matches!(first, SidecarStopOutcome::StillRunning { .. }),
            "a process that is plainly still running is a measured failure: {first:?}"
        );
        assert_eq!(
            sidecar_job_binding(),
            Some(binding),
            "the wire must keep reporting the real binding while the daemon is up"
        );

        let second = stop_sidecar().await;
        assert!(
            matches!(second, SidecarStopOutcome::StillRunning { .. }),
            "the retry found nothing to stop: {second:?}"
        );
        assert!(
            matches!(
                last_sidecar_stop(),
                Some(SidecarStopOutcome::StillRunning { .. })
            ),
            "the recorded failure was replaced: {:?}",
            last_sidecar_stop()
        );

        clear_sidecar_for_test();
        let _ = helper.kill();
        let _ = helper.wait();
    }

    /// Defect 2, end to end. The port-release path reported one immediate
    /// presence read, so a daemon that closed its listener and then spent a few
    /// milliseconds flushing before exiting was called `StillRunning` — and,
    /// with defect 1, had its retry handle destroyed on the way out. Both paths
    /// now measure at one site.
    #[tokio::test]
    #[serial_test::serial]
    async fn a_daemon_that_exits_just_after_releasing_its_port_is_reported_ended() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
        let _ports = point_the_client_at_a_port_nothing_can_hold();
        clear_sidecar_for_test();

        let mut helper = spawn_a_live_process();
        install_sidecar_for_test(identity_of(&helper), JobBinding::NotSupported);

        // The flush: still in the process table when the port is already free,
        // gone well inside SIDECAR_REAP_LIMIT.
        let flushing = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            let _ = helper.kill();
            let _ = helper.wait();
        });

        let outcome = stop_sidecar().await;
        flushing.join().expect("the helper's exit");

        assert_eq!(
            outcome,
            SidecarStopOutcome::Ended,
            "the port-release path reported the first read instead of waiting for the reap"
        );
        assert_eq!(
            sidecar_job_binding(),
            None,
            "a measured end is the one case that clears the slot"
        );
        clear_sidecar_for_test();
    }

    // An empty slot means this app owns no sidecar — launchd or the Task
    // Scheduler holds the daemon, or it exited. That is not a binding, and it
    // must never answer with the reassuring one.
    #[test]
    #[serial_test::serial]
    fn an_empty_sidecar_slot_reports_no_binding_rather_than_a_bound_one() {
        assert_eq!(sidecar_job_binding(), None);
    }

    // The identity check that makes a kill-by-pid safe, and the reading the
    // stop outcome is built from. Only one of these four is "our process is
    // running"; the pid nothing holds and the pid a *later* process reused are
    // both real measurements that ours is gone; and an identity that was never
    // captured, beside a pid something still holds, is the one honest unknown.
    #[test]
    fn an_unmeasured_process_identity_is_unknown_not_a_match() {
        assert_eq!(
            presence_for(Some(1_700_000_000), Some(1_700_000_000)),
            ProcessPresence::Running
        );
        assert_eq!(
            presence_for(Some(1_700_000_000), Some(1_700_000_042)),
            ProcessPresence::Gone,
            "a pid reused by a later process means ours ended; it must never match"
        );
        assert_eq!(
            presence_for(Some(1_700_000_000), None),
            ProcessPresence::Gone,
            "a pid nothing holds is a measured end"
        );
        assert_eq!(
            presence_for(None, Some(1_700_000_000)),
            ProcessPresence::Unknown,
            "an identity that was never captured cannot be matched against anything"
        );
    }

    // `capture` reads a real start time for a live process, so the recorded
    // identity of a running sidecar matches itself. Uses this test process,
    // which is certainly alive; nothing is killed here.
    #[test]
    fn capture_records_a_start_time_for_a_live_process() {
        let me = sysinfo::get_current_pid().expect("current process id");
        let identity = SidecarIdentity::capture(me.as_u32());
        assert_eq!(identity.pid, me);
        assert!(
            identity.started_at.is_some(),
            "a live process must yield a start time, or every later kill check degrades to 'unknown'"
        );
        assert_eq!(identity.presence(), ProcessPresence::Running);
    }

    /// A3. `stop_sidecar` used to return `()`, so an identity that was never
    /// captured, a refused kill and a clean shutdown were the same nothing.
    /// The mapping is now a value, and the uncapturable identity is
    /// `CouldNotMeasure` — never `Ended`.
    #[test]
    fn an_uncapturable_identity_reports_could_not_measure_not_success() {
        assert_eq!(
            stop_outcome_for(ProcessPresence::Gone, Some(KillAttempt::Issued)),
            SidecarStopOutcome::Ended
        );
        assert_eq!(
            stop_outcome_for(ProcessPresence::Gone, None),
            SidecarStopOutcome::Ended
        );

        let refused = stop_outcome_for(
            ProcessPresence::Unknown,
            Some(KillAttempt::RefusedUnidentified),
        );
        assert!(
            matches!(refused, SidecarStopOutcome::CouldNotMeasure { .. }),
            "an identity that could not be captured is not a stop: {refused:?}"
        );
        assert_ne!(refused, SidecarStopOutcome::Ended);

        let failed = stop_outcome_for(ProcessPresence::Running, Some(KillAttempt::Failed));
        match failed {
            SidecarStopOutcome::StillRunning { reason } => {
                assert!(reason.contains("kill"), "the reason must name it: {reason}")
            }
            other => panic!("a live process after the kill is not a stop: {other:?}"),
        }
    }

    // Startup owner: launchd first, sidecar only when no LaunchAgent took the
    // daemon, nothing at all when the preflight failed, and — the guard this
    // path never had — nothing when a daemon already answers the port.
    #[test]
    fn startup_sidecar_is_the_fallback_owner() {
        assert_eq!(
            decide_startup_sidecar(true, false, LaunchdOwnership::Owns),
            StartupSidecar::SkipLaunchdOwns
        );
        assert_eq!(
            decide_startup_sidecar(true, false, LaunchdOwnership::DoesNot),
            StartupSidecar::Spawn
        );
        assert_eq!(
            decide_startup_sidecar(false, false, LaunchdOwnership::DoesNot),
            StartupSidecar::SkipPreflightFailed
        );
        assert_eq!(
            decide_startup_sidecar(false, false, LaunchdOwnership::Owns),
            StartupSidecar::SkipPreflightFailed
        );
        assert_eq!(
            decide_startup_sidecar(true, true, LaunchdOwnership::DoesNot),
            StartupSidecar::SkipAlreadyServing
        );
    }

    /// A2, startup half. `settle_startup_owner` took the probe's flattened
    /// `false` and spawned as if the owner were known. It now spawns under a
    /// decision of its own, and only after the caller has measured the port —
    /// which is the check the old comment claimed prevented a double start and
    /// which this path did not in fact perform.
    #[test]
    fn an_unmeasured_owner_at_startup_spawns_only_against_a_silent_port() {
        assert_eq!(
            decide_startup_sidecar(true, false, LaunchdOwnership::Unknown),
            StartupSidecar::SpawnOnUnknownOwner
        );
        assert_ne!(
            decide_startup_sidecar(true, false, LaunchdOwnership::Unknown),
            StartupSidecar::Spawn,
            "'could not measure' must not collapse into 'measured not owned'"
        );
        assert_eq!(
            decide_startup_sidecar(true, true, LaunchdOwnership::Unknown),
            StartupSidecar::SkipAlreadyServing,
            "an answering daemon is never double-started, least of all on a guess"
        );
    }
}
