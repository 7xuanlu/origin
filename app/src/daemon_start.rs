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
//! and [`stop_sidecar`] ends it on quit, and on Windows the process is bound
//! to a kill-on-close job object so a hard kill of the app takes it too.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// The sidecar this app spawned, so quitting can stop it. Empty when launchd
/// or the Task Scheduler owns the daemon, or once the sidecar exited. The
/// handle used to be dropped at spawn, so the daemon outlived the app on
/// every exit path (first-run gauntlet finding F14); the shell plugin only
/// kills children spawned through its JS `execute` command.
static SIDECAR: Mutex<Option<(u64, CommandChild)>> = Mutex::new(None);

/// Counts spawns. The shell plugin's own wait thread reaps a child that exits
/// on its own, and a later spawn can reuse its PID, so a termination event
/// clears the slot only when it names the same spawn.
static SIDECAR_GENERATION: AtomicU64 = AtomicU64::new(0);

/// How long [`stop_sidecar`] waits for the daemon to release its port after
/// the shutdown request before it kills the child: the daemon drains and
/// exits in about a second.
const SIDECAR_STOP_LIMIT: Duration = Duration::from_secs(3);

/// Remember a freshly spawned sidecar; returns its spawn generation. A child
/// still in the slot is killed first: two sidecars would race for one port,
/// and the first would be left unowned.
fn remember_sidecar(child: CommandChild) -> u64 {
    let generation = SIDECAR_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    let previous = SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .replace((generation, child));
    if let Some((old_generation, old)) = previous {
        log::warn!(
            "[daemon-start] sidecar {} (spawn {old_generation}) was still in the slot; killing it",
            old.pid()
        );
        if let Err(e) = old.kill() {
            log::warn!("[daemon-start] could not kill the earlier sidecar: {e}");
        }
    }
    generation
}

fn forget_sidecar(generation: u64) {
    let mut slot = SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot
        .as_ref()
        .is_some_and(|(spawned, _)| *spawned == generation)
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
pub async fn stop_sidecar() {
    let Some((_, child)) = SIDECAR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    else {
        return;
    };
    let pid = child.pid();
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
    if crate::lifecycle::wait_for_daemon_to_stop(SIDECAR_STOP_LIMIT).await {
        log::info!("[daemon-start] sidecar {pid} released the port after the shutdown request");
        return;
    }
    log::warn!(
        "[daemon-start] sidecar {pid} still holds the port after {SIDECAR_STOP_LIMIT:?}; killing it"
    );
    match child.kill() {
        Ok(()) => log::info!("[daemon-start] killed sidecar {pid}"),
        Err(e) => log::warn!("[daemon-start] could not kill sidecar {pid}: {e}"),
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
    /// launchd owns a matching daemon — it will restart it; don't fight it.
    LaunchdManaged,
    /// The startup plist preflight failed — same skip as `setup()`.
    PreflightFailed,
    /// Nothing is serving and it's safe to spawn our own sidecar.
    Spawn,
}

/// Guard order for the on-demand start. Pure so it is unit-testable without a
/// running app. Mirrors `setup()`'s guards, plus a port-health pre-check the
/// button needs that `setup()` does not: `setup()` runs before any daemon
/// could answer, but the button runs when one might already be back up.
pub fn decide_daemon_start(
    port_healthy: bool,
    launchd_managed: bool,
    preflight_ok: bool,
) -> DaemonStartDecision {
    if port_healthy {
        DaemonStartDecision::AlreadyRunning
    } else if launchd_managed {
        DaemonStartDecision::LaunchdManaged
    } else if !preflight_ok {
        DaemonStartDecision::PreflightFailed
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
    #[cfg(windows)]
    if let Err(e) = job::bind(pid) {
        log::warn!("[daemon-start] sidecar {pid} is not bound to the app's job object: {e}");
    }
    let generation = remember_sidecar(child);
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

/// Start the daemon sidecar if — and only if — nothing already serves it.
/// Probes the port first (a daemon that came back on its own must not be
/// double-spawned), then defers to launchd, then honors the startup preflight,
/// and only then spawns. Returns a discriminated result the UI renders inline.
#[tauri::command]
pub async fn start_daemon_sidecar(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<RwLock<AppState>>>,
) -> Result<DaemonStartResult, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let port_healthy = client.health().await.is_ok();
    let launchd_managed = crate::lifecycle::current_server_plist_matches_selected_data_dir();

    Ok(
        match decide_daemon_start(port_healthy, launchd_managed, startup_preflight_ok()) {
            DaemonStartDecision::AlreadyRunning => DaemonStartResult::AlreadyRunning,
            DaemonStartDecision::LaunchdManaged => DaemonStartResult::LaunchdManaged,
            DaemonStartDecision::PreflightFailed => DaemonStartResult::Failed {
                message: "Wenlan's startup configuration needs repair. Restart the app to fix it."
                    .to_string(),
            },
            DaemonStartDecision::Spawn => match spawn_daemon_sidecar(&app) {
                Ok(()) => DaemonStartResult::Started,
                Err(e) => DaemonStartResult::Failed { message: e },
            },
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port-health wins over everything: a daemon that came back on its own must
    // never be double-spawned, even if launchd or preflight would say otherwise.
    #[test]
    fn healthy_port_reports_already_running_without_spawning() {
        assert_eq!(
            decide_daemon_start(true, false, true),
            DaemonStartDecision::AlreadyRunning
        );
        assert_eq!(
            decide_daemon_start(true, true, false),
            DaemonStartDecision::AlreadyRunning
        );
    }

    // launchd-owned but not answering: defer to launchd, do not spawn a rival.
    #[test]
    fn launchd_managed_defers_without_spawning() {
        assert_eq!(
            decide_daemon_start(false, true, true),
            DaemonStartDecision::LaunchdManaged
        );
    }

    // Startup plist repair failed → same skip as setup(); do not spawn.
    #[test]
    fn preflight_failure_skips_spawn() {
        assert_eq!(
            decide_daemon_start(false, false, false),
            DaemonStartDecision::PreflightFailed
        );
    }

    // Nothing serving, launchd absent, preflight ok → the one case that spawns.
    #[test]
    fn clear_field_spawns() {
        assert_eq!(
            decide_daemon_start(false, false, true),
            DaemonStartDecision::Spawn
        );
    }
}
