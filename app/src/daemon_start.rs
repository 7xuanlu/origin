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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
/// spawn identically. Returns `Err(msg)` when the sidecar command can't be
/// created or spawned; the caller decides how to surface it.
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
    log::info!(
        "[daemon-start] Spawned wenlan-server daemon (pid {}, {}={})",
        child.pid(),
        data_dir_env,
        data_dir.display()
    );
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
