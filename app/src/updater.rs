use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Listener, Manager};
use tauri_plugin_updater::UpdaterExt;

const SUPPRESS_TTL: Duration = Duration::from_secs(24 * 3600);
const STARTUP_DELAY: Duration = Duration::from_secs(3);

#[derive(Serialize, Deserialize, Debug)]
struct DismissedUpdate {
    version: String,
    dismissed_at_secs: u64,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn dismissal_path(app: &AppHandle) -> Option<PathBuf> {
    let dir = app.path().app_data_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("updater-dismissed.json"))
}

fn was_recently_dismissed(app: &AppHandle, version: &str) -> bool {
    let Some(path) = dismissal_path(app) else {
        return false;
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return false;
    };
    let Ok(entry) = serde_json::from_slice::<DismissedUpdate>(&bytes) else {
        return false;
    };
    entry.version == version
        && now_secs().saturating_sub(entry.dismissed_at_secs) < SUPPRESS_TTL.as_secs()
}

fn record_dismissal(app: &AppHandle, version: &str) {
    if let Some(path) = dismissal_path(app) {
        let entry = DismissedUpdate {
            version: version.to_string(),
            dismissed_at_secs: now_secs(),
        };
        if let Ok(bytes) = serde_json::to_vec(&entry) {
            let _ = std::fs::write(path, bytes);
        }
    }
}

/// Emit `updater://available` to the main webview and wait for the user's
/// choice via the `updater://action` event. The actual UI is rendered by
/// `UpdaterDialog` inside the main window's React tree (see
/// `src/components/UpdaterDialog.tsx`), so the dialog stays attached to the
/// app window and travels with it.
async fn prompt_via_overlay(app: &AppHandle, version: &str) -> bool {
    let _ = app.emit(
        "updater://available",
        serde_json::json!({ "version": version }),
    );

    let (tx, rx) = tokio::sync::oneshot::channel::<bool>();
    let tx = Arc::new(Mutex::new(Some(tx)));

    let tx_action = Arc::clone(&tx);
    let action_id = app.listen("updater://action", move |event| {
        let payload = event.payload();
        let install = payload.contains("install");
        if let Ok(mut g) = tx_action.lock() {
            if let Some(sender) = g.take() {
                let _ = sender.send(install);
            }
        }
    });

    let accepted = rx.await.unwrap_or(false);
    app.unlisten(action_id);
    accepted
}

/// Check for an update on startup. If one exists and the user hasn't dismissed
/// it within the last 24 hours, prompt via an in-app overlay; on accept,
/// download + install + relaunch with progress events. Failures are logged
/// and surfaced in the overlay, never blocking.
pub async fn check_and_prompt(app: AppHandle) {
    tokio::time::sleep(STARTUP_DELAY).await;

    let updater = match app.updater() {
        Ok(u) => u,
        Err(e) => {
            log::warn!("updater unavailable: {e}");
            return;
        }
    };

    let update = match updater.check().await {
        Ok(Some(u)) => u,
        Ok(None) => return,
        Err(e) => {
            log::warn!("update check failed: {e}");
            return;
        }
    };

    let version = update.version.clone();

    if was_recently_dismissed(&app, &version) {
        log::info!("update v{version} suppressed (dismissed within 24h)");
        return;
    }

    let accepted = prompt_via_overlay(&app, &version).await;

    if !accepted {
        record_dismissal(&app, &version);
        return;
    }

    let app_chunk = app.clone();
    let on_chunk = move |chunk_len: usize, total: Option<u64>| {
        let _ = app_chunk.emit(
            "updater://progress",
            serde_json::json!({
                "chunk": chunk_len,
                "total": total,
            }),
        );
    };
    let app_done = app.clone();
    let on_done = move || {
        let _ = app_done.emit("updater://progress", serde_json::json!({ "done": true }));
    };

    if let Err(e) = update.download_and_install(on_chunk, on_done).await {
        log::error!("update install failed: {e}");
        let _ = app.emit(
            "updater://progress",
            serde_json::json!({ "error": format!("{e}") }),
        );
        return;
    }

    tokio::time::sleep(Duration::from_millis(800)).await;
    app.restart();
}
