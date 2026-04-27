use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::UpdaterExt;

/// How long a "Later" dismissal of a given version suppresses re-prompts.
const SUPPRESS_TTL: Duration = Duration::from_secs(24 * 3600);

/// Delay before the first update check so the main app window has time to
/// paint — otherwise the dialog can appear before the window, leaving the
/// user unsure which app is asking.
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

/// Check for an update on startup. If one exists and the user hasn't dismissed
/// it within the last 24 hours, prompt the user; on accept, download + install
/// + relaunch. Failures are logged, never blocking.
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

    // OkCancelCustom: clicking the red X / pressing Esc returns false (Cancel).
    let accepted = app
        .dialog()
        .message(format!(
            "Origin {version} is available. Download and install now?"
        ))
        .kind(MessageDialogKind::Info)
        .title("Update available")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Install".into(),
            "Later".into(),
        ))
        .blocking_show();

    if !accepted {
        record_dismissal(&app, &version);
        return;
    }

    if let Err(e) = update.download_and_install(|_, _| {}, || {}).await {
        log::error!("update install failed: {e}");
        return;
    }

    app.restart();
}
