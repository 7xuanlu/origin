use tauri::AppHandle;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::UpdaterExt;

/// Check for an update on startup. If one exists, prompt the user; on accept,
/// download + install + relaunch. Failures are logged, never blocking.
pub async fn check_and_prompt(app: AppHandle) {
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
        return;
    }

    if let Err(e) = update.download_and_install(|_, _| {}, || {}).await {
        log::error!("update install failed: {e}");
        return;
    }

    app.restart();
}
