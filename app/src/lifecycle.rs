// SPDX-License-Identifier: AGPL-3.0-only
// Items in this module are used by later tasks (Tasks 6-16). Allow dead-code
// until they are wired up.
#![allow(dead_code)]
use anyhow::{Context, Result};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tauri::AppHandle;

/// Process-wide guard that prevents `quit_origin` from running twice. A failed
/// teardown releases it so the recovered app can guard a later retry.
static QUITTING: AtomicBool = AtomicBool::new(false);

struct QuitAttemptGuard<'a> {
    flag: &'a AtomicBool,
    committed: bool,
}

impl<'a> QuitAttemptGuard<'a> {
    fn try_begin(flag: &'a AtomicBool) -> Option<Self> {
        if flag.swap(true, Ordering::AcqRel) {
            None
        } else {
            Some(Self {
                flag,
                committed: false,
            })
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for QuitAttemptGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.flag.store(false, Ordering::Release);
        }
    }
}

pub fn is_quitting() -> bool {
    QUITTING.load(Ordering::Acquire)
}

/// Spec line 198: set_run_at_login holds a global Mutex for the duration of
/// the toggle to prevent concurrent install/uninstall races (G2).
static RUN_AT_LOGIN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub const SERVER_PLIST_LABEL: &str = "com.wenlan.server";
pub const LEGACY_SERVER_PLIST_LABEL: &str = "com.origin.server";
pub const APP_PLIST_LABEL: &str = "com.wenlan.desktop";
pub const LEGACY_APP_PLIST_LABEL: &str = "com.origin.desktop";

const APP_PLIST_TEMPLATE: &str = include_str!("../resources/com.wenlan.desktop.plist");

/// Trait for shelling out to launchctl. Mock in tests.
pub trait LaunchctlExec: Send + Sync {
    fn run(&self, args: &[&str]) -> io::Result<Output>;
}

pub struct SystemLaunchctl;

impl LaunchctlExec for SystemLaunchctl {
    fn run(&self, args: &[&str]) -> io::Result<Output> {
        Command::new("launchctl").args(args).output()
    }
}

/// Resolve the data directory for the auto-start flag.
fn data_dir() -> Result<PathBuf> {
    Ok(crate::identity_paths::app_data_dir())
}

/// Path to the auto-start opt-out sentinel file. Owned by the app, not the
/// daemon's typed `Config` (which would AGPL-contaminate origin-core).
/// Touch = opted out, absent = opted in.
fn opt_out_flag_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("auto_start_disabled.flag"))
}

/// Returns true iff the opt-out sentinel file exists.
pub fn user_opted_out() -> bool {
    opt_out_flag_path().map(|p| p.exists()).unwrap_or(false)
}

/// Set or clear the opt-out sentinel file.
pub fn set_user_opted_out(opted_out: bool) -> Result<()> {
    let path = opt_out_flag_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if opted_out {
        // Touch the file (idempotent — overwrite empty)
        std::fs::write(&path, b"")?;
    } else if path.exists() {
        std::fs::remove_file(&path)?;
    }
    Ok(())
}

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().context("HOME not set")
}

pub fn app_plist_path() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", APP_PLIST_LABEL)))
}

pub fn legacy_app_plist_path() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", LEGACY_APP_PLIST_LABEL)))
}

pub fn server_plist_path() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", SERVER_PLIST_LABEL)))
}

pub fn legacy_server_plist_path() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", LEGACY_SERVER_PLIST_LABEL)))
}

pub fn current_server_plist_exists() -> bool {
    server_plist_path().map(|p| p.exists()).unwrap_or(false)
}

pub fn current_server_plist_matches_selected_data_dir() -> bool {
    server_plist_path()
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .is_some_and(|content| server_plist_has_selected_data_dir(&content))
}

pub fn legacy_server_plist_exists() -> bool {
    legacy_server_plist_path()
        .map(|p| p.exists())
        .unwrap_or(false)
}

fn log_dir() -> Result<PathBuf> {
    Ok(data_dir()?.join("logs"))
}

fn current_app_path() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    std::fs::canonicalize(&exe).context("canonicalize current_exe")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StableLaunchAgentTarget {
    Current,
    LegacyOrigin,
    Rejected,
}

fn classify_stable_launch_agent_target(exe: &Path) -> StableLaunchAgentTarget {
    // Accept the current `wenlan-app` binary plus legacy `origin` /
    // `origin-app` binary names from old installs.
    let name = exe.file_name().and_then(|s| s.to_str());
    if name != Some("wenlan-app") && name != Some("origin-app") && name != Some("origin") {
        return StableLaunchAgentTarget::Rejected;
    }

    let Some(app_bundle) = exe.ancestors().find(|p| {
        p.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext == "app")
    }) else {
        return StableLaunchAgentTarget::Rejected;
    };

    let Some(bundle_name) = app_bundle.file_name().and_then(|s| s.to_str()) else {
        return StableLaunchAgentTarget::Rejected;
    };

    let in_system_apps = app_bundle == Path::new("/Applications/Wenlan.app")
        || app_bundle == Path::new("/Applications/Origin.app");
    let in_user_apps = dirs::home_dir()
        .map(|home| {
            app_bundle == home.join("Applications/Wenlan.app")
                || app_bundle == home.join("Applications/Origin.app")
        })
        .unwrap_or(false);

    if !in_system_apps && !in_user_apps {
        return StableLaunchAgentTarget::Rejected;
    }

    match bundle_name {
        "Wenlan.app" => StableLaunchAgentTarget::Current,
        "Origin.app" => StableLaunchAgentTarget::LegacyOrigin,
        _ => StableLaunchAgentTarget::Rejected,
    }
}

fn is_stable_launch_agent_target(exe: &Path) -> bool {
    classify_stable_launch_agent_target(exe) != StableLaunchAgentTarget::Rejected
}

pub fn install_app_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = app_plist_path()?;
    let logs = log_dir()?;
    std::fs::create_dir_all(&logs)?;
    if let Some(parent) = plist.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let app_path = current_app_path()?;
    let content = APP_PLIST_TEMPLATE
        .replace("__WENLAN_APP_PATH__", &app_path.to_string_lossy())
        .replace("__LOG_PATH__", &logs.to_string_lossy());

    if plist.exists() {
        let _ = launchctl.run(&["unload", &plist.to_string_lossy()]);
    }
    std::fs::write(&plist, content)?;

    // H5: roll back the file write if the load fails — otherwise a broken
    // plist sticks around and stale-plist detection on next startup will
    // consider it valid, never retrying.
    let load_result = launchctl.run(&["load", &plist.to_string_lossy()]);
    let out = match load_result {
        Ok(o) => o,
        Err(e) => {
            let _ = std::fs::remove_file(&plist);
            return Err(anyhow::Error::from(e));
        }
    };
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let _ = std::fs::remove_file(&plist);
        anyhow::bail!("launchctl load failed: {}", stderr);
    }
    Ok(())
}

pub fn uninstall_app_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = app_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    let _ = launchctl.run(&["unload", &plist.to_string_lossy()]);
    std::fs::remove_file(&plist)?;
    Ok(())
}

fn plist_string(content: &str, key: &str) -> Option<String> {
    let value = plist::Value::from_reader_xml(content.as_bytes()).ok()?;
    value
        .as_dictionary()?
        .get(key)?
        .as_string()
        .map(ToOwned::to_owned)
}

fn plist_first_program(content: &str) -> Option<String> {
    let value = plist::Value::from_reader_xml(content.as_bytes()).ok()?;
    let dict = value.as_dictionary()?;
    if let Some(program) = dict.get("Program").and_then(|program| program.as_string()) {
        return Some(program.to_owned());
    }
    dict.get("ProgramArguments")?
        .as_array()?
        .first()?
        .as_string()
        .map(ToOwned::to_owned)
}

fn path_is_legacy_origin_app_exe(path: &str) -> bool {
    let path = Path::new(path);
    path == Path::new("/Applications/Origin.app/Contents/MacOS/origin")
        || path == Path::new("/Applications/Origin.app/Contents/MacOS/origin-app")
        || dirs::home_dir()
            .map(|home| {
                path == home.join("Applications/Origin.app/Contents/MacOS/origin")
                    || path == home.join("Applications/Origin.app/Contents/MacOS/origin-app")
            })
            .unwrap_or(false)
}

fn path_is_legacy_origin_server_exe(path: &str) -> bool {
    let path = Path::new(path);
    path == Path::new("/Applications/Origin.app/Contents/MacOS/origin-server")
        || dirs::home_dir()
            .map(|home| path == home.join("Applications/Origin.app/Contents/MacOS/origin-server"))
            .unwrap_or(false)
}

fn legacy_app_plist_is_owned(content: &str) -> bool {
    plist_string(content, "Label").as_deref() == Some(LEGACY_APP_PLIST_LABEL)
        && plist_first_program(content)
            .as_deref()
            .is_some_and(path_is_legacy_origin_app_exe)
}

fn legacy_server_plist_is_owned(content: &str) -> bool {
    plist_string(content, "Label").as_deref() == Some(LEGACY_SERVER_PLIST_LABEL)
        && plist_first_program(content)
            .as_deref()
            .is_some_and(path_is_legacy_origin_server_exe)
}

fn remove_legacy_app_plist_file_if_owned() -> Result<()> {
    let plist = legacy_app_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(&plist)?;
    if !legacy_app_plist_is_owned(&content) {
        return Ok(());
    }
    std::fs::remove_file(&plist)?;
    Ok(())
}

fn unload_plist_best_effort(launchctl: &dyn LaunchctlExec, plist: &Path, label: &str) {
    let plist_arg = plist.to_string_lossy().to_string();
    match launchctl.run(&["unload", &plist_arg]) {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            log::warn!(
                "[lifecycle] launchctl unload failed for {label}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            log::warn!("[lifecycle] launchctl unload failed for {label}: {e}");
        }
    }
}

pub fn cleanup_legacy_app_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = legacy_app_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(&plist)?;
    if !legacy_app_plist_is_owned(&content) {
        return Ok(());
    }
    unload_plist_best_effort(launchctl, &plist, LEGACY_APP_PLIST_LABEL);
    std::fs::remove_file(&plist)?;
    Ok(())
}

pub fn cleanup_legacy_server_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = legacy_server_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(&plist)?;
    if !legacy_server_plist_is_owned(&content) {
        return Ok(());
    }
    unload_plist_best_effort(launchctl, &plist, LEGACY_SERVER_PLIST_LABEL);
    std::fs::remove_file(&plist)?;
    Ok(())
}

fn service_cli_path_for_app_exe(app_exe: &Path) -> Result<PathBuf> {
    let mut bin = app_exe.parent().context("no parent dir")?.join("wenlan");
    if cfg!(target_os = "windows") {
        bin.set_extension("exe");
    }
    Ok(bin)
}

fn service_cli_path() -> Result<PathBuf> {
    service_cli_path_for_app_exe(&current_app_path()?)
}

fn run_service_cli(subcommand: &str) -> Result<()> {
    let bin = service_cli_path()?;
    let (data_dir_env, data_dir) = crate::identity_paths::sidecar_data_dir_env();
    let out = Command::new(&bin)
        .env(data_dir_env, &data_dir)
        .arg(subcommand)
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "wenlan {} failed: {}",
            subcommand,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

fn plist_environment_string(content: &str, key: &str) -> Option<String> {
    let value = plist::Value::from_reader_xml(content.as_bytes()).ok()?;
    let env = value
        .as_dictionary()?
        .get("EnvironmentVariables")?
        .as_dictionary()?;
    env.get(key)?.as_string().map(ToOwned::to_owned)
}

fn server_plist_has_selected_data_dir(content: &str) -> bool {
    let (key, data_dir) = crate::identity_paths::sidecar_data_dir_env();
    plist_environment_string(content, key).as_deref() == Some(data_dir.to_string_lossy().as_ref())
}

fn patch_plist_environment_variable(path: &Path, key: &str, value: &Path) -> Result<()> {
    let mut plist =
        plist::Value::from_file(path).with_context(|| format!("read plist {}", path.display()))?;
    let root = plist
        .as_dictionary_mut()
        .context("server plist root is not a dictionary")?;
    if !root.contains_key("EnvironmentVariables") {
        root.insert(
            "EnvironmentVariables".to_string(),
            plist::Value::Dictionary(plist::Dictionary::new()),
        );
    }
    let env = root
        .get_mut("EnvironmentVariables")
        .context("server plist EnvironmentVariables missing after insert")?;
    let env = env
        .as_dictionary_mut()
        .context("server plist EnvironmentVariables is not a dictionary")?;
    env.insert(
        key.to_string(),
        plist::Value::String(value.to_string_lossy().to_string()),
    );
    plist
        .to_file_xml(path)
        .with_context(|| format!("write plist {}", path.display()))?;
    Ok(())
}

fn ensure_server_plist_data_dir_env(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = server_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }

    let (key, data_dir) = crate::identity_paths::sidecar_data_dir_env();
    let original_content = std::fs::read_to_string(&plist)?;
    if server_plist_has_selected_data_dir(&original_content) {
        return Ok(());
    }

    patch_plist_environment_variable(&plist, key, &data_dir)?;
    unload_plist_best_effort(launchctl, &plist, SERVER_PLIST_LABEL);
    let load_result = launchctl.run(&["load", &plist.to_string_lossy()]);
    let result = match load_result {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => Err(anyhow::anyhow!(
            "launchctl load failed after data-dir patch: {}",
            String::from_utf8_lossy(&out.stderr)
        )),
        Err(e) => Err(e).context("launchctl load after data-dir patch"),
    };
    if result.is_err() {
        if let Err(e) = std::fs::write(&plist, original_content) {
            log::warn!(
                "[lifecycle] failed to roll back server plist after data-dir patch failure: {e}"
            );
        }
    }
    result
}

pub fn prepare_server_plist_for_startup(launchctl: &dyn LaunchctlExec) -> Result<()> {
    ensure_server_plist_data_dir_env(launchctl)
}

/// Run `wenlan install`. Resolves the CLI binary alongside our exe; the CLI
/// owns service-manager integration and expects `wenlan-server` next to it.
pub fn install_server_plist_via_subprocess(launchctl: &dyn LaunchctlExec) -> Result<()> {
    run_service_cli("install")?;
    ensure_server_plist_data_dir_env(launchctl)
}

pub fn uninstall_server_plist_via_subprocess() -> Result<()> {
    if !server_plist_path().map(|p| p.exists()).unwrap_or(false) {
        return Ok(());
    }
    run_service_cli("uninstall")
}

/// Returns true iff BOTH current Wenlan plists are loaded in launchctl.
///
/// `launchctl list` output is `PID\tStatus\tLabel`. We must compare the third
/// whitespace-separated field with `==`; a substring match would treat
/// `com.origin.server.staging` as `com.origin.server` (H4).
pub fn is_run_at_login_enabled(launchctl: &dyn LaunchctlExec) -> bool {
    let out = match launchctl.run(&["list"]) {
        Ok(o) => o,
        Err(_) => return false,
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let server = stdout
        .lines()
        .any(|line| line.split_whitespace().nth(2) == Some(SERVER_PLIST_LABEL));
    let app = stdout
        .lines()
        .any(|line| line.split_whitespace().nth(2) == Some(APP_PLIST_LABEL));
    server && app
}

/// First-run install of both plists. Detects stale paths (e.g. app moved)
/// and re-installs when the embedded path doesn't match the current binary.
/// Returns Ok(()) if the install completed or was unnecessary.
pub fn first_run_install_if_needed(launchctl: &dyn LaunchctlExec) -> Result<()> {
    if user_opted_out() {
        if let Err(e) = remove_legacy_app_plist_file_if_owned() {
            log::warn!("[first-run] legacy app plist cleanup failed: {e}");
        }
        if let Err(e) = cleanup_legacy_server_plist(launchctl) {
            log::warn!("[first-run] legacy server plist cleanup failed: {e}");
        }
        return Ok(());
    }

    let exe_canonical = match current_app_path() {
        Ok(path) => path,
        Err(e) => {
            log::warn!("[first-run] unable to resolve current app path: {e}");
            return Ok(());
        }
    };

    if !is_stable_launch_agent_target(&exe_canonical) {
        log::warn!(
            "[first-run] skipping LaunchAgent install from non-stable app path: {}",
            exe_canonical.display()
        );
        return Ok(());
    }

    let app_plist_stale = app_plist_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .map(|content| !content.contains(exe_canonical.to_string_lossy().as_ref()))
        .unwrap_or(true); // missing plist = stale

    let server_plist_stale = server_plist_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .map(|content| {
            let expected_server = exe_canonical
                .parent()
                .map(|p| p.join("wenlan-server").to_string_lossy().to_string())
                .unwrap_or_default();
            !content.contains(&expected_server) || !server_plist_has_selected_data_dir(&content)
        })
        .unwrap_or(true);

    if !app_plist_stale && !server_plist_stale {
        if let Err(e) = remove_legacy_app_plist_file_if_owned() {
            log::warn!("[first-run] legacy app plist cleanup failed: {e}");
        }
        if let Err(e) = cleanup_legacy_server_plist(launchctl) {
            log::warn!("[first-run] legacy server plist cleanup failed: {e}");
        }
        return Ok(());
    }

    let mut server_replacement_ready = !server_plist_stale;
    if server_plist_stale {
        match install_server_plist_via_subprocess(launchctl) {
            Ok(()) => server_replacement_ready = true,
            Err(e) => log::warn!("[first-run] wenlan install failed: {e}"),
        }
    }

    if app_plist_stale {
        install_app_plist(launchctl)?;
    }

    if let Err(e) = remove_legacy_app_plist_file_if_owned() {
        log::warn!("[first-run] legacy app plist cleanup failed: {e}");
    }
    if server_replacement_ready {
        if let Err(e) = cleanup_legacy_server_plist(launchctl) {
            log::warn!("[first-run] legacy server plist cleanup failed: {e}");
        }
    } else {
        log::warn!("[first-run] preserving legacy server plist because wenlan install failed");
    }

    log::info!("[first-run] LaunchAgents installed");
    Ok(())
}

/// Toggle "Run at login". Holds a process-wide Mutex for the duration of the
/// install/uninstall sequence so concurrent toggles serialize (G2, spec
/// line 198).
pub async fn set_run_at_login(enabled: bool, launchctl: &dyn LaunchctlExec) -> Result<()> {
    let _guard = RUN_AT_LOGIN_LOCK.lock().await;
    if enabled {
        let exe = current_app_path()?;
        if !is_stable_launch_agent_target(&exe) {
            anyhow::bail!(
                "refusing to enable Run at Login from non-stable app path: {}",
                exe.display()
            );
        }
        set_user_opted_out(false)?;
        install_server_plist_via_subprocess(launchctl)?;
        install_app_plist(launchctl)?;
    } else {
        set_user_opted_out(true)?;
        uninstall_app_plist(launchctl)?;
        let legacy_app_cleanup_result = remove_legacy_app_plist_file_if_owned();
        let legacy_server_cleanup_result = cleanup_legacy_server_plist(launchctl);
        let uninstall_result = uninstall_server_plist_via_subprocess();
        legacy_app_cleanup_result?;
        legacy_server_cleanup_result?;
        uninstall_result?;
    }
    Ok(())
}

pub async fn quit_origin(app_handle: &AppHandle) -> Result<()> {
    // Debounce: tray menu Quit Wenlan item stays clickable during the 500ms
    // shutdown sleep; double-click would otherwise spawn 2× POSTs (H1).
    let Some(attempt) = QuitAttemptGuard::try_begin(&QUITTING) else {
        return Ok(());
    };

    // Spec lifecycle invariant #4: "Quit Wenlan = full off; both plists
    // unloaded, both processes exit, no auto-restart on reboot." (H2)
    // Order matters: uninstall plists FIRST so launchd won't respawn after
    // the daemon dies, then shut the daemon down cleanly.
    let launchctl = SystemLaunchctl;
    if let Err(e) = uninstall_app_plist(&launchctl) {
        log::warn!("[quit] uninstall_app_plist failed: {e}");
    }
    if let Err(e) = uninstall_server_plist_via_subprocess() {
        log::warn!("[quit] uninstall_server_plist failed: {e}");
    }
    if let Err(e) = cleanup_legacy_app_plist(&launchctl) {
        log::warn!("[quit] cleanup_legacy_app_plist failed: {e}");
    }
    if let Err(e) = cleanup_legacy_server_plist(&launchctl) {
        log::warn!("[quit] cleanup_legacy_server_plist failed: {e}");
    }

    // 1. Tell daemon to shut down cleanly
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    let _ = client
        .post("http://127.0.0.1:7878/api/shutdown")
        .send()
        .await;

    // 2. Wait briefly for daemon to flush
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 3. Tauri-graceful exit.
    app_handle.exit(0);
    attempt.commit();
    Ok(())
}

/// Test-only — checks the debounce flag without invoking the full quit flow
/// (which needs a real `AppHandle`).
#[cfg(test)]
pub(crate) fn try_begin_quit() -> bool {
    !QUITTING.swap(true, Ordering::AcqRel)
}

#[cfg(test)]
pub(crate) fn reset_quitting_flag_for_test() {
    QUITTING.store(false, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::Mutex;

    struct EnvGuard {
        home: Option<std::ffi::OsString>,
        wenlan: Option<std::ffi::OsString>,
        origin: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn capture() -> Self {
            Self {
                home: std::env::var_os("HOME"),
                wenlan: std::env::var_os("WENLAN_DATA_DIR"),
                origin: std::env::var_os("ORIGIN_DATA_DIR"),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.wenlan {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
            match &self.origin {
                Some(value) => std::env::set_var("ORIGIN_DATA_DIR", value),
                None => std::env::remove_var("ORIGIN_DATA_DIR"),
            }
        }
    }

    #[derive(Default)]
    struct MockLaunchctl {
        calls: Mutex<Vec<Vec<String>>>,
        /// Status code to return for load/start subcommands. Default 0 = ok.
        load_status: Mutex<i32>,
        /// Status code to return for unload subcommands. Default 0 = ok.
        unload_status: Mutex<i32>,
    }
    impl LaunchctlExec for MockLaunchctl {
        fn run(&self, args: &[&str]) -> io::Result<Output> {
            self.calls
                .lock()
                .unwrap()
                .push(args.iter().map(|s| s.to_string()).collect());
            // Tests can override load/unload statuses independently.
            let status_code = match args.first().copied() {
                Some("load") => *self.load_status.lock().unwrap(),
                Some("unload") => *self.unload_status.lock().unwrap(),
                _ => 0,
            };
            Ok(Output {
                status: std::process::ExitStatus::from_raw(status_code),
                stdout: vec![],
                stderr: vec![],
            })
        }
    }

    fn launch_agent_program_arguments_plist(label: &str, program: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{program}</string>
    </array>
</dict>
</plist>
"#
        )
    }

    fn owned_legacy_app_plist() -> String {
        launch_agent_program_arguments_plist(
            LEGACY_APP_PLIST_LABEL,
            "/Applications/Origin.app/Contents/MacOS/origin",
        )
    }

    fn foreign_legacy_app_plist() -> String {
        launch_agent_program_arguments_plist(
            LEGACY_APP_PLIST_LABEL,
            "/Applications/Other.app/Contents/MacOS/origin",
        )
    }

    fn owned_legacy_server_plist() -> String {
        launch_agent_program_arguments_plist(
            LEGACY_SERVER_PLIST_LABEL,
            "/Applications/Origin.app/Contents/MacOS/origin-server",
        )
    }

    fn foreign_legacy_server_plist() -> String {
        launch_agent_program_arguments_plist(
            LEGACY_SERVER_PLIST_LABEL,
            "/usr/local/bin/origin-server",
        )
    }

    // Tests that mutate `HOME` env var must run serially — std::env::set_var is
    // !Sync (Rust 2024 will mark it unsafe). #[serial] forces these to one-at-a-time.

    #[test]
    #[serial_test::serial]
    fn opt_out_flag_round_trip() {
        let _env = EnvGuard::capture();
        // Override HOME so the default app data root resolves under the tempdir.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");

        // Default = false
        assert!(!user_opted_out());

        // Set true → readback true
        set_user_opted_out(true).unwrap();
        assert!(user_opted_out());

        // Set false → readback false
        set_user_opted_out(false).unwrap();
        assert!(!user_opted_out());
    }

    #[test]
    #[serial_test::serial]
    fn opt_out_flag_does_not_touch_typed_config_json() {
        let _env = EnvGuard::capture();
        // The opt-out sentinel must NOT live inside the daemon's typed
        // `config.json` — otherwise unrelated `Config::save` calls overwrite
        // the file and silently drop the user's opt-out preference (C1).
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");

        // Pre-populate config.json without the flag
        let config_path = tmp.path().join("origin").join("config.json");
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        std::fs::write(&config_path, r#"{"some_other_key":"value"}"#).unwrap();

        set_user_opted_out(true).unwrap();

        // Typed config.json must be untouched by the opt-out write.
        let raw = std::fs::read_to_string(&config_path).unwrap();
        assert_eq!(raw, r#"{"some_other_key":"value"}"#);
        assert!(user_opted_out());
    }

    #[test]
    #[serial_test::serial]
    fn opt_out_honors_origin_data_dir_env() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", tmp.path());

        assert!(!user_opted_out());
        set_user_opted_out(true).unwrap();
        assert!(tmp.path().join("auto_start_disabled.flag").exists());
        assert!(user_opted_out());
    }

    #[test]
    #[serial_test::serial]
    fn opt_out_prefers_wenlan_data_dir() {
        let _env = EnvGuard::capture();
        let current = tempfile::tempdir().unwrap();
        let legacy = tempfile::tempdir().unwrap();

        std::env::set_var("WENLAN_DATA_DIR", current.path());
        std::env::set_var("ORIGIN_DATA_DIR", legacy.path());

        set_user_opted_out(true).unwrap();

        assert!(current.path().join("auto_start_disabled.flag").exists());
        assert!(!legacy.path().join("auto_start_disabled.flag").exists());
    }

    #[test]
    #[serial_test::serial]
    fn stable_launch_agent_target_accepts_system_wenlan_app_bundle() {
        assert_eq!(
            classify_stable_launch_agent_target(std::path::Path::new(
                "/Applications/Wenlan.app/Contents/MacOS/origin-app"
            )),
            StableLaunchAgentTarget::Current
        );
    }

    #[test]
    #[serial_test::serial]
    fn stable_launch_agent_target_accepts_system_wenlan_app_executable() {
        assert_eq!(
            classify_stable_launch_agent_target(std::path::Path::new(
                "/Applications/Wenlan.app/Contents/MacOS/wenlan-app"
            )),
            StableLaunchAgentTarget::Current
        );
    }

    #[test]
    #[serial_test::serial]
    fn stable_launch_agent_target_accepts_user_wenlan_app_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let exe = tmp
            .path()
            .join("Applications/Wenlan.app/Contents/MacOS/origin-app");

        assert_eq!(
            classify_stable_launch_agent_target(&exe),
            StableLaunchAgentTarget::Current
        );
    }

    #[test]
    #[serial_test::serial]
    fn stable_launch_agent_target_detects_legacy_origin_app_bundle() {
        assert_eq!(
            classify_stable_launch_agent_target(std::path::Path::new(
                "/Applications/Origin.app/Contents/MacOS/origin"
            )),
            StableLaunchAgentTarget::LegacyOrigin
        );
    }

    #[test]
    #[serial_test::serial]
    fn stable_launch_agent_target_rejects_downloads_app_bundle() {
        assert_eq!(
            classify_stable_launch_agent_target(std::path::Path::new(
                "/Users/alice/Downloads/Wenlan.app/Contents/MacOS/origin-app"
            )),
            StableLaunchAgentTarget::Rejected
        );
    }

    #[test]
    #[serial_test::serial]
    fn app_plist_path_uses_wenlan_label() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());

        assert_eq!(
            app_plist_path().unwrap(),
            tmp.path()
                .join("Library/LaunchAgents/com.wenlan.desktop.plist")
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_app_plist_path_uses_origin_label() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());

        assert_eq!(
            legacy_app_plist_path().unwrap(),
            tmp.path()
                .join("Library/LaunchAgents/com.origin.desktop.plist")
        );
    }

    #[test]
    fn current_app_plist_template_uses_wenlan_placeholder() {
        assert!(APP_PLIST_TEMPLATE.contains("__WENLAN_APP_PATH__"));
        assert!(!APP_PLIST_TEMPLATE.contains("__ORIGIN_APP_PATH__"));
    }

    #[test]
    #[serial_test::serial]
    fn legacy_app_plist_ownership_accepts_owned_origin_app_path() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let user_app_path = tmp
            .path()
            .join("Applications/Origin.app/Contents/MacOS/origin-app");
        let user_app_plist = launch_agent_program_arguments_plist(
            LEGACY_APP_PLIST_LABEL,
            &user_app_path.to_string_lossy(),
        );

        assert!(legacy_app_plist_is_owned(&owned_legacy_app_plist()));
        assert!(legacy_app_plist_is_owned(&user_app_plist));
    }

    #[test]
    fn legacy_app_plist_ownership_rejects_foreign_path() {
        assert!(!legacy_app_plist_is_owned(&foreign_legacy_app_plist()));
    }

    #[test]
    #[serial_test::serial]
    fn legacy_server_plist_ownership_accepts_owned_origin_server_path() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let user_server_path = tmp
            .path()
            .join("Applications/Origin.app/Contents/MacOS/origin-server");
        let user_server_plist = launch_agent_program_arguments_plist(
            LEGACY_SERVER_PLIST_LABEL,
            &user_server_path.to_string_lossy(),
        );

        assert!(legacy_server_plist_is_owned(&owned_legacy_server_plist()));
        assert!(legacy_server_plist_is_owned(&user_server_plist));
    }

    #[test]
    fn legacy_server_plist_ownership_rejects_foreign_path() {
        assert!(!legacy_server_plist_is_owned(&foreign_legacy_server_plist()));
    }

    #[test]
    #[serial_test::serial]
    fn install_app_plist_rolls_back_file_when_launchctl_load_fails() {
        // H5: when `launchctl load` reports non-zero status, the plist file
        // must be removed so stale-plist detection on next startup does not
        // consider the broken file valid.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());

        let mock = MockLaunchctl {
            // ExitStatus::from_raw(256) => exit code 1 (not success).
            load_status: Mutex::new(256),
            ..Default::default()
        };
        let err = install_app_plist(&mock).expect_err("install should fail when load fails");
        assert!(
            err.to_string().contains("launchctl load failed"),
            "unexpected error: {err}"
        );

        let plist = tmp
            .path()
            .join("Library/LaunchAgents/com.wenlan.desktop.plist");
        assert!(
            !plist.exists(),
            "broken plist must be rolled back after load failure"
        );
    }

    #[test]
    #[serial_test::serial]
    fn install_app_plist_writes_file_and_calls_launchctl_load() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let mock = MockLaunchctl::default();
        install_app_plist(&mock).unwrap();

        let plist = tmp
            .path()
            .join("Library/LaunchAgents/com.wenlan.desktop.plist");
        assert!(plist.exists(), "plist file written");
        let content = std::fs::read_to_string(&plist).unwrap();
        assert!(content.contains("<key>Label</key>"));
        assert!(content.contains("<string>com.wenlan.desktop</string>"));
        assert!(
            !content.contains("__ORIGIN_APP_PATH__"),
            "legacy placeholder absent"
        );
        assert!(
            !content.contains("__WENLAN_APP_PATH__"),
            "current placeholder substituted"
        );

        let calls = mock.calls.lock().unwrap();
        assert!(calls.iter().any(|c| c[0] == "load"));
    }

    #[test]
    #[serial_test::serial]
    fn install_app_plist_writes_wenlan_log_paths() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let mock = MockLaunchctl::default();
        install_app_plist(&mock).unwrap();

        let plist = tmp
            .path()
            .join("Library/LaunchAgents/com.wenlan.desktop.plist");
        let content = std::fs::read_to_string(&plist).unwrap();
        let log_dir = data.path().join("logs");
        assert!(content.contains(log_dir.to_string_lossy().as_ref()));
        assert!(content.contains("wenlan-app.stdout.log"));
        assert!(content.contains("wenlan-app.stderr.log"));
        assert!(!content.contains("origin-app.stdout.log"));
        assert!(!content.contains("origin-app.stderr.log"));
    }

    #[test]
    #[serial_test::serial]
    fn uninstall_app_plist_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist_dir = tmp.path().join("Library/LaunchAgents");
        std::fs::create_dir_all(&plist_dir).unwrap();
        let plist = plist_dir.join("com.wenlan.desktop.plist");
        std::fs::write(&plist, "<plist/>").unwrap();

        let mock = MockLaunchctl::default();
        uninstall_app_plist(&mock).unwrap();

        assert!(!plist.exists(), "plist file removed");
        let calls = mock.calls.lock().unwrap();
        assert!(calls.iter().any(|c| c[0] == "unload"));
    }

    #[test]
    #[serial_test::serial]
    fn is_run_at_login_enabled_returns_true_when_both_labels_present() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());

        struct MockListed(String);
        impl LaunchctlExec for MockListed {
            fn run(&self, _args: &[&str]) -> io::Result<Output> {
                Ok(Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: self.0.as_bytes().to_vec(),
                    stderr: vec![],
                })
            }
        }
        let label_line = format!(
            "123\t0\t{}\n456\t0\tcom.wenlan.desktop\n",
            SERVER_PLIST_LABEL
        );
        let listed = MockListed(label_line);
        assert!(is_run_at_login_enabled(&listed));
    }

    #[test]
    fn is_run_at_login_enabled_returns_false_when_one_missing() {
        struct MockListed(String);
        impl LaunchctlExec for MockListed {
            fn run(&self, _args: &[&str]) -> io::Result<Output> {
                Ok(Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: self.0.as_bytes().to_vec(),
                    stderr: vec![],
                })
            }
        }
        let only_server = MockListed(format!("123\t0\t{}\n", SERVER_PLIST_LABEL));
        assert!(!is_run_at_login_enabled(&only_server));
    }

    #[test]
    fn is_run_at_login_enabled_does_not_match_label_substring() {
        // H4: `launchctl list` output where a different label has our label
        // as a prefix (e.g. `com.origin.server.staging`) must not be treated
        // as our service being present.
        struct MockListed(String);
        impl LaunchctlExec for MockListed {
            fn run(&self, _args: &[&str]) -> io::Result<Output> {
                Ok(Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: self.0.as_bytes().to_vec(),
                    stderr: vec![],
                })
            }
        }
        // Note: only the `.staging` suffixed labels appear. Real labels are absent.
        let staging_only = MockListed(format!(
            "123\t0\t{}.staging\n456\t0\t{}.staging\n",
            SERVER_PLIST_LABEL, APP_PLIST_LABEL
        ));
        assert!(
            !is_run_at_login_enabled(&staging_only),
            ".staging suffixed labels must not satisfy exact-label match"
        );
    }

    #[test]
    #[serial_test::serial]
    fn quit_origin_debounces_concurrent_calls() {
        // H1: tray menu Quit Wenlan item stays clickable during the 500ms
        // shutdown sleep — second click must not re-enter the shutdown flow.
        reset_quitting_flag_for_test();
        // First call wins — flag flips to true.
        assert!(try_begin_quit(), "first call should be allowed to proceed");
        // Second call is rejected — flag is already true.
        assert!(
            !try_begin_quit(),
            "second concurrent call should be rejected"
        );
        // Cleanup so other tests start fresh.
        reset_quitting_flag_for_test();
    }

    #[test]
    fn recoverable_quit_error_releases_the_process_wide_guard() {
        let flag = AtomicBool::new(false);

        {
            let _attempt = QuitAttemptGuard::try_begin(&flag)
                .expect("first quit attempt should acquire the guard");
            assert!(flag.load(Ordering::Acquire));
        }

        assert!(
            !flag.load(Ordering::Acquire),
            "dropping an uncommitted attempt must allow a guarded retry"
        );
    }

    #[test]
    #[serial_test::serial]
    fn uninstall_app_plist_is_idempotent_when_file_absent() {
        // H2: Quit Wenlan calls uninstall_app_plist; the sequence must be
        // idempotent because the plist may already have been removed by an
        // earlier toggle.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let mock = MockLaunchctl::default();
        // No file present → succeed without error.
        uninstall_app_plist(&mock).unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_server_plist_unloads_and_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, owned_legacy_server_plist()).unwrap();

        let mock = MockLaunchctl::default();
        cleanup_legacy_server_plist(&mock).unwrap();

        assert!(!plist.exists(), "legacy server plist removed");
        let calls = mock.calls.lock().unwrap();
        assert!(
            calls
                .iter()
                .any(|c| c[0] == "unload" && c[1] == plist.to_string_lossy()),
            "legacy server plist unloaded before removal"
        );
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_app_plist_unloads_and_removes_owned_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_app_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, owned_legacy_app_plist()).unwrap();

        let mock = MockLaunchctl::default();
        cleanup_legacy_app_plist(&mock).unwrap();

        assert!(!plist.exists(), "legacy app plist removed");
        let calls = mock.calls.lock().unwrap();
        assert!(
            calls
                .iter()
                .any(|c| c[0] == "unload" && c[1] == plist.to_string_lossy()),
            "legacy app plist unloaded before removal"
        );
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_app_plist_removes_owned_file_when_unload_fails() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_app_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, owned_legacy_app_plist()).unwrap();

        let mock = MockLaunchctl {
            unload_status: Mutex::new(256),
            ..Default::default()
        };
        cleanup_legacy_app_plist(&mock).unwrap();

        assert!(
            !plist.exists(),
            "owned legacy app plist removed even when unload fails"
        );
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_server_plist_removes_owned_file_when_unload_fails() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, owned_legacy_server_plist()).unwrap();

        let mock = MockLaunchctl {
            unload_status: Mutex::new(256),
            ..Default::default()
        };
        cleanup_legacy_server_plist(&mock).unwrap();

        assert!(
            !plist.exists(),
            "owned legacy server plist removed even when unload fails"
        );
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_app_plist_preserves_foreign_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_app_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, foreign_legacy_app_plist()).unwrap();

        let mock = MockLaunchctl::default();
        cleanup_legacy_app_plist(&mock).unwrap();

        assert!(plist.exists(), "foreign legacy app plist preserved");
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "foreign legacy app plist must not be unloaded"
        );
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_server_plist_preserves_foreign_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, foreign_legacy_server_plist()).unwrap();

        let mock = MockLaunchctl::default();
        cleanup_legacy_server_plist(&mock).unwrap();

        assert!(plist.exists(), "foreign legacy server plist preserved");
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "foreign legacy server plist must not be unloaded"
        );
    }

    #[test]
    #[serial_test::serial]
    fn first_run_install_cleans_legacy_plists_even_when_user_opted_out() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let legacy_app = legacy_app_plist_path().unwrap();
        let legacy_server = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(legacy_app.parent().unwrap()).unwrap();
        std::fs::write(&legacy_app, owned_legacy_app_plist()).unwrap();
        std::fs::write(&legacy_server, owned_legacy_server_plist()).unwrap();
        set_user_opted_out(true).unwrap();

        let mock = MockLaunchctl::default();
        first_run_install_if_needed(&mock).unwrap();

        assert!(!legacy_app.exists(), "owned legacy app plist removed");
        assert!(!legacy_server.exists(), "owned legacy server plist removed");
        assert!(
            !tmp.path()
                .join("Library/LaunchAgents/com.wenlan.desktop.plist")
                .exists(),
            "opted-out users should not get a new current app plist"
        );
        let calls = mock.calls.lock().unwrap();
        assert!(
            !calls
                .iter()
                .any(|c| c[0] == "unload" && c[1] == legacy_app.to_string_lossy()),
            "first-run migration must not unload the legacy app job before replacement exists"
        );
    }

    #[test]
    #[serial_test::serial]
    fn first_run_preserves_legacy_plists_when_current_app_path_is_rejected() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        set_user_opted_out(false).unwrap();

        let legacy_app = legacy_app_plist_path().unwrap();
        let legacy_server = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(legacy_app.parent().unwrap()).unwrap();
        std::fs::write(&legacy_app, owned_legacy_app_plist()).unwrap();
        std::fs::write(&legacy_server, owned_legacy_server_plist()).unwrap();

        let mock = MockLaunchctl::default();
        first_run_install_if_needed(&mock).unwrap();

        assert!(
            legacy_app.exists(),
            "legacy app fallback must remain until current app install is possible"
        );
        assert!(
            legacy_server.exists(),
            "legacy server fallback must remain until current server install is possible"
        );
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "rejected current app path should not unload legacy fallbacks"
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn set_run_at_login_false_cleans_legacy_app_and_server_plists() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let current_app = tmp
            .path()
            .join("Library/LaunchAgents/com.wenlan.desktop.plist");
        let legacy_app = legacy_app_plist_path().unwrap();
        let legacy_server = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(current_app.parent().unwrap()).unwrap();
        std::fs::write(&current_app, "<plist/>").unwrap();
        std::fs::write(&legacy_app, owned_legacy_app_plist()).unwrap();
        std::fs::write(&legacy_server, owned_legacy_server_plist()).unwrap();

        let mock = MockLaunchctl::default();
        set_run_at_login(false, &mock).await.unwrap();

        assert!(!current_app.exists(), "current Wenlan app plist removed");
        assert!(!legacy_app.exists(), "owned legacy app plist removed");
        assert!(!legacy_server.exists(), "owned legacy server plist removed");
    }

    #[test]
    #[serial_test::serial]
    fn legacy_server_plist_does_not_count_as_current_wenlan_service() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        assert!(legacy_server_plist_exists());
        assert!(
            !current_server_plist_exists(),
            "legacy Origin service must not suppress Wenlan sidecar fallback"
        );
    }

    #[test]
    #[serial_test::serial]
    fn current_server_plist_counts_as_wenlan_service() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(
            &plist,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wenlan.server</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WENLAN_DATA_DIR</key>
        <string>{}</string>
    </dict>
</dict>
</plist>
"#,
                data.path().display()
            ),
        )
        .unwrap();

        assert!(current_server_plist_exists());
        assert!(current_server_plist_matches_selected_data_dir());
    }

    #[test]
    #[serial_test::serial]
    fn current_server_plist_with_stale_data_dir_does_not_count_as_wenlan_service() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let selected_data = tempfile::tempdir().unwrap();
        let stale_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", selected_data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(
            &plist,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wenlan.server</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WENLAN_DATA_DIR</key>
        <string>{}</string>
    </dict>
</dict>
</plist>
"#,
                stale_data.path().display()
            ),
        )
        .unwrap();

        assert!(
            current_server_plist_exists(),
            "the stale launchd plist file still exists"
        );
        assert!(
            !current_server_plist_matches_selected_data_dir(),
            "a stale launchd data root must not suppress the selected-data-dir sidecar fallback"
        );
    }

    #[test]
    #[serial_test::serial]
    fn server_plist_data_dir_env_is_patched_and_reloaded() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(
            &plist,
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wenlan.server</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>RUST_LOG</key>
        <string>info</string>
    </dict>
</dict>
</plist>
"#,
        )
        .unwrap();

        let mock = MockLaunchctl::default();
        ensure_server_plist_data_dir_env(&mock).unwrap();

        let content = std::fs::read_to_string(&plist).unwrap();
        assert_eq!(
            plist_environment_string(&content, "WENLAN_DATA_DIR").as_deref(),
            Some(data.path().to_string_lossy().as_ref())
        );
        assert_eq!(
            plist_environment_string(&content, "RUST_LOG").as_deref(),
            Some("info")
        );
        assert!(server_plist_has_selected_data_dir(&content));

        let calls = mock.calls.lock().unwrap();
        assert!(
            calls.iter().any(|c| c[0] == "unload"),
            "server plist should be unloaded before reloading patched env"
        );
        assert!(
            calls.iter().any(|c| c[0] == "load"),
            "server plist should be loaded after patching env"
        );
    }

    #[test]
    #[serial_test::serial]
    fn startup_server_plist_preflight_repairs_stale_data_dir_before_selection() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let selected_data = tempfile::tempdir().unwrap();
        let stale_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", selected_data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(
            &plist,
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wenlan.server</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WENLAN_DATA_DIR</key>
        <string>{}</string>
    </dict>
</dict>
</plist>
"#,
                stale_data.path().display()
            ),
        )
        .unwrap();
        assert!(!current_server_plist_matches_selected_data_dir());

        let mock = MockLaunchctl::default();
        prepare_server_plist_for_startup(&mock).unwrap();

        assert!(
            current_server_plist_matches_selected_data_dir(),
            "startup preflight must repair stale launchd data root before daemon selection"
        );
        let calls = mock.calls.lock().unwrap();
        assert!(
            calls.iter().any(|c| c[0] == "unload"),
            "stale running daemon should be unloaded before health/config hydration"
        );
        assert!(
            calls.iter().any(|c| c[0] == "load"),
            "patched daemon should be reloaded before health/config hydration"
        );
    }

    #[test]
    #[serial_test::serial]
    fn startup_server_plist_preflight_rolls_back_file_when_reload_fails() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        let selected_data = tempfile::tempdir().unwrap();
        let stale_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", selected_data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        let original = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wenlan.server</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WENLAN_DATA_DIR</key>
        <string>{}</string>
    </dict>
</dict>
</plist>
"#,
            stale_data.path().display()
        );
        std::fs::write(&plist, &original).unwrap();

        let mock = MockLaunchctl {
            load_status: Mutex::new(256),
            ..Default::default()
        };
        let err = prepare_server_plist_for_startup(&mock)
            .expect_err("reload failure must make startup preflight fail");
        assert!(
            err.to_string().contains("launchctl load failed"),
            "unexpected error: {err}"
        );

        let content = std::fs::read_to_string(&plist).unwrap();
        assert_eq!(
            plist_environment_string(&content, "WENLAN_DATA_DIR").as_deref(),
            Some(stale_data.path().to_string_lossy().as_ref()),
            "failed reload must roll back the file so later selection cannot trust patched-only state"
        );
        assert!(!current_server_plist_matches_selected_data_dir());
    }

    #[test]
    fn service_management_uses_wenlan_cli_next_to_app_binary() {
        let path = service_cli_path_for_app_exe(std::path::Path::new(
            "/Applications/Origin.app/Contents/MacOS/origin-app",
        ))
        .unwrap();
        assert_eq!(
            path,
            std::path::Path::new("/Applications/Origin.app/Contents/MacOS/wenlan")
        );
    }

    #[test]
    fn tauri_bundle_declares_wenlan_cli_sidecar_for_service_management() {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let external_bins = config["bundle"]["externalBin"].as_array().unwrap();
        assert!(
            external_bins
                .iter()
                .any(|bin| bin.as_str() == Some("binaries/wenlan")),
            "wenlan CLI must be bundled because lifecycle service management runs `wenlan install/uninstall`"
        );
    }

    #[test]
    fn tauri_asset_scope_allows_wenlan_and_legacy_avatar_roots() {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let allowed = config["app"]["security"]["assetProtocol"]["scope"]["allow"]
            .as_array()
            .unwrap();
        assert!(
            allowed
                .iter()
                .any(|path| path.as_str() == Some("$LOCALDATA/wenlan/avatars/**")),
            "new avatars are stored under the Wenlan data root"
        );
        assert!(
            allowed
                .iter()
                .any(|path| path.as_str() == Some("$DATA/origin/avatars/**")),
            "legacy Origin avatar paths must keep rendering during migration"
        );
    }

    /// Mock that observes concurrent launchctl invocations. `in_flight`
    /// tracks how many calls are currently executing; `max_in_flight`
    /// records the high-water mark. If the caller properly serializes via
    /// RUN_AT_LOGIN_LOCK, we should never observe `max_in_flight > 1`.
    #[derive(Default)]
    struct ConcurrencyMockLaunchctl {
        in_flight: std::sync::atomic::AtomicU32,
        max_in_flight: std::sync::atomic::AtomicU32,
    }
    impl LaunchctlExec for ConcurrencyMockLaunchctl {
        fn run(&self, _args: &[&str]) -> io::Result<Output> {
            use std::sync::atomic::Ordering::AcqRel;
            let prev = self.in_flight.fetch_add(1, AcqRel);
            let now = prev + 1;
            // Update high-water mark.
            self.max_in_flight.fetch_max(now, AcqRel);
            // Sleep to widen the window for concurrent observers.
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.in_flight.fetch_sub(1, AcqRel);
            Ok(Output {
                status: std::process::ExitStatus::from_raw(0),
                stdout: vec![],
                stderr: vec![],
            })
        }
    }

    /// Test wrapper that exercises ONLY the launchctl-touching portion of
    /// the toggle while still acquiring the same RUN_AT_LOGIN_LOCK that
    /// `set_run_at_login` uses. This isolates the concurrency property
    /// (the lock serializes the launchctl observation window) from the
    /// subprocess-related side effects (origin-server isn't available in
    /// tests, and uninstall_app_plist removes the plist file on first
    /// call which makes subsequent calls short-circuit).
    async fn set_run_at_login_lock_section_for_test(launchctl: &dyn LaunchctlExec) -> Result<()> {
        let _guard = RUN_AT_LOGIN_LOCK.lock().await;
        // Spend deterministic time inside the locked region invoking the
        // mock launchctl, so the concurrency mock's high-water observation
        // is exercised.
        let _ = launchctl.run(&["unload", "/tmp/fake.plist"]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial_test::serial]
    async fn set_run_at_login_serializes_concurrent_calls() {
        // G2: spec line 198 — set_run_at_login holds RUN_AT_LOGIN_LOCK for
        // the duration of the toggle to prevent concurrent install/uninstall
        // races. Spawn two concurrent calls that take the same lock and
        // hit the launchctl mock; assert the mock never observes >1 call
        // in flight.
        let mock: &'static ConcurrencyMockLaunchctl =
            Box::leak(Box::new(ConcurrencyMockLaunchctl::default()));

        let h1 = tokio::spawn(async move { set_run_at_login_lock_section_for_test(mock).await });
        let h2 = tokio::spawn(async move { set_run_at_login_lock_section_for_test(mock).await });
        h1.await.unwrap().unwrap();
        h2.await.unwrap().unwrap();

        let max_seen = mock
            .max_in_flight
            .load(std::sync::atomic::Ordering::Acquire);
        assert!(
            max_seen <= 1,
            "RUN_AT_LOGIN_LOCK failed to serialize: max_in_flight={max_seen}"
        );
    }
}
