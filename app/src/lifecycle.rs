// SPDX-License-Identifier: AGPL-3.0-only
// Items in this module are used by later tasks (Tasks 6-16). Allow dead-code
// until they are wired up.
#![allow(dead_code)]
use crate::daemon_start::SidecarStopOutcome;
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

/// The newer app bundle to reopen once this quit has finished, set by the
/// single-instance handover (see `handover.rs`). Only ever read on macOS.
#[cfg(target_os = "macos")]
static HANDOVER_BUNDLE: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Remember the newer bundle that the running quit must hand over to.
#[cfg(target_os = "macos")]
pub fn set_handover_bundle(bundle: PathBuf) {
    *HANDOVER_BUNDLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(bundle);
}

/// Drop a pending handover: the frontend refused the quit (it could not
/// save), so the user keeps this app and a later, unrelated quit must not
/// reopen the newer bundle.
#[cfg(target_os = "macos")]
pub fn clear_handover_bundle() {
    take_handover_bundle();
}

#[cfg(target_os = "macos")]
fn take_handover_bundle() -> Option<PathBuf> {
    HANDOVER_BUNDLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

/// Finish a quit: schedule the handover reopen when one is pending, then exit
/// with `code`. A failed teardown exits non-zero but still hands over, so the
/// newer bundle opens either way.
pub(crate) fn exit_after_quit(app_handle: &AppHandle, code: i32) {
    #[cfg(target_os = "macos")]
    if let Some(bundle) = take_handover_bundle() {
        crate::handover::relaunch_after_exit(&bundle);
    }
    app_handle.exit(code);
}

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
pub(crate) const RUN_AT_LOGIN_UNSUPPORTED: &str = "Run at Login is not supported on this platform";
pub(crate) const FULL_QUIT_BREADCRUMB: &str = "[quit] full quit command accepted";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuitPlan {
    clean_launch_agents: bool,
    shutdown_daemon: bool,
    exit_app: bool,
}

pub(crate) fn run_at_login_capability(target_os: &str) -> Result<(), &'static str> {
    if target_os == "macos" {
        Ok(())
    } else {
        Err(RUN_AT_LOGIN_UNSUPPORTED)
    }
}

fn quit_plan_for_target_os(target_os: &str) -> QuitPlan {
    QuitPlan {
        clean_launch_agents: target_os == "macos",
        shutdown_daemon: true,
        exit_app: true,
    }
}

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
    #[cfg(test)]
    {
        // Every plist path below hangs off this, and `install_app_plist`
        // *writes* one. `dirs::home_dir()` reads `FOLDERID_Profile` on Windows
        // and never looks at `HOME`, so an unset `HOME` here is a test about
        // to install a LaunchAgent into the developer's real profile — the
        // same failure that `identity_paths::refuse_real_profile` catches for
        // the data roots. `~/Library/LaunchAgents` on the Windows dev host is
        // the residue of exactly this.
        match std::env::var_os("HOME") {
            Some(home) => Ok(PathBuf::from(home)),
            None => panic!(
                "lifecycle::home_dir() reached the developer's real profile from a unit test. \
                 Set HOME to a tempdir (and keep the guard alive for the whole test) before \
                 touching a plist path."
            ),
        }
    }
    #[cfg(not(test))]
    {
        dirs::home_dir().context("HOME not set")
    }
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

/// What the server LaunchAgent file says about the selected data root. A file
/// that is not there is a measured negative; a file that is there and cannot
/// be *read* is not — a permission error says nothing about its contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerPlistMatch {
    Matches,
    DoesNotMatch,
    Unknown,
}

fn server_plist_data_dir_match() -> ServerPlistMatch {
    let Ok(path) = server_plist_path() else {
        log::warn!("[lifecycle] no LaunchAgents path for this user; server plist unreadable");
        return ServerPlistMatch::Unknown;
    };
    match std::fs::read_to_string(&path) {
        Ok(content) if server_plist_has_selected_data_dir(&content) => ServerPlistMatch::Matches,
        Ok(_) => ServerPlistMatch::DoesNotMatch,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ServerPlistMatch::DoesNotMatch,
        Err(e) => {
            log::warn!(
                "[lifecycle] could not read {} to see whether launchd targets the selected data \
                 root: {e}",
                path.display()
            );
            ServerPlistMatch::Unknown
        }
    }
}

/// Whether launchd owns the daemon, as far as it could be measured.
///
/// Tri-state on purpose: a log line inside a probe does not preserve
/// tri-state, so the `Unknown` arm has to leave the function. Callers that
/// genuinely only want the positive case write `== ServerPlistMatch::Matches`
/// at their own site, where the discarded third value is visible in the diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchdOwnership {
    /// Measured: the server LaunchAgent targets the selected data root *and*
    /// launchd has the job loaded.
    Owns,
    /// Measured: it does not. Either no plist for this data root, or a plist
    /// with no loaded job behind it.
    DoesNot,
    /// Could not be measured: launchctl would not run, would not answer, or
    /// did not print a table; or the plist exists and could not be read.
    Unknown,
}

/// Whether launchd owns the daemon: the server LaunchAgent targets the
/// selected data root *and* launchd has the job loaded. The file alone is not
/// ownership — `wenlan background on` writes it before `launchctl load`, and a
/// failed load leaves it behind with no job to run the daemon.
///
/// This function no longer decides what an unmeasurable launchctl *means*.
/// Falling back to the app's own sidecar is still the right move (assuming
/// launchd owns a daemon it may have no job for leaves the user with nothing
/// serving), but that is a caller's decision, taken beside a port-health
/// measurement the caller makes, and recorded where it can be seen —
/// `daemon_start::spawned_on_unknown_owner` and the diagnostics wire.
pub fn launchd_owns_server_daemon(launchctl: &dyn LaunchctlExec) -> LaunchdOwnership {
    match server_plist_data_dir_match() {
        ServerPlistMatch::DoesNotMatch => return LaunchdOwnership::DoesNot,
        ServerPlistMatch::Unknown => return LaunchdOwnership::Unknown,
        ServerPlistMatch::Matches => {}
    }
    // `LaunchctlReading::label_state`, never a bare table lookup: a table cut
    // at a row boundary before `com.wenlan.server` would land here as
    // `NotLoaded` -> `DoesNot`, sending `daemon_start` down the CLEAN spawn
    // branch against a job that is in fact loaded.
    match LaunchctlReading::take(launchctl).label_state(SERVER_PLIST_LABEL) {
        LabelState::Loaded => LaunchdOwnership::Owns,
        LabelState::NotLoaded => LaunchdOwnership::DoesNot,
        LabelState::Unknown => LaunchdOwnership::Unknown,
    }
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

/// True when the process was launched with an explicit data-dir override
/// (WENLAN_DATA_DIR / ORIGIN_DATA_DIR) — an isolated dev/smoke run. Such a
/// run must never mutate the user's LaunchAgents or the shared daemon.
pub fn data_dir_env_overridden() -> bool {
    std::env::var_os("WENLAN_DATA_DIR").is_some() || std::env::var_os("ORIGIN_DATA_DIR").is_some()
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
    let in_user_apps = home_dir()
        .ok()
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
        || home_dir()
            .ok()
            .map(|home| {
                path == home.join("Applications/Origin.app/Contents/MacOS/origin")
                    || path == home.join("Applications/Origin.app/Contents/MacOS/origin-app")
            })
            .unwrap_or(false)
}

fn path_is_legacy_origin_server_exe(path: &str) -> bool {
    let path = Path::new(path);
    path == Path::new("/Applications/Origin.app/Contents/MacOS/origin-server")
        || home_dir()
            .ok()
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

/// Argv the app hands the bundled `wenlan` CLI to register and start the
/// daemon LaunchAgent. Must stay in step with `BackgroundCommand::On` in
/// `crates/wenlan-cli/src/main.rs`, which maps to `service::install()`.
///
/// Limit worth knowing: this cannot be type-checked against the CLI's clap
/// definition from here — `Cli`/`Commands` live in the CLI's `main.rs`, not
/// its library target, and the app crate does not depend on wenlan-cli. The
/// other half of the guard is `crates/wenlan-cli/tests/cli_integration.rs`,
/// which exercises `["background", "on"]` end to end and asserts the removed
/// `install`/`uninstall` verbs are gone.
const SERVICE_CLI_BACKGROUND_ON: [&str; 2] = ["background", "on"];

fn run_service_cli(args: &[&str]) -> Result<()> {
    let bin = service_cli_path()?;
    let (data_dir_env, data_dir) = crate::identity_paths::sidecar_data_dir_env();
    let out = Command::new(&bin)
        .env(data_dir_env, &data_dir)
        .args(args)
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "wenlan {} failed: {}",
            args.join(" "),
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
    // Isolated launches and non-stable app paths leave the shared LaunchAgent
    // untouched. In the isolated case, lib.rs then sees that the plist does
    // not match the selected scratch data dir and starts the sidecar, whose
    // child inherits the app process environment (including WENLAN_PORT).
    if data_dir_env_overridden() {
        log::warn!(
            "[lifecycle] skipping server plist startup preflight: isolated run (data-dir env override)"
        );
        return Ok(());
    }

    let app_path = match current_app_path() {
        Ok(path) => path,
        Err(error) => {
            log::info!(
                "[lifecycle] skipping server plist startup preflight: non-stable app path ({error})"
            );
            return Ok(());
        }
    };
    if !is_stable_launch_agent_target(&app_path) {
        log::info!(
            "[lifecycle] skipping server plist startup preflight: non-stable app path ({})",
            app_path.display()
        );
        return Ok(());
    }

    ensure_server_plist_data_dir_env(launchctl)
}

/// Run `wenlan background on`. Resolves the CLI binary alongside our exe; the
/// CLI owns service-manager integration and expects `wenlan-server` next to it.
pub fn install_server_plist_via_subprocess(launchctl: &dyn LaunchctlExec) -> Result<()> {
    run_service_cli(&SERVICE_CLI_BACKGROUND_ON)?;
    ensure_server_plist_data_dir_env(launchctl)
}

/// Unload `plist`; if launchctl refuses, succeed only when `label` is
/// *measured* to be no longer loaded, otherwise return an error so the caller
/// keeps the plist file.
///
/// Only `NotLoaded` may pass. When the unload failed and the follow-up
/// `launchctl list` also failed, nothing was learned about the job, and the
/// caller deleting the plist on that non-answer is precisely the shipped bug:
/// a still-loaded `KeepAlive` daemon left with no registration file to unload
/// it by, and Run at Login reading "off" while it is on.
///
/// The `NotLoaded` this acts on can only come from
/// [`LaunchctlReading::label_state`]: absence is produced by exactly one
/// thing, a targeted probe with a working control.
fn unload_plist_or_verify_absent(
    launchctl: &dyn LaunchctlExec,
    plist: &Path,
    label: &str,
) -> Result<()> {
    let plist_arg = plist.to_string_lossy().to_string();
    let failure = match launchctl.run(&["unload", &plist_arg]) {
        Ok(out) if out.status.success() => return Ok(()),
        Ok(out) => String::from_utf8_lossy(&out.stderr).to_string(),
        Err(e) => e.to_string(),
    };
    match LaunchctlReading::take(launchctl).label_state(label) {
        LabelState::NotLoaded => {
            log::warn!(
                "[lifecycle] launchctl unload failed for {label} but it is not loaded (stale plist?): {failure}"
            );
            Ok(())
        }
        LabelState::Loaded => {
            anyhow::bail!("launchctl unload failed for {label}: {failure}")
        }
        LabelState::Unknown => anyhow::bail!(
            "launchctl unload failed for {label} and launchctl could not say whether it is still \
             loaded, so its plist is being kept: {failure}"
        ),
    }
}

/// Deregister the server LaunchAgent: unload it, then delete its plist.
///
/// Done in-process rather than through the CLI, because the CLI has no
/// deregister verb. `wenlan background off` is a reversible runtime stop that
/// deliberately keeps the launchd registration, so it would leave the plist's
/// `RunAtLoad true` respawn in place — breaking both the Quit invariant below
/// ("both plists unloaded … no auto-restart on reboot") and the Run-at-Login
/// toggle, whose state `is_run_at_login_enabled` reads back out of launchctl.
/// Unlike `cleanup_legacy_server_plist`, this is strict on purpose: the
/// server LaunchAgent has `KeepAlive`/`RunAtLoad`, so deleting the plist
/// while the job is still loaded would leave it running with no
/// registration left to unload later.
pub fn uninstall_server_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = server_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    unload_plist_or_verify_absent(launchctl, &plist, SERVER_PLIST_LABEL)?;
    std::fs::remove_file(&plist)?;
    Ok(())
}

/// What launchctl could tell us about a label. `Unknown` is a *failed
/// measurement*, not an absence: a launchctl that would not run, or that
/// answered nonzero, knows nothing about the job either way. Folding it into
/// `NotLoaded` is what let `uninstall_server_plist` delete the plist of a
/// still-loaded `KeepAlive` job, leaving a daemon running with no
/// registration left to unload and a Settings toggle reading "off".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelState {
    Loaded,
    NotLoaded,
    Unknown,
}

/// The header `launchctl list` prints above its table. Compared field-wise so
/// the tabs the real output uses and the spaces a fixture might use are the
/// same header.
const LAUNCHCTL_LIST_HEADER: [&str; 3] = ["PID", "Status", "Label"];

/// Fewest data rows below which this code stops believing something that
/// answered `launchctl list` was launchd at all.
///
/// READ THIS AS PROVENANCE, NOT COMPLETENESS (round 4, defect A). launchd's
/// user domain is never empty on a machine a person is logged into: the
/// session is *started* by launchd and carries its own `com.apple.*` agents
/// before Wenlan is installed at all. Ten rows plus a `com.apple.*` job
/// therefore distinguishes "launchd answered" from "a stub on `PATH`
/// answered". It says NOTHING about whether the answer was the whole answer: a
/// table cut after row eleven and before our label clears both checks and is
/// indistinguishable, from the outside, from a domain that genuinely has no
/// Wenlan job. That is why no plist is deleted on this table's word alone —
/// see [`label_absent_enough_to_delete_plist`].
///
/// The floor's own failure mode is `Unknown`, which is the safe side: a
/// legitimately sparse domain is called unmeasurable rather than empty.
const LAUNCHCTL_MIN_ROWS: usize = 10;

/// Prefix of the jobs whose presence witnesses a real launchd user domain.
/// The row-count floor alone can be cleared by any ten well-formed lines; a
/// `com.apple.*` job cannot be, and every macOS login session has them. Mirrors
/// the `WINPID 4` (System process) witness in `ps_w_row_for`. Provenance
/// evidence, like the floor above — not evidence of completeness.
const LAUNCHD_SYSTEM_JOB_PREFIX: &str = "com.apple.";

/// The labels in a `launchctl list` table, or `None` when the text is not one.
///
/// WHAT THESE CHECKS ESTABLISH, stated exactly:
///
/// * header / row shape / row floor / `com.apple.*` job — PROVENANCE. Something
///   that speaks launchd's table dialect answered. A stub, an error message, or
///   an empty pipe does not clear them.
/// * final newline — the one COMPLETENESS check available here, and only a
///   partial one. `launchctl` terminates every row it writes; stdout that ends
///   mid-line was cut in transit, so it is `None`. A cut that lands exactly on
///   a row boundary is still invisible, and no arrangement of these checks can
///   see it.
///
/// So: a `Some(..)` here is "launchd answered with a table whose last line was
/// whole", NOT "this is every job launchd has". Presence of a label in the
/// result is sound (a truncated table cannot invent a row); ABSENCE from it is
/// not, and every caller that acts destructively on absence must get a second,
/// non-truncatable witness first.
///
/// A row is exactly three whitespace-separated fields — `PID` (a number or `-`
/// for a job that is not currently running), `Status` (a signed exit code),
/// and `Label`. THE RESIDUAL, stated because it is not fixed: a label
/// containing whitespace would make its own row unparseable and the whole
/// table `Unknown`; launchd labels are reverse-DNS and none observed carries a
/// space, and `Unknown` is the safe side of that guess in every caller here.
fn launchctl_list_labels(stdout: &str) -> Option<Vec<&str>> {
    // A table whose last line was cut MID-ROW is a short read the row-shape
    // checks cannot see. `launchctl` ends every row with a newline, so stdout
    // that does not end with one is missing an unknown amount of table.
    //
    // WHAT THIS DOES NOT CATCH, and the reason no caller may read a missing
    // label out of this vector as an absence: a cut at a ROW BOUNDARY. Ten
    // whole rows and a trailing newline is a valid table by every check here
    // and by every check that could be written here, because a short complete
    // table and a complete short table are the same bytes. See
    // `BulkLabelReading`.
    if !stdout.is_empty() && !stdout.ends_with('\n') {
        return None;
    }
    let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<&str> = lines.next()?.split_whitespace().collect();
    if header != LAUNCHCTL_LIST_HEADER {
        return None;
    }
    let mut labels = Vec::new();
    let mut saw_system_job = false;
    for line in lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [pid, status, label] = fields[..] else {
            return None;
        };
        if pid != "-" && pid.parse::<i64>().is_err() {
            return None;
        }
        if status.parse::<i64>().is_err() {
            return None;
        }
        saw_system_job |= label.starts_with(LAUNCHD_SYSTEM_JOB_PREFIX);
        labels.push(label);
    }
    if labels.len() < LAUNCHCTL_MIN_ROWS || !saw_system_job {
        return None;
    }
    Some(labels)
}

/// One `launchctl list` read, parsed once.
///
/// Exists so a caller that has two questions asks launchd ONCE (defect B).
/// Two `launchctl list` calls are two different instants, and launchd can load
/// or unload a job between them — which let `run_at_login_state` report a pair
/// state that was never true at any single moment.
enum LaunchctlSnapshot {
    /// launchd answered with a table whose last line was whole. Owned rather
    /// than borrowed so the stdout buffer can be dropped.
    Table(Vec<String>),
    /// launchctl would not run, exited nonzero, or did not print a table.
    /// A failed measurement, not an empty domain.
    Unreadable,
}

/// Take one bare `launchctl list` reading.
///
/// Deliberately the *bare* form for this snapshot: it answers about every
/// label at once, so N questions cost one instant instead of N. Its weakness --
/// a table truncated at a row boundary is a silent short answer -- is handled
/// by [`BulkLabelReading`] refusing to call a missing row an absence, for every
/// caller rather than for the one that deletes files.
fn launchctl_snapshot(launchctl: &dyn LaunchctlExec) -> LaunchctlSnapshot {
    let out = match launchctl.run(&["list"]) {
        Ok(out) => out,
        Err(e) => {
            log::warn!("[lifecycle] could not run launchctl list: {e}");
            return LaunchctlSnapshot::Unreadable;
        }
    };
    if !out.status.success() {
        log::warn!(
            "[lifecycle] launchctl list exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        return LaunchctlSnapshot::Unreadable;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    match launchctl_list_labels(&stdout) {
        Some(labels) => LaunchctlSnapshot::Table(labels.into_iter().map(str::to_string).collect()),
        None => {
            log::warn!(
                "[lifecycle] launchctl list exited 0 but did not print a whole `{}` table with at \
                 least {LAUNCHCTL_MIN_ROWS} rows and a {LAUNCHD_SYSTEM_JOB_PREFIX}* job; treating \
                 it as could-not-measure ({} bytes of stdout)",
                LAUNCHCTL_LIST_HEADER.join("/"),
                out.stdout.len()
            );
            LaunchctlSnapshot::Unreadable
        }
    }
}

/// What a BULK `launchctl list` table establishes about ONE label.
///
/// A label missing from the table is NOT a measured negative.
///
/// A table cut at a ROW BOUNDARY -- the process died between rows, a pipe
/// filled and the tail was dropped, launchd stopped enumerating -- is
/// byte-for-byte a well-formed SHORTER table: right header, right three-field
/// rows, `com.apple.*` jobs present, and a trailing newline. Nothing in the
/// table's own shape distinguishes it from a complete one. The trailing-newline
/// check in [`launchctl_list_labels`] does not see it either: that check only
/// catches a cut in the MIDDLE of a row, which is the easy half of the problem
/// and not the half that matters.
///
/// So: a `launchctl list` table can prove PRESENCE (a row that is there was
/// not invented by truncation) and can NEVER prove ABSENCE. This type says
/// exactly that much and no more, which forces every caller to go get its
/// absence from a measurement that can actually produce one -- see
/// [`LaunchctlReading::label_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BulkLabelReading {
    /// The row is in the table. A measured positive.
    Present,
    /// The row is not in the table we were given. Either the job is not
    /// loaded, or the table stopped before its row. NOT an absence.
    NotInTable,
    /// There was no table at all.
    Unreadable,
}

/// `label` must match the exact third (`Label`) field of some row; comparing
/// with `==` rather than a substring match keeps `com.origin.server.staging`
/// from reading as `com.origin.server` (H4).
fn bulk_label_reading(snapshot: &LaunchctlSnapshot, label: &str) -> BulkLabelReading {
    match snapshot {
        LaunchctlSnapshot::Unreadable => BulkLabelReading::Unreadable,
        LaunchctlSnapshot::Table(labels) => {
            if labels.iter().any(|l| l == label) {
                BulkLabelReading::Present
            } else {
                BulkLabelReading::NotInTable
            }
        }
    }
}

/// What `launchctl list <label>` establishes about ONE job.
///
/// This is the question whose answer cannot be truncated: it is asked about a
/// single label and answered through an exit status, so there is no table for
/// a short read to cut. What it costs is that the exit status is coarse.
///
/// ONLY exit 0 is mapped, and it is mapped to `Present`: `launchctl list
/// <label>` prints the job's dictionary and exits 0 when it finds the job.
/// EVERY nonzero value is `NotPresentOrFailed` -- a lumped value, deliberately.
/// The not-found code has differed across macOS releases (1, 113, and
/// `EX_UNAVAILABLE` have all been reported), and there is NO macOS host in this
/// worktree to settle which this release uses. "We cannot verify the exit code"
/// is itself a could-not-measure, so it is encoded as one rather than guessed.
///
/// Being honest costs this value its usefulness ON ITS OWN: it lumps "launchd
/// has no such job" together with "launchctl could not be spawned", so nothing
/// may spend a `NotPresentOrFailed` as an absence. That honesty is kept, and
/// the RULE built on top of it is what changed -- see
/// [`TargetedProbeApparatus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetedLabelProbe {
    /// launchctl found the job.
    Present,
    /// launchctl did not answer "found". Could be a genuine not-found, could be
    /// launchctl failing for its own reasons. NOT an absence on its own.
    NotPresentOrFailed,
}

fn targeted_label_probe(launchctl: &dyn LaunchctlExec, label: &str) -> TargetedLabelProbe {
    match launchctl.run(&["list", label]) {
        Ok(out) if out.status.success() => TargetedLabelProbe::Present,
        Ok(out) => {
            log::info!(
                "[lifecycle] launchctl list {label} exited {:?}; on its own that is 'not found' \
                 OR a launchctl error and this code does not claim to know which",
                out.status.code()
            );
            TargetedLabelProbe::NotPresentOrFailed
        }
        Err(e) => {
            log::warn!("[lifecycle] could not run launchctl list {label}: {e}");
            TargetedLabelProbe::NotPresentOrFailed
        }
    }
}

/// Whether `launchctl list <label>` is answering questions on this host right
/// now -- established by asking it one whose answer is already known.
///
/// WHY THIS EXISTS. `NotPresentOrFailed` includes a spawn failure, and the
/// bulk table's silence is not an absence at all (see [`BulkLabelReading`]).
/// Two non-measurements do not add up to a measurement, so a transient
/// targeted failure beside a row-boundary-truncated table must not be read as
/// two agreeing witnesses. The repair is not to make the probe dishonest -- it
/// is to give it a control.
///
/// A POSITIVE CONTROL settles it without needing to know a single macOS exit
/// code. The bulk table just taken names jobs that ARE loaded: its
/// `com.apple.*` rows, whose presence is a measured positive. Ask
/// `launchctl list <one of those>`:
///
/// * exit 0 -- the apparatus works. It just found a job that is really there.
///   A nonzero exit for a DIFFERENT label, from the same binary in the same
///   form moments later, is then a working instrument reporting a different
///   answer. THAT is a measured absence.
/// * anything else -- `launchctl list <label>` is not answering on this host,
///   whatever the reason. Nothing it says about any label means anything, and
///   every label it is asked about comes back `CouldNotMeasure`.
///
/// The control is drawn from the SAME snapshot the question is answered
/// against, so "known loaded" is known at the right instant. Wenlan's own
/// labels are `com.wenlan.*` / `com.origin.*`, so the control is never the
/// subject of the question.
///
/// WHAT THIS DOES NOT ESTABLISH, stated because the gap is real: it does not
/// prove the two calls failed for the same reason, only that the instrument
/// worked once, close by, in the same form. A launchctl that answered for
/// `com.apple.*` and refused for user-domain labels specifically would defeat
/// it. That is a far narrower hole than "any transient spawn failure may
/// delete a plist", and it is the narrowest one reachable without a macOS host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetedProbeApparatus {
    /// A job known to be loaded answered exit 0.
    Working,
    /// No control was available, or the control did not exit 0.
    Unproven,
}

/// What the targeted probe established, with the apparatus taken into account.
/// The third value is the one the old two-witness rule could not express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetedLabelState {
    /// Measured: launchd has the job.
    Present,
    /// Measured: launchd does not have the job -- said by an instrument proven
    /// to be working moments earlier.
    Absent,
    /// Nothing was established.
    CouldNotMeasure,
}

/// One launchctl consultation: a single bulk snapshot, the targeted follow-ups
/// the snapshot cannot answer, and the control that makes those follow-ups mean
/// anything. Held together in one value so the control is paid for at most once
/// per consultation, and so no caller can get hold of an absence that no
/// instrument produced.
struct LaunchctlReading<'a> {
    launchctl: &'a dyn LaunchctlExec,
    snapshot: LaunchctlSnapshot,
    apparatus: std::cell::OnceCell<TargetedProbeApparatus>,
}

impl<'a> LaunchctlReading<'a> {
    fn take(launchctl: &'a dyn LaunchctlExec) -> Self {
        Self {
            launchctl,
            snapshot: launchctl_snapshot(launchctl),
            apparatus: std::cell::OnceCell::new(),
        }
    }

    /// A label the snapshot just showed as loaded, to use as the control.
    /// `launchctl_list_labels` already refuses a table with no `com.apple.*`
    /// job in it, so a `Table` always has one.
    fn control_label(&self) -> Option<&str> {
        match &self.snapshot {
            LaunchctlSnapshot::Table(labels) => labels
                .iter()
                .find(|l| l.starts_with(LAUNCHD_SYSTEM_JOB_PREFIX))
                .map(String::as_str),
            LaunchctlSnapshot::Unreadable => None,
        }
    }

    /// See [`TargetedProbeApparatus`]. Computed at most once per reading, and
    /// only when something actually needs an absence -- a label the snapshot
    /// already shows as loaded costs no control call at all.
    fn apparatus(&self) -> TargetedProbeApparatus {
        *self.apparatus.get_or_init(|| {
            let Some(control) = self.control_label() else {
                log::warn!(
                    "[lifecycle] no launchctl table, so no job known to be loaded to use as a \
                     control; `launchctl list <label>` cannot be trusted to mean absence here"
                );
                return TargetedProbeApparatus::Unproven;
            };
            match targeted_label_probe(self.launchctl, control) {
                TargetedLabelProbe::Present => TargetedProbeApparatus::Working,
                TargetedLabelProbe::NotPresentOrFailed => {
                    log::warn!(
                        "[lifecycle] control probe failed: `launchctl list {control}` did not exit \
                         0 for a job the table had just listed as loaded, so a nonzero exit for \
                         any other label establishes nothing"
                    );
                    TargetedProbeApparatus::Unproven
                }
            }
        })
    }

    fn targeted_label_state(&self, label: &str) -> TargetedLabelState {
        match targeted_label_probe(self.launchctl, label) {
            TargetedLabelProbe::Present => TargetedLabelState::Present,
            TargetedLabelProbe::NotPresentOrFailed => match self.apparatus() {
                TargetedProbeApparatus::Working => TargetedLabelState::Absent,
                TargetedProbeApparatus::Unproven => TargetedLabelState::CouldNotMeasure,
            },
        }
    }

    /// Whether `label` is loaded. THE answer -- there is no weaker one left in
    /// this file to reach for by mistake.
    ///
    /// PRESENCE from either instrument is presence. ABSENCE comes only from the
    /// targeted probe backed by a working control.
    fn label_state(&self, label: &str) -> LabelState {
        match bulk_label_reading(&self.snapshot, label) {
            BulkLabelReading::Present => LabelState::Loaded,
            // The two remaining readings mean the same thing here: the table
            // did not establish an absence and cannot. `Unreadable` also leaves
            // no control available, so it comes back `CouldNotMeasure` below
            // rather than quietly borrowing the targeted probe's coarseness.
            BulkLabelReading::NotInTable | BulkLabelReading::Unreadable => {
                match self.targeted_label_state(label) {
                    TargetedLabelState::Present => LabelState::Loaded,
                    TargetedLabelState::Absent => LabelState::NotLoaded,
                    TargetedLabelState::CouldNotMeasure => LabelState::Unknown,
                }
            }
        }
    }
}

/// Whether BOTH current Wenlan plists are loaded, tri-state. `Unknown`
/// propagates to the Settings toggle, which must not paint an unread state as
/// "off" — the user would then see "Run at Login" disabled while launchd is
/// in fact still starting Wenlan every boot.
///
/// ONE bulk snapshot answers both labels: a pair assembled from two tables
/// taken at two instants is not a state of anything — launchd loading the
/// second label in between reports `NotLoaded` while both are loaded.
///
/// THE TRADE, stated rather than hidden: a label the snapshot does not carry is
/// escalated to its own `launchctl list <label>`, which is a later instant, so
/// the single-instant property holds for every complete table and not for a
/// truncated one. That is the right way round: spending a truncated table as an
/// absence paints the toggle "off" while launchd is starting Wenlan every boot,
/// where the escalation can at worst give a briefly stale answer.
pub fn run_at_login_state(launchctl: &dyn LaunchctlExec) -> LabelState {
    // An isolated dev app has its own bundle identifier and never owns the
    // installed LaunchAgents, so reporting the user's production state here
    // would be a lie the Settings toggle cannot act on. This is a measured
    // negative — it is decided without asking launchctl at all.
    #[cfg(debug_assertions)]
    if std::env::var_os("WENLAN_DEV_APP_ID").is_some() {
        return LabelState::NotLoaded;
    }
    let reading = LaunchctlReading::take(launchctl);
    match (
        reading.label_state(SERVER_PLIST_LABEL),
        reading.label_state(APP_PLIST_LABEL),
    ) {
        (LabelState::Loaded, LabelState::Loaded) => LabelState::Loaded,
        // "Both loaded" is false the moment either one is *measured* absent,
        // whatever the other side did: that much is known. The match is left
        // total so a future third label cannot quietly change the rule.
        (LabelState::NotLoaded, _) | (_, LabelState::NotLoaded) => LabelState::NotLoaded,
        _ => LabelState::Unknown,
    }
}

/// First-run install of both plists. Detects stale paths (e.g. app moved)
/// and re-installs when the embedded path doesn't match the current binary.
/// Returns Ok(()) if the install completed or was unnecessary.
pub fn first_run_install_if_needed(launchctl: &dyn LaunchctlExec) -> Result<()> {
    if data_dir_env_overridden() {
        log::warn!(
            "[lifecycle] skipping first-run LaunchAgent install: isolated run (data-dir env override)"
        );
        return Ok(());
    }

    let exe_canonical = match current_app_path() {
        Ok(path) => path,
        Err(e) => {
            log::warn!("[first-run] unable to resolve current app path: {e}");
            return Ok(());
        }
    };
    first_run_install_if_needed_at_path(launchctl, &exe_canonical)
}

/// The eligibility check runs before the opted-out branch: a dev or
/// human-review binary must not unload the user's legacy LaunchAgents either.
fn first_run_install_if_needed_at_path(
    launchctl: &dyn LaunchctlExec,
    exe_canonical: &Path,
) -> Result<()> {
    if !is_stable_launch_agent_target(exe_canonical) {
        log::warn!(
            "[first-run] skipping LaunchAgent install from non-stable app path: {}",
            exe_canonical.display()
        );
        return Ok(());
    }

    if user_opted_out() {
        if let Err(e) = remove_legacy_app_plist_file_if_owned() {
            log::warn!("[first-run] legacy app plist cleanup failed: {e}");
        }
        if let Err(e) = cleanup_legacy_server_plist(launchctl) {
            log::warn!("[first-run] legacy server plist cleanup failed: {e}");
        }
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
            Err(e) => log::warn!("[first-run] wenlan background on failed: {e}"),
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
        log::warn!(
            "[first-run] preserving legacy server plist because wenlan background on failed"
        );
    }

    log::info!("[first-run] LaunchAgents installed");
    Ok(())
}

/// Why a launchd handover was refused. Typed rather than a bare string so a
/// caller can tell it apart from the other `set_run_at_login` failures, and so
/// the sentence the user reads is defined in exactly one place. `Display` is
/// what `search::set_run_at_login`'s `Result<(), String>` carries to the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandoverRefused {
    /// The app's own sidecar was measured still running after the stop.
    SidecarStillRunning { reason: String },
    /// The stop could not be measured either way.
    SidecarStopUnmeasured { reason: String },
}

impl std::fmt::Display for HandoverRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SidecarStillRunning { reason } => write!(
                f,
                "Run at Login was not changed: Wenlan's own daemon is still running and still \
                 holds the port, so handing it to the system launcher would leave two owners \
                 ({reason}). Quit and reopen Wenlan, then try again."
            ),
            Self::SidecarStopUnmeasured { reason } => write!(
                f,
                "Run at Login was not changed: Wenlan could not confirm its own daemon stopped, \
                 so handing the port to the system launcher is not safe ({reason}). Quit and \
                 reopen Wenlan, then try again."
            ),
        }
    }
}

impl std::error::Error for HandoverRefused {}

/// Whether the launchd handover may proceed given what the stop established
/// about this app's own sidecar.
///
/// The rule the collapse broke: a handover is a *transfer* of one port from
/// this app's sidecar to launchd, and only `Ended` (or "there was never one of
/// ours") establishes that the port was let go. `StillRunning` and
/// `CouldNotMeasure` are the two states in which registering the launchd job
/// creates a rival owner or a restart loop, and they are the two the old code
/// logged and then walked past. Pure, so the rule is testable without
/// launchctl — which a unit test cannot reach anyway, because
/// `set_run_at_login(true)` rejects any binary outside a stable app bundle.
pub fn handover_may_proceed(outcome: &SidecarStopOutcome) -> Result<(), HandoverRefused> {
    match outcome {
        // Measured gone, or there was nothing of ours holding the port.
        SidecarStopOutcome::Ended | SidecarStopOutcome::NoSidecar => Ok(()),
        SidecarStopOutcome::StillRunning { reason } => {
            log::error!(
                "[lifecycle] refusing to hand the daemon to launchd: this app's sidecar is still \
                 running ({reason}); registering the job now would give the port two owners"
            );
            Err(HandoverRefused::SidecarStillRunning {
                reason: reason.clone(),
            })
        }
        SidecarStopOutcome::CouldNotMeasure { reason } => {
            log::error!(
                "[lifecycle] refusing to hand the daemon to launchd: this app's sidecar stop \
                 could not be measured ({reason})"
            );
            Err(HandoverRefused::SidecarStopUnmeasured {
                reason: reason.clone(),
            })
        }
    }
}

/// Toggle "Run at login". Holds a process-wide Mutex for the duration of the
/// install/uninstall sequence so concurrent toggles serialize (G2, spec
/// line 198).
pub async fn set_run_at_login(enabled: bool, launchctl: &dyn LaunchctlExec) -> Result<()> {
    let _guard = RUN_AT_LOGIN_LOCK.lock().await;
    if data_dir_env_overridden() {
        log::info!(
            "[lifecycle] skipping Run at Login change: isolated run (data-dir env override)"
        );
        anyhow::bail!(
            "refusing to change Run at Login during an isolated run with a data-dir env override"
        );
    }

    if enabled {
        let exe = current_app_path()?;
        if !is_stable_launch_agent_target(&exe) {
            anyhow::bail!(
                "refusing to enable Run at Login from non-stable app path: {}",
                exe.display()
            );
        }
        // The "Start Wenlan" button must not spawn a sidecar between the stop
        // below and the launchd registration; the count releases on drop.
        let _pending = crate::daemon_start::LaunchdInstallPending::begin();
        // A sidecar this app spawned holds the port that `wenlan background
        // on` is about to hand to launchd, and the CLI's pre-install shutdown
        // request failed against it (first-run gauntlet finding F16). Stop it
        // through its child handle first; a launchd-owned daemon is never in
        // that slot, so this is a no-op once launchd owns the daemon.
        //
        // The stop outcome is a decision input, not a diagnostic: a handover
        // this code has just established is unsafe is not performed, because
        // registering launchd against a still-held port makes a second owner
        // or a restart loop. The toggle stays usable — a failed stop does not
        // discard the sidecar record (so the next attempt can retry the kill),
        // and `search::set_run_at_login` restarts the daemon when this returns
        // `Err` and nothing owns the port.
        handover_may_proceed(&crate::daemon_start::stop_sidecar().await)?;
        set_user_opted_out(false)?;
        install_server_plist_via_subprocess(launchctl)?;
        install_app_plist(launchctl)?;
    } else {
        set_user_opted_out(true)?;
        uninstall_app_plist(launchctl)?;
        let legacy_app_cleanup_result = remove_legacy_app_plist_file_if_owned();
        let legacy_server_cleanup_result = cleanup_legacy_server_plist(launchctl);
        let uninstall_result = uninstall_server_plist(launchctl);
        legacy_app_cleanup_result?;
        legacy_server_cleanup_result?;
        uninstall_result?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn handover_pending() -> bool {
    HANDOVER_BUNDLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
}

/// Wait until the daemon stops answering its health route *and* its listener
/// is closed, or `limit` passes. The limit is a hard ceiling: it bounds the
/// health calls themselves, not only the gaps between them. The closed port
/// is the fact the reopened app needs; health can stop answering while the
/// listener lingers. Returns whether the port was released within `limit`.
pub(crate) async fn wait_for_daemon_to_stop(limit: Duration) -> bool {
    let client = crate::api::WenlanClient::new();
    let listener = listener_addr(client.base_url());
    let released = tokio::time::timeout(limit, async {
        while client.health().await.is_ok() {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        if let Some(addr) = listener.as_deref() {
            while tokio::net::TcpStream::connect(addr).await.is_ok() {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    })
    .await;
    if released.is_err() {
        log::warn!(
            "[lifecycle] daemon still holding {} after {limit:?}",
            client.base_url()
        );
    }
    released.is_ok()
}

/// `host:port` of the daemon listener behind a base URL, for a raw TCP probe.
fn listener_addr(base_url: &str) -> Option<String> {
    let rest = base_url
        .strip_prefix("http://")
        .or_else(|| base_url.strip_prefix("https://"))?;
    let authority = rest.split('/').next()?;
    (!authority.is_empty()).then(|| authority.to_string())
}

pub(crate) fn shutdown_url_for(client: &crate::api::WenlanClient) -> String {
    format!("{}/api/shutdown", client.base_url())
}

pub async fn quit_origin(app_handle: &AppHandle) -> Result<()> {
    // Debounce: tray menu Quit Wenlan item stays clickable during the 500ms
    // shutdown sleep; double-click would otherwise spawn 2× POSTs (H1).
    let Some(attempt) = QuitAttemptGuard::try_begin(&QUITTING) else {
        return Ok(());
    };
    log::info!("{FULL_QUIT_BREADCRUMB}");

    if data_dir_env_overridden() {
        log::info!("[lifecycle] skipping quit teardown: isolated run (data-dir env override)");
        // The isolated run has no launchd job: the daemon is our sidecar.
        log_sidecar_stop_on_quit(crate::daemon_start::stop_sidecar().await);
        exit_after_quit(app_handle, 0);
        attempt.commit();
        return Ok(());
    }

    let quit_plan = quit_plan_for_target_os(std::env::consts::OS);

    if quit_plan.clean_launch_agents {
        // Spec lifecycle invariant #4: "Quit Wenlan = full off; both plists
        // unloaded, both processes exit, no auto-restart on reboot." (H2)
        // Order matters: uninstall plists FIRST so launchd won't respawn after
        // the daemon dies, then shut the daemon down cleanly. macOS only —
        // no other platform may invoke launchctl or manufacture a
        // LaunchAgents path.
        let launchctl = SystemLaunchctl;
        if let Err(e) = uninstall_app_plist(&launchctl) {
            log::warn!("[quit] uninstall_app_plist failed: {e}");
        }
        if let Err(e) = uninstall_server_plist(&launchctl) {
            log::warn!("[quit] uninstall_server_plist failed: {e}");
        }
        if let Err(e) = cleanup_legacy_app_plist(&launchctl) {
            log::warn!("[quit] cleanup_legacy_app_plist failed: {e}");
        }
        if let Err(e) = cleanup_legacy_server_plist(&launchctl) {
            log::warn!("[quit] cleanup_legacy_server_plist failed: {e}");
        }
    } else {
        log::info!("[lifecycle] LaunchAgent cleanup is not applicable on this platform");
    }

    if quit_plan.shutdown_daemon {
        // Tell the daemon this app selected to shut down cleanly.
        let shutdown_url = shutdown_url_for(&crate::api::WenlanClient::new());
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()?;
        let _ = client.post(shutdown_url).send().await;

        // Wait briefly for the daemon to flush.
        tokio::time::sleep(Duration::from_millis(500)).await;
        // A handover reopens a newer app right after this one exits, and
        // that app starts its own daemon on the same port: give the old
        // daemon a bounded moment to release it.
        #[cfg(target_os = "macos")]
        if handover_pending() {
            wait_for_daemon_to_stop(Duration::from_secs(5)).await;
        }
    }

    if quit_plan.exit_app {
        // A sidecar we spawned (no launchd job, or a respawn from Diagnostics)
        // must not outlive the app: the next launch would adopt it.
        log_sidecar_stop_on_quit(crate::daemon_start::stop_sidecar().await);
        exit_after_quit(app_handle, 0);
        attempt.commit();
    }
    Ok(())
}

/// The quit path's branch on [`SidecarStopOutcome`]. There is nothing left to
/// *do* here — the app is about to exit and the outcome already says the kill
/// was attempted — but the two failing outcomes are the exact shape the next
/// launch will hit as "a stale daemon already holds the port", so the log the
/// user sends must contain them rather than a uniform silence.
fn log_sidecar_stop_on_quit(outcome: SidecarStopOutcome) {
    match outcome {
        SidecarStopOutcome::Ended => log::info!("[quit] the app's sidecar daemon ended"),
        SidecarStopOutcome::NoSidecar => {
            log::info!("[quit] this app owned no sidecar; nothing to stop")
        }
        SidecarStopOutcome::StillRunning { reason } => log::error!(
            "[quit] the app's sidecar daemon is STILL RUNNING after the stop ({reason}); the next \
             launch will find the port held"
        ),
        SidecarStopOutcome::CouldNotMeasure { reason } => log::error!(
            "[quit] could not establish whether the app's sidecar daemon ended ({reason})"
        ),
    }
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
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    #[cfg(windows)]
    use std::os::windows::process::ExitStatusExt;
    use std::sync::Mutex;

    use crate::test_env::EnvGuard;

    /// Build an ExitStatus from a plain exit code. The two platforms disagree
    /// about what `from_raw` takes — unix wants the wait(2) encoding, where
    /// the code sits in the high byte, Windows wants the code itself — so
    /// every mock status goes through here rather than hard-coding either.
    fn exit_status(code: u32) -> std::process::ExitStatus {
        #[cfg(unix)]
        {
            std::process::ExitStatus::from_raw((code as i32) << 8)
        }
        #[cfg(windows)]
        {
            std::process::ExitStatus::from_raw(code)
        }
    }

    #[test]
    fn synthetic_exit_status_preserves_success_semantics() {
        assert!(exit_status(0).success());
        assert!(!exit_status(1).success());
    }

    /// The environment every lifecycle test mutates: the home directory the
    /// plist paths hang off, both data-dir overrides, and the dev bundle id.
    const LIFECYCLE_ENV_KEYS: &[&str] = &[
        "HOME",
        "WENLAN_DATA_DIR",
        "ORIGIN_DATA_DIR",
        "WENLAN_DEV_APP_ID",
    ];

    /// A `launchctl list` table shaped like the real one: the `PID Status
    /// Label` header, enough `com.apple.*` rows to clear the witnesses in
    /// [`launchctl_list_labels`], and then whatever labels the test wants
    /// loaded. Fixtures go through this rather than writing a bare row,
    /// because a bare row is no longer a table and the code under test is
    /// right to say so.
    fn launchctl_table(loaded: &[&str]) -> String {
        let mut table = format!("{}\n", LAUNCHCTL_LIST_HEADER.join("\t"));
        for i in 0..LAUNCHCTL_MIN_ROWS {
            table.push_str(&format!(
                "-\t0\t{LAUNCHD_SYSTEM_JOB_PREFIX}fixture.job{i}\n"
            ));
        }
        for (i, label) in loaded.iter().enumerate() {
            table.push_str(&format!("{}\t0\t{label}\n", 100 + i));
        }
        table
    }

    /// A server LaunchAgent plist that targets `data_dir`.
    fn server_plist_for_data_dir(data_dir: &Path) -> String {
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
            data_dir.display()
        )
    }

    /// Stage that plist at the real `server_plist_path()`, creating the
    /// LaunchAgents directory. Returns the path it was written to.
    fn write_server_plist_for(data_dir: &Path) -> PathBuf {
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, server_plist_for_data_dir(data_dir)).unwrap();
        plist
    }

    /// A nonzero exit for the targeted `launchctl list <label>` form. The real
    /// not-found code has varied across macOS releases and this worktree has no
    /// macOS host to settle it — which is the whole reason `targeted_label_probe`
    /// maps only exit 0 and lumps everything else. 113 is one of the reported
    /// values and stands in for "some nonzero", nothing more; no production code
    /// compares against a specific number, so the fixture's choice is not load-
    /// bearing.
    const TARGETED_LIST_NOT_FOUND_FIXTURE: u32 = 113;

    struct MockLaunchctl {
        calls: Mutex<Vec<Vec<String>>>,
        /// Status code to return for load/start subcommands. Default 0 = ok.
        load_status: Mutex<u32>,
        /// Status code to return for unload subcommands. Default 0 = ok.
        unload_status: Mutex<u32>,
        /// Status code to return for the BARE `launchctl list`. Default 0 = the
        /// table on stdout is the whole answer; nonzero = launchctl answered
        /// nothing usable, which callers must read as could-not-measure, not as
        /// absence.
        list_status: Mutex<u32>,
        /// How the TARGETED `launchctl list <label>` answers. A different
        /// question from the bare `list`, with a different failure mode, so it
        /// needs its own knob -- and it needs to answer PER LABEL, because the
        /// control probe that decides whether a nonzero exit means anything is
        /// itself a targeted call. See [`TargetedList`].
        targeted: Mutex<TargetedList>,
        /// Canned bare-`launchctl list` stdout body.
        list_stdout: Mutex<String>,
    }

    impl MockLaunchctl {
        /// Every `launchctl` invocation the mock saw, as argv vectors.
        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }

        /// How many times the BARE `launchctl list` was run. The pair-state
        /// single-snapshot invariant is stated in this number.
        fn bare_list_calls(&self) -> usize {
            self.calls()
                .iter()
                .filter(|c| c.len() == 1 && c[0] == "list")
                .count()
        }
    }

    impl Default for MockLaunchctl {
        fn default() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                load_status: Mutex::new(0),
                unload_status: Mutex::new(0),
                list_status: Mutex::new(0),
                targeted: Mutex::new(TargetedList::FromTable),
                // A well-formed table that simply carries no Wenlan label —
                // the measured negative. An EMPTY stdout is NOT this: it
                // means could-not-measure.
                list_stdout: Mutex::new(launchctl_table(&[])),
            }
        }
    }
    /// How [`MockLaunchctl`] answers the TARGETED `launchctl list <label>`.
    ///
    /// Per-label, not one status for all of them, because the production rule
    /// now asks a CONTROL question through the same call: a nonzero exit for
    /// the label under test only means "absent" if a job known to be loaded
    /// answered 0 moments earlier. A fixture that cannot answer those two
    /// differently cannot reach either side of that rule.
    enum TargetedList {
        /// The honest host: exit 0 for exactly the labels the bulk table
        /// prints. Table and targeted probe agree, because on a healthy
        /// machine they do.
        FromTable,
        /// The host whose bulk table is a TRUNCATED VIEW of the real domain:
        /// exit 0 for these labels whatever the table happens to show. This is
        /// how a row-boundary truncation is staged -- the table lies by
        /// omission, the targeted probe does not.
        Loaded(Vec<String>),
        /// Every targeted call exits with this code, the control included: a
        /// launchctl whose targeted form says "no" to everything, which is
        /// indistinguishable from one that is broken.
        Always(u32),
        /// The targeted form cannot be SPAWNED, while the bulk form still
        /// works. The C1.1 attack in one value: `NotPresentOrFailed` produced
        /// by a FAILURE, not by an absence.
        SpawnFails,
    }

    impl LaunchctlExec for MockLaunchctl {
        fn run(&self, args: &[&str]) -> io::Result<Output> {
            self.calls
                .lock()
                .unwrap()
                .push(args.iter().map(|s| s.to_string()).collect());
            // `list` and `list <label>` are two different questions; the mock
            // has to be able to answer them differently or the tests cannot
            // reach the branch where they disagree.
            let targeted_list = args.first().copied() == Some("list") && args.len() > 1;
            if targeted_list {
                let label = args[1];
                let code = match &*self.targeted.lock().unwrap() {
                    TargetedList::FromTable => {
                        let table = self.list_stdout.lock().unwrap().clone();
                        let listed = launchctl_list_labels(&table)
                            .map(|labels| labels.contains(&label))
                            .unwrap_or(false);
                        if listed {
                            0
                        } else {
                            TARGETED_LIST_NOT_FOUND_FIXTURE
                        }
                    }
                    TargetedList::Loaded(labels) => {
                        if labels.iter().any(|l| l == label) {
                            0
                        } else {
                            TARGETED_LIST_NOT_FOUND_FIXTURE
                        }
                    }
                    TargetedList::Always(code) => *code,
                    TargetedList::SpawnFails => {
                        return Err(io::Error::new(
                            io::ErrorKind::NotFound,
                            "launchctl: no such file or directory",
                        ))
                    }
                };
                return Ok(Output {
                    status: exit_status(code),
                    stdout: vec![],
                    stderr: vec![],
                });
            }
            // Tests can override load/unload/list statuses independently.
            let status_code = match args.first().copied() {
                Some("load") => *self.load_status.lock().unwrap(),
                Some("unload") => *self.unload_status.lock().unwrap(),
                Some("list") => *self.list_status.lock().unwrap(),
                _ => 0,
            };
            let stdout = match args.first().copied() {
                Some("list") => self.list_stdout.lock().unwrap().as_bytes().to_vec(),
                _ => vec![],
            };
            Ok(Output {
                status: exit_status(status_code),
                stdout,
                stderr: vec![],
            })
        }
    }

    /// The jobs really loaded on the fixture host: everything the default
    /// table carries (the `com.apple.*` control jobs included) plus `extra`.
    /// Used to state a truth the bulk table is then allowed to under-report.
    fn loaded_domain(extra: &[&str]) -> Vec<String> {
        launchctl_list_labels(&launchctl_table(extra))
            .expect("the fixture table parses")
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// A launchctl that cannot be run at all — the spawn itself fails, so no
    /// subcommand ever produces an answer. Every probe against it is a failed
    /// measurement and none of them is evidence of absence.
    struct UnrunnableLaunchctl;
    impl LaunchctlExec for UnrunnableLaunchctl {
        fn run(&self, _args: &[&str]) -> io::Result<Output> {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "launchctl: no such file or directory",
            ))
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        // Point the data dir at the tempdir directly. Overriding HOME only
        // relocates the app data root on unix; Windows resolves it from its
        // own known folders and would write into the real profile.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("WENLAN_DATA_DIR", tmp.path());
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        // The opt-out sentinel must NOT live inside the daemon's typed
        // `config.json` — otherwise unrelated `Config::save` calls overwrite
        // the file and silently drop the user's opt-out preference (C1).
        let tmp = tempfile::tempdir().unwrap();
        // The data root has to be relocated explicitly: `HOME` does not move
        // it on Windows, and this test writes the opt-out flag.
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
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
    fn data_dir_env_overridden_reports_either_override_key() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        assert!(!data_dir_env_overridden());

        std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-isolated");
        assert!(data_dir_env_overridden());

        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-isolated");
        assert!(data_dir_env_overridden());
    }

    #[test]
    #[serial_test::serial]
    fn stable_launch_agent_target_classifies_system_paths() {
        for (exe, expected) in [
            (
                "/Applications/Wenlan.app/Contents/MacOS/origin-app",
                StableLaunchAgentTarget::Current,
            ),
            (
                "/Applications/Wenlan.app/Contents/MacOS/wenlan-app",
                StableLaunchAgentTarget::Current,
            ),
            (
                "/Applications/Origin.app/Contents/MacOS/origin",
                StableLaunchAgentTarget::LegacyOrigin,
            ),
            (
                "/Users/alice/Downloads/Wenlan.app/Contents/MacOS/origin-app",
                StableLaunchAgentTarget::Rejected,
            ),
        ] {
            assert_eq!(
                classify_stable_launch_agent_target(std::path::Path::new(exe)),
                expected,
                "{exe}"
            );
        }
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        // The plist template embeds log paths under `app_data_dir()`, which
        // `HOME` does not move on Windows.
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
        std::env::set_var("HOME", tmp.path());

        let mock = MockLaunchctl {
            load_status: Mutex::new(1),
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        // See `install_app_plist_rolls_back_file_when_launchctl_load_fails`.
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
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

        let listed = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[SERVER_PLIST_LABEL, "com.wenlan.desktop"])),
            ..Default::default()
        };
        assert_eq!(run_at_login_state(&listed), LabelState::Loaded);
        assert_eq!(
            listed.bare_list_calls(),
            1,
            "both labels are in the snapshot, so nothing needs escalating and one bulk read \
             answers the pair"
        );
    }

    #[test]
    fn is_run_at_login_enabled_returns_false_when_one_missing() {
        let only_server = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[SERVER_PLIST_LABEL])),
            ..Default::default()
        };
        assert_eq!(run_at_login_state(&only_server), LabelState::NotLoaded);
    }

    /// The failed measurement must not read as "Run at Login is off". A
    /// launchctl that will not run knows nothing about either job; reporting
    /// `NotLoaded` here paints the Settings toggle disabled while launchd may
    /// still be starting Wenlan every boot.
    #[test]
    #[serial_test::serial]
    fn a_launchctl_that_cannot_run_is_unknown_not_disabled() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        std::env::remove_var("WENLAN_DEV_APP_ID");
        assert_eq!(
            run_at_login_state(&UnrunnableLaunchctl),
            LabelState::Unknown,
            "a launchctl that could not be executed is a failed measurement, not an absence"
        );
    }

    /// Same for a launchctl that ran and answered nonzero: the `list` table it
    /// was supposed to print is the whole contract, and without a successful
    /// exit there is no table to conclude anything from.
    #[test]
    #[serial_test::serial]
    fn a_nonzero_launchctl_list_is_unknown_not_disabled() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        std::env::remove_var("WENLAN_DEV_APP_ID");
        let refused = MockLaunchctl {
            list_status: Mutex::new(1),
            ..Default::default()
        };
        assert_eq!(run_at_login_state(&refused), LabelState::Unknown);
    }

    /// The fixture makes two instants disagree in the most damaging way: at T1
    /// only the server label is loaded, at T2 only the app label is. Asking
    /// once per label would conclude `Loaded` for a pair that was never
    /// simultaneously loaded; one snapshot can only answer from T1, where the
    /// app label is absent, so the honest answer is `NotLoaded`. The call count
    /// is asserted directly — a passing result with two calls is a coincidence.
    #[test]
    #[serial_test::serial]
    fn run_at_login_state_answers_both_labels_from_one_snapshot() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        std::env::remove_var("WENLAN_DEV_APP_ID");

        /// A launchd whose domain changes between calls: server-only, then
        /// app-only.
        struct ShiftingDomain {
            bare_list_calls: Mutex<usize>,
            tables: Vec<String>,
        }
        impl LaunchctlExec for ShiftingDomain {
            fn run(&self, args: &[&str]) -> io::Result<Output> {
                // A TARGETED `list <label>` is answered out of the table
                // the caller was last handed -- the domain as of the snapshot
                // it is holding -- so the escalation path cannot accidentally
                // read the *next* instant and hide a torn bulk read.
                if args.len() > 1 {
                    let served = *self.bare_list_calls.lock().unwrap();
                    let table =
                        self.tables[served.saturating_sub(1).min(self.tables.len() - 1)].clone();
                    let listed = launchctl_list_labels(&table)
                        .map(|labels| labels.contains(&args[1]))
                        .unwrap_or(false);
                    return Ok(Output {
                        status: exit_status(if listed {
                            0
                        } else {
                            TARGETED_LIST_NOT_FOUND_FIXTURE
                        }),
                        stdout: vec![],
                        stderr: vec![],
                    });
                }
                let mut n = self.bare_list_calls.lock().unwrap();
                let table = self.tables[(*n).min(self.tables.len() - 1)].clone();
                *n += 1;
                Ok(Output {
                    status: exit_status(0),
                    stdout: table.into_bytes(),
                    stderr: vec![],
                })
            }
        }

        let shifting = ShiftingDomain {
            bare_list_calls: Mutex::new(0),
            tables: vec![
                launchctl_table(&[SERVER_PLIST_LABEL]),
                launchctl_table(&[APP_PLIST_LABEL]),
            ],
        };
        assert_eq!(
            run_at_login_state(&shifting),
            LabelState::NotLoaded,
            "a pair state stitched from two launchctl readings can report a pair that was never \
             loaded at the same time"
        );
        assert_eq!(
            *shifting.bare_list_calls.lock().unwrap(),
            1,
            "both labels must be answered from ONE launchctl list; a second call is a second \
             instant and re-opens the torn read"
        );

        // The `Unknown` half of the contract, from the same one snapshot: a
        // launchctl that answers nothing settles neither label.
        let unreadable = MockLaunchctl {
            list_status: Mutex::new(1),
            ..Default::default()
        };
        assert_eq!(run_at_login_state(&unreadable), LabelState::Unknown);
        assert_eq!(unreadable.bare_list_calls(), 1);
    }

    #[test]
    fn is_run_at_login_enabled_does_not_match_label_substring() {
        // H4: `launchctl list` output where a different label has our label
        // as a prefix (e.g. `com.origin.server.staging`) must not be treated
        // as our service being present.
        // Note: only the `.staging` suffixed labels appear. Real labels are
        // absent -- from the table AND, via `FromTable`, from the targeted
        // probe, so the exact-match rule is tested on both instruments.
        let staging_only = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[
                &format!("{SERVER_PLIST_LABEL}.staging"),
                &format!("{APP_PLIST_LABEL}.staging"),
            ])),
            ..Default::default()
        };
        assert_eq!(
            run_at_login_state(&staging_only),
            LabelState::NotLoaded,
            ".staging suffixed labels must not satisfy exact-label match"
        );
    }

    #[test]
    fn listener_addr_is_the_authority_of_the_base_url() {
        assert_eq!(
            listener_addr("http://127.0.0.1:7878").as_deref(),
            Some("127.0.0.1:7878")
        );
        assert_eq!(
            listener_addr("http://127.0.0.1:7878/").as_deref(),
            Some("127.0.0.1:7878")
        );
        assert_eq!(
            listener_addr("https://localhost:7878/api").as_deref(),
            Some("localhost:7878")
        );
        assert_eq!(listener_addr("127.0.0.1:7878"), None);
        assert_eq!(listener_addr("http://"), None);
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

    /// The two legacy plists (app and server) each with their path, an owned
    /// body, a foreign body, and their cleanup verb — the three cleanup tests
    /// below assert the same contract for both.
    #[allow(clippy::type_complexity)]
    const LEGACY_CLEANUPS: [(
        &str,
        fn() -> Result<PathBuf>,
        fn() -> String,
        fn() -> String,
        fn(&dyn LaunchctlExec) -> Result<()>,
    ); 2] = [
        (
            "app",
            legacy_app_plist_path,
            owned_legacy_app_plist,
            foreign_legacy_app_plist,
            cleanup_legacy_app_plist,
        ),
        (
            "server",
            legacy_server_plist_path,
            owned_legacy_server_plist,
            foreign_legacy_server_plist,
            cleanup_legacy_server_plist,
        ),
    ];

    fn stage_legacy_plist(path_of: fn() -> Result<PathBuf>, body: fn() -> String) -> PathBuf {
        let plist = path_of().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, body()).unwrap();
        plist
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_plist_unloads_and_removes_owned_file() {
        for (kind, path_of, owned, _foreign, cleanup) in LEGACY_CLEANUPS {
            let tmp = tempfile::tempdir().unwrap();
            std::env::set_var("HOME", tmp.path());
            let plist = stage_legacy_plist(path_of, owned);

            let mock = MockLaunchctl::default();
            cleanup(&mock).unwrap();

            assert!(!plist.exists(), "legacy {kind} plist removed");
            let calls = mock.calls.lock().unwrap();
            assert!(
                calls
                    .iter()
                    .any(|c| c[0] == "unload" && c[1] == plist.to_string_lossy()),
                "legacy {kind} plist unloaded before removal"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_plist_removes_owned_file_when_unload_fails() {
        for (kind, path_of, owned, _foreign, cleanup) in LEGACY_CLEANUPS {
            let tmp = tempfile::tempdir().unwrap();
            std::env::set_var("HOME", tmp.path());
            let plist = stage_legacy_plist(path_of, owned);

            let mock = MockLaunchctl {
                unload_status: Mutex::new(1),
                ..Default::default()
            };
            cleanup(&mock).unwrap();

            assert!(
                !plist.exists(),
                "owned legacy {kind} plist removed even when unload fails"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn cleanup_legacy_plist_preserves_foreign_file() {
        for (kind, path_of, _owned, foreign, cleanup) in LEGACY_CLEANUPS {
            let tmp = tempfile::tempdir().unwrap();
            std::env::set_var("HOME", tmp.path());
            let plist = stage_legacy_plist(path_of, foreign);

            let mock = MockLaunchctl::default();
            cleanup(&mock).unwrap();

            assert!(plist.exists(), "foreign legacy {kind} plist preserved");
            assert!(
                mock.calls.lock().unwrap().is_empty(),
                "foreign legacy {kind} plist must not be unloaded"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn uninstall_server_plist_keeps_file_when_unload_fails_and_still_loaded() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        // Both instruments agree the job is loaded: it is in the table AND
        // the targeted `launchctl list <label>` exits 0.
        let mock = MockLaunchctl {
            unload_status: Mutex::new(1),
            list_stdout: Mutex::new(launchctl_table(&[SERVER_PLIST_LABEL])),
            targeted: Mutex::new(TargetedList::FromTable),
            ..Default::default()
        };

        let err = uninstall_server_plist(&mock).unwrap_err();
        assert!(err.to_string().contains(SERVER_PLIST_LABEL));
        assert!(
            plist.exists(),
            "plist must survive a failed unload while the job is still loaded"
        );
    }

    #[test]
    #[serial_test::serial]
    fn uninstall_server_plist_removes_file_when_unload_fails_but_not_loaded() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        let mock = MockLaunchctl {
            unload_status: Mutex::new(1),
            ..Default::default()
        };

        uninstall_server_plist(&mock).unwrap();
        assert!(
            !plist.exists(),
            "stale plist removed once launchctl confirms the job is not loaded"
        );
    }

    /// A server plist that is THERE and cannot be READ says nothing about which
    /// data root launchd targets, so it must not answer the same as one that
    /// genuinely points elsewhere. The unreadable fixture is a DIRECTORY at the
    /// plist path: `read_to_string` refuses it on every platform, with an error
    /// that is not `NotFound`.
    #[test]
    #[serial_test::serial]
    fn a_server_plist_that_cannot_be_read_is_unknown_not_a_mismatch() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();

        // No plist at all is a MEASURED negative and must stay one.
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        assert_eq!(
            server_plist_data_dir_match(),
            ServerPlistMatch::DoesNotMatch
        );

        // A plist path that cannot be read is NOT.
        std::fs::create_dir_all(&plist).unwrap();
        assert_eq!(
            server_plist_data_dir_match(),
            ServerPlistMatch::Unknown,
            "a plist whose contents could not be read was reported as a measured mismatch"
        );

        // And the ownership answer built on it stays a failed measurement
        // rather than becoming "launchd does not own the daemon".
        assert_eq!(
            launchd_owns_server_daemon(&MockLaunchctl::default()),
            LaunchdOwnership::Unknown
        );
    }

    /// `launchctl unload` fails and the follow-up `launchctl list` also fails.
    /// Reading that non-answer as "not loaded" deletes the plist of a live
    /// `KeepAlive` daemon, leaving no registration to unload it by: a
    /// could-not-measure must keep the file and fail the uninstall.
    #[test]
    #[serial_test::serial]
    fn uninstall_server_plist_keeps_file_when_launchctl_cannot_be_measured() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        // launchctl will not run at all: neither the unload nor the readback
        // produced an answer.
        let err = uninstall_server_plist(&UnrunnableLaunchctl).unwrap_err();
        assert!(
            err.to_string().contains(SERVER_PLIST_LABEL),
            "the error must name the job it could not verify: {err}"
        );
        assert!(
            plist.exists(),
            "a plist must never be deleted on a launchctl reading that never happened"
        );

        // Same when launchctl runs but its `list` exits nonzero: the table
        // that is the whole contract of `launchctl list` never arrived.
        let refused = MockLaunchctl {
            unload_status: Mutex::new(1),
            list_status: Mutex::new(1),
            ..Default::default()
        };
        uninstall_server_plist(&refused).unwrap_err();
        assert!(
            plist.exists(),
            "an empty stdout from a failed `launchctl list` is not evidence the job is gone"
        );
    }

    /// The same non-answer must not make the app think launchd owns the
    /// daemon, and must not be flattened into "measured not owned" on the way
    /// out either.
    #[test]
    #[serial_test::serial]
    fn launchd_ownership_is_not_claimed_from_an_unreadable_launchctl() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        // A plist that DOES target the selected data root, so the launchctl
        // probe is the only thing left that can answer.
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, server_plist_for_data_dir(data.path())).unwrap();
        assert_eq!(server_plist_data_dir_match(), ServerPlistMatch::Matches);

        assert_eq!(
            LaunchctlReading::take(&UnrunnableLaunchctl).label_state(SERVER_PLIST_LABEL),
            LabelState::Unknown,
            "the probe behind launchd ownership must report could-not-measure"
        );
        assert_eq!(
            launchd_owns_server_daemon(&UnrunnableLaunchctl),
            LaunchdOwnership::Unknown,
            "an unreadable launchctl is not evidence launchd does not own the daemon"
        );
        assert_ne!(
            launchd_owns_server_daemon(&UnrunnableLaunchctl),
            LaunchdOwnership::DoesNot,
            "the could-not-measure must survive the return, not just the log"
        );
    }

    /// The other input to the ownership decision has the same shape. A plist
    /// that is not there is a measured "launchd does not own this"; a plist
    /// that is there and cannot be READ says nothing about its contents. A
    /// directory at the plist path stands in for the unreadable file, because
    /// it fails with a non-`NotFound` error on every platform this ships to.
    #[test]
    #[serial_test::serial]
    fn an_unreadable_server_plist_is_not_a_measured_absence() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        // No plist at all: a real negative.
        assert_eq!(
            server_plist_data_dir_match(),
            ServerPlistMatch::DoesNotMatch
        );
        let loaded = MockLaunchctl::default();
        *loaded.list_stdout.lock().unwrap() = launchctl_table(&[SERVER_PLIST_LABEL]);
        assert_eq!(
            launchd_owns_server_daemon(&loaded),
            LaunchdOwnership::DoesNot
        );

        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(&plist).unwrap();
        assert_eq!(server_plist_data_dir_match(), ServerPlistMatch::Unknown);
        assert_eq!(
            launchd_owns_server_daemon(&loaded),
            LaunchdOwnership::Unknown,
            "a plist that could not be read is not a plist that says no"
        );
    }

    /// A1. `launchctl list` exits 0 and prints nothing — a stub on `PATH`, a
    /// truncated pipe, a launchctl that lost its domain. The old code took
    /// *any* exit-0 stdout as a complete table, so "our label is not in it"
    /// came out of an empty one. That answer is what authorized deleting the
    /// plist of a job that may still have been loaded.
    #[test]
    fn an_exit_zero_launchctl_list_that_is_not_a_table_is_unknown() {
        for (name, body) in [
            ("empty stdout", String::new()),
            ("only whitespace", "\n\n  \n".to_string()),
            // Rows, no header: the shape `launchctl list` never prints.
            (
                "header missing",
                launchctl_table(&[SERVER_PLIST_LABEL])
                    .lines()
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            // A header and one row: a table cut off after the first write.
            (
                "truncated table",
                format!(
                    "{}\n-\t0\tcom.apple.only\n",
                    LAUNCHCTL_LIST_HEADER.join("\t")
                ),
            ),
            // Long enough, but every row is a diagnostic rather than a row.
            (
                "a message, not a table",
                format!(
                    "{}\n{}",
                    LAUNCHCTL_LIST_HEADER.join("\t"),
                    "Could not connect to the bootstrap server\n".repeat(20)
                ),
            ),
            // Well-formed and long enough, but no launchd user domain has no
            // com.apple.* job in it — so this is not one.
            ("no system jobs", {
                let mut table = format!("{}\n", LAUNCHCTL_LIST_HEADER.join("\t"));
                for i in 0..LAUNCHCTL_MIN_ROWS {
                    table.push_str(&format!("-\t0\tlocal.invented.job{i}\n"));
                }
                table
            }),
        ] {
            let mock = MockLaunchctl::default();
            *mock.list_stdout.lock().unwrap() = body;
            assert_eq!(
                LaunchctlReading::take(&mock).label_state(SERVER_PLIST_LABEL),
                LabelState::Unknown,
                "{name}: exit 0 without the launchctl table is a failed measurement, not an absence"
            );
        }
    }

    /// The other half: a real table still answers both ways, so the witnesses
    /// above did not simply turn every measurement into `Unknown`.
    #[test]
    fn a_well_formed_launchctl_table_still_answers_loaded_and_not_loaded() {
        let loaded = MockLaunchctl::default();
        *loaded.list_stdout.lock().unwrap() = launchctl_table(&[SERVER_PLIST_LABEL]);
        assert_eq!(
            LaunchctlReading::take(&loaded).label_state(SERVER_PLIST_LABEL),
            LabelState::Loaded
        );
        // A complete table without our label is still a real negative -- but
        // now it is one because the TARGETED probe said so with a control
        // behind it, not because the table was silent. `FromTable` is the
        // honest host: the table and the targeted probe agree.
        assert_eq!(
            LaunchctlReading::take(&MockLaunchctl::default()).label_state(SERVER_PLIST_LABEL),
            LabelState::NotLoaded,
            "a measured absence, confirmed by a probe whose control answered, must stay a negative"
        );
    }

    /// A `launchctl list` whose stdout was cut mid-write ends without a
    /// newline, and every other parser check passes on such a body (intact
    /// header, well-formed complete rows, more than ten of them, a
    /// `com.apple.*` job among them). The fixture cuts at a point where the
    /// partial tail is still a SHAPE-VALID row — the only cut the row checks
    /// cannot already catch.
    #[test]
    fn a_launchctl_table_cut_mid_row_is_not_a_table() {
        let mut cut = launchctl_table(&[]);
        cut.push_str("-\t0\tcom.apple.cut.off"); // no trailing newline: short read
        assert_eq!(
            launchctl_list_labels(&cut),
            None,
            "stdout that does not end in a newline lost an unknown amount of table"
        );

        let mock = MockLaunchctl::default();
        *mock.list_stdout.lock().unwrap() = cut;
        assert_eq!(
            LaunchctlReading::take(&mock).label_state(SERVER_PLIST_LABEL),
            LabelState::Unknown,
            "a short read is a failed measurement, not an absent job"
        );

        // And the check does not fire on the whole table it is meant to accept.
        assert_eq!(
            LaunchctlReading::take(&MockLaunchctl::default()).label_state(SERVER_PLIST_LABEL),
            LabelState::NotLoaded
        );
    }

    /// The defeating table: a valid header, ten valid rows, a `com.apple.*` job
    /// among them, ending in a newline — and cut before the Wenlan row. Every
    /// witness in `launchctl_list_labels` passes, yet the job IS loaded.
    /// Answering `NotLoaded` from it would delete the plist of a live
    /// `KeepAlive` job. The targeted `launchctl list <label>` — which has no
    /// table to truncate — exits 0 for that job, and no bulk-derived absence
    /// may outvote it.
    #[test]
    #[serial_test::serial]
    fn a_truncated_table_can_never_authorize_deleting_a_plist() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        let truncated_but_well_formed = MockLaunchctl {
            unload_status: Mutex::new(1),
            // Cut before the Wenlan row: indistinguishable, from its own
            // shape, from a domain that has no Wenlan job.
            list_stdout: Mutex::new(launchctl_table(&[])),
            // The job is in fact loaded, and the question that cannot be
            // truncated says so. The `com.apple.*` control jobs are in the
            // real domain too, so the apparatus is provably working -- this
            // fixture is a truncated table, not a broken launchctl.
            targeted: Mutex::new(TargetedList::Loaded(loaded_domain(&[SERVER_PLIST_LABEL]))),
            ..Default::default()
        };

        let err = uninstall_server_plist(&truncated_but_well_formed).unwrap_err();
        assert!(
            err.to_string().contains(SERVER_PLIST_LABEL),
            "the error must name the job whose plist was kept: {err}"
        );
        assert!(
            plist.exists(),
            "a well-formed but truncated launchctl table deleted the plist of a loaded KeepAlive \
             job"
        );
        assert!(
            truncated_but_well_formed
                .calls()
                .iter()
                .any(|c| c.len() == 2 && c[0] == "list" && c[1] == SERVER_PLIST_LABEL),
            "the destructive path must ask the question that cannot be truncated; calls were {:?}",
            truncated_but_well_formed.calls()
        );
    }

    /// The bulk table carries the label and the targeted probe did not exit 0.
    ///
    /// The rule that decides this changed in the round-4 follow-up, and the new
    /// answer is the stronger one. A row that IS in the table cannot have been
    /// invented by truncation, so it is a measured POSITIVE; a targeted probe
    /// that did not exit 0 may be a failure. Presence therefore wins, the state
    /// is `Loaded`, and the plist stays. The previous rule called this a
    /// contradiction and returned `Unknown` -- which also kept the file, but
    /// for a reason that treated a measurement and a non-measurement as
    /// equals.
    #[test]
    #[serial_test::serial]
    fn a_table_row_outranks_a_targeted_probe_that_did_not_answer() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        let disagreeing = MockLaunchctl {
            unload_status: Mutex::new(1),
            list_stdout: Mutex::new(launchctl_table(&[SERVER_PLIST_LABEL])),
            targeted: Mutex::new(TargetedList::Always(TARGETED_LIST_NOT_FOUND_FIXTURE)),
            ..Default::default()
        };
        uninstall_server_plist(&disagreeing).unwrap_err();
        assert!(plist.exists());

        assert_eq!(
            LaunchctlReading::take(&disagreeing).label_state(SERVER_PLIST_LABEL),
            LabelState::Loaded,
            "a row that is in the table is a measured positive and outranks a probe that did not \
             answer"
        );
    }

    /// The other side of the rule, so the fix is not "return Unknown forever":
    /// on an honest host -- a whole table, and a targeted probe whose control
    /// answers -- a genuinely stale plist is still deleted. Without this, a
    /// safety change that simply stopped deleting anything would pass every
    /// test above.
    #[test]
    #[serial_test::serial]
    fn a_measured_absence_still_deletes_the_stale_plist() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        let measured_absent = MockLaunchctl {
            unload_status: Mutex::new(1),
            list_stdout: Mutex::new(launchctl_table(&[])),
            // FromTable: the control (`com.apple.*`, in the table) exits 0 and
            // the Wenlan label (not in the table) does not. A working
            // instrument reporting a real absence.
            targeted: Mutex::new(TargetedList::FromTable),
            ..Default::default()
        };
        uninstall_server_plist(&measured_absent).unwrap();
        assert!(
            !plist.exists(),
            "a genuinely stale plist must still be removable"
        );
    }

    /// C1.1, THE REFUTATION. The rule this replaces authorized deleting a
    /// plist when the targeted probe returned `NotPresentOrFailed` and the
    /// bulk table did not carry the label, and called that two agreeing
    /// witnesses.
    ///
    /// Neither of those is an absence. `NotPresentOrFailed` lumps a SPAWN
    /// FAILURE in with "launchd has no such job" -- deliberately, because the
    /// not-found exit code cannot be verified from here -- and a bulk table
    /// that does not carry a label may simply have stopped before its row. So
    /// this fixture, a launchctl whose bulk form works perfectly and whose
    /// targeted form cannot be spawned at all, was enough to delete the plist
    /// of a job that is loaded. It must now establish nothing.
    #[test]
    #[serial_test::serial]
    fn a_targeted_probe_that_could_not_be_spawned_is_never_an_absence() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        let bulk_works_targeted_broken = MockLaunchctl {
            unload_status: Mutex::new(1),
            // A whole, well-formed table that does not carry the label. From
            // its own shape this is indistinguishable from a table cut at a
            // row boundary just before the Wenlan row.
            list_stdout: Mutex::new(launchctl_table(&[])),
            targeted: Mutex::new(TargetedList::SpawnFails),
            ..Default::default()
        };

        assert_eq!(
            LaunchctlReading::take(&bulk_works_targeted_broken).label_state(SERVER_PLIST_LABEL),
            LabelState::Unknown,
            "a targeted probe that never ran is not half of an agreeing absence"
        );

        let err = uninstall_server_plist(&bulk_works_targeted_broken).unwrap_err();
        assert!(
            err.to_string().contains(SERVER_PLIST_LABEL),
            "the error must name the job whose plist was kept: {err}"
        );
        assert!(
            plist.exists(),
            "a transient targeted failure beside a table that could be truncated deleted the \
             plist of a possibly-loaded job"
        );
    }

    /// The control is the whole of the difference between "absent" and "the
    /// instrument said nothing", so it gets its own test with the two fixtures
    /// differing in exactly one thing: whether `launchctl list <a job the
    /// table just listed>` exits 0.
    #[test]
    fn an_absence_is_only_an_absence_when_the_control_answered() {
        // Honest host: the control answers, so the same nonzero exit for our
        // label is a working instrument reporting a real absence.
        let control_answers = MockLaunchctl::default();
        assert_eq!(
            LaunchctlReading::take(&control_answers).label_state(SERVER_PLIST_LABEL),
            LabelState::NotLoaded
        );
        assert!(
            control_answers.calls().iter().any(|c| c.len() == 2
                && c[0] == "list"
                && c[1].starts_with(LAUNCHD_SYSTEM_JOB_PREFIX)),
            "an absence must be backed by a control probe that was actually run; calls were {:?}",
            control_answers.calls()
        );

        // Same table, same nonzero exit for our label -- but now the targeted
        // form also refuses a job the table just said is loaded. The
        // instrument is not answering, so nothing it says means anything.
        let control_silent = MockLaunchctl {
            targeted: Mutex::new(TargetedList::Always(TARGETED_LIST_NOT_FOUND_FIXTURE)),
            ..Default::default()
        };
        assert_eq!(
            LaunchctlReading::take(&control_silent).label_state(SERVER_PLIST_LABEL),
            LabelState::Unknown,
            "a nonzero exit from an instrument that also denies a known-loaded job is not a \
             measurement"
        );
    }

    /// The same unsound bulk negative, spent OUTSIDE the deletion path: a table
    /// cut at a row boundary making `launchd_owns_server_daemon` answer
    /// `DoesNot` sends `daemon_start` down the CLEAN spawn branch — a second
    /// daemon against a job that is in fact loaded, with no
    /// `spawned_on_unknown_owner` latch to show for it.
    #[test]
    #[serial_test::serial]
    fn a_truncated_table_does_not_make_launchd_disown_the_daemon() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        // A plist that DOES target the selected data root, so launchctl is the
        // only thing left that can answer.
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, server_plist_for_data_dir(data.path())).unwrap();
        assert_eq!(server_plist_data_dir_match(), ServerPlistMatch::Matches);

        // The job IS loaded. The table simply stopped before its row.
        let truncated = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[])),
            targeted: Mutex::new(TargetedList::Loaded(loaded_domain(&[SERVER_PLIST_LABEL]))),
            ..Default::default()
        };
        assert_eq!(
            launchd_owns_server_daemon(&truncated),
            LaunchdOwnership::Owns,
            "a table that stopped early is not evidence launchd stopped owning the daemon"
        );

        // And when nothing can answer, the answer is `Unknown` -- the value
        // that reaches `StartupSidecar::SpawnOnUnknownOwner` -- never
        // `DoesNot`.
        let unmeasurable = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[])),
            targeted: Mutex::new(TargetedList::SpawnFails),
            ..Default::default()
        };
        assert_eq!(
            launchd_owns_server_daemon(&unmeasurable),
            LaunchdOwnership::Unknown
        );
        assert_ne!(
            launchd_owns_server_daemon(&unmeasurable),
            LaunchdOwnership::DoesNot,
            "an unmeasurable domain must not read as a measured hand-off to the app's sidecar"
        );
    }

    /// The same table also paints the Settings toggle. "Run at Login: off"
    /// while launchd starts Wenlan every boot is a lie the user acts on -- they
    /// turn it on, and the handover unloads and reloads a job already there.
    #[test]
    #[serial_test::serial]
    fn a_truncated_table_does_not_paint_run_at_login_off() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        std::env::remove_var("WENLAN_DEV_APP_ID");

        let truncated = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[])),
            targeted: Mutex::new(TargetedList::Loaded(loaded_domain(&[
                SERVER_PLIST_LABEL,
                APP_PLIST_LABEL,
            ]))),
            ..Default::default()
        };
        assert_eq!(
            run_at_login_state(&truncated),
            LabelState::Loaded,
            "both jobs are loaded; a table that stopped before their rows must not report the \
             feature off"
        );

        let unmeasurable = MockLaunchctl {
            list_stdout: Mutex::new(launchctl_table(&[])),
            targeted: Mutex::new(TargetedList::SpawnFails),
            ..Default::default()
        };
        assert_eq!(run_at_login_state(&unmeasurable), LabelState::Unknown);
        assert_ne!(
            run_at_login_state(&unmeasurable),
            LabelState::NotLoaded,
            "the toggle must surface the read failure, not paint itself off"
        );
    }

    /// The table parser on its own, including the `-` PID launchd prints for a
    /// job that is registered but not currently running.
    #[test]
    fn launchctl_list_labels_requires_the_whole_table_shape() {
        let table = launchctl_table(&[SERVER_PLIST_LABEL]);
        let labels = launchctl_list_labels(&table).expect("a well-formed table parses");
        assert!(labels.contains(&SERVER_PLIST_LABEL));
        assert_eq!(labels.len(), LAUNCHCTL_MIN_ROWS + 1);

        // One malformed line anywhere poisons the whole table: absence from a
        // table that could not be fully parsed is not absence.
        let mut contaminated = table.clone();
        contaminated.push_str("launchctl: Couldn't stat some path\n");
        assert_eq!(launchctl_list_labels(&contaminated), None);

        // A status column that is not a number is not a row either.
        let mut bad_status = table.clone();
        bad_status.push_str("-\tSTOPPED\tcom.apple.something\n");
        assert_eq!(launchctl_list_labels(&bad_status), None);
    }

    /// The shipped consequence of A1, at the caller that deletes files: an
    /// unload that failed followed by a `launchctl list` that exits 0 with
    /// nothing must keep the plist, exactly as a nonzero `list` already did.
    #[test]
    #[serial_test::serial]
    fn uninstall_server_plist_keeps_the_file_when_launchctl_list_prints_no_table() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, "<plist/>").unwrap();

        let silent = MockLaunchctl {
            unload_status: Mutex::new(1),
            list_stdout: Mutex::new(String::new()),
            ..Default::default()
        };
        let err = uninstall_server_plist(&silent).unwrap_err();
        assert!(
            err.to_string().contains(SERVER_PLIST_LABEL),
            "the error must name the job it could not verify: {err}"
        );
        assert!(
            plist.exists(),
            "an exit-0 `launchctl list` that printed no table is not evidence the job is gone"
        );
    }

    #[test]
    #[serial_test::serial]
    fn first_run_install_cleans_legacy_plists_even_when_user_opted_out() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        // `HOME` for the LaunchAgents paths, `isolate_app_roots` for the
        // opt-out flag: the data-dir env vars stay CLEARED because
        // `first_run_install_if_needed*` treats either of them as an isolated
        // dev run and skips the whole install this test is exercising.
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");

        let legacy_app = legacy_app_plist_path().unwrap();
        let legacy_server = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(legacy_app.parent().unwrap()).unwrap();
        std::fs::write(&legacy_app, owned_legacy_app_plist()).unwrap();
        std::fs::write(&legacy_server, owned_legacy_server_plist()).unwrap();
        set_user_opted_out(true).unwrap();

        let mock = MockLaunchctl::default();
        // The legacy cleanup is reachable only from an installed app path, so
        // the stable-path seam stands in for one here.
        first_run_install_if_needed_at_path(
            &mock,
            Path::new("/Applications/Wenlan.app/Contents/MacOS/wenlan-app"),
        )
        .unwrap();

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
    fn non_stable_first_run_preserves_opted_out_legacy_registrations() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let home = tempfile::tempdir().unwrap();
        // Same split as above: the data-dir overrides must stay clear so
        // `first_run_install_if_needed` does not take its isolated-run exit,
        // so the write root is relocated by the test-only hook instead.
        let _roots = crate::test_env::isolate_app_roots(home.path());
        std::env::set_var("HOME", home.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        set_user_opted_out(true).unwrap();

        let current_exe = current_app_path().unwrap();
        assert_eq!(
            classify_stable_launch_agent_target(&current_exe),
            StableLaunchAgentTarget::Rejected,
            "the test executable must exercise the non-stable startup path"
        );

        let legacy_app = legacy_app_plist_path().unwrap();
        let legacy_server = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(legacy_app.parent().unwrap()).unwrap();
        std::fs::write(&legacy_app, owned_legacy_app_plist()).unwrap();
        std::fs::write(&legacy_server, owned_legacy_server_plist()).unwrap();

        let mock = MockLaunchctl::default();
        first_run_install_if_needed(&mock).unwrap();

        assert!(
            legacy_app.exists(),
            "a non-stable startup must preserve the legacy app registration"
        );
        assert!(
            legacy_server.exists(),
            "a non-stable startup must preserve the legacy server registration"
        );
        assert!(
            mock.calls.lock().unwrap().is_empty(),
            "a non-stable startup must not unload global legacy LaunchAgents"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[serial_test::serial]
    fn isolated_dev_app_reports_run_at_login_disabled_without_querying_launchctl() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        std::env::set_var("WENLAN_DEV_APP_ID", "com.wenlan.desktop.dev.123");
        let mock = MockLaunchctl::default();

        assert_eq!(run_at_login_state(&mock), LabelState::NotLoaded);
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn run_at_login_policy_is_macos_only() {
        assert_eq!(run_at_login_capability("macos"), Ok(()));
        assert_eq!(
            run_at_login_capability("windows"),
            Err(RUN_AT_LOGIN_UNSUPPORTED)
        );
        assert_eq!(
            run_at_login_capability("linux"),
            Err(RUN_AT_LOGIN_UNSUPPORTED)
        );
    }

    #[test]
    fn full_quit_breadcrumb_is_stable() {
        // The quit path logs this verbatim and a Windows lifecycle proof
        // greps for it; changing the string silently blinds that check.
        assert_eq!(FULL_QUIT_BREADCRUMB, "[quit] full quit command accepted");
    }

    #[test]
    fn full_quit_plan_keeps_cross_platform_shutdown_but_limits_launchagents_to_macos() {
        assert_eq!(
            quit_plan_for_target_os("macos"),
            QuitPlan {
                clean_launch_agents: true,
                shutdown_daemon: true,
                exit_app: true,
            }
        );
        assert_eq!(
            quit_plan_for_target_os("windows"),
            QuitPlan {
                clean_launch_agents: false,
                shutdown_daemon: true,
                exit_app: true,
            }
        );
    }

    #[test]
    fn quit_targets_the_selected_daemon_base_url() {
        let client = crate::api::WenlanClient::with_base_url("http://127.0.0.1:17734".to_string());

        assert_eq!(
            shutdown_url_for(&client),
            "http://127.0.0.1:17734/api/shutdown"
        );
    }

    #[test]
    #[serial_test::serial]
    fn first_run_preserves_legacy_plists_when_current_app_path_is_rejected() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        // The data-dir overrides stay cleared so the `wenlan` / `origin` root
        // selection runs for real; `isolate_app_roots` supplies the base it
        // selects from, which is the leg `HOME` cannot reach on Windows.
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        // See `first_run_preserves_legacy_plists_when_current_app_path_is_rejected`.
        let _roots = crate::test_env::isolate_app_roots(tmp.path());
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
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
    fn prepare_server_plist_skips_isolated_override_without_mutation() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let selected_data = tempfile::tempdir().unwrap();
        let live_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", selected_data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let original = server_plist_for_data_dir(live_data.path());
        let plist = server_plist_path().unwrap();
        std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
        std::fs::write(&plist, &original).unwrap();

        let mock = MockLaunchctl::default();
        // This assertion pins both gates together: the env override and the
        // stable-path check. current_exe() cannot be faked to /Applications
        // from a unit test, so removing only the env gate still leaves this
        // test protected by the non-stable-path gate.
        prepare_server_plist_for_startup(&mock).unwrap();

        assert_eq!(std::fs::read_to_string(&plist).unwrap(), original);
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    #[test]
    #[serial_test::serial]
    fn first_run_install_skips_isolated_override_without_touching_plists() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let app_plist = app_plist_path().unwrap();
        let server_plist = server_plist_path().unwrap();
        let legacy_app = legacy_app_plist_path().unwrap();
        let legacy_server = legacy_server_plist_path().unwrap();
        std::fs::create_dir_all(app_plist.parent().unwrap()).unwrap();
        let legacy_app_content = owned_legacy_app_plist();
        let legacy_server_content = owned_legacy_server_plist();
        let originals = [
            (app_plist.as_path(), b"current app".as_slice()),
            (server_plist.as_path(), b"current server".as_slice()),
            (legacy_app.as_path(), legacy_app_content.as_bytes()),
            (legacy_server.as_path(), legacy_server_content.as_bytes()),
        ];
        for (path, content) in originals {
            std::fs::write(path, content).unwrap();
        }
        let original_bytes: Vec<_> = [&app_plist, &server_plist, &legacy_app, &legacy_server]
            .into_iter()
            .map(|path| std::fs::read(path).unwrap())
            .collect();

        let mock = MockLaunchctl::default();
        // This assertion pins both gates together: the env override and the
        // stable-path check. current_exe() cannot be faked to /Applications
        // from a unit test, so removing only the env gate still leaves this
        // test protected by the non-stable-path gate.
        first_run_install_if_needed(&mock).unwrap();

        for (path, original) in [&app_plist, &server_plist, &legacy_app, &legacy_server]
            .into_iter()
            .zip(original_bytes)
        {
            assert_eq!(std::fs::read(path).unwrap(), original);
        }
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn set_run_at_login_rejects_isolated_override_without_launchctl() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        let mock = MockLaunchctl::default();

        let enable_error = set_run_at_login(true, &mock)
            .await
            .expect_err("isolated enable must be rejected");
        let disable_error = set_run_at_login(false, &mock)
            .await
            .expect_err("isolated disable must be rejected");

        assert!(enable_error.to_string().contains("isolated run"));
        assert!(disable_error.to_string().contains("isolated run"));
        assert!(mock.calls.lock().unwrap().is_empty());
    }

    /// `set_run_at_login(true)` stops the app's own sidecar and then registers
    /// launchd against the same port. Registering after a `StillRunning` /
    /// `CouldNotMeasure` stop makes two owners, or a launchd restart loop.
    ///
    /// The rule is pinned here rather than through `set_run_at_login` itself
    /// because that function rejects any binary outside `/Applications/*.app`
    /// (`is_stable_launch_agent_target`) and `current_exe()` cannot be faked
    /// from a unit test, so no unit test can reach the stop at all.
    #[test]
    fn a_handover_is_refused_after_a_stop_that_did_not_confirm_the_daemon_ended() {
        handover_may_proceed(&SidecarStopOutcome::Ended)
            .expect("a measured end is the whole point of the handover");
        handover_may_proceed(&SidecarStopOutcome::NoSidecar)
            .expect("nothing of ours ever held the port");

        let still_running = handover_may_proceed(&SidecarStopOutcome::StillRunning {
            reason: "the process is still there after the shutdown request and the kill"
                .to_string(),
        })
        .expect_err("registering launchd against a held port makes a second owner");
        assert_eq!(
            still_running,
            HandoverRefused::SidecarStillRunning {
                reason: "the process is still there after the shutdown request and the kill"
                    .to_string()
            }
        );

        let unmeasured = handover_may_proceed(&SidecarStopOutcome::CouldNotMeasure {
            reason: "the sidecar's start time was never captured".to_string(),
        })
        .expect_err("'could not measure' must not be spent as if it were 'ended'");
        assert_eq!(
            unmeasured,
            HandoverRefused::SidecarStopUnmeasured {
                reason: "the sidecar's start time was never captured".to_string()
            }
        );

        // What `search::set_run_at_login`'s `Result<(), String>` carries to the
        // Settings row. It has to say what was *not* done, or the user reads a
        // failure as a success.
        for message in [still_running.to_string(), unmeasured.to_string()] {
            assert!(
                message.starts_with("Run at Login was not changed:"),
                "the user must be told the toggle did not take: {message}"
            );
        }
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
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        write_server_plist_for(data.path());

        assert!(current_server_plist_exists());
        assert_eq!(server_plist_data_dir_match(), ServerPlistMatch::Matches);
    }

    // The owner test behind the startup sidecar decision and the "Start
    // Wenlan" button: a server plist that targets the selected data root is
    // only launchd's daemon when launchd has the job loaded. `wenlan
    // background on` writes the file before `launchctl load`, so a failed
    // load leaves a matching file and no job; trusting the file alone would
    // skip the sidecar and leave the user with no daemon.
    #[test]
    #[serial_test::serial]
    fn launchd_owns_the_daemon_only_when_its_job_is_loaded() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        let loaded = MockLaunchctl::default();
        *loaded.list_stdout.lock().unwrap() = launchctl_table(&[SERVER_PLIST_LABEL]);
        let unloaded = MockLaunchctl::default();

        // A loaded job with no plist for the selected data root is not ours.
        assert_eq!(
            launchd_owns_server_daemon(&loaded),
            LaunchdOwnership::DoesNot
        );

        write_server_plist_for(data.path());
        assert_eq!(server_plist_data_dir_match(), ServerPlistMatch::Matches);

        assert_eq!(
            launchd_owns_server_daemon(&unloaded),
            LaunchdOwnership::DoesNot,
            "a matching plist that launchd never loaded must not count as launchd's daemon"
        );
        assert_eq!(launchd_owns_server_daemon(&loaded), LaunchdOwnership::Owns);
    }

    #[test]
    #[serial_test::serial]
    fn current_server_plist_with_stale_data_dir_does_not_count_as_wenlan_service() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let selected_data = tempfile::tempdir().unwrap();
        let stale_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", selected_data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        write_server_plist_for(stale_data.path());

        assert!(
            current_server_plist_exists(),
            "the stale launchd plist file still exists"
        );
        assert_eq!(
            server_plist_data_dir_match(),
            ServerPlistMatch::DoesNotMatch,
            "a stale launchd data root must not suppress the selected-data-dir sidecar fallback \
             — and it must answer DoesNotMatch, a MEASURED negative, not the bare `false` the \
             deleted bool also produced for a plist it could not read at all"
        );
    }

    #[test]
    #[serial_test::serial]
    fn server_plist_data_dir_env_is_patched_and_reloaded() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
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
    fn startup_server_plist_preflight_skips_non_stable_app_path() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let stale_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");

        let plist = write_server_plist_for(stale_data.path());
        let original = std::fs::read(&plist).unwrap();

        let mock = MockLaunchctl::default();
        prepare_server_plist_for_startup(&mock).unwrap();

        assert_eq!(std::fs::read(&plist).unwrap(), original);
        let calls = mock.calls.lock().unwrap();
        assert!(
            calls.is_empty(),
            "non-stable app path must not call launchctl"
        );
    }

    #[test]
    #[serial_test::serial]
    fn server_plist_data_dir_env_rolls_back_file_when_reload_fails() {
        let _env = EnvGuard::capture(LIFECYCLE_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();
        let selected_data = tempfile::tempdir().unwrap();
        let stale_data = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("WENLAN_DATA_DIR", selected_data.path());
        std::env::remove_var("ORIGIN_DATA_DIR");

        let plist = write_server_plist_for(stale_data.path());

        let mock = MockLaunchctl {
            load_status: Mutex::new(1),
            ..Default::default()
        };
        let err = ensure_server_plist_data_dir_env(&mock)
            .expect_err("reload failure must make data-dir patch fail");
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
        assert_eq!(
            server_plist_data_dir_match(),
            ServerPlistMatch::DoesNotMatch
        );
    }

    #[test]
    fn service_management_uses_wenlan_cli_next_to_app_binary() {
        let app_exe = std::path::Path::new("/Applications/Origin.app/Contents/MacOS/origin-app");
        let path = service_cli_path_for_app_exe(app_exe).unwrap();
        let mut expected = app_exe.parent().unwrap().join("wenlan");
        if cfg!(target_os = "windows") {
            expected.set_extension("exe");
        }

        assert_eq!(path, expected);
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
            "wenlan CLI must be bundled because lifecycle service management runs `wenlan background on`"
        );
    }

    /// The app used to send `wenlan install` / `wenlan uninstall`; both verbs
    /// were removed from the CLI, so clap exited 2 and every Run-at-Login
    /// toggle and the Quit teardown failed. See the constant's doc comment for
    /// why this cannot be checked against clap directly from this crate.
    #[test]
    fn background_on_is_the_argv_the_app_sends_to_register_the_daemon() {
        assert_eq!(SERVICE_CLI_BACKGROUND_ON, ["background", "on"]);
        assert!(
            !SERVICE_CLI_BACKGROUND_ON.contains(&"install"),
            "`wenlan install` was removed from the CLI in v0.10"
        );
        assert!(
            !SERVICE_CLI_BACKGROUND_ON.contains(&"uninstall"),
            "`wenlan uninstall` was removed from the CLI in v0.10; deregistering \
             is done in-process by uninstall_server_plist"
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
                status: exit_status(0),
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
