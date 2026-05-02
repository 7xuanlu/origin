// SPDX-License-Identifier: AGPL-3.0-only
// Items in this module are used by later tasks (Tasks 6-16). Allow dead-code
// until they are wired up.
#![allow(dead_code)]
use anyhow::{Context, Result};
use std::io;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tauri::AppHandle;

/// Process-wide guard that prevents `quit_origin` from running twice. Set on
/// first entry; never cleared (the process is exiting).
static QUITTING: AtomicBool = AtomicBool::new(false);

pub const SERVER_PLIST_LABEL: &str = "com.origin.server";
pub const APP_PLIST_LABEL: &str = "com.origin.desktop";

const APP_PLIST_TEMPLATE: &str = include_str!("../resources/com.origin.desktop.plist");

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

/// Resolve the data directory for the auto-start flag. Honors `ORIGIN_DATA_DIR`
/// for isolated testing per CLAUDE.md.
fn data_dir() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("ORIGIN_DATA_DIR") {
        return Ok(PathBuf::from(custom));
    }
    Ok(dirs::data_dir().context("HOME not set")?.join("origin"))
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

pub fn server_plist_path() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", SERVER_PLIST_LABEL)))
}

fn log_dir() -> Result<PathBuf> {
    Ok(dirs::data_local_dir()
        .context("data_local_dir unavailable")?
        .join("origin")
        .join("logs"))
}

fn current_app_path() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    std::fs::canonicalize(&exe).context("canonicalize current_exe")
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
        .replace("__ORIGIN_APP_PATH__", &app_path.to_string_lossy())
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

/// Run `origin-server install`. Resolves the binary alongside our exe.
pub fn install_server_plist_via_subprocess() -> Result<()> {
    let bin = current_app_path()?
        .parent()
        .context("no parent dir")?
        .join("origin-server");
    let out = Command::new(&bin).arg("install").output()?;
    if !out.status.success() {
        anyhow::bail!(
            "origin-server install failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

pub fn uninstall_server_plist_via_subprocess() -> Result<()> {
    let bin = current_app_path()?
        .parent()
        .context("no parent dir")?
        .join("origin-server");
    let out = Command::new(&bin).arg("uninstall").output()?;
    if !out.status.success() {
        anyhow::bail!(
            "origin-server uninstall failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// Returns true iff BOTH plists are loaded in launchctl.
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
        return Ok(());
    }

    let exe_canonical = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::canonicalize(&p).ok());

    let app_plist_stale = app_plist_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .map(|content| {
            if let Some(exe) = &exe_canonical {
                !content.contains(exe.to_string_lossy().as_ref())
            } else {
                false
            }
        })
        .unwrap_or(true); // missing plist = stale

    let server_plist_stale = server_plist_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .map(|content| {
            if let Some(exe) = &exe_canonical {
                let expected_server = exe
                    .parent()
                    .map(|p| p.join("origin-server").to_string_lossy().to_string())
                    .unwrap_or_default();
                !content.contains(&expected_server)
            } else {
                false
            }
        })
        .unwrap_or(true);

    if !app_plist_stale && !server_plist_stale {
        return Ok(());
    }

    if let Err(e) = install_server_plist_via_subprocess() {
        log::warn!("[first-run] origin-server install failed: {e}");
    }
    install_app_plist(launchctl)?;
    log::info!("[first-run] LaunchAgents installed");
    Ok(())
}

/// Toggle "Run at login". Mutex-guarded at the caller via app state.
pub async fn set_run_at_login(enabled: bool, launchctl: &dyn LaunchctlExec) -> Result<()> {
    if enabled {
        set_user_opted_out(false)?;
        install_server_plist_via_subprocess()?;
        install_app_plist(launchctl)?;
    } else {
        set_user_opted_out(true)?;
        uninstall_app_plist(launchctl)?;
        uninstall_server_plist_via_subprocess()?;
    }
    Ok(())
}

pub async fn quit_origin(app_handle: &AppHandle) -> Result<()> {
    // Debounce: tray menu Quit Origin item stays clickable during the 500ms
    // shutdown sleep; double-click would otherwise spawn 2× POSTs (H1).
    if QUITTING.swap(true, Ordering::AcqRel) {
        return Ok(());
    }

    // Spec lifecycle invariant #4: "Quit Origin = full off; both plists
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

    #[derive(Default)]
    struct MockLaunchctl {
        calls: Mutex<Vec<Vec<String>>>,
        /// Status code to return for load/start subcommands. Default 0 = ok.
        load_status: Mutex<i32>,
    }
    impl LaunchctlExec for MockLaunchctl {
        fn run(&self, args: &[&str]) -> io::Result<Output> {
            self.calls
                .lock()
                .unwrap()
                .push(args.iter().map(|s| s.to_string()).collect());
            // Only `load` honors the configurable status — other subcommands
            // (unload, list) always succeed so test setup is simpler.
            let status_code = if args.first().copied() == Some("load") {
                *self.load_status.lock().unwrap()
            } else {
                0
            };
            Ok(Output {
                status: std::process::ExitStatus::from_raw(status_code),
                stdout: vec![],
                stderr: vec![],
            })
        }
    }

    // Tests that mutate `HOME` env var must run serially — std::env::set_var is
    // !Sync (Rust 2024 will mark it unsafe). #[serial] forces these to one-at-a-time.

    #[test]
    #[serial_test::serial]
    fn opt_out_flag_round_trip() {
        // Override HOME so dirs::data_dir() returns the tempdir on macOS.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
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
        // The opt-out sentinel must NOT live inside the daemon's typed
        // `config.json` — otherwise unrelated `Config::save` calls overwrite
        // the file and silently drop the user's opt-out preference (C1).
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
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
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("ORIGIN_DATA_DIR", tmp.path());

        assert!(!user_opted_out());
        set_user_opted_out(true).unwrap();
        assert!(tmp.path().join("auto_start_disabled.flag").exists());
        assert!(user_opted_out());

        std::env::remove_var("ORIGIN_DATA_DIR");
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
            .join("Library/LaunchAgents/com.origin.desktop.plist");
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
            .join("Library/LaunchAgents/com.origin.desktop.plist");
        assert!(plist.exists(), "plist file written");
        let content = std::fs::read_to_string(&plist).unwrap();
        assert!(content.contains("<key>Label</key>"));
        assert!(
            !content.contains("__ORIGIN_APP_PATH__"),
            "placeholder substituted"
        );

        let calls = mock.calls.lock().unwrap();
        assert!(calls.iter().any(|c| c[0] == "load"));
    }

    #[test]
    #[serial_test::serial]
    fn uninstall_app_plist_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let plist_dir = tmp.path().join("Library/LaunchAgents");
        std::fs::create_dir_all(&plist_dir).unwrap();
        let plist = plist_dir.join("com.origin.desktop.plist");
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
            "123\t0\t{}\n456\t0\t{}\n",
            SERVER_PLIST_LABEL, APP_PLIST_LABEL
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
        // H1: tray menu Quit Origin item stays clickable during the 500ms
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
    #[serial_test::serial]
    fn uninstall_app_plist_is_idempotent_when_file_absent() {
        // H2: Quit Origin calls uninstall_app_plist; the sequence must be
        // idempotent because the plist may already have been removed by an
        // earlier toggle.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let mock = MockLaunchctl::default();
        // No file present → succeed without error.
        uninstall_app_plist(&mock).unwrap();
    }
}
