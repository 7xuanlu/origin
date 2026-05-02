// SPDX-License-Identifier: AGPL-3.0-only
// Items in this module are used by later tasks (Tasks 6-16). Allow dead-code
// until they are wired up.
#![allow(dead_code)]
use anyhow::{Context, Result};
use std::io;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::Duration;
use tauri::AppHandle;

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

/// Path to the user-opted-out flag in config.json.
fn config_path() -> Result<PathBuf> {
    Ok(dirs::data_dir()
        .context("HOME not set")?
        .join("origin")
        .join("config.json"))
}

/// Read auto_start_disabled flag from config.json. False if file/key missing.
pub fn user_opted_out() -> bool {
    let path = match config_path() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let json: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return false,
    };
    json.get("auto_start_disabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Write auto_start_disabled flag to config.json. Creates file if absent.
pub fn set_user_opted_out(opted_out: bool) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json: serde_json::Value = if path.exists() {
        let raw = std::fs::read_to_string(&path)?;
        serde_json::from_str(&raw).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    json["auto_start_disabled"] = serde_json::Value::Bool(opted_out);
    std::fs::write(&path, serde_json::to_string_pretty(&json)?)?;
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

    let out = launchctl.run(&["load", &plist.to_string_lossy()])?;
    if !out.status.success() {
        anyhow::bail!(
            "launchctl load failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
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
pub fn is_run_at_login_enabled(launchctl: &dyn LaunchctlExec) -> bool {
    let out = match launchctl.run(&["list"]) {
        Ok(o) => o,
        Err(_) => return false,
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let server = stdout.lines().any(|l| l.contains(SERVER_PLIST_LABEL));
    let app = stdout.lines().any(|l| l.contains(APP_PLIST_LABEL));
    server && app
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

    // 3. Tauri-graceful exit. KeepAlive.SuccessfulExit=false means
    //    launchd does NOT respawn after clean exit.
    app_handle.exit(0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockLaunchctl {
        calls: Mutex<Vec<Vec<String>>>,
        next_status: Mutex<i32>,
    }
    impl LaunchctlExec for MockLaunchctl {
        fn run(&self, args: &[&str]) -> io::Result<Output> {
            self.calls
                .lock()
                .unwrap()
                .push(args.iter().map(|s| s.to_string()).collect());
            Ok(Output {
                status: std::process::ExitStatus::from_raw(*self.next_status.lock().unwrap()),
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
}
