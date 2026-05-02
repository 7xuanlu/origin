// SPDX-License-Identifier: AGPL-3.0-only
// Items in this module are used by later tasks (Tasks 6-16). Allow dead-code
// until they are wired up.
#![allow(dead_code)]
use anyhow::{Context, Result};
use std::io;
use std::path::PathBuf;
use std::process::{Command, Output};

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

    #[test]
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
}
