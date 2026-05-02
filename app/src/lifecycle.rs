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

#[cfg(test)]
mod tests {
    use super::*;

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
}
