// SPDX-License-Identifier: AGPL-3.0-only
//! App-local activity tracking (macOS focus event ring buffer).
//! Copied from origin-core; kept here because activities are an in-process
//! sensor record, never served through the daemon.
use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const ACTIVITY_GAP_SECS: i64 = 1800; // 30-min inactivity → new activity

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    pub id: String,
    pub started_at: i64,
    pub ended_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySummary {
    pub id: String,
    pub started_at: i64,
    pub ended_at: i64,
    pub is_live: bool,
    pub app_names: Vec<String>,
}

impl Activity {
    pub fn new(now: i64) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            started_at: now,
            ended_at: now,
        }
    }

    pub fn new_with_range(started_at: i64, ended_at: i64) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            started_at,
            ended_at,
        }
    }

    pub fn to_summary(&self, is_live: bool) -> ActivitySummary {
        ActivitySummary {
            id: self.id.clone(),
            started_at: self.started_at,
            ended_at: self.ended_at,
            is_live,
            app_names: Vec::new(),
        }
    }
}

fn activities_path() -> PathBuf {
    crate::identity_paths::app_data_dir().join("activities.json")
}

pub fn load_activities() -> Vec<Activity> {
    let path = activities_path();
    match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

pub fn save_activities(activities: &[Activity]) -> Result<(), AppError> {
    let path = activities_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(activities)?;
    std::fs::write(&path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

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

    #[test]
    #[serial_test::serial]
    fn activities_path_prefers_wenlan_data_dir() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let current = tempfile::tempdir().unwrap();
        let legacy = tempfile::tempdir().unwrap();
        std::env::set_var("WENLAN_DATA_DIR", current.path());
        std::env::set_var("ORIGIN_DATA_DIR", legacy.path());

        save_activities(&[Activity::new_with_range(1, 2)]).unwrap();

        assert!(current.path().join("activities.json").exists());
        assert!(!legacy.path().join("activities.json").exists());
    }
}
