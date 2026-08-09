// SPDX-License-Identifier: AGPL-3.0-only
//! App-local config file I/O. Reads the shared Wenlan config path, with legacy
//! Origin path fallback, during app startup so local sensors have bootstrap
//! state before the UI talks to the daemon. Settings writes that affect daemon
//! config should go through the daemon HTTP client, then mirror successful
//! values into app-local runtime state when the process needs them immediately.
//! Remaining local compatibility writes preserve daemon-only JSON fields.
//!
//! Copied from origin-core::config; uses AppError instead of OriginError.
use crate::error::AppError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use wenlan_types::sources::{Source, SourceType, SyncStatus};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Legacy field — kept for backward compat with old config files.
    /// Use `sources` instead. Migrated to Source structs by `migrate()`.
    #[serde(default)]
    pub watch_paths: Vec<PathBuf>,
    #[serde(default)]
    pub sources: Vec<Source>,
    #[serde(default)]
    pub knowledge_path: Option<PathBuf>,
    #[serde(default)]
    pub setup_completed: bool,
    #[serde(default)]
    pub anthropic_api_key: Option<String>,
    #[serde(default)]
    pub routine_model: Option<String>,
    #[serde(default)]
    pub synthesis_model: Option<String>,
    #[serde(default)]
    pub remote_access_enabled: bool,
    #[serde(default)]
    pub on_device_model: Option<String>,
    #[serde(default)]
    pub external_llm_endpoint: Option<String>,
    #[serde(default)]
    pub external_llm_model: Option<String>,
}

/// Generate a source ID slug from a directory path (last component, lowercased, sanitized).
fn slug_from_path(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| "dir".to_string())
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "-")
}

impl Config {
    /// Migrate legacy `watch_paths` entries into `sources` vec.
    /// Idempotent — only converts paths not already represented in `sources`.
    pub fn migrate(&mut self) {
        if self.watch_paths.is_empty() {
            return;
        }
        let existing_paths: std::collections::HashSet<PathBuf> =
            self.sources.iter().map(|s| s.path.clone()).collect();

        for path in &self.watch_paths {
            if existing_paths.contains(path) {
                continue;
            }
            let slug = slug_from_path(path);
            self.sources.push(Source {
                id: format!("dir-{}", slug),
                source_type: SourceType::Directory,
                path: path.clone(),
                status: SyncStatus::Active,
                last_sync: None,
                file_count: 0,
                memory_count: 0,
                last_sync_errors: 0,
                last_sync_error_detail: None,
            });
        }
        // Clear legacy field so it doesn't re-migrate on next load
        self.watch_paths.clear();
    }

    /// Returns the configured knowledge path, or the product default.
    /// Existing `~/Origin/knowledge` directories remain readable during rename.
    pub fn knowledge_path_or_default(&self) -> PathBuf {
        if let Some(path) = self.knowledge_path.clone() {
            return path;
        }
        let current = default_knowledge_path();
        let legacy = legacy_knowledge_path();
        if !current.exists() && legacy.exists() {
            return legacy;
        }
        current
    }

    /// Returns paths for all active Directory-type sources (for indexer compat).
    pub fn directory_source_paths(&self) -> Vec<PathBuf> {
        self.sources
            .iter()
            .filter(|s| s.source_type == SourceType::Directory)
            .filter(|s| matches!(s.status, SyncStatus::Active))
            .map(|s| s.path.clone())
            .collect()
    }
}

fn config_path() -> PathBuf {
    crate::identity_paths::app_data_dir().join("config.json")
}

fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

fn default_knowledge_path() -> PathBuf {
    home_dir().join("Wenlan/knowledge")
}

fn legacy_knowledge_path() -> PathBuf {
    home_dir().join("Origin/knowledge")
}

pub fn load_config() -> Config {
    let path = config_path();
    let mut config = match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => Config::default(),
    };
    config.migrate();
    config
}

pub fn save_config(config: &Config) -> Result<(), AppError> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let value = merge_with_existing_json(&path, serde_json::to_value(config)?);
    let json = serde_json::to_string_pretty(&value)?;
    std::fs::write(&path, &json)?;

    // Restrict file permissions when API key is present (user-only read/write)
    #[cfg(unix)]
    if config.anthropic_api_key.is_some() {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&path, perms).ok();
    }

    Ok(())
}

fn merge_with_existing_json(path: &std::path::Path, next: Value) -> Value {
    let Value::Object(next) = next else {
        return next;
    };

    let mut merged = std::fs::read_to_string(path)
        .ok()
        .and_then(|contents| serde_json::from_str::<Value>(&contents).ok())
        .and_then(|value| match value {
            Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default();

    for (key, value) in next {
        merged.insert(key, value);
    }

    Value::Object(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn config_path_prefers_wenlan_data_dir() {
        let _env = EnvGuard::capture();
        std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-config-test");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-config-test");

        assert_eq!(
            config_path(),
            PathBuf::from("/tmp/wenlan-config-test/config.json")
        );
    }

    #[test]
    #[serial_test::serial]
    fn config_path_falls_back_to_origin_data_dir() {
        let _env = EnvGuard::capture();
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-config-test");

        assert_eq!(
            config_path(),
            PathBuf::from("/tmp/origin-config-test/config.json")
        );
    }

    #[test]
    fn test_config_default_values() {
        let config = Config::default();
        assert!(config.watch_paths.is_empty());
    }

    #[test]
    fn test_config_roundtrip_serde() {
        let mut config = Config {
            setup_completed: false,
            anthropic_api_key: None,
            remote_access_enabled: false,
            ..Config::default()
        };
        config.watch_paths = vec![PathBuf::from("/tmp/test")];
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.watch_paths, config.watch_paths);
    }

    #[test]
    fn test_config_deserialize_missing_fields_uses_defaults() {
        let json = r#"{"watch_paths": ["/tmp/a"]}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.watch_paths, vec![PathBuf::from("/tmp/a")]);
    }

    #[test]
    fn test_config_deserialize_empty_json() {
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(config.watch_paths.is_empty());
    }

    // --- save_config / load_config I/O roundtrip ---

    #[test]
    #[serial_test::serial]
    fn save_load_config_roundtrip() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        // Point config_path() at our temp dir via the env override.
        // Env mutation is process-wide, so this must not race with other tests
        // that read or write ORIGIN_DATA_DIR.
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", tmp.path());
        let config = Config {
            watch_paths: vec![PathBuf::from("/test/path")],
            ..Config::default()
        };
        save_config(&config).unwrap();
        let loaded = load_config();
        // After load_config, migrate() runs: watch_paths -> sources, watch_paths cleared.
        assert_eq!(loaded.sources.len(), 1);
        assert_eq!(loaded.sources[0].path, PathBuf::from("/test/path"));
        assert!(loaded.watch_paths.is_empty());
    }

    #[test]
    #[serial_test::serial]
    fn save_config_preserves_daemon_only_fields() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", tmp.path());
        let path = config_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            r#"{"private_browsing_detection":true,"reranker_mode":"hybrid","future_flag":{"enabled":true}}"#,
        )
        .unwrap();

        let mut config = load_config();
        config.setup_completed = true;
        save_config(&config).unwrap();

        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(saved["private_browsing_detection"], true);
        assert_eq!(saved["reranker_mode"], "hybrid");
        assert_eq!(saved["future_flag"], serde_json::json!({ "enabled": true }));
        assert_eq!(saved["setup_completed"], true);
    }

    // --- setup_completed ---

    #[test]
    fn test_setup_completed_defaults_to_false() {
        let config = Config::default();
        assert!(!config.setup_completed);
    }

    #[test]
    fn test_setup_completed_roundtrip() {
        let config = Config {
            setup_completed: true,
            ..Config::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert!(restored.setup_completed);
    }

    #[test]
    fn test_setup_completed_missing_in_json_defaults_false() {
        let json = r#"{"clipboard_enabled": true}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(!config.setup_completed);
    }

    // --- remote_access_enabled ---

    #[test]
    fn test_remote_access_enabled_defaults_to_false() {
        let config = Config::default();
        assert!(!config.remote_access_enabled);
    }

    #[test]
    fn test_remote_access_enabled_roundtrip() {
        let config = Config {
            remote_access_enabled: true,
            ..Config::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert!(restored.remote_access_enabled);
    }

    #[test]
    fn test_remote_access_enabled_missing_in_json_defaults_false() {
        let json = r#"{"clipboard_enabled": true}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(!config.remote_access_enabled);
    }

    // --- migrate() / watch_paths / sources / knowledge_path ---

    #[test]
    fn test_config_defaults_empty_sources() {
        let config: Config = serde_json::from_str("{}").unwrap();
        let mut config = config;
        config.migrate();
        assert!(config.sources.is_empty());
        assert!(config.knowledge_path.is_none());
    }

    #[test]
    fn config_watch_paths_migration() {
        let old_json = r#"{
            "watch_paths": ["/Users/x/docs", "/Users/x/notes"],
            "clipboard_enabled": false
        }"#;
        let mut config: Config = serde_json::from_str(old_json).unwrap();
        config.migrate();
        assert_eq!(config.sources.len(), 2);
        assert_eq!(config.sources[0].source_type, SourceType::Directory);
        assert_eq!(config.sources[0].path, PathBuf::from("/Users/x/docs"));
        assert_eq!(config.sources[1].path, PathBuf::from("/Users/x/notes"));
        // Legacy field cleared after migration.
        assert!(config.watch_paths.is_empty());
    }

    #[test]
    #[serial_test::serial]
    fn config_knowledge_path_default_uses_wenlan_when_no_legacy_exists() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let config: Config = serde_json::from_str("{}").unwrap();
        let default_path = tmp.path().join("Wenlan/knowledge");
        assert_eq!(config.knowledge_path_or_default(), default_path);
    }

    #[test]
    #[serial_test::serial]
    fn config_knowledge_path_default_uses_legacy_when_only_legacy_exists() {
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        let legacy = tmp.path().join("Origin/knowledge");
        std::fs::create_dir_all(&legacy).unwrap();
        let config: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(config.knowledge_path_or_default(), legacy);
    }

    #[test]
    fn config_knowledge_path_custom() {
        let json = r#"{"knowledge_path": "/my/custom/path"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.knowledge_path_or_default(),
            PathBuf::from("/my/custom/path")
        );
    }

    #[test]
    fn directory_source_paths() {
        let json = r#"{"sources": [
            {"id": "d1", "source_type": "directory", "path": "/a", "status": "Active", "last_sync": null, "file_count": 0, "memory_count": 0},
            {"id": "o1", "source_type": "obsidian", "path": "/b", "status": "Active", "last_sync": null, "file_count": 0, "memory_count": 0}
        ]}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let paths = config.directory_source_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], PathBuf::from("/a"));
    }

    // --- unknown-field tolerance ---

    #[test]
    fn dwell_enabled_alias() {
        // dwell_enabled was removed with ambient capture; verify unknown fields are ignored.
        let json = r#"{"dwell_enabled": true}"#;
        let _config: Config = serde_json::from_str(json).unwrap();
    }
}
