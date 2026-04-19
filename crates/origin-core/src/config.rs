// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use crate::sources::{Source, SourceType, SyncStatus};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_true() -> bool {
    true
}

fn default_skip_apps() -> Vec<String> {
    vec![
        "Window Server".into(),
        "Dock".into(),
        "SystemUIServer".into(),
        "Control Center".into(),
        "Notification Center".into(),
        "loginwindow".into(),
        "Spotlight".into(),
        "Origin".into(),
        "1Password".into(),
        "Keychain Access".into(),
        "LastPass".into(),
        "Bitwarden".into(),
        "Dashlane".into(),
        "KeePass".into(),
    ]
}

fn default_skip_title_patterns() -> Vec<String> {
    vec![]
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    pub clipboard_enabled: bool,
    #[serde(default = "default_skip_apps")]
    pub skip_apps: Vec<String>,
    #[serde(default = "default_skip_title_patterns")]
    pub skip_title_patterns: Vec<String>,
    #[serde(default = "default_true")]
    pub private_browsing_detection: bool,
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
    pub screen_capture_enabled: bool,
    #[serde(default)]
    pub selection_capture_enabled: bool,
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

    /// Returns the configured knowledge path, or `~/Origin/knowledge/` as default.
    pub fn knowledge_path_or_default(&self) -> PathBuf {
        self.knowledge_path.clone().unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("Origin/knowledge")
        })
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
    // Honor the `ORIGIN_DATA_DIR` override so a scratch daemon (e.g.
    // `origin-server --data-dir /tmp/origin-demo`) reads and writes its own
    // config file rather than clobbering the user's real one.
    let root = std::env::var_os("ORIGIN_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("origin")
        });
    root.join("config.json")
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

pub fn save_config(config: &Config) -> Result<(), OriginError> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(config)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let config = Config::default();
        assert!(config.watch_paths.is_empty());
        assert!(!config.clipboard_enabled);
    }

    #[test]
    fn test_config_roundtrip_serde() {
        let mut config = Config {
            clipboard_enabled: true,
            skip_apps: vec!["TestApp".into()],
            skip_title_patterns: vec!["secret*".into()],
            private_browsing_detection: false,
            setup_completed: false,
            anthropic_api_key: None,
            remote_access_enabled: false,
            screen_capture_enabled: false,
            selection_capture_enabled: false,
            ..Config::default()
        };
        config.watch_paths = vec![PathBuf::from("/tmp/test")];
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.watch_paths, config.watch_paths);
        assert!(restored.clipboard_enabled);
        assert_eq!(restored.skip_apps, vec!["TestApp".to_string()]);
        assert!(!restored.private_browsing_detection);
    }

    #[test]
    fn test_config_deserialize_missing_fields_uses_defaults() {
        let json = r#"{"watch_paths": ["/tmp/a"]}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.watch_paths, vec![PathBuf::from("/tmp/a")]);
        assert!(config.private_browsing_detection);
        assert!(!config.skip_apps.is_empty());
    }

    #[test]
    fn test_config_deserialize_empty_json() {
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(config.watch_paths.is_empty());
    }

    #[test]
    fn test_save_load_config_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let config_file = tmp.path().join("config.json");
        let mut config = Config {
            clipboard_enabled: true,
            ..Config::default()
        };
        config.watch_paths = vec![PathBuf::from("/test/path")];
        let json = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&config_file, &json).unwrap();
        let contents = std::fs::read_to_string(&config_file).unwrap();
        let restored: Config = serde_json::from_str(&contents).unwrap();
        assert_eq!(restored.watch_paths, vec![PathBuf::from("/test/path")]);
        assert!(restored.clipboard_enabled);
    }

    #[test]
    fn test_dwell_enabled_alias() {
        // dwell_enabled was removed with ambient capture; verify unknown fields are ignored
        let json = r#"{"dwell_enabled": true}"#;
        let _config: Config = serde_json::from_str(json).unwrap();
    }

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

    #[test]
    fn test_screen_capture_enabled_defaults_to_false() {
        let config = Config::default();
        assert!(!config.screen_capture_enabled);
    }

    #[test]
    fn test_screen_capture_enabled_roundtrip() {
        let config = Config {
            screen_capture_enabled: true,
            ..Config::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert!(restored.screen_capture_enabled);
    }

    #[test]
    fn test_screen_capture_enabled_missing_in_json_defaults_false() {
        let json = r#"{"clipboard_enabled": true}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(!config.screen_capture_enabled);
    }

    #[test]
    fn config_model_choice_defaults_none() {
        let config = Config::default();
        assert!(config.routine_model.is_none());
        assert!(config.synthesis_model.is_none());
    }

    #[test]
    fn config_model_choice_roundtrip() {
        let json = r#"{
            "anthropic_api_key": "sk-test",
            "routine_model": "claude-haiku-4-5-20251001",
            "synthesis_model": "claude-opus-4-6"
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.routine_model.as_deref(),
            Some("claude-haiku-4-5-20251001")
        );
        assert_eq!(config.synthesis_model.as_deref(), Some("claude-opus-4-6"));
    }

    #[test]
    fn config_backward_compat_without_model_fields() {
        let json = r#"{"anthropic_api_key": "sk-test"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(config.routine_model.is_none());
        assert!(config.synthesis_model.is_none());
    }

    // --- New sources/knowledge_path tests ---

    #[test]
    fn test_config_defaults_empty_sources() {
        let config: Config = serde_json::from_str("{}").unwrap();
        let mut config = config;
        config.migrate();
        assert!(config.sources.is_empty());
        assert!(config.knowledge_path.is_none());
    }

    #[test]
    fn test_config_watch_paths_migration() {
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
    }

    #[test]
    fn test_config_knowledge_path_default() {
        let config: Config = serde_json::from_str("{}").unwrap();
        let default_path = dirs::home_dir().unwrap().join("Origin/knowledge");
        assert_eq!(config.knowledge_path_or_default(), default_path);
    }

    #[test]
    fn test_config_knowledge_path_custom() {
        let json = r#"{"knowledge_path": "/my/custom/path"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.knowledge_path_or_default(),
            PathBuf::from("/my/custom/path")
        );
    }

    #[test]
    fn test_directory_source_paths() {
        let json = r#"{"sources": [
            {"id": "d1", "source_type": "directory", "path": "/a", "status": "Active", "last_sync": null, "file_count": 0, "memory_count": 0},
            {"id": "o1", "source_type": "obsidian", "path": "/b", "status": "Active", "last_sync": null, "file_count": 0, "memory_count": 0}
        ]}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        let paths = config.directory_source_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], PathBuf::from("/a"));
    }
}
