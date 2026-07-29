// SPDX-License-Identifier: Apache-2.0
use crate::error::WenlanError;
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
        "Wenlan".into(),
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
    pub on_device_model: Option<String>,
    #[serde(default)]
    pub external_llm_endpoint: Option<String>,
    #[serde(default)]
    pub external_llm_model: Option<String>,
    /// Bearer key for the external OpenAI-compatible endpoint. Never returned
    /// by any API response — see the key-lifecycle contract in the design spec.
    #[serde(default)]
    pub external_llm_api_key: Option<String>,
    /// Persistent cross-encoder reranker mode (`off`/`lite`/`full`). Daemon-read
    /// at startup via `reranker_mode_resolved`; the `WENLAN_RERANKER_MODE` env
    /// var overrides it. Set with `wenlan models reranker <mode>`.
    #[serde(default)]
    pub reranker_mode: Option<String>,
    /// Optional per-job routing pin for everyday work (recap, extraction, bulk
    /// enrich): `"anthropic"` | `"external"` | `"on_device"`. Selects the SOURCE
    /// only; the model comes from that source's own knobs. Absent disables
    /// model-backed background work for this job class.
    #[serde(default)]
    pub everyday_source: Option<String>,
    /// Optional per-job routing pin for synthesis (page distillation):
    /// `"anthropic"` | `"external"` | `"on_device"`. Selects the SOURCE only.
    /// Absent disables model-backed background work for this job class.
    #[serde(default)]
    pub synthesis_source: Option<String>,
    /// Gates ONLY the proactive Page-Map suggestion phase in the refinery
    /// scheduler (`Phase::PageMaps`). The explicit `POST /api/pages/{id}/map/improve`
    /// route is NEVER gated by this flag.
    ///
    /// DEFAULT FALSE since 2026-07-19: a Canvas that fills itself with a dozen
    /// `suggested` boxes before the user has drawn anything reads as clutter to
    /// triage, not as help — the user asked to start from blank and reach for
    /// suggestions deliberately via the Improve button. Opt back in by setting
    /// `page_map_auto_suggest: true` in config.json (or via `PUT /api/config`).
    #[serde(default)]
    pub page_map_auto_suggest: bool,
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

    /// Returns the configured pages path, or `~/.wenlan/pages/` as default.
    /// Field name remains `knowledge_path` for back-compat with existing
    /// config files; the default folder is rebranded to `pages/` to match
    /// the user-facing `Page` concept and the consolidated `~/.wenlan/`
    /// data layout.
    pub fn knowledge_path_or_default(&self) -> PathBuf {
        self.knowledge_path.clone().unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".wenlan/pages")
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

/// The real config root: the platform data dir.
#[cfg(not(any(test, feature = "scratch-config")))]
fn default_config_root() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("wenlan")
}

/// A disposable per-process root, used when the crate is compiled for tests.
///
/// ~25 tests call `save_config`, and what keeps them off the developer's real
/// `config.json` today is a *process-global env var*: seven hand-rolled copies
/// of a `DataDirGuard` set `WENLAN_DATA_DIR` to a tempdir and remove it on drop.
/// That is a race, not a guarantee — the guard's `remove_var` on one thread can
/// land between another thread's `set_var` and its `save_config`, and then the
/// write goes to the real file. `Config::default()` carries no API key and no
/// reranker mode, so losing that race wipes both. Two holes are already open:
/// `page_map_routes.rs` uses a guard without taking `TEST_DATA_DIR_LOCK`, and
/// `spaces.rs` sets the same var raw, which `temp_env`'s lock does not see.
///
/// Redirecting the *default* root closes it at the one place that resolves it,
/// so the env guards become a convenience rather than the safety mechanism.
/// `WENLAN_DATA_DIR` still wins when set; the rest share one temp dir per
/// process, which is the same save-then-load semantics they had before.
#[cfg(any(test, feature = "scratch-config"))]
fn default_config_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        // Under `cfg(test)` this is the point. Reached from a *binary*, it means
        // the binary was built with the dev-dependency feature on (`cargo test`
        // leaves such a `target/debug/wenlan-server` behind), so say so loudly:
        // the config it reads is empty, not lost.
        #[cfg(not(test))]
        eprintln!(
            "[wenlan] WARNING: built with wenlan-core/scratch-config — config \
             reads and writes go to a temporary directory, not the real data \
             dir. This is a test-only feature; rebuild without it."
        );
        let root =
            std::env::temp_dir().join(format!("wenlan-scratch-config-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&root);
        root
    })
    .clone()
}

fn config_path() -> PathBuf {
    // Honor the `WENLAN_DATA_DIR` override so a scratch daemon (e.g.
    // `wenlan-server --data-dir /tmp/wenlan-demo`) reads and writes its own
    // config file rather than clobbering the user's real one.
    let root = crate::env_compat::var_compat("WENLAN_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_config_root);
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

/// True when the config holds any credential — used to tighten file perms.
#[cfg_attr(not(unix), allow(dead_code))]
fn stores_secret(config: &Config) -> bool {
    config.anthropic_api_key.is_some() || config.external_llm_api_key.is_some()
}

pub fn save_config(config: &Config) -> Result<(), WenlanError> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(&path, &json)?;

    // Restrict file permissions when any credential is present (user-only read/write)
    #[cfg(unix)]
    if stores_secret(config) {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&path, perms).ok();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test binary may never resolve the developer's real `config.json`.
    /// ~25 test call sites reach `save_config`, which writes `Config::default()`
    /// — no API key, no reranker mode — so a test that lost the env-var race
    /// described on `default_config_root` wiped both. Unsets the legacy
    /// `ORIGIN_DATA_DIR` too: `var_compat` falls back to it, so leaving it set
    /// would mask the real default.
    #[test]
    fn config_path_never_resolves_the_real_data_dir_in_tests() {
        temp_env::with_vars(
            [
                ("WENLAN_DATA_DIR", None::<&str>),
                ("ORIGIN_DATA_DIR", None::<&str>),
            ],
            || {
                let real = dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("wenlan")
                    .join("config.json");
                assert_ne!(
                    config_path(),
                    real,
                    "test builds must not read or write the real config"
                );
            },
        );
    }

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

    #[test]
    fn config_reranker_mode_roundtrip_and_default_none() {
        // Persisted reranker_mode survives a serialize -> deserialize round-trip,
        // and an old config file without the field deserializes to None (serde default).
        let cfg = Config {
            reranker_mode: Some("full".to_string()),
            ..Default::default()
        };
        let back: Config = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(back.reranker_mode.as_deref(), Some("full"));

        let old: Config = serde_json::from_str(r#"{"anthropic_api_key": "sk-test"}"#).unwrap();
        assert_eq!(old.reranker_mode, None);
    }

    #[test]
    fn page_map_auto_suggest_defaults_false_on_missing_field() {
        // A config file without the field deserializes to false — a Canvas the
        // user has not asked to populate stays blank. This is the teeth on the
        // start-from-blank decision; flipping the serde default back to true
        // fails here.
        let config: Config = serde_json::from_str(r#"{"clipboard_enabled": true}"#).unwrap();
        assert!(!config.page_map_auto_suggest);
    }

    #[test]
    fn page_map_auto_suggest_roundtrip_true() {
        // Opting back in survives a serialize/deserialize cycle, so a user who
        // turns proactive suggestions on keeps them.
        let config = Config {
            page_map_auto_suggest: true,
            ..Config::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&json).unwrap();
        assert!(restored.page_map_auto_suggest);
    }

    #[test]
    fn page_map_auto_suggest_tolerates_unknown_fields() {
        // Unknown/adjacent fields don't disturb the flag's serde default.
        let json = r#"{"page_map_auto_suggest": false, "some_future_field": 42}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(!config.page_map_auto_suggest);
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
        let default_path = dirs::home_dir().unwrap().join(".wenlan/pages");
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

    #[test]
    fn test_external_llm_api_key_roundtrip_and_default() {
        let cfg: Config = serde_json::from_str("{}").unwrap();
        assert!(cfg.external_llm_api_key.is_none());
        let cfg = Config {
            external_llm_api_key: Some("sk-test".into()),
            ..Config::default()
        };
        let restored: Config = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(restored.external_llm_api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn test_stores_secret_covers_both_keys() {
        assert!(!stores_secret(&Config::default()));
        assert!(stores_secret(&Config {
            anthropic_api_key: Some("k".into()),
            ..Config::default()
        }));
        assert!(stores_secret(&Config {
            external_llm_api_key: Some("k".into()),
            ..Config::default()
        }));
    }
}
