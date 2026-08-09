// SPDX-License-Identifier: AGPL-3.0-only
//! App-local Obsidian helpers used at source registration time.
//! Only the filesystem-scanning utilities are needed by the app.
//! The heavy note_to_documents conversion stays in origin-core (used by daemon).
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Directories to skip when scanning a vault.
const SKIP_DIRS: &[&str] = &[".obsidian", ".trash", ".git", "templates"];

/// Check if a path should be skipped (any component matches a skip directory).
fn should_skip(path: &Path) -> bool {
    path.components().any(|c| {
        let name = c.as_os_str().to_string_lossy();
        SKIP_DIRS.contains(&name.as_ref())
    })
}

/// Short-circuit check: returns `true` as soon as any `.md` file is found in
/// `root` or any (non-skipped) subdirectory. Used at source registration time
/// so we don't need to walk the whole vault.
pub fn has_any_markdown(root: &Path) -> bool {
    has_any_markdown_recursive(root, root)
}

/// Convert a title string into a URL-safe slug (lowercase, spaces to hyphens,
/// non-alphanumeric chars removed). Inlined from origin-core::export::obsidian.
pub fn slugify(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else if c == ' ' {
                '-'
            } else {
                '\0'
            }
        })
        .filter(|&c| c != '\0')
        .collect::<String>()
}

fn has_any_markdown_recursive(root: &Path, dir: &Path) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if should_skip(relative) {
            continue;
        }

        if path.is_dir() {
            if has_any_markdown_recursive(root, &path) {
                return true;
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            return true;
        }
    }
    false
}

// ── Vault discovery via Obsidian's own registry ─────────────────────────
//
// Obsidian keeps a machine-local registry of every vault it has ever opened
// at `obsidian.json`. Reading it lets the connect flow offer the user's real
// vaults as one-tap chips instead of always sending them through the folder
// picker. This is a convenience, never a dependency: any failure to read or
// parse the registry returns an empty list rather than an error, so a user
// without Obsidian (or with a registry we can't parse) sees the card behave
// exactly as it does without this feature.

/// One entry from Obsidian's vault registry, filtered to vaults that still
/// exist on disk.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ObsidianVault {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Deserialize)]
struct ObsidianRegistry {
    #[serde(default)]
    vaults: HashMap<String, ObsidianRegistryEntry>,
}

#[derive(Debug, Deserialize)]
struct ObsidianRegistryEntry {
    path: String,
    #[serde(default)]
    ts: i64,
}

/// Path to Obsidian's vault registry on macOS.
pub fn obsidian_registry_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Library/Application Support/obsidian/obsidian.json")
}

/// Read and parse the Obsidian vault registry at `registry_path`, sorted by
/// most-recently-opened first. Never errors: a missing file, unparseable
/// JSON, or a missing `vaults` key all resolve to an empty list. Vaults whose
/// path no longer exists on disk are dropped.
pub fn discover_vaults(registry_path: &Path) -> Vec<ObsidianVault> {
    let contents = match std::fs::read_to_string(registry_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let registry: ObsidianRegistry = match serde_json::from_str(&contents) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut entries: Vec<ObsidianRegistryEntry> = registry
        .vaults
        .into_values()
        .filter(|entry| Path::new(&entry.path).is_dir())
        .collect();
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.ts));

    entries
        .into_iter()
        .map(|entry| {
            let name = Path::new(&entry.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&entry.path)
                .to_string();
            ObsidianVault {
                name,
                path: entry.path,
            }
        })
        .collect()
}

#[cfg(test)]
mod discover_vaults_tests {
    use super::*;
    use std::fs;

    fn write_registry(dir: &Path, json: &str) -> PathBuf {
        let registry_path = dir.join("obsidian.json");
        fs::write(&registry_path, json).unwrap();
        registry_path
    }

    #[test]
    fn happy_path_sorts_most_recently_opened_first() {
        let tmp = tempfile::tempdir().unwrap();
        let vault_a = tmp.path().join("Vault A");
        let vault_b = tmp.path().join("Vault B");
        fs::create_dir_all(&vault_a).unwrap();
        fs::create_dir_all(&vault_b).unwrap();

        let registry_path = write_registry(
            tmp.path(),
            &format!(
                r#"{{"vaults": {{
                    "id-a": {{"path": "{}", "ts": 1000, "open": true}},
                    "id-b": {{"path": "{}", "ts": 2000}}
                }}}}"#,
                vault_a.display().to_string().replace('\\', "\\\\"),
                vault_b.display().to_string().replace('\\', "\\\\"),
            ),
        );

        let result = discover_vaults(&registry_path);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "Vault B");
        assert_eq!(result[0].path, vault_b.to_string_lossy());
        assert_eq!(result[1].name, "Vault A");
        assert_eq!(result[1].path, vault_a.to_string_lossy());
    }

    #[test]
    fn missing_registry_file_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let registry_path = tmp.path().join("does-not-exist.json");
        assert_eq!(discover_vaults(&registry_path), Vec::new());
    }

    #[test]
    fn malformed_json_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let registry_path = write_registry(tmp.path(), "{ not valid json");
        assert_eq!(discover_vaults(&registry_path), Vec::new());
    }

    #[test]
    fn vault_path_no_longer_on_disk_is_filtered_out() {
        let tmp = tempfile::tempdir().unwrap();
        let vault_a = tmp.path().join("Still Here");
        fs::create_dir_all(&vault_a).unwrap();
        let gone_path = tmp.path().join("Deleted Vault");

        let registry_path = write_registry(
            tmp.path(),
            &format!(
                r#"{{"vaults": {{
                    "id-a": {{"path": "{}", "ts": 1000}},
                    "id-gone": {{"path": "{}", "ts": 2000}}
                }}}}"#,
                vault_a.display(),
                gone_path.display(),
            ),
        );

        let result = discover_vaults(&registry_path);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Still Here");
    }
}
