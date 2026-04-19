// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClient {
    pub name: String,
    pub client_type: String,
    pub config_path: String,
    pub detected: bool,
    pub already_configured: bool,
}

/// Returns the expected config file path for each MCP client on macOS.
pub fn client_config_path(client_type: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    match client_type {
        "claude_desktop" => {
            // macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
            dirs::config_dir().map(|d| d.join("Claude").join("claude_desktop_config.json"))
        }
        "cursor" => Some(home.join(".cursor").join("mcp.json")),
        "claude_code" => Some(home.join(".claude.json")),
        _ => None,
    }
}

/// Check if a JSON config string already has an "origin" entry in mcpServers.
fn has_origin_entry(json_str: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json_str)
        .ok()
        .and_then(|v| v.get("mcpServers")?.get("origin").cloned())
        .is_some()
}

/// Detect installed MCP-compatible tools and whether Origin is already configured.
pub fn detect_mcp_clients() -> Vec<McpClient> {
    // Claude Desktop skipped — uses remote MCP, not local stdio.
    // Will need a separate setup flow once remote MCP support is added.
    let clients = [("Cursor", "cursor"), ("Claude Code", "claude_code")];

    clients
        .iter()
        .filter_map(|(name, client_type)| {
            let config_path = client_config_path(client_type)?;
            let config_path_str = config_path.to_string_lossy().to_string();

            let (detected, already_configured) = if client_type == &"cursor" {
                // Cursor: detect by app bundle, not config file
                let app_exists = std::path::Path::new("/Applications/Cursor.app").exists()
                    || dirs::home_dir()
                        .map(|h| h.join("Applications/Cursor.app").exists())
                        .unwrap_or(false);
                let configured = config_path.exists()
                    && std::fs::read_to_string(&config_path)
                        .map(|s| has_origin_entry(&s))
                        .unwrap_or(false);
                (app_exists, configured)
            } else {
                // Claude Desktop & Claude Code: detect by config file existence
                let exists = config_path.exists();
                let configured = exists
                    && std::fs::read_to_string(&config_path)
                        .map(|s| has_origin_entry(&s))
                        .unwrap_or(false);
                (exists, configured)
            };

            Some(McpClient {
                name: name.to_string(),
                client_type: client_type.to_string(),
                config_path: config_path_str,
                detected,
                already_configured,
            })
        })
        .collect()
}

/// Search for a local origin-mcp binary (dev fallback).
fn find_origin_mcp_binary() -> Option<PathBuf> {
    let candidates = [
        dirs::home_dir().map(|h| h.join(".cargo/bin/origin-mcp")),
        dirs::home_dir().map(|h| h.join("Repos/origin-mcp/target/release/origin-mcp")),
        Some(PathBuf::from("/usr/local/bin/origin-mcp")),
    ];
    candidates.into_iter().flatten().find(|p| p.exists())
}

/// The MCP config entry Origin writes into client config files.
/// Default: npx (production path, requires origin-mcp published to npm).
/// Dev fallback: uses local binary if found on disk.
pub fn origin_mcp_entry() -> serde_json::Value {
    // Dev fallback: use local binary if available
    if let Some(binary_path) = find_origin_mcp_binary() {
        return serde_json::json!({
            "command": binary_path.to_string_lossy(),
            "args": []
        });
    }
    // Production default
    serde_json::json!({
        "command": "npx",
        "args": ["-y", "origin-mcp"]
    })
}

/// Write the origin MCP server entry into a client's config file.
/// If `is_claude_code` is true and the file doesn't exist, returns an error
/// (Claude Code manages its own config file).
pub fn write_origin_entry(
    config_path: &std::path::Path,
    is_claude_code: bool,
) -> Result<(), OriginError> {
    let mut root = if config_path.exists() {
        // Back up existing file
        let backup_path = config_path.with_extension("json.bak");
        std::fs::copy(config_path, &backup_path)?;

        let contents = std::fs::read_to_string(config_path)?;
        serde_json::from_str::<serde_json::Value>(&contents).map_err(|e| {
            OriginError::Generic(format!("Invalid JSON in {}: {}", config_path.display(), e))
        })?
    } else if is_claude_code {
        return Err(OriginError::Generic(
            "Claude Code config file not found — Claude Code manages this file internally".into(),
        ));
    } else {
        // Create minimal skeleton for Claude Desktop / Cursor
        serde_json::json!({})
    };

    // Ensure mcpServers key exists
    if root.get("mcpServers").is_none() {
        root["mcpServers"] = serde_json::json!({});
    }
    root["mcpServers"]["origin"] = origin_mcp_entry();

    // Write back with pretty formatting
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let formatted =
        serde_json::to_string_pretty(&root).map_err(|e| OriginError::Generic(e.to_string()))?;
    std::fs::write(config_path, formatted)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_path_claude_desktop() {
        let path = client_config_path("claude_desktop").unwrap();
        assert!(path.to_string_lossy().contains("Claude"));
        assert!(path
            .to_string_lossy()
            .ends_with("claude_desktop_config.json"));
    }

    #[test]
    fn test_client_config_path_cursor() {
        let path = client_config_path("cursor").unwrap();
        assert!(path.to_string_lossy().contains(".cursor"));
        assert!(path.to_string_lossy().ends_with("mcp.json"));
    }

    #[test]
    fn test_client_config_path_claude_code() {
        let path = client_config_path("claude_code").unwrap();
        assert!(path.to_string_lossy().ends_with(".claude.json"));
    }

    #[test]
    fn test_client_config_path_unknown() {
        assert!(client_config_path("unknown").is_none());
    }

    #[test]
    fn test_check_already_configured_finds_origin() {
        let json =
            r#"{"mcpServers": {"origin": {"command": "npx", "args": ["-y", "origin-mcp"]}}}"#;
        assert!(has_origin_entry(json));
    }

    #[test]
    fn test_check_already_configured_not_found() {
        let json = r#"{"mcpServers": {"other-server": {}}}"#;
        assert!(!has_origin_entry(json));
    }

    #[test]
    fn test_check_already_configured_no_mcp_servers_key() {
        let json = r#"{"theme": "dark"}"#;
        assert!(!has_origin_entry(json));
    }

    #[test]
    fn test_check_already_configured_invalid_json() {
        assert!(!has_origin_entry("not json"));
    }

    #[test]
    fn test_write_origin_entry_creates_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        write_origin_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        // Command is either a local binary path or npx (depending on host)
        let cmd = parsed["mcpServers"]["origin"]["command"].as_str().unwrap();
        assert!(
            cmd == "npx" || cmd.ends_with("origin-mcp"),
            "expected npx or path to origin-mcp, got: {cmd}"
        );
    }

    #[test]
    fn test_write_origin_entry_preserves_existing_servers() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing = r#"{"mcpServers": {"other": {"command": "other-cmd"}}}"#;
        std::fs::write(&config_path, existing).unwrap();
        write_origin_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(parsed["mcpServers"]["other"].is_object());
        assert!(parsed["mcpServers"]["origin"].is_object());
    }

    #[test]
    fn test_write_origin_entry_creates_mcp_servers_key() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, r#"{"theme": "dark"}"#).unwrap();
        write_origin_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["theme"], "dark");
        assert!(parsed["mcpServers"]["origin"].is_object());
    }

    #[test]
    fn test_write_origin_entry_creates_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, r#"{"original": true}"#).unwrap();
        write_origin_entry(&config_path, false).unwrap();
        let backup = tmp.path().join("config.json.bak");
        assert!(backup.exists());
        let backup_contents = std::fs::read_to_string(&backup).unwrap();
        assert!(backup_contents.contains("original"));
    }

    #[test]
    fn test_write_origin_entry_errors_on_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, "not valid json").unwrap();
        let result = write_origin_entry(&config_path, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_origin_entry_refuses_create_for_claude_code() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("claude.json");
        // is_claude_code = true, file doesn't exist → should error
        let result = write_origin_entry(&config_path, true);
        assert!(result.is_err());
    }
}
