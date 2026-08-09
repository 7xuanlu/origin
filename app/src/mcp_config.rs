// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClient {
    pub name: String,
    pub client_type: String,
    pub config_path: String,
    pub detected: bool,
    pub already_configured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WenlanMcpEntry {
    pub command: String,
    pub args: Vec<String>,
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
        "gemini_cli" => Some(home.join(".gemini").join("settings.json")),
        "codex_cli" => Some(home.join(".codex").join("config.toml")),
        _ => None,
    }
}

const MCP_SERVER_KEY: &str = "wenlan";
const LEGACY_MCP_SERVER_KEY: &str = "origin";

/// Check if a JSON config string already has a Wenlan entry or legacy Origin entry.
fn has_configured_entry(json_str: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json_str)
        .ok()
        .and_then(|v| {
            let servers = v.get("mcpServers")?;
            Some(
                servers.get(MCP_SERVER_KEY).is_some()
                    || servers.get(LEGACY_MCP_SERVER_KEY).is_some(),
            )
        })
        .unwrap_or(false)
}

/// TOML variant for Codex CLI (`[mcp_servers.*]` tables).
fn has_configured_entry_toml(toml_str: &str) -> bool {
    toml_str
        .parse::<toml_edit::DocumentMut>()
        .ok()
        .and_then(|doc| {
            let servers = doc.get("mcp_servers")?;
            Some(
                servers.get(MCP_SERVER_KEY).is_some()
                    || servers.get(LEGACY_MCP_SERVER_KEY).is_some(),
            )
        })
        .unwrap_or(false)
}

/// Check whether a JSON config holds BOTH the live `wenlan` entry AND the
/// legacy `origin` entry under `mcpServers` — the raw+raw duplicate a client
/// with no plugin path (Cursor, Gemini CLI) lands in after the origin→wenlan
/// rename, where both entries launch a server against the same daemon. Distinct
/// from `has_configured_entry`, which is an OR: the fix here removes only the
/// stale `origin`, so detection has to know both are present, not just one.
fn has_both_raw_entries(json_str: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json_str)
        .ok()
        .and_then(|v| {
            let servers = v.get("mcpServers")?;
            Some(
                servers.get(MCP_SERVER_KEY).is_some()
                    && servers.get(LEGACY_MCP_SERVER_KEY).is_some(),
            )
        })
        .unwrap_or(false)
}

/// TOML variant of `has_both_raw_entries` for Codex CLI (`[mcp_servers.*]`).
fn has_both_raw_entries_toml(toml_str: &str) -> bool {
    toml_str
        .parse::<toml_edit::DocumentMut>()
        .ok()
        .and_then(|doc| {
            let servers = doc.get("mcp_servers")?;
            Some(
                servers.get(MCP_SERVER_KEY).is_some()
                    && servers.get(LEGACY_MCP_SERVER_KEY).is_some(),
            )
        })
        .unwrap_or(false)
}

/// Whether a Claude Code `settings.json` blob has the Wenlan plugin enabled.
/// `enabledPlugins` keys are `<plugin>@<marketplace>`, and the marketplace
/// name varies by install (`wenlan@7xuanlu` fresh, `wenlan@7xuanlu-wenlan` on
/// a machine that added the old self-marketplace) — match the `wenlan@`
/// prefix, never a literal marketplace name, or the check breaks for exactly
/// one of the two populations. Malformed JSON or a missing key is "no
/// plugin", never an error: a user who has never touched plugins should not
/// see this crash the detector.
fn claude_code_plugin_enabled(settings_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(settings_json)
        .ok()
        .and_then(|v| v.get("enabledPlugins")?.as_object().cloned())
        .map(|plugins| {
            plugins
                .iter()
                .any(|(key, val)| key.starts_with("wenlan@") && val.as_bool() == Some(true))
        })
        .unwrap_or(false)
}

/// Reads the real `~/.claude/settings.json` and checks it via
/// `claude_code_plugin_enabled`. Split out so the matching logic stays a pure,
/// directly testable function.
fn claude_code_plugin_enabled_on_disk() -> bool {
    let Some(home) = dirs::home_dir() else {
        return false;
    };
    std::fs::read_to_string(home.join(".claude").join("settings.json"))
        .map(|s| claude_code_plugin_enabled(&s))
        .unwrap_or(false)
}

/// Whether a Codex CLI `config.toml` blob has the Wenlan plugin enabled —
/// `[plugins."wenlan@<marketplace>"] enabled = true`. The marketplace name
/// varies (`wenlan-local` pre-7xuanlu/wenlan#348, `7xuanlu-wenlan` after),
/// so match the `wenlan@` prefix, never a literal marketplace name — same
/// reasoning as `claude_code_plugin_enabled`.
fn codex_cli_plugin_enabled(toml_str: &str) -> bool {
    toml_str
        .parse::<toml_edit::DocumentMut>()
        .ok()
        .and_then(|doc| {
            let plugins = doc.get("plugins")?.as_table_like()?;
            Some(plugins.iter().any(|(key, item)| {
                key.starts_with("wenlan@")
                    && item.get("enabled").and_then(|v| v.as_bool()) == Some(true)
            }))
        })
        .unwrap_or(false)
}

/// Reads the real `~/.codex/config.toml` and checks it via
/// `codex_cli_plugin_enabled`. Split out so the matching logic stays a pure,
/// directly testable function.
fn codex_cli_plugin_enabled_on_disk() -> bool {
    let Some(home) = dirs::home_dir() else {
        return false;
    };
    std::fs::read_to_string(home.join(".codex").join("config.toml"))
        .map(|s| codex_cli_plugin_enabled(&s))
        .unwrap_or(false)
}

/// Whether a Claude Desktop chat-side plugin manifest (`rpm/manifest.json`
/// under a session directory) lists the Wenlan plugin. True iff `plugins[]`
/// contains an entry whose `name` field is exactly `"wenlan"` — matching
/// `marketplaceName` instead would be wrong, since a user's own upload
/// marketplace can be named anything (`marketplaceName` values seen in the
/// wild: `"My Uploads"`, `"knowledge-work-plugins"`). Malformed JSON or a
/// missing `plugins` key is "no plugin", never an error — same policy as
/// `claude_code_plugin_enabled`.
fn claude_desktop_plugin_enabled(manifest_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(manifest_json)
        .ok()
        .and_then(|v| v.get("plugins")?.as_array().cloned())
        .map(|plugins| {
            plugins
                .iter()
                .any(|p| p.get("name").and_then(|n| n.as_str()) == Some("wenlan"))
        })
        .unwrap_or(false)
}

/// Extract the pinned account id (`lastKnownAccountUuid`) from a Claude
/// Desktop `config.json` blob. `None` on malformed JSON or a missing key.
fn claude_desktop_account_id(config_json: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(config_json)
        .ok()?
        .get("lastKnownAccountUuid")?
        .as_str()
        .map(String::from)
}

/// The directory holding one subdirectory per chat-side session id for a
/// given account: `<support_dir>/local-agent-mode-sessions/<account_id>`.
/// Pure path construction, so a typo in either hardcoded segment fails a
/// test that reads the returned path directly, rather than surfacing as an
/// unexplained `false` several calls downstream. Scoping to the pinned
/// account id also means the `skills-plugin` sentinel directory that lives
/// alongside real account-id directories under `local-agent-mode-sessions/`
/// is never visited — it isn't a UUID, so it can never be `account_id`.
fn claude_desktop_account_sessions_dir(support_dir: &Path, account_id: &str) -> PathBuf {
    support_dir
        .join("local-agent-mode-sessions")
        .join(account_id)
}

/// Whether any session under `account_sessions_dir` has a
/// `rpm/manifest.json` listing the Wenlan plugin. One directory per session
/// id; any single session counting is enough (a user can have several open
/// at once). A session directory without `rpm/manifest.json`, or with one
/// that fails to parse, is silently skipped — never a panic, never treated
/// as a match.
fn claude_desktop_plugin_enabled_in_sessions_dir(account_sessions_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(account_sessions_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        std::fs::read_to_string(entry.path().join("rpm").join("manifest.json"))
            .map(|s| claude_desktop_plugin_enabled(&s))
            .unwrap_or(false)
    })
}

/// Whether the Wenlan plugin is enabled for Claude Desktop, given the
/// already-resolved support directory (normally
/// `~/Library/Application Support/Claude`). Composes
/// `claude_desktop_account_id` + `claude_desktop_account_sessions_dir` +
/// `claude_desktop_plugin_enabled_in_sessions_dir` end to end, so the full
/// real-world path — `config.json` -> account id -> sessions dir -> manifest
/// scan — is exercised under test with a tempdir standing in for
/// `support_dir`. That leaves nothing for `claude_desktop_plugin_enabled_on_disk`
/// or its `detect_mcp_clients` call site to quietly sever.
fn claude_desktop_plugin_enabled_for_support_dir(support_dir: &Path) -> bool {
    let Some(account_id) = std::fs::read_to_string(support_dir.join("config.json"))
        .ok()
        .and_then(|s| claude_desktop_account_id(&s))
    else {
        return false;
    };
    claude_desktop_plugin_enabled_in_sessions_dir(&claude_desktop_account_sessions_dir(
        support_dir,
        &account_id,
    ))
}

/// Reads the real Claude Desktop support directory
/// (`~/Library/Application Support/Claude`) and checks it via
/// `claude_desktop_plugin_enabled_for_support_dir`. Split out so the
/// matching/composition logic stays directly testable — mirrors
/// `claude_code_plugin_enabled_on_disk` and `codex_cli_plugin_enabled_on_disk`.
/// Missing home dir is "no plugin", never an error: a user who has never
/// opened Claude Desktop must not see this break detection. READ-ONLY: never
/// creates, writes, or modifies anything under Claude Desktop's support
/// directory — that state belongs to another vendor's app.
fn claude_desktop_plugin_enabled_on_disk() -> bool {
    let Some(support_dir) = dirs::config_dir().map(|d| d.join("Claude")) else {
        return false;
    };
    claude_desktop_plugin_enabled_for_support_dir(&support_dir)
}

/// Where ChatGPT desktop can be installed. Its Codex pane reads the same
/// `~/.codex/config.toml` as Codex CLI (OpenAI merged Codex into the ChatGPT
/// app), so finding the bundle means the `codex_cli` row applies.
fn chatgpt_app_candidates(home: Option<&Path>) -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("/Applications/ChatGPT.app")];
    if let Some(home) = home {
        out.push(home.join("Applications/ChatGPT.app"));
    }
    out
}

/// Whether the Codex CLI row should be detected: either its shared
/// `~/.codex/config.toml` exists, or ChatGPT desktop is installed. Feeds the
/// single `codex_cli` row — never a second row for ChatGPT.
///
/// `exists` is injected so the bundle paths themselves are under test: a typo
/// in a candidate path fails `codex_cli_detected_finds_chatgpt_in_*`, and the
/// call site cannot silently opt out of the probe (there is no bool to pass).
fn codex_cli_detected(
    config_exists: bool,
    home: Option<&Path>,
    exists: impl Fn(&Path) -> bool,
) -> bool {
    config_exists
        || chatgpt_app_candidates(home)
            .iter()
            .any(|p| exists(p.as_path()))
}

/// Whether `client_type`'s own config file has a raw wenlan/origin
/// `mcpServers` (or Codex's `[mcp_servers.*]`) entry — the file-based half of
/// what `detect_mcp_clients` folds into a single `already_configured` bool.
/// `wire_state` needs the two halves kept apart: a raw entry and a missing
/// plugin point at different fixes. Same "exists + parses + has the key"
/// logic as `detect_mcp_clients`'s own `config_has_entry` closure — sharing
/// `has_configured_entry`/`has_configured_entry_toml` keeps the two readings
/// from drifting.
pub(crate) fn client_config_has_raw_entry(client_type: &str, config_path: &Path) -> bool {
    if !config_path.exists() {
        return false;
    }
    std::fs::read_to_string(config_path)
        .map(|s| {
            if client_type == "codex_cli" {
                has_configured_entry_toml(&s)
            } else {
                has_configured_entry(&s)
            }
        })
        .unwrap_or(false)
}

/// Whether `client_type`'s own config file holds BOTH the `wenlan` entry and
/// the legacy `origin` entry — the raw+raw duplicate. Mirrors
/// `client_config_has_raw_entry`'s file handling and TOML/JSON split, sharing
/// `has_both_raw_entries`/`has_both_raw_entries_toml` so detection and the
/// `remove_legacy_origin_entry` fix stay symmetric. This is the one signal a
/// no-plugin client (cursor, gemini_cli) needs: those can never trip the
/// plugin+raw double-registration path in `wire_state`, so without it their
/// raw+raw duplicate is invisible.
pub(crate) fn client_config_has_both_raw_entries(client_type: &str, config_path: &Path) -> bool {
    if !config_path.exists() {
        return false;
    }
    std::fs::read_to_string(config_path)
        .map(|s| {
            if client_type == "codex_cli" {
                has_both_raw_entries_toml(&s)
            } else {
                has_both_raw_entries(&s)
            }
        })
        .unwrap_or(false)
}

/// Whether `client_type`'s Wenlan plugin is enabled — the plugin half of
/// `already_configured` for the three clients that support one. `cursor` and
/// `gemini_cli` have no plugin path, so they're always `false` here (and
/// route to `"config"` in `wire_state`, never `"plugin"`).
pub(crate) fn client_plugin_enabled(client_type: &str) -> bool {
    match client_type {
        "claude_code" => claude_code_plugin_enabled_on_disk(),
        "codex_cli" => codex_cli_plugin_enabled_on_disk(),
        "claude_desktop" => claude_desktop_plugin_enabled_on_disk(),
        _ => false,
    }
}

/// Detect installed MCP-compatible tools and whether Wenlan is already configured.
pub fn detect_mcp_clients() -> Vec<McpClient> {
    let clients = [
        ("Cursor", "cursor"),
        ("Claude Code", "claude_code"),
        ("Claude Desktop", "claude_desktop"),
        ("Gemini CLI", "gemini_cli"),
        ("Codex CLI", "codex_cli"),
    ];

    clients
        .iter()
        .filter_map(|(name, client_type)| {
            let config_path = client_config_path(client_type)?;
            let config_path_str = config_path.to_string_lossy().to_string();

            let is_toml = *client_type == "codex_cli";
            let config_has_entry = || {
                config_path.exists()
                    && std::fs::read_to_string(&config_path)
                        .map(|s| {
                            if is_toml {
                                has_configured_entry_toml(&s)
                            } else {
                                has_configured_entry(&s)
                            }
                        })
                        .unwrap_or(false)
            };

            let (detected, already_configured) = if client_type == &"cursor" {
                // Cursor: detect by app bundle, not config file
                let app_exists = std::path::Path::new("/Applications/Cursor.app").exists()
                    || dirs::home_dir()
                        .map(|h| h.join("Applications/Cursor.app").exists())
                        .unwrap_or(false);
                (app_exists, config_has_entry())
            } else if client_type == &"claude_code" {
                // Claude Code also counts as configured via the Wenlan plugin
                // (`enabledPlugins` in `~/.claude/settings.json`), which
                // registers its own MCP server without touching
                // `~/.claude.json` — see claude_code_plugin_enabled.
                (
                    config_path.exists(),
                    config_has_entry() || claude_code_plugin_enabled_on_disk(),
                )
            } else if client_type == &"codex_cli" {
                // Codex CLI also counts as configured via the Wenlan plugin
                // (`[plugins."wenlan@<marketplace>"]` in
                // `~/.codex/config.toml`), which registers its own MCP
                // server without a separate `[mcp_servers.wenlan]` entry —
                // see codex_cli_plugin_enabled.
                //
                // Detection also fires off ChatGPT desktop's app bundle:
                // ChatGPT desktop's Codex pane reads the same
                // `~/.codex/config.toml` as Codex CLI, so a user who only
                // has ChatGPT desktop (never ran Codex CLI) still gets this
                // row — see codex_cli_detected.
                (
                    codex_cli_detected(config_path.exists(), dirs::home_dir().as_deref(), |p| {
                        p.exists()
                    }),
                    config_has_entry() || codex_cli_plugin_enabled_on_disk(),
                )
            } else if client_type == &"claude_desktop" {
                // Claude Desktop also counts as configured via the chat-side
                // Wenlan plugin (`rpm/manifest.json` under a session
                // directory in `local-agent-mode-sessions/<account_id>/`),
                // which registers its own MCP server without touching
                // `claude_desktop_config.json` — see
                // claude_desktop_plugin_enabled_on_disk.
                (
                    config_path.exists(),
                    config_has_entry() || claude_desktop_plugin_enabled_on_disk(),
                )
            } else {
                // Everything else: detect by config file existence
                (config_path.exists(), config_has_entry())
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

/// The backend release this app ships against. Deriving the npm fallback from
/// the same file the sidecar download uses means the two can never disagree:
/// bumping the pin bumps the fallback.
const BACKEND_VERSION_PIN: &str = include_str!("../../.wenlan-backend-version");

/// `wenlan-mcp@^<pinned version>`, e.g. `wenlan-mcp@^0.12.0`. Falls back to the
/// bare package name only if the pin file is unparseable — an unpinned `npx`
/// can silently pull a backend the app was never tested against.
fn pinned_wenlan_mcp_package(pin_file: &str) -> String {
    let version = pin_file
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .trim_start_matches('v');
    if version.is_empty() || !version.starts_with(|c: char| c.is_ascii_digit()) {
        return "wenlan-mcp".to_string();
    }
    format!("wenlan-mcp@^{version}")
}

/// Each `wenlan-mcp` candidate paired with where it came from, most-specific
/// first — the single source of truth `wenlan_mcp_candidates` (the plain
/// path list `find_wenlan_mcp_binary` resolves against) and `wire_state`'s
/// candidate trail both derive from, so the two can never disagree about
/// what was tried. Mirrors the plugin's own `wenlan-mcp-runner.sh`
/// resolution order.
///
/// Deliberately does *not* probe a cargo target dir. `~/Repos/wenlan/target/release`
/// used to rank above the installed binary here, so the wizard baked a maintainer's
/// build-artifact path into real users' client configs — and the entry died the next
/// `cargo clean`. A target dir is a build output, not an install location.
pub(crate) fn wenlan_mcp_candidate_sources(
    home: Option<&Path>,
    dev_bin: Option<&str>,
    exe_dir: Option<&Path>,
) -> Vec<(PathBuf, &'static str)> {
    let mut candidates = Vec::new();
    if let Some(dev_bin) = dev_bin.filter(|p| !p.trim().is_empty()) {
        candidates.push((PathBuf::from(dev_bin), "WENLAN_MCP_DEV_BIN"));
    }
    if let Some(home) = home {
        candidates.push((home.join(".wenlan/bin/wenlan-mcp"), "installed"));
    }
    if let Some(exe_dir) = exe_dir {
        candidates.push((exe_dir.join("wenlan-mcp"), "bundled"));
    }
    if let Some(home) = home {
        candidates.push((home.join(".cargo/bin/wenlan-mcp"), "cargo"));
    }
    candidates
}

fn wenlan_mcp_candidates(
    home: Option<&Path>,
    dev_bin: Option<&str>,
    exe_dir: Option<&Path>,
) -> Vec<PathBuf> {
    wenlan_mcp_candidate_sources(home, dev_bin, exe_dir)
        .into_iter()
        .map(|(path, _source)| path)
        .collect()
}

fn find_wenlan_mcp_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let dev_bin = std::env::var("WENLAN_MCP_DEV_BIN").ok();
    wenlan_mcp_candidates(
        dirs::home_dir().as_deref(),
        dev_bin.as_deref(),
        exe.as_ref().and_then(|exe| exe.parent()),
    )
    .into_iter()
    .find(|p| p.exists())
}

/// The MCP config entry Wenlan writes into client config files: an installed
/// binary when one exists, otherwise a version-pinned `npx`.
fn wenlan_mcp_entry_for(binary: Option<PathBuf>, npm_package: &str) -> WenlanMcpEntry {
    match binary {
        Some(path) => WenlanMcpEntry {
            command: path.to_string_lossy().to_string(),
            args: Vec::new(),
        },
        None => WenlanMcpEntry {
            command: "npx".to_string(),
            args: vec!["-y".to_string(), npm_package.to_string()],
        },
    }
}

pub fn wenlan_mcp_entry() -> WenlanMcpEntry {
    wenlan_mcp_entry_for(
        find_wenlan_mcp_binary(),
        &pinned_wenlan_mcp_package(BACKEND_VERSION_PIN),
    )
}

/// Write the Wenlan MCP server entry into a client's config file.
/// Existing legacy `origin` entries are preserved and still detected.
/// If `is_claude_code` is true and the file doesn't exist, returns an error
/// (Claude Code manages its own config file).
pub fn write_wenlan_entry(
    config_path: &std::path::Path,
    is_claude_code: bool,
) -> Result<(), AppError> {
    let mut root = if config_path.exists() {
        // Back up existing file
        let backup_path = config_path.with_extension("json.bak");
        std::fs::copy(config_path, &backup_path)?;

        let contents = std::fs::read_to_string(config_path)?;
        serde_json::from_str::<serde_json::Value>(&contents).map_err(|e| {
            AppError::Generic(format!("Invalid JSON in {}: {}", config_path.display(), e))
        })?
    } else if is_claude_code {
        return Err(AppError::Generic(
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
    root["mcpServers"][MCP_SERVER_KEY] =
        serde_json::to_value(wenlan_mcp_entry()).map_err(|e| AppError::Generic(e.to_string()))?;

    // Write back with pretty formatting
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let formatted =
        serde_json::to_string_pretty(&root).map_err(|e| AppError::Generic(e.to_string()))?;
    std::fs::write(config_path, formatted)?;

    Ok(())
}

/// Upsert the Wenlan entry into a Codex CLI `config.toml` — format-preserving:
/// user comments, key order, and unrelated tables survive byte-for-byte
/// (toml_edit round-trips everything it didn't touch).
pub fn write_wenlan_entry_toml(config_path: &std::path::Path) -> Result<(), AppError> {
    use toml_edit::{DocumentMut, Item, Table};

    let mut doc: DocumentMut = if config_path.exists() {
        let backup_path = config_path.with_extension("toml.bak");
        std::fs::copy(config_path, &backup_path)?;
        let contents = std::fs::read_to_string(config_path)?;
        contents.parse().map_err(|e| {
            AppError::Generic(format!("Invalid TOML in {}: {}", config_path.display(), e))
        })?
    } else {
        DocumentMut::new()
    };

    if doc.get("mcp_servers").is_none() {
        let mut parent = Table::new();
        parent.set_implicit(true); // render only [mcp_servers.wenlan], no bare [mcp_servers]
        doc.insert("mcp_servers", Item::Table(parent));
    }

    let entry = wenlan_mcp_entry();
    let mut server = Table::new();
    server.insert("command", toml_edit::value(entry.command));
    let mut args = toml_edit::Array::new();
    for a in entry.args {
        args.push(a);
    }
    server.insert("args", toml_edit::value(args));
    doc["mcp_servers"][MCP_SERVER_KEY] = Item::Table(server);

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(config_path, doc.to_string())?;
    Ok(())
}

/// Remove the raw `wenlan`/legacy `origin` `mcpServers` entries from a JSON
/// client config — the inverse of `write_wenlan_entry`, and the fix for the
/// double-registration Diagnostics surfaces (a plugin *and* a raw entry for
/// one client). Symmetric with detection: it removes exactly the keys
/// `has_configured_entry` recognizes, so `client_config_has_raw_entry` reads
/// `false` afterwards. Every sibling server and unrelated key survives.
/// A missing file, or a file with neither key present, is `Err` — there is
/// nothing to remove, and the caller surfaces that verbatim. Backs the file
/// up first (like `write_wenlan_entry`), but only once a removal is certain,
/// so the no-op error path leaves no stray `.bak`.
pub fn remove_wenlan_entry(config_path: &std::path::Path) -> Result<(), AppError> {
    if !config_path.exists() {
        return Err(AppError::Generic(
            "No config file found — nothing to remove".into(),
        ));
    }
    let contents = std::fs::read_to_string(config_path)?;
    let mut root = serde_json::from_str::<serde_json::Value>(&contents).map_err(|e| {
        AppError::Generic(format!("Invalid JSON in {}: {}", config_path.display(), e))
    })?;

    let removed = root
        .get_mut("mcpServers")
        .and_then(|servers| servers.as_object_mut())
        .map(|servers| {
            let wenlan = servers.remove(MCP_SERVER_KEY).is_some();
            let legacy = servers.remove(LEGACY_MCP_SERVER_KEY).is_some();
            wenlan || legacy
        })
        .unwrap_or(false);

    if !removed {
        return Err(AppError::Generic(
            "No Wenlan MCP entry found to remove".into(),
        ));
    }

    let backup_path = config_path.with_extension("json.bak");
    std::fs::copy(config_path, &backup_path)?;
    let formatted =
        serde_json::to_string_pretty(&root).map_err(|e| AppError::Generic(e.to_string()))?;
    std::fs::write(config_path, formatted)?;
    Ok(())
}

/// TOML variant for Codex CLI (`[mcp_servers.*]` tables) — mirrors
/// `remove_wenlan_entry`'s contract and `has_configured_entry_toml`'s key set,
/// using the same format-preserving `toml_edit` round-trip
/// `write_wenlan_entry_toml` writes with.
pub fn remove_wenlan_entry_toml(config_path: &std::path::Path) -> Result<(), AppError> {
    use toml_edit::DocumentMut;

    if !config_path.exists() {
        return Err(AppError::Generic(
            "No config file found — nothing to remove".into(),
        ));
    }
    let contents = std::fs::read_to_string(config_path)?;
    let mut doc: DocumentMut = contents.parse().map_err(|e| {
        AppError::Generic(format!("Invalid TOML in {}: {}", config_path.display(), e))
    })?;

    let removed = doc
        .get_mut("mcp_servers")
        .and_then(|servers| servers.as_table_like_mut())
        .map(|servers| {
            let wenlan = servers.remove(MCP_SERVER_KEY).is_some();
            let legacy = servers.remove(LEGACY_MCP_SERVER_KEY).is_some();
            wenlan || legacy
        })
        .unwrap_or(false);

    if !removed {
        return Err(AppError::Generic(
            "No Wenlan MCP entry found to remove".into(),
        ));
    }

    let backup_path = config_path.with_extension("toml.bak");
    std::fs::copy(config_path, &backup_path)?;
    std::fs::write(config_path, doc.to_string())?;
    Ok(())
}

/// Remove ONLY the legacy `origin` `mcpServers` entry from a JSON client
/// config, keeping the live `wenlan` entry — the fix for the raw+raw
/// duplicate a no-plugin client (Cursor, Gemini CLI) lands in after the
/// rename. Critically different from `remove_wenlan_entry`, which drops both
/// keys: that is correct only where a plugin still provides the server, so
/// applying it here would delete the client's only working connection. Every
/// other server and unrelated key survives. A missing file, or one with no
/// `origin` entry, is `Err` (nothing to remove) — surfaced verbatim by the
/// caller. Backs the file up first (like `remove_wenlan_entry`), but only once
/// a removal is certain, so the no-op error path leaves no stray `.bak`.
pub fn remove_legacy_origin_entry(config_path: &std::path::Path) -> Result<(), AppError> {
    if !config_path.exists() {
        return Err(AppError::Generic(
            "No config file found — nothing to remove".into(),
        ));
    }
    let contents = std::fs::read_to_string(config_path)?;
    let mut root = serde_json::from_str::<serde_json::Value>(&contents).map_err(|e| {
        AppError::Generic(format!("Invalid JSON in {}: {}", config_path.display(), e))
    })?;

    let removed = root
        .get_mut("mcpServers")
        .and_then(|servers| servers.as_object_mut())
        .map(|servers| servers.remove(LEGACY_MCP_SERVER_KEY).is_some())
        .unwrap_or(false);

    if !removed {
        return Err(AppError::Generic(
            "No legacy origin MCP entry found to remove".into(),
        ));
    }

    let backup_path = config_path.with_extension("json.bak");
    std::fs::copy(config_path, &backup_path)?;
    let formatted =
        serde_json::to_string_pretty(&root).map_err(|e| AppError::Generic(e.to_string()))?;
    std::fs::write(config_path, formatted)?;
    Ok(())
}

/// TOML variant for Codex CLI (`[mcp_servers.*]` tables) — mirrors
/// `remove_legacy_origin_entry`'s contract (removes only `origin`, keeps
/// `wenlan`) using the same format-preserving `toml_edit` round-trip.
pub fn remove_legacy_origin_entry_toml(config_path: &std::path::Path) -> Result<(), AppError> {
    use toml_edit::DocumentMut;

    if !config_path.exists() {
        return Err(AppError::Generic(
            "No config file found — nothing to remove".into(),
        ));
    }
    let contents = std::fs::read_to_string(config_path)?;
    let mut doc: DocumentMut = contents.parse().map_err(|e| {
        AppError::Generic(format!("Invalid TOML in {}: {}", config_path.display(), e))
    })?;

    let removed = doc
        .get_mut("mcp_servers")
        .and_then(|servers| servers.as_table_like_mut())
        .map(|servers| servers.remove(LEGACY_MCP_SERVER_KEY).is_some())
        .unwrap_or(false);

    if !removed {
        return Err(AppError::Generic(
            "No legacy origin MCP entry found to remove".into(),
        ));
    }

    let backup_path = config_path.with_extension("toml.bak");
    std::fs::copy(config_path, &backup_path)?;
    std::fs::write(config_path, doc.to_string())?;
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
    fn test_check_already_configured_finds_legacy_origin() {
        let json =
            r#"{"mcpServers": {"origin": {"command": "npx", "args": ["-y", "origin-mcp"]}}}"#;
        assert!(has_configured_entry(json));
    }

    #[test]
    fn test_check_already_configured_finds_wenlan() {
        let json =
            r#"{"mcpServers": {"wenlan": {"command": "npx", "args": ["-y", "wenlan-mcp"]}}}"#;
        assert!(has_configured_entry(json));
    }

    #[test]
    fn test_check_already_configured_not_found() {
        let json = r#"{"mcpServers": {"other-server": {}}}"#;
        assert!(!has_configured_entry(json));
    }

    #[test]
    fn test_check_already_configured_no_mcp_servers_key() {
        let json = r#"{"theme": "dark"}"#;
        assert!(!has_configured_entry(json));
    }

    #[test]
    fn test_check_already_configured_invalid_json() {
        assert!(!has_configured_entry("not json"));
    }

    #[test]
    fn test_claude_code_plugin_enabled_matches_fresh_install_prefix() {
        // Fresh install: marketplace name is the short form.
        let json = r#"{"enabledPlugins": {"wenlan@7xuanlu": true}}"#;
        assert!(claude_code_plugin_enabled(json));
    }

    #[test]
    fn test_claude_code_plugin_enabled_matches_legacy_marketplace_name() {
        // A machine that added the old self-hosted marketplace before it was
        // deleted upstream (048d77a8) — must still match, since matching is
        // by the `wenlan@` prefix, not a literal marketplace name.
        let json = r#"{"enabledPlugins": {"wenlan@7xuanlu-wenlan": true}}"#;
        assert!(claude_code_plugin_enabled(json));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_when_disabled() {
        let json = r#"{"enabledPlugins": {"wenlan@7xuanlu": false}}"#;
        assert!(!claude_code_plugin_enabled(json));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_when_no_wenlan_entry() {
        let json = r#"{"enabledPlugins": {"other-plugin@somewhere": true}}"#;
        assert!(!claude_code_plugin_enabled(json));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_when_no_enabled_plugins_key() {
        let json = r#"{"theme": "dark"}"#;
        assert!(!claude_code_plugin_enabled(json));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_on_malformed_json() {
        assert!(!claude_code_plugin_enabled("not json"));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_matches_pre_rename_marketplace() {
        let toml = "[plugins.\"wenlan@wenlan-local\"]\nenabled = true\n";
        assert!(codex_cli_plugin_enabled(toml));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_matches_post_rename_marketplace() {
        // 7xuanlu/wenlan#348 renames the marketplace to match Claude's — must
        // still match, since matching is by the `wenlan@` prefix, not a
        // literal marketplace name.
        let toml = "[plugins.\"wenlan@7xuanlu-wenlan\"]\nenabled = true\n";
        assert!(codex_cli_plugin_enabled(toml));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_when_disabled() {
        let toml = "[plugins.\"wenlan@wenlan-local\"]\nenabled = false\n";
        assert!(!codex_cli_plugin_enabled(toml));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_when_no_wenlan_entry() {
        let toml = "[plugins.\"other@somewhere\"]\nenabled = true\n";
        assert!(!codex_cli_plugin_enabled(toml));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_when_no_plugins_key() {
        let toml = "model = \"gpt-5.5\"\n";
        assert!(!codex_cli_plugin_enabled(toml));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_on_malformed_toml() {
        assert!(!codex_cli_plugin_enabled("not toml ["));
    }

    /// Manifest fixture matching the real shape seen on a live machine:
    /// `wenlan` present, plus another entry whose `marketplaceName` ("My
    /// Uploads") deliberately differs from its `name` ("social-media-skills")
    /// — a name/marketplaceName mixup would false-positive on this fixture.
    fn manifest_with_wenlan() -> &'static str {
        r#"{"plugins": [
            {"id": "plugin_1", "name": "social-media-skills", "marketplaceId": "m1", "marketplaceName": "My Uploads"},
            {"id": "plugin_2", "name": "wenlan", "marketplaceId": "m2", "marketplaceName": "wenlan"}
        ]}"#
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_true_for_exact_name() {
        assert!(claude_desktop_plugin_enabled(manifest_with_wenlan()));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_for_similar_name() {
        // Guards against a `starts_with`/`contains` match instead of exact
        // equality: "wenlan-old" must not count as "wenlan".
        let json =
            r#"{"plugins": [{"id": "p1", "name": "wenlan-old", "marketplaceName": "wenlan"}]}"#;
        assert!(!claude_desktop_plugin_enabled(json));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_when_only_marketplace_name_matches() {
        // The entry's `name` is "other-plugin"; only `marketplaceName` says
        // "wenlan". Matching the wrong field would false-positive here.
        let json =
            r#"{"plugins": [{"id": "p1", "name": "other-plugin", "marketplaceName": "wenlan"}]}"#;
        assert!(!claude_desktop_plugin_enabled(json));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_case_mismatch() {
        let json = r#"{"plugins": [{"id": "p1", "name": "Wenlan"}]}"#;
        assert!(!claude_desktop_plugin_enabled(json));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_when_no_plugins_key() {
        assert!(!claude_desktop_plugin_enabled(r#"{"lastUpdated": 1}"#));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_on_malformed_json() {
        assert!(!claude_desktop_plugin_enabled("not json"));
    }

    #[test]
    fn test_claude_desktop_account_id_extracts_last_known_account_uuid() {
        let json = r#"{"lastKnownAccountUuid": "acct-123", "locale": "en-US"}"#;
        assert_eq!(
            claude_desktop_account_id(json),
            Some("acct-123".to_string())
        );
    }

    #[test]
    fn test_claude_desktop_account_id_none_when_key_missing() {
        assert_eq!(claude_desktop_account_id(r#"{"locale": "en-US"}"#), None);
    }

    #[test]
    fn test_claude_desktop_account_id_none_on_malformed_json() {
        assert_eq!(claude_desktop_account_id("not json"), None);
    }

    #[test]
    fn test_claude_desktop_account_sessions_dir_joins_expected_segments() {
        let support_dir = Path::new("/support");
        let dir = claude_desktop_account_sessions_dir(support_dir, "acct-1");
        assert_eq!(dir, Path::new("/support/local-agent-mode-sessions/acct-1"));
    }

    #[test]
    fn test_sessions_dir_true_when_one_session_has_wenlan() {
        let tmp = tempfile::tempdir().unwrap();
        let session_dir = tmp.path().join("sess-1").join("rpm");
        std::fs::create_dir_all(&session_dir).unwrap();
        std::fs::write(session_dir.join("manifest.json"), manifest_with_wenlan()).unwrap();
        assert!(claude_desktop_plugin_enabled_in_sessions_dir(tmp.path()));
    }

    #[test]
    fn test_sessions_dir_true_when_second_of_two_sessions_has_wenlan() {
        let tmp = tempfile::tempdir().unwrap();
        let no_wenlan = tmp.path().join("sess-a").join("rpm");
        std::fs::create_dir_all(&no_wenlan).unwrap();
        std::fs::write(
            no_wenlan.join("manifest.json"),
            r#"{"plugins": [{"id": "p1", "name": "engineering"}]}"#,
        )
        .unwrap();

        let with_wenlan = tmp.path().join("sess-b").join("rpm");
        std::fs::create_dir_all(&with_wenlan).unwrap();
        std::fs::write(with_wenlan.join("manifest.json"), manifest_with_wenlan()).unwrap();

        assert!(claude_desktop_plugin_enabled_in_sessions_dir(tmp.path()));
    }

    #[test]
    fn test_sessions_dir_false_when_dir_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!claude_desktop_plugin_enabled_in_sessions_dir(
            &tmp.path().join("does-not-exist")
        ));
    }

    #[test]
    fn test_sessions_dir_false_when_no_rpm_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let session_dir = tmp.path().join("sess-1");
        std::fs::create_dir_all(&session_dir).unwrap();
        // manifest.json exists but not under rpm/
        std::fs::write(session_dir.join("manifest.json"), manifest_with_wenlan()).unwrap();
        assert!(!claude_desktop_plugin_enabled_in_sessions_dir(tmp.path()));
    }

    #[test]
    fn test_sessions_dir_tolerates_malformed_manifest_alongside_a_valid_one() {
        let tmp = tempfile::tempdir().unwrap();
        let broken = tmp.path().join("sess-broken").join("rpm");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("manifest.json"), "not json").unwrap();

        let good = tmp.path().join("sess-good").join("rpm");
        std::fs::create_dir_all(&good).unwrap();
        std::fs::write(good.join("manifest.json"), manifest_with_wenlan()).unwrap();

        assert!(claude_desktop_plugin_enabled_in_sessions_dir(tmp.path()));
    }

    /// Builds `<tmp>/config.json` (with `lastKnownAccountUuid`) plus
    /// `<tmp>/local-agent-mode-sessions/<account_id>/<session_id>/rpm/manifest.json`
    /// — the exact shape verified on a live Claude Desktop install — so
    /// `claude_desktop_plugin_enabled_for_support_dir` is exercised
    /// end-to-end against a fake `support_dir`.
    fn write_support_dir_fixture(root: &Path, account_id: &str, session_id: &str, manifest: &str) {
        std::fs::write(
            root.join("config.json"),
            format!(r#"{{"lastKnownAccountUuid": "{account_id}"}}"#),
        )
        .unwrap();
        let rpm_dir = root
            .join("local-agent-mode-sessions")
            .join(account_id)
            .join(session_id)
            .join("rpm");
        std::fs::create_dir_all(&rpm_dir).unwrap();
        std::fs::write(rpm_dir.join("manifest.json"), manifest).unwrap();
    }

    #[test]
    fn test_support_dir_true_when_pinned_account_session_has_wenlan() {
        let tmp = tempfile::tempdir().unwrap();
        write_support_dir_fixture(tmp.path(), "acct-1", "sess-1", manifest_with_wenlan());
        assert!(claude_desktop_plugin_enabled_for_support_dir(tmp.path()));
    }

    #[test]
    fn test_support_dir_false_when_wenlan_only_under_a_different_account() {
        let tmp = tempfile::tempdir().unwrap();
        // config.json pins "acct-1", but the manifest with wenlan lives
        // under a *different* account id — must not count.
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"lastKnownAccountUuid": "acct-1"}"#,
        )
        .unwrap();
        let rpm_dir = tmp
            .path()
            .join("local-agent-mode-sessions")
            .join("acct-2")
            .join("sess-1")
            .join("rpm");
        std::fs::create_dir_all(&rpm_dir).unwrap();
        std::fs::write(rpm_dir.join("manifest.json"), manifest_with_wenlan()).unwrap();

        assert!(!claude_desktop_plugin_enabled_for_support_dir(tmp.path()));
    }

    #[test]
    fn test_support_dir_false_when_config_json_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!claude_desktop_plugin_enabled_for_support_dir(tmp.path()));
    }

    #[test]
    fn test_support_dir_never_reads_skills_plugin_sentinel() {
        // The `skills-plugin` sentinel sits alongside the real account-id
        // directory under `local-agent-mode-sessions/` on a real machine.
        // It is not a UUID, so it can never be `lastKnownAccountUuid` — a
        // manifest planted only under it must never count.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"lastKnownAccountUuid": "acct-1"}"#,
        )
        .unwrap();
        let sentinel_rpm = tmp
            .path()
            .join("local-agent-mode-sessions")
            .join("skills-plugin")
            .join("sess-1")
            .join("rpm");
        std::fs::create_dir_all(&sentinel_rpm).unwrap();
        std::fs::write(sentinel_rpm.join("manifest.json"), manifest_with_wenlan()).unwrap();

        assert!(!claude_desktop_plugin_enabled_for_support_dir(tmp.path()));
    }

    /// Live-machine sanity check, not part of the default gating suite (this
    /// machine's Claude Desktop state is not portable to CI or other dev
    /// machines) — run explicitly with `cargo test --lib -- --ignored
    /// claude_desktop_detected_via_real_plugin_manifest`. This machine's
    /// `rpm/manifest.json`, under the account pinned by
    /// `~/Library/Application Support/Claude/config.json`, lists a plugin
    /// named "wenlan" (verified by hand before writing this test). If
    /// `detect_mcp_clients`'s `claude_desktop` branch is ever severed from
    /// `claude_desktop_plugin_enabled_on_disk`, this is the one test in this
    /// file that will catch it — everything else here exercises the logic
    /// against a fake `support_dir`, never the real one.
    #[test]
    #[ignore]
    fn claude_desktop_detected_via_real_plugin_manifest() {
        let claude_desktop = detect_mcp_clients()
            .into_iter()
            .find(|c| c.client_type == "claude_desktop")
            .expect("claude_desktop row always present");
        assert!(
            claude_desktop.already_configured,
            "expected already_configured=true: this machine has the Wenlan chat-side plugin installed"
        );
    }

    #[test]
    fn test_wenlan_mcp_entry_is_typed_command_args() {
        let entry = wenlan_mcp_entry();

        assert!(!entry.command.is_empty());
        if entry.command == "npx" {
            assert_eq!(entry.args.len(), 2);
            assert_eq!(entry.args[0], "-y");
            assert!(entry.args[1].starts_with("wenlan-mcp@^"));
        } else {
            assert!(entry.command.ends_with("wenlan-mcp"));
            assert!(entry.args.is_empty());
        }
    }

    /// The bug that broke a real machine: a maintainer's cargo target dir outranked
    /// the installed binary, so the absolute dev path was written into the user's
    /// client config and died on the next `cargo clean`.
    #[test]
    fn wenlan_mcp_candidates_never_probe_a_build_artifact_dir() {
        let home = PathBuf::from("/Users/someone");
        let candidates = wenlan_mcp_candidates(
            Some(home.as_path()),
            None,
            Some(Path::new("/Applications/Wenlan.app/Contents/MacOS")),
        );
        assert!(!candidates.is_empty());
        for candidate in &candidates {
            let path = candidate.to_string_lossy();
            assert!(
                !path.contains("/target/release/") && !path.contains("/target/debug/"),
                "candidate probes a cargo build artifact, which is not an install location: {path}"
            );
            assert!(
                !path.contains("/Repos/"),
                "candidate hardcodes a maintainer's checkout layout: {path}"
            );
        }
    }

    #[test]
    fn wenlan_mcp_candidates_rank_the_installed_binary_first() {
        let home = PathBuf::from("/Users/someone");
        let candidates = wenlan_mcp_candidates(Some(home.as_path()), None, None);
        assert_eq!(
            candidates.first().unwrap(),
            &home.join(".wenlan/bin/wenlan-mcp")
        );
        assert!(candidates.contains(&home.join(".cargo/bin/wenlan-mcp")));
    }

    #[test]
    fn wenlan_mcp_candidates_let_a_dev_override_win() {
        let home = PathBuf::from("/Users/someone");
        let candidates = wenlan_mcp_candidates(
            Some(home.as_path()),
            Some("/tmp/dev/wenlan-mcp"),
            Some(Path::new("/Applications/Wenlan.app/Contents/MacOS")),
        );
        assert_eq!(candidates[0], PathBuf::from("/tmp/dev/wenlan-mcp"));
        assert_eq!(candidates[1], home.join(".wenlan/bin/wenlan-mcp"));
        assert_eq!(
            candidates[2],
            PathBuf::from("/Applications/Wenlan.app/Contents/MacOS/wenlan-mcp")
        );
    }

    #[test]
    fn wenlan_mcp_candidates_survive_a_missing_home_and_empty_override() {
        let candidates = wenlan_mcp_candidates(
            None,
            Some("   "),
            Some(Path::new("/Applications/Wenlan.app/Contents/MacOS")),
        );
        assert_eq!(
            candidates,
            vec![PathBuf::from(
                "/Applications/Wenlan.app/Contents/MacOS/wenlan-mcp"
            )]
        );
    }

    #[test]
    fn pinned_wenlan_mcp_package_tracks_the_backend_pin_file() {
        assert_eq!(
            pinned_wenlan_mcp_package("v0.13.0\ndeadbeef\n"),
            "wenlan-mcp@^0.13.0"
        );
        assert_eq!(pinned_wenlan_mcp_package("0.12.0"), "wenlan-mcp@^0.12.0");
    }

    #[test]
    fn pinned_wenlan_mcp_package_falls_back_when_the_pin_is_unparseable() {
        assert_eq!(pinned_wenlan_mcp_package(""), "wenlan-mcp");
        assert_eq!(pinned_wenlan_mcp_package("latest\n"), "wenlan-mcp");
    }

    /// The npx fallback must carry the version this app was built against, or a
    /// `.dmg`-only user silently gets whatever backend npm serves today.
    #[test]
    fn npx_fallback_is_pinned_to_the_shipped_backend_version() {
        let entry = wenlan_mcp_entry_for(None, &pinned_wenlan_mcp_package(BACKEND_VERSION_PIN));
        assert_eq!(entry.command, "npx");
        assert_eq!(entry.args[0], "-y");
        assert!(
            entry.args[1].starts_with("wenlan-mcp@^"),
            "npx fallback is unpinned: {}",
            entry.args[1]
        );
        assert!(
            entry.args[1]
                .trim_start_matches("wenlan-mcp@^")
                .starts_with(|c: char| c.is_ascii_digit()),
            "npx fallback carries no version: {}",
            entry.args[1]
        );
    }

    #[test]
    fn wenlan_mcp_entry_prefers_a_found_binary_over_npx() {
        let entry = wenlan_mcp_entry_for(
            Some(PathBuf::from("/Users/someone/.wenlan/bin/wenlan-mcp")),
            "wenlan-mcp@^9.9.9",
        );
        assert_eq!(entry.command, "/Users/someone/.wenlan/bin/wenlan-mcp");
        assert!(entry.args.is_empty());
    }

    #[test]
    fn test_write_wenlan_entry_creates_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        write_wenlan_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(
            parsed["mcpServers"]["wenlan"],
            serde_json::to_value(wenlan_mcp_entry()).unwrap()
        );
        // Command is either a local binary path or npx (depending on host)
        let cmd = parsed["mcpServers"]["wenlan"]["command"].as_str().unwrap();
        assert!(
            cmd == "npx" || cmd.ends_with("wenlan-mcp"),
            "expected npx or path to wenlan-mcp, got: {cmd}"
        );
        assert!(parsed["mcpServers"]["origin"].is_null());
    }

    #[test]
    fn test_write_wenlan_entry_preserves_existing_servers() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing = r#"{"mcpServers": {"other": {"command": "other-cmd"}}}"#;
        std::fs::write(&config_path, existing).unwrap();
        write_wenlan_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(parsed["mcpServers"]["other"].is_object());
        assert!(parsed["mcpServers"]["wenlan"].is_object());
    }

    #[test]
    fn test_write_wenlan_entry_preserves_legacy_origin_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing =
            r#"{"mcpServers": {"origin": {"command": "npx", "args": ["-y", "origin-mcp"]}}}"#;
        std::fs::write(&config_path, existing).unwrap();
        write_wenlan_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(
            parsed["mcpServers"]["origin"]["args"],
            serde_json::json!(["-y", "origin-mcp"])
        );
        assert!(parsed["mcpServers"]["wenlan"].is_object());
    }

    #[test]
    fn test_write_wenlan_entry_creates_mcp_servers_key() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, r#"{"theme": "dark"}"#).unwrap();
        write_wenlan_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["theme"], "dark");
        assert!(parsed["mcpServers"]["wenlan"].is_object());
    }

    #[test]
    fn test_write_wenlan_entry_creates_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, r#"{"original": true}"#).unwrap();
        write_wenlan_entry(&config_path, false).unwrap();
        let backup = tmp.path().join("config.json.bak");
        assert!(backup.exists());
        let backup_contents = std::fs::read_to_string(&backup).unwrap();
        assert!(backup_contents.contains("original"));
    }

    #[test]
    fn test_write_wenlan_entry_errors_on_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, "not valid json").unwrap();
        let result = write_wenlan_entry(&config_path, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_wenlan_entry_refuses_create_for_claude_code() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("claude.json");
        // is_claude_code = true, file doesn't exist → should error
        let result = write_wenlan_entry(&config_path, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_client_config_path_gemini_cli() {
        let path = client_config_path("gemini_cli").unwrap();
        assert!(path.to_string_lossy().ends_with(".gemini/settings.json"));
    }

    #[test]
    fn test_client_config_path_codex_cli() {
        let path = client_config_path("codex_cli").unwrap();
        assert!(path.to_string_lossy().ends_with(".codex/config.toml"));
    }

    /// `exists` that answers true for exactly one path — so a test failure
    /// means the probed path is wrong, not merely that some boolean was false.
    fn only(hit: &str) -> impl Fn(&Path) -> bool + '_ {
        move |p: &Path| p == Path::new(hit)
    }

    #[test]
    fn codex_cli_detected_finds_chatgpt_in_applications() {
        let home = PathBuf::from("/Users/someone");
        assert!(codex_cli_detected(
            false,
            Some(&home),
            only("/Applications/ChatGPT.app")
        ));
    }

    #[test]
    fn codex_cli_detected_finds_chatgpt_in_user_applications() {
        let home = PathBuf::from("/Users/someone");
        assert!(codex_cli_detected(
            false,
            Some(&home),
            only("/Users/someone/Applications/ChatGPT.app")
        ));
    }

    #[test]
    fn codex_cli_detected_via_config_when_chatgpt_absent() {
        let home = PathBuf::from("/Users/someone");
        assert!(codex_cli_detected(true, Some(&home), |_| false));
    }

    #[test]
    fn codex_cli_not_detected_when_neither_present() {
        let home = PathBuf::from("/Users/someone");
        assert!(!codex_cli_detected(false, Some(&home), |_| false));
        // A *different* Mac app must not be mistaken for ChatGPT desktop.
        assert!(!codex_cli_detected(
            false,
            Some(&home),
            only("/Applications/Cursor.app")
        ));
    }

    #[test]
    fn codex_cli_detected_survives_missing_home() {
        assert!(codex_cli_detected(
            false,
            None,
            only("/Applications/ChatGPT.app")
        ));
    }

    #[test]
    fn test_detect_mcp_clients_has_exactly_one_codex_cli_row() {
        // ChatGPT desktop shares ~/.codex/config.toml with Codex CLI — it
        // must fold into the existing codex_cli row, never add a second row.
        let codex_rows: Vec<_> = detect_mcp_clients()
            .into_iter()
            .filter(|c| c.client_type == "codex_cli")
            .collect();
        assert_eq!(
            codex_rows.len(),
            1,
            "ChatGPT.app detection must reuse the codex_cli row, not add a second one"
        );
    }

    #[test]
    fn test_detect_includes_new_clients() {
        let types: Vec<String> = detect_mcp_clients()
            .into_iter()
            .map(|c| c.client_type)
            .collect();
        for expected in [
            "cursor",
            "claude_code",
            "claude_desktop",
            "gemini_cli",
            "codex_cli",
        ] {
            assert!(types.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn test_has_configured_entry_toml() {
        assert!(has_configured_entry_toml(
            "[mcp_servers.wenlan]\ncommand = \"npx\"\nargs = [\"-y\", \"wenlan-mcp\"]\n"
        ));
        assert!(has_configured_entry_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n"
        ));
        assert!(!has_configured_entry_toml(
            "[mcp_servers.other]\ncommand = \"x\"\n"
        ));
        assert!(!has_configured_entry_toml("model = \"gpt-5.5\"\n"));
        assert!(!has_configured_entry_toml("not toml ["));
    }

    #[test]
    fn test_write_wenlan_entry_toml_creates_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        write_wenlan_entry_toml(&config_path).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        assert!(has_configured_entry_toml(&contents));
        let parsed: toml::Value = toml::from_str(&contents).unwrap();
        let wenlan = &parsed["mcp_servers"]["wenlan"];
        assert!(wenlan.get("command").is_some());
    }

    #[test]
    fn test_write_wenlan_entry_toml_preserves_formatting_byte_for_byte() {
        // Council change (d): a user's hand-edited config must survive the
        // upsert byte-for-byte — comments, spacing, key order, other tables.
        let fixture = r#"# my codex config — do not touch
model = "gpt-5.5"   # inline comment

[profiles.fast]
model   = "gpt-5.5-mini"

[mcp_servers.other]
command = "other-cmd"  # keep me
args = ["--flag"]
"#;
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, fixture).unwrap();
        write_wenlan_entry_toml(&config_path).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        // Everything that existed before is preserved verbatim; the wenlan
        // table is appended after it.
        assert!(
            contents.starts_with(fixture),
            "existing content was reformatted:\n{contents}"
        );
        assert!(has_configured_entry_toml(&contents));
    }

    #[test]
    fn test_write_wenlan_entry_toml_upsert_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        write_wenlan_entry_toml(&config_path).unwrap();
        let first = std::fs::read_to_string(&config_path).unwrap();
        write_wenlan_entry_toml(&config_path).unwrap();
        let second = std::fs::read_to_string(&config_path).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn test_write_wenlan_entry_toml_creates_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, "model = \"gpt-5.5\"\n").unwrap();
        write_wenlan_entry_toml(&config_path).unwrap();
        let backup = tmp.path().join("config.toml.bak");
        assert!(backup.exists());
        assert!(std::fs::read_to_string(&backup)
            .unwrap()
            .contains("gpt-5.5"));
    }

    #[test]
    fn test_write_wenlan_entry_toml_errors_on_invalid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, "not toml [").unwrap();
        assert!(write_wenlan_entry_toml(&config_path).is_err());
    }

    // ── remove_wenlan_entry (JSON) ──────────────────────────────────────

    #[test]
    fn test_remove_wenlan_entry_removes_only_the_wenlan_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing =
            r#"{"mcpServers": {"wenlan": {"command": "npx"}, "other": {"command": "other-cmd"}}}"#;
        std::fs::write(&config_path, existing).unwrap();

        remove_wenlan_entry(&config_path).unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert!(parsed["mcpServers"]["wenlan"].is_null());
        // The sibling server survives untouched.
        assert_eq!(parsed["mcpServers"]["other"]["command"], "other-cmd");
    }

    #[test]
    fn test_remove_wenlan_entry_preserves_unrelated_structure() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing = r#"{"theme": "dark", "mcpServers": {"wenlan": {"command": "npx"}}}"#;
        std::fs::write(&config_path, existing).unwrap();

        remove_wenlan_entry(&config_path).unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(parsed["theme"], "dark");
        assert!(parsed["mcpServers"]["wenlan"].is_null());
    }

    #[test]
    fn test_remove_wenlan_entry_removes_legacy_origin() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing =
            r#"{"mcpServers": {"origin": {"command": "npx", "args": ["-y", "origin-mcp"]}}}"#;
        std::fs::write(&config_path, existing).unwrap();

        remove_wenlan_entry(&config_path).unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert!(parsed["mcpServers"]["origin"].is_null());
    }

    #[test]
    fn test_remove_wenlan_entry_errs_when_no_entry_present() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, r#"{"mcpServers": {"other": {}}}"#).unwrap();
        assert!(remove_wenlan_entry(&config_path).is_err());
        // No-op error path leaves no stray backup behind.
        assert!(!config_path.with_extension("json.bak").exists());
    }

    #[test]
    fn test_remove_wenlan_entry_errs_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("does-not-exist.json");
        assert!(remove_wenlan_entry(&config_path).is_err());
    }

    #[test]
    fn test_remove_wenlan_entry_leaves_client_config_has_raw_entry_false() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing =
            r#"{"mcpServers": {"wenlan": {"command": "npx"}, "other": {"command": "x"}}}"#;
        std::fs::write(&config_path, existing).unwrap();
        // Precondition: detection sees the raw entry before removal.
        assert!(client_config_has_raw_entry("cursor", &config_path));

        remove_wenlan_entry(&config_path).unwrap();

        // The written file still parses and detection no longer sees an entry.
        assert!(!client_config_has_raw_entry("cursor", &config_path));
    }

    // ── remove_wenlan_entry_toml (Codex CLI) ────────────────────────────

    #[test]
    fn test_remove_wenlan_entry_toml_removes_only_the_wenlan_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        let fixture = r#"# my codex config
model = "gpt-5.5"

[mcp_servers.other]
command = "other-cmd"

[mcp_servers.wenlan]
command = "npx"
args = ["-y", "wenlan-mcp"]
"#;
        std::fs::write(&config_path, fixture).unwrap();

        remove_wenlan_entry_toml(&config_path).unwrap();

        let contents = std::fs::read_to_string(&config_path).unwrap();
        // The wenlan entry is gone; the sibling server and unrelated keys stay.
        assert!(!has_configured_entry_toml(&contents));
        let parsed: toml::Value = toml::from_str(&contents).unwrap();
        assert_eq!(parsed["model"], toml::Value::from("gpt-5.5"));
        assert_eq!(
            parsed["mcp_servers"]["other"]["command"],
            toml::Value::from("other-cmd")
        );
        assert!(parsed["mcp_servers"].get("wenlan").is_none());
    }

    #[test]
    fn test_remove_wenlan_entry_toml_errs_when_no_entry_present() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, "model = \"gpt-5.5\"\n").unwrap();
        assert!(remove_wenlan_entry_toml(&config_path).is_err());
        assert!(!config_path.with_extension("toml.bak").exists());
    }

    #[test]
    fn test_remove_wenlan_entry_toml_errs_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("does-not-exist.toml");
        assert!(remove_wenlan_entry_toml(&config_path).is_err());
    }

    #[test]
    fn test_remove_wenlan_entry_toml_leaves_client_config_has_raw_entry_false() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[mcp_servers.wenlan]\ncommand = \"npx\"\nargs = [\"-y\", \"wenlan-mcp\"]\n",
        )
        .unwrap();
        assert!(client_config_has_raw_entry("codex_cli", &config_path));

        remove_wenlan_entry_toml(&config_path).unwrap();

        assert!(!client_config_has_raw_entry("codex_cli", &config_path));
    }

    // ── has_both_raw_entries (raw+raw duplicate detection) ──────────────

    #[test]
    fn test_has_both_raw_entries_true_when_wenlan_and_origin_present() {
        // The real ~/.cursor/mcp.json shape on this machine.
        let json = r#"{"mcpServers": {
            "origin": {"command": "npx", "args": ["-y", "origin-mcp"]},
            "wenlan": {"command": "npx", "args": ["-y", "wenlan-mcp"]}
        }}"#;
        assert!(has_both_raw_entries(json));
    }

    #[test]
    fn test_has_both_raw_entries_false_when_only_wenlan() {
        let json = r#"{"mcpServers": {"wenlan": {"command": "npx"}}}"#;
        assert!(!has_both_raw_entries(json));
    }

    #[test]
    fn test_has_both_raw_entries_false_when_only_origin() {
        let json = r#"{"mcpServers": {"origin": {"command": "npx"}}}"#;
        assert!(!has_both_raw_entries(json));
    }

    #[test]
    fn test_has_both_raw_entries_false_when_neither() {
        assert!(!has_both_raw_entries(r#"{"mcpServers": {"other": {}}}"#));
        assert!(!has_both_raw_entries(r#"{"theme": "dark"}"#));
        assert!(!has_both_raw_entries("not json"));
    }

    #[test]
    fn test_has_both_raw_entries_toml() {
        assert!(has_both_raw_entries_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n[mcp_servers.wenlan]\ncommand = \"npx\"\n"
        ));
        assert!(!has_both_raw_entries_toml(
            "[mcp_servers.wenlan]\ncommand = \"npx\"\n"
        ));
        assert!(!has_both_raw_entries_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n"
        ));
        assert!(!has_both_raw_entries_toml("model = \"gpt-5.5\"\n"));
        assert!(!has_both_raw_entries_toml("not toml ["));
    }

    /// HEADLINE (a): a raw+raw duplicate on a no-plugin client (cursor) IS
    /// flagged through the public detector, and neither single-entry case is.
    #[test]
    fn test_client_config_has_both_raw_entries_flags_cursor_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let both = tmp.path().join("both.json");
        std::fs::write(
            &both,
            r#"{"mcpServers": {"origin": {"command": "npx"}, "wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        assert!(client_config_has_both_raw_entries("cursor", &both));

        let only_wenlan = tmp.path().join("only_wenlan.json");
        std::fs::write(
            &only_wenlan,
            r#"{"mcpServers": {"wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        assert!(!client_config_has_both_raw_entries("cursor", &only_wenlan));

        // A file that doesn't exist has no duplicate.
        assert!(!client_config_has_both_raw_entries(
            "cursor",
            &tmp.path().join("missing.json")
        ));
    }

    #[test]
    fn test_client_config_has_both_raw_entries_toml_for_codex() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[mcp_servers.origin]\ncommand = \"npx\"\n[mcp_servers.wenlan]\ncommand = \"npx\"\n",
        )
        .unwrap();
        assert!(client_config_has_both_raw_entries(
            "codex_cli",
            &config_path
        ));
    }

    // ── remove_legacy_origin_entry (removes origin, keeps wenlan) ────────

    /// HEADLINE (b): the fix removes `origin` and KEEPS `wenlan`. Mutating
    /// `remove_legacy_origin_entry` to also drop `wenlan` fails this test.
    #[test]
    fn test_remove_legacy_origin_entry_keeps_wenlan() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let existing = r#"{"mcpServers": {
            "origin": {"command": "npx", "args": ["-y", "origin-mcp"]},
            "wenlan": {"command": "npx", "args": ["-y", "wenlan-mcp"]},
            "other": {"command": "other-cmd"}
        }}"#;
        std::fs::write(&config_path, existing).unwrap();

        remove_legacy_origin_entry(&config_path).unwrap();

        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        // origin is gone; wenlan and the sibling server stay.
        assert!(parsed["mcpServers"]["origin"].is_null());
        assert!(
            parsed["mcpServers"]["wenlan"].is_object(),
            "the live wenlan entry must survive — removing it would sever the client's connection"
        );
        assert_eq!(parsed["mcpServers"]["other"]["command"], "other-cmd");
    }

    #[test]
    fn test_remove_legacy_origin_entry_clears_the_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"mcpServers": {"origin": {"command": "npx"}, "wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        assert!(client_config_has_both_raw_entries("cursor", &config_path));

        remove_legacy_origin_entry(&config_path).unwrap();

        // The duplicate is resolved, and a single wenlan entry remains.
        assert!(!client_config_has_both_raw_entries("cursor", &config_path));
        assert!(client_config_has_raw_entry("cursor", &config_path));
    }

    #[test]
    fn test_remove_legacy_origin_entry_creates_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"mcpServers": {"origin": {"command": "npx"}, "wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();

        remove_legacy_origin_entry(&config_path).unwrap();

        let backup = tmp.path().join("config.json.bak");
        assert!(backup.exists());
        assert!(std::fs::read_to_string(&backup).unwrap().contains("origin"));
    }

    #[test]
    fn test_remove_legacy_origin_entry_errs_when_only_wenlan_present() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"mcpServers": {"wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        assert!(remove_legacy_origin_entry(&config_path).is_err());
        // No-op error path leaves no stray backup behind.
        assert!(!config_path.with_extension("json.bak").exists());
    }

    #[test]
    fn test_remove_legacy_origin_entry_errs_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(remove_legacy_origin_entry(&tmp.path().join("nope.json")).is_err());
    }

    #[test]
    fn test_remove_legacy_origin_entry_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"mcpServers": {"origin": {"command": "npx"}, "wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        remove_legacy_origin_entry(&config_path).unwrap();
        // Second run: origin already gone, so it's an Err (nothing to remove),
        // and wenlan is left untouched.
        assert!(remove_legacy_origin_entry(&config_path).is_err());
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert!(parsed["mcpServers"]["wenlan"].is_object());
    }

    // ── remove_legacy_origin_entry_toml (Codex CLI) ─────────────────────

    #[test]
    fn test_remove_legacy_origin_entry_toml_keeps_wenlan() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        let fixture = r#"# my codex config
model = "gpt-5.5"

[mcp_servers.origin]
command = "npx"
args = ["-y", "origin-mcp"]

[mcp_servers.wenlan]
command = "npx"
args = ["-y", "wenlan-mcp"]
"#;
        std::fs::write(&config_path, fixture).unwrap();

        remove_legacy_origin_entry_toml(&config_path).unwrap();

        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: toml::Value = toml::from_str(&contents).unwrap();
        assert!(parsed["mcp_servers"].get("origin").is_none());
        assert!(
            parsed["mcp_servers"].get("wenlan").is_some(),
            "the live wenlan entry must survive"
        );
        assert_eq!(parsed["model"], toml::Value::from("gpt-5.5"));
        assert!(client_config_has_raw_entry("codex_cli", &config_path));
        assert!(!client_config_has_both_raw_entries(
            "codex_cli",
            &config_path
        ));
    }

    #[test]
    fn test_remove_legacy_origin_entry_toml_creates_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(
            &config_path,
            "[mcp_servers.origin]\ncommand = \"npx\"\n[mcp_servers.wenlan]\ncommand = \"npx\"\n",
        )
        .unwrap();
        remove_legacy_origin_entry_toml(&config_path).unwrap();
        assert!(config_path.with_extension("toml.bak").exists());
    }

    #[test]
    fn test_remove_legacy_origin_entry_toml_errs_when_only_wenlan_present() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, "[mcp_servers.wenlan]\ncommand = \"npx\"\n").unwrap();
        assert!(remove_legacy_origin_entry_toml(&config_path).is_err());
        assert!(!config_path.with_extension("toml.bak").exists());
    }

    #[test]
    fn test_remove_legacy_origin_entry_toml_errs_when_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(remove_legacy_origin_entry_toml(&tmp.path().join("nope.toml")).is_err());
    }
}
