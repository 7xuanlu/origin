// SPDX-License-Identifier: AGPL-3.0-only
//! `wire_state()` — the real, resolved wiring of Wenlan on this machine:
//! whether the daemon answers, which `wenlan-mcp` binary would actually be
//! written into a client config (plus the full candidate trail, missing
//! paths included), and per-client MCP routing. Single source of truth
//! behind the onboarding wizard's "Setting up" step and Settings →
//! Diagnostics.
//!
//! The bug this exists to make visible: the app once resolved `wenlan-mcp`
//! to a maintainer's cargo build output, wrote that absolute path into a
//! user's `claude_desktop_config.json`, and the binary was later deleted by
//! `cargo clean`. Claude Desktop failed with "cannot connect mcpserver
//! wenlan" and nothing in the app surfaced it. A trail that only lists paths
//! that exist can't show a missing one — so `mcp_binary_wire` below keeps
//! every candidate `mcp_config::wenlan_mcp_candidate_sources` returns,
//! existent or not.

use crate::api::WenlanClient;
use crate::mcp_config;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireState {
    pub daemon: DaemonWire,
    pub mcp_binary: BinaryWire,
    pub clients: Vec<ClientWire>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonWire {
    pub base_url: String,
    pub reachable: bool,
    pub version: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryWire {
    pub command: String,
    pub args: Vec<String>,
    pub candidates: Vec<BinaryCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryCandidate {
    pub path: String,
    pub exists: bool,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientWire {
    pub client_type: String,
    pub name: String,
    pub detected: bool,
    pub config_path: String,
    pub has_raw_entry: bool,
    /// Config holds BOTH a `wenlan` and a legacy `origin` raw entry — the
    /// raw+raw duplicate. Load-bearing for no-plugin clients (cursor,
    /// gemini_cli): they never trip the plugin+raw double-registration
    /// (`has_plugin && has_raw_entry`), so this is the only way their
    /// duplicate surfaces, and it routes to a fix that removes only `origin`.
    pub has_raw_duplicate: bool,
    pub has_plugin: bool,
    pub route: String,
}

/// `route` for a client — what setup would do *now*, not what is already there.
///
/// The plugin always wins over a raw MCP entry, because every Wenlan plugin
/// declares its own `mcpServers`: writing `~/.claude.json` /
/// `[mcp_servers.wenlan]` / `claude_desktop_config.json` as well registers the
/// server twice.
///
/// `has_plugin` is load-bearing here, not decoration. Claude Desktop has two
/// independent plugin surfaces, and its *chat*-side plugin ships an MCP server
/// of its own — so a Desktop that already has the plugin must be `"skip"`, not
/// `"config"`. Routing it to `"config"` is exactly the double-registration that
/// broke a real machine (see commit 5d7a364); `claude_code` / `codex_cli` reach
/// the same conclusion via their own `"plugin"` arm.
fn route_for(client_type: &str, detected: bool, has_plugin: bool) -> &'static str {
    if !detected {
        "skip"
    } else if client_type == "claude_code" || client_type == "codex_cli" {
        "plugin"
    } else if has_plugin {
        "skip"
    } else {
        "config"
    }
}

/// Builds the daemon's wire state from an already-attempted health check —
/// pure, so it's testable without a live daemon. Always succeeds: an
/// unreachable daemon is `reachable: false` with `error` set, never a
/// propagated `Err` — surfacing exactly that without crashing is
/// `wire_state`'s whole point.
fn daemon_wire_for(base_url: String, health: Result<String, String>) -> DaemonWire {
    match health {
        Ok(version) => DaemonWire {
            base_url,
            reachable: true,
            version: Some(version),
            error: None,
        },
        Err(error) => DaemonWire {
            base_url,
            reachable: false,
            version: None,
            error: Some(error),
        },
    }
}

/// Builds the mcp-binary wire state: the resolved `entry` (what setup would
/// actually write) plus the full candidate trail from
/// `mcp_config::wenlan_mcp_candidate_sources` — the same function
/// `mcp_config::wenlan_mcp_candidates` resolves `entry` from, so the two can
/// never disagree. `exists` is injected so a candidate that doesn't exist on
/// disk still appears in the trail with `exists: false` instead of being
/// silently dropped — that's the one thing this whole command exists to
/// show.
fn mcp_binary_wire(
    home: Option<&Path>,
    dev_bin: Option<&str>,
    exe_dir: Option<&Path>,
    exists: impl Fn(&Path) -> bool,
    entry: mcp_config::WenlanMcpEntry,
) -> BinaryWire {
    let candidates = mcp_config::wenlan_mcp_candidate_sources(home, dev_bin, exe_dir)
        .into_iter()
        .map(|(path, source)| BinaryCandidate {
            exists: exists(path.as_path()),
            path: path.to_string_lossy().to_string(),
            source: source.to_string(),
        })
        .collect();
    BinaryWire {
        command: entry.command,
        args: entry.args,
        candidates,
    }
}

/// Builds one client's wire state from `mcp_config::detect_mcp_clients()`'s
/// output plus the plugin/raw-entry breakdown `detect_mcp_clients` folds
/// into a single `already_configured` bool.
fn client_wire(client: &mcp_config::McpClient) -> ClientWire {
    let config_path = Path::new(&client.config_path);
    let has_plugin = mcp_config::client_plugin_enabled(&client.client_type);
    ClientWire {
        route: route_for(&client.client_type, client.detected, has_plugin).to_string(),
        client_type: client.client_type.clone(),
        name: client.name.clone(),
        detected: client.detected,
        has_raw_entry: mcp_config::client_config_has_raw_entry(&client.client_type, config_path),
        has_raw_duplicate: mcp_config::client_config_has_both_raw_entries(
            &client.client_type,
            config_path,
        ),
        has_plugin,
        config_path: client.config_path.clone(),
    }
}

/// Assembles the full `WireState` for the real machine: reads `HOME`,
/// `WENLAN_MCP_DEV_BIN`, and the running executable's directory exactly as
/// `mcp_config::wenlan_mcp_entry()` does internally, so the binary trail and
/// the resolved command describe the same candidates. Never errors — a down
/// daemon shows up as `daemon.reachable: false`, not a failed command.
pub async fn compute(client: &WenlanClient) -> WireState {
    let home = dirs::home_dir();
    let dev_bin = std::env::var("WENLAN_MCP_DEV_BIN").ok();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));

    let health = client.health().await.map(|h| h.version);
    let daemon = daemon_wire_for(client.base_url().to_string(), health);

    let mcp_binary = mcp_binary_wire(
        home.as_deref(),
        dev_bin.as_deref(),
        exe_dir.as_deref(),
        |p| p.exists(),
        mcp_config::wenlan_mcp_entry(),
    );

    let clients = mcp_config::detect_mcp_clients()
        .iter()
        .map(client_wire)
        .collect();

    WireState {
        daemon,
        mcp_binary,
        clients,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── route_for ───────────────────────────────────────────────────────

    #[test]
    fn plugin_clients_route_to_plugin_never_config() {
        assert_eq!(route_for("claude_code", true, false), "plugin");
        assert_eq!(route_for("codex_cli", true, false), "plugin");
    }

    #[test]
    fn non_plugin_clients_route_to_config() {
        assert_eq!(route_for("cursor", true, false), "config");
        assert_eq!(route_for("claude_desktop", true, false), "config");
        assert_eq!(route_for("gemini_cli", true, false), "config");
    }

    #[test]
    fn undetected_clients_route_to_skip_regardless_of_type() {
        assert_eq!(route_for("claude_code", false, false), "skip");
        assert_eq!(route_for("cursor", false, false), "skip");
    }

    /// The bug that broke a real machine. Claude Desktop's *chat*-side plugin
    /// ships its own MCP server, so a Desktop that already has the plugin must
    /// not also be written a raw `claude_desktop_config.json` entry — that is
    /// the double registration. `route` has to see `has_plugin` to know this;
    /// a route computed from `(client_type, detected)` alone cannot.
    #[test]
    fn a_client_that_already_has_the_plugin_is_never_routed_to_config() {
        assert_eq!(route_for("claude_desktop", true, true), "skip");
        assert_eq!(route_for("cursor", true, true), "skip");
        // …and without the plugin it still needs its config written.
        assert_eq!(route_for("claude_desktop", true, false), "config");
    }

    // ── mcp_binary_wire ─────────────────────────────────────────────────

    fn fake_entry() -> mcp_config::WenlanMcpEntry {
        mcp_config::WenlanMcpEntry {
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "wenlan-mcp@^0.12.0".to_string()],
        }
    }

    #[test]
    fn candidate_trail_keeps_missing_paths_with_exists_false() {
        let home = PathBuf::from("/Users/someone");
        let wire = mcp_binary_wire(Some(&home), None, None, |_| false, fake_entry());

        // ~/.wenlan/bin/wenlan-mcp and ~/.cargo/bin/wenlan-mcp — neither exists.
        assert_eq!(wire.candidates.len(), 2);
        assert!(
            wire.candidates.iter().all(|c| !c.exists),
            "a candidate that `exists` says is missing must still be in the trail: {:?}",
            wire.candidates
        );
        assert!(wire
            .candidates
            .iter()
            .any(|c| c.path.ends_with(".wenlan/bin/wenlan-mcp") && c.source == "installed"));
        assert!(wire
            .candidates
            .iter()
            .any(|c| c.path.ends_with(".cargo/bin/wenlan-mcp") && c.source == "cargo"));
    }

    #[test]
    fn candidate_trail_reports_exists_true_when_the_probe_says_so() {
        let home = PathBuf::from("/Users/someone");
        let installed = home.join(".wenlan/bin/wenlan-mcp");
        let wire = mcp_binary_wire(
            Some(&home),
            None,
            None,
            move |p| p == installed,
            fake_entry(),
        );

        let installed_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "installed")
            .expect("installed candidate present");
        assert!(installed_candidate.exists);
        let cargo_candidate = wire
            .candidates
            .iter()
            .find(|c| c.source == "cargo")
            .expect("cargo candidate present");
        assert!(!cargo_candidate.exists);
    }

    #[test]
    fn mcp_binary_wire_carries_the_resolved_command_and_args_through() {
        let wire = mcp_binary_wire(None, None, None, |_| false, fake_entry());
        assert_eq!(wire.command, "npx");
        assert_eq!(wire.args, vec!["-y", "wenlan-mcp@^0.12.0"]);
    }

    // ── daemon_wire_for ─────────────────────────────────────────────────

    #[test]
    fn daemon_wire_for_reachable_carries_version_and_no_error() {
        let wire = daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            Ok("0.12.0".to_string()),
        );
        assert!(wire.reachable);
        assert_eq!(wire.version.as_deref(), Some("0.12.0"));
        assert!(wire.error.is_none());
    }

    #[test]
    fn daemon_wire_for_unreachable_carries_error_and_no_version() {
        let wire = daemon_wire_for(
            "http://127.0.0.1:7878".to_string(),
            Err("connection refused".to_string()),
        );
        assert!(!wire.reachable);
        assert!(wire.version.is_none());
        assert_eq!(wire.error.as_deref(), Some("connection refused"));
    }

    // ── compute() end to end against an unreachable daemon ─────────────

    /// Port 1 is a privileged port nothing listens on in this sandbox, so the
    /// connection is refused immediately — no live server needed, and no
    /// dependence on the real `WENLAN_PORT`/`ORIGIN_PORT` env vars (which
    /// `WenlanClient::new()` reads and which are process-global, shared with
    /// every other test in this binary).
    #[tokio::test]
    async fn compute_never_errors_when_the_daemon_is_unreachable() {
        let client = WenlanClient::with_base_url("http://127.0.0.1:1".to_string());
        let wire = compute(&client).await;

        assert!(!wire.daemon.reachable);
        assert!(wire.daemon.error.is_some());
        assert_eq!(wire.daemon.base_url, "http://127.0.0.1:1");
        // Reaching this line at all proves compute() returned instead of
        // panicking or being unreachable through a propagated Err — its
        // signature (`-> WireState`, not `-> Result<WireState, _>`) makes
        // that structurally true, and this test exercises the down-daemon
        // path that would have to trigger it.
    }

    // ── client_wire ──────────────────────────────────────────────────────

    #[test]
    fn client_wire_carries_detect_mcp_clients_fields_through() {
        let client = mcp_config::McpClient {
            name: "Cursor".to_string(),
            client_type: "cursor".to_string(),
            config_path: "/nonexistent/mcp.json".to_string(),
            detected: true,
            already_configured: false,
        };
        let wire = client_wire(&client);
        assert_eq!(wire.client_type, "cursor");
        assert_eq!(wire.name, "Cursor");
        assert!(wire.detected);
        assert_eq!(wire.config_path, "/nonexistent/mcp.json");
        assert_eq!(wire.route, "config");
        // A config file that doesn't exist has no raw entry.
        assert!(!wire.has_raw_entry);
        assert!(!wire.has_raw_duplicate);
        assert!(!wire.has_plugin);
    }

    /// A no-plugin client whose config carries both `wenlan` and `origin`
    /// surfaces `has_raw_duplicate: true` — the raw+raw case that plugin+raw
    /// detection can never reach for cursor/gemini_cli.
    #[test]
    fn client_wire_flags_raw_duplicate_for_a_no_plugin_client() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("mcp.json");
        std::fs::write(
            &config_path,
            r#"{"mcpServers": {"origin": {"command": "npx"}, "wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        let client = mcp_config::McpClient {
            name: "Cursor".to_string(),
            client_type: "cursor".to_string(),
            config_path: config_path.to_string_lossy().to_string(),
            detected: true,
            already_configured: true,
        };
        let wire = client_wire(&client);
        assert!(wire.has_raw_duplicate);
        assert!(wire.has_raw_entry);
        assert!(!wire.has_plugin);
    }
}
