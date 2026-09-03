// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A yes/no fact about an MCP client's install, in three values.
///
/// ROUND 5, DEFECT 4. Every one of these was a `bool`, and every one was
/// produced by `std::fs::read_to_string(..).map(..).unwrap_or(false)` or by
/// `Path::exists()` — both of which answer `false` for "measured: no" AND for
/// "the OS would not tell me". A `~/.claude/settings.json` that a permission
/// denies, holding an ENABLED Wenlan plugin, reported byte-for-byte identically
/// to one with the plugin switched off. Those are different situations: the
/// second is one the user can act on, and the first is one they have to be told
/// about — the wizard's "not configured" row invites them to write a SECOND
/// registration over a working one, which is the double-registration this
/// codebase already has a warning box for.
///
/// Same three values and the same rule as [`CandidateProbe`]: only a measured
/// absence is an absence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Reading {
    /// Measured: yes.
    Yes,
    /// Measured: no.
    No,
    /// Could not be measured. NOT a `No`.
    Unreadable { error: String },
}

impl Reading {
    fn of(value: bool) -> Self {
        if value {
            Reading::Yes
        } else {
            Reading::No
        }
    }

    /// True only for a MEASURED yes. Named `is_yes` rather than exposed as a
    /// bool field so a call site cannot treat "unreadable" as "no" by writing
    /// the shorter thing.
    pub fn is_yes(&self) -> bool {
        matches!(self, Reading::Yes)
    }

    /// OR over two readings of the same question, ranked so a failed read can
    /// never turn a measured yes into a no: `Yes` wins, then `Unreadable`, then
    /// `No`. A client counts as configured through EITHER a plugin or a raw
    /// entry, so if either half is a measured yes the answer is yes whatever
    /// happened to the other — and if neither is, an unread half means the
    /// answer is unknown, not "no".
    fn or(self, other: Reading) -> Reading {
        match (self, other) {
            (Reading::Yes, _) | (_, Reading::Yes) => Reading::Yes,
            (Reading::Unreadable { error }, _) | (_, Reading::Unreadable { error }) => {
                Reading::Unreadable { error }
            }
            (Reading::No, Reading::No) => Reading::No,
        }
    }
}

/// What reading one client config file actually answered.
///
/// Replaces the `Path::exists()` + `read_to_string(..).unwrap_or(false)` pair
/// that ran everywhere below. That pair had two collapses and a third problem:
/// `exists()` answers `false` when `metadata` fails for ANY reason, the read
/// answered `false` for every read error, and the two together were TWO
/// instants answering one question — the same shape as round 4's defect F.
/// One read, three answers.
enum ConfigRead {
    Contents(String),
    /// Measured absent.
    Absent,
    /// Could not look. NOT an absence.
    Unreadable(String),
}

fn read_config(path: &Path) -> ConfigRead {
    match std::fs::read_to_string(path) {
        Ok(contents) => ConfigRead::Contents(contents),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ConfigRead::Absent,
        Err(e) => ConfigRead::Unreadable(e.to_string()),
    }
}

impl ConfigRead {
    /// Whether the file is there at all — the honest replacement for
    /// `Path::exists()`, taken from the same read that answers the content
    /// questions so the two cannot come from different instants.
    fn present(&self) -> Reading {
        match self {
            ConfigRead::Contents(_) => Reading::Yes,
            ConfigRead::Absent => Reading::No,
            ConfigRead::Unreadable(error) => Reading::Unreadable {
                error: error.clone(),
            },
        }
    }

    /// Ask a yes/no question of the contents. An absent file is a measured
    /// `No` — it genuinely holds no entry. A file that could not be read is
    /// neither, and NEITHER IS ONE THAT COULD NOT BE PARSED.
    ///
    /// ROUND 6, DEFECT D2 — where the tri-state used to die. This took an
    /// `impl FnOnce(&str) -> bool`, and every question passed to it ended in
    /// `serde_json::from_str(..).ok()..unwrap_or(false)` or
    /// `parse::<DocumentMut>().ok()..unwrap_or(false)`. So a present Gemini
    /// `settings.json` holding `not json` produced `detected = Yes`,
    /// `has_raw_entry = No`, `has_plugin = No`, `already_configured = No`: the
    /// file WAS read, whether it holds an entry could NOT be measured, and the
    /// UI was told "no". The I/O layer above kept three values all the way here
    /// and then handed them to a function that only has two.
    ///
    /// The question is now fallible, and a body that would not parse comes back
    /// as `Unreadable` — a failed measurement, which is what it is.
    fn asks(&self, question: impl FnOnce(&str) -> Result<bool, String>) -> Reading {
        match self {
            ConfigRead::Contents(contents) => match question(contents) {
                Ok(answer) => Reading::of(answer),
                Err(error) => Reading::Unreadable { error },
            },
            ConfigRead::Absent => Reading::No,
            ConfigRead::Unreadable(error) => Reading::Unreadable {
                error: error.clone(),
            },
        }
    }
}

/// Parse a config body as JSON, or say why it could not be parsed.
///
/// The `Err` string is user-facing: it lands in `Reading::Unreadable { error }`
/// and from there in the "Setup state unknown" chip's detail and in the pasted
/// diagnostics report, so it has to name the FILE's problem rather than a
/// serde type name.
fn parse_json(body: &str) -> Result<serde_json::Value, String> {
    serde_json::from_str::<serde_json::Value>(body)
        .map_err(|e| format!("the file is not valid JSON ({e})"))
}

/// TOML counterpart of [`parse_json`]. Same contract, same reason.
fn parse_toml(body: &str) -> Result<toml_edit::DocumentMut, String> {
    body.parse::<toml_edit::DocumentMut>()
        .map_err(|e| format!("the file is not valid TOML ({e})"))
}

/// Is anything at `path`? Three answers from ONE `metadata` call.
/// `Path::exists()` is the two-answer version and gets the failure case wrong —
/// see [`CandidateProbe`], which exists for exactly this reason on the binary
/// side.
fn path_exists_reading(path: &Path) -> Reading {
    match std::fs::metadata(path) {
        Ok(_) => Reading::Yes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Reading::No,
        Err(e) => Reading::Unreadable {
            error: e.to_string(),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClient {
    pub name: String,
    pub client_type: String,
    /// `None` when the directory the path hangs off could not be determined,
    /// so there is no path to show and nothing about this client was measured.
    pub config_path: Option<String>,
    pub detected: Reading,
    /// `has_raw_entry` OR `has_plugin`, by [`Reading::or`] — kept alongside its
    /// two halves rather than replacing them, because they point at different
    /// fixes and the UI needs to tell them apart.
    pub already_configured: Reading,
    /// A raw `wenlan`/legacy `origin` entry in the client's own config file.
    pub has_raw_entry: Reading,
    /// BOTH the `wenlan` and the legacy `origin` raw entry — the raw+raw
    /// duplicate.
    pub has_raw_duplicate: Reading,
    /// The Wenlan plugin, for the three clients that have a plugin surface.
    pub has_plugin: Reading,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WenlanMcpEntry {
    pub command: String,
    pub args: Vec<String>,
}

/// Where a client's config file is, or why that could not be said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientConfigPath {
    Known(PathBuf),
    /// Not a client type this app knows.
    UnknownClient,
    /// The directory the path hangs off could not be determined, so nothing
    /// about this client can be measured — including whether it is installed.
    Undetermined(String),
}

/// The expected config file path for each MCP client.
pub fn client_config_path(client_type: &str) -> ClientConfigPath {
    client_config_path_for(
        client_type,
        dirs::home_dir().as_deref(),
        dirs::config_dir().as_deref(),
    )
}

/// Path construction, split from the directory lookups so the "the platform
/// would not report a home directory" branch is reachable from a test. It is
/// not reachable through the environment on Windows — `dirs::home_dir()`
/// resolves the known folder and IGNORES `$HOME` — so without this seam the
/// branch could only be reasoned about, never measured.
pub(crate) fn client_config_path_for(
    client_type: &str,
    home: Option<&Path>,
    config_dir: Option<&Path>,
) -> ClientConfigPath {
    // ROUND 5, DEFECT 4, THE STARKEST CASE. This function used to open with
    // `let home = dirs::home_dir()?;` — one `?` that turned EVERY client's path
    // into `None` when that lookup failed, `claude_desktop` included, whose
    // path does not use `home` at all. `detect_mcp_clients` then dropped all
    // five rows through `filter_map` and returned an EMPTY VECTOR, which the
    // Diagnostics card renders as "no MCP client detected" and the wizard as
    // nothing to configure. A lookup that failed became a confident negative
    // about the user's whole machine.
    //
    // Each arm now asks only for the directory it actually needs, and a
    // directory that could not be determined is a VALUE — a row that says so —
    // rather than a row that vanishes.
    let under = |dir: Option<&Path>, what: &str, tail: &[&str]| match dir {
        Some(dir) => ClientConfigPath::Known(
            tail.iter()
                .fold(dir.to_path_buf(), |acc, part| acc.join(part)),
        ),
        None => ClientConfigPath::Undetermined(format!("the platform would not report {what}")),
    };
    const HOME: &str = "a home directory";
    match client_type {
        // macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
        "claude_desktop" => under(
            config_dir,
            "an application-configuration directory",
            &["Claude", "claude_desktop_config.json"],
        ),
        "cursor" => under(home, HOME, &[".cursor", "mcp.json"]),
        "claude_code" => under(home, HOME, &[".claude.json"]),
        "gemini_cli" => under(home, HOME, &[".gemini", "settings.json"]),
        "codex_cli" => under(home, HOME, &[".codex", "config.toml"]),
        _ => ClientConfigPath::UnknownClient,
    }
}

const MCP_SERVER_KEY: &str = "wenlan";
const LEGACY_MCP_SERVER_KEY: &str = "origin";

/// Check if a JSON config string already has a Wenlan entry or legacy Origin
/// entry. `Ok(false)` is a MEASURED no — the file parsed and holds no such
/// entry. A body that would not parse is `Err`, never `Ok(false)`: see
/// [`ConfigRead::asks`].
///
/// A parsed file with no `mcpServers` key, or one whose `mcpServers` is not a
/// container, is a genuine `Ok(false)`: it demonstrably holds no
/// `mcpServers.wenlan` entry. Only the PARSE is a failed measurement.
fn has_configured_entry(json_str: &str) -> Result<bool, String> {
    let value = parse_json(json_str)?;
    Ok(match value.get("mcpServers") {
        Some(servers) => {
            servers.get(MCP_SERVER_KEY).is_some() || servers.get(LEGACY_MCP_SERVER_KEY).is_some()
        }
        None => false,
    })
}

/// TOML variant for Codex CLI (`[mcp_servers.*]` tables).
fn has_configured_entry_toml(toml_str: &str) -> Result<bool, String> {
    let doc = parse_toml(toml_str)?;
    Ok(match doc.get("mcp_servers") {
        Some(servers) => {
            servers.get(MCP_SERVER_KEY).is_some() || servers.get(LEGACY_MCP_SERVER_KEY).is_some()
        }
        None => false,
    })
}

/// Check whether a JSON config holds BOTH the live `wenlan` entry AND the
/// legacy `origin` entry under `mcpServers` — the raw+raw duplicate a client
/// with no plugin path (Cursor, Gemini CLI) lands in after the origin→wenlan
/// rename, where both entries launch a server against the same daemon. Distinct
/// from `has_configured_entry`, which is an OR: the fix here removes only the
/// stale `origin`, so detection has to know both are present, not just one.
fn has_both_raw_entries(json_str: &str) -> Result<bool, String> {
    let value = parse_json(json_str)?;
    Ok(match value.get("mcpServers") {
        Some(servers) => {
            servers.get(MCP_SERVER_KEY).is_some() && servers.get(LEGACY_MCP_SERVER_KEY).is_some()
        }
        None => false,
    })
}

/// TOML variant of `has_both_raw_entries` for Codex CLI (`[mcp_servers.*]`).
fn has_both_raw_entries_toml(toml_str: &str) -> Result<bool, String> {
    let doc = parse_toml(toml_str)?;
    Ok(match doc.get("mcp_servers") {
        Some(servers) => {
            servers.get(MCP_SERVER_KEY).is_some() && servers.get(LEGACY_MCP_SERVER_KEY).is_some()
        }
        None => false,
    })
}

/// Whether a Claude Code `settings.json` blob has the Wenlan plugin enabled.
/// `enabledPlugins` keys are `<plugin>@<marketplace>`, and the marketplace
/// name varies by install (`wenlan@7xuanlu` fresh, `wenlan@7xuanlu-wenlan` on
/// a machine that added the old self-marketplace) — match the `wenlan@`
/// prefix, never a literal marketplace name, or the check breaks for exactly
/// one of the two populations.
///
/// A missing `enabledPlugins` key is a measured "no plugin" — the file was
/// understood and it enables nothing. MALFORMED JSON IS NOT. That was the
/// shipped policy ("malformed JSON or a missing key is no plugin, never an
/// error") and it is round 6's D2: a `~/.claude/settings.json` that a partial
/// write or a merge conflict left unparseable answered exactly what a
/// plugins-off one answers, and the wizard invited the user to write a SECOND
/// registration over a plugin it had simply failed to read.
fn claude_code_plugin_enabled(settings_json: &str) -> Result<bool, String> {
    let value = parse_json(settings_json)?;
    Ok(value
        .get("enabledPlugins")
        .and_then(|plugins| plugins.as_object())
        .is_some_and(|plugins| {
            plugins
                .iter()
                .any(|(key, val)| key.starts_with("wenlan@") && val.as_bool() == Some(true))
        }))
}

/// Reads the real `~/.claude/settings.json` and checks it via
/// `claude_code_plugin_enabled`. Split out so the matching logic stays a pure,
/// directly testable function.
fn claude_code_plugin_enabled_on_disk(home: Option<&Path>) -> Reading {
    match home {
        // Round 5, D4: a denied `settings.json` used to answer `false` here,
        // which is what "the plugin is off" answers. One read, three answers.
        Some(home) => read_config(&home.join(".claude").join("settings.json"))
            .asks(claude_code_plugin_enabled),
        None => Reading::Unreadable {
            error: "the platform would not report a home directory".to_string(),
        },
    }
}

/// Whether a Codex CLI `config.toml` blob has the Wenlan plugin enabled —
/// `[plugins."wenlan@<marketplace>"] enabled = true`. The marketplace name
/// varies (`wenlan-local` pre-7xuanlu/wenlan#348, `7xuanlu-wenlan` after),
/// so match the `wenlan@` prefix, never a literal marketplace name — same
/// reasoning as `claude_code_plugin_enabled`.
fn codex_cli_plugin_enabled(toml_str: &str) -> Result<bool, String> {
    let doc = parse_toml(toml_str)?;
    Ok(doc
        .get("plugins")
        .and_then(|plugins| plugins.as_table_like())
        .is_some_and(|plugins| {
            plugins.iter().any(|(key, item)| {
                key.starts_with("wenlan@")
                    && item.get("enabled").and_then(|v| v.as_bool()) == Some(true)
            })
        }))
}

/// Reads the real `~/.codex/config.toml` and checks it via
/// `codex_cli_plugin_enabled`. Split out so the matching logic stays a pure,
/// directly testable function.
fn codex_cli_plugin_enabled_on_disk(home: Option<&Path>) -> Reading {
    match home {
        Some(home) => {
            read_config(&home.join(".codex").join("config.toml")).asks(codex_cli_plugin_enabled)
        }
        None => Reading::Unreadable {
            error: "the platform would not report a home directory".to_string(),
        },
    }
}

/// Whether a Claude Desktop chat-side plugin manifest (`rpm/manifest.json`
/// under a session directory) lists the Wenlan plugin. True iff `plugins[]`
/// contains an entry whose `name` field is exactly `"wenlan"` — matching
/// `marketplaceName` instead would be wrong, since a user's own upload
/// marketplace can be named anything (`marketplaceName` values seen in the
/// wild: `"My Uploads"`, `"knowledge-work-plugins"`). A missing `plugins` key
/// is a measured "no plugin"; a manifest that would not parse is `Err` — same
/// policy as `claude_code_plugin_enabled`, for the same round-6 reason.
fn claude_desktop_plugin_enabled(manifest_json: &str) -> Result<bool, String> {
    let value = parse_json(manifest_json)?;
    Ok(value
        .get("plugins")
        .and_then(|plugins| plugins.as_array())
        .is_some_and(|plugins| {
            plugins
                .iter()
                .any(|p| p.get("name").and_then(|n| n.as_str()) == Some("wenlan"))
        }))
}

/// Extract the pinned account id (`lastKnownAccountUuid`) from a Claude
/// Desktop `config.json` blob.
///
/// THREE outcomes, not two. `Ok(None)` is "the file was understood and pins no
/// account" — a real negative, and the reason the sessions scan is skipped.
/// `Err` is "the file could not be parsed", which establishes nothing about
/// whether an account is pinned. These used to be the same `None`, and the
/// caller turned that `None` into `Reading::No` — "Claude Desktop does not have
/// the Wenlan plugin", asserted from a file nothing could read.
fn claude_desktop_account_id(config_json: &str) -> Result<Option<String>, String> {
    let value = parse_json(config_json)?;
    Ok(value
        .get("lastKnownAccountUuid")
        .and_then(|id| id.as_str())
        .map(String::from))
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
fn claude_desktop_plugin_enabled_in_sessions_dir(account_sessions_dir: &Path) -> Reading {
    let entries = match std::fs::read_dir(account_sessions_dir) {
        Ok(entries) => entries,
        // No sessions directory: Desktop's chat side has never run. A real
        // negative.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Reading::No,
        Err(e) => {
            return Reading::Unreadable {
                error: e.to_string(),
            }
        }
    };
    // A single matching session is enough (a user can have several open). A
    // session whose manifest could NOT be read is neither a match nor an
    // absence: remembered, and reported only if nothing else matched. Round 5,
    // D4 — `entries.flatten()` plus `unwrap_or(false)` used to discard both the
    // per-entry errors and the per-manifest ones.
    let mut unreadable: Option<String> = None;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                unreadable.get_or_insert_with(|| e.to_string());
                continue;
            }
        };
        match read_config(&entry.path().join("rpm").join("manifest.json"))
            .asks(claude_desktop_plugin_enabled)
        {
            Reading::Yes => return Reading::Yes,
            Reading::No => {}
            Reading::Unreadable { error } => {
                unreadable.get_or_insert(error);
            }
        }
    }
    match unreadable {
        Some(error) => Reading::Unreadable { error },
        None => Reading::No,
    }
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
fn claude_desktop_plugin_enabled_for_support_dir(support_dir: &Path) -> Reading {
    let account_id = match read_config(&support_dir.join("config.json")) {
        ConfigRead::Contents(contents) => match claude_desktop_account_id(&contents) {
            Ok(Some(account_id)) => account_id,
            // The file was read and simply pins no account: a measured no.
            // There is no account whose sessions could hold a manifest.
            Ok(None) => return Reading::No,
            // ROUND 6, D2. A `config.json` that would not parse used to land on
            // the same `None` — and therefore the same `No` — as one that
            // parsed and pinned nothing. The sessions directory is named after
            // the account id, so an unparseable `config.json` means the scan
            // never happened at all; that is not "the plugin is not there".
            Err(error) => return Reading::Unreadable { error },
        },
        ConfigRead::Absent => return Reading::No,
        ConfigRead::Unreadable(error) => return Reading::Unreadable { error },
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
fn claude_desktop_plugin_enabled_on_disk(config_dir: Option<&Path>) -> Reading {
    match config_dir {
        Some(config_dir) => {
            claude_desktop_plugin_enabled_for_support_dir(&config_dir.join("Claude"))
        }
        None => Reading::Unreadable {
            error: "the platform would not report an application-configuration directory"
                .to_string(),
        },
    }
}

/// The reading contributed by the home-directory lookup ITSELF, before any
/// candidate under it is probed. A `None` home is not "this user has no home
/// directory" — it is a lookup that failed, and every `~/...` candidate that
/// would have been built from it went unprobed. Seeding a bundle search with
/// this keeps a search that could only look at half its candidates from
/// reporting the confident `No` that means "not installed".
fn home_lookup_reading(home: Option<&Path>) -> Reading {
    match home {
        Some(_) => Reading::No,
        None => Reading::Unreadable {
            error: "could not determine the home directory".to_string(),
        },
    }
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
    config_exists: Reading,
    home: Option<&Path>,
    exists: impl Fn(&Path) -> Reading,
) -> Reading {
    chatgpt_app_candidates(home)
        .iter()
        .map(|p| exists(p.as_path()))
        // `Reading::or`, not `any`: a bundle path the OS refused to stat is not
        // a bundle that is absent, and `any` over `Path::exists()` said it was.
        //
        // The seed is the home lookup itself: with no home, the
        // `~/Applications` candidate was never built and never probed, so a
        // `No` here would be a verdict from half a search.
        .fold(config_exists.or(home_lookup_reading(home)), Reading::or)
}

/// Whether Cursor is installed — by app bundle, not by config file, since
/// Cursor writes `~/.cursor/mcp.json` only once something configures it.
/// Same tri-state rule as [`codex_cli_detected`].
fn cursor_detected(home: Option<&Path>, exists: impl Fn(&Path) -> Reading) -> Reading {
    let mut candidates = vec![PathBuf::from("/Applications/Cursor.app")];
    if let Some(home) = home {
        candidates.push(home.join("Applications/Cursor.app"));
    }
    candidates
        .iter()
        .map(|p| exists(p.as_path()))
        .fold(home_lookup_reading(home), Reading::or)
}

/// Whether a client config's contents hold a raw wenlan/origin `mcpServers`
/// (or Codex's `[mcp_servers.*]`) entry — the file-based half of
/// `already_configured`. `wire_state` needs the two halves kept apart: a raw
/// entry and a missing plugin point at different fixes.
fn raw_entry_reading(client_type: &str, config: &ConfigRead) -> Reading {
    config.asks(|s| {
        if client_type == "codex_cli" {
            has_configured_entry_toml(s)
        } else {
            has_configured_entry(s)
        }
    })
}

/// Whether a client config's contents hold BOTH the `wenlan` entry and the
/// legacy `origin` entry — the raw+raw duplicate. Mirrors
/// `raw_entry_reading`'s TOML/JSON split, sharing
/// `has_both_raw_entries`/`has_both_raw_entries_toml` so detection and the
/// `remove_legacy_origin_entry` fix stay symmetric. This is the one signal a
/// no-plugin client (cursor, gemini_cli) needs: those can never trip the
/// plugin+raw double-registration path in `wire_state`, so without it their
/// raw+raw duplicate is invisible.
fn raw_duplicate_reading(client_type: &str, config: &ConfigRead) -> Reading {
    config.asks(|s| {
        if client_type == "codex_cli" {
            has_both_raw_entries_toml(s)
        } else {
            has_both_raw_entries(s)
        }
    })
}

/// `raw_entry_reading` against a path, for callers holding one rather than an
/// already-read file (the removal verbs' round-trip tests).
#[cfg(test)]
pub(crate) fn client_config_has_raw_entry(client_type: &str, config_path: &Path) -> Reading {
    raw_entry_reading(client_type, &read_config(config_path))
}

/// `raw_duplicate_reading` against a path. Same reason.
#[cfg(test)]
pub(crate) fn client_config_has_both_raw_entries(client_type: &str, config_path: &Path) -> Reading {
    raw_duplicate_reading(client_type, &read_config(config_path))
}

/// Whether `client_type`'s Wenlan plugin is enabled — the plugin half of
/// `already_configured` for the three clients that support one. `cursor` and
/// `gemini_cli` have no plugin path, so they are a MEASURED `No` here (and
/// route to `"config"` in `wire_state`, never `"plugin"`): there is no plugin
/// surface to fail to read.
fn client_plugin_enabled_for(
    client_type: &str,
    home: Option<&Path>,
    config_dir: Option<&Path>,
) -> Reading {
    match client_type {
        "claude_code" => claude_code_plugin_enabled_on_disk(home),
        "codex_cli" => codex_cli_plugin_enabled_on_disk(home),
        "claude_desktop" => claude_desktop_plugin_enabled_on_disk(config_dir),
        _ => Reading::No,
    }
}

/// Detect installed MCP-compatible tools and whether Wenlan is already
/// configured — in three values per fact, never two. See [`Reading`].
pub fn detect_mcp_clients() -> Vec<McpClient> {
    detect_mcp_clients_from(
        dirs::home_dir().as_deref(),
        dirs::config_dir().as_deref(),
        path_exists_reading,
    )
}

/// The body, with the two directory lookups and the existence probe handed in.
///
/// The seam exists because the failure this function was rewritten for cannot
/// be staged any other way: `dirs::home_dir()` on Windows resolves the known
/// folder and IGNORES `$HOME`, so "the platform would not report a home
/// directory" is unreachable through the environment on the platform this is
/// most often run on.
pub(crate) fn detect_mcp_clients_from(
    home: Option<&Path>,
    config_dir: Option<&Path>,
    exists: impl Fn(&Path) -> Reading,
) -> Vec<McpClient> {
    let clients = [
        ("Cursor", "cursor"),
        ("Claude Code", "claude_code"),
        ("Claude Desktop", "claude_desktop"),
        ("Gemini CLI", "gemini_cli"),
        ("Codex CLI", "codex_cli"),
    ];

    clients
        .iter()
        // ROUND 5, DEFECT 4: `map`, not `filter_map`. A client whose config
        // path could not even be built used to be DROPPED from the returned
        // vector — and since one failed `dirs::home_dir()` broke every arm of
        // `client_config_path`, that meant an empty vector: "no MCP client is
        // installed", stated confidently, from a lookup that failed. A row that
        // says "could not read" is the only honest output here, and it can only
        // exist if the row exists.
        .map(|(name, client_type)| {
            let path = client_config_path_for(client_type, home, config_dir);
            let (config_path, config) = match &path {
                ClientConfigPath::Known(path) => {
                    // ONE read answers both "is the file there" and "what does
                    // it say" — see `ConfigRead`. It used to be an `exists()`
                    // call plus a separate `read_to_string`, two instants and
                    // two collapses.
                    (Some(path.to_string_lossy().to_string()), read_config(path))
                }
                ClientConfigPath::UnknownClient => (
                    None,
                    ConfigRead::Unreadable(format!(
                        "{client_type} is not a client type this app knows"
                    )),
                ),
                ClientConfigPath::Undetermined(why) => (None, ConfigRead::Unreadable(why.clone())),
            };

            let has_raw_entry = raw_entry_reading(client_type, &config);
            let has_raw_duplicate = raw_duplicate_reading(client_type, &config);
            let has_plugin = client_plugin_enabled_for(client_type, home, config_dir);

            let detected = match *client_type {
                // Cursor: detect by app bundle, not config file — Cursor writes
                // `~/.cursor/mcp.json` only once something configures it.
                "cursor" => cursor_detected(home, &exists),
                // Codex CLI also detects off ChatGPT desktop's bundle: its Codex
                // pane reads the same `~/.codex/config.toml`, so a user who
                // only has ChatGPT desktop still gets this row.
                "codex_cli" => codex_cli_detected(config.present(), home, &exists),
                // Everything else: the config file's own presence.
                _ => config.present(),
            };

            McpClient {
                name: name.to_string(),
                client_type: client_type.to_string(),
                config_path,
                // A raw entry OR a plugin. `Reading::or` ranks a measured yes
                // above an unread half, so a client that is demonstrably
                // configured through one route is not dragged to "unknown" by
                // the other — and a client that is demonstrably configured
                // through NEITHER, with one route unread, is unknown rather
                // than "not configured".
                detected,
                already_configured: has_raw_entry.clone().or(has_plugin.clone()),
                has_raw_entry,
                has_raw_duplicate,
                has_plugin,
            }
        })
        .collect()
}

/// The backend release this app ships against. `app/Cargo.toml`'s version is
/// lockstepped to the daemon release (Milestone B phase 4c: the app builds
/// from the same tagged commit as `wenlan-server`/`wenlan-mcp`), so the
/// crate's own compile-time version can never disagree with the daemon this
/// build was tested against — no separate pin file needed.
const BACKEND_VERSION_PIN: &str = env!("CARGO_PKG_VERSION");

/// `wenlan-mcp@^<pinned version>`, e.g. `wenlan-mcp@^0.12.0`. Falls back to the
/// bare package name only if the pinned version string is unparseable — an
/// unpinned `npx` can silently pull a backend the app was never tested against.
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

/// Give a candidate binary path the platform's executable suffix —
/// `wenlan-mcp.exe` on Windows. Same idiom as
/// `lifecycle::service_cli_path_for_app_exe`, and for the same reason: a
/// suffix-less candidate never matches a real Windows install, so every
/// Windows desktop skipped the bundled binary sitting next to the app exe and
/// fell through to the `npx` fallback — a network dependency and a
/// version-skew hazard on a machine that already shipped the right binary.
fn with_exe_suffix(mut bin: PathBuf) -> PathBuf {
    if cfg!(target_os = "windows") {
        bin.set_extension("exe");
    }
    bin
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
        candidates.push((
            with_exe_suffix(home.join(".wenlan/bin/wenlan-mcp")),
            "installed",
        ));
    }
    if let Some(exe_dir) = exe_dir {
        candidates.push((with_exe_suffix(exe_dir.join("wenlan-mcp")), "bundled"));
    }
    if let Some(home) = home {
        candidates.push((with_exe_suffix(home.join(".cargo/bin/wenlan-mcp")), "cargo"));
    }
    candidates
}

#[cfg(test)]
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

/// Env var a test sets to point the resolver at a fixture install tree.
/// Mirrors `lifecycle::home_dir`'s `#[cfg(test)]` HOME hook. Without it a test
/// can only assert whatever the host it runs on happens to hold, which is how
/// the missing `.exe` suffix above shipped behind two tests that accepted the
/// bundled binary and the `npx` fallback interchangeably.
#[cfg(test)]
pub(crate) const MCP_RESOLVER_HOME_ENV: &str = "WENLAN_TEST_MCP_HOME";

/// Home directory the `installed` and `cargo` candidates hang off.
fn resolver_home_dir() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(home) = std::env::var_os(MCP_RESOLVER_HOME_ENV) {
        return Some(PathBuf::from(home));
    }
    dirs::home_dir()
}

/// What one candidate path is, as far as the filesystem would say.
///
/// `Path::exists()` answered this in two states and got both failure modes
/// wrong. It returns false when `metadata` fails for *any* reason, so a
/// `wenlan-mcp.exe` the OS refused to stat — a locked file, a denied ACL, a
/// disconnected network path — read as "not installed" and the app quietly
/// fell through to `npx`. And it returns true for anything at the path, so a
/// *directory* named `wenlan-mcp.exe` read as an executable and would be
/// written into the user's client config as the command to run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateProbe {
    /// A regular file that nothing here could rule OUT as a program. The only
    /// state that may be used as the command -- and it is a "not disqualified",
    /// not a "certified runnable". See [`file_probe`] for exactly what was and
    /// was not established, per platform.
    File,
    /// Something is there and it is not a regular file — a directory, most
    /// plausibly. Measured, and measured unusable.
    NotAFile,
    /// A regular file that is measured NOT to be runnable: empty, or (on Unix)
    /// carrying no execute bit. Measured, and measured unusable — the same
    /// standing as `NotAFile`, and deliberately NOT `Unreadable`: nothing
    /// failed here, the answer is just no.
    NotExecutable { reason: String },
    /// Measured absent.
    Absent,
    /// Could not look. NOT an absence.
    Unreadable { error: String },
}

/// Probe one candidate. `NotFound` is the only error that is an absence;
/// every other error is a failed look and says so.
pub fn probe_candidate(path: &Path) -> CandidateProbe {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => file_probe(&metadata),
        Ok(_) => CandidateProbe::NotAFile,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => CandidateProbe::Absent,
        Err(e) => CandidateProbe::Unreadable {
            error: e.to_string(),
        },
    }
}

/// Round 4, defect E: `metadata.is_file()` is a FILE witness, not an
/// EXECUTABLE witness, and the resolver writes the winner into a user's client
/// config as a command the OS then has to run.
///
/// WHAT THIS ESTABLISHES, PER PLATFORM — stated exactly, because the honest
/// answer differs and inventing a uniform one would be another check that
/// cannot fail:
///
/// * Everywhere: a zero-length file is not a program. No OS on any platform
///   executes an empty file — there is no interpreter line, no PE header, no
///   ELF header, nothing to load. This is a real check with a real failure
///   mode, and it is the one that catches the fixture problem below.
/// * Unix: the execute bit. `metadata.permissions().mode() & 0o111 == 0` means
///   the kernel will refuse `execve` with `EACCES` no matter what the bytes
///   are. A real, decidable negative.
/// * WINDOWS: THERE IS NO EXECUTE BIT, and nothing here pretends otherwise.
///   NTFS ACLs carry no "executable" attribute that `std::fs::Metadata`
///   exposes, and executability on Windows is decided by the loader reading the
///   image header at `CreateProcess` time. A non-empty regular file therefore
///   probes as `File` on Windows, and that is the strongest claim available
///   without opening and parsing the file.
///
/// THE RESIDUAL, STATED IN FULL, because a partial statement of it was itself
/// a defect in the last round. NEITHER CHECK CERTIFIES A RUNNABLE IMAGE.
/// `File` means "nothing here could rule it out", and the ways it can still be
/// wrong are not confined to Windows:
///
/// * Windows: a non-empty but corrupt or truncated `wenlan-mcp.exe` probes as
///   `File`. Nothing read the PE header.
/// * UNIX, EQUALLY: the execute bit is a permission, not a format. A corrupt
///   ELF, a shell script whose `#!` line names an interpreter that is not
///   installed, or any non-empty file at all with mode 0755 probes as `File`
///   and fails only when something tries to run it. The execute bit is a sound
///   NEGATIVE (no bit means `execve` returns `EACCES`, whatever the bytes are)
///   and a very weak positive, and only the negative is relied on here.
///
/// Parsing the image was considered and not done: a failed *read* of a real
/// binary would then have to be told apart from a real bad one, which is the
/// defect class this file exists to avoid, and the resolver's job is to pick a
/// path rather than to certify a program. What is claimed is exactly what is
/// checked, and the trail on the diagnostics wire shows the reading rather than
/// a verdict.
fn file_probe(metadata: &std::fs::Metadata) -> CandidateProbe {
    if metadata.len() == 0 {
        return CandidateProbe::NotExecutable {
            reason: "the file is empty (0 bytes), so it is not a program".to_string(),
        };
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        if mode & 0o111 == 0 {
            return CandidateProbe::NotExecutable {
                reason: format!("mode {:04o} has no execute bit set", mode & 0o7777),
            };
        }
    }
    CandidateProbe::File
}

/// One input the candidate paths hang off, in three values.
///
/// ROUND 4 FOLLOW-UP, C1.4. The tri-state was fixed at the PROBE and left
/// collapsing at candidate CONSTRUCTION, one layer upstream, where
/// `dirs::home_dir()`, `current_exe()` and a non-Unicode `WENLAN_MCP_DEV_BIN`
/// were all flattened into `None` by `Option` and `.ok()`. `None` then meant
/// "that candidate is not in play", the candidate was never built, it never
/// appeared in the trail, and a search that had failed to look at four paths
/// could end in `NoneInstalled` -- and write `npx` over a user's working local
/// command, with no unresolved trail to show why. A failed measurement
/// upstream of a hardened one is still a failed measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RootInput<T> {
    /// Measured: here it is.
    Known(T),
    /// Measured: there is no such input. `WENLAN_MCP_DEV_BIN` unset is a real
    /// negative -- the developer override simply is not in play, and no
    /// candidate is missing because of it.
    NotSet,
    /// Could NOT be determined. Every candidate hanging off it was never
    /// constructed, so the search did not cover the paths it claims to.
    Undetermined(String),
}

impl<T> RootInput<T> {
    fn known(&self) -> Option<&T> {
        match self {
            RootInput::Known(value) => Some(value),
            RootInput::NotSet | RootInput::Undetermined(_) => None,
        }
    }
}

/// An input that could not be determined, named the way a user would recognise
/// it. Deliberately not a path: there is no path, and that is the point.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UndeterminedInput {
    /// What could not be determined.
    pub input: String,
    /// The candidate sources that were therefore never probed.
    pub blocked: String,
    pub error: String,
}

/// Everything a search failed to turn into an answer it may act on, in one
/// value. An empty one is the precondition for `NoneInstalled` -- i.e. for
/// writing anything.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Unmeasured {
    /// Candidate paths that produced no usable answer: one the OS would not
    /// let this process look at, or -- round 6, additional defect 2 -- the
    /// winner it DID look at and cannot NAME, because a path that is not valid
    /// Unicode cannot be written into a JSON or TOML config without silently
    /// naming a different file. Both are "there is a path here and no answer
    /// about it that may be acted on", which is what the callers do with this.
    pub unreadable: Vec<(PathBuf, String)>,
    /// Inputs that could not be determined, so the candidates hanging off them
    /// were never built and never looked at at all.
    pub undetermined: Vec<UndeterminedInput>,
}

/// Why a found binary is still not writable as a command. Lives beside
/// `Unmeasured` because it is one of the two things `unreadable` can hold.
const UNREPRESENTABLE_BINARY_PATH: &str =
    "a usable wenlan-mcp binary is here, but its path is not valid Unicode, so it cannot be \
     written into a config file without naming a different file";

impl Unmeasured {
    pub(crate) fn is_empty(&self) -> bool {
        self.unreadable.is_empty() && self.undetermined.is_empty()
    }
}

/// The three inputs the candidate list is built from, each of which can fail
/// to be determined rather than merely be absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolverInputs {
    pub home: RootInput<PathBuf>,
    pub dev_bin: RootInput<String>,
    pub exe_dir: RootInput<PathBuf>,
}

impl ResolverInputs {
    /// Read the real environment.
    fn from_env() -> Self {
        Self::from_reads(
            resolver_home_dir(),
            std::env::var("WENLAN_MCP_DEV_BIN"),
            std::env::current_exe(),
        )
    }

    /// The classification, split from the reads so a test can hand it a
    /// failure directly. Without this seam the only way to exercise the
    /// NotUnicode arm would be to mutate the real process environment, and the
    /// arm that matters most -- `current_exe()` failing -- could not be
    /// reached at all.
    fn from_reads(
        home: Option<PathBuf>,
        dev_bin: Result<String, std::env::VarError>,
        exe: std::io::Result<PathBuf>,
    ) -> Self {
        Self {
            // `dirs::home_dir()` answering `None` is not "this user has no
            // home directory" -- it is the platform declining to say. Two
            // candidates (`installed`, `cargo`) hang off it.
            home: match home {
                Some(home) => RootInput::Known(home),
                None => RootInput::Undetermined(
                    "the platform would not report a home directory".to_string(),
                ),
            },
            // `env::var` returns `Err` for BOTH "unset" and "not valid
            // Unicode". ONLY THE FIRST IS AN ABSENCE. `.ok()` erased that
            // distinction, which is half of C1.4.
            dev_bin: match dev_bin {
                Ok(value) => RootInput::Known(value),
                Err(std::env::VarError::NotPresent) => RootInput::NotSet,
                Err(e) => RootInput::Undetermined(e.to_string()),
            },
            exe_dir: match exe {
                Ok(exe) => match exe.parent() {
                    Some(dir) => RootInput::Known(dir.to_path_buf()),
                    None => RootInput::Undetermined(
                        "the running executable has no parent directory".to_string(),
                    ),
                },
                Err(e) => RootInput::Undetermined(format!(
                    "the running executable's own path could not be read: {e}"
                )),
            },
        }
    }

    /// Only the inputs that were actually determined reach the path builder.
    fn as_paths(&self) -> (Option<&Path>, Option<&str>, Option<&Path>) {
        (
            self.home.known().map(PathBuf::as_path),
            self.dev_bin.known().map(String::as_str),
            self.exe_dir.known().map(PathBuf::as_path),
        )
    }

    fn undetermined(&self) -> Vec<UndeterminedInput> {
        let mut out = Vec::new();
        let mut push = |input: &str, blocked: &str, error: &String| {
            log::warn!(
                "[mcp] {input} could not be determined ({error}); the {blocked} candidate(s) were \
                 never probed, so this search cannot report that nothing is installed"
            );
            out.push(UndeterminedInput {
                input: input.to_string(),
                blocked: blocked.to_string(),
                error: error.clone(),
            });
        };
        if let RootInput::Undetermined(error) = &self.dev_bin {
            push("WENLAN_MCP_DEV_BIN", "WENLAN_MCP_DEV_BIN", error);
        }
        if let RootInput::Undetermined(error) = &self.home {
            push("the home directory", "installed and cargo", error);
        }
        if let RootInput::Undetermined(error) = &self.exe_dir {
            push("the application's own directory", "bundled", error);
        }
        out
    }
}

/// How the `wenlan-mcp` binary search ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum McpBinaryResolution {
    /// A candidate measured to be a regular file, in probe order — plus any
    /// input that could NOT be determined along the way.
    ///
    /// ROUND 5, DEFECT D4. `Found` used to be a bare path, and
    /// `inputs.undetermined()` was consulted only in the empty branch below.
    /// So `home = Known`, `dev_bin = Undetermined("not valid Unicode")`,
    /// `exe_dir = NotSet` with a hit under `home` returned `Found` and the
    /// undetermined input vanished from the resolution AND from the trail (an
    /// input that could not be determined builds no candidate path, so it has
    /// no trail entry to vanish from) — externally indistinguishable from
    /// `dev_bin = NotSet`, a MEASURED absence. A developer whose override was
    /// silently unreadable got the installed binary written into their config
    /// with nothing anywhere saying the override had been skipped.
    ///
    /// A find does not retroactively make an unread input read. The winner is
    /// still the first `File` in probe order — this changes nothing about the
    /// ranking, only about what a successful search is allowed to leave out.
    Found {
        path: PathBuf,
        undetermined: Vec<UndeterminedInput>,
    },
    /// EVERY candidate was constructed AND measured, and none is a usable
    /// file. `npx` is the right answer, and it is an answer, not a fallback
    /// from ignorance.
    NoneInstalled,
    /// No candidate is a usable file, and something was not measured: a path
    /// that could not be looked at, or an input that could not be determined
    /// so its paths were never built. "No binary is installed" was never
    /// established.
    Unresolved(Unmeasured),
}

/// One candidate as the resolver actually saw it. Carried out of the resolver
/// so `wire_state` can put THE DECISION'S OWN probe results on the diagnostics
/// wire instead of re-probing every path afterwards (round 4, defect F): a
/// second probe pass is a second instant, and a permission that changed in
/// between produced a trail that contradicted the command beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbedCandidate {
    pub path: PathBuf,
    pub source: &'static str,
    pub state: CandidateProbe,
}

/// A resolution plus the exact probe readings it was decided from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McpResolutionTrail {
    pub resolution: McpBinaryResolution,
    pub trail: Vec<ProbedCandidate>,
}

/// First candidate measured to be a usable file, in `wenlan_mcp_candidates`
/// order, together with every reading taken to decide that. Split from
/// [`find_wenlan_mcp_binary`] so the probe order can be exercised against an
/// explicit tree rather than the ambient one.
///
/// RANKING, decided and stated: a candidate that could not be looked at does
/// NOT stop the search — a readable bundled binary still outranks an
/// unreadable dev override, because a measured file beats an unmeasured
/// anything. But it does change the *shape* of an empty result. Ending the
/// search with no file and at least one unreadable candidate is
/// [`McpBinaryResolution::Unresolved`], never `NoneInstalled`: the silent
/// `npx` fall-through those cases used to take is the shipped defect, and the
/// two are now different values that different code paths handle.
///
/// Every candidate is probed exactly ONCE, and probing does not stop at the
/// winner. That costs a few `stat` calls and buys the property defect F is
/// about: there is one set of readings, the decision came from it, and the
/// diagnostics trail IS it. The winner is still the first `File` in order, so
/// the ranking above is unchanged.
/// The key two candidate SLOTS share when they name one filesystem object.
///
/// ROUND 6, ADDITIONAL DEFECT 3 — a FALSE-GREEN CONTROL. The dedup below
/// claimed "one reading per filesystem object" while keying a
/// `HashMap<PathBuf, _>` on PATHNAME EQUALITY, and its test supplied the same
/// spelling twice — so the test could not fail for the reason the property is
/// about, and could not establish the claim in its own title. Two spellings of
/// one file (`~/.wenlan/./bin/wenlan-mcp` as a dev override beside
/// `~/.wenlan/bin/wenlan-mcp` as the installed candidate; a symlink; a
/// differently-cased alias on Windows) were probed TWICE — two instants, which
/// is round 4's defect F, the thing the dedup exists to prevent.
///
/// `canonicalize` resolves `.`/`..`, symlinks, and (on Windows) the real
/// on-disk casing, so those three all collapse to one key. It is a look that
/// can fail — most often because the candidate simply is not there — and a
/// failure falls back to the literal path: two absent candidates then get two
/// readings, which costs one `stat` and cannot produce a contradiction, since
/// there is nothing there for either reading to disagree about.
///
/// THE RESIDUAL, STATED: a HARDLINK is still two keys. Two directory entries
/// for one inode are genuinely two paths, and `canonicalize` returns each
/// unchanged; telling them apart needs a file identity `std` does not expose
/// portably (`std::os::windows::fs::MetadataExt::file_index` is unstable, and
/// inventing an identity would be a check that cannot fail — the thing this
/// file exists to avoid). What is claimed is exactly what is done.
fn probe_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn resolve_wenlan_mcp_with_trail(
    inputs: &ResolverInputs,
    probe: impl Fn(&Path) -> CandidateProbe,
) -> McpResolutionTrail {
    let (home, dev_bin, exe_dir) = inputs.as_paths();
    // C1.6: one reading per FILESYSTEM OBJECT, not one per vector entry. The
    // candidate list is four SLOTS, and two of them can name the same file --
    // `WENLAN_MCP_DEV_BIN` pointed at the installed binary is the ordinary
    // case for a developer. Probing that file twice is two instants, which is
    // defect F in miniature: the same path could appear in the trail as both
    // `file` and `unreadable`, contradicting itself and the command beside it.
    // Every slot still appears in the trail -- the user should see that both
    // named the same file -- but they share the one reading.
    let mut seen: std::collections::HashMap<PathBuf, CandidateProbe> =
        std::collections::HashMap::new();
    let trail: Vec<ProbedCandidate> = wenlan_mcp_candidate_sources(home, dev_bin, exe_dir)
        .into_iter()
        .map(|(path, source)| {
            let key = probe_key(path.as_path());
            let state = match seen.get(key.as_path()) {
                Some(already) => already.clone(),
                None => {
                    let measured = probe(path.as_path());
                    seen.insert(key, measured.clone());
                    measured
                }
            };
            match &state {
                CandidateProbe::NotAFile => log::warn!(
                    "[mcp] {} exists but is not a regular file; it is not the wenlan-mcp binary \
                     and will not be written into any client config",
                    path.display()
                ),
                CandidateProbe::NotExecutable { reason } => log::warn!(
                    "[mcp] {} is a file but not a runnable one ({reason}); it will not be written \
                     into any client config",
                    path.display()
                ),
                CandidateProbe::Unreadable { error } => {
                    log::warn!("[mcp] could not look at {}: {error}", path.display())
                }
                CandidateProbe::File | CandidateProbe::Absent => {}
            }
            ProbedCandidate {
                path,
                source,
                state,
            }
        })
        .collect();

    let found = trail
        .iter()
        .find(|c| c.state == CandidateProbe::File)
        .map(|c| c.path.clone());
    // Round 5, D4: computed BEFORE the outcome is known, and unconditionally.
    // This used to live inside the `None` arm, which made "which inputs could
    // not be read" a property of an empty result rather than of the search --
    // so a hit anywhere short-circuited it out of existence, and the warnings
    // this call logs were never emitted either.
    let undetermined = inputs.undetermined();
    let resolution = match found {
        Some(path) => McpBinaryResolution::Found { path, undetermined },
        None => {
            let unmeasured = Unmeasured {
                unreadable: trail
                    .iter()
                    .filter_map(|c| match &c.state {
                        CandidateProbe::Unreadable { error } => {
                            Some((c.path.clone(), error.clone()))
                        }
                        _ => None,
                    })
                    .collect(),
                undetermined,
            };
            if unmeasured.is_empty() {
                McpBinaryResolution::NoneInstalled
            } else {
                McpBinaryResolution::Unresolved(unmeasured)
            }
        }
    };
    McpResolutionTrail { resolution, trail }
}

/// Inputs that were all determined -- the shape the older tests state their
/// fixtures in. `None` here means `NotSet`, a MEASURED absence; a test that
/// wants an undetermined input builds [`ResolverInputs`] itself.
#[cfg(test)]
pub(crate) fn determined_inputs(
    home: Option<&Path>,
    dev_bin: Option<&str>,
    exe_dir: Option<&Path>,
) -> ResolverInputs {
    fn known<T>(value: Option<T>) -> RootInput<T> {
        match value {
            Some(value) => RootInput::Known(value),
            None => RootInput::NotSet,
        }
    }
    ResolverInputs {
        home: known(home.map(Path::to_path_buf)),
        dev_bin: known(dev_bin.map(str::to_string)),
        exe_dir: known(exe_dir.map(Path::to_path_buf)),
    }
}

/// The resolution alone, for callers that do not need the trail.
#[cfg(test)]
fn resolve_wenlan_mcp(
    home: Option<&Path>,
    dev_bin: Option<&str>,
    exe_dir: Option<&Path>,
    probe: impl Fn(&Path) -> CandidateProbe,
) -> McpBinaryResolution {
    resolve_wenlan_mcp_with_trail(&determined_inputs(home, dev_bin, exe_dir), probe).resolution
}

fn find_wenlan_mcp_binary_with_trail() -> McpResolutionTrail {
    resolve_wenlan_mcp_with_trail(&ResolverInputs::from_env(), probe_candidate)
}

/// What to do with a client config, given how the binary search ended.
///
/// The third value exists because "write nothing" is a thing this code has to
/// be able to say, and before round 4 it could not: `Unresolved` returned the
/// same `npx` entry as `NoneInstalled`, so a call site had no way to express
/// "leave the user's config alone".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum McpEntryDecision {
    /// Write this entry — and, separately, the inputs that could not be
    /// determined while deciding it (round 5, D4).
    ///
    /// `undetermined` is NOT a reason to withhold the entry: a candidate that
    /// was measured to be a usable file is still the right command, and the
    /// round-4 rule ("write nothing rather than a guess") applies to an EMPTY
    /// search, not to a search that found something while one of its inputs
    /// went unread. It is carried because the alternative is to drop it, and a
    /// dropped failed measurement is indistinguishable from a negative one.
    Write {
        entry: WenlanMcpEntry,
        undetermined: Vec<UndeterminedInput>,
    },
    /// Write NOTHING. No candidate was confirmed usable and at least one could
    /// not be looked at, so no measured absence exists to justify any entry —
    /// including `npx`. Whatever is in the user's config stays there.
    PreserveExisting { unmeasured: Unmeasured },
}

/// The message the UI shows and the writers fail with. Names every path that
/// could not be read, and says plainly that nothing was changed.
pub(crate) fn unresolved_message(unmeasured: &Unmeasured) -> String {
    let mut reasons: Vec<String> = unmeasured
        .unreadable
        .iter()
        .map(|(path, error)| format!("{} ({error})", path.display()))
        .collect();
    reasons.extend(unmeasured.undetermined.iter().map(|u| {
        format!(
            "{} could not be determined ({}), so the {} candidate(s) were never checked",
            u.input, u.error, u.blocked
        )
    }));
    format!(
        "Could not determine the wenlan-mcp binary: {}. Nothing was written — your existing MCP \
         configuration is unchanged.",
        reasons.join("; ")
    )
}

/// The MCP config entry Wenlan writes into client config files: an installed
/// binary when one was measured, a version-pinned `npx` when nothing is
/// installed, and NOTHING AT ALL when the search could not be completed.
///
/// ROUND 4, DEFECT D — the collapse this removes. `Unresolved` used to yield
/// the same `npx` entry as `NoneInstalled`, justified as "a config entry has to
/// be something a client can run". That justification does not survive its own
/// scenario: a real `wenlan-mcp.exe` that is momentarily unstatable — an ACL,
/// an antivirus lock, a disconnected network path — made the app overwrite a
/// working local command with `npx`, which on an offline machine, or one with
/// no Node, is precisely the entry that is NOT runnable. A guess is not more
/// runnable than the thing it replaced; it is only less measured.
///
/// So `Unresolved` now maps to `PreserveExisting`, and the writers make no
/// change and report the failure. Writing nothing cannot break a working
/// config; writing a guess can, and did.
fn wenlan_mcp_entry_for(resolution: McpBinaryResolution, npm_package: &str) -> McpEntryDecision {
    match resolution {
        // ROUND 6, ADDITIONAL DEFECT 2 — a RELEASE ARTIFACT, not a diagnostic.
        // This was `path.to_string_lossy().to_string()`, which never fails and
        // therefore never reports. A real, measured-usable `wenlan-mcp` under a
        // path that is not valid Unicode resolved as `Found`, and the command
        // written into the user's client config was the same bytes with the
        // unrepresentable ones replaced by U+FFFD — a DIFFERENT path, naming a
        // file that does not exist. The client then fails to launch the server
        // with an error pointing at a filename the user cannot find on disk.
        //
        // `to_str()` is the fallible version, and a path that cannot be spelled
        // is exactly the case round 4's defect D settled: no measured answer
        // exists, so write nothing and say why. `to_string_lossy` is still used
        // for the MESSAGE, where an approximate spelling is what a human needs
        // and cannot be mistaken for a command.
        McpBinaryResolution::Found { path, undetermined } => match path.to_str() {
            Some(command) => McpEntryDecision::Write {
                entry: WenlanMcpEntry {
                    command: command.to_string(),
                    args: Vec::new(),
                },
                // Round 5, D4: a find does not erase an input that could not be
                // read. Carried through so the diagnostics wire can say "this
                // command was chosen, AND these inputs were never determined"
                // instead of showing a clean, complete search.
                undetermined,
            },
            None => {
                log::error!(
                    "[mcp] {} is a usable wenlan-mcp binary, but its path is not valid Unicode; \
                     writing it into a config file would name a different, nonexistent file, so \
                     nothing will be written",
                    path.display()
                );
                McpEntryDecision::PreserveExisting {
                    unmeasured: Unmeasured {
                        unreadable: vec![(path, UNREPRESENTABLE_BINARY_PATH.to_string())],
                        undetermined,
                    },
                }
            }
        },
        // `NoneInstalled` is only reachable when `Unmeasured::is_empty()`, so
        // by construction there is nothing undetermined to carry here.
        McpBinaryResolution::NoneInstalled => McpEntryDecision::Write {
            entry: WenlanMcpEntry {
                command: "npx".to_string(),
                args: vec!["-y".to_string(), npm_package.to_string()],
            },
            undetermined: Vec::new(),
        },
        McpBinaryResolution::Unresolved(unmeasured) => {
            for (path, error) in &unmeasured.unreadable {
                log::error!(
                    "[mcp] refusing to write any wenlan entry while {} could not be read \
                     ({error}); this is NOT a measured absence, so `npx {npm_package}` would be a \
                     guess overwriting a possibly-working config",
                    path.display()
                );
            }
            McpEntryDecision::PreserveExisting { unmeasured }
        }
    }
}

/// [`wenlan_mcp_decision`] against an explicit candidate tree and probe, so
/// `wire_state`'s tests can exercise the one-pass property without the ambient
/// machine deciding the outcome.
#[cfg(test)]
pub(crate) fn wenlan_mcp_decision_for(
    home: Option<&Path>,
    dev_bin: Option<&str>,
    exe_dir: Option<&Path>,
    probe: impl Fn(&Path) -> CandidateProbe,
    npm_package: &str,
) -> (McpEntryDecision, Vec<ProbedCandidate>) {
    wenlan_mcp_decision_from(
        &determined_inputs(home, dev_bin, exe_dir),
        probe,
        npm_package,
    )
}

/// The same, from inputs a test built itself -- the only way to stage an
/// UNDETERMINED input, which is C1.4's whole subject.
#[cfg(test)]
pub(crate) fn wenlan_mcp_decision_from(
    inputs: &ResolverInputs,
    probe: impl Fn(&Path) -> CandidateProbe,
    npm_package: &str,
) -> (McpEntryDecision, Vec<ProbedCandidate>) {
    let McpResolutionTrail { resolution, trail } = resolve_wenlan_mcp_with_trail(inputs, probe);
    (wenlan_mcp_entry_for(resolution, npm_package), trail)
}

/// One probe pass over the real machine: what to write, and the readings that
/// decided it. Every caller that needs either takes both from here, so the
/// decision and the trail describing it can never come from different instants.
pub(crate) fn wenlan_mcp_decision() -> (McpEntryDecision, Vec<ProbedCandidate>) {
    let McpResolutionTrail { resolution, trail } = find_wenlan_mcp_binary_with_trail();
    let decision =
        wenlan_mcp_entry_for(resolution, &pinned_wenlan_mcp_package(BACKEND_VERSION_PIN));
    (decision, trail)
}

/// The entry that would be written, together with every input that could not
/// be determined while deciding it.
///
/// ROUND 6, D3's BOUNDARY DEFECT. `McpEntryDecision::Write` carries
/// `undetermined` correctly — and then `wenlan_mcp_entry()` matched
/// `Write { entry, .. }` and threw it away, so the PUBLIC result of the search
/// was `{command, args}` either way. With a non-Unicode `WENLAN_MCP_DEV_BIN`
/// and a readable installed binary, that result was byte-for-byte what an
/// UNSET override produces — a measured absence — and the developer whose
/// override was silently skipped had nothing anywhere to tell them. A vector
/// that reaches the boundary and is dropped at it never travelled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WenlanMcpEntryReport {
    pub entry: WenlanMcpEntry,
    /// Inputs that could not be determined. EMPTY is itself a measurement:
    /// every input was read. Non-empty means this command was chosen by a
    /// search that did not cover the paths it looks like it covered.
    pub undetermined: Vec<UndeterminedInput>,
}

/// The entry that would be written, or an error naming what could not be read.
///
/// The `Err` arm is the only thing a caller can do with `PreserveExisting`
/// besides make no change: there is no entry to report because none was
/// established.
pub fn wenlan_mcp_entry() -> Result<WenlanMcpEntryReport, AppError> {
    match wenlan_mcp_decision().0 {
        McpEntryDecision::Write {
            entry,
            undetermined,
        } => Ok(WenlanMcpEntryReport {
            entry,
            undetermined,
        }),
        McpEntryDecision::PreserveExisting { unmeasured } => {
            Err(AppError::Generic(unresolved_message(&unmeasured)))
        }
    }
}

/// The name of a JSON value's type, for a schema-error message a user can act
/// on ("`mcpServers` is a list" beats "expected object").
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a true/false value",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "an object",
    }
}

/// The shapes [`write_wenlan_entry_with`] is able to write into, checked
/// BEFORE anything is backed up or written.
///
/// ROUND 6, ADDITIONAL DEFECT 1 — THE CRASH, AND IT CRASHED AFTER THE BACKUP.
/// `root["mcpServers"][MCP_SERVER_KEY] = ..` is `serde_json`'s `IndexMut`,
/// which PANICS — not errors — when the thing being indexed is neither an
/// object nor null. `{"mcpServers": []}` is perfectly valid JSON: it parsed,
/// it passed the "does the key exist" check, and then the assignment panicked
/// with `cannot access key "wenlan" in JSON array`. Any valid non-object top
/// level (`[]`, `"x"`, `3`) panicked one line earlier, on the insert. A panic
/// inside a Tauri command is not an error the UI can show — the user gets a
/// dead button and, on the `mcpServers` variant, a `.json.bak` proving
/// something started.
///
/// There was no schema-error path at all, so this adds one: an unexpected
/// shape is an ordinary `Err`, raised before the backup, and the user's file
/// is untouched.
fn check_json_config_shape(
    config_path: &std::path::Path,
    root: &serde_json::Value,
) -> Result<(), AppError> {
    let unchanged = "Nothing was written — your existing MCP configuration is unchanged.";
    if !root.is_object() {
        return Err(AppError::Generic(format!(
            "Unexpected shape in {}: the top level of the file is {}, but an MCP client config \
             has to be an object. {unchanged}",
            config_path.display(),
            json_type_name(root)
        )));
    }
    match root.get("mcpServers") {
        // Absent, or present-and-null: both are places a fresh `mcpServers`
        // object can be put without destroying anything.
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(servers) if servers.is_object() => Ok(()),
        Some(servers) => Err(AppError::Generic(format!(
            "Unexpected shape in {}: `mcpServers` is {}, but it has to be an object for Wenlan to \
             add its entry to it. {unchanged}",
            config_path.display(),
            json_type_name(servers)
        ))),
    }
}

/// Back the config up from the bytes that were PARSED — and only while those
/// are still the bytes on disk.
///
/// ROUND 6, D1, CONDITION 1. The backup used to be `fs::copy(config_path,
/// backup_path)`, which RE-READS the path. Read-parse-then-back-up fixed the
/// ordering but not the object: the copy captured whatever was at that
/// pathname at a LATER instant than the parse. Race, on both writers — the
/// writer reads and parses valid A; another process replaces the path with
/// malformed B; `fs::copy` puts B on top of the last good backup; the writer
/// then overwrites B from its stale parse of A. The one recoverable copy is
/// destroyed and B's contents are lost, silently.
///
/// Two changes, one per half. The backup is WRITTEN FROM `contents`, so it can
/// only ever hold bytes this process parsed — `fs::copy` cannot be made to
/// promise that. And the file is re-read first: if it no longer holds those
/// bytes, the update is abandoned rather than applied from a stale parse, so a
/// concurrent change is reported instead of overwritten.
///
/// THE RESIDUAL, STATED: this narrows the window, it does not close it.
/// Between this check and the caller's `fs::write` the file can still change,
/// and closing that properly needs a lock this codebase does not take on
/// another vendor's config file. What it does guarantee unconditionally is the
/// part that was destructive: the backup is never bytes nothing parsed.
fn back_up_parsed(
    config_path: &std::path::Path,
    contents: &str,
    backup_extension: &str,
) -> Result<(), AppError> {
    match read_config(config_path) {
        ConfigRead::Contents(now) if now == contents => {}
        ConfigRead::Contents(_) | ConfigRead::Absent => {
            return Err(AppError::Generic(format!(
                "{} changed while Wenlan was updating it, so this update was built from bytes that \
                 are no longer there. Nothing was written and no backup was taken — try again.",
                config_path.display()
            )))
        }
        ConfigRead::Unreadable(error) => {
            return Err(AppError::Generic(format!(
                "Could not re-read {} to confirm it had not changed ({error}). Nothing was written \
                 — your existing MCP configuration is unchanged.",
                config_path.display()
            )))
        }
    }
    std::fs::write(config_path.with_extension(backup_extension), contents)?;
    Ok(())
}

/// The message for a config file that is THERE but could not be read.
///
/// ROUND 6, D1, CONDITION 2. Both writers opened with `if config_path.exists()`
/// — the two-answer API this file condemns everywhere else. `exists()` is
/// `false` for a metadata denial as well as for an absence, so a file that
/// permits writing but not stat-ing took the NEW-FILE branch and was
/// TRUNCATED, with no backup, from a `json!({})` skeleton. `read_config` is
/// the three-answer replacement, and it also removes the second instant: one
/// read answers "is it there" and "what does it say".
fn unreadable_config_message(config_path: &std::path::Path, error: &str) -> AppError {
    AppError::Generic(format!(
        "Could not read {} ({error}), so Wenlan cannot tell what is in it. Nothing was written — \
         your existing MCP configuration is unchanged.",
        config_path.display()
    ))
}

/// Write the Wenlan MCP server entry into a client's config file.
/// Existing legacy `origin` entries are preserved and still detected.
/// If `is_claude_code` is true and the file doesn't exist, returns an error
/// (Claude Code manages its own config file).
///
/// `Ok` carries the inputs that could NOT be determined while deciding what to
/// write (round 6, D3's boundary defect). A write that succeeded while one of
/// its inputs went unread is not the same event as one where everything was
/// measured, and returning `()` for both made them the same event to every
/// caller.
pub fn write_wenlan_entry(
    config_path: &std::path::Path,
    is_claude_code: bool,
) -> Result<Vec<UndeterminedInput>, AppError> {
    write_wenlan_entry_with(config_path, is_claude_code, wenlan_mcp_decision().0)
}

/// Body of [`write_wenlan_entry`] with the decision handed in, so a test can
/// stage `PreserveExisting` deterministically. The real OS refusal that
/// produces it (a denied ACL, a disconnected share) is not reproducible on
/// every platform a test runs on — Windows answers `NotFound` for several
/// shapes Unix answers `ENOTDIR` for — and the branch under test is this
/// function's, not the OS's.
pub(crate) fn write_wenlan_entry_with(
    config_path: &std::path::Path,
    is_claude_code: bool,
    decision: McpEntryDecision,
) -> Result<Vec<UndeterminedInput>, AppError> {
    // Decide BEFORE touching anything. An unresolvable binary must leave the
    // file exactly as it was — no rewrite, and no `.json.bak` either, since a
    // backup of an unchanged file is just litter that suggests a change
    // happened. (Round 4, defect D.)
    let (entry, undetermined) = match decision {
        McpEntryDecision::Write {
            entry,
            undetermined,
        } => (entry, undetermined),
        McpEntryDecision::PreserveExisting { unmeasured } => {
            return Err(AppError::Generic(unresolved_message(&unmeasured)))
        }
    };
    // ONE read, three answers — `read_config`, not `exists()` plus a separate
    // `read_to_string`. See `unreadable_config_message` for the branch that
    // used to truncate a file it could not stat.
    let existing = match read_config(config_path) {
        ConfigRead::Contents(contents) => Some(contents),
        ConfigRead::Absent => None,
        ConfigRead::Unreadable(error) => {
            return Err(unreadable_config_message(config_path, &error))
        }
    };
    let mut root = match &existing {
        // ROUND 5, DEFECT 1 — the ordering that destroyed the user's only
        // recoverable copy. The backup used to run BEFORE this parse, so a
        // malformed `config.json` beside a `config.json.bak` holding the last
        // GOOD configuration overwrote the good backup with the broken file
        // and only then returned `Invalid JSON`.
        //
        // So: read, parse, CHECK THE SHAPE, build the new document, and only
        // then back up — see `back_up_parsed`, which also fixes which BYTES
        // the backup holds. A backup is a copy of a configuration that was
        // understood; a copy of bytes nothing could parse is not a backup, it
        // is the loss of one.
        Some(contents) => {
            let parsed = serde_json::from_str::<serde_json::Value>(contents).map_err(|e| {
                AppError::Generic(format!("Invalid JSON in {}: {}", config_path.display(), e))
            })?;
            check_json_config_shape(config_path, &parsed)?;
            parsed
        }
        None if is_claude_code => {
            return Err(AppError::Generic(
                "Claude Code config file not found — Claude Code manages this file internally"
                    .into(),
            ))
        }
        // Create minimal skeleton for Claude Desktop / Cursor
        None => serde_json::json!({}),
    };

    // Ensure `mcpServers` is an OBJECT, not merely present — `check_json_config_shape`
    // has already ruled out every shape but object/null/absent, and a present
    // `null` still has to be replaced before it can be indexed.
    if !root.get("mcpServers").is_some_and(|v| v.is_object()) {
        root["mcpServers"] = serde_json::json!({});
    }
    root["mcpServers"][MCP_SERVER_KEY] =
        serde_json::to_value(entry).map_err(|e| AppError::Generic(e.to_string()))?;

    // Write back with pretty formatting
    let formatted =
        serde_json::to_string_pretty(&root).map_err(|e| AppError::Generic(e.to_string()))?;
    // Everything that could fail has failed by now, so the backup cannot be
    // left behind by a change that never happened (round 4, defect D).
    if let Some(contents) = &existing {
        back_up_parsed(config_path, contents, "json.bak")?;
    }
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(config_path, formatted)?;

    Ok(undetermined)
}

/// Upsert the Wenlan entry into a Codex CLI `config.toml` — format-preserving:
/// user comments, key order, and unrelated tables survive byte-for-byte
/// (toml_edit round-trips everything it didn't touch).
pub fn write_wenlan_entry_toml(
    config_path: &std::path::Path,
) -> Result<Vec<UndeterminedInput>, AppError> {
    write_wenlan_entry_toml_with(config_path, wenlan_mcp_decision().0)
}

/// Body of [`write_wenlan_entry_toml`] with the decision handed in. Same
/// reason as [`write_wenlan_entry_with`].
pub(crate) fn write_wenlan_entry_toml_with(
    config_path: &std::path::Path,
    decision: McpEntryDecision,
) -> Result<Vec<UndeterminedInput>, AppError> {
    use toml_edit::{DocumentMut, Item, Table};

    // Same ordering rule as `write_wenlan_entry`: an unresolvable binary must
    // not produce a `.toml.bak` or a rewrite. (Round 4, defect D.)
    let (entry, undetermined) = match decision {
        McpEntryDecision::Write {
            entry,
            undetermined,
        } => (entry, undetermined),
        McpEntryDecision::PreserveExisting { unmeasured } => {
            return Err(AppError::Generic(unresolved_message(&unmeasured)))
        }
    };
    // Same `read_config`, same reason as `write_wenlan_entry_with`: `exists()`
    // read a metadata denial as an absence and truncated the file.
    let existing = match read_config(config_path) {
        ConfigRead::Contents(contents) => Some(contents),
        ConfigRead::Absent => None,
        ConfigRead::Unreadable(error) => {
            return Err(unreadable_config_message(config_path, &error))
        }
    };
    let mut doc: DocumentMut = match &existing {
        // Same ordering, same reason as `write_wenlan_entry_with` (round 5,
        // defect 1): read and parse FIRST, back up only what parsed.
        Some(contents) => {
            let parsed: DocumentMut = contents.parse().map_err(|e| {
                AppError::Generic(format!("Invalid TOML in {}: {}", config_path.display(), e))
            })?;
            // ROUND 6, ADDITIONAL DEFECT 1, THE TOML HALF. `doc["mcp_servers"][key] = ..`
            // is `toml_edit`'s `IndexMut`, which is `index_mut(..).expect("index
            // not found")` — a PANIC, not an error, for any `mcp_servers` that
            // is not table-like. `mcp_servers = 5` in a user's `config.toml`
            // parses fine, survives the presence check, and panics on the
            // assignment. Same class as the JSON crash, same fix: a real
            // schema-error path, raised before any backup or write.
            match parsed.get("mcp_servers") {
                None => {}
                Some(servers) if servers.as_table_like().is_some() => {}
                Some(servers) => {
                    return Err(AppError::Generic(format!(
                        "Unexpected shape in {}: `mcp_servers` is {}, but it has to be a table for \
                         Wenlan to add its entry to it. Nothing was written — your existing MCP \
                         configuration is unchanged.",
                        config_path.display(),
                        servers.type_name()
                    )))
                }
            }
            parsed
        }
        None => DocumentMut::new(),
    };

    if doc.get("mcp_servers").is_none() {
        let mut parent = Table::new();
        parent.set_implicit(true); // render only [mcp_servers.wenlan], no bare [mcp_servers]
        doc.insert("mcp_servers", Item::Table(parent));
    }

    let mut server = Table::new();
    server.insert("command", toml_edit::value(entry.command));
    let mut args = toml_edit::Array::new();
    for a in entry.args {
        args.push(a);
    }
    server.insert("args", toml_edit::value(args));
    doc["mcp_servers"][MCP_SERVER_KEY] = Item::Table(server);

    let formatted = doc.to_string();
    // Backup last, from the bytes that were parsed — see `back_up_parsed`.
    if let Some(contents) = &existing {
        back_up_parsed(config_path, contents, "toml.bak")?;
    }
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(config_path, formatted)?;
    Ok(undetermined)
}

/// Shared body of the four JSON removal verbs below: exists-check, parse,
/// remove every key in `keys` from `mcpServers`, then — only once a removal is
/// certain — back the file up before writing it back, so a no-op leaves no
/// stray `.bak`. `not_found` is the caller's nothing-was-removed message,
/// which the UI surfaces verbatim. Every sibling server and unrelated key
/// survives.
fn remove_json_keys(
    config_path: &std::path::Path,
    keys: &[&str],
    not_found: &str,
) -> Result<(), AppError> {
    // ROUND 6, ADDITIONAL DEFECT 4 — the same boundary collapse as the
    // writers. `if !config_path.exists()` reported "No config file found" for a
    // METADATA DENIAL as well as for an absence, so a config that is right
    // there and holds the entry the user asked to remove was presented as a
    // measured absence — and the read was never attempted. `read_config` gives
    // the third answer.
    let contents = match read_config(config_path) {
        ConfigRead::Contents(contents) => contents,
        ConfigRead::Absent => {
            return Err(AppError::Generic(
                "No config file found — nothing to remove".into(),
            ))
        }
        ConfigRead::Unreadable(error) => {
            return Err(AppError::Generic(format!(
                "Could not read {} ({error}), so Wenlan cannot tell whether there is anything to \
                 remove. Nothing was changed.",
                config_path.display()
            )))
        }
    };
    let mut root = serde_json::from_str::<serde_json::Value>(&contents).map_err(|e| {
        AppError::Generic(format!("Invalid JSON in {}: {}", config_path.display(), e))
    })?;

    let removed = root
        .get_mut("mcpServers")
        .and_then(|servers| servers.as_object_mut())
        .map(|servers| {
            // Deliberately not `any`/`fold`: every key must be removed, so
            // this must not short-circuit on the first match.
            let mut removed = false;
            for key in keys {
                removed |= servers.remove(*key).is_some();
            }
            removed
        })
        .unwrap_or(false);

    if !removed {
        return Err(AppError::Generic(not_found.into()));
    }

    let formatted =
        serde_json::to_string_pretty(&root).map_err(|e| AppError::Generic(e.to_string()))?;
    // Same backup rule as the writers (round 6, D1 condition 1): written from
    // the bytes that were parsed, and only while they are still on disk.
    back_up_parsed(config_path, &contents, "json.bak")?;
    std::fs::write(config_path, formatted)?;
    Ok(())
}

/// TOML counterpart of [`remove_json_keys`] for Codex CLI's `[mcp_servers.*]`
/// tables, using the format-preserving `toml_edit` round-trip
/// `write_wenlan_entry_toml` writes with. Same contract, same ordering.
fn remove_toml_keys(
    config_path: &std::path::Path,
    keys: &[&str],
    not_found: &str,
) -> Result<(), AppError> {
    use toml_edit::DocumentMut;

    // Same three-answer read as `remove_json_keys` (round 6, additional
    // defect 4).
    let contents = match read_config(config_path) {
        ConfigRead::Contents(contents) => contents,
        ConfigRead::Absent => {
            return Err(AppError::Generic(
                "No config file found — nothing to remove".into(),
            ))
        }
        ConfigRead::Unreadable(error) => {
            return Err(AppError::Generic(format!(
                "Could not read {} ({error}), so Wenlan cannot tell whether there is anything to \
                 remove. Nothing was changed.",
                config_path.display()
            )))
        }
    };
    let mut doc: DocumentMut = contents.parse().map_err(|e| {
        AppError::Generic(format!("Invalid TOML in {}: {}", config_path.display(), e))
    })?;

    let removed = doc
        .get_mut("mcp_servers")
        .and_then(|servers| servers.as_table_like_mut())
        .map(|servers| {
            // Deliberately not `any`/`fold`: every key must be removed, so
            // this must not short-circuit on the first match.
            let mut removed = false;
            for key in keys {
                removed |= servers.remove(key).is_some();
            }
            removed
        })
        .unwrap_or(false);

    if !removed {
        return Err(AppError::Generic(not_found.into()));
    }

    let formatted = doc.to_string();
    back_up_parsed(config_path, &contents, "toml.bak")?;
    std::fs::write(config_path, formatted)?;
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
    remove_json_keys(
        config_path,
        &[MCP_SERVER_KEY, LEGACY_MCP_SERVER_KEY],
        "No Wenlan MCP entry found to remove",
    )
}

/// TOML variant for Codex CLI (`[mcp_servers.*]` tables) — mirrors
/// `remove_wenlan_entry`'s contract and `has_configured_entry_toml`'s key set,
/// using the same format-preserving `toml_edit` round-trip
/// `write_wenlan_entry_toml` writes with.
pub fn remove_wenlan_entry_toml(config_path: &std::path::Path) -> Result<(), AppError> {
    remove_toml_keys(
        config_path,
        &[MCP_SERVER_KEY, LEGACY_MCP_SERVER_KEY],
        "No Wenlan MCP entry found to remove",
    )
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
    remove_json_keys(
        config_path,
        &[LEGACY_MCP_SERVER_KEY],
        "No legacy origin MCP entry found to remove",
    )
}

/// TOML variant for Codex CLI (`[mcp_servers.*]` tables) — mirrors
/// `remove_legacy_origin_entry`'s contract (removes only `origin`, keeps
/// `wenlan`) using the same format-preserving `toml_edit` round-trip.
pub fn remove_legacy_origin_entry_toml(config_path: &std::path::Path) -> Result<(), AppError> {
    remove_toml_keys(
        config_path,
        &[LEGACY_MCP_SERVER_KEY],
        "No legacy origin MCP entry found to remove",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::EnvGuard;

    /// A [`Reading`] as a bool, for the many tests below that are about the
    /// FACT and not about readability.
    ///
    /// `Unreadable` PANICS rather than counting as `false`. That is the whole
    /// point of the tri-state carried into the test helper: a test whose
    /// fixture the OS refused to read must fail loudly, never quietly pass as
    /// "measured: no" — which is the shipped defect this round is about,
    /// re-enacted inside the suite.
    #[track_caller]
    fn yes(reading: Reading) -> bool {
        match reading {
            Reading::Yes => true,
            Reading::No => false,
            Reading::Unreadable { error } => {
                panic!("the fixture could not be read, which is not a `no`: {error}")
            }
        }
    }

    /// A content question's answer as a bool, for the many tests below that are
    /// about the FACT and not about whether the fixture parses.
    ///
    /// A parse failure PANICS rather than counting as `false` — the same rule
    /// as [`yes`], and round 6's D2 is the reason it is needed here too. Before
    /// this round these predicates returned a bare `bool` and every one of them
    /// answered `false` for a body nothing could parse, so a fixture with a
    /// typo in it passed a "…_false_when_no_wenlan_entry" test for the wrong
    /// reason.
    #[track_caller]
    fn parsed(answer: Result<bool, String>) -> bool {
        match answer {
            Ok(answer) => answer,
            Err(error) => {
                panic!("the fixture could not be parsed, which is not a `no`: {error}")
            }
        }
    }

    /// The parse error from a body that must NOT be measurable, for the tests
    /// that pin the failure itself.
    #[track_caller]
    fn unparseable(answer: Result<bool, String>) -> String {
        match answer {
            Err(error) => error,
            Ok(answer) => panic!(
                "expected an unparseable body to be unmeasurable, but it measured {answer:?}"
            ),
        }
    }

    impl ClientConfigPath {
        #[track_caller]
        fn unwrap(self) -> PathBuf {
            match self {
                ClientConfigPath::Known(path) => path,
                other => panic!("expected a known config path, got {other:?}"),
            }
        }
    }

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
        assert_eq!(
            client_config_path("unknown"),
            ClientConfigPath::UnknownClient
        );
    }

    #[test]
    fn test_check_already_configured_finds_legacy_origin() {
        let json =
            r#"{"mcpServers": {"origin": {"command": "npx", "args": ["-y", "origin-mcp"]}}}"#;
        assert!(parsed(has_configured_entry(json)));
    }

    #[test]
    fn test_check_already_configured_finds_wenlan() {
        let json =
            r#"{"mcpServers": {"wenlan": {"command": "npx", "args": ["-y", "wenlan-mcp"]}}}"#;
        assert!(parsed(has_configured_entry(json)));
    }

    #[test]
    fn test_check_already_configured_not_found() {
        let json = r#"{"mcpServers": {"other-server": {}}}"#;
        assert!(!parsed(has_configured_entry(json)));
    }

    #[test]
    fn test_check_already_configured_no_mcp_servers_key() {
        let json = r#"{"theme": "dark"}"#;
        assert!(!parsed(has_configured_entry(json)));
    }

    /// CHANGED IN ROUND 6 WITH THE CODE IT PINNED. This test used to be
    /// `assert!(!has_configured_entry("not json"))` — it REQUIRED a body that
    /// could not be parsed to answer exactly what a parsed body with no entry
    /// answers, which is D2 stated as a requirement. Its three siblings
    /// (`*_false_on_malformed_json`, `*_false_on_malformed_toml`) required the
    /// same thing and changed the same way.
    ///
    /// The property worth keeping from it is the one it was really guarding:
    /// garbage in a config file must not panic or hang the detector. That is
    /// still checked — the call returns — and the answer is now the honest one.
    #[test]
    fn an_unparseable_config_is_unmeasurable_not_a_measured_absence() {
        let error = unparseable(has_configured_entry("not json"));
        assert!(
            error.contains("not valid JSON"),
            "the reason reaches the user's chip and the pasted report, so it has to name the \
             file's problem: {error}"
        );
        // The same body, through the layer the UI actually reads.
        let reading =
            ConfigRead::Contents("not json".to_string()).asks(|s| has_configured_entry(s));
        assert!(
            matches!(reading, Reading::Unreadable { .. }),
            "a present-but-unparseable config is a failed measurement, not `no entry`: \
             {reading:?}"
        );
    }

    #[test]
    fn test_claude_code_plugin_enabled_matches_fresh_install_prefix() {
        // Fresh install: marketplace name is the short form.
        let json = r#"{"enabledPlugins": {"wenlan@7xuanlu": true}}"#;
        assert!(parsed(claude_code_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_code_plugin_enabled_matches_legacy_marketplace_name() {
        // A machine that added the old self-hosted marketplace before it was
        // deleted upstream (048d77a8) — must still match, since matching is
        // by the `wenlan@` prefix, not a literal marketplace name.
        let json = r#"{"enabledPlugins": {"wenlan@7xuanlu-wenlan": true}}"#;
        assert!(parsed(claude_code_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_when_disabled() {
        let json = r#"{"enabledPlugins": {"wenlan@7xuanlu": false}}"#;
        assert!(!parsed(claude_code_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_when_no_wenlan_entry() {
        let json = r#"{"enabledPlugins": {"other-plugin@somewhere": true}}"#;
        assert!(!parsed(claude_code_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_code_plugin_enabled_false_when_no_enabled_plugins_key() {
        let json = r#"{"theme": "dark"}"#;
        assert!(!parsed(claude_code_plugin_enabled(json)));
    }

    /// CHANGED IN ROUND 6 WITH THE CODE — see
    /// `an_unparseable_config_is_unmeasurable_not_a_measured_absence`. This one
    /// is the case Codex's D6a is built on: a `~/.claude/settings.json` that
    /// would not parse used to be indistinguishable from one with the plugin
    /// switched off, and "switched off" is what licenses writing a raw entry.
    #[test]
    fn an_unparseable_claude_code_settings_file_is_unmeasurable() {
        let error = unparseable(claude_code_plugin_enabled("not json"));
        assert!(error.contains("not valid JSON"), "{error}");
    }

    #[test]
    fn test_codex_cli_plugin_enabled_matches_pre_rename_marketplace() {
        let toml = "[plugins.\"wenlan@wenlan-local\"]\nenabled = true\n";
        assert!(parsed(codex_cli_plugin_enabled(toml)));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_matches_post_rename_marketplace() {
        // 7xuanlu/wenlan#348 renames the marketplace to match Claude's — must
        // still match, since matching is by the `wenlan@` prefix, not a
        // literal marketplace name.
        let toml = "[plugins.\"wenlan@7xuanlu-wenlan\"]\nenabled = true\n";
        assert!(parsed(codex_cli_plugin_enabled(toml)));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_when_disabled() {
        let toml = "[plugins.\"wenlan@wenlan-local\"]\nenabled = false\n";
        assert!(!parsed(codex_cli_plugin_enabled(toml)));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_when_no_wenlan_entry() {
        let toml = "[plugins.\"other@somewhere\"]\nenabled = true\n";
        assert!(!parsed(codex_cli_plugin_enabled(toml)));
    }

    #[test]
    fn test_codex_cli_plugin_enabled_false_when_no_plugins_key() {
        let toml = "model = \"gpt-5.5\"\n";
        assert!(!parsed(codex_cli_plugin_enabled(toml)));
    }

    /// CHANGED IN ROUND 6 WITH THE CODE, same reason as the JSON siblings.
    #[test]
    fn an_unparseable_codex_config_is_unmeasurable() {
        let error = unparseable(codex_cli_plugin_enabled("not toml ["));
        assert!(error.contains("not valid TOML"), "{error}");
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
        assert!(parsed(
            claude_desktop_plugin_enabled(manifest_with_wenlan())
        ));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_for_similar_name() {
        // Guards against a `starts_with`/`contains` match instead of exact
        // equality: "wenlan-old" must not count as "wenlan".
        let json =
            r#"{"plugins": [{"id": "p1", "name": "wenlan-old", "marketplaceName": "wenlan"}]}"#;
        assert!(!parsed(claude_desktop_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_when_only_marketplace_name_matches() {
        // The entry's `name` is "other-plugin"; only `marketplaceName` says
        // "wenlan". Matching the wrong field would false-positive here.
        let json =
            r#"{"plugins": [{"id": "p1", "name": "other-plugin", "marketplaceName": "wenlan"}]}"#;
        assert!(!parsed(claude_desktop_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_case_mismatch() {
        let json = r#"{"plugins": [{"id": "p1", "name": "Wenlan"}]}"#;
        assert!(!parsed(claude_desktop_plugin_enabled(json)));
    }

    #[test]
    fn test_claude_desktop_plugin_enabled_false_when_no_plugins_key() {
        assert!(!parsed(claude_desktop_plugin_enabled(
            r#"{"lastUpdated": 1}"#
        )));
    }

    /// CHANGED IN ROUND 6 WITH THE CODE, same reason as the siblings.
    #[test]
    fn an_unparseable_desktop_manifest_is_unmeasurable() {
        let error = unparseable(claude_desktop_plugin_enabled("not json"));
        assert!(error.contains("not valid JSON"), "{error}");
    }

    #[test]
    fn test_claude_desktop_account_id_extracts_last_known_account_uuid() {
        let json = r#"{"lastKnownAccountUuid": "acct-123", "locale": "en-US"}"#;
        assert_eq!(
            claude_desktop_account_id(json),
            Ok(Some("acct-123".to_string()))
        );
    }

    #[test]
    fn test_claude_desktop_account_id_none_when_key_missing() {
        assert_eq!(
            claude_desktop_account_id(r#"{"locale": "en-US"}"#),
            Ok(None)
        );
    }

    /// CHANGED IN ROUND 6 WITH THE CODE. This required a malformed
    /// `config.json` to answer `None` — the same value a file that parsed and
    /// pinned no account answers — and the caller turned that `None` into
    /// `Reading::No`, i.e. "Claude Desktop does not have the Wenlan plugin".
    /// The account id names the sessions directory, so an unparseable
    /// `config.json` means the manifest scan never happened at all.
    #[test]
    fn an_unparseable_desktop_config_cannot_answer_which_account_is_pinned() {
        let error = claude_desktop_account_id("not json")
            .expect_err("an unparseable config.json pins no account it could report");
        assert!(error.contains("not valid JSON"), "{error}");
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
        assert!(yes(claude_desktop_plugin_enabled_in_sessions_dir(
            tmp.path()
        )));
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

        assert!(yes(claude_desktop_plugin_enabled_in_sessions_dir(
            tmp.path()
        )));
    }

    #[test]
    fn test_sessions_dir_false_when_dir_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!yes(claude_desktop_plugin_enabled_in_sessions_dir(
            &tmp.path().join("does-not-exist")
        )));
    }

    #[test]
    fn test_sessions_dir_false_when_no_rpm_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let session_dir = tmp.path().join("sess-1");
        std::fs::create_dir_all(&session_dir).unwrap();
        // manifest.json exists but not under rpm/
        std::fs::write(session_dir.join("manifest.json"), manifest_with_wenlan()).unwrap();
        assert!(!yes(claude_desktop_plugin_enabled_in_sessions_dir(
            tmp.path()
        )));
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

        assert!(yes(claude_desktop_plugin_enabled_in_sessions_dir(
            tmp.path()
        )));
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
        assert!(yes(claude_desktop_plugin_enabled_for_support_dir(
            tmp.path()
        )));
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

        assert!(!yes(claude_desktop_plugin_enabled_for_support_dir(
            tmp.path()
        )));
    }

    #[test]
    fn test_support_dir_false_when_config_json_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!yes(claude_desktop_plugin_enabled_for_support_dir(
            tmp.path()
        )));
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

        assert!(!yes(claude_desktop_plugin_enabled_for_support_dir(
            tmp.path()
        )));
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
        assert_eq!(
            claude_desktop.already_configured,
            Reading::Yes,
            "expected a MEASURED yes: this machine has the Wenlan chat-side plugin installed"
        );
    }

    /// `wenlan-mcp` under `dir`, named from `std::env::consts::EXE_SUFFIX`
    /// rather than from `wenlan_mcp_in`. A fixture built from the code under
    /// test only ever agrees with itself; std's suffix is the independent
    /// answer that lets these tests fail when the resolver forgets `.exe`.
    fn installed_wenlan_mcp(dir: &Path) -> PathBuf {
        dir.join(format!("wenlan-mcp{}", std::env::consts::EXE_SUFFIX))
    }

    /// Write a fixture that is plausibly a program: non-empty, and on Unix
    /// carrying the execute bit.
    ///
    /// ROUND 4, DEFECT E, THE TEST HALF. These fixtures used to be
    /// `std::fs::write(path, b"")` — ZERO BYTES — and the assertions below
    /// certified them as the resolved `wenlan-mcp` binary. The test encoded the
    /// defect as the expected behaviour: a probe that answers `File` for an
    /// empty file is a probe that will write an unrunnable command into a real
    /// user's client config, and every test agreed that was correct.
    ///
    /// `MZ` is the DOS/PE signature; nothing here parses it, it is simply
    /// honest fixture content rather than nothing at all.
    fn write_binary_fixture(path: &Path) {
        std::fs::write(path, b"MZ\x90\x00 wenlan-mcp test fixture\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    /// Unwrap the entry a decision says to write, failing loudly (with the
    /// unreadable paths) when the decision was "write nothing".
    fn written_entry(decision: McpEntryDecision) -> WenlanMcpEntry {
        match decision {
            McpEntryDecision::Write { entry, .. } => entry,
            McpEntryDecision::PreserveExisting { unmeasured } => panic!(
                "expected an entry to be written, but the resolution was unresolved: \
                 {unmeasured:?}"
            ),
        }
    }

    /// Create a fixture home whose `.wenlan/bin` holds a real `wenlan-mcp`
    /// executable file, and point the resolver at it. Returns the binary path.
    fn install_wenlan_mcp_into(home: &Path) -> PathBuf {
        let bin_dir = home.join(".wenlan").join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let installed = installed_wenlan_mcp(&bin_dir);
        write_binary_fixture(&installed);
        installed
    }

    /// The entry the app writes into client configs must be the installed
    /// binary whenever one exists. `npx` here is a failure, not an
    /// alternative: it is a network dependency and a version-skew hazard, and
    /// it is exactly what a Windows install got while the candidate paths
    /// carried no `.exe` suffix. This test used to accept either answer.
    #[test]
    #[serial_test::serial]
    fn wenlan_mcp_entry_takes_the_installed_binary_over_npx() {
        let _env = EnvGuard::capture(&[MCP_RESOLVER_HOME_ENV, "WENLAN_MCP_DEV_BIN"]);
        let tmp = tempfile::tempdir().unwrap();
        let installed = install_wenlan_mcp_into(tmp.path());
        std::env::remove_var("WENLAN_MCP_DEV_BIN");
        std::env::set_var(MCP_RESOLVER_HOME_ENV, tmp.path());

        let report = wenlan_mcp_entry().expect("a readable installed binary resolves");
        // Round 6, D3's boundary defect: an empty `undetermined` is itself a
        // measurement — every resolver input was read. This fixture determines
        // all three, so the report has to SAY that rather than have nowhere to
        // say it.
        assert!(
            report.undetermined.is_empty(),
            "nothing was undetermined in this fixture: {:?}",
            report.undetermined
        );
        let entry = report.entry;

        assert_ne!(
            entry.command,
            "npx",
            "an installed {} was ignored and the npx fallback was written instead",
            installed.display()
        );
        // Compared as paths: `join` mixes separators on Windows, and the
        // question here is which file was chosen, not how it was spelled.
        assert_eq!(Path::new(&entry.command), installed);
        assert!(entry.args.is_empty());
    }

    /// The other arm, made just as deterministic: with nothing installed the
    /// entry is the pinned `npx` fallback, typed as command + args.
    #[test]
    #[serial_test::serial]
    fn wenlan_mcp_entry_is_the_pinned_npx_fallback_when_nothing_is_installed() {
        let _env = EnvGuard::capture(&[MCP_RESOLVER_HOME_ENV, "WENLAN_MCP_DEV_BIN"]);
        let tmp = tempfile::tempdir().unwrap();
        std::env::remove_var("WENLAN_MCP_DEV_BIN");
        std::env::set_var(MCP_RESOLVER_HOME_ENV, tmp.path());

        let entry = wenlan_mcp_entry()
            .expect("an empty tree is a MEASURED absence, not a failure")
            .entry;

        assert_eq!(entry.command, "npx");
        assert_eq!(entry.args.len(), 2);
        assert_eq!(entry.args[0], "-y");
        assert!(entry.args[1].starts_with("wenlan-mcp@^"));
    }

    /// The bundled binary — the one a Windows installer drops next to the app
    /// exe — must resolve on every platform. Without the executable suffix
    /// this returns `None` on Windows and the caller writes `npx`.
    #[test]
    fn bundled_wenlan_mcp_next_to_the_app_exe_resolves_on_this_platform() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("Programs").join("Wenlan");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let bundled = installed_wenlan_mcp(&exe_dir);
        write_binary_fixture(&bundled);

        assert_eq!(
            resolve_wenlan_mcp(None, None, Some(&exe_dir), probe_candidate),
            McpBinaryResolution::Found {
                path: bundled.clone(),
                undetermined: Vec::new(),
            },
            "the bundled {} was not found; the app would write the npx fallback",
            bundled.display()
        );
    }

    /// A directory named `wenlan-mcp[.exe]` is not the binary. `Path::exists()`
    /// said it was, and the resolver wrote that directory into the user's MCP
    /// client config as the command to run.
    #[test]
    fn a_directory_named_like_the_binary_is_never_resolved_as_the_command() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("Programs").join("Wenlan");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let impostor = installed_wenlan_mcp(&exe_dir);
        std::fs::create_dir_all(&impostor).unwrap();

        assert_eq!(
            probe_candidate(&impostor),
            CandidateProbe::NotAFile,
            "a directory at the candidate path must not probe as a usable binary"
        );
        assert_eq!(
            resolve_wenlan_mcp(None, None, Some(&exe_dir), probe_candidate),
            McpBinaryResolution::NoneInstalled,
            "a directory named {} must not resolve as the wenlan-mcp binary",
            impostor.display()
        );
        let entry = written_entry(wenlan_mcp_entry_for(
            resolve_wenlan_mcp(None, None, Some(&exe_dir), probe_candidate),
            "wenlan-mcp@^9.9.9",
        ));
        assert_eq!(
            entry.command, "npx",
            "a directory was written into a client config as the command"
        );
    }

    /// A candidate the OS refuses to stat is not an absence. Stubbed rather
    /// than produced with real ACLs, because the permission shape differs per
    /// platform and the branch under test is the resolver's, not the OS's.
    #[test]
    fn an_unreadable_candidate_is_not_a_measured_absence() {
        let home = PathBuf::from("/Users/someone");
        let installed = installed_wenlan_mcp(&home.join(".wenlan/bin"));
        let denied = installed.clone();
        let resolution = resolve_wenlan_mcp(Some(home.as_path()), None, None, move |p| {
            if p == denied {
                CandidateProbe::Unreadable {
                    error: "Access is denied. (os error 5)".to_string(),
                }
            } else {
                CandidateProbe::Absent
            }
        });
        match &resolution {
            McpBinaryResolution::Unresolved(unmeasured) => {
                assert_eq!(unmeasured.unreadable.len(), 1);
                assert_eq!(unmeasured.unreadable[0].0, installed);
                assert!(unmeasured.unreadable[0].1.contains("os error 5"));
                assert!(
                    unmeasured.undetermined.is_empty(),
                    "every input was determined; only the path could not be read"
                );
            }
            other => panic!("an unreadable candidate must not resolve as absent: {other:?}"),
        }
        assert_ne!(
            resolution,
            McpBinaryResolution::NoneInstalled,
            "'could not look' must never be reported as 'nothing is installed'"
        );
    }

    /// The documented ranking: an unreadable candidate does not end the
    /// search, so a later candidate that IS a measured file still wins. Before
    /// this, `exists()` returning false for a denied stat had the same effect
    /// — but it also had the same effect when nothing later existed, and that
    /// case was the silent `npx`.
    #[test]
    fn an_unreadable_candidate_does_not_outrank_a_later_measured_file() {
        let home = PathBuf::from("/Users/someone");
        let installed = installed_wenlan_mcp(&home.join(".wenlan/bin"));
        let cargo = installed_wenlan_mcp(&home.join(".cargo/bin"));
        let denied = installed.clone();
        let real = cargo.clone();
        let resolution = resolve_wenlan_mcp(Some(home.as_path()), None, None, move |p| {
            if p == denied {
                CandidateProbe::Unreadable {
                    error: "Access is denied. (os error 5)".to_string(),
                }
            } else if p == real {
                CandidateProbe::File
            } else {
                CandidateProbe::Absent
            }
        });
        assert_eq!(
            resolution,
            McpBinaryResolution::Found {
                path: cargo,
                undetermined: Vec::new()
            }
        );
    }

    /// `Absent` is the only error state that is an absence; everything else the
    /// filesystem can say is its own answer.
    #[test]
    fn probe_candidate_separates_absence_from_a_failed_look() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("wenlan-mcp");
        write_binary_fixture(&file);
        assert_eq!(probe_candidate(&file), CandidateProbe::File);
        assert_eq!(probe_candidate(tmp.path()), CandidateProbe::NotAFile);
        assert_eq!(
            probe_candidate(&tmp.path().join("nope")),
            CandidateProbe::Absent
        );
    }

    /// C1.4, AND IT IS THE SHIPPED-USER ONE. Round 4 hardened the tri-state at
    /// the PROBE and left it collapsing one layer upstream, at candidate
    /// CONSTRUCTION: `dirs::home_dir()` answering `None`, `current_exe()`
    /// failing, and a non-Unicode `WENLAN_MCP_DEV_BIN` were all flattened into
    /// `None` by `Option`/`.ok()`. The candidates that hang off those inputs
    /// were then never built, never probed, and never appeared in the trail --
    /// and a search that had looked at NOTHING could end in `NoneInstalled`
    /// and write `npx` over a user's working local command.
    ///
    /// The probe here panics if it is ever called, which states the fixture
    /// exactly: no candidate path exists to probe, because none could be
    /// constructed. The answer must still not be "nothing is installed".
    #[test]
    fn an_input_that_could_not_be_determined_is_never_none_installed() {
        let never_probed = |path: &Path| -> CandidateProbe {
            panic!("no candidate could be constructed, so nothing should be probed: {path:?}")
        };

        let no_home = ResolverInputs {
            home: RootInput::Undetermined("the platform would not report a home directory".into()),
            dev_bin: RootInput::NotSet,
            exe_dir: RootInput::NotSet,
        };
        let (decision, trail) =
            wenlan_mcp_decision_from(&no_home, never_probed, "wenlan-mcp@^9.9.9");
        assert!(
            trail.is_empty(),
            "the fixture is not staging the defect if candidates were built: {trail:?}"
        );
        match decision {
            McpEntryDecision::PreserveExisting { unmeasured } => {
                assert!(unmeasured.unreadable.is_empty());
                assert_eq!(unmeasured.undetermined.len(), 1);
                assert_eq!(unmeasured.undetermined[0].input, "the home directory");
                assert_eq!(
                    unmeasured.undetermined[0].blocked, "installed and cargo",
                    "the message has to name the candidates that were never checked"
                );
                let message = unresolved_message(&unmeasured);
                assert!(
                    message.contains("home directory") && message.contains("unchanged"),
                    "the user-facing message must name what could not be determined and say \
                     nothing was written: {message}"
                );
            }
            McpEntryDecision::Write { entry, .. } => panic!(
                "a search that could not even build its candidate paths produced `{} {}` -- a \
                 measured-absence outcome manufactured from an unread input",
                entry.command,
                entry.args.join(" ")
            ),
        }

        // Same shape, different input: `env::var` returns `Err` for BOTH
        // "unset" and "not valid Unicode", and only the first is an absence.
        let bad_dev_bin = ResolverInputs {
            home: RootInput::NotSet,
            dev_bin: RootInput::Undetermined("not valid unicode".into()),
            exe_dir: RootInput::NotSet,
        };
        match wenlan_mcp_decision_from(&bad_dev_bin, never_probed, "wenlan-mcp@^9.9.9").0 {
            McpEntryDecision::PreserveExisting { unmeasured } => {
                assert_eq!(unmeasured.undetermined[0].input, "WENLAN_MCP_DEV_BIN");
            }
            other => panic!("an unreadable dev override must not be an absence: {other:?}"),
        }

        // The control, so this is not just "always refuse": when every input is
        // genuinely NOT SET, nothing failed, and `npx` is the right answer.
        let all_absent = ResolverInputs {
            home: RootInput::NotSet,
            dev_bin: RootInput::NotSet,
            exe_dir: RootInput::NotSet,
        };
        assert_eq!(
            written_entry(
                wenlan_mcp_decision_from(&all_absent, never_probed, "wenlan-mcp@^9.9.9").0
            )
            .command,
            "npx",
            "a measured absence must still resolve to the npx entry"
        );
    }

    /// ROUND 5, DEFECT D4 — a `Found` short-circuited the undetermined inputs
    /// out of existence.
    ///
    /// `inputs.undetermined()` was consulted only in the resolver's EMPTY
    /// branch, so the moment any candidate probed as a file the failed input
    /// stopped being reported. It could not even be inferred from the trail:
    /// an input that could not be determined builds no candidate path, so it
    /// has no trail row to be missing from — a three-row trail beside a chosen
    /// command looks exactly like a complete search.
    ///
    /// This is Codex's stated input verbatim: `home` determined, `dev_bin`
    /// undetermined ("not valid Unicode"), `exe_dir` not set, and a real file
    /// under `home`. The developer's override was silently skipped; the only
    /// question is whether anything says so.
    #[test]
    fn a_found_binary_still_reports_an_input_that_could_not_be_read() {
        let tmp = tempfile::tempdir().unwrap();
        let installed = install_wenlan_mcp_into(tmp.path());

        let inputs = ResolverInputs {
            home: RootInput::Known(tmp.path().to_path_buf()),
            dev_bin: RootInput::Undetermined("not valid Unicode".to_string()),
            exe_dir: RootInput::NotSet,
        };

        let resolved = resolve_wenlan_mcp_with_trail(&inputs, probe_candidate);
        match &resolved.resolution {
            McpBinaryResolution::Found { path, undetermined } => {
                // Compared as filesystem objects, not as strings: the resolver
                // spells this `<home>\.wenlan/bin/wenlan-mcp.exe` (a literal
                // `/`-joined tail) while the fixture spells it with separators
                // — the same file, two spellings. That difference is Codex's
                // third residual (per-spelling, not per-object, deduplication)
                // and is out of scope here; the test must not depend on it.
                assert_eq!(
                    std::fs::canonicalize(path).unwrap(),
                    std::fs::canonicalize(&installed).unwrap()
                );
                assert_eq!(
                    undetermined.len(),
                    1,
                    "a hit under one input erased the input that could not be read: {undetermined:?}"
                );
                assert_eq!(undetermined[0].input, "WENLAN_MCP_DEV_BIN");
                assert_eq!(undetermined[0].blocked, "WENLAN_MCP_DEV_BIN");
                assert!(undetermined[0].error.contains("not valid Unicode"));
            }
            other => panic!("the fixture must resolve to the installed binary: {other:?}"),
        }

        // The trail cannot stand in for it: `WENLAN_MCP_DEV_BIN` has no
        // candidate row, because no path could be built from it. That is the
        // whole reason the resolution has to carry it separately.
        assert!(
            !resolved
                .trail
                .iter()
                .any(|c| c.source == "WENLAN_MCP_DEV_BIN"),
            "the fixture is not staging the defect if a dev-override candidate was built"
        );

        // And it survives the step that reaches the writers and the wire.
        match wenlan_mcp_decision_from(&inputs, probe_candidate, "wenlan-mcp@^9.9.9").0 {
            McpEntryDecision::Write {
                entry,
                undetermined,
            } => {
                assert_eq!(
                    std::fs::canonicalize(&entry.command).unwrap(),
                    std::fs::canonicalize(&installed).unwrap()
                );
                assert_eq!(
                    undetermined.len(),
                    1,
                    "the decision the writers and the diagnostics wire act on dropped it"
                );
            }
            other => panic!("a measured file must still be written: {other:?}"),
        }

        // The control, so this is not "always report something": with the same
        // find and every input determined, there is nothing to report and the
        // successful search says so.
        let clean = determined_inputs(Some(tmp.path()), None, None);
        match resolve_wenlan_mcp_with_trail(&clean, probe_candidate).resolution {
            McpBinaryResolution::Found { undetermined, .. } => assert!(
                undetermined.is_empty(),
                "a fully determined search must not invent a failure: {undetermined:?}"
            ),
            other => panic!("the control must also find the installed binary: {other:?}"),
        }
    }

    /// The other half of C1.4: the classification itself. `.ok()` on
    /// `env::var` and on `current_exe()` is what erased the difference between
    /// "not set" and "could not be read", so the arms are pinned here directly
    /// rather than through whatever the host process happens to have.
    #[test]
    fn a_read_that_failed_is_not_an_input_that_is_absent() {
        let determined = ResolverInputs::from_reads(
            Some(PathBuf::from("/home/someone")),
            Err(std::env::VarError::NotPresent),
            Ok(PathBuf::from(
                "/Applications/Wenlan.app/Contents/MacOS/wenlan",
            )),
        );
        assert_eq!(determined.dev_bin, RootInput::NotSet);
        assert!(
            determined.undetermined().is_empty(),
            "an unset env var is a measured absence and must not be reported as a failure"
        );

        let failed = ResolverInputs::from_reads(
            None,
            Err(std::env::VarError::NotUnicode("\u{fffd}".into())),
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "current_exe: Access is denied. (os error 5)",
            )),
        );
        let undetermined = failed.undetermined();
        let names: Vec<&str> = undetermined.iter().map(|u| u.input.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "WENLAN_MCP_DEV_BIN",
                "the home directory",
                "the application's own directory",
            ],
            "every read that failed must survive as a failure, not be flattened into 'not set'"
        );
    }

    /// C1.6. The candidate list is four SLOTS and two of them can name the same
    /// FILE -- `WENLAN_MCP_DEV_BIN` pointed at the installed binary is the
    /// ordinary developer setup. Probing that file twice is two instants, so
    /// the trail could carry the same path as both `file` and `unreadable`,
    /// contradicting itself and the command printed beside it. One reading per
    /// filesystem object; both slots still shown, sharing it.
    #[test]
    fn the_same_file_named_by_two_slots_is_probed_once() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        let installed = install_wenlan_mcp_into(home);
        // ROUND 6, ADDITIONAL DEFECT 3 — THE FALSE-GREEN CONTROL. This fixture
        // used to pass `installed.to_str()` as the dev override: the SAME
        // SPELLING, twice. `HashMap<PathBuf, _>` keys on pathname equality, so
        // the test passed by construction and could not fail for the reason
        // its own name states — it measured "one reading per vector entry
        // spelled identically", not "one reading per FILESYSTEM OBJECT".
        //
        // Two spellings, one file: the ordinary developer case of an override
        // built by walking a directory back (`$bin/../bin/wenlan-mcp`).
        //
        // It has to be `..` rather than `.`: `Path`'s `Eq` and `Hash` both run
        // over `components()`, which DROPS a `CurDir` component, so `a/./b`
        // already hashes equal to `a/b` and would have passed against the old
        // pathname-keyed map. `ParentDir` components are kept, so these two are
        // genuinely different keys — asserted below, so the fixture cannot
        // silently decay back into the one that could not fail.
        let walked = home
            .join(".wenlan")
            .join("bin")
            .join("..")
            .join("bin")
            .join(installed.file_name().unwrap());
        assert_ne!(
            walked, installed,
            "the fixture must be two SPELLINGS of one file; if these are equal the test is \
             back to measuring pathname equality against itself"
        );

        let probes: std::sync::Mutex<std::collections::HashMap<PathBuf, usize>> =
            std::sync::Mutex::new(std::collections::HashMap::new());
        let counting = |path: &Path| {
            *probes
                .lock()
                .unwrap()
                .entry(path.to_path_buf())
                .or_insert(0) += 1;
            probe_candidate(path)
        };

        let (decision, trail) = wenlan_mcp_decision_for(
            Some(home),
            Some(walked.to_str().unwrap()),
            None,
            counting,
            "wenlan-mcp@^9.9.9",
        );

        let readings_of_the_object = {
            let probes = probes.lock().unwrap();
            probes.get(&installed).copied().unwrap_or(0) + probes.get(&walked).copied().unwrap_or(0)
        };
        assert_eq!(
            readings_of_the_object, 1,
            "one file, named twice, was probed {readings_of_the_object} time(s); two readings of \
             one object is two instants, which is how a trail comes to contradict the command \
             beside it"
        );

        let for_this_file: Vec<&ProbedCandidate> = trail
            .iter()
            .filter(|c| c.path == installed || c.path == walked)
            .collect();
        assert_eq!(
            for_this_file.len(),
            2,
            "both slots must still appear in the trail -- the user should see that the dev \
             override and the installed path are the same file"
        );
        assert_eq!(
            for_this_file[0].state, for_this_file[1].state,
            "one object, one reading: the two slots must carry the SAME state"
        );
        // The dev override still wins the ranking, spelled the way the user
        // spelled it. Sharing a reading must not rewrite anyone's path.
        assert_eq!(written_entry(decision).command, walked.to_str().unwrap());
    }

    /// DEFECT E. `metadata.is_file()` is a FILE witness, not an EXECUTABLE
    /// witness, and the resolver hands its winner to a client as a command to
    /// run. A zero-byte file is not a program on any operating system, and this
    /// is the check that says so — the same check whose absence let the FIXTURES
    /// in this file certify `b""` as the installed binary.
    ///
    /// It must reach the resolver, not just the probe: a `NotExecutable`
    /// candidate is measured-unusable, so the search moves on and — with no
    /// unreadable candidate anywhere — ends in the MEASURED `NoneInstalled`,
    /// never `Unresolved`. Nothing failed; the answer is just no.
    #[test]
    fn an_empty_file_is_never_resolved_as_the_binary() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("Programs").join("Wenlan");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let empty = installed_wenlan_mcp(&exe_dir);
        std::fs::write(&empty, b"").unwrap();

        match probe_candidate(&empty) {
            CandidateProbe::NotExecutable { reason } => {
                assert!(
                    reason.contains("empty"),
                    "the reason must say what was wrong: {reason}"
                );
            }
            other => panic!("a zero-byte file must not probe as a usable binary: {other:?}"),
        }
        assert_eq!(
            resolve_wenlan_mcp(None, None, Some(&exe_dir), probe_candidate),
            McpBinaryResolution::NoneInstalled,
            "an empty {} was resolved as the wenlan-mcp binary and would be written into a \
             client config as the command to run",
            empty.display()
        );
    }

    /// The Unix half of the same rule, and the one that has a real execute bit
    /// to read. Skipped on Windows on purpose: there IS no execute bit there,
    /// `std::fs::Metadata` exposes nothing equivalent, and a check invented to
    /// make the platforms look symmetrical would be a witness that cannot fail.
    /// What Windows can establish is non-emptiness, which the test above covers;
    /// a non-empty but corrupt `.exe` remains an unfixed residual there.
    #[cfg(unix)]
    #[test]
    fn a_file_with_no_execute_bit_is_never_resolved_as_the_binary() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("Programs").join("Wenlan");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let not_executable = installed_wenlan_mcp(&exe_dir);
        std::fs::write(&not_executable, b"MZ\x90\x00 real content\n").unwrap();
        std::fs::set_permissions(&not_executable, std::fs::Permissions::from_mode(0o644)).unwrap();

        match probe_candidate(&not_executable) {
            CandidateProbe::NotExecutable { reason } => {
                assert!(reason.contains("execute"), "reason was: {reason}");
            }
            other => panic!("a non-executable file must not probe as a usable binary: {other:?}"),
        }
        assert_eq!(
            resolve_wenlan_mcp(None, None, Some(&exe_dir), probe_candidate),
            McpBinaryResolution::NoneInstalled
        );
    }

    /// Every probed candidate carries the platform's executable suffix. The
    /// dev override is exempt: it is a full path the developer supplies.
    #[test]
    fn wenlan_mcp_candidates_carry_the_platform_executable_suffix() {
        let home = PathBuf::from("/Users/someone");
        let expected = format!("wenlan-mcp{}", std::env::consts::EXE_SUFFIX);
        for (path, source) in wenlan_mcp_candidate_sources(
            Some(home.as_path()),
            None,
            Some(Path::new("/Applications/Wenlan.app/Contents/MacOS")),
        ) {
            assert_eq!(
                path.file_name().unwrap().to_string_lossy(),
                expected.as_str(),
                "the {source} candidate cannot match a real install on this platform: {}",
                path.display()
            );
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
            &installed_wenlan_mcp(&home.join(".wenlan/bin"))
        );
        assert!(candidates.contains(&installed_wenlan_mcp(&home.join(".cargo/bin"))));
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
        assert_eq!(
            candidates[1],
            installed_wenlan_mcp(&home.join(".wenlan/bin"))
        );
        assert_eq!(
            candidates[2],
            installed_wenlan_mcp(Path::new("/Applications/Wenlan.app/Contents/MacOS"))
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
            vec![installed_wenlan_mcp(Path::new(
                "/Applications/Wenlan.app/Contents/MacOS"
            ))]
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
        let entry = written_entry(wenlan_mcp_entry_for(
            McpBinaryResolution::NoneInstalled,
            &pinned_wenlan_mcp_package(BACKEND_VERSION_PIN),
        ));
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
        let entry = written_entry(wenlan_mcp_entry_for(
            McpBinaryResolution::Found {
                path: PathBuf::from("/Users/someone/.wenlan/bin/wenlan-mcp"),
                undetermined: Vec::new(),
            },
            "wenlan-mcp@^9.9.9",
        ));
        assert_eq!(entry.command, "/Users/someone/.wenlan/bin/wenlan-mcp");
        assert!(entry.args.is_empty());
    }

    /// DEFECT D, at the decision. `Unresolved` used to produce the SAME `npx`
    /// entry as `NoneInstalled` — a measured absence manufactured out of a
    /// failed look. The third value existed and was discarded one function
    /// later, which is the exact shape Codex's round-3 review named.
    ///
    /// It must now be "write nothing". `npx` is not a safe default: it needs
    /// Node and a network, and the user whose local binary was momentarily
    /// unstatable has neither guaranteed.
    #[test]
    fn an_unresolved_search_writes_nothing_at_all() {
        let denied = PathBuf::from("/Users/someone/.wenlan/bin/wenlan-mcp");
        let unresolved = McpBinaryResolution::Unresolved(Unmeasured {
            unreadable: vec![(denied.clone(), "Access is denied. (os error 5)".to_string())],
            undetermined: Vec::new(),
        });
        assert_ne!(unresolved, McpBinaryResolution::NoneInstalled);

        match wenlan_mcp_entry_for(unresolved, "wenlan-mcp@^9.9.9") {
            McpEntryDecision::PreserveExisting { unmeasured } => {
                assert_eq!(unmeasured.unreadable.len(), 1);
                assert_eq!(unmeasured.unreadable[0].0, denied);
                let message = unresolved_message(&unmeasured);
                assert!(
                    message.contains("os error 5") && message.contains("unchanged"),
                    "the user-facing message must name the failure and say nothing changed: \
                     {message}"
                );
            }
            McpEntryDecision::Write { entry, .. } => panic!(
                "a candidate that could not be LOOKED AT produced a written entry `{} {}` — a \
                 measured-absence outcome manufactured from a failed measurement",
                entry.command,
                entry.args.join(" ")
            ),
        }
    }

    /// DEFECT D, at the mutation. The scenario Codex described end to end: a
    /// user HAS a local `wenlan-mcp` in a client config, and the candidate is
    /// momentarily unstatable (ACL, antivirus, disconnected network path).
    /// Rewriting that config to `npx` breaks an offline machine, or one with no
    /// Node, that was working a second earlier.
    ///
    /// The file must come back BYTE-IDENTICAL, and no `.bak` may be left
    /// either: a backup implies a change happened.
    #[test]
    fn an_unreadable_candidate_leaves_an_existing_config_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let unresolved = || McpEntryDecision::PreserveExisting {
            unmeasured: Unmeasured {
                unreadable: vec![(
                    PathBuf::from("/Users/someone/.wenlan/bin/wenlan-mcp"),
                    "Access is denied. (os error 5)".to_string(),
                )],
                undetermined: Vec::new(),
            },
        };

        let config_path = tmp.path().join("config.json");
        let existing = "{\n  \"mcpServers\": {\n    \"wenlan\": {\n      \"command\": \
                        \"/opt/wenlan/bin/wenlan-mcp\",\n      \"args\": []\n    }\n  }\n}\n";
        std::fs::write(&config_path, existing).unwrap();

        let err = write_wenlan_entry_with(&config_path, false, unresolved())
            .expect_err("an unresolvable binary must not silently rewrite a client config");
        assert!(
            err.to_string().contains("unchanged"),
            "the error must tell the user nothing was written: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(&config_path).unwrap(),
            existing,
            "the user's working local command was overwritten with a guess"
        );
        assert!(
            !config_path.with_extension("json.bak").exists(),
            "a backup was written for a change that never happened"
        );

        // Same rule for the Codex TOML writer.
        let toml_path = tmp.path().join("config.toml");
        let existing_toml = "# hand-written\n[mcp_servers.wenlan]\ncommand = \
                             \"/opt/wenlan/bin/wenlan-mcp\"\nargs = []\n";
        std::fs::write(&toml_path, existing_toml).unwrap();
        write_wenlan_entry_toml_with(&toml_path, unresolved())
            .expect_err("same rule for the TOML writer");
        assert_eq!(std::fs::read_to_string(&toml_path).unwrap(), existing_toml);
        assert!(!toml_path.with_extension("toml.bak").exists());

        // A config file that does NOT exist yet must not be created either:
        // "write nothing" has to mean nothing, not an empty skeleton.
        let fresh = tmp.path().join("fresh.json");
        write_wenlan_entry_with(&fresh, false, unresolved()).unwrap_err();
        assert!(
            !fresh.exists(),
            "an unresolvable search created a config file"
        );
    }

    /// A decision that is definitely `Write`, so these tests measure the
    /// WRITER's ordering rather than whatever binary the host running them
    /// happens to have installed.
    #[cfg(test)]
    fn staged_write() -> McpEntryDecision {
        McpEntryDecision::Write {
            entry: WenlanMcpEntry {
                command: "/opt/wenlan/bin/wenlan-mcp".to_string(),
                args: Vec::new(),
            },
            undetermined: Vec::new(),
        }
    }

    /// ROUND 5, DEFECT 1 — the writers destroyed the user's only recoverable
    /// copy on the way to reporting the problem.
    ///
    /// The scenario is the ordinary one: `config.json` got corrupted (a crash
    /// mid-write, a bad hand-edit) and `config.json.bak` still holds the last
    /// GOOD configuration — which is exactly what a `.bak` is for. The writer
    /// copied the current file over the backup FIRST and only then read and
    /// parsed it, so the malformed bytes landed on top of the good ones and
    /// the call returned `Invalid JSON`. The recovery copy was gone at the one
    /// moment it mattered.
    ///
    /// The old tests could not see this: they asserted `is_err()`, and the
    /// destructive path returns exactly the same `Err`. The assertion that
    /// separates the two is on the BACKUP's bytes, not on the return value.
    #[test]
    fn a_malformed_config_does_not_overwrite_the_last_good_backup() {
        let tmp = tempfile::tempdir().unwrap();

        let config_path = tmp.path().join("config.json");
        let backup_path = tmp.path().join("config.json.bak");
        let good = "{\n  \"mcpServers\": {\n    \"wenlan\": {\n      \"command\": \
                    \"/opt/wenlan/bin/wenlan-mcp\",\n      \"args\": []\n    }\n  }\n}\n";
        let malformed = "{\n  \"mcpServers\": {\n    \"wenlan\": {\n  <<<< truncated";
        std::fs::write(&backup_path, good).unwrap();
        std::fs::write(&config_path, malformed).unwrap();

        let err = write_wenlan_entry_with(&config_path, false, staged_write())
            .expect_err("a malformed config must still be reported as one");
        assert!(
            err.to_string().contains("Invalid JSON"),
            "the failure must still name what could not be parsed: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(&backup_path).unwrap(),
            good,
            "the last good backup was overwritten with the malformed file — the user's only \
             recoverable copy, destroyed by the failure that was about to be reported"
        );
        // And the broken file itself is left exactly as found: nothing was
        // written anywhere.
        assert_eq!(std::fs::read_to_string(&config_path).unwrap(), malformed);
    }

    /// Same defect, same fix, in the Codex TOML writer — Codex's review named
    /// both, and a fix to one of two identical orderings is half a fix.
    #[test]
    fn a_malformed_toml_config_does_not_overwrite_the_last_good_backup() {
        let tmp = tempfile::tempdir().unwrap();

        let config_path = tmp.path().join("config.toml");
        let backup_path = tmp.path().join("config.toml.bak");
        let good = "# hand-written\n[mcp_servers.wenlan]\ncommand = \
                    \"/opt/wenlan/bin/wenlan-mcp\"\nargs = []\n";
        let malformed = "[mcp_servers.wenlan\ncommand = ";
        std::fs::write(&backup_path, good).unwrap();
        std::fs::write(&config_path, malformed).unwrap();

        let err = write_wenlan_entry_toml_with(&config_path, staged_write())
            .expect_err("a malformed config must still be reported as one");
        assert!(
            err.to_string().contains("Invalid TOML"),
            "the failure must still name what could not be parsed: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(&backup_path).unwrap(),
            good,
            "the last good backup was overwritten with the malformed file"
        );
        assert_eq!(std::fs::read_to_string(&config_path).unwrap(), malformed);
    }

    /// The other half of the ordering: a config that DOES parse must still be
    /// backed up. Without this, "fix the ordering" could be satisfied by never
    /// writing a backup at all, and the two tests above would still pass.
    #[test]
    fn a_config_that_parses_is_still_backed_up_before_it_is_rewritten() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let original = "{\"mcpServers\":{\"other\":{\"command\":\"other-cmd\"}}}";
        std::fs::write(&config_path, original).unwrap();

        write_wenlan_entry_with(&config_path, false, staged_write()).unwrap();

        assert_eq!(
            std::fs::read_to_string(config_path.with_extension("json.bak")).unwrap(),
            original,
            "the pre-change bytes must be recoverable after a successful write"
        );

        let toml_path = tmp.path().join("config.toml");
        let original_toml = "model = \"gpt-5.5\"\n";
        std::fs::write(&toml_path, original_toml).unwrap();
        write_wenlan_entry_toml_with(&toml_path, staged_write()).unwrap();
        assert_eq!(
            std::fs::read_to_string(toml_path.with_extension("toml.bak")).unwrap(),
            original_toml
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_write_wenlan_entry_creates_new_file() {
        let _env = EnvGuard::capture(&[MCP_RESOLVER_HOME_ENV, "WENLAN_MCP_DEV_BIN"]);
        let tmp = tempfile::tempdir().unwrap();
        // Stand up the install the resolver is supposed to find, so the
        // written command is decided by the fixture and not by whatever this
        // host happens to have. The old "npx or a path" assertion passed on
        // both answers, which is why it never saw the missing `.exe` suffix.
        let installed = install_wenlan_mcp_into(tmp.path());
        std::env::remove_var("WENLAN_MCP_DEV_BIN");
        std::env::set_var(MCP_RESOLVER_HOME_ENV, tmp.path());

        let config_path = tmp.path().join("config.json");
        write_wenlan_entry(&config_path, false).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        // `.entry`, not the whole report: the report's `undetermined` half is
        // for the caller, and must never be written into the user's config
        // file (round 6, D3 — the report type exists precisely so the entry
        // shape written to disk stays exactly `{command, args}`).
        assert_eq!(
            parsed["mcpServers"]["wenlan"],
            serde_json::to_value(wenlan_mcp_entry().unwrap().entry).unwrap()
        );
        let cmd = parsed["mcpServers"]["wenlan"]["command"].as_str().unwrap();
        assert_eq!(
            Path::new(cmd),
            installed,
            "wrote {cmd} into the client config while {} was installed",
            installed.display()
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
        assert!(path.ends_with(Path::new(".gemini").join("settings.json")));
    }

    #[test]
    fn test_client_config_path_codex_cli() {
        let path = client_config_path("codex_cli").unwrap();
        assert!(path.ends_with(Path::new(".codex").join("config.toml")));
    }

    /// `exists` that answers true for exactly one path — so a test failure
    /// means the probed path is wrong, not merely that some boolean was false.
    fn only(hit: &str) -> impl Fn(&Path) -> Reading + '_ {
        move |p: &Path| Reading::of(p == Path::new(hit))
    }

    #[test]
    fn codex_cli_detected_finds_chatgpt_in_applications() {
        let home = PathBuf::from("/Users/someone");
        assert!(yes(codex_cli_detected(
            Reading::No,
            Some(&home),
            only("/Applications/ChatGPT.app")
        )));
    }

    #[test]
    fn codex_cli_detected_finds_chatgpt_in_user_applications() {
        let home = PathBuf::from("/Users/someone");
        assert!(yes(codex_cli_detected(
            Reading::No,
            Some(&home),
            only("/Users/someone/Applications/ChatGPT.app")
        )));
    }

    #[test]
    fn codex_cli_detected_via_config_when_chatgpt_absent() {
        let home = PathBuf::from("/Users/someone");
        assert!(yes(codex_cli_detected(Reading::Yes, Some(&home), |_| {
            Reading::No
        })));
    }

    #[test]
    fn codex_cli_not_detected_when_neither_present() {
        let home = PathBuf::from("/Users/someone");
        assert!(!yes(codex_cli_detected(Reading::No, Some(&home), |_| {
            Reading::No
        })));
        // A *different* Mac app must not be mistaken for ChatGPT desktop.
        assert!(!yes(codex_cli_detected(
            Reading::No,
            Some(&home),
            only("/Applications/Cursor.app")
        )));
    }

    #[test]
    fn codex_cli_detected_survives_missing_home() {
        assert!(yes(codex_cli_detected(
            Reading::No,
            None,
            only("/Applications/ChatGPT.app")
        )));
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

    // ── Round 5, defect 4: tri-state client detection ────────────────────

    /// A path that could not be built is not a client that is absent.
    ///
    /// `client_config_path` opened with `let home = dirs::home_dir()?;` — ONE
    /// `?` that turned every client's path into `None` when that lookup failed,
    /// `claude_desktop` included, whose path is built from the CONFIG dir and
    /// never touches `home`. Both halves are pinned here: the home-based
    /// clients report the failure, and the one that does not need home is
    /// unaffected by it.
    #[test]
    fn a_home_that_could_not_be_determined_is_not_a_missing_client() {
        let config_dir = PathBuf::from("/tmp/fixture-config");
        for client_type in ["cursor", "claude_code", "gemini_cli", "codex_cli"] {
            match client_config_path_for(client_type, None, Some(&config_dir)) {
                ClientConfigPath::Undetermined(why) => assert!(
                    why.contains("home directory"),
                    "the reason must name what could not be determined: {why}"
                ),
                other => panic!("{client_type} reported {other:?} for an unreadable home"),
            }
        }
        // The coupling the single `?` created: Claude Desktop's path does not
        // use `home` at all, and must not fail with it.
        assert_eq!(
            client_config_path_for("claude_desktop", None, Some(&config_dir)),
            ClientConfigPath::Known(config_dir.join("Claude").join("claude_desktop_config.json")),
        );
    }

    /// THE STARKEST CASE. With no home and no config dir, every client's path
    /// was `None`, `filter_map` dropped every row, and `detect_mcp_clients`
    /// returned an EMPTY VECTOR — which the Diagnostics card renders as "no MCP
    /// client detected" and the wizard as nothing to set up. A lookup that
    /// failed became a confident statement about the user's whole machine.
    #[test]
    fn a_search_that_could_not_look_is_never_an_empty_client_list() {
        let clients = detect_mcp_clients_from(None, None, |_| Reading::No);
        assert_eq!(
            clients.len(),
            5,
            "a client whose path could not be built must still be a row that says so"
        );
        for client in &clients {
            assert!(
                matches!(client.detected, Reading::Unreadable { .. }),
                "{} reported {:?} when nothing could be looked at",
                client.client_type,
                client.detected
            );
            assert!(
                matches!(client.already_configured, Reading::Unreadable { .. }),
                "{} claimed a configuration state it never read",
                client.client_type
            );
            assert!(
                client.config_path.is_none(),
                "there is no path to show when the directory could not be determined"
            );
        }
    }

    /// Codex's own scenario: a `~/.claude/settings.json` the OS will not hand
    /// over. It used to answer `false` through
    /// `read_to_string(..).map(..).unwrap_or(false)` — byte-identical to a
    /// settings file that is perfectly readable and has the plugin switched
    /// OFF. The user is then invited to configure a client that may already be
    /// configured, which is the double registration this app has a warning box
    /// for.
    ///
    /// The unreadable file is staged as a DIRECTORY at the settings path: it is
    /// the one read failure that reproduces on every platform (Windows answers
    /// `Access is denied`, Unix `EISDIR`), and a chmod-based fixture would be
    /// a no-op on Windows, which is where this app most needs the distinction.
    #[test]
    fn a_settings_file_that_could_not_be_read_is_not_a_plugin_that_is_off() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();

        // Control first, so the fixture is known to be able to say "no".
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::write(home.join(".claude").join("settings.json"), "{}").unwrap();
        assert_eq!(
            claude_code_plugin_enabled_on_disk(Some(home)),
            Reading::No,
            "a readable settings file with no plugin is a measured no"
        );

        std::fs::remove_file(home.join(".claude").join("settings.json")).unwrap();
        std::fs::create_dir_all(home.join(".claude").join("settings.json")).unwrap();
        match claude_code_plugin_enabled_on_disk(Some(home)) {
            Reading::Unreadable { error } => assert!(!error.is_empty()),
            other => panic!("a settings file that could not be read reported {other:?}"),
        }

        // And a home that could not be determined is not a plugin that is off
        // either — the `dirs::home_dir()` guard used to `return false`.
        match claude_code_plugin_enabled_on_disk(None) {
            Reading::Unreadable { error } => assert!(error.contains("home directory")),
            other => panic!("an undetermined home reported {other:?}"),
        }
    }

    /// The same collapse on the client's own config file, end to end through
    /// `detect_mcp_clients_from`: a `~/.gemini/settings.json` that cannot be
    /// read used to report exactly like one with no Wenlan entry.
    #[test]
    fn a_config_that_could_not_be_read_is_not_a_config_without_an_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        std::fs::create_dir_all(home.join(".gemini").join("settings.json")).unwrap();

        let clients = detect_mcp_clients_from(Some(home), Some(home), |_| Reading::No);
        let gemini = clients
            .iter()
            .find(|c| c.client_type == "gemini_cli")
            .expect("the gemini_cli row is always present");
        assert!(
            matches!(gemini.has_raw_entry, Reading::Unreadable { .. }),
            "an unreadable config reported {:?}",
            gemini.has_raw_entry
        );
        assert!(matches!(gemini.detected, Reading::Unreadable { .. }));

        // The control: a readable config with a real entry is a measured yes,
        // and a readable config without one is a measured no.
        let cursor_dir = home.join(".cursor");
        std::fs::create_dir_all(&cursor_dir).unwrap();
        std::fs::write(
            cursor_dir.join("mcp.json"),
            r#"{"mcpServers": {"wenlan": {"command": "wenlan-mcp"}}}"#,
        )
        .unwrap();
        let clients = detect_mcp_clients_from(Some(home), Some(home), |_| Reading::No);
        let cursor = clients
            .iter()
            .find(|c| c.client_type == "cursor")
            .expect("the cursor row is always present");
        assert_eq!(cursor.has_raw_entry, Reading::Yes);
        assert_eq!(cursor.already_configured, Reading::Yes);
        assert_eq!(cursor.has_plugin, Reading::No);
        // Cursor detects by app bundle, and the injected probe says no bundle.
        assert_eq!(cursor.detected, Reading::No);
    }

    /// `Reading::or` is the OR that `already_configured` is built from, and the
    /// ranking is the whole content: a failed read must not be able to turn a
    /// measured yes into a no, and two halves that are "no" and "unread" are
    /// unread, not "no".
    #[test]
    fn or_never_lets_a_failed_read_outrank_a_measurement() {
        let unreadable = || Reading::Unreadable {
            error: "Access is denied. (os error 5)".to_string(),
        };
        assert_eq!(Reading::Yes.or(unreadable()), Reading::Yes);
        assert_eq!(unreadable().or(Reading::Yes), Reading::Yes);
        assert!(matches!(
            Reading::No.or(unreadable()),
            Reading::Unreadable { .. }
        ));
        assert!(matches!(
            unreadable().or(Reading::No),
            Reading::Unreadable { .. }
        ));
        assert_eq!(Reading::No.or(Reading::No), Reading::No);
        assert_eq!(Reading::Yes.or(Reading::No), Reading::Yes);
    }

    /// `read_config` is the replacement for `Path::exists()` +
    /// `read_to_string(..).unwrap_or(false)`. Only `NotFound` is an absence.
    #[test]
    fn read_config_separates_absence_from_a_failed_look() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("there.json"), "{}").unwrap();

        assert_eq!(
            read_config(&tmp.path().join("there.json")).present(),
            Reading::Yes
        );
        assert_eq!(
            read_config(&tmp.path().join("nope.json")).present(),
            Reading::No
        );
        // A directory where a config should be: measured, and not an absence.
        assert!(matches!(
            read_config(tmp.path()).present(),
            Reading::Unreadable { .. }
        ));
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
        assert!(parsed(has_configured_entry_toml(
            "[mcp_servers.wenlan]\ncommand = \"npx\"\nargs = [\"-y\", \"wenlan-mcp\"]\n"
        )));
        assert!(parsed(has_configured_entry_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n"
        )));
        assert!(!parsed(has_configured_entry_toml(
            "[mcp_servers.other]\ncommand = \"x\"\n"
        )));
        assert!(!parsed(has_configured_entry_toml("model = \"gpt-5.5\"\n")));
        // CHANGED IN ROUND 6 WITH THE CODE: this line was
        // `assert!(!has_configured_entry_toml("not toml ["))`, which required
        // an unparseable `config.toml` to answer what a parsed one with no
        // entry answers. Codex CLI's row is the one that then reads
        // `already_configured = No`.
        assert!(unparseable(has_configured_entry_toml("not toml [")).contains("not valid TOML"));
    }

    #[test]
    fn test_write_wenlan_entry_toml_creates_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        write_wenlan_entry_toml(&config_path).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        assert!(parsed(has_configured_entry_toml(&contents)));
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
        assert!(parsed(has_configured_entry_toml(&contents)));
    }

    // Serial with a pinned resolver home, because both writes have to resolve
    // the *same* binary: `MCP_RESOLVER_HOME_ENV` is process-global, and a
    // concurrent serial test changing it between the two writes made this
    // compare an installed path against the `npx` fallback and fail on a
    // difference that is not the upsert's.
    #[test]
    #[serial_test::serial]
    fn test_write_wenlan_entry_toml_upsert_is_idempotent() {
        let _env = EnvGuard::capture(&[MCP_RESOLVER_HOME_ENV, "WENLAN_MCP_DEV_BIN"]);
        let home = tempfile::tempdir().unwrap();
        std::env::remove_var("WENLAN_MCP_DEV_BIN");
        std::env::set_var(MCP_RESOLVER_HOME_ENV, home.path());
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
        assert!(yes(client_config_has_raw_entry("cursor", &config_path)));

        remove_wenlan_entry(&config_path).unwrap();

        // The written file still parses and detection no longer sees an entry.
        assert!(!yes(client_config_has_raw_entry("cursor", &config_path)));
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
        assert!(!parsed(has_configured_entry_toml(&contents)));
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
        assert!(yes(client_config_has_raw_entry("codex_cli", &config_path)));

        remove_wenlan_entry_toml(&config_path).unwrap();

        assert!(!yes(client_config_has_raw_entry("codex_cli", &config_path)));
    }

    // ── has_both_raw_entries (raw+raw duplicate detection) ──────────────

    #[test]
    fn test_has_both_raw_entries_true_when_wenlan_and_origin_present() {
        // The real ~/.cursor/mcp.json shape on this machine.
        let json = r#"{"mcpServers": {
            "origin": {"command": "npx", "args": ["-y", "origin-mcp"]},
            "wenlan": {"command": "npx", "args": ["-y", "wenlan-mcp"]}
        }}"#;
        assert!(parsed(has_both_raw_entries(json)));
    }

    #[test]
    fn test_has_both_raw_entries_false_when_only_wenlan() {
        let json = r#"{"mcpServers": {"wenlan": {"command": "npx"}}}"#;
        assert!(!parsed(has_both_raw_entries(json)));
    }

    #[test]
    fn test_has_both_raw_entries_false_when_only_origin() {
        let json = r#"{"mcpServers": {"origin": {"command": "npx"}}}"#;
        assert!(!parsed(has_both_raw_entries(json)));
    }

    #[test]
    fn test_has_both_raw_entries_false_when_neither() {
        assert!(!parsed(has_both_raw_entries(
            r#"{"mcpServers": {"other": {}}}"#
        )));
        assert!(!parsed(has_both_raw_entries(r#"{"theme": "dark"}"#)));
        // CHANGED IN ROUND 6 WITH THE CODE, same reason as the TOML sibling.
        assert!(unparseable(has_both_raw_entries("not json")).contains("not valid JSON"));
    }

    #[test]
    fn test_has_both_raw_entries_toml() {
        assert!(parsed(has_both_raw_entries_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n[mcp_servers.wenlan]\ncommand = \"npx\"\n"
        )));
        assert!(!parsed(has_both_raw_entries_toml(
            "[mcp_servers.wenlan]\ncommand = \"npx\"\n"
        )));
        assert!(!parsed(has_both_raw_entries_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n"
        )));
        assert!(!parsed(has_both_raw_entries_toml("model = \"gpt-5.5\"\n")));
        // CHANGED IN ROUND 6 WITH THE CODE, same reason as the sibling above.
        assert!(unparseable(has_both_raw_entries_toml("not toml [")).contains("not valid TOML"));
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
        assert!(yes(client_config_has_both_raw_entries("cursor", &both)));

        let only_wenlan = tmp.path().join("only_wenlan.json");
        std::fs::write(
            &only_wenlan,
            r#"{"mcpServers": {"wenlan": {"command": "npx"}}}"#,
        )
        .unwrap();
        assert!(!yes(client_config_has_both_raw_entries(
            "cursor",
            &only_wenlan
        )));

        // A file that doesn't exist has no duplicate.
        assert!(!yes(client_config_has_both_raw_entries(
            "cursor",
            &tmp.path().join("missing.json")
        )));
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
        assert!(yes(client_config_has_both_raw_entries(
            "codex_cli",
            &config_path
        )));
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
        assert!(yes(client_config_has_both_raw_entries(
            "cursor",
            &config_path
        )));

        remove_legacy_origin_entry(&config_path).unwrap();

        // The duplicate is resolved, and a single wenlan entry remains.
        assert!(!yes(client_config_has_both_raw_entries(
            "cursor",
            &config_path
        )));
        assert!(yes(client_config_has_raw_entry("cursor", &config_path)));
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
        assert!(yes(client_config_has_raw_entry("codex_cli", &config_path)));
        assert!(!yes(client_config_has_both_raw_entries(
            "codex_cli",
            &config_path
        )));
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

    // ── Round 6 ─────────────────────────────────────────────────────────

    /// A decision that names a real binary, for the write tests below — so
    /// they exercise this file's branches rather than whatever the host has
    /// installed.
    fn write_decision() -> McpEntryDecision {
        McpEntryDecision::Write {
            entry: WenlanMcpEntry {
                command: "/opt/wenlan/bin/wenlan-mcp".to_string(),
                args: Vec::new(),
            },
            undetermined: Vec::new(),
        }
    }

    /// ADDITIONAL DEFECT 1 — THE CRASH, and it crashed AFTER the backup.
    ///
    /// `{"mcpServers": []}` is valid JSON a user can genuinely have (an editor
    /// that serialises an empty map as `[]`, a hand-edit). It parsed, it passed
    /// the presence check, and then `root["mcpServers"]["wenlan"] = ..` — which
    /// is `serde_json`'s `IndexMut`, and PANICS rather than erroring when it is
    /// handed an array. By then `fs::copy` had already taken the backup, so the
    /// user was left with a `.json.bak`, an unchanged config, a dead button and
    /// no message: a panic inside a Tauri command has no error the UI can show.
    ///
    /// Three things are asserted, and the third is the one that makes this a
    /// regression test rather than a smoke test: the call RETURNS (a panic
    /// fails the test), the message names the shape, and NO BACKUP EXISTS —
    /// nothing started.
    #[test]
    fn an_mcp_servers_array_is_a_schema_error_not_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let original = r#"{"mcpServers": []}"#;
        std::fs::write(&config_path, original).unwrap();

        let error = write_wenlan_entry_with(&config_path, false, write_decision())
            .expect_err("an array under `mcpServers` cannot take an entry");
        let error = error.to_string();
        assert!(
            error.contains("mcpServers") && error.contains("a list"),
            "the message has to name the shape the user has to fix: {error}"
        );
        assert_eq!(
            std::fs::read_to_string(&config_path).unwrap(),
            original,
            "the user's config was modified by a write that could not be completed"
        );
        assert!(
            !config_path.with_extension("json.bak").exists(),
            "a backup was left behind by a change that never happened"
        );
    }

    /// The same crash one line earlier: any valid non-object top level panics
    /// on the `root[\"mcpServers\"] = ..` insert instead.
    #[test]
    fn a_non_object_json_config_is_a_schema_error_not_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        for original in [r#"[]"#, r#""wenlan""#, r#"3"#] {
            let config_path = tmp.path().join(format!("config{}.json", original.len()));
            std::fs::write(&config_path, original).unwrap();
            let error = write_wenlan_entry_with(&config_path, false, write_decision())
                .expect_err("a non-object top level cannot take an entry")
                .to_string();
            assert!(
                error.contains("top level"),
                "the message has to say what is wrong with the file: {error}"
            );
            assert_eq!(std::fs::read_to_string(&config_path).unwrap(), original);
            assert!(!config_path.with_extension("json.bak").exists());
        }
    }

    /// A present-but-null `mcpServers` is NOT a schema error: it is a place an
    /// entry can go. (`serde_json`'s `IndexMut` turns a null into an object, so
    /// this one never crashed — it is here so the schema check cannot be
    /// tightened into refusing a file it used to handle.)
    #[test]
    fn a_null_mcp_servers_key_is_still_writable() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, r#"{"mcpServers": null, "theme": "dark"}"#).unwrap();

        write_wenlan_entry_with(&config_path, false, write_decision()).unwrap();

        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(
            written["mcpServers"]["wenlan"]["command"],
            "/opt/wenlan/bin/wenlan-mcp"
        );
        assert_eq!(written["theme"], "dark", "unrelated keys must survive");
    }

    /// ADDITIONAL DEFECT 1, THE TOML HALF. `doc[\"mcp_servers\"][key] = ..` is
    /// `toml_edit`'s `IndexMut`, i.e. `.expect(\"index not found\")` — a panic
    /// for any `mcp_servers` that is not table-like.
    #[test]
    fn a_scalar_mcp_servers_key_is_a_schema_error_not_a_panic_in_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        let original = "mcp_servers = 5\n";
        std::fs::write(&config_path, original).unwrap();

        let error = write_wenlan_entry_toml_with(&config_path, write_decision())
            .expect_err("a scalar `mcp_servers` cannot take a table")
            .to_string();
        assert!(
            error.contains("mcp_servers"),
            "the message has to name the key the user has to fix: {error}"
        );
        assert_eq!(std::fs::read_to_string(&config_path).unwrap(), original);
        assert!(!config_path.with_extension("toml.bak").exists());
    }

    /// D1, CONDITION 1 — the backup is written from the bytes that were
    /// PARSED, and only while those are still the bytes on disk.
    ///
    /// `back_up_parsed` is exercised directly because the race it closes cannot
    /// be staged through the writers without a hook into the middle of them:
    /// the interleaving is read+parse A, another process writes B, backup. This
    /// asserts the guard itself — handed contents that no longer match the
    /// file, it refuses and leaves NO backup, instead of copying B over the
    /// user's last good one.
    #[test]
    fn a_config_that_changed_under_the_writer_is_not_backed_up_from_the_new_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::write(&config_path, "{\"replaced\": true}").unwrap();

        let error = back_up_parsed(&config_path, "{\"parsed\": true}", "json.bak")
            .expect_err("the parse is stale, so the update it produced must not be applied")
            .to_string();
        assert!(
            error.contains("changed while Wenlan was updating it"),
            "{error}"
        );
        assert!(
            !config_path.with_extension("json.bak").exists(),
            "the last good backup was replaced with bytes this process never parsed"
        );
    }

    /// The ordinary path: unchanged file, backup holds exactly the parsed
    /// bytes.
    #[test]
    fn the_backup_holds_the_bytes_that_were_parsed() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let contents = "{\"mcpServers\": {}}";
        std::fs::write(&config_path, contents).unwrap();

        back_up_parsed(&config_path, contents, "json.bak").unwrap();

        assert_eq!(
            std::fs::read_to_string(config_path.with_extension("json.bak")).unwrap(),
            contents
        );
    }

    /// D1, CONDITION 2 / ADDITIONAL DEFECT 4 — `exists()` was a two-answer API
    /// standing in for a three-answer question, on all four verbs.
    ///
    /// The OS refusal the defect is really about (a file that permits writing
    /// but not `metadata`) cannot be staged portably. A DIRECTORY at the config
    /// path reaches the SAME branch: `read_config` answers `Unreadable`, not
    /// `Absent`, so the writer must refuse and say so rather than take the
    /// new-file branch — which is what truncated an unstattable file, from a
    /// `json!({})` skeleton, with no backup.
    #[test]
    fn a_config_path_that_cannot_be_read_is_never_treated_as_a_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::create_dir(&config_path).unwrap();
        assert!(
            matches!(read_config(&config_path), ConfigRead::Unreadable(_)),
            "fixture: this path must READ as unreadable, not as an absence"
        );

        let error = write_wenlan_entry_with(&config_path, false, write_decision())
            .expect_err("a config that could not be read must not be replaced")
            .to_string();
        assert!(error.contains("Nothing was written"), "{error}");
        assert!(
            config_path.is_dir(),
            "the path was replaced by a file built from an empty skeleton"
        );
    }

    /// The removal half of the same collapse: `if !config_path.exists()`
    /// reported "No config file found — nothing to remove" for a metadata
    /// denial, presenting a failed look as a measured absence and never
    /// attempting the read.
    #[test]
    fn a_config_that_cannot_be_read_is_not_reported_as_nothing_to_remove() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        std::fs::create_dir(&config_path).unwrap();

        let error = remove_wenlan_entry(&config_path)
            .expect_err("an unreadable config establishes nothing about what it holds")
            .to_string();
        assert!(
            !error.contains("No config file found"),
            "a failed look was reported as a measured absence: {error}"
        );
        assert!(error.contains("Could not read"), "{error}");

        // …while a genuinely absent file still says exactly that.
        let missing = tmp.path().join("nope.json");
        assert!(remove_wenlan_entry(&missing)
            .expect_err("nothing to remove")
            .to_string()
            .contains("No config file found"));
    }

    /// ADDITIONAL DEFECT 2 — a RELEASE ARTIFACT. `to_string_lossy` turned a
    /// path that cannot be spelled in UTF-8 into one that CAN, made of U+FFFD,
    /// and wrote it into the user's client config as the command to run. That
    /// names a different, nonexistent file, and the failure surfaces later as
    /// the client failing to launch a filename the user cannot find.
    #[test]
    fn a_binary_under_a_non_unicode_path_is_never_written_as_a_lossy_command() {
        #[cfg(windows)]
        let path: PathBuf = {
            use std::os::windows::ffi::OsStringExt;
            // A lone high surrogate: a valid Windows filename, not valid
            // Unicode, so `to_str()` is `None` and `to_string_lossy()` is a
            // DIFFERENT string.
            std::ffi::OsString::from_wide(&[0x0043, 0x003A, 0x005C, 0xD800, 0x002E, 0x0065]).into()
        };
        #[cfg(unix)]
        let path: PathBuf = {
            use std::os::unix::ffi::OsStringExt;
            std::ffi::OsString::from_vec(b"/opt/\xff/wenlan-mcp".to_vec()).into()
        };
        assert!(
            path.to_str().is_none(),
            "fixture: this path must not be representable as UTF-8"
        );

        let decision = wenlan_mcp_entry_for(
            McpBinaryResolution::Found {
                path: path.clone(),
                undetermined: Vec::new(),
            },
            "wenlan-mcp@^9.9.9",
        );

        match decision {
            McpEntryDecision::Write { entry, .. } => panic!(
                "wrote a command naming a file that does not exist: {:?} (the real path is \
                 {:?})",
                entry.command, path
            ),
            McpEntryDecision::PreserveExisting { unmeasured } => {
                assert_eq!(unmeasured.unreadable.len(), 1);
                assert_eq!(unmeasured.unreadable[0].0, path);
                assert!(
                    unmeasured.unreadable[0].1.contains("not valid Unicode"),
                    "{:?}",
                    unmeasured.unreadable[0].1
                );
                // …and the writers act on it the way they act on any other
                // unresolved search: no write, no backup, message names it.
                let message = unresolved_message(&unmeasured);
                assert!(message.contains("Nothing was written"), "{message}");
            }
        }
    }

    /// D2, END TO END — the shape the UI actually receives.
    ///
    /// A present Gemini `settings.json` holding `not json`. The file WAS read,
    /// so `detected` is a measured yes; whether it holds a Wenlan entry could
    /// NOT be measured, so `has_raw_entry` and `already_configured` must not be
    /// `no`. Before this round every one of them was `no`, and "no" is what
    /// puts an unqualified "Set up" button in front of the user.
    #[test]
    fn a_present_but_unparseable_client_config_is_detected_and_unmeasurable() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        std::fs::create_dir_all(home.join(".gemini")).unwrap();
        std::fs::write(home.join(".gemini").join("settings.json"), "not json").unwrap();

        let clients = detect_mcp_clients_from(Some(home), None, |_| Reading::No);
        let gemini = clients
            .iter()
            .find(|c| c.client_type == "gemini_cli")
            .expect("every client keeps a row");

        assert_eq!(
            gemini.detected,
            Reading::Yes,
            "the file is there and was read; only its CONTENTS were unmeasurable"
        );
        assert!(
            matches!(gemini.has_raw_entry, Reading::Unreadable { .. }),
            "an unparseable config answered `no entry`: {:?}",
            gemini.has_raw_entry
        );
        assert!(
            matches!(gemini.has_raw_duplicate, Reading::Unreadable { .. }),
            "{:?}",
            gemini.has_raw_duplicate
        );
        assert!(
            matches!(gemini.already_configured, Reading::Unreadable { .. }),
            "`already_configured` is what the wizard and the Settings list read, and it said \
             `not configured`: {:?}",
            gemini.already_configured
        );
    }

    /// The D6b half: a malformed `~/.claude/settings.json` used to reach
    /// `has_plugin = No` through the same collapse, and a MEASURED `no` for the
    /// plugin is the gate on Diagnostics' destructive raw-duplicate fix.
    #[test]
    fn an_unparseable_claude_settings_file_leaves_the_plugin_state_unmeasured() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::write(home.join(".claude").join("settings.json"), "{oops").unwrap();

        let clients = detect_mcp_clients_from(Some(home), None, |_| Reading::No);
        let claude_code = clients
            .iter()
            .find(|c| c.client_type == "claude_code")
            .expect("every client keeps a row");

        assert!(
            matches!(claude_code.has_plugin, Reading::Unreadable { .. }),
            "an unparseable settings.json answered `the plugin is off`: {:?}",
            claude_code.has_plugin
        );
    }

    /// …and the client with no plugin surface at all is still a MEASURED no,
    /// so nothing above turns Cursor's honest `no` into an unknown.
    #[test]
    fn a_client_with_no_plugin_surface_is_still_a_measured_no() {
        let tmp = tempfile::tempdir().unwrap();
        let clients = detect_mcp_clients_from(Some(tmp.path()), None, |_| Reading::No);
        for client_type in ["cursor", "gemini_cli"] {
            let client = clients
                .iter()
                .find(|c| c.client_type == client_type)
                .unwrap();
            assert_eq!(
                client.has_plugin,
                Reading::No,
                "{client_type} has no plugin surface to fail to read"
            );
        }
    }

    /// D3's BOUNDARY DEFECT, at the writers. A write that succeeded while one
    /// of the resolver's inputs went unread is not the same event as one where
    /// everything was measured; both used to return `()`.
    #[test]
    fn a_write_reports_the_inputs_that_could_not_be_determined() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        let undetermined = vec![UndeterminedInput {
            input: "WENLAN_MCP_DEV_BIN".to_string(),
            blocked: "WENLAN_MCP_DEV_BIN".to_string(),
            error: "environment variable was not valid Unicode".to_string(),
        }];

        let reported = write_wenlan_entry_with(
            &config_path,
            false,
            McpEntryDecision::Write {
                entry: WenlanMcpEntry {
                    command: "/opt/wenlan/bin/wenlan-mcp".to_string(),
                    args: Vec::new(),
                },
                undetermined: undetermined.clone(),
            },
        )
        .unwrap();

        assert_eq!(
            reported, undetermined,
            "the write succeeded off a search that skipped a candidate it never built, and said \
             exactly what an all-measured write says"
        );

        // The TOML writer is the same boundary.
        let toml_path = tmp.path().join("config.toml");
        let reported = write_wenlan_entry_toml_with(
            &toml_path,
            McpEntryDecision::Write {
                entry: WenlanMcpEntry {
                    command: "/opt/wenlan/bin/wenlan-mcp".to_string(),
                    args: Vec::new(),
                },
                undetermined: undetermined.clone(),
            },
        )
        .unwrap();
        assert_eq!(reported, undetermined);
    }
}
