//! Installs Wenlan's Claude Code and Codex plugins by shelling out to each
//! CLI's supported, non-interactive plugin subcommands, instead of
//! hand-writing either CLI's private plugin-state file
//! (`~/.claude/plugins/installed_plugins.json` is versioned, undocumented,
//! and carries a git cache with commit SHAs; the `.bak-*` files next to it
//! show the format churns). Both plugins declare their own `mcpServers`, so
//! callers must never ALSO write a raw MCP entry for `claude_code` /
//! `codex_cli` via `mcp_config.rs` — that would duplicate it.
//!
//! Binary resolution tries a LOGIN SHELL first, static install locations
//! only as a fallback: both CLIs are commonly installed via `npm i -g` under
//! a version manager (nvm) whose bin dir no static probe can guess — on the
//! machine this was verified against, `codex` lives at
//! `~/.nvm/versions/node/v24.11.1/bin/codex`. A Tauri app launched from
//! Finder inherits a minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`), so a
//! plain `which` at spawn time would not see it either; a login shell
//! (`zsh -lic 'command -v <bin>'`) sources the same rc files the user's own
//! terminal does.
//!
//! Marketplace *selectors* are resolved at runtime, never hardcoded: the
//! name a CLI derives from a GitHub source can differ from the repo slug
//! (Codex derived `wenlan-local` for `7xuanlu/wenlan` as of codex-cli
//! 0.144.0, pending a rename to `7xuanlu-wenlan` in 7xuanlu/wenlan#348,
//! not yet merged as of 2026-07-12) — so after `marketplace add`, each
//! installer reads the real name back out of `plugin marketplace list
//! --json` and only falls back to a hardcoded default if that lookup fails.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The GitHub source both CLIs' `plugin marketplace add` clone from.
const WENLAN_REPO: &str = "7xuanlu/wenlan";
/// Last-resort marketplace names if runtime resolution fails for any reason
/// (unexpected CLI output, `marketplace list` unsupported, etc) — per
/// client, not shared: Claude's derivation is stable (owner/repo slug), but
/// Codex's is `wenlan-local` *today*, correct only until 7xuanlu/wenlan#348
/// merges. A shared "post-rename" fallback would be wrong for every Codex
/// user until that PR lands; this way the fallback matches current reality,
/// and is only ever reached if the primary `marketplace list --json` lookup
/// itself fails — which stays correct across the rename either way.
const FALLBACK_MARKETPLACE_CLAUDE: &str = "7xuanlu-wenlan";
const FALLBACK_MARKETPLACE_CODEX: &str = "wenlan-local";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PluginInstallError {
    #[error("{0} CLI not found")]
    CliNotFound(&'static str),
    #[error("{0}")]
    StepFailed(String),
    #[error("unsupported client type: {0}")]
    UnknownClient(String),
}

// ── Binary resolution ───────────────────────────────────────────────────

/// Static, fixed-location fallback candidates for `binary_name`, in probe
/// order. Only consulted when the login shell can't resolve the binary.
/// `.claude/local/<bin>` is Claude Code's own self-managed install dir and
/// only applies to `claude`.
pub fn static_binary_candidates(home: &Path, binary_name: &str) -> Vec<PathBuf> {
    let mut candidates = vec![home.join(".local/bin").join(binary_name)];
    if binary_name == "claude" {
        candidates.push(home.join(".claude/local").join(binary_name));
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin").join(binary_name));
    candidates.push(PathBuf::from("/usr/local/bin").join(binary_name));
    candidates
}

fn first_existing(candidates: &[PathBuf], exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    candidates.iter().find(|p| exists(p)).cloned()
}

/// Full resolution with every filesystem/process dependency injected, so the
/// probe order (login shell first, static candidates as fallback) is
/// unit-testable without spawning a shell or touching the real filesystem.
fn resolve_binary_with(
    binary_name: &str,
    home: Option<PathBuf>,
    exists: impl Fn(&Path) -> bool,
    login_shell: impl FnOnce(&str) -> Option<PathBuf>,
) -> Option<PathBuf> {
    login_shell(binary_name).or_else(|| {
        let candidates = home
            .map(|h| static_binary_candidates(&h, binary_name))
            .unwrap_or_default();
        first_existing(&candidates, exists)
    })
}

fn login_shell_binary_path(binary_name: &str) -> Option<PathBuf> {
    let output = Command::new("zsh")
        .arg("-lic")
        .arg(format!("command -v {binary_name}"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

/// Resolve a CLI binary by name ("claude" / "codex"): login shell first (it
/// sees the user's real PATH, including version-managed installs), static
/// locations as fallback. Never panics — a missing CLI is expected, not
/// exceptional.
pub fn resolve_binary(binary_name: &str) -> Option<PathBuf> {
    resolve_binary_with(
        binary_name,
        dirs::home_dir(),
        |p| p.exists(),
        login_shell_binary_path,
    )
}

// ── Marketplace selector resolution ─────────────────────────────────────

/// Reads the registered marketplace name for `7xuanlu/wenlan` out of
/// `claude plugin marketplace list --json` output, e.g.
/// `[{"name":"7xuanlu-wenlan","source":"github","repo":"7xuanlu/wenlan",...}]`.
fn find_marketplace_name_claude(json: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    value.as_array()?.iter().find_map(|entry| {
        (entry.get("repo")?.as_str()? == WENLAN_REPO)
            .then(|| entry.get("name")?.as_str().map(String::from))
            .flatten()
    })
}

/// Reads the registered marketplace name for `7xuanlu/wenlan` out of
/// `codex plugin marketplace list --json` output, e.g.
/// `{"marketplaces":[{"name":"wenlan-local","marketplaceSource":{"sourceType":"git","source":"https://github.com/7xuanlu/wenlan.git"}}]}`.
/// Matches by substring on the git URL rather than exact equality, since the
/// URL carries a protocol and `.git` suffix the bare repo slug doesn't.
fn find_marketplace_name_codex(json: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    value
        .get("marketplaces")?
        .as_array()?
        .iter()
        .find_map(|entry| {
            let source = entry.get("marketplaceSource")?.get("source")?.as_str()?;
            source
                .contains(WENLAN_REPO)
                .then(|| entry.get("name")?.as_str().map(String::from))
                .flatten()
        })
}

/// Builds the `wenlan@<marketplace>` selector from a `marketplace list
/// --json` result, falling back to `fallback` if the process failed or the
/// output didn't parse. Pure — the process call is injected via `(stdout,
/// succeeded)` so probe/parse logic is unit-testable without spawning a CLI.
fn build_selector_with(
    stdout: &str,
    succeeded: bool,
    parse: impl Fn(&str) -> Option<String>,
    fallback: &str,
) -> String {
    let name = succeeded
        .then(|| parse(stdout))
        .flatten()
        .unwrap_or_else(|| fallback.to_string());
    format!("wenlan@{name}")
}

fn resolve_selector(bin: &Path, parse: impl Fn(&str) -> Option<String>, fallback: &str) -> String {
    match Command::new(bin)
        .args(["plugin", "marketplace", "list", "--json"])
        .output()
    {
        Ok(output) => build_selector_with(
            &String::from_utf8_lossy(&output.stdout),
            output.status.success(),
            parse,
            fallback,
        ),
        Err(_) => build_selector_with("", false, parse, fallback),
    }
}

// ── Output classification ───────────────────────────────────────────────

/// Strips ANSI SGR escape sequences (`ESC [ ... m`) — both CLIs colorize
/// failure output even without a TTY, but not success output.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Codex prints a benign PATH-alias warning on stderr on both success and
/// failure paths; strip it so a failure message doesn't lead with noise.
fn strip_warning_lines(s: &str) -> String {
    s.lines()
        .filter(|line| !line.trim_start().starts_with("WARNING:"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Classifies a completed step: success on exit 0, success if the combined
/// output mentions "already" (both CLIs exit 0 on repeat too, but this
/// covers a hypothetical future CLI version that doesn't), otherwise a
/// `StepFailed` carrying the CLI's own message with ANSI codes and the
/// benign Codex warning line stripped out.
fn classify_step(
    step: &'static str,
    exit_code: Option<i32>,
    stdout: &str,
    stderr: &str,
) -> Result<(), PluginInstallError> {
    if exit_code == Some(0) {
        return Ok(());
    }
    let combined = format!("{stdout} {stderr}").to_lowercase();
    if combined.contains("already") {
        return Ok(());
    }
    let raw = if !stderr.trim().is_empty() {
        stderr
    } else {
        stdout
    };
    let filtered = strip_warning_lines(raw);
    let message_source = if filtered.trim().is_empty() {
        raw
    } else {
        &filtered
    };
    Err(PluginInstallError::StepFailed(format!(
        "{step}: {}",
        strip_ansi(message_source).trim()
    )))
}

fn run_step(bin: &Path, step: &'static str, args: &[&str]) -> Result<(), PluginInstallError> {
    let output = Command::new(bin)
        .args(args)
        .output()
        .map_err(|e| PluginInstallError::StepFailed(format!("{step}: {e}")))?;
    classify_step(
        step,
        output.status.code(),
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
    )
}

// ── Public install entry points ─────────────────────────────────────────

/// Installs the Wenlan Claude Code plugin: `claude plugin marketplace add
/// 7xuanlu/wenlan`, then `claude plugin install wenlan@<resolved>`.
/// Idempotent — succeeds if the marketplace or plugin is already present.
pub fn install_claude_code_plugin() -> Result<(), PluginInstallError> {
    let bin = resolve_binary("claude").ok_or(PluginInstallError::CliNotFound("Claude Code"))?;
    run_step(
        &bin,
        "marketplace add",
        &["plugin", "marketplace", "add", WENLAN_REPO],
    )?;
    let selector = resolve_selector(
        &bin,
        find_marketplace_name_claude,
        FALLBACK_MARKETPLACE_CLAUDE,
    );
    run_step(&bin, "plugin install", &["plugin", "install", &selector])?;
    Ok(())
}

/// Installs the Wenlan Codex plugin: `codex plugin marketplace add
/// 7xuanlu/wenlan`, then `codex plugin add wenlan@<resolved>` (note: `add`,
/// not `install` — Codex's subcommand name differs from Claude Code's).
/// Idempotent — succeeds if the marketplace or plugin is already present.
pub fn install_codex_plugin() -> Result<(), PluginInstallError> {
    let bin = resolve_binary("codex").ok_or(PluginInstallError::CliNotFound("Codex"))?;
    run_step(
        &bin,
        "marketplace add",
        &["plugin", "marketplace", "add", WENLAN_REPO],
    )?;
    let selector = resolve_selector(
        &bin,
        find_marketplace_name_codex,
        FALLBACK_MARKETPLACE_CODEX,
    );
    run_step(&bin, "plugin add", &["plugin", "add", &selector])?;
    Ok(())
}

/// Installs the Wenlan plugin for `client_type` (`"claude_code"` /
/// `"codex_cli"`, matching the wizard's client-type strings elsewhere in the
/// codebase — see `mcp_config::detect_mcp_clients`). The single dispatch
/// point `search::install_client_plugin` calls into.
pub fn install_client_plugin(client_type: &str) -> Result<(), PluginInstallError> {
    match client_type {
        "claude_code" => install_claude_code_plugin(),
        "codex_cli" => install_codex_plugin(),
        other => Err(PluginInstallError::UnknownClient(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── static_binary_candidates ────────────────────────────────────────

    #[test]
    fn static_candidates_claude_includes_self_managed_dir() {
        let home = PathBuf::from("/home/u");
        let c = static_binary_candidates(&home, "claude");
        assert_eq!(
            c,
            vec![
                PathBuf::from("/home/u/.local/bin/claude"),
                PathBuf::from("/home/u/.claude/local/claude"),
                PathBuf::from("/opt/homebrew/bin/claude"),
                PathBuf::from("/usr/local/bin/claude"),
            ]
        );
    }

    #[test]
    fn static_candidates_codex_excludes_claude_local_dir() {
        let home = PathBuf::from("/home/u");
        let c = static_binary_candidates(&home, "codex");
        assert_eq!(
            c,
            vec![
                PathBuf::from("/home/u/.local/bin/codex"),
                PathBuf::from("/opt/homebrew/bin/codex"),
                PathBuf::from("/usr/local/bin/codex"),
            ]
        );
    }

    // ── resolve_binary_with: order ──────────────────────────────────────

    #[test]
    fn resolve_binary_prefers_login_shell_over_static_match() {
        // Both the login shell AND a static candidate would resolve to a
        // path that "exists" — but they're different paths. The login
        // shell's answer must win.
        let home = Some(PathBuf::from("/home/u"));
        let exists = |p: &Path| p == Path::new("/home/u/.local/bin/codex");
        let login_shell = |_: &str| Some(PathBuf::from("/nvm/bin/codex"));
        let resolved = resolve_binary_with("codex", home, exists, login_shell);
        assert_eq!(resolved, Some(PathBuf::from("/nvm/bin/codex")));
    }

    #[test]
    fn resolve_binary_falls_back_to_static_when_login_shell_fails() {
        let home = Some(PathBuf::from("/home/u"));
        let exists = |p: &Path| p == Path::new("/home/u/.local/bin/codex");
        let login_shell = |_: &str| None;
        let resolved = resolve_binary_with("codex", home, exists, login_shell);
        assert_eq!(resolved, Some(PathBuf::from("/home/u/.local/bin/codex")));
    }

    #[test]
    fn resolve_binary_none_when_nothing_matches() {
        let home = Some(PathBuf::from("/home/u"));
        let exists = |_: &Path| false;
        let login_shell = |_: &str| None;
        assert_eq!(
            resolve_binary_with("codex", home, exists, login_shell),
            None
        );
    }

    #[test]
    fn resolve_binary_none_home_no_login_shell_is_none() {
        let exists = |_: &Path| true; // would match if home existed
        let login_shell = |_: &str| None;
        assert_eq!(
            resolve_binary_with("codex", None, exists, login_shell),
            None
        );
    }

    // ── find_marketplace_name_claude ────────────────────────────────────

    const CLAUDE_MARKETPLACE_JSON: &str = r#"[
  {
    "name": "7xuanlu-wenlan",
    "source": "github",
    "repo": "7xuanlu/wenlan",
    "installLocation": "/home/u/.claude/plugins/marketplaces/7xuanlu-wenlan"
  }
]"#;

    #[test]
    fn find_marketplace_name_claude_matches_by_repo() {
        assert_eq!(
            find_marketplace_name_claude(CLAUDE_MARKETPLACE_JSON),
            Some("7xuanlu-wenlan".to_string())
        );
    }

    #[test]
    fn find_marketplace_name_claude_ignores_other_repos() {
        let json = r#"[{"name":"other","source":"github","repo":"someone/else"}]"#;
        assert_eq!(find_marketplace_name_claude(json), None);
    }

    #[test]
    fn find_marketplace_name_claude_none_on_garbage() {
        assert_eq!(find_marketplace_name_claude("not json"), None);
    }

    // ── find_marketplace_name_codex ─────────────────────────────────────

    const CODEX_MARKETPLACE_JSON: &str = r#"{
  "marketplaces": [
    {
      "name": "wenlan-local",
      "root": "/home/u/.codex/.tmp/marketplaces/wenlan-local",
      "marketplaceSource": {
        "sourceType": "git",
        "source": "https://github.com/7xuanlu/wenlan.git"
      }
    }
  ]
}"#;

    #[test]
    fn find_marketplace_name_codex_matches_by_source_url_substring() {
        assert_eq!(
            find_marketplace_name_codex(CODEX_MARKETPLACE_JSON),
            Some("wenlan-local".to_string())
        );
    }

    #[test]
    fn find_marketplace_name_codex_matches_renamed_marketplace() {
        // 7xuanlu/wenlan#348 renames the marketplace; the source URL (what
        // we match on) is unchanged, so resolution keeps working post-merge
        // without a code change.
        let json = CODEX_MARKETPLACE_JSON.replace("wenlan-local", "7xuanlu-wenlan");
        assert_eq!(
            find_marketplace_name_codex(&json),
            Some("7xuanlu-wenlan".to_string())
        );
    }

    #[test]
    fn find_marketplace_name_codex_ignores_other_repos() {
        let json = r#"{"marketplaces":[{"name":"other","marketplaceSource":{"sourceType":"git","source":"https://github.com/someone/else.git"}}]}"#;
        assert_eq!(find_marketplace_name_codex(json), None);
    }

    #[test]
    fn find_marketplace_name_codex_none_on_garbage() {
        assert_eq!(find_marketplace_name_codex("not json"), None);
    }

    // ── build_selector_with ─────────────────────────────────────────────

    #[test]
    fn build_selector_uses_resolved_name_on_success() {
        let selector = build_selector_with(
            CODEX_MARKETPLACE_JSON,
            true,
            find_marketplace_name_codex,
            FALLBACK_MARKETPLACE_CODEX,
        );
        assert_eq!(selector, "wenlan@wenlan-local");
    }

    #[test]
    fn build_selector_falls_back_when_process_failed() {
        let selector = build_selector_with(
            CODEX_MARKETPLACE_JSON,
            false,
            find_marketplace_name_codex,
            FALLBACK_MARKETPLACE_CODEX,
        );
        assert_eq!(selector, format!("wenlan@{FALLBACK_MARKETPLACE_CODEX}"));
    }

    #[test]
    fn build_selector_falls_back_when_parse_fails() {
        let selector = build_selector_with(
            "not json",
            true,
            find_marketplace_name_codex,
            FALLBACK_MARKETPLACE_CODEX,
        );
        assert_eq!(selector, format!("wenlan@{FALLBACK_MARKETPLACE_CODEX}"));
    }

    #[test]
    fn build_selector_fallback_differs_per_client() {
        // The whole point of a per-client fallback: Codex's real name today
        // is `wenlan-local`, not the post-rename `7xuanlu-wenlan` Claude
        // uses — using the wrong one would install to a marketplace that
        // doesn't exist yet.
        let claude_selector = build_selector_with(
            "not json",
            false,
            find_marketplace_name_claude,
            FALLBACK_MARKETPLACE_CLAUDE,
        );
        let codex_selector = build_selector_with(
            "not json",
            false,
            find_marketplace_name_codex,
            FALLBACK_MARKETPLACE_CODEX,
        );
        assert_eq!(claude_selector, "wenlan@7xuanlu-wenlan");
        assert_eq!(codex_selector, "wenlan@wenlan-local");
    }

    // ── strip_ansi / strip_warning_lines ────────────────────────────────

    #[test]
    fn strip_ansi_removes_sgr_codes() {
        assert_eq!(
            strip_ansi("\u{1b}[31mError:\u{1b}[0m broke"),
            "Error: broke"
        );
    }

    #[test]
    fn strip_ansi_passes_through_plain_text() {
        assert_eq!(strip_ansi("plain text"), "plain text");
    }

    #[test]
    fn strip_warning_lines_removes_only_warning_prefixed_lines() {
        let input = "WARNING: proceeding, even though...\nError: plugin not found";
        assert_eq!(strip_warning_lines(input), "Error: plugin not found");
    }

    #[test]
    fn strip_warning_lines_keeps_non_warning_content_untouched() {
        assert_eq!(
            strip_warning_lines("Error: plugin not found"),
            "Error: plugin not found"
        );
    }

    // ── classify_step ────────────────────────────────────────────────────

    #[test]
    fn classify_step_success_on_exit_zero() {
        assert_eq!(classify_step("marketplace add", Some(0), "", ""), Ok(()));
    }

    #[test]
    fn classify_step_success_when_output_mentions_already() {
        assert_eq!(
            classify_step("plugin install", Some(1), "already installed", ""),
            Ok(())
        );
    }

    #[test]
    fn classify_step_real_claude_failure_message_is_ansi_stripped() {
        // Captured verbatim from `claude plugin install wenlan@nonexistent`
        // against a throwaway HOME.
        let stderr = "\u{1b}[31mError: Marketplace \"nonexistent\" not found\u{1b}[0m\n";
        let result = classify_step("plugin install", Some(1), "", stderr);
        assert_eq!(
            result,
            Err(PluginInstallError::StepFailed(
                "plugin install: Error: Marketplace \"nonexistent\" not found".to_string()
            ))
        );
    }

    #[test]
    fn classify_step_real_codex_failure_strips_warning_line() {
        // Captured verbatim from `codex plugin add wenlan@nonexistent`
        // against a throwaway HOME.
        let stderr = "WARNING: proceeding, even though we could not create PATH aliases: Refusing to create helper binaries under temporary dir\nError: plugin `wenlan` was not found in marketplace `nonexistent`";
        let result = classify_step("plugin add", Some(1), "", stderr);
        assert_eq!(
            result,
            Err(PluginInstallError::StepFailed(
                "plugin add: Error: plugin `wenlan` was not found in marketplace `nonexistent`"
                    .to_string()
            ))
        );
    }

    #[test]
    fn classify_step_falls_back_to_stdout_when_stderr_empty() {
        let result = classify_step("plugin add", Some(1), "install failed", "");
        assert_eq!(
            result,
            Err(PluginInstallError::StepFailed(
                "plugin add: install failed".to_string()
            ))
        );
    }

    #[test]
    fn classify_step_keeps_raw_message_if_warning_strip_empties_it() {
        // Defensive: if stderr were somehow ONLY a WARNING line, don't
        // collapse the error message to nothing.
        let stderr = "WARNING: only a warning, no error text";
        let result = classify_step("plugin add", Some(1), "", stderr);
        assert_eq!(
            result,
            Err(PluginInstallError::StepFailed(
                "plugin add: WARNING: only a warning, no error text".to_string()
            ))
        );
    }

    // ── install_client_plugin dispatcher ────────────────────────────────

    #[test]
    fn install_client_plugin_rejects_unsupported_client_type() {
        // Only the two error-free branches spawn a process (unreachable in
        // a unit test without a real CLI); the unsupported branch is pure
        // and directly testable.
        assert_eq!(
            install_client_plugin("cursor"),
            Err(PluginInstallError::UnknownClient("cursor".to_string()))
        );
    }
}
