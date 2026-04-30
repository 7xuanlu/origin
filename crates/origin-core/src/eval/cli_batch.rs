// SPDX-License-Identifier: AGPL-3.0-only
//! Shared `claude -p` subprocess primitive for eval CLI batched routes.
//!
//! Contains the truly shared bits across judge / answer-gen / enrichment CLI paths:
//! - `CliCostInfo` — cost telemetry parsed from `claude -p` envelope
//! - `strip_markdown_fence` — pre-parser for cases where `--json-schema` enforcement
//!   drops and the model returns markdown-fenced JSON in the envelope's `.result` field
//! - `run_cli_batch_subprocess` — the actual subprocess invocation. NO
//!   `--no-session-persistence` (it kills sessions immediately, breaking `--resume`).
//!   Optionally passes `--resume <session_id>` for cache reuse. `env_remove`s
//!   `ANTHROPIC_API_KEY` so Max-plan OAuth is used instead of an empty API balance.
//!
//! Each calling phase (judge / answer-gen / enrichment) keeps its own prompt builder
//! and parser on top of this primitive — only the subprocess call is shared.

use crate::error::OriginError;

/// Cost telemetry extracted from `claude -p` JSON envelope.
#[derive(Debug, Clone, Default)]
pub struct CliCostInfo {
    pub cost_usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}

/// Strip markdown code fence (```json ... ``` or ``` ... ```) wrapping JSON.
///
/// Used as a fallback parse strategy when `--json-schema` enforcement drops
/// and the model returns the JSON wrapped in a markdown fence inside the
/// envelope's `.result` field instead of the structured `.structured_output`.
pub fn strip_markdown_fence(text: &str) -> String {
    let t = text.trim();
    let after_open = if let Some(rest) = t
        .strip_prefix("```json\n")
        .or_else(|| t.strip_prefix("```\n"))
    {
        rest
    } else if let Some(rest) = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")) {
        rest
    } else {
        return t.to_string();
    };
    after_open
        .trim_end_matches("```")
        .trim_end_matches('\n')
        .trim()
        .to_string()
}

/// Run one `claude -p` subprocess call.
///
/// Returns `(raw_stdout, cost_info_if_parseable, session_id_if_returned)`.
/// Caller is responsible for parsing the stdout per their own JSON schema.
///
/// - Adds `--resume <sid>` if `session_id` is `Some`.
/// - NEVER passes `--no-session-persistence` (it would discard the session
///   immediately, defeating `--resume` on subsequent calls).
/// - `env_remove`s `ANTHROPIC_API_KEY` so OAuth (Max plan) is used even when
///   an empty `ANTHROPIC_API_KEY` is set in the shell.
/// - Disables tools via `--allowedTools ""`.
pub async fn run_cli_batch_subprocess(
    prompt: &str,
    model: &str,
    json_schema: &str,
    session_id: Option<&str>,
) -> Result<(String, Option<CliCostInfo>, Option<String>), OriginError> {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let mut args: Vec<String> = vec![
        "-p".into(),
        "--model".into(),
        model.into(),
        "--output-format".into(),
        "json".into(),
        "--json-schema".into(),
        json_schema.into(),
        "--allowedTools".into(),
        "".into(),
    ];
    if let Some(sid) = session_id {
        args.push("--resume".into());
        args.push(sid.into());
    }

    let mut child = Command::new("claude")
        .env_remove("ANTHROPIC_API_KEY")
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| OriginError::Generic(format!("claude -p failed to start: {}", e)))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .await
            .map_err(|e| OriginError::Generic(format!("write to claude stdin failed: {}", e)))?;
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| OriginError::Generic(format!("claude -p wait failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(OriginError::Generic(format!(
            "claude -p exited with error: {}",
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    let env: Option<serde_json::Value> = serde_json::from_str(stdout.trim()).ok();
    let cost = env.as_ref().map(|e| {
        let usage = e.get("usage");
        CliCostInfo {
            cost_usd: e
                .get("total_cost_usd")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            input_tokens: usage
                .and_then(|u| u.get("input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: usage
                .and_then(|u| u.get("output_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_creation_tokens: usage
                .and_then(|u| u.get("cache_creation_input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_read_tokens: usage
                .and_then(|u| u.get("cache_read_input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        }
    });

    let new_sid = env
        .as_ref()
        .and_then(|e| e.get("session_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok((stdout, cost, new_sid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_json_fence_with_newline() {
        let input = "```json\n{\"score\": 1}\n```";
        assert_eq!(strip_markdown_fence(input), "{\"score\": 1}");
    }

    #[test]
    fn strips_plain_fence() {
        let input = "```\n{\"a\":1}\n```";
        assert_eq!(strip_markdown_fence(input), "{\"a\":1}");
    }

    #[test]
    fn returns_unchanged_when_no_fence() {
        let input = "{\"score\": 0}";
        assert_eq!(strip_markdown_fence(input), "{\"score\": 0}");
    }

    #[test]
    fn handles_inline_json_fence() {
        let input = "```json{\"x\":1}```";
        assert_eq!(strip_markdown_fence(input), "{\"x\":1}");
    }

    #[test]
    fn cost_info_default() {
        let c = CliCostInfo::default();
        assert_eq!(c.cost_usd, 0.0);
        assert_eq!(c.input_tokens, 0);
    }
}
