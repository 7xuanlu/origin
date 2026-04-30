// SPDX-License-Identifier: AGPL-3.0-only
//! LLM-as-judge infrastructure: types, functions, prompts, Batch API judge.

use crate::error::OriginError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ===== Judge Prompt (shared between CLI and Batch API paths) =====

/// Task-specific judge prompt dispatcher. Both `judge_single_tuple_model` (CLI)
/// and `judge_with_batch_api` (Batch API) call this, so judge behavior is
/// identical regardless of path.
///
/// Dispatches to benchmark-sourced prompts based on `category`:
/// - `temporal-reasoning`: off-by-one tolerance for day/week/month counts
/// - `knowledge-update`: accepts old+new answers if updated answer is correct
/// - `single-session-preference`: rubric-based (not exact-match)
/// - Everything else (LoCoMo categories + LME SSU/SSA/MS): standard benchmark prompt
fn task_judge_prompt(category: &str, question: &str, ground_truth: &str, answer: &str) -> String {
    lme_anscheck_prompt(category, question, ground_truth, answer)
}

// ===== LLM-as-Judge Types =====

/// A single E2E answer to be judged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentTuple {
    pub question: String,
    pub ground_truth: String,
    pub approach: String,
    pub answer: String,
    pub context_tokens: usize,
    /// Task category for task-specific judge prompts (e.g. "temporal-reasoning",
    /// "single-hop"). Defaults to empty for backward compat with existing JSON.
    #[serde(default)]
    pub category: String,
}

/// Result from the LLM judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentResult {
    pub question: String,
    pub approach: String,
    /// 0 or 1.
    pub score: u8,
    pub reason: String,
    pub context_tokens: usize,
}

// Cost telemetry has been moved to `cli_batch::CliCostInfo` and is shared across
// all CLI batched eval routes. Keep `JudgeCostInfo` as an alias for backward compat
// with any external caller; new code should use `CliCostInfo` directly.
pub use crate::eval::cli_batch::CliCostInfo as JudgeCostInfo;

/// Per-approach aggregated result in a judged E2E report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgedApproachResult {
    pub approach: String,
    /// Fraction of questions scoring 1.
    pub accuracy: f64,
    pub total: usize,
    pub correct: usize,
    pub mean_context_tokens: f64,
}

/// Full judged E2E report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgedE2EReport {
    pub judge_model: String,
    pub total_judged: usize,
    pub results_by_approach: Vec<JudgedApproachResult>,
}

// ===== LLM-as-Judge Functions =====

/// Save E2E answer tuples to JSON for offline judging.
pub fn save_judgment_tuples(tuples: &[JudgmentTuple], path: &Path) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(tuples).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Load previously saved judgment tuples from JSON.
pub fn load_judgment_tuples(path: &Path) -> Result<Vec<JudgmentTuple>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(std::io::Error::other)
}

/// Judge answer tuples using Claude via the `claude -p` CLI.
///
/// Requires Claude Code CLI installed (`claude --version` must succeed).
/// Uses Haiku via the user's existing Max subscription — no API key needed.
/// Runs up to `concurrency` judgments in parallel.
pub async fn judge_with_claude(
    tuples: &[JudgmentTuple],
    concurrency: usize,
) -> Result<Vec<JudgmentResult>, OriginError> {
    judge_with_claude_model(tuples, concurrency, "haiku").await
}

/// Judge tuples with a specific Claude model (e.g. "haiku", "sonnet").
///
/// In-memory only. For long runs prefer `judge_with_claude_model_persistent`,
/// which caches each judgment to disk so partial failures aren't lost.
pub async fn judge_with_claude_model(
    tuples: &[JudgmentTuple],
    concurrency: usize,
    model: &str,
) -> Result<Vec<JudgmentResult>, OriginError> {
    use tokio::sync::Semaphore;

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::new();

    for tuple in tuples {
        let sem = semaphore.clone();
        let tuple = tuple.clone();
        let model = model.to_string();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            judge_single_tuple_model(&tuple, &model).await
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    let mut fail_count = 0usize;
    for handle in handles {
        match handle.await {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => {
                fail_count += 1;
                eprintln!("[judge] FAILED: {}", e);
            }
            Err(e) => {
                fail_count += 1;
                eprintln!("[judge] PANICKED: {}", e);
            }
        }
    }
    eprintln!(
        "[judge] {} succeeded, {} failed (no persistence — failures lost)",
        results.len(),
        fail_count
    );

    Ok(results)
}

/// Judge tuples with on-disk persistence + retry + resume.
///
/// Behavior:
/// - Loads any existing judgments from `cache_path` (JSONL, one `JudgmentResult` per line)
/// - Skips tuples whose (question, approach) pair is already judged
/// - Retries each unsuccessful tuple up to `max_retries` times with exponential backoff
/// - Appends each successful judgment to `cache_path` immediately (under file mutex)
/// - Prints visible per-failure messages + final summary to stderr
///
/// Safe to ctrl-C and re-run: re-running picks up where it left off.
pub async fn judge_with_claude_model_persistent(
    tuples: &[JudgmentTuple],
    concurrency: usize,
    model: &str,
    cache_path: &Path,
    max_retries: u32,
) -> Result<Vec<JudgmentResult>, OriginError> {
    use std::collections::HashSet;
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::{Mutex as AsyncMutex, Semaphore};

    let cached: Vec<JudgmentResult> = if cache_path.exists() {
        let f = std::fs::File::open(cache_path).map_err(OriginError::from)?;
        BufReader::new(f)
            .lines()
            .map_while(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<JudgmentResult>(&l).ok())
            .collect()
    } else {
        Vec::new()
    };

    let cached_keys: HashSet<(String, String)> = cached
        .iter()
        .map(|r| (r.question.clone(), r.approach.clone()))
        .collect();

    let todo: Vec<JudgmentTuple> = tuples
        .iter()
        .filter(|t| !cached_keys.contains(&(t.question.clone(), t.approach.clone())))
        .cloned()
        .collect();

    eprintln!(
        "[judge] cache: {} existing, {} to judge ({} total)",
        cached.len(),
        todo.len(),
        tuples.len()
    );

    if todo.is_empty() {
        return Ok(cached);
    }

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(OriginError::from)?;
    }
    let cache_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(cache_path)
        .map_err(OriginError::from)?;
    let cache_writer = Arc::new(AsyncMutex::new(cache_file));

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let retry_count = Arc::new(AtomicUsize::new(0));
    let fail_count = Arc::new(AtomicUsize::new(0));
    let succ_count = Arc::new(AtomicUsize::new(0));
    let progress = Arc::new(AtomicUsize::new(0));
    let total_cost_usd = Arc::new(std::sync::atomic::AtomicU64::new(0)); // cost * 1_000_000
    let total_in_tokens = Arc::new(AtomicUsize::new(0));
    let total_out_tokens = Arc::new(AtomicUsize::new(0));
    let total_cache_create = Arc::new(AtomicUsize::new(0));
    let total_cache_read = Arc::new(AtomicUsize::new(0));
    let total = todo.len();

    let mut handles = Vec::with_capacity(todo.len());
    for tuple in todo {
        let sem = semaphore.clone();
        let model = model.to_string();
        let writer = cache_writer.clone();
        let retries = retry_count.clone();
        let fails = fail_count.clone();
        let succs = succ_count.clone();
        let prog = progress.clone();

        let cost_us = total_cost_usd.clone();
        let in_tok = total_in_tokens.clone();
        let out_tok = total_out_tokens.clone();
        let cc_tok = total_cache_create.clone();
        let cr_tok = total_cache_read.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let mut last_err: Option<String> = None;
            let mut result: Option<JudgmentResult> = None;
            for attempt in 0..=max_retries {
                match judge_single_tuple_model_with_cost(&tuple, &model).await {
                    Ok((r, cost)) => {
                        if let Some(c) = cost {
                            cost_us.fetch_add((c.cost_usd * 1_000_000.0) as u64, Ordering::Relaxed);
                            in_tok.fetch_add(c.input_tokens as usize, Ordering::Relaxed);
                            out_tok.fetch_add(c.output_tokens as usize, Ordering::Relaxed);
                            cc_tok.fetch_add(c.cache_creation_tokens as usize, Ordering::Relaxed);
                            cr_tok.fetch_add(c.cache_read_tokens as usize, Ordering::Relaxed);
                        }
                        result = Some(r);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e.to_string());
                        if attempt < max_retries {
                            retries.fetch_add(1, Ordering::Relaxed);
                            let delay_ms = 500u64 * (1 << attempt);
                            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        }
                    }
                }
            }

            let done = prog.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(50) || done == total {
                eprintln!(
                    "[judge] progress {}/{} (succ={}, fail={}, retries={})",
                    done,
                    total,
                    succs.load(Ordering::Relaxed),
                    fails.load(Ordering::Relaxed),
                    retries.load(Ordering::Relaxed)
                );
            }

            match result {
                Some(r) => {
                    let line = match serde_json::to_string(&r) {
                        Ok(s) => s,
                        Err(e) => {
                            fails.fetch_add(1, Ordering::Relaxed);
                            eprintln!("[judge] serialize FAIL for {:?}: {}", tuple.question, e);
                            return None;
                        }
                    };
                    let mut guard = writer.lock().await;
                    if let Err(e) = writeln!(*guard, "{}", line) {
                        fails.fetch_add(1, Ordering::Relaxed);
                        eprintln!("[judge] write FAIL for {:?}: {}", tuple.question, e);
                        return None;
                    }
                    let _ = guard.flush();
                    drop(guard);
                    succs.fetch_add(1, Ordering::Relaxed);
                    Some(r)
                }
                None => {
                    fails.fetch_add(1, Ordering::Relaxed);
                    let q_preview: String = tuple.question.chars().take(60).collect();
                    eprintln!(
                        "[judge] FAILED after {} retries: {:?} — last_err: {}",
                        max_retries,
                        q_preview,
                        last_err.unwrap_or_else(|| "?".into())
                    );
                    None
                }
            }
        });
        handles.push(handle);
    }

    let mut new_results = Vec::new();
    for handle in handles {
        if let Ok(Some(r)) = handle.await {
            new_results.push(r);
        }
    }

    let cost_total_dollars = total_cost_usd.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let succ_n = succ_count.load(Ordering::Relaxed) as f64;
    let mean_cost = if succ_n > 0.0 {
        cost_total_dollars / succ_n
    } else {
        0.0
    };
    eprintln!(
        "[judge] DONE: {} succ, {} fail, {} retries (cache: {} → {})",
        succ_count.load(Ordering::Relaxed),
        fail_count.load(Ordering::Relaxed),
        retry_count.load(Ordering::Relaxed),
        cached.len(),
        cached.len() + new_results.len()
    );
    eprintln!(
        "[judge] cost: ${:.4} total (${:.5}/call) | in={} out={} cache_create={} cache_read={}",
        cost_total_dollars,
        mean_cost,
        total_in_tokens.load(Ordering::Relaxed),
        total_out_tokens.load(Ordering::Relaxed),
        total_cache_create.load(Ordering::Relaxed),
        total_cache_read.load(Ordering::Relaxed),
    );
    if mean_cost > 0.05 {
        eprintln!(
            "[judge] WARN: mean cost ${:.5}/call is high — claude -p re-creates the system-prompt cache each call. Consider Anthropic API direct (~1000x cheaper) for full runs.",
            mean_cost
        );
    }

    let mut all = cached;
    all.extend(new_results);
    Ok(all)
}

/// Judge a single (question, ground_truth, answer) tuple via `claude -p`.
///
/// Passes the prompt via stdin and disables all tools (`--allowedTools ""`), which
/// prevents Claude Code's agentic tool-calling loop and gets a direct text/JSON response.
/// OAuth auth from the user's existing login is used (no API key required).
pub async fn judge_single_tuple(tuple: &JudgmentTuple) -> Result<JudgmentResult, OriginError> {
    judge_single_tuple_model(tuple, "haiku").await
}

/// Judge a single tuple with a specific Claude model.
///
/// Backward-compat wrapper that drops cost telemetry. New callers should prefer
/// `judge_single_tuple_model_with_cost`.
pub async fn judge_single_tuple_model(
    tuple: &JudgmentTuple,
    model: &str,
) -> Result<JudgmentResult, OriginError> {
    judge_single_tuple_model_with_cost(tuple, model)
        .await
        .map(|(r, _)| r)
}

/// Judge a single tuple, also extracting cost telemetry from the CLI envelope.
pub async fn judge_single_tuple_model_with_cost(
    tuple: &JudgmentTuple,
    model: &str,
) -> Result<(JudgmentResult, Option<JudgeCostInfo>), OriginError> {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let prompt = task_judge_prompt(
        &tuple.category,
        &tuple.question,
        &tuple.ground_truth,
        &tuple.answer,
    );

    let json_schema = r#"{"type":"object","properties":{"score":{"type":"integer","enum":[0,1]},"reason":{"type":"string"}},"required":["score","reason"]}"#;

    // No system prompt — all instructions are in the user prompt (shared with batch API).
    // Strip ANTHROPIC_API_KEY so Claude Code uses Max-plan OAuth instead of an empty
    // API balance (mirrors ClaudeCliProvider in llm_provider.rs).
    let mut child = Command::new("claude")
        .env_remove("ANTHROPIC_API_KEY")
        .args([
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--json-schema",
            json_schema,
            "--no-session-persistence",
            "--allowedTools",
            "",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| OriginError::Generic(format!("claude -p failed to start: {}", e)))?;

    // Write prompt to stdin then close it.
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .await
            .map_err(|e| OriginError::Generic(format!("write to claude stdin failed: {}", e)))?;
        // drop closes stdin
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| OriginError::Generic(format!("claude -p wait failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout_preview = String::from_utf8_lossy(&output.stdout);
        return Err(OriginError::Generic(format!(
            "claude -p exited with error: stderr={} stdout={}",
            stderr.trim(),
            stdout_preview.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Try to also extract cost telemetry from the envelope (best-effort, never fatal).
    let cost = serde_json::from_str::<serde_json::Value>(stdout.trim())
        .ok()
        .map(|env| {
            let usage = env.get("usage");
            JudgeCostInfo {
                cost_usd: env
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

    // Parse the JSON response. `--output-format json` returns an envelope; the structured
    // output lives in the `structured_output` field when `--json-schema` is used.
    let parsed: serde_json::Value = parse_judge_json(&stdout).map_err(|e| {
        OriginError::Generic(format!(
            "judge response parse error: {} — raw: {}",
            e, stdout
        ))
    })?;

    let score = parsed.get("score").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("no reason")
        .to_string();

    Ok((
        JudgmentResult {
            question: tuple.question.clone(),
            approach: tuple.approach.clone(),
            score,
            reason,
            context_tokens: tuple.context_tokens,
        },
        cost,
    ))
}

/// Strict batch judge prompt for N tuples per call.
///
/// Explicit conservative scoring rules counter the leniency bias observed when
/// the model judges multiple tuples in one call (probe 2026-04-29 measured
/// 90% agreement with single-call gold for standard prompt vs 100% for strict).
pub fn strict_batch_judge_prompt(tuples: &[JudgmentTuple]) -> String {
    let mut s = String::with_capacity(2048 + tuples.len() * 256);
    s.push_str(
        "You will judge multiple (question, ground_truth, model_answer) tuples. Be CONSERVATIVE and STRICT.\n\n",
    );
    s.push_str("Scoring rules:\n");
    s.push_str("- Score 1 ONLY if model_answer explicitly contains the exact information in ground_truth\n");
    s.push_str("- Score 0 if answer is a vague paraphrase, implication, or generic equivalent\n");
    s.push_str(
        "- Score 0 if answer is missing any required element of a multi-part ground_truth\n",
    );
    s.push_str(
        "- Score 0 if answer hedges on a key fact (date, name, number) the question requires\n",
    );
    s.push_str(
        "- Off-by-one tolerance allowed only for explicit numeric temporal counts (days/weeks/months)\n\n",
    );
    s.push_str(&format!(
        "Return JSON object with a 'results' array containing exactly {} entries in input order.\n\nTuples:\n",
        tuples.len()
    ));
    for (i, t) in tuples.iter().enumerate() {
        s.push_str(&format!(
            "[{}] Q: {}\n    GT: {}\n    A: {}\n\n",
            i + 1,
            t.question,
            t.ground_truth,
            t.answer
        ));
    }
    s
}

// Markdown fence stripper has moved to `cli_batch::strip_markdown_fence`.
use crate::eval::cli_batch::strip_markdown_fence;

/// Defensive parser for batch judge envelope. Returns Vec<(score, reason)> or None.
///
/// Tries multiple strategies because `--json-schema` enforcement is unreliable
/// after several --resume turns (model drifts to conversational mode and returns
/// markdown-wrapped JSON in `.result` field instead of `.structured_output`).
pub fn parse_batch_judge_envelope(stdout: &str) -> Option<Vec<(u8, String)>> {
    let trimmed = stdout.trim();
    let env: serde_json::Value = serde_json::from_str(trimmed).ok()?;

    // Strategy 1: structured_output.results (primary)
    if let Some(results) = env
        .get("structured_output")
        .and_then(|v| v.get("results"))
        .and_then(|v| v.as_array())
    {
        if !results.is_empty() {
            return Some(extract_score_reason_pairs(results));
        }
    }

    // Strategy 2: result field with markdown fence (schema drift fallback)
    if let Some(result_str) = env.get("result").and_then(|v| v.as_str()) {
        let stripped = strip_markdown_fence(result_str);
        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(&stripped) {
            if let Some(results) = inner.get("results").and_then(|v| v.as_array()) {
                if !results.is_empty() {
                    return Some(extract_score_reason_pairs(results));
                }
            }
        }
    }

    None
}

fn extract_score_reason_pairs(arr: &[serde_json::Value]) -> Vec<(u8, String)> {
    arr.iter()
        .map(|v| {
            let score = v.get("score").and_then(|x| x.as_u64()).unwrap_or(0) as u8;
            let reason = v
                .get("reason")
                .and_then(|x| x.as_str())
                .unwrap_or("no reason")
                .to_string();
            (score, reason)
        })
        .collect()
}

/// Run one batched judge subprocess call.
///
/// If `session_id` is `Some`, uses `--resume` for cache reuse. Returns
/// `(results, cost, new_session_id)`. The new_session_id is the session of THIS
/// call (caller passes it into the next call's `session_id` for --resume).
async fn run_batch_judge_subprocess(
    prompt: &str,
    model: &str,
    session_id: Option<&str>,
) -> Result<(Vec<(u8, String)>, Option<JudgeCostInfo>, Option<String>), OriginError> {
    use crate::eval::cli_batch::run_cli_batch_subprocess;

    let json_schema = r#"{"type":"object","properties":{"results":{"type":"array","items":{"type":"object","properties":{"score":{"type":"integer","enum":[0,1]},"reason":{"type":"string"}},"required":["score","reason"]}}},"required":["results"]}"#;

    let (stdout, cost, new_sid) =
        run_cli_batch_subprocess(prompt, model, json_schema, session_id).await?;

    let results = parse_batch_judge_envelope(&stdout).ok_or_else(|| {
        OriginError::Generic(format!(
            "batch judge parse error — raw envelope head: {}",
            stdout.chars().take(200).collect::<String>()
        ))
    })?;

    Ok((results, cost, new_sid))
}

/// Batched persistent judging with --resume cache reuse + session rotation.
///
/// Cost-optimized path:
/// - Each subprocess call judges `batch_size` tuples (10 = empirical sweet spot).
/// - Within a session, second+ calls use `--resume` so claude-code re-loads cached
///   system prompt cheaply (cache_create drops ~92k → ~3k after first call).
/// - Every `rotation_calls` calls (default 3), session is reset to avoid schema
///   drift — claude treats long --resume conversations as casual chat and stops
///   honoring `--json-schema`, falling back to markdown-wrapped output.
/// - Defensive parser handles markdown fence fallback if schema drift slips through.
/// - Strict prompt counters the leniency bias observed in standard batched judging.
///
/// Persistence + retry semantics match `judge_with_claude_model_persistent`:
/// JSONL append-only cache, resume across runs, exp-backoff retry per batch.
pub async fn judge_with_claude_model_batched_persistent(
    tuples: &[JudgmentTuple],
    batch_size: usize,
    rotation_calls: usize,
    model: &str,
    cache_path: &Path,
    max_retries: u32,
) -> Result<Vec<JudgmentResult>, OriginError> {
    use std::collections::HashSet;
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Write};

    let cached: Vec<JudgmentResult> = if cache_path.exists() {
        let f = std::fs::File::open(cache_path).map_err(OriginError::from)?;
        BufReader::new(f)
            .lines()
            .map_while(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<JudgmentResult>(&l).ok())
            .collect()
    } else {
        Vec::new()
    };

    let cached_keys: HashSet<(String, String)> = cached
        .iter()
        .map(|r| (r.question.clone(), r.approach.clone()))
        .collect();

    let todo: Vec<&JudgmentTuple> = tuples
        .iter()
        .filter(|t| !cached_keys.contains(&(t.question.clone(), t.approach.clone())))
        .collect();

    eprintln!(
        "[judge-batch] cache: {} existing, {} to judge ({} total) | batch_size={} rotation={}",
        cached.len(),
        todo.len(),
        tuples.len(),
        batch_size,
        rotation_calls
    );

    if todo.is_empty() {
        return Ok(cached);
    }

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(OriginError::from)?;
    }
    let mut cache_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(cache_path)
        .map_err(OriginError::from)?;

    let total_batches = todo.len().div_ceil(batch_size);
    let mut new_results: Vec<JudgmentResult> = Vec::new();
    let mut session_id: Option<String> = None;
    let mut calls_in_session: usize = 0;
    let mut total_cost_usd: f64 = 0.0;
    let mut total_cache_create: u64 = 0;
    let mut total_cache_read: u64 = 0;
    let mut total_input: u64 = 0;
    let mut total_output: u64 = 0;
    let mut succ_tuples: usize = 0;
    let mut fail_batches: usize = 0;
    let mut retry_count: usize = 0;

    for (batch_i, chunk) in todo.chunks(batch_size).enumerate() {
        let chunk_owned: Vec<JudgmentTuple> = chunk.iter().map(|t| (*t).clone()).collect();
        let prompt = strict_batch_judge_prompt(&chunk_owned);

        // Rotate session if we hit the cap.
        if calls_in_session >= rotation_calls {
            session_id = None;
            calls_in_session = 0;
        }

        #[allow(clippy::type_complexity)]
        let mut batch_result: Option<(
            Vec<(u8, String)>,
            Option<JudgeCostInfo>,
            Option<String>,
        )> = None;
        let mut last_err: Option<String> = None;

        for attempt in 0..=max_retries {
            match run_batch_judge_subprocess(&prompt, model, session_id.as_deref()).await {
                Ok((results, cost, sid)) if results.len() == chunk_owned.len() => {
                    batch_result = Some((results, cost, sid));
                    break;
                }
                Ok((results, _, _)) => {
                    last_err = Some(format!(
                        "expected {} results, got {}",
                        chunk_owned.len(),
                        results.len()
                    ));
                    if attempt < max_retries {
                        retry_count += 1;
                        // Reset session — schema drift suspected.
                        session_id = None;
                        calls_in_session = 0;
                        let delay_ms = 500u64 * (1 << attempt);
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                    if attempt < max_retries {
                        retry_count += 1;
                        // Reset session on error — may be transient state corruption.
                        session_id = None;
                        calls_in_session = 0;
                        let delay_ms = 500u64 * (1 << attempt);
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        match batch_result {
            Some((results, cost, sid)) => {
                if let Some(c) = cost {
                    total_cost_usd += c.cost_usd;
                    total_cache_create += c.cache_creation_tokens;
                    total_cache_read += c.cache_read_tokens;
                    total_input += c.input_tokens;
                    total_output += c.output_tokens;
                }
                if session_id.is_none() {
                    session_id = sid;
                    calls_in_session = 1;
                } else {
                    calls_in_session += 1;
                }

                for (i, (score, reason)) in results.iter().enumerate() {
                    let tuple = &chunk_owned[i];
                    let r = JudgmentResult {
                        question: tuple.question.clone(),
                        approach: tuple.approach.clone(),
                        score: *score,
                        reason: reason.clone(),
                        context_tokens: tuple.context_tokens,
                    };
                    let line = match serde_json::to_string(&r) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[judge-batch] serialize FAIL: {}", e);
                            continue;
                        }
                    };
                    if let Err(e) = writeln!(cache_file, "{}", line) {
                        eprintln!("[judge-batch] write FAIL: {}", e);
                        continue;
                    }
                    new_results.push(r);
                    succ_tuples += 1;
                }
                let _ = cache_file.flush();
                eprintln!(
                    "[judge-batch] batch {}/{} ok | call_in_session={} | cost so far: ${:.4} | tuples: {}",
                    batch_i + 1,
                    total_batches,
                    calls_in_session,
                    total_cost_usd,
                    succ_tuples
                );
            }
            None => {
                fail_batches += 1;
                eprintln!(
                    "[judge-batch] batch {}/{} FAILED after {} retries: {}",
                    batch_i + 1,
                    total_batches,
                    max_retries,
                    last_err.unwrap_or_else(|| "?".into())
                );
                // Reset session for next batch
                session_id = None;
                calls_in_session = 0;
            }
        }
    }

    let mean_cost_per_call = if total_batches > fail_batches {
        total_cost_usd / (total_batches - fail_batches) as f64
    } else {
        0.0
    };
    eprintln!(
        "[judge-batch] DONE: {} succ tuples / {} target | {} batches failed | {} retries | sessions={} | total_cost=${:.4} (${:.4}/call)",
        succ_tuples,
        todo.len(),
        fail_batches,
        retry_count,
        total_batches.div_ceil(rotation_calls),
        total_cost_usd,
        mean_cost_per_call
    );
    eprintln!(
        "[judge-batch] tokens: input={} output={} cache_create={} cache_read={}",
        total_input, total_output, total_cache_create, total_cache_read
    );

    let mut all = cached;
    all.extend(new_results);
    Ok(all)
}

/// Try to extract the judgment JSON object from `claude -p` output.
///
/// When `--output-format json` is combined with `--json-schema`, Claude Code returns an
/// envelope like:
/// ```json
/// {"type":"result", "structured_output": {"score":1, "reason":"..."}, ...}
/// ```
/// We try several strategies to locate the score/reason object:
/// 1. `structured_output` field in the envelope (primary path).
/// 2. `result` field in the envelope (fallback for older CLI versions).
/// 3. Top-level object if it already contains `score`.
/// 4. Extract any `{...}` substring (last resort).
pub fn parse_judge_json(stdout: &str) -> Result<serde_json::Value, serde_json::Error> {
    let trimmed = stdout.trim();

    if let Ok(envelope) = serde_json::from_str::<serde_json::Value>(trimmed) {
        // Strategy 1: structured_output field (primary — used when --json-schema is set).
        if let Some(so) = envelope.get("structured_output") {
            if so.get("score").is_some() {
                return Ok(so.clone());
            }
        }
        // Strategy 2: result field (text-mode fallback).
        if let Some(result) = envelope.get("result") {
            if result.get("score").is_some() {
                return Ok(result.clone());
            }
            // result may be a JSON string — try to parse it.
            if let Some(s) = result.as_str() {
                if let Ok(inner) = serde_json::from_str::<serde_json::Value>(s) {
                    if inner.get("score").is_some() {
                        return Ok(inner);
                    }
                }
            }
        }
        // Strategy 3: top-level already has score.
        if envelope.get("score").is_some() {
            return Ok(envelope);
        }
    }

    // Strategy 4: extract the first balanced {...} block.
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start <= end {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed[start..=end]) {
                if v.get("score").is_some() {
                    return Ok(v);
                }
            }
        }
    }

    // Final fallback: return a parse error to surface the raw output.
    serde_json::from_str(trimmed)
}

/// Aggregate judgment results into a report sorted by accuracy descending.
pub fn aggregate_judgments(results: &[JudgmentResult], judge_model: &str) -> JudgedE2EReport {
    let mut by_approach: HashMap<String, Vec<&JudgmentResult>> = HashMap::new();
    for r in results {
        by_approach.entry(r.approach.clone()).or_default().push(r);
    }

    let mut approach_results: Vec<JudgedApproachResult> = by_approach
        .iter()
        .map(|(approach, items)| {
            let total = items.len();
            let correct = items.iter().filter(|r| r.score == 1).count();
            let accuracy = correct as f64 / total.max(1) as f64;
            let mean_tokens =
                items.iter().map(|r| r.context_tokens as f64).sum::<f64>() / total.max(1) as f64;
            JudgedApproachResult {
                approach: approach.clone(),
                accuracy,
                total,
                correct,
                mean_context_tokens: mean_tokens,
            }
        })
        .collect();

    approach_results.sort_by(|a, b| {
        b.accuracy
            .partial_cmp(&a.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    JudgedE2EReport {
        judge_model: judge_model.to_string(),
        total_judged: results.len(),
        results_by_approach: approach_results,
    }
}

// ===== Task-Specific Prompt Functions =====

/// LongMemEval answer-check prompt. Returns the appropriate judge prompt for the task type.
pub fn lme_anscheck_prompt(task: &str, question: &str, answer: &str, response: &str) -> String {
    match task {
        // LME "temporal-reasoning" + LoCoMo "temporal" — both test temporal recall
        "temporal-reasoning" | "temporal" => {
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no. \
                 If the response is equivalent to the correct answer or contains all the intermediate \
                 steps to get the correct answer, you should also answer yes. If the response only \
                 contains a subset of the information required by the answer, answer no. In addition, \
                 do not penalize off-by-one errors for the number of days. If the question asks for \
                 the number of days/weeks/months, etc., and the model makes off-by-one errors \
                 (e.g., predicting 19 days when the answer is 18), the model's response is still \
                 correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        "knowledge-update" => {
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no. \
                 If the response contains some previous information along with an updated answer, the \
                 response should be considered as correct as long as the updated answer is the required \
                 answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        "single-session-preference" => {
            format!(
                "I will give you a question, a rubric for desired personalized response, and a \
                 response from a model. Please answer yes if the response satisfies the desired \
                 response. Otherwise, answer no. The model does not need to reflect all the points \
                 in the rubric. The response is correct as long as it recalls and utilizes the \
                 user's personal information correctly.\n\n\
                 Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        _ => {
            // Standard benchmark prompt — same as LoCoMo and LME SSU/SSA/MS.
            // Includes equivalence + subset guidance for fair evaluation.
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no. \
                 If the response is equivalent to the correct answer or contains all the intermediate \
                 steps to get the correct answer, you should also answer yes. If the response only \
                 contains a subset of the information required by the answer, answer no.\n\n\
                 Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
    }
}

/// LongMemEval answer prompt. Returns (user_prompt, system_prompt) for generating answers.
pub fn lme_answer_prompt(question: &str, context: &str, question_type: &str) -> (String, String) {
    if context.is_empty() {
        return (
            format!(
                "Question: {}\n\nAnswer the question as best you can. Be specific and concise.",
                question
            ),
            "Be specific and concise. Respond in 1-3 sentences.".to_string(),
        );
    }
    match question_type {
        "single-session-preference" => {
            let prompt = format!(
                "The following context contains information about a user's preferences, \
                 interests, and past choices:\n\n{}\n\nQuestion: {}\n\n\
                 Use the user's preferences and interests from the context to \
                 personalize your response. Apply their known preferences even if \
                 this specific scenario isn't mentioned.",
                context, question
            );
            let sys = "You are a personalized assistant. Use the user's known preferences \
                to tailor your response. Be specific and concise. Respond in 1-3 sentences."
                .to_string();
            (prompt, sys)
        }
        _ => {
            let prompt = format!(
                "Context:\n{}\n\nQuestion: {}\n\nAnswer the question based on the context provided. \
                 Be specific and concise.",
                context, question
            );
            let sys =
                "Answer the question based on the provided context. Be specific and concise. \
                Respond in 1-3 sentences."
                    .to_string();
            (prompt, sys)
        }
    }
}

/// LoCoMo judge prompt. Standard binary yes/no judge for LoCoMo eval.
pub fn locomo_judge_prompt(question: &str, ground_truth: &str, model_answer: &str) -> String {
    format!(
        "I will give you a question, a correct answer, and a response from a model. \
         Please answer yes if the response contains the correct answer. Otherwise, answer no. \
         If the response is equivalent to the correct answer or contains all the intermediate \
         steps to get the correct answer, you should also answer yes. If the response only \
         contains a subset of the information required by the answer, answer no.\n\n\
         Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
         Is the model response correct? Answer yes or no only.",
        question, ground_truth, model_answer
    )
}

// ===== Batch API Judge =====

/// Judge answer tuples using Anthropic Batch API.
///
/// 50% cheaper than direct API, no rate limits. Cost cap via
/// `EVAL_COST_CAP` env var (default $2).
pub async fn judge_with_batch_api(
    tuples: &[JudgmentTuple],
    judge_model: &str,
    cost_cap: Option<f64>,
) -> Result<Vec<JudgmentResult>, crate::error::OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| crate::error::OriginError::Generic("ANTHROPIC_API_KEY not set".into()))?;
    let cost_cap = cost_cap.unwrap_or_else(|| {
        std::env::var("EVAL_COST_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0)
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| crate::error::OriginError::Generic(format!("reqwest build: {e}")))?;

    // Build batch requests — each tuple becomes a judge prompt
    let requests: Vec<(String, String, Option<String>, usize)> = tuples
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let prompt = task_judge_prompt(&t.category, &t.question, &t.ground_truth, &t.answer);
            (format!("judge_{i}"), prompt, None, 10usize)
        })
        .collect();

    eprintln!(
        "[judge_batch] Submitting {} requests (model={judge_model})...",
        requests.len()
    );

    let batch_id = submit_batch(&client, &api_key, requests, judge_model, cost_cap)
        .await
        .map_err(|e| crate::error::OriginError::Generic(format!("batch submit: {e}")))?;

    let results_url = poll_batch(&client, &api_key, &batch_id)
        .await
        .map_err(|e| crate::error::OriginError::Generic(format!("batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, &api_key, &results_url)
        .await
        .map_err(|e| crate::error::OriginError::Generic(format!("batch download: {e}")))?;

    let mut results = Vec::new();
    for (i, tuple) in tuples.iter().enumerate() {
        let id = format!("judge_{i}");
        let (score, reason) = match raw_results.get(&id) {
            Some(resp) => {
                let lower = resp.to_lowercase();
                if lower.starts_with("yes") {
                    (1u8, resp.clone())
                } else {
                    (0u8, resp.clone())
                }
            }
            None => {
                eprintln!("[judge_batch] missing result for {id}");
                (0, "judge error: missing result".to_string())
            }
        };
        results.push(JudgmentResult {
            question: tuple.question.clone(),
            approach: tuple.approach.clone(),
            score,
            reason,
            context_tokens: tuple.context_tokens,
        });
    }

    eprintln!(
        "[judge_batch] Done. {}/{} judged.",
        results.len(),
        tuples.len()
    );
    Ok(results)
}
