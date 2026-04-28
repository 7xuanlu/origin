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
    for handle in handles {
        match handle.await {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => log::warn!("[judge] judgment failed: {}", e),
            Err(e) => log::warn!("[judge] task panicked: {}", e),
        }
    }

    Ok(results)
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
pub async fn judge_single_tuple_model(
    tuple: &JudgmentTuple,
    model: &str,
) -> Result<JudgmentResult, OriginError> {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let prompt = task_judge_prompt(
        &tuple.category,
        &tuple.question,
        &tuple.ground_truth,
        &tuple.answer,
    );

    let json_schema = r#"{"type":"object","properties":{"score":{"type":"integer","enum":[0,1]},"reason":{"type":"string"}},"required":["score","reason"]}"#;

    // No system prompt — all instructions are in the user prompt (shared with batch API)
    let mut child = Command::new("claude")
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

    Ok(JudgmentResult {
        question: tuple.question.clone(),
        approach: tuple.approach.clone(),
        score,
        reason,
        context_tokens: tuple.context_tokens,
    })
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
