// SPDX-License-Identifier: Apache-2.0
//! E2E answer quality evaluation: generate answers from context, judge quality.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::eval::fixtures::load_fixtures;
use crate::eval::judge::JudgmentTuple;
use crate::eval::shared::{count_tokens, eval_shared_embedder, run_entity_extraction_for_eval};
use crate::events::NoopEmitter;
use crate::read_scope::ReadScope;
use crate::sources::RawDocument;
use crate::tuning::ConfidenceConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ===== Phase 2: End-to-End LLM Answer Evaluation =====

/// End-to-end answer quality for one approach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2EAnswerResult {
    pub approach: String,
    /// 0–1: fraction of relevant info captured in answer
    pub mean_answer_score: f64,
    /// tokens sent as context (measured via tiktoken)
    pub mean_context_tokens: f64,
    /// tokens in LLM response (from API usage field)
    pub mean_answer_tokens: f64,
    /// context + answer
    pub mean_total_tokens: f64,
    pub queries_evaluated: usize,
}

/// Full end-to-end evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2EEvalReport {
    pub results: Vec<E2EAnswerResult>,
    pub model: String,
    pub methodology: String,
}

/// Score an LLM answer against relevant seeds using keyword overlap.
///
/// For each relevant seed, extracts key words (length > 4) and checks
/// whether at least 30% of them appear in the answer. Score = fraction
/// of relevant seeds whose key content appears in the answer.
pub(crate) fn score_answer(answer: &str, relevant_seeds: &[&str]) -> f64 {
    if relevant_seeds.is_empty() {
        return 0.0;
    }

    let answer_lower = answer.to_lowercase();
    let mut found = 0usize;

    for seed_content in relevant_seeds {
        let key_words: Vec<&str> = seed_content
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();

        if key_words.is_empty() {
            continue;
        }

        let matches = key_words
            .iter()
            .filter(|w| answer_lower.contains(&w.to_lowercase() as &str))
            .count();

        if matches as f64 / key_words.len() as f64 >= 0.3 {
            found += 1;
        }
    }

    found as f64 / relevant_seeds.len() as f64
}

/// Call the Anthropic API and return (answer_text, input_tokens, output_tokens).
/// Returns Err on API failure (caller should skip the case rather than panic).
async fn call_llm_for_answer(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> Result<(String, usize, usize), String> {
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    });

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {e}"))?;

    let answer = json["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|block| block["text"].as_str())
        .unwrap_or("")
        .to_string();

    let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;

    Ok((answer, input_tokens, output_tokens))
}

/// End-to-end answer evaluation: send context to LLM, judge answer quality.
///
/// Requires ANTHROPIC_API_KEY environment variable.
///
/// Tests three approaches:
/// - FlatMarkdown: all seeds as markdown context
/// - Wenlan: search results as context
/// - NoContext: no context (LLM baseline)
///
/// For each case, composes a prompt with context + query, sends to Haiku,
/// and scores the answer via keyword overlap against relevant seeds.
///
/// `limit` controls the search top-K; `max_cases` caps API calls for cost control
/// (each case = 3 API calls, one per approach).
pub async fn run_e2e_answer_eval(
    fixture_dir: &Path,
    limit: usize,
    max_cases: usize,
) -> Result<E2EEvalReport, WenlanError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| WenlanError::Generic("ANTHROPIC_API_KEY not set".to_string()))?;

    let model = "claude-haiku-4-5-20251001";
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| WenlanError::Generic(format!("failed to build reqwest client: {e}")))?;

    let cases = load_fixtures(fixture_dir)?;
    let confidence_cfg = ConfidenceConfig::default();

    // Pre-create shared embedder so each case reuses the loaded model.
    let shared_embedder = eval_shared_embedder();

    // Per-approach accumulators: answer_score, context_tokens, answer_tokens
    let approach_keys = ["flat_markdown", "origin", "no_context"];
    let mut scores: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut ctx_tokens: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut ans_tokens: HashMap<&str, Vec<f64>> = HashMap::new();
    for key in &approach_keys {
        scores.insert(key, Vec::new());
        ctx_tokens.insert(key, Vec::new());
        ans_tokens.insert(key, Vec::new());
    }

    let mut cases_done = 0usize;

    for case in &cases {
        if cases_done >= max_cases {
            break;
        }
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }

        // Gather relevant seeds (relevance >= 2) for judging
        let relevant_seed_contents: Vec<&str> = case
            .seeds
            .iter()
            .filter(|s| s.relevance >= 2)
            .map(|s| s.content.as_str())
            .collect();
        if relevant_seed_contents.is_empty() {
            continue;
        }

        let all_seeds: Vec<&crate::eval::fixtures::SeedMemory> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .collect();

        // ---- Build contexts for each approach ----

        // FlatMarkdown: all seeds as numbered markdown sections
        let flat_context = all_seeds
            .iter()
            .enumerate()
            .map(|(i, s)| format!("## Memory {}\n{}", i + 1, s.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Wenlan: seed ephemeral DB, run hybrid search
        let origin_context = {
            let case_tmp = tempfile::tempdir()
                .map_err(|e| WenlanError::Generic(format!("tempdir e2e: {e}")))?;
            let db = MemoryDB::new_with_shared_embedder(
                case_tmp.path(),
                Arc::new(NoopEmitter),
                shared_embedder.clone(),
            )
            .await?;
            let docs: Vec<RawDocument> = all_seeds
                .iter()
                .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
                .collect();
            db.upsert_documents(docs).await?;
            let scope = match case.space.as_deref() {
                None => ReadScope::Global,
                Some("uncategorized") => ReadScope::Uncategorized,
                Some(space) => ReadScope::Space(space.to_string()),
            };
            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    &scope,
                    None,
                    Some(1.0),
                    Some(1.0),
                    None,
                )
                .await?;
            results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("## Result {}\n{}", i + 1, r.content))
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        // ---- Send each approach to the LLM ----

        let approaches: &[(&str, &str)] = &[
            ("flat_markdown", &flat_context),
            ("origin", &origin_context),
            ("no_context", ""),
        ];

        for (approach_key, context) in approaches {
            let prompt = if context.is_empty() {
                format!(
                    "Question: {}\n\nAnswer the question as best you can. Be specific and concise.",
                    case.query
                )
            } else {
                format!(
                    "Context:\n{}\n\nQuestion: {}\n\nAnswer the question using only the context provided. Be specific and concise.",
                    context, case.query
                )
            };

            let ctx_tok_count = if context.is_empty() {
                0usize
            } else {
                count_tokens(context)
            };

            // Rate limit between calls
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            match call_llm_for_answer(&client, &api_key, model, &prompt).await {
                Ok((answer, _input_tok, output_tok)) => {
                    let score = score_answer(&answer, &relevant_seed_contents);
                    scores.get_mut(approach_key).unwrap().push(score);
                    ctx_tokens
                        .get_mut(approach_key)
                        .unwrap()
                        .push(ctx_tok_count as f64);
                    ans_tokens
                        .get_mut(approach_key)
                        .unwrap()
                        .push(output_tok as f64);
                }
                Err(e) => {
                    log::warn!(
                        "[e2e_eval] case '{}' approach '{}' skipped: {}",
                        case.query,
                        approach_key,
                        e
                    );
                }
            }
        }

        cases_done += 1;
    }

    // Aggregate
    let mut results: Vec<E2EAnswerResult> = Vec::new();
    for key in &approach_keys {
        let score_vec = &scores[key];
        let ctx_vec = &ctx_tokens[key];
        let ans_vec = &ans_tokens[key];
        let n = score_vec.len().max(1) as f64;

        let mean_score = score_vec.iter().sum::<f64>() / n;
        let mean_ctx = ctx_vec.iter().sum::<f64>() / n;
        let mean_ans = ans_vec.iter().sum::<f64>() / n;
        let mean_total = mean_ctx + mean_ans;

        results.push(E2EAnswerResult {
            approach: key.to_string(),
            mean_answer_score: mean_score,
            mean_context_tokens: mean_ctx,
            mean_answer_tokens: mean_ans,
            mean_total_tokens: mean_total,
            queries_evaluated: score_vec.len(),
        });
    }

    Ok(E2EEvalReport {
        results,
        model: model.to_string(),
        methodology: "Keyword overlap judge: answer scores 1 for a relevant seed when ≥30% of its \
            key words (len>4) appear in the answer. Final score = fraction of relevant seeds \
            matched. Context tokens counted via cl100k_base; answer tokens from API usage field."
            .to_string(),
    })
}

// ===== E2E LoCoMo Answer Quality Eval (On-Device LLM) =====

/// Per-approach result for the E2E LoCoMo eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2ELocomoResult {
    /// Approach identifier: "origin", "full_replay", "no_context".
    pub approach: String,
    /// Mean keyword-overlap score between LLM answer and ground truth (0–1).
    pub mean_answer_score: f64,
    /// Mean tokens of context sent to the LLM.
    pub mean_context_tokens: f64,
    /// Number of QA pairs evaluated for this approach.
    pub questions_evaluated: usize,
    /// Mean character length of the LLM's response.
    pub mean_answer_length: f64,
}

/// Full E2E LoCoMo benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2ELocomoReport {
    /// Model name used for inference.
    pub model: String,
    /// Number of conversations evaluated.
    pub conversations: usize,
    /// Max QA pairs sampled per conversation.
    pub questions_per_conv: usize,
    /// Total QA pairs evaluated.
    pub total_questions: usize,
    /// Per-approach results.
    pub results: Vec<E2ELocomoResult>,
}

/// Score an LLM answer against a ground-truth string using keyword overlap.
///
/// Splits the ground truth into words longer than 3 characters and measures
/// what fraction appear in the (lowercased) answer.
fn score_answer_against_ground_truth(answer: &str, ground_truth: &str) -> f64 {
    let answer_lower = answer.to_lowercase();
    let gt_words: Vec<&str> = ground_truth
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();
    if gt_words.is_empty() {
        return 0.0;
    }
    let matches = gt_words
        .iter()
        .filter(|w| answer_lower.contains(&w.to_lowercase() as &str))
        .count();
    matches as f64 / gt_words.len() as f64
}

/// Run end-to-end answer quality evaluation on LoCoMo using the on-device LLM.
///
/// For each LoCoMo conversation:
/// 1. Seeds all observations into an ephemeral DB.
/// 2. For up to `max_questions_per_conv` non-adversarial QA pairs:
///    - **origin**: retrieve top-`search_top_k` results, compose prompt, call LLM.
///    - **full_replay**: use ALL observations as context (skipped if > 4000 tokens).
///    - **no_context**: ask the question with no memory context.
/// 3. Scores each LLM answer against the ground-truth via keyword overlap.
///
/// The `llm_provider` must be an `OnDeviceProvider` (or any `LlmProvider`).
/// This function is `async` but LLM calls are routed through the provider's
/// internal worker thread — no extra `spawn_blocking` needed here.
///
/// Returns an `(E2ELocomoReport, Vec<JudgmentTuple>)` tuple.
///
/// The `JudgmentTuple` list contains raw (question, ground_truth, approach, answer,
/// context_tokens) records for every answered question. Save them with
/// [`save_judgment_tuples`] and score offline with [`judge_with_claude`].
pub async fn run_e2e_locomo_eval(
    locomo_path: &Path,
    max_questions_per_conv: usize,
    search_top_k: usize,
    llm_provider: Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<(E2ELocomoReport, Vec<JudgmentTuple>), WenlanError> {
    use crate::eval::locomo::{extract_observations, load_locomo};
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    let samples = load_locomo(locomo_path)?;

    // Accumulators: (answer_score, context_tokens, answer_len)
    // T10: the `origin_compressed` approach is appended only when the master
    // env gate is on, so a flag-OFF run is byte-identical to today's report.
    let compress_on = crate::retrieval::compress::context_compress_enabled();
    let approach_keys: Vec<&str> = if compress_on {
        vec!["origin", "origin_compressed", "full_replay", "no_context"]
    } else {
        vec!["origin", "full_replay", "no_context"]
    };
    let mut scores: std::collections::HashMap<&str, Vec<f64>> =
        approach_keys.iter().map(|k| (*k, Vec::new())).collect();
    let mut ctx_tokens: std::collections::HashMap<&str, Vec<f64>> =
        approach_keys.iter().map(|k| (*k, Vec::new())).collect();
    let mut ans_lens: std::collections::HashMap<&str, Vec<f64>> =
        approach_keys.iter().map(|k| (*k, Vec::new())).collect();

    // Collect raw tuples for offline LLM judging.
    let mut judgment_tuples: Vec<JudgmentTuple> = Vec::new();

    let total_convs = samples.len();

    // Pre-create shared embedder so each conversation reuses the loaded model.
    let shared_embedder = eval_shared_embedder();

    for (conv_idx, sample) in samples.iter().enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }

        // Build full-replay corpus text (all observations concatenated).
        let corpus_text: String = memories
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let corpus_tokens = count_tokens(&corpus_text);
        // Cap full_replay at 4000 tokens to stay within model's synthesis limit.
        const FULL_REPLAY_TOKEN_LIMIT: usize = 4000;
        let full_replay_context: Option<String> = if corpus_tokens <= FULL_REPLAY_TOKEN_LIMIT {
            Some(corpus_text.clone())
        } else {
            // Truncate by taking observations until we reach the limit.
            let mut truncated = String::new();
            for mem in &memories {
                let candidate = if truncated.is_empty() {
                    mem.content.clone()
                } else {
                    format!("{}\n\n{}", truncated, mem.content)
                };
                if count_tokens(&candidate) > FULL_REPLAY_TOKEN_LIMIT {
                    break;
                }
                truncated = candidate;
            }
            if truncated.is_empty() {
                None // even one observation exceeds the limit — skip full_replay
            } else {
                Some(truncated)
            }
        };

        // Seed ephemeral DB for Wenlan retrieval.
        let tmp = tempfile::tempdir()
            .map_err(|e| WenlanError::Generic(format!("tempdir e2e_locomo: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<crate::sources::RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| crate::sources::RawDocument {
                content: mem.content.clone(),
                source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                source: "memory".to_string(),
                title: format!("{} session {}", mem.speaker, mem.session_num),
                memory_type: Some("fact".to_string()),
                space: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Iterate QA pairs (skip adversarial, cap at max_questions_per_conv).
        let mut questions_done = 0usize;
        for qa in &sample.qa {
            if questions_done >= max_questions_per_conv {
                break;
            }
            if qa.category == 5 {
                continue; // skip adversarial
            }

            let ground_truth = qa
                .answer
                .as_ref()
                .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                .unwrap_or_default();

            if ground_truth.is_empty() {
                continue;
            }

            eprintln!(
                "[e2e_locomo] Conv {}/{}, Q {}/{}...",
                conv_idx + 1,
                total_convs,
                questions_done + 1,
                max_questions_per_conv,
            );

            let system_prompt = "Answer the question using only the provided context. \
                Be specific and concise. Respond in 1-3 sentences."
                .to_string();

            // ---- Wenlan approach: hybrid search ----
            let origin_context = {
                let scope = ReadScope::Space("conversation".to_string());
                let results = db
                    .search_memory(
                        &qa.question,
                        search_top_k,
                        None,
                        &scope,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await?;
                results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| format!("{}. {}", i + 1, r.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let origin_ctx_tokens = count_tokens(&origin_context);

            let origin_request = LlmRequest {
                system_prompt: Some(system_prompt.clone()),
                user_prompt: format!("Context:\n{}\n\nQuestion: {}", origin_context, qa.question),
                max_tokens: 200,
                temperature: 0.1,
                label: Some("e2e_locomo_origin".to_string()),
                timeout_secs: None,
            };
            match llm_provider.generate(origin_request).await {
                Ok(raw_answer) => {
                    let answer = strip_think_tags(&raw_answer);
                    let score = score_answer_against_ground_truth(&answer, &ground_truth);
                    scores.get_mut("origin").unwrap().push(score);
                    ctx_tokens
                        .get_mut("origin")
                        .unwrap()
                        .push(origin_ctx_tokens as f64);
                    ans_lens
                        .get_mut("origin")
                        .unwrap()
                        .push(answer.len() as f64);
                    judgment_tuples.push(JudgmentTuple {
                        question: qa.question.clone(),
                        ground_truth: ground_truth.clone(),
                        approach: "origin".to_string(),
                        answer,
                        context_tokens: origin_ctx_tokens,
                        category: String::new(),
                        question_id: String::new(),
                    });
                }
                Err(e) => {
                    log::warn!("[e2e_locomo] origin approach failed: {e}");
                }
            }

            // ---- T10: Wenlan-compressed approach (env-gated) ----
            if compress_on {
                // Env flag is the master gate; force the tuning `enabled`
                // knob on so the env-gated eval actually compresses (other
                // knobs keep their tuning defaults).
                let mut cfg = crate::retrieval::compress::CompressConfig::from(
                    &crate::tuning::ContextCompressConfig::default(),
                );
                cfg.enabled = true;
                let registry = crate::prompts::PromptRegistry::default();
                let compressed = crate::retrieval::compress::compress_context(
                    &origin_context,
                    &qa.question,
                    Some(llm_provider.clone()),
                    &registry.compress_context,
                    &cfg,
                )
                .await;
                let compressed_ctx_tokens = count_tokens(&compressed);
                let compressed_request = LlmRequest {
                    system_prompt: Some(system_prompt.clone()),
                    user_prompt: format!("Context:\n{}\n\nQuestion: {}", compressed, qa.question),
                    max_tokens: 200,
                    temperature: 0.1,
                    label: Some("e2e_locomo_origin_compressed".to_string()),
                    timeout_secs: None,
                };
                match llm_provider.generate(compressed_request).await {
                    Ok(raw_answer) => {
                        let answer = strip_think_tags(&raw_answer);
                        let score = score_answer_against_ground_truth(&answer, &ground_truth);
                        scores.get_mut("origin_compressed").unwrap().push(score);
                        ctx_tokens
                            .get_mut("origin_compressed")
                            .unwrap()
                            .push(compressed_ctx_tokens as f64);
                        ans_lens
                            .get_mut("origin_compressed")
                            .unwrap()
                            .push(answer.len() as f64);
                        judgment_tuples.push(JudgmentTuple {
                            question: qa.question.clone(),
                            ground_truth: ground_truth.clone(),
                            approach: "origin_compressed".to_string(),
                            answer,
                            context_tokens: compressed_ctx_tokens,
                            category: String::new(),
                            question_id: String::new(),
                        });
                    }
                    Err(e) => {
                        log::warn!("[e2e_locomo] origin_compressed approach failed: {e}");
                    }
                }
            }

            // ---- FullReplay approach ----
            if let Some(ref replay_ctx) = full_replay_context {
                let replay_ctx_tokens = count_tokens(replay_ctx);
                let replay_request = LlmRequest {
                    system_prompt: Some(system_prompt.clone()),
                    user_prompt: format!("Context:\n{}\n\nQuestion: {}", replay_ctx, qa.question),
                    max_tokens: 200,
                    temperature: 0.1,
                    label: Some("e2e_locomo_full_replay".to_string()),
                    timeout_secs: None,
                };
                match llm_provider.generate(replay_request).await {
                    Ok(raw_answer) => {
                        let answer = strip_think_tags(&raw_answer);
                        let score = score_answer_against_ground_truth(&answer, &ground_truth);
                        scores.get_mut("full_replay").unwrap().push(score);
                        ctx_tokens
                            .get_mut("full_replay")
                            .unwrap()
                            .push(replay_ctx_tokens as f64);
                        ans_lens
                            .get_mut("full_replay")
                            .unwrap()
                            .push(answer.len() as f64);
                        judgment_tuples.push(JudgmentTuple {
                            question: qa.question.clone(),
                            ground_truth: ground_truth.clone(),
                            approach: "full_replay".to_string(),
                            answer,
                            context_tokens: replay_ctx_tokens,
                            category: String::new(),
                            question_id: String::new(),
                        });
                    }
                    Err(e) => {
                        log::warn!("[e2e_locomo] full_replay approach failed: {e}");
                    }
                }
            }
            // If full_replay was skipped (too long), we don't push to its accumulators.

            // ---- NoContext approach ----
            let no_ctx_request = LlmRequest {
                system_prompt: Some(
                    "Answer the question as best you can from your knowledge. \
                    Be specific and concise. Respond in 1-3 sentences."
                        .to_string(),
                ),
                user_prompt: format!("Question: {}", qa.question),
                max_tokens: 200,
                temperature: 0.1,
                label: Some("e2e_locomo_no_context".to_string()),
                timeout_secs: None,
            };
            match llm_provider.generate(no_ctx_request).await {
                Ok(raw_answer) => {
                    let answer = strip_think_tags(&raw_answer);
                    let score = score_answer_against_ground_truth(&answer, &ground_truth);
                    scores.get_mut("no_context").unwrap().push(score);
                    ctx_tokens.get_mut("no_context").unwrap().push(0.0);
                    ans_lens
                        .get_mut("no_context")
                        .unwrap()
                        .push(answer.len() as f64);
                    judgment_tuples.push(JudgmentTuple {
                        question: qa.question.clone(),
                        ground_truth: ground_truth.clone(),
                        approach: "no_context".to_string(),
                        answer,
                        context_tokens: 0,
                        category: String::new(),
                        question_id: String::new(),
                    });
                }
                Err(e) => {
                    log::warn!("[e2e_locomo] no_context approach failed: {e}");
                }
            }

            questions_done += 1;
        }
    }

    // Aggregate per-approach
    let total_questions = scores["origin"].len();
    let mut results: Vec<E2ELocomoResult> = Vec::new();
    for key in &approach_keys {
        let score_vec = &scores[key];
        let ctx_vec = &ctx_tokens[key];
        let len_vec = &ans_lens[key];
        let n = score_vec.len().max(1) as f64;

        results.push(E2ELocomoResult {
            approach: key.to_string(),
            mean_answer_score: score_vec.iter().sum::<f64>() / n,
            mean_context_tokens: ctx_vec.iter().sum::<f64>() / n,
            questions_evaluated: score_vec.len(),
            mean_answer_length: len_vec.iter().sum::<f64>() / n,
        });
    }

    Ok((
        E2ELocomoReport {
            model: llm_provider.name().to_string(),
            conversations: samples.len(),
            questions_per_conv: max_questions_per_conv,
            total_questions,
            results,
        },
        judgment_tuples,
    ))
}

/// Run E2E answer quality comparison: flat (search_memory) vs structured (search + concepts).
///
/// For each LoCoMo question:
/// 1. Build flat context: search_memory top-K concatenated
/// 2. Build structured context: search_memory + concept articles (like chat-context)
/// 3. Generate answers from both contexts using on-device LLM
/// 4. Return JudgmentTuples for offline Claude Haiku judging
///
/// Requires enrichment + distillation to be run first (concepts must exist).
/// Call this after seeding + enriching a DB, or use the all-in-one wrapper.
async fn generate_e2e_answers_for_question(
    db: &MemoryDB,
    question: &str,
    ground_truth: &str,
    category: &str,
    search_limit: usize,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<Vec<JudgmentTuple>, WenlanError> {
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    let system_prompt = "Answer the question using only the provided context. \
        Be specific and concise. Respond in 1-3 sentences."
        .to_string();

    let mut tuples = Vec::new();

    // --- Flat context: search_memory only ---
    let scope = ReadScope::Space("conversation".to_string());
    let flat_results = db
        .search_memory(question, search_limit, None, &scope, None, None, None, None)
        .await?;
    let flat_context: String = flat_results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}. {}", i + 1, r.content))
        .collect::<Vec<_>>()
        .join("\n");
    let flat_tokens = count_tokens(&flat_context);

    let flat_request = LlmRequest {
        system_prompt: Some(system_prompt.clone()),
        user_prompt: format!("Context:\n{}\n\nQuestion: {}", flat_context, question),
        max_tokens: 200,
        temperature: 0.1,
        label: Some("e2e_flat".to_string()),
        timeout_secs: None,
    };
    if let Ok(raw) = llm.generate(flat_request).await {
        let answer = strip_think_tags(&raw);
        tuples.push(JudgmentTuple {
            question: question.to_string(),
            ground_truth: ground_truth.to_string(),
            approach: format!("flat_{}", category),
            answer,
            context_tokens: flat_tokens,
            category: category.to_string(),
            question_id: String::new(),
        });
    }

    // --- Structured context: search_memory + concept articles ---
    let mut structured_parts: Vec<String> = Vec::new();

    // Concept articles (like chat-context's "Compiled Knowledge" section)
    let concepts = db.search_pages_strict(question, 3, None).await?;
    if !concepts.is_empty() {
        structured_parts.push("## Compiled Knowledge".to_string());
        for c in &concepts {
            let summary = c.summary.as_deref().unwrap_or("");
            structured_parts.push(format!("**{}**: {}\n{}", c.title, summary, c.content));
        }
    }

    // Memory search results
    if !flat_results.is_empty() {
        structured_parts.push("## Relevant Memories".to_string());
        for (i, r) in flat_results.iter().enumerate() {
            structured_parts.push(format!("{}. {}", i + 1, r.content));
        }
    }

    let structured_context = structured_parts.join("\n\n");
    let structured_tokens = count_tokens(&structured_context);

    let structured_request = LlmRequest {
        system_prompt: Some(system_prompt),
        user_prompt: format!("Context:\n{}\n\nQuestion: {}", structured_context, question),
        max_tokens: 200,
        temperature: 0.1,
        label: Some("e2e_structured".to_string()),
        timeout_secs: None,
    };
    if let Ok(raw) = llm.generate(structured_request).await {
        let answer = strip_think_tags(&raw);
        tuples.push(JudgmentTuple {
            question: question.to_string(),
            ground_truth: ground_truth.to_string(),
            approach: format!("structured_{}", category),
            answer,
            context_tokens: structured_tokens,
            category: category.to_string(),
            question_id: String::new(),
        });
    }

    Ok(tuples)
}

/// Run full E2E answer quality eval on LoCoMo: seed, enrich, distill, generate answers.
///
/// Returns JudgmentTuples for offline judging with `judge_with_claude`.
/// Two approaches per question: "flat_{category}" and "structured_{category}".
pub async fn run_e2e_context_eval(
    locomo_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_conversations: usize,
    max_questions_per_conv: usize,
) -> Result<Vec<JudgmentTuple>, WenlanError> {
    use crate::eval::locomo::{category_name, extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    let mut all_tuples: Vec<JudgmentTuple> = Vec::new();
    let conv_limit = max_conversations.min(samples.len());

    // Pre-create shared embedder so each conversation reuses the loaded model.
    let shared_embedder = eval_shared_embedder();

    for (conv_idx, sample) in samples.iter().take(max_conversations).enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }

        eprintln!(
            "[e2e_context] Conv {}/{} ({}): {} observations",
            conv_idx + 1,
            conv_limit,
            sample.sample_id,
            memories.len(),
        );

        // Seed DB
        let tmp = tempfile::tempdir()
            .map_err(|e| WenlanError::Generic(format!("tempdir e2e_context: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| RawDocument {
                content: mem.content.clone(),
                source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                source: "memory".to_string(),
                title: format!("{} session {}", mem.speaker, mem.session_num),
                memory_type: Some("fact".to_string()),
                space: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Enrich + distill
        eprintln!("  [enriching]...");
        let entities = run_entity_extraction_for_eval(&db, &llm).await?;
        let concepts =
            crate::refinery::distill_pages(&db, Some(&llm), &prompts, &tuning, None).await?;
        eprintln!(
            "  [enriched] {} entities, {} concepts. generating answers...",
            entities, concepts
        );

        // Generate answers for each question
        let mut questions_done = 0usize;
        for qa in &sample.qa {
            if questions_done >= max_questions_per_conv {
                break;
            }
            if qa.category == 5 {
                continue;
            }

            let ground_truth = qa
                .answer
                .as_ref()
                .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                .unwrap_or_default();
            if ground_truth.is_empty() {
                continue;
            }

            let category = category_name(qa.category);

            match generate_e2e_answers_for_question(
                &db,
                &qa.question,
                &ground_truth,
                category,
                search_limit,
                &llm,
            )
            .await
            {
                Ok(tuples) => {
                    all_tuples.extend(tuples);
                }
                Err(e) => {
                    log::warn!("[e2e_context] question failed: {e}");
                }
            }

            questions_done += 1;
            if questions_done.is_multiple_of(10) {
                eprintln!(
                    "  [progress] {}/{} questions",
                    questions_done, max_questions_per_conv
                );
            }
        }

        eprintln!(
            "  Conv done: {} answers generated ({} tuples total)",
            questions_done,
            all_tuples.len(),
        );
    }

    eprintln!(
        "[e2e_context] Total: {} judgment tuples ({} questions x 2 approaches)",
        all_tuples.len(),
        all_tuples.len() / 2,
    );

    Ok(all_tuples)
}

/// Same as run_e2e_context_eval but for LongMemEval.
pub async fn run_e2e_context_eval_longmemeval(
    longmemeval_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_questions: usize,
    _max_answers_per_question: usize,
) -> Result<Vec<JudgmentTuple>, WenlanError> {
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Pre-create shared embedder
    eprintln!("[e2e_context_lme] loading shared embedder...");
    let shared_embedder = eval_shared_embedder();

    let mut all_tuples: Vec<JudgmentTuple> = Vec::new();
    let sample_limit = max_questions.min(samples.len());

    for (q_idx, sample) in samples.iter().take(max_questions).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }

        if q_idx % 25 == 0 {
            eprintln!(
                "[e2e_context_lme] Q {}/{} ({}): {} memories",
                q_idx + 1,
                sample_limit,
                sample.question_id,
                memories.len(),
            );
        }

        // Seed DB with shared embedder
        let tmp = tempfile::tempdir()
            .map_err(|e| WenlanError::Generic(format!("tempdir e2e_lme: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| RawDocument {
                content: mem.content.clone(),
                source_id: format!(
                    "lme_{}_{}_t{}",
                    sample.question_id, mem.session_idx, mem.turn_idx
                ),
                source: "memory".to_string(),
                title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                memory_type: Some(
                    if sample.question_type == "single-session-preference" {
                        "preference"
                    } else {
                        "fact"
                    }
                    .to_string(),
                ),
                space: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Enrich + distill
        let _entities = run_entity_extraction_for_eval(&db, &llm).await?;
        let _concepts =
            crate::refinery::distill_pages(&db, Some(&llm), &prompts, &tuning, None).await?;

        // Generate answers
        let ground_truth = sample
            .answer
            .as_str()
            .unwrap_or(&sample.answer.to_string())
            .to_string();
        if ground_truth.is_empty() {
            continue;
        }

        let category = category_name(&sample.question_type);

        if let Ok(tuples) = generate_e2e_answers_for_question(
            &db,
            &sample.question,
            &ground_truth,
            category,
            search_limit,
            &llm,
        )
        .await
        {
            all_tuples.extend(tuples);
        }

        if q_idx % 50 == 49 {
            eprintln!(
                "  [progress] {}/{} questions, {} tuples",
                q_idx + 1,
                sample_limit,
                all_tuples.len()
            );
        }
    }

    eprintln!(
        "[e2e_context_lme] Total: {} judgment tuples",
        all_tuples.len(),
    );

    Ok(all_tuples)
}

// ===== Batch-based full-scale variants =====

/// Metadata for a pending answer request, submitted via Batch API.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingAnswer {
    question: String,
    ground_truth: String,
    approach: String,
    category: String,
    context_tokens: usize,
}

/// System prompt used for all E2E answer generation.
const E2E_SYSTEM_PROMPT: &str =
    "Answer the question using only the provided context. Be specific and concise. Respond in 1-3 sentences.";

/// V2 answer prompt, opt-in via `WENLAN_EVAL_ANSWER_PROMPT_V2`. Two changes over v1:
/// (a) license a concrete preference-aligned recommendation instead of abstaining,
/// (b) work dates step by step for temporal reasoning. Paired with a larger token
/// budget (see `e2e_max_answer_tokens`) so the chain-of-thought has room. Default OFF
/// keeps the v1 path byte-identical.
const E2E_SYSTEM_PROMPT_V2: &str =
    "Answer the question using the provided context. Be specific. For temporal questions \
     (ordering, durations, \"which came first\", \"how many days between\"), work through the \
     relevant dates step by step, then give the answer. For preference or recommendation \
     questions, give a concrete recommendation aligned with the user's stated preferences \
     instead of declining; only say the context lacks the information if no relevant \
     preference is present. Keep the final answer to a few sentences.";

/// Opt-in gate for the v2 answer prompt. Default OFF (v1). Accepts 1/true/yes/on.
fn e2e_answer_prompt_v2_enabled() -> bool {
    matches!(
        std::env::var("WENLAN_EVAL_ANSWER_PROMPT_V2")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Active E2E system prompt: v2 if opted in, else v1 (byte-identical default).
fn e2e_system_prompt() -> String {
    if e2e_answer_prompt_v2_enabled() {
        E2E_SYSTEM_PROMPT_V2.to_string()
    } else {
        E2E_SYSTEM_PROMPT.to_string()
    }
}

/// Max answer tokens: 512 under v2 (CoT room), 200 under v1 (byte-identical default).
fn e2e_max_answer_tokens() -> usize {
    if e2e_answer_prompt_v2_enabled() {
        512
    } else {
        200
    }
}

/// Controls which retrieval path `build_structured_context` uses.
///
/// - `Quick`: existing `search_memory` quick path (unchanged behavior, graph stream ON).
/// - `CrossRerank(r)`: `search_memory_cross_rerank`; `r=None` disables the CE while
///   still routing through cross_rerank so P3 distill-demotion is symmetric across arms.
///   Pass `WENLAN_GRAPH_MEMORY_STREAM=0` to suppress the stream on the None arm.
pub(crate) enum CtxRetrieval {
    Quick,
    CrossRerank(Option<Arc<dyn crate::reranker::Reranker>>),
}

/// Build structured context for a question against an enriched DB.
///
/// Returns the structured context: search_memory results + concept articles.
/// Matches production `/api/context` assembly pattern.
/// Flat baseline comes from the retrieval-only pipeline caches.
async fn build_structured_context(
    db: &MemoryDB,
    question: &str,
    search_limit: usize,
    domain: Option<&str>,
    retrieval: CtxRetrieval,
    include_pages: bool,
) -> Result<(String, usize), WenlanError> {
    let scope = match domain {
        None => ReadScope::Global,
        Some("uncategorized") => ReadScope::Uncategorized,
        Some(space) => ReadScope::Space(space.to_string()),
    };
    let results = match retrieval {
        CtxRetrieval::Quick => {
            db.search_memory(question, search_limit, None, &scope, None, None, None, None)
                .await?
        }
        CtxRetrieval::CrossRerank(reranker) => {
            db.search_memory_cross_rerank(question, search_limit, None, &scope, None, reranker)
                .await?
        }
    };

    // Source IDs from search results — used to gate concept relevance
    let search_source_ids: std::collections::HashSet<String> =
        results.iter().map(|r| r.source_id.clone()).collect();

    let mut parts: Vec<String> = Vec::new();
    if include_pages {
        use crate::pages::filter_pages_by_source_overlap;
        // Structured: concepts + search results (matches production /api/context).
        // EVAL_CONCEPT_MIN_OVERLAP env var lets us sweep thresholds without code changes;
        // defaults to the production tuning value (2).
        let min_overlap: usize = std::env::var("EVAL_CONCEPT_MIN_OVERLAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| crate::tuning::DistillationConfig::default().page_min_overlap);

        let raw_concepts = db.search_pages_strict(question, 3, None).await?;
        let concepts =
            filter_pages_by_source_overlap(&raw_concepts, &search_source_ids, min_overlap);

        if !raw_concepts.is_empty() {
            for c in &raw_concepts {
                let kept = concepts.iter().any(|k| k.id == c.id);
                let overlap = c
                    .source_memory_ids
                    .iter()
                    .filter(|sid| search_source_ids.contains(sid.as_str()))
                    .count();
                log::info!(
                    "[eval:concept] score={:.4} overlap={}/{} {} title={:?} q={:?}",
                    c.relevance_score,
                    overlap,
                    results.len(),
                    if kept { "KEPT" } else { "FILTERED" },
                    c.title.chars().take(40).collect::<String>(),
                    question.chars().take(50).collect::<String>(),
                );
            }
        }
        if !concepts.is_empty() {
            parts.push("## Compiled Knowledge".to_string());
            for c in &concepts {
                let summary = c.summary.as_deref().unwrap_or("");
                parts.push(format!("**{}**: {}\n{}", c.title, summary, c.content));
            }
        }
    }
    if !results.is_empty() {
        parts.push("## Relevant Memories".to_string());
        for (i, r) in results.iter().enumerate() {
            parts.push(format!("{}. {}", i + 1, r.content));
        }
    }
    let structured_context = parts.join("\n\n");
    let tokens = count_tokens(&structured_context);

    Ok((structured_context, tokens))
}

/// Drive Phase 3 answer generation via `claude -p` subprocesses instead of the Batch API.
///
/// Set `EVAL_PHASE3_CLI=1` to use this path. Useful when the Anthropic API balance is
/// insufficient for Batch API calls — this uses the Max subscription via OAuth instead.
///
/// Returns a `HashMap<String, String>` with `req_id -> answer` pairs in the same format that
/// `download_batch_results` returns, so the Phase 4 merge loop needs no changes.
///
/// Per-request failures are logged as warnings and produce an empty answer string; they are
/// skipped by the Phase 4 `pending.get(custom_id)` lookup because the answer is still present
/// as a key — callers that need strict filtering should check `answer.is_empty()`.
async fn run_phase3_via_cli(
    batch_requests: Vec<(String, String, Option<String>, usize)>,
    cli_concurrency: usize,
) -> HashMap<String, String> {
    use crate::llm_provider::{ClaudeCliProvider, LlmProvider, LlmRequest};
    use futures::StreamExt;

    eprintln!(
        "[phase3_cli] Running {} requests via claude -p (concurrency={})",
        batch_requests.len(),
        cli_concurrency,
    );

    let provider = Arc::new(ClaudeCliProvider::haiku());

    let results: HashMap<String, String> = futures::stream::iter(batch_requests)
        .map(|(req_id, user_prompt, system_prompt, max_tokens)| {
            let provider = provider.clone();
            async move {
                let request = LlmRequest {
                    system_prompt,
                    user_prompt,
                    max_tokens: max_tokens as u32,
                    temperature: 0.0,
                    label: Some("phase3_cli".to_string()),
                    timeout_secs: None,
                };
                match provider.generate(request).await {
                    Ok(answer) => (req_id, answer),
                    Err(e) => {
                        eprintln!("[phase3_cli] WARN: req {req_id} failed: {e}");
                        (req_id, String::new())
                    }
                }
            }
        })
        .buffer_unordered(cli_concurrency)
        .collect()
        .await;

    eprintln!("[phase3_cli] {} answers collected", results.len());
    results
}

// ============================================================================
// Phase 3 batched + persistent CLI path (selected by EVAL_PHASE3_BATCH_SIZE>=2)
// ============================================================================
//
// Why this exists: each `claude -p` subprocess re-loads ~190k tokens of Claude
// Code system prompt. The per-question path above pays that cost 1388 times per
// LoCoMo run. The batched path below judges N (default 10) questions per call
// and uses `--resume` to reuse the cached system prompt across calls in the
// same session, with rotation every 3 calls to avoid schema drift. Pattern
// mirrors `judge_with_claude_model_batched_persistent` in `judge.rs`.
//
// Cost target: ~25x cheaper than per-question path on LoCoMo Phase 3.

/// Cache record for one persisted answer. JSONL append-only.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Phase3AnswerRecord {
    /// Format version. Bump and migrate when shape changes; loader filters mismatches.
    schema_version: u32,
    /// SHA-256 of `format!("{system}\n{user_prompt}")` — stable across runs given
    /// deterministic retrieval. Same key = same context = same expected answer.
    key: String,
    /// Question text for human readability + per-category aggregation.
    question: String,
    /// Approach label, e.g. "structured_single-hop".
    approach: String,
    /// The Haiku answer.
    answer: String,
}

const PHASE3_SCHEMA_VERSION: u32 = 1;

/// Strict batch answer prompt — explicit conservative instructions to counter
/// the cognitive-load relaxation observed when batching multiple Q+context
/// pairs in one call. Directs the model to admit uncertainty rather than
/// hallucinate when the provided context lacks the answer.
fn strict_batch_answer_prompt(items: &[(usize, &str, &str)]) -> String {
    let mut s = String::with_capacity(2048 + items.len() * 1024);
    s.push_str(
        "You will answer multiple (context, question) pairs. Be CONSERVATIVE and DISCIPLINED.\n\n",
    );
    s.push_str("Rules:\n");
    s.push_str("- Use ONLY the context provided in each pair. Do not add external knowledge.\n");
    s.push_str("- Do not invent, paraphrase, or speculate beyond what the context states.\n");
    s.push_str("- If the context does not contain enough information to answer, output exactly: \"Information not available\".\n");
    s.push_str("- Be specific and concise. 1-3 sentences per answer.\n\n");
    s.push_str(&format!(
        "Return JSON object with a 'results' array containing exactly {} entries in input order, each with {{ idx, answer }}.\n\nPairs:\n",
        items.len()
    ));
    for (idx, question, context) in items {
        s.push_str(&format!(
            "[{}]\nContext:\n{}\n\nQuestion: {}\n\n",
            idx, context, question
        ));
    }
    s
}

/// Defensive parser for batch answer envelope. Tries `structured_output.results`
/// first; falls back to markdown-fence-stripped JSON in `.result` field if
/// `--json-schema` enforcement drops (which happens after several `--resume`
/// turns when the model treats the conversation as casual chat).
///
/// Returns Vec<(idx, answer)> or None.
fn parse_batch_answer_envelope(stdout: &str) -> Option<Vec<(usize, String)>> {
    use crate::eval::cli_batch::strip_markdown_fence;
    let trimmed = stdout.trim();
    let env: serde_json::Value = serde_json::from_str(trimmed).ok()?;

    let extract = |arr: &[serde_json::Value]| -> Vec<(usize, String)> {
        arr.iter()
            .filter_map(|v| {
                let idx = v.get("idx").and_then(|x| x.as_u64())? as usize;
                let answer = v
                    .get("answer")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                Some((idx, answer))
            })
            .collect()
    };

    if let Some(results) = env
        .get("structured_output")
        .and_then(|v| v.get("results"))
        .and_then(|v| v.as_array())
    {
        if !results.is_empty() {
            return Some(extract(results));
        }
    }

    if let Some(result_str) = env.get("result").and_then(|v| v.as_str()) {
        let stripped = strip_markdown_fence(result_str);
        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(&stripped) {
            if let Some(results) = inner.get("results").and_then(|v| v.as_array()) {
                if !results.is_empty() {
                    return Some(extract(results));
                }
            }
        }
    }

    None
}

/// Compute the cache key for a Phase 3 request.
fn phase3_cache_key(system_prompt: Option<&str>, user_prompt: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(system_prompt.unwrap_or("").as_bytes());
    hasher.update(b"\n");
    hasher.update(user_prompt.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Split a Phase 3 user prompt back into (context, question) for batched re-formatting.
/// Format from caller: `"Context:\n{ctx}\n\nQuestion: {question}"`.
fn split_context_question(user_prompt: &str) -> (String, String) {
    if let Some(stripped) = user_prompt.strip_prefix("Context:\n") {
        if let Some((ctx, question)) = stripped.split_once("\n\nQuestion: ") {
            return (ctx.to_string(), question.to_string());
        }
    }
    (String::new(), user_prompt.to_string())
}

/// Run Phase 3 with batched + persistent CLI subprocesses and cache-aware resume.
///
/// Sequential per `--resume` invariant (single-writer session). Each batch sends
/// up to `batch_size` (Q, context) pairs. Sessions rotate every `rotation_calls`
/// calls to avoid schema-drift after long --resume conversations. Cost is hard-
/// capped at `cost_cap_usd` (process aborts with error if exceeded).
///
/// JSONL cache lives at `cache_path` and is keyed by SHA-256 of
/// `system_prompt + "\n" + user_prompt` so identical (context, question, system)
/// triples reuse cached answers across runs.
#[allow(clippy::too_many_arguments)]
async fn run_phase3_batched_persistent(
    batch_requests: Vec<(String, String, Option<String>, usize)>,
    pending: &HashMap<String, PendingAnswer>,
    batch_size: usize,
    rotation_calls: usize,
    model: &str,
    cache_path: &Path,
    max_retries: u32,
    cost_cap_usd: f64,
) -> HashMap<String, String> {
    use crate::eval::cli_batch::run_cli_batch_subprocess;
    use std::collections::HashMap as StdMap;
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Write};

    eprintln!(
        "[phase3-batch] {} requests | batch_size={} rotation={} retries={} cost_cap=${:.2} | cache={}",
        batch_requests.len(),
        batch_size,
        rotation_calls,
        max_retries,
        cost_cap_usd,
        cache_path.display()
    );

    // Load existing cache (key -> answer).
    let mut cached: StdMap<String, String> = StdMap::new();
    let mut bad_lines = 0usize;
    if cache_path.exists() {
        if let Ok(f) = std::fs::File::open(cache_path) {
            for line in BufReader::new(f).lines().map_while(|l| l.ok()) {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Phase3AnswerRecord>(&line) {
                    Ok(rec) if rec.schema_version == PHASE3_SCHEMA_VERSION => {
                        cached.insert(rec.key, rec.answer);
                    }
                    Ok(rec) => {
                        eprintln!(
                            "[phase3-batch] WARN: skipping cache record with schema_version={} (expected {})",
                            rec.schema_version, PHASE3_SCHEMA_VERSION
                        );
                    }
                    Err(_) => bad_lines += 1,
                }
            }
        }
    }
    if bad_lines > 0 {
        eprintln!(
            "[phase3-batch] WARN: skipped {} corrupt JSONL lines in cache",
            bad_lines
        );
    }
    eprintln!("[phase3-batch] cache: {} existing entries", cached.len());

    let mut results: HashMap<String, String> = HashMap::new();

    // Partition: cache hits return immediately; misses go to CLI.
    let mut todo: Vec<(String, String, String, String, String, String, usize)> = Vec::new();
    // (req_id, key, question, context, approach, system_prompt, max_tokens)
    for (req_id, user_prompt, system, max_tokens) in batch_requests {
        let key = phase3_cache_key(system.as_deref(), &user_prompt);
        if let Some(answer) = cached.get(&key) {
            results.insert(req_id, answer.clone());
            continue;
        }
        let (context, question) = split_context_question(&user_prompt);
        let (approach_label, _gt, _cat, _ctx_tokens) = match pending.get(&req_id) {
            Some(p) => (
                p.approach.clone(),
                p.ground_truth.clone(),
                p.category.clone(),
                p.context_tokens,
            ),
            None => {
                eprintln!(
                    "[phase3-batch] WARN: req {req_id} has no pending metadata; using empty approach"
                );
                (String::new(), String::new(), String::new(), 0usize)
            }
        };
        todo.push((
            req_id,
            key,
            question,
            context,
            approach_label,
            system.unwrap_or_default(),
            max_tokens,
        ));
    }

    eprintln!(
        "[phase3-batch] cache hits: {} | to call: {}",
        results.len(),
        todo.len()
    );

    if todo.is_empty() {
        return results;
    }

    // Open cache file for append.
    if let Some(parent) = cache_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("[phase3-batch] WARN: cache dir create failed: {e}");
        }
    }
    let mut cache_file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(cache_path)
    {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("[phase3-batch] WARN: cache open failed: {e}; running without persistence");
            None
        }
    };

    let json_schema = r#"{"type":"object","properties":{"results":{"type":"array","items":{"type":"object","properties":{"idx":{"type":"integer"},"answer":{"type":"string"}},"required":["idx","answer"]}}},"required":["results"]}"#;

    let total_batches = todo.len().div_ceil(batch_size);
    let mut session_id: Option<String> = None;
    let mut calls_in_session = 0usize;
    let mut total_cost_usd = 0.0f64;
    let mut total_cc_tokens: u64 = 0;
    let mut total_cr_tokens: u64 = 0;
    let mut total_in_tokens: u64 = 0;
    let mut total_out_tokens: u64 = 0;
    let mut succ = 0usize;
    let mut fail_batches = 0usize;
    let mut retries = 0usize;
    let mut aborted = false;

    for (batch_i, chunk) in todo.chunks(batch_size).enumerate() {
        if total_cost_usd > cost_cap_usd {
            eprintln!(
                "[phase3-batch] ABORT: cumulative cost ${:.4} exceeds cost_cap=${:.2}",
                total_cost_usd, cost_cap_usd
            );
            aborted = true;
            break;
        }

        // Build prompt for this batch.
        let items: Vec<(usize, &str, &str)> = chunk
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.2.as_str(), t.3.as_str()))
            .collect();
        let prompt = strict_batch_answer_prompt(&items);

        // Rotate session if at cap.
        if calls_in_session >= rotation_calls {
            session_id = None;
            calls_in_session = 0;
        }

        let mut batch_result: Option<(Vec<(usize, String)>, _, _)> = None;
        let mut last_err: Option<String> = None;
        for attempt in 0..=max_retries {
            match run_cli_batch_subprocess(&prompt, model, json_schema, session_id.as_deref()).await
            {
                Ok((stdout, cost, sid)) => match parse_batch_answer_envelope(&stdout) {
                    Some(parsed) if parsed.len() == chunk.len() => {
                        batch_result = Some((parsed, cost, sid));
                        break;
                    }
                    Some(parsed) => {
                        last_err = Some(format!(
                            "expected {} answers, got {}",
                            chunk.len(),
                            parsed.len()
                        ));
                        if attempt < max_retries {
                            retries += 1;
                            session_id = None;
                            calls_in_session = 0;
                            tokio::time::sleep(std::time::Duration::from_millis(
                                500u64 * (1 << attempt),
                            ))
                            .await;
                        }
                    }
                    None => {
                        last_err = Some(format!(
                            "parse failed — head: {}",
                            stdout.chars().take(200).collect::<String>()
                        ));
                        if attempt < max_retries {
                            retries += 1;
                            session_id = None;
                            calls_in_session = 0;
                            tokio::time::sleep(std::time::Duration::from_millis(
                                500u64 * (1 << attempt),
                            ))
                            .await;
                        }
                    }
                },
                Err(e) => {
                    last_err = Some(e.to_string());
                    if attempt < max_retries {
                        retries += 1;
                        session_id = None;
                        calls_in_session = 0;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            500u64 * (1 << attempt),
                        ))
                        .await;
                    }
                }
            }
        }

        match batch_result {
            Some((parsed, cost, sid)) => {
                if let Some(c) = cost {
                    total_cost_usd += c.cost_usd;
                    total_cc_tokens += c.cache_creation_tokens;
                    total_cr_tokens += c.cache_read_tokens;
                    total_in_tokens += c.input_tokens;
                    total_out_tokens += c.output_tokens;
                }
                if session_id.is_none() {
                    session_id = sid;
                    calls_in_session = 1;
                } else {
                    calls_in_session += 1;
                }

                // Map idx → answer, then write to cache + results map.
                let by_idx: HashMap<usize, String> = parsed.into_iter().collect();
                for (i, t) in chunk.iter().enumerate() {
                    let answer = by_idx.get(&i).cloned().unwrap_or_default();
                    let (req_id, key, question, _ctx, approach, _sys, _max) = t;
                    if let Some(file) = cache_file.as_mut() {
                        let rec = Phase3AnswerRecord {
                            schema_version: PHASE3_SCHEMA_VERSION,
                            key: key.clone(),
                            question: question.clone(),
                            approach: approach.clone(),
                            answer: answer.clone(),
                        };
                        if let Ok(line) = serde_json::to_string(&rec) {
                            let _ = writeln!(file, "{}", line);
                            let _ = file.flush();
                        }
                    }
                    results.insert(req_id.clone(), answer);
                    succ += 1;
                }
                eprintln!(
                    "[phase3-batch] {}/{} ok | call_in_session={} | cost so far: ${:.4} | answers: {}",
                    batch_i + 1,
                    total_batches,
                    calls_in_session,
                    total_cost_usd,
                    succ
                );
            }
            None => {
                fail_batches += 1;
                eprintln!(
                    "[phase3-batch] {}/{} FAILED after {} retries: {}",
                    batch_i + 1,
                    total_batches,
                    max_retries,
                    last_err.unwrap_or_else(|| "?".into())
                );
                // Insert empty answers for this batch's req_ids so the caller's lookup works.
                for t in chunk.iter() {
                    results.insert(t.0.clone(), String::new());
                }
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
        "[phase3-batch] DONE: {} succ / {} target | {} batches failed | {} retries | aborted={}",
        succ,
        todo.len(),
        fail_batches,
        retries,
        aborted
    );
    eprintln!(
        "[phase3-batch] cost: ${:.4} total (${:.5}/call) | tokens: input={} output={} cache_create={} cache_read={}",
        total_cost_usd,
        mean_cost_per_call,
        total_in_tokens,
        total_out_tokens,
        total_cc_tokens,
        total_cr_tokens
    );
    if mean_cost_per_call > 0.05 {
        eprintln!(
            "[phase3-batch] WARN: mean cost ${:.5}/call is high — investigate cache hit rate or batch size",
            mean_cost_per_call
        );
    }

    results
}

/// Full-pipeline LoCoMo eval using Batch API for answer generation.
///
/// **Per-conversation DB**: each LoCoMo conversation gets its own isolated database so
/// enrichment (entity dedup, concept distillation, KG augmentation) cannot cross-pollinate
/// between scenarios. DBs are cached under `baselines_dir/fullpipeline/locomo/{sample_id}/`.
///
/// **Phase 1+2** (on-device, free): For each conversation, open or seed its DB (cached on
/// re-runs), then build contexts for all questions against that conversation's DB.
/// **Phase 3** (Batch API, 50% cheaper): Submit all answer prompts in one batch.
/// **Phase 4** (instant): Merge batch results + cached flat answers into tuples.
pub async fn run_fullpipeline_locomo_batch(
    locomo_path: &Path,
    enrichment: crate::eval::shared::EnrichmentMode,
    api_key: &str,
    answer_model: &str,
    output_path: &Path,
    cost_cap_usd: f64,
) -> Result<Vec<JudgmentTuple>, WenlanError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::eval::judge::save_judgment_tuples;
    use crate::eval::locomo::{category_name, extract_observations, load_locomo};

    let samples_all = load_locomo(locomo_path)?;

    // LOCOMO_LIMIT_CONVS=N: take only the first N conversations.  Useful for a quick
    // smoke-test before committing to a full run.
    let limit_convs: Option<usize> = std::env::var("LOCOMO_LIMIT_CONVS")
        .ok()
        .and_then(|s| s.parse().ok());
    let samples = if let Some(n) = limit_convs {
        eprintln!("[fullpipeline] LOCOMO_LIMIT_CONVS={n}: limiting to first {n} conversations");
        samples_all.into_iter().take(n).collect::<Vec<_>>()
    } else {
        samples_all
    };

    let shared_embedder = eval_shared_embedder();

    // Resume
    let mut finished_tuples: Vec<JudgmentTuple> = if output_path.exists() {
        let existing = crate::eval::judge::load_judgment_tuples(output_path)
            .map_err(|e| WenlanError::Generic(format!("load resume: {e}")))?;
        eprintln!(
            "[fullpipeline] Resuming with {} existing tuples",
            existing.len()
        );
        existing
    } else {
        Vec::new()
    };
    let done_questions: std::collections::HashSet<String> =
        finished_tuples.iter().map(|t| t.question.clone()).collect();

    // --- Phase 1+2 merged: per-conversation DB open (cached) + context build ---
    // output_path is e.g. baselines/fullpipeline_locomo_tuples.json; its parent is baselines/.
    let baselines_dir = output_path
        .parent()
        .ok_or_else(|| WenlanError::Generic("output_path has no parent".to_string()))?;

    let scenario_concurrency: usize = std::env::var("EVAL_SCENARIO_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .min(8);
    eprintln!("[fullpipeline] scenario concurrency = {scenario_concurrency}");

    let mut pending: HashMap<String, PendingAnswer> = HashMap::new();
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();

    // Each scenario yields a Vec of (req_id, prompt, system, max_tokens, PendingAnswer).
    type ScenarioOutput = Vec<(String, String, Option<String>, usize, String, PendingAnswer)>;

    if scenario_concurrency <= 1 {
        // Serial path — preserves existing behavior exactly.
        for sample in &samples {
            let memories = extract_observations(sample);
            if memories.is_empty() {
                continue;
            }

            let scope_dir =
                crate::eval::shared::scenario_db_dir(baselines_dir, "locomo", &sample.sample_id);

            let sample_id = sample.sample_id.clone();
            let memories_owned = memories.clone();
            let db = match crate::eval::shared::open_or_seed_scenario_db(
                &scope_dir,
                shared_embedder.clone(),
                move || {
                    memories_owned
                        .iter()
                        .enumerate()
                        .map(|(i, mem)| RawDocument {
                            content: mem.content.clone(),
                            source_id: format!("locomo_{}_obs_{}", sample_id, i),
                            source: "memory".to_string(),
                            title: format!("{} session {}", mem.speaker, mem.session_num),
                            memory_type: Some("fact".to_string()),
                            space: Some("conversation".to_string()),
                            last_modified: chrono::Utc::now().timestamp(),
                            ..Default::default()
                        })
                        .collect()
                },
                &enrichment,
            )
            .await
            {
                Ok(db) => db,
                Err(e) => {
                    eprintln!(
                        "[fullpipeline] WARN: scenario {} failed to open/seed DB: {}. Skipping.",
                        sample.sample_id, e
                    );
                    continue;
                }
            };

            let db_mem_count = db.memory_count().await.unwrap_or(0);
            let db_enriched = db.enriched_memory_count().await.unwrap_or(0);
            eprintln!(
                "[fullpipeline] Conv {}: {} obs, {}/{} enriched",
                sample.sample_id,
                memories.len(),
                db_enriched,
                db_mem_count,
            );

            let mut q_count = 0usize;

            for qa in &sample.qa {
                if qa.category == 5 {
                    continue;
                }
                if done_questions.contains(&qa.question) {
                    continue;
                }

                let ground_truth = qa
                    .answer
                    .as_ref()
                    .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                    .unwrap_or_default();
                if ground_truth.is_empty() {
                    continue;
                }

                let category = category_name(qa.category);
                let ctx_result = build_structured_context(
                    &db,
                    &qa.question,
                    10,
                    None,
                    CtxRetrieval::Quick,
                    true,
                )
                .await;
                let (ctx, ctx_tokens) = match ctx_result {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "[fullpipeline] WARN: scenario {} question context failed: {}. Skipping question.",
                            sample.sample_id, e
                        );
                        continue;
                    }
                };

                let req_id = format!("q_{}_{}", sample.sample_id, q_count);
                batch_requests.push((
                    req_id.clone(),
                    format!("Context:\n{}\n\nQuestion: {}", ctx, qa.question),
                    Some(e2e_system_prompt()),
                    e2e_max_answer_tokens(),
                ));
                pending.insert(
                    req_id,
                    PendingAnswer {
                        question: qa.question.clone(),
                        ground_truth,
                        approach: format!("structured_{}", category),
                        category: category.to_string(),
                        context_tokens: ctx_tokens,
                    },
                );

                q_count += 1;
            }
            if q_count > 0 {
                eprintln!("  {} — {} questions collected", sample.sample_id, q_count);
            }
        }
    } else {
        // Concurrent path — overlaps CPU/DB/IO across scenarios.
        // Phase 3 (batch API) stays serial after this block.
        use futures::StreamExt;

        let enrichment_arc = Arc::new(enrichment);
        let done_arc = Arc::new(done_questions.clone());
        let baselines_dir_owned = baselines_dir.to_path_buf();
        // Wrap samples in Arc so each future can index without cloning the whole vec.
        let samples_arc = Arc::new(samples);

        let scenario_outputs: Vec<Result<ScenarioOutput, WenlanError>> =
            futures::stream::iter(0..samples_arc.len())
                .map(|s_idx| {
                    let shared_embedder = shared_embedder.clone();
                    let enrichment_arc = enrichment_arc.clone();
                    let done_arc = done_arc.clone();
                    let baselines_dir_owned = baselines_dir_owned.clone();
                    let samples_arc = samples_arc.clone();
                    async move {
                        let sample = &samples_arc[s_idx];
                        let memories = extract_observations(sample);
                        if memories.is_empty() {
                            return Ok(Vec::new());
                        }

                        let scope_dir = crate::eval::shared::scenario_db_dir(
                            &baselines_dir_owned,
                            "locomo",
                            &sample.sample_id,
                        );

                        let sample_id = sample.sample_id.clone();
                        let memories_owned = memories.clone();
                        let db = match crate::eval::shared::open_or_seed_scenario_db(
                            &scope_dir,
                            shared_embedder,
                            move || {
                                memories_owned
                                    .iter()
                                    .enumerate()
                                    .map(|(i, mem)| RawDocument {
                                        content: mem.content.clone(),
                                        source_id: format!("locomo_{}_obs_{}", sample_id, i),
                                        source: "memory".to_string(),
                                        title: format!(
                                            "{} session {}",
                                            mem.speaker, mem.session_num
                                        ),
                                        memory_type: Some("fact".to_string()),
                                        space: Some("conversation".to_string()),
                                        last_modified: chrono::Utc::now().timestamp(),
                                        ..Default::default()
                                    })
                                    .collect()
                            },
                            &enrichment_arc,
                        )
                        .await
                        {
                            Ok(db) => db,
                            Err(e) => {
                                eprintln!(
                                    "[fullpipeline] WARN: scenario {} failed to open/seed DB: {}. Skipping.",
                                    sample.sample_id, e
                                );
                                return Ok(Vec::new());
                            }
                        };

                        let db_mem_count = db.memory_count().await.unwrap_or(0);
                        let db_enriched = db.enriched_memory_count().await.unwrap_or(0);
                        eprintln!(
                            "[fullpipeline] Conv {}: {} obs, {}/{} enriched",
                            sample.sample_id,
                            memories.len(),
                            db_enriched,
                            db_mem_count,
                        );

                        let mut entries: ScenarioOutput = Vec::new();
                        let mut q_count = 0usize;

                        for qa in &sample.qa {
                            if qa.category == 5 {
                                continue;
                            }
                            if done_arc.contains(&qa.question) {
                                continue;
                            }

                            let ground_truth = qa
                                .answer
                                .as_ref()
                                .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                                .unwrap_or_default();
                            if ground_truth.is_empty() {
                                continue;
                            }

                            let category = category_name(qa.category);
                            let ctx_result = build_structured_context(
                                &db,
                                &qa.question,
                                10,
                                None,
                                CtxRetrieval::Quick,
                                true,
                            )
                            .await;
                            let (ctx, ctx_tokens) = match ctx_result {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!(
                                        "[fullpipeline] WARN: scenario {} question context failed: {}. Skipping question.",
                                        sample.sample_id, e
                                    );
                                    continue;
                                }
                            };

                            let req_id = format!("q_{}_{}", sample.sample_id, q_count);
                            entries.push((
                                req_id,
                                format!("Context:\n{}\n\nQuestion: {}", ctx, qa.question),
                                Some(e2e_system_prompt()),
                                e2e_max_answer_tokens(),
                                sample.sample_id.clone(),
                                PendingAnswer {
                                    question: qa.question.clone(),
                                    ground_truth,
                                    approach: format!("structured_{}", category),
                                    category: category.to_string(),
                                    context_tokens: ctx_tokens,
                                },
                            ));
                            q_count += 1;
                        }
                        if q_count > 0 {
                            eprintln!("  {} — {} questions collected", sample.sample_id, q_count);
                        }
                        Ok(entries)
                    }
                })
                .buffer_unordered(scenario_concurrency)
                .collect()
                .await;

        for result in scenario_outputs {
            let entries = match result {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "[fullpipeline] WARN: scenario stream error: {}. Skipping.",
                        e
                    );
                    continue;
                }
            };
            for (req_id, prompt, system, max_tok, _sid, meta) in entries {
                batch_requests.push((req_id.clone(), prompt, system, max_tok));
                pending.insert(req_id, meta);
            }
        }
    }

    if batch_requests.is_empty() {
        eprintln!("[fullpipeline] No new requests — all cached/resumed");
        save_judgment_tuples(&finished_tuples, output_path)
            .map_err(|e| WenlanError::Generic(format!("save: {e}")))?;
        return Ok(finished_tuples);
    }

    // --- Phase 3: Answer generation (Batch API or CLI subprocess) ---
    let use_cli = std::env::var("EVAL_PHASE3_CLI").as_deref() == Ok("1");
    let cli_concurrency: usize = std::env::var("EVAL_PHASE3_CLI_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let phase3_batch_size: usize = std::env::var("EVAL_PHASE3_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let phase3_rotation: usize = std::env::var("EVAL_PHASE3_ROTATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .max(1);
    let phase3_retries: u32 = std::env::var("EVAL_PHASE3_RETRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let phase3_cost_cap: f64 = std::env::var("EVAL_PHASE3_COST_CAP_USD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let raw_results: HashMap<String, String> = if use_cli && phase3_batch_size >= 2 {
        eprintln!(
            "\n[fullpipeline] EVAL_PHASE3_CLI=1, batch_size={}: running {} requests via batched claude -p",
            phase3_batch_size,
            batch_requests.len()
        );
        let cache_path = baselines_dir.join("fullpipeline_locomo_phase3_answers_batch.jsonl");
        run_phase3_batched_persistent(
            batch_requests,
            &pending,
            phase3_batch_size,
            phase3_rotation,
            "haiku",
            &cache_path,
            phase3_retries,
            phase3_cost_cap,
        )
        .await
    } else if use_cli {
        eprintln!(
            "\n[fullpipeline] EVAL_PHASE3_CLI=1: running {} requests via claude -p (per-question, concurrency={})",
            batch_requests.len(),
            cli_concurrency
        );
        run_phase3_via_cli(batch_requests, cli_concurrency).await
    } else {
        eprintln!(
            "\n[fullpipeline] Submitting {} requests via Batch API (model={})",
            batch_requests.len(),
            answer_model
        );

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| WenlanError::Generic(format!("client: {e}")))?;

        let batch_id = submit_batch(&client, api_key, batch_requests, answer_model, cost_cap_usd)
            .await
            .map_err(|e| WenlanError::Generic(format!("batch submit: {e}")))?;
        eprintln!("[fullpipeline] Batch submitted: {}", batch_id);

        let results_url = poll_batch(&client, api_key, &batch_id)
            .await
            .map_err(|e| WenlanError::Generic(format!("batch poll: {e}")))?;

        download_batch_results(&client, api_key, &results_url)
            .await
            .map_err(|e| WenlanError::Generic(format!("batch download: {e}")))?
    };

    // --- Phase 4: Merge ---
    // Incremental save: persist after every 10 tuples so a mid-phase crash loses minimal work.
    let mut matched = 0usize;
    for (custom_id, answer) in &raw_results {
        if let Some(meta) = pending.get(custom_id) {
            finished_tuples.push(JudgmentTuple {
                question: meta.question.clone(),
                ground_truth: meta.ground_truth.clone(),
                approach: meta.approach.clone(),
                answer: answer.clone(),
                context_tokens: meta.context_tokens,
                category: meta.category.clone(),
                question_id: String::new(),
            });
            matched += 1;
            if matched.is_multiple_of(10) {
                if let Err(e) = save_judgment_tuples(&finished_tuples, output_path) {
                    eprintln!("[fullpipeline] WARN: incremental save failed: {e}");
                }
            }
        }
    }

    eprintln!(
        "[fullpipeline] Batch: {} results, {} matched",
        raw_results.len(),
        matched
    );

    save_judgment_tuples(&finished_tuples, output_path)
        .map_err(|e| WenlanError::Generic(format!("save: {e}")))?;
    eprintln!(
        "[fullpipeline] Saved {} total tuples to {:?}",
        finished_tuples.len(),
        output_path
    );

    Ok(finished_tuples)
}

/// Full-pipeline LongMemEval eval using Batch API for answer generation.
///
/// **Per-question DB**: each LME question gets its own isolated database so enrichment
/// (entity dedup, concept distillation, KG augmentation) cannot cross-pollinate between
/// scenarios. DBs are cached under `baselines_dir/fullpipeline/lme/{question_id}/`.
///
/// **Phase 1+2** (on-device, free): For each question, open or seed its DB (cached on
/// re-runs), then build context for that question against its own DB.
/// **Phase 3** (Batch API, 50% cheaper): Submit all answer prompts in one batch.
/// **Phase 4** (instant): Merge batch results + cached flat answers into tuples.
pub async fn run_fullpipeline_lme_batch(
    longmemeval_path: &Path,
    enrichment: crate::eval::shared::EnrichmentMode,
    api_key: &str,
    answer_model: &str,
    output_path: &Path,
    cost_cap_usd: f64,
) -> Result<Vec<JudgmentTuple>, WenlanError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::eval::judge::save_judgment_tuples;
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};

    let mut samples = load_longmemeval(longmemeval_path)?;
    // Optional limit for small test runs (set LME_LIMIT_QUESTIONS=N).
    let limit_active = std::env::var("LME_LIMIT_QUESTIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    if let Some(n) = limit_active {
        let total_before = samples.len();
        samples.truncate(n);
        eprintln!(
            "[fullpipeline_lme] LME_LIMIT_QUESTIONS={n} -> processing {}/{}",
            samples.len(),
            total_before
        );
    }
    let shared_embedder = eval_shared_embedder();

    // Resume
    let mut finished_tuples: Vec<JudgmentTuple> = if output_path.exists() {
        let existing = crate::eval::judge::load_judgment_tuples(output_path)
            .map_err(|e| WenlanError::Generic(format!("load resume: {e}")))?;
        eprintln!(
            "[fullpipeline_lme] Resuming with {} existing tuples",
            existing.len()
        );
        existing
    } else {
        Vec::new()
    };
    // When LME_LIMIT_QUESTIONS is active, restrict resume tuples to questions in the
    // truncated sample window so the returned vec matches the limit (avoids leaking
    // tuples from prior unbounded runs back to caller).
    if limit_active.is_some() {
        let allowed: std::collections::HashSet<String> =
            samples.iter().map(|s| s.question.clone()).collect();
        let before = finished_tuples.len();
        finished_tuples.retain(|t| allowed.contains(&t.question));
        if before != finished_tuples.len() {
            eprintln!(
                "[fullpipeline_lme] resume filter: kept {}/{} tuples within limit window",
                finished_tuples.len(),
                before
            );
        }
    }
    let done_questions: std::collections::HashSet<String> =
        finished_tuples.iter().map(|t| t.question.clone()).collect();

    // --- Phase 1+2 merged: per-question DB open (cached) + context build ---
    // output_path is e.g. baselines/fullpipeline_lme_tuples.json; its parent is baselines/.
    let baselines_dir = output_path
        .parent()
        .ok_or_else(|| WenlanError::Generic("output_path has no parent".to_string()))?;

    let scenario_concurrency: usize = std::env::var("EVAL_SCENARIO_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .min(8);
    eprintln!("[fullpipeline_lme] scenario concurrency = {scenario_concurrency}");

    let mut pending: HashMap<String, PendingAnswer> = HashMap::new();
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();

    // Each scenario yields a Vec of (req_id, prompt, system, max_tokens, PendingAnswer).
    type LmeScenarioOutput = Vec<(String, String, Option<String>, usize, PendingAnswer)>;

    if scenario_concurrency <= 1 {
        // Serial path — preserves existing behavior exactly.
        for (q_idx, sample) in samples.iter().enumerate() {
            if done_questions.contains(&sample.question) {
                continue;
            }

            let ground_truth = sample
                .answer
                .as_str()
                .unwrap_or(&sample.answer.to_string())
                .to_string();
            if ground_truth.is_empty() {
                continue;
            }

            let memories = extract_memories(sample);
            if memories.is_empty() {
                continue;
            }

            let scope_dir =
                crate::eval::shared::scenario_db_dir(baselines_dir, "lme", &sample.question_id);

            let question_id = sample.question_id.clone();
            let question_type = sample.question_type.clone();
            let memories_owned = memories.clone();
            let db = match crate::eval::shared::open_or_seed_scenario_db(
                &scope_dir,
                shared_embedder.clone(),
                move || {
                    memories_owned
                        .iter()
                        .map(|mem| RawDocument {
                            content: mem.content.clone(),
                            source_id: format!(
                                "lme_{}_{}_t{}",
                                question_id, mem.session_idx, mem.turn_idx
                            ),
                            source: "memory".to_string(),
                            title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                            memory_type: Some(
                                if question_type == "single-session-preference" {
                                    "preference"
                                } else {
                                    "fact"
                                }
                                .to_string(),
                            ),
                            space: Some("conversation".to_string()),
                            last_modified: chrono::Utc::now().timestamp(),
                            ..Default::default()
                        })
                        .collect()
                },
                &enrichment,
            )
            .await
            {
                Ok(db) => db,
                Err(e) => {
                    eprintln!(
                        "[fullpipeline_lme] WARN: scenario {} failed to open/seed DB: {}. Skipping.",
                        sample.question_id, e
                    );
                    continue;
                }
            };

            let db_mem_count = db.memory_count().await.unwrap_or(0);
            let db_enriched = db.enriched_memory_count().await.unwrap_or(0);
            eprintln!(
                "[fullpipeline_lme] Q {}: {} memories, {}/{} enriched",
                sample.question_id,
                memories.len(),
                db_enriched,
                db_mem_count,
            );

            let category = category_name(&sample.question_type);
            let ctx_result = build_structured_context(
                &db,
                &sample.question,
                10,
                None,
                CtxRetrieval::Quick,
                true,
            )
            .await;
            let (ctx, ctx_tokens) = match ctx_result {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "[fullpipeline_lme] WARN: scenario {} context build failed: {}. Skipping.",
                        sample.question_id, e
                    );
                    continue;
                }
            };

            let req_id = format!("q_lme_{}", q_idx);
            batch_requests.push((
                req_id.clone(),
                format!("Context:\n{}\n\nQuestion: {}", ctx, sample.question),
                Some(e2e_system_prompt()),
                e2e_max_answer_tokens(),
            ));
            pending.insert(
                req_id,
                PendingAnswer {
                    question: sample.question.clone(),
                    ground_truth,
                    approach: format!("structured_{}", category),
                    category: category.to_string(),
                    context_tokens: ctx_tokens,
                },
            );

            if q_idx % 100 == 99 {
                eprintln!(
                    "  [contexts] {}/{} questions collected",
                    q_idx + 1,
                    samples.len()
                );
            }
        }
    } else {
        // Concurrent path — overlaps CPU/DB/IO across scenarios.
        // Phase 3 (batch API) stays serial after this block.
        use futures::StreamExt;

        let enrichment_arc = Arc::new(enrichment);
        let done_arc = Arc::new(done_questions.clone());
        let baselines_dir_owned = baselines_dir.to_path_buf();
        let total = samples.len();
        // Wrap samples in Arc so each future can index without cloning the whole vec.
        let samples_arc = Arc::new(samples);

        let scenario_outputs: Vec<Result<LmeScenarioOutput, WenlanError>> =
            futures::stream::iter(0..total)
                .map(|q_idx| {
                    let shared_embedder = shared_embedder.clone();
                    let enrichment_arc = enrichment_arc.clone();
                    let done_arc = done_arc.clone();
                    let baselines_dir_owned = baselines_dir_owned.clone();
                    let samples_arc = samples_arc.clone();
                    async move {
                        let sample = &samples_arc[q_idx];
                        if done_arc.contains(&sample.question) {
                            return Ok(Vec::new());
                        }

                        let ground_truth = sample
                            .answer
                            .as_str()
                            .unwrap_or(&sample.answer.to_string())
                            .to_string();
                        if ground_truth.is_empty() {
                            return Ok(Vec::new());
                        }

                        let memories = extract_memories(sample);
                        if memories.is_empty() {
                            return Ok(Vec::new());
                        }

                        let scope_dir = crate::eval::shared::scenario_db_dir(
                            &baselines_dir_owned,
                            "lme",
                            &sample.question_id,
                        );

                        let question_id = sample.question_id.clone();
                        let question_type = sample.question_type.clone();
                        let memories_owned = memories.clone();
                        let db = match crate::eval::shared::open_or_seed_scenario_db(
                            &scope_dir,
                            shared_embedder,
                            move || {
                                memories_owned
                                    .iter()
                                    .map(|mem| RawDocument {
                                        content: mem.content.clone(),
                                        source_id: format!(
                                            "lme_{}_{}_t{}",
                                            question_id, mem.session_idx, mem.turn_idx
                                        ),
                                        source: "memory".to_string(),
                                        title: format!(
                                            "session {} turn {}",
                                            mem.session_idx, mem.turn_idx
                                        ),
                                        memory_type: Some(
                                            if question_type == "single-session-preference" {
                                                "preference"
                                            } else {
                                                "fact"
                                            }
                                            .to_string(),
                                        ),
                                        space: Some("conversation".to_string()),
                                        last_modified: chrono::Utc::now().timestamp(),
                                        ..Default::default()
                                    })
                                    .collect()
                            },
                            &enrichment_arc,
                        )
                        .await
                        {
                            Ok(db) => db,
                            Err(e) => {
                                eprintln!(
                                    "[fullpipeline_lme] WARN: scenario {} failed to open/seed DB: {}. Skipping.",
                                    sample.question_id, e
                                );
                                return Ok(Vec::new());
                            }
                        };

                        let db_mem_count = db.memory_count().await.unwrap_or(0);
                        let db_enriched = db.enriched_memory_count().await.unwrap_or(0);
                        eprintln!(
                            "[fullpipeline_lme] Q {}: {} memories, {}/{} enriched",
                            sample.question_id,
                            memories.len(),
                            db_enriched,
                            db_mem_count,
                        );

                        let category = category_name(&sample.question_type);
                        let ctx_result = build_structured_context(
                            &db,
                            &sample.question,
                            10,
                            None,
                            CtxRetrieval::Quick,
                            true,
                        )
                        .await;
                        let (ctx, ctx_tokens) = match ctx_result {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!(
                                    "[fullpipeline_lme] WARN: scenario {} context build failed: {}. Skipping.",
                                    sample.question_id, e
                                );
                                return Ok(Vec::new());
                            }
                        };

                        let req_id = format!("q_lme_{}", q_idx);
                        if q_idx % 100 == 99 {
                            eprintln!("  [contexts] {}/{} questions collected", q_idx + 1, total);
                        }
                        Ok(vec![(
                            req_id,
                            format!("Context:\n{}\n\nQuestion: {}", ctx, sample.question),
                            Some(e2e_system_prompt()),
                            e2e_max_answer_tokens(),
                            PendingAnswer {
                                question: sample.question.clone(),
                                ground_truth,
                                approach: format!("structured_{}", category),
                                category: category.to_string(),
                                context_tokens: ctx_tokens,
                            },
                        )])
                    }
                })
                .buffer_unordered(scenario_concurrency)
                .collect()
                .await;

        for result in scenario_outputs {
            let entries = match result {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "[fullpipeline_lme] WARN: scenario stream error: {}. Skipping.",
                        e
                    );
                    continue;
                }
            };
            for (req_id, prompt, system, max_tok, meta) in entries {
                batch_requests.push((req_id.clone(), prompt, system, max_tok));
                pending.insert(req_id, meta);
            }
        }
    }

    if batch_requests.is_empty() {
        eprintln!("[fullpipeline_lme] No new requests — all cached/resumed");
        save_judgment_tuples(&finished_tuples, output_path)
            .map_err(|e| WenlanError::Generic(format!("save: {e}")))?;
        return Ok(finished_tuples);
    }

    // --- Phase 3: Answer generation (Batch API or CLI subprocess) ---
    let use_cli = std::env::var("EVAL_PHASE3_CLI").as_deref() == Ok("1");
    let cli_concurrency: usize = std::env::var("EVAL_PHASE3_CLI_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let phase3_batch_size: usize = std::env::var("EVAL_PHASE3_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let phase3_rotation: usize = std::env::var("EVAL_PHASE3_ROTATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .max(1);
    let phase3_retries: u32 = std::env::var("EVAL_PHASE3_RETRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let phase3_cost_cap: f64 = std::env::var("EVAL_PHASE3_COST_CAP_USD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);

    let raw_results: HashMap<String, String> = if use_cli && phase3_batch_size >= 2 {
        eprintln!(
            "\n[fullpipeline_lme] EVAL_PHASE3_CLI=1, batch_size={}: running {} requests via batched claude -p",
            phase3_batch_size,
            batch_requests.len()
        );
        let cache_path = baselines_dir.join("fullpipeline_lme_phase3_answers_batch.jsonl");
        run_phase3_batched_persistent(
            batch_requests,
            &pending,
            phase3_batch_size,
            phase3_rotation,
            "haiku",
            &cache_path,
            phase3_retries,
            phase3_cost_cap,
        )
        .await
    } else if use_cli {
        eprintln!(
            "\n[fullpipeline_lme] EVAL_PHASE3_CLI=1: running {} requests via claude -p (per-question, concurrency={})",
            batch_requests.len(),
            cli_concurrency
        );
        run_phase3_via_cli(batch_requests, cli_concurrency).await
    } else {
        eprintln!(
            "\n[fullpipeline_lme] Submitting {} requests via Batch API (model={})",
            batch_requests.len(),
            answer_model
        );

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| WenlanError::Generic(format!("client: {e}")))?;

        let batch_id = submit_batch(&client, api_key, batch_requests, answer_model, cost_cap_usd)
            .await
            .map_err(|e| WenlanError::Generic(format!("batch submit: {e}")))?;
        eprintln!("[fullpipeline_lme] Batch submitted: {}", batch_id);

        let results_url = poll_batch(&client, api_key, &batch_id)
            .await
            .map_err(|e| WenlanError::Generic(format!("batch poll: {e}")))?;

        download_batch_results(&client, api_key, &results_url)
            .await
            .map_err(|e| WenlanError::Generic(format!("batch download: {e}")))?
    };

    // --- Phase 4: Merge ---
    // Incremental save: persist after every 10 tuples so a mid-phase crash loses minimal work.
    let mut matched = 0usize;
    for (custom_id, answer) in &raw_results {
        if let Some(meta) = pending.get(custom_id) {
            finished_tuples.push(JudgmentTuple {
                question: meta.question.clone(),
                ground_truth: meta.ground_truth.clone(),
                approach: meta.approach.clone(),
                answer: answer.clone(),
                context_tokens: meta.context_tokens,
                category: meta.category.clone(),
                question_id: String::new(),
            });
            matched += 1;
            if matched.is_multiple_of(10) {
                if let Err(e) = save_judgment_tuples(&finished_tuples, output_path) {
                    eprintln!("[fullpipeline_lme] WARN: incremental save failed: {e}");
                }
            }
        }
    }

    eprintln!(
        "[fullpipeline_lme] Batch: {} results, {} matched",
        raw_results.len(),
        matched
    );

    save_judgment_tuples(&finished_tuples, output_path)
        .map_err(|e| WenlanError::Generic(format!("save: {e}")))?;
    eprintln!(
        "[fullpipeline_lme] Saved {} total tuples to {:?}",
        finished_tuples.len(),
        output_path
    );

    Ok(finished_tuples)
}

// ===== Fair CE A/B answer runner =====

/// One arm of an LME answer A/B. Maps to a labeled retrieval mode per category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmKind {
    /// CE off: CrossRerank(None).
    CeOff,
    /// CE on: CrossRerank(Some(reranker)).
    CeOn,
    /// Full production stack, single arm: CrossRerank(Some(reranker)) with all
    /// feature env flags left ON by the operator (page channel / graph stream /
    /// temporal / deep `RERANK_POOL_FLOOR`). Used by the "best-possible ceiling"
    /// run to measure how good Wenlan can answer with everything turned on, NOT
    /// for an isolated A/B. Resolves to the same retrieval as `CeOn`; the
    /// difference is the surrounding env flags + that no confounder isolation is
    /// enforced (there is no OFF arm to perturb asymmetrically). Note: on this
    /// CE path the live graph stream is structurally suppressed
    /// (`allow_graph_stream = reranker.is_none()`, db.rs), so graph contributes
    /// via entity-linked memories already in the pool, not the live stream.
    FullStack,
    // PR-3 will add CeOnGraph -> CrossRerankGraph. Do NOT add it now.
}

/// One arm: a retrieval mode paired with whether to include pages in context.
/// `label_prefix` is the stable identity used in baseline filenames — assigned
/// explicitly by each constructor, independent of `include_pages`.
#[derive(Debug, Clone)]
pub struct Arm {
    pub kind: ArmKind,
    pub include_pages: bool,
    /// Stable baseline label prefix, e.g. "ce_off", "ce_on", "ce_on_pages".
    /// Assigned by the constructor, NOT derived from `include_pages`.
    pub label_prefix: &'static str,
}

/// The set of arms to run, in order.
#[derive(Debug, Clone)]
pub struct ArmSpec {
    arms: Vec<Arm>,
}

impl ArmSpec {
    /// The shipped 2-arm CE A/B: [CeOff+pages, CeOn+pages].
    pub fn ce_two() -> Self {
        ArmSpec {
            arms: vec![
                Arm {
                    kind: ArmKind::CeOff,
                    include_pages: true,
                    label_prefix: "ce_off",
                },
                Arm {
                    kind: ArmKind::CeOn,
                    include_pages: true,
                    label_prefix: "ce_on",
                },
            ],
        }
    }

    /// Single-arm "best-possible ceiling": [FullStack+pages]. All feature env flags are
    /// the operator's responsibility (page channel / graph / temporal / deep
    /// `RERANK_POOL_FLOOR`); this spec just runs one CE-reranked arm and measures
    /// its accuracy, with NO confounder isolation (see `needs_confounder_isolation`).
    pub fn full_stack() -> Self {
        ArmSpec {
            arms: vec![Arm {
                kind: ArmKind::FullStack,
                include_pages: true,
                label_prefix: "fullstack",
            }],
        }
    }

    /// 3-arm pages A/B: [CeOff-no-pages, CeOn-no-pages, CeOn+pages].
    /// Isolates the page-channel contribution on top of the CE baseline.
    pub fn pages_ab() -> Self {
        ArmSpec {
            arms: vec![
                Arm {
                    kind: ArmKind::CeOff,
                    include_pages: false,
                    label_prefix: "ce_off",
                },
                Arm {
                    kind: ArmKind::CeOn,
                    include_pages: false,
                    label_prefix: "ce_on",
                },
                Arm {
                    kind: ArmKind::CeOn,
                    include_pages: true,
                    label_prefix: "ce_on_pages",
                },
            ],
        }
    }

    /// Whether this spec is a CE isolation A/B that must run with the graph
    /// stream / skip-pref / temporal flags OFF. True iff a `CeOff` arm is present:
    /// the OFF arm (reranker=None) is the one those flags would perturb
    /// asymmetrically vs an ON arm. A single-arm ceiling (`full_stack`) has no
    /// OFF arm, so the isolation gate does not apply — features are meant to be ON.
    fn needs_confounder_isolation(&self) -> bool {
        self.arms.iter().any(|a| a.kind == ArmKind::CeOff)
    }

    /// One label per arm for a given category, in order.
    /// Label is `<arm.label_prefix>_<category>`; the prefix is assigned explicitly per
    /// constructor (e.g. "ce_off_cat", "ce_on_pages_cat"), NOT derived from `include_pages`.
    pub fn labels(&self, category: &str) -> Vec<String> {
        self.arms
            .iter()
            .map(|arm| format!("{}_{}", arm.label_prefix, category))
            .collect()
    }

    /// Label prefixes (for the done-set), e.g. `["ce_off", "ce_on"]`.
    fn prefixes(&self) -> Vec<&'static str> {
        self.arms.iter().map(|arm| arm.label_prefix).collect()
    }

    /// Resolve arms to `(label, CtxRetrieval, include_pages)` for a category, given the reranker.
    /// `CeOn` requires `reranker.is_some()` — returns `Err` if an arm needs it and
    /// it is `None`.
    fn resolve(
        &self,
        category: &str,
        reranker: &Option<Arc<dyn crate::reranker::Reranker>>,
    ) -> Result<Vec<(String, CtxRetrieval, bool)>, WenlanError> {
        let mut result = Vec::with_capacity(self.arms.len());
        for arm in &self.arms {
            let label = format!("{}_{}", arm.label_prefix, category);
            let retrieval = match arm.kind {
                ArmKind::CeOff => CtxRetrieval::CrossRerank(None),
                ArmKind::CeOn | ArmKind::FullStack => {
                    let r = reranker.clone().ok_or_else(|| {
                        WenlanError::Generic(format!(
                            "ArmSpec::resolve: arm {} requires a reranker but none was provided",
                            label
                        ))
                    })?;
                    CtxRetrieval::CrossRerank(Some(r))
                }
            };
            result.push((label, retrieval, arm.include_pages));
        }
        Ok(result)
    }
}

/// Where the per-question retrieval pool comes from.
///
/// The cross-encoder is a *selection* stage: it can only change which memories
/// reach the answerer when the candidate pool it reranks is larger than the
/// returned top-k window. Two settings expose very different pools:
///
/// - [`DbSource::SeedPerQuestion`] — seed a fresh scenario DB from the
///   question's own sessions (the LME *oracle* setting). The total pool is just
///   that question's gold turns (N ~ k), so a top-10 rerank reorders an
///   already-complete context and rarely changes membership. This is the
///   default and matches the original `run_fullpipeline_lme_ce_ab` behavior.
/// - [`DbSource::Consolidated`] — reuse ONE pre-seeded consolidated DB (every
///   question's memories in a single store, N >> k). Retrieval must select
///   k-from-many, so the CE has real reselection headroom. The DB is opened
///   ONCE and shared read-only across all questions; per-question seeding and
///   `event_date` injection are skipped (the consolidated DB is already
///   enriched + dated). Pair with `RERANK_POOL_FLOOR` > 10 so the CE fetch pool
///   exceeds the top-10 window (see `compute_rerank_fetch_pool`); with the
///   default floor of 10 the CE reranks exactly the 10 docs it returns and
///   cannot change membership regardless of the consolidated pool size.
#[derive(Debug, Clone)]
pub enum DbSource {
    /// Seed a fresh per-question scenario DB (oracle setting, small pool N~k).
    SeedPerQuestion,
    /// Reuse one pre-seeded consolidated DB for every question (full-corpus
    /// setting, large pool N>>k). The path is the DB *directory* (the one that
    /// contains `origin_memory.db`). Opened once, never mutated.
    Consolidated(std::path::PathBuf),
}

/// Provider-generic LME full-pipeline A/B runner.
///
/// Replaces the hardcoded `run_fullpipeline_lme_ce_ab` with an extensible
/// `ArmSpec`-driven variant. Pass `ArmSpec::ce_two()` to reproduce the
/// existing CE A/B behavior exactly.
///
/// `db_source` selects the candidate pool (see [`DbSource`]):
/// [`DbSource::SeedPerQuestion`] reproduces the original oracle behavior;
/// [`DbSource::Consolidated`] runs every question against one shared large-pool
/// DB so the cross-encoder has real reselection headroom.
///
/// **FAIR design**: all arms route through `search_memory_cross_rerank` so
/// P3 distill-demotion is symmetric, and the caller MUST set
/// `WENLAN_GRAPH_MEMORY_STREAM=0` (enforced at entry) so the OFF arm
/// (reranker=None) does not re-enable the graph stream that the ON arm
/// (reranker=Some) suppresses — without this, the A/B measures
/// graph-stream vs CE, not CE alone.
///
/// Per question, all arms share the same seeded DB (same seed/enrichment
/// as `run_fullpipeline_lme_batch`). Answers are generated on-device by the
/// supplied `llm` provider (Qwen3.5-9B) at `temperature = 0.0` so all arms
/// are deterministic and self-judging is avoided (the downstream judge is a
/// different model). Mirrors `generate_e2e_answers_for_question`'s LLM call.
///
/// Approach labels: `"<arm_prefix>_{category}"` per arm.
///
/// ## MEASUREMENT SCOPE
///
/// This isolates the cross-encoder. The confounder gate forces the graph stream
/// (and skip-preference / temporal boost+filter) OFF on ALL arms, so the only
/// deliberate difference is the CE rescore. Production, however, runs the graph
/// stream DEFAULT-ON, where CE-off keeps the stream and CE-on drops it
/// (allow_graph_stream = reranker.is_none(), db.rs). A positive delta here therefore
/// supports 'the CE rescore adds answer accuracy in isolation' — it does NOT by
/// itself prove the production default-flip (CE-on coupled with stream-off) is a net
/// win; that needs a separate stream-on-vs-CE-on comparison. Interpret accordingly.
///
/// # Errors
///
/// Returns `Err` immediately if `WENLAN_GRAPH_MEMORY_STREAM` is not one of
/// "0"/"false"/"no"/"off" — the A/B is statistically invalid without it.
pub async fn run_fullpipeline_lme(
    longmemeval_path: &std::path::Path,
    enrichment: crate::eval::shared::EnrichmentMode,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    reranker: Option<Arc<dyn crate::reranker::Reranker>>,
    arms: &ArmSpec,
    db_source: DbSource,
    output_path: &std::path::Path,
) -> Result<Vec<crate::eval::judge::JudgmentTuple>, WenlanError> {
    use crate::eval::judge::{save_judgment_tuples, JudgmentTuple};
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    // Safety gate: the CE A/B isolates the cross-encoder only when no other
    // retrieval flag perturbs ONE arm asymmetrically. graph stream: the OFF arm
    // (reranker=None) re-enables it while the ON arm suppresses it; skip-pref: the
    // ON arm bypasses the CE on preference queries; temporal boost/filter: alters
    // the base pool. Reuse the same predicates that gate each feature so the guard
    // cannot drift. Refuse loud, naming the offenders, rather than emit a
    // confounded delta.
    //
    // Only enforced for an isolation A/B (a spec with a CeOff arm). A single-arm
    // ceiling (`ArmSpec::full_stack`) has no OFF arm to perturb, so features are
    // meant to be ON and the gate is skipped — see `needs_confounder_isolation`.
    if arms.needs_confounder_isolation() {
        let confounders: [(&str, bool); 4] = [
            (
                "WENLAN_GRAPH_MEMORY_STREAM",
                crate::db::graph_memory_stream_enabled(),
            ),
            (
                "WENLAN_RERANK_SKIP_PREFERENCE",
                crate::db::rerank_skip_preference_enabled(),
            ),
            (
                "WENLAN_ENABLE_TEMPORAL_SOFT_BOOST",
                crate::db::temporal_soft_boost_enabled(),
            ),
            (
                "WENLAN_ENABLE_TEMPORAL_FILTER",
                crate::db::temporal_filter_enabled(),
            ),
        ];
        let active: Vec<&str> = confounders
            .iter()
            .filter(|(_, on)| *on)
            .map(|(name, _)| *name)
            .collect();
        if !active.is_empty() {
            return Err(WenlanError::Generic(format!(
                "CE A/B refused: confounding flag(s) active: {}. Each perturbs one arm \
                 asymmetrically (graph stream re-enabled on the OFF arm; CE bypass on \
                 preference queries; temporal boost/filter alters the base pool), so the \
                 A/B would not isolate the cross-encoder. Set them all to 0/false/no/off \
                 and retry.",
                active.join(", ")
            )));
        }
    }

    let mut samples = load_longmemeval(longmemeval_path)?;
    let limit_active = std::env::var("LME_LIMIT_QUESTIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    if let Some(n) = limit_active {
        let total_before = samples.len();
        samples.truncate(n);
        eprintln!(
            "[ce_ab_lme] LME_LIMIT_QUESTIONS={n} -> processing {}/{}",
            samples.len(),
            total_before
        );
    }
    let shared_embedder = crate::eval::shared::eval_shared_embedder();

    // Consolidated source: open the ONE shared DB up front and assert it is
    // substrate-live (non-empty pool), so a misconfigured path fails loud here
    // rather than emitting a vacuous A/B. Opened read-only in effect — the
    // consolidated branch performs no writes (no seed, no event_date inject).
    let consolidated_db: Option<Arc<MemoryDB>> = match &db_source {
        DbSource::SeedPerQuestion => None,
        DbSource::Consolidated(dir) => {
            let db = MemoryDB::new_with_shared_embedder(
                dir,
                Arc::new(NoopEmitter),
                shared_embedder.clone(),
            )
            .await
            .map_err(|e| {
                WenlanError::Generic(format!(
                    "[ce_ab_lme] failed to open consolidated DB at {dir:?}: {e}"
                ))
            })?;
            let n = db.memory_count().await.unwrap_or(0);
            if n == 0 {
                return Err(WenlanError::Generic(format!(
                    "EVAL REFUSED [ce_ab_lme]: consolidated DB at {dir:?} has 0 memories. \
                     A CE A/B over an empty substrate measures noise, not CE. \
                     Seed it (scripts/seed-scenario-dbs.sh) and retry."
                )));
            }
            let fetch_pool = crate::db::compute_rerank_fetch_pool(
                10,
                std::env::var("RERANK_POOL_MULTIPLIER").ok().as_deref(),
                std::env::var("RERANK_POOL_FLOOR").ok().as_deref(),
            );
            eprintln!(
                "[ce_ab_lme] consolidated DB {dir:?}: {n} memories (shared pool), \
                 CE fetch_pool={fetch_pool} vs top-10 window{}",
                if fetch_pool <= 10 {
                    " — WARNING: fetch_pool<=10 means the CE reorders the returned \
                     docs without reselecting; set RERANK_POOL_FLOOR>10 to give it \
                     membership headroom"
                } else {
                    ""
                }
            );
            Some(Arc::new(db))
        }
    };

    // Resume: load any already-generated tuples.
    let mut finished_tuples: Vec<JudgmentTuple> = if output_path.exists() {
        let existing = crate::eval::judge::load_judgment_tuples(output_path)
            .map_err(|e| WenlanError::Generic(format!("load resume: {e}")))?;
        eprintln!(
            "[ce_ab_lme] Resuming with {} existing tuples",
            existing.len()
        );
        existing
    } else {
        Vec::new()
    };
    if limit_active.is_some() {
        let allowed: std::collections::HashSet<String> =
            samples.iter().map(|s| s.question.clone()).collect();
        let before = finished_tuples.len();
        finished_tuples.retain(|t| allowed.contains(&t.question));
        if before != finished_tuples.len() {
            eprintln!(
                "[ce_ab_lme] resume filter: kept {}/{} tuples within limit window",
                finished_tuples.len(),
                before
            );
        }
    }

    // A question is done only when ALL arms are present.
    // Build one set per prefix, then intersect them all.
    let done_questions: std::collections::HashSet<String> = {
        let prefixes = arms.prefixes();
        let mut per_prefix: Vec<std::collections::HashSet<String>> = prefixes
            .iter()
            .map(|_| std::collections::HashSet::new())
            .collect();
        for t in &finished_tuples {
            let k = if t.question_id.is_empty() {
                t.question.clone()
            } else {
                t.question_id.clone()
            };
            for (i, prefix) in prefixes.iter().enumerate() {
                if t.approach.starts_with(&format!("{prefix}_")) {
                    per_prefix[i].insert(k.clone());
                }
            }
        }
        // Intersection across all arm-prefix sets.
        let mut iter = per_prefix.into_iter();
        if let Some(first) = iter.next() {
            iter.fold(first, |acc, set| acc.intersection(&set).cloned().collect())
        } else {
            std::collections::HashSet::new()
        }
    };

    let baselines_dir = output_path
        .parent()
        .ok_or_else(|| WenlanError::Generic("output_path has no parent".to_string()))?;

    let scenario_concurrency: usize = std::env::var("EVAL_SCENARIO_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .min(8);
    eprintln!("[ce_ab_lme] scenario concurrency = {scenario_concurrency}");

    let mut processed = 0usize;
    let mut save_counter = 0usize;

    if scenario_concurrency <= 1 {
        for (q_idx, sample) in samples.iter().enumerate() {
            // Match the done-set key (question_id, fallback to text) — see build above.
            let sample_key: &str = if sample.question_id.is_empty() {
                sample.question.as_str()
            } else {
                sample.question_id.as_str()
            };
            if done_questions.contains(sample_key) {
                continue;
            }

            // Ground truth: LME answers are JSON strings; handle non-string values
            // explicitly rather than borrowing a temporary (M1).
            let ground_truth = match &sample.answer {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if ground_truth.is_empty() {
                continue;
            }

            // Acquire this question's retrieval DB. Consolidated reuses the ONE
            // shared large-pool DB opened above (skip per-question seeding +
            // event_date injection — it is already enriched + 100% dated, and is
            // never mutated). SeedPerQuestion seeds a fresh oracle-pool scenario DB
            // from the sample's own sessions, byte-identical to the prior behavior.
            let db: Arc<MemoryDB> = match &db_source {
                DbSource::Consolidated(_) => consolidated_db
                    .clone()
                    .expect("consolidated_db is Some for DbSource::Consolidated"),
                DbSource::SeedPerQuestion => {
                    let memories = extract_memories(sample);
                    if memories.is_empty() {
                        continue;
                    }

                    let scope_dir = crate::eval::shared::scenario_db_dir(
                        baselines_dir,
                        "lme",
                        &sample.question_id,
                    );

                    let question_id = sample.question_id.clone();
                    let question_type = sample.question_type.clone();
                    let memories_owned = memories.clone();

                    let seeded = match crate::eval::shared::open_or_seed_scenario_db(
                        &scope_dir,
                        shared_embedder.clone(),
                        move || {
                            memories_owned
                                .iter()
                                .map(|mem| crate::sources::RawDocument {
                                    content: mem.content.clone(),
                                    source_id: format!(
                                        "lme_{}_{}_t{}",
                                        question_id, mem.session_idx, mem.turn_idx
                                    ),
                                    source: "memory".to_string(),
                                    title: format!(
                                        "session {} turn {}",
                                        mem.session_idx, mem.turn_idx
                                    ),
                                    memory_type: Some(
                                        if question_type == "single-session-preference" {
                                            "preference"
                                        } else {
                                            "fact"
                                        }
                                        .to_string(),
                                    ),
                                    space: Some("conversation".to_string()),
                                    last_modified: chrono::Utc::now().timestamp(),
                                    ..Default::default()
                                })
                                .collect()
                        },
                        &enrichment,
                    )
                    .await
                    {
                        Ok(db) => db,
                        Err(e) => {
                            return Err(WenlanError::Generic(format!(
                                "[ce_ab_lme] scenario {} failed to open/seed DB: {e}. Refusing to \
                                 silently drop the question — re-seed and retry.",
                                sample.question_id
                            )));
                        }
                    };

                    // Substrate-liveness: a per-question DB with zero memories yields a
                    // vacuous A/B (both arms retrieve nothing). Fail loud rather than
                    // emit a misleading null, per the eval ONE-contract discipline (H2).
                    let db_mem_count = seeded.memory_count().await.unwrap_or(0);
                    if db_mem_count == 0 {
                        return Err(WenlanError::Generic(format!(
                            "EVAL REFUSED [ce_ab_lme]: scenario {} seeded DB has 0 memories. \
                             A CE A/B over an empty substrate measures noise, not CE. Re-seed and retry.",
                            sample.question_id
                        )));
                    }

                    // Make the temporal substrate LIVE for this question's scenario DB by
                    // injecting per-session event_date (turn text carries no date; the date
                    // is per-session haystack metadata). This is substrate-prep + a liveness
                    // gate, NOT a direct answer fix: event_date is not rendered into the CE
                    // A/B context (build_structured_context renders only content) and no
                    // temporal ranking flag is set on this path, so dates influence neither
                    // ranking nor the prompt today. Their value is (a) a per-question liveness
                    // signal (the rows-updated count below), (b) a future temporal lever has
                    // live data to read. source_id keys mirror the seed closure's
                    // `lme_{qid}_{sess}_t{turn}` exactly via event_date_map -> memory_source_id.
                    let date_map =
                        crate::eval::longmemeval::event_date_map(std::slice::from_ref(sample));
                    if !date_map.is_empty() {
                        let updates: Vec<(String, i64)> = date_map.into_iter().collect();
                        let n_total = updates.len();
                        let rows = seeded.set_event_dates_by_source_id(&updates).await?;
                        if rows == 0 {
                            // Per-question log + continue (NOT whole-run abort): a non-empty
                            // date_map that updates 0 rows means source_id-format drift for
                            // THIS question. Skip it rather than nuke a multi-hour decision run;
                            // the rows-updated count is an earlier, more precise signal than a
                            // DB-wide liveness assert (which propagated and aborted the run).
                            log::warn!(
                                "[ce_ab_lme] event_date injection updated 0/{} rows for question {} \
                                 — source_id drift; skipping question",
                                n_total,
                                sample.question_id
                            );
                            continue;
                        }
                    }

                    Arc::new(seeded)
                }
            };

            let category = category_name(&sample.question_type);

            // Generate an answer for all arms from the same seeded DB.
            let arm_pairs = arms.resolve(category, &reranker)?;
            for (arm_label, retrieval, include_pages) in arm_pairs {
                let (ctx, ctx_tokens) = match build_structured_context(
                    &db,
                    &sample.question,
                    10,
                    None,
                    retrieval,
                    include_pages,
                )
                .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(WenlanError::Generic(format!(
                            "[ce_ab_lme] scenario {} arm {arm_label} context build failed: {e}",
                            sample.question_id
                        )));
                    }
                };

                let request = LlmRequest {
                    system_prompt: Some(e2e_system_prompt()),
                    user_prompt: format!("Context:\n{}\n\nQuestion: {}", ctx, sample.question),
                    max_tokens: e2e_max_answer_tokens() as u32,
                    // Pinned to 0.0 (vs the 0.1 used elsewhere) so both arms are
                    // deterministic — fairness depends on identical decoding.
                    temperature: 0.0,
                    label: Some(arm_label.clone()),
                    timeout_secs: None,
                };

                let answer = match llm.generate(request).await {
                    Ok(raw) => strip_think_tags(&raw).trim().to_string(),
                    Err(e) => {
                        // Skip ALL arms for this question rather than persisting a
                        // half-pair; on resume the question is reprocessed (H1).
                        eprintln!(
                            "[ce_ab_lme] WARN: scenario {} arm {arm_label} generation failed: {e}. \
                             Dropping all arms for this question; will retry on resume.",
                            sample.question_id
                        );
                        break;
                    }
                };

                // Never persist an empty answer — it would mark the question DONE on
                // resume and silently bias the arm to score 0 (H1).
                if answer.is_empty() {
                    eprintln!(
                        "[ce_ab_lme] WARN: scenario {} arm {arm_label} produced empty answer. \
                         Dropping all arms; will retry on resume.",
                        sample.question_id
                    );
                    break;
                }

                finished_tuples.push(JudgmentTuple {
                    question: sample.question.clone(),
                    ground_truth: ground_truth.clone(),
                    approach: arm_label,
                    answer,
                    context_tokens: ctx_tokens,
                    category: category.to_string(),
                    question_id: sample.question_id.clone(),
                });
            }

            processed += 1;
            save_counter += 1;
            if save_counter >= 10 {
                save_counter = 0;
                if let Err(e) = save_judgment_tuples(&finished_tuples, output_path) {
                    eprintln!("[ce_ab_lme] WARN: incremental save failed: {e}");
                }
            }
            if q_idx % 100 == 99 {
                eprintln!(
                    "[ce_ab_lme] {}/{} questions processed",
                    q_idx + 1,
                    samples.len()
                );
            }
        }
    } else {
        use futures::StreamExt;

        let enrichment_arc = Arc::new(enrichment);
        let done_arc = Arc::new(done_questions.clone());
        let baselines_dir_owned = baselines_dir.to_path_buf();
        let db_source_owned = db_source.clone();
        let consolidated_db_owned = consolidated_db.clone();
        let arms_owned = arms.clone();
        let total = samples.len();

        let scenario_outputs: Vec<Result<Vec<JudgmentTuple>, WenlanError>> =
            futures::stream::iter(samples.iter().enumerate())
                .map(|(q_idx, sample)| {
                    let shared_embedder = shared_embedder.clone();
                    let enrichment_arc = enrichment_arc.clone();
                    let done_arc = done_arc.clone();
                    let baselines_dir_owned = baselines_dir_owned.clone();
                    let db_source = db_source_owned.clone();
                    let consolidated_db = consolidated_db_owned.clone();
                    let arms = arms_owned.clone();
                    let reranker = reranker.clone();
                    let llm = llm.clone();
                    async move {
                        let sample_key: &str = if sample.question_id.is_empty() {
                            sample.question.as_str()
                        } else {
                            sample.question_id.as_str()
                        };
                        if done_arc.contains(sample_key) {
                            return Ok(Vec::new());
                        }

                        let ground_truth = match &sample.answer {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        if ground_truth.is_empty() {
                            return Ok(Vec::new());
                        }

                        let db: Arc<MemoryDB> = match &db_source {
                            DbSource::Consolidated(_) => consolidated_db
                                .clone()
                                .expect("consolidated_db is Some for DbSource::Consolidated"),
                            DbSource::SeedPerQuestion => {
                                let memories = extract_memories(sample);
                                if memories.is_empty() {
                                    return Ok(Vec::new());
                                }

                                let scope_dir = crate::eval::shared::scenario_db_dir(
                                    &baselines_dir_owned,
                                    "lme",
                                    &sample.question_id,
                                );

                                let question_id = sample.question_id.clone();
                                let question_type = sample.question_type.clone();
                                let memories_owned = memories.clone();

                                let seeded = match crate::eval::shared::open_or_seed_scenario_db(
                                    &scope_dir,
                                    shared_embedder,
                                    move || {
                                        memories_owned
                                            .iter()
                                            .map(|mem| crate::sources::RawDocument {
                                                content: mem.content.clone(),
                                                source_id: format!(
                                                    "lme_{}_{}_t{}",
                                                    question_id, mem.session_idx, mem.turn_idx
                                                ),
                                                source: "memory".to_string(),
                                                title: format!(
                                                    "session {} turn {}",
                                                    mem.session_idx, mem.turn_idx
                                                ),
                                                memory_type: Some(
                                                    if question_type
                                                        == "single-session-preference"
                                                    {
                                                        "preference"
                                                    } else {
                                                        "fact"
                                                    }
                                                    .to_string(),
                                                ),
                                                space: Some("conversation".to_string()),
                                                last_modified: chrono::Utc::now().timestamp(),
                                                ..Default::default()
                                            })
                                            .collect()
                                    },
                                    &enrichment_arc,
                                )
                                .await
                                {
                                    Ok(db) => db,
                                    Err(e) => {
                                        return Err(WenlanError::Generic(format!(
                                            "[ce_ab_lme] scenario {} failed to open/seed DB: {e}. Refusing to \
                                             silently drop the question — re-seed and retry.",
                                            sample.question_id
                                        )));
                                    }
                                };

                                let db_mem_count = seeded.memory_count().await.unwrap_or(0);
                                if db_mem_count == 0 {
                                    return Err(WenlanError::Generic(format!(
                                        "EVAL REFUSED [ce_ab_lme]: scenario {} seeded DB has 0 memories. \
                                         A CE A/B over an empty substrate measures noise, not CE. Re-seed and retry.",
                                        sample.question_id
                                    )));
                                }

                                let date_map = crate::eval::longmemeval::event_date_map(
                                    std::slice::from_ref(sample),
                                );
                                if !date_map.is_empty() {
                                    let updates: Vec<(String, i64)> =
                                        date_map.into_iter().collect();
                                    let n_total = updates.len();
                                    let rows = seeded.set_event_dates_by_source_id(&updates).await?;
                                    if rows == 0 {
                                        log::warn!(
                                            "[ce_ab_lme] event_date injection updated 0/{} rows for question {} \
                                             — source_id drift; skipping question",
                                            n_total,
                                            sample.question_id
                                        );
                                        return Ok(Vec::new());
                                    }
                                }

                                Arc::new(seeded)
                            }
                        };

                        let category = category_name(&sample.question_type);
                        let arm_pairs = arms.resolve(category, &reranker)?;
                        let mut tuples = Vec::with_capacity(arm_pairs.len());
                        for (arm_label, retrieval, include_pages) in arm_pairs {
                            let (ctx, ctx_tokens) = match build_structured_context(
                                &db,
                                &sample.question,
                                10,
                                None,
                                retrieval,
                                include_pages,
                            )
                            .await
                            {
                                Ok(v) => v,
                                Err(e) => {
                                    return Err(WenlanError::Generic(format!(
                                        "[ce_ab_lme] scenario {} arm {arm_label} context build failed: {e}",
                                        sample.question_id
                                    )));
                                }
                            };

                            let request = LlmRequest {
                                system_prompt: Some(e2e_system_prompt()),
                                user_prompt: format!(
                                    "Context:\n{}\n\nQuestion: {}",
                                    ctx, sample.question
                                ),
                                max_tokens: e2e_max_answer_tokens() as u32,
                                temperature: 0.0,
                                label: Some(arm_label.clone()),
                                timeout_secs: None,
                            };

                            let answer = match llm.generate(request).await {
                                Ok(raw) => strip_think_tags(&raw).trim().to_string(),
                                Err(e) => {
                                    eprintln!(
                                        "[ce_ab_lme] WARN: scenario {} arm {arm_label} generation failed: {e}. \
                                         Dropping all arms for this question; will retry on resume.",
                                        sample.question_id
                                    );
                                    return Ok(Vec::new());
                                }
                            };

                            if answer.is_empty() {
                                eprintln!(
                                    "[ce_ab_lme] WARN: scenario {} arm {arm_label} produced empty answer. \
                                     Dropping all arms; will retry on resume.",
                                    sample.question_id
                                );
                                return Ok(Vec::new());
                            }

                            tuples.push(JudgmentTuple {
                                question: sample.question.clone(),
                                ground_truth: ground_truth.clone(),
                                approach: arm_label,
                                answer,
                                context_tokens: ctx_tokens,
                                category: category.to_string(),
                                question_id: sample.question_id.clone(),
                            });
                        }

                        if q_idx % 100 == 99 {
                            eprintln!(
                                "[ce_ab_lme] {}/{} questions processed",
                                q_idx + 1,
                                total
                            );
                        }

                        Ok(tuples)
                    }
                })
                .buffer_unordered(scenario_concurrency)
                .collect()
                .await;

        for result in scenario_outputs {
            let tuples = result?;
            if !tuples.is_empty() {
                processed += 1;
                finished_tuples.extend(tuples);
                // Mirror the serial path's incremental checkpoint (see save_counter above):
                // this drain loop is sequential (single-writer), so saving every 10 questions
                // bounds crash loss to ~10 questions instead of the entire run when
                // scenario_concurrency > 1. The final save below flushes the remainder.
                save_counter += 1;
                if save_counter >= 10 {
                    save_counter = 0;
                    if let Err(e) = save_judgment_tuples(&finished_tuples, output_path) {
                        eprintln!("[ce_ab_lme] WARN: incremental save failed: {e}");
                    }
                }
            }
        }
    }

    save_judgment_tuples(&finished_tuples, output_path)
        .map_err(|e| WenlanError::Generic(format!("save: {e}")))?;
    eprintln!(
        "[ce_ab_lme] Processed {processed} new questions; saved {} total tuples to {:?}",
        finished_tuples.len(),
        output_path
    );

    Ok(finished_tuples)
}

// ===== Flat cache loaders =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::longmemeval::{ChatTurn, LongMemEvalSample};
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn answer_prompt_v2_gate_default_off_byte_identical() {
        // Save/restore the env so we don't poison other tests.
        let prev = std::env::var("WENLAN_EVAL_ANSWER_PROMPT_V2").ok();
        std::env::remove_var("WENLAN_EVAL_ANSWER_PROMPT_V2");
        assert!(!e2e_answer_prompt_v2_enabled());
        assert_eq!(e2e_system_prompt(), E2E_SYSTEM_PROMPT);
        assert_eq!(e2e_max_answer_tokens(), 200);

        std::env::set_var("WENLAN_EVAL_ANSWER_PROMPT_V2", "1");
        assert!(e2e_answer_prompt_v2_enabled());
        assert_eq!(e2e_system_prompt(), E2E_SYSTEM_PROMPT_V2);
        assert_eq!(e2e_max_answer_tokens(), 512);

        // restore
        match prev {
            Some(v) => std::env::set_var("WENLAN_EVAL_ANSWER_PROMPT_V2", v),
            None => std::env::remove_var("WENLAN_EVAL_ANSWER_PROMPT_V2"),
        }
    }

    #[tokio::test]
    async fn event_date_injection_lands_on_seeded_rows() {
        let sample = LongMemEvalSample {
            question_id: "qtest".to_string(),
            question_type: "temporal-reasoning".to_string(),
            question: "What happened after the May session?".to_string(),
            answer: serde_json::json!("the answer"),
            question_date: "2023/05/02 (Tue) 10:00".to_string(),
            haystack_dates: vec!["2023/05/01 (Mon) 10:00".to_string()],
            haystack_session_ids: vec!["s0".to_string()],
            haystack_sessions: vec![vec![ChatTurn {
                role: "user".to_string(),
                content: "I had the planning session today.".to_string(),
                has_answer: true,
            }]],
            answer_session_ids: vec!["s0".to_string()],
        };

        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            eval_shared_embedder(),
        )
        .await
        .unwrap();

        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: format!("lme_{}_{}_t{}", sample.question_id, 0usize, 0usize),
            title: "session 0 turn 0".to_string(),
            content: "I had the planning session today.".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            space: Some("conversation".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();

        let updates: Vec<(String, i64)> =
            crate::eval::longmemeval::event_date_map(std::slice::from_ref(&sample))
                .into_iter()
                .collect();
        let updated = db.set_event_dates_by_source_id(&updates).await.unwrap();
        assert_eq!(updated, 1, "event_date_map key must match seeded source_id");

        db.assert_eval_feature_substrate_live("temporal")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn concurrent_collection_preserves_qid_attribution() {
        use futures::StreamExt;

        fn val(qid: &str) -> u64 {
            qid.strip_prefix('q')
                .unwrap()
                .parse::<u64>()
                .unwrap()
                .pow(2)
        }

        // Create 10 qids to increase chance of completion-order mismatch.
        let qids: Vec<String> = (0..10).map(|i| format!("q{i}")).collect();
        let serial_map: HashMap<String, u64> =
            qids.iter().map(|qid| (qid.clone(), val(qid))).collect();

        // Intentionally reverse the sleep order so earlier inputs (smaller qids) sleep LONGER,
        // forcing completion order != input order with very high probability.
        let n = qids.len() as u64;
        let outputs: Vec<(String, u64)> = futures::stream::iter(qids.iter().enumerate())
            .map(|(idx, qid)| async move {
                // q0 sleeps 100ms, q1 sleeps 90ms, ..., q9 sleeps 10ms
                // So completion order will be: q9, q8, ..., q1, q0 (reverse of input)
                tokio::time::sleep(Duration::from_millis((n - idx as u64) * 10)).await;
                (qid.clone(), val(qid))
            })
            .buffer_unordered(8)
            .collect()
            .await;

        // WRONG approach: zip original qids with completion-ordered values.
        // Since completion order is reversed, zipping produces:
        // q0 -> val(q9)=81, q1 -> val(q8)=64, q2 -> val(q7)=49, etc.
        // This is WRONG and must NOT equal serial_map.
        let wrong_index_map: HashMap<String, u64> = qids
            .iter()
            .cloned()
            .zip(outputs.iter().map(|(_, v)| *v))
            .collect();

        // The WRONG map MUST differ from the serial map (the whole point of this test).
        // If this assertion fails, it means the test isn't detecting misattribution properly.
        assert_ne!(
            wrong_index_map, serial_map,
            "CRITICAL: wrong_index_map MUST differ from serial_map to validate this test's effectiveness. \
             This failure means the scrambling didn't work as expected or the test logic is flawed."
        );

        // RIGHT approach: collect tuples that CARRY their qid alongside their value.
        // Each tuple remembers its own qid, so reassembly is correct regardless of completion order.
        let concurrent_map: HashMap<String, u64> = outputs.into_iter().collect();
        assert_eq!(
            concurrent_map, serial_map,
            "concurrent_map MUST match serial_map when tuples carry their own qid"
        );
    }

    #[test]
    fn arm_spec_ce_two_labels() {
        let spec = ArmSpec::ce_two();
        assert_eq!(
            spec.labels("single-session-user"),
            vec![
                "ce_off_single-session-user".to_string(),
                "ce_on_single-session-user".to_string(),
            ],
            "ce_two labels for single-session-user"
        );
        assert_eq!(
            spec.labels("temporal-reasoning"),
            vec![
                "ce_off_temporal-reasoning".to_string(),
                "ce_on_temporal-reasoning".to_string(),
            ],
            "ce_two labels for temporal-reasoning"
        );
        assert_eq!(spec.prefixes(), vec!["ce_off", "ce_on"], "ce_two prefixes");
        assert!(
            spec.needs_confounder_isolation(),
            "ce_two is an isolation A/B (has CeOff) — gate must apply"
        );
    }

    #[test]
    fn arm_spec_full_stack_labels() {
        let spec = ArmSpec::full_stack();
        assert_eq!(
            spec.labels("single-session-user"),
            vec!["fullstack_single-session-user".to_string()],
            "full_stack is a single arm"
        );
        assert_eq!(spec.prefixes(), vec!["fullstack"], "full_stack prefix");
        assert!(
            !spec.needs_confounder_isolation(),
            "full_stack has no CeOff arm — confounder gate must be skipped so features can be ON"
        );
    }

    /// Task 1: `include_pages=true` surfaces `## Compiled Knowledge`; `false` omits it.
    #[tokio::test]
    async fn build_structured_context_include_pages_toggle() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            eval_shared_embedder(),
        )
        .await
        .unwrap();

        // Seed two memories so the page overlap gate (default min_overlap=2) passes.
        let source_id_1 = "test_mem_p1";
        let source_id_2 = "test_mem_p2";
        db.upsert_documents(vec![
            RawDocument {
                source: "memory".to_string(),
                source_id: source_id_1.to_string(),
                title: "planning session".to_string(),
                content: "I discussed project planning with the team.".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                memory_type: Some("fact".to_string()),
                space: Some("conversation".to_string()),
                ..Default::default()
            },
            RawDocument {
                source: "memory".to_string(),
                source_id: source_id_2.to_string(),
                title: "team coordination".to_string(),
                content: "We coordinated the planning tasks across the team.".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                memory_type: Some("fact".to_string()),
                space: Some("conversation".to_string()),
                ..Default::default()
            },
        ])
        .await
        .unwrap();

        // Insert a page referencing both memories (satisfies default min_overlap=2 gate).
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_test_001",
            "Project Planning Guide",
            Some("Overview of planning practices."),
            "This page covers project planning discussions and team coordination.",
            None,
            None,
            &[source_id_1, source_id_2],
            &now,
        )
        .await
        .unwrap();

        let question = "planning";

        let (ctx_on, _) =
            build_structured_context(&db, question, 10, None, CtxRetrieval::Quick, true)
                .await
                .unwrap();

        let (ctx_off, _) =
            build_structured_context(&db, question, 10, None, CtxRetrieval::Quick, false)
                .await
                .unwrap();

        assert!(
            ctx_on.contains("## Compiled Knowledge"),
            "include_pages=true must include ## Compiled Knowledge; got: {ctx_on}"
        );
        assert!(
            !ctx_off.contains("## Compiled Knowledge"),
            "include_pages=false must NOT include ## Compiled Knowledge; got: {ctx_off}"
        );
        // Both should contain memory content when there are hits.
        assert!(
            ctx_on.contains("## Relevant Memories") || ctx_on.contains("planning"),
            "include_pages=true context should contain memory content"
        );
        assert!(
            ctx_off.contains("## Relevant Memories") || ctx_off.contains("planning"),
            "include_pages=false context should contain memory content"
        );
    }

    /// Task 2 (pure, no DB): pages_ab() has 3 arms with the right kinds and include_pages flags.
    #[test]
    fn arm_spec_pages_ab_arms() {
        let spec = ArmSpec::pages_ab();
        let arms: Vec<_> = spec.arms.iter().collect();
        assert_eq!(arms.len(), 3, "pages_ab must have 3 arms");
        assert_eq!(arms[0].kind, ArmKind::CeOff, "arm 0 kind");
        assert_eq!(arms[1].kind, ArmKind::CeOn, "arm 1 kind");
        assert_eq!(arms[2].kind, ArmKind::CeOn, "arm 2 kind");
        assert!(!arms[0].include_pages, "arm 0 include_pages must be false");
        assert!(!arms[1].include_pages, "arm 1 include_pages must be false");
        assert!(arms[2].include_pages, "arm 2 include_pages must be true");
    }

    /// Task 2 (pure, no DB): pages_ab().labels() produces the correct suffixed label strings.
    #[test]
    fn arm_spec_pages_ab_labels() {
        let spec = ArmSpec::pages_ab();
        assert_eq!(
            spec.labels("x"),
            vec![
                "ce_off_x".to_string(),
                "ce_on_x".to_string(),
                "ce_on_pages_x".to_string()
            ],
            "pages_ab labels"
        );
    }

    /// Task 2 (pure, no DB): ce_two() and full_stack() all arms have include_pages=true
    /// (byte-identical preservation guard).
    #[test]
    fn arm_spec_existing_arms_include_pages_true() {
        for arm in &ArmSpec::ce_two().arms {
            assert!(
                arm.include_pages,
                "ce_two arm {:?} must have include_pages=true",
                arm.kind
            );
        }
        for arm in &ArmSpec::full_stack().arms {
            assert!(
                arm.include_pages,
                "full_stack arm {:?} must have include_pages=true",
                arm.kind
            );
        }
    }
}
