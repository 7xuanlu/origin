// SPDX-License-Identifier: AGPL-3.0-only
//! E2E answer quality evaluation: generate answers from context, judge quality.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::fixtures::load_fixtures;
use crate::eval::judge::JudgmentTuple;
use crate::eval::shared::{count_tokens, eval_shared_embedder, run_entity_extraction_for_eval};
use crate::events::NoopEmitter;
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
/// - Origin: search results as context
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
) -> Result<E2EEvalReport, OriginError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| OriginError::Generic("ANTHROPIC_API_KEY not set".to_string()))?;

    let model = "claude-haiku-4-5-20251001";
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| OriginError::Generic(format!("failed to build reqwest client: {e}")))?;

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

        // Origin: seed ephemeral DB, run hybrid search
        let origin_context = {
            let case_tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tempdir e2e: {e}")))?;
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
            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
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
) -> Result<(E2ELocomoReport, Vec<JudgmentTuple>), OriginError> {
    use crate::eval::locomo::{extract_observations, load_locomo};
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    let samples = load_locomo(locomo_path)?;

    // Accumulators: (answer_score, context_tokens, answer_len)
    let approach_keys = ["origin", "full_replay", "no_context"];
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

        // Seed ephemeral DB for Origin retrieval.
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir e2e_locomo: {e}")))?;
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
                domain: Some("conversation".to_string()),
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

            // ---- Origin approach: hybrid search ----
            let origin_context = {
                let results = db
                    .search_memory(
                        &qa.question,
                        search_top_k,
                        None,
                        Some("conversation"),
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
                    });
                }
                Err(e) => {
                    log::warn!("[e2e_locomo] origin approach failed: {e}");
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
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    let system_prompt = "Answer the question using only the provided context. \
        Be specific and concise. Respond in 1-3 sentences."
        .to_string();

    let mut tuples = Vec::new();

    // --- Flat context: search_memory only ---
    let flat_results = db
        .search_memory(
            question,
            search_limit,
            None,
            Some("conversation"),
            None,
            None,
            None,
            None,
        )
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
        });
    }

    // --- Structured context: search_memory + concept articles ---
    let mut structured_parts: Vec<String> = Vec::new();

    // Concept articles (like chat-context's "Compiled Knowledge" section)
    let concepts = db.search_concepts(question, 3).await.unwrap_or_default();
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
) -> Result<Vec<JudgmentTuple>, OriginError> {
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
            .map_err(|e| OriginError::Generic(format!("tempdir e2e_context: {e}")))?;
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
                domain: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Enrich + distill
        eprintln!("  [enriching]...");
        let entities = run_entity_extraction_for_eval(&db, &llm).await?;
        let concepts =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
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
) -> Result<Vec<JudgmentTuple>, OriginError> {
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
            .map_err(|e| OriginError::Generic(format!("tempdir e2e_lme: {e}")))?;
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
                domain: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Enrich + distill
        let _entities = run_entity_extraction_for_eval(&db, &llm).await?;
        let _concepts =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;

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

/// Build structured context for a question against an enriched DB.
///
/// Returns the structured context: search_memory results + concept articles.
/// Matches production `/api/chat-context` assembly pattern.
/// Flat baseline comes from the retrieval-only pipeline caches.
async fn build_structured_context(
    db: &MemoryDB,
    question: &str,
    search_limit: usize,
    domain: Option<&str>,
) -> Result<(String, usize), OriginError> {
    use crate::concepts::filter_concepts_by_source_overlap;

    let results = db
        .search_memory(question, search_limit, None, domain, None, None, None, None)
        .await?;

    // Source IDs from search results — used to gate concept relevance
    let search_source_ids: std::collections::HashSet<String> =
        results.iter().map(|r| r.source_id.clone()).collect();

    // Structured: concepts + search results (matches production /api/chat-context).
    // EVAL_CONCEPT_MIN_OVERLAP env var lets us sweep thresholds without code changes;
    // defaults to the production tuning value (2).
    let min_overlap: usize = std::env::var("EVAL_CONCEPT_MIN_OVERLAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| crate::tuning::DistillationConfig::default().concept_min_overlap);

    let mut parts: Vec<String> = Vec::new();
    let raw_concepts = db.search_concepts(question, 3).await.unwrap_or_default();
    let concepts =
        filter_concepts_by_source_overlap(&raw_concepts, &search_source_ids, min_overlap);

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
) -> Result<Vec<JudgmentTuple>, OriginError> {
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
            .map_err(|e| OriginError::Generic(format!("load resume: {e}")))?;
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
        .ok_or_else(|| OriginError::Generic("output_path has no parent".to_string()))?;

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
                            domain: Some("conversation".to_string()),
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
                let ctx_result = build_structured_context(&db, &qa.question, 10, None).await;
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
                    Some(E2E_SYSTEM_PROMPT.to_string()),
                    200,
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

        let scenario_outputs: Vec<Result<ScenarioOutput, OriginError>> =
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
                                        domain: Some("conversation".to_string()),
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
                            let ctx_result =
                                build_structured_context(&db, &qa.question, 10, None)
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
                                Some(E2E_SYSTEM_PROMPT.to_string()),
                                200,
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
            .map_err(|e| OriginError::Generic(format!("save: {e}")))?;
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
            .map_err(|e| OriginError::Generic(format!("client: {e}")))?;

        let batch_id = submit_batch(&client, api_key, batch_requests, answer_model, cost_cap_usd)
            .await
            .map_err(|e| OriginError::Generic(format!("batch submit: {e}")))?;
        eprintln!("[fullpipeline] Batch submitted: {}", batch_id);

        let results_url = poll_batch(&client, api_key, &batch_id)
            .await
            .map_err(|e| OriginError::Generic(format!("batch poll: {e}")))?;

        download_batch_results(&client, api_key, &results_url)
            .await
            .map_err(|e| OriginError::Generic(format!("batch download: {e}")))?
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
        .map_err(|e| OriginError::Generic(format!("save: {e}")))?;
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
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::eval::judge::save_judgment_tuples;
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};

    let mut samples = load_longmemeval(longmemeval_path)?;
    // Optional limit for small test runs (set LME_LIMIT_QUESTIONS=N).
    if let Some(n) = std::env::var("LME_LIMIT_QUESTIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
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
            .map_err(|e| OriginError::Generic(format!("load resume: {e}")))?;
        eprintln!(
            "[fullpipeline_lme] Resuming with {} existing tuples",
            existing.len()
        );
        existing
    } else {
        Vec::new()
    };
    let done_questions: std::collections::HashSet<String> =
        finished_tuples.iter().map(|t| t.question.clone()).collect();

    // --- Phase 1+2 merged: per-question DB open (cached) + context build ---
    // output_path is e.g. baselines/fullpipeline_lme_tuples.json; its parent is baselines/.
    let baselines_dir = output_path
        .parent()
        .ok_or_else(|| OriginError::Generic("output_path has no parent".to_string()))?;

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
                            domain: Some("conversation".to_string()),
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
            let ctx_result = build_structured_context(&db, &sample.question, 10, None).await;
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
                Some(E2E_SYSTEM_PROMPT.to_string()),
                200,
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

        let scenario_outputs: Vec<Result<LmeScenarioOutput, OriginError>> =
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
                                        domain: Some("conversation".to_string()),
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
                        let ctx_result =
                            build_structured_context(&db, &sample.question, 10, None).await;
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
                            Some(E2E_SYSTEM_PROMPT.to_string()),
                            200usize,
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
            .map_err(|e| OriginError::Generic(format!("save: {e}")))?;
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
            .map_err(|e| OriginError::Generic(format!("client: {e}")))?;

        let batch_id = submit_batch(&client, api_key, batch_requests, answer_model, cost_cap_usd)
            .await
            .map_err(|e| OriginError::Generic(format!("batch submit: {e}")))?;
        eprintln!("[fullpipeline_lme] Batch submitted: {}", batch_id);

        let results_url = poll_batch(&client, api_key, &batch_id)
            .await
            .map_err(|e| OriginError::Generic(format!("batch poll: {e}")))?;

        download_batch_results(&client, api_key, &results_url)
            .await
            .map_err(|e| OriginError::Generic(format!("batch download: {e}")))?
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
        .map_err(|e| OriginError::Generic(format!("save: {e}")))?;
    eprintln!(
        "[fullpipeline_lme] Saved {} total tuples to {:?}",
        finished_tuples.len(),
        output_path
    );

    Ok(finished_tuples)
}

// ===== Flat cache loaders =====
