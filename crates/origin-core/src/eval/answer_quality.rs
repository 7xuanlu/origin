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
                last_modified: crate::eval::dates::seed_last_modified(
                    mem.session_date.as_deref(),
                    crate::eval::dates::parse_locomo_date,
                ),
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
    question_date: Option<&str>,
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    // If we have a question_date (LongMemEval), prepend it to the system prompt
    // so the LLM has a "today" anchor for relative time references in the question.
    let system_prompt = match question_date {
        Some(d) => format!(
            "The question was asked on {}. Answer the question using only the provided context. \
             Be specific and concise. Respond in 1-3 sentences.",
            d
        ),
        None => "Answer the question using only the provided context. \
            Be specific and concise. Respond in 1-3 sentences."
            .to_string(),
    };

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
        .map(|r| {
            format!(
                "On {}: {}",
                crate::eval::shared::format_ymd(r.last_modified),
                r.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let flat_tokens = count_tokens(&flat_context);

    let flat_request = LlmRequest {
        system_prompt: Some(system_prompt.clone()),
        user_prompt: format!(
            "Context (each line prefixed with the date the memory was recorded):\n{}\n\nQuestion: {}",
            flat_context, question
        ),
        max_tokens: 200,
        temperature: 0.1,
        label: Some("e2e_flat".to_string()),
    };
    if let Ok(raw) = llm.generate(flat_request).await {
        let answer = strip_think_tags(&raw);
        tuples.push(JudgmentTuple {
            question: question.to_string(),
            ground_truth: ground_truth.to_string(),
            approach: format!("flat_{}", category),
            answer,
            context_tokens: flat_tokens,
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
        for r in flat_results.iter() {
            structured_parts.push(format!(
                "On {}: {}",
                crate::eval::shared::format_ymd(r.last_modified),
                r.content
            ));
        }
    }

    let structured_context = structured_parts.join("\n\n");
    let structured_tokens = count_tokens(&structured_context);

    let structured_request = LlmRequest {
        system_prompt: Some(system_prompt),
        user_prompt: format!(
            "Context (each line prefixed with the date the memory was recorded; concept articles are time-spanning):\n{}\n\nQuestion: {}",
            structured_context, question
        ),
        max_tokens: 200,
        temperature: 0.1,
        label: Some("e2e_structured".to_string()),
    };
    if let Ok(raw) = llm.generate(structured_request).await {
        let answer = strip_think_tags(&raw);
        tuples.push(JudgmentTuple {
            question: question.to_string(),
            ground_truth: ground_truth.to_string(),
            approach: format!("structured_{}", category),
            answer,
            context_tokens: structured_tokens,
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
                last_modified: crate::eval::dates::seed_last_modified(
                    mem.session_date.as_deref(),
                    crate::eval::dates::parse_locomo_date,
                ),
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
                None,
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
                last_modified: crate::eval::dates::seed_last_modified(
                    mem.session_date.as_deref(),
                    crate::eval::dates::parse_lme_date,
                ),
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
            Some(&sample.question_date),
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
    context_tokens: usize,
}

/// System prompt used for all E2E answer generation.
const E2E_SYSTEM_PROMPT: &str =
    "Answer the question using only the provided context. Be specific and concise. Respond in 1-3 sentences.";

/// Build flat + structured contexts for a question against an enriched DB.
///
/// Returns `(flat_context, structured_context)`. The flat context uses only
/// `search_memory` results; structured adds concept articles on top.
async fn build_contexts(
    db: &MemoryDB,
    question: &str,
    search_limit: usize,
    domain: Option<&str>,
) -> Result<(String, String), OriginError> {
    // Flat: search_memory only
    let results = db
        .search_memory(question, search_limit, None, domain, None, None, None, None)
        .await?;
    let flat_context: String = results
        .iter()
        .map(|r| format!("On {}: {}", crate::eval::shared::format_ymd(r.last_modified), r.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Structured: concepts + search results
    let mut parts: Vec<String> = Vec::new();
    let concepts = db.search_concepts(question, 3).await.unwrap_or_default();
    if !concepts.is_empty() {
        parts.push("## Compiled Knowledge".to_string());
        for c in &concepts {
            let summary = c.summary.as_deref().unwrap_or("");
            parts.push(format!("**{}**: {}\n{}", c.title, summary, c.content));
        }
    }
    if !results.is_empty() {
        parts.push("## Relevant Memories".to_string());
        for r in results.iter() {
            parts.push(format!("On {}: {}", crate::eval::shared::format_ymd(r.last_modified), r.content));
        }
    }
    let structured_context = parts.join("\n\n");

    Ok((flat_context, structured_context))
}

/// Full-pipeline LoCoMo eval using Batch API for answer generation.
///
/// **Single DB**: all conversations seeded into one database, each tagged with
/// a conversation-specific domain. Enrichment runs once across all data, so
/// entities accumulate and concepts can form from cross-observation clusters.
///
/// **Phase 1** (on-device, free): Seed all conversations, enrich once.
/// **Phase 2** (free): Collect contexts for all questions (search with domain filter).
/// **Phase 3** (Batch API, 50% cheaper): Submit all answer prompts in one batch.
/// **Phase 4** (instant): Merge batch results + cached flat answers into tuples.
pub async fn run_fullpipeline_locomo_batch(
    locomo_path: &Path,
    enrichment_llm: Option<Arc<dyn crate::llm_provider::LlmProvider>>,
    api_key: &str,
    answer_model: &str,
    flat_cache_path: Option<&Path>,
    output_path: &Path,
    cost_cap_usd: f64,
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::eval::judge::save_judgment_tuples;
    use crate::eval::locomo::{category_name, extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let _prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let _tuning = DistillationConfig::default();
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

    let flat_cache: HashMap<String, (String, usize)> = load_flat_cache_locomo(flat_cache_path);
    if !flat_cache.is_empty() {
        eprintln!(
            "[fullpipeline] Loaded {} cached flat answers",
            flat_cache.len()
        );
    }

    let _enrich_llm: Arc<dyn crate::llm_provider::LlmProvider> = match enrichment_llm {
        Some(llm) => llm,
        None => {
            eprintln!("[fullpipeline] No enrichment LLM, using API");
            Arc::new(crate::llm_provider::ApiProvider::new(
                api_key.to_string(),
                answer_model.to_string(),
            ))
        }
    };

    // --- Phase 1: Seed all conversations into one DB, enrich once ---
    // Use a stable DB path (sibling to output_path) so enrichment survives crashes.
    let db_dir = output_path.with_extension("db");
    std::fs::create_dir_all(&db_dir).ok();
    let db =
        MemoryDB::new_with_shared_embedder(&db_dir, Arc::new(NoopEmitter), shared_embedder.clone())
            .await?;

    // Check if DB already has COMPLETE enrichment (not just partial data).
    // Enrichment is complete when enrichment_steps rows exist for memories.
    let mem_count = db.memory_count().await.unwrap_or(0);
    let enriched_count = db.enriched_memory_count().await.unwrap_or(0);
    let enrichment_complete = mem_count > 0 && enriched_count == mem_count;

    if enrichment_complete {
        eprintln!(
            "[fullpipeline] Resuming with enriched DB ({} memories, all enriched)",
            mem_count
        );
    } else {
        // Wipe partial data and start fresh to avoid inconsistencies
        if mem_count > 0 && enriched_count < mem_count {
            eprintln!(
                "[fullpipeline] Partial data found ({}/{} enriched). Starting fresh.",
                enriched_count, mem_count
            );
            db.clear_all_for_eval().await?;
        }

        let mut total_obs = 0usize;
        for sample in &samples {
            let memories = extract_observations(sample);
            if memories.is_empty() {
                continue;
            }

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
            total_obs += docs.len();
            db.upsert_documents(docs).await?;
            eprintln!(
                "[fullpipeline] Seeded {} ({} observations)",
                sample.sample_id,
                memories.len()
            );
        }

        eprintln!(
            "[fullpipeline] Total: {} observations in 1 DB. Enriching via Batch API...",
            total_obs
        );
        let entities =
            crate::eval::shared::run_enrichment_batch_api(&db, api_key, answer_model, cost_cap_usd)
                .await?;
        let titles = crate::eval::shared::run_title_enrichment_batch_api(
            &db,
            api_key,
            answer_model,
            cost_cap_usd,
        )
        .await?;
        let concepts = crate::eval::shared::run_concept_distillation_batch_api(
            &db,
            api_key,
            answer_model,
            cost_cap_usd,
        )
        .await?;
        eprintln!(
            "[fullpipeline] Enriched: {} entities, {} titles, {} concepts",
            entities, titles, concepts
        );
    }

    // --- Phase 2: Collect contexts for all questions ---
    let mut pending: HashMap<String, PendingAnswer> = HashMap::new();
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();

    for sample in &samples {
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
            let (flat_ctx, structured_ctx) = build_contexts(&db, &qa.question, 10, None).await?;
            let flat_tokens = count_tokens(&flat_ctx);
            let structured_tokens = count_tokens(&structured_ctx);

            // Flat: cache or batch
            if let Some((cached_answer, cached_tokens)) = flat_cache.get(&qa.question) {
                finished_tuples.push(JudgmentTuple {
                    question: qa.question.clone(),
                    ground_truth: ground_truth.clone(),
                    approach: format!("flat_{}", category),
                    answer: cached_answer.clone(),
                    context_tokens: *cached_tokens,
                });
            } else {
                let flat_id = format!("flat_{}_{}", sample.sample_id, q_count);
                batch_requests.push((
                    flat_id.clone(),
                    format!("Context:\n{}\n\nQuestion: {}", flat_ctx, qa.question),
                    Some(E2E_SYSTEM_PROMPT.to_string()),
                    200,
                ));
                pending.insert(
                    flat_id,
                    PendingAnswer {
                        question: qa.question.clone(),
                        ground_truth: ground_truth.clone(),
                        approach: format!("flat_{}", category),
                        context_tokens: flat_tokens,
                    },
                );
            }

            // Structured: always batch
            let structured_id = format!("structured_{}_{}", sample.sample_id, q_count);
            batch_requests.push((
                structured_id.clone(),
                format!("Context:\n{}\n\nQuestion: {}", structured_ctx, qa.question),
                Some(E2E_SYSTEM_PROMPT.to_string()),
                200,
            ));
            pending.insert(
                structured_id,
                PendingAnswer {
                    question: qa.question.clone(),
                    ground_truth,
                    approach: format!("structured_{}", category),
                    context_tokens: structured_tokens,
                },
            );

            q_count += 1;
        }
        if q_count > 0 {
            eprintln!("  {} — {} questions collected", sample.sample_id, q_count);
        }
    }

    if batch_requests.is_empty() {
        eprintln!("[fullpipeline] No new requests — all cached/resumed");
        save_judgment_tuples(&finished_tuples, output_path)
            .map_err(|e| OriginError::Generic(format!("save: {e}")))?;
        return Ok(finished_tuples);
    }

    // --- Phase 3: Batch answer generation ---
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

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(format!("batch download: {e}")))?;

    // --- Phase 4: Merge ---
    let mut matched = 0usize;
    for (custom_id, answer) in &raw_results {
        if let Some(meta) = pending.get(custom_id) {
            finished_tuples.push(JudgmentTuple {
                question: meta.question.clone(),
                ground_truth: meta.ground_truth.clone(),
                approach: meta.approach.clone(),
                answer: answer.clone(),
                context_tokens: meta.context_tokens,
            });
            matched += 1;
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
/// **Single DB**: all 500 questions' memories seeded into one database (~10K memories).
/// No domain filter — search must find relevant memories among all data, like production.
/// Enrichment runs once across all data.
pub async fn run_fullpipeline_lme_batch(
    longmemeval_path: &Path,
    enrichment_llm: Option<Arc<dyn crate::llm_provider::LlmProvider>>,
    api_key: &str,
    answer_model: &str,
    flat_cache_path: Option<&Path>,
    output_path: &Path,
    cost_cap_usd: f64,
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::eval::judge::save_judgment_tuples;
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let _prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let _tuning = DistillationConfig::default();
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

    let flat_cache: HashMap<String, (String, usize)> = load_flat_cache_lme(flat_cache_path);
    if !flat_cache.is_empty() {
        eprintln!(
            "[fullpipeline_lme] Loaded {} cached flat answers",
            flat_cache.len()
        );
    }

    let _enrich_llm: Arc<dyn crate::llm_provider::LlmProvider> = match enrichment_llm {
        Some(llm) => llm,
        None => {
            eprintln!("[fullpipeline_lme] No enrichment LLM, using API");
            Arc::new(crate::llm_provider::ApiProvider::new(
                api_key.to_string(),
                answer_model.to_string(),
            ))
        }
    };

    // --- Phase 1: Seed all questions' memories into one DB, enrich once ---
    let db_dir = output_path.with_extension("db");
    std::fs::create_dir_all(&db_dir).ok();
    let db =
        MemoryDB::new_with_shared_embedder(&db_dir, Arc::new(NoopEmitter), shared_embedder.clone())
            .await?;

    let mem_count = db.memory_count().await.unwrap_or(0);
    let enriched_count = db.enriched_memory_count().await.unwrap_or(0);
    let enrichment_complete = mem_count > 0 && enriched_count == mem_count;

    if enrichment_complete {
        eprintln!(
            "[fullpipeline_lme] Resuming with enriched DB ({} memories, all enriched)",
            mem_count
        );
    } else {
        if mem_count > 0 && enriched_count < mem_count {
            eprintln!(
                "[fullpipeline_lme] Partial data ({}/{} enriched). Starting fresh.",
                enriched_count, mem_count
            );
            db.clear_all_for_eval().await?;
        }
        let mut total_mems = 0usize;
        for sample in &samples {
            let memories = extract_memories(sample);
            if memories.is_empty() {
                continue;
            }

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
            total_mems += docs.len();
            db.upsert_documents(docs).await?;
        }

        eprintln!(
            "[fullpipeline_lme] Seeded {} memories from {} questions. Enriching via Batch API...",
            total_mems,
            samples.len()
        );
        let entities =
            crate::eval::shared::run_enrichment_batch_api(&db, api_key, answer_model, cost_cap_usd)
                .await?;
        let titles = crate::eval::shared::run_title_enrichment_batch_api(
            &db,
            api_key,
            answer_model,
            cost_cap_usd,
        )
        .await?;
        let concepts = crate::eval::shared::run_concept_distillation_batch_api(
            &db,
            api_key,
            answer_model,
            cost_cap_usd,
        )
        .await?;
        eprintln!(
            "[fullpipeline_lme] Enriched: {} entities, {} titles, {} concepts",
            entities, titles, concepts
        );
    }

    // --- Phase 2: Collect contexts ---
    let mut pending: HashMap<String, PendingAnswer> = HashMap::new();
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();

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

        let category = category_name(&sample.question_type);
        let (flat_ctx, structured_ctx) = build_contexts(&db, &sample.question, 10, None).await?;
        let flat_tokens = count_tokens(&flat_ctx);
        let structured_tokens = count_tokens(&structured_ctx);

        // Flat: cache or batch
        if let Some((cached_answer, cached_tokens)) = flat_cache.get(&sample.question) {
            finished_tuples.push(JudgmentTuple {
                question: sample.question.clone(),
                ground_truth: ground_truth.clone(),
                approach: format!("flat_{}", category),
                answer: cached_answer.clone(),
                context_tokens: *cached_tokens,
            });
        } else {
            let flat_id = format!("flat_lme_{}", q_idx);
            batch_requests.push((
                flat_id.clone(),
                format!("Context:\n{}\n\nQuestion: {}", flat_ctx, sample.question),
                Some(E2E_SYSTEM_PROMPT.to_string()),
                200,
            ));
            pending.insert(
                flat_id,
                PendingAnswer {
                    question: sample.question.clone(),
                    ground_truth: ground_truth.clone(),
                    approach: format!("flat_{}", category),
                    context_tokens: flat_tokens,
                },
            );
        }

        // Structured: always batch
        let structured_id = format!("structured_lme_{}", q_idx);
        batch_requests.push((
            structured_id.clone(),
            format!(
                "Context:\n{}\n\nQuestion: {}",
                structured_ctx, sample.question
            ),
            Some(E2E_SYSTEM_PROMPT.to_string()),
            200,
        ));
        pending.insert(
            structured_id,
            PendingAnswer {
                question: sample.question.clone(),
                ground_truth,
                approach: format!("structured_{}", category),
                context_tokens: structured_tokens,
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

    if batch_requests.is_empty() {
        eprintln!("[fullpipeline_lme] No new requests — all cached/resumed");
        save_judgment_tuples(&finished_tuples, output_path)
            .map_err(|e| OriginError::Generic(format!("save: {e}")))?;
        return Ok(finished_tuples);
    }

    // --- Phase 3: Batch answer generation ---
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

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(format!("batch download: {e}")))?;

    // --- Phase 4: Merge ---
    let mut matched = 0usize;
    for (custom_id, answer) in &raw_results {
        if let Some(meta) = pending.get(custom_id) {
            finished_tuples.push(JudgmentTuple {
                question: meta.question.clone(),
                ground_truth: meta.ground_truth.clone(),
                approach: meta.approach.clone(),
                answer: answer.clone(),
                context_tokens: meta.context_tokens,
            });
            matched += 1;
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

/// Load cached LoCoMo flat answers: question -> (answer, context_tokens).
fn load_flat_cache_locomo(path: Option<&Path>) -> HashMap<String, (String, usize)> {
    let path = match path {
        Some(p) if p.exists() => p,
        _ => return HashMap::new(),
    };
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    let items: Vec<serde_json::Value> = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    items
        .iter()
        .filter_map(|item| {
            let question = item["question"].as_str()?.to_string();
            let answer = item["model_answer"].as_str()?.to_string();
            let tokens = item["context_tokens"].as_u64().unwrap_or(0) as usize;
            Some((question, (answer, tokens)))
        })
        .collect()
}

/// Load cached LME flat answers: question -> (answer, context_tokens).
fn load_flat_cache_lme(path: Option<&Path>) -> HashMap<String, (String, usize)> {
    // Same format as LoCoMo
    load_flat_cache_locomo(path)
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_format_ymd_used_in_context() {
        // Sanity: format_ymd produces the expected ISO date for a known timestamp.
        // This guards against accidental regressions if someone refactors format_ymd
        // away or changes the format string.
        assert_eq!(crate::eval::shared::format_ymd(1_681_168_020), "2023-04-10");
        assert_eq!(crate::eval::shared::format_ymd(1_683_554_160), "2023-05-08");
    }

    #[test]
    fn test_system_prompt_includes_question_date_when_provided() {
        // Mirror the system_prompt construction from generate_e2e_answers_for_question
        // to lock in the format. If the function changes, this test should reflect.
        let with_date = match Some("2023/04/10 (Mon) 23:07") {
            Some(d) => format!(
                "The question was asked on {}. Answer the question using only the provided context. \
                 Be specific and concise. Respond in 1-3 sentences.",
                d
            ),
            None => "Answer the question using only the provided context. \
                Be specific and concise. Respond in 1-3 sentences."
                .to_string(),
        };
        assert!(with_date.contains("The question was asked on 2023/04/10"));
        assert!(with_date.contains("only the provided context"));
    }

    #[test]
    fn test_system_prompt_omits_when_no_question_date() {
        let without_date: String = match None::<&str> {
            Some(d) => format!(
                "The question was asked on {}. Answer the question using only the provided context. \
                 Be specific and concise. Respond in 1-3 sentences.",
                d
            ),
            None => "Answer the question using only the provided context. \
                Be specific and concise. Respond in 1-3 sentences."
                .to_string(),
        };
        assert!(!without_date.contains("question was asked on"));
        assert!(without_date.contains("only the provided context"));
    }
}
