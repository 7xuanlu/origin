// SPDX-License-Identifier: AGPL-3.0-only
//! Context path evaluation: recall-only vs search + concepts + graph coverage.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::shared::{
    count_tokens, eval_shared_embedder, run_entity_extraction_for_eval,
    run_title_enrichment_for_eval,
};
use crate::events::NoopEmitter;
use crate::sources::RawDocument;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Count total tokens across a slice of SearchResults (content field only).
fn count_results_tokens(results: &[crate::db::SearchResult]) -> usize {
    results.iter().map(|r| count_tokens(&r.content)).sum()
}

// ===== Context Path Eval: recall vs context coverage comparison =====

/// Per-question result comparing recall (search_memory) vs context (search + concepts + graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPathResult {
    pub question: String,
    pub category: String,
    /// Evidence source_ids found by search_memory alone.
    pub recall_found: usize,
    /// Evidence source_ids found by search_memory + concepts + graph.
    pub context_found: usize,
    /// Total evidence source_ids for this question.
    pub total_evidence: usize,
    pub recall_coverage: f64,
    pub context_coverage: f64,
    /// Source_ids recovered by context that recall missed.
    pub recovered_ids: Vec<String>,
    /// Tokens: recall context vs full context.
    pub recall_tokens: usize,
    pub context_tokens: usize,
}

/// Aggregate report for context path eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPathReport {
    pub benchmark: String,
    pub total_questions: usize,
    pub mean_recall_coverage: f64,
    pub mean_context_coverage: f64,
    pub coverage_delta: f64,
    pub questions_improved: usize,
    pub total_evidence_recovered: usize,
    pub mean_recall_tokens: f64,
    pub mean_context_tokens: f64,
    pub per_category: Vec<ContextPathCategoryResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPathCategoryResult {
    pub category: String,
    pub count: usize,
    pub mean_recall_coverage: f64,
    pub mean_context_coverage: f64,
    pub delta: f64,
}

impl ContextPathReport {
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Context Path Eval: {} ({} questions)\n\n",
            self.benchmark, self.total_questions
        ));
        out.push_str(&format!(
            "{:<20} | {:<10} | {:<10} | {:<8} | {:<8}\n",
            "Path", "Coverage", "Tok/Q", "Improved", "Recovered"
        ));
        out.push_str(&format!(
            "{:-<20}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<8}\n",
            "", "", "", "", ""
        ));
        out.push_str(&format!(
            "{:<20} | {:<10.4} | {:<10.1} | {:<8} | {:<8}\n",
            "recall (search only)", self.mean_recall_coverage, self.mean_recall_tokens, "-", "-"
        ));
        out.push_str(&format!(
            "{:<20} | {:<10.4} | {:<10.1} | {:<8} | {:<8}\n",
            "context (full)",
            self.mean_context_coverage,
            self.mean_context_tokens,
            self.questions_improved,
            self.total_evidence_recovered
        ));
        out.push_str(&format!(
            "\nCoverage delta: {:+.4} ({:.1}% relative improvement)\n",
            self.coverage_delta,
            if self.mean_recall_coverage > 0.0 {
                self.coverage_delta / self.mean_recall_coverage * 100.0
            } else {
                0.0
            }
        ));

        if !self.per_category.is_empty() {
            out.push_str("\nPer category:\n");
            for cat in &self.per_category {
                out.push_str(&format!(
                    "  {} (n={}): recall={:.3} context={:.3} delta={:+.3}\n",
                    cat.category,
                    cat.count,
                    cat.mean_recall_coverage,
                    cat.mean_context_coverage,
                    cat.delta,
                ));
            }
        }

        out
    }
}

/// Compare recall (search_memory only) vs context (search + concepts + graph) on LoCoMo.
///
/// Seeds one conversation at a time, runs enrichment + distillation, then for each question:
/// 1. recall path: search_memory top-K, check evidence coverage
/// 2. context path: search_memory top-K + search_concepts top-3 source_ids, check coverage
///
/// Reports coverage delta: how many evidence items does the context path recover
/// that recall alone misses.
///
/// Requires on-device LLM for enrichment/distillation. Run with sandbox disabled.
pub async fn run_context_path_eval(
    locomo_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_conversations: usize,
) -> Result<ContextPathReport, OriginError> {
    use crate::eval::locomo::{category_name, extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    let mut all_results: Vec<ContextPathResult> = Vec::new();
    let conv_limit = max_conversations.min(samples.len());

    // Pre-create shared embedder so each conversation reuses the loaded model.
    let shared_embedder = eval_shared_embedder();

    for (conv_idx, sample) in samples.iter().take(max_conversations).enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }

        eprintln!(
            "[context_eval] Conv {}/{} ({}): {} observations",
            conv_idx + 1,
            conv_limit,
            sample.sample_id,
            memories.len(),
        );

        // Build evidence mapping
        let dia_to_source: HashMap<String, String> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                (
                    m.dia_id.clone(),
                    format!("locomo_{}_obs_{}", sample.sample_id, i),
                )
            })
            .collect();

        // Seed DB
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir context_eval: {e}")))?;
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

        // Run enrichment + distillation (entity extraction → titles → distill)
        eprintln!("  [enriching] entity extraction...");
        let entities = run_entity_extraction_for_eval(&db, &llm).await?;
        eprintln!("  [enriching] {} entities. enriching titles...", entities);

        let titles = run_title_enrichment_for_eval(&db, &llm).await?;
        eprintln!("  [enriching] {} titles. distilling...", titles);

        let concepts =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        eprintln!("  [enriched] {} concepts. evaluating...", concepts);

        // Evaluate each question
        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }

            let evidence_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if evidence_ids.is_empty() {
                continue;
            }

            // --- Recall path: search_memory only ---
            let recall_results = db
                .search_memory(
                    &qa.question,
                    search_limit,
                    None,
                    Some("conversation"),
                    None,
                    None,
                    None,
                    None,
                )
                .await?;
            let recall_ids: HashSet<String> =
                recall_results.iter().map(|r| r.source_id.clone()).collect();
            let recall_tokens = count_results_tokens(&recall_results);

            let recall_found = evidence_ids.intersection(&recall_ids).count();

            // --- Context path: search_memory + search_concepts ---
            let mut context_ids = recall_ids.clone();
            let mut context_tokens = recall_tokens;

            // Add concept source_ids
            let concept_results = db
                .search_concepts(&qa.question, 3)
                .await
                .unwrap_or_default();
            for concept in &concept_results {
                context_tokens += count_tokens(&concept.content);
                // Get source memories via concept_sources join table
                let sources = db
                    .get_concept_sources(&concept.id)
                    .await
                    .unwrap_or_default();
                for src in &sources {
                    context_ids.insert(src.memory_source_id.clone());
                }
                // Also use the legacy source_memory_ids field
                for sid in &concept.source_memory_ids {
                    context_ids.insert(sid.clone());
                }
            }

            let context_found = evidence_ids.intersection(&context_ids).count();
            let recovered: Vec<String> = evidence_ids
                .iter()
                .filter(|id| context_ids.contains(*id) && !recall_ids.contains(*id))
                .cloned()
                .collect();

            all_results.push(ContextPathResult {
                question: qa.question.clone(),
                category: category_name(qa.category).to_string(),
                recall_found,
                context_found,
                total_evidence: evidence_ids.len(),
                recall_coverage: recall_found as f64 / evidence_ids.len() as f64,
                context_coverage: context_found as f64 / evidence_ids.len() as f64,
                recovered_ids: recovered,
                recall_tokens,
                context_tokens,
            });
        }

        let conv_results: Vec<&ContextPathResult> = all_results
            .iter()
            .rev()
            .take_while(|r| !r.question.is_empty()) // all from this conv
            .collect();
        let improved = conv_results
            .iter()
            .filter(|r| r.context_found > r.recall_found)
            .count();
        eprintln!(
            "  {} questions, {} improved by context path",
            conv_results.len(),
            improved,
        );
    }

    // Aggregate
    let n = all_results.len().max(1) as f64;
    let mean_recall_cov = all_results.iter().map(|r| r.recall_coverage).sum::<f64>() / n;
    let mean_context_cov = all_results.iter().map(|r| r.context_coverage).sum::<f64>() / n;
    let questions_improved = all_results
        .iter()
        .filter(|r| r.context_found > r.recall_found)
        .count();
    let total_recovered: usize = all_results.iter().map(|r| r.recovered_ids.len()).sum();
    let mean_recall_tok = all_results
        .iter()
        .map(|r| r.recall_tokens as f64)
        .sum::<f64>()
        / n;
    let mean_context_tok = all_results
        .iter()
        .map(|r| r.context_tokens as f64)
        .sum::<f64>()
        / n;

    // Per-category
    let mut cat_map: HashMap<String, Vec<&ContextPathResult>> = HashMap::new();
    for r in &all_results {
        cat_map.entry(r.category.clone()).or_default().push(r);
    }
    let mut per_category: Vec<ContextPathCategoryResult> = cat_map
        .iter()
        .map(|(cat, results)| {
            let cn = results.len().max(1) as f64;
            let rc = results.iter().map(|r| r.recall_coverage).sum::<f64>() / cn;
            let cc = results.iter().map(|r| r.context_coverage).sum::<f64>() / cn;
            ContextPathCategoryResult {
                category: cat.clone(),
                count: results.len(),
                mean_recall_coverage: rc,
                mean_context_coverage: cc,
                delta: cc - rc,
            }
        })
        .collect();
    per_category.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ContextPathReport {
        benchmark: "LoCoMo".to_string(),
        total_questions: all_results.len(),
        mean_recall_coverage: mean_recall_cov,
        mean_context_coverage: mean_context_cov,
        coverage_delta: mean_context_cov - mean_recall_cov,
        questions_improved,
        total_evidence_recovered: total_recovered,
        mean_recall_tokens: mean_recall_tok,
        mean_context_tokens: mean_context_tok,
        per_category,
    })
}

// ===== Context Path Eval: LongMemEval =====

/// Same as run_context_path_eval but for LongMemEval dataset.
/// Each question has its own memory set. Evidence = memories from answer_session_ids.
pub async fn run_context_path_eval_longmemeval(
    longmemeval_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_questions: usize,
) -> Result<ContextPathReport, OriginError> {
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Pre-create shared embedder — loads 140MB ONNX model once instead of per-question
    eprintln!("[context_path_lme] loading shared embedder...");
    let shared_embedder = eval_shared_embedder();
    eprintln!(
        "[context_path_lme] embedder ready, processing {} questions",
        samples.len().min(max_questions)
    );

    let mut all_results: Vec<ContextPathResult> = Vec::new();
    let sample_limit = max_questions.min(samples.len());

    for (q_idx, sample) in samples.iter().take(max_questions).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }

        if q_idx % 25 == 0 {
            eprintln!(
                "[context_path_lme] Q {}/{} ({}): {} memories{}",
                q_idx + 1,
                sample_limit,
                sample.question_id,
                memories.len(),
                if memories.len() < 15 {
                    " (skip distill)"
                } else {
                    ""
                },
            );
        }

        // Build evidence mapping: memories from answer sessions are evidence
        let answer_session_set: HashSet<String> =
            sample.answer_session_ids.iter().cloned().collect();

        let evidence_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| answer_session_set.contains(&m.session_id))
            .map(|m| {
                format!(
                    "lme_{}_{}_t{}",
                    sample.question_id, m.session_idx, m.turn_idx
                )
            })
            .collect();

        if evidence_source_ids.is_empty() {
            continue;
        }

        // Seed DB with shared embedder (skip 10-30s model reload per question)
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir ctx_lme: {e}")))?;
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

        // Only distill if enough memories to form meaningful clusters (min 15).
        // LME has avg 11 memories per question — most won't produce concepts.
        // Each distillation call takes ~30s (LLM inference), so skipping saves hours.
        if memories.len() >= 15 {
            let _concepts =
                crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        }

        // --- Recall path ---
        let recall_results = db
            .search_memory(
                &sample.question,
                search_limit,
                None,
                Some("conversation"),
                None,
                None,
                None,
                None,
            )
            .await?;
        let recall_ids: HashSet<String> =
            recall_results.iter().map(|r| r.source_id.clone()).collect();
        let recall_tokens = count_results_tokens(&recall_results);
        let recall_found = evidence_source_ids.intersection(&recall_ids).count();

        // --- Context path ---
        let mut context_ids = recall_ids.clone();
        let mut context_tokens = recall_tokens;

        let concept_results = db
            .search_concepts(&sample.question, 3)
            .await
            .unwrap_or_default();
        for concept in &concept_results {
            context_tokens += count_tokens(&concept.content);
            let sources = db
                .get_concept_sources(&concept.id)
                .await
                .unwrap_or_default();
            for src in &sources {
                context_ids.insert(src.memory_source_id.clone());
            }
            for sid in &concept.source_memory_ids {
                context_ids.insert(sid.clone());
            }
        }

        let context_found = evidence_source_ids.intersection(&context_ids).count();
        let recovered: Vec<String> = evidence_source_ids
            .iter()
            .filter(|id| context_ids.contains(*id) && !recall_ids.contains(*id))
            .cloned()
            .collect();

        let category = category_name(&sample.question_type);

        all_results.push(ContextPathResult {
            question: sample.question.clone(),
            category: category.to_string(),
            recall_found,
            context_found,
            total_evidence: evidence_source_ids.len(),
            recall_coverage: recall_found as f64 / evidence_source_ids.len() as f64,
            context_coverage: context_found as f64 / evidence_source_ids.len() as f64,
            recovered_ids: recovered,
            recall_tokens,
            context_tokens,
        });

        if q_idx % 50 == 49 {
            let improved = all_results
                .iter()
                .filter(|r| r.context_found > r.recall_found)
                .count();
            eprintln!(
                "  [progress] {}/{} questions, {} improved",
                q_idx + 1,
                sample_limit,
                improved,
            );
        }
    }

    // Aggregate
    let n = all_results.len().max(1) as f64;
    let mean_recall_cov = all_results.iter().map(|r| r.recall_coverage).sum::<f64>() / n;
    let mean_context_cov = all_results.iter().map(|r| r.context_coverage).sum::<f64>() / n;
    let questions_improved = all_results
        .iter()
        .filter(|r| r.context_found > r.recall_found)
        .count();
    let total_recovered: usize = all_results.iter().map(|r| r.recovered_ids.len()).sum();
    let mean_recall_tok = all_results
        .iter()
        .map(|r| r.recall_tokens as f64)
        .sum::<f64>()
        / n;
    let mean_context_tok = all_results
        .iter()
        .map(|r| r.context_tokens as f64)
        .sum::<f64>()
        / n;

    // Per-category
    let mut cat_map: HashMap<String, Vec<&ContextPathResult>> = HashMap::new();
    for r in &all_results {
        cat_map.entry(r.category.clone()).or_default().push(r);
    }
    let mut per_category: Vec<ContextPathCategoryResult> = cat_map
        .iter()
        .map(|(cat, results)| {
            let cn = results.len().max(1) as f64;
            let rc = results.iter().map(|r| r.recall_coverage).sum::<f64>() / cn;
            let cc = results.iter().map(|r| r.context_coverage).sum::<f64>() / cn;
            ContextPathCategoryResult {
                category: cat.clone(),
                count: results.len(),
                mean_recall_coverage: rc,
                mean_context_coverage: cc,
                delta: cc - rc,
            }
        })
        .collect();
    per_category.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ContextPathReport {
        benchmark: "LongMemEval".to_string(),
        total_questions: all_results.len(),
        mean_recall_coverage: mean_recall_cov,
        mean_context_coverage: mean_context_cov,
        coverage_delta: mean_context_cov - mean_recall_cov,
        questions_improved,
        total_evidence_recovered: total_recovered,
        mean_recall_tokens: mean_recall_tok,
        mean_context_tokens: mean_context_tok,
        per_category,
    })
}
