// SPDX-License-Identifier: AGPL-3.0-only
//! Pipeline evaluation: compare Flat vs Enriched vs Distilled conditions.

use super::retrieval::SearchStrategy;
use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::metrics;
use crate::eval::shared::{count_tokens, eval_shared_embedder, run_entity_extraction_for_eval};
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

// ===== Pipeline Eval: LoCoMo + LongMemEval through Origin's full pipeline =====

/// Pipeline condition: what processing has been applied to the DB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineCondition {
    /// Raw observations seeded via upsert_documents, no enrichment.
    Flat,
    /// Flat + entity extraction (auto-linking, entity creation).
    Enriched,
    /// Enriched + distillation via real LLM (merged concept memories).
    Distilled,
}

impl PipelineCondition {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Enriched => "enriched",
            Self::Distilled => "distilled",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Flat => "Flat (raw)",
            Self::Enriched => "Enriched (entities)",
            Self::Distilled => "Distilled (LLM concepts)",
        }
    }
}

/// Metrics for one (condition, strategy) cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCellMetrics {
    pub condition: String,
    pub strategy: String,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub mean_context_tokens: f64,
    pub corpus_tokens: usize,
    pub memory_count: usize,
    /// NDCG per 1K context tokens.
    pub information_density: f64,
    pub queries_evaluated: usize,
}

/// Per-conversation/question result showing all conditions x strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConversationResult {
    pub id: String,
    pub observations_seeded: usize,
    pub queries_evaluated: usize,
    pub distillation_clusters_found: usize,
    pub distilled_concepts_created: usize,
    pub cells: Vec<PipelineCellMetrics>,
}

/// Full pipeline eval report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineBenchmarkReport {
    pub benchmark: String,
    pub timestamp: String,
    pub conversations: usize,
    pub total_queries: usize,
    pub total_observations: usize,
    pub llm_model: String,
    /// Aggregate metrics across all conversations/questions.
    pub aggregate: Vec<PipelineCellMetrics>,
    pub per_conversation: Vec<PipelineConversationResult>,
}

impl PipelineBenchmarkReport {
    /// Format as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Pipeline Eval: {} ({})\n",
            self.benchmark, self.timestamp
        ));
        out.push_str(&format!(
            "LLM: {} | Conversations: {} | Queries: {} | Observations: {}\n\n",
            self.llm_model, self.conversations, self.total_queries, self.total_observations
        ));

        out.push_str(&format!(
            "{:<24} | {:<14} | {:<8} | {:<8} | {:<8} | {:<10} | {:<8}\n",
            "Condition x Strategy", "NDCG@10", "MRR", "R@5", "Tok/Q", "Corpus", "Density"
        ));
        out.push_str(&format!(
            "{:-<24}-+-{:-<14}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<8}\n",
            "", "", "", "", "", "", ""
        ));

        for cell in &self.aggregate {
            out.push_str(&format!(
                "{:<24} | {:<14.4} | {:<8.4} | {:<8.4} | {:<8.1} | {:<10} | {:<8.3}\n",
                format!("{} / {}", cell.condition, cell.strategy),
                cell.ndcg_at_10,
                cell.mrr,
                cell.recall_at_5,
                cell.mean_context_tokens,
                cell.corpus_tokens,
                cell.information_density,
            ));
        }

        // Distillation summary
        let total_clusters: usize = self
            .per_conversation
            .iter()
            .map(|c| c.distillation_clusters_found)
            .sum();
        let total_concepts: usize = self
            .per_conversation
            .iter()
            .map(|c| c.distilled_concepts_created)
            .sum();
        out.push_str(&format!(
            "\nDistillation: {} clusters found, {} concepts created\n",
            total_clusters, total_concepts
        ));

        out
    }
}

/// Search strategies to test in pipeline eval (subset -- no LLM-requiring ones).
const PIPELINE_STRATEGIES: &[SearchStrategy] = &[
    SearchStrategy::Origin,
    SearchStrategy::NaiveRag,
    SearchStrategy::FtsOnly,
    SearchStrategy::VectorPlusFts,
];

/// Run search for a given strategy and return (result_ids, context_tokens).
async fn run_strategy_search(
    db: &MemoryDB,
    query: &str,
    strategy: SearchStrategy,
    limit: usize,
    domain: Option<&str>,
) -> Result<(Vec<String>, usize), OriginError> {
    let results = match strategy {
        SearchStrategy::Origin => {
            db.search_memory(query, limit, None, domain, None, Some(1.0), Some(1.0), None)
                .await?
        }
        SearchStrategy::NaiveRag => db.naive_vector_search(query, limit, domain).await?,
        SearchStrategy::FtsOnly => db.fts_only_search(query, limit, domain).await?,
        SearchStrategy::VectorPlusFts => db.vector_plus_fts_search(query, limit, domain).await?,
        _ => return Ok((vec![], 0)),
    };
    let ids: Vec<String> = results.iter().map(|r| r.source_id.clone()).collect();
    let tokens = count_results_tokens(&results);
    Ok((ids, tokens))
}

/// Evaluate all strategies for a set of QA pairs against a DB, returning per-cell metrics.
///
/// `relevance_map` maps source_id -> relevance grade (0-3).
/// `evidence_sets` maps question index -> set of relevant source_ids.
#[allow(clippy::too_many_arguments)]
async fn evaluate_condition<Q: AsRef<str>>(
    db: &MemoryDB,
    condition: PipelineCondition,
    questions: &[Q],
    evidence_sets: &[HashSet<String>],
    relevance_map: &HashMap<String, u8>,
    corpus_tokens: usize,
    memory_count: usize,
    limit: usize,
    domain: Option<&str>,
) -> Result<Vec<PipelineCellMetrics>, OriginError> {
    let mut cells = Vec::new();
    let total_strategies = PIPELINE_STRATEGIES.len();

    for (si, &strategy) in PIPELINE_STRATEGIES.iter().enumerate() {
        eprintln!(
            "    [{}/{}] {} / {} ({} queries)...",
            condition.name(),
            strategy.name(),
            si + 1,
            total_strategies,
            questions.len(),
        );
        let mut ndcg_vals = Vec::new();
        let mut mrr_vals = Vec::new();
        let mut recall_vals = Vec::new();
        let mut ctx_tokens_vals = Vec::new();

        for (qi, question) in questions.iter().enumerate() {
            let (result_ids, ctx_tokens) =
                run_strategy_search(db, question.as_ref(), strategy, limit, domain).await?;

            let result_refs: Vec<&str> = result_ids.iter().map(|s| s.as_str()).collect();

            // Build grades for this query's results
            let grades: HashMap<&str, u8> = result_refs
                .iter()
                .map(|id| (*id, *relevance_map.get(*id).unwrap_or(&0)))
                .collect();

            let relevant_set: HashSet<&str> =
                evidence_sets[qi].iter().map(|s| s.as_str()).collect();

            ndcg_vals.push(metrics::ndcg_at_k(&result_refs, &grades, 10));
            mrr_vals.push(metrics::mrr(&result_refs, &relevant_set));
            recall_vals.push(metrics::recall_at_k(&result_refs, &relevant_set, 5));
            ctx_tokens_vals.push(ctx_tokens as f64);
        }

        let n = ndcg_vals.len().max(1) as f64;
        let mean_ndcg = ndcg_vals.iter().sum::<f64>() / n;
        let mean_mrr = mrr_vals.iter().sum::<f64>() / n;
        let mean_recall = recall_vals.iter().sum::<f64>() / n;
        let mean_ctx = ctx_tokens_vals.iter().sum::<f64>() / n;
        let density = if mean_ctx > 0.0 {
            mean_ndcg / (mean_ctx / 1000.0)
        } else {
            0.0
        };

        cells.push(PipelineCellMetrics {
            condition: condition.name().to_string(),
            strategy: strategy.name().to_string(),
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_recall,
            mean_context_tokens: mean_ctx,
            corpus_tokens,
            memory_count,
            information_density: density,
            queries_evaluated: questions.len(),
        });
    }

    Ok(cells)
}

/// Count corpus tokens from all memories in a DB.
async fn count_corpus_tokens(db: &MemoryDB) -> Result<(usize, usize), OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT content FROM memories WHERE chunk_index = 0 \
             AND supersede_mode <> 'archive'",
            (),
        )
        .await
        .map_err(|e| OriginError::Generic(format!("count_corpus: {e}")))?;

    let mut total_tokens = 0usize;
    let mut count = 0usize;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::Generic(e.to_string()))?
    {
        let content: String = row
            .get(0)
            .map_err(|e| OriginError::Generic(e.to_string()))?;
        total_tokens += count_tokens(&content);
        count += 1;
    }
    Ok((total_tokens, count))
}

/// Expand evidence sets to include merged/distilled source_ids.
///
/// Expand evidence sets to include concept source_ids via the `concept_sources` join table.
///
/// After distillation, concepts link to their source memories via `concept_sources` (PR4).
/// For each evidence source_id, check if any concept consumed it. If so, add the concept's
/// merged memory source_id to the evidence set so NDCG credits distilled retrieval.
async fn expand_evidence_for_distillation(
    db: &MemoryDB,
    original_evidence: &[HashSet<String>],
) -> Result<Vec<HashSet<String>>, OriginError> {
    let conn = db.conn.lock().await;

    // Build reverse map: memory_source_id -> [concept merged source_ids]
    // Uses the concept_sources join table (PR4) for precise lineage tracking.
    let mut rows = conn
        .query(
            "SELECT cs.memory_source_id, m.source_id AS concept_mem_sid \
             FROM concept_sources cs \
             JOIN concepts c ON cs.concept_id = c.id \
             JOIN memories m ON m.source_id LIKE 'merged_%' \
               AND m.chunk_index = 0 \
               AND m.entity_id = c.entity_id \
             WHERE c.status = 'active'",
            (),
        )
        .await
        .map_err(|e| OriginError::Generic(format!("expand_evidence: {e}")))?;

    let mut mem_to_concepts: HashMap<String, Vec<String>> = HashMap::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::Generic(e.to_string()))?
    {
        let memory_sid: String = row
            .get(0)
            .map_err(|e| OriginError::Generic(e.to_string()))?;
        let concept_sid: String = row
            .get(1)
            .map_err(|e| OriginError::Generic(e.to_string()))?;
        mem_to_concepts
            .entry(memory_sid)
            .or_default()
            .push(concept_sid);
    }
    drop(rows);
    drop(conn);

    let mut expanded: Vec<HashSet<String>> = original_evidence.to_vec();
    for evidence_set in &mut expanded {
        let mut additions: Vec<String> = Vec::new();
        for evidence_id in evidence_set.iter() {
            if let Some(concept_sids) = mem_to_concepts.get(evidence_id) {
                additions.extend(concept_sids.iter().cloned());
            }
        }
        for a in additions {
            evidence_set.insert(a);
        }
    }

    Ok(expanded)
}

/// Run LoCoMo through Origin's full pipeline: flat, enriched, distilled.
///
/// For each conversation:
/// 1. Flat: seed observations, measure retrieval quality + tokens
/// 2. Enriched: run entity extraction on the same DB
/// 3. Distilled: run `distill_concepts()` with real LLM, measure again
///
/// Requires on-device LLM (Metal GPU). Run with sandbox disabled.
pub async fn run_locomo_pipeline_eval(
    locomo_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_conversations: usize,
) -> Result<PipelineBenchmarkReport, OriginError> {
    use crate::eval::locomo::{extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    let mut per_conversation = Vec::new();
    let mut total_queries = 0usize;
    let mut total_observations = 0usize;

    // Per-cell accumulators: (condition, strategy) -> Vec<(ndcg, mrr, recall, ctx_tokens)>
    #[allow(clippy::type_complexity)]
    let mut accumulators: HashMap<(String, String), Vec<(f64, f64, f64, f64, usize, usize)>> =
        HashMap::new();

    let conv_limit = max_conversations.min(samples.len());

    // Pre-create shared embedder so each conversation reuses the loaded model.
    let shared_embedder = eval_shared_embedder();

    for (conv_idx, sample) in samples.iter().take(max_conversations).enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }
        let obs_count = memories.len();
        total_observations += obs_count;

        eprintln!(
            "[pipeline] Conv {}/{} ({}): {} observations",
            conv_idx + 1,
            conv_limit,
            sample.sample_id,
            obs_count,
        );

        // Build evidence mapping: dia_id -> source_id
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

        // Prepare questions and evidence sets (non-adversarial only)
        let mut questions: Vec<String> = Vec::new();
        let mut evidence_sets: Vec<HashSet<String>> = Vec::new();
        let mut relevance_map: HashMap<String, u8> = HashMap::new();

        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }
            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }

            // All seeded source_ids get a relevance grade
            for (idx, _mem) in memories.iter().enumerate() {
                let sid = format!("locomo_{}_obs_{}", sample.sample_id, idx);
                let grade = if relevant_ids.contains(&sid) { 1u8 } else { 0 };
                // Use max in case a source_id is evidence for multiple questions
                let entry = relevance_map.entry(sid).or_insert(0);
                *entry = (*entry).max(grade);
            }

            questions.push(qa.question.clone());
            evidence_sets.push(relevant_ids);
        }

        if questions.is_empty() {
            continue;
        }
        total_queries += questions.len();

        // ---- Create DB and seed flat ----
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir pipeline: {e}")))?;
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

        // ---- Condition 1: Flat ----
        eprintln!("  [flat] evaluating {} queries...", questions.len());
        let (flat_corpus_tokens, flat_mem_count) = count_corpus_tokens(&db).await?;
        let flat_cells = evaluate_condition(
            &db,
            PipelineCondition::Flat,
            &questions,
            &evidence_sets,
            &relevance_map,
            flat_corpus_tokens,
            flat_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Condition 2: Enriched (entity extraction) ----
        eprintln!("  [enriched] running entity extraction...");
        // Run entity extraction via refinery's extract functions.
        // We use the LLM to extract entities from each memory.
        let extract_count = run_entity_extraction_for_eval(&db, &llm).await?;
        eprintln!("    extracted {} entities", extract_count);

        let (enriched_corpus_tokens, enriched_mem_count) = count_corpus_tokens(&db).await?;

        // Rebuild relevance map to include any new source_ids from entity extraction
        // (entity extraction doesn't create new memories, just links, so map is unchanged)
        let enriched_cells = evaluate_condition(
            &db,
            PipelineCondition::Enriched,
            &questions,
            &evidence_sets,
            &relevance_map,
            enriched_corpus_tokens,
            enriched_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Condition 3: Distilled (real LLM) ----
        eprintln!("  [distilled] running distillation...");
        let distilled_count =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        eprintln!("    distilled {} concepts", distilled_count);

        // Count clusters that were found (for reporting)
        let clusters = db
            .find_distillation_clusters(
                tuning.similarity_threshold,
                tuning.concept_min_cluster_size,
                tuning.max_clusters_per_steep,
                llm.synthesis_token_limit(),
                tuning.concept_min_cluster_size,
            )
            .await
            .unwrap_or_default();

        let (distilled_corpus_tokens, distilled_mem_count) = count_corpus_tokens(&db).await?;

        // Expand evidence sets to credit merged memories
        let expanded_evidence = expand_evidence_for_distillation(&db, &evidence_sets).await?;

        // Rebuild relevance map to include merged source_ids
        let mut distilled_relevance = relevance_map.clone();
        for evidence_set in &expanded_evidence {
            for id in evidence_set {
                if id.starts_with("merged_") {
                    distilled_relevance.entry(id.clone()).or_insert(1);
                }
            }
        }

        let distilled_cells = evaluate_condition(
            &db,
            PipelineCondition::Distilled,
            &questions,
            &expanded_evidence,
            &distilled_relevance,
            distilled_corpus_tokens,
            distilled_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // Collect into per-conversation result
        let mut all_cells = Vec::new();
        all_cells.extend(flat_cells);
        all_cells.extend(enriched_cells);
        all_cells.extend(distilled_cells);

        for cell in &all_cells {
            accumulators
                .entry((cell.condition.clone(), cell.strategy.clone()))
                .or_default()
                .push((
                    cell.ndcg_at_10,
                    cell.mrr,
                    cell.recall_at_5,
                    cell.mean_context_tokens,
                    cell.corpus_tokens,
                    cell.memory_count,
                ));
        }

        per_conversation.push(PipelineConversationResult {
            id: sample.sample_id.clone(),
            observations_seeded: obs_count,
            queries_evaluated: questions.len(),
            distillation_clusters_found: clusters.len(),
            distilled_concepts_created: distilled_count,
            cells: all_cells,
        });

        // Print per-conversation summary
        if let Some(flat_origin) = per_conversation.last().and_then(|c| {
            c.cells
                .iter()
                .find(|c| c.condition == "flat" && c.strategy == "origin")
        }) {
            if let Some(dist_origin) = per_conversation.last().and_then(|c| {
                c.cells
                    .iter()
                    .find(|c| c.condition == "distilled" && c.strategy == "origin")
            }) {
                eprintln!(
                    "  Flat NDCG={:.3} @{:.0}tok -> Distilled NDCG={:.3} @{:.0}tok ({}c, {}m->{}m)",
                    flat_origin.ndcg_at_10,
                    flat_origin.mean_context_tokens,
                    dist_origin.ndcg_at_10,
                    dist_origin.mean_context_tokens,
                    distilled_count,
                    flat_mem_count,
                    distilled_mem_count,
                );
            }
        }
    }

    // Aggregate across conversations
    let mut aggregate = Vec::new();
    for ((condition, strategy), vals) in &accumulators {
        let n = vals.len().max(1) as f64;
        let mean_ndcg = vals.iter().map(|v| v.0).sum::<f64>() / n;
        let mean_mrr = vals.iter().map(|v| v.1).sum::<f64>() / n;
        let mean_recall = vals.iter().map(|v| v.2).sum::<f64>() / n;
        let mean_ctx = vals.iter().map(|v| v.3).sum::<f64>() / n;
        let mean_corpus = (vals.iter().map(|v| v.4).sum::<usize>() as f64 / n) as usize;
        let mean_mem_count = (vals.iter().map(|v| v.5).sum::<usize>() as f64 / n) as usize;
        let density = if mean_ctx > 0.0 {
            mean_ndcg / (mean_ctx / 1000.0)
        } else {
            0.0
        };

        aggregate.push(PipelineCellMetrics {
            condition: condition.clone(),
            strategy: strategy.clone(),
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_recall,
            mean_context_tokens: mean_ctx,
            corpus_tokens: mean_corpus,
            memory_count: mean_mem_count,
            information_density: density,
            queries_evaluated: total_queries,
        });
    }
    // Sort: condition order (flat, enriched, distilled) then strategy
    let condition_order = |c: &str| match c {
        "flat" => 0,
        "enriched" => 1,
        "distilled" => 2,
        _ => 3,
    };
    aggregate.sort_by(|a, b| {
        condition_order(&a.condition)
            .cmp(&condition_order(&b.condition))
            .then(a.strategy.cmp(&b.strategy))
    });

    Ok(PipelineBenchmarkReport {
        benchmark: "LoCoMo".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        conversations: per_conversation.len(),
        total_queries,
        total_observations,
        llm_model: llm.name().to_string(),
        aggregate,
        per_conversation,
    })
}

/// Run LongMemEval through Origin's full pipeline: flat, enriched, distilled.
///
/// Same approach as LoCoMo but adapted for LongMemEval's per-question structure.
/// LongMemEval has 500 questions with ~22 turns each (oracle variant).
/// Smaller per-question corpora mean distillation may find fewer clusters.
pub async fn run_longmemeval_pipeline_eval(
    longmemeval_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_questions: usize,
) -> Result<PipelineBenchmarkReport, OriginError> {
    use crate::eval::longmemeval::{extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Pre-create shared embedder
    eprintln!("[pipeline-lme] loading shared embedder...");
    let shared_embedder = eval_shared_embedder();

    let mut per_conversation = Vec::new();
    let mut total_queries = 0usize;
    let mut total_observations = 0usize;

    #[allow(clippy::type_complexity)]
    let mut accumulators: HashMap<(String, String), Vec<(f64, f64, f64, f64, usize, usize)>> =
        HashMap::new();

    let sample_count = samples.len().min(max_questions);

    for (q_idx, sample) in samples.iter().take(max_questions).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }
        let obs_count = memories.len();
        total_observations += obs_count;

        if q_idx % 25 == 0 {
            eprintln!(
                "[pipeline-lme] Q {}/{} ({}): {} memories, type={}",
                q_idx + 1,
                sample_count,
                sample.question_id,
                obs_count,
                sample.question_type,
            );
        }

        // Build evidence: turns with has_answer=true
        let evidence_session_set: HashSet<&str> = sample
            .answer_session_ids
            .iter()
            .map(|s| s.as_str())
            .collect();

        let mut evidence_ids: HashSet<String> = HashSet::new();
        let mut relevance_map: HashMap<String, u8> = HashMap::new();

        for mem in memories.iter() {
            let sid = format!(
                "lme_{}_{}_t{}",
                sample.question_id, mem.session_idx, mem.turn_idx
            );
            let grade = if mem.has_answer {
                3
            } else if evidence_session_set.contains(mem.session_id.as_str()) {
                2
            } else {
                0
            };
            if grade >= 2 {
                evidence_ids.insert(sid.clone());
            }
            relevance_map.insert(sid, grade);
        }

        if evidence_ids.is_empty() {
            continue;
        }

        let questions = vec![sample.question.clone()];
        let evidence_sets = vec![evidence_ids];
        total_queries += 1;

        // Create DB with shared embedder
        let tmp =
            tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir lme: {e}")))?;
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

        // ---- Flat ----
        let (flat_corpus, flat_count) = count_corpus_tokens(&db).await?;
        let flat_cells = evaluate_condition(
            &db,
            PipelineCondition::Flat,
            &questions,
            &evidence_sets,
            &relevance_map,
            flat_corpus,
            flat_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Enriched ----
        let _extract_count = run_entity_extraction_for_eval(&db, &llm).await?;
        let (enriched_corpus, enriched_count) = count_corpus_tokens(&db).await?;
        let enriched_cells = evaluate_condition(
            &db,
            PipelineCondition::Enriched,
            &questions,
            &evidence_sets,
            &relevance_map,
            enriched_corpus,
            enriched_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Distilled ----
        let distilled_count =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;

        let clusters = db
            .find_distillation_clusters(
                tuning.similarity_threshold,
                tuning.concept_min_cluster_size,
                tuning.max_clusters_per_steep,
                llm.synthesis_token_limit(),
                tuning.concept_min_cluster_size,
            )
            .await
            .unwrap_or_default();

        let (distilled_corpus, distilled_mem_count) = count_corpus_tokens(&db).await?;

        let expanded_evidence = expand_evidence_for_distillation(&db, &evidence_sets).await?;
        let mut distilled_relevance = relevance_map.clone();
        for evidence_set in &expanded_evidence {
            for id in evidence_set {
                if id.starts_with("merged_") {
                    distilled_relevance.entry(id.clone()).or_insert(1);
                }
            }
        }

        let distilled_cells = evaluate_condition(
            &db,
            PipelineCondition::Distilled,
            &questions,
            &expanded_evidence,
            &distilled_relevance,
            distilled_corpus,
            distilled_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        let mut all_cells = Vec::new();
        all_cells.extend(flat_cells);
        all_cells.extend(enriched_cells);
        all_cells.extend(distilled_cells);

        for cell in &all_cells {
            accumulators
                .entry((cell.condition.clone(), cell.strategy.clone()))
                .or_default()
                .push((
                    cell.ndcg_at_10,
                    cell.mrr,
                    cell.recall_at_5,
                    cell.mean_context_tokens,
                    cell.corpus_tokens,
                    cell.memory_count,
                ));
        }

        per_conversation.push(PipelineConversationResult {
            id: sample.question_id.clone(),
            observations_seeded: obs_count,
            queries_evaluated: 1,
            distillation_clusters_found: clusters.len(),
            distilled_concepts_created: distilled_count,
            cells: all_cells,
        });

        if q_idx % 50 == 49 {
            eprintln!("  [progress] {}/{} questions done", q_idx + 1, sample_count);
        }
    }

    // Aggregate
    let mut aggregate = Vec::new();
    for ((condition, strategy), vals) in &accumulators {
        let n = vals.len().max(1) as f64;
        let mean_ndcg = vals.iter().map(|v| v.0).sum::<f64>() / n;
        let mean_mrr = vals.iter().map(|v| v.1).sum::<f64>() / n;
        let mean_recall = vals.iter().map(|v| v.2).sum::<f64>() / n;
        let mean_ctx = vals.iter().map(|v| v.3).sum::<f64>() / n;
        let mean_corpus = (vals.iter().map(|v| v.4).sum::<usize>() as f64 / n) as usize;
        let mean_mem_count = (vals.iter().map(|v| v.5).sum::<usize>() as f64 / n) as usize;
        let density = if mean_ctx > 0.0 {
            mean_ndcg / (mean_ctx / 1000.0)
        } else {
            0.0
        };

        aggregate.push(PipelineCellMetrics {
            condition: condition.clone(),
            strategy: strategy.clone(),
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_recall,
            mean_context_tokens: mean_ctx,
            corpus_tokens: mean_corpus,
            memory_count: mean_mem_count,
            information_density: density,
            queries_evaluated: total_queries,
        });
    }
    let condition_order = |c: &str| match c {
        "flat" => 0,
        "enriched" => 1,
        "distilled" => 2,
        _ => 3,
    };
    aggregate.sort_by(|a, b| {
        condition_order(&a.condition)
            .cmp(&condition_order(&b.condition))
            .then(a.strategy.cmp(&b.strategy))
    });

    Ok(PipelineBenchmarkReport {
        benchmark: "LongMemEval".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        conversations: per_conversation.len(),
        total_queries,
        total_observations,
        llm_model: llm.name().to_string(),
        aggregate,
        per_conversation,
    })
}
