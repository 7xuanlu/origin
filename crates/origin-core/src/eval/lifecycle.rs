// SPDX-License-Identifier: AGPL-3.0-only
//! Lifecycle eval mode — measures how each pipeline phase affects retrieval quality.
//!
//! Runs the full memory pipeline (post-ingest → entity extraction → distillation → insights)
//! between memory seeding and search, measuring retrieval metrics after each phase.
//! Identifies which phases help, hurt, or have no effect on retrieval quality.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::fixtures::load_fixtures;
use crate::eval::metrics;
use crate::eval::runner::seed_to_doc;
use crate::llm_provider::{LlmError, LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use crate::sources::RawDocument;
use crate::tuning::{ConfidenceConfig, DistillationConfig, RefineryConfig};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Pipeline phases evaluated in order. Each phase builds on the previous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifecyclePhase {
    /// Raw seeds, no processing.
    Baseline,
    /// After post-ingest enrichment (dedup, entity auto-link, contradiction check).
    PostIngest,
    /// After LLM entity extraction (creates entities + observations + relations).
    EntityExtraction,
    /// After LLM distillation (consolidate clusters, archive originals).
    Distillation,
    /// After concept compilation — measures whether FTS5 concept search finds compiled knowledge.
    ConceptRetrieval,
    /// After LLM insights generation (recaps + decision logs).
    Insights,
}

impl LifecyclePhase {
    fn name(&self) -> &'static str {
        match self {
            Self::Baseline => "Baseline",
            Self::PostIngest => "PostIngest",
            Self::EntityExtraction => "EntityExtraction",
            Self::Distillation => "Distillation",
            Self::ConceptRetrieval => "ConceptRetrieval",
            Self::Insights => "Insights",
        }
    }

    /// Returns true if this phase needs an LLM provider to produce meaningful results.
    pub fn requires_llm(&self) -> bool {
        matches!(
            self,
            Self::EntityExtraction | Self::Distillation | Self::Insights
        )
    }
}

/// Retrieval metrics captured after each pipeline phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub phase: LifecyclePhase,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
    pub precision_at_5: f64,
    pub memory_count: usize,
    pub entity_count: usize,
    pub archived_count: usize,
    /// Number of compiled concepts in the DB (only meaningful for ConceptRetrieval+).
    #[serde(default)]
    pub concept_count: usize,
    /// Unbounded coverage when concept source_memory_ids supplement search_memory results.
    /// Fraction of relevant memories found in (concept sources ∪ search results), no rank cutoff.
    /// Only computed for ConceptRetrieval phase; 0.0 for other phases. Oracle ceiling metric.
    #[serde(default)]
    pub concept_coverage: f64,
    pub duration_ms: u64,
}

/// Delta between two consecutive phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDelta {
    pub from: LifecyclePhase,
    pub to: LifecyclePhase,
    pub ndcg_at_10_delta: f64,
    pub mrr_delta: f64,
    pub recall_at_5_delta: f64,
    /// Combined recall delta (search_memory ∪ concept source_ids). Non-zero only for
    /// transitions into ConceptRetrieval phase.
    #[serde(default)]
    pub combined_recall_delta: f64,
    pub verdict: String,
}

impl PhaseDelta {
    fn compute(from: &PhaseMetrics, to: &PhaseMetrics) -> Self {
        let ndcg_delta = to.ndcg_at_10 - from.ndcg_at_10;
        let combined_delta =
            if to.phase == LifecyclePhase::ConceptRetrieval && to.concept_coverage > 0.0 {
                to.concept_coverage - from.recall_at_5
            } else {
                0.0
            };
        // Use combined coverage for concept transitions when computed, NDCG otherwise
        let effective_delta = if combined_delta.abs() > 1e-9 {
            combined_delta
        } else {
            ndcg_delta
        };
        let verdict = if effective_delta > 0.005 {
            "helped".to_string()
        } else if effective_delta < -0.005 {
            "hurt".to_string()
        } else {
            "neutral".to_string()
        };
        Self {
            from: from.phase,
            to: to.phase,
            ndcg_at_10_delta: ndcg_delta,
            mrr_delta: to.mrr - from.mrr,
            recall_at_5_delta: to.recall_at_5 - from.recall_at_5,
            combined_recall_delta: combined_delta,
            verdict,
        }
    }
}

/// Per-case lifecycle breakdown (for debugging worst cases).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleCaseResult {
    pub query: String,
    pub phase_ndcg: Vec<(LifecyclePhase, f64)>,
}

/// Archive leakage result — do archived memories still appear in search?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveLeakageResult {
    pub total_archived: usize,
    pub leaked: usize,
    pub leaked_ids: Vec<String>,
    pub leakage_rate: f64,
}

/// Round-trip fidelity result — does the pipeline preserve retrieval quality?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundTripResult {
    pub total_cases: usize,
    pub regressions: Vec<RoundTripRegression>,
    pub loss_rate: f64,
    pub mean_delta: f64,
}

/// A single case where the pipeline degraded retrieval quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundTripRegression {
    pub query: String,
    pub baseline_ndcg: f64,
    pub final_ndcg: f64,
    pub delta: f64,
    pub worst_phase: LifecyclePhase,
}

/// Temporal preservation result — do supersession chains survive the pipeline?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPreservationResult {
    pub total_chains: usize,
    pub preserved: usize,
    pub violated: usize,
    pub preservation_rate: f64,
}

/// Full lifecycle eval report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleReport {
    pub benchmark: String,
    pub case_count: usize,
    pub phases: Vec<PhaseMetrics>,
    pub deltas: Vec<PhaseDelta>,
    pub per_case: Vec<LifecycleCaseResult>,
    pub llm_provider: String,
    pub total_duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive_leakage: Option<ArchiveLeakageResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round_trip: Option<RoundTripResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_preservation: Option<TemporalPreservationResult>,
}

// ---------------------------------------------------------------------------
// SupersedesTracker — maps merged memories back to their originals
// ---------------------------------------------------------------------------

/// Tracks which merged memories supersede which originals so that
/// relevance grades can be inherited by distilled memories.
struct SupersedesTracker {
    /// merged_source_id → Vec<original_source_id>
    merged_to_originals: HashMap<String, Vec<String>>,
}

impl SupersedesTracker {
    fn new() -> Self {
        Self {
            merged_to_originals: HashMap::new(),
        }
    }

    /// Build the mapping by correlating pre-scanned clusters with newly created merged memories.
    ///
    /// `pre_clusters` are the cluster source_ids from `find_distillation_clusters()` before distillation.
    /// `merged_before` is the set of merged_* source_ids that existed before distillation.
    async fn build(
        &mut self,
        db: &MemoryDB,
        pre_clusters: &[Vec<String>],
        merged_before: &HashSet<String>,
    ) -> Result<(), OriginError> {
        // Find all merged_* memories that exist now but didn't before
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT DISTINCT source_id, supersedes FROM memories \
             WHERE source_id LIKE 'merged_%' AND source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("supersedes scan: {e}")))?;

        let mut new_merged: Vec<(String, Option<String>)> = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let source_id: String = row.get(0).unwrap_or_default();
            if !merged_before.contains(&source_id) {
                let supersedes: Option<String> = row.get(1).unwrap_or(None);
                new_merged.push((source_id, supersedes));
            }
        }
        drop(rows);
        drop(conn);

        // Match each new merged memory to its cluster via supersedes field
        for (merged_id, supersedes) in &new_merged {
            if let Some(sup_id) = supersedes {
                // Find the cluster containing this superseded ID
                if let Some(cluster) = pre_clusters.iter().find(|c| c.contains(sup_id)) {
                    self.merged_to_originals
                        .insert(merged_id.clone(), cluster.clone());
                }
            }
        }

        Ok(())
    }

    /// Extend relevance grades to include merged memories.
    /// A merged memory inherits the max grade from its originals.
    fn extend_grades(&self, original: &HashMap<String, u8>) -> HashMap<String, u8> {
        let mut extended = original.clone();
        for (merged_id, originals) in &self.merged_to_originals {
            let max_grade = originals
                .iter()
                .filter_map(|id| original.get(id))
                .max()
                .copied()
                .unwrap_or(0);
            if max_grade > 0 {
                extended.insert(merged_id.clone(), max_grade);
            }
        }
        extended
    }

    /// Extend relevant set to include merged memories that supersede any relevant original.
    fn extend_relevant(&self, original: &HashSet<String>) -> HashSet<String> {
        let mut extended = original.clone();
        for (merged_id, originals) in &self.merged_to_originals {
            if originals.iter().any(|id| original.contains(id)) {
                extended.insert(merged_id.clone());
            }
        }
        extended
    }
}

// ---------------------------------------------------------------------------
// EvalMockLlm — public mock for CI tests
// ---------------------------------------------------------------------------

/// Mock LLM provider that inspects the system prompt to return
/// phase-appropriate canned responses. Deterministic, no GPU needed.
pub struct EvalMockLlm {
    call_count: AtomicUsize,
}

impl Default for EvalMockLlm {
    fn default() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }
}

impl EvalMockLlm {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl LlmProvider for EvalMockLlm {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let system = request.system_prompt.as_deref().unwrap_or("");

        if system.contains("merge") || system.contains("distill") || system.contains("consolidat") {
            // Distillation: return first meaningful line from user prompt
            let response = request
                .user_prompt
                .lines()
                .skip(1) // skip "Topic: ..." line
                .find(|l| !l.trim().is_empty())
                .unwrap_or("Distilled memory content.")
                .trim()
                .to_string();
            Ok(response)
        } else if system.contains("knowledge graph") || system.contains("extract") {
            // Entity extraction: return JSON array format expected by parse_kg_response
            let entity_name = request
                .user_prompt
                .split_whitespace()
                .find(|w| w.len() > 3 && w.chars().next().is_some_and(|c| c.is_uppercase()))
                .unwrap_or("TestEntity");
            let entity_name: String = entity_name
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            Ok(format!(
                r#"[{{"i": 0, "entities": [{{"name": "{entity_name}", "type": "concept"}}], "observations": [{{"entity": "{entity_name}", "content": "observed in eval"}}], "relations": []}}]"#
            ))
        } else if system.contains("pattern") || system.contains("recap") || system.contains("burst")
        {
            // Recap: return a summary sentence
            Ok("Activity summary: multiple items observed in this session.".to_string())
        } else if system.contains("decision") {
            // Decision log
            Ok("Decision summary: key decisions were made in this session.".to_string())
        } else if system.contains("classify") {
            // Classification
            Ok(r#"{"memory_type": "fact", "domain": "general", "quality": "medium"}"#.to_string())
        } else if system.contains("contradict") {
            // Contradiction detection
            Ok("Consistent".to_string())
        } else if system.contains("title")
            || system.contains("3-5 word")
            || system.contains("short")
        {
            // Title generation: first 5 words
            let title: String = request
                .user_prompt
                .split_whitespace()
                .take(5)
                .collect::<Vec<_>>()
                .join(" ");
            Ok(title)
        } else {
            // Fallback
            Ok(request.user_prompt.chars().take(100).collect())
        }
    }

    fn is_available(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "eval_mock"
    }
    fn backend(&self) -> crate::llm_provider::LlmBackend {
        crate::llm_provider::LlmBackend::OnDevice
    }
}

// ---------------------------------------------------------------------------
// Measure helper — search + score + count DB state
// ---------------------------------------------------------------------------

/// Search and score against known relevance, returning metrics + per-case NDCG.
#[allow(clippy::type_complexity)]
async fn measure_phase(
    db: &MemoryDB,
    queries: &[(String, HashMap<String, u8>, HashSet<String>, Option<String>)],
    phase: LifecyclePhase,
    phase_start: std::time::Instant,
) -> Result<(PhaseMetrics, Vec<f64>), OriginError> {
    let mut total_ndcg_5 = 0.0;
    let mut total_ndcg_10 = 0.0;
    let mut total_mrr = 0.0;
    let mut total_recall_5 = 0.0;
    let mut total_hr_1 = 0.0;
    let mut total_p5 = 0.0;
    let mut per_case_ndcg: Vec<f64> = Vec::with_capacity(queries.len());
    let count = queries.len().max(1) as f64;

    let mut total_combined_recall_5 = 0.0;

    // Check concept count ONCE before the loop — skip FTS5 queries when no concepts exist.
    let (memory_count, archived_count, entity_count, concept_count) = count_db_state(db).await?;
    let should_search_concepts = phase == LifecyclePhase::ConceptRetrieval && concept_count > 0;

    for (query, grades, relevant, domain) in queries {
        let results = db
            .search_memory(query, 10, None, domain.as_deref(), None, None, None, None)
            .await?;

        let ranked_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        let grades_ref: HashMap<&str, u8> = ranked_ids
            .iter()
            .map(|id| (*id, grades.get(*id).copied().unwrap_or(0)))
            .collect();
        let relevant_ref: HashSet<&str> = relevant.iter().map(|s| s.as_str()).collect();

        let ndcg_10 = metrics::ndcg_at_k(&ranked_ids, &grades_ref, 10);
        total_ndcg_5 += metrics::ndcg_at_k(&ranked_ids, &grades_ref, 5);
        total_ndcg_10 += ndcg_10;
        total_mrr += metrics::mrr(&ranked_ids, &relevant_ref);
        total_recall_5 += metrics::recall_at_k(&ranked_ids, &relevant_ref, 5);
        total_hr_1 += metrics::hit_rate_at_k(&ranked_ids, &relevant_ref, 1);
        total_p5 += metrics::precision_at_k(&ranked_ids, &relevant_ref, 5);
        per_case_ndcg.push(ndcg_10);

        // Combined recall: search_memory ∪ concept source_ids (simulates chat-context Tier 2.5 + 3)
        if should_search_concepts {
            let concepts = db.search_concepts(query, 3).await.unwrap_or_default();
            let mut combined: Vec<String> = Vec::new();
            for concept in &concepts {
                for sid in &concept.source_memory_ids {
                    if !combined.contains(sid) {
                        combined.push(sid.clone());
                    }
                }
            }
            for id in &ranked_ids {
                let s = id.to_string();
                if !combined.contains(&s) {
                    combined.push(s);
                }
            }
            let combined_refs: Vec<&str> = combined.iter().map(|s| s.as_str()).collect();
            total_combined_recall_5 +=
                metrics::recall_at_k(&combined_refs, &relevant_ref, combined_refs.len());
        }
    }

    Ok((
        PhaseMetrics {
            phase,
            ndcg_at_5: total_ndcg_5 / count,
            ndcg_at_10: total_ndcg_10 / count,
            mrr: total_mrr / count,
            recall_at_5: total_recall_5 / count,
            hit_rate_at_1: total_hr_1 / count,
            precision_at_5: total_p5 / count,
            memory_count,
            entity_count,
            archived_count,
            concept_count,
            concept_coverage: total_combined_recall_5 / count,
            duration_ms: phase_start.elapsed().as_millis() as u64,
        },
        per_case_ndcg,
    ))
}

/// Count memories, archived memories, entities, and concepts in the DB.
async fn count_db_state(db: &MemoryDB) -> Result<(usize, usize, usize, usize), OriginError> {
    let conn = db.conn.lock().await;

    let memory_count = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories WHERE source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("count memories: {e}")))?;
        if let Ok(Some(row)) = rows.next().await {
            row.get::<i64>(0).unwrap_or(0) as usize
        } else {
            0
        }
    };

    let archived_count = {
        let mut rows = conn.query(
            "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND supersede_mode = 'archive'",
            libsql::params![],
        ).await.map_err(|e| OriginError::VectorDb(format!("count archived: {e}")))?;
        if let Ok(Some(row)) = rows.next().await {
            row.get::<i64>(0).unwrap_or(0) as usize
        } else {
            0
        }
    };

    let entity_count = {
        let mut rows = conn
            .query("SELECT COUNT(*) FROM entities", libsql::params![])
            .await
            .map_err(|e| OriginError::VectorDb(format!("count entities: {e}")))?;
        if let Ok(Some(row)) = rows.next().await {
            row.get::<i64>(0).unwrap_or(0) as usize
        } else {
            0
        }
    };

    let concept_count = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM concepts WHERE status = 'active'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("count concepts: {e}")))?;
        if let Ok(Some(row)) = rows.next().await {
            row.get::<i64>(0).unwrap_or(0) as usize
        } else {
            0
        }
    };

    drop(conn);
    Ok((memory_count, archived_count, entity_count, concept_count))
}

/// Get all merged_* source_ids currently in the DB.
async fn get_merged_ids(db: &MemoryDB) -> Result<HashSet<String>, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn.query(
        "SELECT DISTINCT source_id FROM memories WHERE source_id LIKE 'merged_%' AND source = 'memory'",
        libsql::params![],
    ).await.map_err(|e| OriginError::VectorDb(format!("merged ids: {e}")))?;

    let mut ids = HashSet::new();
    while let Ok(Some(row)) = rows.next().await {
        if let Ok(id) = row.get::<String>(0) {
            ids.insert(id);
        }
    }
    drop(rows);
    drop(conn);
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Core lifecycle runner — runs all phases on a seeded DB
// ---------------------------------------------------------------------------

/// Internal case representation for lifecycle eval.
struct LifecycleCase {
    query: String,
    domain: Option<String>,
    /// Owned grades: source_id -> relevance grade
    grades: HashMap<String, u8>,
    /// Owned relevant set: source_ids with grade >= 2
    relevant: HashSet<String>,
    /// Source IDs and content for post-ingest (source_id, content, memory_type, domain)
    seeds_meta: Vec<(String, String, Option<String>, Option<String>)>,
}

/// Run lifecycle eval phases on a single seeded DB with multiple queries.
/// Returns per-phase metrics aggregated across all queries.
async fn run_lifecycle_phases(
    db: &MemoryDB,
    cases: &mut [LifecycleCase],
    llm: Option<&Arc<dyn LlmProvider>>,
) -> Result<
    (
        Vec<PhaseMetrics>,
        Vec<LifecycleCaseResult>,
        Option<ArchiveLeakageResult>,
    ),
    OriginError,
> {
    let prompts = PromptRegistry::default();
    let refinery_cfg = RefineryConfig::default();
    let distillation_cfg = DistillationConfig::default();
    let mut all_phases: Vec<PhaseMetrics> = Vec::new();
    let mut per_case_ndcg: Vec<Vec<(LifecyclePhase, f64)>> =
        cases.iter().map(|_| Vec::new()).collect();

    // Helper to build the query tuples for measure_phase
    #[allow(clippy::type_complexity)]
    let build_queries = |cases: &[LifecycleCase]| -> Vec<(String, HashMap<String, u8>, HashSet<String>, Option<String>)> {
        cases.iter().map(|c| {
            (c.query.clone(), c.grades.clone(), c.relevant.clone(), c.domain.clone())
        }).collect()
    };

    // Helper: run measure_phase and record per-case NDCG in one call
    macro_rules! measure {
        ($db:expr, $cases:expr, $phase:expr, $start:expr, $all:expr, $pcn:expr) => {{
            let queries = build_queries($cases);
            let (pm, case_ndcgs) = measure_phase($db, &queries, $phase, $start).await?;
            for (i, ndcg) in case_ndcgs.into_iter().enumerate() {
                $pcn[i].push(($phase, ndcg));
            }
            $all.push(pm);
        }};
    }

    // --- Phase 0: Baseline ---
    let phase_start = std::time::Instant::now();
    measure!(
        db,
        cases,
        LifecyclePhase::Baseline,
        phase_start,
        all_phases,
        per_case_ndcg
    );

    // --- Phase 1: Post-Ingest ---
    let phase_start = std::time::Instant::now();
    let all_seeds_meta: Vec<(String, String, Option<String>, Option<String>)> =
        cases.iter().flat_map(|c| c.seeds_meta.clone()).collect();
    let mut seen = HashSet::new();
    for (source_id, content, memory_type, domain) in &all_seeds_meta {
        if !seen.insert(source_id.clone()) {
            continue;
        }
        // Pass None for LLM during post-ingest so recap generation is
        // deferred to the Insights phase (otherwise PostIngest generates recaps
        // and Insights has nothing left to do).
        let _ = crate::post_ingest::run_post_ingest_enrichment(
            db,
            source_id,
            content,
            None, // entity_id
            memory_type.as_deref(),
            domain.as_deref(),
            None, // structured_fields
            None, // llm — defer recap/title to Insights phase
            &prompts,
            &refinery_cfg,
            &distillation_cfg,
            None, // knowledge_path — eval should not write to knowledge directory
        )
        .await;
    }
    measure!(
        db,
        cases,
        LifecyclePhase::PostIngest,
        phase_start,
        all_phases,
        per_case_ndcg
    );

    // --- Phase 2: Entity Extraction (LLM required) ---
    let phase_start = std::time::Instant::now();
    if let Some(llm_ref) = llm {
        let _ =
            crate::refinery::extract_entities_from_memories(db, Some(llm_ref), &prompts, 100).await;
    }
    measure!(
        db,
        cases,
        LifecyclePhase::EntityExtraction,
        phase_start,
        all_phases,
        per_case_ndcg
    );

    // --- Phase 3: Distillation (LLM required) ---
    let phase_start = std::time::Instant::now();
    if let Some(llm_ref) = llm {
        let clusters = db
            .find_distillation_clusters(
                distillation_cfg.similarity_threshold,
                distillation_cfg.min_cluster_size,
                distillation_cfg.max_clusters_per_steep,
                3500,
            )
            .await?;
        let pre_cluster_ids: Vec<Vec<String>> =
            clusters.iter().map(|c| c.source_ids.clone()).collect();
        let merged_before = get_merged_ids(db).await?;

        let _ =
            crate::refinery::distill_concepts(db, Some(llm_ref), &prompts, &distillation_cfg, None)
                .await;

        let mut tracker = SupersedesTracker::new();
        tracker.build(db, &pre_cluster_ids, &merged_before).await?;

        for case in cases.iter_mut() {
            case.grades = tracker.extend_grades(&case.grades);
            case.relevant = tracker.extend_relevant(&case.relevant);
        }
    }
    measure!(
        db,
        cases,
        LifecyclePhase::Distillation,
        phase_start,
        all_phases,
        per_case_ndcg
    );

    // --- Phase 3.5: ConceptRetrieval (FTS on compiled concepts) ---
    // Measures the DB state after distillation has created concepts.
    // Captures whether distilled concepts + archived originals improve retrieval.
    let phase_start = std::time::Instant::now();
    measure!(
        db,
        cases,
        LifecyclePhase::ConceptRetrieval,
        phase_start,
        all_phases,
        per_case_ndcg
    );

    // --- Forgetting check: do archived memories still appear in search? ---
    let archive_leakage_result = if llm.is_some() {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, content FROM memories \
             WHERE supersede_mode = 'archive' AND source = 'memory' AND chunk_index = 0",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("archive scan: {e}")))?;

        let mut archived: Vec<(String, String)> = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let sid: String = row.get(0).unwrap_or_default();
            let content: String = row.get(1).unwrap_or_default();
            archived.push((sid, content));
        }
        drop(rows);
        drop(conn);

        if !archived.is_empty() {
            let archived_ids: std::collections::HashSet<String> =
                archived.iter().map(|(id, _)| id.clone()).collect();
            let mut leaked_ids: Vec<String> = Vec::new();

            for (sid, content) in &archived {
                let results = db
                    .search_memory(content, 5, None, None, None, None, None, None)
                    .await?;
                if results.iter().any(|r| r.source_id == *sid) {
                    leaked_ids.push(sid.clone());
                }
            }

            let leakage_rate = leaked_ids.len() as f64 / archived_ids.len() as f64;
            Some(ArchiveLeakageResult {
                total_archived: archived_ids.len(),
                leaked: leaked_ids.len(),
                leaked_ids,
                leakage_rate,
            })
        } else {
            None
        }
    } else {
        None
    };

    // --- Phase 4: Insights (LLM required) ---
    let phase_start = std::time::Instant::now();
    if let Some(llm_ref) = llm {
        let _ = crate::refinery::generate_recaps(db, Some(llm_ref), &prompts, &refinery_cfg).await;
        let _ = crate::refinery::generate_decision_logs(db, Some(llm_ref), &prompts, &refinery_cfg)
            .await;
    }
    measure!(
        db,
        cases,
        LifecyclePhase::Insights,
        phase_start,
        all_phases,
        per_case_ndcg
    );

    // Build per-case results
    let per_case: Vec<LifecycleCaseResult> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| LifecycleCaseResult {
            query: c.query.clone(),
            phase_ndcg: per_case_ndcg[i].clone(),
        })
        .collect();

    Ok((all_phases, per_case, archive_leakage_result))
}

// ---------------------------------------------------------------------------
// Fixture lifecycle runner
// ---------------------------------------------------------------------------

/// Run lifecycle eval on TOML fixture files.
pub async fn run_lifecycle_fixture(
    fixture_dir: &Path,
    llm: Option<Arc<dyn LlmProvider>>,
) -> Result<LifecycleReport, OriginError> {
    let eval_start = std::time::Instant::now();
    let eval_cases = load_fixtures(fixture_dir)?;
    let llm_name = llm
        .as_ref()
        .map(|l| l.name().to_string())
        .unwrap_or_else(|| "none".to_string());

    let mut all_phase_metrics: Vec<Vec<PhaseMetrics>> = Vec::new();
    let mut all_per_case: Vec<LifecycleCaseResult> = Vec::new();
    let mut archive_leakage: Option<ArchiveLeakageResult> = None;
    let mut all_temporal_chains = 0usize;
    let mut temporal_preserved = 0usize;
    let mut temporal_violated = 0usize;

    for case in &eval_cases {
        // Fresh ephemeral DB per case
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all memories
        let confidence_cfg = ConfidenceConfig::default();
        let docs: Vec<RawDocument> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .map(|seed| seed_to_doc(seed, &confidence_cfg))
            .collect();
        db.upsert_documents(docs).await?;

        // Seed entities and observations
        let mut obs_grades: HashMap<String, u8> = HashMap::new();
        for entity in &case.entities {
            let entity_id = db
                .store_entity(
                    &entity.name,
                    &entity.entity_type,
                    entity.domain.as_deref(),
                    Some("eval"),
                    None,
                )
                .await?;
            for obs in &entity.observations {
                let obs_id = db
                    .add_observation(&entity_id, obs.content(), Some("eval"), None)
                    .await?;
                let grade = obs.relevance();
                if grade > 0 {
                    obs_grades.insert(format!("obs_{}", obs_id), grade);
                }
            }
        }

        // Build grades and relevant set (owned strings)
        let mut grades: HashMap<String, u8> = case
            .seeds
            .iter()
            .map(|s| (s.id.clone(), s.relevance))
            .collect();
        for (obs_id, grade) in &obs_grades {
            grades.insert(obs_id.clone(), *grade);
        }
        let relevant: HashSet<String> = grades
            .iter()
            .filter(|(_, &v)| v >= 2)
            .map(|(k, _)| k.clone())
            .collect();

        // Build seeds metadata for post-ingest.
        // Only positive seeds are enriched — negative seeds are noise that shouldn't
        // benefit from enrichment (entity linking would make them more findable).
        let seeds_meta: Vec<(String, String, Option<String>, Option<String>)> = case
            .seeds
            .iter()
            .map(|s| {
                (
                    s.id.clone(),
                    s.content.clone(),
                    Some(s.memory_type.clone()),
                    s.domain.clone(),
                )
            })
            .collect();

        let mut lifecycle_cases = vec![LifecycleCase {
            query: case.query.clone(),
            domain: case.domain.clone(),
            grades,
            relevant,
            seeds_meta,
        }];

        let llm_ref = llm.clone();
        let (phases, per_case, case_archive_leakage) =
            run_lifecycle_phases(&db, &mut lifecycle_cases, llm_ref.as_ref()).await?;

        all_phase_metrics.push(phases);
        all_per_case.extend(per_case);
        if case_archive_leakage.is_some() {
            archive_leakage = case_archive_leakage;
        }

        // Temporal preservation check — runs after the full pipeline (including Insights)
        // to verify supersession chains survive end-to-end, not just post-distillation.
        // One search per case, then check all supersession pairs against the result set.
        let chains: Vec<(&str, &str)> = case
            .seeds
            .iter()
            .filter_map(|s| s.supersedes.as_deref().map(|old| (s.id.as_str(), old)))
            .collect();
        if !chains.is_empty() {
            let results = db
                .search_memory(&case.query, 5, None, None, None, None, None, None)
                .await?;
            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            for (newer_id, older_id) in &chains {
                all_temporal_chains += 1;
                if crate::eval::metrics::temporal_ordering(&result_ids, newer_id, older_id) {
                    temporal_preserved += 1;
                } else {
                    temporal_violated += 1;
                }
            }
        }
    }

    // Aggregate across cases
    let phases = aggregate_phase_metrics(&all_phase_metrics);
    let deltas = compute_deltas(&phases);
    let round_trip = Some(compute_round_trip(&all_per_case));

    let temporal_preservation = if all_temporal_chains > 0 {
        Some(TemporalPreservationResult {
            total_chains: all_temporal_chains,
            preserved: temporal_preserved,
            violated: temporal_violated,
            preservation_rate: temporal_preserved as f64 / all_temporal_chains as f64,
        })
    } else {
        None
    };

    Ok(LifecycleReport {
        benchmark: "fixture".to_string(),
        case_count: eval_cases.len(),
        phases,
        deltas,
        per_case: all_per_case,
        llm_provider: llm_name,
        total_duration_ms: eval_start.elapsed().as_millis() as u64,
        archive_leakage,
        round_trip,
        temporal_preservation,
    })
}

// ---------------------------------------------------------------------------
// LoCoMo lifecycle runner
// ---------------------------------------------------------------------------

/// Run lifecycle eval on the LoCoMo benchmark.
pub async fn run_lifecycle_locomo(
    path: &Path,
    llm: Option<Arc<dyn LlmProvider>>,
) -> Result<LifecycleReport, OriginError> {
    use crate::eval::locomo::{extract_observations, load_locomo};

    let eval_start = std::time::Instant::now();
    let samples = load_locomo(path)?;
    let llm_name = llm
        .as_ref()
        .map(|l| l.name().to_string())
        .unwrap_or_else(|| "none".to_string());

    let mut all_phase_metrics: Vec<Vec<PhaseMetrics>> = Vec::new();
    let mut all_per_case: Vec<LifecycleCaseResult> = Vec::new();
    let mut total_cases = 0usize;

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all observations as memories
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

        // Build dia_id -> source_id map
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

        // Build seeds metadata for post-ingest
        let seeds_meta: Vec<(String, String, Option<String>, Option<String>)> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                (
                    format!("locomo_{}_obs_{}", sample.sample_id, i),
                    m.content.clone(),
                    Some("fact".to_string()),
                    Some("conversation".to_string()),
                )
            })
            .collect();

        // Build lifecycle cases for each QA pair
        let mut lifecycle_cases: Vec<LifecycleCase> = Vec::new();
        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            } // skip adversarial

            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }

            // Binary relevance grades — only include evidence items in the grades map.
            // This matches run_locomo_eval's methodology (grades from result IDs only)
            // so IDCG is computed correctly and numbers are comparable.
            // LoCoMo doesn't provide graded relevance, so binary (1 = evidence) is used.
            let grades: HashMap<String, u8> =
                relevant_ids.iter().map(|sid| (sid.clone(), 1u8)).collect();

            lifecycle_cases.push(LifecycleCase {
                query: qa.question.clone(),
                domain: None, // match run_locomo_eval — no domain filter on search
                grades,
                relevant: relevant_ids,
                seeds_meta: seeds_meta.clone(),
            });
        }

        if lifecycle_cases.is_empty() {
            continue;
        }
        total_cases += lifecycle_cases.len();

        let llm_ref = llm.clone();
        let (phases, per_case, _archive_leakage) =
            run_lifecycle_phases(&db, &mut lifecycle_cases, llm_ref.as_ref()).await?;

        all_phase_metrics.push(phases);
        all_per_case.extend(per_case);
    }

    let phases = aggregate_phase_metrics(&all_phase_metrics);
    let deltas = compute_deltas(&phases);

    Ok(LifecycleReport {
        benchmark: "locomo".to_string(),
        case_count: total_cases,
        phases,
        deltas,
        per_case: all_per_case,
        llm_provider: llm_name,
        total_duration_ms: eval_start.elapsed().as_millis() as u64,
        archive_leakage: None,
        round_trip: None,
        temporal_preservation: None,
    })
}

// ---------------------------------------------------------------------------
// LongMemEval lifecycle runner
// ---------------------------------------------------------------------------

/// Run lifecycle eval on the LongMemEval benchmark.
pub async fn run_lifecycle_longmemeval(
    path: &Path,
    llm: Option<Arc<dyn LlmProvider>>,
) -> Result<LifecycleReport, OriginError> {
    use crate::eval::longmemeval::{extract_memories, load_longmemeval};

    let eval_start = std::time::Instant::now();
    let samples = load_longmemeval(path)?;
    let llm_name = llm
        .as_ref()
        .map(|l| l.name().to_string())
        .unwrap_or_else(|| "none".to_string());

    let mut all_phase_metrics: Vec<Vec<PhaseMetrics>> = Vec::new();
    let mut all_per_case: Vec<LifecycleCaseResult> = Vec::new();

    fn memory_source_id(question_id: &str, session_idx: usize, turn_idx: usize) -> String {
        format!("lme_{}_{}_t{}", question_id, session_idx, turn_idx)
    }

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all extracted memories
        let memory_type = match sample.question_type.as_str() {
            "single-session-preference" => "preference",
            _ => "fact",
        };
        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| RawDocument {
                content: mem.content.clone(),
                source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                source: "memory".to_string(),
                title: format!("{} session {}", mem.role, mem.session_idx),
                memory_type: Some(memory_type.to_string()),
                domain: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Build relevance: has_answer turns are relevant.
        // Only include evidence items in grades map — matches run_longmemeval_eval's
        // methodology so IDCG is comparable.
        let relevant: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant.is_empty() {
            continue;
        }

        let grades: HashMap<String, u8> = relevant.iter().map(|sid| (sid.clone(), 1u8)).collect();

        let seeds_meta: Vec<(String, String, Option<String>, Option<String>)> = memories
            .iter()
            .map(|m| {
                (
                    memory_source_id(&m.question_id, m.session_idx, m.turn_idx),
                    m.content.clone(),
                    Some(memory_type.to_string()),
                    Some("conversation".to_string()),
                )
            })
            .collect();

        let mut lifecycle_cases = vec![LifecycleCase {
            query: sample.question.clone(),
            domain: None, // match run_longmemeval_eval — no domain filter on search
            grades,
            relevant,
            seeds_meta,
        }];

        let llm_ref = llm.clone();
        let (phases, per_case, _archive_leakage) =
            run_lifecycle_phases(&db, &mut lifecycle_cases, llm_ref.as_ref()).await?;

        all_phase_metrics.push(phases);
        all_per_case.extend(per_case);
    }

    let phases = aggregate_phase_metrics(&all_phase_metrics);
    let deltas = compute_deltas(&phases);

    Ok(LifecycleReport {
        benchmark: "longmemeval".to_string(),
        case_count: all_per_case.len(),
        phases,
        deltas,
        per_case: all_per_case,
        llm_provider: llm_name,
        total_duration_ms: eval_start.elapsed().as_millis() as u64,
        archive_leakage: None,
        round_trip: None,
        temporal_preservation: None,
    })
}

// ---------------------------------------------------------------------------
// Aggregation helpers
// ---------------------------------------------------------------------------

/// Aggregate per-case phase metrics into a single set of phase metrics.
fn aggregate_phase_metrics(all: &[Vec<PhaseMetrics>]) -> Vec<PhaseMetrics> {
    if all.is_empty() {
        return Vec::new();
    }

    let num_phases = all[0].len();
    let n = all.len() as f64;

    (0..num_phases)
        .map(|phase_idx| {
            let phase = all[0][phase_idx].phase;
            PhaseMetrics {
                phase,
                ndcg_at_5: all.iter().map(|p| p[phase_idx].ndcg_at_5).sum::<f64>() / n,
                ndcg_at_10: all.iter().map(|p| p[phase_idx].ndcg_at_10).sum::<f64>() / n,
                mrr: all.iter().map(|p| p[phase_idx].mrr).sum::<f64>() / n,
                recall_at_5: all.iter().map(|p| p[phase_idx].recall_at_5).sum::<f64>() / n,
                hit_rate_at_1: all.iter().map(|p| p[phase_idx].hit_rate_at_1).sum::<f64>() / n,
                precision_at_5: all.iter().map(|p| p[phase_idx].precision_at_5).sum::<f64>() / n,
                // Sum counts (not average)
                memory_count: all.iter().map(|p| p[phase_idx].memory_count).sum(),
                entity_count: all.iter().map(|p| p[phase_idx].entity_count).sum(),
                archived_count: all.iter().map(|p| p[phase_idx].archived_count).sum(),
                concept_count: all.iter().map(|p| p[phase_idx].concept_count).sum(),
                concept_coverage: all
                    .iter()
                    .map(|p| p[phase_idx].concept_coverage)
                    .sum::<f64>()
                    / n,
                // Sum durations
                duration_ms: all.iter().map(|p| p[phase_idx].duration_ms).sum(),
            }
        })
        .collect()
}

/// Compute deltas between consecutive phases.
fn compute_deltas(phases: &[PhaseMetrics]) -> Vec<PhaseDelta> {
    phases
        .windows(2)
        .map(|w| PhaseDelta::compute(&w[0], &w[1]))
        .collect()
}

/// Compute round-trip fidelity from per-case lifecycle results.
/// A regression is any case where final NDCG dropped > 0.2 from Baseline.
pub(crate) fn compute_round_trip(per_case: &[LifecycleCaseResult]) -> RoundTripResult {
    let mut regressions = Vec::new();
    let mut total_delta = 0.0;

    for case in per_case {
        if case.phase_ndcg.len() < 2 {
            continue;
        }
        let baseline = case.phase_ndcg[0].1;
        let final_ndcg = case.phase_ndcg.last().unwrap().1;
        let delta = final_ndcg - baseline;
        total_delta += delta;

        if delta < -0.2 {
            // Find the phase with the largest single-step drop
            let worst_phase = case
                .phase_ndcg
                .windows(2)
                .map(|w| (w[1].0, w[1].1 - w[0].1))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(phase, _)| phase)
                .unwrap_or(LifecyclePhase::Baseline);

            regressions.push(RoundTripRegression {
                query: case.query.clone(),
                baseline_ndcg: baseline,
                final_ndcg,
                delta,
                worst_phase,
            });
        }
    }

    let analyzed = per_case.iter().filter(|c| c.phase_ndcg.len() >= 2).count();
    let loss_rate = if analyzed > 0 {
        regressions.len() as f64 / analyzed as f64
    } else {
        0.0
    };
    let mean_delta = if analyzed > 0 {
        total_delta / analyzed as f64
    } else {
        0.0
    };
    RoundTripResult {
        total_cases: analyzed,
        regressions,
        loss_rate,
        mean_delta,
    }
}

// ---------------------------------------------------------------------------
// Terminal report
// ---------------------------------------------------------------------------

impl LifecycleReport {
    /// Format the report for terminal output.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "\nLifecycle Eval — {} ({} cases), LLM: {}\n",
            self.benchmark, self.case_count, self.llm_provider,
        ));
        out.push_str(&"═".repeat(92));
        out.push('\n');
        out.push_str(&format!(
            "{:<20} {:>8} {:>7} {:>7} {:>8} {:>7} {:>6} {:>6} {:>6} {:>6}\n",
            "Phase", "NDCG@10", "MRR", "R@5", "Cov+C*", "HR@1", "Mems", "Ents", "Arch", "Cpts",
        ));
        out.push_str(&"─".repeat(92));
        out.push('\n');

        for pm in &self.phases {
            let concepts_str = if pm.concept_count > 0 {
                format!("{:>6}", pm.concept_count)
            } else {
                format!("{:>6}", "-")
            };
            let combined_str = if pm.concept_coverage > 0.0 {
                format!("{:>8.3}", pm.concept_coverage)
            } else {
                format!("{:>8}", "-")
            };
            out.push_str(&format!(
                "{:<20} {:>8.3} {:>7.3} {:>7.3} {} {:>7.3} {:>6} {:>6} {:>6} {}\n",
                pm.phase.name(),
                pm.ndcg_at_10,
                pm.mrr,
                pm.recall_at_5,
                combined_str,
                pm.hit_rate_at_1,
                pm.memory_count,
                pm.entity_count,
                pm.archived_count,
                concepts_str,
            ));
        }

        if !self.deltas.is_empty() {
            out.push('\n');
            out.push_str("Deltas:\n");
            for d in &self.deltas {
                let sign = if d.ndcg_at_10_delta >= 0.0 { "+" } else { "" };
                let verdict_marker = match d.verdict.as_str() {
                    "helped" => " HELPED",
                    "hurt" => " HURT",
                    _ => "",
                };
                let combined_str = if d.to == LifecyclePhase::ConceptRetrieval
                    && d.combined_recall_delta.abs() > 1e-6
                {
                    format!(
                        "  Cov+C* {}{:.3}",
                        if d.combined_recall_delta >= 0.0 {
                            "+"
                        } else {
                            ""
                        },
                        d.combined_recall_delta
                    )
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "  {} → {}:  NDCG {}{:.3}  MRR {}{:.3}  R@5 {}{:.3}{}{}\n",
                    d.from.name(),
                    d.to.name(),
                    sign,
                    d.ndcg_at_10_delta,
                    if d.mrr_delta >= 0.0 { "+" } else { "" },
                    d.mrr_delta,
                    if d.recall_at_5_delta >= 0.0 { "+" } else { "" },
                    d.recall_at_5_delta,
                    combined_str,
                    verdict_marker,
                ));
            }
        }

        if let Some(ref al) = self.archive_leakage {
            out.push_str(&format!(
                "\nForgetting: {}/{} archived leaked (rate={:.3})\n",
                al.leaked, al.total_archived, al.leakage_rate,
            ));
            if !al.leaked_ids.is_empty() {
                for id in &al.leaked_ids {
                    out.push_str(&format!("  LEAKED: {}\n", id));
                }
            }
        }

        if let Some(ref rt) = self.round_trip {
            out.push_str(&format!(
                "\nRound-trip: {}/{} regressed (loss_rate={:.3}, mean_delta={:+.3})\n",
                rt.regressions.len(),
                rt.total_cases,
                rt.loss_rate,
                rt.mean_delta,
            ));
            for r in &rt.regressions {
                let query_short: String = r.query.chars().take(40).collect();
                out.push_str(&format!(
                    "  {:.2} -> {:.2} ({:+.2}) worst={} | {}\n",
                    r.baseline_ndcg,
                    r.final_ndcg,
                    r.delta,
                    r.worst_phase.name(),
                    query_short,
                ));
            }
        }

        if let Some(ref tp) = self.temporal_preservation {
            out.push_str(&format!(
                "\nTemporal: {}/{} chains preserved (rate={:.3})\n",
                tp.preserved, tp.total_chains, tp.preservation_rate,
            ));
        }

        // Concept impact summary — only when ConceptRetrieval phase has data
        let concept_phase = self
            .phases
            .iter()
            .find(|p| p.phase == LifecyclePhase::ConceptRetrieval);
        let distill_phase = self
            .phases
            .iter()
            .find(|p| p.phase == LifecyclePhase::Distillation);
        if let (Some(cp), Some(dp)) = (concept_phase, distill_phase) {
            out.push_str("\nConcept Impact:\n");
            out.push_str(&format!("  Concepts compiled: {}\n", cp.concept_count));
            if cp.concept_coverage > 0.0 {
                let delta = cp.concept_coverage - dp.recall_at_5;
                out.push_str(&format!(
                    "  R@5 (memory only):    {:.3}\n  Cov+C* (combined):    {:.3}  ({:+.3})\n",
                    dp.recall_at_5, cp.concept_coverage, delta,
                ));
            }
        }

        // Footnote for Cov+C*
        let has_combined = self.phases.iter().any(|p| p.concept_coverage > 0.0);
        if has_combined {
            out.push_str(
                "\n  * Cov+C = oracle ceiling (unbounded coverage, synthetic concepts from\n",
            );
            out.push_str(
                "    known-relevant seeds). Not rank-limited like R@5. Real LLM-generated\n",
            );
            out.push_str("    concepts will score lower. Not comparable to LoCoMo/LongMemEval.\n");
        }

        out.push_str(&format!("\nTotal: {}ms\n", self.total_duration_ms));
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_delta_helped() {
        let from = PhaseMetrics {
            phase: LifecyclePhase::Baseline,
            ndcg_at_5: 0.7,
            ndcg_at_10: 0.75,
            mrr: 0.8,
            recall_at_5: 0.6,
            hit_rate_at_1: 0.5,
            precision_at_5: 0.4,
            memory_count: 10,
            entity_count: 0,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 10,
        };
        let to = PhaseMetrics {
            phase: LifecyclePhase::EntityExtraction,
            ndcg_at_5: 0.8,
            ndcg_at_10: 0.85,
            mrr: 0.9,
            recall_at_5: 0.7,
            hit_rate_at_1: 0.6,
            precision_at_5: 0.5,
            memory_count: 10,
            entity_count: 5,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 50,
        };
        let delta = PhaseDelta::compute(&from, &to);
        assert_eq!(delta.verdict, "helped");
        assert!((delta.ndcg_at_10_delta - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_phase_delta_hurt() {
        let from = PhaseMetrics {
            phase: LifecyclePhase::Distillation,
            ndcg_at_5: 0.8,
            ndcg_at_10: 0.85,
            mrr: 0.9,
            recall_at_5: 0.7,
            hit_rate_at_1: 0.6,
            precision_at_5: 0.5,
            memory_count: 10,
            entity_count: 5,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 50,
        };
        let to = PhaseMetrics {
            phase: LifecyclePhase::Insights,
            ndcg_at_5: 0.75,
            ndcg_at_10: 0.80,
            mrr: 0.85,
            recall_at_5: 0.65,
            hit_rate_at_1: 0.55,
            precision_at_5: 0.45,
            memory_count: 12,
            entity_count: 5,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 100,
        };
        let delta = PhaseDelta::compute(&from, &to);
        assert_eq!(delta.verdict, "hurt");
    }

    #[test]
    fn test_phase_delta_neutral() {
        let from = PhaseMetrics {
            phase: LifecyclePhase::Baseline,
            ndcg_at_5: 0.8,
            ndcg_at_10: 0.85,
            mrr: 0.9,
            recall_at_5: 0.7,
            hit_rate_at_1: 0.6,
            precision_at_5: 0.5,
            memory_count: 10,
            entity_count: 0,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 10,
        };
        let to = PhaseMetrics {
            phase: LifecyclePhase::PostIngest,
            ndcg_at_5: 0.8,
            ndcg_at_10: 0.852,
            mrr: 0.9,
            recall_at_5: 0.7,
            hit_rate_at_1: 0.6,
            precision_at_5: 0.5,
            memory_count: 10,
            entity_count: 0,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 20,
        };
        let delta = PhaseDelta::compute(&from, &to);
        assert_eq!(delta.verdict, "neutral");
    }

    #[test]
    fn test_phase_delta_concept_coverage_helped() {
        let from = PhaseMetrics {
            phase: LifecyclePhase::Distillation,
            ndcg_at_5: 0.8,
            ndcg_at_10: 0.85,
            mrr: 0.9,
            recall_at_5: 0.6,
            hit_rate_at_1: 0.6,
            precision_at_5: 0.5,
            memory_count: 10,
            entity_count: 5,
            archived_count: 0,
            concept_count: 0,
            concept_coverage: 0.0,
            duration_ms: 50,
        };
        let to = PhaseMetrics {
            phase: LifecyclePhase::ConceptRetrieval,
            ndcg_at_5: 0.8,
            ndcg_at_10: 0.85,
            mrr: 0.9,
            recall_at_5: 0.6,
            hit_rate_at_1: 0.6,
            precision_at_5: 0.5,
            memory_count: 10,
            entity_count: 5,
            archived_count: 0,
            concept_count: 3,
            concept_coverage: 0.85,
            duration_ms: 50,
        };
        let delta = PhaseDelta::compute(&from, &to);
        assert_eq!(delta.verdict, "helped");
        assert!(
            (delta.combined_recall_delta - 0.25).abs() < 0.01,
            "combined_recall_delta should be 0.85 - 0.6 = 0.25, got {}",
            delta.combined_recall_delta
        );
    }

    #[test]
    fn test_supersedes_tracker_extend_grades() {
        let mut tracker = SupersedesTracker::new();
        tracker.merged_to_originals.insert(
            "merged_abc".to_string(),
            vec![
                "seed_1".to_string(),
                "seed_2".to_string(),
                "seed_3".to_string(),
            ],
        );

        let mut original = HashMap::new();
        original.insert("seed_1".to_string(), 3u8);
        original.insert("seed_2".to_string(), 1u8);
        original.insert("seed_3".to_string(), 2u8);
        original.insert("seed_4".to_string(), 0u8);

        let extended = tracker.extend_grades(&original);
        assert_eq!(extended.get("merged_abc"), Some(&3)); // max of 3, 1, 2
        assert_eq!(extended.get("seed_1"), Some(&3)); // originals preserved
    }

    #[test]
    fn test_supersedes_tracker_extend_relevant() {
        let mut tracker = SupersedesTracker::new();
        tracker.merged_to_originals.insert(
            "merged_abc".to_string(),
            vec!["seed_1".to_string(), "seed_2".to_string()],
        );
        tracker.merged_to_originals.insert(
            "merged_xyz".to_string(),
            vec!["seed_5".to_string(), "seed_6".to_string()],
        );

        let mut relevant = HashSet::new();
        relevant.insert("seed_1".to_string());
        relevant.insert("seed_3".to_string());

        let extended = tracker.extend_relevant(&relevant);
        assert!(extended.contains("merged_abc")); // seed_1 is relevant
        assert!(!extended.contains("merged_xyz")); // seed_5/6 not relevant
        assert!(extended.contains("seed_1")); // original preserved
    }

    #[test]
    fn test_mock_llm_distillation_response() {
        let mock = EvalMockLlm::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(mock.generate(LlmRequest {
            system_prompt: Some("You are a merge assistant. Consolidate these memories.".to_string()),
            user_prompt: "Topic: coding\n\nRust is a systems language.\n\nRust has ownership semantics.".to_string(),
            max_tokens: 512,
            temperature: 0.1,
            label: None,
        })).unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("Rust"));
    }

    #[test]
    fn test_mock_llm_entity_response() {
        let mock = EvalMockLlm::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt
            .block_on(mock.generate(LlmRequest {
                system_prompt: Some("Extract a knowledge graph from this text.".to_string()),
                user_prompt: "1. Alice uses PostgreSQL for data storage".to_string(),
                max_tokens: 512,
                temperature: 0.3,
                label: None,
            }))
            .unwrap();
        assert!(result.contains("Alice"));
        // Must be valid JSON array for parse_kg_response
        assert!(
            result.starts_with('['),
            "Entity response must be JSON array, got: {}",
            result
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&result).expect("Entity response must be valid JSON");
        assert!(parsed.is_array());
    }

    #[test]
    fn test_lifecycle_report_to_terminal() {
        let report = LifecycleReport {
            benchmark: "fixture".to_string(),
            case_count: 5,
            phases: vec![
                PhaseMetrics {
                    phase: LifecyclePhase::Baseline,
                    ndcg_at_5: 0.7,
                    ndcg_at_10: 0.75,
                    mrr: 0.8,
                    recall_at_5: 0.6,
                    hit_rate_at_1: 0.5,
                    precision_at_5: 0.4,
                    memory_count: 10,
                    entity_count: 0,
                    archived_count: 0,
                    concept_count: 0,
                    concept_coverage: 0.0,
                    duration_ms: 10,
                },
                PhaseMetrics {
                    phase: LifecyclePhase::PostIngest,
                    ndcg_at_5: 0.7,
                    ndcg_at_10: 0.75,
                    mrr: 0.8,
                    recall_at_5: 0.6,
                    hit_rate_at_1: 0.5,
                    precision_at_5: 0.4,
                    memory_count: 10,
                    entity_count: 0,
                    archived_count: 0,
                    concept_count: 0,
                    concept_coverage: 0.0,
                    duration_ms: 20,
                },
            ],
            deltas: vec![PhaseDelta {
                from: LifecyclePhase::Baseline,
                to: LifecyclePhase::PostIngest,
                ndcg_at_10_delta: 0.0,
                mrr_delta: 0.0,
                recall_at_5_delta: 0.0,
                combined_recall_delta: 0.0,
                verdict: "neutral".to_string(),
            }],
            per_case: vec![],
            llm_provider: "mock".to_string(),
            total_duration_ms: 30,
            archive_leakage: None,
            round_trip: None,
            temporal_preservation: None,
        };

        let output = report.to_terminal();
        assert!(output.contains("Lifecycle Eval"));
        assert!(output.contains("Baseline"));
        assert!(output.contains("PostIngest"));
        assert!(output.contains("NDCG +0.000"));
    }

    #[test]
    fn test_archive_leakage_result_construction() {
        let result = ArchiveLeakageResult {
            total_archived: 5,
            leaked: 1,
            leaked_ids: vec!["leaked_abc".to_string()],
            leakage_rate: 0.2,
        };
        assert_eq!(result.leakage_rate, 0.2);
        assert_eq!(result.leaked_ids.len(), 1);
    }

    #[test]
    fn test_compute_deltas() {
        let phases = vec![
            PhaseMetrics {
                phase: LifecyclePhase::Baseline,
                ndcg_at_5: 0.7,
                ndcg_at_10: 0.75,
                mrr: 0.8,
                recall_at_5: 0.6,
                hit_rate_at_1: 0.5,
                precision_at_5: 0.4,
                memory_count: 10,
                entity_count: 0,
                archived_count: 0,
                concept_count: 0,
                concept_coverage: 0.0,
                duration_ms: 10,
            },
            PhaseMetrics {
                phase: LifecyclePhase::PostIngest,
                ndcg_at_5: 0.7,
                ndcg_at_10: 0.75,
                mrr: 0.8,
                recall_at_5: 0.6,
                hit_rate_at_1: 0.5,
                precision_at_5: 0.4,
                memory_count: 10,
                entity_count: 0,
                archived_count: 0,
                concept_count: 0,
                concept_coverage: 0.0,
                duration_ms: 20,
            },
            PhaseMetrics {
                phase: LifecyclePhase::EntityExtraction,
                ndcg_at_5: 0.8,
                ndcg_at_10: 0.85,
                mrr: 0.9,
                recall_at_5: 0.7,
                hit_rate_at_1: 0.6,
                precision_at_5: 0.5,
                memory_count: 10,
                entity_count: 5,
                archived_count: 0,
                concept_count: 0,
                concept_coverage: 0.0,
                duration_ms: 50,
            },
        ];
        let deltas = compute_deltas(&phases);
        assert_eq!(deltas.len(), 2);
        assert_eq!(deltas[0].verdict, "neutral");
        assert_eq!(deltas[1].verdict, "helped");
    }

    #[test]
    fn test_round_trip_detection() {
        let per_case = vec![
            LifecycleCaseResult {
                query: "good case".to_string(),
                phase_ndcg: vec![
                    (LifecyclePhase::Baseline, 0.8),
                    (LifecyclePhase::PostIngest, 0.85),
                    (LifecyclePhase::EntityExtraction, 0.85),
                    (LifecyclePhase::Distillation, 0.82),
                    (LifecyclePhase::Insights, 0.83),
                ],
            },
            LifecycleCaseResult {
                query: "regressed case".to_string(),
                phase_ndcg: vec![
                    (LifecyclePhase::Baseline, 0.9),
                    (LifecyclePhase::PostIngest, 0.88),
                    (LifecyclePhase::EntityExtraction, 0.85),
                    (LifecyclePhase::Distillation, 0.55),
                    (LifecyclePhase::Insights, 0.55),
                ],
            },
        ];

        let result = compute_round_trip(&per_case);
        assert_eq!(result.total_cases, 2);
        assert_eq!(result.regressions.len(), 1);
        assert_eq!(result.regressions[0].query, "regressed case");
        assert_eq!(
            result.regressions[0].worst_phase,
            LifecyclePhase::Distillation
        );
        assert!((result.regressions[0].delta - (-0.35)).abs() < 0.01);
        assert!((result.loss_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_round_trip_no_regressions() {
        let per_case = vec![LifecycleCaseResult {
            query: "improving case".to_string(),
            phase_ndcg: vec![
                (LifecyclePhase::Baseline, 0.6),
                (LifecyclePhase::PostIngest, 0.7),
                (LifecyclePhase::EntityExtraction, 0.75),
                (LifecyclePhase::Distillation, 0.78),
                (LifecyclePhase::Insights, 0.80),
            ],
        }];

        let result = compute_round_trip(&per_case);
        assert_eq!(result.total_cases, 1);
        assert!(result.regressions.is_empty());
        assert_eq!(result.loss_rate, 0.0);
        assert!(result.mean_delta > 0.0);
    }

    #[test]
    fn test_temporal_preservation_result() {
        let result = TemporalPreservationResult {
            total_chains: 3,
            preserved: 2,
            violated: 1,
            preservation_rate: 2.0 / 3.0,
        };
        assert!((result.preservation_rate - 0.667).abs() < 0.01);
    }

    #[test]
    fn concept_retrieval_phase_exists() {
        let phase = LifecyclePhase::ConceptRetrieval;
        assert_eq!(phase.name(), "ConceptRetrieval");
        assert!(!phase.requires_llm());
    }
}
