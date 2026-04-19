// SPDX-License-Identifier: AGPL-3.0-only
//! Eval runner: seeds ephemeral DB, runs queries, scores results.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::fixtures::{load_fixtures, SeedMemory};
use crate::eval::metrics;
use crate::eval::report::{CaseResult, EvalReport};
use crate::quality_gate::QualityGate;
use crate::sources::RawDocument;
use crate::tuning::{ConfidenceConfig, GateConfig, SearchScoringConfig};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Cosine similarity between two vectors (used for raw score comparison).
fn raw_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Controls which quality gate checks run on negative seeds during eval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateMode {
    /// No filtering — insert all negatives as-is.
    Off,
    /// Rule-based content check only (sync, no DB needed).
    ContentOnly,
    /// Content check + novelty check against already-seeded DB.
    Full,
}

/// Build a RawDocument from a SeedMemory, auto-deriving scoring fields.
pub(crate) fn seed_to_doc(seed: &SeedMemory, _confidence_cfg: &ConfidenceConfig) -> RawDocument {
    // Auto-derive is_recap from memory_type when not explicitly set
    let is_recap = seed.is_recap.unwrap_or(seed.memory_type == "recap");

    // Compute timestamp: subtract age_days from now, or use current time
    let last_modified = match seed.age_days {
        Some(days) => chrono::Utc::now().timestamp() - (days as i64 * 86400),
        None => chrono::Utc::now().timestamp(),
    };

    RawDocument {
        content: seed.content.clone(),
        source_id: seed.id.clone(),
        source: "memory".to_string(),
        title: format!("Eval: {}", seed.id.chars().take(30).collect::<String>()),
        memory_type: Some(seed.memory_type.clone()),
        domain: seed.domain.clone(),
        structured_fields: seed.structured_fields.clone(),
        confidence: seed.confidence,
        confirmed: seed.confirmed,
        quality: seed.quality.clone(),
        is_recap,
        source_agent: seed.source_agent.clone(),
        last_modified,
        supersedes: seed.supersedes.clone(),
        ..Default::default()
    }
}

/// Run eval against fixture files in `fixture_dir`.
/// Creates a fresh ephemeral database per case, seeds it, runs queries, computes metrics.
/// Pass `scoring` to override confirmation_boost / recap_penalty (None uses DB defaults).
pub async fn run_eval(
    fixture_dir: &Path,
    _db_dir: &Path,
    scoring: Option<&SearchScoringConfig>,
    baseline_path: Option<&Path>,
    gate_mode: GateMode,
) -> Result<EvalReport, OriginError> {
    let cases = load_fixtures(fixture_dir)?;
    let file_count = std::fs::read_dir(fixture_dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
                .count()
        })
        .unwrap_or(0);

    let mut case_results = Vec::new();
    let mut total_ndcg_10 = 0.0;
    let mut total_ndcg_5 = 0.0;
    let mut total_map_5 = 0.0;
    let mut total_map_10 = 0.0;
    let mut total_mrr = 0.0;
    let mut total_recall_1 = 0.0;
    let mut total_recall_3 = 0.0;
    let mut total_recall_5 = 0.0;
    let mut total_hit_rate_1 = 0.0;
    let mut total_hit_rate_3 = 0.0;
    let mut total_precision_3 = 0.0;
    let mut total_precision_5 = 0.0;
    let mut total_neg_leakage = 0usize;
    let mut total_neg_above = 0usize;
    let mut total_negatives = 0usize;
    let mut total_gate_content_filtered = 0usize;
    let mut total_gate_novelty_filtered = 0usize;
    let mut empty_set_cosines: Vec<f64> = Vec::new();
    let mut normal_top1_cosines: Vec<f64> = Vec::new();
    let mut empty_set_count = 0usize;
    let mut temporal_ordering_correct = 0usize;
    let mut temporal_ordering_total = 0usize;

    let gate = if gate_mode != GateMode::Off {
        Some(QualityGate::new(GateConfig::default()))
    } else {
        None
    };

    for case in &cases {
        // Fresh ephemeral DB per case — tempdir is dropped after each iteration
        let case_tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("create temp dir for eval case: {}", e)))?;
        let db = MemoryDB::new(
            case_tmp.path(),
            std::sync::Arc::new(crate::events::NoopEmitter),
        )
        .await?;

        // Seed only this case's memories with auto-derived scoring fields
        let confidence_cfg = ConfidenceConfig::default();

        // For Full mode: insert positive seeds first so novelty has something to compare against
        let positive_docs: Vec<RawDocument> = case
            .seeds
            .iter()
            .map(|seed| seed_to_doc(seed, &confidence_cfg))
            .collect();
        if gate_mode == GateMode::Full {
            db.upsert_documents(positive_docs.clone()).await?;
        }

        // Process negative seeds through the gate
        let mut negative_docs = Vec::new();
        for neg in &case.negative_seeds {
            if let Some(ref g) = gate {
                // Content check (sync, rule-based) — runs for both ContentOnly and Full
                let content_result = g.check_content(&neg.content);
                if !content_result.admitted {
                    total_gate_content_filtered += 1;
                    continue;
                }

                // Novelty check (async, needs DB) — only for Full mode
                if gate_mode == GateMode::Full {
                    let (novelty_result, _similar_id) = g.evaluate(&neg.content, &db).await?;
                    if !novelty_result.admitted {
                        total_gate_novelty_filtered += 1;
                        continue;
                    }
                }
            }
            negative_docs.push(seed_to_doc(neg, &confidence_cfg));
        }

        // Insert all documents into the DB
        if gate_mode == GateMode::Full {
            // Positives already inserted; only add surviving negatives
            db.upsert_documents(negative_docs).await?;
        } else {
            // Off or ContentOnly: insert positives + surviving negatives together
            let mut all_docs = positive_docs;
            all_docs.extend(negative_docs);
            db.upsert_documents(all_docs).await?;
        }

        // Seed entities and observations — observations boost scores via graph augmentation
        // but are filtered from search output, so grades aren't used for retrieval metrics.
        let mut _obs_grades: HashMap<String, u8> = HashMap::new();
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
                    _obs_grades.insert(format!("obs_{}", obs_id), grade);
                }
            }
        }

        let confirmation_boost = scoring.map(|s| s.confirmation_boost);
        let recap_penalty = scoring.map(|s| s.recap_penalty);

        let results = db
            .search_memory(
                &case.query,
                10,
                None,
                case.domain.as_deref(),
                None,
                confirmation_boost,
                recap_penalty,
                scoring,
            )
            .await?;

        let ranked_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        if case.empty_set {
            // Empty-set case: compute raw cosine similarity between query and top result.
            // We use raw cosine (not the normalized RRF score) because search_memory
            // always normalizes top-1 to 1.0, making the score useless for false-confidence.
            empty_set_count += 1;
            if let Some(top) = results.first() {
                if let Ok(embs) = db.generate_embeddings(&[case.query.clone(), top.content.clone()])
                {
                    if embs.len() == 2 {
                        empty_set_cosines.push(raw_cosine_similarity(&embs[0], &embs[1]));
                    }
                }
            }
            case_results.push(CaseResult {
                query: case.query.clone(),
                ndcg_at_10: 0.0,
                ndcg_at_5: 0.0,
                map_at_10: 0.0,
                mrr: 0.0,
                recall_at_5: 0.0,
                hit_rate_at_1: 0.0,
                precision_at_3: 0.0,
                negative_leakage: 0,
                neg_above_relevant: 0,
            });
            continue;
        }

        // Normal case: compute raw cosine similarity for score gap calculation.
        // Raw cosine is used instead of the normalized RRF score (always 1.0 for top-1)
        // to get an absolute measure of semantic match quality.
        if let Some(top) = results.first() {
            if let Ok(embs) = db.generate_embeddings(&[case.query.clone(), top.content.clone()]) {
                if embs.len() == 2 {
                    normal_top1_cosines.push(raw_cosine_similarity(&embs[0], &embs[1]));
                }
            }
        }

        // Build relevant set (relevance >= 2) — memories only.
        // Entity observations are seeded for score boosting via graph augmentation
        // but are filtered from search output, so they don't count for retrieval metrics.
        let relevant: HashSet<&str> = case
            .seeds
            .iter()
            .filter(|s| s.relevance >= 2)
            .map(|s| s.id.as_str())
            .collect();

        // Build relevance grades map — memories only
        let grades: HashMap<&str, u8> = case
            .seeds
            .iter()
            .map(|s| (s.id.as_str(), s.relevance))
            .collect();

        let negatives: HashSet<&str> = case.negative_seeds.iter().map(|s| s.id.as_str()).collect();

        let case_ndcg_10 = metrics::ndcg_at_k(&ranked_ids, &grades, 10);
        let case_ndcg_5 = metrics::ndcg_at_k(&ranked_ids, &grades, 5);
        let case_map_5 = metrics::map_at_k(&ranked_ids, &relevant, 5);
        let case_map_10 = metrics::map_at_k(&ranked_ids, &relevant, 10);
        let case_mrr = metrics::mrr(&ranked_ids, &relevant);
        let case_recall_1 = metrics::recall_at_k(&ranked_ids, &relevant, 1);
        let case_recall_3 = metrics::recall_at_k(&ranked_ids, &relevant, 3);
        let case_recall_5 = metrics::recall_at_k(&ranked_ids, &relevant, 5);
        let case_hit_1 = metrics::hit_rate_at_k(&ranked_ids, &relevant, 1);
        let case_hit_3 = metrics::hit_rate_at_k(&ranked_ids, &relevant, 3);
        let case_p3 = metrics::precision_at_k(&ranked_ids, &relevant, 3);
        let case_p5 = metrics::precision_at_k(&ranked_ids, &relevant, 5);
        let case_neg = metrics::negative_leakage(&ranked_ids, &negatives, 5);
        let case_neg_above = metrics::negatives_above_relevant(&ranked_ids, &relevant, &negatives);

        total_ndcg_10 += case_ndcg_10;
        total_ndcg_5 += case_ndcg_5;
        total_map_5 += case_map_5;
        total_map_10 += case_map_10;
        total_mrr += case_mrr;
        total_recall_1 += case_recall_1;
        total_recall_3 += case_recall_3;
        total_recall_5 += case_recall_5;
        total_hit_rate_1 += case_hit_1;
        total_hit_rate_3 += case_hit_3;
        total_precision_3 += case_p3;
        total_precision_5 += case_p5;
        total_neg_leakage += case_neg;
        total_neg_above += case_neg_above;
        total_negatives += case.negative_seeds.len();

        // Check temporal ordering for supersession pairs
        for seed in &case.seeds {
            if let Some(ref superseded_id) = seed.supersedes {
                temporal_ordering_total += 1;
                if metrics::temporal_ordering(&ranked_ids, &seed.id, superseded_id) {
                    temporal_ordering_correct += 1;
                }
            }
        }

        case_results.push(CaseResult {
            query: case.query.clone(),
            ndcg_at_10: case_ndcg_10,
            ndcg_at_5: case_ndcg_5,
            map_at_10: case_map_10,
            mrr: case_mrr,
            recall_at_5: case_recall_5,
            hit_rate_at_1: case_hit_1,
            precision_at_3: case_p3,
            negative_leakage: case_neg,
            neg_above_relevant: case_neg_above,
        });
    }

    let n = cases.iter().filter(|c| !c.empty_set).count().max(1) as f64;

    // false_confidence: mean raw cosine similarity for empty-set top results.
    // Uses raw embedding cosine (not the normalized RRF score that is always 1.0).
    // Lower = better: a well-calibrated system shows low similarity for irrelevant results.
    let empty_set_false_confidence = if !empty_set_cosines.is_empty() {
        Some(empty_set_cosines.iter().sum::<f64>() / empty_set_cosines.len() as f64)
    } else {
        None
    };

    // score_gap: how much higher does raw cosine similarity go for truly relevant
    // results vs irrelevant ones? Larger gap = better separation.
    let score_gap = if !empty_set_cosines.is_empty() && !normal_top1_cosines.is_empty() {
        let normal_mean =
            normal_top1_cosines.iter().sum::<f64>() / normal_top1_cosines.len() as f64;
        let empty_mean = empty_set_cosines.iter().sum::<f64>() / empty_set_cosines.len() as f64;
        Some(normal_mean - empty_mean)
    } else {
        None
    };

    Ok(EvalReport {
        fixture_count: cases.len(),
        file_count,
        search_mode: "search_memory".to_string(),
        ndcg_at_10: total_ndcg_10 / n,
        ndcg_at_5: total_ndcg_5 / n,
        map_at_5: total_map_5 / n,
        map_at_10: total_map_10 / n,
        mrr: total_mrr / n,
        recall_at_1: total_recall_1 / n,
        recall_at_3: total_recall_3 / n,
        recall_at_5: total_recall_5 / n,
        hit_rate_at_1: total_hit_rate_1 / n,
        hit_rate_at_3: total_hit_rate_3 / n,
        precision_at_3: total_precision_3 / n,
        precision_at_5: total_precision_5 / n,
        negative_leakage: total_neg_leakage,
        neg_above_relevant: total_neg_above,
        total_negatives,
        gate_content_filtered: total_gate_content_filtered,
        gate_novelty_filtered: total_gate_novelty_filtered,
        empty_set_count,
        empty_set_false_confidence,
        score_gap,
        temporal_ordering_total,
        temporal_ordering_correct,
        temporal_ordering_rate: if temporal_ordering_total > 0 {
            Some(temporal_ordering_correct as f64 / temporal_ordering_total as f64)
        } else {
            None
        },
        baseline: baseline_path.and_then(crate::eval::report::EvalReport::load_baseline),
        per_case: case_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::fixtures::SeedMemory;

    fn make_seed(id: &str, age_days: Option<u32>, supersedes: Option<&str>) -> SeedMemory {
        SeedMemory {
            id: id.to_string(),
            content: "test content".to_string(),
            memory_type: "fact".to_string(),
            domain: Some("test".to_string()),
            relevance: 2,
            structured_fields: None,
            confidence: None,
            confirmed: None,
            quality: None,
            is_recap: None,
            source_agent: None,
            age_days,
            supersedes: supersedes.map(|s| s.to_string()),
        }
    }

    #[test]
    fn test_seed_to_doc_age_days() {
        let cfg = ConfidenceConfig::default();
        let now = chrono::Utc::now().timestamp();

        let fresh = seed_to_doc(&make_seed("fresh", None, None), &cfg);
        assert!(
            (fresh.last_modified - now).abs() < 2,
            "No age_days should use current time"
        );

        let aged = seed_to_doc(&make_seed("old", Some(30), None), &cfg);
        let expected = now - (30 * 86400);
        assert!(
            (aged.last_modified - expected).abs() < 2,
            "age_days=30 should subtract 30 days"
        );
    }

    #[test]
    fn test_seed_to_doc_supersedes() {
        let cfg = ConfidenceConfig::default();

        let doc = seed_to_doc(&make_seed("new", None, Some("old_id")), &cfg);
        assert_eq!(doc.supersedes, Some("old_id".to_string()));

        let doc_none = seed_to_doc(&make_seed("standalone", None, None), &cfg);
        assert_eq!(doc_none.supersedes, None);
    }

    #[test]
    fn test_raw_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((raw_cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_raw_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(raw_cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_raw_cosine_similarity_partial() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = raw_cosine_similarity(&a, &b);
        // cos(45 degrees) = 1/sqrt(2) ~ 0.707
        assert!((sim - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_raw_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(raw_cosine_similarity(&a, &b), 0.0);
    }
}
