// SPDX-License-Identifier: AGPL-3.0-only
//! LifeBench benchmark adapter (shallow) — multi-source long-horizon memory.
//!
//! LifeBench (arXiv:2603.03781, Mar 2026) tests memory retrieval from diverse
//! digital traces: calendar events, social interactions, routines, location
//! traces, and conversations. SOTA is 55.2%.
//!
//! This shallow adapter treats all events as flat memories with domain/memory_type
//! inferred from source type and content keywords. Procedural inference,
//! temporal chains, and multi-source fusion are deferred.
//!
//! Dataset: linked from <https://arxiv.org/abs/2603.03781>

#[allow(unused_imports)]
use crate::db::MemoryDB;
use crate::error::OriginError;
#[allow(unused_imports)]
use crate::eval::metrics;
#[allow(unused_imports)]
use crate::sources::RawDocument;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORK_KEYWORDS: &[&str] = &[
    "meeting",
    "standup",
    "sprint",
    "deploy",
    "review",
    "deadline",
    "presentation",
    "client",
    "project",
    "office",
    "coworker",
    "manager",
    "budget",
];
#[allow(dead_code)]
const SOURCE_TYPE_ORDER: &[&str] = &["calendar", "social", "routine", "location", "conversation"];

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LifeBenchSample {
    pub question_id: String,
    pub question: String,
    pub answer: serde_json::Value,
    #[serde(default)]
    pub question_type: Option<String>,
    pub events: Vec<LifeBenchEvent>,
    pub evidence_event_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct LifeBenchEvent {
    pub event_id: String,
    pub content: String,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
}

// ---------------------------------------------------------------------------
// Domain inference
// ---------------------------------------------------------------------------

/// Infer the domain ("work" or "personal") from event content using
/// word-boundary-aware keyword matching.
///
/// A keyword matches if the character immediately before it (if any) is
/// non-alphanumeric — i.e. the keyword starts at a word boundary. Prefix
/// matches such as "deployment" (keyword: "deploy") are intentionally
/// included; only matches embedded mid-word (e.g. "review" in "unreviewed")
/// are excluded.
pub fn infer_domain(content: &str) -> &'static str {
    let lower = content.to_lowercase();
    for kw in WORK_KEYWORDS {
        if let Some(pos) = lower.find(kw) {
            // Only check the character before the match (left word boundary).
            // Prefix matches (keyword at start of a longer word) are allowed.
            let before_ok = pos == 0 || !lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
            if before_ok {
                return "work";
            }
        }
    }
    "personal"
}

// ---------------------------------------------------------------------------
// Source type mapping
// ---------------------------------------------------------------------------

/// Map a LifeBench source_type to an Origin memory_type.
pub fn source_type_to_memory_type(source_type: &str) -> &'static str {
    match source_type {
        "routine" => "preference",
        _ => "fact",
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a LifeBench dataset file (JSON array of LifeBenchSample).
pub fn load_lifebench(path: &Path) -> Result<Vec<LifeBenchSample>, OriginError> {
    let data = std::fs::read_to_string(path)?;
    let samples: Vec<LifeBenchSample> = serde_json::from_str(&data)?;
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Report structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct LifeBenchSourceResult {
    pub source_type: String,
    pub count: usize,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LifeBenchReport {
    pub aggregate_ndcg_at_10: f64,
    pub aggregate_mrr: f64,
    pub aggregate_recall_at_5: f64,
    pub aggregate_hit_rate_at_1: f64,
    pub total_questions: usize,
    pub total_events: usize,
    pub per_source_type: Vec<LifeBenchSourceResult>,
}

impl LifeBenchReport {
    /// Format as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("LifeBench Benchmark (Multi-Source Memory)\n");
        out.push_str("==========================================\n");
        out.push_str(&format!("Questions: {}\n", self.total_questions));
        out.push_str(&format!("Total events seeded: {}\n\n", self.total_events));

        out.push_str(&format!(
            "  NDCG@10:     {:.4}  <- primary\n",
            self.aggregate_ndcg_at_10
        ));
        out.push_str(&format!("  MRR:         {:.4}\n", self.aggregate_mrr));
        out.push_str(&format!(
            "  Recall@5:    {:.4}\n",
            self.aggregate_recall_at_5
        ));
        out.push_str(&format!(
            "  Hit Rate@1:  {:.4}\n\n",
            self.aggregate_hit_rate_at_1
        ));

        out.push_str("Per source type:\n");
        for src in &self.per_source_type {
            out.push_str(&format!(
                "  {} (n={:>3}): NDCG@10={:.3} MRR={:.3} R@5={:.3}\n",
                src.source_type, src.count, src.ndcg_at_10, src.mrr, src.recall_at_5,
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// End-to-end benchmark runner
// ---------------------------------------------------------------------------

/// Returns `"lb_{question_id}_{event_id}"`.
fn event_source_id(question_id: &str, event_id: &str) -> String {
    format!("lb_{}_{}", question_id, event_id)
}

/// Run LifeBench benchmark. For each sample:
/// 1. Create fresh ephemeral DB
/// 2. Seed all events as RawDocuments
/// 3. Search with the question, score against evidence event ids
/// 4. Aggregate per-source-type and overall metrics
pub async fn run_lifebench_eval(path: &Path) -> Result<LifeBenchReport, OriginError> {
    let samples = load_lifebench(path)?;
    // (source_type, ndcg_10, mrr, recall_5, hr_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64)> = Vec::new();
    let mut total_events: usize = 0;

    for sample in &samples {
        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all events as RawDocuments
        let docs: Vec<RawDocument> = sample
            .events
            .iter()
            .map(|event| {
                let source_type = event.source_type.as_deref().unwrap_or("unknown");
                let memory_type = source_type_to_memory_type(source_type);
                let domain = infer_domain(&event.content);
                RawDocument {
                    content: event.content.clone(),
                    source_id: event_source_id(&sample.question_id, &event.event_id),
                    source: "memory".to_string(),
                    title: format!("{} {}", source_type, event.event_id),
                    memory_type: Some(memory_type.to_string()),
                    domain: Some(domain.to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        total_events += docs.len();
        db.upsert_documents(docs).await?;

        // Build relevance judgments from evidence_event_ids
        let relevant_source_ids: HashSet<String> = sample
            .evidence_event_ids
            .iter()
            .map(|eid| event_source_id(&sample.question_id, eid))
            .collect();

        if relevant_source_ids.is_empty() {
            continue; // Skip if no evidence events
        }

        // Search
        let results = db
            .search_memory(&sample.question, 10, None, None, None, None, None, None)
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        // Binary relevance grades
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    if relevant_source_ids.contains(*id) {
                        1
                    } else {
                        0
                    },
                )
            })
            .collect();

        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
        let mrr_val = metrics::mrr(&result_ids, &relevant_set);
        let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
        let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

        // Primary source_type from the first evidence event
        let primary_source_type = sample
            .evidence_event_ids
            .first()
            .and_then(|eid| {
                sample
                    .events
                    .iter()
                    .find(|e| &e.event_id == eid)
                    .and_then(|e| e.source_type.as_deref())
            })
            .unwrap_or("unknown")
            .to_string();

        all_scores.push((primary_source_type, ndcg_10, mrr_val, recall_5, hr_1));
    }

    let per_source_type = aggregate_by_source_type(&all_scores);

    Ok(LifeBenchReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.1),
        aggregate_mrr: avg_field(&all_scores, |s| s.2),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.3),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.4),
        total_questions: all_scores.len(),
        total_events,
        per_source_type,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Average a field across a score slice.
fn avg_field(
    scores: &[(String, f64, f64, f64, f64)],
    f: impl Fn(&(String, f64, f64, f64, f64)) -> f64,
) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let sum: f64 = scores.iter().map(&f).sum();
    sum / scores.len() as f64
}

/// Aggregate scores by source_type.
///
/// SOURCE_TYPE_ORDER entries appear first (in that order), then any unknown
/// source types in sorted order — so output is deterministic.
fn aggregate_by_source_type(scores: &[(String, f64, f64, f64, f64)]) -> Vec<LifeBenchSourceResult> {
    let mut results = Vec::new();

    // Known source types in canonical order
    for &src_type in SOURCE_TYPE_ORDER {
        let type_scores: Vec<_> = scores.iter().filter(|s| s.0 == src_type).cloned().collect();
        if type_scores.is_empty() {
            continue;
        }
        results.push(LifeBenchSourceResult {
            source_type: src_type.to_string(),
            count: type_scores.len(),
            ndcg_at_10: avg_field(&type_scores, |s| s.1),
            mrr: avg_field(&type_scores, |s| s.2),
            recall_at_5: avg_field(&type_scores, |s| s.3),
        });
    }

    // Unknown source types not in SOURCE_TYPE_ORDER, in sorted order
    let known: HashSet<&str> = SOURCE_TYPE_ORDER.iter().copied().collect();
    let mut unknown_types: Vec<String> = scores
        .iter()
        .filter(|s| !known.contains(s.0.as_str()))
        .map(|s| s.0.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unknown_types.sort();

    for src_type in unknown_types {
        let type_scores: Vec<_> = scores.iter().filter(|s| s.0 == src_type).cloned().collect();
        results.push(LifeBenchSourceResult {
            source_type: src_type.clone(),
            count: type_scores.len(),
            ndcg_at_10: avg_field(&type_scores, |s| s.1),
            mrr: avg_field(&type_scores, |s| s.2),
            recall_at_5: avg_field(&type_scores, |s| s.3),
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lifebench_sample() {
        let json = r#"[{
            "question_id": "q001",
            "question": "What did Alice do on Monday?",
            "answer": "Attended standup",
            "question_type": "factual",
            "events": [
                {
                    "event_id": "e1",
                    "content": "Team standup at 9am",
                    "source_type": "calendar",
                    "timestamp": "2026-03-01T09:00:00Z"
                },
                {
                    "event_id": "e2",
                    "content": "Lunch with Bob",
                    "source_type": "social",
                    "timestamp": "2026-03-01T12:00:00Z"
                },
                {
                    "event_id": "e3",
                    "content": "Morning run at 7am",
                    "source_type": "routine"
                }
            ],
            "evidence_event_ids": ["e1"]
        }]"#;

        let samples: Vec<LifeBenchSample> = serde_json::from_str(json).unwrap();
        assert_eq!(samples.len(), 1);
        let s = &samples[0];
        assert_eq!(s.question_id, "q001");
        assert_eq!(s.question, "What did Alice do on Monday?");
        assert_eq!(
            s.answer,
            serde_json::Value::String("Attended standup".to_string())
        );
        assert_eq!(s.question_type.as_deref(), Some("factual"));
        assert_eq!(s.events.len(), 3);

        let e1 = &s.events[0];
        assert_eq!(e1.event_id, "e1");
        assert_eq!(e1.source_type.as_deref(), Some("calendar"));
        assert_eq!(e1.timestamp.as_deref(), Some("2026-03-01T09:00:00Z"));

        let e3 = &s.events[2];
        assert_eq!(e3.source_type.as_deref(), Some("routine"));
        assert!(e3.timestamp.is_none());

        assert_eq!(s.evidence_event_ids, vec!["e1"]);
    }

    #[test]
    fn test_infer_domain_work() {
        assert_eq!(infer_domain("Team meeting at 10am"), "work");
        assert_eq!(infer_domain("Standup with engineering"), "work");
        assert_eq!(infer_domain("Sprint planning session"), "work");
        assert_eq!(infer_domain("Deadline for client deliverable"), "work");
        assert_eq!(infer_domain("Presentation to the manager"), "work");
    }

    #[test]
    fn test_infer_domain_personal() {
        assert_eq!(infer_domain("Went for a morning run"), "personal");
        assert_eq!(infer_domain("Called mom to catch up"), "personal");
        assert_eq!(infer_domain("Grocery shopping at 5pm"), "personal");
        assert_eq!(infer_domain("Watched a movie at home"), "personal");
    }

    #[test]
    fn test_infer_domain_word_boundary() {
        // "deployment" contains "deploy" at position 0 — left boundary OK (start of string).
        // Prefix matches are intentionally allowed.
        assert_eq!(infer_domain("the deployment went smoothly"), "work");
        // "unreviewed": "review" appears at position 2 — char before is 'n' (alphanumeric) → no match.
        assert_eq!(infer_domain("unreviewed pull request"), "personal");
    }

    #[test]
    fn test_source_type_to_memory_type() {
        assert_eq!(source_type_to_memory_type("routine"), "preference");
        assert_eq!(source_type_to_memory_type("calendar"), "fact");
        assert_eq!(source_type_to_memory_type("social"), "fact");
        assert_eq!(source_type_to_memory_type("location"), "fact");
        assert_eq!(source_type_to_memory_type("conversation"), "fact");
        assert_eq!(source_type_to_memory_type("unknown"), "fact");
    }

    #[test]
    fn test_report_to_terminal() {
        let report = LifeBenchReport {
            aggregate_ndcg_at_10: 0.4321,
            aggregate_mrr: 0.5678,
            aggregate_recall_at_5: 0.6789,
            aggregate_hit_rate_at_1: 0.3456,
            total_questions: 50,
            total_events: 500,
            per_source_type: vec![
                LifeBenchSourceResult {
                    source_type: "calendar".to_string(),
                    count: 20,
                    ndcg_at_10: 0.45,
                    mrr: 0.55,
                    recall_at_5: 0.65,
                },
                LifeBenchSourceResult {
                    source_type: "social".to_string(),
                    count: 15,
                    ndcg_at_10: 0.38,
                    mrr: 0.48,
                    recall_at_5: 0.58,
                },
            ],
        };

        let output = report.to_terminal();
        assert!(output.contains("LifeBench Benchmark"), "missing header");
        assert!(output.contains("NDCG@10"), "missing NDCG@10 label");
        assert!(output.contains("calendar"), "missing calendar source type");
        assert!(output.contains("social"), "missing social source type");
        assert!(output.contains("50"), "missing question count");
        assert!(output.contains("500"), "missing event count");
    }
}
