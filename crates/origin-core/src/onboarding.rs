// SPDX-License-Identifier: Apache-2.0
//! Onboarding milestone tracking — fire-once events that drive post-wizard UX.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::events::EventEmitter;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;

/// Canonical identifier for each onboarding milestone. The string form
/// (produced by `as_str`, `FromStr`, and serde's kebab-case rename) is
/// the single source of truth for (a) the DB primary key, (b) the JSON
/// wire format, (c) `MilestoneRecord.id`. All four forms must stay in sync —
/// the round-trip test in this module enforces that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MilestoneId {
    IntelligenceReady,
    FirstMemory,
    FirstRecall,
    FirstConcept,
    GraphAlive,
    SecondAgent,
}

impl MilestoneId {
    pub fn as_str(&self) -> &'static str {
        match self {
            MilestoneId::IntelligenceReady => "intelligence-ready",
            MilestoneId::FirstMemory => "first-memory",
            MilestoneId::FirstRecall => "first-recall",
            MilestoneId::FirstConcept => "first-concept",
            MilestoneId::GraphAlive => "graph-alive",
            MilestoneId::SecondAgent => "second-agent",
        }
    }
}

impl FromStr for MilestoneId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "intelligence-ready" => Ok(MilestoneId::IntelligenceReady),
            "first-memory" => Ok(MilestoneId::FirstMemory),
            "first-recall" => Ok(MilestoneId::FirstRecall),
            "first-concept" => Ok(MilestoneId::FirstConcept),
            "graph-alive" => Ok(MilestoneId::GraphAlive),
            "second-agent" => Ok(MilestoneId::SecondAgent),
            _ => Err(format!("unknown milestone id: {}", s)),
        }
    }
}

/// A recorded milestone, returned by DB queries and API endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneRecord {
    pub id: MilestoneId,
    pub first_triggered_at: i64,
    pub acknowledged_at: Option<i64>,
    /// Optional JSON payload (e.g. concept_id, agent_name).
    pub payload: Option<serde_json::Value>,
}

/// Evaluates onboarding milestones at well-defined callsites. Each `check_*`
/// method is idempotent — it delegates to `MemoryDB::record_milestone` which
/// is an `INSERT ... ON CONFLICT DO NOTHING`, so milestones fire exactly once
/// even if the callsite runs many times. On first fire, the evaluator emits
/// an `onboarding-milestone` event with the serialized `MilestoneRecord` as
/// payload so the UI can toast it.
///
/// Borrowed reference to `MemoryDB` by design — the evaluator is a short-lived
/// helper constructed per callsite. Callers typically hold an `Arc<MemoryDB>`
/// in state and pass `&*db` here.
pub struct MilestoneEvaluator<'a> {
    db: &'a MemoryDB,
    emitter: Arc<dyn EventEmitter>,
}

impl<'a> MilestoneEvaluator<'a> {
    pub fn new(db: &'a MemoryDB, emitter: Arc<dyn EventEmitter>) -> Self {
        Self { db, emitter }
    }

    /// Record a milestone and, if it's the first fire, emit an event. Emission
    /// is best-effort — a failing emitter never fails the evaluator.
    async fn fire(
        &self,
        id: MilestoneId,
        payload: Option<serde_json::Value>,
    ) -> Result<(), OriginError> {
        if let Some(record) = self.db.record_milestone(id, payload).await? {
            let json = serde_json::to_string(&record)
                .map_err(|e| OriginError::Generic(format!("serialize milestone: {}", e)))?;
            let _ = self.emitter.emit("onboarding-milestone", &json);
        }
        Ok(())
    }

    /// Called after a memory is successfully ingested. Fires `first-memory`
    /// on any non-manual source. Manual entries are user-driven and don't
    /// count as an agent writing memory — they shouldn't claim the onboarding
    /// milestone for the user.
    ///
    /// Payload includes a short preview of the memory content (first ~100
    /// chars, char-safe) and the source agent name, so the UI toast can
    /// render a subtitle ("Claude: I prefer Rust for CLI tools…") without
    /// an extra fetch.
    pub async fn check_after_ingest(
        &self,
        memory_id: &str,
        source: &str,
    ) -> Result<(), OriginError> {
        if source == "manual" {
            return Ok(());
        }
        let preview = self
            .db
            .get_memory_contents(&[memory_id.to_string()])
            .await
            .ok()
            .and_then(|v| v.into_iter().next())
            .map(|content| {
                // Char-safe truncation (UTF-8 safety: never byte-index).
                let truncated: String = content.chars().take(100).collect();
                if truncated.chars().count() < content.chars().count() {
                    format!("{}…", truncated.trim_end())
                } else {
                    truncated
                }
            });
        let payload = serde_json::json!({
            "memory_id": memory_id,
            "source": source,
            "preview": preview,
        });
        self.fire(MilestoneId::FirstMemory, Some(payload)).await?;
        Ok(())
    }

    /// Called after a refinery pass completes. May fire `first-concept`
    /// (≥1 active concept exists) and/or `graph-alive` (≥5 entities AND
    /// ≥1 relation). Both checks are independent and idempotent.
    pub async fn check_after_refinery_pass(&self) -> Result<(), OriginError> {
        let active_count = self.db.count_active_concepts().await?;
        if active_count >= 1 {
            if let Some(first) = self.db.oldest_active_concept().await? {
                let payload = serde_json::json!({
                    "concept_id": first.id,
                    "title": first.title,
                });
                self.fire(MilestoneId::FirstConcept, Some(payload)).await?;
            }
        }

        let entity_count = self.db.count_entities().await?;
        let relation_count = self.db.count_relations().await?;
        if entity_count >= 5 && relation_count >= 1 {
            self.fire(MilestoneId::GraphAlive, None).await?;
        }
        Ok(())
    }

    /// Called after a `/api/context` or equivalent recall call. Fires
    /// `first-recall` when the agent actually got results back — an empty
    /// recall is not a success story worth toasting.
    ///
    /// Payload carries the agent name and, when available, a short preview
    /// of the top-ranked hit so the UI toast can quote what was actually
    /// surfaced. `top_preview` is char-truncated upstream — we don't
    /// re-truncate here.
    pub async fn check_after_context_call(
        &self,
        agent: &str,
        results_count: usize,
        top_preview: Option<&str>,
    ) -> Result<(), OriginError> {
        if results_count == 0 {
            return Ok(());
        }
        let payload = serde_json::json!({
            "agent": agent,
            "preview": top_preview,
        });
        self.fire(MilestoneId::FirstRecall, Some(payload)).await?;
        Ok(())
    }

    /// Called after an agent registers or records a write. Fires
    /// `second-agent` once ≥2 distinct agents have actually written memory
    /// (per `agent_connections.memory_count >= 1`).
    ///
    /// Payload carries the triggering agent name so the UI toast can render a
    /// specific subtitle ("Cursor just joined — your memories now follow you
    /// across tools.") without re-fetching.
    pub async fn check_after_agent_register(&self, agent: &str) -> Result<(), OriginError> {
        if agent == "unknown" || agent == "manual" {
            return Ok(());
        }
        let written_count = self.db.count_agents_with_writes().await?;
        if written_count >= 2 {
            let payload = serde_json::json!({ "agent": agent });
            self.fire(MilestoneId::SecondAgent, Some(payload)).await?;
        }
        Ok(())
    }

    /// Called once the on-device LLM is downloaded and warm. Fires
    /// `intelligence-ready`.
    pub async fn check_after_llm_ready(&self) -> Result<(), OriginError> {
        self.fire(MilestoneId::IntelligenceReady, None).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[MilestoneId] = &[
        MilestoneId::IntelligenceReady,
        MilestoneId::FirstMemory,
        MilestoneId::FirstRecall,
        MilestoneId::FirstConcept,
        MilestoneId::GraphAlive,
        MilestoneId::SecondAgent,
    ];

    #[test]
    fn as_str_roundtrips_through_from_str() {
        for m in ALL {
            assert_eq!(MilestoneId::from_str(m.as_str()).unwrap(), *m);
        }
    }

    #[test]
    fn as_str_matches_serde_wire_format() {
        for m in ALL {
            let json = serde_json::to_string(m).unwrap();
            assert_eq!(json, format!("\"{}\"", m.as_str()));
        }
    }

    #[test]
    fn from_str_rejects_unknown() {
        assert!(MilestoneId::from_str("not-a-milestone").is_err());
    }

    // ===== MilestoneEvaluator tests =====

    use crate::events::EventEmitter;
    use std::sync::Arc;

    /// Test emitter that captures every (event, payload) pair passed to `emit`.
    /// Uses a std::sync::Mutex — we never hold it across `.await`, so sync is fine.
    struct CapturingEmitter {
        events: std::sync::Mutex<Vec<(String, String)>>,
    }
    impl CapturingEmitter {
        fn new() -> Self {
            Self {
                events: std::sync::Mutex::new(Vec::new()),
            }
        }
    }
    impl EventEmitter for CapturingEmitter {
        fn emit(&self, event: &str, payload: &str) -> anyhow::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push((event.to_string(), payload.to_string()));
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_check_after_ingest_fires_first_memory_once() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let emitter = Arc::new(CapturingEmitter::new());
        let ev = MilestoneEvaluator::new(&db, emitter.clone() as Arc<dyn EventEmitter>);

        ev.check_after_ingest("m1", "claude").await.unwrap();
        ev.check_after_ingest("m2", "claude").await.unwrap();

        let events = emitter.events.lock().unwrap();
        let fires: Vec<_> = events
            .iter()
            .filter(|(k, _)| k == "onboarding-milestone")
            .collect();
        assert_eq!(
            fires.len(),
            1,
            "first-memory should fire exactly once across two ingests"
        );
        assert!(fires[0].1.contains("first-memory"));
    }

    #[tokio::test]
    async fn test_check_after_ingest_skips_manual_source() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let emitter = Arc::new(CapturingEmitter::new());
        let ev = MilestoneEvaluator::new(&db, emitter.clone() as Arc<dyn EventEmitter>);

        ev.check_after_ingest("m1", "manual").await.unwrap();
        assert_eq!(emitter.events.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_check_after_context_call_fires_first_recall_with_agent() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let emitter = Arc::new(CapturingEmitter::new());
        let ev = MilestoneEvaluator::new(&db, emitter.clone() as Arc<dyn EventEmitter>);

        // Zero results should NOT fire.
        ev.check_after_context_call("claude", 0, None)
            .await
            .unwrap();
        assert_eq!(emitter.events.lock().unwrap().len(), 0);

        // Non-zero results fire once, with agent name in payload.
        ev.check_after_context_call("claude", 3, Some("top hit preview"))
            .await
            .unwrap();
        ev.check_after_context_call("claude", 5, Some("another preview"))
            .await
            .unwrap();

        let events = emitter.events.lock().unwrap();
        let fires: Vec<_> = events
            .iter()
            .filter(|(k, _)| k == "onboarding-milestone")
            .collect();
        assert_eq!(fires.len(), 1);
        assert!(fires[0].1.contains("first-recall"));
        assert!(fires[0].1.contains("claude"));
    }

    #[tokio::test]
    async fn test_check_after_llm_ready_fires_once() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let emitter = Arc::new(CapturingEmitter::new());
        let ev = MilestoneEvaluator::new(&db, emitter.clone() as Arc<dyn EventEmitter>);

        ev.check_after_llm_ready().await.unwrap();
        ev.check_after_llm_ready().await.unwrap();

        let events = emitter.events.lock().unwrap();
        let fires: Vec<_> = events
            .iter()
            .filter(|(k, _)| k == "onboarding-milestone")
            .collect();
        assert_eq!(fires.len(), 1);
        assert!(fires[0].1.contains("intelligence-ready"));
    }
}
