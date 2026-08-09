// SPDX-License-Identifier: Apache-2.0
//! HTTP routes for the refinement queue surface (read + reject).
//!
//! Convergent with daemon-side `synthesis::refinement_queue::resolve_proposal`.

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_core::synthesis::refinement_queue::{
    apply_refinement_with_decision, resolve_proposal, RefinementDecision, ResolveStatus,
};
use wenlan_types::requests::AcceptRefinementRequest;
use wenlan_types::responses::{
    AcceptRefinementResponse, ListRefinementsResponse, ProposalAction, RefinementPayload,
    RefinementProposalSummary, RejectRefinementResponse,
};

use crate::error::ServerError;
use crate::memory_routes::extract_agent_name;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::{ServerState, SharedState};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/refinery/queue", get(handle_list_refinements))
        .route(
            "/api/refinery/queue/{id}/reject",
            post(handle_reject_refinement),
        )
        .route(
            "/api/refinery/queue/{id}/accept",
            post(handle_accept_refinement),
        )
}

#[derive(Debug, Deserialize)]
pub struct ListRefinementsQuery {
    pub action: Option<String>,
    pub limit: Option<usize>,
}

pub async fn handle_list_refinements(
    State(state): State<Arc<RwLock<ServerState>>>,
    Query(q): Query<ListRefinementsQuery>,
) -> Result<Json<ListRefinementsResponse>, ServerError> {
    let limit = q.limit.unwrap_or(500).min(500);

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let pending = db
        .get_pending_refinements()
        .await
        .map_err(ServerError::from)?;

    let proposals: Vec<RefinementProposalSummary> = pending
        .into_iter()
        .filter(|p| p.status == "awaiting_review")
        .filter(|p| match &q.action {
            Some(a) => &p.action == a,
            None => true,
        })
        .filter_map(|p| {
            let action = match parse_action(&p.action) {
                Some(a) => a,
                None => {
                    tracing::warn!(
                        proposal_id = %p.id,
                        action = %p.action,
                        "refinery: skipping proposal with unknown action variant"
                    );
                    return None;
                }
            };
            let payload =
                parse_payload_for_proposal(&p.id, &p.source_ids, p.payload.as_deref(), &p.action);
            if matches!(action, ProposalAction::LintRepairReview) && payload.is_none() {
                tracing::warn!(
                    proposal_id = %p.id,
                    "refinery: skipping malformed lint repair review payload"
                );
                return None;
            }
            Some(RefinementProposalSummary {
                action,
                payload,
                id: p.id,
                source_ids: p.source_ids,
                confidence: p.confidence,
                created_at: p.created_at,
            })
        })
        .take(limit)
        .collect();

    Ok(Json(ListRefinementsResponse { proposals }))
}

pub async fn handle_reject_refinement(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<RejectRefinementResponse>, ServerError> {
    let agent = extract_agent_name(&headers, None);

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let result = resolve_proposal(&db, &id, ResolveStatus::Dismissed, &agent)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(RejectRefinementResponse { id: result.id }))
}

pub async fn handle_accept_refinement(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Bytes,
) -> Result<Json<AcceptRefinementResponse>, ServerError> {
    let agent = extract_agent_name(&headers, None);

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let decision = parse_accept_decision(&body)?;
    let outcome = apply_refinement_with_decision(&db, &id, &agent, decision)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(AcceptRefinementResponse {
        id: outcome.id,
        action_applied: outcome.action_applied,
    }))
}

fn parse_action(s: &str) -> Option<ProposalAction> {
    match s {
        "entity_merge" => Some(ProposalAction::EntityMerge),
        "relation_conflict" => Some(ProposalAction::RelationConflict),
        "detect_contradiction" => Some(ProposalAction::DetectContradiction),
        "suggest_entity" => Some(ProposalAction::SuggestEntity),
        "dedup_merge" => Some(ProposalAction::DedupMerge),
        "page_merge" => Some(ProposalAction::PageMerge),
        "cross_space_discovery" => Some(ProposalAction::CrossSpaceDiscovery),
        "page_keep_or_archive" => Some(ProposalAction::PageKeepOrArchive),
        "lint_repair_review" => Some(ProposalAction::LintRepairReview),
        "vocab_promote" => Some(ProposalAction::VocabPromote),
        _ => None,
    }
}

fn parse_accept_decision(body: &[u8]) -> Result<RefinementDecision, ServerError> {
    if body.is_empty() {
        return Ok(RefinementDecision::Accept { notes: None });
    }
    let parsed: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ServerError::BadRequest(format!("invalid accept body: {e}")))?;
    if parsed.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(RefinementDecision::Accept { notes: None });
    }
    let request: AcceptRefinementRequest = serde_json::from_value(parsed)
        .map_err(|e| ServerError::BadRequest(format!("invalid accept body: {e}")))?;
    Ok(match request {
        AcceptRefinementRequest::Accept { notes } => RefinementDecision::Accept { notes },
        AcceptRefinementRequest::PickSpace { space, notes } => {
            RefinementDecision::PickSpace { space, notes }
        }
    })
}

/// Convert the raw daemon payload string into the typed wire enum.
///
/// Variant decode rules:
/// - `detect_contradiction` and `dedup_merge`: unit variants, return Some regardless of payload
/// - `suggest_entity`: raw string payload lifted into `name_hint`
/// - `entity_merge`, `relation_conflict`: JSON object payload; inject `action` tag.
///   Returns `None` if payload is malformed JSON or fails to deserialize.
/// - Unknown action: falls through to JSON-decode path; returns `None` for unknown shapes.
fn parse_payload(payload: Option<&str>, action: &str) -> Option<RefinementPayload> {
    match action {
        "detect_contradiction" => return Some(RefinementPayload::DetectContradiction),
        "dedup_merge" => return Some(RefinementPayload::DedupMerge),
        "suggest_entity" => {
            return Some(RefinementPayload::SuggestEntity {
                name_hint: payload.map(|s| s.to_string()),
            });
        }
        "vocab_promote" => {
            let raw = payload?;
            return serde_json::from_str::<RefinementPayload>(raw).ok();
        }
        _ => {}
    }

    let raw = payload?;
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(raw).ok()?;
    let mut tagged = map;
    tagged.insert(
        "action".to_string(),
        serde_json::Value::String(action.to_string()),
    );
    serde_json::from_value(serde_json::Value::Object(tagged)).ok()
}

fn parse_payload_for_proposal(
    id: &str,
    source_ids: &[String],
    payload: Option<&str>,
    action: &str,
) -> Option<RefinementPayload> {
    if action == "lint_repair_review" {
        return wenlan_core::db::validate_lint_review_contract(id, source_ids, payload?).ok();
    }
    parse_payload(payload, action)
}

#[cfg(test)]
mod parse_payload_tests {
    use super::*;

    #[test]
    fn parses_entity_merge() {
        let raw = r#"{"existing_id":"e1","new_id":"e2","similarity":0.86}"#;
        let parsed = parse_payload(Some(raw), "entity_merge").unwrap();
        match parsed {
            RefinementPayload::EntityMerge {
                existing_id,
                new_id,
                similarity,
            } => {
                assert_eq!(existing_id, "e1");
                assert_eq!(new_id, "e2");
                assert!((similarity - 0.86).abs() < 1e-9);
            }
            _ => panic!("expected EntityMerge"),
        }
    }

    #[test]
    fn parses_relation_conflict_with_from_to() {
        let raw = r#"{"existing_id":"r1","new_id":"r2","from":"e_a","to":"e_b","old_type":"works_at","new_type":"founded"}"#;
        let parsed = parse_payload(Some(raw), "relation_conflict").unwrap();
        match parsed {
            RefinementPayload::RelationConflict { from, to, .. } => {
                assert_eq!(from, "e_a");
                assert_eq!(to, "e_b");
            }
            _ => panic!("expected RelationConflict"),
        }
    }

    #[test]
    fn detect_contradiction_unit_even_when_payload_none() {
        let parsed = parse_payload(None, "detect_contradiction").unwrap();
        assert!(matches!(parsed, RefinementPayload::DetectContradiction));
    }

    #[test]
    fn suggest_entity_lifts_raw_string_into_name_hint() {
        let parsed = parse_payload(Some("PostgreSQL"), "suggest_entity").unwrap();
        match parsed {
            RefinementPayload::SuggestEntity { name_hint } => {
                assert_eq!(name_hint.as_deref(), Some("PostgreSQL"));
            }
            _ => panic!("expected SuggestEntity"),
        }
    }

    #[test]
    fn dedup_merge_unit_with_no_payload() {
        let parsed = parse_payload(None, "dedup_merge").unwrap();
        assert!(matches!(parsed, RefinementPayload::DedupMerge));
    }

    #[test]
    fn parses_cross_space_discovery_payload() {
        let raw = r#"{"memory_count":3,"spaces":["personal","work"],"allowed_actions":["dismiss","pick_space"]}"#;
        let parsed = parse_payload(Some(raw), "cross_space_discovery").unwrap();
        match parsed {
            RefinementPayload::CrossSpaceDiscovery {
                memory_count,
                spaces,
                ..
            } => {
                assert_eq!(memory_count, 3);
                assert_eq!(spaces, vec!["personal".to_string(), "work".to_string()]);
            }
            _ => panic!("expected CrossSpaceDiscovery"),
        }
    }

    #[test]
    fn parses_page_keep_or_archive_payload() {
        let raw =
            r#"{"page_id":"page_stub","source_count":1,"allowed_actions":["dismiss","accept"]}"#;
        let parsed = parse_payload(Some(raw), "page_keep_or_archive").unwrap();
        match parsed {
            RefinementPayload::PageKeepOrArchive {
                page_id,
                source_count,
                allowed_actions,
            } => {
                assert_eq!(page_id, "page_stub");
                assert_eq!(source_count, 1);
                assert_eq!(
                    allowed_actions,
                    vec![
                        wenlan_types::responses::RefinementCardAction::Dismiss,
                        wenlan_types::responses::RefinementCardAction::Accept
                    ]
                );
            }
            _ => panic!("expected PageKeepOrArchive"),
        }
    }

    #[test]
    fn parses_lint_repair_review_action_and_payload() {
        assert_eq!(
            parse_action("lint_repair_review"),
            Some(ProposalAction::LintRepairReview)
        );

        let occurrence = wenlan_types::repair::RepairDigest::parse(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        .unwrap();
        let source_ids = vec!["page-a".to_string()];
        let owner_binding_digest = wenlan_types::repair::RepairDigest::parse(
            "9564fbd4194ceeb40ff8305b376a6263acb872733009ce606dfd896d1c747d25",
        )
        .unwrap();
        let raw = serde_json::json!({
            "action": "lint_repair_review",
            "check_id": "pages.links.orphan_labels",
            "occurrence_digest": occurrence,
            "owner_binding_digest": owner_binding_digest,
            "issue": "Review the ambiguous target.",
            "choices": ["keep", "retarget", "remove"],
            "suggested_research_queries": ["Find the canonical page"]
        })
        .to_string();
        let parsed = parse_payload_for_proposal(
            "lint_review_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &source_ids,
            Some(&raw),
            "lint_repair_review",
        )
        .unwrap();
        match parsed {
            RefinementPayload::LintRepairReview {
                check_id,
                owner_binding_digest,
                choices,
                suggested_research_queries,
                ..
            } => {
                assert_eq!(check_id, "pages.links.orphan_labels");
                assert_eq!(
                    owner_binding_digest.as_str(),
                    "9564fbd4194ceeb40ff8305b376a6263acb872733009ce606dfd896d1c747d25"
                );
                assert_eq!(choices, vec!["keep", "retarget", "remove"]);
                assert_eq!(suggested_research_queries, vec!["Find the canonical page"]);
            }
            _ => panic!("expected LintRepairReview"),
        }
    }

    #[test]
    fn lint_repair_review_payload_must_bind_id_and_source_owners_before_surfacing() {
        let occurrence = wenlan_types::repair::RepairDigest::parse(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        .unwrap();
        let bound_source_ids = vec!["page-a".to_string()];
        let raw = serde_json::json!({
            "action": "lint_repair_review",
            "check_id": "pages.links.orphan_labels",
            "occurrence_digest": occurrence,
            "owner_binding_digest":
                "9564fbd4194ceeb40ff8305b376a6263acb872733009ce606dfd896d1c747d25",
            "issue": "Review the ambiguous target.",
            "choices": ["keep", "retarget", "remove"],
            "suggested_research_queries": []
        })
        .to_string();

        assert!(parse_payload_for_proposal(
            "lint_review_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            &bound_source_ids,
            Some(&raw),
            "lint_repair_review",
        )
        .is_none());
        assert!(parse_payload_for_proposal(
            "lint_review_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &["page-b".to_string()],
            Some(&raw),
            "lint_repair_review",
        )
        .is_none());
        assert!(parse_payload_for_proposal(
            "lint_review_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &bound_source_ids,
            Some(
                r#"{"action":"lint_repair_review","occurrence_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#,
            ),
            "lint_repair_review",
        )
        .is_none());
    }

    #[test]
    fn parses_pick_space_accept_body() {
        let decision = parse_accept_decision(
            br#"{"action":"pick_space","space":"work","notes":"keep it with engineering"}"#,
        )
        .unwrap();
        match decision {
            RefinementDecision::PickSpace { space, notes } => {
                assert_eq!(space, "work");
                assert_eq!(notes.as_deref(), Some("keep it with engineering"));
            }
            _ => panic!("expected PickSpace decision"),
        }
    }

    #[test]
    fn malformed_json_returns_none() {
        let parsed = parse_payload(Some("not json"), "entity_merge");
        assert!(
            parsed.is_none(),
            "malformed payload should decode to None, not crash"
        );
    }

    #[test]
    fn parse_action_unknown_returns_none() {
        assert!(super::parse_action("entity_split_v2").is_none());
        assert!(super::parse_action("").is_none());
    }
}
