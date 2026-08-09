// SPDX-License-Identifier: Apache-2.0
//! Integration tests for /api/refinery/queue endpoints.
//!
//! Drives the axum Router via tower::ServiceExt::oneshot — no TCP socket.
//! Tests: list filtering, typed payload decode, empty queue,
//! reject happy + default agent + 404 + 422.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;
use wenlan_core::sources::RawDocument;
use wenlan_server::router::{build_router, AppRouter};
use wenlan_server::state::ServerState;

fn owner_binding_digest(digest: &str, source_ids: &[String]) -> String {
    Sha256::digest(
        serde_json::to_vec(&serde_json::json!({
            "occurrence_digest": digest,
            "source_ids": source_ids,
        }))
        .unwrap(),
    )
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect()
}

async fn test_app() -> (AppRouter, tempfile::TempDir, Arc<MemoryDB>) {
    let dir = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    let db_arc = Arc::new(db);
    let state = ServerState {
        db: Some(db_arc.clone()),
        ..ServerState::default()
    };
    let router = build_router(Arc::new(RwLock::new(state)));
    (router, dir, db_arc)
}

/// Insert a proposal and immediately transition it to `awaiting_review` so the
/// list route (which filters on that status) can surface it.
async fn seed_awaiting(
    db: &Arc<MemoryDB>,
    id: &str,
    action: &str,
    payload: Option<&str>,
    confidence: f64,
) {
    db.insert_refinement_proposal(
        id,
        action,
        &["src_a".to_string(), "src_b".to_string()],
        payload,
        confidence,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open(id, "awaiting_review")
        .await
        .unwrap();
}

async fn read_body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn seed_memory(db: &Arc<MemoryDB>, source_id: &str, content: &str, space: &str) {
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        content: content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        space: Some(space.to_string()),
        source_agent: Some("test".to_string()),
        confidence: Some(0.9),
        ..Default::default()
    };
    db.upsert_documents(vec![doc]).await.unwrap();
}

#[tokio::test]
async fn list_queue_returns_only_awaiting_review() {
    let (app, _tmp, db) = test_app().await;
    seed_awaiting(&db, "ref_a", "entity_merge", None, 0.86).await;
    seed_awaiting(&db, "ref_b", "entity_merge", None, 0.87).await;
    // ref_c: insert then immediately dismiss — should not appear
    db.insert_refinement_proposal("ref_c", "entity_merge", &["src_a".to_string()], None, 0.88)
        .await
        .unwrap();
    db.resolve_refinement_if_open("ref_c", "dismissed")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    let proposals = body["proposals"].as_array().unwrap();
    assert_eq!(proposals.len(), 2, "dismissed proposal must not appear");
}

#[tokio::test]
async fn list_queue_filters_by_action() {
    let (app, _tmp, db) = test_app().await;
    seed_awaiting(&db, "r1", "entity_merge", None, 0.86).await;
    seed_awaiting(&db, "r2", "relation_conflict", None, 0.7).await;
    seed_awaiting(&db, "r3", "suggest_entity", Some("\"X\""), 0.9).await;

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue?action=entity_merge")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    let proposals = body["proposals"].as_array().unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0]["action"], "entity_merge");
}

#[tokio::test]
async fn list_queue_decodes_typed_payload() {
    let (app, _tmp, db) = test_app().await;
    let payload = r#"{"existing_id":"e1","new_id":"e2","similarity":0.86}"#;
    seed_awaiting(&db, "ref_typed", "entity_merge", Some(payload), 0.86).await;

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    // RefinementPayload is a serde internally-tagged enum with tag "action",
    // so the serialized object carries `"action": "entity_merge"` alongside the fields.
    let payload_val = &body["proposals"][0]["payload"];
    assert_eq!(payload_val["action"], "entity_merge");
    assert_eq!(payload_val["existing_id"], "e1");
    assert_eq!(payload_val["new_id"], "e2");
}

#[tokio::test]
async fn list_queue_surfaces_typed_lint_repair_review() {
    let (app, _tmp, db) = test_app().await;
    let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let source_ids = vec!["page-a".to_string(), "page-b".to_string()];
    let payload = serde_json::json!({
        "action": "lint_repair_review",
        "check_id": "pages.links.orphan_labels",
        "occurrence_digest": digest,
        "owner_binding_digest": owner_binding_digest(digest, &source_ids),
        "issue": "Review the ambiguous target.",
        "choices": ["keep", "retarget", "remove"],
        "suggested_research_queries": ["Find the canonical page"],
    })
    .to_string();
    assert!(db
        .insert_lint_review_if_absent(
            "lint_review_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &source_ids,
            &payload,
        )
        .await
        .unwrap());

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue?action=lint_repair_review")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    let proposals = body["proposals"].as_array().unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0]["action"], "lint_repair_review");
    assert_eq!(proposals[0]["payload"]["action"], "lint_repair_review");
    assert_eq!(
        proposals[0]["payload"]["check_id"],
        "pages.links.orphan_labels"
    );
    assert_eq!(
        proposals[0]["payload"]["choices"],
        serde_json::json!(["keep", "retarget", "remove"])
    );
}

#[tokio::test]
async fn list_queue_default_does_not_hide_lint_review_backlog() {
    let (app, _tmp, db) = test_app().await;
    for index in 0..81 {
        let digest = format!("{:064x}", index + 1);
        let source_ids = vec!["page-a".to_string()];
        let payload = serde_json::json!({
            "action": "lint_repair_review",
            "check_id": "pages.links.orphan_labels",
            "occurrence_digest": digest,
            "owner_binding_digest": owner_binding_digest(&digest, &source_ids),
            "issue": "Review the ambiguous target.",
            "choices": ["keep", "retarget", "remove"],
            "suggested_research_queries": [],
        })
        .to_string();
        assert!(db
            .insert_lint_review_if_absent(&format!("lint_review_{digest}"), &source_ids, &payload,)
            .await
            .unwrap());
    }

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue?action=lint_repair_review")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert_eq!(body["proposals"].as_array().unwrap().len(), 81);
}

#[tokio::test]
async fn list_queue_skips_malformed_lint_repair_review_instead_of_returning_null_payload() {
    let (app, tmp, db) = test_app().await;
    let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let id = format!("lint_review_{digest}");
    let source_ids = vec!["page-a".to_string()];
    let payload = serde_json::json!({
        "action": "lint_repair_review",
        "check_id": "pages.links.orphan_labels",
        "occurrence_digest": digest,
        "owner_binding_digest": owner_binding_digest(digest, &source_ids),
        "issue": "Review the ambiguous target.",
        "choices": ["keep", "retarget", "remove"],
        "suggested_research_queries": [],
    })
    .to_string();
    assert!(db
        .insert_lint_review_if_absent(&id, &source_ids, &payload)
        .await
        .unwrap());

    let db_file = tmp.path().join("origin_memory.db");
    let raw_db = libsql::Builder::new_local(db_file.to_str().unwrap())
        .build()
        .await
        .unwrap();
    raw_db
        .connect()
        .unwrap()
        .execute(
            "UPDATE refinement_queue SET payload=?1 WHERE id=?2",
            libsql::params![
                r#"{"action":"lint_repair_review","occurrence_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#,
                id
            ],
        )
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue?action=lint_repair_review")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert!(body["proposals"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn list_queue_empty_returns_200() {
    let (app, _tmp, _db) = test_app().await;

    let req = Request::builder()
        .method(Method::GET)
        .uri("/api/refinery/queue")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert_eq!(body["proposals"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn reject_marks_dismissed_and_logs_activity() {
    let (app, _tmp, db) = test_app().await;
    // Reject works from 'pending' status directly (resolve_proposal only guards terminal states)
    db.insert_refinement_proposal(
        "ref_rej_1",
        "detect_contradiction",
        &["src_a".to_string()],
        None,
        0.8,
    )
    .await
    .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_rej_1/reject")
        .header("x-agent-name", "claude-code")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify queue row dismissed
    let pending = db.get_pending_refinements().await.unwrap();
    assert!(!pending.iter().any(|p| p.id == "ref_rej_1"));

    // Verify activity row logged with x-agent-name value
    let acts = db
        .list_agent_activity(10, Some("claude-code"), None)
        .await
        .unwrap_or_default();
    assert!(
        acts.iter().any(|a| a.action == "refinement_resolve"),
        "activity row should be logged with x-agent-name value"
    );
}

#[tokio::test]
async fn reject_default_agent_when_header_missing() {
    let (app, _tmp, db) = test_app().await;
    db.insert_refinement_proposal(
        "ref_no_agent",
        "entity_merge",
        &["src_a".to_string()],
        None,
        0.86,
    )
    .await
    .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_no_agent/reject")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify queue row dismissed regardless of which agent name was used
    let pending = db.get_pending_refinements().await.unwrap();
    assert!(!pending.iter().any(|p| p.id == "ref_no_agent"));

    // extract_agent_name defaults to "unknown" when header is absent
    let acts = db
        .list_agent_activity(10, Some("unknown"), None)
        .await
        .unwrap_or_default();
    assert!(
        acts.iter().any(|a| a.action == "refinement_resolve"),
        "default-agent path should log activity under 'unknown'"
    );
}

#[tokio::test]
async fn reject_unknown_id_returns_404() {
    let (app, _tmp, _db) = test_app().await;

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_does_not_exist/reject")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn reject_already_terminal_returns_422() {
    let (app, _tmp, db) = test_app().await;
    db.insert_refinement_proposal(
        "ref_done",
        "entity_merge",
        &["src_a".to_string()],
        None,
        0.86,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_done", "dismissed")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_done/reject")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// ── accept endpoint tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn accept_entity_merge_flow() {
    let (app, _tmp, db) = test_app().await;

    let new_id = db
        .store_entity("Acme Corporation", "organization", None, None, None)
        .await
        .unwrap();
    let existing_id = db
        .store_entity("Acme Corp", "organization", None, None, None)
        .await
        .unwrap();
    db.insert_refinement_proposal(
        "ref_accept_em",
        "entity_merge",
        &[new_id.clone(), existing_id.clone()],
        None,
        0.87,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_accept_em", "awaiting_review")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_accept_em/accept")
        .header("x-agent-name", "test-agent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert_eq!(body["id"], "ref_accept_em");
    assert_eq!(body["action_applied"], "entity_merge");

    // existing_id (canonical) survives; new_id (alias) is merged away
    assert!(
        db.get_entity_detail(&existing_id).await.is_ok(),
        "canonical entity should still exist"
    );
    assert!(
        db.get_entity_detail(&new_id).await.is_err(),
        "merged-away entity should be deleted"
    );
}

#[tokio::test]
async fn accept_stale_empty_json_body_applies_default_accept() {
    let (app, _tmp, db) = test_app().await;
    db.insert_refinement_proposal(
        "ref_accept_empty_json",
        "detect_contradiction",
        &["src_a".to_string(), "src_b".to_string()],
        None,
        0.8,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_accept_empty_json", "awaiting_review")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_accept_empty_json/accept")
        .header("x-agent-name", "test-agent")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert_eq!(body["id"], "ref_accept_empty_json");
    assert_eq!(body["action_applied"], "detect_contradiction");
}

#[tokio::test]
async fn accept_tagged_accept_body_applies_default_accept() {
    let (app, _tmp, db) = test_app().await;
    db.insert_refinement_proposal(
        "ref_accept_tagged",
        "detect_contradiction",
        &["src_a".to_string(), "src_b".to_string()],
        None,
        0.8,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_accept_tagged", "awaiting_review")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_accept_tagged/accept")
        .header("x-agent-name", "test-agent")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"action":"accept"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert_eq!(body["id"], "ref_accept_tagged");
    assert_eq!(body["action_applied"], "detect_contradiction");
}

#[tokio::test]
async fn accept_pick_space_body_applies_cross_space_discovery() {
    let (app, _tmp, db) = test_app().await;
    for (source_id, space) in [
        ("route_pick_work_a", "work"),
        ("route_pick_personal_a", "personal"),
        ("route_pick_work_b", "work"),
    ] {
        seed_memory(
            &db,
            source_id,
            "Incremental Rust compilation cache tuning for shared developer machines.",
            space,
        )
        .await;
    }
    let source_ids = vec![
        "route_pick_work_a".to_string(),
        "route_pick_personal_a".to_string(),
        "route_pick_work_b".to_string(),
    ];
    db.insert_refinement_proposal(
        "ref_accept_pick_space",
        "cross_space_discovery",
        &source_ids,
        Some(
            r#"{"action":"cross_space_discovery","memory_count":3,"spaces":["personal","work"],"allowed_actions":["dismiss","pick_space"]}"#,
        ),
        1.0,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_accept_pick_space", "awaiting_review")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_accept_pick_space/accept")
        .header("x-agent-name", "test-agent")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"action":"pick_space","space":"work"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body_json(resp).await;
    assert_eq!(body["id"], "ref_accept_pick_space");
    assert_eq!(body["action_applied"], "cross_space_discovery");
    let pages = db.list_pages("active", 10, 0).await.unwrap();
    assert_eq!(pages.len(), 1, "pick_space should create exactly one page");
    assert_eq!(pages[0].workspace.as_deref(), Some("work"));
    assert_eq!(pages[0].space.as_deref(), Some("work"));
}

#[tokio::test]
async fn accept_garbage_body_returns_400() {
    let (app, _tmp, _db) = test_app().await;
    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_parse_first/accept")
        .header("x-agent-name", "test-agent")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn accept_returns_404_for_unknown_id() {
    let (app, _tmp, _db) = test_app().await;
    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_nonexistent/accept")
        .header("x-agent-name", "test-agent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn accept_returns_422_for_suggest_entity() {
    let (app, _tmp, db) = test_app().await;
    db.insert_refinement_proposal(
        "ref_se_route",
        "suggest_entity",
        &["x".into()],
        Some("\"Acme\""),
        0.9,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_se_route", "awaiting_review")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_se_route/accept")
        .header("x-agent-name", "test-agent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn accept_lint_repair_review_returns_choice_specific_422() {
    let (app, _tmp, db) = test_app().await;
    let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let id = format!("lint_review_{digest}");
    let source_ids = vec!["page-a".to_string()];
    let payload = serde_json::json!({
        "action": "lint_repair_review",
        "check_id": "pages.links.orphan_labels",
        "occurrence_digest": digest,
        "owner_binding_digest": owner_binding_digest(digest, &source_ids),
        "issue": "Review the ambiguous target.",
        "choices": ["keep", "retarget", "remove"],
        "suggested_research_queries": [],
    })
    .to_string();
    db.insert_lint_review_if_absent(&id, &source_ids, &payload)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/api/refinery/queue/{id}/accept"))
        .header("x-agent-name", "test-agent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let body = read_body_json(resp).await;
    assert_eq!(
        body["error"],
        "action 'lint_repair_review' requires a choice-specific repair flow"
    );
    assert_eq!(
        db.get_refinement_proposal(&id)
            .await
            .unwrap()
            .unwrap()
            .status,
        "awaiting_review"
    );
}

#[tokio::test]
async fn accept_returns_422_for_already_resolved() {
    let (app, _tmp, db) = test_app().await;
    db.insert_refinement_proposal(
        "ref_done_accept",
        "entity_merge",
        &["a".into(), "b".into()],
        None,
        0.85,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_done_accept", "resolved")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_done_accept/accept")
        .header("x-agent-name", "test-agent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn accept_logs_apply_and_resolve_activity_with_agent() {
    let (app, _tmp, db) = test_app().await;
    let new_id = db
        .store_entity("Acme Corporation", "organization", None, None, None)
        .await
        .unwrap();
    let existing_id = db
        .store_entity("Acme Corp", "organization", None, None, None)
        .await
        .unwrap();
    db.insert_refinement_proposal(
        "ref_acc_log",
        "entity_merge",
        &[new_id, existing_id],
        None,
        0.87,
    )
    .await
    .unwrap();
    db.resolve_refinement_if_open("ref_acc_log", "awaiting_review")
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/refinery/queue/ref_acc_log/accept")
        .header("x-agent-name", "claude-code")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let acts = db
        .list_agent_activity(20, Some("claude-code"), None)
        .await
        .unwrap();
    assert!(
        acts.iter().any(|a| a.action == "refinement_apply"),
        "should log refinement_apply"
    );
    assert!(
        acts.iter().any(|a| a.action == "refinement_resolve"),
        "should log refinement_resolve"
    );
}
