// SPDX-License-Identifier: Apache-2.0
//! Compatibility acceptance for the deprecated `/api/context` route.

mod common;

use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use tower::ServiceExt;
use wenlan_core::sources::RawDocument;
use wenlan_types::{
    responses::ChatContextResponse, BriefItemState, BriefMutation, BriefUpdateRequest,
};

async fn call_context(
    app: wenlan_server::router::AppRouter,
    body: serde_json::Value,
) -> ChatContextResponse {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/context")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn add_request(space: &str, operation_id: &str, text: &str) -> BriefUpdateRequest {
    BriefUpdateRequest {
        space: space.into(),
        caller_id: "context-compat-test".into(),
        operation_id: operation_id.into(),
        summary: None,
        mutations: vec![BriefMutation::Add {
            text: text.into(),
            state: BriefItemState::Active,
            added_at: None,
            gate: None,
        }],
    }
}

#[tokio::test]
async fn context_without_topic_adapts_brief_without_semantic_results() {
    let (app, _temp, db) = common::test_app().await;
    db.apply_brief_update(&add_request("work", "create", "ship the release"))
        .await
        .unwrap();
    db.upsert_documents(vec![RawDocument {
        source: "memory".into(),
        source_id: "fake-recent-context-hit".into(),
        title: "recent context".into(),
        content: "This must not be returned by a no-topic context call.".into(),
        last_modified: 1,
        confirmed: Some(true),
        space: Some("work".into()),
        ..Default::default()
    }])
    .await
    .unwrap();

    let response = call_context(app, serde_json::json!({"space": "work"})).await;

    assert!(response.context.contains("## Space Brief — work"));
    assert!(response.context.contains("ship the release"));
    assert!(!response.context.contains("This must not be returned"));
    assert!(response.knowledge.relevant_memories.is_empty());
    assert!(response.profile.identity.is_empty());
    assert!(response.profile.preferences.is_empty());
}

#[tokio::test]
async fn context_with_topic_keeps_brief_and_adds_only_same_space_results() {
    let (app, _temp, db) = common::test_app().await;
    db.apply_brief_update(&add_request("work", "create", "keep the complete brief"))
        .await
        .unwrap();
    db.create_space("personal", None, false).await.unwrap();
    for (source_id, content, space) in [
        ("work-topic", "release gate WORK_CONTEXT_MARKER", "work"),
        (
            "personal-topic",
            "release gate PERSONAL_CONTEXT_MARKER",
            "personal",
        ),
    ] {
        db.upsert_documents(vec![RawDocument {
            source: "memory".into(),
            source_id: source_id.into(),
            title: source_id.into(),
            content: content.into(),
            last_modified: 1,
            confirmed: Some(true),
            space: Some(space.into()),
            ..Default::default()
        }])
        .await
        .unwrap();
    }

    let response = call_context(
        app,
        serde_json::json!({"space": "work", "query": "release gate"}),
    )
    .await;

    assert!(response.context.contains("keep the complete brief"));
    assert!(response
        .context
        .contains("## Related Context — release gate"));
    assert!(response.context.contains("WORK_CONTEXT_MARKER"));
    assert!(!response.context.contains("PERSONAL_CONTEXT_MARKER"));
    assert!(response
        .knowledge
        .relevant_memories
        .iter()
        .all(|memory| memory.space.as_deref() == Some("work")));
}

#[tokio::test]
async fn context_without_a_resolved_space_returns_a_legacy_explanation_without_writing() {
    let (app, _temp, db) = common::test_app().await;

    let response = call_context(app, serde_json::json!({})).await;

    assert!(response.context.contains("No Space resolved"));
    assert!(response.knowledge.relevant_memories.is_empty());
    assert!(db.list_spaces().await.unwrap().is_empty());
}
