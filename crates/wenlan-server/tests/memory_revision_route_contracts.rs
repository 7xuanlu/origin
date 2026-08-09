// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_server::{router::build_router, state::ServerState};
use wenlan_types::responses::{ListMemoryRevisionsResponse, PendingRevision, PendingRevisionItem};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_bytes(
    router: &common::AppRouter,
    uri: &str,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(Method::GET).uri(uri);
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "memory-revision-route-contract");
    }
    let response = router
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

async fn request_typed<T>(
    router: &common::AppRouter,
    uri: &str,
    space: Option<&str>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, uri, false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "GET {uri} returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

async fn seed_memory(
    db: &Arc<wenlan_core::db::MemoryDB>,
    source_id: &str,
    content: &str,
    supersedes: Option<&str>,
    pending_revision: bool,
    last_modified: i64,
    space: &str,
) {
    common::insert_memory(
        db,
        source_id,
        content,
        "memory",
        Some("memory-revision-route-contract"),
        supersedes,
        pending_revision,
        last_modified,
    )
    .await;
    db.update_memory_space(source_id, space).await.unwrap();
}

#[tokio::test]
async fn moved_memory_revision_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();

    for (source_id, content, supersedes, last_modified, space) in [
        ("work-chain-base", "Work chain base.", None, 100, "work"),
        (
            "work-chain-current",
            "Work chain current.",
            Some("work-chain-base"),
            200,
            "work",
        ),
        (
            "personal-chain-base",
            "Personal chain base.",
            None,
            300,
            "personal",
        ),
        (
            "personal-chain-current",
            "Personal chain current.",
            Some("personal-chain-base"),
            400,
            "personal",
        ),
        (
            "work-pending-target",
            "Work pending target.",
            None,
            500,
            "work",
        ),
        (
            "personal-pending-target",
            "Personal pending target.",
            None,
            600,
            "personal",
        ),
    ] {
        seed_memory(
            &db,
            source_id,
            content,
            supersedes,
            false,
            last_modified,
            space,
        )
        .await;
    }
    for (source_id, content, target, last_modified, space) in [
        (
            "work-pending-revision",
            "Work staged revision.",
            "work-pending-target",
            700,
            "work",
        ),
        (
            "personal-pending-revision",
            "Personal staged revision.",
            "personal-pending-target",
            800,
            "personal",
        ),
    ] {
        seed_memory(
            &db,
            source_id,
            content,
            Some(target),
            true,
            last_modified,
            space,
        )
        .await;
    }

    let (status, history): (StatusCode, ListMemoryRevisionsResponse) = request_typed(
        &router,
        "/api/memory/work-chain-current/revisions",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(history.current_source_id, "work-chain-current");
    assert_eq!(history.chain_depth, 1);
    assert_eq!(history.entries.len(), 2);
    assert_eq!(history.entries[0].source_id, "work-chain-current");
    assert_eq!(history.entries[1].source_id, "work-chain-base");

    let (status, items): (StatusCode, Vec<PendingRevisionItem>) = request_typed(
        &router,
        "/api/memory/pending-revisions?limit=10",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].target_source_id, "work-pending-target");
    assert_eq!(items[0].revision_source_id, "work-pending-revision");
    assert_eq!(items[0].revision_content, "Work staged revision.");

    let (status, revision): (StatusCode, Option<PendingRevision>) = request_typed(
        &router,
        "/api/memory/pending-revision/work-pending-target",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let revision = revision.expect("work pending revision");
    assert_eq!(revision.source_id, "work-pending-revision");
    assert_eq!(revision.content, "Work staged revision.");

    for uri in [
        "/api/memory/personal-chain-current/revisions",
        "/api/memory/missing/revisions",
        "/api/memory/pending-revision/personal-pending-target",
        "/api/memory/pending-revision/missing",
    ] {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&router, uri, Some("work")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error.error, "memory not found");
    }

    for uri in [
        "/api/memory/work-chain-current/revisions",
        "/api/memory/pending-revisions",
        "/api/memory/pending-revision/work-pending-target",
    ] {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&router, uri, Some("missing-space")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error.error, "unknown Space: missing-space");

        let (status, body) = request_bytes(&router, uri, true, None).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
        assert!(refusal.error.contains("reader-intent marker"));
    }

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for uri in [
        "/api/memory/missing/revisions",
        "/api/memory/pending-revisions",
        "/api/memory/pending-revision/missing",
    ] {
        let (status, error): (StatusCode, ErrorEnvelope) = request_typed(&no_db, uri, None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.error, "Database not initialized");
    }
}
