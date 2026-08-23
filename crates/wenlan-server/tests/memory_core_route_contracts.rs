// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::db::MemoryDB;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_server::{router::build_router, state::ServerState};
use wenlan_types::responses::{
    ConfirmResponse, DeleteResponse, ListMemoriesResponse, MemoryStatsResponse,
    NurtureCardsResponse, ReclassifyMemoryResponse, SearchMemoryResponse, StoreMemoryResponse,
    SuccessResponse, VersionChainResponse,
};
use wenlan_types::{
    ContradictionDismissResponse, EnrichmentStatusResponse, HomeStats, RecentActivityItem,
    RejectionRecord, RevisionAcceptResponse, RevisionDismissResponse,
};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

#[derive(Clone)]
struct RouteCase {
    method: Method,
    uri: &'static str,
    body: Option<&'static str>,
}

fn core_routes() -> Vec<RouteCase> {
    vec![
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/recent",
            body: None,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/unconfirmed",
            body: None,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/store",
            body: Some(r#"{"content":"route probe","memory_type":"fact"}"#),
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/search",
            body: Some(r#"{"query":"route probe","limit":5}"#),
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/confirm/probe",
            body: Some(r#"{"confirmed":true}"#),
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/list",
            body: Some(r#"{"limit":5}"#),
        },
        RouteCase {
            method: Method::DELETE,
            uri: "/api/memory/delete/probe",
            body: None,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/reclassify/probe",
            body: Some(r#"{"memory_type":"fact"}"#),
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/probe/enrichment-status",
            body: None,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/revision/probe/accept",
            body: None,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/revision/probe/dismiss",
            body: None,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/contradiction/probe/dismiss",
            body: None,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/stats",
            body: None,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/home-stats",
            body: None,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/nurture",
            body: None,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/rejections",
            body: None,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/memory/probe/versions",
            body: None,
        },
        RouteCase {
            method: Method::PUT,
            uri: "/api/memory/probe/update",
            body: Some(r#"{"content":"updated route probe"}"#),
        },
        RouteCase {
            method: Method::PUT,
            uri: "/api/memory/probe/stability",
            body: Some(r#"{"stability":"confirmed"}"#),
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/memory/probe/correct",
            body: Some(r#"{"correction_prompt":"make it exact"}"#),
        },
    ]
}

async fn request_bytes(
    router: &common::AppRouter,
    case: &RouteCase,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(case.method.clone()).uri(case.uri);
    if case.body.is_some() {
        request = request.header("content-type", "application/json");
    }
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "memory-core-route-contract");
    }
    let response = router
        .clone()
        .oneshot(
            request
                .body(case.body.map(Body::from).unwrap_or_else(Body::empty))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 512 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

async fn request_typed<T>(
    router: &common::AppRouter,
    method: Method,
    uri: &'static str,
    body: Option<&'static str>,
    space: Option<&str>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let case = RouteCase { method, uri, body };
    let (status, bytes) = request_bytes(router, &case, false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{} {uri} returned a non-contract response ({error}): {}",
            case.method,
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

async fn seed_memory(
    db: &Arc<MemoryDB>,
    source_id: &str,
    content: &str,
    supersedes: Option<&str>,
    pending_revision: bool,
    last_modified: i64,
) {
    common::insert_memory(
        db,
        source_id,
        content,
        "memory",
        Some("memory-core-route-contract"),
        supersedes,
        pending_revision,
        last_modified,
    )
    .await;
    db.update_memory_space(source_id, "work").await.unwrap();
}

#[tokio::test]
async fn memory_core_routes_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    for (source_id, content, supersedes, pending, modified) in [
        ("core-target", "core target memory", None, false, 100),
        ("core-delete", "delete this memory", None, false, 110),
        ("core-version-base", "base version", None, false, 120),
        (
            "core-version-current",
            "current version",
            Some("core-version-base"),
            false,
            130,
        ),
        ("accept-target", "accept target", None, false, 140),
        (
            "accept-revision",
            "accepted revision",
            Some("accept-target"),
            true,
            150,
        ),
        ("dismiss-target", "dismiss target", None, false, 160),
        (
            "dismiss-revision",
            "dismissed revision",
            Some("dismiss-target"),
            true,
            170,
        ),
    ] {
        seed_memory(&db, source_id, content, supersedes, pending, modified).await;
    }
    db.record_enrichment_step("core-target", "classify", "ok", None)
        .await
        .unwrap();

    let (status, stored): (StatusCode, StoreMemoryResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/store",
        Some(
            r#"{"content":"stored through core route","memory_type":"fact","space":"work","source_agent":"contract"}"#,
        ),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!stored.source_id.is_empty());
    assert_eq!(stored.space.as_deref(), Some("work"));

    let (status, search): (StatusCode, SearchMemoryResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/search",
        Some(r#"{"query":"core target","limit":10,"space":"work"}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(search.took_ms >= 0.0);

    let (status, confirmed): (StatusCode, ConfirmResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/confirm/core-target",
        Some(r#"{"confirmed":true}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(confirmed.confirmed);
    assert!(confirmed.updated, "a seeded id must report a touched row");

    // An unknown id must not report success: the UPDATE matches nothing.
    let (status, missing_confirm): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/memory/confirm/no-such-memory",
        Some(r#"{"confirmed":true}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing_confirm.error, "memory no-such-memory");

    let (status, listed): (StatusCode, ListMemoriesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/list",
        Some(r#"{"space":"work","limit":20}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(listed
        .memories
        .iter()
        .any(|memory| memory.source_id == "core-target"));

    let (status, reclassified): (StatusCode, ReclassifyMemoryResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/reclassify/core-target",
        Some(r#"{"memory_type":"decision"}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reclassified.source_id, "core-target");
    assert_eq!(reclassified.memory_type, "decision");

    let (status, enrichment): (StatusCode, EnrichmentStatusResponse) = request_typed(
        &router,
        Method::GET,
        "/api/memory/core-target/enrichment-status",
        None,
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(enrichment.source_id, "core-target");
    assert_eq!(enrichment.steps.len(), 1);

    let (status, accepted): (StatusCode, RevisionAcceptResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/revision/accept-target/accept",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(accepted.revision_source_id, "accept-revision");
    assert!(accepted.wrote);

    let (status, dismissed): (StatusCode, RevisionDismissResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/revision/dismiss-revision/dismiss",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(dismissed.target_source_id, "dismiss-target");
    assert!(dismissed.wrote);

    let (status, contradiction): (StatusCode, ContradictionDismissResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/contradiction/core-target/dismiss",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(contradiction.source_id, "core-target");
    assert!(contradiction.wrote);

    let (status, stats): (StatusCode, MemoryStatsResponse) =
        request_typed(&router, Method::GET, "/api/memory/stats", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(stats.stats.total >= 1);

    let (status, home): (StatusCode, HomeStats) =
        request_typed(&router, Method::GET, "/api/home-stats", None, Some("work")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(home.total >= 1);

    let (status, nurture): (StatusCode, NurtureCardsResponse) = request_typed(
        &router,
        Method::GET,
        "/api/memory/nurture?space=work&limit=3",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(nurture.cards.len() <= 3);

    let (status, rejections): (StatusCode, Vec<RejectionRecord>) = request_typed(
        &router,
        Method::GET,
        "/api/memory/rejections?limit=5",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(rejections.is_empty());

    let (status, versions): (StatusCode, VersionChainResponse) = request_typed(
        &router,
        Method::GET,
        "/api/memory/core-version-current/versions",
        None,
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(versions.versions.len(), 2);

    let (status, updated): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        "/api/memory/core-target/update",
        Some(r#"{"content":"updated core target"}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(updated.ok);

    let (status, stable): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        "/api/memory/core-target/stability",
        Some(r#"{"stability":"confirmed"}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(stable.ok);

    let (status, recent): (StatusCode, Vec<RecentActivityItem>) = request_typed(
        &router,
        Method::GET,
        "/api/memory/recent?limit=5",
        None,
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(recent.len() <= 5);

    let (status, unconfirmed): (StatusCode, Vec<RecentActivityItem>) = request_typed(
        &router,
        Method::GET,
        "/api/memory/unconfirmed?limit=5",
        None,
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(unconfirmed.len() <= 5);

    let (status, correction_error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/memory/core-target/correct",
        Some(r#"{"correction_prompt":"make it exact"}"#),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(correction_error.error, "LLM not available for correction");

    let (status, missing): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::GET,
        "/api/memory/missing-core-memory/versions",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing.error, "memory not found");

    let (status, deleted): (StatusCode, DeleteResponse) = request_typed(
        &router,
        Method::DELETE,
        "/api/memory/delete/core-delete",
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.deleted);

    for case in core_routes() {
        let (status, bytes) = request_bytes(&router, &case, true, None).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "{} {}",
            case.method,
            case.uri
        );
        let refusal: ErrorEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert!(refusal.error.contains("reader-intent marker"));
    }

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for case in core_routes() {
        let (status, bytes) = request_bytes(&no_db, &case, false, None).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{} {}: {}",
            case.method,
            case.uri,
            String::from_utf8_lossy(&bytes)
        );
        let error: ErrorEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error.error, "Database not initialized");
    }
}
