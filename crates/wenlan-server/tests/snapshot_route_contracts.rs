// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::db::{MemoryDB, SessionSnapshotRow};
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_server::{router::build_router, state::ServerState};
use wenlan_types::responses::SuccessResponse;
use wenlan_types::{SessionSnapshot, SnapshotCapture, SnapshotCaptureWithContent};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_bytes(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "snapshot-route-contract");
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
    method: Method,
    uri: &str,
    space: Option<&str>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, method.clone(), uri, false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{} {uri} returned a non-contract response ({error}): {}",
            method,
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

async fn seed_capture(
    db: &Arc<MemoryDB>,
    source_id: &str,
    content: &str,
    space: &str,
    snapshot_id: &str,
    timestamp: i64,
) {
    common::insert_memory(
        db,
        source_id,
        content,
        "focus_capture",
        Some("snapshot-route-contract"),
        None,
        false,
        timestamp,
    )
    .await;
    db.update_memory_space(source_id, space).await.unwrap();
    db.insert_capture_ref(
        source_id,
        "activity-mixed",
        Some(snapshot_id),
        &format!("{space} app"),
        &format!("{space} window"),
        timestamp,
        "focus",
    )
    .await
    .unwrap();
}

async fn seed_snapshot(db: &Arc<MemoryDB>, id: &str, started_at: i64, capture_count: usize) {
    db.insert_snapshot(&SessionSnapshotRow {
        id: id.to_string(),
        activity_id: format!("activity-{id}"),
        started_at,
        ended_at: started_at + 10,
        primary_apps: vec![format!("{id} app")],
        summary: format!("{id} summary"),
        tags: vec![format!("{id} tag")],
        capture_count,
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn moved_snapshot_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();

    seed_snapshot(&db, "snapshot-old", 100, 0).await;
    seed_snapshot(&db, "snapshot-personal", 200, 1).await;
    seed_snapshot(&db, "snapshot-mixed", 300, 2).await;
    seed_capture(
        &db,
        "work-capture",
        "work capture content",
        "work",
        "snapshot-mixed",
        301,
    )
    .await;
    seed_capture(
        &db,
        "personal-capture",
        "personal capture content",
        "personal",
        "snapshot-mixed",
        302,
    )
    .await;
    seed_capture(
        &db,
        "personal-only-capture",
        "personal-only capture content",
        "personal",
        "snapshot-personal",
        201,
    )
    .await;

    let (status, snapshots): (StatusCode, Vec<SessionSnapshot>) = request_typed(
        &router,
        Method::GET,
        "/api/snapshots?limit=1",
        Some("missing-space"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].id, "snapshot-mixed");
    assert_eq!(snapshots[0].capture_count, 2);

    let (status, captures): (StatusCode, Vec<SnapshotCapture>) = request_typed(
        &router,
        Method::GET,
        "/api/snapshots/snapshot-mixed/captures",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].source_id, "work-capture");
    assert_eq!(captures[0].window_title, "work window");

    let (status, captures): (StatusCode, Vec<SnapshotCaptureWithContent>) = request_typed(
        &router,
        Method::GET,
        "/api/snapshots/snapshot-mixed/captures-with-content",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].source_id, "work-capture");
    assert_eq!(captures[0].content, "work capture content");

    for suffix in ["captures", "captures-with-content"] {
        for id in ["snapshot-personal", "snapshot-missing"] {
            let uri = format!("/api/snapshots/{id}/{suffix}");
            let (status, error): (StatusCode, ErrorEnvelope) =
                request_typed(&router, Method::GET, &uri, Some("work")).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            assert_eq!(error.error, "snapshot not found");
        }

        let uri = format!("/api/snapshots/snapshot-mixed/{suffix}");
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&router, Method::GET, &uri, Some("missing-space")).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error.error, "unknown Space: missing-space");
    }

    let routes = [
        (Method::GET, "/api/snapshots"),
        (Method::GET, "/api/snapshots/snapshot-mixed/captures"),
        (
            Method::GET,
            "/api/snapshots/snapshot-mixed/captures-with-content",
        ),
        (Method::POST, "/api/snapshots/snapshot-mixed/delete"),
    ];
    for (method, uri) in &routes {
        let (status, bytes) = request_bytes(&router, method.clone(), uri, true, None).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let refusal: ErrorEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert!(refusal.error.contains("reader-intent marker"));
    }

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for (method, uri) in &routes {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&no_db, method.clone(), uri, None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.error, "Database not initialized");
    }

    let (status, deleted): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::POST,
        "/api/snapshots/snapshot-mixed/delete",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.ok);

    let (status, snapshots): (StatusCode, Vec<SessionSnapshot>) =
        request_typed(&router, Method::GET, "/api/snapshots?limit=10", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        snapshots
            .iter()
            .map(|snapshot| snapshot.id.as_str())
            .collect::<Vec<_>>(),
        ["snapshot-personal", "snapshot-old"]
    );
}
