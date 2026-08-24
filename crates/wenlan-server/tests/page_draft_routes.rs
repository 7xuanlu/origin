// SPDX-License-Identifier: Apache-2.0
//! HTTP-surface contract for the page-draft routes.
//!
//! The desktop editor's autosave/publish/discard chain calls these four
//! endpoints through `app/src/search.rs`, and `src/lib/tauri.ts`
//! (`parsePageDraftError`) parses the error bodies for a top-level `code`.
//! These tests pin the wire shapes at the daemon boundary — paths, envelopes,
//! status codes, and the code-tagged conflict bodies — against the executable
//! spec in `e2e/tauriMock/runtime.ts`.
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

const DRAFT_A: &str = "page_00000000-0000-4000-8000-0000000000a1";
const DRAFT_B: &str = "page_00000000-0000-4000-8000-0000000000b2";

async fn send(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Option<Value>,
    space_header: Option<&str>,
) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(uri);
    if body.is_some() {
        request = request.header("content-type", "application/json");
    }
    if let Some(space) = space_header {
        request = request.header("x-wenlan-space", space);
    }
    let response = router
        .clone()
        .oneshot(
            request
                .body(
                    body.map(|value| Body::from(serde_json::to_vec(&value).unwrap()))
                        .unwrap_or_else(Body::empty),
                )
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "non-JSON response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, value)
}

#[tokio::test]
async fn draft_lifecycle_create_update_publish_discard() {
    let (router, _tmp, _db) = common::test_app().await;

    // Create: envelope is {"page": ...}, the draft starts at version 1.
    let create = json!({
        "draft_id": DRAFT_A,
        "title": "  Launch checklist ",
        "content": "- verify the artifacts",
    });
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(create.clone()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["id"], DRAFT_A);
    assert_eq!(body["page"]["status"], "draft");
    assert_eq!(body["page"]["version"], 1);
    assert_eq!(body["page"]["title"], "  Launch checklist ");

    // An ambiguous create retry (same id, same snapshot) replays the draft.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(create),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["version"], 1);

    // Autosave: complete-snapshot replace bumps the version.
    let (status, body) = send(
        &router,
        Method::PUT,
        &format!("/api/pages/drafts/{DRAFT_A}"),
        Some(json!({
            "expected_version": 1,
            "title": "  Launch checklist ",
            "content": "- verify the artifacts\n- write the findings doc",
            "space": null,
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["version"], 2);

    // Publish flips the row active, trims the title, and bumps the version.
    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 2})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["status"], "active");
    assert_eq!(body["page"]["title"], "Launch checklist");
    assert_eq!(body["page"]["version"], 3);

    // An exact publish retry replays the already-active page.
    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 2})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["version"], 3);

    // Discard: a fresh draft deletes with {"status":"deleted"}.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({
            "draft_id": DRAFT_B,
            "title": "Scratch",
            "content": "",
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = send(
        &router,
        Method::DELETE,
        &format!("/api/pages/drafts/{DRAFT_B}"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"status": "deleted"}));
}

#[tokio::test]
async fn draft_error_bodies_carry_the_editor_codes() {
    let (router, _tmp, _db) = common::test_app().await;

    // Unknown draft -> 404 page_draft_not_found (update, publish, discard).
    for (method, uri, body) in [
        (
            Method::PUT,
            format!("/api/pages/drafts/{DRAFT_A}"),
            json!({"expected_version": 1, "title": "t", "content": "c", "space": null}),
        ),
        (
            Method::POST,
            format!("/api/pages/drafts/{DRAFT_A}/publish"),
            json!({"expected_version": 1}),
        ),
        (
            Method::DELETE,
            format!("/api/pages/drafts/{DRAFT_A}"),
            json!({"expected_version": 1}),
        ),
    ] {
        let (status, body) = send(&router, method.clone(), &uri, Some(body), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {uri}: {body}");
        assert_eq!(body["code"], "page_draft_not_found", "{method} {uri}");
        assert_eq!(body["error"], "Page draft not found");
    }

    // An empty snapshot is a plain 422 with no editor code.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_A, "title": " ", "content": ""})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body.get("code").is_none(), "{body}");

    // Seed a draft for the conflict cases.
    let (status, _) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_A, "title": "Draft", "content": "Body"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Reusing the id with a different snapshot -> 409 page_draft_id_conflict.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_A, "title": "Different", "content": "Body"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "page_draft_id_conflict");

    // A stale version -> 409 draft_version_conflict carrying current_version.
    for (method, uri, body) in [
        (
            Method::PUT,
            format!("/api/pages/drafts/{DRAFT_A}"),
            json!({"expected_version": 9, "title": "t", "content": "c", "space": null}),
        ),
        (
            Method::POST,
            format!("/api/pages/drafts/{DRAFT_A}/publish"),
            json!({"expected_version": 9}),
        ),
        (
            Method::DELETE,
            format!("/api/pages/drafts/{DRAFT_A}"),
            json!({"expected_version": 9}),
        ),
    ] {
        let (status, body) = send(&router, method.clone(), &uri, Some(body), None).await;
        assert_eq!(status, StatusCode::CONFLICT, "{method} {uri}: {body}");
        assert_eq!(body["code"], "draft_version_conflict", "{method} {uri}");
        assert_eq!(body["error"], "Page draft changed since it was loaded");
        assert_eq!(body["current_version"], 1, "{method} {uri}");
    }

    // Publish the draft, then a same-scope draft with a case-insensitively
    // equal title -> 409 page_title_conflict naming the existing page.
    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, _) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_B, "title": "  DRAFT ", "content": "Other body"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_B}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "page_title_conflict");
    assert_eq!(body["error"], "A Page with this title already exists");
    assert_eq!(body["existing_page_id"], DRAFT_A);
    assert_eq!(body["existing_page_title"], "Draft");
}

#[tokio::test]
async fn create_space_semantics_header_inheritance_and_registration() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("work", None, false).await.unwrap();

    // Omitted `space` key inherits the X-Wenlan-Space header.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_A, "title": "Scoped", "content": ""})),
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["space"], "work");

    // An explicit `"space": null` overrides the header: unscoped draft.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_B, "title": "Unscoped", "content": "", "space": null})),
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["page"]["space"], Value::Null);

    // An unregistered space is rejected before anything persists.
    let (status, body) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({
            "draft_id": "page_00000000-0000-4000-8000-0000000000c3",
            "title": "Nowhere",
            "content": "",
            "space": "no-such-space",
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body.get("code").is_none(), "{body}");
}
