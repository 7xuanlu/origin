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

    // A queued update or discard racing in after publish gets the structured
    // 404 (only draft rows are findable as drafts) — the editor's discard
    // path treats page_draft_not_found as completed cleanup.
    let (status, body) = send(
        &router,
        Method::PUT,
        &format!("/api/pages/drafts/{DRAFT_A}"),
        Some(json!({
            "expected_version": 3,
            "title": "Launch checklist",
            "content": "Late autosave",
            "space": null,
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["code"], "page_draft_not_found");
    let (status, body) = send(
        &router,
        Method::DELETE,
        &format!("/api/pages/drafts/{DRAFT_A}"),
        Some(json!({"expected_version": 3})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["code"], "page_draft_not_found");

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

/// Stamp a judged-Unsupported verdict on an active page.
///
/// The daemon deliberately exposes no verdict-writing API, so this mirrors the
/// finalizer's two writes directly against the db file — the same raw-SQL seam
/// `refinery_routes.rs` uses. `extractor_version` is hardcoded to mirror
/// `claim_derivation::EXTRACTOR_VERSION` (private); if that constant bumps,
/// the verdict decays to `Unevaluated`, the page stays visible, and the hiding
/// assertions below fail loudly at this comment.
async fn mark_page_unsupported(
    tmp: &tempfile::TempDir,
    db: &std::sync::Arc<wenlan_core::db::MemoryDB>,
    page_id: &str,
) {
    let page = db
        .get_page(page_id)
        .await
        .unwrap()
        .expect("page under test exists");
    let digest = wenlan_core::provenance::revision_content_digest(&page.content);
    let raw = libsql::Builder::new_local(tmp.path().join("origin_memory.db").to_str().unwrap())
        .build()
        .await
        .unwrap();
    let conn = raw.connect().unwrap();
    conn.execute(
        "INSERT INTO page_truth_state
             (page_id, page_version, support_status, provisional_reason,
              evaluated_at, human_reviewed, updated_at)
         VALUES (?1, ?2, 'provisional', 'refuted by fixture', 1, 0, 1)
         ON CONFLICT(page_id) DO UPDATE SET
             page_version = ?2, support_status = 'provisional',
             provisional_reason = 'refuted by fixture', evaluated_at = 1,
             human_reviewed = 0, updated_at = 1",
        libsql::params![page_id, page.version],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO claim_derivation_markers
             (page_id, page_version, page_version_digest, extractor_version,
              inventory_count, created_at)
         VALUES (?1, ?2, ?3, 1, 1, 0)",
        libsql::params![page_id, page.version, digest],
    )
    .await
    .unwrap();
    db.set_app_metadata("claim_promoter_enforcement", "1")
        .await
        .unwrap();
    db.set_truth_cutover_generation(1).await.unwrap();
}

/// Live-cutover teeth for `filter_draft_echo`: every other test in this file
/// runs at generation 0, where the truth adapters are pass-through — removing
/// the filter calls would leave them all green. This one hides the published
/// page and requires the publish replay to stop echoing it.
#[tokio::test]
async fn publish_replay_of_a_hidden_page_reports_draft_not_found() {
    let (router, tmp, db) = common::test_app().await;
    let (status, _) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_A, "title": "Hidden plan", "content": "Body"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Control: with no verdict the exact replay echoes the active page, so
    // the 404 below can only come from the verdict stamped in between.
    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    mark_page_unsupported(&tmp, &db, DRAFT_A).await;

    // For this automatic caller the hidden page is not there, and the replay
    // must say so rather than echo prose the truth contract withholds.
    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["code"], "page_draft_not_found", "{body}");
}

/// The 409 must survive, but a hidden page's id and title are that page's
/// content and stay off the wire for a caller the truth contract hides it
/// from. Both optional fields are asserted present before the verdict and
/// absent after it, so this fails if either the leak returns or the fixture
/// stops hiding the page.
#[tokio::test]
async fn title_conflict_with_a_hidden_page_omits_its_identity() {
    let (router, tmp, db) = common::test_app().await;
    let (status, _) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_A, "title": "Secret plan", "content": "Body"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_A}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &router,
        Method::POST,
        "/api/pages/drafts",
        Some(json!({"draft_id": DRAFT_B, "title": "  SECRET PLAN ", "content": "Other"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Control: while the page is visible the conflict names it.
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
    assert_eq!(body["existing_page_id"], DRAFT_A, "{body}");
    assert_eq!(body["existing_page_title"], "Secret plan", "{body}");

    mark_page_unsupported(&tmp, &db, DRAFT_A).await;

    let (status, body) = send(
        &router,
        Method::POST,
        &format!("/api/pages/drafts/{DRAFT_B}/publish"),
        Some(json!({"expected_version": 1})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["code"], "page_title_conflict", "{body}");
    assert_eq!(body["error"], "A Page with this title already exists");
    assert!(body.get("existing_page_id").is_none(), "{body}");
    assert!(body.get("existing_page_title").is_none(), "{body}");
}
