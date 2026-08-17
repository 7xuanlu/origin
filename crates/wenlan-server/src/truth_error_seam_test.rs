// SPDX-License-Identifier: Apache-2.0
//! The error-seam invariant from `crates/wenlan-core/contracts/m5-reader-manifest-inventory.md`
//! (teeth check 14): an error response body must never contain a page's title
//! or prose, at any cutover generation. Error bodies routinely embed the
//! thing that failed ("page 'X' not found", a serde error quoting the
//! payload, an anyhow chain carrying a row's content) -- that makes the error
//! seam a disclosure channel bypassing every adapter. This is ONE invariant
//! (assert the sentinel is absent from every non-2xx body), not a per-route
//! classification, so this file drives many error paths through one shared
//! assertion rather than hand-checking each route's shape.

use axum::body::{to_bytes, Body};
use axum::http::{Request, Response, StatusCode};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

use crate::space_header::HEADER_NAME as SPACE_HEADER_NAME;
use crate::state::ServerState;

/// A state with a real database, so the seeded page has somewhere to land.
/// Mirrors `truth_guard_test::db_state` exactly -- same harness, same shape.
async fn db_state() -> (
    Arc<RwLock<ServerState>>,
    Arc<wenlan_core::db::MemoryDB>,
    tempfile::TempDir,
) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
        Arc::new(wenlan_core::events::NoopEmitter);
    let db = Arc::new(
        wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new"),
    );
    let state = Arc::new(RwLock::new(ServerState {
        db: Some(db.clone()),
        ..Default::default()
    }));
    (state, db, tmp)
}

const SENTINEL_TITLE: &str = "ZZQPROVISIONALTITLESENTINEL";
const SENTINEL_PROSE: &str = "ZZQPROVISIONALPROSESENTINEL";

async fn send(state: &Arc<RwLock<ServerState>>, req: Request<Body>) -> Response<Body> {
    crate::router::build_router(state.clone())
        .oneshot(req)
        .await
        .unwrap()
}

/// Seed a page carrying unmistakable sentinel strings and return its id.
/// Goes through the real `POST /api/pages` route (`handle_create_page`) --
/// the only page-creation seam reachable from wenlan-server; `MemoryDB::insert_page`
/// is `pub(crate)` to wenlan-core and not callable here.
async fn seed_sentinel_page(state: &Arc<RwLock<ServerState>>) -> String {
    // creation_kind "authored": zero cited source memories is a legitimate
    // human-authored page, unlike the default "distilled" kind which requires
    // at least one `source_memory_ids` entry (see `create_page_impl`).
    let body = serde_json::json!({
        "title": SENTINEL_TITLE,
        "content": SENTINEL_PROSE,
        "creation_kind": "authored",
    });
    let response = send(
        state,
        Request::builder()
            .method("POST")
            .uri("/api/pages")
            .header("Content-Type", "application/json")
            .header("x-agent-name", "test-agent")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await;
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "seeding the sentinel page failed: {}",
        String::from_utf8_lossy(&bytes),
    );
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    parsed["id"]
        .as_str()
        .expect("create-page response carried no id")
        .to_string()
}

/// Assert on the raw response bytes, not a parsed struct -- the point is that
/// nothing leaks, including via a field nobody thought to check.
async fn assert_response_has_no_sentinel(response: Response<Body>, label: &str) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_text = String::from_utf8_lossy(&bytes);
    assert!(
        !body_text.contains(SENTINEL_TITLE),
        "{label}: {status} response leaked the sentinel title -- body: {body_text}"
    );
    assert!(
        !body_text.contains(SENTINEL_PROSE),
        "{label}: {status} response leaked the sentinel prose -- body: {body_text}"
    );
}

/// The minimum the plan asks for: a 404 on a page id nothing was stored under.
///
/// The sentinel page is seeded first and left live, so `SENTINEL_TITLE` and
/// `SENTINEL_PROSE` genuinely exist in this process and the assertion has
/// something to find. Without that the test could not fail no matter what the
/// 404 arm echoed. The bogus id is the real one with a suffix, so a prefix or
/// `LIKE`-shaped lookup that resolved the wrong row would surface here too.
#[tokio::test]
async fn a_404_on_a_nonexistent_page_id_carries_no_sentinel() {
    let (state, _db, _tmp) = db_state().await;
    let seeded = seed_sentinel_page(&state).await;
    let missing = format!("{seeded}-zzq-does-not-exist");

    let response = send(
        &state,
        Request::builder()
            .method("GET")
            .uri(format!("/api/pages/{missing}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_response_has_no_sentinel(response, "GET /api/pages/{bogus}").await;
}

/// The payload-echo channel named in this module's header ("a serde error
/// quoting the payload").
///
/// Axum's `Json` rejection fires before `handle_export_page` runs, so no
/// DB-layer regression is reachable through this route -- the request body
/// carries the sentinel instead, which is what gives the assertion a way to go
/// RED. Today `JsonSyntaxError` reports a line and column and nothing else; a
/// rejection formatter that started quoting the offending input would fail
/// here.
#[tokio::test]
async fn a_json_rejection_does_not_echo_the_payload() {
    let (state, _db, _tmp) = db_state().await;
    let id = seed_sentinel_page(&state).await;

    // Valid JSON up to the missing closing brace, so the parse gets far enough
    // to have the sentinel in hand before it fails.
    let malformed = format!(r#"{{"vault_path": "{SENTINEL_TITLE}", "note": "{SENTINEL_PROSE}""#);
    let response = send(
        &state,
        Request::builder()
            .method("POST")
            .uri(format!("/api/pages/{id}/export"))
            .header("Content-Type", "application/json")
            .body(Body::from(malformed))
            .unwrap(),
    )
    .await;
    assert!(
        response.status().is_client_error(),
        "malformed JSON should be rejected, got {}",
        response.status()
    );
    assert_response_has_no_sentinel(response, "POST /api/pages/{id}/export (malformed body)").await;
}

/// An error path that names the page while it still exists: an unregistered
/// `X-Wenlan-Space` header fails scope resolution with `ValidationError`
/// before any handler reads the row, on every read shape that accepts the
/// header. The page is fully live and sentinel-bearing at the moment of the
/// error -- this is the sharpest version of the invariant.
#[tokio::test]
async fn scope_errors_on_a_live_sentinel_page_carry_no_sentinel() {
    let (state, _db, _tmp) = db_state().await;
    let id = seed_sentinel_page(&state).await;

    for (method, uri, body) in [
        ("GET", format!("/api/pages/{id}"), None),
        ("GET", format!("/api/pages/{id}/sources"), None),
        ("GET", format!("/api/pages/{id}/links"), None),
        (
            "POST",
            format!("/api/pages/{id}/export"),
            Some(serde_json::json!({ "vault_path": "/tmp/zzq-export" }).to_string()),
        ),
    ] {
        let mut builder = Request::builder()
            .method(method)
            .uri(&uri)
            .header(SPACE_HEADER_NAME, "zzq-unregistered-space");
        let body = match body {
            Some(b) => {
                builder = builder.header("Content-Type", "application/json");
                Body::from(b)
            }
            None => Body::empty(),
        };
        let response = send(&state, builder.body(body).unwrap()).await;
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "{method} {uri} with an unregistered Space header"
        );
        assert_response_has_no_sentinel(response, &format!("{method} {uri} (unknown space)")).await;
    }
}

/// The id that used to hold the sentinel, after the page is gone. Every route
/// that resolves this id now either 404s or comes back empty -- either way,
/// nothing from the page's last-known content may resurface in the body.
#[tokio::test]
async fn a_deleted_pages_former_id_carries_no_sentinel_anywhere() {
    let (state, _db, _tmp) = db_state().await;
    let id = seed_sentinel_page(&state).await;

    let delete_response = send(
        &state,
        Request::builder()
            .method("DELETE")
            .uri(format!("/api/pages/{id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(delete_response.status(), StatusCode::OK);

    let get_response = send(
        &state,
        Request::builder()
            .method("GET")
            .uri(format!("/api/pages/{id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(get_response.status(), StatusCode::NOT_FOUND);
    assert_response_has_no_sentinel(get_response, "GET /api/pages/{id} (deleted)").await;

    let export_response = send(
        &state,
        Request::builder()
            .method("POST")
            .uri(format!("/api/pages/{id}/export"))
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({ "vault_path": "/tmp/zzq-export" }).to_string(),
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(export_response.status(), StatusCode::NOT_FOUND);
    assert_response_has_no_sentinel(export_response, "POST /api/pages/{id}/export (deleted)").await;

    // These two don't error on a resolved-empty page (they answer with an
    // empty collection rather than 404) -- asserted anyway, on whatever
    // status comes back, since the invariant is "never in any body", not
    // "never in an error body of a route we predicted would error".
    let sources_response = send(
        &state,
        Request::builder()
            .method("GET")
            .uri(format!("/api/pages/{id}/sources"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_response_has_no_sentinel(sources_response, "GET /api/pages/{id}/sources (deleted)")
        .await;

    let links_response = send(
        &state,
        Request::builder()
            .method("GET")
            .uri(format!("/api/pages/{id}/links"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_response_has_no_sentinel(links_response, "GET /api/pages/{id}/links (deleted)").await;
}
