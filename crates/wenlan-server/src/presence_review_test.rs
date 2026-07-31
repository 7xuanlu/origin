// SPDX-License-Identifier: Apache-2.0
//! End-to-end teeth for `POST /api/pages/{id}/review`.
//!
//! The protocol's own rows are proven beside the mutation, in
//! `wenlan-core/src/db/presence_review_test.rs`. What only this file can prove
//! is that the route exists, that it resolves the install secret from the same
//! directory the app writes it to, and that a refusal reaches the client as a
//! coarse code and a status rather than as a 500 or a leak.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

use crate::state::ServerState;

const SECRET: &[u8; 32] = b"01234567890123456789012345678901";
const PAGE: &str = "page_00000000-0000-4000-8000-0000000000b1";
const BODY: &str = "the prose a human read";

/// A daemon with a database, a page, and the app's secret sitting where the app
/// would have written it.
async fn wired() -> (
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
    db.create_page_draft_with_id(PAGE, "A Page", BODY, None, None)
        .await
        .expect("page");
    plant_secret(tmp.path());

    let state = Arc::new(RwLock::new(ServerState {
        db: Some(db.clone()),
        presence_root: Some(tmp.path().to_path_buf()),
        ..Default::default()
    }));
    (state, db, tmp)
}

/// Plants the secret with the mode the app writes. The default mode depends on
/// the umask of whoever runs the tests, and the daemon refuses a secret other
/// users can read, so an install fixture has to be explicit about it.
fn plant_secret(root: &std::path::Path) {
    let path = wenlan_core::presence::secret_path(root);
    std::fs::write(&path, SECRET).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
}

/// Mints the way the app does. Duplicated from the core test on purpose — a
/// shared helper would let one encoder drift and take both sides with it.
fn mint(operation_id: &str, base_digest: &str, nonce: &[u8], secret: &[u8]) -> serde_json::Value {
    fn push(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        out.extend_from_slice(bytes);
    }
    // Against the real clock, because the handler stamps and expires with it.
    // A fixed mint time would put every capability 56 years in the past.
    let minted_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let expires_at = minted_at + 60;
    let mut out = Vec::new();
    push(&mut out, &1_u32.to_be_bytes());
    push(&mut out, b"review_page");
    out.extend_from_slice(&1_u64.to_be_bytes());
    push(&mut out, PAGE.as_bytes());
    push(&mut out, base_digest.as_bytes());
    push(&mut out, b"app");
    push(&mut out, operation_id.as_bytes());
    push(&mut out, nonce);
    push(&mut out, &minted_at.to_be_bytes());
    push(&mut out, &expires_at.to_be_bytes());

    let mut mac = <Hmac<Sha256>>::new_from_slice(secret).unwrap();
    mac.update(&out);
    serde_json::json!({
        "protocol_version": 1,
        "action": "review_page",
        "target_ids": [PAGE],
        "base_digest": base_digest,
        "caller_id": "app",
        "operation_id": operation_id,
        "minted_at": minted_at,
        "expires_at": expires_at,
        "nonce": hex::encode(nonce),
        "mac": hex::encode(mac.finalize().into_bytes()),
    })
}

fn body_digest() -> String {
    wenlan_core::provenance::revision_content_digest(BODY)
}

async fn review(
    state: Arc<RwLock<ServerState>>,
    presence: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = crate::router::build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/pages/{PAGE}/review"))
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({ "presence": presence }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

#[tokio::test]
async fn the_review_route_exists() {
    let state = Arc::new(RwLock::new(ServerState::default()));
    let response = crate::router::build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/pages/page-1/review")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(
        response.status(),
        StatusCode::NOT_FOUND,
        "a client with a minted capability has nowhere to submit it"
    );
}

/// The whole point, over HTTP: a gesture the app minted marks the page, and the
/// caller gets back a receipt it can show.
#[tokio::test]
async fn a_gesture_from_the_app_marks_the_page() {
    let (state, db, _tmp) = wired().await;
    let (status, body) = review(state, mint("op-1", &body_digest(), &[0xb1; 16], SECRET)).await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["human_reviewed"], true);
    assert_eq!(body["page_id"], PAGE);
    assert_eq!(body["reviewed_page_digest"], body_digest());

    let states = db.page_truth_states(&[PAGE.to_string()]).await.unwrap();
    assert!(
        states.get(PAGE).is_some_and(|t| t.human_reviewed),
        "the durable row carries the human axis"
    );
}

/// §5, at the route. A daemon whose data directory has no secret in it cannot
/// check presence, and must refuse rather than proceed.
#[tokio::test]
async fn without_the_install_secret_every_review_is_refused() {
    let (state, db, tmp) = wired().await;
    std::fs::remove_file(wenlan_core::presence::secret_path(tmp.path())).unwrap();

    let (status, body) = review(state, mint("op-1", &body_digest(), &[0xb1; 16], SECRET)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "presence_unavailable");

    // No row at all is the honest reading here: a page distilled after
    // migration 99's backfill has none until something writes one, and a
    // refused review writes nothing.
    let states = db.page_truth_states(&[PAGE.to_string()]).await.unwrap();
    assert!(
        !states.get(PAGE).is_some_and(|t| t.human_reviewed),
        "an unavailable secret must not degrade to trusting the caller"
    );
}

/// T2 over the wire: an HTTP client without the secret gets a coarse refusal.
#[tokio::test]
async fn a_client_that_cannot_read_the_secret_is_refused() {
    let (state, _db, _tmp) = wired().await;
    let (status, body) = review(
        state,
        mint(
            "op-1",
            &body_digest(),
            &[0xb1; 16],
            b"not the install secret, thirty-two",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"], "presence_invalid");
}

/// T6 over the wire. The refusal says only that presence did not check out —
/// not which digest was expected, which would hand a caller the exact value it
/// needs to re-mint against content it never saw.
#[tokio::test]
async fn a_stale_gesture_is_refused_without_naming_the_live_digest() {
    let (state, _db, _tmp) = wired().await;
    let (status, body) = review(
        state,
        mint("op-1", "a digest of some older text", &[0xb1; 16], SECRET),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"], "presence_invalid");
    assert!(
        !body.to_string().contains(&body_digest()),
        "the live digest must not appear in a refusal: {body}"
    );
}

/// T7 over the wire: the retry gets the same 200 and the same receipt.
#[tokio::test]
async fn a_retry_returns_the_stored_receipt_and_the_same_status() {
    let (state, _db, _tmp) = wired().await;
    let cap = mint("op-1", &body_digest(), &[0xb1; 16], SECRET);

    let (first_status, first) = review(state.clone(), cap.clone()).await;
    let (retry_status, retry) = review(state, cap).await;

    assert_eq!(first_status, StatusCode::OK);
    assert_eq!(retry_status, StatusCode::OK);
    assert_eq!(
        first, retry,
        "a retry must not look different to the client"
    );
}

/// T8 over the wire.
#[tokio::test]
async fn a_reused_operation_id_with_new_content_conflicts() {
    let (state, _db, _tmp) = wired().await;
    review(
        state.clone(),
        mint("op-1", &body_digest(), &[0xb1; 16], SECRET),
    )
    .await;

    let (status, body) = review(
        state,
        mint("op-1", "some other content digest", &[0xb2; 16], SECRET),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"], "presence_conflict");
}

/// T3 over the wire.
#[tokio::test]
async fn a_second_use_of_one_nonce_is_refused() {
    let (state, _db, _tmp) = wired().await;
    review(
        state.clone(),
        mint("op-1", &body_digest(), &[0xb1; 16], SECRET),
    )
    .await;

    let (status, body) = review(state, mint("op-2", &body_digest(), &[0xb1; 16], SECRET)).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"], "presence_replayed");
}

/// The §7 sentinel test.
///
/// A capability carrying a distinctive string in its HMAC goes in; the string
/// must appear in no response body and in no formatting of the request. It is
/// deliberately placed in the `mac` field, because that is the one component
/// whose whole job is to be secret — an error handler that echoed "the value we
/// could not parse" would put it straight into the response.
#[tokio::test]
async fn a_capability_containing_a_sentinel_never_comes_back_out() {
    const SENTINEL: &str = "zzsentinelzz";
    let (state, _db, _tmp) = wired().await;

    let mut cap = mint("op-1", &body_digest(), &[0xb1; 16], SECRET);
    cap["mac"] = serde_json::Value::String(format!("{SENTINEL}deadbeef"));
    cap["nonce"] = serde_json::Value::String(format!("{SENTINEL}00"));

    let (status, body) = review(state, cap.clone()).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(
        !body.to_string().contains(SENTINEL),
        "the response echoed capability material: {body}"
    );

    // And the same string must not survive being formatted, which is how it
    // would reach a log file.
    let parsed: wenlan_types::requests::PresenceCapability = serde_json::from_value(cap).unwrap();
    let printed = format!("{parsed:?}");
    assert!(
        !printed.contains(SENTINEL),
        "Debug printed capability material: {printed}"
    );
}

/// The classification decision, pinned rather than inherited. The receipt
/// carries a version, two digests, and identity — no title, no excerpt, no
/// prose — so the route stays `page_bearing: no` and refuses a marker.
#[test]
fn the_review_route_carries_no_page_prose() {
    use wenlan_core::truth_manifest::{
        self, Builder, MarkerShape, PageBearing, ReaderMethod, TruthClass,
    };
    let row =
        truth_manifest::http_reader(Builder::Main, ReaderMethod::Post, "/api/pages/{id}/review")
            .expect("the review route must be classified");
    assert_eq!(row.page_bearing, PageBearing::No);
    assert_eq!(row.class, TruthClass::NotApplicable);
    assert_eq!(row.marker_shape, MarkerShape::None);
}

/// `MarkerShape::None` refuses rather than ignores, and that holds here too: a
/// mutation with no prose to gate is not a place a reader marker may do
/// anything.
#[tokio::test]
async fn a_marker_is_refused_at_the_review_route() {
    let (state, _db, _tmp) = wired().await;
    let response = crate::router::build_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/pages/{PAGE}/review"))
                .header("content-type", "application/json")
                .header(wenlan_core::truth_contract::INTENT_HEADER, "explicit")
                .header(wenlan_core::truth_contract::CONTRACT_HEADER, "1")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}
