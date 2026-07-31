// SPDX-License-Identifier: Apache-2.0
//! The client-queryable half of the cutover: can a caller ask this daemon
//! whether truth enforcement is live, and what generation it is at?
//!
//! Until `/api/status` carried it, the durable generation was reachable only
//! from the daemon's own `truth-cutover` maintenance subcommand, which no app
//! can call. That left every client-side action whose precondition is "the
//! cutover is live" with nothing to gate on.
//!
//! The classification half is tested here too. Adding a field to a response is
//! exactly the change that can turn a `page_bearing: no` route into a leak
//! without anyone editing the manifest, so both `/api/status` rows and the
//! guard's behaviour at that route are pinned rather than assumed.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER, TRUTH_CONTRACT_VERSION};
use wenlan_core::truth_manifest::{
    self, Builder, MarkerShape, PageBearing, ReaderMethod, TruthClass,
};

use crate::state::ServerState;

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

async fn get_status(state: Arc<RwLock<ServerState>>) -> serde_json::Value {
    let response = crate::router::build_router(state)
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1_048_576)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// The question the app could not ask. A pass-through daemon says so in the
/// same shape an enforcing one does, so a client reads one field rather than
/// inferring enforcement from the absence of evidence.
#[tokio::test]
async fn status_reports_the_durable_cutover_generation() {
    let (state, db, _tmp) = db_state().await;

    let before = get_status(state.clone()).await;
    assert_eq!(
        before["truth"]["cutover_generation"], 0,
        "a daemon at generation 0 must say so, not omit the field"
    );
    assert_eq!(
        before["truth"]["contract_version"], TRUTH_CONTRACT_VERSION,
        "the contract version a client should declare is the one the daemon serves"
    );

    db.set_truth_cutover_generation(7)
        .await
        .expect("advance the generation");

    let after = get_status(state).await;
    assert_eq!(
        after["truth"]["cutover_generation"], 7,
        "status must report the live generation, not a compiled-in constant"
    );
}

/// Without a database the daemon cannot establish that enforcement is off, so
/// it declines to claim anything. `null` is the fail-closed reading a client is
/// contracted to treat as "not live".
#[tokio::test]
async fn status_omits_truth_when_the_generation_cannot_be_read() {
    let state = Arc::new(RwLock::new(ServerState::default()));
    let status = get_status(state).await;
    assert!(
        status["truth"].is_null(),
        "an unreadable generation must not be reported as generation 0: {status}"
    );
}

/// Deserializing a payload from a daemon that predates the field must not fail.
/// The released app is on the other end of this.
#[test]
fn a_status_payload_without_truth_still_deserializes() {
    let legacy = serde_json::json!({
        "is_running": true,
        "files_indexed": 0,
        "files_total": 0,
        "sources_connected": [],
    });
    let parsed: wenlan_types::responses::StatusResponse = serde_json::from_value(legacy).unwrap();
    assert!(parsed.truth.is_none());
}

/// The classification decision, made deliberately rather than inherited.
///
/// `TruthStatus` describes the daemon — a generation counter and a protocol
/// version. No page id, no title, no excerpt can appear in it, so `/api/status`
/// stays `page_bearing: no`, needs no adapter, and refuses a marker. Both
/// builders register the route, and both rows are pinned.
#[test]
fn status_stays_unclassified_for_page_prose_in_both_builders() {
    for builder in [Builder::Main, Builder::Repair] {
        let row = truth_manifest::http_reader(builder, ReaderMethod::Get, "/api/status")
            .unwrap_or_else(|| panic!("/api/status must be classified in {builder:?}"));
        assert_eq!(
            row.page_bearing,
            PageBearing::No,
            "{builder:?} /api/status carries a cutover generation, never page prose"
        );
        assert_eq!(row.class, TruthClass::NotApplicable);
        assert_eq!(
            row.marker_shape,
            MarkerShape::None,
            "{builder:?} /api/status has no prose to gate, so a marker may do nothing here"
        );
    }
}

/// `MarkerShape::None` refuses rather than ignores, and that has to hold at the
/// route a client now polls for cutover state. A daemon that answered a marked
/// status poll would be treating the marker as a blanket header.
#[tokio::test]
async fn a_marker_is_still_refused_at_status() {
    let (state, _db, _tmp) = db_state().await;
    let response = crate::router::build_router(state)
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .header(INTENT_HEADER, "explicit")
                .header(CONTRACT_HEADER, "1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::FORBIDDEN,
        "a reader-intent marker at /api/status must be refused, not ignored"
    );
}
