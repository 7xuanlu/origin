// SPDX-License-Identifier: Apache-2.0
//! The wire half of the exposure contract, driven through the real router.
//!
//! The policy itself is unit-tested in `wenlan_core::truth_contract`. What only
//! shows up here is whether the guard is actually *wired*: that a marker aimed
//! at a `none`-shaped route comes back 403 rather than being quietly ignored,
//! and that a marked call leaves a row behind. Both are properties of the
//! middleware being installed, and neither is observable from the core crate.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};

use crate::state::ServerState;

/// A state with a real database, so the audit write has somewhere to land.
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

struct Call {
    method: &'static str,
    uri: &'static str,
    marker: bool,
    contract: bool,
}

impl Call {
    const fn new(method: &'static str, uri: &'static str) -> Self {
        Self {
            method,
            uri,
            marker: false,
            contract: false,
        }
    }
    const fn marked(mut self) -> Self {
        self.marker = true;
        self
    }
    const fn declaring(mut self) -> Self {
        self.contract = true;
        self
    }

    async fn send(&self, state: Arc<RwLock<ServerState>>) -> StatusCode {
        let mut builder = Request::builder()
            .method(self.method)
            .uri(self.uri)
            .header("Content-Type", "application/json")
            .header("x-agent-name", "test-agent");
        if self.marker {
            builder = builder.header(INTENT_HEADER, "explicit");
        }
        if self.contract {
            builder = builder.header(CONTRACT_HEADER, "1");
        }
        crate::router::build_router(state)
            .oneshot(builder.body(Body::from("{}")).unwrap())
            .await
            .unwrap()
            .status()
    }
}

async fn status_of(call: Call) -> StatusCode {
    let state = Arc::new(RwLock::new(ServerState::default()));
    call.send(state).await
}

/// The routes D3 excludes unconditionally: automatic context, search, and both
/// exports. An agent wired to transmit the marker gets nothing from them, and
/// gets told so, whoever it claims to be.
#[tokio::test]
async fn a_marker_is_refused_at_context_search_and_both_exports() {
    for (method, uri) in [
        ("POST", "/api/context"),
        ("POST", "/api/search"),
        ("POST", "/api/pages/export"),
        ("POST", "/api/pages/pg1/export"),
    ] {
        let status = status_of(Call::new(method, uri).marked().declaring()).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "{method} {uri} accepted a reader-intent marker"
        );
    }
}

/// Refusal is a fact about the route, not the caller. A client that never
/// declared the contract is refused on exactly the same routes -- otherwise
/// "don't declare the contract" would be a way around the hard gate.
#[tokio::test]
async fn refusal_does_not_require_the_client_to_have_declared_anything() {
    assert_eq!(
        status_of(Call::new("POST", "/api/context").marked()).await,
        StatusCode::FORBIDDEN
    );
}

/// The other half of the same tooth: the guard must refuse *markers*, not
/// requests. Every ordinary call to these routes still goes through.
#[tokio::test]
async fn an_unmarked_call_at_a_none_shaped_route_is_not_refused() {
    for (method, uri) in [
        ("POST", "/api/context"),
        ("POST", "/api/search"),
        ("POST", "/api/pages/export"),
    ] {
        let status = status_of(Call::new(method, uri)).await;
        assert_ne!(
            status,
            StatusCode::FORBIDDEN,
            "{method} {uri} refused a call that sent no marker"
        );
    }
}

#[tokio::test]
async fn a_marker_at_a_shaped_route_is_served() {
    for (method, uri) in [
        ("GET", "/api/pages"),
        ("GET", "/api/pages/pg1"),
        ("GET", "/api/pages/pg1/links"),
    ] {
        // A real database, because this asserts a property of the route's
        // *shape*. Without one the grant is unauditable and the fail-closed
        // rule below refuses it, which would make this test pass or fail for a
        // reason that has nothing to do with shape.
        let (state, _db, _tmp) = db_state().await;
        let status = Call::new(method, uri)
            .marked()
            .declaring()
            .send(state)
            .await;
        assert_ne!(
            status,
            StatusCode::FORBIDDEN,
            "{method} {uri} refused a marker it is shaped to accept"
        );
    }
}

/// The two recent-activity feeds advertised `collection` until PR-C. Their wire
/// types cannot carry what that shape promises -- `RecentActivityItem` has a
/// prose `snippet` and no axes, `PageChange` a bare title -- so they were
/// demoted to `none`, and `none` refuses rather than downgrades. A client that
/// sends a marker here is misconfigured, and this is where it finds out.
#[tokio::test]
async fn the_recent_feeds_refuse_a_marker_they_cannot_honour() {
    for (method, uri) in [
        ("GET", "/api/pages/recent"),
        ("GET", "/api/pages/recent-changes"),
    ] {
        let status = status_of(Call::new(method, uri).marked().declaring()).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "{method} {uri} accepted a marker its wire type cannot carry"
        );
    }
}

/// The audit row is the only compensating control for the one attack D3
/// concedes: `Collection` and `NamedPage` compose, so a caller willing to forge
/// the marker can enumerate provisional IDs and fetch them one at a time. A
/// grant that leaves no row is that walk, unobserved -- so it is refused rather
/// than served.
///
/// A server with no database is the reachable form of "the row could not be
/// written": it also cannot be asked what generation it is on, and unknown is
/// not permission. The other form -- a live database at generation >= 1 whose
/// audit INSERT fails -- takes the same branch by construction but cannot be
/// provoked from here without a fault-injection seam this does not have.
#[tokio::test]
async fn a_grant_that_cannot_be_audited_is_refused() {
    for (method, uri) in [("GET", "/api/pages"), ("GET", "/api/pages/pg1")] {
        let status = status_of(Call::new(method, uri).marked().declaring()).await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "{method} {uri} issued an exposure grant it could not record"
        );
    }
}

// ---- the audit row: the only compensating control there is -----------------

#[tokio::test]
async fn a_granted_call_records_the_caller_and_the_page_it_named() {
    let (state, db, _tmp) = db_state().await;
    Call::new("GET", "/api/pages/pg1")
        .marked()
        .declaring()
        .send(state)
        .await;

    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows.len(), 1, "a marked call left no audit row");
    assert_eq!(rows[0].caller, "test-agent");
    assert_eq!(rows[0].method, "GET");
    assert_eq!(
        rows[0].path, "/api/pages/{id}",
        "the route template, not the URI"
    );
    assert_eq!(rows[0].page_ids, vec!["pg1".to_string()]);
    assert_eq!(rows[0].outcome, "granted_named_page");
}

/// A refusal is the signature of a misconfigured integration, so it is worth
/// exactly as much as a grant in the log.
#[tokio::test]
async fn a_refused_call_is_recorded_too() {
    let (state, db, _tmp) = db_state().await;
    Call::new("POST", "/api/context")
        .marked()
        .declaring()
        .send(state)
        .await;

    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].outcome, "refused");
    assert_eq!(rows[0].path, "/api/context");
}

/// The 10-second sidebar poll from a fully M5-aware app. It declares the
/// contract, it hits a `collection` route, and it must be indistinguishable
/// from any other machine read -- including in the audit log, which would
/// otherwise fill with noise and hide the walk it exists to show.
#[tokio::test]
async fn a_declared_clients_unmarked_poll_records_nothing() {
    let (state, db, _tmp) = db_state().await;
    Call::new("GET", "/api/pages").declaring().send(state).await;
    assert!(
        db.recent_truth_markers(10).await.unwrap().is_empty(),
        "an unmarked call wrote an audit row"
    );
}

/// Marked, shaped, but the client never declared it renders both axes. The
/// marker did nothing, and the log says so rather than claiming a grant.
#[tokio::test]
async fn a_marked_call_from_an_undeclared_client_records_automatic() {
    let (state, db, _tmp) = db_state().await;
    Call::new("GET", "/api/pages/pg1")
        .marked()
        .send(state)
        .await;

    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].outcome, "automatic");
}

/// Caller identity is the whole point of the row. An anonymous call is recorded
/// as `unknown` -- honest-unknown, the same convention the rest of agent
/// attribution uses -- rather than guessed at or dropped.
#[tokio::test]
async fn a_caller_that_does_not_identify_itself_is_recorded_as_unknown() {
    let (state, db, _tmp) = db_state().await;
    crate::router::build_router(state)
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/pages/pg1")
                .header(INTENT_HEADER, "explicit")
                .header(CONTRACT_HEADER, "1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].caller, "unknown");
}

/// PR-B installs the guard; it does not switch anything on. If this ever fails,
/// the adapters have gone live without the PR-C ceremony.
#[tokio::test]
async fn the_cutover_is_off_on_a_freshly_migrated_database() {
    let (_state, db, _tmp) = db_state().await;
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}
