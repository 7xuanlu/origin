// SPDX-License-Identifier: Apache-2.0
//
// #708: the page surface must refuse to archive or delete an entity's page,
// and the refusal must be usable -- it names the entity and carries its id, so
// the app can offer to open it in Entities instead of showing a dead end.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use crate::state::ServerState;

/// A server state holding one entity, plus the entity id and the id of the
/// page that stands for it.
async fn state_with_entity() -> (
    Arc<tokio::sync::RwLock<ServerState>>,
    tempfile::TempDir,
    String,
    String,
) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
        Arc::new(wenlan_core::events::NoopEmitter);
    let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
        .await
        .expect("MemoryDB::new");
    let entity_id = db
        .store_entity("Guarded Org", "organization", None, None, Some(0.9))
        .await
        .expect("store_entity");
    // Established, so its page is one a person can actually reach from the
    // Wiki -- which is where they would try to archive it from. The browse
    // read is also how the id gets back out: an entity's page carries its
    // entity id there (#708).
    db.confirm_entity(&entity_id, true)
        .await
        .expect("confirm_entity");
    let page_id = db
        .list_pages_browse("active", 50, 0)
        .await
        .expect("list_pages_browse")
        .into_iter()
        .find(|page| page.entity_id.as_deref() == Some(entity_id.as_str()))
        .expect("the entity's page must be visible in browse")
        .id;
    let mut server_state = ServerState::new();
    server_state.db = Some(Arc::new(db));
    (
        Arc::new(tokio::sync::RwLock::new(server_state)),
        tmp,
        entity_id,
        page_id,
    )
}

async fn send(
    state: Arc<tokio::sync::RwLock<ServerState>>,
    method: &str,
    uri: String,
) -> (StatusCode, serde_json::Value) {
    let app = crate::router::build_router(state);
    let response = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body)
}

#[tokio::test]
async fn archiving_an_entity_page_names_the_entity_and_carries_its_id() {
    let (state, _tmp, entity_id, page_id) = state_with_entity().await;
    let (status, body) = send(state, "POST", format!("/api/pages/{page_id}/archive")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body["error"],
        "This page belongs to the entity 'Guarded Org'. Archive or delete it from Entities",
        "the refusal must name the entity and where to act: {body}"
    );
    assert_eq!(body["entity_id"], serde_json::Value::String(entity_id));
    assert_eq!(body["entity_name"], "Guarded Org");
}

#[tokio::test]
async fn deleting_an_entity_page_is_refused_the_same_way() {
    let (state, _tmp, entity_id, page_id) = state_with_entity().await;
    let (status, body) = send(state, "DELETE", format!("/api/pages/{page_id}")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body["error"],
        "This page belongs to the entity 'Guarded Org'. Archive or delete it from Entities"
    );
    assert_eq!(body["entity_id"], serde_json::Value::String(entity_id));
}

#[tokio::test]
async fn a_page_that_is_not_an_entitys_is_untouched_by_the_guard() {
    let (state, _tmp, _entity_id, _page_id) = state_with_entity().await;
    let (status, body) = send(state, "POST", "/api/pages/no-such-page/archive".to_string()).await;
    assert_ne!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "the guard must not fire for a page that is not an entity's: {body}"
    );
}
