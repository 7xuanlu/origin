// SPDX-License-Identifier: Apache-2.0
//! Typed, versioned M4 community reads.

use axum::{
    extract::{Query, State},
    Json,
};
use serde::Deserialize;

use crate::{
    error::ServerError,
    route_registry::{get, TrackedRouter},
    space_header::SpaceHeader,
    state::SharedState,
};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/communities", get(handle_list_communities))
        .route(
            "/api/communities/members",
            get(handle_list_community_members),
        )
}

#[derive(Debug, Default, Deserialize)]
pub struct ListCommunitiesQuery {
    pub space: Option<String>,
    pub cursor: Option<String>,
    pub limit: Option<usize>,
}

pub async fn handle_list_communities(
    State(state): State<SharedState>,
    SpaceHeader(header_space): SpaceHeader,
    Query(query): Query<ListCommunitiesQuery>,
) -> Result<Json<wenlan_types::CommunityListResponse>, ServerError> {
    let db = {
        let state = state.read().await;
        state.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(
        &db,
        query.space.as_deref(),
        header_space.as_deref(),
    )
    .await?;
    let response = db
        .list_communities(&scope, query.cursor.as_deref(), query.limit.unwrap_or(100))
        .await
        .map_err(ServerError::from)?;
    Ok(Json(response))
}

#[derive(Debug, Default, Deserialize)]
pub struct ListCommunityMembersQuery {
    pub space: Option<String>,
    pub cursor_space: Option<String>,
    pub cursor_node_id: Option<String>,
    pub limit: Option<usize>,
}

pub async fn handle_list_community_members(
    State(state): State<SharedState>,
    SpaceHeader(header_space): SpaceHeader,
    Query(query): Query<ListCommunityMembersQuery>,
) -> Result<Json<wenlan_types::CommunityMembersResponse>, ServerError> {
    let db = {
        let state = state.read().await;
        state.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(
        &db,
        query.space.as_deref(),
        header_space.as_deref(),
    )
    .await?;
    let cursor = match (query.cursor_space, query.cursor_node_id) {
        (Some(space), Some(node_id)) => {
            Some(wenlan_types::CommunityMemberCursor { space, node_id })
        }
        (None, None) => None,
        _ => {
            return Err(ServerError::from(
                wenlan_core::error::WenlanError::Validation(
                    "community member cursor requires both cursor_space and cursor_node_id"
                        .to_string(),
                ),
            ));
        }
    };
    let response = db
        .list_community_members(&scope, cursor.as_ref(), query.limit.unwrap_or(100))
        .await
        .map_err(ServerError::from)?;
    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    #[tokio::test]
    async fn communities_routes_return_separate_paginated_versioned_wires() {
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(root.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        let state = crate::state::ServerState {
            db: Some(db),
            ..Default::default()
        };
        let router = crate::router::build_router(Arc::new(RwLock::new(state)));
        for uri in [
            "/api/communities?limit=10",
            "/api/communities/members?limit=10",
        ] {
            let response = router
                .clone()
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK, "{uri}");
            let body = axum::body::to_bytes(response.into_body(), 1_048_576)
                .await
                .unwrap();
            let decoded: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                decoded["schema_version"],
                wenlan_types::communities::COMMUNITY_READ_SCHEMA_VERSION,
                "{uri}"
            );
        }
    }

    #[tokio::test]
    async fn communities_routes_reject_unknown_space_and_half_cursors_with_422() {
        // KG review #26: the query -> scope -> ServerError -> 422 translation
        // is what the app keys its unregistered-space fallback on, and the
        // members cursor must be all-or-nothing.
        let root = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(root.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        let state = crate::state::ServerState {
            db: Some(db),
            ..Default::default()
        };
        let router = crate::router::build_router(Arc::new(RwLock::new(state)));
        for (uri, expected) in [
            (
                "/api/communities?space=missing",
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/communities/members?space=missing",
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/communities/members?cursor_node_id=node-1",
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/communities/members?cursor_space=space-a",
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                "/api/communities/members?cursor_space=space-a&cursor_node_id=node-1",
                axum::http::StatusCode::OK,
            ),
        ] {
            let response = router
                .clone()
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "{uri}");
        }
    }
}
