// SPDX-License-Identifier: Apache-2.0
//! Knowledge directory inspection endpoints.

use crate::error::ServerError;
use crate::route_registry::{get, TrackedRouter};
use crate::state::SharedState;
use axum::extract::{Query, State};
use axum::response::Json;
use serde::Deserialize;
use wenlan_types::responses::{KnowledgeCountResponse, KnowledgePathResponse};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/knowledge/path", get(handle_get_knowledge_path))
        .route("/api/knowledge/count", get(handle_get_knowledge_count))
        .route(
            "/api/knowledge/recent-relations",
            get(handle_list_recent_relations),
        )
}

/// GET /api/knowledge/path
pub async fn handle_get_knowledge_path() -> Result<Json<KnowledgePathResponse>, ServerError> {
    let cfg = wenlan_core::config::load_config();
    let path = cfg.knowledge_path_or_default();
    Ok(Json(KnowledgePathResponse {
        path: path.to_string_lossy().to_string(),
    }))
}

/// GET /api/knowledge/count
pub async fn handle_get_knowledge_count() -> Result<Json<KnowledgeCountResponse>, ServerError> {
    let cfg = wenlan_core::config::load_config();
    let path = cfg.knowledge_path_or_default();
    if !path.exists() {
        return Ok(Json(KnowledgeCountResponse { count: 0 }));
    }
    let count = std::fs::read_dir(&path)
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("md"))
                .unwrap_or(false)
        })
        .count();
    Ok(Json(KnowledgeCountResponse {
        count: count as u64,
    }))
}

/// Query params for `GET /api/knowledge/recent-relations`.
#[derive(Debug, Deserialize)]
pub struct RecentRelationsQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub since_ms: Option<i64>,
}

/// GET /api/knowledge/recent-relations?limit=&since_ms=
pub async fn handle_list_recent_relations(
    State(state): State<SharedState>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Query(params): Query<RecentRelationsQuery>,
) -> Result<Json<Vec<wenlan_types::RecentRelation>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let limit = params.limit.unwrap_or(10).min(50);
    let relations = db
        .list_recent_relations_scoped(limit, params.since_ms, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(relations))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn get_knowledge_count_returns_ok() {
        let result = handle_get_knowledge_count().await;
        assert!(result.is_ok());
    }

    #[test]
    fn count_md_files_in_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.md"), "x").unwrap();
        std::fs::write(dir.path().join("b.md"), "y").unwrap();
        std::fs::write(dir.path().join("c.txt"), "z").unwrap();

        let count = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("md"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(count, 2);
    }
}
