// SPDX-License-Identifier: Apache-2.0

use crate::{
    brief_files::project_brief_receipt, error::ServerError, space_header::SpaceHeader,
    state::SharedState,
};
use axum::{extract::State, response::Json};
use std::{future::Future, sync::Arc};
use wenlan_core::{db::MemoryDB, read_scope::ReadScope};
use wenlan_types::{
    BriefReadRequest, BriefReadResponse, BriefReadState, BriefRelatedContext, BriefUpdateReceipt,
    BriefUpdateRequest, SearchResult,
};

const RELATED_CONTEXT_LIMIT: usize = 20;

async fn resolve_space(
    db: &MemoryDB,
    explicit: Option<&str>,
    header: Option<&str>,
) -> Result<Option<String>, ServerError> {
    let selected = explicit
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| header.map(str::trim).filter(|value| !value.is_empty()));

    if let Some(name) = selected {
        return db
            .get_space(name)
            .await?
            .map(|space| Some(space.name))
            .ok_or_else(|| ServerError::ValidationError(format!("unknown Space: {name}")));
    }

    Ok(db.get_default_space().await?.map(|space| space.name))
}

async fn related_context_if_requested<F, Fut>(
    topic: Option<&str>,
    load: F,
) -> Result<Option<BriefRelatedContext>, ServerError>
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = Result<Vec<SearchResult>, ServerError>>,
{
    let Some(query) = topic
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
    else {
        return Ok(None);
    };
    let results = load(query.clone()).await?;
    Ok(Some(BriefRelatedContext { query, results }))
}

/// POST /api/brief — read the complete Space Brief, optionally composed with
/// separately-labelled recall results for the same Space.
pub async fn handle_read_brief(
    State(state): State<SharedState>,
    SpaceHeader(header_space): SpaceHeader,
    Json(request): Json<BriefReadRequest>,
) -> Result<Json<BriefReadResponse>, ServerError> {
    let db = {
        let state = state.read().await;
        state.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let Some(space) = resolve_space(&db, request.space.as_deref(), header_space.as_deref()).await?
    else {
        return Ok(Json(BriefReadResponse {
            state: BriefReadState::SpaceNotResolved,
            space: None,
            brief: None,
            related_context: None,
        }));
    };
    let Some(brief) = db.get_brief_by_space_name(&space).await? else {
        return Ok(Json(BriefReadResponse {
            state: BriefReadState::BriefNotCreated,
            space: Some(space),
            brief: None,
            related_context: None,
        }));
    };

    let legacy_context_limit = request
        .legacy_context_limit
        .unwrap_or(RELATED_CONTEXT_LIMIT);
    let recall_db = Arc::clone(&db);
    let recall_space = space.clone();
    let related_context = related_context_if_requested(request.topic.as_deref(), move |query| {
        let scope = ReadScope::Space(recall_space);
        async move {
            recall_db
                .search_memory(
                    &query,
                    legacy_context_limit,
                    None,
                    &scope,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .map_err(|error| ServerError::SearchFailed(error.to_string()))
        }
    })
    .await?;

    Ok(Json(BriefReadResponse {
        state: BriefReadState::Ready,
        space: Some(space),
        brief: Some(brief),
        related_context,
    }))
}

/// PATCH /api/brief — apply item-level handoff deltas, then best-effort
/// project the committed Brief as a human-readable Markdown receipt.
pub async fn handle_update_brief(
    State(state): State<SharedState>,
    Json(request): Json<BriefUpdateRequest>,
) -> Result<Json<BriefUpdateReceipt>, ServerError> {
    let (db, status_root, projection_lock) = {
        let state = state.read().await;
        (
            state.db.clone().ok_or(ServerError::DbNotInitialized)?,
            state.brief_status_root.clone(),
            Arc::clone(&state.brief_projection_lock),
        )
    };
    let mut receipt = db.apply_brief_update(&request).await?;
    let _projection_guard = projection_lock.lock().await;

    match (
        status_root,
        db.get_brief_by_space_name(&receipt.space).await?,
    ) {
        (Some(root), Some(brief)) => {
            match project_brief_receipt(&root, &brief, chrono::Utc::now().timestamp()) {
                Ok(path) => receipt.projection_path = Some(path.display().to_string()),
                Err(error) => receipt
                    .warnings
                    .push(format!("Brief receipt projection failed: {error}")),
            }
        }
        (None, _) => receipt
            .warnings
            .push("Brief receipt projection is not configured".to_string()),
        (Some(_), None) => receipt.warnings.push(
            "Brief receipt projection skipped because the committed Brief was not found".into(),
        ),
    }

    Ok(Json(receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn no_topic_never_invokes_related_context_loader() {
        let calls = AtomicUsize::new(0);
        let result = related_context_if_requested(None, |_| async {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        })
        .await
        .unwrap();

        assert!(result.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn topic_invokes_related_context_loader_once_without_rewriting_it() {
        let calls = AtomicUsize::new(0);
        let result = related_context_if_requested(Some("  release gate  "), |query| {
            calls.fetch_add(1, Ordering::SeqCst);
            async move {
                assert_eq!(query, "release gate");
                Ok(Vec::new())
            }
        })
        .await
        .unwrap()
        .unwrap();

        assert_eq!(result.query, "release gate");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
