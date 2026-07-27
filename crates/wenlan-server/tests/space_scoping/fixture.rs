// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Method, Request, Response};
use serde_json::Value;
use tower::ServiceExt;
use wenlan_core::db::MemoryDB;
use wenlan_core::sources::RawDocument;
use wenlan_types::requests::CreateConceptRequest;

pub struct ScopeFixture {
    pub router: super::super::common::AppRouter,
    pub db: std::sync::Arc<MemoryDB>,
    pub _tmp: tempfile::TempDir,
}

impl ScopeFixture {
    pub async fn new() -> Self {
        let (router, tmp, db) = super::super::common::test_app_no_gate().await;
        db.create_space("work", None, false).await.unwrap();
        db.create_space("personal", None, false).await.unwrap();
        Self {
            router,
            db,
            _tmp: tmp,
        }
    }

    pub async fn seed_wave_1_memory(
        &self,
        source_id: &str,
        space: Option<&str>,
        last_modified: i64,
    ) {
        self.db
            .upsert_documents(vec![RawDocument {
                source: "memory".to_string(),
                source_id: source_id.to_string(),
                title: format!("title-{source_id}"),
                content: format!("scope canary {source_id}"),
                last_modified,
                memory_type: Some("fact".to_string()),
                space: space.map(str::to_string),
                confirmed: Some(false),
                stability: Some("new".to_string()),
                pending_revision: false,
                ..Default::default()
            }])
            .await
            .unwrap();
        self.db.pin_memory(source_id).await.unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn seed_record(
        &self,
        source: &str,
        source_id: &str,
        space: Option<&str>,
        memory_type: &str,
        last_modified: i64,
        supersedes: Option<&str>,
        pending_revision: bool,
    ) {
        self.db
            .upsert_documents(vec![RawDocument {
                source: source.to_string(),
                source_id: source_id.to_string(),
                title: format!("title-{source_id}"),
                content: format!("record canary {source_id}"),
                last_modified,
                memory_type: Some(memory_type.to_string()),
                space: space.map(str::to_string),
                confirmed: Some(true),
                stability: Some("confirmed".to_string()),
                supersedes: supersedes.map(str::to_string),
                pending_revision,
                ..Default::default()
            }])
            .await
            .unwrap();
    }

    pub async fn seed_page(&self, title: &str, workspace: &str) -> String {
        wenlan_core::post_write::create_page(
            &self.db,
            CreateConceptRequest {
                title: title.to_string(),
                content: format!("page canary {title}"),
                summary: None,
                entity_id: None,
                space: (Some(workspace.to_string())).into(),
                source_memory_ids: Vec::new(),
                creation_kind: Some("authored".to_string()),
                workspace: Some(workspace.to_string()),
            },
            "test",
            None,
        )
        .await
        .unwrap()
        .id
    }

    pub async fn send(
        &self,
        method: Method,
        uri: &str,
        body: Option<Value>,
        header_space: Option<&str>,
    ) -> Response<Body> {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(space) = header_space {
            builder = builder.header("x-wenlan-space", space);
        }
        let body = match body {
            Some(value) => {
                builder = builder.header("content-type", "application/json");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        self.router
            .clone()
            .oneshot(builder.body(body).unwrap())
            .await
            .unwrap()
    }
}
