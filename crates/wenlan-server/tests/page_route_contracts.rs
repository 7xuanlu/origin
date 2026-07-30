// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_server::{router::build_router, state::ServerState};
use wenlan_types::requests::{
    CreateConceptRequest, ExportPageRequest, ExportPagesRequest, RefreshPageRequest,
    SearchPagesRequest, UpdatePageRequest,
};
use wenlan_types::responses::{
    CreatePageResponse, ExportPageResponse, ExportStats, ListPageRevisionsResponse,
    OrphanLinksResponse, PageLinksResponse, PageWriteResponse, SearchPagesResponse,
    SuccessResponse,
};
use wenlan_types::{Page, PageSourceWithMemory, RawDocument, WriteSpaceTarget};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

#[derive(Debug, Deserialize)]
struct PageEnvelope {
    page: Page,
}

#[derive(Debug, Deserialize)]
struct StatusEnvelope {
    status: String,
}

#[derive(Clone)]
struct RouteCase {
    method: Method,
    uri: String,
    body: Option<Vec<u8>>,
    marked: bool,
}

fn encoded<T: Serialize>(value: &T) -> Option<Vec<u8>> {
    Some(serde_json::to_vec(value).unwrap())
}

fn page_routes(id: &str) -> Vec<RouteCase> {
    vec![
        RouteCase {
            method: Method::GET,
            uri: "/api/pages".to_string(),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::GET,
            uri: format!("/api/pages/{id}"),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::GET,
            uri: format!("/api/pages/{id}/sources"),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::POST,
            uri: format!("/api/pages/{id}/archive"),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::DELETE,
            uri: format!("/api/pages/{id}"),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/pages/search".to_string(),
            body: encoded(&SearchPagesRequest {
                query: "route probe".to_string(),
                limit: Some(5),
                page_type: None,
                space: None,
            }),
            marked: false,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/pages".to_string(),
            body: encoded(&CreateConceptRequest {
                title: "Route probe".to_string(),
                content: "Route probe content".to_string(),
                summary: None,
                entity_id: None,
                space: WriteSpaceTarget::Inherit,
                source_memory_ids: Vec::new(),
                creation_kind: Some("authored".to_string()),
                workspace: None,
            }),
            marked: false,
        },
        RouteCase {
            method: Method::POST,
            uri: "/api/pages/export".to_string(),
            body: encoded(&ExportPagesRequest {
                vault_path: Some("/tmp/wenlan-page-route-contract-no-db".to_string()),
            }),
            marked: false,
        },
        RouteCase {
            method: Method::POST,
            uri: format!("/api/pages/{id}/export"),
            body: encoded(&ExportPageRequest {
                vault_path: "/tmp/wenlan-page-route-contract-no-db".to_string(),
            }),
            marked: false,
        },
        RouteCase {
            method: Method::GET,
            uri: format!("/api/pages/{id}/links"),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::GET,
            uri: "/api/pages/orphan-links".to_string(),
            body: None,
            marked: false,
        },
        RouteCase {
            method: Method::POST,
            uri: format!("/api/memory/{id}/update-page"),
            body: encoded(&UpdatePageRequest {
                content: "Manual route probe".to_string(),
                source_memory_ids: Vec::new(),
                expected_version: None,
                caller_id: None,
                operation_id: None,
            }),
            marked: false,
        },
        RouteCase {
            method: Method::PUT,
            uri: format!("/api/pages/{id}"),
            body: encoded(&RefreshPageRequest {
                content: "Refreshed route probe".to_string(),
                source_memory_ids: vec!["route-probe-source".to_string()],
                summary: None,
            }),
            marked: false,
        },
        RouteCase {
            method: Method::GET,
            uri: format!("/api/pages/{id}/revisions"),
            body: None,
            marked: false,
        },
    ]
}

async fn request_bytes(router: &common::AppRouter, case: &RouteCase) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder()
        .method(case.method.clone())
        .uri(&case.uri);
    if case.body.is_some() {
        request = request.header("content-type", "application/json");
    }
    if case.marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "page-route-contract");
    }
    let response = router
        .clone()
        .oneshot(
            request
                .body(
                    case.body
                        .clone()
                        .map(Body::from)
                        .unwrap_or_else(Body::empty),
                )
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

async fn request_typed<T>(router: &common::AppRouter, case: RouteCase) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, &case).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{} {} returned a non-contract response ({error}): {}",
            case.method,
            case.uri,
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

fn get(uri: impl Into<String>, marked: bool) -> RouteCase {
    RouteCase {
        method: Method::GET,
        uri: uri.into(),
        body: None,
        marked,
    }
}

fn mutation<T: Serialize>(method: Method, uri: impl Into<String>, body: Option<&T>) -> RouteCase {
    RouteCase {
        method,
        uri: uri.into(),
        body: body.and_then(encoded),
        marked: false,
    }
}

struct WritableKnowledgeConfig {
    previous: Option<std::ffi::OsString>,
    tmp: tempfile::TempDir,
}

impl WritableKnowledgeConfig {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let pages = tmp.path().join("pages");
        std::fs::create_dir_all(&pages).unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::json!({ "knowledge_path": pages.to_string_lossy() }).to_string(),
        )
        .unwrap();
        let previous = std::env::var_os("WENLAN_DATA_DIR");
        std::env::set_var("WENLAN_DATA_DIR", tmp.path());
        Self { previous, tmp }
    }

    fn pages_dir(&self) -> std::path::PathBuf {
        self.tmp.path().join("pages")
    }
}

impl Drop for WritableKnowledgeConfig {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
            None => std::env::remove_var("WENLAN_DATA_DIR"),
        }
    }
}

#[tokio::test]
async fn page_routes_preserve_typed_contracts() {
    let config = WritableKnowledgeConfig::new();
    assert_eq!(
        wenlan_core::config::load_config().knowledge_path_or_default(),
        config.pages_dir(),
        "the contract must never resolve the default live Page vault"
    );

    let (router, tmp, db) = common::test_app_no_gate_with_page_root().await;
    let source_content = "A grounded source for Page route contracts with [[Missing Route Topic]].";
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: "page-route-source".to_string(),
        title: "Page route source".to_string(),
        content: source_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("page-route-contract".to_string()),
        confidence: Some(0.9),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();

    let read_id = common::create_page_fixture(
        &db,
        "Page Route Read",
        source_content,
        None,
        &["page-route-source"],
        "authored",
    )
    .await;
    let manual_id = common::create_page_fixture(
        &db,
        "Page Route Manual",
        "Manual edit before",
        None,
        &[],
        "authored",
    )
    .await;
    let refresh_id = common::create_page_fixture(
        &db,
        "Page Route Refresh",
        source_content,
        None,
        &["page-route-source"],
        "distilled",
    )
    .await;
    let archive_id = common::create_page_fixture(
        &db,
        "Page Route Archive",
        "Archive content",
        None,
        &[],
        "authored",
    )
    .await;
    let delete_id = common::create_page_fixture(
        &db,
        "Page Route Delete",
        "Delete content",
        None,
        &[],
        "authored",
    )
    .await;

    let create_request = CreateConceptRequest {
        title: "Page Route Created".to_string(),
        content: "Created through the HTTP Page route".to_string(),
        summary: Some("Created Page route".to_string()),
        entity_id: None,
        space: WriteSpaceTarget::Inherit,
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };
    let (status, created): (StatusCode, CreatePageResponse) = request_typed(
        &router,
        mutation(Method::POST, "/api/pages", Some(&create_request)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!created.id.is_empty());

    let (status, listed): (StatusCode, SearchPagesResponse) =
        request_typed(&router, get("/api/pages?limit=50", true)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(listed.pages.iter().any(|page| page.id == read_id));

    let (status, page): (StatusCode, PageEnvelope) =
        request_typed(&router, get(format!("/api/pages/{read_id}"), true)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page.page.title, "Page Route Read");

    let (status, sources): (StatusCode, Vec<PageSourceWithMemory>) =
        request_typed(&router, get(format!("/api/pages/{read_id}/sources"), true)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].source.memory_source_id, "page-route-source");

    let search_request = SearchPagesRequest {
        query: "Page Route Read".to_string(),
        limit: Some(10),
        page_type: None,
        space: None,
    };
    let (status, searched): (StatusCode, SearchPagesResponse) = request_typed(
        &router,
        RouteCase {
            marked: true,
            ..mutation(Method::POST, "/api/pages/search", Some(&search_request))
        },
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(searched.pages.iter().any(|page| page.id == read_id));

    let (status, links): (StatusCode, PageLinksResponse) =
        request_typed(&router, get(format!("/api/pages/{read_id}/links"), true)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(links.outbound.len(), 1);

    let (status, orphans): (StatusCode, OrphanLinksResponse) =
        request_typed(&router, get("/api/pages/orphan-links", false)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(orphans.min_count, 2);

    let (status, revisions): (StatusCode, ListPageRevisionsResponse) = request_typed(
        &router,
        get(format!("/api/pages/{read_id}/revisions"), true),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(revisions.page_id, read_id);

    let bulk_export_dir = tmp.path().join("bulk-export");
    let bulk_export = ExportPagesRequest {
        vault_path: Some(bulk_export_dir.to_string_lossy().into_owned()),
    };
    let (status, exported): (StatusCode, ExportStats) = request_typed(
        &router,
        mutation(Method::POST, "/api/pages/export", Some(&bulk_export)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(exported.exported >= 1);

    let single_export_dir = tmp.path().join("single-export");
    let single_export = ExportPageRequest {
        vault_path: single_export_dir.to_string_lossy().into_owned(),
    };
    let (status, exported): (StatusCode, ExportPageResponse) = request_typed(
        &router,
        mutation(
            Method::POST,
            format!("/api/pages/{read_id}/export"),
            Some(&single_export),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(std::path::Path::new(&exported.path).starts_with(&single_export_dir));

    let manual = UpdatePageRequest {
        content: "Manual edit after".to_string(),
        source_memory_ids: Vec::new(),
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let (status, updated): (StatusCode, SuccessResponse) = request_typed(
        &router,
        mutation(
            Method::POST,
            format!("/api/memory/{manual_id}/update-page"),
            Some(&manual),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(updated.ok);

    let refresh = RefreshPageRequest {
        content: "Refresh content after, grounded in its source memory.".to_string(),
        source_memory_ids: vec!["page-route-source".to_string()],
        summary: Some("Refreshed Page route".to_string()),
    };
    let (status, refreshed): (StatusCode, PageWriteResponse) = request_typed(
        &router,
        mutation(
            Method::PUT,
            format!("/api/pages/{refresh_id}"),
            Some(&refresh),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(refreshed.ok);

    let (status, archived): (StatusCode, StatusEnvelope) = request_typed(
        &router,
        mutation::<()>(
            Method::POST,
            format!("/api/pages/{archive_id}/archive"),
            None,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(archived.status, "archived");

    let (status, deleted): (StatusCode, StatusEnvelope) = request_typed(
        &router,
        mutation::<()>(Method::DELETE, format!("/api/pages/{delete_id}"), None),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted.status, "deleted");

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for case in page_routes("missing-page") {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&no_db, case.clone()).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{} {}: {}",
            case.method,
            case.uri,
            error.error
        );
        assert_eq!(error.error, "Database not initialized");
    }
}
