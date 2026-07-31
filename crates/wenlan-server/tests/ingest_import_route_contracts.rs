// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_server::state::ServerState;
use wenlan_types::import::{ImportChatExportRequest, ImportChatExportResponse, PendingImport};
use wenlan_types::requests::ImportMemoriesRequest;
use wenlan_types::responses::{DeleteResponse, ImportMemoriesResponse, IngestResponse};
use wenlan_types::WriteSpaceTarget;

#[derive(Debug, Serialize)]
struct IngestTextRequest {
    source: String,
    source_id: String,
    title: String,
    content: String,
    url: Option<String>,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
struct IngestWebpageRequest {
    url: String,
    title: String,
    content: String,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
struct IngestMemoryRequest {
    source: String,
    source_id: String,
    title: String,
    content: String,
    url: Option<String>,
    tags: Option<Vec<String>>,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_typed<T>(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Option<Vec<u8>>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let mut builder = Request::builder().method(&method).uri(uri);
    let body = match body {
        Some(bytes) => {
            builder = builder.header("content-type", "application/json");
            Body::from(bytes)
        }
        None => Body::empty(),
    };
    let response = router
        .clone()
        .oneshot(builder.body(body).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap();
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{method} {uri} returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

fn json_body<T: Serialize>(body: &T) -> Option<Vec<u8>> {
    Some(serde_json::to_vec(body).unwrap())
}

fn write_empty_claude_export(path: &std::path::Path) {
    let file = std::fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options: zip::write::FileOptions<()> =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("users.json", options).unwrap();
    zip.write_all(b"[]").unwrap();
    zip.start_file("conversations.json", options).unwrap();
    zip.write_all(b"[]").unwrap();
    zip.finish().unwrap();
}

#[tokio::test]
async fn moved_ingest_and_import_routes_preserve_typed_success_contracts() {
    let (router, tmp, db) = common::test_app_no_gate().await;

    let text_request = IngestTextRequest {
        source: "typed-contract".to_string(),
        source_id: "typed-text".to_string(),
        title: "Typed text".to_string(),
        content: "Typed text contract content.".to_string(),
        url: None,
        metadata: None,
    };
    let (status, text): (StatusCode, IngestResponse) = request_typed(
        &router,
        Method::POST,
        "/api/ingest/text",
        json_body(&text_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(text.document_id, "typed-text");
    assert!(text.chunks_created > 0);

    let webpage_request = IngestWebpageRequest {
        url: "https://typed.example/page".to_string(),
        title: "Typed webpage".to_string(),
        content: "Typed webpage contract content.".to_string(),
        metadata: None,
    };
    let (status, webpage): (StatusCode, IngestResponse) = request_typed(
        &router,
        Method::POST,
        "/api/ingest/webpage",
        json_body(&webpage_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(webpage.document_id, "https://typed.example/page");
    assert!(webpage.chunks_created > 0);

    let memory_request = IngestMemoryRequest {
        source: "typed-contract".to_string(),
        source_id: "typed-memory".to_string(),
        title: "Typed memory".to_string(),
        content: "Typed memory contract content.".to_string(),
        url: None,
        tags: Some(vec!["contract".to_string()]),
        metadata: None,
    };
    let (status, memory): (StatusCode, IngestResponse) = request_typed(
        &router,
        Method::POST,
        "/api/ingest/memory",
        json_body(&memory_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(memory.document_id, "typed-memory");
    assert!(memory.chunks_created > 0);

    let (status, deleted): (StatusCode, DeleteResponse) = request_typed(
        &router,
        Method::DELETE,
        "/api/documents/typed-contract/typed-text",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.deleted);

    let import_request = ImportMemoriesRequest {
        source: "other".to_string(),
        content: "- Typed import contract canary remains parseable.".to_string(),
        label: None,
        space: WriteSpaceTarget::Inherit,
    };
    let (status, imported): (StatusCode, ImportMemoriesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/import/memories",
        json_body(&import_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(imported.imported, 1);
    assert_eq!(imported.skipped, 0);
    assert!(!imported.batch_id.is_empty());

    let archive_path = tmp.path().join("empty-claude-export.zip");
    write_empty_claude_export(&archive_path);
    let chat_request = ImportChatExportRequest {
        path: archive_path.to_string_lossy().into_owned(),
    };
    let (status, chat_import): (StatusCode, ImportChatExportResponse) = request_typed(
        &router,
        Method::POST,
        "/api/import/chat-export",
        json_body(&chat_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(chat_import.vendor, "claude");
    assert_eq!(chat_import.conversations_total, 0);
    assert_eq!(chat_import.conversations_new, 0);
    assert_eq!(chat_import.conversations_skipped_existing, 0);
    assert_eq!(chat_import.memories_stored, 0);
    assert!(!chat_import.import_id.is_empty());

    db.start_import_state(
        "imp-typed-pending",
        wenlan_core::chat_import::Vendor::Claude,
        "/tmp/typed-pending.zip",
    )
    .await
    .unwrap();
    let (status, pending): (StatusCode, Vec<PendingImport>) =
        request_typed(&router, Method::GET, "/api/import/state", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, "imp-typed-pending");
    assert_eq!(pending[0].vendor, "claude");
    assert_eq!(pending[0].stage, "parsing");
}

#[tokio::test]
async fn moved_ingest_and_import_routes_preserve_typed_error_contracts() {
    let router = wenlan_server::router::build_router(Arc::new(RwLock::new(ServerState::default())));

    let text_request = IngestTextRequest {
        source: "typed-contract".to_string(),
        source_id: "typed-text".to_string(),
        title: "Typed text".to_string(),
        content: "Typed text contract content.".to_string(),
        url: None,
        metadata: None,
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/ingest/text",
        json_body(&text_request),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error.error, "Database not initialized");

    let webpage_request = IngestWebpageRequest {
        url: "https://typed.example/page".to_string(),
        title: "Typed webpage".to_string(),
        content: "Typed webpage contract content.".to_string(),
        metadata: None,
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/ingest/webpage",
        json_body(&webpage_request),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error.error, "Database not initialized");

    let memory_request = IngestMemoryRequest {
        source: "typed-contract".to_string(),
        source_id: "typed-memory".to_string(),
        title: "Typed memory".to_string(),
        content: "Typed memory contract content.".to_string(),
        url: None,
        tags: Some(vec!["contract".to_string()]),
        metadata: None,
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/ingest/memory",
        json_body(&memory_request),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error.error, "Database not initialized");

    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::DELETE,
        "/api/documents/typed-contract/typed-text",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error.error, "Database not initialized");

    let invalid_import = ImportMemoriesRequest {
        source: "invalid".to_string(),
        content: "- Typed invalid import remains long enough.".to_string(),
        label: None,
        space: WriteSpaceTarget::Inherit,
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/import/memories",
        json_body(&invalid_import),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.error.contains("Invalid source"));

    let missing_dir = tempfile::tempdir().unwrap();
    let missing_archive = ImportChatExportRequest {
        path: missing_dir
            .path()
            .join("missing-chat-export.zip")
            .to_string_lossy()
            .into_owned(),
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/import/chat-export",
        json_body(&missing_archive),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error.error.contains("failed to stat"));

    let (status, error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Method::GET, "/api/import/state", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error.error, "database not initialized");
}
