// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tower::ServiceExt;
use wenlan_types::requests::{AddSourceRequest, OnDeviceModelRequest, UpdateConfigRequest};
use wenlan_types::responses::{ConfigResponse, OnDeviceModelResponse, SuccessResponse};
use wenlan_types::sources::Source;

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

#[derive(Debug, Serialize)]
struct AnthropicKeyRequest {
    api_key: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SetupStatusResponse {
    setup_completed: bool,
    mode: String,
    anthropic_key_configured: bool,
    local_model_selected: Option<String>,
    local_model_loaded: Option<String>,
    local_model_cached: bool,
    external_llm: ExternalLlmStatus,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ExternalLlmStatus {
    configured: bool,
    loaded: bool,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ResolvedRoutingResponse {
    everyday: JobRoute,
    synthesis: JobRoute,
    pool: RoutingPool,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct JobRoute {
    source: String,
    model: Option<String>,
    mode: String,
    pin: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct RoutingPool {
    anthropic: AnthropicPool,
    external: Option<ExternalPool>,
    on_device: Option<OnDevicePool>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AnthropicPool {
    configured: bool,
    everyday_model: Option<String>,
    synthesis_model: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ExternalPool {
    endpoint: String,
    model: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OnDevicePool {
    selected: Option<String>,
    loaded: bool,
    loading: bool,
}

struct DataDirGuard {
    previous: Option<std::ffi::OsString>,
    _root: tempfile::TempDir,
}

impl DataDirGuard {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("WENLAN_DATA_DIR");
        std::env::set_var("WENLAN_DATA_DIR", root.path());
        Self {
            previous,
            _root: root,
        }
    }
}

impl Drop for DataDirGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var("WENLAN_DATA_DIR", previous);
        } else {
            std::env::remove_var("WENLAN_DATA_DIR");
        }
    }
}

async fn request_bytes(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Option<Vec<u8>>,
) -> (StatusCode, Vec<u8>) {
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
        .unwrap()
        .to_vec();
    (status, bytes)
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
    let (status, bytes) = request_bytes(router, method.clone(), uri, body).await;
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

fn update_config_request(
    clipboard_enabled: Option<bool>,
    everyday_source: Option<&str>,
) -> UpdateConfigRequest {
    UpdateConfigRequest {
        skip_apps: None,
        skip_title_patterns: None,
        private_browsing_detection: None,
        setup_completed: None,
        clipboard_enabled,
        screen_capture_enabled: None,
        remote_access_enabled: None,
        routine_model: None,
        synthesis_model: None,
        external_llm_endpoint: None,
        external_llm_model: None,
        external_llm_api_key: None,
        everyday_source: everyday_source.map(str::to_string),
        synthesis_source: None,
        page_map_auto_suggest: None,
    }
}

#[tokio::test]
async fn moved_source_and_config_routes_preserve_typed_http_contracts() {
    let _config_root = DataDirGuard::new();
    wenlan_core::config::save_config(&wenlan_core::config::Config::default()).unwrap();
    let (router, _db_tmp, _db) = common::test_app_no_gate().await;

    let (status, sources): (StatusCode, Vec<Source>) =
        request_typed(&router, Method::GET, "/api/sources", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(sources.is_empty());

    let source_root = tempfile::tempdir().unwrap();
    let note_path = source_root.path().join("contract-note.md");
    std::fs::write(
        &note_path,
        "# Contract note\n\nSource route registration stays typed and reachable.",
    )
    .unwrap();
    let add_source = AddSourceRequest {
        source_type: "directory".to_string(),
        path: source_root.path().to_string_lossy().into_owned(),
    };
    let (status, source): (StatusCode, Source) = request_typed(
        &router,
        Method::POST,
        "/api/sources",
        json_body(&add_source),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(source.path, source_root.path());
    assert_eq!(source.source_type.as_str(), "directory");

    let sync_uri = format!("/api/sources/{}/sync", source.id);
    let (status, sync): (StatusCode, wenlan_server::source_routes::SyncStatsResponse) =
        request_typed(&router, Method::POST, &sync_uri, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sync.files_found, 1);
    assert_eq!(sync.errors, 0);

    let delete_uri = format!("/api/sources/{}", source.id);
    let (status, body) = request_bytes(&router, Method::DELETE, &delete_uri, None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(body.is_empty());

    let (status, sources): (StatusCode, Vec<Source>) =
        request_typed(&router, Method::GET, "/api/sources", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(sources.is_empty());

    let invalid_source = AddSourceRequest {
        source_type: "invalid".to_string(),
        path: source_root.path().to_string_lossy().into_owned(),
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/sources",
        json_body(&invalid_source),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.error.contains("Unknown source type"));

    let (status, error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Method::DELETE, "/api/sources/missing", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(error.error.contains("Source not found"));

    let (status, error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Method::POST, "/api/sources/missing/sync", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(error.error.contains("Source not found"));

    let (status, config): (StatusCode, ConfigResponse) =
        request_typed(&router, Method::GET, "/api/config", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!config.clipboard_enabled);

    let update = update_config_request(Some(true), None);
    let (status, config): (StatusCode, ConfigResponse) =
        request_typed(&router, Method::PUT, "/api/config", json_body(&update)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(config.clipboard_enabled);

    let invalid_update = update_config_request(None, Some("invalid"));
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::PUT,
        "/api/config",
        json_body(&invalid_update),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.error.contains("invalid everyday_source"));

    let (status, routing): (StatusCode, ResolvedRoutingResponse) =
        request_typed(&router, Method::GET, "/api/config/routing", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(routing.everyday.source, "basic");
    assert_eq!(routing.synthesis.source, "none");
    assert!(!routing.pool.anthropic.configured);

    let (status, setup): (StatusCode, SetupStatusResponse) =
        request_typed(&router, Method::GET, "/api/setup/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(setup.mode, "basic-memory");
    assert!(!setup.anthropic_key_configured);

    let key_request = AnthropicKeyRequest {
        api_key: "typed-contract-key".to_string(),
    };
    let (status, updated): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        "/api/setup/anthropic-key",
        json_body(&key_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(updated.ok);

    let (status, setup): (StatusCode, SetupStatusResponse) =
        request_typed(&router, Method::GET, "/api/setup/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(setup.anthropic_key_configured);

    let (status, cleared): (StatusCode, SuccessResponse) =
        request_typed(&router, Method::DELETE, "/api/setup/anthropic-key", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(cleared.ok);

    let blank_key = AnthropicKeyRequest {
        api_key: "   ".to_string(),
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::PUT,
        "/api/setup/anthropic-key",
        json_body(&blank_key),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.error.contains("api_key cannot be empty"));

    let (status, models): (StatusCode, OnDeviceModelResponse) =
        request_typed(&router, Method::GET, "/api/on-device-model", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(models.loaded.is_none());
    assert!(!models.models.is_empty());

    let unknown_model = OnDeviceModelRequest {
        model_id: "r5-no-such-model".to_string(),
    };
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/on-device-model/download",
        json_body(&unknown_model),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.error.contains("unknown on-device model id"));
}
