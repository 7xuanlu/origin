// SPDX-License-Identifier: Apache-2.0
//! Config endpoints — read/write the daemon's persistent Config.

use crate::error::ServerError;
use crate::route_registry::{get, post, put, TrackedRouter};
use crate::state::SharedState;
use axum::extract::State;
use axum::response::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wenlan_core::config;
use wenlan_core::on_device_models::{self, OnDeviceModel};
use wenlan_types::requests::{OnDeviceModelRequest, UpdateConfigRequest};
use wenlan_types::responses::{ConfigResponse, OnDeviceModelEntry, OnDeviceModelResponse};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route(
            "/api/config",
            get(handle_get_config).put(handle_update_config),
        )
        .route("/api/config/routing", get(handle_get_resolved_routing))
        .route("/api/setup/status", get(handle_get_setup_status))
        .route(
            "/api/setup/anthropic-key",
            put(handle_set_anthropic_key).delete(handle_clear_anthropic_key),
        )
        .route("/api/on-device-model", get(handle_get_on_device_model))
        .route(
            "/api/on-device-model/download",
            post(handle_download_on_device_model),
        )
}

fn config_to_response(cfg: &config::Config) -> ConfigResponse {
    ConfigResponse {
        skip_apps: cfg.skip_apps.clone(),
        skip_title_patterns: cfg.skip_title_patterns.clone(),
        private_browsing_detection: cfg.private_browsing_detection,
        setup_completed: cfg.setup_completed,
        clipboard_enabled: cfg.clipboard_enabled,
        screen_capture_enabled: cfg.screen_capture_enabled,
        remote_access_enabled: cfg.remote_access_enabled,
        routine_model: cfg.routine_model.clone(),
        synthesis_model: cfg.synthesis_model.clone(),
        external_llm_endpoint: cfg.external_llm_endpoint.clone(),
        external_llm_model: cfg.external_llm_model.clone(),
        external_llm_api_key_configured: cfg
            .external_llm_api_key
            .as_deref()
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false),
        everyday_source: cfg.everyday_source.clone(),
        synthesis_source: cfg.synthesis_source.clone(),
        page_map_auto_suggest: cfg.page_map_auto_suggest,
    }
}

/// GET /api/config — return current config.
pub async fn handle_get_config() -> Result<Json<ConfigResponse>, ServerError> {
    let cfg = config::load_config();
    Ok(Json(config_to_response(&cfg)))
}

/// PUT /api/config — update config fields (partial update).
pub async fn handle_update_config(
    State(state): State<SharedState>,
    Json(req): Json<UpdateConfigRequest>,
) -> Result<Json<ConfigResponse>, ServerError> {
    let mut cfg = config::load_config();
    let external_touched = req.external_llm_endpoint.is_some()
        || req.external_llm_model.is_some()
        || req.external_llm_api_key.is_some();
    if let Some(v) = req.skip_apps {
        cfg.skip_apps = v;
    }
    if let Some(v) = req.skip_title_patterns {
        cfg.skip_title_patterns = v;
    }
    if let Some(v) = req.private_browsing_detection {
        cfg.private_browsing_detection = v;
    }
    if let Some(v) = req.setup_completed {
        cfg.setup_completed = v;
    }
    if let Some(v) = req.clipboard_enabled {
        cfg.clipboard_enabled = v;
    }
    if let Some(v) = req.screen_capture_enabled {
        cfg.screen_capture_enabled = v;
    }
    if let Some(v) = req.remote_access_enabled {
        cfg.remote_access_enabled = v;
    }
    if let Some(v) = req.routine_model {
        cfg.routine_model = Some(v);
    }
    if let Some(v) = req.synthesis_model {
        cfg.synthesis_model = Some(v);
    }
    if let Some(v) = req.external_llm_endpoint {
        cfg.external_llm_endpoint = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.external_llm_model {
        cfg.external_llm_model = if v.is_empty() { None } else { Some(v) };
    }
    // Key lifecycle contract: omitted = preserve; null/"" = clear; value = replace.
    match req.external_llm_api_key {
        None => {}
        Some(None) => cfg.external_llm_api_key = None,
        Some(Some(v)) => {
            let trimmed = v.trim();
            cfg.external_llm_api_key = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
        }
    }
    // Per-job source pins. Validated before save so an invalid value 4xxes
    // without persisting; `""` clears the pin, omitted preserves it.
    if let Some(v) = req.everyday_source {
        cfg.everyday_source = validate_everyday_source(&v)?;
    }
    if let Some(v) = req.synthesis_source {
        cfg.synthesis_source = validate_synthesis_source(&v)?;
    }
    if let Some(v) = req.page_map_auto_suggest {
        cfg.page_map_auto_suggest = v;
    }
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;
    if external_touched {
        let mut s = state.write().await;
        apply_external_provider(&mut s, &cfg);
    }
    Ok(Json(config_to_response(&cfg)))
}

#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub ok: bool,
}

// ── Setup/status endpoints ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct SetupStatusResponse {
    pub setup_completed: bool,
    pub mode: String,
    pub anthropic_key_configured: bool,
    pub local_model_selected: Option<String>,
    pub local_model_loaded: Option<String>,
    pub local_model_cached: bool,
    pub external_llm: ExternalLlmStatus,
}

#[derive(Debug, Serialize)]
pub struct ExternalLlmStatus {
    pub configured: bool,
    pub loaded: bool,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicKeyRequest {
    pub api_key: String,
}

fn has_anthropic_key(cfg: &config::Config) -> bool {
    cfg.anthropic_api_key
        .as_deref()
        .map(|key| !key.trim().is_empty())
        .unwrap_or(false)
}

fn apply_anthropic_provider(state: &mut crate::state::ServerState, cfg: &config::Config) {
    if let Some(ref key) = cfg.anthropic_api_key {
        if !key.trim().is_empty() {
            let routine_model = cfg
                .routine_model
                .clone()
                .unwrap_or_else(|| wenlan_core::llm_provider::DEFAULT_ROUTINE_MODEL.to_string());
            let synthesis_model = cfg
                .synthesis_model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-6".to_string());
            state.api_llm = Some(Arc::new(wenlan_core::llm_provider::ApiProvider::new(
                key.clone(),
                routine_model,
            )));
            state.synthesis_llm = Some(Arc::new(wenlan_core::llm_provider::ApiProvider::new(
                key.clone(),
                synthesis_model,
            )));
            return;
        }
    }
    state.api_llm = None;
    state.synthesis_llm = None;
}

/// Validate an `everyday_source` PATCH value. `""` clears the pin; a known
/// source stores; anything else is a 4xx. No vendor is privileged — all three
/// sources are equally valid pins.
fn validate_everyday_source(v: &str) -> Result<Option<String>, ServerError> {
    match v {
        "" => Ok(None),
        "anthropic" | "external" | "on_device" => Ok(Some(v.to_string())),
        other => Err(ServerError::ValidationError(format!(
            "invalid everyday_source '{other}' (expected anthropic | external | on_device)"
        ))),
    }
}

/// Validate a `synthesis_source` PATCH value. `""` clears; a known source
/// stores; anything else is a 4xx. Pins carry no capability restriction —
/// on-device is a first-class synthesis pin. The legacy
/// `WENLAN_PREFER_ON_DEVICE_COMPILE` gate applies only to explicit foreground
/// callers of `synthesis_route`, not to background routing.
fn validate_synthesis_source(v: &str) -> Result<Option<String>, ServerError> {
    match v {
        "" => Ok(None),
        "anthropic" | "external" | "on_device" => Ok(Some(v.to_string())),
        other => Err(ServerError::ValidationError(format!(
            "invalid synthesis_source '{other}' (expected anthropic | external | on_device)"
        ))),
    }
}

/// (Re)build or clear the external OpenAI-compatible provider from config.
/// Mirrors `apply_anthropic_provider` so `PUT /api/config` hot-swaps the slot.
fn apply_external_provider(state: &mut crate::state::ServerState, cfg: &config::Config) {
    match (&cfg.external_llm_endpoint, &cfg.external_llm_model) {
        (Some(endpoint), Some(model)) if !endpoint.is_empty() && !model.is_empty() => {
            state.external_llm = Some(Arc::new(
                wenlan_core::llm_provider::OpenAICompatibleProvider::new_with_key(
                    endpoint.clone(),
                    model.clone(),
                    cfg.external_llm_api_key.clone(),
                ),
            ));
        }
        _ => {
            state.external_llm = None;
        }
    }
}

/// GET /api/setup/status — return setup + model/key status for every client.
pub async fn handle_get_setup_status(
    State(state): State<SharedState>,
) -> Result<Json<SetupStatusResponse>, ServerError> {
    let cfg = config::load_config();
    let selected_model = cfg
        .on_device_model
        .as_deref()
        .map(|id| on_device_models::resolve_or_default(Some(id)));
    let local_model_cached = selected_model
        .map(on_device_models::is_cached)
        .unwrap_or(false);
    let (local_model_loaded, external_loaded) = {
        let s = state.read().await;
        (s.loaded_on_device_model.clone(), s.external_llm.is_some())
    };
    let anthropic_key_configured = has_anthropic_key(&cfg);
    let mode = if anthropic_key_configured {
        "anthropic-key"
    } else if selected_model.is_some() {
        "local-model"
    } else {
        "basic-memory"
    };
    let external_configured = matches!(
        (&cfg.external_llm_endpoint, &cfg.external_llm_model),
        (Some(e), Some(m)) if !e.is_empty() && !m.is_empty()
    );

    Ok(Json(SetupStatusResponse {
        setup_completed: cfg.setup_completed,
        mode: mode.to_string(),
        anthropic_key_configured,
        local_model_selected: selected_model.map(|model| model.id.to_string()),
        local_model_loaded,
        local_model_cached,
        external_llm: ExternalLlmStatus {
            configured: external_configured,
            loaded: external_loaded,
        },
    }))
}

// ── Resolved routing endpoint ───────────────────────────────────────────────

/// Resolved route for one job class: the source that serves it, the model, and
/// how it was chosen.
#[derive(Debug, Serialize)]
pub struct JobRoute {
    /// The RESOLVED source that serves this job.
    /// everyday: "anthropic" | "external" | "on_device" | "basic";
    /// synthesis: "anthropic" | "external" | "on_device" | "none".
    pub source: String,
    pub model: Option<String>,
    /// "pinned" (explicit source pin honored), "pinned_unavailable" (the exact
    /// pinned source is not loaded), or "unconfigured" (no source was approved).
    pub mode: String,
    /// The raw configured source pin for this job, or `null` when unpinned.
    /// On `pinned_unavailable`, `source` remains the exact pinned source while
    /// `model` is null; the daemon never silently crosses to another source.
    /// everyday: "anthropic" | "external" | "on_device"; synthesis: "anthropic"
    /// | "external".
    pub pin: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicPool {
    pub configured: bool,
    pub everyday_model: Option<String>,
    pub synthesis_model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExternalPool {
    pub endpoint: String,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct OnDevicePool {
    pub selected: Option<String>,
    pub loaded: bool,
    pub loading: bool,
}

#[derive(Debug, Serialize)]
pub struct RoutingPool {
    pub anthropic: AnthropicPool,
    pub external: Option<ExternalPool>,
    pub on_device: Option<OnDevicePool>,
}

#[derive(Debug, Serialize)]
pub struct ResolvedRoutingResponse {
    pub everyday: JobRoute,
    pub synthesis: JobRoute,
    pub pool: RoutingPool,
}

/// GET /api/config/routing — resolved per-job routing derived at request time
/// from the live provider slots, using the SAME hard-pin resolvers the refinery
/// runs so what the app displays cannot drift from what the daemon does. Never
/// includes key material.
pub async fn handle_get_resolved_routing(
    State(state): State<SharedState>,
) -> Result<Json<ResolvedRoutingResponse>, ServerError> {
    let cfg = config::load_config();
    let s = state.read().await;

    // Live resolution — the exact hard-pin resolvers background work runs, so
    // display cannot drift from behavior. Missing pins authorize no inference;
    // an unavailable pin remains pinned and never falls back across sources.
    let everyday_pin = wenlan_core::refinery::EverydaySource::parse(cfg.everyday_source.as_deref());
    let everyday_route = wenlan_core::refinery::resolve_everyday(
        everyday_pin,
        s.api_llm.as_ref(),
        s.external_llm.as_ref(),
        s.llm.as_ref(),
    );
    let everyday = JobRoute {
        source: everyday_route.source.as_str().to_string(),
        model: everyday_route.llm.map(|p| p.model_id()),
        mode: everyday_route.mode.as_str().to_string(),
        pin: cfg.everyday_source.clone(),
    };
    let synthesis_pin =
        wenlan_core::refinery::SynthesisSource::parse(cfg.synthesis_source.as_deref());
    let synth_route = wenlan_core::refinery::resolve_synthesis(
        synthesis_pin,
        s.synthesis_llm.as_ref(),
        s.api_llm.as_ref(),
        s.external_llm.as_ref(),
        s.llm.as_ref(),
    );
    let synthesis = JobRoute {
        source: synth_route.source.as_str().to_string(),
        model: synth_route.llm.map(|p| p.model_id()),
        mode: synth_route.mode.as_str().to_string(),
        pin: cfg.synthesis_source.clone(),
    };

    // Pool — the configuration view (what each lane WOULD use). Anthropic model
    // names come from the live slot when loaded, else the configured default,
    // matching `apply_anthropic_provider`.
    let anthropic_configured = has_anthropic_key(&cfg);
    let anthropic = AnthropicPool {
        configured: anthropic_configured,
        everyday_model: anthropic_configured.then(|| {
            s.api_llm.as_ref().map(|p| p.model_id()).unwrap_or_else(|| {
                cfg.routine_model
                    .clone()
                    .unwrap_or_else(|| wenlan_core::llm_provider::DEFAULT_ROUTINE_MODEL.to_string())
            })
        }),
        synthesis_model: anthropic_configured.then(|| {
            s.synthesis_llm
                .as_ref()
                .map(|p| p.model_id())
                .unwrap_or_else(|| {
                    cfg.synthesis_model
                        .clone()
                        .unwrap_or_else(|| "claude-sonnet-4-6".to_string())
                })
        }),
    };
    let external = match (&cfg.external_llm_endpoint, &cfg.external_llm_model) {
        (Some(endpoint), Some(model)) if !endpoint.is_empty() && !model.is_empty() => {
            Some(ExternalPool {
                endpoint: endpoint.clone(),
                model: model.clone(),
            })
        }
        _ => None,
    };
    let selected = cfg.on_device_model.as_deref().map(|id| {
        on_device_models::resolve_or_default(Some(id))
            .id
            .to_string()
    });
    let on_device = if selected.is_some() || s.loaded_on_device_model.is_some() || s.llm.is_some() {
        Some(OnDevicePool {
            selected,
            loaded: s.loaded_on_device_model.is_some(),
            loading: s
                .startup_model_load_reserved
                .load(std::sync::atomic::Ordering::Acquire),
        })
    } else {
        None
    };

    Ok(Json(ResolvedRoutingResponse {
        everyday,
        synthesis,
        pool: RoutingPool {
            anthropic,
            external,
            on_device,
        },
    }))
}

/// PUT /api/setup/anthropic-key — save and hot-load the Anthropic provider.
pub async fn handle_set_anthropic_key(
    State(state): State<SharedState>,
    Json(req): Json<AnthropicKeyRequest>,
) -> Result<Json<SuccessResponse>, ServerError> {
    let key = req.api_key.trim().to_string();
    if key.is_empty() {
        return Err(ServerError::ValidationError(
            "api_key cannot be empty".into(),
        ));
    }
    let mut cfg = config::load_config();
    cfg.setup_completed = true;
    cfg.anthropic_api_key = Some(key);
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;
    {
        let mut s = state.write().await;
        apply_anthropic_provider(&mut s, &cfg);
    }
    Ok(Json(SuccessResponse { ok: true }))
}

/// DELETE /api/setup/anthropic-key — clear the Anthropic provider.
pub async fn handle_clear_anthropic_key(
    State(state): State<SharedState>,
) -> Result<Json<SuccessResponse>, ServerError> {
    let mut cfg = config::load_config();
    cfg.anthropic_api_key = None;
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;
    {
        let mut s = state.write().await;
        apply_anthropic_provider(&mut s, &cfg);
    }
    Ok(Json(SuccessResponse { ok: true }))
}

// ── On-device model endpoints ──────────────────────────────────────────────

fn model_entry(model: &OnDeviceModel) -> OnDeviceModelEntry {
    OnDeviceModelEntry {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        param_count: model.param_count.to_string(),
        ram_required_gb: model.ram_required_gb,
        file_size_gb: model.file_size_gb,
        cached: on_device_models::is_cached(model),
    }
}

/// GET /api/on-device-model — returns the list of models with cache/load state.
pub async fn handle_get_on_device_model(
    State(state): State<SharedState>,
) -> Result<Json<OnDeviceModelResponse>, ServerError> {
    let cfg = config::load_config();
    let loaded = {
        let s = state.read().await;
        s.loaded_on_device_model.clone()
    };
    let models: Vec<OnDeviceModelEntry> =
        on_device_models::MODELS.iter().map(model_entry).collect();
    // Resolve selected against registry so stale config values map to the default,
    // but keep local memory distinct from "default local model selected".
    let selected = cfg
        .on_device_model
        .as_deref()
        .map(|id| on_device_models::resolve_or_default(Some(id)))
        .map(|model| model.id.to_string());
    Ok(Json(OnDeviceModelResponse {
        loaded,
        selected,
        models,
    }))
}

/// POST /api/on-device-model/download — download (if needed) and hot-load a model.
///
/// This is a long-running endpoint: the HTTP request stays open until the
/// download + engine init completes. For a 2.7GB model on a fresh laptop this
/// can take minutes. The client should set a generous timeout.
pub async fn handle_download_on_device_model(
    State(state): State<SharedState>,
    Json(req): Json<OnDeviceModelRequest>,
) -> Result<Json<SuccessResponse>, ServerError> {
    // Validate the id against the registry.
    let Some(model) = on_device_models::get_model(&req.model_id) else {
        return Err(ServerError::ValidationError(format!(
            "unknown on-device model id: {}",
            req.model_id
        )));
    };
    let model_id = model.id.to_string();

    // Run the blocking download + engine init on a dedicated thread so the
    // async runtime stays responsive.
    let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
        tokio::task::spawn_blocking(move || {
            let provider =
                wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some(&model_id))?;
            Ok::<_, wenlan_core::error::WenlanError>(
                Arc::new(provider) as Arc<dyn wenlan_core::llm_provider::LlmProvider>
            )
        })
        .await
        .map_err(|e| ServerError::Internal(format!("download task panicked: {}", e)))?
        .map_err(|e| ServerError::Internal(format!("download failed: {}", e)))?;

    // Persist the selection.
    let mut cfg = config::load_config();
    cfg.setup_completed = true;
    cfg.on_device_model = Some(req.model_id.clone());
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;

    // Hot-swap the provider in ServerState. The old provider (if any) is
    // dropped here; its worker thread exits when the channel closes.
    {
        let mut s = state.write().await;
        s.llm = Some(provider);
        s.loaded_on_device_model = Some(req.model_id.clone());
    }

    tracing::info!("[on-device] model {} downloaded and loaded", req.model_id);
    Ok(Json(SuccessResponse { ok: true }))
}

#[cfg(test)]
mod setup_status_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use super::*;
    use crate::state::ServerState;

    struct WenlanDataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl WenlanDataDirGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let previous = std::env::var_os("WENLAN_DATA_DIR");
            std::env::set_var("WENLAN_DATA_DIR", tmp.path());
            Self {
                previous,
                _tmp: tmp,
            }
        }
    }

    impl Drop for WenlanDataDirGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
        }
    }

    async fn response_json(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Minimal provider for seeding routing slots in endpoint tests: a fixed
    /// backend + model id, always available; `generate` is never called.
    struct PinTestProvider {
        backend: wenlan_core::llm_provider::LlmBackend,
        model: &'static str,
    }

    #[async_trait::async_trait]
    impl wenlan_core::llm_provider::LlmProvider for PinTestProvider {
        async fn generate(
            &self,
            _request: wenlan_core::llm_provider::LlmRequest,
        ) -> Result<String, wenlan_core::llm_provider::LlmError> {
            unreachable!("routing endpoint never calls generate()")
        }
        fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            self.model
        }
        fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
            self.backend
        }
        fn model_id(&self) -> String {
            self.model.to_string()
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn setup_status_defaults_to_basic_memory() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert_eq!(body["setup_completed"], false);
        assert_eq!(body["mode"], "basic-memory");
        assert_eq!(body["anthropic_key_configured"], false);
        assert_eq!(body["local_model_selected"], Value::Null);
        assert_eq!(body["external_llm"]["configured"], false);
        assert_eq!(body["external_llm"]["loaded"], false);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn setup_status_reports_external_llm_state() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state.clone());

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = response_json(resp).await;
        assert_eq!(body["external_llm"]["configured"], false);
        assert_eq!(body["external_llm"]["loaded"], false);

        // Configure via PUT /api/config -> hot-swap makes it configured AND loaded.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"external_llm_endpoint":"http://localhost:11434/v1","external_llm_model":"llama3"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = response_json(resp).await;
        assert_eq!(body["external_llm"]["configured"], true);
        assert_eq!(body["external_llm"]["loaded"], true);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn routing_defaults_to_basic_and_none() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/routing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert_eq!(body["everyday"]["source"], "basic");
        assert_eq!(body["everyday"]["model"], Value::Null);
        assert_eq!(body["everyday"]["mode"], "unconfigured");
        // No pins configured → pin is null on both jobs (present key, null value).
        assert_eq!(body["everyday"]["pin"], Value::Null);
        assert_eq!(body["synthesis"]["source"], "none");
        assert_eq!(body["synthesis"]["mode"], "unconfigured");
        assert_eq!(body["synthesis"]["pin"], Value::Null);
        assert_eq!(body["pool"]["anthropic"]["configured"], false);
        assert_eq!(body["pool"]["anthropic"]["everyday_model"], Value::Null);
        assert_eq!(body["pool"]["external"], Value::Null);
        assert_eq!(body["pool"]["on_device"], Value::Null);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn routing_reports_selected_model_waiting_for_startup_admission() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let mut cfg = config::load_config();
        cfg.on_device_model = Some("qwen3-4b".to_string());
        config::save_config(&cfg).unwrap();

        let state = Arc::new(RwLock::new(ServerState::default()));
        state
            .read()
            .await
            .startup_model_load_reserved
            .store(true, std::sync::atomic::Ordering::Release);
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/routing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert_eq!(body["pool"]["on_device"]["selected"], "qwen3-4b");
        assert_eq!(body["pool"]["on_device"]["loaded"], false);
        assert_eq!(body["pool"]["on_device"]["loading"], true);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn routing_reports_external_pool_without_using_it_unpinned_and_omits_keys() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        // Hot-swap an external provider via PUT /api/config, including a key.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"external_llm_endpoint":"http://localhost:11434/v1","external_llm_model":"llama3","external_llm_api_key":"sk-secret-should-not-leak"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/routing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        // Being configured is capability, not consent: neither background job
        // may use the external provider until its source is explicitly pinned.
        assert_eq!(body["everyday"]["source"], "basic");
        assert_eq!(body["everyday"]["model"], Value::Null);
        assert_eq!(body["everyday"]["mode"], "unconfigured");
        assert_eq!(body["synthesis"]["source"], "none");
        assert_eq!(body["synthesis"]["model"], Value::Null);
        assert_eq!(body["synthesis"]["mode"], "unconfigured");
        assert_eq!(body["pool"]["anthropic"]["configured"], false);
        assert_eq!(
            body["pool"]["external"]["endpoint"],
            "http://localhost:11434/v1"
        );
        assert_eq!(body["pool"]["external"]["model"], "llama3");
        // The API key must never appear anywhere in the response.
        assert!(!serde_json::to_string(&body).unwrap().contains("sk-secret"));
    }

    /// THE headline: everyday pinned to on-device while an Anthropic key is
    /// configured — a mix that was unreachable before pins (api_llm always won).
    /// The pin is stored through the real PATCH route, then reflected by GET.
    #[tokio::test(flavor = "current_thread")]
    async fn routing_honors_everyday_on_device_pin_over_anthropic_key() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();

        // Seed an Anthropic key on disk so the pool reports it configured.
        let mut cfg = config::load_config();
        cfg.anthropic_api_key = Some("sk-ant-test-key".to_string());
        config::save_config(&cfg).unwrap();

        // Providers can't be built from config alone in a unit test, so seed the
        // live slots directly: an on-device model plus the Anthropic slots.
        let state = Arc::new(RwLock::new(ServerState::default()));
        {
            let mut s = state.write().await;
            s.llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::OnDevice,
                model: "qwen3-4b",
            }));
            s.loaded_on_device_model = Some("qwen3-4b".to_string());
            s.api_llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::Api,
                model: "claude-haiku",
            }));
            s.synthesis_llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::Api,
                model: "claude-sonnet",
            }));
        }
        let app = crate::router::build_router(state.clone());

        // Pin everyday to on-device through the real PATCH route.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"everyday_source":"on_device"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/routing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        // Everyday follows the pin to on-device even though Anthropic is present.
        assert_eq!(body["everyday"]["source"], "on_device");
        assert_eq!(body["everyday"]["mode"], "pinned");
        assert_eq!(body["everyday"]["model"], "qwen3-4b");
        // The raw pin is echoed alongside the resolved source.
        assert_eq!(body["everyday"]["pin"], "on_device");
        // Anthropic is configured but cannot serve unpinned synthesis work.
        assert_eq!(body["pool"]["anthropic"]["configured"], true);
        assert_eq!(body["synthesis"]["source"], "none");
        assert_eq!(body["synthesis"]["model"], Value::Null);
        assert_eq!(body["synthesis"]["mode"], "unconfigured");
        assert_eq!(body["synthesis"]["pin"], Value::Null);
    }

    /// Synthesis counterpart of the headline: pinned to on-device while an
    /// Anthropic key is configured. Proves GET /api/config/routing reports
    /// on_device for synthesis with no compile gate.
    #[tokio::test(flavor = "current_thread")]
    async fn routing_honors_synthesis_on_device_pin() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();

        // Seed an Anthropic key so the pool reports it configured.
        let mut cfg = config::load_config();
        cfg.anthropic_api_key = Some("sk-ant-test-key".to_string());
        config::save_config(&cfg).unwrap();

        let state = Arc::new(RwLock::new(ServerState::default()));
        {
            let mut s = state.write().await;
            s.llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::OnDevice,
                model: "qwen3-4b",
            }));
            s.loaded_on_device_model = Some("qwen3-4b".to_string());
            s.api_llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::Api,
                model: "claude-haiku",
            }));
            s.synthesis_llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::Api,
                model: "claude-sonnet",
            }));
        }
        let app = crate::router::build_router(state.clone());

        // Pin synthesis to on-device through the real PATCH route.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"synthesis_source":"on_device"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/routing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        // Synthesis follows the pin to on-device even though Anthropic is present.
        assert_eq!(body["synthesis"]["source"], "on_device");
        assert_eq!(body["synthesis"]["mode"], "pinned");
        assert_eq!(body["synthesis"]["model"], "qwen3-4b");
        assert_eq!(body["synthesis"]["pin"], "on_device");
        // Anthropic is still configured; unpinned everyday inference is disabled.
        assert_eq!(body["pool"]["anthropic"]["configured"], true);
        assert_eq!(body["everyday"]["source"], "basic");
        assert_eq!(body["everyday"]["model"], Value::Null);
        assert_eq!(body["everyday"]["mode"], "unconfigured");
        assert_eq!(body["everyday"]["pin"], Value::Null);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn routing_keeps_unavailable_pins_without_cross_source_fallback() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();

        let mut cfg = config::load_config();
        cfg.everyday_source = Some("external".to_string());
        cfg.synthesis_source = Some("external".to_string());
        cfg.anthropic_api_key = Some("sk-ant-test-key".to_string());
        config::save_config(&cfg).unwrap();

        // Anthropic is live, but the explicitly pinned external slot is absent.
        let state = Arc::new(RwLock::new(ServerState::default()));
        {
            let mut s = state.write().await;
            s.api_llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::Api,
                model: "claude-haiku",
            }));
            s.synthesis_llm = Some(Arc::new(PinTestProvider {
                backend: wenlan_core::llm_provider::LlmBackend::Api,
                model: "claude-sonnet",
            }));
        }
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config/routing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;

        for job in ["everyday", "synthesis"] {
            assert_eq!(body[job]["source"], "external");
            assert_eq!(body[job]["model"], Value::Null);
            assert_eq!(body[job]["mode"], "pinned_unavailable");
            assert_eq!(body[job]["pin"], "external");
        }
    }

    /// Divergence case: config on disk has endpoint+model set (so `configured`
    /// is true), but ServerState was built fresh — never hot-swapped by a PUT
    /// /api/config call — so `external_llm` in state is still `None`
    /// (`loaded` is false). This is the case a daemon restart hits: config
    /// persists across restarts, but the in-memory provider slot does not
    /// until something re-loads it.
    #[tokio::test(flavor = "current_thread")]
    async fn setup_status_reports_configured_but_not_loaded_when_state_untouched() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();

        // Seed the config file directly — bypass PUT /api/config entirely so
        // no hot-swap ever runs.
        let mut cfg = config::load_config();
        cfg.external_llm_endpoint = Some("http://localhost:11434/v1".to_string());
        cfg.external_llm_model = Some("llama3".to_string());
        config::save_config(&cfg).unwrap();

        // Fresh ServerState: external_llm defaults to None (never hot-swapped).
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert_eq!(body["external_llm"]["configured"], true);
        assert_eq!(body["external_llm"]["loaded"], false);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn set_anthropic_key_marks_setup_completed_and_hot_loads_provider() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state.clone());

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/setup/anthropic-key")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"api_key":"sk-ant-test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let cfg = config::load_config();
        assert!(cfg.setup_completed);
        assert_eq!(cfg.anthropic_api_key.as_deref(), Some("sk-ant-test"));
        {
            let s = state.read().await;
            assert!(s.api_llm.is_some());
            assert!(s.synthesis_llm.is_some());
        }

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert_eq!(body["setup_completed"], true);
        assert_eq!(body["mode"], "anthropic-key");
        assert_eq!(body["anthropic_key_configured"], true);
    }
}

#[cfg(test)]
mod config_model_fields_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use super::*;
    use crate::state::ServerState;

    struct DataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl DataDirGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let previous = std::env::var_os("WENLAN_DATA_DIR");
            std::env::set_var("WENLAN_DATA_DIR", tmp.path());
            Self {
                previous,
                _tmp: tmp,
            }
        }
    }

    impl Drop for DataDirGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
        }
    }

    async fn response_json(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn get_config_returns_null_model_fields_by_default() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        // Optional fields absent when None
        assert_eq!(body["routine_model"], Value::Null);
        assert_eq!(body["synthesis_model"], Value::Null);
        assert_eq!(body["external_llm_endpoint"], Value::Null);
        assert_eq!(body["external_llm_model"], Value::Null);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn put_config_round_trips_model_fields() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        let body = serde_json::json!({
            "routine_model": "claude-haiku-4-5-20251001",
            "synthesis_model": "claude-opus-4-6",
            "external_llm_endpoint": "http://localhost:11434/v1",
            "external_llm_model": "llama3"
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body = response_json(resp).await;
        assert_eq!(resp_body["routine_model"], "claude-haiku-4-5-20251001");
        assert_eq!(resp_body["synthesis_model"], "claude-opus-4-6");
        assert_eq!(
            resp_body["external_llm_endpoint"],
            "http://localhost:11434/v1"
        );
        assert_eq!(resp_body["external_llm_model"], "llama3");

        // Verify persisted to disk
        let cfg = config::load_config();
        assert_eq!(
            cfg.routine_model.as_deref(),
            Some("claude-haiku-4-5-20251001")
        );
        assert_eq!(cfg.synthesis_model.as_deref(), Some("claude-opus-4-6"));
        assert_eq!(
            cfg.external_llm_endpoint.as_deref(),
            Some("http://localhost:11434/v1")
        );
        assert_eq!(cfg.external_llm_model.as_deref(), Some("llama3"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn put_config_toggles_page_map_auto_suggest_and_preserves_when_omitted() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        async fn put(app: &crate::router::AppRouter, body: serde_json::Value) -> Value {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("PUT")
                        .uri("/api/config")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            response_json(resp).await
        }

        // Set true explicitly — response echoes it and it persists.
        let body = put(&app, serde_json::json!({"page_map_auto_suggest": true})).await;
        assert_eq!(body["page_map_auto_suggest"], true);
        assert!(config::load_config().page_map_auto_suggest);

        // Omitting the field preserves the stored value.
        let body = put(&app, serde_json::json!({"clipboard_enabled": true})).await;
        assert_eq!(body["page_map_auto_suggest"], true);
        assert!(config::load_config().page_map_auto_suggest);

        // Flip to false — response echoes it and it persists.
        let body = put(&app, serde_json::json!({"page_map_auto_suggest": false})).await;
        assert_eq!(body["page_map_auto_suggest"], false);
        assert!(!config::load_config().page_map_auto_suggest);
    }
}

#[cfg(test)]
mod external_llm_lifecycle_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use super::*;
    use crate::state::ServerState;

    struct DataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl DataDirGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let previous = std::env::var_os("WENLAN_DATA_DIR");
            std::env::set_var("WENLAN_DATA_DIR", tmp.path());
            Self {
                previous,
                _tmp: tmp,
            }
        }
    }

    impl Drop for DataDirGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
        }
    }

    async fn response_json(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn put_config(
        app: &crate::router::AppRouter,
        body: serde_json::Value,
    ) -> (StatusCode, Value) {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        (status, response_json(resp).await)
    }

    async fn get_config(app: &crate::router::AppRouter) -> Value {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        response_json(resp).await
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_key_lifecycle_and_hot_swap() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state.clone());

        // 1. Set endpoint + model + key: hot-swap ON, flag true, value never echoed.
        let (status, body) = put_config(
            &app,
            serde_json::json!({
                "external_llm_endpoint": "http://localhost:11434/v1",
                "external_llm_model": "llama3",
                "external_llm_api_key": "sk-secret-123"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["external_llm_api_key_configured"], true);
        assert!(
            !body.to_string().contains("sk-secret-123"),
            "key value must never be serialized"
        );
        assert!(
            state.read().await.external_llm.is_some(),
            "hot-swap must load the slot"
        );

        // 2. Omitted field preserves the stored key.
        let (_, body) = put_config(&app, serde_json::json!({"clipboard_enabled": true})).await;
        assert_eq!(body["external_llm_api_key_configured"], true);
        assert_eq!(
            config::load_config().external_llm_api_key.as_deref(),
            Some("sk-secret-123")
        );

        // 3. Explicit null clears the key (endpoint+model remain -> slot stays loaded, keyless).
        let (_, body) = put_config(&app, serde_json::json!({"external_llm_api_key": null})).await;
        assert_eq!(body["external_llm_api_key_configured"], false);
        assert!(config::load_config().external_llm_api_key.is_none());
        assert!(state.read().await.external_llm.is_some());

        // 4. Empty string also clears.
        put_config(&app, serde_json::json!({"external_llm_api_key": "sk-2"})).await;
        let (_, body) = put_config(&app, serde_json::json!({"external_llm_api_key": ""})).await;
        assert_eq!(body["external_llm_api_key_configured"], false);

        // 4b. A pasted key with trailing whitespace/newline is stored trimmed.
        put_config(&app, serde_json::json!({"external_llm_api_key": "sk-x\n"})).await;
        assert_eq!(
            config::load_config().external_llm_api_key.as_deref(),
            Some("sk-x"),
            "stored key must be trimmed of pasted whitespace"
        );

        // 5. Clearing the endpoint clears the slot.
        let (_, body) = put_config(&app, serde_json::json!({"external_llm_endpoint": ""})).await;
        assert_eq!(body["external_llm_endpoint"], Value::Null);
        assert!(state.read().await.external_llm.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn patch_stores_valid_source_pins_and_rejects_unknown() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        // Valid pins persist. No vendor is privileged — on_device is a first-
        // class everyday pin.
        let (status, body) = put_config(
            &app,
            serde_json::json!({"everyday_source": "on_device", "synthesis_source": "external"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let cfg = config::load_config();
        assert_eq!(cfg.everyday_source.as_deref(), Some("on_device"));
        assert_eq!(cfg.synthesis_source.as_deref(), Some("external"));
        // The PUT response (ConfigResponse) echoes the stored pins.
        assert_eq!(body["everyday_source"], "on_device");
        assert_eq!(body["synthesis_source"], "external");

        // GET /api/config also echoes the pins (the app reads them here).
        let cfg_body = get_config(&app).await;
        assert_eq!(cfg_body["everyday_source"], "on_device");
        assert_eq!(cfg_body["synthesis_source"], "external");

        // Empty string clears a pin — the cleared pin reads null via GET.
        let (status, _) = put_config(&app, serde_json::json!({"everyday_source": ""})).await;
        assert_eq!(status, StatusCode::OK);
        assert!(config::load_config().everyday_source.is_none());
        let cfg_body = get_config(&app).await;
        assert_eq!(cfg_body["everyday_source"], Value::Null);
        assert_eq!(cfg_body["synthesis_source"], "external");

        // Unknown value is a 4xx and does not persist.
        let (status, _) = put_config(&app, serde_json::json!({"everyday_source": "gpt4"})).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(config::load_config().everyday_source.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn patch_accepts_synthesis_on_device_pin() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        // On-device synthesis is a first-class pin — accepted and persisted even
        // with the legacy foreground fallback gate unset.
        std::env::remove_var("WENLAN_PREFER_ON_DEVICE_COMPILE");
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);

        let (status, body) =
            put_config(&app, serde_json::json!({"synthesis_source": "on_device"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            config::load_config().synthesis_source.as_deref(),
            Some("on_device")
        );
        assert_eq!(body["synthesis_source"], "on_device");
    }
}
