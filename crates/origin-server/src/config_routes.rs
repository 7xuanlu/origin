// SPDX-License-Identifier: Apache-2.0
//! Config endpoints — read/write the daemon's persistent Config.

use crate::error::ServerError;
use crate::state::SharedState;
use axum::extract::State;
use axum::response::Json;
use origin_core::config;
use origin_core::on_device_models::{self, OnDeviceModel};
use origin_types::requests::UpdateConfigRequest;
use origin_types::responses::ConfigResponse;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

fn config_to_response(cfg: &config::Config) -> ConfigResponse {
    ConfigResponse {
        skip_apps: cfg.skip_apps.clone(),
        skip_title_patterns: cfg.skip_title_patterns.clone(),
        private_browsing_detection: cfg.private_browsing_detection,
        setup_completed: cfg.setup_completed,
        clipboard_enabled: cfg.clipboard_enabled,
        screen_capture_enabled: cfg.screen_capture_enabled,
        selection_capture_enabled: cfg.selection_capture_enabled,
        remote_access_enabled: cfg.remote_access_enabled,
    }
}

/// GET /api/config — return current config.
pub async fn handle_get_config() -> Result<Json<ConfigResponse>, ServerError> {
    let cfg = config::load_config();
    Ok(Json(config_to_response(&cfg)))
}

/// PUT /api/config — update config fields (partial update).
pub async fn handle_update_config(
    Json(req): Json<UpdateConfigRequest>,
) -> Result<Json<ConfigResponse>, ServerError> {
    let mut cfg = config::load_config();
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
    if let Some(v) = req.selection_capture_enabled {
        cfg.selection_capture_enabled = v;
    }
    if let Some(v) = req.remote_access_enabled {
        cfg.remote_access_enabled = v;
    }
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(config_to_response(&cfg)))
}

/// GET /api/config/skip-apps — return skip-apps list.
pub async fn handle_get_skip_apps() -> Result<Json<Vec<String>>, ServerError> {
    let cfg = config::load_config();
    Ok(Json(cfg.skip_apps))
}

#[derive(serde::Deserialize)]
pub struct SkipAppsRequest {
    pub apps: Vec<String>,
}

/// PUT /api/config/skip-apps — update skip-apps list.
pub async fn handle_update_skip_apps(
    Json(req): Json<SkipAppsRequest>,
) -> Result<Json<SuccessResponse>, ServerError> {
    let mut cfg = config::load_config();
    cfg.skip_apps = req.apps;
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(SuccessResponse { ok: true }))
}

#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub ok: bool,
}

// ── On-device model endpoints ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct OnDeviceModelEntry {
    pub id: String,
    pub display_name: String,
    pub param_count: String,
    pub ram_required_gb: f64,
    pub file_size_gb: f64,
    pub cached: bool,
}

#[derive(Debug, Serialize)]
pub struct OnDeviceModelResponse {
    /// ID of the model currently loaded in the daemon (if any).
    pub loaded: Option<String>,
    /// ID the user has selected in config (may differ from loaded if a
    /// download is pending or a restart is needed).
    pub selected: Option<String>,
    /// All available models with per-model cache/download state.
    pub models: Vec<OnDeviceModelEntry>,
}

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
    // Resolve selected against registry so stale config values map to the default.
    let selected = on_device_models::resolve_or_default(cfg.on_device_model.as_deref())
        .id
        .to_string();
    Ok(Json(OnDeviceModelResponse {
        loaded,
        selected: Some(selected),
        models,
    }))
}

#[derive(Debug, Deserialize)]
pub struct OnDeviceModelRequest {
    pub model_id: String,
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
    let provider: Arc<dyn origin_core::llm_provider::LlmProvider> =
        tokio::task::spawn_blocking(move || {
            let provider =
                origin_core::llm_provider::OnDeviceProvider::new_with_model(Some(&model_id))?;
            Ok::<_, origin_core::error::OriginError>(
                Arc::new(provider) as Arc<dyn origin_core::llm_provider::LlmProvider>
            )
        })
        .await
        .map_err(|e| ServerError::Internal(format!("download task panicked: {}", e)))?
        .map_err(|e| ServerError::Internal(format!("download failed: {}", e)))?;

    // Persist the selection.
    let mut cfg = config::load_config();
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
