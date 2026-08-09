// SPDX-License-Identifier: AGPL-3.0-only
//! Application state — thin client version.
//!
//! Heavy fields (MemoryDB, LLM providers, prompts, tuning, quality_gate)
//! have been removed — the daemon owns those.  This struct keeps only
//! Tauri-specific UI state: app_handle, sensors, feature flags, and the
//! HTTP client that proxies data requests to the daemon.

use crate::api::WenlanClient;
use crate::remote_access::RemoteAccessState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::Emitter;

use crate::activity::{Activity, ActivitySummary, ACTIVITY_GAP_SECS};
use crate::sources::{DataSource, SourceStatus};

/// Retention period for completed activities (90 days).
const ACTIVITY_RETENTION_SECS: i64 = 90 * 86400;

#[derive(Debug, Clone, Serialize)]
pub struct CaptureEvent {
    pub source: String,
    pub source_id: String,
    pub summary: String,
    pub chunks: usize,
    /// When true, AI is still processing this capture (LLM Pass 2 pending).
    pub processing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatus {
    pub is_running: bool,
    pub files_indexed: u64,
    pub files_total: u64,
    pub last_error: Option<String>,
    pub sources_connected: Vec<String>,
    pub reranker: wenlan_types::responses::RerankerStatus,
    pub reranker_light: wenlan_types::responses::RerankerStatus,
    pub reranker_mode: String,
}

pub struct AppState {
    /// HTTP client for the Wenlan daemon.
    pub client: WenlanClient,
    pub index_status: IndexStatus,
    pub sources: HashMap<String, Box<dyn DataSource>>,
    pub watch_paths: Vec<PathBuf>,
    pub last_ingestion_at: i64,
    pub current_activity: Option<Activity>,
    pub completed_activities: Vec<Activity>,
    pub app_handle: Option<tauri::AppHandle>,
    /// Remote access tunnel state.
    pub remote_access: tokio::sync::Mutex<RemoteAccessState>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            client: WenlanClient::new(),
            index_status: IndexStatus {
                is_running: false,
                files_indexed: 0,
                files_total: 0,
                last_error: None,
                sources_connected: vec![],
                reranker: wenlan_types::responses::RerankerStatus::Disabled,
                reranker_light: wenlan_types::responses::RerankerStatus::Disabled,
                reranker_mode: "off".to_string(),
            },
            sources: HashMap::new(),
            watch_paths: vec![],
            last_ingestion_at: 0,
            current_activity: None,
            completed_activities: vec![],
            app_handle: None,
            remote_access: tokio::sync::Mutex::new(RemoteAccessState::default()),
        }
    }
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit_capture_event(&self, event: CaptureEvent) {
        if let Some(ref handle) = self.app_handle {
            if let Err(e) = handle.emit("capture-event", &event) {
                log::warn!("[state] emit capture-event failed: {}", e);
            }
        }
    }

    pub(crate) fn save_all_activities(&self) {
        let cutoff = chrono::Utc::now().timestamp() - ACTIVITY_RETENTION_SECS;
        let mut all: Vec<Activity> = self
            .completed_activities
            .iter()
            .filter(|a| a.ended_at >= cutoff)
            .cloned()
            .collect();
        if let Some(ref current) = self.current_activity {
            all.push(current.clone());
        }
        if let Err(e) = crate::activity::save_activities(&all) {
            log::error!("Failed to save activities: {}", e);
        }
    }

    pub fn touch_activity(&mut self, now: i64) {
        let needs_new = self
            .current_activity
            .as_ref()
            .is_none_or(|a| now - a.ended_at > ACTIVITY_GAP_SECS);
        if needs_new {
            if let Some(old) = self.current_activity.take() {
                self.completed_activities.push(old);
            }
            self.current_activity = Some(Activity::new(now));
        } else if let Some(ref mut a) = self.current_activity {
            a.ended_at = now;
        }
        self.save_all_activities();
    }

    pub fn list_activity_summaries(&self) -> Vec<ActivitySummary> {
        let mut summaries: Vec<ActivitySummary> = self
            .completed_activities
            .iter()
            .map(|a| a.to_summary(false))
            .collect();
        if let Some(ref a) = self.current_activity {
            summaries.push(a.to_summary(true));
        }
        summaries.sort_by_key(|s| std::cmp::Reverse(s.started_at));
        summaries
    }

    /// Initialize after daemon is confirmed healthy.
    /// Loads local file-based state only — no DB or LLM.
    pub async fn initialize_local(&mut self) -> Result<Vec<PathBuf>, crate::error::AppError> {
        use crate::sources::local_files::LocalFilesSource;

        // Register local files source
        self.sources
            .insert("local_files".to_string(), Box::new(LocalFilesSource::new()));

        // Load completed activities
        let cutoff = chrono::Utc::now().timestamp() - ACTIVITY_RETENTION_SECS;
        self.completed_activities = crate::activity::load_activities()
            .into_iter()
            .filter(|a| a.ended_at >= cutoff)
            .collect();

        // Restore persisted config
        let config = crate::config::load_config();

        let mut restored_paths = Vec::new();
        for path in config.directory_source_paths() {
            self.watch_paths.push(path.clone());
            if path.is_dir() {
                if let Some(source) = self.sources.get_mut("local_files") {
                    if let Some(local) = source.as_any_mut().downcast_mut::<LocalFilesSource>() {
                        local.add_watch_path(path.clone());
                    }
                }
                restored_paths.push(path.clone());
            }
        }

        log::info!(
            "App state initialized (thin client) with {} sources, {} restored watch paths",
            self.sources.len(),
            restored_paths.len()
        );
        Ok(restored_paths)
    }

    pub async fn list_sources(&self) -> Vec<SourceStatus> {
        let mut statuses = Vec::new();
        for source in self.sources.values() {
            statuses.push(source.status().await);
        }
        statuses.sort_by(|a, b| a.name.cmp(&b.name));
        statuses
    }
}
