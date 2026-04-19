// SPDX-License-Identifier: AGPL-3.0-only
//! Application state — thin client version.
//!
//! Heavy fields (MemoryDB, LLM providers, prompts, tuning, quality_gate)
//! have been removed — the daemon owns those.  This struct keeps only
//! Tauri-specific UI state: app_handle, sensors, feature flags, and the
//! HTTP client that proxies data requests to the daemon.

use crate::api::OriginClient;
use crate::remote_access::RemoteAccessState;
use origin_core::working_memory::WorkingMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
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
}

pub struct AppState {
    /// HTTP client for the origin-server daemon.
    pub client: OriginClient,
    pub index_status: IndexStatus,
    pub sources: HashMap<String, Box<dyn DataSource>>,
    pub watch_paths: Vec<PathBuf>,
    pub clipboard_enabled: bool,
    pub last_ingestion_at: i64,
    pub current_activity: Option<Activity>,
    pub completed_activities: Vec<Activity>,
    pub app_handle: Option<tauri::AppHandle>,
    /// Unified trigger channel sender.
    pub trigger_tx: Option<tokio::sync::mpsc::Sender<crate::trigger::types::TriggerEvent>>,
    /// Last context bundle received from the router.
    pub last_context_bundle: Option<crate::router::bundle::ContextBundle>,
    /// Ambient overlay dismissal records.
    pub ambient_dismissals: Vec<crate::ambient::types::DismissalRecord>,
    /// Ambient overlay mode: proactive, on_demand, or off.
    pub ambient_mode: crate::ambient::types::AmbientMode,
    /// Screen capture enabled.
    pub screen_capture_enabled: bool,
    /// Selection trigger enabled.
    pub selection_capture_enabled: bool,
    /// Stop flag for selection sensor thread.
    pub selection_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Remote access tunnel state.
    pub remote_access: tokio::sync::Mutex<RemoteAccessState>,
    /// Rolling in-memory buffer of recent captures for zero-query Spotlight.
    ///
    /// Populated by the context consumer (`router/intent.rs`) when it ingests
    /// a screen/quick-thought capture. Served by the `get_working_memory`
    /// Tauri command. Lives in the app process because only the app has
    /// active sensors — captures happen only in the app process.
    pub working_memory: Arc<tokio::sync::Mutex<WorkingMemory>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            client: OriginClient::new(),
            index_status: IndexStatus {
                is_running: false,
                files_indexed: 0,
                files_total: 0,
                last_error: None,
                sources_connected: vec![],
            },
            sources: HashMap::new(),
            watch_paths: vec![],
            clipboard_enabled: false,
            last_ingestion_at: 0,
            current_activity: None,
            completed_activities: vec![],
            app_handle: None,
            trigger_tx: None,
            last_context_bundle: None,
            ambient_dismissals: Vec::new(),
            ambient_mode: crate::ambient::types::AmbientMode::default(),
            screen_capture_enabled: false,
            selection_capture_enabled: false,
            selection_stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            remote_access: tokio::sync::Mutex::new(RemoteAccessState::default()),
            working_memory: Arc::new(tokio::sync::Mutex::new(WorkingMemory::new())),
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
    pub async fn initialize_local(&mut self) -> Result<Vec<PathBuf>, crate::error::OriginError> {
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
        self.clipboard_enabled = config.clipboard_enabled;
        self.screen_capture_enabled = config.screen_capture_enabled;
        self.selection_capture_enabled = config.selection_capture_enabled;

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
