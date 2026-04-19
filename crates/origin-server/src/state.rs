// SPDX-License-Identifier: Apache-2.0
//! Server state — shared application state for the standalone HTTP daemon.

use crate::ingest_batcher::IngestBatcher;
use crate::scheduler::WriteSignal;
use origin_core::access_tracker::AccessTracker;
use origin_core::db::MemoryDB;
use origin_core::llm_provider::LlmProvider;
use origin_core::prompts::PromptRegistry;
use origin_core::quality_gate::QualityGate;
use origin_core::spaces::SpaceStore;
use origin_core::tuning::TuningConfig;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared state for the HTTP daemon.
///
/// This mirrors the subset of `AppState` (from the Tauri app) that HTTP route
/// handlers actually need. It does NOT include Tauri-specific fields (app_handle,
/// sensors, triggers, ambient overlay, etc.).
pub struct ServerState {
    pub db: Option<Arc<MemoryDB>>,
    /// On-device LLM provider (Qwen via llama-cpp).
    pub llm: Option<Arc<dyn LlmProvider>>,
    /// Registry id of the currently-loaded on-device model (e.g. "qwen3-4b").
    /// Set whenever `llm` is populated with an `OnDeviceProvider`. `None` when
    /// the daemon has no on-device model loaded.
    pub loaded_on_device_model: Option<String>,
    /// API-based LLM provider for routine tasks (Anthropic Haiku by default).
    pub api_llm: Option<Arc<dyn LlmProvider>>,
    /// API-based LLM provider for synthesis tasks (Anthropic Sonnet by default).
    /// Falls back to api_llm when not set.
    pub synthesis_llm: Option<Arc<dyn LlmProvider>>,
    /// External LLM provider (Ollama, LM Studio, etc.)
    pub external_llm: Option<Arc<dyn LlmProvider>>,
    /// Intelligence prompt templates.
    pub prompts: PromptRegistry,
    /// Intelligence tuning parameters.
    pub tuning: TuningConfig,
    /// Debounced access tracker.
    pub access_tracker: AccessTracker,
    /// Pre-store quality gate.
    pub quality_gate: QualityGate,
    /// Spaces, tags & document assignments.
    pub space_store: SpaceStore,
    /// Configured directory watch paths.
    pub watch_paths: Vec<PathBuf>,
    /// Write-event tracker for the event-driven steep scheduler.
    pub write_signal: WriteSignal,
    /// Coalescing batcher for concurrent `/api/memory/store` calls. Groups
    /// requests that arrive within a short window into a single batched
    /// upsert (one FastEmbed call, one libSQL transaction) instead of N
    /// independent ones. `None` when the DB is not initialized — handlers
    /// fall back to the direct per-request upsert path in that case.
    pub ingest_batcher: Option<IngestBatcher>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            db: None,
            llm: None,
            loaded_on_device_model: None,
            api_llm: None,
            synthesis_llm: None,
            external_llm: None,
            prompts: PromptRegistry::default(),
            tuning: TuningConfig::default(),
            access_tracker: AccessTracker::new(),
            quality_gate: QualityGate::new(origin_core::tuning::GateConfig::default()),
            space_store: SpaceStore::default(),
            watch_paths: Vec::new(),
            write_signal: WriteSignal::new(),
            ingest_batcher: None,
        }
    }
}

impl ServerState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The shared state type threaded through all Axum handlers.
pub type SharedState = Arc<RwLock<ServerState>>;
