// SPDX-License-Identifier: Apache-2.0
//! Server state — shared application state for the standalone HTTP daemon.

use crate::ingest_batcher::IngestBatcher;
use crate::lifecycle::ShutdownHandle;
use crate::maintenance_coordinator::MaintenanceCoordinator;
use crate::reflection_debounce::ReflectionDebouncer;
use crate::scheduler::WriteSignal;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_core::access_tracker::AccessTracker;
use wenlan_core::db::MemoryDB;
use wenlan_core::lint::observation::{LintRunObserver, NoopLintRunObserver};
use wenlan_core::llm_provider::LlmProvider;
use wenlan_core::prompts::PromptRegistry;
use wenlan_core::quality_gate::QualityGate;
use wenlan_core::reranker::Reranker;
use wenlan_core::tuning::TuningConfig;
use wenlan_types::responses::RerankerStatus;
use wenlan_types::sources::Source;

#[derive(Clone, Default)]
pub struct LintServerConfig {
    sources: Vec<Source>,
    page_root: Option<PathBuf>,
    clock_epoch_seconds: Option<i64>,
}

impl LintServerConfig {
    pub(crate) const fn new(sources: Vec<Source>, page_root: Option<PathBuf>) -> Self {
        Self {
            sources,
            page_root,
            clock_epoch_seconds: None,
        }
    }

    fn capture() -> Self {
        let config = wenlan_core::config::load_config();
        let page_root = config.knowledge_path_or_default();
        Self::new(config.sources, Some(page_root)).with_clock_epoch_seconds(
            std::env::var("WENLAN_TEST_LINT_EPOCH")
                .ok()
                .and_then(|value| value.parse().ok()),
        )
    }

    pub(crate) const fn with_clock_epoch_seconds(mut self, value: Option<i64>) -> Self {
        self.clock_epoch_seconds = value;
        self
    }

    pub(crate) fn clock(&self) -> wenlan_core::lint::context::LintClock {
        self.clock_epoch_seconds.map_or_else(
            wenlan_core::lint::context::LintClock::capture,
            wenlan_core::lint::context::LintClock::fixed_at,
        )
    }

    pub(crate) fn sources(&self) -> &[Source] {
        &self.sources
    }

    pub(crate) fn page_root(&self) -> Option<&std::path::Path> {
        self.page_root.as_deref()
    }
}

/// Shared state for the HTTP daemon.
///
/// This mirrors the subset of `AppState` (from the Tauri app) that HTTP route
/// handlers actually need. It does NOT include Tauri-specific fields (app_handle,
/// sensors, triggers, ambient overlay, etc.).
pub struct ServerState {
    /// Sticky daemon-lifecycle signal shared by HTTP and background workers.
    pub shutdown: ShutdownHandle,
    pub db: Option<Arc<MemoryDB>>,
    /// One-way, human-readable projection of the daemon-owned Space Brief.
    /// `None` disables projection without affecting the authoritative DB state.
    pub brief_status_root: Option<PathBuf>,
    /// On-device LLM provider (Qwen via llama-cpp).
    pub llm: Option<Arc<dyn LlmProvider>>,
    /// Registry id of the currently-loaded on-device model (e.g. "qwen3-4b").
    /// Set whenever `llm` is populated with an `OnDeviceProvider`. `None` when
    /// the daemon has no on-device model loaded.
    pub loaded_on_device_model: Option<String>,
    /// Sticky reservation spanning the two-sample admission window and the
    /// blocking startup model load. The scheduler observes it so no automatic
    /// heavy turn can race the load after both sampled the same quiet window.
    pub startup_model_load_reserved: Arc<std::sync::atomic::AtomicBool>,
    /// API-based LLM provider for routine tasks (Anthropic Haiku by default).
    pub api_llm: Option<Arc<dyn LlmProvider>>,
    /// API-based LLM provider for synthesis tasks (Anthropic Sonnet by default).
    /// Falls back to api_llm when not set.
    pub synthesis_llm: Option<Arc<dyn LlmProvider>>,
    /// External LLM provider (Ollama, LM Studio, etc.)
    pub external_llm: Option<Arc<dyn LlmProvider>>,
    /// Cross-encoder reranker for retrieval candidates. Wired by the daemon at
    /// startup via `wenlan_core::reranker::init_cross_encoder_reranker`. `None`
    /// means search falls back to embedding+FTS ordering with no rerank pass.
    pub reranker: Option<Arc<dyn Reranker>>,
    /// Observable reranker state for `/api/status`. Distinguishes "never
    /// enabled" (`Disabled`) from "requested but init failed" (`Failed`) —
    /// both leave `reranker == None`. This tracks the DEEP path
    /// (`/api/memory/search` rerank=true); for `full` mode it flips to `Active`
    /// once the background bge-base load completes.
    pub reranker_status: RerankerStatus,
    /// Cross-encoder for the LIGHT paths — quick (`/api/search`) + context
    /// (`/api/context`). Wired (turbo) when `WENLAN_RERANKER_MODE` is
    /// `lite`/`full`. `None` => those paths use plain hybrid ordering.
    pub reranker_light: Option<Arc<dyn Reranker>>,
    /// Observable light-path reranker state for `/api/status`.
    pub reranker_light_status: RerankerStatus,
    /// Resolved `WENLAN_RERANKER_MODE` string ("off"|"lite"|"full") for `/api/status`.
    pub reranker_mode: String,
    /// Intelligence prompt templates.
    pub prompts: PromptRegistry,
    /// Intelligence tuning parameters.
    pub tuning: TuningConfig,
    /// Debounced access tracker.
    pub access_tracker: AccessTracker,
    /// Pre-store quality gate.
    pub quality_gate: QualityGate,
    /// Configured directory watch paths.
    pub watch_paths: Vec<PathBuf>,
    /// Write-event tracker for the event-driven steep scheduler.
    pub write_signal: WriteSignal,
    /// Excludes daemon-owned background writers from an approved apply-to-verify window.
    pub maintenance_coordinator: MaintenanceCoordinator,
    /// Per-agent debouncer for background reflection (T22). Coalesces
    /// mid-burst enrichment spawns when `WENLAN_ENABLE_REFLECTION_DEBOUNCE`
    /// is truthy; inert (never consulted) when the flag is unset/0.
    pub reflection_debouncer: ReflectionDebouncer,
    /// Coalescing batcher for concurrent `/api/memory/store` calls. Groups
    /// requests that arrive within a short window into a single batched
    /// upsert (one FastEmbed call, one libSQL transaction) instead of N
    /// independent ones. `None` when the DB is not initialized — handlers
    /// fall back to the direct per-request upsert path in that case.
    pub ingest_batcher: Option<IngestBatcher>,
    /// Durable, private artifacts for workflow-approved lint repairs.
    /// Core rejects repair operations on platforms without the required
    /// owner-only permissions and directory-sync durability.
    pub repair_root: Option<PathBuf>,
    /// True only while startup recovery intentionally suppresses optional
    /// providers, rerankers, and background runtime workers.
    pub optional_runtime_workers_suspended: bool,
    pub lint_config: LintServerConfig,
    pub lint_observer: Arc<dyn LintRunObserver>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            shutdown: ShutdownHandle::default(),
            db: None,
            brief_status_root: None,
            llm: None,
            loaded_on_device_model: None,
            startup_model_load_reserved: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            api_llm: None,
            synthesis_llm: None,
            external_llm: None,
            reranker: None,
            reranker_status: RerankerStatus::Disabled,
            reranker_light: None,
            reranker_light_status: RerankerStatus::Disabled,
            reranker_mode: String::from("off"),
            prompts: PromptRegistry::default(),
            tuning: TuningConfig::default(),
            access_tracker: AccessTracker::new(),
            quality_gate: QualityGate::new(wenlan_core::tuning::GateConfig::default()),
            watch_paths: Vec::new(),
            write_signal: WriteSignal::new(),
            maintenance_coordinator: MaintenanceCoordinator::default(),
            reflection_debouncer: ReflectionDebouncer::new(),
            ingest_batcher: None,
            repair_root: None,
            optional_runtime_workers_suspended: false,
            lint_config: LintServerConfig::default(),
            lint_observer: Arc::new(NoopLintRunObserver),
        }
    }
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            lint_config: LintServerConfig::capture(),
            ..Self::default()
        }
    }

    /// Override the Page projection root for an isolated server instance.
    pub fn with_page_root(mut self, page_root: PathBuf) -> Self {
        self.lint_config.page_root = Some(page_root);
        self
    }

    /// Override the human-readable Brief receipt root for an isolated server.
    pub fn with_brief_status_root(mut self, root: PathBuf) -> Self {
        self.brief_status_root = Some(root);
        self
    }
}

/// The shared state type threaded through all Axum handlers.
pub type SharedState = Arc<RwLock<ServerState>>;
