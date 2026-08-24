use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AmbientJob {
    Document,
    Classification,
    StructuredExtract,
    Entity,
    Title,
    PageGrowth,
    Reconcile,
    Citation,
    EdgeGroundingPromote,
}

impl AmbientJob {
    pub(super) const ALL: [Self; 9] = [
        Self::Document,
        Self::Classification,
        Self::StructuredExtract,
        Self::Entity,
        Self::Title,
        Self::PageGrowth,
        Self::Reconcile,
        Self::Citation,
        Self::EdgeGroundingPromote,
    ];

    /// Stable key for this job, used for the `app_metadata` last-run record
    /// and the `/api/ambient/status` wire response. Never renamed once
    /// shipped — it is a persisted key, not just a debug label.
    pub(super) const fn as_key(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Classification => "classification",
            Self::StructuredExtract => "structured_extract",
            Self::Entity => "entity",
            Self::Title => "title",
            Self::PageGrowth => "page_growth",
            Self::Reconcile => "reconcile",
            Self::Citation => "citation",
            Self::EdgeGroundingPromote => "edge_grounding_promote",
        }
    }
}

const AMBIENT_LAST_RUN_KEY_PREFIX: &str = "ambient_last_run_v1";

/// `app_metadata` key recording the last time this job was attempted
/// (selected or not), by either the passive scheduler tick or a forced sweep.
pub(super) fn ambient_last_run_key(job: AmbientJob) -> String {
    format!("{AMBIENT_LAST_RUN_KEY_PREFIX}_{}", job.as_key())
}

async fn persist_ambient_last_run(db: &wenlan_core::db::MemoryDB, job: AmbientJob) {
    let epoch = chrono::Utc::now().timestamp().to_string();
    if let Err(error) = db
        .set_app_metadata(&ambient_last_run_key(job), &epoch)
        .await
    {
        tracing::warn!(
            "[scheduler] failed to persist ambient last-run timestamp for {job:?}: {error}"
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AmbientAvailability {
    pub(super) document: bool,
    pub(super) classification: bool,
    pub(super) structured_extract: bool,
    pub(super) entity: bool,
    pub(super) title: bool,
    pub(super) page_growth: bool,
    pub(super) reconcile: bool,
    pub(super) citation: bool,
    pub(super) edge_grounding_promote: bool,
}

impl AmbientAvailability {
    /// Automatic lanes are a consent boundary as well as a capability check.
    /// Deterministic document preparation remains available without a model so
    /// source data becomes searchable; all inference lanes stay parked until
    /// the pinned provider is both authorized and healthy.
    pub(super) fn for_provider(provider_available: bool) -> Self {
        Self {
            document: true,
            classification: provider_available,
            structured_extract: provider_available,
            entity: provider_available && wenlan_core::db::entity_sweep_enabled(),
            title: provider_available,
            page_growth: provider_available,
            reconcile: provider_available && wenlan_core::db::doc_reconcile_enabled(),
            citation: provider_available && wenlan_core::db::citation_backfill_enabled(),
            // Provider-gated: the mandatory entailment check spends an LLM call,
            // so the lane parks unless the pinned provider is authorized+healthy
            // AND the opt-in flag is set (mirroring reconcile / citation).
            edge_grounding_promote: provider_available
                && wenlan_core::db::edge_grounding_promote_enabled(),
        }
    }

    pub(super) const fn supports(self, job: AmbientJob) -> bool {
        match job {
            AmbientJob::Document => self.document,
            AmbientJob::Classification => self.classification,
            AmbientJob::StructuredExtract => self.structured_extract,
            AmbientJob::Entity => self.entity,
            AmbientJob::Title => self.title,
            AmbientJob::PageGrowth => self.page_growth,
            AmbientJob::Reconcile => self.reconcile,
            AmbientJob::Citation => self.citation,
            AmbientJob::EdgeGroundingPromote => self.edge_grounding_promote,
        }
    }
}

pub(super) struct AmbientSchedule {
    cursor: usize,
    pub(super) next_allowed_at: Instant,
    last_classification: Option<Instant>,
    last_structured_extract: Option<Instant>,
    pub(super) last_entity: Option<Instant>,
    last_title: Option<Instant>,
    last_page_growth: Option<Instant>,
    pub(super) last_reconcile: Option<Instant>,
    pub(super) last_citation: Option<Instant>,
    last_edge_grounding_promote: Option<Instant>,
}

impl AmbientSchedule {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            cursor: 0,
            next_allowed_at: now,
            last_classification: None,
            last_structured_extract: None,
            last_entity: None,
            last_title: None,
            last_page_growth: None,
            last_reconcile: None,
            last_citation: None,
            last_edge_grounding_promote: None,
        }
    }

    pub(super) fn select_due(
        &mut self,
        now: Instant,
        availability: AmbientAvailability,
    ) -> Option<AmbientJob> {
        for _ in 0..AmbientJob::ALL.len() {
            let job = AmbientJob::ALL[self.cursor];
            self.cursor = (self.cursor + 1) % AmbientJob::ALL.len();
            if !availability.supports(job) {
                continue;
            }
            let due = match job {
                AmbientJob::Document => true,
                AmbientJob::Classification => self
                    .last_classification
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::StructuredExtract => self
                    .last_structured_extract
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::Entity => self
                    .last_entity
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::Title => self
                    .last_title
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::PageGrowth => self
                    .last_page_growth
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::Reconcile => self
                    .last_reconcile
                    .is_none_or(|last| now.duration_since(last) >= RECONCILE_SWEEP_INTERVAL),
                AmbientJob::Citation => self
                    .last_citation
                    .is_none_or(|last| now.duration_since(last) >= CITATION_SWEEP_INTERVAL),
                AmbientJob::EdgeGroundingPromote => self
                    .last_edge_grounding_promote
                    .is_none_or(|last| now.duration_since(last) >= EDGE_GROUNDING_SWEEP_INTERVAL),
            };
            if !due {
                continue;
            }
            return Some(job);
        }
        None
    }

    /// Enumerate every currently-due job for one bounded scheduler tick — at
    /// most one full round-robin lap.
    ///
    /// Without this, a quiet tick only ever fires the single job the cursor
    /// lands on; an always-due job at an earlier cursor position (Document,
    /// which has no interval) can starve a job several slots further round
    /// (e.g. Citation) for many ticks.
    ///
    /// The loop bound is "stop the first time `select_due` repeats a job
    /// already collected this call", not merely "call `select_due` at most
    /// `ALL.len()` times": collecting due jobs up front does not call
    /// `note_job_result` in between (that only happens once each job has
    /// actually run, back in the caller), so a job's due-ness is invariant
    /// for the whole `drain_due` call. If fewer than `ALL.len()` jobs are
    /// currently due-and-supported, `select_due`'s own internal scan (itself
    /// bounded to one lap) can still cross a full circuit within a single
    /// call and land back on an already-returned job — counting outer
    /// `select_due` *calls* does not bound total *probes*. Detecting the
    /// repeat, rather than counting calls, is what actually keeps every job
    /// in the result at most once.
    pub(super) fn drain_due(
        &mut self,
        now: Instant,
        availability: AmbientAvailability,
    ) -> Vec<AmbientJob> {
        let mut due = Vec::with_capacity(AmbientJob::ALL.len());
        for _ in 0..AmbientJob::ALL.len() {
            match self.select_due(now, availability) {
                Some(job) if due.contains(&job) => break,
                Some(job) => due.push(job),
                None => break,
            }
        }
        due
    }

    /// Back off an empty periodic lane, but leave known backlog due. The global
    /// thermal cooldown still limits actual work; this only prevents a second
    /// 30-minute delay from turning catch-up into a multi-week drain.
    pub(super) fn note_job_result(&mut self, job: AmbientJob, now: Instant, selected: bool) {
        if selected {
            return;
        }
        match job {
            AmbientJob::Document => {}
            AmbientJob::Classification => self.last_classification = Some(now),
            AmbientJob::StructuredExtract => self.last_structured_extract = Some(now),
            AmbientJob::Entity => self.last_entity = Some(now),
            AmbientJob::Title => self.last_title = Some(now),
            AmbientJob::PageGrowth => self.last_page_growth = Some(now),
            AmbientJob::Reconcile => self.last_reconcile = Some(now),
            AmbientJob::Citation => self.last_citation = Some(now),
            // Backlog drainer, not a full pass: this arm only runs when the
            // slice made no progress (empty backlog), so stamp the interval to
            // back off. A progressing slice returned early above, staying due.
            AmbientJob::EdgeGroundingPromote => self.last_edge_grounding_promote = Some(now),
        }
    }

    pub(super) fn note_thermal_work_completion(
        &mut self,
        now: Instant,
        elapsed: Duration,
        policy: ThermalPolicy,
    ) {
        self.next_allowed_at = now + policy.cooldown_after(elapsed);
    }
}

pub(super) fn ambient_turn_allowed(
    system_resources_idle: bool,
    now: Instant,
    next_allowed_at: Instant,
) -> bool {
    system_resources_idle && now >= next_allowed_at
}

pub(super) fn should_backoff_ambient_lane(selected: bool, llm_calls: usize) -> bool {
    !selected && llm_calls == 0
}

pub(super) fn ambient_work_consumes_thermal_turn(
    job: AmbientJob,
    selected: bool,
    llm_calls: usize,
    page_growth_terminal_no_match_committed: bool,
) -> bool {
    llm_calls > 0
        || (selected
            && (matches!(job, AmbientJob::Document | AmbientJob::Reconcile)
                || (matches!(job, AmbientJob::PageGrowth)
                    && !page_growth_terminal_no_match_committed)))
}

#[derive(Debug)]
pub(super) struct AmbientTurnReport {
    pub(super) job: AmbientJob,
    pub(super) selected: bool,
    pub(super) page_growth_terminal_no_match_committed: bool,
    pub(super) llm_calls: usize,
    pub(super) panicked: bool,
    pub(super) elapsed: Duration,
}

pub(super) fn resolve_ambient_provider(
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
) -> Option<Arc<dyn wenlan_core::llm_provider::LlmProvider>> {
    wenlan_core::refinery::resolve_everyday(everyday_pin, api_llm, external_llm, llm)
        .llm
        .cloned()
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_ambient_job_safe(
    job: AmbientJob,
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery: &wenlan_core::tuning::RefineryConfig,
    distillation: &wenlan_core::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> AmbientTurnReport {
    let started = Instant::now();
    let future = std::panic::AssertUnwindSafe(run_ambient_job(
        job,
        db,
        llm,
        api_llm,
        external_llm,
        everyday_pin,
        prompts,
        refinery,
        distillation,
        knowledge_path,
    ));
    let report = match futures::FutureExt::catch_unwind(future).await {
        Ok(report) => report,
        Err(error) => {
            let message = if let Some(message) = error.downcast_ref::<&str>() {
                (*message).to_string()
            } else if let Some(message) = error.downcast_ref::<String>() {
                message.clone()
            } else {
                "unknown panic".to_string()
            };
            tracing::error!(
                "[scheduler] ambient job={job:?} PANICKED — scheduler continues: {message}"
            );
            AmbientTurnReport {
                job,
                selected: true,
                page_growth_terminal_no_match_committed: false,
                llm_calls: 0,
                panicked: true,
                elapsed: started.elapsed(),
            }
        }
    };
    // Record every attempt (selected or not) so `/api/ambient/status` can
    // show liveness even for a job that keeps finding nothing to do.
    persist_ambient_last_run(db, job).await;
    report
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_ambient_job(
    job: AmbientJob,
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery: &wenlan_core::tuning::RefineryConfig,
    distillation: &wenlan_core::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> AmbientTurnReport {
    let started = Instant::now();
    let observed = resolve_ambient_provider(everyday_pin, api_llm, external_llm, llm)
        .map(|provider| Arc::new(AmbientBudgetProvider::new(provider)));
    let provider: Option<Arc<dyn wenlan_core::llm_provider::LlmProvider>> = observed
        .as_ref()
        .map(|provider| provider.clone() as Arc<dyn wenlan_core::llm_provider::LlmProvider>);

    let mut page_growth_terminal_no_match_committed = false;
    let selected = match job {
        AmbientJob::Document => {
            run_document_enrichment_slice_tick(db, provider.as_ref(), prompts).await > 0
        }
        AmbientJob::Classification => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::ingest::run_classification_enrichment_slice(db, provider, prompts)
                .await
            {
                Ok(report) => report.selected,
                Err(error) => {
                    tracing::warn!("[scheduler] classification slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::StructuredExtract => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::ingest::run_structured_extract_slice(db, provider, prompts).await {
                Ok(report) => report.selected,
                Err(error) => {
                    tracing::warn!("[scheduler] structured extraction slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Entity => {
            let Some(provider) = provider.clone() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            let prompts = prompts.clone();
            match db
                .run_entity_enrichment_slice_with_auto_link(
                    refinery.entity_link_distance as f32,
                    move |content: String| {
                        let provider = provider.clone();
                        let prompts = prompts.clone();
                        async move {
                            wenlan_core::kg::entity_extraction::extract_kg(
                                &provider, &prompts, &content,
                            )
                            .await
                        }
                    },
                )
                .await
            {
                Ok(selected) => selected > 0,
                Err(error) => {
                    tracing::warn!("[scheduler] entity enrichment slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Title => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::post_ingest::run_title_enrichment_slice(db, provider).await {
                Ok(report) => report.selected,
                Err(error) => {
                    tracing::warn!("[scheduler] title enrichment slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::PageGrowth => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::post_ingest::run_page_growth_slice(
                db,
                provider,
                prompts,
                distillation.page_growth_threshold,
                knowledge_path,
            )
            .await
            {
                Ok(report) => {
                    page_growth_terminal_no_match_committed =
                        report.terminal_no_match && report.committed;
                    report.selected
                }
                Err(error) => {
                    tracing::warn!("[scheduler] page growth slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Reconcile => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::reconcile::run_reconcile_slice(
                db,
                provider,
                prompts,
                refinery,
                distillation,
            )
            .await
            {
                Ok(report) => report.progressed,
                Err(error) => {
                    tracing::warn!("[scheduler] reconcile slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Citation => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::citations::run_citation_backfill_slice(db, provider, prompts).await {
                Ok(selected) => selected > 0,
                Err(error) => {
                    tracing::warn!("[scheduler] citation backfill slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::EdgeGroundingPromote => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::edge_grounding::run_edge_grounding_slice(db, provider, prompts).await
            {
                Ok(report) => report.progressed,
                Err(error) => {
                    tracing::warn!("[scheduler] edge grounding slice error: {error}");
                    false
                }
            }
        }
    };

    AmbientTurnReport {
        job,
        selected,
        page_growth_terminal_no_match_committed,
        llm_calls: observed
            .as_ref()
            .map_or(0, |provider| provider.call_count()),
        panicked: false,
        elapsed: started.elapsed(),
    }
}

/// Claim at most one document and advance it by at most one LLM request.
/// Paused rows retain their existing backoff through `claim_next_pending`.
pub(super) async fn run_document_enrichment_slice_tick(
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    prompts: &wenlan_core::prompts::PromptRegistry,
) -> usize {
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    match db.claim_next_pending_for_provider(llm.is_some()).await {
        Ok(Some(entry)) => {
            let slice = std::panic::AssertUnwindSafe(
                wenlan_core::document_enrichment::run_document_enrichment_slice(
                    db,
                    &entry,
                    Some(&knowledge_path),
                    llm,
                    prompts,
                ),
            );
            match futures::FutureExt::catch_unwind(slice).await {
                Ok(_) => 1,
                Err(panic) => {
                    wenlan_core::document_enrichment::pause_document_enrichment_after_panic(
                        db, &entry,
                    )
                    .await;
                    std::panic::resume_unwind(panic);
                }
            }
        }
        Ok(None) => 0,
        Err(error) => {
            tracing::warn!("[scheduler] claim_next_pending failed: {error}");
            0
        }
    }
}
