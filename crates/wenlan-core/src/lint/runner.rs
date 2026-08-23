use super::catalog::{catalog_entry, catalog_for_profile, catalog_group, LintCheckGroup};
use super::context::{CancellationToken, ExecutionGate, LintClock, LintContext, PopulationBasis};
use super::observation::{LintRunEvent, LintRunObserver, NoopLintRunObserver};
use super::pages::fs::{
    scan_page_root_controlled, PageFsError, PageFsReceipt, PageScan, PageScanControl,
};
use super::run_config::{EffectiveLintConfig, SemanticProviderConfig};
use super::snapshot::{SnapshotError, SnapshotReceipt};
use crate::db::MemoryDB;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
#[cfg(test)]
use wenlan_types::lint::LintConfigFingerprint;
use wenlan_types::lint::{
    LintAgentSubmission, LintAgentWork, LintApplicability, LintCapabilityContext, LintCheckResult,
    LintCheckResultInput, LintContractError, LintCoverage, LintDbSnapshotMode,
    LintDbSnapshotReceipt, LintDigest, LintOutcome, LintPageSnapshotMode, LintPageSnapshotReceipt,
    LintPrecondition, LintProducerReceipt, LintProfile, LintQuery, LintRecommendationCode,
    LintReport, LintScope, LintSeverity, LintSnapshotReceipts, LintSummaryCode,
    LintValidationMethod, LINT_MAX_EVIDENCE_PER_CHECK,
};

mod configuration;
mod scope;

#[derive(Debug, thiserror::Error)]
pub enum LintRunError {
    #[error("invalid_scope")]
    InvalidScope,
    #[error("lint catalog/result mismatch")]
    CatalogMismatch,
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    #[error(transparent)]
    PageScan(#[from] PageFsError),
    #[error(transparent)]
    Contract(#[from] LintContractError),
}

pub struct LintRunner {
    clock: LintClock,
    gate: ExecutionGate,
    #[cfg(test)]
    scenario: Option<TestScenario>,
    #[cfg(test)]
    synchronization: Option<TestSynchronization>,
    #[cfg(test)]
    memory_features: Option<super::memories::TestMemoryFeatures>,
    #[cfg(test)]
    kg_config: Option<super::kg::KgRunConfig>,
    operations_config: super::operations::OperationsRunConfig,
    runtime_config: super::runtime::RuntimeRunConfig,
    semantic_provider: Option<Arc<dyn crate::llm_provider::LlmProvider>>,
    semantic_external_egress_enabled: bool,
    semantic_agent_request: super::semantic::AgentRequest,
    observer: Arc<dyn LintRunObserver>,
    #[cfg(test)]
    run_timeout_override: Option<std::time::Duration>,
    #[cfg(test)]
    page_timeout_override: Option<std::time::Duration>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TestScenario {
    BlockedGroup,
    MixedQueryFailure,
    PageGroupTimeout,
    ScopedDenominators,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TestSyncPoint {
    AfterTrackerSample,
    BeforeReceipts,
}

#[cfg(test)]
pub(super) struct TestSynchronization {
    point: TestSyncPoint,
    reached: std::sync::Arc<tokio::sync::Barrier>,
    resume: std::sync::Arc<tokio::sync::Barrier>,
}

#[cfg(test)]
pub(super) struct TestSynchronizationControl {
    reached: std::sync::Arc<tokio::sync::Barrier>,
    resume: std::sync::Arc<tokio::sync::Barrier>,
}

#[cfg(test)]
impl TestSynchronization {
    pub(super) fn new(point: TestSyncPoint) -> (Self, TestSynchronizationControl) {
        let reached = std::sync::Arc::new(tokio::sync::Barrier::new(2));
        let resume = std::sync::Arc::new(tokio::sync::Barrier::new(2));
        (
            Self {
                point,
                reached: std::sync::Arc::clone(&reached),
                resume: std::sync::Arc::clone(&resume),
            },
            TestSynchronizationControl { reached, resume },
        )
    }

    async fn hit(&self, point: TestSyncPoint) {
        if self.point == point {
            self.reached.wait().await;
            self.resume.wait().await;
        }
    }
}

#[cfg(test)]
impl TestSynchronizationControl {
    pub(super) async fn wait_until_reached(&self) {
        self.reached.wait().await;
    }

    pub(super) async fn resume(&self) {
        self.resume.wait().await;
    }
}

impl LintRunner {
    pub fn new(clock: LintClock, cancellation: CancellationToken) -> Self {
        Self {
            clock,
            gate: ExecutionGate::new(cancellation),
            #[cfg(test)]
            scenario: None,
            #[cfg(test)]
            synchronization: None,
            #[cfg(test)]
            memory_features: None,
            #[cfg(test)]
            kg_config: None,
            operations_config: super::operations::OperationsRunConfig::unavailable(),
            runtime_config: super::runtime::RuntimeRunConfig::capture(),
            semantic_provider: None,
            semantic_external_egress_enabled: false,
            semantic_agent_request: super::semantic::AgentRequest::Disabled,
            observer: Arc::new(NoopLintRunObserver),
            #[cfg(test)]
            run_timeout_override: None,
            #[cfg(test)]
            page_timeout_override: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_test_scenario(mut self, scenario: TestScenario) -> Self {
        self.scenario = Some(scenario);
        self
    }

    #[cfg(test)]
    pub(super) fn with_test_synchronization(
        mut self,
        synchronization: TestSynchronization,
    ) -> Self {
        self.synchronization = Some(synchronization);
        self
    }

    #[cfg(test)]
    pub(super) fn with_test_memory_features(
        mut self,
        features: super::memories::TestMemoryFeatures,
    ) -> Self {
        self.memory_features = Some(features);
        self
    }

    #[cfg(test)]
    pub(super) fn with_test_kg_config(mut self, config: super::kg::KgRunConfig) -> Self {
        self.kg_config = Some(config);
        self
    }

    pub async fn run(
        &self,
        database: &MemoryDB,
        query: &LintQuery,
        page_root: Option<&Path>,
        page_projection_enabled: bool,
    ) -> Result<LintReport, LintRunError> {
        let profile = query.applied_profile();
        let memory_config = {
            #[cfg(test)]
            if let Some(features) = self.memory_features {
                super::memories::MemoryFeatureConfig::for_test(database, features)
            } else {
                super::memories::MemoryFeatureConfig::capture(database, page_projection_enabled)
            }
            #[cfg(not(test))]
            super::memories::MemoryFeatureConfig::capture(database, page_projection_enabled)
        };
        let kg_config = {
            #[cfg(test)]
            if let Some(config) = self.kg_config {
                config
            } else {
                super::kg::KgRunConfig::capture()
            }
            #[cfg(not(test))]
            super::kg::KgRunConfig::capture()
        };
        let serving_config = super::serving::capture(memory_config, kg_config.serving_enabled);
        let semantic_provider_ready = self
            .semantic_provider
            .as_deref()
            .is_some_and(crate::llm_provider::LlmProvider::is_available);
        let semantic_provider_on_device =
            self.semantic_provider.as_deref().is_some_and(|provider| {
                provider.backend() == crate::llm_provider::LlmBackend::OnDevice
            });
        let semantic_external_egress_enabled = self.semantic_external_egress_enabled
            || self.semantic_provider.as_deref().is_some_and(|provider| {
                provider.backend() != crate::llm_provider::LlmBackend::OnDevice
            });
        let effective_config = EffectiveLintConfig::new(
            page_projection_enabled,
            memory_config,
            kg_config,
            self.operations_config.clone(),
            serving_config,
            self.runtime_config
                .clone()
                .with_clock_epoch_seconds(self.clock.epoch_seconds()),
            SemanticProviderConfig::new(
                semantic_provider_ready,
                semantic_provider_on_device,
                semantic_external_egress_enabled,
                self.semantic_agent_request.is_enabled(),
            ),
        );
        let projection_tracker = database.page_projection_tracker();
        let tracker_before = projection_tracker.sample();
        #[cfg(test)]
        if let Some(synchronization) = &self.synchronization {
            synchronization.hit(TestSyncPoint::AfterTrackerSample).await;
        }
        let snapshot = database
            .open_unpinned_lint_snapshot(Arc::clone(&self.observer))
            .await?;
        let scope = scope::validate(&snapshot, query, self.observer.as_ref()).await?;
        let snapshot = snapshot.pin_analysis().await?;
        let mut execution_failed = self.run_budget_exceeded(profile);

        let page_started = self.clock.elapsed();
        let page_scan = if page_projection_enabled && !execution_failed {
            self.observer.observe(LintRunEvent::PageScan);
            let page_scan = match page_root {
                Some(root) => self.scan_pages(root, profile).await.map(Some),
                None => Ok(None),
            };
            match page_scan {
                Ok(scan) => scan,
                Err(_) => {
                    execution_failed = true;
                    None
                }
            }
        } else {
            None
        };
        let context = LintContext::new(
            &snapshot,
            &scope,
            page_scan.as_ref(),
            &self.clock,
            &self.gate,
            profile,
        );
        let (mut checks, page_elapsed, agent_work) = if execution_failed {
            record_zero_populations(&context, profile)?;
            (
                failed_results_for_profile(&self.clock, profile),
                std::time::Duration::ZERO,
                None,
            )
        } else {
            self.observer.observe(LintRunEvent::AggregateChecks);
            let elapsed = self.clock.elapsed();
            let remaining = self.run_budget(profile).saturating_sub(elapsed);
            let group_result = tokio::select! {
                biased;
                _ = self.gate.cancelled() => None,
                result = tokio::time::timeout(
                    remaining,
                    self.run_groups(&context, &effective_config, page_started),
                ) => result.ok(),
            };
            match group_result {
                Some(result) => result?,
                None => (
                    failed_results_for_context(&context, profile)?,
                    self.page_elapsed(page_started),
                    None,
                ),
            }
        };
        if effective_config.memory.artifact_state_changed(database) {
            checks = inconsistent_selected_results(&self.clock, &checks, |check_id| {
                catalog_entry(check_id).is_some_and(|entry| entry.group == LintCheckGroup::Memories)
            })?;
        }
        if self.page_budget_exceeded(profile, page_elapsed) {
            checks = failed_selected_results(&self.clock, &checks, |check_id| {
                catalog_entry(check_id).is_some_and(|entry| entry.group == LintCheckGroup::Pages)
            })?;
        }
        if self.run_budget_exceeded(profile) {
            checks = failed_results_from_checks(&self.clock, &checks)?;
        }
        validate_profile_catalog_results(profile, &mut checks)?;
        validate_scope_policy(&context, &checks)?;
        drop(context);
        #[cfg(test)]
        if let Some(synchronization) = &self.synchronization {
            synchronization.hit(TestSyncPoint::BeforeReceipts).await;
        }

        let page_before = page_scan
            .as_ref()
            .map(PageScan::normalized_bytes)
            .unwrap_or([0; 32]);
        let (page_after, page_receipt_failed) = match (page_root, page_scan.as_ref()) {
            (Some(root), Some(scan)) => {
                match self.verify_pages(root, scan, profile, page_elapsed).await {
                    Ok(after) => (after, false),
                    Err(_) => (page_before, true),
                }
            }
            _ => (page_before, false),
        };
        let page_changed = page_before != page_after;
        let db_receipt = snapshot.finish().await?;
        let tracker_after = projection_tracker.sample();
        let tracker_unstable = tracker_before.has_active_writes()
            || tracker_after.has_active_writes()
            || tracker_before.generation() != tracker_after.generation();
        if page_changed {
            checks =
                inconsistent_selected_results(&self.clock, &checks, super::pages::uses_filesystem)?;
        }
        if page_receipt_failed {
            checks = failed_selected_results(&self.clock, &checks, super::pages::uses_filesystem)?;
        }
        if !db_receipt.is_consistent() {
            checks = inconsistent_selected_results(&self.clock, &checks, |_| true)?;
        } else if page_projection_enabled && tracker_unstable {
            checks = inconsistent_selected_results(
                &self.clock,
                &checks,
                super::pages::uses_cross_store,
            )?;
        }
        if self.run_budget_exceeded(profile) {
            checks = failed_results_from_checks(&self.clock, &checks)?;
        }
        validate_profile_catalog_results(profile, &mut checks)?;
        self.observer.observe(LintRunEvent::ReportBuild);
        build_report(
            profile,
            scope.into_report(),
            db_receipt,
            page_before,
            page_after,
            effective_config,
            checks,
            agent_work,
        )
    }

    pub fn with_semantic_provider(
        mut self,
        provider: Option<Arc<dyn crate::llm_provider::LlmProvider>>,
    ) -> Self {
        self.semantic_provider = provider;
        self
    }

    pub fn with_semantic_external_egress_enabled(mut self, enabled: bool) -> Self {
        self.semantic_external_egress_enabled = enabled;
        self
    }

    pub fn with_semantic_agent_assist(mut self) -> Self {
        self.semantic_agent_request = super::semantic::AgentRequest::Prepare;
        self
    }

    pub fn with_semantic_agent_submission(mut self, submission: LintAgentSubmission) -> Self {
        self.semantic_agent_request = super::semantic::AgentRequest::Submit(submission);
        self
    }

    /// Run `work` in a cancellable, deadline-bounded blocking task. Shared by
    /// `scan_pages` and `verify_pages`, which differ only in the closure and
    /// the budget arithmetic that produces `timeout`.
    async fn run_page_task<T: Send + 'static>(
        &self,
        timeout: std::time::Duration,
        work: impl FnOnce(PageScanControl) -> Result<T, PageFsError> + Send + 'static,
    ) -> Result<T, PageFsError> {
        let control = PageScanControl::with_timeout(timeout);
        let worker_control = control.clone();
        let mut task = tokio::task::spawn_blocking(move || work(worker_control));
        tokio::select! {
            biased;
            _ = self.gate.cancelled() => {
                control.cancel();
                let _ = task.await;
                Err(PageFsError::Canceled)
            },
            result = tokio::time::timeout(timeout, &mut task) => match result {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err(PageFsError::ReadDirectory),
                Err(_) => {
                    control.cancel();
                    let _ = task.await;
                    Err(PageFsError::DeadlineExceeded)
                },
            },
        }
    }

    async fn scan_pages(&self, root: &Path, profile: LintProfile) -> Result<PageScan, PageFsError> {
        let root = root.to_path_buf();
        let run_remaining = self
            .run_budget(profile)
            .saturating_sub(self.clock.elapsed());
        let timeout = self.page_budget(profile).min(run_remaining);
        let deep = profile == LintProfile::Deep;
        self.run_page_task(timeout, move |control| {
            scan_page_root_controlled(&root, deep, &control)
        })
        .await
    }

    async fn verify_pages(
        &self,
        root: &Path,
        scan: &PageScan,
        profile: LintProfile,
        page_elapsed: std::time::Duration,
    ) -> Result<[u8; 32], PageFsError> {
        let page_remaining = self.page_budget(profile).saturating_sub(page_elapsed);
        let run_remaining = self
            .run_budget(profile)
            .saturating_sub(self.clock.elapsed());
        let timeout = page_remaining.min(run_remaining);
        let root = root.to_path_buf();
        let scan = scan.clone();
        self.run_page_task(timeout, move |control| {
            scan.verify_unchanged_with_control(&root, &control)
                .map(PageFsReceipt::after_normalized_bytes)
        })
        .await
    }

    #[cfg(test)]
    pub(super) fn with_test_run_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.run_timeout_override = Some(timeout);
        self
    }

    #[cfg(test)]
    pub(super) fn with_test_page_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.page_timeout_override = Some(timeout);
        self
    }

    fn run_budget(&self, profile: LintProfile) -> std::time::Duration {
        #[cfg(test)]
        if let Some(timeout) = self.run_timeout_override {
            return timeout;
        }
        ExecutionGate::run_budget_for(profile)
    }

    fn page_budget(&self, profile: LintProfile) -> std::time::Duration {
        #[cfg(test)]
        if let Some(timeout) = self.page_timeout_override {
            return timeout;
        }
        ExecutionGate::page_budget_for(profile)
    }

    fn page_budget_exceeded(&self, profile: LintProfile, elapsed: std::time::Duration) -> bool {
        self.gate.check_for(profile, elapsed).is_err() || elapsed > self.page_budget(profile)
    }

    fn run_budget_exceeded(&self, profile: LintProfile) -> bool {
        let elapsed = self.clock.elapsed();
        self.gate.check_run_for(profile, elapsed).is_err() || elapsed > self.run_budget(profile)
    }

    async fn run_groups(
        &self,
        context: &LintContext<'_, '_>,
        config: &EffectiveLintConfig,
        page_started: std::time::Duration,
    ) -> Result<
        (
            Vec<LintCheckResult>,
            std::time::Duration,
            Option<LintAgentWork>,
        ),
        LintRunError,
    > {
        #[cfg(test)]
        if let Some(scenario) = self.scenario {
            let results = run_test_scenario(context, scenario).await?;
            return Ok((results, self.page_elapsed(page_started), None));
        }
        let mut results = super::pages::run(context, config.page_projection_enabled).await;
        let page_elapsed = self.page_elapsed(page_started);
        results.extend(super::identity::run(context).await);
        results.extend(super::memories::run(context, config.memory).await);
        results.extend(super::kg::run(context, config.kg).await);
        results.extend(super::operations::run(context, config.operations.clone()).await);
        results.extend(super::runtime::run(context, &config.runtime).await);
        results.extend(super::serving::run(context, config.serving).await);
        if context.profile() == LintProfile::Deep {
            results.extend(super::deep::run(context, &config.operations).await);
            let semantic = super::semantic::run(
                context,
                self.semantic_provider.as_deref(),
                &self.semantic_agent_request,
            )
            .await;
            results.extend(semantic.results);
            return Ok((results, page_elapsed, semantic.agent_work));
        }
        Ok((results, page_elapsed, None))
    }
}

#[cfg(test)]
async fn run_test_scenario(
    context: &LintContext<'_, '_>,
    scenario: TestScenario,
) -> Result<Vec<LintCheckResult>, LintRunError> {
    match scenario {
        TestScenario::BlockedGroup => std::future::pending().await,
        TestScenario::MixedQueryFailure => {
            record_zero_populations(context, context.profile())?;
            let failed = match context
                .snapshot()
                .query(
                    "SELECT CANARY_RAW_QUERY_ERROR_7f31 FROM pages",
                    libsql::params::Params::None,
                )
                .await
            {
                Err(_) => true,
                Ok(mut rows) => rows.next().await.is_err(),
            };
            if !failed {
                return Err(LintRunError::CatalogMismatch);
            }
            catalog_for_profile(context.profile())
                .enumerate()
                .map(|(index, entry)| match index {
                    0 => make_result(
                        entry.id,
                        LintOutcome::Pass,
                        LintSeverity::Info,
                        LintApplicability::Applicable,
                        LintPrecondition::Ready,
                        LintSummaryCode::CheckPassed,
                        None,
                        context.clock().duration_ms(),
                    ),
                    1 => make_result(
                        entry.id,
                        LintOutcome::Finding,
                        LintSeverity::Warning,
                        LintApplicability::Applicable,
                        LintPrecondition::Ready,
                        LintSummaryCode::FindingDetected,
                        Some(LintRecommendationCode::ReviewFinding),
                        context.clock().duration_ms(),
                    ),
                    2 => make_result(
                        entry.id,
                        LintOutcome::FailedToRun,
                        LintSeverity::Error,
                        LintApplicability::Applicable,
                        LintPrecondition::Ready,
                        LintSummaryCode::ExecutionFailed,
                        Some(LintRecommendationCode::InspectRuntime),
                        context.clock().duration_ms(),
                    ),
                    _ => make_result(
                        entry.id,
                        LintOutcome::Pass,
                        LintSeverity::Info,
                        LintApplicability::Applicable,
                        LintPrecondition::Ready,
                        LintSummaryCode::CheckPassed,
                        None,
                        context.clock().duration_ms(),
                    ),
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(LintRunError::from)
        }
        TestScenario::PageGroupTimeout => {
            let mut population = super::context::PopulationAccounting::new(101);
            for ordinal in 1..=100 {
                population.validate(ordinal, true);
            }
            if population.coverage().is_ok() {
                return Err(LintRunError::CatalogMismatch);
            }
            catalog_for_profile(context.profile())
                .map(|entry| {
                    context
                        .record_population(entry.id, PopulationBasis::Global, 101)
                        .map_err(|_| LintRunError::CatalogMismatch)?;
                    make_result_with_denominator(
                        entry.id,
                        LintOutcome::Pass,
                        LintSeverity::Info,
                        LintApplicability::Applicable,
                        LintPrecondition::Ready,
                        LintSummaryCode::CheckPassed,
                        None,
                        context.clock().duration_ms(),
                        101,
                    )
                    .map_err(LintRunError::from)
                })
                .collect()
        }
        TestScenario::ScopedDenominators => scoped_denominator_results(context),
    }
}

#[cfg(test)]
fn scoped_denominator_results(
    context: &LintContext<'_, '_>,
) -> Result<Vec<LintCheckResult>, LintRunError> {
    if !context.scope().filter().is_selected() {
        return Err(LintRunError::CatalogMismatch);
    }
    catalog_for_profile(context.profile())
        .map(|entry| {
            let selected_policy = matches!(
                entry.scope_policy,
                super::catalog::ScopePolicy::ScopedRows
                    | super::catalog::ScopePolicy::DbAnchoredProjection
            );
            let (basis, denominator) = if selected_policy {
                (PopulationBasis::SelectedScope, 3)
            } else {
                (PopulationBasis::Global, 9)
            };
            context
                .record_population(entry.id, basis, denominator)
                .map_err(|_| LintRunError::CatalogMismatch)?;
            make_result_with_denominator(
                entry.id,
                LintOutcome::Pass,
                LintSeverity::Info,
                LintApplicability::Applicable,
                LintPrecondition::Ready,
                LintSummaryCode::CheckPassed,
                None,
                context.clock().duration_ms(),
                denominator,
            )
            .map_err(LintRunError::from)
        })
        .collect()
}

pub fn configured_off_results(clock: LintClock) -> Vec<LintCheckResult> {
    catalog_for_profile(LintProfile::General)
        .map(|entry| {
            make_result(
                entry.id,
                LintOutcome::Pass,
                LintSeverity::Info,
                LintApplicability::ExpectedEmpty,
                LintPrecondition::ConfiguredOff,
                LintSummaryCode::ExpectedEmpty,
                None,
                clock.duration_ms(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("static configured-off lint results are valid")
}

pub(crate) fn configured_off_results_for_group(
    clock: &LintClock,
    group: LintCheckGroup,
) -> Vec<LintCheckResult> {
    configured_off_results(clock.clone())
        .into_iter()
        .filter(|result| catalog_entry(result.check_id()).is_some_and(|entry| entry.group == group))
        .collect()
}

pub(crate) fn prerequisite_results_for_group(
    clock: &LintClock,
    group: LintCheckGroup,
) -> Vec<LintCheckResult> {
    terminal_results_for_group(
        clock,
        group,
        LintOutcome::NotRunPrerequisite,
        LintPrecondition::MissingPrerequisite,
        LintSummaryCode::PrerequisiteUnavailable,
        LintRecommendationCode::RestorePrerequisite,
    )
}

fn failed_results_for_profile(clock: &LintClock, profile: LintProfile) -> Vec<LintCheckResult> {
    terminal_results(
        clock,
        profile,
        LintOutcome::FailedToRun,
        LintPrecondition::Ready,
        LintSummaryCode::ExecutionFailed,
        LintRecommendationCode::InspectRuntime,
    )
}

pub(crate) fn failed_results_for_group(
    clock: &LintClock,
    group: LintCheckGroup,
) -> Vec<LintCheckResult> {
    terminal_results_for_group(
        clock,
        group,
        LintOutcome::FailedToRun,
        LintPrecondition::Ready,
        LintSummaryCode::ExecutionFailed,
        LintRecommendationCode::InspectRuntime,
    )
}

fn failed_results_from_checks(
    clock: &LintClock,
    checks: &[LintCheckResult],
) -> Result<Vec<LintCheckResult>, LintRunError> {
    terminal_results_from_checks(
        clock,
        checks,
        LintOutcome::FailedToRun,
        LintPrecondition::Ready,
        LintSummaryCode::ExecutionFailed,
        LintRecommendationCode::InspectRuntime,
    )
}

fn inconsistent_selected_results(
    clock: &LintClock,
    checks: &[LintCheckResult],
    affected: impl Fn(&str) -> bool,
) -> Result<Vec<LintCheckResult>, LintRunError> {
    checks
        .iter()
        .map(|check| {
            if affected(check.check_id()) {
                make_result_with_denominator(
                    check.check_id(),
                    LintOutcome::InconsistentSnapshot,
                    LintSeverity::Error,
                    LintApplicability::Applicable,
                    LintPrecondition::SnapshotUnstable,
                    LintSummaryCode::SnapshotInconsistent,
                    Some(LintRecommendationCode::RerunAfterSnapshotStabilizes),
                    clock.duration_ms(),
                    check.coverage().denominator(),
                )
                .map_err(LintRunError::from)
            } else {
                Ok(check.clone())
            }
        })
        .collect()
}

fn failed_selected_results(
    clock: &LintClock,
    checks: &[LintCheckResult],
    affected: impl Fn(&str) -> bool,
) -> Result<Vec<LintCheckResult>, LintRunError> {
    checks
        .iter()
        .map(|check| {
            if affected(check.check_id()) {
                make_result_with_denominator(
                    check.check_id(),
                    LintOutcome::FailedToRun,
                    LintSeverity::Error,
                    LintApplicability::Applicable,
                    LintPrecondition::Ready,
                    LintSummaryCode::ExecutionFailed,
                    Some(LintRecommendationCode::InspectRuntime),
                    clock.duration_ms(),
                    check.coverage().denominator(),
                )
                .map_err(LintRunError::from)
            } else {
                Ok(check.clone())
            }
        })
        .collect()
}

fn terminal_results_from_checks(
    clock: &LintClock,
    checks: &[LintCheckResult],
    outcome: LintOutcome,
    precondition: LintPrecondition,
    summary: LintSummaryCode,
    recommendation: LintRecommendationCode,
) -> Result<Vec<LintCheckResult>, LintRunError> {
    checks
        .iter()
        .map(|check| {
            let denominator = checks
                .iter()
                .find(|candidate| candidate.check_id() == check.check_id())
                .ok_or(LintRunError::CatalogMismatch)?
                .coverage()
                .denominator();
            make_result_with_denominator(
                check.check_id(),
                outcome,
                LintSeverity::Error,
                LintApplicability::Applicable,
                precondition,
                summary,
                Some(recommendation),
                clock.duration_ms(),
                denominator,
            )
            .map_err(LintRunError::from)
        })
        .collect()
}

fn terminal_results(
    clock: &LintClock,
    profile: LintProfile,
    outcome: LintOutcome,
    precondition: LintPrecondition,
    summary: LintSummaryCode,
    recommendation: LintRecommendationCode,
) -> Vec<LintCheckResult> {
    catalog_for_profile(profile)
        .map(|entry| {
            make_result(
                entry.id,
                outcome,
                LintSeverity::Error,
                LintApplicability::Applicable,
                precondition,
                summary,
                Some(recommendation),
                clock.duration_ms(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("static terminal lint results are valid")
}

fn terminal_results_for_group(
    clock: &LintClock,
    group: LintCheckGroup,
    outcome: LintOutcome,
    precondition: LintPrecondition,
    summary: LintSummaryCode,
    recommendation: LintRecommendationCode,
) -> Vec<LintCheckResult> {
    let applicability = if outcome == LintOutcome::NotRunPrerequisite {
        LintApplicability::NotApplicable
    } else {
        LintApplicability::Applicable
    };
    catalog_group(group)
        .map(|entry| {
            make_result(
                entry.id,
                outcome,
                LintSeverity::Error,
                applicability,
                precondition,
                summary,
                Some(recommendation),
                clock.duration_ms(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("static group lint results are valid")
}

#[allow(clippy::too_many_arguments)]
fn make_result(
    check_id: &str,
    outcome: LintOutcome,
    severity: LintSeverity,
    applicability: LintApplicability,
    precondition: LintPrecondition,
    summary_code: LintSummaryCode,
    recommendation_code: Option<LintRecommendationCode>,
    duration_ms: u64,
) -> Result<LintCheckResult, LintContractError> {
    make_result_with_denominator(
        check_id,
        outcome,
        severity,
        applicability,
        precondition,
        summary_code,
        recommendation_code,
        duration_ms,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn make_result_with_denominator(
    check_id: &str,
    outcome: LintOutcome,
    severity: LintSeverity,
    applicability: LintApplicability,
    precondition: LintPrecondition,
    summary_code: LintSummaryCode,
    recommendation_code: Option<LintRecommendationCode>,
    duration_ms: u64,
    denominator: u64,
) -> Result<LintCheckResult, LintContractError> {
    let input = LintCheckResultInput {
        check_id: check_id.to_string(),
        outcome,
        severity,
        applicability,
        precondition,
        coverage: LintCoverage::new(
            LintValidationMethod::FullEnumeration,
            denominator,
            denominator,
            LINT_MAX_EVIDENCE_PER_CHECK,
            false,
            0,
        )?,
        metrics: Vec::new(),
        summary_code,
        recommendation_code,
        evidence: Vec::new(),
        duration_ms,
    };
    LintCheckResult::try_new_with_gate_effect(
        input,
        catalog_entry(check_id)
            .map(|entry| entry.gate_effect)
            .unwrap_or_default(),
    )
}

pub fn validate_catalog_results(checks: &mut [LintCheckResult]) -> Result<(), LintRunError> {
    validate_profile_catalog_results(LintProfile::General, checks)
}

fn validate_profile_catalog_results(
    profile: LintProfile,
    checks: &mut [LintCheckResult],
) -> Result<(), LintRunError> {
    checks.sort_by(|left, right| left.check_id().cmp(right.check_id()));
    let expected = catalog_for_profile(profile)
        .map(|entry| entry.id)
        .collect::<Vec<_>>();
    let actual = checks
        .iter()
        .map(LintCheckResult::check_id)
        .collect::<Vec<_>>();
    let unique = actual.iter().copied().collect::<BTreeSet<_>>();
    let gates_match = checks.iter().all(|check| {
        catalog_entry(check.check_id())
            .is_some_and(|entry| entry.gate_effect == check.gate_effect())
    });
    if actual != expected || unique.len() != actual.len() || !gates_match {
        return Err(LintRunError::CatalogMismatch);
    }
    Ok(())
}

fn validate_scope_policy(
    context: &LintContext<'_, '_>,
    checks: &[LintCheckResult],
) -> Result<(), LintRunError> {
    for check in checks {
        let entry =
            super::catalog::catalog_entry(check.check_id()).ok_or(LintRunError::CatalogMismatch)?;
        let receipt = context
            .population(check.check_id())
            .map_err(|_| LintRunError::CatalogMismatch)?
            .ok_or(LintRunError::CatalogMismatch)?;
        let selected = context.scope().filter().is_selected();
        let expected_basis = if selected
            && matches!(
                entry.scope_policy,
                super::catalog::ScopePolicy::ScopedRows
                    | super::catalog::ScopePolicy::DbAnchoredProjection
            ) {
            PopulationBasis::SelectedScope
        } else {
            PopulationBasis::Global
        };
        let global_policy = matches!(
            entry.scope_policy,
            super::catalog::ScopePolicy::GlobalAggregateOnly
                | super::catalog::ScopePolicy::GlobalOnly
        );
        if receipt.basis != expected_basis
            || receipt.denominator != check.coverage().denominator()
            || (selected && global_policy && !check.evidence().is_empty())
        {
            return Err(LintRunError::CatalogMismatch);
        }
    }
    Ok(())
}

fn record_zero_populations(
    context: &LintContext<'_, '_>,
    profile: LintProfile,
) -> Result<(), LintRunError> {
    let selected = context.scope().filter().is_selected();
    for entry in catalog_for_profile(profile) {
        let basis = if selected
            && matches!(
                entry.scope_policy,
                super::catalog::ScopePolicy::ScopedRows
                    | super::catalog::ScopePolicy::DbAnchoredProjection
            ) {
            PopulationBasis::SelectedScope
        } else {
            PopulationBasis::Global
        };
        context
            .record_population(entry.id, basis, 0)
            .map_err(|_| LintRunError::CatalogMismatch)?;
    }
    Ok(())
}

fn failed_results_for_context(
    context: &LintContext<'_, '_>,
    profile: LintProfile,
) -> Result<Vec<LintCheckResult>, LintRunError> {
    let selected = context.scope().filter().is_selected();
    catalog_for_profile(profile)
        .map(|entry| {
            let receipt = match context
                .population(entry.id)
                .map_err(|_| LintRunError::CatalogMismatch)?
            {
                Some(receipt) => receipt,
                None => {
                    let basis = if selected
                        && matches!(
                            entry.scope_policy,
                            super::catalog::ScopePolicy::ScopedRows
                                | super::catalog::ScopePolicy::DbAnchoredProjection
                        ) {
                        PopulationBasis::SelectedScope
                    } else {
                        PopulationBasis::Global
                    };
                    context
                        .record_population(entry.id, basis, 0)
                        .map_err(|_| LintRunError::CatalogMismatch)?;
                    super::context::PopulationReceipt {
                        basis,
                        denominator: 0,
                    }
                }
            };
            make_result_with_denominator(
                entry.id,
                LintOutcome::FailedToRun,
                LintSeverity::Error,
                LintApplicability::Applicable,
                LintPrecondition::Ready,
                LintSummaryCode::ExecutionFailed,
                Some(LintRecommendationCode::InspectRuntime),
                context.clock().duration_ms(),
                receipt.denominator,
            )
            .map_err(LintRunError::from)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    profile: LintProfile,
    scope: LintScope,
    db_receipt: SnapshotReceipt,
    page_before: [u8; 32],
    page_after: [u8; 32],
    effective_config: EffectiveLintConfig,
    checks: Vec<LintCheckResult>,
    agent_work: Option<LintAgentWork>,
) -> Result<LintReport, LintRunError> {
    LintReport::try_new_for_profile_with_agent_work(
        profile,
        scope,
        LintCapabilityContext::daemon_operator_endpoint(),
        receipts(db_receipt, page_before, page_after),
        effective_config.fingerprint(),
        LintProducerReceipt::new(effective_config.runtime.runtime_commit()),
        checks,
        agent_work,
    )
    .map_err(LintRunError::from)
}

fn receipts(
    db: SnapshotReceipt,
    page_before: [u8; 32],
    page_after: [u8; 32],
) -> LintSnapshotReceipts {
    LintSnapshotReceipts::new(
        LintDbSnapshotReceipt::new(
            LintDbSnapshotMode::TransactionalReadOnly,
            digest(db.analysis_receipt_digest().as_bytes()),
            Some(digest(db.post_run_receipt_digest().as_bytes())),
        ),
        LintPageSnapshotReceipt::new(
            LintPageSnapshotMode::BestEffort,
            digest(page_before),
            Some(digest(page_after)),
        ),
    )
}

fn digest(bytes: [u8; 32]) -> LintDigest {
    LintDigest::from_u64(u64::from_le_bytes(
        bytes[..8].try_into().expect("eight bytes"),
    ))
}

#[cfg(test)]
pub(super) fn synthetic_report(checks: Vec<LintCheckResult>) -> LintReport {
    let zero = LintDigest::from_u64(0);
    LintReport::try_new(
        LintScope::global(),
        LintCapabilityContext::daemon_operator_endpoint(),
        LintSnapshotReceipts::new(
            LintDbSnapshotReceipt::new(
                LintDbSnapshotMode::TransactionalReadOnly,
                zero.clone(),
                Some(zero.clone()),
            ),
            LintPageSnapshotReceipt::new(
                LintPageSnapshotMode::BestEffort,
                zero.clone(),
                Some(zero),
            ),
        ),
        LintConfigFingerprint::from_effective_config(&[]),
        LintProducerReceipt::new(None),
        checks,
    )
    .expect("synthetic report is valid")
}
