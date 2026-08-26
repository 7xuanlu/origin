use super::config::KgRunConfig;
use super::query::RowCheck;
use crate::lint::context::{LintContext, PopulationBasis, PopulationLedgerError};
use wenlan_types::lint::{
    LintActionCode, LintApplicability, LintCheckResult, LintCheckResultInput, LintContractError,
    LintCoverage, LintEvidenceRef, LintMetric, LintMetricCode, LintMetricStringCode,
    LintMetricValue, LintOpaqueId, LintOutcome, LintPrecondition, LintRecommendationCode,
    LintSeverity, LintSummaryCode, LintValidationMethod, LINT_MAX_EVIDENCE_PER_CHECK,
};

pub(super) struct Assessment {
    id: &'static str,
    population: u64,
    affected: u64,
    severity: LintSeverity,
    applicability: LintApplicability,
    precondition: LintPrecondition,
    metrics: Vec<LintMetric>,
    evidence_positions: Vec<usize>,
    method: LintValidationMethod,
    basis: PopulationBasis,
    /// Whether this assessment can point at individual records at all.
    ///
    /// `truncated` means "there was more evidence than we listed". An
    /// aggregate-only assessment never produces evidence positions, so its
    /// affected count can never be evidence that was cut off — reporting
    /// `truncated=true` there tells the reader to go look for a list that does
    /// not exist. Only an assessment that does emit positions can truncate.
    reports_evidence: bool,
    /// Advice to attach when the check passes. A finding always recommends
    /// reviewing it; a pass is silent unless it is waiting on the owner.
    /// Carried as `action_code`, not `recommendation_code` — see
    /// [`LintActionCode`] for why the wire keeps the two apart.
    passing_action: Option<LintActionCode>,
}

impl Assessment {
    pub(super) fn structural(id: &'static str, rows: RowCheck) -> Self {
        Self {
            id,
            population: rows.population,
            affected: rows.affected,
            severity: LintSeverity::Error,
            applicability: LintApplicability::Applicable,
            precondition: LintPrecondition::Ready,
            metrics: base_metrics(rows.population, rows.affected),
            evidence_positions: rows.evidence_positions,
            method: LintValidationMethod::FullEnumeration,
            basis: PopulationBasis::SelectedScope,
            reports_evidence: true,
            passing_action: None,
        }
    }

    pub(super) fn inventory(id: &'static str, population: u64, metrics: Vec<LintMetric>) -> Self {
        Self::inventory_with_basis(id, population, metrics, PopulationBasis::SelectedScope)
    }

    pub(super) fn global_inventory(
        id: &'static str,
        population: u64,
        metrics: Vec<LintMetric>,
    ) -> Self {
        Self::inventory_with_basis(id, population, metrics, PopulationBasis::Global)
    }

    fn inventory_with_basis(
        id: &'static str,
        population: u64,
        metrics: Vec<LintMetric>,
        basis: PopulationBasis,
    ) -> Self {
        Self {
            id,
            population,
            affected: 0,
            severity: LintSeverity::Info,
            applicability: LintApplicability::Inventory,
            precondition: LintPrecondition::Ready,
            metrics,
            evidence_positions: Vec::new(),
            method: LintValidationMethod::ExactAggregate,
            basis,
            reports_evidence: false,
            passing_action: None,
        }
    }

    pub(super) fn liveness(config: KgRunConfig, eligible: u64, linked: u64) -> Self {
        // An empty graph substrate is only a defect once the graph could
        // actually have been built. Two states make it the expected shape:
        // the channel is switched off, or nobody has chosen a model source yet,
        // so the enrichment that would populate it has never been authorized to
        // run. `serving_enabled` is checked first: an owner who turned the
        // channel off should not be told to go pick a model source for it.
        let waiting_on_model_source = config.serving_enabled && !config.model_source_configured;
        let live = config.serving_enabled && config.model_source_configured;
        let affected = if live && eligible > 0 && linked == 0 {
            eligible
        } else {
            0
        };
        Self {
            id: super::LIVENESS,
            population: eligible,
            affected,
            severity: LintSeverity::Warning,
            applicability: if live {
                LintApplicability::Applicable
            } else {
                LintApplicability::ExpectedEmpty
            },
            precondition: if live {
                LintPrecondition::Ready
            } else if waiting_on_model_source {
                LintPrecondition::ExpectedEmpty
            } else {
                LintPrecondition::ConfiguredOff
            },
            metrics: vec![
                metric(LintMetricCode::EligibleRecords, eligible),
                metric(LintMetricCode::ObservedRecords, linked),
                metric(LintMetricCode::AffectedRecords, affected),
                status_metric(LintMetricCode::KgServingStatus, config.serving_enabled),
                status_metric(LintMetricCode::KgSweepStatus, config.sweep_enabled),
                LintMetric::new(
                    LintMetricCode::KgProviderReadiness,
                    LintMetricValue::CatalogCode {
                        code: if config.provider_ready {
                            LintMetricStringCode::Ready
                        } else {
                            LintMetricStringCode::Missing
                        },
                    },
                ),
            ],
            evidence_positions: Vec::new(),
            method: LintValidationMethod::ExactAggregate,
            basis: PopulationBasis::SelectedScope,
            reports_evidence: false,
            passing_action: waiting_on_model_source.then_some(LintActionCode::ChooseModelSource),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum BuildError {
    #[error(transparent)]
    Contract(#[from] LintContractError),
    #[error(transparent)]
    Population(#[from] PopulationLedgerError),
}

pub(super) fn finish(
    context: &LintContext<'_, '_>,
    assessment: Assessment,
) -> Result<LintCheckResult, BuildError> {
    let basis = if context.scope().filter().is_selected() {
        assessment.basis
    } else {
        PopulationBasis::Global
    };
    let evidence = assessment
        .evidence_positions
        .iter()
        .filter_map(|position| LintOpaqueId::from_sorted_position(*position))
        .map(|opaque_id| LintEvidenceRef::OpaqueId { opaque_id })
        .collect::<Vec<_>>();
    let finding = assessment.affected > 0;
    let result = LintCheckResult::try_new(LintCheckResultInput {
        check_id: assessment.id.to_string(),
        outcome: if finding {
            LintOutcome::Finding
        } else {
            LintOutcome::Pass
        },
        severity: if finding {
            assessment.severity
        } else {
            LintSeverity::Info
        },
        applicability: assessment.applicability,
        precondition: assessment.precondition,
        coverage: LintCoverage::new(
            assessment.method,
            assessment.population,
            assessment.population,
            LINT_MAX_EVIDENCE_PER_CHECK,
            assessment.reports_evidence
                && assessment.affected > u64::try_from(evidence.len()).unwrap_or(u64::MAX),
            u64::try_from(evidence.len()).unwrap_or(u64::MAX),
        )?,
        metrics: assessment.metrics,
        summary_code: if finding {
            LintSummaryCode::FindingDetected
        } else if assessment.applicability == LintApplicability::ExpectedEmpty {
            LintSummaryCode::ExpectedEmpty
        } else {
            LintSummaryCode::CheckPassed
        },
        recommendation_code: finding.then_some(LintRecommendationCode::ReviewFinding),
        evidence,
        duration_ms: context.clock().duration_ms(),
    })?
    .with_action_code(assessment.passing_action);
    context.record_population(assessment.id, basis, assessment.population)?;
    Ok(result)
}

fn base_metrics(eligible: u64, affected: u64) -> Vec<LintMetric> {
    vec![
        metric(LintMetricCode::EligibleRecords, eligible),
        metric(LintMetricCode::AffectedRecords, affected),
    ]
}

fn metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}

pub(super) fn status_metric(code: LintMetricCode, enabled: bool) -> LintMetric {
    LintMetric::new(
        code,
        LintMetricValue::CatalogCode {
            code: if enabled {
                LintMetricStringCode::Enabled
            } else {
                LintMetricStringCode::Disabled
            },
        },
    )
}
