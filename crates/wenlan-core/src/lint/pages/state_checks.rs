mod assessment;
mod identity;
mod version;

use self::assessment::{Assessment, Level};
pub(crate) use self::identity::projection_target_is_exclusive_page_markdown;
use self::identity::{evaluate_identity, evaluate_state};
pub(crate) use self::identity::{
    normalize_id, stale_projection_ownership, StaleProjectionOwnership,
};
use self::version::evaluate_versions;
pub(crate) use self::version::{projection_version_mismatches, ProjectionVersionMismatch};
use super::fs::PageScan;
use super::state::RawStateKind;
use crate::lint::context::{LintContext, PopulationBasis, ScopeFilter};
use std::collections::BTreeSet;
use wenlan_types::lint::{
    LintApplicability, LintCheckResult, LintCheckResultInput, LintCoverage, LintOutcome,
    LintPrecondition, LintRecommendationCode, LintSeverity, LintSummaryCode, LintValidationMethod,
    LINT_MAX_EVIDENCE_PER_CHECK,
};

pub(crate) const STATE_CONTRACT_ID: &str = "pages.projection.state_contract";
pub(crate) const IDENTITY_ID: &str = "pages.projection.identity";
pub(crate) const VERSION_ALIGNMENT_ID: &str = "pages.projection.version_alignment";

#[derive(Debug, Clone)]
pub(crate) struct DbPage {
    pub(crate) id: String,
    pub(crate) status: String,
    pub(crate) version: i64,
}

pub(crate) async fn run(context: &LintContext<'_, '_>) -> Vec<LintCheckResult> {
    let pages = match load_pages(context).await {
        Ok(pages) => pages,
        Err(()) => return failed_results(context),
    };
    let selected = context.scope().filter().is_selected();
    let scan = context
        .page_scan()
        .expect("Page checks require a Page scan");
    let selected_ids = pages
        .iter()
        .map(|page| page.id.as_str())
        .collect::<BTreeSet<_>>();
    evaluate_all(scan, &pages, selected, &selected_ids)
        .into_iter()
        .map(|(check_id, assessment)| {
            let denominator = assessment.population();
            let basis = if selected {
                PopulationBasis::SelectedScope
            } else {
                PopulationBasis::Global
            };
            if context
                .record_population(check_id, basis, denominator)
                .is_err()
            {
                return failed_result(check_id, context.clock().duration_ms());
            }
            assessment.result(check_id, context.clock().duration_ms())
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|_| failed_results(context))
}

fn evaluate_all(
    scan: &PageScan,
    pages: &[DbPage],
    selected: bool,
    selected_ids: &BTreeSet<&str>,
) -> [(&'static str, Assessment); 3] {
    if matches!(scan.raw_state.kind, RawStateKind::FutureU32(_)) {
        return [
            (STATE_CONTRACT_ID, prerequisite_assessment()),
            (IDENTITY_ID, prerequisite_assessment()),
            (VERSION_ALIGNMENT_ID, prerequisite_assessment()),
        ];
    }
    [
        (
            STATE_CONTRACT_ID,
            evaluate_state(scan, selected.then_some(selected_ids)),
        ),
        (IDENTITY_ID, evaluate_identity(scan, pages, selected)),
        (
            VERSION_ALIGNMENT_ID,
            evaluate_versions(scan, pages, selected),
        ),
    ]
}

fn prerequisite_assessment() -> Assessment {
    let mut assessment = Assessment::default();
    assessment.push(Level::Prerequisite, false);
    assessment
}

async fn load_pages(context: &LintContext<'_, '_>) -> Result<Vec<DbPage>, ()> {
    let (sql, params) = match context.scope().filter() {
        ScopeFilter::Global => (
            "SELECT id, status, version FROM pages ORDER BY id",
            libsql::params::Params::None,
        ),
        ScopeFilter::Registered(workspace) => (
            "SELECT id, status, version FROM pages WHERE workspace = ?1 ORDER BY id",
            libsql::params::Params::Positional(vec![libsql::Value::Text(workspace.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            "SELECT id, status, version FROM pages WHERE workspace = '00000000-0000-4000-8000-000000000001' ORDER BY id",
            libsql::params::Params::None,
        ),
    };
    let mut rows = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let mut pages = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        pages.push(DbPage {
            id: row.get(0).map_err(|_| ())?,
            status: row.get(1).map_err(|_| ())?,
            version: row.get(2).map_err(|_| ())?,
        });
    }
    Ok(pages)
}

fn failed_results(context: &LintContext<'_, '_>) -> Vec<LintCheckResult> {
    [STATE_CONTRACT_ID, IDENTITY_ID, VERSION_ALIGNMENT_ID]
        .into_iter()
        .map(|check_id| {
            let _ = context.record_population(
                check_id,
                if context.scope().filter().is_selected() {
                    PopulationBasis::SelectedScope
                } else {
                    PopulationBasis::Global
                },
                0,
            );
            failed_result(check_id, context.clock().duration_ms())
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("static failed state-check results are valid")
}

fn failed_result(
    check_id: &str,
    duration_ms: u64,
) -> Result<LintCheckResult, wenlan_types::lint::LintContractError> {
    LintCheckResult::try_new(LintCheckResultInput {
        check_id: check_id.to_string(),
        outcome: LintOutcome::FailedToRun,
        severity: LintSeverity::Error,
        applicability: LintApplicability::Applicable,
        precondition: LintPrecondition::Ready,
        coverage: LintCoverage::new(
            LintValidationMethod::FullEnumeration,
            0,
            0,
            LINT_MAX_EVIDENCE_PER_CHECK,
            false,
            0,
        )?,
        metrics: Vec::new(),
        summary_code: LintSummaryCode::ExecutionFailed,
        recommendation_code: Some(LintRecommendationCode::InspectRuntime),
        evidence: Vec::new(),
        duration_ms,
    })
}

#[cfg(test)]
#[path = "state_checks_test.rs"]
mod tests;
