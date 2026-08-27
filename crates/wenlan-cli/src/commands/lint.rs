// SPDX-License-Identifier: Apache-2.0
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wenlan_types::lint::{LintAgentSubmission, LintProfile, LintReport};

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

const MAX_AGENT_SUBMISSION_BYTES: usize = 64 * 1024;

#[allow(clippy::too_many_arguments)]
pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    profile: Option<LintProfile>,
    space: Option<String>,
    external_egress: bool,
    agent_assist: bool,
    agent_submission: Option<PathBuf>,
) -> ExitCode {
    if external_egress && profile != Some(LintProfile::Deep) {
        eprintln!("wenlan lint: --allow-external requires --profile deep");
        return ExitCode::from(2);
    }
    if (agent_assist || agent_submission.is_some()) && profile != Some(LintProfile::Deep) {
        eprintln!(
            "wenlan lint: {} requires --profile deep",
            if agent_submission.is_some() {
                "--agent-submission"
            } else {
                "--agent-assist"
            }
        );
        return ExitCode::from(2);
    }
    let submission = match agent_submission {
        Some(path) => match read_agent_submission(&path) {
            Ok(submission) => Some(submission),
            Err(error) => {
                eprintln!(
                    "wenlan lint: reading agent submission {} failed: {error:#}",
                    path.display()
                );
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let report = match client
        .lint(
            profile,
            space,
            external_egress,
            agent_assist,
            submission.as_ref(),
        )
        .await
    {
        Ok(report) => report,
        Err(error) => {
            eprintln!("wenlan lint: {error:#}");
            return ExitCode::from(2);
        }
    };
    let code = exit_code(&report);
    let rendered = if quiet {
        Ok(())
    } else {
        match format {
            ResolvedFormat::Json => print_json(&report),
            ResolvedFormat::Table => {
                print!("{}", report.render_text());
                Ok(())
            }
        }
    };
    if let Err(error) = rendered {
        eprintln!("wenlan lint: rendering report failed: {error:#}");
        return ExitCode::from(2);
    }
    ExitCode::from(code)
}

fn read_agent_submission(path: &Path) -> anyhow::Result<LintAgentSubmission> {
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(u64::try_from(MAX_AGENT_SUBMISSION_BYTES + 1).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_AGENT_SUBMISSION_BYTES {
        anyhow::bail!("agent submission exceeds {MAX_AGENT_SUBMISSION_BYTES}-byte limit");
    }
    serde_json::from_slice(&bytes).map_err(Into::into)
}

pub const fn exit_code(report: &LintReport) -> u8 {
    if !report.complete() {
        2
    } else if report.totals().actionable_findings() > 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::exit_code;
    use wenlan_types::lint::{
        LintActionCode, LintApplicability, LintCapabilityContext, LintCheckResult,
        LintCheckResultInput, LintConfigFingerprint, LintCoverage, LintDbSnapshotMode,
        LintDbSnapshotReceipt, LintDigest, LintOutcome, LintPageSnapshotMode,
        LintPageSnapshotReceipt, LintPrecondition, LintPrecondition as Pre, LintProducerReceipt,
        LintProfile, LintRecommendationCode, LintReport, LintScope, LintSeverity,
        LintSnapshotReceipts, LintSummaryCode, LintValidationMethod, LINT_MAX_EVIDENCE_PER_CHECK,
    };

    /// The two checks a fresh store with no model source used to fail on, as the
    /// daemon now produces them: complete, passing, expected-empty, each asking
    /// the owner to pick a model source.
    fn fresh_store_report() -> LintReport {
        report(vec![
            model_source_check("kg.substrate_liveness"),
            model_source_check("serving.channel.graph"),
        ])
    }

    fn model_source_check(check_id: &str) -> LintCheckResult {
        check(
            check_id,
            LintOutcome::Pass,
            LintSeverity::Info,
            LintApplicability::ExpectedEmpty,
            Pre::ExpectedEmpty,
            LintSummaryCode::ExpectedEmpty,
            None,
        )
        .with_action_code(Some(LintActionCode::ChooseModelSource))
    }

    #[allow(clippy::too_many_arguments)]
    fn check(
        check_id: &str,
        outcome: LintOutcome,
        severity: LintSeverity,
        applicability: LintApplicability,
        precondition: LintPrecondition,
        summary_code: LintSummaryCode,
        recommendation_code: Option<LintRecommendationCode>,
    ) -> LintCheckResult {
        LintCheckResult::try_new(LintCheckResultInput {
            check_id: check_id.to_string(),
            outcome,
            severity,
            applicability,
            precondition,
            coverage: LintCoverage::new(
                LintValidationMethod::ExactAggregate,
                2,
                2,
                LINT_MAX_EVIDENCE_PER_CHECK,
                false,
                0,
            )
            .expect("valid synthetic coverage"),
            metrics: vec![],
            summary_code,
            recommendation_code,
            evidence: vec![],
            duration_ms: 1,
        })
        .expect("valid synthetic check")
    }

    fn report(checks: Vec<LintCheckResult>) -> LintReport {
        LintReport::try_new_for_profile(
            LintProfile::General,
            LintScope::global(),
            LintCapabilityContext::daemon_operator_endpoint(),
            LintSnapshotReceipts::new(
                LintDbSnapshotReceipt::new(
                    LintDbSnapshotMode::TransactionalReadOnly,
                    LintDigest::from_u64(1),
                    Some(LintDigest::from_u64(1)),
                ),
                LintPageSnapshotReceipt::new(
                    LintPageSnapshotMode::BestEffort,
                    LintDigest::from_u64(2),
                    Some(LintDigest::from_u64(2)),
                ),
            ),
            LintConfigFingerprint::from_effective_config(&[]),
            LintProducerReceipt::new(None),
            checks,
        )
        .expect("valid synthetic report")
    }

    // Tracker row 17: a first-hour install must not read as broken to a script.
    #[test]
    fn a_fresh_store_without_a_model_source_exits_zero() {
        let report = fresh_store_report();

        assert!(report.complete());
        assert_eq!(report.totals().actionable_findings(), 0);
        assert_eq!(exit_code(&report), 0);
    }

    #[test]
    fn exit_code_is_one_only_for_actionable_findings_and_two_for_incomplete() {
        let finding = report(vec![check(
            "pages.projection.identity",
            LintOutcome::Finding,
            LintSeverity::Error,
            LintApplicability::Applicable,
            Pre::Ready,
            LintSummaryCode::FindingDetected,
            Some(LintRecommendationCode::ReviewFinding),
        )]);
        assert_eq!(exit_code(&finding), 1);

        let incomplete = report(vec![check(
            "runtime.provider_inventory",
            LintOutcome::NotRunPrerequisite,
            LintSeverity::Error,
            LintApplicability::NotApplicable,
            Pre::MissingPrerequisite,
            LintSummaryCode::PrerequisiteUnavailable,
            Some(LintRecommendationCode::RestorePrerequisite),
        )]);
        assert_eq!(exit_code(&incomplete), 2);
    }
}
