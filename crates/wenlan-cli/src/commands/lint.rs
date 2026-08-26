// SPDX-License-Identifier: Apache-2.0
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wenlan_types::lint::{
    LintActionCode, LintAgentSubmission, LintCheckGroup, LintEvidenceRef, LintGateEffect,
    LintMetricCode, LintMetricValue, LintOutcome, LintProfile, LintReasonCode,
    LintRecommendationCode, LintReport, LintSummaryCode,
};

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
                print!("{}", render_human(&report));
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

fn render_human(report: &LintReport) -> String {
    let totals = report.totals();
    let mut groups = [(0_u32, 0_u32, 0_u32); 7];
    for check in report.checks() {
        let Some(group) = LintCheckGroup::for_check_id(check.check_id()) else {
            continue;
        };
        let counts = &mut groups[group_index(group)];
        counts.0 += 1;
        match check.outcome() {
            LintOutcome::Pass => {}
            LintOutcome::Finding => counts.1 += 1,
            LintOutcome::NotRunPrerequisite
            | LintOutcome::InconsistentSnapshot
            | LintOutcome::FailedToRun => counts.2 += 1,
        }
    }
    let mut output = format!(
        "Lint: {} checks, {} passed, {} actionable findings, {} advisor{}, {} incomplete\nGroups:\n",
        totals.checks(),
        totals.passed(),
        totals.actionable_findings(),
        totals.advisory_findings(),
        if totals.advisory_findings() == 1 { "y" } else { "ies" },
        totals.incomplete()
    );
    for group in LintCheckGroup::ALL {
        let (checks, findings, incomplete) = groups[group_index(group)];
        if checks == 0 {
            continue;
        }
        output.push_str(&format!(
            "  {}: {checks} check{}, {findings} findings, {incomplete} incomplete\n",
            group.as_str(),
            if checks == 1 { "" } else { "s" }
        ));
    }
    if let Some(work) = report.agent_work() {
        output.push_str(&format!(
            "Agent work: {} bounded records; digest={}\n",
            work.records().len(),
            work.work_digest().as_str()
        ));
    }
    append_findings(&mut output, "Findings", report, LintGateEffect::Actionable);
    append_findings(&mut output, "Advisories", report, LintGateEffect::Advisory);
    // A check that passed but still carries advice is waiting on the owner, not
    // broken — e.g. a graph channel with no model source chosen. It is not a
    // finding and does not change the exit code, but the reader still has to see
    // the one thing they can do about it.
    output.push_str("Waiting on you");
    let waiting: Vec<_> = report
        .checks()
        .iter()
        .filter(|check| check.outcome() == LintOutcome::Pass && check.action_code().is_some())
        .collect();
    append_selected(&mut output, &waiting);
    output.push_str("Incomplete");
    let incomplete: Vec<_> = report
        .checks()
        .iter()
        .filter(|check| !matches!(check.outcome(), LintOutcome::Pass | LintOutcome::Finding))
        .collect();
    append_selected(&mut output, &incomplete);
    output
}

fn append_findings(
    output: &mut String,
    label: &str,
    report: &LintReport,
    gate_effect: LintGateEffect,
) {
    output.push_str(label);
    let selected: Vec<_> = report
        .checks()
        .iter()
        .filter(|check| {
            check.outcome() == LintOutcome::Finding && check.gate_effect() == gate_effect
        })
        .collect();
    append_selected(output, &selected);
}

fn append_selected(output: &mut String, checks: &[&wenlan_types::lint::LintCheckResult]) {
    if checks.is_empty() {
        output.push_str(": none\n");
        return;
    }
    output.push_str(&format!(" ({}):\n", checks.len()));
    for check in checks {
        let summary = summary_name(check.summary_code());
        let code_suffix = match (check.recommendation_code(), check.action_code()) {
            (Some(recommendation), _) => {
                format!("; recommendation: {}", recommendation_name(recommendation))
            }
            (None, Some(action)) => format!("; action: {}", action_name(action)),
            (None, None) => String::new(),
        };
        output.push_str(&format!("  {}: {summary}{code_suffix}\n", check.check_id()));
        // The codes above are stable identifiers for scripts. This line is the
        // same thing in plain English for the person reading the terminal.
        output.push_str(&format!(
            "    {}{}\n",
            check.summary_code().meaning(),
            match (check.recommendation_code(), check.action_code()) {
                (Some(recommendation), _) => format!(" {}", recommendation.action()),
                (None, Some(action)) => format!(" {}", action.action()),
                (None, None) => String::new(),
            }
        ));
        let affected = check.metrics().iter().find_map(|metric| {
            if metric.code() == LintMetricCode::AffectedRecords {
                match metric.value() {
                    LintMetricValue::Count { value } => Some(*value),
                    LintMetricValue::Boolean { .. } | LintMetricValue::CatalogCode { .. } => None,
                }
            } else {
                None
            }
        });
        let mut evidence_items = check
            .evidence()
            .iter()
            .take(8)
            .map(evidence_name)
            .collect::<Vec<_>>();
        if check.evidence().len() > evidence_items.len() {
            evidence_items.push(format!(
                "+{}_more",
                check.evidence().len() - evidence_items.len()
            ));
        }
        let evidence = evidence_items.join(",");
        output.push_str(&format!(
            "    affected={}; evaluated={}/{}; evidence={}; truncated={}\n",
            affected.map_or_else(|| "unknown".to_string(), |value| value.to_string()),
            check.coverage().evaluated(),
            check.coverage().denominator(),
            if evidence.is_empty() {
                "none"
            } else {
                &evidence
            },
            check.coverage().truncated(),
        ));
    }
}

const fn group_index(group: LintCheckGroup) -> usize {
    match group {
        LintCheckGroup::Identity => 0,
        LintCheckGroup::KnowledgeGraph => 1,
        LintCheckGroup::Memories => 2,
        LintCheckGroup::Operations => 3,
        LintCheckGroup::Pages => 4,
        LintCheckGroup::Runtime => 5,
        LintCheckGroup::Serving => 6,
    }
}

fn evidence_name(evidence: &LintEvidenceRef) -> String {
    match evidence {
        LintEvidenceRef::OpaqueId { opaque_id } => format!("opaque:{}", opaque_id.ordinal()),
        LintEvidenceRef::OpaqueDigest { opaque_digest } => {
            format!("opaque-digest:{}", opaque_digest.as_str())
        }
        LintEvidenceRef::ReasonCode { reason_code } => {
            format!("reason:{}", reason_name(*reason_code))
        }
        LintEvidenceRef::SafeRootRelativePath {
            safe_root_relative_path,
        } => format!("path:{safe_root_relative_path:?}"),
        LintEvidenceRef::SemanticFinding { finding } => format!(
            "semantic:{}:{:?}:{:?}:{}:{:?}",
            finding.candidate_id().ordinal(),
            finding.proposed_action(),
            finding.reason_code(),
            finding.confidence_basis_points(),
            finding.provider_route(),
        ),
    }
}

const fn reason_name(reason: LintReasonCode) -> &'static str {
    match reason {
        LintReasonCode::MissingArtifact => "missing_artifact",
        LintReasonCode::InvalidCatalogState => "invalid_catalog_state",
        LintReasonCode::ExpectedEmptySubstrate => "expected_empty_substrate",
        LintReasonCode::InvalidSourceConfiguration => "invalid_source_configuration",
        LintReasonCode::TerminalOperationFailure => "terminal_operation_failure",
        LintReasonCode::ExpiredRetry => "expired_retry",
        LintReasonCode::InvalidOperationState => "invalid_operation_state",
        LintReasonCode::DurableNoProgress => "durable_no_progress",
        LintReasonCode::SemanticProviderUnavailable => "semantic_provider_unavailable",
        LintReasonCode::InsufficientSemanticEvidence => "insufficient_semantic_evidence",
        LintReasonCode::SemanticExecutionFailure => "semantic_execution_failure",
        LintReasonCode::SemanticAgentAdjudicationRequired => "semantic_agent_adjudication_required",
        LintReasonCode::SemanticAgentWorkStale => "semantic_agent_work_stale",
        LintReasonCode::SemanticAgentSubmissionInvalid => "semantic_agent_submission_invalid",
        LintReasonCode::SemanticCandidateGenerationFailure => {
            "semantic_candidate_generation_failure"
        }
        LintReasonCode::SemanticPopulationIncomplete => "semantic_population_incomplete",
        LintReasonCode::SemanticDisagreementUnresolved => "semantic_disagreement_unresolved",
        LintReasonCode::SemanticSecondJudgeRequired => "semantic_second_judge_required",
    }
}

const fn recommendation_name(recommendation: LintRecommendationCode) -> &'static str {
    match recommendation {
        LintRecommendationCode::ReviewFinding => "review_finding",
        LintRecommendationCode::RestorePrerequisite => "restore_prerequisite",
        LintRecommendationCode::RerunAfterSnapshotStabilizes => "rerun_after_snapshot_stabilizes",
        LintRecommendationCode::InspectRuntime => "inspect_runtime",
    }
}

const fn action_name(action: LintActionCode) -> &'static str {
    match action {
        LintActionCode::ChooseModelSource => "choose_model_source",
    }
}

const fn summary_name(summary: LintSummaryCode) -> &'static str {
    match summary {
        LintSummaryCode::CheckPassed => "check_passed",
        LintSummaryCode::FindingDetected => "finding_detected",
        LintSummaryCode::PrerequisiteUnavailable => "prerequisite_unavailable",
        LintSummaryCode::SnapshotInconsistent => "snapshot_inconsistent",
        LintSummaryCode::ExecutionFailed => "execution_failed",
        LintSummaryCode::ExpectedEmpty => "expected_empty",
    }
}

#[cfg(test)]
mod tests {
    use super::{exit_code, render_human};
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

    #[test]
    fn the_model_source_checks_render_with_a_plain_sentence_and_no_findings() {
        let rendered = render_human(&fresh_store_report());

        assert!(rendered.contains("0 actionable findings"), "{rendered}");
        assert!(rendered.contains("Findings: none\n"), "{rendered}");
        assert!(rendered.contains("Waiting on you (2):"), "{rendered}");
        for check_id in ["kg.substrate_liveness", "serving.channel.graph"] {
            assert!(
                rendered.contains(&format!(
                    "  {check_id}: expected_empty; action: choose_model_source\n"
                )),
                "{rendered}"
            );
        }
        assert!(
            rendered.contains(
                "    There was nothing here to check, which is the expected state right now. \
                 Run `wenlan setup` and choose a model source so Wenlan can build this in the \
                 background.\n"
            ),
            "{rendered}"
        );
    }

    // Every rendered line carries the sentence, not just the new one -- the
    // codes stay for scripts, the English is for the person reading.
    #[test]
    fn a_rendered_finding_carries_both_the_code_and_the_plain_sentence() {
        let rendered = render_human(&report(vec![check(
            "pages.projection.identity",
            LintOutcome::Finding,
            LintSeverity::Error,
            LintApplicability::Applicable,
            Pre::Ready,
            LintSummaryCode::FindingDetected,
            Some(LintRecommendationCode::ReviewFinding),
        )]));

        assert!(
            rendered.contains(
                "  pages.projection.identity: finding_detected; recommendation: review_finding\n"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains(
                "    This check found records that do not match what Wenlan expects. \
                 Look at the listed records and fix or dismiss them.\n"
            ),
            "{rendered}"
        );
        assert!(rendered.contains("Waiting on you: none\n"), "{rendered}");
    }
}
