// SPDX-License-Identifier: Apache-2.0
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wenlan_types::lint::{
    LintAgentSubmission, LintCheckGroup, LintEvidenceRef, LintGateEffect, LintMetricCode,
    LintMetricValue, LintOutcome, LintProfile, LintReasonCode, LintRecommendationCode, LintReport,
    LintSummaryCode,
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
        match check.recommendation_code() {
            Some(recommendation) => output.push_str(&format!(
                "  {}: {summary}; recommendation: {}\n",
                check.check_id(),
                recommendation_name(recommendation)
            )),
            None => output.push_str(&format!("  {}: {summary}\n", check.check_id())),
        }
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
