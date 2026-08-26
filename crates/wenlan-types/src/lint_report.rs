// SPDX-License-Identifier: Apache-2.0
use super::*;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintTotals {
    checks: u32,
    passed: u32,
    findings: u32,
    actionable_findings: u32,
    advisory_findings: u32,
    incomplete: u32,
}
impl LintTotals {
    fn from_checks(checks: &[LintCheckResult]) -> Result<Self, LintContractError> {
        let checks_count =
            u32::try_from(checks.len()).map_err(|_| LintContractError::TooManyChecks)?;
        let mut totals = Self {
            checks: checks_count,
            passed: 0,
            findings: 0,
            actionable_findings: 0,
            advisory_findings: 0,
            incomplete: 0,
        };
        for check in checks {
            match check.outcome {
                LintOutcome::Pass => totals.passed += 1,
                LintOutcome::Finding => {
                    totals.findings += 1;
                    match check.gate_effect() {
                        LintGateEffect::Actionable => totals.actionable_findings += 1,
                        LintGateEffect::Advisory => totals.advisory_findings += 1,
                    }
                }
                LintOutcome::NotRunPrerequisite
                | LintOutcome::InconsistentSnapshot
                | LintOutcome::FailedToRun => totals.incomplete += 1,
            }
        }
        Ok(totals)
    }
    pub const fn checks(&self) -> u32 {
        self.checks
    }
    pub const fn findings(&self) -> u32 {
        self.findings
    }
    pub const fn actionable_findings(&self) -> u32 {
        self.actionable_findings
    }
    pub const fn advisory_findings(&self) -> u32 {
        self.advisory_findings
    }
    pub const fn passed(&self) -> u32 {
        self.passed
    }
    pub const fn incomplete(&self) -> u32 {
        self.incomplete
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LintReport {
    report_schema_version: u16,
    check_catalog_version: u16,
    profile: LintProfile,
    scope: LintScope,
    capability_context: LintCapabilityContext,
    snapshots: LintSnapshotReceipts,
    config_fingerprint: LintConfigFingerprint,
    producer_receipt: LintProducerReceipt,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_work: Option<LintAgentWork>,
    checks: Vec<LintCheckResult>,
    totals: LintTotals,
    complete: bool,
}
#[derive(Deserialize)]
struct LintReportWire {
    report_schema_version: u16,
    check_catalog_version: u16,
    profile: LintProfile,
    scope: LintScope,
    capability_context: LintCapabilityContext,
    snapshots: LintSnapshotReceipts,
    config_fingerprint: LintConfigFingerprint,
    producer_receipt: LintProducerReceipt,
    #[serde(default)]
    agent_work: Option<LintAgentWork>,
    checks: Vec<LintCheckResult>,
    totals: LintTotals,
    complete: bool,
}
impl LintReport {
    pub fn try_new(
        scope: LintScope,
        capability_context: LintCapabilityContext,
        snapshots: LintSnapshotReceipts,
        config_fingerprint: LintConfigFingerprint,
        producer_receipt: LintProducerReceipt,
        checks: Vec<LintCheckResult>,
    ) -> Result<Self, LintContractError> {
        Self::try_new_for_profile(
            LintProfile::General,
            scope,
            capability_context,
            snapshots,
            config_fingerprint,
            producer_receipt,
            checks,
        )
    }

    pub fn try_new_for_profile(
        profile: LintProfile,
        scope: LintScope,
        capability_context: LintCapabilityContext,
        snapshots: LintSnapshotReceipts,
        config_fingerprint: LintConfigFingerprint,
        producer_receipt: LintProducerReceipt,
        checks: Vec<LintCheckResult>,
    ) -> Result<Self, LintContractError> {
        Self::try_new_for_profile_with_agent_work(
            profile,
            scope,
            capability_context,
            snapshots,
            config_fingerprint,
            producer_receipt,
            checks,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_for_profile_with_agent_work(
        profile: LintProfile,
        scope: LintScope,
        capability_context: LintCapabilityContext,
        snapshots: LintSnapshotReceipts,
        config_fingerprint: LintConfigFingerprint,
        producer_receipt: LintProducerReceipt,
        mut checks: Vec<LintCheckResult>,
        agent_work: Option<LintAgentWork>,
    ) -> Result<Self, LintContractError> {
        if profile == LintProfile::General && agent_work.is_some() {
            return Err(LintContractError::InvalidAgentWork);
        }
        checks.sort_by(|left, right| left.check_id().cmp(right.check_id()));
        let totals = LintTotals::from_checks(&checks)?;
        let complete = checks.iter().all(|check| check.outcome.is_complete());
        Ok(Self {
            report_schema_version: LINT_REPORT_SCHEMA_VERSION,
            check_catalog_version: LINT_CHECK_CATALOG_VERSION,
            profile,
            scope,
            capability_context,
            snapshots,
            config_fingerprint,
            producer_receipt,
            agent_work,
            checks,
            totals,
            complete,
        })
    }
    pub const fn complete(&self) -> bool {
        self.complete
    }
    pub const fn profile(&self) -> LintProfile {
        self.profile
    }
    pub const fn totals(&self) -> &LintTotals {
        &self.totals
    }
    pub fn checks(&self) -> &[LintCheckResult] {
        &self.checks
    }
    pub fn scope(&self) -> &LintScope {
        &self.scope
    }
    pub const fn capability_context(&self) -> LintCapabilityContext {
        self.capability_context
    }
    pub fn snapshots(&self) -> &LintSnapshotReceipts {
        &self.snapshots
    }
    pub fn config_fingerprint(&self) -> &LintConfigFingerprint {
        &self.config_fingerprint
    }
    pub fn producer_receipt(&self) -> &LintProducerReceipt {
        &self.producer_receipt
    }
    pub fn agent_work(&self) -> Option<&LintAgentWork> {
        self.agent_work.as_ref()
    }
}
impl<'de> Deserialize<'de> for LintReport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LintReportWire::deserialize(deserializer)?;
        if wire.report_schema_version != LINT_REPORT_SCHEMA_VERSION {
            return Err(D::Error::custom(LintContractError::UnsupportedReportSchema));
        }
        if wire.check_catalog_version != LINT_CHECK_CATALOG_VERSION {
            return Err(D::Error::custom(LintContractError::UnsupportedCheckCatalog));
        }
        let expected_checks = match wire.profile {
            LintProfile::General => LINT_GENERAL_CHECK_COUNT,
            LintProfile::Deep => LINT_DEEP_CHECK_COUNT,
        };
        let unique_ids = wire
            .checks
            .iter()
            .map(LintCheckResult::check_id)
            .collect::<BTreeSet<_>>();
        if wire.checks.len() != expected_checks || unique_ids.len() != wire.checks.len() {
            return Err(D::Error::custom(LintContractError::InvalidCatalogShape));
        }
        if wire.checks.iter().any(|check| {
            canonical_gate_effect(wire.profile, check.check_id()) != Some(check.gate_effect())
        }) {
            return Err(D::Error::custom(LintContractError::InvalidCatalogShape));
        }
        let report = Self::try_new_for_profile_with_agent_work(
            wire.profile,
            wire.scope,
            wire.capability_context,
            wire.snapshots,
            wire.config_fingerprint,
            wire.producer_receipt,
            wire.checks,
            wire.agent_work,
        )
        .map_err(D::Error::custom)?;
        if report.totals != wire.totals {
            return Err(D::Error::custom(LintContractError::InvalidTotals));
        }
        if report.complete != wire.complete {
            return Err(D::Error::custom(LintContractError::InvalidCompleteness));
        }
        Ok(report)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<LintProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
}

impl LintQuery {
    pub const fn new(profile: Option<LintProfile>, space: Option<String>) -> Self {
        Self { profile, space }
    }

    pub fn applied_profile(&self) -> LintProfile {
        self.profile.unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintRequestQuery {
    #[serde(flatten)]
    lint: LintQuery,
    #[serde(default, skip_serializing_if = "is_false")]
    external_egress: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    agent_assist: bool,
}

impl LintRequestQuery {
    pub const fn new(lint: LintQuery, external_egress: bool, agent_assist: bool) -> Self {
        Self {
            lint,
            external_egress,
            agent_assist,
        }
    }

    pub const fn lint(&self) -> &LintQuery {
        &self.lint
    }

    pub const fn external_egress(&self) -> bool {
        self.external_egress
    }

    pub const fn agent_assist(&self) -> bool {
        self.agent_assist
    }
}

const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintErrorResponse {
    error: String,
}

impl LintErrorResponse {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
        }
    }

    pub fn error(&self) -> &str {
        &self.error
    }
}

impl LintReport {
    /// The report as the text `wenlan lint` prints and the MCP `lint` tool
    /// returns: totals, per-group counts, then every finding, advisory and
    /// incomplete check with its recommendation and coverage. The typed
    /// report stays the contract; this is what a person or an agent reads.
    pub fn render_text(&self) -> String {
        let totals = self.totals();
        let mut groups = [(0_u32, 0_u32, 0_u32); 7];
        for check in self.checks() {
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
        if let Some(work) = self.agent_work() {
            output.push_str(&format!(
                "Agent work: {} bounded records; digest={}\n",
                work.records().len(),
                work.work_digest().as_str()
            ));
        }
        append_findings(&mut output, "Findings", self, LintGateEffect::Actionable);
        append_findings(&mut output, "Advisories", self, LintGateEffect::Advisory);
        // A check that passed but still carries advice is waiting on the owner, not
        // broken — e.g. a graph channel with no model source chosen. It is not a
        // finding and does not change the exit code, but the reader still has to see
        // the one thing they can do about it.
        output.push_str("Waiting on you");
        let waiting: Vec<_> = self
            .checks()
            .iter()
            .filter(|check| check.outcome() == LintOutcome::Pass && check.action_code().is_some())
            .collect();
        append_selected(&mut output, &waiting);
        output.push_str("Incomplete");
        let incomplete: Vec<_> = self
            .checks()
            .iter()
            .filter(|check| !matches!(check.outcome(), LintOutcome::Pass | LintOutcome::Finding))
            .collect();
        append_selected(&mut output, &incomplete);
        output
    }
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

fn append_selected(output: &mut String, checks: &[&LintCheckResult]) {
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
