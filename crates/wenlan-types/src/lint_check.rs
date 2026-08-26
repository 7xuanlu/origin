// SPDX-License-Identifier: Apache-2.0
use super::*;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LintCheckResult {
    check_id: String,
    pub(crate) outcome: LintOutcome,
    gate_effect: LintGateEffect,
    severity: LintSeverity,
    applicability: LintApplicability,
    precondition: LintPrecondition,
    coverage: LintCoverage,
    metrics: Vec<LintMetric>,
    summary_code: LintSummaryCode,
    recommendation_code: Option<LintRecommendationCode>,
    /// Additive since schema 5; see [`LintActionCode`] for why this is a field
    /// and not a fifth `recommendation_code` variant. Omitted from the wire
    /// entirely when absent, so a report that carries no action serializes
    /// exactly as it did before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    action_code: Option<LintActionCode>,
    evidence: Vec<LintEvidenceRef>,
    duration_ms: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LintCheckResultInput {
    pub check_id: String,
    pub outcome: LintOutcome,
    pub severity: LintSeverity,
    pub applicability: LintApplicability,
    pub precondition: LintPrecondition,
    pub coverage: LintCoverage,
    pub metrics: Vec<LintMetric>,
    pub summary_code: LintSummaryCode,
    pub recommendation_code: Option<LintRecommendationCode>,
    pub evidence: Vec<LintEvidenceRef>,
    pub duration_ms: u64,
}
#[derive(Deserialize)]
struct LintCheckResultWire {
    #[serde(flatten)]
    input: LintCheckResultInput,
    #[serde(default)]
    gate_effect: LintGateEffect,
    /// Read outside `LintCheckResultInput` on purpose: the input struct is the
    /// frozen schema-5 shape that older readers also parse, and keeping the
    /// additive field out of it means the two stay the same struct.
    #[serde(default)]
    action_code: Option<LintActionCode>,
}
impl LintCheckResult {
    pub fn try_new(input: LintCheckResultInput) -> Result<Self, LintContractError> {
        Self::try_new_with_gate_effect(input, LintGateEffect::Actionable)
    }

    pub fn try_new_with_gate_effect(
        input: LintCheckResultInput,
        gate_effect: LintGateEffect,
    ) -> Result<Self, LintContractError> {
        let LintCheckResultInput {
            check_id,
            outcome,
            severity,
            applicability,
            precondition,
            coverage,
            metrics,
            summary_code,
            recommendation_code,
            evidence,
            duration_ms,
        } = input;
        let legal_severity = match outcome {
            LintOutcome::Pass => severity == LintSeverity::Info,
            LintOutcome::Finding => match gate_effect {
                LintGateEffect::Actionable => {
                    severity == LintSeverity::Warning || severity == LintSeverity::Error
                }
                LintGateEffect::Advisory => severity == LintSeverity::Warning,
            },
            LintOutcome::NotRunPrerequisite
            | LintOutcome::InconsistentSnapshot
            | LintOutcome::FailedToRun => severity == LintSeverity::Error,
        };
        if !legal_severity {
            return Err(
                if outcome == LintOutcome::Finding && gate_effect == LintGateEffect::Advisory {
                    LintContractError::InvalidGateEffect
                } else {
                    LintContractError::InvalidOutcomeSeverity
                },
            );
        }
        let legal_context = match outcome {
            LintOutcome::Pass => match applicability {
                LintApplicability::Applicable | LintApplicability::Inventory => {
                    precondition == LintPrecondition::Ready
                }
                LintApplicability::ExpectedEmpty => {
                    precondition == LintPrecondition::ExpectedEmpty
                        || precondition == LintPrecondition::ConfiguredOff
                }
                LintApplicability::NotApplicable => false,
            },
            LintOutcome::Finding => {
                applicability == LintApplicability::Applicable
                    && precondition == LintPrecondition::Ready
            }
            LintOutcome::NotRunPrerequisite => {
                applicability == LintApplicability::NotApplicable
                    && precondition == LintPrecondition::MissingPrerequisite
            }
            LintOutcome::InconsistentSnapshot => {
                applicability == LintApplicability::Applicable
                    && precondition == LintPrecondition::SnapshotUnstable
            }
            LintOutcome::FailedToRun => {
                applicability == LintApplicability::Applicable
                    && precondition == LintPrecondition::Ready
            }
        };
        if !legal_context {
            return Err(LintContractError::InvalidApplicabilityPrecondition);
        }
        if evidence.len() > usize::from(LINT_MAX_EVIDENCE_PER_CHECK) {
            return Err(LintContractError::EvidenceLimitExceeded);
        }
        coverage.validate(evidence.len())?;
        if evidence
            .iter()
            .filter_map(LintEvidenceRef::opaque_id)
            .any(|opaque_id| opaque_id.ordinal() > coverage.authorized_denominator())
        {
            return Err(LintContractError::EvidenceOutsideAuthorizedDenominator);
        }
        Ok(Self {
            check_id,
            outcome,
            gate_effect,
            severity,
            applicability,
            precondition,
            coverage,
            metrics,
            summary_code,
            recommendation_code,
            action_code: None,
            evidence,
            duration_ms,
        })
    }

    /// Attach an owner-facing action to an already legal result.
    ///
    /// Separate from `try_new` so the ~20 existing `LintCheckResultInput`
    /// literals in the tree keep compiling unchanged; only a producer that has
    /// something to ask of the owner calls this.
    #[must_use]
    pub fn with_action_code(mut self, action_code: Option<LintActionCode>) -> Self {
        self.action_code = action_code;
        self
    }
    pub fn check_id(&self) -> &str {
        &self.check_id
    }
    pub const fn outcome(&self) -> LintOutcome {
        self.outcome
    }
    pub const fn gate_effect(&self) -> LintGateEffect {
        self.gate_effect
    }
    pub const fn severity(&self) -> LintSeverity {
        self.severity
    }
    pub const fn applicability(&self) -> LintApplicability {
        self.applicability
    }
    pub const fn precondition(&self) -> LintPrecondition {
        self.precondition
    }
    pub fn coverage(&self) -> &LintCoverage {
        &self.coverage
    }
    pub fn metrics(&self) -> &[LintMetric] {
        &self.metrics
    }
    pub const fn summary_code(&self) -> LintSummaryCode {
        self.summary_code
    }
    pub const fn recommendation_code(&self) -> Option<LintRecommendationCode> {
        self.recommendation_code
    }
    pub const fn action_code(&self) -> Option<LintActionCode> {
        self.action_code
    }
    pub fn evidence(&self) -> &[LintEvidenceRef] {
        &self.evidence
    }
    pub const fn duration_ms(&self) -> u64 {
        self.duration_ms
    }
}
impl<'de> Deserialize<'de> for LintCheckResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LintCheckResultWire::deserialize(deserializer)?;
        Self::try_new_with_gate_effect(wire.input, wire.gate_effect)
            .map(|result| result.with_action_code(wire.action_code))
            .map_err(D::Error::custom)
    }
}
