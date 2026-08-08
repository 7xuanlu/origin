// SPDX-License-Identifier: Apache-2.0
use super::LintGateEffect;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};
use std::{fmt, num::NonZeroU64};

pub const LINT_REPORT_SCHEMA_VERSION: u16 = 5;
pub const LINT_CHECK_CATALOG_VERSION: u16 = 3;
pub const LINT_MAX_EVIDENCE_PER_CHECK: u16 = 100;
pub const LINT_GENERAL_CHECK_COUNT: usize = 56;
pub const LINT_DEEP_CHECK_COUNT: usize = 74;

pub(crate) const LINT_CANONICAL_CHECK_IDS: [&str; LINT_DEEP_CHECK_COUNT] = [
    "entities.alias_collision_inventory",
    "entities.partition_inventory",
    "entities.structural_integrity",
    "identity.cache_inventory",
    "identity.memory_state_integrity",
    "identity.registry_integrity",
    "identity.session_structure",
    "identity.tag_integrity",
    "kg.advisory_inventory",
    "kg.aggregate_inventory",
    "kg.semantic.entity_relations",
    "kg.semantic.memory_entity_links",
    "kg.substrate_liveness",
    "memories.derived.episode",
    "memories.derived.fact",
    "memories.derived.page_links",
    "memories.derived.summary",
    "memories.derived.temporal",
    "memories.duplicate_inventory",
    "memories.embedding_integrity",
    "memories.enrichment_failures",
    "memories.lifecycle_integrity",
    "memories.partition_inventory",
    "memories.retrieval_substrate_inventory",
    "memories.semantic.classification",
    "memories.semantic.contradiction",
    "memories.semantic.staleness",
    "memories.structured_conflict_inventory",
    "memories.supersession_integrity",
    "memory_entities.integrity",
    "observations.duplicate_inventory",
    "observations.integrity",
    "operations.document_queue",
    "operations.import_checkpoints",
    "operations.maintenance_backlogs",
    "operations.refinement_inventory",
    "operations.rejection_inventory",
    "operations.source_configuration",
    "operations.source_lifecycle_residue",
    "pages.archive_inventory",
    "pages.citations.partitions",
    "pages.db.partitions",
    "pages.duplicate_active_titles",
    "pages.duplicate_body_inventory",
    "pages.links.orphan_labels",
    "pages.project.artifact_inventory",
    "pages.projection.body_alignment",
    "pages.projection.identity",
    "pages.projection.manifest_inventory",
    "pages.projection.state_contract",
    "pages.projection.version_alignment",
    "pages.provenance.source_evidence_coverage",
    "pages.review_status_inventory",
    "pages.semantic.evidence_links",
    "pages.semantic.faithfulness",
    "pages.semantic.provenance_adequacy",
    "pages.source_page_integrity",
    "relations.integrity",
    "relations.vocabulary_integrity",
    "runtime.ingest_worker_liveness",
    "runtime.provider_inventory",
    "runtime.schema_contract",
    "runtime.search_index_contract",
    "runtime.status_parity",
    "serving.channel.episode",
    "serving.channel.fact",
    "serving.channel.graph",
    "serving.channel.page",
    "serving.channel.summary",
    "serving.fact_scope_starvation",
    "serving.observability_inventory",
    "serving.reranker_fallback_inventory",
    "serving.route_scope_contracts",
    "serving.semantic.retrieval_quality",
];

const LINT_DEEP_ONLY_CHECK_IDS: [&str; LINT_DEEP_CHECK_COUNT - LINT_GENERAL_CHECK_COUNT] = [
    "entities.alias_collision_inventory",
    "kg.semantic.entity_relations",
    "kg.semantic.memory_entity_links",
    "memories.duplicate_inventory",
    "memories.retrieval_substrate_inventory",
    "memories.semantic.classification",
    "memories.semantic.contradiction",
    "memories.semantic.staleness",
    "memories.structured_conflict_inventory",
    "observations.duplicate_inventory",
    "operations.source_lifecycle_residue",
    "pages.duplicate_body_inventory",
    "pages.projection.body_alignment",
    "pages.semantic.evidence_links",
    "pages.semantic.faithfulness",
    "pages.semantic.provenance_adequacy",
    "relations.vocabulary_integrity",
    "serving.semantic.retrieval_quality",
];

const LINT_ADVISORY_CHECK_IDS: [&str; 15] = [
    "entities.alias_collision_inventory",
    "kg.semantic.entity_relations",
    "kg.semantic.memory_entity_links",
    "memories.duplicate_inventory",
    "memories.retrieval_substrate_inventory",
    "memories.semantic.classification",
    "memories.semantic.contradiction",
    "memories.semantic.staleness",
    "memories.structured_conflict_inventory",
    "observations.duplicate_inventory",
    "pages.duplicate_body_inventory",
    "pages.semantic.evidence_links",
    "pages.semantic.faithfulness",
    "pages.semantic.provenance_adequacy",
    "serving.semantic.retrieval_quality",
];

#[doc(hidden)]
pub fn canonical_check_ids(profile: LintProfile) -> impl Iterator<Item = &'static str> {
    LINT_CANONICAL_CHECK_IDS
        .into_iter()
        .filter(move |check_id| {
            profile == LintProfile::Deep
                || LINT_DEEP_ONLY_CHECK_IDS.binary_search(check_id).is_err()
        })
}

#[doc(hidden)]
pub fn canonical_gate_effect(profile: LintProfile, check_id: &str) -> Option<LintGateEffect> {
    LINT_CANONICAL_CHECK_IDS.binary_search(&check_id).ok()?;
    if profile == LintProfile::General && LINT_DEEP_ONLY_CHECK_IDS.binary_search(&check_id).is_ok()
    {
        return None;
    }
    Some(
        if LINT_ADVISORY_CHECK_IDS.binary_search(&check_id).is_ok() {
            LintGateEffect::Advisory
        } else {
            LintGateEffect::Actionable
        },
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintContractError {
    InvalidDigest,
    InvalidCommit,
    InvalidScope,
    InvalidOutcomeSeverity,
    InvalidGateEffect,
    InvalidApplicabilityPrecondition,
    InvalidCoverage,
    EvidenceLimitExceeded,
    EvidenceOutsideAuthorizedDenominator,
    UnsupportedReportSchema,
    UnsupportedCheckCatalog,
    InvalidTotals,
    InvalidCompleteness,
    InvalidCatalogShape,
    InvalidAgentRecord,
    InvalidAgentWork,
    InvalidAgentSubmission,
    TooManyChecks,
}

impl LintContractError {
    const fn code(self) -> &'static str {
        match self {
            Self::InvalidDigest => "invalid_lint_digest",
            Self::InvalidCommit => "invalid_lint_commit",
            Self::InvalidScope => "invalid_lint_scope",
            Self::InvalidOutcomeSeverity => "invalid_lint_outcome_severity",
            Self::InvalidGateEffect => "invalid_lint_gate_effect",
            Self::InvalidApplicabilityPrecondition => "invalid_lint_applicability_precondition",
            Self::InvalidCoverage => "invalid_lint_coverage",
            Self::EvidenceLimitExceeded => "lint_evidence_limit_exceeded",
            Self::EvidenceOutsideAuthorizedDenominator => "lint_evidence_outside_denominator",
            Self::UnsupportedReportSchema => "unsupported_lint_report_schema",
            Self::UnsupportedCheckCatalog => "unsupported_lint_check_catalog",
            Self::InvalidTotals => "invalid_lint_totals",
            Self::InvalidCompleteness => "invalid_lint_completeness",
            Self::InvalidCatalogShape => "invalid_lint_catalog_shape",
            Self::InvalidAgentRecord => "invalid_lint_agent_record",
            Self::InvalidAgentWork => "invalid_lint_agent_work",
            Self::InvalidAgentSubmission => "invalid_lint_agent_submission",
            Self::TooManyChecks => "too_many_lint_checks",
        }
    }
}
impl fmt::Display for LintContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}
impl std::error::Error for LintContractError {}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct LintDigest(String);
impl LintDigest {
    pub fn from_u64(value: u64) -> Self {
        Self(format!("{value:016x}"))
    }
    pub fn from_hex(value: &str) -> Result<Self, LintContractError> {
        if is_lower_hex(value, 16) {
            Ok(Self(value.to_string()))
        } else {
            Err(LintContractError::InvalidDigest)
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for LintDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_hex(&String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct LintOpaqueDigest(String);
impl LintOpaqueDigest {
    pub fn from_hex(value: &str) -> Result<Self, LintContractError> {
        if is_lower_hex(value, 64) {
            Ok(Self(value.to_string()))
        } else {
            Err(LintContractError::InvalidDigest)
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for LintOpaqueDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_hex(&String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct LintCommitReceipt(String);
impl LintCommitReceipt {
    pub fn new(value: &str) -> Result<Self, LintContractError> {
        if is_lower_hex(value, 40) {
            Ok(Self(value.to_string()))
        } else {
            Err(LintContractError::InvalidCommit)
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for LintCommitReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(&String::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintProfile {
    #[default]
    General,
    Deep,
}
impl fmt::Display for LintProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::General => "general",
            Self::Deep => "deep",
        })
    }
}
impl std::str::FromStr for LintProfile {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "general" => Ok(Self::General),
            "deep" => Ok(Self::Deep),
            _ => Err("expected `general` or `deep`"),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintScopeKind {
    Global,
    Registered,
    Uncategorized,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LintOpaqueId(NonZeroU64);
impl LintOpaqueId {
    pub fn from_sorted_position(position: usize) -> Option<Self> {
        position
            .checked_add(1)
            .and_then(|ordinal| u64::try_from(ordinal).ok())
            .and_then(NonZeroU64::new)
            .map(Self)
    }
    pub const fn ordinal(self) -> u64 {
        self.0.get()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LintScope {
    kind: LintScopeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    opaque_scope_ref: Option<LintOpaqueId>,
}
#[derive(Deserialize)]
struct LintScopeWire {
    kind: LintScopeKind,
    opaque_scope_ref: Option<LintOpaqueId>,
}
impl LintScope {
    pub const fn global() -> Self {
        Self {
            kind: LintScopeKind::Global,
            opaque_scope_ref: None,
        }
    }
    pub const fn registered(opaque_scope_ref: LintOpaqueId) -> Self {
        Self {
            kind: LintScopeKind::Registered,
            opaque_scope_ref: Some(opaque_scope_ref),
        }
    }
    pub const fn uncategorized() -> Self {
        Self {
            kind: LintScopeKind::Uncategorized,
            opaque_scope_ref: None,
        }
    }
    pub const fn kind(&self) -> LintScopeKind {
        self.kind
    }
    pub const fn opaque_scope_ref(&self) -> Option<LintOpaqueId> {
        self.opaque_scope_ref
    }
    const fn is_valid(&self) -> bool {
        match (self.kind, self.opaque_scope_ref) {
            (LintScopeKind::Global, None)
            | (LintScopeKind::Registered, Some(_))
            | (LintScopeKind::Uncategorized, None) => true,
            (LintScopeKind::Global, Some(_))
            | (LintScopeKind::Registered, None)
            | (LintScopeKind::Uncategorized, Some(_)) => false,
        }
    }
}
impl<'de> Deserialize<'de> for LintScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LintScopeWire::deserialize(deserializer)?;
        let scope = Self {
            kind: wire.kind,
            opaque_scope_ref: wire.opaque_scope_ref,
        };
        if scope.is_valid() {
            Ok(scope)
        } else {
            Err(D::Error::custom(LintContractError::InvalidScope))
        }
    }
}
