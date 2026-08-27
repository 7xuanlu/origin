// SPDX-License-Identifier: Apache-2.0
use super::agent::LintSemanticFinding;
use super::contract::{LintOpaqueDigest, LintOpaqueId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintMetricCode {
    EligibleRecords,
    ObservedRecords,
    AffectedRecords,
    PendingRecords,
    ReturnedEvidence,
    MemoryClassifiedHeads,
    MemoryEventDatedHeads,
    MemoryEpisodeHeads,
    MemoryFactVectorHeads,
    MemoryPageLinkedHeads,
    MemorySummaryLinkedHeads,
    MemoryReembedPendingHeads,
    MemoryFailedEnrichmentSteps,
    CitationNullPages,
    CitationEmptyPages,
    CitationNonemptyPages,
    CitationVerifiedOccurrences,
    CitationUnverifiedOccurrences,
    CitationSentenceOccurrences,
    CitationParagraphOccurrences,
    CitationMemoryOccurrences,
    CitationExternalFileOccurrences,
    CitationExternalUrlOccurrences,
    CitationAuthoredOccurrences,
    PageOrphanLabels,
    PageManifestPages,
    PageManifestReferences,
    PageSourceStubs,
    PageManifestDivergences,
    ProjectPurposeArtifacts,
    ProjectSchemaArtifacts,
    ProjectIndexArtifacts,
    ProjectLogArtifacts,
    ProjectOverviewArtifacts,
    ProjectArchiveRecords,
    ProjectOutboundLinks,
    ProjectInboundLinks,
    ProjectBrokenLinks,
    KgEntities,
    KgEntitiesConfirmed,
    KgEntitiesScoped,
    KgEntitiesUncategorized,
    KgRelations,
    KgObservations,
    KgMemoryEntityLinks,
    KgDuplicateEntityNames,
    KgHubEntities,
    KgSemanticSuspicions,
    KgServingStatus,
    KgSweepStatus,
    KgProviderReadiness,
    SourceConfigurations,
    SourceInvalidConfigurations,
    SourceTerminalFailures,
    SourceSyncCheckpoints,
    SourceConfigurationStatus,
    OperationActiveRetries,
    OperationExpiredRetries,
    OperationPending,
    OperationAwaitingReview,
    OperationTerminal,
    OperationTerminalFailures,
    OperationInvalidStates,
    OperationDurableNoProgress,
    OperationMissingProgressOracles,
    OperationAgeUnderHour,
    OperationAgeOneTo24Hours,
    OperationAgeOneTo7Days,
    OperationAgeSevenDaysOrMore,
    AccessTelemetryRows,
    AgentActivityTelemetryRows,
    UnattributedServingChannels,
    RerankerConfiguredPaths,
    RerankerRuntimeReadinessUnavailable,
    IdentityProfiles,
    IdentityAgents,
    IdentitySpaces,
    DecisionMemories,
    TaggedDocuments,
    PinnedMemories,
    StableMemories,
    SessionActivities,
    SessionCaptures,
    SessionSnapshots,
    BriefingCacheRows,
    NarrativeCacheRows,
    WorkingMemoryTelemetryRows,
    WorkingMemoryTelemetryUnavailable,
    WorkingMemoryNewestAgeSeconds,
    DeepDuplicateRecords,
    DeepConflictCandidates,
    DeepVocabularyDriftRecords,
    DeepLifecycleResidueRecords,
    DeepRetrievalSubstrateMissingRecords,
    DeepPageBodyMismatchRecords,
    SemanticEligibleRecords,
    SemanticCandidateRecords,
    SemanticPacketCandidates,
    SemanticJudgedRecords,
    SemanticUnresolvedDisagreements,
    SemanticModelCalls,
    SemanticProviderOnDevice,
    SemanticAgentSubmissions,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintMetricStringCode {
    Ready,
    Enabled,
    Disabled,
    Present,
    Missing,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LintMetricValue {
    Count { value: u64 },
    Boolean { value: bool },
    CatalogCode { code: LintMetricStringCode },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintMetric {
    code: LintMetricCode,
    value: LintMetricValue,
}
impl LintMetric {
    pub fn new(code: LintMetricCode, value: LintMetricValue) -> Self {
        Self { code, value }
    }
    pub const fn code(&self) -> LintMetricCode {
        self.code
    }
    pub fn value(&self) -> &LintMetricValue {
        &self.value
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintSummaryCode {
    CheckPassed,
    FindingDetected,
    PrerequisiteUnavailable,
    SnapshotInconsistent,
    ExecutionFailed,
    ExpectedEmpty,
}
impl LintSummaryCode {
    /// One plain sentence saying what this outcome means for the reader.
    ///
    /// Every surface that shows a check line pairs this with
    /// [`LintRecommendationCode::action`] so the reader never has to decode a
    /// snake_case code. The table lives here, beside the codes, because the
    /// CLI and the MCP tool both render it; `wenlan-types` stays dependency
    /// free, so these are plain `&'static str`.
    pub const fn meaning(self) -> &'static str {
        match self {
            Self::CheckPassed => "This check looked at everything it covers and found nothing wrong.",
            Self::FindingDetected => {
                "This check found records that do not match what Wenlan expects."
            }
            Self::PrerequisiteUnavailable => {
                "This check could not run because something it depends on was missing."
            }
            Self::SnapshotInconsistent => {
                "Your data changed while this check was reading it, so its answer is not trustworthy."
            }
            Self::ExecutionFailed => "This check hit an error and did not finish.",
            Self::ExpectedEmpty => {
                "There was nothing here to check, which is the expected state right now."
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintRecommendationCode {
    ReviewFinding,
    RestorePrerequisite,
    RerunAfterSnapshotStabilizes,
    InspectRuntime,
}
impl LintRecommendationCode {
    /// One plain sentence saying what to do next. See [`LintSummaryCode::meaning`].
    pub const fn action(self) -> &'static str {
        match self {
            Self::ReviewFinding => "Look at the listed records and fix or dismiss them.",
            Self::RestorePrerequisite => {
                "Restore the missing piece this check needs, then run `wenlan lint` again."
            }
            Self::RerunAfterSnapshotStabilizes => {
                "Run `wenlan lint` again once writes have settled."
            }
            Self::InspectRuntime => {
                "Check the daemon logs for the error behind this, then run `wenlan lint` again."
            }
        }
    }
}
/// Something the owner can do about a check that is *not* a defect.
///
/// This is deliberately a separate field from `recommendation_code` rather than
/// a fifth variant of it. `recommendation_code` is part of the frozen
/// `LINT_REPORT_SCHEMA_VERSION` 5 wire shape, and a patch-ahead daemon is
/// explicitly compatible with an older CLI or MCP client
/// (`wenlan-mcp/src/version_check.rs`). Serde rejects an unknown enum variant,
/// so teaching the daemon to emit a fifth `recommendation_code` would make an
/// old reader fail to parse the whole report; an added optional *field* is
/// simply ignored by that same reader, because `LintCheckResultInput` does not
/// deny unknown fields. Additive field now, enum variant at the next schema
/// bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintActionCode {
    /// The check needs a model source and none is chosen yet. Pairs with a
    /// passing, expected-empty verdict: nothing is broken, the owner simply has
    /// not picked where inference runs.
    ChooseModelSource,
}
impl LintActionCode {
    /// One plain sentence saying what to do next. See [`LintSummaryCode::meaning`].
    pub const fn action(self) -> &'static str {
        match self {
            Self::ChooseModelSource => {
                "Run `wenlan setup` and choose a model source so Wenlan can build this in the background."
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintReasonCode {
    MissingArtifact,
    InvalidCatalogState,
    ExpectedEmptySubstrate,
    InvalidSourceConfiguration,
    TerminalOperationFailure,
    ExpiredRetry,
    InvalidOperationState,
    DurableNoProgress,
    SemanticProviderUnavailable,
    InsufficientSemanticEvidence,
    SemanticExecutionFailure,
    SemanticAgentAdjudicationRequired,
    SemanticAgentWorkStale,
    SemanticAgentSubmissionInvalid,
    SemanticCandidateGenerationFailure,
    SemanticPopulationIncomplete,
    SemanticDisagreementUnresolved,
    SemanticSecondJudgeRequired,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LintSafeRootRelativePath {
    #[serde(rename = "pages")]
    PagesRoot,
    #[serde(rename = "pages/.wenlan/state.json")]
    PagesState,
    #[serde(rename = "pages/.wenlan/manifest.json")]
    PagesManifest,
    #[serde(rename = "pages/.wenlan/stubs")]
    PagesStubs,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LintEvidenceRef {
    OpaqueId {
        opaque_id: LintOpaqueId,
    },
    OpaqueDigest {
        opaque_digest: LintOpaqueDigest,
    },
    ReasonCode {
        reason_code: LintReasonCode,
    },
    SafeRootRelativePath {
        safe_root_relative_path: LintSafeRootRelativePath,
    },
    SemanticFinding {
        finding: LintSemanticFinding,
    },
}
impl LintEvidenceRef {
    pub(crate) const fn opaque_id(&self) -> Option<LintOpaqueId> {
        match self {
            Self::OpaqueId { opaque_id } => Some(*opaque_id),
            Self::SemanticFinding { finding } => Some(finding.candidate_id()),
            Self::OpaqueDigest { .. }
            | Self::ReasonCode { .. }
            | Self::SafeRootRelativePath { .. } => None,
        }
    }
}
