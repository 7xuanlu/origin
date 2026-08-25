use crate::client::{WenlanClient, WenlanError};
use crate::types::*;
use rmcp::{
    handler::server::router::tool::ToolRouter,
    handler::server::tool::ToolCallContext,
    handler::server::wrapper::Parameters,
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation, InitializeResult,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, Tool,
    },
    service::{NotificationContext, RequestContext, RoleServer},
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize an `Option<usize>` that also accepts stringified numbers (e.g. `"10"`).
/// MCP clients like Claude Desktop sometimes send numeric params as strings.
fn deserialize_optional_usize_lenient<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        Number(usize),
        Str(String),
    }

    match Option::<StringOrNumber>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrNumber::Number(n)) => Ok(Some(n)),
        Some(StringOrNumber::Str(s)) => s
            .parse::<usize>()
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

/// Return the effective space for a tool call: strict process pin, then an
/// explicit non-empty tool argument, then the overridable process fallback.
pub fn effective_space(inbound: &Option<String>) -> Option<String> {
    if let Some(locked) = crate::lock_state::locked_space() {
        if let Some(passed) = inbound.as_ref() {
            if passed != &locked {
                tracing::warn!(
                    inbound = %passed,
                    locked = %locked,
                    "model passed inbound space while WENLAN_SPACE is locked; using locked value"
                );
            }
        }
        Some(locked)
    } else if let Some(explicit) = inbound
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
    {
        Some(explicit)
    } else {
        crate::lock_state::default_space()
    }
}

/// Controls which operations are allowed based on transport.
#[derive(Clone, Debug, PartialEq)]
pub enum TransportMode {
    /// Local stdio — full access, all tools
    Stdio,
    /// Remote HTTP — block deletes, inject source_agent
    Http,
}

const LINT_AGENT_WORK_CACHE_CAPACITY: usize = 4;
const LINT_REPORT_CACHE_CAPACITY: usize = 4;

#[derive(Clone, Default)]
struct LintAgentWorkCache {
    entries: std::collections::VecDeque<wenlan_types::lint::LintAgentWork>,
}

impl LintAgentWorkCache {
    fn insert(&mut self, work: wenlan_types::lint::LintAgentWork) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|cached| cached.work_digest() == work.work_digest())
        {
            self.entries.remove(index);
        }
        self.entries.push_back(work);
        while self.entries.len() > LINT_AGENT_WORK_CACHE_CAPACITY {
            self.entries.pop_front();
        }
    }

    fn get(
        &self,
        digest: &wenlan_types::lint::LintDigest,
    ) -> Option<&wenlan_types::lint::LintAgentWork> {
        self.entries
            .iter()
            .find(|work| work.work_digest() == digest)
    }

    fn remove(&mut self, digest: &wenlan_types::lint::LintDigest) -> bool {
        let Some(index) = self
            .entries
            .iter()
            .position(|work| work.work_digest() == digest)
        else {
            return false;
        };
        self.entries.remove(index);
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LintReportCacheToken {
    lint_scope: wenlan_types::repair::RepairLintScope,
    generation: u64,
    profile: wenlan_types::lint::LintProfile,
    agent_submission: bool,
}

#[derive(Clone)]
struct CachedLintScopeState {
    lint_scope: wenlan_types::repair::RepairLintScope,
    generation: u64,
    general_report: Option<wenlan_types::lint::LintReport>,
    deep_report: Option<wenlan_types::lint::LintReport>,
}

#[derive(Clone, Default)]
struct LintReportCache {
    entries: std::collections::VecDeque<CachedLintScopeState>,
    next_generation: u64,
}

impl LintReportCache {
    fn issue_generation(&mut self) -> Result<u64, &'static str> {
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or("lint report cache generation exhausted")?;
        Ok(self.next_generation)
    }

    fn remove_scope(
        &mut self,
        lint_scope: &wenlan_types::repair::RepairLintScope,
    ) -> Option<CachedLintScopeState> {
        let index = self
            .entries
            .iter()
            .position(|entry| &entry.lint_scope == lint_scope)?;
        self.entries.remove(index)
    }

    fn push_scope(&mut self, cached: CachedLintScopeState) {
        self.entries.push_back(cached);
        while self.entries.len() > LINT_REPORT_CACHE_CAPACITY {
            self.entries.pop_front();
        }
    }

    #[cfg(test)]
    fn record_general(
        &mut self,
        lint_scope: wenlan_types::repair::RepairLintScope,
        report: wenlan_types::lint::LintReport,
    ) {
        let token = self
            .begin_lint_call(
                &lint_scope,
                wenlan_types::lint::LintProfile::General,
                false,
                false,
            )
            .expect("test cache generation")
            .expect("General cache token");
        self.record_from_lint_call(&token, report);
    }

    #[cfg(test)]
    fn record_final_deep(
        &mut self,
        lint_scope: wenlan_types::repair::RepairLintScope,
        report: wenlan_types::lint::LintReport,
    ) {
        let token = self
            .begin_lint_call(
                &lint_scope,
                wenlan_types::lint::LintProfile::Deep,
                true,
                true,
            )
            .expect("test cache generation")
            .expect("agent-assisted Deep cache token");
        self.record_from_lint_call(&token, report);
    }

    fn record_from_lint_call(
        &mut self,
        token: &LintReportCacheToken,
        report: wenlan_types::lint::LintReport,
    ) -> bool {
        let Some(cached) = self.entries.iter_mut().find(|cached| {
            cached.lint_scope == token.lint_scope && cached.generation == token.generation
        }) else {
            return false;
        };
        if report.profile() != token.profile
            || !token.lint_scope.matches_report_scope_kind(report.scope())
        {
            return false;
        }
        match token.profile {
            wenlan_types::lint::LintProfile::General => {
                cached.general_report = report.complete().then_some(report);
                cached.deep_report = None;
            }
            wenlan_types::lint::LintProfile::Deep => {
                cached.deep_report = cached
                    .general_report
                    .as_ref()
                    .filter(|general| {
                        report.complete()
                            && (report.agent_work().is_none() || token.agent_submission)
                            && report.scope() == general.scope()
                    })
                    .map(|_| report);
            }
        }
        true
    }

    fn begin_lint_call(
        &mut self,
        lint_scope: &wenlan_types::repair::RepairLintScope,
        profile: wenlan_types::lint::LintProfile,
        agent_assist: bool,
        agent_submission: bool,
    ) -> Result<Option<LintReportCacheToken>, &'static str> {
        if profile == wenlan_types::lint::LintProfile::Deep && !agent_assist {
            return Ok(None);
        }
        let generation = self.issue_generation()?;
        let mut cached = self
            .remove_scope(lint_scope)
            .unwrap_or_else(|| CachedLintScopeState {
                lint_scope: lint_scope.clone(),
                generation,
                general_report: None,
                deep_report: None,
            });
        cached.generation = generation;
        match profile {
            wenlan_types::lint::LintProfile::General => {
                cached.general_report = None;
                cached.deep_report = None;
            }
            wenlan_types::lint::LintProfile::Deep => {
                cached.deep_report = None;
            }
        }
        self.push_scope(cached);
        Ok(Some(LintReportCacheToken {
            lint_scope: lint_scope.clone(),
            generation,
            profile,
            agent_submission,
        }))
    }

    #[cfg(test)]
    fn reports_for_plan(
        &self,
        lint_scope: &wenlan_types::repair::RepairLintScope,
    ) -> Option<(
        wenlan_types::lint::LintReport,
        Option<wenlan_types::lint::LintReport>,
    )> {
        let cached = self
            .entries
            .iter()
            .find(|entry| &entry.lint_scope == lint_scope)?;
        Some((cached.general_report.clone()?, cached.deep_report.clone()))
    }

    fn take_reports_for_plan(
        &mut self,
        lint_scope: &wenlan_types::repair::RepairLintScope,
    ) -> Result<
        Option<(
            wenlan_types::lint::LintReport,
            Option<wenlan_types::lint::LintReport>,
        )>,
        &'static str,
    > {
        let generation = self.issue_generation()?;
        let mut cached = self
            .remove_scope(lint_scope)
            .unwrap_or_else(|| CachedLintScopeState {
                lint_scope: lint_scope.clone(),
                generation,
                general_report: None,
                deep_report: None,
            });
        cached.generation = generation;
        let reports = cached
            .general_report
            .take()
            .map(|general| (general, cached.deep_report.take()));
        self.push_scope(cached);
        Ok(reports)
    }
}

fn compact_lint_report_metadata(report: &wenlan_types::lint::LintReport) -> serde_json::Value {
    serde_json::json!({
        "profile": report.profile(),
        "scope": report.scope(),
        "complete": report.complete(),
        "check_count": report.checks().len(),
        "totals": report.totals(),
    })
}

fn add_repair_plan_source_reports(
    mut plan_summary: serde_json::Value,
    general_report: &wenlan_types::lint::LintReport,
    deep_report: Option<&wenlan_types::lint::LintReport>,
) -> Result<serde_json::Value, &'static str> {
    let object = plan_summary
        .as_object_mut()
        .ok_or("repair plan summary must serialize as an object")?;
    if object.contains_key("source_reports") {
        return Err("repair plan summary already contains source_reports");
    }
    object.insert(
        "source_reports".to_string(),
        serde_json::json!({
            "general": compact_lint_report_metadata(general_report),
            "deep": deep_report.map(compact_lint_report_metadata),
        }),
    );
    Ok(plan_summary)
}

#[derive(Clone)]
pub struct WenlanMcpServer {
    tool_router: ToolRouter<Self>,
    client: WenlanClient,
    transport: TransportMode,
    agent_name: String,
    /// Client name from MCP initialize handshake (e.g., "Claude Code", "Claude Desktop")
    client_name: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    agent_work_cache: std::sync::Arc<std::sync::Mutex<LintAgentWorkCache>>,
    lint_report_cache: std::sync::Arc<std::sync::Mutex<LintReportCache>>,
    user_id: Option<String>,
}

// ===== Parameter Structs =====

// --- Primary tool params ---

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CaptureParams {
    #[schemars(
        description = "The memory content. Write as a complete statement with context and reasoning, not shorthand. One idea per memory."
    )]
    pub content: String,
    #[schemars(description = wenlan_types::MEMORY_TYPE_CAPTURE_DESCRIPTION)]
    pub memory_type: Option<String>,
    #[schemars(
        description = "Topic scope (e.g. 'rust', 'work', 'health', 'origin'). Auto-detected if omitted."
    )]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[schemars(
        description = "Person, project, or tool name to anchor to (e.g. 'Alice', 'Wenlan', 'PostgreSQL'). Helps build the knowledge graph."
    )]
    pub entity: Option<String>,
    #[schemars(
        description = "0.0-1.0. Leave unset for auto-calculation based on type and trust level. Set low (0.3-0.5) for uncertain info, high (0.8-1.0) for user-stated facts."
    )]
    pub confidence: Option<f32>,
    #[schemars(
        description = "source_id of a memory this replaces. Use when correcting or updating an existing memory — get the ID from recall first. If the user has set your agent's trust level below full, a capture that sets supersedes is staged for review rather than applied: the response says gated: true, and the memory you meant to replace keeps answering searches until a human accepts your change. Tell the user the correction is waiting for their approval; do not repeat the capture."
    )]
    pub supersedes: Option<String>,
    #[schemars(
        description = "Pre-extracted structured fields as a JSON object. Auto-extracted by backend; only supply if you have high-quality structured data already."
    )]
    pub structured_fields: Option<serde_json::Map<String, serde_json::Value>>,
    #[schemars(
        description = "A question this memory answers, for search matching. Auto-generated by backend; only supply to override."
    )]
    pub retrieval_cue: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    #[schemars(
        description = "Natural language search. Be specific: 'Alice database preference' finds more than 'database stuff'."
    )]
    pub query: String,
    #[schemars(
        description = "Max memory results (distilled pages are returned separately), default 10, capped at 100. Use 3-5 for quick lookups, 10-20 for exploration."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    #[schemars(description = wenlan_types::MEMORY_TYPE_FILTER_DESCRIPTION)]
    pub memory_type: Option<String>,
    #[schemars(description = "Filter by topic scope.")]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[schemars(
        description = "Enable cross-encoder reranking. Slower (model inference) but higher retrieval quality. Off by default. Requires WENLAN_RERANKER_ENABLED=1 on the daemon; otherwise the daemon falls back to the plain hybrid ordering."
    )]
    #[serde(default)]
    pub rerank: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BriefParams {
    #[schemars(
        description = "Optional topic. Omit it to read only the complete Space Brief; provide it to append separately labeled related context from the same Space."
    )]
    pub topic: Option<String>,
    #[schemars(
        description = "Space (project) whose Brief to read. Uses the locked or configured default Space when omitted."
    )]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ContextParams {
    #[schemars(
        description = "Topic or conversation summary to focus context retrieval. Omit at session start for general orientation; provide when shifting topics."
    )]
    pub topic: Option<String>,
    #[schemars(
        description = "Max context chunks, default 20. Increase for complex topics, decrease for quick check-ins."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    #[schemars(
        description = "Scope context to a space (e.g. 'work', 'personal'). Auto-detected from conversation if omitted."
    )]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LintProfileParam {
    General,
    Deep,
}

impl From<LintProfileParam> for wenlan_types::lint::LintProfile {
    fn from(value: LintProfileParam) -> Self {
        match value {
            LintProfileParam::General => Self::General,
            LintProfileParam::Deep => Self::Deep,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LintParams {
    #[schemars(description = "Diagnostic depth. Defaults to general.")]
    pub profile: Option<LintProfileParam>,
    #[schemars(description = "Optional registered space, or uncategorized.")]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[schemars(
        description = "Prepare bounded high-recall semantic candidates for the calling agent. Deep only."
    )]
    #[serde(default)]
    pub agent_assist: bool,
    #[schemars(description = "Typed verdicts over a prior agent-assist work packet. Deep only.")]
    #[serde(default)]
    pub agent_submission: Option<LintAgentSubmissionParam>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetLintAgentWorkPageParams {
    pub work_digest: String,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RepairLintScopeParam {
    Global,
    Registered { space: String },
    Uncategorized,
}

impl TryFrom<RepairLintScopeParam> for wenlan_types::repair::RepairLintScope {
    type Error = wenlan_types::repair::RepairContractError;

    fn try_from(value: RepairLintScopeParam) -> Result<Self, Self::Error> {
        match value {
            RepairLintScopeParam::Global => Ok(Self::global()),
            RepairLintScopeParam::Registered { space } => Self::registered(space),
            RepairLintScopeParam::Uncategorized => Ok(Self::uncategorized()),
        }
    }
}

fn effective_repair_lint_scope(
    inbound: RepairLintScopeParam,
) -> Result<wenlan_types::repair::RepairLintScope, wenlan_types::repair::RepairContractError> {
    match crate::lock_state::locked_space() {
        Some(locked) => wenlan_types::repair::RepairLintScope::registered(locked),
        None => inbound.try_into(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PrepareLintRepairChoiceParam {
    ReclassifyMemory {
        #[schemars(with = "std::collections::BTreeMap<String, serde_json::Value>")]
        selected_finding: wenlan_types::lint::LintSemanticFinding,
        #[schemars(description = "Canonical target memory type.")]
        after_memory_type: String,
    },
    RenamePageTitle {
        review_id: String,
        page_id: String,
        before_title: String,
        after_title: String,
    },
    CompleteEntityExtraction {
        review_id: String,
        memory_id: String,
        entity_ids: Vec<String>,
    },
}

impl TryFrom<PrepareLintRepairChoiceParam> for wenlan_types::repair::RepairChoice {
    type Error = wenlan_types::repair::RepairContractError;

    fn try_from(value: PrepareLintRepairChoiceParam) -> Result<Self, Self::Error> {
        match value {
            PrepareLintRepairChoiceParam::ReclassifyMemory {
                selected_finding,
                after_memory_type,
            } => {
                let after_memory_type = after_memory_type
                    .parse::<wenlan_types::MemoryType>()
                    .map_err(|_| {
                        wenlan_types::repair::RepairContractError::InvalidPrepareRequest
                    })?;
                Self::reclassify_memory(selected_finding, after_memory_type)
            }
            PrepareLintRepairChoiceParam::RenamePageTitle {
                review_id,
                page_id,
                before_title,
                after_title,
            } => Self::rename_page_title(review_id, page_id, before_title, after_title),
            PrepareLintRepairChoiceParam::CompleteEntityExtraction {
                review_id,
                memory_id,
                entity_ids,
            } => Self::complete_entity_extraction(review_id, memory_id, entity_ids),
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareLintRepairParams {
    pub lint_scope: RepairLintScopeParam,
    #[schemars(with = "std::collections::BTreeMap<String, serde_json::Value>")]
    pub general_report: wenlan_types::lint::LintReport,
    #[serde(default)]
    #[schemars(with = "Option<std::collections::BTreeMap<String, serde_json::Value>>")]
    pub deep_report: Option<wenlan_types::lint::LintReport>,
    pub choice: PrepareLintRepairChoiceParam,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct PrepareLintRepairPlanParams {
    pub lint_scope: RepairLintScopeParam,
    #[serde(default)]
    #[schemars(skip)]
    pub general_report: Option<wenlan_types::lint::LintReport>,
    #[serde(default)]
    #[schemars(skip)]
    pub deep_report: Option<wenlan_types::lint::LintReport>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetLintRepairPlanEntriesParams {
    pub plan_id: String,
    pub plan_digest: String,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ApplyLintRepairParams {
    pub manifest_id: String,
    pub approved_manifest_digest: String,
    #[schemars(description = "Exact user approval: apply repair <manifest-id> <manifest-digest>")]
    pub approval: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct VerifyLintRepairParams {
    pub manifest_id: String,
    pub manifest_digest: String,
    pub apply_receipt_digest: String,
    #[schemars(with = "std::collections::BTreeMap<String, serde_json::Value>")]
    pub general_report: wenlan_types::lint::LintReport,
    #[schemars(
        with = "Option<std::collections::BTreeMap<String, serde_json::Value>>",
        description = "Post-repair Deep report for Deep-backed manifests. Omit for General-only deterministic manifests."
    )]
    #[serde(default)]
    pub deep_report: Option<wenlan_types::lint::LintReport>,
    #[schemars(description = "Exact next approved apply in the same displayed repair plan.")]
    #[serde(default)]
    pub next_apply: Option<ApplyLintRepairParams>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LintSemanticDecisionParam {
    Pass,
    Finding,
}

impl From<LintSemanticDecisionParam> for wenlan_types::lint::LintSemanticDecision {
    fn from(value: LintSemanticDecisionParam) -> Self {
        match value {
            LintSemanticDecisionParam::Pass => Self::Pass,
            LintSemanticDecisionParam::Finding => Self::Finding,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LintSemanticReasonParam {
    ClassificationMismatch,
    PotentialContradiction,
    PotentialStaleness,
    MentionWithoutLink,
    ExistingLinkMismatch,
    SharedContextWithoutRelation,
    ExistingRelationMismatch,
    PotentialUnfaithfulClaim,
    PotentialInadequateProvenance,
    ClaimOverlapWithoutEvidence,
    ExistingEvidenceMismatch,
    PotentialRetrievalMiss,
    DanglingOwner,
    TemporalEvolution,
    RelatedButNotEvidence,
}

impl From<LintSemanticReasonParam> for wenlan_types::lint::LintSemanticReasonCode {
    fn from(value: LintSemanticReasonParam) -> Self {
        use LintSemanticReasonParam as Param;
        match value {
            Param::ClassificationMismatch => Self::ClassificationMismatch,
            Param::PotentialContradiction => Self::PotentialContradiction,
            Param::PotentialStaleness => Self::PotentialStaleness,
            Param::MentionWithoutLink => Self::MentionWithoutLink,
            Param::ExistingLinkMismatch => Self::ExistingLinkMismatch,
            Param::SharedContextWithoutRelation => Self::SharedContextWithoutRelation,
            Param::ExistingRelationMismatch => Self::ExistingRelationMismatch,
            Param::PotentialUnfaithfulClaim => Self::PotentialUnfaithfulClaim,
            Param::PotentialInadequateProvenance => Self::PotentialInadequateProvenance,
            Param::ClaimOverlapWithoutEvidence => Self::ClaimOverlapWithoutEvidence,
            Param::ExistingEvidenceMismatch => Self::ExistingEvidenceMismatch,
            Param::PotentialRetrievalMiss => Self::PotentialRetrievalMiss,
            Param::DanglingOwner => Self::DanglingOwner,
            Param::TemporalEvolution => Self::TemporalEvolution,
            Param::RelatedButNotEvidence => Self::RelatedButNotEvidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct LintAgentVerdictParam {
    pub candidate_ref: u16,
    pub decision: LintSemanticDecisionParam,
    #[schemars(
        description = "Independent second judgment for high-risk removal or supersession candidates; omit for other candidates."
    )]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_decision: Option<LintSemanticDecisionParam>,
    pub reason_code: LintSemanticReasonParam,
    pub confidence_basis_points: u16,
    #[schemars(
        description = "Sorted unique subset of the candidate's authorized record refs (`evidence_refs` plus `counterevidence_refs`). Include only records the verdict actually treats as counterevidence; use [] when there are none. Do not mechanically copy every evidence ref."
    )]
    pub counterevidence_refs: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct LintAgentSubmissionParam {
    pub work_digest: String,
    pub verdicts: Vec<LintAgentVerdictParam>,
}

impl TryFrom<LintAgentSubmissionParam> for wenlan_types::lint::LintAgentSubmission {
    type Error = wenlan_types::lint::LintContractError;

    fn try_from(value: LintAgentSubmissionParam) -> Result<Self, Self::Error> {
        let verdicts = value
            .verdicts
            .into_iter()
            .map(|verdict| {
                wenlan_types::lint::LintAgentVerdict::try_new(
                    verdict.candidate_ref,
                    verdict.decision.into(),
                    verdict.second_decision.map(Into::into),
                    verdict.reason_code.into(),
                    verdict.confidence_basis_points,
                    verdict.counterevidence_refs,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(
            wenlan_types::lint::LintDigest::from_hex(&value.work_digest)?,
            verdicts,
        )
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    #[schemars(
        description = "The source_id of the memory to delete. Get this from recall results first."
    )]
    pub memory_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DistillParams {
    #[schemars(
        description = "Optional target scope. Accepts a page id (`page_*` or `concept_*`) to re-distill that single page, an entity name (e.g. `Wenlan`, `Alice`) to scope clustering to that entity, or a space value (e.g. `work`, `personal`) to scope to that space. Omit for a full pass over any clusters with new sources. The daemon resolves the string and falls back with a hint payload if nothing matches."
    )]
    #[serde(default, alias = "page_id")]
    pub target: Option<String>,

    #[schemars(
        description = "When true, clears the user_edited flag on the target page before recompile. Use for /distill rebuild <page> to explicitly wipe user prose and regenerate from sources. Only valid when target is a single page id; the daemon ignores it otherwise. Requires daemon LLM."
    )]
    #[serde(default)]
    pub force: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListPendingParams {
    #[schemars(
        description = "Max results, default 20. Increase for full audit, decrease for quick check-in."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    #[schemars(
        description = "Scope to a space (e.g. 'work', 'personal'). Auto-detected from conversation if omitted."
    )]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConfirmMemoryParams {
    #[schemars(
        description = "The source_id of the memory to confirm. Get this from list_pending or recall results."
    )]
    pub memory_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListRefinementsParams {
    #[schemars(
        description = "Optional action filter. One of: entity_merge, relation_conflict, detect_contradiction, suggest_entity, dedup_merge, page_merge, cross_space_discovery, page_keep_or_archive, lint_repair_review, vocab_promote."
    )]
    #[serde(default)]
    pub action: Option<String>,
    #[schemars(description = "Max number of proposals to return. Default 500, max 500.")]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RejectRefinementParams {
    #[schemars(description = "The review proposal id to dismiss.")]
    pub id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AcceptRefinementParams {
    #[schemars(description = "The review proposal id (e.g. \"merge_abc123_def456\").")]
    pub id: String,
    #[schemars(
        description = "Selected destination space for cross_space_discovery cards. Omit for ordinary accept actions."
    )]
    #[serde(default)]
    pub space: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateEntityParams {
    #[schemars(
        description = "Canonical entity name (e.g. 'Alice', 'Wenlan', 'PostgreSQL'). Use the exact, full name; the daemon resolves an existing canonical entity idempotently when present."
    )]
    pub name: String,
    #[schemars(
        description = "Entity category: 'person', 'project', 'tool', 'place', 'organization', etc. Free-form string; choose the noun that best describes what it is."
    )]
    pub entity_type: String,
    #[schemars(description = "Topic scope (e.g. 'work', 'wenlan'). Optional.")]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[schemars(
        description = "0.0-1.0 confidence in the entity assertion. Leave unset for caller-default."
    )]
    pub confidence: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateRelationParams {
    #[schemars(
        description = "Resolved source entity id (e.g. 'ent_alice'). Obtain it from create_entity; names are not accepted or auto-created."
    )]
    pub from_entity_id: String,
    #[schemars(
        description = "Resolved target entity id (e.g. 'ent_wenlan'). Obtain it from create_entity; names are not accepted or auto-created."
    )]
    pub to_entity_id: String,
    #[schemars(
        description = "Verb describing the directed relation (e.g. 'works_on', 'prefers', 'uses', 'depends_on'). Snake_case, present-tense."
    )]
    pub relation_type: String,
    #[schemars(
        description = "Required source memory id returned by capture for the user's explicit relation statement."
    )]
    pub source_memory_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct WritePageParams {
    #[schemars(
        description = "Omit to create a new page. Pass an existing page id (e.g. 'page_abc', from the `stale_pages` block in distill output) to refresh that page in place."
    )]
    pub page_id: Option<String>,
    #[schemars(
        description = "Short noun phrase that names the page (e.g. 'Wenlan daemon architecture'). Required when creating; ignored on refresh."
    )]
    pub title: Option<String>,
    #[schemars(
        description = "Markdown body — 3-7 paragraphs of wiki prose with [[wikilinks]]. Do not cite source ids inline; pass them in source_memory_ids and the daemon attaches provenance automatically."
    )]
    pub content: String,
    #[schemars(
        description = "Optional one-sentence summary — the durable claim. On refresh: omit to keep the existing summary; pass empty string to clear it."
    )]
    pub summary: Option<String>,
    #[schemars(
        description = "Optional entity_id (e.g. 'ent_abc') to anchor the page to a knowledge-graph entity. Create only."
    )]
    pub entity_id: Option<String>,
    #[schemars(description = "Topic scope (e.g. 'origin', 'work'). Optional. Create only.")]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[schemars(
        description = "Memory source_ids the page is distilled from. Required for traceability."
    )]
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DeletePageParams {
    #[schemars(
        description = "Page id (e.g. 'page_abc' or legacy 'concept_abc'). Get it from distill output."
    )]
    pub page_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetPageSourcesParams {
    #[schemars(
        description = "Page id (e.g. 'page_abc'). Returns the source memories that distilled into this page, each enriched with the memory's metadata for display."
    )]
    pub page_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetMemoryRevisionsParams {
    #[schemars(
        description = "Memory source id (e.g. 'mem_abc' or 'merged_<uuid>'). Returns the full supersede chain ordered by depth (0 = current)."
    )]
    pub memory_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetPageRevisionsParams {
    #[schemars(
        description = "Page id (e.g. 'page_abc'). Returns the version changelog ordered newest-first."
    )]
    pub page_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AcceptRevisionRequest {
    /// The source_id of the memory whose pending revision should be accepted.
    pub target_source_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DismissRevisionRequest {
    /// The source_id of the memory whose pending revision should be dismissed.
    pub target_source_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListPendingImportsParams {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListRejectionsParams {
    /// Maximum records to return. Default 50. Clamped to 1..=500.
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    /// Filter by rejection reason code (e.g. "duplicate", "low_quality").
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListPendingRevisionsParams {
    /// Maximum rows to return. Server defaults to 50, clamps to 500.
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
}

// ===== Internal Implementations =====

fn format_capture_success(resp: &StoreMemoryResponse) -> String {
    let mut msg = if resp.gated {
        format!(
            "Staged {} for review. Not live yet. The memory it replaces still \
             answers searches until a human accepts it.\nsource_memory_id: {}\ngated: true",
            resp.source_id, resp.source_id
        )
    } else {
        format!(
            "Stored {}\nsource_memory_id: {}",
            resp.source_id, resp.source_id
        )
    };
    append_write_receipt(
        &mut msg,
        resp.space.as_deref(),
        resp.space_source,
        resp.write_outcome,
    );
    if let Some(near_duplicate) = &resp.near_duplicate {
        msg.push_str(&format!(
            "\nnear_duplicate: {} ({:.2})",
            near_duplicate.source_id, near_duplicate.similarity
        ));
    }
    if !resp.warnings.is_empty() {
        msg.push_str("\nWarnings:");
        for warning in &resp.warnings {
            msg.push_str(&format!("\n  - {}", warning));
        }
    }
    msg
}

fn daemon_setup_hint() -> &'static str {
    "Install the local Wenlan runtime and run `wenlan setup`.

Setup choices:
- Local Memory: store, search, and recall now. No model download or API key.
- On-device Model: private local extraction and distill cycles after model download.
- Anthropic Key: richer extraction and distill cycles using your API key.

Install:
  curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | bash
  export PATH=\"$HOME/.wenlan/bin:$PATH\"
  wenlan setup
  wenlan background on
  wenlan status"
}

/// Convert a backend error into a tool-level error result (isError: true)
/// with an actionable message. This keeps the MCP transport healthy
/// (no protocol-level McpError) while telling the caller what happened.
fn tool_error(e: WenlanError, verb: &str) -> CallToolResult {
    let msg = match &e {
        WenlanError::Unreachable(_) => format!(
            "Wenlan daemon is not reachable (retried 3x over ~6s). \
             The {verb} was NOT completed.\n\n{}",
            daemon_setup_hint()
        ),
        WenlanError::Api { status, reason } => match reason {
            Some(reason) => format!(
                "Wenlan daemon returned HTTP {status}: {reason}. The {verb} may not have completed."
            ),
            None => {
                format!("Wenlan daemon returned HTTP {status}. The {verb} may not have completed.")
            }
        },
        WenlanError::Deserialize => String::from(
            "Failed to parse daemon response. \
             This may indicate a version mismatch between wenlan-mcp and the daemon.",
        ),
        WenlanError::ResponseTooLarge => format!(
            "The daemon response exceeded Wenlan MCP's size limit. The {verb} was not completed."
        ),
    };
    CallToolResult::error(vec![Content::text(msg)])
}

/// Call a daemon HTTP method and short-circuit on transport error.
///
/// `$call` — the client method expression (without `.await`)
/// `$label` — the verb string passed to `tool_error` on failure
///
/// Expands to the `match … { Ok(r) => r, Err(e) => return Ok(tool_error(e, $label)) }`
/// boilerplate that every `*_impl` method repeats. The per-site `let binding: T =`
/// type annotation is preserved outside the macro so typed deserialization is
/// still driven by the concrete response type (AGENTS.md hard invariant, PR #77).
macro_rules! try_call {
    ($call:expr, $label:expr) => {
        match $call.await {
            Ok(r) => r,
            Err(e) => return Ok(tool_error(e, $label)),
        }
    };
}

impl WenlanMcpServer {
    /// Resolve the source_agent for a write operation.
    /// Priority: explicit param > MCP client name (from initialize) > configured agent_name.
    fn resolve_source_agent(&self, param_agent: Option<String>) -> Option<String> {
        // 1. Explicit param from tool call
        if let Some(ref agent) = param_agent {
            if !agent.is_empty() {
                return param_agent;
            }
        }
        // 2. Client name captured from MCP initialize handshake
        if let Ok(guard) = self.client_name.lock() {
            if let Some(ref name) = *guard {
                return Some(name.clone());
            }
        }
        // 3. Configured --agent-name flag
        Some(self.agent_name.clone())
    }

    /// Resolve a local user_id for logging or future use.
    /// This value is intentionally not sent on the wire (D4).
    fn resolve_user_id(&self, param_user_id: Option<String>) -> Option<String> {
        if self.transport == TransportMode::Http {
            self.user_id.clone().or(param_user_id)
        } else {
            param_user_id
        }
    }

    pub async fn capture_impl(&self, params: CaptureParams) -> Result<CallToolResult, McpError> {
        // Tool was renamed `remember -> capture` in v0.4. The HTTP request
        // body shape (StoreMemoryRequest) is unchanged; only the MCP-facing
        // tool name shifted.
        let source_agent = self.resolve_source_agent(None);
        if let Some(uid) = self.resolve_user_id(None) {
            tracing::debug!(user_id = %uid, "capture invoked");
        }
        let space_arg = effective_space(&params.space);

        let req = StoreMemoryRequest {
            content: params.content,
            memory_type: params.memory_type,
            space: (space_arg).into(),
            source_agent,
            title: None,
            confidence: params.confidence,
            supersedes: params.supersedes,
            entity: params.entity,
            entity_id: None,
            structured_fields: params.structured_fields.map(serde_json::Value::Object),
            retrieval_cue: params.retrieval_cue,
        };

        let resp: StoreMemoryResponse =
            try_call!(self.client.post("/api/memory/store", &req), "memory store");

        Ok(CallToolResult::success(vec![Content::text(
            format_capture_success(&resp),
        )]))
    }

    pub async fn recall_impl(&self, params: RecallParams) -> Result<CallToolResult, McpError> {
        let space_arg = effective_space(&params.space);
        let req = SearchMemoryRequest {
            query: params.query,
            // Clamp: an unbounded caller-supplied limit either trips the MCP
            // client's 8MB response cap (client.rs MAX_RESPONSE_BYTES) as an
            // opaque transport error, or floods the agent's context window.
            limit: params.limit.unwrap_or(10).clamp(1, 100),
            memory_type: params.memory_type,
            space: space_arg,
            source_agent: self.resolve_source_agent(None),
            // Opt-in cross-encoder rerank. Default `false` preserves the
            // current cost/latency for callers that don't pass the flag.
            // Requires WENLAN_RERANKER_ENABLED=1 on the daemon to take
            // effect; otherwise the daemon logs and falls back to plain
            // hybrid ordering.
            rerank: params.rerank.unwrap_or(false),
        };

        let resp: SearchMemoryResponse =
            try_call!(self.client.post("/api/memory/search", &req), "search");

        let json = serde_json::to_string_pretty(&resp.results)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let mut output = format!(
            "{} results ({:.1}ms)\n{}",
            resp.results.len(),
            resp.took_ms,
            json
        );

        if let Some(pages) = resp.supplemental_pages.as_ref().filter(|p| !p.is_empty()) {
            let pages_json = serde_json::to_string_pretty(pages)
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            output.push_str(&format!("\n\nCompiled pages:\n{}", pages_json));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    pub async fn brief_impl(&self, params: BriefParams) -> Result<CallToolResult, McpError> {
        let request = wenlan_types::BriefReadRequest {
            topic: params.topic,
            space: effective_space(&params.space),
            legacy_context_limit: None,
        };
        let response: wenlan_types::BriefReadResponse =
            try_call!(self.client.post("/api/brief", &request), "brief load");

        let output = match response.state {
            wenlan_types::BriefReadState::SpaceNotResolved => {
                "Space not resolved. Pass `space` or configure a default Space.".to_string()
            }
            wenlan_types::BriefReadState::BriefNotCreated => format!(
                "No Brief exists for Space `{}` yet. Reads never create one; the first handoff update can.",
                response.space.as_deref().unwrap_or("unknown")
            ),
            wenlan_types::BriefReadState::Ready => {
                let brief = response.brief.as_ref().ok_or_else(|| {
                    McpError::internal_error("ready Brief response omitted brief", None)
                })?;
                let brief_json = serde_json::to_string_pretty(brief)
                    .map_err(|error| McpError::internal_error(error.to_string(), None))?;
                let mut output = format!("Space Brief:\n{brief_json}");
                if let Some(related) = response.related_context.as_ref() {
                    let results_json = serde_json::to_string_pretty(&related.results)
                        .map_err(|error| McpError::internal_error(error.to_string(), None))?;
                    output.push_str(&format!(
                        "\n\nRelated Context (query: {}):\n{}",
                        related.query, results_json
                    ));
                }
                output
            }
        };

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    pub async fn context_impl(&self, params: ContextParams) -> Result<CallToolResult, McpError> {
        let space_arg = effective_space(&params.space);
        #[allow(deprecated)]
        let req = ChatContextRequest {
            query: None,
            conversation_id: params.topic,
            max_chunks: params.limit.unwrap_or(20),
            relevance_threshold: None,
            include_goals: true,
            space: space_arg,
        };

        // Extract only the `context` string field from the response.
        //
        // The full ChatContextResponse embeds Vec<SearchResult> which may
        // contain fields added after the published wenlan-types version.
        // Since context_impl only uses `resp.context`, we parse the raw
        // JSON and pull that field directly — this makes the tool forward-
        // compatible with any new fields the daemon might add.
        let raw: serde_json::Value =
            try_call!(self.client.post("/api/context", &req), "context load");

        let context = raw
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        if context.is_empty() {
            Ok(CallToolResult::success(vec![Content::text(
                "No relevant context found".to_string(),
            )]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(context)]))
        }
    }

    fn remove_submitted_agent_work(
        &self,
        digest: &wenlan_types::lint::LintDigest,
    ) -> Result<(), McpError> {
        self.agent_work_cache
            .lock()
            .map_err(|_| McpError::internal_error("lint agent work cache poisoned", None))?
            .remove(digest);
        Ok(())
    }

    fn repair_scope_for_lint_request(
        effective_space: Option<&str>,
        space_locked: bool,
    ) -> Option<wenlan_types::repair::RepairLintScope> {
        match effective_space {
            None => Some(wenlan_types::repair::RepairLintScope::global()),
            Some("uncategorized") if !space_locked => {
                Some(wenlan_types::repair::RepairLintScope::uncategorized())
            }
            Some(space) => {
                wenlan_types::repair::RepairLintScope::registered(space.to_string()).ok()
            }
        }
    }

    pub async fn lint_impl(&self, params: LintParams) -> Result<CallToolResult, McpError> {
        let profile = params.profile.map(Into::into);
        let submission: Option<wenlan_types::lint::LintAgentSubmission> = params
            .agent_submission
            .map(TryInto::try_into)
            .transpose()
            .map_err(|error: wenlan_types::lint::LintContractError| {
                McpError::invalid_params(error.to_string(), None)
            })?;
        let agent_assist = params.agent_assist || submission.is_some();
        if agent_assist && profile != Some(wenlan_types::lint::LintProfile::Deep) {
            return Err(McpError::invalid_params("agent_assist_requires_deep", None));
        }
        let effective_space = effective_space(&params.space);
        let repair_scope = Self::repair_scope_for_lint_request(
            effective_space.as_deref(),
            crate::lock_state::is_locked(),
        );
        let cache_token = match repair_scope.as_ref() {
            Some(lint_scope) => self
                .lint_report_cache
                .lock()
                .map_err(|_| McpError::internal_error("lint report cache poisoned", None))?
                .begin_lint_call(
                    lint_scope,
                    profile.unwrap_or_default(),
                    agent_assist,
                    submission.is_some(),
                )
                .map_err(|error| McpError::internal_error(error, None))?,
            None => None,
        };
        let query = wenlan_types::lint::LintRequestQuery::new(
            wenlan_types::lint::LintQuery::new(profile, effective_space.clone()),
            false,
            agent_assist,
        );
        let report: wenlan_types::lint::LintReport = match submission.as_ref() {
            Some(submission) => try_call!(
                self.client.post_with_query("/api/lint", &query, submission),
                "lint"
            ),
            None => try_call!(self.client.get_with_query("/api/lint", &query), "lint"),
        };
        if let Some(cache_token) = cache_token.as_ref() {
            self.lint_report_cache
                .lock()
                .map_err(|_| McpError::internal_error("lint report cache poisoned", None))?
                .record_from_lint_call(cache_token, report.clone());
        }
        if submission.is_none() {
            if let Some(work) = report.agent_work() {
                self.agent_work_cache
                    .lock()
                    .map_err(|_| McpError::internal_error("lint agent work cache poisoned", None))?
                    .insert(work.clone());
                let value = serde_json::json!({
                    "profile": report.profile(),
                    "scope": report.scope(),
                    "complete": report.complete(),
                    "totals": report.totals(),
                    "agent_work": {
                        "work_digest": work.work_digest(),
                        "populations": work.populations(),
                        "record_count": work.records().len(),
                        "candidate_count": work.candidates().len(),
                        "candidate_page_limit": 10,
                    }
                });
                return Ok(CallToolResult::structured(value));
            }
        } else if let Some(submission) = submission.as_ref() {
            self.remove_submitted_agent_work(submission.work_digest())?;
        }
        let value = serde_json::to_value(report)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::structured(value))
    }

    pub async fn get_lint_agent_work_page_impl(
        &self,
        params: GetLintAgentWorkPageParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Lint agent work pages are available only over local stdio MCP.".to_string(),
            )]));
        }
        let digest = wenlan_types::lint::LintDigest::from_hex(&params.work_digest)
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let offset = params.offset.unwrap_or(0);
        let limit = params.limit.unwrap_or(10);
        if !(1..=10).contains(&limit) {
            return Err(McpError::invalid_params(
                "lint agent work page limit must be between 1 and 10",
                None,
            ));
        }
        let cache = self
            .agent_work_cache
            .lock()
            .map_err(|_| McpError::internal_error("lint agent work cache poisoned", None))?;
        let work = cache
            .get(&digest)
            .ok_or_else(|| McpError::invalid_params("lint agent work is not cached", None))?;
        if offset > work.candidates().len() {
            return Err(McpError::invalid_params(
                "lint agent work page offset is out of range",
                None,
            ));
        }
        let end = offset.saturating_add(limit).min(work.candidates().len());
        let candidates = &work.candidates()[offset..end];
        let references = candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .evidence_refs()
                    .iter()
                    .chain(candidate.counterevidence_refs())
            })
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let records = references
            .into_iter()
            .map(|reference| {
                work.records()
                    .get(usize::from(reference).saturating_sub(1))
                    .cloned()
                    .ok_or_else(|| McpError::internal_error("lint agent work record missing", None))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next_offset = (end < work.candidates().len()).then_some(end);
        let value = serde_json::json!({
            "work_digest": work.work_digest(),
            "offset": offset,
            "next_offset": next_offset,
            "total_candidates": work.candidates().len(),
            "candidates": candidates,
            "records": records,
        });
        Ok(CallToolResult::structured(value))
    }

    fn reject_remote_lint_repair(&self) -> Option<CallToolResult> {
        (self.transport == TransportMode::Http).then(|| {
            CallToolResult::error(vec![Content::text(
                "Lint repair operations are not available over remote connections. Use local stdio MCP on the machine running Wenlan."
                    .to_string(),
            )])
        })
    }

    pub async fn prepare_lint_repair_impl(
        &self,
        params: PrepareLintRepairParams,
    ) -> Result<CallToolResult, McpError> {
        if let Some(blocked) = self.reject_remote_lint_repair() {
            return Ok(blocked);
        }
        let lint_scope = effective_repair_lint_scope(params.lint_scope).map_err(
            |error: wenlan_types::repair::RepairContractError| {
                McpError::invalid_params(error.to_string(), None)
            },
        )?;
        let choice = wenlan_types::repair::RepairChoice::try_from(params.choice)
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let request = wenlan_types::repair::PrepareRepairRequest::try_new_with_choice(
            lint_scope,
            params.general_report,
            params.deep_report,
            choice,
        )
        .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let manifest: wenlan_types::repair::RepairManifest = try_call!(
            self.client.post("/api/repairs/prepare", &request),
            "repair prepare"
        );
        let value = serde_json::to_value(manifest)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::structured(value))
    }

    pub async fn prepare_lint_repair_plan_impl(
        &self,
        params: PrepareLintRepairPlanParams,
    ) -> Result<CallToolResult, McpError> {
        if let Some(blocked) = self.reject_remote_lint_repair() {
            return Ok(blocked);
        }
        let lint_scope = effective_repair_lint_scope(params.lint_scope).map_err(
            |error: wenlan_types::repair::RepairContractError| {
                McpError::invalid_params(error.to_string(), None)
            },
        )?;
        let (general_report, deep_report) = match (params.general_report, params.deep_report) {
            (Some(general_report), deep_report) => (general_report, deep_report),
            (None, Some(_)) => {
                return Err(McpError::invalid_params(
                    "deep_report requires general_report",
                    None,
                ));
            }
            (None, None) => {
                // Consume before POST and never restore on transport failure: the daemon may
                // already have written immutable plan artifacts even when the response is lost.
                let cached_reports = {
                    self.lint_report_cache
                        .lock()
                        .map_err(|_| McpError::internal_error("lint report cache poisoned", None))?
                        .take_reports_for_plan(&lint_scope)
                }
                .map_err(|error| McpError::internal_error(error, None))?;
                let (general_report, deep_report) = cached_reports.ok_or_else(|| {
                    McpError::invalid_params(
                        "a complete General report is not cached for this lint scope",
                        None,
                    )
                })?;
                (general_report, deep_report)
            }
        };
        let request = wenlan_types::repair_plan::RepairPlanRequest::try_new(
            lint_scope,
            general_report,
            deep_report,
        )
        .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let summary: wenlan_types::repair_plan::RepairPlanSummary = try_call!(
            self.client.post("/api/repairs/plan", &request),
            "repair plan prepare"
        );
        let value = serde_json::to_value(summary)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        let value =
            add_repair_plan_source_reports(value, request.general_report(), request.deep_report())
                .map_err(|error| McpError::internal_error(error, None))?;
        Ok(CallToolResult::structured(value))
    }

    pub async fn get_lint_repair_plan_entries_impl(
        &self,
        params: GetLintRepairPlanEntriesParams,
    ) -> Result<CallToolResult, McpError> {
        if let Some(blocked) = self.reject_remote_lint_repair() {
            return Ok(blocked);
        }
        let plan_digest = wenlan_types::repair::RepairDigest::parse(&params.plan_digest)
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let request = wenlan_types::repair_plan::RepairPlanEntriesRequest::try_new(
            params.plan_id,
            plan_digest,
            params.offset.unwrap_or(0),
            params.limit.unwrap_or(50),
        )
        .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let page: wenlan_types::repair_plan::RepairPlanEntriesPage = try_call!(
            self.client.post("/api/repairs/plan/entries", &request),
            "repair plan entries"
        );
        let value = serde_json::to_value(page)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::structured(value))
    }

    pub async fn apply_lint_repair_impl(
        &self,
        params: ApplyLintRepairParams,
    ) -> Result<CallToolResult, McpError> {
        if let Some(blocked) = self.reject_remote_lint_repair() {
            return Ok(blocked);
        }
        let digest = wenlan_types::repair::RepairDigest::parse(&params.approved_manifest_digest)
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let request = wenlan_types::repair::ApplyRepairRequest::try_new(
            params.manifest_id,
            digest,
            params.approval,
        )
        .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let receipt: wenlan_types::repair::RepairApplyReceipt = try_call!(
            self.client.post("/api/repairs/apply", &request),
            "repair apply"
        );
        let value = serde_json::to_value(receipt)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::structured(value))
    }

    pub async fn verify_lint_repair_impl(
        &self,
        params: VerifyLintRepairParams,
    ) -> Result<CallToolResult, McpError> {
        if let Some(blocked) = self.reject_remote_lint_repair() {
            return Ok(blocked);
        }
        let manifest_digest = wenlan_types::repair::RepairDigest::parse(&params.manifest_digest)
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let apply_receipt_digest =
            wenlan_types::repair::RepairDigest::parse(&params.apply_receipt_digest)
                .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let next_apply = params
            .next_apply
            .map(|next| {
                let digest =
                    wenlan_types::repair::RepairDigest::parse(&next.approved_manifest_digest)?;
                wenlan_types::repair::ApplyRepairRequest::try_new(
                    next.manifest_id,
                    digest,
                    next.approval,
                )
            })
            .transpose()
            .map_err(|error: wenlan_types::repair::RepairContractError| {
                McpError::invalid_params(error.to_string(), None)
            })?;
        let request =
            wenlan_types::repair::VerifyRepairRequest::try_new_with_optional_deep_and_next_apply(
                params.manifest_id,
                manifest_digest,
                apply_receipt_digest,
                params.general_report,
                params.deep_report,
                next_apply,
            )
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let receipt: wenlan_types::repair::RepairVerificationReceipt = try_call!(
            self.client.post("/api/repairs/verify", &request),
            "repair verify"
        );
        let value = serde_json::to_value(receipt)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::structured(value))
    }

    pub async fn forget_impl(&self, memory_id: &str) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Delete operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to delete memories."
                    .to_string(),
            )]));
        }

        let resp: DeleteResponse = try_call!(
            self.client
                .delete(&format!("/api/memory/delete/{}", memory_id)),
            "delete"
        );

        Ok(CallToolResult::success(vec![Content::text(
            if resp.deleted {
                "Memory deleted"
            } else {
                "Memory not found"
            }
            .to_string(),
        )]))
    }

    pub async fn distill_impl(&self, params: DistillParams) -> Result<CallToolResult, McpError> {
        let mut body = serde_json::Map::new();
        if let Some(t) = params.target.as_deref().filter(|t| !t.is_empty()) {
            body.insert("target".into(), serde_json::Value::String(t.to_string()));
        }
        if params.force.unwrap_or(false) {
            body.insert("force".into(), serde_json::Value::Bool(true));
        }
        let body = serde_json::Value::Object(body);
        match self
            .client
            .post::<serde_json::Value, serde_json::Value>("/api/distill", &body)
            .await
        {
            Ok(resp) => {
                if let Some(unresolved) = resp.get("unresolved").and_then(|v| v.as_str()) {
                    let hint = resp
                        .get("hint")
                        .and_then(|v| v.as_str())
                        .unwrap_or("no matching target");
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Could not resolve target `{}`. {}",
                        unresolved, hint
                    ))]));
                }
                // Return the daemon's structured response verbatim. The caller
                // (agent in Claude Code, Cursor, etc.) reads `pending` from the
                // payload, synthesizes each cluster in-session, and POSTs the
                // resulting pages back to /api/pages. The MCP tool stays as a
                // thin wrapper; the synthesis lives where the LLM is.
                let pretty =
                    serde_json::to_string_pretty(&resp).unwrap_or_else(|_| resp.to_string());
                Ok(CallToolResult::success(vec![Content::text(pretty)]))
            }
            Err(e) => Ok(tool_error(e, "distill")),
        }
    }

    pub async fn list_pending_impl(
        &self,
        params: ListPendingParams,
    ) -> Result<CallToolResult, McpError> {
        let limit = params.limit.unwrap_or(20).min(100);
        let req = ListMemoriesRequest {
            memory_type: None,
            space: effective_space(&params.space),
            confirmed: Some(false),
            limit,
        };
        let resp: ListMemoriesResponse =
            try_call!(self.client.post("/api/memory/list", &req), "list_pending");
        let body = serde_json::to_string_pretty(&resp.memories)
            .unwrap_or_else(|e| format!("serialization error: {e}"));
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    pub async fn confirm_memory_impl(&self, memory_id: &str) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Confirm operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan for review."
                    .to_string(),
            )]));
        }
        let path = format!("/api/memory/confirm/{}", memory_id);
        match self
            .client
            .post::<serde_json::Value, ConfirmResponse>(&path, &serde_json::json!({}))
            .await
        {
            // The daemon 404s an unknown id, but an older daemon returns 200
            // without `updated`, which defaults to `true` so that response
            // still reads as success (mirrors forget_impl's flag branch).
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(
                if resp.updated {
                    format!("Memory {} confirmed.", memory_id)
                } else {
                    format!("Memory {} not found.", memory_id)
                },
            )])),
            Err(e) => Ok(tool_error(e, "confirm_memory")),
        }
    }

    pub async fn create_entity_impl(
        &self,
        params: CreateEntityParams,
    ) -> Result<CallToolResult, McpError> {
        let req = CreateEntityRequest {
            name: params.name,
            entity_type: params.entity_type,
            space: (effective_space(&params.space)).into(),
            source_agent: self.resolve_source_agent(None),
            confidence: params.confidence,
        };
        let resp: CreateEntityResponse = try_call!(
            self.client.post("/api/memory/entities", &req),
            "create_entity"
        );
        Ok(CallToolResult::success(vec![Content::text(
            format_entity_ready_response(resp),
        )]))
    }

    pub async fn list_refinements_impl(
        &self,
        params: ListRefinementsParams,
    ) -> Result<CallToolResult, McpError> {
        let mut path = String::from("/api/refinery/queue");
        let mut query = Vec::new();
        if let Some(action) = params.action.as_deref() {
            query.push(format!("action={}", url_encode_simple(action)));
        }
        if let Some(limit) = params.limit {
            query.push(format!("limit={limit}"));
        }
        if !query.is_empty() {
            path.push('?');
            path.push_str(&query.join("&"));
        }

        let resp: ListRefinementsResponse = try_call!(self.client.get(&path), "list_refinements");
        let pretty = serde_json::to_string_pretty(&resp.proposals)
            .map_err(|error| McpError::internal_error(error.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} pending review proposals\n{}",
            resp.proposals.len(),
            pretty
        ))]))
    }

    pub async fn reject_refinement_impl(
        &self,
        params: RejectRefinementParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Review proposal operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to reject proposals."
                    .to_string(),
            )]));
        }
        let path = format!(
            "/api/refinery/queue/{}/reject",
            url_encode_simple(&params.id)
        );
        let resp: RejectRefinementResponse = try_call!(
            self.client.post(&path, &serde_json::json!({})),
            "reject_refinement"
        );
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Review proposal {} dismissed.",
            resp.id
        ))]))
    }

    pub async fn accept_refinement_impl(
        &self,
        params: AcceptRefinementParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Review proposal operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to accept proposals."
                    .to_string(),
            )]));
        }
        let path = format!(
            "/api/refinery/queue/{}/accept",
            url_encode_simple(&params.id)
        );
        let request = match params.space {
            Some(space) => {
                wenlan_types::requests::AcceptRefinementRequest::PickSpace { space, notes: None }
            }
            None => wenlan_types::requests::AcceptRefinementRequest::Accept { notes: None },
        };
        let resp: AcceptRefinementResponse =
            try_call!(self.client.post(&path, &request), "accept_refinement");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Review proposal {} accepted (action={}).",
            resp.id, resp.action_applied
        ))]))
    }

    pub async fn create_relation_impl(
        &self,
        params: CreateRelationParams,
    ) -> Result<CallToolResult, McpError> {
        if params.source_memory_id.trim().is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "source_memory_id must not be blank; pass the id returned by capture.".to_string(),
            )]));
        }
        let detail_path = format!(
            "/api/memory/{}/detail",
            url_encode_simple(&params.source_memory_id)
        );
        let source: wenlan_types::responses::MemoryDetailResponse = try_call!(
            self.client.get(&detail_path),
            "create_relation source preflight"
        );
        if source.memory.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Source memory {} is missing or not visible; relation was not created.",
                params.source_memory_id
            ))]));
        }

        let req = CreateRelationRequest {
            from_entity: params.from_entity_id,
            to_entity: params.to_entity_id,
            relation_type: params.relation_type,
            source_agent: self.resolve_source_agent(None),
            confidence: None,
            explanation: None,
            source_memory_id: Some(params.source_memory_id),
            span: None,
            model_version: None,
            prompt_version: None,
        };
        let resp: CreateRelationResponse = try_call!(
            self.client.post("/api/memory/relations", &req),
            "create_relation"
        );
        Ok(CallToolResult::success(vec![Content::text(
            format_relation_ready_response(resp),
        )]))
    }

    pub async fn write_page_impl(
        &self,
        params: WritePageParams,
    ) -> Result<CallToolResult, McpError> {
        match params.page_id {
            Some(page_id) => {
                if params.source_memory_ids.is_empty() {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "source_memory_ids is required when refreshing — carry through \
                         the stale page's existing list from distill output"
                            .to_string(),
                    )]));
                }
                if self.transport == TransportMode::Http {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "Update operations are not available over remote connections. \
                         Use local MCP on the machine running Wenlan to update pages."
                            .to_string(),
                    )]));
                }
                let req = wenlan_types::requests::RefreshPageRequest {
                    content: params.content,
                    source_memory_ids: params.source_memory_ids,
                    summary: params.summary,
                };
                let path = format!("/api/pages/{page_id}");
                // Typed end-to-end: a wire-shape drift on the daemon side fails at
                // deserialize instead of silently returning the no-op "Refreshed"
                // line. Same discipline as the other typed page responses.
                let resp: wenlan_types::responses::PageWriteResponse =
                    try_call!(self.client.put(&path, &req), "write_page");
                // Ownership gate (spec §5.2): a human-owned page is never overwritten
                // in place; the daemon stages a revision card. Surface that so the
                // caller does not believe the prose was rewritten.
                let msg = format_update_page_response(&page_id, resp);
                Ok(CallToolResult::success(vec![Content::text(msg)]))
            }
            None => {
                let Some(title) = params.title else {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "title is required when creating a page (no page_id given)".to_string(),
                    )]));
                };
                let req = CreateConceptRequest {
                    title,
                    content: params.content,
                    summary: params.summary,
                    entity_id: params.entity_id,
                    space: (effective_space(&params.space)).into(),
                    source_memory_ids: params.source_memory_ids,
                    creation_kind: None,
                    workspace: None,
                };
                let resp: CreatePageResponse =
                    try_call!(self.client.post("/api/pages", &req), "write_page");
                let text = format_create_page_response(resp);
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
        }
    }

    pub async fn delete_page_impl(&self, page_id: &str) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Delete operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to delete pages."
                    .to_string(),
            )]));
        }

        // Envelope for DELETE /api/pages/{id} (handle_delete_page,
        // memory_routes.rs:2023): a bare `{"status": ...}`, unrelated to the
        // `DeleteResponse{deleted: bool}` shape used elsewhere. No exported
        // wenlan-types struct matches it, so a local type stands in for
        // serde_json::Value -- a wire-shape drift now fails at deserialize
        // instead of round-tripping silently as untyped JSON. `deny_unknown_fields`
        // makes an ADDITIVE envelope key fail loud too, not just a renamed one.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct DeletePageResponse {
            status: String,
        }

        let path = format!("/api/pages/{}", page_id);
        let resp: DeletePageResponse = try_call!(self.client.delete(&path), "delete_page");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Page {} {}",
            page_id, resp.status
        ))]))
    }

    pub async fn get_page_sources_impl(&self, page_id: &str) -> Result<CallToolResult, McpError> {
        let path = format!("/api/pages/{}/sources", page_id);
        // Daemon returns Vec<PageSourceWithMemory> directly (no envelope key).
        let resp: Vec<PageSourceWithMemory> = try_call!(self.client.get(&path), "get_page_sources");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} sources\n{}",
            resp.len(),
            pretty
        ))]))
    }

    pub async fn get_memory_revisions_impl(
        &self,
        memory_id: &str,
    ) -> Result<CallToolResult, McpError> {
        let path = format!("/api/memory/{}/revisions", memory_id);
        let resp: ListMemoryRevisionsResponse =
            try_call!(self.client.get(&path), "get_memory_revisions");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "chain depth {}\n{}",
            resp.chain_depth, pretty
        ))]))
    }

    pub async fn get_page_revisions_impl(&self, page_id: &str) -> Result<CallToolResult, McpError> {
        let path = format!("/api/pages/{}/revisions", page_id);
        let resp: ListPageRevisionsResponse =
            try_call!(self.client.get(&path), "get_page_revisions");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "version {} ({} entries)\n{}",
            resp.current_version,
            resp.entries.len(),
            pretty
        ))]))
    }

    pub async fn accept_revision_impl(
        &self,
        req: AcceptRevisionRequest,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Revision operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to accept memory revisions."
                    .to_string(),
            )]));
        }
        let path = format!("/api/memory/revision/{}/accept", req.target_source_id);
        let response = try_call!(
            self.client.post_empty::<RevisionAcceptResponse>(&path),
            "accept_revision"
        );
        let pretty = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(pretty)]))
    }

    pub async fn dismiss_revision_impl(
        &self,
        req: DismissRevisionRequest,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Revision operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to dismiss memory revisions."
                    .to_string(),
            )]));
        }
        let path = format!("/api/memory/revision/{}/dismiss", req.target_source_id);
        let response = try_call!(
            self.client.post_empty::<RevisionDismissResponse>(&path),
            "dismiss_revision"
        );
        let pretty = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(pretty)]))
    }

    pub async fn list_pending_imports_impl(
        &self,
        _params: ListPendingImportsParams,
    ) -> Result<CallToolResult, McpError> {
        let resp: Vec<wenlan_types::import::PendingImport> =
            try_call!(self.client.get("/api/import/state"), "list_pending_imports");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} pending import(s)\n{}",
            resp.len(),
            pretty
        ))]))
    }

    pub async fn list_rejections_impl(
        &self,
        params: ListRejectionsParams,
    ) -> Result<CallToolResult, McpError> {
        let mut path = String::from("/api/memory/rejections");
        let mut query = Vec::new();
        if let Some(limit) = params.limit {
            query.push(format!("limit={}", limit.clamp(1, 500)));
        }
        if let Some(reason) = params.reason.as_deref().filter(|reason| !reason.is_empty()) {
            query.push(format!("reason={}", url_encode_simple(reason)));
        }
        if !query.is_empty() {
            path.push('?');
            path.push_str(&query.join("&"));
        }

        let resp: Vec<wenlan_types::memory::RejectionRecord> =
            try_call!(self.client.get(&path), "list_rejections");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} rejection(s)\n{}",
            resp.len(),
            pretty
        ))]))
    }

    pub async fn list_pending_revisions_impl(
        &self,
        params: ListPendingRevisionsParams,
    ) -> Result<CallToolResult, McpError> {
        let path = match params.limit {
            Some(l) => format!("/api/memory/pending-revisions?limit={}", l.clamp(1, 500)),
            None => "/api/memory/pending-revisions".to_string(),
        };
        let resp: Vec<wenlan_types::responses::PendingRevisionItem> =
            try_call!(self.client.get(&path), "list_pending_revisions");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} pending revision(s)\n{}",
            resp.len(),
            pretty
        ))]))
    }
}

fn format_update_page_response(
    page_id: &str,
    resp: wenlan_types::responses::PageWriteResponse,
) -> String {
    if resp.gated {
        match resp.revision_card_id {
            Some(card) => format!(
                "Page {page_id} is human-owned; staged revision card for review — prose left unchanged.\ngated: true\nrevision_card_id: {card}"
            ),
            None => format!(
                "Page {page_id} is human-owned; staged a revision card for review — prose left unchanged.\ngated: true"
            ),
        }
    } else {
        format!("Refreshed page {page_id}")
    }
}

fn format_create_page_response(resp: CreatePageResponse) -> String {
    let mut text = match resp.attached_to {
        Some(page_id) => format!("Attached to existing page {page_id}"),
        None => format!("Created page {}", resp.id),
    };
    append_write_receipt(
        &mut text,
        resp.space.as_deref(),
        resp.space_source,
        resp.write_outcome,
    );
    for warning in resp.warnings {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    text
}

fn format_entity_ready_response(resp: CreateEntityResponse) -> String {
    let mut text = format!("Entity {} ready", resp.id);
    append_write_receipt(
        &mut text,
        resp.space.as_deref(),
        resp.space_source,
        resp.write_outcome,
    );
    for warning in resp.warnings {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    text
}

fn append_write_receipt(
    text: &mut String,
    space: Option<&str>,
    source: Option<wenlan_types::WriteSpaceSource>,
    outcome: Option<wenlan_types::WriteOutcome>,
) {
    if space.is_some() || source.is_some() || outcome.is_some() {
        text.push_str("\nspace: ");
        text.push_str(space.unwrap_or("Uncategorized"));
    }
    if let Some(source) = source {
        text.push_str("\nspace_source: ");
        text.push_str(match source {
            wenlan_types::WriteSpaceSource::Request => "request",
            wenlan_types::WriteSpaceSource::Header => "header",
            wenlan_types::WriteSpaceSource::Default => "default",
            wenlan_types::WriteSpaceSource::Uncategorized => "uncategorized",
            wenlan_types::WriteSpaceSource::Existing => "existing",
            wenlan_types::WriteSpaceSource::Unknown => "unknown",
        });
    }
    if let Some(outcome) = outcome {
        text.push_str("\nwrite_outcome: ");
        text.push_str(match outcome {
            wenlan_types::WriteOutcome::Created => "created",
            wenlan_types::WriteOutcome::ResolvedExisting => "resolved_existing",
            wenlan_types::WriteOutcome::AttachedExisting => "attached_existing",
            wenlan_types::WriteOutcome::Unknown => "unknown",
        });
    }
}

fn format_relation_ready_response(resp: CreateRelationResponse) -> String {
    let mut text = format!("Relation {} ready", resp.id);
    for warning in resp.warnings {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    text
}

fn url_encode_simple(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte))
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

// ===== Tool Registrations =====

#[tool_router]
impl WenlanMcpServer {
    pub fn new(
        client: WenlanClient,
        transport: TransportMode,
        agent_name: String,
        user_id: Option<String>,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            client,
            transport,
            agent_name,
            client_name: std::sync::Arc::new(std::sync::Mutex::new(None)),
            agent_work_cache: std::sync::Arc::new(std::sync::Mutex::new(
                LintAgentWorkCache::default(),
            )),
            lint_report_cache: std::sync::Arc::new(std::sync::Mutex::new(
                LintReportCache::default(),
            )),
            user_id,
        }
    }

    // --- Primary Tools ---

    #[tool(
        description = "Capture a memory. Call PROACTIVELY when you learn something durable about the user — preferences, decisions, corrections, or facts about people/projects/tools they care about. Don't wait for the user to say 'remember this' or 'capture that' — that phrasing is a floor, not a trigger.\n\nWrite content as a complete, self-contained statement — someone reading it months later with no conversation context should understand it. Include the WHY, not just the WHAT. Name people, projects, and tools explicitly.\n\nThe backend auto-classifies type, extracts structured fields, detects entities, and links to the knowledge graph. You don't need to set memory_type or structured_fields unless you're confident — omitting them gets better results than guessing wrong.\n\nDo NOT store: system prompts, boot logs, heartbeat/health checks, transient task state ('currently working on...'), tool output/responses, architecture dumps, single-word acknowledgments, or content you have already stored. Focus on durable facts, preferences, decisions, lessons, gotchas, and identity information. Each call is one atomic idea — \"prefers TDD\" and \"uses pytest\" are two calls, not one.",
        annotations(
            title = "Capture",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn capture(
        &self,
        Parameters(params): Parameters<CaptureParams>,
    ) -> Result<CallToolResult, McpError> {
        self.capture_impl(params).await
    }

    #[tool(
        description = "Search memories by query. Use when the user asks 'do you remember', 'what do you know about', 'look up', or when you need a specific fact before acting.\n\nWrite queries as natural language — the search engine handles semantic matching. For precision, use filters (memory_type, space) to narrow results. If you get too many results, add filters rather than making the query longer.\n\nFor higher retrieval quality at the cost of latency, pass `rerank: true` to opt into the cross-encoder reranker (requires WENLAN_RERANKER_ENABLED=1 on the daemon).\n\nThis is for targeted lookups. To resume work from a Space-owned project snapshot, use brief instead.",
        annotations(title = "Recall", read_only_hint = true, open_world_hint = false)
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        self.recall_impl(params).await
    }

    #[tool(
        description = "Read the current Space Brief when resuming project work or when the user asks to catch up. Omit `topic` to return only the complete Brief. Provide `topic` to append separately labeled related context scoped to the same Space. Reads never create or mutate a Brief; the first handoff update can create it.",
        annotations(title = "Brief", read_only_hint = true, open_world_hint = false)
    )]
    async fn brief(
        &self,
        Parameters(params): Parameters<BriefParams>,
    ) -> Result<CallToolResult, McpError> {
        self.brief_impl(params).await
    }

    #[tool(
        description = "Deprecated compatibility adapter for known callers. Use brief for Space-owned project orientation and recall for targeted memory lookup.",
        annotations(
            title = "Context (deprecated)",
            read_only_hint = true,
            open_world_hint = false
        )
    )]
    async fn context(
        &self,
        Parameters(params): Parameters<ContextParams>,
    ) -> Result<CallToolResult, McpError> {
        self.context_impl(params).await
    }

    #[tool(
        description = "Run Wenlan's read-only system lint on demand. General is the default bounded deterministic profile. Deep adds expensive deterministic checks plus full-store local semantic candidate generation; bounded candidate packets are adjudicated either by the daemon's configured provider or, with explicit agent_assist consent, by the calling agent through a typed prepare-and-submit protocol. Results are the canonical typed report; incomplete takes precedence over findings.",
        annotations(title = "Lint", read_only_hint = true, open_world_hint = false)
    )]
    async fn lint(
        &self,
        Parameters(params): Parameters<LintParams>,
    ) -> Result<CallToolResult, McpError> {
        self.lint_impl(params).await
    }

    #[tool(
        description = "Read one bounded page from the exact agent-assisted Deep work packet returned by lint. Requires the work_digest from the compact lint prepare response. Each page includes candidates plus only their referenced records; follow next_offset until absent, then submit exactly one verdict per candidate in the second lint call. Local stdio only.",
        annotations(
            title = "Get lint agent work page",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_lint_agent_work_page(
        &self,
        Parameters(params): Parameters<GetLintAgentWorkPageParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_lint_agent_work_page_impl(params).await
    }

    #[tool(
        description = "Prepare one approval-gated lint repair manifest from fresh reports and one exact tagged choice. Reclassification requires complete General plus agent-assisted Deep; deterministic Review Item choices may use General alone. This binds one durable owner, exact mutation, rollback artifact, and post-repair assertion without mutating the Review Item or canonical data. Page-title choices contain title intent only; the fully initialized daemon computes the canonical embedding before manifest persistence. Local stdio only.",
        annotations(
            title = "Prepare lint repair",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn prepare_lint_repair(
        &self,
        Parameters(params): Parameters<PrepareLintRepairParams>,
    ) -> Result<CallToolResult, McpError> {
        self.prepare_lint_repair_impl(params).await
    }

    #[tool(
        description = "Prepare a complete lint-repair plan from the latest complete General report and, when available, the final agent-assisted Deep report cached for the exact lint scope in this local MCP process. Pass only lint_scope; report payloads stay inside the MCP process. A missing Deep report keeps semantic planning incomplete but does not block General-only deterministic planning. Returns a compact verified summary, exact plan digest, counts, source report metadata, and durable artifact path without inlining the potentially large entry set. Fetch every ready, review, system_action, or blocked entry with get_lint_repair_plan_entries before presenting or applying anything. This writes only repair-control-plane artifacts and durable Review Items, never canonical memory or Page data. Local stdio only.",
        annotations(
            title = "Prepare lint repair plan",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn prepare_lint_repair_plan(
        &self,
        Parameters(params): Parameters<PrepareLintRepairPlanParams>,
    ) -> Result<CallToolResult, McpError> {
        self.prepare_lint_repair_plan_impl(params).await
    }

    #[tool(
        description = "Read one verified byte-bounded page of entries from a previously prepared lint-repair plan. Requires the exact plan id and digest returned by prepare_lint_repair_plan. The daemon may return fewer entries than the requested limit to keep the response safe; follow the returned next_offset until absent. Returns stable ordered entries and total_entries. This is read-only and local stdio only.",
        annotations(
            title = "Get lint repair plan entries",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_lint_repair_plan_entries(
        &self,
        Parameters(params): Parameters<GetLintRepairPlanEntriesParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_lint_repair_plan_entries_impl(params).await
    }

    #[tool(
        description = "Apply exactly one prepared lint repair through the daemon's CAS canonical writer. Requires the exact user approval string `apply repair <manifest-id> <manifest-digest>` and refuses stale targets. Local stdio only.",
        annotations(
            title = "Apply lint repair",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn apply_lint_repair(
        &self,
        Parameters(params): Parameters<ApplyLintRepairParams>,
    ) -> Result<CallToolResult, McpError> {
        self.apply_lint_repair_impl(params).await
    }

    #[tool(
        description = "Record verification for an applied lint repair after rerunning complete General and applicable agent-assisted Deep lint. Rejects surviving target evidence, new actionable findings, stale reports, or any non-target state change. Local stdio only.",
        annotations(
            title = "Verify lint repair",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn verify_lint_repair(
        &self,
        Parameters(params): Parameters<VerifyLintRepairParams>,
    ) -> Result<CallToolResult, McpError> {
        self.verify_lint_repair_impl(params).await
    }

    #[tool(
        description = "Delete a memory by ID. Use when the user says 'forget this', 'delete that', 'that's wrong and should be removed'. Requires the source_id — get it from recall first.\n\nThis is destructive and cannot be undone. For corrections, prefer storing a new memory with the supersedes param pointing to the old one — this preserves history. If your agent's trust level is below full, that supersedes write may stage for human review instead of taking effect immediately.",
        annotations(
            title = "Forget",
            read_only_hint = false,
            destructive_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn forget(
        &self,
        Parameters(params): Parameters<ForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        self.forget_impl(&params.memory_id).await
    }

    #[tool(
        description = "Trigger Wenlan's distillation pass. With no `target`, runs a full pass that clusters new memories into pages and refreshes the wiki view. With a `target`, scopes the pass: a page id (`page_*` or `concept_*`) re-distills that single page, an entity name scopes clustering to that entity, a space value (e.g. `work`, `personal`) scopes to that space. Use when the user explicitly asks to synthesize, distill, or rebuild a page. The daemon also runs distillation periodically in the background, so don't trigger redundantly during normal flow.",
        annotations(
            title = "Distill",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn distill(
        &self,
        Parameters(params): Parameters<DistillParams>,
    ) -> Result<CallToolResult, McpError> {
        self.distill_impl(params).await
    }

    #[tool(
        description = "List unconfirmed memories pending review. Use when the user wants to audit what got captured before it becomes authoritative — typical phrases: 'review pending', 'show unconfirmed', 'what got captured'. Pair with `confirm_memory` to accept and `forget` to reject.",
        annotations(title = "List pending", read_only_hint = true, open_world_hint = false)
    )]
    async fn list_pending(
        &self,
        Parameters(params): Parameters<ListPendingParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_pending_impl(params).await
    }

    #[tool(
        description = "Confirm a pending memory by source_id. Use during review to accept a memory the agent captured. The user typically picks from a `list_pending` result. To reject instead, call `forget` with the same `memory_id`.",
        annotations(
            title = "Confirm memory",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn confirm_memory(
        &self,
        Parameters(params): Parameters<ConfirmMemoryParams>,
    ) -> Result<CallToolResult, McpError> {
        self.confirm_memory_impl(&params.memory_id).await
    }

    #[tool(
        description = "List pending daemon refinement proposals only when the user explicitly asks to inspect or review the proposal/refinement queue; never poll this queue ambiently. Returns typed actions and payloads, including vocab_promote. Filter by action and bound the result with limit. Pair with accept_refinement or reject_refinement only after an unambiguous item-level user decision.",
        annotations(
            title = "List refinement proposals",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_refinements(
        &self,
        Parameters(params): Parameters<ListRefinementsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_refinements_impl(params).await
    }

    #[tool(
        description = "Accept one listed refinement proposal by id only after the user gives an unambiguous item-level accept decision. For cross_space_discovery, pass the selected destination space. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Accept refinement proposal",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn accept_refinement(
        &self,
        Parameters(params): Parameters<AcceptRefinementParams>,
    ) -> Result<CallToolResult, McpError> {
        self.accept_refinement_impl(params).await
    }

    #[tool(
        description = "Reject one listed refinement proposal by id only after the user gives an unambiguous item-level reject decision. Skip or cancel is a no-op. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Reject refinement proposal",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn reject_refinement(
        &self,
        Parameters(params): Parameters<RejectRefinementParams>,
    ) -> Result<CallToolResult, McpError> {
        self.reject_refinement_impl(params).await
    }

    #[tool(
        description = "Resolve a durable named entity idempotently. Use only as a support step when the user explicitly establishes a durable named entity, or when resolved ids are needed for a relation the user explicitly stated. Do not call create_entity for every ordinary capture; capture already handles its primary entity and daemon enrichment handles routine extraction.",
        annotations(
            title = "Resolve entity",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn create_entity(
        &self,
        Parameters(params): Parameters<CreateEntityParams>,
    ) -> Result<CallToolResult, McpError> {
        self.create_entity_impl(params).await
    }

    #[tool(
        description = "Create a source-backed directed relation only when the user explicitly states a durable relation. Resolve both endpoint ids with create_entity, capture the user's relation statement, then pass those ids plus the capture result's source_memory_id. Entity names are not accepted or auto-created. Never infer an unstated relation.",
        annotations(
            title = "Create relation",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn create_relation(
        &self,
        Parameters(params): Parameters<CreateRelationParams>,
    ) -> Result<CallToolResult, McpError> {
        self.create_relation_impl(params).await
    }

    #[tool(
        description = "Create or refresh a distilled wiki page. Omit page_id to create a new page (title required); the daemon writes the DB row and the on-disk <pages dir>/<slug>.md projection atomically (default ~/.wenlan/pages/, slug made unique on collision). Pass page_id (from the `stale_pages` block in distill output) to refresh that page in place — replaces content + source_memory_ids + optional summary, clears stale_reason, preserves page_id and created_at, bumps version monotonically so external [[wikilinks]] keep working. Never delete_page + recreate to refresh: that churns ids and loses version history. Refresh is not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Write page",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn write_page(
        &self,
        Parameters(params): Parameters<WritePageParams>,
    ) -> Result<CallToolResult, McpError> {
        self.write_page_impl(params).await
    }

    #[tool(
        description = "Delete a page by id. Destructive — removes both the DB row and the on-disk md projection. Use during a /distill refresh to drop a stale page before creating its replacement, or when the user explicitly asks to remove a page. Pages without sources can be re-derived by running /distill again on the same scope.",
        annotations(
            title = "Delete page",
            read_only_hint = false,
            destructive_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn delete_page(
        &self,
        Parameters(params): Parameters<DeletePageParams>,
    ) -> Result<CallToolResult, McpError> {
        self.delete_page_impl(&params.page_id).await
    }

    #[tool(
        description = "Fetch the source memories of a page — the memory ids the page was distilled from, each enriched with the memory's title, content, type, and space. The /distill skill uses this on the stale-page refresh path after obtaining the page id from the `stale_pages` block in distill output.",
        annotations(
            title = "Get page sources",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_page_sources(
        &self,
        Parameters(params): Parameters<GetPageSourcesParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_page_sources_impl(&params.page_id).await
    }

    #[tool(
        description = "Fetch the supersede chain for one memory, ordered by depth (0 = current). Use after recall only when the user explicitly asks for that memory's history or evolution, or to verify that a correction was recorded.",
        annotations(
            title = "Get memory revisions",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_memory_revisions(
        &self,
        Parameters(params): Parameters<GetMemoryRevisionsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_memory_revisions_impl(&params.memory_id).await
    }

    #[tool(
        description = "Fetch the version changelog for one page, ordered newest-first. Use only when the user explicitly asks for a page's changelog, version history, or which source memories triggered a re-distill.",
        annotations(
            title = "Get page revisions",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_page_revisions(
        &self,
        Parameters(params): Parameters<GetPageRevisionsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_page_revisions_impl(&params.page_id).await
    }

    #[tool(
        description = "Accept a pending memory revision. Replaces the target memory's content \
                       with the proposed revision content and removes the revision row from the \
                       pending list. Returns the consumed revision id. Returns an error if no \
                       pending revision exists for that target. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Accept revision",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn accept_revision(
        &self,
        Parameters(req): Parameters<AcceptRevisionRequest>,
    ) -> Result<CallToolResult, McpError> {
        self.accept_revision_impl(req).await
    }

    #[tool(
        description = "Dismiss a pending memory revision. Deletes the revision row; the original \
                       memory is unchanged. Returns an error if no pending revision exists for \
                       that target. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Dismiss revision",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn dismiss_revision(
        &self,
        Parameters(req): Parameters<DismissRevisionRequest>,
    ) -> Result<CallToolResult, McpError> {
        self.dismiss_revision_impl(req).await
    }

    #[tool(
        description = "List in-flight chat-history imports awaiting processing or completion. Use only when the user asks what imports are running, whether an export is done, or requests import progress. Returns id, vendor, stage, source path, and processed/total conversation counts.",
        annotations(
            title = "List pending imports",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_pending_imports(
        &self,
        Parameters(params): Parameters<ListPendingImportsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_pending_imports_impl(params).await
    }

    #[tool(
        description = "List quality-gate flags and rejections: memories the daemon discarded before storing (low quality or another filter), plus near-duplicate memories that were stored anyway but logged for review (reason \"near_duplicate\"). Use only to diagnose a missing capture, a flagged near duplicate, or when the user asks what Wenlan rejected. Returns reason codes, detail, and similarity info; optional limit and reason filters narrow the audit.",
        annotations(
            title = "List rejections",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_rejections(
        &self,
        Parameters(params): Parameters<ListRejectionsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_rejections_impl(params).await
    }

    #[tool(
        description = "List memories awaiting human accept/dismiss because a newer version \
                       was proposed (Protected tier supersede). Use when the user asks \
                       'what revisions are pending', 'show me memories awaiting approval'. \
                       Each item carries target_source_id (the memory being revised: pass \
                       THIS to accept_pending_revision in PR2) and revision_content for \
                       display. Optional `limit` caps results (default 50, max 500).",
        annotations(
            title = "List pending revisions",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_pending_revisions(
        &self,
        Parameters(params): Parameters<ListPendingRevisionsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_pending_revisions_impl(params).await
    }
}

// ===== Schema gating =====

/// Return a copy of `tool` with the `space` field removed from its
/// `inputSchema.properties` (and from `required` if present).
///
/// Called when `WENLAN_SPACE` is locked so the model never sees the field.
/// The runtime guard in `effective_space()` is the load-bearing safety net;
/// this is UX polish on top.
fn strip_space_from_tool_schema(mut tool: Tool) -> Tool {
    let mut schema = (*tool.input_schema).clone();
    if let Some(props) = schema.get_mut("properties").and_then(|v| v.as_object_mut()) {
        props.remove("space");
    }
    if let Some(required) = schema.get_mut("required").and_then(|v| v.as_array_mut()) {
        required.retain(|v| v.as_str() != Some("space"));
    }
    tool.input_schema = std::sync::Arc::new(schema);
    tool
}

/// Deprecated tools remain routable for known callers but are not advertised.
const HIDDEN_LEGACY_TOOL_NAMES: &[&str] = &["context"];

const LOCAL_ONLY_TOOL_NAMES: &[&str] = &[
    "get_lint_agent_work_page",
    "prepare_lint_repair_plan",
    "get_lint_repair_plan_entries",
    "prepare_lint_repair",
    "apply_lint_repair",
    "verify_lint_repair",
    "forget",
    "confirm_memory",
    "delete_page",
    "accept_refinement",
    "reject_refinement",
    "accept_revision",
    "dismiss_revision",
];

impl WenlanMcpServer {
    fn visible_tools(&self) -> Vec<Tool> {
        let mut tools = Self::tool_router().list_all();
        tools.retain(|tool| !HIDDEN_LEGACY_TOOL_NAMES.contains(&tool.name.as_ref()));
        if self.transport == TransportMode::Http {
            tools.retain(|tool| !LOCAL_ONLY_TOOL_NAMES.contains(&tool.name.as_ref()));
        }
        tools
    }

    /// Refuse a `LOCAL_ONLY_TOOL_NAMES` tool reaching this server over HTTP.
    /// Unlike `visible_tools()`, which only hides the name from `list_tools`,
    /// this is consulted by `call_tool` before dispatch, so a name added to
    /// `LOCAL_ONLY_TOOL_NAMES` without a matching inline guard is still refused
    /// rather than merely unlisted.
    fn local_only_refusal(&self, name: &str) -> Option<CallToolResult> {
        (self.transport == TransportMode::Http && LOCAL_ONLY_TOOL_NAMES.contains(&name)).then(
            || {
                CallToolResult::error(vec![Content::text(
                    "This tool is not available over remote connections. Use local stdio MCP on the machine running Wenlan."
                        .to_string(),
                )])
            },
        )
    }
}

// ===== ServerHandler =====

#[tool_handler]
impl ServerHandler for WenlanMcpServer {
    // Overrides the `#[tool_handler]`-generated `call_tool` (which dispatches
    // purely by name via `self.tool_router`) so a `LOCAL_ONLY_TOOL_NAMES` tool
    // is refused before dispatch even if the caller never saw it in
    // `list_tools`. Keep the belt-and-braces inline guards in each `*_impl`
    // for now (removing them requires rewriting the tests that call `*_impl`
    // directly, tools.rs:4405 et al.).
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if let Some(refusal) = self.local_only_refusal(request.name.as_ref()) {
            return Ok(refusal);
        }
        let tcc = ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = self.visible_tools();
        let tools = if crate::lock_state::is_locked() {
            tools
                .into_iter()
                .map(strip_space_from_tool_schema)
                .collect()
        } else {
            tools
        };
        Ok(ListToolsResult {
            tools,
            meta: None,
            next_cursor: None,
        })
    }

    async fn on_initialized(&self, context: NotificationContext<RoleServer>) {
        // Capture client name from MCP initialize handshake
        if let Some(client_info) = context.peer.peer_info() {
            let name = &client_info.client_info.name;
            if !name.is_empty() {
                if let Ok(mut guard) = self.client_name.lock() {
                    tracing::info!("MCP client identified: {}", name);
                    *guard = Some(name.clone());
                }
            }
        }
    }

    fn get_info(&self) -> InitializeResult {
        InitializeResult::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(
            Implementation::new("wenlan-mcp", env!("CARGO_PKG_VERSION"))
        )
        .with_instructions(
            "Wenlan is your personal memory layer — a local knowledge base that persists across sessions and tools.\n\
             Think of yourself as a curator, not a logger. Store insights, not conversation artifacts.\n\n\
             Wenlan is cumulative: each memory you store can be recalled, linked, and distilled into knowledge over time. \
             It's also shared across all the user's tools: what you write, other agents (Claude Desktop, Claude Code, \
             ChatGPT, Cursor, etc.) will read later. Write for any future reader, not just this conversation.\n\n\
             BRIEF: When resuming project work or when the user asks to catch up, call brief for the current Space. \
             Omit topic for the complete Brief alone; provide topic only when same-Space related context is useful. \
             Brief reads never create or mutate state.\n\n\
             STORE PROACTIVELY — don't wait for the user to ask.\n\
             - The user states a preference (\"I use X because...\", \"I prefer Y over Z\")\n\
             - The user makes a decision (\"going with approach A\", \"switching to B\")\n\
             - The user corrects you or prior info (\"actually, it's C, not D\") — store the correction so it sticks\n\
             - The user shares a durable fact about themselves, their work, or people/projects/tools they care about — \
               anchor it to the entity\n\n\
             If the user asks explicitly (\"remember this\", \"save this\", \"don't forget\"), that's a floor — you \
             should have already stored it.\n\n\
             WHEN NOT TO STORE:\n\
             - Conversation filler (\"ok\", \"thanks\", \"let's move on\")\n\
             - Things the user can trivially re-derive (file paths, recent git history)\n\
             - Anything already stored — recall first if unsure\n\
             - Tool output or command results (file contents, git history, build logs) — these are derivable\n\
             - General world facts or documentation that aren't personal to this user (e.g., \"Rust has a borrow \
               checker\", \"PostgreSQL supports JSONB\") — those are not memory material.\n\
             - Your own inferences about the user that they didn't express. Store what they said; infer from that \
               when responding.\n\
             - Agent operating rules — standing \"always X\" / \"never Y\" directives about how an agent should \
               behave (workflow, escalation, tooling). Those are obey-tier instructions for the agent's own config \
               (CLAUDE.md / AGENTS.md / MEMORY.md), not shared memory. Store the user's preference as a fact \
               (\"prefers TDD because…\"); never the agent-facing rule (\"always run TDD first\").\n\n\
             CONTENT QUALITY — this is where you make the biggest difference:\n\
             - Specific beats vague: \"prefers Rust for CLI tools because of compile-time safety\" > \"likes Rust\"\n\
             - Include the WHY: the backend can classify \"dark mode\" as a preference, but only you know\n\
               \"switched to dark mode because of migraines from bright screens\"\n\
             - Name the entities: mention people, projects, tools by name — this powers the knowledge graph\n\
             - Atomic: one idea per memory — \"prefers TDD\" and \"uses pytest\" should be two memories, not one\n\
             - Declarative, not narrative: \"User prefers X because Y\" — not \"User said today they prefer X\". \
               Memories outlive the conversation that produced them.\n\n\
             MEMORY TYPES — omit and trust the backend.\n\n\
             By default, do NOT set memory_type. The backend auto-classifies into identity / preference / \
             decision / lesson / gotcha / fact with more context than you have. Agents that over-specify \
             types tend to pick wrong.\n\n\
             Opt-in specification:\n\
             - \"profile\"   — you're sure it's about the user (identity / preference)\n\
             - \"knowledge\" — you're sure it's about the world (decision / lesson / gotcha / fact)\n\
             - Precise type — only if you're confident and the distinction matters.\n\n\
             EXCEPTION — decisions carry structured fields (alternatives considered, reversibility, domain) \
             that power the Decision Log view. Set memory_type=\"decision\" explicitly ONLY when the user \
             articulated alternatives weighed AND the reasoning for the choice. A bare \"I'm switching to Cursor\" \
             is just a preference change — omit the type. \"Switching to Cursor over VSCode because of better \
             Claude integration, and we can always go back\" — that's a decision.\n\n\
             BRIEF vs RECALL:\n\
             - brief: Space-owned project snapshot; topic optionally adds same-Space related context\n\
             - recall: specific lookup (\"what's Alice's role?\", \"database preferences\", \"our auth decision\")\n\n\
             The backend handles classification, entity extraction, structured fields, quality scoring,\n\
             and dedup — you don't need to replicate that logic. Focus on what only you know:\n\
             the conversational context, why something matters, and what the user actually cares about."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::WenlanClient;
    use crate::types::{
        ChatContextRequest, ChatContextResponse, SearchMemoryRequest, SearchResult,
        StoreMemoryRequest, StoreMemoryResponse,
    };

    fn make_server(
        transport: TransportMode,
        agent_name: &str,
        user_id: Option<&str>,
    ) -> WenlanMcpServer {
        let client = WenlanClient::new("http://127.0.0.1:19999".into());
        WenlanMcpServer::new(
            client,
            transport,
            agent_name.into(),
            user_id.map(String::from),
        )
    }

    fn make_lint_agent_work(seed: u64) -> wenlan_types::lint::LintAgentWork {
        use wenlan_types::lint::{
            LintAgentCandidate, LintAgentRecord, LintAgentRecordKind, LintAgentWork, LintDigest,
            LintSemanticAction, LintSemanticCandidateKind, LintSemanticCheckId,
            LintSemanticPopulation, LintSemanticReasonCode,
        };

        let populations = LintSemanticCheckId::ALL
            .into_iter()
            .map(|check_id| {
                let count = u64::from(check_id == LintSemanticCheckId::MemoryClassification);
                LintSemanticPopulation::try_new(check_id, count, count, count, false).unwrap()
            })
            .collect();
        LintAgentWork::try_new(
            LintDigest::from_u64(seed),
            populations,
            vec![LintAgentRecord::try_new(
                1,
                LintAgentRecordKind::Memory,
                format!("record excerpt {seed}"),
                Some("fact".to_string()),
                None,
                None,
            )
            .unwrap()],
            vec![LintAgentCandidate::try_new(
                1,
                LintSemanticCheckId::MemoryClassification,
                LintSemanticCandidateKind::RecordReview,
                vec![1],
                vec![],
                LintSemanticAction::ReclassifyMemory,
                LintSemanticReasonCode::ClassificationMismatch,
            )
            .unwrap()],
        )
        .unwrap()
    }

    fn make_complete_lint_report(
        profile: wenlan_types::lint::LintProfile,
        scope: wenlan_types::lint::LintScope,
        seed: u64,
        agent_work: Option<wenlan_types::lint::LintAgentWork>,
    ) -> wenlan_types::lint::LintReport {
        use wenlan_types::lint::{
            canonical_check_ids, canonical_gate_effect, LintApplicability, LintCapabilityContext,
            LintCheckResult, LintCheckResultInput, LintConfigFingerprint, LintCoverage,
            LintDbSnapshotMode, LintDbSnapshotReceipt, LintDigest, LintOutcome,
            LintPageSnapshotMode, LintPageSnapshotReceipt, LintPrecondition, LintProducerReceipt,
            LintSeverity, LintSnapshotReceipts, LintSummaryCode, LintValidationMethod,
        };

        let checks = canonical_check_ids(profile)
            .map(|check_id| {
                LintCheckResult::try_new_with_gate_effect(
                    LintCheckResultInput {
                        check_id: check_id.to_string(),
                        outcome: LintOutcome::Pass,
                        severity: LintSeverity::Info,
                        applicability: LintApplicability::Applicable,
                        precondition: LintPrecondition::Ready,
                        coverage: LintCoverage::new(
                            LintValidationMethod::FullEnumeration,
                            0,
                            0,
                            100,
                            false,
                            0,
                        )
                        .unwrap(),
                        metrics: vec![],
                        summary_code: LintSummaryCode::CheckPassed,
                        recommendation_code: None,
                        evidence: vec![],
                        duration_ms: 0,
                    },
                    canonical_gate_effect(profile, check_id).unwrap(),
                )
                .unwrap()
            })
            .collect();
        wenlan_types::lint::LintReport::try_new_for_profile_with_agent_work(
            profile,
            scope,
            LintCapabilityContext::daemon_operator_endpoint(),
            LintSnapshotReceipts::new(
                LintDbSnapshotReceipt::new(
                    LintDbSnapshotMode::TransactionalReadOnly,
                    LintDigest::from_u64(seed),
                    Some(LintDigest::from_u64(seed)),
                ),
                LintPageSnapshotReceipt::new(
                    LintPageSnapshotMode::BestEffort,
                    LintDigest::from_u64(seed + 1),
                    Some(LintDigest::from_u64(seed + 1)),
                ),
            ),
            LintConfigFingerprint::from_effective_config(&[]),
            LintProducerReceipt::new(None),
            checks,
            agent_work,
        )
        .unwrap()
    }

    #[test]
    fn tool_error_surfaces_exact_api_reason() {
        let result = tool_error(
            WenlanError::Api {
                status: 409,
                reason: Some("repair_background_writer_busy".to_string()),
            },
            "repair apply",
        );
        assert_eq!(result.is_error, Some(true));
        match &result.content[0].raw {
            rmcp::model::RawContent::Text(text) => {
                assert!(text.text.contains("HTTP 409"));
                assert!(text.text.contains("repair_background_writer_busy"));
            }
            other => panic!("unexpected tool error content: {other:?}"),
        }
    }

    // ===== call_tool boundary: every LOCAL_ONLY_TOOL_NAMES entry is refused =====
    //
    // Table test for the hole the finder flagged: LOCAL_ONLY_TOOL_NAMES only fed
    // `visible_tools()` (hiding from `list_tools`), so a name added there without
    // a matching hand-rolled `if` guard in its handler stayed fully callable over
    // HTTP. `local_only_refusal` is now the single source of truth `call_tool`
    // consults before dispatch; this asserts every declared name is actually
    // covered, not just the ones someone remembered to guard by hand.

    #[test]
    fn local_only_refusal_covers_every_declared_name_over_http() {
        let server = make_server(TransportMode::Http, "agent", None);
        for name in LOCAL_ONLY_TOOL_NAMES {
            let result = server
                .local_only_refusal(name)
                .unwrap_or_else(|| panic!("expected {name} to be refused over HTTP"));
            let content = &result.content[0];
            match content.raw {
                rmcp::model::RawContent::Text(ref tc) => {
                    assert!(
                        tc.text.contains("not available over remote connections"),
                        "refusal for {name} missing the standard message: {}",
                        tc.text
                    );
                }
                _ => panic!("expected text content for {name}"),
            }
        }
    }

    #[test]
    fn local_only_refusal_allows_every_declared_name_over_stdio() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        for name in LOCAL_ONLY_TOOL_NAMES {
            assert!(
                server.local_only_refusal(name).is_none(),
                "{name} should not be refused over stdio"
            );
        }
    }

    #[test]
    fn local_only_refusal_ignores_unlisted_name_over_http() {
        let server = make_server(TransportMode::Http, "agent", None);
        assert!(server.local_only_refusal("recall").is_none());
    }

    // ===== Transport resolution (existing) =====

    #[test]
    fn test_http_mode_prefers_param_over_agent_name() {
        let server = make_server(TransportMode::Http, "claude.ai", None);
        // Explicit param has highest priority
        let result = server.resolve_source_agent(Some("user-provided".into()));
        assert_eq!(result, Some("user-provided".into()));
    }

    #[test]
    fn test_http_mode_sets_source_agent_when_none() {
        let server = make_server(TransportMode::Http, "chatgpt", None);
        let result = server.resolve_source_agent(None);
        assert_eq!(result, Some("chatgpt".into()));
    }

    #[test]
    fn test_stdio_mode_passes_through_source_agent() {
        let server = make_server(TransportMode::Stdio, "ignored", None);
        let result = server.resolve_source_agent(Some("user-provided".into()));
        assert_eq!(result, Some("user-provided".into()));
    }

    #[test]
    fn test_stdio_mode_falls_back_to_agent_name() {
        let server = make_server(TransportMode::Stdio, "fallback", None);
        // No param, no client_name → falls back to configured agent_name
        let result = server.resolve_source_agent(None);
        assert_eq!(result, Some("fallback".into()));
    }

    #[test]
    fn test_http_mode_resolves_configured_user_id_for_local_use() {
        let server = make_server(TransportMode::Http, "agent", Some("lucian"));
        let result = server.resolve_user_id(None);
        assert_eq!(result, Some("lucian".into()));
    }

    #[test]
    fn test_transport_mode_equality() {
        assert_eq!(TransportMode::Stdio, TransportMode::Stdio);
        assert_eq!(TransportMode::Http, TransportMode::Http);
        assert_ne!(TransportMode::Stdio, TransportMode::Http);
    }

    #[test]
    fn fully_local_tools_are_visible_only_over_stdio() {
        let mut stdio = make_server(TransportMode::Stdio, "agent", None)
            .visible_tools()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        stdio.sort();
        let mut expected = expected_tool_surface()
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(stdio, expected, "stdio must retain the full locked surface");

        let http = make_server(TransportMode::Http, "agent", None)
            .visible_tools()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        for name in [
            "get_lint_agent_work_page",
            "prepare_lint_repair_plan",
            "get_lint_repair_plan_entries",
            "prepare_lint_repair",
            "apply_lint_repair",
            "verify_lint_repair",
            "forget",
            "confirm_memory",
            "delete_page",
            "accept_refinement",
            "reject_refinement",
            "accept_revision",
            "dismiss_revision",
        ] {
            assert!(stdio.iter().any(|candidate| candidate == name));
            assert!(!http.iter().any(|candidate| candidate == name));
        }
    }

    #[tokio::test]
    async fn interleaved_lint_agent_work_packets_remain_pageable_and_digest_bound() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        {
            let mut cache = server.agent_work_cache.lock().unwrap();
            cache.insert(make_lint_agent_work(7));
            cache.insert(make_lint_agent_work(8));
        }

        for digest in ["0000000000000007", "0000000000000008"] {
            let result = server
                .get_lint_agent_work_page_impl(GetLintAgentWorkPageParams {
                    work_digest: digest.to_string(),
                    offset: Some(0),
                    limit: Some(1),
                })
                .await
                .unwrap();
            let value = result.structured_content.unwrap();
            assert_eq!(value["work_digest"], digest);
            assert_eq!(value["total_candidates"], 1);
            assert_eq!(value["candidates"].as_array().unwrap().len(), 1);
            assert_eq!(value["records"].as_array().unwrap().len(), 1);
            assert!(value.get("next_offset").is_none() || value["next_offset"].is_null());
        }

        assert!(server
            .get_lint_agent_work_page_impl(GetLintAgentWorkPageParams {
                work_digest: "0000000000000009".to_string(),
                offset: Some(0),
                limit: Some(1),
            })
            .await
            .is_err());
    }

    #[test]
    fn submitted_lint_agent_work_evicts_only_its_exact_digest() {
        use wenlan_types::lint::LintDigest;

        let server = make_server(TransportMode::Stdio, "agent", None);
        {
            let mut cache = server.agent_work_cache.lock().unwrap();
            cache.insert(make_lint_agent_work(7));
            cache.insert(make_lint_agent_work(8));
        }

        server
            .remove_submitted_agent_work(&LintDigest::from_u64(7))
            .unwrap();

        let cache = server.agent_work_cache.lock().unwrap();
        assert!(cache.get(&LintDigest::from_u64(7)).is_none());
        assert!(cache.get(&LintDigest::from_u64(8)).is_some());
    }

    #[test]
    fn lint_agent_work_cache_replaces_digest_and_evicts_oldest_above_capacity() {
        use wenlan_types::lint::LintDigest;

        let mut cache = LintAgentWorkCache::default();
        for seed in 1..=4 {
            cache.insert(make_lint_agent_work(seed));
        }
        cache.insert(make_lint_agent_work(2));
        assert_eq!(cache.entries.len(), LINT_AGENT_WORK_CACHE_CAPACITY);

        cache.insert(make_lint_agent_work(5));
        assert_eq!(cache.entries.len(), LINT_AGENT_WORK_CACHE_CAPACITY);
        assert!(cache.get(&LintDigest::from_u64(1)).is_none());
        for seed in 2..=5 {
            assert!(cache.get(&LintDigest::from_u64(seed)).is_some());
        }
    }

    #[test]
    fn prepare_lint_repair_plan_schema_requires_scope_only_but_legacy_reports_deserialize() {
        let schema =
            serde_json::to_value(schemars::schema_for!(PrepareLintRepairPlanParams)).unwrap();
        assert_eq!(
            schema["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["lint_scope"]
        );
        assert_eq!(schema["required"], serde_json::json!(["lint_scope"]));

        let scope_only: PrepareLintRepairPlanParams = serde_json::from_value(serde_json::json!({
            "lint_scope": { "kind": "global" }
        }))
        .unwrap();
        assert!(scope_only.general_report.is_none());
        assert!(scope_only.deep_report.is_none());

        let general = make_complete_lint_report(
            wenlan_types::lint::LintProfile::General,
            wenlan_types::lint::LintScope::global(),
            10,
            None,
        );
        let legacy: PrepareLintRepairPlanParams = serde_json::from_value(serde_json::json!({
            "lint_scope": { "kind": "global" },
            "general_report": general
        }))
        .unwrap();
        assert!(legacy.general_report.is_some());
        assert!(legacy.deep_report.is_none());
    }

    #[test]
    fn lint_repair_typed_payloads_are_advertised_as_json_objects() {
        fn advertises_object(
            schema: &serde_json::Value,
            root: &serde_json::Map<String, serde_json::Value>,
        ) -> bool {
            if let Some(reference) = schema.get("$ref").and_then(serde_json::Value::as_str) {
                let Some(definition) = reference.strip_prefix("#/$defs/") else {
                    return false;
                };
                return root
                    .get("$defs")
                    .and_then(|definitions| definitions.get(definition))
                    .is_some_and(|resolved| advertises_object(resolved, root));
            }
            match schema.get("type") {
                Some(serde_json::Value::String(kind)) => kind == "object",
                Some(serde_json::Value::Array(kinds)) => {
                    kinds.iter().any(|kind| kind.as_str() == Some("object"))
                }
                _ => schema
                    .get("anyOf")
                    .and_then(serde_json::Value::as_array)
                    .or_else(|| schema.get("oneOf").and_then(serde_json::Value::as_array))
                    .is_some_and(|variants| {
                        variants
                            .iter()
                            .any(|variant| advertises_object(variant, root))
                    }),
            }
        }

        let tools = make_server(TransportMode::Stdio, "agent", None).visible_tools();
        for (tool_name, fields) in [
            ("prepare_lint_repair", &["general_report", "choice"][..]),
            ("verify_lint_repair", &["general_report", "deep_report"][..]),
        ] {
            let tool = tools
                .iter()
                .find(|tool| tool.name.as_ref() == tool_name)
                .unwrap_or_else(|| panic!("missing tool {tool_name}"));
            let properties = tool.input_schema["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("{tool_name} properties must be an object"));
            for field in fields {
                let schema = properties
                    .get(*field)
                    .unwrap_or_else(|| panic!("{tool_name}.{field} must be advertised"));
                assert!(
                    schema.is_object() && advertises_object(schema, tool.input_schema.as_ref()),
                    "{tool_name}.{field} must advertise a JSON object, got {schema}"
                );
            }
        }
    }

    #[test]
    fn prepare_lint_repair_schema_and_serde_expose_one_exact_tagged_choice() {
        let schema = serde_json::to_value(schemars::schema_for!(PrepareLintRepairParams)).unwrap();
        let properties = schema["properties"].as_object().unwrap();
        assert!(properties.contains_key("choice"));
        assert!(!properties.contains_key("selected_finding"));
        assert!(!properties.contains_key("after_memory_type"));
        let required = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            required,
            ["choice", "general_report", "lint_scope"]
                .into_iter()
                .collect()
        );

        let rename: PrepareLintRepairChoiceParam = serde_json::from_value(serde_json::json!({
            "kind": "rename_page_title",
            "review_id": "lint_review_exact",
            "page_id": "page_exact",
            "before_title": "Before",
            "after_title": "After"
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(rename).unwrap(),
            serde_json::json!({
                "kind": "rename_page_title",
                "review_id": "lint_review_exact",
                "page_id": "page_exact",
                "before_title": "Before",
                "after_title": "After"
            })
        );

        for rejected in [
            serde_json::json!({
                "kind": "rename_page_title",
                "review_id": "lint_review_exact",
                "page_id": "page_exact",
                "before_title": "Before"
            }),
            serde_json::json!({
                "kind": "rename_page_title",
                "review_id": "lint_review_exact",
                "page_id": "page_exact",
                "before_title": "Before",
                "after_title": "After",
                "memory_id": "mem_mixed"
            }),
            serde_json::json!({
                "review_id": "lint_review_exact",
                "page_id": "page_exact",
                "before_title": "Before",
                "after_title": "After"
            }),
        ] {
            assert!(
                serde_json::from_value::<PrepareLintRepairChoiceParam>(rejected).is_err(),
                "partial, mixed, and untagged choices must fail before any HTTP request"
            );
        }
    }

    #[test]
    fn lint_report_cache_is_exact_scope_bounded_and_new_general_resets_deep() {
        use wenlan_types::{
            lint::{LintOpaqueId, LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let mut cache = LintReportCache::default();
        for seed in 1..=4_u64 {
            let scope = RepairLintScope::registered(format!("space-{seed}")).unwrap();
            let report_scope =
                LintScope::registered(LintOpaqueId::from_sorted_position(seed as usize).unwrap());
            cache.record_general(
                scope.clone(),
                make_complete_lint_report(LintProfile::General, report_scope.clone(), seed, None),
            );
            cache.record_final_deep(
                scope.clone(),
                make_complete_lint_report(LintProfile::Deep, report_scope, seed + 10, None),
            );
            assert!(cache.reports_for_plan(&scope).unwrap().1.is_some());
        }

        let first = RepairLintScope::registered("space-1".to_string()).unwrap();
        let fifth = RepairLintScope::registered("space-5".to_string()).unwrap();
        let fifth_report_scope =
            LintScope::registered(LintOpaqueId::from_sorted_position(5).unwrap());
        cache.record_general(
            fifth.clone(),
            make_complete_lint_report(LintProfile::General, fifth_report_scope.clone(), 5, None),
        );
        cache.record_final_deep(
            fifth.clone(),
            make_complete_lint_report(LintProfile::Deep, fifth_report_scope.clone(), 15, None),
        );
        assert_eq!(cache.entries.len(), LINT_REPORT_CACHE_CAPACITY);
        assert!(cache.reports_for_plan(&first).is_none());
        assert!(cache.reports_for_plan(&fifth).unwrap().1.is_some());

        cache.record_general(
            fifth.clone(),
            make_complete_lint_report(LintProfile::General, fifth_report_scope, 25, None),
        );
        assert!(cache.reports_for_plan(&fifth).unwrap().1.is_none());
    }

    #[test]
    fn lint_report_cache_never_treats_work_packet_as_final_deep() {
        use wenlan_types::{
            lint::{LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let scope = RepairLintScope::global();
        let mut cache = LintReportCache::default();
        cache.record_general(
            scope.clone(),
            make_complete_lint_report(LintProfile::General, LintScope::global(), 1, None),
        );
        let token = cache
            .begin_lint_call(&scope, LintProfile::Deep, true, false)
            .unwrap()
            .unwrap();
        assert!(cache.record_from_lint_call(
            &token,
            make_complete_lint_report(
                LintProfile::Deep,
                LintScope::global(),
                2,
                Some(make_lint_agent_work(2)),
            ),
        ));
        assert!(cache.reports_for_plan(&scope).unwrap().1.is_none());
    }

    #[test]
    fn lint_report_cache_preserves_completed_agent_submission_with_work_packet() {
        use wenlan_types::{
            lint::{LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let scope = RepairLintScope::global();
        let mut cache = LintReportCache::default();
        cache.record_general(
            scope.clone(),
            make_complete_lint_report(LintProfile::General, LintScope::global(), 1, None),
        );
        cache.record_final_deep(
            scope.clone(),
            make_complete_lint_report(
                LintProfile::Deep,
                LintScope::global(),
                2,
                Some(make_lint_agent_work(2)),
            ),
        );

        assert!(cache.reports_for_plan(&scope).unwrap().1.is_some());
    }

    #[test]
    fn lint_report_cache_rejects_registered_scope_receipt_mismatch() {
        use wenlan_types::{
            lint::{LintOpaqueId, LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let scope = RepairLintScope::registered("wenlan".to_string()).unwrap();
        let mut cache = LintReportCache::default();
        cache.record_general(
            scope.clone(),
            make_complete_lint_report(
                LintProfile::General,
                LintScope::registered(LintOpaqueId::from_sorted_position(0).unwrap()),
                1,
                None,
            ),
        );
        cache.record_final_deep(
            scope.clone(),
            make_complete_lint_report(
                LintProfile::Deep,
                LintScope::registered(LintOpaqueId::from_sorted_position(1).unwrap()),
                2,
                None,
            ),
        );
        assert!(cache.reports_for_plan(&scope).unwrap().1.is_none());
    }

    #[test]
    fn lint_report_cache_attaches_deep_only_for_completed_agent_assist_flow() {
        use wenlan_types::{
            lint::{LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let scope = RepairLintScope::global();
        let mut cache = LintReportCache::default();
        let general_token = cache
            .begin_lint_call(&scope, LintProfile::General, false, false)
            .unwrap()
            .unwrap();
        assert!(cache.record_from_lint_call(
            &general_token,
            make_complete_lint_report(LintProfile::General, LintScope::global(), 1, None),
        ));
        assert!(cache
            .begin_lint_call(&scope, LintProfile::Deep, false, false)
            .unwrap()
            .is_none());
        assert!(cache.reports_for_plan(&scope).unwrap().1.is_none());

        let deep_token = cache
            .begin_lint_call(&scope, LintProfile::Deep, true, true)
            .unwrap()
            .unwrap();
        assert!(cache.record_from_lint_call(
            &deep_token,
            make_complete_lint_report(LintProfile::Deep, LintScope::global(), 3, None),
        ));
        assert!(cache.reports_for_plan(&scope).unwrap().1.is_some());
    }

    #[test]
    fn lint_report_cache_generation_rejects_overlapping_and_consumed_late_responses() {
        use wenlan_types::{
            lint::{LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let scope = RepairLintScope::global();
        let mut cache = LintReportCache::default();
        let general_a =
            make_complete_lint_report(LintProfile::General, LintScope::global(), 10, None);
        let general_b =
            make_complete_lint_report(LintProfile::General, LintScope::global(), 20, None);

        let token_a = cache
            .begin_lint_call(&scope, LintProfile::General, false, false)
            .unwrap()
            .unwrap();
        let token_b = cache
            .begin_lint_call(&scope, LintProfile::General, false, false)
            .unwrap()
            .unwrap();
        assert!(cache.record_from_lint_call(&token_b, general_b.clone()));
        assert!(!cache.record_from_lint_call(&token_a, general_a.clone()));
        assert_eq!(
            cache.reports_for_plan(&scope).unwrap().0.snapshots(),
            general_b.snapshots()
        );

        let consumed = cache.take_reports_for_plan(&scope).unwrap().unwrap();
        assert_eq!(consumed.0.snapshots(), general_b.snapshots());
        assert!(!cache.record_from_lint_call(&token_a, general_a.clone()));
        assert!(cache.reports_for_plan(&scope).is_none());

        let token_after_miss = cache
            .begin_lint_call(&scope, LintProfile::General, false, false)
            .unwrap()
            .unwrap();
        assert!(cache.take_reports_for_plan(&scope).unwrap().is_none());
        assert!(!cache.record_from_lint_call(&token_after_miss, general_a));
        assert!(cache.reports_for_plan(&scope).is_none());
    }

    #[tokio::test]
    async fn lint_call_start_invalidates_stale_same_scope_reports_before_network_failure() {
        use wenlan_types::{
            lint::{LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let _guard = crate::lock_state::ENV_LOCK.lock().await;
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();
        let server = make_server(TransportMode::Stdio, "agent", None);
        let scope = RepairLintScope::global();
        {
            let mut cache = server.lint_report_cache.lock().unwrap();
            cache.record_general(
                scope.clone(),
                make_complete_lint_report(LintProfile::General, LintScope::global(), 1, None),
            );
            cache.record_final_deep(
                scope.clone(),
                make_complete_lint_report(LintProfile::Deep, LintScope::global(), 2, None),
            );
        }

        let general_failure = server
            .lint_impl(LintParams {
                profile: None,
                space: None,
                agent_assist: false,
                agent_submission: None,
            })
            .await
            .unwrap();
        assert_eq!(general_failure.is_error, Some(true));
        assert!(server
            .lint_report_cache
            .lock()
            .unwrap()
            .reports_for_plan(&scope)
            .is_none());

        {
            let mut cache = server.lint_report_cache.lock().unwrap();
            cache.record_general(
                scope.clone(),
                make_complete_lint_report(LintProfile::General, LintScope::global(), 3, None),
            );
            cache.record_final_deep(
                scope.clone(),
                make_complete_lint_report(LintProfile::Deep, LintScope::global(), 4, None),
            );
        }
        let deep_failure = server
            .lint_impl(LintParams {
                profile: Some(LintProfileParam::Deep),
                space: None,
                agent_assist: true,
                agent_submission: None,
            })
            .await
            .unwrap();
        assert_eq!(deep_failure.is_error, Some(true));
        let cached = server
            .lint_report_cache
            .lock()
            .unwrap()
            .reports_for_plan(&scope)
            .unwrap();
        assert_eq!(cached.0.profile(), LintProfile::General);
        assert!(cached.1.is_none());
    }

    #[tokio::test]
    async fn prepare_lint_repair_plan_rejects_deep_only_and_exact_scope_cache_miss() {
        use wenlan_types::{
            lint::{LintOpaqueId, LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let server = make_server(TransportMode::Stdio, "agent", None);
        let deep_only = server
            .prepare_lint_repair_plan_impl(PrepareLintRepairPlanParams {
                lint_scope: RepairLintScopeParam::Global,
                general_report: None,
                deep_report: Some(make_complete_lint_report(
                    LintProfile::Deep,
                    LintScope::global(),
                    1,
                    None,
                )),
            })
            .await
            .unwrap_err();
        assert!(deep_only
            .message
            .contains("deep_report requires general_report"));

        let wenlan_scope = RepairLintScope::registered("wenlan".to_string()).unwrap();
        let report_scope = LintScope::registered(LintOpaqueId::from_sorted_position(0).unwrap());
        {
            let mut cache = server.lint_report_cache.lock().unwrap();
            cache.record_general(
                wenlan_scope.clone(),
                make_complete_lint_report(LintProfile::General, report_scope.clone(), 2, None),
            );
            cache.record_final_deep(
                wenlan_scope,
                make_complete_lint_report(LintProfile::Deep, report_scope, 3, None),
            );
        }
        let wrong_scope = server
            .prepare_lint_repair_plan_impl(PrepareLintRepairPlanParams {
                lint_scope: RepairLintScopeParam::Registered {
                    space: "other".to_string(),
                },
                general_report: None,
                deep_report: None,
            })
            .await
            .unwrap_err();
        assert!(wrong_scope
            .message
            .contains("General report is not cached for this lint scope"));
    }

    #[tokio::test]
    async fn scope_only_prepare_consumes_cached_reports_before_network_failure() {
        use wenlan_types::{
            lint::{LintProfile, LintScope},
            repair::RepairLintScope,
        };

        let _guard = crate::lock_state::ENV_LOCK.lock().await;
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();
        let server = make_server(TransportMode::Stdio, "agent", None);
        let scope = RepairLintScope::global();
        server.lint_report_cache.lock().unwrap().record_general(
            scope.clone(),
            make_complete_lint_report(LintProfile::General, LintScope::global(), 1, None),
        );

        let first = server
            .prepare_lint_repair_plan_impl(PrepareLintRepairPlanParams {
                lint_scope: RepairLintScopeParam::Global,
                general_report: None,
                deep_report: None,
            })
            .await
            .unwrap();
        assert_eq!(first.is_error, Some(true));
        assert!(server
            .lint_report_cache
            .lock()
            .unwrap()
            .reports_for_plan(&scope)
            .is_none());

        let second = server
            .prepare_lint_repair_plan_impl(PrepareLintRepairPlanParams {
                lint_scope: RepairLintScopeParam::Global,
                general_report: None,
                deep_report: None,
            })
            .await
            .unwrap_err();
        assert!(second
            .message
            .contains("General report is not cached for this lint scope"));
    }

    #[test]
    fn repair_plan_summary_includes_compact_source_report_metadata() {
        use wenlan_types::lint::{
            LintProfile, LintScope, LINT_DEEP_CHECK_COUNT, LINT_GENERAL_CHECK_COUNT,
        };

        let general = make_complete_lint_report(LintProfile::General, LintScope::global(), 1, None);
        let deep = make_complete_lint_report(LintProfile::Deep, LintScope::global(), 2, None);
        let value = add_repair_plan_source_reports(
            serde_json::json!({"plan_id": "plan_1", "entry_count": 0}),
            &general,
            Some(&deep),
        )
        .unwrap();
        assert_eq!(value["plan_id"], "plan_1");
        assert_eq!(value["source_reports"]["general"]["profile"], "general");
        assert_eq!(
            value["source_reports"]["general"]["check_count"],
            u64::try_from(LINT_GENERAL_CHECK_COUNT).unwrap()
        );
        assert_eq!(value["source_reports"]["general"]["complete"], true);
        assert_eq!(value["source_reports"]["deep"]["profile"], "deep");
        assert_eq!(
            value["source_reports"]["deep"]["check_count"],
            u64::try_from(LINT_DEEP_CHECK_COUNT).unwrap()
        );
        assert_eq!(value["source_reports"]["deep"]["complete"], true);
        assert!(value["source_reports"]["deep"].get("totals").is_some());

        let general_only = add_repair_plan_source_reports(
            serde_json::json!({"plan_id": "plan_2", "entry_count": 0}),
            &general,
            None,
        )
        .unwrap();
        assert!(general_only["source_reports"]["deep"].is_null());

        let collision = add_repair_plan_source_reports(
            serde_json::json!({
                "plan_id": "plan_3",
                "entry_count": 0,
                "source_reports": {"daemon": true}
            }),
            &general,
            None,
        );
        assert_eq!(
            collision.unwrap_err(),
            "repair plan summary already contains source_reports"
        );
    }

    #[tokio::test]
    async fn all_lint_repair_execution_calls_refuse_http_before_network() {
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let server = make_server(TransportMode::Http, "agent", None);
        let general = make_complete_lint_report(
            wenlan_types::lint::LintProfile::General,
            wenlan_types::lint::LintScope::global(),
            1,
            None,
        );
        let prepare = server
            .prepare_lint_repair_impl(PrepareLintRepairParams {
                lint_scope: RepairLintScopeParam::Global,
                general_report: general.clone(),
                deep_report: None,
                choice: PrepareLintRepairChoiceParam::CompleteEntityExtraction {
                    review_id: "lint_review_exact".to_string(),
                    memory_id: "mem_exact".to_string(),
                    entity_ids: vec!["entity_exact".to_string()],
                },
            })
            .await
            .unwrap();
        let apply = server
            .apply_lint_repair_impl(ApplyLintRepairParams {
                manifest_id: manifest_id.to_string(),
                approved_manifest_digest: digest.to_string(),
                approval: format!("apply repair {manifest_id} {digest}"),
            })
            .await
            .unwrap();
        let verify = server
            .verify_lint_repair_impl(VerifyLintRepairParams {
                manifest_id: manifest_id.to_string(),
                manifest_digest: digest.to_string(),
                apply_receipt_digest: digest.to_string(),
                general_report: general,
                deep_report: None,
                next_apply: None,
            })
            .await
            .unwrap();
        for result in [prepare, apply, verify] {
            match &result.content[0].raw {
                rmcp::model::RawContent::Text(text) => {
                    assert!(text.text.contains("not available over remote connections"));
                }
                other => panic!("unexpected content: {other:?}"),
            }
        }
    }

    // ===== Param deserialization: CaptureParams =====

    #[test]
    fn test_capture_params_minimal() {
        let json = r#"{"content": "Lucian prefers dark mode"}"#;
        let params: CaptureParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.content, "Lucian prefers dark mode");
        assert!(params.memory_type.is_none());
        assert!(params.space.is_none());
        assert!(params.entity.is_none());
        assert!(params.confidence.is_none());
        assert!(params.supersedes.is_none());
    }

    #[test]
    fn test_capture_params_full() {
        let json = r#"{
            "content": "We chose PostgreSQL over MongoDB",
            "memory_type": "decision",
            "space": "origin",
            "entity": "PostgreSQL",
            "confidence": 0.95,
            "supersedes": "mem_abc123"
        }"#;
        let params: CaptureParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.content, "We chose PostgreSQL over MongoDB");
        assert_eq!(params.memory_type.as_deref(), Some("decision"));
        assert_eq!(params.space.as_deref(), Some("origin"));
        assert_eq!(params.entity.as_deref(), Some("PostgreSQL"));
        assert_eq!(params.confidence, Some(0.95));
        assert_eq!(params.supersedes.as_deref(), Some("mem_abc123"));
    }

    #[test]
    fn test_capture_params_missing_content_fails() {
        let json = r#"{"memory_type": "fact"}"#;
        let result = serde_json::from_str::<CaptureParams>(json);
        assert!(result.is_err());
    }

    // ===== Param deserialization: RecallParams =====

    #[test]
    fn test_recall_params_minimal() {
        let json = r#"{"query": "what does Alice work on?"}"#;
        let params: RecallParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "what does Alice work on?");
        assert!(params.limit.is_none());
        assert!(
            params.rerank.is_none(),
            "rerank omitted must remain None so the daemon receives default false"
        );
    }

    #[test]
    fn test_recall_params_full() {
        let json = r#"{
            "query": "database preferences",
            "limit": 5,
            "memory_type": "decision",
            "space": "origin",
            "rerank": true
        }"#;
        let params: RecallParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "database preferences");
        assert_eq!(params.limit, Some(5));
        assert_eq!(params.memory_type.as_deref(), Some("decision"));
        assert_eq!(params.space.as_deref(), Some("origin"));
        assert_eq!(params.rerank, Some(true));
    }

    #[test]
    fn test_recall_params_limit_as_string() {
        let json = r#"{"query": "test", "limit": "10"}"#;
        let params: RecallParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(10));
    }

    #[test]
    fn test_recall_params_missing_query_fails() {
        let json = r#"{"limit": 5}"#;
        let result = serde_json::from_str::<RecallParams>(json);
        assert!(result.is_err());
    }

    // ===== Param deserialization: ContextParams =====

    #[test]
    fn test_context_params_empty() {
        let json = r#"{}"#;
        let params: ContextParams = serde_json::from_str(json).unwrap();
        assert!(params.topic.is_none());
        assert!(params.limit.is_none());
        assert!(params.space.is_none());
    }

    #[test]
    fn test_context_params_full() {
        let json = r#"{"topic": "project Wenlan architecture", "limit": 30, "space": "work"}"#;
        let params: ContextParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.topic.as_deref(), Some("project Wenlan architecture"));
        assert_eq!(params.limit, Some(30));
        assert_eq!(params.space.as_deref(), Some("work"));
    }

    #[test]
    fn test_context_params_limit_as_string() {
        let json = r#"{"limit": "20"}"#;
        let params: ContextParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(20));
    }

    #[test]
    fn legacy_domain_alias_still_deserializes() {
        // Cached MCP clients (pre-0.7.0 schema) send `"domain"` instead of `"space"`.
        // The serde alias must accept legacy JSON so they don't break for the one-release window.
        let json = r#"{"topic": "project work", "domain": "work"}"#;
        let params: ContextParams =
            serde_json::from_str(json).expect("legacy 'domain' key must deserialize");
        assert_eq!(
            params.space.as_deref(),
            Some("work"),
            "alias must map domain → space"
        );
    }

    #[test]
    fn store_memory_request_serialization_excludes_user_id() {
        let req = StoreMemoryRequest {
            content: "test content".into(),
            memory_type: None,
            space: (None).into(),
            source_agent: Some("test-agent".into()),
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        let obj = json.as_object().unwrap();
        assert!(
            !obj.contains_key("user_id"),
            "user_id must not be on the wire; got: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn capture_success_message_uses_persisted_receipt() {
        let resp = StoreMemoryResponse {
            source_id: "mem_abc".into(),
            chunks_created: 3,
            memory_type: "fact".into(),
            entity_id: Some("ent_xyz".into()),
            quality: Some("high".into()),
            warnings: vec![],
            near_duplicate: None,
            gated: false,
            extraction_method: "llm".into(),
            enrichment: String::new(),
            hint: String::new(),
            space: Some("work".into()),
            space_source: Some(wenlan_types::WriteSpaceSource::Default),
            write_outcome: Some(wenlan_types::WriteOutcome::Created),
        };
        let msg = format_capture_success(&resp);
        assert_eq!(
            msg,
            "Stored mem_abc\nsource_memory_id: mem_abc\nspace: work\nspace_source: default\nwrite_outcome: created"
        );
        assert!(!msg.contains("chunks"));
        assert!(!msg.contains("quality"));
        assert!(!msg.contains("entity"));
    }

    #[test]
    fn capture_success_message_says_gated() {
        let resp = StoreMemoryResponse {
            source_id: "mem_staged".into(),
            chunks_created: 1,
            memory_type: "fact".into(),
            entity_id: None,
            quality: None,
            warnings: vec![],
            near_duplicate: None,
            gated: true,
            extraction_method: "agent".into(),
            enrichment: String::new(),
            hint: String::new(),
            space: None,
            space_source: None,
            write_outcome: Some(wenlan_types::WriteOutcome::Created),
        };
        let msg = format_capture_success(&resp);
        assert!(msg.contains("gated: true"), "got: {msg}");
        assert!(!msg.starts_with("Stored"), "got: {msg}");
    }

    #[test]
    fn capture_success_message_surfaces_warnings() {
        let resp = StoreMemoryResponse {
            source_id: "mem_abc".into(),
            chunks_created: 1,
            memory_type: "decision".into(),
            entity_id: None,
            quality: None,
            warnings: vec!["decision memory missing required 'claim' field".into()],
            near_duplicate: None,
            gated: false,
            extraction_method: "agent".into(),
            enrichment: String::new(),
            hint: String::new(),
            space: None,
            space_source: None,
            write_outcome: None,
        };
        let msg = format_capture_success(&resp);
        assert!(msg.starts_with("Stored mem_abc"));
        assert!(msg.contains("Warnings:"));
        assert!(msg.contains("decision memory missing required 'claim' field"));
    }

    #[test]
    fn format_capture_success_omits_section_when_empty() {
        let resp = StoreMemoryResponse {
            source_id: "mem_new".into(),
            chunks_created: 1,
            memory_type: "fact".into(),
            entity_id: None,
            quality: None,
            warnings: vec![],
            near_duplicate: None,
            gated: false,
            extraction_method: "agent".into(),
            enrichment: String::new(),
            hint: String::new(),
            space: None,
            space_source: None,
            write_outcome: None,
        };
        let out = format_capture_success(&resp);
        assert!(!out.contains("Triggered revisions"));
    }

    #[test]
    fn search_memory_request_serialization_excludes_entity() {
        let req = SearchMemoryRequest {
            query: "test".into(),
            limit: 10,
            memory_type: None,
            space: None,
            source_agent: None,
            rerank: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        let obj = json.as_object().unwrap();
        assert!(
            !obj.contains_key("entity"),
            "entity must not be on the wire; got keys: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn chat_context_request_serialization_includes_domain() {
        #[allow(deprecated)]
        let req = ChatContextRequest {
            query: None,
            conversation_id: Some("topic".into()),
            max_chunks: 20,
            relevance_threshold: None,
            include_goals: true,
            space: Some("work".into()),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["space"], serde_json::json!("work"));
        assert_eq!(json["conversation_id"], serde_json::json!("topic"));
    }

    #[test]
    fn chat_context_response_deserializes_with_profile_and_knowledge() {
        let json = r#"{
            "context": "user is Lucian, prefers Rust",
            "profile": {
                "narrative": "n",
                "identity": ["rust"],
                "preferences": [],
                "goals": []
            },
            "knowledge": {
                "pages": [],
                "decisions": [],
                "relevant_memories": [],
                "graph_context": []
            },
            "took_ms": 42.0,
            "token_estimates": {
                "tier1_identity": 10,
                "tier2_project": 20,
                "tier3_relevant": 30,
                "total": 60
            }
        }"#;
        let parsed: ChatContextResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.context, "user is Lucian, prefers Rust");
        assert_eq!(parsed.profile.identity, vec!["rust"]);
        assert_eq!(parsed.token_estimates.total, 60);
    }

    #[test]
    fn capture_params_structured_fields_schema_is_object() {
        use schemars::schema_for;

        let schema = schema_for!(CaptureParams);
        let json = serde_json::to_value(&schema).unwrap();
        let sf_schema = json
            .pointer("/properties/structured_fields")
            .expect("structured_fields property in schema");
        let type_val = sf_schema
            .pointer("/type")
            .unwrap_or(&serde_json::Value::Null);
        let type_str = match type_val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(","),
            other => panic!(
                "structured_fields schema lacks type constraint; got: {:?}",
                other
            ),
        };
        assert!(
            type_str.contains("object"),
            "expected object type, got: {}",
            type_str
        );
    }

    #[test]
    fn lint_agent_verdict_schema_describes_the_authorized_record_union() {
        let schema = serde_json::to_string(&schemars::schema_for!(LintAgentVerdictParam))
            .expect("LintAgentVerdictParam schema serializes");

        assert!(
            schema.contains(
                "candidate's authorized record refs (`evidence_refs` plus `counterevidence_refs`)"
            ),
            "counterevidence schema must state the full authorized source set: {schema}"
        );
        assert!(
            schema.contains("Do not mechanically copy every evidence ref"),
            "counterevidence schema must require judgment rather than bulk copying: {schema}"
        );
    }

    // ===== Param deserialization: ForgetParams =====

    #[test]
    fn test_forget_params() {
        let json = r#"{"memory_id": "mem_abc123"}"#;
        let params: ForgetParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.memory_id, "mem_abc123");
    }

    #[test]
    fn test_forget_params_missing_id_fails() {
        let json = r#"{}"#;
        let result = serde_json::from_str::<ForgetParams>(json);
        assert!(result.is_err());
    }

    // ===== Request serialization: StoreMemoryRequest =====

    #[test]
    fn test_store_request_includes_new_fields() {
        let req = StoreMemoryRequest {
            content: "test".into(),
            memory_type: Some("decision".into()),
            space: (None).into(),
            source_agent: Some("claude".into()),
            title: None,
            confidence: Some(0.9),
            supersedes: Some("old_id".into()),
            entity: Some("PostgreSQL".into()),
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["entity"], "PostgreSQL");
        assert_eq!(json["supersedes"], "old_id");
        assert!(json["confidence"].as_f64().unwrap() > 0.89);
        assert_eq!(json["source_agent"], "claude");
        assert!(json.get("user_id").is_none());
    }

    #[test]
    fn test_store_request_minimal() {
        let req = StoreMemoryRequest {
            content: "hello".into(),
            memory_type: Some("fact".into()),
            space: (None).into(),
            source_agent: None,
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["content"], "hello");
        assert_eq!(json["memory_type"], "fact");
        assert!(json.get("user_id").is_none());
    }

    // ===== Response deserialization: StoreMemoryResponse =====

    #[test]
    fn test_store_response_with_new_fields() {
        let json = r#"{
            "source_id": "mem_xyz",
            "chunks_created": 2,
            "memory_type": "fact",
            "entity_id": "ent_abc",
            "quality": "high",
            "warnings": ["decision memory missing claim"],
            "extraction_method": "agent"
        }"#;
        let resp: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.source_id, "mem_xyz");
        assert_eq!(resp.chunks_created, 2);
        assert_eq!(resp.memory_type, "fact");
        assert_eq!(resp.entity_id.as_deref(), Some("ent_abc"));
        assert_eq!(resp.quality.as_deref(), Some("high"));
        assert_eq!(resp.warnings, vec!["decision memory missing claim"]);
        assert_eq!(resp.extraction_method, "agent");
    }

    #[test]
    fn test_store_response_backward_compat_no_new_fields() {
        // Old backend response without warnings/extraction_method
        let json = r#"{
            "source_id": "mem_old",
            "chunks_created": 1,
            "memory_type": "fact"
        }"#;
        let resp: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.source_id, "mem_old");
        assert_eq!(resp.chunks_created, 1);
        assert_eq!(resp.memory_type, "fact");
        assert!(resp.entity_id.is_none());
        assert!(resp.quality.is_none());
        assert!(resp.warnings.is_empty());
        assert_eq!(resp.extraction_method, "unknown");
    }

    #[test]
    fn test_store_response_with_warnings_and_extraction_method() {
        let json = r#"{
            "source_id": "mem_xyz",
            "chunks_created": 1,
            "memory_type": "decision",
            "warnings": ["decision memory missing required 'claim' field"],
            "extraction_method": "llm"
        }"#;
        let resp: StoreMemoryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.memory_type, "decision");
        assert_eq!(
            resp.warnings,
            vec!["decision memory missing required 'claim' field"]
        );
        assert_eq!(resp.extraction_method, "llm");
    }

    // ===== Response deserialization: SearchResult =====

    #[test]
    fn test_search_result_with_new_fields() {
        let json = r#"{
            "id": "1",
            "content": "We chose Postgres",
            "source": "memory",
            "source_id": "mem_1",
            "title": "DB decision",
            "url": null,
            "chunk_index": 0,
            "last_modified": 1711000000,
            "score": 0.95,
            "chunk_type": "memory",
            "language": "en",
            "semantic_unit": "sentence",
            "memory_type": "decision",
            "space": "origin",
            "source_agent": "claude",
            "confidence": 0.9,
            "confirmed": true,
            "stability": "standard",
            "supersedes": "mem_0",
            "summary": "DB choice",
            "entity_id": "ent_pg",
            "entity_name": "PostgreSQL",
            "quality": "high",
            "is_archived": false,
            "is_recap": false,
            "source_text": "We chose Postgres",
            "raw_score": 0.42
        }"#;
        let result: SearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.chunk_type.as_deref(), Some("memory"));
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.semantic_unit.as_deref(), Some("sentence"));
        assert_eq!(result.stability.as_deref(), Some("standard"));
        assert_eq!(result.supersedes.as_deref(), Some("mem_0"));
        assert_eq!(result.summary.as_deref(), Some("DB choice"));
        assert_eq!(result.entity_id.as_deref(), Some("ent_pg"));
        assert_eq!(result.entity_name.as_deref(), Some("PostgreSQL"));
        assert_eq!(result.quality.as_deref(), Some("high"));
        assert!(!result.is_archived);
        assert!(!result.is_recap);
        assert_eq!(result.source_text.as_deref(), Some("We chose Postgres"));
        assert!((result.raw_score - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn test_search_result_backward_compat_no_new_fields() {
        // Old backend response without entity/quality/archive/recap
        let json = r#"{
            "id": "1",
            "content": "test",
            "source": "memory",
            "source_id": "mem_1",
            "title": "test",
            "url": null,
            "chunk_index": 0,
            "last_modified": 1711000000,
            "score": 0.8,
            "memory_type": "fact",
            "space": null,
            "source_agent": null,
            "confidence": null,
            "confirmed": null
        }"#;
        let result: SearchResult = serde_json::from_str(json).unwrap();
        assert!(result.entity_id.is_none());
        assert!(result.entity_name.is_none());
        assert!(result.quality.is_none());
        assert!(!result.is_archived);
        assert!(!result.is_recap);
        assert!(result.structured_fields.is_none());
        assert!(result.retrieval_cue.is_none());
        assert_eq!(result.raw_score, 0.0);
    }

    #[test]
    fn test_search_result_with_structured_fields_and_retrieval_cue() {
        let json = r#"{
            "id": "1",
            "content": "Lucian prefers dark mode",
            "source": "memory",
            "source_id": "mem_1",
            "title": "Dark mode preference",
            "url": null,
            "chunk_index": 0,
            "last_modified": 1711000000,
            "score": 0.92,
            "memory_type": "preference",
            "space": null,
            "source_agent": null,
            "confidence": null,
            "confirmed": null,
            "structured_fields": "{\"theme\":\"dark\",\"applies_to\":\"all_apps\"}",
            "retrieval_cue": "What UI theme does Lucian prefer?"
        }"#;
        let result: SearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(
            result.structured_fields.as_deref(),
            Some("{\"theme\":\"dark\",\"applies_to\":\"all_apps\"}")
        );
        assert_eq!(
            result.retrieval_cue.as_deref(),
            Some("What UI theme does Lucian prefer?")
        );
        assert!(!result.is_archived);
        assert!(!result.is_recap);
        assert_eq!(result.raw_score, 0.0);
    }

    #[test]
    fn test_search_result_knowledge_graph_source() {
        // Entity-boosted observation results from knowledge graph
        let json = r#"{
            "id": "obs_1",
            "content": "Prefers Rust over Go",
            "source": "knowledge_graph",
            "source_id": "ent_lucian",
            "title": "Lucian",
            "url": null,
            "chunk_index": 0,
            "last_modified": 1711000000,
            "score": 1.14,
            "memory_type": null,
            "space": null,
            "source_agent": null,
            "confidence": null,
            "confirmed": null,
            "entity_id": "ent_lucian",
            "entity_name": "Lucian"
        }"#;
        let result: SearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.source, "knowledge_graph");
        assert_eq!(result.entity_id.as_deref(), Some("ent_lucian"));
        assert_eq!(result.entity_name.as_deref(), Some("Lucian"));
        assert!(!result.is_archived);
        assert!(!result.is_recap);
        assert_eq!(result.raw_score, 0.0);
    }

    // ===== Transport security: forget blocks on HTTP =====

    #[tokio::test]
    async fn test_forget_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let result = server.forget_impl("mem_123").await.unwrap();
        // Should return error content, not an Err
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_forget_allowed_on_stdio_transport() {
        // This will fail with connection error (no server), which proves
        // the transport check passed and it tried to make the HTTP call.
        // The error comes back as CallToolResult with is_error: true
        // (tool-level failure), not McpError (protocol-level).
        let server = make_server(TransportMode::Stdio, "agent", None);
        let result = server.forget_impl("mem_123").await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    // ===== Transport security: revision wrappers block on HTTP =====

    #[tokio::test]
    async fn test_accept_revision_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let req = AcceptRevisionRequest {
            target_source_id: "mem_x".into(),
        };
        let result = server.accept_revision_impl(req).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_accept_revision_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let req = AcceptRevisionRequest {
            target_source_id: "mem_x".into(),
        };
        let result = server.accept_revision_impl(req).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_dismiss_revision_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let req = DismissRevisionRequest {
            target_source_id: "mem_x".into(),
        };
        let result = server.dismiss_revision_impl(req).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_dismiss_revision_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let req = DismissRevisionRequest {
            target_source_id: "mem_x".into(),
        };
        let result = server.dismiss_revision_impl(req).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_write_page_refresh_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = WritePageParams {
            page_id: Some("page_x".into()),
            title: None,
            content: "body".into(),
            source_memory_ids: vec!["mem_a".into()],
            summary: None,
            entity_id: None,
            space: None,
        };
        let result = server.write_page_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_write_page_refresh_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = WritePageParams {
            page_id: Some("page_x".into()),
            title: None,
            content: "body".into(),
            source_memory_ids: vec!["mem_a".into()],
            summary: None,
            entity_id: None,
            space: None,
        };
        let result = server.write_page_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[test]
    fn create_page_reports_attached_to_existing_page_truthfully() {
        let text = format_create_page_response(CreatePageResponse {
            id: "page_existing".into(),
            attached_to: Some("page_existing".into()),
            warnings: Vec::new(),
            space: Some("work".into()),
            space_source: Some(wenlan_types::WriteSpaceSource::Existing),
            write_outcome: Some(wenlan_types::WriteOutcome::AttachedExisting),
        });

        assert!(
            text.starts_with("Attached to existing page page_existing"),
            "attached response must not claim a new page was created: {text}"
        );
        assert!(
            !text.contains("Created page"),
            "attached response must distinguish reuse from creation: {text}"
        );
        assert!(text.contains("space: work"));
        assert!(text.contains("space_source: existing"));
        assert!(text.contains("write_outcome: attached_existing"));
    }

    #[tokio::test]
    async fn write_page_refresh_rejects_empty_sources_before_network() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            server.write_page_impl(WritePageParams {
                page_id: Some("page_x".into()),
                title: None,
                content: "body".into(),
                source_memory_ids: Vec::new(),
                summary: None,
                entity_id: None,
                space: None,
            }),
        )
        .await
        .expect("empty-source guard must return before any daemon retry")
        .unwrap();
        let text = match &result.content[0].raw {
            rmcp::model::RawContent::Text(text) => &text.text,
            other => panic!("expected text content, got {other:?}"),
        };
        assert!(
            result.is_error.unwrap_or(false) && text.contains("source_memory_ids is required"),
            "empty refresh sources must fail locally before any daemon call: {text}"
        );
    }

    #[tokio::test]
    async fn test_write_page_refresh_surfaces_gated_revision_card_fields() {
        let text = format_update_page_response(
            "page_x",
            wenlan_types::responses::PageWriteResponse {
                ok: true,
                gated: true,
                revision_card_id: Some("mem_page_card_1".to_string()),
            },
        );
        assert!(
            text.contains("gated: true"),
            "gated field missing from write_page refresh response: {text}"
        );
        assert!(
            text.contains("revision_card_id: mem_page_card_1"),
            "revision_card_id missing from write_page refresh response: {text}"
        );
    }

    // ===== Context default limit =====

    #[test]
    fn test_context_request_default_limit() {
        let params = ContextParams {
            topic: Some("test".into()),
            limit: None,
            space: None,
        };
        #[allow(deprecated)]
        let req = ChatContextRequest {
            query: None,
            conversation_id: params.topic,
            max_chunks: params.limit.unwrap_or(20),
            relevance_threshold: None,
            include_goals: true,
            space: params.space,
        };
        assert_eq!(req.max_chunks, 20);
    }

    #[test]
    fn test_context_request_custom_limit() {
        let params = ContextParams {
            topic: None,
            limit: Some(5),
            space: Some("work".into()),
        };
        #[allow(deprecated)]
        let req = ChatContextRequest {
            query: None,
            conversation_id: params.topic,
            max_chunks: params.limit.unwrap_or(20),
            relevance_threshold: None,
            include_goals: true,
            space: params.space,
        };
        assert_eq!(req.max_chunks, 5);
        assert_eq!(req.space.as_deref(), Some("work"));
    }

    #[test]
    fn test_context_maps_topic_to_conversation_id() {
        let params = ContextParams {
            topic: Some("project Wenlan".into()),
            limit: None,
            space: None,
        };
        #[allow(deprecated)]
        let req = ChatContextRequest {
            query: None,
            conversation_id: params.topic.clone(),
            max_chunks: params.limit.unwrap_or(20),
            relevance_threshold: None,
            include_goals: true,
            space: params.space,
        };
        assert_eq!(req.conversation_id.as_deref(), Some("project Wenlan"));
    }

    // ===== Remember request construction =====

    #[test]
    fn test_capture_constructs_store_request_with_entity() {
        let server = make_server(TransportMode::Stdio, "claude", None);
        let params = CaptureParams {
            content: "Alice manages the frontend team".into(),
            memory_type: Some("fact".into()),
            space: Some("work".into()),
            entity: Some("Alice".into()),
            confidence: Some(0.9),
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        };

        // Replicate capture_impl's request construction
        let source_agent = server.resolve_source_agent(None);

        let req = StoreMemoryRequest {
            content: params.content,
            memory_type: params.memory_type,
            space: (params.space).into(),
            source_agent,
            title: None,
            confidence: params.confidence,
            supersedes: params.supersedes,
            entity: params.entity,
            entity_id: None,
            structured_fields: params.structured_fields.map(serde_json::Value::Object),
            retrieval_cue: params.retrieval_cue,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["content"], "Alice manages the frontend team");
        assert_eq!(json["memory_type"], "fact");
        assert_eq!(json["space"], "work");
        assert_eq!(json["entity"], "Alice");
        assert!(json["confidence"].as_f64().unwrap() > 0.89);
        // stdio mode: no param, no client_name → falls back to agent_name "claude"
        assert_eq!(json["source_agent"], "claude");
    }

    #[test]
    fn test_remember_http_mode_injects_agent() {
        let server = make_server(TransportMode::Http, "claude.ai", Some("lucian"));
        let source_agent = server.resolve_source_agent(None);

        assert_eq!(source_agent, Some("claude.ai".into()));
    }

    // ===== Recall request construction =====

    #[test]
    fn test_recall_constructs_search_request() {
        let params = RecallParams {
            query: "database choices".into(),
            limit: Some(5),
            memory_type: Some("decision".into()),
            space: None,
            rerank: None,
        };

        let req = SearchMemoryRequest {
            query: params.query,
            limit: params.limit.unwrap_or(10),
            memory_type: params.memory_type,
            space: params.space,
            source_agent: None,
            rerank: params.rerank.unwrap_or(false),
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["query"], "database choices");
        assert_eq!(json["limit"], 5);
        assert_eq!(json["memory_type"], "decision");
        assert!(json.get("entity").is_none());
        assert!(json["space"].is_null());
        assert!(json["source_agent"].is_null());
        assert_eq!(json["rerank"], false);
    }

    #[test]
    fn test_recall_forwards_rerank_flag() {
        // When the caller passes rerank: Some(true), the constructed
        // SearchMemoryRequest must carry rerank=true through to the daemon.
        let params = RecallParams {
            query: "database choices".into(),
            limit: None,
            memory_type: None,
            space: None,
            rerank: Some(true),
        };

        let req = SearchMemoryRequest {
            query: params.query,
            limit: params.limit.unwrap_or(10),
            memory_type: params.memory_type,
            space: params.space,
            source_agent: None,
            rerank: params.rerank.unwrap_or(false),
        };

        assert!(
            req.rerank,
            "RecallParams.rerank=Some(true) must flow through to SearchMemoryRequest.rerank=true"
        );
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["rerank"], true);
    }

    #[test]
    fn test_recall_params_schema_advertises_rerank() {
        // The schemars-derived JSON Schema for RecallParams must advertise
        // the rerank field so MCP clients (Claude Desktop, Cursor, etc.) see
        // it as an available parameter.
        let params_schema = serde_json::to_string(&schemars::schema_for!(RecallParams))
            .expect("RecallParams schema serializes");
        assert!(
            params_schema.contains("rerank"),
            "RecallParams schema must advertise the `rerank` field, got: {params_schema}"
        );
        assert!(
            params_schema.contains("cross-encoder"),
            "RecallParams.rerank description must mention cross-encoder so models understand the tradeoff, got: {params_schema}"
        );
    }

    // ===== Memory type pass-through =====

    /// CaptureParams must pass every canonical memory_type through to the
    /// daemon verbatim. The MCP layer is dumb wire — it doesn't validate or
    /// rewrite the value; the daemon owns that. Drift test sourced from
    /// `MemoryType::all_values()` so adding a variant extends coverage
    /// automatically.
    #[test]
    fn test_capture_passes_through_all_canonical_types() {
        for t in wenlan_types::MemoryType::all_values() {
            let params = CaptureParams {
                content: "test".into(),
                memory_type: Some((*t).to_string()),
                space: None,
                entity: None,
                confidence: None,
                supersedes: None,
                structured_fields: None,
                retrieval_cue: None,
            };
            assert_eq!(params.memory_type.as_deref(), Some(*t));
        }
    }

    /// Legacy "goal" alias still flows through the wire untouched —
    /// `MemoryType::FromStr` folds it to "identity" daemon-side. The MCP
    /// layer must not pre-reject it (the daemon owns the fold decision).
    #[test]
    fn test_capture_passes_through_legacy_goal_alias() {
        let params = CaptureParams {
            content: "test".into(),
            memory_type: Some("goal".into()),
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        };
        assert_eq!(params.memory_type.as_deref(), Some("goal"));
    }

    // ===== Structured fields in remember params =====

    #[test]
    fn test_capture_params_with_structured_fields_and_cue() {
        let json = r#"{
            "content": "Lucian prefers dark mode",
            "structured_fields": {"theme":"dark"},
            "retrieval_cue": "What theme does Lucian prefer?"
        }"#;
        let params: CaptureParams = serde_json::from_str(json).unwrap();
        let structured_fields = params.structured_fields.expect("structured_fields");
        assert_eq!(
            structured_fields.get("theme"),
            Some(&serde_json::Value::String("dark".into()))
        );
        assert_eq!(
            params.retrieval_cue.as_deref(),
            Some("What theme does Lucian prefer?")
        );
    }

    #[test]
    fn test_store_request_with_structured_fields() {
        let req = StoreMemoryRequest {
            content: "test".into(),
            memory_type: Some("fact".into()),
            space: (None).into(),
            source_agent: None,
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: Some(serde_json::json!({"key":"val"})),
            retrieval_cue: Some("What is the key?".into()),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["structured_fields"], serde_json::json!({"key":"val"}));
        assert_eq!(json["retrieval_cue"], "What is the key?");
    }

    // ===== ChatContextResponse deserialization =====

    #[test]
    fn test_chat_context_response() {
        let json = r#"{
            "context": "User prefers dark mode. Works on Wenlan project.",
            "profile": {
                "narrative": "narrative",
                "identity": [],
                "preferences": [],
                "goals": []
            },
            "knowledge": {
                "pages": [],
                "decisions": [],
                "relevant_memories": [],
                "graph_context": []
            },
            "took_ms": 12.5,
            "token_estimates": {
                "tier1_identity": 1,
                "tier2_project": 2,
                "tier3_relevant": 3,
                "total": 6
            }
        }"#;
        let resp: ChatContextResponse = serde_json::from_str(json).unwrap();
        assert!(!resp.context.is_empty());
        assert!(resp.profile.identity.is_empty());
        assert_eq!(resp.took_ms, 12.5);
        assert_eq!(resp.token_estimates.total, 6);
    }

    #[test]
    fn test_chat_context_response_empty() {
        let json = r#"{
            "context": "",
            "profile": {
                "narrative": "",
                "identity": [],
                "preferences": [],
                "goals": []
            },
            "knowledge": {
                "pages": [],
                "decisions": [],
                "relevant_memories": [],
                "graph_context": []
            },
            "took_ms": 1.0,
            "token_estimates": {
                "tier1_identity": 0,
                "tier2_project": 0,
                "tier3_relevant": 0,
                "total": 0
            }
        }"#;
        let resp: ChatContextResponse = serde_json::from_str(json).unwrap();
        assert!(resp.context.is_empty());
    }

    // ===== with_instructions content assertions =====
    // These tests lock in the refined agent-facing guidance. If any
    // assertion fails, either the rule was intentionally changed
    // (update the test) or the refinement was accidentally dropped
    // (restore the rule).

    fn server_instructions() -> String {
        let s = make_server(TransportMode::Stdio, "test", None);
        s.get_info()
            .instructions
            .expect("server must ship with_instructions")
    }

    #[test]
    fn instructions_mention_cumulative_knowledge() {
        assert!(
            server_instructions().contains("cumulative"),
            "with_instructions must describe Wenlan as cumulative"
        );
    }

    #[test]
    fn instructions_mention_shared_across_tools() {
        assert!(
            server_instructions().contains("shared across all"),
            "with_instructions must tell agents the store is shared across tools"
        );
    }

    #[test]
    fn instructions_define_brief_recall_boundary() {
        let instructions = server_instructions();
        assert!(instructions.contains("BRIEF vs RECALL"));
        assert!(instructions.contains("Brief reads never create or mutate state"));
        assert!(!instructions.contains("FIRST THING EVERY SESSION"));
    }

    #[test]
    fn instructions_use_proactive_framing() {
        assert!(
            server_instructions().contains("STORE PROACTIVELY"),
            "with_instructions must use STORE PROACTIVELY framing (not passive WHEN TO STORE)"
        );
    }

    #[test]
    fn instructions_ban_tool_output_storage() {
        assert!(
            server_instructions().contains("Tool output or command results"),
            "with_instructions must explicitly rule out tool output as storage material"
        );
    }

    #[test]
    fn instructions_ban_ghost_inferences() {
        assert!(
            server_instructions().contains("Your own inferences"),
            "with_instructions must rule out storing agent's own inferences user didn't express"
        );
    }

    #[test]
    fn instructions_call_out_atomic_memory() {
        assert!(
            server_instructions().contains("Atomic: one idea per memory"),
            "with_instructions must call out the atomic-memory rule explicitly by name"
        );
    }

    #[test]
    fn instructions_specify_declarative_writing() {
        assert!(
            server_instructions().contains("Declarative, not narrative"),
            "with_instructions must require declarative (not narrative) writing style"
        );
    }

    #[test]
    fn instructions_default_to_omit_memory_type() {
        let i = server_instructions();
        assert!(
            i.contains("omit and trust the backend"),
            "with_instructions must default agents to omitting memory_type"
        );
        assert!(
            i.contains("do NOT set memory_type"),
            "with_instructions must explicitly say do NOT set memory_type by default"
        );
    }

    #[test]
    fn instructions_list_every_canonical_memory_type() {
        let i = server_instructions();
        for ty in wenlan_types::MemoryType::all_values() {
            assert!(
                contains_word(&i, ty),
                "with_instructions must list canonical memory type \"{ty}\" so MCP clients see the full vocabulary",
            );
        }
    }

    #[test]
    fn instructions_omit_legacy_goal_type() {
        let i = server_instructions();
        // "goal" (singular) is a legacy memory_type folded to Identity by
        // MemoryType::FromStr. The plural English noun "goals" (life goals,
        // profile.goals chat-context field) is a separate concern and must
        // NOT trigger this test — tokenizing on word boundaries lets one
        // through while still catching the legacy memory-type token.
        assert!(
            !contains_word(&i, "goal"),
            "with_instructions must not advertise legacy \"goal\" memory_type"
        );
    }

    /// Tokenize on non-alphanumeric boundaries and check whether `needle`
    /// appears as a standalone token. Mirrors the helper used by the
    /// wenlan-types drift tests so "goals" (plural noun) does not false-match
    /// the legacy "goal" memory_type token.
    fn contains_word(haystack: &str, needle: &str) -> bool {
        haystack
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .any(|tok| tok == needle)
    }

    #[test]
    fn instructions_carve_out_decisions_for_decision_log() {
        let i = server_instructions();
        assert!(
            i.contains("Decision Log"),
            "with_instructions must name the Decision Log as the reason for explicit decision typing"
        );
        assert!(
            i.contains("memory_type=\"decision\""),
            "with_instructions must tell agents to set memory_type=\"decision\" explicitly for decisions"
        );
    }

    // ===== tool-level and param-level description assertions =====

    fn tool_descriptions() -> std::collections::HashMap<String, String> {
        let server = make_server(TransportMode::Stdio, "test", None);
        server
            .tool_router
            .list_all()
            .into_iter()
            .filter_map(|t| {
                let desc = t.description.as_ref()?.to_string();
                Some((t.name.to_string(), desc))
            })
            .collect()
    }

    #[test]
    fn capture_description_calls_out_atomic() {
        let descriptions = tool_descriptions();
        let capture = descriptions.get("capture").expect("capture tool exists");
        assert!(
            capture.contains("Each call is one atomic idea"),
            "capture description must call out atomic-per-call explicitly, got: {capture}"
        );
    }

    #[test]
    fn brief_description_defines_topic_boundary() {
        let descriptions = tool_descriptions();
        let brief = descriptions.get("brief").expect("brief tool exists");
        assert!(
            brief.contains("Omit `topic`") && brief.contains("same Space"),
            "brief description must distinguish the full Brief from topic-related context: {brief}"
        );
    }

    #[test]
    fn recall_memory_type_param_lists_two_level_filter() {
        let params_schema = serde_json::to_string(&schemars::schema_for!(RecallParams))
            .expect("RecallParams schema serializes");
        assert!(
            params_schema.contains("Two-level filter"),
            "RecallParams.memory_type must advertise the two-level filter, got schema: {params_schema}"
        );
        assert!(
            params_schema.contains("profile"),
            "RecallParams.memory_type must mention profile alias"
        );
        assert!(
            params_schema.contains("knowledge"),
            "RecallParams.memory_type must mention knowledge alias"
        );
    }

    // ===== Explicit KG writes =====

    #[test]
    fn create_entity_params_require_name_and_type() {
        let params: CreateEntityParams =
            serde_json::from_value(serde_json::json!({"name": "Alice", "entity_type": "person"}))
                .unwrap();
        assert_eq!(params.name, "Alice");
        assert_eq!(params.entity_type, "person");
        assert!(params.space.is_none());
        assert!(params.confidence.is_none());
        assert!(serde_json::from_value::<CreateEntityParams>(
            serde_json::json!({"entity_type": "person"})
        )
        .is_err());
        assert!(
            serde_json::from_value::<CreateEntityParams>(serde_json::json!({"name": "Alice"}))
                .is_err()
        );
    }

    #[test]
    fn create_relation_params_require_resolved_ids_and_source_memory() {
        let params: CreateRelationParams = serde_json::from_value(serde_json::json!({
            "from_entity_id": "ent_alice",
            "to_entity_id": "ent_wenlan",
            "relation_type": "works_on",
            "source_memory_id": "mem_user_statement"
        }))
        .unwrap();
        assert_eq!(params.from_entity_id, "ent_alice");
        assert_eq!(params.to_entity_id, "ent_wenlan");
        assert_eq!(params.relation_type, "works_on");
        assert_eq!(params.source_memory_id, "mem_user_statement");

        assert!(
            serde_json::from_value::<CreateRelationParams>(serde_json::json!({
                "from_entity_id": "ent_alice",
                "to_entity_id": "ent_wenlan",
                "relation_type": "works_on"
            }))
            .is_err(),
            "source_memory_id must be required"
        );
        assert!(
            serde_json::from_value::<CreateRelationParams>(serde_json::json!({
                "from_entity": "Alice",
                "to_entity": "Wenlan",
                "relation_type": "works_on",
                "source_memory_id": "mem_user_statement"
            }))
            .is_err(),
            "entity names must not satisfy the id-based contract"
        );
    }

    #[test]
    fn create_relation_request_maps_ids_and_source_memory() {
        let params = CreateRelationParams {
            from_entity_id: "ent_alice".into(),
            to_entity_id: "ent_wenlan".into(),
            relation_type: "works_on".into(),
            source_memory_id: "mem_user_statement".into(),
        };
        let req = CreateRelationRequest {
            from_entity: params.from_entity_id,
            to_entity: params.to_entity_id,
            relation_type: params.relation_type,
            source_agent: Some("claude".into()),
            confidence: None,
            explanation: None,
            source_memory_id: Some(params.source_memory_id),
            span: None,
            model_version: None,
            prompt_version: None,
        };
        let json = serde_json::to_value(req).unwrap();
        assert_eq!(json["from_entity"], "ent_alice");
        assert_eq!(json["to_entity"], "ent_wenlan");
        assert_eq!(json["relation_type"], "works_on");
        assert_eq!(json["source_memory_id"], "mem_user_statement");
        assert_eq!(json["source_agent"], "claude");
    }

    #[test]
    fn kg_write_outputs_do_not_claim_new_creation() {
        let entity = format_entity_ready_response(CreateEntityResponse {
            id: "ent_alice".into(),
            warnings: vec!["resolved existing canonical entity".into()],
            space: None,
            space_source: Some(wenlan_types::WriteSpaceSource::Uncategorized),
            write_outcome: Some(wenlan_types::WriteOutcome::ResolvedExisting),
        });
        assert!(entity.starts_with("Entity ent_alice ready"));
        assert!(!entity.contains("Created"));
        assert!(entity.contains("warning: resolved existing canonical entity"));
        assert!(entity.contains("space: Uncategorized"));
        assert!(entity.contains("write_outcome: resolved_existing"));

        let relation = format_relation_ready_response(CreateRelationResponse {
            id: "rel_alice_wenlan".into(),
            warnings: Vec::new(),
        });
        assert_eq!(relation, "Relation rel_alice_wenlan ready");
        assert!(!relation.contains("Created"));
    }

    // ===== Refinement queue guards =====

    #[tokio::test]
    async fn test_reject_refinement_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let result = server
            .reject_refinement_impl(RejectRefinementParams {
                id: "merge_abc_def".into(),
            })
            .await
            .unwrap();
        let rmcp::model::RawContent::Text(text) = &result.content[0].raw else {
            panic!("expected text content");
        };
        assert!(text.text.contains("not available over remote connections"));
    }

    #[tokio::test]
    async fn test_reject_refinement_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let result = server
            .reject_refinement_impl(RejectRefinementParams {
                id: "merge_abc_def".into(),
            })
            .await
            .unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_accept_refinement_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let result = server
            .accept_refinement_impl(AcceptRefinementParams {
                id: "merge_abc_def".into(),
                space: None,
            })
            .await
            .unwrap();
        let rmcp::model::RawContent::Text(text) = &result.content[0].raw else {
            panic!("expected text content");
        };
        assert!(text.text.contains("not available over remote connections"));
    }

    #[tokio::test]
    async fn test_accept_refinement_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let result = server
            .accept_refinement_impl(AcceptRefinementParams {
                id: "merge_abc_def".into(),
                space: None,
            })
            .await
            .unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[test]
    fn list_refinements_description_mentions_vocab_promote() {
        let descriptions = tool_descriptions();
        let list = descriptions
            .get("list_refinements")
            .expect("list_refinements tool exists");
        assert!(
            list.contains("vocab_promote"),
            "list_refinements description must enumerate vocab_promote, got: {list}"
        );
    }

    #[test]
    fn accept_refinement_response_typed_deserialize() {
        let raw = r#"{"id":"ref_xyz","action_applied":"entity_merge"}"#;
        let parsed: AcceptRefinementResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.id, "ref_xyz");
        assert_eq!(parsed.action_applied, "entity_merge");
    }

    #[test]
    fn accept_refinement_response_rejects_extra_envelope() {
        let wrong = r#"{"data":{"id":"ref_xyz","action_applied":"entity_merge"}}"#;
        let result: Result<AcceptRefinementResponse, _> = serde_json::from_str(wrong);
        assert!(
            result.is_err(),
            "envelope-wrapped response must fail typed deserialize"
        );
    }

    // ===== Page CRUD =====

    #[test]
    fn write_page_params_accept_create_and_refresh_shapes() {
        let create: WritePageParams = serde_json::from_value(serde_json::json!({
            "title": "T",
            "content": "body"
        }))
        .unwrap();
        assert!(create.page_id.is_none());
        assert_eq!(create.title.as_deref(), Some("T"));
        assert!(create.source_memory_ids.is_empty());

        let refresh: WritePageParams = serde_json::from_value(serde_json::json!({
            "page_id": "page_abc",
            "content": "body",
            "source_memory_ids": ["mem_1"]
        }))
        .unwrap();
        assert_eq!(refresh.page_id.as_deref(), Some("page_abc"));
        assert!(refresh.title.is_none());
        assert_eq!(refresh.source_memory_ids, vec!["mem_1"]);
    }

    // --- DeletePageParams ---

    #[test]
    fn test_delete_page_params() {
        let json = r#"{"page_id": "page_abc"}"#;
        let params: DeletePageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.page_id, "page_abc");
    }

    #[test]
    fn test_delete_page_params_missing_fails() {
        let json = r#"{}"#;
        let result = serde_json::from_str::<DeletePageParams>(json);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_page_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let result = server.delete_page_impl("page_123").await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_delete_page_allowed_on_stdio_transport() {
        // No daemon running → falls through to connection error (not transport block).
        let server = make_server(TransportMode::Stdio, "agent", None);
        let result = server.delete_page_impl("page_123").await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    // --- Tool registration ---

    #[test]
    fn new_crud_tools_are_registered() {
        let descriptions = tool_descriptions();
        for name in ["write_page", "delete_page"] {
            assert!(
                descriptions.contains_key(name),
                "tool `{name}` must be registered, got: {:?}",
                descriptions.keys().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn explicit_kg_write_tools_advertise_source_backed_narrow_contract() {
        let tools = WenlanMcpServer::tool_router().list_all();
        let entity = tools
            .iter()
            .find(|tool| tool.name == "create_entity")
            .expect("create_entity must be registered");
        assert!(
            entity
                .description
                .as_deref()
                .is_some_and(|description| description.contains("explicitly establishes")),
            "create_entity must advertise its narrow explicit-user trigger"
        );

        let relation = tools
            .iter()
            .find(|tool| tool.name == "create_relation")
            .expect("create_relation must be registered");
        let description = relation.description.as_deref().unwrap_or_default();
        assert!(
            description.contains("Never infer an unstated relation"),
            "create_relation must prohibit inferred writes: {description}"
        );
        let schema = serde_json::Value::Object((*relation.input_schema).clone());
        let required = schema["required"]
            .as_array()
            .expect("create_relation schema must declare required fields");
        for field in [
            "from_entity_id",
            "to_entity_id",
            "relation_type",
            "source_memory_id",
        ] {
            assert!(
                required.iter().any(|value| value == field),
                "create_relation must require {field}: {schema}"
            );
        }
        assert!(
            schema["properties"].get("from_entity").is_none()
                && schema["properties"].get("to_entity").is_none(),
            "create_relation must take resolved ids, not entity names: {schema}"
        );
    }

    #[test]
    fn unique_read_only_detail_tools_are_registered_for_explicit_queries() {
        let tools = WenlanMcpServer::tool_router().list_all();
        for (name, trigger) in [
            ("get_memory_revisions", "history"),
            ("get_page_revisions", "changelog"),
            ("list_pending_imports", "import progress"),
            ("list_rejections", "missing capture"),
        ] {
            let tool = tools
                .iter()
                .find(|tool| tool.name == name)
                .unwrap_or_else(|| panic!("{name} must be registered"));
            assert_eq!(
                tool.annotations
                    .as_ref()
                    .and_then(|annotations| annotations.read_only_hint),
                Some(true),
                "{name} must remain read-only"
            );
            assert!(
                tool.description
                    .as_deref()
                    .is_some_and(|description| description.contains(trigger)),
                "{name} must advertise explicit trigger phrase {trigger:?}"
            );
        }
    }

    #[test]
    fn unique_detail_tool_params_enforce_ids_and_accept_filters() {
        let memory: GetMemoryRevisionsParams =
            serde_json::from_value(serde_json::json!({"memory_id": "mem_1"})).unwrap();
        assert_eq!(memory.memory_id, "mem_1");
        assert!(serde_json::from_value::<GetMemoryRevisionsParams>(serde_json::json!({})).is_err());

        let page: GetPageRevisionsParams =
            serde_json::from_value(serde_json::json!({"page_id": "page_1"})).unwrap();
        assert_eq!(page.page_id, "page_1");
        assert!(serde_json::from_value::<GetPageRevisionsParams>(serde_json::json!({})).is_err());

        serde_json::from_value::<ListPendingImportsParams>(serde_json::json!({})).unwrap();
        let rejections: ListRejectionsParams = serde_json::from_value(serde_json::json!({
            "limit": "30",
            "reason": "duplicate"
        }))
        .unwrap();
        assert_eq!(rejections.limit, Some(30));
        assert_eq!(rejections.reason.as_deref(), Some("duplicate"));
    }

    #[test]
    fn capture_memory_type_schema_lists_every_canonical_type() {
        let params_schema = serde_json::to_string(&schemars::schema_for!(CaptureParams))
            .expect("CaptureParams schema serializes");
        for ty in wenlan_types::MemoryType::all_values() {
            assert!(
                params_schema.contains(ty),
                "CaptureParams.memory_type schema must list canonical type \"{ty}\", got: {params_schema}"
            );
        }
    }

    #[test]
    fn recall_memory_type_schema_lists_every_canonical_type() {
        let params_schema = serde_json::to_string(&schemars::schema_for!(RecallParams))
            .expect("RecallParams schema serializes");
        for ty in wenlan_types::MemoryType::all_values() {
            assert!(
                params_schema.contains(ty),
                "RecallParams.memory_type schema must list canonical type \"{ty}\", got: {params_schema}"
            );
        }
    }

    #[test]
    fn write_page_schema_documents_traceability() {
        let schema = serde_json::to_string(&schemars::schema_for!(WritePageParams))
            .expect("WritePageParams schema serializes");
        assert!(
            schema.contains("traceability"),
            "schema must spell out why source_memory_ids matter"
        );
    }

    #[test]
    fn delete_page_tool_is_marked_destructive() {
        let server = make_server(TransportMode::Stdio, "test", None);
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "delete_page")
            .expect("delete_page registered");
        let ann = tool.annotations.as_ref().expect("annotations present");
        assert_eq!(
            ann.destructive_hint,
            Some(true),
            "delete_page must declare destructive_hint=true"
        );
    }

    // ===== DistillParams force field =====

    #[test]
    fn distill_params_deserializes_force() {
        let p: DistillParams =
            serde_json::from_str(r#"{"target":"page_xyz","force":true}"#).unwrap();
        assert_eq!(p.target.as_deref(), Some("page_xyz"));
        assert_eq!(p.force, Some(true));
    }

    #[test]
    fn distill_params_defaults_force_to_none() {
        let p: DistillParams = serde_json::from_str(r#"{"target":"foo"}"#).unwrap();
        assert_eq!(p.force, None);
    }

    // ===== effective_space =====

    #[test]
    fn locked_overrides_inbound_space() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::set_var("WENLAN_SPACE", "career");
        crate::lock_state::init_from_env();

        let inbound = Some("ideas".to_string());
        let resolved = effective_space(&inbound);
        assert_eq!(resolved.as_deref(), Some("career"));
    }

    #[test]
    fn unlocked_passes_inbound_through() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();

        let inbound = Some("ideas".to_string());
        let resolved = effective_space(&inbound);
        assert_eq!(resolved.as_deref(), Some("ideas"));
    }

    #[test]
    fn locked_with_no_inbound_yields_locked() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::set_var("WENLAN_SPACE", "career");
        crate::lock_state::init_from_env();

        let inbound: Option<String> = None;
        let resolved = effective_space(&inbound);
        assert_eq!(resolved.as_deref(), Some("career"));
    }

    #[test]
    fn unlocked_with_no_inbound_yields_none() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();

        let inbound: Option<String> = None;
        let resolved = effective_space(&inbound);
        assert_eq!(resolved, None);
    }

    #[test]
    fn default_space_fills_an_omitted_argument() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        std::env::set_var("WENLAN_DEFAULT_SPACE", "fallback");
        crate::lock_state::init_from_env();

        let resolved = effective_space(&None);
        assert_eq!(resolved.as_deref(), Some("fallback"));

        std::env::remove_var("WENLAN_DEFAULT_SPACE");
        crate::lock_state::init_from_env();
    }

    #[test]
    fn explicit_tool_space_beats_default_space() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        std::env::set_var("WENLAN_DEFAULT_SPACE", "fallback");
        crate::lock_state::init_from_env();

        let resolved = effective_space(&Some("explicit".to_string()));
        assert_eq!(resolved.as_deref(), Some("explicit"));

        std::env::remove_var("WENLAN_DEFAULT_SPACE");
        crate::lock_state::init_from_env();
    }

    #[test]
    fn unlocked_empty_inbound_yields_none() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();

        let inbound = Some(String::new());
        let resolved = effective_space(&inbound);
        assert_eq!(resolved, None);
    }

    #[test]
    fn unlocked_whitespace_inbound_yields_none() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();

        let inbound = Some("   ".to_string());
        let resolved = effective_space(&inbound);
        assert_eq!(resolved, None);
    }

    // ===== effective_repair_lint_scope =====

    #[test]
    fn locked_repair_scope_overrides_global_inbound() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::set_var("WENLAN_SPACE", "career");
        crate::lock_state::init_from_env();

        let resolved = effective_repair_lint_scope(RepairLintScopeParam::Global).unwrap();
        assert_eq!(
            resolved,
            wenlan_types::repair::RepairLintScope::registered("career".to_string()).unwrap()
        );

        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();
    }

    #[test]
    fn locked_repair_scope_overrides_other_registered_inbound() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::set_var("WENLAN_SPACE", "career");
        crate::lock_state::init_from_env();

        let resolved = effective_repair_lint_scope(RepairLintScopeParam::Registered {
            space: "ideas".to_string(),
        })
        .unwrap();
        assert_eq!(
            resolved,
            wenlan_types::repair::RepairLintScope::registered("career".to_string()).unwrap()
        );

        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();
    }

    #[test]
    fn unlocked_repair_scope_preserves_explicit_inbound() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();

        let resolved = effective_repair_lint_scope(RepairLintScopeParam::Registered {
            space: "ideas".to_string(),
        })
        .unwrap();
        assert_eq!(
            resolved,
            wenlan_types::repair::RepairLintScope::registered("ideas".to_string()).unwrap()
        );
    }

    // ===== Schema gating =====

    /// Baseline: the raw `capture` schema from the tool router includes `space`.
    #[test]
    fn capture_schema_has_space_in_raw_router() {
        let tools = WenlanMcpServer::tool_router().list_all();
        let capture = tools
            .into_iter()
            .find(|t| t.name == "capture")
            .expect("capture tool registered");
        let props = capture
            .input_schema
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("capture has properties");
        assert!(
            props.contains_key("space"),
            "baseline: capture schema must have space before gating"
        );
    }

    /// When locked, `strip_space_from_tool_schema` removes `space` from properties.
    #[test]
    fn capture_tool_schema_omits_space_when_locked() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::set_var("WENLAN_SPACE", "career");
        crate::lock_state::init_from_env();

        let tools = WenlanMcpServer::tool_router().list_all();
        let tools: Vec<_> = tools
            .into_iter()
            .map(strip_space_from_tool_schema)
            .collect();
        let capture = tools
            .iter()
            .find(|t| t.name == "capture")
            .expect("capture tool registered");
        let props = capture
            .input_schema
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("capture has properties");
        assert!(
            !props.contains_key("space"),
            "space field must be omitted from capture schema when WENLAN_SPACE is locked"
        );

        // Clean up.
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();
    }

    /// Unlocked: `list_tools` equivalent — raw router listing preserves `space`.
    #[test]
    fn capture_tool_schema_includes_space_when_unlocked() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        crate::lock_state::init_from_env();

        // When not locked, tools are returned as-is (no stripping).
        let tools = WenlanMcpServer::tool_router().list_all();
        let capture = tools
            .iter()
            .find(|t| t.name == "capture")
            .expect("capture tool registered");
        let props = capture
            .input_schema
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("capture has properties");
        assert!(
            props.contains_key("space"),
            "space field must be present in capture schema when WENLAN_SPACE is not locked"
        );
    }

    #[test]
    fn default_space_keeps_explicit_space_in_tool_schema() {
        let _guard = crate::lock_state::ENV_LOCK.blocking_lock();
        std::env::remove_var("WENLAN_SPACE");
        std::env::set_var("WENLAN_DEFAULT_SPACE", "fallback");
        crate::lock_state::init_from_env();

        let tools = WenlanMcpServer::tool_router().list_all();
        let capture = tools
            .iter()
            .find(|tool| tool.name == "capture")
            .expect("capture tool registered");
        let props = capture.input_schema["properties"].as_object().unwrap();
        assert!(props.contains_key("space"));

        std::env::remove_var("WENLAN_DEFAULT_SPACE");
        crate::lock_state::init_from_env();
    }

    /// Collect `properties.<name>` entries whose schema is a bare boolean.
    fn boolean_property_schemas(node: &serde_json::Value, path: &str, out: &mut Vec<String>) {
        match node {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::Object(props)) = map.get("properties") {
                    for (name, schema) in props {
                        if schema.is_boolean() {
                            out.push(format!("{path}.{name}"));
                        }
                    }
                }
                for value in map.values() {
                    boolean_property_schemas(value, path, out);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    boolean_property_schemas(item, path, out);
                }
            }
            _ => {}
        }
    }

    /// No tool may expose a bare-boolean property schema.
    ///
    /// `#[schemars(with = "serde_json::Value")]` renders as the schema `true`. That is
    /// legal JSON Schema 2020-12, but MCP clients that validate with Zod (Claude Code
    /// among them) reject it. `tools/list` is all-or-nothing, so ONE such property makes
    /// the whole listing fail and every tool disappears from the client — the server
    /// still connects, which is why it presents as "tools fetch failed" rather than a
    /// crash. Use `serde_json::Map<String, serde_json::Value>` instead: it renders as
    /// `{"type": "object", "additionalProperties": true}` and validates.
    #[test]
    fn tool_schemas_have_no_boolean_property_schemas() {
        let mut offenders = Vec::new();
        for tool in WenlanMcpServer::tool_router().list_all() {
            let schema = serde_json::Value::Object((*tool.input_schema).clone());
            boolean_property_schemas(&schema, &tool.name, &mut offenders);
        }
        assert!(
            offenders.is_empty(),
            "bare-boolean property schemas would make Claude Code reject tools/list \
             entirely, hiding EVERY tool: {offenders:?}"
        );
    }

    /// The 29 tools in the final 2026-07 Phase A MCP surface.
    /// Deleting or adding a tool must edit this list deliberately.
    fn expected_tool_surface() -> Vec<&'static str> {
        vec![
            "accept_refinement",
            "accept_revision",
            "apply_lint_repair",
            "capture",
            "confirm_memory",
            "brief",
            "create_entity",
            "create_relation",
            "delete_page",
            "dismiss_revision",
            "distill",
            "forget",
            "get_lint_agent_work_page",
            "get_lint_repair_plan_entries",
            "get_memory_revisions",
            "get_page_revisions",
            "get_page_sources",
            "lint",
            "list_pending",
            "list_pending_imports",
            "list_pending_revisions",
            "list_refinements",
            "list_rejections",
            "prepare_lint_repair",
            "prepare_lint_repair_plan",
            "recall",
            "reject_refinement",
            "verify_lint_repair",
            "write_page",
        ]
    }

    #[test]
    fn legacy_context_is_registered_but_not_visible() {
        let raw = WenlanMcpServer::tool_router().list_all();
        assert!(raw.iter().any(|tool| tool.name == "context"));
        assert!(make_server(TransportMode::Stdio, "agent", None)
            .visible_tools()
            .iter()
            .all(|tool| tool.name != "context"));
    }

    #[test]
    fn tool_surface_is_locked() {
        let mut actual: Vec<String> = make_server(TransportMode::Stdio, "agent", None)
            .visible_tools()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect();
        actual.sort();
        let mut expected: Vec<String> = expected_tool_surface()
            .into_iter()
            .map(String::from)
            .collect();
        expected.sort();
        assert_eq!(
            actual, expected,
            "MCP tool surface drifted — every add/remove must edit expected_tool_surface() deliberately"
        );
    }

    /// Every MCP tool token in either skill tree must name a live tool.
    #[test]
    fn skills_reference_only_live_tools() {
        let live: std::collections::HashSet<&str> = expected_tool_surface().into_iter().collect();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let mut stale = Vec::new();
        for tree in ["plugin/skills", "plugin-codex/skills"] {
            let dir = root.join(tree);
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let file = entry.path().join("SKILL.md");
                let Ok(text) = std::fs::read_to_string(&file) else {
                    continue;
                };
                for prefix in ["mcp__plugin_wenlan_wenlan__", "mcp__wenlan__"] {
                    for (start, _) in text.match_indices(prefix) {
                        let name: String = text[start + prefix.len()..]
                            .chars()
                            .take_while(|ch| ch.is_ascii_lowercase() || *ch == '_')
                            .collect();
                        if !name.is_empty() && !live.contains(name.as_str()) {
                            stale.push(format!("{}: {name}", file.display()));
                        }
                    }
                }
            }
        }
        assert!(
            stale.is_empty(),
            "skills name tools that do not exist: {stale:?}"
        );
    }
}
