use crate::client::{WenlanClient, WenlanError};
use crate::types::*;
use rmcp::{
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{
        CallToolResult, Content, Implementation, InitializeResult, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, Tool,
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

/// Deserialize an `Option<i64>` that also accepts stringified numbers (e.g. `"1715000000000"`).
/// Same lenient shape as `deserialize_optional_usize_lenient`, for params that map onto
/// signed daemon fields (timestamps, badge windows, etc.).
fn deserialize_optional_i64_lenient<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        Number(i64),
        Str(String),
    }

    match Option::<StringOrNumber>::deserialize(deserializer)? {
        None => Ok(None),
        Some(StringOrNumber::Number(n)) => Ok(Some(n)),
        Some(StringOrNumber::Str(s)) => {
            s.parse::<i64>().map(Some).map_err(serde::de::Error::custom)
        }
    }
}

/// Return the effective space for a tool call: when locked, always the
/// locked value (warns if model attempted to override); otherwise the
/// non-empty inbound value passed by the model.
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
    } else {
        inbound
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
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
    #[allow(dead_code)]
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
        description = "source_id of a memory this replaces. Use when correcting or updating an existing memory — get the ID from recall first."
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
        description = "Max memory results (distilled pages are returned separately), default 10. Use 3-5 for quick lookups, 10-20 for exploration."
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

// --- Review proposal params ---

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

// --- Knowledge graph CRUD params ---

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateEntityParams {
    #[schemars(
        description = "Canonical entity name (e.g. 'Alice', 'Wenlan', 'PostgreSQL'). Use the exact, full name — aliases resolve to this canonical form."
    )]
    pub name: String,
    #[schemars(
        description = "Entity category: 'person', 'project', 'tool', 'place', 'organization', etc. Free-form string; choose the noun that best describes what it is."
    )]
    pub entity_type: String,
    #[schemars(description = "Topic scope (e.g. 'work', 'origin'). Optional.")]
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
        description = "Canonical name of the source entity (e.g. 'Alice'). Must exist or will be created on the daemon side."
    )]
    pub from_entity: String,
    #[schemars(
        description = "Canonical name of the target entity (e.g. 'Wenlan'). Must exist or will be created on the daemon side."
    )]
    pub to_entity: String,
    #[schemars(
        description = "Verb describing the directed relation (e.g. 'works_on', 'prefers', 'uses', 'depends_on'). Snake_case, present-tense."
    )]
    pub relation_type: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreateObservationParams {
    pub entity_id: String,
    pub content: String,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConfirmEntityParams {
    pub entity_id: String,
    #[serde(default = "default_confirmed")]
    pub confirmed: bool,
}

fn default_confirmed() -> bool {
    true
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct UpdateObservationParams {
    pub observation_id: String,
    pub content: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConfirmObservationParams {
    pub observation_id: String,
    #[serde(default = "default_confirmed")]
    pub confirmed: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DeleteObservationParams {
    pub observation_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CreatePageParams {
    #[schemars(
        description = "Short noun phrase that names the page (e.g. 'Wenlan daemon architecture')."
    )]
    pub title: String,
    #[schemars(
        description = "Markdown body — 3-7 paragraphs of wiki prose with [[wikilinks]]. Do not cite source ids inline; pass them in source_memory_ids and the daemon attaches provenance automatically."
    )]
    pub content: String,
    #[schemars(description = "Optional one-sentence summary — the durable claim.")]
    pub summary: Option<String>,
    #[schemars(
        description = "Optional entity_id (e.g. 'ent_abc') to anchor the page to a knowledge-graph entity."
    )]
    pub entity_id: Option<String>,
    #[schemars(description = "Topic scope (e.g. 'origin', 'work'). Optional.")]
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
        description = "Page id (e.g. 'page_abc' or legacy 'concept_abc'). Get it from get_page or distill output."
    )]
    pub page_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct UpdatePageParams {
    #[schemars(
        description = "Page id (e.g. 'page_abc' or legacy 'concept_abc'). Get it from the `stale_pages` block in distill output."
    )]
    pub page_id: String,
    #[schemars(
        description = "Refreshed markdown body — same wiki-prose style as create_page. Replaces the existing content."
    )]
    pub content: String,
    #[schemars(
        description = "Full source_memory_ids list for the refreshed page — typically the stale page's existing list (carry through from distill output)."
    )]
    pub source_memory_ids: Vec<String>,
    #[schemars(
        description = "Optional one-sentence summary. Omit to keep the existing summary; pass empty string to clear it."
    )]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetPageParams {
    #[schemars(
        description = "Page id (e.g. 'page_abc' or legacy 'concept_abc'). For title-based lookup, search via recall or the daemon's /api/pages/search."
    )]
    pub page_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetPageLinksParams {
    #[schemars(
        description = "Page id (e.g. 'page_abc'). Returns inbound + outbound wikilink graph for that page."
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
pub struct ListMemoriesParams {
    #[schemars(
        description = "Filter by memory type (e.g. 'fact', 'preference', 'decision'). Optional."
    )]
    pub memory_type: Option<String>,
    #[schemars(description = "Filter by topic/space. Optional.")]
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    #[schemars(
        description = "Max results, default 100. Increase for bulk listings, decrease for quick scans."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchPagesParams {
    #[schemars(
        description = "Natural-language search over page title + body content (e.g. 'mutex deadlock', 'distillation architecture')."
    )]
    pub query: String,
    #[schemars(
        description = "Max results, default 20. Use 1 to resolve a title to its id before calling get_page; higher for broader search."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    #[schemars(
        description = "Optional page type filter (e.g. 'recap', 'decision'). Narrows results to one type. Omit to search all types."
    )]
    #[serde(default)]
    pub page_type: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListPagesRecentParams {
    #[schemars(
        description = "Max results, default 10. Use higher (up to ~50) for a wider sweep of recent activity."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    #[schemars(
        description = "Optional Unix milliseconds. Items modified before this timestamp lose their 'new'/'updated' badge; the feed itself is still top-N by recency. This is not a date filter — items before `since_ms` are still returned, just without badges. Omit for default badge behavior."
    )]
    #[serde(default, deserialize_with = "deserialize_optional_i64_lenient")]
    pub since_ms: Option<i64>,
}

// --- Curation read params ---

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListNurtureParams {
    /// Maximum cards to return. Default 50. Clamped to 1..=500.
    #[serde(default, deserialize_with = "deserialize_optional_usize_lenient")]
    pub limit: Option<usize>,
    /// Restrict to a single space.
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListEntitySuggestionsParams {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListSpacesParams {}

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
pub struct DismissContradictionRequest {
    /// The source_id of the memory whose contradiction flags should be dismissed.
    pub source_id: String,
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

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListOrphanLinksParams {
    /// Minimum reference count a label must have to appear. Default 1. Daemon clamps via `.max(1)`.
    #[serde(default, deserialize_with = "deserialize_optional_i64_lenient")]
    pub min_count: Option<i64>,
}

// ===== Internal Implementations =====

fn format_capture_success(resp: &StoreMemoryResponse) -> String {
    let mut msg = format!("Stored {}", resp.source_id);
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

fn format_doctor_message(status: &serde_json::Value) -> String {
    let mode = status
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let setup_completed = status
        .get("setup_completed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let anthropic_key_configured = status
        .get("anthropic_key_configured")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let local_model_selected = status.get("local_model_selected").and_then(|v| v.as_str());
    let local_model_loaded = status.get("local_model_loaded").and_then(|v| v.as_str());
    let local_model_cached = status
        .get("local_model_cached")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mode_label = match mode {
        "basic-memory" => "Local Memory",
        "local-model" => "On-device Model",
        "anthropic-key" => "Anthropic Key",
        other => other,
    };
    let local_model_line = match local_model_selected {
        Some(id) => {
            let cache_status = if local_model_cached {
                "downloaded"
            } else {
                "not downloaded"
            };
            let loaded_status = if Some(id) == local_model_loaded {
                ", loaded"
            } else {
                ""
            };
            format!("{id} ({cache_status}{loaded_status})")
        }
        None => "not selected".to_string(),
    };
    let refinement_line = if anthropic_key_configured || local_model_loaded.is_some() {
        "enabled (richer extraction and page synthesis are active)"
    } else if setup_completed {
        "off (local memory stores, searches, and recalls now. Choose an on-device model or Anthropic key for richer extraction.)"
    } else {
        "not configured"
    };

    let mut msg = format!(
        "Wenlan daemon: running\n\
         Setup: {}\n\
         Mode: {mode_label}\n\
         Anthropic key: {}\n\
         On-device model: {local_model_line}\n\
         Distill cycles: {refinement_line}",
        if setup_completed {
            "completed"
        } else {
            "not completed"
        },
        if anthropic_key_configured {
            "configured"
        } else {
            "not configured"
        }
    );

    if !setup_completed {
        msg.push_str(
            "\n\nRun `wenlan setup` to choose Local Memory, On-device Model, or Anthropic Key.",
        );
    } else if !anthropic_key_configured && local_model_loaded.is_none() {
        msg.push_str(
            "\n\nLocal memory works now: capture, recall, and context are available. \
             To enable richer extraction and distill cycles, run `wenlan models install` \
             or `wenlan keys set anthropic`.",
        );
    }

    msg
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
            space: space_arg,
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
            limit: params.limit.unwrap_or(10),
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

    pub async fn doctor_impl(&self) -> Result<CallToolResult, McpError> {
        let status: serde_json::Value = match self.client.get("/api/setup/status").await {
            Ok(r) => r,
            Err(WenlanError::Api { status: 404, .. }) => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Wenlan daemon is running, but it does not expose /api/setup/status. \
                     Update Wenlan, then run `wenlan doctor`."
                        .to_string(),
                )]));
            }
            Err(e) => return Ok(tool_error(e, "status check")),
        };

        Ok(CallToolResult::success(vec![Content::text(
            format_doctor_message(&status),
        )]))
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
            .post::<serde_json::Value, serde_json::Value>(&path, &serde_json::json!({}))
            .await
        {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Memory {} confirmed.",
                memory_id
            ))])),
            Err(e) => Ok(tool_error(e, "confirm_memory")),
        }
    }

    pub async fn create_entity_impl(
        &self,
        params: CreateEntityParams,
    ) -> Result<CallToolResult, McpError> {
        let source_agent = self.resolve_source_agent(None);
        let space_arg = effective_space(&params.space);
        let req = CreateEntityRequest {
            name: params.name,
            entity_type: params.entity_type,
            space: space_arg,
            source_agent,
            confidence: params.confidence,
        };
        let resp: CreateEntityResponse = try_call!(
            self.client.post("/api/memory/entities", &req),
            "create_entity"
        );
        let mut text = format!("Created entity {}", resp.id);
        for w in &resp.warnings {
            text.push_str(&format!("\nwarning: {w}"));
        }
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    pub async fn create_relation_impl(
        &self,
        params: CreateRelationParams,
    ) -> Result<CallToolResult, McpError> {
        let source_agent = self.resolve_source_agent(None);
        let req = CreateRelationRequest {
            from_entity: params.from_entity,
            to_entity: params.to_entity,
            relation_type: params.relation_type,
            source_agent,
            confidence: None,
            explanation: None,
            source_memory_id: None,
        };
        let resp: CreateRelationResponse = try_call!(
            self.client.post("/api/memory/relations", &req),
            "create_relation"
        );
        let mut text = format!("Created relation {}", resp.id);
        for w in &resp.warnings {
            text.push_str(&format!("\nwarning: {w}"));
        }
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    pub async fn create_observation_impl(
        &self,
        params: CreateObservationParams,
    ) -> Result<CallToolResult, McpError> {
        let req = wenlan_types::requests::AddObservationRequest {
            entity_id: params.entity_id,
            content: params.content,
            source_agent: params.source_agent,
            confidence: params.confidence,
        };
        let resp: wenlan_types::responses::AddObservationResponse = try_call!(
            self.client.post("/api/memory/observations", &req),
            "create_observation"
        );
        let mut text = format!("Created observation {}", resp.id);
        for w in &resp.warnings {
            text.push_str(&format!("\nwarning: {w}"));
        }
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    pub async fn confirm_entity_impl(
        &self,
        params: ConfirmEntityParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Confirm operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to confirm entities."
                    .to_string(),
            )]));
        }
        let req = wenlan_types::requests::ConfirmEntityRequest {
            confirmed: params.confirmed,
        };
        let path = format!("/api/memory/entities/{}/confirm", params.entity_id);
        let _: wenlan_types::responses::SuccessResponse =
            try_call!(self.client.put(&path, &req), "confirm_entity");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Entity {} {}",
            params.entity_id,
            if params.confirmed {
                "confirmed"
            } else {
                "unconfirmed"
            }
        ))]))
    }

    pub async fn update_observation_impl(
        &self,
        params: UpdateObservationParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Update operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to update observations."
                    .to_string(),
            )]));
        }
        let req = wenlan_types::requests::UpdateObservationRequest {
            content: params.content,
        };
        let path = format!("/api/memory/observations/{}", params.observation_id);
        let _: wenlan_types::responses::SuccessResponse =
            try_call!(self.client.put(&path, &req), "update_observation");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Updated observation {}",
            params.observation_id
        ))]))
    }

    pub async fn confirm_observation_impl(
        &self,
        params: ConfirmObservationParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Confirm operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to confirm observations."
                    .to_string(),
            )]));
        }
        let req = wenlan_types::requests::ConfirmObservationRequest {
            confirmed: params.confirmed,
        };
        let path = format!("/api/memory/observations/{}/confirm", params.observation_id);
        let _: wenlan_types::responses::SuccessResponse =
            try_call!(self.client.put(&path, &req), "confirm_observation");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Observation {} {}",
            params.observation_id,
            if params.confirmed {
                "confirmed"
            } else {
                "unconfirmed"
            }
        ))]))
    }

    pub async fn delete_observation_impl(
        &self,
        params: DeleteObservationParams,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Delete operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to delete observations."
                    .to_string(),
            )]));
        }
        let path = format!("/api/memory/observations/{}", params.observation_id);
        let _: wenlan_types::responses::SuccessResponse =
            try_call!(self.client.delete(&path), "delete_observation");
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Observation {} deleted",
            params.observation_id
        ))]))
    }

    pub async fn create_page_impl(
        &self,
        params: CreatePageParams,
    ) -> Result<CallToolResult, McpError> {
        let space_arg = effective_space(&params.space);
        let req = CreateConceptRequest {
            title: params.title,
            content: params.content,
            summary: params.summary,
            entity_id: params.entity_id,
            space: space_arg,
            source_memory_ids: params.source_memory_ids,
            creation_kind: None,
            workspace: None,
        };
        let resp: CreatePageResponse =
            try_call!(self.client.post("/api/pages", &req), "create_page");
        let mut text = format!("Created page {}", resp.id);
        for w in &resp.warnings {
            text.push_str(&format!("\nwarning: {w}"));
        }
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    pub async fn update_page_impl(
        &self,
        params: UpdatePageParams,
    ) -> Result<CallToolResult, McpError> {
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
        let path = format!("/api/pages/{}", params.page_id);
        // Typed end-to-end: a wire-shape drift on the daemon side fails at
        // deserialize instead of silently returning the no-op "Refreshed"
        // line. Same discipline as PR #77's search_pages / list_pages_recent.
        let resp: wenlan_types::responses::PageWriteResponse =
            try_call!(self.client.put(&path, &req), "update_page");
        // Ownership gate (spec §5.2): a human-owned page is never overwritten in
        // place; the daemon stages a revision card. Surface that so the caller
        // does not believe the prose was rewritten.
        let msg = format_update_page_response(&params.page_id, resp);
        Ok(CallToolResult::success(vec![Content::text(msg)]))
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

    pub async fn get_page_impl(&self, page_id: &str) -> Result<CallToolResult, McpError> {
        // Envelope for GET /api/pages/{id} (handle_get_page,
        // memory_routes.rs:1938): `{"page": Page}`. No exported wenlan-types
        // envelope matches this shape (PageDraftResponse{page: Page} is the
        // page-draft feature's response, not this route's), so a local type
        // stands in for serde_json::Value -- a wire-shape drift now fails at
        // deserialize instead of round-tripping silently as untyped JSON.
        // `deny_unknown_fields` makes an ADDITIVE envelope key fail loud too
        // (the top-level object; the inner Page keeps its own field contract).
        #[derive(Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct GetPageResponse {
            page: wenlan_types::Page,
        }

        let path = format!("/api/pages/{}", page_id);
        let resp: GetPageResponse = try_call!(self.client.get(&path), "get_page");
        let pretty = serde_json::to_string_pretty(&resp).unwrap_or_else(|_| String::new());
        Ok(CallToolResult::success(vec![Content::text(pretty)]))
    }

    pub async fn get_page_links_impl(&self, page_id: &str) -> Result<CallToolResult, McpError> {
        let path = format!("/api/pages/{}/links", page_id);
        // Typed end-to-end via PageLinksResponse — keeps wire shape pinned.
        let resp: wenlan_types::responses::PageLinksResponse =
            try_call!(self.client.get(&path), "get_page_links");
        let pretty = serde_json::to_string_pretty(&resp).unwrap_or_else(|_| String::new());
        Ok(CallToolResult::success(vec![Content::text(pretty)]))
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

    pub async fn list_memories_impl(
        &self,
        params: ListMemoriesParams,
    ) -> Result<CallToolResult, McpError> {
        let space_arg = effective_space(&params.space);
        let req = ListMemoriesRequest {
            memory_type: params.memory_type,
            space: space_arg,
            limit: params.limit.unwrap_or(100),
            confirmed: None,
        };
        let resp: ListMemoriesResponse =
            try_call!(self.client.post("/api/memory/list", &req), "list_memories");
        let pretty = serde_json::to_string_pretty(&resp.memories)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} memories\n{}",
            resp.memories.len(),
            pretty
        ))]))
    }

    pub async fn search_pages_impl(
        &self,
        params: SearchPagesParams,
    ) -> Result<CallToolResult, McpError> {
        let req = SearchPagesRequest {
            query: params.query,
            limit: params.limit,
            page_type: params.page_type,
            space: None,
        };
        let resp: SearchPagesResponse =
            try_call!(self.client.post("/api/pages/search", &req), "search_pages");
        // Metadata-only render: never serialize page bodies into the agent's
        // context. Fetch a single body on demand with get_page.
        let body = format_page_list(&resp.pages);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} pages\n{}",
            resp.pages.len(),
            body
        ))]))
    }

    pub async fn list_pages_recent_impl(
        &self,
        params: ListPagesRecentParams,
    ) -> Result<CallToolResult, McpError> {
        let path = build_recent_pages_path(params.limit, params.since_ms);
        let resp: Vec<RecentActivityItem> = try_call!(self.client.get(&path), "list_pages_recent");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} recent pages\n{}",
            resp.len(),
            pretty
        ))]))
    }

    pub async fn list_spaces_impl(
        &self,
        _params: ListSpacesParams,
    ) -> Result<CallToolResult, McpError> {
        let resp: Vec<Space> = try_call!(self.client.get("/api/spaces"), "list_spaces");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} spaces\n{}",
            resp.len(),
            pretty
        ))]))
    }

    pub async fn list_refinements_impl(
        &self,
        params: ListRefinementsParams,
    ) -> Result<CallToolResult, McpError> {
        let mut path = String::from("/api/refinery/queue");
        let mut q: Vec<String> = Vec::new();
        if let Some(a) = params.action.as_deref() {
            q.push(format!("action={}", url_encode_simple(a)));
        }
        if let Some(l) = params.limit {
            q.push(format!("limit={l}"));
        }
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q.join("&"));
        }

        let resp: ListRefinementsResponse = try_call!(self.client.get(&path), "list_refinements");

        let pretty = serde_json::to_string_pretty(&resp.proposals)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
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
        let req = match params.space {
            Some(space) => {
                wenlan_types::requests::AcceptRefinementRequest::PickSpace { space, notes: None }
            }
            None => wenlan_types::requests::AcceptRefinementRequest::Accept { notes: None },
        };
        let resp: AcceptRefinementResponse =
            try_call!(self.client.post(&path, &req), "accept_refinement");

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Review proposal {} accepted (action={}).",
            resp.id, resp.action_applied
        ))]))
    }

    pub async fn list_nurture_impl(
        &self,
        params: ListNurtureParams,
    ) -> Result<CallToolResult, McpError> {
        let space_arg = effective_space(&params.space);
        let mut path = String::from("/api/memory/nurture");
        let mut q: Vec<String> = Vec::new();
        if let Some(l) = params.limit {
            q.push(format!("limit={}", l.clamp(1, 500)));
        }
        if let Some(s) = space_arg.as_deref().filter(|s| !s.is_empty()) {
            q.push(format!("space={}", url_encode_simple(s)));
        }
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q.join("&"));
        }

        let resp: wenlan_types::responses::NurtureCardsResponse =
            try_call!(self.client.get(&path), "list_nurture");

        let pretty = serde_json::to_string_pretty(&resp.cards)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} nurture cards\n{}",
            resp.cards.len(),
            pretty
        ))]))
    }

    pub async fn list_entity_suggestions_impl(
        &self,
        _params: ListEntitySuggestionsParams,
    ) -> Result<CallToolResult, McpError> {
        let resp: Vec<wenlan_types::entities::EntitySuggestion> = try_call!(
            self.client.get("/api/memory/entity-suggestions"),
            "list_entity_suggestions"
        );
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} entity suggestion(s)\n{}",
            resp.len(),
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

    pub async fn dismiss_contradiction_impl(
        &self,
        req: DismissContradictionRequest,
    ) -> Result<CallToolResult, McpError> {
        if self.transport == TransportMode::Http {
            return Ok(CallToolResult::error(vec![Content::text(
                "Contradiction operations are not available over remote connections. \
                 Use local MCP on the machine running Wenlan to dismiss contradictions."
                    .to_string(),
            )]));
        }
        let path = format!("/api/memory/contradiction/{}/dismiss", req.source_id);
        let response = try_call!(
            self.client
                .post_empty::<ContradictionDismissResponse>(&path),
            "dismiss_contradiction"
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
        let mut q: Vec<String> = Vec::new();
        if let Some(l) = params.limit {
            q.push(format!("limit={}", l.clamp(1, 500)));
        }
        if let Some(r) = params.reason.as_deref().filter(|s| !s.is_empty()) {
            q.push(format!("reason={}", url_encode_simple(r)));
        }
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q.join("&"));
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

    pub async fn list_orphan_links_impl(
        &self,
        params: ListOrphanLinksParams,
    ) -> Result<CallToolResult, McpError> {
        let path = match params.min_count {
            Some(n) => format!("/api/pages/orphan-links?min_count={}", n.max(1)),
            None => "/api/pages/orphan-links".to_string(),
        };
        let resp: wenlan_types::responses::OrphanLinksResponse =
            try_call!(self.client.get(&path), "list_orphan_links");
        let pretty = serde_json::to_string_pretty(&resp)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} orphan link(s)\n{}",
            resp.orphan_labels.len(),
            pretty
        ))]))
    }
}

/// Build the `/api/pages/recent` URL with optional `limit` + `since_ms` query
/// params. Pure function so the test can exercise the actual builder rather
/// than a duplicate.
fn build_recent_pages_path(limit: Option<usize>, since_ms: Option<i64>) -> String {
    let mut path = String::from("/api/pages/recent");
    let mut q: Vec<String> = Vec::new();
    if let Some(l) = limit {
        q.push(format!("limit={}", l));
    }
    if let Some(s) = since_ms {
        q.push(format!("since_ms={}", s));
    }
    if !q.is_empty() {
        path.push('?');
        path.push_str(&q.join("&"));
    }
    path
}

/// Render a metadata-only listing of pages for MCP tool output: one line
/// per page as `<id>  <title>  — <summary>`. Deliberately omits `content`
/// so browsing/searching never dumps full page bodies into the agent's
/// context — fetch a single body on demand with `get_page`.
fn format_page_list(pages: &[wenlan_types::Page]) -> String {
    if pages.is_empty() {
        return "no pages".to_string();
    }
    pages
        .iter()
        .map(|p| {
            let summary = p.summary.as_deref().unwrap_or("(no summary)");
            format!("{}  {}  — {}", p.id, p.title, summary)
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Percent-encode a string for use in URL query parameter values.
/// Encodes all characters except unreserved ones (A-Z, a-z, 0-9, `-`, `_`, `.`, `~`).
fn url_encode_simple(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c]
            }
            _ => format!("%{:02X}", c as u32).chars().collect(),
        })
        .collect()
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
        description = "Search memories by query. Use when the user asks 'do you remember', 'what do you know about', 'look up', or when you need a specific fact before acting.\n\nWrite queries as natural language — the search engine handles semantic matching. For precision, use filters (memory_type, space) to narrow results. If you get too many results, add filters rather than making the query longer.\n\nFor higher retrieval quality at the cost of latency, pass `rerank: true` to opt into the cross-encoder reranker (requires WENLAN_RERANKER_ENABLED=1 on the daemon).\n\nThis is for targeted lookups. For broad session orientation, use context instead.",
        annotations(title = "Recall", read_only_hint = true, open_world_hint = false)
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        self.recall_impl(params).await
    }

    #[tool(
        description = "Load session context — identity, preferences, goals, and topic-relevant memories. Call this FIRST at the start of every session before doing anything else. Also call on major topic shifts or when the user says 'catch me up' or 'what's the background on'.\n\nThis returns a curated blend of who the user is and what's relevant. For specific factual lookups, use recall instead. Use the result to model how the user thinks, not just to look things up — their preferences and corrections tell you how they want to be helped.",
        annotations(title = "Context", read_only_hint = true, open_world_hint = false)
    )]
    async fn context(
        &self,
        Parameters(params): Parameters<ContextParams>,
    ) -> Result<CallToolResult, McpError> {
        self.context_impl(params).await
    }

    #[tool(
        description = "Diagnose the local Wenlan runtime. This is not part of the memory loop. Use only when Wenlan tools fail, when onboarding a new MCP client, or when the user asks why setup, extraction, or distill cycles are off. Reports daemon reachability, setup mode, Local Memory, On-device Model, Anthropic key state, and on-device model state.",
        annotations(title = "Doctor", read_only_hint = true, open_world_hint = false)
    )]
    async fn doctor(&self) -> Result<CallToolResult, McpError> {
        self.doctor_impl().await
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
        description = "Delete a memory by ID. Use when the user says 'forget this', 'delete that', 'that's wrong and should be removed'. Requires the source_id — get it from recall first.\n\nThis is destructive and cannot be undone. For corrections, prefer storing a new memory with the supersedes param pointing to the old one — this preserves history.",
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

    // --- Knowledge graph CRUD ---

    #[tool(
        description = "Create an entity in the knowledge graph. Use when the user names a person, project, tool, or place that isn't yet linked, or when you need a stable id to anchor memories or pages to. The daemon's post-ingest enrichment usually creates entities automatically when a model or Anthropic key is configured — call this explicitly when distill cycles are off or you need the id back synchronously.",
        annotations(
            title = "Create entity",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
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
        description = "Create a directed relation between two entities in the knowledge graph. Use sparingly — most relations come out of the daemon's enrichment when a model or Anthropic key is configured. Call this explicitly to record a relation the user articulated that the daemon couldn't infer, or when distill cycles are off.",
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
        description = "Attach a factual observation to an existing entity in the knowledge graph. Use sparingly — most observations come from daemon extraction. Call explicitly when the user articulates a fact about a person/project/tool that the daemon couldn't infer, or when distill cycles are off. Requires the entity_id; resolve via search_entities first if you only have the name. Returns 422 if entity does not exist.",
        annotations(
            title = "Create observation",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn create_observation(
        &self,
        Parameters(params): Parameters<CreateObservationParams>,
    ) -> Result<CallToolResult, McpError> {
        self.create_observation_impl(params).await
    }

    #[tool(
        description = "Confirm (or unconfirm) an entity in the knowledge graph — flips its stability flag from tentative to durable. Call when the user explicitly affirms or revokes an extracted entity (\"yes that's right\", \"no that's wrong\"), or when you have high confidence after seeing the entity reused across multiple contexts. Unconfirmed entities may be pruned by distill cycles; confirmed ones persist. Defaults confirmed=true if omitted. Do NOT call for every extracted entity — most should stay unconfirmed and let distill cycles decide. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Confirm entity",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn confirm_entity(
        &self,
        Parameters(params): Parameters<ConfirmEntityParams>,
    ) -> Result<CallToolResult, McpError> {
        self.confirm_entity_impl(params).await
    }

    #[tool(
        description = "Update the content of an existing observation. Use when the user corrects a fact (\"actually X not Y\") or when you find that a prior observation needs refinement based on new context. Only the content text changes — the entity attachment stays the same. To move an observation to a different entity, delete and recreate. Prefer this over delete+recreate when the entity attachment is correct, so history is preserved. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Update observation",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn update_observation(
        &self,
        Parameters(params): Parameters<UpdateObservationParams>,
    ) -> Result<CallToolResult, McpError> {
        self.update_observation_impl(params).await
    }

    #[tool(
        description = "Confirm (or unconfirm) an observation — flips its stability flag from tentative to durable. Call when the user explicitly affirms a specific fact attached to an entity (\"yes Alice does prefer tabs\"), or when you observe the same fact restated across multiple sources. Unconfirmed observations may be pruned by distill cycles; confirmed ones persist. Defaults confirmed=true if omitted. Do NOT call for every observation you create — let distill cycles promote them when warranted. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Confirm observation",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn confirm_observation(
        &self,
        Parameters(params): Parameters<ConfirmObservationParams>,
    ) -> Result<CallToolResult, McpError> {
        self.confirm_observation_impl(params).await
    }

    #[tool(
        description = "Delete an observation by ID. Destructive and cannot be undone — for corrections, prefer update_observation. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Delete observation",
            read_only_hint = false,
            destructive_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn delete_observation(
        &self,
        Parameters(params): Parameters<DeleteObservationParams>,
    ) -> Result<CallToolResult, McpError> {
        self.delete_observation_impl(params).await
    }

    #[tool(
        description = "Create a distilled wiki page from a memory cluster. The /distill flow uses this to post agent-synthesized pages back to the daemon. Provide a markdown body with [[wikilinks]]. Do not cite source ids inline; pass them in source_memory_ids and the daemon attaches provenance automatically. The daemon writes both the DB row and the on-disk .origin/pages/<slug>.md projection atomically.",
        annotations(
            title = "Create page",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn create_page(
        &self,
        Parameters(params): Parameters<CreatePageParams>,
    ) -> Result<CallToolResult, McpError> {
        self.create_page_impl(params).await
    }

    #[tool(
        description = "Refresh a stale page in place. Replaces content + source_memory_ids + optional summary, clears the daemon's stale_reason in the same call. Preserves page_id, created_at, and bumps version monotonically — external [[wikilinks]] keep working. Use this on entries in the /distill response's `stale_pages` block instead of delete_page + create_page (which churned ids and lost version history). Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Refresh page",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false,
            open_world_hint = false
        )
    )]
    async fn update_page(
        &self,
        Parameters(params): Parameters<UpdatePageParams>,
    ) -> Result<CallToolResult, McpError> {
        self.update_page_impl(params).await
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
        description = "Fetch a page by id. Returns the full page row including title, summary, body, source memory ids, and metadata. The /pages skill uses this for the preview block — agents reading a page should call this rather than guessing the on-disk path, because the md slug is daemon-controlled.",
        annotations(title = "Get page", read_only_hint = true, open_world_hint = false)
    )]
    async fn get_page(
        &self,
        Parameters(params): Parameters<GetPageParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_page_impl(&params.page_id).await
    }

    #[tool(
        description = "Fetch the wikilink graph centered on one page: `outbound` (labels parsed out of this page's body, with target_page_id set when matched; NULL means broken/orphan) and `inbound` (active pages whose body cites this title). Use this for the /pages preview to surface 'N inbound, M broken' without parsing the full body.",
        annotations(
            title = "Get page links",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn get_page_links(
        &self,
        Parameters(params): Parameters<GetPageLinksParams>,
    ) -> Result<CallToolResult, McpError> {
        self.get_page_links_impl(&params.page_id).await
    }

    #[tool(
        description = "Fetch the source memories of a page — the memory ids the page was distilled from, each enriched with the memory's title, content, type, and space. The /distill skill uses this on the stale-page refresh path: get_page returns ids, get_page_sources returns the full memory content needed to re-synthesize prose.",
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
        description = "Fetch the supersede chain for a memory — all prior versions ordered by depth (0 = current, 1 = immediate predecessor, …). Use after recall when you need to understand how a memory evolved or verify that a correction was recorded.",
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
        description = "Fetch the version changelog for a page — all distillation rounds ordered newest-first. Use after get_page when you need to understand what changed between versions or which source memories triggered a re-distill.",
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
        description = "List memories filtered by type and/or space. Returns the raw memory rows — useful for bulk review, type audits, or feeding a downstream tool. For semantic search use recall; for orientation use context. This is the listing path: predictable order, no relevance ranking.",
        annotations(
            title = "List memories",
            read_only_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_memories(
        &self,
        Parameters(params): Parameters<ListMemoriesParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_memories_impl(params).await
    }

    #[tool(
        description = "Search pages by query. Use to resolve a page title to its id before calling get_page (set `limit: 1` for that), or to browse pages on a topic. Returns matching pages with id, title, and summary. Optional `page_type` filter narrows to one type (e.g. `recap`, `decision`). For listing recent activity instead, use list_pages_recent.",
        annotations(title = "Search pages", read_only_hint = true, open_world_hint = false)
    )]
    async fn search_pages(
        &self,
        Parameters(params): Parameters<SearchPagesParams>,
    ) -> Result<CallToolResult, McpError> {
        self.search_pages_impl(params).await
    }

    #[tool(
        description = "List recently created or updated pages. Use when the user asks 'what's new', 'recent pages', 'what got synthesized lately'. Returns top-N pages by activity timestamp with optional badge deltas (`since_ms` scopes the badge window). For a topic search instead, use search_pages.",
        annotations(title = "Recent pages", read_only_hint = true, open_world_hint = false)
    )]
    async fn list_pages_recent(
        &self,
        Parameters(params): Parameters<ListPagesRecentParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_pages_recent_impl(params).await
    }

    #[tool(
        description = "List all spaces in this Wenlan instance. Use when the user asks 'what spaces exist', 'list my topics', or to discover space names before passing one as a filter to search_memory / list_nurture. Returns each space's name, description, memory_count, entity_count, and timestamps.",
        annotations(title = "List spaces", read_only_hint = true, open_world_hint = false)
    )]
    async fn list_spaces(
        &self,
        Parameters(params): Parameters<ListSpacesParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_spaces_impl(params).await
    }

    // --- Review proposal tools ---

    #[tool(
        description = "List pending review proposals from Wenlan's daemon-side queue. Use when the user wants to audit what the daemon has queued for review — phrases like 'pending proposals', 'what's queued', 'check review queue'. Returns proposals with typed actions and payloads, including vocab_promote and lint_repair_review items created by `/lint repair`. Filter by action with optional `action` param. Pair with `reject_refinement` to dismiss noise. lint_repair_review choices are advisory until a choice-specific repair is prepared and separately approved; generic accept does not apply them.",
        annotations(
            title = "List review proposals",
            read_only_hint = true,
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
        description = "Reject (dismiss) a review proposal by id. Use when reviewing the daemon queue and the user decides a proposal is wrong or noise. Marks the queue row dismissed and logs the agent activity. Idempotent: already-dismissed proposals return 422. Keeping a proposal is a no-op (it stays queued). Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Reject review proposal",
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
        description = "Apply a review queue proposal using sensible defaults. \
            entity_merge: existing entity wins as canonical. \
            relation_conflict: new relation supersedes. \
            detect_contradiction: previously-stored memory flagged for revision. \
            cross_space_discovery: pass `space` to choose the destination space. \
            vocab_promote: promote a non-canonical entity or relation type to a first-class vocabulary type. \
            Returns 422 for suggest_entity (no producer), dedup_merge (deprecated), \
            and lint_repair_review (requires a choice-specific repair flow). \
            Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Accept review proposal",
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

    // --- Curation read tools ---

    #[tool(
        description = "List nurture cards: memories flagged for human attention because they are unconfirmed, low-confidence, or have been queued for review by the daemon. Use when the user wants to audit what needs review: phrases like 'what needs my attention', 'unconfirmed memories', 'nurture queue'. Returns memory items with metadata. Optional `limit` caps results (default 50, max 500). Optional `space` restricts to one topic space. Distinct from `list_pending` (which lists all unconfirmed captures) and `list_refinements` (which lists daemon-generated merge/conflict proposals).",
        annotations(
            title = "List nurture cards",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_nurture(
        &self,
        Parameters(params): Parameters<ListNurtureParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_nurture_impl(params).await
    }

    #[tool(
        description = "List entity-suggestion proposals from the daemon review queue \
                       (action='suggest_entity'). Use when the user asks 'what entities \
                       does the daemon want to create' or wants to triage merge-vs-create \
                       decisions. Returns id, proposed entity_name, source_ids, confidence. \
                       Pair with PR2's approve/dismiss verbs once they land.",
        annotations(
            title = "List entity suggestions",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_entity_suggestions(
        &self,
        Parameters(params): Parameters<ListEntitySuggestionsParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_entity_suggestions_impl(params).await
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
        description = "Dismiss all awaiting-review contradiction flags for a memory. Idempotent. \
                       Returns wrote:true even if no rows matched. Not available over remote HTTP MCP transport (local stdio only).",
        annotations(
            title = "Dismiss contradiction",
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn dismiss_contradiction(
        &self,
        Parameters(req): Parameters<DismissContradictionRequest>,
    ) -> Result<CallToolResult, McpError> {
        self.dismiss_contradiction_impl(req).await
    }

    #[tool(
        description = "List in-flight chat-history imports awaiting processing or completion. \
                       Use when the user asks 'what imports are running', 'is my Claude.ai \
                       export done', or to surface import progress. Returns id, vendor, \
                       stage, source path, processed/total conversation counts.",
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
        description = "List quality-gate rejections: memories the daemon discarded before storing, due to low quality, duplication, or other filters. Use when the user asks 'what did Wenlan reject', 'what was filtered out', or to diagnose why captures are not appearing. Returns rejection records with reason code, detail, and similarity info. Optional `limit` caps results (default 50, max 500). Optional `reason` filters by rejection reason code (e.g. 'duplicate', 'low_quality').",
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

    #[tool(
        description = "List wiki-link labels that appear in page bodies but have no matching \
                       page title. Use when the user asks 'what links are broken', 'orphan links', \
                       or wants to find knowledge gaps. Returns label names and reference counts. \
                       Optional `min_count` filters to labels referenced at least N times \
                       (default 1, minimum 1).",
        annotations(
            title = "List orphan links",
            read_only_hint = true,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    async fn list_orphan_links(
        &self,
        Parameters(params): Parameters<ListOrphanLinksParams>,
    ) -> Result<CallToolResult, McpError> {
        self.list_orphan_links_impl(params).await
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

const LINT_REPAIR_TOOL_NAMES: &[&str] = &[
    "get_lint_agent_work_page",
    "prepare_lint_repair_plan",
    "get_lint_repair_plan_entries",
    "prepare_lint_repair",
    "apply_lint_repair",
    "verify_lint_repair",
];

impl WenlanMcpServer {
    fn visible_tools(&self) -> Vec<Tool> {
        let mut tools = Self::tool_router().list_all();
        if self.transport == TransportMode::Http {
            tools.retain(|tool| !LINT_REPAIR_TOOL_NAMES.contains(&tool.name.as_ref()));
        }
        tools
    }
}

// ===== ServerHandler =====

#[tool_handler]
impl ServerHandler for WenlanMcpServer {
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
             FIRST THING EVERY SESSION: Call context to load the user's identity, preferences, goals, and\n\
             topic-relevant memories. This is how you know who you're talking to. Use the result to model how the \
             user thinks — their preferences, corrections, and past decisions tell you how they want to be helped, \
             not just what they already know.\n\n\
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
             RECALL vs CONTEXT:\n\
             - context: broad orientation, session start, topic shifts, \"catch me up\"\n\
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

    // ===== Page list render (metadata-only) =====

    #[test]
    fn format_page_list_omits_body() {
        let page: wenlan_types::Page = serde_json::from_value(serde_json::json!({
            "id": "page_abc",
            "title": "Mutex deadlock notes",
            "summary": "How to avoid self-deadlock with tokio Mutex",
            "content": "SECRET_BODY_TOKEN should never reach the agent context",
            "entity_id": null,
            "source_memory_ids": ["mem_1", "mem_2"],
            "version": 3,
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
            "last_compiled": "2026-01-01T00:00:00Z",
            "last_modified": "2026-01-01T00:00:00Z",
            "sources_updated_count": 0,
            "stale_reason": null,
            "user_edited": false,
            "last_edited_by": null,
            "last_edited_at": null,
            "last_delta_summary": null,
            "changelog": null
        }))
        .expect("construct Page from json");

        let rendered = format_page_list(std::slice::from_ref(&page));
        assert!(
            !rendered.contains("SECRET_BODY_TOKEN"),
            "page body leaked into list render: {rendered}"
        );
        assert!(rendered.contains("page_abc"), "id missing: {rendered}");
        assert!(
            rendered.contains("Mutex deadlock notes"),
            "title missing: {rendered}"
        );
        assert!(
            rendered.contains("How to avoid"),
            "summary missing: {rendered}"
        );
    }

    #[test]
    fn format_page_list_empty() {
        assert_eq!(format_page_list(&[]), "no pages");
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
    fn lint_repair_tools_are_visible_only_over_stdio() {
        let stdio = make_server(TransportMode::Stdio, "agent", None)
            .visible_tools()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
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
            space: None,
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
    fn capture_success_message_is_terse() {
        let resp = StoreMemoryResponse {
            source_id: "mem_abc".into(),
            chunks_created: 3,
            memory_type: "fact".into(),
            entity_id: Some("ent_xyz".into()),
            quality: Some("high".into()),
            warnings: vec![],
            extraction_method: "llm".into(),
            enrichment: String::new(),
            hint: String::new(),
        };
        let msg = format_capture_success(&resp);
        assert_eq!(msg, "Stored mem_abc");
        assert!(!msg.contains("chunks"));
        assert!(!msg.contains("quality"));
        assert!(!msg.contains("entity"));
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
            extraction_method: "agent".into(),
            enrichment: String::new(),
            hint: String::new(),
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
            extraction_method: "agent".into(),
            enrichment: String::new(),
            hint: String::new(),
        };
        let out = format_capture_success(&resp);
        assert!(!out.contains("Triggered revisions"));
    }

    #[test]
    fn doctor_local_memory_message_sets_expectations() {
        let msg = format_doctor_message(&serde_json::json!({
            "setup_completed": true,
            "mode": "basic-memory",
            "anthropic_key_configured": false,
            "local_model_selected": null,
            "local_model_loaded": null,
            "local_model_cached": false
        }));

        assert!(msg.contains("Mode: Local Memory"));
        assert!(msg.contains("On-device model: not selected"));
        assert!(msg.contains("Distill cycles: off"));
        assert!(msg.contains("Local memory works now: capture, recall, and context are available"));
        assert!(msg.contains("wenlan models install"));
        assert!(msg.contains("wenlan keys set anthropic"));
    }

    #[test]
    fn doctor_on_device_model_message_shows_loaded_model() {
        let msg = format_doctor_message(&serde_json::json!({
            "setup_completed": true,
            "mode": "local-model",
            "anthropic_key_configured": false,
            "local_model_selected": "qwen3-1.7b",
            "local_model_loaded": "qwen3-1.7b",
            "local_model_cached": true
        }));

        assert!(msg.contains("Mode: On-device Model"), "{msg}");
        assert!(
            msg.contains("On-device model: qwen3-1.7b (downloaded, loaded)"),
            "{msg}"
        );
        assert!(msg.contains("Distill cycles: enabled"), "{msg}");
        assert!(!msg.contains("Local memory works now"));
    }

    #[test]
    fn doctor_unconfigured_message_names_three_setup_paths() {
        let msg = format_doctor_message(&serde_json::json!({
            "setup_completed": false,
            "mode": "unknown",
            "anthropic_key_configured": false,
            "local_model_selected": null,
            "local_model_loaded": null,
            "local_model_cached": false
        }));

        assert!(msg.contains("Setup: not completed"));
        assert!(msg.contains("Run `wenlan setup`"));
        assert!(msg.contains("Local Memory, On-device Model, or Anthropic Key"));
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
            space: None,
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
            space: None,
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
    async fn test_dismiss_contradiction_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let req = DismissContradictionRequest {
            source_id: "mem_x".into(),
        };
        let result = server.dismiss_contradiction_impl(req).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_dismiss_contradiction_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let req = DismissContradictionRequest {
            source_id: "mem_x".into(),
        };
        let result = server.dismiss_contradiction_impl(req).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_confirm_entity_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = ConfirmEntityParams {
            entity_id: "ent_x".into(),
            confirmed: true,
        };
        let result = server.confirm_entity_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_confirm_entity_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = ConfirmEntityParams {
            entity_id: "ent_x".into(),
            confirmed: true,
        };
        let result = server.confirm_entity_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_confirm_observation_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = ConfirmObservationParams {
            observation_id: "obs_x".into(),
            confirmed: true,
        };
        let result = server.confirm_observation_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_confirm_observation_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = ConfirmObservationParams {
            observation_id: "obs_x".into(),
            confirmed: true,
        };
        let result = server.confirm_observation_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_update_observation_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = UpdateObservationParams {
            observation_id: "obs_x".into(),
            content: "new content".into(),
        };
        let result = server.update_observation_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_update_observation_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = UpdateObservationParams {
            observation_id: "obs_x".into(),
            content: "new content".into(),
        };
        let result = server.update_observation_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_update_page_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = UpdatePageParams {
            page_id: "page_x".into(),
            content: "body".into(),
            source_memory_ids: vec!["mem_a".into()],
            summary: None,
        };
        let result = server.update_page_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_update_page_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = UpdatePageParams {
            page_id: "page_x".into(),
            content: "body".into(),
            source_memory_ids: vec!["mem_a".into()],
            summary: None,
        };
        let result = server.update_page_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_update_page_surfaces_gated_revision_card_fields() {
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
            "gated field missing from update_page response: {text}"
        );
        assert!(
            text.contains("revision_card_id: mem_page_card_1"),
            "revision_card_id missing from update_page response: {text}"
        );
    }

    // ===== Refinement queue guards =====

    #[tokio::test]
    async fn test_reject_refinement_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = RejectRefinementParams {
            id: "merge_abc_def".into(),
        };
        let result = server.reject_refinement_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_reject_refinement_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = RejectRefinementParams {
            id: "merge_abc_def".into(),
        };
        let result = server.reject_refinement_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
        );
    }

    #[tokio::test]
    async fn test_accept_refinement_blocked_on_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = AcceptRefinementParams {
            id: "merge_abc_def".into(),
            space: None,
        };
        let result = server.accept_refinement_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_accept_refinement_allowed_on_stdio_transport() {
        let server = make_server(TransportMode::Stdio, "agent", None);
        let params = AcceptRefinementParams {
            id: "merge_abc_def".into(),
            space: None,
        };
        let result = server.accept_refinement_impl(params).await.unwrap();
        assert!(
            result.is_error.unwrap_or(false),
            "should fail with connection error, not transport block"
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
            space: params.space,
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
            space: None,
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
    fn instructions_mention_how_user_thinks() {
        assert!(
            server_instructions().contains("how the user thinks"),
            "with_instructions must frame context as modeling how the user thinks"
        );
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
    fn context_description_frames_modeling_user() {
        let descriptions = tool_descriptions();
        let ctx = descriptions.get("context").expect("context tool exists");
        assert!(
            ctx.contains("how the user thinks"),
            "context description must frame the result as modeling how the user thinks, got: {ctx}"
        );
    }

    #[test]
    fn list_refinements_description_mentions_vocab_promote() {
        // Closed-set contract (spec §2.4): the list_refinements action enumeration
        // must include vocab_promote, or the action ships half-wired (Task 6 wired
        // the enum/parse/apply but missed this doc string).
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
    fn doctor_description_mentions_setup_mode() {
        let descriptions = tool_descriptions();
        let status = descriptions.get("doctor").expect("doctor tool exists");
        assert!(
            status.contains("Local Memory"),
            "doctor description must mention setup modes, got: {status}"
        );
        assert!(
            status.contains("On-device Model"),
            "doctor description must mention on-device setup, got: {status}"
        );
        assert!(
            status.contains("not part of the memory loop"),
            "doctor description must frame itself as diagnostic-only, got: {status}"
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

    // ===== Knowledge graph / page CRUD =====

    // --- CreateEntityParams ---

    #[test]
    fn test_create_entity_params_minimal() {
        let json = r#"{"name": "Alice", "entity_type": "person"}"#;
        let params: CreateEntityParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.name, "Alice");
        assert_eq!(params.entity_type, "person");
        assert!(params.space.is_none());
        assert!(params.confidence.is_none());
    }

    #[test]
    fn test_create_entity_params_full() {
        let json = r#"{
            "name": "PostgreSQL",
            "entity_type": "tool",
            "space": "origin",
            "confidence": 0.9
        }"#;
        let params: CreateEntityParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.name, "PostgreSQL");
        assert_eq!(params.entity_type, "tool");
        assert_eq!(params.space.as_deref(), Some("origin"));
        assert_eq!(params.confidence, Some(0.9));
    }

    #[test]
    fn test_create_entity_params_missing_name_fails() {
        let json = r#"{"entity_type": "person"}"#;
        let result = serde_json::from_str::<CreateEntityParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_entity_params_missing_type_fails() {
        let json = r#"{"name": "Alice"}"#;
        let result = serde_json::from_str::<CreateEntityParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_entity_request_body_shape() {
        let server = make_server(TransportMode::Stdio, "claude", None);
        let params = CreateEntityParams {
            name: "Wenlan".into(),
            entity_type: "project".into(),
            space: Some("origin".into()),
            confidence: Some(0.95),
        };
        let source_agent = server.resolve_source_agent(None);
        let req = CreateEntityRequest {
            name: params.name,
            entity_type: params.entity_type,
            space: params.space,
            source_agent,
            confidence: params.confidence,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["name"], "Wenlan");
        assert_eq!(json["entity_type"], "project");
        assert_eq!(json["space"], "origin");
        assert_eq!(json["source_agent"], "claude");
        assert!(json["confidence"].as_f64().unwrap() > 0.94);
    }

    // --- CreateRelationParams ---

    #[test]
    fn test_create_relation_params() {
        let json = r#"{
            "from_entity": "Alice",
            "to_entity": "Wenlan",
            "relation_type": "works_on"
        }"#;
        let params: CreateRelationParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.from_entity, "Alice");
        assert_eq!(params.to_entity, "Wenlan");
        assert_eq!(params.relation_type, "works_on");
    }

    #[test]
    fn test_create_relation_params_missing_field_fails() {
        let json = r#"{"from_entity": "Alice", "to_entity": "Wenlan"}"#;
        let result = serde_json::from_str::<CreateRelationParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_relation_request_body_shape() {
        let server = make_server(TransportMode::Stdio, "claude", None);
        let params = CreateRelationParams {
            from_entity: "Alice".into(),
            to_entity: "Wenlan".into(),
            relation_type: "prefers".into(),
        };
        let source_agent = server.resolve_source_agent(None);
        let req = CreateRelationRequest {
            from_entity: params.from_entity,
            to_entity: params.to_entity,
            relation_type: params.relation_type,
            source_agent,
            confidence: None,
            explanation: None,
            source_memory_id: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["from_entity"], "Alice");
        assert_eq!(json["to_entity"], "Wenlan");
        assert_eq!(json["relation_type"], "prefers");
        assert_eq!(json["source_agent"], "claude");
    }

    // --- CreatePageParams ---

    #[test]
    fn test_create_page_params_minimal() {
        let json = r#"{"title": "Wenlan daemon", "content": "Body text."}"#;
        let params: CreatePageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.title, "Wenlan daemon");
        assert_eq!(params.content, "Body text.");
        assert!(params.summary.is_none());
        assert!(params.entity_id.is_none());
        assert!(params.space.is_none());
        assert!(params.source_memory_ids.is_empty());
    }

    #[test]
    fn test_create_page_params_full() {
        let json = r##"{
            "title": "Wenlan daemon",
            "content": "Markdown body with [[wikilinks]].",
            "summary": "The headless HTTP daemon at the heart of Wenlan.",
            "entity_id": "ent_origin",
            "space": "origin",
            "source_memory_ids": ["mem_1", "mem_2"]
        }"##;
        let params: CreatePageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.title, "Wenlan daemon");
        assert_eq!(
            params.summary.as_deref(),
            Some("The headless HTTP daemon at the heart of Wenlan.")
        );
        assert_eq!(params.entity_id.as_deref(), Some("ent_origin"));
        assert_eq!(params.space.as_deref(), Some("origin"));
        assert_eq!(params.source_memory_ids, vec!["mem_1", "mem_2"]);
    }

    #[test]
    fn test_create_page_params_missing_required_fails() {
        let json = r#"{"title": "Only title"}"#;
        let result = serde_json::from_str::<CreatePageParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_page_request_body_shape() {
        let params = CreatePageParams {
            title: "Page".into(),
            content: "Body".into(),
            summary: Some("S".into()),
            entity_id: Some("ent_1".into()),
            space: Some("origin".into()),
            source_memory_ids: vec!["mem_1".into()],
        };
        let req = CreateConceptRequest {
            title: params.title,
            content: params.content,
            summary: params.summary,
            entity_id: params.entity_id,
            space: params.space,
            source_memory_ids: params.source_memory_ids,
            creation_kind: None,
            workspace: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["title"], "Page");
        assert_eq!(json["content"], "Body");
        assert_eq!(json["summary"], "S");
        assert_eq!(json["entity_id"], "ent_1");
        assert_eq!(json["space"], "origin");
        assert_eq!(json["source_memory_ids"], serde_json::json!(["mem_1"]));
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

    #[tokio::test]
    async fn delete_observation_refuses_http_transport() {
        let server = make_server(TransportMode::Http, "agent", None);
        let params = DeleteObservationParams {
            observation_id: "obs_123".to_string(),
        };
        let result = server.delete_observation_impl(params).await.unwrap();
        let content = &result.content[0];
        match content.raw {
            rmcp::model::RawContent::Text(ref tc) => {
                assert!(tc.text.contains("not available over remote connections"));
            }
            _ => panic!("expected text content"),
        }
    }

    // --- GetPageParams ---

    #[test]
    fn test_get_page_params() {
        let json = r#"{"page_id": "page_abc"}"#;
        let params: GetPageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.page_id, "page_abc");
    }

    #[test]
    fn test_get_page_params_missing_fails() {
        let json = r#"{}"#;
        let result = serde_json::from_str::<GetPageParams>(json);
        assert!(result.is_err());
    }

    // --- ListMemoriesParams ---

    #[test]
    fn test_list_memories_params_empty() {
        let json = r#"{}"#;
        let params: ListMemoriesParams = serde_json::from_str(json).unwrap();
        assert!(params.memory_type.is_none());
        assert!(params.space.is_none());
        assert!(params.limit.is_none());
    }

    #[test]
    fn test_list_memories_params_full() {
        let json = r#"{"memory_type": "decision", "space": "origin", "limit": 50}"#;
        let params: ListMemoriesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.memory_type.as_deref(), Some("decision"));
        assert_eq!(params.space.as_deref(), Some("origin"));
        assert_eq!(params.limit, Some(50));
    }

    #[test]
    fn test_list_memories_params_limit_as_string() {
        // MCP clients sometimes serialize numeric params as strings.
        let json = r#"{"limit": "25"}"#;
        let params: ListMemoriesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(25));
    }

    #[test]
    fn test_list_memories_request_body_shape() {
        let params = ListMemoriesParams {
            memory_type: Some("fact".into()),
            space: None,
            limit: Some(10),
        };
        let req = ListMemoriesRequest {
            memory_type: params.memory_type,
            space: params.space,
            limit: params.limit.unwrap_or(100),
            confirmed: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["memory_type"], "fact");
        assert!(json["space"].is_null());
        assert_eq!(json["limit"], 10);
    }

    #[test]
    fn test_list_memories_request_default_limit() {
        let params = ListMemoriesParams {
            memory_type: None,
            space: None,
            limit: None,
        };
        let req = ListMemoriesRequest {
            memory_type: params.memory_type,
            space: params.space,
            limit: params.limit.unwrap_or(100),
            confirmed: None,
        };
        assert_eq!(req.limit, 100);
    }

    // --- UpdatePageParams ---

    #[test]
    fn test_update_page_params_minimal() {
        let json =
            r#"{"page_id": "page_abc", "content": "fresh body", "source_memory_ids": ["mem_1"]}"#;
        let params: UpdatePageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.page_id, "page_abc");
        assert_eq!(params.content, "fresh body");
        assert_eq!(params.source_memory_ids, vec!["mem_1"]);
        assert!(params.summary.is_none());
    }

    #[test]
    fn test_update_page_params_with_summary() {
        let json = r#"{
            "page_id": "page_abc",
            "content": "body",
            "source_memory_ids": ["mem_1", "mem_2"],
            "summary": "Refreshed claim."
        }"#;
        let params: UpdatePageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.summary.as_deref(), Some("Refreshed claim."));
        assert_eq!(params.source_memory_ids.len(), 2);
    }

    #[test]
    fn test_update_page_params_missing_required_fails() {
        // Missing source_memory_ids is a hard fail — refresh without sources
        // would orphan the page from its provenance trail.
        let json = r#"{"page_id": "page_abc", "content": "body"}"#;
        let result = serde_json::from_str::<UpdatePageParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_page_request_body_shape() {
        let params = UpdatePageParams {
            page_id: "page_abc".into(),
            content: "Body".into(),
            source_memory_ids: vec!["mem_1".into()],
            summary: Some("S".into()),
        };
        let req = wenlan_types::requests::RefreshPageRequest {
            content: params.content,
            source_memory_ids: params.source_memory_ids,
            summary: params.summary,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["content"], "Body");
        assert_eq!(json["source_memory_ids"], serde_json::json!(["mem_1"]));
        assert_eq!(json["summary"], "S");
        // page_id stays in the URL, never the body.
        assert!(json.get("page_id").is_none());
    }

    // --- Tool registration ---

    #[test]
    fn new_crud_tools_are_registered() {
        let descriptions = tool_descriptions();
        for name in [
            "create_entity",
            "create_relation",
            "create_observation",
            "confirm_entity",
            "update_observation",
            "confirm_observation",
            "delete_observation",
            "create_page",
            "update_page",
            "delete_page",
            "get_page",
            "get_page_links",
            "list_memories",
            "search_pages",
            "list_pages_recent",
            "list_spaces",
        ] {
            assert!(
                descriptions.contains_key(name),
                "tool `{name}` must be registered, got: {:?}",
                descriptions.keys().collect::<Vec<_>>()
            );
        }
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
    fn create_entity_schema_documents_name_and_type() {
        let schema = serde_json::to_string(&schemars::schema_for!(CreateEntityParams))
            .expect("CreateEntityParams schema serializes");
        assert!(
            schema.contains("Canonical entity name"),
            "schema must describe `name` field"
        );
        assert!(
            schema.contains("Entity category"),
            "schema must describe `entity_type` field"
        );
    }

    #[test]
    fn create_page_schema_documents_traceability() {
        let schema = serde_json::to_string(&schemars::schema_for!(CreatePageParams))
            .expect("CreatePageParams schema serializes");
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

    // --- SearchPagesParams ---

    #[test]
    fn test_search_pages_params_minimal() {
        let json = r#"{"query": "mutex deadlock"}"#;
        let params: SearchPagesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "mutex deadlock");
        assert!(params.limit.is_none());
    }

    #[test]
    fn test_search_pages_params_full() {
        let json = r#"{"query": "distill architecture", "limit": 5}"#;
        let params: SearchPagesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "distill architecture");
        assert_eq!(params.limit, Some(5));
    }

    #[test]
    fn test_search_pages_params_missing_query_fails() {
        let json = r#"{"limit": 10}"#;
        let result = serde_json::from_str::<SearchPagesParams>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_pages_params_limit_as_string() {
        let json = r#"{"query": "x", "limit": "3"}"#;
        let params: SearchPagesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(3));
    }

    #[test]
    fn test_search_pages_request_body_shape() {
        let params = SearchPagesParams {
            query: "mutex".into(),
            limit: Some(7),
            page_type: None,
        };
        let req = SearchPagesRequest {
            query: params.query,
            limit: params.limit,
            page_type: params.page_type,
            space: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["query"], "mutex");
        assert_eq!(json["limit"], 7);
        assert!(json.get("space").is_none());
    }

    // --- ListPagesRecentParams ---

    #[test]
    fn test_list_pages_recent_params_empty() {
        let json = r#"{}"#;
        let params: ListPagesRecentParams = serde_json::from_str(json).unwrap();
        assert!(params.limit.is_none());
        assert!(params.since_ms.is_none());
    }

    #[test]
    fn test_list_pages_recent_params_full() {
        let json = r#"{"limit": 20, "since_ms": 1715000000000}"#;
        let params: ListPagesRecentParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(20));
        assert_eq!(params.since_ms, Some(1715000000000));
    }

    #[test]
    fn test_list_pages_recent_params_string_numbers() {
        let json = r#"{"limit": "15", "since_ms": "1715000000000"}"#;
        let params: ListPagesRecentParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(15));
        assert_eq!(params.since_ms, Some(1715000000000));
    }

    #[test]
    fn list_pages_recent_url_construction() {
        // Exercises the actual builder used by `list_pages_recent_impl` so the
        // test cannot drift from production behavior.
        assert_eq!(build_recent_pages_path(None, None), "/api/pages/recent");
        assert_eq!(
            build_recent_pages_path(Some(5), None),
            "/api/pages/recent?limit=5"
        );
        assert_eq!(
            build_recent_pages_path(None, Some(123)),
            "/api/pages/recent?since_ms=123"
        );
        assert_eq!(
            build_recent_pages_path(Some(10), Some(456)),
            "/api/pages/recent?limit=10&since_ms=456"
        );
        // Negative since_ms (i64 — sentinel like "-1" must still serialize).
        assert_eq!(
            build_recent_pages_path(None, Some(-1)),
            "/api/pages/recent?since_ms=-1"
        );
    }

    #[test]
    fn search_pages_and_list_pages_recent_are_read_only() {
        let server = make_server(TransportMode::Stdio, "test", None);
        for name in ["search_pages", "list_pages_recent"] {
            let tool = server
                .tool_router
                .list_all()
                .into_iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("`{name}` registered"));
            let ann = tool.annotations.as_ref().expect("annotations present");
            assert_eq!(
                ann.read_only_hint,
                Some(true),
                "`{name}` must declare read_only_hint=true"
            );
        }
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
        // Daemon must not wrap successful response under an extra key — the
        // lesson_mcp_typed_deserialize guard. This test verifies a non-typed
        // shape fails to deserialize loud.
        let wrong = r#"{"data":{"id":"ref_xyz","action_applied":"entity_merge"}}"#;
        let result: Result<AcceptRefinementResponse, _> = serde_json::from_str(wrong);
        assert!(
            result.is_err(),
            "envelope-wrapped response must fail typed deserialize"
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
}
