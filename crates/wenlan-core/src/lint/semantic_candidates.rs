use super::super::context::{LintContext, ScopeFilter};
use regex::{Regex, RegexSet};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use wenlan_types::lint::{
    LintAgentCandidate, LintAgentRecord, LintAgentRecordKind, LintAgentWork, LintDigest,
    LintSemanticAction, LintSemanticCandidateKind, LintSemanticCheckId, LintSemanticPopulation,
    LintSemanticReasonCode, LINT_AGENT_CANDIDATE_CAP, LINT_AGENT_EXCERPT_CHAR_CAP,
    LINT_AGENT_RECORD_CAP,
};

const PER_CHECK_CANDIDATE_CAP: u64 = 6;
const PAGE_MEMORY_TOP_K: usize = 5;
const STALE_AFTER_SECONDS: i64 = 365 * 24 * 60 * 60;

#[derive(Clone)]
struct Record {
    key: String,
    kind: LintAgentRecordKind,
    excerpt: String,
    memory_type: Option<String>,
    evidence_count: Option<u64>,
    source_excerpt: Option<String>,
}

#[derive(Clone)]
struct Memory {
    id: String,
    content: String,
    memory_type: Option<String>,
    space: Option<String>,
    last_modified: i64,
    content_tokens: BTreeSet<String>,
}

#[derive(Clone)]
struct Entity {
    id: String,
    name: String,
    space: Option<String>,
    selected: bool,
}

struct Relation {
    from_entity: String,
    to_entity: String,
    relation_type: String,
}

#[derive(Clone)]
struct Page {
    id: String,
    content: String,
    workspace: Option<String>,
    creation_kind: String,
    review_status: String,
    content_tokens: BTreeSet<String>,
}

struct PageEvidence {
    page_id: String,
    source_kind: String,
    locator: String,
}

struct EntityMatcher {
    entity_indexes: Vec<usize>,
    patterns: RegexSet,
}

struct MemoryTokenIndex {
    memory_count: usize,
    postings: BTreeMap<String, Vec<usize>>,
}

pub(super) struct CandidateSet {
    work: LintAgentWork,
    record_ids: Vec<LintDigest>,
}

impl CandidateSet {
    pub(super) fn work(&self) -> &LintAgentWork {
        &self.work
    }

    pub(super) fn record_id(&self, reference: u16) -> Option<LintDigest> {
        self.record_ids
            .get(usize::from(reference.checked_sub(1)?))
            .cloned()
    }
}

pub(super) async fn load(context: &LintContext<'_, '_>) -> Result<CandidateSet, ()> {
    let LoadedMemories {
        memories,
        population_digest: memory_population_digest,
    } = load_memories(context).await?;
    let links = load_memory_entity_links(context).await?;
    let relations = load_relations(context).await?;
    let entities = load_entities(context).await?;
    let pages = load_pages(context).await?;
    let page_evidence = load_page_evidence(context).await?;
    let mut builder = Builder::new(context, &memories, &entities, &pages);
    let mut work_units = 0_usize;

    for memory in &memories {
        candidate_checkpoint(context, &mut work_units).await?;
        if memory.memory_type.as_deref().is_none_or(str::is_empty) {
            builder.add(
                LintSemanticCheckId::MemoryClassification,
                LintSemanticCandidateKind::RecordReview,
                vec![memory_record(memory)],
                Vec::new(),
                LintSemanticAction::ReclassifyMemory,
                LintSemanticReasonCode::ClassificationMismatch,
            );
        }
        if context
            .clock()
            .epoch_seconds()
            .saturating_sub(memory.last_modified)
            > STALE_AFTER_SECONDS
        {
            builder.add(
                LintSemanticCheckId::MemoryStaleness,
                LintSemanticCandidateKind::RecordReview,
                vec![memory_record(memory)],
                Vec::new(),
                LintSemanticAction::ReviewStaleness,
                LintSemanticReasonCode::PotentialStaleness,
            );
        }
    }

    let memory_by_id = memories
        .iter()
        .map(|memory| (memory.id.as_str(), memory))
        .collect::<BTreeMap<_, _>>();
    let entity_by_id = entities
        .iter()
        .map(|entity| (entity.id.as_str(), entity))
        .collect::<BTreeMap<_, _>>();
    let entity_matchers = build_entity_matchers(&entities)?;
    let memory_token_indexes = build_memory_token_indexes(&memories);
    let faithfulness_eligible = pages
        .iter()
        .filter(|page| {
            page_evidence.iter().any(|evidence| {
                evidence.page_id == page.id
                    && evidence.source_kind == "memory"
                    && memory_by_id.contains_key(evidence.locator.as_str())
            })
        })
        .count() as u64;
    builder.set_eligible(LintSemanticCheckId::PageFaithfulness, faithfulness_eligible);
    let mut associated = BTreeMap::<String, BTreeSet<String>>::new();
    let mut mentioned_by_memory = BTreeMap::<String, Vec<usize>>::new();
    let mut missing_memory_entity_links = Vec::<(usize, usize)>::new();
    for (memory_index, memory) in memories.iter().enumerate() {
        candidate_checkpoint(context, &mut work_units).await?;
        let mentioned = mentioned_entity_indexes(memory, &entity_matchers);
        for entity_index in &mentioned {
            let entity = &entities[*entity_index];
            associated
                .entry(entity.id.clone())
                .or_default()
                .insert(memory.id.clone());
            if !links.contains(&(memory.id.clone(), entity.id.clone())) {
                missing_memory_entity_links.push((memory_index, *entity_index));
            }
        }
        mentioned_by_memory.insert(memory.id.clone(), mentioned);
    }
    for (memory_id, entity_id) in &links {
        candidate_checkpoint(context, &mut work_units).await?;
        let (Some(memory), Some(entity)) = (
            memory_by_id.get(memory_id.as_str()),
            entity_by_id.get(entity_id.as_str()),
        ) else {
            continue;
        };
        associated
            .entry(entity.id.clone())
            .or_default()
            .insert(memory.id.clone());
        if !same_scope(memory.space.as_deref(), entity.space.as_deref())
            || !contains_phrase(&memory.content, &entity.name)
        {
            builder.add(
                LintSemanticCheckId::MemoryEntityLinks,
                LintSemanticCandidateKind::ExistingLink,
                vec![memory_record(memory), entity_record(entity)],
                Vec::new(),
                LintSemanticAction::RemoveMemoryEntityLink,
                LintSemanticReasonCode::ExistingLinkMismatch,
            );
        }
    }
    for (memory_index, entity_index) in missing_memory_entity_links {
        candidate_checkpoint(context, &mut work_units).await?;
        builder.add(
            LintSemanticCheckId::MemoryEntityLinks,
            LintSemanticCandidateKind::MissingLink,
            vec![
                memory_record(&memories[memory_index]),
                entity_record(&entities[entity_index]),
            ],
            Vec::new(),
            LintSemanticAction::AddMemoryEntityLink,
            LintSemanticReasonCode::MentionWithoutLink,
        );
    }

    for memory_ids in associated.values() {
        candidate_checkpoint(context, &mut work_units).await?;
        let ordered = memory_ids
            .iter()
            .filter_map(|id| memory_by_id.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        for pair in ordered.windows(2) {
            candidate_checkpoint(context, &mut work_units).await?;
            builder.add(
                LintSemanticCheckId::MemoryContradiction,
                LintSemanticCandidateKind::PairReview,
                vec![memory_record(pair[0]), memory_record(pair[1])],
                Vec::new(),
                LintSemanticAction::ReviewContradiction,
                LintSemanticReasonCode::PotentialContradiction,
            );
        }
    }

    let relation_pairs = relations
        .iter()
        .map(|relation| (relation.from_entity.clone(), relation.to_entity.clone()))
        .collect::<BTreeSet<_>>();
    for relation in &relations {
        candidate_checkpoint(context, &mut work_units).await?;
        let (Some(left), Some(right)) = (
            entity_by_id.get(relation.from_entity.as_str()),
            entity_by_id.get(relation.to_entity.as_str()),
        ) else {
            continue;
        };
        if same_scope(left.space.as_deref(), right.space.as_deref()) {
            continue;
        }
        builder.add(
            LintSemanticCheckId::EntityRelations,
            LintSemanticCandidateKind::ExistingLink,
            vec![
                relation_entity_record(left, &relation.relation_type, "from"),
                relation_entity_record(right, &relation.relation_type, "to"),
            ],
            Vec::new(),
            LintSemanticAction::RemoveEntityRelation,
            LintSemanticReasonCode::ExistingRelationMismatch,
        );
    }
    for memory in &memories {
        candidate_checkpoint(context, &mut work_units).await?;
        let mentioned = mentioned_by_memory
            .get(&memory.id)
            .into_iter()
            .flatten()
            .map(|index| &entities[*index])
            .collect::<Vec<_>>();
        for pair in mentioned.windows(2) {
            candidate_checkpoint(context, &mut work_units).await?;
            if !relation_pairs.contains(&(pair[0].id.clone(), pair[1].id.clone()))
                && !relation_pairs.contains(&(pair[1].id.clone(), pair[0].id.clone()))
            {
                builder.add(
                    LintSemanticCheckId::EntityRelations,
                    LintSemanticCandidateKind::MissingLink,
                    vec![
                        entity_record(pair[0]),
                        entity_record(pair[1]),
                        memory_record(memory),
                    ],
                    Vec::new(),
                    LintSemanticAction::AddEntityRelation,
                    LintSemanticReasonCode::SharedContextWithoutRelation,
                );
            }
        }
    }

    let evidence_set = page_evidence
        .iter()
        .filter(|evidence| evidence.source_kind == "memory")
        .map(|evidence| (evidence.page_id.clone(), evidence.locator.clone()))
        .collect::<BTreeSet<_>>();
    let mut missing_page_evidence = Vec::<(u16, usize, usize)>::new();
    for (page_index, page) in pages.iter().enumerate() {
        candidate_checkpoint(context, &mut work_units).await?;
        let linked = page_evidence
            .iter()
            .filter(|evidence| evidence.page_id == page.id && evidence.source_kind == "memory")
            .filter_map(|evidence| memory_by_id.get(evidence.locator.as_str()).copied())
            .collect::<Vec<_>>();
        let has_provenance = page_evidence
            .iter()
            .any(|evidence| evidence.page_id == page.id);
        if !has_provenance {
            builder.add(
                LintSemanticCheckId::PageProvenanceAdequacy,
                LintSemanticCandidateKind::RecordReview,
                vec![page_record(page, 0, None)],
                Vec::new(),
                LintSemanticAction::ReviewPageClaim,
                LintSemanticReasonCode::PotentialInadequateProvenance,
            );
        }
        if let Some(linked_memory) = linked.first() {
            builder.add(
                LintSemanticCheckId::PageFaithfulness,
                LintSemanticCandidateKind::PairReview,
                vec![
                    page_record(page, linked.len() as u64, Some(&linked_memory.content)),
                    memory_record(linked_memory),
                ],
                Vec::new(),
                LintSemanticAction::ReviewPageClaim,
                LintSemanticReasonCode::PotentialUnfaithfulClaim,
            );
        }
        for memory in linked {
            if same_scope(page.workspace.as_deref(), memory.space.as_deref())
                && token_overlap_count(&page.content_tokens, &memory.content_tokens) > 0
            {
                continue;
            }
            builder.add(
                LintSemanticCheckId::PageEvidenceLinks,
                LintSemanticCandidateKind::ExistingLink,
                vec![
                    page_record(page, 1, Some(&memory.content)),
                    memory_record(memory),
                ],
                Vec::new(),
                LintSemanticAction::RemovePageEvidence,
                LintSemanticReasonCode::ExistingEvidenceMismatch,
            );
        }
        for (memory_index, shared_tokens) in related_memory_indexes(page, &memory_token_indexes) {
            let memory = &memories[memory_index];
            if evidence_set.contains(&(page.id.clone(), memory.id.clone())) {
                continue;
            }
            missing_page_evidence.push((shared_tokens, page_index, memory_index));
        }
    }
    missing_page_evidence.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| pages[left.1].id.cmp(&pages[right.1].id))
            .then_with(|| memories[left.2].id.cmp(&memories[right.2].id))
    });
    for (_, page_index, memory_index) in missing_page_evidence {
        candidate_checkpoint(context, &mut work_units).await?;
        builder.add(
            LintSemanticCheckId::PageEvidenceLinks,
            LintSemanticCandidateKind::MissingLink,
            vec![
                page_record(&pages[page_index], 0, None),
                memory_record(&memories[memory_index]),
            ],
            Vec::new(),
            LintSemanticAction::AddPageEvidence,
            LintSemanticReasonCode::ClaimOverlapWithoutEvidence,
        );
    }

    let population_digest = source_population_digest(
        memory_population_digest,
        &entities,
        &links,
        &relation_pairs,
        &pages,
        &page_evidence,
    );
    builder.finish(context, population_digest)
}

async fn candidate_checkpoint(
    context: &LintContext<'_, '_>,
    work_units: &mut usize,
) -> Result<(), ()> {
    context
        .gate()
        .check_run_for(context.profile(), context.clock().elapsed())
        .map_err(|_| ())?;
    *work_units = work_units.saturating_add(1);
    if !work_units.is_multiple_of(128) {
        return Ok(());
    }
    tokio::task::yield_now().await;
    context
        .gate()
        .check_run_for(context.profile(), context.clock().elapsed())
        .map_err(|_| ())
}

struct Builder {
    eligible: BTreeMap<LintSemanticCheckId, u64>,
    candidate_counts: BTreeMap<LintSemanticCheckId, u64>,
    packet_counts: BTreeMap<LintSemanticCheckId, u64>,
    pending: BTreeMap<LintSemanticCheckId, Vec<PendingCandidate>>,
    records: Vec<Record>,
    record_refs: BTreeMap<String, u16>,
    candidates: Vec<LintAgentCandidate>,
    candidate_keys: BTreeSet<String>,
}

struct PendingCandidate {
    key: String,
    risk: (u16, usize),
    kind: LintSemanticCandidateKind,
    evidence: Vec<Record>,
    counterevidence: Vec<Record>,
    action: LintSemanticAction,
    reason: LintSemanticReasonCode,
}

impl Builder {
    fn new(
        _context: &LintContext<'_, '_>,
        memories: &[Memory],
        entities: &[Entity],
        pages: &[Page],
    ) -> Self {
        let mut eligible = LintSemanticCheckId::ALL
            .into_iter()
            .map(|id| (id, 0))
            .collect::<BTreeMap<_, _>>();
        for id in [
            LintSemanticCheckId::MemoryClassification,
            LintSemanticCheckId::MemoryContradiction,
            LintSemanticCheckId::MemoryStaleness,
            LintSemanticCheckId::MemoryEntityLinks,
        ] {
            eligible.insert(id, memories.len() as u64);
        }
        eligible.insert(
            LintSemanticCheckId::EntityRelations,
            entities.iter().filter(|entity| entity.selected).count() as u64,
        );
        for id in [
            LintSemanticCheckId::PageFaithfulness,
            LintSemanticCheckId::PageProvenanceAdequacy,
            LintSemanticCheckId::PageEvidenceLinks,
        ] {
            eligible.insert(id, pages.len() as u64);
        }
        Self {
            eligible,
            candidate_counts: BTreeMap::new(),
            packet_counts: BTreeMap::new(),
            pending: BTreeMap::new(),
            records: Vec::new(),
            record_refs: BTreeMap::new(),
            candidates: Vec::new(),
            candidate_keys: BTreeSet::new(),
        }
    }

    fn set_eligible(&mut self, check_id: LintSemanticCheckId, eligible: u64) {
        self.eligible.insert(check_id, eligible);
    }

    fn add(
        &mut self,
        check_id: LintSemanticCheckId,
        kind: LintSemanticCandidateKind,
        evidence: Vec<Record>,
        counterevidence: Vec<Record>,
        action: LintSemanticAction,
        reason: LintSemanticReasonCode,
    ) {
        let mut evidence_keys = evidence
            .iter()
            .map(|record| record.key.as_str())
            .collect::<Vec<_>>();
        evidence_keys.sort_unstable();
        evidence_keys.dedup();
        let mut counterevidence_keys = counterevidence
            .iter()
            .map(|record| record.key.as_str())
            .collect::<Vec<_>>();
        counterevidence_keys.sort_unstable();
        counterevidence_keys.dedup();
        let candidate_key = format!(
            "{check_id:?}|{kind:?}|{action:?}|{reason:?}|{}|{}",
            evidence_keys.join(","),
            counterevidence_keys.join(",")
        );
        if !self.candidate_keys.insert(candidate_key.clone()) {
            return;
        }
        *self.candidate_counts.entry(check_id).or_insert(0) += 1;
        let risk = candidate_risk(action, &evidence);
        let pending = self.pending.entry(check_id).or_default();
        pending.push(PendingCandidate {
            key: candidate_key,
            risk,
            kind,
            evidence,
            counterevidence,
            action,
            reason,
        });
        pending.sort_by(compare_pending_candidates);
        pending.truncate(PER_CHECK_CANDIDATE_CAP as usize);
    }

    fn materialize(&mut self, check_id: LintSemanticCheckId, pending: PendingCandidate) {
        if self.packet_counts.get(&check_id).copied().unwrap_or(0) >= PER_CHECK_CANDIDATE_CAP
            || self.candidates.len() >= LINT_AGENT_CANDIDATE_CAP
        {
            return;
        }
        let needed = pending
            .evidence
            .iter()
            .chain(&pending.counterevidence)
            .filter(|record| !self.record_refs.contains_key(&record.key))
            .count();
        if self.records.len().saturating_add(needed) > LINT_AGENT_RECORD_CAP {
            return;
        }
        let mut evidence_refs = pending
            .evidence
            .into_iter()
            .map(|record| self.record_ref(record))
            .collect::<Vec<_>>();
        evidence_refs.sort_unstable();
        evidence_refs.dedup();
        let mut counterevidence_refs = pending
            .counterevidence
            .into_iter()
            .map(|record| self.record_ref(record))
            .collect::<Vec<_>>();
        counterevidence_refs.sort_unstable();
        counterevidence_refs.dedup();
        let reference = u16::try_from(self.candidates.len() + 1).unwrap_or(u16::MAX);
        self.candidates.push(
            LintAgentCandidate::try_new(
                reference,
                check_id,
                pending.kind,
                evidence_refs,
                counterevidence_refs,
                pending.action,
                pending.reason,
            )
            .expect("candidate builder produces valid typed work"),
        );
        *self.packet_counts.entry(check_id).or_insert(0) += 1;
    }

    fn record_ref(&mut self, record: Record) -> u16 {
        if let Some(reference) = self.record_refs.get(&record.key) {
            return *reference;
        }
        let reference = u16::try_from(self.records.len() + 1).unwrap_or(u16::MAX);
        self.record_refs.insert(record.key.clone(), reference);
        self.records.push(record);
        reference
    }

    fn finish(
        mut self,
        context: &LintContext<'_, '_>,
        population_digest: [u8; 32],
    ) -> Result<CandidateSet, ()> {
        for check_id in LintSemanticCheckId::ALL {
            let mut pending = self.pending.remove(&check_id).unwrap_or_default();
            pending.sort_by(compare_pending_candidates);
            for candidate in pending {
                self.materialize(check_id, candidate);
            }
        }
        let populations = LintSemanticCheckId::ALL
            .into_iter()
            .map(|check_id| {
                let candidates = self.candidate_counts.get(&check_id).copied().unwrap_or(0);
                let packet = self.packet_counts.get(&check_id).copied().unwrap_or(0);
                LintSemanticPopulation::try_new(
                    check_id,
                    self.eligible.get(&check_id).copied().unwrap_or(0),
                    candidates,
                    packet,
                    packet < candidates,
                )
                .map_err(|_| ())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let typed_records = self
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                LintAgentRecord::try_new(
                    u16::try_from(index + 1).map_err(|_| ())?,
                    record.kind,
                    record.excerpt.clone(),
                    record.memory_type.clone(),
                    record.evidence_count,
                    record.source_excerpt.clone(),
                )
                .map_err(|_| ())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let record_ids = self
            .records
            .iter()
            .map(|record| super::semantic_record_key_digest(&record.key))
            .collect::<Vec<_>>();
        let mut digest = Sha256::new();
        digest.update(b"wenlan-lint-semantic-candidates-v2");
        match context.scope().filter() {
            ScopeFilter::Global => digest.update(b"scope:global"),
            ScopeFilter::Registered(value) => {
                digest.update(b"scope:registered");
                digest_value(&mut digest, value.as_bytes());
            }
            ScopeFilter::Uncategorized => digest.update(b"scope:uncategorized"),
        }
        digest.update(population_digest);
        digest.update(
            context
                .snapshot()
                .analysis_digest()
                .map_err(|_| ())?
                .as_bytes(),
        );
        digest.update(
            context
                .page_scan()
                .map(|scan| scan.normalized_bytes())
                .unwrap_or([0; 32]),
        );
        digest.update(serde_json::to_vec(&populations).map_err(|_| ())?);
        digest.update(serde_json::to_vec(&typed_records).map_err(|_| ())?);
        digest.update(serde_json::to_vec(&self.candidates).map_err(|_| ())?);
        let digest: [u8; 32] = digest.finalize().into();
        let work_digest =
            LintDigest::from_u64(u64::from_le_bytes(digest[..8].try_into().map_err(|_| ())?));
        let work = LintAgentWork::try_new(work_digest, populations, typed_records, self.candidates)
            .map_err(|_| ())?;
        Ok(CandidateSet { work, record_ids })
    }
}

fn compare_pending_candidates(
    left: &PendingCandidate,
    right: &PendingCandidate,
) -> std::cmp::Ordering {
    right
        .risk
        .cmp(&left.risk)
        .then_with(|| left.key.cmp(&right.key))
}

fn candidate_risk(action: LintSemanticAction, evidence: &[Record]) -> (u16, usize) {
    let action_priority = match action {
        LintSemanticAction::RemoveMemoryEntityLink
        | LintSemanticAction::RemoveEntityRelation
        | LintSemanticAction::RemovePageEvidence => 500,
        LintSemanticAction::SupersedeMemory | LintSemanticAction::ReviewContradiction => 400,
        LintSemanticAction::ReviewPageClaim | LintSemanticAction::ReviewRetrieval => 300,
        LintSemanticAction::ReclassifyMemory | LintSemanticAction::ReviewStaleness => 200,
        LintSemanticAction::AddMemoryEntityLink
        | LintSemanticAction::AddEntityRelation
        | LintSemanticAction::AddPageEvidence => 100,
    };
    let shared_tokens = evidence
        .first()
        .zip(evidence.get(1))
        .map(|(left, right)| {
            let left = content_tokens(&left.excerpt);
            let right = content_tokens(&right.excerpt);
            left.intersection(&right).count()
        })
        .unwrap_or(0);
    (action_priority, shared_tokens)
}

struct LoadedMemories {
    memories: Vec<Memory>,
    population_digest: [u8; 32],
}

async fn load_memories(context: &LintContext<'_, '_>) -> Result<LoadedMemories, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "m.space");
    let mut rows = context.snapshot().query(
        &format!("SELECT m.id,m.source_id,m.chunk_index,m.content,m.memory_type,m.space,m.last_modified FROM memories m WHERE m.source='memory' AND m.pending_revision=0 AND COALESCE(m.is_recap,0)=0 AND m.supersede_mode!='evicted'{scope} ORDER BY m.source_id,m.chunk_index,m.id"),
        params,
    ).await.map_err(|_| ())?;
    let mut output = Vec::new();
    let mut digest = Sha256::new();
    digest.update(b"wenlan-lint-semantic-memory-population-v2");
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let row_id: String = row.get(0).map_err(|_| ())?;
        let source_id: String = row.get(1).map_err(|_| ())?;
        let chunk_index: i64 = row.get(2).map_err(|_| ())?;
        let content: String = row.get(3).map_err(|_| ())?;
        let memory_type: Option<String> = row.get(4).map_err(|_| ())?;
        // M3 PR-1 stage e: `memories.space` is NOT NULL as of migration 91,
        // so an unscoped memory now carries the reserved `UNFILED_SPACE_ID`
        // sentinel rather than NULL. Translate it back to None here, same as
        // load_pages already does for `p.workspace` (M1), so uncategorized
        // memories keep grouping/matching against uncategorized (NULL-space)
        // entities in build_entity_matchers/mentioned_entity_indexes and
        // same_scope below.
        let space: Option<String> = row
            .get::<Option<String>>(5)
            .map_err(|_| ())?
            .filter(|s| s != crate::db::UNFILED_SPACE_ID);
        let last_modified: i64 = row.get(6).map_err(|_| ())?;
        digest_value(&mut digest, row_id.as_bytes());
        digest_value(&mut digest, source_id.as_bytes());
        digest.update(chunk_index.to_le_bytes());
        digest_value(&mut digest, content.as_bytes());
        digest_optional(&mut digest, memory_type.as_deref());
        digest_optional(&mut digest, space.as_deref());
        digest.update(last_modified.to_le_bytes());
        if chunk_index == 0 {
            output.push(Memory {
                id: source_id,
                content_tokens: content_tokens(&content),
                content,
                memory_type,
                space,
                last_modified,
            });
        }
    }
    Ok(LoadedMemories {
        memories: output,
        population_digest: digest.finalize().into(),
    })
}

async fn load_entities(context: &LintContext<'_, '_>) -> Result<Vec<Entity>, ()> {
    let (scope, params) = entity_scope_clause(context.scope().filter());
    let mut rows = context.snapshot().query(
        &format!("SELECT e.id,e.name,e.space FROM entities e WHERE TRIM(e.name)!=''{scope} ORDER BY e.id"),
        params,
    ).await.map_err(|_| ())?;
    let mut output = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let space: Option<String> = row.get(2).map_err(|_| ())?;
        output.push(Entity {
            id: row.get(0).map_err(|_| ())?,
            name: row.get(1).map_err(|_| ())?,
            selected: scope_matches(context.scope().filter(), space.as_deref()),
            space,
        });
    }
    Ok(output)
}

fn entity_scope_clause(scope: &ScopeFilter) -> (String, libsql::params::Params) {
    let memory_filter = "m.source='memory' AND m.pending_revision=0 AND COALESCE(m.is_recap,0)=0 AND m.supersede_mode!='evicted'";
    match scope {
        ScopeFilter::Global => (String::new(), libsql::params::Params::None),
        ScopeFilter::Registered(value) => (
            format!(
                " AND (e.space=?1
                    OR e.id IN (
                        SELECT me.entity_id FROM memory_entities me
                        JOIN memories m ON m.source_id=me.memory_id
                        WHERE {memory_filter} AND m.space=?1
                    )
                    OR e.id IN (
                        SELECT r.to_entity FROM relations r
                        JOIN entities source ON source.id=r.from_entity
                        WHERE source.space=?1
                    ))"
            ),
            libsql::params::Params::Positional(vec![libsql::Value::Text(value.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            format!(
                " AND (e.space IS NULL
                    OR e.id IN (
                        SELECT me.entity_id FROM memory_entities me
                        JOIN memories m ON m.source_id=me.memory_id
                        WHERE {memory_filter} AND m.space IS NULL
                    )
                    OR e.id IN (
                        SELECT r.to_entity FROM relations r
                        JOIN entities source ON source.id=r.from_entity
                        WHERE source.space IS NULL
                    ))"
            ),
            libsql::params::Params::None,
        ),
    }
}

fn scope_matches(scope: &ScopeFilter, value: Option<&str>) -> bool {
    match scope {
        ScopeFilter::Global => true,
        ScopeFilter::Registered(selected) => value == Some(selected.as_str()),
        ScopeFilter::Uncategorized => value.is_none(),
    }
}

async fn load_memory_entity_links(
    context: &LintContext<'_, '_>,
) -> Result<BTreeSet<(String, String)>, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "m.space");
    let mut rows = context.snapshot().query(
        &format!("SELECT me.memory_id,me.entity_id FROM memory_entities me JOIN memories m ON m.source_id=me.memory_id AND m.source='memory' WHERE 1=1{scope} ORDER BY me.memory_id,me.entity_id"),
        params,
    ).await.map_err(|_| ())?;
    let mut output = BTreeSet::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        output.insert((row.get(0).map_err(|_| ())?, row.get(1).map_err(|_| ())?));
    }
    Ok(output)
}

async fn load_relations(context: &LintContext<'_, '_>) -> Result<Vec<Relation>, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "source.space");
    let mut rows = context.snapshot().query(
        &format!("SELECT r.from_entity,r.to_entity,r.relation_type FROM relations r JOIN entities source ON source.id=r.from_entity WHERE 1=1{scope} ORDER BY r.from_entity,r.to_entity,r.id"),
        params,
    ).await.map_err(|_| ())?;
    let mut output = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        output.push(Relation {
            from_entity: row.get(0).map_err(|_| ())?,
            to_entity: row.get(1).map_err(|_| ())?,
            relation_type: row.get(2).map_err(|_| ())?,
        });
    }
    Ok(output)
}

async fn load_pages(context: &LintContext<'_, '_>) -> Result<Vec<Page>, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "p.workspace");
    let mut rows = context.snapshot().query(
        &format!("SELECT p.id,p.content,p.workspace,p.creation_kind,p.review_status FROM pages p WHERE p.status='active'{scope} ORDER BY p.id"),
        params,
    ).await.map_err(|_| ())?;
    let mut output = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let content: String = row.get(1).map_err(|_| ())?;
        output.push(Page {
            id: row.get(0).map_err(|_| ())?,
            content_tokens: content_tokens(&content),
            content,
            // M1: pages store the reserved scope sentinel id; translate it back
            // to None here (as row_to_page does) so uncategorized pages compare
            // symmetric with uncategorized memories (space = NULL) in same_scope.
            workspace: row
                .get::<Option<String>>(2)
                .map_err(|_| ())?
                .filter(|s| s != crate::db::UNFILED_SPACE_ID),
            creation_kind: row.get(3).map_err(|_| ())?,
            review_status: row.get(4).map_err(|_| ())?,
        });
    }
    Ok(output)
}

async fn load_page_evidence(context: &LintContext<'_, '_>) -> Result<Vec<PageEvidence>, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "p.workspace");
    let mut rows = context.snapshot().query(
        &format!("SELECT pe.page_id,pe.source_kind,pe.locator FROM page_evidence pe JOIN pages p ON p.id=pe.page_id WHERE pe.locator IS NOT NULL{scope} ORDER BY pe.page_id,pe.source_kind,pe.locator"),
        params,
    ).await.map_err(|_| ())?;
    let mut output = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        output.push(PageEvidence {
            page_id: row.get(0).map_err(|_| ())?,
            source_kind: row.get(1).map_err(|_| ())?,
            locator: row.get(2).map_err(|_| ())?,
        });
    }
    Ok(output)
}

fn scope_clause(scope: &ScopeFilter, column: &str) -> (String, libsql::params::Params) {
    match scope {
        ScopeFilter::Global => (String::new(), libsql::params::Params::None),
        ScopeFilter::Registered(value) => (
            format!(" AND {column}=?1"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(value.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            format!(" AND {column} IS NULL"),
            libsql::params::Params::None,
        ),
    }
}

fn memory_record(memory: &Memory) -> Record {
    Record {
        key: format!("memory:{}", memory.id),
        kind: LintAgentRecordKind::Memory,
        excerpt: contextual_excerpt(memory.space.as_deref(), &memory.content),
        memory_type: memory.memory_type.clone(),
        evidence_count: None,
        source_excerpt: None,
    }
}

fn entity_record(entity: &Entity) -> Record {
    Record {
        key: format!("entity:{}", entity.id),
        kind: LintAgentRecordKind::Entity,
        excerpt: contextual_excerpt(entity.space.as_deref(), &entity.name),
        memory_type: None,
        evidence_count: None,
        source_excerpt: None,
    }
}

fn relation_entity_record(entity: &Entity, relation_type: &str, endpoint: &str) -> Record {
    let scope = entity.space.as_deref().unwrap_or("uncategorized");
    Record {
        key: format!("relation-entity:{}:{relation_type}:{endpoint}", entity.id),
        kind: LintAgentRecordKind::Entity,
        excerpt: contextual_excerpt_with(
            &[
                ("scope", scope),
                ("relation_type", relation_type),
                ("endpoint", endpoint),
            ],
            &entity.name,
        ),
        memory_type: None,
        evidence_count: None,
        source_excerpt: None,
    }
}

fn page_record(page: &Page, evidence_count: u64, source: Option<&str>) -> Record {
    let scope = page.workspace.as_deref().unwrap_or("uncategorized");
    Record {
        key: format!("page:{}", page.id),
        kind: LintAgentRecordKind::Page,
        excerpt: contextual_excerpt_with(
            &[
                ("scope", scope),
                ("creation_kind", &page.creation_kind),
                ("review_status", &page.review_status),
            ],
            &page.content,
        ),
        memory_type: None,
        evidence_count: Some(evidence_count),
        source_excerpt: source.map(bounded),
    }
}

fn contextual_excerpt(scope: Option<&str>, value: &str) -> String {
    contextual_excerpt_with(&[("scope", scope.unwrap_or("uncategorized"))], value)
}

fn contextual_excerpt_with(fields: &[(&str, &str)], value: &str) -> String {
    let metadata = fields
        .iter()
        .map(|(key, value)| format!("{key}={}", context_value(value)))
        .collect::<Vec<_>>()
        .join(" ");
    let prefix = format!("[lint_context {metadata}]\n");
    let content = redact_excerpt(value);
    prefix
        .chars()
        .chain(content.chars())
        .take(LINT_AGENT_EXCERPT_CHAR_CAP)
        .collect()
}

fn context_value(value: &str) -> String {
    let redacted = redact_excerpt(value);
    redacted
        .trim_matches(['[', ']'])
        .chars()
        .map(|character| {
            if character.is_whitespace() || matches!(character, '[' | ']') {
                '_'
            } else {
                character
            }
        })
        .collect()
}

fn bounded(value: &str) -> String {
    redact_excerpt(value)
        .chars()
        .take(LINT_AGENT_EXCERPT_CHAR_CAP)
        .collect()
}

fn redact_excerpt(value: &str) -> String {
    static SENSITIVE: OnceLock<Regex> = OnceLock::new();
    let pattern = SENSITIVE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:\b(?:api[_-]?key|password|passwd|secret|access[_-]?token|authorization)\s*[:=]\s*[^\s,;]+|\b(?:sk-|gh[pousr]_|github_pat_)[a-z0-9_-]{8,}|\bAKIA[A-Z0-9]{16}\b|https?://[^\s)\]}]+|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|(?:/Users/|/home/|[a-z]:\\)[^\s]+)",
        )
        .expect("static semantic redaction pattern")
    });
    pattern.replace_all(value, "[redacted]").into_owned()
}

fn same_scope(left: Option<&str>, right: Option<&str>) -> bool {
    left == right
}

fn build_entity_matchers(
    entities: &[Entity],
) -> Result<BTreeMap<Option<String>, EntityMatcher>, ()> {
    let mut grouped = BTreeMap::<Option<String>, Vec<usize>>::new();
    for (index, entity) in entities.iter().enumerate() {
        grouped.entry(entity.space.clone()).or_default().push(index);
    }
    grouped
        .into_iter()
        .map(|(space, entity_indexes)| {
            let entries = entity_indexes
                .into_iter()
                .filter_map(|index| {
                    let name = normalize(&entities[index].name);
                    if name.is_empty() {
                        return None;
                    }
                    let escaped = regex::escape(&name);
                    let pattern = if !name.is_ascii() {
                        escaped
                    } else {
                        format!(r"(?:^| ){escaped}(?: |$)")
                    };
                    Some((index, pattern))
                })
                .collect::<Vec<_>>();
            let (entity_indexes, patterns): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
            let patterns = RegexSet::new(patterns).map_err(|_| ())?;
            Ok((
                space,
                EntityMatcher {
                    entity_indexes,
                    patterns,
                },
            ))
        })
        .collect()
}

fn mentioned_entity_indexes(
    memory: &Memory,
    matchers: &BTreeMap<Option<String>, EntityMatcher>,
) -> Vec<usize> {
    let Some(matcher) = matchers.get(&memory.space) else {
        return Vec::new();
    };
    let content = normalize(&memory.content);
    matcher
        .patterns
        .matches(&content)
        .into_iter()
        .map(|pattern_index| matcher.entity_indexes[pattern_index])
        .collect()
}

fn build_memory_token_indexes(memories: &[Memory]) -> BTreeMap<Option<String>, MemoryTokenIndex> {
    let mut grouped = BTreeMap::<Option<String>, Vec<usize>>::new();
    for (index, memory) in memories.iter().enumerate() {
        grouped.entry(memory.space.clone()).or_default().push(index);
    }
    grouped
        .into_iter()
        .map(|(space, memory_indexes)| {
            let mut postings = BTreeMap::<String, Vec<usize>>::new();
            for memory_index in &memory_indexes {
                for token in &memories[*memory_index].content_tokens {
                    postings
                        .entry(token.clone())
                        .or_default()
                        .push(*memory_index);
                }
            }
            (
                space,
                MemoryTokenIndex {
                    memory_count: memory_indexes.len(),
                    postings,
                },
            )
        })
        .collect()
}

fn related_memory_indexes(
    page: &Page,
    indexes: &BTreeMap<Option<String>, MemoryTokenIndex>,
) -> Vec<(usize, u16)> {
    let Some(index) = indexes.get(&page.workspace) else {
        return Vec::new();
    };
    let max_posting = (index.memory_count / 20).max(64);
    let mut overlap = BTreeMap::<usize, u16>::new();
    for token in &page.content_tokens {
        let Some(postings) = index.postings.get(token) else {
            continue;
        };
        if postings.len() > max_posting {
            continue;
        }
        for memory_index in postings {
            *overlap.entry(*memory_index).or_insert(0) += 1;
        }
    }
    let mut ranked = overlap
        .into_iter()
        .filter(|(_, shared_tokens)| *shared_tokens >= 3)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.truncate(PAGE_MEMORY_TOP_K);
    ranked
}

fn contains_phrase(content: &str, phrase: &str) -> bool {
    let content = normalize(content);
    let phrase = normalize(phrase);
    if phrase.is_empty() {
        return false;
    }
    if !phrase.is_ascii() {
        return content.contains(&phrase);
    }
    format!(" {content} ").contains(&format!(" {phrase} "))
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn content_tokens(value: &str) -> BTreeSet<String> {
    normalize(value)
        .split_whitespace()
        .filter(|token| token.len() >= 4)
        .map(str::to_string)
        .collect()
}

fn token_overlap_count(left: &BTreeSet<String>, right: &BTreeSet<String>) -> usize {
    left.intersection(right).count()
}

fn source_population_digest(
    memory_population_digest: [u8; 32],
    entities: &[Entity],
    links: &BTreeSet<(String, String)>,
    relations: &BTreeSet<(String, String)>,
    pages: &[Page],
    page_evidence: &[PageEvidence],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"wenlan-lint-semantic-source-population-v2");
    digest.update(memory_population_digest);
    for entity in entities {
        digest.update(b"entity");
        digest_value(&mut digest, entity.id.as_bytes());
        digest_value(&mut digest, entity.name.as_bytes());
        digest_optional(&mut digest, entity.space.as_deref());
    }
    for (memory_id, entity_id) in links {
        digest.update(b"memory_entity");
        digest_value(&mut digest, memory_id.as_bytes());
        digest_value(&mut digest, entity_id.as_bytes());
    }
    for (from, to) in relations {
        digest.update(b"relation");
        digest_value(&mut digest, from.as_bytes());
        digest_value(&mut digest, to.as_bytes());
    }
    for page in pages {
        digest.update(b"page");
        digest_value(&mut digest, page.id.as_bytes());
        digest_value(&mut digest, page.content.as_bytes());
        digest_optional(&mut digest, page.workspace.as_deref());
    }
    for evidence in page_evidence {
        digest.update(b"page_evidence");
        digest_value(&mut digest, evidence.page_id.as_bytes());
        digest_value(&mut digest, evidence.source_kind.as_bytes());
        digest_value(&mut digest, evidence.locator.as_bytes());
    }
    digest.finalize().into()
}

fn digest_optional(digest: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            digest.update([1]);
            digest_value(digest, value.as_bytes());
        }
        None => digest.update([0]),
    }
}

fn digest_value(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

#[cfg(test)]
mod excerpt_tests {
    use super::contextual_excerpt;

    #[test]
    fn redacted_scope_keeps_context_marker_well_formed() {
        let excerpt = contextual_excerpt(Some("/Users/alice/private"), "visible evidence");

        assert_eq!(excerpt, "[lint_context scope=redacted]\nvisible evidence");
        assert!(!excerpt.contains("/Users/alice"));
    }
}
