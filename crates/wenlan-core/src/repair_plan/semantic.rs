// SPDX-License-Identifier: Apache-2.0
use crate::{error::WenlanError, lint::snapshot::LintReadSnapshot};
use std::collections::BTreeMap;
use wenlan_types::{
    lint::{LintEvidenceRef, LintReport, LintSemanticAction, LintSemanticFinding},
    repair_plan::{RepairAffectedRecord, RepairAffectedRecordKind},
};

#[derive(Debug)]
pub(crate) struct SemanticReviewCandidate {
    pub(crate) check_id: String,
    pub(crate) finding: LintSemanticFinding,
    pub(crate) affected_records: Vec<RepairAffectedRecord>,
}

#[derive(Debug)]
pub(crate) struct SemanticBlockedCandidate {
    pub(crate) check_id: String,
    pub(crate) finding: LintSemanticFinding,
    pub(crate) affected_records: Vec<RepairAffectedRecord>,
    pub(crate) detail: String,
}

#[derive(Debug)]
pub(crate) enum SemanticResolution {
    Review(SemanticReviewCandidate),
    Blocked(SemanticBlockedCandidate),
}

pub(crate) async fn resolve_current(
    snapshot: &LintReadSnapshot<'_>,
    report: &LintReport,
) -> Result<Vec<SemanticResolution>, WenlanError> {
    let inventory = load_record_inventory(snapshot).await?;
    let mut resolutions = Vec::new();
    for check in report.checks() {
        for evidence in check.evidence() {
            let LintEvidenceRef::SemanticFinding { finding } = evidence else {
                continue;
            };
            let mut affected_records = Vec::new();
            let mut missing = Vec::new();
            for digest in finding
                .evidence_ids()
                .iter()
                .chain(finding.counterevidence_ids())
            {
                match inventory.get(digest.as_str()) {
                    Some(record) => affected_records.push(record.clone()),
                    None => missing.push(digest.as_str().to_string()),
                }
            }
            affected_records.sort();
            affected_records.dedup();
            if affected_records.is_empty() || !missing.is_empty() {
                resolutions.push(SemanticResolution::Blocked(SemanticBlockedCandidate {
                    check_id: check.check_id().to_string(),
                    finding: finding.clone(),
                    affected_records,
                    detail: if missing.is_empty() {
                        "semantic finding has no durable owner".to_string()
                    } else {
                        format!(
                            "semantic evidence no longer resolves to canonical records: {}",
                            missing.join(",")
                        )
                    },
                }));
                continue;
            }
            resolutions.push(SemanticResolution::Review(SemanticReviewCandidate {
                check_id: check.check_id().to_string(),
                finding: finding.clone(),
                affected_records,
            }));
        }
    }
    Ok(resolutions)
}

pub(crate) fn review_choices(action: LintSemanticAction) -> Vec<String> {
    use LintSemanticAction::*;
    match action {
        ReclassifyMemory => vec![
            "choose the canonical memory type".to_string(),
            "keep the current classification".to_string(),
        ],
        ReviewContradiction => vec![
            "mark the memories as compatible".to_string(),
            "identify the superseded memory".to_string(),
            "defer until more evidence exists".to_string(),
        ],
        ReviewStaleness => vec![
            "confirm the memory is still current".to_string(),
            "identify a replacement".to_string(),
            "defer until more evidence exists".to_string(),
        ],
        SupersedeMemory => vec![
            "select the predecessor and replacement direction".to_string(),
            "keep both memories active".to_string(),
        ],
        AddMemoryEntityLink => vec![
            "add the proposed memory-entity link".to_string(),
            "leave the records unlinked".to_string(),
        ],
        RemoveMemoryEntityLink => vec![
            "remove the proposed memory-entity link".to_string(),
            "keep the existing link".to_string(),
        ],
        AddEntityRelation => vec![
            "add the proposed entity relation".to_string(),
            "leave the entities unrelated".to_string(),
        ],
        RemoveEntityRelation => vec![
            "remove the proposed entity relation".to_string(),
            "keep the existing relation".to_string(),
        ],
        ReviewPageClaim => vec![
            "confirm the Page claim".to_string(),
            "revise or remove the Page claim".to_string(),
            "research supporting sources".to_string(),
        ],
        AddPageEvidence => vec![
            "add the proposed Page evidence link".to_string(),
            "leave the claim unsupported".to_string(),
            "research a stronger source".to_string(),
        ],
        RemovePageEvidence => vec![
            "remove the proposed Page evidence link".to_string(),
            "keep the existing evidence link".to_string(),
            "research the source relationship".to_string(),
        ],
        ReviewRetrieval => vec![
            "accept the retrieval result".to_string(),
            "record a retrieval miss".to_string(),
            "research the expected answer source".to_string(),
        ],
    }
}

pub(crate) fn suggested_research_queries(
    action: LintSemanticAction,
    affected_records: &[RepairAffectedRecord],
) -> Vec<String> {
    if !matches!(
        action,
        LintSemanticAction::ReviewContradiction
            | LintSemanticAction::ReviewStaleness
            | LintSemanticAction::ReviewPageClaim
            | LintSemanticAction::AddPageEvidence
            | LintSemanticAction::RemovePageEvidence
            | LintSemanticAction::ReviewRetrieval
    ) {
        return Vec::new();
    }
    let ids = affected_records
        .iter()
        .map(RepairAffectedRecord::durable_id)
        .collect::<Vec<_>>()
        .join(" ");
    vec![format!("find authoritative context for {ids}")]
}

async fn load_record_inventory(
    snapshot: &LintReadSnapshot<'_>,
) -> Result<BTreeMap<String, RepairAffectedRecord>, WenlanError> {
    let mut inventory = BTreeMap::new();
    load_simple_records(
        snapshot,
        "SELECT source_id FROM memories WHERE source='memory' AND chunk_index=0
          AND pending_revision=0 AND COALESCE(is_recap,0)=0 AND supersede_mode!='evicted'
          ORDER BY source_id,id",
        "memory",
        RepairAffectedRecordKind::Memory,
        &mut inventory,
    )
    .await?;
    // G6 Stage 1.5a: id-only enumeration moved onto `entity_page_map` (1:1
    // with `entities` by the shadow-page invariant); no space touch, order
    // is by the entity id itself (not a name), so it is unaffected by the
    // mirror-invariant name equivalence.
    load_simple_records(
        snapshot,
        "SELECT epm.entity_id AS id FROM entity_page_map epm
         JOIN pages p ON p.id = epm.page_id
         WHERE p.kind = 'entity' AND p.status = 'active' ORDER BY epm.entity_id",
        "entity",
        RepairAffectedRecordKind::Entity,
        &mut inventory,
    )
    .await?;
    load_simple_records(
        snapshot,
        "SELECT id FROM pages WHERE status='active' ORDER BY id",
        "page",
        RepairAffectedRecordKind::Page,
        &mut inventory,
    )
    .await?;

    // G6 Stage 1.2 reader #12: read `edges`, matching #13's
    // (`lint::semantic_candidates::load_relations`) query shape -- the two
    // readers mint the same `relation-entity:{id}:{type}:{endpoint}` keys, so
    // reading different stores here would leave an edge-only relation with no
    // inventory record, permanently Blocking its repair.
    let mut rows = snapshot
        .query(
            "SELECT src_id,dst_id,semantic_type
             FROM edges WHERE edge_type='relates' AND valid_until IS NULL
                   AND semantic_type IS NOT NULL
             ORDER BY src_id,dst_id,edge_id",
            libsql::params::Params::None,
        )
        .await
        .map_err(snapshot_error)?;
    while let Some(row) = rows.next().await.map_err(snapshot_error)? {
        let from_entity = row.get::<String>(0).map_err(database_error)?;
        let to_entity = row.get::<String>(1).map_err(database_error)?;
        let relation_type = row.get::<String>(2).map_err(database_error)?;
        insert_record(
            &mut inventory,
            format!("relation-entity:{from_entity}:{relation_type}:from"),
            RepairAffectedRecordKind::Entity,
            from_entity,
        )?;
        insert_record(
            &mut inventory,
            format!("relation-entity:{to_entity}:{relation_type}:to"),
            RepairAffectedRecordKind::Entity,
            to_entity,
        )?;
    }
    Ok(inventory)
}

async fn load_simple_records(
    snapshot: &LintReadSnapshot<'_>,
    sql: &str,
    key_kind: &str,
    record_kind: RepairAffectedRecordKind,
    inventory: &mut BTreeMap<String, RepairAffectedRecord>,
) -> Result<(), WenlanError> {
    let mut rows = snapshot
        .query(sql, libsql::params::Params::None)
        .await
        .map_err(snapshot_error)?;
    while let Some(row) = rows.next().await.map_err(snapshot_error)? {
        let durable_id = row.get::<String>(0).map_err(database_error)?;
        insert_record(
            inventory,
            format!("{key_kind}:{durable_id}"),
            record_kind,
            durable_id,
        )?;
    }
    Ok(())
}

fn insert_record(
    inventory: &mut BTreeMap<String, RepairAffectedRecord>,
    semantic_key: String,
    kind: RepairAffectedRecordKind,
    durable_id: String,
) -> Result<(), WenlanError> {
    let digest = crate::lint::semantic_record_key_digest(&semantic_key);
    let record = RepairAffectedRecord::try_new(kind, durable_id)
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    match inventory.entry(digest.as_str().to_string()) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(record);
        }
        std::collections::btree_map::Entry::Occupied(entry) if entry.get() != &record => {
            return Err(WenlanError::Conflict(
                "semantic record digest collision".to_string(),
            ));
        }
        std::collections::btree_map::Entry::Occupied(_) => {}
    }
    Ok(())
}

fn database_error(error: libsql::Error) -> WenlanError {
    WenlanError::VectorDb(format!("repair semantic resolver: {error}"))
}

fn snapshot_error(error: crate::lint::snapshot::SnapshotError) -> WenlanError {
    WenlanError::VectorDb(format!("repair semantic resolver: {error}"))
}
