mod aggregate;

use crate::lint::context::{LintContext, ScopeFilter};
pub(in crate::lint::kg) use aggregate::AggregateCounts;
use aggregate::{advisory_metrics, aggregate_counts, entity_partitions, substrate_counts};
use wenlan_types::lint::{LintMetric, LINT_MAX_EVIDENCE_PER_CHECK};

pub(super) struct RowCheck {
    pub(super) population: u64,
    pub(super) affected: u64,
    pub(super) evidence_positions: Vec<usize>,
}

pub(super) struct KgSnapshot {
    pub(super) entities: RowCheck,
    pub(super) observations: RowCheck,
    pub(super) relations: RowCheck,
    pub(super) links: RowCheck,
    pub(super) partitions: Vec<LintMetric>,
    pub(super) aggregates: AggregateCounts,
    pub(super) advisory: Vec<LintMetric>,
    pub(super) eligible_memories: u64,
    pub(super) linked_memories: u64,
}

pub(super) async fn load(context: &LintContext<'_, '_>, hub_cap: u64) -> Result<KgSnapshot, ()> {
    let entities = entity_integrity(context).await?;
    let observations = observation_integrity(context).await?;
    let relations = relation_integrity(context).await?;
    let links = link_integrity(context).await?;
    let partitions = entity_partitions(context).await?;
    let aggregates = aggregate_counts(context).await?;
    let advisory = advisory_metrics(context, hub_cap).await?;
    let (eligible_memories, linked_memories) = substrate_counts(context).await?;
    Ok(KgSnapshot {
        entities,
        observations,
        relations,
        links,
        partitions,
        aggregates,
        advisory,
        eligible_memories,
        linked_memories,
    })
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: the scope clause now uses `scope_clause`, and
// the integrity predicate excludes the `UNFILED_SPACE_ID` sentinel itself
// (never a registered `spaces.name`) so a folded unfiled entity no longer
// vacuously flags as an integrity violation. G6 Stage 3 retirement lint
// track: the prior disposition here ("audits the legacy entities store's
// own data quality") no longer holds -- every column this check validates
// (name, entity_type, confirmed, confidence, space) is mirrored verbatim
// onto the `kind='entity'` shadow page (item 5, writer flip), so validating
// the shadow page validates the live store, not a stale mirror. Ported to
// the canonical projected subquery (same pattern as `link_integrity` above
// and `semantic_candidates.rs::load_entities`), aliased so every downstream
// `e.name`/`e.entity_type`/`e.confirmed`/`e.confidence`/`e.space`/`e.id`
// reference is unchanged.
async fn entity_integrity(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (clause, params) = scope_clause(context.scope().filter(), "e", false);
    row_check(
        context,
        &format!(
            "SELECT CASE WHEN TRIM(e.name)='' OR TRIM(e.entity_type)=''
                    OR COALESCE(e.confirmed,-1) NOT IN (0,1)
                    OR (e.confidence IS NOT NULL AND (e.confidence < 0 OR e.confidence > 1))
                    OR (e.space IS NOT NULL AND e.space != '{unfiled}' AND NOT EXISTS(
                        SELECT 1 FROM spaces s WHERE s.name=e.space))
                  THEN 1 ELSE 0 END
               FROM (SELECT epm.entity_id AS id, p.title AS name, p.entity_type AS entity_type,
                            p.entity_confirmed AS confirmed, p.confidence AS confidence,
                            p.space AS space
                       FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id
                      WHERE p.kind='entity' AND p.status='active') e{clause} ORDER BY e.id",
            unfiled = crate::db::UNFILED_SPACE_ID
        ),
        params,
    )
    .await
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: `e.space`/`e.id` now uses `scope_clause`,
// sentinel-aware (entity-existence side only -- observation content itself
// is out of scope for this stage regardless, see the spec's 1.5b
// observations ruling). G6 Stage 3 retirement lint track: the prior
// disposition here ("audits the legacy entities store's own data quality")
// was only half right -- `TRIM(o.content)=''`/confirmed/confidence bounds
// are genuine observation-quality checks with nothing to do with `entities`,
// and `e.id IS NULL` is a real orphan-observation check (does this
// observation's owning entity still exist), which stays meaningful once
// the store retires. Only the join needed porting, same pattern as
// `link_integrity` above: a canonical-only entity's observations no longer
// read as orphaned just because it never got an `entities` row.
async fn observation_integrity(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (clause, params) = scope_clause(context.scope().filter(), "e", true);
    row_check(
        context,
        &format!(
            "SELECT CASE WHEN e.id IS NULL OR TRIM(o.content)=''
                    OR COALESCE(o.confirmed,-1) NOT IN (0,1)
                    OR (o.confidence IS NOT NULL AND (o.confidence < 0 OR o.confidence > 1))
                  THEN 1 ELSE 0 END
               FROM observations o
               LEFT JOIN (SELECT epm.entity_id AS id, p.space AS space
                            FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id
                           WHERE p.kind='entity' AND p.status='active') e
                 ON e.id=o.entity_id
               {clause} ORDER BY o.id"
        ),
        params,
    )
    .await
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: `f.space`/`f.id` (the src-entity alias) now uses
// `scope_clause`, sentinel-aware. G6 Stage 3 retirement lint track:
// the prior disposition here ("audits the legacy entities store's own data
// quality") was only half right -- `TRIM(r.semantic_type)=''` has nothing to
// do with `entities`, and `f.id`/`t.id IS NULL` are real orphan-relation
// checks (does this edge's src/dst entity still exist), which stay
// meaningful once the store retires. Both joins ported to the canonical
// projected subquery, same pattern as `link_integrity` above; two separate
// subquery instances since `f` and `t` each need their own derived table.
async fn relation_integrity(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (clause, params) = scope_clause(context.scope().filter(), "f", true);
    row_check(
        context,
        &format!(
            "SELECT CASE WHEN f.id IS NULL OR t.id IS NULL OR TRIM(r.semantic_type)=''
                  THEN 1 ELSE 0 END
               FROM (SELECT * FROM edges WHERE edge_type='relates' AND valid_until IS NULL) r
               LEFT JOIN (SELECT epm.entity_id AS id, p.space AS space
                            FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id
                           WHERE p.kind='entity' AND p.status='active') f
                 ON f.id=r.src_id
               LEFT JOIN (SELECT epm.entity_id AS id, p.space AS space
                            FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id
                           WHERE p.kind='entity' AND p.status='active') t
                 ON t.id=r.dst_id
               {clause} ORDER BY r.edge_id"
        ),
        params,
    )
    .await
}

// G6 Stage 1.5a: the entity-existence side (`e.id IS NULL`) moved onto the
// `kind='entity'` shadow page via `entity_page_map` -- safe because the
// scope clause here filters on the memories-derived alias `m`, never on
// `entities`/the entity side's space at all.
async fn link_integrity(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (clause, params) = scope_clause(context.scope().filter(), "m", true);
    row_check(
        context,
        &format!(
            "SELECT CASE WHEN m.source_id IS NULL OR e.entity_id IS NULL THEN 1 ELSE 0 END
               FROM memory_entities me
               LEFT JOIN (SELECT source_id, MAX(id) AS id, MAX(space) AS space FROM memories
                           GROUP BY source_id) m
                 ON m.source_id=me.memory_id
               LEFT JOIN (SELECT epm.entity_id FROM entity_page_map epm
                          JOIN pages p ON p.id = epm.page_id
                          WHERE p.kind = 'entity' AND p.status = 'active') e
                 ON e.entity_id=me.entity_id
               {clause} ORDER BY me.memory_id, me.entity_id"
        ),
        params,
    )
    .await
}

async fn row_check(
    context: &LintContext<'_, '_>,
    sql: &str,
    params: libsql::params::Params,
) -> Result<RowCheck, ()> {
    let mut rows = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let mut population = 0_u64;
    let mut affected = 0_u64;
    let mut evidence_positions = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        if row.get::<i64>(0).map_err(|_| ())? != 0 {
            affected = affected.saturating_add(1);
            if evidence_positions.len() < usize::from(LINT_MAX_EVIDENCE_PER_CHECK) {
                evidence_positions.push(usize::try_from(population).map_err(|_| ())?);
            }
        }
        population = population.saturating_add(1);
    }
    Ok(RowCheck {
        population,
        affected,
        evidence_positions,
    })
}

/// Scope predicate for a space column folded by the 1.5b space-sentinel
/// migration (`memories.space`, `entities.space`, `pages.workspace`): an
/// unfiled row stores `UNFILED_SPACE_ID`, not SQL NULL, so `Uncategorized`
/// must match either. Mirrors `push_read_scope_filter_folded` (db.rs).
pub(super) fn scope_clause(
    scope: &ScopeFilter,
    alias: &str,
    exclude_missing_owner: bool,
) -> (String, libsql::params::Params) {
    match scope {
        ScopeFilter::Global => (String::new(), libsql::params::Params::None),
        ScopeFilter::Registered(space) => (
            format!(" WHERE {alias}.space=?1"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            format!(
                " WHERE ({alias}.space IS NULL OR {alias}.space = '{}'){}",
                crate::db::UNFILED_SPACE_ID,
                if exclude_missing_owner {
                    format!(" AND {alias}.id IS NOT NULL")
                } else {
                    String::new()
                }
            ),
            libsql::params::Params::None,
        ),
    }
}
