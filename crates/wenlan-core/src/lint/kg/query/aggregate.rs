use super::{scope_clause, scope_clause_folded};
use crate::lint::context::LintContext;
use wenlan_types::lint::{LintMetric, LintMetricCode, LintMetricValue};

pub(in crate::lint::kg) struct AggregateCounts {
    pub(in crate::lint::kg) entities: u64,
    relations: u64,
    observations: u64,
    links: u64,
}

impl AggregateCounts {
    pub(in crate::lint::kg) fn sum(&self) -> u64 {
        self.entities
            .saturating_add(self.relations)
            .saturating_add(self.observations)
            .saturating_add(self.links)
    }

    pub(in crate::lint::kg) fn metrics(&self) -> Vec<LintMetric> {
        vec![
            metric(LintMetricCode::KgEntities, self.entities),
            metric(LintMetricCode::KgRelations, self.relations),
            metric(LintMetricCode::KgObservations, self.observations),
            metric(LintMetricCode::KgMemoryEntityLinks, self.links),
        ]
    }
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: `KgEntitiesScoped` vs `KgEntitiesUncategorized` is
// still the "has a real space" vs "unfiled" split -- the SUM predicates below
// now test against the `UNFILED_SPACE_ID` sentinel too, so a folded row keeps
// classifying as Uncategorized instead of silently reporting 0. G6 Stage 3
// retirement lint track: this resolves the "known transitional skew" the
// prior comment flagged -- `KgEntities` here was legacy-derived
// (`COUNT(*) FROM entities`) while `aggregate_counts` below already emits the
// same metric code shadow-derived (`entity_page_map` JOIN `pages`); porting
// this query onto the identical canonical projection makes the two counts
// the same query in different shapes, not two independently-drifting
// sources of truth.
pub(super) async fn entity_partitions(
    context: &LintContext<'_, '_>,
) -> Result<Vec<LintMetric>, ()> {
    let (clause, params) = scope_clause_folded(context.scope().filter(), "e", false);
    let values = scalar_row(
        context,
        &format!(
            "SELECT COUNT(*), SUM(CASE WHEN e.confirmed=1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN e.space IS NOT NULL AND e.space != '{unfiled}' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN e.space IS NULL OR e.space = '{unfiled}' THEN 1 ELSE 0 END)
               FROM (SELECT epm.entity_id AS id, p.entity_confirmed AS confirmed,
                            p.space AS space
                       FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id
                      WHERE p.kind='entity' AND p.status='active') e{clause}",
            unfiled = crate::db::UNFILED_SPACE_ID
        ),
        params,
        4,
    )
    .await?;
    Ok(vec![
        metric(LintMetricCode::KgEntities, values[0]),
        metric(LintMetricCode::KgEntitiesConfirmed, values[1]),
        metric(LintMetricCode::KgEntitiesScoped, values[2]),
        metric(LintMetricCode::KgEntitiesUncategorized, values[3]),
    ])
}

// G6 Stage 1.5a: the `entities` count moved onto `entity_page_map` (1:1 with
// `entities` by the shadow-page invariant); no space touch here.
pub(super) async fn aggregate_counts(context: &LintContext<'_, '_>) -> Result<AggregateCounts, ()> {
    let values = scalar_row(
        context,
        "SELECT (SELECT COUNT(*) FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id
                  WHERE p.kind = 'entity' AND p.status = 'active'),
                (SELECT COUNT(*) FROM edges WHERE edge_type='relates' AND valid_until IS NULL),
                (SELECT COUNT(*) FROM observations), (SELECT COUNT(*) FROM memory_entities)",
        libsql::params::Params::None,
        4,
    )
    .await?;
    Ok(AggregateCounts {
        entities: values[0],
        relations: values[1],
        observations: values[2],
        links: values[3],
    })
}

pub(super) async fn advisory_metrics(
    context: &LintContext<'_, '_>,
    hub_cap: u64,
) -> Result<Vec<LintMetric>, ()> {
    let cap = i64::try_from(hub_cap).map_err(|_| ())?;
    // G6 Stage 1.5a: `e.id`/`e.name`/`e.entity_type` moved onto the
    // `kind='entity'` shadow page via `entity_page_map`; no space touch.
    let values = scalar_row(
        context,
        "WITH degree AS (
             SELECT e.entity_id AS id, LOWER(TRIM(p.title)) AS normalized_name,
                    LOWER(TRIM(p.entity_type)) AS entity_type, COUNT(me.memory_id) AS links
               FROM entity_page_map e
               JOIN pages p ON p.id = e.page_id AND p.kind = 'entity' AND p.status = 'active'
               LEFT JOIN memory_entities me ON me.entity_id=e.entity_id GROUP BY e.entity_id
         ), duplicate AS (
             SELECT COALESCE(SUM(amount-1),0) AS extras FROM (
                 SELECT COUNT(*) AS amount FROM degree GROUP BY normalized_name HAVING amount>1
             )
         )
         SELECT (SELECT extras FROM duplicate),
                SUM(CASE WHEN links>?1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN links>?1 AND entity_type IN ('person','speaker','people','user')
                         THEN 1 ELSE 0 END)
           FROM degree",
        libsql::params::Params::Positional(vec![libsql::Value::Integer(cap)]),
        3,
    )
    .await?;
    Ok(vec![
        metric(LintMetricCode::KgDuplicateEntityNames, values[0]),
        metric(LintMetricCode::KgHubEntities, values[1]),
        metric(LintMetricCode::KgSemanticSuspicions, values[2]),
    ])
}

pub(super) async fn substrate_counts(context: &LintContext<'_, '_>) -> Result<(u64, u64), ()> {
    let (clause, params) = scope_clause(context.scope().filter(), "m", false);
    let values = scalar_row(
        context,
        &format!(
            "SELECT COUNT(DISTINCT m.source_id),
                    COUNT(DISTINCT CASE WHEN me.memory_id IS NOT NULL THEN m.source_id END)
               FROM (SELECT DISTINCT source_id, space FROM memories
                      WHERE source='memory' AND chunk_index=0 AND TRIM(content)!='') m
               LEFT JOIN memory_entities me ON me.memory_id=m.source_id{clause}"
        ),
        params,
        2,
    )
    .await?;
    Ok((values[0], values[1]))
}

async fn scalar_row(
    context: &LintContext<'_, '_>,
    sql: &str,
    params: libsql::params::Params,
    columns: usize,
) -> Result<Vec<u64>, ()> {
    let mut rows = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let row = rows.next().await.map_err(|_| ())?.ok_or(())?;
    (0..columns)
        .map(|index| {
            let index = i32::try_from(index).map_err(|_| ())?;
            let value = row.get::<Option<i64>>(index).map_err(|_| ())?.unwrap_or(0);
            u64::try_from(value).map_err(|_| ())
        })
        .collect()
}

fn metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}
