use super::scope_clause;
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

pub(super) async fn entity_partitions(
    context: &LintContext<'_, '_>,
) -> Result<Vec<LintMetric>, ()> {
    let (clause, params) = scope_clause(context.scope().filter(), "e", false);
    let values = scalar_row(
        context,
        &format!(
            "SELECT COUNT(*), SUM(CASE WHEN e.confirmed=1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN e.space IS NOT NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN e.space IS NULL THEN 1 ELSE 0 END)
               FROM entities e{clause}"
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

pub(super) async fn aggregate_counts(context: &LintContext<'_, '_>) -> Result<AggregateCounts, ()> {
    let values = scalar_row(
        context,
        "SELECT (SELECT COUNT(*) FROM entities),
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
    let values = scalar_row(
        context,
        "WITH degree AS (
             SELECT e.id, LOWER(TRIM(e.name)) AS normalized_name,
                    LOWER(TRIM(e.entity_type)) AS entity_type, COUNT(me.memory_id) AS links
               FROM entities e LEFT JOIN memory_entities me ON me.entity_id=e.id GROUP BY e.id
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
