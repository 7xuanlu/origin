use super::super::provenance_checks::result::Assessment;
use crate::lint::context::{LintContext, ScopeFilter};
use std::collections::BTreeSet;
use wenlan_types::lint::{
    LintMetric, LintMetricCode, LintMetricValue, LINT_MAX_EVIDENCE_PER_CHECK,
};

pub(super) async fn load(context: &LintContext<'_, '_>) -> Result<Assessment, ()> {
    let (where_sql, params) = scoped_active_source_filter(context.scope().filter());
    let sql = format!(
        "SELECT pl.source_page_id, pl.label_key, COUNT(target.id) \
         FROM page_links pl \
         INNER JOIN pages p ON p.id = pl.source_page_id \
         LEFT JOIN pages target \
           ON LOWER(target.title) = LOWER(pl.label_key) \
          AND target.status = 'active' \
          AND target.space = p.space \
         WHERE pl.target_page_id IS NULL AND p.status = 'active'{where_sql} \
         GROUP BY pl.source_page_id, pl.label_key, p.space \
         ORDER BY pl.source_page_id, pl.label_key"
    );
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let mut population = 0_u64;
    let mut warning_count = 0_u64;
    let mut evidence_positions = Vec::new();
    let mut distinct_labels = BTreeSet::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let _: String = row.get(0).map_err(|_| ())?;
        let label_key: String = row.get(1).map_err(|_| ())?;
        let target_count: i64 = row.get(2).map_err(|_| ())?;
        let target_count = u64::try_from(target_count).map_err(|_| ())?;
        distinct_labels.insert(label_key);
        if target_count > 0 {
            warning_count = warning_count.saturating_add(1);
            if evidence_positions.len() < usize::from(LINT_MAX_EVIDENCE_PER_CHECK) {
                evidence_positions.push(usize::try_from(population).map_err(|_| ())?);
            }
        }
        population = population.saturating_add(1);
    }
    let distinct_label_count = u64::try_from(distinct_labels.len()).map_err(|_| ())?;
    let mut assessment =
        Assessment::from_aggregate(population, warning_count, 0, true, evidence_positions);
    assessment.set_metrics(vec![LintMetric::new(
        LintMetricCode::PageOrphanLabels,
        LintMetricValue::Count {
            value: distinct_label_count,
        },
    )]);
    Ok(assessment)
}

fn scoped_active_source_filter(filter: &ScopeFilter) -> (&'static str, libsql::params::Params) {
    match filter {
        ScopeFilter::Global => ("", libsql::params::Params::None),
        ScopeFilter::Registered(workspace) => (
            " AND p.space = ?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(workspace.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            " AND p.space = '00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    }
}

#[cfg(test)]
mod architecture_tests {
    #[test]
    fn lint_orphan_module_has_no_mutating_resolver_dependency() {
        let source = include_str!("orphans.rs");
        let forbidden = concat!("resolve_orphan_page_", "links");
        assert!(!source.contains(forbidden));
    }
}
