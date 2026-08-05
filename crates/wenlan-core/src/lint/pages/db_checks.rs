use super::provenance_checks::result::{failed_result, Assessment, Level};
use crate::lint::context::{LintContext, PopulationBasis, ScopeFilter};
use std::collections::BTreeMap;
use wenlan_types::lint::{LintCheckResult, LintMetric, LintMetricCode, LintMetricValue};

pub(crate) const PARTITIONS_ID: &str = "pages.db.partitions";
pub(crate) const DUPLICATE_TITLES_ID: &str = "pages.duplicate_active_titles";
pub(crate) const ARCHIVE_ID: &str = "pages.archive_inventory";
pub(crate) const REVIEW_ID: &str = "pages.review_status_inventory";
pub(crate) const SOURCE_INTEGRITY_ID: &str = "pages.source_page_integrity";

const STATUSES: [&str; 2] = ["active", "archived"];
const CREATION_KINDS: [&str; 5] = ["authored", "distilled", "imported", "research", "source"];
const REVIEW_STATUSES: [&str; 2] = ["confirmed", "unconfirmed"];

#[derive(Debug, Clone)]
struct PageRow {
    title_key: String,
    effective_scope: Option<String>,
    status: String,
    creation_kind: String,
    review_status: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Partitions {
    statuses: BTreeMap<String, u64>,
    creation_kinds: BTreeMap<String, u64>,
    review_statuses: BTreeMap<String, u64>,
}

impl Partitions {
    fn from_rows(rows: &[PageRow]) -> Self {
        let mut partitions = Self::default();
        for row in rows {
            increment(&mut partitions.statuses, &row.status);
            increment(&mut partitions.creation_kinds, &row.creation_kind);
            increment(&mut partitions.review_statuses, &row.review_status);
        }
        partitions
    }

    fn is_exact(&self, total: u64) -> bool {
        [
            self.statuses.values().copied().sum::<u64>(),
            self.creation_kinds.values().copied().sum::<u64>(),
            self.review_statuses.values().copied().sum::<u64>(),
        ]
        .into_iter()
        .all(|sum| sum == total)
    }
}

pub(crate) async fn run(context: &LintContext<'_, '_>) -> Vec<LintCheckResult> {
    let rows = match load_rows(context).await {
        Ok(rows) => rows,
        Err(()) => return failed_results(context),
    };
    let source_integrity = match load_and_assess_source_integrity(context).await {
        Ok(assessment) => assessment,
        Err(()) => return failed_results(context),
    };
    let assessments = assess_all(&rows, source_integrity);
    let basis = if context.scope().filter().is_selected() {
        PopulationBasis::SelectedScope
    } else {
        PopulationBasis::Global
    };
    assessments
        .into_iter()
        .map(|(check_id, assessment)| {
            let population = assessment.population();
            if context
                .record_population(check_id, basis, population)
                .is_err()
            {
                return failed_result(check_id, context.clock().duration_ms());
            }
            assessment
                .result(check_id, context.clock().duration_ms())
                .unwrap_or_else(|_| failed_result(check_id, context.clock().duration_ms()))
        })
        .collect()
}

fn assess_all(rows: &[PageRow], source_integrity: Assessment) -> [(&'static str, Assessment); 5] {
    [
        (PARTITIONS_ID, assess_partitions(rows)),
        (DUPLICATE_TITLES_ID, assess_duplicates(rows)),
        (ARCHIVE_ID, assess_archive(rows)),
        (REVIEW_ID, assess_review(rows)),
        (SOURCE_INTEGRITY_ID, source_integrity),
    ]
}

async fn load_and_assess_source_integrity(context: &LintContext<'_, '_>) -> Result<Assessment, ()> {
    let (scope_clause, params) = match context.scope().filter() {
        ScopeFilter::Global => ("", libsql::params::Params::None),
        ScopeFilter::Registered(workspace) => (
            " AND p.space=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(workspace.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            " AND p.space='00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    };
    // G6 Stage 1.3: page provenance reads `edges` (`cites`).
    // S2 (2026-08-05 review): the evidence EXISTS reads `page_evidence` OR
    // `edges` -- a NULL-locator `authored` row IS provenance (it has no
    // edge twin; `insert_resolved_page_evidence` only dual-writes when a
    // real locator resolves), and this check is well-formedness over
    // provenance rows, not a reader of the canonical `edges` store alone.
    // Closure review (2026-08-05): the former page_sources-analog EXISTS
    // (edges restricted to dst_kind='memory') is a strict subset of this
    // evidence arm's edges half (no dst_kind filter) -- any edge satisfying
    // it also satisfies the OR here, so it never changed `valid`. Deleted.
    let sql = format!(
        "SELECT p.source_memory_ids,
                EXISTS(SELECT 1 FROM page_evidence pe WHERE pe.page_id=p.id)
                OR EXISTS(SELECT 1 FROM edges pe WHERE pe.edge_type='cites' AND pe.valid_until IS NULL
                       AND pe.src_id=p.id)
           FROM pages p
          WHERE p.status='active' AND p.creation_kind='source'{scope_clause}
          ORDER BY p.id"
    );
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let mut assessment = Assessment::default();
    let mut affected = 0_u64;
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let source_ids = row.get::<String>(0).map_err(|_| ())?;
        let parsed_nonempty =
            serde_json::from_str::<Vec<String>>(&source_ids).is_ok_and(|ids| !ids.is_empty());
        let has_page_evidence = row.get::<i64>(1).map_err(|_| ())? != 0;
        let valid = parsed_nonempty || has_page_evidence;
        assessment.push(if valid { Level::Pass } else { Level::Error });
        affected = affected.saturating_add(u64::from(!valid));
    }
    assessment.set_metrics(metrics(assessment.population(), affected));
    Ok(assessment)
}

pub(crate) async fn run_source_integrity(context: &LintContext<'_, '_>) -> LintCheckResult {
    let basis = if context.scope().filter().is_selected() {
        PopulationBasis::SelectedScope
    } else {
        PopulationBasis::Global
    };
    let Ok(assessment) = load_and_assess_source_integrity(context).await else {
        let _ = context.record_population(SOURCE_INTEGRITY_ID, basis, 0);
        return failed_result(SOURCE_INTEGRITY_ID, context.clock().duration_ms());
    };
    let population = assessment.population();
    if context
        .record_population(SOURCE_INTEGRITY_ID, basis, population)
        .is_err()
    {
        return failed_result(SOURCE_INTEGRITY_ID, context.clock().duration_ms());
    }
    assessment
        .result(SOURCE_INTEGRITY_ID, context.clock().duration_ms())
        .unwrap_or_else(|_| failed_result(SOURCE_INTEGRITY_ID, context.clock().duration_ms()))
}

fn assess_partitions(rows: &[PageRow]) -> Assessment {
    let partitions = Partitions::from_rows(rows);
    let total = u64::try_from(rows.len()).unwrap_or(u64::MAX);
    let exact = partitions.is_exact(total);
    let mut assessment = Assessment::default();
    let mut affected = 0_u64;
    for row in rows {
        let valid = exact
            && STATUSES.contains(&row.status.as_str())
            && CREATION_KINDS.contains(&row.creation_kind.as_str())
            && REVIEW_STATUSES.contains(&row.review_status.as_str());
        assessment.push(if valid { Level::Pass } else { Level::Error });
        affected = affected.saturating_add(u64::from(!valid));
    }
    assessment.set_metrics(metrics(
        u64::try_from(rows.len()).unwrap_or(u64::MAX),
        affected,
    ));
    assessment
}

fn assess_duplicates(rows: &[PageRow]) -> Assessment {
    let mut counts = BTreeMap::<(Option<&str>, &str), u64>::new();
    for row in rows.iter().filter(|row| row.status == "active") {
        *counts
            .entry((row.effective_scope.as_deref(), &row.title_key))
            .or_default() += 1;
    }
    let mut assessment = Assessment::default();
    let mut affected = 0_u64;
    for row in rows.iter().filter(|row| row.status == "active") {
        let duplicate = counts
            .get(&(row.effective_scope.as_deref(), row.title_key.as_str()))
            .copied()
            .unwrap_or(0)
            > 1;
        assessment.push(if duplicate {
            Level::Warning
        } else {
            Level::Pass
        });
        affected = affected.saturating_add(u64::from(duplicate));
    }
    assessment.set_metrics(metrics(counts.values().copied().sum(), affected));
    assessment
}

fn assess_archive(rows: &[PageRow]) -> Assessment {
    let mut assessment = Assessment::default();
    assessment.mark_inventory();
    for _ in rows.iter().filter(|row| row.status == "archived") {
        assessment.push(Level::Pass);
    }
    assessment.set_metrics(metrics(assessment.population(), 0));
    assessment
}

fn assess_review(rows: &[PageRow]) -> Assessment {
    let mut assessment = Assessment::default();
    assessment.mark_inventory();
    let mut affected = 0_u64;
    for row in rows {
        let valid = CREATION_KINDS.contains(&row.creation_kind.as_str())
            && REVIEW_STATUSES.contains(&row.review_status.as_str());
        assessment.push(if valid { Level::Pass } else { Level::Error });
        affected = affected.saturating_add(u64::from(!valid));
    }
    assessment.set_metrics(metrics(
        u64::try_from(rows.len()).unwrap_or(u64::MAX),
        affected,
    ));
    assessment
}

async fn load_rows(context: &LintContext<'_, '_>) -> Result<Vec<PageRow>, ()> {
    let (sql, params) = match context.scope().filter() {
        ScopeFilter::Global => (
            "SELECT LOWER(title), space, status, creation_kind, review_status FROM pages ORDER BY id",
            libsql::params::Params::None,
        ),
        ScopeFilter::Registered(workspace) => (
            "SELECT LOWER(title), space, status, creation_kind, review_status FROM pages WHERE space = ?1 ORDER BY id",
            libsql::params::Params::Positional(vec![libsql::Value::Text(workspace.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            "SELECT LOWER(title), space, status, creation_kind, review_status FROM pages WHERE space = '00000000-0000-4000-8000-000000000001' ORDER BY id",
            libsql::params::Params::None,
        ),
    };
    let mut query = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let mut rows = Vec::new();
    while let Some(row) = query.next().await.map_err(|_| ())? {
        rows.push(PageRow {
            title_key: row.get(0).map_err(|_| ())?,
            effective_scope: row.get(1).map_err(|_| ())?,
            status: row.get(2).map_err(|_| ())?,
            creation_kind: row.get(3).map_err(|_| ())?,
            review_status: row.get(4).map_err(|_| ())?,
        });
    }
    Ok(rows)
}

fn failed_results(context: &LintContext<'_, '_>) -> Vec<LintCheckResult> {
    [
        PARTITIONS_ID,
        DUPLICATE_TITLES_ID,
        ARCHIVE_ID,
        REVIEW_ID,
        SOURCE_INTEGRITY_ID,
    ]
    .into_iter()
    .map(|check_id| {
        let basis = if context.scope().filter().is_selected() {
            PopulationBasis::SelectedScope
        } else {
            PopulationBasis::Global
        };
        let _ = context.record_population(check_id, basis, 0);
        failed_result(check_id, context.clock().duration_ms())
    })
    .collect()
}

fn metrics(eligible: u64, affected: u64) -> Vec<LintMetric> {
    vec![
        LintMetric::new(
            LintMetricCode::EligibleRecords,
            LintMetricValue::Count { value: eligible },
        ),
        LintMetric::new(
            LintMetricCode::AffectedRecords,
            LintMetricValue::Count { value: affected },
        ),
    ]
}

fn increment(counts: &mut BTreeMap<String, u64>, key: &str) {
    *counts.entry(key.to_string()).or_default() += 1;
}

#[cfg(test)]
#[path = "db_checks_test.rs"]
mod tests;
