use super::result::Assessment;
#[cfg(test)]
use super::result::Level;
use crate::lint::context::{LintContext, ScopeFilter};
#[cfg(test)]
use std::collections::BTreeMap;
use wenlan_types::lint::{LintMetric, LintMetricCode, LintMetricValue};

#[cfg(test)]
const VALID_KINDS: [&str; 4] = ["authored", "external_file", "external_url", "memory"];

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct SourceRecord {
    #[cfg(test)]
    pub(super) locator: String,
    pub(super) owner_present: bool,
    pub(super) expected_kind: String,
    pub(super) evidence_kinds: Vec<String>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct ExtraEvidence {
    pub(super) source_kind: String,
    pub(super) locator_present: bool,
}

#[cfg(test)]
#[derive(Debug)]
struct SourceBuilder {
    owner_present: bool,
    expected_kind: String,
    evidence_kinds: Vec<String>,
}

// G6 Stage 1.3 carryover (2026-08-05): this module validates `page_sources`
// and `page_evidence` AGAINST EACH OTHER (cross-store consistency), which is
// definitionally inexpressible over a single canonical `edges` store -- there
// is no second legacy copy left to disagree with once migrated. Stays on the
// legacy tables through Stage 1 and Stage 2; retires with the stores at
// Stage 3, same as `load_sources`/`source_ctes` below.
pub(super) async fn load_and_assess_sources(
    context: &LintContext<'_, '_>,
) -> Result<Assessment, ()> {
    let (scope_sql, params) = scoped_pages(context.scope().filter());
    let sql = format!(
        "{},
         evidence_stats AS MATERIALIZED (
           SELECT pe.page_id, pe.locator, COUNT(*) AS evidence_count,
                  SUM(CASE WHEN pe.source_kind = 'memory' THEN 1 ELSE 0 END) AS memory_count,
                  SUM(CASE WHEN pe.source_kind = 'external_url' THEN 1 ELSE 0 END) AS external_url_count,
                  SUM(CASE WHEN pe.source_kind = 'external_file' THEN 1 ELSE 0 END) AS external_file_count,
                  SUM(CASE WHEN pe.source_kind = 'authored' THEN 1 ELSE 0 END) AS authored_count,
                  SUM(CASE WHEN pe.source_kind NOT IN ('memory','external_url','external_file','authored') THEN 1 ELSE 0 END) AS invalid_count
             FROM page_evidence pe
             JOIN scoped_pages sp ON sp.id = pe.page_id
            WHERE pe.locator IS NOT NULL
            GROUP BY pe.page_id, pe.locator
         ),
         classified AS MATERIALIZED (
           SELECT r.page_id, r.memory_source_id,
                  CASE
                    WHEN r.owner_present = 0 THEN 2
                    WHEN es.evidence_count IS NULL THEN 2
                    WHEN es.invalid_count > 0 THEN 2
                    WHEN es.evidence_count = 1 AND CASE r.expected_kind
                      WHEN 'memory' THEN es.memory_count
                      WHEN 'external_url' THEN es.external_url_count
                      WHEN 'external_file' THEN es.external_file_count
                      WHEN 'authored' THEN es.authored_count
                      ELSE 0 END = 1 THEN 0
                    ELSE 1
                  END AS level
             FROM resolved r
             LEFT JOIN evidence_stats es
               ON es.page_id = r.page_id AND es.locator = r.memory_source_id
         ),
         extras AS (
           SELECT COUNT(*) AS extra_count,
                  COALESCE(SUM(CASE WHEN pe.source_kind NOT IN ('memory','external_url','external_file','authored') THEN 1 ELSE 0 END), 0) AS invalid_count
             FROM page_evidence pe
             JOIN scoped_pages sp ON sp.id = pe.page_id
             LEFT JOIN page_sources ps
               ON ps.page_id = pe.page_id AND ps.memory_source_id = pe.locator
            WHERE ps.page_id IS NULL
         ),
         ordered AS MATERIALIZED (
           SELECT ROW_NUMBER() OVER (ORDER BY page_id, memory_source_id) AS ordinal,
                  level
             FROM classified
         ),
         summary AS (
           SELECT COUNT(*) + COALESCE((SELECT invalid_count FROM extras), 0) AS population,
                  COALESCE(SUM(CASE WHEN ordered.level = 1 THEN 1 ELSE 0 END), 0) AS warning_count,
                  COALESCE(SUM(CASE WHEN ordered.level = 2 THEN 1 ELSE 0 END), 0)
                    + COALESCE((SELECT invalid_count FROM extras), 0) AS error_count,
                  COALESCE((SELECT extra_count FROM extras), 0) AS extra_count
             FROM ordered
         )
         SELECT 0 AS row_kind, population, warning_count, error_count, extra_count
           FROM summary
         UNION ALL
         SELECT 1 AS row_kind, ordinal, level, 0, 0
           FROM ordered
          WHERE level != 0
          ORDER BY row_kind, 2
          LIMIT 101",
        source_ctes(scope_sql)
    );
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let summary = rows.next().await.map_err(|_| ())?.ok_or(())?;
    let population = to_u64(summary.get::<i64>(1).map_err(|_| ())?);
    let warnings = to_u64(summary.get::<i64>(2).map_err(|_| ())?);
    let errors = to_u64(summary.get::<i64>(3).map_err(|_| ())?);
    let extras = to_u64(summary.get::<i64>(4).map_err(|_| ())?);
    let mut evidence_positions = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let ordinal = to_u64(row.get::<i64>(1).map_err(|_| ())?);
        let position = ordinal
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok());
        if let Some(position) = position {
            evidence_positions.push(position);
        }
    }
    let mut assessment =
        Assessment::from_aggregate(population, warnings, errors, extras > 0, evidence_positions);
    assessment.set_metrics(vec![
        count_metric(LintMetricCode::EligibleRecords, population),
        count_metric(LintMetricCode::ObservedRecords, extras),
        count_metric(
            LintMetricCode::AffectedRecords,
            warnings.saturating_add(errors),
        ),
    ]);
    Ok(assessment)
}

#[cfg(test)]
pub(super) fn assess_sources(records: &[SourceRecord], extras: &[ExtraEvidence]) -> Assessment {
    let mut assessment = Assessment::default();
    let mut affected = 0_u64;
    for record in records {
        let unknown = record
            .evidence_kinds
            .iter()
            .any(|kind| !VALID_KINDS.contains(&kind.as_str()));
        let level = if !record.owner_present || unknown || record.evidence_kinds.is_empty() {
            Level::Error
        } else if record.evidence_kinds.len() == 1
            && record.evidence_kinds[0] == record.expected_kind
        {
            Level::Pass
        } else {
            Level::Warning
        };
        if level != Level::Pass {
            affected = affected.saturating_add(1);
        }
        assessment.push(level);
    }
    for extra in extras {
        assessment.mark_inventory();
        if !VALID_KINDS.contains(&extra.source_kind.as_str()) {
            assessment.push(Level::Error);
            affected = affected.saturating_add(1);
        }
        if !extra.locator_present {
            assessment.mark_inventory();
        }
    }
    assessment.set_metrics(vec![
        count_metric(
            LintMetricCode::EligibleRecords,
            u64::try_from(records.len()).unwrap_or(u64::MAX),
        ),
        count_metric(
            LintMetricCode::ObservedRecords,
            u64::try_from(extras.len()).unwrap_or(u64::MAX),
        ),
        count_metric(LintMetricCode::AffectedRecords, affected),
    ]);
    assessment
}

#[cfg(test)]
pub(super) async fn load_sources(context: &LintContext<'_, '_>) -> Result<Vec<SourceRecord>, ()> {
    let (scope_sql, params) = scoped_pages(context.scope().filter());
    let sql = format!(
        "{}
         SELECT r.page_id, r.memory_source_id, r.owner_present,
                r.expected_kind, pe.source_kind
           FROM resolved r
           LEFT JOIN page_evidence pe
             ON pe.page_id = r.page_id AND pe.locator = r.memory_source_id
          ORDER BY r.page_id, r.memory_source_id, pe.source_kind",
        source_ctes(scope_sql)
    );
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let mut grouped = BTreeMap::<(String, String), SourceBuilder>::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let page_id = row.get::<String>(0).map_err(|_| ())?;
        let locator = row.get::<String>(1).map_err(|_| ())?;
        let owner_present = row.get::<i64>(2).map_err(|_| ())? != 0;
        let expected_kind = row.get::<String>(3).map_err(|_| ())?;
        let builder = grouped
            .entry((page_id, locator))
            .or_insert_with(|| SourceBuilder {
                owner_present,
                expected_kind,
                evidence_kinds: Vec::new(),
            });
        if let Ok(Some(kind)) = row.get::<Option<String>>(4) {
            builder.evidence_kinds.push(kind);
        }
    }
    Ok(grouped
        .into_iter()
        .map(|((_, locator), builder)| SourceRecord {
            locator,
            owner_present: builder.owner_present,
            expected_kind: builder.expected_kind,
            evidence_kinds: builder.evidence_kinds,
        })
        .collect())
}

fn source_ctes(scope_sql: &str) -> String {
    format!(
        "WITH scoped_pages AS ({scope_sql}),
         sources AS MATERIALIZED (
           SELECT ps.page_id, ps.memory_source_id
             FROM page_sources ps
             JOIN scoped_pages sp ON sp.id = ps.page_id
         ),
         source_id_matches AS MATERIALIZED (
           SELECT s.page_id, s.memory_source_id, MIN(m.id) AS matched_id
             FROM sources s
             JOIN memories m
               ON m.source_id = s.memory_source_id AND m.source != 'episode'
            GROUP BY s.page_id, s.memory_source_id
         ),
         resolved_shapes AS MATERIALIZED (
           SELECT s.page_id, s.memory_source_id,
                  COALESCE(ms.source, mi.source, '') AS source,
                  COALESCE(ms.source_agent, mi.source_agent) AS source_agent,
                  COALESCE(ms.source_id, mi.source_id, s.memory_source_id) AS resolved_source_id,
                  CASE WHEN ms.id IS NOT NULL OR mi.id IS NOT NULL THEN 1 ELSE 0 END AS owner_present
             FROM sources s
             LEFT JOIN source_id_matches sm
               ON sm.page_id = s.page_id AND sm.memory_source_id = s.memory_source_id
             LEFT JOIN memories ms ON ms.id = sm.matched_id
             LEFT JOIN memories mi
               ON mi.id = s.memory_source_id AND sm.matched_id IS NULL
              AND mi.source != 'episode'
         ),
         resolved AS MATERIALIZED (
           SELECT page_id, memory_source_id, owner_present,
                  CASE
                    WHEN LOWER(source) = 'authored' OR LOWER(COALESCE(source_agent, '')) = 'authored' THEN 'authored'
                    WHEN SUBSTR(resolved_source_id, 1, 7) = 'http://' OR SUBSTR(resolved_source_id, 1, 8) = 'https://' THEN 'external_url'
                    WHEN LOWER(COALESCE(source_agent, '')) = 'folder' AND INSTR(resolved_source_id, '::') > 0 THEN 'external_file'
                    ELSE 'memory'
                  END AS expected_kind
             FROM resolved_shapes
         )"
    )
}

fn scoped_pages(filter: &ScopeFilter) -> (&'static str, libsql::params::Params) {
    match filter {
        ScopeFilter::Global => ("SELECT id FROM pages", libsql::params::Params::None),
        ScopeFilter::Registered(workspace) => (
            "SELECT id FROM pages WHERE workspace = ?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(workspace.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            "SELECT id FROM pages WHERE workspace = '00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    }
}

fn count_metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}

fn to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}
