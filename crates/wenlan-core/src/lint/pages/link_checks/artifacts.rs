use super::super::fs::{EntryKind, EntryScope, PageScan};
use super::super::provenance_checks::result::{Assessment, Level};
use crate::lint::context::LintContext;
use wenlan_types::lint::{LintMetric, LintMetricCode, LintMetricValue};

pub(super) async fn load(context: &LintContext<'_, '_>) -> Result<Assessment, ()> {
    let scan = context.page_scan().ok_or(())?;
    let counts = ArtifactCounts::from_scan(scan);
    let links = load_link_counts(context).await?;
    let mut assessment = Assessment::default();
    assessment.mark_inventory();
    for _ in 0..9 {
        assessment.push(Level::Pass);
    }
    assessment.push(if links.broken == 0 {
        Level::Pass
    } else {
        Level::Warning
    });
    assessment.set_metrics(vec![
        metric(LintMetricCode::ProjectPurposeArtifacts, counts.purpose),
        metric(LintMetricCode::ProjectSchemaArtifacts, counts.schema),
        metric(LintMetricCode::ProjectIndexArtifacts, counts.index),
        metric(LintMetricCode::ProjectLogArtifacts, counts.log),
        metric(LintMetricCode::ProjectOverviewArtifacts, counts.overview),
        metric(LintMetricCode::PageSourceStubs, counts.source_stubs),
        metric(LintMetricCode::ProjectArchiveRecords, links.archived),
        metric(LintMetricCode::ProjectOutboundLinks, links.outbound),
        metric(LintMetricCode::ProjectInboundLinks, links.inbound),
        metric(LintMetricCode::ProjectBrokenLinks, links.broken),
    ]);
    Ok(assessment)
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ArtifactCounts {
    pub(super) purpose: u64,
    pub(super) schema: u64,
    pub(super) index: u64,
    pub(super) log: u64,
    pub(super) overview: u64,
    pub(super) source_stubs: u64,
}

impl ArtifactCounts {
    pub(super) fn from_scan(scan: &PageScan) -> Self {
        let mut counts = Self::default();
        for entry in scan
            .entries
            .iter()
            .filter(|entry| entry.kind == EntryKind::File)
        {
            let basename = entry
                .path
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase();
            match basename.as_str() {
                "purpose.md" => counts.purpose += 1,
                "schema.md" => counts.schema += 1,
                "index.md" | "_index.md" => counts.index += 1,
                "log.md" => counts.log += 1,
                "overview.md" => counts.overview += 1,
                _ => {}
            }
            if entry.scope == EntryScope::SourceInventory && basename.ends_with(".md") {
                counts.source_stubs += 1;
            }
        }
        counts
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct LinkCounts {
    archived: u64,
    outbound: u64,
    inbound: u64,
    broken: u64,
}

async fn load_link_counts(context: &LintContext<'_, '_>) -> Result<LinkCounts, ()> {
    let mut rows = context
        .snapshot()
        .query(
            "SELECT \
               (SELECT COUNT(*) FROM pages WHERE status = 'archived'), \
               COUNT(*) + (SELECT COUNT(*) FROM page_links WHERE target_page_id IS NULL), \
               COALESCE(SUM(CASE WHEN target.status = 'active' THEN 1 ELSE 0 END), 0), \
               COALESCE(SUM(CASE WHEN target.id IS NULL OR target.status != 'active' \
                                 THEN 1 ELSE 0 END), 0) \
             FROM edges e \
             LEFT JOIN pages target ON target.id = e.dst_id \
             WHERE e.edge_type = 'links' AND e.src_kind = 'page' \
               AND e.dst_kind = 'page' AND e.valid_until IS NULL",
            libsql::params::Params::None,
        )
        .await
        .map_err(|_| ())?;
    let row = rows.next().await.map_err(|_| ())?.ok_or(())?;
    Ok(LinkCounts {
        archived: count_column(&row, 0)?,
        outbound: count_column(&row, 1)?,
        inbound: count_column(&row, 2)?,
        broken: count_column(&row, 3)?,
    })
}

fn count_column(row: &libsql::Row, index: i32) -> Result<u64, ()> {
    u64::try_from(row.get::<i64>(index).map_err(|_| ())?).map_err(|_| ())
}

fn metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}
