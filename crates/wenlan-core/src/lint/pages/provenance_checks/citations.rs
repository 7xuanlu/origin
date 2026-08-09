use super::result::{Assessment, Level};
use crate::lint::context::{LintContext, ScopeFilter};
use wenlan_types::lint::{LintMetric, LintMetricCode, LintMetricValue};
use wenlan_types::pages::PageCitation;

const VALID_KINDS: [&str; 4] = ["authored", "external_file", "external_url", "memory"];
const VALID_STATUSES: [&str; 2] = ["unverified", "verified"];
const VALID_SCOPES: [&str; 2] = ["paragraph", "sentence"];

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct CitationPartitions {
    pub(super) null_pages: u64,
    pub(super) empty_pages: u64,
    pub(super) nonempty_pages: u64,
    pub(super) occurrences: u64,
    pub(super) verified: u64,
    pub(super) unverified: u64,
    pub(super) sentence: u64,
    pub(super) paragraph: u64,
    pub(super) memory: u64,
    pub(super) external_file: u64,
    pub(super) external_url: u64,
    pub(super) authored: u64,
}

impl CitationPartitions {
    pub(super) fn page_total(&self) -> u64 {
        self.null_pages
            .saturating_add(self.empty_pages)
            .saturating_add(self.nonempty_pages)
    }

    #[cfg(test)]
    pub(super) fn partitions_are_exact(&self) -> bool {
        self.verified.saturating_add(self.unverified) == self.occurrences
            && self.sentence.saturating_add(self.paragraph) == self.occurrences
            && self
                .memory
                .saturating_add(self.external_file)
                .saturating_add(self.external_url)
                .saturating_add(self.authored)
                == self.occurrences
    }
}

pub(super) async fn load_and_assess_citations(
    context: &LintContext<'_, '_>,
) -> Result<Assessment, ()> {
    let rows = load_citation_rows(context).await?;
    Ok(assess_citations(&rows).0)
}

pub(super) fn assess_citations(rows: &[Option<String>]) -> (Assessment, CitationPartitions) {
    let mut assessment = Assessment::default();
    let mut partitions = CitationPartitions::default();
    let mut affected = 0_u64;
    for raw in rows {
        let level = match raw {
            None => {
                partitions.null_pages = partitions.null_pages.saturating_add(1);
                assessment.mark_inventory();
                Level::Pass
            }
            Some(raw) => match serde_json::from_str::<Vec<PageCitation>>(raw) {
                Ok(citations) if citations.is_empty() => {
                    partitions.empty_pages = partitions.empty_pages.saturating_add(1);
                    assessment.mark_inventory();
                    Level::Pass
                }
                Ok(citations) => {
                    partitions.nonempty_pages = partitions.nonempty_pages.saturating_add(1);
                    assessment.mark_inventory();
                    if add_occurrences(&mut partitions, &citations) {
                        Level::Pass
                    } else {
                        Level::Error
                    }
                }
                Err(_) => {
                    partitions.nonempty_pages = partitions.nonempty_pages.saturating_add(1);
                    Level::Error
                }
            },
        };
        if level != Level::Pass {
            affected = affected.saturating_add(1);
        }
        assessment.push(level);
    }
    debug_assert_eq!(
        partitions.page_total(),
        u64::try_from(rows.len()).unwrap_or(u64::MAX)
    );
    assessment.set_metrics(vec![
        count_metric(
            LintMetricCode::EligibleRecords,
            u64::try_from(rows.len()).unwrap_or(u64::MAX),
        ),
        count_metric(LintMetricCode::ObservedRecords, partitions.occurrences),
        count_metric(LintMetricCode::AffectedRecords, affected),
        count_metric(LintMetricCode::PendingRecords, partitions.null_pages),
        count_metric(LintMetricCode::CitationNullPages, partitions.null_pages),
        count_metric(LintMetricCode::CitationEmptyPages, partitions.empty_pages),
        count_metric(
            LintMetricCode::CitationNonemptyPages,
            partitions.nonempty_pages,
        ),
        count_metric(
            LintMetricCode::CitationVerifiedOccurrences,
            partitions.verified,
        ),
        count_metric(
            LintMetricCode::CitationUnverifiedOccurrences,
            partitions.unverified,
        ),
        count_metric(
            LintMetricCode::CitationSentenceOccurrences,
            partitions.sentence,
        ),
        count_metric(
            LintMetricCode::CitationParagraphOccurrences,
            partitions.paragraph,
        ),
        count_metric(LintMetricCode::CitationMemoryOccurrences, partitions.memory),
        count_metric(
            LintMetricCode::CitationExternalFileOccurrences,
            partitions.external_file,
        ),
        count_metric(
            LintMetricCode::CitationExternalUrlOccurrences,
            partitions.external_url,
        ),
        count_metric(
            LintMetricCode::CitationAuthoredOccurrences,
            partitions.authored,
        ),
    ]);
    (assessment, partitions)
}

fn count_metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}

fn add_occurrences(partitions: &mut CitationPartitions, citations: &[PageCitation]) -> bool {
    let mut valid = true;
    for citation in citations {
        partitions.occurrences = partitions.occurrences.saturating_add(1);
        match citation.status.as_str() {
            "verified" => partitions.verified = partitions.verified.saturating_add(1),
            "unverified" => partitions.unverified = partitions.unverified.saturating_add(1),
            _ => valid = false,
        }
        match citation.scope.as_str() {
            "sentence" => partitions.sentence = partitions.sentence.saturating_add(1),
            "paragraph" => partitions.paragraph = partitions.paragraph.saturating_add(1),
            _ => valid = false,
        }
        match citation.source_kind.as_str() {
            "memory" => partitions.memory = partitions.memory.saturating_add(1),
            "external_file" => {
                partitions.external_file = partitions.external_file.saturating_add(1)
            }
            "external_url" => partitions.external_url = partitions.external_url.saturating_add(1),
            "authored" => partitions.authored = partitions.authored.saturating_add(1),
            _ => valid = false,
        }
        valid &= VALID_STATUSES.contains(&citation.status.as_str())
            && VALID_SCOPES.contains(&citation.scope.as_str())
            && VALID_KINDS.contains(&citation.source_kind.as_str());
    }
    valid
}

async fn load_citation_rows(context: &LintContext<'_, '_>) -> Result<Vec<Option<String>>, ()> {
    let (sql, params) = match context.scope().filter() {
        ScopeFilter::Global => (
            "SELECT citations FROM pages ORDER BY id",
            libsql::params::Params::None,
        ),
        ScopeFilter::Registered(workspace) => (
            "SELECT citations FROM pages WHERE workspace = ?1 ORDER BY id",
            libsql::params::Params::Positional(vec![libsql::Value::Text(workspace.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            "SELECT citations FROM pages WHERE workspace = '00000000-0000-4000-8000-000000000001' ORDER BY id",
            libsql::params::Params::None,
        ),
    };
    let mut rows = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let mut citations = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        citations.push(row.get(0).map_err(|_| ())?);
    }
    Ok(citations)
}
