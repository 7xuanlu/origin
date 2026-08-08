use super::catalog::catalog_entry;
use super::context::{LintContext, PopulationBasis, ScopeFilter};
use super::operations::OperationsRunConfig;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use wenlan_types::lint::{
    LintApplicability, LintCheckResult, LintCheckResultInput, LintCoverage, LintEvidenceRef,
    LintGateEffect, LintMetric, LintMetricCode, LintMetricValue, LintOpaqueId, LintOutcome,
    LintPrecondition, LintRecommendationCode, LintSeverity, LintSummaryCode, LintValidationMethod,
    LINT_MAX_EVIDENCE_PER_CHECK,
};

const ALIASES: &str = "entities.alias_integrity";
const MEMORY_DUPLICATES: &str = "memories.duplicate_inventory";
const RETRIEVAL_SUBSTRATE: &str = "memories.retrieval_substrate_inventory";
const CONFLICTS: &str = "memories.structured_conflict_inventory";
const OBSERVATION_DUPLICATES: &str = "observations.duplicate_inventory";
const SOURCE_RESIDUE: &str = "operations.source_lifecycle_residue";
const PAGE_DUPLICATES: &str = "pages.duplicate_body_inventory";
const PAGE_BODY: &str = "pages.projection.body_alignment";
const RELATION_VOCABULARY: &str = "relations.vocabulary_integrity";

#[derive(Default)]
struct RowCheck {
    population: u64,
    affected: u64,
    evidence_positions: Vec<usize>,
}

pub(super) async fn run(
    context: &LintContext<'_, '_>,
    operations: &OperationsRunConfig,
) -> Vec<LintCheckResult> {
    vec![
        query_result(
            context,
            ALIASES,
            alias_integrity(context).await,
            LintMetricCode::AffectedRecords,
            LintSeverity::Error,
        ),
        query_result(
            context,
            MEMORY_DUPLICATES,
            memory_duplicates(context).await,
            LintMetricCode::DeepDuplicateRecords,
            LintSeverity::Warning,
        ),
        query_result(
            context,
            RETRIEVAL_SUBSTRATE,
            retrieval_substrate(context).await,
            LintMetricCode::DeepRetrievalSubstrateMissingRecords,
            LintSeverity::Warning,
        ),
        query_result(
            context,
            CONFLICTS,
            structured_conflicts(context).await,
            LintMetricCode::DeepConflictCandidates,
            LintSeverity::Warning,
        ),
        query_result(
            context,
            OBSERVATION_DUPLICATES,
            observation_duplicates(context).await,
            LintMetricCode::DeepDuplicateRecords,
            LintSeverity::Warning,
        ),
        source_residue_result(context, operations).await,
        query_result(
            context,
            PAGE_DUPLICATES,
            page_duplicates(context).await,
            LintMetricCode::DeepDuplicateRecords,
            LintSeverity::Warning,
        ),
        page_body_result(context).await,
        query_result(
            context,
            RELATION_VOCABULARY,
            relation_vocabulary(context).await,
            LintMetricCode::DeepVocabularyDriftRecords,
            LintSeverity::Error,
        ),
    ]
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: `e.space` now uses `scope_clause_folded`,
// sentinel-aware. The `entity_aliases`/`entities` join itself is a separate,
// still-legacy concern. 1.5b Part 3 (item 8) disposition: not a
// reader-migration target -- this audits the legacy `entities` store's own
// data quality, and reading the shadow-page mirror instead would validate
// the mirror rather than the store it exists to check. Stays on `entities`
// through Stage 1 and Stage 2; retires with the store at Stage 3, same as
// `lint/pages/provenance_checks/source.rs`.
async fn alias_integrity(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause_folded(context.scope().filter(), "e.space", true);
    rows(
        context,
        &format!(
            "SELECT CASE WHEN TRIM(a.alias_name)='' OR e.id IS NULL THEN 1 ELSE 0 END
               FROM entity_aliases a
               LEFT JOIN entities e ON e.id=a.canonical_entity_id
              WHERE 1=1{scope}
              ORDER BY a.alias_name, a.canonical_entity_id"
        ),
        params,
    )
    .await
}

async fn memory_duplicates(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "m.space", false);
    rows(
        context,
        &format!(
            "WITH heads AS (
                 SELECT m.source_id,
                        LOWER(TRIM(MAX(CASE WHEN m.chunk_index=0 THEN m.content END))) AS normalized
                   FROM memories m
                  WHERE m.source='memory' AND m.pending_revision=0
                    AND COALESCE(m.is_recap,0)=0 AND m.supersede_mode!='evicted'{scope}
                    AND NOT EXISTS (
                        SELECT 1 FROM memories r
                         WHERE r.source='memory' AND r.pending_revision=0
                           AND r.supersede_mode='hide' AND r.supersedes=m.source_id)
                  GROUP BY m.source_id
             ), duplicates AS (
                 SELECT normalized FROM heads WHERE normalized!=''
                  GROUP BY normalized HAVING COUNT(*)>1
             )
             SELECT CASE WHEN d.normalized IS NULL THEN 0 ELSE 1 END
               FROM heads h LEFT JOIN duplicates d ON d.normalized=h.normalized
              ORDER BY h.source_id"
        ),
        params,
    )
    .await
}

// G6 Stage 1.3 reader #12 disposition: MIGRATE. The `page_evidence` EXISTS
// below is one of five independent per-channel well-formedness checks (fact,
// graph, page, summary, episode) over the canonical retrieval substrate, not a
// cross-store consistency check between two legacy tables -- so it migrates to
// `edges` like the rest of the table, unlike the provenance-consistency lints
// in `lint/pages/provenance_checks/source.rs`, which stay legacy carryovers.
// S1 (2026-08-05): "page channel present" MEANS "an active `cites` edge
// exists" -- the EXISTS below reads `edges` directly and needs no other
// change. G6 Stage 2 PR 2b (item 4, 2026-08-06): `pages.citations`' D7
// edge-backing role (`cites_backed_by_page_citations`) retired -- an edge
// once kept alive only by that column's backing no longer survives orphan
// cleanup, so this check's substrate is simply "edges", full stop.
async fn retrieval_substrate(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "m.space", false);
    rows(
        context,
        &format!(
            "WITH heads AS (
                 SELECT m.source_id,
                        MAX(CASE WHEN m.embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
                   FROM memories m
                  WHERE m.source='memory' AND m.pending_revision=0
                    AND COALESCE(m.is_recap,0)=0 AND m.supersede_mode!='evicted'{scope}
                    AND NOT EXISTS (
                        SELECT 1 FROM memories r
                         WHERE r.source='memory' AND r.pending_revision=0
                           AND r.supersede_mode='hide' AND r.supersedes=m.source_id)
                  GROUP BY m.source_id
             )
             SELECT CASE WHEN h.embedded=0 OR (
                    NOT EXISTS(SELECT 1 FROM child_vectors c
                                WHERE c.parent_kind='memory' AND c.parent_id=h.source_id)
                AND NOT EXISTS(SELECT 1 FROM memory_entities me WHERE me.memory_id=h.source_id)
                AND NOT EXISTS(SELECT 1 FROM edges pe
                                WHERE pe.edge_type='cites' AND pe.valid_until IS NULL
                                  AND pe.dst_kind='memory' AND pe.dst_id=h.source_id)
                AND NOT EXISTS(SELECT 1 FROM summary_node_sources s
                                WHERE s.memory_source_id=h.source_id)
                AND NOT EXISTS(SELECT 1 FROM memories ep
                                WHERE ep.source='episode' AND ep.episode_of=h.source_id)
             ) THEN 1 ELSE 0 END
               FROM heads h ORDER BY h.source_id"
        ),
        params,
    )
    .await
}

async fn structured_conflicts(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "m.space", false);
    rows(
        context,
        &format!(
            "WITH heads AS (
                 SELECT m.source_id, COALESCE(m.memory_type,'') AS memory_type,
                        COALESCE(m.entity_id,'') AS subject_key,
                        COALESCE(m.structured_fields,'') AS structured_fields
                   FROM memories m
                  WHERE m.source='memory' AND m.pending_revision=0
                    AND COALESCE(m.is_recap,0)=0 AND m.supersede_mode!='evicted'{scope}
                  GROUP BY m.source_id
             ), conflict_keys AS (
                 SELECT memory_type, subject_key FROM heads
                  WHERE memory_type!='' AND subject_key!='' AND structured_fields!=''
                  GROUP BY memory_type, subject_key
                 HAVING COUNT(DISTINCT structured_fields)>1
             )
             SELECT CASE WHEN c.subject_key IS NULL THEN 0 ELSE 1 END
               FROM heads h LEFT JOIN conflict_keys c
                 ON c.memory_type=h.memory_type AND c.subject_key=h.subject_key
              ORDER BY h.source_id"
        ),
        params,
    )
    .await
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: `e.space` now uses `scope_clause_folded`,
// sentinel-aware (entity-existence side only -- observation content itself
// is out of scope for this stage regardless, see the spec's 1.5b
// observations ruling). 1.5b Part 3 (item 8) disposition: not a
// reader-migration target -- this audits the legacy `entities` store's own
// data quality, and reading the shadow-page mirror instead would validate
// the mirror rather than the store it exists to check. Stays on `entities`
// through Stage 1 and Stage 2; retires with the store at Stage 3, same as
// `lint/pages/provenance_checks/source.rs`.
async fn observation_duplicates(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause_folded(context.scope().filter(), "e.space", true);
    rows(
        context,
        &format!(
            "WITH candidates AS (
                 SELECT o.id, o.entity_id, LOWER(TRIM(o.content)) AS normalized
                   FROM observations o JOIN entities e ON e.id=o.entity_id
                  WHERE 1=1{scope}
             ), duplicates AS (
                 SELECT entity_id, normalized FROM candidates WHERE normalized!=''
                  GROUP BY entity_id, normalized HAVING COUNT(*)>1
             )
             SELECT CASE WHEN d.entity_id IS NULL THEN 0 ELSE 1 END
               FROM candidates c LEFT JOIN duplicates d
                 ON d.entity_id=c.entity_id AND d.normalized=c.normalized
              ORDER BY c.id"
        ),
        params,
    )
    .await
}

async fn page_duplicates(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause(context.scope().filter(), "p.workspace", false);
    rows(
        context,
        &format!(
            "WITH candidates AS (
                 SELECT p.id, TRIM(p.content) AS normalized
                   FROM pages p WHERE p.status='active'{scope}
             ), duplicates AS (
                 SELECT normalized FROM candidates WHERE normalized!=''
                  GROUP BY normalized HAVING COUNT(*)>1
             )
             SELECT CASE WHEN d.normalized IS NULL THEN 0 ELSE 1 END
               FROM candidates c LEFT JOIN duplicates d ON d.normalized=c.normalized
              ORDER BY c.id"
        ),
        params,
    )
    .await
}

// G6 Stage 1.5a carryover (2026-08-05), resolved by the 1.5b Part 2
// space-sentinel fold: `f.space` (the src-entity alias) now uses
// `scope_clause_folded`, sentinel-aware. The `entities` join itself is a
// separate, still-legacy concern. 1.5b Part 3 (item 8) disposition: not a
// reader-migration target -- this audits the legacy `entities` store's own
// data quality, and reading the shadow-page mirror instead would validate
// the mirror rather than the store it exists to check. Stays on `entities`
// through Stage 1 and Stage 2; retires with the store at Stage 3, same as
// `lint/pages/provenance_checks/source.rs`.
async fn relation_vocabulary(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let (scope, params) = scope_clause_folded(context.scope().filter(), "f.space", true);
    rows(
        context,
        &format!(
            "SELECT CASE WHEN v.canonical IS NULL THEN 1 ELSE 0 END
               FROM (SELECT * FROM edges WHERE edge_type='relates' AND valid_until IS NULL) r
               LEFT JOIN entities f ON f.id=r.src_id
               LEFT JOIN relation_type_vocabulary v ON v.canonical=r.semantic_type
              WHERE 1=1{scope}
              ORDER BY r.edge_id"
        ),
        params,
    )
    .await
}

async fn source_residue_result(
    context: &LintContext<'_, '_>,
    operations: &OperationsRunConfig,
) -> LintCheckResult {
    if !operations.captured {
        return prerequisite(context, SOURCE_RESIDUE);
    }
    let mut result = RowCheck::default();
    let query = context
        .snapshot()
        .query(
            "SELECT DISTINCT source_id FROM source_sync_state ORDER BY source_id",
            libsql::params::Params::None,
        )
        .await;
    let Ok(mut records) = query else {
        return failed(context, SOURCE_RESIDUE);
    };
    loop {
        match records.next().await {
            Ok(Some(row)) => {
                let Ok(source_id) = row.get::<String>(0) else {
                    return failed(context, SOURCE_RESIDUE);
                };
                let affected = !operations.configured_ids.contains(&source_id);
                push_row(&mut result, affected);
            }
            Ok(None) => break,
            Err(_) => return failed(context, SOURCE_RESIDUE),
        }
    }
    finish(
        context,
        SOURCE_RESIDUE,
        result,
        LintMetricCode::DeepLifecycleResidueRecords,
        LintSeverity::Warning,
    )
}

async fn page_body_result(context: &LintContext<'_, '_>) -> LintCheckResult {
    let Some(scan) = context.page_scan() else {
        return prerequisite(context, PAGE_BODY);
    };
    let digests = scan
        .entries
        .iter()
        .filter_map(|entry| {
            entry
                .body_digest
                .map(|digest| (entry.path.as_str(), digest))
        })
        .collect::<BTreeMap<_, _>>();
    let projected = scan
        .raw_state
        .edges
        .iter()
        .filter_map(|edge| {
            edge.target_path.as_deref().and_then(|path| {
                digests
                    .get(path)
                    .copied()
                    .map(|digest| (edge.state_id.as_str(), digest))
            })
        })
        .collect::<BTreeMap<_, _>>();
    let (scope, params) = scope_clause(context.scope().filter(), "p.workspace", false);
    let query = context
        .snapshot()
        .query(
            &format!(
                "SELECT p.id, p.content FROM pages p
                  WHERE p.status='active'{scope} ORDER BY p.id"
            ),
            params,
        )
        .await;
    let Ok(mut pages) = query else {
        return failed(context, PAGE_BODY);
    };
    let mut result = RowCheck::default();
    loop {
        match pages.next().await {
            Ok(Some(row)) => {
                let (Ok(id), Ok(content)) = (row.get::<String>(0), row.get::<String>(1)) else {
                    return failed(context, PAGE_BODY);
                };
                let Some(projected_digest) = projected.get(id.as_str()) else {
                    continue;
                };
                let canonical = crate::export::provenance::canonicalize_page_body(&content);
                let digest: [u8; 32] = Sha256::digest(canonical.as_bytes()).into();
                push_row(&mut result, projected_digest != &digest);
            }
            Ok(None) => break,
            Err(_) => return failed(context, PAGE_BODY),
        }
    }
    finish(
        context,
        PAGE_BODY,
        result,
        LintMetricCode::DeepPageBodyMismatchRecords,
        LintSeverity::Error,
    )
}

async fn rows(
    context: &LintContext<'_, '_>,
    sql: &str,
    params: libsql::params::Params,
) -> Result<RowCheck, ()> {
    context
        .gate()
        .check_run_for(context.profile(), context.clock().elapsed())
        .map_err(|_| ())?;
    let mut rows = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let mut result = RowCheck::default();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        push_row(&mut result, row.get::<i64>(0).map_err(|_| ())? != 0);
    }
    Ok(result)
}

fn push_row(result: &mut RowCheck, affected: bool) {
    if affected {
        result.affected = result.affected.saturating_add(1);
        if result.evidence_positions.len() < usize::from(LINT_MAX_EVIDENCE_PER_CHECK) {
            if let Ok(position) = usize::try_from(result.population) {
                result.evidence_positions.push(position);
            }
        }
    }
    result.population = result.population.saturating_add(1);
}

fn query_result(
    context: &LintContext<'_, '_>,
    id: &'static str,
    result: Result<RowCheck, ()>,
    metric_code: LintMetricCode,
    severity: LintSeverity,
) -> LintCheckResult {
    match result {
        Ok(result) => finish(context, id, result, metric_code, severity),
        Err(()) => failed(context, id),
    }
}

fn finish(
    context: &LintContext<'_, '_>,
    id: &'static str,
    result: RowCheck,
    metric_code: LintMetricCode,
    severity: LintSeverity,
) -> LintCheckResult {
    let gate_effect = catalog_entry(id)
        .expect("deep check is cataloged")
        .gate_effect;
    let finding = result.affected > 0;
    let evidence = result
        .evidence_positions
        .iter()
        .filter_map(|position| LintOpaqueId::from_sorted_position(*position))
        .map(|opaque_id| LintEvidenceRef::OpaqueId { opaque_id })
        .collect::<Vec<_>>();
    let returned = u64::try_from(evidence.len()).unwrap_or(u64::MAX);
    let check = LintCheckResult::try_new_with_gate_effect(
        LintCheckResultInput {
            check_id: id.to_string(),
            outcome: if finding {
                LintOutcome::Finding
            } else {
                LintOutcome::Pass
            },
            severity: if finding {
                severity
            } else {
                LintSeverity::Info
            },
            applicability: if finding {
                LintApplicability::Applicable
            } else if gate_effect == LintGateEffect::Advisory {
                LintApplicability::Inventory
            } else {
                LintApplicability::Applicable
            },
            precondition: LintPrecondition::Ready,
            coverage: LintCoverage::new(
                LintValidationMethod::FullEnumeration,
                result.population,
                result.population,
                LINT_MAX_EVIDENCE_PER_CHECK,
                result.affected > returned,
                returned,
            )
            .expect("deep row coverage is valid"),
            metrics: {
                let mut metrics = vec![
                    metric(LintMetricCode::ObservedRecords, result.population),
                    metric(LintMetricCode::AffectedRecords, result.affected),
                ];
                if metric_code != LintMetricCode::AffectedRecords {
                    metrics.push(metric(metric_code, result.affected));
                }
                metrics
            },
            summary_code: if finding {
                LintSummaryCode::FindingDetected
            } else {
                LintSummaryCode::CheckPassed
            },
            recommendation_code: finding.then_some(LintRecommendationCode::ReviewFinding),
            evidence,
            duration_ms: context.clock().duration_ms(),
        },
        gate_effect,
    )
    .expect("deep result contract is static");
    context
        .record_population(id, population_basis(context, id), result.population)
        .expect("deep check population is recorded once");
    check
}

fn prerequisite(context: &LintContext<'_, '_>, id: &'static str) -> LintCheckResult {
    terminal(context, id, LintOutcome::NotRunPrerequisite)
}

fn failed(context: &LintContext<'_, '_>, id: &'static str) -> LintCheckResult {
    terminal(context, id, LintOutcome::FailedToRun)
}

fn terminal(
    context: &LintContext<'_, '_>,
    id: &'static str,
    outcome: LintOutcome,
) -> LintCheckResult {
    let prerequisite = outcome == LintOutcome::NotRunPrerequisite;
    let check = LintCheckResult::try_new_with_gate_effect(
        LintCheckResultInput {
            check_id: id.to_string(),
            outcome,
            severity: LintSeverity::Error,
            applicability: if prerequisite {
                LintApplicability::NotApplicable
            } else {
                LintApplicability::Applicable
            },
            precondition: if prerequisite {
                LintPrecondition::MissingPrerequisite
            } else {
                LintPrecondition::Ready
            },
            coverage: LintCoverage::new(
                LintValidationMethod::FullEnumeration,
                0,
                0,
                LINT_MAX_EVIDENCE_PER_CHECK,
                false,
                0,
            )
            .expect("empty deep coverage is valid"),
            metrics: Vec::new(),
            summary_code: if prerequisite {
                LintSummaryCode::PrerequisiteUnavailable
            } else {
                LintSummaryCode::ExecutionFailed
            },
            recommendation_code: Some(if prerequisite {
                LintRecommendationCode::RestorePrerequisite
            } else {
                LintRecommendationCode::InspectRuntime
            }),
            evidence: Vec::new(),
            duration_ms: context.clock().duration_ms(),
        },
        catalog_entry(id)
            .expect("deep check is cataloged")
            .gate_effect,
    )
    .expect("deep terminal result contract is static");
    context
        .record_population(id, population_basis(context, id), 0)
        .expect("deep check population is recorded once");
    check
}

fn population_basis(context: &LintContext<'_, '_>, id: &str) -> PopulationBasis {
    if id == SOURCE_RESIDUE || !context.scope().filter().is_selected() {
        PopulationBasis::Global
    } else {
        PopulationBasis::SelectedScope
    }
}

fn scope_clause(
    scope: &ScopeFilter,
    column: &str,
    exclude_missing_owner: bool,
) -> (String, libsql::params::Params) {
    match scope {
        ScopeFilter::Global => (String::new(), libsql::params::Params::None),
        ScopeFilter::Registered(value) => (
            format!(" AND {column}=?1"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(value.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            format!(
                " AND {column} IS NULL{}",
                if exclude_missing_owner {
                    format!(
                        " AND {} IS NOT NULL",
                        column.trim_end_matches(".space").to_owned() + ".id"
                    )
                } else {
                    String::new()
                }
            ),
            libsql::params::Params::None,
        ),
    }
}

/// Same as `scope_clause`, but for an `entities.space` column folded by the
/// 1.5b space-sentinel migration: an unfiled row stores `UNFILED_SPACE_ID`,
/// not SQL NULL, so `Uncategorized` must match either.
fn scope_clause_folded(
    scope: &ScopeFilter,
    column: &str,
    exclude_missing_owner: bool,
) -> (String, libsql::params::Params) {
    match scope {
        ScopeFilter::Global => (String::new(), libsql::params::Params::None),
        ScopeFilter::Registered(value) => (
            format!(" AND {column}=?1"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(value.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            format!(
                " AND ({column} IS NULL OR {column} = '{}'){}",
                crate::db::UNFILED_SPACE_ID,
                if exclude_missing_owner {
                    format!(
                        " AND {} IS NOT NULL",
                        column.trim_end_matches(".space").to_owned() + ".id"
                    )
                } else {
                    String::new()
                }
            ),
            libsql::params::Params::None,
        ),
    }
}

fn metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}

#[cfg(test)]
#[path = "deep_test.rs"]
mod tests;
