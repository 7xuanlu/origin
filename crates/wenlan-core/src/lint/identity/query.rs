use super::{CACHES, MEMORY, REGISTRY, SESSIONS, TAGS};
use crate::lint::context::{LintContext, PopulationBasis, ScopeFilter};
use sha2::{Digest as _, Sha256};
use wenlan_types::lint::{
    LintEvidenceRef, LintMetric, LintMetricCode, LintMetricValue, LintOpaqueDigest, LintOpaqueId,
    LintSeverity, LINT_MAX_EVIDENCE_PER_CHECK,
};

pub(super) struct RowCheck {
    population: u64,
    affected: u64,
    evidence: Vec<LintEvidenceRef>,
}

pub(super) struct IdentitySnapshot {
    registry: RowCheck,
    memory: RowCheck,
    tags: RowCheck,
    sessions: RowCheck,
    caches: RowCheck,
    inventory: Inventory,
}

struct Inventory {
    profiles: u64,
    agents: u64,
    spaces: u64,
    decisions: u64,
    pinned: u64,
    stable: u64,
    tags: u64,
    activities: u64,
    captures: u64,
    snapshots: u64,
    briefing: u64,
    narrative: u64,
}

impl IdentitySnapshot {
    pub(super) fn assessments(self) -> [super::result::Assessment; 5] {
        [
            assessment(
                REGISTRY,
                self.registry,
                PopulationBasis::Global,
                vec![
                    metric(LintMetricCode::IdentityProfiles, self.inventory.profiles),
                    metric(LintMetricCode::IdentityAgents, self.inventory.agents),
                    metric(LintMetricCode::IdentitySpaces, self.inventory.spaces),
                ],
            ),
            assessment(
                MEMORY,
                self.memory,
                PopulationBasis::SelectedScope,
                vec![
                    metric(LintMetricCode::DecisionMemories, self.inventory.decisions),
                    metric(LintMetricCode::PinnedMemories, self.inventory.pinned),
                    metric(LintMetricCode::StableMemories, self.inventory.stable),
                ],
            ),
            assessment(
                TAGS,
                self.tags,
                PopulationBasis::Global,
                vec![metric(LintMetricCode::TaggedDocuments, self.inventory.tags)],
            ),
            assessment(
                SESSIONS,
                self.sessions,
                PopulationBasis::Global,
                vec![
                    metric(LintMetricCode::SessionActivities, self.inventory.activities),
                    metric(LintMetricCode::SessionCaptures, self.inventory.captures),
                    metric(LintMetricCode::SessionSnapshots, self.inventory.snapshots),
                ],
            ),
            assessment(
                CACHES,
                self.caches,
                PopulationBasis::Global,
                vec![
                    metric(LintMetricCode::BriefingCacheRows, self.inventory.briefing),
                    metric(LintMetricCode::NarrativeCacheRows, self.inventory.narrative),
                ],
            ),
        ]
    }
}

pub(super) async fn load(context: &LintContext<'_, '_>) -> Result<IdentitySnapshot, ()> {
    let registry = row_check(context, REGISTRY_SQL, libsql::params::Params::None).await?;
    let (scope, params) = memory_scope(context.scope().filter());
    let memory = row_check(context, &MEMORY_SQL.replace("{scope}", &scope), params).await?;
    let tags = tag_check(context).await?;
    let sessions = row_check(context, super::session::SQL, libsql::params::Params::None).await?;
    let caches = row_check(context, CACHE_SQL, libsql::params::Params::None).await?;
    let inventory = inventory(context, context.scope().filter()).await?;
    Ok(IdentitySnapshot {
        registry,
        memory,
        tags,
        sessions,
        caches,
        inventory,
    })
}

const REGISTRY_SQL: &str = "
SELECT invalid FROM (
 SELECT 'profile' AS key, CASE WHEN COUNT(*)=1 THEN 0 ELSE 1 END AS invalid FROM profiles
 UNION ALL SELECT 'profile:'||id, CASE WHEN TRIM(name)='' OR updated_at<created_at THEN 1 ELSE 0 END FROM profiles
 UNION ALL SELECT 'agent:'||id, CASE WHEN TRIM(name)='' OR enabled NOT IN (0,1) OR trust_level NOT IN ('full','review','unknown') OR memory_count<0 OR updated_at<created_at THEN 1 ELSE 0 END FROM agent_connections
 UNION ALL SELECT 'space:'||id, CASE WHEN TRIM(name)='' OR updated_at<created_at THEN 1 ELSE 0 END FROM spaces
) ORDER BY key";

// M3 PR-1 stage e: memories.space is NOT NULL as of migration 91, so an
// unscoped memory carries the reserved sentinel id rather than NULL --
// `!=sentinel` (was `IS NOT NULL`) keeps the "claims a real space name that
// isn't registered" check from false-flagging every unfiled memory.
const MEMORY_SQL: &str = "
SELECT CASE WHEN (m.confirmed IS NOT NULL AND m.confirmed NOT IN (0,1)) OR COALESCE(m.pinned,-1) NOT IN (0,1)
 OR COALESCE(m.pending_revision,-1) NOT IN (0,1) OR (m.pinned=1 AND COALESCE(m.confirmed,0)!=1)
 OR m.stability NOT IN ('new','learned','confirmed') OR m.supersedes=m.source_id
 OR (m.pending_revision=1 AND (m.supersedes IS NULL OR NOT EXISTS(SELECT 1 FROM memories prior WHERE prior.source='memory' AND prior.source_id=m.supersedes)))
 OR (m.space!='00000000-0000-4000-8000-000000000001' AND NOT EXISTS(SELECT 1 FROM spaces s WHERE s.name=m.space))
 OR (m.source_agent IS NOT NULL AND TRIM(m.source_agent)='')
 THEN 1 ELSE 0 END
FROM memories m WHERE m.source='memory' AND m.chunk_index=0{scope} ORDER BY m.source_id";

const TAG_SQL: &str = "
SELECT t.source,t.source_id,t.tag,CASE WHEN TRIM(t.tag)='' OR t.source NOT IN ('memory','page')
 OR (t.source='memory' AND NOT EXISTS(SELECT 1 FROM memories m WHERE m.source_id=t.source_id))
 OR (t.source='page' AND NOT EXISTS(SELECT 1 FROM pages p WHERE p.id=t.source_id))
 THEN 1 ELSE 0 END FROM document_tags t ORDER BY t.source,t.source_id,t.tag";

const CACHE_SQL: &str = "
SELECT invalid FROM (
 SELECT 'briefing:'||id AS key, CASE WHEN id!=1 OR generated_at<0 OR memory_count<0 THEN 1 ELSE 0 END AS invalid FROM briefing_cache
 UNION ALL SELECT 'narrative:'||id, CASE WHEN id!=1 OR generated_at<0 OR memory_count<0 THEN 1 ELSE 0 END FROM narrative_cache
) ORDER BY key";

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
    let mut evidence = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        if row.get::<i64>(0).map_err(|_| ())? != 0 {
            affected = affected.saturating_add(1);
            if evidence.len() < usize::from(LINT_MAX_EVIDENCE_PER_CHECK) {
                let opaque_id = LintOpaqueId::from_sorted_position(
                    usize::try_from(population).map_err(|_| ())?,
                )
                .ok_or(())?;
                evidence.push(LintEvidenceRef::OpaqueId { opaque_id });
            }
        }
        population = population.saturating_add(1);
    }
    Ok(RowCheck {
        population,
        affected,
        evidence,
    })
}

async fn tag_check(context: &LintContext<'_, '_>) -> Result<RowCheck, ()> {
    let mut rows = context
        .snapshot()
        .query(TAG_SQL, libsql::params::Params::None)
        .await
        .map_err(|_| ())?;
    let mut population = 0_u64;
    let mut affected = 0_u64;
    let mut evidence = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        if row.get::<i64>(3).map_err(|_| ())? != 0 {
            affected = affected.saturating_add(1);
            if evidence.len() < usize::from(LINT_MAX_EVIDENCE_PER_CHECK) {
                evidence.push(LintEvidenceRef::OpaqueDigest {
                    opaque_digest: tag_evidence_digest(
                        &row.get::<String>(0).map_err(|_| ())?,
                        &row.get::<String>(1).map_err(|_| ())?,
                        &row.get::<String>(2).map_err(|_| ())?,
                    ),
                });
            }
        }
        population = population.saturating_add(1);
    }
    Ok(RowCheck {
        population,
        affected,
        evidence,
    })
}

fn tag_evidence_digest(source: &str, source_id: &str, tag: &str) -> LintOpaqueDigest {
    let mut digest = Sha256::new();
    digest.update(b"wenlan-lint-tag-evidence-v1");
    for value in [source, source_id, tag] {
        digest.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(value.as_bytes());
    }
    LintOpaqueDigest::from_hex(&format!("{:x}", digest.finalize()))
        .expect("SHA-256 is canonical lowercase hex")
}

fn memory_scope(scope: &ScopeFilter) -> (String, libsql::params::Params) {
    match scope {
        ScopeFilter::Global => (String::new(), libsql::params::Params::None),
        ScopeFilter::Registered(space) => (
            " AND m.space=?1".into(),
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            " AND m.space='00000000-0000-4000-8000-000000000001'".into(),
            libsql::params::Params::None,
        ),
    }
}

async fn inventory(context: &LintContext<'_, '_>, scope: &ScopeFilter) -> Result<Inventory, ()> {
    Ok(Inventory {
        profiles: scalar(context, "SELECT COUNT(*) FROM profiles").await?,
        agents: scalar(context, "SELECT COUNT(*) FROM agent_connections").await?,
        spaces: scalar(context, "SELECT COUNT(*) FROM spaces").await?,
        decisions: memory_count(context, scope, "memory_type='decision'").await?,
        pinned: memory_count(context, scope, "pinned=1").await?,
        stable: memory_count(context, scope, "stability IN ('learned','confirmed')").await?,
        tags: scalar(context, "SELECT COUNT(*) FROM document_tags").await?,
        activities: scalar(context, "SELECT COUNT(*) FROM activities").await?,
        captures: scalar(context, "SELECT COUNT(*) FROM capture_refs").await?,
        snapshots: scalar(context, "SELECT COUNT(*) FROM session_snapshots").await?,
        briefing: scalar(context, "SELECT COUNT(*) FROM briefing_cache").await?,
        narrative: scalar(context, "SELECT COUNT(*) FROM narrative_cache").await?,
    })
}

async fn scalar(context: &LintContext<'_, '_>, sql: &str) -> Result<u64, ()> {
    scalar_params(context, sql, libsql::params::Params::None).await
}

async fn memory_count(
    context: &LintContext<'_, '_>,
    scope: &ScopeFilter,
    predicate: &str,
) -> Result<u64, ()> {
    let (suffix, params) = memory_scope(scope);
    scalar_params(
        context,
        &format!("SELECT COUNT(*) FROM memories m WHERE source='memory' AND chunk_index=0 AND {predicate}{suffix}"),
        params,
    )
    .await
}

async fn scalar_params(
    context: &LintContext<'_, '_>,
    sql: &str,
    params: libsql::params::Params,
) -> Result<u64, ()> {
    let mut rows = context
        .snapshot()
        .query(sql, params)
        .await
        .map_err(|_| ())?;
    let value = rows
        .next()
        .await
        .map_err(|_| ())?
        .ok_or(())?
        .get::<i64>(0)
        .map_err(|_| ())?;
    u64::try_from(value).map_err(|_| ())
}

fn assessment(
    id: &'static str,
    rows: RowCheck,
    basis: PopulationBasis,
    metrics: Vec<LintMetric>,
) -> super::result::Assessment {
    super::result::Assessment {
        id,
        population: rows.population,
        affected: rows.affected,
        evidence: rows.evidence,
        severity: LintSeverity::Error,
        basis,
        metrics,
    }
}

fn metric(code: LintMetricCode, value: u64) -> LintMetric {
    LintMetric::new(code, LintMetricValue::Count { value })
}
