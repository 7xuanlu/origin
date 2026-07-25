use crate::lint::context::{LintContext, ScopeFilter};

pub(super) struct Counts {
    pub(super) episode: ChannelCount,
    pub(super) fact: ChannelCount,
    pub(super) graph: ChannelCount,
    pub(super) page: ChannelCount,
    pub(super) summary: ChannelCount,
}

pub(super) struct ChannelCount {
    pub(super) eligible: u64,
    pub(super) observed: u64,
}

pub(super) async fn load(context: &LintContext<'_, '_>) -> Result<Counts, ()> {
    let (memory_filter, page_filter, params) = match context.scope().filter() {
        ScopeFilter::Global => ("", " WHERE p.status='active'", libsql::params::Params::None),
        ScopeFilter::Registered(scope) => (
            " AND m.space=?1",
            " WHERE p.status='active' AND p.workspace=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(scope.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            // M3 PR-1 stage e: memories.space is NOT NULL as of migration 91,
            // same honest-columns treatment pages.workspace already got in M1
            // (see the page_filter arm above) -- an unscoped memory now
            // carries the reserved sentinel id, never NULL.
            " AND m.space='00000000-0000-4000-8000-000000000001'",
            " WHERE p.status='active' AND p.workspace='00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    };
    let episode_gate = crate::db::episode_word_gate();
    let summary_eligible = crate::derived_artifact_state::summary_eligible_predicate("m");
    let sql = format!("SELECT
      (SELECT COUNT(DISTINCT CASE WHEN COALESCE(m.word_count,0)>={episode_gate} THEN m.source_id END) FROM memories m WHERE m.source='memory' AND m.chunk_index=0{memory_filter}),
      (SELECT COUNT(DISTINCT e.episode_of) FROM memories e JOIN memories m ON m.source='memory' AND m.chunk_index=0 AND m.source_id=e.episode_of WHERE e.source='episode'{memory_filter}),
      (SELECT COUNT(DISTINCT CASE WHEN TRIM(m.content)!='' THEN m.source_id END) FROM memories m WHERE m.source='memory'{memory_filter}),
      (SELECT COUNT(DISTINCT m.source_id) FROM memories m JOIN child_vectors c ON c.parent_id=m.source_id AND c.parent_kind='memory' AND c.embedding IS NOT NULL WHERE m.source='memory' AND m.chunk_index=0{memory_filter}),
      (SELECT COUNT(DISTINCT m.source_id) FROM memories m WHERE m.source='memory' AND m.chunk_index=0{memory_filter}),
      (SELECT COUNT(DISTINCT m.source_id) FROM memories m JOIN memory_entities e ON e.memory_id=m.source_id WHERE m.source='memory' AND m.chunk_index=0{memory_filter}),
      (SELECT COUNT(DISTINCT m.source_id) FROM memories m WHERE m.source='memory' AND m.chunk_index=0{memory_filter}),
      (SELECT COUNT(*) FROM pages p{page_filter}),
      (SELECT COUNT(DISTINCT CASE WHEN {summary_eligible} THEN m.source_id END) FROM memories m WHERE m.source='memory'{memory_filter}),
      (SELECT COUNT(DISTINCT m.source_id) FROM memories m JOIN summary_node_sources s ON s.memory_source_id=m.source_id WHERE m.source='memory' AND m.chunk_index=0{memory_filter})");
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let row = rows.next().await.map_err(|_| ())?.ok_or(())?;
    Ok(Counts {
        episode: ChannelCount {
            eligible: load_episode_eligible(context).await?,
            observed: count(&row, 1)?,
        },
        fact: channel(&row, 2)?,
        graph: channel(&row, 4)?,
        page: channel(&row, 6)?,
        summary: channel(&row, 8)?,
    })
}

async fn load_episode_eligible(context: &LintContext<'_, '_>) -> Result<u64, ()> {
    let (filter, params) = match context.scope().filter() {
        ScopeFilter::Global => ("", libsql::params::Params::None),
        ScopeFilter::Registered(scope) => (
            " AND m.space=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(scope.clone())]),
        ),
        ScopeFilter::Uncategorized => (
            " AND m.space='00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    };
    let sql = format!(
        "SELECT m.source_id,m.source_text,m.content FROM memories m WHERE m.source='memory' AND m.chunk_index=0{filter} ORDER BY m.source_id"
    );
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let gate = crate::db::episode_word_gate();
    let mut eligible = 0u64;
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let source_id = row.get::<String>(0).map_err(|_| ())?;
        let source_text = row.get::<Option<String>>(1).map_err(|_| ())?;
        let content = row.get::<String>(2).map_err(|_| ())?;
        if crate::db::derive_episode(&source_id, source_text.as_deref(), &content, gate).is_some() {
            eligible = eligible.checked_add(1).ok_or(())?;
        }
    }
    Ok(eligible)
}

fn channel(row: &libsql::Row, index: i32) -> Result<ChannelCount, ()> {
    Ok(ChannelCount {
        eligible: count(row, index)?,
        observed: count(row, index + 1)?,
    })
}

fn count(row: &libsql::Row, index: i32) -> Result<u64, ()> {
    u64::try_from(row.get::<i64>(index).map_err(|_| ())?).map_err(|_| ())
}

pub(super) struct TelemetryCounts {
    pub(super) access: u64,
    pub(super) activity: u64,
}

pub(super) async fn load_telemetry(context: &LintContext<'_, '_>) -> Result<TelemetryCounts, ()> {
    let mut rows = context
        .snapshot()
        .query(
            "SELECT (SELECT COUNT(*) FROM access_log), (SELECT COUNT(*) FROM agent_activity)",
            libsql::params::Params::None,
        )
        .await
        .map_err(|_| ())?;
    let row = rows.next().await.map_err(|_| ())?.ok_or(())?;
    Ok(TelemetryCounts {
        access: count(&row, 0)?,
        activity: count(&row, 1)?,
    })
}
