use super::MemoryRecord;
use crate::lint::context::{LintContext, ScopeFilter};
use std::collections::BTreeSet;

pub(super) async fn load_records(context: &LintContext<'_, '_>) -> Result<Vec<MemoryRecord>, ()> {
    let episode_gate = crate::db::episode_word_gate();
    let summary_eligible = crate::derived_artifact_state::summary_eligible_predicate("m");
    let (where_scope, params) = match context.scope().filter() {
        ScopeFilter::Global => ("", libsql::params::Params::None),
        ScopeFilter::Registered(space) => (
            " AND m.space = ?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        ScopeFilter::Uncategorized => (" AND m.space IS NULL", libsql::params::Params::None),
    };
    let sql = format!(
        "WITH grouped AS (
           SELECT m.source_id,
             CASE WHEN COUNT(*) = COUNT(DISTINCT m.chunk_index)
                    AND COUNT(DISTINCT COALESCE(m.supersedes,'')) = 1
                    AND COUNT(DISTINCT m.pending_revision) = 1
                    AND COUNT(DISTINCT COALESCE(m.is_recap,0)) = 1
                    AND COUNT(DISTINCT COALESCE(m.stability,'')) = 1
                    AND COUNT(DISTINCT COALESCE(m.supersede_mode,'')) = 1
                    AND SUM(CASE WHEN COALESCE(m.pending_revision,-1) NOT IN (0,1)
                                       OR COALESCE(m.is_recap,0) NOT IN (0,1)
                                       OR COALESCE(m.stability,'') NOT IN ('new','learned','confirmed')
                                       OR COALESCE(m.supersede_mode,'') NOT IN ('hide','archive','evicted')
                                  THEN 1 ELSE 0 END) = 0
                   THEN 1 ELSE 0 END AS lifecycle_valid,
             MIN(m.supersedes) AS supersedes,
             MAX(m.pending_revision) AS pending_revision,
             MAX(COALESCE(m.is_recap,0)) AS is_recap,
             MAX(CASE WHEN m.supersede_mode='evicted' THEN 1 ELSE 0 END) AS evicted,
             CASE WHEN SUM(CASE WHEN m.embedding IS NULL
                                      AND COALESCE(m.needs_reembed,0)=0
                                 THEN 1 ELSE 0 END)=0
                  THEN 1 ELSE 0 END AS embedding_valid,
             MAX(COALESCE(m.needs_reembed,0)) AS needs_reembed,
             (SELECT COUNT(*) FROM enrichment_steps es
               WHERE es.source_id=m.source_id
                 AND es.status IN ('failed','abandoned')) AS failed_steps,
             MAX(CASE WHEN m.memory_type IS NOT NULL AND m.memory_type != ''
                      THEN 1 ELSE 0 END) AS classified,
             MAX(CASE WHEN m.event_date IS NOT NULL THEN 1 ELSE 0 END) AS event_dated,
             EXISTS(SELECT 1 FROM memories e
                     WHERE e.source='episode' AND e.episode_of=m.source_id) AS episode,
             EXISTS(SELECT 1 FROM child_vectors c
                     WHERE c.parent_kind='memory' AND c.parent_id=m.source_id
                       AND c.embedding IS NOT NULL) AS fact,
             -- G6 Stage 1.3: collapses the page_sources/page_evidence OR-of-two
             -- EXISTS into one EXISTS over `edges` (`cites`, `dst_kind='memory'`).
             EXISTS(SELECT 1 FROM edges p WHERE p.edge_type='cites' AND p.valid_until IS NULL
                     AND p.dst_kind='memory' AND p.dst_id=m.source_id) AS page_link,
             EXISTS(SELECT 1 FROM summary_node_sources s
                     WHERE s.memory_source_id=m.source_id) AS summary,
             MAX(CASE WHEN m.chunk_index=0 THEN m.source_text END) AS episode_source_text,
             MAX(CASE WHEN m.chunk_index=0 THEN m.content END) AS episode_content,
             MAX(CASE WHEN TRIM(m.content) != '' THEN 1 ELSE 0 END) AS fact_eligible,
             MAX(CASE WHEN {summary_eligible} THEN 1 ELSE 0 END) AS summary_eligible
           FROM memories m
          WHERE m.source='memory'{where_scope}
          GROUP BY m.source_id
        )
        SELECT grouped.source_id, grouped.lifecycle_valid, grouped.supersedes,
               CASE WHEN grouped.supersedes IS NULL OR EXISTS(
                   SELECT 1 FROM memories t
                    WHERE t.source='memory' AND t.source_id=grouped.supersedes
               ) THEN 1 ELSE 0 END,
               EXISTS(SELECT 1 FROM memories r
                       WHERE r.source='memory' AND r.pending_revision=0
                         AND r.supersede_mode='hide'
                         AND r.supersedes=grouped.source_id),
               grouped.pending_revision, grouped.is_recap,
               grouped.evicted, grouped.embedding_valid, grouped.needs_reembed,
               grouped.failed_steps, grouped.classified, grouped.event_dated,
               grouped.episode, grouped.fact, grouped.page_link, grouped.summary,
               grouped.episode_source_text, grouped.episode_content,
               grouped.fact_eligible, grouped.summary_eligible,
               episode_sweep.first_missing_at, episode_sweep.completed_sweeps,
               fact_sweep.first_missing_at, fact_sweep.completed_sweeps,
               summary_sweep.first_missing_at, summary_sweep.completed_sweeps
          FROM grouped
          LEFT JOIN derived_artifact_sweeps episode_sweep
            ON episode_sweep.feature='episode' AND episode_sweep.source_id=grouped.source_id
          LEFT JOIN derived_artifact_sweeps fact_sweep
            ON fact_sweep.feature='fact' AND fact_sweep.source_id=grouped.source_id
          LEFT JOIN derived_artifact_sweeps summary_sweep
            ON summary_sweep.feature='summary' AND summary_sweep.source_id=grouped.source_id
         ORDER BY grouped.source_id"
    );
    let mut rows = context
        .snapshot()
        .query(&sql, params)
        .await
        .map_err(|_| ())?;
    let mut records = Vec::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        let source_id = row.get::<String>(0).map_err(|_| ())?;
        let episode_source_text = row.get::<Option<String>>(17).map_err(|_| ())?;
        let episode_content = row.get::<Option<String>>(18).map_err(|_| ())?;
        let episode_eligible = episode_content.as_deref().is_some_and(|content| {
            crate::db::derive_episode(
                &source_id,
                episode_source_text.as_deref(),
                content,
                episode_gate,
            )
            .is_some()
        });
        records.push(MemoryRecord {
            source_id,
            lifecycle_valid: row.get::<i64>(1).map_err(|_| ())? != 0,
            supersedes: row.get(2).ok(),
            target_exists: row.get::<i64>(3).map_err(|_| ())? != 0,
            replaced_by_active: row.get::<i64>(4).map_err(|_| ())? != 0,
            pending_revision: row.get::<i64>(5).map_err(|_| ())? != 0,
            recap: row.get::<i64>(6).map_err(|_| ())? != 0,
            evicted: row.get::<i64>(7).map_err(|_| ())? != 0,
            embedding_valid: row.get::<i64>(8).map_err(|_| ())? != 0,
            needs_reembed: row.get::<i64>(9).map_err(|_| ())? != 0,
            failed_steps: u64::try_from(row.get::<i64>(10).map_err(|_| ())?).map_err(|_| ())?,
            classified: row.get::<i64>(11).map_err(|_| ())? != 0,
            event_dated: row.get::<i64>(12).map_err(|_| ())? != 0,
            episode: row.get::<i64>(13).map_err(|_| ())? != 0,
            fact: row.get::<i64>(14).map_err(|_| ())? != 0,
            page_link: row.get::<i64>(15).map_err(|_| ())? != 0,
            summary: row.get::<i64>(16).map_err(|_| ())? != 0,
            episode_eligible,
            fact_eligible: row.get::<i64>(19).map_err(|_| ())? != 0,
            summary_eligible: row.get::<i64>(20).map_err(|_| ())? != 0,
            episode_receipt: receipt(&row, 21, 22)?,
            fact_receipt: receipt(&row, 23, 24)?,
            summary_receipt: receipt(&row, 25, 26)?,
            head: false,
        });
    }
    Ok(records)
}

fn receipt(
    row: &libsql::Row,
    first_index: i32,
    sweep_index: i32,
) -> Result<Option<super::DerivedSweepReceipt>, ()> {
    let Some(first_missing_at) = row.get::<Option<i64>>(first_index).map_err(|_| ())? else {
        return Ok(None);
    };
    let completed_sweeps = row
        .get::<Option<i64>>(sweep_index)
        .map_err(|_| ())?
        .unwrap_or(0);
    Ok(Some(super::DerivedSweepReceipt {
        first_missing_at,
        completed_sweeps: u64::try_from(completed_sweeps).map_err(|_| ())?,
    }))
}

pub(super) async fn apply_fact_index_visibility(
    context: &LintContext<'_, '_>,
    records: &mut [MemoryRecord],
) -> Result<(), ()> {
    let mut probe_rows = context
        .snapshot()
        .query(
            "SELECT embedding FROM child_vectors
              WHERE parent_kind='memory' AND embedding IS NOT NULL
              ORDER BY rowid LIMIT 1",
            libsql::params::Params::None,
        )
        .await
        .map_err(|_| ())?;
    let Some(probe_row) = probe_rows.next().await.map_err(|_| ())? else {
        return Ok(());
    };
    let probe = probe_row.get::<Vec<u8>>(0).map_err(|_| ())?;
    drop(probe_rows);
    let mut count_rows = context
        .snapshot()
        .query(
            "SELECT COUNT(*) FROM child_vectors
              WHERE parent_kind='memory' AND embedding IS NOT NULL",
            libsql::params::Params::None,
        )
        .await
        .map_err(|_| ())?;
    let total = count_rows
        .next()
        .await
        .map_err(|_| ())?
        .ok_or(())?
        .get::<i64>(0)
        .map_err(|_| ())?;
    drop(count_rows);
    let mut indexed_rows = context
        .snapshot()
        .query(
            "SELECT DISTINCT cv.parent_id
               FROM vector_top_k('child_vectors_vec_idx', ?1, ?2) AS vt
               JOIN child_vectors cv ON cv.rowid=vt.id
              WHERE cv.parent_kind='memory'
              ORDER BY cv.parent_id",
            libsql::params::Params::Positional(vec![
                libsql::Value::Blob(probe),
                libsql::Value::Integer(total),
            ]),
        )
        .await
        .map_err(|_| ())?;
    let mut indexed = BTreeSet::new();
    while let Some(row) = indexed_rows.next().await.map_err(|_| ())? {
        indexed.insert(row.get::<String>(0).map_err(|_| ())?);
    }
    for record in records {
        record.fact = record.fact && indexed.contains(&record.source_id);
    }
    Ok(())
}
