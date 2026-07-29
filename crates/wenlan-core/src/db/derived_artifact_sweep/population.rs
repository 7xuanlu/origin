use std::collections::BTreeSet;

use super::SWEEP_INTERVAL_SECONDS;
use crate::error::WenlanError;

pub(super) async fn missing_episode_sources(
    conn: &libsql::Connection,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT m.source_id, m.source_text, m.content
               FROM memories m
              WHERE m.source='memory' AND m.chunk_index=0
                AND m.pending_revision=0 AND COALESCE(m.is_recap,0)=0
                AND m.supersede_mode!='evicted'
                AND NOT EXISTS (
                    SELECT 1 FROM memories r
                     WHERE r.source='memory' AND r.pending_revision=0
                       AND r.supersede_mode='hide' AND r.supersedes=m.source_id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM memories e
                     WHERE e.source='episode' AND e.episode_of=m.source_id
                )
              ORDER BY m.source_id",
            (),
        )
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("derived episode population query: {error}"))
        })?;
    let word_gate = crate::db::episode_word_gate();
    let mut missing = Vec::new();
    while let Some(row) = rows.next().await.map_err(|error| {
        WenlanError::VectorDb(format!("derived episode population row: {error}"))
    })? {
        let source_id = row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("derived episode source id: {error}"))
        })?;
        let source_text = row.get::<Option<String>>(1).map_err(|error| {
            WenlanError::VectorDb(format!("derived episode source text: {error}"))
        })?;
        let content = row
            .get::<String>(2)
            .map_err(|error| WenlanError::VectorDb(format!("derived episode content: {error}")))?;
        if crate::db::derive_episode(&source_id, source_text.as_deref(), &content, word_gate)
            .is_some()
        {
            missing.push(source_id);
        }
    }
    Ok(missing)
}

pub(super) async fn reconcile_receipts(
    conn: &libsql::Connection,
    feature: &str,
    source_ids: &[String],
    observed_at: i64,
) -> Result<(), WenlanError> {
    let current = source_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut rows = conn
        .query(
            "SELECT source_id FROM derived_artifact_sweeps WHERE feature=?1",
            libsql::params![feature],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derived receipt query: {error}")))?;
    let mut prior = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derived receipt row: {error}")))?
    {
        prior.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("derived receipt source id: {error}"))
        })?);
    }
    drop(rows);

    for stale in prior
        .iter()
        .filter(|source_id| !current.contains(source_id.as_str()))
    {
        conn.execute(
            "DELETE FROM derived_artifact_sweeps WHERE feature=?1 AND source_id=?2",
            libsql::params![feature, stale.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derived receipt cleanup: {error}")))?;
    }
    for source_id in current {
        conn.execute(
            &format!(
                "INSERT INTO derived_artifact_sweeps (
                     feature, source_id, first_missing_at, last_sweep_at, completed_sweeps
                 ) VALUES (?1, ?2, ?3, ?3, 0)
                 ON CONFLICT(feature, source_id) DO UPDATE SET
                   completed_sweeps = completed_sweeps +
                     CASE WHEN ?3 - last_sweep_at >= {SWEEP_INTERVAL_SECONDS}
                          THEN 1 ELSE 0 END,
                   last_sweep_at = CASE WHEN ?3 - last_sweep_at >= {SWEEP_INTERVAL_SECONDS}
                                        THEN ?3 ELSE last_sweep_at END"
            ),
            libsql::params![feature, source_id, observed_at],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derived receipt upsert: {error}")))?;
    }
    Ok(())
}
