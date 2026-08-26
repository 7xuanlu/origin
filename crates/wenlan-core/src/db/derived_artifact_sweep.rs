use super::{commit_or_rollback, MemoryDB};
use crate::derived_artifact_state::summary_eligible_predicate;
use crate::error::WenlanError;

mod population;

const SWEEP_INTERVAL_SECONDS: i64 = 30 * 60;

#[derive(Debug, Clone, Copy)]
struct DerivedFeatureSelection {
    episode: bool,
    fact: bool,
    summary: bool,
}

impl DerivedFeatureSelection {
    fn capture() -> Self {
        Self {
            episode: crate::db::episode_channel_enabled(),
            fact: crate::retrieval::fact_channel::fact_channel_enabled(),
            summary: crate::db::global_prelude_enabled(),
        }
    }
}

impl MemoryDB {
    pub async fn record_derived_artifact_sweep(&self) -> Result<(), WenlanError> {
        self.record_derived_artifact_sweep_with(
            chrono::Utc::now().timestamp(),
            DerivedFeatureSelection::capture(),
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn record_derived_artifact_sweep_at(
        &self,
        observed_at: i64,
    ) -> Result<(), WenlanError> {
        self.record_derived_artifact_sweep_with(
            observed_at,
            DerivedFeatureSelection {
                episode: true,
                fact: false,
                summary: false,
            },
        )
        .await
    }

    async fn record_derived_artifact_sweep_with(
        &self,
        observed_at: i64,
        selection: DerivedFeatureSelection,
    ) -> Result<(), WenlanError> {
        let features = [
            (
                "fact",
                selection.fact,
                "TRIM(m.content) != ''
                 AND NOT EXISTS (
                     SELECT 1 FROM child_vectors c
                      WHERE c.parent_kind='memory' AND c.parent_id=m.source_id
                        AND c.embedding IS NOT NULL
                 )"
                .to_string(),
            ),
            (
                "summary",
                selection.summary,
                format!(
                    "{}
                 AND NOT EXISTS (
                     SELECT 1 FROM summary_node_sources s
                      WHERE s.memory_source_id=m.source_id
                 )",
                    summary_eligible_predicate("m")
                ),
            ),
        ];
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derived sweep begin: {error}")))?;
        let result = async {
            if selection.episode {
                let missing = population::missing_episode_sources(&conn).await?;
                population::reconcile_receipts(&conn, "episode", &missing, observed_at).await?;
            } else {
                conn.execute(
                    "DELETE FROM derived_artifact_sweeps WHERE feature = 'episode'",
                    (),
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("derived sweep disabled cleanup: {error}"))
                })?;
            }
            for (feature, enabled, missing) in features {
                if !enabled {
                    conn.execute(
                        "DELETE FROM derived_artifact_sweeps WHERE feature = ?1",
                        libsql::params![feature],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("derived sweep disabled cleanup: {error}"))
                    })?;
                    continue;
                }
                let eligible = format!(
                    "m.source='memory' AND m.chunk_index=0
                     AND m.pending_revision=0 AND COALESCE(m.is_recap,0)=0
                     AND m.supersede_mode!='evicted'
                     AND NOT EXISTS (
                         SELECT 1 FROM memories r
                          WHERE r.source='memory' AND r.pending_revision=0
                            AND r.supersede_mode='hide' AND r.supersedes=m.source_id
                     ) AND ({missing})"
                );
                conn.execute(
                    &format!(
                        "DELETE FROM derived_artifact_sweeps
                          WHERE feature=?1 AND source_id NOT IN (
                              SELECT m.source_id FROM memories m WHERE {eligible}
                          )"
                    ),
                    libsql::params![feature],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("derived sweep stale cleanup: {error}"))
                })?;
                conn.execute(
                    &format!(
                        "INSERT INTO derived_artifact_sweeps (
                             feature, source_id, first_missing_at, last_sweep_at, completed_sweeps
                         )
                         SELECT ?1, m.source_id, ?2, ?2, 0
                           FROM memories m WHERE {eligible}
                         ON CONFLICT(feature, source_id) DO UPDATE SET
                           completed_sweeps = completed_sweeps +
                             CASE WHEN ?2 - last_sweep_at >= {SWEEP_INTERVAL_SECONDS}
                                  THEN 1 ELSE 0 END,
                           last_sweep_at =
                             CASE WHEN ?2 - last_sweep_at >= {SWEEP_INTERVAL_SECONDS}
                                  THEN ?2 ELSE last_sweep_at END"
                    ),
                    libsql::params![feature, observed_at],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("derived sweep upsert: {error}")))?;
            }
            Ok::<(), WenlanError>(())
        }
        .await;
        match result {
            Ok(()) => commit_or_rollback(&conn)
                .await
                .map(|_| ())
                .map_err(|error| WenlanError::VectorDb(format!("derived sweep commit: {error}"))),
            Err(error) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(error)
            }
        }
    }
}
