// SPDX-License-Identifier: Apache-2.0
//! Durable cursor and lease-cleanup state for community grouping.

use super::MemoryDB;
use crate::community_grouping::CommunityGroupingError;
use std::sync::Arc;

/// Durable queue position only; advancing it never acknowledges publication.
const COMMUNITY_GROUPING_SPACE_CURSOR_KEY: &str = "community_grouping_space_cursor";

pub(crate) struct CommunityGroupingLeaseCleanup {
    conn: Arc<tokio::sync::Mutex<libsql::Connection>>,
    space: String,
    input_generation: i64,
    token: String,
    armed: bool,
}

impl std::fmt::Debug for CommunityGroupingLeaseCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommunityGroupingLeaseCleanup")
            .field("space", &self.space)
            .field("input_generation", &self.input_generation)
            .field("armed", &self.armed)
            .finish_non_exhaustive()
    }
}

impl CommunityGroupingLeaseCleanup {
    fn new(
        conn: Arc<tokio::sync::Mutex<libsql::Connection>>,
        space: String,
        input_generation: i64,
        token: String,
    ) -> Self {
        Self {
            conn,
            space,
            input_generation,
            token,
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CommunityGroupingLeaseCleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let conn = Arc::clone(&self.conn);
        let space = self.space.clone();
        let input_generation = self.input_generation;
        let token = self.token.clone();
        runtime.spawn(async move {
            let conn = conn.lock().await;
            let _ = conn
                .execute(
                    "DELETE FROM grouping_leases
                     WHERE phase = 'community' AND space = ?1
                       AND input_generation = ?2 AND token = ?3",
                    libsql::params![space, input_generation, token],
                )
                .await;
        });
    }
}

impl MemoryDB {
    pub(crate) async fn claim_next_dirty_community_space(
        &self,
    ) -> Result<Option<String>, CommunityGroupingError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT space FROM space_graph_state \
                 WHERE dirty = 1 \
                 ORDER BY CASE \
                     WHEN space > ( \
                         SELECT value FROM app_metadata WHERE key = ?1) \
                     THEN 0 ELSE 1 \
                 END, space \
                 LIMIT 1",
                libsql::params![COMMUNITY_GROUPING_SPACE_CURSOR_KEY],
            )
            .await
            .map_err(|error| {
                CommunityGroupingError::Database(format!(
                    "select next dirty community space: {error}"
                ))
            })?;
        let space = rows
            .next()
            .await
            .map_err(|error| {
                CommunityGroupingError::Database(format!(
                    "read next dirty community space: {error}"
                ))
            })?
            .map(|row| {
                row.get::<String>(0).map_err(|error| {
                    CommunityGroupingError::Database(format!(
                        "decode next dirty community space: {error}"
                    ))
                })
            })
            .transpose()?;
        drop(rows);
        if let Some(selected_space) = space.as_deref() {
            conn.execute(
                "INSERT INTO app_metadata (key, value) VALUES (?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                libsql::params![COMMUNITY_GROUPING_SPACE_CURSOR_KEY, selected_space],
            )
            .await
            .map_err(|error| {
                CommunityGroupingError::Database(format!(
                    "advance dirty community space cursor: {error}"
                ))
            })?;
        }
        Ok(space)
    }

    pub(super) fn mint_community_grouping_lease_cleanup(
        &self,
        space: String,
        input_generation: i64,
        token: String,
    ) -> CommunityGroupingLeaseCleanup {
        CommunityGroupingLeaseCleanup::new(Arc::clone(&self.conn), space, input_generation, token)
    }
}

#[cfg(test)]
#[path = "community_grouping_state_test.rs"]
mod tests;
