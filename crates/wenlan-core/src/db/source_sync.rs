// SPDX-License-Identifier: Apache-2.0

use super::{FileSyncState, MemoryDB};
use crate::error::WenlanError;

impl MemoryDB {
    /// Insert or update sync state for a file tracked by a knowledge source.
    pub async fn upsert_sync_state(
        &self,
        source_id: &str,
        file_path: &str,
        mtime_ns: i64,
        content_hash: &str,
    ) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT INTO source_sync_state (source_id, file_path, mtime_ns, content_hash, last_synced_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(source_id, file_path) DO UPDATE SET
                mtime_ns = excluded.mtime_ns,
                content_hash = excluded.content_hash,
                last_synced_at = excluded.last_synced_at",
            libsql::params![
                source_id.to_string(),
                file_path.to_string(),
                mtime_ns,
                content_hash.to_string(),
                now
            ],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("upsert_sync_state: {}", e)))?;
        Ok(())
    }

    /// Get sync state for a specific file in a source.
    pub async fn get_sync_state(
        &self,
        source_id: &str,
        file_path: &str,
    ) -> Result<Option<FileSyncState>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, file_path, mtime_ns, content_hash, last_synced_at
                 FROM source_sync_state WHERE source_id = ?1 AND file_path = ?2",
                libsql::params![source_id.to_string(), file_path.to_string()],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("get_sync_state: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("get_sync_state row: {}", e)))?
        {
            Ok(Some(FileSyncState {
                source_id: row
                    .get::<String>(0)
                    .map_err(|e| WenlanError::VectorDb(format!("sync source_id: {e}")))?,
                file_path: row
                    .get::<String>(1)
                    .map_err(|e| WenlanError::VectorDb(format!("sync file_path: {e}")))?,
                mtime_ns: row
                    .get::<i64>(2)
                    .map_err(|e| WenlanError::VectorDb(format!("sync mtime_ns: {e}")))?,
                content_hash: row
                    .get::<String>(3)
                    .map_err(|e| WenlanError::VectorDb(format!("sync content_hash: {e}")))?,
                last_synced_at: row
                    .get::<i64>(4)
                    .map_err(|e| WenlanError::VectorDb(format!("sync last_synced_at: {e}")))?,
            }))
        } else {
            Ok(None)
        }
    }

    /// List all tracked file paths for a source.
    pub async fn list_sync_state_paths(&self, source_id: &str) -> Result<Vec<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT file_path FROM source_sync_state WHERE source_id = ?1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("list_sync_state_paths: {}", e)))?;
        let mut paths = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("list_sync_state_paths row: {}", e)))?
        {
            paths.push(
                row.get::<String>(0)
                    .map_err(|e| WenlanError::VectorDb(format!("sync path: {e}")))?,
            );
        }
        Ok(paths)
    }

    /// Delete sync state for a specific file in a source.
    pub async fn delete_sync_state(
        &self,
        source_id: &str,
        file_path: &str,
    ) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM source_sync_state WHERE source_id = ?1 AND file_path = ?2",
            libsql::params![source_id.to_string(), file_path.to_string()],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("delete_sync_state: {}", e)))?;
        Ok(())
    }

    /// Delete all sync state entries for a source.
    pub async fn delete_all_sync_state(&self, source_id: &str) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM source_sync_state WHERE source_id = ?1",
            libsql::params![source_id.to_string()],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("delete_all_sync_state: {}", e)))?;
        Ok(())
    }
}
