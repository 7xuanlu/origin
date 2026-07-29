// SPDX-License-Identifier: Apache-2.0
use super::{MemoryDB, WenlanError};

impl MemoryDB {
    // Migration 4: Refinement pipeline columns + queue table
    pub(super) async fn migrate_4_refinement_pipeline(&self) -> Result<(), WenlanError> {
        let chunk_cols = self.get_table_columns("memories").await?;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration4 begin: {}", e)))?;

        if !chunk_cols.contains("access_count") {
            conn.execute(
                "ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0",
                (),
            )
            .await
            .map_err(|e| {
                WenlanError::VectorDb(format!("alter memories add access_count: {}", e))
            })?;
        }
        if !chunk_cols.contains("last_accessed") {
            conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT", ())
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("alter memories add last_accessed: {}", e))
                })?;
        }
        if !chunk_cols.contains("refinement_status") {
            conn.execute("ALTER TABLE memories ADD COLUMN refinement_status TEXT", ())
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("alter memories add refinement_status: {}", e))
                })?;
        }
        if !chunk_cols.contains("effective_confidence") {
            conn.execute(
                "ALTER TABLE memories ADD COLUMN effective_confidence REAL",
                (),
            )
            .await
            .map_err(|e| {
                WenlanError::VectorDb(format!("alter memories add effective_confidence: {}", e))
            })?;
        }

        conn.execute(
            "CREATE TABLE IF NOT EXISTS refinement_queue (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                source_ids TEXT NOT NULL,
                payload TEXT,
                confidence REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                resolved_at TEXT
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create refinement_queue: {}", e)))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_refinement_status ON refinement_queue(status)
             WHERE status IN ('pending', 'awaiting_review')",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create refinement_status index: {}", e)))?;

        conn.execute("PRAGMA user_version = 4", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("set user_version=4: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration4 commit: {}", e)))?;

        log::info!("[memory_db] migration 4: added refinement columns + queue table");
        Ok(())
    }

    // Migration 5: Session tables consolidated from session_db
    pub(super) async fn migrate_5_session_tables(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration5 begin: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS activities (
                id TEXT PRIMARY KEY,
                started_at INTEGER NOT NULL,
                ended_at INTEGER NOT NULL
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create activities: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS capture_refs (
                source_id TEXT PRIMARY KEY,
                activity_id TEXT NOT NULL,
                snapshot_id TEXT,
                app_name TEXT NOT NULL,
                window_title TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                source TEXT NOT NULL
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create capture_refs: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_snapshots (
                id TEXT PRIMARY KEY,
                activity_id TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                ended_at INTEGER NOT NULL,
                primary_apps TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL,
                capture_count INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create session_snapshots: {}", e)))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_captures_activity ON capture_refs(activity_id)",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create capture_refs idx: {}", e)))?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_unpackaged ON capture_refs(activity_id) WHERE snapshot_id IS NULL", ())
            .await.map_err(|e| WenlanError::VectorDb(format!("create unpackaged idx: {}", e)))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_time ON session_snapshots(started_at DESC)",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create snapshots time idx: {}", e)))?;

        conn.execute("PRAGMA user_version = 5", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("set user_version=5: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration5 commit: {}", e)))?;

        log::info!("[memory_db] migration 5: session tables consolidated from session_db");
        Ok(())
    }

    // Migration 6: word_count column on memories + access_log table
    pub(super) async fn migrate_6_access_tracking(&self) -> Result<(), WenlanError> {
        let chunk_cols = self.get_table_columns("memories").await?;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration6 begin: {}", e)))?;

        if !chunk_cols.contains("word_count") {
            conn.execute(
                "ALTER TABLE memories ADD COLUMN word_count INTEGER NOT NULL DEFAULT 0",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("alter memories add word_count: {}", e)))?;
        }

        conn.execute(
            "UPDATE memories SET word_count = LENGTH(content) - LENGTH(REPLACE(content, ' ', '')) + 1 WHERE content IS NOT NULL AND content != ''",
            (),
        ).await.map_err(|e| WenlanError::VectorDb(format!("backfill word_count: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS access_log (source_id TEXT NOT NULL, accessed_at INTEGER NOT NULL)",
            (),
        ).await.map_err(|e| WenlanError::VectorDb(format!("create access_log: {}", e)))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_log_time ON access_log(accessed_at)",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create access_log time index: {}", e)))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_log_source ON access_log(source_id)",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create access_log source index: {}", e)))?;

        conn.execute("PRAGMA user_version = 6", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("set user_version=6: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration6 commit: {}", e)))?;

        log::info!("[memory_db] migration 6: added word_count column + access_log table");
        Ok(())
    }

    // Migration 7: Briefing cache table
    pub(super) async fn migrate_7_briefing_cache(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration7 begin: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS briefing_cache (
                id INTEGER PRIMARY KEY DEFAULT 1,
                content TEXT NOT NULL,
                generated_at INTEGER NOT NULL,
                memory_count INTEGER NOT NULL
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create briefing_cache: {}", e)))?;

        conn.execute("PRAGMA user_version = 7", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("set user_version=7: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration7 commit: {}", e)))?;

        log::info!("[memory_db] migration 7: added briefing_cache table");
        Ok(())
    }

    // Migration 8: Narrative cache table
    pub(super) async fn migrate_8_narrative_cache(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration8 begin: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS narrative_cache (
                id INTEGER PRIMARY KEY DEFAULT 1,
                content TEXT NOT NULL,
                generated_at INTEGER NOT NULL,
                memory_count INTEGER NOT NULL
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create narrative_cache: {}", e)))?;

        conn.execute("PRAGMA user_version = 8", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("set user_version=8: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration8 commit: {}", e)))?;

        log::info!("[memory_db] migration 8: added narrative_cache table");
        Ok(())
    }

    // Migration 9: Agent activity table (impact tracking)
    pub(super) async fn migrate_9_agent_activity(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration9 begin: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                action TEXT NOT NULL,
                memory_ids TEXT,
                query TEXT,
                detail TEXT
            )",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create agent_activity: {}", e)))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_time ON agent_activity(timestamp)",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create idx_activity_time: {}", e)))?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_agent ON agent_activity(agent_name)",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("create idx_activity_agent: {}", e)))?;

        conn.execute("PRAGMA user_version = 9", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("set user_version=9: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("migration9 commit: {}", e)))?;

        log::info!("[memory_db] migration 9: added agent_activity table");
        Ok(())
    }
}
