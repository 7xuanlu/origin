// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

const BRIEF_TABLES_DDL: &str = "
CREATE TABLE IF NOT EXISTS briefs (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id) ON DELETE CASCADE,
    last_session_summary TEXT NOT NULL DEFAULT '',
    last_handoff_at INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS brief_items (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    text TEXT NOT NULL CHECK(length(trim(text)) > 0),
    state TEXT NOT NULL CHECK(state IN ('active', 'backlog')),
    added_at INTEGER NOT NULL,
    gate TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_brief_items_space_state
    ON brief_items(space_id, state, added_at, id);
";

impl MemoryDB {
    pub(super) async fn ensure_brief_tables(tx: &libsql::Transaction) -> Result<(), WenlanError> {
        tx.execute_batch(BRIEF_TABLES_DDL)
            .await
            .map(|_| ())
            .map_err(|error| WenlanError::VectorDb(format!("brief tables DDL: {error}")))
    }

    pub(super) async fn migrate_100_brief(&self, prior_version: i64) -> Result<(), WenlanError> {
        self.backup_before_migration(100, prior_version).await?;

        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m100 brief begin: {error}")))?;
        Self::ensure_brief_tables(&tx).await?;
        tx.commit()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m100 brief commit: {error}")))?;

        conn.execute("PRAGMA user_version = 100", ())
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m100 brief bump: {error}")))?;
        log::info!("[migration] Migration 100 applied: Space-owned Brief tables");
        Ok(())
    }
}

fn validate_update_request(request: &wenlan_types::BriefUpdateRequest) -> Result<(), WenlanError> {
    if request.space.trim().is_empty() {
        return Err(WenlanError::Validation(
            "brief update requires a non-empty space".into(),
        ));
    }
    if request.caller_id.trim().is_empty() || request.operation_id.trim().is_empty() {
        return Err(WenlanError::Validation(
            "brief update requires caller_id and operation_id".into(),
        ));
    }
    if let Some(summary) = &request.summary {
        if summary.text.trim().is_empty() {
            return Err(WenlanError::Validation(
                "brief summary must not be empty".into(),
            ));
        }
        if summary.expected_version < 0 {
            return Err(WenlanError::Validation(
                "brief summary expected_version must be non-negative".into(),
            ));
        }
    }
    for mutation in &request.mutations {
        match mutation {
            wenlan_types::BriefMutation::Add { text, added_at, .. } => {
                if text.trim().is_empty() {
                    return Err(WenlanError::Validation(
                        "brief item text must not be empty".into(),
                    ));
                }
                if added_at.is_some_and(|value| value <= 0) {
                    return Err(WenlanError::Validation(
                        "brief item added_at must be positive".into(),
                    ));
                }
            }
            wenlan_types::BriefMutation::Edit {
                text,
                expected_version,
                ..
            } => {
                if text.trim().is_empty() {
                    return Err(WenlanError::Validation(
                        "brief item text must not be empty".into(),
                    ));
                }
                if *expected_version < 1 {
                    return Err(WenlanError::Validation(
                        "brief item expected_version must be positive".into(),
                    ));
                }
            }
            wenlan_types::BriefMutation::Move {
                expected_version, ..
            }
            | wenlan_types::BriefMutation::SetGate {
                expected_version, ..
            }
            | wenlan_types::BriefMutation::Complete {
                expected_version, ..
            } => {
                if *expected_version < 1 {
                    return Err(WenlanError::Validation(
                        "brief item expected_version must be positive".into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn update_request_digest(
    request: &wenlan_types::BriefUpdateRequest,
) -> Result<String, WenlanError> {
    use sha2::{Digest, Sha256};

    let encoded = serde_json::to_vec(request)?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

async fn current_item_version(
    tx: &libsql::Transaction,
    space_id: &str,
    item_id: &str,
) -> Result<Option<i64>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT version FROM brief_items WHERE space_id=?1 AND id=?2",
            libsql::params![space_id, item_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("brief item version: {error}")))?;
    let version = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("brief item version row: {error}")))?
        .map(|row| row.get::<i64>(0).unwrap_or_default());
    Ok(version)
}

fn item_conflict(
    mutation_index: usize,
    item_id: &str,
    current_version: Option<i64>,
) -> wenlan_types::BriefMutationConflict {
    wenlan_types::BriefMutationConflict {
        mutation_index: Some(mutation_index),
        item_id: Some(item_id.to_string()),
        reason: if current_version.is_some() {
            wenlan_types::BriefConflictReason::VersionMismatch
        } else {
            wenlan_types::BriefConflictReason::ItemNotFound
        },
        current_version,
    }
}

impl MemoryDB {
    pub async fn get_brief_by_space_name(
        &self,
        space_name: &str,
    ) -> Result<Option<wenlan_types::Brief>, WenlanError> {
        let conn = self.conn.lock().await;
        let header = {
            let mut rows = conn
                .query(
                    "SELECT b.space_id, s.name, b.last_session_summary,
                            b.last_handoff_at, b.version
                       FROM briefs b
                       JOIN spaces s ON s.id=b.space_id
                      WHERE s.name=?1",
                    libsql::params![space_name],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("get brief: {error}")))?;
            let Some(row) = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("get brief row: {error}")))?
            else {
                return Ok(None);
            };
            (
                row.get::<String>(0).unwrap_or_default(),
                row.get::<String>(1).unwrap_or_default(),
                row.get::<String>(2).unwrap_or_default(),
                row.get::<Option<i64>>(3).unwrap_or(None),
                row.get::<i64>(4).unwrap_or(1),
            )
        };

        let mut rows = conn
            .query(
                "SELECT id, text, state, added_at, gate, version
                   FROM brief_items
                  WHERE space_id=?1
                  ORDER BY CASE state WHEN 'active' THEN 0 ELSE 1 END,
                           added_at, id",
                libsql::params![header.0.as_str()],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("get brief items: {error}")))?;
        let mut active = Vec::new();
        let mut backlog = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("get brief item row: {error}")))?
        {
            let state_text = row.get::<String>(2).unwrap_or_default();
            let state = match state_text.as_str() {
                "active" => wenlan_types::BriefItemState::Active,
                "backlog" => wenlan_types::BriefItemState::Backlog,
                other => {
                    return Err(WenlanError::VectorDb(format!(
                        "brief item has invalid state '{other}'"
                    )));
                }
            };
            let item = wenlan_types::BriefItem {
                id: row.get::<String>(0).unwrap_or_default(),
                text: row.get::<String>(1).unwrap_or_default(),
                state,
                added_at: row.get::<i64>(3).unwrap_or_default(),
                gate: row.get::<Option<String>>(4).unwrap_or(None),
                version: row.get::<i64>(5).unwrap_or(1),
            };
            match state {
                wenlan_types::BriefItemState::Active => active.push(item),
                wenlan_types::BriefItemState::Backlog => backlog.push(item),
            }
        }

        Ok(Some(wenlan_types::Brief {
            space_id: header.0,
            space: header.1,
            last_session_summary: header.2,
            last_handoff_at: header.3,
            version: header.4,
            active,
            backlog,
        }))
    }

    pub async fn apply_brief_update(
        &self,
        request: &wenlan_types::BriefUpdateRequest,
    ) -> Result<wenlan_types::BriefUpdateReceipt, WenlanError> {
        validate_update_request(request)?;
        let request_digest = update_request_digest(request)?;
        let _space_write_guard = self.space_write_lock.lock().await;
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("brief update begin: {error}")))?;

        {
            let mut rows = tx
                .query(
                    "SELECT request_digest, response
                       FROM operation_receipts
                      WHERE caller_id=?1 AND operation_id=?2",
                    libsql::params![request.caller_id.as_str(), request.operation_id.as_str()],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("brief operation receipt lookup: {error}"))
                })?;
            if let Some(row) = rows.next().await.map_err(|error| {
                WenlanError::VectorDb(format!("brief operation receipt row: {error}"))
            })? {
                let stored_digest = row.get::<String>(0).unwrap_or_default();
                if stored_digest != request_digest {
                    return Err(WenlanError::Conflict(format!(
                        "operation id '{}' for '{}' was already used with a different request",
                        request.operation_id, request.caller_id
                    )));
                }
                let response = row.get::<String>(1).unwrap_or_default();
                let receipt = serde_json::from_str(&response)?;
                drop(rows);
                tx.commit().await.map_err(|error| {
                    WenlanError::VectorDb(format!("brief replay commit: {error}"))
                })?;
                return Ok(receipt);
            }
        }

        let now = chrono::Utc::now().timestamp();
        let space_id = {
            let existing = {
                let mut rows = tx
                    .query(
                        "SELECT id FROM spaces WHERE name=?1",
                        libsql::params![request.space.trim()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("brief space lookup: {error}"))
                    })?;
                rows.next()
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("brief space row: {error}")))?
                    .map(|row| row.get::<String>(0).unwrap_or_default())
            };
            if let Some(id) = existing {
                id
            } else {
                let id = uuid::Uuid::new_v4().to_string();
                tx.execute(
                    "INSERT INTO spaces
                        (id, name, description, suggested, sort_order, created_at, updated_at)
                     VALUES (
                        ?1, ?2, NULL, 0,
                        (SELECT COALESCE(MAX(sort_order), -1) + 1 FROM spaces),
                        ?3, ?3
                     )",
                    libsql::params![id.as_str(), request.space.trim(), now as f64],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("brief create space: {error}")))?;
                id
            }
        };

        let existing_brief_version = {
            let mut rows = tx
                .query(
                    "SELECT version FROM briefs WHERE space_id=?1",
                    libsql::params![space_id.as_str()],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("brief version lookup: {error}")))?;
            rows.next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("brief version row: {error}")))?
                .map(|row| row.get::<i64>(0).unwrap_or(1))
        };
        let created_brief = existing_brief_version.is_none();
        let starting_brief_version = existing_brief_version.unwrap_or(1);
        if created_brief {
            tx.execute(
                "INSERT INTO briefs
                    (space_id, last_session_summary, last_handoff_at, version, updated_at)
                 VALUES (?1, '', NULL, 1, ?2)",
                libsql::params![space_id.as_str(), now],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("brief create: {error}")))?;
        }

        let mut applied = Vec::new();
        let mut conflicts = Vec::new();
        let mut changed = created_brief;

        if let Some(summary) = &request.summary {
            let version_matches = if created_brief {
                summary.expected_version == 0
            } else {
                summary.expected_version == starting_brief_version
            };
            if version_matches {
                tx.execute(
                    "UPDATE briefs SET last_session_summary=?1, updated_at=?2
                      WHERE space_id=?3",
                    libsql::params![summary.text.trim(), now, space_id.as_str()],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("brief summary update: {error}")))?;
                applied.push(wenlan_types::BriefAppliedMutation {
                    mutation_index: None,
                    kind: wenlan_types::BriefMutationKind::Summary,
                    item_id: None,
                    version: None,
                });
                changed = true;
            } else {
                conflicts.push(wenlan_types::BriefMutationConflict {
                    mutation_index: None,
                    item_id: None,
                    reason: wenlan_types::BriefConflictReason::BriefVersionMismatch,
                    current_version: Some(starting_brief_version),
                });
            }
        }

        for (mutation_index, mutation) in request.mutations.iter().enumerate() {
            match mutation {
                wenlan_types::BriefMutation::Add {
                    text,
                    state,
                    added_at,
                    gate,
                } => {
                    let item_id = format!("brief_item_{}", uuid::Uuid::new_v4().simple());
                    let gate = gate
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty());
                    tx.execute(
                        "INSERT INTO brief_items
                            (id, space_id, text, state, added_at, gate, version, updated_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, ?7)",
                        libsql::params![
                            item_id.as_str(),
                            space_id.as_str(),
                            text.trim(),
                            state.as_str(),
                            added_at.unwrap_or(now),
                            gate,
                            now
                        ],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("brief add item: {error}")))?;
                    applied.push(wenlan_types::BriefAppliedMutation {
                        mutation_index: Some(mutation_index),
                        kind: mutation.kind(),
                        item_id: Some(item_id),
                        version: Some(1),
                    });
                    changed = true;
                }
                wenlan_types::BriefMutation::Edit {
                    item_id,
                    expected_version,
                    text,
                } => {
                    let updated = tx
                        .execute(
                            "UPDATE brief_items
                                SET text=?1, version=version+1, updated_at=?2
                              WHERE space_id=?3 AND id=?4 AND version=?5",
                            libsql::params![
                                text.trim(),
                                now,
                                space_id.as_str(),
                                item_id.as_str(),
                                *expected_version
                            ],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!("brief edit item: {error}"))
                        })?;
                    if updated == 1 {
                        applied.push(wenlan_types::BriefAppliedMutation {
                            mutation_index: Some(mutation_index),
                            kind: mutation.kind(),
                            item_id: Some(item_id.clone()),
                            version: Some(expected_version + 1),
                        });
                        changed = true;
                    } else {
                        conflicts.push(item_conflict(
                            mutation_index,
                            item_id,
                            current_item_version(&tx, &space_id, item_id).await?,
                        ));
                    }
                }
                wenlan_types::BriefMutation::Move {
                    item_id,
                    expected_version,
                    state,
                } => {
                    let updated = tx
                        .execute(
                            "UPDATE brief_items
                                SET state=?1, version=version+1, updated_at=?2
                              WHERE space_id=?3 AND id=?4 AND version=?5",
                            libsql::params![
                                state.as_str(),
                                now,
                                space_id.as_str(),
                                item_id.as_str(),
                                *expected_version
                            ],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!("brief move item: {error}"))
                        })?;
                    if updated == 1 {
                        applied.push(wenlan_types::BriefAppliedMutation {
                            mutation_index: Some(mutation_index),
                            kind: mutation.kind(),
                            item_id: Some(item_id.clone()),
                            version: Some(expected_version + 1),
                        });
                        changed = true;
                    } else {
                        conflicts.push(item_conflict(
                            mutation_index,
                            item_id,
                            current_item_version(&tx, &space_id, item_id).await?,
                        ));
                    }
                }
                wenlan_types::BriefMutation::SetGate {
                    item_id,
                    expected_version,
                    gate,
                } => {
                    let gate = gate
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty());
                    let updated = tx
                        .execute(
                            "UPDATE brief_items
                                SET gate=?1, version=version+1, updated_at=?2
                              WHERE space_id=?3 AND id=?4 AND version=?5",
                            libsql::params![
                                gate,
                                now,
                                space_id.as_str(),
                                item_id.as_str(),
                                *expected_version
                            ],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!("brief gate item: {error}"))
                        })?;
                    if updated == 1 {
                        applied.push(wenlan_types::BriefAppliedMutation {
                            mutation_index: Some(mutation_index),
                            kind: mutation.kind(),
                            item_id: Some(item_id.clone()),
                            version: Some(expected_version + 1),
                        });
                        changed = true;
                    } else {
                        conflicts.push(item_conflict(
                            mutation_index,
                            item_id,
                            current_item_version(&tx, &space_id, item_id).await?,
                        ));
                    }
                }
                wenlan_types::BriefMutation::Complete {
                    item_id,
                    expected_version,
                } => {
                    let deleted = tx
                        .execute(
                            "DELETE FROM brief_items
                              WHERE space_id=?1 AND id=?2 AND version=?3",
                            libsql::params![space_id.as_str(), item_id.as_str(), *expected_version],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!("brief complete item: {error}"))
                        })?;
                    if deleted == 1 {
                        applied.push(wenlan_types::BriefAppliedMutation {
                            mutation_index: Some(mutation_index),
                            kind: mutation.kind(),
                            item_id: Some(item_id.clone()),
                            version: None,
                        });
                        changed = true;
                    } else {
                        conflicts.push(item_conflict(
                            mutation_index,
                            item_id,
                            current_item_version(&tx, &space_id, item_id).await?,
                        ));
                    }
                }
            }
        }

        let final_brief_version = if changed && !created_brief {
            tx.execute(
                "UPDATE briefs
                    SET version=version+1, last_handoff_at=?1, updated_at=?1
                  WHERE space_id=?2",
                libsql::params![now, space_id.as_str()],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("brief version bump: {error}")))?;
            starting_brief_version + 1
        } else {
            if changed {
                tx.execute(
                    "UPDATE briefs SET last_handoff_at=?1, updated_at=?1 WHERE space_id=?2",
                    libsql::params![now, space_id.as_str()],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("brief handoff timestamp: {error}"))
                })?;
            }
            starting_brief_version
        };
        for entry in &mut applied {
            if entry.kind == wenlan_types::BriefMutationKind::Summary {
                entry.version = Some(final_brief_version);
            }
        }

        let receipt = wenlan_types::BriefUpdateReceipt {
            space: request.space.trim().to_string(),
            brief_version: final_brief_version,
            applied,
            conflicts,
            projection_path: None,
            warnings: Vec::new(),
        };
        let response = serde_json::to_string(&receipt)?;
        tx.execute(
            "INSERT INTO operation_receipts
                (caller_id, operation_id, request_digest, response, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                request.caller_id.as_str(),
                request.operation_id.as_str(),
                request_digest.as_str(),
                response,
                now
            ],
        )
        .await
        .map_err(|error| {
            WenlanError::Conflict(format!(
                "operation id '{}' for '{}' was already used: {error}",
                request.operation_id, request.caller_id
            ))
        })?;

        tx.commit()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("brief update commit: {error}")))?;
        Ok(receipt)
    }
}
