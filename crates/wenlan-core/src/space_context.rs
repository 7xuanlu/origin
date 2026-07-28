// SPDX-License-Identifier: Apache-2.0

use crate::db::{MemoryDB, UNFILED_SPACE_ID};
use crate::WenlanError;
use std::path::Path;
use wenlan_types::{Space, WriteSpaceSource, WriteSpaceTarget};

const LEGACY_DEFAULT_WATERMARK: &str = "legacy_default_imported";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedWriteSpace {
    pub space_id: Option<String>,
    pub space_name: Option<String>,
    pub source: WriteSpaceSource,
}

impl ResolvedWriteSpace {
    pub fn uncategorized() -> Self {
        Self {
            space_id: None,
            space_name: None,
            source: WriteSpaceSource::Uncategorized,
        }
    }
}

#[derive(Debug)]
pub struct LegacyDefaultImport {
    pub already_processed: bool,
    pub imported_space: Option<Space>,
    pub invalid_name: Option<String>,
}

impl MemoryDB {
    async fn get_space_by_id(&self, id: &str) -> Result<Option<Space>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT s.id, s.name, s.description, s.suggested, s.created_at, s.updated_at,
                        (SELECT COUNT(DISTINCT c.source_id) FROM memories c WHERE c.space = s.name AND c.source = 'memory' AND c.pending_revision = 0 AND COALESCE(c.is_recap, 0) = 0 AND c.source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)) as mem_count,
                        (SELECT COUNT(*) FROM entities e WHERE e.space = s.name) as ent_count,
                        s.sort_order, s.starred, s.is_default
                 FROM spaces s
                 WHERE s.id = ?1 AND s.id != ?2",
                libsql::params![id, UNFILED_SPACE_ID],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("get_space_by_id: {e}")))?;

        let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("get_space_by_id row: {e}")))?
        else {
            return Ok(None);
        };

        Ok(Some(Space {
            id: row.get::<String>(0).unwrap_or_default(),
            name: row.get::<String>(1).unwrap_or_default(),
            description: row.get::<Option<String>>(2).unwrap_or(None),
            suggested: row.get::<i32>(3).unwrap_or(0) != 0,
            created_at: row.get::<f64>(4).unwrap_or(0.0),
            updated_at: row.get::<f64>(5).unwrap_or(0.0),
            memory_count: row.get::<u64>(6).unwrap_or(0),
            entity_count: row.get::<u64>(7).unwrap_or(0),
            sort_order: row.get::<i64>(8).unwrap_or(0),
            starred: row.get::<i32>(9).unwrap_or(0) != 0,
            is_default: row.get::<i32>(10).unwrap_or(0) != 0,
        }))
    }

    pub async fn get_default_space(&self) -> Result<Option<Space>, WenlanError> {
        let id = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT id FROM spaces
                     WHERE is_default = 1 AND id != ?1
                     LIMIT 1",
                    libsql::params![UNFILED_SPACE_ID],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("get_default_space: {e}")))?;
            let id = if let Some(row) = rows
                .next()
                .await
                .map_err(|e| WenlanError::VectorDb(format!("get_default_space row: {e}")))?
            {
                Some(
                    row.get::<String>(0)
                        .map_err(|e| WenlanError::VectorDb(format!("get_default_space id: {e}")))?,
                )
            } else {
                None
            };
            id
        };

        match id {
            Some(id) => self.get_space_by_id(&id).await,
            None => Ok(None),
        }
    }

    pub async fn set_default_space(&self, space_id: &str) -> Result<Space, WenlanError> {
        if space_id == UNFILED_SPACE_ID {
            return Err(WenlanError::Validation(
                "Uncategorized cannot be the Default save space".to_string(),
            ));
        }

        let _space_write_guard = self.lock_space_writes().await;
        {
            let conn = self.conn.lock().await;
            let tx = conn
                .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
                .await
                .map_err(|e| WenlanError::VectorDb(format!("set_default_space begin: {e}")))?;

            let result: Result<(), WenlanError> = async {
                let mut rows = tx
                    .query(
                        "SELECT name FROM spaces WHERE id = ?1 AND id != ?2",
                        libsql::params![space_id, UNFILED_SPACE_ID],
                    )
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("set_default_space lookup: {e}")))?;
                rows.next()
                    .await
                    .map_err(|e| {
                        WenlanError::VectorDb(format!("set_default_space lookup row: {e}"))
                    })?
                    .map(|row| row.get::<String>(0))
                    .transpose()
                    .map_err(|e| WenlanError::VectorDb(format!("set_default_space name: {e}")))?
                    .ok_or_else(|| {
                        WenlanError::Validation(format!(
                            "Default save space id '{space_id}' is not registered"
                        ))
                    })?;
                drop(rows);

                tx.execute("UPDATE spaces SET is_default = 0 WHERE is_default != 0", ())
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("set_default_space clear: {e}")))?;
                let changed = tx
                    .execute(
                        "UPDATE spaces SET is_default = 1 WHERE id = ?1 AND id != ?2",
                        libsql::params![space_id, UNFILED_SPACE_ID],
                    )
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("set_default_space select: {e}")))?;
                if changed != 1 {
                    return Err(WenlanError::Validation(format!(
                        "Default save space id '{space_id}' is not registered"
                    )));
                }
                Ok(())
            }
            .await;

            match result {
                Ok(()) => {
                    tx.commit().await.map_err(|e| {
                        WenlanError::VectorDb(format!("set_default_space commit: {e}"))
                    })?;
                }
                Err(error) => {
                    let _ = tx.rollback().await;
                    return Err(error);
                }
            }
        }

        self.get_space_by_id(space_id)
            .await?
            .ok_or_else(|| WenlanError::VectorDb("default Space disappeared after commit".into()))
    }

    pub async fn clear_default_space(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("UPDATE spaces SET is_default = 0 WHERE is_default != 0", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("clear_default_space: {e}")))?;
        Ok(())
    }

    pub async fn resolve_write_space(
        &self,
        target: &WriteSpaceTarget,
        header_space: Option<&str>,
    ) -> Result<ResolvedWriteSpace, WenlanError> {
        match target {
            WriteSpaceTarget::Uncategorized => return Ok(ResolvedWriteSpace::uncategorized()),
            WriteSpaceTarget::Named(name) => {
                return self
                    .resolve_registered_write_space(name, WriteSpaceSource::Request)
                    .await;
            }
            WriteSpaceTarget::Inherit => {}
        }

        if let Some(header) = header_space.map(str::trim).filter(|name| !name.is_empty()) {
            return self
                .resolve_registered_write_space(header, WriteSpaceSource::Header)
                .await;
        }

        if let Some(space) = self.get_default_space().await? {
            return Ok(ResolvedWriteSpace {
                space_id: Some(space.id),
                space_name: Some(space.name),
                source: WriteSpaceSource::Default,
            });
        }

        Ok(ResolvedWriteSpace::uncategorized())
    }

    async fn resolve_registered_write_space(
        &self,
        name: &str,
        source: WriteSpaceSource,
    ) -> Result<ResolvedWriteSpace, WenlanError> {
        let normalized = name.trim();
        if normalized.is_empty() {
            return Err(WenlanError::Validation(
                "Space name must not be empty".to_string(),
            ));
        }
        let space = self.get_space(normalized).await?.ok_or_else(|| {
            WenlanError::Validation(format!(
                "Space '{normalized}' is not registered; create it first or choose Uncategorized"
            ))
        })?;
        if space.id == UNFILED_SPACE_ID {
            return Err(WenlanError::Validation(
                "Uncategorized must be selected with an explicit null Space".to_string(),
            ));
        }
        Ok(ResolvedWriteSpace {
            space_id: Some(space.id),
            space_name: Some(space.name),
            source,
        })
    }

    pub async fn finalize_write_space(
        &self,
        resolved: &ResolvedWriteSpace,
    ) -> Result<ResolvedWriteSpace, WenlanError> {
        let Some(space_id) = resolved.space_id.as_deref() else {
            return Ok(ResolvedWriteSpace::uncategorized());
        };

        if let Some(space) = self.get_space_by_id(space_id).await? {
            return Ok(ResolvedWriteSpace {
                space_id: Some(space.id),
                space_name: Some(space.name),
                source: resolved.source,
            });
        }

        if resolved.source == WriteSpaceSource::Default {
            Ok(ResolvedWriteSpace::uncategorized())
        } else {
            Err(WenlanError::Validation(format!(
                "Selected Space no longer exists (id '{space_id}'); choose another destination"
            )))
        }
    }

    pub async fn import_legacy_default_once(
        &self,
        path: &Path,
    ) -> Result<LegacyDefaultImport, WenlanError> {
        {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT value FROM app_metadata WHERE key = ?1",
                    libsql::params![LEGACY_DEFAULT_WATERMARK],
                )
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("legacy default watermark lookup: {e}"))
                })?;
            if rows
                .next()
                .await
                .map_err(|e| WenlanError::VectorDb(format!("legacy default watermark row: {e}")))?
                .is_some()
            {
                return Ok(LegacyDefaultImport {
                    already_processed: true,
                    imported_space: None,
                    invalid_name: None,
                });
            }
        }

        let existing_default = self.get_default_space().await?;
        let legacy_name = if existing_default.is_some() {
            None
        } else {
            read_legacy_default_name(path)?
        };

        let (imported_id, invalid_name) = {
            let conn = self.conn.lock().await;
            let tx = conn
                .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
                .await
                .map_err(|e| WenlanError::VectorDb(format!("legacy default import begin: {e}")))?;

            let result: Result<(Option<String>, Option<String>), WenlanError> = async {
                let mut watermark_rows = tx
                    .query(
                        "SELECT value FROM app_metadata WHERE key = ?1",
                        libsql::params![LEGACY_DEFAULT_WATERMARK],
                    )
                    .await
                    .map_err(|e| {
                        WenlanError::VectorDb(format!("legacy default watermark recheck: {e}"))
                    })?;
                if watermark_rows
                    .next()
                    .await
                    .map_err(|e| {
                        WenlanError::VectorDb(format!("legacy default watermark recheck row: {e}"))
                    })?
                    .is_some()
                {
                    return Ok((None, None));
                }
                drop(watermark_rows);

                let mut default_rows = tx
                    .query(
                        "SELECT id FROM spaces
                         WHERE is_default = 1 AND id != ?1
                         LIMIT 1",
                        libsql::params![UNFILED_SPACE_ID],
                    )
                    .await
                    .map_err(|e| {
                        WenlanError::VectorDb(format!("legacy default current lookup: {e}"))
                    })?;
                let has_default = default_rows
                    .next()
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("legacy default current row: {e}")))?
                    .is_some();
                drop(default_rows);

                let mut imported_id = None;
                let mut invalid_name = None;
                if !has_default {
                    if let Some(name) = legacy_name.as_deref() {
                        let mut rows = tx
                            .query(
                                "SELECT id FROM spaces
                                 WHERE name = ?1 AND id != ?2",
                                libsql::params![name, UNFILED_SPACE_ID],
                            )
                            .await
                            .map_err(|e| {
                                WenlanError::VectorDb(format!("legacy default Space lookup: {e}"))
                            })?;
                        let matched_id = rows
                            .next()
                            .await
                            .map_err(|e| {
                                WenlanError::VectorDb(format!(
                                    "legacy default Space lookup row: {e}"
                                ))
                            })?
                            .map(|row| row.get::<String>(0))
                            .transpose()
                            .map_err(|e| {
                                WenlanError::VectorDb(format!("legacy default Space id: {e}"))
                            })?;
                        drop(rows);

                        if let Some(space_id) = matched_id {
                            tx.execute(
                                "UPDATE spaces SET is_default = 1 WHERE id = ?1",
                                libsql::params![space_id.clone()],
                            )
                            .await
                            .map_err(|e| {
                                WenlanError::VectorDb(format!("legacy default Space select: {e}"))
                            })?;
                            imported_id = Some(space_id);
                        } else {
                            invalid_name = Some(name.to_string());
                        }
                    }
                }

                tx.execute(
                    "INSERT INTO app_metadata (key, value) VALUES (?1, '1')",
                    libsql::params![LEGACY_DEFAULT_WATERMARK],
                )
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("legacy default watermark insert: {e}"))
                })?;
                Ok((imported_id, invalid_name))
            }
            .await;

            match result {
                Ok(outcome) => {
                    tx.commit().await.map_err(|e| {
                        WenlanError::VectorDb(format!("legacy default import commit: {e}"))
                    })?;
                    outcome
                }
                Err(error) => {
                    let _ = tx.rollback().await;
                    return Err(error);
                }
            }
        };

        let imported_space = match imported_id {
            Some(id) => self.get_space_by_id(&id).await?,
            None => None,
        };
        Ok(LegacyDefaultImport {
            already_processed: false,
            imported_space,
            invalid_name,
        })
    }
}

fn read_legacy_default_name(path: &Path) -> Result<Option<String>, WenlanError> {
    let body = match std::fs::read_to_string(path) {
        Ok(body) => body,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(WenlanError::Io(error)),
    };
    let parsed: toml::Value = match toml::from_str(&body) {
        Ok(value) => value,
        Err(error) => {
            log::warn!(
                "[space_context] could not parse legacy {}: {error}; leaving Default save space unset",
                path.display()
            );
            return Ok(None);
        }
    };
    Ok(parsed
        .get("default")
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string))
}

#[cfg(test)]
mod tests {
    use crate::db::tests::test_db;
    use crate::db::UNFILED_SPACE_ID;
    use crate::events::NoopEmitter;
    use crate::WenlanError;
    use wenlan_types::{WriteSpaceSource, WriteSpaceTarget};

    #[tokio::test]
    async fn default_space_set_replace_rename_delete_lifecycle() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();
        let personal = db.create_space("personal", None, false).await.unwrap();

        let sentinel_error = db.set_default_space(UNFILED_SPACE_ID).await.unwrap_err();
        assert!(matches!(sentinel_error, WenlanError::Validation(_)));

        let selected = db.set_default_space(&work.id).await.unwrap();
        assert_eq!(selected.id, work.id);
        assert!(selected.is_default);
        assert_eq!(
            db.list_spaces()
                .await
                .unwrap()
                .into_iter()
                .filter(|space| space.is_default)
                .count(),
            1
        );

        db.update_space("work", "career", None).await.unwrap();
        let renamed = db.get_default_space().await.unwrap().unwrap();
        assert_eq!(renamed.id, work.id);
        assert_eq!(renamed.name, "career");

        db.set_default_space(&personal.id).await.unwrap();
        assert_eq!(
            db.get_default_space().await.unwrap().unwrap().id,
            personal.id
        );
        assert!(!db.get_space("career").await.unwrap().unwrap().is_default);

        db.delete_space("personal", "keep").await.unwrap();
        assert!(db.get_default_space().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn legacy_default_import_runs_once_without_rewriting_toml() {
        let (db, dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();
        let path = dir.path().join("spaces.toml");
        let original =
            b"default = \"work\"\n\n[[mapping]]\npath = \"/tmp/work\"\nspace = \"work\"\n";
        std::fs::write(&path, original).unwrap();

        let imported = db.import_legacy_default_once(&path).await.unwrap();
        assert!(!imported.already_processed);
        assert_eq!(
            imported
                .imported_space
                .as_ref()
                .map(|space| space.id.as_str()),
            Some(work.id.as_str())
        );
        assert_eq!(std::fs::read(&path).unwrap(), original);

        db.clear_default_space().await.unwrap();
        std::fs::write(&path, b"default = \"personal\"\n").unwrap();
        let second = db.import_legacy_default_once(&path).await.unwrap();
        assert!(second.already_processed);
        assert!(second.imported_space.is_none());
        assert!(db.get_default_space().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn write_space_resolution_carries_stable_id_through_rename() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();

        let resolved = db
            .resolve_write_space(&WriteSpaceTarget::Named("work".into()), None)
            .await
            .unwrap();
        assert_eq!(resolved.space_id.as_deref(), Some(work.id.as_str()));
        assert_eq!(resolved.source, WriteSpaceSource::Request);

        db.update_space("work", "career", None).await.unwrap();
        let finalized = db.finalize_write_space(&resolved).await.unwrap();
        assert_eq!(finalized.space_id, resolved.space_id);
        assert_eq!(finalized.space_name.as_deref(), Some("career"));
        assert_eq!(finalized.source, WriteSpaceSource::Request);

        let explicit = db
            .resolve_write_space(&WriteSpaceTarget::Named("career".into()), None)
            .await
            .unwrap();
        db.delete_space("career", "keep").await.unwrap();
        assert!(matches!(
            db.finalize_write_space(&explicit).await,
            Err(WenlanError::Validation(_))
        ));
    }

    #[tokio::test]
    async fn deleted_default_falls_back_to_uncategorized_at_finalization() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&NoopEmitter).await.unwrap();
        let work = db.create_space("work", None, false).await.unwrap();
        db.set_default_space(&work.id).await.unwrap();

        let resolved = db
            .resolve_write_space(&WriteSpaceTarget::Inherit, None)
            .await
            .unwrap();
        assert_eq!(resolved.source, WriteSpaceSource::Default);

        db.delete_space("work", "keep").await.unwrap();
        let finalized = db.finalize_write_space(&resolved).await.unwrap();
        assert_eq!(finalized.space_id, None);
        assert_eq!(finalized.space_name, None);
        assert_eq!(finalized.source, WriteSpaceSource::Uncategorized);
    }
}
