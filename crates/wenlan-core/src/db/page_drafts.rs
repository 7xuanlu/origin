// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;
use crate::pages::{Page, PageDraftDeleteOutcome, PageDraftPublishOutcome, PageDraftUpdateOutcome};

const PAGE_COLUMNS: &str = "id, title, summary, content, entity_id, space,
    source_memory_ids, version, status, created_at, last_compiled, last_modified,
    COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0),
    COALESCE(changelog, '[]'), COALESCE(creation_kind, 'distilled'),
    COALESCE(review_status, 'confirmed'), workspace, citations,
    COALESCE(kind, 'concept')";

fn ensure_meaningful_snapshot(title: &str, content: &str) -> Result<(), WenlanError> {
    if title.trim().is_empty() && content.trim().is_empty() {
        return Err(WenlanError::Validation(
            "a Page draft needs a title or body".to_string(),
        ));
    }
    Ok(())
}

fn ensure_meaningful_draft_snapshot(title: &str, content: &str) -> Result<(), WenlanError> {
    let content = crate::export::provenance::validate_canonical_page_content(content)?;
    ensure_meaningful_snapshot(title, content)
}

fn ensure_client_page_draft_id(id: &str) -> Result<(), WenlanError> {
    let Some(uuid_text) = id.strip_prefix("page_") else {
        return Err(WenlanError::Validation(
            "Page draft id must use the page_<uuid-v4> format".to_string(),
        ));
    };
    let uuid = uuid::Uuid::parse_str(uuid_text).map_err(|_| {
        WenlanError::Validation("Page draft id must use the page_<uuid-v4> format".to_string())
    })?;
    if uuid.get_version_num() != 4
        || uuid.get_variant() != uuid::Variant::RFC4122
        || uuid.hyphenated().to_string() != uuid_text
    {
        return Err(WenlanError::Validation(
            "Page draft id must use the page_<uuid-v4> format".to_string(),
        ));
    }
    Ok(())
}

fn ensure_draft(page: &Page) -> Result<(), WenlanError> {
    if page.status != "draft" {
        return Err(WenlanError::Validation(format!(
            "Page {} is not a draft",
            page.id
        )));
    }
    Ok(())
}

impl MemoryDB {
    async fn registered_page_draft_space_on_conn(
        conn: &libsql::Connection,
        requested: Option<&str>,
    ) -> Result<Option<String>, WenlanError> {
        let Some(space) = requested.map(str::trim).filter(|space| !space.is_empty()) else {
            return Ok(None);
        };
        let mut rows = conn
            .query(
                "SELECT 1 FROM spaces WHERE name=?1 LIMIT 1",
                libsql::params![space],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("validate Page draft Space: {error}"))
            })?;
        if rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("validate Page draft Space: {error}")))?
            .is_some()
        {
            Ok(Some(space.to_string()))
        } else {
            Err(WenlanError::Validation(format!(
                "Space {space:?} is not registered"
            )))
        }
    }

    async fn page_draft_on_conn(
        conn: &libsql::Connection,
        id: &str,
    ) -> Result<Option<Page>, WenlanError> {
        let mut rows = conn
            .query(
                &format!("SELECT {PAGE_COLUMNS} FROM pages WHERE id=?1"),
                libsql::params![id],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("load Page draft: {error}")))?;
        match rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("load Page draft row: {error}")))?
        {
            Some(row) => Ok(Some(Self::row_to_page(&row)?)),
            None => Ok(None),
        }
    }

    async fn required_page_draft_on_conn(
        conn: &libsql::Connection,
        id: &str,
    ) -> Result<Page, WenlanError> {
        Self::page_draft_on_conn(conn, id)
            .await?
            .ok_or_else(|| WenlanError::NotFound(format!("Page draft {id}")))
    }

    async fn page_draft_create_request_matches_on_conn(
        conn: &libsql::Connection,
        id: &str,
        title: &str,
        content: &str,
        space: Option<&str>,
        workspace: Option<&str>,
    ) -> Result<bool, WenlanError> {
        let mut rows = conn
            .query(
                "SELECT 1
                   FROM page_draft_create_requests
                  WHERE page_id=?1
                    AND title=?2
                    AND content=?3
                    AND space IS ?4
                    AND workspace IS ?5
                  LIMIT 1",
                libsql::params![id, title, content, space, workspace],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("load Page draft create request: {error}"))
            })?;
        Ok(rows
            .next()
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("load Page draft create request row: {error}"))
            })?
            .is_some())
    }

    async fn page_draft_create_request_exists_on_conn(
        conn: &libsql::Connection,
        id: &str,
    ) -> Result<bool, WenlanError> {
        let mut rows = conn
            .query(
                "SELECT 1 FROM page_draft_create_requests WHERE page_id=?1 LIMIT 1",
                libsql::params![id],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("check Page draft create request: {error}"))
            })?;
        Ok(rows
            .next()
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("check Page draft create request row: {error}"))
            })?
            .is_some())
    }

    /// Create the first durable, meaningful Page draft snapshot.
    pub async fn create_page_draft(
        &self,
        title: &str,
        content: &str,
        space: Option<&str>,
        workspace: Option<&str>,
    ) -> Result<Page, WenlanError> {
        let id = crate::pages::new_page_id();
        self.create_page_draft_with_id(&id, title, content, space, workspace)
            .await
    }

    /// Create a Page draft under a stable client-generated id.
    ///
    /// Replaying the immutable first request is idempotent even when mutable
    /// Page scope has since changed on the server. Reusing the id for any other
    /// request, or for an active Page, is a conflict.
    pub async fn create_page_draft_with_id(
        &self,
        id: &str,
        title: &str,
        content: &str,
        space: Option<&str>,
        workspace: Option<&str>,
    ) -> Result<Page, WenlanError> {
        self.create_page_draft_with_id_impl(id, title, content, space, workspace, false)
            .await
    }

    /// Validate and insert the requested Space in one `Immediate` transaction.
    pub async fn create_page_draft_with_id_in_registered_space(
        &self,
        id: &str,
        title: &str,
        content: &str,
        space: Option<&str>,
    ) -> Result<Page, WenlanError> {
        self.create_page_draft_with_id_impl(id, title, content, space, space, true)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn create_page_draft_with_id_impl(
        &self,
        id: &str,
        title: &str,
        content: &str,
        space: Option<&str>,
        workspace: Option<&str>,
        validate_space: bool,
    ) -> Result<Page, WenlanError> {
        ensure_client_page_draft_id(id)?;
        ensure_meaningful_draft_snapshot(title, content)?;
        let requested_space = if validate_space {
            space
                .map(str::trim)
                .filter(|space| !space.is_empty())
                .map(str::to_string)
        } else {
            space.map(str::to_string)
        };
        let requested_workspace = if validate_space {
            requested_space.clone()
        } else {
            workspace.map(str::to_string)
        };
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("create Page draft begin: {error}")))?;
        if let Some(existing) = Self::page_draft_on_conn(&tx, id).await? {
            if existing.status == "draft"
                && Self::page_draft_create_request_matches_on_conn(
                    &tx,
                    id,
                    title,
                    content,
                    requested_space.as_deref(),
                    requested_workspace.as_deref(),
                )
                .await?
            {
                return Ok(existing);
            }
            return Err(WenlanError::PageDraftIdConflict(id.to_string()));
        }
        if Self::page_draft_create_request_exists_on_conn(&tx, id).await? {
            return Err(WenlanError::PageDraftIdConflict(id.to_string()));
        }
        let normalized_space = if validate_space {
            Self::registered_page_draft_space_on_conn(&tx, requested_space.as_deref()).await?
        } else {
            requested_space
        };
        let normalized_workspace = if validate_space {
            normalized_space.clone()
        } else {
            requested_workspace
        };
        #[cfg(test)]
        if validate_space {
            super::page_drafts_test::transaction_test_hooks::after_space_validation(id).await;
        }

        // M1: the pages scope columns are NOT NULL. Resolve the draft's scope via
        // the Option A ladder (workspace wins, else space, else the reserved
        // sentinel id) and mirror it onto BOTH columns so the read-collapse reads
        // a single honest scope. The create-request ledger below keeps the raw
        // (possibly-None) values so replaying the original request still matches.
        let page_scope = normalized_workspace
            .as_deref()
            .or(normalized_space.as_deref())
            .unwrap_or(super::UNFILED_SPACE_ID);
        tx.execute(
            "INSERT INTO pages (
                    id, title, summary, content, entity_id, space, source_memory_ids,
                    version, status, embedding, created_at, last_compiled,
                    last_modified, sources_updated_count, stale_reason, user_edited,
                    changelog, creation_kind, review_status, workspace, citations, kind
                 ) VALUES (
                    ?1, ?2, NULL, ?3, NULL, ?4, '[]',
                    1, 'draft', NULL, ?5, ?5,
                    ?5, 0, NULL, 1,
                    '[]', 'authored', 'unconfirmed', ?4, '[]', ?6
                 )",
            libsql::params![
                id,
                title,
                content,
                page_scope,
                now,
                crate::pages::page_kind_for(title, "authored", "draft")
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("create Page draft: {error}")))?;
        tx.execute(
            "INSERT INTO page_draft_create_requests (
                page_id, title, content, space, workspace
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                id,
                title,
                content,
                normalized_space.as_deref(),
                normalized_workspace.as_deref()
            ],
        )
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("create Page draft request fingerprint: {error}"))
        })?;
        #[cfg(test)]
        super::page_drafts_test::transaction_test_hooks::after_create_insert(id).await;
        let page = Self::required_page_draft_on_conn(&tx, id).await?;
        tx.commit()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("create Page draft commit: {error}")))?;
        Ok(page)
    }

    /// Replace one complete draft snapshot if the caller still holds its version.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_page_draft(
        &self,
        id: &str,
        expected_version: i64,
        title: &str,
        content: &str,
        space: Option<&str>,
        workspace: Option<&str>,
    ) -> Result<PageDraftUpdateOutcome, WenlanError> {
        self.update_page_draft_impl(
            id,
            expected_version,
            title,
            content,
            space,
            workspace,
            false,
        )
        .await
    }

    /// Recognize an exact ambiguous retry, otherwise preserve version-conflict
    /// precedence, then validate and write the requested Space atomically.
    pub async fn update_page_draft_in_registered_space(
        &self,
        id: &str,
        expected_version: i64,
        title: &str,
        content: &str,
        space: Option<&str>,
    ) -> Result<PageDraftUpdateOutcome, WenlanError> {
        self.update_page_draft_impl(id, expected_version, title, content, space, space, true)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn update_page_draft_impl(
        &self,
        id: &str,
        expected_version: i64,
        title: &str,
        content: &str,
        space: Option<&str>,
        workspace: Option<&str>,
        validate_space: bool,
    ) -> Result<PageDraftUpdateOutcome, WenlanError> {
        ensure_meaningful_draft_snapshot(title, content)?;
        let requested_space = if validate_space {
            space
                .map(str::trim)
                .filter(|space| !space.is_empty())
                .map(str::to_string)
        } else {
            space.map(str::to_string)
        };
        let requested_workspace = if validate_space {
            requested_space.clone()
        } else {
            workspace.map(str::to_string)
        };
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("update Page draft begin: {error}")))?;

        let current = Self::required_page_draft_on_conn(&tx, id).await?;
        if current.status != "draft" {
            // Wire contract: only draft rows are findable as drafts. A queued
            // update racing after publish gets the structured 404, not a 422 —
            // the editor treats "draft not found" as the terminal answer.
            return Err(WenlanError::NotFound(format!("Page draft {id}")));
        }
        // M1 read-collapse: the write below mirrors ONE resolved scope onto both
        // NOT NULL columns via the Option A ladder (workspace wins, else space,
        // else the reserved sentinel id), and `row_to_page` translates that
        // sentinel back to None. An exact retry must therefore compare against
        // that SAME resolved wire scope, not the raw (possibly-divergent)
        // requested columns -- otherwise a divergent-but-idempotent replay
        // (e.g. Some("work"), None, which stores space=workspace="work") misses
        // the fast-path and falls through to a spurious VersionConflict. Both
        // callers agree here: on the registered path requested_space ==
        // requested_workspace, so the ladder is a no-op. The filter drops the
        // sentinel id (not the word "unfiled", which is a legal user scope) so a
        // caller passing it aligns with the wire-hidden `current.space` (None).
        let requested_scope = requested_workspace
            .as_deref()
            .or(requested_space.as_deref())
            .filter(|s| *s != super::UNFILED_SPACE_ID);
        if expected_version.checked_add(1) == Some(current.version)
            && current.title == title
            && current.content == content
            && current.space.as_deref() == requested_scope
        {
            return Ok(PageDraftUpdateOutcome::Updated(current));
        }
        if current.version != expected_version {
            return Ok(PageDraftUpdateOutcome::VersionConflict {
                current_version: current.version,
            });
        }
        let normalized_space = if validate_space {
            Self::registered_page_draft_space_on_conn(&tx, requested_space.as_deref()).await?
        } else {
            requested_space
        };
        let normalized_workspace = if validate_space {
            normalized_space.clone()
        } else {
            requested_workspace
        };
        #[cfg(test)]
        if validate_space {
            super::page_drafts_test::transaction_test_hooks::after_space_validation(id).await;
        }
        // M1: mirror the resolved scope onto both NOT NULL columns via the Option A
        // ladder (workspace wins, else space, else the reserved sentinel id), so
        // an uncategorized draft update writes the sentinel instead of a NULL that
        // the NOT NULL constraint rejects. The idempotency/replay comparison above
        // reads translated (sentinel-hidden) values, so it is unaffected.
        let page_scope = normalized_workspace
            .as_deref()
            .or(normalized_space.as_deref())
            .unwrap_or(super::UNFILED_SPACE_ID);
        let affected = tx
            .execute(
                "UPDATE pages
                     SET title=?1, content=?2, space=?3, workspace=?3,
                         version=version+1, last_modified=?4
                     WHERE id=?5 AND status='draft' AND version=?6",
                libsql::params![title, content, page_scope, now, id, expected_version],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("update Page draft row: {error}")))?;
        if affected != 1 {
            return Err(WenlanError::Conflict(format!(
                "Page draft {id} changed during update"
            )));
        }
        let outcome =
            PageDraftUpdateOutcome::Updated(Self::required_page_draft_on_conn(&tx, id).await?);
        tx.commit()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("update Page draft commit: {error}")))?;
        Ok(outcome)
    }

    /// Delete a draft only when the version supplied by the client is current.
    pub async fn delete_page_draft(
        &self,
        id: &str,
        expected_version: i64,
    ) -> Result<PageDraftDeleteOutcome, WenlanError> {
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("delete Page draft begin: {error}")))?;
        let current = Self::required_page_draft_on_conn(&tx, id).await?;
        if current.status != "draft" {
            // Same wire contract as update: a discard racing after publish gets
            // the structured 404, which the editor treats as completed cleanup.
            return Err(WenlanError::NotFound(format!("Page draft {id}")));
        }
        if current.version != expected_version {
            return Ok(PageDraftDeleteOutcome::VersionConflict {
                current_version: current.version,
            });
        }
        let affected = tx
            .execute(
                "DELETE FROM pages WHERE id=?1 AND status='draft' AND version=?2",
                libsql::params![id, expected_version],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("delete Page draft row: {error}")))?;
        if affected != 1 {
            return Err(WenlanError::Conflict(format!(
                "Page draft {id} changed during delete"
            )));
        }
        tx.execute(
            "INSERT INTO page_draft_create_requests (page_id)
             VALUES (?1)
             ON CONFLICT(page_id) DO UPDATE SET
                title=NULL,
                content=NULL,
                space=NULL,
                workspace=NULL",
            libsql::params![id],
        )
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("scrub Page draft create request: {error}"))
        })?;
        tx.commit()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("delete Page draft commit: {error}")))?;
        Ok(PageDraftDeleteOutcome::Deleted)
    }

    /// Publish a draft as an active Page if the caller still holds its version.
    ///
    /// Editor contract (mirrored by `e2e/tauriMock/runtime.ts`): an exact retry
    /// of a publish that already landed — the row is active at
    /// `expected_version + 1` — replays the published Page; any other version
    /// mismatch is a `VersionConflict`; an active Page in the same scope whose
    /// trimmed title matches case-insensitively blocks with `TitleConflict`.
    /// Publishing requires both a trimmed title and non-empty content, stamps
    /// the trimmed title, flips `status` to active, re-derives `kind` from the
    /// one shared rule, bumps the version, and stamps `last_compiled` /
    /// `last_modified`. The page embedding is computed best-effort like every
    /// other page insert; the FTS reindex is the `pages_fts_update` trigger's
    /// job.
    pub async fn publish_page_draft(
        &self,
        id: &str,
        expected_version: i64,
    ) -> Result<PageDraftPublishOutcome, WenlanError> {
        // Snapshot outside the write transaction so the embedding compute
        // (slow on first call while the model loads) never holds it. Any
        // concurrent draft write bumps the version, so if the in-transaction
        // recheck still sees `expected_version` the snapshot's title and
        // content are exactly what gets published.
        let snapshot = {
            let conn = self.conn.lock().await;
            Self::required_page_draft_on_conn(&conn, id).await?
        };
        let embedding_sql = if snapshot.status == "draft" && snapshot.version == expected_version {
            let embed_text = crate::pages::page_embedding_text(
                snapshot.title.trim(),
                snapshot.summary.as_deref(),
                &snapshot.content,
            );
            match self.generate_embeddings(&[embed_text]) {
                Ok(vecs) if !vecs.is_empty() => Some(Self::vec_to_sql(&vecs[0])),
                Ok(_) => {
                    log::warn!("publish_page_draft: empty embedding result for {id}");
                    None
                }
                Err(e) => {
                    log::warn!("publish_page_draft: embedding failed for {id}: {e}");
                    None
                }
            }
        } else {
            None
        };

        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("publish Page draft begin: {error}")))?;
        let current = Self::required_page_draft_on_conn(&tx, id).await?;
        if current.status == "active" && expected_version.checked_add(1) == Some(current.version) {
            return Ok(PageDraftPublishOutcome::Published(current));
        }
        if current.version != expected_version {
            return Ok(PageDraftPublishOutcome::VersionConflict {
                current_version: current.version,
            });
        }
        ensure_draft(&current)?;
        let title = current.title.trim().to_string();
        if title.is_empty() || current.content.trim().is_empty() {
            return Err(WenlanError::Validation(
                "Title and content are required".to_string(),
            ));
        }
        // Same-scope title uniqueness among active Pages, compared on the
        // stored (sentinel-mirrored) scope column so unfiled matches unfiled.
        // The case fold runs in Rust: the bundled SQLite lower() is ASCII-only
        // (no ICU), while the editor contract folds Unicode titles too — the
        // same reason migration 31 re-ran canonicalization in Rust.
        let scope = current
            .space
            .clone()
            .unwrap_or_else(|| super::UNFILED_SPACE_ID.to_string());
        let wanted = title.to_lowercase();
        let mut rows = tx
            .query(
                "SELECT id, title FROM pages
                 WHERE id<>?1 AND status='active' AND space=?2",
                libsql::params![id, scope.as_str()],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("publish Page draft title check: {error}"))
            })?;
        let mut conflict: Option<(String, String)> = None;
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("publish Page draft title row: {error}"))
        })? {
            let existing_page_id: String = row
                .get(0)
                .map_err(|error| WenlanError::VectorDb(format!("title conflict id: {error}")))?;
            let existing_page_title: String = row
                .get(1)
                .map_err(|error| WenlanError::VectorDb(format!("title conflict title: {error}")))?;
            if existing_page_title.trim().to_lowercase() == wanted {
                conflict = Some((existing_page_id, existing_page_title));
                break;
            }
        }
        drop(rows);
        if let Some((existing_page_id, existing_page_title)) = conflict {
            return Ok(PageDraftPublishOutcome::TitleConflict {
                existing_page_id,
                existing_page_title,
            });
        }

        let kind = crate::pages::page_kind_for(&title, &current.creation_kind, "active");
        let publish = match &embedding_sql {
            Some(emb) => {
                tx.execute(
                    "UPDATE pages
                     SET title=?1, status='active', kind=?2, version=version+1,
                         last_compiled=?3, last_modified=?3, embedding=vector32(?4)
                     WHERE id=?5 AND status='draft' AND version=?6",
                    libsql::params![
                        title.as_str(),
                        kind,
                        now.as_str(),
                        emb.as_str(),
                        id,
                        expected_version
                    ],
                )
                .await
            }
            None => {
                tx.execute(
                    "UPDATE pages
                     SET title=?1, status='active', kind=?2, version=version+1,
                         last_compiled=?3, last_modified=?3
                     WHERE id=?4 AND status='draft' AND version=?5",
                    libsql::params![title.as_str(), kind, now.as_str(), id, expected_version],
                )
                .await
            }
        };
        let affected = publish
            .map_err(|error| WenlanError::VectorDb(format!("publish Page draft row: {error}")))?;
        if affected != 1 {
            return Err(WenlanError::Conflict(format!(
                "Page draft {id} changed during publish"
            )));
        }
        // Every path that bumps `pages.version` appends its immutable history
        // row in the same transaction (`append_page_history` is the ONE site).
        Self::append_page_history(&tx, id, "publish", chrono::Utc::now().timestamp())
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("publish Page draft history: {error}"))
            })?;
        let outcome =
            PageDraftPublishOutcome::Published(Self::required_page_draft_on_conn(&tx, id).await?);
        tx.commit().await.map_err(|error| {
            WenlanError::VectorDb(format!("publish Page draft commit: {error}"))
        })?;
        drop(conn);
        if let PageDraftPublishOutcome::Published(page) = &outcome {
            // Wikilink upkeep, the same best-effort post-write as insert_page /
            // update_page_content: a stale link index is recoverable on the
            // next save and must not fail a durable publish.
            if let Err(e) = self.refresh_page_wikilinks(&page.id, &page.content).await {
                log::warn!(
                    "[publish_page_draft] wikilink refresh failed for {}: {e}",
                    page.id
                );
            }
            if let Err(e) = self.resolve_orphan_page_links().await {
                log::warn!("[publish_page_draft] orphan link resolve failed: {e}");
            }
        }
        Ok(outcome)
    }
}
