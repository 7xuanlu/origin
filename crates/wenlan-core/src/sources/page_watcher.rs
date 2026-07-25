// SPDX-License-Identifier: Apache-2.0
//! Poll-based filesystem watcher for the page projection at
//! `~/.wenlan/pages/*.md`.
//!
//! md is canonical. When the user opens a page in Obsidian / VS Code / etc.
//! and saves a change, the daemon's DB row goes stale. This module walks the
//! projection on a scheduler tick, compares each file against its DB
//! counterpart, and applies the edit back through `update_page_content`
//! with `link_reason = "fs_edit"`. That flips `user_edited` (same effect
//! as a `manual_edit` write via POST /api/pages/{id}) so the refinery's
//! re-distill escalation branch fires instead of clobbering the user's
//! prose.
//!
//! Conflict direction: if the md's `origin_version` is less than the DB's,
//! the daemon wrote last and the md on disk is stale (probably from a
//! refinery cycle that hasn't re-projected yet). We skip rather than
//! roll the DB back. The reverse (md ahead, DB behind) doesn't happen in
//! practice — the daemon is the only writer that bumps the version
//! column.
//!
//! v1 is poll-only, runs on the same cadence as the refinery scheduler
//! tick. An event-driven `notify` watcher is a follow-up if 30-second
//! latency proves too slow.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::sources::obsidian;
use std::path::{Path, PathBuf};

/// Counts the watcher reports back to the scheduler tick. Useful for the
/// daily-status feed once the UI surface lands; for now the daemon just
/// logs `applied` so an operator can see edits flowing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WatcherStats {
    /// Total .md files inspected this pass.
    pub scanned: usize,
    /// Edits successfully written back to the DB.
    pub applied: usize,
    /// File had no `origin_id` in the frontmatter — most likely a user-
    /// added md the daemon hasn't claimed. Skipped, not auto-promoted.
    pub skipped_no_origin_id: usize,
    /// Frontmatter named a page_id the DB doesn't know. Dangling md from
    /// an archived/deleted page or a manually-pasted file.
    pub skipped_unknown_page: usize,
    /// md version stamp is behind the DB. Daemon wrote last; md will get
    /// re-projected on the next daemon write, no need to reflect it back.
    pub skipped_daemon_ahead: usize,
    /// md body matches DB content. No-op.
    pub skipped_unchanged: usize,
    /// IO or parse failures — surfaced via `log::warn!` per file.
    pub errors: usize,
}

/// Scan `knowledge_path` for page md files, reflect external edits back
/// into the DB. Idempotent and safe to call on every scheduler tick.
pub async fn sync_filesystem_edits(
    db: &MemoryDB,
    knowledge_path: &Path,
) -> Result<WatcherStats, WenlanError> {
    let mut stats = WatcherStats::default();
    if !knowledge_path.exists() {
        return Ok(stats);
    }
    let entries = match std::fs::read_dir(knowledge_path) {
        Ok(e) => e,
        Err(e) => {
            log::warn!(
                "[page-watcher] read_dir({}) failed: {e}",
                knowledge_path.display()
            );
            return Ok(stats);
        }
    };
    let knowledge_path_buf: PathBuf = knowledge_path.to_path_buf();
    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        // Pages live flat under knowledge_path. Subdirectories (e.g. the
        // `.wenlan/` state dir) aren't recursed.
        if !path.is_file() || ext != "md" {
            continue;
        }
        stats.scanned += 1;
        match sync_one_file(db, &path, &knowledge_path_buf).await {
            Ok(Outcome::Applied { page_id }) => {
                stats.applied += 1;
                log::info!(
                    "[page-watcher] applied fs_edit for {page_id} from {}",
                    path.display()
                );
            }
            Ok(Outcome::SkippedNoOriginId) => stats.skipped_no_origin_id += 1,
            Ok(Outcome::SkippedUnknownPage { page_id }) => {
                stats.skipped_unknown_page += 1;
                log::debug!(
                    "[page-watcher] {} references unknown page {page_id}",
                    path.display()
                );
            }
            Ok(Outcome::SkippedDaemonAhead { page_id }) => {
                stats.skipped_daemon_ahead += 1;
                log::debug!("[page-watcher] {page_id}: md trails DB version, skipping");
            }
            Ok(Outcome::Unchanged) => stats.skipped_unchanged += 1,
            Err(e) => {
                stats.errors += 1;
                log::warn!("[page-watcher] {}: {e}", path.display());
            }
        }
    }
    Ok(stats)
}

enum Outcome {
    Applied { page_id: String },
    SkippedNoOriginId,
    SkippedUnknownPage { page_id: String },
    SkippedDaemonAhead { page_id: String },
    Unchanged,
}

async fn sync_one_file(
    db: &MemoryDB,
    path: &Path,
    knowledge_path: &Path,
) -> Result<Outcome, WenlanError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| WenlanError::VectorDb(format!("read {}: {e}", path.display())))?;
    let (fm, body) = obsidian::extract_frontmatter(&raw);
    let page_id = match fm.get_str("origin_id") {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return Ok(Outcome::SkippedNoOriginId),
    };

    let existing = match db.get_page(&page_id).await? {
        Some(p) => p,
        None => return Ok(Outcome::SkippedUnknownPage { page_id }),
    };

    // Skip when md is stale relative to DB. `origin_version` in the
    // frontmatter is written by KnowledgeWriter::render_markdown and
    // tracks the DB's version column at projection time. If the daemon
    // bumped version without re-projecting (e.g. refinery write without
    // KnowledgeWriter call), the md is behind — we'd otherwise roll the
    // DB back to a stale body.
    // Shared with the startup reconcile, which repairs exactly the files this
    // branch declines to touch — they must read the stamp identically or
    // reconcile would overwrite what this skip protects.
    let md_version = crate::export::knowledge::projected_origin_version(&fm);
    if md_version < existing.version {
        return Ok(Outcome::SkippedDaemonAhead { page_id });
    }

    // Strip the export-only delimiter Sources block from BOTH sides before
    // the diff. The projected file always carries the generated block while
    // Page.content never does; a raw compare would mark every page
    // user_edited on the first tick and lock the refinery out.
    let body_norm = crate::export::provenance::canonicalize_page_body(body);
    let db_norm = crate::export::provenance::canonicalize_page_body(&existing.content);
    if body_norm == db_norm {
        // Canonicalized prose is identical, but the raw bytes may differ
        // (user touched the protected Sources block). If so, re-project from
        // DB truth so the file stops lying about its sources; otherwise it
        // really is unchanged.
        let raw_body = body.trim_end_matches('\n');
        let fresh = crate::export::knowledge::render_markdown_for(&existing);
        let (_, fresh_body) = obsidian::extract_frontmatter(&fresh);
        if raw_body != fresh_body.trim_end_matches('\n') {
            let writer =
                crate::export::knowledge::KnowledgeWriter::new(knowledge_path.to_path_buf(), db);
            // Only re-project if the writer already maps this page to a file (state
            // entry present) — otherwise write_page's unique_filename would fork a
            // `<slug>-2.md` duplicate against the on-disk file we're reading (real on a
            // vault synced without `.wenlan/state.json`). Without state we can't safely
            // pick the canonical file; leave it as-is until a genuine re-distill
            // re-establishes the mapping. The page row is untouched either way.
            if writer.page_filename(&existing.id).is_some() {
                writer.begin_projection_write().write_page(&existing)?;
            }
        }
        return Ok(Outcome::Unchanged);
    }

    // Preserve existing source list — the user edited prose, not the
    // memory provenance. Sources change only via /distill refresh or
    // explicit POST.
    let req = wenlan_types::requests::UpdatePageRequest {
        content: body_norm,
        source_memory_ids: existing.source_memory_ids.clone(),
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    // Pass knowledge_path so update_page re-projects the md with the new
    // version stamp; without that the next tick would see origin_version
    // trailing the DB and skip as SkippedDaemonAhead.
    // require_stale=false: user edits are unconditional.
    // knowledge_path=Some: page_watcher IS the fs writer; update_page
    //   re-projects rather than skipping the write.
    crate::post_write::update_page(
        db,
        &page_id,
        req,
        "fs_edit",
        false,
        Some(knowledge_path),
        None,
    )
    .await?;

    Ok(Outcome::Applied { page_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MemoryDB;
    use crate::events::NoopEmitter;
    use crate::export::knowledge::KnowledgeWriter;
    use crate::pages::Page;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn fresh_db() -> (MemoryDB, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let emitter: Arc<dyn crate::events::EventEmitter> = Arc::new(NoopEmitter);
        let db = MemoryDB::new(&path, emitter).await.unwrap();
        (db, dir)
    }

    fn tracked_writer(db: &MemoryDB, path: &Path) -> KnowledgeWriter {
        KnowledgeWriter::new(path.to_path_buf(), db)
    }

    fn project_page(db: &MemoryDB, writer: &KnowledgeWriter, page: &Page) -> String {
        let guard = db.begin_page_projection_write();
        writer.write_page(&guard, page).unwrap()
    }

    fn write_page_md(dir: &std::path::Path, page: &Page, body_override: Option<&str>) {
        let body = body_override.unwrap_or(&page.content);
        let domain_line = match &page.space {
            Some(d) => format!("space: {d}\n"),
            None => String::new(),
        };
        let content = format!(
            "---\ntitle: \"{}\"\n{}origin_id: {}\norigin_version: {}\ncreated: {}\nmodified: {}\n---\n\n{}\n",
            page.title,
            domain_line,
            page.id,
            page.version,
            page.created_at.chars().take(10).collect::<String>(),
            page.last_modified.chars().take(10).collect::<String>(),
            body
        );
        std::fs::write(dir.join(format!("{}.md", slug(&page.title))), content).unwrap();
    }

    fn slug(s: &str) -> String {
        s.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect()
    }

    fn sample_page(id: &str, title: &str, content: &str) -> Page {
        let now = chrono::Utc::now().to_rfc3339();
        Page {
            id: id.to_string(),
            title: title.to_string(),
            summary: None,
            content: content.to_string(),
            entity_id: None,
            space: None,
            source_memory_ids: vec!["mem_seed".to_string()],
            version: 1,
            status: "active".to_string(),
            created_at: now.clone(),
            last_compiled: now.clone(),
            last_modified: now,
            sources_updated_count: 0,
            stale_reason: None,
            pending_rebuild: None,
            user_edited: false,
            relevance_score: 0.0,
            last_edited_by: None,
            last_edited_at: None,
            last_delta_summary: None,
            changelog: None,
            creation_kind: "distilled".to_string(),
            review_status: "confirmed".to_string(),
            workspace: None,
            citations: Vec::new(),
            kind: "concept".to_string(),
        }
    }

    #[tokio::test]
    async fn applies_user_edit_when_md_body_differs() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page("page_a", "Topic A", "original body line");
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            &page.id,
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();

        write_page_md(knowledge_dir.path(), &page, Some("user-edited body line"));
        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(stats.applied, 1);

        let p = db.get_page("page_a").await.unwrap().unwrap();
        assert_eq!(p.content, "user-edited body line");
        // fs_edit must flip user_edited so refinery escalates instead of
        // overwriting on the next re-distill.
        assert!(p.user_edited);
    }

    #[tokio::test]
    async fn no_op_when_md_matches_db() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page("page_b", "Topic B", "matching content");
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            &page.id,
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();
        write_page_md(knowledge_dir.path(), &page, None);

        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(stats.applied, 0);
        assert_eq!(stats.skipped_unchanged, 1);
    }

    #[tokio::test]
    async fn skips_md_with_no_origin_id() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        // User dropped a plain md in the page dir with no origin_id.
        std::fs::write(
            knowledge_dir.path().join("freeform.md"),
            "---\ntitle: \"Random\"\n---\n\njust a note",
        )
        .unwrap();

        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(stats.applied, 0);
        assert_eq!(stats.skipped_no_origin_id, 1);
    }

    #[tokio::test]
    async fn skips_md_pointing_at_unknown_page_id() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let ghost = sample_page("page_ghost", "Ghost", "ghost body");
        // DB never inserted page_ghost — md is dangling.
        write_page_md(knowledge_dir.path(), &ghost, None);

        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(stats.applied, 0);
        assert_eq!(stats.skipped_unknown_page, 1);
    }

    #[tokio::test]
    async fn skips_when_md_version_trails_db() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_c",
            "Topic C",
            None,
            "daemon body v3",
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();
        // Daemon bumped version to 3 via two updates without re-projecting md.
        db.update_page_content("page_c", "daemon body v2", &["mem_seed"], "re_distill")
            .await
            .unwrap();
        db.update_page_content("page_c", "daemon body v3", &["mem_seed"], "re_distill")
            .await
            .unwrap();

        // md frontmatter stamps origin_version: 1 — the projection from
        // the original insert. md body differs from DB.
        let mut stale = sample_page("page_c", "Topic C", "old body from disk");
        stale.version = 1;
        write_page_md(knowledge_dir.path(), &stale, None);

        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        // Daemon ahead — don't roll DB back to the disk's old body.
        assert_eq!(stats.applied, 0);
        assert_eq!(stats.skipped_daemon_ahead, 1);
        let p = db.get_page("page_c").await.unwrap().unwrap();
        assert_eq!(p.content, "daemon body v3");
        assert!(!p.user_edited);
    }

    #[tokio::test]
    async fn applies_consecutive_user_edits() {
        // Regression: after apply, md must be re-projected so the frontmatter
        // origin_version stays in sync with the DB. Otherwise the second
        // edit lands as SkippedDaemonAhead.
        //
        // Seed the page via KnowledgeWriter so state.json + filename + DB
        // all line up the way they would in production. Subsequent edits
        // rewrite that same file in place.
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page("page_x", "Topic X", "original body");
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            &page.id,
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();
        let writer = tracked_writer(&db, knowledge_dir.path());
        project_page(&db, &writer, &page);
        let md_path = knowledge_dir
            .path()
            .join(writer.page_filename(&page.id).expect("state has file"));

        // Edit #1 — replace body.
        let projected = std::fs::read_to_string(&md_path).unwrap();
        let header_end = projected.find("\n---\n\n").expect("frontmatter present");
        let head = &projected[..header_end + "\n---\n\n".len()];
        std::fs::write(&md_path, format!("{head}edit one\n")).unwrap();
        let s1 = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(s1.applied, 1, "edit one must apply; got {:?}", s1);

        // Edit #2 — same path, new body. The watcher's re-projection
        // bumped origin_version, so a fresh read picks up the new header.
        let projected = std::fs::read_to_string(&md_path).unwrap();
        let header_end = projected.find("\n---\n\n").expect("frontmatter present");
        let head = &projected[..header_end + "\n---\n\n".len()];
        std::fs::write(&md_path, format!("{head}edit two — second pass\n")).unwrap();
        let s2 = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(s2.applied, 1, "edit two must apply; got {:?}", s2);
        let p = db.get_page("page_x").await.unwrap().unwrap();
        assert_eq!(p.content, "edit two — second pass");
    }

    #[tokio::test]
    async fn nonexistent_knowledge_path_is_a_noop() {
        let (db, _ddir) = fresh_db().await;
        let missing = std::path::PathBuf::from("/tmp/origin-page-watcher-nonexistent");
        let stats = sync_filesystem_edits(&db, &missing).await.unwrap();
        assert_eq!(stats, WatcherStats::default());
    }

    #[tokio::test]
    async fn freshly_projected_page_is_unchanged_not_user_edited() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page("page_fresh", "Fresh Topic", "## Overview\nbody prose here");
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            &page.id,
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();
        // Project via the live writer — the .md now carries the delimiter
        // Sources block + sources: frontmatter that Page.content lacks.
        let writer = tracked_writer(&db, knowledge_dir.path());
        project_page(&db, &writer, &page);

        // First watcher tick must see the projection as Unchanged, NOT a
        // user edit — otherwise the refinery is locked out of every page.
        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(
            stats.applied, 0,
            "fresh projection must not apply; got {:?}",
            stats
        );
        assert_eq!(
            stats.scanned, 1,
            "exactly one md file should be scanned; got {:?}",
            stats
        );
        assert_eq!(
            stats.skipped_unchanged, 1,
            "fresh projection must be skipped_unchanged; got {:?}",
            stats
        );
        let p = db.get_page("page_fresh").await.unwrap().unwrap();
        assert!(!p.user_edited);
    }

    #[tokio::test]
    async fn genuine_prose_edit_with_block_present_applies_block_free() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page("page_edit", "Edit Topic", "## Overview\noriginal prose");
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            &page.id,
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_e"],
            &now,
        )
        .await
        .unwrap();
        let writer = tracked_writer(&db, knowledge_dir.path());
        let fname = project_page(&db, &writer, &page);
        let md_path = knowledge_dir.path().join(&fname);

        // The projected file has the Sources block. User edits the PROSE only,
        // leaving the block intact.
        let projected = std::fs::read_to_string(&md_path).unwrap();
        let edited = projected.replace("original prose", "edited prose by the user");
        assert_ne!(edited, projected, "prose replacement must have matched");
        std::fs::write(&md_path, &edited).unwrap();

        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(
            stats.applied, 1,
            "genuine prose edit must apply; got {:?}",
            stats
        );
        let p = db.get_page("page_edit").await.unwrap().unwrap();
        assert!(p.content.contains("edited prose by the user"));
        assert!(
            !p.content
                .contains(crate::export::provenance::SOURCES_BLOCK_START),
            "the generated block must NOT be persisted into Page.content"
        );
        assert!(p.user_edited, "a real prose edit flags user_edited");

        // The block-free body means no `mem_*` rows reach the wikilink graph
        // via replace_page_links.
        let links = db.get_page_outbound_links("page_edit").await.unwrap();
        assert!(
            !links.iter().any(|l| l.label.starts_with("mem_")),
            "no mem_* link rows should persist; got {:?}",
            links
        );
    }

    #[tokio::test]
    async fn edited_sources_block_on_disk_is_reprojected_to_db_truth() {
        use crate::export::provenance::{SOURCES_BLOCK_END, SOURCES_BLOCK_START};
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let mut page = sample_page("page_block", "Block Topic", "## Overview\ncanonical prose");
        page.source_memory_ids = vec!["mem_real".to_string()];
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_block",
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_real"],
            &now,
        )
        .await
        .unwrap();
        let writer = tracked_writer(&db, knowledge_dir.path());
        project_page(&db, &writer, &page);
        let md_path = knowledge_dir
            .path()
            .join(writer.page_filename(&page.id).expect("state has file"));

        // User tampers with ONLY the protected block (adds a fake source),
        // leaving the canonicalized prose identical.
        let projected = std::fs::read_to_string(&md_path).unwrap();
        let tampered = projected.replace(
            &format!("{SOURCES_BLOCK_START}\n## Sources\n- [[mem_real]]\n{SOURCES_BLOCK_END}"),
            &format!("{SOURCES_BLOCK_START}\n## Sources\n- [[mem_real]]\n- [[mem_FAKE]]\n{SOURCES_BLOCK_END}"),
        );
        assert_ne!(
            tampered, projected,
            "replacement must have matched the block"
        );
        std::fs::write(&md_path, &tampered).unwrap();

        let stats = sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();
        assert_eq!(stats.applied, 0); // canonicalized prose unchanged → not a user edit
        let p = db.get_page("page_block").await.unwrap().unwrap();
        assert!(!p.content.contains(SOURCES_BLOCK_START)); // DB stays canonical
        assert!(!p.content.contains("mem_FAKE"));
        assert!(!p.user_edited);
        // The file on disk was re-projected from DB truth — fake source gone.
        let after = std::fs::read_to_string(&md_path).unwrap();
        assert!(!after.contains("mem_FAKE"));
        assert!(after.contains("[[mem_real]]"));
    }

    #[tokio::test]
    async fn no_duplicate_file_when_state_missing_on_reprojection() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page("page_nostate", "No State", "## Overview\nold body");
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_nostate",
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_z"],
            &now,
        )
        .await
        .unwrap();
        // Write an OLD-format .md directly (no write_page → no state.json entry),
        // with frontmatter origin_id matching the DB row. The prose body equals
        // the DB content, but the file lacks the delimiter Sources block a fresh
        // render adds — so the canonicalized bodies are EQUAL (reaching the
        // equal-canon branch) while raw_body != fresh_body (which would trigger
        // re-projection if not for the missing-state guard). origin_version: 1
        // matches the DB row so the watcher does not bail via SkippedDaemonAhead.
        let md = format!(
            "---\ntitle: \"{}\"\norigin_id: page_nostate\norigin_version: 1\n---\n\n## Overview\nold body\n",
            page.title
        );
        std::fs::write(knowledge_dir.path().join("no-state.md"), &md).unwrap();

        sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();

        // No duplicate page file forked.
        let md_files: Vec<_> = std::fs::read_dir(knowledge_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !md_files.iter().any(|n| n.contains("-2")),
            "must not fork a duplicate; got {:?}",
            md_files
        );
    }

    /// Task 5: pure projection + watcher tick with NO user edit must produce
    /// zero `mem_*` page_link rows.  Distinct from Task 4's test, which edits
    /// prose; here we never touch the file after `KnowledgeWriter::write_page`.
    /// The on-disk file carries `[[mem_alpha]]`/`[[mem_beta]]` inside the
    /// generated Sources block, but the canonicalized body written back to the
    /// DB must never include them, so no `mem_*` link rows are ever created.
    #[tokio::test]
    async fn projecting_provenance_creates_no_mem_page_links_no_edit() {
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let page = sample_page(
            "page_links",
            "Links Topic",
            "## Overview\nrefers to [[Real Topic]] but not memories",
        );
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_links",
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_alpha", "mem_beta"],
            &now,
        )
        .await
        .unwrap();
        let writer = tracked_writer(&db, knowledge_dir.path());
        project_page(&db, &writer, &page);

        // Pure projection + a watcher tick with NO user edit. The on-disk file
        // carries the Sources block (with [[mem_alpha]]/[[mem_beta]]), but the
        // canonicalized body written to the DB must never include them, so no
        // mem_* link rows are ever created.
        sync_filesystem_edits(&db, knowledge_dir.path())
            .await
            .unwrap();

        let links = db.get_page_outbound_links("page_links").await.unwrap();
        let mem_links: Vec<_> = links
            .iter()
            .filter(|l| l.label.starts_with("mem_"))
            .collect();
        assert!(
            mem_links.is_empty(),
            "no page_links row may target a mem_* id; got {:?}",
            mem_links.iter().map(|l| &l.label).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn full_provenance_round_trip_is_stable() {
        use crate::export::provenance::SOURCES_BLOCK_START;
        let (db, _ddir) = fresh_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let mut page = sample_page(
            "page_rt",
            "RT Topic",
            "## Overview\nstable prose with [[Other]]",
        );
        page.source_memory_ids = vec!["mem_one".to_string(), "mem_two".to_string()];
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_rt",
            &page.title,
            None,
            &page.content,
            None,
            None,
            &["mem_one", "mem_two"],
            &now,
        )
        .await
        .unwrap();
        let writer = tracked_writer(&db, knowledge_dir.path());
        project_page(&db, &writer, &page);

        // Two consecutive ticks — both Unchanged, DB never gains the block.
        for _ in 0..2 {
            let stats = sync_filesystem_edits(&db, knowledge_dir.path())
                .await
                .unwrap();
            assert_eq!(stats.applied, 0);
            assert_eq!(stats.skipped_unchanged, 1);
        }
        let p = db.get_page("page_rt").await.unwrap().unwrap();
        assert!(!p.content.contains(SOURCES_BLOCK_START));
        assert!(!p.user_edited);
        // Stubs resolve for both cited memories.
        assert!(knowledge_dir
            .path()
            .join("_sources")
            .join("mem_one.md")
            .exists());
        assert!(knowledge_dir
            .path()
            .join("_sources")
            .join("mem_two.md")
            .exists());
        // No mem_* page_links.
        let links = db.get_page_outbound_links("page_rt").await.unwrap();
        assert!(links.iter().all(|l| !l.label.starts_with("mem_")));
    }
}
