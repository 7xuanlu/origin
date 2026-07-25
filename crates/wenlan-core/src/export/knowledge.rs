// SPDX-License-Identifier: Apache-2.0
//! Knowledge writer — exports pages as `.md` files with state tracking.

use crate::error::WenlanError;
use crate::export::obsidian::{convert_links_to_wikilinks, slugify};
use crate::pages::Page;
use cap_fs_ext::{DirExt as _, FollowSymlinks, OpenOptionsFollowExt as _};
use cap_std::fs::{Dir, OpenOptions};
use fs2::FileExt as _;
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::{Read as _, Write as _};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const KNOWLEDGE_STATE_SCHEMA_V2: u32 = 2;
static PROJECTION_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
static PROJECTION_STATE_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const REPAIR_CAPABILITY_READ_MAX_BYTES: u64 = 16 * 1024 * 1024;
const ORPHAN_BASELINE_ENTRY_BUDGET_BYTES: u64 = 4096;
const PROJECTION_UNLINK_STAGE_FILE: &str = "source";
const PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE: &str = "rollback-quarantine";
const PROJECTION_STATE_STAGE_FILE: &str = "state";
const PROJECTION_STAGE_OWNER_FILE: &str = "owner.json";
#[cfg(unix)]
type ProjectionStateMode = u32;
#[cfg(not(unix))]
type ProjectionStateMode = ();

/// Process-local monotonic counter, combined with the pid, so concurrent
/// `write_page` calls for the same page never pick the same temp filename.
static TEMP_FILE_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize)]
struct KnowledgeState {
    #[serde(default = "default_schema_v2")]
    schema_version: u32,
    pages: HashMap<String, PageFileState>,
}

impl Default for KnowledgeState {
    fn default() -> Self {
        Self {
            schema_version: KNOWLEDGE_STATE_SCHEMA_V2,
            pages: HashMap::new(),
        }
    }
}

fn default_schema_v2() -> u32 {
    KNOWLEDGE_STATE_SCHEMA_V2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PageFileState {
    file: String,
    version: i64,
    last_written: String,
}

/// What one `reconcile` pass repaired. Logged as a single summary line at
/// startup — per-page detail would be thousands of lines on a real corpus.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileStats {
    /// Pages with a `state.json` entry, i.e. ones we could check at all.
    pub checked: usize,
    /// Projections found behind the DB and rewritten.
    pub rewritten: usize,
    /// `write_page` leftovers swept from the pages directory.
    pub temp_files_removed: usize,
    /// Repairs that failed; the page keeps its stale file until next boot.
    pub errors: usize,
}

/// The `origin_version` stamp `render_markdown` writes into every projected
/// file, or 0 when the file has no parseable frontmatter (empty, truncated, or
/// never ours). Shared by the page-watcher and the startup reconcile: they
/// make opposite decisions off this number, so reading it two different ways
/// would let reconcile overwrite exactly what the watcher protects.
///
/// Tolerant of `3`, `3.0` and `"3"` — YAML gives each a different serde shape,
/// and a hand-retyped frontmatter should not be mistaken for a stale one.
pub(crate) fn projected_origin_version(fm: &crate::sources::obsidian::NoteFrontmatter) -> i64 {
    fm.fields
        .get("origin_version")
        .and_then(|v| {
            v.as_i64()
                .or_else(|| v.as_f64().map(|f| f as i64))
                .or_else(|| v.as_str().and_then(|s| s.trim().parse::<i64>().ok()))
        })
        .unwrap_or(0)
}

/// True only for the `.{page_id}.{pid}.{seq}.tmp` names `write_page` creates.
/// Page ids carry no dots, so "exactly three components, the last two all
/// digits" is our writer's shape and nobody else's — an editor swap file or a
/// user's own `.tmp` scratch in the pages directory does not match.
fn is_write_page_temp_file(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('.').and_then(|n| n.strip_suffix(".tmp")) else {
        return false;
    };
    let mut parts = rest.rsplitn(3, '.');
    let seq = parts.next().unwrap_or_default();
    let pid = parts.next().unwrap_or_default();
    let page_id = parts.next().unwrap_or_default();
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    !page_id.is_empty() && !page_id.contains('.') && all_digits(pid) && all_digits(seq)
}

/// Legacy v1 KnowledgeState. The Phase 0 (Page) taxonomy refactor renamed
/// `concepts` → `pages` and changed the page-id prefix `concept_<uuid>` →
/// `page_<uuid>`. v1 state.json files are auto-migrated on read. Drop in
/// next minor release.
#[derive(Debug, Default, Deserialize)]
struct LegacyKnowledgeStateV1 {
    concepts: HashMap<String, PageFileState>,
}

pub struct KnowledgeWriter {
    path: PathBuf,
    tracker: std::sync::Arc<crate::page_projection_tracker::PageProjectionTracker>,
    reap_orphan_stubs: bool,
    write_provenance: bool,
}

impl KnowledgeWriter {
    pub fn new(path: PathBuf, database: &crate::db::MemoryDB) -> Self {
        Self {
            path,
            tracker: database.page_projection_tracker(),
            reap_orphan_stubs: true,
            write_provenance: true,
        }
    }

    fn new_repair(path: PathBuf, database: &crate::db::MemoryDB) -> Self {
        Self {
            path,
            tracker: database.page_projection_tracker(),
            reap_orphan_stubs: false,
            write_provenance: false,
        }
    }

    pub fn begin_projection_write(&self) -> KnowledgeProjectionWriteRef<'_> {
        KnowledgeProjectionWriteRef {
            writer: self,
            guard: self.tracker.begin_write(),
        }
    }

    #[cfg(test)]
    fn new_for_test(path: PathBuf) -> Self {
        Self {
            path,
            tracker: crate::page_projection_tracker::PageProjectionTracker::new(),
            reap_orphan_stubs: true,
            write_provenance: true,
        }
    }

    #[cfg(test)]
    fn begin_test_write(&self) -> crate::page_projection_tracker::PageProjectionWriteGuard {
        self.tracker.begin_write()
    }

    #[cfg(test)]
    fn write_page_for_test(&self, page: &Page) -> Result<String, WenlanError> {
        let guard = self.begin_test_write();
        self.write_page(&guard, page)
    }

    #[cfg(test)]
    fn write_page_after_open_for_test<F>(
        &self,
        page: &Page,
        after_open: F,
    ) -> Result<String, WenlanError>
    where
        F: FnOnce() -> Result<(), WenlanError>,
    {
        let guard = self.begin_test_write();
        self.validate_guard(&guard)?;
        create_projection_root_nofollow(&self.path)?;
        KnowledgeProjectionWrite::with_projection_capabilities(&self.path, |capabilities| {
            after_open()?;
            self.write_page_with_lock_held(capabilities, &guard, page)
        })
    }

    #[cfg(test)]
    fn remove_page_for_test(&self, page_id: &str) -> Result<(), WenlanError> {
        let guard = self.begin_test_write();
        self.remove_page(&guard, page_id)
    }

    #[cfg(test)]
    fn reconcile_for_test(&self, pages: &[Page]) -> Result<ReconcileStats, WenlanError> {
        let guard = self.begin_test_write();
        self.reconcile(&guard, pages)
    }

    pub fn write_page(
        &self,
        guard: &crate::page_projection_tracker::PageProjectionWriteGuard,
        page: &Page,
    ) -> Result<String, WenlanError> {
        self.validate_guard(guard)?;
        create_projection_root_nofollow(&self.path)?;
        KnowledgeProjectionWrite::with_projection_capabilities(&self.path, |capabilities| {
            self.write_page_with_lock_held(capabilities, guard, page)
        })
    }

    fn write_page_with_lock_held(
        &self,
        capabilities: &ProjectionCapabilities,
        guard: &crate::page_projection_tracker::PageProjectionWriteGuard,
        page: &Page,
    ) -> Result<String, WenlanError> {
        self.write_page_with_lock_held_and_hook(capabilities, guard, page, || Ok(()))
    }

    fn write_page_with_lock_held_and_hook<F>(
        &self,
        capabilities: &ProjectionCapabilities,
        guard: &crate::page_projection_tracker::PageProjectionWriteGuard,
        page: &Page,
        after_target_write: F,
    ) -> Result<String, WenlanError>
    where
        F: FnOnce() -> Result<(), WenlanError>,
    {
        self.validate_guard(guard)?;
        let mut state = self.load_state_cap(&capabilities.wenlan);
        let filename =
            self.unique_filename_cap(&capabilities.root, &page.id, &page.title, &state)?;
        let file_path = self.path.join(&filename);

        let content = render_markdown(page);
        // Write to a temp file in the same capability directory, then rename
        // over the target: rename within one directory is atomic, so a reader
        // (Obsidian, the fs watcher) never observes a half-written page file.
        // Truncating the target in place — what `write_regular_nofollow` does —
        // cannot give that: it leaves a window where the file is short or empty.
        //
        // The symlink and regular-file guarantees are preserved: the temp is
        // opened `create_new` + nofollow, and an existing target is checked to be
        // a regular file before it is replaced.
        let temp_filename = format!(
            ".{}.{}.{}.tmp",
            page.id,
            std::process::id(),
            TEMP_FILE_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        write_page_atomically_nofollow(
            &capabilities.root,
            &filename,
            &temp_filename,
            content.as_bytes(),
        )?;
        after_target_write()?;

        if self.write_provenance {
            // Project read-only source stubs so [[mem_*]] resolves in Obsidian.
            if let Err(e) = crate::export::provenance::project_stubs_for_page_in(
                &capabilities.root,
                &page.id,
                &page.source_memory_ids,
            ) {
                log::warn!("[knowledge] stub projection failed for {}: {e}", page.id);
            }

            // Manifest is a cosmetic projection index, updated load-modify-save
            // without a lock. HTTP page-write handlers can run concurrently with
            // scheduler distillation, so a lost update is possible — but the blast
            // radius is one stub and it self-heals: a spuriously-GC'd stub for a
            // live page is regenerated on that page's next write_page (project runs
            // before this update), and a leaked orphan is reaped on the next GC. A
            // real lock is a P2 concern, unjustified for a regenerable nav aid.
            let mut manifest =
                crate::export::provenance::StubManifest::load_from(&capabilities.root);
            manifest.record(&page.id, &page.source_memory_ids);
            let _ = manifest.save_to(&capabilities.root);
            if self.reap_orphan_stubs {
                let _ =
                    crate::export::provenance::gc_orphan_stubs_in(&capabilities.root, &manifest);
            }
        }

        state.pages.insert(
            page.id.clone(),
            PageFileState {
                file: filename,
                version: page.version,
                last_written: page.last_modified.clone(),
            },
        );
        self.save_state_cap(&capabilities.wenlan, &state)?;

        Ok(file_path.to_string_lossy().to_string())
    }

    /// Repair the markdown projection from the DB, which is authoritative.
    ///
    /// `write_page` deliberately skips fsync, so a crash can leave a page's
    /// file missing, holding the previous version's bytes (rename undone), or
    /// zero-length (rename durable, bytes not). This brings those forward
    /// through the same atomic `write_page` path, and sweeps `.tmp` leftovers
    /// from writes that died between write and rename.
    ///
    /// Staleness is judged by the `origin_version` stamp, NOT by comparing
    /// bytes against `render_markdown`. md is canonical for prose (see
    /// `sources::page_watcher`): a body that differs while the stamp is
    /// current is a user edit made while the daemon was down, and the watcher
    /// reflects it back into the DB on its next tick. Rewriting on
    /// content-inequality would delete that edit before the watcher ever saw
    /// it. A stale stamp, by contrast, can only mean the daemon wrote last and
    /// the file did not keep up.
    ///
    /// ponytail: two known ceilings, both cheap to live with. (1) A file
    /// corrupted in the BODY while its frontmatter still reads current is
    /// indistinguishable from a user edit without a per-page checksum, so it
    /// is left alone — add a content hash to `PageFileState` if that ever
    /// bites. (2) Pages with no `state.json` entry are skipped: we cannot tell
    /// which file on disk is theirs, and guessing forks a `<slug>-2.md`
    /// duplicate. The empty-directory case is already covered by the daemon's
    /// one-time backfill.
    pub fn reconcile(
        &self,
        guard: &crate::page_projection_tracker::PageProjectionWriteGuard,
        pages: &[Page],
    ) -> Result<ReconcileStats, WenlanError> {
        self.validate_guard(guard)?;
        let mut stats = ReconcileStats {
            temp_files_removed: self.sweep_temp_leftovers(),
            ..ReconcileStats::default()
        };

        let state = self.load_state();
        for page in pages {
            let Some(entry) = state.pages.get(&page.id) else {
                continue;
            };
            stats.checked += 1;
            if !self.projection_is_behind(&entry.file, page) {
                continue;
            }
            match self.write_page(guard, page) {
                Ok(_) => stats.rewritten += 1,
                Err(e) => {
                    log::warn!("[reconcile] repair failed for {}: {e}", page.id);
                    stats.errors += 1;
                }
            }
        }
        Ok(stats)
    }

    /// Whether `filename` is missing, unreadable, or stamped with a version
    /// older than the DB's. Unreadable counts as behind: the repair attempt
    /// will surface the real error rather than silently skipping the page.
    fn projection_is_behind(&self, filename: &str, page: &Page) -> bool {
        let Ok(raw) = std::fs::read_to_string(self.path.join(filename)) else {
            return true;
        };
        let (fm, _body) = crate::sources::obsidian::extract_frontmatter(&raw);
        projected_origin_version(&fm) < page.version
    }

    /// Remove `write_page` temp files orphaned by a crash between write and
    /// rename. Safe to run unconditionally at startup: the daemon holds the
    /// port by this point, so no live writer owns any of these.
    fn sweep_temp_leftovers(&self) -> usize {
        let Ok(entries) = std::fs::read_dir(&self.path) else {
            return 0;
        };
        entries
            .flatten()
            .filter(|e| is_write_page_temp_file(&e.file_name().to_string_lossy()))
            .filter(|e| std::fs::remove_file(e.path()).is_ok())
            .count()
    }

    fn unique_filename_cap(
        &self,
        root: &Dir,
        page_id: &str,
        title: &str,
        state: &KnowledgeState,
    ) -> Result<String, WenlanError> {
        if let Some(existing) = state.pages.get(page_id) {
            return Ok(existing.file.clone());
        }
        let base = slugify(title);
        let mut candidate = format!("{base}.md");
        let mut n = 2;
        let taken: std::collections::HashSet<&str> = state
            .pages
            .iter()
            .filter(|(id, _)| id.as_str() != page_id)
            .map(|(_, state)| state.file.as_str())
            .collect();
        loop {
            let collides_state = taken.contains(candidate.as_str());
            let collides_disk = match root.symlink_metadata(&candidate) {
                Ok(_) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => return Err(WenlanError::Io(error)),
            };
            if !collides_state && !collides_disk {
                return Ok(candidate);
            }
            candidate = format!("{base}-{n}.md");
            n += 1;
        }
    }

    /// Resolve the filename currently recorded in `state.json` for a page,
    /// or `None` if the page has no projection yet. Used by the PUT route's
    /// rollback path so a failed DB update can restore the prior md bytes
    /// instead of orphaning the file.
    pub fn page_filename(&self, page_id: &str) -> Option<String> {
        self.load_state().pages.get(page_id).map(|s| s.file.clone())
    }

    pub fn remove_page(
        &self,
        guard: &crate::page_projection_tracker::PageProjectionWriteGuard,
        page_id: &str,
    ) -> Result<(), WenlanError> {
        self.validate_guard(guard)?;
        KnowledgeProjectionWrite::with_projection_capabilities(&self.path, |capabilities| {
            let mut state = self.load_state_cap(&capabilities.wenlan);

            if let Some(entry) = state.pages.remove(page_id) {
                // Delete the file *before* persisting state. If the file remove
                // fails we keep the state entry so the daemon can retry; if the
                // state save fails we have at most a stale empty entry pointing
                // at a missing file (detectable, recoverable) instead of an
                // orphan file with no DB or state reference.
                match capabilities.root.symlink_metadata(&entry.file) {
                    Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
                        capabilities.root.remove_file(&entry.file)?;
                    }
                    Ok(_) => {
                        return Err(WenlanError::Conflict(
                            "page_projection_target_invalid".to_string(),
                        ))
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(WenlanError::Io(error)),
                }
                self.save_state_cap(&capabilities.wenlan, &state)?;

                let mut manifest =
                    crate::export::provenance::StubManifest::load_from(&capabilities.root);
                manifest.forget_page(page_id);
                let _ = manifest.save_to(&capabilities.root);
                let _ =
                    crate::export::provenance::gc_orphan_stubs_in(&capabilities.root, &manifest);
            }

            Ok(())
        })
    }

    fn validate_guard(
        &self,
        guard: &crate::page_projection_tracker::PageProjectionWriteGuard,
    ) -> Result<(), WenlanError> {
        if guard.belongs_to(&self.tracker) {
            Ok(())
        } else {
            Err(WenlanError::VectorDb(
                "page projection write guard does not own this writer".to_string(),
            ))
        }
    }

    fn load_state(&self) -> KnowledgeState {
        let state_path = self.path.join(".wenlan/state.json");
        let data = match std::fs::read_to_string(&state_path) {
            Ok(d) => d,
            Err(_) => return KnowledgeState::default(),
        };
        self.parse_state(&data)
    }

    fn load_state_cap(&self, wenlan: &Dir) -> KnowledgeState {
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = match wenlan.open_with("state.json", &options) {
            Ok(file) => file,
            Err(_) => return KnowledgeState::default(),
        };
        let mut data = String::new();
        if (&mut file)
            .take(crate::lint::pages::fs::STATE_MAX_BYTES.saturating_add(1))
            .read_to_string(&mut data)
            .is_err()
            || u64::try_from(data.len()).unwrap_or(u64::MAX)
                > crate::lint::pages::fs::STATE_MAX_BYTES
        {
            return KnowledgeState::default();
        }
        self.parse_state(&data)
    }

    fn parse_state(&self, data: &str) -> KnowledgeState {
        // v1 detection: has "concepts" key, no "pages" key. Migrate inline.
        // Heuristic on raw bytes is good enough — the legacy file has at most
        // a thousand small entries and we only enter this branch once per boot.
        if data.contains("\"concepts\"") && !data.contains("\"pages\"") {
            let v1: LegacyKnowledgeStateV1 = serde_json::from_str(data).unwrap_or_default();
            log::info!(
                "[knowledge] migrating state.json v1 -> v2 ({} entries; rewriting concept_ -> page_ id prefix)",
                v1.concepts.len()
            );
            let pages: HashMap<String, PageFileState> = v1
                .concepts
                .into_iter()
                .map(|(id, st)| {
                    let new_id = if let Some(rest) = id.strip_prefix("concept_") {
                        format!("page_{rest}")
                    } else {
                        id
                    };
                    (new_id, st)
                })
                .collect();
            return KnowledgeState {
                schema_version: KNOWLEDGE_STATE_SCHEMA_V2,
                pages,
            };
        }

        let mut state: KnowledgeState = serde_json::from_str(data).unwrap_or_default();
        if state.schema_version == 0 {
            state.schema_version = KNOWLEDGE_STATE_SCHEMA_V2;
        }
        state
    }

    #[cfg(test)]
    fn save_state(&self, state: &KnowledgeState) -> Result<(), WenlanError> {
        let state_path = self.path.join(".wenlan/state.json");
        let data = serde_json::to_string_pretty(state)?;
        std::fs::write(&state_path, data)?;
        Ok(())
    }

    fn save_state_cap(&self, wenlan: &Dir, state: &KnowledgeState) -> Result<(), WenlanError> {
        let data = serde_json::to_vec_pretty(state)?;
        write_regular_nofollow(wenlan, "state.json", &data)
    }
}

pub struct KnowledgeProjectionWriteRef<'writer> {
    writer: &'writer KnowledgeWriter,
    guard: crate::page_projection_tracker::PageProjectionWriteGuard,
}

impl KnowledgeProjectionWriteRef<'_> {
    pub fn write_page(&self, page: &Page) -> Result<String, WenlanError> {
        self.writer.write_page(&self.guard, page)
    }

    pub fn remove_page(&self, page_id: &str) -> Result<(), WenlanError> {
        self.writer.remove_page(&self.guard, page_id)
    }
}

pub struct KnowledgeProjectionWrite {
    writer: KnowledgeWriter,
    guard: crate::page_projection_tracker::PageProjectionWriteGuard,
}

pub(crate) struct LockedRepairProjection<'a> {
    write: &'a KnowledgeProjectionWrite,
    capabilities: &'a ProjectionCapabilities,
}

pub(crate) struct LockedProjection<'a> {
    capabilities: &'a ProjectionCapabilities,
}

/// Repair-only projection ownership that can safely remain alive across
/// awaited database work. It owns the tracker guard, capability roots, and
/// advisory file lock; it never borrows `PROJECTION_WRITE_LOCK`.
pub(crate) struct OwnedRepairProjectionSession {
    write: KnowledgeProjectionWrite,
    capabilities: ProjectionCapabilities,
}

pub(crate) struct RepairReadBudget {
    remaining: u64,
}

impl RepairReadBudget {
    pub(crate) fn new() -> Self {
        Self {
            remaining: REPAIR_CAPABILITY_READ_MAX_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProjectionFileIdentity {
    device: u64,
    inode: u64,
}

fn projection_file_identity(
    metadata: &cap_std::fs::Metadata,
) -> Result<ProjectionFileIdentity, WenlanError> {
    use cap_fs_ext::MetadataExt as _;
    Ok(ProjectionFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

impl LockedProjection<'_> {
    pub(crate) fn scan_page_root_controlled(
        &self,
        include_body_digests: bool,
        control: &crate::lint::pages::fs::PageScanControl,
    ) -> Result<crate::lint::pages::fs::PageScan, WenlanError> {
        crate::lint::pages::fs::scan_page_root_capability_controlled(
            &self.capabilities.root,
            include_body_digests,
            control,
        )
        .map_err(|error| WenlanError::Validation(format!("repair projection scan: {error}")))
    }

    pub(crate) fn capture_stale_page_projection_current(
        &self,
        page_id: &str,
        source_path: &str,
        quarantine_path: &str,
    ) -> Result<crate::repair::StoredRollbackArtifact, WenlanError> {
        crate::repair::capture_stale_page_projection_current_locked(
            self,
            page_id,
            source_path,
            quarantine_path,
        )
    }

    #[cfg(test)]
    pub(crate) fn read_relative_regular_nofollow(
        &self,
        relative: &str,
        max_bytes: u64,
    ) -> Result<Option<Vec<u8>>, WenlanError> {
        let mut budget = RepairReadBudget {
            remaining: max_bytes,
        };
        self.read_relative_regular_nofollow_inner(relative, &mut budget, || Ok(()))
    }

    #[cfg(test)]
    pub(crate) fn read_relative_regular_nofollow_with_after_metadata<F>(
        &self,
        relative: &str,
        max_bytes: u64,
        after_metadata: F,
    ) -> Result<Option<Vec<u8>>, WenlanError>
    where
        F: FnOnce() -> Result<(), WenlanError>,
    {
        let mut budget = RepairReadBudget {
            remaining: max_bytes,
        };
        self.read_relative_regular_nofollow_inner(relative, &mut budget, after_metadata)
    }

    pub(crate) fn read_relative_regular_nofollow_budget(
        &self,
        relative: &str,
        budget: &mut RepairReadBudget,
    ) -> Result<Option<Vec<u8>>, WenlanError> {
        self.read_relative_regular_nofollow_inner(relative, budget, || Ok(()))
    }

    fn read_relative_regular_nofollow_inner<F>(
        &self,
        relative: &str,
        budget: &mut RepairReadBudget,
        after_metadata: F,
    ) -> Result<Option<Vec<u8>>, WenlanError>
    where
        F: FnOnce() -> Result<(), WenlanError>,
    {
        crate::repair::validate_projection_relative_path(relative)?;
        let relative = Path::new(relative);
        let mut components = relative.components().peekable();
        let mut directory = self.capabilities.root.try_clone()?;
        while let Some(component) = components.next() {
            let Component::Normal(component) = component else {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            };
            if components.peek().is_none() {
                return read_optional_regular_nofollow_bounded(
                    &directory,
                    component,
                    budget,
                    after_metadata,
                );
            }
            directory = match directory.open_dir_nofollow(Path::new(component)) {
                Ok(directory) => directory,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(WenlanError::Io(error)),
            };
        }
        Err(WenlanError::Conflict("repair_target_stale".to_string()))
    }

    pub(crate) fn orphaned_baseline_nofollow(
        &self,
        budget: &mut RepairReadBudget,
    ) -> Result<Option<Vec<(String, String)>>, WenlanError> {
        match self.capabilities.wenlan.symlink_metadata("orphaned") {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                let orphaned = self.capabilities.wenlan.open_dir_nofollow("orphaned")?;
                ensure_orphaned_private(&orphaned)?;
                Ok(Some(orphaned_baseline_nofollow(&orphaned, None, budget)?))
            }
            Ok(_) => Err(WenlanError::Conflict("repair_target_stale".to_string())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(WenlanError::Io(error)),
        }
    }

    pub(crate) fn capture_rename_page_projection(
        &self,
        page_id: &str,
    ) -> Result<(String, Vec<wenlan_types::repair::RepairRollbackFileEntry>), WenlanError> {
        let mut budget = RepairReadBudget::new();
        let state_bytes = self
            .read_relative_regular_nofollow_budget(".wenlan/state.json", &mut budget)?
            .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
        let state = serde_json::from_slice::<KnowledgeState>(&state_bytes)
            .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
        if state.schema_version != KNOWLEDGE_STATE_SCHEMA_V2 {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let target_path = state
            .pages
            .get(page_id)
            .map(|entry| entry.file.clone())
            .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
        let normalized = crate::lint::pages::fs::normalize_target_path(&target_path)
            .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?
            .as_str()
            .to_string();
        let target = Path::new(&normalized);
        if normalized != target_path
            || target.components().count() != 1
            || target.extension().and_then(|value| value.to_str()) != Some("md")
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let target_bytes = self
            .read_relative_regular_nofollow_budget(&normalized, &mut budget)
            .map_err(|error| match error {
                WenlanError::Io(_) => WenlanError::Conflict("repair_target_stale".to_string()),
                other => other,
            })?
            .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
        let scan = self.scan_page_root_controlled(
            true,
            &crate::lint::pages::fs::PageScanControl::with_timeout(std::time::Duration::from_secs(
                30,
            )),
        )?;
        if matches!(
            scan.raw_state.kind,
            crate::lint::pages::state::RawStateKind::Malformed
        ) {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        if !crate::lint::pages::state_checks::projection_target_is_exclusive_page_markdown(
            &scan,
            page_id,
            &normalized,
        ) {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let entries = vec![
            wenlan_types::repair::RepairRollbackFileEntry::file(
                ".wenlan/state.json".to_string(),
                state_bytes,
            )
            .map_err(|error| WenlanError::Validation(error.to_string()))?,
            wenlan_types::repair::RepairRollbackFileEntry::file(normalized.clone(), target_bytes)
                .map_err(|error| WenlanError::Validation(error.to_string()))?,
        ];
        Ok((normalized, entries))
    }

    fn restore_rename_page_projection(
        &self,
        target_path: &str,
        entries: &[wenlan_types::repair::RepairRollbackFileEntry],
    ) -> Result<(), WenlanError> {
        if entries.len() != 2
            || entries[0].relative_path() != ".wenlan/state.json"
            || entries[1].relative_path() != target_path
            || entries
                .iter()
                .any(|entry| entry.kind() != wenlan_types::repair::RepairRollbackFileKind::File)
        {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        let state_bytes = hex::decode(entries[0].content_hex()).map_err(|_| {
            WenlanError::Validation("repair_projection_rollback_invalid".to_string())
        })?;
        let target_bytes = hex::decode(entries[1].content_hex()).map_err(|_| {
            WenlanError::Validation("repair_projection_rollback_invalid".to_string())
        })?;
        write_regular_nofollow(&self.capabilities.root, target_path, &target_bytes)?;
        let mut budget = RepairReadBudget::new();
        let (_, state_mode) = read_state_nofollow(&self.capabilities.wenlan, &mut budget)?;
        write_state_atomically(&self.capabilities.wenlan, &state_bytes, state_mode)?;
        #[cfg(unix)]
        {
            sync_dir_capability(&self.capabilities.wenlan)?;
            sync_dir_capability(&self.capabilities.root)?;
        }
        Ok(())
    }
}

impl LockedRepairProjection<'_> {
    fn locked_projection(&self) -> LockedProjection<'_> {
        LockedProjection {
            capabilities: self.capabilities,
        }
    }

    pub(crate) fn scan_page_root_controlled(
        &self,
        include_body_digests: bool,
        control: &crate::lint::pages::fs::PageScanControl,
    ) -> Result<crate::lint::pages::fs::PageScan, WenlanError> {
        self.locked_projection()
            .scan_page_root_controlled(include_body_digests, control)
    }

    pub(crate) fn capture_stale_page_projection_current(
        &self,
        page_id: &str,
        source_path: &str,
        quarantine_path: &str,
    ) -> Result<crate::repair::StoredRollbackArtifact, WenlanError> {
        self.locked_projection()
            .capture_stale_page_projection_current(page_id, source_path, quarantine_path)
    }

    pub(crate) fn write_page(&self, page: &Page) -> Result<String, WenlanError> {
        self.write
            .writer
            .write_page_with_lock_held(self.capabilities, &self.write.guard, page)
    }

    pub(crate) fn write_page_with_after_target_write<F>(
        &self,
        page: &Page,
        after_target_write: F,
    ) -> Result<String, WenlanError>
    where
        F: FnOnce() -> Result<(), WenlanError>,
    {
        self.write.writer.write_page_with_lock_held_and_hook(
            self.capabilities,
            &self.write.guard,
            page,
            after_target_write,
        )
    }

    pub(crate) fn capture_rename_page_projection(
        &self,
        page_id: &str,
    ) -> Result<(String, Vec<wenlan_types::repair::RepairRollbackFileEntry>), WenlanError> {
        self.locked_projection()
            .capture_rename_page_projection(page_id)
    }

    pub(crate) fn restore_rename_page_projection(
        &self,
        target_path: &str,
        entries: &[wenlan_types::repair::RepairRollbackFileEntry],
    ) -> Result<(), WenlanError> {
        self.locked_projection()
            .restore_rename_page_projection(target_path, entries)
    }

    #[cfg(test)]
    pub(crate) fn quarantine_stale_page(
        &self,
        page_id: &str,
        source_path: &str,
        quarantine_path: &str,
    ) -> Result<(), WenlanError> {
        let approved =
            self.capture_stale_page_projection_current(page_id, source_path, quarantine_path)?;
        self.pin_stale_page_projection(page_id, source_path, quarantine_path, &approved, page_id)?
            .quarantine()
    }

    pub(crate) fn pin_stale_page_projection<'lock>(
        &'lock self,
        page_id: &str,
        source_path: &str,
        quarantine_path: &str,
        approved: &crate::repair::StoredRollbackArtifact,
        stage_owner: &str,
    ) -> Result<PinnedStalePageProjection<'lock>, WenlanError> {
        crate::repair::validate_projection_relative_path(source_path)?;
        crate::repair::validate_projection_relative_path(quarantine_path)?;
        let (approved_source_path, approved_quarantine_path) =
            crate::repair::stale_page_projection_paths(approved)?;
        if approved.source_id != page_id
            || approved_source_path != source_path
            || approved_quarantine_path != quarantine_path
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let approved_state_bytes = rollback_row_bytes(approved, ".wenlan/state.json")?;
        let approved_source_bytes = rollback_row_bytes(approved, source_path)?;
        let approved_quarantine = approved
            .rows
            .iter()
            .find(|row| row[0] == quarantine_path)
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
        if approved_quarantine[1] != "missing" || !approved_quarantine[2].is_empty() {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        let approved_orphaned = crate::repair::stale_page_projection_orphaned_baseline(approved)?;
        let quarantine = Path::new(quarantine_path);
        if quarantine.parent() != Some(Path::new(".wenlan/orphaned")) {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let quarantine_name = quarantine
            .file_name()
            .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?
            .to_os_string();
        let (source_parent, source_name) =
            open_relative_parent_nofollow(&self.capabilities.root, Path::new(source_path))?;
        let stage_name = projection_unlink_stage_name(stage_owner);

        let mut read_budget = RepairReadBudget::new();
        let (expected_state_bytes, state_mode) =
            read_state_nofollow(&self.capabilities.wenlan, &mut read_budget)?;
        if expected_state_bytes != approved_state_bytes {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let next_state_bytes =
            crate::lint::pages::state::remove_unique_page_member(&expected_state_bytes, page_id)
                .map_err(|()| WenlanError::Conflict("repair_target_stale".to_string()))?;
        let (expected_source_bytes, expected_source_identity) =
            read_regular_identity_nofollow_bounded(&source_parent, &source_name, &mut read_budget)?;
        if expected_source_bytes != approved_source_bytes {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let stage_owner_bytes = projection_stage_owner_bytes(
            stage_owner,
            page_id,
            source_path,
            &expected_source_bytes,
        )?;
        let orphaned = match (
            approved_orphaned.as_ref(),
            self.capabilities.wenlan.symlink_metadata("orphaned"),
        ) {
            (Some(approved_baseline), Ok(metadata))
                if metadata.is_dir() && !metadata.file_type().is_symlink() =>
            {
                let orphaned = self.capabilities.wenlan.open_dir_nofollow("orphaned")?;
                ensure_orphaned_private(&orphaned)?;
                if orphaned_baseline_nofollow(&orphaned, None, &mut read_budget)?
                    != *approved_baseline
                {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                orphaned
            }
            (None, Ok(metadata)) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                let orphaned = self.capabilities.wenlan.open_dir_nofollow("orphaned")?;
                ensure_orphaned_private(&orphaned)?;
                if !orphaned_baseline_nofollow(&orphaned, None, &mut read_budget)?.is_empty() {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                orphaned
            }
            (None, Err(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                #[cfg(unix)]
                {
                    use cap_std::fs::DirBuilderExt as _;
                    let mut builder = cap_std::fs::DirBuilder::new();
                    builder.mode(0o700);
                    self.capabilities
                        .wenlan
                        .create_dir_with("orphaned", &builder)?;
                }
                #[cfg(not(unix))]
                self.capabilities.wenlan.create_dir("orphaned")?;
                let orphaned = self.capabilities.wenlan.open_dir_nofollow("orphaned")?;
                ensure_orphaned_private(&orphaned)?;
                orphaned
            }
            (_, Err(error)) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(WenlanError::Io(error))
            }
            _ => return Err(WenlanError::Conflict("repair_target_stale".to_string())),
        };
        match orphaned.symlink_metadata(&quarantine_name) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => return Err(WenlanError::Conflict("repair_target_stale".to_string())),
            Err(error) => return Err(WenlanError::Io(error)),
        }

        let root = self.capabilities.root.try_clone()?;
        let wenlan = self.capabilities.wenlan.try_clone()?;
        let stage = match OwnedProjectionStage::open_existing(
            &wenlan,
            stage_name.clone(),
            stage_owner_bytes.clone(),
        )? {
            Some(stage) => {
                if !matches!(
                    inspect_owned_projection_stage(Some(&stage), &mut read_budget)?,
                    StageContents::Present {
                        source: None,
                        rollback_quarantine: None,
                        state: None,
                    }
                ) {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                stage
            }
            None => OwnedProjectionStage::create_noclobber(&wenlan, stage_name, stage_owner_bytes)?,
        };
        Ok(PinnedStalePageProjection {
            root,
            wenlan,
            source_parent,
            source_name,
            source_path: source_path.to_string(),
            stage: Some(stage),
            orphaned: Some(orphaned),
            quarantine_name,
            quarantine_path: quarantine_path.to_string(),
            expected_state_bytes,
            next_state_bytes,
            expected_source_bytes,
            expected_source_identity,
            expected_orphaned_baseline: approved_orphaned.unwrap_or_default(),
            state_mode,
            mutation_started: true,
            _lock: std::marker::PhantomData,
        })
    }

    pub(crate) fn recover_stale_page_projection(
        &self,
        rollback: &crate::repair::StoredRollbackArtifact,
        stage_owner: &str,
        restore_post: bool,
    ) -> Result<StalePageProjectionRecoveryState, WenlanError> {
        let (source_path, quarantine_path) = crate::repair::stale_page_projection_paths(rollback)?;
        let state = rollback_row_bytes(rollback, ".wenlan/state.json")?;
        let source = rollback_row_bytes(rollback, &source_path)?;
        let quarantine = rollback
            .rows
            .iter()
            .find(|row| row[0] == quarantine_path)
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
        if quarantine[1] != "missing" || !quarantine[2].is_empty() {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        let exact_post =
            crate::lint::pages::state::remove_unique_page_member(&state, &rollback.source_id)
                .map_err(|()| {
                    WenlanError::Validation("repair_projection_rollback_invalid".to_string())
                })?;
        let (source_parent, source_name) =
            match open_relative_parent_nofollow(&self.capabilities.root, Path::new(&source_path)) {
                Ok(value) => value,
                Err(_) => return Ok(StalePageProjectionRecoveryState::Unknown),
            };
        let quarantine = Path::new(&quarantine_path);
        if quarantine.parent() != Some(Path::new(".wenlan/orphaned")) {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        let quarantine_name = quarantine
            .file_name()
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?
            .to_os_string();
        let orphaned_baseline = crate::repair::stale_page_projection_orphaned_baseline(rollback)?;
        let stage_owner_bytes =
            projection_stage_owner_bytes(stage_owner, &rollback.source_id, &source_path, &source)?;
        let mut recovery = match StalePageProjectionRecovery::open(
            self.capabilities,
            source_parent,
            source_name,
            projection_unlink_stage_name(stage_owner),
            stage_owner_bytes,
            quarantine_name,
            state,
            exact_post,
            source,
            orphaned_baseline,
        ) {
            Ok(recovery) => recovery,
            Err(_) => return Ok(StalePageProjectionRecoveryState::Unknown),
        };
        recovery.normalize_state_stage()?;
        let state = recovery.classify()?;
        if matches!(
            state,
            StalePageProjectionRecoveryState::PreparedStage
                | StalePageProjectionRecoveryState::AfterLink
                | StalePageProjectionRecoveryState::AfterStage
                | StalePageProjectionRecoveryState::RestoredSource
                | StalePageProjectionRecoveryState::AfterUnlink
                | StalePageProjectionRecoveryState::PostStaged
                | StalePageProjectionRecoveryState::QuarantineStaged
                | StalePageProjectionRecoveryState::OriginalCleanup
        ) || (restore_post && matches!(state, StalePageProjectionRecoveryState::Post))
        {
            recovery.restore_original()?;
            return recovery.classify();
        }
        Ok(state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StalePageProjectionRecoveryState {
    Original,
    PreparedStage,
    AfterLink,
    AfterStage,
    RestoredSource,
    AfterUnlink,
    PostStaged,
    Post,
    QuarantineStaged,
    OriginalCleanup,
    Unknown,
}

struct StalePageProjectionRecovery {
    root: Dir,
    wenlan: Dir,
    source_parent: Dir,
    source_name: OsString,
    stage: Option<OwnedProjectionStage>,
    orphaned: Option<Dir>,
    quarantine_name: OsString,
    original_state: Vec<u8>,
    post_state: Vec<u8>,
    source_bytes: Vec<u8>,
    state_mode: ProjectionStateMode,
    orphaned_baseline: Option<Vec<(String, String)>>,
}

impl StalePageProjectionRecovery {
    #[allow(clippy::too_many_arguments)]
    fn open(
        capabilities: &ProjectionCapabilities,
        source_parent: Dir,
        source_name: OsString,
        stage_name: OsString,
        stage_owner_bytes: Vec<u8>,
        quarantine_name: OsString,
        original_state: Vec<u8>,
        post_state: Vec<u8>,
        source_bytes: Vec<u8>,
        orphaned_baseline: Option<Vec<(String, String)>>,
    ) -> Result<Self, WenlanError> {
        let orphaned = match capabilities.wenlan.symlink_metadata("orphaned") {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                let directory = capabilities.wenlan.open_dir_nofollow("orphaned")?;
                ensure_orphaned_private(&directory)?;
                Some(directory)
            }
            Ok(_) => return Err(WenlanError::Conflict("repair_target_stale".to_string())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(WenlanError::Io(error)),
        };
        let stage = match OwnedProjectionStage::open_existing(
            &capabilities.wenlan,
            stage_name.clone(),
            stage_owner_bytes.clone(),
        )? {
            Some(stage) => stage,
            None => OwnedProjectionStage::create_noclobber(
                &capabilities.wenlan,
                stage_name,
                stage_owner_bytes,
            )?,
        };
        let mut state_budget = RepairReadBudget::new();
        let state_mode = match read_optional_projection_state_identity_nofollow(
            &capabilities.wenlan,
            OsStr::new("state.json"),
            &mut state_budget,
        )? {
            Some((_, mode, _)) => mode,
            None => {
                let (_, mode, _) = read_projection_state_identity_nofollow(
                    &stage.directory,
                    OsStr::new(PROJECTION_STATE_STAGE_FILE),
                    &mut state_budget,
                )?;
                mode
            }
        };
        Ok(Self {
            root: capabilities.root.try_clone()?,
            wenlan: capabilities.wenlan.try_clone()?,
            source_parent,
            source_name,
            stage: Some(stage),
            orphaned,
            quarantine_name,
            original_state,
            post_state,
            source_bytes,
            state_mode,
            orphaned_baseline,
        })
    }

    fn normalize_state_stage(&mut self) -> Result<(), WenlanError> {
        let stage = self
            .stage
            .as_ref()
            .ok_or_else(|| WenlanError::Conflict("repair_apply_recovery_required".to_string()))?;
        let mut staged_budget = RepairReadBudget::new();
        let Some((staged, staged_mode, staged_identity)) =
            read_optional_projection_state_identity_nofollow(
                &stage.directory,
                OsStr::new(PROJECTION_STATE_STAGE_FILE),
                &mut staged_budget,
            )?
        else {
            return Ok(());
        };
        if staged_mode != self.state_mode
            || (staged != self.original_state && staged != self.post_state)
        {
            return Ok(());
        }

        let mut public_budget = RepairReadBudget::new();
        let public = read_optional_projection_state_identity_nofollow(
            &self.wenlan,
            OsStr::new("state.json"),
            &mut public_budget,
        )?;
        match public {
            None => {
                match stage.directory.hard_link(
                    PROJECTION_STATE_STAGE_FILE,
                    &self.wenlan,
                    "state.json",
                ) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        return Ok(())
                    }
                    Err(error) => return Err(WenlanError::Io(error)),
                }
                let mut restored_budget = RepairReadBudget::new();
                let (restored, restored_mode, restored_identity) =
                    read_projection_state_identity_nofollow(
                        &self.wenlan,
                        OsStr::new("state.json"),
                        &mut restored_budget,
                    )?;
                if restored != staged
                    || restored_mode != staged_mode
                    || restored_identity != staged_identity
                {
                    return Err(WenlanError::Conflict(
                        "repair_apply_recovery_required".to_string(),
                    ));
                }
            }
            Some((public, public_mode, _))
                if public_mode == self.state_mode
                    && ((staged == self.original_state && public == self.post_state)
                        || (staged == self.post_state && public == self.original_state)) => {}
            Some(_) => return Ok(()),
        }

        let mut final_budget = RepairReadBudget::new();
        let (final_staged, final_mode, final_identity) = read_projection_state_identity_nofollow(
            &stage.directory,
            OsStr::new(PROJECTION_STATE_STAGE_FILE),
            &mut final_budget,
        )?;
        if final_staged != staged || final_mode != staged_mode || final_identity != staged_identity
        {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        stage.directory.remove_file(PROJECTION_STATE_STAGE_FILE)?;
        #[cfg(unix)]
        {
            sync_dir_capability(&stage.directory)?;
            sync_dir_capability(&self.wenlan)?;
            sync_dir_capability(&self.root)?;
        }
        Ok(())
    }

    fn classify(&self) -> Result<StalePageProjectionRecoveryState, WenlanError> {
        let mut read_budget = RepairReadBudget::new();
        let (state, current_mode) = match read_state_nofollow(&self.wenlan, &mut read_budget) {
            Ok(value) => value,
            Err(_) => return Ok(StalePageProjectionRecoveryState::Unknown),
        };
        if current_mode != self.state_mode {
            return Ok(StalePageProjectionRecoveryState::Unknown);
        }
        let source = match read_optional_regular_nofollow(
            &self.source_parent,
            &self.source_name,
            &mut read_budget,
        ) {
            Ok(value) => value,
            Err(_) => return Ok(StalePageProjectionRecoveryState::Unknown),
        };
        let (stage, staged_quarantine, stage_prepared) =
            match inspect_owned_projection_stage(self.stage.as_ref(), &mut read_budget) {
                Ok(StageContents::Absent) => (None, None, false),
                Ok(StageContents::Present {
                    source,
                    rollback_quarantine,
                    state: None,
                }) => (source, rollback_quarantine, true),
                Ok(StageContents::Present { state: Some(_), .. }) => {
                    return Ok(StalePageProjectionRecoveryState::Unknown)
                }
                Err(_) => return Ok(StalePageProjectionRecoveryState::Unknown),
            };
        let quarantine = match &self.orphaned {
            Some(orphaned) => match read_optional_regular_nofollow(
                orphaned,
                &self.quarantine_name,
                &mut read_budget,
            ) {
                Ok(value) => value,
                Err(_) => return Ok(StalePageProjectionRecoveryState::Unknown),
            },
            None => None,
        };
        let orphaned_shape_matches = match (&self.orphaned_baseline, &self.orphaned) {
            (None, None) => true,
            (None, Some(orphaned)) => {
                orphaned_baseline_nofollow(orphaned, Some(&self.quarantine_name), &mut read_budget)?
                    .is_empty()
            }
            (Some(expected), Some(orphaned)) => {
                orphaned_baseline_nofollow(orphaned, Some(&self.quarantine_name), &mut read_budget)?
                    == *expected
            }
            (Some(_), None) => false,
        };
        if !orphaned_shape_matches {
            return Ok(StalePageProjectionRecoveryState::Unknown);
        }
        let source_original = source.as_deref() == Some(self.source_bytes.as_slice());
        let source_absent = source.is_none();
        let stage_original = stage.as_deref() == Some(self.source_bytes.as_slice());
        let stage_absent = stage.is_none() && !stage_prepared;
        let stage_empty = stage.is_none() && stage_prepared;
        let quarantine_original = quarantine.as_deref() == Some(self.source_bytes.as_slice());
        let quarantine_absent = quarantine.is_none();
        let staged_quarantine_original =
            staged_quarantine.as_deref() == Some(self.source_bytes.as_slice());
        let staged_quarantine_absent = staged_quarantine.is_none();
        let restored_links_match = if source_original && stage_original && quarantine_original {
            let identities = (|| {
                let source = projection_file_identity(
                    &self.source_parent.symlink_metadata(&self.source_name)?,
                )?;
                let stage = projection_file_identity(
                    &self
                        .stage
                        .as_ref()
                        .ok_or_else(|| {
                            WenlanError::Conflict("repair_apply_recovery_required".to_string())
                        })?
                        .directory
                        .symlink_metadata(PROJECTION_UNLINK_STAGE_FILE)?,
                )?;
                let quarantine = projection_file_identity(
                    &self
                        .orphaned
                        .as_ref()
                        .ok_or_else(|| {
                            WenlanError::Conflict("repair_apply_recovery_required".to_string())
                        })?
                        .symlink_metadata(&self.quarantine_name)?,
                )?;
                Ok::<_, WenlanError>(source == stage && stage == quarantine)
            })();
            identities.unwrap_or(false)
        } else {
            false
        };
        Ok(
            if state == self.original_state
                && source_original
                && (stage_absent || stage_empty)
                && quarantine_absent
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::Original
            } else if state == self.original_state
                && source_original
                && stage_empty
                && quarantine_absent
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::PreparedStage
            } else if state == self.original_state
                && source_original
                && (stage_absent || stage_empty)
                && quarantine_absent
                && staged_quarantine_absent
                && self.orphaned_baseline.is_none()
                && self.orphaned.is_some()
            {
                StalePageProjectionRecoveryState::OriginalCleanup
            } else if state == self.original_state
                && source_original
                && (stage_absent || stage_empty)
                && quarantine_original
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::AfterLink
            } else if state == self.original_state
                && source_absent
                && stage_original
                && quarantine_original
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::AfterStage
            } else if state == self.original_state
                && source_original
                && stage_original
                && quarantine_original
                && staged_quarantine_absent
                && restored_links_match
            {
                StalePageProjectionRecoveryState::RestoredSource
            } else if state == self.original_state
                && source_absent
                && (stage_absent || stage_empty)
                && quarantine_original
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::AfterUnlink
            } else if state == self.post_state
                && source_absent
                && stage_original
                && quarantine_original
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::PostStaged
            } else if state == self.post_state
                && source_absent
                && (stage_absent || stage_empty)
                && quarantine_original
                && staged_quarantine_absent
            {
                StalePageProjectionRecoveryState::Post
            } else if state == self.original_state
                && source_original
                && stage_empty
                && quarantine_absent
                && staged_quarantine_original
            {
                StalePageProjectionRecoveryState::QuarantineStaged
            } else {
                StalePageProjectionRecoveryState::Unknown
            },
        )
    }

    fn restore_original(&mut self) -> Result<(), WenlanError> {
        let state = self.classify()?;
        let mut read_budget = RepairReadBudget::new();
        if !matches!(
            state,
            StalePageProjectionRecoveryState::PreparedStage
                | StalePageProjectionRecoveryState::AfterLink
                | StalePageProjectionRecoveryState::AfterStage
                | StalePageProjectionRecoveryState::RestoredSource
                | StalePageProjectionRecoveryState::AfterUnlink
                | StalePageProjectionRecoveryState::PostStaged
                | StalePageProjectionRecoveryState::Post
                | StalePageProjectionRecoveryState::QuarantineStaged
                | StalePageProjectionRecoveryState::OriginalCleanup
        ) {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        if matches!(
            state,
            StalePageProjectionRecoveryState::Post | StalePageProjectionRecoveryState::PostStaged
        ) {
            write_state_atomically_if_matches(
                &self.wenlan,
                self.stage.as_ref().ok_or_else(|| {
                    WenlanError::Conflict("repair_apply_recovery_required".to_string())
                })?,
                &self.post_state,
                &self.original_state,
                self.state_mode,
                &mut read_budget,
            )?;
            #[cfg(unix)]
            {
                sync_dir_capability(&self.wenlan)?;
                sync_dir_capability(&self.root)?;
            }
            let expected = if state == StalePageProjectionRecoveryState::PostStaged {
                StalePageProjectionRecoveryState::AfterStage
            } else {
                StalePageProjectionRecoveryState::AfterUnlink
            };
            if self.classify()? != expected {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
        }
        if matches!(
            state,
            StalePageProjectionRecoveryState::AfterStage
                | StalePageProjectionRecoveryState::PostStaged
        ) {
            let stage = self.stage.as_ref().ok_or_else(|| {
                WenlanError::Conflict("repair_apply_recovery_required".to_string())
            })?;
            stage
                .directory
                .hard_link(
                    PROJECTION_UNLINK_STAGE_FILE,
                    &self.source_parent,
                    &self.source_name,
                )
                .map_err(|error| {
                    WenlanError::Io(std::io::Error::new(
                        error.kind(),
                        format!("repair source recovery no-clobber hard link failed: {error}"),
                    ))
                })?;
            let mut stage_budget = RepairReadBudget::new();
            let (source, source_identity) = read_regular_identity_nofollow_bounded(
                &self.source_parent,
                &self.source_name,
                &mut stage_budget,
            )?;
            let (stage_bytes, stage_identity) = read_regular_identity_nofollow_bounded(
                &stage.directory,
                OsStr::new(PROJECTION_UNLINK_STAGE_FILE),
                &mut stage_budget,
            )?;
            if source != self.source_bytes
                || stage_bytes != self.source_bytes
                || source_identity != stage_identity
            {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
            stage.directory.remove_file(PROJECTION_UNLINK_STAGE_FILE)?;
            #[cfg(unix)]
            {
                sync_dir_capability(&self.source_parent)?;
                sync_dir_capability(&self.wenlan)?;
                sync_dir_capability(&self.root)?;
            }
            if self.classify()? != StalePageProjectionRecoveryState::AfterLink {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
        }
        if state == StalePageProjectionRecoveryState::RestoredSource {
            let stage = self.stage.as_ref().ok_or_else(|| {
                WenlanError::Conflict("repair_apply_recovery_required".to_string())
            })?;
            let source_identity =
                projection_file_identity(&self.source_parent.symlink_metadata(&self.source_name)?)?;
            let stage_identity = projection_file_identity(
                &stage
                    .directory
                    .symlink_metadata(PROJECTION_UNLINK_STAGE_FILE)?,
            )?;
            let orphaned = self.orphaned.as_ref().ok_or_else(|| {
                WenlanError::Conflict("repair_apply_recovery_required".to_string())
            })?;
            let quarantine_identity =
                projection_file_identity(&orphaned.symlink_metadata(&self.quarantine_name)?)?;
            if source_identity != stage_identity || stage_identity != quarantine_identity {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
            stage.directory.remove_file(PROJECTION_UNLINK_STAGE_FILE)?;
            #[cfg(unix)]
            sync_dir_capability(&stage.directory)?;
            if self.classify()? != StalePageProjectionRecoveryState::AfterLink {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
        }
        if matches!(
            state,
            StalePageProjectionRecoveryState::AfterUnlink | StalePageProjectionRecoveryState::Post
        ) {
            write_new_file_nofollow(&self.source_parent, &self.source_name, &self.source_bytes)?;
            #[cfg(unix)]
            {
                sync_dir_capability(&self.source_parent)?;
                sync_dir_capability(&self.root)?;
            }
            if self.classify()? != StalePageProjectionRecoveryState::AfterLink {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
        }
        if state == StalePageProjectionRecoveryState::QuarantineStaged {
            self.stage
                .as_ref()
                .ok_or_else(|| WenlanError::Conflict("repair_apply_recovery_required".to_string()))?
                .delete_staged_quarantine_if_exact(&self.source_bytes, None)?;
        } else if !matches!(
            state,
            StalePageProjectionRecoveryState::PreparedStage
                | StalePageProjectionRecoveryState::OriginalCleanup
        ) {
            let orphaned = self.orphaned.as_ref().ok_or_else(|| {
                WenlanError::Conflict("repair_apply_recovery_required".to_string())
            })?;
            self.stage
                .as_ref()
                .ok_or_else(|| WenlanError::Conflict("repair_apply_recovery_required".to_string()))?
                .move_quarantine_for_exact_delete(
                    orphaned,
                    &self.quarantine_name,
                    &self.source_bytes,
                    None,
                )?;
            sync_projection_dirs(&self.root, &self.wenlan, &self.source_parent, orphaned)?;
        }
        if let Some(stage) = self.stage.as_ref() {
            stage.remove_if_empty(&self.wenlan)?;
        }
        Ok(())
    }
}

fn orphaned_baseline_nofollow(
    orphaned: &Dir,
    excluded: Option<&OsStr>,
    budget: &mut RepairReadBudget,
) -> Result<Vec<(String, String)>, WenlanError> {
    let mut baseline = Vec::new();
    for entry in orphaned.entries()? {
        let entry = entry?;
        let name = entry.file_name();
        charge_repair_read_budget(
            budget,
            ORPHAN_BASELINE_ENTRY_BUDGET_BYTES
                .checked_add(os_str_storage_len(&name)?)
                .ok_or_else(repair_projection_rollback_too_large)?,
        )?;
        if excluded.is_some_and(|excluded| excluded == name) {
            continue;
        }
        let name_string = name
            .to_str()
            .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?
            .to_string();
        baseline.push((
            name_string,
            hash_regular_nofollow_bounded(orphaned, &name, budget)?,
        ));
    }
    baseline.sort();
    Ok(baseline)
}

fn rollback_row_bytes(
    rollback: &crate::repair::StoredRollbackArtifact,
    path: &str,
) -> Result<Vec<u8>, WenlanError> {
    let row = rollback
        .rows
        .iter()
        .find(|row| row[0] == path && row[1] == "file_hex")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    hex::decode(&row[2])
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))
}

pub(crate) fn projection_unlink_stage_name(owner: &str) -> OsString {
    format!(
        ".projection-unlink-{}",
        hex::encode(sha2::Sha256::digest(owner.as_bytes()))
    )
    .into()
}

#[derive(Serialize)]
struct ProjectionStageOwner<'a> {
    format_version: u32,
    manifest_id: &'a str,
    page_id: &'a str,
    source_path: &'a str,
    source_digest: String,
}

pub(crate) fn projection_stage_owner_bytes(
    manifest_id: &str,
    page_id: &str,
    source_path: &str,
    source_bytes: &[u8],
) -> Result<Vec<u8>, WenlanError> {
    serde_json::to_vec(&ProjectionStageOwner {
        format_version: 1,
        manifest_id,
        page_id,
        source_path,
        source_digest: hex::encode(sha2::Sha256::digest(source_bytes)),
    })
    .map_err(WenlanError::from)
}

struct OwnedProjectionStage {
    directory: Dir,
    owner_bytes: Vec<u8>,
}

enum StageContents {
    Absent,
    Present {
        source: Option<Vec<u8>>,
        rollback_quarantine: Option<Vec<u8>>,
        state: Option<Vec<u8>>,
    },
}

fn inspect_owned_projection_stage(
    stage: Option<&OwnedProjectionStage>,
    budget: &mut RepairReadBudget,
) -> Result<StageContents, WenlanError> {
    let Some(stage) = stage else {
        return Ok(StageContents::Absent);
    };
    let mut source = None;
    let mut rollback_quarantine = None;
    let mut state = None;
    let mut owner_seen = false;
    for entry in stage.directory.entries()? {
        let entry = entry?;
        match entry.file_name().to_str() {
            Some(PROJECTION_STAGE_OWNER_FILE) if !owner_seen => {
                if read_regular_nofollow(
                    &stage.directory,
                    OsStr::new(PROJECTION_STAGE_OWNER_FILE),
                    budget,
                )? != stage.owner_bytes
                {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                owner_seen = true;
            }
            Some(PROJECTION_UNLINK_STAGE_FILE) if source.is_none() => {
                source = Some(read_regular_nofollow(
                    &stage.directory,
                    OsStr::new(PROJECTION_UNLINK_STAGE_FILE),
                    budget,
                )?);
            }
            Some(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE) if rollback_quarantine.is_none() => {
                rollback_quarantine = Some(read_regular_nofollow(
                    &stage.directory,
                    OsStr::new(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE),
                    budget,
                )?);
            }
            Some(PROJECTION_STATE_STAGE_FILE) if state.is_none() => {
                state = Some(read_regular_nofollow(
                    &stage.directory,
                    OsStr::new(PROJECTION_STATE_STAGE_FILE),
                    budget,
                )?);
            }
            _ => return Err(WenlanError::Conflict("repair_target_stale".to_string())),
        }
    }
    if !owner_seen {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(StageContents::Present {
        source,
        rollback_quarantine,
        state,
    })
}

impl OwnedProjectionStage {
    fn create_noclobber(
        wenlan: &Dir,
        name: OsString,
        owner_bytes: Vec<u8>,
    ) -> Result<Self, WenlanError> {
        let create_result = {
            #[cfg(unix)]
            {
                use cap_std::fs::DirBuilderExt as _;
                let mut builder = cap_std::fs::DirBuilder::new();
                builder.mode(0o700);
                wenlan.create_dir_with(Path::new(&name), &builder)
            }
            #[cfg(not(unix))]
            {
                wenlan.create_dir(Path::new(&name))
            }
        };
        match create_result {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()))
            }
            Err(error) => return Err(WenlanError::Io(error)),
        }

        let directory = wenlan.open_dir_nofollow(Path::new(&name))?;
        ensure_orphaned_private(&directory)?;
        write_new_file_nofollow(
            &directory,
            OsStr::new(PROJECTION_STAGE_OWNER_FILE),
            &owner_bytes,
        )?;
        #[cfg(unix)]
        sync_dir_capability(&directory)?;
        Ok(Self {
            directory,
            owner_bytes,
        })
    }

    fn open_existing(
        wenlan: &Dir,
        name: OsString,
        owner_bytes: Vec<u8>,
    ) -> Result<Option<Self>, WenlanError> {
        match wenlan.symlink_metadata(Path::new(&name)) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(WenlanError::Io(error)),
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                let directory = wenlan.open_dir_nofollow(Path::new(&name))?;
                ensure_orphaned_private(&directory)?;
                let mut budget = RepairReadBudget::new();
                if read_regular_nofollow(
                    &directory,
                    OsStr::new(PROJECTION_STAGE_OWNER_FILE),
                    &mut budget,
                )? != owner_bytes
                {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                Ok(Some(Self {
                    directory,
                    owner_bytes,
                }))
            }
            Ok(_) => Err(WenlanError::Conflict("repair_target_stale".to_string())),
        }
    }

    fn remove_if_empty(&self, _wenlan: &Dir) -> Result<(), WenlanError> {
        let mut budget = RepairReadBudget::new();
        if !matches!(
            inspect_owned_projection_stage(Some(self), &mut budget)?,
            StageContents::Present {
                source: None,
                rollback_quarantine: None,
                state: None,
            }
        ) {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        #[cfg(unix)]
        sync_dir_capability(&self.directory)?;
        Ok(())
    }

    fn delete_staged_quarantine_if_exact(
        &self,
        expected_bytes: &[u8],
        expected_identity: Option<ProjectionFileIdentity>,
    ) -> Result<(), WenlanError> {
        let mut budget = RepairReadBudget::new();
        let (bytes, identity) = read_regular_identity_nofollow_bounded(
            &self.directory,
            OsStr::new(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE),
            &mut budget,
        )?;
        if bytes != expected_bytes || expected_identity.is_some_and(|expected| expected != identity)
        {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        self.directory
            .remove_file(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE)?;
        #[cfg(unix)]
        sync_dir_capability(&self.directory)?;
        Ok(())
    }

    fn move_quarantine_for_exact_delete(
        &self,
        orphaned: &Dir,
        quarantine_name: &OsStr,
        expected_bytes: &[u8],
        expected_identity: Option<ProjectionFileIdentity>,
    ) -> Result<(), WenlanError> {
        match self
            .directory
            .symlink_metadata(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE)
        {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ))
            }
            Err(error) => return Err(WenlanError::Io(error)),
        }
        orphaned.rename(
            Path::new(quarantine_name),
            &self.directory,
            Path::new(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE),
        )?;
        #[cfg(unix)]
        {
            sync_dir_capability(orphaned)?;
            sync_dir_capability(&self.directory)?;
        }

        let mut budget = RepairReadBudget::new();
        let moved = read_regular_identity_nofollow_bounded(
            &self.directory,
            OsStr::new(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE),
            &mut budget,
        );
        let exact = matches!(
            &moved,
            Ok((bytes, identity))
                if bytes == expected_bytes
                    && expected_identity.is_none_or(|expected| expected == *identity)
        );
        if !exact {
            if let Ok((_, moved_identity)) = moved {
                match self.directory.hard_link(
                    PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE,
                    orphaned,
                    quarantine_name,
                ) {
                    Ok(()) => {
                        let restored =
                            projection_file_identity(&orphaned.symlink_metadata(quarantine_name)?)?;
                        if restored == moved_identity {
                            self.directory
                                .remove_file(PROJECTION_ROLLBACK_QUARANTINE_STAGE_FILE)?;
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(WenlanError::Io(error)),
                }
            }
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        self.delete_staged_quarantine_if_exact(expected_bytes, expected_identity)
    }
}

pub(crate) struct PinnedStalePageProjection<'lock> {
    root: Dir,
    wenlan: Dir,
    source_parent: Dir,
    source_name: OsString,
    source_path: String,
    stage: Option<OwnedProjectionStage>,
    orphaned: Option<Dir>,
    quarantine_name: OsString,
    quarantine_path: String,
    expected_state_bytes: Vec<u8>,
    next_state_bytes: Vec<u8>,
    expected_source_bytes: Vec<u8>,
    expected_source_identity: ProjectionFileIdentity,
    expected_orphaned_baseline: Vec<(String, String)>,
    state_mode: ProjectionStateMode,
    mutation_started: bool,
    _lock: std::marker::PhantomData<&'lock ProjectionCapabilities>,
}

impl PinnedStalePageProjection<'_> {
    fn orphaned(&self) -> &Dir {
        self.orphaned
            .as_ref()
            .expect("pinned orphaned directory retained until rollback")
    }

    fn stage(&self) -> Result<&OwnedProjectionStage, WenlanError> {
        self.stage
            .as_ref()
            .ok_or_else(|| WenlanError::Conflict("repair_apply_recovery_required".to_string()))
    }

    pub(crate) fn mutation_started(&self) -> bool {
        self.mutation_started
    }

    pub(crate) fn quarantine(&mut self) -> Result<(), WenlanError> {
        self.quarantine_inner(|| Ok(()), || Ok(()), || Ok(()), || Ok(()), || Ok(()))
    }

    #[cfg(test)]
    pub(crate) fn quarantine_with_hooks<B, S>(
        &mut self,
        before_link: B,
        before_source_stage: S,
    ) -> Result<(), WenlanError>
    where
        B: FnOnce() -> Result<(), WenlanError>,
        S: FnOnce() -> Result<(), WenlanError>,
    {
        self.quarantine_inner(
            before_link,
            || Ok(()),
            before_source_stage,
            || Ok(()),
            || Ok(()),
        )
    }

    #[cfg(test)]
    pub(crate) fn quarantine_with_after_link<A>(&mut self, after_link: A) -> Result<(), WenlanError>
    where
        A: FnOnce() -> Result<(), WenlanError>,
    {
        self.quarantine_inner(|| Ok(()), after_link, || Ok(()), || Ok(()), || Ok(()))
    }

    #[cfg(test)]
    pub(crate) fn quarantine_with_after_source_stage<A>(
        &mut self,
        after_source_stage: A,
    ) -> Result<(), WenlanError>
    where
        A: FnOnce() -> Result<(), WenlanError>,
    {
        self.quarantine_inner(
            || Ok(()),
            || Ok(()),
            || Ok(()),
            after_source_stage,
            || Ok(()),
        )
    }

    #[cfg(test)]
    pub(crate) fn quarantine_with_before_state_swap<B>(
        &mut self,
        before_state_swap: B,
    ) -> Result<(), WenlanError>
    where
        B: FnOnce() -> Result<(), WenlanError>,
    {
        self.quarantine_inner(
            || Ok(()),
            || Ok(()),
            || Ok(()),
            || Ok(()),
            before_state_swap,
        )
    }

    fn quarantine_inner<B, L, S, A, T>(
        &mut self,
        before_link: B,
        after_link: L,
        before_source_stage: S,
        after_source_stage: A,
        before_state_swap: T,
    ) -> Result<(), WenlanError>
    where
        B: FnOnce() -> Result<(), WenlanError>,
        L: FnOnce() -> Result<(), WenlanError>,
        S: FnOnce() -> Result<(), WenlanError>,
        A: FnOnce() -> Result<(), WenlanError>,
        T: FnOnce() -> Result<(), WenlanError>,
    {
        let mut initial_budget = RepairReadBudget::new();
        let (state_bytes, current_mode) = read_state_nofollow(&self.wenlan, &mut initial_budget)?;
        if state_bytes != self.expected_state_bytes || current_mode != self.state_mode {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let (source_bytes, source_identity) = read_regular_identity_nofollow_bounded(
            &self.source_parent,
            &self.source_name,
            &mut initial_budget,
        )?;
        if source_bytes != self.expected_source_bytes
            || source_identity != self.expected_source_identity
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        before_link()?;
        let mut mutation_budget = RepairReadBudget::new();
        if orphaned_baseline_nofollow(self.orphaned(), None, &mut mutation_budget)?
            != self.expected_orphaned_baseline
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let (source_bytes, source_identity) = read_regular_identity_nofollow_bounded(
            &self.source_parent,
            &self.source_name,
            &mut mutation_budget,
        )?;
        if source_bytes != self.expected_source_bytes
            || source_identity != self.expected_source_identity
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        match self.orphaned().symlink_metadata(&self.quarantine_name) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => return Err(WenlanError::Conflict("repair_target_stale".to_string())),
            Err(error) => return Err(WenlanError::Io(error)),
        }
        self.source_parent
            .hard_link(&self.source_name, self.orphaned(), &self.quarantine_name)
            .map_err(|error| {
                WenlanError::Io(std::io::Error::new(
                    error.kind(),
                    format!("repair quarantine no-clobber hard link failed: {error}"),
                ))
            })?;
        self.mutation_started = true;
        after_link()?;
        let mut linked_budget = RepairReadBudget::new();
        let linked = read_regular_identity_nofollow_bounded(
            self.orphaned(),
            &self.quarantine_name,
            &mut linked_budget,
        );
        let source_after_link = self
            .source_parent
            .symlink_metadata(&self.source_name)
            .and_then(|metadata| {
                projection_file_identity(&metadata)
                    .map_err(|error| std::io::Error::other(error.to_string()))
            });
        let link_is_exact = matches!(
            (&linked, &source_after_link),
            (Ok((linked_bytes, linked_identity)), Ok(source_identity))
                if linked_bytes == &self.expected_source_bytes
                    && *linked_identity == self.expected_source_identity
                    && *source_identity == self.expected_source_identity
        );
        if !link_is_exact {
            if let Ok((linked_bytes, linked_identity)) = &linked {
                if linked_bytes == &self.expected_source_bytes
                    && *linked_identity == self.expected_source_identity
                {
                    self.stage()?.move_quarantine_for_exact_delete(
                        self.orphaned(),
                        &self.quarantine_name,
                        &self.expected_source_bytes,
                        Some(self.expected_source_identity),
                    )?;
                    self.mutation_started = false;
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
            }
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        let mut final_source_budget = RepairReadBudget::new();
        let (source_bytes, source_identity) = read_regular_identity_nofollow_bounded(
            &self.source_parent,
            &self.source_name,
            &mut final_source_budget,
        )?;
        if source_bytes != self.expected_source_bytes
            || source_identity != self.expected_source_identity
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        before_source_stage()?;
        if !matches!(
            inspect_owned_projection_stage(self.stage.as_ref(), &mut final_source_budget)?,
            StageContents::Present {
                source: None,
                rollback_quarantine: None,
                state: None,
            }
        ) {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        self.source_parent.rename(
            Path::new(&self.source_name),
            &self.stage()?.directory,
            Path::new(PROJECTION_UNLINK_STAGE_FILE),
        )?;
        #[cfg(unix)]
        {
            sync_dir_capability(&self.source_parent)?;
            sync_dir_capability(&self.stage()?.directory)?;
            sync_dir_capability(&self.root)?;
        }
        after_source_stage()?;
        let mut staged_budget = RepairReadBudget::new();
        let staged = read_regular_identity_nofollow_bounded(
            &self.stage()?.directory,
            OsStr::new(PROJECTION_UNLINK_STAGE_FILE),
            &mut staged_budget,
        );
        let source_after_stage = read_optional_regular_nofollow(
            &self.source_parent,
            &self.source_name,
            &mut staged_budget,
        );
        let stage_is_exact = matches!(
            (&staged, &source_after_stage),
            (Ok((staged_bytes, staged_identity)), Ok(None))
                if staged_bytes == &self.expected_source_bytes
                    && *staged_identity == self.expected_source_identity
        );
        if !stage_is_exact {
            self.restore_failed_source_stage(staged, source_after_stage)?;
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let mut state_budget = RepairReadBudget::new();
        write_state_atomically_if_matches_with_hook(
            &self.wenlan,
            self.stage()?,
            &self.expected_state_bytes,
            &self.next_state_bytes,
            self.state_mode,
            &mut state_budget,
            before_state_swap,
        )?;
        let staged_metadata = self
            .stage()?
            .directory
            .symlink_metadata(PROJECTION_UNLINK_STAGE_FILE)?;
        if projection_file_identity(&staged_metadata)? != self.expected_source_identity {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        self.stage()?
            .directory
            .remove_file(PROJECTION_UNLINK_STAGE_FILE)?;
        self.stage()?.remove_if_empty(&self.wenlan)?;
        sync_projection_dirs(
            &self.root,
            &self.wenlan,
            &self.source_parent,
            self.orphaned(),
        )?;
        Ok(())
    }

    fn restore_failed_source_stage(
        &mut self,
        staged: Result<(Vec<u8>, ProjectionFileIdentity), WenlanError>,
        source_after_stage: Result<Option<Vec<u8>>, WenlanError>,
    ) -> Result<(), WenlanError> {
        let (staged_bytes, staged_identity) = staged?;
        let source_after_stage = source_after_stage?;
        match source_after_stage {
            None => {
                self.stage()?
                    .directory
                    .hard_link(
                        PROJECTION_UNLINK_STAGE_FILE,
                        &self.source_parent,
                        &self.source_name,
                    )
                    .map_err(|error| {
                        WenlanError::Io(std::io::Error::new(
                            error.kind(),
                            format!("repair source restore no-clobber hard link failed: {error}"),
                        ))
                    })?;
                let mut restored_budget = RepairReadBudget::new();
                let (restored_bytes, restored_identity) = read_regular_identity_nofollow_bounded(
                    &self.source_parent,
                    &self.source_name,
                    &mut restored_budget,
                )?;
                if restored_bytes != staged_bytes || restored_identity != staged_identity {
                    return Err(WenlanError::Conflict(
                        "repair_apply_recovery_required".to_string(),
                    ));
                }
            }
            Some(_) => {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ))
            }
        }
        let stage_metadata = self
            .stage()?
            .directory
            .symlink_metadata(PROJECTION_UNLINK_STAGE_FILE)?;
        if projection_file_identity(&stage_metadata)? != staged_identity {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        self.stage()?
            .directory
            .remove_file(PROJECTION_UNLINK_STAGE_FILE)?;

        self.stage()?.move_quarantine_for_exact_delete(
            self.orphaned(),
            &self.quarantine_name,
            &self.expected_source_bytes,
            Some(self.expected_source_identity),
        )?;
        self.stage()?.remove_if_empty(&self.wenlan)?;
        sync_projection_dirs(
            &self.root,
            &self.wenlan,
            &self.source_parent,
            self.orphaned(),
        )?;
        self.mutation_started = false;
        Ok(())
    }

    pub(crate) fn restore_snapshot(
        &mut self,
        rollback: &crate::repair::StoredRollbackArtifact,
    ) -> Result<(), WenlanError> {
        let (source_path, quarantine_path) = crate::repair::stale_page_projection_paths(rollback)?;
        if source_path != self.source_path || quarantine_path != self.quarantine_path {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        let state = rollback
            .rows
            .iter()
            .find(|row| row[0] == ".wenlan/state.json" && row[1] == "file_hex")
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
        let source = rollback
            .rows
            .iter()
            .find(|row| row[0] == source_path && row[1] == "file_hex")
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
        let state_bytes = hex::decode(&state[2]).map_err(|_| {
            WenlanError::Validation("repair_projection_rollback_invalid".to_string())
        })?;
        let source_bytes = hex::decode(&source[2]).map_err(|_| {
            WenlanError::Validation("repair_projection_rollback_invalid".to_string())
        })?;

        let mut read_budget = RepairReadBudget::new();
        let (current_state, current_mode) = read_state_nofollow(&self.wenlan, &mut read_budget)?;
        if current_mode != self.state_mode
            || (current_state != state_bytes && current_state != self.next_state_bytes)
        {
            return Err(WenlanError::Conflict(
                "repair_projection_rollback_state_collision".to_string(),
            ));
        }
        let mut source_budget = RepairReadBudget::new();
        let source_exists = match read_optional_regular_nofollow(
            &self.source_parent,
            &self.source_name,
            &mut source_budget,
        ) {
            Ok(Some(bytes)) if bytes == source_bytes => true,
            Ok(None) => false,
            _ => {
                return Err(WenlanError::Conflict(
                    "repair_projection_rollback_source_collision".to_string(),
                ))
            }
        };
        let mut stage_budget = RepairReadBudget::new();
        let stage_identity =
            match inspect_owned_projection_stage(self.stage.as_ref(), &mut stage_budget)? {
                StageContents::Absent
                | StageContents::Present {
                    source: None,
                    rollback_quarantine: None,
                    state: None,
                } => None,
                StageContents::Present {
                    source: Some(bytes),
                    rollback_quarantine: None,
                    state: None,
                } if bytes == source_bytes => {
                    let metadata = self
                        .stage()?
                        .directory
                        .symlink_metadata(PROJECTION_UNLINK_STAGE_FILE)?;
                    Some(projection_file_identity(&metadata)?)
                }
                StageContents::Present { .. } => {
                    return Err(WenlanError::Conflict(
                        "repair_projection_rollback_source_collision".to_string(),
                    ))
                }
            };
        let stage_exists = stage_identity.is_some();
        let stage_prepared = self.stage.is_some();
        let mut quarantine_budget = RepairReadBudget::new();
        let quarantine_exists = match read_optional_regular_nofollow(
            self.orphaned(),
            &self.quarantine_name,
            &mut quarantine_budget,
        ) {
            Ok(Some(bytes)) if bytes == source_bytes => true,
            Ok(None) => false,
            _ => {
                return Err(WenlanError::Conflict(
                    "repair_projection_rollback_quarantine_collision".to_string(),
                ))
            }
        };
        let original = current_state == state_bytes;
        let post = current_state == self.next_state_bytes;
        if !matches!(
            (
                original,
                post,
                source_exists,
                quarantine_exists,
                stage_exists,
                stage_prepared
            ),
            (true, false, true, false, false, true)
                | (true, false, true, true, false, true)
                | (true, false, false, true, false, false)
                | (true, false, false, true, false, true)
                | (false, true, false, true, false, false)
                | (false, true, false, true, false, true)
                | (true, false, false, true, true, true)
                | (false, true, false, true, true, true)
        ) {
            return Err(WenlanError::Conflict(
                "repair_projection_rollback_state_collision".to_string(),
            ));
        }
        if post {
            write_state_atomically_if_matches(
                &self.wenlan,
                self.stage()?,
                &self.next_state_bytes,
                &state_bytes,
                self.state_mode,
                &mut read_budget,
            )?;
            #[cfg(unix)]
            {
                sync_dir_capability(&self.wenlan)?;
                sync_dir_capability(&self.root)?;
            }
        }
        if !source_exists {
            if let Some(stage_identity) = stage_identity {
                self.stage()?
                    .directory
                    .hard_link(
                        PROJECTION_UNLINK_STAGE_FILE,
                        &self.source_parent,
                        &self.source_name,
                    )
                    .map_err(|error| {
                        WenlanError::Io(std::io::Error::new(
                            error.kind(),
                            format!("repair source rollback no-clobber hard link failed: {error}"),
                        ))
                    })?;
                let restored = self.source_parent.symlink_metadata(&self.source_name)?;
                if projection_file_identity(&restored)? != stage_identity {
                    return Err(WenlanError::Conflict(
                        "repair_projection_rollback_source_collision".to_string(),
                    ));
                }
                self.stage()?
                    .directory
                    .remove_file(PROJECTION_UNLINK_STAGE_FILE)?;
            } else {
                write_new_file_nofollow(&self.source_parent, &self.source_name, &source_bytes)?;
            }
            #[cfg(unix)]
            {
                sync_dir_capability(&self.source_parent)?;
                sync_dir_capability(&self.root)?;
            }
        }
        if quarantine_exists {
            self.stage()?.move_quarantine_for_exact_delete(
                self.orphaned(),
                &self.quarantine_name,
                &source_bytes,
                Some(self.expected_source_identity),
            )?;
        }
        sync_projection_dirs(
            &self.root,
            &self.wenlan,
            &self.source_parent,
            self.orphaned(),
        )?;
        if let Some(stage) = self.stage.as_ref() {
            stage.remove_if_empty(&self.wenlan)?;
        }

        self.mutation_started = false;
        Ok(())
    }
}

fn open_relative_parent_nofollow(
    root: &Dir,
    relative: &Path,
) -> Result<(Dir, OsString), WenlanError> {
    let mut components = relative.components().peekable();
    let mut directory = root.try_clone()?;
    while let Some(component) = components.next() {
        let Component::Normal(component) = component else {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        };
        if components.peek().is_none() {
            return Ok((directory, component.to_os_string()));
        }
        directory = directory.open_dir_nofollow(Path::new(component))?;
    }
    Err(WenlanError::Conflict("repair_target_stale".to_string()))
}

fn create_projection_root_nofollow(path: &Path) -> Result<(), WenlanError> {
    let parent = path
        .parent()
        .ok_or_else(|| WenlanError::Validation("repair_projection_root_invalid".to_string()))?;
    let basename = path
        .file_name()
        .ok_or_else(|| WenlanError::Validation("repair_projection_root_invalid".to_string()))?;
    let parent = Dir::open_ambient_dir(parent, cap_std::ambient_authority())?;
    match parent.symlink_metadata(Path::new(basename)) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(WenlanError::Validation(
            "repair_projection_root_invalid".to_string(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match parent.create_dir(Path::new(basename)) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let metadata = parent.symlink_metadata(Path::new(basename))?;
                    if metadata.is_dir() && !metadata.file_type().is_symlink() {
                        Ok(())
                    } else {
                        Err(WenlanError::Validation(
                            "repair_projection_root_invalid".to_string(),
                        ))
                    }
                }
                Err(error) => Err(WenlanError::Io(error)),
            }
        }
        Err(error) => Err(WenlanError::Io(error)),
    }
}

fn repair_projection_rollback_too_large() -> WenlanError {
    WenlanError::Validation("repair_projection_rollback_too_large".to_string())
}

fn charge_repair_read_budget(
    budget: &mut RepairReadBudget,
    amount: u64,
) -> Result<(), WenlanError> {
    budget.remaining = budget
        .remaining
        .checked_sub(amount)
        .ok_or_else(repair_projection_rollback_too_large)?;
    Ok(())
}

fn os_str_storage_len(value: &OsStr) -> Result<u64, WenlanError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        u64::try_from(value.as_bytes().len()).map_err(|_| repair_projection_rollback_too_large())
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt as _;
        u64::try_from(value.encode_wide().count())
            .ok()
            .and_then(|units| units.checked_mul(2))
            .ok_or_else(repair_projection_rollback_too_large)
    }
    #[cfg(not(any(unix, windows)))]
    {
        u64::try_from(value.to_string_lossy().len())
            .map_err(|_| repair_projection_rollback_too_large())
    }
}

fn regular_read_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    options
}

fn read_regular_nofollow(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
) -> Result<Vec<u8>, WenlanError> {
    read_optional_regular_nofollow(directory, name, budget)?
        .ok_or_else(|| WenlanError::Io(std::io::Error::from(std::io::ErrorKind::NotFound)))
}

fn read_optional_regular_nofollow(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
) -> Result<Option<Vec<u8>>, WenlanError> {
    read_optional_regular_nofollow_bounded(directory, name, budget, || Ok(()))
}

fn read_optional_regular_nofollow_bounded(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
    after_metadata: impl FnOnce() -> Result<(), WenlanError>,
) -> Result<Option<Vec<u8>>, WenlanError> {
    let mut file = match directory.open_with(Path::new(name), &regular_read_options()) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(WenlanError::Io(error)),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    if metadata.len() > budget.remaining {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_too_large".to_string(),
        ));
    }
    after_metadata()?;
    let max_bytes = usize::try_from(budget.remaining)
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_too_large".to_string()))?;
    let initial_capacity = usize::try_from(metadata.len())
        .unwrap_or(max_bytes)
        .min(max_bytes);
    let mut bytes = Vec::with_capacity(initial_capacity);
    let read_limit = u64::try_from(max_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut limited = (&mut file).take(read_limit);
    let mut buffer = [0_u8; 8192];
    loop {
        let read = limited.read(&mut buffer)?;
        if read == 0 {
            return Ok(Some(bytes));
        }
        if bytes
            .len()
            .checked_add(read)
            .is_none_or(|length| length > max_bytes)
        {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_too_large".to_string(),
            ));
        }
        budget.remaining = budget
            .remaining
            .checked_sub(u64::try_from(read).map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_too_large".to_string())
            })?)
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_too_large".to_string())
            })?;
        bytes.extend_from_slice(&buffer[..read]);
    }
}

fn read_regular_identity_nofollow_bounded(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
) -> Result<(Vec<u8>, ProjectionFileIdentity), WenlanError> {
    let mut file = directory.open_with(Path::new(name), &regular_read_options())?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > budget.remaining {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let identity = projection_file_identity(&metadata)?;
    let max_bytes = usize::try_from(budget.remaining)
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_too_large".to_string()))?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .unwrap_or(max_bytes)
            .min(max_bytes),
    );
    let mut limited = (&mut file).take(budget.remaining.saturating_add(1));
    let mut buffer = [0_u8; 8192];
    loop {
        let read = limited.read(&mut buffer)?;
        if read == 0 {
            return Ok((bytes, identity));
        }
        if bytes
            .len()
            .checked_add(read)
            .is_none_or(|length| length > max_bytes)
        {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_too_large".to_string(),
            ));
        }
        budget.remaining = budget
            .remaining
            .checked_sub(u64::try_from(read).map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_too_large".to_string())
            })?)
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_too_large".to_string())
            })?;
        bytes.extend_from_slice(&buffer[..read]);
    }
}

fn hash_regular_nofollow_bounded(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
) -> Result<String, WenlanError> {
    let mut file = directory.open_with(Path::new(name), &regular_read_options())?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    if metadata.len() > budget.remaining {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_too_large".to_string(),
        ));
    }
    let read_limit = budget.remaining.saturating_add(1);
    let mut limited = (&mut file).take(read_limit);
    let mut digest = sha2::Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = limited.read(&mut buffer)?;
        if read == 0 {
            return Ok(hex::encode(digest.finalize()));
        }
        let read = u64::try_from(read).map_err(|_| {
            WenlanError::Validation("repair_projection_rollback_too_large".to_string())
        })?;
        budget.remaining = budget.remaining.checked_sub(read).ok_or_else(|| {
            WenlanError::Validation("repair_projection_rollback_too_large".to_string())
        })?;
        digest.update(&buffer[..usize::try_from(read).unwrap_or(buffer.len())]);
    }
}

fn read_state_nofollow(
    wenlan: &Dir,
    budget: &mut RepairReadBudget,
) -> Result<(Vec<u8>, ProjectionStateMode), WenlanError> {
    let (bytes, mode, _) =
        read_projection_state_identity_nofollow(wenlan, OsStr::new("state.json"), budget)?;
    Ok((bytes, mode))
}

fn read_optional_projection_state_identity_nofollow(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
) -> Result<Option<(Vec<u8>, ProjectionStateMode, ProjectionFileIdentity)>, WenlanError> {
    match directory.open_with(Path::new(name), &regular_read_options()) {
        Ok(file) => read_projection_state_identity_from_file(file, budget).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(WenlanError::Io(error)),
    }
}

fn read_projection_state_identity_nofollow(
    directory: &Dir,
    name: &OsStr,
    budget: &mut RepairReadBudget,
) -> Result<(Vec<u8>, ProjectionStateMode, ProjectionFileIdentity), WenlanError> {
    let file = directory.open_with(Path::new(name), &regular_read_options())?;
    read_projection_state_identity_from_file(file, budget)
}

fn read_projection_state_identity_from_file(
    mut file: cap_std::fs::File,
    budget: &mut RepairReadBudget,
) -> Result<(Vec<u8>, ProjectionStateMode, ProjectionFileIdentity), WenlanError> {
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.len() > crate::lint::pages::fs::STATE_MAX_BYTES
        || metadata.len() > budget.remaining
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let max_bytes = budget
        .remaining
        .min(crate::lint::pages::fs::STATE_MAX_BYTES);
    let mut limited = (&mut file).take(max_bytes.saturating_add(1));
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .unwrap_or(0)
            .min(4 * 1024 * 1024),
    );
    let mut buffer = [0_u8; 8192];
    loop {
        let read = limited.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let next = bytes.len().checked_add(read).ok_or_else(|| {
            WenlanError::Validation("repair_projection_rollback_too_large".to_string())
        })?;
        if u64::try_from(next).unwrap_or(u64::MAX) > max_bytes {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_too_large".to_string(),
            ));
        }
        budget.remaining = budget
            .remaining
            .checked_sub(u64::try_from(read).map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_too_large".to_string())
            })?)
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_too_large".to_string())
            })?;
        bytes.extend_from_slice(&buffer[..read]);
    }
    let identity = projection_file_identity(&metadata)?;
    #[cfg(unix)]
    {
        use cap_std::fs::PermissionsExt as _;
        Ok((bytes, metadata.permissions().mode(), identity))
    }
    #[cfg(not(unix))]
    Ok((bytes, (), identity))
}

fn ensure_orphaned_private(orphaned: &Dir) -> Result<(), WenlanError> {
    let metadata = orphaned.dir_metadata()?;
    if !metadata.is_dir() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    #[cfg(unix)]
    {
        use cap_std::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
    }
    Ok(())
}

fn write_new_file_nofollow(directory: &Dir, name: &OsStr, bytes: &[u8]) -> Result<(), WenlanError> {
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = directory.open_with(Path::new(name), &options)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Atomic, symlink-refusing page write: temp file in the same capability
/// directory, fsync, then rename over the target.
///
/// `write_regular_nofollow` truncates the target and writes in place, so a
/// concurrent reader can observe a short or empty page. A rename within one
/// directory is atomic: a reader sees the old bytes or the new ones, never a
/// torn file. `reconcile` repairs the remaining failure mode (temp written,
/// rename never happened), which leaves the target's `origin_version` behind
/// the DB's.
fn write_page_atomically_nofollow(
    directory: &Dir,
    name: &str,
    temporary: &str,
    bytes: &[u8],
) -> Result<(), WenlanError> {
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let result = (|| {
        let mut file = directory.open_with(temporary, &options)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        match directory.open_with(name, &regular_read_options()) {
            Ok(current) => {
                if !current.metadata()?.is_file() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "page_projection_target_invalid",
                    ));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        directory.rename(temporary, directory, name)?;
        Ok::<(), std::io::Error>(())
    })();
    if result.is_err() {
        let _ = directory.remove_file(temporary);
    }
    result.map_err(WenlanError::Io)
}

fn write_regular_nofollow(directory: &Dir, name: &str, bytes: &[u8]) -> Result<(), WenlanError> {
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create(true)
        .truncate(true)
        .follow(FollowSymlinks::No);
    let mut file = directory.open_with(name, &options)?;
    if !file.metadata()?.is_file() {
        return Err(WenlanError::Conflict(
            "page_projection_target_invalid".to_string(),
        ));
    }
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_state_atomically(
    wenlan: &Dir,
    bytes: &[u8],
    mode: ProjectionStateMode,
) -> Result<(), WenlanError> {
    let sequence = PROJECTION_STATE_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = format!(".projection-state-{}-{sequence}.tmp", std::process::id());
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let result = (|| {
        let mut file = wenlan.open_with(&temporary, &options)?;
        #[cfg(unix)]
        {
            use cap_std::fs::PermissionsExt as _;
            file.set_permissions(cap_std::fs::Permissions::from_mode(mode))?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        let current = wenlan.open_with("state.json", &regular_read_options())?;
        if !current.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "projection state is not a regular file",
            ));
        }
        wenlan.rename(&temporary, wenlan, "state.json")?;
        Ok::<(), std::io::Error>(())
    })();
    if result.is_err() {
        let _ = wenlan.remove_file(&temporary);
    }
    result.map_err(WenlanError::Io)
}

fn write_state_atomically_if_matches(
    wenlan: &Dir,
    stage: &OwnedProjectionStage,
    expected: &[u8],
    replacement: &[u8],
    mode: ProjectionStateMode,
    budget: &mut RepairReadBudget,
) -> Result<(), WenlanError> {
    write_state_atomically_if_matches_with_hook(
        wenlan,
        stage,
        expected,
        replacement,
        mode,
        budget,
        || Ok(()),
    )
}

fn write_state_atomically_if_matches_with_hook<B>(
    wenlan: &Dir,
    stage: &OwnedProjectionStage,
    expected: &[u8],
    replacement: &[u8],
    mode: ProjectionStateMode,
    budget: &mut RepairReadBudget,
    before_swap: B,
) -> Result<(), WenlanError>
where
    B: FnOnce() -> Result<(), WenlanError>,
{
    let (current, current_mode, current_identity) =
        read_projection_state_identity_nofollow(wenlan, OsStr::new("state.json"), budget)?;
    if current != expected || current_mode != mode {
        return Err(WenlanError::Conflict(
            "repair_projection_rollback_state_collision".to_string(),
        ));
    }
    let mut stage_budget = RepairReadBudget::new();
    if matches!(
        inspect_owned_projection_stage(Some(stage), &mut stage_budget)?,
        StageContents::Present { state: Some(_), .. }
    ) {
        return Err(WenlanError::Conflict(
            "repair_apply_recovery_required".to_string(),
        ));
    }

    let sequence = PROJECTION_STATE_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = format!(".projection-state-{}-{sequence}.tmp", std::process::id());
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut temporary_file = wenlan.open_with(&temporary, &options)?;
    #[cfg(unix)]
    {
        use cap_std::fs::PermissionsExt as _;
        temporary_file.set_permissions(cap_std::fs::Permissions::from_mode(mode))?;
    }
    temporary_file.write_all(replacement)?;
    temporary_file.sync_all()?;
    let temporary_identity = projection_file_identity(&temporary_file.metadata()?)?;
    drop(temporary_file);

    let result = (|| {
        before_swap()?;
        wenlan.rename("state.json", &stage.directory, PROJECTION_STATE_STAGE_FILE)?;
        #[cfg(unix)]
        {
            sync_dir_capability(wenlan)?;
            sync_dir_capability(&stage.directory)?;
        }

        let mut moved_budget = RepairReadBudget::new();
        let (moved, moved_mode, moved_identity) = read_projection_state_identity_nofollow(
            &stage.directory,
            OsStr::new(PROJECTION_STATE_STAGE_FILE),
            &mut moved_budget,
        )?;
        if moved != expected || moved_mode != mode || moved_identity != current_identity {
            match stage
                .directory
                .hard_link(PROJECTION_STATE_STAGE_FILE, wenlan, "state.json")
            {
                Ok(()) => {
                    let mut restored_budget = RepairReadBudget::new();
                    let (_, _, restored_identity) = read_projection_state_identity_nofollow(
                        wenlan,
                        OsStr::new("state.json"),
                        &mut restored_budget,
                    )?;
                    if restored_identity != moved_identity {
                        return Err(WenlanError::Conflict(
                            "repair_apply_recovery_required".to_string(),
                        ));
                    }
                    #[cfg(unix)]
                    sync_dir_capability(wenlan)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(WenlanError::Io(error)),
            }
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }

        match wenlan.hard_link(&temporary, wenlan, "state.json") {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ))
            }
            Err(error) => return Err(WenlanError::Io(error)),
        }
        let mut installed_budget = RepairReadBudget::new();
        let (installed, installed_mode, installed_identity) =
            read_projection_state_identity_nofollow(
                wenlan,
                OsStr::new("state.json"),
                &mut installed_budget,
            )?;
        if installed != replacement
            || installed_mode != mode
            || installed_identity != temporary_identity
        {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }

        let mut staged_budget = RepairReadBudget::new();
        let (staged, staged_mode, staged_identity) = read_projection_state_identity_nofollow(
            &stage.directory,
            OsStr::new(PROJECTION_STATE_STAGE_FILE),
            &mut staged_budget,
        )?;
        if staged != expected || staged_mode != mode || staged_identity != current_identity {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
        stage.directory.remove_file(PROJECTION_STATE_STAGE_FILE)?;
        #[cfg(unix)]
        {
            sync_dir_capability(wenlan)?;
            sync_dir_capability(&stage.directory)?;
        }
        Ok(())
    })();

    let temporary_cleanup = wenlan.remove_file(&temporary);
    match (result, temporary_cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(error)) => Err(WenlanError::Io(error)),
        (Err(error), _) => Err(error),
    }
}

fn sync_projection_dirs(
    root: &Dir,
    wenlan: &Dir,
    source_parent: &Dir,
    orphaned: &Dir,
) -> Result<(), WenlanError> {
    #[cfg(unix)]
    {
        sync_dir_capability(orphaned)?;
        sync_dir_capability(source_parent)?;
        sync_dir_capability(wenlan)?;
        sync_dir_capability(root)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_dir_capability(directory: &Dir) -> Result<(), WenlanError> {
    // cap-std opens directory capabilities with O_PATH on Linux. Cloning that
    // descriptor preserves O_PATH, which cannot be fsynced (EBADF). Reopen the
    // same directory through the capability to obtain a syncable descriptor.
    directory.open(".")?.sync_all()?;
    Ok(())
}

impl KnowledgeProjectionWrite {
    pub fn new(path: PathBuf, database: &crate::db::MemoryDB) -> Self {
        Self {
            writer: KnowledgeWriter::new(path, database),
            guard: database.begin_page_projection_write(),
        }
    }

    pub(crate) fn new_repair(path: PathBuf, database: &crate::db::MemoryDB) -> Self {
        Self {
            writer: KnowledgeWriter::new_repair(path, database),
            guard: database.begin_page_projection_write(),
        }
    }

    pub(crate) fn with_repair_lock<T, F>(
        path: PathBuf,
        database: &crate::db::MemoryDB,
        operation: F,
    ) -> Result<T, WenlanError>
    where
        F: FnOnce(&LockedRepairProjection<'_>) -> Result<T, WenlanError>,
    {
        let write = Self::new_repair(path, database);
        Self::with_projection_capabilities(&write.writer.path, |capabilities| {
            operation(&LockedRepairProjection {
                write: &write,
                capabilities,
            })
        })
    }

    pub(crate) fn with_projection_lock<T, F>(path: &Path, operation: F) -> Result<T, WenlanError>
    where
        F: FnOnce(&LockedProjection<'_>) -> Result<T, WenlanError>,
    {
        Self::with_projection_capabilities(path, |capabilities| {
            operation(&LockedProjection { capabilities })
        })
    }

    pub(crate) fn begin_owned_repair_session(
        path: PathBuf,
        database: &crate::db::MemoryDB,
    ) -> Result<OwnedRepairProjectionSession, WenlanError> {
        let write = Self::new_repair(path, database);
        let _write_lock = PROJECTION_WRITE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let capabilities = ProjectionCapabilities::open(&write.writer.path)?;
        Ok(OwnedRepairProjectionSession {
            write,
            capabilities,
        })
    }

    fn with_projection_capabilities<T, F>(path: &Path, operation: F) -> Result<T, WenlanError>
    where
        F: FnOnce(&ProjectionCapabilities) -> Result<T, WenlanError>,
    {
        let _write_lock = PROJECTION_WRITE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let capabilities = ProjectionCapabilities::open(path)?;
        operation(&capabilities)
    }

    pub fn write_page(&self, page: &Page) -> Result<String, WenlanError> {
        self.writer.write_page(&self.guard, page)
    }

    pub fn remove_page(&self, page_id: &str) -> Result<(), WenlanError> {
        self.writer.remove_page(&self.guard, page_id)
    }

    pub fn reconcile(&self, pages: &[Page]) -> Result<ReconcileStats, WenlanError> {
        self.writer.reconcile(&self.guard, pages)
    }
}

impl OwnedRepairProjectionSession {
    pub(crate) fn locked(&self) -> LockedRepairProjection<'_> {
        LockedRepairProjection {
            write: &self.write,
            capabilities: &self.capabilities,
        }
    }
}

struct ProjectionCapabilities {
    root: Dir,
    wenlan: Dir,
    // Advisory only: processes which do not participate in this protocol can
    // still mutate the Page tree. Every Wenlan projection writer acquires it
    // after `PROJECTION_WRITE_LOCK`, so cooperating processes fail closed.
    // Normal writers still use their existing ambient filesystem operations;
    // participating in this lock does not make those operations capability-safe.
    _file_lock: std::fs::File,
}

impl ProjectionCapabilities {
    fn open(path: &Path) -> Result<Self, WenlanError> {
        let parent = path
            .parent()
            .ok_or_else(|| WenlanError::Validation("repair_projection_root_invalid".to_string()))?;
        let basename = path
            .file_name()
            .ok_or_else(|| WenlanError::Validation("repair_projection_root_invalid".to_string()))?;
        let ambient_parent = Dir::open_ambient_dir(parent, cap_std::ambient_authority())?;
        let root_metadata = ambient_parent.symlink_metadata(Path::new(basename))?;
        if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
            return Err(WenlanError::Validation(
                "repair_projection_root_invalid".to_string(),
            ));
        }
        let root = ambient_parent.open_dir_nofollow(Path::new(basename))?;
        match root.symlink_metadata(".wenlan") {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return Err(WenlanError::Conflict(
                    "repair_projection_control_invalid".to_string(),
                ))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                root.create_dir(".wenlan")?;
            }
            Err(error) => return Err(WenlanError::Io(error)),
        }
        let wenlan = root.open_dir_nofollow(".wenlan")?;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create(true)
            .follow(FollowSymlinks::No);
        let file_lock = wenlan.open_with(".projection.lock", &options)?.into_std();
        match file_lock.try_lock_exclusive() {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(WenlanError::Conflict("page_projection_locked".to_string()))
            }
            Err(error) => return Err(WenlanError::Io(error)),
        }
        Ok(Self {
            root,
            wenlan,
            _file_lock: file_lock,
        })
    }
}

/// Public projection of a page's markdown (frontmatter + body + the delimiter
/// Sources block), so callers outside this module (the watcher's
/// protected-block re-projection check) can compute the canonical projection
/// without duplicating it.
pub fn render_markdown_for(page: &Page) -> String {
    render_markdown(page)
}

fn render_markdown(page: &Page) -> String {
    use crate::export::provenance::{
        related_frontmatter, render_sources_block, sources_frontmatter, yaml_quoted,
    };
    let mut out = String::new();

    // Frontmatter
    out.push_str("---\n");
    out.push_str(&format!("title: {}\n", yaml_quoted(&page.title)));
    if let Some(ref space) = page.space {
        out.push_str(&format!("space: {}\n", space));
    }
    out.push_str(&format!("origin_id: {}\n", page.id));
    out.push_str(&format!("origin_version: {}\n", page.version));
    let created_date: String = page.created_at.chars().take(10).collect();
    let modified_date: String = page.last_modified.chars().take(10).collect();
    out.push_str(&format!("created: {}\n", created_date));
    out.push_str(&format!("modified: {}\n", modified_date));
    // Read-only provenance projection (one-way; the watcher never reads it back).
    out.push_str(&sources_frontmatter(&page.source_memory_ids));
    let related = related_page_titles(&page.content);
    out.push_str(&related_frontmatter(&related));
    out.push_str("---\n\n");

    // Body with wikilinks
    out.push_str(&convert_links_to_wikilinks(&page.content));

    // Export-only delimiter-wrapped Sources block, generated from DB truth.
    let block = render_sources_block(&page.source_memory_ids);
    if !block.is_empty() {
        out.push_str("\n\n");
        out.push_str(&block);
    } else {
        out.push('\n');
    }

    out
}

/// Page-to-page wikilink targets in the body, for the read-only `related:`
/// frontmatter. `mem_*` targets are excluded — those are provenance, not
/// topic links.
fn related_page_titles(content: &str) -> Vec<String> {
    let wikified = convert_links_to_wikilinks(content);
    crate::sources::obsidian::extract_wikilinks(&wikified)
        .into_iter()
        .map(|w| w.target)
        .filter(|t| !t.starts_with("mem_"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pages::Page;

    fn test_concept() -> Page {
        Page {
            id: "concept_test123".to_string(),
            title: "Rust Ownership".to_string(),
            summary: Some("Memory safety without GC".to_string()),
            content: "## Overview\nRust uses ownership for memory safety.\n\n## Related\n- [Borrowing](concept_borrow1)".to_string(),
            entity_id: None,
            space: Some("rust".to_string()),
            source_memory_ids: vec!["m1".to_string()],
            version: 2,
            status: "active".to_string(),
            created_at: "2026-04-01T00:00:00+00:00".to_string(),
            last_compiled: "2026-04-09T00:00:00+00:00".to_string(),
            last_modified: "2026-04-09T00:00:00+00:00".to_string(),
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
    async fn repair_projection_lock_covers_the_whole_transaction() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_root = tempfile::TempDir::new().unwrap();

        KnowledgeProjectionWrite::with_repair_lock(
            page_root.path().to_path_buf(),
            &db,
            |_repair| {
                let excluded = std::thread::spawn(|| {
                    matches!(
                        PROJECTION_WRITE_LOCK.try_lock(),
                        Err(std::sync::TryLockError::WouldBlock)
                    )
                })
                .join()
                .unwrap();
                assert!(excluded);
                Ok(())
            },
        )
        .unwrap();
        assert!(PROJECTION_WRITE_LOCK.try_lock().is_ok());
    }

    #[tokio::test]
    async fn projection_file_lock_rejects_second_process() {
        const CHILD_ROOT: &str = "WENLAN_PROJECTION_LOCK_TEST_CHILD_ROOT";
        const CHILD_MODE: &str = "WENLAN_PROJECTION_LOCK_TEST_CHILD_MODE";
        if let Some(page_root) = std::env::var_os(CHILD_ROOT) {
            let page_root = PathBuf::from(page_root);
            let error = match std::env::var(CHILD_MODE).as_deref() {
                Ok("repair") => {
                    let (db, _dir) = crate::db::tests::test_db().await;
                    KnowledgeProjectionWrite::with_repair_lock(page_root, &db, |_repair| Ok(()))
                        .expect_err("a second repair process must not acquire the projection lock")
                }
                Ok("write") => KnowledgeWriter::new_for_test(page_root)
                    .write_page_for_test(&test_concept())
                    .expect_err("a normal writer must participate in the projection lock"),
                Ok("remove") => KnowledgeWriter::new_for_test(page_root)
                    .remove_page_for_test("page_absent")
                    .expect_err("a normal remover must participate in the projection lock"),
                mode => panic!("unexpected child lock mode: {mode:?}"),
            };
            assert_eq!(error.to_string(), "Conflict: page_projection_locked");
            return;
        }

        let (db, _dir) = crate::db::tests::test_db().await;
        let page_root = tempfile::TempDir::new().unwrap();
        KnowledgeProjectionWrite::with_repair_lock(
            page_root.path().to_path_buf(),
            &db,
            |_repair| {
                for mode in ["repair", "write", "remove"] {
                    let output = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "export::knowledge::tests::projection_file_lock_rejects_second_process",
                            "--nocapture",
                        ])
                        .env(CHILD_ROOT, page_root.path())
                        .env(CHILD_MODE, mode)
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{mode} child lock probe failed:\nstdout:\n{}\nstderr:\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr),
                    );
                }
                Ok(())
            },
        )
        .unwrap();
    }

    #[tokio::test]
    async fn repair_quarantine_rejects_duplicate_state_keys_before_mutation() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let cases = [
            br#"{"schema_version":2,"pages":{"page_other":{"file":"other.md"}},"pages":{"page_duplicate":{"file":"target.md"}}}"#
                .as_slice(),
            br#"{"schema_version":2,"pages":{"page_duplicate":{"file":"other.md"},"page_duplicate":{"file":"target.md"}}}"#
                .as_slice(),
            br#"{"schema_version":2,"pages":{"page_duplicate":{"file":"other.md","file":"target.md","version":1}}}"#
                .as_slice(),
            br#"{"schema_version":2,"pages":{"page_duplicate":{"file":"target.md","version":0,"version":1}}}"#
                .as_slice(),
        ];

        for raw_state in cases {
            let page_root = tempfile::TempDir::new().unwrap();
            std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
            std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
            std::fs::write(page_root.path().join("target.md"), b"target bytes").unwrap();

            let error = KnowledgeProjectionWrite::with_repair_lock(
                page_root.path().to_path_buf(),
                &db,
                |repair| {
                    repair.quarantine_stale_page(
                        "page_duplicate",
                        "target.md",
                        ".wenlan/orphaned/page_duplicate.md",
                    )
                },
            )
            .unwrap_err();

            assert!(error.to_string().contains("repair_target_stale"));
            assert_eq!(
                std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
                raw_state
            );
            assert_eq!(
                std::fs::read(page_root.path().join("target.md")).unwrap(),
                b"target bytes"
            );
            assert!(!page_root.path().join(".wenlan/orphaned").exists());
        }
    }

    #[tokio::test]
    async fn stale_projection_source_reappearing_after_stage_preserves_all_versions() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        let raw_state =
            br#"{"schema_version":2,"pages":{"page_reappears":{"file":"target.md","version":1}}}"#;
        let approved = b"---\norigin_id: page_reappears\norigin_version: 1\n---\napproved\n";
        let replacement = b"---\norigin_id: page_reappears\norigin_version: 1\n---\nreplacement\n";
        std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
        std::fs::write(page_root.path().join("target.md"), approved).unwrap();
        let rollback = crate::repair::capture_stale_page_projection_current(
            page_root.path(),
            "page_reappears",
            "target.md",
            ".wenlan/orphaned/page_reappears.md",
        )
        .unwrap();

        let error = KnowledgeProjectionWrite::with_repair_lock(
            page_root.path().to_path_buf(),
            &db,
            |repair| {
                let mut pinned = repair.pin_stale_page_projection(
                    "page_reappears",
                    "target.md",
                    ".wenlan/orphaned/page_reappears.md",
                    &rollback,
                    "manifest_source_reappears",
                )?;
                let source = page_root.path().join("target.md");
                let result = pinned.quarantine_with_after_source_stage(|| {
                    std::fs::write(&source, replacement)?;
                    Ok(())
                });
                assert!(pinned.mutation_started());
                result
            },
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Conflict: repair_apply_recovery_required"
        );
        assert_eq!(
            std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
            raw_state
        );
        assert_eq!(
            std::fs::read(page_root.path().join("target.md")).unwrap(),
            replacement
        );
        assert_eq!(
            std::fs::read(page_root.path().join(".wenlan/orphaned/page_reappears.md")).unwrap(),
            approved
        );
        let stage = page_root
            .path()
            .join(".wenlan")
            .join(projection_unlink_stage_name("manifest_source_reappears"));
        assert_eq!(std::fs::read(stage.join("source")).unwrap(), approved);
        assert!(stage.join(PROJECTION_STAGE_OWNER_FILE).is_file());
    }

    #[tokio::test]
    async fn stale_projection_foreign_quarantine_replacement_is_never_deleted() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        let raw_state = br#"{"schema_version":2,"pages":{"page_quarantine_race":{"file":"target.md","version":1}}}"#;
        let approved = b"---\norigin_id: page_quarantine_race\norigin_version: 1\n---\napproved\n";
        let foreign = b"foreign quarantine replacement";
        std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
        std::fs::write(page_root.path().join("target.md"), approved).unwrap();
        let rollback = crate::repair::capture_stale_page_projection_current(
            page_root.path(),
            "page_quarantine_race",
            "target.md",
            ".wenlan/orphaned/page_quarantine_race.md",
        )
        .unwrap();

        let error = KnowledgeProjectionWrite::with_repair_lock(
            page_root.path().to_path_buf(),
            &db,
            |repair| {
                let mut pinned = repair.pin_stale_page_projection(
                    "page_quarantine_race",
                    "target.md",
                    ".wenlan/orphaned/page_quarantine_race.md",
                    &rollback,
                    "manifest_quarantine_race",
                )?;
                let quarantine = page_root
                    .path()
                    .join(".wenlan/orphaned/page_quarantine_race.md");
                pinned.quarantine_with_after_link(|| {
                    let replacement = page_root.path().join("foreign-quarantine.tmp");
                    std::fs::write(&replacement, foreign)?;
                    std::fs::rename(replacement, quarantine)?;
                    Ok(())
                })
            },
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Conflict: repair_apply_recovery_required"
        );
        assert_eq!(
            std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
            raw_state
        );
        assert_eq!(
            std::fs::read(page_root.path().join("target.md")).unwrap(),
            approved
        );
        assert_eq!(
            std::fs::read(
                page_root
                    .path()
                    .join(".wenlan/orphaned/page_quarantine_race.md")
            )
            .unwrap(),
            foreign
        );
    }

    #[tokio::test]
    async fn stale_projection_state_replacement_between_compare_and_swap_is_preserved() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        let raw_state =
            br#"{"schema_version":2,"pages":{"page_state_race":{"file":"target.md","version":1}}}"#;
        let foreign_state =
            br#"{"schema_version":2,"pages":{"foreign":{"file":"foreign.md","version":9}}}"#;
        let approved = b"---\norigin_id: page_state_race\norigin_version: 1\n---\napproved\n";
        std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
        std::fs::write(page_root.path().join("target.md"), approved).unwrap();
        let rollback = crate::repair::capture_stale_page_projection_current(
            page_root.path(),
            "page_state_race",
            "target.md",
            ".wenlan/orphaned/page_state_race.md",
        )
        .unwrap();

        let error = KnowledgeProjectionWrite::with_repair_lock(
            page_root.path().to_path_buf(),
            &db,
            |repair| {
                let mut pinned = repair.pin_stale_page_projection(
                    "page_state_race",
                    "target.md",
                    ".wenlan/orphaned/page_state_race.md",
                    &rollback,
                    "manifest_state_race",
                )?;
                let state = page_root.path().join(".wenlan/state.json");
                pinned.quarantine_with_before_state_swap(|| {
                    let replacement = page_root.path().join(".wenlan/foreign-state.tmp");
                    std::fs::write(&replacement, foreign_state)?;
                    std::fs::rename(replacement, state)?;
                    Ok(())
                })
            },
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Conflict: repair_apply_recovery_required"
        );
        assert_eq!(
            std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
            foreign_state
        );
        assert!(!page_root.path().join("target.md").exists());
        assert_eq!(
            std::fs::read(page_root.path().join(".wenlan/orphaned/page_state_race.md")).unwrap(),
            approved
        );
        let stage = page_root
            .path()
            .join(".wenlan")
            .join(projection_unlink_stage_name("manifest_state_race"));
        assert_eq!(std::fs::read(stage.join("source")).unwrap(), approved);
        assert_eq!(std::fs::read(stage.join("state")).unwrap(), foreign_state);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pinned_quarantine_and_rollback_ignore_ancestor_swap_canary() {
        use std::os::unix::fs::symlink;

        let (db, _dir) = crate::db::tests::test_db().await;
        let parent = tempfile::TempDir::new().unwrap();
        let page_root = parent.path().join("page-root");
        let pinned_root = parent.path().join("pinned-root");
        let external = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.join(".wenlan")).unwrap();
        let raw_state =
            br#"{"schema_version":2,"pages":{"page_stale":{"file":"target.md","version":1}}}"#;
        let source_bytes = b"---\norigin_id: page_stale\n---\noriginal\n";
        std::fs::write(page_root.join(".wenlan/state.json"), raw_state).unwrap();
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(
                page_root.join(".wenlan/state.json"),
                std::fs::Permissions::from_mode(0o640),
            )
            .unwrap();
        }
        std::fs::write(page_root.join("target.md"), source_bytes).unwrap();
        std::fs::write(external.path().join("canary"), b"external").unwrap();
        std::fs::create_dir_all(external.path().join(".wenlan/orphaned")).unwrap();
        std::fs::write(
            external.path().join(".wenlan/state.json"),
            b"external state",
        )
        .unwrap();
        std::fs::write(external.path().join("target.md"), b"external source").unwrap();
        std::fs::write(
            external.path().join(".wenlan/orphaned/page_stale.md"),
            b"external quarantine",
        )
        .unwrap();

        let before = crate::repair::capture_stale_page_projection_current(
            &page_root,
            "page_stale",
            "target.md",
            ".wenlan/orphaned/page_stale.md",
        )
        .unwrap();
        KnowledgeProjectionWrite::with_repair_lock(page_root.clone(), &db, |repair| {
            let mut pinned = repair.pin_stale_page_projection(
                "page_stale",
                "target.md",
                ".wenlan/orphaned/page_stale.md",
                &before,
                "page_stale",
            )?;
            std::fs::rename(&page_root, &pinned_root)?;
            symlink(external.path(), &page_root)?;

            pinned.quarantine()?;
            {
                use std::os::unix::fs::PermissionsExt as _;
                assert_eq!(
                    std::fs::metadata(pinned_root.join(".wenlan/state.json"))
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777,
                    0o640,
                );
            }
            pinned.restore_snapshot(&before)?;
            Ok(())
        })
        .unwrap();

        assert_eq!(
            std::fs::read(external.path().join("canary")).unwrap(),
            b"external"
        );
        assert_eq!(
            std::fs::read(external.path().join(".wenlan/state.json")).unwrap(),
            b"external state"
        );
        assert_eq!(
            std::fs::read(external.path().join("target.md")).unwrap(),
            b"external source"
        );
        assert_eq!(
            std::fs::read(external.path().join(".wenlan/orphaned/page_stale.md")).unwrap(),
            b"external quarantine"
        );
        assert_eq!(
            std::fs::read(pinned_root.join(".wenlan/state.json")).unwrap(),
            raw_state
        );
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(pinned_root.join(".wenlan/state.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o640,
            );
        }
        assert_eq!(
            std::fs::read(pinned_root.join("target.md")).unwrap(),
            source_bytes
        );
        assert!(pinned_root.join(".wenlan/orphaned").is_dir());
        assert!(std::fs::read_dir(pinned_root.join(".wenlan/orphaned"))
            .unwrap()
            .next()
            .is_none());
    }

    #[cfg(unix)]
    #[test]
    fn projection_locked_capture_and_scan_ignore_ancestor_swap_canary() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::TempDir::new().unwrap();
        let page_root = parent.path().join("page-root");
        let pinned_root = parent.path().join("pinned-root");
        let external = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.join(".wenlan")).unwrap();
        let raw_state =
            br#"{"schema_version":2,"pages":{"page_stale":{"file":"target.md","version":1}}}"#;
        let source_bytes = b"---\norigin_id: page_stale\n---\noriginal\n";
        std::fs::write(page_root.join(".wenlan/state.json"), raw_state).unwrap();
        std::fs::write(page_root.join("target.md"), source_bytes).unwrap();
        std::fs::write(page_root.join("pinned-canary.md"), b"pinned").unwrap();
        std::fs::create_dir(page_root.join(".wenlan/orphaned")).unwrap();
        std::fs::write(
            page_root.join(".wenlan/orphaned/existing.md"),
            b"pinned orphan",
        )
        .unwrap();

        std::fs::create_dir_all(external.path().join(".wenlan")).unwrap();
        std::fs::write(
            external.path().join(".wenlan/state.json"),
            br#"{"schema_version":2,"pages":{"page_external":{"file":"external.md","version":1}}}"#,
        )
        .unwrap();
        std::fs::write(external.path().join("target.md"), b"external source").unwrap();
        std::fs::write(external.path().join("external-canary.md"), b"external").unwrap();
        std::fs::create_dir(external.path().join(".wenlan/orphaned")).unwrap();
        std::fs::write(
            external.path().join(".wenlan/orphaned/external.md"),
            b"external orphan",
        )
        .unwrap();
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(
                page_root.join(".wenlan/orphaned"),
                std::fs::Permissions::from_mode(0o700),
            )
            .unwrap();
            std::fs::set_permissions(
                external.path().join(".wenlan/orphaned"),
                std::fs::Permissions::from_mode(0o700),
            )
            .unwrap();
        }

        KnowledgeProjectionWrite::with_projection_lock(&page_root, |projection| {
            std::fs::rename(&page_root, &pinned_root)?;
            symlink(external.path(), &page_root)?;

            let scan = projection.scan_page_root_controlled(
                true,
                &crate::lint::pages::fs::PageScanControl::with_timeout(
                    std::time::Duration::from_secs(30),
                ),
            )?;
            assert!(scan.entry("pinned-canary.md").is_some());
            assert!(scan.entry("external-canary.md").is_none());

            let captured = projection.capture_stale_page_projection_current(
                "page_stale",
                "target.md",
                ".wenlan/orphaned/page_stale.md",
            )?;
            assert_eq!(
                captured
                    .rows
                    .iter()
                    .find(|row| row[0] == ".wenlan/state.json")
                    .unwrap()[2],
                hex::encode(raw_state)
            );
            assert_eq!(
                captured
                    .rows
                    .iter()
                    .find(|row| row[0] == "target.md")
                    .unwrap()[2],
                hex::encode(source_bytes)
            );
            let orphaned_row = captured
                .rows
                .iter()
                .find(|row| row[0] == ".wenlan/orphaned")
                .unwrap();
            let baseline: Vec<(String, String)> =
                serde_json::from_slice(&hex::decode(&orphaned_row[2]).unwrap()).unwrap();
            assert_eq!(
                baseline,
                vec![(
                    "existing.md".to_string(),
                    hex::encode(sha2::Sha256::digest(b"pinned orphan"))
                )]
            );
            Ok(())
        })
        .unwrap();

        assert_eq!(
            std::fs::read(external.path().join("target.md")).unwrap(),
            b"external source"
        );
    }

    #[test]
    fn projection_capability_read_rejects_oversized_file_at_bound() {
        const CAPTURE_LIMIT: u64 = 16 * 1024 * 1024;

        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        std::fs::write(
            page_root.path().join(".wenlan/state.json"),
            br#"{"schema_version":2,"pages":{}}"#,
        )
        .unwrap();
        let oversized = std::fs::File::create(page_root.path().join("oversized.md")).unwrap();
        oversized.set_len(CAPTURE_LIMIT + 1).unwrap();

        let error =
            KnowledgeProjectionWrite::with_projection_lock(page_root.path(), |projection| {
                projection
                    .read_relative_regular_nofollow("oversized.md", CAPTURE_LIMIT)
                    .map(|_| ())
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("repair_projection_rollback_too_large"),
            "{error}"
        );
    }

    #[test]
    fn projection_capability_read_rejects_file_growth_after_metadata() {
        const CAPTURE_LIMIT: u64 = 1024;

        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        std::fs::write(
            page_root.path().join(".wenlan/state.json"),
            br#"{"schema_version":2,"pages":{}}"#,
        )
        .unwrap();
        let growing_path = page_root.path().join("growing.md");
        let growing = std::fs::File::create(&growing_path).unwrap();
        growing.set_len(CAPTURE_LIMIT).unwrap();

        let error =
            KnowledgeProjectionWrite::with_projection_lock(page_root.path(), |projection| {
                projection
                    .read_relative_regular_nofollow_with_after_metadata(
                        "growing.md",
                        CAPTURE_LIMIT,
                        || {
                            std::fs::OpenOptions::new()
                                .write(true)
                                .open(&growing_path)?
                                .set_len(CAPTURE_LIMIT + 1)?;
                            Ok(())
                        },
                    )
                    .map(|_| ())
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("repair_projection_rollback_too_large"),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn projection_capture_rejects_aggregate_orphan_baseline_over_budget() {
        use std::os::unix::fs::PermissionsExt as _;

        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan/orphaned")).unwrap();
        std::fs::set_permissions(
            page_root.path().join(".wenlan/orphaned"),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        std::fs::write(
            page_root.path().join(".wenlan/state.json"),
            br#"{"schema_version":2,"pages":{"page_stale_many_orphans":{"file":"source.md","version":1}}}"#,
        )
        .unwrap();
        std::fs::write(
            page_root.path().join("source.md"),
            b"---\norigin_id: page_stale_many_orphans\norigin_version: 1\n---\nbody\n",
        )
        .unwrap();
        for index in 0..17 {
            std::fs::File::create(
                page_root
                    .path()
                    .join(format!(".wenlan/orphaned/{index:02}.md")),
            )
            .unwrap()
            .set_len(1024 * 1024)
            .unwrap();
        }

        let error =
            KnowledgeProjectionWrite::with_projection_lock(page_root.path(), |projection| {
                projection.capture_stale_page_projection_current(
                    "page_stale_many_orphans",
                    "source.md",
                    ".wenlan/orphaned/page_stale_many_orphans.md",
                )
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("repair_projection_rollback_too_large"),
            "{error}"
        );
        assert!(page_root.path().join("source.md").is_file());
        assert!(!page_root
            .path()
            .join(".wenlan/orphaned/page_stale_many_orphans.md")
            .exists());
    }

    #[cfg(unix)]
    #[test]
    fn projection_orphan_baseline_charges_many_empty_file_entries() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempfile::TempDir::new().unwrap();
        let orphaned_path = root.path().join("orphaned");
        std::fs::create_dir(&orphaned_path).unwrap();
        std::fs::set_permissions(&orphaned_path, std::fs::Permissions::from_mode(0o700)).unwrap();
        for index in 0..65 {
            std::fs::File::create(orphaned_path.join(format!("{index:02}.md"))).unwrap();
        }
        let orphaned = Dir::open_ambient_dir(&orphaned_path, cap_std::ambient_authority()).unwrap();
        let mut budget = RepairReadBudget {
            remaining: 64 * ORPHAN_BASELINE_ENTRY_BUDGET_BYTES,
        };

        let error = orphaned_baseline_nofollow(&orphaned, None, &mut budget).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("repair_projection_rollback_too_large"),
            "{error}"
        );
    }

    #[test]
    fn first_page_write_creates_missing_projection_root_nofollow() {
        let parent = tempfile::TempDir::new().unwrap();
        let page_root = parent.path().join("new-page-root");
        let writer = KnowledgeWriter::new_for_test(page_root.clone());

        let path = writer.write_page_for_test(&test_concept()).unwrap();

        assert!(page_root.is_dir());
        assert!(page_root.join(".wenlan").is_dir());
        assert_eq!(
            std::fs::read_to_string(path).unwrap(),
            render_markdown(&test_concept())
        );
    }

    #[cfg(unix)]
    #[test]
    fn first_page_write_root_swap_uses_pinned_capability() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::TempDir::new().unwrap();
        let page_root = parent.path().join("page-root");
        let pinned_root = parent.path().join("pinned-root");
        let external = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(external.path().join(".wenlan")).unwrap();
        std::fs::write(external.path().join("canary"), b"external").unwrap();
        let writer = KnowledgeWriter::new_for_test(page_root.clone());

        writer
            .write_page_after_open_for_test(&test_concept(), || {
                std::fs::rename(&page_root, &pinned_root)?;
                symlink(external.path(), &page_root)?;
                Ok(())
            })
            .unwrap();

        assert_eq!(
            std::fs::read(external.path().join("canary")).unwrap(),
            b"external"
        );
        assert!(!external.path().join("rust-ownership.md").exists());
        assert!(!external.path().join("_sources").exists());
        assert_eq!(
            std::fs::read_to_string(pinned_root.join("rust-ownership.md")).unwrap(),
            render_markdown(&test_concept())
        );
        assert!(pinned_root.join(".wenlan/state.json").is_file());
        assert!(pinned_root.join("_sources/m1.md").is_file());
        assert!(pinned_root.join("_sources/.manifest.json").is_file());
    }

    #[test]
    fn test_write_page_creates_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        let page = test_concept();

        let path = writer.write_page_for_test(&page).unwrap();
        assert!(path.ends_with("rust-ownership.md"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("title: \"Rust Ownership\""));
        assert!(content.contains("space: rust"));
        assert!(content.contains("origin_id: concept_test123"));
        assert!(content.contains("origin_version: 2"));
        // Wikilinks converted
        assert!(content.contains("[[Borrowing]]"));
        assert!(!content.contains("(concept_borrow1)"));
    }

    #[test]
    fn test_write_updates_state() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        writer.write_page_for_test(&test_concept()).unwrap();

        let state = writer.load_state();
        assert_eq!(state.schema_version, KNOWLEDGE_STATE_SCHEMA_V2);
        assert!(state.pages.contains_key("concept_test123"));
        assert_eq!(state.pages["concept_test123"].file, "rust-ownership.md");
        assert_eq!(state.pages["concept_test123"].version, 2);
    }

    #[test]
    fn test_remove_page_deletes_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        let path = writer.write_page_for_test(&test_concept()).unwrap();
        assert!(std::path::Path::new(&path).exists());

        writer.remove_page_for_test("concept_test123").unwrap();
        assert!(!std::path::Path::new(&path).exists());

        let state = writer.load_state();
        assert!(!state.pages.contains_key("concept_test123"));
    }

    #[test]
    fn test_remove_nonexistent_page_noop() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        writer.remove_page_for_test("nonexistent").unwrap();
    }

    #[test]
    fn test_write_multiple_pages() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        let c1 = Page {
            id: "concept_a".to_string(),
            title: "Alpha".to_string(),
            ..test_concept()
        };
        let c2 = Page {
            id: "concept_b".to_string(),
            title: "Beta".to_string(),
            ..test_concept()
        };

        writer.write_page_for_test(&c1).unwrap();
        writer.write_page_for_test(&c2).unwrap();

        assert!(dir.path().join("alpha.md").exists());
        assert!(dir.path().join("beta.md").exists());

        let state = writer.load_state();
        assert_eq!(state.pages.len(), 2);
    }

    #[test]
    fn test_knowledge_writer_overwrite_on_version_change() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        let mut page = test_concept();
        writer.write_page_for_test(&page).unwrap();

        let v1 = std::fs::read_to_string(dir.path().join("rust-ownership.md")).unwrap();
        assert!(v1.contains("origin_version: 2"));

        // Update version and content
        page.version = 3;
        page.content = "## Updated\nNew content.".to_string();
        writer.write_page_for_test(&page).unwrap();

        let v2 = std::fs::read_to_string(dir.path().join("rust-ownership.md")).unwrap();
        assert!(v2.contains("origin_version: 3"));
        assert!(v2.contains("## Updated"));
        assert!(!v2.contains("memory safety")); // old content replaced

        // State reflects new version
        let state = writer.load_state();
        assert_eq!(state.pages["concept_test123"].version, 3);
    }

    /// `write_page` must replace the target file via temp-file-then-rename,
    /// never via in-place truncate+write, so a concurrent reader can never
    /// observe a half-written page file. A plain `fs::write`
    /// truncates and rewrites the *same* inode, so its inode number is
    /// unchanged across writes; an atomic rename swaps in a new inode. That
    /// difference is deterministic (no thread timing needed) and is what
    /// actually distinguishes this from the old, non-atomic implementation —
    /// unlike the leftover-temp-file check below, which passes trivially on
    /// old code too (it never created temp files to begin with).
    #[cfg(unix)]
    #[test]
    fn write_page_replaces_file_via_atomic_rename_not_in_place_write() {
        use std::os::unix::fs::MetadataExt;

        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        let mut page = test_concept();
        let path = writer.write_page_for_test(&page).unwrap();
        let ino_before = std::fs::metadata(&path).unwrap().ino();

        page.version = 3;
        page.content = "## Updated\nNew content for atomicity check.".to_string();
        writer.write_page_for_test(&page).unwrap();

        let ino_after = std::fs::metadata(&path).unwrap().ino();
        assert_ne!(
            ino_before, ino_after,
            "rewrite kept the same inode — target was truncated in place, not atomically renamed"
        );

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, render_markdown_for(&page));

        let leftover_temp_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(
            leftover_temp_files.is_empty(),
            "leftover temp file(s): {leftover_temp_files:?}"
        );
    }

    /// A projection file that vanished (rename never landed, user deleted it)
    /// is rewritten from DB truth.
    #[test]
    fn reconcile_rewrites_page_whose_projection_file_vanished() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        let page = test_concept();
        let path = writer.write_page_for_test(&page).unwrap();
        std::fs::remove_file(&path).unwrap();

        let stats = writer
            .reconcile_for_test(std::slice::from_ref(&page))
            .unwrap();

        assert_eq!(stats.checked, 1);
        assert_eq!(stats.rewritten, 1);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            render_markdown_for(&page),
            "vanished projection was not restored from the DB"
        );
    }

    /// The fsync-free write buys atomicity, not durability: after a power loss
    /// the rename may be undone, leaving the *previous* version's bytes under
    /// the target name. The `origin_version` stamp is what proves it — the DB
    /// moved on, the file did not.
    #[test]
    fn reconcile_rewrites_projection_left_behind_by_a_torn_rename() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        let mut page = test_concept();
        let path = writer.write_page_for_test(&page).unwrap();

        // DB advanced to v3; the file still holds the v2 projection.
        page.version = 3;
        page.content = "## Updated\nv3 body that never reached disk.".to_string();

        let stats = writer
            .reconcile_for_test(std::slice::from_ref(&page))
            .unwrap();

        assert_eq!(stats.rewritten, 1);
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            on_disk,
            render_markdown_for(&page),
            "stale projection was not brought forward to the DB version"
        );
    }

    /// The other power-loss shape: the rename is durable but the bytes are
    /// not, so the target is zero-length. No frontmatter parses, so the
    /// version stamp reads 0 — behind any real page (pages start at 1).
    #[test]
    fn reconcile_rewrites_projection_truncated_to_zero_bytes() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        let page = test_concept();
        let path = writer.write_page_for_test(&page).unwrap();
        std::fs::write(&path, "").unwrap();

        let stats = writer
            .reconcile_for_test(std::slice::from_ref(&page))
            .unwrap();

        assert_eq!(stats.rewritten, 1);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            render_markdown_for(&page)
        );
    }

    /// The safety boundary. md is canonical for prose (see `page_watcher`), so
    /// a body that differs from `render_markdown` while the version stamp is
    /// current is a USER EDIT made while the daemon was down — the watcher
    /// owns it, and reconcile must not clobber it on the way past. The temp
    /// sweep is likewise scoped to `write_page`'s own name shape, never to a
    /// file the user happens to have parked in the pages directory.
    #[test]
    fn reconcile_preserves_offline_user_edits_and_sweeps_only_our_temp_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        let page = test_concept();
        let path = writer.write_page_for_test(&page).unwrap();

        let edited = std::fs::read_to_string(&path)
            .unwrap()
            .replace("Rust uses ownership", "Hand-written prose the user typed");
        std::fs::write(&path, &edited).unwrap();

        let ours = dir.path().join(format!(".{}.4242.7.tmp", page.id));
        let user_scratch = dir.path().join("draft-notes.tmp");
        let editor_swap = dir.path().join(".obsidian-swap.tmp");
        for f in [&ours, &user_scratch, &editor_swap] {
            std::fs::write(f, "scratch").unwrap();
        }

        let stats = writer
            .reconcile_for_test(std::slice::from_ref(&page))
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            edited,
            "reconcile clobbered an offline user edit"
        );
        assert_eq!(stats.rewritten, 0);
        assert_eq!(stats.temp_files_removed, 1);
        assert!(!ours.exists(), "our own leftover temp file survived");
        assert!(
            user_scratch.exists() && editor_swap.exists(),
            "reconcile deleted a temp file that isn't ours"
        );
    }

    /// Without a `state.json` entry we cannot tell which file on disk is a
    /// page's projection, and guessing forks a `<slug>-2.md` duplicate against
    /// the real one (the hazard `page_watcher` calls out for vaults synced
    /// without `.wenlan/`). Reconcile skips those rather than fork.
    #[test]
    fn reconcile_skips_pages_with_no_state_entry_rather_than_forking_a_duplicate() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        let page = test_concept();
        // A projection exists on disk, but state.json does not know about it.
        std::fs::write(
            dir.path().join("rust-ownership.md"),
            render_markdown_for(&page),
        )
        .unwrap();

        let stats = writer
            .reconcile_for_test(std::slice::from_ref(&page))
            .unwrap();

        assert_eq!(stats.checked, 0);
        assert_eq!(stats.rewritten, 0);
        assert!(
            !dir.path().join("rust-ownership-2.md").exists(),
            "reconcile forked a duplicate projection"
        );
    }

    #[test]
    fn test_load_state_migrates_v1_concept_keys_to_page() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        // Write a legacy v1 state.json by hand.
        let v1_json = r#"{
            "concepts": {
                "concept_aaa": { "file": "a.md", "version": 1, "last_written": "2026-04-01T00:00:00+00:00" },
                "concept_bbb": { "file": "b.md", "version": 2, "last_written": "2026-04-02T00:00:00+00:00" }
            }
        }"#;
        std::fs::create_dir_all(dir.path().join(".wenlan")).unwrap();
        std::fs::write(dir.path().join(".wenlan/state.json"), v1_json).unwrap();

        let state = writer.load_state();
        // v1 keys are auto-rewritten to page_ prefix.
        assert!(state.pages.contains_key("page_aaa"));
        assert!(state.pages.contains_key("page_bbb"));
        assert!(!state.pages.contains_key("concept_aaa"));
        assert_eq!(state.pages["page_aaa"].file, "a.md");
        assert_eq!(state.pages["page_bbb"].version, 2);
        assert_eq!(state.schema_version, KNOWLEDGE_STATE_SCHEMA_V2);

        // After save_state, the file is rewritten in v2 form (no "concepts" key).
        writer.save_state(&state).unwrap();
        let written = std::fs::read_to_string(dir.path().join(".wenlan/state.json")).unwrap();
        assert!(written.contains("\"pages\""));
        assert!(!written.contains("\"concepts\""));
        assert!(written.contains("\"schema_version\""));
    }

    #[test]
    fn test_load_state_upgrades_writer_default_v0() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        std::fs::create_dir_all(dir.path().join(".wenlan")).unwrap();
        std::fs::write(
            dir.path().join(".wenlan/state.json"),
            r#"{"schema_version":0,"pages":{}}"#,
        )
        .unwrap();

        assert_eq!(
            writer.load_state().schema_version,
            KNOWLEDGE_STATE_SCHEMA_V2
        );
    }

    #[test]
    fn test_knowledge_writer_no_domain() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());

        let page = Page {
            space: None,
            ..test_concept()
        };
        let path = writer.write_page_for_test(&page).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(!content.contains("space:"));
    }

    #[test]
    fn render_markdown_emits_sources_frontmatter_and_delimiter_block() {
        let mut page = test_concept();
        page.source_memory_ids = vec!["mem_1".to_string(), "mem_2".to_string()];
        let md = render_markdown(&page);
        // Read-only frontmatter property.
        assert!(md.contains("sources: [\"[[mem_1]]\", \"[[mem_2]]\"]"));
        // Delimiter-wrapped Sources block in the body.
        assert!(md.contains(crate::export::provenance::SOURCES_BLOCK_START));
        assert!(md.contains(crate::export::provenance::SOURCES_BLOCK_END));
        assert!(md.contains("## Sources"));
        // The projected body, run back through the canonicalizer, drops the
        // generated block — the round-trip invariant the watcher relies on.
        let (_, body) = crate::sources::obsidian::extract_frontmatter(&md);
        let canon = crate::export::provenance::canonicalize_page_body(body);
        assert!(!canon.contains(crate::export::provenance::SOURCES_BLOCK_START));
        assert!(!canon.contains("## Sources"));
        // Substring of test_concept()'s actual body text.
        assert!(canon.contains("Rust uses ownership"));
    }

    #[test]
    fn stubs_resolve_then_gc_when_page_removed() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new_for_test(dir.path().to_path_buf());
        let mut page = test_concept();
        page.source_memory_ids = vec!["mem_x".to_string()];
        writer.write_page_for_test(&page).unwrap();
        let stub = dir.path().join("_sources").join("mem_x.md");
        assert!(stub.exists(), "stub projected on write");

        writer.remove_page_for_test(&page.id).unwrap();
        assert!(!stub.exists(), "orphan stub GC'd on page removal");
    }

    #[test]
    fn render_markdown_source_less_page_has_no_sources_artifacts() {
        let mut page = test_concept();
        page.source_memory_ids = vec![];
        let md = render_markdown(&page);
        assert!(!md.contains("sources:"));
        assert!(!md.contains(crate::export::provenance::SOURCES_BLOCK_START));
    }

    #[test]
    fn render_markdown_escapes_title_so_frontmatter_survives_quotes() {
        let mut page = test_concept();
        page.title = "The \"Real\" Architecture".to_string();
        let md = render_markdown(&page);
        let (fm, _body) = crate::sources::obsidian::extract_frontmatter(&md);
        // Frontmatter must NOT have collapsed to empty: origin_id still parses.
        assert_eq!(
            fm.get_str("origin_id"),
            Some(page.id.as_str()),
            "quote-bearing title collapsed the frontmatter map"
        );
        // And the title round-trips intact.
        assert_eq!(fm.get_str("title"), Some("The \"Real\" Architecture"));
    }
}
