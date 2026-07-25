// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::{
    db::tests::test_db,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
        snapshot::LintReadSnapshot,
    },
    repair::RepairArtifactStore,
};
use wenlan_types::{
    lint::{LintProfile, LintQuery},
    repair::{ApplyRepairRequest, RepairManifest, RepairTarget},
    repair_plan::{RepairPlanRequest, RepairResolution},
};

async fn prepared_manifest(
    db: &crate::db::MemoryDB,
    store: &RepairArtifactStore,
    writer: RepairWriter,
) -> RepairManifest {
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let plan = crate::repair_plan::prepare_repair_plan(
        db,
        store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    plan.entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest } if manifest.writer() == writer => {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing ready manifest for {writer:?}: {plan:#?}"))
}

async fn prepared_page_manifest(
    db: &crate::db::MemoryDB,
    store: &RepairArtifactStore,
    page_root: &Path,
    writer: RepairWriter,
) -> RepairManifest {
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root),
            true,
        )
        .await
        .unwrap();
    let plan = crate::repair_plan::prepare_repair_plan(
        db,
        store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        Some(page_root),
        1_721_000_000,
    )
    .await
    .unwrap();
    plan.entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest } if manifest.writer() == writer => {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing ready page manifest for {writer:?}: {plan:#?}"))
}

async fn memory_row_state(
    db: &crate::db::MemoryDB,
    source_id: &str,
) -> Vec<(String, String, i64, Option<String>, Option<String>, i64)> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT id,content,pending_revision,supersedes,space,version
               FROM memories WHERE source_id=?1 ORDER BY chunk_index,id",
            libsql::params![source_id],
        )
        .await
        .unwrap();
    let mut values = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        values.push((
            row.get::<String>(0).unwrap(),
            row.get::<String>(1).unwrap(),
            row.get::<i64>(2).unwrap(),
            row.get::<Option<String>>(3).unwrap(),
            row.get::<Option<String>>(4).unwrap(),
            row.get::<i64>(5).unwrap(),
        ));
    }
    values
}

async fn complete_memory_rollback_snapshot(
    db: &crate::db::MemoryDB,
    source_id: &str,
) -> (Vec<String>, Vec<Vec<String>>) {
    let conn = db.conn.lock().await;
    let mut column_rows = conn.query("PRAGMA table_info(memories)", ()).await.unwrap();
    let mut columns = Vec::new();
    while let Some(row) = column_rows.next().await.unwrap() {
        columns.push(row.get::<String>(1).unwrap());
    }
    drop(column_rows);
    let selected = columns
        .iter()
        .map(|column| format!("quote(\"{}\")", column.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(",");
    let mut query_rows = conn
        .query(
            &format!(
                "SELECT {selected} FROM memories
                 WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id"
            ),
            libsql::params![source_id],
        )
        .await
        .unwrap();
    let mut rows = Vec::new();
    while let Some(row) = query_rows.next().await.unwrap() {
        let mut values = Vec::with_capacity(columns.len());
        for index in 0..columns.len() {
            values.push(row.get::<String>(i32::try_from(index).unwrap()).unwrap());
        }
        rows.push(values);
    }
    (columns, rows)
}

fn exact_apply_request(manifest: &RepairManifest) -> ApplyRepairRequest {
    ApplyRepairRequest::try_new(
        manifest.manifest_id().to_string(),
        manifest.manifest_digest().clone(),
        format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        ),
    )
    .unwrap()
}

struct StaleProjectionPlanFixture {
    db: crate::db::MemoryDB,
    _db_dir: tempfile::TempDir,
    page_root: tempfile::TempDir,
    _repair_root: tempfile::TempDir,
    store: RepairArtifactStore,
    manifests: Vec<RepairManifest>,
}

async fn stale_projection_plan_fixture(page_ids: &[&str]) -> StaleProjectionPlanFixture {
    let (db, db_dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let pages = page_ids
        .iter()
        .enumerate()
        .map(|(index, page_id)| {
            (
                (*page_id).to_string(),
                serde_json::json!({
                    "file": format!("{index}.md"),
                    "version": 1,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 2,
            "pages": pages,
        }))
        .unwrap(),
    )
    .unwrap();
    for (index, page_id) in page_ids.iter().enumerate() {
        std::fs::write(
            page_root.path().join(format!("{index}.md")),
            format!("---\norigin_id: {page_id}\norigin_version: 1\n---\n{page_id}\n"),
        )
        .unwrap();
    }
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let plan = crate::repair_plan::prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifests = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(manifests.len(), page_ids.len(), "plan: {plan:#?}");
    StaleProjectionPlanFixture {
        db,
        _db_dir: db_dir,
        page_root,
        _repair_root: repair_root,
        store,
        manifests,
    }
}

fn stale_projection_manifest_paths(manifest: &RepairManifest) -> (&str, &str) {
    match manifest.mutation() {
        RepairMutation::QuarantineStalePageProjection {
            source_path,
            quarantine_path,
        } => (source_path, quarantine_path),
        mutation => panic!("unexpected stale projection mutation: {mutation:?}"),
    }
}

fn stale_projection_manifest_page_id(manifest: &RepairManifest) -> &str {
    match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id,
        target => panic!("unexpected stale projection target: {target:?}"),
    }
}

fn persist_test_stale_projection_apply_journal(
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    rollback: &crate::repair::StoredRollbackArtifact,
) {
    #[derive(serde::Serialize)]
    struct Draft<'a> {
        format_version: u16,
        manifest_id: &'a str,
        manifest_digest: &'a wenlan_types::repair::RepairDigest,
        rollback: &'a crate::repair::StoredRollbackArtifact,
    }
    #[derive(serde::Serialize)]
    struct Journal<'a> {
        format_version: u16,
        manifest_id: &'a str,
        manifest_digest: &'a wenlan_types::repair::RepairDigest,
        rollback: &'a crate::repair::StoredRollbackArtifact,
        journal_digest: wenlan_types::repair::RepairDigest,
    }
    let draft = Draft {
        format_version: 1,
        manifest_id: manifest.manifest_id(),
        manifest_digest: manifest.manifest_digest(),
        rollback,
    };
    let journal_digest = crate::repair::repair_digest(&serde_json::to_vec(&draft).unwrap());
    let journal = Journal {
        format_version: draft.format_version,
        manifest_id: draft.manifest_id,
        manifest_digest: draft.manifest_digest,
        rollback: draft.rollback,
        journal_digest,
    };
    std::fs::write(
        store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join(".stale-page-projection-apply-journal-v1.json"),
        serde_json::to_vec_pretty(&journal).unwrap(),
    )
    .unwrap();
}

async fn apply_and_verify_stale_projection_manifest(
    fixture: &StaleProjectionPlanFixture,
    manifest: &RepairManifest,
    applied_at: i64,
) -> wenlan_types::repair::RepairApplyReceipt {
    let receipt = crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_apply_request(manifest),
        Some(fixture.page_root.path()),
        applied_at,
    )
    .await
    .unwrap_or_else(|error| panic!("apply failed for {:?}: {error:?}", manifest.target()));
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &fixture.db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(fixture.page_root.path()),
            true,
        )
        .await
        .unwrap();
    crate::repair::record_repair_verification(
        &fixture.db,
        &fixture.store,
        wenlan_types::repair::VerifyRepairRequest::try_new_general_only(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            receipt.receipt_digest().clone(),
            general,
        )
        .unwrap(),
        Some(fixture.page_root.path()),
        applied_at + 1,
    )
    .await
    .unwrap_or_else(|error| panic!("verification failed for {:?}: {error:?}", manifest.target()));
    receipt
}

struct StaleRecoveryFixture {
    db: crate::db::MemoryDB,
    _db_dir: tempfile::TempDir,
    page_root: tempfile::TempDir,
    _repair_root: tempfile::TempDir,
    store: RepairArtifactStore,
    manifest: RepairManifest,
    original_state: Vec<u8>,
    post_state: Vec<u8>,
    source_bytes: Vec<u8>,
    source_path: String,
    quarantine_path: String,
}

impl StaleRecoveryFixture {
    fn pending_path(&self) -> std::path::PathBuf {
        self.store
            .manifest_dir(self.manifest.manifest_id())
            .unwrap()
            .join(".apply-receipt.json.pending")
    }

    fn final_path(&self) -> std::path::PathBuf {
        self.store
            .manifest_dir(self.manifest.manifest_id())
            .unwrap()
            .join("apply-receipt.json")
    }

    fn source(&self) -> std::path::PathBuf {
        self.page_root.path().join(&self.source_path)
    }

    fn quarantine(&self) -> std::path::PathBuf {
        self.page_root.path().join(&self.quarantine_path)
    }

    fn state(&self) -> std::path::PathBuf {
        self.page_root.path().join(".wenlan/state.json")
    }

    fn source_stage_dir(&self) -> std::path::PathBuf {
        self.page_root.path().join(".wenlan").join(
            crate::export::knowledge::projection_unlink_stage_name(self.manifest.manifest_id()),
        )
    }

    fn source_stage(&self) -> std::path::PathBuf {
        self.source_stage_dir().join("source")
    }

    fn source_stage_owner(&self) -> std::path::PathBuf {
        self.source_stage_dir().join("owner.json")
    }

    fn staged_quarantine(&self) -> std::path::PathBuf {
        self.source_stage_dir().join("rollback-quarantine")
    }

    fn page_id(&self) -> &str {
        match self.manifest.target() {
            RepairTarget::PageProjection { page_id, .. } => page_id,
            target => panic!("unexpected stale recovery target: {target:?}"),
        }
    }

    fn ensure_source_stage_dir(&self) {
        let stage = self.source_stage_dir();
        if !stage.exists() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt as _;
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&stage)
                    .unwrap();
            }
            #[cfg(not(unix))]
            std::fs::create_dir(&stage).unwrap();
        }
        let owner = crate::export::knowledge::projection_stage_owner_bytes(
            self.manifest.manifest_id(),
            self.page_id(),
            &self.source_path,
            &self.source_bytes,
        )
        .unwrap();
        std::fs::write(self.source_stage_owner(), owner).unwrap();
    }

    fn ensure_orphaned(&self) {
        let orphaned = self.page_root.path().join(".wenlan/orphaned");
        if orphaned.exists() {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&orphaned)
                .unwrap();
        }
        #[cfg(not(unix))]
        std::fs::create_dir(&orphaned).unwrap();
    }

    fn stage_after_link(&self) {
        self.ensure_orphaned();
        self.ensure_source_stage_dir();
        std::fs::hard_link(self.source(), self.quarantine()).unwrap();
    }

    fn stage_after_unlink(&self) {
        self.stage_after_link();
        std::fs::remove_file(self.source()).unwrap();
    }

    fn stage_after_source_stage(&self) {
        self.stage_after_link();
        std::fs::rename(self.source(), self.source_stage()).unwrap();
    }

    fn stage_after_source_restore_link(&self) {
        self.stage_after_source_stage();
        std::fs::hard_link(self.source_stage(), self.source()).unwrap();
    }

    fn stage_after_quarantine_move(&self) {
        self.stage_after_link();
        std::fs::rename(self.quarantine(), self.staged_quarantine()).unwrap();
    }

    fn stage_post(&self) {
        self.stage_after_unlink();
        std::fs::write(self.state(), &self.post_state).unwrap();
    }

    async fn apply(
        &self,
        now_epoch: i64,
    ) -> Result<wenlan_types::repair::RepairApplyReceipt, crate::error::WenlanError> {
        crate::repair::apply_repair_with_pages(
            &self.db,
            &self.store,
            exact_apply_request(&self.manifest),
            Some(self.page_root.path()),
            now_epoch,
        )
        .await
    }
}

async fn stale_recovery_fixture(page_id: &str, preexisting_orphan: bool) -> StaleRecoveryFixture {
    let (db, db_dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    if preexisting_orphan {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(page_root.path().join(".wenlan/orphaned"))
                .unwrap();
        }
        #[cfg(not(unix))]
        std::fs::create_dir(page_root.path().join(".wenlan/orphaned")).unwrap();
        std::fs::write(
            page_root.path().join(".wenlan/orphaned/baseline.md"),
            b"baseline orphan bytes",
        )
        .unwrap();
    }
    let source_path = format!("{page_id}.md");
    let original_state = format!(
        "{{\r\n  \"custom\":{{\"preserve\":true}},\r\n  \"pages\":{{\"{page_id}\":{{\"file\":\"{source_path}\",\"version\":1}},\"page_other\":{{\"file\":\"other.md\",\"version\":2}}}},\r\n  \"schema_version\":2\r\n}}\r\n"
    )
    .into_bytes();
    let post_state =
        crate::lint::pages::state::remove_unique_page_member(&original_state, page_id).unwrap();
    let mut source_bytes =
        format!("---\r\norigin_id: {page_id}\r\norigin_version: 1\r\n---\r\n").into_bytes();
    source_bytes.extend_from_slice(b"\xffrecovery\r\n");
    std::fs::write(page_root.path().join(".wenlan/state.json"), &original_state).unwrap();
    std::fs::write(page_root.path().join(&source_path), &source_bytes).unwrap();
    std::fs::write(page_root.path().join("other.md"), b"other bytes").unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let (manifest_source, quarantine_path) = match manifest.mutation() {
        RepairMutation::QuarantineStalePageProjection {
            source_path,
            quarantine_path,
        } => (source_path.clone(), quarantine_path.clone()),
        mutation => panic!("unexpected recovery fixture mutation: {mutation:?}"),
    };
    assert_eq!(manifest_source, source_path);
    StaleRecoveryFixture {
        db,
        _db_dir: db_dir,
        page_root,
        _repair_root: repair_root,
        store,
        manifest,
        original_state,
        post_state,
        source_bytes,
        source_path,
        quarantine_path,
    }
}

async fn stale_projection_resolution_kind(
    db: &crate::db::MemoryDB,
    page_root: &Path,
    page_id: &str,
    scope: RepairLintScope,
) -> Option<&'static str> {
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
    let resolutions = resolve_current(&snapshot, &scope, Some(page_root))
        .await
        .unwrap();
    let kind = resolutions.iter().find_map(|resolution| match resolution {
        DeterministicResolution::Exact(exact)
            if matches!(
                &exact.target,
                RepairTarget::PageProjection { page_id: target, .. } if target == page_id
            ) =>
        {
            Some("exact")
        }
        DeterministicResolution::Review(review)
            if review.check_id == IDENTITY_ID && review.affected_record.durable_id() == page_id =>
        {
            Some("review")
        }
        DeterministicResolution::Blocked(blocked)
            if blocked.check_id == IDENTITY_ID
                && blocked.affected_record.durable_id() == page_id =>
        {
            Some("blocked")
        }
        _ => None,
    });
    snapshot.finish().await.unwrap();
    kind
}

#[tokio::test]
async fn memory_and_tag_predicates_split_exact_repairs_from_review() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  supersedes,pinned,confirmed,stability)
             VALUES
                 ('row_exact','exact','memory','mem_exact','exact',0,1,'text',
                  0,0,'hide','fact','   ','mem_exact',0,0,'new'),
                 ('row_review','review','memory','mem_review','review',0,1,'text',
                  0,0,'hide','fact',NULL,NULL,2,0,'unknown');
             INSERT INTO document_tags(source,source_id,tag)
             VALUES ('future','missing','stale'),('memory','missing','');",
        )
        .await
        .unwrap();
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(&snapshot, &RepairLintScope::global(), None)
        .await
        .unwrap();
    let exact_writers = resolutions
        .iter()
        .filter_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact) => Some(exact.writer),
            DeterministicResolution::Review(_) | DeterministicResolution::Blocked(_) => None,
        })
        .collect::<Vec<_>>();
    assert!(exact_writers.contains(&RepairWriter::NormalizeMemorySourceAgent));
    assert!(exact_writers.contains(&RepairWriter::ClearMemorySupersedes));
    assert_eq!(
        exact_writers
            .iter()
            .filter(|writer| **writer == RepairWriter::DeleteTagRow)
            .count(),
        2
    );
    assert!(resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Review(review)
            if review.check_id == MEMORY_STATE
                && review.affected_record.durable_id() == "mem_review"
    )));
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn orphan_revision_is_one_exact_repair_for_both_checks() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO spaces(id,name,created_at,updated_at)
             VALUES ('space_wenlan','wenlan',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  supersedes,pinned,confirmed,stability,space,version)
             VALUES
                 ('row_orphan_revision','independent capture','memory',
                  'mem_orphan_revision','independent',0,1,'text',
                  1,0,'hide','fact','claude-code',
                  NULL,0,0,'learned','wenlan',1);",
        )
        .await
        .unwrap();
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(&snapshot, &RepairLintScope::global(), None)
        .await
        .unwrap();
    let exact = resolutions
        .iter()
        .filter_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact)
                if exact.writer == RepairWriter::UnstageOrphanRevision =>
            {
                Some(exact)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(exact.len(), 1, "one record must not become two proposals");
    assert_eq!(exact[0].source_check_ids, vec![MEMORY_STATE, SUPERSESSION]);
    assert!(matches!(
        exact[0].mutation,
        RepairMutation::UnstageOrphanRevision
    ));
    assert!(!resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Review(review)
            if review.affected_record.durable_id() == "mem_orphan_revision"
    )));
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn orphan_memory_entity_link_resolves_to_one_exact_pair_delete() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "PRAGMA foreign_keys=OFF;
             INSERT INTO entities
                 (id,name,entity_type,confirmed,created_at,updated_at)
             VALUES ('entity_present','present','concept',0,1,1);
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('mem_missing','entity_present');
             PRAGMA foreign_keys=ON;",
        )
        .await
        .unwrap();
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(&snapshot, &RepairLintScope::global(), None)
        .await
        .unwrap();
    let exact = resolutions
        .iter()
        .filter_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact)
                if exact.writer == RepairWriter::DeleteMemoryEntityLink =>
            {
                Some(exact)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].source_check_ids, vec![MEMORY_ENTITY_INTEGRITY]);
    assert!(matches!(
        &exact[0].target,
        RepairTarget::MemoryEntityLink {
            memory_id,
            entity_id,
            ..
        } if memory_id == "mem_missing" && entity_id == "entity_present"
    ));
    assert!(matches!(
        &exact[0].mutation,
        RepairMutation::DeleteMemoryEntityLink {
            memory_id,
            entity_id,
        } if memory_id == "mem_missing" && entity_id == "entity_present"
    ));
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn missing_entity_link_uses_memory_scope_and_exact_pair_rollback() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO spaces(id,name,created_at,updated_at)
             VALUES ('space_wenlan','wenlan',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  supersedes,pinned,confirmed,stability,space,version)
             VALUES
                 ('row_present','present','memory','mem_present','present',0,1,'text',
                  0,0,'hide','fact','test',NULL,0,0,'new','wenlan',1);
             PRAGMA foreign_keys=OFF;
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('mem_present','entity_missing');
             PRAGMA foreign_keys=ON;",
        )
        .await
        .unwrap();
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(
        &snapshot,
        &RepairLintScope::registered("wenlan".to_string()).unwrap(),
        None,
    )
    .await
    .unwrap();
    let exact = resolutions
        .iter()
        .find_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact)
                if exact.writer == RepairWriter::DeleteMemoryEntityLink =>
            {
                Some(exact)
            }
            _ => None,
        })
        .expect("missing entity pair is repairable in the memory scope");
    assert!(matches!(
        &exact.target,
        RepairTarget::MemoryEntityLink {
            memory_id,
            entity_id,
            scope: RepairScope::Registered { space },
        } if memory_id == "mem_present"
            && entity_id == "entity_missing"
            && space == "wenlan"
    ));
    assert_eq!(exact.rollback.table, "memory_entities");
    assert_eq!(
        exact.rollback.rows,
        vec![vec![
            "mem_present".to_string(),
            "entity_missing".to_string()
        ]]
    );
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn imported_document_owner_with_missing_entity_prepares_and_applies_in_scope() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type)
             VALUES
                 ('row_doc','Imported Topic','local_files','/tmp/doc.md','doc',0,1,'text');
             PRAGMA foreign_keys=OFF;
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('/tmp/doc.md','entity_missing');
             PRAGMA foreign_keys=ON;",
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_manifest(&db, &store, RepairWriter::DeleteMemoryEntityLink).await;
    assert!(matches!(
        manifest.target(),
        RepairTarget::MemoryEntityLink {
            memory_id,
            entity_id,
            scope: RepairScope::Uncategorized,
        } if memory_id == "/tmp/doc.md" && entity_id == "entity_missing"
    ));

    let receipt =
        crate::repair::apply_repair(&db, &store, exact_apply_request(&manifest), 1_721_000_001)
            .await
            .expect("an imported document row is a legitimate junction owner");
    assert_eq!(receipt.writer(), RepairWriter::DeleteMemoryEntityLink);
    let conn = db.conn.lock().await;
    let link_count = conn
        .query(
            "SELECT COUNT(*) FROM memory_entities
             WHERE memory_id='/tmp/doc.md' AND entity_id='entity_missing'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    let owner_count = conn
        .query(
            "SELECT COUNT(*) FROM memories
             WHERE source='local_files' AND source_id='/tmp/doc.md'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    assert_eq!(link_count, 0);
    assert_eq!(owner_count, 1);
}

#[tokio::test]
async fn orphan_revision_cas_rejects_scope_state_and_noop_then_changes_only_pending_revision() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO spaces(id,name,created_at,updated_at)
             VALUES ('space_wenlan','wenlan',1,1),('space_other','other',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  supersedes,pinned,confirmed,stability,space,version)
             VALUES
                 ('row_orphan_0','first','memory','mem_orphan','first',0,1,'text',
                  1,0,'hide','fact','test',NULL,0,0,'new','wenlan',7),
                 ('row_orphan_1','second','memory','mem_orphan','second',1,1,'text',
                  1,0,'hide','fact','test',NULL,0,0,'new','wenlan',7),
                 ('row_unrelated','untouched','memory','mem_unrelated','untouched',0,1,'text',
                  0,0,'hide','fact','test',NULL,0,0,'new','wenlan',1);",
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let expected_rollback = complete_memory_rollback_snapshot(&db, "mem_orphan").await;
    let manifest = prepared_manifest(&db, &store, RepairWriter::UnstageOrphanRevision).await;
    assert_eq!(
        manifest.post_assertions().allowed_non_target_check_deltas(),
        ["memories.supersession_integrity"]
    );
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(rollback.table, "memories");
    assert_eq!(rollback.source_id, "mem_orphan");
    assert_eq!(rollback.columns, expected_rollback.0);
    assert_eq!(rollback.rows, expected_rollback.1);
    assert_eq!(rollback.rows.len(), 2);

    let direct_apply = || async {
        crate::post_write::apply_deterministic_repair_cas(&db, &manifest, &[], |_| Ok(())).await
    };

    db.conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET space='other' WHERE source_id='mem_orphan'",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET space='wenlan' WHERE source_id='mem_orphan'",
            (),
        )
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET supersedes='mem_missing' WHERE source_id='mem_orphan'",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET supersedes=NULL WHERE source_id='mem_orphan'",
            (),
        )
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET pending_revision=0 WHERE source_id='mem_orphan'",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET pending_revision=1 WHERE source_id='mem_orphan'",
            (),
        )
        .await
        .unwrap();

    let before = memory_row_state(&db, "mem_orphan").await;
    let receipt =
        crate::repair::apply_repair(&db, &store, exact_apply_request(&manifest), 1_721_000_001)
            .await
            .unwrap();
    assert_eq!(receipt.writer(), RepairWriter::UnstageOrphanRevision);
    assert_eq!(receipt.actual_effects(), manifest.allowed_effects());
    let after = memory_row_state(&db, "mem_orphan").await;
    assert_eq!(before.len(), after.len());
    for (before, after) in before.iter().zip(&after) {
        assert_eq!(before.0, after.0);
        assert_eq!(before.1, after.1);
        assert_eq!(before.3, after.3);
        assert_eq!(before.4, after.4);
        assert_eq!(before.5, after.5);
        assert_eq!(before.2, 1);
        assert_eq!(after.2, 0);
    }
    let conn = db.conn.lock().await;
    let orphan_pending = conn
        .query(
            "SELECT pending_revision FROM memories
             WHERE source_id='mem_orphan' ORDER BY chunk_index",
            (),
        )
        .await
        .unwrap();
    let mut orphan_pending = orphan_pending;
    let mut values = Vec::new();
    while let Some(row) = orphan_pending.next().await.unwrap() {
        values.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(values, vec![0, 0]);
    let unrelated = conn
        .query(
            "SELECT content,pending_revision,space,version FROM memories
             WHERE source_id='mem_unrelated'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(unrelated.get::<String>(0).unwrap(), "untouched");
    assert_eq!(unrelated.get::<i64>(1).unwrap(), 0);
    assert_eq!(
        unrelated.get::<Option<String>>(2).unwrap().as_deref(),
        Some("wenlan")
    );
    assert_eq!(unrelated.get::<i64>(3).unwrap(), 1);
    drop(conn);
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_link_cas_stales_on_restored_owner_and_deletes_only_the_exact_pair() {
    let (db, _dir) = test_db().await;
    let db = std::sync::Arc::new(db);
    db.conn
        .lock()
        .await
        .execute_batch(
            "PRAGMA foreign_keys=OFF;
             INSERT INTO entities(id,name,entity_type,confirmed,created_at,updated_at)
             VALUES
                 ('entity_target','target','concept',0,1,1),
                 ('entity_other','other','concept',0,1,1);
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES
                 ('mem_missing','entity_target'),
                 ('mem_other_missing','entity_other');
             PRAGMA foreign_keys=ON;",
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_manifest(&db, &store, RepairWriter::DeleteMemoryEntityLink).await;
    assert_eq!(manifest.rollback().format_version(), 1);
    assert_eq!(
        manifest.post_assertions().target_check_id(),
        MEMORY_ENTITY_INTEGRITY
    );
    let (target_memory, target_entity) = match manifest.target() {
        RepairTarget::MemoryEntityLink {
            memory_id,
            entity_id,
            ..
        } => (memory_id.clone(), entity_id.clone()),
        other => panic!("unexpected target: {other:?}"),
    };
    let rollback: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        rollback["rows"],
        serde_json::json!([[target_memory, target_entity]])
    );

    {
        let conn = db.conn.lock().await;
        let mut rows = conn.query("PRAGMA busy_timeout=2000", ()).await.unwrap();
        while rows.next().await.unwrap().is_some() {}
    }
    let concurrent = db._db.connect().unwrap();
    let mut rows = concurrent
        .query("PRAGMA busy_timeout=2000", ())
        .await
        .unwrap();
    while rows.next().await.unwrap().is_some() {}
    concurrent.execute("BEGIN IMMEDIATE", ()).await.unwrap();
    concurrent
        .execute(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type)
             VALUES (?1,'restored','memory',?2,'restored',0,1,'text')",
            libsql::params![format!("row_{target_memory}"), target_memory.clone()],
        )
        .await
        .unwrap();
    let (first_poll_tx, first_poll_rx) = tokio::sync::oneshot::channel();
    let cas_db = std::sync::Arc::clone(&db);
    let cas_manifest = manifest.clone();
    let mut cas_task = tokio::spawn(async move {
        let mut cas = Box::pin(crate::post_write::apply_deterministic_repair_cas(
            &cas_db,
            &cas_manifest,
            &[],
            |_| Ok(()),
        ));
        let mut first_poll_tx = Some(first_poll_tx);
        std::future::poll_fn(move |cx| {
            if let Some(first_poll_tx) = first_poll_tx.take() {
                first_poll_tx.send(()).unwrap();
            }
            std::future::Future::poll(cas.as_mut(), cx)
        })
        .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(1), first_poll_rx)
        .await
        .expect("the CAS future must be polled while the competing lock is held")
        .unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), &mut cas_task)
            .await
            .is_err(),
        "the CAS future must remain pending while the competing lock is held"
    );
    concurrent.execute("COMMIT", ()).await.unwrap();
    let write_result = tokio::time::timeout(std::time::Duration::from_secs(5), &mut cas_task)
        .await
        .expect("the blocked BEGIN IMMEDIATE resumes after the competing commit")
        .unwrap();
    let error = write_result.unwrap_err();
    assert!(error.to_string().contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "DELETE FROM memories WHERE source_id=?1",
            libsql::params![target_memory.clone()],
        )
        .await
        .unwrap();

    let receipt =
        crate::repair::apply_repair(&db, &store, exact_apply_request(&manifest), 1_721_000_001)
            .await
            .unwrap();
    assert_eq!(receipt.writer(), RepairWriter::DeleteMemoryEntityLink);
    assert_eq!(receipt.actual_effects(), manifest.allowed_effects());
    let conn = db.conn.lock().await;
    let target_count = conn
        .query(
            "SELECT COUNT(*) FROM memory_entities
             WHERE memory_id=?1 AND entity_id=?2",
            libsql::params![target_memory, target_entity],
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    let all_link_count = conn
        .query("SELECT COUNT(*) FROM memory_entities", ())
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    let entity_count = conn
        .query("SELECT COUNT(*) FROM entities", ())
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    assert_eq!(target_count, 0);
    assert_eq!(all_link_count, 1);
    assert_eq!(entity_count, 2);
    drop(conn);
    assert!(
        crate::post_write::apply_deterministic_repair_cas(&db, &manifest, &[], |_| Ok(()))
            .await
            .unwrap_err()
            .to_string()
            .contains("repair_target_stale")
    );
}

#[tokio::test]
async fn registered_scope_does_not_resolve_global_tag_findings() {
    let (db, _dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO document_tags(source,source_id,tag)
             VALUES ('future','missing','stale'),('memory','missing','');",
        )
        .await
        .unwrap();
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(
        &snapshot,
        &RepairLintScope::registered("work".to_string()).unwrap(),
        None,
    )
    .await
    .unwrap();
    let tag_repairs = resolutions
        .iter()
        .filter_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact) if exact.writer == RepairWriter::DeleteTagRow => {
                Some(&exact.target)
            }
            DeterministicResolution::Exact(_)
            | DeterministicResolution::Review(_)
            | DeterministicResolution::Blocked(_) => None,
        })
        .collect::<Vec<_>>();

    assert!(
        tag_repairs.is_empty(),
        "a registered repair scope must not expose global tag mutations"
    );
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn orphan_links_bind_only_one_active_same_scope_target() {
    let (db, _dir) = test_db().await;
    let now = "2026-07-15T00:00:00Z";
    db.insert_page(
        "source_unique",
        "Source Unique",
        None,
        "see [[Unique Topic]]",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    db.insert_page(
        "source_ambiguous",
        "Source Ambiguous",
        None,
        "see [[Ambiguous Topic]]",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    db.insert_page(
        "source_missing",
        "Source Missing",
        None,
        "see [[Never Created Topic]]",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    db.insert_page(
        "target_unique",
        "Unique Topic",
        None,
        "target",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    for id in ["target_ambiguous_a", "target_ambiguous_b"] {
        db.insert_page(
            id,
            "Ambiguous Topic",
            None,
            "target",
            None,
            Some("work"),
            &[],
            now,
        )
        .await
        .unwrap();
    }
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(
        &snapshot,
        &RepairLintScope::registered("work".to_string()).unwrap(),
        None,
    )
    .await
    .unwrap();
    assert!(resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Exact(exact)
            if exact.writer == RepairWriter::BindPageLink
                && matches!(
                    &exact.mutation,
                    RepairMutation::BindPageLink { after_target_page_id, .. }
                        if after_target_page_id == "target_unique"
                )
    )));
    assert!(resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Review(review)
            if review.check_id == ORPHAN_LABELS
                && review.affected_record.durable_id().contains("source_ambiguous")
    )));
    assert!(resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Blocked(blocked)
            if blocked.check_id == ORPHAN_LABELS
                && blocked.affected_record.durable_id().contains("source_missing")
                && blocked.next_action.contains("create")
    )));
    assert!(!resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Review(review)
            if review.check_id == ORPHAN_LABELS
                && review.affected_record.durable_id().contains("source_missing")
    )));
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn page_projection_reserved_target_is_blocked() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    let page_id = "page_reserved_projection";
    let now = "2026-07-15T00:00:00Z";
    db.insert_page_with_kind(
        page_id,
        "Reserved Projection",
        None,
        "canonical body",
        None,
        Some("work"),
        &[],
        now,
        "distilled",
        "confirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    let page = db.get_page(page_id).await.unwrap().unwrap();
    crate::export::knowledge::KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
        .write_page(&page)
        .unwrap();
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET version=version+1 WHERE id=?1",
            libsql::params![page_id],
        )
        .await
        .unwrap();

    let state_path = page_root.path().join(".wenlan/state.json");
    let mut state =
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&state_path).unwrap()).unwrap();
    state["pages"][page_id]["file"] = serde_json::Value::String("_sources/reserved.md".to_string());
    std::fs::write(&state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();
    std::fs::create_dir_all(page_root.path().join("_sources")).unwrap();
    std::fs::write(
        page_root.path().join("_sources/reserved.md"),
        format!("---\norigin_id: {page_id}\norigin_version: 1\n---\n\nsource bytes\n"),
    )
    .unwrap();

    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
    let resolutions = resolve_current(
        &snapshot,
        &RepairLintScope::registered("work".to_string()).unwrap(),
        Some(page_root.path()),
    )
    .await
    .unwrap();

    assert!(
        resolutions.iter().any(|resolution| matches!(
            resolution,
            DeterministicResolution::Blocked(blocked)
                if blocked.check_id == PROJECTION_VERSION
                    && blocked.affected_record.durable_id() == page_id
        )),
        "resolutions: {resolutions:#?}"
    );
    assert!(!resolutions.iter().any(|resolution| matches!(
        resolution,
        DeterministicResolution::Exact(exact)
            if exact.writer == RepairWriter::RegeneratePageProjection
                && matches!(
                    &exact.target,
                    RepairTarget::PageProjection { page_id: target_id, .. }
                        if target_id == page_id
                )
    )));
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn stale_page_projection_owner_is_one_exact_global_repair() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        br#"{"schema_version":2,"pages":{"page_stale_owner":{"file":"stale-owner.md","version":7,"last_written":"raw"}}}"#,
    )
    .unwrap();
    std::fs::write(
        page_root.path().join("stale-owner.md"),
        b"---\r\norigin_id: page_stale_owner\r\norigin_version: 7\r\n---\r\nbody\r\n",
    )
    .unwrap();
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(
        &snapshot,
        &RepairLintScope::global(),
        Some(page_root.path()),
    )
    .await
    .unwrap();
    let exact = resolutions
        .iter()
        .filter_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact) if exact.source_check_ids == [IDENTITY_ID] => {
                Some(exact)
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(exact.len(), 1, "resolutions: {resolutions:#?}");
    assert!(matches!(
        exact[0].target,
        RepairTarget::PageProjection { ref page_id, .. } if page_id == "page_stale_owner"
    ));
    assert_eq!(exact[0].expected_version, None);
    assert_eq!(exact[0].rollback.source_id, "page_stale_owner");
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn stale_page_projection_prepare_rejects_cooperating_writer_contention() {
    const CHILD_ROOT: &str = "WENLAN_STALE_PREPARE_LOCK_CHILD_ROOT";
    if let Some(page_root) = std::env::var_os(CHILD_ROOT) {
        let (db, _dir) = test_db().await;
        let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
        let error = capture_stale_page_projection_rollback(
            &snapshot,
            Path::new(&page_root),
            "page_stale_prepare_lock",
            "stale.md",
            ".wenlan/orphaned/page_stale_prepare_lock.md",
        )
        .await
        .expect_err("prepare capture must participate in the projection lock");
        assert_eq!(error.to_string(), "Conflict: page_projection_locked");
        snapshot.finish().await.unwrap();
        return;
    }

    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        br#"{"schema_version":2,"pages":{"page_stale_prepare_lock":{"file":"stale.md","version":1}}}"#,
    )
    .unwrap();
    std::fs::write(
        page_root.path().join("stale.md"),
        b"---\norigin_id: page_stale_prepare_lock\norigin_version: 1\n---\nbody\n",
    )
    .unwrap();

    crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        page_root.path().to_path_buf(),
        &db,
        |_repair| {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "repair_plan::deterministic::tests::stale_page_projection_prepare_rejects_cooperating_writer_contention",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ROOT, page_root.path())
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "prepare lock child probe failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            Ok(())
        },
    )
    .unwrap();
}

#[tokio::test]
async fn stale_page_projection_ownership_hazards_are_review_or_blocked() {
    let (db, _dir) = test_db().await;
    let cases = [
        (
            "page_wrong_origin",
            br#"{"schema_version":2,"pages":{"page_wrong_origin":{"file":"wrong.md","version":1}}}"#
                .as_slice(),
            &[(
                "wrong.md",
                b"---\norigin_id: page_someone_else\norigin_version: 1\n---\nbody\n".as_slice(),
            )][..],
            "review",
        ),
        (
            "page_missing_target",
            br#"{"schema_version":2,"pages":{"page_missing_target":{"file":"missing.md","version":1}}}"#
                .as_slice(),
            &[][..],
            "blocked",
        ),
        (
            "page_malformed_entry",
            br#"{"schema_version":2,"pages":{"page_malformed_entry":"bad"}}"#.as_slice(),
            &[][..],
            "blocked",
        ),
        (
            "page_reserved_target",
            br#"{"schema_version":2,"pages":{"page_reserved_target":{"file":"_sources/reserved.md","version":1}}}"#
                .as_slice(),
            &[(
                "_sources/reserved.md",
                b"---\norigin_id: page_reserved_target\n---\n".as_slice(),
            )][..],
            "blocked",
        ),
        (
            "page_shared_target",
            br#"{"schema_version":2,"pages":{"page_shared_target":{"file":"shared.md","version":1},"page_other":{"file":"shared.md","version":1}}}"#
                .as_slice(),
            &[(
                "shared.md",
                b"---\norigin_id: page_shared_target\n---\n".as_slice(),
            )][..],
            "blocked",
        ),
        (
            "page_casefold_target",
            br#"{"schema_version":2,"pages":{"page_casefold_target":{"file":"Case.md","version":1},"page_other":{"file":"case.md","version":1}}}"#
                .as_slice(),
            &[
                (
                    "Case.md",
                    b"---\norigin_id: page_casefold_target\n---\n".as_slice(),
                ),
                ("case.md", b"---\norigin_id: page_other\n---\n".as_slice()),
            ][..],
            "blocked",
        ),
        (
            "page_quarantine_collision",
            br#"{"schema_version":2,"pages":{"page_quarantine_collision":{"file":"collision.md","version":1}}}"#
                .as_slice(),
            &[
                (
                    "collision.md",
                    b"---\norigin_id: page_quarantine_collision\n---\n".as_slice(),
                ),
                (
                    ".wenlan/orphaned/page_quarantine_collision.md",
                    b"do not clobber".as_slice(),
                ),
            ][..],
            "blocked",
        ),
    ];
    for (page_id, state, files, expected) in cases {
        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        std::fs::write(page_root.path().join(".wenlan/state.json"), state).unwrap();
        for (relative, bytes) in files {
            let path = page_root.path().join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, bytes).unwrap();
        }
        assert_eq!(
            stale_projection_resolution_kind(
                &db,
                page_root.path(),
                page_id,
                RepairLintScope::global(),
            )
            .await,
            Some(expected),
            "case {page_id}"
        );
    }
}

#[tokio::test]
async fn stale_page_projection_duplicate_state_keys_are_blocked() {
    let (db, _dir) = test_db().await;
    let cases = [
        (
            "duplicate root pages",
            "page_duplicate_root",
            br#"{"schema_version":2,"pages":{"page_other":{"file":"other.md","version":1}},"pages":{"page_duplicate_root":{"file":"target.md","version":1}}}"#
                .as_slice(),
        ),
        (
            "duplicate target page id",
            "page_duplicate_target",
            br#"{"schema_version":2,"pages":{"page_duplicate_target":{"file":"other.md","version":1},"page_duplicate_target":{"file":"target.md","version":1}}}"#
                .as_slice(),
        ),
        (
            "duplicate target file field",
            "page_duplicate_file",
            br#"{"schema_version":2,"pages":{"page_duplicate_file":{"file":"other.md","file":"target.md","version":1}}}"#
                .as_slice(),
        ),
        (
            "duplicate target version field",
            "page_duplicate_version",
            br#"{"schema_version":2,"pages":{"page_duplicate_version":{"file":"target.md","version":0,"version":1}}}"#
                .as_slice(),
        ),
    ];

    for (case, page_id, raw_state) in cases {
        let page_root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
        std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
        std::fs::write(
            page_root.path().join("target.md"),
            format!("---\norigin_id: {page_id}\norigin_version: 1\n---\nbody\n"),
        )
        .unwrap();
        std::fs::write(
            page_root.path().join("other.md"),
            format!("---\norigin_id: {page_id}\norigin_version: 1\n---\nbody\n"),
        )
        .unwrap();

        assert_eq!(
            stale_projection_resolution_kind(
                &db,
                page_root.path(),
                page_id,
                RepairLintScope::global(),
            )
            .await,
            Some("blocked"),
            "case {case}"
        );
    }
}

#[tokio::test]
async fn stale_page_projection_registered_scope_does_not_infer_authority() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        br#"{"schema_version":2,"pages":{"page_stale_scoped":{"file":"scoped.md","version":1}}}"#,
    )
    .unwrap();
    std::fs::write(
        page_root.path().join("scoped.md"),
        b"---\norigin_id: page_stale_scoped\norigin_version: 1\n---\nbody\n",
    )
    .unwrap();

    assert_eq!(
        stale_projection_resolution_kind(
            &db,
            page_root.path(),
            "page_stale_scoped",
            RepairLintScope::registered("work".to_string()).unwrap(),
        )
        .await,
        None
    );
}

#[cfg(unix)]
#[tokio::test]
async fn stale_page_projection_symlink_target_and_quarantine_ancestor_are_blocked() {
    use std::os::unix::fs::symlink;

    let (db, _dir) = test_db().await;
    let outside = tempfile::TempDir::new().unwrap();
    std::fs::write(
        outside.path().join("stale.md"),
        b"---\norigin_id: page_symlink_target\n---\n",
    )
    .unwrap();

    let target_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(target_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        target_root.path().join(".wenlan/state.json"),
        br#"{"schema_version":2,"pages":{"page_symlink_target":{"file":"linked/stale.md","version":1}}}"#,
    )
    .unwrap();
    symlink(outside.path(), target_root.path().join("linked")).unwrap();
    assert_eq!(
        stale_projection_resolution_kind(
            &db,
            target_root.path(),
            "page_symlink_target",
            RepairLintScope::global(),
        )
        .await,
        Some("blocked")
    );

    let ancestor_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(ancestor_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        ancestor_root.path().join(".wenlan/state.json"),
        br#"{"schema_version":2,"pages":{"page_symlink_quarantine":{"file":"stale.md","version":1}}}"#,
    )
    .unwrap();
    std::fs::write(
        ancestor_root.path().join("stale.md"),
        b"---\norigin_id: page_symlink_quarantine\n---\n",
    )
    .unwrap();
    symlink(
        outside.path(),
        ancestor_root.path().join(".wenlan/orphaned"),
    )
    .unwrap();
    assert_eq!(
        stale_projection_resolution_kind(
            &db,
            ancestor_root.path(),
            "page_symlink_quarantine",
            RepairLintScope::global(),
        )
        .await,
        Some("blocked")
    );
}

#[tokio::test]
async fn stale_page_projection_apply_quarantines_exact_bytes_and_verifies() {
    use wenlan_types::repair::VerifyRepairRequest;

    let (db, _dir) = test_db().await;
    db.insert_page(
        "page_other",
        "Other",
        None,
        "other",
        None,
        None,
        &[],
        "2026-07-17T00:00:00Z",
    )
    .await
    .unwrap();
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let raw_state = b"{\r\n\t\"custom\": { \"preserve\":true },\r\n\t\"pages\": {\r\n\t\t\"page_before\": {\"file\":\"before.md\", \"version\":3},\r\n\t\t\"page_stale_apply\": { \"version\":7, \"file\":\"stale.md\" },\r\n\t\t\"page_other\":{\t\"file\":\"other.md\",\"version\":1}\r\n\t},\r\n\t\"schema_version\": 2\r\n}\r\n";
    let expected_state = b"{\r\n\t\"custom\": { \"preserve\":true },\r\n\t\"pages\": {\r\n\t\t\"page_before\": {\"file\":\"before.md\", \"version\":3},\r\n\t\t\"page_other\":{\t\"file\":\"other.md\",\"version\":1}\r\n\t},\r\n\t\"schema_version\": 2\r\n}\r\n";
    let target_bytes =
        b"---\r\norigin_id: page_stale_apply\r\norigin_version: 7\r\n---\r\nbody:\xff\r\n";
    let other_bytes = b"---\norigin_id: page_other\norigin_version: 1\n---\nother\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("stale.md"), target_bytes).unwrap();
    std::fs::write(page_root.path().join("other.md"), other_bytes).unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;

    let receipt = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();
    let quarantine = page_root
        .path()
        .join(".wenlan/orphaned/page_stale_apply.md");
    assert!(!page_root.path().join("stale.md").exists());
    assert_eq!(std::fs::read(&quarantine).unwrap(), target_bytes);
    assert_eq!(
        std::fs::read(page_root.path().join("other.md")).unwrap(),
        other_bytes
    );
    let written_state = std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap();
    assert_eq!(written_state, expected_state);
    let state: serde_json::Value = serde_json::from_slice(&written_state).unwrap();
    assert!(state["pages"].get("page_stale_apply").is_none());
    assert_eq!(state["pages"]["page_other"]["file"], "other.md");
    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert_eq!(
        crate::repair::apply_repair_with_pages(
            &db,
            &store,
            exact_apply_request(&manifest),
            Some(page_root.path()),
            1_721_000_002,
        )
        .await
        .unwrap()
        .receipt_digest(),
        receipt.receipt_digest()
    );

    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    crate::repair::record_repair_verification(
        &db,
        &store,
        VerifyRepairRequest::try_new_general_only(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            receipt.receipt_digest().clone(),
            general,
        )
        .unwrap(),
        Some(page_root.path()),
        1_721_000_003,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn stale_page_projection_manifests_from_one_plan_apply_sequentially() {
    let fixture = stale_projection_plan_fixture(&["page_stale_a", "page_stale_b"]).await;
    for (index, manifest) in fixture.manifests.iter().enumerate() {
        apply_and_verify_stale_projection_manifest(
            &fixture,
            manifest,
            1_721_000_001 + 2 * i64::try_from(index).unwrap(),
        )
        .await;
    }

    for manifest in &fixture.manifests {
        let (source_path, quarantine_path) = stale_projection_manifest_paths(manifest);
        assert!(!fixture.page_root.path().join(source_path).exists());
        assert!(fixture.page_root.path().join(quarantine_path).is_file());
    }
}

#[tokio::test]
async fn stale_page_projection_plan_allows_an_ordered_subset() {
    let fixture = stale_projection_plan_fixture(&[
        "page_stale_subset_a",
        "page_stale_subset_b",
        "page_stale_subset_c",
    ])
    .await;
    for (sequence, index) in [0_usize, 2].into_iter().enumerate() {
        let manifest = &fixture.manifests[index];
        apply_and_verify_stale_projection_manifest(
            &fixture,
            manifest,
            1_721_000_101 + 2 * i64::try_from(sequence).unwrap(),
        )
        .await;
    }

    for index in [0_usize, 2] {
        let (source_path, quarantine_path) =
            stale_projection_manifest_paths(&fixture.manifests[index]);
        assert!(!fixture.page_root.path().join(source_path).exists());
        assert!(fixture.page_root.path().join(quarantine_path).is_file());
    }
    let (skipped_source, skipped_quarantine) =
        stale_projection_manifest_paths(&fixture.manifests[1]);
    assert!(fixture.page_root.path().join(skipped_source).is_file());
    assert!(!fixture.page_root.path().join(skipped_quarantine).exists());
}

#[tokio::test]
async fn stale_page_projection_plan_allows_only_the_last_manifest() {
    let fixture =
        stale_projection_plan_fixture(&["page_stale_last_only_a", "page_stale_last_only_b"]).await;
    let skipped = &fixture.manifests[0];
    let selected = &fixture.manifests[1];

    apply_and_verify_stale_projection_manifest(&fixture, selected, 1_721_000_151).await;

    let (skipped_source, skipped_quarantine) = stale_projection_manifest_paths(skipped);
    let (selected_source, selected_quarantine) = stale_projection_manifest_paths(selected);
    assert!(fixture.page_root.path().join(skipped_source).is_file());
    assert!(!fixture.page_root.path().join(skipped_quarantine).exists());
    assert!(!fixture.page_root.path().join(selected_source).exists());
    assert!(fixture.page_root.path().join(selected_quarantine).is_file());
}

#[tokio::test]
async fn stale_page_projection_recovery_preserves_an_earlier_selected_repair() {
    let fixture =
        stale_projection_plan_fixture(&["page_stale_recovery_a", "page_stale_recovery_b"]).await;
    let first = &fixture.manifests[0];
    let second = &fixture.manifests[1];
    apply_and_verify_stale_projection_manifest(&fixture, first, 1_721_000_201).await;
    let (_, first_quarantine) = stale_projection_manifest_paths(first);
    let first_quarantine = fixture.page_root.path().join(first_quarantine);
    let first_quarantine_bytes = std::fs::read(&first_quarantine).unwrap();

    let (second_source, second_quarantine) = stale_projection_manifest_paths(second);
    let dynamic_rollback = crate::repair::capture_stale_page_projection_current(
        fixture.page_root.path(),
        stale_projection_manifest_page_id(second),
        second_source,
        second_quarantine,
    )
    .unwrap();
    let second_manifest_dir = fixture.store.manifest_dir(second.manifest_id()).unwrap();
    std::fs::write(second_manifest_dir.join(".apply-receipt.json.pending"), b"").unwrap();
    persist_test_stale_projection_apply_journal(&fixture.store, second, &dynamic_rollback);
    let crash = crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        fixture.page_root.path().to_path_buf(),
        &fixture.db,
        |write| {
            let mut pinned = write.pin_stale_page_projection(
                stale_projection_manifest_page_id(second),
                second_source,
                second_quarantine,
                &dynamic_rollback,
                second.manifest_id(),
            )?;
            pinned.quarantine_with_after_source_stage(|| {
                Err(crate::error::WenlanError::VectorDb(
                    "simulated process crash after source stage".to_string(),
                ))
            })
        },
    )
    .unwrap_err();
    assert!(
        crash
            .to_string()
            .contains("simulated process crash after source stage"),
        "{crash}"
    );

    crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_apply_request(second),
        Some(fixture.page_root.path()),
        1_721_000_203,
    )
    .await
    .unwrap();

    assert_eq!(
        std::fs::read(&first_quarantine).unwrap(),
        first_quarantine_bytes
    );
    assert!(!fixture.page_root.path().join(second_source).exists());
    assert!(fixture.page_root.path().join(second_quarantine).is_file());
}

#[tokio::test]
async fn stale_page_projection_legacy_pending_without_journal_fails_closed() {
    let fixture =
        stale_projection_plan_fixture(&["page_stale_pre_journal_a", "page_stale_pre_journal_b"])
            .await;
    let first = &fixture.manifests[0];
    let second = &fixture.manifests[1];
    apply_and_verify_stale_projection_manifest(&fixture, first, 1_721_000_301).await;
    let (_, first_quarantine) = stale_projection_manifest_paths(first);
    let first_quarantine = fixture.page_root.path().join(first_quarantine);
    let first_quarantine_bytes = std::fs::read(&first_quarantine).unwrap();

    std::fs::write(
        fixture
            .store
            .manifest_dir(second.manifest_id())
            .unwrap()
            .join(".apply-receipt.json.pending"),
        b"",
    )
    .unwrap();

    let error = crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_apply_request(second),
        Some(fixture.page_root.path()),
        1_721_000_303,
    )
    .await
    .unwrap_err();

    let (second_source, second_quarantine) = stale_projection_manifest_paths(second);
    assert_eq!(
        error.to_string(),
        "Conflict: repair_apply_recovery_required"
    );
    assert_eq!(
        std::fs::read(&first_quarantine).unwrap(),
        first_quarantine_bytes
    );
    assert!(fixture.page_root.path().join(second_source).is_file());
    assert!(!fixture.page_root.path().join(second_quarantine).exists());
    assert!(fixture
        .store
        .manifest_dir(second.manifest_id())
        .unwrap()
        .join(".apply-receipt.json.pending")
        .is_file());
}

#[tokio::test]
async fn stale_page_projection_journal_without_pending_preserves_earlier_repair() {
    let fixture = stale_projection_plan_fixture(&[
        "page_stale_valid_pending_a",
        "page_stale_valid_pending_b",
    ])
    .await;
    let first = &fixture.manifests[0];
    let second = &fixture.manifests[1];
    apply_and_verify_stale_projection_manifest(&fixture, first, 1_721_000_401).await;
    let (_, first_quarantine) = stale_projection_manifest_paths(first);
    let first_quarantine = fixture.page_root.path().join(first_quarantine);
    let first_quarantine_bytes = std::fs::read(&first_quarantine).unwrap();

    let (second_source, second_quarantine) = stale_projection_manifest_paths(second);
    let dynamic_rollback = crate::repair::capture_stale_page_projection_current(
        fixture.page_root.path(),
        stale_projection_manifest_page_id(second),
        second_source,
        second_quarantine,
    )
    .unwrap();
    persist_test_stale_projection_apply_journal(&fixture.store, second, &dynamic_rollback);
    let second_manifest_dir = fixture.store.manifest_dir(second.manifest_id()).unwrap();
    assert!(second_manifest_dir
        .join(".stale-page-projection-apply-journal-v1.json")
        .is_file());
    assert!(!second_manifest_dir
        .join(".apply-receipt.json.pending")
        .exists());

    crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_apply_request(second),
        Some(fixture.page_root.path()),
        1_721_000_403,
    )
    .await
    .unwrap();

    assert_eq!(
        std::fs::read(&first_quarantine).unwrap(),
        first_quarantine_bytes
    );
    assert!(!fixture.page_root.path().join(second_source).exists());
    assert!(fixture.page_root.path().join(second_quarantine).is_file());
}

#[tokio::test]
async fn stale_page_projection_pending_journal_is_promoted_and_recovered() {
    let fixture = stale_projection_plan_fixture(&[
        "page_stale_pending_journal_a",
        "page_stale_pending_journal_b",
    ])
    .await;
    let first = &fixture.manifests[0];
    let second = &fixture.manifests[1];
    apply_and_verify_stale_projection_manifest(&fixture, first, 1_721_000_451).await;
    let (_, first_quarantine) = stale_projection_manifest_paths(first);
    let first_quarantine = fixture.page_root.path().join(first_quarantine);
    let first_quarantine_bytes = std::fs::read(&first_quarantine).unwrap();

    let (second_source, second_quarantine) = stale_projection_manifest_paths(second);
    let dynamic_rollback = crate::repair::capture_stale_page_projection_current(
        fixture.page_root.path(),
        stale_projection_manifest_page_id(second),
        second_source,
        second_quarantine,
    )
    .unwrap();
    persist_test_stale_projection_apply_journal(&fixture.store, second, &dynamic_rollback);
    let second_manifest_dir = fixture.store.manifest_dir(second.manifest_id()).unwrap();
    let journal = second_manifest_dir.join(".stale-page-projection-apply-journal-v1.json");
    let pending_journal =
        second_manifest_dir.join(".stale-page-projection-apply-journal-v1.json.pending");
    std::fs::rename(&journal, &pending_journal).unwrap();

    crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_apply_request(second),
        Some(fixture.page_root.path()),
        1_721_000_453,
    )
    .await
    .unwrap();

    assert!(!pending_journal.exists());
    assert_eq!(
        std::fs::read(&first_quarantine).unwrap(),
        first_quarantine_bytes
    );
    assert!(!fixture.page_root.path().join(second_source).exists());
    assert!(fixture.page_root.path().join(second_quarantine).is_file());
}

#[tokio::test]
async fn stale_page_projection_after_plan_collision_is_no_clobber_stale() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let raw_state =
        br#"{"schema_version":2,"pages":{"page_stale_collision":{"file":"stale.md","version":1}}}"#;
    let target_bytes = b"---\norigin_id: page_stale_collision\norigin_version: 1\n---\nsource\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("stale.md"), target_bytes).unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(page_root.path().join(".wenlan/orphaned"))
            .unwrap();
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(page_root.path().join(".wenlan/orphaned")).unwrap();
    let quarantine = page_root
        .path()
        .join(".wenlan/orphaned/page_stale_collision.md");
    std::fs::write(&quarantine, b"preexisting quarantine artifact").unwrap();

    let error = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("repair_target_stale"));
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        raw_state
    );
    assert_eq!(
        std::fs::read(page_root.path().join("stale.md")).unwrap(),
        target_bytes
    );
    assert_eq!(
        std::fs::read(&quarantine).unwrap(),
        b"preexisting quarantine artifact"
    );
    let pending = store
        .manifest_dir(manifest.manifest_id())
        .unwrap()
        .join(".apply-receipt.json.pending");
    assert!(
        !pending.exists(),
        "a pre-journal collision proves zero mutation and must not retain a writer fence"
    );

    std::fs::remove_file(&quarantine).unwrap();
    let receipt = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap();
    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert!(!pending.exists());
    assert_eq!(std::fs::read(&quarantine).unwrap(), target_bytes);
}

#[tokio::test]
async fn stale_page_projection_rejects_db_owner_created_after_plan() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        br#"{"schema_version":2,"pages":{"page_owner_returns":{"file":"owner.md","version":1}}}"#,
    )
    .unwrap();
    let target_bytes = b"---\norigin_id: page_owner_returns\norigin_version: 1\n---\nsource\n";
    std::fs::write(page_root.path().join("owner.md"), target_bytes).unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    db.insert_page(
        "page_owner_returns",
        "Owner Returns",
        None,
        "database owner",
        None,
        None,
        &[],
        "2026-07-17T00:00:00Z",
    )
    .await
    .unwrap();

    let error = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("repair_target_stale"));
    assert_eq!(
        std::fs::read(page_root.path().join("owner.md")).unwrap(),
        target_bytes
    );
    assert!(!page_root.path().join(".wenlan/orphaned").exists());
}

#[tokio::test]
async fn stale_page_projection_forced_failure_restores_raw_state_and_file_bytes() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let raw_state = b"{\"schema_version\":2,\r\n\"pages\":{\"page_stale_rollback\":{\"file\":\"rollback.md\",\"version\":1}}}";
    let target_bytes =
        b"---\r\norigin_id: page_stale_rollback\r\norigin_version: 1\r\n---\r\n\xffrollback\r\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("rollback.md"), target_bytes).unwrap();
    std::fs::write(page_root.path().join("canary.bin"), b"\0canary\xff").unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let error = crate::post_write::quarantine_stale_page_projection_cas(
        &db,
        &manifest,
        &rollback,
        page_root.path(),
        |_| {
            Err(crate::error::WenlanError::Validation(
                "forced_pre_commit".to_string(),
            ))
        },
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Validation error: forced_pre_commit",
        "rollback must not append a hidden cleanup failure"
    );
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        raw_state
    );
    assert_eq!(
        std::fs::read(page_root.path().join("rollback.md")).unwrap(),
        target_bytes
    );
    assert_eq!(
        std::fs::read(page_root.path().join("canary.bin")).unwrap(),
        b"\0canary\xff"
    );
    assert!(!page_root
        .path()
        .join(".wenlan/orphaned/page_stale_rollback.md")
        .exists());
    assert!(page_root.path().join(".wenlan/orphaned").is_dir());
    assert!(std::fs::read_dir(page_root.path().join(".wenlan/orphaned"))
        .unwrap()
        .next()
        .is_none());
}

#[tokio::test]
async fn stale_page_projection_target_edit_between_capture_and_pin_is_zero_mutation_stale() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let raw_state =
        br#"{"schema_version":2,"pages":{"page_stale_pin_race":{"file":"race.md","version":1}}}"#;
    let approved_bytes = b"---\norigin_id: page_stale_pin_race\norigin_version: 1\n---\napproved\n";
    let raced_bytes =
        b"---\norigin_id: page_stale_pin_race\norigin_version: 1\n---\nnoncooperating edit\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("race.md"), approved_bytes).unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let error = crate::post_write::quarantine_stale_page_projection_cas_with_before_pin(
        &db,
        &manifest,
        &rollback,
        page_root.path(),
        || {
            std::fs::write(page_root.path().join("race.md"), raced_bytes)?;
            Ok(())
        },
        |_| Ok(()),
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("repair_target_stale"), "{error}");
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        raw_state
    );
    assert_eq!(
        std::fs::read(page_root.path().join("race.md")).unwrap(),
        raced_bytes
    );
    assert!(!page_root
        .path()
        .join(".wenlan/orphaned/page_stale_pin_race.md")
        .exists());
    assert!(!page_root.path().join(".wenlan/orphaned").exists());
}

#[tokio::test]
async fn stale_page_projection_orphan_addition_after_pin_is_zero_mutation_stale() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let raw_state = br#"{"schema_version":2,"pages":{"page_stale_orphan_race":{"file":"race.md","version":1}}}"#;
    let source_bytes =
        b"---\norigin_id: page_stale_orphan_race\norigin_version: 1\n---\napproved\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("race.md"), source_bytes).unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let error = crate::post_write::quarantine_stale_page_projection_cas_with_after_pin(
        &db,
        &manifest,
        &rollback,
        page_root.path(),
        || {
            std::fs::write(
                page_root.path().join(".wenlan/orphaned/injected.md"),
                b"noncooperating orphan",
            )?;
            Ok(())
        },
        |_| Ok(()),
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("repair_target_stale"), "{error}");
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        raw_state
    );
    assert_eq!(
        std::fs::read(page_root.path().join("race.md")).unwrap(),
        source_bytes
    );
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/orphaned/injected.md")).unwrap(),
        b"noncooperating orphan"
    );
    assert!(!page_root
        .path()
        .join(".wenlan/orphaned/page_stale_orphan_race.md")
        .exists());
}

#[tokio::test]
async fn stale_page_projection_source_replacement_after_pin_is_zero_mutation_stale() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan/orphaned")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(
            page_root.path().join(".wenlan/orphaned"),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
    }
    let raw_state = br#"{"schema_version":2,"pages":{"page_stale_source_race":{"file":"race.md","version":1}}}"#;
    let approved_bytes =
        b"---\norigin_id: page_stale_source_race\norigin_version: 1\n---\napproved\n";
    let replacement_bytes =
        b"---\norigin_id: page_stale_source_race\norigin_version: 1\n---\nreplacement\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("race.md"), approved_bytes).unwrap();
    std::fs::write(
        page_root.path().join(".wenlan/orphaned/existing.md"),
        b"baseline",
    )
    .unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let error = crate::post_write::quarantine_stale_page_projection_cas_with_after_pin(
        &db,
        &manifest,
        &rollback,
        page_root.path(),
        || {
            let replacement = page_root.path().join("replacement.tmp");
            std::fs::write(&replacement, replacement_bytes)?;
            std::fs::rename(replacement, page_root.path().join("race.md"))?;
            Ok(())
        },
        |_| Ok(()),
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("repair_target_stale"), "{error}");
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        raw_state
    );
    assert_eq!(
        std::fs::read(page_root.path().join("race.md")).unwrap(),
        replacement_bytes
    );
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/orphaned/existing.md")).unwrap(),
        b"baseline"
    );
    assert!(!page_root
        .path()
        .join(".wenlan/orphaned/page_stale_source_race.md")
        .exists());
}

#[tokio::test]
async fn stale_page_projection_source_replacement_before_unlink_preserves_replacement() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan/orphaned")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(
            page_root.path().join(".wenlan/orphaned"),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
    }
    let raw_state = br#"{"schema_version":2,"pages":{"page_stale_unlink_race":{"file":"race.md","version":1}}}"#;
    let approved_bytes =
        b"---\norigin_id: page_stale_unlink_race\norigin_version: 1\n---\napproved\n";
    let replacement_bytes =
        b"---\norigin_id: page_stale_unlink_race\norigin_version: 1\n---\nreplacement\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("race.md"), approved_bytes).unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let error = crate::post_write::quarantine_stale_page_projection_cas_with_before_source_stage(
        &db,
        &manifest,
        &rollback,
        page_root.path(),
        || {
            let replacement = page_root.path().join("replacement.tmp");
            std::fs::write(&replacement, replacement_bytes)?;
            std::fs::rename(replacement, page_root.path().join("race.md"))?;
            Ok(())
        },
        |_| Ok(()),
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("repair_target_stale"), "{error}");
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        raw_state
    );
    assert_eq!(
        std::fs::read(page_root.path().join("race.md")).unwrap(),
        replacement_bytes
    );
    assert!(!page_root
        .path()
        .join(".wenlan/orphaned/page_stale_unlink_race.md")
        .exists());
    let stage = page_root.path().join(".wenlan").join(
        crate::export::knowledge::projection_unlink_stage_name(manifest.manifest_id()),
    );
    assert!(stage.join("owner.json").is_file());
    assert!(!stage.join("source").exists());
}

#[tokio::test]
async fn stale_page_projection_rollback_failure_retains_marker_for_later_safe_recovery() {
    let fixture = stale_recovery_fixture("page_rollback_failure_recovery", false).await;
    std::fs::write(fixture.pending_path(), b"partial").unwrap();
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            fixture
                .store
                .manifest_dir(fixture.manifest.manifest_id())
                .unwrap()
                .join(fixture.manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();
    let quarantine = fixture.quarantine();

    let error = crate::post_write::quarantine_stale_page_projection_cas(
        &fixture.db,
        &fixture.manifest,
        &rollback,
        fixture.page_root.path(),
        |_| {
            std::fs::write(&quarantine, b"rollback collision").unwrap();
            Err(crate::error::WenlanError::Validation(
                "forced_pre_commit".to_string(),
            ))
        },
    )
    .await
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("stale projection rollback failed"));
    assert_eq!(std::fs::read(fixture.pending_path()).unwrap(), b"partial");
    assert_eq!(std::fs::read(fixture.state()).unwrap(), fixture.post_state);
    assert!(!fixture.source().exists());
    assert_eq!(
        std::fs::read(fixture.quarantine()).unwrap(),
        b"rollback collision"
    );

    std::fs::write(fixture.quarantine(), &fixture.source_bytes).unwrap();
    let receipt = fixture.apply(1_721_000_401).await.unwrap();

    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert!(!fixture.pending_path().exists());
    assert!(fixture.final_path().is_file());
    assert_eq!(
        std::fs::read(fixture.quarantine()).unwrap(),
        fixture.source_bytes
    );
}

#[cfg(unix)]
#[tokio::test]
async fn stale_page_projection_partial_writer_error_retains_marker_and_retry_recovers() {
    use std::os::unix::fs::PermissionsExt as _;

    let fixture = stale_recovery_fixture("page_partial_writer_error", true).await;
    let original_mode = std::fs::metadata(fixture.page_root.path())
        .unwrap()
        .permissions()
        .mode();
    std::fs::set_permissions(
        fixture.page_root.path(),
        std::fs::Permissions::from_mode(0o500),
    )
    .unwrap();

    let result = fixture.apply(1_721_000_451).await;
    std::fs::set_permissions(
        fixture.page_root.path(),
        std::fs::Permissions::from_mode(original_mode),
    )
    .unwrap();
    let error = result.unwrap_err();

    assert!(
        error.to_string().contains("Permission denied"),
        "unexpected partial-writer error: {error}"
    );
    assert!(fixture.pending_path().is_file());
    assert_eq!(
        std::fs::read(fixture.state()).unwrap(),
        fixture.original_state
    );
    assert_eq!(
        std::fs::read(fixture.source()).unwrap(),
        fixture.source_bytes
    );
    assert!(!fixture.quarantine().exists());

    let receipt = fixture.apply(1_721_000_452).await.unwrap();

    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert!(!fixture.pending_path().exists());
    assert!(fixture.final_path().is_file());
    assert!(!fixture.source().exists());
    assert_eq!(
        std::fs::read(fixture.quarantine()).unwrap(),
        fixture.source_bytes
    );
}

#[tokio::test]
async fn stale_page_projection_empty_pending_after_link_restores_and_retries() {
    let (db, _dir) = test_db().await;
    let page_root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let raw_state = b"{\r\n\t\"custom\":{\"preserve\":true},\r\n\t\"pages\":{\"page_stale_recovery\":{\"file\":\"recovery.md\",\"version\":1},\"page_other\":{\"file\":\"other.md\",\"version\":2}},\r\n\t\"schema_version\":2\r\n}\r\n";
    let exact_post_state =
        crate::lint::pages::state::remove_unique_page_member(raw_state, "page_stale_recovery")
            .unwrap();
    let target_bytes =
        b"---\r\norigin_id: page_stale_recovery\r\norigin_version: 1\r\n---\r\n\xffrecovery\r\n";
    std::fs::write(page_root.path().join(".wenlan/state.json"), raw_state).unwrap();
    std::fs::write(page_root.path().join("recovery.md"), target_bytes).unwrap();
    std::fs::write(page_root.path().join("other.md"), b"other bytes").unwrap();
    let repair_root = tempfile::TempDir::new().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::QuarantineStalePageProjection,
    )
    .await;
    let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
    std::fs::write(manifest_dir.join(".apply-receipt.json.pending"), b"").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(page_root.path().join(".wenlan/orphaned"))
            .unwrap();
    }
    #[cfg(not(unix))]
    std::fs::create_dir(page_root.path().join(".wenlan/orphaned")).unwrap();
    let quarantine = page_root
        .path()
        .join(".wenlan/orphaned/page_stale_recovery.md");
    std::fs::hard_link(page_root.path().join("recovery.md"), &quarantine).unwrap();

    let receipt = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();

    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert_eq!(
        std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
        exact_post_state
    );
    assert!(!page_root.path().join("recovery.md").exists());
    assert_eq!(std::fs::read(&quarantine).unwrap(), target_bytes);
    assert_eq!(
        std::fs::read(page_root.path().join("other.md")).unwrap(),
        b"other bytes"
    );
    assert!(!manifest_dir.join(".apply-receipt.json.pending").exists());
    assert!(manifest_dir.join("apply-receipt.json").is_file());
    assert_eq!(
        crate::repair::apply_repair_with_pages(
            &db,
            &store,
            exact_apply_request(&manifest),
            Some(page_root.path()),
            1_721_000_002,
        )
        .await
        .unwrap()
        .receipt_digest(),
        receipt.receipt_digest()
    );
}

#[tokio::test]
async fn stale_page_projection_crash_after_source_stage_restores_exact_original() {
    let fixture = stale_recovery_fixture("page_recover_after_source_stage", true).await;
    std::fs::write(fixture.pending_path(), b"partial receipt").unwrap();
    fixture.stage_after_source_stage();
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            fixture
                .store
                .manifest_dir(fixture.manifest.manifest_id())
                .unwrap()
                .join(fixture.manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let recovered = crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        fixture.page_root.path().to_path_buf(),
        &fixture.db,
        |write| {
            write.recover_stale_page_projection(&rollback, fixture.manifest.manifest_id(), true)
        },
    )
    .unwrap();

    assert_eq!(
        recovered,
        crate::export::knowledge::StalePageProjectionRecoveryState::Original
    );
    assert_eq!(
        std::fs::read(fixture.state()).unwrap(),
        fixture.original_state
    );
    assert_eq!(
        std::fs::read(fixture.source()).unwrap(),
        fixture.source_bytes
    );
    assert!(!fixture.source_stage().exists());
    assert!(fixture.source_stage_owner().is_file());
    assert!(!fixture.quarantine().exists());
    assert_eq!(
        std::fs::read(
            fixture
                .page_root
                .path()
                .join(".wenlan/orphaned/baseline.md")
        )
        .unwrap(),
        b"baseline orphan bytes"
    );
}

#[tokio::test]
async fn stale_page_projection_crash_after_source_relink_recovers_idempotently() {
    let fixture = stale_recovery_fixture("page_recover_after_source_relink", true).await;
    std::fs::write(fixture.pending_path(), b"partial receipt").unwrap();
    fixture.stage_after_source_restore_link();
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            fixture
                .store
                .manifest_dir(fixture.manifest.manifest_id())
                .unwrap()
                .join(fixture.manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    for _ in 0..2 {
        let recovered = crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
            fixture.page_root.path().to_path_buf(),
            &fixture.db,
            |write| {
                write.recover_stale_page_projection(&rollback, fixture.manifest.manifest_id(), true)
            },
        )
        .unwrap();
        assert_eq!(
            recovered,
            crate::export::knowledge::StalePageProjectionRecoveryState::Original
        );
    }
    assert_eq!(
        std::fs::read(fixture.source()).unwrap(),
        fixture.source_bytes
    );
    assert!(!fixture.source_stage().exists());
    assert!(!fixture.quarantine().exists());
    assert!(fixture.source_stage_owner().is_file());
}

#[tokio::test]
async fn stale_page_projection_crash_after_private_quarantine_move_recovers_and_retries() {
    let fixture = stale_recovery_fixture("page_recover_after_quarantine_move", false).await;
    std::fs::write(fixture.pending_path(), b"partial receipt").unwrap();
    fixture.stage_after_quarantine_move();
    assert!(!fixture.quarantine().exists());
    assert_eq!(
        std::fs::read(fixture.staged_quarantine()).unwrap(),
        fixture.source_bytes
    );

    let receipt = fixture.apply(1_721_000_047).await.unwrap();

    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert_eq!(std::fs::read(fixture.state()).unwrap(), fixture.post_state);
    assert!(!fixture.source().exists());
    assert_eq!(
        std::fs::read(fixture.quarantine()).unwrap(),
        fixture.source_bytes
    );
    assert!(!fixture.staged_quarantine().exists());
    assert!(fixture.source_stage_owner().is_file());
    assert!(!fixture.pending_path().exists());
}

#[tokio::test]
async fn stale_page_projection_crash_after_private_stage_prepare_restores_exact_original() {
    let fixture = stale_recovery_fixture("page_recover_after_stage_prepare", true).await;
    std::fs::write(fixture.pending_path(), b"partial receipt").unwrap();
    fixture.ensure_source_stage_dir();
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            fixture
                .store
                .manifest_dir(fixture.manifest.manifest_id())
                .unwrap()
                .join(fixture.manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();

    let recovered = crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        fixture.page_root.path().to_path_buf(),
        &fixture.db,
        |write| {
            write.recover_stale_page_projection(&rollback, fixture.manifest.manifest_id(), true)
        },
    )
    .unwrap();

    assert_eq!(
        recovered,
        crate::export::knowledge::StalePageProjectionRecoveryState::Original
    );
    assert_eq!(
        std::fs::read(fixture.state()).unwrap(),
        fixture.original_state
    );
    assert_eq!(
        std::fs::read(fixture.source()).unwrap(),
        fixture.source_bytes
    );
    assert!(fixture.source_stage_dir().is_dir());
    assert!(fixture.source_stage_owner().is_file());
    assert!(!fixture.quarantine().exists());

    let receipt = fixture.apply(1_721_000_049).await.unwrap();
    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert!(fixture.source_stage_owner().is_file());
    assert!(!fixture.source_stage().exists());
}

#[tokio::test]
async fn stale_page_projection_preexisting_private_stage_is_zero_mutation_stale() {
    let fixture = stale_recovery_fixture("page_preexisting_private_stage", true).await;
    fixture.ensure_source_stage_dir();
    std::fs::write(fixture.source_stage_owner(), b"foreign owner").unwrap();
    std::fs::write(fixture.source_stage(), b"foreign stage").unwrap();

    let error = fixture.apply(1_721_000_051).await.unwrap_err();

    assert!(error.to_string().contains("repair_target_stale"), "{error}");
    assert_eq!(
        std::fs::read(fixture.state()).unwrap(),
        fixture.original_state
    );
    assert_eq!(
        std::fs::read(fixture.source()).unwrap(),
        fixture.source_bytes
    );
    assert_eq!(
        std::fs::read(fixture.source_stage()).unwrap(),
        b"foreign stage"
    );
    assert!(!fixture.quarantine().exists());
}

#[tokio::test]
async fn stale_page_projection_partial_pending_recovers_after_unlink_and_post() {
    for (page_id, stage_post, preexisting_orphan) in [
        ("page_recover_after_unlink", false, false),
        ("page_recover_post", true, true),
    ] {
        let fixture = stale_recovery_fixture(page_id, preexisting_orphan).await;
        std::fs::write(fixture.pending_path(), b"partial receipt").unwrap();
        if stage_post {
            fixture.stage_post();
        } else {
            fixture.stage_after_unlink();
        }

        let receipt = fixture.apply(1_721_000_101).await.unwrap();

        assert_eq!(std::fs::read(fixture.state()).unwrap(), fixture.post_state);
        assert!(!fixture.source().exists());
        assert_eq!(
            std::fs::read(fixture.quarantine()).unwrap(),
            fixture.source_bytes
        );
        assert!(!fixture.pending_path().exists());
        assert!(fixture.final_path().is_file());
        assert_eq!(
            fixture.apply(1_721_000_102).await.unwrap().receipt_digest(),
            receipt.receipt_digest()
        );
        if preexisting_orphan {
            assert_eq!(
                std::fs::read(
                    fixture
                        .page_root
                        .path()
                        .join(".wenlan/orphaned/baseline.md")
                )
                .unwrap(),
                b"baseline orphan bytes"
            );
        }
    }
}

#[tokio::test]
async fn stale_page_projection_valid_pending_publishes_post_and_recovers_partial() {
    let post = stale_recovery_fixture("page_valid_pending_post", false).await;
    let receipt = post.apply(1_721_000_201).await.unwrap();
    std::fs::rename(post.final_path(), post.pending_path()).unwrap();

    let published = post.apply(1_721_000_202).await.unwrap();

    assert_eq!(published, receipt);
    assert!(post.final_path().is_file());
    assert!(!post.pending_path().exists());
    assert_eq!(std::fs::read(post.state()).unwrap(), post.post_state);
    assert!(!post.source().exists());
    assert_eq!(std::fs::read(post.quarantine()).unwrap(), post.source_bytes);

    let partial = stale_recovery_fixture("page_valid_pending_partial", false).await;
    let partial_receipt = partial.apply(1_721_000_203).await.unwrap();
    let pending_bytes = std::fs::read(partial.final_path()).unwrap();
    std::fs::remove_file(partial.final_path()).unwrap();
    std::fs::write(partial.state(), &partial.original_state).unwrap();
    std::fs::write(partial.source(), &partial.source_bytes).unwrap();
    std::fs::remove_file(partial.quarantine()).unwrap();
    std::fs::remove_dir(partial.page_root.path().join(".wenlan/orphaned")).unwrap();
    partial.stage_after_unlink();
    std::fs::write(partial.pending_path(), pending_bytes).unwrap();

    let retried = partial.apply(1_721_000_204).await.unwrap();

    assert_ne!(retried.receipt_digest(), partial_receipt.receipt_digest());
    assert!(partial.final_path().is_file());
    assert!(!partial.pending_path().exists());
    assert_eq!(
        std::fs::read(partial.quarantine()).unwrap(),
        partial.source_bytes
    );
}

#[tokio::test]
async fn stale_page_projection_valid_pending_with_wrong_post_digest_stays_pending() {
    use wenlan_types::repair::{RepairApplyReceipt, RepairApplyReceiptDraft, RepairDigest};

    let fixture = stale_recovery_fixture("page_wrong_pending_post", false).await;
    let receipt = fixture.apply(1_721_000_251).await.unwrap();
    std::fs::remove_file(fixture.final_path()).unwrap();
    let wrong_after =
        RepairDigest::parse("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            .unwrap();
    let draft = RepairApplyReceiptDraft::try_new(
        fixture.manifest.manifest_id().to_string(),
        fixture.manifest.manifest_digest().clone(),
        1_721_000_251,
        receipt.before_target_receipt().clone(),
        wrong_after,
        receipt.non_target_before().clone(),
        receipt.non_target_after().clone(),
        receipt.post_apply_db_digest().unwrap().clone(),
        receipt.actual_effects().clone(),
        receipt.writer(),
    )
    .unwrap();
    let digest = crate::repair::repair_digest(&draft.canonical_bytes().unwrap());
    let wrong_receipt = RepairApplyReceipt::from_draft(draft, digest);
    let pending_bytes = serde_json::to_vec_pretty(&wrong_receipt).unwrap();
    std::fs::write(fixture.pending_path(), &pending_bytes).unwrap();
    let before_state = std::fs::read(fixture.state()).unwrap();
    let before_quarantine = std::fs::read(fixture.quarantine()).unwrap();

    let error = fixture.apply(1_721_000_252).await.unwrap_err();

    assert_eq!(
        error.to_string(),
        "Conflict: repair_apply_recovery_required"
    );
    assert_eq!(
        std::fs::read(fixture.pending_path()).unwrap(),
        pending_bytes
    );
    assert!(!fixture.final_path().exists());
    assert_eq!(std::fs::read(fixture.state()).unwrap(), before_state);
    assert!(!fixture.source().exists());
    assert_eq!(
        std::fs::read(fixture.quarantine()).unwrap(),
        before_quarantine
    );
}

#[cfg(unix)]
#[tokio::test]
async fn stale_page_projection_unknown_recovery_states_preserve_every_target_path() {
    use std::os::unix::fs::symlink;

    for case in [
        "unrelated_state",
        "source",
        "quarantine",
        "source_symlink",
        "unexpected_orphan",
        "unexpected_preexisting_orphan",
    ] {
        let page_id = format!("page_unknown_{case}");
        let fixture =
            stale_recovery_fixture(&page_id, case == "unexpected_preexisting_orphan").await;
        std::fs::write(fixture.pending_path(), b"partial").unwrap();
        match case {
            "unrelated_state" => {
                let mut changed = fixture.original_state.clone();
                changed.extend_from_slice(b" ");
                std::fs::write(fixture.state(), changed).unwrap();
            }
            "source" => std::fs::write(fixture.source(), b"changed source").unwrap(),
            "quarantine" => {
                fixture.ensure_orphaned();
                std::fs::write(fixture.quarantine(), b"changed quarantine").unwrap();
            }
            "source_symlink" => {
                let outside = fixture.page_root.path().join("outside.md");
                std::fs::write(&outside, b"outside").unwrap();
                std::fs::remove_file(fixture.source()).unwrap();
                symlink(&outside, fixture.source()).unwrap();
            }
            "unexpected_orphan" | "unexpected_preexisting_orphan" => {
                fixture.ensure_orphaned();
                std::fs::write(
                    fixture
                        .page_root
                        .path()
                        .join(".wenlan/orphaned/unexpected.md"),
                    b"unexpected",
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        let before_state = std::fs::read(fixture.state()).unwrap();
        let before_source = std::fs::symlink_metadata(fixture.source()).unwrap();
        let before_source_bytes = if before_source.file_type().is_symlink() {
            None
        } else {
            Some(std::fs::read(fixture.source()).unwrap())
        };
        let before_quarantine = std::fs::read(fixture.quarantine()).ok();
        let before_unexpected = std::fs::read(
            fixture
                .page_root
                .path()
                .join(".wenlan/orphaned/unexpected.md"),
        )
        .ok();

        let error = fixture.apply(1_721_000_301).await.unwrap_err();

        assert_eq!(
            error.to_string(),
            "Conflict: repair_apply_recovery_required",
            "{case}"
        );
        assert_eq!(
            std::fs::read(fixture.state()).unwrap(),
            before_state,
            "{case}"
        );
        assert_eq!(
            std::fs::symlink_metadata(fixture.source())
                .unwrap()
                .file_type()
                .is_symlink(),
            before_source.file_type().is_symlink(),
            "{case}"
        );
        if let Some(bytes) = before_source_bytes {
            assert_eq!(std::fs::read(fixture.source()).unwrap(), bytes, "{case}");
        }
        assert_eq!(
            std::fs::read(fixture.quarantine()).ok(),
            before_quarantine,
            "{case}"
        );
        assert_eq!(
            std::fs::read(
                fixture
                    .page_root
                    .path()
                    .join(".wenlan/orphaned/unexpected.md")
            )
            .ok(),
            before_unexpected,
            "{case}"
        );
        assert_eq!(std::fs::read(fixture.pending_path()).unwrap(), b"partial");
        assert!(!fixture.final_path().exists());
    }
}

#[tokio::test]
async fn only_empty_machine_owned_unconfirmed_source_page_is_exact_archive_target() {
    let (db, _dir) = test_db().await;
    let now = "2026-07-17T00:00:00Z";
    for (id, content, review_status, user_edited) in [
        ("source_ready", " \n", "unconfirmed", 0_i64),
        ("source_with_content", "keep me", "unconfirmed", 0_i64),
        ("source_confirmed", "", "confirmed", 0_i64),
        ("source_user_edited", "", "unconfirmed", 1_i64),
    ] {
        db.insert_page_with_kind(
            id,
            id,
            None,
            content,
            None,
            Some("work"),
            &[],
            now,
            "source",
            review_status,
            Some("work"),
            None,
        )
        .await
        .unwrap();
        if user_edited != 0 {
            db.conn
                .lock()
                .await
                .execute(
                    "UPDATE pages SET user_edited=1 WHERE id=?1",
                    libsql::params![id],
                )
                .await
                .unwrap();
        }
    }
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();

    let resolutions = resolve_current(
        &snapshot,
        &RepairLintScope::registered("work".to_string()).unwrap(),
        None,
    )
    .await
    .unwrap();
    let archives = resolutions
        .iter()
        .filter_map(|resolution| match resolution {
            DeterministicResolution::Exact(exact)
                if exact.writer == RepairWriter::ArchiveEmptySourcePage =>
            {
                Some(exact)
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(archives.len(), 1, "resolutions: {resolutions:#?}");
    assert!(matches!(
        archives[0].target,
        RepairTarget::Page { ref page_id, .. } if page_id == "source_ready"
    ));
    assert_eq!(archives[0].expected_version, Some(1));
    assert!(matches!(
        archives[0].mutation,
        RepairMutation::ArchiveEmptySourcePage { ref before_status, ref after_status }
            if before_status == "active" && after_status == "archived"
    ));
    assert_eq!(archives[0].rollback.table, "pages");
    assert_eq!(archives[0].rollback.source_id, "source_ready");
    assert_eq!(archives[0].rollback.rows.len(), 1);
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn source_page_archive_accepts_rust_whitespace_persisted_below_ingress() {
    let (db, _dir) = test_db().await;
    let now = "2026-07-17T00:00:00Z";
    db.insert_page_with_kind(
        "source_whitespace",
        "source_whitespace",
        None,
        "",
        None,
        Some("work"),
        &[],
        now,
        "source",
        "unconfirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    let persisted_whitespace = "\t\n\u{2003}";
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET content=?1 WHERE id='source_whitespace'",
            libsql::params![persisted_whitespace],
        )
        .await
        .unwrap();
    assert_eq!(
        db.get_page("source_whitespace")
            .await
            .unwrap()
            .unwrap()
            .content,
        persisted_whitespace
    );

    let page_root = tempfile::tempdir().unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::ArchiveEmptySourcePage,
    )
    .await;
    assert!(matches!(
        manifest.target(),
        RepairTarget::Page { page_id, .. } if page_id == "source_whitespace"
    ));

    let receipt = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();

    assert_eq!(receipt.writer(), RepairWriter::ArchiveEmptySourcePage);
    assert_eq!(
        db.get_page("source_whitespace")
            .await
            .unwrap()
            .unwrap()
            .status,
        "archived"
    );
}

#[tokio::test]
async fn source_page_archive_cas_changes_only_status_and_clears_lint_findings() {
    let (db, _dir) = test_db().await;
    let now = "2026-07-17T00:00:00Z";
    for id in ["source_target", "source_canary"] {
        db.insert_page_with_kind(
            id,
            id,
            None,
            if id == "source_target" {
                " \n"
            } else {
                "canary"
            },
            None,
            Some("work"),
            &[],
            now,
            if id == "source_target" {
                "source"
            } else {
                "distilled"
            },
            if id == "source_target" {
                "unconfirmed"
            } else {
                "confirmed"
            },
            Some("work"),
            None,
        )
        .await
        .unwrap();
    }
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET status='archived' WHERE id='source_canary'",
            (),
        )
        .await
        .unwrap();
    let page_root = tempfile::tempdir().unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::ArchiveEmptySourcePage,
    )
    .await;
    assert_eq!(manifest.expected_state().version(), Some(1));
    assert_eq!(
        manifest.post_assertions().allowed_non_target_check_deltas(),
        ["pages.projection.identity"]
    );
    let rollback: crate::repair::StoredRollbackArtifact = serde_json::from_slice(
        &std::fs::read(
            repair_root
                .path()
                .join(manifest.manifest_id())
                .join(manifest.rollback().relative_path()),
        )
        .unwrap(),
    )
    .unwrap();
    let page_column_count = {
        let conn = db.conn.lock().await;
        let mut rows = conn.query("PRAGMA table_info(pages)", ()).await.unwrap();
        let mut count = 0;
        while rows.next().await.unwrap().is_some() {
            count += 1;
        }
        count
    };
    assert_eq!(rollback.table, "pages");
    assert_eq!(rollback.source_id, "source_target");
    assert_eq!(rollback.columns.len(), page_column_count);
    assert_eq!(rollback.rows.len(), 1);
    assert_eq!(rollback.rows[0].len(), page_column_count);

    let before_target = serde_json::to_value(db.get_page("source_target").await.unwrap()).unwrap();
    let before_canary = serde_json::to_value(db.get_page("source_canary").await.unwrap()).unwrap();
    let direct_apply = || async {
        crate::post_write::apply_deterministic_repair_cas(&db, &manifest, &[], |_| Ok(())).await
    };

    db.conn
        .lock()
        .await
        .execute("UPDATE pages SET version=2 WHERE id='source_target'", ())
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute("UPDATE pages SET version=1 WHERE id='source_target'", ())
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET workspace='other' WHERE id='source_target'",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET workspace='work' WHERE id='source_target'",
            (),
        )
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET source_memory_ids='[\"restored-json\"]'
             WHERE id='source_target'",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET source_memory_ids='[]' WHERE id='source_target'",
            (),
        )
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "INSERT INTO page_sources(page_id,memory_source_id,linked_at,link_reason)
             VALUES ('source_target','restored-join',1,'stale proof')",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute("DELETE FROM page_sources WHERE page_id='source_target'", ())
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "INSERT INTO page_evidence(page_id,source_kind,locator,linked_at,link_reason)
             VALUES ('source_target','authored',NULL,1,'stale proof')",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "DELETE FROM page_evidence WHERE page_id='source_target'",
            (),
        )
        .await
        .unwrap();

    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET status='archived' WHERE id='source_target'",
            (),
        )
        .await
        .unwrap();
    assert!(direct_apply()
        .await
        .unwrap_err()
        .to_string()
        .contains("repair_target_stale"));
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET status='active' WHERE id='source_target'",
            (),
        )
        .await
        .unwrap();

    let receipt = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_apply_request(&manifest),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();
    assert_eq!(receipt.writer(), RepairWriter::ArchiveEmptySourcePage);
    assert_eq!(receipt.actual_effects(), manifest.allowed_effects());
    let mut after_target =
        serde_json::to_value(db.get_page("source_target").await.unwrap()).unwrap();
    assert_eq!(after_target["status"], "archived");
    after_target["status"] = before_target["status"].clone();
    assert_eq!(after_target, before_target, "only status may change");
    assert_eq!(
        serde_json::to_value(db.get_page("source_canary").await.unwrap()).unwrap(),
        before_canary
    );
    assert!(!page_root.path().join("source_target.md").exists());

    let after_lint = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    for check_id in [SOURCE_PAGE_INTEGRITY, IDENTITY_ID] {
        let check = after_lint
            .checks()
            .iter()
            .find(|check| check.check_id() == check_id)
            .unwrap();
        assert_eq!(check.outcome(), wenlan_types::lint::LintOutcome::Pass);
    }
}
