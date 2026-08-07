// SPDX-License-Identifier: Apache-2.0
//! P2 typed-evidence integration tests.
use std::{collections::BTreeMap, sync::Arc};
use wenlan_core::db::MemoryDB;
use wenlan_core::sources::RawDocument;
use wenlan_core::{EventEmitter, NoopEmitter};
use wenlan_types::requests::CreateConceptRequest;

async fn make_db() -> (Arc<MemoryDB>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("test.db");
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let db = MemoryDB::new(&db_path, emitter)
        .await
        .expect("MemoryDB::new");
    (Arc::new(db), dir)
}

/// Seed a memory via the canonical upsert path (replaces the plan's
/// non-existent `store_memory_for_test`).
async fn seed_memory(db: &MemoryDB, id: &str, content: &str) {
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: id.to_string(),
        title: id.to_string(),
        summary: None,
        content: content.to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        space: Some("technology".to_string()),
        source_agent: Some("test-agent".to_string()),
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };
    db.upsert_documents(vec![doc]).await.expect("seed memory");
}

struct PageFixture<'a> {
    title: &'a str,
    summary: Option<&'a str>,
    content: &'a str,
    space: Option<&'a str>,
    source_ids: &'a [&'a str],
    creation_kind: &'a str,
    workspace: Option<&'a str>,
}

async fn create_page_fixture(db: &MemoryDB, fixture: PageFixture<'_>) -> String {
    let PageFixture {
        title,
        summary,
        content,
        space,
        source_ids,
        creation_kind,
        workspace,
    } = fixture;
    if !source_ids.is_empty() {
        db.upsert_documents(
            source_ids
                .iter()
                .map(|source_id| RawDocument {
                    source: "memory".to_string(),
                    source_id: (*source_id).to_string(),
                    title: (*source_id).to_string(),
                    content: content.to_string(),
                    last_modified: chrono::Utc::now().timestamp(),
                    memory_type: Some("fact".to_string()),
                    space: space.map(str::to_string),
                    source_agent: Some("test-agent".to_string()),
                    confirmed: Some(true),
                    ..Default::default()
                })
                .collect(),
        )
        .await
        .expect("seed page sources");
    }
    let req = CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: summary.map(str::to_string),
        entity_id: None,
        space: (space.map(str::to_string)).into(),
        source_memory_ids: source_ids.iter().map(|id| (*id).to_string()).collect(),
        creation_kind: Some(creation_kind.to_string()),
        workspace: workspace.map(str::to_string),
    };
    let result = if creation_kind == "distilled" {
        wenlan_core::post_write::create_page_with_tuning(
            db,
            req,
            "test",
            None,
            source_ids.len().max(1),
            1.1,
        )
        .await
        .expect("create distilled page fixture")
    } else {
        wenlan_core::post_write::create_page(db, req, "test", None)
            .await
            .expect("create page fixture")
    };
    db.set_page_review_status(&result.id, "confirmed")
        .await
        .expect("confirm page fixture");
    result.id
}

/// Seed a folder-document memory: `source="memory"`, `source_agent="folder"`,
/// and the `{source_id}::{provenance}` id shape that
/// `sources::directory::document_source_id` stamps (directory.rs:167,372). The
/// resolver must map this shape to `source_kind='external_file'`, not the
/// plain-capture default `'memory'`.
async fn seed_folder_doc(db: &MemoryDB, source_id: &str, content: &str) {
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: content.to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        space: Some("technology".to_string()),
        source_agent: Some("folder".to_string()),
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };
    db.upsert_documents(vec![doc])
        .await
        .expect("seed folder doc");
}

/// Spec §5.1 (True source kinds): PageWrite must create typed evidence for
/// every source kind it carries into the DB write. This goes through the
/// public PageWrite create path, not the raw `insert_page` helper; hardcoding
/// every emitted row to `'memory'` leaves the folder doc and URL rows wrong.
#[tokio::test]
async fn pagewrite_create_records_resolved_source_kinds_for_file_and_url_sources() {
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_a", "Rust workspaces share Cargo configuration.").await;
    seed_folder_doc(
        &db,
        "folder-notes::rust/workspace.md",
        "Folder documents describe Rust workspace layouts.",
    )
    .await;
    seed_memory(
        &db,
        "https://example.com/rust-workspaces",
        "Web docs explain Rust workspace member crates.",
    )
    .await;
    let req = CreateConceptRequest { title: "Rust Workspaces".to_string(),
    content: "Rust workspaces share Cargo configuration. Folder documents describe Rust workspace layouts. Web docs explain Rust workspace member crates.".to_string(),
    summary: Some("Rust workspace sources".to_string()),
    entity_id: None, space: (Some("technology".to_string())).into(), source_memory_ids: vec![
        "mem_a".to_string(),
        "folder-notes::rust/workspace.md".to_string(),
        "https://example.com/rust-workspaces".to_string(),
    ],
    creation_kind: Some("distilled".to_string()),
    workspace: None, };

    let result = wenlan_core::post_write::create_page(&db, req, "test", None)
        .await
        .unwrap();

    let ev = db.get_page_evidence(&result.id).await.unwrap();
    let actual: BTreeMap<String, String> = ev
        .iter()
        .filter_map(|e| Some((e.locator.clone()?, e.source_kind.clone())))
        .collect();
    let expected = BTreeMap::from([
        ("mem_a".to_string(), "memory".to_string()),
        (
            "folder-notes::rust/workspace.md".to_string(),
            "external_file".to_string(),
        ),
        (
            "https://example.com/rust-workspaces".to_string(),
            "external_url".to_string(),
        ),
    ]);
    assert_eq!(
        actual, expected,
        "PageWrite create must resolve page_evidence.source_kind per source"
    );
    assert_eq!(
        ev.len(),
        expected.len(),
        "PageWrite create should emit one evidence row per source"
    );
    let kind_of = |loc: &str| -> Option<String> {
        ev.iter()
            .find(|e| e.locator.as_deref() == Some(loc))
            .map(|e| e.source_kind.clone())
    };
    assert_eq!(
        kind_of("mem_a").as_deref(),
        Some("memory"),
        "a plain agent capture must record source_kind='memory'"
    );
    assert_eq!(
        kind_of("folder-notes::rust/workspace.md").as_deref(),
        Some("external_file"),
        "a PageWrite-created page whose source is a folder doc must record source_kind='external_file', not 'memory'"
    );
    assert_eq!(
        kind_of("https://example.com/rust-workspaces").as_deref(),
        Some("external_url"),
        "a PageWrite-created page whose source is a URL must record source_kind='external_url', not 'memory'"
    );
}

#[tokio::test]
async fn page_evidence_backfill_matches_legacy_page_sources() {
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_a", "alpha content about rust").await;
    seed_memory(&db, "mem_b", "beta content about rust").await;
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "Rust",
            summary: Some("rust topic"),
            content: "body",
            space: None,
            source_ids: &["mem_a", "mem_b"],
            creation_kind: "distilled",
            workspace: None,
        },
    )
    .await;
    let ev = db.get_page_evidence(&page_id).await.unwrap();
    let mut locs: Vec<String> = ev
        .iter()
        .filter(|e| e.source_kind == "memory")
        .map(|e| e.locator.clone().unwrap())
        .collect();
    locs.sort();
    assert_eq!(locs, vec!["mem_a".to_string(), "mem_b".to_string()]);
    assert!(ev.iter().all(|e| e.source_kind == "memory"));
}

/// Sorted memory-locator set from page_evidence vs sorted memory_source_id
/// set from page_sources for a page. They must be equal (dual-write contract).
async fn evidence_vs_sources(db: &MemoryDB, page_id: &str) -> (Vec<String>, Vec<String>) {
    let mut ev: Vec<String> = db
        .get_page_evidence(page_id)
        .await
        .unwrap()
        .into_iter()
        .filter(|e| e.source_kind == "memory")
        .filter_map(|e| e.locator)
        .collect();
    let mut ps: Vec<String> = db
        .get_page_sources(page_id)
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.memory_source_id)
        .collect();
    ev.sort();
    ps.sort();
    (ev, ps)
}

#[tokio::test]
async fn dual_write_keeps_page_evidence_consistent_with_page_sources() {
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_a", "alpha").await;
    seed_memory(&db, "mem_b", "beta").await;
    seed_memory(&db, "mem_c", "gamma").await;
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "T",
            summary: Some("s"),
            content: "body",
            space: None,
            source_ids: &["mem_a", "mem_b"],
            creation_kind: "distilled",
            workspace: None,
        },
    )
    .await;
    let (ev, ps) = evidence_vs_sources(&db, &page_id).await;
    assert_eq!(ev, ps, "insert_page diverged");
    db.update_page_content(&page_id, "body2", &["mem_a", "mem_c"], "manual_edit")
        .await
        .unwrap();
    let (ev, ps) = evidence_vs_sources(&db, &page_id).await;
    assert_eq!(ev, ps, "update_page_content diverged");
    db.link_page_source(&page_id, "mem_b", "page_growth")
        .await
        .unwrap();
    let (ev, ps) = evidence_vs_sources(&db, &page_id).await;
    assert_eq!(ev, ps, "link_page_source diverged");
}

// `update_prunes_memory_evidence_but_preserves_external` relocated to
// `crates/wenlan-core/src/db/main_tests.rs` (G6 Stage 2 PR 2b): its one
// manual `external_url` evidence attach called `link_page_evidence`, which
// closed to `#[cfg(test)]` under the Q2 ruling (the authored/typed-manual
// entry point had zero live rows). `#[cfg(test)]` does not cross the
// `tests/` integration-binary boundary, so the test moved into the crate's
// own unit-test suite where it compiles. Assertions unchanged.

#[tokio::test]
async fn distilled_page_defaults_creation_kind_distilled() {
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_a", "alpha").await;
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "T",
            summary: Some("s"),
            content: "body",
            space: None,
            source_ids: &["mem_a"],
            creation_kind: "distilled",
            workspace: None,
        },
    )
    .await;
    let p = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(p.creation_kind, "distilled");
}

/// Guards the `dist` read index in `find_matching_page` against creation_kind
/// column-shift drift. After adding creation_kind at index 16, the vector
/// distance moved to index 17. If the read regresses to index 16, it parses
/// the creation_kind TEXT column as f64 -> unwrap_or(1.0) -> uniform distance
/// 1.0 -> similarity 0.0 < threshold -> `find_matching_page` returns None.
/// With the correct index 17 it reads the real distance and returns the near
/// page. The None-vs-Some(near) boundary is binary, so this is non-flaky.
#[tokio::test]
async fn find_matching_page_reads_real_distance_not_creation_kind() {
    let (db, _d) = make_db().await;
    let rust_page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "Rust",
            summary: Some("rust async tokio runtime"),
            content: "Rust ownership and the borrow checker.",
            space: None,
            source_ids: &[],
            creation_kind: "research",
            workspace: None,
        },
    )
    .await;
    create_page_fixture(
        &db,
        PageFixture {
            title: "Python",
            summary: Some("python decorators metaclasses"),
            content: "Python data model and descriptors.",
            space: None,
            source_ids: &[],
            creation_kind: "research",
            workspace: None,
        },
    )
    .await;

    // Query embedding near the Rust page (title + summary is what insert_page embeds).
    let q = db
        .generate_embeddings(&["Rust rust async tokio runtime ownership".to_string()])
        .unwrap()
        .remove(0);

    // entity_id=None forces the embedding path (skips the entity-first branch).
    // Threshold 0.5: the near page clears it; under a misread (uniform distance
    // 1.0 -> similarity 0.0) NO page clears it, so the result would be None.
    let matched = db.find_matching_page(None, &q, 0.5).await.unwrap();
    let page = matched.expect("near page must clear the similarity threshold");
    assert_eq!(
        page.id, rust_page_id,
        "must match the embedding-nearest page, proving real distance was read"
    );
}

#[tokio::test]
async fn create_page_fixture_forces_review_status_confirmed() {
    // Pages are born "unconfirmed" (see post_write::create_page_borns_distilled_unconfirmed);
    // this guards the fixture's explicit set_page_review_status("confirmed") override,
    // which the other provenance tests rely on.
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_a", "alpha").await;
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "T",
            summary: Some("s"),
            content: "body",
            space: None,
            source_ids: &["mem_a"],
            creation_kind: "distilled",
            workspace: None,
        },
    )
    .await;
    let p = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(p.review_status, "confirmed");
}

#[tokio::test]
async fn distilled_zero_source_page_rejected() {
    let (db, dir) = make_db().await;
    let req = CreateConceptRequest {
        title: "T".into(),
        content: "body".into(),
        summary: Some("s".into()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![],
        creation_kind: Some("distilled".into()),
        workspace: None,
    };
    let r = wenlan_core::post_write::create_page(&db, req, "test", Some(dir.path())).await;
    assert!(matches!(
        r,
        Err(wenlan_core::error::WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn authored_zero_source_page_accepted_unconfirmed() {
    let (db, dir) = make_db().await;
    let req = CreateConceptRequest {
        title: "Authored".into(),
        content: "hand written body".into(),
        summary: Some("sum".into()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![],
        creation_kind: Some("authored".into()),
        workspace: None,
    };
    let wr = wenlan_core::post_write::create_page(&db, req, "test", Some(dir.path()))
        .await
        .unwrap();
    let p = db.get_page(&wr.id).await.unwrap().unwrap();
    assert_eq!(p.creation_kind, "authored");
    assert_eq!(p.review_status, "unconfirmed");
    assert!(
        db.get_page_evidence(&wr.id).await.unwrap().is_empty(),
        "authored page has zero evidence"
    );
}

#[tokio::test]
async fn garbage_creation_kind_rejected() {
    let (db, dir) = make_db().await;
    let req = CreateConceptRequest {
        title: "T".into(),
        content: "body".into(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![],
        creation_kind: Some("garbage".into()),
        workspace: None,
    };
    let r = wenlan_core::post_write::create_page(&db, req, "test", Some(dir.path())).await;
    assert!(
        matches!(r, Err(wenlan_core::error::WenlanError::Validation(_))),
        "bad kind must be a 422-class Validation error, not a DB CHECK 500"
    );
}

#[tokio::test]
async fn scoped_matcher_excludes_user_edited_when_disallowed() {
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_a", "rust async runtime tokio").await;
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "Rust",
            summary: Some("rust async tokio"),
            content: "body",
            space: None,
            source_ids: &["mem_a"],
            creation_kind: "distilled",
            workspace: Some("work"),
        },
    )
    .await;
    // Mark user_edited via the manual-edit path (sets user_edited=1, demotes review_status to 'unconfirmed').
    db.update_page_content(&page_id, "edited body", &["mem_a"], "manual_edit")
        .await
        .unwrap();
    // Restore review_status to 'confirmed' so the control below tests user_edited filtering,
    // not review_status filtering (scoped_matcher only surfaces confirmed pages).
    db.set_page_review_status(&page_id, "confirmed")
        .await
        .unwrap();
    let emb = db
        .generate_embeddings(&["rust async tokio".to_string()])
        .unwrap()
        .remove(0);

    // CONTROL: with allow_user_edited=true, the page IS returned (proves it matches by embedding + workspace + confirmed).
    let allowed = db
        .find_matching_page_scoped(None, &emb, 0.5, Some("work"), true)
        .await
        .unwrap();
    assert_eq!(allowed.as_ref().map(|p| p.id.as_str()), Some(page_id.as_str()),
        "control: a confirmed in-workspace user_edited page must match when user_edited is allowed (else threshold/embedding is the problem, fix the test seed)");

    // TEST: with allow_user_edited=false, the SAME page is REFUSED.
    let refused = db
        .find_matching_page_scoped(None, &emb, 0.5, Some("work"), false)
        .await
        .unwrap();
    assert!(
        refused.map(|p| p.id != page_id).unwrap_or(true),
        "must not return a user_edited page when disallowed"
    );
}

#[tokio::test]
async fn source_less_page_embeds_title_plus_content() {
    let (db, _d) = make_db().await;
    // Source-less confirmed authored page: OFF-TOPIC title ("Cooking Recipes"),
    // Rust-specific body, NO summary.
    // Pre-Task-11 (title-only embed): "Cooking Recipes" embeds far from the Rust
    // query — cosine < 0.5 → find_matching_page_scoped returns None.
    // Post-Task-11 (title+content embed): the Rust-dense body pulls the vector
    // close to the query — cosine > 0.5 → returns the page.
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "Cooking Recipes",
            summary: None,
            content: "Detailed notes about Rust ownership, borrowing, and the lifetime system used in async code.",
            space: None,
            source_ids: &[],
            creation_kind: "authored",
            workspace: None,
        },
    )
    .await;
    let q = db
        .generate_embeddings(&["Rust ownership borrowing lifetimes".to_string()])
        .unwrap()
        .remove(0);
    let m = db
        .find_matching_page_scoped(None, &q, 0.5, None, true)
        .await
        .unwrap();
    assert_eq!(m.as_ref().map(|p| p.id.as_str()), Some(page_id.as_str()),
        "source-less page must embed its content so a content-related query matches above 0.5 (title alone would not)");
}

#[tokio::test]
async fn scoped_matcher_never_crosses_workspace() {
    let (db, _d) = make_db().await;
    seed_memory(&db, "mem_w", "kubernetes deployment rollout strategy").await;
    // A confirmed page in workspace "personal".
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "K8s",
            summary: Some("kubernetes deployment rollout strategy"),
            content: "Kubernetes rollout strategies including blue-green and canary deployments.",
            space: None,
            source_ids: &["mem_w"],
            creation_kind: "distilled",
            workspace: Some("personal"),
        },
    )
    .await;
    let q = db
        .generate_embeddings(&["kubernetes rollout strategy".to_string()])
        .unwrap()
        .remove(0);

    // Scoped to "work": the high-similarity "personal" page must NOT leak.
    let leaked = db
        .find_matching_page_scoped(None, &q, 0.5, Some("work"), true)
        .await
        .unwrap();
    assert!(
        leaked.is_none(),
        "SECURITY: scoped matcher leaked a cross-workspace page"
    );

    // CONTROL — scoped to "personal": it SHOULD match (proves the None above is a
    // workspace exclusion, not a threshold miss).
    let same = db
        .find_matching_page_scoped(None, &q, 0.5, Some("personal"), true)
        .await
        .unwrap();
    assert_eq!(
        same.as_ref().map(|p| p.id.as_str()),
        Some(page_id.as_str()),
        "control: the page must match within its own workspace"
    );
}

#[tokio::test]
async fn fs_edit_demotes_review_status_but_agent_refresh_does_not() {
    let (db, _d) = make_db().await;
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "T",
            summary: Some("s"),
            content: "body",
            space: None,
            source_ids: &[],
            creation_kind: "authored",
            workspace: None,
        },
    )
    .await;
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().review_status,
        "confirmed"
    );
    // fs_edit (markdown curation) demotes — same trust rule as manual_edit.
    db.update_page_content(&page_id, "edited on disk", &[], "fs_edit")
        .await
        .unwrap();
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().review_status,
        "unconfirmed",
        "fs_edit must demote"
    );
    // agent_refresh is a faithful re-synth — must NOT demote.
    db.set_page_review_status(&page_id, "confirmed")
        .await
        .unwrap();
    db.update_page_content(&page_id, "refreshed body", &[], "agent_refresh")
        .await
        .unwrap();
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().review_status,
        "confirmed",
        "agent_refresh must NOT demote"
    );
}
