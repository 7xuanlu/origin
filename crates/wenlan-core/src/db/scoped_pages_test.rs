// SPDX-License-Identifier: Apache-2.0

use super::tests::test_db;
use crate::pages::Page;
use crate::read_scope::ReadScope;
use crate::sources::RawDocument;
use std::collections::{HashMap, HashSet};

fn memory_doc(source_id: &str, space: &str) -> RawDocument {
    RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: "deliberately unrelated candidate text".to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        metadata: HashMap::new(),
        memory_type: Some("fact".to_string()),
        space: Some(space.to_string()),
        source_agent: None,
        confidence: Some(0.9),
        confirmed: Some(true),
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    }
}

fn page(id: &str, workspace: Option<&str>) -> Page {
    Page {
        id: id.to_string(),
        title: id.to_string(),
        summary: None,
        content: String::new(),
        entity_id: None,
        space: None,
        source_memory_ids: vec!["work-memory".to_string()],
        version: 1,
        status: "active".to_string(),
        created_at: String::new(),
        last_compiled: String::new(),
        last_modified: String::new(),
        sources_updated_count: 0,
        stale_reason: None,
        pending_rebuild: None,
        user_edited: false,
        relevance_score: 1.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: None,
        changelog: None,
        creation_kind: "distilled".to_string(),
        review_status: "confirmed".to_string(),
        workspace: workspace.map(str::to_string),
        citations: Vec::new(),
        kind: "concept".to_string(),
    }
}

#[tokio::test]
async fn search_memory_scopes_before_vector_limit() {
    let (db, _tmp) = test_db().await;
    db.upsert_documents(vec![memory_doc("work-memory", "work")])
        .await
        .unwrap();
    for index in 0..8 {
        db.upsert_documents(vec![memory_doc(&format!("personal-{index}"), "personal")])
            .await
            .unwrap();
    }

    let query_embedding = db.get_or_compute_embedding("quasar nebula").unwrap();
    let exact = super::MemoryDB::vec_to_sql(&query_embedding);
    let opposite = super::MemoryDB::vec_to_sql(
        &query_embedding
            .iter()
            .map(|value| -*value)
            .collect::<Vec<_>>(),
    );
    let conn = db.conn.lock().await;
    conn.execute(
        "UPDATE memories SET embedding = vector32(?1) WHERE source_id = 'work-memory'",
        libsql::params![opposite],
    )
    .await
    .unwrap();
    conn.execute(
        "UPDATE memories SET embedding = vector32(?1) WHERE source_id LIKE 'personal-%'",
        libsql::params![exact],
    )
    .await
    .unwrap();
    drop(conn);

    let results = db
        .search_memory(
            "quasar nebula",
            1,
            None,
            &ReadScope::Space("work".to_string()),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_id, "work-memory");
}

#[tokio::test]
async fn legacy_search_boundary_scopes_before_vector_limit() {
    let (db, _tmp) = test_db().await;
    db.upsert_documents(vec![memory_doc("work-search", "work")])
        .await
        .unwrap();
    for index in 0..8 {
        db.upsert_documents(vec![memory_doc(
            &format!("personal-search-{index}"),
            "personal",
        )])
        .await
        .unwrap();
    }

    let query_embedding = db.get_or_compute_embedding("legacy route query").unwrap();
    let exact = super::MemoryDB::vec_to_sql(&query_embedding);
    let opposite = super::MemoryDB::vec_to_sql(
        &query_embedding
            .iter()
            .map(|value| -*value)
            .collect::<Vec<_>>(),
    );
    let conn = db.conn.lock().await;
    conn.execute(
        "UPDATE memories SET embedding = vector32(?1) WHERE source_id = 'work-search'",
        libsql::params![opposite],
    )
    .await
    .unwrap();
    conn.execute(
        "UPDATE memories SET embedding = vector32(?1) WHERE source_id LIKE 'personal-search-%'",
        libsql::params![exact],
    )
    .await
    .unwrap();
    drop(conn);

    let results = db
        .search(
            "legacy route query",
            1,
            Some("memory"),
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_id, "work-search");
}

#[tokio::test]
async fn search_pages_scopes_before_vector_limit() {
    let (db, _tmp) = test_db().await;
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        "work-page",
        "Work page",
        None,
        "deliberately unrelated page text",
        None,
        None,
        &[],
        &now,
        "distilled",
        "confirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    for index in 0..8 {
        db.insert_page_with_kind(
            &format!("personal-page-{index}"),
            "Personal page",
            None,
            "deliberately unrelated page text",
            None,
            None,
            &[],
            &now,
            "distilled",
            "confirmed",
            Some("personal"),
            None,
        )
        .await
        .unwrap();
    }

    let query_embedding = db.get_or_compute_embedding("quasar nebula").unwrap();
    let exact = super::MemoryDB::vec_to_sql(&query_embedding);
    let opposite = super::MemoryDB::vec_to_sql(
        &query_embedding
            .iter()
            .map(|value| -*value)
            .collect::<Vec<_>>(),
    );
    let conn = db.conn.lock().await;
    conn.execute(
        "UPDATE pages SET embedding = vector32(?1) WHERE id = 'work-page'",
        libsql::params![opposite],
    )
    .await
    .unwrap();
    conn.execute(
        "UPDATE pages SET embedding = vector32(?1) WHERE id LIKE 'personal-page-%'",
        libsql::params![exact],
    )
    .await
    .unwrap();
    drop(conn);

    let results = db
        .search_pages_scoped(
            "quasar nebula",
            1,
            None,
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "work-page");
}

#[tokio::test]
async fn selected_page_search_propagates_total_database_failure() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute_batch("ALTER TABLE pages RENAME TO pages_unavailable;")
        .await
        .unwrap();
    drop(conn);

    let result = db
        .search_pages_scoped("query", 10, None, &ReadScope::Space("work".to_string()))
        .await;

    assert!(
        result.is_err(),
        "total DB failure must not become a clean 200"
    );
}

#[tokio::test]
async fn cross_space_superseder_does_not_hide_selected_memory() {
    let (db, _tmp) = test_db().await;
    let mut target = memory_doc("work-target", "work");
    target.content = "deliberately unrelated candidate text work target".to_string();
    let mut superseder = memory_doc("personal-superseder", "personal");
    superseder.content = "deliberately unrelated candidate text personal superseder".to_string();
    superseder.supersedes = Some("work-target".to_string());
    superseder.supersede_mode = "hide".to_string();
    let mut archive_target = memory_doc("work-archive-target", "work");
    archive_target.content =
        "deliberately unrelated candidate text work archive target".to_string();
    let mut archive_superseder = memory_doc("personal-archive-superseder", "personal");
    archive_superseder.content =
        "deliberately unrelated candidate text personal archive superseder".to_string();
    archive_superseder.supersedes = Some("work-archive-target".to_string());
    archive_superseder.supersede_mode = "archive".to_string();
    db.upsert_documents(vec![target, superseder, archive_target, archive_superseder])
        .await
        .unwrap();
    assert_eq!(
        db.get_memory_space("personal-archive-superseder")
            .await
            .unwrap()
            .as_deref(),
        Some("personal")
    );
    let scope = ReadScope::Space("work".to_string());

    let ranked = db
        .search_memory(
            "deliberately unrelated candidate text",
            10,
            None,
            &scope,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let generic = db
        .search(
            "deliberately unrelated candidate text",
            10,
            Some("memory"),
            &scope,
        )
        .await
        .unwrap();
    let listed = db
        .list_memories_scoped(&scope, None, None, None, 10)
        .await
        .unwrap();
    let typed = db
        .load_memories_by_type_scoped("fact", 10, &scope)
        .await
        .unwrap();
    let filtered = db
        .list_filtered_scoped(Some("memory"), None, &scope, 10)
        .await
        .unwrap();

    assert!(
        ranked
            .iter()
            .any(|memory| memory.source_id == "work-target"),
        "a personal superseder must not alter work search visibility"
    );
    assert!(
        generic
            .iter()
            .any(|memory| memory.source_id == "work-target"),
        "the generic ranked path must share scoped superseder isolation"
    );
    assert!(
        listed
            .iter()
            .any(|memory| memory.source_id == "work-target"),
        "a personal superseder must not alter work collection visibility"
    );
    assert!(
        typed.iter().any(|memory| memory.source_id == "work-target"),
        "typed collections must share scoped superseder isolation"
    );
    assert!(
        filtered
            .iter()
            .any(|memory| memory.source_id == "work-target"),
        "filtered collections must share scoped superseder isolation"
    );
    assert!(
        ranked
            .iter()
            .find(|memory| memory.source_id == "work-archive-target")
            .is_some_and(|memory| !memory.is_archived),
        "a personal archive superseder must not mark a work result archived: {ranked:#?}"
    );

    db.set_stability("work-target", "new").await.unwrap();
    let nurture = db.get_nurture_cards_scoped(10, &scope).await.unwrap();
    assert!(
        nurture
            .iter()
            .any(|memory| memory.source_id == "work-target"),
        "nurture collections must share scoped superseder isolation"
    );
}

#[tokio::test]
async fn selected_page_visibility_requires_matching_workspace() {
    let (db, _tmp) = test_db().await;
    let source_ids = HashSet::from(["work-memory".to_string()]);
    let visible = db
        .select_visible_pages_scoped(
            vec![
                page("work", Some("work")),
                page("personal", Some("personal")),
            ],
            &ReadScope::Space("work".to_string()),
            &source_ids,
            "full",
            10,
        )
        .await;

    assert_eq!(
        visible
            .iter()
            .map(|page| page.id.as_str())
            .collect::<Vec<_>>(),
        vec!["work"]
    );
}

#[tokio::test]
async fn search_pages_route_helpers_bind_by_scope() {
    let (db, _tmp) = test_db().await;
    let now = chrono::Utc::now().to_rfc3339();
    // Single-axis (spec §1): a page's scope is one honest column. `workspace`
    // wins when present; otherwise `space` is the scope. An uncategorized page
    // has NEITHER set (both resolve to 'unfiled'). The former "category
    // independent of workspace" premise is exactly what M1 deletes -- so
    // null-page-route seeds no scope at all rather than a leftover category.
    for (id, space, workspace) in [
        ("work-page-route", None, Some("work")),
        ("personal-page-route", None, Some("personal")),
        ("null-page-route", None, None),
    ] {
        db.insert_page_with_kind(
            id,
            id,
            None,
            "page route scope canary",
            None,
            space,
            &[],
            &now,
            "authored",
            "confirmed",
            workspace,
            None,
        )
        .await
        .unwrap();
    }

    let scope = ReadScope::Space("work".to_string());
    let listed = db.list_pages_scoped("active", 10, 0, &scope).await.unwrap();
    assert_eq!(
        listed
            .iter()
            .map(|page| page.id.as_str())
            .collect::<Vec<_>>(),
        vec!["work-page-route"]
    );

    let recent = db
        .list_recent_pages_with_badges_scoped(10, None, &scope)
        .await
        .unwrap();
    assert_eq!(
        recent
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["work-page-route"]
    );

    let changes = db.list_recent_changes_scoped(10, &scope).await.unwrap();
    assert_eq!(
        changes
            .iter()
            .map(|change| change.page_id.as_str())
            .collect::<Vec<_>>(),
        vec!["work-page-route"]
    );

    assert!(db
        .get_page_scoped("work-page-route", &scope)
        .await
        .unwrap()
        .is_some());
    assert!(db
        .get_page_scoped("personal-page-route", &scope)
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        db.get_page_changelog_scoped("personal-page-route", &scope).await,
        Err(crate::WenlanError::NotFound(message)) if message == "page not found"
    ));

    let null_pages = db
        .list_pages_scoped("active", 10, 0, &ReadScope::Uncategorized)
        .await
        .unwrap();
    assert_eq!(
        null_pages
            .iter()
            .map(|page| page.id.as_str())
            .collect::<Vec<_>>(),
        vec!["null-page-route"]
    );
}

#[tokio::test]
async fn page_links_scoped_gate_parent_and_filter_source_pages() {
    let (db, _tmp) = test_db().await;
    let now = chrono::Utc::now().to_rfc3339();
    for (id, workspace) in [
        ("work-target", "work"),
        ("work-source", "work"),
        ("personal-source", "personal"),
        ("personal-parent", "personal"),
    ] {
        db.insert_page_with_kind(
            id,
            id,
            None,
            "page link scope canary",
            None,
            Some("decision"),
            &[],
            &now,
            "authored",
            "confirmed",
            Some(workspace),
            None,
        )
        .await
        .unwrap();
    }
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO page_links (source_page_id, target_page_id, label, label_key)
         VALUES ('work-source', 'work-target', 'Work target', 'work target'),
                ('personal-source', 'work-target', 'Work target', 'work target'),
                ('work-source', NULL, 'Work orphan', 'work orphan'),
                ('personal-source', NULL, 'Personal orphan', 'personal orphan')",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    let scope = ReadScope::Space("work".to_string());
    let outbound = db
        .get_page_outbound_links_scoped("work-source", &scope)
        .await
        .unwrap();
    assert_eq!(outbound.len(), 2);
    let inbound = db
        .get_page_inbound_links_scoped("work-target", &scope)
        .await
        .unwrap();
    assert_eq!(
        inbound,
        vec![("work-source".to_string(), "Work target".to_string())]
    );
    let orphans = db.list_orphan_link_labels_scoped(1, &scope).await.unwrap();
    assert_eq!(orphans, vec![("Work orphan".to_string(), 1)]);
    assert!(matches!(
        db.get_page_outbound_links_scoped("personal-parent", &scope).await,
        Err(crate::WenlanError::NotFound(message)) if message == "page not found"
    ));
    assert!(matches!(
        db.get_page_sources_scoped("personal-parent", &scope).await,
        Err(crate::WenlanError::NotFound(message)) if message == "page not found"
    ));
}

#[tokio::test]
async fn selected_summary_requires_nonempty_all_matching_sources() {
    let (db, _tmp) = test_db().await;
    db.upsert_documents(vec![
        memory_doc("work-summary-source", "work"),
        memory_doc("personal-summary-source", "personal"),
    ])
    .await
    .unwrap();
    let embedding = db.get_or_compute_embedding("summary scope query").unwrap();
    db.insert_summary_node(
        "work-summary",
        0,
        Some("work"),
        "Work summary",
        "summary scope query",
        &embedding,
        1,
        1,
        &["work-summary-source".to_string()],
    )
    .await
    .unwrap();
    db.insert_summary_node(
        "mixed-summary",
        0,
        Some("mixed"),
        "Mixed summary",
        "summary scope query",
        &embedding,
        2,
        1,
        &[
            "work-summary-source".to_string(),
            "personal-summary-source".to_string(),
        ],
    )
    .await
    .unwrap();
    db.insert_summary_node(
        "empty-summary",
        0,
        Some("empty"),
        "Empty summary",
        "summary scope query",
        &embedding,
        0,
        1,
        &[],
    )
    .await
    .unwrap();

    let results = db
        .search_summary_nodes_scoped(
            "summary scope query",
            10,
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert_eq!(
        results
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["work-summary"]
    );
}
