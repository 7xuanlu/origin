use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use crate::lint::snapshot::LintReadSnapshot;
use crate::lint::test_support::DbSemanticFingerprint;
use wenlan_types::lint::{
    LintEvidenceRef, LintMetricCode, LintMetricValue, LintOutcome, LintQuery,
};

const REGISTRY: &str = "identity.registry_integrity";
const MEMORY: &str = "identity.memory_state_integrity";
const TAGS: &str = "identity.tag_integrity";
const SESSIONS: &str = "identity.session_structure";
const CACHES: &str = "identity.cache_inventory";

#[tokio::test]
async fn scoped_rows_are_isolated_redacted_and_read_only() {
    let (db, _temp) = test_db().await;
    seed_spaces(&db).await;
    seed_invalid_memory(&db, "alpha-secret-id", "alpha").await;
    let before = fingerprint(&db).await;
    let first = run_lint(&db, Some("alpha")).await;
    assert_eq!(before, fingerprint(&db).await);
    assert_eq!(check(&first, MEMORY).coverage().denominator(), 1);
    assert_eq!(check(&first, MEMORY).outcome(), LintOutcome::Finding);
    assert_eq!(
        metric(check(&first, MEMORY), LintMetricCode::DecisionMemories),
        1
    );

    seed_invalid_memory(&db, "beta-secret-id", "beta").await;
    let second = run_lint(&db, Some("alpha")).await;
    assert_eq!(check(&first, MEMORY), check(&second, MEMORY));
    assert!(check(&second, SESSIONS).evidence().is_empty());

    let json = serde_json::to_string(&second).unwrap();
    for secret in [
        "alpha-secret-id",
        "beta-secret-id",
        "secret title",
        "/secret/path",
    ] {
        assert!(!json.contains(secret), "privacy canary leaked: {secret}");
    }
}

#[tokio::test]
async fn impossible_registry_session_cache_and_tag_rows_are_findings() {
    let (db, _temp) = test_db().await;
    db.conn.lock().await.execute_batch(
        "INSERT INTO agent_connections (id,name,agent_type,enabled,trust_level,memory_count,created_at,updated_at) VALUES ('bad-agent',' ','api',2,'forged',-1,2,1);
         INSERT INTO document_tags (source,source_id,tag) VALUES ('memory','missing-memory','secret-tag');
         INSERT INTO capture_refs (source_id,activity_id,snapshot_id,app_name,window_title,timestamp,source) VALUES ('capture-secret','missing-activity','missing-snapshot','secret app','secret title',10,'/secret/path');
         INSERT INTO briefing_cache (id,content,generated_at,memory_count) VALUES (2,'secret briefing',-1,-1);
         INSERT INTO narrative_cache (id,content,generated_at,memory_count) VALUES (2,'secret narrative',-1,-1);
         INSERT INTO agent_activity (timestamp,agent_name,action,memory_ids,query,detail) VALUES (10,' ','','raw-secret-id','secret query','secret detail');",
    ).await.unwrap();

    let report = run_lint(&db, None).await;
    for id in [REGISTRY, TAGS, SESSIONS, CACHES] {
        assert_eq!(check(&report, id).outcome(), LintOutcome::Finding, "{id}");
    }
    let json = serde_json::to_string(&report).unwrap();
    for secret in [
        "secret-tag",
        "secret title",
        "secret briefing",
        "raw-secret-id",
    ] {
        assert!(!json.contains(secret), "privacy canary leaked: {secret}");
    }
}

#[tokio::test]
async fn tag_evidence_identity_survives_earlier_row_deletion() {
    let (db, _temp) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('tag-owner-row','valid','memory','tag-owner','valid',0,7,'text',
                     0,0,'hide','fact','test',0,0,'new');
             INSERT INTO document_tags(source,source_id,tag)
             VALUES
                 ('aaa','missing-first','invalid-first'),
                 ('memory','tag-owner','valid-middle'),
                 ('zzz','missing-second','invalid-second');",
        )
        .await
        .unwrap();

    let before = run_lint(&db, None).await;
    let before_evidence = check(&before, TAGS).evidence().to_vec();
    assert_eq!(before_evidence.len(), 2);
    assert!(before_evidence
        .iter()
        .all(|evidence| matches!(evidence, LintEvidenceRef::OpaqueDigest { .. })));

    db.conn
        .lock()
        .await
        .execute(
            "DELETE FROM document_tags
             WHERE source='aaa' AND source_id='missing-first' AND tag='invalid-first'",
            (),
        )
        .await
        .unwrap();
    let after = run_lint(&db, None).await;
    assert_eq!(check(&after, TAGS).evidence(), &before_evidence[1..]);
}

#[tokio::test]
async fn importer_unknown_confirmation_and_provenance_agent_are_valid() {
    let (db, _temp) = test_db().await;
    db.conn
        .lock()
        .await
        .execute(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  confirmed,pinned,pending_revision,stability,supersede_mode,source_agent)
             VALUES ('import-row','imported','memory','import-source','imported',0,1,'text',
                     NULL,0,0,'new','hide','folder')",
            (),
        )
        .await
        .unwrap();

    let report = run_lint(&db, None).await;
    assert_eq!(check(&report, MEMORY).outcome(), LintOutcome::Pass);
}

#[tokio::test]
async fn cross_owner_and_out_of_window_session_rows_are_findings() {
    let (db, _temp) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO activities (id,started_at,ended_at) VALUES
             ('activity-a',100,200),('activity-b',300,400);
         INSERT INTO session_snapshots
             (id,activity_id,started_at,ended_at,primary_apps,summary,tags,capture_count,created_at)
             VALUES
             ('snapshot-cross','activity-b',310,390,'[]','','[]',1,390),
             ('snapshot-wide','activity-a',90,210,'[]','','[]',1,210);
         INSERT INTO capture_refs
             (source_id,activity_id,snapshot_id,app_name,window_title,timestamp,source)
             VALUES
             ('capture-cross','activity-a','snapshot-cross','','',150,''),
             ('capture-late','activity-a','snapshot-wide','','',250,'');",
        )
        .await
        .unwrap();

    let report = run_lint(&db, None).await;
    assert_eq!(check(&report, SESSIONS).outcome(), LintOutcome::Finding);
    assert!(affected(check(&report, SESSIONS)) >= 3);
}

#[tokio::test]
async fn full_population_is_evaluated_while_opaque_examples_stop_at_one_hundred() {
    let (db, _temp) = test_db().await;
    for index in 0..101 {
        seed_invalid_memory(&db, &format!("secret-{index:03}"), "uncategorized").await;
    }
    let result = check(&run_lint(&db, None).await, MEMORY).clone();
    assert_eq!(result.coverage().denominator(), 101);
    assert_eq!(result.coverage().evaluated(), 101);
    assert!(result.coverage().truncated());
    assert_eq!(result.evidence().len(), 100);
    assert_eq!(affected(&result), 101);
}

async fn run_lint(db: &crate::db::MemoryDB, space: Option<&str>) -> wenlan_types::lint::LintReport {
    LintRunner::new(LintClock::fixed_at(100), CancellationToken::new())
        .run(
            db,
            &LintQuery {
                profile: None,
                space: space.map(str::to_string),
            },
            None,
            false,
        )
        .await
        .unwrap()
}

async fn seed_spaces(db: &crate::db::MemoryDB) {
    db.conn.lock().await.execute_batch(
        "INSERT INTO spaces (id,name,created_at,updated_at) VALUES ('space-a','alpha',1,1),('space-b','beta',1,1);",
    ).await.unwrap();
}

async fn seed_invalid_memory(db: &crate::db::MemoryDB, id: &str, space: &str) {
    // M3 PR-1 stage e: memories.space is NOT NULL as of migration 91, so
    // "uncategorized" must bind the reserved sentinel id, not NULL.
    let space = if space == "uncategorized" {
        crate::db::UNFILED_SPACE_ID
    } else {
        space
    };
    db.conn.lock().await.execute(
        "INSERT INTO memories (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,confirmed,pinned,pending_revision,stability,supersede_mode,memory_type,space) VALUES (?1,'secret body','memory',?1,'secret title',0,1,'text',0,1,1,'impossible','hide','decision',?2)",
        libsql::params![id, space],
    ).await.unwrap();
}

fn check<'a>(
    report: &'a wenlan_types::lint::LintReport,
    id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|check| check.check_id() == id)
        .unwrap()
}

fn affected(result: &wenlan_types::lint::LintCheckResult) -> u64 {
    metric(result, LintMetricCode::AffectedRecords)
}

fn metric(result: &wenlan_types::lint::LintCheckResult, code: LintMetricCode) -> u64 {
    result
        .metrics()
        .iter()
        .find_map(|metric| match (metric.code(), metric.value()) {
            (observed, LintMetricValue::Count { value }) if observed == code => Some(*value),
            _ => None,
        })
        .unwrap()
}

async fn fingerprint(db: &crate::db::MemoryDB) -> DbSemanticFingerprint {
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
    let fingerprint = DbSemanticFingerprint::capture(&snapshot).await.unwrap();
    snapshot.finish().await.unwrap();
    fingerprint
}
