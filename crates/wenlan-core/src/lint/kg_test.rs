use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use crate::lint::snapshot::LintReadSnapshot;
use crate::lint::test_support::DbSemanticFingerprint;
use wenlan_types::lint::{
    LintApplicability, LintMetricCode, LintMetricValue, LintOutcome, LintPrecondition, LintQuery,
    LintSeverity,
};

const ENTITY_INTEGRITY: &str = "entities.structural_integrity";
const ENTITY_PARTITIONS: &str = "entities.partition_inventory";
const ADVISORY: &str = "kg.advisory_inventory";
const AGGREGATES: &str = "kg.aggregate_inventory";
const LIVENESS: &str = "kg.substrate_liveness";
const LINKS: &str = "memory_entities.integrity";
const OBSERVATIONS: &str = "observations.integrity";
const RELATIONS: &str = "relations.integrity";

#[tokio::test]
async fn structural_orphans_and_invalid_rows_are_errors_without_mutation() {
    let (db, _tmp) = test_db().await;
    seed_corrupt_graph(&db).await;
    let before = fingerprint(&db).await;

    let report = run(&db, None, test_config(true)).await;

    for (id, affected) in [
        (ENTITY_INTEGRITY, 1),
        (LINKS, 3),
        (OBSERVATIONS, 2),
        (RELATIONS, 2),
    ] {
        let result = check(&report, id);
        assert_eq!(result.outcome(), LintOutcome::Finding, "{id}");
        assert_eq!(result.severity(), LintSeverity::Error, "{id}");
        assert_eq!(metric(result, LintMetricCode::AffectedRecords), affected);
        assert_eq!(result.evidence().len(), affected as usize);
    }
    assert_eq!(before, fingerprint(&db).await);
    let json = serde_json::to_string(&report).unwrap();
    for secret in [
        "Secret Entity",
        "secret observation",
        "secret_relation",
        "missing-id",
    ] {
        assert!(!json.contains(secret), "privacy canary leaked: {secret}");
    }
}

#[tokio::test]
async fn imported_document_entity_links_have_a_valid_memory_owner() {
    let (db, _tmp) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO entities
                 (id,name,entity_type,confirmed,created_at,updated_at)
             VALUES ('entity-doc','Imported Topic','concept',0,1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type)
             VALUES ('row-doc','Imported Topic','local_files','/tmp/doc.md','doc',0,1,'text');
             INSERT INTO memory_entities (memory_id,entity_id)
             VALUES ('/tmp/doc.md','entity-doc');",
        )
        .await
        .unwrap();

    let report = run(&db, None, test_config(true)).await;
    assert_eq!(check(&report, LINKS).outcome(), LintOutcome::Pass);
}

#[tokio::test]
async fn scoped_rows_follow_memory_or_entity_ownership_but_aggregates_stay_global() {
    let (db, _tmp) = test_db().await;
    seed_valid_scoped_graph(&db).await;

    let report = run(&db, Some("alpha"), test_config(true)).await;

    assert_eq!(check(&report, ENTITY_INTEGRITY).coverage().denominator(), 1);
    assert_eq!(
        check(&report, ENTITY_PARTITIONS).coverage().denominator(),
        1
    );
    assert_eq!(check(&report, OBSERVATIONS).coverage().denominator(), 1);
    assert_eq!(check(&report, RELATIONS).coverage().denominator(), 1);
    assert_eq!(check(&report, LINKS).coverage().denominator(), 1);
    let aggregates = check(&report, AGGREGATES);
    assert_eq!(aggregates.coverage().denominator(), 8);
    assert!(aggregates.evidence().is_empty());
    assert_eq!(metric(aggregates, LintMetricCode::KgEntities), 2);
    assert_eq!(metric(aggregates, LintMetricCode::KgRelations), 1);
    assert_eq!(metric(aggregates, LintMetricCode::KgObservations), 2);
    assert_eq!(metric(aggregates, LintMetricCode::KgMemoryEntityLinks), 3);
}

#[tokio::test]
async fn configured_off_is_expected_empty_and_enabled_empty_substrate_is_actionable() {
    let (db, _tmp) = test_db().await;
    insert_memory(&db, "eligible", None).await;

    let off = run(&db, None, test_config(false)).await;
    let off_liveness = check(&off, LIVENESS);
    assert_eq!(off_liveness.outcome(), LintOutcome::Pass);
    assert_eq!(
        off_liveness.applicability(),
        LintApplicability::ExpectedEmpty
    );
    assert_eq!(off_liveness.precondition(), LintPrecondition::ConfiguredOff);

    let on = run(&db, None, test_config(true)).await;
    let on_liveness = check(&on, LIVENESS);
    assert_eq!(on_liveness.outcome(), LintOutcome::Finding);
    assert_eq!(on_liveness.severity(), LintSeverity::Warning);
    assert_eq!(metric(on_liveness, LintMetricCode::EligibleRecords), 1);
}

#[tokio::test]
async fn duplicate_names_hubs_and_semantic_suspicion_are_advisory_only() {
    let (db, _tmp) = test_db().await;
    seed_advisory_graph(&db).await;

    let result = check(&run(&db, None, test_config(true)).await, ADVISORY).clone();

    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.severity(), LintSeverity::Info);
    assert_eq!(result.applicability(), LintApplicability::Inventory);
    assert!(result.evidence().is_empty());
    assert_eq!(metric(&result, LintMetricCode::KgDuplicateEntityNames), 1);
    assert_eq!(metric(&result, LintMetricCode::KgHubEntities), 1);
    assert_eq!(metric(&result, LintMetricCode::KgSemanticSuspicions), 1);
}

#[tokio::test]
async fn full_population_is_validated_while_opaque_evidence_is_capped() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute("PRAGMA foreign_keys = OFF", ()).await.unwrap();
    for index in 0..101 {
        conn.execute(
            "INSERT INTO observations (id, entity_id, content, confirmed, created_at) VALUES (?1, 'missing-id', 'secret observation', 0, 1)",
            libsql::params![format!("obs-{index:03}")],
        ).await.unwrap();
    }
    drop(conn);

    let report = run(&db, None, test_config(true)).await;
    let result = check(&report, OBSERVATIONS);
    assert_eq!(result.coverage().denominator(), 101);
    assert_eq!(result.coverage().evaluated(), 101);
    assert!(result.coverage().truncated());
    assert_eq!(result.evidence().len(), 100);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), 101);
}

async fn run(
    db: &crate::db::MemoryDB,
    space: Option<&str>,
    config: super::KgRunConfig,
) -> wenlan_types::lint::LintReport {
    LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .with_test_kg_config(config)
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

fn test_config(serving_enabled: bool) -> super::KgRunConfig {
    super::KgRunConfig::for_test(serving_enabled, true, true, 20)
}

fn check<'a>(
    report: &'a wenlan_types::lint::LintReport,
    id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|result| result.check_id() == id)
        .unwrap()
}

fn metric(result: &wenlan_types::lint::LintCheckResult, code: LintMetricCode) -> u64 {
    result
        .metrics()
        .iter()
        .find_map(|metric| {
            (metric.code() == code).then(|| match metric.value() {
                LintMetricValue::Count { value } => *value,
                _ => 0,
            })
        })
        .unwrap()
}

async fn fingerprint(db: &crate::db::MemoryDB) -> DbSemanticFingerprint {
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
    let fingerprint = DbSemanticFingerprint::capture(&snapshot).await.unwrap();
    snapshot.finish().await.unwrap();
    fingerprint
}

async fn insert_memory(db: &crate::db::MemoryDB, id: &str, space: Option<&str>) {
    // M3 PR-1 stage e: memories.space is NOT NULL as of migration 91, so
    // "no space" must bind the reserved sentinel id, not NULL.
    let space = space.unwrap_or(crate::db::UNFILED_SPACE_ID);
    db.conn.lock().await.execute(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, stability, supersede_mode, needs_reembed, memory_type, space) VALUES (?1, 'eligible body', 'memory', ?1, ?1, 0, 1, 'text', 'new', 'hide', 1, 'fact', ?2)",
        libsql::params![id, space],
    ).await.unwrap();
}

async fn seed_corrupt_graph(db: &crate::db::MemoryDB) {
    insert_memory(db, "memory-ok", None).await;
    let conn = db.conn.lock().await;
    conn.execute("PRAGMA foreign_keys = OFF", ()).await.unwrap();
    conn.execute_batch(
        "INSERT INTO entities (id,name,entity_type,space,confidence,confirmed,created_at,updated_at) VALUES ('entity-ok','Secret Entity','concept',NULL,0.8,0,1,1),('entity-bad',' ','', 'missing-space',1.5,2,1,1);
         INSERT INTO observations (id,entity_id,content,confidence,confirmed,created_at) VALUES ('obs-orphan','missing-id','secret observation',0.5,0,1),('obs-invalid','entity-ok',' ',2.0,2,1);
         INSERT INTO relations (id,from_entity,to_entity,relation_type,created_at) VALUES ('relation-from','missing-id','entity-ok','secret_relation',1),('relation-to','entity-ok','missing-id','secret_relation',1);
         INSERT INTO memory_entities (memory_id,entity_id) VALUES ('missing-id','entity-ok'),('memory-ok','missing-id'),('entity-ok','entity-ok');"
    ).await.unwrap();
}

async fn seed_valid_scoped_graph(db: &crate::db::MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute_batch("INSERT INTO spaces (id,name,created_at,updated_at) VALUES ('s-a','alpha',1,1),('s-b','beta',1,1);").await.unwrap();
    drop(conn);
    insert_memory(db, "mem-alpha", Some("alpha")).await;
    insert_memory(db, "mem-beta", Some("beta")).await;
    insert_memory(db, "mem-none", None).await;
    db.conn.lock().await.execute_batch(
        "INSERT INTO entities (id,name,entity_type,space,confirmed,created_at,updated_at) VALUES ('ent-a','Alpha','concept','alpha',0,1,1),('ent-b','Beta','concept','beta',0,1,1);
         INSERT INTO observations (id,entity_id,content,confirmed,created_at) VALUES ('obs-a','ent-a','a',0,1),('obs-b','ent-b','b',0,1);
         INSERT INTO relations (id,from_entity,to_entity,relation_type,created_at) VALUES ('rel-a','ent-a','ent-b','related',1);
         INSERT INTO memory_entities (memory_id,entity_id) VALUES ('mem-alpha','ent-a'),('mem-beta','ent-b'),('mem-none','ent-a');"
    ).await.unwrap();
}

async fn seed_advisory_graph(db: &crate::db::MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute_batch("INSERT INTO entities (id,name,entity_type,confirmed,created_at,updated_at) VALUES ('hub','Shared','person',0,1,1),('dupe',' shared ','concept',0,1,1);").await.unwrap();
    drop(conn);
    for index in 0..21 {
        let id = format!("mem-{index:02}");
        insert_memory(db, &id, None).await;
        db.conn
            .lock()
            .await
            .execute(
                "INSERT INTO memory_entities (memory_id,entity_id) VALUES (?1,'hub')",
                libsql::params![id],
            )
            .await
            .unwrap();
    }
}
