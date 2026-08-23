use super::*;
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use wenlan_types::lint::{LintGateEffect, LintProfile, LintQuery};

fn check<'a>(report: &'a wenlan_types::lint::LintReport, id: &str) -> &'a LintCheckResult {
    report
        .checks()
        .iter()
        .find(|check| check.check_id() == id)
        .unwrap_or_else(|| panic!("missing check {id}"))
}

async fn deep_report(
    db: &crate::db::MemoryDB,
    page_root: Option<&std::path::Path>,
) -> wenlan_types::lint::LintReport {
    LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            db,
            &LintQuery {
                profile: Some(LintProfile::Deep),
                space: None,
            },
            page_root,
            page_root.is_some(),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn deep_profile_detects_structural_and_advisory_counterexamples() {
    let (db, _dir) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "-- G6 Stage 3 retirement lint track: migration 123 drops
             -- `entities` outright, so there is no legacy row to raw-INSERT
             -- for entity_a any more -- observation_duplicates ports to the
             -- canonical entity_page_map/pages join (INNER, same as the
             -- legacy `entities` join it replaces), so entity_a needs a
             -- shadow page or its observations drop out of that check's
             -- population entirely. observations/relations/edges/
             -- memory_entities below reference 'entity_a' as a bare id --
             -- no FK to `entities` remains to satisfy.
             -- obs_a/obs_b are deliberately near-duplicates (not identical):
             -- migration 125's idx_observations_identity is a UNIQUE index
             -- on (entity_id, lower(trim(content))), so two rows with the
             -- exact same content can no longer coexist here. Observation
             -- content dedup is covered by an explicit Pass assertion below
             -- instead of this structural fixture.
             INSERT INTO pages
                 (id,title,summary,content,kind,entity_type,confidence,entity_confirmed,
                  embedding,space,workspace,source_memory_ids,version,status,
                  created_at,last_compiled,last_modified,creation_kind,review_status,aliases)
             VALUES
                 ('page_entity_a','A',NULL,'','entity','person',NULL,0,
                  NULL,'unfiled','unfiled','[]',1,'active',0,0,0,'entity','unconfirmed','[]');
             INSERT INTO entity_page_map (entity_id,page_id,created_at)
             VALUES ('entity_a','page_entity_a',0);
             INSERT INTO observations
                 (id,entity_id,content,created_at)
             VALUES ('obs_a','entity_a','same a',0),('obs_b','entity_a','same b',0);
             INSERT INTO relations
                 (id,from_entity,to_entity,relation_type,created_at)
             VALUES ('rel_a','entity_a','entity_a','legacy_custom_type',0);
             -- G6 Stage 1.2: relation_vocabulary (reader #9) reads `edges`,
             -- not `relations` -- mirror the dual-write here.
             INSERT INTO edges
                 (edge_id,src_id,src_kind,dst_id,dst_kind,edge_type,lineage,grounded,space,created_at,semantic_type)
             VALUES ('edge-rel_a','entity_a','entity','entity_a','entity','relates','legacy',0,
                     '00000000-0000-4000-8000-000000000001',0,'legacy_custom_type');
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  memory_type,entity_id,pending_revision,is_recap,supersede_mode,structured_fields)
             VALUES
                 ('mem_row_a','same memory','memory','mem_a','a',0,0,'text',
                  'fact','entity_a',0,0,'hide','{\"value\":1}'),
                 ('mem_row_b','same memory','memory','mem_b','b',0,0,'text',
                  'fact','entity_a',0,0,'hide','{\"value\":2}');
             INSERT INTO pages
                 (id,title,content,source_memory_ids,version,status,created_at,last_compiled,
                  last_modified,creation_kind,review_status)
             VALUES
                 ('page_a','a','same page','[]',1,'active','now','now','now','distilled','confirmed'),
                 ('page_b','b','same page','[]',1,'active','now','now','now','distilled','confirmed');",
        )
        .await
        .unwrap();

    let report = deep_report(&db, None).await;
    assert_eq!(
        check(&report, RELATION_VOCABULARY).outcome(),
        LintOutcome::Finding
    );
    assert_eq!(
        check(&report, RELATION_VOCABULARY).gate_effect(),
        LintGateEffect::Actionable
    );
    for id in [
        MEMORY_DUPLICATES,
        RETRIEVAL_SUBSTRATE,
        CONFLICTS,
        PAGE_DUPLICATES,
    ] {
        assert_eq!(check(&report, id).outcome(), LintOutcome::Finding, "{id}");
        assert_eq!(check(&report, id).gate_effect(), LintGateEffect::Advisory);
        assert!(!check(&report, id).evidence().is_empty());
    }
    // obs_a/obs_b no longer share identical content (migration 125's unique
    // index forbids seeding that here -- see the fixture comment above), so
    // observation content dedup is now a Pass: this check has become a
    // tripwire for the invariant the index enforces at write time.
    let observation_duplicates = check(&report, OBSERVATION_DUPLICATES);
    assert_eq!(observation_duplicates.outcome(), LintOutcome::Pass);
    assert_eq!(
        observation_duplicates.gate_effect(),
        LintGateEffect::Advisory
    );
    assert!(observation_duplicates.evidence().is_empty());
    assert_eq!(
        check(&report, PAGE_BODY).outcome(),
        LintOutcome::NotRunPrerequisite
    );
    assert_eq!(
        check(&report, SOURCE_RESIDUE).outcome(),
        LintOutcome::NotRunPrerequisite
    );
}

// G6 Stage 3 retirement lint track: dropping `entity_aliases`' own
// UNIQUE(alias_name) means two active entity shadow pages can now claim the
// same lowercased alias (see `MemoryDB::add_entity_alias`'s write-time
// guard, db.rs) -- `alias_collisions` is the lint-visible detector for that.
#[tokio::test]
async fn alias_collision_inventory_flags_the_same_alias_on_two_active_pages() {
    let (db, _dir) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO pages
                 (id,title,summary,content,kind,entity_type,confidence,entity_confirmed,
                  embedding,space,workspace,source_memory_ids,version,status,
                  created_at,last_compiled,last_modified,creation_kind,review_status,aliases)
             VALUES
                 ('page_older','Older',NULL,'','entity','concept',NULL,0,
                  NULL,'unfiled','unfiled','[]',1,'active','2020-01-01T00:00:00Z',
                  '2020-01-01T00:00:00Z','2020-01-01T00:00:00Z','entity','unconfirmed',
                  '[\"collide alias\"]'),
                 ('page_newer','Newer',NULL,'','entity','concept',NULL,0,
                  NULL,'unfiled','unfiled','[]',1,'active','2021-01-01T00:00:00Z',
                  '2021-01-01T00:00:00Z','2021-01-01T00:00:00Z','entity','unconfirmed',
                  '[\"collide alias\"]');
             INSERT INTO entity_page_map (entity_id,page_id,created_at)
             VALUES ('collide-older','page_older','2020-01-01T00:00:00Z'),
                    ('collide-newer','page_newer','2021-01-01T00:00:00Z');",
        )
        .await
        .unwrap();

    let report = deep_report(&db, None).await;
    let result = check(&report, ALIAS_COLLISIONS);
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.gate_effect(), LintGateEffect::Advisory);
    assert!(!result.evidence().is_empty());
}

#[tokio::test]
async fn alias_collision_inventory_is_silent_on_a_clean_store() {
    let (db, _dir) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO pages
                 (id,title,summary,content,kind,entity_type,confidence,entity_confirmed,
                  embedding,space,workspace,source_memory_ids,version,status,
                  created_at,last_compiled,last_modified,creation_kind,review_status,aliases)
             VALUES
                 ('page_solo','Solo',NULL,'','entity','concept',NULL,0,
                  NULL,'unfiled','unfiled','[]',1,'active','2020-01-01T00:00:00Z',
                  '2020-01-01T00:00:00Z','2020-01-01T00:00:00Z','entity','unconfirmed',
                  '[\"unique alias\"]');
             INSERT INTO entity_page_map (entity_id,page_id,created_at)
             VALUES ('solo-entity','page_solo','2020-01-01T00:00:00Z');",
        )
        .await
        .unwrap();

    let report = deep_report(&db, None).await;
    assert_eq!(
        check(&report, ALIAS_COLLISIONS).outcome(),
        LintOutcome::Pass
    );
}

#[tokio::test]
async fn duplicate_inventory_compares_memory_heads_not_shared_later_chunks() {
    let (db, _dir) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  memory_type,pending_revision,is_recap,supersede_mode)
             VALUES
                 ('row_a0','session alpha','memory','mem_a','a',0,0,'text',
                  'fact',0,0,'hide'),
                 ('row_a1','## Notes','memory','mem_a','a',1,0,'text',
                  'fact',0,0,'hide'),
                 ('row_b0','session beta','memory','mem_b','b',0,0,'text',
                  'fact',0,0,'hide'),
                 ('row_b1','## Notes','memory','mem_b','b',1,0,'text',
                  'fact',0,0,'hide');",
        )
        .await
        .unwrap();

    let report = deep_report(&db, None).await;
    assert_eq!(
        check(&report, MEMORY_DUPLICATES).outcome(),
        LintOutcome::Pass
    );
    assert_eq!(
        check(&report, MEMORY_DUPLICATES).coverage().denominator(),
        2
    );
}

#[tokio::test]
async fn deep_page_body_alignment_uses_state_mapping_and_canonical_body() {
    let (db, _dir) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO pages
                 (id,title,content,source_memory_ids,version,status,created_at,last_compiled,
                  last_modified,creation_kind,review_status)
             VALUES
                 ('page_a','a','body','[]',1,'active','now','now','now','distilled','confirmed'),
                 ('page_unprojected','missing','other','[]',1,'active','now','now','now','distilled','confirmed');",
        )
        .await
        .unwrap();
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join(".wenlan")).unwrap();
    std::fs::write(
        root.path().join(".wenlan/state.json"),
        r#"{"schema_version":2,"pages":{"page_a":{"file":"page.md","version":1}}}"#,
    )
    .unwrap();
    std::fs::write(
        root.path().join("page.md"),
        "---\norigin_id: page_a\norigin_version: 1\n---\nbody\n\n<!-- origin:sources:start -->\n## Sources\n- [[mem_a]]\n<!-- origin:sources:end -->\n",
    )
    .unwrap();

    let aligned = deep_report(&db, Some(root.path())).await;
    assert_eq!(check(&aligned, PAGE_BODY).outcome(), LintOutcome::Pass);
    assert_eq!(check(&aligned, PAGE_BODY).coverage().denominator(), 1);

    std::fs::write(
        root.path().join("page.md"),
        "---\norigin_id: page_a\norigin_version: 1\n---\ndrifted\n",
    )
    .unwrap();
    let drifted = deep_report(&db, Some(root.path())).await;
    assert_eq!(check(&drifted, PAGE_BODY).outcome(), LintOutcome::Finding);
    assert_eq!(check(&drifted, PAGE_BODY).severity(), LintSeverity::Error);
}
