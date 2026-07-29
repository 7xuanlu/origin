// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use wenlan_types::repair::{RepairEnrichmentStep, RepairScope, RepairTarget};

async fn seed_entity_extraction_target(db: &MemoryDB) {
    let connection = db.conn.lock().await;
    connection
        .execute_batch(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('space-work','work',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,space)
             VALUES ('row-receipt','Target memory','memory','mem-receipt','target',0,10,
                     'text',0,0,'hide','fact','work');
             INSERT INTO entities
                 (id,name,entity_type,space,created_at,updated_at)
             VALUES ('ent-existing','Existing','concept','work',1,1),
                    ('ent-new','New','concept','work',1,1);
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('mem-receipt','ent-existing');
             INSERT INTO enrichment_steps
                 (source_id,step_name,status,error,attempts,updated_at)
             VALUES ('mem-receipt','entity_extract','failed','transient',2,1721000000);",
        )
        .await
        .unwrap();
}

fn entity_extraction_target(scope: RepairScope) -> RepairTarget {
    RepairTarget::MemoryEntityExtraction {
        memory_id: "mem-receipt".to_string(),
        step: RepairEnrichmentStep::EntityExtract,
        entity_ids: vec!["ent-new".to_string()],
        scope,
    }
}

#[tokio::test]
async fn repair_receipt_matches_connection_helper_and_preserves_scope_guard() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_entity_extraction_target(&db).await;
    let target = entity_extraction_target(RepairScope::registered("work".to_string()).unwrap());

    let expected = {
        let connection = db.conn.lock().await;
        crate::repair::repair_target_receipt_on_connection(&connection, &target)
            .await
            .unwrap()
    };
    let actual = db.read_repair_target_receipt(&target).await.unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.1, 1);
    assert_ne!(
        actual.0.as_str(),
        "0000000000000000000000000000000000000000000000000000000000000000"
    );

    let wrong_scope =
        entity_extraction_target(RepairScope::registered("personal".to_string()).unwrap());
    assert!(matches!(
        db.read_repair_target_receipt(&wrong_scope).await,
        Err(crate::error::WenlanError::Conflict(message))
            if message == "repair_target_stale"
    ));
}
