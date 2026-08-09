use super::*;
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use wenlan_types::lint::{LintOutcome, LintQuery};

async fn assert_mixed_lifecycle(
    pending: [i64; 2],
    recap: [i64; 2],
    stability: [&str; 2],
    mode: [&str; 2],
) {
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (
            id, content, source, source_id, title, chunk_index, last_modified,
            chunk_type, stability, supersede_mode, pending_revision, is_recap,
            needs_reembed, memory_type, word_count
         ) VALUES
            ('lifecycle-0', 'first', 'memory', 'mixed-lifecycle', 'mixed', 0, 1,
             'text', ?1, ?2, ?3, ?4, 1, 'fact', 1),
            ('lifecycle-1', 'second', 'memory', 'mixed-lifecycle', 'mixed', 1, 1,
             'text', ?5, ?6, ?7, ?8, 1, 'fact', 1)",
        libsql::params![
            stability[0],
            mode[0],
            pending[0],
            recap[0],
            stability[1],
            mode[1],
            pending[1],
            recap[1]
        ],
    )
    .await
    .unwrap();
    drop(conn);

    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap();
    let lifecycle = report
        .checks()
        .iter()
        .find(|check| check.check_id() == LIFECYCLE_ID)
        .unwrap();
    assert_eq!(lifecycle.outcome(), LintOutcome::Finding);
}

#[tokio::test]
async fn pending_revision_must_agree_across_chunks() {
    assert_mixed_lifecycle([0, 1], [0, 0], ["new", "new"], ["hide", "hide"]).await;
}

#[tokio::test]
async fn recap_state_must_agree_across_chunks() {
    assert_mixed_lifecycle([0, 0], [0, 1], ["new", "new"], ["hide", "hide"]).await;
}

#[tokio::test]
async fn stability_must_agree_across_chunks() {
    assert_mixed_lifecycle([0, 0], [0, 0], ["new", "confirmed"], ["hide", "hide"]).await;
}

#[tokio::test]
async fn supersede_mode_must_agree_across_chunks() {
    assert_mixed_lifecycle([0, 0], [0, 0], ["new", "new"], ["hide", "archive"]).await;
}
