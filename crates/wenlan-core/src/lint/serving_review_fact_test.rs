use super::super::FACT_STARVATION_ID;
use super::review_tests::{check, insert_memory, metric, run};
use crate::db::tests::test_db;
use crate::lint::context::{
    AppliedScope, CancellationToken, ExecutionGate, LintClock, LintContext,
};
use crate::lint::serving::fact_probe::{run_with_ann, AnnTopK, RankedChild};
use std::cell::Cell;
use wenlan_types::lint::{LintMetricCode, LintOutcome};

#[tokio::test]
async fn snapshot_fact_probe_distinguishes_starvation_from_non_starvation() {
    let (starved, _tmp) = fact_fixture(false).await;
    let finding = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_FACT_CHANNEL", Some("1")),
            ("WENLAN_FACT_CHANNEL_LIMIT", Some("1")),
        ],
        run(&starved, Some("work")),
    )
    .await;
    let finding = check(&finding, FACT_STARVATION_ID);
    assert_eq!(finding.outcome(), LintOutcome::Finding);
    assert_eq!(metric(finding, LintMetricCode::EligibleRecords), 1);
    assert_eq!(metric(finding, LintMetricCode::AffectedRecords), 1);
    assert_eq!(finding.evidence().len(), 1);

    let (live, _tmp) = fact_fixture(true).await;
    let pass = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_FACT_CHANNEL", Some("1")),
            ("WENLAN_FACT_CHANNEL_LIMIT", Some("1")),
        ],
        run(&live, Some("work")),
    )
    .await;
    let pass = check(&pass, FACT_STARVATION_ID);
    assert_eq!(pass.outcome(), LintOutcome::Pass);
    assert_eq!(metric(pass, LintMetricCode::EligibleRecords), 1);
    assert_eq!(metric(pass, LintMetricCode::AffectedRecords), 0);
    assert!(pass.evidence().is_empty());
}

#[tokio::test]
async fn snapshot_fact_probe_uses_production_ann_limit_before_parent_join() {
    let (db, _tmp) = exact_ann_limit_fixture().await;
    let report = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_FACT_CHANNEL", Some("1")),
            ("WENLAN_FACT_CHANNEL_LIMIT", Some("1")),
        ],
        run(&db, Some("work")),
    )
    .await;
    let finding = check(&report, FACT_STARVATION_ID);
    assert_eq!(finding.outcome(), LintOutcome::Finding);
    assert_eq!(metric(finding, LintMetricCode::EligibleRecords), 1);
    assert_eq!(metric(finding, LintMetricCode::AffectedRecords), 1);
    assert_eq!(finding.evidence().len(), 1);
}

#[derive(Default)]
struct RecordingAnn {
    requested_k: Cell<Option<usize>>,
}

impl AnnTopK for RecordingAnn {
    async fn query(
        &self,
        _context: &LintContext<'_, '_>,
        _embedding: Vec<u8>,
        k: usize,
    ) -> Result<Vec<RankedChild>, ()> {
        self.requested_k.set(Some(k));
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn fact_probe_passes_three_times_parent_limit_to_ann_query() {
    let (db, _tmp) = exact_ann_limit_fixture().await;
    let snapshot = db
        .open_isolated_lint_snapshot_for_test()
        .await
        .expect("snapshot");
    let scope = AppliedScope::registered(
        wenlan_types::lint::LintOpaqueId::from_sorted_position(0).expect("opaque scope"),
        "work".to_string(),
    );
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let ann = RecordingAnn::default();

    run_with_ann(&context, 7, &ann).await.expect("fact probe");

    assert_eq!(ann.requested_k.get(), Some(21));
    snapshot.finish().await.expect("finish snapshot");
}

async fn fact_fixture(selected_near: bool) -> (crate::db::MemoryDB, tempfile::TempDir) {
    let (db, tmp) = test_db().await;
    db.create_space("work", None, false).await.expect("space");
    let memories = if selected_near {
        vec![("work-a", "work")]
    } else {
        vec![
            ("other-a", "other"),
            ("other-b", "other"),
            ("other-c", "other"),
            ("work-a", "work"),
        ]
    };
    for (id, space) in memories {
        insert_memory(&db, id, space).await;
    }
    insert_children(&db, selected_near).await;
    (db, tmp)
}

async fn insert_children(db: &crate::db::MemoryDB, selected_near: bool) {
    let near = vector(1.0, 0.0);
    let far = vector(0.0, 1.0);
    let work = if selected_near { &near } else { &far };
    let children = if selected_near {
        vec![("a-work", "work-a", work)]
    } else {
        vec![
            ("a-probe", "other-a", &near),
            ("b-other", "other-b", &near),
            ("c-other", "other-c", &near),
            ("d-work", "work-a", work),
        ]
    };
    let conn = db.conn.lock().await;
    for (id, parent, embedding) in children {
        conn.execute(
            "INSERT INTO child_vectors (id,parent_kind,parent_id,field,content,embedding) VALUES (?1,'memory',?2,'fact','body',vector32(?3))",
            libsql::params![id, parent, embedding.clone()],
        )
        .await
        .expect("child vector");
    }
}

async fn exact_ann_limit_fixture() -> (crate::db::MemoryDB, tempfile::TempDir) {
    let (db, tmp) = test_db().await;
    db.create_space("work", None, false).await.expect("space");
    for (id, space) in [
        ("other-a", "other"),
        ("other-b", "other"),
        ("work-a", "work"),
        ("other-far", "other"),
    ] {
        insert_memory(&db, id, space).await;
    }
    let children = [
        ("00-orphan", "missing-parent", vector_pair(1.0, 0.0)),
        ("a-probe", "other-a", vector_pair(0.99, 0.10)),
        ("b-other", "other-b", vector_pair(0.98, 0.20)),
        ("c-work", "work-a", vector_pair(0.97, 0.30)),
        ("d-far", "other-far", vector_pair(0.0, 1.0)),
    ];
    let conn = db.conn.lock().await;
    for (id, parent, embedding) in children {
        conn.execute(
            "INSERT INTO child_vectors (id,parent_kind,parent_id,field,content,embedding) VALUES (?1,'memory',?2,'fact','body',vector32(?3))",
            libsql::params![id, parent, embedding],
        )
        .await
        .expect("child vector");
    }
    drop(conn);
    (db, tmp)
}

fn vector(x: f32, y: f32) -> String {
    let mut values = vec!["0"; 768];
    values[0] = if x == 0.0 { "0" } else { "1" };
    values[1] = if y == 0.0 { "0" } else { "1" };
    format!("[{}]", values.join(","))
}

fn vector_pair(x: f32, y: f32) -> String {
    let mut values = vec!["0".to_string(); 768];
    values[0] = x.to_string();
    values[1] = y.to_string();
    format!("[{}]", values.join(","))
}
