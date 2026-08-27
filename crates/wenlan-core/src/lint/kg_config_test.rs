use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use wenlan_types::lint::{
    LintApplicability, LintOutcome, LintPrecondition, LintQuery, LintSeverity,
};

const LIVENESS: &str = "kg.substrate_liveness";

#[tokio::test]
async fn production_capture_separates_stream_sweep_and_hub_cap_in_fingerprint() {
    let (db, _tmp) = test_db().await;
    insert_memory(&db).await;

    let sweep_off = production_run(&db, "1", "0", "20").await;
    let liveness = check(&sweep_off);
    assert_eq!(liveness.outcome(), LintOutcome::Finding);
    assert_eq!(liveness.severity(), LintSeverity::Warning);
    assert_eq!(liveness.applicability(), LintApplicability::Applicable);
    assert_eq!(liveness.precondition(), LintPrecondition::Ready);

    let different_cap = production_run(&db, "1", "0", "21").await;
    assert_ne!(
        sweep_off.config_fingerprint(),
        different_cap.config_fingerprint()
    );

    let sweep_on = production_run(&db, "1", "1", "20").await;
    assert_eq!(check(&sweep_on).outcome(), LintOutcome::Finding);
    assert_ne!(
        sweep_off.config_fingerprint(),
        sweep_on.config_fingerprint()
    );

    let stream_off = production_run(&db, "0", "0", "20").await;
    let liveness = check(&stream_off);
    assert_eq!(liveness.outcome(), LintOutcome::Pass);
    assert_eq!(liveness.applicability(), LintApplicability::ExpectedEmpty);
    assert_eq!(liveness.precondition(), LintPrecondition::ConfiguredOff);
    assert_ne!(
        sweep_off.config_fingerprint(),
        stream_off.config_fingerprint()
    );
}

#[test]
fn production_capture_reads_process_provider_readiness_once() {
    let config = super::KgRunConfig::capture();
    assert_eq!(
        config.provider_ready,
        crate::llm_provider::llm_provider_ready()
    );
    let opposite = super::KgRunConfig::for_test(
        config.serving_enabled,
        config.sweep_enabled,
        !config.provider_ready,
        config.model_source_configured,
        config.hub_cap,
    );
    let captured = wenlan_types::lint::LintConfigFingerprint::from_effective_config(
        &config.fingerprint_selections(),
    );
    let changed = wenlan_types::lint::LintConfigFingerprint::from_effective_config(
        &opposite.fingerprint_selections(),
    );
    assert_ne!(captured, changed);
}

async fn production_run(
    db: &crate::db::MemoryDB,
    stream: &str,
    sweep: &str,
    hub_cap: &str,
) -> wenlan_types::lint::LintReport {
    // `capture()` reads the everyday-source pin from config, and an empty
    // substrate is only a finding once a model source exists to have filled it.
    // Point the config at a scratch data dir that has one, so this test keeps
    // measuring what it is about: fingerprint separation.
    let data_dir = tempfile::tempdir().expect("scratch data dir");
    std::fs::write(
        data_dir.path().join("config.json"),
        br#"{"everyday_source":"on_device"}"#,
    )
    .expect("scratch config");
    let config = temp_env::with_vars(
        [
            ("WENLAN_GRAPH_MEMORY_STREAM", Some(stream)),
            ("WENLAN_ENABLE_ENTITY_SWEEP", Some(sweep)),
            ("WENLAN_GRAPH_HUB_CAP", Some(hub_cap)),
            (
                "WENLAN_DATA_DIR",
                Some(data_dir.path().to_str().expect("utf-8 scratch path")),
            ),
        ],
        super::KgRunConfig::capture,
    );
    assert!(config.model_source_configured);
    LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .with_test_kg_config(config)
        .run(
            db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap()
}

fn check(report: &wenlan_types::lint::LintReport) -> &wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|result| result.check_id() == LIVENESS)
        .unwrap()
}

async fn insert_memory(db: &crate::db::MemoryDB) {
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, stability, supersede_mode, needs_reembed, memory_type) VALUES ('eligible', 'eligible body', 'memory', 'eligible', 'eligible', 0, 1, 'text', 'new', 'hide', 1, 'fact')",
            (),
        )
        .await
        .unwrap();
}
