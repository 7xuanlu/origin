use super::*;

#[tokio::test]
async fn equal_readiness_counts_cannot_hide_wrong_provider_model_or_reranker_path() {
    let (db, _temp) = test_db().await;
    let requested = RuntimeConfigSnapshot::disabled()
        .with_provider_request(ProviderClass::OnDevice, "model-a")
        .with_reranker_request(RerankerPath::Light, "reranker-a");
    let wrong_identity = RuntimeObservation::unavailable()
        .with_provider(ProviderClass::External, "model-a", RuntimeReadiness::Ready)
        .with_reranker(RerankerPath::Deep, "reranker-a", RuntimeReadiness::Ready)
        .with_ingest_worker_closed(false)
        .with_status_files(StatusFilesObservation::Direct(1));
    let report = run(
        &db,
        RuntimeRunConfig::for_test(requested.clone(), wrong_identity, None),
    )
    .await;
    assert_eq!(check(&report, PROVIDERS).outcome(), LintOutcome::Finding);

    let wrong_models = RuntimeObservation::unavailable()
        .with_provider(ProviderClass::OnDevice, "model-b", RuntimeReadiness::Ready)
        .with_reranker(RerankerPath::Light, "reranker-b", RuntimeReadiness::Ready)
        .with_ingest_worker_closed(false)
        .with_status_files(StatusFilesObservation::Direct(1));
    let report = run(
        &db,
        RuntimeRunConfig::for_test(requested, wrong_models, None),
    )
    .await;
    assert_eq!(check(&report, PROVIDERS).outcome(), LintOutcome::Finding);
}

#[tokio::test]
async fn status_mismatch_finds_and_direct_error_stays_incomplete_after_zero_fallback() {
    let (db, _temp) = test_db().await;
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type) VALUES ('status-row', '', 'memory', 'status-row', '', 0, 1, 'text')",
            (),
        )
        .await
        .unwrap();
    let mismatch = run(&db, config(RuntimeObservation::open(0))).await;
    assert_eq!(check(&mismatch, STATUS).outcome(), LintOutcome::Finding);

    let observation = RuntimeObservation::unavailable()
        .with_ingest_worker_closed(false)
        .with_status_files(StatusFilesObservation::DirectError {
            fallback_files_indexed: 0,
        });
    let report = run(&db, config(observation)).await;
    assert_eq!(check(&report, STATUS).outcome(), LintOutcome::FailedToRun);
}

#[tokio::test]
async fn available_working_memory_timestamp_detects_stale_recency() {
    let (db, _temp) = test_db().await;
    let stale =
        RuntimeObservation::open(1).with_working_memory(WorkingMemoryObservation::Available {
            entries: 1,
            newest_timestamp: Some(1_000),
        });
    let report = run_at(&db, config(stale), 2_000).await;
    assert_eq!(check(&report, WORKER).outcome(), LintOutcome::Finding);
    assert_eq!(
        metric_value(
            check(&report, WORKER),
            LintMetricCode::WorkingMemoryNewestAgeSeconds,
        ),
        1_000
    );

    let fresh =
        RuntimeObservation::open(1).with_working_memory(WorkingMemoryObservation::Available {
            entries: 1,
            newest_timestamp: Some(1_500),
        });
    assert_eq!(
        check(&run_at(&db, config(fresh), 2_000).await, WORKER).outcome(),
        LintOutcome::Pass
    );
}
