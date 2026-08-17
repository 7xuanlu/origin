// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::{Arc, OnceLock};
use tempfile::TempDir;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;
use wenlan_server::router::build_router;
use wenlan_server::state::ServerState;
use wenlan_types::requests::StoreMemoryRequest;
use wenlan_types::{
    BriefSummaryUpdate, BriefUpdateRequest, OutboxDrainReport, OutboxEnvelope, OutboxPayload,
    WriteSpaceTarget, OUTBOX_SCHEMA,
};

fn data_dir_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

struct DataDirGuard {
    previous: Option<std::ffi::OsString>,
    _temp: TempDir,
}

impl DataDirGuard {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("WENLAN_DATA_DIR");
        std::env::set_var("WENLAN_DATA_DIR", temp.path());
        Self {
            previous,
            _temp: temp,
        }
    }
}

impl Drop for DataDirGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
            None => std::env::remove_var("WENLAN_DATA_DIR"),
        }
    }
}

fn envelope_file(envelope: &OutboxEnvelope, timestamp: u128) {
    let directory = wenlan_core::config::outbox_dir();
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join(envelope.file_name(timestamp)),
        serde_json::to_vec_pretty(envelope).unwrap(),
    )
    .unwrap();
}

async fn response_json<T: serde::de::DeserializeOwned>(
    response: axum::response::Response<Body>,
) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn drain_replays_both_kinds_and_handles_duplicates_and_malformed_files() {
    let _env_lock = data_dir_lock().lock().await;
    let _data = DataDirGuard::new();
    let db_path = wenlan_core::config::data_root().join("memorydb");
    let db = Arc::new(
        MemoryDB::new(&db_path, Arc::new(NoopEmitter))
            .await
            .unwrap(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = Arc::new(RwLock::new(
        ServerState {
            db: Some(Arc::clone(&db)),
            ..ServerState::default()
        }
        .with_bound_port(port),
    ));
    let app = build_router(state);
    let server = tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });

    let brief = OutboxEnvelope {
        schema: OUTBOX_SCHEMA,
        created_at: "2026-08-16T12:00:00.000Z".into(),
        caller_id: "test".into(),
        operation_id: "brief-drain".into(),
        space: Some("demo".into()),
        agent_name: None,
        reconcile_summary_version: true,
        payload: OutboxPayload::BriefUpdate(BriefUpdateRequest {
            space: "demo".into(),
            caller_id: "test".into(),
            operation_id: "brief-drain".into(),
            summary: Some(BriefSummaryUpdate {
                text: "drained summary".into(),
                expected_version: 999,
            }),
            mutations: vec![],
        }),
    };
    let memory = OutboxEnvelope {
        schema: OUTBOX_SCHEMA,
        created_at: "2026-08-16T12:00:01.000Z".into(),
        caller_id: "test".into(),
        operation_id: "memory-drain".into(),
        space: Some("demo".into()),
        agent_name: None,
        reconcile_summary_version: false,
        payload: OutboxPayload::MemoryStore(StoreMemoryRequest {
            content: "outbox replay stores this durable memory".into(),
            memory_type: Some("fact".into()),
            space: WriteSpaceTarget::Named("demo".into()),
            source_agent: None,
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        }),
    };
    envelope_file(&brief, 1);
    envelope_file(&memory, 2);

    let client = reqwest::Client::new();
    let report: OutboxDrainReport = client
        .post(format!("http://127.0.0.1:{port}/api/outbox/drain"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(report.applied, 2, "{report:?}");
    assert_eq!(report.failed, 0, "{report:?}");
    assert!(wenlan_core::config::outbox_dir()
        .read_dir()
        .unwrap()
        .next()
        .is_none());
    assert_eq!(
        db.get_brief_by_space_name("demo")
            .await
            .unwrap()
            .unwrap()
            .version,
        1
    );
    assert!(db
        .has_memory_content("outbox replay stores this durable memory")
        .await
        .unwrap());

    let second: OutboxDrainReport = client
        .post(format!("http://127.0.0.1:{port}/api/outbox/drain"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(second.applied, 0);

    envelope_file(&memory, 3);
    let duplicate: OutboxDrainReport = client
        .post(format!("http://127.0.0.1:{port}/api/outbox/drain"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(duplicate.duplicate, 1, "{duplicate:?}");
    assert_eq!(duplicate.remaining, 0);

    std::fs::write(
        wenlan_core::config::outbox_dir().join("bad.json"),
        b"not json",
    )
    .unwrap();
    let malformed: OutboxDrainReport = client
        .post(format!("http://127.0.0.1:{port}/api/outbox/drain"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(malformed.failed, 1, "{malformed:?}");
    assert!(wenlan_core::config::outbox_dir()
        .join("failed/bad.json")
        .is_file());
    assert!(wenlan_core::config::outbox_dir()
        .join("failed/bad.json.receipt.json")
        .is_file());

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn drain_moves_a_terminal_4xx_rejection_to_failed_with_a_receipt() {
    let _env_lock = data_dir_lock().lock().await;
    let _data = DataDirGuard::new();
    let db_path = wenlan_core::config::data_root().join("memorydb");
    let db = Arc::new(
        MemoryDB::new(&db_path, Arc::new(NoopEmitter))
            .await
            .unwrap(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = Arc::new(RwLock::new(
        ServerState {
            db: Some(Arc::clone(&db)),
            ..ServerState::default()
        }
        .with_bound_port(port),
    ));
    let app = build_router(state);
    let server = tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });

    let rejected = OutboxEnvelope {
        schema: OUTBOX_SCHEMA,
        created_at: "2026-08-16T12:00:02.000Z".into(),
        caller_id: "test".into(),
        operation_id: "quality-gate-reject".into(),
        space: Some("demo".into()),
        agent_name: None,
        reconcile_summary_version: false,
        payload: OutboxPayload::MemoryStore(StoreMemoryRequest {
            content: "too short".into(),
            memory_type: None,
            space: WriteSpaceTarget::Named("demo".into()),
            source_agent: None,
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        }),
    };
    let name = rejected.file_name(1);
    envelope_file(&rejected, 1);

    let client = reqwest::Client::new();
    let report: OutboxDrainReport = client
        .post(format!("http://127.0.0.1:{port}/api/outbox/drain"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(report.applied, 0, "{report:?}");
    assert_eq!(report.failed, 1, "{report:?}");
    assert_eq!(report.remaining, 0, "{report:?}");

    let failed_dir = wenlan_core::config::outbox_dir().join("failed");
    assert!(failed_dir.join(&name).is_file());
    let receipt: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(failed_dir.join(format!("{name}.receipt.json"))).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["http_status"], 422);

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn outbox_status_counts_envelopes_without_counting_receipts() {
    let _env_lock = data_dir_lock().lock().await;
    let _data = DataDirGuard::new();
    let directory = wenlan_core::config::outbox_dir();
    std::fs::create_dir_all(directory.join("failed")).unwrap();
    std::fs::write(directory.join("queued.json"), b"{}").unwrap();
    std::fs::write(directory.join("failed/failed.json"), b"{}").unwrap();
    std::fs::write(directory.join("failed/failed.json.receipt.json"), b"{}").unwrap();

    let state = ServerState::default();
    let response = build_router(Arc::new(RwLock::new(state)))
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/outbox/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response_json(response).await;
    assert_eq!(body["queued"], 1);
    assert_eq!(body["failed"], 1);
}
