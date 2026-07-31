// SPDX-License-Identifier: Apache-2.0
//! Process-level proof for graceful and bounded daemon shutdown.

use axum::{routing::post, Json, Router};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct RunningDaemon {
    child: ChildGuard,
    port: u16,
    _tempdir: tempfile::TempDir,
}

fn binary_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_wenlan-server"))
}

fn prepare_isolated_data_dir(root: &Path) -> PathBuf {
    let data_dir = root.join("data");
    let pages_dir = root.join("pages");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&pages_dir).unwrap();
    std::fs::write(
        data_dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "knowledge_path": pages_dir,
        }))
        .unwrap(),
    )
    .unwrap();
    data_dir
}

async fn start_mock_provider(
    delay: Duration,
) -> (SocketAddr, Arc<Notify>, tokio::task::JoinHandle<()>) {
    let request_started = Arc::new(Notify::new());
    let started_for_handler = request_started.clone();
    let mock = Router::new().route(
        "/chat/completions",
        post(move || {
            let started = started_for_handler.clone();
            async move {
                started.notify_one();
                tokio::time::sleep(delay).await;
                Json(serde_json::json!({
                    "choices": [{"message": {"content": "hello"}}]
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, mock).await.unwrap();
    });
    (addr, request_started, task)
}

async fn start_daemon() -> RunningDaemon {
    let tempdir = tempfile::tempdir().unwrap();
    let data_dir = prepare_isolated_data_dir(tempdir.path());
    let port_file = tempdir.path().join("port");
    let child = Command::new(binary_path())
        .env("WENLAN_BIND_ADDR", "127.0.0.1:0")
        // Keep the child isolated from both durable user config and data. The
        // reranker overrides below are safe only while this tempdir-scoped
        // WENLAN_DATA_DIR remains part of the process-test contract.
        .env("WENLAN_DATA_DIR", data_dir)
        .env("WENLAN_PORT_FILE", &port_file)
        // Parent developer settings must never make this process test download
        // or eagerly load a reranker before announcing its port.
        .env_remove("WENLAN_RERANKER_MODE")
        .env_remove("WENLAN_RERANKER_ENABLED")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wenlan-server");
    let mut child = ChildGuard(child);

    let port = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                panic!("daemon exited before announcing its port: {status}");
            }
            if let Ok(contents) = std::fs::read_to_string(&port_file) {
                break contents.trim().parse::<u16>().expect("valid daemon port");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("daemon port discovery timed out");

    RunningDaemon {
        child,
        port,
        _tempdir: tempdir,
    }
}

fn start_llm_request(
    client: &reqwest::Client,
    daemon_port: u16,
    provider_addr: SocketAddr,
) -> tokio::task::JoinHandle<Result<reqwest::Response, reqwest::Error>> {
    let client = client.clone();
    tokio::spawn(async move {
        client
            .post(format!("http://127.0.0.1:{daemon_port}/api/llm/test"))
            .json(&serde_json::json!({
                "endpoint": format!("http://{provider_addr}"),
                "model": "test-model",
                "prompt": "hello"
            }))
            .send()
            .await
    })
}

async fn wait_for_exit(child: &mut ChildGuard, bound: Duration) -> ExitStatus {
    tokio::time::timeout(bound, async {
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("daemon did not exit within the shutdown bound")
}

async fn assert_successful_llm_response(
    request: tokio::task::JoinHandle<Result<reqwest::Response, reqwest::Error>>,
) {
    let response = tokio::time::timeout(Duration::from_secs(3), request)
        .await
        .expect("in-flight request did not drain")
        .expect("LLM request task panicked")
        .expect("daemon exited before the in-flight response completed");
    assert!(response.status().is_success());
}

#[tokio::test]
async fn shutdown_drains_completable_request_before_process_exit() {
    let (mock_addr, request_started, mock_task) =
        start_mock_provider(Duration::from_millis(350)).await;
    let mut daemon = start_daemon().await;
    let client = reqwest::Client::new();
    let llm_request = start_llm_request(&client, daemon.port, mock_addr);

    tokio::time::timeout(Duration::from_secs(5), request_started.notified())
        .await
        .expect("daemon never started the in-flight LLM request");

    let shutdown = client
        .post(format!("http://127.0.0.1:{}/api/shutdown", daemon.port))
        .send()
        .await
        .expect("shutdown response");
    assert!(shutdown.status().is_success());
    assert_eq!(
        shutdown
            .headers()
            .get("x-wenlan-process-id")
            .and_then(|value| value.to_str().ok()),
        Some(daemon.child.0.id().to_string().as_str()),
        "shutdown response must identify the exact daemon process being stopped"
    );
    assert_eq!(shutdown.text().await.unwrap(), "shutting down");

    assert_successful_llm_response(llm_request).await;
    let status = wait_for_exit(&mut daemon.child, Duration::from_secs(5)).await;
    assert!(status.success(), "requested shutdown status: {status}");
    mock_task.abort();
}

#[tokio::test]
async fn shutdown_forces_exit_when_request_exceeds_drain_deadline() {
    let (mock_addr, request_started, mock_task) =
        start_mock_provider(Duration::from_secs(10)).await;
    let mut daemon = start_daemon().await;
    let client = reqwest::Client::new();
    let llm_request = start_llm_request(&client, daemon.port, mock_addr);

    tokio::time::timeout(Duration::from_secs(5), request_started.notified())
        .await
        .expect("daemon never started the long in-flight request");
    let started = Instant::now();
    let response = client
        .post(format!("http://127.0.0.1:{}/api/shutdown", daemon.port))
        .send()
        .await
        .expect("shutdown response");
    assert!(response.status().is_success());

    let status = wait_for_exit(&mut daemon.child, Duration::from_secs(4)).await;
    assert!(status.success(), "forced shutdown status: {status}");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "shutdown exceeded the 1.5s drain deadline: {:?}",
        started.elapsed()
    );
    llm_request.abort();
    mock_task.abort();
}

#[cfg(unix)]
#[tokio::test]
async fn sigterm_uses_the_same_graceful_drain_path() {
    let (mock_addr, request_started, mock_task) =
        start_mock_provider(Duration::from_millis(350)).await;
    let mut daemon = start_daemon().await;
    let client = reqwest::Client::new();
    let llm_request = start_llm_request(&client, daemon.port, mock_addr);

    tokio::time::timeout(Duration::from_secs(5), request_started.notified())
        .await
        .expect("daemon never started the SIGTERM in-flight request");
    let signal_status = Command::new("kill")
        .arg("-TERM")
        .arg(daemon.child.0.id().to_string())
        .status()
        .expect("invoke kill -TERM");
    assert!(signal_status.success());

    assert_successful_llm_response(llm_request).await;
    let status = wait_for_exit(&mut daemon.child, Duration::from_secs(5)).await;
    assert!(status.success(), "SIGTERM shutdown status: {status}");
    mock_task.abort();
}

#[cfg(unix)]
#[tokio::test]
async fn sigterm_after_bind_during_startup_uses_cooperative_shutdown() {
    let tempdir = tempfile::tempdir().unwrap();
    let data_dir = prepare_isolated_data_dir(tempdir.path());
    let barrier = tempdir.path().join("startup-signal-barrier");
    std::fs::create_dir_all(&barrier).unwrap();
    let child = Command::new(binary_path())
        .env("WENLAN_BIND_ADDR", "127.0.0.1:0")
        // Preserve the same isolated-config invariant as start_daemon(); the
        // removed parent reranker settings must not expose real user state.
        .env("WENLAN_DATA_DIR", data_dir)
        .env("WENLAN_TEST_STARTUP_SIGNAL_BARRIER", &barrier)
        .env_remove("WENLAN_RERANKER_MODE")
        .env_remove("WENLAN_RERANKER_ENABLED")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wenlan-server at startup barrier");
    let mut child = ChildGuard(child);

    tokio::time::timeout(Duration::from_secs(10), async {
        while !barrier.join("ready").exists() {
            if let Some(status) = child.0.try_wait().unwrap() {
                panic!("daemon exited before startup signal barrier: {status}");
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("startup signal barrier was never reached");

    let signal_status = Command::new("kill")
        .arg("-TERM")
        .arg(child.0.id().to_string())
        .status()
        .expect("invoke startup kill -TERM");
    assert!(signal_status.success());
    std::fs::write(barrier.join("release"), b"release").unwrap();

    let status = wait_for_exit(&mut child, Duration::from_secs(30)).await;
    assert!(
        status.success(),
        "startup SIGTERM bypassed cooperative shutdown: {status}"
    );
}
