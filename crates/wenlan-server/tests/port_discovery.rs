// SPDX-License-Identifier: Apache-2.0
//! Integration test: spawn origin-server with WENLAN_BIND_ADDR=127.0.0.1:0 and verify
//! both port-discovery channels (stdout printline + WENLAN_PORT_FILE) work.

#[path = "common/daemon_binary.rs"]
mod daemon_binary;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const STARTUP_READY_TIMEOUT: Duration = Duration::from_secs(60);

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

#[test]
fn port_discovery_via_stdout() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = prepare_isolated_data_dir(tmp.path());
    let mut child = Command::new(daemon_binary::wenlan_server_binary())
        .env("WENLAN_BIND_ADDR", "127.0.0.1:0")
        .env("WENLAN_DATA_DIR", data_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn origin-server");

    let stdout = child.stdout.take().expect("stdout pipe");
    let mut reader = BufReader::new(stdout);

    let deadline = Instant::now() + STARTUP_READY_TIMEOUT;
    let mut line = String::new();
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for WENLAN_LISTENING_ON line");
        }
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            let _ = child.kill();
            panic!("daemon stdout closed before announcing port");
        }
        if line.starts_with("WENLAN_LISTENING_ON=") {
            let addr = line
                .trim_end()
                .strip_prefix("WENLAN_LISTENING_ON=")
                .unwrap();
            assert!(addr.starts_with("127.0.0.1:"), "bad addr: {}", addr);
            let port_str = addr.split(':').next_back().unwrap();
            let _port: u16 = port_str.parse().expect("port number");
            break;
        }
    }

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn port_discovery_via_port_file() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = prepare_isolated_data_dir(tmp.path());
    let port_file = tmp.path().join("port");
    let mut child = Command::new(daemon_binary::wenlan_server_binary())
        .env("WENLAN_BIND_ADDR", "127.0.0.1:0")
        .env("WENLAN_DATA_DIR", data_dir)
        .env("WENLAN_PORT_FILE", &port_file)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn origin-server");

    let deadline = Instant::now() + STARTUP_READY_TIMEOUT;
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for WENLAN_PORT_FILE");
        }
        if let Ok(contents) = std::fs::read_to_string(&port_file) {
            let _port: u16 = contents.trim().parse().expect("valid port in file");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn port_file_is_published_only_after_startup_preparation() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = prepare_isolated_data_dir(tmp.path());
    let barrier = tmp.path().join("startup-signal-barrier");
    let port_file = tmp.path().join("port");
    std::fs::create_dir_all(&barrier).unwrap();
    let mut child = Command::new(daemon_binary::wenlan_server_binary())
        .env("WENLAN_BIND_ADDR", "127.0.0.1:0")
        .env("WENLAN_DATA_DIR", data_dir)
        .env("WENLAN_PORT_FILE", &port_file)
        .env("WENLAN_TEST_STARTUP_SIGNAL_BARRIER", &barrier)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn origin-server at startup barrier");

    let deadline = Instant::now() + Duration::from_secs(10);
    while !barrier.join("ready").exists() {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for startup signal barrier");
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("daemon exited before startup signal barrier: {status}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !port_file.exists(),
        "daemon must not publish its port before startup preparation completes"
    );

    std::fs::write(barrier.join("release"), b"release").unwrap();
    let deadline = Instant::now() + STARTUP_READY_TIMEOUT;
    loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("timed out waiting for post-preparation WENLAN_PORT_FILE");
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("daemon exited before publishing its port: {status}");
        }
        if let Ok(contents) = std::fs::read_to_string(&port_file) {
            let _port: u16 = contents.trim().parse().expect("valid port in file");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let _ = child.kill();
    let _ = child.wait();
}
