// SPDX-License-Identifier: Apache-2.0
//! Integration test: when the port is already held by a healthy daemon, the
//! server must bail out BEFORE initializing MemoryDB. A launchd KeepAlive
//! retry loop otherwise re-runs full DB init (schema writes + embedder load)
//! against the live daemon's SQLite file every ~10s — enough lock/CPU
//! pressure to wedge the daemon that actually owns the port.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[path = "common/daemon_binary.rs"]
mod daemon_binary;

/// Minimal "healthy daemon": answers HTTP 200 to every request on an
/// ephemeral port.
fn spawn_fake_healthy_daemon() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0_u8; 1024];
            let _ = stream.read(&mut buf);
            let body = r#"{"status":"ok","db_initialized":true,"version":"0.0.0-test"}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    port
}

/// Holds a port but never answers HTTP: connections queue in the accept
/// backlog and get no response. The server must error out ("no healthy
/// daemon"), not hang forever on its health probe.
#[test]
fn port_taken_by_mute_listener_errors_instead_of_hanging() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    // Keep the listener alive but never accept/respond.

    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");

    let mut child = Command::new(daemon_binary::wenlan_server_binary())
        .env("WENLAN_PORT", port.to_string())
        .env_remove("WENLAN_BIND_ADDR")
        .env_remove("XPC_SERVICE_NAME")
        .env("WENLAN_DATA_DIR", &data_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wenlan-server");

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("server hung on the health probe against a mute port-holder");
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    assert!(
        !status.success(),
        "expected error exit (port in use, no healthy daemon), got {status:?}"
    );
    assert!(
        !data_dir.join("memorydb").exists(),
        "MemoryDB was initialized even though the port was already taken"
    );
    drop(listener);
}

#[test]
fn port_taken_by_healthy_daemon_exits_before_db_init() {
    let port = spawn_fake_healthy_daemon();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");

    let mut child = Command::new(daemon_binary::wenlan_server_binary())
        .env("WENLAN_PORT", port.to_string())
        .env_remove("WENLAN_BIND_ADDR")
        // Non-launchd path: a healthy daemon on the port means clean exit 0.
        .env_remove("XPC_SERVICE_NAME")
        .env("WENLAN_DATA_DIR", &data_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wenlan-server");

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("server did not exit; it should bail when the port is taken");
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    assert!(status.success(), "expected clean exit, got {status:?}");
    assert!(
        !data_dir.join("memorydb").exists(),
        "MemoryDB was initialized even though the port was already taken"
    );
}
