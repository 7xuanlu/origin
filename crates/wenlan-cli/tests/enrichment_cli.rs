// SPDX-License-Identifier: Apache-2.0
//! Cross-surface consent contract for the Wenlan CLI.

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;

fn cli(base: &str) -> Command {
    let mut cmd = Command::cargo_bin("wenlan").expect("wenlan binary built");
    cmd.env("WENLAN_HOST", base);
    cmd
}

fn response(body: &str) -> String {
    response_with_status("200 OK", body)
}

fn response_with_status(status: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn spawn_stub(responses: Vec<String>) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind consent stub");
    let base = format!("http://{}", listener.local_addr().expect("stub address"));
    let (sent, received) = mpsc::channel();
    thread::spawn(move || {
        for response in responses {
            let (stream, _) = listener.accept().expect("accept consent request");
            let mut reader = BufReader::new(stream);
            let mut request = String::new();
            reader.read_line(&mut request).expect("request line");
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).expect("request header");
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = value.trim().parse().expect("content length");
                }
            }
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body).expect("request body");
            request.push_str(&String::from_utf8(body).expect("utf8 body"));
            sent.send(request).expect("record request");
            reader
                .get_mut()
                .write_all(response.as_bytes())
                .expect("write response");
        }
    });
    (base, received)
}

fn routing(mode_everyday: &str, mode_synthesis: &str) -> String {
    serde_json::json!({
        "everyday": {
            "source": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "mode": mode_everyday,
            "pin": if mode_everyday == "unconfigured" { serde_json::Value::Null } else { serde_json::json!("anthropic") }
        },
        "synthesis": {
            "source": "on_device",
            "model": if mode_synthesis == "pinned" { serde_json::json!("qwen3-4b") } else { serde_json::Value::Null },
            "mode": mode_synthesis,
            "pin": if mode_synthesis == "unconfigured" { serde_json::Value::Null } else { serde_json::json!("on_device") }
        },
        "pool": {
            "anthropic": { "configured": true, "everyday_model": "claude-haiku-4-5-20251001", "synthesis_model": "claude-sonnet-4-6" },
            "external": null,
            "on_device": { "selected": "qwen3-4b", "loaded": mode_synthesis == "pinned", "loading": false }
        }
    })
    .to_string()
}

#[test]
fn status_uses_canonical_ready_paused_off_vocabulary() {
    let (base, requests) = spawn_stub(vec![response(&routing("pinned", "pinned_unavailable"))]);

    cli(&base)
        .args(["enrichment", "status"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Everyday organization: ready [anthropic]",
        ))
        .stdout(predicate::str::contains(
            "Page synthesis: paused (exact source unavailable; no fallback) [on_device]",
        ));

    assert!(requests
        .recv()
        .unwrap()
        .starts_with("GET /api/config/routing "));
}

#[test]
fn configure_discloses_and_writes_the_exact_confirmed_mapping() {
    let (base, requests) = spawn_stub(vec![
        response(&routing("unconfigured", "unconfigured")),
        response("{}"),
        response(&routing("pinned", "pinned")),
    ]);

    cli(&base)
        .args([
            "enrichment",
            "configure",
            "--everyday",
            "anthropic",
            "--synthesis",
            "on-device",
            "--yes",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Everyday organization: Anthropic"))
        .stdout(predicate::str::contains("On-device work uses CPU/GPU/RAM"))
        .stdout(predicate::str::contains(
            "Anthropic receives relevant memory content",
        ));

    assert!(requests
        .recv()
        .unwrap()
        .starts_with("GET /api/config/routing "));
    let put = requests.recv().unwrap();
    assert!(put.starts_with("PUT /api/config "), "{put}");
    assert!(put.contains(r#""everyday_source":"anthropic""#), "{put}");
    assert!(put.contains(r#""synthesis_source":"on_device""#), "{put}");
    assert!(requests
        .recv()
        .unwrap()
        .starts_with("GET /api/config/routing "));
}

#[test]
fn disable_clears_both_pins_without_removing_providers() {
    let (base, requests) = spawn_stub(vec![response("{}")]);

    cli(&base)
        .args(["enrichment", "disable"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Providers and downloaded models were kept",
        ));

    let put = requests.recv().unwrap();
    assert!(put.starts_with("PUT /api/config "), "{put}");
    assert!(put.contains(r#""everyday_source":"""#), "{put}");
    assert!(put.contains(r#""synthesis_source":"""#), "{put}");
}

#[test]
fn disable_surfaces_live_daemon_rejection_without_overwriting_disk() {
    let data = tempfile::tempdir().expect("consent data dir");
    let config_path = data.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"everyday_source":"anthropic","synthesis_source":"on_device"}"#,
    )
    .expect("seed pinned config");
    let (base, requests) = spawn_stub(vec![response_with_status(
        "500 Internal Server Error",
        r#"{"error":"write rejected"}"#,
    )]);

    cli(&base)
        .env("WENLAN_DATA_DIR", data.path())
        .args(["enrichment", "disable"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("HTTP 500"));

    assert!(requests.recv().unwrap().starts_with("PUT /api/config "));
    let config = std::fs::read_to_string(config_path).expect("config remains readable");
    assert!(
        config.contains(r#""everyday_source":"anthropic""#),
        "{config}"
    );
    assert!(
        config.contains(r#""synthesis_source":"on_device""#),
        "{config}"
    );
}

#[test]
fn disable_clears_local_pins_only_when_daemon_is_unreachable() {
    let data = tempfile::tempdir().expect("consent data dir");
    let config_path = data.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"everyday_source":"anthropic","synthesis_source":"on_device"}"#,
    )
    .expect("seed pinned config");
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve unreachable port");
    let base = format!("http://{}", listener.local_addr().expect("stub address"));
    drop(listener);

    cli(&base)
        .env("WENLAN_DATA_DIR", data.path())
        .args(["enrichment", "disable"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Background enrichment disabled in local config",
        ));

    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(config_path).expect("config remains readable"),
    )
    .expect("saved config json");
    assert!(config["everyday_source"].is_null(), "{config}");
    assert!(config["synthesis_source"].is_null(), "{config}");
}
