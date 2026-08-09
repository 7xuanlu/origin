// SPDX-License-Identifier: Apache-2.0

use assert_cmd::Command;
use predicates::prelude::*;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    sync::{mpsc, Arc, Mutex},
    thread,
};

#[derive(Debug)]
struct Recorded {
    method: String,
    path: String,
    body: String,
}

fn cli() -> Command {
    Command::cargo_bin("wenlan").unwrap()
}

fn response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn spawn_stub(responses: Vec<String>) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let host = format!("http://{}", listener.local_addr().unwrap());
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let recorded_for_thread = Arc::clone(&recorded);
    thread::spawn(move || {
        for response in responses {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap().to_string();
            let path = parts.next().unwrap().to_string();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                if let Some(value) = line
                    .strip_prefix("content-length:")
                    .or_else(|| line.strip_prefix("Content-Length:"))
                {
                    content_length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; content_length];
            reader.read_exact(&mut body).unwrap();
            recorded_for_thread.lock().unwrap().push(Recorded {
                method,
                path,
                body: String::from_utf8(body).unwrap(),
            });
            reader.get_mut().write_all(response.as_bytes()).unwrap();
        }
    });
    (host, recorded)
}

fn space_list() -> String {
    serde_json::json!([{
        "id": "space-id",
        "name": "work",
        "description": null,
        "suggested": false,
        "created_at": 1.0,
        "updated_at": 1.0,
        "memory_count": 0,
        "entity_count": 0,
        "sort_order": 0,
        "starred": false,
        "is_default": false
    }])
    .to_string()
}

fn brief_response() -> String {
    serde_json::json!({
        "state": "ready",
        "space": "work",
        "brief": {
            "space_id": "space-machine-id",
            "space": "work",
            "last_session_summary": "finished daemon routes",
            "last_handoff_at": 1,
            "version": 3,
            "active": [{
                "id": "item-machine-id",
                "text": "open the PR",
                "state": "active",
                "added_at": 1,
                "version": 2
            }],
            "backlog": []
        }
    })
    .to_string()
}

#[test]
fn brief_read_posts_topic_and_json_keeps_machine_ids() {
    let (host, recorded) = spawn_stub(vec![response(&space_list()), response(&brief_response())]);

    cli()
        .env("WENLAN_HOST", host)
        .args([
            "--space",
            "work",
            "--format",
            "json",
            "brief",
            "release gate",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("item-machine-id"))
        .stdout(predicate::str::contains("space-machine-id"));

    let recorded = recorded.lock().unwrap();
    assert_eq!(recorded.len(), 2);
    assert_eq!(
        (recorded[1].method.as_str(), recorded[1].path.as_str()),
        ("POST", "/api/brief")
    );
    let body: serde_json::Value = serde_json::from_str(&recorded[1].body).unwrap();
    assert_eq!(body["topic"], "release gate");
}

#[test]
fn brief_human_output_omits_machine_ids() {
    let (host, _recorded) = spawn_stub(vec![response(&space_list()), response(&brief_response())]);

    cli()
        .env("WENLAN_HOST", host)
        .args(["--space", "work", "--format", "table", "brief"])
        .assert()
        .success()
        .stdout(predicate::str::contains("open the PR"))
        .stdout(predicate::str::contains("item-machine-id").not())
        .stdout(predicate::str::contains("space-machine-id").not());
}

#[test]
fn brief_update_uses_patch_with_the_typed_file() {
    let temp = tempfile::tempdir().unwrap();
    let update = temp.path().join("update.json");
    std::fs::write(
        &update,
        serde_json::json!({
            "space": "work",
            "caller_id": "handoff",
            "operation_id": "operation-1",
            "mutations": [{
                "kind": "add",
                "text": "publish PR",
                "state": "active"
            }]
        })
        .to_string(),
    )
    .unwrap();
    let receipt = serde_json::json!({
        "space": "work",
        "brief_version": 1,
        "applied": [],
        "conflicts": [],
        "projection_path": "/tmp/work.md",
        "warnings": []
    })
    .to_string();
    let (host, recorded) = spawn_stub(vec![response(&receipt)]);

    cli()
        .env("WENLAN_HOST", host)
        .args([
            "--space",
            "work",
            "brief",
            "update",
            "--file",
            update.to_str().unwrap(),
        ])
        .assert()
        .success();

    let recorded = recorded.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        (recorded[0].method.as_str(), recorded[0].path.as_str()),
        ("PATCH", "/api/brief")
    );
    let body: serde_json::Value = serde_json::from_str(&recorded[0].body).unwrap();
    assert_eq!(body["operation_id"], "operation-1");
}

#[test]
fn invalid_update_fails_before_any_network_request() {
    let temp = tempfile::tempdir().unwrap();
    let update = temp.path().join("invalid.json");
    std::fs::write(&update, "{not json").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let host = format!("http://{}", listener.local_addr().unwrap());

    cli()
        .env("WENLAN_HOST", host)
        .args(["brief", "update", "--file", update.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid Brief update JSON"));

    let (sent, received) = mpsc::channel();
    thread::spawn(move || {
        thread::sleep(std::time::Duration::from_millis(100));
        sent.send(listener.accept().is_ok()).unwrap();
    });
    assert!(!received.recv().unwrap(), "invalid JSON sent a request");
}

#[test]
fn recall_uses_memory_search_instead_of_legacy_context() {
    let search = serde_json::json!({
        "results": [],
        "took_ms": 1.0
    })
    .to_string();
    let (host, recorded) = spawn_stub(vec![response(&space_list()), response(&search)]);

    cli()
        .env("WENLAN_HOST", host)
        .args(["--space", "work", "recall", "release gate"])
        .assert()
        .success();

    let recorded = recorded.lock().unwrap();
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[1].path, "/api/memory/search");
    assert!(recorded
        .iter()
        .all(|request| request.path != "/api/context"));
}
