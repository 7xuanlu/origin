// SPDX-License-Identifier: Apache-2.0
//! Integration tests for `wenlan sources add <path>`.
//!
//! The CLI is a thin HTTP client: `sources add` must POST /api/sources to
//! register a Directory source, then POST /api/sources/{id}/sync, then render the stats.
//! These tests drive the real `wenlan` binary against a tiny in-process stub
//! daemon (there is no HTTP-mock crate in dev-deps) and assert on the recorded
//! requests plus rendered output.

use assert_cmd::Command;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

fn cli() -> Command {
    let mut command = Command::cargo_bin("wenlan").expect("wenlan binary built");
    command.env("WENLAN_NO_AUTOSTART", "1");
    command
}

#[derive(Clone, Debug)]
struct Recorded {
    method: String,
    path: String,
    #[allow(dead_code)]
    body: String,
}

/// Spawn a one-shot stub HTTP/1.1 server that answers the next request with the
/// next canned response body (200 OK, JSON). Returns the base URL and a handle
/// to the recorded requests. Threads are detached — fine for a test process.
fn spawn_stub(responses: Vec<String>) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
    let base = format!("http://{}", listener.local_addr().unwrap());
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    let recorded: Arc<Mutex<Vec<Recorded>>> = Arc::new(Mutex::new(Vec::new()));

    let responses_outer = responses.clone();
    let recorded_outer = recorded.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            // One thread per connection so a reqwest connection-pool reuse OR a
            // fresh connection are both handled without the accept loop blocking.
            let responses = responses_outer.clone();
            let recorded = recorded_outer.clone();
            thread::spawn(move || {
                let mut reader = BufReader::new(stream);
                loop {
                    let mut request_line = String::new();
                    match reader.read_line(&mut request_line) {
                        Ok(0) | Err(_) => break, // connection closed
                        Ok(_) => {}
                    }
                    let mut it = request_line.split_whitespace();
                    let method = it.next().unwrap_or("").to_string();
                    let path = it.next().unwrap_or("").to_string();

                    let mut content_length = 0usize;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 {
                            break;
                        }
                        if line == "\r\n" || line == "\n" {
                            break;
                        }
                        let lower = line.to_ascii_lowercase();
                        if let Some(rest) = lower.strip_prefix("content-length:") {
                            content_length = rest.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut body = vec![0u8; content_length];
                    if content_length > 0 && reader.read_exact(&mut body).is_err() {
                        break;
                    }

                    recorded.lock().unwrap().push(Recorded {
                        method,
                        path,
                        body: String::from_utf8_lossy(&body).to_string(),
                    });

                    let resp_body = responses.lock().unwrap().pop_front();
                    let (status, resp_body) = match resp_body {
                        Some(b) => ("200 OK", b),
                        None => ("500 Internal Server Error", "{}".to_string()),
                    };
                    let resp = format!(
                        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        status,
                        resp_body.len(),
                        resp_body
                    );
                    let stream = reader.get_mut();
                    if stream.write_all(resp.as_bytes()).is_err() {
                        break;
                    }
                    let _ = stream.flush();
                }
            });
        }
    });

    (base, recorded)
}

fn source_json(id: &str, path: &str) -> String {
    // json! (not format!) so Windows backslash paths are escaped as valid JSON.
    serde_json::json!({
        "id": id,
        "source_type": "directory",
        "path": path,
        "status": "Active",
        "last_sync": null,
        "file_count": 0,
        "memory_count": 0,
        "last_sync_errors": 0,
        "last_sync_error_detail": null,
    })
    .to_string()
}

fn stats_json(found: usize, ingested: usize, skipped: usize, errors: usize) -> String {
    format!(
        r#"{{"files_found":{found},"ingested":{ingested},"skipped":{skipped},"errors":{errors}}}"#
    )
}

#[test]
fn ingest_registers_source_then_syncs_and_renders_stats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let abs = std::fs::canonicalize(dir.path()).expect("canonicalize");
    let abs_str = abs.to_string_lossy().to_string();

    let (base, recorded) = spawn_stub(vec![
        source_json("directory-notes", &abs_str),
        stats_json(3, 2, 1, 0),
    ]);

    cli()
        .env("WENLAN_HOST", &base)
        .args(["sources", "add", dir.path().to_str().unwrap()])
        .assert()
        .success()
        // Default (piped stdout) format is JSON — the stats must render.
        .stdout(predicates::str::contains("\"ingested\": 2"))
        .stdout(predicates::str::contains("\"files_found\": 3"));

    let reqs = recorded.lock().unwrap().clone();
    assert!(
        reqs.iter()
            .any(|r| r.method == "POST" && r.path == "/api/sources"),
        "expected POST /api/sources, got {:?}",
        reqs.iter()
            .map(|r| (&r.method, &r.path))
            .collect::<Vec<_>>()
    );
    assert!(
        reqs.iter()
            .any(|r| r.method == "POST" && r.path == "/api/sources/directory-notes/sync"),
        "expected POST /api/sources/{{id}}/sync, got {:?}",
        reqs.iter()
            .map(|r| (&r.method, &r.path))
            .collect::<Vec<_>>()
    );
    // Ordering: registration must precede sync.
    let reg = reqs.iter().position(|r| r.path == "/api/sources").unwrap();
    let sync = reqs.iter().position(|r| r.path.ends_with("/sync")).unwrap();
    assert!(reg < sync, "register must precede sync: {reqs:?}");
}

#[test]
fn ingest_accepts_a_single_file_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("note.md");
    std::fs::write(&file, "# hi").expect("write file");
    let abs = std::fs::canonicalize(&file).expect("canonicalize");
    let abs_str = abs.to_string_lossy().to_string();

    let (base, recorded) = spawn_stub(vec![
        source_json("directory-note-md", &abs_str),
        stats_json(1, 1, 0, 0),
    ]);

    cli()
        .env("WENLAN_HOST", &base)
        .args(["sources", "add", file.to_str().unwrap()])
        .assert()
        .success();

    let reqs = recorded.lock().unwrap().clone();
    assert!(
        reqs.iter()
            .any(|r| r.method == "POST" && r.path == "/api/sources"),
        "single-file ingest must still POST /api/sources, got {:?}",
        reqs.iter()
            .map(|r| (&r.method, &r.path))
            .collect::<Vec<_>>()
    );
}

#[test]
fn ingest_table_format_renders_human_summary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let abs = std::fs::canonicalize(dir.path()).expect("canonicalize");
    let abs_str = abs.to_string_lossy().to_string();

    let (base, _recorded) = spawn_stub(vec![
        source_json("directory-notes", &abs_str),
        stats_json(5, 4, 1, 0),
    ]);

    cli()
        .env("WENLAN_HOST", &base)
        .args([
            "sources",
            "add",
            dir.path().to_str().unwrap(),
            "--format",
            "table",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("file(s) found"));
}

#[test]
fn sources_add_help_exits_zero() {
    cli().args(["sources", "add", "--help"]).assert().success();
}
