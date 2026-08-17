// SPDX-License-Identifier: Apache-2.0
//! Daemon-owned replay of CLI outbox envelopes through the normal HTTP routes.

use crate::state::SharedState;
use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wenlan_core::db::MemoryDB;
use wenlan_types::{
    BriefUpdateReceipt, OutboxDrainDetail, OutboxDrainReport, OutboxEnvelope, OutboxPayload,
    OUTBOX_SCHEMA,
};

pub async fn drain_once(state: SharedState) -> OutboxDrainReport {
    let (lock, port, db) = {
        let state = state.read().await;
        (
            Arc::clone(&state.outbox_drain_lock),
            state.bound_port,
            state.db.clone(),
        )
    };
    let Some(_drain_guard) = lock.try_lock().ok() else {
        let remaining = queued_files().map(|files| files.len() as u32).unwrap_or(0);
        return OutboxDrainReport {
            remaining,
            details: vec![OutboxDrainDetail {
                file: String::new(),
                kind: "drain".into(),
                outcome: "failed".into(),
                error: Some("another outbox drain is already running".into()),
            }],
            ..OutboxDrainReport::default()
        };
    };

    let files = match queued_files() {
        Ok(files) => files,
        Err(error) => {
            return OutboxDrainReport {
                failed: 1,
                details: vec![OutboxDrainDetail {
                    file: String::new(),
                    kind: "drain".into(),
                    outcome: "failed".into(),
                    error: Some(error.to_string()),
                }],
                ..OutboxDrainReport::default()
            }
        }
    };
    if files.is_empty() {
        return OutboxDrainReport::default();
    }

    let client = reqwest::Client::new();
    let mut report = OutboxDrainReport::default();
    for path in files {
        let file = path.display().to_string();
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.clone());
        let envelope = match fs::read(&path)
            .with_context(|| format!("read {file}"))
            .and_then(|bytes| serde_json::from_slice::<OutboxEnvelope>(&bytes).context("parse"))
        {
            Ok(envelope) if envelope.schema == OUTBOX_SCHEMA => envelope,
            Ok(envelope) => {
                let error = format!("unsupported outbox schema {}", envelope.schema);
                fail_malformed(&path, &name, &error, &mut report);
                continue;
            }
            Err(error) => {
                fail_malformed(&path, &name, &error.to_string(), &mut report);
                continue;
            }
        };

        let kind = envelope.payload.kind().to_string();
        match replay(&client, port, db.as_ref(), &envelope).await {
            Ok(ReplayOutcome::Applied { receipt }) => {
                if let Some(receipt) = receipt.filter(|receipt| !receipt.conflicts.is_empty()) {
                    let error = format!(
                        "brief receipt contains {} conflict(s)",
                        receipt.conflicts.len()
                    );
                    let receipt_value = serde_json::to_value(&receipt).unwrap_or_else(
                        |_| serde_json::json!({"error": "failed to serialize receipt"}),
                    );
                    let move_error = move_to_failed(&path, &name, &receipt_value).err();
                    report.failed += 1;
                    report.details.push(detail(
                        &file,
                        &kind,
                        "failed",
                        Some(match move_error {
                            Some(move_error) => format!("{error}; move to failed: {move_error}"),
                            None => error,
                        }),
                    ));
                } else if let Err(error) = fs::remove_file(&path) {
                    report.failed += 1;
                    report.details.push(detail(
                        &file,
                        &kind,
                        "failed",
                        Some(format!("delete after apply: {error}")),
                    ));
                } else {
                    report.applied += 1;
                    report.details.push(detail(&file, &kind, "applied", None));
                }
            }
            Ok(ReplayOutcome::Duplicate) => {
                if let Err(error) = fs::remove_file(&path) {
                    report.failed += 1;
                    report.details.push(detail(
                        &file,
                        &kind,
                        "failed",
                        Some(format!("delete duplicate: {error}")),
                    ));
                } else {
                    report.duplicate += 1;
                    report.details.push(detail(&file, &kind, "duplicate", None));
                }
            }
            Err(error) => {
                report.failed += 1;
                report
                    .details
                    .push(detail(&file, &kind, "failed", Some(error.to_string())));
            }
        }
    }

    report.remaining = queued_files().map(|files| files.len() as u32).unwrap_or(0);
    tracing::info!(
        applied = report.applied,
        duplicate = report.duplicate,
        failed = report.failed,
        remaining = report.remaining,
        "outbox drain complete"
    );
    report
}

pub fn queued_files() -> Result<Vec<PathBuf>> {
    json_files(&wenlan_core::config::outbox_dir())
}

pub fn failed_files() -> Result<Vec<PathBuf>> {
    json_files(&wenlan_core::config::outbox_dir().join("failed"))
}

fn json_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Ok(Vec::new());
    };
    let mut files = entries
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
                && !path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().ends_with(".receipt.json"))
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    Ok(files)
}

enum ReplayOutcome {
    Applied { receipt: Option<BriefUpdateReceipt> },
    Duplicate,
}

async fn replay(
    client: &reqwest::Client,
    port: u16,
    db: Option<&Arc<MemoryDB>>,
    envelope: &OutboxEnvelope,
) -> Result<ReplayOutcome> {
    if port == 0 {
        anyhow::bail!("daemon bound port is not initialized")
    }
    let base_url = format!("http://127.0.0.1:{port}");
    let mut request = match &envelope.payload {
        OutboxPayload::BriefUpdate(request) => {
            let mut request = request.clone();
            if envelope.reconcile_summary_version && request.summary.is_some() {
                let db = db.context("database is not initialized")?;
                let current_version = db
                    .get_brief_by_space_name(&request.space)
                    .await
                    .context("read current Brief version")?
                    .map(|brief| brief.version)
                    .unwrap_or(0);
                if let Some(summary) = request.summary.as_mut() {
                    summary.expected_version = current_version;
                }
            }
            client.patch(format!("{base_url}/api/brief")).json(&request)
        }
        OutboxPayload::MemoryStore(request) => client
            .post(format!("{base_url}/api/memory/store"))
            .json(request),
    };
    if let Some(space) = envelope.space.as_deref() {
        request = request.header("x-wenlan-space", space);
    }
    if let Some(agent_name) = envelope.agent_name.as_deref() {
        request = request.header("x-agent-name", agent_name);
    }
    let response = request.send().await.context("loopback replay request")?;
    let status = response.status();
    let body = response.text().await.context("read replay response")?;
    if status.is_success() {
        if matches!(&envelope.payload, OutboxPayload::BriefUpdate(_)) {
            let receipt: BriefUpdateReceipt =
                serde_json::from_str(&body).context("parse Brief replay receipt")?;
            return Ok(ReplayOutcome::Applied {
                receipt: Some(receipt),
            });
        }
        return Ok(ReplayOutcome::Applied { receipt: None });
    }
    if status.as_u16() == 422 && matches!(&envelope.payload, OutboxPayload::MemoryStore(_)) {
        let body_json: Value = serde_json::from_str(&body).unwrap_or_default();
        let reason = body_json.get("reason").and_then(Value::as_str);
        let exact_duplicate = body_json
            .get("error")
            .and_then(Value::as_str)
            .is_some_and(|error| error.starts_with("Duplicate:"));
        if reason == Some("not_novel") || exact_duplicate {
            return Ok(ReplayOutcome::Duplicate);
        }
    }
    anyhow::bail!("loopback replay returned {status}: {body}")
}

fn detail(file: &str, kind: &str, outcome: &str, error: Option<String>) -> OutboxDrainDetail {
    OutboxDrainDetail {
        file: file.into(),
        kind: kind.into(),
        outcome: outcome.into(),
        error,
    }
}

fn fail_malformed(path: &Path, name: &str, error: &str, report: &mut OutboxDrainReport) {
    let receipt = serde_json::json!({"status": "failed", "file": name, "error": error});
    let move_error = move_to_failed(path, name, &receipt).err();
    report.failed += 1;
    report.details.push(detail(
        &path.display().to_string(),
        "unknown",
        "failed",
        Some(match move_error {
            Some(move_error) => format!("{error}; move to failed: {move_error}"),
            None => error.to_string(),
        }),
    ));
}

fn move_to_failed(path: &Path, name: &str, receipt: &Value) -> Result<()> {
    let failed = wenlan_core::config::outbox_dir().join("failed");
    fs::create_dir_all(&failed).context("create outbox failed directory")?;
    fs::rename(path, failed.join(name)).context("move outbox file to failed")?;
    fs::write(
        failed.join(format!("{name}.receipt.json")),
        serde_json::to_vec_pretty(receipt).context("serialize outbox failure receipt")?,
    )
    .context("write outbox failure receipt")?;
    Ok(())
}
