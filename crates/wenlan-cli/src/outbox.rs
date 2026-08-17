// SPDX-License-Identifier: Apache-2.0
//! Offline queue storage for writes that could not reach the daemon.

use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use wenlan_types::OutboxEnvelope;

#[derive(Debug, Clone)]
pub struct OutboxStatus {
    pub queued: Vec<PathBuf>,
    pub failed: Vec<PathBuf>,
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

pub fn enqueue(envelope: &OutboxEnvelope) -> Result<PathBuf> {
    let directory = wenlan_core::config::outbox_dir();
    fs::create_dir_all(&directory)
        .with_context(|| format!("creating outbox directory {}", directory.display()))?;

    let unix_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis();
    let name = envelope.file_name(unix_millis);
    let path = directory.join(&name);
    let temporary = directory.join(format!(".tmp-{name}"));
    let json = serde_json::to_vec_pretty(envelope).context("serializing outbox envelope")?;

    fs::write(&temporary, json)
        .with_context(|| format!("writing temporary outbox file {}", temporary.display()))?;
    if let Err(error) = fs::rename(&temporary, &path) {
        let _ = fs::remove_file(&temporary);
        return Err(error).with_context(|| format!("renaming outbox file {}", path.display()));
    }
    Ok(path)
}

pub fn status() -> Result<OutboxStatus> {
    let directory = wenlan_core::config::outbox_dir();
    Ok(OutboxStatus {
        queued: json_files(&directory)?,
        failed: json_files(&directory.join("failed"))?,
    })
}

fn json_files(directory: &std::path::Path) -> Result<Vec<PathBuf>> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Ok(Vec::new());
    };
    let mut files = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
                && !path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().ends_with(".receipt.json"))
                && !path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with(".tmp-"))
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    Ok(files)
}

pub fn is_daemon_unreachable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .is_some_and(reqwest::Error::is_connect)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    use tempfile::TempDir;
    use wenlan_types::{BriefUpdateRequest, OutboxPayload};

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    fn enqueue_writes_atomically_without_a_temporary_file() {
        let _guard = env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let previous = std::env::var_os("WENLAN_DATA_DIR");
        std::env::set_var("WENLAN_DATA_DIR", temp.path());
        let envelope = OutboxEnvelope {
            schema: wenlan_types::OUTBOX_SCHEMA,
            created_at: now_rfc3339(),
            caller_id: "wenlan-cli".into(),
            operation_id: "atomic-test".into(),
            space: None,
            agent_name: None,
            reconcile_summary_version: false,
            payload: OutboxPayload::BriefUpdate(BriefUpdateRequest {
                space: "demo".into(),
                caller_id: "wenlan-cli".into(),
                operation_id: "atomic-test".into(),
                summary: None,
                mutations: vec![],
            }),
        };

        let path = enqueue(&envelope).unwrap();
        assert!(path.is_file());
        let entries = fs::read_dir(temp.path().join("outbox"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert!(entries
            .iter()
            .all(|name| !name.to_string_lossy().starts_with(".tmp-")));

        match previous {
            Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
            None => std::env::remove_var("WENLAN_DATA_DIR"),
        }
    }

    #[test]
    fn json_files_skips_tmp_and_receipt_files() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("x.json"), b"{}").unwrap();
        std::fs::write(temp.path().join(".tmp-x.json"), b"{}").unwrap();
        std::fs::write(temp.path().join("x.json.receipt.json"), b"{}").unwrap();

        let files = json_files(temp.path()).unwrap();

        assert_eq!(files, vec![temp.path().join("x.json")]);
    }

    #[tokio::test]
    async fn connect_errors_are_detected_through_anyhow_context() {
        let client = reqwest::Client::new();
        let error = client.get("http://127.0.0.1:9").send().await.unwrap_err();
        let wrapped = anyhow::Error::new(error).context("closed daemon");
        assert!(is_daemon_unreachable(&wrapped));
    }
}
