// SPDX-License-Identifier: Apache-2.0
//! `wenlan capture [text] [--file <path>] [--type <type>]` — POST /api/memory/store.

use anyhow::{Context, Result};
use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use wenlan_types::{OutboxEnvelope, OutboxPayload, OUTBOX_SCHEMA};

use crate::client::WenlanClient;
use crate::outbox;
use crate::output::{print_json, ResolvedFormat};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    text: Option<String>,
    file: Option<PathBuf>,
    memory_type: Option<String>,
    effective_space: Option<String>,
    agent_name: Option<&str>,
) -> Result<()> {
    let content = match (text, file) {
        (Some(t), None) => t,
        (None, Some(p)) => {
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?
        }
        (None, None) => {
            // Reject interactive TTY stdin — would block indefinitely waiting for EOF.
            // User must provide text, --file, or pipe content.
            if std::io::stdin().is_terminal() {
                anyhow::bail!("No input. Provide text, --file <path>, or pipe content via stdin.");
            }
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .context("reading stdin")?;
            s
        }
        (Some(_), Some(_)) => {
            anyhow::bail!("Provide either positional text or --file, not both");
        }
    };
    let content = content.trim().to_string();
    if content.is_empty() {
        anyhow::bail!("Empty content. Provide text, --file, or pipe content via stdin.");
    }
    let request = WenlanClient::store_request(content.clone(), memory_type.clone());
    let resp = match client.store(content, memory_type).await {
        Ok(response) => response,
        Err(error) if client.is_local() && outbox::is_daemon_unreachable(&error) => {
            let operation_id = uuid::Uuid::new_v4().to_string();
            let envelope = OutboxEnvelope {
                schema: OUTBOX_SCHEMA,
                created_at: outbox::now_rfc3339(),
                caller_id: "wenlan-cli".into(),
                operation_id: operation_id.clone(),
                space: effective_space,
                agent_name: agent_name.map(str::to_owned),
                reconcile_summary_version: false,
                payload: OutboxPayload::MemoryStore(request),
            };
            let path = outbox::enqueue(&envelope)?;
            if !quiet {
                match format {
                    ResolvedFormat::Json => print_json(&serde_json::json!({
                        "status": "queued",
                        "path": path.display().to_string(),
                        "operation_id": operation_id,
                    }))?,
                    ResolvedFormat::Table => println!("queued: {}", path.display()),
                }
            }
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&resp)?,
        ResolvedFormat::Table => {
            let destination = resp.space.as_deref().unwrap_or("Uncategorized");
            println!(
                "Stored memory {} in {} ({} chunk(s))",
                resp.source_id, destination, resp.chunks_created
            );
        }
    }
    Ok(())
}
