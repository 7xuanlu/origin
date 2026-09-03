// SPDX-License-Identifier: Apache-2.0
//! `wenlan brief [topic]` and typed handoff updates.

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use wenlan_types::{
    Brief, BriefItem, BriefReadRequest, BriefReadResponse, BriefReadState, BriefUpdateReceipt,
    BriefUpdateRequest, OutboxEnvelope, OutboxPayload, OUTBOX_SCHEMA,
};

use crate::client::WenlanClient;
use crate::outbox;
use crate::output::{print_json, ResolvedFormat};

#[derive(Debug, Args)]
pub struct BriefArgs {
    /// Optional topic for a separately-labelled same-Space recall.
    #[arg(value_name = "TOPIC")]
    pub topic: Option<String>,

    #[command(subcommand)]
    pub command: Option<BriefCommand>,
}

#[derive(Debug, Subcommand)]
pub enum BriefCommand {
    /// Apply a typed item-level handoff update from JSON.
    Update {
        /// JSON file containing a BriefUpdateRequest.
        #[arg(long, value_name = "JSON_FILE")]
        file: PathBuf,
    },
}

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    args: BriefArgs,
    effective_space: Option<String>,
    agent_name: Option<&str>,
) -> Result<()> {
    match args.command {
        Some(BriefCommand::Update { file }) => {
            let request = load_update(&file)?;
            if let Some(space) = effective_space.as_deref() {
                if request.space.trim() != space {
                    bail!(
                        "Brief update Space '{}' conflicts with resolved Space '{}'",
                        request.space,
                        space
                    );
                }
            }
            let receipt = match client.update_brief(&request).await {
                Ok(receipt) => receipt,
                Err(error) if client.is_local() && outbox::is_daemon_unreachable(&error) => {
                    let envelope = OutboxEnvelope {
                        schema: OUTBOX_SCHEMA,
                        created_at: outbox::now_rfc3339(),
                        caller_id: request.caller_id.clone(),
                        operation_id: request.operation_id.clone(),
                        space: Some(request.space.clone()),
                        agent_name: agent_name.map(str::to_owned),
                        reconcile_summary_version: true,
                        payload: OutboxPayload::BriefUpdate(request),
                    };
                    let path = outbox::enqueue(&envelope)?;
                    if !quiet {
                        print_queued(format, &path, &envelope.operation_id)?;
                    }
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
            if !quiet {
                match format {
                    ResolvedFormat::Json => print_json(&receipt)?,
                    ResolvedFormat::Table => print!("{}", format_update_receipt(&receipt)),
                }
            }
        }
        None => {
            let response = client
                .brief(&BriefReadRequest {
                    topic: args.topic,
                    space: None,
                    legacy_context_limit: None,
                })
                .await?;
            if !quiet {
                match format {
                    ResolvedFormat::Json => print_json(&response)?,
                    ResolvedFormat::Table => print!("{}", format_brief(&response)),
                }
            }
        }
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct QueuedOutput<'a> {
    status: &'static str,
    path: String,
    operation_id: &'a str,
}

fn print_queued(format: ResolvedFormat, path: &Path, operation_id: &str) -> Result<()> {
    match format {
        ResolvedFormat::Json => print_json(&QueuedOutput {
            status: "queued",
            path: path.display().to_string(),
            operation_id,
        }),
        ResolvedFormat::Table => {
            println!("queued: {}", path.display());
            Ok(())
        }
    }
}

pub fn load_update(path: &Path) -> Result<BriefUpdateRequest> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("reading Brief update {}", path.display()))?;
    serde_json::from_str(&body)
        .with_context(|| format!("invalid Brief update JSON in {}", path.display()))
}

fn format_brief(response: &BriefReadResponse) -> String {
    match response.state {
        BriefReadState::SpaceNotResolved => {
            "No Space resolved. Use --space or configure a project mapping.\n".into()
        }
        BriefReadState::BriefNotCreated => format!(
            "No Brief exists for Space '{}'. Run `wenlan brief update --file <json>` \
             (or the /handoff plugin command) to create it.\n",
            response.space.as_deref().unwrap_or("unknown")
        ),
        BriefReadState::Ready => response
            .brief
            .as_ref()
            .map(|brief| format_ready_brief(brief, response))
            .unwrap_or_else(|| "Brief response was marked ready without Brief data.\n".into()),
    }
}

fn format_ready_brief(brief: &Brief, response: &BriefReadResponse) -> String {
    let mut output = format!("Space: {}\n\nLast session\n", brief.space);
    if brief.last_session_summary.trim().is_empty() {
        output.push_str("  (none)\n");
    } else {
        for line in brief.last_session_summary.lines() {
            output.push_str("  ");
            output.push_str(line.trim());
            output.push('\n');
        }
    }
    format_items(&mut output, "Active", &brief.active);
    format_items(&mut output, "Backlog", &brief.backlog);

    if let Some(related) = &response.related_context {
        output.push_str(&format!("\nRelated Context: {}\n", related.query));
        if related.results.is_empty() {
            output.push_str("  (no related memories)\n");
        } else {
            for result in related.results.iter().take(10) {
                let title = if result.title.is_empty() {
                    result.content.lines().next().unwrap_or("(no title)")
                } else {
                    &result.title
                };
                output.push_str(&format!("  [{:.3}] {}\n", result.score, title));
            }
        }
    }
    output
}

fn format_items(output: &mut String, heading: &str, items: &[BriefItem]) {
    output.push_str(&format!("\n{heading}\n"));
    if items.is_empty() {
        output.push_str("  (none)\n");
        return;
    }
    for item in items {
        output.push_str("  - ");
        output.push_str(&item.text);
        if let Some(gate) = item.gate.as_deref() {
            output.push_str(" [gate: ");
            output.push_str(gate);
            output.push(']');
        }
        output.push('\n');
    }
}

fn format_update_receipt(receipt: &BriefUpdateReceipt) -> String {
    let mut output = format!(
        "Updated Brief for '{}': {} applied, {} conflict(s).\n",
        receipt.space,
        receipt.applied.len(),
        receipt.conflicts.len()
    );
    if let Some(path) = &receipt.projection_path {
        output.push_str(&format!("Receipt: {path}\n"));
    }
    for warning in &receipt.warnings {
        output.push_str(&format!("Warning: {warning}\n"));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use wenlan_types::{BriefItemState, BriefRelatedContext, SearchResult};

    fn result() -> SearchResult {
        serde_json::from_value(serde_json::json!({
            "id": "memory-1",
            "content": "related memory",
            "source": "memory",
            "source_id": "memory-1",
            "title": "Related",
            "chunk_index": 0,
            "last_modified": 0,
            "score": 0.9
        }))
        .unwrap()
    }

    #[test]
    fn human_brief_omits_machine_ids_and_keeps_related_context_separate() {
        let response = BriefReadResponse {
            state: BriefReadState::Ready,
            space: Some("work".into()),
            brief: Some(Brief {
                space_id: "secret-space-id".into(),
                space: "work".into(),
                last_session_summary: "shipped storage".into(),
                last_handoff_at: Some(1),
                version: 3,
                active: vec![BriefItem {
                    id: "secret-item-id".into(),
                    text: "open PR".into(),
                    state: BriefItemState::Active,
                    added_at: 1,
                    gate: Some("CI green".into()),
                    version: 2,
                }],
                backlog: Vec::new(),
            }),
            related_context: Some(BriefRelatedContext {
                query: "release".into(),
                results: vec![result()],
            }),
        };

        let output = format_brief(&response);

        assert!(output.contains("open PR"));
        assert!(output.contains("Related Context: release"));
        assert!(!output.contains("secret-space-id"));
        assert!(!output.contains("secret-item-id"));
    }

    #[test]
    fn no_brief_hint_names_a_command_the_cli_actually_has() {
        let response = BriefReadResponse {
            state: BriefReadState::BriefNotCreated,
            space: Some("work".into()),
            brief: None,
            related_context: None,
        };

        let output = format_brief(&response);

        // The CLI has no `handoff` subcommand; the hint must point at
        // something a CLI user can actually run.
        assert!(!output.contains("The first handoff"));
        assert!(output.contains("wenlan brief update"));
    }
}
