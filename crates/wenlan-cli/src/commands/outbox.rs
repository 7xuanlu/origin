// SPDX-License-Identifier: Apache-2.0
//! `wenlan outbox status|drain`.

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

use crate::client::WenlanClient;
use crate::outbox;
use crate::output::{print_json, ResolvedFormat};

#[derive(Debug, Subcommand)]
pub enum OutboxCommand {
    /// List queued and failed envelope files without contacting the daemon.
    Status,
    /// Ask the daemon to replay queued envelopes.
    Drain,
}

#[derive(Debug, Serialize)]
struct StatusOutput {
    queued: usize,
    failed: usize,
    queued_files: Vec<String>,
    failed_files: Vec<String>,
}

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    command: OutboxCommand,
) -> Result<()> {
    match command {
        OutboxCommand::Status => {
            let status = outbox::status()?;
            let output = StatusOutput {
                queued: status.queued.len(),
                failed: status.failed.len(),
                queued_files: status
                    .queued
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect(),
                failed_files: status
                    .failed
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect(),
            };
            if !quiet {
                match format {
                    ResolvedFormat::Json => print_json(&output)?,
                    ResolvedFormat::Table => {
                        println!("Queued: {}", output.queued);
                        for file in &output.queued_files {
                            println!("  {file}");
                        }
                        println!("Failed: {}", output.failed);
                        for file in &output.failed_files {
                            println!("  {file}");
                        }
                    }
                }
            }
        }
        OutboxCommand::Drain => {
            let report = client.drain_outbox().await?;
            if !quiet {
                match format {
                    ResolvedFormat::Json => print_json(&report)?,
                    ResolvedFormat::Table => {
                        println!(
                            "Applied: {}, duplicate: {}, failed: {}, remaining: {}",
                            report.applied, report.duplicate, report.failed, report.remaining
                        );
                        for detail in report.details {
                            if let Some(error) = detail.error {
                                println!("{}: {} ({})", detail.outcome, detail.file, error);
                            } else {
                                println!("{}: {}", detail.outcome, detail.file);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
