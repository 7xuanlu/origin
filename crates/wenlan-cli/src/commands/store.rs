// SPDX-License-Identifier: Apache-2.0
//! `wenlan capture [text] [--file <path>] [--type <type>]` — POST /api/memory/store.

use anyhow::{Context, Result};
use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use crate::client::WenlanClient;
use crate::output::{print_json, OutputFormat};

pub async fn run(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    text: Option<String>,
    file: Option<PathBuf>,
    memory_type: Option<String>,
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
    let resp = client.store(content, memory_type).await?;
    if quiet {
        return Ok(());
    }
    match format {
        OutputFormat::Json => print_json(&resp)?,
        OutputFormat::Table => {
            let destination = resp.space.as_deref().unwrap_or("Uncategorized");
            println!(
                "Stored memory {} in {} ({} chunk(s))",
                resp.source_id, destination, resp.chunks_created
            );
        }
        OutputFormat::Auto => unreachable!("Auto resolved by main before dispatch"),
    }
    Ok(())
}
