// SPDX-License-Identifier: Apache-2.0
//! `wenlan sources add <path>` — register a Directory source and sync it now.
//!
//! Thin HTTP client: POST /api/sources to register the path as a Directory
//! source (idempotent — an already-registered path is treated as success),
//! then POST /api/sources/{id}/sync, then render the returned stats. No DB
//! access, no new endpoints (per AGENTS.md crate boundaries).

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::client::{SyncStats, WenlanClient};
use crate::output::{print_json, ResolvedFormat};

#[derive(clap::Subcommand)]
pub enum SourcesCommand {
    /// Add a folder or file source and sync it now.
    Add {
        /// Path to a directory or file to add as a source.
        path: PathBuf,
    },
}

pub async fn run_sources(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    command: SourcesCommand,
) -> Result<()> {
    match command {
        SourcesCommand::Add { path } => run(client, format, quiet, path).await,
    }
}

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    path: PathBuf,
) -> Result<()> {
    // Resolve to an absolute, canonical path so the daemon stores a stable key
    // and an idempotent re-run matches the existing registration.
    let abs = std::fs::canonicalize(&path)
        .with_context(|| format!("resolving path: {}", path.display()))?;
    let abs_str = abs.to_string_lossy().to_string();

    let id = register_source(client, &abs_str).await?;
    let stats = client.sync_source(&id).await?;

    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&stats)?,
        ResolvedFormat::Table => print!("{}", format_stats(&id, &stats)),
    }
    Ok(())
}

/// Register `abs_path` as a Directory source, idempotently, returning its id.
///
/// The happy path is a single POST /api/sources. If the path is already
/// registered the POST fails (the daemon rejects a duplicate), so we recover
/// the existing id from the source list instead of erroring. A genuine failure
/// (path missing, reserved root, daemon down) surfaces the original POST error
/// because the list will not contain the path.
async fn register_source(client: &WenlanClient, abs_path: &str) -> Result<String> {
    match client.add_source("directory", abs_path).await {
        Ok(source) => Ok(source.id),
        Err(add_err) => {
            let existing = client.list_sources().await.ok().and_then(|sources| {
                sources
                    .into_iter()
                    .find(|s| s.path.to_string_lossy() == abs_path)
            });
            match existing {
                Some(s) => Ok(s.id),
                None => Err(add_err),
            }
        }
    }
}

fn format_stats(id: &str, stats: &SyncStats) -> String {
    let mut out = format!(
        "Synced {}: {} file(s) found, {} ingested, {} skipped, {} error(s)\n",
        id, stats.files_found, stats.ingested, stats.skipped, stats.errors,
    );
    if let Some(detail) = &stats.error_detail {
        out.push_str(&format!("  error detail: {}\n", detail));
    }
    if let Some(paused) = &stats.paused {
        out.push_str(&format!("  enrichment paused: {}\n", paused));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(found: usize, ingested: usize, skipped: usize, errors: usize) -> SyncStats {
        SyncStats {
            files_found: found,
            ingested,
            skipped,
            errors,
            error_detail: None,
            paused: None,
        }
    }

    #[test]
    fn format_stats_renders_counts() {
        let out = format_stats("directory-notes", &stats(3, 2, 1, 0));
        assert!(out.contains("directory-notes"));
        assert!(out.contains("3 file(s) found"));
        assert!(out.contains("2 ingested"));
        assert!(out.contains("1 skipped"));
        assert!(out.contains("0 error(s)"));
    }

    #[test]
    fn format_stats_surfaces_error_detail_and_pause() {
        let s = SyncStats {
            error_detail: Some("file_read_errors".to_string()),
            paused: Some("llm backoff".to_string()),
            ..stats(4, 1, 0, 3)
        };
        let out = format_stats("directory-docs", &s);
        assert!(out.contains("file_read_errors"));
        assert!(out.contains("llm backoff"));
    }
}
