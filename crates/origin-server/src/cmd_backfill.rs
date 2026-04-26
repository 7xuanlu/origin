// SPDX-License-Identifier: Apache-2.0
//! `backfill-stale-concepts` CLI subcommand.
//!
//! Deletes archived concepts that look like Mode B failures (large
//! source_memory_ids, no entity, no domain, not user-edited). Source memories
//! are NOT modified — see spec 2026-04-25-bad-concept-distill-fix-design.md
//! for the rationale and follow-up steps required to re-distill them.
//!
//! `concept_sources` rows are deleted automatically via ON DELETE CASCADE.

use anyhow::{anyhow, Context, Result};
use origin_core::db::MemoryDB;
use origin_core::events::NoopEmitter;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

const DAEMON_PROBE_URL: &str = "http://127.0.0.1:7878/api/health";
const DAEMON_PROBE_TIMEOUT: Duration = Duration::from_millis(500);

pub async fn run(dry_run: bool) -> anyhow::Result<()> {
    // Step 1: refuse if daemon is running on :7878.
    check_daemon_not_running().await?;

    // Step 2: open the DB directly (not via daemon).
    // Mirrors the path computation in `run_daemon()` in main.rs.
    let origin_root = std::env::var_os("ORIGIN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("origin")
        });
    let data_dir = origin_root.join("memorydb");

    let db = MemoryDB::new(&data_dir, Arc::new(NoopEmitter))
        .await
        .with_context(|| format!("opening MemoryDB at {}", data_dir.display()))?;

    // Step 3: query candidates.
    let candidates = db
        .find_stale_archived_concepts()
        .await
        .context("querying stale concepts")?;

    if candidates.is_empty() {
        println!("No stale archived concepts found. Nothing to do.");
        return Ok(());
    }

    println!("Found {} candidate concept(s):\n", candidates.len());
    for c in &candidates {
        println!(
            "  {} \"{}\" — {} sources — created {}",
            c.id,
            c.title,
            c.source_memory_ids.len(),
            c.created_at,
        );
    }
    println!();

    if dry_run {
        println!("--dry-run: no changes made.");
        return Ok(());
    }

    // Step 4: confirm.
    print!(
        "Delete {} concept(s) and their concept_sources rows? (y/N): ",
        candidates.len()
    );
    io::stdout().flush().ok();
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("reading confirmation")?;
    let answer = answer.trim().to_lowercase();
    if answer != "y" && answer != "yes" {
        println!("Aborted.");
        return Ok(());
    }

    // Step 5: delete.
    // concept_sources rows cascade automatically (ON DELETE CASCADE FK).
    let mut deleted = 0usize;
    for c in &candidates {
        db.delete_concept(&c.id)
            .await
            .with_context(|| format!("deleting concept {}", c.id))?;
        deleted += 1;
    }

    println!(
        "Deleted {} concept(s). Source memories were NOT modified.",
        deleted
    );
    println!();
    println!("Next steps to re-distill the freed sources:");
    println!("  - Sources with enrichment_steps rows will be eligible on next refinery tick.");
    println!("  - Raw sources need re-enrichment first. Either:");
    println!("    (a) Re-import: touch the original source files (e.g., `touch ~/second-brain/inbox/*.md`)");
    println!("    (b) Wait for entity_backfill to gradually backfill entity_ids");

    Ok(())
}

async fn check_daemon_not_running() -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(DAEMON_PROBE_TIMEOUT)
        .build()
        .context("building reqwest client")?;
    match client.get(DAEMON_PROBE_URL).send().await {
        Ok(_) => Err(anyhow!(
            "Daemon is running on :7878. Stop it before running backfill:\n  \
             launchctl unload ~/Library/LaunchAgents/com.origin.server.plist\n  \
             # or: kill -9 $(lsof -ti :7878)"
        )),
        // Connection refused / timeout / etc — daemon is not running, proceed.
        Err(_) => Ok(()),
    }
}
