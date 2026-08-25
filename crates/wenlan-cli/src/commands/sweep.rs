// SPDX-License-Identifier: Apache-2.0
//! `wenlan sweep` — force one bounded pass over every due ambient job
//! (chunking, citation backfill, relations/communities, ...), bypassing the
//! daemon's idle gate. Each job's own per-call slice bound is unchanged; only
//! the wait for a quiet turn is skipped.

use anyhow::Result;

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

/// We still hit the daemon under `quiet` so a connection failure still
/// surfaces via exit code, matching `commands::status::run`.
pub async fn run(client: &WenlanClient, format: ResolvedFormat, quiet: bool) -> Result<()> {
    let report = client.sweep_ambient().await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&report)?,
        ResolvedFormat::Table => {
            for phase in &report.phases {
                if !phase.attempted {
                    println!("{}: skipped (not available)", phase.job);
                    continue;
                }
                let outcome = if phase.panicked {
                    "panicked"
                } else if phase.selected {
                    "ran"
                } else {
                    "nothing due"
                };
                println!(
                    "{}: {} (llm_calls={}, elapsed_ms={})",
                    phase.job, outcome, phase.llm_calls, phase.elapsed_ms
                );
            }
        }
    }
    Ok(())
}
