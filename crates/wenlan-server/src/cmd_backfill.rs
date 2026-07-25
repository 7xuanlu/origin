// SPDX-License-Identifier: Apache-2.0
//! `backfill-stale-pages` internal CLI subcommand.
//!
//! Deletes archived pages that look like old distillation failures (large
//! source_memory_ids, no entity, no domain, not user-edited). Source memories
//! are NOT modified.
//!
//! `page_sources` rows are deleted automatically via ON DELETE CASCADE.

use anyhow::{anyhow, Context, Result};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;
use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;

const DAEMON_PROBE_TIMEOUT: Duration = Duration::from_millis(500);

pub async fn run(dry_run: bool) -> anyhow::Result<()> {
    // Step 1a: refuse if the platform service manager has the daemon
    // registered. With auto-restart enabled, killing the daemon manually
    // wouldn't be enough — the service manager respawns it, creating a race
    // where the daemon could start writing between our probe and our SQLite
    // writes.
    check_service_unloaded()?;

    // Step 1b: refuse if a daemon is currently running (covers manually-started
    // instances and the brief window between service unload and respawn).
    check_daemon_not_running().await?;

    // Step 2: open the DB directly (not via daemon).
    // Mirrors the path computation in `run_daemon()` in main.rs.
    let origin_root = std::env::var_os("WENLAN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("wenlan")
        });
    let data_dir = origin_root.join("memorydb");
    let _data_root_lock = super::DaemonDataLock::acquire(&origin_root, true)?;

    let db = MemoryDB::new(&data_dir, Arc::new(NoopEmitter))
        .await
        .with_context(|| format!("opening MemoryDB at {}", data_dir.display()))?;

    // Step 3: query candidates.
    let candidates = db
        .find_stale_archived_pages()
        .await
        .context("querying stale pages")?;

    if candidates.is_empty() {
        println!("No stale archived pages found. Nothing to do.");
        return Ok(());
    }

    println!("Found {} candidate page(s):\n", candidates.len());
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
        "Delete {} page(s) and their page_sources rows? (y/N): ",
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
    // page_sources rows cascade automatically (ON DELETE CASCADE FK).
    let mut deleted = 0usize;
    for c in &candidates {
        db.delete_page(&c.id)
            .await
            .with_context(|| format!("deleting page {}", c.id))?;
        deleted += 1;
    }

    println!(
        "Deleted {} page(s). Source memories were NOT modified.",
        deleted
    );
    println!();
    println!("Next steps to re-distill the freed sources:");
    println!("  - Sources with enrichment_steps rows will be eligible on the next distill cycle.");
    println!("  - Raw sources need re-enrichment first. Either:");
    println!("    (a) Re-import: touch the original source files (e.g., `touch ~/second-brain/inbox/*.md`)");
    println!("    (b) Wait for entity_backfill to gradually backfill entity_ids");

    Ok(())
}

/// Service label registered with the host service manager. Must match
/// `wenlan_cli::commands::service::SERVICE_LABEL` — `service_unit_path_matches_cli`
/// pins both copies to the on-disk paths `service-manager` 0.11 actually writes.
const SERVICE_LABEL: &str = "com.wenlan.server";

/// Resolves the platform-specific path to the Wenlan service unit file on
/// Unix-likes. Mirrors the on-disk path that `service-manager` 0.11 writes:
/// - macOS (launchd): `~/Library/LaunchAgents/com.wenlan.server.plist`
///   (uses `ServiceLabel::to_qualified_name()` — qualifier kept).
/// - Linux (systemd-user): `~/.config/systemd/user/wenlan-server.service`
///   (uses `ServiceLabel::to_script_name()` — qualifier DROPPED, org+app
///   joined with `-`).
///
/// Windows uses `sc.exe` which writes no on-disk unit file, so this is
/// `#[cfg]`-gated off Windows. Kept in sync with
/// `wenlan-cli::commands::service::service_unit_path` via
/// `service_unit_path_matches_cli` below.
#[cfg(not(target_os = "windows"))]
fn service_unit_path() -> Result<std::path::PathBuf> {
    let label: service_manager::ServiceLabel =
        SERVICE_LABEL.parse().context("invalid service label")?;
    #[cfg(target_os = "macos")]
    {
        Ok(dirs::home_dir()
            .context("HOME not set")?
            .join("Library/LaunchAgents")
            .join(format!("{}.plist", label.to_qualified_name())))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(dirs::config_dir()
            .context("XDG_CONFIG_HOME not set")?
            .join("systemd/user")
            .join(format!("{}.service", label.to_script_name())))
    }
}

#[cfg(not(target_os = "windows"))]
fn check_service_unit_absent(unit: &std::path::Path) -> Result<()> {
    if unit.exists() {
        Err(anyhow!(
            "The Wenlan service is registered with the platform service manager at:\n  {}\n\
             Turn it off first to prevent auto-restart:\n  wenlan background off\n\
             Then re-run this command. (Restart after with `wenlan background on`.)",
            unit.display()
        ))
    } else {
        Ok(())
    }
}

/// Returns Ok if no service manager has the origin daemon registered.
/// Returns Err with instructions if a service unit file is present.
fn check_service_unloaded() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        // `sc.exe query <label>` exits 0 when the service is registered with
        // the Windows Service Control Manager, 1060 when it is not.
        let registered = std::process::Command::new("sc.exe")
            .args(["query", SERVICE_LABEL])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if registered {
            Err(anyhow!(
                "The Wenlan service is registered with the Windows Service Control Manager as \
                 '{SERVICE_LABEL}'.\n\
                 Turn it off first to prevent auto-restart:\n  wenlan background off\n\
                 Then re-run this command. (Restart after with `wenlan background on`.)"
            ))
        } else {
            Ok(())
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let unit = service_unit_path()?;
        check_service_unit_absent(&unit)
    }
}

async fn check_daemon_not_running() -> Result<()> {
    // Mirror the port-reading logic from cmd_status in main.rs.
    let port: u16 = std::env::var("WENLAN_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7878);
    let probe_url = format!("http://127.0.0.1:{}/api/health", port);

    let client = reqwest::Client::builder()
        .timeout(DAEMON_PROBE_TIMEOUT)
        .build()
        .context("building reqwest client")?;
    match client.get(&probe_url).send().await {
        Ok(_) => Err(anyhow!(
            "Daemon is running on :{port}. Stop it before running backfill:\n  \
             wenlan background off\n  \
             # or: kill -9 $(lsof -ti :{port})"
        )),
        // Truly refused (nothing listening): safe to proceed.
        Err(e) if e.is_connect() => Ok(()),
        // Timeout: daemon may be alive but wedged (e.g. GPU inference). Refuse.
        Err(e) if e.is_timeout() => Err(anyhow!(
            "Daemon probe to :{port} timed out after {}ms. \
             Daemon may be busy. Stop it explicitly and retry:\n  \
             wenlan background off\n  \
             # or: kill -9 $(lsof -ti :{port})",
            DAEMON_PROBE_TIMEOUT.as_millis()
        )),
        // Any other network error: surface it.
        Err(e) => Err(anyhow!("Daemon probe to :{port} failed unexpectedly: {e}")),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn check_service_unloaded_returns_ok_when_no_service_installed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let unit = tmp.path().join("com.wenlan.server.plist");

        super::check_service_unit_absent(&unit).expect("expected Ok for absent test unit");
    }

    /// Pin both copies (CLI + server) to the on-disk paths `service-manager`
    /// 0.11 actually writes. If service-manager changes its label-to-path
    /// rules in a future major bump, this test must be re-derived from the
    /// crate source (`launchd.rs`, `systemd.rs`), not from prior intuition.
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn service_unit_path_matches_cli() {
        let path = super::service_unit_path().expect("service_unit_path should not fail");
        let p = path.to_string_lossy();

        #[cfg(target_os = "macos")]
        assert!(
            p.ends_with("Library/LaunchAgents/com.wenlan.server.plist"),
            "unexpected macOS path: {p}"
        );
        #[cfg(target_os = "linux")]
        assert!(
            p.ends_with(".config/systemd/user/wenlan-server.service"),
            "unexpected Linux path: {p}"
        );
    }
}
