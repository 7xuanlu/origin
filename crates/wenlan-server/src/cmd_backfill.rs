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

#[cfg(target_os = "linux")]
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

/// Exit status `launchctl print` uses for a target launchd does not have.
#[cfg(target_os = "macos")]
const LAUNCHCTL_NO_SUCH_SERVICE: i32 = 113;

/// The current user's launchd domain id, for `gui/<uid>/<label>` targets.
/// Mirrors `current_user_id` in wenlan-cli's `commands::service`; the CLI is a
/// separate crate this daemon does not depend on, so the lookup is repeated
/// rather than shared.
#[cfg(target_os = "macos")]
fn current_user_id() -> Result<String> {
    let output = std::process::Command::new("id")
        .arg("-u")
        .output()
        .context("run id -u for the launchd user domain")?;
    if !output.status.success() {
        anyhow::bail!("id -u failed (exit {})", output.status.code().unwrap_or(-1));
    }
    let uid = std::str::from_utf8(&output.stdout)
        .context("id -u returned non-UTF-8 output")?
        .trim()
        .to_owned();
    if uid.is_empty() || !uid.bytes().all(|byte| byte.is_ascii_digit()) {
        anyhow::bail!("id -u returned invalid user id: {uid:?}");
    }
    Ok(uid)
}

#[cfg(target_os = "macos")]
fn launchctl_print_exit_code(target: &str) -> Result<Option<i32>> {
    let output = std::process::Command::new("launchctl")
        .args(["print", target])
        .output()
        .context("spawn launchctl print")?;
    Ok(output.status.code())
}

/// Refuses while launchd still has the daemon job loaded.
///
/// The plist file is not the answer on macOS: `wenlan background off` boots the
/// job out and deliberately keeps
/// `~/Library/LaunchAgents/com.wenlan.server.plist`, so a file check refused
/// precisely after the command this error tells the user to run.
/// `launchctl print gui/<uid>/<label>` exits 113 for a target launchd does not
/// have — the same signal `wenlan background off` itself reads (wenlan-cli
/// `commands::service::stop_registered_service`).
///
/// `print_exit_code` is injected so tests can drive every launchd answer
/// without loading or unloading the real service.
#[cfg(target_os = "macos")]
fn check_launchd_job_unloaded(
    uid: &str,
    print_exit_code: impl FnOnce(&str) -> Result<Option<i32>>,
) -> Result<()> {
    let target = format!("gui/{uid}/{SERVICE_LABEL}");
    match print_exit_code(&target)? {
        Some(LAUNCHCTL_NO_SUCH_SERVICE) => Ok(()),
        Some(0) => Err(anyhow!(
            "launchd still has the Wenlan daemon loaded as '{}', so it can restart in the \
             middle of this command.\nIts LaunchAgent is at:\n  {}\n\
             Turn it off first to prevent auto-restart:\n  wenlan background off\n\
             Then re-run this command. (Restart after with `wenlan background on`.)",
            target,
            service_unit_path()?.display()
        )),
        other => Err(anyhow!(
            "Could not read launchd state for '{}': launchctl print exit status {}. \
             Refusing while it is unknown whether the daemon can restart mid-run.\n\
             Turn it off first:\n  wenlan background off\n\
             Then re-run this command. (Restart after with `wenlan background on`.)",
            target,
            match other {
                Some(code) => code.to_string(),
                None => "killed by a signal".to_string(),
            }
        )),
    }
}

/// Returns Ok when nothing can restart the daemon under this command.
/// Returns Err with instructions when something can.
pub(crate) fn check_service_unloaded() -> Result<()> {
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
    #[cfg(target_os = "macos")]
    {
        check_launchd_job_unloaded(&current_user_id()?, launchctl_print_exit_code)
    }
    #[cfg(target_os = "linux")]
    {
        let unit = service_unit_path()?;
        check_service_unit_absent(&unit)
    }
}

pub(crate) async fn check_daemon_not_running() -> Result<()> {
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
    #[cfg(target_os = "linux")]
    #[test]
    fn check_service_unloaded_returns_ok_when_no_service_installed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let unit = tmp.path().join("wenlan-server.service");

        super::check_service_unit_absent(&unit).expect("expected Ok for absent test unit");
    }

    /// `wenlan background off` boots the launchd job out and keeps the plist on
    /// disk, so an unloaded job must let the command run even though the file
    /// is still there.
    #[cfg(target_os = "macos")]
    #[test]
    fn an_unloaded_launchd_job_lets_the_command_run() {
        let mut asked = Vec::new();
        super::check_launchd_job_unloaded("501", |target| {
            asked.push(target.to_string());
            Ok(Some(113))
        })
        .expect("an unloaded launchd job must not block the command");
        assert_eq!(asked, vec!["gui/501/com.wenlan.server".to_string()]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_loaded_launchd_job_blocks_and_names_background_off() {
        let error = super::check_launchd_job_unloaded("501", |_| Ok(Some(0)))
            .expect_err("a loaded launchd job must block the command");
        let message = format!("{error}");
        assert!(message.contains("gui/501/com.wenlan.server"), "{message}");
        assert!(message.contains("wenlan background off"), "{message}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn an_unreadable_launchd_state_blocks() {
        let error = super::check_launchd_job_unloaded("501", |_| Ok(Some(2)))
            .expect_err("an unknown launchd state must block the command");
        assert!(
            format!("{error}").contains("Could not read launchd state"),
            "{error}"
        );
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
