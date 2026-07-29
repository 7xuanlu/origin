// SPDX-License-Identifier: Apache-2.0
//! `truth-cutover` internal CLI subcommand — the M5 ceremony.
//!
//! Advancing the truth cutover generation stops the projection directory
//! carrying prose no evidence supports. It does that by DELETING those pages'
//! `.md` files out of the user's vault, so this is the most destructive
//! operation the daemon owns.
//!
//! # Why this is not an HTTP route
//!
//! The daemon is loopback and unauthenticated. Anything that can reach :7878 can
//! call any route on it, agents included. A vault-emptying operation behind an
//! admin route would be one forged request away, and the audit log — the
//! compensating control for reads — does not undo a delete. So the ceremony
//! lives here, on a hidden subcommand of the daemon binary, reachable only by
//! someone with a shell on the machine and only while the daemon is stopped.
//!
//! # The dry run is not advisory
//!
//! `--apply` refuses without a stored dry-run digest that still matches the
//! projection's current state. The digest covers the generation, every projected
//! page, whether each is supported, and the file each would cost — so a
//! distillation cycle that ran between the two invalidates it, and the operator
//! reads the new plan instead of applying an old one.

use anyhow::{anyhow, Context, Result};
use std::io::{self, Write};
use std::sync::Arc;
use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;

/// `app_metadata` key holding the digest of the last dry run.
const DRY_RUN_DIGEST_KEY: &str = "truth_cutover_dryrun_digest";

pub async fn run(generation: i64, apply: bool) -> Result<()> {
    if generation < 1 {
        return Err(anyhow!(
            "generation must be 1 or higher; 0 is the inert state and there is \
             nothing to advance to"
        ));
    }

    // Same two refusals as `backfill-stale-pages`, for the same reason: a daemon
    // that respawns between our probe and our writes would be a second writer
    // against the file we are deleting from.
    super::cmd_backfill::check_service_unloaded()?;
    super::cmd_backfill::check_daemon_not_running().await?;

    let wenlan_root = std::env::var_os("WENLAN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("wenlan")
        });
    let data_dir = wenlan_root.join("memorydb");
    let _data_root_lock = super::DaemonDataLock::acquire(&wenlan_root, true)?;

    let db = MemoryDB::new(&data_dir, Arc::new(NoopEmitter))
        .await
        .with_context(|| format!("opening MemoryDB at {}", data_dir.display()))?;

    // Read through the same resolver the daemon uses. `WENLAN_DATA_DIR` does NOT
    // cover the page vault — the projection directory comes from the config's
    // `knowledge_path`, which defaults to `~/.wenlan/pages` — so an isolated data
    // dir alone would point this at the user's real vault.
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    println!("Projection directory: {}", knowledge_path.display());
    if !knowledge_path.exists() {
        println!("It does not exist yet. Nothing is projected, so nothing can be evicted.");
    }

    let current = db.truth_cutover_generation().await?;
    let fence = db.cutover_fence().await?;
    println!("Current generation: {current}");
    println!("Fence: epoch {} phase {:?}", fence.epoch, fence.phase);
    if current >= generation {
        return Err(anyhow!(
            "already at generation {current}; the cutover is forward-only"
        ));
    }

    let writer = wenlan_core::export::knowledge::KnowledgeWriter::new(knowledge_path.clone(), &db);
    let plan = writer
        .plan_truth_cutover(&db, generation)
        .await
        .context("planning the cutover")?;

    println!(
        "\nAt generation {}, {} of {} projected page(s) would stop being projected:\n",
        plan.generation,
        plan.evictions.len(),
        plan.projected
    );
    for eviction in &plan.evictions {
        if eviction.files.is_empty() {
            println!(
                "  {}  ->  no file on disk (stale state entry)",
                eviction.page_id
            );
        } else {
            println!(
                "  {}  ->  move to archive/: {}",
                eviction.page_id,
                eviction.files.join(", ")
            );
        }
    }
    println!("\nPlan digest: {}", plan.digest);

    if !apply {
        db.set_app_metadata(DRY_RUN_DIGEST_KEY, &plan.digest)
            .await?;
        println!(
            "\nDry run recorded. Re-run with `--apply` to advance. The apply refuses \
             if anything above changes in the meantime."
        );
        return Ok(());
    }

    let recorded = db.get_app_metadata(DRY_RUN_DIGEST_KEY).await?;
    match recorded.as_deref() {
        // Blank is the spent marker a completed apply leaves behind. Falling
        // through to the mismatch arm would report "recorded , now <digest>",
        // which reads like corruption rather than "you already did this".
        None | Some("") => {
            return Err(anyhow!(
                "no unspent dry run on record. Run this command without `--apply` first \
                 and read the eviction list — it names files that will be deleted from \
                 the user's vault."
            ))
        }
        Some(digest) if digest != plan.digest => {
            return Err(anyhow!(
                "the projection changed since the dry run (recorded {digest}, now {}). \
                 Re-run without `--apply` and read the new plan.",
                plan.digest
            ))
        }
        Some(_) => {}
    }

    print!(
        "\nDelete {} file(s) from {} and advance to generation {}? This cannot be \
         undone by rolling the generation back. (y/N): ",
        plan.evictions.len(),
        knowledge_path.display(),
        generation
    );
    io::stdout().flush().ok();
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    if !matches!(answer.trim(), "y" | "Y") {
        println!("Aborted. Nothing was changed.");
        return Ok(());
    }

    let applied = writer
        .run_truth_cutover(&db, generation, &plan.digest)
        .await
        .context("running the cutover")?;
    // Spend the dry run. Leaving it would let a later apply reuse an approval
    // the operator gave for a plan that has already been carried out.
    db.set_app_metadata(DRY_RUN_DIGEST_KEY, "").await?;

    println!(
        "\nCutover committed at generation {}. {} page(s) evicted.",
        applied.generation,
        applied.evictions.len()
    );
    println!("Restart the daemon with `wenlan background on`.");
    Ok(())
}
