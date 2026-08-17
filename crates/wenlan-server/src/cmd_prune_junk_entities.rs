// SPDX-License-Identifier: Apache-2.0
//! `prune-junk-entities` internal CLI subcommand.
//!
//! One-off cleanup for the entity pages the extractor minted before the
//! admission gate existed: quantities, percentages, file paths, source
//! filenames, commit shas, and test residue. Candidates come from exactly the
//! `wenlan_core::kg::entity_name_quality` rules that now run at write time, so
//! the cleanup and the gate can never disagree about what junk is.
//!
//! Two tiers, and the difference matters:
//!   * `reject` — structurally impossible as an entity name. Archived by
//!     default.
//!   * `review` — ambiguous (a bare version, a two-letter name, a URL). Listed
//!     always, archived only under `--include-review`, so a bulk run cannot
//!     quietly take out something real.
//!
//! Dry run is the default, and `--apply` archives rather than deletes: the
//! shadow page flips to `status='archived'`, which hides the entity from every
//! reader while leaving the row, its observations, its aliases, and its memory
//! links in place; its live edges are retracted alongside it, stamped so they
//! come back. `--restore <ENTITY_ID>` is the way back: it un-archives the
//! entity and re-activates those edges (a bare `UPDATE pages SET status`
//! would bring the entity back with none of its relations).
//!
//! Like the other one-off commands here, it refuses to run while the daemon is
//! up, so it never writes to a live WAL.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;
use wenlan_core::db::{EntityLinkSummary, MemoryDB};
use wenlan_core::events::NoopEmitter;
use wenlan_core::kg::entity_name_quality::{assess, Reason, Tier};

/// One entity page the gate would not admit today.
struct Candidate {
    entity_id: String,
    name: String,
    entity_type: String,
    reason: Reason,
    tier: Tier,
    links: EntityLinkSummary,
}

/// Open the store the way every one-off command here does: daemon down,
/// data-root lock held for the life of the returned guard.
async fn open_store() -> Result<(MemoryDB, super::DaemonDataLock)> {
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
    let data_root_lock = super::DaemonDataLock::acquire(&wenlan_root, true)?;

    let db = MemoryDB::new(&data_dir, Arc::new(NoopEmitter))
        .await
        .with_context(|| format!("opening MemoryDB at {}", data_dir.display()))?;
    Ok((db, data_root_lock))
}

/// `--restore <ENTITY_ID>`: un-archive each entity and re-activate the edges
/// its archive retracted. Reports per id; an id that is not archived (or does
/// not exist) is a no-op, not an error.
pub async fn restore(entity_ids: &[String]) -> Result<()> {
    let (db, _data_root_lock) = open_store().await?;
    let mut restored = 0usize;
    for id in entity_ids {
        let before = db
            .entity_link_summary(id)
            .await
            .with_context(|| format!("link summary for {id}"))?;
        if db
            .restore_entity(id)
            .await
            .with_context(|| format!("restoring entity {id}"))?
        {
            let after = db
                .entity_link_summary(id)
                .await
                .with_context(|| format!("link summary for {id}"))?;
            println!(
                "restored {id}: active edges {} -> {}",
                before.active_edges, after.active_edges
            );
            restored += 1;
        } else {
            println!("skipped {id}: no archived entity page with that id");
        }
    }
    println!(
        "Restored {restored} of {} entity page(s).",
        entity_ids.len()
    );
    Ok(())
}

pub async fn run(apply: bool, include_review: bool) -> Result<()> {
    let (db, _data_root_lock) = open_store().await?;

    let candidates = collect_candidates(&db).await?;
    if candidates.is_empty() {
        println!("No junk entity pages found. Nothing to do.");
        return Ok(());
    }

    println!(
        "Found {} entity page(s) the admission gate would refuse today.\n",
        candidates.len()
    );
    println!(
        "  {:<7} {:<18} {:<13} {:<44} links",
        "tier", "rule", "type", "name"
    );
    for c in &candidates {
        println!(
            "  {:<7} {:<18} {:<13} {:<44} mem={} obs={} alias={} edge={} primary={}",
            c.tier.as_str(),
            c.reason.as_str(),
            c.entity_type,
            truncate(&c.name, 44),
            c.links.memory_links,
            c.links.observations,
            c.links.aliases,
            c.links.active_edges,
            c.links.primary_memories,
        );
    }

    let mut by_rule: BTreeMap<(&'static str, &'static str), usize> = BTreeMap::new();
    for c in &candidates {
        *by_rule
            .entry((c.tier.as_str(), c.reason.as_str()))
            .or_default() += 1;
    }
    println!("\nBy rule:");
    for ((tier, reason), count) in &by_rule {
        println!("  {tier:<7} {reason:<18} {count}");
    }

    let targets: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| include_review || c.tier == Tier::Reject)
        .collect();
    let held_back = candidates.len() - targets.len();
    println!(
        "\n{} candidate(s) in scope for archiving; {held_back} review-tier candidate(s) held back{}.",
        targets.len(),
        if include_review { " (none: --include-review)" } else { ", pass --include-review to include them" }
    );

    if !apply {
        println!("Dry run: nothing was changed. Re-run with --apply to archive the in-scope set.");
        return Ok(());
    }
    if targets.is_empty() {
        println!("Nothing in scope. Nothing to do.");
        return Ok(());
    }

    print!(
        "Archive {} entity page(s)? Observations, aliases, and memory links are kept; live \
         edges are retracted and come back with `prune-junk-entities --restore <ENTITY_ID>`. \
         (y/N): ",
        targets.len()
    );
    io::stdout().flush().ok();
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("reading confirmation")?;
    if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
        println!("Aborted.");
        return Ok(());
    }

    let mut archived = 0usize;
    for c in &targets {
        if db
            .archive_entity(&c.entity_id)
            .await
            .with_context(|| format!("archiving entity {} ({:?})", c.entity_id, c.name))?
        {
            archived += 1;
        }
    }
    println!(
        "Archived {archived} entity page(s). Nothing was deleted; each entity's live edges were \
         retracted alongside it. To bring one back with its edges: \
         `prune-junk-entities --restore <ENTITY_ID>`."
    );
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
}

/// Every active entity the gate would not admit, with its blast radius. Pure
/// read: nothing here writes.
async fn collect_candidates(db: &MemoryDB) -> Result<Vec<Candidate>> {
    let entities = db
        .list_entities(None, None)
        .await
        .context("listing entities")?;
    let mut out: Vec<Candidate> = Vec::new();
    for e in entities {
        let Some(reason) = assess(&e.name).reason() else {
            continue;
        };
        let links = db
            .entity_link_summary(&e.id)
            .await
            .with_context(|| format!("link summary for {}", e.id))?;
        out.push(Candidate {
            entity_id: e.id,
            name: e.name,
            entity_type: e.entity_type,
            reason,
            tier: reason.tier(),
            links,
        });
    }
    out.sort_by(|a, b| {
        a.tier
            .as_str()
            .cmp(b.tier.as_str())
            .then_with(|| a.reason.as_str().cmp(b.reason.as_str()))
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .expect("MemoryDB::new");
        (db, dir)
    }

    /// The candidate set is exactly what the write-time gate refuses, split by
    /// tier: junk in, real entities untouched.
    #[tokio::test]
    async fn collects_what_the_gate_refuses_and_tags_each_with_its_tier() {
        let (db, _dir) = test_db().await;
        // Seeded through store_entity, the pre-gate write path -- precisely the
        // shape of the rows already in production.
        for (name, kind) in [
            ("30 minutes", "concept"),
            ("~/.claude/CLAUDE.md", "technology"),
            ("types.rs", "technology"),
            ("bd691cc", "project"),
            ("v1.2.3", "technology"),
            ("de", "concept"),
            ("Alice Chen", "person"),
            ("Rust", "technology"),
            ("Next.js", "technology"),
            ("7xuanlu/wenlan", "project"),
            ("Redis Pub/Sub", "technology"),
        ] {
            db.store_entity(name, kind, None, Some("post_ingest"), None)
                .await
                .expect("store_entity");
        }

        let found = collect_candidates(&db).await.expect("collect");
        let mut rejected: Vec<&str> = found
            .iter()
            .filter(|c| c.tier == Tier::Reject)
            .map(|c| c.name.as_str())
            .collect();
        rejected.sort_unstable();
        assert_eq!(
            rejected,
            vec!["30 minutes", "bd691cc", "types.rs", "~/.claude/CLAUDE.md"]
        );
        let mut review: Vec<&str> = found
            .iter()
            .filter(|c| c.tier == Tier::Review)
            .map(|c| c.name.as_str())
            .collect();
        review.sort_unstable();
        assert_eq!(review, vec!["de", "v1.2.3"]);
    }

    /// A candidate reports what archiving it would hide.
    #[tokio::test]
    async fn each_candidate_carries_its_blast_radius() {
        let (db, _dir) = test_db().await;
        let junk = db
            .store_entity("30 minutes", "concept", None, Some("post_ingest"), None)
            .await
            .unwrap();
        let real = db
            .store_entity("Rust", "technology", None, Some("post_ingest"), None)
            .await
            .unwrap();
        db.add_observation(&junk, "took half an hour", Some("post_ingest"), None)
            .await
            .unwrap();
        db.create_relation(
            &junk,
            &real,
            "related_to",
            Some("post_ingest"),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let found = collect_candidates(&db).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].links.observations, 1);
        assert_eq!(found[0].links.active_edges, 1);
        assert!(!found[0].links.is_empty());
    }

    /// Archiving hides the entity from every reader without deleting it, and
    /// leaves the real entities alone.
    #[tokio::test]
    async fn archiving_a_candidate_hides_it_but_keeps_the_real_ones() {
        let (db, _dir) = test_db().await;
        db.store_entity("30 minutes", "concept", None, Some("post_ingest"), None)
            .await
            .unwrap();
        db.store_entity("Alice Chen", "person", None, Some("post_ingest"), None)
            .await
            .unwrap();

        let candidates = collect_candidates(&db).await.unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(db.archive_entity(&candidates[0].entity_id).await.unwrap());

        let remaining: Vec<String> = db
            .list_entities(None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(remaining, vec!["Alice Chen".to_string()]);
        assert!(
            db.search_entities_by_name("30 minutes")
                .await
                .unwrap()
                .is_empty(),
            "an archived entity must not resolve by name"
        );
        assert!(
            collect_candidates(&db).await.unwrap().is_empty(),
            "a second run must find nothing left to do"
        );
    }

    /// Archiving is idempotent: a second call on the same entity reports no
    /// change rather than erroring.
    #[tokio::test]
    async fn archiving_twice_is_a_no_op() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("17%", "concept", None, Some("post_ingest"), None)
            .await
            .unwrap();
        assert!(db.archive_entity(&id).await.unwrap());
        assert!(!db.archive_entity(&id).await.unwrap());
        assert!(!db.archive_entity("ent_does_not_exist").await.unwrap());
    }
}
