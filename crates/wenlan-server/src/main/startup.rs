// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Keep each connection-holding reconciliation slice short enough for the
/// post-serve continuation to yield between slices.
pub(super) const SUPPORT_RECONCILE_BATCH: usize = 25;

pub(super) struct PreparedStartupState {
    pub(super) server_state: ServerState,
    pub(super) db_arc: Arc<wenlan_core::db::MemoryDB>,
    pub(super) repair_recovery_pending: bool,
    pub(super) config: wenlan_core::config::Config,
    pub(super) reranker_cache_dir: Option<std::path::PathBuf>,
    pub(super) deep_bgebase_pending: bool,
}

pub(super) async fn prepare_startup_state(
    wenlan_root: std::path::PathBuf,
    data_dir: std::path::PathBuf,
    brief_status_root: std::path::PathBuf,
    startup_repair_claim: &Option<StartupRepairClaim>,
    startup_repair_claimed: bool,
) -> anyhow::Result<PreparedStartupState> {
    let repair_store = wenlan_core::repair::RepairArtifactStore::new(wenlan_root.join("repairs"));
    if let Some(claim) = startup_repair_claim.as_ref() {
        validate_startup_repair_claim(&repair_store, claim)?;
        tracing::warn!(
            manifest_id = claim.manifest_id(),
            "validated exact repair-only startup claim before opening the database"
        );
    }

    // Inspect durable repair state before opening the database. A prepared
    // exact claim and any applied-unverified receipt must identify the same
    // manifest; otherwise startup fails without touching canonical data.
    let pending_repairs = repair_store
        .pending_verification_manifest_ids()
        .map_err(|error| anyhow::anyhow!("inspect durable repair state: {error}"))?;
    let startup_repair_fence =
        select_startup_repair_fence(&pending_repairs, startup_repair_claim.as_ref())?;
    let repair_recovery_pending = startup_repair_fence.is_some();

    // One-time origin -> wenlan migration is an ordinary startup write. Run it
    // only after durable repair inspection has proved no repair fence exists.
    if !repair_recovery_pending && wenlan_core::env_compat::var_compat("WENLAN_DATA_DIR").is_none()
    {
        if let Some(dl) = dirs::data_local_dir() {
            wenlan_core::migrate_rename::migrate_and_log(&dl.join("origin"), &dl.join("wenlan"));
        }
    }
    if !repair_recovery_pending {
        if let Some(home) = dirs::home_dir() {
            wenlan_core::migrate_rename::migrate_and_log(
                &home.join(".origin"),
                &home.join(".wenlan"),
            );
        }
    }

    // Build state and restore the process-local fence while recovery is still
    // sealed. No background acquisition can start before `finish_recovery`.
    let mut server_state = ServerState::new();
    server_state.brief_status_root = Some(brief_status_root.clone());
    server_state.optional_runtime_workers_suspended = repair_recovery_pending;
    server_state.repair_root = Some(repair_store.root().to_path_buf());
    server_state.presence_root = Some(wenlan_root.clone());
    let startup_repair_authority = match startup_repair_claim.as_ref() {
        Some(claim) => Some(claim.apply_request()?),
        None => startup_repair_fence
            .as_deref()
            .map(|manifest_id| stored_repair_apply_request(&repair_store, manifest_id))
            .transpose()?,
    };
    if let Some(request) = startup_repair_authority {
        let manifest_id = request.manifest_id().to_string();
        server_state
            .maintenance_coordinator
            .rearm_approved_repair(request)
            .map_err(|error| anyhow::anyhow!("restore exact repair writer fence: {error}"))?;
        tracing::warn!(
            manifest_id,
            prepared_claim = startup_repair_claimed,
            "restored exact repair authority before startup writers"
        );
    }
    // Repair mode refuses schema drift and skips every ordinary constructor
    // side effect (schema/migrations/profile bootstrap/embedder load). Normal
    // startup retains the existing fully initialized path.
    let db = if repair_recovery_pending {
        tracing::warn!("Opening current database in side-effect-free repair mode");
        wenlan_core::db::MemoryDB::open_for_repair(&data_dir).await?
    } else {
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::NoopEmitter);
        tracing::info!("Initializing MemoryDB at {}", data_dir.display());
        wenlan_core::db::MemoryDB::new(&data_dir, emitter).await?
    };
    let db_arc = Arc::new(db);
    server_state.db = Some(db_arc.clone());

    // Legacy Markdown is imported once into daemon-owned Brief state. The source
    // remains untouched until a later successful update replaces it with a receipt.
    if !repair_recovery_pending {
        match brief_files::import_legacy_status_files(&db_arc, &brief_status_root).await {
            Ok(report) => {
                if report.imported > 0 {
                    tracing::info!("[brief] imported {} legacy Space Brief(s)", report.imported);
                }
                for warning in report.warnings {
                    tracing::warn!("[brief] legacy import: {warning}");
                }
            }
            Err(error) => tracing::warn!("[brief] legacy import failed: {error}"),
        }
    }

    // Run migration-55 backfill (event_date regex Pass A + memory_entities Pass B)
    // before the HTTP listener binds so no ingest races the backfill. Idempotent.
    if repair_recovery_pending {
        tracing::warn!("skipping first-boot data backfill until repair verification completes");
    } else {
        tracing::info!(
            "Running first-boot data backfill (event dates + knowledge-graph links); \
             this can take a moment on large databases…"
        );
        let m55 = db_arc.run_migration_55().await.map_err(|e| {
            anyhow::anyhow!("running migration 55 (event_date + memory_entities backfill): {e}")
        })?;
        tracing::info!(
            "First-boot backfill complete: scanned {} memories for dates, inserted {} entity links",
            m55.event_dates_scanned,
            m55.entity_links_inserted
        );
    }

    // Requeue any document-enrichment rows left `in_progress` by a previous run
    // (a crash / restart mid-enrichment). Their per-chunk checkpoint is
    // preserved, so the scheduler resumes them from where they stopped rather
    // than re-analyzing from scratch — restart-from-checkpoint, no manual step.
    if !repair_recovery_pending {
        match db_arc.reset_in_progress_documents().await {
            Ok(0) => {}
            Ok(n) => tracing::info!("[doc-enrich] requeued {n} in-progress document(s) for resume"),
            Err(e) => tracing::warn!("[doc-enrich] reset_in_progress_documents failed: {e}"),
        }
    }

    // Consolidate user-facing assets under ~/.wenlan/.
    // - Ensure ~/.wenlan/{pages, sessions, sessions/_status} exist
    // - Symlink ~/.wenlan/db -> <data_dir> (cosmetic alias; DB stays at
    //   the platform data directory (resolved via `dirs::data_local_dir()` per OS)
    //   under `wenlan/memorydb/`, to avoid moving live SQLite/WAL files mid-flight).
    // - Migrate legacy ~/Origin/knowledge/ md files into ~/.wenlan/pages/ if
    //   the new dir is empty. Never deletes the old dir; user can clean up
    //   manually after verifying.
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        if let Some(home) = dirs::home_dir() {
            let wenlan_dot = home.join(".wenlan");
            for sub in ["pages", "sessions", "sessions/_status"] {
                if let Err(e) = std::fs::create_dir_all(wenlan_dot.join(sub)) {
                    tracing::warn!("[wenlan-dir] create {} failed: {}", sub, e);
                }
            }

            let db_link = wenlan_dot.join("db");
            let link_target_already_correct = std::fs::read_link(&db_link)
                .map(|t| t == data_dir)
                .unwrap_or(false);
            if !link_target_already_correct && !db_link.exists() {
                #[cfg(unix)]
                if let Err(e) = std::os::unix::fs::symlink(&data_dir, &db_link) {
                    tracing::warn!(
                        "[wenlan-dir] symlink {} -> {} failed: {}",
                        db_link.display(),
                        data_dir.display(),
                        e
                    );
                }
                #[cfg(windows)]
                {
                    tracing::info!(
                        "Database at {} (no shortcut created; Windows symlinks require admin).",
                        data_dir.display()
                    );
                }
            }

            let legacy_pages = home.join("Origin/knowledge");
            let new_pages = wenlan_dot.join("pages");
            let legacy_has_md = std::fs::read_dir(&legacy_pages)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
                })
                .unwrap_or(false);
            let new_is_empty = std::fs::read_dir(&new_pages)
                .map(|entries| {
                    !entries
                        .filter_map(|e| e.ok())
                        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
                })
                .unwrap_or(true);
            if startup_projection_writes_allowed(repair_recovery_pending)
                && legacy_has_md
                && new_is_empty
            {
                tracing::info!(
                    "[migrate] copying md files from {} to {}",
                    legacy_pages.display(),
                    new_pages.display()
                );
                if let Ok(entries) = std::fs::read_dir(&legacy_pages) {
                    let mut copied = 0usize;
                    for entry in entries.filter_map(|e| e.ok()) {
                        let src = entry.path();
                        if src.extension().and_then(|s| s.to_str()) != Some("md") {
                            continue;
                        }
                        if let Some(name) = src.file_name() {
                            let dst = new_pages.join(name);
                            if dst.exists() {
                                continue;
                            }
                            match std::fs::copy(&src, &dst) {
                                Ok(_) => copied += 1,
                                Err(e) => tracing::warn!(
                                    "[migrate] copy {} -> {} failed: {}",
                                    src.display(),
                                    dst.display(),
                                    e
                                ),
                            }
                        }
                    }
                    tracing::info!("[migrate] copied {} md files from legacy path", copied);
                }
            }

            // Initialize ~/.wenlan/ as a git repo so users get version history
            // of pages + sessions for free. Defensive — silent skip if git is
            // missing or any step fails. Skills (/handoff, /distill, /forget)
            // commit per logical batch; daemon only does the initial bring-up
            // here.
            let dot_git = wenlan_dot.join(".git");
            let git_available = std::process::Command::new("git")
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !dot_git.exists() && git_available {
                let gitignore = wenlan_dot.join(".gitignore");
                if !gitignore.exists() {
                    // No trailing slash on `db` / `bin` — those entries are
                    // symlinks in the consolidated layout, and pattern `db/`
                    // would only match real directories.
                    let _ = std::fs::write(
                        &gitignore,
                        "db\nbin\nlogs/\nsessions/_status/handoff-*.json\n",
                    );
                }
                let run = |args: &[&str]| {
                    std::process::Command::new("git")
                        .args(args)
                        .current_dir(&wenlan_dot)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status()
                        .ok()
                        .filter(|s| s.success())
                };
                if run(&["init", "--quiet"]).is_some() {
                    let _ = run(&[
                        "-c",
                        "user.name=Wenlan",
                        "-c",
                        "user.email=daemon@origin.local",
                        "commit",
                        "--allow-empty",
                        "--quiet",
                        "-m",
                        "Wenlan initialized",
                    ]);
                    let _ = run(&["add", "-A"]);
                    let _ = run(&[
                        "-c",
                        "user.name=Wenlan",
                        "-c",
                        "user.email=daemon@origin.local",
                        "commit",
                        "--quiet",
                        "-m",
                        "backfill: initial pages from DB",
                    ]);
                    tracing::info!("[wenlan-dir] git init complete at {}", wenlan_dot.display());
                }
            }
        }
    }

    // One-time backfill: if the knowledge directory is empty but the DB has
    // active pages, write them all to disk. Handles the case where pages were
    // created before KnowledgeWriter was wired up, or via a code path that
    // bypasses the writer.
    //
    // We gate on a `.origin/.backfill-attempted` marker file (created on
    // first attempt regardless of outcome) so this block only runs once per
    // daemon install. Without the marker, a persistent write_page
    // failure — e.g. permission error on the destination directory — would
    // re-trigger a full DB scan + write attempt on every single startup.
    {
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        let marker_path = knowledge_path.join(".wenlan").join(".backfill-attempted");

        let already_attempted = marker_path.exists();
        let has_md_files = !already_attempted
            && knowledge_path.exists()
            && std::fs::read_dir(&knowledge_path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .and_then(|s| s.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("md"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

        if startup_projection_writes_allowed(repair_recovery_pending)
            && !already_attempted
            && !has_md_files
        {
            match db_arc.list_pages("active", 10_000, 0).await {
                Ok(pages) if !pages.is_empty() => {
                    tracing::info!(
                        "[backfill] knowledge dir empty; writing {} pages to {}",
                        pages.len(),
                        knowledge_path.display()
                    );
                    let projection = wenlan_core::export::knowledge::KnowledgeProjectionWrite::new(
                        knowledge_path.clone(),
                        &db_arc,
                    );
                    let mut written = 0usize;
                    let mut failed = 0usize;
                    for page in &pages {
                        match projection.write_page_gated(&db_arc, page).await {
                            Ok(Some(_)) => written += 1,
                            // gated: not projected, not a failure
                            Ok(None) => {}
                            Err(e) => {
                                tracing::warn!(
                                    "[backfill] write_page failed for {}: {}",
                                    page.id,
                                    e
                                );
                                failed += 1;
                            }
                        }
                    }
                    tracing::info!("[backfill] wrote {} pages, {} failed", written, failed);

                    // Create the marker file so we don't re-run the
                    // backfill on every subsequent startup — even if every
                    // write_page above failed. The user can delete
                    // `.origin/.backfill-attempted` to force a retry.
                    if let Some(parent) = marker_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if let Err(e) = std::fs::write(&marker_path, "") {
                        tracing::warn!(
                            "[backfill] failed to write marker {}: {}",
                            marker_path.display(),
                            e
                        );
                    }
                }
                Ok(_) => {
                    // DB has no pages yet — nothing to backfill. Don't create
                    // the marker; the next startup after pages exist should retry.
                }
                Err(e) => {
                    tracing::warn!("[backfill] list_pages failed: {}", e);
                }
            }
        }
    }

    // Startup reconcile: repair the markdown projection from the DB.
    //
    // `write_page` renames a temp file over the target without an fsync — that
    // buys readers atomicity, not crash durability. So a crash can leave a
    // page's file missing, holding the previous version's bytes, or
    // zero-length, plus `.tmp` orphans from a write that died mid-rename.
    // This is the pass that makes "the md is a repairable projection" true.
    //
    // Runs synchronously, before `axum::serve`, for the same reason the
    // backfill above does: no HTTP write and no scheduler tick can race the
    // repair, so the pass needs no locking. The listener is already bound
    // (see the bind-first block up top), so a slow pass on a large corpus
    // delays serving, never the port handoff.
    //
    // ponytail: same 10k page ceiling as the backfill, and one pass reads
    // every projected file. If a corpus ever outgrows that, page the scan or
    // move it behind the listener — do NOT background it naively, since a
    // concurrent page write would race the repair.
    {
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        if knowledge_path.exists() {
            match db_arc.list_pages("active", 10_000, 0).await {
                Ok(pages) => {
                    let projection = wenlan_core::export::knowledge::KnowledgeProjectionWrite::new(
                        knowledge_path.clone(),
                        &db_arc,
                    );
                    match projection.reconcile(&pages) {
                        Ok(stats)
                            if stats.rewritten > 0
                                || stats.temp_files_removed > 0
                                || stats.errors > 0 =>
                        {
                            tracing::info!(
                                "[reconcile] projection repaired: {} checked, {} rewritten, \
                                 {} temp leftover(s) swept, {} failed",
                                stats.checked,
                                stats.rewritten,
                                stats.temp_files_removed,
                                stats.errors
                            );
                        }
                        Ok(stats) => {
                            tracing::debug!(
                                "[reconcile] {} page(s) checked, all clean",
                                stats.checked
                            );
                        }
                        Err(e) => tracing::warn!("[reconcile] pass failed: {e}"),
                    }

                    // A cutover ceremony that was killed mid-flight left the
                    // fence at `preparing`, and `preparing` refuses every page
                    // write. The ceremony cannot run while this daemon is up
                    // (it takes the data-root lock, probes the port, and
                    // refuses on a registered service unit), so a fence still
                    // reading `preparing` here belongs to a process that is
                    // gone. Release it before the invariant pass, so the pass
                    // and everything after it can write.
                    match db_arc.release_stranded_cutover_fence().await {
                        Ok(false) => {}
                        Ok(true) => tracing::warn!(
                            "[truth] a cutover ceremony did not finish; released its fence so \
                             page writes can proceed. Re-run `wenlan-server truth-cutover` if \
                             the cutover was intended."
                        ),
                        Err(e) => tracing::error!(
                            "[truth] could not read or release the cutover fence: {e}. Page \
                             writes may be refused until this is resolved."
                        ),
                    }

                    // M5 PR-B: the projection directory is the enforcement
                    // boundary for `wenlan pages`, which reads Markdown off
                    // disk and cannot negotiate a truth contract. Inert until
                    // the PR-C cutover — `page_visibility` answers `Full` for
                    // every page at generation 0, so this removes nothing and
                    // there is no generation branch here to weaken. It runs
                    // anyway, so the pass is live rather than proven-and-
                    // unwired, and it runs AFTER reconcile so it evicts from a
                    // directory that is already consistent with the DB.
                    match projection
                        .enforce_projection_directory_invariant(&db_arc)
                        .await
                    {
                        Ok(0) => {}
                        Ok(removed) => tracing::info!(
                            "[truth] projection invariant evicted {removed} unsupported page(s)"
                        ),
                        // M5 PR-C item 4. A failure means a file the reader may
                        // not see is still on disk. `wenlan pages` reads that
                        // directory directly — there is no HTTP response to
                        // filter and no wire gate in front of it — so at
                        // generation >= 1 the daemon refuses to open a door it
                        // cannot hold. At generation 0 the pass removes nothing,
                        // so a failure records the absence of a restriction that
                        // is not in force: `error!` and serve, as before.
                        //
                        // Refusing to start is acceptable rather than a brick
                        // because advancing the generation is a deliberate
                        // ceremony with an operator present.
                        Err(e) => {
                            let live = db_arc.truth_cutover_generation().await.unwrap_or(1) != 0;
                            if live {
                                let msg = format!(
                                    "[truth] projection invariant pass failed at cutover \
                                     generation >= 1; refusing to serve page traffic: {e}"
                                );
                                report_bootstrap_error(&wenlan_root, &msg);
                                return Err(anyhow::anyhow!(msg));
                            }
                            tracing::error!("[truth] projection invariant pass failed: {e}");
                        }
                    }
                }
                Err(e) => tracing::warn!("[reconcile] list_pages failed: {e}"),
            }
        }
    }

    // M5: finish the claim-derivation backlog the migration could only start.
    //
    // Migration 105 sweeps one bounded batch, because it runs holding the single
    // connection's mutex and a full scan of a large vault would delay boot. That
    // bound truncates rather than defers unless something continues it: a vault
    // with more pages than the batch leaves the remainder reachable only by
    // being edited, and the derivation queue is then quietly partial on exactly
    // the installs the backlog scan exists for.
    //
    // Once per boot is the right cadence and not merely the convenient one. The
    // three inputs that can strand a page here — a new binary's
    // EXTRACTOR_VERSION, a new binary's SUPPORT_THRESHOLD, and an interrupted
    // migration — are all things that change across a restart and not during
    // one. Everything that moves while the daemon runs already has a trigger.
    //
    // Fenced, because the drain WRITES. Repair recovery opens the database
    // through `open_for_repair`, which deliberately performs no ordinary
    // constructor side effect, and the whole point of that mode is that an
    // operator inspecting or repairing a damaged database sees only what the
    // repair itself does. The drain deletes stale `done` jobs and inserts
    // pending ones, so leaving it unfenced put queue mutations into exactly the
    // startup that promised none. The backlog is not lost — it is a boot-time
    // sweep, so the next ordinary start performs it.
    // The drain also DEMOTES. Row 15's whole point is that a page whose support
    // stopped clearing the live bar must stop asserting `supported` now rather
    // than whenever a worker reaches it, and `reconcile_supported_pages` extends
    // that to the conditions no SQL scan can see — a missing or stale marker, an
    // extractor bump, a citation whose bytes moved. Both are reconciliation, so
    // both are treated the same way when they fail.
    //
    // BOUNDED, and it has to be. The reconcile pass holds the single connection
    // every foreground request needs and spends a multi-query evaluator per
    // page, per claim, per candidate; run to completion here it would occupy
    // the port for as long as a large supported vault takes, which is the very
    // thing the batch above the drain exists to avoid. So it runs in batches
    // under a wall-clock deadline, and what it does not reach is withheld
    // rather than served: the durable cursor it leaves makes every unproved
    // `supported` read `Unevaluated` until a later batch gets there. Truncating
    // WITHOUT that gate would serve exactly the stale state the pass retracts.
    //
    // The deadline is a boot-latency budget, not a correctness one. Overrunning
    // it costs re-derivation and a slower path back to `supported`; it can
    // never publish an unproved verdict.
    if !repair_recovery_pending {
        const RECONCILE_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

        // Start a new pass after a completed one, or resume an incomplete pass
        // under the same ruleset. The durable ruleset binds the cursor to the
        // extractor, threshold, and reconciliation algorithm that earned it;
        // a changed rule restarts from the beginning, while a short-lived
        // process cannot starve a large vault by repeatedly proving one prefix.
        let drained = match db_arc.drain_stale_derivation_jobs(500).await {
            Ok(enqueued) => db_arc
                .begin_support_reconcile_pass()
                .await
                .map(|()| enqueued),
            Err(e) => Err(e),
        };
        let reconciled = match drained {
            Ok(enqueued) => {
                let deadline = std::time::Instant::now() + RECONCILE_BUDGET;
                let mut demoted = 0usize;
                let mut outcome = Ok(());
                loop {
                    match db_arc
                        .reconcile_supported_pages(SUPPORT_RECONCILE_BATCH)
                        .await
                    {
                        Ok(pass) => {
                            demoted += pass.demoted;
                            if pass.complete {
                                break;
                            }
                            if std::time::Instant::now() >= deadline {
                                tracing::info!(
                                    "[truth] support reconciliation paused at its boot budget; \
                                     unproved pages read as unevaluated until the runtime worker \
                                     resumes"
                                );
                                break;
                            }
                        }
                        Err(e) => {
                            outcome = Err(e);
                            break;
                        }
                    }
                }
                outcome.map(|()| (enqueued, demoted))
            }
            Err(e) => Err(e),
        };
        match reconciled {
            Ok((0, 0)) => {}
            Ok((enqueued, demoted)) => tracing::info!(
                "[truth] claim-derivation reconciliation: {enqueued} page(s) enqueued, \
                 {demoted} demoted out of supported"
            ),
            // Fail CLOSED, on the same rule and through the same helper as the
            // projection invariant above. This pass is what withdraws a stored
            // `supported` whose evidence no longer holds; when it does not
            // complete, the daemon does not know which pages those are, and
            // logging and serving anyway means serving exactly the stale
            // `supported` state the pass exists to retract. At generation 0
            // every adapter is pass-through and no truth state reaches a
            // reader, so the failure records the absence of a restriction that
            // is not in force: `error!` and serve.
            Err(e) => {
                let live = db_arc.truth_cutover_generation().await.unwrap_or(1) != 0;
                if live {
                    let msg = format!(
                        "[truth] claim-derivation reconciliation failed at cutover generation \
                         >= 1; refusing to serve possibly-stale supported state: {e}"
                    );
                    report_bootstrap_error(&wenlan_root, &msg);
                    return Err(anyhow::anyhow!(msg));
                }
                tracing::error!("[truth] claim-derivation reconciliation failed: {e}");
            }
        }
    }

    // Load intelligence config
    server_state.prompts = wenlan_core::prompts::PromptRegistry::load(
        &wenlan_core::prompts::PromptRegistry::override_dir(),
    );
    server_state.tuning =
        wenlan_core::tuning::TuningConfig::load(&wenlan_core::tuning::TuningConfig::config_path());
    server_state.quality_gate =
        wenlan_core::quality_gate::QualityGate::new(server_state.tuning.gate.clone());

    // Load API LLM providers if configured
    let config = wenlan_core::config::load_config();
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        if let Some(ref key) = config.anthropic_api_key {
            if !key.is_empty() {
                let routine_model = config.routine_model.clone().unwrap_or_else(|| {
                    wenlan_core::llm_provider::DEFAULT_ROUTINE_MODEL.to_string()
                });
                let provider =
                    wenlan_core::llm_provider::ApiProvider::new(key.clone(), routine_model);
                server_state.api_llm = Some(Arc::new(provider));
                tracing::info!("API LLM provider initialized (routine)");

                let synthesis_model = config
                    .synthesis_model
                    .clone()
                    .unwrap_or_else(|| "claude-sonnet-4-6".to_string());
                let provider =
                    wenlan_core::llm_provider::ApiProvider::new(key.clone(), synthesis_model);
                server_state.synthesis_llm = Some(Arc::new(provider));
                tracing::info!("Synthesis LLM provider initialized");
            }
        }

        // Load external LLM provider if configured
        if let (Some(ref endpoint), Some(ref model)) =
            (&config.external_llm_endpoint, &config.external_llm_model)
        {
            if !endpoint.is_empty() && !model.is_empty() {
                let provider = wenlan_core::llm_provider::OpenAICompatibleProvider::new_with_key(
                    endpoint.clone(),
                    model.clone(),
                    config.external_llm_api_key.clone(),
                );
                server_state.external_llm = Some(Arc::new(provider));
                tracing::info!("External LLM provider initialized from config");
            }
        }
    }

    // Cross-encoder reranker wiring. `WENLAN_RERANKER_MODE = off|lite|full` (default
    // off) selects which retrieval paths get a CE and which model; the legacy
    // `WENLAN_RERANKER_ENABLED=1` (with MODE unset) maps to deep-only CE using the
    // configured model — exactly the pre-mode behavior. First construction downloads
    // weights (turbo ~146MB, bge-base ~1.1GB) into the shared FastEmbed cache;
    // failure is non-fatal (the affected path falls back to embedding+FTS ordering).
    let reranker_cache_dir = wenlan_core::db::resolve_fastembed_cache_dir(&data_dir);
    let mut deep_bgebase_pending = false;
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        use wenlan_core::reranker::{RerankerMode, RerankerPick};
        use wenlan_types::responses::RerankerStatus;
        let mode = wenlan_core::reranker::reranker_mode_resolved(&config);
        let legacy_enabled = std::env::var("WENLAN_RERANKER_ENABLED").as_deref() == Ok("1");
        let plan = wenlan_core::reranker::resolve_reranker_plan(mode, legacy_enabled);
        server_state.reranker_mode = match mode {
            RerankerMode::Off => "off",
            RerankerMode::Lite => "lite",
            RerankerMode::Full => "full",
        }
        .to_string();
        tracing::info!(
            "[reranker] mode={} (legacy_enabled={legacy_enabled}); light={:?} deep={:?}",
            server_state.reranker_mode,
            plan.light,
            plan.deep
        );

        // Light paths (quick `/api/search` + context `/api/context`): turbo
        // (~146MB), eager-load — small enough not to meaningfully block startup.
        let mut light_reranker: Option<Arc<dyn wenlan_core::reranker::Reranker>> = None;
        if let Some(pick) = plan.light {
            let cache = reranker_cache_dir.clone();
            match tokio::task::spawn_blocking(move || {
                wenlan_core::reranker::init_cross_encoder_reranker_pick(pick, cache)
            })
            .await
            {
                Ok(Ok(r)) => {
                    let model_id = r.model_id().to_string();
                    tracing::info!("[reranker] light paths active (model={model_id})");
                    server_state.reranker_light_status = RerankerStatus::Active { model_id };
                    light_reranker = Some(r.clone());
                    server_state.reranker_light = Some(r);
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "[reranker] light init failed; quick + context fall back to plain hybrid: {e}"
                    );
                    server_state.reranker_light_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                }
                Err(e) => {
                    tracing::warn!("[reranker] light init join failed: {e}");
                    server_state.reranker_light_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                }
            }
        }

        // Deep path (`/api/memory/search` with rerank=true).
        match plan.deep {
            // Back-compat: ENABLED=1 + mode unset -> eager-load the configured model
            // (+ BYO via WENLAN_RERANKER_ONNX_DIR), blocking startup, exactly as before.
            Some(RerankerPick::Configured) => {
                tracing::info!(
                    "[reranker] deep path (legacy WENLAN_RERANKER_ENABLED); first run downloads \
                     weights (~1.1GB). The daemon finishes starting once the model is ready\u{2026}"
                );
                let cache = reranker_cache_dir.clone();
                match tokio::task::spawn_blocking(move || {
                    wenlan_core::reranker::init_cross_encoder_reranker(cache)
                })
                .await
                {
                    Ok(Ok(r)) => {
                        let model_id = r.model_id().to_string();
                        tracing::info!("[reranker] deep path active (model={model_id})");
                        server_state.reranker_status = RerankerStatus::Active { model_id };
                        server_state.reranker = Some(r);
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(
                            "[reranker] deep init failed; rerank=true falls back to plain hybrid: {e}"
                        );
                        server_state.reranker_status = RerankerStatus::Failed {
                            reason: e.to_string(),
                        };
                    }
                    Err(e) => {
                        tracing::warn!("[reranker] deep init join failed: {e}");
                        server_state.reranker_status = RerankerStatus::Failed {
                            reason: e.to_string(),
                        };
                    }
                }
            }
            // lite: the deep path reuses the already-loaded turbo (no second load).
            // Mirror the light status either way so a FAILED turbo load surfaces as
            // deep=failed (not a misleading deep=disabled) on /api/status; the missing
            // Arc still makes rerank=true fall back to plain hybrid. (review fix)
            Some(RerankerPick::Turbo) => {
                server_state.reranker_status = server_state.reranker_light_status.clone();
                if let Some(r) = light_reranker.clone() {
                    server_state.reranker = Some(r);
                }
            }
            // full: heavy bge-base. Council fix #3 — do NOT block startup; load it in
            // the background after the state is shared (rerank=true falls back to plain
            // until ready). Status stays Disabled until the background load completes.
            Some(RerankerPick::BgeBase) => {
                deep_bgebase_pending = true;
            }
            None => {}
        }
    }

    // One-time compatibility import for the pre-daemon top-level
    // `default = "..."` key. The file remains user-owned mapping config and
    // is never rewritten; the durable watermark prevents stale older clients
    // from resurrecting it after this first check.
    if !repair_recovery_pending {
        if let Some(home) = dirs::home_dir() {
            let legacy_default_path = home.join(".wenlan/spaces.toml");
            match db_arc
                .import_legacy_default_once(&legacy_default_path)
                .await
            {
                Ok(outcome) => {
                    if let Some(space) = outcome.imported_space {
                        tracing::info!(
                            "[startup] imported legacy Default save space '{}'",
                            space.name
                        );
                    } else if let Some(name) = outcome.invalid_name {
                        tracing::warn!(
                            "[startup] legacy Default save space '{}' is not registered; \
                             leaving Default save space unset. Run `wenlan spaces add {}` \
                             and then `wenlan spaces default {}`.",
                            name,
                            name,
                            name
                        );
                    }
                }
                Err(error) => {
                    tracing::warn!("[startup] legacy Default save space import failed: {error}")
                }
            }
        }
    }

    // Import any legacy tag data from the pre-PR-B2 spaces.db file.
    if !repair_recovery_pending {
        match wenlan_core::spaces::import_legacy_tags(&db_arc).await {
            Ok(n) if n > 0 => {
                tracing::info!("[startup] imported {} legacy tag triples from spaces.db", n)
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("[startup] legacy tags import failed: {e}"),
        }
    }

    // Spawn the ingest coalescer. HTTP `/api/memory/store` handlers submit
    // fully-built RawDocuments + pre-computed chunk counts; the coalescer
    // runs the full ingest pipeline (batched quality gate, partition,
    // upsert survivors) per flush window. This amortizes both the FastEmbed
    // invocation (one batched call per flush for gate's novelty check) AND
    // the libSQL transaction (one per flush for the survivors) across
    // concurrent writes.
    //
    // See `crates/wenlan-server/src/ingest_batcher.rs` for the design and
    // contract tests.
    {
        let db_for_batcher = db_arc.clone();
        let gate_for_batcher = server_state.quality_gate.clone();
        let maintenance_for_batcher = server_state.maintenance_coordinator.clone();
        let process: ingest_batcher::BatchProcessFn = Arc::new(
            move |items: Vec<(
                wenlan_core::sources::RawDocument,
                usize,
                Option<wenlan_core::space_context::ResolvedWriteSpace>,
            )>| {
                let db = db_for_batcher.clone();
                let gate = gate_for_batcher.clone();
                let maintenance = maintenance_for_batcher.clone();
                Box::pin(async move {
                    let _maintenance_guard = maintenance.begin_background().await;
                    ingest_batch_process(db, gate, items).await
                })
            },
        );
        server_state.ingest_batcher = Some(ingest_batcher::IngestBatcher::spawn(
            process,
            ingest_batcher::BatcherConfig::default(),
        ));
    }
    Ok(PreparedStartupState {
        server_state,
        db_arc,
        repair_recovery_pending,
        config,
        reranker_cache_dir,
        deep_bgebase_pending,
    })
}
