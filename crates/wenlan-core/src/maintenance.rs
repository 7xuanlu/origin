// SPDX-License-Identifier: Apache-2.0

mod duplicates;
mod page_merge_order;

#[cfg(test)]
mod survivor_tests;

use std::path::Path;
use std::sync::Arc;

use crate::db::{AutomaticRetroPageScan, MemoryDB, StalePageCursor};
use crate::error::WenlanError;
use crate::llm_provider::LlmProvider;
use crate::post_write::page_is_human_owned;
use crate::prompts::PromptRegistry;
use crate::synthesis::distill::{
    automatic_refresh_exceeds_source_cap, refresh_page, RefreshReason,
    AUTOMATIC_PAGE_REFRESH_SOURCE_CAP,
};
use wenlan_types::pages::Page;

const RETRO_SWEEP_COMPLETE_KEY: &str = "maintenance_retro_sweep_v1_complete";
const RETRO_SWEEP_PAUSE_KEY: &str = "maintenance_retro_sweep_v1_pause";
// v1 stored a numeric OFFSET that cannot identify a row after stale-set
// mutation. v2 deliberately starts a fresh keyset pass on upgrade.
const MAINTENANCE_STALE_CURSOR_KEY: &str = "maintenance_stale_page_cursor_v2";
const AUTOMATIC_RETRO_CURSOR_KEY: &str = "automatic_maintenance_retro_cursor_v1";
const AUTOMATIC_RETRO_COMPLETE_KEY: &str = "automatic_maintenance_retro_complete_v1";
const AUTOMATIC_NEAR_DUPLICATE_CURSOR_KEY: &str = "automatic_maintenance_near_duplicate_cursor_v1";
const AUTOMATIC_CROSS_SPACE_CURSOR_KEY: &str = "automatic_maintenance_cross_space_cursor_v1";
const AUTOMATIC_DISCOVERY_SEED_BUDGET: usize = 8;
const AUTOMATIC_DISCOVERY_NEIGHBOR_BUDGET: usize = 64;
const STUB_PAGE_SOURCE_FLOOR: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceStage {
    RetroReview,
    NearDuplicate,
    OrphanInventory,
    CrossSpaceDiscovery,
    StalePage,
    Overview,
}

impl MaintenanceStage {
    pub const ALL: &'static [Self] = &[
        Self::RetroReview,
        Self::NearDuplicate,
        Self::OrphanInventory,
        Self::CrossSpaceDiscovery,
        Self::StalePage,
        Self::Overview,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RetroReview => "retro_review",
            Self::NearDuplicate => "near_duplicate",
            Self::OrphanInventory => "orphan_inventory",
            Self::CrossSpaceDiscovery => "cross_space_discovery",
            Self::StalePage => "stale_page",
            Self::Overview => "overview",
        }
    }
}

impl std::fmt::Display for MaintenanceStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct MaintenanceTickConfig {
    pub page_match_threshold: f64,
    pub formation_threshold: f64,
    pub page_min_cluster_size: usize,
    pub token_limit: usize,
    pub max_unlinked_cluster_size: usize,
    pub max_grouped_cluster_size: usize,
    pub max_per_tick: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MaintenanceTickResult {
    pub merge_cards_emitted: usize,
    pub stale_machine_refreshed: usize,
    pub stale_machine_queued: usize,
    pub stale_human_cards: usize,
    pub stale_human_queued: usize,
    pub orphan_labels_checked: usize,
    pub overview_refreshed: usize,
    pub discovery_cards_emitted: usize,
    pub retro_expected_card_volume: usize,
    pub retro_cards_emitted: usize,
    pub retro_stub_cards_emitted: usize,
    pub retro_paused: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceSliceReport {
    pub stage: MaintenanceStage,
    pub result: MaintenanceTickResult,
    pub selected: bool,
    pub progressed: bool,
    pub more: bool,
    pub retryable: bool,
    pub paused: bool,
    pub work: MaintenanceSliceWork,
}

/// Observable proof that an automatic maintenance slice stayed inside its
/// cooperative work envelope. These are rows/items actually inspected, not
/// merely results emitted after an unbounded query.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MaintenanceSliceWork {
    pub pages_examined: usize,
    pub pairs_examined: usize,
    pub source_rows_examined: usize,
    pub seeds_examined: usize,
    pub eligible_seeds_probed: usize,
    pub neighbor_rows_examined: usize,
    pub fully_filtered_seeds: usize,
    pub truncated: bool,
}

pub async fn run_maintenance_tick(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    config: &MaintenanceTickConfig,
    knowledge_path: Option<&Path>,
) -> Result<MaintenanceTickResult, WenlanError> {
    let mut result = MaintenanceTickResult::default();
    let max_per_tick = config.max_per_tick.max(1);

    let retro = run_retro_sweep(db, config, max_per_tick).await?;
    result.retro_expected_card_volume = retro.expected_card_volume;
    result.retro_cards_emitted = retro.cards_emitted;
    result.retro_stub_cards_emitted = retro.stub_cards_emitted;
    result.retro_paused = retro.paused;
    result.merge_cards_emitted += retro.merge_cards_emitted;

    let near_duplicates =
        duplicates::detect_near_duplicate_pages(db, config.page_match_threshold, max_per_tick)
            .await?;
    for pair in near_duplicates.iter().take(max_per_tick) {
        if emit_page_merge_card(db, pair).await? {
            result.merge_cards_emitted += 1;
        }
    }

    result.orphan_labels_checked = db.list_orphan_link_labels(1).await?.len();

    let discovery_clusters = db
        .find_cross_space_distillation_clusters(
            config.formation_threshold,
            config.page_min_cluster_size,
            max_per_tick,
            config.token_limit,
            config.max_unlinked_cluster_size,
            config.max_grouped_cluster_size,
        )
        .await?;
    for cluster in discovery_clusters.iter().take(max_per_tick) {
        if emit_cross_space_discovery_card(db, &cluster.source_ids).await? {
            result.discovery_cards_emitted += 1;
        }
    }

    let Some(provider) = llm.filter(|provider| provider.is_available()) else {
        let stale = db.list_stale_pages("source_updated").await?;
        for page in stale.iter().take(max_per_tick) {
            if page_is_human_owned(page) {
                result.stale_human_queued += 1;
            } else {
                result.stale_machine_queued += 1;
            }
        }
        return Ok(result);
    };

    let stale = db.list_stale_pages("source_updated").await?;
    for page in stale.iter().take(max_per_tick) {
        let human_owned = page_is_human_owned(page);
        let outcome = refresh_page(
            db,
            provider,
            prompts,
            &page.id,
            RefreshReason::SourceChanged,
            knowledge_path,
        )
        .await?;
        if human_owned || outcome.gated {
            result.stale_human_cards += usize::from(outcome.gated);
        } else {
            result.stale_machine_refreshed += usize::from(outcome.wrote);
        }
    }

    let overview = crate::synthesis::overview::refresh_overview_page(
        db,
        provider,
        prompts,
        "maintenance",
        knowledge_path,
    )
    .await?;
    result.overview_refreshed = usize::from(overview.wrote);

    Ok(result)
}

/// Run one maintenance stage and at most one emitted/refreshed item. The
/// explicit full-tick API above remains available to foreground callers; the
/// automatic scheduler uses this cooperative boundary.
async fn load_maintenance_stale_cursor(
    db: &MemoryDB,
) -> Result<Option<StalePageCursor>, WenlanError> {
    Ok(db
        .get_app_metadata(MAINTENANCE_STALE_CURSOR_KEY)
        .await?
        .and_then(|value| serde_json::from_str(&value).ok()))
}

async fn persist_maintenance_stale_cursor(
    db: &MemoryDB,
    cursor: Option<&StalePageCursor>,
) -> Result<(), WenlanError> {
    let value = cursor
        .map(serde_json::to_string)
        .transpose()?
        .unwrap_or_default();
    db.set_app_metadata(MAINTENANCE_STALE_CURSOR_KEY, &value)
        .await
}

pub async fn run_maintenance_stage_slice(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    config: &MaintenanceTickConfig,
    knowledge_path: Option<&Path>,
    stage: MaintenanceStage,
) -> Result<MaintenanceSliceReport, WenlanError> {
    let mut result = MaintenanceTickResult::default();
    let mut work = MaintenanceSliceWork::default();
    let mut selected = false;
    let mut progressed = false;
    let mut more = false;
    let mut retryable = false;
    let mut paused = false;

    match stage {
        MaintenanceStage::RetroReview => {
            if db.has_pending_retro_review().await? {
                paused = true;
                result.retro_paused = true;
            } else if db
                .get_app_metadata(AUTOMATIC_RETRO_COMPLETE_KEY)
                .await?
                .as_deref()
                != Some("1")
            {
                // The automatic retro lane audits one machine Page at a time.
                // Near-duplicate history belongs to the following recurring
                // stage, so this one-time backfill does not repeat that scan.
                let cursor = db
                    .get_app_metadata(AUTOMATIC_RETRO_CURSOR_KEY)
                    .await?
                    .filter(|value| !value.is_empty());
                let slice: AutomaticRetroPageScan = db
                    .scan_automatic_retro_stub_slice(cursor.as_deref(), STUB_PAGE_SOURCE_FLOOR)
                    .await?;
                selected = slice.next_cursor.is_some();
                progressed = selected;
                more = slice.more;
                work.pages_examined = usize::from(selected);
                work.source_rows_examined = slice.eligible_source_count.unwrap_or(0);
                let candidate = slice
                    .eligible_source_count
                    .filter(|source_count| *source_count < STUB_PAGE_SOURCE_FLOOR)
                    .zip(slice.next_cursor.as_ref())
                    .map(|(source_count, page_id)| StubPageCandidate {
                        page_id: page_id.clone(),
                        source_count,
                    });
                if let Some(candidate) = candidate {
                    result.retro_expected_card_volume = 1;
                    let emitted =
                        emit_keep_or_archive_card(db, &candidate.page_id, candidate.source_count)
                            .await?;
                    result.retro_cards_emitted = usize::from(emitted);
                    result.retro_stub_cards_emitted = usize::from(emitted);
                    paused = emitted;
                    result.retro_paused = emitted;
                }
                if let Some(cursor) = slice.next_cursor {
                    db.set_app_metadata(AUTOMATIC_RETRO_CURSOR_KEY, &cursor)
                        .await?;
                }
                if !more {
                    db.set_app_metadata(AUTOMATIC_RETRO_COMPLETE_KEY, "1")
                        .await?;
                }
            }
        }
        MaintenanceStage::NearDuplicate => {
            if db.has_pending_retro_review().await? {
                paused = true;
            } else {
                let cursor = db
                    .get_app_metadata(AUTOMATIC_NEAR_DUPLICATE_CURSOR_KEY)
                    .await?
                    .filter(|value| !value.is_empty())
                    .and_then(|value| {
                        serde_json::from_str::<duplicates::NearDuplicateCursor>(&value).ok()
                    });
                let reader = db.begin_near_duplicate_slice_reader().await;
                let pair_rows = reader
                    .scan_near_duplicate_slice(
                        cursor
                            .as_ref()
                            .map(|cursor| (cursor.left_id.as_str(), cursor.right_id.as_str())),
                        duplicates::AUTOMATIC_PAIR_BUDGET,
                    )
                    .await?;
                let slice = duplicates::evaluate_near_duplicate_slice(
                    &reader,
                    pair_rows,
                    config.page_match_threshold,
                )
                .await?;
                drop(reader);
                selected = slice.pairs_examined > 0;
                progressed = selected;
                more = slice.more;
                work.pages_examined = slice.pages_examined;
                work.pairs_examined = slice.pairs_examined;
                work.source_rows_examined = slice.source_rows_examined;
                work.truncated = slice.truncated;
                if slice.truncated {
                    log::warn!(
                        "[maintenance-slice] Page source evidence exceeded the {}-row cap; source-overlap was suppressed for that Page",
                        duplicates::AUTOMATIC_SOURCE_CAP
                    );
                }
                if let Some(pair) = slice.candidate.as_ref() {
                    let emitted = emit_page_merge_card(db, pair).await?;
                    result.merge_cards_emitted = usize::from(emitted);
                    paused = emitted;
                }
                if let Some(cursor) = slice.next_cursor {
                    db.set_app_metadata(
                        AUTOMATIC_NEAR_DUPLICATE_CURSOR_KEY,
                        &serde_json::to_string(&cursor)?,
                    )
                    .await?;
                }
                if !more {
                    // EOF ends this cooperative pass. The next maintenance
                    // round starts at the beginning so Pages inserted behind
                    // the cursor are eventually seen without an offset treadmill.
                    db.set_app_metadata(AUTOMATIC_NEAR_DUPLICATE_CURSOR_KEY, "")
                        .await?;
                }
            }
        }
        MaintenanceStage::OrphanInventory => {
            let count = db.list_orphan_link_labels(1).await?.len().min(1);
            result.orphan_labels_checked = count;
            selected = count > 0;
            progressed = selected;
        }
        MaintenanceStage::CrossSpaceDiscovery => {
            if db.has_pending_cross_space_discovery().await? {
                paused = true;
            } else {
                let cursor = db
                    .get_app_metadata(AUTOMATIC_CROSS_SPACE_CURSOR_KEY)
                    .await?
                    .filter(|value| !value.is_empty());
                let mut slice = db
                    .find_cross_space_distillation_cluster_slice(
                        config.formation_threshold,
                        config.page_min_cluster_size,
                        config.token_limit,
                        config.max_unlinked_cluster_size,
                        config.max_grouped_cluster_size,
                        cursor.as_deref(),
                        AUTOMATIC_DISCOVERY_SEED_BUDGET,
                        AUTOMATIC_DISCOVERY_NEIGHBOR_BUDGET,
                    )
                    .await?;
                selected = slice.seeds_examined > 0;
                progressed = selected;
                more = slice.more;
                work.seeds_examined = slice.seeds_examined;
                work.eligible_seeds_probed = slice.eligible_seeds_probed;
                work.neighbor_rows_examined = slice.neighbor_rows_examined;
                work.fully_filtered_seeds = slice.fully_filtered_seeds;
                if slice.seeds_examined > 0 {
                    log::info!(
                        "[maintenance-slice] cross-space raw_seeds={}, eligible_probes={}, ANN rows={}, no-cross-space-neighborhood={}",
                        slice.seeds_examined,
                        slice.eligible_seeds_probed,
                        slice.neighbor_rows_examined,
                        slice.fully_filtered_seeds,
                    );
                }
                if let Some(cluster) = slice.cluster.take() {
                    let emitted = emit_cross_space_discovery_card(db, &cluster.source_ids).await?;
                    result.discovery_cards_emitted = usize::from(emitted);
                    // Keep the human queue at one discovery card. A dismissed
                    // card retains its deterministic id across cursor wraps.
                    paused = emitted;
                }
                if let Some(cursor) = slice.next_cursor {
                    db.set_app_metadata(AUTOMATIC_CROSS_SPACE_CURSOR_KEY, &cursor)
                        .await?;
                }
                if !more {
                    db.set_app_metadata(AUTOMATIC_CROSS_SPACE_CURSOR_KEY, "")
                        .await?;
                }
            }
        }
        MaintenanceStage::StalePage => {
            let cursor = load_maintenance_stale_cursor(db).await?;
            let page = match db
                .get_stale_page_after("source_updated", cursor.as_ref())
                .await?
            {
                Some(page) => Some(page),
                None if cursor.is_some() => {
                    persist_maintenance_stale_cursor(db, None).await?;
                    db.get_stale_page_after("source_updated", None).await?
                }
                None => None,
            };
            if let Some(page) = page {
                selected = true;
                let human_owned = page_is_human_owned(&page);
                let Some(provider) = llm.filter(|provider| provider.is_available()) else {
                    paused = true;
                    if human_owned {
                        result.stale_human_queued = 1;
                    } else {
                        result.stale_machine_queued = 1;
                    }
                    return Ok(MaintenanceSliceReport {
                        stage,
                        result,
                        selected,
                        progressed,
                        more,
                        retryable,
                        paused,
                        work,
                    });
                };
                // Advance before provider code so a panic/error rotates the
                // next maintenance round beyond this Page. Persisting row
                // identity avoids OFFSET drift when a completed Page leaves.
                let selected_cursor = StalePageCursor::for_page(&page);
                persist_maintenance_stale_cursor(db, Some(&selected_cursor)).await?;
                if automatic_refresh_exceeds_source_cap(db, &page).await? {
                    log::warn!(
                        "[maintenance-slice] stale Page '{}' exceeds the automatic {}-source cap; keeping it stale for explicit refresh",
                        page.id,
                        AUTOMATIC_PAGE_REFRESH_SOURCE_CAP
                    );
                    progressed = true;
                    more = true;
                    retryable = true;
                } else {
                    match refresh_page(
                        db,
                        provider,
                        prompts,
                        &page.id,
                        RefreshReason::SourceChanged,
                        knowledge_path,
                    )
                    .await
                    {
                        Ok(outcome) if outcome.wrote || outcome.gated => {
                            db.clear_page_staleness(&page.id).await?;
                            progressed = true;
                            more = true;
                            if human_owned || outcome.gated {
                                result.stale_human_cards = usize::from(outcome.gated);
                            } else {
                                result.stale_machine_refreshed = usize::from(outcome.wrote);
                            }
                        }
                        Ok(_) => retryable = true,
                        Err(error) => {
                            log::warn!(
                                "[maintenance-slice] stale Page '{}' failed: {error}",
                                page.id
                            );
                            retryable = true;
                        }
                    }
                }
            }
        }
        MaintenanceStage::Overview => {
            let Some(provider) = llm.filter(|provider| provider.is_available()) else {
                paused = true;
                return Ok(MaintenanceSliceReport {
                    stage,
                    result,
                    selected,
                    progressed,
                    more,
                    retryable,
                    paused,
                    work,
                });
            };
            selected = true;
            let overview = crate::synthesis::overview::refresh_overview_page(
                db,
                provider,
                prompts,
                "maintenance",
                knowledge_path,
            )
            .await?;
            progressed = overview.wrote;
            result.overview_refreshed = usize::from(overview.wrote);
        }
    }

    Ok(MaintenanceSliceReport {
        stage,
        result,
        selected,
        progressed,
        more,
        retryable,
        paused,
        work,
    })
}

#[derive(Debug, Default)]
struct RetroSweepResult {
    expected_card_volume: usize,
    cards_emitted: usize,
    merge_cards_emitted: usize,
    stub_cards_emitted: usize,
    paused: bool,
}

#[derive(Debug)]
enum RetroCardCandidate {
    PageMerge(duplicates::NearDuplicatePair),
    KeepOrArchive(StubPageCandidate),
}

#[derive(Debug)]
struct StubPageCandidate {
    page_id: String,
    source_count: usize,
}

async fn run_retro_sweep(
    db: &MemoryDB,
    config: &MaintenanceTickConfig,
    max_per_tick: usize,
) -> Result<RetroSweepResult, WenlanError> {
    if db
        .get_app_metadata(RETRO_SWEEP_COMPLETE_KEY)
        .await?
        .as_deref()
        == Some("1")
    {
        return Ok(RetroSweepResult::default());
    }

    let paused = db.get_app_metadata(RETRO_SWEEP_PAUSE_KEY).await?.as_deref() == Some("1");
    if paused && pending_retro_review_count(db).await? > 0 {
        return Ok(RetroSweepResult {
            paused: true,
            ..RetroSweepResult::default()
        });
    }
    if paused {
        db.set_app_metadata(RETRO_SWEEP_PAUSE_KEY, "0").await?;
    }

    let candidates = collect_retro_candidates(db, config).await?;
    let expected_card_volume = candidates.len();
    if expected_card_volume == 0 {
        db.set_app_metadata(RETRO_SWEEP_COMPLETE_KEY, "1").await?;
        db.set_app_metadata(RETRO_SWEEP_PAUSE_KEY, "0").await?;
        return Ok(RetroSweepResult::default());
    }

    let mut result = RetroSweepResult {
        expected_card_volume,
        ..RetroSweepResult::default()
    };
    for candidate in candidates.iter().take(max_per_tick) {
        match candidate {
            RetroCardCandidate::PageMerge(pair) => {
                let emitted = emit_page_merge_card(db, pair).await?;
                result.cards_emitted += usize::from(emitted);
                result.merge_cards_emitted += usize::from(emitted);
            }
            RetroCardCandidate::KeepOrArchive(stub) => {
                let emitted =
                    emit_keep_or_archive_card(db, &stub.page_id, stub.source_count).await?;
                result.cards_emitted += usize::from(emitted);
                result.stub_cards_emitted += usize::from(emitted);
            }
        }
    }

    if expected_card_volume > result.cards_emitted {
        db.set_app_metadata(RETRO_SWEEP_PAUSE_KEY, "1").await?;
        result.paused = true;
    } else {
        db.set_app_metadata(RETRO_SWEEP_COMPLETE_KEY, "1").await?;
        db.set_app_metadata(RETRO_SWEEP_PAUSE_KEY, "0").await?;
    }
    Ok(result)
}

async fn collect_retro_candidates(
    db: &MemoryDB,
    config: &MaintenanceTickConfig,
) -> Result<Vec<RetroCardCandidate>, WenlanError> {
    let mut candidates = Vec::new();
    for pair in duplicates::detect_all_near_duplicate_pages(db, config.page_match_threshold).await?
    {
        let id = page_merge_card_id(&pair.left_id, &pair.right_id);
        if db.get_refinement_proposal(&id).await?.is_none() {
            candidates.push(RetroCardCandidate::PageMerge(pair));
        }
    }
    for stub in list_distilled_stub_pages(db).await? {
        let id = keep_or_archive_card_id(&stub.page_id);
        if db.get_refinement_proposal(&id).await?.is_none() {
            candidates.push(RetroCardCandidate::KeepOrArchive(stub));
        }
    }
    Ok(candidates)
}

async fn list_distilled_stub_pages(db: &MemoryDB) -> Result<Vec<StubPageCandidate>, WenlanError> {
    let mut out = Vec::new();
    for page in db.list_pages("active", i64::MAX, 0).await? {
        if !page_is_retro_stub_candidate(&page) {
            continue;
        }
        let source_count = effective_page_source_count(db, &page).await?;
        if source_count < STUB_PAGE_SOURCE_FLOOR {
            out.push(StubPageCandidate {
                page_id: page.id,
                source_count,
            });
        }
    }
    out.sort_by(|left, right| {
        left.source_count
            .cmp(&right.source_count)
            .then_with(|| left.page_id.cmp(&right.page_id))
    });
    Ok(out)
}

fn page_is_retro_stub_candidate(page: &Page) -> bool {
    page.creation_kind == "distilled"
        && !page_is_human_owned(page)
        && !page.title.eq_ignore_ascii_case("overview")
}

async fn effective_page_source_count(db: &MemoryDB, page: &Page) -> Result<usize, WenlanError> {
    let sources = db.get_page_sources(&page.id).await?;
    if sources.is_empty() {
        Ok(page.source_memory_ids.len())
    } else {
        Ok(sources.len())
    }
}

pub(crate) async fn emit_keep_or_archive_card(
    db: &MemoryDB,
    page_id: &str,
    source_count: usize,
) -> Result<bool, WenlanError> {
    let id = keep_or_archive_card_id(page_id);
    if db.get_refinement_proposal(&id).await?.is_some() {
        return Ok(false);
    }
    let legacy_id = legacy_keep_or_archive_card_id(page_id);
    if let Some(proposal) = db.get_refinement_proposal(&legacy_id).await? {
        if proposal.source_ids.as_slice() == [page_id] {
            return Ok(false);
        }
    }

    let payload = serde_json::json!({
        "page_id": page_id,
        "source_count": source_count,
        "allowed_actions": ["dismiss", "accept"],
    })
    .to_string();
    db.insert_refinement_proposal(
        &id,
        "page_keep_or_archive",
        &[page_id.to_string()],
        Some(&payload),
        1.0,
    )
    .await?;
    db.resolve_refinement_if_open(&id, "awaiting_review")
        .await?;
    Ok(true)
}

async fn pending_retro_review_count(db: &MemoryDB) -> Result<usize, WenlanError> {
    Ok(db
        .get_pending_refinements()
        .await?
        .into_iter()
        .filter(|proposal| {
            matches!(proposal.status.as_str(), "pending" | "awaiting_review")
                && retro_review_action(&proposal.action)
        })
        .count())
}

fn retro_review_action(action: &str) -> bool {
    matches!(action, "page_merge" | "page_keep_or_archive")
}

async fn emit_page_merge_card(
    db: &MemoryDB,
    pair: &duplicates::NearDuplicatePair,
) -> Result<bool, WenlanError> {
    let id = page_merge_card_id(&pair.left_id, &pair.right_id);
    if db.get_refinement_proposal(&id).await?.is_some() {
        return Ok(false);
    }
    let legacy_id = legacy_page_merge_card_id(&pair.left_id, &pair.right_id);
    if let Some(proposal) = db.get_refinement_proposal(&legacy_id).await? {
        let same_pair = proposal.source_ids.len() == 2
            && proposal
                .source_ids
                .iter()
                .all(|id| id == &pair.left_id || id == &pair.right_id);
        if same_pair {
            return Ok(false);
        }
    }
    let Some(order) = page_merge_order::order_survivor(db, &pair.left_id, &pair.right_id).await?
    else {
        return Ok(false);
    };
    let survivor_id = order.survivor_id;
    let absorbed_id = order.absorbed_id;

    let payload = serde_json::json!({
        "left_page_id": &survivor_id,
        "right_page_id": &absorbed_id,
        "similarity": pair.similarity,
        "source_overlap": pair.source_overlap,
        "source_overlap_ratio": pair.source_overlap_ratio,
    })
    .to_string();
    let confidence = pair
        .similarity
        .unwrap_or(pair.source_overlap_ratio)
        .clamp(0.0, 1.0);
    db.insert_refinement_proposal(
        &id,
        "page_merge",
        &[survivor_id, absorbed_id],
        Some(&payload),
        confidence,
    )
    .await?;
    db.resolve_refinement_if_open(&id, "awaiting_review")
        .await?;
    Ok(true)
}

fn page_merge_card_id(left: &str, right: &str) -> String {
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!(
        "page_merge_{}_{}_{:016x}",
        stable_fragment(first),
        stable_fragment(second),
        stable_hash(&[first, second]),
    )
}

fn keep_or_archive_card_id(page_id: &str) -> String {
    format!(
        "page_keep_or_archive_{}_{:016x}",
        stable_fragment(page_id),
        stable_hash(&[page_id]),
    )
}

fn legacy_page_merge_card_id(left: &str, right: &str) -> String {
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    format!(
        "page_merge_{}_{}",
        stable_fragment(first),
        stable_fragment(second)
    )
}

fn legacy_keep_or_archive_card_id(page_id: &str) -> String {
    format!("page_keep_or_archive_{}", stable_fragment(page_id))
}

async fn emit_cross_space_discovery_card(
    db: &MemoryDB,
    source_ids: &[String],
) -> Result<bool, WenlanError> {
    let spaces = spaces_for_sources(db, source_ids).await?;
    if spaces.len() < 2 {
        return Ok(false);
    }

    let id = cross_space_discovery_card_id(source_ids);
    if db.get_refinement_proposal(&id).await?.is_some() {
        return Ok(false);
    }

    let payload = serde_json::json!({
        "memory_count": source_ids.len(),
        "spaces": spaces,
        "allowed_actions": ["dismiss", "pick_space"],
    })
    .to_string();
    db.insert_refinement_proposal(
        &id,
        "cross_space_discovery",
        source_ids,
        Some(&payload),
        1.0,
    )
    .await?;
    db.resolve_refinement_if_open(&id, "awaiting_review")
        .await?;
    Ok(true)
}

async fn spaces_for_sources(
    db: &MemoryDB,
    source_ids: &[String],
) -> Result<Vec<String>, WenlanError> {
    let mut spaces = std::collections::BTreeSet::new();
    for source_id in source_ids {
        if let Some(space) = db.get_memory_space(source_id).await? {
            if !space.is_empty() {
                spaces.insert(space);
            }
        }
    }
    Ok(spaces.into_iter().collect())
}

fn cross_space_discovery_card_id(source_ids: &[String]) -> String {
    let mut ids = source_ids.to_vec();
    ids.sort();
    let refs = ids.iter().map(String::as_str).collect::<Vec<_>>();
    format!("cross_space_discovery_{:016x}", stable_hash(&refs))
}

fn stable_hash(ids: &[&str]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for id in ids {
        for byte in id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn stable_fragment(id: &str) -> String {
    id.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MemoryDB;
    use crate::events::NoopEmitter;
    use crate::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};
    use std::sync::Arc;

    struct TestProvider {
        body: String,
    }

    #[async_trait::async_trait]
    impl LlmProvider for TestProvider {
        async fn generate(&self, _request: LlmRequest) -> Result<String, LlmError> {
            Ok(self.body.clone())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "maintenance-test"
        }

        fn backend(&self) -> LlmBackend {
            LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    async fn new_test_db() -> (MemoryDB, tempfile::TempDir) {
        let db_dir = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(db_dir.path(), Arc::new(NoopEmitter))
            .await
            .unwrap();
        (db, db_dir)
    }

    async fn store_test_memory(db: &MemoryDB, id: &str, content: &str) {
        db.upsert_documents(vec![wenlan_types::RawDocument {
            source: "memory".to_string(),
            source_id: id.to_string(),
            title: id.to_string(),
            content: content.to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confirmed: Some(true),
            ..Default::default()
        }])
        .await
        .unwrap();
    }

    async fn insert_test_page(db: &MemoryDB, id: &str, content: &str, source_ids: &[&str]) {
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page_with_kind(
            id,
            id,
            None,
            content,
            None,
            None,
            source_ids,
            &now,
            "research",
            "confirmed",
            Some("work"),
            Some("[]"),
        )
        .await
        .unwrap();
    }

    async fn insert_distilled_test_page(
        db: &MemoryDB,
        id: &str,
        content: &str,
        source_ids: &[&str],
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page_with_kind(
            id,
            id,
            None,
            content,
            None,
            None,
            source_ids,
            &now,
            "distilled",
            "confirmed",
            Some("work"),
            Some("[]"),
        )
        .await
        .unwrap();
    }

    fn config() -> MaintenanceTickConfig {
        MaintenanceTickConfig {
            page_match_threshold: 0.85,
            formation_threshold: 0.60,
            page_min_cluster_size: 3,
            token_limit: 3500,
            max_unlinked_cluster_size: 20,
            max_grouped_cluster_size: 20,
            max_per_tick: 5,
        }
    }

    fn discovery_config() -> MaintenanceTickConfig {
        MaintenanceTickConfig {
            page_match_threshold: 0.85,
            formation_threshold: 0.80,
            page_min_cluster_size: 3,
            token_limit: 3500,
            max_unlinked_cluster_size: 20,
            max_grouped_cluster_size: 20,
            max_per_tick: 5,
        }
    }

    fn vec_to_sql(v: &[f32]) -> String {
        let mut out = String::with_capacity(v.len() * 10);
        out.push('[');
        for (i, value) in v.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            use std::fmt::Write;
            let _ = write!(out, "{value:.6}");
        }
        out.push(']');
        out
    }

    fn unit_vec(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; 768];
        v[axis] = 1.0;
        v
    }

    async fn insert_staging_memory(
        db: &MemoryDB,
        source_id: &str,
        content: &str,
        space: &str,
        embedding: &[f32],
        last_modified: i64,
    ) {
        let embedding_sql = vec_to_sql(embedding);
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (
                id, content, source, source_id, title, chunk_index, last_modified,
                chunk_type, memory_type, space, source_agent, confidence,
                confirmed, word_count, enrichment_status, quality, is_recap,
                supersede_mode, stability, embedding
             )
             VALUES (
                ?1, ?2, 'memory', ?1, ?3, 0, ?4, 'document', 'fact',
                ?5, 'codex-test', 0.9, 1, 8, 'enriched', 'high', 0,
                'hide', 'learned', vector32(?6)
             )",
            libsql::params![
                source_id,
                content,
                content.chars().take(40).collect::<String>(),
                last_modified,
                space,
                embedding_sql.as_str(),
            ],
        )
        .await
        .unwrap();
    }

    async fn active_page_count(db: &MemoryDB) -> i64 {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM pages WHERE status = 'active'", ())
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
    }

    async fn page_status(db: &MemoryDB, page_id: &str) -> String {
        db.get_page(page_id)
            .await
            .unwrap()
            .expect("page exists")
            .status
    }

    async fn page_source_count(db: &MemoryDB) -> i64 {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM page_sources", ())
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
    }

    #[tokio::test]
    async fn cross_space_topic_emits_one_discovery_card_without_page_mutation() {
        let (db, _db_dir) = new_test_db().await;
        let embedding = unit_vec(42);
        let now = chrono::Utc::now().timestamp();
        for (source_id, space) in [
            ("cross_disc_work_a", "work"),
            ("cross_disc_personal_a", "personal"),
            ("cross_disc_work_b", "work"),
        ] {
            insert_staging_memory(
                &db,
                source_id,
                "Incremental Rust compilation cache tuning for shared developer machines.",
                space,
                &embedding,
                now,
            )
            .await;
        }

        let result = run_maintenance_tick(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.discovery_cards_emitted, 1);
        assert_eq!(active_page_count(&db).await, 0);
        assert_eq!(page_source_count(&db).await, 0);

        let discovery_cards: Vec<_> = db
            .get_pending_refinements()
            .await
            .unwrap()
            .into_iter()
            .filter(|p| p.action == "cross_space_discovery")
            .collect();
        assert_eq!(
            discovery_cards.len(),
            1,
            "one cross-space cluster should produce exactly one discovery card"
        );
        assert_eq!(discovery_cards[0].source_ids.len(), 3);
        let payload = discovery_cards[0]
            .payload
            .as_deref()
            .expect("discovery card should carry a typed payload");
        assert!(
            payload.contains("\"memory_count\":3"),
            "payload should describe the card prompt facts, got {payload}"
        );
        assert!(
            payload.contains("\"work\"") && payload.contains("\"personal\""),
            "payload should name every space involved, got {payload}"
        );
    }

    #[tokio::test]
    async fn dismissed_cross_space_discovery_card_stays_dismissed_across_next_tick() {
        let (db, _db_dir) = new_test_db().await;
        let embedding = unit_vec(43);
        let now = chrono::Utc::now().timestamp();
        for (source_id, space) in [
            ("cross_dismiss_work_a", "work"),
            ("cross_dismiss_personal_a", "personal"),
            ("cross_dismiss_work_b", "work"),
        ] {
            insert_staging_memory(
                &db,
                source_id,
                "Cross-space topic cards should not resurrect after dismissal.",
                space,
                &embedding,
                now,
            )
            .await;
        }

        let first = run_maintenance_tick(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(first.discovery_cards_emitted, 1);
        let card = db
            .get_pending_refinements()
            .await
            .unwrap()
            .into_iter()
            .find(|p| p.action == "cross_space_discovery")
            .expect("cross-space discovery card exists");
        db.resolve_refinement_if_open(&card.id, "dismissed")
            .await
            .unwrap();

        let second = run_maintenance_tick(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(second.discovery_cards_emitted, 0);
        let dismissed = db
            .get_refinement_proposal(&card.id)
            .await
            .unwrap()
            .expect("dismissed discovery card remains");
        assert_eq!(dismissed.status, "dismissed");
    }

    #[tokio::test]
    async fn dismissed_page_merge_card_stays_dismissed_across_next_tick() {
        let (db, _db_dir) = new_test_db().await;
        let source = "Rust ownership prevents data races at compile time.";
        for id in ["mem_dup_a", "mem_dup_b", "mem_dup_c"] {
            store_test_memory(&db, id, source).await;
        }
        insert_test_page(
            &db,
            "page_dup_a",
            source,
            &["mem_dup_a", "mem_dup_b", "mem_dup_c"],
        )
        .await;
        insert_test_page(
            &db,
            "page_dup_b",
            source,
            &["mem_dup_a", "mem_dup_b", "mem_dup_c"],
        )
        .await;

        let first = run_maintenance_tick(&db, None, &PromptRegistry::default(), &config(), None)
            .await
            .unwrap();
        assert_eq!(first.merge_cards_emitted, 1);
        let card = db
            .get_pending_refinements()
            .await
            .unwrap()
            .into_iter()
            .find(|p| p.action == "page_merge")
            .expect("page_merge card exists");
        db.resolve_refinement_if_open(&card.id, "dismissed")
            .await
            .unwrap();

        let second = run_maintenance_tick(&db, None, &PromptRegistry::default(), &config(), None)
            .await
            .unwrap();
        assert_eq!(second.merge_cards_emitted, 0);
        let dismissed = db
            .get_refinement_proposal(&card.id)
            .await
            .unwrap()
            .expect("dismissed card remains");
        assert_eq!(dismissed.status, "dismissed");
    }

    #[tokio::test]
    async fn retro_sweep_cards_near_duplicate_pages_and_distilled_stub_pages_without_mutation() {
        let (db, _db_dir) = new_test_db().await;
        let source = "Rust ownership prevents data races at compile time.";
        for id in ["retro_mem_dup_a", "retro_mem_dup_b", "retro_mem_dup_c"] {
            store_test_memory(&db, id, source).await;
        }
        insert_distilled_test_page(
            &db,
            "retro_page_dup_a",
            source,
            &["retro_mem_dup_a", "retro_mem_dup_b", "retro_mem_dup_c"],
        )
        .await;
        insert_distilled_test_page(
            &db,
            "retro_page_dup_b",
            source,
            &["retro_mem_dup_a", "retro_mem_dup_b", "retro_mem_dup_c"],
        )
        .await;
        insert_distilled_test_page(
            &db,
            "retro_stub_page",
            "Thin machine stub.",
            &["retro_mem_dup_a"],
        )
        .await;

        let mut retro_config = config();
        retro_config.page_match_threshold = 1.0;
        let result =
            run_maintenance_tick(&db, None, &PromptRegistry::default(), &retro_config, None)
                .await
                .unwrap();

        assert_eq!(result.retro_expected_card_volume, 2);
        assert_eq!(result.retro_cards_emitted, 2);
        assert_eq!(result.retro_stub_cards_emitted, 1);
        assert_eq!(page_status(&db, "retro_page_dup_b").await, "active");
        assert_eq!(page_status(&db, "retro_stub_page").await, "active");

        let pending = db.get_pending_refinements().await.unwrap();
        assert_eq!(
            pending.iter().filter(|p| p.action == "page_merge").count(),
            1,
            "retro sweep should card the near-duplicate pair"
        );
        let stub = pending
            .iter()
            .find(|p| p.action == "page_keep_or_archive")
            .expect("retro sweep should card a <3-source distilled stub");
        assert_eq!(stub.source_ids, vec!["retro_stub_page".to_string()]);
        assert!(
            stub.payload
                .as_deref()
                .unwrap_or_default()
                .contains("\"source_count\":1"),
            "stub payload should carry the source count"
        );
    }

    #[tokio::test]
    async fn dismissed_keep_or_archive_card_stays_dismissed_across_next_tick() {
        let (db, _db_dir) = new_test_db().await;
        store_test_memory(&db, "stub_dismiss_mem", "Small source.").await;
        insert_distilled_test_page(
            &db,
            "stub_dismiss_page",
            "Thin machine stub.",
            &["stub_dismiss_mem"],
        )
        .await;

        let first = run_maintenance_tick(&db, None, &PromptRegistry::default(), &config(), None)
            .await
            .unwrap();
        assert_eq!(first.retro_stub_cards_emitted, 1);
        let card = db
            .get_pending_refinements()
            .await
            .unwrap()
            .into_iter()
            .find(|p| p.action == "page_keep_or_archive")
            .expect("keep-or-archive card exists");
        db.resolve_refinement_if_open(&card.id, "dismissed")
            .await
            .unwrap();

        let second = run_maintenance_tick(&db, None, &PromptRegistry::default(), &config(), None)
            .await
            .unwrap();
        assert_eq!(second.retro_stub_cards_emitted, 0);
        let dismissed = db
            .get_refinement_proposal(&card.id)
            .await
            .unwrap()
            .expect("dismissed keep-or-archive card remains");
        assert_eq!(dismissed.status, "dismissed");
    }

    #[tokio::test]
    async fn retro_sweep_large_predicted_volume_emits_one_batch_and_pauses() {
        let (db, _db_dir) = new_test_db().await;
        for idx in 0..5 {
            let mem_id = format!("retro_batch_mem_{idx}");
            let page_id = format!("retro_batch_stub_{idx}");
            let content = format!("Thin machine stub {idx} with distinct audit context.");
            store_test_memory(&db, &mem_id, &content).await;
            insert_distilled_test_page(&db, &page_id, &content, &[&mem_id]).await;
        }
        let mut batch_config = config();
        batch_config.max_per_tick = 2;
        batch_config.page_match_threshold = 1.0;

        let first =
            run_maintenance_tick(&db, None, &PromptRegistry::default(), &batch_config, None)
                .await
                .unwrap();

        assert_eq!(first.retro_expected_card_volume, 5);
        assert_eq!(first.retro_cards_emitted, 2);
        assert_eq!(first.retro_stub_cards_emitted, 2);
        assert!(first.retro_paused);
        assert_eq!(
            db.get_pending_refinements()
                .await
                .unwrap()
                .into_iter()
                .filter(|p| p.action == "page_keep_or_archive")
                .count(),
            2,
            "large retro sweep should not flood all predicted cards at once"
        );
        assert_eq!(
            db.get_app_metadata("maintenance_retro_sweep_v1_pause")
                .await
                .unwrap()
                .as_deref(),
            Some("1")
        );

        let second =
            run_maintenance_tick(&db, None, &PromptRegistry::default(), &batch_config, None)
                .await
                .unwrap();
        assert_eq!(second.retro_cards_emitted, 0);
        assert!(second.retro_paused);
    }

    #[tokio::test]
    async fn stale_machine_page_survives_noop_refresh_for_retry() {
        let (db, _db_dir) = new_test_db().await;
        let source = "Rust ownership prevents data races at compile time.";
        store_test_memory(&db, "mem_stale", source).await;
        insert_test_page(&db, "page_stale", "Old machine prose.", &["mem_stale"]).await;
        db.set_page_stale("page_stale", "source_updated")
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(TestProvider {
            body: String::new(),
        });
        let result =
            run_maintenance_tick(&db, Some(&llm), &PromptRegistry::default(), &config(), None)
                .await
                .unwrap();

        assert_eq!(result.stale_machine_refreshed, 0);
        let page = db
            .get_page("page_stale")
            .await
            .unwrap()
            .expect("stale page remains");
        assert_eq!(page.stale_reason.as_deref(), Some("source_updated"));
    }

    #[tokio::test]
    async fn maintenance_stale_stage_runs_exactly_one_item_and_no_other_stage() {
        let (db, _db_dir) = new_test_db().await;
        for suffix in ["a", "b"] {
            let memory_id = format!("maintenance_slice_memory_{suffix}");
            let page_id = format!("maintenance_slice_page_{suffix}");
            store_test_memory(&db, &memory_id, "bounded maintenance source").await;
            insert_distilled_test_page(
                &db,
                &page_id,
                "old maintenance body",
                &[memory_id.as_str()],
            )
            .await;
            db.set_page_stale(&page_id, "source_updated").await.unwrap();
        }
        let llm: Arc<dyn LlmProvider> = Arc::new(TestProvider {
            body: "bounded maintenance source [1]".to_string(),
        });

        let report = run_maintenance_stage_slice(
            &db,
            Some(&llm),
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::StalePage,
        )
        .await
        .unwrap();

        assert_eq!(report.stage, MaintenanceStage::StalePage);
        assert!(report.selected);
        assert!(report.progressed);
        assert!(report.more);
        assert!(!report.retryable);
        assert!(!report.paused);
        assert_eq!(report.result.stale_machine_refreshed, 1);
        assert_eq!(
            db.list_stale_pages("source_updated").await.unwrap().len(),
            1
        );
        assert!(
            db.find_active_page_id_by_title("Overview")
                .await
                .unwrap()
                .is_none(),
            "the stale stage must not also run Overview"
        );
        assert!(
            db.get_pending_refinements().await.unwrap().is_empty(),
            "the stale stage must not emit retro/duplicate/discovery cards"
        );
    }

    #[tokio::test]
    async fn maintenance_stale_success_does_not_skip_row_shifted_by_stale_removal() {
        let (db, _db_dir) = new_test_db().await;
        for (suffix, last_modified) in [
            ("a", "2026-07-16T03:00:00+00:00"),
            ("b", "2026-07-16T02:00:00+00:00"),
            ("c", "2026-07-16T01:00:00+00:00"),
        ] {
            let memory_id = format!("maintenance_shift_memory_{suffix}");
            let page_id = format!("maintenance_shift_page_{suffix}");
            store_test_memory(&db, &memory_id, "ordered maintenance source").await;
            insert_distilled_test_page(
                &db,
                &page_id,
                "old maintenance body",
                &[memory_id.as_str()],
            )
            .await;
            {
                let conn = db.conn.lock().await;
                conn.execute(
                    "UPDATE pages SET last_modified = ?1 WHERE id = ?2",
                    libsql::params![last_modified, page_id.as_str()],
                )
                .await
                .unwrap();
            }
            db.set_page_stale(&page_id, "source_updated").await.unwrap();
        }
        let llm: Arc<dyn LlmProvider> = Arc::new(TestProvider {
            body: "ordered maintenance source [1]".to_string(),
        });

        for _ in 0..2 {
            run_maintenance_stage_slice(
                &db,
                Some(&llm),
                &PromptRegistry::default(),
                &config(),
                None,
                MaintenanceStage::StalePage,
            )
            .await
            .unwrap();
        }

        assert!(
            db.get_page_stale_reason("maintenance_shift_page_a")
                .await
                .unwrap()
                .is_none(),
            "the first stale Page should complete"
        );
        assert!(
            db.get_page_stale_reason("maintenance_shift_page_b")
                .await
                .unwrap()
                .is_none(),
            "removing the first row must not shift the second Page behind a persisted OFFSET"
        );
        assert_eq!(
            db.get_page_stale_reason("maintenance_shift_page_c")
                .await
                .unwrap()
                .as_deref(),
            Some("source_updated"),
            "two slices should leave only the third Page stale"
        );
    }

    #[tokio::test]
    async fn maintenance_stale_over_source_cap_makes_zero_provider_calls_and_stays_stale() {
        let (db, _db_dir) = new_test_db().await;
        let source_ids = (0..65)
            .map(|index| format!("maintenance_over_cap_memory_{index:02}"))
            .collect::<Vec<_>>();
        for source_id in &source_ids {
            store_test_memory(&db, source_id, "bounded maintenance source").await;
        }
        let source_refs = source_ids.iter().map(String::as_str).collect::<Vec<_>>();
        insert_distilled_test_page(
            &db,
            "maintenance_over_cap_page",
            "old maintenance body",
            &source_refs,
        )
        .await;
        db.set_page_stale("maintenance_over_cap_page", "source_updated")
            .await
            .unwrap();
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            "bounded maintenance source [1]",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let report = run_maintenance_stage_slice(
            &db,
            Some(&llm),
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::StalePage,
        )
        .await
        .unwrap();

        assert!(report.selected);
        assert!(report.retryable);
        assert_eq!(report.result.stale_machine_refreshed, 0);
        assert_eq!(provider.call_count(), 0);
        assert_eq!(
            db.get_page_stale_reason("maintenance_over_cap_page")
                .await
                .unwrap()
                .as_deref(),
            Some("source_updated")
        );
    }

    #[tokio::test]
    async fn maintenance_stale_retry_rotates_without_erasing_retry() {
        let (db, _db_dir) = new_test_db().await;
        for suffix in ["a", "b"] {
            let memory_id = format!("maintenance_retry_memory_{suffix}");
            let page_id = format!("maintenance_retry_page_{suffix}");
            store_test_memory(&db, &memory_id, "bounded retry source").await;
            insert_distilled_test_page(&db, &page_id, "old retry body", &[memory_id.as_str()])
                .await;
            db.set_page_stale(&page_id, "source_updated").await.unwrap();
        }
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            "",
            "bounded retry source [1]",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let first = run_maintenance_stage_slice(
            &db,
            Some(&llm),
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::StalePage,
        )
        .await
        .unwrap();
        let second = run_maintenance_stage_slice(
            &db,
            Some(&llm),
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::StalePage,
        )
        .await
        .unwrap();

        assert!(first.retryable);
        assert!(!first.progressed);
        assert!(second.progressed);
        assert!(!second.retryable);
        assert_eq!(provider.call_count(), 2);
        assert_eq!(
            db.list_stale_pages("source_updated").await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn automatic_retro_slice_examines_one_page_and_three_sources_at_most() {
        let (db, _db_dir) = new_test_db().await;
        for idx in 0..4 {
            let memory_id = format!("bounded_retro_memory_{idx}");
            let page_id = format!("bounded_retro_page_{idx}");
            store_test_memory(&db, &memory_id, "small retro source").await;
            insert_distilled_test_page(&db, &page_id, "thin machine page", &[&memory_id]).await;
        }

        let report = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::RetroReview,
        )
        .await
        .unwrap();

        assert_eq!(report.work.pages_examined, 1);
        assert!(report.work.source_rows_examined <= 3);
        assert!(report.result.retro_cards_emitted <= 1);
        assert!(
            report.more,
            "the durable stub cursor must expose remaining work"
        );
    }

    #[tokio::test]
    async fn automatic_retro_ineligible_page_advances_cursor_without_emitting_card() {
        let (db, _db_dir) = new_test_db().await;
        insert_test_page(&db, "a_ineligible", "authored control", &[]).await;
        insert_distilled_test_page(&db, "b_eligible", "thin machine page", &[]).await;

        let report = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::RetroReview,
        )
        .await
        .unwrap();

        assert!(report.selected);
        assert!(report.progressed);
        assert!(report.more);
        assert_eq!(report.work.pages_examined, 1);
        assert_eq!(report.work.source_rows_examined, 0);
        assert_eq!(report.result.retro_cards_emitted, 0);
        assert_eq!(
            db.get_app_metadata(AUTOMATIC_RETRO_CURSOR_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some("a_ineligible")
        );
        assert!(db.get_pending_refinements().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn automatic_near_duplicate_slice_reports_bounded_pair_and_source_work() {
        let (db, _db_dir) = new_test_db().await;
        let source = "bounded duplicate source";
        for id in ["bounded_dup_a", "bounded_dup_b", "bounded_dup_c"] {
            store_test_memory(&db, id, source).await;
        }
        insert_distilled_test_page(
            &db,
            "bounded_dup_page_a",
            source,
            &["bounded_dup_a", "bounded_dup_b", "bounded_dup_c"],
        )
        .await;
        insert_distilled_test_page(
            &db,
            "bounded_dup_page_b",
            source,
            &["bounded_dup_a", "bounded_dup_b", "bounded_dup_c"],
        )
        .await;

        let report = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::NearDuplicate,
        )
        .await
        .unwrap();

        assert!((1..=128).contains(&report.work.pairs_examined));
        assert!(report.work.pages_examined <= 129);
        assert!(report.work.source_rows_examined <= 129 * 257);
        assert!(report.result.merge_cards_emitted <= 1);
    }

    #[tokio::test]
    async fn automatic_cross_space_slice_reports_bounded_seed_and_ann_work() {
        let (db, _db_dir) = new_test_db().await;
        let embedding = unit_vec(77);
        let now = chrono::Utc::now().timestamp();
        for (source_id, space) in [
            ("bounded_cross_a", "work"),
            ("bounded_cross_b", "personal"),
            ("bounded_cross_c", "work"),
        ] {
            insert_staging_memory(
                &db,
                source_id,
                "bounded cross-space maintenance topic",
                space,
                &embedding,
                now,
            )
            .await;
        }

        let report = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
            MaintenanceStage::CrossSpaceDiscovery,
        )
        .await
        .unwrap();

        assert!((1..=8).contains(&report.work.seeds_examined));
        assert!(report.work.neighbor_rows_examined <= 8 * 64);
        assert!(report.result.discovery_cards_emitted <= 1);
    }

    #[tokio::test]
    async fn open_review_cards_pause_automatic_stages_before_scan_or_cursor_work() {
        let (db, _db_dir) = new_test_db().await;
        let no_sources = Vec::<String>::new();
        db.insert_refinement_proposal("open-retro-control", "page_merge", &no_sources, None, 1.0)
            .await
            .unwrap();

        let retro = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::RetroReview,
        )
        .await
        .unwrap();
        assert!(retro.paused);
        assert!(retro.result.retro_paused);
        assert!(!retro.selected && !retro.progressed && !retro.more);
        assert_eq!(retro.work, MaintenanceSliceWork::default());
        assert_eq!(
            db.get_app_metadata(AUTOMATIC_RETRO_CURSOR_KEY)
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            db.get_app_metadata(AUTOMATIC_RETRO_COMPLETE_KEY)
                .await
                .unwrap(),
            None,
            "the paused probe must return before the empty scan marks retro complete"
        );

        let near_duplicate = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::NearDuplicate,
        )
        .await
        .unwrap();
        assert!(near_duplicate.paused);
        assert!(!near_duplicate.result.retro_paused);
        assert!(!near_duplicate.selected && !near_duplicate.progressed && !near_duplicate.more);
        assert_eq!(near_duplicate.work, MaintenanceSliceWork::default());
        assert_eq!(
            db.get_app_metadata(AUTOMATIC_NEAR_DUPLICATE_CURSOR_KEY)
                .await
                .unwrap(),
            None
        );

        db.insert_refinement_proposal(
            "open-cross-space-control",
            "cross_space_discovery",
            &no_sources,
            None,
            1.0,
        )
        .await
        .unwrap();
        let cross_space = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
            MaintenanceStage::CrossSpaceDiscovery,
        )
        .await
        .unwrap();
        assert!(cross_space.paused);
        assert!(!cross_space.result.retro_paused);
        assert!(!cross_space.selected && !cross_space.progressed && !cross_space.more);
        assert_eq!(cross_space.work, MaintenanceSliceWork::default());
        assert_eq!(
            db.get_app_metadata(AUTOMATIC_CROSS_SPACE_CURSOR_KEY)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn automatic_retro_slice_resumes_after_dismissal_without_rescanning_page() {
        let (db, _db_dir) = new_test_db().await;
        for suffix in ["a", "b"] {
            let memory_id = format!("automatic_retro_resume_memory_{suffix}");
            let page_id = format!("automatic_retro_resume_page_{suffix}");
            store_test_memory(&db, &memory_id, "resume retro source").await;
            insert_distilled_test_page(&db, &page_id, "thin resume page", &[&memory_id]).await;
        }

        let first = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::RetroReview,
        )
        .await
        .unwrap();
        assert!(first.paused);
        assert!(first.more);
        let first_card = db
            .get_pending_refinements()
            .await
            .unwrap()
            .into_iter()
            .find(|proposal| proposal.action == "page_keep_or_archive")
            .unwrap();
        assert_eq!(
            first_card.source_ids,
            vec!["automatic_retro_resume_page_a".to_string()]
        );
        db.resolve_refinement_if_open(&first_card.id, "dismissed")
            .await
            .unwrap();

        let second = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &config(),
            None,
            MaintenanceStage::RetroReview,
        )
        .await
        .unwrap();
        assert!(second.paused);
        assert!(!second.more);
        let second_card = db
            .get_pending_refinements()
            .await
            .unwrap()
            .into_iter()
            .find(|proposal| proposal.action == "page_keep_or_archive")
            .unwrap();
        assert_eq!(
            second_card.source_ids,
            vec!["automatic_retro_resume_page_b".to_string()]
        );
    }

    #[tokio::test]
    async fn automatic_near_duplicate_cursor_reaches_pair_beyond_first_budget() {
        let (db, _db_dir) = new_test_db().await;
        for idx in 0..16 {
            let memory_id = format!("pair_cursor_memory_{idx:02}");
            let page_id = format!("pair_cursor_page_{idx:02}");
            store_test_memory(&db, &memory_id, &format!("unique source {idx}")).await;
            insert_distilled_test_page(
                &db,
                &page_id,
                &format!("unique machine page {idx}"),
                &[&memory_id],
            )
            .await;
        }
        for source_id in [
            "pair_cursor_shared_a",
            "pair_cursor_shared_b",
            "pair_cursor_shared_c",
        ] {
            store_test_memory(&db, source_id, "shared cursor source").await;
        }
        for idx in 16..18 {
            let page_id = format!("pair_cursor_page_{idx:02}");
            insert_distilled_test_page(
                &db,
                &page_id,
                "late duplicate candidate",
                &[
                    "pair_cursor_shared_a",
                    "pair_cursor_shared_b",
                    "pair_cursor_shared_c",
                ],
            )
            .await;
        }
        let mut bounded_config = config();
        bounded_config.page_match_threshold = 1.1;

        let first = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &bounded_config,
            None,
            MaintenanceStage::NearDuplicate,
        )
        .await
        .unwrap();
        assert_eq!(first.work.pairs_examined, 128);
        assert!(first.more);
        assert_eq!(first.result.merge_cards_emitted, 0);
        assert!(db
            .get_app_metadata(AUTOMATIC_NEAR_DUPLICATE_CURSOR_KEY)
            .await
            .unwrap()
            .is_some_and(|cursor| !cursor.is_empty()));

        let second = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &bounded_config,
            None,
            MaintenanceStage::NearDuplicate,
        )
        .await
        .unwrap();
        assert!(second.work.pairs_examined <= 128);
        assert_eq!(second.result.merge_cards_emitted, 1);
    }

    #[tokio::test]
    async fn automatic_cross_space_cursor_reaches_seed_beyond_first_batch_and_caps_queue() {
        let (db, _db_dir) = new_test_db().await;
        let now = chrono::Utc::now().timestamp();
        for idx in 0..10 {
            insert_staging_memory(
                &db,
                &format!("cross_cursor_filler_{idx:02}"),
                &format!("unrelated filler topic {idx}"),
                "solo",
                &unit_vec(idx),
                now,
            )
            .await;
        }
        let target_embedding = unit_vec(77);
        for (source_id, space) in [
            ("cross_cursor_target_a", "work"),
            ("cross_cursor_target_b", "personal"),
            ("cross_cursor_target_c", "work"),
        ] {
            insert_staging_memory(
                &db,
                source_id,
                "late cross-space target topic",
                space,
                &target_embedding,
                now,
            )
            .await;
        }

        let first = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
            MaintenanceStage::CrossSpaceDiscovery,
        )
        .await
        .unwrap();
        assert_eq!(first.work.seeds_examined, 8);
        assert!(first.more);
        assert_eq!(first.result.discovery_cards_emitted, 0);

        let second = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
            MaintenanceStage::CrossSpaceDiscovery,
        )
        .await
        .unwrap();
        assert_eq!(second.result.discovery_cards_emitted, 1);
        assert!(second.paused);

        let third = run_maintenance_stage_slice(
            &db,
            None,
            &PromptRegistry::default(),
            &discovery_config(),
            None,
            MaintenanceStage::CrossSpaceDiscovery,
        )
        .await
        .unwrap();
        assert!(third.paused);
        assert_eq!(third.work.seeds_examined, 0);
        assert_eq!(third.result.discovery_cards_emitted, 0);
    }
}
