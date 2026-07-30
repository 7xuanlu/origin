// SPDX-License-Identifier: Apache-2.0
//! The M5 truth gate on the ambient re-distill lane.
//!
//! Separate from `distill.rs`'s inline `mod tests` for one reason: advancing
//! the cutover generation is a test-only lever, and
//! `db/truth_exposure_test.rs::the_cutover_setter_has_no_production_caller`
//! excuses callers by FILE rather than by `#[cfg(test)]` region -- deliberately,
//! because a scan that tried to recognize test regions inside a source file
//! would fail OPEN when it guessed the end wrong. So a test that needs the
//! setter lives in a `*_test.rs`, and this is it.

use super::*;
use crate::llm_provider::MockProvider;
use std::sync::Arc;

/// At generation 0 every adapter is pass-through, so the sweep still
/// re-distills a page with no truth row; at generation 1 that same page reads as
/// unsupported and the sweep must not spend an LLM round-trip on it. Both
/// generations run against the same substrate -- the pair is the point, per
/// `truth_adapter_test.rs`. A test that only ran at generation 1 would pass a
/// build that had quietly gone live.
#[tokio::test]
async fn refresh_page_skips_re_distill_for_a_page_the_automatic_reader_may_not_see() {
    let (db, _db_dir) = crate::db::tests::test_db().await;
    let now = chrono::Utc::now().to_rfc3339();
    let now_ts = chrono::Utc::now().timestamp();

    {
        let conn = db.test_primary_session().await;
        conn.execute(
            "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) \
             VALUES (?1, ?1, ?1, 'recompiled body reference material', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
            libsql::params!["mem_gate_seed".to_string(), now_ts],
        )
        .await
        .unwrap();
    }

    // No `page_truth_state` row inserted -- a page with no row reads as
    // unsupported once the cutover is live, the same post-migration normal case
    // `truth_adapter_test.rs` seeds as `p3`.
    db.insert_page(
        "page_gate",
        "Gated Topic",
        None,
        "original body",
        None,
        None,
        &["mem_gate_seed"],
        &now,
    )
    .await
    .unwrap();

    let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new("recompiled body [1]"));
    let prompts = PromptRegistry::default();

    // Generation 0: pass-through.
    db.set_page_stale("page_gate", "source_updated")
        .await
        .unwrap();
    let outcome = refresh_page(
        &db,
        &llm,
        &prompts,
        "page_gate",
        RefreshReason::SourceChanged,
        None,
    )
    .await
    .unwrap();
    assert!(
        outcome.wrote,
        "generation 0 must not gate the ambient re-distill sweep"
    );

    // Generation 1: the same page, still with no truth row, is unsupported.
    db.set_truth_cutover_generation(1).await.unwrap();
    db.set_page_stale("page_gate", "source_updated")
        .await
        .unwrap();
    let outcome = refresh_page(
        &db,
        &llm,
        &prompts,
        "page_gate",
        RefreshReason::SourceChanged,
        None,
    )
    .await
    .unwrap();
    assert!(
        !outcome.wrote,
        "an automatic reader may not see this page, so the ambient sweep must skip it"
    );
}
