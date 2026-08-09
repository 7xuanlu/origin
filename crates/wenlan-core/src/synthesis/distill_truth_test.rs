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

/// At generation 0 every adapter is pass-through. The strict-policy half then
/// seeds a current evaluated provisional verdict and explicitly enables machine
/// enforcement before proving the ambient sweep skips it. Production keeps the
/// same verdict advisory until that separate switch is enabled.
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

    // Generation 1 plus explicit enforcement: seed a verdict about the page's
    // current bytes. Without the marker/version match this would be unevaluated,
    // and without the independent enforcement bit it would remain advisory.
    let current = db.get_page("page_gate").await.unwrap().unwrap();
    {
        let conn = db.test_primary_session().await;
        conn.execute(
            "INSERT OR REPLACE INTO page_truth_state
                 (page_id, page_version, support_status, human_reviewed,
                  updated_at, evaluated_at)
             VALUES ('page_gate', ?1, 'provisional', 0, ?2, ?2)",
            libsql::params![current.version, now_ts],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO claim_derivation_markers
                 (page_id, page_version, page_version_digest, extractor_version,
                  inventory_count, created_at)
             VALUES ('page_gate', ?1, ?2, ?3, 1, ?4)",
            libsql::params![
                current.version,
                crate::provenance::revision_content_digest(&current.content),
                crate::db::EXTRACTOR_VERSION,
                now_ts
            ],
        )
        .await
        .unwrap();
    }
    db.set_truth_cutover_generation(1).await.unwrap();
    db.set_app_metadata("claim_promoter_enforcement", "1")
        .await
        .unwrap();
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
