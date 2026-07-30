// SPDX-License-Identifier: Apache-2.0

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::post_write::{page_write, PageWrite};
use crate::tuning::DistillationConfig;
use std::collections::BTreeSet;

const DETECT_DECLINED_MEMBERSHIPS_KEY: &str = "distill_detect_declined_memberships_v1";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetectReport {
    pub candidates_processed: usize,
    pub attached: usize,
    pub skipped_unchanged: usize,
}

pub async fn detect_page_candidates(
    db: &MemoryDB,
    tuning: &DistillationConfig,
) -> Result<DetectReport, WenlanError> {
    let candidate_bound = tuning.detect_max_candidates_per_tick;
    if candidate_bound == 0 {
        return Ok(DetectReport::default());
    }

    // DETECT is embeddings + cosine only, but it is not free at corpus scale:
    // greedy running-centroid clustering is O(N*C) per space (N staging
    // memories, C candidate clusters). After bucket removal and the retro
    // sweep, N can be the whole per-space staging pool, so each tick also has
    // an explicit cap on candidate attempts/writes.
    let clusters = db
        .find_distillation_clusters_scoped(
            tuning.formation_threshold,
            tuning.page_min_cluster_size,
            candidate_bound,
            tuning.ondevice_token_limit,
            tuning.max_unlinked_cluster_size,
            tuning.max_grouped_cluster_size,
            None,
            None,
        )
        .await?;

    let mut declined_memberships = load_declined_memberships(db).await?;
    let mut declined_memberships_changed = false;
    let mut report = DetectReport::default();
    for cluster in clusters {
        let signature = membership_signature(&cluster.source_ids);
        if declined_memberships.contains(&signature) {
            report.skipped_unchanged += 1;
            continue;
        }

        report.candidates_processed += 1;
        let Some(centroid) = cluster.centroid_embedding.as_deref() else {
            declined_memberships_changed |= declined_memberships.insert(signature);
            continue;
        };
        let Some(page) = db
            .find_matching_page_scoped(
                cluster.entity_id.as_deref(),
                centroid,
                tuning.page_match_threshold,
                cluster.space.as_deref(),
                false,
            )
            .await?
        else {
            declined_memberships_changed |= declined_memberships.insert(signature);
            continue;
        };
        if let Err(e) = page_write(
            db,
            PageWrite::Attach {
                page_id: &page.id,
                source_memory_ids: &cluster.source_ids,
                link_reason: "detect_attach",
                agent: "detect",
            },
        )
        .await
        {
            declined_memberships.insert(signature);
            save_declined_memberships(db, &declined_memberships).await?;
            return Err(e);
        }
        report.attached += 1;
    }
    if declined_memberships_changed {
        save_declined_memberships(db, &declined_memberships).await?;
    }
    Ok(report)
}

fn membership_signature(source_ids: &[String]) -> String {
    let mut ids = source_ids.to_vec();
    ids.sort();
    serde_json::to_string(&ids).unwrap_or_else(|_| "[]".to_string())
}

async fn load_declined_memberships(db: &MemoryDB) -> Result<BTreeSet<String>, WenlanError> {
    Ok(db
        .get_app_metadata(DETECT_DECLINED_MEMBERSHIPS_KEY)
        .await?
        .and_then(|raw| serde_json::from_str::<BTreeSet<String>>(&raw).ok())
        .unwrap_or_default())
}

async fn save_declined_memberships(
    db: &MemoryDB,
    memberships: &BTreeSet<String>,
) -> Result<(), WenlanError> {
    let raw = serde_json::to_string(memberships).unwrap_or_else(|_| "[]".to_string());
    db.set_app_metadata(DETECT_DECLINED_MEMBERSHIPS_KEY, &raw)
        .await
}

#[cfg(test)]
mod tests {
    use crate::db::MemoryDB;
    use crate::tuning::DistillationConfig;
    use libsql::params;

    fn vec_to_sql(v: &[f32]) -> String {
        let mut s = String::with_capacity(v.len() * 10);
        s.push('[');
        for (i, f) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            use std::fmt::Write;
            let _ = write!(s, "{f:.6}");
        }
        s.push(']');
        s
    }

    fn unit_vec(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; 768];
        v[axis] = 1.0;
        v
    }

    fn tuning_with_bound(bound: usize) -> DistillationConfig {
        DistillationConfig {
            formation_threshold: 0.80,
            page_min_cluster_size: 3,
            page_match_threshold: 0.80,
            detect_max_candidates_per_tick: bound,
            max_clusters_per_steep: 20,
            max_grouped_cluster_size: 20,
            max_unlinked_cluster_size: 20,
            ..Default::default()
        }
    }

    async fn insert_memory(
        db: &MemoryDB,
        source_id: &str,
        content: &str,
        space: &str,
        embedding: &[f32],
        last_modified: i64,
    ) {
        let embedding_sql = vec_to_sql(embedding);
        let conn = db.test_primary_session().await;
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
            params![
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

    async fn insert_page(
        db: &MemoryDB,
        page_id: &str,
        title: &str,
        space: &str,
        embedding: &[f32],
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        let embedding_sql = vec_to_sql(embedding);
        let conn = db.test_primary_session().await;
        conn.execute(
            "INSERT INTO pages (
                id, title, summary, content, space, source_memory_ids, version,
                status, embedding, created_at, last_compiled, last_modified,
                creation_kind, review_status, workspace
             )
             VALUES (
                ?1, ?2, ?2, ?2, ?3, '[]', 1, 'active', vector32(?4),
                ?5, ?5, ?5, 'distilled', 'confirmed', ?3
             )",
            params![page_id, title, space, embedding_sql.as_str(), now],
        )
        .await
        .unwrap();
    }

    async fn insert_candidate_cluster(
        db: &MemoryDB,
        prefix: &str,
        space: &str,
        topic: &str,
        embedding: &[f32],
        members: usize,
        base_last_modified: i64,
    ) {
        for i in 0..members {
            insert_memory(
                db,
                &format!("{prefix}_{i}"),
                &format!("{topic} supporting capture {i}"),
                space,
                embedding,
                base_last_modified + i as i64,
            )
            .await;
        }
    }

    async fn detect_link_count(db: &MemoryDB) -> i64 {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM page_sources WHERE link_reason = 'detect_attach'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get::<i64>(0).unwrap()
    }

    #[tokio::test]
    async fn detect_phase_processes_at_most_configured_bound_per_tick() {
        let (db, _db_dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().timestamp();

        for cluster in 0..5 {
            let embedding = unit_vec(cluster);
            let space = format!("detect_bound_space_{cluster}");
            let topic = format!("Detect bounded topic {cluster}");
            insert_page(
                &db,
                &format!("detect_bound_page_{cluster}"),
                &topic,
                &space,
                &embedding,
            )
            .await;
            insert_candidate_cluster(
                &db,
                &format!("detect_bound_mem_{cluster}"),
                &space,
                &topic,
                &embedding,
                3,
                now + (cluster as i64 * 10),
            )
            .await;
        }

        let report = super::detect_page_candidates(&db, &tuning_with_bound(2))
            .await
            .unwrap();

        assert_eq!(report.candidates_processed, 2);
        assert_eq!(report.attached, 2);
        assert_eq!(detect_link_count(&db).await, 6);
    }

    #[tokio::test]
    async fn detect_phase_skips_unchanged_failed_membership_until_new_member_joins() {
        let (db, _db_dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().timestamp();
        let embedding = unit_vec(25);
        let tuning = tuning_with_bound(5);

        insert_candidate_cluster(
            &db,
            "detect_retry_mem",
            "detect_retry_space",
            "Unmatched retry topic",
            &embedding,
            3,
            now,
        )
        .await;

        let first = super::detect_page_candidates(&db, &tuning).await.unwrap();
        assert_eq!(first.candidates_processed, 1);
        assert_eq!(first.attached, 0);
        assert_eq!(first.skipped_unchanged, 0);

        let second = super::detect_page_candidates(&db, &tuning).await.unwrap();
        assert_eq!(second.candidates_processed, 0);
        assert_eq!(second.attached, 0);
        assert_eq!(second.skipped_unchanged, 1);

        insert_memory(
            &db,
            "detect_retry_mem_new",
            "Unmatched retry topic supporting capture new",
            "detect_retry_space",
            &embedding,
            now + 100,
        )
        .await;

        let third = super::detect_page_candidates(&db, &tuning).await.unwrap();
        assert_eq!(third.candidates_processed, 1);
        assert_eq!(third.attached, 0);
        assert_eq!(third.skipped_unchanged, 0);
    }
}
