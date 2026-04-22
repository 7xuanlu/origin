// SPDX-License-Identifier: AGPL-3.0-only
//! Topic-matching: identifies whether an incoming memory is an update to an
//! existing topic, enabling in-place upsert instead of creating duplicates.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::tuning::TopicMatchConfig;

/// Result of a topic-match check.
#[derive(Debug)]
pub struct TopicMatchResult {
    /// The matched memory's source_id, if a topic match was found.
    pub matched_source_id: Option<String>,
    /// The matched memory's content at the time of matching (for changelog delta).
    pub old_content: Option<String>,
    /// The matched memory's embedding at the time of matching (for change classification).
    pub old_embedding: Option<Vec<f32>>,
    /// Signals that contributed to the match decision (for logging / debugging).
    pub signals: MatchSignals,
}

/// Signals that contributed to the match decision.
#[derive(Debug, Default)]
pub struct MatchSignals {
    pub entity_match: bool,
    pub fts_title_hit: bool,
    pub embedding_similarity: Option<f64>,
    pub candidate_count: usize,
}

/// A lightweight candidate memory for topic matching.
#[derive(Debug, Clone)]
pub struct TopicMatchCandidate {
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub entity_id: Option<String>,
    pub embedding: Vec<f32>,
}

/// Check if an incoming memory matches an existing topic for in-place upsert.
///
/// Runs pre-batcher in `handle_store_memory`. Returns the matched memory's
/// source_id if a topic match is found, `None` otherwise.
///
/// Matching strategy (in priority order):
/// 1. Entity ID overlap (strongest signal — same entity = same topic)
/// 2. FTS5 title hit AND embedding similarity above threshold
/// 3. Embedding-only similarity above threshold (weaker fallback)
pub async fn find_topic_match(
    db: &MemoryDB,
    title: &str,
    memory_type: Option<&str>,
    domain: Option<&str>,
    entity_id: Option<&str>,
    content_embedding: &[f32],
    config: &TopicMatchConfig,
) -> Result<TopicMatchResult, OriginError> {
    let no_match = TopicMatchResult {
        matched_source_id: None,
        old_content: None,
        old_embedding: None,
        signals: MatchSignals::default(),
    };

    // Both domain and memory_type must be present for a meaningful topic match.
    let (Some(_d), Some(_mt)) = (domain, memory_type) else {
        return Ok(no_match);
    };

    // Step 1: Fetch candidates (same domain + same memory_type, most recent first).
    let candidates = db
        .topic_match_candidates(domain, memory_type, config.max_candidates)
        .await?;

    if candidates.is_empty() {
        return Ok(no_match);
    }

    let mut signals = MatchSignals {
        candidate_count: candidates.len(),
        ..Default::default()
    };

    // Step 2: Entity ID overlap (strongest signal).
    if let Some(eid) = entity_id {
        if let Some(matched) = candidates
            .iter()
            .find(|c| c.entity_id.as_deref() == Some(eid))
        {
            signals.entity_match = true;
            log::info!(
                "[topic_match] entity match: entity={eid} → source_id={}",
                matched.source_id
            );
            return Ok(TopicMatchResult {
                matched_source_id: Some(matched.source_id.clone()),
                old_content: Some(matched.content.clone()),
                old_embedding: Some(matched.embedding.clone()),
                signals,
            });
        }
    }

    // Step 3: FTS5 title query against memories_fts (DB-backed, uses title: column filter).
    let candidate_ids: Vec<&str> = candidates.iter().map(|c| c.source_id.as_str()).collect();
    let fts_hits: std::collections::HashSet<String> = db
        .topic_match_title_fts(title, &candidate_ids)
        .await?
        .into_iter()
        .collect();

    // Rank candidates by embedding similarity.
    let mut ranked: Vec<(&TopicMatchCandidate, f64)> = candidates
        .iter()
        .map(|c| {
            let sim = cosine_similarity(content_embedding, &c.embedding);
            (c, sim)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (candidate, similarity) in &ranked {
        let title_hit = fts_hits.contains(&candidate.source_id);

        if similarity > &config.embedding_threshold {
            if title_hit {
                signals.fts_title_hit = true;
                signals.embedding_similarity = Some(*similarity);
                log::info!(
                    "[topic_match] title+embedding match: sim={:.3} source_id={}",
                    similarity,
                    candidate.source_id
                );
            } else {
                signals.embedding_similarity = Some(*similarity);
                log::info!(
                    "[topic_match] embedding-only match: sim={:.3} source_id={}",
                    similarity,
                    candidate.source_id
                );
            }
            return Ok(TopicMatchResult {
                matched_source_id: Some(candidate.source_id.clone()),
                old_content: Some(candidate.content.clone()),
                old_embedding: Some(candidate.embedding.clone()),
                signals,
            });
        }
    }

    Ok(no_match)
}

/// Compute cosine similarity between two f32 embedding vectors.
/// Returns 0.0 for empty or mismatched-length vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_similarity_mismatched() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
