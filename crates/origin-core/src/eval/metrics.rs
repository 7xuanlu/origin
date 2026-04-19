// SPDX-License-Identifier: AGPL-3.0-only
//! IR metrics: MRR, NDCG@K, Precision@K, negative leakage.

use std::collections::{HashMap, HashSet};

/// Mean Reciprocal Rank — how high does the first relevant result rank?
pub fn mrr(ranked_ids: &[&str], relevant_ids: &HashSet<&str>) -> f64 {
    for (i, id) in ranked_ids.iter().enumerate() {
        if relevant_ids.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Normalized Discounted Cumulative Gain at K.
/// `relevance_grades` maps ID → grade (0-3).
pub fn ndcg_at_k(ranked_ids: &[&str], relevance_grades: &HashMap<&str, u8>, k: usize) -> f64 {
    let k = k.min(ranked_ids.len());
    if k == 0 {
        return 0.0;
    }

    // DCG: sum of (2^rel - 1) / log2(i + 2) for i in 0..k
    let dcg: f64 = ranked_ids[..k]
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let rel = *relevance_grades.get(id).unwrap_or(&0) as f64;
            (2.0f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    // Ideal DCG: sort grades descending, compute same formula
    let mut ideal_grades: Vec<f64> = relevance_grades.values().map(|&g| g as f64).collect();
    ideal_grades.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let ideal_k = k.min(ideal_grades.len());
    let idcg: f64 = ideal_grades[..ideal_k]
        .iter()
        .enumerate()
        .map(|(i, &rel)| (2.0f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Precision at K — fraction of top-K results that are relevant.
pub fn precision_at_k(ranked_ids: &[&str], relevant_ids: &HashSet<&str>, k: usize) -> f64 {
    let k = k.min(ranked_ids.len());
    if k == 0 {
        return 0.0;
    }
    let hits = ranked_ids[..k]
        .iter()
        .filter(|id| relevant_ids.contains(*id))
        .count();
    hits as f64 / k as f64
}

/// Mean Average Precision at K.
/// AP = (1/min(|R|,k)) * sum_{i=1}^{k} [P(i) * rel(i)]
/// where R = total relevant docs, P(i) = precision at position i.
/// Follows the TREC definition: denominator is min(|relevant|, k) to handle
/// cases where there are more relevant docs than k.
pub fn map_at_k(ranked_ids: &[&str], relevant_ids: &HashSet<&str>, k: usize) -> f64 {
    let k = k.min(ranked_ids.len());
    if k == 0 || relevant_ids.is_empty() {
        return 0.0;
    }

    let mut hits = 0;
    let mut sum_precision = 0.0;

    for (i, ranked_id) in ranked_ids.iter().enumerate().take(k) {
        if relevant_ids.contains(ranked_id) {
            hits += 1;
            sum_precision += hits as f64 / (i as f64 + 1.0);
        }
    }

    let denominator = (relevant_ids.len() as f64).min(k as f64);
    sum_precision / denominator
}

/// Recall at K — fraction of all relevant docs found in top-K.
/// Recall@k = |relevant ∩ top_k| / |relevant|
pub fn recall_at_k(ranked_ids: &[&str], relevant_ids: &HashSet<&str>, k: usize) -> f64 {
    let k = k.min(ranked_ids.len());
    if relevant_ids.is_empty() {
        return 0.0;
    }
    let found = ranked_ids[..k]
        .iter()
        .filter(|id| relevant_ids.contains(*id))
        .count();
    found as f64 / relevant_ids.len() as f64
}

/// Hit Rate at K (Success@k) — binary: 1.0 if at least one relevant doc in top-K, else 0.0.
pub fn hit_rate_at_k(ranked_ids: &[&str], relevant_ids: &HashSet<&str>, k: usize) -> f64 {
    let k = k.min(ranked_ids.len());
    if ranked_ids[..k].iter().any(|id| relevant_ids.contains(id)) {
        1.0
    } else {
        0.0
    }
}

/// Negative leakage — count of negative IDs that appear in top-K results.
pub fn negative_leakage(ranked_ids: &[&str], negative_ids: &HashSet<&str>, k: usize) -> usize {
    let k = k.min(ranked_ids.len());
    ranked_ids[..k]
        .iter()
        .filter(|id| negative_ids.contains(*id))
        .count()
}

/// Temporal ordering — does the newer memory rank above the older one?
/// Returns true if newer_id appears before older_id in results,
/// or if newer_id is present but older_id is absent.
/// Returns false if both are absent or if older ranks above newer.
pub fn temporal_ordering(ranked_ids: &[&str], newer_id: &str, older_id: &str) -> bool {
    let newer_pos = ranked_ids.iter().position(|&id| id == newer_id);
    let older_pos = ranked_ids.iter().position(|&id| id == older_id);

    match (newer_pos, older_pos) {
        (Some(n), Some(o)) => n < o,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => false,
    }
}

/// Archive leakage — fraction of archived memories that still appear in search results.
/// `search_results_per_archived` is a vec of (archived_id, top-K result IDs).
/// An archived memory "leaks" if it appears in the search results for its own content.
pub fn archive_leakage(
    archived_ids: &HashSet<&str>,
    search_results_per_archived: &[(&str, Vec<&str>)],
) -> f64 {
    if archived_ids.is_empty() {
        return 0.0;
    }
    let leaked = search_results_per_archived
        .iter()
        .filter(|(archived_id, results)| results.contains(archived_id))
        .count();
    leaked as f64 / archived_ids.len() as f64
}

/// Negatives above relevant — count of negative IDs that rank above the LAST relevant result.
/// This measures ranking quality: negatives below all positives don't count as leakage.
/// Returns 0 when no relevant results appear in rankings (nothing to leak above).
pub fn negatives_above_relevant(
    ranked_ids: &[&str],
    relevant_ids: &HashSet<&str>,
    negative_ids: &HashSet<&str>,
) -> usize {
    // Find position of last relevant result
    let last_relevant_pos = ranked_ids
        .iter()
        .enumerate()
        .filter(|(_, id)| relevant_ids.contains(*id))
        .map(|(i, _)| i)
        .next_back();

    match last_relevant_pos {
        Some(pos) => {
            // Count negatives before or at this position
            ranked_ids[..=pos]
                .iter()
                .filter(|id| negative_ids.contains(*id))
                .count()
        }
        None => 0, // No relevant results → can't leak above them
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrr_first_result_relevant() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["a"].into_iter().collect();
        assert_eq!(mrr(&ranked, &relevant), 1.0);
    }

    #[test]
    fn test_mrr_second_result_relevant() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["b"].into_iter().collect();
        assert_eq!(mrr(&ranked, &relevant), 0.5);
    }

    #[test]
    fn test_mrr_no_relevant() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["x"].into_iter().collect();
        assert_eq!(mrr(&ranked, &relevant), 0.0);
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        let ranked = vec!["a", "b", "c"];
        let mut grades = HashMap::new();
        grades.insert("a", 3u8);
        grades.insert("b", 2);
        grades.insert("c", 1);
        let score = ndcg_at_k(&ranked, &grades, 3);
        assert!(
            (score - 1.0).abs() < 0.001,
            "Perfect ranking should give NDCG ≈ 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_ndcg_reversed_ranking() {
        let ranked = vec!["c", "b", "a"];
        let mut grades = HashMap::new();
        grades.insert("a", 3u8);
        grades.insert("b", 2);
        grades.insert("c", 1);
        let score = ndcg_at_k(&ranked, &grades, 3);
        assert!(
            score < 1.0,
            "Reversed ranking should give NDCG < 1.0, got {}",
            score
        );
        assert!(
            score > 0.0,
            "Reversed ranking should give NDCG > 0.0, got {}",
            score
        );
    }

    #[test]
    fn test_ndcg_empty() {
        let ranked: Vec<&str> = vec![];
        let grades = HashMap::new();
        assert_eq!(ndcg_at_k(&ranked, &grades, 5), 0.0);
    }

    #[test]
    fn test_precision_at_k_all_relevant() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        assert_eq!(precision_at_k(&ranked, &relevant, 3), 1.0);
    }

    #[test]
    fn test_precision_at_k_none_relevant() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["x"].into_iter().collect();
        assert_eq!(precision_at_k(&ranked, &relevant, 3), 0.0);
    }

    #[test]
    fn test_precision_at_k_partial() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["a", "c"].into_iter().collect();
        let p = precision_at_k(&ranked, &relevant, 3);
        assert!((p - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_negative_leakage_none() {
        let ranked = vec!["a", "b", "c"];
        let negatives: HashSet<&str> = ["x"].into_iter().collect();
        assert_eq!(negative_leakage(&ranked, &negatives, 3), 0);
    }

    #[test]
    fn test_negative_leakage_some() {
        let ranked = vec!["a", "neg1", "c", "neg2"];
        let negatives: HashSet<&str> = ["neg1", "neg2"].into_iter().collect();
        assert_eq!(negative_leakage(&ranked, &negatives, 3), 1);
        assert_eq!(negative_leakage(&ranked, &negatives, 4), 2);
    }

    #[test]
    fn test_negatives_above_relevant_none_above() {
        // All positives before negatives → 0 leakage
        let ranked = vec!["a", "b", "neg1"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        let negatives: HashSet<&str> = ["neg1"].into_iter().collect();
        assert_eq!(negatives_above_relevant(&ranked, &relevant, &negatives), 0);
    }

    #[test]
    fn test_negatives_above_relevant_one_above() {
        // neg1 ranks between two positives → 1 leakage
        let ranked = vec!["a", "neg1", "b"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        let negatives: HashSet<&str> = ["neg1"].into_iter().collect();
        assert_eq!(negatives_above_relevant(&ranked, &relevant, &negatives), 1);
    }

    #[test]
    fn test_negatives_above_relevant_below_all() {
        // Negative below all relevant → 0
        let ranked = vec!["a", "b", "neg1", "neg2"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        let negatives: HashSet<&str> = ["neg1", "neg2"].into_iter().collect();
        assert_eq!(negatives_above_relevant(&ranked, &relevant, &negatives), 0);
    }

    #[test]
    fn test_negatives_above_relevant_no_relevant() {
        let ranked = vec!["neg1", "neg2"];
        let relevant: HashSet<&str> = HashSet::new();
        let negatives: HashSet<&str> = ["neg1", "neg2"].into_iter().collect();
        assert_eq!(negatives_above_relevant(&ranked, &relevant, &negatives), 0);
    }

    #[test]
    fn test_map_perfect_ranking() {
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        let score = map_at_k(&ranked, &relevant, 4);
        assert!(
            (score - 1.0).abs() < 0.001,
            "Perfect MAP should be 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_map_interleaved() {
        let ranked = vec!["a", "neg", "b", "neg2"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        let score = map_at_k(&ranked, &relevant, 4);
        assert!(
            (score - 0.833).abs() < 0.01,
            "Interleaved MAP should be ~0.833, got {}",
            score
        );
    }

    #[test]
    fn test_map_no_relevant() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["x"].into_iter().collect();
        assert_eq!(map_at_k(&ranked, &relevant, 3), 0.0);
    }

    #[test]
    fn test_map_relevant_beyond_k() {
        let ranked = vec!["neg", "a", "b"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        let score = map_at_k(&ranked, &relevant, 2);
        assert!(
            (score - 0.25).abs() < 0.01,
            "MAP@2 should be ~0.25, got {}",
            score
        );
    }

    #[test]
    fn test_recall_all_found() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        assert!((recall_at_k(&ranked, &relevant, 3) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_partial() {
        let ranked = vec!["a", "neg", "neg2", "b"];
        let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
        assert!((recall_at_k(&ranked, &relevant, 2) - 0.5).abs() < 0.001);
        assert!((recall_at_k(&ranked, &relevant, 4) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_recall_none_found() {
        let ranked = vec!["a", "b"];
        let relevant: HashSet<&str> = ["x", "y"].into_iter().collect();
        assert_eq!(recall_at_k(&ranked, &relevant, 2), 0.0);
    }

    #[test]
    fn test_recall_no_relevant() {
        let ranked = vec!["a", "b"];
        let relevant: HashSet<&str> = HashSet::new();
        assert_eq!(recall_at_k(&ranked, &relevant, 2), 0.0);
    }

    #[test]
    fn test_hit_rate_hit() {
        let ranked = vec!["neg", "a", "neg2"];
        let relevant: HashSet<&str> = ["a"].into_iter().collect();
        assert_eq!(hit_rate_at_k(&ranked, &relevant, 3), 1.0);
        assert_eq!(hit_rate_at_k(&ranked, &relevant, 1), 0.0);
    }

    #[test]
    fn test_hit_rate_miss() {
        let ranked = vec!["neg", "neg2"];
        let relevant: HashSet<&str> = ["x"].into_iter().collect();
        assert_eq!(hit_rate_at_k(&ranked, &relevant, 2), 0.0);
    }

    #[test]
    fn test_temporal_ordering_newer_first() {
        let ranked = vec!["new_1", "old_1", "other"];
        assert!(temporal_ordering(&ranked, "new_1", "old_1"));
    }

    #[test]
    fn test_temporal_ordering_older_first() {
        let ranked = vec!["old_1", "new_1", "other"];
        assert!(!temporal_ordering(&ranked, "new_1", "old_1"));
    }

    #[test]
    fn test_temporal_ordering_newer_missing() {
        let ranked = vec!["old_1", "other"];
        assert!(!temporal_ordering(&ranked, "new_1", "old_1"));
    }

    #[test]
    fn test_temporal_ordering_older_missing() {
        let ranked = vec!["new_1", "other"];
        assert!(temporal_ordering(&ranked, "new_1", "old_1"));
    }

    #[test]
    fn test_temporal_ordering_both_missing() {
        let ranked = vec!["other1", "other2"];
        assert!(!temporal_ordering(&ranked, "new_1", "old_1"));
    }

    #[test]
    fn test_archive_leakage_none() {
        let archived: HashSet<&str> = ["a1", "a2"].into_iter().collect();
        let results = vec![
            ("a1", vec!["merged_1", "other"]),
            ("a2", vec!["merged_2", "other"]),
        ];
        assert_eq!(archive_leakage(&archived, &results), 0.0);
    }

    #[test]
    fn test_archive_leakage_one_leaked() {
        let archived: HashSet<&str> = ["a1", "a2"].into_iter().collect();
        let results = vec![
            ("a1", vec!["a1", "merged_1"]),
            ("a2", vec!["merged_2", "other"]),
        ];
        assert!((archive_leakage(&archived, &results) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_archive_leakage_all_leaked() {
        let archived: HashSet<&str> = ["a1", "a2"].into_iter().collect();
        let results = vec![("a1", vec!["a1", "other"]), ("a2", vec!["a2", "other"])];
        assert!((archive_leakage(&archived, &results) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_archive_leakage_empty() {
        let archived: HashSet<&str> = HashSet::new();
        let results: Vec<(&str, Vec<&str>)> = vec![];
        assert_eq!(archive_leakage(&archived, &results), 0.0);
    }
}
