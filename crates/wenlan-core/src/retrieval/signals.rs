/// T8 salience prior multiplier. Maps a per-memory `importance` rating (1-10,
/// LLM-assigned at write time) to a multiplicative ranking factor in
/// `[floor, ceil]`.
///
/// - `None` short-circuits to exactly `1.0` (neutral) — NOT the band midpoint —
///   so cold-start rows with no importance never move in ranking. This is the
///   load-bearing safety property: the band default `[0.85, 1.15]` straddles 1.0
///   but its midpoint is 1.0 only by coincidence; relying on it would silently
///   demote/boost un-rated rows. The explicit short-circuit guarantees neutrality.
/// - `Some(i)` clamps `i` to `[1, 10]`, then linearly maps `1 -> floor`,
///   `10 -> ceil`. Strictly monotone in `i`. Never panics.
///
/// Plain `pub(crate) fn` matching the other signal helpers; wired into the
/// `search_memory` ranking closure behind `db::salience_prior_enabled()`.
pub(crate) fn salience_multiplier(importance: Option<u8>, floor: f64, ceil: f64) -> f64 {
    match importance {
        None => 1.0,
        Some(i) => {
            let i = i.clamp(1, 10) as f64;
            floor + (i - 1.0) / 9.0 * (ceil - floor)
        }
    }
}

/// Temporal SOFT-boost multiplier: binary in-window boost.
///
/// Returns `1.0 + bonus` when `event_date` is `Some(t)` and `t` falls inside the
/// inclusive parsed query window `[window_start, window_end]`; returns `1.0`
/// (neutral) for outside-window dated memories AND for `None` (undated) rows.
/// The result is NEVER less than `1.0`, so a memory can only be lifted, never
/// demoted or excluded — the load-bearing safety property of the soft boost
/// (cf. the T4a hard filter, which dropped outside-window rows entirely).
///
/// Plain `pub(crate) fn` matching the other signal helpers; wired into the
/// `search_memory` ranking closure behind `db::temporal_soft_boost_enabled()`.
pub(crate) fn temporal_interval_boost(
    event_date: Option<i64>,
    window_start: i64,
    window_end: i64,
    bonus: f64,
) -> f64 {
    match event_date {
        Some(d) if (window_start..=window_end).contains(&d) => 1.0 + bonus,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// T9 — Wide-pool-seeded graph expansion helpers
// ---------------------------------------------------------------------------

/// Parse `WENLAN_GRAPH_HOP_DEPTH` → usize in [0, 3]. Default 1 on unset or parse failure.
pub(crate) fn parse_hop_depth(val: Option<&str>) -> usize {
    let raw = match val {
        Some(s) => s,
        None => return 1,
    };
    match raw.trim().parse::<isize>() {
        Ok(n) if n >= 0 => (n as usize).min(3),
        _ => 1,
    }
}

/// Parse `WENLAN_GRAPH_SEED_TOP_K` → usize in [1, 50]. Default 10 on unset or parse failure.
pub(crate) fn parse_seed_top_k(val: Option<&str>) -> usize {
    let raw = match val {
        Some(s) => s,
        None => return 10,
    };
    match raw.trim().parse::<isize>() {
        Ok(n) if n >= 1 => (n as usize).min(50),
        _ => 10,
    }
}

/// Parse `WENLAN_GRAPH_FRONTIER_CAP` → usize in [1, 512]. Default 64 on unset or parse failure.
pub(crate) fn parse_frontier_cap(val: Option<&str>) -> usize {
    let raw = match val {
        Some(s) => s,
        None => return 64,
    };
    match raw.trim().parse::<isize>() {
        Ok(n) if n >= 1 => (n as usize).min(512),
        _ => 64,
    }
}

/// Dedup entity IDs by best (lowest) rank, sort ascending by rank, truncate to `top_k`.
///
/// Input: `(entity_id, rank)` pairs from the wide pool (rank = enumeration index).
/// Multiple entries for the same entity keep only the lowest rank (best position).
/// Empty pool → empty result (no panic).
pub(crate) fn seed_entities_by_rank(pool: &[(String, usize)], top_k: usize) -> Vec<String> {
    if pool.is_empty() || top_k == 0 {
        return Vec::new();
    }
    // dedup by best (lowest) rank
    let mut best: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (id, rank) in pool {
        let entry = best.entry(id.as_str()).or_insert(*rank);
        if *rank < *entry {
            *entry = *rank;
        }
    }
    let mut sorted: Vec<(&str, usize)> = best.into_iter().collect();
    sorted.sort_by_key(|(_, rank)| *rank);
    sorted
        .into_iter()
        .take(top_k)
        .map(|(id, _)| id.to_string())
        .collect()
}

/// Capitalized words that commonly start a query but are NOT entity anchors
/// (question words, articles, pronouns, common verbs). Compared lowercased.
const NON_ENTITY_CAP_WORDS: &[&str] = &[
    "what", "when", "where", "who", "why", "how", "which", "whose", "whom", "is", "are", "was",
    "were", "do", "does", "did", "the", "a", "an", "i", "my", "me", "you", "your", "we", "our",
    "it", "this", "that", "can", "could", "should", "would", "tell", "show", "find", "give",
    "list", "search",
];

/// Zero-LLM check: does the query contain an entity anchor — a quoted phrase or a
/// capitalized non-stopword token (proper noun)? Mirrors the entity-presence gate
/// Mem0 / agentmemory use to decide whether a graph hop is worthwhile.
pub(crate) fn query_has_entity_anchor(query: &str) -> bool {
    // Quoted phrase: at least one pair of double quotes.
    if query.matches('"').count() >= 2 {
        return true;
    }
    query.split_whitespace().any(|raw| {
        let tok = raw.trim_matches(|c: char| !c.is_alphanumeric());
        match tok.chars().next() {
            Some(first) if first.is_uppercase() && tok.chars().count() >= 2 => {
                !NON_ENTITY_CAP_WORDS.contains(&tok.to_lowercase().as_str())
            }
            _ => false,
        }
    })
}

/// Cheap, zero-LLM predicate gating the otherwise-unconditional
/// `augment_with_graph` call (opt-in via `WENLAN_ENABLE_GRAPH_GATE`). Returns true
/// when the query is worth a graph hop: relational/temporal phrasing
/// (`classify_query().use_graph`) OR an entity anchor. Single-fact lookups like
/// "what is the database password" return false so the graph entity-embedding +
/// traversal is skipped.
///
/// KNOWN LIMITATION (why the gate defaults OFF and must be eval-gated before it
/// is ever defaulted ON): the entity anchor relies on capitalization, so an
/// entity mentioned in all-lowercase with no relational/temporal keyword (e.g.
/// "tell me about libsql") is treated as a single-fact lookup and its graph hop
/// is skipped when the gate is ON. This is a recall trade-off, not a "conservative
/// in all cases" guarantee — measure per-category (N>=3) before flipping the
/// default. A gazetteer of common lowercase proper nouns is a deliberate follow-up.
pub(crate) fn query_warrants_graph(query: &str) -> bool {
    if crate::router::classify::classify_query(query, "", "", false).use_graph {
        return true;
    }
    query_has_entity_anchor(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- T3 graph-gate predicate (WENLAN_ENABLE_GRAPH_GATE) ---

    #[test]
    fn graph_gate_fires_on_relational_phrasing() {
        assert!(query_warrants_graph(
            "what is the relationship between Alice and Bob"
        ));
    }

    #[test]
    fn graph_gate_fires_on_temporal_phrasing() {
        // lowercase, no proper noun, but "changed recently" is a temporal cue.
        assert!(query_warrants_graph("what changed recently in the project"));
    }

    #[test]
    fn graph_gate_fires_on_capitalized_entity_anchor() {
        // No relational/temporal keyword, but "Postgres" is a proper-noun entity.
        assert!(query_warrants_graph("what is Postgres"));
    }

    #[test]
    fn graph_gate_fires_on_quoted_phrase() {
        assert!(query_warrants_graph(
            r#"find notes about "machine learning""#
        ));
    }

    #[test]
    fn graph_gate_skips_plain_single_fact() {
        // No keyword, no proper-noun entity (stopwords + lowercase) → skip graph.
        assert!(!query_warrants_graph("what is the database password"));
    }

    #[test]
    fn entity_anchor_ignores_leading_question_word() {
        // "What"/"Where" are capitalized sentence-initial stopwords, not entities.
        assert!(!query_has_entity_anchor("What is the password"));
        assert!(!query_has_entity_anchor("Where do i keep notes"));
    }

    #[test]
    fn entity_anchor_detects_proper_noun() {
        assert!(query_has_entity_anchor("tell me about Alice"));
        assert!(query_has_entity_anchor("the React migration"));
    }

    #[test]
    fn entity_anchor_detects_quoted_phrase() {
        assert!(query_has_entity_anchor(r#"search for "exact phrase" here"#));
    }

    #[test]
    fn entity_anchor_false_on_all_lowercase() {
        assert!(!query_has_entity_anchor("what is the database password"));
    }

    // ---------------------------------------------------------------------------
    // T9 pure-helper tests (no DB needed)
    // ---------------------------------------------------------------------------

    #[test]
    fn graph_seed_top_k_ranks_by_carrier_rank() {
        let pool = vec![
            ("a".to_string(), 5usize),
            ("b".to_string(), 1usize),
            ("a".to_string(), 0usize),
            ("c".to_string(), 3usize),
        ];
        let result = seed_entities_by_rank(&pool, 2);
        assert_eq!(result, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn graph_seed_top_k_empty_pool_is_empty() {
        assert!(seed_entities_by_rank(&[], 10).is_empty());
    }

    #[test]
    fn graph_seed_top_k_truncates_to_k() {
        let pool: Vec<(String, usize)> = (0..20).map(|i| (format!("e{}", i), i)).collect();
        let result = seed_entities_by_rank(&pool, 5);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], "e0");
        assert_eq!(result[4], "e4");
    }

    #[test]
    fn graph_hop_depth_parse_clamps() {
        assert_eq!(parse_hop_depth(None), 1);
        assert_eq!(parse_hop_depth(Some("0")), 0);
        assert_eq!(parse_hop_depth(Some("2")), 2);
        assert_eq!(parse_hop_depth(Some("3")), 3);
        assert_eq!(parse_hop_depth(Some("99")), 3);
        assert_eq!(parse_hop_depth(Some("-1")), 1);
        assert_eq!(parse_hop_depth(Some("abc")), 1);
    }

    #[test]
    fn graph_frontier_cap_parse_defaults() {
        assert_eq!(parse_frontier_cap(None), 64);
        assert_eq!(parse_frontier_cap(Some("16")), 16);
        assert_eq!(parse_frontier_cap(Some("999")), 512);
        assert_eq!(parse_frontier_cap(Some("0")), 64);
        assert_eq!(parse_frontier_cap(Some("bad")), 64);
    }

    #[test]
    fn graph_seed_top_k_parse_defaults() {
        assert_eq!(parse_seed_top_k(None), 10);
        assert_eq!(parse_seed_top_k(Some("5")), 5);
        assert_eq!(parse_seed_top_k(Some("100")), 50);
        assert_eq!(parse_seed_top_k(Some("0")), 10);
        assert_eq!(parse_seed_top_k(Some("bad")), 10);
    }

    // ---------------------------------------------------------------------------
    // T8 — salience prior multiplier (L1 tests)
    // ---------------------------------------------------------------------------

    #[test]
    fn salience_none_is_neutral() {
        // None short-circuits to exactly 1.0, NOT the band midpoint.
        assert_eq!(salience_multiplier(None, 0.85, 1.15), 1.0);
    }

    #[test]
    fn salience_max_is_ceil() {
        assert!((salience_multiplier(Some(10), 0.85, 1.15) - 1.15).abs() < 1e-6);
    }

    #[test]
    fn salience_min_is_floor() {
        assert!((salience_multiplier(Some(1), 0.85, 1.15) - 0.85).abs() < 1e-6);
    }

    #[test]
    fn salience_mid_near_one() {
        // importance 5 and 6 straddle the band center; both within ~0.05 of 1.0.
        let m5 = salience_multiplier(Some(5), 0.85, 1.15);
        let m6 = salience_multiplier(Some(6), 0.85, 1.15);
        assert!((m5 - 1.0).abs() < 0.05, "m5={m5}");
        assert!((m6 - 1.0).abs() < 0.05, "m6={m6}");
        // 5 is just below 1.0, 6 just above.
        assert!(m5 < 1.0 && m6 > 1.0, "m5={m5} m6={m6}");
    }

    #[test]
    fn salience_monotone() {
        let a = salience_multiplier(Some(1), 0.85, 1.15);
        let b = salience_multiplier(Some(5), 0.85, 1.15);
        let c = salience_multiplier(Some(10), 0.85, 1.15);
        assert!(a < b, "1 < 5: {a} < {b}");
        assert!(b < c, "5 < 10: {b} < {c}");
    }

    #[test]
    fn salience_clamps_out_of_range() {
        // 0 clamps to floor (treated as 1), >10 clamps to ceil (treated as 10).
        assert!((salience_multiplier(Some(0), 0.85, 1.15) - 0.85).abs() < 1e-6);
        assert!((salience_multiplier(Some(255), 0.85, 1.15) - 1.15).abs() < 1e-6);
        // never panics, always within [floor, ceil].
        for i in 0u8..=255 {
            let m = salience_multiplier(Some(i), 0.85, 1.15);
            assert!((0.85..=1.15).contains(&m), "i={i} m={m}");
        }
    }

    // ---------------------------------------------------------------------------
    // Temporal SOFT-boost — binary in-window boost multiplier (L1 unit test)
    // ---------------------------------------------------------------------------

    /// In-window dated event -> `1.0 + bonus`; outside-window -> `1.0`;
    /// `None` (undated) -> `1.0`. Result is NEVER < 1.0 for any input — the
    /// soft boost only lifts, never demotes or excludes.
    ///
    /// RED until impl: the stub returns 1.0 even for in-window dates, so the
    /// in-window assertion (`1.0 + bonus`) fails.
    #[test]
    fn temporal_interval_boost_unit() {
        let start = 1_779_724_800; // 2026-05-26 00:00:00Z (window low)
        let end = 1_779_811_199; // 2026-05-26 23:59:59Z (window high)
        let bonus = 0.5;

        // In-window dated event -> boosted.
        let in_window = temporal_interval_boost(Some(1_779_775_200), start, end, bonus);
        assert!(
            (in_window - (1.0 + bonus)).abs() < 1e-9,
            "in-window event must be boosted to 1.0 + bonus; got {in_window}"
        );

        // Inclusive boundaries are in-window.
        assert!(
            (temporal_interval_boost(Some(start), start, end, bonus) - (1.0 + bonus)).abs() < 1e-9,
            "lower boundary must be in-window"
        );
        assert!(
            (temporal_interval_boost(Some(end), start, end, bonus) - (1.0 + bonus)).abs() < 1e-9,
            "upper boundary must be in-window"
        );

        // Outside-window dated event -> neutral 1.0.
        assert_eq!(
            temporal_interval_boost(Some(1_779_602_400), start, end, bonus),
            1.0,
            "outside-window event must be neutral (1.0)"
        );

        // Undated (None) -> neutral 1.0.
        assert_eq!(
            temporal_interval_boost(None, start, end, bonus),
            1.0,
            "undated (None) event must be neutral (1.0)"
        );

        // Safety invariant: NEVER < 1.0 for any input (sampled across the line).
        for ed in [
            None,
            Some(i64::MIN),
            Some(start - 1),
            Some(start),
            Some(1_779_775_200),
            Some(end),
            Some(end + 1),
            Some(i64::MAX),
        ] {
            let m = temporal_interval_boost(ed, start, end, bonus);
            assert!(
                m >= 1.0,
                "multiplier must never drop below 1.0; ed={ed:?} m={m}"
            );
        }
    }

    #[test]
    fn graph_seed_enabled_truthy_table() {
        for val in &["1", "true", "yes"] {
            std::env::set_var("WENLAN_ENABLE_GRAPH_SEED", val);
            assert!(crate::db::graph_seed_enabled(), "expected true for {val}");
        }
        for val in &["0", "false", ""] {
            std::env::set_var("WENLAN_ENABLE_GRAPH_SEED", val);
            assert!(!crate::db::graph_seed_enabled(), "expected false for {val}");
        }
        std::env::remove_var("WENLAN_ENABLE_GRAPH_SEED");
        assert!(
            !crate::db::graph_seed_enabled(),
            "expected false when unset"
        );
    }
}
