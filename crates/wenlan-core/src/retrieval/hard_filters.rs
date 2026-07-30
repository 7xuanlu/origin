// SPDX-License-Identifier: Apache-2.0
//! Hard filter cascade: supersession exclusion, temporal cue window, space,
//! and memory_type. Produces a SQL WHERE snippet (prefix ` AND ...`) suitable
//! for appending after a `WHERE 1=1` base clause on the `memories c` alias.
//!
//! Moved out of composite/ in PR-A (2026-05-27); used by both legacy search_memory* paths and future page-channel callers.

use crate::temporal_query::{CueConfidence, ExtractedCue};

/// Declarative hard-filter spec. Each enabled filter becomes a mandatory
/// WHERE clause. All fields are `pub(crate)` so callers in this crate can
/// construct by name.
#[allow(dead_code)]
pub(crate) struct HardFilters<'a> {
    /// Filter to a specific space (column `c.space`). `None` = no space filter.
    pub(crate) space: Option<&'a str>,
    /// Filter to a specific memory_type. `None` = no type filter.
    pub(crate) memory_type: Option<&'a str>,
    /// When `true`, hide superseded memories (supersede_mode = 'hide').
    /// Uses the verbatim subquery from db.rs search_memory_llm_rerank.
    pub(crate) exclude_superseded: bool,
    /// When `Some` and confidence is `High`, constrain `c.event_date` to the
    /// cue's range (OR allow NULL so undated memories pass through).
    /// `Low`-confidence cues are not applied as hard filters.
    pub(crate) temporal_cue: Option<ExtractedCue>,
}

impl<'a> HardFilters<'a> {
    /// All-permissive default: no filters applied. Useful for callers that want
    /// to start open and selectively add constraints (e.g. page-channel callers
    /// in follow-up PRs).
    #[allow(dead_code)]
    pub(crate) fn default_open() -> HardFilters<'static> {
        HardFilters {
            space: None,
            memory_type: None,
            exclude_superseded: false,
            temporal_cue: None,
        }
    }
}

/// Build a SQL WHERE snippet from the given filters.
///
/// Returns an empty string when no filters are active, or a string beginning
/// with ` AND ` that can be appended directly after `WHERE 1=1`.
///
/// The supersession subquery is verbatim from `db.rs` (`search_memory_llm_rerank`)
/// so both code paths stay in sync.
#[allow(dead_code)]
pub(crate) fn build_where(f: &HardFilters) -> String {
    let mut clauses: Vec<String> = Vec::new();

    // NOTE: SQL92 single-quote escaping (safe for SQLite). Switch to bind-params
    // when this function gains a params-return signature (deferred to PR-B wiring).
    if let Some(s) = f.space {
        clauses.push(format!("c.space = '{}'", s.replace('\'', "''")));
    }
    if let Some(t) = f.memory_type {
        clauses.push(format!("c.memory_type = '{}'", t.replace('\'', "''")));
    }
    if f.exclude_superseded {
        clauses.push(
            "c.pending_revision = 0 AND c.source_id NOT IN (\
                SELECT supersedes FROM memories \
                WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' \
                AND supersede_mode = 'hide' \
                GROUP BY supersedes\
            )"
            .into(),
        );
    }
    if let Some(cue) = f.temporal_cue {
        if cue.confidence == CueConfidence::High {
            clauses.push(format!(
                "(c.event_date BETWEEN {} AND {} OR c.event_date IS NULL)",
                cue.range.start, cue.range.end,
            ));
        }
        // Low-confidence cue: hard filter not applied.
        // Low-confidence cues are reserved for the soft-scoring layer added in follow-up work.
    }

    if clauses.is_empty() {
        String::new()
    } else {
        format!(" AND {}", clauses.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::tests::test_db;

    #[tokio::test]
    async fn hard_filters_apply_supersession_and_temporal_cue() {
        let (db, _dir) = test_db().await;
        let conn = db.test_primary_session().await;

        // Fixed "now" = 2026-05-27T12:00:00Z
        // "yesterday" cue = 2026-05-26 00:00:00Z .. 2026-05-26 23:59:59Z
        // event_date values are absolute timestamps chosen to fall clearly inside
        // or outside that range, independent of midnight boundary semantics.
        //
        // 2026-05-26T06:00:00Z = 1779775200  (inside)
        // 2026-05-24T06:00:00Z = 1779602400  (outside — two days before)

        conn.execute_batch(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type)
             VALUES
               -- mem_old: will be superseded (hide mode) by mem_new
               ('c_old',          '', 'memory', 'mem_old',          '', 0, 0, 'text'),
               -- mem_new: active revision, references mem_old via supersedes
               ('c_new',          '', 'memory', 'mem_new',          '', 0, 0, 'text'),
               -- mem_archived: superseded (archive mode) by mem_replacement — stays visible
               ('c_archived',     '', 'memory', 'mem_archived',     '', 0, 0, 'text'),
               -- mem_replacement: the archive-mode superseder (has event_date in range)
               ('c_replacement',  '', 'memory', 'mem_replacement',  '', 0, 0, 'text'),
               -- mem_in_range: event_date inside the cue window (2026-05-26T06:00:00Z)
               ('c_in_range',     '', 'memory', 'mem_in_range',     '', 0, 0, 'text'),
               -- mem_out_range: event_date outside the cue window (2026-05-24T06:00:00Z)
               ('c_out_range',    '', 'memory', 'mem_out_range',    '', 0, 0, 'text'),
               -- mem_null_date: event_date NULL (OR-NULL clause lets it pass)
               ('c_null_date',    '', 'memory', 'mem_null_date',    '', 0, 0, 'text');",
        )
        .await
        .expect("seed base rows");

        // Set supersedes + supersede_mode relationships
        conn.execute_batch(
            // mem_new supersedes mem_old in 'hide' mode → mem_old disappears
            "UPDATE memories SET supersedes = 'mem_old', supersede_mode = 'hide'  WHERE source_id = 'mem_new';
             -- mem_replacement supersedes mem_archived in 'archive' mode → mem_archived stays visible
             UPDATE memories SET supersedes = 'mem_archived', supersede_mode = 'archive' WHERE source_id = 'mem_replacement';",
        )
        .await
        .expect("set supersedes");

        // Set event_date values
        // Inside range: 2026-05-26T06:00:00Z = 1779775200
        // Outside range: 2026-05-24T06:00:00Z = 1779602400
        conn.execute_batch(
            "UPDATE memories SET event_date = 1779775200 WHERE source_id = 'mem_in_range';
             UPDATE memories SET event_date = 1779775200 WHERE source_id = 'mem_replacement';
             UPDATE memories SET event_date = 1779602400 WHERE source_id = 'mem_out_range';
             -- mem_null_date, mem_old, mem_new, mem_archived left with event_date NULL",
        )
        .await
        .expect("set event_dates");

        drop(conn);

        // "yesterday" at 2026-05-27T12:00:00Z → 2026-05-26 full day, High confidence
        let now: chrono::DateTime<chrono::Utc> = "2026-05-27T12:00:00Z".parse().unwrap();
        let cue = crate::temporal_query::extract_cue("yesterday", now)
            .expect("should extract yesterday cue");
        assert_eq!(
            cue.confidence,
            CueConfidence::High,
            "yesterday must be High confidence"
        );

        let filters = HardFilters {
            space: None,
            memory_type: None,
            exclude_superseded: true,
            temporal_cue: Some(cue),
        };
        let where_clause = build_where(&filters);

        let conn2 = db.test_primary_session().await;
        let sql = format!("SELECT source_id FROM memories c WHERE 1=1{where_clause}");
        let mut rows = conn2.query(&sql, ()).await.expect("query");
        let mut got: Vec<String> = Vec::new();
        while let Some(r) = rows.next().await.unwrap() {
            got.push(r.get::<String>(0).unwrap());
        }
        got.sort();

        assert!(
            got.contains(&"mem_new".into()),
            "mem_new (active) must pass"
        );
        assert!(
            got.contains(&"mem_archived".into()),
            "mem_archived (archive-mode) must pass"
        );
        assert!(
            got.contains(&"mem_in_range".into()),
            "mem_in_range must pass (event_date in range)"
        );
        assert!(
            got.contains(&"mem_null_date".into()),
            "mem_null_date must pass (NULL passes via OR)"
        );
        assert!(
            !got.contains(&"mem_old".into()),
            "mem_old must be filtered (hide-mode superseded)"
        );
        assert!(
            !got.contains(&"mem_out_range".into()),
            "mem_out_range must be filtered (outside cue range)"
        );
    }

    #[test]
    fn build_where_empty_returns_empty_string() {
        let f = HardFilters::default_open();
        assert_eq!(build_where(&f), "");
    }

    #[test]
    fn build_where_space_filter() {
        let f = HardFilters {
            space: Some("work"),
            memory_type: None,
            exclude_superseded: false,
            temporal_cue: None,
        };
        let w = build_where(&f);
        assert!(w.contains("c.space = 'work'"), "space clause present: {w}");
    }

    #[test]
    fn build_where_low_confidence_cue_not_applied() {
        use crate::temporal_query::{CueConfidence, DateRange, ExtractedCue};
        let cue = ExtractedCue {
            range: DateRange {
                start: 100,
                end: 200,
            },
            confidence: CueConfidence::Low,
        };
        let f = HardFilters {
            space: None,
            memory_type: None,
            exclude_superseded: false,
            temporal_cue: Some(cue),
        };
        let w = build_where(&f);
        assert!(
            !w.contains("event_date"),
            "Low confidence must not add temporal clause: {w}"
        );
    }

    #[test]
    fn build_where_single_quote_escaping() {
        let f = HardFilters {
            space: Some("o'malley"),
            memory_type: Some("user's"),
            exclude_superseded: false,
            temporal_cue: None,
        };
        let w = build_where(&f);
        assert!(w.contains("o''malley"), "space single-quote escaped: {w}");
        assert!(
            w.contains("user''s"),
            "memory_type single-quote escaped: {w}"
        );
    }
}
