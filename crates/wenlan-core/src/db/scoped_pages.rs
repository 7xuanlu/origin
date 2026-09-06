// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::pages::Page;
use crate::read_scope::ReadScope;
use crate::WenlanError;
use std::collections::{HashMap, HashSet};

fn numbered_conditions(conditions: &[String], start: usize) -> String {
    let mut parameter = start;
    conditions
        .iter()
        .map(|condition| {
            let mut numbered = String::new();
            for character in condition.chars() {
                if character == '?' {
                    numbered.push_str(&format!("?{parameter}"));
                    parameter += 1;
                } else {
                    numbered.push(character);
                }
            }
            numbered
        })
        .collect::<Vec<_>>()
        .join(" AND ")
}

/// The one page-scoping predicate. `pub(super)` so the bulk graph read in
/// `scoped_entities` scopes its page half by the same rule rather than
/// spelling a second copy of it.
pub(super) fn page_scope_clause(
    scope: &ReadScope,
    column: &str,
    parameter: usize,
) -> (String, Option<libsql::Value>) {
    match scope {
        ReadScope::Global => (String::new(), None),
        ReadScope::Space(space) => (
            format!(" AND {column} = ?{parameter}"),
            Some(libsql::Value::Text(space.clone())),
        ),
        ReadScope::Uncategorized => (
            format!(
                " AND ({column} IS NULL OR {column} = '{}')",
                super::UNFILED_SPACE_ID
            ),
            None,
        ),
    }
}

fn page_not_found() -> WenlanError {
    WenlanError::NotFound("page not found".to_string())
}

fn page_link_label_key(label: &str) -> String {
    label.to_lowercase()
}

/// One unresolved wikilink, still attached to the page whose body it came from.
///
/// This is the row `list_orphan_link_labels_scoped` throws away:
/// `COUNT(DISTINCT pl.source_page_id)` folds the source page out inside SQL,
/// and the source page is the only thing the exposure contract has a verdict
/// about -- an orphan label is `[[wikilink]]` text lifted out of ITS body, not
/// a name for the page it fails to resolve to. A reader that must filter by
/// source page therefore needs the rows, not the aggregate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrphanLinkRow {
    pub label_key: String,
    pub label: String,
    pub source_page_id: String,
}

impl MemoryDB {
    /// Route-facing Page search. Selected scopes use a workspace predicate in
    /// the same query that ranks and limits candidates; Global preserves the
    /// existing indexed search path.
    ///
    /// Retrieval arm: `kind='entity'` dual-write shadow pages stay fenced
    /// out. Used by `search_memory_cross_rerank_cued`'s direct page channel,
    /// which bypasses `select_visible_pages`. For the browse-facing twin
    /// that shows stub pages (Q1), see `search_pages_scoped_browse`.
    pub async fn search_pages_scoped(
        &self,
        query: &str,
        limit: usize,
        page_type: Option<&str>,
        scope: &ReadScope,
    ) -> Result<Vec<Page>, WenlanError> {
        self.search_pages_scoped_inner(query, limit, page_type, scope, true)
            .await
    }

    /// Browse-facing twin of `search_pages_scoped`: backs `/api/pages/search`
    /// (+ MCP `search_pages` via the same HTTP route), where Q1 rules
    /// `kind='entity'` dual-write shadow pages must appear. Never call this
    /// from a retrieval/context channel -- those must stay fenced
    /// (`search_pages_scoped`).
    pub async fn search_pages_scoped_browse(
        &self,
        query: &str,
        limit: usize,
        page_type: Option<&str>,
        scope: &ReadScope,
    ) -> Result<Vec<Page>, WenlanError> {
        self.search_pages_scoped_inner(query, limit, page_type, scope, false)
            .await
    }

    async fn search_pages_scoped_inner(
        &self,
        query: &str,
        limit: usize,
        page_type: Option<&str>,
        scope: &ReadScope,
        fence_entity: bool,
    ) -> Result<Vec<Page>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return if fence_entity {
                self.search_pages(query, limit, page_type).await
            } else {
                self.search_pages_browse(query, limit, page_type).await
            };
        }

        let embedding = self.get_or_compute_embedding(query)?;
        let vec_str = Self::vec_to_sql(&embedding);
        let fetch_limit = (limit * 3) as i64;
        let conn = self.conn.lock().await;

        // #708: `entity_id` filled in from `entity_page_map` for the
        // `kind='entity'` shadow rows, whose own column is NULL.
        let entity_id_column = super::page_entity_id_column("c.");
        let select = format!(
            "c.id, c.title, c.summary, c.content, {entity_id_column}, c.space, \
             c.source_memory_ids, c.version, c.status, c.created_at, \
             c.last_compiled, c.last_modified, \
             COALESCE(c.sources_updated_count, 0), c.stale_reason, \
             COALESCE(c.user_edited, 0), COALESCE(c.changelog, '[]'), \
             COALESCE(c.creation_kind, 'distilled'), \
             COALESCE(c.review_status, 'confirmed'), c.workspace, c.citations, COALESCE(c.kind, 'concept')"
        );
        let mut conditions = vec!["c.status = 'active'".to_string()];
        // Fence (M3 PR-1 stage f; Q1 lift, stage c; #708): the retrieval arm
        // (`fence_entity=true`) excludes every `kind='entity'` shadow page --
        // this channel is used by `search_memory_cross_rerank_cued`'s direct
        // page channel, which bypasses `select_visible_pages`. The browse arm
        // (`search_pages_scoped_browse`) keeps established, active entity
        // rows and drops detected and archived ones, matching the Wiki list.
        // In SQL rather than a post-fusion Rust filter so the fence lands
        // before `LIMIT`.
        conditions.push(if fence_entity {
            "COALESCE(c.kind, 'concept') != 'entity'".to_string()
        } else {
            super::entity_row_browse_predicate("c.")
        });
        let mut values = Vec::new();
        super::push_read_scope_filter_folded(scope, "c.workspace", &mut conditions, &mut values);
        if let Some(page_type) = page_type {
            conditions.push("c.space = ?".to_string());
            values.push(libsql::Value::Text(page_type.to_string()));
        }
        let where_clause = numbered_conditions(&conditions, 3);

        let vector_sql = format!(
            "SELECT {select}, vector_distance_cos(c.embedding, vector32(?1)) AS dist \
             FROM pages c \
             WHERE c.embedding IS NOT NULL AND {where_clause} \
             ORDER BY dist ASC LIMIT ?2"
        );
        let mut vector_params = vec![
            libsql::Value::Text(vec_str),
            libsql::Value::Integer(fetch_limit),
        ];
        vector_params.extend(values.clone());
        let mut vector_results = Vec::new();
        let mut channel_errors = Vec::new();
        let mut vector_succeeded = false;
        match conn.query(&vector_sql, vector_params).await {
            Ok(mut rows) => {
                vector_succeeded = true;
                while let Some(row) = rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("search_pages_scoped vector row: {error}"))
                })? {
                    let page = Self::row_to_page(&row).map_err(|error| {
                        WenlanError::VectorDb(format!("search_pages_scoped vector decode: {error}"))
                    })?;
                    let distance = row.get::<f64>(21).map_err(|error| {
                        WenlanError::VectorDb(format!(
                            "search_pages_scoped vector distance: {error}"
                        ))
                    })?;
                    vector_results.push((page.id.clone(), distance, page));
                }
            }
            Err(error) => {
                log::warn!("[search_pages_scoped] vector search failed: {error}");
                channel_errors.push(format!("vector: {error}"));
            }
        }

        use crate::retrieval::fts_query::{
            fts_length_exceeded, fts_recall_hardening_enabled, sanitize_fts_query,
        };
        let fts_queries = if fts_recall_hardening_enabled() {
            if fts_length_exceeded(query, 128) {
                Vec::new()
            } else {
                vec![sanitize_fts_query(query)]
            }
        } else {
            vec![query.to_string(), Self::fts_or_query(query)]
        };
        let fts_sql = format!(
            "SELECT {select} FROM pages c \
             JOIN pages_fts f ON c.rowid = f.rowid \
             WHERE pages_fts MATCH ?1 AND {where_clause} \
             ORDER BY rank LIMIT ?2"
        );
        let mut fts_results = Vec::new();
        let mut fts_succeeded = false;
        for fts_query in fts_queries {
            let mut params = vec![
                libsql::Value::Text(fts_query),
                libsql::Value::Integer(fetch_limit),
            ];
            params.extend(values.clone());
            match conn.query(&fts_sql, params).await {
                Ok(mut rows) => {
                    fts_succeeded = true;
                    while let Some(row) = rows.next().await.map_err(|error| {
                        WenlanError::VectorDb(format!("search_pages_scoped FTS row: {error}"))
                    })? {
                        let page = Self::row_to_page(&row).map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "search_pages_scoped FTS decode: {error}"
                            ))
                        })?;
                        fts_results.push((page.id.clone(), page));
                    }
                }
                Err(error) => {
                    log::debug!("[search_pages_scoped] FTS search failed: {error}");
                    channel_errors.push(format!("fts: {error}"));
                }
            }
            if !fts_results.is_empty() {
                break;
            }
        }
        if !vector_succeeded && !fts_succeeded {
            return Err(WenlanError::VectorDb(format!(
                "search_pages_scoped has no available search channel: {}",
                channel_errors.join("; ")
            )));
        }
        drop(conn);

        let rrf_k = 60.0f32;
        let fts_weight = 0.2f32;
        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut pages: HashMap<String, Page> = HashMap::new();
        for (rank, (id, distance, page)) in vector_results.into_iter().enumerate() {
            let similarity = (1.0 - distance as f32).max(0.01);
            *scores.entry(id.clone()).or_insert(0.0) += similarity / (rrf_k + rank as f32);
            pages.entry(id).or_insert(page);
        }
        for (rank, (id, page)) in fts_results.into_iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += fts_weight / (rrf_k + rank as f32);
            pages.entry(id).or_insert(page);
        }

        let theoretical_max = (1.0 + fts_weight) / rrf_k;
        for score in scores.values_mut() {
            *score = (*score / theoretical_max).min(1.0);
        }
        // The entity fence is a SQL condition on both channels above (see
        // `conditions`), so nothing is filtered out here.
        let mut results: Vec<Page> = pages.into_values().collect();
        results.sort_by(|left, right| {
            scores
                .get(&right.id)
                .unwrap_or(&0.0)
                .partial_cmp(scores.get(&left.id).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        results.truncate(limit);
        for page in &mut results {
            page.relevance_score = *scores.get(&page.id).unwrap_or(&0.0);
        }
        Ok(results)
    }

    pub async fn list_recent_pages_with_badges_scoped(
        &self,
        limit: i64,
        since_ms: Option<i64>,
        scope: &ReadScope,
    ) -> Result<Vec<wenlan_types::RecentActivityItem>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.list_recent_pages_with_badges(limit, since_ms).await;
        }

        use wenlan_types::{ActivityBadge, ActivityKind, RecentActivityItem};

        // Q1 (stage c): `kind='entity'` dual-write shadow pages appear in this
        // recent-activity listing -- a browse surface, so it reads the unfenced
        // `list_pages_scoped_browse` twin.
        let pages: Vec<Page> = self
            .list_pages_scoped_browse("active", limit, 0, scope)
            .await?;
        let all_source_ids = pages
            .iter()
            .flat_map(|page| page.source_memory_ids.iter().cloned())
            .collect::<Vec<_>>();
        let visible_source_ids = self
            .get_memories_by_source_ids_scoped(&all_source_ids, scope)
            .await?
            .into_iter()
            .map(|memory| memory.source_id)
            .collect::<HashSet<_>>();
        let flagged = self
            .pending_review_memory_ids(&visible_source_ids.iter().cloned().collect::<Vec<String>>())
            .await?;
        let since_rfc = since_ms.and_then(|ms| {
            chrono::DateTime::from_timestamp(ms / 1000, 0).map(|dt| dt.to_rfc3339())
        });
        let since_s = since_ms.map(|ms| ms / 1000);

        let mut items = Vec::with_capacity(pages.len());
        for page in pages {
            let visible_members = page
                .source_memory_ids
                .iter()
                .filter(|id| visible_source_ids.contains(id.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            let needs_review = visible_members.iter().any(|id| flagged.contains(id));
            let badge = if needs_review {
                ActivityBadge::NeedsReview
            } else if let Some(ref since) = since_rfc {
                if page.created_at >= *since {
                    ActivityBadge::New
                } else {
                    let growing_count = if visible_members.is_empty() {
                        0
                    } else {
                        let placeholders = (2..visible_members.len() + 2)
                            .map(|index| format!("?{index}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let scope_parameter = visible_members.len() + 2;
                        let (scope_sql, scope_value) =
                            page_scope_clause(scope, "space", scope_parameter);
                        let sql = format!(
                            "SELECT COUNT(*) FROM memories \
                             WHERE source != 'episode' AND source_id IN ({placeholders}) \
                               AND created_at >= ?1{scope_sql}"
                        );
                        let mut params = vec![libsql::Value::Integer(since_s.unwrap_or(0))];
                        params.extend(visible_members.iter().cloned().map(libsql::Value::Text));
                        if let Some(value) = scope_value {
                            params.push(value);
                        }
                        let conn = self.conn.lock().await;
                        let mut rows = conn.query(&sql, params).await.map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "list_recent_pages_with_badges_scoped growth count: {error}"
                            ))
                        })?;
                        rows.next()
                            .await
                            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
                            .map(|row| row.get::<i64>(0).unwrap_or(0))
                            .unwrap_or(0)
                    };
                    if growing_count > 0 {
                        ActivityBadge::Growing {
                            added: growing_count as u32,
                        }
                    } else if page.version > 1 && page.last_modified >= *since {
                        ActivityBadge::Revised
                    } else {
                        ActivityBadge::None
                    }
                }
            } else {
                ActivityBadge::None
            };
            let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&page.last_modified)
                .map(|dt| dt.timestamp_millis() as u64)
                .unwrap_or(0);
            items.push(RecentActivityItem {
                kind: ActivityKind::Page,
                id: page.id,
                title: page.title,
                snippet: page.summary.filter(|summary| !summary.is_empty()),
                timestamp_ms,
                badge,
            });
        }
        Ok(items)
    }

    pub async fn list_recent_changes_scoped(
        &self,
        limit: i64,
        scope: &ReadScope,
    ) -> Result<Vec<wenlan_types::PageChange>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.list_recent_changes(limit).await;
        }
        let (scope_sql, scope_value) = page_scope_clause(scope, "c.workspace", 2);
        let sql = format!(
            "SELECT c.id, c.title, c.version, c.created_at, c.last_modified \
             FROM pages c WHERE c.status = 'active'{scope_sql} \
             ORDER BY c.last_modified DESC LIMIT ?1"
        );
        let mut params = vec![libsql::Value::Integer(limit)];
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("list_recent_changes_scoped: {error}"))
        })?;
        let mut changes = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            let version = row.get::<i64>(2).unwrap_or(1);
            let created_at = row.get::<String>(3).unwrap_or_default();
            let last_modified = row.get::<String>(4).unwrap_or_default();
            let change_kind = if version > 1 || created_at != last_modified {
                wenlan_types::PageChangeKind::Revised
            } else {
                wenlan_types::PageChangeKind::Created
            };
            changes.push(wenlan_types::PageChange {
                page_id: row.get(0).unwrap_or_default(),
                title: row.get(1).unwrap_or_default(),
                change_kind,
                changed_at_ms: chrono::DateTime::parse_from_rfc3339(&last_modified)
                    .map(|dt| dt.timestamp_millis())
                    .unwrap_or(0),
            });
        }
        Ok(changes)
    }

    /// Fenced: `kind='entity'` dual-write shadow pages stay excluded on every
    /// scope arm (Global delegates to the fenced `list_pages`). Export and
    /// other internal consumers that must never emit a stub call this, so the
    /// fence lands in SQL before `LIMIT` -- a burst of fresh stubs can never
    /// crowd real pages out of a bounded window. For the browse-facing twin
    /// (Q1 read surfaces), see `list_pages_scoped_browse`.
    pub async fn list_pages_scoped(
        &self,
        status: &str,
        limit: i64,
        offset: i64,
        scope: &ReadScope,
    ) -> Result<Vec<Page>, WenlanError> {
        self.list_pages_scoped_inner(status, limit, offset, scope, true)
            .await
    }

    /// Browse-facing twin of `list_pages_scoped`: backs `GET /api/pages` and
    /// the recent-with-badges browse path, where Q1 rules `kind='entity'`
    /// dual-write shadow pages must appear on every scope arm (Global delegates
    /// to `list_pages_browse`). Never call this from export or any internal
    /// non-browse path -- those must stay fenced (`list_pages_scoped`).
    pub async fn list_pages_scoped_browse(
        &self,
        status: &str,
        limit: i64,
        offset: i64,
        scope: &ReadScope,
    ) -> Result<Vec<Page>, WenlanError> {
        self.list_pages_scoped_inner(status, limit, offset, scope, false)
            .await
    }

    async fn list_pages_scoped_inner(
        &self,
        status: &str,
        limit: i64,
        offset: i64,
        scope: &ReadScope,
        fence_entity: bool,
    ) -> Result<Vec<Page>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return if fence_entity {
                self.list_pages(status, limit, offset).await
            } else {
                self.list_pages_browse(status, limit, offset).await
            };
        }
        // #708: `entity_id` is filled in from `entity_page_map` for the
        // `kind='entity'` shadow rows, whose own column is NULL.
        let entity_id_column = super::page_entity_id_column("c.");
        let select = format!(
            "c.id, c.title, c.summary, c.content, {entity_id_column}, c.space, \
             c.source_memory_ids, c.version, c.status, c.created_at, \
             c.last_compiled, c.last_modified, \
             COALESCE(c.sources_updated_count, 0), c.stale_reason, \
             COALESCE(c.user_edited, 0), COALESCE(c.changelog, '[]'), \
             COALESCE(c.creation_kind, 'distilled'), \
             COALESCE(c.review_status, 'confirmed'), c.workspace, c.citations, COALESCE(c.kind, 'concept')"
        );
        // #708: the browse twin keeps established, active entity rows and
        // drops detected and archived ones; the fenced twin drops every
        // entity row as before. Both land in SQL before `LIMIT`, so a burst
        // of fresh detected entities can never crowd real pages out of a
        // bounded window.
        let fence_sql = if fence_entity {
            " AND COALESCE(c.kind, 'concept') != 'entity'".to_string()
        } else {
            super::entity_row_browse_fence("c.")
        };
        let (sql, params) = match scope {
            ReadScope::Space(workspace) => (
                format!(
                    "SELECT {select} FROM pages c \
                     WHERE c.status = ?1{fence_sql} AND c.workspace = ?2 \
                     ORDER BY c.last_modified DESC LIMIT ?3 OFFSET ?4"
                ),
                vec![
                    libsql::Value::Text(status.to_string()),
                    libsql::Value::Text(workspace.clone()),
                    libsql::Value::Integer(limit),
                    libsql::Value::Integer(offset),
                ],
            ),
            ReadScope::Uncategorized => (
                format!(
                    "SELECT {select} FROM pages c \
                     WHERE c.status = ?1{fence_sql} AND c.workspace = '00000000-0000-4000-8000-000000000001' \
                     ORDER BY c.last_modified DESC LIMIT ?2 OFFSET ?3"
                ),
                vec![
                    libsql::Value::Text(status.to_string()),
                    libsql::Value::Integer(limit),
                    libsql::Value::Integer(offset),
                ],
            ),
            ReadScope::Global => unreachable!(),
        };
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(&sql, params)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("list_pages_scoped: {error}")))?;
        let mut pages = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            pages.push(Self::row_to_page(&row)?);
        }
        Ok(pages)
    }

    /// Every orphan link as a row, source page intact, for readers that have to
    /// filter before they fold.
    ///
    /// Same predicate as `list_orphan_link_labels_scoped` -- unresolved target,
    /// active source page, the scope clause -- with the aggregation removed.
    /// Splitting it that way is the whole point: the aggregate applies
    /// `HAVING n >= ?` and `LIMIT 100` inside SQL, so a caller that filters the
    /// aggregate's output reports counts and a truncation point computed over
    /// pages it may not see. Pair this with [`MemoryDB::fold_orphan_labels`],
    /// which reproduces the aggregate over whatever survives the filter.
    ///
    /// Global is handled here rather than delegating to a separate twin: with
    /// no `min_count` parameter the scope clause is the only difference between
    /// the arms, and `page_scope_clause` already answers empty for Global.
    ///
    /// ponytail: the flat `LIMIT 20000` is a ceiling, not a page. Orphan links
    /// are a per-page-body artifact, and a graph that produces more than 20k of
    /// them has a bigger problem than this feed. If one ever does, the upgrade
    /// is a keyset walk over `label_key` feeding the same fold, not a larger
    /// number here.
    pub async fn list_orphan_link_rows_scoped(
        &self,
        scope: &ReadScope,
    ) -> Result<Vec<OrphanLinkRow>, WenlanError> {
        let (scope_sql, scope_value) = page_scope_clause(scope, "p.workspace", 1);
        let sql = format!(
            "SELECT pl.label_key, pl.label, pl.source_page_id \
             FROM page_links pl INNER JOIN pages p ON p.id = pl.source_page_id \
             WHERE pl.target_page_id IS NULL AND p.status = 'active'{scope_sql} \
             LIMIT 20000"
        );
        let mut params = Vec::new();
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("list_orphan_link_rows_scoped: {error}"))
        })?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            out.push(OrphanLinkRow {
                label_key: row.get(0).unwrap_or_default(),
                label: row.get(1).unwrap_or_default(),
                source_page_id: row.get(2).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// The aggregate twin's `GROUP BY label_key HAVING n >= ? ORDER BY n DESC,
    /// display_label ASC LIMIT 100`, in Rust, over rows a caller has already
    /// filtered.
    ///
    /// It has to be a re-implementation rather than a second query, because the
    /// point is to count and cap over the pages *this* caller may see and SQL
    /// cannot know that. The semantics are copied deliberately: display label is
    /// the lexicographic minimum in the group (`MIN(pl.label)`, BINARY collation,
    /// which is `String::cmp`), and `n` counts DISTINCT source pages.
    /// `page_links` is keyed `(source_page_id, label_key)`, so a duplicate pair
    /// cannot exist today -- the DISTINCT logic is written out anyway rather
    /// than leaning on a primary key this function has no way to check.
    ///
    /// Pure and DB-free; it hangs off `MemoryDB` only because that is what makes
    /// it nameable outside `db`'s private module tree.
    pub fn fold_orphan_labels(rows: Vec<OrphanLinkRow>, min_count: i64) -> Vec<(String, i64)> {
        let mut groups: HashMap<String, (String, HashSet<String>)> = HashMap::new();
        for row in rows {
            let group = groups
                .entry(row.label_key)
                .or_insert_with(|| (row.label.clone(), HashSet::new()));
            if row.label < group.0 {
                group.0 = row.label;
            }
            group.1.insert(row.source_page_id);
        }
        let mut folded: Vec<(String, i64)> = groups
            .into_values()
            .map(|(label, sources)| (label, sources.len() as i64))
            .filter(|(_, n)| *n >= min_count)
            .collect();
        folded.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        folded.truncate(100);
        folded
    }

    pub async fn list_orphan_link_labels_scoped(
        &self,
        min_count: i64,
        scope: &ReadScope,
    ) -> Result<Vec<(String, i64)>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.list_orphan_link_labels(min_count).await;
        }
        let (scope_sql, scope_value) = page_scope_clause(scope, "p.workspace", 2);
        let sql = format!(
            "SELECT MIN(pl.label) AS display_label, \
                    COUNT(DISTINCT pl.source_page_id) AS n \
             FROM page_links pl INNER JOIN pages p ON p.id = pl.source_page_id \
             WHERE pl.target_page_id IS NULL AND p.status = 'active'{scope_sql} \
             GROUP BY pl.label_key HAVING n >= ?1 \
             ORDER BY n DESC, display_label ASC LIMIT 100"
        );
        let mut params = vec![libsql::Value::Integer(min_count)];
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("list_orphan_link_labels_scoped: {error}"))
        })?;
        let mut labels = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            labels.push((row.get(0).unwrap_or_default(), row.get(1).unwrap_or(0)));
        }
        Ok(labels)
    }

    /// Fenced: `kind='entity'` dual-write shadow pages stay excluded, so a
    /// by-id read on a mutation/export path resolves a stub as absent (None).
    /// For the browse-facing twin that returns stub pages by id (Q1 read
    /// surfaces), see `get_page_scoped_browse`.
    pub async fn get_page_scoped(
        &self,
        id: &str,
        scope: &ReadScope,
    ) -> Result<Option<Page>, WenlanError> {
        self.get_page_scoped_inner(id, scope, true).await
    }

    /// Browse-facing twin of `get_page_scoped`: backs `GET /api/pages/{id}` and
    /// its advisory revisions read, where Q1 rules `kind='entity'` dual-write
    /// shadow pages must resolve. Never call this from a mutation/export path --
    /// those must stay fenced (`get_page_scoped`).
    pub async fn get_page_scoped_browse(
        &self,
        id: &str,
        scope: &ReadScope,
    ) -> Result<Option<Page>, WenlanError> {
        self.get_page_scoped_inner(id, scope, false).await
    }

    async fn get_page_scoped_inner(
        &self,
        id: &str,
        scope: &ReadScope,
        fence_entity: bool,
    ) -> Result<Option<Page>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return if fence_entity {
                self.get_page(id).await
            } else {
                self.get_page_browse(id).await
            };
        }
        let select = "c.id, c.title, c.summary, c.content, c.entity_id, c.space, \
                      c.source_memory_ids, c.version, c.status, c.created_at, \
                      c.last_compiled, c.last_modified, \
                      COALESCE(c.sources_updated_count, 0), c.stale_reason, \
                      COALESCE(c.user_edited, 0), COALESCE(c.changelog, '[]'), \
                      COALESCE(c.creation_kind, 'distilled'), \
                      COALESCE(c.review_status, 'confirmed'), c.workspace, c.citations, COALESCE(c.kind, 'concept')";
        let (scope_sql, scope_value) = page_scope_clause(scope, "c.workspace", 2);
        let fence_sql = if fence_entity {
            " AND COALESCE(c.kind, 'concept') != 'entity'"
        } else {
            ""
        };
        let sql = format!("SELECT {select} FROM pages c WHERE c.id = ?1{fence_sql}{scope_sql}");
        let mut params = vec![libsql::Value::Text(id.to_string())];
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(&sql, params)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("get_page_scoped: {error}")))?;
        match rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            Some(row) => Ok(Some(Self::row_to_page(&row)?)),
            None => Ok(None),
        }
    }

    /// G6 Stage 1.3: the joined side reads `edges` (`cites`) instead of
    /// `page_sources`, spanning all `dst_kind`s (no `dst_kind` filter) -- see
    /// the unscoped `get_page_sources` doc comment for the ruling. `linked_at`
    /// is COALESCE-total over the payload-mirrored legacy value and
    /// `edges.created_at` (ordering trap, same discipline as the unscoped
    /// reader).
    pub async fn get_page_sources_scoped(
        &self,
        page_id: &str,
        scope: &ReadScope,
    ) -> Result<Vec<wenlan_types::PageSource>, WenlanError> {
        let (scope_sql, scope_value) = page_scope_clause(scope, "c.workspace", 2);
        let sql = format!(
            "SELECT c.id, ps.src_id, ps.dst_id, \
                    COALESCE(json_extract(ps.payload,'$.linked_at'), ps.created_at), \
                    json_extract(ps.payload,'$.link_reason') \
             FROM pages c LEFT JOIN edges ps ON ps.src_id = c.id AND ps.edge_type = 'cites' \
                    AND ps.valid_until IS NULL \
             WHERE c.id = ?1{scope_sql} \
             ORDER BY COALESCE(json_extract(ps.payload,'$.linked_at'), ps.created_at) ASC, ps.dst_id ASC"
        );
        let mut params = vec![libsql::Value::Text(page_id.to_string())];
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(&sql, params)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("get_page_sources_scoped: {error}")))?;
        let mut found = false;
        let mut sources = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            found = true;
            let Some(source_page_id) = row.get::<Option<String>>(1).unwrap_or(None) else {
                continue;
            };
            sources.push(wenlan_types::PageSource {
                page_id: source_page_id,
                memory_source_id: row.get(2).unwrap_or_default(),
                linked_at: row.get(3).unwrap_or(0),
                link_reason: row.get::<Option<String>>(4).unwrap_or(None),
            });
        }
        if !found {
            return Err(page_not_found());
        }
        Ok(sources)
    }

    pub async fn get_page_outbound_links_scoped(
        &self,
        source_page_id: &str,
        scope: &ReadScope,
    ) -> Result<Vec<crate::synthesis::wikilinks::Wikilink>, WenlanError> {
        let (scope_sql, scope_value) = page_scope_clause(scope, "c.workspace", 2);
        let conn = self.conn.lock().await;
        let existence_sql = format!("SELECT c.id FROM pages c WHERE c.id = ?1{scope_sql}");
        let mut existence_params = vec![libsql::Value::Text(source_page_id.to_string())];
        if let Some(value) = scope_value.clone() {
            existence_params.push(value);
        }
        let mut rows = conn
            .query(&existence_sql, existence_params)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_page_outbound_links_scoped: {error}"))
            })?;
        let found = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
            .is_some();
        drop(rows);
        if !found {
            return Err(page_not_found());
        }

        let edge_sql = format!(
            "SELECT e.dst_id, json_extract(e.payload, '$.label') \
             FROM edges e INNER JOIN pages c ON c.id = e.src_id \
             WHERE c.id = ?1 AND e.edge_type = 'links' \
               AND e.src_kind = 'page' AND e.dst_kind = 'page' \
               AND e.valid_until IS NULL{scope_sql}"
        );
        let mut edge_params = vec![libsql::Value::Text(source_page_id.to_string())];
        if let Some(value) = scope_value {
            edge_params.push(value);
        }
        let mut rows = conn.query(&edge_sql, edge_params).await.map_err(|error| {
            WenlanError::VectorDb(format!("get_page_outbound_links_scoped: {error}"))
        })?;
        let mut links = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            let label = row
                .get::<Option<String>>(1)
                .unwrap_or(None)
                .unwrap_or_default();
            links.push((
                page_link_label_key(&label),
                crate::synthesis::wikilinks::Wikilink {
                    target_page_id: row.get::<String>(0).ok(),
                    label,
                },
            ));
        }
        drop(rows);

        let mut rows = conn
            .query(
                "SELECT target_page_id, label FROM page_links \
                 WHERE source_page_id = ?1 AND target_page_id IS NULL",
                libsql::params![source_page_id],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_page_outbound_links_scoped: {error}"))
            })?;
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            let Some(label) = row.get::<Option<String>>(1).unwrap_or(None) else {
                continue;
            };
            links.push((
                page_link_label_key(&label),
                crate::synthesis::wikilinks::Wikilink {
                    target_page_id: row.get::<Option<String>>(0).unwrap_or(None),
                    label,
                },
            ));
        }
        links.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(links.into_iter().map(|(_, link)| link).collect())
    }

    pub async fn get_page_inbound_links_scoped(
        &self,
        target_page_id: &str,
        scope: &ReadScope,
    ) -> Result<Vec<(String, String)>, WenlanError> {
        let (source_scope_sql, source_scope_value) =
            page_scope_clause(scope, "source.workspace", 2);
        let (target_scope_sql, target_scope_value) =
            page_scope_clause(scope, "target.workspace", 3);
        let sql = format!(
            "SELECT target.id, source.id, json_extract(e.payload, '$.label') \
             FROM pages target \
             LEFT JOIN edges e ON e.dst_id = target.id \
               AND e.edge_type = 'links' AND e.src_kind = 'page' \
               AND e.dst_kind = 'page' AND e.valid_until IS NULL \
             LEFT JOIN pages source ON source.id = e.src_id \
               AND source.status = 'active'{source_scope_sql} \
             WHERE target.id = ?1{target_scope_sql} \
             ORDER BY source.last_modified DESC, source.id ASC"
        );
        let mut params = vec![libsql::Value::Text(target_page_id.to_string())];
        if let Some(value) = source_scope_value {
            params.push(value);
        }
        if let Some(value) = target_scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("get_page_inbound_links_scoped: {error}"))
        })?;
        let mut found = false;
        let mut links = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            found = true;
            let Some(source_id) = row.get::<Option<String>>(1).unwrap_or(None) else {
                continue;
            };
            links.push((source_id, row.get(2).unwrap_or_default()));
        }
        if !found {
            return Err(page_not_found());
        }
        Ok(links)
    }

    pub async fn get_page_changelog_scoped(
        &self,
        page_id: &str,
        scope: &ReadScope,
    ) -> Result<String, WenlanError> {
        let (scope_sql, scope_value) = page_scope_clause(scope, "c.workspace", 2);
        let sql =
            format!("SELECT COALESCE(c.changelog, '[]') FROM pages c WHERE c.id = ?1{scope_sql}");
        let mut params = vec![libsql::Value::Text(page_id.to_string())];
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("get_page_changelog_scoped: {error}"))
        })?;
        match rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
        {
            Some(row) => Ok(row.get(0).unwrap_or_else(|_| "[]".to_string())),
            None => Err(page_not_found()),
        }
    }

    pub async fn select_visible_pages_scoped(
        &self,
        raw_pages: Vec<Page>,
        scope: &ReadScope,
        memory_source_ids: &HashSet<String>,
        caller_trust: &str,
        cap: usize,
    ) -> Vec<Page> {
        let scoped = raw_pages
            .into_iter()
            .filter(|page| scope.matches(page.workspace.as_deref()))
            .collect();
        self.select_visible_pages(scoped, None, memory_source_ids, caller_trust, cap)
            .await
    }
}
