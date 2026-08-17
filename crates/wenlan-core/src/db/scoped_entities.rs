// SPDX-License-Identifier: Apache-2.0

use super::{
    Entity, EntityDetail, EntitySearchResult, MemoryDB, Observation, RefinementProposal,
    RelationWithEntity, SearchResult,
};
use crate::read_scope::ReadScope;
use crate::WenlanError;
use std::collections::{HashMap, HashSet};

impl MemoryDB {
    pub async fn list_entities_scoped(
        &self,
        entity_type: Option<&str>,
        scope: &ReadScope,
    ) -> Result<Vec<Entity>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.list_entities(entity_type, None).await;
        }

        let conn = self.conn.lock().await;
        // G6 Stage 1.5b Part 3: unconditional hard cutover onto the
        // `kind='entity'` shadow page (same program contract as 1.5a; spec
        // item 9 collapses this reader's `reader_uses_entity_pages` gate --
        // Part 1's scalar mirror extension and Part 2's space fold made
        // every remaining field safe to trust off the mirror, so this
        // never touches `entities` directly).
        let mut conditions = Vec::new();
        let mut values = Vec::new();
        if let Some(entity_type) = entity_type {
            conditions.push("p.entity_type = ?".to_string());
            values.push(libsql::Value::Text(entity_type.to_string()));
        }
        // G6 Stage 1.5b: `entities.space` is folded (never NULL; unfiled rows
        // carry `UNFILED_SPACE_ID`), so `Uncategorized` must match either.
        super::push_read_scope_filter_folded(scope, "p.space", &mut conditions, &mut values);
        let sql = format!(
            "SELECT epm.entity_id, p.title, p.entity_type, p.space, p.source_agent, \
                    p.confidence, p.entity_confirmed, p.entity_created_at, p.entity_updated_at \
             FROM entity_page_map epm \
             JOIN pages p ON p.id = epm.page_id \
             WHERE p.kind = 'entity' AND p.status = 'active' AND {} \
             ORDER BY p.entity_updated_at DESC, epm.entity_id ASC",
            conditions.join(" AND ")
        );
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(values))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("list_entities_scoped query: {error}"))
            })?;
        let mut entities = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("list_entities_scoped next: {error}")))?
        {
            entities.push(entity_from_row(&row, "list_entities_scoped")?);
        }
        Ok(entities)
    }

    pub async fn get_entity_detail_scoped(
        &self,
        entity_id: &str,
        scope: &ReadScope,
    ) -> Result<EntityDetail, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.get_entity_detail(entity_id).await;
        }

        let conn = self.conn.lock().await;
        // G6 Stage 1.5b Part 3: unconditional hard cutover onto the
        // `kind='entity'` shadow page, for the primary entity row and (G6
        // Stage 2 PR 2c sub-step 3 item 5) the relations query's endpoint
        // hydration below -- `store_entity` no longer writes `entities`, so
        // the old `entities` JOIN silently excluded any shadow-only
        // endpoint. Ported onto `entity_page_map`/`pages`, same shape as
        // `list_recent_relations_scoped` below. The observations query
        // never touched `entities`, so it's unchanged. Spec item 9
        // collapses this reader's `reader_uses_entity_pages` gate (same
        // program contract as 1.5a and `list_entities_scoped` above).
        let mut conditions = vec!["epm.entity_id = ?".to_string()];
        let mut values = vec![libsql::Value::Text(entity_id.to_string())];
        // G6 Stage 1.5b: `entities.space` is folded (never NULL; unfiled rows
        // carry `UNFILED_SPACE_ID`), so `Uncategorized` must match either.
        super::push_read_scope_filter_folded(scope, "p.space", &mut conditions, &mut values);
        let entity_sql = format!(
            "SELECT epm.entity_id, p.title, p.entity_type, p.space, p.source_agent, \
                    p.confidence, p.entity_confirmed, p.entity_created_at, p.entity_updated_at \
             FROM entity_page_map epm \
             JOIN pages p ON p.id = epm.page_id \
             WHERE p.kind = 'entity' AND p.status = 'active' AND {} LIMIT 1",
            conditions.join(" AND ")
        );
        let mut rows = conn
            .query(&entity_sql, libsql::params_from_iter(values))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_entity_detail_scoped entity query: {error}"))
            })?;
        let entity = match rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("get_entity_detail_scoped entity next: {error}"))
        })? {
            Some(row) => entity_from_row(&row, "get_entity_detail_scoped entity")?,
            None => return Err(WenlanError::NotFound("entity not found".to_string())),
        };
        drop(rows);

        let mut observation_rows = conn
            .query(
                "SELECT id, entity_id, content, source_agent, confidence, confirmed, created_at \
                 FROM observations WHERE entity_id = ?1 ORDER BY created_at DESC",
                libsql::params![entity_id],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!(
                    "get_entity_detail_scoped observations query: {error}"
                ))
            })?;
        let mut observations = Vec::new();
        while let Some(row) = observation_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!(
                "get_entity_detail_scoped observations next: {error}"
            ))
        })? {
            observations.push(Observation {
                id: row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped observation id: {error}"
                    ))
                })?,
                entity_id: row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped observation entity_id: {error}"
                    ))
                })?,
                content: row.get(2).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped observation content: {error}"
                    ))
                })?,
                source_agent: row.get::<Option<String>>(3).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped observation source_agent: {error}"
                    ))
                })?,
                confidence: row
                    .get::<Option<f64>>(4)
                    .map_err(|error| {
                        WenlanError::VectorDb(format!(
                            "get_entity_detail_scoped observation confidence: {error}"
                        ))
                    })?
                    .map(|value| value as f32),
                confirmed: row.get::<i64>(5).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped observation confirmed: {error}"
                    ))
                })? != 0,
                created_at: row.get(6).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped observation created_at: {error}"
                    ))
                })?,
            });
        }
        drop(observation_rows);

        // G6 Stage 1.5b: `entities.space` is folded (never NULL; unfiled rows
        // carry `UNFILED_SPACE_ID`), so `Uncategorized` must match either.
        let (endpoint_filter, endpoint_value) = match scope {
            ReadScope::Space(space) => {
                ("AND e.space = ?2", Some(libsql::Value::Text(space.clone())))
            }
            ReadScope::Uncategorized => (
                "AND (e.space IS NULL OR e.space = '00000000-0000-4000-8000-000000000001')",
                None,
            ),
            ReadScope::Global => unreachable!(),
        };
        let relation_sql = format!(
            "SELECT r.edge_id, r.semantic_type, json_extract(r.payload, '$.source_agent'), \
                    COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at), \
                    'outgoing' AS direction, r.dst_id AS entity_id, \
                    e.title AS entity_name, e.entity_type AS entity_type \
             FROM edges r \
             JOIN entity_page_map epm ON epm.entity_id = r.dst_id \
             JOIN pages e ON e.id = epm.page_id AND e.kind = 'entity' AND e.status = 'active' \
             WHERE r.edge_type = 'relates' AND r.valid_until IS NULL \
               AND r.semantic_type IS NOT NULL \
               AND r.src_id = ?1 {endpoint_filter} \
             UNION ALL \
             SELECT r.edge_id, r.semantic_type, json_extract(r.payload, '$.source_agent'), \
                    COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at), \
                    'incoming' AS direction, r.src_id AS entity_id, \
                    e.title AS entity_name, e.entity_type AS entity_type \
             FROM edges r \
             JOIN entity_page_map epm ON epm.entity_id = r.src_id \
             JOIN pages e ON e.id = epm.page_id AND e.kind = 'entity' AND e.status = 'active' \
             WHERE r.edge_type = 'relates' AND r.valid_until IS NULL \
               AND r.semantic_type IS NOT NULL \
               AND r.dst_id = ?1 {endpoint_filter} \
             ORDER BY 4 DESC"
        );
        let mut relation_values = vec![libsql::Value::Text(entity_id.to_string())];
        if let Some(value) = endpoint_value {
            relation_values.push(value);
        }
        let mut relation_rows = conn
            .query(&relation_sql, libsql::params_from_iter(relation_values))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_entity_detail_scoped relations query: {error}"))
            })?;
        let mut relations = Vec::new();
        while let Some(row) = relation_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("get_entity_detail_scoped relations next: {error}"))
        })? {
            relations.push(RelationWithEntity {
                id: row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!("get_entity_detail_scoped relation id: {error}"))
                })?,
                relation_type: row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation type: {error}"
                    ))
                })?,
                source_agent: row.get::<Option<String>>(2).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation source_agent: {error}"
                    ))
                })?,
                created_at: row.get(3).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation created_at: {error}"
                    ))
                })?,
                direction: row.get(4).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation direction: {error}"
                    ))
                })?,
                entity_id: row.get(5).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation entity_id: {error}"
                    ))
                })?,
                entity_name: row.get(6).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation entity_name: {error}"
                    ))
                })?,
                entity_type: row.get(7).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_entity_detail_scoped relation entity_type: {error}"
                    ))
                })?,
            });
        }

        Ok(EntityDetail {
            entity,
            observations,
            relations,
        })
    }

    pub async fn list_recent_relations_scoped(
        &self,
        limit: usize,
        since_ms: Option<i64>,
        scope: &ReadScope,
    ) -> Result<Vec<wenlan_types::RecentRelation>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.list_recent_relations(limit, since_ms).await;
        }

        let conn = self.conn.lock().await;
        // G6 Stage 1.5b Part 3: unconditional hard cutover onto the
        // `kind='entity'` shadow page (spec item 9 collapses this reader's
        // `reader_uses_entity_pages` gate -- the last of the four
        // `scoped_entities.rs` call sites). Item 7's structural rework: the
        // join itself now runs on `entity_page_map`/`pages` for both row set
        // and scope filter, replacing the former tie-safe-flip design (base
        // query always legacy `entities`, gated hydration query overlaying
        // just the display title).
        let (scope_filter, scope_value) = match scope {
            ReadScope::Space(space) => (
                "AND p1.space = ?3 AND p2.space = ?3",
                Some(libsql::Value::Text(space.clone())),
            ),
            ReadScope::Uncategorized => (
                "AND (p1.space IS NULL OR p1.space = '00000000-0000-4000-8000-000000000001') \
                 AND (p2.space IS NULL OR p2.space = '00000000-0000-4000-8000-000000000001')",
                None,
            ),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT r.edge_id, r.src_id, r.semantic_type, r.dst_id, \
                    p1.title, p2.title, \
                    COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at) AS asserted_ts \
             FROM edges r \
             JOIN entity_page_map epm1 ON r.src_id = epm1.entity_id \
             JOIN pages p1 ON p1.id = epm1.page_id AND p1.kind = 'entity' AND p1.status = 'active' \
             JOIN entity_page_map epm2 ON r.dst_id = epm2.entity_id \
             JOIN pages p2 ON p2.id = epm2.page_id AND p2.kind = 'entity' AND p2.status = 'active' \
             WHERE r.edge_type = 'relates' AND r.valid_until IS NULL \
               AND r.semantic_type IS NOT NULL \
               AND (?1 IS NULL OR COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at) >= ?1) \
               AND p1.title IS NOT NULL AND p1.title != '' \
               AND p2.title IS NOT NULL AND p2.title != '' \
               {scope_filter} \
             ORDER BY asserted_ts DESC, r.edge_id DESC LIMIT ?2"
        );
        let mut values = vec![
            since_ms
                .map(libsql::Value::Integer)
                .unwrap_or(libsql::Value::Null),
            libsql::Value::Integer(limit as i64),
        ];
        if let Some(value) = scope_value {
            values.push(value);
        }
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(values))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("list_recent_relations_scoped query: {error}"))
            })?;
        let mut relations = Vec::new();
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("list_recent_relations_scoped next: {error}"))
        })? {
            relations.push(wenlan_types::RecentRelation {
                id: row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!("list_recent_relations_scoped id: {error}"))
                })?,
                from_entity_id: row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped from_entity: {error}"
                    ))
                })?,
                relation_type: row.get(2).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped relation_type: {error}"
                    ))
                })?,
                to_entity_id: row.get(3).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped to_entity: {error}"
                    ))
                })?,
                from_entity_name: row.get(4).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped from_name: {error}"
                    ))
                })?,
                to_entity_name: row.get(5).map_err(|error| {
                    WenlanError::VectorDb(format!("list_recent_relations_scoped to_name: {error}"))
                })?,
                created_at_ms: row.get(6).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped created_at: {error}"
                    ))
                })?,
            });
        }

        Ok(relations)
    }

    pub async fn list_entity_suggestions_scoped(
        &self,
        scope: &ReadScope,
    ) -> Result<Vec<RefinementProposal>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.get_pending_refinements().await;
        }

        let conn = self.conn.lock().await;
        // M3 PR-2 stage c: NOT a cutover consumer. A suggestion proposes an
        // entity that does not exist yet, so this query sources nothing
        // from `entities`/`pages` (it reads `refinement_queue` + `memories`
        // only) -- it stays on its own substrate until the retirement rung
        // widens the mirror.
        let safe_sources = "CASE WHEN json_valid(rq.source_ids) \
                            THEN rq.source_ids ELSE '[]' END";
        let (matching_owner, mismatching_owner, values) = match scope {
            ReadScope::Space(space) => (
                "m.space = ?1",
                "(m.space IS NULL OR m.space != ?1)",
                vec![libsql::Value::Text(space.clone())],
            ),
            ReadScope::Uncategorized => (
                "(m.space IS NULL OR m.space = '00000000-0000-4000-8000-000000000001')",
                "(m.space IS NOT NULL AND m.space != '00000000-0000-4000-8000-000000000001')",
                Vec::new(),
            ),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT rq.id, rq.action, rq.source_ids, rq.payload, rq.confidence, \
                    rq.status, rq.created_at \
             FROM refinement_queue rq \
             WHERE rq.action = 'suggest_entity' \
               AND rq.status IN ('pending', 'awaiting_review') \
               AND json_valid(rq.source_ids) \
               AND json_type({safe_sources}) = 'array' \
               AND json_array_length({safe_sources}) > 0 \
               AND NOT EXISTS ( \
                   SELECT 1 FROM json_each({safe_sources}) sid \
                   WHERE sid.type != 'text' \
                      OR NOT EXISTS ( \
                          SELECT 1 FROM memories m \
                          WHERE m.source = 'memory' AND m.pending_revision = 0 \
                            AND m.source_id = sid.value AND {matching_owner} \
                      ) \
                      OR EXISTS ( \
                          SELECT 1 FROM memories m \
                          WHERE m.source = 'memory' AND m.pending_revision = 0 \
                            AND m.source_id = sid.value AND {mismatching_owner} \
                      ) \
               ) \
             ORDER BY rq.created_at, rq.id"
        );
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(values))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("list_entity_suggestions_scoped query: {error}"))
            })?;
        let mut proposals = Vec::new();
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("list_entity_suggestions_scoped next: {error}"))
        })? {
            let source_ids_json: String = row.get(2).map_err(|error| {
                WenlanError::VectorDb(format!(
                    "list_entity_suggestions_scoped source_ids: {error}"
                ))
            })?;
            proposals.push(RefinementProposal {
                id: row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!("list_entity_suggestions_scoped id: {error}"))
                })?,
                action: row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!("list_entity_suggestions_scoped action: {error}"))
                })?,
                source_ids: serde_json::from_str(&source_ids_json).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_entity_suggestions_scoped source_ids JSON: {error}"
                    ))
                })?,
                payload: row.get::<Option<String>>(3).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_entity_suggestions_scoped payload: {error}"
                    ))
                })?,
                confidence: row
                    .get::<Option<f64>>(4)
                    .map_err(|error| {
                        WenlanError::VectorDb(format!(
                            "list_entity_suggestions_scoped confidence: {error}"
                        ))
                    })?
                    .unwrap_or(0.0),
                status: row.get(5).map_err(|error| {
                    WenlanError::VectorDb(format!("list_entity_suggestions_scoped status: {error}"))
                })?,
                created_at: row.get(6).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_entity_suggestions_scoped created_at: {error}"
                    ))
                })?,
            });
        }
        Ok(proposals)
    }

    /// G6 Stage 1.5a: scope-filters via the `kind='entity'` shadow page's
    /// `space` column (through `entity_page_map`) instead of `entities.space`
    /// directly. Safe under the space-sentinel fold: `ReadScope::Uncategorized`
    /// matches `pages.space = UNFILED_SPACE_ID` (the NULL fold target), and
    /// `ReadScope::Space(name)` only ever carries a name resolved through
    /// `resolve_read_scope`/`db.get_space` against a REGISTERED space —
    /// `create_space`/`update_space` reject `UNFILED_SPACE_ID` as a
    /// user-supplied name (db.rs `UNFILED_SPACE_ID` doc comment), so the two
    /// branches can never collide.
    pub(super) async fn filter_entity_ids_scoped(
        &self,
        entity_ids: &[String],
        scope: &ReadScope,
    ) -> Result<Vec<String>, WenlanError> {
        if matches!(scope, ReadScope::Global) || entity_ids.is_empty() {
            return Ok(entity_ids.to_vec());
        }
        let placeholders = (1..=entity_ids.len())
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(",");
        let scope_index = entity_ids.len() + 1;
        let (scope_sql, scope_value) = match scope {
            ReadScope::Space(space) => (
                format!("AND p.space = ?{scope_index}"),
                Some(libsql::Value::Text(space.clone())),
            ),
            ReadScope::Uncategorized => (
                format!("AND p.space = ?{scope_index}"),
                Some(libsql::Value::Text(crate::db::UNFILED_SPACE_ID.to_string())),
            ),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT epm.entity_id FROM entity_page_map epm
             JOIN pages p ON p.id = epm.page_id
             WHERE epm.entity_id IN ({placeholders}) AND p.kind = 'entity' AND p.status = 'active' {scope_sql}"
        );
        let mut params: Vec<libsql::Value> = entity_ids
            .iter()
            .map(|id| libsql::Value::Text(id.clone()))
            .collect();
        if let Some(value) = scope_value {
            params.push(value);
        }
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(&sql, params)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("filter_entity_ids_scoped: {error}")))?;
        let mut visible = std::collections::HashSet::new();
        while let Ok(Some(row)) = rows.next().await {
            if let Ok(id) = row.get::<String>(0) {
                visible.insert(id);
            }
        }
        Ok(entity_ids
            .iter()
            .filter(|id| visible.contains(id.as_str()))
            .cloned()
            .collect())
    }

    /// Route/retrieval-facing Entity vector search. The Global path retains
    /// DiskANN; selected scopes rank via a brute-force scan.
    pub async fn search_entities_by_vector_scoped(
        &self,
        query: &str,
        limit: usize,
        scope: &ReadScope,
    ) -> Result<Vec<EntitySearchResult>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.search_entities_by_vector(query, limit).await;
        }

        let embedding = self.get_or_compute_embedding(query)?;
        let vec_str = Self::vec_to_sql(&embedding);

        let conn = self.conn.lock().await;
        // G6 Stage 2 PR 2c sub-step 1: selection now ranks on the
        // `kind='entity'` shadow page's own embedding/space (via
        // `entity_page_map` joined to `pages`) instead of brute-forcing
        // `entities`. The hydration query below re-reads the same shadow
        // page for the same rows, so it's now a no-op overlay -- kept
        // in place (rather than deleted) for the same unconditional
        // program contract 1.5a established, and because dropping it is
        // a structural change outside this sub-step's query-swap scope.
        // Both queries still run under this one held `conn` guard, no
        // re-lock in between.

        // G6 Stage 2 PR 2c item 2: `pages.space` is folded the same way
        // `entities.space` was (never NULL; unfiled rows carry
        // `UNFILED_SPACE_ID`, confirmed by the same parity receipt), so
        // `Uncategorized` must match either.
        let (scope_sql, scope_value) = match scope {
            ReadScope::Space(space) => {
                ("AND p.space = ?3", Some(libsql::Value::Text(space.clone())))
            }
            ReadScope::Uncategorized => (
                "AND (p.space IS NULL OR p.space = '00000000-0000-4000-8000-000000000001')",
                None,
            ),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT m.entity_id, p.title, p.entity_type, p.space, p.source_agent, p.confidence, \
                    p.entity_confirmed, p.entity_created_at, p.entity_updated_at, \
                    vector_distance_cos(p.embedding, vector32(?1)) AS distance \
             FROM entity_page_map m \
             JOIN pages p ON p.id = m.page_id \
             WHERE p.kind = 'entity' AND p.status = 'active' AND p.embedding IS NOT NULL {scope_sql} \
             ORDER BY distance ASC LIMIT ?2"
        );
        let mut params = vec![
            libsql::Value::Text(vec_str),
            libsql::Value::Integer(limit as i64),
        ];
        if let Some(value) = scope_value {
            params.push(value);
        }
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("search_entities_by_vector_scoped: {error}"))
        })?;
        let mut results = Vec::new();
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("search_entities_by_vector_scoped next: {error}"))
        })? {
            results.push(EntitySearchResult {
                entity: entity_from_row(&row, "search_entities_by_vector_scoped")?,
                distance: row.get::<f64>(9).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "search_entities_by_vector_scoped distance: {error}"
                    ))
                })? as f32,
            });
        }

        if !results.is_empty() {
            struct Mirror {
                title: String,
                entity_type: String,
                confidence: Option<f64>,
                confirmed: i64,
            }
            let entity_ids: Vec<String> = results
                .iter()
                .map(|result| result.entity.id.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let placeholders = (1..=entity_ids.len())
                .map(|index| format!("?{index}"))
                .collect::<Vec<_>>()
                .join(",");
            let hydrate_sql = format!(
                "SELECT m.entity_id, p.title, p.entity_type, p.confidence, p.entity_confirmed \
                 FROM entity_page_map m \
                 JOIN pages p ON p.id = m.page_id AND p.kind = 'entity' AND p.status = 'active' \
                 WHERE m.entity_id IN ({placeholders})"
            );
            let hydrate_params: Vec<libsql::Value> = entity_ids
                .iter()
                .map(|id| libsql::Value::Text(id.clone()))
                .collect();
            let mut hydrate_rows =
                conn.query(&hydrate_sql, hydrate_params)
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!(
                            "search_entities_by_vector_scoped hydrate: {error}"
                        ))
                    })?;
            let mut mirrors: HashMap<String, Mirror> = HashMap::new();
            while let Some(row) = hydrate_rows.next().await.map_err(|error| {
                WenlanError::VectorDb(format!(
                    "search_entities_by_vector_scoped hydrate next: {error}"
                ))
            })? {
                let entity_id: String = row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "search_entities_by_vector_scoped hydrate entity_id: {error}"
                    ))
                })?;
                mirrors.insert(
                    entity_id,
                    Mirror {
                        title: row.get(1).map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "search_entities_by_vector_scoped hydrate title: {error}"
                            ))
                        })?,
                        entity_type: row.get(2).map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "search_entities_by_vector_scoped hydrate entity_type: {error}"
                            ))
                        })?,
                        confidence: row.get::<Option<f64>>(3).map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "search_entities_by_vector_scoped hydrate confidence: {error}"
                            ))
                        })?,
                        confirmed: row.get::<i64>(4).map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "search_entities_by_vector_scoped hydrate confirmed: {error}"
                            ))
                        })?,
                    },
                );
            }
            // A hydration miss (id with no live shadow row -- unreachable
            // under a clean gate) keeps the legacy-sourced fields already in
            // `results` rather than dropping or erroring the row.
            for result in &mut results {
                if let Some(mirror) = mirrors.get(&result.entity.id) {
                    result.entity.name = mirror.title.clone();
                    result.entity.entity_type = mirror.entity_type.clone();
                    result.entity.confidence = mirror.confidence.map(|value| value as f32);
                    result.entity.confirmed = mirror.confirmed != 0;
                }
            }
        }

        Ok(results)
    }

    pub async fn get_memories_for_entities_scoped(
        &self,
        ranked_anchor_ids: &[String],
        limit: usize,
        scope: &ReadScope,
    ) -> Result<Vec<SearchResult>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self
                .get_memories_for_entities(ranked_anchor_ids, limit)
                .await;
        }
        if ranked_anchor_ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;
        // M3 PR-2 stage c: NOT a cutover consumer. This query returns
        // MEMORIES linked to the given entity ids (`c.*`/`memory_entities`),
        // not entity rows, and sources no entity/page-mirrored field -- it
        // stays on its own substrate until the retirement rung widens the
        // mirror.
        let anchor_rank: HashMap<&str, usize> = ranked_anchor_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (id.as_str(), index))
            .collect();
        let placeholders = (1..=ranked_anchor_ids.len())
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(",");
        let scope_index = ranked_anchor_ids.len() + 1;
        let (scope_sql, scope_value) = match scope {
            ReadScope::Space(space) => (
                format!("AND c.space = ?{scope_index}"),
                Some(libsql::Value::Text(space.clone())),
            ),
            ReadScope::Uncategorized => (
                "AND (c.space IS NULL OR c.space = '00000000-0000-4000-8000-000000000001')"
                    .to_string(),
                None,
            ),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                    c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                    c.byte_end, c.semantic_unit, c.memory_type, c.space, c.source_agent,
                    c.confidence, c.confirmed, c.stability, c.supersedes,
                    c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                    c.structured_fields, c.retrieval_cue, c.source_text,
                    c.version, c.pending_revision,
                    0.0, c.importance, c.event_date, c.content_hash, me.entity_id
             FROM memories c
             JOIN memory_entities me ON me.memory_id = c.source_id
             WHERE me.entity_id IN ({placeholders})
               AND c.source = 'memory' AND c.chunk_index = 0 {scope_sql}"
        );
        let mut params: Vec<libsql::Value> = ranked_anchor_ids
            .iter()
            .map(|id| libsql::Value::Text(id.clone()))
            .collect();
        if let Some(value) = scope_value {
            params.push(value);
        }
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("get_memories_for_entities_scoped: {error}"))
        })?;
        let mut best = HashMap::new();
        while let Ok(Some(row)) = rows.next().await {
            let entity_id: String = row.get(34).unwrap_or_default();
            let rank = anchor_rank
                .get(entity_id.as_str())
                .copied()
                .unwrap_or(usize::MAX);
            let Ok(result) = Self::row_to_search_result(&row, 0.0) else {
                continue;
            };
            best.entry(result.source_id.clone())
                .and_modify(|(best_rank, _): &mut (usize, SearchResult)| {
                    *best_rank = (*best_rank).min(rank);
                })
                .or_insert((rank, result));
        }
        let mut ordered: Vec<(usize, SearchResult)> = best.into_values().collect();
        ordered.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.source_id.cmp(&right.1.source_id))
        });
        ordered.truncate(limit);
        Ok(ordered.into_iter().map(|(_, result)| result).collect())
    }

    pub(super) async fn get_observations_for_entities_scoped(
        &self,
        entity_ids: &[String],
        limit: usize,
        scope: &ReadScope,
    ) -> Result<Vec<SearchResult>, WenlanError> {
        if matches!(scope, ReadScope::Global) {
            return self.get_observations_for_entities(entity_ids, limit).await;
        }
        let visible_ids = self.filter_entity_ids_scoped(entity_ids, scope).await?;
        if visible_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = (1..=visible_ids.len())
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(",");
        let limit_parameter = visible_ids.len() + 1;
        let sql = format!(
            "SELECT o.id, o.content, e.title, o.created_at, o.source_agent, o.confidence \
             FROM observations o \
             JOIN entity_page_map epm ON o.entity_id = epm.entity_id \
             JOIN pages e ON e.id = epm.page_id AND e.kind = 'entity' AND e.status = 'active' \
             WHERE o.entity_id IN ({placeholders}) \
             ORDER BY o.confidence DESC NULLS LAST, o.created_at DESC \
             LIMIT ?{limit_parameter}"
        );
        let mut params: Vec<libsql::Value> = visible_ids
            .iter()
            .map(|id| libsql::Value::Text(id.clone()))
            .collect();
        params.push(libsql::Value::Integer(limit as i64));
        let conn = self.conn.lock().await;
        let mut rows = conn.query(&sql, params).await.map_err(|error| {
            WenlanError::VectorDb(format!("get_observations_for_entities_scoped: {error}"))
        })?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let id: String = row.get(0).unwrap_or_default();
            let entity_name: String = row.get(2).unwrap_or_default();
            results.push(SearchResult {
                id: id.clone(),
                content: row.get(1).unwrap_or_default(),
                source: "knowledge_graph".to_string(),
                source_id: format!("obs_{id}"),
                title: entity_name.clone(),
                url: None,
                chunk_index: 0,
                last_modified: row.get(3).unwrap_or(0),
                score: 0.0,
                chunk_type: None,
                language: None,
                semantic_unit: None,
                memory_type: None,
                space: None,
                source_agent: row.get::<Option<String>>(4).unwrap_or(None),
                confidence: row.get::<Option<f64>>(5).unwrap_or(None).map(|v| v as f32),
                confirmed: None,
                stability: None,
                supersedes: None,
                summary: None,
                entity_id: None,
                entity_name: Some(entity_name),
                quality: None,
                importance: None,
                event_date: None,
                is_archived: false,
                is_recap: false,
                structured_fields: None,
                retrieval_cue: None,
                source_text: None,
                content_hash: None,
                raw_score: 0.0,
                version: 0,
                pending_revision: false,
                merged_from: None,
                last_delta_summary: None,
            });
        }
        Ok(results)
    }

    /// One bulk read of the whole graph the scope can see: every entity, every
    /// live entity<->entity relation whose BOTH endpoints are in that entity
    /// set, every wiki page, and the memories those entities and pages reach.
    ///
    /// The desktop Graph view used to fan out per-entity detail fetches capped
    /// at the first 20 entities, which drew every other connected entity as an
    /// isolate. This is its single replacement request.
    ///
    /// Scoping: the entity half reuses `list_entities_scoped`, so relations
    /// need no scope predicate of their own — an endpoint outside the scope is
    /// simply not in the id set, and the relation is dropped. Memories carry
    /// their OWN scope filter (same rule as `list_memories_scoped`), so an
    /// in-scope entity linked to another space's memory does not leak it.
    /// Pages carry theirs (`page_scope_clause`, the same predicate the page
    /// readers use), so page links need none either: every endpoint is checked
    /// against an already-scoped set.
    ///
    /// Truth exposure is NOT decided here. The page half arrives whole and the
    /// route applies the caller's grant (`truth_adapter`), which is where every
    /// other page-bearing reader decides it.
    pub async fn get_knowledge_graph_scoped(
        &self,
        scope: &ReadScope,
    ) -> Result<wenlan_types::KnowledgeGraphResponse, WenlanError> {
        let entities = self.list_entities_scoped(None, scope).await?;
        // No early return on an empty entity set: wiki pages exist
        // independently of entities, and a scope with pages but no entities
        // still has a graph to draw.
        let entity_ids: HashSet<&str> = entities.iter().map(|entity| entity.id.as_str()).collect();

        // One held guard for both queries, no `.await` on anything else while
        // it is held (crate invariant).
        let conn = self.conn.lock().await;

        // Same live-relation predicate as `get_entity_detail_scoped`:
        // `edge_type='relates' AND valid_until IS NULL AND semantic_type IS
        // NOT NULL`. Endpoint membership is checked in Rust against the
        // already-scoped id set above rather than re-joining
        // `entity_page_map`/`pages` twice more here.
        let mut relation_rows = conn
            .query(
                "SELECT edge_id, src_id, dst_id, semantic_type, \
                        json_extract(payload, '$.source_agent'), \
                        COALESCE(json_extract(payload, '$.asserted_at'), created_at) AS asserted_ts \
                 FROM edges \
                 WHERE edge_type = 'relates' AND valid_until IS NULL \
                   AND semantic_type IS NOT NULL \
                 ORDER BY asserted_ts DESC, edge_id ASC",
                (),
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped relations query: {error}"))
            })?;
        let mut relations = Vec::new();
        while let Some(row) = relation_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!(
                "get_knowledge_graph_scoped relations next: {error}"
            ))
        })? {
            let from_entity: String = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped relation src: {error}"))
            })?;
            let to_entity: String = row.get(2).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped relation dst: {error}"))
            })?;
            if !entity_ids.contains(from_entity.as_str())
                || !entity_ids.contains(to_entity.as_str())
            {
                continue;
            }
            relations.push(wenlan_types::GraphRelation {
                id: row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_knowledge_graph_scoped relation id: {error}"
                    ))
                })?,
                from_entity,
                to_entity,
                relation_type: row.get(3).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_knowledge_graph_scoped relation type: {error}"
                    ))
                })?,
                source_agent: row.get::<Option<String>>(4).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_knowledge_graph_scoped relation source_agent: {error}"
                    ))
                })?,
                created_at: row.get(5).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "get_knowledge_graph_scoped relation created_at: {error}"
                    ))
                })?,
            });
        }
        drop(relation_rows);

        // Memory rows resolve the same way `list_memories_scoped` does:
        // `source = 'memory'`, no pending revision, superseded rows hidden,
        // fields folded with MAX over the memory's chunks.
        let hidden_by_superseder = super::superseder_not_exists(
            scope,
            "m",
            "superseder.pending_revision = 0 AND superseder.source = 'memory'",
        );
        let memory_scope_sql = match scope {
            ReadScope::Global => "",
            ReadScope::Space(_) => "AND m.space = ?1",
            ReadScope::Uncategorized => {
                "AND (m.space IS NULL OR m.space = '00000000-0000-4000-8000-000000000001')"
            }
        };
        // Rebuilt rather than moved: both the entity-linked and the
        // page-cited memory queries below bind the same `?1`.
        let memory_scope_params = || -> Vec<libsql::Value> {
            match scope {
                ReadScope::Space(space) => vec![libsql::Value::Text(space.clone())],
                _ => Vec::new(),
            }
        };
        let memory_sql = format!(
            "SELECT me.entity_id, m.source_id, MAX(m.title), MAX(m.memory_type), \
                    MAX(m.space), MAX(m.confirmed), MAX(m.last_modified) \
             FROM memory_entities me \
             JOIN memories m ON m.source_id = me.memory_id \
             WHERE m.source = 'memory' AND m.pending_revision = 0 \
               AND {hidden_by_superseder} {memory_scope_sql} \
             GROUP BY me.entity_id, m.source_id \
             ORDER BY m.source_id ASC, me.entity_id ASC"
        );
        let mut memory_rows = conn
            .query(&memory_sql, libsql::params_from_iter(memory_scope_params()))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!(
                    "get_knowledge_graph_scoped memories query: {error}"
                ))
            })?;
        let mut memories: HashMap<String, wenlan_types::GraphMemoryNode> = HashMap::new();
        let mut memory_links = Vec::new();
        while let Some(row) = memory_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("get_knowledge_graph_scoped memories next: {error}"))
        })? {
            let entity_id: String = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!(
                    "get_knowledge_graph_scoped link entity_id: {error}"
                ))
            })?;
            if !entity_ids.contains(entity_id.as_str()) {
                continue;
            }
            let source_id: String = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!(
                    "get_knowledge_graph_scoped memory source_id: {error}"
                ))
            })?;
            if !memories.contains_key(&source_id) {
                memories.insert(
                    source_id.clone(),
                    wenlan_types::GraphMemoryNode {
                        source_id: source_id.clone(),
                        title: row
                            .get::<Option<String>>(2)
                            .unwrap_or(None)
                            .unwrap_or_default(),
                        memory_type: row.get::<Option<String>>(3).unwrap_or(None),
                        // The wire contract keeps `null` for "unfiled", same
                        // normalization the entity rows get.
                        space: crate::space_context::normalize_unfiled_space(
                            row.get::<Option<String>>(4).unwrap_or(None),
                        ),
                        confirmed: row.get::<i64>(5).unwrap_or(0) != 0,
                        last_modified: row.get::<i64>(6).unwrap_or(0),
                    },
                );
            }
            memory_links.push(wenlan_types::GraphMemoryLink {
                memory_id: source_id,
                entity_id,
            });
        }
        drop(memory_rows);

        // ---- wiki pages ------------------------------------------------
        //
        // Every page that is NOT an entity's `kind='entity'` dual-write
        // shadow. Both markers are checked: a shadow carries `kind='entity'`
        // and `creation_kind='entity'`, and a row that somehow lost one must
        // still not arrive here as a wiki page — the entity is already a node,
        // so its shadow would draw the same thing twice.
        let mut page_links: Vec<wenlan_types::GraphPageLink> = Vec::new();
        let (page_scope_sql, page_scope_value) =
            super::scoped_pages::page_scope_clause(scope, "p.workspace", 1);
        let page_sql = format!(
            "SELECT p.id, p.title, p.space, COALESCE(p.creation_kind, 'distilled'), \
                    p.entity_id, p.last_modified \
             FROM pages p \
             WHERE p.status = 'active' \
               AND COALESCE(p.kind, 'concept') != 'entity' \
               AND COALESCE(p.creation_kind, 'distilled') != 'entity'\
               {page_scope_sql} \
             ORDER BY p.id ASC"
        );
        let mut page_rows = conn
            .query(
                &page_sql,
                libsql::params_from_iter(page_scope_value.into_iter().collect::<Vec<_>>()),
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped pages query: {error}"))
            })?;
        let mut pages = Vec::new();
        while let Some(row) = page_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("get_knowledge_graph_scoped pages next: {error}"))
        })? {
            let id: String = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped page id: {error}"))
            })?;
            let entity_id = row.get::<Option<String>>(4).unwrap_or(None);
            // `pages.entity_id` says this page is ABOUT that entity — a
            // different thing from being its shadow. Emitted only when the
            // entity is one the scope can already see.
            if let Some(entity_id) = entity_id
                .as_deref()
                .filter(|entity_id| entity_ids.contains(entity_id))
            {
                page_links.push(page_link(&id, PAGE, entity_id, ENTITY, "about"));
            }
            pages.push(wenlan_types::GraphPageNode {
                id,
                title: row
                    .get::<Option<String>>(1)
                    .unwrap_or(None)
                    .unwrap_or_default(),
                // Same `null`-for-unfiled normalization the entity and memory
                // rows get, so all three collections speak one space vocabulary.
                space: crate::space_context::normalize_unfiled_space(
                    row.get::<Option<String>>(2).unwrap_or(None),
                ),
                creation_kind: row
                    .get::<Option<String>>(3)
                    .unwrap_or(None)
                    .unwrap_or_else(|| "distilled".to_string()),
                entity_id,
                last_modified: row
                    .get::<Option<String>>(5)
                    .unwrap_or(None)
                    .unwrap_or_default(),
            });
        }
        drop(page_rows);
        let page_ids: HashSet<&str> = pages.iter().map(|page| page.id.as_str()).collect();

        // ---- resolved wikilinks ----------------------------------------
        //
        // Read from `edges`, NOT from `page_links`. Since the G6 Stage 2
        // orphan-only narrowing, `replace_page_links` writes a `page_links`
        // row only for an UNRESOLVED `[[link]]` and deletes the page's whole
        // row set on every re-link; a resolved link's sole canonical
        // representation is the `links` edge minted beside it. This is the
        // same store `get_page_outbound_links_scoped` reads.
        //
        // A dangling link reaches nothing and is simply not an edge. A
        // resolved target is another wiki page; it can also be an entity's
        // shadow page for a legacy row (today's title resolver excludes
        // `kind='entity'`), and that is drawn as a link to the ENTITY, since
        // the shadow itself is not on the map. No scope predicate: both
        // endpoints are checked against the sets above, which are already
        // scoped. Several labels can resolve to one target, so pairs are
        // de-duplicated.
        let mut seen_wikilinks: HashSet<(String, String)> = HashSet::new();
        let mut wikilink_rows = conn
            .query(
                "SELECT e.src_id, e.dst_id, \
                        COALESCE(t.kind, 'concept'), epm.entity_id \
                 FROM edges e \
                 JOIN pages t ON t.id = e.dst_id \
                 LEFT JOIN entity_page_map epm ON epm.page_id = t.id \
                 WHERE e.edge_type = 'links' AND e.src_kind = 'page' \
                   AND e.dst_kind = 'page' AND e.valid_until IS NULL \
                 ORDER BY e.src_id ASC, e.dst_id ASC",
                (),
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!(
                    "get_knowledge_graph_scoped wikilinks query: {error}"
                ))
            })?;
        while let Some(row) = wikilink_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!(
                "get_knowledge_graph_scoped wikilinks next: {error}"
            ))
        })? {
            let source: String = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped wikilink src: {error}"))
            })?;
            if !page_ids.contains(source.as_str()) {
                continue;
            }
            let target: String = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped wikilink dst: {error}"))
            })?;
            let target_kind: String = row.get(2).unwrap_or_else(|_| "concept".to_string());
            if target_kind == "entity" {
                let Some(entity_id) = row.get::<Option<String>>(3).unwrap_or(None) else {
                    continue;
                };
                if entity_ids.contains(entity_id.as_str())
                    && seen_wikilinks.insert((source.clone(), entity_id.clone()))
                {
                    page_links.push(page_link(&source, PAGE, &entity_id, ENTITY, "wikilink"));
                }
                continue;
            }
            // A page linking to itself is not a relationship between two
            // things; it would draw a self-loop on the map for nothing.
            if target != source
                && page_ids.contains(target.as_str())
                && seen_wikilinks.insert((source.clone(), target.clone()))
            {
                page_links.push(page_link(&source, PAGE, &target, PAGE, "wikilink"));
            }
        }
        drop(wikilink_rows);

        // ---- page -> memory citations ----------------------------------
        //
        // Same rows `get_page_sources_scoped` reads. Joined straight to
        // `memories` so one query both resolves the cited memory row (which
        // may not be linked to any entity, and so may not be in the map yet)
        // and yields the edge — no runtime id list, no parameter-count ceiling.
        // A memory row that is gone, pending, superseded or out of scope
        // simply does not join, and the citation is dropped with it.
        let cites_sql = format!(
            "SELECT e.src_id, m.source_id, MAX(m.title), MAX(m.memory_type), \
                    MAX(m.space), MAX(m.confirmed), MAX(m.last_modified) \
             FROM edges e \
             JOIN memories m ON m.source_id = e.dst_id \
             WHERE e.edge_type = 'cites' AND e.valid_until IS NULL \
               AND m.source = 'memory' AND m.pending_revision = 0 \
               AND {hidden_by_superseder} {memory_scope_sql} \
             GROUP BY e.src_id, m.source_id \
             ORDER BY e.src_id ASC, m.source_id ASC"
        );
        let mut cites_rows = conn
            .query(&cites_sql, libsql::params_from_iter(memory_scope_params()))
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped cites query: {error}"))
            })?;
        while let Some(row) = cites_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("get_knowledge_graph_scoped cites next: {error}"))
        })? {
            let page_id: String = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped cites src: {error}"))
            })?;
            if !page_ids.contains(page_id.as_str()) {
                continue;
            }
            let source_id: String = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!("get_knowledge_graph_scoped cites dst: {error}"))
            })?;
            memories
                .entry(source_id.clone())
                .or_insert_with(|| wenlan_types::GraphMemoryNode {
                    source_id: source_id.clone(),
                    title: row
                        .get::<Option<String>>(2)
                        .unwrap_or(None)
                        .unwrap_or_default(),
                    memory_type: row.get::<Option<String>>(3).unwrap_or(None),
                    space: crate::space_context::normalize_unfiled_space(
                        row.get::<Option<String>>(4).unwrap_or(None),
                    ),
                    confirmed: row.get::<i64>(5).unwrap_or(0) != 0,
                    last_modified: row.get::<i64>(6).unwrap_or(0),
                });
            page_links.push(page_link(&page_id, PAGE, &source_id, MEMORY, "cites"));
        }
        drop(cites_rows);
        drop(conn);

        let mut memories: Vec<wenlan_types::GraphMemoryNode> = memories.into_values().collect();
        memories.sort_by(|left, right| left.source_id.cmp(&right.source_id));

        Ok(wenlan_types::KnowledgeGraphResponse {
            entities,
            relations,
            memories,
            memory_links,
            pages,
            page_links,
        })
    }
}

/// The three collections a [`wenlan_types::GraphRef`] can point into. Spelled
/// once so a typo cannot send a caller looking in the wrong one.
const PAGE: &str = "page";
const ENTITY: &str = "entity";
const MEMORY: &str = "memory";

fn page_link(
    from_id: &str,
    from_kind: &str,
    to_id: &str,
    to_kind: &str,
    link_type: &str,
) -> wenlan_types::GraphPageLink {
    wenlan_types::GraphPageLink {
        from: wenlan_types::GraphRef {
            kind: from_kind.to_string(),
            id: from_id.to_string(),
        },
        to: wenlan_types::GraphRef {
            kind: to_kind.to_string(),
            id: to_id.to_string(),
        },
        link_type: link_type.to_string(),
    }
}

fn entity_from_row(row: &libsql::Row, context: &str) -> Result<Entity, WenlanError> {
    Ok(Entity {
        id: row
            .get(0)
            .map_err(|error| WenlanError::VectorDb(format!("{context} id: {error}")))?,
        name: row
            .get(1)
            .map_err(|error| WenlanError::VectorDb(format!("{context} name: {error}")))?,
        entity_type: row
            .get(2)
            .map_err(|error| WenlanError::VectorDb(format!("{context} type: {error}")))?,
        // G6 Stage 1.5b Part 2: `entities.space` is folded (never NULL;
        // unfiled rows carry `UNFILED_SPACE_ID`), but the wire contract
        // keeps `null` for "unfiled" -- normalize at this serialization
        // boundary.
        space: crate::space_context::normalize_unfiled_space(
            row.get::<Option<String>>(3)
                .map_err(|error| WenlanError::VectorDb(format!("{context} space: {error}")))?,
        ),
        source_agent: row
            .get::<Option<String>>(4)
            .map_err(|error| WenlanError::VectorDb(format!("{context} source_agent: {error}")))?,
        confidence: row
            .get::<Option<f64>>(5)
            .map_err(|error| WenlanError::VectorDb(format!("{context} confidence: {error}")))?
            .map(|value| value as f32),
        confirmed: row
            .get::<i64>(6)
            .map_err(|error| WenlanError::VectorDb(format!("{context} confirmed: {error}")))?
            != 0,
        created_at: row
            .get(7)
            .map_err(|error| WenlanError::VectorDb(format!("{context} created_at: {error}")))?,
        updated_at: row
            .get(8)
            .map_err(|error| WenlanError::VectorDb(format!("{context} updated_at: {error}")))?,
    })
}
