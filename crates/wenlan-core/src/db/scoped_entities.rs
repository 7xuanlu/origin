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
        // M3 PR-2 stage c: the vanguard flip. `name`/`entity_type`/
        // `confidence`/`confirmed` are mirrored fields, sourced from the
        // shadow page under a clean, current parity proof; `space` etc.
        // stay live off `entities` (see `reader_uses_entity_pages`'s docs).
        // Filter/scope predicates are unchanged either way, only qualified
        // to the `e.` alias on the hybrid path -- clean parity means both
        // columns agree, so which side a predicate checks cannot change
        // the result set.
        let cutover_on = Self::reader_uses_entity_pages(&conn, Self::SCOPED_ENTITIES_CONSUMER)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("list_entities_scoped gate: {error}"))
            })?;
        let entity_prefix = if cutover_on { "e." } else { "" };

        let mut conditions = Vec::new();
        let mut values = Vec::new();
        if let Some(entity_type) = entity_type {
            conditions.push(format!("{entity_prefix}entity_type = ?"));
            values.push(libsql::Value::Text(entity_type.to_string()));
        }
        super::push_read_scope_filter(
            scope,
            &format!("{entity_prefix}space"),
            &mut conditions,
            &mut values,
        );
        let sql = if cutover_on {
            format!(
                "SELECT e.id, p.title, p.entity_type, e.space, e.source_agent, p.confidence, \
                        p.entity_confirmed, e.created_at, e.updated_at \
                 FROM entities e \
                 JOIN entity_page_map m ON m.entity_id = e.id \
                 JOIN pages p ON p.id = m.page_id AND p.kind = 'entity' AND p.status = 'active' \
                 WHERE {} ORDER BY e.updated_at DESC, e.id ASC",
                conditions.join(" AND ")
            )
        } else {
            format!(
                "SELECT id, name, entity_type, space, source_agent, confidence, confirmed, \
                        created_at, updated_at \
                 FROM entities WHERE {} ORDER BY updated_at DESC, id ASC",
                conditions.join(" AND ")
            )
        };
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
        // M3 PR-2 stage c: the vanguard flip, for the primary entity row
        // only -- the observations query below never touched `entities`,
        // and the relations query's own `entities` JOIN hydrates the
        // OTHER side of each relation (a distinct case the design deferred
        // to a later retirement rung), so neither changes here.
        let cutover_on = Self::reader_uses_entity_pages(&conn, Self::SCOPED_ENTITIES_CONSUMER)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("get_entity_detail_scoped gate: {error}"))
            })?;
        let entity_prefix = if cutover_on { "e." } else { "" };

        let mut conditions = vec![format!("{entity_prefix}id = ?")];
        let mut values = vec![libsql::Value::Text(entity_id.to_string())];
        super::push_read_scope_filter(
            scope,
            &format!("{entity_prefix}space"),
            &mut conditions,
            &mut values,
        );
        let entity_sql = if cutover_on {
            format!(
                "SELECT e.id, p.title, p.entity_type, e.space, e.source_agent, p.confidence, \
                        p.entity_confirmed, e.created_at, e.updated_at \
                 FROM entities e \
                 JOIN entity_page_map m ON m.entity_id = e.id \
                 JOIN pages p ON p.id = m.page_id AND p.kind = 'entity' AND p.status = 'active' \
                 WHERE {} LIMIT 1",
                conditions.join(" AND ")
            )
        } else {
            format!(
                "SELECT id, name, entity_type, space, source_agent, confidence, confirmed, \
                        created_at, updated_at \
                 FROM entities WHERE {} LIMIT 1",
                conditions.join(" AND ")
            )
        };
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

        let (endpoint_filter, endpoint_value) = match scope {
            ReadScope::Space(space) => {
                ("AND e.space = ?2", Some(libsql::Value::Text(space.clone())))
            }
            ReadScope::Uncategorized => ("AND e.space IS NULL", None),
            ReadScope::Global => unreachable!(),
        };
        let relation_sql = format!(
            "SELECT r.edge_id, r.semantic_type, json_extract(r.payload, '$.source_agent'), \
                    COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at), \
                    'outgoing' AS direction, r.dst_id AS entity_id, \
                    e.name AS entity_name, e.entity_type AS entity_type \
             FROM edges r JOIN entities e ON e.id = r.dst_id \
             WHERE r.edge_type = 'relates' AND r.valid_until IS NULL \
               AND r.semantic_type IS NOT NULL \
               AND r.src_id = ?1 {endpoint_filter} \
             UNION ALL \
             SELECT r.edge_id, r.semantic_type, json_extract(r.payload, '$.source_agent'), \
                    COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at), \
                    'incoming' AS direction, r.src_id AS entity_id, \
                    e.name AS entity_name, e.entity_type AS entity_type \
             FROM edges r JOIN entities e ON e.id = r.src_id \
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
        // M3 PR-2 stage f (Sol review fix 3+4): tie-safe flip. Selection
        // always runs the EXACT legacy query below -- row set and order
        // (ties included) are therefore identical to the OFF path by
        // construction, not by keeping a second hand-written arm in sync.
        // When the gate is ON, a hydration query below overlays
        // `from_entity_name`/`to_entity_name` from each side's shadow page,
        // falling back to the legacy `e1.name`/`e2.name` already selected
        // on a hydration miss. Both queries run under this one held `conn`
        // guard, no re-lock in between.
        let cutover_on = Self::reader_uses_entity_pages(&conn, Self::SCOPED_ENTITIES_CONSUMER)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("list_recent_relations_scoped gate: {error}"))
            })?;

        let (scope_filter, scope_value) = match scope {
            ReadScope::Space(space) => (
                "AND e1.space = ?3 AND e2.space = ?3",
                Some(libsql::Value::Text(space.clone())),
            ),
            ReadScope::Uncategorized => ("AND e1.space IS NULL AND e2.space IS NULL", None),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT r.edge_id, r.src_id, r.semantic_type, r.dst_id, \
                    e1.name, e2.name, \
                    COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at) AS asserted_ts \
             FROM edges r \
             JOIN entities e1 ON r.src_id = e1.id \
             JOIN entities e2 ON r.dst_id = e2.id \
             WHERE r.edge_type = 'relates' AND r.valid_until IS NULL \
               AND r.semantic_type IS NOT NULL \
               AND (?1 IS NULL OR COALESCE(json_extract(r.payload, '$.asserted_at'), r.created_at) >= ?1) \
               AND e1.name IS NOT NULL AND e1.name != '' \
               AND e2.name IS NOT NULL AND e2.name != '' \
               {scope_filter} \
             ORDER BY asserted_ts DESC LIMIT ?2"
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

        if cutover_on && !relations.is_empty() {
            let entity_ids: Vec<String> = relations
                .iter()
                .flat_map(|relation| {
                    [
                        relation.from_entity_id.clone(),
                        relation.to_entity_id.clone(),
                    ]
                })
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
                            "list_recent_relations_scoped hydrate: {error}"
                        ))
                    })?;
            let mut titles: HashMap<String, String> = HashMap::new();
            while let Some(row) = hydrate_rows.next().await.map_err(|error| {
                WenlanError::VectorDb(format!(
                    "list_recent_relations_scoped hydrate next: {error}"
                ))
            })? {
                let entity_id: String = row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped hydrate entity_id: {error}"
                    ))
                })?;
                let title: String = row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "list_recent_relations_scoped hydrate title: {error}"
                    ))
                })?;
                titles.insert(entity_id, title);
            }
            // A hydration miss (id with no live shadow row -- unreachable
            // under a clean gate) keeps the legacy-sourced name already in
            // `relations` rather than dropping or erroring the row.
            for relation in &mut relations {
                if let Some(title) = titles.get(&relation.from_entity_id) {
                    relation.from_entity_name = title.clone();
                }
                if let Some(title) = titles.get(&relation.to_entity_id) {
                    relation.to_entity_name = title.clone();
                }
            }
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
    /// DiskANN; selected scopes rank only rows satisfying `entities.space`.
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
        // M3 PR-2 stage f (Sol review fix 3+4): tie-safe flip. Selection
        // always runs the EXACT legacy ANN query below -- row set and order
        // (tied distances included) are therefore identical to the OFF path
        // by construction. The ANN ordering/filter stay on `e.embedding`
        // (the entity-side DiskANN index; the design deliberately does not
        // swap ranking infrastructure -- the page embedding is byte-identical
        // by parity, not a different index). When the gate is ON, a
        // hydration query below overlays the mirrored columns
        // (name/entity_type/confidence/confirmed) from each row's shadow
        // page, falling back to the legacy entity-side value already
        // selected on a hydration miss. Both queries run under this one held
        // `conn` guard, no re-lock in between.
        let cutover_on = Self::reader_uses_entity_pages(&conn, Self::SCOPED_ENTITIES_CONSUMER)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("search_entities_by_vector_scoped gate: {error}"))
            })?;

        let (scope_sql, scope_value) = match scope {
            ReadScope::Space(space) => {
                ("AND e.space = ?3", Some(libsql::Value::Text(space.clone())))
            }
            ReadScope::Uncategorized => ("AND e.space IS NULL", None),
            ReadScope::Global => unreachable!(),
        };
        let sql = format!(
            "SELECT e.id, e.name, e.entity_type, e.space, e.source_agent, e.confidence, \
                    e.confirmed, e.created_at, e.updated_at, \
                    vector_distance_cos(e.embedding, vector32(?1)) AS distance \
             FROM entities e \
             WHERE e.embedding IS NOT NULL {scope_sql} \
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

        if cutover_on && !results.is_empty() {
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
            "SELECT o.id, o.content, e.name, o.created_at, o.source_agent, o.confidence \
             FROM observations o JOIN entities e ON o.entity_id = e.id \
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
        space: row
            .get::<Option<String>>(3)
            .map_err(|error| WenlanError::VectorDb(format!("{context} space: {error}")))?,
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
