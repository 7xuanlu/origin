# origin-types 0.1.0 public-surface audit

Date: 2026-04-12
Purpose: classify each publicly-reachable type into WIRE / TAURI / INTERNAL to
determine `#[doc(hidden)]` placement per D0 of the shared-types drift-fix spec.

## Methodology

1. Enumerated all `pub struct` / `pub enum` across all 6 files in
   `crates/origin-types/src/` — 119 unique types.
2. For each type, grepped `crates/origin-server/src/` and
   `crates/origin-core/src/` for:
   - `Json<Type>` extractors and return types on axum handlers
   - `State<>` extraction
   - Response literal construction
   - Field nesting inside other wire types
   - Direct path-qualified import (`origin_types::requests::Type`,
     `origin_types::responses::Type`, `requests::Type`, `responses::Type`).
3. For each type, grepped `app/src/` for:
   - `#[tauri::command]` fn returns/parameters
   - Client-side construction of origin-types request bodies to POST/PUT
     against the daemon (these traverse the same HTTP wire contract)
   - Client-side deserialization of responses from the daemon.
4. Types where origin-server has a **local shadow** `pub struct TypeName`
   *and* the origin-types version is NOT directly imported on server OR app
   were flagged INTERNAL (the wire is served by the shadow; the origin-types
   copy is not on any actual wire path today).

Rules (per Task 7 plan):
- Dual-use types (HTTP + Tauri) are WIRE; Tauri column just records the
  additional surface.
- Nested types inside a wire parent (e.g. `TopMemory` inside `HomeStats`,
  `Observation` inside `EntityDetail`) are WIRE by transitive serialization.
- Be conservative: default to WIRE/TAURI when unsure.

## Classification

### entities.rs

| Type | Origin file | HTTP wire? | Tauri wire? | Verdict |
|---|---|---|---|---|
| Entity | entities.rs | YES (nested in EntityDetail, ListEntitiesResponse) | YES (app/src/search.rs:1397 `list_entities_cmd -> Vec<Entity>`) | WIRE — stays pub |
| EntityDetail | entities.rs | YES (`Json<origin_core::db::EntityDetail>` in memory_routes.rs:1100; re-export of origin_types) | YES (app/src/search.rs:1432 `get_entity_detail_cmd -> EntityDetail`) | WIRE — stays pub |
| EntitySearchResult | entities.rs | YES (nested in SearchEntitiesResponse) | YES (app/src/search.rs:1415 `search_entities_cmd -> Vec<EntitySearchResult>`) | WIRE — stays pub |
| EntitySuggestion | entities.rs | YES (`Vec<origin_types::EntitySuggestion>` consumed at app/src/search.rs:1932 from `/api/memory/entity-suggestions`) | YES (app/src/search.rs:1930 `get_entity_suggestions_cmd -> Vec<EntitySuggestion>`) | WIRE — stays pub |
| Observation | entities.rs | YES (nested in EntityDetail, constructed in origin-core/db.rs:6558) | YES (transitive via EntityDetail Tauri return) | WIRE — stays pub |
| Relation | entities.rs | YES (nested via origin_core::db re-export; constructed in origin-core schemas) | NO | WIRE — stays pub |
| RelationWithEntity | entities.rs | YES (nested in EntityDetail, constructed in origin-core/db.rs:6595) | YES (transitive via EntityDetail Tauri return) | WIRE — stays pub |

### import.rs

| Type | Origin file | HTTP wire? | Tauri wire? | Verdict |
|---|---|---|---|---|
| ImportChatExportRequest | import.rs | YES (`Json<ImportChatExportRequest>` in import_routes.rs:84) | YES (app/src/api.rs:171 constructs for `import_chat_export` Tauri cmd chain) | WIRE — stays pub |
| ImportChatExportResponse | import.rs | YES (`Json<ImportChatExportResponse>` in import_routes.rs:85) | YES (app/src/search.rs:1332 `import_chat_export -> ImportChatExportResponse`) | WIRE — stays pub |
| PendingImport | import.rs | YES (`Json<Vec<origin_types::import::PendingImport>>` in import_routes.rs:231) | YES (app/src/search.rs:1340 `list_pending_imports -> Vec<PendingImport>`) | WIRE — stays pub |

### memory.rs

| Type | Origin file | HTTP wire? | Tauri wire? | Verdict |
|---|---|---|---|---|
| AgentActivityRow | memory.rs | YES (nested in ActivityResponse, constructed in origin-core/db) | YES (app/src/search.rs:1903 `list_agent_activity -> Vec<AgentActivityRow>`) | WIRE — stays pub |
| AgentConnection | memory.rs | YES (constructed in origin-core, used by app for `/api/agents` round-trip) | YES (app/src/search.rs:1583 `list_agents -> Vec<AgentConnection>`) | WIRE — stays pub |
| DomainInfo | memory.rs | YES (nested `domains: Vec<DomainInfo>` in MemoryStats; constructed in origin-core/db.rs:7173) | YES (transitive via MemoryStats Tauri return) | WIRE — stays pub |
| HomeStats | memory.rs | YES (`Json<origin_types::HomeStats>` in memory_routes.rs:1163) | YES (app/src/search.rs:1071 `get_home_stats -> HomeStats`) | WIRE — stays pub |
| IndexedFileInfo | memory.rs | YES (nested in ListMemoriesResponse, IndexedFilesResponse) | YES (app/src/search.rs:951 `list_memories -> Vec<IndexedFileInfo>`; :1117 `list_indexed_files`) | WIRE — stays pub |
| MemoryItem | memory.rs | YES (nested in MemoryDetailResponse, NurtureCardsResponse, DecisionsResponse, PinnedMemoriesResponse) | YES (app/src/search.rs:1000 `list_memories_cmd -> Vec<MemoryItem>` + many) | WIRE — stays pub |
| MemoryStats | memory.rs | YES (nested in MemoryStatsResponse.stats) | YES (app/src/search.rs:1064 `get_memory_stats_cmd -> MemoryStats`) | WIRE — stays pub |
| MemoryVersionItem | memory.rs | YES (nested in VersionChainResponse) | YES (app/src/search.rs:1103 `get_version_chain_cmd -> Vec<MemoryVersionItem>`) | WIRE — stays pub |
| Profile | memory.rs | YES (constructed in origin-core; proxied by app from ProfileResponse) | YES (app/src/search.rs:1539 `get_profile -> Option<Profile>`) | WIRE — stays pub |
| RejectionRecord | memory.rs | YES (returned by `/api/memory/rejections`; app consumes at search.rs:2554) | YES (app/src/search.rs:2554 `get_rejection_log -> Vec<RejectionRecord>`) | WIRE — stays pub |
| SearchResult | memory.rs | YES (nested in SearchMemoryResponse, SearchResponse, KnowledgeContext.relevant_memories) | YES (app/src/search.rs:846 `search -> Vec<SearchResult>`; :862 `search_memory`) | WIRE — stays pub |
| SessionSnapshot | memory.rs | YES (`GET /api/snapshots` returns `Vec<SessionSnapshot>`) | YES (app/src/search.rs:2276 `get_session_snapshots -> Vec<origin_types::SessionSnapshot>`) | WIRE — stays pub |
| SnapshotCapture | memory.rs | YES (`GET /api/snapshots/{id}/captures`) | YES (app/src/search.rs:2286 `get_snapshot_captures -> Vec<origin_types::SnapshotCapture>`) | WIRE — stays pub |
| SnapshotCaptureWithContent | memory.rs | YES (`GET /api/snapshots/{id}/captures-with-content`) | YES (app/src/search.rs:2297 `get_snapshot_captures_with_content -> Vec<origin_types::SnapshotCaptureWithContent>`) | WIRE — stays pub |
| Space | memory.rs | YES (`/api/spaces` returns `Vec<Space>`, constructed in origin-core) | YES (app/src/search.rs:1983 `list_spaces -> Vec<Space>` + create/update/delete/pin) | WIRE — stays pub |
| TopMemory | memory.rs | YES (nested `top_memories: Vec<TopMemory>` in HomeStats; constructed in origin-core/db.rs:7533) | YES (transitive via HomeStats Tauri return) | WIRE — stays pub |
| TypeBreakdown | memory.rs | YES (nested `by_type: Vec<TypeBreakdown>` in MemoryStats; constructed in origin-core/db.rs:7196) | YES (transitive via MemoryStats Tauri return) | WIRE — stays pub |

### requests.rs

| Type | Origin file | HTTP wire? | Tauri wire? | Verdict |
|---|---|---|---|---|
| AddEntityObservationRequest | requests.rs | YES (`Json<origin_types::requests::AddEntityObservationRequest>` in memory_routes.rs:1860) | NO | WIRE — stays pub |
| AddObservationRequest | requests.rs | YES (app/src/search.rs:1525 constructs `requests::AddObservationRequest` and POSTs to `/api/memory/observations`) | YES (transitive via `add_observation_cmd` Tauri cmd) | WIRE — stays pub |
| AddSourceRequest | requests.rs | NO — origin-server has local shadow at source_routes.rs:23; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| BulkDeleteItem | requests.rs | YES (nested in BulkDeleteRequest.items) | YES (app/src/search.rs:1195 `delete_bulk(items: Vec<BulkDeleteItem>)` — via `use origin_types::*`; note app also has a local shadow) | WIRE — stays pub |
| BulkDeleteRequest | requests.rs | YES (constructed in app/src/search.rs:1198 and POSTed to `/api/memory/bulk-delete`) | YES (transitive via `delete_bulk` Tauri cmd) | WIRE — stays pub |
| ChatContextRequest | requests.rs | YES (`Json<ChatContextRequest>` in routes.rs:194; imported at routes.rs:10) | NO | WIRE — stays pub |
| ConfirmEntityRequest | requests.rs | YES (`Json<origin_types::requests::ConfirmEntityRequest>` in memory_routes.rs:1829) | YES (app/src/search.rs:1490 constructs `requests::ConfirmEntityRequest` in `confirm_entity_cmd`) | WIRE — stays pub |
| ConfirmObservationRequest | requests.rs | YES (`Json<origin_types::requests::ConfirmObservationRequest>` in memory_routes.rs:1913) | YES (app/src/search.rs:1505 constructs `requests::ConfirmObservationRequest` in `confirm_observation_cmd`) | WIRE — stays pub |
| ConfirmRequest | requests.rs | YES (imported in memory_routes.rs:11) | YES (app/src/search.rs:914 constructs `requests::ConfirmRequest` in `confirm_memory`) | WIRE — stays pub |
| ContextRequest | requests.rs | NO — origin-server has local shadow at routes.rs:130; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| CorrectMemoryRequest | requests.rs | YES (app/src/search.rs:2340 constructs `requests::CorrectMemoryRequest` for `correct_memory_cmd` Tauri cmd → daemon) | YES (`correct_memory_cmd`) | WIRE — stays pub |
| CreateConceptRequest | requests.rs | NO — origin-server has local shadow at memory_routes.rs:1611; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| CreateEntityRequest | requests.rs | YES (app/src/search.rs:1380 constructs `requests::CreateEntityRequest` for Tauri cmd → daemon) | YES (`create_entity_cmd`) | WIRE — stays pub (origin-server has local shadow; client path is origin-types-based) |
| CreateRelationRequest | requests.rs | NO — origin-server has local shadow at memory_routes.rs:103; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| CreateSpaceRequest | requests.rs | YES (app/src/search.rs:2006 constructs `requests::CreateSpaceRequest` to POST `/api/spaces`) | YES (Tauri `create_space`, `add_space`) | WIRE — stays pub (origin-server has local shadow) |
| DeleteByTimeRangeRequest | requests.rs | YES (app/src/search.rs:1183 constructs `requests::DeleteByTimeRangeRequest`) | YES (Tauri `delete_by_time_range`) | WIRE — stays pub |
| ExportConceptRequest | requests.rs | YES (app/src/search.rs:2506 constructs `requests::ExportConceptRequest`) | YES (Tauri `export_concept_to_obsidian`) | WIRE — stays pub |
| ImportMemoriesRequest | requests.rs | YES (app/src/search.rs:1313 constructs `requests::ImportMemoriesRequest` for daemon POST) | YES (transitive via `import_memories_cmd` Tauri cmd) | WIRE — stays pub (origin-server has local shadow at import_routes.rs) |
| IngestMemoryRequest | requests.rs | YES (app/src/search.rs:1245 constructs `requests::IngestMemoryRequest`) | YES (transitive via `ingest_clipboard` Tauri cmd) | WIRE — stays pub (origin-server has local shadow at ingest_routes.rs) |
| IngestTextRequest | requests.rs | YES (app/src/router/intent.rs:17 imports and POSTs to `/api/ingest/text`) | NO (internal app → daemon HTTP call, not a Tauri-exposed signature) | WIRE — stays pub |
| IngestWebpageRequest | requests.rs | NO — origin-server has local shadow at ingest_routes.rs:27; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| LinkEntityRequest | requests.rs | NO — origin-server has local shadow at memory_routes.rs:127; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| ListActivityRequest | requests.rs | NO — origin-types is the only place defining it; no references in server, core, or app | NO | **DEAD — delete candidate** (not referenced anywhere, even within origin-types) |
| ListEntitiesRequest | requests.rs | YES (app/src/search.rs:1399 constructs `requests::ListEntitiesRequest` for Tauri cmd → daemon) | YES (`list_entities_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| ListMemoriesRequest | requests.rs | YES (imported in memory_routes.rs:11 as `Json<ListMemoriesRequest>`) | YES (app/src/search.rs:953 constructs `requests::ListMemoriesRequest` in `list_memories` Tauri cmd; app has its own local shadow input) | WIRE — stays pub |
| ReclassifyMemoryRequest | requests.rs | YES (app/src/search.rs:983 constructs `requests::ReclassifyMemoryRequest`) | YES (`reclassify_memory_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| ReorderSpaceRequest | requests.rs | YES (`Json<origin_types::requests::ReorderSpaceRequest>` at memory_routes.rs:1963) | YES (app/src/search.rs:2058 constructs in `reorder_space`) | WIRE — stays pub |
| SearchConceptsRequest | requests.rs | YES (app/src/search.rs:2469 constructs `requests::SearchConceptsRequest` for `search_concepts` Tauri cmd) | YES | WIRE — stays pub (origin-server has local shadow) |
| SearchEntitiesRequest | requests.rs | YES (app/src/search.rs:1417 constructs `requests::SearchEntitiesRequest`) | YES (`search_entities_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| SearchMemoryRequest | requests.rs | YES (`Json<SearchMemoryRequest>` at memory_routes.rs:714; imported at :11) | YES (app/src/search.rs:864 + app/src/ambient/monitor.rs:18 construct for daemon POST; `search_memory` Tauri cmd) | WIRE — stays pub |
| SearchRequest | requests.rs | YES (app/src/search.rs:848 constructs `requests::SearchRequest` for `search` Tauri cmd → daemon POST `/api/search`) | YES (transitive via `search` Tauri cmd) | WIRE — stays pub (origin-server has local shadow at routes.rs:21) |
| SetDocumentSpaceRequest | requests.rs | YES (`Json<origin_types::requests::SetDocumentSpaceRequest>` at memory_routes.rs:1995) | YES (app/src/search.rs:2088 constructs in `set_document_space`) | WIRE — stays pub |
| SetDocumentTagsRequest | requests.rs | YES (`Json<origin_types::requests::SetDocumentTagsRequest>` at memory_routes.rs:2058) | YES (app/src/search.rs:2181 constructs) | WIRE — stays pub |
| SetStabilityRequest | requests.rs | YES (constructed in app and POSTed for `set_stability_cmd` Tauri cmd) | YES | WIRE — stays pub |
| StoreMemoryRequest | requests.rs | YES (`Json<StoreMemoryRequest>` at memory_routes.rs:215) | YES (app/src/search.rs:886 constructs `requests::StoreMemoryRequest` in `store_memory` Tauri cmd; app has local-shadow Tauri cmd input) | WIRE — stays pub |
| UpdateAgentRequest | requests.rs | YES (app/src/search.rs:1643 constructs `requests::UpdateAgentRequest` for `update_agent` Tauri cmd → daemon) | YES (`update_agent`) | WIRE — stays pub (origin-server has local shadow at memory_routes.rs:70) |
| UpdateChunkRequest | requests.rs | YES (app/src/search.rs:1144 constructs `requests::UpdateChunkRequest` for `update_chunk` Tauri cmd) | YES | WIRE — stays pub |
| UpdateConceptRequest | requests.rs | YES (app/src/search.rs:2394 constructs `requests::UpdateConceptRequest`) | YES (`update_concept`) | WIRE — stays pub |
| UpdateConfigRequest | requests.rs | YES (`Json<UpdateConfigRequest>` in config_routes.rs) | NO | WIRE — stays pub |
| UpdateMemoryRequest | requests.rs | YES (`update_memory_cmd` Tauri cmd constructs `requests::UpdateMemoryRequest`) | YES | WIRE — stays pub |
| UpdateObservationRequest | requests.rs | YES (app/src/search.rs:1446 constructs `requests::UpdateObservationRequest`) | YES (`update_observation_cmd`) | WIRE — stays pub |
| UpdateProfileRequest | requests.rs | YES (app/src/search.rs:1571, 1697, 1760 construct `requests::UpdateProfileRequest` across `update_profile`, `set_avatar`, `remove_avatar`) | YES (`update_profile`, `set_avatar`, `remove_avatar`) | WIRE — stays pub (origin-server has local shadow at memory_routes.rs:38) |
| UpdateSpaceRequest | requests.rs | YES (app/src/search.rs:2018, 2131 construct `requests::UpdateSpaceRequest`) | YES (`update_space`, `rename_space`) | WIRE — stays pub |

### responses.rs

| Type | Origin file | HTTP wire? | Tauri wire? | Verdict |
|---|---|---|---|---|
| ActivityResponse | responses.rs | YES (`{"activities": Vec<AgentActivityRow>}` wire shape for `/api/activities`) | YES (transitive: app deserializes to unwrap `.activities` for `list_agent_activity`) | WIRE — stays pub |
| AddObservationResponse | responses.rs | YES (`Json<origin_types::responses::AddObservationResponse>` in memory_routes.rs:1861; constructed at :1875) | YES (app/src/search.rs:1531 deserializes `responses::AddObservationResponse`) | WIRE — stays pub |
| AgentResponse | responses.rs | YES (app/src/search.rs:1585 deserializes `responses::AgentResponse` from `/api/agents`) | YES (transitive via `list_agents`, `get_agent` Tauri cmds) | WIRE — stays pub (origin-server has local shadow at memory_routes.rs; canonical client-facing def is origin-types) |
| ChatContextResponse | responses.rs | YES (`Json<ChatContextResponse>` at routes.rs:195) | NO | WIRE — stays pub |
| ConfigResponse | responses.rs | YES (imported and returned in config_routes.rs) | NO | WIRE — stays pub |
| ConfirmResponse | responses.rs | YES (imported in memory_routes.rs:14) | YES (app/src/search.rs:915 deserializes `responses::ConfirmResponse` in `confirm_memory`) | WIRE — stays pub |
| ContextResponse | responses.rs | NO — origin-server has local shadow at routes.rs:145; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| ContextSuggestion | responses.rs | NO — origin-server has local shadow at routes.rs:138; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| CreateEntityResponse | responses.rs | YES (app/src/search.rs:1387 deserializes `responses::CreateEntityResponse` after POST) | YES (transitive via `create_entity_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| CreateRelationResponse | responses.rs | NO — origin-server has local shadow at memory_routes.rs:112; origin-types version never imported anywhere | NO | INTERNAL — #[doc(hidden)] candidate |
| DecisionDomainsResponse | responses.rs | YES (JSON on wire; app consumes `responses::DecisionDomainsResponse` at search.rs:2581) | YES (transitive via `list_decision_domains_cmd`) | WIRE — stays pub |
| DecisionsResponse | responses.rs | YES (JSON on wire; app consumes `responses::DecisionsResponse` for `list_decisions_cmd`) | YES (transitive) | WIRE — stays pub |
| DeleteCountResponse | responses.rs | YES (returned by `DELETE /api/memory/bulk` and related endpoints) | YES (transitive) | WIRE — stays pub |
| DeleteResponse | responses.rs | YES (imported in memory_routes.rs:14; used in `/api/memory/delete/{id}`) | YES (app/src/search.rs:969 deserializes `responses::DeleteResponse`) | WIRE — stays pub |
| ExportConceptResponse | responses.rs | YES (app consumes after POST `/api/concepts/{id}/export`) | YES (transitive via `export_concept_to_obsidian`) | WIRE — stays pub |
| HealthResponse | responses.rs | YES (app/src/api.rs:7 imports and consumes from `/api/health`) | NO (internal app use) | WIRE — stays pub (origin-server has local shadow at routes.rs:41; origin-types is the canonical client-facing def) |
| ImportMemoriesResponse | responses.rs | YES (app/src/search.rs:1311 deserializes `responses::ImportMemoriesResponse`) | YES (`import_memories_cmd` Tauri cmd returns this type) | WIRE — stays pub |
| IndexedFilesResponse | responses.rs | YES (app/src/search.rs:1119 deserializes `responses::IndexedFilesResponse` to unwrap `.files`) | YES (transitive) | WIRE — stays pub |
| IngestResponse | responses.rs | YES (app/src/router/intent.rs:18 imports and receives from `/api/ingest/text`) | NO | WIRE — stays pub (origin-server has local shadow at ingest_routes.rs:46; origin-types is the canonical client-facing def) |
| KnowledgeContext | responses.rs | YES (nested in ChatContextResponse) | NO | WIRE — stays pub |
| KnowledgeCountResponse | responses.rs | YES (imported in knowledge_routes.rs:6; app consumes at search.rs:2520) | NO | WIRE — stays pub |
| KnowledgePathResponse | responses.rs | YES (imported in knowledge_routes.rs:6; app consumes at search.rs:2513) | NO | WIRE — stays pub |
| ListEntitiesResponse | responses.rs | YES (app/src/search.rs:1403 deserializes `responses::ListEntitiesResponse`) | YES (transitive via `list_entities_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| ListMemoriesResponse | responses.rs | YES (imported in memory_routes.rs:14) | YES (transitive via list-memories unwrap) | WIRE — stays pub |
| MemoryDetailResponse | responses.rs | YES (app/src/search.rs:1057 deserializes `responses::MemoryDetailResponse`) | YES (transitive via `get_memory_detail`) | WIRE — stays pub |
| MemoryStatsResponse | responses.rs | YES (app/src/search.rs:1066 deserializes `responses::MemoryStatsResponse`) | YES (transitive via `get_memory_stats_cmd`) | WIRE — stays pub (origin-server has local shadow at memory_routes.rs) |
| Nudge | responses.rs | NO — origin-core/refinery.rs:85 has the authoritative local def; origin-types version never referenced outside its own crate | NO | INTERNAL — #[doc(hidden)] candidate (parallel definition in origin-core; origin-types copy is dead today) |
| NurtureCardsResponse | responses.rs | YES (app/src/search.rs:2329 deserializes `responses::NurtureCardsResponse`) | YES (transitive via `get_nurture_cards_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| PhaseResult | responses.rs | NO — origin-server's local SteepResponse uses `origin_core::refinery::PhaseResult`; origin-types version never referenced outside its own crate | NO | INTERNAL — #[doc(hidden)] candidate (parallel definition in origin-core) |
| PinnedMemoriesResponse | responses.rs | YES (app/src/search.rs:1798 deserializes `responses::PinnedMemoriesResponse`) | YES (transitive via `list_pinned_memories`) | WIRE — stays pub |
| ProfileContext | responses.rs | YES (nested in ChatContextResponse) | NO | WIRE — stays pub |
| ProfileResponse | responses.rs | YES (app/src/search.rs:1543 deserializes `responses::ProfileResponse`) | YES (transitive via `get_profile`, `get_avatar_data_url`, `remove_avatar`) | WIRE — stays pub (origin-server has local shadow) |
| ReclassifyMemoryResponse | responses.rs | YES (app/src/search.rs:984 deserializes `responses::ReclassifyMemoryResponse`) | YES (transitive via `reclassify_memory_cmd`) | WIRE — stays pub |
| SearchEntitiesResponse | responses.rs | YES (app/src/search.rs:1421 deserializes `responses::SearchEntitiesResponse`) | YES (transitive via `search_entities_cmd`) | WIRE — stays pub (origin-server has local shadow) |
| SearchMemoryResponse | responses.rs | YES (imported in memory_routes.rs:14; app imports in :18 of ambient/monitor.rs) | YES (app/src/search.rs:871 deserializes) | WIRE — stays pub |
| SearchResponse | responses.rs | YES (app/src/search.rs:854 deserializes `responses::SearchResponse` from `/api/search`) | YES (transitive via `search` Tauri cmd) | WIRE — stays pub (origin-server has local shadow at routes.rs:35) |
| StatusResponse | responses.rs | YES (app/src/search.rs:469 deserializes `responses::StatusResponse`) | NO (internal app use in `get_capture_stats`) | WIRE — stays pub (origin-server has local shadow at routes.rs:47) |
| SteepResponse | responses.rs | NO — origin-server has its own local `SteepResponse` at routes.rs:587 wrapping `origin_core::refinery::PhaseResult`; origin-types version never referenced | NO | INTERNAL — #[doc(hidden)] candidate |
| StoreMemoryResponse | responses.rs | YES (imported in memory_routes.rs:15; `Json<StoreMemoryResponse>` return) | YES (app/src/search.rs `store_memory -> StoreMemoryResponse`; app has local shadow for cmd output too) | WIRE — stays pub |
| SuccessResponse | responses.rs | YES (extensively used by app: app/src/search.rs:1146, :1192, :1463, :1776, etc. deserializes `responses::SuccessResponse` from many endpoints) | YES (transitive via dozens of Tauri cmds) | WIRE — stays pub (origin-server has a local shadow in config_routes.rs but origin-types is the canonical client-facing def) |
| SyncStatsResponse | responses.rs | NO — origin-server has its own `SyncStatsResponse` at source_routes.rs:29 with an extra `error_detail` field; origin-types version never referenced | NO | INTERNAL — #[doc(hidden)] candidate (server shadow has an extra field; reconcile in future D10 pass) |
| TagsResponse | responses.rs | YES (`GET /api/tags` returns this wire shape) | YES (transitive) | WIRE — stays pub |
| TierTokenEstimates | responses.rs | YES (nested in ChatContextResponse) | NO | WIRE — stays pub |
| VersionChainResponse | responses.rs | YES (`GET /api/memory/{id}/versions`; app/src/search.rs:1105 deserializes `responses::VersionChainResponse`) | YES (transitive via `get_version_chain_cmd`) | WIRE — stays pub |

### sources.rs

| Type | Origin file | HTTP wire? | Tauri wire? | Verdict |
|---|---|---|---|---|
| MemoryType | sources.rs | YES (validated at API boundary in memory_routes.rs:17; re-exported via origin_core::sources) | YES (transitive via memory_type string fields on returned types) | WIRE — stays pub |
| RawDocument | sources.rs | YES (imported in memory_routes.rs:17, ingest_routes.rs:8, websocket.rs:10; constructed throughout ingest) | YES (via origin_core::db re-export in app) | WIRE — stays pub |
| SourceType | sources.rs | YES (imported at source_routes.rs:13) | YES (via origin_core::sources re-export in app) | WIRE — stays pub |
| StabilityTier | sources.rs | YES (imported at memory_routes.rs:17; used in confidence calculation) | NO (not directly returned; internal logic type) | WIRE — stays pub |
| SyncStatus | sources.rs | YES (imported at source_routes.rs:13) | YES (via origin_core re-export in app) | WIRE — stays pub |

## Summary

- **WIRE (HTTP and/or TAURI)**: 105 types
- **INTERNAL candidates** (`#[doc(hidden)]` in Task 8): 13 types
- **DEAD** (not referenced anywhere, delete candidates): 1 type

Total classified: 119 (7 entities + 3 import + 17 memory + 43 requests + 44
responses + 5 sources). 105 + 13 + 1 = 119. ✓

### INTERNAL candidates (13)

These types are defined in origin-types but have a parallel local definition in
origin-server or origin-core that is the one actually on the wire. The
origin-types copies are not imported or referenced by any handler, Tauri
command, or client helper today. They remain `pub` at the source level but
should be hidden from rustdoc in Task 8.

From `requests.rs` (6):
1. `AddSourceRequest` (origin-server shadow at source_routes.rs:23)
2. `ContextRequest` (origin-server shadow at routes.rs:130)
3. `CreateConceptRequest` (origin-server shadow at memory_routes.rs:1611)
4. `CreateRelationRequest` (origin-server shadow at memory_routes.rs:103)
5. `IngestWebpageRequest` (origin-server shadow at ingest_routes.rs:27)
6. `LinkEntityRequest` (origin-server shadow at memory_routes.rs:127)

From `responses.rs` (7):
7. `ContextResponse` (origin-server shadow at routes.rs:145)
8. `ContextSuggestion` (origin-server shadow at routes.rs:138)
9. `CreateRelationResponse` (origin-server shadow at memory_routes.rs:112)
10. `Nudge` (origin-core/refinery.rs:85 has authoritative def; origin-types copy dead)
11. `PhaseResult` (origin-core/refinery.rs:198 has authoritative def; origin-types copy dead)
12. `SteepResponse` (origin-server has local at routes.rs:587 using origin-core/refinery::PhaseResult)
13. `SyncStatsResponse` (origin-server local in source_routes.rs:29 has extra `error_detail` field)

### DEAD — delete candidate (1)

- `ListActivityRequest` (requests.rs:412): defined in origin-types but never
  imported, constructed, or referenced anywhere — including within origin-types
  itself. The `/api/activities` endpoint uses query-param destructuring (see
  `app/src/search.rs:1898`) instead of a typed request body. Recommend
  deleting rather than `#[doc(hidden)]` since it serves no purpose.

## Notes on shadowing

A recurring theme in this audit: origin-server and origin-core have local
definitions that duplicate origin-types names. Tasks 5-6 migrated the wire
types in `memory_routes.rs` and the chat-context types in `routes.rs`; many
other shadows remain. D10 of the spec anticipates future migrations — the
audit deliberately keeps WIRE classifications for types where the origin-types
version is actually consumed by *client* code (app/src/), even when
origin-server has a local shadow serving the same endpoint. This matches the
spec's framing: origin-types is the canonical contract that external consumers
(origin-mcp, desktop app) depend on.

The INTERNAL set is therefore small (13) — limited to types with NO consumer
usage of the origin-types variant anywhere. These are true candidates for
`#[doc(hidden)]` because exposing them in rustdoc would mislead readers about
the actual wire surface.

The lone exception is `ListActivityRequest`, which has no analog anywhere —
flagged for deletion.
