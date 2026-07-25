// SPDX-License-Identifier: Apache-2.0
//! Wire-shape freeze tests for the M3 entity_id-carrying wire surface (spec M3 D4).
//!
//! Every struct/field/enum-variant here is listed in the FREEZE set of
//! `docs/plans/2026-07-22-m3-pr1-caller-inventory.md` ("FREEZE set" section):
//! wire/MCP-facing entity_id-carrying surfaces whose shape must never change
//! while `entities`-internal readers flip to page-backed storage behind
//! adapters (D4). Each test asserts `serde_json::to_value(&sample)` against a
//! committed `serde_json::json!({...})` literal -- exact keys and values, so a
//! rename/removal/retype/addition fails the build. Where the struct derives
//! `Deserialize`, the literal is also asserted to deserialize back and
//! re-serialize to the same value (round trip).
//!
//! These are freeze tests, not correctness tests: a surprising shape here
//! (e.g. an `Option` field with no `skip_serializing_if` sitting next to
//! siblings that have one) is frozen as-is, never "fixed". See the stage
//! report for the ones worth a human's attention.
//!
//! All float samples use binary-exact values (0.0, 0.5) rather than decimals
//! like 0.9: `serde_json::to_value` widens an `f32` to `f64` via a raw `as`
//! cast, not a decimal round-trip, so `0.9f32` does not compare equal to
//! `json!(0.9)` (verified empirically -- `0.9f32 as f64 == 0.8999999761581421`).
//!
//! Freezing a sample only proves what it covers: a NEW `Option` field added
//! with `skip_serializing_if` and left `None` in every sample here passes
//! silently without ever freezing its `Some` shape. When adding a field to a
//! frozen struct, populate it (`Some(...)`) in the fully-populated sample, not
//! just the struct literal's default.

use crate::entities::{
    Entity, EntityDetail, EntitySearchResult, EntitySuggestion, Observation, RecentRelation,
    Relation, RelationWithEntity,
};
use crate::memory::{MemoryItem, SearchResult, Space};
use crate::pages::Page;
use crate::repair::{
    RepairChoice, RepairEnrichmentStep, RepairMutation, RepairRollbackPayloadV2, RepairScope,
    RepairTarget,
};
use crate::requests::{
    AddEntityObservationRequest, ConfirmEntityRequest, CreateEntityRequest, LinkEntityRequest,
    StoreMemoryRequest,
};
use crate::responses::{
    CreateEntityResponse, ListEntitiesResponse, ProposalAction, RefinementPayload,
    SearchEntitiesResponse,
};
use crate::sources::RawDocument;
use serde_json::json;
use std::collections::HashMap;

// ===== entities.rs:8-93 =====

fn sample_entity() -> Entity {
    Entity {
        id: "entity_1".into(),
        name: "Alice".into(),
        entity_type: "person".into(),
        space: Some("work".into()),
        source_agent: Some("claude-code".into()),
        confidence: Some(0.5),
        confirmed: true,
        created_at: 1_700_000_000,
        updated_at: 1_700_000_100,
    }
}

fn entity_json() -> serde_json::Value {
    json!({
        "id": "entity_1",
        "name": "Alice",
        "entity_type": "person",
        "space": "work",
        "source_agent": "claude-code",
        "confidence": 0.5,
        "confirmed": true,
        "created_at": 1_700_000_000,
        "updated_at": 1_700_000_100,
    })
}

fn sample_observation() -> Observation {
    Observation {
        id: "obs_1".into(),
        entity_id: "entity_1".into(),
        content: "likes tea".into(),
        source_agent: Some("claude-code".into()),
        confidence: Some(0.5),
        confirmed: true,
        created_at: 1_700_000_000,
    }
}

fn observation_json() -> serde_json::Value {
    json!({
        "id": "obs_1",
        "entity_id": "entity_1",
        "content": "likes tea",
        "source_agent": "claude-code",
        "confidence": 0.5,
        "confirmed": true,
        "created_at": 1_700_000_000,
    })
}

fn sample_relation_with_entity() -> RelationWithEntity {
    RelationWithEntity {
        id: "rel_1".into(),
        relation_type: "knows".into(),
        direction: "outgoing".into(),
        entity_id: "entity_2".into(),
        entity_name: "Bob".into(),
        entity_type: "person".into(),
        source_agent: Some("claude-code".into()),
        created_at: 1_700_000_000,
    }
}

fn relation_with_entity_json() -> serde_json::Value {
    json!({
        "id": "rel_1",
        "relation_type": "knows",
        "direction": "outgoing",
        "entity_id": "entity_2",
        "entity_name": "Bob",
        "entity_type": "person",
        "source_agent": "claude-code",
        "created_at": 1_700_000_000,
    })
}

#[test]
fn entity_and_wrappers_freeze() {
    assert_eq!(
        serde_json::to_value(sample_entity()).unwrap(),
        entity_json()
    );
    let back: Entity = serde_json::from_value(entity_json()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), entity_json());

    let search_result = EntitySearchResult {
        entity: sample_entity(),
        distance: 0.5,
    };
    let search_result_json = json!({ "entity": entity_json(), "distance": 0.5 });
    assert_eq!(
        serde_json::to_value(&search_result).unwrap(),
        search_result_json
    );
    let back: EntitySearchResult = serde_json::from_value(search_result_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), search_result_json);

    let detail = EntityDetail {
        entity: sample_entity(),
        observations: vec![sample_observation()],
        relations: vec![sample_relation_with_entity()],
    };
    let detail_json = json!({
        "entity": entity_json(),
        "observations": [observation_json()],
        "relations": [relation_with_entity_json()],
    });
    assert_eq!(serde_json::to_value(&detail).unwrap(), detail_json);
    let back: EntityDetail = serde_json::from_value(detail_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), detail_json);
}

#[test]
fn observation_freeze() {
    assert_eq!(
        serde_json::to_value(sample_observation()).unwrap(),
        observation_json()
    );
    let back: Observation = serde_json::from_value(observation_json()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), observation_json());
}

#[test]
fn relation_family_freeze() {
    let relation = Relation {
        id: "rel_1".into(),
        from_entity: "entity_1".into(),
        to_entity: "entity_2".into(),
        relation_type: "knows".into(),
        source_agent: Some("claude-code".into()),
        created_at: 1_700_000_000,
    };
    let relation_json = json!({
        "id": "rel_1",
        "from_entity": "entity_1",
        "to_entity": "entity_2",
        "relation_type": "knows",
        "source_agent": "claude-code",
        "created_at": 1_700_000_000,
    });
    assert_eq!(serde_json::to_value(&relation).unwrap(), relation_json);
    let back: Relation = serde_json::from_value(relation_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), relation_json);

    assert_eq!(
        serde_json::to_value(sample_relation_with_entity()).unwrap(),
        relation_with_entity_json()
    );
    let back: RelationWithEntity = serde_json::from_value(relation_with_entity_json()).unwrap();
    assert_eq!(
        serde_json::to_value(&back).unwrap(),
        relation_with_entity_json()
    );

    let recent = RecentRelation {
        id: "rel_1".into(),
        from_entity_id: "entity_1".into(),
        relation_type: "knows".into(),
        to_entity_id: "entity_2".into(),
        from_entity_name: "Alice".into(),
        to_entity_name: "Bob".into(),
        created_at_ms: 1_700_000_000_000,
    };
    let recent_json = json!({
        "id": "rel_1",
        "from_entity_id": "entity_1",
        "relation_type": "knows",
        "to_entity_id": "entity_2",
        "from_entity_name": "Alice",
        "to_entity_name": "Bob",
        "created_at_ms": 1_700_000_000_000i64,
    });
    assert_eq!(serde_json::to_value(&recent).unwrap(), recent_json);
    let back: RecentRelation = serde_json::from_value(recent_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), recent_json);
}

#[test]
fn entity_suggestion_freeze() {
    let some = EntitySuggestion {
        id: "sugg_1".into(),
        entity_name: Some("Alice".into()),
        source_ids: vec!["mem_1".into()],
        confidence: 0.6,
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    let some_json = json!({
        "id": "sugg_1",
        "entity_name": "Alice",
        "source_ids": ["mem_1"],
        "confidence": 0.6,
        "created_at": "2026-01-01T00:00:00Z",
    });
    assert_eq!(serde_json::to_value(&some).unwrap(), some_json);
    let back: EntitySuggestion = serde_json::from_value(some_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), some_json);

    // entity_name has no skip_serializing_if: None still serializes as a null
    // key (freezing today's shape, not the omission pattern used elsewhere).
    let none = EntitySuggestion {
        id: "sugg_2".into(),
        entity_name: None,
        source_ids: vec![],
        confidence: 0.0,
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    let none_json = json!({
        "id": "sugg_2",
        "entity_name": null,
        "source_ids": [],
        "confidence": 0.0,
        "created_at": "2026-01-01T00:00:00Z",
    });
    assert_eq!(serde_json::to_value(&none).unwrap(), none_json);
}

// ===== responses.rs:284/317/322,697-745 =====

#[test]
fn entity_crud_responses_freeze() {
    let with_warnings = CreateEntityResponse {
        id: "entity_1".into(),
        warnings: vec!["low confidence".into()],
    };
    assert_eq!(
        serde_json::to_value(&with_warnings).unwrap(),
        json!({ "id": "entity_1", "warnings": ["low confidence"] })
    );

    // warnings has skip_serializing_if = Vec::is_empty: an empty Vec omits the key.
    let no_warnings = CreateEntityResponse {
        id: "entity_1".into(),
        warnings: vec![],
    };
    assert_eq!(
        serde_json::to_value(&no_warnings).unwrap(),
        json!({ "id": "entity_1" })
    );

    let list = ListEntitiesResponse {
        entities: vec![sample_entity()],
    };
    assert_eq!(
        serde_json::to_value(&list).unwrap(),
        json!({ "entities": [entity_json()] })
    );

    let search = SearchEntitiesResponse {
        results: vec![EntitySearchResult {
            entity: sample_entity(),
            distance: 0.5,
        }],
    };
    assert_eq!(
        serde_json::to_value(&search).unwrap(),
        json!({ "results": [{ "entity": entity_json(), "distance": 0.5 }] })
    );
}

#[test]
fn proposal_action_entity_variants_freeze() {
    assert_eq!(
        serde_json::to_value(&ProposalAction::EntityMerge).unwrap(),
        json!("entity_merge")
    );
    assert_eq!(
        serde_json::from_value::<ProposalAction>(json!("entity_merge")).unwrap(),
        ProposalAction::EntityMerge
    );

    assert_eq!(
        serde_json::to_value(&ProposalAction::SuggestEntity).unwrap(),
        json!("suggest_entity")
    );
    assert_eq!(
        serde_json::from_value::<ProposalAction>(json!("suggest_entity")).unwrap(),
        ProposalAction::SuggestEntity
    );
}

#[test]
fn refinement_payload_entity_variants_freeze() {
    let merge = RefinementPayload::EntityMerge {
        existing_id: "entity_1".into(),
        new_id: "entity_2".into(),
        similarity: 0.9,
    };
    let merge_json = json!({
        "action": "entity_merge",
        "existing_id": "entity_1",
        "new_id": "entity_2",
        "similarity": 0.9,
    });
    assert_eq!(serde_json::to_value(&merge).unwrap(), merge_json);
    assert_eq!(
        serde_json::from_value::<RefinementPayload>(merge_json).unwrap(),
        merge
    );

    let suggest_some = RefinementPayload::SuggestEntity {
        name_hint: Some("Alice".into()),
    };
    let suggest_some_json = json!({ "action": "suggest_entity", "name_hint": "Alice" });
    assert_eq!(
        serde_json::to_value(&suggest_some).unwrap(),
        suggest_some_json
    );
    assert_eq!(
        serde_json::from_value::<RefinementPayload>(suggest_some_json).unwrap(),
        suggest_some
    );

    // name_hint has skip_serializing_if = Option::is_none: None omits the key.
    let suggest_none = RefinementPayload::SuggestEntity { name_hint: None };
    let suggest_none_json = json!({ "action": "suggest_entity" });
    assert_eq!(
        serde_json::to_value(&suggest_none).unwrap(),
        suggest_none_json
    );
    assert_eq!(
        serde_json::from_value::<RefinementPayload>(suggest_none_json).unwrap(),
        suggest_none
    );
}

// ===== requests.rs:29,136,175-177,564,570 =====

#[test]
fn store_memory_request_entity_id_freeze() {
    let some = StoreMemoryRequest {
        content: "note".into(),
        memory_type: Some("fact".into()),
        space: Some("work".into()),
        source_agent: Some("claude-code".into()),
        title: Some("Title".into()),
        confidence: Some(0.5),
        supersedes: None,
        entity: Some("Alice".into()),
        entity_id: Some("entity_1".into()),
        structured_fields: None,
        retrieval_cue: None,
    };
    let some_json = json!({
        "content": "note",
        "memory_type": "fact",
        "space": "work",
        "source_agent": "claude-code",
        "title": "Title",
        "confidence": 0.5,
        "supersedes": null,
        "entity": "Alice",
        "entity_id": "entity_1",
        "structured_fields": null,
        "retrieval_cue": null,
    });
    assert_eq!(serde_json::to_value(&some).unwrap(), some_json);
    let back: StoreMemoryRequest = serde_json::from_value(some_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), some_json);

    // entity_id has no skip_serializing_if: None still serializes as a null key.
    let none = StoreMemoryRequest {
        content: "note".into(),
        memory_type: None,
        space: None,
        source_agent: None,
        title: None,
        confidence: None,
        supersedes: None,
        entity: None,
        entity_id: None,
        structured_fields: None,
        retrieval_cue: None,
    };
    let none_json = json!({
        "content": "note",
        "memory_type": null,
        "space": null,
        "source_agent": null,
        "title": null,
        "confidence": null,
        "supersedes": null,
        "entity": null,
        "entity_id": null,
        "structured_fields": null,
        "retrieval_cue": null,
    });
    assert_eq!(serde_json::to_value(&none).unwrap(), none_json);
}

#[test]
fn create_and_link_entity_request_freeze() {
    let create = CreateEntityRequest {
        name: "Alice".into(),
        entity_type: "person".into(),
        space: Some("work".into()),
        source_agent: Some("claude-code".into()),
        confidence: Some(0.5),
    };
    let create_json = json!({
        "name": "Alice",
        "entity_type": "person",
        "space": "work",
        "source_agent": "claude-code",
        "confidence": 0.5,
    });
    assert_eq!(serde_json::to_value(&create).unwrap(), create_json);
    let back: CreateEntityRequest = serde_json::from_value(create_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), create_json);

    let link = LinkEntityRequest {
        source_id: "mem_1".into(),
        entity_id: "entity_1".into(),
    };
    let link_json = json!({ "source_id": "mem_1", "entity_id": "entity_1" });
    assert_eq!(serde_json::to_value(&link).unwrap(), link_json);
    let back: LinkEntityRequest = serde_json::from_value(link_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), link_json);
}

#[test]
fn confirm_and_observation_request_freeze() {
    let confirm = ConfirmEntityRequest { confirmed: true };
    let confirm_json = json!({ "confirmed": true });
    assert_eq!(serde_json::to_value(&confirm).unwrap(), confirm_json);
    let back: ConfirmEntityRequest = serde_json::from_value(confirm_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), confirm_json);

    let obs = AddEntityObservationRequest {
        content: "likes tea".into(),
        source_agent: Some("claude-code".into()),
        confidence: Some(0.5),
    };
    let obs_json = json!({
        "content": "likes tea",
        "source_agent": "claude-code",
        "confidence": 0.5,
    });
    assert_eq!(serde_json::to_value(&obs).unwrap(), obs_json);
    let back: AddEntityObservationRequest = serde_json::from_value(obs_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), obs_json);
}

// ===== pages.rs:15 =====

fn sample_page(entity_id: Option<String>) -> Page {
    Page {
        id: "page_1".into(),
        title: "Title".into(),
        summary: Some("Summary".into()),
        content: "Body".into(),
        entity_id,
        space: Some("work".into()),
        source_memory_ids: vec!["mem_1".into()],
        version: 1,
        status: "active".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        last_compiled: "2026-01-01T00:00:00Z".into(),
        last_modified: "2026-01-01T00:00:00Z".into(),
        sources_updated_count: 0,
        stale_reason: None,
        pending_rebuild: None,
        user_edited: false,
        relevance_score: 0.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: None,
        changelog: None,
        workspace: None,
        creation_kind: "distilled".into(),
        review_status: "confirmed".into(),
        citations: vec![],
        kind: "concept".into(),
    }
}

/// `Page.entity_id` (spec M3, FREEZE set). `kind`'s permanent wire absence is
/// already tested at `pages.rs`'s `kind_is_never_serialized_onto_the_wire` --
/// not duplicated here.
#[test]
fn page_entity_id_freeze() {
    let some_json = json!({
        "id": "page_1",
        "title": "Title",
        "summary": "Summary",
        "content": "Body",
        "entity_id": "entity_1",
        "space": "work",
        "source_memory_ids": ["mem_1"],
        "version": 1,
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "last_compiled": "2026-01-01T00:00:00Z",
        "last_modified": "2026-01-01T00:00:00Z",
        "sources_updated_count": 0,
        "stale_reason": null,
        "user_edited": false,
        "workspace": null,
        "creation_kind": "distilled",
        "review_status": "confirmed",
    });
    assert_eq!(
        serde_json::to_value(sample_page(Some("entity_1".into()))).unwrap(),
        some_json
    );
    let back: Page = serde_json::from_value(some_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), some_json);

    // entity_id has no skip_serializing_if: None still serializes as a null key.
    let mut none_json = some_json;
    none_json["entity_id"] = serde_json::Value::Null;
    assert_eq!(serde_json::to_value(sample_page(None)).unwrap(), none_json);
}

// ===== memory.rs:41-43,102,356 =====

#[allow(clippy::too_many_arguments)]
fn sample_search_result(entity_id: Option<String>, entity_name: Option<String>) -> SearchResult {
    SearchResult {
        id: "sr_1".into(),
        content: "content".into(),
        source: "memory".into(),
        source_id: "mem_1".into(),
        title: "Title".into(),
        url: None,
        chunk_index: 0,
        last_modified: 1_700_000_000,
        score: 0.5,
        chunk_type: None,
        language: None,
        semantic_unit: None,
        memory_type: None,
        space: None,
        source_agent: None,
        confidence: None,
        confirmed: None,
        stability: None,
        supersedes: None,
        summary: None,
        entity_id,
        entity_name,
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
    }
}

/// `SearchResult.entity_id`/`.entity_name` (spec M3, FREEZE set). The existing
/// `search_result_serializes` test (lib.rs) only checks omission by substring
/// for the None case; this freezes the exact shape for both Some and None.
#[test]
fn search_result_entity_fields_freeze() {
    let some_json = json!({
        "id": "sr_1",
        "content": "content",
        "source": "memory",
        "source_id": "mem_1",
        "title": "Title",
        "url": null,
        "chunk_index": 0,
        "last_modified": 1_700_000_000,
        "score": 0.5,
        "entity_id": "entity_1",
        "entity_name": "Alice",
        "is_archived": false,
        "is_recap": false,
        "raw_score": 0.0,
        "version": 0,
        "pending_revision": false,
    });
    assert_eq!(
        serde_json::to_value(sample_search_result(
            Some("entity_1".into()),
            Some("Alice".into())
        ))
        .unwrap(),
        some_json
    );
    let back: SearchResult = serde_json::from_value(some_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), some_json);

    // entity_id/entity_name both have skip_serializing_if = Option::is_none.
    let mut none_json = some_json;
    let none_obj = none_json.as_object_mut().unwrap();
    none_obj.remove("entity_id");
    none_obj.remove("entity_name");
    assert_eq!(
        serde_json::to_value(sample_search_result(None, None)).unwrap(),
        none_json
    );
}

fn sample_memory_item(entity_id: Option<String>) -> MemoryItem {
    MemoryItem {
        source_id: "mem_1".into(),
        title: "Title".into(),
        content: "Body".into(),
        summary: None,
        memory_type: Some("fact".into()),
        space: Some("work".into()),
        source_agent: Some("claude-code".into()),
        confidence: Some(0.5),
        confirmed: true,
        stability: None,
        pinned: false,
        supersedes: None,
        last_modified: 1_700_000_000,
        chunk_count: 1,
        entity_id,
        quality: None,
        is_recap: false,
        enrichment_status: "complete".into(),
        supersede_mode: "hide".into(),
        structured_fields: None,
        retrieval_cue: None,
        access_count: 0,
        source_text: None,
        version: 1,
        changelog: None,
        pending_revision: false,
        merged_from: None,
    }
}

/// `MemoryItem.entity_id` and `Space.entity_count` (spec M3, FREEZE set).
#[test]
fn memory_item_entity_id_and_space_entity_count_freeze() {
    let some_json = json!({
        "source_id": "mem_1",
        "title": "Title",
        "content": "Body",
        "summary": null,
        "memory_type": "fact",
        "space": "work",
        "source_agent": "claude-code",
        "confidence": 0.5,
        "confirmed": true,
        "pinned": false,
        "supersedes": null,
        "last_modified": 1_700_000_000,
        "chunk_count": 1,
        "entity_id": "entity_1",
        "is_recap": false,
        "enrichment_status": "complete",
        "supersede_mode": "hide",
        "access_count": 0,
        "version": 1,
        "pending_revision": false,
    });
    assert_eq!(
        serde_json::to_value(sample_memory_item(Some("entity_1".into()))).unwrap(),
        some_json
    );
    let back: MemoryItem = serde_json::from_value(some_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), some_json);

    // entity_id has skip_serializing_if = Option::is_none: None omits the key.
    let mut none_json = some_json;
    none_json.as_object_mut().unwrap().remove("entity_id");
    assert_eq!(
        serde_json::to_value(sample_memory_item(None)).unwrap(),
        none_json
    );

    let space = Space {
        id: "space_1".into(),
        name: "Work".into(),
        description: Some("Work space".into()),
        suggested: false,
        starred: true,
        sort_order: 0,
        memory_count: 10,
        entity_count: 3,
        created_at: 1_700_000_000.0,
        updated_at: 1_700_000_100.0,
    };
    let space_json = json!({
        "id": "space_1",
        "name": "Work",
        "description": "Work space",
        "suggested": false,
        "starred": true,
        "sort_order": 0,
        "memory_count": 10,
        "entity_count": 3,
        "created_at": 1_700_000_000.0,
        "updated_at": 1_700_000_100.0,
    });
    assert_eq!(serde_json::to_value(&space).unwrap(), space_json);
    let back: Space = serde_json::from_value(space_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), space_json);
}

// ===== repair.rs:1684-1731,2338,2353,3871-3894,294-327 =====
//
// RepairTarget/RepairMutation/RepairChoice/RepairRollbackPayloadV2 derive
// Serialize only; Deserialize is a hand-written impl that routes through a
// private `*Wire` enum + the same smart constructors used here, validating
// as it goes (see e.g. `impl<'de> Deserialize<'de> for RepairTarget`). The
// wire shape under test is identical either way -- these are the same public
// types the daemon/MCP layer actually serializes and deserializes.

#[test]
fn repair_target_entity_variants_freeze() {
    let link =
        RepairTarget::memory_entity_link("mem_1".into(), "entity_1".into(), RepairScope::global())
            .unwrap();
    let link_json = json!({
        "kind": "memory_entity_link",
        "memory_id": "mem_1",
        "entity_id": "entity_1",
        "scope": { "kind": "global" },
    });
    assert_eq!(serde_json::to_value(&link).unwrap(), link_json);
    assert_eq!(
        serde_json::from_value::<RepairTarget>(link_json).unwrap(),
        link
    );

    let extraction = RepairTarget::memory_entity_extraction(
        "mem_1".into(),
        RepairEnrichmentStep::EntityExtract,
        vec!["entity_1".into(), "entity_2".into()],
        RepairScope::global(),
    )
    .unwrap();
    let extraction_json = json!({
        "kind": "memory_entity_extraction",
        "memory_id": "mem_1",
        "step": "entity_extract",
        "entity_ids": ["entity_1", "entity_2"],
        "scope": { "kind": "global" },
    });
    assert_eq!(serde_json::to_value(&extraction).unwrap(), extraction_json);
    assert_eq!(
        serde_json::from_value::<RepairTarget>(extraction_json).unwrap(),
        extraction
    );
}

#[test]
fn repair_mutation_entity_variants_freeze() {
    let complete =
        RepairMutation::complete_entity_extraction(vec!["entity_1".into(), "entity_2".into()])
            .unwrap();
    let complete_json = json!({
        "kind": "complete_entity_extraction",
        "entity_ids": ["entity_1", "entity_2"],
    });
    assert_eq!(serde_json::to_value(&complete).unwrap(), complete_json);
    assert_eq!(
        serde_json::from_value::<RepairMutation>(complete_json).unwrap(),
        complete
    );

    let delete = RepairMutation::delete_memory_entity_link("mem_1", "entity_1").unwrap();
    let delete_json = json!({
        "kind": "delete_memory_entity_link",
        "memory_id": "mem_1",
        "entity_id": "entity_1",
    });
    assert_eq!(serde_json::to_value(&delete).unwrap(), delete_json);
    assert_eq!(
        serde_json::from_value::<RepairMutation>(delete_json).unwrap(),
        delete
    );
}

#[test]
fn repair_choice_complete_entity_extraction_freeze() {
    let choice = RepairChoice::complete_entity_extraction(
        "review_1".into(),
        "mem_1".into(),
        vec!["entity_1".into(), "entity_2".into()],
    )
    .unwrap();
    let choice_json = json!({
        "kind": "complete_entity_extraction",
        "review_id": "review_1",
        "memory_id": "mem_1",
        "entity_ids": ["entity_1", "entity_2"],
    });
    assert_eq!(serde_json::to_value(&choice).unwrap(), choice_json);
    assert_eq!(
        serde_json::from_value::<RepairChoice>(choice_json).unwrap(),
        choice
    );
}

#[test]
fn repair_rollback_v2_complete_entity_extraction_freeze() {
    let with_error = RepairRollbackPayloadV2::complete_entity_extraction(
        "mem_1".into(),
        vec!["id".into(), "status".into()],
        vec!["mem_1".into(), "raw".into()],
        vec!["entity_1".into(), "entity_2".into()],
        "raw".into(),
        Some("boom".into()),
        1,
        1_700_000_000,
    )
    .unwrap();
    let with_error_json = json!({
        "kind": "complete_entity_extraction",
        "memory_id": "mem_1",
        "memory_columns": ["id", "status"],
        "before_memory_row": ["mem_1", "raw"],
        "before_entity_ids": ["entity_1", "entity_2"],
        "enrichment_status": "raw",
        "enrichment_error": "boom",
        "enrichment_attempts": 1,
        "enrichment_updated_at": 1_700_000_000i64,
    });
    assert_eq!(serde_json::to_value(&with_error).unwrap(), with_error_json);
    assert_eq!(
        serde_json::from_value::<RepairRollbackPayloadV2>(with_error_json).unwrap(),
        with_error
    );

    // enrichment_error has skip_serializing_if = Option::is_none: None omits the key.
    let no_error = RepairRollbackPayloadV2::complete_entity_extraction(
        "mem_1".into(),
        vec!["id".into(), "status".into()],
        vec!["mem_1".into(), "raw".into()],
        vec!["entity_1".into(), "entity_2".into()],
        "raw".into(),
        None,
        1,
        1_700_000_000,
    )
    .unwrap();
    let no_error_json = json!({
        "kind": "complete_entity_extraction",
        "memory_id": "mem_1",
        "memory_columns": ["id", "status"],
        "before_memory_row": ["mem_1", "raw"],
        "before_entity_ids": ["entity_1", "entity_2"],
        "enrichment_status": "raw",
        "enrichment_attempts": 1,
        "enrichment_updated_at": 1_700_000_000i64,
    });
    assert_eq!(serde_json::to_value(&no_error).unwrap(), no_error_json);
    assert_eq!(
        serde_json::from_value::<RepairRollbackPayloadV2>(no_error_json).unwrap(),
        no_error
    );
}

// ===== sources.rs:172 =====

fn sample_raw_document(entity_id: Option<String>) -> RawDocument {
    RawDocument {
        source: "gmail".into(),
        source_id: "msg_1".into(),
        title: "Title".into(),
        summary: Some("Summary".into()),
        content: "Body".into(),
        url: Some("https://example.com".into()),
        last_modified: 1_700_000_000,
        metadata: HashMap::from([("key".to_string(), "value".to_string())]),
        memory_type: Some("fact".into()),
        space: Some("work".into()),
        source_agent: Some("claude-code".into()),
        confidence: Some(0.5),
        confirmed: Some(true),
        stability: Some("confirmed".into()),
        supersedes: Some("mem_0".into()),
        pending_revision: true,
        entity_id,
        quality: Some("high".into()),
        importance: Some(8),
        is_recap: true,
        enrichment_status: "complete".into(),
        supersede_mode: "archive".into(),
        structured_fields: Some("{\"claim\":\"x\"}".into()),
        retrieval_cue: Some("cue".into()),
        source_text: Some("original".into()),
        content_hash: Some("abc123".into()),
    }
}

/// `RawDocument.entity_id` (spec M3, FREEZE set).
#[test]
fn raw_document_entity_id_freeze() {
    let some_json = json!({
        "source": "gmail",
        "source_id": "msg_1",
        "title": "Title",
        "summary": "Summary",
        "content": "Body",
        "url": "https://example.com",
        "last_modified": 1_700_000_000,
        "metadata": { "key": "value" },
        "memory_type": "fact",
        "space": "work",
        "source_agent": "claude-code",
        "confidence": 0.5,
        "confirmed": true,
        "stability": "confirmed",
        "supersedes": "mem_0",
        "pending_revision": true,
        "entity_id": "entity_1",
        "quality": "high",
        "importance": 8,
        "is_recap": true,
        "enrichment_status": "complete",
        "supersede_mode": "archive",
        "structured_fields": "{\"claim\":\"x\"}",
        "retrieval_cue": "cue",
        "source_text": "original",
        "content_hash": "abc123",
    });
    assert_eq!(
        serde_json::to_value(sample_raw_document(Some("entity_1".into()))).unwrap(),
        some_json
    );
    let back: RawDocument = serde_json::from_value(some_json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&back).unwrap(), some_json);

    // entity_id has skip_serializing_if = Option::is_none: None omits the key.
    let mut none_json = some_json;
    none_json.as_object_mut().unwrap().remove("entity_id");
    assert_eq!(
        serde_json::to_value(sample_raw_document(None)).unwrap(),
        none_json
    );
}
