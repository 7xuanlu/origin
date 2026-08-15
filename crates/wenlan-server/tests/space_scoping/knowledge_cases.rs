// SPDX-License-Identifier: Apache-2.0

use super::case_runner::{
    assert_wave_4_knowledge_catalog_contract, assert_wave_4_knowledge_executed_keys,
};
use super::fixture::ScopeFixture;
use axum::body::{to_bytes, Body};
use axum::http::{Method as HttpMethod, Response, StatusCode};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use wenlan_server::sensitive_read_routes::Method;

async fn body_bytes(response: Response<Body>) -> (StatusCode, Vec<u8>) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, bytes.to_vec())
}

async fn json_body(response: Response<Body>) -> (StatusCode, Value) {
    let (status, bytes) = body_bytes(response).await;
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn seed_entity(fixture: &ScopeFixture, name: &str, space: Option<&str>) -> String {
    fixture
        .db
        .store_entity(name, "topic", space, Some("scope-test"), Some(0.9))
        .await
        .unwrap()
}

fn entity_ids(body: &Value, key: &str) -> BTreeSet<String> {
    body[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array: {body}"))
        .iter()
        .filter_map(|row| row["id"].as_str().or_else(|| row["entity"]["id"].as_str()))
        .map(str::to_string)
        .collect()
}

pub async fn unknown_selectors_are_rejected() {
    let fixture = ScopeFixture::new().await;
    let entity = seed_entity(&fixture, "Unknown selector entity", Some("work")).await;
    let probes = [
        (
            HttpMethod::POST,
            "/api/memory/entities/list".to_string(),
            Some(json!({"space":"missing-space"})),
            "/api/memory/entities/list",
            Method::Post,
        ),
        (
            HttpMethod::POST,
            "/api/memory/entities/search".to_string(),
            Some(json!({"query":"entity","space":"missing-space"})),
            "/api/memory/entities/search",
            Method::Post,
        ),
        (
            HttpMethod::GET,
            format!("/api/memory/entities/{entity}"),
            None,
            "/api/memory/entities/{entity_id}",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            "/api/memory/graph".to_string(),
            None,
            "/api/memory/graph",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            "/api/memory/entity-suggestions".to_string(),
            None,
            "/api/memory/entity-suggestions",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            "/api/knowledge/recent-relations".to_string(),
            None,
            "/api/knowledge/recent-relations",
            Method::Get,
        ),
    ];

    let mut executed = Vec::new();
    for (method, uri, body, path, catalog_method) in probes {
        let response = fixture
            .send(method, &uri, body, Some("missing-space"))
            .await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY, "{uri}");
        executed.push((catalog_method, path));
    }
    assert_wave_4_knowledge_executed_keys(executed);
}

pub async fn entity_collections_and_search_are_scoped() {
    let fixture = ScopeFixture::new().await;
    let work_shared = seed_entity(&fixture, "Shared scope name", Some("work")).await;
    let personal_shared = seed_entity(&fixture, "Shared scope name", Some("personal")).await;
    let null_entity = seed_entity(&fixture, "Null scope entity", None).await;
    let literal_uncategorized = seed_entity(
        &fixture,
        "Literal uncategorized entity",
        Some("uncategorized"),
    )
    .await;
    let distant_work = seed_entity(&fixture, "Distant cedar manual", Some("work")).await;
    for index in 0..8 {
        seed_entity(
            &fixture,
            &format!("Quasar nebula personal {index}"),
            Some("personal"),
        )
        .await;
    }

    let (status, list) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/memory/entities/list",
                Some(json!({"space":"work"})),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{list}");
    let listed = entity_ids(&list, "entities");
    assert!(listed.contains(&work_shared));
    assert!(listed.contains(&distant_work));
    assert!(!listed.contains(&personal_shared));

    let (status, search) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/memory/entities/search",
                Some(json!({"query":"Quasar nebula","limit":1,"space":"work"})),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{search}");
    assert_eq!(
        entity_ids(&search, "results"),
        BTreeSet::from([distant_work])
    );

    let (status, uncategorized) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/memory/entities/list",
                Some(json!({"space":"uncategorized"})),
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{uncategorized}");
    assert_eq!(
        entity_ids(&uncategorized, "entities"),
        BTreeSet::from([null_entity])
    );

    let (status, global) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/memory/entities/list",
                Some(json!({})),
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{global}");
    assert!(entity_ids(&global, "entities").contains(&literal_uncategorized));
}

pub async fn detail_and_relation_endpoints_are_scoped() {
    let fixture = ScopeFixture::new().await;
    let work = seed_entity(&fixture, "Work anchor", Some("work")).await;
    let work_peer = seed_entity(&fixture, "Work peer", Some("work")).await;
    let personal = seed_entity(&fixture, "Personal peer", Some("personal")).await;
    fixture
        .db
        .create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    fixture
        .db
        .create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();
    // G6 Stage 1.2 wire-visible id swap (spec Trap 2): the wire `id` is now
    // the active edge's content-addressed edge_id, not the `relations` row
    // uuid `create_relation` returns.
    let work_relation = wenlan_core::provenance::compute_edge_id(
        "relates",
        "entity",
        &work,
        "entity",
        &work_peer,
        "related_to",
    );
    let mixed_relation = wenlan_core::provenance::compute_edge_id(
        "relates",
        "entity",
        &work,
        "entity",
        &personal,
        "related_to",
    );

    let (status, detail) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/memory/entities/{work}"),
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{detail}");
    let detail_relations = detail["relations"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert!(detail_relations.contains(work_relation.as_str()));
    assert!(!detail_relations.contains(mixed_relation.as_str()));

    let mismatch = body_bytes(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/memory/entities/{personal}"),
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    let missing = body_bytes(
        fixture
            .send(
                HttpMethod::GET,
                "/api/memory/entities/missing-entity",
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(mismatch.0, StatusCode::NOT_FOUND);
    assert_eq!(mismatch, missing);
    assert_eq!(mismatch.1, br#"{"error":"entity not found"}"#);

    let global_missing = body_bytes(
        fixture
            .send(
                HttpMethod::GET,
                "/api/memory/entities/missing-entity",
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(global_missing, missing);

    let (status, graph) = json_body(
        fixture
            .send(HttpMethod::GET, "/api/memory/graph", None, Some("work"))
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{graph}");
    let graph_entities = entity_ids(&graph, "entities");
    assert!(graph_entities.contains(&work));
    assert!(graph_entities.contains(&work_peer));
    assert!(!graph_entities.contains(&personal));
    let graph_relations = graph["relations"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        graph_relations,
        BTreeSet::from([work_relation.as_str()]),
        "the cross-space relation must not survive a work-scoped graph read"
    );

    let (status, global_graph) = json_body(
        fixture
            .send(HttpMethod::GET, "/api/memory/graph", None, None)
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{global_graph}");
    let global_graph_relations = global_graph["relations"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert!(global_graph_relations.contains(work_relation.as_str()));
    assert!(global_graph_relations.contains(mixed_relation.as_str()));

    let (status, selected) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/knowledge/recent-relations?limit=20",
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{selected}");
    let selected_ids = selected
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(selected_ids, BTreeSet::from([work_relation.as_str()]));

    let (status, global) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/knowledge/recent-relations?limit=20",
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{global}");
    let global_ids = global
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert!(global_ids.contains(work_relation.as_str()));
    assert!(global_ids.contains(mixed_relation.as_str()));
}

pub async fn suggestions_require_all_sources_in_scope() {
    let fixture = ScopeFixture::new().await;
    fixture
        .seed_record(
            "memory",
            "work-source",
            Some("work"),
            "fact",
            1,
            None,
            false,
        )
        .await;
    fixture
        .seed_record(
            "memory",
            "personal-source",
            Some("personal"),
            "fact",
            2,
            None,
            false,
        )
        .await;
    for (id, sources) in [
        ("work-only", vec!["work-source".to_string()]),
        (
            "mixed",
            vec!["work-source".to_string(), "personal-source".to_string()],
        ),
        ("missing", vec!["missing-source".to_string()]),
        ("empty", Vec::new()),
    ] {
        fixture
            .db
            .insert_refinement_proposal(id, "suggest_entity", &sources, Some(id), 0.9)
            .await
            .unwrap();
    }

    let (status, selected) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/memory/entity-suggestions",
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{selected}");
    let ids = selected
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(ids, BTreeSet::from(["work-only"]));

    let (status, global) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/memory/entity-suggestions",
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{global}");
    let global_ids = global
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        global_ids,
        BTreeSet::from(["empty", "missing", "mixed", "work-only"])
    );
}

pub fn registry_matches_completed_contracts() {
    assert_wave_4_knowledge_catalog_contract();
}
