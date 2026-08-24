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

async fn seed_page(
    fixture: &ScopeFixture,
    title: &str,
    content: &str,
    space: Option<&str>,
) -> String {
    let request = wenlan_types::requests::CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: None,
        entity_id: None,
        space: (space.map(str::to_string)).into(),
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: space.map(str::to_string),
    };
    let created = wenlan_core::post_write::create_page(&fixture.db, request, "scope-test", None)
        .await
        .expect("page fixture must be created through PageWrite");
    fixture
        .db
        .set_page_review_status(&created.id, "confirmed")
        .await
        .unwrap();
    created.id
}

fn string_set(body: &Value, key: &str, field: &str) -> BTreeSet<String> {
    body[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array: {body}"))
        .iter()
        .filter_map(|row| row[field].as_str())
        .map(str::to_string)
        .collect()
}

fn page_link_triples(body: &Value) -> BTreeSet<(String, String, String)> {
    body["page_links"]
        .as_array()
        .unwrap_or_else(|| panic!("page_links must be an array: {body}"))
        .iter()
        .map(|row| {
            (
                format!(
                    "{}:{}",
                    row["from"]["kind"].as_str().unwrap(),
                    row["from"]["id"].as_str().unwrap()
                ),
                format!(
                    "{}:{}",
                    row["to"]["kind"].as_str().unwrap(),
                    row["to"]["id"].as_str().unwrap()
                ),
                row["link_type"].as_str().unwrap().to_string(),
            )
        })
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

    // Wiki pages ride the same scoped graph read: a work page linking to
    // another work page, and a personal page that must not cross over.
    let work_target = seed_page(&fixture, "Scoped graph target", "target body", Some("work")).await;
    let work_source = seed_page(
        &fixture,
        "Scoped graph source",
        "links to [[Scoped graph target]]",
        Some("work"),
    )
    .await;
    let personal_page = seed_page(
        &fixture,
        "Scoped graph personal",
        "personal body",
        Some("personal"),
    )
    .await;

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
    let graph_pages = string_set(&graph, "pages", "id");
    assert_eq!(
        graph_pages,
        BTreeSet::from([work_target.clone(), work_source.clone()]),
        "a work-scoped graph read carries the work wiki pages and nothing else"
    );
    assert_eq!(
        page_link_triples(&graph),
        BTreeSet::from([(
            format!("page:{work_source}"),
            format!("page:{work_target}"),
            "wikilink".to_string()
        )]),
        "the resolved wikilink between two in-scope pages is on the map"
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
    let global_graph_pages = string_set(&global_graph, "pages", "id");
    assert!(global_graph_pages.contains(&personal_page));
    assert!(global_graph_pages.contains(&work_source));
}

/// #576: the five id-addressed entity write routes (confirm, delete,
/// observations, merge, aliases) are scoped like the reads above -- an id
/// outside the request space is indistinguishable from a missing one, an
/// unknown space is a 422, and the in-scope (or header-less) request still
/// succeeds. These routes stay in `NON_SENSITIVE_PATHS` (not the wave_4
/// catalog): the oracle here proves the scope directly against the core
/// write instead.
pub async fn entity_writes_are_scoped() {
    let fixture = ScopeFixture::new().await;

    // ---- confirm: mismatch == missing, unknown space is 422, in-scope OKs ----
    let work_confirm = seed_entity(&fixture, "Write Scope Confirm", Some("work")).await;
    let confirm_uri = format!("/api/memory/entities/{work_confirm}/confirm");
    let confirm_body = json!({"confirmed": true});
    let mismatch = body_bytes(
        fixture
            .send(
                HttpMethod::PUT,
                &confirm_uri,
                Some(confirm_body.clone()),
                Some("personal"),
            )
            .await,
    )
    .await;
    let missing = body_bytes(
        fixture
            .send(
                HttpMethod::PUT,
                "/api/memory/entities/missing-entity/confirm",
                Some(confirm_body.clone()),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(mismatch.0, StatusCode::NOT_FOUND);
    assert_eq!(mismatch, missing);
    assert_eq!(mismatch.1, br#"{"error":"entity not found"}"#);
    let (status, _) = json_body(
        fixture
            .send(
                HttpMethod::PUT,
                &confirm_uri,
                Some(confirm_body.clone()),
                Some("missing-space"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, confirmed) = json_body(
        fixture
            .send(
                HttpMethod::PUT,
                &confirm_uri,
                Some(confirm_body),
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{confirmed}");

    // ---- observations: keeps its 422 "does not exist" shape (id-bearing) ----
    let work_obs = seed_entity(&fixture, "Write Scope Observation", Some("work")).await;
    let obs_uri = format!("/api/memory/entities/{work_obs}/observations");
    let obs_body = json!({"content": "An observation on the scoped entity."});
    let (status, error) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                &obs_uri,
                Some(obs_body.clone()),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{error}");
    assert!(error["error"].as_str().unwrap().contains("does not exist"));
    let (status, _) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                &obs_uri,
                Some(obs_body.clone()),
                Some("missing-space"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, added) = json_body(
        fixture
            .send(HttpMethod::POST, &obs_uri, Some(obs_body), Some("work"))
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{added}");

    // ---- aliases: mismatch == missing, unknown space is 422, in-scope OKs ----
    let work_alias = seed_entity(&fixture, "Write Scope Alias", Some("work")).await;
    let alias_uri = format!("/api/memory/entities/{work_alias}/aliases");
    let alias_body = json!({"alias": "Scoped Alias"});
    let alias_mismatch = body_bytes(
        fixture
            .send(
                HttpMethod::POST,
                &alias_uri,
                Some(alias_body.clone()),
                Some("personal"),
            )
            .await,
    )
    .await;
    let alias_missing = body_bytes(
        fixture
            .send(
                HttpMethod::POST,
                "/api/memory/entities/missing-entity/aliases",
                Some(alias_body.clone()),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(alias_mismatch.0, StatusCode::NOT_FOUND);
    assert_eq!(alias_mismatch, alias_missing);
    assert_eq!(alias_mismatch.1, br#"{"error":"entity not found"}"#);
    let (status, _) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                &alias_uri,
                Some(alias_body.clone()),
                Some("missing-space"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, aliases) = json_body(
        fixture
            .send(HttpMethod::POST, &alias_uri, Some(alias_body), Some("work"))
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{aliases}");

    // ---- merge: dry-run and apply, either id out of scope, either direction --
    // 404 and both entities left untouched; unknown space is 422; in-scope OKs.
    let work_canon = seed_entity(&fixture, "Write Scope Canon", Some("work")).await;
    let personal_loser = seed_entity(&fixture, "Write Scope Loser", Some("personal")).await;
    let personal_canon =
        seed_entity(&fixture, "Write Scope Personal Canon", Some("personal")).await;
    let work_loser = seed_entity(&fixture, "Write Scope Work Loser", Some("work")).await;

    for (loser, canon, dry_run) in [
        (&personal_loser, &work_canon, true),
        (&personal_loser, &work_canon, false),
        (&work_loser, &personal_canon, true),
        (&work_loser, &personal_canon, false),
    ] {
        let merge_uri = format!("/api/memory/entities/{loser}/merge");
        let (status, error) = json_body(
            fixture
                .send(
                    HttpMethod::POST,
                    &merge_uri,
                    Some(json!({"into": canon, "dry_run": dry_run})),
                    Some("work"),
                )
                .await,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{error}");
    }
    for id in [&personal_loser, &work_canon, &personal_canon, &work_loser] {
        let (status, _) = json_body(
            fixture
                .send(
                    HttpMethod::GET,
                    &format!("/api/memory/entities/{id}"),
                    None,
                    None,
                )
                .await,
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "entity {id} must survive a scoped-out merge"
        );
    }
    let merge_uri = format!("/api/memory/entities/{work_loser}/merge");
    let (status, _) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                &merge_uri,
                Some(json!({"into": work_canon, "dry_run": true})),
                Some("missing-space"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, applied) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                &merge_uri,
                Some(json!({"into": work_canon, "dry_run": false})),
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{applied}");

    // ---- delete: mismatch == missing, unknown space is 422, in-scope OKs ----
    let work_delete = seed_entity(&fixture, "Write Scope Delete", Some("work")).await;
    let delete_uri = format!("/api/memory/entities/{work_delete}/delete");
    let delete_mismatch = body_bytes(
        fixture
            .send(HttpMethod::DELETE, &delete_uri, None, Some("personal"))
            .await,
    )
    .await;
    let delete_missing = body_bytes(
        fixture
            .send(
                HttpMethod::DELETE,
                "/api/memory/entities/missing-entity/delete",
                None,
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(delete_mismatch.0, StatusCode::NOT_FOUND);
    assert_eq!(delete_mismatch, delete_missing);
    assert_eq!(delete_mismatch.1, br#"{"error":"entity not found"}"#);
    let (status, _) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/memory/entities/{work_delete}"),
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the mismatched delete must not have deleted anything"
    );
    let (status, _) = json_body(
        fixture
            .send(HttpMethod::DELETE, &delete_uri, None, Some("missing-space"))
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, deleted) = json_body(
        fixture
            .send(HttpMethod::DELETE, &delete_uri, None, Some("work"))
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    let (status, _) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/memory/entities/{work_delete}"),
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "the in-scope delete must have deleted it"
    );
}

pub fn registry_matches_completed_contracts() {
    assert_wave_4_knowledge_catalog_contract();
}
