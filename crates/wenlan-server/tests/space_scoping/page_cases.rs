// SPDX-License-Identifier: Apache-2.0

use super::case_runner::{assert_wave_3_pages_catalog_contract, assert_wave_3_pages_executed_keys};
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

async fn seed_page(
    fixture: &ScopeFixture,
    title: &str,
    category: &str,
    workspace: Option<&str>,
    content: &str,
    sources: &[&str],
) -> String {
    let id = fixture.seed_page(title, category).await;
    fixture.db.set_page_workspace(&id, workspace).await.unwrap();
    fixture
        .db
        .update_page_content(&id, content, sources, "wave_3_test")
        .await
        .unwrap();
    id
}

fn ids(body: &Value, key: &str) -> Vec<String> {
    body[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array: {body}"))
        .iter()
        .filter_map(|row| row["id"].as_str().or_else(|| row["page_id"].as_str()))
        .map(str::to_string)
        .collect()
}

pub async fn unknown_selectors_are_rejected() {
    let fixture = ScopeFixture::new().await;
    let page = seed_page(
        &fixture,
        "unknown selector canary",
        "decision",
        Some("work"),
        "unknown selector canary",
        &[],
    )
    .await;
    let probes = [
        (
            HttpMethod::GET,
            "/api/pages/recent".to_string(),
            None,
            "/api/pages/recent",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            "/api/pages/recent-changes".to_string(),
            None,
            "/api/pages/recent-changes",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            "/api/pages?space=missing-space".to_string(),
            None,
            "/api/pages",
            Method::Get,
        ),
        (
            HttpMethod::POST,
            "/api/pages/search".to_string(),
            Some(json!({"query":"canary","space":"missing-space"})),
            "/api/pages/search",
            Method::Post,
        ),
        (
            HttpMethod::POST,
            "/api/pages/export".to_string(),
            Some(json!({"vault_path":"/tmp/wenlan-route-scope-export"})),
            "/api/pages/export",
            Method::Post,
        ),
        (
            HttpMethod::GET,
            "/api/pages/orphan-links".to_string(),
            None,
            "/api/pages/orphan-links",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            format!("/api/pages/{page}"),
            None,
            "/api/pages/{id}",
            Method::Get,
        ),
        (
            HttpMethod::POST,
            format!("/api/pages/{page}/export"),
            Some(json!({"vault_path":"/tmp/wenlan-route-scope-export"})),
            "/api/pages/{id}/export",
            Method::Post,
        ),
        (
            HttpMethod::GET,
            format!("/api/pages/{page}/sources"),
            None,
            "/api/pages/{id}/sources",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            format!("/api/pages/{page}/links"),
            None,
            "/api/pages/{id}/links",
            Method::Get,
        ),
        (
            HttpMethod::GET,
            format!("/api/pages/{page}/revisions"),
            None,
            "/api/pages/{id}/revisions",
            Method::Get,
        ),
    ];

    let mut executed = Vec::new();
    for (method, uri, body, key, catalog_method) in probes {
        let header = if uri.contains("space=missing-space") {
            Some("work")
        } else {
            Some("missing-space")
        };
        let response = fixture.send(method, &uri, body, header).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY, "{uri}");
        executed.push((catalog_method, key));
    }
    assert_wave_3_pages_executed_keys(executed);
}

pub async fn collections_and_precedence_are_scoped() {
    let fixture = ScopeFixture::new().await;
    fixture
        .seed_record(
            "memory",
            "work-source",
            Some("work"),
            "fact",
            10,
            None,
            false,
        )
        .await;
    let work = seed_page(
        &fixture,
        "work decision page",
        "decision",
        Some("work"),
        "workspace collection canary",
        &["work-source"],
    )
    .await;
    let personal = seed_page(
        &fixture,
        "personal recap page",
        "recap",
        Some("personal"),
        "workspace collection canary [[personal orphan]]",
        &["work-source"],
    )
    .await;
    let uncategorized = seed_page(
        &fixture,
        "null workspace page",
        "decision",
        None,
        "workspace collection canary",
        &[],
    )
    .await;
    let literal = seed_page(
        &fixture,
        "literal uncategorized workspace page",
        "recap",
        Some("uncategorized"),
        "workspace collection canary",
        &[],
    )
    .await;

    let (status, list) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/pages?space=work&limit=20",
                None,
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(ids(&list, "pages"), vec![work.clone()]);

    let (status, fallback) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/pages?space=&limit=20",
                None,
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{fallback}");
    assert_eq!(ids(&fallback, "pages"), vec![personal.clone()]);

    // M1 (one honest column): `page_type` and `space` both filter c.space, so a
    // page cannot carry a category independent of its scope -- the dual-axis
    // `page_type=decision` + `space=work` cross-filter is the deleted three-state
    // model. What survives, and what this case verifies, is scope precedence: an
    // explicit body `space` wins over the header. (page_type's own
    // filter-on-scope is covered by `search_pages_filters_by_page_type`.)
    let (status, body_wins) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/pages/search",
                Some(json!({"query":"workspace collection canary","limit":20,"space":"work"})),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body_wins}");
    assert_eq!(ids(&body_wins, "pages"), vec![work.clone()]);

    // Blank body `space` falls back to the header scope (same single-axis rule;
    // page_type rider dropped for the reason above).
    let (status, header_fallback) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/pages/search",
                Some(json!({"query":"workspace collection canary","limit":20,"space":""})),
                Some("personal"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{header_fallback}");
    assert_eq!(ids(&header_fallback, "pages"), vec![personal.clone()]);

    let (status, legacy_global) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/pages/search",
                Some(json!({"query":"workspace collection canary","limit":20})),
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{legacy_global}");
    assert_eq!(
        ids(&legacy_global, "pages")
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            work.clone(),
            personal.clone(),
            uncategorized.clone(),
            literal.clone(),
        ])
    );

    for uri in [
        "/api/pages/recent?limit=20",
        "/api/pages/recent-changes?limit=20",
    ] {
        let (status, body) =
            json_body(fixture.send(HttpMethod::GET, uri, None, Some("work")).await).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {body}");
        let serialized = body.to_string();
        assert!(serialized.contains(&work), "{uri}: {serialized}");
        assert!(!serialized.contains(&personal), "{uri}: {serialized}");
        assert!(!serialized.contains(&uncategorized), "{uri}: {serialized}");
    }

    let (status, global) = json_body(
        fixture
            .send(HttpMethod::GET, "/api/pages?limit=20", None, None)
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{global}");
    assert!(ids(&global, "pages").contains(&literal));

    let (status, null_only) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/pages?space=uncategorized&limit=20",
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{null_only}");
    assert_eq!(ids(&null_only, "pages"), vec![uncategorized]);

    let fixture = ScopeFixture::new().await;
    let only = seed_page(
        &fixture,
        "truthful short list",
        "decision",
        Some("work"),
        "truthful short list query",
        &[],
    )
    .await;
    let (status, short) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/pages/search",
                Some(json!({"query":"truthful short list query","limit":10,"space":"work"})),
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{short}");
    assert_eq!(ids(&short, "pages"), vec![only]);
}

pub async fn ranked_candidates_are_filtered_before_limit() {
    let fixture = ScopeFixture::new().await;
    let work = seed_page(
        &fixture,
        "distant cedar manual",
        "decision",
        Some("work"),
        "deliberately unrelated low-ranked content",
        &[],
    )
    .await;
    for index in 0..8 {
        seed_page(
            &fixture,
            &format!("quasar nebula personal {index}"),
            "recap",
            Some("personal"),
            "quasar nebula exact high ranked content",
            &[],
        )
        .await;
    }

    let (status, body) = json_body(
        fixture
            .send(
                HttpMethod::POST,
                "/api/pages/search",
                Some(json!({"query":"quasar nebula","limit":1})),
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids(&body, "pages"), vec![work]);
}

pub async fn direct_and_child_routes_are_gated() {
    let fixture = ScopeFixture::new().await;
    fixture
        .seed_record(
            "memory",
            "work-memory",
            Some("work"),
            "fact",
            10,
            None,
            false,
        )
        .await;
    fixture
        .seed_record(
            "memory",
            "personal-memory",
            Some("personal"),
            "fact",
            11,
            None,
            false,
        )
        .await;
    fixture
        .seed_record(
            "file",
            "external-file",
            Some("personal"),
            "document",
            12,
            None,
            false,
        )
        .await;
    let work = seed_page(
        &fixture,
        "work parent page",
        "decision",
        Some("work"),
        "work parent [[missing work topic]]",
        &["work-memory", "personal-memory", "external-file"],
    )
    .await;
    let personal = seed_page(
        &fixture,
        "personal parent page",
        "recap",
        Some("personal"),
        "personal parent [[missing personal topic]]",
        &["work-memory"],
    )
    .await;
    let second_work = seed_page(
        &fixture,
        "second work page",
        "decision",
        Some("work"),
        "second work [[missing work topic]]",
        &[],
    )
    .await;
    seed_page(
        &fixture,
        "second personal page",
        "recap",
        Some("personal"),
        "second personal [[missing personal topic]]",
        &[],
    )
    .await;

    let (status, page) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/pages/{work}"),
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(page["page"]["id"], work);

    let (status, sources) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/pages/{work}/sources"),
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{sources}");
    let serialized = sources.to_string();
    assert!(serialized.contains("work-memory"), "{serialized}");
    assert!(
        serialized.contains("personal-memory"),
        "source metadata may remain: {serialized}"
    );
    assert!(
        !serialized.contains("record canary personal-memory"),
        "cross-Space Memory content leaked: {serialized}"
    );
    assert!(
        !serialized.contains("record canary external-file"),
        "selected Page source materialized cross-Space external content: {serialized}"
    );

    let (status, global_sources) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/pages/{work}/sources"),
                None,
                None,
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{global_sources}");
    assert!(
        global_sources
            .to_string()
            .contains("record canary external-file"),
        "Global Page source compatibility lost external content: {global_sources}"
    );

    let (status, orphan) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                "/api/pages/orphan-links?min_count=2",
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{orphan}");
    let serialized = orphan.to_string();
    assert!(serialized.contains("missing work topic"), "{serialized}");
    assert!(
        !serialized.contains("missing personal topic"),
        "{serialized}"
    );

    for suffix in ["", "/sources", "/links", "/revisions"] {
        let mismatch = body_bytes(
            fixture
                .send(
                    HttpMethod::GET,
                    &format!("/api/pages/{personal}{suffix}"),
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
                    &format!("/api/pages/missing{suffix}"),
                    None,
                    Some("work"),
                )
                .await,
        )
        .await;
        assert_eq!(mismatch.0, StatusCode::NOT_FOUND, "{suffix}");
        assert_eq!(missing.0, StatusCode::NOT_FOUND, "{suffix}");
        assert_eq!(mismatch.1, missing.1, "{suffix}");
        assert_eq!(mismatch.1, br#"{"error":"page not found"}"#);
    }

    let (status, links) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/pages/{work}/links"),
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{links}");
    assert!(links.to_string().contains("missing work topic"));

    let (status, revisions) = json_body(
        fixture
            .send(
                HttpMethod::GET,
                &format!("/api/pages/{second_work}/revisions"),
                None,
                Some("work"),
            )
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{revisions}");
}

pub fn registry_matches_completed_contracts() {
    assert_wave_3_pages_catalog_contract();
}
