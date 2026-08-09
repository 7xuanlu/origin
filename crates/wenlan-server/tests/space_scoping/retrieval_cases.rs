// SPDX-License-Identifier: Apache-2.0

use super::fixture::ScopeFixture;
use axum::http::{Method as HttpMethod, StatusCode};
use serde_json::json;
use wenlan_server::sensitive_read_routes::Method as CatalogMethod;

async fn json_body(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 512 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

pub async fn unknown_selectors_are_rejected() {
    let fixture = ScopeFixture::new().await;
    let probes = [
        (
            CatalogMethod::Post,
            HttpMethod::POST,
            "/api/search",
            "/api/search",
            Some(json!({"query":"probe","space":"missing"})),
            None,
        ),
        (
            CatalogMethod::Post,
            HttpMethod::POST,
            "/api/context",
            "/api/context",
            Some(json!({"query":"probe","space":"missing"})),
            None,
        ),
        (
            CatalogMethod::Post,
            HttpMethod::POST,
            "/api/brief",
            "/api/brief",
            Some(json!({"space":"missing"})),
            None,
        ),
        (
            CatalogMethod::Get,
            HttpMethod::GET,
            "/api/memory/recent",
            "/api/memory/recent",
            None,
            Some("missing"),
        ),
        (
            CatalogMethod::Get,
            HttpMethod::GET,
            "/api/memory/unconfirmed",
            "/api/memory/unconfirmed",
            None,
            Some("missing"),
        ),
        (
            CatalogMethod::Post,
            HttpMethod::POST,
            "/api/memory/search",
            "/api/memory/search",
            Some(json!({"query":"probe","space":"missing"})),
            None,
        ),
        (
            CatalogMethod::Post,
            HttpMethod::POST,
            "/api/memory/list",
            "/api/memory/list",
            Some(json!({"space":"missing"})),
            None,
        ),
        (
            CatalogMethod::Get,
            HttpMethod::GET,
            "/api/memory/nurture",
            "/api/memory/nurture?space=missing",
            None,
            None,
        ),
        (
            CatalogMethod::Get,
            HttpMethod::GET,
            "/api/memory/pinned",
            "/api/memory/pinned",
            None,
            Some("missing"),
        ),
    ];

    super::case_runner::assert_wave_1_executed_keys(
        probes
            .iter()
            .map(|(method, _, catalog_path, ..)| (*method, *catalog_path)),
    );

    for (_, method, _, uri, body, header) in probes {
        let response = fixture.send(method, uri, body, header).await;
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "{uri} must reject an unknown Space"
        );
    }
}

pub async fn primary_and_header_precedence() {
    let fixture = ScopeFixture::new().await;

    for (uri, body) in [
        ("/api/search", json!({"query":"probe","space":"work"})),
        ("/api/context", json!({"query":"probe","space":"work"})),
        ("/api/brief", json!({"space":"work"})),
        (
            "/api/memory/search",
            json!({"query":"probe","space":"work"}),
        ),
        ("/api/memory/list", json!({"space":"work"})),
    ] {
        let response = fixture
            .send(HttpMethod::POST, uri, Some(body), Some("missing"))
            .await;
        assert_eq!(response.status(), StatusCode::OK, "body must win for {uri}");
    }

    let response = fixture
        .send(
            HttpMethod::GET,
            "/api/memory/nurture?space=work",
            None,
            Some("missing"),
        )
        .await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "query must win for nurture"
    );
}

pub async fn collections_filter_before_limit() {
    let fixture = ScopeFixture::new().await;
    fixture
        .seed_wave_1_memory("work-canary", Some("work"), 100)
        .await;
    fixture.seed_wave_1_memory("null-canary", None, 200).await;
    fixture
        .seed_wave_1_memory("personal-canary", Some("personal"), 300)
        .await;

    for (uri, envelope) in [
        ("/api/memory/recent?limit=1", None),
        ("/api/memory/unconfirmed?limit=1", None),
        ("/api/memory/pinned", Some("memories")),
    ] {
        let response = fixture.send(HttpMethod::GET, uri, None, Some("work")).await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
        let body = json_body(response).await;
        let rows = envelope.map_or(&body, |key| &body[key]);
        let ids = rows
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row["source_id"]
                    .as_str()
                    .or_else(|| row["id"].as_str())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["work-canary"], "{uri} must scope before limit");
    }

    for (uri, body, envelope) in [
        (
            "/api/memory/list",
            json!({"space":"work","confirmed":false,"limit":1}),
            "memories",
        ),
        (
            "/api/memory/nurture?space=work&limit=1",
            serde_json::Value::Null,
            "cards",
        ),
    ] {
        let method = if uri == "/api/memory/list" {
            HttpMethod::POST
        } else {
            HttpMethod::GET
        };
        let response = fixture
            .send(method, uri, (!body.is_null()).then_some(body), None)
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
        let payload = json_body(response).await;
        let ids = payload[envelope]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["source_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["work-canary"], "{uri} must scope before limit");
    }
}

pub async fn blank_primary_falls_back_and_reserved_collision_rejects() {
    let fixture = ScopeFixture::new().await;
    fixture
        .seed_wave_1_memory("work-blank-canary", Some("work"), 100)
        .await;
    fixture
        .seed_wave_1_memory("personal-blank-canary", Some("personal"), 200)
        .await;

    let response = fixture
        .send(
            HttpMethod::POST,
            "/api/memory/list",
            Some(json!({"space":"  ","limit":20})),
            Some("work"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let payload = json_body(response).await;
    let ids = payload["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["source_id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["work-blank-canary"]);

    fixture
        .db
        .create_space("uncategorized", None, false)
        .await
        .unwrap();
    let response = fixture
        .send(
            HttpMethod::GET,
            "/api/memory/pinned",
            None,
            Some("uncategorized"),
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

pub async fn ranked_routes_exclude_cross_scope_rows() {
    let fixture = ScopeFixture::new().await;
    fixture
        .seed_wave_1_memory("work-search-canary", Some("work"), 100)
        .await;
    fixture
        .seed_wave_1_memory("personal-search-canary", Some("personal"), 200)
        .await;
    fixture
        .seed_wave_1_memory("null-search-canary", None, 300)
        .await;

    for (uri, body, envelope) in [
        (
            "/api/search",
            json!({"query":"scope canary","space":"work","source_filter":"memory","limit":20}),
            "results",
        ),
        (
            "/api/memory/search",
            json!({"query":"scope canary","space":"work","limit":20}),
            "results",
        ),
    ] {
        let response = fixture
            .send(HttpMethod::POST, uri, Some(body), Some("personal"))
            .await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
        let payload = json_body(response).await;
        let rows = &payload[envelope];
        let rows = rows.as_array().unwrap();
        assert!(
            rows.iter()
                .any(|row| row["source_id"] == "work-search-canary"),
            "{uri} must retain an in-scope positive canary: {rows:?}"
        );
        assert!(
            rows.iter().all(|row| row["space"] == "work"),
            "{uri} leaked a cross-scope row: {rows:?}"
        );
    }
}
