// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

async fn request_json(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Option<Value>,
    header_space: Option<&str>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(space) = header_space {
        builder = builder.header("x-wenlan-space", space);
    }
    let body = if let Some(body) = body {
        builder = builder.header("content-type", "application/json");
        Body::from(body.to_string())
    } else {
        Body::empty()
    };
    let response = router
        .clone()
        .oneshot(builder.body(body).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            panic!(
                "response must be JSON ({error}): {}",
                String::from_utf8_lossy(&bytes)
            )
        })
    };
    (status, value)
}

#[tokio::test]
async fn default_space_api_lifecycle_and_status_capability() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    let work = db.create_space("work", None, false).await.unwrap();
    let personal = db.create_space("personal", None, false).await.unwrap();

    let (status, body) =
        request_json(&router, Method::GET, "/api/spaces/default", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"space": null}));

    let (status, body) = request_json(
        &router,
        Method::PUT,
        "/api/spaces/default",
        Some(json!({"space_id": work.id})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["space"]["name"], "work");
    assert_eq!(body["space"]["is_default"], true);

    let (status, body) = request_json(
        &router,
        Method::PUT,
        "/api/spaces/default",
        Some(json!({"space_id": personal.id})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["space"]["name"], "personal");
    assert_eq!(
        db.list_spaces()
            .await
            .unwrap()
            .into_iter()
            .filter(|space| space.is_default)
            .count(),
        1
    );

    let (status, _) =
        request_json(&router, Method::DELETE, "/api/spaces/default", None, None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(db.get_default_space().await.unwrap().is_none());

    let (status, _) = request_json(
        &router,
        Method::PUT,
        "/api/spaces/default",
        Some(json!({"space_id": "00000000-0000-4000-8000-000000000001"})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let (status, body) = request_json(&router, Method::GET, "/api/status", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("default_save_space")),
        "status must advertise the additive default_save_space capability: {body}"
    );
}

#[tokio::test]
async fn store_space_precedence_and_truthful_receipts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    let default = db.create_space("default", None, false).await.unwrap();
    db.create_space("header", None, false).await.unwrap();
    db.create_space("body", None, false).await.unwrap();
    db.set_default_space(&default.id).await.unwrap();

    let cases = [
        (
            json!({"content":"body Space precedence canary", "space":"body"}),
            Some("header"),
            Some("body"),
            "request",
        ),
        (
            json!({"content":"explicit Uncategorized precedence canary", "space":null}),
            Some("header"),
            None,
            "uncategorized",
        ),
        (
            json!({"content":"header Space precedence canary"}),
            Some("header"),
            Some("header"),
            "header",
        ),
        (
            json!({"content":"daemon Default precedence canary"}),
            None,
            Some("default"),
            "default",
        ),
    ];

    for (body, header, expected_space, expected_source) in cases {
        let (status, response) = request_json(
            &router,
            Method::POST,
            "/api/memory/store",
            Some(body),
            header,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(
            response.get("space").and_then(Value::as_str),
            expected_space,
            "{response}"
        );
        assert_eq!(response["space_source"], expected_source, "{response}");
        assert_eq!(response["write_outcome"], "created", "{response}");
    }

    let (status, body) = request_json(
        &router,
        Method::POST,
        "/api/memory/store",
        Some(json!({
            "content":"unknown named Space must fail loudly",
            "space":"missing"
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("not registered"),
        "{body}"
    );
}

#[tokio::test]
async fn import_memories_inherits_default_without_moving_duplicates() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    let work = db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    db.set_default_space(&work.id).await.unwrap();

    let content = "- Import Space canary remembers the daemon default destination.";
    let (status, imported) = request_json(
        &router,
        Method::POST,
        "/api/import/memories",
        Some(json!({
            "source":"other",
            "content":content
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{imported}");
    assert_eq!(imported["imported"], 1);
    assert_eq!(imported["space"], "work");
    assert_eq!(imported["space_source"], "default");

    let source_id = format!("import_{}_0", imported["batch_id"].as_str().unwrap());
    assert_eq!(
        db.get_memory_space(&source_id).await.unwrap().as_deref(),
        Some("work")
    );

    let (status, duplicate) = request_json(
        &router,
        Method::POST,
        "/api/import/memories",
        Some(json!({
            "source":"other",
            "content":content,
            "space":"personal"
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{duplicate}");
    assert_eq!(duplicate["imported"], 0);
    assert_eq!(duplicate["skipped"], 1);
    assert_eq!(
        db.get_memory_space(&source_id).await.unwrap().as_deref(),
        Some("work"),
        "a duplicate import must not re-file the existing memory"
    );
}

#[tokio::test]
async fn create_entity_reports_existing_owner_without_moving() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();

    let (status, created) = request_json(
        &router,
        Method::POST,
        "/api/memory/entities",
        Some(json!({
            "name":"Default Space Entity Canary",
            "entity_type":"concept",
            "space":"work"
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(created["space"], "work");
    assert_eq!(created["space_source"], "request");
    assert_eq!(created["write_outcome"], "created");

    let (status, existing) = request_json(
        &router,
        Method::POST,
        "/api/memory/entities",
        Some(json!({
            "name":"Default Space Entity Canary",
            "entity_type":"concept",
            "space":"personal"
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{existing}");
    assert_eq!(existing["id"], created["id"]);
    assert_eq!(existing["space"], "work");
    assert_eq!(existing["space_source"], "existing");
    assert_eq!(existing["write_outcome"], "resolved_existing");

    let detail = db
        .get_entity_detail(created["id"].as_str().unwrap())
        .await
        .unwrap();
    assert_eq!(detail.entity.space.as_deref(), Some("work"));
}

#[tokio::test]
async fn create_page_rejects_conflicting_aliases_and_mirrors_default() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    let work = db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    db.set_default_space(&work.id).await.unwrap();

    let (status, conflict) = request_json(
        &router,
        Method::POST,
        "/api/pages",
        Some(json!({
            "title":"Conflicting Space Alias Canary",
            "content":"This page must never be created under conflicting aliases.",
            "space":"work",
            "workspace":"personal",
            "creation_kind":"authored"
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{conflict}");

    let (status, created) = request_json(
        &router,
        Method::POST,
        "/api/pages",
        Some(json!({
            "title":"Default Space Page Canary",
            "content":"This page proves the daemon Default save space is mirrored.",
            "creation_kind":"authored"
        })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(created["space"], "work");
    assert_eq!(created["space_source"], "default");
    assert_eq!(created["write_outcome"], "created");

    let page = db
        .get_page(created["id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(page.space.as_deref(), Some("work"));
    assert_eq!(page.workspace.as_deref(), Some("work"));
}
