//! Type-contract tests: verify origin-mcp deserializes origin-server's wire shapes correctly.
//!
//! Unlike `serve_integration.rs` (which tests origin-mcp's own HTTP endpoints without a
//! backend), these tests stand up a live `wiremock` listening on a real port, point
//! origin-mcp's HTTP client at it, and invoke the `_impl` methods directly.
//!
//! The wiremock responses are constructed from `wenlan_types::*` types via
//! `serde_json::to_value` - by construction, a passing test proves origin-mcp deserializes
//! the same JSON origin-server would emit for that shape.

use rmcp::model::{CallToolResult, RawContent};
use wenlan_mcp::client::WenlanClient;
use wenlan_mcp::tools::{
    CaptureParams, ContextParams, CreateRelationParams, ListPendingImportsParams,
    ListPendingParams, ListRejectionsParams, RecallParams, TransportMode, WenlanMcpServer,
};
use wenlan_types::import::PendingImport;
use wenlan_types::memory::{IndexedFileInfo, SearchResult};
use wenlan_types::memory::{MemoryItem, RejectionRecord};
use wenlan_types::responses::{
    ChatContextResponse, CreateRelationResponse, DeleteResponse, KnowledgeContext,
    ListMemoriesResponse, MemoryDetailResponse, ProfileContext, SearchMemoryResponse,
    StoreMemoryResponse, TierTokenEstimates,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn setup() -> (MockServer, WenlanClient) {
    let mock = MockServer::start().await;
    let client = WenlanClient::new(mock.uri());
    (mock, client)
}

fn make_server(client: WenlanClient) -> WenlanMcpServer {
    WenlanMcpServer::new(client, TransportMode::Stdio, "test-agent".into(), None)
}

fn text_of(result: &CallToolResult) -> String {
    for content in &result.content {
        match &content.raw {
            RawContent::Text(text) => return text.text.clone(),
            _ => continue,
        }
    }
    panic!(
        "expected at least one text Content block; got: {:?}",
        result.content
    );
}

async fn captured_body(mock: &MockServer) -> serde_json::Value {
    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(
        received.len(),
        1,
        "expected exactly 1 captured request, got {}",
        received.len()
    );
    serde_json::from_slice(&received[0].body).expect("captured body is not valid JSON")
}

fn sample_search_result() -> SearchResult {
    SearchResult {
        id: "1".into(),
        content: "some memory content".into(),
        source: "memory".into(),
        source_id: "mem_r1".into(),
        title: "title".into(),
        url: None,
        chunk_index: 0,
        last_modified: 0,
        score: 0.9,
        chunk_type: None,
        language: None,
        semantic_unit: None,
        memory_type: Some("fact".into()),
        space: None,
        source_agent: None,
        confidence: None,
        confirmed: None,
        stability: None,
        supersedes: None,
        summary: None,
        entity_id: None,
        entity_name: None,
        quality: None,
        importance: None,
        event_date: None,
        content_hash: None,
        is_archived: false,
        is_recap: false,
        structured_fields: None,
        retrieval_cue: None,
        source_text: None,
        raw_score: 0.0,
        version: 0,
        pending_revision: false,
        merged_from: None,
        last_delta_summary: None,
    }
}

fn sample_memory(source_id: &str) -> MemoryItem {
    MemoryItem {
        source_id: source_id.into(),
        title: "Explicit relation".into(),
        content: "Alice works on Wenlan.".into(),
        summary: None,
        memory_type: Some("fact".into()),
        space: None,
        source_agent: Some("test-agent".into()),
        confidence: Some(1.0),
        confirmed: false,
        stability: None,
        pinned: false,
        supersedes: None,
        last_modified: 0,
        chunk_count: 1,
        entity_id: None,
        quality: None,
        is_recap: false,
        enrichment_status: String::new(),
        supersede_mode: String::new(),
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

#[tokio::test]
async fn t1_remember_roundtrip() {
    let (mock, client) = setup().await;
    let response = StoreMemoryResponse {
        source_id: "mem_t1".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "none".into(),

        enrichment: String::new(),

        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "anything".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let text = text_of(&result);
    assert_eq!(text, "Stored mem_t1\nsource_memory_id: mem_t1");

    let body = captured_body(&mock).await;
    assert_eq!(body["content"], serde_json::json!("anything"));
    assert_eq!(body["source_agent"], serde_json::json!("test-agent"));
}

#[tokio::test]
async fn create_relation_rejects_blank_source_before_network() {
    let (mock, client) = setup().await;
    let server = make_server(client);

    let result = server
        .create_relation_impl(CreateRelationParams {
            from_entity_id: "ent_alice".into(),
            to_entity_id: "ent_wenlan".into(),
            relation_type: "works_on".into(),
            source_memory_id: "   ".into(),
        })
        .await
        .unwrap();

    assert!(result.is_error.unwrap_or(false));
    assert!(
        mock.received_requests().await.unwrap().is_empty(),
        "blank source must reject before any HTTP request"
    );
}

#[tokio::test]
async fn create_relation_missing_source_preflight_prevents_post() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/mem_missing/detail"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&mock)
        .await;
    let server = make_server(client);

    let result = server
        .create_relation_impl(CreateRelationParams {
            from_entity_id: "ent_alice".into(),
            to_entity_id: "ent_wenlan".into(),
            relation_type: "works_on".into(),
            source_memory_id: "mem_missing".into(),
        })
        .await
        .unwrap();

    assert!(result.is_error.unwrap_or(false));
    let received = mock.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        1,
        "missing source must stop after preflight"
    );
    assert_eq!(received[0].method.as_str(), "GET");
}

#[tokio::test]
async fn create_relation_successful_preflight_maps_source_into_post() {
    let (mock, client) = setup().await;
    let source_id = "mem source/1";
    Mock::given(method("GET"))
        .and(path("/api/memory/mem%20source%2F1/detail"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(MemoryDetailResponse {
                memory: Some(sample_memory(source_id)),
            }),
        )
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/memory/relations"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(CreateRelationResponse {
                id: "rel_1".into(),
                warnings: Vec::new(),
            }),
        )
        .mount(&mock)
        .await;
    let server = make_server(client);

    let result = server
        .create_relation_impl(CreateRelationParams {
            from_entity_id: "ent_alice".into(),
            to_entity_id: "ent_wenlan".into(),
            relation_type: "works_on".into(),
            source_memory_id: source_id.into(),
        })
        .await
        .unwrap();

    assert!(!result.is_error.unwrap_or(false));
    let received = mock.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        2,
        "preflight GET must precede relation POST"
    );
    assert_eq!(received[0].method.as_str(), "GET");
    assert_eq!(received[1].method.as_str(), "POST");
    let post_body: serde_json::Value = serde_json::from_slice(&received[1].body).unwrap();
    assert_eq!(post_body["source_memory_id"], source_id);
}

#[tokio::test]
async fn t2_remember_surfaces_warnings_when_present() {
    let (mock, client) = setup().await;
    let response = StoreMemoryResponse {
        source_id: "mem_t2".into(),
        chunks_created: 1,
        memory_type: "decision".into(),
        entity_id: None,
        quality: None,
        warnings: vec!["decision memory missing required 'claim' field".into()],
        near_duplicate: None,
        extraction_method: "agent".into(),

        enrichment: String::new(),

        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "anything".into(),
            memory_type: Some("decision".into()),
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let text = text_of(&result);
    assert!(
        text.starts_with("Stored mem_t2"),
        "expected source_id line first; got: {text}"
    );
    assert!(
        text.contains("Warnings:"),
        "expected Warnings: section; got: {text}"
    );
    assert!(
        text.contains("decision memory missing required 'claim' field"),
        "expected validation text; got: {text}"
    );
}

#[tokio::test]
async fn t3_structured_fields_schema_is_object() {
    use schemars::schema_for;

    let (mock, client) = setup().await;
    let response = StoreMemoryResponse {
        source_id: "mem_t3".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "agent".into(),

        enrichment: String::new(),

        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let mut structured_fields = serde_json::Map::new();
    structured_fields.insert("theme".into(), serde_json::json!("dark"));

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "prefers dark mode".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: Some(structured_fields),
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let text = text_of(&result);
    assert_eq!(text, "Stored mem_t3\nsource_memory_id: mem_t3");

    let body = captured_body(&mock).await;
    assert_eq!(
        body["structured_fields"],
        serde_json::json!({ "theme": "dark" })
    );

    let schema = schema_for!(CaptureParams);
    let json = serde_json::to_value(&schema).unwrap();
    let sf = json
        .pointer("/properties/structured_fields")
        .expect("structured_fields property present in schema");
    let t = sf.pointer("/type").expect("type constraint present");
    match t {
        serde_json::Value::Array(arr) => {
            assert!(
                arr.iter().any(|v| v.as_str() == Some("object")),
                "expected 'object' among type array; got: {:?}",
                arr
            );
            assert!(
                arr.iter().any(|v| v.as_str() == Some("null")),
                "expected 'null' among type array; got: {:?}",
                arr
            );
        }
        serde_json::Value::String(s) => panic!("expected nullable object schema, got {}", s),
        other => panic!(
            "schema type constraint is not a string or array: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn t4_recall_roundtrip() {
    let (mock, client) = setup().await;
    let response = SearchMemoryResponse {
        results: vec![sample_search_result()],
        took_ms: 10.0,
        supplemental_pages: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .recall_impl(RecallParams {
            query: "anything".into(),
            limit: None,
            memory_type: None,
            space: None,
            rerank: None,
        })
        .await
        .expect("recall_impl failed");

    let text = text_of(&result);
    assert!(
        text.contains("1 results"),
        "expected result count line; got: {text}"
    );
    assert!(
        text.contains("mem_r1"),
        "expected source_id in rendered JSON; got: {text}"
    );

    let body = captured_body(&mock).await;
    assert_eq!(body["query"], serde_json::json!("anything"));
    assert_eq!(body["limit"], serde_json::json!(10));
    assert!(body["memory_type"].is_null());
    assert!(body["space"].is_null());
    assert_eq!(
        body["source_agent"],
        serde_json::json!("test-agent"),
        "recall should send resolved agent name, not null"
    );
}

#[tokio::test]
async fn t5_memory_type_hint_preserved_without_forcing_domain() {
    let (mock, client) = setup().await;
    let response = StoreMemoryResponse {
        source_id: "mem_t5".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: Some("medium".into()),
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "llm".into(),

        enrichment: String::new(),

        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "some content".into(),
            memory_type: Some("fact".into()),
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let text = text_of(&result);
    assert_eq!(text, "Stored mem_t5\nsource_memory_id: mem_t5");

    let body = captured_body(&mock).await;
    assert_eq!(body["memory_type"], serde_json::json!("fact"));
    assert!(body["space"].is_null());
}

#[tokio::test]
async fn t6_context_roundtrip_bug_regression() {
    let (mock, client) = setup().await;
    #[allow(deprecated)]
    let response = ChatContextResponse {
        context: "you are Lucian".into(),
        profile: ProfileContext {
            narrative: "n".into(),
            identity: vec!["rust developer".into()],
            preferences: vec![],
            goals: vec![],
        },
        knowledge: KnowledgeContext {
            pages: vec![],
            decisions: vec![],
            relevant_memories: vec![],
            graph_context: vec![],
        },
        took_ms: 12.0,
        token_estimates: TierTokenEstimates {
            tier1_identity: 5,
            tier2_project: 10,
            tier3_relevant: 15,
            total: 30,
        },
    };
    Mock::given(method("POST"))
        .and(path("/api/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .context_impl(ContextParams {
            topic: Some("orientation".into()),
            limit: None,
            space: None,
        })
        .await
        .expect("context_impl failed");

    let text = text_of(&result);
    assert_eq!(text, "you are Lucian");

    let body = captured_body(&mock).await;
    assert_eq!(body["conversation_id"], serde_json::json!("orientation"));
    assert_eq!(body["max_chunks"], serde_json::json!(20));
    assert_eq!(body["include_goals"], serde_json::json!(true));
    assert!(body["space"].is_null());
}

#[tokio::test]
async fn t7_context_with_domain() {
    let (mock, client) = setup().await;
    #[allow(deprecated)]
    let response = ChatContextResponse {
        context: "work context".into(),
        profile: ProfileContext {
            narrative: String::new(),
            identity: vec![],
            preferences: vec![],
            goals: vec![],
        },
        knowledge: KnowledgeContext {
            pages: vec![],
            decisions: vec![],
            relevant_memories: vec![],
            graph_context: vec![],
        },
        took_ms: 5.0,
        token_estimates: TierTokenEstimates {
            tier1_identity: 0,
            tier2_project: 0,
            tier3_relevant: 0,
            total: 0,
        },
    };
    Mock::given(method("POST"))
        .and(path("/api/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .context_impl(ContextParams {
            topic: None,
            limit: None,
            space: Some("work".into()),
        })
        .await
        .expect("context_impl failed");

    let text = text_of(&result);
    assert_eq!(text, "work context");

    let body = captured_body(&mock).await;
    assert_eq!(body["space"], serde_json::json!("work"));
}

#[tokio::test]
async fn t8_forget_roundtrip() {
    let (deleted_mock, deleted_client) = setup().await;
    let deleted_response = DeleteResponse { deleted: true };
    Mock::given(method("DELETE"))
        .and(path("/api/memory/delete/mem_xyz"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&deleted_response))
        .mount(&deleted_mock)
        .await;

    let deleted_server = make_server(deleted_client);
    let deleted_result = deleted_server
        .forget_impl("mem_xyz")
        .await
        .expect("forget_impl failed for deleted=true");

    let deleted_text = text_of(&deleted_result);
    assert_eq!(deleted_text, "Memory deleted");

    let (missing_mock, missing_client) = setup().await;
    let missing_response = DeleteResponse { deleted: false };
    Mock::given(method("DELETE"))
        .and(path("/api/memory/delete/mem_missing"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&missing_response))
        .mount(&missing_mock)
        .await;

    let missing_server = make_server(missing_client);
    let missing_result = missing_server
        .forget_impl("mem_missing")
        .await
        .expect("forget_impl failed for deleted=false");

    let missing_text = text_of(&missing_result);
    assert_eq!(missing_text, "Memory not found");
}

#[tokio::test]
async fn t9_recall_request_does_not_contain_entity() {
    let (mock, client) = setup().await;
    let response = SearchMemoryResponse {
        results: vec![],
        took_ms: 1.0,
        supplemental_pages: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .recall_impl(RecallParams {
            query: "anything".into(),
            limit: None,
            memory_type: None,
            space: None,
            rerank: None,
        })
        .await
        .expect("recall_impl failed");

    let text = text_of(&result);
    assert!(
        text.contains("0 results"),
        "expected empty result count; got: {text}"
    );

    let body = captured_body(&mock).await;
    let obj = body.as_object().expect("body is an object");
    assert!(
        !obj.contains_key("entity"),
        "entity field leaked into wire body: {:?}",
        obj.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn t10_remember_request_does_not_contain_user_id() {
    let (mock, client) = setup().await;
    let response = StoreMemoryResponse {
        source_id: "mem_t10".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "none".into(),

        enrichment: String::new(),

        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "anything".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let text = text_of(&result);
    assert_eq!(text, "Stored mem_t10\nsource_memory_id: mem_t10");

    let body = captured_body(&mock).await;
    let obj = body.as_object().expect("body is an object");
    assert!(
        !obj.contains_key("user_id"),
        "user_id field leaked into wire body: {:?}",
        obj.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn t11_extraction_method_none_not_in_text() {
    let (mock, client) = setup().await;
    let response = StoreMemoryResponse {
        source_id: "mem_t11".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "none".into(),

        enrichment: String::new(),

        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "anything".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let text = text_of(&result);
    assert!(
        !text.contains("extraction_method"),
        "extraction_method label leaked into text: {text}"
    );
    assert_eq!(text, "Stored mem_t11\nsource_memory_id: mem_t11");
}

/// Regression test: context_impl must succeed even when the daemon returns
/// extra fields in the ChatContextResponse that origin-types v0.1.0 doesn't know
/// about (e.g. future enrichment fields added to SearchResult or top-level).
#[tokio::test]
async fn t13_context_forward_compat_with_extra_fields() {
    // Simulate a future daemon response that adds unknown fields to
    // relevant_memories items and to the top-level response.
    let raw_json = serde_json::json!({
        "context": "you are Lucian",
        "profile": {
            "narrative": "n",
            "identity": ["rust developer"],
            "preferences": [],
            "goals": []
        },
        "knowledge": {
            "relevant_memories": [{
                "id": "1",
                "content": "some memory",
                "source": "memory",
                "source_id": "mem_r1",
                "title": "title",
                "url": null,
                "chunk_index": 0,
                "last_modified": 0,
                "score": 0.9,
                "is_archived": false,
                "is_recap": false,
                "raw_score": 0.0,
                "unknown_future_field": "this should not break deserialization",
                "another_new_field": {"nested": "object"}
            }],
            "graph_context": [],
            "pages": [],
            "decisions": []
        },
        "took_ms": 12.0,
        "token_estimates": {
            "tier1_identity": 5,
            "tier2_project": 10,
            "tier3_relevant": 15,
            "total": 30
        },
        "top_level_future_field": "also ignored"
    });

    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&raw_json))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .context_impl(ContextParams {
            topic: None,
            limit: None,
            space: None,
        })
        .await
        .expect("context_impl must succeed even with extra unknown fields in response");

    let text = text_of(&result);
    assert_eq!(text, "you are Lucian");
}

#[tokio::test]
async fn t12_forward_compat_response_missing_extraction_method() {
    let raw_json = serde_json::json!({
        "source_id": "mem_t12",
        "chunks_created": 2,
        "memory_type": "fact"
    });

    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&raw_json))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .capture_impl(CaptureParams {
            content: "anything".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed against pre-D9 response");

    let text = text_of(&result);
    assert_eq!(text, "Stored mem_t12\nsource_memory_id: mem_t12");

    let parsed: StoreMemoryResponse = serde_json::from_value(raw_json).unwrap();
    assert_eq!(parsed.extraction_method, "unknown");
    assert!(parsed.warnings.is_empty());
}

#[tokio::test]
async fn origin_client_sends_x_agent_name_header() {
    let mock = MockServer::start().await;
    let client = WenlanClient::new(mock.uri()).with_agent_name("claude-code".into());

    let response = StoreMemoryResponse {
        source_id: "mem_xan1".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "none".into(),
        enrichment: String::new(),
        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    server
        .capture_impl(CaptureParams {
            content: "header test".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 request");
    let headers = &received[0].headers;
    let value = headers
        .get("x-agent-name")
        .expect("x-agent-name header must be present");
    assert_eq!(
        value.to_str().expect("header value is valid utf-8"),
        "claude-code",
        "x-agent-name header must equal the configured agent name"
    );
}

#[tokio::test]
async fn origin_client_omits_x_agent_name_when_unset() {
    let (mock, client) = setup().await;

    let response = StoreMemoryResponse {
        source_id: "mem_xan2".into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        near_duplicate: None,
        extraction_method: "none".into(),
        enrichment: String::new(),
        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    server
        .capture_impl(CaptureParams {
            content: "no header test".into(),
            memory_type: None,
            space: None,
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl failed");

    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 request");
    let headers = &received[0].headers;
    assert!(
        headers.get("x-agent-name").is_none(),
        "x-agent-name header must be absent when agent_name is not set"
    );
}

// ===== list_pending_imports =====

fn sample_pending_import(id: &str) -> PendingImport {
    PendingImport {
        id: id.into(),
        vendor: "claude".into(),
        stage: "ingest".into(),
        source_path: "/tmp/import.zip".into(),
        processed_conversations: 5,
        total_conversations: Some(20),
    }
}

#[tokio::test]
async fn list_pending_imports_happy_path() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/import/state"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(vec![sample_pending_import("imp_1")]),
        )
        .mount(&mock)
        .await;

    let result = make_server(client)
        .list_pending_imports_impl(ListPendingImportsParams {})
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(text.starts_with("1 pending import(s)"), "got: {text}");
    assert!(text.contains("imp_1"), "got: {text}");
}

#[tokio::test]
async fn list_pending_imports_envelope_guard() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/import/state"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"items": []})))
        .mount(&mock)
        .await;

    let result = make_server(client)
        .list_pending_imports_impl(ListPendingImportsParams {})
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        result.is_error.unwrap_or(false),
        "wrong response shape must surface as a tool error; got: {text}"
    );
    assert!(text.contains("Failed to parse"), "got: {text}");
}

// ===== list_rejections =====

fn sample_rejection_record(id: &str) -> RejectionRecord {
    RejectionRecord {
        id: id.into(),
        content: "Low quality content.".into(),
        source_agent: Some("test-agent".into()),
        rejection_reason: "low_quality".into(),
        rejection_detail: Some("Quality score below threshold.".into()),
        similarity_score: None,
        similar_to_source_id: None,
        created_at: 1_715_000_000,
    }
}

#[tokio::test]
async fn list_rejections_happy_path() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/rejections"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(vec![sample_rejection_record("rej_abc1")]),
        )
        .mount(&mock)
        .await;

    let result = make_server(client)
        .list_rejections_impl(ListRejectionsParams {
            limit: None,
            reason: None,
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(text.starts_with("1 rejection(s)"), "got: {text}");
    assert!(text.contains("rej_abc1"), "got: {text}");
    assert!(text.contains("low_quality"), "got: {text}");
}

#[tokio::test]
async fn list_rejections_envelope_guard() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/rejections"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": []})))
        .mount(&mock)
        .await;

    let result = make_server(client)
        .list_rejections_impl(ListRejectionsParams {
            limit: None,
            reason: None,
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        result.is_error.unwrap_or(false),
        "envelope-wrapped response must surface as tool error; got: {text}"
    );
    assert!(text.contains("Failed to parse"), "got: {text}");
}

#[tokio::test]
async fn list_rejections_passes_query_params() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/rejections"))
        .respond_with(ResponseTemplate::new(200).set_body_json(Vec::<RejectionRecord>::new()))
        .mount(&mock)
        .await;

    make_server(client)
        .list_rejections_impl(ListRejectionsParams {
            limit: Some(30),
            reason: Some("duplicate".into()),
        })
        .await
        .unwrap();

    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1);
    let url = received[0].url.to_string();
    assert!(url.contains("limit=30"), "got: {url}");
    assert!(url.contains("reason=duplicate"), "got: {url}");
}

// ===== list_pending_revisions =====

use wenlan_mcp::tools::ListPendingRevisionsParams;
use wenlan_types::responses::PendingRevisionItem;

fn sample_pending_revision_item(target: &str, rev: &str) -> PendingRevisionItem {
    PendingRevisionItem {
        target_source_id: target.into(),
        revision_source_id: rev.into(),
        revision_content: "Revised body".into(),
        source_agent: Some("claude-code".into()),
        last_modified: 1_715_000_000,
        target_kind: wenlan_types::responses::RevisionTargetKind::Memory,
        grounded_in: None,
    }
}

#[tokio::test]
async fn list_pending_revisions_happy_path() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/pending-revisions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(vec![sample_pending_revision_item("mem_target", "mem_rev")]),
        )
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .list_pending_revisions_impl(ListPendingRevisionsParams { limit: None })
        .await
        .expect("list_pending_revisions_impl failed");

    let text = text_of(&result);
    assert!(
        text.contains("mem_target"),
        "target_source_id must appear in output: {text}"
    );
    assert!(
        text.contains("Revised body"),
        "revision_content must appear in output: {text}"
    );
}

#[tokio::test]
async fn list_pending_revisions_envelope_guard() {
    // Daemon must return target_source_id, not target.
    // Wrong key: "target" instead of "target_source_id". Typed deserialization
    // must fail loud without echoing daemon response details. This is the C1 regression guard.
    let wrong = serde_json::json!([
        {
            "target": "mem_t",
            "revision_source_id": "mem_r",
            "revision_content": "x",
            "source_agent": null,
            "last_modified": 0
        }
    ]);
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/pending-revisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&wrong))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .list_pending_revisions_impl(ListPendingRevisionsParams { limit: None })
        .await
        .expect("list_pending_revisions_impl returned Err unexpectedly");

    let text = text_of(&result);
    assert!(
        result.is_error.unwrap_or(false),
        "wrong key 'target' instead of 'target_source_id' must surface as tool error; got: {text}"
    );
    assert!(
        text.contains("Failed to parse"),
        "error message must mention parse failure; got: {text}"
    );
}

#[tokio::test]
async fn list_pending_revisions_passes_limit() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/pending-revisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(Vec::<PendingRevisionItem>::new()))
        .mount(&mock)
        .await;

    let server = make_server(client);
    server
        .list_pending_revisions_impl(ListPendingRevisionsParams { limit: Some(25) })
        .await
        .expect("list_pending_revisions_impl failed");

    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1);
    let url = received[0].url.to_string();
    assert!(
        url.contains("limit=25"),
        "expected limit=25 in query; got: {url}"
    );
}

#[tokio::test]
async fn list_pending_revisions_http_500() {
    let (mock, client) = setup().await;
    Mock::given(method("GET"))
        .and(path("/api/memory/pending-revisions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .list_pending_revisions_impl(ListPendingRevisionsParams { limit: None })
        .await
        .expect("list_pending_revisions_impl must not return Err on HTTP 500");

    assert!(
        result.is_error.unwrap_or(false),
        "HTTP 500 must surface as tool error"
    );
    let text = text_of(&result);
    assert!(
        text.contains("500"),
        "error message must mention HTTP 500; got: {text}"
    );
}

// ===== WenlanClient::post_empty =====

#[tokio::test]
async fn origin_client_post_empty_uses_post_verb() {
    let (mock, client) = setup().await;
    let response = DeleteResponse { deleted: true };
    Mock::given(method("POST"))
        .and(path("/api/memory/confirm/mem_abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let _: DeleteResponse = client
        .post_empty("/api/memory/confirm/mem_abc")
        .await
        .expect("post_empty should succeed");

    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 request");
    assert_eq!(
        received[0].method.as_str(),
        "POST",
        "expected POST verb, got: {}",
        received[0].method
    );
    assert!(
        received[0].body.is_empty(),
        "expected empty request body, got {} bytes",
        received[0].body.len()
    );
}

#[tokio::test]
async fn origin_client_post_empty_forwards_x_agent_name() {
    let mock = MockServer::start().await;
    let client = WenlanClient::new(mock.uri()).with_agent_name("test-agent".into());
    let response = DeleteResponse { deleted: true };
    Mock::given(method("POST"))
        .and(path("/api/memory/confirm/mem_xyz"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let _: DeleteResponse = client
        .post_empty("/api/memory/confirm/mem_xyz")
        .await
        .expect("post_empty should succeed");

    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 request");
    let headers = &received[0].headers;
    let value = headers
        .get("x-agent-name")
        .expect("x-agent-name header must be present");
    assert_eq!(
        value.to_str().expect("header value is valid utf-8"),
        "test-agent",
        "x-agent-name header must equal configured agent name"
    );
}

#[tokio::test]
async fn origin_client_post_empty_deserializes_typed_response() {
    let (mock, client) = setup().await;
    let response = DeleteResponse { deleted: true };
    Mock::given(method("POST"))
        .and(path("/api/memory/confirm/mem_typed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let result: DeleteResponse = client
        .post_empty("/api/memory/confirm/mem_typed")
        .await
        .expect("post_empty should deserialize typed response");

    assert!(
        result.deleted,
        "expected deleted=true in deserialized response"
    );
}

// ===== accept_revision =====

use wenlan_mcp::tools::AcceptRevisionRequest;

#[tokio::test]
async fn accept_revision_happy_path() {
    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_target/accept"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "target_source_id": "mem_target",
            "revision_source_id": "mem_rev",
            "wrote": true,
        })))
        .mount(&mock)
        .await;
    let server = make_server(client);
    let result = server
        .accept_revision_impl(AcceptRevisionRequest {
            target_source_id: "mem_target".into(),
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        text.contains("mem_target"),
        "expected target_source_id in output; got: {text}"
    );
    assert!(
        text.contains("mem_rev"),
        "expected revision_source_id in output; got: {text}"
    );
    assert!(
        text.contains("true"),
        "expected wrote=true in output; got: {text}"
    );
}

#[tokio::test]
async fn accept_revision_envelope_guard_ignores_extra_fields() {
    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_target/accept"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "target_source_id": "mem_target",
            "revision_source_id": "mem_rev",
            "wrote": true,
            "unexpected_field": "should be ignored",
        })))
        .mount(&mock)
        .await;
    let server = make_server(client);
    let result = server
        .accept_revision_impl(AcceptRevisionRequest {
            target_source_id: "mem_target".into(),
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        text.contains("mem_target"),
        "expected target_source_id in output; got: {text}"
    );
}

#[tokio::test]
async fn accept_revision_404() {
    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_missing/accept"))
        .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
        .mount(&mock)
        .await;
    let server = make_server(client);
    let result = server
        .accept_revision_impl(AcceptRevisionRequest {
            target_source_id: "mem_missing".into(),
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        text.to_lowercase().contains("error") || text.contains("404"),
        "expected error signal on 404; got: {text}"
    );
}

#[tokio::test]
async fn accept_revision_forwards_x_agent_name() {
    let mock = MockServer::start().await;
    let client = WenlanClient::new(mock.uri()).with_agent_name("test-agent".into());
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_hdr/accept"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "target_source_id": "mem_hdr",
            "revision_source_id": "mem_rev",
            "wrote": true,
        })))
        .mount(&mock)
        .await;
    let server = make_server(client);
    server
        .accept_revision_impl(AcceptRevisionRequest {
            target_source_id: "mem_hdr".into(),
        })
        .await
        .unwrap();
    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 request");
    let value = received[0]
        .headers
        .get("x-agent-name")
        .expect("x-agent-name header must be present");
    assert_eq!(
        value.to_str().expect("header value is valid utf-8"),
        "test-agent",
        "x-agent-name header must equal configured agent name"
    );
}

// ===== dismiss_revision =====

use wenlan_mcp::tools::DismissRevisionRequest;

#[tokio::test]
async fn dismiss_revision_happy_path() {
    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_target/dismiss"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "target_source_id": "mem_target",
            "wrote": true,
        })))
        .mount(&mock)
        .await;
    let server = make_server(client);
    let result = server
        .dismiss_revision_impl(DismissRevisionRequest {
            target_source_id: "mem_target".into(),
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        text.contains("mem_target"),
        "expected target_source_id in output; got: {text}"
    );
    assert!(
        text.contains("true"),
        "expected wrote=true in output; got: {text}"
    );
}

#[tokio::test]
async fn dismiss_revision_envelope_guard() {
    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_target/dismiss"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "target_source_id": "mem_target",
            "wrote": true,
            "unexpected_field": "should be ignored",
        })))
        .mount(&mock)
        .await;
    let server = make_server(client);
    let result = server
        .dismiss_revision_impl(DismissRevisionRequest {
            target_source_id: "mem_target".into(),
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        text.contains("mem_target"),
        "expected target_source_id in output; got: {text}"
    );
}

#[tokio::test]
async fn dismiss_revision_404() {
    let (mock, client) = setup().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_missing/dismiss"))
        .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
        .mount(&mock)
        .await;
    let server = make_server(client);
    let result = server
        .dismiss_revision_impl(DismissRevisionRequest {
            target_source_id: "mem_missing".into(),
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(
        text.to_lowercase().contains("error") || text.contains("404"),
        "expected error signal on 404; got: {text}"
    );
}

#[tokio::test]
async fn dismiss_revision_forwards_x_agent_name() {
    let mock = MockServer::start().await;
    let client = WenlanClient::new(mock.uri()).with_agent_name("test-agent".into());
    Mock::given(method("POST"))
        .and(path("/api/memory/revision/mem_hdr/dismiss"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "target_source_id": "mem_hdr",
            "wrote": true,
        })))
        .mount(&mock)
        .await;
    let server = make_server(client);
    server
        .dismiss_revision_impl(DismissRevisionRequest {
            target_source_id: "mem_hdr".into(),
        })
        .await
        .unwrap();
    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 request");
    let value = received[0]
        .headers
        .get("x-agent-name")
        .expect("x-agent-name header must be present");
    assert_eq!(
        value.to_str().expect("header value is valid utf-8"),
        "test-agent",
        "x-agent-name header must equal configured agent name"
    );
}
#[tokio::test]
async fn t_list_pending_uses_post_with_confirmed_false() {
    let (mock, client) = setup().await;

    let response = ListMemoriesResponse {
        memories: vec![IndexedFileInfo {
            source_id: "mem_pending1".into(),
            title: "Pending capture".into(),
            source: "memory".into(),
            url: None,
            chunk_count: 1,
            last_modified: 1_000_000,
            summary: None,
            processing: false,
            memory_type: Some("fact".into()),
            space: None,
            source_agent: Some("claude-code".into()),
            confidence: None,
            confirmed: Some(false),
            stability: None,
            pinned: false,
            created_at: 1_000_000,
            content: String::new(),
        }],
    };

    Mock::given(method("POST"))
        .and(path("/api/memory/list"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&mock)
        .await;

    let server = make_server(client);
    let result = server
        .list_pending_impl(ListPendingParams {
            limit: Some(10),
            space: None,
        })
        .await
        .expect("list_pending_impl failed");

    let text = text_of(&result);
    assert!(
        text.contains("mem_pending1"),
        "expected source_id in output; got: {text}"
    );

    // Verify the request used POST with confirmed=false in the body
    let body = captured_body(&mock).await;
    assert_eq!(
        body["confirmed"],
        serde_json::json!(false),
        "list_pending must send confirmed=false in POST body; got: {body}"
    );
}
