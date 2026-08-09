//! E2E: MCP layer respects space scoping; legacy `domain` JSON key still deserializes.
//!
//! ## Test 1: `mcp_capture_and_recall_respects_space`
//! Uses wiremock to stand in for the daemon (same pattern as `type_contract.rs`).
//! Verifies that `space` is forwarded on both the capture and the recall requests,
//! and that results returned for the wrong space do not appear in the output.
//!
//! ## Test 2: `mcp_legacy_domain_key_still_works`
//! Pure serde deserialization: JSON with `domain` key must map to `space` via the
//! `#[serde(alias = "domain")]` on `CaptureParams`, `RecallParams`, and also exercises
//! the tool dispatch path to confirm the alias survives the MCP parameter layer.
//!
//! Task 9 already covers `ContextParams` + the in-module `legacy_domain_alias_still_deserializes`
//! unit test. This file extends coverage to `CaptureParams` and `RecallParams` and also
//! verifies the alias round-trips through the `capture_impl` / `recall_impl` dispatch path.

use rmcp::model::RawContent;
use wenlan_mcp::client::WenlanClient;
use wenlan_mcp::tools::{CaptureParams, RecallParams, TransportMode, WenlanMcpServer};
use wenlan_types::memory::SearchResult;
use wenlan_types::responses::{SearchMemoryResponse, StoreMemoryResponse};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_server(client: WenlanClient) -> WenlanMcpServer {
    WenlanMcpServer::new(client, TransportMode::Stdio, "test-agent".into(), None)
}

fn text_of(result: &rmcp::model::CallToolResult) -> String {
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

fn alpha_search_result() -> SearchResult {
    SearchResult {
        id: "1".into(),
        content: "alpha fact one".into(),
        source: "memory".into(),
        source_id: "mem_alpha1".into(),
        title: "alpha title".into(),
        url: None,
        chunk_index: 0,
        last_modified: 0,
        score: 0.9,
        chunk_type: None,
        language: None,
        semantic_unit: None,
        memory_type: Some("fact".into()),
        space: Some("alpha".into()),
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

fn store_response(source_id: &str) -> StoreMemoryResponse {
    StoreMemoryResponse {
        source_id: source_id.into(),
        chunks_created: 1,
        memory_type: "fact".into(),
        entity_id: None,
        quality: None,
        warnings: vec![],
        extraction_method: "none".into(),
        enrichment: String::new(),
        hint: String::new(),
        space: None,
        space_source: None,
        write_outcome: None,
    }
}

/// Test 1: space filter is forwarded through capture and recall, and the daemon's
/// space-filtered response (alpha only) does not contain beta content.
///
/// The wiremock layer here plays the role of the daemon: it accepts the store
/// requests (returning a minimal ok), then on recall returns only alpha results.
/// We assert that:
///   (a) the recall request body carries `space = "alpha"`
///   (b) the rendered output contains alpha content
///   (c) the rendered output does not contain any beta content
///
/// This confirms the MCP layer threads `space` through without stripping or
/// overwriting it, and that `recall_impl` renders exactly what the daemon returns.
#[tokio::test]
async fn mcp_capture_and_recall_respects_space() {
    let mock = MockServer::start().await;

    // Daemon accepts both captures (store endpoint returns ok for each).
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(store_response("mem_alpha1")))
        .expect(2)
        .mount(&mock)
        .await;

    // Recall with space=alpha: daemon returns only the alpha result.
    let search_response = SearchMemoryResponse {
        results: vec![alpha_search_result()],
        took_ms: 5.0,
        supplemental_pages: None,
    };
    Mock::given(method("POST"))
        .and(path("/api/memory/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&search_response))
        .mount(&mock)
        .await;

    let client = WenlanClient::new(mock.uri());
    let server = make_server(client);

    // Capture alpha-space memory.
    server
        .capture_impl(CaptureParams {
            content: "alpha fact one".into(),
            memory_type: None,
            space: Some("alpha".into()),
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl alpha failed");

    // Capture beta-space memory (same store endpoint — space is in the body).
    server
        .capture_impl(CaptureParams {
            content: "beta fact two".into(),
            memory_type: None,
            space: Some("beta".into()),
            entity: None,
            confidence: None,
            supersedes: None,
            structured_fields: None,
            retrieval_cue: None,
        })
        .await
        .expect("capture_impl beta failed");

    // Recall with space=alpha.
    let recall_result = server
        .recall_impl(RecallParams {
            query: "fact".into(),
            limit: None,
            memory_type: None,
            space: Some("alpha".into()),
            rerank: None,
        })
        .await
        .expect("recall_impl failed");

    let text = text_of(&recall_result);

    // Alpha content must appear.
    assert!(
        text.contains("alpha fact one"),
        "alpha result must appear in recall output; got: {text}"
    );

    // Beta content must not appear (daemon filtered it out; MCP layer must not re-add it).
    assert!(
        !text.contains("beta fact two"),
        "beta result must not appear when space=alpha; got: {text}"
    );

    // Verify the recall wire request carried `space = "alpha"`.
    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    // 2 store + 1 search = 3 requests total.
    assert_eq!(received.len(), 3, "expected 2 captures + 1 recall request");

    let recall_body: serde_json::Value =
        serde_json::from_slice(&received[2].body).expect("recall body must be valid JSON");
    assert_eq!(
        recall_body["space"],
        serde_json::json!("alpha"),
        "recall request must carry space=alpha in the wire body"
    );

    // Also verify the two store requests forwarded their respective spaces.
    let alpha_body: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("alpha capture body must be valid JSON");
    assert_eq!(
        alpha_body["space"],
        serde_json::json!("alpha"),
        "first capture must carry space=alpha"
    );
    let beta_body: serde_json::Value =
        serde_json::from_slice(&received[1].body).expect("beta capture body must be valid JSON");
    assert_eq!(
        beta_body["space"],
        serde_json::json!("beta"),
        "second capture must carry space=beta"
    );
}

/// Test 2: pre-0.7.0 clients that send `"domain"` instead of `"space"` still work.
///
/// Task 9 covers `ContextParams` with a unit test in `tools.rs`. This test
/// extends coverage to `CaptureParams` and `RecallParams` (the other two
/// tools that accept a space filter), and also exercises the `capture_impl`
/// dispatch path end-to-end through the tool layer so we know the alias
/// survives MCP parameter deserialization, not just raw serde.
#[tokio::test]
async fn mcp_legacy_domain_key_still_works() {
    // --- Pure serde layer: CaptureParams ---
    let capture_json = r#"{"content": "legacy capture fact", "domain": "alpha"}"#;
    let params: CaptureParams = serde_json::from_str(capture_json)
        .expect("legacy 'domain' key must deserialize for CaptureParams");
    assert_eq!(
        params.space.as_deref(),
        Some("alpha"),
        "domain alias must map to space in CaptureParams"
    );

    // --- Pure serde layer: RecallParams ---
    let recall_json = r#"{"query": "some fact", "domain": "alpha"}"#;
    let rparams: RecallParams = serde_json::from_str(recall_json)
        .expect("legacy 'domain' key must deserialize for RecallParams");
    assert_eq!(
        rparams.space.as_deref(),
        Some("alpha"),
        "domain alias must map to space in RecallParams"
    );

    // --- Tool dispatch path: confirm alias survives capture_impl ---
    // Use wiremock so the impl can complete the HTTP round-trip.
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/memory/store"))
        .respond_with(ResponseTemplate::new(200).set_body_json(store_response("mem_legacy1")))
        .mount(&mock)
        .await;

    let client = WenlanClient::new(mock.uri());
    let server = make_server(client);

    // Deserialize from legacy JSON to simulate a cached MCP client sending `domain`.
    let legacy_params: CaptureParams =
        serde_json::from_str(r#"{"content": "legacy capture fact", "domain": "alpha"}"#)
            .expect("legacy JSON must parse");

    let result = server
        .capture_impl(legacy_params)
        .await
        .expect("capture_impl must succeed with params parsed from legacy domain JSON");

    let text = text_of(&result);
    assert!(
        text.contains("mem_legacy1"),
        "capture_impl must succeed and return the source_id; got: {text}"
    );

    // Verify the wire request sent space=alpha (not null), confirming alias propagated.
    let received = mock
        .received_requests()
        .await
        .expect("wiremock captured no requests");
    assert_eq!(received.len(), 1, "expected exactly 1 store request");
    let body: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("request body must be valid JSON");
    assert_eq!(
        body["space"],
        serde_json::json!("alpha"),
        "space=alpha must appear in the wire request when parsed from legacy domain JSON; body={body}"
    );
}
