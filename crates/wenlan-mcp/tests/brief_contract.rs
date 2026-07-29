// SPDX-License-Identifier: Apache-2.0

use rmcp::model::{CallToolResult, RawContent};
use wenlan_mcp::{
    client::WenlanClient,
    tools::{BriefParams, TransportMode, WenlanMcpServer},
};
use wenlan_types::{
    Brief, BriefItem, BriefItemState, BriefReadResponse, BriefReadState, BriefRelatedContext,
};
use wiremock::{
    matchers::{body_json, method, path},
    Mock, MockServer, ResponseTemplate,
};

fn make_server(mock: &MockServer) -> WenlanMcpServer {
    WenlanMcpServer::new(
        WenlanClient::new(mock.uri()),
        TransportMode::Stdio,
        "test-agent".into(),
        None,
    )
}

fn text_of(result: &CallToolResult) -> &str {
    result
        .content
        .iter()
        .find_map(|content| match &content.raw {
            RawContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .expect("expected text content")
}

fn ready_response(topic: Option<&str>) -> BriefReadResponse {
    BriefReadResponse {
        state: BriefReadState::Ready,
        space: Some("wenlan".into()),
        brief: Some(Brief {
            space_id: "space-1".into(),
            space: "wenlan".into(),
            last_session_summary: "MCP work is next.".into(),
            last_handoff_at: Some(1_785_196_800),
            version: 7,
            active: vec![BriefItem {
                id: "item-1".into(),
                text: "Replace the public context tool.".into(),
                state: BriefItemState::Active,
                added_at: 1_785_110_400,
                gate: Some("MCP tests pass".into()),
                version: 3,
            }],
            backlog: Vec::new(),
        }),
        related_context: topic.map(|query| BriefRelatedContext {
            query: query.into(),
            results: Vec::new(),
        }),
    }
}

#[tokio::test]
async fn no_topic_reads_only_the_complete_brief_and_preserves_versions() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/brief"))
        .and(body_json(serde_json::json!({
            "topic": null,
            "space": "wenlan"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ready_response(None)))
        .mount(&mock)
        .await;

    let result = make_server(&mock)
        .brief_impl(BriefParams {
            topic: None,
            space: Some("wenlan".into()),
        })
        .await
        .expect("brief read succeeds");
    let text = text_of(&result);

    assert!(text.starts_with("Space Brief:\n"));
    assert!(text.contains("\"id\": \"item-1\""));
    assert!(text.contains("\"version\": 3"));
    assert!(!text.contains("Related Context"));
}

#[tokio::test]
async fn topic_appends_separately_labeled_same_space_context() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/brief"))
        .and(body_json(serde_json::json!({
            "topic": "release readiness",
            "space": "wenlan"
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(ready_response(Some("release readiness"))),
        )
        .mount(&mock)
        .await;

    let result = make_server(&mock)
        .brief_impl(BriefParams {
            topic: Some("release readiness".into()),
            space: Some("wenlan".into()),
        })
        .await
        .expect("brief topic read succeeds");
    let text = text_of(&result);

    assert!(text.contains("Space Brief:"));
    assert!(text.contains("Related Context (query: release readiness):"));
}

#[tokio::test]
async fn missing_brief_explains_that_reads_do_not_create_it() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/brief"))
        .respond_with(ResponseTemplate::new(200).set_body_json(BriefReadResponse {
            state: BriefReadState::BriefNotCreated,
            space: Some("wenlan".into()),
            brief: None,
            related_context: None,
        }))
        .mount(&mock)
        .await;

    let result = make_server(&mock)
        .brief_impl(BriefParams {
            topic: None,
            space: Some("wenlan".into()),
        })
        .await
        .expect("missing brief is a successful read state");

    assert_eq!(
        text_of(&result),
        "No Brief exists for Space `wenlan` yet. Reads never create one; the first handoff update can."
    );
}
