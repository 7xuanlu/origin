// SPDX-License-Identifier: Apache-2.0
mod common;

use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::Message;
use wenlan_server::{router::build_router, state::ServerState};

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ClientMessage<'a> {
    #[serde(rename = "subscribe")]
    Subscribe { channels: &'a [&'a str] },
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "index_progress")]
    IndexProgress { data: IndexProgressData },
    #[serde(rename = "ingest_complete")]
    IngestComplete { data: IngestCompleteData },
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, Deserialize)]
struct IndexProgressData {
    files_indexed: u64,
    files_total: u64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct IngestCompleteData {
    document_id: String,
    chunks: usize,
}

#[tokio::test]
async fn moved_websocket_route_preserves_typed_success_and_error_contracts() {
    let app = build_router(Arc::new(RwLock::new(ServerState::default())));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    let (mut socket, response) =
        tokio_tungstenite::connect_async(format!("ws://{address}/ws/updates"))
            .await
            .unwrap();
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SWITCHING_PROTOCOLS
    );

    let subscribe = ClientMessage::Subscribe {
        channels: &["index_progress"],
    };
    socket
        .send(Message::Text(
            serde_json::to_string(&subscribe).unwrap().into(),
        ))
        .await
        .unwrap();

    let message = timeout(Duration::from_secs(2), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Message::Text(message) = message else {
        panic!("subscription returned a non-text WebSocket message");
    };
    let response: ServerMessage = serde_json::from_str(&message).unwrap();
    match response {
        ServerMessage::IndexProgress { data } => {
            assert_eq!(data.files_indexed, 0);
            assert_eq!(data.files_total, 0);
        }
        other => panic!("subscription returned the wrong response: {other:?}"),
    }

    socket.send(Message::Text("not json".into())).await.unwrap();
    let message = timeout(Duration::from_secs(2), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Message::Text(message) = message else {
        panic!("invalid input returned a non-text WebSocket message");
    };
    let response: ServerMessage = serde_json::from_str(&message).unwrap();
    match response {
        ServerMessage::Error { message } => {
            assert!(message.contains("Invalid message"));
        }
        other => panic!("invalid input returned the wrong response: {other:?}"),
    }

    socket.close(None).await.unwrap();
    shutdown_tx.send(()).unwrap();
    timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap();
}
