// SPDX-License-Identifier: Apache-2.0
use crate::state::ServerState;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::Response,
};
use origin_types::sources::RawDocument;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

// ===== WebSocket Message Types =====

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum WsClientMessage {
    #[serde(rename = "subscribe")]
    Subscribe { channels: Vec<String> },
    #[serde(rename = "ingest")]
    Ingest { data: IngestData },
}

#[derive(Debug, Deserialize)]
pub struct IngestData {
    pub source: String,
    pub source_id: String,
    pub title: Option<String>,
    pub content: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum WsServerMessage {
    #[serde(rename = "index_progress")]
    IndexProgress { data: IndexProgressData },
    #[serde(rename = "ingest_complete")]
    IngestComplete { data: IngestCompleteData },
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, Serialize)]
pub struct IndexProgressData {
    pub files_indexed: u64,
    pub files_total: u64,
}

#[derive(Debug, Serialize)]
pub struct IngestCompleteData {
    pub document_id: String,
    pub chunks: usize,
}

// ===== WebSocket Handler =====

/// WS /ws/updates
pub async fn handle_ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
}

async fn handle_ws_connection(mut socket: WebSocket, state: Arc<RwLock<ServerState>>) {
    tracing::info!("WebSocket client connected");

    #[allow(unused_assignments)]
    let mut subscribed_channels: Vec<String> = vec![];

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(msg) => msg,
            Err(e) => {
                tracing::warn!("WebSocket receive error: {}", e);
                break;
            }
        };

        match msg {
            Message::Text(text) => match serde_json::from_str::<WsClientMessage>(&text) {
                Ok(WsClientMessage::Subscribe { channels }) => {
                    subscribed_channels = channels;
                    tracing::info!("WebSocket client subscribed to: {:?}", subscribed_channels);

                    if subscribed_channels.contains(&"index_progress".to_string()) {
                        let progress = get_index_progress(&state).await;
                        let msg = WsServerMessage::IndexProgress { data: progress };
                        if let Ok(json) = serde_json::to_string(&msg) {
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                Ok(WsClientMessage::Ingest { data }) => {
                    let result = handle_ws_ingest(&state, data).await;
                    let msg = match result {
                        Ok(complete) => WsServerMessage::IngestComplete { data: complete },
                        Err(e) => WsServerMessage::Error {
                            message: e.to_string(),
                        },
                    };
                    if let Ok(json) = serde_json::to_string(&msg) {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let msg = WsServerMessage::Error {
                        message: format!("Invalid message: {}", e),
                    };
                    if let Ok(json) = serde_json::to_string(&msg) {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                }
            },
            Message::Close(_) => {
                tracing::info!("WebSocket client disconnected");
                break;
            }
            _ => {}
        }
    }

    tracing::info!("WebSocket connection closed");
}

async fn get_index_progress(state: &Arc<RwLock<ServerState>>) -> IndexProgressData {
    let s = state.read().await;

    let files_indexed = if let Some(db) = &s.db {
        db.count().await.unwrap_or(0)
    } else {
        0
    };

    IndexProgressData {
        files_indexed,
        files_total: 0,
    }
}

async fn handle_ws_ingest(
    state: &Arc<RwLock<ServerState>>,
    data: IngestData,
) -> Result<IngestCompleteData, String> {
    let document_id = data.source_id.clone();

    let doc = RawDocument {
        source: data.source,
        source_id: data.source_id,
        title: data.title.unwrap_or_else(|| "Untitled".to_string()),
        summary: None,
        content: data.content,
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        metadata: std::collections::HashMap::new(),
        memory_type: None,
        source_agent: None,
        domain: None,
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };

    let s = state.read().await;
    let db = s.db.as_ref().ok_or("Database not initialized")?;

    let chunks = db
        .upsert_documents(vec![doc])
        .await
        .map_err(|e| e.to_string())?;

    Ok(IngestCompleteData {
        document_id,
        chunks,
    })
}
