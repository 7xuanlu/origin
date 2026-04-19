// SPDX-License-Identifier: Apache-2.0
//! Types for the chat-export import endpoint.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportChatExportRequest {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportChatExportResponse {
    pub import_id: String,
    pub vendor: String,
    pub conversations_total: usize,
    pub conversations_new: usize,
    pub conversations_skipped_existing: usize,
    pub memories_stored: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingImport {
    pub id: String,
    pub vendor: String,
    pub stage: String,
    pub source_path: String,
    pub processed_conversations: i64,
    pub total_conversations: Option<i64>,
}
