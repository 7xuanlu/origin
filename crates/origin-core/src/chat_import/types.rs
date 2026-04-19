// SPDX-License-Identifier: Apache-2.0
//! Shared types for the chat conversation importer.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Vendor of a chat conversation export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Vendor {
    ChatGpt,
    Claude,
}

impl Vendor {
    pub fn source_id_prefix(&self) -> &'static str {
        match self {
            Vendor::ChatGpt => "import_chatgpt_",
            Vendor::Claude => "import_claude_",
        }
    }

    pub fn build_source_id(&self, external_id: &str) -> String {
        format!("{}{}", self.source_id_prefix(), external_id)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Vendor::ChatGpt => "chatgpt",
            Vendor::Claude => "claude",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "chatgpt" => Some(Vendor::ChatGpt),
            "claude" => Some(Vendor::Claude),
            _ => None,
        }
    }
}

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

/// One message in a parsed conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedMessage {
    pub role: MessageRole,
    pub content: String,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// One conversation parsed from a vendor export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedConversation {
    pub external_id: String,
    pub vendor: Vendor,
    pub title: Option<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub summary: Option<String>,
    pub messages: Vec<ParsedMessage>,
}

/// Errors emitted by the chat import subsystem.
#[derive(Debug, Error)]
pub enum ImportError {
    #[error("Failed to read zip archive: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("Failed to read file '{path}' from archive: {source}")]
    ZipEntryRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Invalid export format: {reason}")]
    InvalidFormat { reason: String },

    #[error("JSON parse error in '{path}': {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("No parser could handle the provided archive")]
    NoParserMatched,

    #[error("Database error: {0}")]
    Db(#[from] crate::error::OriginError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A parser for a specific vendor's conversation export format.
pub trait ChatExportParser: Send + Sync {
    fn vendor(&self) -> Vendor;
    fn can_parse(&self, archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>) -> bool;
    fn parse(
        &self,
        archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    ) -> Result<Vec<ParsedConversation>, ImportError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_source_id_prefix_chatgpt() {
        assert_eq!(Vendor::ChatGpt.source_id_prefix(), "import_chatgpt_");
    }

    #[test]
    fn vendor_source_id_prefix_claude() {
        assert_eq!(Vendor::Claude.source_id_prefix(), "import_claude_");
    }

    #[test]
    fn vendor_build_source_id() {
        let id = Vendor::Claude.build_source_id("abc-123");
        assert_eq!(id, "import_claude_abc-123");
    }

    #[test]
    fn message_role_user_maps_both_vendors() {
        let r: MessageRole = MessageRole::User;
        assert!(matches!(r, MessageRole::User));
    }

    #[test]
    fn parsed_conversation_serializable() {
        let conv = ParsedConversation {
            external_id: "conv-1".into(),
            vendor: Vendor::Claude,
            title: Some("Hello".into()),
            created_at: None,
            summary: None,
            messages: vec![],
        };
        let json = serde_json::to_string(&conv).unwrap();
        let back: ParsedConversation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.external_id, "conv-1");
        assert_eq!(back.vendor, Vendor::Claude);
    }

    #[test]
    fn import_error_display() {
        let e = ImportError::InvalidFormat {
            reason: "not a zip".into(),
        };
        assert!(format!("{e}").contains("not a zip"));
    }
}
