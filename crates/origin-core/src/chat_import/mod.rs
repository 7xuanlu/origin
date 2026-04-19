// SPDX-License-Identifier: Apache-2.0
//! Chat conversation history importer.

pub mod bulk_ingest;
pub mod chatgpt;
pub mod claude;
pub mod types;

pub use types::{
    ChatExportParser, ImportError, MessageRole, ParsedConversation, ParsedMessage, Vendor,
};

use std::io::Cursor;

/// Combine a chronological list of individual messages into user-assistant
/// memory pairs. Each pair becomes one message with role=Assistant (the
/// response) and combined content.
///
/// An unpaired trailing user message (no assistant response yet) is dropped
/// for v1 — see the spec's "Memory splitting strategy" section.
pub(crate) fn pair_messages(msgs: Vec<ParsedMessage>) -> Vec<ParsedMessage> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < msgs.len() {
        if msgs[i].role == MessageRole::User
            && i + 1 < msgs.len()
            && msgs[i + 1].role == MessageRole::Assistant
        {
            let user = &msgs[i];
            let asst = &msgs[i + 1];
            out.push(ParsedMessage {
                role: MessageRole::Assistant,
                content: format!("User: {}\nAssistant: {}", user.content, asst.content),
                created_at: asst.created_at,
            });
            i += 2;
        } else {
            // Unpaired message (user without response, or assistant without
            // preceding user) — skip.
            i += 1;
        }
    }
    out
}

/// Result of a successful parse — tagged with the vendor that matched.
pub struct ParsedConversationBatch {
    pub vendor: Vendor,
    pub conversations: Vec<ParsedConversation>,
}

/// Attempt to parse a chat-export ZIP by trying each known parser in sequence.
pub fn dispatch_parse(zip_bytes: &[u8]) -> Result<ParsedConversationBatch, ImportError> {
    let parsers: Vec<Box<dyn ChatExportParser>> = vec![
        Box::new(claude::ClaudeParser::new()),
        Box::new(chatgpt::ChatGptParser::new()),
    ];

    for parser in parsers {
        let cursor = Cursor::new(zip_bytes);
        let mut archive = zip::ZipArchive::new(cursor)?;
        if parser.can_parse(&mut archive) {
            let cursor = Cursor::new(zip_bytes);
            let mut archive = zip::ZipArchive::new(cursor)?;
            let conversations = parser.parse(&mut archive)?;
            return Ok(ParsedConversationBatch {
                vendor: parser.vendor(),
                conversations,
            });
        }
    }

    Err(ImportError::NoParserMatched)
}

#[cfg(test)]
mod dispatcher_tests {
    use super::*;
    use std::io::Write;

    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options: zip::write::FileOptions<()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, data) in entries {
                zip.start_file(*name, options).unwrap();
                zip.write_all(data).unwrap();
            }
            zip.finish().unwrap();
        }
        buf
    }

    #[test]
    fn dispatcher_returns_no_parser_matched_for_unknown_zip() {
        let zip_bytes = make_zip(&[("random.txt", b"hello")]);
        let result = dispatch_parse(&zip_bytes);
        assert!(matches!(result, Err(ImportError::NoParserMatched)));
    }

    #[test]
    fn dispatcher_picks_chatgpt_for_chatgpt_zip() {
        // An empty conversations.json is fine — parse() will return Ok(vec![]).
        // The test just verifies that the dispatcher selects ChatGptParser.
        let zip_bytes = make_zip(&[
            ("conversations.json", b"[]"),
            ("message_feedback.json", b"[]"),
        ]);
        let result =
            dispatch_parse(&zip_bytes).expect("dispatch_parse should succeed for ChatGPT zip");
        assert_eq!(
            result.vendor,
            Vendor::ChatGpt,
            "dispatcher should pick ChatGPT vendor for a zip containing message_feedback.json"
        );
        assert_eq!(
            result.conversations.len(),
            0,
            "empty conversations.json should yield zero conversations"
        );
    }
}
