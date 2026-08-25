// SPDX-License-Identifier: Apache-2.0
//! Daemon-authoritative origin classification (KG unified-model spec v3 §5.2 /
//! §5.6; close plan `docs/plans/2026-08-04-kg-revamp-close-plan.md` Part A
//! items 1 + 2).
//!
//! Grounding promotion has to know whether a memory came from OUTSIDE the
//! system (a document a human put in front of it) or from INSIDE (something
//! the system wrote). That question used to be answered by comparing
//! `memories.source_agent` against the literal string `"folder"`, which was
//! wrong twice over:
//!
//! * **Too narrow** — an Obsidian vault import (`source_agent="obsidian"`) is
//!   a genuine document ingest and could never ground, and every new connector
//!   re-broke the gate by inventing its own string.
//! * **Spoofable** — `/api/memory/store` persisted the client-supplied
//!   `source_agent` verbatim, so any agent could send `"folder"` and make its
//!   own extracted relations promotion-eligible.
//!
//! The fix is one classification the daemon records at the ingest boundary
//! (`memories.origin_class`, migration 112) which promotion reads instead of
//! string-matching, plus a wire guard that stops a request from selecting an
//! origin-bearing `source_agent` at all. The two halves are load-bearing
//! together: the classifier is only trustworthy because
//! [`normalize_wire_source_agent`] keeps the reserved values internal to the
//! daemon's own ingest paths.
//!
//! **Adding a connector.** A new document-ingest connector adds its
//! `source_agent` to [`DOCUMENT_INGEST_SOURCE_AGENTS`]. That single edit both
//! makes its documents groundable and makes the string un-claimable from the
//! wire; nothing else changes. This is the mechanism the spec left a vacuum
//! for, and it replaces the old "invent a new string" connector guidance in
//! `sources/obsidian.rs`.

/// Where a memory came from, as decided by the daemon at the ingest boundary.
///
/// Stored in `memories.origin_class` as [`OriginClass::as_str`] and read back
/// with [`OriginClass::from_stored`], which fails closed to
/// [`OriginClass::Generated`] on NULL or an unrecognised value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginClass {
    /// A document ingested from outside the system — folder watcher, vault
    /// import, file importer. The only class the grounding sweep promotes.
    DocumentIngest,
    /// A statement a human made directly. RESERVED: no code path mints it
    /// today, because spec §5.6 makes every agent capture `Generated`. It
    /// exists so a human-authored surface has an honest class to land on
    /// instead of borrowing `DocumentIngest`.
    HumanCapture,
    /// Anything the system produced about itself: agent captures, distilled
    /// prose, reconcile rewrites, recaps. Never groundable (invariants
    /// #11/#13) — this is the class that stops the graph voting for its own
    /// output.
    Generated,
}

impl OriginClass {
    /// The stored spelling. Stable — it is persisted, so changing one of these
    /// strings is a migration, not a rename.
    pub fn as_str(self) -> &'static str {
        match self {
            OriginClass::DocumentIngest => "document_ingest",
            OriginClass::HumanCapture => "human_capture",
            OriginClass::Generated => "generated",
        }
    }

    /// Read a stored `memories.origin_class`. NULL (a row written by a build
    /// older than migration 112) and any unrecognised value both fail closed
    /// to `Generated`, so an unclassified row can never ground.
    pub fn from_stored(stored: Option<&str>) -> Self {
        match stored {
            Some("document_ingest") => OriginClass::DocumentIngest,
            Some("human_capture") => OriginClass::HumanCapture,
            _ => OriginClass::Generated,
        }
    }
}

/// The `source_agent` values that mean "this row is a document ingest".
///
/// Doubles as the reserved list the wire guard enforces: exactly the values
/// that decide grounding are the values a request may not select. `reconcile`
/// is deliberately absent — a doc-reconcile revision is LLM-rewritten prose
/// about a document, not the document, so it is `Generated` and unpromotable.
pub const DOCUMENT_INGEST_SOURCE_AGENTS: &[&str] = &["folder", "obsidian"];

/// Whether `source_agent` names a document-ingest path. Case- and
/// whitespace-insensitive so the guard and the classifier can never disagree
/// about `" Folder"`.
pub fn is_document_ingest_source_agent(source_agent: &str) -> bool {
    let canonical = source_agent.trim().to_ascii_lowercase();
    DOCUMENT_INGEST_SOURCE_AGENTS.contains(&canonical.as_str())
}

/// Strip a wire-claimed page-revision marker out of `structured_fields`.
///
/// A memory row whose `structured_fields` carry `revision_kind: "page_write"`
/// and `target_kind: "page"` is a page revision card: accepting it rewrites a
/// page's prose, and dismissing it deletes the row. Only
/// `post_write::stage_page_revision_card` may mint one. `structured_fields` is
/// persisted verbatim from `POST /api/memory/store`, so without this guard a
/// request could hand itself that meaning, the same way a request could once
/// hand itself a document-ingest origin. Daemon-authoritative, like
/// `source_agent` above.
///
/// Returns the sanitised value and whether a claim was dropped. Only the four
/// card keys are removed; anything else the caller stored is kept.
pub fn strip_wire_page_revision_marker(
    fields: Option<serde_json::Value>,
) -> (Option<serde_json::Value>, bool) {
    let Some(serde_json::Value::Object(mut map)) = fields else {
        return (fields, false);
    };
    let claims_card = map.get("revision_kind").and_then(|v| v.as_str()) == Some("page_write")
        || map.get("target_kind").and_then(|v| v.as_str()) == Some("page");
    if !claims_card {
        return (Some(serde_json::Value::Object(map)), false);
    }
    for key in [
        "revision_kind",
        "target_kind",
        "revises_page",
        "page_version",
    ] {
        map.remove(key);
    }
    (Some(serde_json::Value::Object(map)), true)
}

/// The daemon's origin verdict for a row about to be written.
///
/// Called from the `RawDocument` ingest path (`db.rs upsert_documents`), which
/// is where every external document and every agent capture enters. It is not
/// the only writer that touches `memories` — the chat-export importer, the
/// episode backfill, and the in-place memory rewrite each have their own INSERT
/// — so those pin `origin_class` explicitly rather than inheriting a
/// classification, and no consumer re-derives one. Nothing anywhere reads
/// `source_agent` to decide grounding.
pub fn classify_origin(source_agent: Option<&str>) -> OriginClass {
    match source_agent {
        Some(agent) if is_document_ingest_source_agent(agent) => OriginClass::DocumentIngest,
        _ => OriginClass::Generated,
    }
}

/// Strip an origin-bearing `source_agent` off a wire request (spec §5.6: no
/// wire request may select anything that decides grounding).
///
/// Returns `(value to persist, the rejected claim)`. Normalising rather than
/// rejecting keeps existing clients working; the rejected claim is returned so
/// the caller can log it loudly rather than swallow a spoof attempt.
///
/// Dropping the claim — rather than keeping the string and relying on
/// `origin_class` alone — matters because `source_agent = 'folder'` is read as
/// "this is a document" by more than grounding: page genesis refuses to seed
/// from it (`db.rs` `can_seed_page`) and doc-reconcile treats it as the
/// authoritative side of a capture/document contradiction. A row that keeps a
/// spoofed string would still buy those privileges.
pub fn normalize_wire_source_agent(claimed: Option<String>) -> (Option<String>, Option<String>) {
    match claimed {
        Some(agent) if is_document_ingest_source_agent(&agent) => (None, Some(agent)),
        other => (other, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_ingest_agents_classify_as_external() {
        assert_eq!(
            classify_origin(Some("folder")),
            OriginClass::DocumentIngest,
            "the folder watcher is the original external origin"
        );
        assert_eq!(
            classify_origin(Some("obsidian")),
            OriginClass::DocumentIngest,
            "an Obsidian vault import is a document ingest too — the narrowing this fixes"
        );
    }

    #[test]
    fn everything_else_is_generated() {
        for agent in [
            Some("claude-code"),
            Some("cursor"),
            Some("reconcile"),
            Some("chatgpt"),
            Some(""),
            None,
        ] {
            assert_eq!(
                classify_origin(agent),
                OriginClass::Generated,
                "{agent:?} must not be groundable"
            );
        }
    }

    #[test]
    fn reserved_matching_ignores_case_and_padding() {
        assert!(is_document_ingest_source_agent("  FoLdEr "));
        assert_eq!(
            classify_origin(Some("  FoLdEr ")),
            OriginClass::DocumentIngest,
            "guard and classifier must agree on the same canonical form"
        );
    }

    #[test]
    fn wire_guard_drops_reserved_claims_and_reports_them() {
        let (kept, rejected) = normalize_wire_source_agent(Some("folder".to_string()));
        assert_eq!(kept, None);
        assert_eq!(rejected.as_deref(), Some("folder"));

        let (kept, rejected) = normalize_wire_source_agent(Some("OBSIDIAN".to_string()));
        assert_eq!(kept, None);
        assert_eq!(rejected.as_deref(), Some("OBSIDIAN"));
    }

    #[test]
    fn wire_guard_leaves_ordinary_agents_alone() {
        let (kept, rejected) = normalize_wire_source_agent(Some("claude-code".to_string()));
        assert_eq!(kept.as_deref(), Some("claude-code"));
        assert_eq!(rejected, None);

        let (kept, rejected) = normalize_wire_source_agent(None);
        assert_eq!(kept, None);
        assert_eq!(rejected, None);
    }

    #[test]
    fn unclassified_rows_fail_closed() {
        assert_eq!(OriginClass::from_stored(None), OriginClass::Generated);
        assert_eq!(
            OriginClass::from_stored(Some("nonsense")),
            OriginClass::Generated
        );
        assert_eq!(
            OriginClass::from_stored(Some("document_ingest")),
            OriginClass::DocumentIngest
        );
        assert_eq!(
            OriginClass::from_stored(Some("human_capture")),
            OriginClass::HumanCapture
        );
    }

    #[test]
    fn stored_spellings_round_trip() {
        for class in [
            OriginClass::DocumentIngest,
            OriginClass::HumanCapture,
            OriginClass::Generated,
        ] {
            assert_eq!(OriginClass::from_stored(Some(class.as_str())), class);
        }
    }

    #[test]
    fn wire_page_revision_marker_is_stripped_but_other_fields_survive() {
        let claimed = serde_json::json!({
            "revision_kind": "page_write",
            "target_kind": "page",
            "revises_page": "page_victim",
            "page_version": 3,
            "mine": "kept"
        });
        let (sanitised, dropped) = strip_wire_page_revision_marker(Some(claimed));
        assert!(dropped, "a claimed card must be reported as dropped");
        let out = sanitised.expect("the object survives, minus the markers");
        for key in [
            "revision_kind",
            "target_kind",
            "revises_page",
            "page_version",
        ] {
            assert!(out.get(key).is_none(), "{key} must be removed");
        }
        assert_eq!(out.get("mine").and_then(|v| v.as_str()), Some("kept"));
    }

    #[test]
    fn ordinary_structured_fields_pass_through_untouched() {
        let plain = serde_json::json!({ "grounded_in": "doc_1", "page_version": 2 });
        let (sanitised, dropped) = strip_wire_page_revision_marker(Some(plain.clone()));
        assert!(!dropped, "no card claim, nothing dropped");
        assert_eq!(sanitised, Some(plain));
        let (none, dropped_none) = strip_wire_page_revision_marker(None);
        assert!(none.is_none() && !dropped_none);
    }
}
