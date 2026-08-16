// SPDX-License-Identifier: Apache-2.0
//! The single seam every entity write passes through.
//!
//! There are two production lanes that mint entity shadow pages, and they do
//! not share a code path:
//!
//! | lane | entry | writer |
//! |---|---|---|
//! | ingest and import | `post_write::create_entity`, `importer::resolve_entity_bulk` | `MemoryDB::resolve_or_create_entity` |
//! | ambient backfill | `run_entity_enrichment_slice_inner` | `MemoryDB::commit_entity_enrichment_at_version` |
//!
//! The second lane is the one that produced the junk: it deduplicates the
//! extractor's names, embeds them, and inserts shadow pages directly, never
//! touching `resolve_or_create_entity`. A gate installed on only the first lane
//! would have missed the traffic that caused the incident.
//!
//! [`admit`] is therefore pure and cheap, and both lanes call it at the point
//! where they normalize a name — in the backfill lane, before embeddings are
//! generated, so a refused name costs no model time either.

use crate::kg::entity_name_quality::{self, Reason, Tier, Verdict};
use crate::vocab::entity_type;

/// A name and type that passed the gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted {
    /// The name, trimmed. Otherwise unchanged: casing and punctuation are the
    /// entity's identity.
    pub name: String,
    /// A canonical member of `entity_type_vocabulary`.
    pub entity_type: String,
    /// The raw type string when no band resolved it. `Some` means the caller
    /// should queue a `vocab_promote` proposal so a human sees the new value.
    pub unrecognized_type: Option<String>,
}

/// A name the gate declined to store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    pub tier: Tier,
    pub reason: Reason,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "entity name failed the {} rule ({})",
            self.reason, self.tier
        )
    }
}

/// What [`admit`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    Admit(Admitted),
    Refuse(Refused),
}

impl Admission {
    pub fn admitted(&self) -> Option<&Admitted> {
        match self {
            Admission::Admit(a) => Some(a),
            Admission::Refuse(_) => None,
        }
    }
}

/// Decide whether `name` may become an entity, and canonicalize its type.
///
/// Refusals are logged at debug and counted in
/// [`crate::kg::entity_name_quality::rejected_total`] /
/// [`crate::kg::entity_name_quality::reviewed_total`]. Never panics: every
/// input, including empty and non-UTF-8-adjacent junk, produces a verdict.
pub fn admit(name: &str, raw_type: &str, canonicals: &[String]) -> Admission {
    let trimmed = name.trim();
    let refuse = |reason: Reason| {
        let tier = reason.tier();
        entity_name_quality::record(tier);
        log::debug!("[entity-gate] {tier} {reason}: {trimmed:?} (type {raw_type:?})");
        Admission::Refuse(Refused { tier, reason })
    };

    match entity_name_quality::assess(trimmed) {
        Verdict::Reject(reason) | Verdict::Review(reason) => return refuse(reason),
        Verdict::Accept => {}
    }

    let normalized = entity_type::normalize(raw_type, canonicals);
    Admission::Admit(Admitted {
        name: trimmed.to_string(),
        entity_type: normalized.canonical,
        unrecognized_type: normalized
            .unrecognized
            .then(|| raw_type.trim().to_string())
            .filter(|s| !s.is_empty()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> Vec<String> {
        [
            "concept",
            "technology",
            "project",
            "organization",
            "person",
            "place",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn a_real_entity_is_admitted_with_a_canonical_type() {
        let out = admit("Node.js", "language", &seed());
        let a = out.admitted().expect("Node.js must be admitted");
        assert_eq!(a.name, "Node.js");
        assert_eq!(a.entity_type, "technology");
        assert_eq!(a.unrecognized_type, None);
    }

    #[test]
    fn the_name_is_trimmed_but_never_rewritten() {
        let a = admit("  GPT-4o  ", "Technology", &seed())
            .admitted()
            .cloned()
            .expect("admitted");
        assert_eq!(a.name, "GPT-4o");
    }

    #[test]
    fn junk_is_refused_at_the_tier_the_rule_carries() {
        for (name, tier) in [
            ("30 minutes", Tier::Reject),
            ("types.rs", Tier::Reject),
            ("~/.claude/CLAUDE.md", Tier::Reject),
            ("v1.2.3", Tier::Review),
            ("de", Tier::Review),
        ] {
            match admit(name, "concept", &seed()) {
                Admission::Refuse(r) => assert_eq!(r.tier, tier, "{name:?}"),
                Admission::Admit(_) => panic!("{name:?} must be refused"),
            }
        }
    }

    #[test]
    fn an_unresolvable_type_keeps_the_entity_and_reports_the_raw_value() {
        let a = admit("Constellation Map", "widget", &seed())
            .admitted()
            .cloned()
            .expect("a novel type must not drop a real entity");
        assert_eq!(a.entity_type, "concept");
        assert_eq!(a.unrecognized_type.as_deref(), Some("widget"));
    }

    #[test]
    fn refusals_are_counted_per_tier() {
        let rejected = entity_name_quality::rejected_total();
        let reviewed = entity_name_quality::reviewed_total();
        let _ = admit("256 tokens", "concept", &seed());
        assert!(entity_name_quality::rejected_total() > rejected);
        assert_eq!(entity_name_quality::reviewed_total(), reviewed);
        let _ = admit("v1", "concept", &seed());
        assert!(entity_name_quality::reviewed_total() > reviewed);
    }

    #[test]
    fn empty_and_whitespace_names_never_panic() {
        for name in ["", "   ", "\u{200b}", "\n"] {
            assert!(matches!(
                admit(name, "concept", &seed()),
                Admission::Refuse(_)
            ));
        }
    }
}
