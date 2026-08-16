// SPDX-License-Identifier: Apache-2.0
//! Write-time canonicalization of `entity_type` against `entity_type_vocabulary`.
//!
//! Nothing checked the type string the extractor returned, so the production
//! corpus carries 37 off-vocabulary entity pages across 18 distinct values
//! ("concepts", "concepth", "platform", "language", "activity", ...). The
//! vocabulary already had an after-the-fact healer
//! (`kg_quality::heal_entity_vocabulary`); this module applies the decision at
//! write time so the column is canonical the moment it is stored.
//!
//! Bands, in order:
//!   1. already canonical, case-folded
//!   2. [`ALIASES`] — an explicit, reviewed mapping. Every off-vocabulary value
//!      observed in the corpus appears here with a deliberate target, so
//!      "language" becomes `technology` and "technique" becomes `concept`
//!      rather than everything collapsing into one bucket.
//!   3. [`crate::vocab::safe_transform`]: casefold, `-`/space to `_`, unique
//!      trailing-`s` singularize — catches plurals no alias anticipated.
//!   4. unresolved: the entity is still stored under [`DEFAULT_ENTITY_TYPE`]
//!      and [`Normalized::unrecognized`] is set, which makes the caller queue a
//!      `vocab_promote` proposal so a human sees the new value.
//!
//! Band 4 keeps the entity. A novel type name is a metadata gap, not evidence
//! that the entity is junk: "Python" arrived typed `language`, and dropping it
//! over that would be a worse failure than storing it as `technology`. The name
//! rules in [`crate::kg::entity_name_quality`] are what decide whether an
//! entity is real.

/// Where a type that no band resolved is stored. `concept` is the catch-all of
/// the six seeded canonicals and matches the extraction prompt's own fallback.
pub const DEFAULT_ENTITY_TYPE: &str = "concept";

/// Explicit alias to canonical. The left column is every off-vocabulary value
/// present in the production corpus plus the obvious synonyms of the six
/// canonicals; the right column was chosen per value, not by a blanket rule.
///
/// Corpus counts at the time of writing: concepts 7, platform 5, document 3,
/// interest 3, product 3, domain 2, file 2, function 2, and one each of
/// activity, command, component, concepth, feature, language, problem,
/// solution, system, technique.
pub const ALIASES: &[(&str, &str)] = &[
    // typos and plurals of the canonicals
    ("concepts", "concept"),
    ("concepth", "concept"),
    ("technologies", "technology"),
    ("organisation", "organization"),
    ("organisations", "organization"),
    ("organizations", "organization"),
    ("projects", "project"),
    ("persons", "person"),
    ("people", "person"),
    ("places", "place"),
    // software and platforms -> technology
    ("platform", "technology"),
    ("product", "technology"),
    ("system", "technology"),
    ("language", "technology"),
    ("component", "technology"),
    ("domain", "technology"),
    ("tool", "technology"),
    ("framework", "technology"),
    ("library", "technology"),
    ("service", "technology"),
    ("software", "technology"),
    ("api", "technology"),
    ("protocol", "technology"),
    ("database", "technology"),
    ("model", "technology"),
    // ideas, artifacts, and activities -> concept
    ("document", "concept"),
    ("file", "concept"),
    ("function", "concept"),
    ("command", "concept"),
    ("technique", "concept"),
    ("solution", "concept"),
    ("problem", "concept"),
    ("feature", "concept"),
    ("activity", "concept"),
    ("interest", "concept"),
    ("topic", "concept"),
    ("method", "concept"),
    ("practice", "concept"),
    ("process", "concept"),
    ("idea", "concept"),
    ("event", "concept"),
    // groups -> organization
    ("company", "organization"),
    ("org", "organization"),
    ("team", "organization"),
    ("group", "organization"),
    ("institution", "organization"),
    ("community", "organization"),
    // individuals -> person
    ("individual", "person"),
    ("human", "person"),
    ("author", "person"),
    ("user", "person"),
    // geography -> place
    ("location", "place"),
    ("city", "place"),
    ("country", "place"),
    ("region", "place"),
    ("venue", "place"),
    // codebases -> project
    ("repo", "project"),
    ("repository", "project"),
    ("codebase", "project"),
];

/// Outcome of normalizing one type string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Normalized {
    /// Always a member of `canonicals`, or [`DEFAULT_ENTITY_TYPE`] when the
    /// vocabulary is somehow empty.
    pub canonical: String,
    /// True when no band matched and the value was defaulted. The caller uses
    /// this to queue a promote candidate for human review.
    pub unrecognized: bool,
}

fn pick<'a>(target: &str, canonicals: &'a [String]) -> Option<&'a String> {
    canonicals.iter().find(|c| c.to_lowercase() == target)
}

/// Normalize `raw` against `canonicals` (the `entity_type_vocabulary` seed).
pub fn normalize(raw: &str, canonicals: &[String]) -> Normalized {
    let trimmed = raw.trim();
    let lower = trimmed.to_lowercase();

    if let Some(hit) = pick(&lower, canonicals) {
        return Normalized {
            canonical: hit.clone(),
            unrecognized: false,
        };
    }
    // An alias only counts when its target is actually in the vocabulary; a
    // custom seed that dropped "technology" must not receive it by the back
    // door.
    if let Some((_, target)) = ALIASES.iter().find(|(alias, _)| *alias == lower) {
        if let Some(hit) = pick(target, canonicals) {
            return Normalized {
                canonical: hit.clone(),
                unrecognized: false,
            };
        }
    }
    if let Some(hit) = super::safe_transform::safe_transform(trimmed, canonicals) {
        return Normalized {
            canonical: hit,
            unrecognized: false,
        };
    }
    // Unresolved: store the default, and let the caller surface the raw value.
    let canonical = pick(DEFAULT_ENTITY_TYPE, canonicals)
        .cloned()
        .unwrap_or_else(|| DEFAULT_ENTITY_TYPE.to_string());
    Normalized {
        canonical,
        unrecognized: true,
    }
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
    fn canonical_passes_through_unchanged() {
        for t in ["concept", "person", "place"] {
            let out = normalize(t, &seed());
            assert_eq!(out.canonical, t);
            assert!(!out.unrecognized);
        }
    }

    #[test]
    fn casefold_and_whitespace_fold_to_canonical() {
        for t in ["Concept", "PERSON", "  place  "] {
            let out = normalize(t, &seed());
            assert!(!out.unrecognized, "{t:?} should resolve deterministically");
            assert!(seed().contains(&out.canonical));
        }
    }

    #[test]
    fn every_off_vocabulary_value_in_the_corpus_has_a_deliberate_target() {
        // The 20 distinct off-vocabulary values behind the 37 production rows,
        // each with the canonical the alias table assigns. Not a blanket
        // rewrite: nine of them land somewhere other than "concept".
        let expected: &[(&str, &str)] = &[
            ("concepts", "concept"),
            ("concepth", "concept"),
            ("platform", "technology"),
            ("product", "technology"),
            ("system", "technology"),
            ("language", "technology"),
            ("component", "technology"),
            ("domain", "technology"),
            ("document", "concept"),
            ("file", "concept"),
            ("function", "concept"),
            ("command", "concept"),
            ("technique", "concept"),
            ("solution", "concept"),
            ("problem", "concept"),
            ("feature", "concept"),
            ("activity", "concept"),
            ("interest", "concept"),
        ];
        for (raw, canonical) in expected {
            let out = normalize(raw, &seed());
            assert_eq!(out.canonical, *canonical, "{raw:?}");
            assert!(!out.unrecognized, "{raw:?} must resolve, not default");
        }
    }

    #[test]
    fn plural_singularizes_even_without_an_alias() {
        let out = normalize("Projects", &seed());
        assert_eq!(out.canonical, "project");
        assert!(!out.unrecognized);
    }

    #[test]
    fn a_genuinely_new_type_keeps_the_entity_and_is_flagged_for_review() {
        for t in ["widget", "constellation", "   "] {
            let out = normalize(t, &seed());
            assert_eq!(out.canonical, DEFAULT_ENTITY_TYPE, "{t:?}");
            assert!(out.unrecognized, "{t:?} must be queued for review");
        }
    }

    #[test]
    fn an_alias_whose_target_is_missing_from_the_seed_does_not_smuggle_it_in() {
        let narrow = vec!["concept".to_string(), "person".to_string()];
        let out = normalize("platform", &narrow);
        assert_eq!(out.canonical, "concept");
        assert!(out.unrecognized, "unmappable alias must reach review");
    }

    #[test]
    fn output_is_always_a_vocabulary_member() {
        for t in ["concepts", "Widget", "", "PERSON", "places", "language"] {
            let out = normalize(t, &seed());
            assert!(
                seed().contains(&out.canonical),
                "{t:?} produced non-canonical {:?}",
                out.canonical
            );
        }
    }

    #[test]
    fn empty_vocabulary_still_yields_the_default() {
        let out = normalize("anything", &[]);
        assert_eq!(out.canonical, DEFAULT_ENTITY_TYPE);
        assert!(out.unrecognized);
    }

    #[test]
    fn alias_table_is_internally_consistent() {
        for (alias, target) in ALIASES {
            assert_eq!(
                *alias,
                alias.to_lowercase(),
                "alias {alias:?} must be lowercase"
            );
            assert!(
                seed().contains(&target.to_string()),
                "alias {alias:?} targets {target:?}, which is not a seeded canonical"
            );
            assert_ne!(alias, target, "{alias:?} is already canonical");
        }
    }
}
