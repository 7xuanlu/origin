// SPDX-License-Identifier: Apache-2.0
//! Shape rules for knowledge-graph entity names.
//!
//! The LLM extractor returns spans that are not entities at all — durations
//! ("30 minutes"), percentages ("17%"), commit shas, file paths, source
//! filenames, and leftover test-fixture strings. Every one of those became an
//! entity plus a `kind='entity'` shadow page.
//!
//! # Three verdicts, not two
//!
//! [`assess`] answers with [`Verdict::Accept`], [`Verdict::Review`], or
//! [`Verdict::Reject`], and the split is the whole point:
//!
//! * **Reject** is reserved for shapes that cannot be a stable named referent
//!   under any reading: no letter or digit at all, a bare number, a quantity, a
//!   commit sha or UUID, a filesystem path, a bare source filename, or an exact
//!   test-fixture string. These are safe to archive in bulk.
//! * **Review** is for names that merely *look* suspicious: bare version
//!   strings, one- and two-character Latin names, URLs, and slash-bearing names
//!   with no strong path marker. The write path declines to create them and
//!   logs, but a bulk cleanup must not archive them without a human looking.
//!
//! A name is **never** rejected merely for being short, containing digits,
//! containing punctuation, or containing a dot or a slash. Those signals feed
//! Review at most. That rule is what keeps `Go`, `C#`, `3M`, `.NET`, `GPT-4o`,
//! `Node.js`, `llama.cpp`, `7xuanlu/wenlan`, `@huggingface/transformers`,
//! `Redis Pub/Sub`, and `建築/房地產公司` in the graph.
//!
//! # Scope
//!
//! Shape only. A rule earns a place here only if it can be decided from the
//! characters of the name with no world knowledge. Names that are merely vague
//! ("full", "duration", "steep") are a semantic problem and belong to prompt
//! work or a judged pass, not to a deterministic filter that must never eat a
//! real entity.
//!
//! Length rules apply to all-ASCII names only. Two characters is a whole word
//! in Han, kana, and Hangul, and `ΔΥ` is an ordinary Greek abbreviation.

use std::sync::atomic::{AtomicU64, Ordering};

/// How confident the rule is that the name is not an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Structurally impossible as a named referent. Safe to archive in bulk.
    Reject,
    /// Ambiguous. Do not create it, but do not bulk-archive it either.
    Review,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Reject => "reject",
            Tier::Review => "review",
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The specific rule that fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    // ── Reject tier ──
    /// Not one letter or digit in the whole string (".", "***").
    NoLetterOrDigit,
    /// The entire name is a number, optionally with a percent sign
    /// ("17", "15%", "3.5").
    BareNumber,
    /// A number with a unit or a currency amount ("30 minutes", "256 tokens",
    /// "280K", "1,200 users", "$24M").
    Quantity,
    /// A short git sha or a UUID ("bd691cc", "9f97751").
    HexOrUuid,
    /// A filesystem path: an absolute or relative prefix, a dot-directory, a
    /// cache or build directory, or a directory plus a source filename
    /// ("~/.claude/CLAUDE.md", "node_modules/.vite", "src/index.ts").
    FilesystemPath,
    /// A bare source or asset filename ("types.rs", "config.json",
    /// "entity-1280x900.png").
    SourceFilename,
    /// An exact test-harness string ("manual test memory number 3", "test").
    TestFixture,

    // ── Review tier ──
    /// A URL. It refers to something real, but a URL is a poor entity name.
    Url,
    /// A version string with no product attached ("v1", "v1.2.3"). Contrast
    /// "Python 3.12", which is anchored to a product and is accepted.
    BareVersion,
    /// One or two ASCII characters that are not a known short name.
    VeryShortName,
    /// Slashes and dots but no strong path marker ("app/eval/fixtures",
    /// "7xuanlu/origin.git"). Could be a repo slug; could be a path fragment.
    AmbiguousSlashes,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::NoLetterOrDigit => "no_letter_or_digit",
            Reason::BareNumber => "bare_number",
            Reason::Quantity => "quantity",
            Reason::HexOrUuid => "hex_or_uuid",
            Reason::FilesystemPath => "filesystem_path",
            Reason::SourceFilename => "source_filename",
            Reason::TestFixture => "test_fixture",
            Reason::Url => "url",
            Reason::BareVersion => "bare_version",
            Reason::VeryShortName => "very_short_name",
            Reason::AmbiguousSlashes => "ambiguous_slashes",
        }
    }

    /// Which bucket this rule belongs to.
    pub fn tier(self) -> Tier {
        match self {
            Reason::NoLetterOrDigit
            | Reason::BareNumber
            | Reason::Quantity
            | Reason::HexOrUuid
            | Reason::FilesystemPath
            | Reason::SourceFilename
            | Reason::TestFixture => Tier::Reject,
            Reason::Url
            | Reason::BareVersion
            | Reason::VeryShortName
            | Reason::AmbiguousSlashes => Tier::Review,
        }
    }
}

impl std::fmt::Display for Reason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The answer for one name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Review(Reason),
    Reject(Reason),
}

impl Verdict {
    /// The rule that fired, if any.
    pub fn reason(self) -> Option<Reason> {
        match self {
            Verdict::Accept => None,
            Verdict::Review(r) | Verdict::Reject(r) => Some(r),
        }
    }
}

/// One- and two-character names that are real entities. Only needed for names
/// that are entirely ASCII: anything with a non-ASCII character skips the
/// length rule outright, because two characters is a whole word in Han, kana,
/// and Hangul, and `ΔΥ` is an ordinary Greek abbreviation.
const SHORT_ALLOW: &[&str] = &[
    "c", "r", "go", "js", "ts", "py", "rs", "ai", "ml", "ui", "ux", "db", "os", "id", "ip", "s3",
    "d3", "2d", "3d", "3m", "c#", "f#", "hn", "ci", "cd", "qa", "io", "us", "uk", "eu", "tw", "jp",
    "kr", "cn", "tv", "hr", "pm", "vc",
];

/// Extensions that mark a source, config, or asset file. Deliberately excludes
/// `js`, `cpp`, `c`, `h`, `go`, `java`, `rb`, `net`, `io`, `ai`, `org`, and
/// `com`, because real product names and hostnames end in them: Next.js,
/// Node.js, llama.cpp, whisper.cpp, .NET, nowledge.ai, arxiv.org.
const FILE_EXT: &[&str] = &[
    "rs", "ts", "tsx", "jsx", "mjs", "cjs", "py", "json", "toml", "yml", "yaml", "md", "sh",
    "bash", "zsh", "lock", "css", "scss", "sql", "plist", "ini", "cfg", "csv", "tsv", "txt", "log",
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico",
];

/// Directory names that only ever appear inside a filesystem path.
const CACHE_SEGMENTS: &[&str] = &[
    "node_modules/",
    "target/debug",
    "target/release",
    "__pycache__",
    "/dist/",
    "/build/",
    ".vite",
    ".next/",
    ".cache/",
];

/// Units accepted glued straight onto the digits ("280K", "500ms"). Much
/// narrower than [`UNITS`], and additionally guarded by a two-digit minimum,
/// because a single digit plus a single letter is far more likely to be a name:
/// "3M" is a company, "4b" is not four bytes.
const GLUED_UNITS: &[&str] = &[
    "%", "ms", "s", "m", "h", "k", "x", "kb", "mb", "gb", "tb", "px",
];

/// Units that turn a leading number into a measurement when they stand as their
/// own word ("30 minutes", "1,200 users").
const UNITS: &[&str] = &[
    "ms", "s", "sec", "secs", "second", "seconds", "m", "min", "mins", "minute", "minutes", "h",
    "hr", "hrs", "hour", "hours", "d", "day", "days", "week", "weeks", "month", "months", "year",
    "years", "x", "k", "kb", "mb", "gb", "tb", "b", "bytes", "px", "rem", "em", "pt", "tokens",
    "token", "chars", "rows", "items", "files", "lines", "calls", "users", "requests", "percent",
    "%",
];

const CURRENCY: &[char] = &['$', '€', '£', '¥', '₩', '₹'];

/// Magnitude letters that may follow a currency amount ("$24M", "$1.5bn").
const CURRENCY_MAGNITUDE: &[&str] = &["", "k", "m", "b", "bn", "t", "mm"];

/// Test-harness phrases, matched only at the START of the name. Anchoring is
/// load-bearing: an unanchored "test entity" also matches the real name
/// "Convergence Test Entity", which the existing suite caught.
const FIXTURE_PREFIXES: &[&str] = &[
    "manual test memory",
    "burst test memory",
    "burst test",
    "manual test",
    "test memory",
    "sample memory",
];

/// Phrases that cannot occur in a real name at any position.
const FIXTURE_ANYWHERE: &[&str] = &["lorem ipsum", "foo bar"];

/// Words that are only ever placeholders. Matched against the WHOLE name, so
/// TestFlight, Testcontainers, "A/B testing", and "regression testing" are all
/// untouched — only a name that is exactly one of these fires.
const PLACEHOLDERS: &[&str] = &[
    "test",
    "tests",
    "testing",
    "placeholder",
    "dummy",
    "untitled",
    "tbd",
    "n/a",
    "asdf",
    "todo",
    "xxx",
    "sample",
    "foo",
    "bar",
    "baz",
    "example",
];

static REJECTED_TOTAL: AtomicU64 = AtomicU64::new(0);
static REVIEWED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// How many entity names this process has hard-rejected since start.
pub fn rejected_total() -> u64 {
    REJECTED_TOTAL.load(Ordering::Relaxed)
}

/// How many entity names this process has sent to review since start.
pub fn reviewed_total() -> u64 {
    REVIEWED_TOTAL.load(Ordering::Relaxed)
}

/// Bump the counter for a refusal. Called by the admission seam so both write
/// lanes report through the same pair of counters.
pub(crate) fn record(tier: Tier) {
    match tier {
        Tier::Reject => REJECTED_TOTAL.fetch_add(1, Ordering::Relaxed),
        Tier::Review => REVIEWED_TOTAL.fetch_add(1, Ordering::Relaxed),
    };
}

fn digits_only(s: &str) -> String {
    s.chars().filter(|c| *c != ',' && *c != '_').collect()
}

/// True for "17", "3.5", "1,200", "15%". Requires at least one ASCII digit, so
/// `f64`'s acceptance of "NaN" and "inf" cannot swallow a real name.
fn is_bare_number(name: &str) -> bool {
    if !name.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    let body = digits_only(name);
    let body = body.trim_end_matches('%');
    !body.is_empty() && body.parse::<f64>().is_ok()
}

/// "$24M", "€1.5bn". A leading currency symbol makes the whole string a number.
fn is_currency_amount(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !CURRENCY.contains(&first) {
        return false;
    }
    let rest = digits_only(chars.as_str()).trim().to_lowercase();
    let split = rest
        .find(|c: char| c.is_ascii_alphabetic())
        .unwrap_or(rest.len());
    let (num, magnitude) = rest.split_at(split);
    num.chars().any(|c| c.is_ascii_digit())
        && num.parse::<f64>().is_ok()
        && CURRENCY_MAGNITUDE.contains(&magnitude)
}

fn is_quantity(name: &str) -> bool {
    if is_currency_amount(name) {
        return true;
    }
    let mut parts = name.split_whitespace();
    let Some(head) = parts.next() else {
        return false;
    };
    // "30 minutes", "1,200 users": a number followed by exactly one unit word.
    if let Some(unit) = parts.next() {
        if parts.next().is_some() {
            return false;
        }
        let unit = unit.trim_end_matches('.').to_lowercase();
        return is_bare_number(head) && UNITS.contains(&unit.as_str());
    }
    // "280K", "500ms": digits then a glued unit, no space. Two digits minimum,
    // so "3M" (a company) and "4b" survive while "280K" does not.
    let digits: String = name
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',' || *c == '-' || *c == '+')
        .collect();
    let digit_count = digits.chars().filter(|c| c.is_ascii_digit()).count();
    if digit_count == 0 {
        return false;
    }
    let suffix = name[digits.len()..].to_lowercase();
    if suffix.is_empty() || !GLUED_UNITS.contains(&suffix.as_str()) {
        return false;
    }
    suffix == "%" || digit_count >= 2
}

fn is_hex_or_uuid(name: &str) -> bool {
    let hex_run = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
    // Short git sha: 7-40 hex characters carrying at least one digit and one
    // letter, which keeps real words like "decade" and "facade" out.
    if (7..=40).contains(&name.len())
        && hex_run(name)
        && name.chars().any(|c| c.is_ascii_digit())
        && name.chars().any(|c| c.is_ascii_alphabetic())
    {
        return true;
    }
    let groups: Vec<&str> = name.split('-').collect();
    groups.len() == 5
        && [8, 4, 4, 4, 12] == groups.iter().map(|g| g.len()).collect::<Vec<_>>()[..]
        && groups.iter().all(|g| hex_run(g))
}

fn is_url(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// The last `/`-separated segment, or the whole name when there is no slash.
fn last_segment(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

/// A bare filename: no slash, a dot, and a known source or asset extension.
fn is_source_filename(name: &str) -> bool {
    if name.contains(char::is_whitespace) || name.contains('/') {
        return false;
    }
    match name.rsplit_once('.') {
        Some((_, ext)) => FILE_EXT.contains(&ext.to_lowercase().as_str()),
        None => false,
    }
}

/// High-confidence filesystem path. Every branch needs a marker a repo slug or
/// a scoped package name cannot have.
fn is_filesystem_path(name: &str) -> bool {
    // `~/`, `./` and `../` are unambiguous path markers, so they are decided
    // before the whitespace guard below: macOS paths routinely contain a space
    // ("~/Library/Application Support/...") and must not escape on that basis.
    if name.len() > 2
        && (name.starts_with("~/") || name.starts_with("./") || name.starts_with("../"))
    {
        return true;
    }
    // Every remaining rule reads a bare slash or a dot as a path marker, which
    // is only safe on a single whitespace-free token. "Redis Pub/Sub" and
    // "AuthN/AuthZ & On-Chain Identity" are entities, not paths.
    if name.contains(char::is_whitespace) {
        return false;
    }
    // Absolute.
    if name.len() > 2 && name.starts_with('/') {
        return true;
    }
    // A dot-directory at the front: ".git/config", ".claude/worktrees/".
    if name.starts_with('.') && name.contains('/') {
        return true;
    }
    // A trailing slash on a single token is a directory: "notes/", "target/".
    if name.ends_with('/') && name.len() > 1 {
        return true;
    }
    // A cache or build directory anywhere in the string.
    let lower = name.to_lowercase();
    if CACHE_SEGMENTS.iter().any(|seg| lower.contains(seg)) {
        return true;
    }
    // A directory plus a source filename: "src/index.ts", "scripts/dev-sync.sh".
    name.contains('/') && is_source_filename(last_segment(name))
}

/// Slashes or dots with no strong marker. Could be a repo slug, could be a path
/// fragment; a human decides. A single slash with no dot is NOT flagged, which
/// is what keeps "TCP/IP", "@huggingface/transformers", "7xuanlu/wenlan", and
/// "建築/房地產公司" clean.
fn has_ambiguous_slashes(name: &str) -> bool {
    if name.contains(char::is_whitespace) || !name.contains('/') {
        return false;
    }
    name.contains('.') || name.matches('/').count() >= 2
}

/// A version with no product attached: "v1", "v2", "1.2.3", "v0.9.1".
/// "Python 3.12" and "Opus 4.6" are anchored to a product and are accepted.
fn is_bare_version(name: &str) -> bool {
    if name.contains(char::is_whitespace) {
        return false;
    }
    let body = name.strip_prefix(['v', 'V']).unwrap_or(name);
    if body.is_empty() || !body.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return false;
    }
    body.split('.')
        .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

/// One or two ASCII characters that are not a known short name. Non-ASCII names
/// are exempt: two characters is a whole word in Han, kana, and Hangul, and
/// "ΔΥ" is an ordinary Greek abbreviation.
fn is_very_short(name: &str) -> bool {
    if name.chars().count() > 2 || !name.is_ascii() {
        return false;
    }
    !SHORT_ALLOW.contains(&name.to_lowercase().as_str())
}

fn is_test_fixture(name: &str) -> bool {
    let lower = name.to_lowercase();
    PLACEHOLDERS.contains(&lower.as_str())
        || FIXTURE_PREFIXES.iter().any(|p| lower.starts_with(p))
        || FIXTURE_ANYWHERE.iter().any(|p| lower.contains(p))
}

/// The verdict for one name. Pure: no logging, no counters, no I/O.
///
/// Order matters. URLs are checked before the path rules so a URL is reviewed
/// rather than rejected, and the reject-tier rules run before the review-tier
/// ones so the reported reason is the strongest one that applies.
pub fn assess(name: &str) -> Verdict {
    let name = name.trim();
    if name.is_empty() || !name.chars().any(char::is_alphanumeric) {
        return Verdict::Reject(Reason::NoLetterOrDigit);
    }
    if is_url(name) {
        return Verdict::Review(Reason::Url);
    }
    if is_bare_number(name) {
        return Verdict::Reject(Reason::BareNumber);
    }
    if is_quantity(name) {
        return Verdict::Reject(Reason::Quantity);
    }
    if is_hex_or_uuid(name) {
        return Verdict::Reject(Reason::HexOrUuid);
    }
    if is_filesystem_path(name) {
        return Verdict::Reject(Reason::FilesystemPath);
    }
    if is_source_filename(name) {
        return Verdict::Reject(Reason::SourceFilename);
    }
    if is_test_fixture(name) {
        return Verdict::Reject(Reason::TestFixture);
    }
    if is_bare_version(name) {
        return Verdict::Review(Reason::BareVersion);
    }
    if is_very_short(name) {
        return Verdict::Review(Reason::VeryShortName);
    }
    if has_ambiguous_slashes(name) {
        return Verdict::Review(Reason::AmbiguousSlashes);
    }
    Verdict::Accept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reason_of(name: &str) -> Option<Reason> {
        assess(name).reason()
    }

    #[test]
    fn hard_rejects_are_shapes_that_cannot_be_a_named_referent() {
        let cases: &[(&str, Reason)] = &[
            (".", Reason::NoLetterOrDigit),
            (". ", Reason::NoLetterOrDigit),
            ("***", Reason::NoLetterOrDigit),
            ("   ", Reason::NoLetterOrDigit),
            ("17", Reason::BareNumber),
            ("2%", Reason::BareNumber),
            ("15%", Reason::BareNumber),
            ("17%", Reason::BareNumber),
            ("3.5", Reason::BareNumber),
            ("$24M", Reason::Quantity),
            ("30 minutes", Reason::Quantity),
            ("5 minutes", Reason::Quantity),
            ("256 tokens", Reason::Quantity),
            ("1,200 users", Reason::Quantity),
            ("280K", Reason::Quantity),
            ("500ms", Reason::Quantity),
            ("bd691cc", Reason::HexOrUuid),
            ("9f97751", Reason::HexOrUuid),
            ("a310415", Reason::HexOrUuid),
            ("3f444117-ef36-4205-b00b-2c0dc830683e", Reason::HexOrUuid),
            ("~/.claude/CLAUDE.md", Reason::FilesystemPath),
            ("~/.claude/skills", Reason::FilesystemPath),
            ("~/Library/WebKit/origin/", Reason::FilesystemPath),
            // A home-relative path is a path even with a space in it; the
            // whitespace guard must not be what saves these.
            (
                "~/Library/Application Support/{bundle-identifier}/",
                Reason::FilesystemPath,
            ),
            ("~/.claude.json config", Reason::FilesystemPath),
            ("notes/", Reason::FilesystemPath),
            ("target/", Reason::FilesystemPath),
            ("./target/debug/origin", Reason::FilesystemPath),
            ("/no_think", Reason::FilesystemPath),
            ("app/target/debug/binaries/", Reason::FilesystemPath),
            ("node_modules/.vite", Reason::FilesystemPath),
            (".git/config", Reason::FilesystemPath),
            (".claude/worktrees/", Reason::FilesystemPath),
            (".origin/.backfill-attempted", Reason::FilesystemPath),
            ("src/index.ts", Reason::FilesystemPath),
            ("scripts/dev-sync.sh", Reason::FilesystemPath),
            ("preview/mocks/core.ts", Reason::FilesystemPath),
            (
                "e2e/spaces-navigation.visual.spec.ts",
                Reason::FilesystemPath,
            ),
            ("skills/claude-profile/SKILL.md", Reason::FilesystemPath),
            (".mcp.json", Reason::SourceFilename),
            (".yml", Reason::SourceFilename),
            ("config.json", Reason::SourceFilename),
            ("types.rs", Reason::SourceFilename),
            ("db.rs", Reason::SourceFilename),
            ("App.tsx", Reason::SourceFilename),
            ("release.yml", Reason::SourceFilename),
            ("AGENTS.md", Reason::SourceFilename),
            ("tauri.conf.json", Reason::SourceFilename),
            ("entity-1280x900.png", Reason::SourceFilename),
            ("live-smoke-*.sh", Reason::SourceFilename),
            ("burst test", Reason::TestFixture),
            ("manual test memory number 2", Reason::TestFixture),
            ("manual test memory number 3", Reason::TestFixture),
            ("test", Reason::TestFixture),
            ("tests", Reason::TestFixture),
            ("Placeholder", Reason::TestFixture),
        ];
        for (name, expected) in cases {
            assert_eq!(
                assess(name),
                Verdict::Reject(*expected),
                "expected {name:?} to be hard-rejected as {expected}"
            );
        }
    }

    #[test]
    fn ambiguous_shapes_go_to_review_never_to_reject() {
        let cases: &[(&str, Reason)] = &[
            ("https://lucian-site-gilt.vercel.app/", Reason::Url),
            ("v1", Reason::BareVersion),
            ("v2", Reason::BareVersion),
            ("v1.2.3", Reason::BareVersion),
            ("v0.9.1", Reason::BareVersion),
            ("1.2.3", Reason::BareVersion),
            ("de", Reason::VeryShortName),
            ("g", Reason::VeryShortName),
            ("v*", Reason::VeryShortName),
            ("+g", Reason::VeryShortName),
            ("4b", Reason::VeryShortName),
            ("E4", Reason::VeryShortName),
            ("pr", Reason::VeryShortName),
            ("app/eval/fixtures", Reason::AmbiguousSlashes),
            ("7xuanlu/origin.git", Reason::AmbiguousSlashes),
        ];
        for (name, expected) in cases {
            assert_eq!(
                assess(name),
                Verdict::Review(*expected),
                "expected {name:?} to be sent to review as {expected}"
            );
        }
    }

    #[test]
    fn real_entities_that_look_junk_shaped_are_accepted() {
        // A regression here is a false rejection: a name the graph silently
        // loses. Every entry is either a real entity from the production corpus
        // or a name the cross-check flagged as a must-keep.
        let keep = [
            // short and punctuated
            "C++",
            "C#",
            "Go",
            "C",
            "R",
            "S3",
            "d3",
            "HN",
            "CI",
            "AI",
            "UX",
            "3D",
            "2D",
            "3M",
            ".NET",
            "iOS",
            "k8s",
            "Web3",
            "ES6",
            "x86_64",
            "501(c)(3)",
            "macOS",
            "ΔΥ",
            // versions anchored to a product
            "GPT-4",
            "GPT-4o",
            "gpt-4",
            "Opus 4.6",
            "Qwen3-4B",
            "Python 3.12",
            "Windows 11",
            // dotted product names, which a filename rule would eat
            "Next.js",
            "Node.js",
            "llama.cpp",
            "whisper.cpp",
            // hostnames, which a filename rule would also eat
            "nowledge.ai",
            "momtestbook.com",
            "example.io",
            "arxiv.org",
            // repo slugs and scoped packages, which a path rule would eat
            "7xuanlu/wenlan",
            "DeusData/codebase-memory-mcp",
            "googleapis/release-please-action@v4",
            "@huggingface/transformers",
            "@origin-memory/mcp",
            "@react-native-google-signin/google-signin",
            // prose containing slashes
            "TCP/IP",
            "CI/CD",
            "Redis Pub/Sub",
            "AuthN/AuthZ & On-Chain Identity",
            "OAuth Client ID/Secret",
            "PRs #118/#119",
            // non-Latin scripts, where the length rule must not apply
            "台湾",
            "靜心",
            "小老",
            "東京",
            "한국",
            "建築/房地產公司",
            // words that brush the hex rule
            "decade",
            "facade",
            // names containing "test" that are real
            "Convergence Test Entity",
            "A/B test memory pool",
            "regression testing",
            "TestFlight",
            "Testcontainers",
            "Rust",
            "Claude Code",
        ];
        for name in keep {
            assert_eq!(
                assess(name),
                Verdict::Accept,
                "{name:?} is a real entity and must be accepted"
            );
        }
    }

    #[test]
    fn no_name_is_rejected_only_for_being_short_or_punctuated() {
        // The tier split is the safety property: every short, digit-bearing,
        // dotted, or slashed name that fires a rule must land in Review.
        for name in [
            "v1",
            "de",
            "g",
            "+g",
            "4b",
            "E4",
            "1.2.3",
            "app/eval/fixtures",
            "7xuanlu/origin.git",
            "https://example.com/",
        ] {
            match assess(name) {
                Verdict::Review(_) => {}
                other => panic!("{name:?} must be review-tier, got {other:?}"),
            }
        }
    }

    #[test]
    fn trims_before_judging() {
        assert_eq!(reason_of("  types.rs  "), Some(Reason::SourceFilename));
        assert_eq!(assess("  Rust  "), Verdict::Accept);
    }

    #[test]
    fn every_reason_reports_a_stable_tier() {
        for (reason, tier) in [
            (Reason::BareNumber, Tier::Reject),
            (Reason::FilesystemPath, Tier::Reject),
            (Reason::TestFixture, Tier::Reject),
            (Reason::Url, Tier::Review),
            (Reason::VeryShortName, Tier::Review),
        ] {
            assert_eq!(reason.tier(), tier, "{reason} tier");
        }
    }
}
