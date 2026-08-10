// SPDX-License-Identifier: AGPL-3.0-only
//
// TypeScript mirrors of the daemon's heading helpers.
//
// A `section` node's `ref_id` is `"{page_id}#{slug}"`, and the daemon
// recomputes liveness on every read by re-extracting the page's headings and
// slugifying them (`page_map_routes.rs::compute_ref_state`). Nothing validates
// the ref at create time, so drift here is silent: the node is written fine and
// then renders dangling forever. Keep these byte-compatible with
// `wenlan_core::page_map_improve::extract_headings` and
// `wenlan_core::export::obsidian::slugify` — quirks included.

/**
 * Port of `wenlan_core::export::obsidian::slugify`.
 *
 * Two quirks are deliberate, not bugs to fix here:
 * - `split("--").join("-")` collapses *pairs* of dashes left to right, so
 *   `"a---b"` becomes `"a--b"`, not `"a-b"`.
 * - Rust's `char::is_alphanumeric` is Unicode-aware, so CJK headings keep their
 *   characters. An `[a-z0-9]` class here would empty every Chinese slug and
 *   dangle every box on a Chinese page.
 */
export function slugify(title: string): string {
  const kept = Array.from(title.toLowerCase())
    .map((c) => {
      if (/[\p{Alphabetic}\p{N}]/u.test(c) || c === "-") return c;
      if (c === " ") return "-";
      return "";
    })
    .join("");
  return kept.split("--").join("-").replace(/^-+|-+$/gu, "");
}

/**
 * Port of `wenlan_core::page_map_improve::extract_headings` — every ATX
 * heading, de-duplicated case-insensitively in first-seen order.
 */
export function extractHeadings(content: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const line of content.split(/\r?\n/u)) {
    const trimmed = line.replace(/^\s+/u, "");
    if (!trimmed.startsWith("#")) continue;
    const heading = trimmed.replace(/^#+/u, "").trim();
    if (!heading) continue;
    const key = heading.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(heading);
  }
  return out;
}

/**
 * The page content that makes `heading` resolvable, appending an `##` section
 * only when no existing heading already slugifies to the same thing.
 *
 * Comparing slugs rather than text is what matters: the daemon's liveness
 * check is slug equality, so "Open Questions" and "open questions" are the
 * same anchor and appending the second would create a duplicate that resolves
 * to the first anyway.
 */
export function withHeading(content: string, heading: string): string {
  const slug = slugify(heading);
  if (!slug) return content;
  if (extractHeadings(content).some((h) => slugify(h) === slug)) return content;
  const body = content.replace(/\s+$/u, "");
  return body ? `${body}\n\n## ${heading}\n` : `## ${heading}\n`;
}
