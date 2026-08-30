// SPDX-License-Identifier: AGPL-3.0-only
// The lede is a page's first sentence, shown as the pull quote under the
// title. A model sometimes opens the body with a "TLDR:" label; the label is
// not part of the sentence, so it must not reach the quote. Mirrors
// `strip_summary_label` in crates/wenlan-core/src/synthesis/distill.rs, which
// does the same for the stored summary.

// Optional markdown emphasis, the label, optional emphasis, then a separator.
// The separator is required, so prose that merely starts with the letters
// ("TLDRs are no substitute for the page") is left alone.
const LEDE_LABEL =
  /^\s*(?:\*{1,2}|_{1,2})?\s*TL;?DR\s*(?:\*{1,2}|_{1,2})?\s*[:\-–—]\s*(?:\*{1,2}|_{1,2})?\s*/i;

/** Drop a leading "TLDR:" label from a lede sentence or a stored summary. */
export function stripLedeLabel(text: string): string {
  return text.replace(LEDE_LABEL, "");
}
