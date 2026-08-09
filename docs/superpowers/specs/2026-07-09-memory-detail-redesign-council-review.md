# Memory detail redesign — external design review (codex + agy)

Date: 2026-07-09. Reviewers: gpt-5.5 (codex CLI, xhigh) and Gemini 3.1 Pro High (agy CLI),
each given the current-page facts + the hero-first single-column mockup
(`memory-detail-redesign-preview.html`, session job 0299e8a7).

## Verdicts

| Reviewer | Verdict | Headline |
| --- | --- | --- |
| gpt-5.5 | approve-with-changes (high confidence) | Hero-first is right; fixed 780px column underuses desktop; plan the scale cases |
| Gemini 3.1 Pro | approve-with-changes (prose) | "Hero-first typography is a massive upgrade"; single column oversimplifies a 300-connection graph |

## Unanimous changes (both reviewers, adopt in v2)

1. **Hybrid layout**: keep the 780px centered reading column for the hero, but add a
   slim sticky right rail on ≥1280px for connections/graph navigation + heavy
   metadata editing. (Converges back to the original wireframe, away from the pure
   single-column mockup.)
2. **Scale handling**: hero type scales down (~27px → 18–20px past ~200 chars);
   tags get their own wrapping line below the primary metadata line; connections
   need filter/sort (and virtualization at hundreds), not just "show N more".
3. **Editing affordances**: the strip must read as editable — explicit hover/edit
   states per field; "+ tag" must not jump position with flex-wrap; keep an obvious
   Edit for the body text.
4. **Accessibility**: 10px mono meta is too small (contrast + target size) — floor
   12px; icon-only buttons get labels/tooltips; facet color never the only signal.
5. **Structure**: separate memories vs entities inside Connections (subheaders or
   tabs) with a rank/why-related signal.

## Single-reviewer flags (consider, not blocking)

- gpt-5.5: move destructive delete out of the primary top bar; add source/evidence
  view; add empty/loading/error states.
- Gemini: agent identity needs stronger ownership marker (avatar) in a multi-agent
  memory layer; define whether "history ▸" expands inline vs overlay; where do the
  other ~800 entities go (navigation path, not a dump).

## Decision

v2 = reading column + slim sticky rail (wide widths only) + items 2–5 above.
Implementation lands in `MemoryDetail.tsx` on branch `memory-ui-consistency`.

## v3 balance pass (user feedback 2026-07-08: "left and right visual balance")

- Rail de-boxed to marginalia: no panel border/background, a single hairline
  `border-left` — same airy texture as the reading column.
- Actions (Confirm / Pinned / Edit / Delete) move to the topbar; the rail
  carries connections only.
- Source excerpt block (when `source_text` differs from `content`) fills the
  reading column's lower half so short memories don't trail into void.
- Length-adaptive hero (user requirement: "memory content might be long or
  short"): ≤160 chars → 24px serif; ≤280 → 19px serif; longer → 15px body
  with a 17px lede first paragraph. Thresholds tightened 2026-07-08 after
  user feedback ("every words squeezed into title… not impressive and ugly"):
  display serif only holds ~4 lines, so a 400–600-char technical memory must
  render as body + lede, never whole-block serif.
  Grid `minmax(0,1fr) 300px`, gap 48px, frame 1120px, reading column ≤720px;
  single column below 980px.
