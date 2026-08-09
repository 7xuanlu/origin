# Design QA — Planet SpaceMark

## Visual truth

- Approved runtime preview: `.omo/evidence/task-7-spaces-navigation-redesign/planet-preview/planet-preview-board.png` (`1800 × 1280`, SHA-256 `0a568a612d47cf64d840a59c5ed67cba35cf022f3aabf358e85f7bf3602189ec`).
- Fidelity contract: use the exact two official Tabler `Planet` paths frozen in `DESIGN.md`; keep Wenlan's existing `14 × 14` navigation footprint, round `2` stroke, theme tokens, active rail, and parent-owned focus/label behavior.
- Implemented selected state: `.omo/evidence/task-7-spaces-navigation-redesign/space-mark/space-mark-spaces-selected-light-1280x900.png`.
- Focused physical proof: `.omo/evidence/task-7-spaces-navigation-redesign/space-mark/space-mark-selected-light-dpr2.png` (`28 × 30` pixels at DPR2).
- Machine-readable state evidence: `.omo/evidence/task-7-spaces-navigation-redesign/space-mark/space-mark-qa.json`.

## Coverage

- Desktop light: default, selected, and keyboard focus.
- Desktop dark: default and selected.
- Mobile overlay: selected at `375 × 812`.
- Density: default and selected at physical DPR2.
- Localization: zh-Hant route and Space state.
- Browser errors: zero console errors and zero page errors.

## Pass A — design-system and functional integrity

**VERDICT: PASS. CONFIDENCE: HIGH.** `SpaceMark` is a live React/SVG primitive with the approved two-path Planet geometry. It reuses the existing navigation size, `currentColor` tokens, selected rail, hover, and parent focus behavior. The glyph is decorative (`aria-hidden`); the localized parent button owns the accessible name and `aria-current`. No raster asset, raw color, motion, dependency, or alternate icon rail was added.

The focused component contract passes `5/5`. The rendered E2E contract also proves the old Folder and Map paths occur zero times.

Blocking findings: none.

## Pass B — visual fidelity and CJK precision

**VERDICT: PASS. CONFIDENCE: HIGH.** Direct inspection of all eight fresh captures shows the ringed Planet remains unclipped and aligned in light, dark, selected, focus, mobile overlay, and DPR2 states. Its diagonal orbit is visibly different from Graph's three connected nodes and Sources' horizontal layers. No label geometry changed, so English and zh-Hant remain unclipped.

The strict RGB comparison in `planet-preview-vs-implementation-diff.json` reports only `17/100` because the preview and implementation icon crops use different flat row backgrounds (`rgb(243,244,246)` versus `rgb(250,250,251)`). A background-normalized foreground-mask check found `283` foreground pixels in each image, identical masks, `0` differing mask pixels, and IoU `1.0`. DOM evidence independently asserts the exact two approved paths, `14 × 14` CSS geometry, `2` stroke, and selected token color.

Blocking findings: none.

## Comparison history

1. Replaced the earlier folded Map direction after user review.
2. Implemented the approved Planet preview without a repair loop; no P0, P1, or P2 mismatch was found.

final result: passed

---

# Design QA — Home title and sidebar action convergence

## Latest-main comparison

- `HEAD` and `origin/main` were both `8927a7673c2e1a09bda0a676b826d216da359418` at audit time.
- Wenlan's existing type tokens remain unchanged: Fraunces for headings, Instrument Sans for interface copy, and JetBrains Mono for metadata.
- `Today in Wenlan` now shares the Wiki and Spaces page-title treatment: semantic `h1`, `clamp(32px, 3vw, 42px)`, weight `500`, line-height `1.16`, and letter-spacing `-0.035em`.
- The standalone sidebar Space creation button was removed. Space creation remains available from the full Spaces surface.

## Rendered verification

- Fresh fixture captures covered Home, Wiki, Spaces, Graph, Memories, and Sources at `1280 × 900`, plus Home at `768 × 800` and `393 × 852`.
- Home, Wiki, and Spaces now form one page-title hierarchy; sidebar rows keep a consistent full width and rhythm.
- The narrow layouts had no horizontal overflow or title/dateline collision. Existing focus-visible treatment was preserved.
- The fixture contains no CJK content on these screens; the change adds no locale-dependent copy, and the complete i18n suite passed.

## Verification

- Vitest: `128` files passed; `1169` tests passed and `1` skipped.
- i18n: `50/50` passed.
- TypeScript, production build, and native Review bundle: passed.
- Existing production chunk warnings and review-feature Rust unused-code warnings remain non-blocking.

Blocking findings: none.

final result: passed

---

# Design QA — Spaces, Wiki, Home, and Entity redesign

## Final information architecture

- Global order is `Home → Wiki → Spaces → Graph → Memories → Sources`, followed by a separate Recent Spaces section.
- Confirmed recent Spaces disappear on the full Spaces management surface and never become a nested Spaces inventory.
- Wiki is the first-class Page inventory. A Page may name a Space; explicit `null` or blank stays visually blank, while only a missing `space` field may fall back to legacy `domain` data.
- Opening a Page keeps Wiki selected. Opening an Entity uses its distinct wiki dossier treatment.

## Rendered matrix

- Thirty deterministic light/dark screenshots cover Home, Spaces, Wiki, Space detail, and Entity at `1280 × 900`, `768 × 900`, and `375 × 812`.
- Current Wiki reference: `.omo/evidence/wiki-implementation/wiki-light-1487x1058.png`.
- The matrix was regenerated after the last source edit. The strict no-update Playwright comparison passed `1/1`.
- CJK coverage checks Page, Space, and Entity text for clipping.

## Verification

- Vitest: `97` files passed; `773` tests passed and `1` skipped.
- i18n: `45/45` passed.
- Playwright: `30/30` passed; the regenerated visual matrix also passed independently in no-update mode.
- TypeScript and production build: passed.
- TypeScript no-excuse rules for the new Pages module: `4/4` files, zero violations.
- Five final review lanes passed: goal/visual integrity, visual QA, code quality, security, and design-context consistency.

## Non-blocking debt

- Date and count formatting on several existing Space surfaces still follows the OS locale rather than the selected app locale.
- The E2E reorder mock updates the moved item but does not model sibling order shifts as faithfully as the daemon.
- The production build retains existing mixed-import and large-chunk warnings.
- Security review found no issue introduced by this redesign. Existing repository debt remains: dependency audit advisories (`4` high, `3` moderate, `3` low) and a pre-existing unencoded Space-name path-segment bridge risk.

Blocking findings: none.

final redesign result: passed

---

# Design QA — Wiki implementation final gate

## Source and implementation

- Approved source: `/Users/lucian/.codex/generated_images/019f4ab8-aa4d-7b30-8c03-2f50c2a7bfb1/exec-4f9a606c-2bb2-475c-8c7a-feaf74964407.png`, `1487 × 1058`, SHA-256 `4ab30a3fdd2ecb033c23ff0ab1b0310ca57e877e1652625a09cd82c2ca39e5d6`.
- Live implementation: `src/components/memory/pages/PagesOverview.tsx`, `src/components/memory/navigation/PrimaryNavigation.tsx`, `src/components/memory/Main.tsx`, and `src/components/memory/navigation/navigation-shell.css`.
- Primary proof: `.omo/evidence/wiki-implementation/wiki-light-1487x1058.png`.

## Rendered states

- Resting surfaces: desktop light at `1487 × 1058`, desktop dark at `1440 × 1024`, Traditional Chinese tablet at `768 × 900`, and Traditional Chinese mobile at `375 × 812`.
- Interactions: filter focus, filtered Entity inventory, page two with Previous enabled, page-link hover at transition midpoint and settled state, mobile navigation overlay, mobile global-search overlay, and a real narrow-screen Cmd+K event path.
- Responsive semantics: desktop table retains Page/Type/Space/Updated; below `640px`, Page remains the visible column and Type/Space/Updated move into row metadata. Empty optional Space remains blank.
- Runtime errors: zero page errors and zero console errors in the Wiki Playwright gate.

## Verification

- Focused RED-to-green search/navigation contract: `41/41` passed.
- Full Vitest: `97` files, `773` passed, `1` skipped.
- Localization: `45/45` passed.
- Production build: passed; pre-existing mixed-import and large-chunk warnings remain non-blocking.
- Full Playwright: `30/30` passed, including contrast, keyboard focus, reduced motion, CJK, DPR2, 200% zoom, responsive Cmd+K, and the no-update screenshot matrix.

## Comparison history

1. Replaced the generated reference's competing recent/type rail with the approved full-width inventory.
2. Preserved Wenlan's 240px shell, editorial typography, theme tokens, and existing Graph icon while adding the folded-page Wiki mark, brain Memories mark, and Planet Spaces mark.
3. Corrected reviewer findings for source search routing, mobile search access, landmark naming, control/metadata contrast, design-contract drift, and missing interaction evidence.

## Independent final review

- Design-system and interaction-state gate: PASS, high confidence.
- Code and reference-fidelity gate: APPROVE, high confidence.
- Pixel comparison gate: APPROVE.
- CJK and responsive visual gate: PASS, high confidence.

Blocking findings: none.

final result: passed
