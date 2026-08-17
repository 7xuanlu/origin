# AGENTS.md - e2e/

## OVERVIEW

Playwright browser tests for user-visible flows that need a real rendered app,
run through a Tauri runtime mock installed before page load. The suite groups
into: i18n (`i18n.spec.ts`, `spaces-zh-hans-visual.spec.ts`), page canvas
(`page-canvas-*.spec.ts`, `page-editor.spec.ts`, `page-draft.visual.spec.ts`),
spaces (`spaces-*.spec.ts`, `space-mark.visual.spec.ts`), wiki
(`wiki-*.spec.ts`), graph (`graph-rendering.visual.spec.ts`), plus a handful of
standalone suites (`memory-list.spec.ts`, `review-flavor.review.spec.ts`,
`no-third-party-requests.spec.ts`, `tauriMock.self-test.spec.ts`). `*.visual.spec.ts`
files are screenshot-diff tests.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Add browser scenario | `*.spec.ts` | `playwright.config.ts` sets `testDir: ./e2e` |
| Mock Tauri runtime | `tauriMock.ts` | installs `__TAURI_INTERNALS__` via `addInitScript` |
| Localization flow | `i18n.spec.ts` | asserts shell labels and raw action labels |
| Server config | `../playwright.config.ts` | Vite on `E2E_PORT` or `14320` |

## CONVENTIONS

- Install Tauri mocks before `page.goto("/")` so app bootstrap sees the mocked
  window internals immediately.
- Keep e2e tests focused on behavior not covered well by Vitest component tests:
  browser bootstrap, rendered localization, navigation, and full-page flows.
- Capture page and console errors in tests that validate shell rendering.
- Prefer accessible roles and labels over text-node scraping for interactions.
- The web server is Vite, not Tauri; do not assume native desktop APIs are real.

## ANTI-PATTERNS

- Do not depend on a live daemon from e2e unless the test and CI setup create
  one explicitly.
- Do not write tests that pass only because untranslated English fallback copy
  remains visible.
- Do not mutate shared browser storage without resetting it through the mock or
  test setup.
- Do not broaden timeouts before first checking whether the Tauri mock is
  missing a command the UI needs.

## COMMANDS

```bash
pnpm test:e2e
pnpm exec playwright test e2e/i18n.spec.ts
```
