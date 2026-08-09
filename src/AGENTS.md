# AGENTS.md - src/

## OVERVIEW

React 19 frontend for the Tauri app. This tree owns UI state, routing inside
the desktop windows, i18n, browser-side preview compatibility, and the typed
frontend side of the Tauri command contract.

## STRUCTURE

```text
src/
|-- main.tsx             # window hash bootstrap: app, toast, quick capture
|-- App.tsx              # top-level window/event shell
|-- components/
|   |-- memory/          # main product UI; has its own AGENTS.md
|   |-- ChatImport/      # chat import flow and ZIP/drop handling
|   |-- onboarding/      # first-page modal, milestone toasts/hooks
|   `-- intelligence/    # local/external model setup surface
|-- hooks/               # small shared hooks
|-- i18n/                # locale resolution and canonical resources
|-- lib/                 # Tauri wrappers, storage, classifiers, stores
`-- test/                # Vitest setup and Tauri mocks
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| New IPC wrapper | `lib/tauri.ts` | centralize invoke payloads and response defaults |
| Locale/copy change | `i18n/resources.ts` | keep parity across supported locales |
| Locale resolution | `i18n/index.ts`, `i18n/locales.ts` | handles browser-safe language mapping |
| Main memory UI | `components/memory/AGENTS.md` | navigation, settings, citations, sources |
| Chat import | `components/ChatImport/` | polling and notifications can hide failures |
| Onboarding | `components/onboarding/` | focus trap, milestone cache, dismissal policy |
| Test harness | `test/setup.ts` | Tauri API mocks and Testing Library setup |

## CONVENTIONS

- `src/lib/tauri.ts` is the frontend-to-Rust boundary. Add or update wrappers
  there before wiring components directly to a command.
- Translation resources are the canonical fixed-copy inventory. New visible
  interface copy should be keyed in `src/i18n/resources.ts` and covered by the
  i18n tests when the key is behaviorally important.
- Tests are colocated as `*.test.ts` / `*.test.tsx`, with some `__tests__`
  folders for grouped component suites.
- Vitest runs in jsdom with `src/test/setup.ts`. Do not require a real Tauri
  runtime from unit/component tests unless the test explicitly opts into it.
- Tailwind v4 is configured through CSS and the Vite plugin. There is no
  separate Tailwind config file to update.
- `pnpm test:e2e` is a Playwright surface outside this tree; use it for browser
  flows that unit tests cannot observe.

## ANTI-PATTERNS

- Do not bypass `lib/tauri.ts` with raw `invoke` calls from components.
- Do not add hardcoded JSX copy without considering `src/i18n/hardcodedCopyGuard.test.ts`.
- Do not broaden the Vitest coverage gate accidentally; `vitest.config.ts`
  intentionally whitelists only tested modules.
- Do not key behavior to placeholder text when i18n can change it; use roles,
  labels, refs, or test ids as appropriate.

## COMMANDS

```bash
pnpm exec tsc -b
pnpm test
pnpm test:i18n
pnpm test:e2e
pnpm test:coverage
```
