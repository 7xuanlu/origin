# AGENTS.md - preview/

## OVERVIEW

Browser-only harness for inspecting app UI with Tauri APIs shimmed. The root
path can proxy live daemon calls; `/preview/` uses fixtures for page-detail
citation states.

## STRUCTURE

```text
preview/
|-- index.html           # fixture harness entry with __PREVIEW_FIXTURES__
|-- main.tsx             # PageDetail preview controls and theme toggle
|-- fixtures.ts          # citation/page/source/link/revision fixture data
`-- mocks/               # Vite aliases for Tauri APIs and live invoke bridge
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Add preview state | `fixtures.ts` | keep shapes aligned with `src/lib/tauri.ts` |
| Change preview UI | `main.tsx` | fixture selectors and theme toggle live here |
| Mock Tauri invoke | `mocks/core.ts` | fixture mode switch plus live daemon fallback |
| Live app preview | `mocks/live-invoke.ts` | proxies daemon calls through `/daemon` |
| Vite aliases | `../vite.preview.config.ts` | plugin API shims and port `1421` |

## CONVENTIONS

- Run with `pnpm exec vite --config vite.preview.config.ts`.
- Use `/preview/` for deterministic fixture states; use `/` only when a live
  daemon on `:7878` is part of the check.
- Keep fixture command responses shaped like the real Tauri wrappers consume.
- Add mocks in `mocks/core.ts` only for commands the preview actually renders.
- Preserve the page-detail citation variants unless the corresponding UI states
  are removed from the product.

## ANTI-PATTERNS

- Do not let fixture data drift into product code.
- Do not make preview-only commands look like validated backend behavior.
- Do not add network calls to fixture mode; use the live root path for that.
- Do not ignore console warnings for unmocked invokes when adding UI surfaces.

## COMMANDS

```bash
pnpm exec vite --config vite.preview.config.ts
```
