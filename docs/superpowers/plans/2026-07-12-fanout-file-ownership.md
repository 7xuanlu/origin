# Fan-out plan — file ownership, not rule slices

**Branch:** `redesign-onboarding-settings` · **Date:** 2026-07-12 · **Owner:** team-lead

## Why the spec's slicing can't be executed as written

The design spec (`2026-07-12-onboarding-settings-design-system.md`, §5) closes with:

> Dependency shape: S1 → everything; S2–S8 are mutually independent and land in any order.

That is true of the **diffs** — each slice is a coherent, shippable change. It is false of
the **execution**, because every implementation agent on this branch shares one working
directory. The slices are cut by *rule*, and the same file appears in up to three of them:

| File | Claimed by |
|---|---|
| `SetupWizard.tsx` | S3 + S4 + S5 |
| `AgentsSection.tsx` | S2 + S3 |
| `GeneralSection.tsx` | S2 + S6 |
| `AddMemoryForm.tsx` | S2 + S8 |
| `src/i18n/resources.ts` | S3 + S4 + S7 + S8 |

Five agents dispatched on five rule-slices would edit the same files concurrently and stage
each other's half-finished work. The spec's rules stay authoritative; only the work-splitting
changes: **each agent owns a disjoint set of files and applies every rule inside them.**

## Ownership map — exclusive write access

An agent may edit only the files in its own row. Stage with explicit paths; never `git add -A`.

| Agent | Owns (exclusive) | Spec rows |
|---|---|---|
| `impl-intel` | `settings/primitives.tsx`, `primitives.test.tsx`, `src/index.css` | **S1** |
| `impl-foundation` | `RemoteAccessPanel.tsx`, `AddMemoryForm.tsx`, `hardcodedCopyBaseline.tsv` | **S7**-i18n, **S8**, + AddMemoryForm's `Select` (S2) |
| `impl-wizard` | `SetupWizard.tsx` (+ test) | S3/S4/S5 wizard rows |
| `impl-cards` | `intelligence/*`, `connect/*`, `VaultConnectCard`, `DropZone`, `ImportFlow` | S2/S4/S5 card rows |
| `impl-settings` | `settings/sections/*`, `SettingsPage.tsx`, `SettingsSidebar.tsx`, `lib/agents.ts`, `ActivityFeed.tsx` | S2/S3/S4/S5/S6 settings rows |
| team-lead | dead-key sweep in `resources.ts` | S3/S4 deletions |

`AddMemoryForm.tsx` is deliberately given whole to `impl-foundation` — it is the one file the
spec puts in two slices at once (S2 select + S8 i18n). One owner, both changes.

## The two shared files

`src/i18n/resources.ts` and `src/i18n/hardcodedCopyBaseline.tsv` cannot be partitioned.
They are a **lock: one holder at a time**, handed out by team-lead.

- `impl-foundation` holds it now (adds `remoteAccess.*` + `addMemory.*`, shrinks the ratchet by 26).
- `impl-settings` takes it next (adds `settings.agents.trustSummary.*`).
- `impl-wizard` and `impl-cards` never take it, and must not edit either file.

`impl-wizard` is lock-free because its only `resources.ts` work is **deletions**
(`setup.connect.detected`, `setup.connect.installFirst`, `setup.import.vaultPathTitle`).
An orphaned key breaks nothing — locale parity still holds, the key is merely dead — so the
deletion is deferred to a team-lead sweep after the wizard lands. Verified those three keys
have no consumer outside `SetupWizard.tsx`.

The ratchet may only shrink. `impl-wizard`, `impl-cards`, and `impl-settings` must leave
`hardcodedCopyBaseline.tsv` byte-identical; if a change would add a row, stop and report.

## Order

```
        impl-intel ──► S1 (primitives + tokens)   ── blocks everything ──┐
                                                                          │
   impl-foundation ──► S7-i18n + S8  [holds lock] ──────────┐             │
                                                            │             │
                                          ┌─────────────────┴─────────────▼──────┐
                                          │  impl-settings  [takes lock]         │
                                          └──────────────────────────────────────┘
                                          ┌──────────────────────────────────────┐
                                          │  impl-wizard   (lock-free) ── parallel│
                                          │  impl-cards    (lock-free) ── parallel│
                                          └──────────────────────────────────────┘
                                                            │
                                              team-lead ──► dead-key sweep
```

`impl-wizard` and `impl-cards` start the moment S1 lands — they do not wait on the lock.
`impl-settings` starts when `impl-foundation` releases it.

## Standing rules for every agent

- Mutation-prove every load-bearing test: break the product code, watch the test fail, paste
  the verbatim failure, revert with a **targeted edit** — never `git checkout HEAD -- <file>`.
- `--mem-*` tokens only. No new dependencies. IPC only through `src/lib/tauri.ts`.
- A11y: decorative glyphs `aria-hidden`; status text via `aria-describedby`, never inside a
  `<label>`; toggles `aria-pressed`.
- Never touch the privacy footer (`settings.footer`, `settings.localOnly`,
  `setup.privacyBody`) — it is false copy awaiting the user's personal sign-off.
- Never delete `CaptureSection` (coordinator override, `c03da71`).
- Never merge, never force-push.

## What stays with team-lead (not delegated)

Three product calls are parked with the user and no agent may pre-empt them:

1. Does ambient capture return, or get deleted?
2. On-device is the default while cloud wears "Recommended" (`SetupWizard.tsx:246` vs `:336`)
   — which one is the real recommendation?
3. The privacy-footer rewrite.

The wizard's double-ending (`verify` then `done`) and the ~500px dead space are composition
calls; they are held until the vocabulary lands, so they are not chased into a moving target.
