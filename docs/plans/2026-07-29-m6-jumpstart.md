# M6 jumpstart

Paste the block below into a fresh session. Everything above it is context for
whoever is deciding whether to start.

## Is M5 PR-D a blocker for M6?

**No.** Three reasons, each verifiable:

1. **PR-D is inert.** `truth_cutover_generation` is 0, and at 0
   `page_visibility` returns `Full` for every page before it reads anything.
   Nothing PR-D added changes a single byte of behavior until someone runs
   `wenlan truth cutover --apply`.
2. **M6's dependencies are already on `main`.** M4 (the durable community
   substrate) merged as `5ba8a3b4`. The M5 reader adapters merged as PR-C
   `3932e3d5`. M6 consumes both.
3. **The one thing M6 inherits from M5 is already on `main`,** and it is a
   design constraint rather than code: any new page-bearing reader must declare
   a truth grant (`TruthGrant::{Automatic, CollectionEntries, NamedPages}`).
   That contract landed in PR-B/PR-C. A new M6 reader that skips it fails
   `truth_manifest`'s teeth, not PR-D's.

So M6 can start now, in parallel, in another session. The only coordination
point is the eventual flip, which is a separate ceremony and is not M6's.

## What M6 is

The **app / client plane**. Three specs defer work to it by name:

| Deferred item | Named at |
|---|---|
| Map-region / overview rollups — the spec's third §3 community consumer, client-side today (the app's degree heuristic), no daemon reader exists | `docs/plans/2026-07-25-m4-communities-mechanics.md:89, 119, 625` |
| `genesis_candidate_roots` and the provenance floor's consumption | `docs/plans/2026-07-21-m2-edge-assignment-matrix.md:30` |

The desktop app lives in a **separate repo**: `7xuanlu/wenlan-app`. Whether M6
lands there, here, or both is the first thing the session has to settle.

---

## The prompt

> M6 — the app/client plane. Start by settling scope, then build.
>
> **Read first, in this order:**
> - `docs/plans/2026-07-25-m4-communities-mechanics.md` sections 1, 3, and 11.
>   Section 1 lists the three §3 consumers and marks which two M4 already cut
>   over daemon-side; section 11 is the non-interference list.
> - `docs/plans/2026-07-21-m2-edge-assignment-matrix.md:20-40` for why every
>   edge carries `root_id = NULL` today and what `genesis_candidate_roots` was
>   deferred *for*.
> - `crates/wenlan-core/AGENTS.md`, the `WENLAN_ENABLE_COMMUNITY_LEIDEN` entry
>   — M4's substrate is write-only and default-OFF, so anything M6 reads is
>   empty on a stock daemon until that flag is on and a grouping cycle has run.
>
> **Settle these three before writing code. They are genuine forks, not
> details:**
>
> 1. **Which repo.** The desktop app is `7xuanlu/wenlan-app`, a separate repo
>    (AGPL-3.0; this workspace is Apache-2.0). Map-region rendering is app code.
>    But a *daemon reader* for community rollups would live here, in
>    `wenlan-server`, and the app would consume it over HTTP like everything
>    else. The daemon-is-the-single-source-of-truth architecture argues for the
>    second: put the rollup logic here, keep the app thin. Confirm with the user
>    rather than assuming — it decides which repo the session works in.
>
> 2. **Whether the app's degree heuristic gets replaced or supplemented.** Today
>    the app groups by node degree, client-side. M4 built a real per-space
>    Leiden partition. Swapping one for the other changes what users see, on
>    data they have already formed a mental model of. That is a product call.
>
> 3. **Whether `genesis_candidate_roots` is in scope at all.** It is a
>    provenance-floor concept from M2 and it is only adjacent to the map work.
>    Two rungs in one milestone is how a milestone stops shipping.
>
> **Standing constraints, non-negotiable:**
> - Every push, PR mutation, merge, or release needs explicit user approval
>   immediately before that action. Local branches, worktrees, RED tests, and
>   verification proceed freely.
> - Never auto-merge to `main`. PR-first.
> - No pushes Mon–Fri 09:00–17:00 PT.
> - `gh` must run with the sandbox disabled — inside it `gh` cannot reach macOS
>   `trustd` and misreports the TLS failure as an invalid token.
> - SSH is broken on this machine. Push over explicit HTTPS:
>   `git push https://github.com/7xuanlu/wenlan.git <branch>:<branch>`.
>
> **What M5 leaves you, and what it does not.** Any new page-bearing reader must
> declare a truth grant (`crates/wenlan-core/src/truth_contract.rs`) and appear
> in `truth_manifest`, which has teeth that fail the build otherwise. That is
> already on `main` and applies regardless of what M5 PR-D does. PR-D itself is
> inert — `truth_cutover_generation` is 0 — so do not wait on it, and do not
> advance it.
>
> Start with the scope questions. Do not open a PR before they are answered.
