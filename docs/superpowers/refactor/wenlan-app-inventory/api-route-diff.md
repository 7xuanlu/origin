# Wenlan App API Route Diff

- **App root:** `.`
- **Backend root:** `wenlan`

## Counts

- backend route paths: 123
- app source route paths: 115
- backend routes with no direct app source path: 8
- classified backend route gaps: 8
- unclassified backend route gaps: 0
- app source paths with no backend router path: 0

## Backend Routes With No Direct App Source Path

| Route | Category | Status | Next Action |
|---|---|---|---|
| `/api/config/skip-apps` | `superseded_route` | intentional | No app work unless the daemon removes the legacy route. |
| `/api/context` | `agent_route` | intentional | Keep hidden unless a diagnostics context panel is designed. |
| `/api/memory/entities/{entity_id}/observations` | `alternate_route` | intentional | No direct wrapper unless the generic observation route is deprecated. |
| `/api/memory/link-entity` | `graph_authoring` | design_required | Design a graph-authoring or entity-suggestion acceptance flow before exposing. |
| `/api/memory/relations` | `graph_authoring` | design_required | Design a relation editor/review flow with provenance and undo states. |
| `/api/ping` | `redundant_health` | intentional | No app work. |
| `/api/steep` | `operator_maintenance` | deferred | Keep deferred; do not expose until a separate maintenance design covers cost, capability, progress, and failure states. |
| `/ws/updates` | `architecture_gap` | deferred | Revisit only if replacing the app event flow with daemon events becomes an explicit product goal. |

## Unclassified Backend Route Gaps

- none

## App Source Paths With No Backend Router Path

- none

