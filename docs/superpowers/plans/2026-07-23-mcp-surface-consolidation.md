# Wenlan MCP Surface Consolidation Implementation Plan

**Goal:** Reduce the default Wenlan MCP surface from 46 tools to exactly 26
without removing unique read-only workflows or weakening source provenance for
knowledge-graph writes.

**Scope:** `crates/wenlan-mcp`, the Claude and Codex plugin skills, plugin
contract validators, and truthful `~/.wenlan` cadence documentation. Phase B
permission redesign remains out of scope.

## Locked outcome

| Surface | Tool count | Advertised schema bytes |
|---|---:|---:|
| Baseline | 46 | 52,336 |
| Final | 26 | 34,349 |
| Reduction | 20 (43.5%) | 17,987 (34.4%) |

Both byte measurements use the same definition: the sum of
`len(json.dumps(tool))` for every tool returned by a live MCP `tools/list`
response.

The final 26 tools, alphabetically:

1. `accept_revision`
2. `apply_lint_repair`
3. `capture`
4. `confirm_memory`
5. `context`
6. `create_entity`
7. `create_relation`
8. `delete_page`
9. `dismiss_revision`
10. `distill`
11. `forget`
12. `get_lint_agent_work_page`
13. `get_lint_repair_plan_entries`
14. `get_memory_revisions`
15. `get_page_revisions`
16. `get_page_sources`
17. `lint`
18. `list_pending`
19. `list_pending_imports`
20. `list_pending_revisions`
21. `list_rejections`
22. `prepare_lint_repair`
23. `prepare_lint_repair_plan`
24. `recall`
25. `verify_lint_repair`
26. `write_page`

Count accounting:

| Change | Count |
|---|---:|
| Start | 46 |
| Permanently remove 17 legacy or redundant tools | 29 |
| Demote `doctor` and `list_spaces` to CLI workflows | 27 |
| Merge `create_page` and `update_page` into `write_page` | 26 |
| Retain source-backed `create_entity` and `create_relation` | 26 |

The four unique read-only tools
`get_memory_revisions`, `get_page_revisions`, `list_pending_imports`, and
`list_rejections` remain available and are documented only in their explicit
history, changelog, import-progress, and rejection workflows.

## Implementation checkpoints

1. Lock the exact final surface in a unit test and remove the 17 obsolete tool
   implementations, schemas, router entries, skill references, and dead tests.
2. Keep `create_entity` idempotent and keep `create_relation` narrow: both
   endpoint IDs and `source_memory_id` are required. The capture skills must
   describe the explicit sequence `capture` → `create_entity` →
   `create_relation`; ordinary capture does not imply graph writes.
3. Move `doctor` and `list_spaces` to truthful CLI fallbacks:
   `wenlan doctor` and `wenlan spaces list`.
4. Replace `create_page` and `update_page` with `write_page`. Creation accepts
   a title; refresh requires an existing page ID and a non-empty source-memory
   list. Preserve typed envelopes and revision-card output.
5. Remove the unsupported promise that `context` includes goals.
6. Exclude exactly these local mutation tools from streamable HTTP:
   `prepare_lint_repair`, `get_lint_agent_work_page`,
   `prepare_lint_repair_plan`, `get_lint_repair_plan_entries`,
   `apply_lint_repair`, `verify_lint_repair`, `forget`, `confirm_memory`, and
   `delete_page`.
7. Make lint routing and fallback copy truthful. The CLI can submit agent
   results with `wenlan lint --profile deep --agent_assist
   --agent-submission <file>`; MCP repair-manifest tools have no CLI equivalent.
8. Correct plugin guidance for `~/.wenlan`: commits land at session boundaries
   (handoff or daemon events), not per capture, and uncommitted page edits
   between sessions are normal.

## Explicitly deferred

Do not restore `create_observation`, `update_observation`,
`delete_observation`, `confirm_observation`, or `confirm_entity` in this phase.
The former observation-writing workflow referenced a nonexistent
`search_entities` tool, and the family lacks a complete source-backed authoring
contract. Design that contract separately before advertising observation CRUD
or entity confirmation again.

The revision decision trio remains: `list_pending_revisions`,
`accept_revision`, and `dismiss_revision`.

## Verification gate

Run all commands from the worktree root:

```bash
cargo fmt --all -- --check
env CARGO_TARGET_DIR=/private/tmp/wenlan-226e-target cargo clippy -p wenlan-mcp --all-targets -- -D warnings
env CARGO_TARGET_DIR=/private/tmp/wenlan-226e-target cargo test -p wenlan-mcp --lib
env CARGO_TARGET_DIR=/private/tmp/wenlan-226e-target cargo check -p wenlan-mcp --all-targets
python3 scripts/validate-plugin-contract.py
bash scripts/validate-plugin-contract.test.sh
python3 scripts/validate-codex-plugin-slice.py
git diff --check
git status --short
```

Also prove that the locked list contains exactly the 26 names above and that no
Claude or Codex skill contains a stale fully-qualified token for removed,
demoted, or merged tools.

Suggested pull-request title:

```text
fix: consolidate MCP tool surface (46→26 tools)
```
