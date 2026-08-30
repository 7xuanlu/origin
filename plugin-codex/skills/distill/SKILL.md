---
name: distill
description: >
  Synthesize or refresh source-backed Wenlan pages from Codex. Invoked as
  /distill [target], /distill deep, or /distill rebuild <page-id>.
argument-hint: "[target | deep | rebuild <page-id>]"
allowed-tools: ["Bash", "mcp__wenlan__recall", "mcp__wenlan__distill", "mcp__wenlan__write_page", "mcp__wenlan__delete_page", "mcp__wenlan__get_page_sources"]
user-invocable: true
---

# /distill

Run a deliberate Wenlan distillation pass. The daemon finds clusters and stale
pages; Codex synthesizes any pending clusters the daemon cannot finish.

## Argument parsing

Accept one optional `space:<name>` token:

```bash
raw_args="<the full argument string passed to /distill>"
space_arg="$(printf '%s\n' "$raw_args" | grep -oE 'space:[A-Za-z0-9_-]+' | head -1 | cut -d: -f2)"
target="$(printf '%s\n' "$raw_args" | sed -E 's/[[:space:]]*space:[A-Za-z0-9_-]+[[:space:]]*/ /g' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
```

Resolve space:

```bash
# The resolver ships inside the installed plugin; the relative path only
# exists in a wenlan checkout.
resolver="$(find "$HOME/.codex/plugins/cache" -path '*/wenlan/*/bin/resolve-space.sh' 2>/dev/null | head -1)"
resolved="$("${resolver:-plugin-codex/bin/resolve-space.sh}" --cwd "$PWD" ${space_arg:+--arg "$space_arg"} 2>/dev/null)"
space="$(printf '%s\n' "$resolved" | cut -f1)"
source_layer="$(printf '%s\n' "$resolved" | cut -f2)"
```

Use an explicit target first. For bare `/distill`, use the resolved space as
the target only when `space` is non-empty. If the resolver returns unscoped,
omit the target.

## Scope rules

- `rebuild <page-id>`: destructive single-page rebuild.
- `deep`: global pass; no target.
- `<target>`: pass the target as supplied.
- bare command with resolved space: pass the resolved space as target.
- bare command without resolved space: omit target.

## Destructive rebuild confirmation

`rebuild <page-id>` calls:

```text
mcp__wenlan__distill(target="<page-id>", force=true)
```

This is destructive: user-edited page prose is wiped and regenerated from source
memories. Require explicit same-turn confirmation before calling the tool unless
the user already wrote an unambiguous rebuild command with the exact id.

If the id is missing or ambiguous, ask for this exact shape:

```text
rebuild <page-id>
```

A bare page id is not confirmation.

## Call the daemon

For non-rebuild flows, call:

```text
mcp__wenlan__distill(target="<scope when present>")
```

Parse the JSON result. If it contains `unresolved` or `hint`, relay it and stop.

The response may include:

- `pending`: clusters to synthesize in this Codex session.
- `stale_pages`: existing pages whose source memories changed.
- `orphan_topics`: wikilink labels referenced by pages but not yet present.
- `created_ids` or `pages_created`: pages the daemon already handled.

## Synthesize pending clusters

For each `pending` cluster:

1. Read all `contents`.
2. Skip low-coherence clusters that merely share an entity while topics scatter.
3. For a new cluster, call:

```text
mcp__wenlan__write_page(
  title="<short noun phrase>",
  summary="<one durable claim>",
  content="<3-7 paragraphs with inline (source: mem_...) citations>",
  entity_id="<cluster entity id if present>",
  space="<cluster space if present>",
  source_memory_ids=[...]
)
```

4. For a refresh candidate, call:

```text
mcp__wenlan__write_page(
  page_id="<existing_page_id>",
  content="<refreshed source-cited prose>",
  source_memory_ids=[...],
  summary="<one refreshed claim>"
)
```

Never create an empty stub page.

## Refresh stale pages

For each `stale_pages` item:

- If `user_edited == true`, do not auto-rewrite. Report the conflict and tell
  the user they can run `/distill rebuild <page-id>` if they want to wipe edits.
- If `user_edited == false`, call `mcp__wenlan__get_page_sources(page_id=...)`,
  synthesize refreshed prose from the sources, then call
  `mcp__wenlan__write_page` with that `page_id`.

## Resolve Markdown paths

After each successful create or update, resolve the on-disk path from
`~/.wenlan/pages/.wenlan/state.json`. Do not derive slugs in Codex.

```bash
python3 - "$page_id" <<'PY'
import json, os, sys
state_path = os.path.expanduser("~/.wenlan/pages/.wenlan/state.json")
pid = sys.argv[1]
filename = None
try:
    with open(state_path) as f:
        filename = json.load(f).get("pages", {}).get(pid, {}).get("file")
except FileNotFoundError:
    pass
print(f"~/.wenlan/pages/{filename}" if filename else "(no md projection on disk)")
PY
```

## Report

Use titles and file paths, not page bodies:

```text
Distilled N page(s) from K memories in scope `<scope>`:
  - <Title>  v1, synthesized from <K> sources
    Open: ~/.wenlan/pages/<slug>.md
```

Also report:

- skipped low-coherence clusters,
- stale pages skipped because the user edited them,
- orphan topic suggestions.

## Auto-commit `~/.wenlan`

Best effort only:

```bash
git -C ~/.wenlan add -A && \
  git -C ~/.wenlan -c user.name=Wenlan -c user.email=daemon@wenlan.local \
    commit --quiet -m "distill: <N> pages" 2>/dev/null || \
(sleep 1 && git -C ~/.wenlan add -A && \
  git -C ~/.wenlan -c user.name=Wenlan -c user.email=daemon@wenlan.local \
    commit --quiet -m "distill: <N> pages" 2>/dev/null) || true
```

Do not fail the distill pass if the audit-trail commit fails.
