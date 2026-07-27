---
name: pages
description: >
  List or open distilled Wenlan pages from Codex by delegating to the local
  `wenlan pages` CLI. Invoked as /pages [query].
argument-hint: "[query]"
allowed-tools: ["Bash", "mcp__wenlan__get_page_revisions"]
user-invocable: true
---

# /pages

Open or list distilled Wenlan pages without pulling page bodies into chat.
The `wenlan pages` CLI owns title and filename matching plus the OS editor
open. This skill is only the Codex slash workflow wrapper.

Never read a page body. Never render a metadata card. Resolve by title or
filename only; the page opens in the user's editor where Markdown search and
navigation belong.

## Resolve the CLI

Use the agent shell's PATH first, then the standard Wenlan install path:

```bash
if command -v wenlan >/dev/null 2>&1; then
  W="$(command -v wenlan)"
elif [ -x "$HOME/.wenlan/bin/wenlan" ]; then
  W="$HOME/.wenlan/bin/wenlan"
else
  echo "wenlan not found"
  exit 127
fi
```

If `wenlan` is not found, tell the user to run `/setup`.

## `/pages <query>`

Run one Bash block:

```bash
if command -v wenlan >/dev/null 2>&1; then
  W="$(command -v wenlan)"
elif [ -x "$HOME/.wenlan/bin/wenlan" ]; then
  W="$HOME/.wenlan/bin/wenlan"
else
  echo "wenlan not found"
  exit 127
fi
"$W" pages "<query>"
```

Act only on the command output:

- `Opened <path>`: print that line. The page is open.
- A bare path: print it and say the OS launcher did not open it automatically.
- `no page matches: ...`: print it and suggest `/distill <query>`.
- `N matches for ...` followed by `title  ·  filename` lines: several pages
  matched. If several pages match, print the CLI output. Ask the user to rerun
  `/pages` with one of the filenames or a narrower query.

Do not use a picker. Do not re-fetch, list again, or read any page file.

## `/pages`

Run one Bash block:

```bash
if command -v wenlan >/dev/null 2>&1; then
  W="$(command -v wenlan)"
elif [ -x "$HOME/.wenlan/bin/wenlan" ]; then
  W="$HOME/.wenlan/bin/wenlan"
else
  echo "wenlan not found"
  exit 127
fi
"$W" --format table pages
```

Print the newest topics exactly as the CLI returns them. The CLI caps the list
by default and prints `--limit 0` when the user wants all topics. Tell the user
to run `/pages <query>` to open one.

## Explicit changelog lookup

Only when the user explicitly asks for a page's changelog or version history,
call:

```text
mcp__wenlan__get_page_revisions(page_id="<page-id>")
```

Do not fetch changelogs during ordinary `/pages` list/open flows.

## When not to use

- Raw memory lookup: use `/recall`.
- Synthesize a new page: use `/distill`.
