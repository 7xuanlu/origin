---
name: pages
description: >
  Browse or open distilled pages, reusing the `wenlan pages` CLI for the actual
  open (one implementation on your PATH, no setup). `/pages <query>` opens the
  page when one matches, or shows a small native picker when 2-4 match. `/pages`
  lists recent pages. Invoked as `/pages [query]`.
argument-hint: "[query]"
allowed-tools: ["Bash", "AskUserQuestion", "mcp__plugin_wenlan_wenlan__get_page_revisions"]
---

# /pages

Open a distilled page fast. The **`wenlan pages`** CLI does the open (one local
command on your PATH — nothing to register, source, or install). This skill only
adds a small native picker for the ambiguous case.

**Never read a page body. Never render a metadata card.** Resolve by title /
filename only — the file opens in your editor, which renders and searches it
better than chat. Keeping bodies out is what keeps this cheap.

Resolve the binary once (the agent shell's PATH may omit `~/.wenlan/bin`):

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
```

## `/pages <query>` — open, or pick when ambiguous

Run ONE Bash block **with the command sandbox DISABLED** (opening calls the OS
launcher `open`/`xdg-open`, which a sandboxed shell can't reach, and would else
fail with `kLSUnknownErr`):

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
"$W" pages "<query>"
```

Act on its output — do NOT re-fetch, list again, or read bodies:

- **`Opened <path>`** → one page matched and is now open. Print that line. Done.
- **`no page matches: …`** → print it; suggest `/distill <query>`.
- **`N matches for … `** followed by `title · filename` lines → several matched:
  - **2-4 matches** → present them with `AskUserQuestion`, one option each:
    **label** = title, **description** = filename (titles can repeat, so the
    filename is the real handle). On pick, open it:

    ```bash
    "$W" pages "<picked-filename>"
    ```
    Print the `Opened` line it returns.
  - **more than 4 matches** → print the list verbatim and ask the user to narrow
    the query or pass one of the filenames. No picker — it can't show more than 4.

The picker stays small by construction: ≤4 short options, no page bodies, and
the open is a single CLI call that prints one line.

## `/pages` — list recent

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
"$W" --format table pages
```

Prints the newest ~20 titles (CLI-capped; `--limit 0` for all). Show them and
tell the user to `/pages <query>` to open one. No picker over the whole set —
that's the lossy, 4-cap case the CLI list avoids.

If `wenlan` isn't found, the CLI isn't installed — tell the user to run `/setup`.

## Explicit changelog lookup

Only when the user explicitly asks for a page's changelog or version history,
resolve its stable id without reading the page body. Resolve the page ID first
with `wenlan pages "<query-or-filename>" --resolve-id`.

```bash
"$W" pages "<query-or-filename>" --resolve-id
```

If several pages match, use the same picker/narrowing flow above and rerun with
the selected filename. Then call `get_page_revisions(page_id="<resolved-id>")`.
Do not fetch changelogs during ordinary `/pages` list/open flows.

## When NOT to use

- Raw memory lookups → `/recall`.
- Synthesize a new page → `/distill`.
