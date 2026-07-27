---
name: forget
description: >
  Delete a memory from Wenlan by ID. Destructive and cannot be undone —
  prefer `/capture` with `supersedes` for corrections. Invoked as
  `/forget <source_id>`.
argument-hint: "<source_id>"
allowed-tools: ["Bash", "mcp__plugin_wenlan_wenlan__forget", "mcp__plugin_wenlan_wenlan__recall"]
---

# /forget

Permanently delete a memory by its `source_id`.

## How to invoke

You need the `source_id`. If the user did not provide it, call
`/recall` first to find the matching memory and confirm with the
user before deleting.

```
forget(memory_id="<source_id>")
```

## When to use

- User says "forget this", "delete that", "that's wrong, remove it".
- User explicitly identifies a memory by ID.

## When NOT to use

- For corrections, prefer storing a new memory with `supersedes` pointing
  at the old one. That preserves history. Use `/capture` with the
  `supersedes` arg instead.
- Bulk deletions — call `/curate` first, confirm with the user,
  then delete one at a time.

## Safety

Deletion is destructive. Always confirm with the user before calling forget,
unless the user has already given an explicit, unambiguous instruction in
the same turn (e.g. they pasted the ID and said "delete this").

## Auto-commit ~/.wenlan/

If the deletion removed a page md (daemon archives the page and
KnowledgeWriter unlinks the file), snapshot the change. Defensive —
silent skip if `git` missing, `~/.wenlan/` not a repo, or no diff.

```
Bash: git -C ~/.wenlan add -A && \
      git -C ~/.wenlan -c user.name=Wenlan -c user.email=daemon@wenlan.local \
          commit --quiet -m "forget: <source_id>" 2>/dev/null || \
      (sleep 1 && git -C ~/.wenlan add -A && \
       git -C ~/.wenlan -c user.name=Wenlan -c user.email=daemon@wenlan.local \
           commit --quiet -m "forget: <source_id>" 2>/dev/null) || true
```

The retry handles index.lock races — the daemon may be writing to
`~/.wenlan/` at the same moment. Commits land at session boundaries
(handoff or daemon events), not per capture; uncommitted page edits
between sessions are normal. One second is enough for the daemon to
release the lock.
