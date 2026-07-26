---
name: setup
description: >
  Frictionless setup. Detects a missing local runtime, installs or repairs it,
  configures local memory, and verifies the full plugin -> MCP -> local runtime
  round-trip. Run after `/plugin install wenlan@7xuanlu-wenlan`, or any time the user
  says "set up wenlan", "is wenlan working", "fix wenlan".
allowed-tools: ["Bash", "mcp__plugin_wenlan_wenlan__doctor", "mcp__plugin_wenlan_wenlan__context"]
---

# /setup

Self-healing setup. Goal: two user actions max: install plugin, type `/setup`.
Default backend is local memory: no local model, no API key, no prompts. Local
model and Anthropic key are opt-in upgrades documented in `/help`.

## Steps

Run in order. Stop and report at the first failure that needs human attention.
Otherwise, push through automatically.

### 1. Health probe

```bash
for i in 1 2 3; do
  curl -fsS -m 3 http://127.0.0.1:7878/api/health && break
  sleep 1
done
```

- 200 OK: continue to version drift probe.
- Anything else: continue to bootstrap.

### 2. Version drift probe

Compare daemon version vs plugin manifest version:

```bash
PLUGIN_JSON="${CLAUDE_PLUGIN_ROOT:-plugin}/.claude-plugin/plugin.json"
[ -r "$PLUGIN_JSON" ] || PLUGIN_JSON=".claude-plugin/plugin.json"
if command -v python3 >/dev/null 2>&1 && [ -r "$PLUGIN_JSON" ]; then
  RESP="$(curl -fsS -m 3 http://127.0.0.1:7878/api/health)"
  DAEMON_VER="$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))')"
  EXPECTED_VER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("version",""))' "$PLUGIN_JSON")"
  RELEASE_VER="${EXPECTED_VER%%+*}"
  printf 'daemon=%s expected=%s release=%s\n' "$DAEMON_VER" "$EXPECTED_VER" "$RELEASE_VER"
else
  echo "version_check=skipped"
fi
```

- Same version: continue to doctor.
- If the probe cannot run because the runtime is down: continue to bootstrap.
- If `PLUGIN_JSON` is unreadable or `python3` is missing: continue to doctor;
  the session hook will keep surfacing a mismatch if one exists.
- If mismatch, repair the runtime:

```bash
PLUGIN_JSON="${CLAUDE_PLUGIN_ROOT:-plugin}/.claude-plugin/plugin.json"
[ -r "$PLUGIN_JSON" ] || PLUGIN_JSON=".claude-plugin/plugin.json"
EXPECTED_VER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("version",""))' "$PLUGIN_JSON")"
RELEASE_VER="${EXPECTED_VER%%+*}"
curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/v${RELEASE_VER}/install.sh | bash
export PATH="$HOME/.wenlan/bin:$PATH"
wenlan setup --basic
wenlan background on
```

Then continue to the health and version re-probe below. The installer deliberately
targets the latest stable runtime. Do not downgrade a newer runtime to match a
stale plugin cache.

### 3. Bootstrap

Detect whether the `wenlan` CLI is on PATH:

```bash
command -v wenlan >/dev/null 2>&1 && echo present || echo absent
```

If absent, install and configure local memory:

```bash
curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/v0.15.0/install.sh | bash
export PATH="$HOME/.wenlan/bin:$PATH"
wenlan setup --basic
wenlan background on
```

If present but the local runtime is down:

```bash
wenlan setup --basic 2>/dev/null || true
wenlan background on
```

`wenlan setup --basic` is idempotent. `wenlan background on` starts the
managed background process.

### 4. Re-probe health and version

```bash
for i in 1 2 3 4 5; do
  curl -fsS -m 3 http://127.0.0.1:7878/api/health && break
  sleep 1
done
```

If the local runtime still is not reachable after about five seconds, surface
the error and stop. Likely causes: launchd load failure, port 7878 already in
use, or a local runtime crash.

Once healthy, repeat the version comparison from step 2. If the versions still
differ, stop instead of claiming setup succeeded:

```text
Runtime and plugin versions still differ after repair; update the Wenlan plugin,
restart Claude Code, then run /wenlan:setup again.
```

This usually means the runtime is newer than the plugin cached by the current
Claude Code process. Updating the plugin is safer than silently downgrading the
runtime, and the restart is required before this session can load new plugin
code.

### 5. Doctor

Call the Wenlan MCP `doctor` tool.

```text
doctor()
```

Expected: local memory configured. Capture the mode string for the final report.

### 6. MCP round-trip

Call the Wenlan MCP `context` tool.

```text
context()
```

Pass: continue. Fail: MCP is not wired. Tell the user:
"wenlan-mcp did not respond. Restart Claude Code so the plugin's `.mcp.json`
respawns the server."

### 7. Ready report

Print:

```text
Wenlan ready.
  Runtime:  up on 127.0.0.1:7878
  Mode:     <mode from doctor()>
  MCP:      connected
  Data:     ~/.wenlan/  (pages, sessions, db symlink)
  Try:      /brief, /capture <thing>, /recall <query>, /help
```

If this was the first `/setup` invocation in the session, dispatch `/help` once
so the user sees the verb cheat sheet without asking.

## Optional upgrades

Mention these only if the user asks for richer synthesis:

- `wenlan models install` for local model-backed distillation.
- `wenlan keys set anthropic` for stronger synthesis.

Installing a model or key only makes that provider available; it does not
authorize background inference. If the user wants automatic enrichment, run
`wenlan enrichment status`, help them choose the exact Everyday and Synthesis
sources, then run `wenlan enrichment configure --everyday <source> --synthesis
<source>`. The CLI itself shows the task mapping and cloud/on-device disclosure
and obtains the one confirmation. Never add `--yes` unless the user already
stated that exact mapping. `wenlan enrichment disable` reverses the consent
without deleting the model or key.

Default flow ignores both. Storage, search, recall, and MCP memory all work in
local memory mode; deterministic indexing and sync continue while model-backed
background enrichment is off.

## When to use

- Right after `/plugin install wenlan@7xuanlu-wenlan`.
- Hook printed "local runtime down -- run /wenlan:setup".
- Hook printed "Run /wenlan:setup to repair" for a version mismatch.
- User says "set up wenlan", "is it working", "reinstall wenlan".

## When NOT to use

- Runtime already verified this session: `/brief` instead.
- Editing one config field: `wenlan doctor` or settings file directly.
