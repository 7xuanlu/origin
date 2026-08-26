---
name: setup
description: >
  Frictionless Wenlan setup for Codex. Detects a missing local runtime, installs
  or repairs it, and verifies the plugin to MCP to local runtime round-trip. Run
  when the user says "set up wenlan", "is wenlan working", or "fix wenlan".
allowed-tools: ["Bash", "mcp__wenlan__brief"]
user-invocable: true
---

# /setup

Self-healing setup for Codex. Default backend is local memory: no local model,
no API key, no prompt ceremony. Local model and Anthropic key are optional
upgrades after the basic path works.

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
PLUGIN_JSON="${CODEX_PLUGIN_ROOT:-plugin-codex}/.codex-plugin/plugin.json"
[ -r "$PLUGIN_JSON" ] || PLUGIN_JSON=".codex-plugin/plugin.json"
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
  Codex will keep using this slice until the plugin cache is updated.
- If mismatch, check the direction before repairing. Compare the release part
  of `daemon=` (strip the `+g<sha>` suffix) against `release=`, numeric per
  component, not lexicographic. Daemon release equal or newer → the plugin
  cache is stale, not the runtime: skip the repair below (it would only
  reinstall the same-or-latest runtime and restart a healthy daemon) and go
  straight to the stop message in step 4 — update the plugin, restart, rerun.
- Only if the daemon release is older than the plugin release, repair the
  runtime:

```bash
PLUGIN_JSON="${CODEX_PLUGIN_ROOT:-plugin-codex}/.codex-plugin/plugin.json"
[ -r "$PLUGIN_JSON" ] || PLUGIN_JSON=".codex-plugin/plugin.json"
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
curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/v0.17.0/install.sh | bash
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

A first boot downloads the embedding model before it answers, which can take
most of a minute on a slow connection, so poll for up to 60 seconds:

```bash
for i in $(seq 1 60); do
  curl -fsS -m 3 http://127.0.0.1:7878/api/health && break
  sleep 1
done
```

If the local runtime still is not reachable after 60 seconds, print the doctor
output and stop; it names the daemon state and the log files to read:

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
"$W" doctor
```

Likely causes: the model download is still running or failed (read the daemon
log the doctor output points at), launchd load failure, port 7878 already in
use, or a local runtime crash.

Once healthy, repeat the version comparison from step 2. If the versions still
differ, stop instead of claiming setup succeeded:

```text
Runtime and plugin versions still differ after repair; update the Wenlan plugin,
restart Codex, then run /wenlan:setup again.
```

This usually means the runtime is newer than the plugin cached by the current
Codex process. Updating the plugin is safer than silently downgrading the
runtime, and the restart is required before this session can load new plugin
code.

### 5. Doctor

Run the Wenlan CLI doctor through the resolved binary path:

```bash
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
"$W" doctor
```

Expected: local memory configured. Capture the mode string for the final report.

### 6. MCP round-trip

Call the Wenlan MCP `brief` tool as a read-only round-trip.

```text
brief()
```

If it fails, report: "wenlan-mcp did not respond through Codex. Start a new
Codex thread after reinstalling the plugin so Codex respawns the MCP server."

### 7. Ready report

Print:

```text
Wenlan ready.
  Runtime:  up on 127.0.0.1:7878
  Mode:     <mode from CLI doctor>
  MCP:      connected
  Data:     ~/.wenlan/
  Try:      /brief, /capture <thing>
```

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

## Codex note

This Codex slice has no session-start hook. `/setup` is the explicit health and
version check until Codex exposes a lifecycle hook this plugin can use.
