# Wenlan Codex plugin

The Codex plugin wraps the shared `wenlan-mcp` server with Codex slash skills.
Use it when you want the session workflow commands in Codex, not just raw MCP
tools.

## Install

Install the runtime using the host-specific path in [Set up Wenlan with your AI client](../docs/setup-with-ai.md). On macOS Apple Silicon, that step is:

```bash
npx -y wenlan setup
```

Then install the Codex plugin:

```bash
codex plugin marketplace add 7xuanlu/wenlan
codex plugin add wenlan@7xuanlu-wenlan
```

Start a new Codex thread after installing so the skills and MCP server load.
Then try `/setup`, `/brief`, `/capture <memory>`, `/recall <query>`,
`/lint [deep|repair] [scope]`, `/pages <query>`, or `/handoff`.

## Development

After editing `plugin-codex`, reinstall it into Codex's plugin cache:

```bash
python3 ~/.codex/skills/.system/plugin-creator/scripts/update_plugin_cachebuster.py plugin-codex
codex plugin add wenlan@7xuanlu-wenlan
```

The plugin runner uses `~/.wenlan/bin/wenlan-mcp` when available and falls back
to `npx -y wenlan-mcp@^0.15.0`. It passes `--agent-name codex` so captures are
labeled as Codex writes.

Before changing plugin skills, manifests, MCP runner wiring, or the local
marketplace, run the shared Claude/Codex contract checks:

```bash
python3 scripts/validate-codex-plugin-slice.py
python3 scripts/validate-plugin-contract.py
bash scripts/validate-plugin-contract.test.sh
```
