# wenlan-cli

Wenlan's product CLI. Use it to set up the local runtime, control background mode, search and recall memory, capture new memories, configure models/API keys, and run doctor checks.

License: Apache-2.0.

## Install

Recommended user setup on macOS Apple Silicon:

```bash
npx -y wenlan setup
```

The setup package downloads `wenlan`, `wenlan-server`, and `wenlan-mcp` into `~/.wenlan/bin/`, configures local memory, starts the background runtime, and verifies status. It currently supports macOS Apple Silicon.

Linux x64/ARM64 has an automated shell path:

```bash
curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | bash
export PATH="$HOME/.wenlan/bin:$PATH"
wenlan setup --basic
wenlan background on
wenlan status
```

Windows x64 users can unpack `wenlan-windows-x64.zip` from [Releases](https://github.com/7xuanlu/wenlan/releases/latest) into a directory on `PATH`. Keep the bundled `onnxruntime.dll` beside the three `.exe` files, then run the same three `wenlan` commands.

### macOS Intel

There is currently no supported complete-runtime install for macOS Intel. The pinned ONNX Runtime dependency does not publish an x86_64 macOS binary, so a custom build requires compiling ONNX Runtime separately and providing it through `ORT_LIB_LOCATION`. A plain `cargo build` is not a working installation path.

For CLI-only local development:

```bash
cargo install --path crates/wenlan-cli
```

Or build only the CLI from the workspace:

```bash
cargo build -p wenlan --release
./target/release/wenlan --help
```

## Configuration

Set `WENLAN_HOST` to point at a remote daemon:

```bash
export WENLAN_HOST=http://127.0.0.1:7878  # default
```

## Subcommands

### `wenlan status`

Show background runtime, model, and API key state.

```bash
wenlan status
wenlan status --format json
```

### `wenlan setup`

Configure Wenlan's runtime mode.

```bash
wenlan setup                  # interactive
wenlan setup --basic          # local memory, no local model or API key
wenlan setup --model qwen3-4b # download/select a local model
wenlan setup --anthropic-api-key-env ANTHROPIC_API_KEY
```

### `wenlan background <on|off>`

Start or stop the per-user background runtime. `wenlan background off` stops the current daemon while preserving its login registration; run `wenlan background on` to start it again. Most users run `wenlan background on` once after setup and `wenlan restart` after upgrades installed outside `wenlan setup`.

```bash
wenlan background on
wenlan background off
wenlan restart
```

### `wenlan enrichment <status|configure|disable>`

Models and provider keys are capability only; they do not authorize automatic
model calls. Inspect the two job groups, confirm exact hard-pinned sources, or
turn model-backed background work off while keeping providers installed:

```bash
wenlan enrichment status
wenlan enrichment configure --everyday on-device --synthesis on-device
wenlan enrichment disable
```

`configure` shows what each job does and discloses cloud cost/data transfer or
on-device CPU/GPU/RAM use before confirmation. An unavailable hard pin pauses;
Wenlan never substitutes another provider. Deterministic storage, indexing, and
sync continue when enrichment is off.

### `wenlan doctor`

Diagnose runtime reachability, background mode, model setup, and API key setup.

```bash
wenlan doctor
```

### `wenlan models`

Manage opt-in local models.

```bash
wenlan models list
wenlan models status
wenlan models install qwen3-4b
wenlan models reranker lite
```

### `wenlan keys`

Manage provider API keys.

```bash
wenlan keys status
wenlan keys set anthropic --env ANTHROPIC_API_KEY
wenlan keys clear anthropic
```

### `wenlan connect <client>`

Configure Wenlan MCP for a supported client. This is the MCP-only path for Claude Code users who do not want the plugin, and for Codex, Cursor, Claude Desktop, VS Code, and Gemini CLI.

```bash
wenlan connect claude-code
wenlan connect codex
wenlan connect cursor --dry-run
```

Supported clients: `claude-code`, `codex`, `gemini`, `cursor`, `claude-desktop`, `vscode`.

When `wenlan-mcp` is installed next to the `wenlan` CLI, the generated config points at that binary. Otherwise it falls back to `npx -y wenlan-mcp`.

Use `--dry-run` to preview JSON config edits before writing them:

```bash
wenlan connect cursor --dry-run
```

### `wenlan search <query>`

Search memories (vector + FTS hybrid).

```bash
wenlan search "embedding model"
wenlan search "rust" --limit 5
wenlan search "rust" --format json | jq '.results[].score'
```

### `wenlan recall <query>`

Get the working memory bundle for a query (pages + decisions + relevant memories + graph context).

```bash
wenlan recall "what we agreed on for the API"
wenlan recall "memory layer" --format json
```

### `wenlan capture [text] [--file <path>] [--type <type>]`

Store a memory. Provide content positionally, via `--file`, or pipe via stdin.

```bash
wenlan capture "remember this insight" --type fact
wenlan capture --file notes.md --type page
echo "stdin pipe content" | wenlan capture --type quick_thought
```

### `wenlan memories [--limit N] [--type X]`

List recent memories.

```bash
wenlan memories
wenlan memories --limit 5
wenlan memories --type fact --format json
```

### `wenlan agents list/show/edit`

Manage registered agents.

```bash
wenlan agents list
wenlan agents show claude-code
wenlan agents edit claude-code --trust trusted --enabled true
```

### `wenlan spaces <list|add|default|move|show>`

Manage memory spaces (buckets).

```bash
wenlan spaces list
wenlan spaces add ideas --default
wenlan spaces show career
wenlan spaces default work
wenlan spaces move scratch career
```

### `wenlan sources add <path>`

Register or resync a file or folder source.

```bash
wenlan sources add ~/Notes
wenlan sources add ~/Notes/project.md
```

## Output formats

- `--format auto` (default): table on TTY, JSON when piped.
- `--format json`: pretty-printed JSON.
- `--format table`: human-readable table.
- `--quiet` / `-q`: suppress success output (errors still go to stderr).

## Links

- [wenlan.app](https://wenlan.app) — project home
- [wenlan.app/docs/get-started](https://wenlan.app/docs/get-started) — install + verify the first memory loop
- [wenlan.app/docs/commands](https://wenlan.app/docs/commands) — Claude Code commands and MCP tools reference
- [wenlan.app/docs/troubleshooting](https://wenlan.app/docs/troubleshooting) — common failure modes
- [github.com/7xuanlu/wenlan](https://github.com/7xuanlu/wenlan) — source
- [github.com/7xuanlu/wenlan/releases/latest](https://github.com/7xuanlu/wenlan/releases/latest) — desktop app and runtime downloads

## License

Apache-2.0. See top-level LICENSE.
