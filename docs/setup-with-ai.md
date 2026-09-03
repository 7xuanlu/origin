# Set up Wenlan with your AI client

Use this guide when an AI assistant is setting up Wenlan on the user's behalf. Configure only the client in the current conversation unless the user explicitly asks for additional clients.

## Target outcome

Setup is complete only when:

1. The Wenlan runtime is healthy.
2. The current AI client has either the Wenlan plugin or an MCP connection.
3. A small capture can be recalled from the same client.

Do not report success after editing configuration alone. A live round trip is the proof.

On the Claude Code plugin path, `/wenlan:setup` already proves the round trip with a read-only `brief` call, which satisfies step 3.

## Install the runtime

Detect the host before choosing the runtime install path:

| Host | Install path |
|---|---|
| macOS Apple Silicon | Run `npx -y wenlan setup` (requires Node.js). Without Node.js, run `curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh \| bash` then `wenlan setup --basic`. |
| Linux x64 or ARM64 | Run the shell installer below. |
| Windows x64 | Download `wenlan-windows-x64.zip` from [Releases](https://github.com/7xuanlu/wenlan/releases/latest). |
| macOS Intel | No supported complete-runtime install; see the [platform note](../crates/wenlan-cli/README.md#macos-intel). |

On Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | bash
export PATH="$HOME/.wenlan/bin:$PATH"
wenlan setup --basic
wenlan background on
wenlan status
```

On Windows, extract the archive as a unit into a user-owned directory on `PATH`. Keep `onnxruntime.dll` beside `wenlan.exe`, `wenlan-server.exe`, and `wenlan-mcp.exe`, then run:

```bash
wenlan setup --basic
wenlan background on
wenlan status
```

## Claude Code

Install the Wenlan plugin from its public marketplace:

```bash
claude plugin marketplace add 7xuanlu/wenlan
claude plugin install wenlan@7xuanlu-wenlan
```

Start a new Claude Code session if requested, then run `/setup`. The setup skill installs or repairs the local runtime and verifies the MCP round trip. Detailed workflows: [Claude Code plugin](../plugin/.claude-plugin/README.md).

## Codex

Install the runtime using the host-specific path above, then install the plugin:

```bash
codex plugin marketplace add 7xuanlu/wenlan
codex plugin add wenlan@7xuanlu-wenlan
```

Start a new Codex task so the plugin and MCP server load, then run `/setup`. Detailed workflows: [Codex plugin](../plugin-codex/README.md).

## MCP-only setup and other clients

Install the runtime using the host-specific path above, then configure only the client the user named:

```bash
wenlan connect <client>
```

Use `claude-code` or `codex` when the user wants MCP without the plugin. Supported values are `claude-code`, `codex`, `cursor`, `claude-desktop`, `vscode`, and `gemini`. The CLI makes a backup before replacing an existing JSON configuration. Full command reference: [CLI and MCP setup](../crates/wenlan-cli/README.md).

## Verify

First confirm the local runtime:

```bash
wenlan status
```

Then use the current client's Wenlan tools to capture a disposable sentence and recall it. Delete the test memory afterward if the client exposes `forget`. If the tools are not visible, start a new client session and verify again before declaring setup complete.
