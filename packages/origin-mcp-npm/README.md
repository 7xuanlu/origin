# origin-mcp

Personal memory layer for AI agents — MCP server integration.

## Quick Start

Add to your MCP client config (Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "origin": {
      "command": "npx",
      "args": ["-y", "origin-mcp"]
    }
  }
}
```

For the stable `latest` channel, on first run, `npx origin-mcp` will:

1. Download the `origin-mcp` and `origin-server` binaries for your platform into `~/.origin/bin/`
2. Start the Origin daemon (`origin-server`) if it isn't already running
3. Launch the MCP server

Subsequent runs skip the download and reuse the cached binaries.

## Prerelease Channels

Stable installs follow the latest published full GitHub release.

The Quick Start and runtime details above describe the stable `latest` channel. Prerelease npm installs use tagged binary caches and isolated runtime defaults instead.

To test a prerelease without changing the default channel, publish the npm wrapper under a prerelease version (for example `0.2.0-alpha.1`) and optionally point an npm dist-tag like `alpha` or `beta` at it:

```sh
npx -y origin-mcp@alpha
```

Or pin an exact release tag:

```sh
ORIGIN_RELEASE_TAG=v0.2.0-alpha.1 npx -y origin-mcp
```

The same exact tag can be downloaded with the shell installer for a manual isolated daemon setup:

```sh
curl -fsSL https://raw.githubusercontent.com/7xuanlu/origin/v0.2.0-alpha.1/install.sh | \
  ORIGIN_RELEASE_TAG=v0.2.0-alpha.1 bash
```

Prerelease binaries are cached under `~/.origin/releases/<tag>/` so they do not overwrite the default stable install.

The prerelease wrapper also uses an isolated local daemon port and data dir by default, so `origin-mcp@alpha` does not take over the stable `latest` runtime.

If you want custom isolation instead of the tag-derived defaults, override them explicitly:

```sh
ORIGIN_RELEASE_TAG=v0.2.0-alpha.1 \
ORIGIN_PORT=7879 \
ORIGIN_DATA_DIR="$HOME/Library/Application Support/origin-alpha" \
npx -y origin-mcp
```

## What It Does

[Origin](https://github.com/7xuanlu/origin) is a local-first memory server for macOS. AI agents write what they learn; you curate. The MCP server exposes your memory store to any MCP-compatible client.

Stable `latest` defaults:

- All data stays on your machine (`~/Library/Application Support/origin/`)
- The daemon runs headlessly in the background on `127.0.0.1:7878`
- No cloud account required

## Manual Install

If you prefer to install without `npx`:

```sh
curl -fsSL https://raw.githubusercontent.com/7xuanlu/origin/main/install.sh | bash
export PATH="$HOME/.origin/bin:$PATH"
origin-server install
origin-server status
```

Or download binaries directly from the [Releases page](https://github.com/7xuanlu/origin/releases).

## Requirements

- macOS (Apple Silicon)
- Node.js 18+ (for `npx`)

## License

This npm wrapper is MIT.

Downloaded binaries keep their own licenses:

- `origin-mcp`: MIT
- `origin-server`: Apache-2.0

The desktop app and frontend in the `origin` repo are AGPL-3.0-only and are not installed by this package.
