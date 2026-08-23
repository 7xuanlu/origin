# Privacy Policy

Wenlan is a local-first personal memory system. This policy covers the Wenlan daemon, CLI, MCP server, and Claude Code plugin.

## What data Wenlan stores

Only what you explicitly capture: decisions, lessons, observations, project context, and wiki pages synthesized from those memories. Wenlan does not monitor, scrape, or ingest anything automatically.

## Where data is stored

All data stays on your machine:

- `~/.wenlan/pages/` -- wiki pages (Markdown)
- `~/.wenlan/sessions/` -- session logs (Markdown)
- `~/.wenlan/sources/` -- uploaded source documents
- `~/.wenlan/spaces.toml` -- Space mapping config
- `~/.wenlan/db/` -- symlink to the platform data directory (macOS: `~/Library/Application Support/wenlan/`; Linux: `~/.local/share/wenlan/`; Windows: `%LOCALAPPDATA%\wenlan\`), with the libSQL database at `<data dir>/memorydb/`
- `~/.wenlan/bin/` -- installed binaries

The daemon listens on `127.0.0.1:7878` (localhost only). No data is sent to any remote server by default.

## Third-party services

None by default. Two opt-in integrations exist:

- **Anthropic API (BYOK):** If you run `wenlan keys set anthropic`, your memories are sent to the Anthropic API for richer extraction and synthesis. Anthropic's privacy policy applies to that data. Wenlan does not store or relay your API key beyond the local config file.
- **On-device model:** If you run `wenlan models install`, a Qwen model is downloaded from Hugging Face Hub. No memory data leaves your machine in this mode.

## Telemetry

None. Wenlan collects no usage analytics, crash reports, or diagnostics.

## Data deletion

- Delete individual memories: `/forget` skill.
- Delete everything: remove `~/.wenlan/` and your platform data directory (`~/Library/Application Support/wenlan/` on macOS, `~/.local/share/wenlan/` on Linux, `%LOCALAPPDATA%\wenlan\` on Windows).
- Uninstall the daemon: run `wenlan background off` to stop it and disable autostart (this is a reversible runtime stop, not an uninstall), then remove the service registration for your platform -- `~/Library/LaunchAgents/com.wenlan.server.plist` (macOS), `~/.config/systemd/user/wenlan-server.service` (Linux), or the `WenlanServer` scheduled task (Windows) -- and delete `~/.wenlan/bin/`.

## Contact

Questions or concerns: open an issue at https://github.com/7xuanlu/wenlan/issues.

Last updated: 2026-05-10.
