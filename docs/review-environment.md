# Wenlan Review environment

Use the Review flavor for UI work that needs realistic native rendering without touching the production Wenlan daemon or user data.

```bash
pnpm review
```

This builds and opens `target/debug/bundle/macos/Wenlan Review.app` with bundle identifier `com.wenlan.desktop.review`. It is a compile-time fixture shell: no production commands, daemon, launchd repair, sidecars, source sync, file watchers, remote tunnel, updater, tray, MCP socket, or global shortcuts are registered. Its WebView imports resolve to the deterministic in-memory runtime in `review/tauri-core.ts`, unknown commands fail closed, and the Review CSP disallows network connections.

Review data starts from `e2e/fixtures/spacesNavigation.ts` on every launch. Edits are process-local. The in-app reset control clears Review-only browser state and reloads the fixture; the production bundle identifier, WebKit storage, daemon, logs, and data paths are separate.

Useful commands:

```bash
pnpm dev:review:web   # fixture-only browser surface on port 1422
pnpm build:review:web # build only the fixture frontend
pnpm review:build     # build the native fixture-only app
pnpm review:verify    # verify Review identity and absence of sidecars
pnpm review:open      # clean-restart the already-built native app
```

Do not reuse `.omo` audit bundles for native testing. They predate the compile-time Review runner and may execute the production startup path.

Before calling a Review bundle isolated, verify its Info.plist identity, confirm `Contents/MacOS` contains no sidecars, and check the running process has no TCP connection to port 7878.
