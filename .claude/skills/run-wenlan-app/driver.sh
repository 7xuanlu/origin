#!/usr/bin/env bash
# Agent driver for the wenlan-app Tauri dev build. See SKILL.md next to this file.
# Subcommands: build | vite | launch | shot [out.png] | stop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)" # unit root (repo or worktree)
# Reuse the main checkout's warm cargo cache and sibling backend even from
# a linked worktree (resolve-backend-dir.sh fallbacks don't reach
# .claude/worktrees/<name>, which is 3 levels deep).
MAIN="$(cd "$(git -C "$ROOT" rev-parse --git-common-dir)/.." && pwd)"
export WENLAN_BACKEND_DIR="${WENLAN_BACKEND_DIR:-$MAIN/../wenlan}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$MAIN/target}"
BIN="$CARGO_TARGET_DIR/debug/wenlan-app"
LOG="${TMPDIR:-/tmp}"

# A debug wenlan-app refuses to start without a complete isolated runtime
# identity (see validate_debug_runtime_isolation in app/src/lib.rs), so the
# driver takes the same worktree-scoped config `pnpm dev:all` uses. This is
# also what keeps an agent launch off the user's daemon and LaunchAgents.
load_dev_runtime_config() {
  while IFS='=' read -r key value; do
    case "$key" in
      WENLAN_PORT | WENLAN_DEV_UI_PORT | WENLAN_DEV_REMOTE_PORT_START | WENLAN_DEV_APP_ID | WENLAN_DEV_TAURI_MCP_SOCKET | WENLAN_DATA_DIR | WENLAN_DEV_STATE_DIR)
        export "$key=$value"
        ;;
    esac
  done < <(bash "$ROOT/scripts/dev-runtime.sh" print-config)
}

case "${1:-}" in
  build)
    # Decoupled from `tauri dev`: its beforeDevCommand compiles sidecars and
    # times out the CLI's 180s wait for vite on a cold cache.
    cd "$ROOT"
    pnpm prepare:sidecars
    cargo build --manifest-path app/Cargo.toml
    ;;
  vite)
    cd "$ROOT"
    # Pinned to :1420 on purpose: a directly-launched prebuilt binary loads the
    # devUrl baked into app/tauri.conf.json, which the tauri CLI's --config
    # override (used by scripts/dev-all.sh) cannot reach here.
    if ! curl -s -o /dev/null http://localhost:1420/; then
      WENLAN_DEV_UI_PORT=1420 nohup pnpm dev >"$LOG/wenlan-vite.log" 2>&1 &
    fi
    for _ in $(seq 1 30); do
      curl -s -o /dev/null http://localhost:1420/ && break
      sleep 2
    done
    curl -s -o /dev/null -w 'vite: %{http_code}\n' http://localhost:1420/
    ;;
  launch)
    # The app inherits this shell's seatbelt sandbox, and the sandbox denies
    # the mach lookup for the ViewBridge XPC service that backs NSOpenPanel.
    # The app starts fine and dies hours later, on the first file/folder
    # picker: +[NSOpenPanel openPanel] returns NULL and objc2 panics, taking
    # the whole process with it — far enough from the launch to look like a
    # product bug. pbpaste needs the same class of mach lookup, so it is a
    # free canary for "this shell is sandboxed".
    if ! /usr/bin/pbpaste >/dev/null 2>&1; then
      echo "REFUSING TO LAUNCH: this shell is sandboxed." >&2
      echo "The app would inherit the sandbox and hard-crash on the first" >&2
      echo "file/folder picker (NSOpenPanel -> NULL -> panic). Re-run this" >&2
      echo "command with the sandbox disabled." >&2
      exit 1
    fi
    load_dev_runtime_config
    # The app now selects the worktree daemon port, so give it a daemon there.
    # This never touches the user's :7878 instance.
    bash "$ROOT/scripts/dev-runtime.sh" start >"$LOG/wenlan-dev-daemon.log" 2>&1 ||
      echo "warning: isolated dev daemon did not start; see $LOG/wenlan-dev-daemon.log" >&2
    "$0" vite
    [ -x "$BIN" ] || "$0" build
    # Launch the binary directly: spawning through the pnpm/tauri chain in a
    # sandboxed shell panics on the rolling-log appender (PermissionDenied,
    # ~/Library/Logs/com.wenlan.desktop/).
    nohup "$BIN" >"$LOG/wenlan-app.log" 2>&1 &
    sleep 5
    if pgrep -f 'target/debug/wenlan-app' >/dev/null; then
      echo "APP UP"
    else
      echo "APP DEAD"
      tail -20 "$LOG/wenlan-app.log"
      exit 1
    fi
    ;;
  shot)
    # ScreenCaptureKit window capture — works while occluded; plain
    # `screencapture -l/-R` both fail ("could not create image from window").
    swift "$(dirname "$0")/wincap.swift" "${2:-$LOG/wenlan-shot.png}"
    ;;
  stop)
    # Only what we started. NEVER kill the user's daemon on :7878 —
    # dev-runtime.sh stop refuses any PID it did not record for this worktree.
    pkill -f 'target/debug/wenlan-app' || true
    lsof -ti:1420 | xargs kill 2>/dev/null || true
    bash "$ROOT/scripts/dev-runtime.sh" stop || true
    ;;
  *)
    echo "usage: driver.sh build|vite|launch|shot [out.png]|stop"
    exit 2
    ;;
esac
