#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKTREE_ID="$(printf '%s' "$REPO_ROOT" | cksum | awk '{ print $1 }')"
TMP_BASE="${TMPDIR:-/tmp/}"
STATE_DIR="${WENLAN_DEV_STATE_DIR:-${TMP_BASE%/}/wenlan-app-dev/$WORKTREE_ID}"
DEV_PORT="${WENLAN_DEV_PORT:-$((17000 + WORKTREE_ID % 1000))}"
DEV_UI_PORT="${WENLAN_DEV_UI_PORT:-$((18000 + WORKTREE_ID % 1000))}"
DEV_REMOTE_PORT_START="${WENLAN_DEV_REMOTE_PORT_START:-$((20000 + (WORKTREE_ID % 1000) * 4))}"
DEV_APP_ID="${WENLAN_DEV_APP_ID:-com.wenlan.desktop.dev.$WORKTREE_ID}"
DEV_DATA_DIR="${WENLAN_DEV_DATA_DIR:-$STATE_DIR/data}"
DEV_TAURI_MCP_SOCKET="${WENLAN_DEV_TAURI_MCP_SOCKET:-$STATE_DIR/tauri-mcp.sock}"
PID_FILE="$STATE_DIR/wenlan-server.pid"
SERVER_PATH_FILE="$STATE_DIR/wenlan-server.path"
PORT_FILE="$STATE_DIR/wenlan-server.port"
DATA_DIR_FILE="$STATE_DIR/wenlan-server.data-dir"
SERVER_LOG="$STATE_DIR/wenlan-server.log"
LOCK_DIR="$STATE_DIR/runtime.lock"
LOCK_OWNER_FILE="$LOCK_DIR/pid"
STARTED_RUNTIME=0
DAEMON_STAGE_DIR="$STATE_DIR/daemon"

# --- Host process primitives -------------------------------------------------
# Everything below keeps its original POSIX implementation and gains a Git Bash
# branch. Windows needed one because none of the POSIX pieces exist there: no
# `lsof`, and an MSYS `ps` with neither `-p` nor `-o`. A process launched from
# bash also carries two identities, the MSYS pid that `$!` yields and the
# Windows pid that every Windows tool reports, so the runtime records the
# Windows pid and compares like with like.
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) HOST_IS_WINDOWS=1 ;;
  *) HOST_IS_WINDOWS=0 ;;
esac

# Values printed for a consumer outside this shell. The desktop app calls
# `std::fs::canonicalize` on the directories it is handed, and a Windows build
# cannot resolve an MSYS path like /tmp/wenlan-app-dev/<id>. Only the boundary
# converts; the script keeps working in the shell's own form.
native_path() {
  if (( HOST_IS_WINDOWS == 1 )); then
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

# `ps -W` reports the MSYS spelling for a process it launched and the Windows
# spelling for anything else, drops the .exe from the first, and does not
# promise a case. The recorded server path is whatever the caller wrote. Both
# sides go through this so the identity check compares one spelling.
normalize_program_path() {
  local path="${1%.[eE][xX][eE]}"
  if (( HOST_IS_WINDOWS == 1 )); then
    path="$(cygpath -m "$path" 2>/dev/null || printf '%s' "$path")"
    printf '%s' "$path" | tr '[:upper:]' '[:lower:]'
  else
    printf '%s' "$path"
  fi
}

process_is_alive() {
  if (( HOST_IS_WINDOWS == 1 )); then
    tasklist //FI "PID eq $1" //NH //FO CSV 2>/dev/null | grep -q '^"'
  else
    kill -0 "$1" 2>/dev/null
  fi
}

process_image_path() {
  if (( HOST_IS_WINDOWS == 1 )); then
    # Column 4 is WINPID and column 8 onward is the image path, which can
    # contain spaces.
    ps -W | awk -v want="$1" '
      $4 == want {
        for (i = 8; i <= NF; i++) printf "%s%s", (i > 8 ? " " : ""), $i
        print ""
        exit
      }
    '
  else
    ps -p "$1" -o command= 2>/dev/null || true
  fi
}

# Kill a Windows process only if it is still the recorded executable.
#
# `taskkill` can filter on a pid and on an image *name*, and a production
# install and every other worktree's dev daemon are all called
# wenlan-server.exe, so that filter cannot tell them apart from this one.
# Checking the full path in a separate call would leave the reuse window open:
# the daemon can exit and Windows can hand its pid to one of those neighbours
# before the kill lands.
#
# Opening the process first closes that window. Windows keeps a pid reserved for
# as long as any handle to it is open, so once `$p.Handle` materializes, the
# path this reads and the process this kills are the same one. `MainModule` and
# the kill both address it by pid, which is now pinned. A path that cannot be
# read is not a match, so an identity this cannot prove kills nothing; the
# caller sees the process fail to exit rather than a stranger disappear.
kill_by_image_path() {
  MSYS2_ARG_CONV_EXCL='*' WENLAN_KILL_PID="$1" WENLAN_KILL_PATH="$2" \
    powershell.exe -NoProfile -NonInteractive -Command '
      $want = ($env:WENLAN_KILL_PATH -replace "/", "\")
      try { $p = [System.Diagnostics.Process]::GetProcessById([int]$env:WENLAN_KILL_PID) }
      catch { exit 0 }
      try { $null = $p.Handle } catch { exit 0 }
      try { $got = $p.MainModule.FileName } catch { exit 0 }
      if ($got -ieq $want) { $p.Kill() }
    ' >/dev/null 2>&1 || true
}

# Windows cannot deliver SIGTERM to a console process that owns no console, so
# there is no graceful stop to try first; a forced kill is the only stop that
# works. The daemon's durability is SQLite WAL, which is crash-safe, and the
# POSIX path already force-kills anything that ignores SIGTERM for 5s.
#
# Both take the recorded server path as a second argument. Neither kills a
# process tree: `kill -KILL` does not on POSIX, so the Windows side does not
# either, and a tree kill is exactly what should not follow an identity check
# that only covers the root.
terminate_process() {
  if (( HOST_IS_WINDOWS == 1 )); then
    kill_by_image_path "$1" "$2"
  else
    kill "$1"
  fi
}

force_terminate_process() {
  if (( HOST_IS_WINDOWS == 1 )); then
    kill_by_image_path "$1" "$2"
  else
    kill -KILL "$1"
  fi
}

# The Windows pid of a job just backgrounded from this shell.
#
# `nohup env ... server &` is a chain of MSYS processes that ends in an exec of
# a native binary, and MSYS spawns a fresh Windows process for that last step.
# Until it does, the MSYS pid still maps to the WINPID of `env`, which then
# exits. Taking the first WINPID that appears therefore records a pid that is
# dead moments later, so this waits for the row to name the program that was
# actually launched.
windows_pid_for_job() {
  local job_pid="$1" program="$2" want row winpid command
  want="$(normalize_program_path "$program")"
  for _ in $(seq 1 100); do
    # One snapshot yields both fields, so the pid and the image it names cannot
    # come from either side of an exec.
    row="$(ps -W | awk -v p="$job_pid" '
      $1 == p {
        printf "%s", $4
        for (i = 8; i <= NF; i++) printf " %s", $i
        print ""
        exit
      }
    ')"
    winpid="${row%% *}"
    command="${row#* }"
    if [[ "$winpid" =~ ^[0-9]+$ ]] && (( winpid > 0 )) &&
      [[ "$(normalize_program_path "$command")" == "$want" ]]; then
      printf '%s\n' "$winpid"
      return 0
    fi
    sleep 0.1
  done
  return 1
}

# Stop a daemon that was launched but never resolved to a pid.
#
# `windows_pid_for_job` gives up after ten seconds, and a daemon that is merely
# slow is still alive when it does. Nothing recorded it, and `dev-all.sh` only
# registers the runtime once `start` succeeds, so no later `stop` will find it
# either -- it goes on to bind the dev port and stays there unmanaged.
#
# Looking for the port is not enough: the process may not have bound it yet at
# the moment the snapshot is taken. The staging directory is private to this
# worktree, so the executable itself is the identity. Anything running that
# exact image is this runtime's own daemon, bound or not, and this keeps looking
# for as long as the health probe downstream would have waited. The `ps` scan is
# prefiltered by name so only the handful of candidate rows are canonicalized.
reap_staged_daemon() {
  local server="$1" want winpid image
  want="$(normalize_program_path "$server")"
  for _ in $(seq 1 50); do
    while read -r winpid image; do
      [[ "$winpid" =~ ^[0-9]+$ ]] || continue
      [[ "$(normalize_program_path "$image")" == "$want" ]] || continue
      force_terminate_process "$winpid" "$server"
      for _ in $(seq 1 50); do
        process_is_alive "$winpid" || return 0
        sleep 0.1
      done
      return 1
    done < <(ps -W | awk 'tolower($0) ~ /wenlan-server/ {
      printf "%s", $4
      for (i = 8; i <= NF; i++) printf " %s", $i
      print ""
    }')
    sleep 0.2
  done
  return 1
}

# Tauri's build script copies the staged sidecars and their runtime DLLs into
# the cargo target directory. A daemon running out of that directory holds
# onnxruntime.dll open, and Windows fails the copy with os error 32, so
# `pnpm dev:all` would deadlock against its own daemon. Running from a private
# copy under the dev state directory keeps the two out of each other's way, and
# incidentally makes the recorded server path unique to this worktree.
stage_windows_daemon() {
  local source="$1" source_dir dll
  source_dir="$(dirname "$source")"
  mkdir -p "$DAEMON_STAGE_DIR"
  if ! cp -u "$source" "$DAEMON_STAGE_DIR/wenlan-server.exe"; then
    echo "error: could not stage the dev daemon in $DAEMON_STAGE_DIR" >&2
    echo "       a daemon may still be running from it; try dev-runtime.sh stop" >&2
    return 1
  fi
  # The loader resolves these from the executable's own directory, so they have
  # to travel with it. app/binaries is the staging area prepare-sidecars.sh
  # fills, and is the only source on a checkout that has not built the app yet.
  for dll in "$REPO_ROOT/app/binaries"/*.dll "$source_dir"/*.dll; do
    [[ -f "$dll" ]] || continue
    cp -u "$dll" "$DAEMON_STAGE_DIR/" || true
  done
}

# One directory has many spellings on Windows -- a different case, a DOS 8.3
# alias, a `\\?\` prefix, a trailing dot, a junction -- and the production-root
# guard below compares these as strings, so every spelling it cannot reduce is a
# way past it. Normalizing them by hand is a losing game, so this asks the OS
# instead: `fs.realpathSync.native` goes through GetFinalPathNameByHandle on
# Windows and answers with the real on-disk path, and is realpath(3) on POSIX,
# which is what this always used. Only something that exists can be resolved, so
# the non-existent tail is peeled off first and re-attached afterwards.
canonicalize_path() {
  node -e '
    const fs = require("node:fs");
    const path = require("node:path");
    let resolved = path.resolve(process.argv[1]);
    // Win32 drops trailing dots and spaces from every component, so
    // `...\wenlan.` opens `...\wenlan`. Node resolves through the extended
    // `\\?\` form, where they are literal instead, so it would call that a
    // different and missing directory and hand the guard a path that never
    // matches -- while the daemon, going through Win32, writes to the real one.
    // Drop them here so both see the same directory. The drive is left alone,
    // and `.` and `..` are already gone by now.
    //
    // Only ordinary Win32 paths reach this: a caller that writes `\\?\` or
    // `\\.\` is refused upstream, because nothing here can carry that prefix
    // through -- `realpathSync.native` answers without it -- so the literal
    // trailing characters it promises would stop being literal right here.
    if (process.platform === "win32") {
      resolved = resolved
        .split(path.sep)
        .map((part, index) => (index === 0 ? part : part.replace(/[. ]+$/, "")))
        .join(path.sep);
    }
    let suffix = "";
    for (;;) {
      try {
        resolved = fs.realpathSync.native(resolved);
        break;
      } catch {
        const parent = path.dirname(resolved);
        if (parent === resolved) break;
        suffix = "/" + path.basename(resolved) + suffix;
        resolved = parent;
      }
    }
    process.stdout.write(resolved.split(path.sep).join("/") + suffix);
  ' "$1"
}

# Windows resolves paths case-insensitively, and `realpath` hands back whatever
# casing the caller wrote, so %LOCALAPPDATA%\WENLAN and %LOCALAPPDATA%\wenlan
# reach here as two different strings naming one directory. Comparing them
# literally lets the second spelling walk past the production-root guard and
# point the dev daemon at the real data directory. Fold case there, and only
# there: on Linux those are genuinely two directories, and folding would refuse
# a path that is not production at all.
path_is_within() {
  local child="$1" parent="$2"
  if (( HOST_IS_WINDOWS == 1 )); then
    child="$(printf '%s' "$child" | tr '[:upper:]' '[:lower:]')"
    parent="$(printf '%s' "$parent" | tr '[:upper:]' '[:lower:]')"
  fi
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

# Refuse an extended-length or device path instead of quietly rewriting it.
#
# `\\?\` and `\\.\` mean "pass this to the filesystem unchanged": trailing dots
# and spaces are part of the name rather than something Win32 discards. Nothing
# downstream can honour that. `realpathSync.native` answers without the prefix,
# MSYS `mkdir` and `realpath` go through Win32, and so does the daemon -- so a
# verbatim path arrives at the guard as an ordinary one whose components no
# longer mean what the caller wrote. `\\?\%LOCALAPPDATA%\wenlan.\dev` would pass
# the production check as a sibling directory and then be opened as
# `...\wenlan\dev`, inside production.
#
# Stripping the components instead redirects the path, which is the opposite
# error: it would send a genuine `\\?\C:\scratch\dev.\data` to `...\dev\data`.
# Neither is acceptable silently, so this says no and names the fix. Windows
# only: on POSIX a leading `\\` is an ordinary relative filename.
reject_verbatim_path() {
  local label="$1" value="$2"
  (( HOST_IS_WINDOWS == 1 )) || return 0
  case "$value" in
    '\\?\'* | '\\.\'* | '//?/'* | '//./'*)
      echo "error: refusing extended-length or device path for $label: $value" >&2
      echo "       write it without the \\\\?\\ or \\\\.\\ prefix" >&2
      exit 2
      ;;
  esac
}

refuse_production_path() {
  local label="$1" value="$2" canonical root
  local -a roots=(
    "$HOME/Library/Application Support/wenlan"
    "$HOME/Library/Application Support/origin"
    "$HOME/Library/LaunchAgents"
    "$HOME/Library/Logs/com.wenlan.desktop"
    "$HOME/Library/Logs/com.origin.desktop"
    "$HOME/.config/wenlan-mcp"
    "$HOME/.config/origin-mcp"
    "$HOME/.wenlan"
    "$HOME/.origin"
  )
  # The Windows half of the same list the app enforces in
  # `production_runtime_roots`. Only appended when the variable is actually set:
  # an empty root would canonicalize to the working directory and refuse the
  # whole checkout.
  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    roots+=("$LOCALAPPDATA/wenlan" "$LOCALAPPDATA/origin")
  fi
  canonical="$(canonicalize_path "$value")"
  for root in "${roots[@]}"; do
    root="$(canonicalize_path "$root")"
    if path_is_within "$canonical" "$root"; then
      echo "error: refusing production path for $label: $value" >&2
      exit 2
    fi
  done
}

if [[ ! "$DEV_PORT" =~ ^[0-9]+$ ]] || (( DEV_PORT < 1 || DEV_PORT > 65535 )); then
  echo "error: invalid WENLAN_DEV_PORT: $DEV_PORT" >&2
  exit 2
fi
if [[ ! "$DEV_UI_PORT" =~ ^[0-9]+$ ]] || (( DEV_UI_PORT < 1 || DEV_UI_PORT > 65535 )); then
  echo "error: invalid WENLAN_DEV_UI_PORT: $DEV_UI_PORT" >&2
  exit 2
fi
if [[ ! "$DEV_REMOTE_PORT_START" =~ ^[0-9]+$ ]] ||
  (( DEV_REMOTE_PORT_START < 1 || DEV_REMOTE_PORT_START > 65532 )); then
  echo "error: invalid WENLAN_DEV_REMOTE_PORT_START: $DEV_REMOTE_PORT_START" >&2
  exit 2
fi
if (( DEV_PORT == 7878 )); then
  echo "error: refusing production daemon port 7878" >&2
  exit 2
fi
if (( DEV_UI_PORT == 1420 )); then
  echo "error: refusing production UI identity on port 1420" >&2
  exit 2
fi
if (( DEV_REMOTE_PORT_START <= 18083 && DEV_REMOTE_PORT_START + 3 >= 18080 )); then
  echo "error: refusing production remote-access port range 18080-18083" >&2
  exit 2
fi
if [[ "$DEV_APP_ID" == "com.wenlan.desktop" || "$DEV_APP_ID" == "com.origin.desktop" ]]; then
  echo "error: refusing production app identifier: $DEV_APP_ID" >&2
  exit 2
fi
reject_verbatim_path "WENLAN_DEV_STATE_DIR" "$STATE_DIR"
reject_verbatim_path "WENLAN_DEV_DATA_DIR" "$DEV_DATA_DIR"
reject_verbatim_path "WENLAN_DEV_TAURI_MCP_SOCKET" "$DEV_TAURI_MCP_SOCKET"
if [[ "$(canonicalize_path "$DEV_TAURI_MCP_SOCKET")" == "$(canonicalize_path "/tmp/tauri-mcp.sock")" ]]; then
  echo "error: refusing production Tauri MCP socket: $DEV_TAURI_MCP_SOCKET" >&2
  exit 2
fi
refuse_production_path "WENLAN_DEV_STATE_DIR" "$STATE_DIR"
refuse_production_path "WENLAN_DEV_DATA_DIR" "$DEV_DATA_DIR"
refuse_production_path "WENLAN_DEV_TAURI_MCP_SOCKET" "$DEV_TAURI_MCP_SOCKET"

print_config() {
  printf 'WENLAN_PORT=%s\n' "$DEV_PORT"
  printf 'WENLAN_DEV_UI_PORT=%s\n' "$DEV_UI_PORT"
  printf 'WENLAN_DEV_REMOTE_PORT_START=%s\n' "$DEV_REMOTE_PORT_START"
  printf 'WENLAN_DEV_APP_ID=%s\n' "$DEV_APP_ID"
  printf 'WENLAN_DEV_TAURI_MCP_SOCKET=%s\n' "$(native_path "$DEV_TAURI_MCP_SOCKET")"
  printf 'WENLAN_DATA_DIR=%s\n' "$(native_path "$DEV_DATA_DIR")"
  printf 'WENLAN_DEV_STATE_DIR=%s\n' "$(native_path "$STATE_DIR")"
}

read_owned_pid() {
  [[ -f "$PID_FILE" && -f "$SERVER_PATH_FILE" && -f "$PORT_FILE" ]] || return 1
  OWNED_PID="$(sed -n '1p' "$PID_FILE")"
  OWNED_SERVER="$(sed -n '1p' "$SERVER_PATH_FILE")"
  OWNED_PORT="$(sed -n '1p' "$PORT_FILE")"
  OWNED_DATA_DIR="$(sed -n '1p' "$DATA_DIR_FILE" 2>/dev/null || true)"
  [[ "$OWNED_PID" =~ ^[0-9]+$ && -n "$OWNED_SERVER" &&
    "$OWNED_PORT" =~ ^[0-9]+$ ]] || return 1
}

listener_pid_for_port() {
  if (( HOST_IS_WINDOWS == 1 )); then
    # netstat is the only listener table Windows offers here. Anchoring the
    # port keeps :17895 from matching :178950 or a foreign address.
    netstat -ano | awk -v port=":$1\$" '
      $1 == "TCP" && $2 ~ port && $4 == "LISTENING" { print $5; exit }
    '
  else
    lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null | sed -n '1p'
  fi
}

has_owned_command_identity() {
  local command
  process_is_alive "$OWNED_PID" || return 1
  command="$(process_image_path "$OWNED_PID")"
  if (( HOST_IS_WINDOWS == 1 )); then
    [[ "$(normalize_program_path "$command")" == "$(normalize_program_path "$OWNED_SERVER")" ]]
  else
    [[ "$command" == "$OWNED_SERVER" || "$command" == "$OWNED_SERVER "* ]]
  fi
}

is_owned_process() {
  has_owned_command_identity &&
    [[ "$(listener_pid_for_port "$OWNED_PORT")" == "$OWNED_PID" ]]
}

release_runtime_lock() {
  if [[ -f "$LOCK_OWNER_FILE" ]] && [[ "$(sed -n '1p' "$LOCK_OWNER_FILE")" == "$$" ]]; then
    rm -f "$LOCK_OWNER_FILE"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

acquire_runtime_lock() {
  local owner
  mkdir -p "$STATE_DIR"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    owner="$(sed -n '1p' "$LOCK_OWNER_FILE" 2>/dev/null || true)"
    if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
      echo "error: another dev runtime command is active (PID $owner)" >&2
      return 1
    fi
    rm -f "$LOCK_OWNER_FILE"
    rmdir "$LOCK_DIR" 2>/dev/null || {
      echo "error: stale dev runtime lock could not be recovered: $LOCK_DIR" >&2
      return 1
    }
    mkdir "$LOCK_DIR"
  fi
  printf '%s\n' "$$" >"$LOCK_OWNER_FILE"
  trap release_runtime_lock EXIT HUP INT TERM
}

stop_runtime() {
  if ! read_owned_pid; then
    echo "No worktree-owned Wenlan dev daemon is recorded."
    return 0
  fi
  if ! process_is_alive "$OWNED_PID"; then
    rm -f "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE" "$DATA_DIR_FILE"
    echo "Removed stale Wenlan dev daemon state."
    return 0
  fi
  if ! is_owned_process; then
    echo "error: refusing to stop PID $OWNED_PID because it is not $OWNED_SERVER" >&2
    return 1
  fi

  terminate_process "$OWNED_PID" "$OWNED_SERVER"
  for _ in $(seq 1 50); do
    if ! process_is_alive "$OWNED_PID"; then
      rm -f "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE" "$DATA_DIR_FILE"
      echo "Stopped worktree-owned Wenlan dev daemon (PID $OWNED_PID)."
      return 0
    fi
    sleep 0.1
  done

  if has_owned_command_identity; then
    force_terminate_process "$OWNED_PID" "$OWNED_SERVER"
  fi
  for _ in $(seq 1 50); do
    if ! process_is_alive "$OWNED_PID"; then
      rm -f "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE" "$DATA_DIR_FILE"
      echo "Force-stopped unresponsive worktree-owned Wenlan dev daemon (PID $OWNED_PID)."
      return 0
    fi
    sleep 0.1
  done
  echo "error: worktree-owned Wenlan dev daemon PID $OWNED_PID did not exit" >&2
  return 1
}

start_runtime() {
  local backend build_dir server pid job_pid listener_pid
  STARTED_RUNTIME=0
  DEV_DATA_DIR="$(canonicalize_path "$DEV_DATA_DIR")"
  backend="$(bash "$SCRIPT_DIR/resolve-backend-dir.sh" "$REPO_ROOT")"
  # The same target-root resolution prepare-sidecars.sh uses. A short
  # CARGO_TARGET_DIR is how a Windows checkout stays under MAX_PATH, and reading
  # the built binary from the wrong root either fails to launch or, worse,
  # launches a stale one.
  build_dir="${CARGO_TARGET_DIR:-$backend/target}/debug"
  server="$build_dir/wenlan-server"
  if (( HOST_IS_WINDOWS == 1 )); then
    # Spelled natively so the recorded identity is one string no matter whether
    # this run inherited WENLAN_DEV_STATE_DIR from `print-config` or derived it.
    server="$(native_path "$DAEMON_STAGE_DIR")/wenlan-server.exe"
  fi

  if read_owned_pid && is_owned_process; then
    if [[ "$OWNED_SERVER" != "$server" || "$OWNED_PORT" != "$DEV_PORT" ||
      "$OWNED_DATA_DIR" != "$DEV_DATA_DIR" ]]; then
      echo "error: recorded dev daemon identity does not match this runtime configuration" >&2
      echo "recorded: server=$OWNED_SERVER port=$OWNED_PORT data=$OWNED_DATA_DIR" >&2
      echo "selected: server=$server port=$DEV_PORT data=$DEV_DATA_DIR" >&2
      return 1
    fi
    print_config
    echo "Wenlan dev daemon is already running (PID $OWNED_PID)."
    return 0
  fi

  mkdir -p "$STATE_DIR" "$DEV_DATA_DIR"
  rm -f "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE" "$DATA_DIR_FILE"

  if [[ -n "$(listener_pid_for_port "$DEV_PORT")" ]]; then
    echo "error: isolated dev port $DEV_PORT is already in use; set WENLAN_DEV_PORT" >&2
    return 1
  fi

  cargo build --manifest-path "$backend/Cargo.toml" -p wenlan-server
  if (( HOST_IS_WINDOWS == 1 )); then
    stage_windows_daemon "$build_dir/wenlan-server.exe" || return 1
  fi
  nohup env WENLAN_PORT="$DEV_PORT" WENLAN_DATA_DIR="$DEV_DATA_DIR" \
    "$server" </dev/null >"$SERVER_LOG" 2>&1 &
  pid=$!
  job_pid=$pid
  if (( HOST_IS_WINDOWS == 1 )); then
    # `$!` is the MSYS pid. Every Windows listener and process table reports the
    # Windows one, so that is the identity worth recording and comparing.
    if ! pid="$(windows_pid_for_job "$job_pid" "$server")"; then
      # Nothing downstream can identify this daemon, so it cannot be left
      # running. The MSYS pid is not the handle to reach for: the resolution
      # above waits up to ten seconds, MSYS recycles pids, and by this branch
      # the job may already be gone, so signalling it can land on an unrelated
      # process. Reap by image identity instead, which needs neither the pid nor
      # the port to have settled.
      if ! reap_staged_daemon "$server"; then
        # Nothing was recorded, so `stop` has no pid to read and cannot clean
        # this up. Name the executable so the port can be freed by hand.
        echo "warning: a daemon may still be starting from $server" >&2
        echo "         nothing recorded it; end that process before retrying" >&2
      fi
      echo "error: could not resolve the Windows pid of the dev daemon" >&2
      tail -n 40 "$SERVER_LOG" >&2 || true
      return 1
    fi
  fi
  printf '%s\n' "$pid" >"$PID_FILE"
  printf '%s\n' "$server" >"$SERVER_PATH_FILE"
  printf '%s\n' "$DEV_PORT" >"$PORT_FILE"
  printf '%s\n' "$DEV_DATA_DIR" >"$DATA_DIR_FILE"

  for _ in $(seq 1 50); do
    if curl --fail --silent --max-time 1 \
      "http://127.0.0.1:$DEV_PORT/api/health" >/dev/null 2>&1; then
      listener_pid="$(listener_pid_for_port "$DEV_PORT")"
      if process_is_alive "$pid" && [[ "$listener_pid" == "$pid" ]]; then
        print_config
        echo "Started worktree-owned Wenlan dev daemon (PID $pid)."
        STARTED_RUNTIME=1
        return 0
      fi
      break
    fi
    if ! process_is_alive "$pid"; then
      break
    fi
    sleep 0.2
  done

  tail -n 40 "$SERVER_LOG" >&2 || true
  stop_runtime || true
  echo "error: Wenlan dev daemon did not become healthy on port $DEV_PORT" >&2
  return 1
}

case "${1:-}" in
  print-config)
    print_config
    ;;
  start)
    acquire_runtime_lock
    start_runtime
    ;;
  start-for-session)
    acquire_runtime_lock
    start_runtime
    if (( STARTED_RUNTIME == 0 )); then
      exit 10
    fi
    ;;
  stop)
    acquire_runtime_lock
    stop_runtime
    ;;
  *)
    echo "usage: $0 {print-config|start|start-for-session|stop}" >&2
    exit 2
    ;;
esac
