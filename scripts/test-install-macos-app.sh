#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INSTALLER="$ROOT_DIR/scripts/install-macos-app.sh"
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/wenlan-app-installer-test.XXXXXX")
STUB_SERVER=
cleanup() {
  if [[ -n $STUB_SERVER ]]; then
    kill "$STUB_SERVER" 2>/dev/null || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

make_fixture() {
  local fixture_dir=$1
  local digest=$2

  mkdir -p "$fixture_dir/archive/Wenlan.app/Contents/MacOS"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$fixture_dir/archive/Wenlan.app/Contents/MacOS/wenlan-app"
  chmod +x "$fixture_dir/archive/Wenlan.app/Contents/MacOS/wenlan-app"
  xattr -w com.apple.quarantine '0081;test;Codex;' "$fixture_dir/archive/Wenlan.app"
  xattr -w com.wenlan.test-marker 'preserve-me' "$fixture_dir/archive/Wenlan.app"
  tar -czf "$fixture_dir/Wenlan_aarch64.app.tar.gz" -C "$fixture_dir/archive" Wenlan.app

  local actual_digest
  actual_digest=$(shasum -a 256 "$fixture_dir/Wenlan_aarch64.app.tar.gz" | awk '{print $1}')
  if [[ "$digest" == "actual" ]]; then
    digest=$actual_digest
  fi

  cat > "$fixture_dir/release.json" <<JSON
{
  "tag_name": "v-test",
  "assets": [
    {
      "name": "Wenlan_aarch64.app.tar.gz",
      "browser_download_url": "file://$fixture_dir/Wenlan_aarch64.app.tar.gz",
      "digest": "sha256:$digest"
    }
  ]
}
JSON
}

test_installs_verified_app_without_quarantine() {
  local fixture_dir="$TMP_DIR/success"
  local install_dir="$fixture_dir/Applications"
  mkdir -p "$fixture_dir"
  make_fixture "$fixture_dir" actual

  WENLAN_APP_RELEASE_JSON_URL="file://$fixture_dir/release.json" \
    WENLAN_APP_INSTALL_DIR="$install_dir" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER"

  test -x "$install_dir/Wenlan.app/Contents/MacOS/wenlan-app"
  if xattr -p com.apple.quarantine "$install_dir/Wenlan.app" >/dev/null 2>&1; then
    echo "installed app still has com.apple.quarantine" >&2
    return 1
  fi
  if [[ $(xattr -p com.wenlan.test-marker "$install_dir/Wenlan.app") != preserve-me ]]; then
    echo "installer removed a non-quarantine extended attribute" >&2
    return 1
  fi
}

test_rejects_bad_digest_before_replacing_existing_app() {
  local fixture_dir="$TMP_DIR/bad-digest"
  local install_dir="$fixture_dir/Applications"
  mkdir -p "$fixture_dir" "$install_dir/Wenlan.app"
  printf 'keep me\n' > "$install_dir/Wenlan.app/existing-marker"
  make_fixture "$fixture_dir" "0000000000000000000000000000000000000000000000000000000000000000"

  if WENLAN_APP_RELEASE_JSON_URL="file://$fixture_dir/release.json" \
    WENLAN_APP_INSTALL_DIR="$install_dir" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER"; then
    echo "installer accepted a mismatched digest" >&2
    return 1
  fi

  test -f "$install_dir/Wenlan.app/existing-marker"
}

test_restores_existing_app_when_interrupted_after_backup() {
  local fixture_dir="$TMP_DIR/interrupted"
  local install_dir="$fixture_dir/Applications"
  local fake_bin="$fixture_dir/fake-bin"
  mkdir -p "$fixture_dir" "$install_dir/Wenlan.app" "$fake_bin"
  printf 'keep me\n' > "$install_dir/Wenlan.app/existing-marker"
  make_fixture "$fixture_dir" actual

  cat > "$fake_bin/mv" <<'SH'
#!/usr/bin/env bash
/bin/mv "$@"
case ${2:-} in
  *.Wenlan.app.backup.*) kill -TERM "$PPID" ;;
esac
SH
  chmod +x "$fake_bin/mv"

  if PATH="$fake_bin:$PATH" \
    WENLAN_APP_RELEASE_JSON_URL="file://$fixture_dir/release.json" \
    WENLAN_APP_INSTALL_DIR="$install_dir" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER"; then
    echo "interrupted installer unexpectedly succeeded" >&2
    return 1
  fi

  test -f "$install_dir/Wenlan.app/existing-marker"
  if compgen -G "$install_dir/.Wenlan.app.backup.*" >/dev/null; then
    echo "interrupted installer left a hidden backup behind" >&2
    return 1
  fi
}


# The quit path runs against a stubbed `pgrep` and `osascript`: the stub `pgrep`
# reports a running Wenlan until the stub `osascript` has been asked to quit it.
make_quit_stubs() {
  local fake_bin=$1
  local quit_marker=$2
  local quits=$3

  mkdir -p "$fake_bin"
  cat > "$fake_bin/pgrep" <<SH
#!/usr/bin/env bash
[[ -e "$quit_marker" ]] && exit 1
exit 0
SH
  cat > "$fake_bin/osascript" <<SH
#!/usr/bin/env bash
printf '%s\n' "\$*" > "$quit_marker.request"
if [[ "$quits" == yes ]]; then
  sleep 0.3
  : > "$quit_marker"
fi
sleep 30
SH
  chmod +x "$fake_bin/pgrep" "$fake_bin/osascript"
}

test_quits_running_app_before_replacing_it() {
  local fixture_dir="$TMP_DIR/quits"
  local install_dir="$fixture_dir/Applications"
  local fake_bin="$fixture_dir/fake-bin"
  local quit_marker="$fixture_dir/quit"
  mkdir -p "$fixture_dir" "$install_dir/Wenlan.app"
  printf 'keep me\n' > "$install_dir/Wenlan.app/existing-marker"
  make_fixture "$fixture_dir" actual
  make_quit_stubs "$fake_bin" "$quit_marker" yes

  PATH="$fake_bin:$PATH" \
    WENLAN_APP_RELEASE_JSON_URL="file://$fixture_dir/release.json" \
    WENLAN_APP_INSTALL_DIR="$install_dir" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER"

  if ! grep -q 'tell application id "com.wenlan.desktop" to quit' "$quit_marker.request"; then
    echo "installer did not ask the running Wenlan to quit" >&2
    return 1
  fi
  test -x "$install_dir/Wenlan.app/Contents/MacOS/wenlan-app"
  if [[ -f "$install_dir/Wenlan.app/existing-marker" ]]; then
    echo "installer kept the old app after the running Wenlan quit" >&2
    return 1
  fi
}

test_fails_when_running_app_does_not_quit() {
  local fixture_dir="$TMP_DIR/never-quits"
  local install_dir="$fixture_dir/Applications"
  local fake_bin="$fixture_dir/fake-bin"
  local quit_marker="$fixture_dir/quit"
  mkdir -p "$fixture_dir" "$install_dir/Wenlan.app"
  printf 'keep me\n' > "$install_dir/Wenlan.app/existing-marker"
  make_fixture "$fixture_dir" actual
  make_quit_stubs "$fake_bin" "$quit_marker" no

  if PATH="$fake_bin:$PATH" \
    WENLAN_APP_RELEASE_JSON_URL="file://$fixture_dir/release.json" \
    WENLAN_APP_INSTALL_DIR="$install_dir" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER" > "$fixture_dir/installer.log" 2>&1; then
    echo "installer replaced an app that never quit" >&2
    return 1
  fi

  if ! grep -q 'did not quit within 10 s' "$fixture_dir/installer.log"; then
    echo "installer failed without naming the app that would not quit:" >&2
    cat "$fixture_dir/installer.log" >&2
    return 1
  fi
  test -f "$install_dir/Wenlan.app/existing-marker"
}

test_names_rate_limiting_without_sending_the_token_elsewhere() {
  local fixture_dir="$TMP_DIR/rate-limit"
  local install_dir="$fixture_dir/Applications"
  mkdir -p "$fixture_dir"

  # A loopback server that answers 403 like a rate limit and records the
  # Authorization header it was sent. It is not GitHub's API, so the token
  # must not reach it.
  cat > "$fixture_dir/server.py" <<'PY'
import http.server
import pathlib
import sys

auth_log = pathlib.Path(sys.argv[1])
port_file = pathlib.Path(sys.argv[2])


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        auth_log.write_text(self.headers.get("Authorization", "") + "\n")
        body = b'{"message":"API rate limit exceeded"}'
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
port_file.write_text(str(server.server_address[1]))
server.serve_forever()
PY
  python3 "$fixture_dir/server.py" "$fixture_dir/auth.log" "$fixture_dir/port" &
  STUB_SERVER=$!
  disown "$STUB_SERVER"
  for _ in $(seq 1 50); do
    [[ -s "$fixture_dir/port" ]] && break
    sleep 0.1
  done
  if [[ ! -s "$fixture_dir/port" ]]; then
    echo "rate-limit stub server did not start" >&2
    return 1
  fi

  local status=0
  GITHUB_TOKEN=test-token \
    WENLAN_APP_RELEASE_JSON_URL="http://127.0.0.1:$(cat "$fixture_dir/port")/releases/latest" \
    WENLAN_APP_INSTALL_DIR="$install_dir" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER" > "$fixture_dir/installer.log" 2>&1 || status=$?
  kill "$STUB_SERVER" 2>/dev/null || true
  STUB_SERVER=

  if [[ $status -eq 0 ]]; then
    echo "installer succeeded against a rate-limited release lookup" >&2
    return 1
  fi
  if ! grep -q "GitHub's API rate limit" "$fixture_dir/installer.log"; then
    echo "installer did not name rate limiting:" >&2
    cat "$fixture_dir/installer.log" >&2
    return 1
  fi
  if [[ -s "$fixture_dir/auth.log" && $(cat "$fixture_dir/auth.log") != "" ]]; then
    echo "installer sent the token to a host that is not api.github.com: $(cat "$fixture_dir/auth.log")" >&2
    return 1
  fi
}

test_sends_the_token_to_the_github_api() {
  local fixture_dir="$TMP_DIR/token"
  local fake_bin="$fixture_dir/fake-bin"
  mkdir -p "$fake_bin"

  # A stand-in curl that records its arguments and answers like an expired
  # token, so nothing reaches the network.
  cat > "$fake_bin/curl" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$CURL_RECORD"
printf '401'
SH
  chmod +x "$fake_bin/curl"

  local status=0
  PATH="$fake_bin:$PATH" \
    CURL_RECORD="$fixture_dir/curl-args" \
    GH_TOKEN=test-token \
    WENLAN_APP_RELEASE_JSON_URL="https://api.github.com/repos/7xuanlu/wenlan/releases/latest" \
    WENLAN_APP_INSTALL_DIR="$fixture_dir/Applications" \
    WENLAN_APP_NO_LAUNCH=1 \
    WENLAN_APP_SKIP_PLATFORM_CHECK=1 \
    bash "$INSTALLER" > "$fixture_dir/installer.log" 2>&1 || status=$?

  if [[ $status -eq 0 ]]; then
    echo "installer succeeded against a rejected token" >&2
    return 1
  fi
  if ! grep -q "invalid or expired" "$fixture_dir/installer.log"; then
    echo "installer did not name the rejected token:" >&2
    cat "$fixture_dir/installer.log" >&2
    return 1
  fi
  if ! grep -qx "Authorization: Bearer test-token" "$fixture_dir/curl-args"; then
    echo "installer did not send GH_TOKEN as a bearer token to api.github.com:" >&2
    cat "$fixture_dir/curl-args" >&2
    return 1
  fi
}

test_installs_verified_app_without_quarantine
test_names_rate_limiting_without_sending_the_token_elsewhere
test_sends_the_token_to_the_github_api
test_rejects_bad_digest_before_replacing_existing_app
test_restores_existing_app_when_interrupted_after_backup
test_quits_running_app_before_replacing_it
test_fails_when_running_app_does_not_quit
echo "install-macos-app tests passed"
