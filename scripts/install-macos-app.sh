#!/usr/bin/env bash
set -euo pipefail

RELEASE_JSON_URL=${WENLAN_APP_RELEASE_JSON_URL:-https://api.github.com/repos/7xuanlu/wenlan/releases/latest}
ASSET_NAME=${WENLAN_APP_ASSET_NAME:-Wenlan_aarch64.app.tar.gz}

die() {
  echo "Wenlan app install failed: $*" >&2
  exit 1
}

if [[ ${WENLAN_APP_SKIP_PLATFORM_CHECK:-0} != 1 ]]; then
  [[ $(uname -s) == Darwin ]] || die "the desktop app installer currently supports macOS only"

  machine=$(uname -m)
  if [[ $machine == x86_64 ]] && [[ $(sysctl -in sysctl.proc_translated 2>/dev/null || true) == 1 ]]; then
    machine=arm64
  fi
  [[ $machine == arm64 ]] || die "the prebuilt desktop app currently supports Apple Silicon only"
fi

if [[ -n ${WENLAN_APP_INSTALL_DIR:-} ]]; then
  install_dir=$WENLAN_APP_INSTALL_DIR
elif [[ -w /Applications ]]; then
  install_dir=/Applications
else
  install_dir="$HOME/Applications"
fi

mkdir -p "$install_dir"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/wenlan-app-install.XXXXXX")
incoming="$install_dir/.Wenlan.app.installing.$$"
backup="$install_dir/.Wenlan.app.backup.$$"
target="$install_dir/Wenlan.app"

cleanup() {
  status=$?
  trap - EXIT INT TERM

  if [[ -e $backup || -L $backup ]]; then
    if [[ ! -e $target && ! -L $target ]]; then
      if ! mv "$backup" "$target"; then
        echo "Wenlan app install warning: could not restore the previous app from $backup" >&2
      fi
    else
      rm -rf "$backup"
    fi
  fi
  rm -rf "$tmp_dir" "$incoming"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

release_json="$tmp_dir/release.json"
archive="$tmp_dir/$ASSET_NAME"
extract_dir="$tmp_dir/extracted"

# The release lookup is this installer's one GitHub API call (the asset's
# SHA-256 digest comes from it). Anonymous callers get 60 calls per hour per IP
# address, so a token is sent when the user has one, and a rate-limited answer
# is named as such instead of surfacing as a bare curl error.
fetch_release_json() {
  local curl_args=(-sSL -o "$release_json" -w '%{http_code}')
  local token=${GITHUB_TOKEN:-${GH_TOKEN:-}}
  if [[ -n $token && $RELEASE_JSON_URL == http* ]]; then
    curl_args+=(-H "Authorization: Bearer $token")
  fi

  local http_code
  http_code=$(curl "${curl_args[@]}" "$RELEASE_JSON_URL") || die "could not fetch release metadata from $RELEASE_JSON_URL"
  case $http_code in
    000 | 2??) ;;
    403 | 429)
      die "GitHub's API rate limit blocked the release lookup (HTTP $http_code). Wait an hour, or set GITHUB_TOKEN to a GitHub token and run the command again."
      ;;
    *) die "release lookup at $RELEASE_JSON_URL failed (HTTP $http_code)" ;;
  esac
}

echo "Finding the latest Wenlan app release..."
fetch_release_json

asset_count=$(plutil -extract assets raw -o - "$release_json" 2>/dev/null) || die "release metadata has no assets"
asset_url=
expected_digest=

for ((index = 0; index < asset_count; index++)); do
  name=$(plutil -extract "assets.$index.name" raw -o - "$release_json")
  if [[ $name == "$ASSET_NAME" ]]; then
    asset_url=$(plutil -extract "assets.$index.browser_download_url" raw -o - "$release_json")
    expected_digest=$(plutil -extract "assets.$index.digest" raw -o - "$release_json")
    break
  fi
done

[[ -n $asset_url ]] || die "release asset $ASSET_NAME was not found"
[[ $expected_digest == sha256:* ]] || die "release asset has no SHA-256 digest"
expected_digest=${expected_digest#sha256:}

echo "Downloading $ASSET_NAME..."
curl -fL "$asset_url" -o "$archive"

actual_digest=$(shasum -a 256 "$archive" | awk '{print $1}')
[[ $actual_digest == "$expected_digest" ]] || die "download checksum did not match the GitHub release"

mkdir -p "$extract_dir"
tar -xzf "$archive" -C "$extract_dir"
source_app="$extract_dir/Wenlan.app"
[[ -d $source_app ]] || die "archive did not contain Wenlan.app"

# The command itself is the user's explicit consent to install this unnotarized preview.
xattr -dr com.apple.quarantine "$source_app" 2>/dev/null || true
ditto "$source_app" "$incoming"
xattr -dr com.apple.quarantine "$incoming" 2>/dev/null || true

# A Wenlan that is still running would only come to the front when the new one
# is opened (its single-instance socket sends the newcomer straight back to
# exit), so ask it to quit before its bundle is replaced. Apps from 0.18.0 on
# hand over by themselves; this covers upgrades from 0.17.0 and older, wherever
# the running bundle lives. The Apple event is bounded, so a stuck or
# unanswered permission prompt cannot hold the install, and a Wenlan that will
# not quit is an error rather than a success message.
running_wenlan() {
  pgrep -f '\.app/Contents/MacOS/wenlan-app$' >/dev/null 2>&1
}
if running_wenlan; then
  echo "Asking the running Wenlan to quit so the new version can start..."
  osascript -e 'tell application id "com.wenlan.desktop" to quit' >/dev/null 2>&1 &
  quit_request=$!
  for _ in $(seq 1 100); do
    running_wenlan || break
    sleep 0.1
  done
  kill "$quit_request" 2>/dev/null || true
  wait "$quit_request" 2>/dev/null || true
  if running_wenlan; then
    die "the running Wenlan did not quit within 10 s. Quit it from its menu bar icon, then run this command again."
  fi
fi

if [[ -e $target ]]; then
  mv "$target" "$backup"
fi

if ! mv "$incoming" "$target"; then
  if [[ -e $backup ]]; then
    mv "$backup" "$target"
  fi
  die "could not place Wenlan.app in $install_dir"
fi

rm -rf "$backup"

echo "Installed Wenlan at $target"
if [[ ${WENLAN_APP_NO_LAUNCH:-0} != 1 ]]; then
  open "$target"
  echo "Opened Wenlan. Follow the in-app setup to connect your sources and AI tools."
fi
