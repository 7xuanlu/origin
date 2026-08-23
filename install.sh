#!/usr/bin/env bash
set -euo pipefail

# Wenlan installer — downloads Wenlan runtime binaries to ~/.wenlan/bin/
# Usage:      curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | bash
# Prerelease: curl -fsSL ... | WENLAN_RELEASE_TAG=v0.2.0-alpha.1 bash
#
# Supported platforms: macOS (arm64), Linux (aarch64, x86_64).
# Windows users: download wenlan-windows-x64.zip from the GitHub release page.

REPO="7xuanlu/wenlan"
REQUESTED_TAG="${WENLAN_RELEASE_TAG:-${WENLAN_TAG:-${ORIGIN_RELEASE_TAG:-${ORIGIN_TAG:-}}}}"

if [[ -n "${REQUESTED_TAG}" ]]; then
  SAFE_TAG="$(printf '%s' "${REQUESTED_TAG}" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')"
  BIN_DIR="${HOME}/.wenlan/releases/${SAFE_TAG}"
  API_URL="https://api.github.com/repos/${REPO}/releases/tags/${REQUESTED_TAG}"
  RELEASE_PAGE="https://github.com/${REPO}/releases/tag/${REQUESTED_TAG}"
else
  BIN_DIR="${HOME}/.wenlan/bin"
  API_URL="https://api.github.com/repos/${REPO}/releases/latest"
  RELEASE_PAGE="https://github.com/${REPO}/releases"
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m==> \033[0m%s\n' "$*"; }
ok()    { printf '\033[1;32m  ✓ \033[0m%s\n' "$*"; }
warn()  { printf '\033[1;33mwarn: \033[0m%s\n' "$*" >&2; }
die()   { printf '\033[1;31merror: \033[0m%s\n' "$*" >&2; exit 1; }

derive_isolated_port() {
  local tag="$1"
  local hash=0
  local i char ord

  for (( i=0; i<${#tag}; i++ )); do
    char="${tag:i:1}"
    ord=$(printf '%d' "'${char}")
    hash=$(( ((hash * 33) + ord) & 0xFFFFFFFF ))
  done

  printf '%s' "$((8800 + (hash % 1000)))"
}

# ── Platform detection ──────────────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
  Darwin-arm64)         ASSET="wenlan-darwin-arm64.tar.gz" ;;
  Linux-aarch64|Linux-arm64)
                        ASSET="wenlan-linux-arm64.tar.gz" ;;
  Linux-x86_64)         ASSET="wenlan-linux-x64.tar.gz" ;;
  *)
    die "Unsupported platform: ${OS}-${ARCH}
Supported: Darwin-arm64, Linux-aarch64, Linux-x86_64.
For Windows, download wenlan-windows-x64.zip from the GitHub release page:
  ${RELEASE_PAGE}"
    ;;
esac

info "Detected platform: ${OS}-${ARCH} (${ASSET})"

# ── Fetch latest release tag ──────────────────────────────────────────────────

info "Fetching latest release from GitHub..."

TAG="$(curl -fsSL "${API_URL}" \
  | grep '"tag_name"' \
  | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"

[[ -n "${TAG}" ]] || die "Could not determine release tag. Is the GitHub API reachable?"

if [[ -n "${REQUESTED_TAG}" ]]; then
  ok "Requested release: ${TAG}"
else
  ok "Latest release: ${TAG}"
fi

# ── Download & extract ───────────────────────────────────────────────────────

mkdir -p "${BIN_DIR}"

RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG}"
DOWNLOAD_URL="${RELEASE_BASE}/${ASSET}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

info "Downloading ${ASSET}..."
if ! curl -fSL --progress-bar -o "${TMP_DIR}/${ASSET}" "${DOWNLOAD_URL}"; then
  die "Failed to download ${ASSET} from ${DOWNLOAD_URL}"
fi
ok "Downloaded ${ASSET}"

info "Extracting..."
if ! tar -xzf "${TMP_DIR}/${ASSET}" -C "${TMP_DIR}"; then
  die "Failed to extract ${ASSET}"
fi
ok "Extracted"

# ── Install binaries ─────────────────────────────────────────────────────────

for bin in wenlan wenlan-server wenlan-mcp; do
  if [[ ! -f "${TMP_DIR}/${bin}" ]]; then
    die "Archive ${ASSET} missing expected binary: ${bin}"
  fi
  install -m 0755 "${TMP_DIR}/${bin}" "${BIN_DIR}/${bin}"
done

# Clear macOS quarantine attribute (unsigned binaries downloaded from the internet)
if [[ "${OS}" == "Darwin" ]]; then
  xattr -cr "${BIN_DIR}/wenlan"        2>/dev/null || true
  xattr -cr "${BIN_DIR}/wenlan-server" 2>/dev/null || true
  xattr -cr "${BIN_DIR}/wenlan-mcp"    2>/dev/null || true
fi

ok "Installed wenlan, wenlan-server, wenlan-mcp to ${BIN_DIR}"

# ── PATH setup ────────────────────────────────────────────────────────────────

add_to_path() {
  local rc_file="$1"
  local line='export PATH="${HOME}/.wenlan/bin:${PATH}"'

  if [[ -f "${rc_file}" ]] && grep -qF '.wenlan/bin' "${rc_file}"; then
    ok "${rc_file} already has ~/.wenlan/bin in PATH — skipping"
    return
  fi

  printf '\n# Added by Wenlan installer\n%s\n' "${line}" >> "${rc_file}"
  ok "Added ~/.wenlan/bin to PATH in ${rc_file}"
}

# Detect current shell and preferred rc file
CURRENT_SHELL="$(basename "${SHELL:-/bin/zsh}")"
case "${CURRENT_SHELL}" in
  zsh)  RC_FILE="${HOME}/.zshrc" ;;
  bash) RC_FILE="${HOME}/.bashrc" ;;
  *)
    warn "Unknown shell '${CURRENT_SHELL}'. Defaulting to ~/.zshrc"
    RC_FILE="${HOME}/.zshrc"
    ;;
esac

if [[ -z "${REQUESTED_TAG}" ]]; then
  add_to_path "${RC_FILE}"
else
  warn "Exact-tag install requested (${TAG}); not modifying ${RC_FILE}"
fi

# Also export for the rest of this script session
export PATH="${BIN_DIR}:${PATH}"

if [[ -n "${REQUESTED_TAG}" ]]; then
  EXACT_RUNTIME_PORT="$(derive_isolated_port "${REQUESTED_TAG}")"
  if [[ "${OS}" == "Darwin" ]]; then
    EXACT_RUNTIME_DATA_DIR="${HOME}/Library/Application Support/wenlan/releases/${SAFE_TAG}"
  else
    EXACT_RUNTIME_DATA_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/wenlan/releases/${SAFE_TAG}"
  fi
fi

# ── Next steps ────────────────────────────────────────────────────────────────

printf '\n'
printf '\033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\033[1;32m  Wenlan installed successfully!\033[0m\n'
printf '\033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\n'
printf 'Next steps:\n\n'
if [[ -z "${REQUESTED_TAG}" ]]; then
  printf '  1. Reload your shell (or open a new terminal):\n'
  printf '\n'
  printf '       source %s\n' "${RC_FILE}"
  printf '\n'
  printf '  2. Set up Wenlan:\n'
  printf '\n'
  printf '       wenlan setup --basic\n'
  printf '\n'
  printf '  3. Register Wenlan as a background service (launchd):\n'
  printf '\n'
  printf '       wenlan background on\n'
  printf '\n'
  printf '  4. Verify the daemon and memory setup:\n'
  printf '\n'
  printf '       wenlan status\n'
  printf '\n'
  printf '  5. Add the MCP server to Claude Desktop or Cursor:\n'
  printf '\n'
  printf '       {\n'
  printf '         "mcpServers": {\n'
  printf '           "wenlan": {\n'
  printf '             "command": "%s/wenlan-mcp"\n' "${BIN_DIR}"
  printf '           }\n'
  printf '         }\n'
  printf '       }\n'
  printf '\n'
else
  printf '  1. Use this exact tagged release in the current shell session:\n'
  printf '\n'
  printf '       export PATH="%s:$PATH"\n' "${BIN_DIR}"
  printf '\n'
  printf '     Installed under: %s\n' "${BIN_DIR}"
  printf '\n'
  printf '  2. Start this exact tagged daemon in an isolated runtime:\n'
  printf '\n'
  printf '       wenlan-server --port %s --data-dir "%s"\n' "${EXACT_RUNTIME_PORT}" "${EXACT_RUNTIME_DATA_DIR}"
  printf '\n'
  printf '  3. Add this exact-release MCP server to Claude Desktop or Cursor:\n'
  printf '\n'
  printf '       {\n'
  printf '         "mcpServers": {\n'
  printf '           "wenlan-exact": {\n'
  printf '             "command": "%s/wenlan-mcp",\n' "${BIN_DIR}"
  printf '             "args": ["--origin-url", "http://127.0.0.1:%s"]\n' "${EXACT_RUNTIME_PORT}"
  printf '           }\n'
  printf '         }\n'
  printf '       }\n'
  printf '\n'
  printf '     Data dir: %s\n' "${EXACT_RUNTIME_DATA_DIR}"
  printf '\n'
  printf '  4. Do not run wenlan background on for exact tagged installs.\n'
  printf '\n'
  printf '     That replaces the stable com.wenlan.server LaunchAgent.\n'
  printf '\n'
fi
printf '\033[1;33mNote:\033[0m Wenlan can store and retrieve memories without a local model or API key.\n'
printf '      Distill cycles are opt-in with `wenlan models install`.\n'
printf '      Anthropic can be configured with `wenlan keys set anthropic`.\n'
if [[ -n "${REQUESTED_TAG}" ]]; then
  printf '      Manual release page for this install: %s\n' "${RELEASE_PAGE}"
fi
printf '\n'
