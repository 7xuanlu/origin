#!/usr/bin/env bash
set -euo pipefail

# Origin installer — downloads origin-server and origin-mcp to ~/.origin/bin/
# Usage: curl -fsSL https://raw.githubusercontent.com/7xuanlu/origin/main/install.sh | bash
# Prerelease: curl -fsSL ... | ORIGIN_RELEASE_TAG=v0.2.0-alpha.1 bash

REPO="7xuanlu/origin"
REQUESTED_TAG="${ORIGIN_RELEASE_TAG:-${ORIGIN_TAG:-}}"

if [[ -n "${REQUESTED_TAG}" ]]; then
  SAFE_TAG="$(printf '%s' "${REQUESTED_TAG}" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')"
  BIN_DIR="${HOME}/.origin/releases/${SAFE_TAG}"
  API_URL="https://api.github.com/repos/${REPO}/releases/tags/${REQUESTED_TAG}"
  RELEASE_PAGE="https://github.com/${REPO}/releases/tag/${REQUESTED_TAG}"
else
  BIN_DIR="${HOME}/.origin/bin"
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

# ── OS check ─────────────────────────────────────────────────────────────────

OS="$(uname -s)"
[[ "${OS}" == "Darwin" ]] || die "Origin requires macOS. Detected OS: ${OS}"

# ── Architecture ─────────────────────────────────────────────────────────────

ARCH="$(uname -m)"
case "${ARCH}" in
  arm64)   TARGET="aarch64-apple-darwin" ;;
  x86_64)  die "Intel Macs are not yet supported. Origin requires Apple Silicon (M1+)." ;;
  *)       die "Unsupported architecture: ${ARCH}" ;;
esac

info "Detected platform: ${TARGET}"

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

# ── Download ──────────────────────────────────────────────────────────────────

mkdir -p "${BIN_DIR}"

RELEASE_BASE="https://github.com/${REPO}/releases/download/${TAG}"

download_binary() {
  local name="$1"
  local url="${RELEASE_BASE}/${name}-${TARGET}"
  local dest="${BIN_DIR}/${name}"

  info "Downloading ${name}..."
  if ! curl -fSL --progress-bar -o "${dest}" "${url}"; then
    die "Failed to download ${name} from ${url}"
  fi
  ok "Downloaded ${name}"
}

download_binary "origin-server"
download_binary "origin-mcp"

# ── Permissions ───────────────────────────────────────────────────────────────

chmod +x "${BIN_DIR}/origin-server" "${BIN_DIR}/origin-mcp"

# Clear macOS quarantine attribute (unsigned binaries downloaded from the internet)
xattr -cr "${BIN_DIR}/origin-server" 2>/dev/null || true
xattr -cr "${BIN_DIR}/origin-mcp"    2>/dev/null || true

ok "Permissions set"

# ── PATH setup ────────────────────────────────────────────────────────────────

add_to_path() {
  local rc_file="$1"
  local line='export PATH="${HOME}/.origin/bin:${PATH}"'

  if [[ -f "${rc_file}" ]] && grep -qF '.origin/bin' "${rc_file}"; then
    ok "${rc_file} already has ~/.origin/bin in PATH — skipping"
    return
  fi

  printf '\n# Added by Origin installer\n%s\n' "${line}" >> "${rc_file}"
  ok "Added ~/.origin/bin to PATH in ${rc_file}"
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
  EXACT_RUNTIME_DATA_DIR="${HOME}/Library/Application Support/origin/releases/${SAFE_TAG}"
fi

# ── Next steps ────────────────────────────────────────────────────────────────

printf '\n'
printf '\033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\033[1;32m  Origin installed successfully!\033[0m\n'
printf '\033[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\n'
printf 'Next steps:\n\n'
if [[ -z "${REQUESTED_TAG}" ]]; then
  printf '  1. Reload your shell (or open a new terminal):\n'
  printf '\n'
  printf '       source %s\n' "${RC_FILE}"
  printf '\n'
  printf '  2. Register Origin as a background service (launchd):\n'
  printf '\n'
  printf '       origin-server install\n'
  printf '\n'
  printf '  3. Verify the daemon is running:\n'
  printf '\n'
  printf '       origin-server status\n'
  printf '\n'
  printf '  4. Add the MCP server to Claude Desktop or Cursor:\n'
  printf '\n'
  printf '       {\n'
  printf '         "mcpServers": {\n'
  printf '           "origin": {\n'
  printf '             "command": "%s/origin-mcp"\n' "${BIN_DIR}"
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
  printf '       origin-server --port %s --data-dir "%s"\n' "${EXACT_RUNTIME_PORT}" "${EXACT_RUNTIME_DATA_DIR}"
  printf '\n'
  printf '  3. Add this exact-release MCP server to Claude Desktop or Cursor:\n'
  printf '\n'
  printf '       {\n'
  printf '         "mcpServers": {\n'
  printf '           "origin-exact": {\n'
  printf '             "command": "%s/origin-mcp",\n' "${BIN_DIR}"
  printf '             "args": ["--origin-url", "http://127.0.0.1:%s"]\n' "${EXACT_RUNTIME_PORT}"
  printf '           }\n'
  printf '         }\n'
  printf '       }\n'
  printf '\n'
  printf '     Data dir: %s\n' "${EXACT_RUNTIME_DATA_DIR}"
  printf '\n'
  printf '  4. Do not run origin-server install for exact tagged installs.\n'
  printf '\n'
  printf '     That replaces the stable com.origin.server LaunchAgent.\n'
  printf '\n'
fi
printf '\033[1;33mNote:\033[0m Origin can store and retrieve memories without downloading an on-device model.\n'
printf '      Local synthesis models are opt-in from the desktop app settings.\n'
if [[ -n "${REQUESTED_TAG}" ]]; then
  printf '      Manual release page for this install: %s\n' "${RELEASE_PAGE}"
fi
printf '\n'
