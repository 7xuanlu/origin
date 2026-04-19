#!/usr/bin/env node
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { execFileSync, spawn } = require('child_process');
const { resolveReleaseTarget } = require('../scripts/release-target');

const {
  binDir: BIN_DIR,
  label,
  pageUrl,
  runtimeDataDir,
  runtimeEnv,
  runtimePort,
  usesIsolatedDefaults,
} = resolveReleaseTarget();
const MCP_BIN = path.join(BIN_DIR, 'origin-mcp');
const SERVER_BIN = path.join(BIN_DIR, 'origin-server');
const INSTALL_SCRIPT = path.join(__dirname, '..', 'scripts', 'install.js');
const DAEMON_URL = `http://127.0.0.1:${runtimePort}/api/health`;

function checkHealth() {
  return new Promise((resolve) => {
    const req = http.get(DAEMON_URL, { timeout: 2000 }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 300);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForDaemon(maxAttempts = 15, maxDelayMs = 3000) {
  for (let i = 0; i < maxAttempts; i++) {
    const healthy = await checkHealth();
    if (healthy) return true;
    const delay = Math.min(200 * Math.pow(2, i), maxDelayMs);
    await sleep(delay);
  }
  return false;
}

async function main() {
  // Recover partial installs by rerunning the installer if either binary is missing.
  if (!fs.existsSync(MCP_BIN) || !fs.existsSync(SERVER_BIN)) {
    console.error(`[origin-mcp] Required binaries for ${label} missing. Running installer...`);
    try {
      execFileSync(process.execPath, [INSTALL_SCRIPT], { stdio: 'inherit' });
    } catch (err) {
      console.error(`[origin-mcp] Installer failed: ${err.message}`);
      process.exit(1);
    }
  }

  // Check if binary exists after install attempt
  if (!fs.existsSync(MCP_BIN)) {
    console.error('[origin-mcp] origin-mcp binary not found after install. Aborting.');
    console.error(`[origin-mcp] Install manually: ${pageUrl}`);
    process.exit(1);
  }

  // Check daemon health
  const daemonRunning = await checkHealth();

  if (usesIsolatedDefaults) {
    console.error(`[origin-mcp] Using isolated prerelease runtime on port ${runtimePort}.`);
    console.error(`[origin-mcp] Data dir: ${runtimeDataDir}`);
  }

  if (!daemonRunning) {
    if (fs.existsSync(SERVER_BIN)) {
      // Spawn daemon detached so it outlives this process
      const child = spawn(SERVER_BIN, [], {
        detached: true,
        env: runtimeEnv,
        stdio: 'ignore',
      });
      child.unref();

      // Wait for daemon to become healthy with exponential backoff
      const ready = await waitForDaemon(10);
      if (!ready) {
        // Not fatal — origin-mcp itself may handle a missing daemon gracefully
        console.error('[origin-mcp] Warning: daemon did not become healthy in time. Proceeding anyway.');
      }
    } else {
      console.error('[origin-mcp] Warning: origin-server binary not found, daemon not started.');
    }
  }

  // Spawn the real MCP binary with signal forwarding
  const child = spawn(
    MCP_BIN,
    ['--origin-url', `http://127.0.0.1:${runtimePort}`, ...process.argv.slice(2)],
    { env: runtimeEnv, stdio: 'inherit' },
  );
  child.on('exit', (code, signal) => {
    process.exit(signal ? 128 : (code ?? 1));
  });
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.on(sig, () => child.kill(sig));
  }
}

main().catch((err) => {
  console.error('[origin-mcp]', err.message);
  process.exit(1);
});
