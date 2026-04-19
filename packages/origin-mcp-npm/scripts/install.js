#!/usr/bin/env node
'use strict';

const https = require('https');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { resolveReleaseTarget } = require('./release-target');

function platformTarget() {
  if (process.platform !== 'darwin') {
    throw new Error(`Unsupported platform: ${process.platform}. Origin currently supports macOS only.`);
  }
  const arch = process.arch;
  if (arch === 'arm64') return 'aarch64-apple-darwin';
  throw new Error(`Unsupported architecture: ${arch}. Origin requires Apple Silicon (M1+).`);
}

function httpsGet(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: { 'User-Agent': 'origin-mcp-npm-installer/0.1.0' },
    }, (res) => {
      // Follow redirects (GitHub returns 302 for asset downloads)
      if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307 || res.statusCode === 308) {
        const location = res.headers['location'];
        if (!location) {
          reject(new Error(`Redirect with no Location header (status ${res.statusCode})`));
          return;
        }
        res.resume(); // drain
        resolve(httpsGet(location));
        return;
      }
      if (res.statusCode < 200 || res.statusCode >= 300) {
        res.resume();
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      resolve(res);
    });
    req.on('error', reject);
  });
}

function downloadToFile(url, destPath) {
  return new Promise(async (resolve, reject) => {
    try {
      const res = await httpsGet(url);
      const tmpPath = destPath + '.tmp';
      const out = fs.createWriteStream(tmpPath);
      res.pipe(out);
      out.on('finish', () => {
        out.close(() => {
          fs.renameSync(tmpPath, destPath);
          resolve();
        });
      });
      out.on('error', (err) => {
        fs.unlink(tmpPath, () => {});
        reject(err);
      });
      res.on('error', (err) => {
        fs.unlink(tmpPath, () => {});
        reject(err);
      });
    } catch (err) {
      reject(err);
    }
  });
}

function getJson(url) {
  return new Promise(async (resolve, reject) => {
    try {
      const res = await httpsGet(url);
      let data = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`Failed to parse JSON: ${e.message}`));
        }
      });
      res.on('error', reject);
    } catch (err) {
      reject(err);
    }
  });
}

async function main() {
  const releaseTarget = resolveReleaseTarget();
  const { apiUrl, binDir, label, pageUrl } = releaseTarget;

  let target;
  try {
    target = platformTarget();
  } catch (err) {
    console.error(`[origin-mcp] ${err.message}`);
    console.error(`[origin-mcp] Skipping binary download. Install manually: ${pageUrl}`);
    process.exit(0);
  }

  const mcpBin = path.join(binDir, 'origin-mcp');
  const serverBin = path.join(binDir, 'origin-server');

  // Skip if both binaries already exist
  if (fs.existsSync(mcpBin) && fs.existsSync(serverBin)) {
    console.log(`[origin-mcp] Binaries for ${label} already installed, skipping download.`);
    return;
  }

  console.log(`[origin-mcp] Fetching release info for ${label}...`);
  let release;
  try {
    release = await getJson(apiUrl);
  } catch (err) {
    console.error(`[origin-mcp] Could not fetch release info: ${err.message}`);
    console.error('[origin-mcp] Postinstall download skipped. First run will retry automatically.');
    console.error(`[origin-mcp] Install manually: ${pageUrl}`);
    process.exit(0);
  }

  const assets = release.assets || [];
  const mcpAssetName = `origin-mcp-${target}`;
  const serverAssetName = `origin-server-${target}`;

  const mcpAsset = assets.find((a) => a.name === mcpAssetName);
  const serverAsset = assets.find((a) => a.name === serverAssetName);

  if (!mcpAsset || !serverAsset) {
    console.error(`[origin-mcp] Could not find release assets for target "${target}".`);
    console.error(`[origin-mcp] Expected: ${mcpAssetName}, ${serverAssetName}`);
    console.error('[origin-mcp] Postinstall download skipped. First run will retry automatically.');
    console.error(`[origin-mcp] Install manually: ${pageUrl}`);
    process.exit(0);
  }

  // Ensure bin directory exists
  fs.mkdirSync(binDir, { recursive: true });

  try {
    if (!fs.existsSync(mcpBin)) {
      console.log(`[origin-mcp] Downloading ${mcpAssetName}...`);
      await downloadToFile(mcpAsset.browser_download_url, mcpBin);
      fs.chmodSync(mcpBin, 0o755);
      try { execFileSync('xattr', ['-cr', mcpBin]); } catch (_) {}
      console.log('[origin-mcp] origin-mcp installed.');
    }

    if (!fs.existsSync(serverBin)) {
      console.log(`[origin-mcp] Downloading ${serverAssetName}...`);
      await downloadToFile(serverAsset.browser_download_url, serverBin);
      fs.chmodSync(serverBin, 0o755);
      try { execFileSync('xattr', ['-cr', serverBin]); } catch (_) {}
      console.log('[origin-mcp] origin-server installed.');
    }

    console.log('[origin-mcp] Installation complete.');
  } catch (err) {
    console.error(`[origin-mcp] Download failed: ${err.message}`);
    console.error('[origin-mcp] Postinstall download skipped. First run will retry automatically.');
    console.error(`[origin-mcp] Install manually: ${pageUrl}`);
    process.exit(0);
  }
}

main();
