#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0

const childProcess = require("child_process");
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const REPO = "7xuanlu/wenlan";
const ASSET = "wenlan-darwin-arm64.tar.gz";
const REQUESTED_TAG =
  process.env.WENLAN_RELEASE_TAG ||
  process.env.WENLAN_TAG ||
  process.env.ORIGIN_RELEASE_TAG ||
  process.env.ORIGIN_TAG ||
  "";
const BINARIES = ["wenlan", "wenlan-server", "wenlan-mcp"];

const RELEASE_TAG_PATTERN = /^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;

function safeTag(tag) {
  return tag.replace(/[^A-Za-z0-9._-]/g, "_");
}

function checkTag(tag) {
  if (!RELEASE_TAG_PATTERN.test(tag)) {
    throw new Error(
      `'${tag}' is not a Wenlan release tag (expected vX.Y.Z, like v0.17.0). ` +
        `See https://github.com/${REPO}/releases.`
    );
  }
  return tag;
}

function binDir() {
  if (REQUESTED_TAG) {
    return path.join(os.homedir(), ".wenlan", "releases", safeTag(REQUESTED_TAG));
  }
  return path.join(os.homedir(), ".wenlan", "bin");
}

// The public releases page redirects to the latest tag. Unlike the GitHub API
// it has no limit of 60 anonymous calls per hour per IP address, which a
// shared network can exhaust before a first install.
function latestTag() {
  const url = `https://github.com/${REPO}/releases/latest`;
  const tagPage = `https://github.com/${REPO}/releases/tag/`;
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: { "User-Agent": "wenlan" } }, (res) => {
        res.resume();
        try {
          const redirected = res.statusCode >= 300 && res.statusCode < 400;
          const location = (res.headers.location || "").split(/[?#]/)[0];
          if (!redirected || !location.startsWith(tagPage)) {
            throw new Error(
              `Could not find the latest Wenlan release at ${url} (HTTP ${res.statusCode}). ` +
                "Check the network, or pin one with WENLAN_RELEASE_TAG=vX.Y.Z."
            );
          }
          resolve(checkTag(decodeURIComponent(location.slice(tagPage.length))));
        } catch (err) {
          reject(err);
        }
      })
      .on("error", reject);
  });
}

function request(url, headers = {}) {
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: { "User-Agent": "wenlan", ...headers } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          resolve(request(res.headers.location, headers));
          return;
        }
        if (res.statusCode < 200 || res.statusCode >= 300) {
          res.resume();
          reject(new Error(`HTTP ${res.statusCode} for ${url}`));
          return;
        }
        resolve(res);
      })
      .on("error", reject);
  });
}

async function download(url, dest) {
  const res = await request(url);
  await new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    res.pipe(file);
    file.on("finish", () => file.close(resolve));
    file.on("error", reject);
  });
}

function extractBinaries(archivePath, dir) {
  const result = childProcess.spawnSync(
    "tar",
    ["-xzf", archivePath, "-C", dir, ...BINARIES],
    { stdio: "inherit" }
  );
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`tar exited with status ${result.status}`);
}

function run(command, args) {
  const result = childProcess.spawnSync(command, args, { stdio: "inherit" });
  if (result.error) throw result.error;
  process.exitCode = result.status || 0;
  if (process.exitCode !== 0) {
    process.exit(process.exitCode);
  }
}

function printPathHint(dir) {
  const entries = (process.env.PATH || "").split(path.delimiter);
  if (entries.includes(dir)) return;

  process.stderr.write(
    `\nWenlan binaries are installed in ${dir}.\n` +
      `Add them to your shell PATH if you want to run wenlan directly:\n\n` +
      `  export PATH="${dir}:$PATH"\n\n`
  );
}

async function installBinaries() {
  if (process.platform !== "darwin" || process.arch !== "arm64") {
    throw new Error("Wenlan setup currently supports macOS Apple Silicon only.");
  }

  const tag = REQUESTED_TAG ? checkTag(REQUESTED_TAG) : await latestTag();

  const dir = binDir();
  fs.mkdirSync(dir, { recursive: true });

  const archivePath = path.join(os.tmpdir(), ASSET);
  const url = `https://github.com/${REPO}/releases/download/${tag}/${ASSET}`;
  process.stderr.write(`Downloading Wenlan ${tag}...\n`);

  try {
    try {
      await download(url, archivePath);
    } catch (err) {
      throw new Error(
        `${err.message}. Check that the release exists at https://github.com/${REPO}/releases/tag/${tag}`
      );
    }
    extractBinaries(archivePath, dir);
  } finally {
    try {
      if (fs.existsSync(archivePath)) fs.unlinkSync(archivePath);
    } catch (_) {
      // Best-effort cleanup only.
    }
  }

  for (const name of BINARIES) {
    const dest = path.join(dir, name);
    fs.chmodSync(dest, 0o755);
  }

  return dir;
}

async function main() {
  const args = process.argv.slice(2);
  const dir = await installBinaries();
  const wenlan = path.join(dir, "wenlan");

  if (args[0] === "setup") {
    const setupArgs = args.slice(1);
    run(wenlan, ["setup", ...(setupArgs.length ? setupArgs : ["--basic"])]);
    run(wenlan, ["background", "on"]);
    run(wenlan, ["status", "--format", "table"]);
    printPathHint(dir);
    return;
  }

  run(wenlan, args.length ? args : ["--help"]);
}

main().catch((err) => {
  console.error(`wenlan setup failed: ${err.message}`);
  process.exit(1);
});
