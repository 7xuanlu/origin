#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
//
// Runs the Tauri CLI with TAURI_DIR pointing at app/, which is where this repo
// keeps tauri.conf.json. Without it the CLI does not find the app at all:
// `tauri info` reports no App section, and a build has nothing to build.
//
// package.json used to do this as `TAURI_DIR=app tauri`. That is POSIX shell
// syntax, and pnpm hands the script to cmd.exe on Windows, which reads it as a
// command named TAURI_DIR and fails with "'TAURI_DIR' is not recognized as an
// internal or external command". Setting the variable from Node works the same
// way everywhere.
import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

// Located through the manifest's own bin entry rather than by resolving
// "@tauri-apps/cli/tauri.js" directly. That deep path works today only because
// the package declares no `exports` field; the day it adds one, a deep specifier
// throws ERR_PACKAGE_PATH_NOT_EXPORTED and every build breaks at once. A
// package.json is always resolvable.
const require = createRequire(import.meta.url);
const manifestPath = require.resolve("@tauri-apps/cli/package.json");
const manifest = require("@tauri-apps/cli/package.json");
const cli = resolve(dirname(manifestPath), manifest.bin.tauri);

// Spawned through the current Node binary rather than node_modules/.bin/tauri,
// because that entry is a cmd shim on Windows and would put a shell back in the
// path this script exists to avoid.
const child = spawn(process.execPath, [cli, ...process.argv.slice(2)], {
  stdio: "inherit",
  env: { ...process.env, TAURI_DIR: "app" },
});

child.on("error", (error) => {
  console.error(`failed to start the Tauri CLI: ${error.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  // Re-raise rather than translating, so Ctrl-C and timeouts still look like
  // signals to whatever invoked pnpm.
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
