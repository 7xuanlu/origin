#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
//
// Runs one of this repo's POSIX shell scripts under a POSIX bash.
//
// package.json used to spell these as `bash scripts/<name>.sh`. On Windows that
// resolves through PATH, and on any machine with WSL installed the first `bash`
// on PATH is C:\Windows\System32\bash.exe -- the Linux distro, not Git Bash. The
// scripts then run inside Ubuntu, where the Windows toolchain does not exist, and
// die with `rustc: command not found` before doing anything. `pnpm tauri dev` and
// `pnpm tauri build` both hit this through beforeDevCommand/beforeBuildCommand,
// so the desktop app could not be started on Windows at all.
//
// Resolving Git Bash explicitly is the fix. Everywhere else `bash` is already the
// right answer and this is a passthrough.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, join } from "node:path";

function windowsBash() {
  const candidates = [];
  // An explicit override wins, for shells installed somewhere unusual.
  if (process.env.WENLAN_BASH) {
    candidates.push(process.env.WENLAN_BASH);
  }
  for (const root of [
    process.env.ProgramFiles,
    process.env["ProgramFiles(x86)"],
    process.env.ProgramW6432,
    process.env.LOCALAPPDATA && join(process.env.LOCALAPPDATA, "Programs"),
  ]) {
    if (root) {
      candidates.push(join(root, "Git", "bin", "bash.exe"));
    }
  }
  // Git is usually on PATH as <root>\cmd\git.exe even when <root>\bin is not,
  // so derive the sibling bash from wherever git itself lives.
  for (const entry of (process.env.PATH ?? "").split(delimiter)) {
    if (/[\\/]git[\\/](cmd|bin)$/i.test(entry)) {
      candidates.push(join(entry, "..", "bin", "bash.exe"));
    }
  }
  return candidates.find((candidate) => existsSync(candidate));
}

const [script, ...rest] = process.argv.slice(2);
if (!script) {
  console.error("usage: node scripts/run-bash.mjs <script.sh> [args...]");
  process.exit(2);
}

let bash = "bash";
if (process.platform === "win32") {
  bash = windowsBash();
  if (!bash) {
    console.error(
      "no Git Bash found. Install Git for Windows, or point WENLAN_BASH at a " +
        "POSIX bash. The `bash` on PATH is WSL and cannot see the Windows " +
        "toolchain this script needs.",
    );
    process.exit(1);
  }
}

const child = spawn(bash, [script, ...rest], { stdio: "inherit" });

child.on("error", (error) => {
  console.error(`failed to start ${bash}: ${error.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  // Re-raise rather than translating, so Ctrl-C still looks like a signal to
  // whatever invoked pnpm.
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
