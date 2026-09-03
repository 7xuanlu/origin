// SPDX-License-Identifier: AGPL-3.0-only
//
// The POSIX bash these suites drive their shell fixtures with, resolved ONCE.
//
// On any Windows machine with WSL installed the first `bash` on PATH is
// C:\Windows\System32\bash.exe -- the Linux distro, not Git Bash -- so a bare
// `bash` would run the fixtures on a different host entirely. scripts/run-bash.mjs
// makes the same guarantee for package.json's script entries; this is the same
// candidate search for callers that already are node, so a case does not pay for
// a second node process (measured on this host: 223 ms through run-bash.mjs
// against 51 ms spawning Git Bash directly, times the ~90 spawns these two
// suites make).
//
// There is no fallback to a bare `bash` on Windows: resolveTestBash throws, so a
// machine without Git Bash reports that rather than silently measuring WSL.
import { existsSync } from "node:fs";
import { delimiter, join } from "node:path";

function windowsBash(): string | undefined {
  const candidates: string[] = [];
  // An explicit override wins, for shells installed somewhere unusual.
  if (process.env.WENLAN_BASH) candidates.push(process.env.WENLAN_BASH);
  for (const root of [
    process.env.ProgramFiles,
    process.env["ProgramFiles(x86)"],
    process.env.ProgramW6432,
    process.env.LOCALAPPDATA && join(process.env.LOCALAPPDATA, "Programs"),
  ]) {
    if (root) candidates.push(join(root, "Git", "bin", "bash.exe"));
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

export function resolveTestBash(): string {
  if (process.platform !== "win32") return "bash";
  const bash = windowsBash();
  if (!bash) {
    throw new Error(
      "no Git Bash found. Install Git for Windows, or point WENLAN_BASH at a " +
        "POSIX bash. The `bash` on PATH is WSL and cannot see the Windows " +
        "toolchain these fixtures measure.",
    );
  }
  return bash;
}
