// Point git at .githooks and make the hooks executable.
//
// Runs from package.json's `postinstall`. pnpm executes lifecycle scripts with
// cmd.exe on Windows, so the previous inline `git config ... 2>/dev/null &&
// chmod +x ... || true` was a parse error there and failed the whole
// `pnpm install` — the Windows app could not install its dependencies at all.
// Node is by definition available in a lifecycle script, so do the work here.
//
// Best-effort by design, exactly like the `|| true` it replaces: a checkout
// without git, or a read-only config, must not fail `pnpm install`.
import { chmodSync, readdirSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";

const hooksDir = ".githooks";

try {
  execFileSync("git", ["config", "core.hooksPath", hooksDir], {
    stdio: "ignore",
  });
} catch {
  process.exit(0);
}

// Windows has no POSIX execute bit; git for Windows ignores the mode entirely.
if (process.platform !== "win32") {
  try {
    for (const entry of readdirSync(hooksDir)) {
      const path = join(hooksDir, entry);
      if (statSync(path).isFile()) {
        chmodSync(path, 0o755);
      }
    }
  } catch {
    // A missing or unreadable .githooks is not an install failure.
  }
}
