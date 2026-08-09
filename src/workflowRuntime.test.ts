// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// After the 2026-07-20 fold into the wenlan monorepo, the app's frontend
// toolchain lives in exactly one wholly-app-owned workflow: app-release.yml.
// The root ci.yml / release.yml are now the daemon's Rust workflows (SHA-pinned
// # v4 actions, quoted node-version) — not the app's to police, and the old
// standalone-repo backend-pin-bump.yml is gone (backend is in-tree now).
const workflows = [".github/workflows/app-release.yml"] as const;

describe("GitHub Actions runtime floor", () => {
  it.each(workflows)("%s avoids Node 20 action releases", (path) => {
    const workflow = readFileSync(resolve(path), "utf8");

    expect(workflow).not.toMatch(
      /(?:actions\/checkout|actions\/setup-node|pnpm\/action-setup)@v4/,
    );
  });

  it.each(workflows)("%s runs project scripts on Node 24", (path) => {
    const workflow = readFileSync(resolve(path), "utf8");

    expect(workflow).toContain("node-version: 24");
    expect(workflow).not.toContain("node-version: 20");
  });
});
