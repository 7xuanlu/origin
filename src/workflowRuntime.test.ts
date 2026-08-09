// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// After the 2026-07-20 fold into the wenlan monorepo, the app's frontend
// toolchain built in its own standalone app-release.yml. That workflow was
// deleted once release.yml's app-bundle job proved the unified in-tree build
// (v0.15.7), so this now polices the app-bundle job's toolchain floor
// directly in release.yml, which uses SHA-pinned actions and a quoted
// node-version — the old standalone-repo backend-pin-bump.yml is gone
// (backend is in-tree now).
const workflows = [".github/workflows/release.yml"] as const;

describe("GitHub Actions runtime floor", () => {
  it.each(workflows)("%s avoids Node 20 action releases", (path) => {
    const workflow = readFileSync(resolve(path), "utf8");

    expect(workflow).not.toMatch(
      /(?:actions\/checkout|actions\/setup-node|pnpm\/action-setup)@v4/,
    );
  });

  it.each(workflows)("%s runs project scripts on Node 24", (path) => {
    const workflow = readFileSync(resolve(path), "utf8");

    expect(workflow).toContain('node-version: "24"');
    expect(workflow).not.toContain("node-version: 20");
    expect(workflow).not.toContain('node-version: "20"');
  });
});
