import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function jsonVersion(path: string): string {
  return JSON.parse(readFileSync(resolve(root, path), "utf8")).version;
}

function cargoVersionLine(path: string): string {
  const cargoToml = readFileSync(resolve(root, path), "utf8");
  const match = cargoToml.match(/^version = "[^"]+".*$/m);
  if (!match) {
    throw new Error(`${path} is missing a package version line`);
  }
  return match[0];
}

function cargoVersion(path: string): string {
  const match = cargoVersionLine(path).match(/^version = "([^"]+)"/);
  if (!match) {
    throw new Error(`${path} version line has no quoted version`);
  }
  return match[1];
}

function workspaceVersion(): string {
  return readFileSync(resolve(root, "version.txt"), "utf8").trim();
}

describe("release version sync", () => {
  it("keeps the desktop app version in lockstep with the workspace version", () => {
    const workspace = workspaceVersion();
    const versions = {
      tauri: jsonVersion("app/tauri.conf.json"),
      packageJson: jsonVersion("package.json"),
      appCargo: cargoVersion("app/Cargo.toml"),
    };

    expect(versions).toEqual({
      tauri: workspace,
      packageJson: workspace,
      appCargo: workspace,
    });
  });

  it("keeps the x-release-please-version marker on the app/Cargo.toml version line", () => {
    // release-please's generic updater locates the line to bump via this marker;
    // losing it silently breaks lockstep instead of failing loud.
    const appCargoLine = cargoVersionLine("app/Cargo.toml");
    expect(appCargoLine).toContain("# x-release-please-version");
  });

  it("syncs the app version trio via scripts/bump-version.sh, not release-please extra-files", () => {
    // release-please's `extra-files` mechanism collides with
    // scripts/validate-release-candidate.py's closed release-please-config.json
    // schema and its pinned release-PR changed-file set, so the app trio rides
    // the same bump-version.sh mechanism as the daemon crates and the CC
    // plugin manifest instead.
    const bumpScript = readFileSync(resolve(root, "scripts/bump-version.sh"), "utf8");
    for (const path of ["app/Cargo.toml", "app/tauri.conf.json", "package.json"]) {
      expect(bumpScript).toContain(path);
    }
  });

  it("registers the app version trio as release-managed paths in validate-release-candidate.py", () => {
    const validator = readFileSync(
      resolve(root, "scripts/validate-release-candidate.py"),
      "utf8",
    );
    const match = validator.match(/RELEASE_MANAGED_PATHS = frozenset\(\s*\{([\s\S]*?)\}\s*\)/);
    if (!match) {
      throw new Error("RELEASE_MANAGED_PATHS block not found in validate-release-candidate.py");
    }
    const block = match[1];
    for (const path of ["app/Cargo.toml", "app/tauri.conf.json", "package.json"]) {
      expect(block).toContain(`"${path}"`);
    }
  });
});
