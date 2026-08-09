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
  const match = cargoToml.match(/^version = "([^"]+)"(.*)$/m);
  if (!match) {
    throw new Error(`${path} is missing a package version`);
  }
  return match[0];
}

function workspaceVersion(): string {
  return readFileSync(resolve(root, "version.txt"), "utf8").trim();
}

describe("release version sync", () => {
  it("keeps the desktop app version in lockstep with the workspace version", () => {
    const workspace = workspaceVersion();
    const appCargoLine = cargoVersionLine("app/Cargo.toml");
    const appCargoMatch = appCargoLine.match(/^version = "([^"]+)"/);
    const versions = {
      tauri: jsonVersion("app/tauri.conf.json"),
      packageJson: jsonVersion("package.json"),
      appCargo: appCargoMatch ? appCargoMatch[1] : undefined,
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

  it("registers the app version trio as release-please extra-files", () => {
    const config = JSON.parse(
      readFileSync(resolve(root, "release-please-config.json"), "utf8"),
    );
    const extraFiles: unknown[] = config.packages["."]["extra-files"];

    expect(extraFiles).toContain("app/Cargo.toml");
    expect(extraFiles).toContainEqual({
      type: "json",
      path: "app/tauri.conf.json",
      jsonpath: "$.version",
    });
    expect(extraFiles).toContainEqual({
      type: "json",
      path: "package.json",
      jsonpath: "$.version",
    });
  });
});
