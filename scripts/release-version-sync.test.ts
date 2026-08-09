import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function jsonVersion(path: string): string {
  return JSON.parse(readFileSync(resolve(root, path), "utf8")).version;
}

function cargoVersion(): string {
  const cargoToml = readFileSync(resolve(root, "app/Cargo.toml"), "utf8");
  const match = cargoToml.match(/^version = "([^"]+)"/m);
  if (!match) {
    throw new Error("app/Cargo.toml is missing a package version");
  }
  return match[1];
}

function pinnedDaemonVersion(): string {
  const pin = readFileSync(resolve(root, ".wenlan-backend-version"), "utf8")
    .split(/\r?\n/)
    .find((line) => line.trim().length > 0);
  if (!pin) {
    throw new Error(".wenlan-backend-version is missing a daemon tag");
  }
  return pin.replace(/^v/, "");
}

describe("release version sync", () => {
  it("keeps the public app release version in lockstep with the pinned daemon release", () => {
    const versions = {
      tauri: jsonVersion("app/tauri.conf.json"),
      packageJson: jsonVersion("package.json"),
      cargo: cargoVersion(),
      pinnedDaemon: pinnedDaemonVersion(),
    };

    expect(versions).toEqual({
      tauri: versions.pinnedDaemon,
      packageJson: versions.pinnedDaemon,
      cargo: versions.pinnedDaemon,
      pinnedDaemon: versions.pinnedDaemon,
    });
  });
});
