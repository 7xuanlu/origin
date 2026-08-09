// SPDX-License-Identifier: AGPL-3.0-only
import { execFileSync } from "node:child_process";
import {
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const bundle = resolve(
  root,
  process.argv[2] ?? "target/debug/bundle/macos/Wenlan Review.app",
);
const contents = resolve(bundle, "Contents");
const plist = resolve(contents, "Info.plist");

function assert(condition, message) {
  if (!condition) throw new Error(`Review bundle verification failed: ${message}`);
}

function plistValue(key) {
  return execFileSync(
    "/usr/libexec/PlistBuddy",
    ["-c", `Print :${key}`, plist],
    { encoding: "utf8" },
  ).trim();
}

function filesBelow(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    return entry.isDirectory() ? filesBelow(path) : [path];
  });
}

assert(
  plistValue("CFBundleIdentifier") === "com.wenlan.desktop.review",
  "unexpected bundle identifier",
);
assert(
  plistValue("CFBundleDisplayName") === "Wenlan Review",
  "unexpected display name",
);
assert(
  plistValue("CFBundleExecutable") === "wenlan-review",
  "unexpected executable name",
);

const macosFiles = filesBelow(resolve(contents, "MacOS"))
  .map((path) => relative(resolve(contents, "MacOS"), path));
assert(
  macosFiles.length === 1 && macosFiles[0] === "wenlan-review",
  `Contents/MacOS must contain only wenlan-review; found ${macosFiles.join(", ")}`,
);
assert(
  (statSync(resolve(contents, "MacOS/wenlan-review")).mode & 0o111) !== 0,
  "wenlan-review is not executable",
);

const forbiddenBundleNames = [
  "wenlan-server",
  "wenlan-mcp",
  "wenlan-cli",
  "cloudflared",
];
const bundledFiles = filesBelow(contents).map((path) => relative(contents, path));
const forbiddenFiles = bundledFiles.filter((path) =>
  forbiddenBundleNames.some((name) => path.toLowerCase().includes(name))
);
assert(
  forbiddenFiles.length === 0,
  `production integration binaries are bundled: ${forbiddenFiles.join(", ")}`,
);

console.log(
  `Verified fixture-only Review bundle: ${bundle} (${bundledFiles.length} files, no sidecars)`,
);
