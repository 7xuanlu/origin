// SPDX-License-Identifier: AGPL-3.0-only
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, dirname, delimiter, resolve } from "node:path";

let cachedBashExecutable: string | undefined;

export const shellIntegrationTestTimeout = process.platform === "win32" ? 60_000 : 10_000;

export function prependNativePath(directory: string, inherited = process.env.PATH ?? ""): string {
  return inherited ? `${directory}${delimiter}${inherited}` : directory;
}

export function canonicalizePathEnvironment(
  environment: Record<string, string | undefined>,
  platform = process.platform,
): Record<string, string> {
  const normalized: Record<string, string> = {};
  for (const [key, value] of Object.entries(environment)) {
    if (value === undefined) continue;
    const normalizedKey = platform === "win32" && key.toLowerCase() === "path"
      ? "PATH"
      : key;
    normalized[normalizedKey] = value;
  }
  return normalized;
}

export function bashExecutable(): string {
  if (cachedBashExecutable) return cachedBashExecutable;
  if (process.platform !== "win32") return "bash";

  const gitRoots = whereExecutables("git.exe")
    .map((git) => dirname(git))
    .map((directory) => (
      ["bin", "cmd"].includes(basename(directory).toLowerCase())
        ? resolve(directory, "..")
        : directory
    ));
  for (const programFiles of [
    process.env.ProgramFiles,
    process.env["ProgramFiles(x86)"],
  ]) {
    if (programFiles) gitRoots.push(resolve(programFiles, "Git"));
  }

  const gitBashCandidates = gitRoots.map((gitRoot) => resolve(gitRoot, "bin", "bash.exe"));
  const pathCandidates = whereExecutables("bash.exe");
  const launcher = [...gitBashCandidates, ...pathCandidates].find(
    (candidate) => (
      existsSync(candidate)
      && !/[\\/]Windows[\\/]System32[\\/]bash\.exe$/i.test(candidate)
      && !/[\\/]Microsoft[\\/]WindowsApps[\\/]bash\.exe$/i.test(candidate)
    ),
  );
  if (!launcher) {
    throw new Error(
      "Git for Windows Bash was not found; WSL bash.exe cannot run native Wenlan test tools",
    );
  }
  cachedBashExecutable = launcher;
  return launcher;
}

function whereExecutables(name: string): string[] {
  try {
    return execFileSync("where.exe", [name], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    })
      .split(/\r?\n/)
      .map((candidate) => candidate.trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

export function minimalBashPath(): string {
  if (process.platform !== "win32") return ["/usr/bin", "/bin"].join(delimiter);

  const launcherDirectory = dirname(bashExecutable());
  const paths = [launcherDirectory];
  const launcherParent = resolve(launcherDirectory, "..");
  if (
    basename(launcherDirectory).toLowerCase() === "bin"
    && basename(launcherParent).toLowerCase() === "git"
  ) {
    paths.push(resolve(launcherDirectory, "..", "usr", "bin"));
  }
  return [...new Set(paths)].join(delimiter);
}

export function canonicalBashPath(directory: string): string {
  return execFileSync(bashExecutable(), ["-lc", "pwd -P"], {
    cwd: directory,
    encoding: "utf8",
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}
