// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { relative } from "node:path";

export function toPosixPath(value: string): string {
  return value.replaceAll("\\", "/");
}

export function repoRelativePath(file: string, cwd = process.cwd()): string {
  return toPosixPath(relative(cwd, file));
}

export function normalizeSourceText(value: string): string {
  return value.replace(/\r\n?/g, "\n");
}

export function readSourceText(file: string): string {
  return normalizeSourceText(readFileSync(file, "utf8"));
}
