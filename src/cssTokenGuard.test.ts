// SPDX-License-Identifier: AGPL-3.0-only
// Guards against the ConstellationMap.tsx "--mem-text-primary" bug: a
// var(--mem-*) / var(--kg-*) reference to a token that was never defined in
// index.css computes to `initial` (often invisible text) instead of erroring.
import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { repoRelativePath } from "./test/sourceText";

const SRC_DIR = path.resolve("src");
const CSS_FILE = path.resolve("src/index.css");
const TOKEN_RE = /var\(\s*(--(?:mem|kg)-[a-zA-Z0-9-]+)/g;

// Pre-existing undefined-token reference this guard caught but that isn't an
// obvious one-line typo (no existing token is an unambiguous stand-in for a
// literal "blue" accent — the hardcoded #60a5fa fallback suggests a new token
// was intended, not a rename of an existing one). Reported upstream instead
// of guessed at. New occurrences anywhere else still fail.
const KNOWN_MISSING_TOKENS = new Set([
  "src/components/memory/page/PageInfo.tsx\t--mem-accent-blue",
]);

function listTsxFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  return entries.flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) return listTsxFiles(fullPath);
    if (!entry.name.endsWith(".tsx")) return [];
    if (entry.name.endsWith(".test.tsx")) return [];
    return [fullPath];
  });
}

function definedTokens(css: string): Set<string> {
  const defined = new Set<string>();
  const re = /(--(?:mem|kg)-[a-zA-Z0-9-]+)\s*:/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(css))) defined.add(match[1]);
  return defined;
}

describe("CSS custom property guard", () => {
  it("does not reference a --mem-*/--kg-* token that index.css never defines", () => {
    const css = fs.readFileSync(CSS_FILE, "utf8");
    const defined = definedTokens(css);

    const missing: string[] = [];
    for (const file of listTsxFiles(SRC_DIR)) {
      const text = fs.readFileSync(file, "utf8");
      const relativeFile = repoRelativePath(file);
      const seenInFile = new Set<string>();
      let match: RegExpExecArray | null;
      TOKEN_RE.lastIndex = 0;
      while ((match = TOKEN_RE.exec(text))) {
        const token = match[1];
        const key = `${relativeFile}\t${token}`;
        if (!defined.has(token) && !seenInFile.has(token) && !KNOWN_MISSING_TOKENS.has(key)) {
          seenInFile.add(token);
          missing.push(key);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});
