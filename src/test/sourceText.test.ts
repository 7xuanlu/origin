// SPDX-License-Identifier: AGPL-3.0-only
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  normalizeSourceText,
  readSourceText,
  repoRelativePath,
  toPosixPath,
} from "./sourceText";

describe("source text portability", () => {
  it("canonicalizes Windows separators for persisted keys", () => {
    expect(toPosixPath(String.raw`src\components\memory\PageInfo.tsx`)).toBe(
      "src/components/memory/PageInfo.tsx",
    );
  });

  it("returns repository-relative paths with forward slashes", () => {
    const file = resolve(process.cwd(), "src", "test", "sourceText.test.ts");

    expect(repoRelativePath(file)).toBe("src/test/sourceText.test.ts");
  });

  it("normalizes CRLF and lone CR without changing other text", () => {
    expect(normalizeSourceText("one\r\ntwo\rthree\n")).toBe(
      "one\ntwo\nthree\n",
    );
  });

  it("normalizes a source file at its read boundary", () => {
    const root = mkdtempSync(resolve(tmpdir(), "wenlan-source-text-"));
    const file = resolve(root, "fixture.ts");
    try {
      writeFileSync(file, "const one = 1;\r\nconst two = 2;\r\n");

      expect(readSourceText(file)).toBe(
        "const one = 1;\nconst two = 2;\n",
      );
    } finally {
      rmSync(root, { force: true, recursive: true });
    }
  });
});
