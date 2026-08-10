// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import ts from "typescript";
import fs from "node:fs";
import path from "node:path";
import {
  readSourceText,
  repoRelativePath,
} from "../test/sourceText";

const COMPONENTS_DIR = path.resolve("src/components");
const USER_FACING_ATTRIBUTES = new Set([
  "aria-description",
  "aria-label",
  "alt",
  "placeholder",
  "title",
]);

const BASELINE_FILE = path.resolve("src/i18n/hardcodedCopyBaseline.tsv");

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

function normalizeCopy(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function looksLikeHumanEnglish(value: string): boolean {
  const normalized = normalizeCopy(value);
  if (!/[A-Za-z]/.test(normalized)) return false;
  if (/^[a-z][a-z0-9_-]*$/i.test(normalized) && normalized.length <= 3) {
    return false;
  }
  if (/^[A-Z0-9_./:-]+$/.test(normalized)) return false;
  if (/^[a-z][A-Za-z0-9]*(\.[A-Za-z0-9]+)+$/.test(normalized)) return false;
  if (/^[a-z0-9_-]+(\s+[a-z0-9_-]+)+$/.test(normalized) && normalized.includes("-")) {
    return false;
  }
  return /[a-z]/.test(normalized);
}

function nearestJsxAttribute(node: ts.Node): ts.JsxAttribute | null {
  let current: ts.Node | undefined = node.parent;
  while (current) {
    if (ts.isJsxAttribute(current)) return current;
    if (ts.isFunctionLike(current) || ts.isSourceFile(current)) return null;
    current = current.parent;
  }
  return null;
}

function jsxAttributeName(attribute: ts.JsxAttribute): string | null {
  return ts.isIdentifier(attribute.name) ? attribute.name.text : null;
}

function unwrapParentheses(node: ts.Node): ts.Node {
  let current = node;
  while (ts.isParenthesizedExpression(current)) current = current.expression;
  return current;
}

function isRenderedExpressionNode(node: ts.Node): boolean {
  let current: ts.Node = node;

  while (current.parent && ts.isParenthesizedExpression(current.parent)) {
    current = current.parent;
  }

  const parent = current.parent;
  if (!parent) return false;

  if (ts.isJsxExpression(parent) && unwrapParentheses(parent.expression ?? node) === current) {
    return true;
  }

  if (ts.isConditionalExpression(parent) && (parent.whenTrue === current || parent.whenFalse === current)) {
    let expression: ts.Node = parent;
    while (expression.parent && ts.isConditionalExpression(expression.parent)) {
      expression = expression.parent;
    }
    while (expression.parent && ts.isParenthesizedExpression(expression.parent)) {
      expression = expression.parent;
    }
    return ts.isJsxExpression(expression.parent) && expression.parent.expression === expression;
  }

  return false;
}

function isConstStringDeclaration(node: ts.Node): node is ts.VariableDeclaration & {
  name: ts.Identifier;
  initializer: ts.StringLiteral;
} {
  return (
    ts.isVariableDeclaration(node) &&
    ts.isIdentifier(node.name) &&
    node.initializer !== undefined &&
    ts.isStringLiteral(node.initializer) &&
    ts.isVariableDeclarationList(node.parent) &&
    (node.parent.flags & ts.NodeFlags.Const) !== 0
  );
}

describe("hardcoded UI copy guard", () => {
  it("does not add untranslated English copy in JSX surfaces", () => {
    const currentCounts = new Map<string, number>();
    const baselineCounts = readBaselineCounts();

    for (const file of listTsxFiles(COMPONENTS_DIR)) {
      const sourceText = readSourceText(file);
      const sourceFile = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
      const relativeFile = repoRelativePath(file);
      const constStringValues = new Map<string, string>();

      const record = (value: string) => {
        const copy = normalizeCopy(value);
        if (!copy || !looksLikeHumanEnglish(copy)) return;
        const key = baselineKey(relativeFile, copy);
        currentCounts.set(key, (currentCounts.get(key) ?? 0) + 1);
      };

      const collectConstants = (node: ts.Node) => {
        if (isConstStringDeclaration(node) && looksLikeHumanEnglish(node.initializer.text)) {
          constStringValues.set(node.name.text, node.initializer.text);
        }
        ts.forEachChild(node, collectConstants);
      };

      collectConstants(sourceFile);

      const visit = (node: ts.Node) => {
        if (ts.isJsxText(node)) {
          record(node.getText(sourceFile));
        }

        if (ts.isJsxAttribute(node) && USER_FACING_ATTRIBUTES.has(jsxAttributeName(node) ?? "")) {
          if (node.initializer && ts.isStringLiteral(node.initializer)) {
            record(node.initializer.text);
          }
        }

        if (ts.isStringLiteral(node)) {
          const attribute = nearestJsxAttribute(node);
          if (attribute) {
            if (USER_FACING_ATTRIBUTES.has(jsxAttributeName(attribute) ?? "")) record(node.text);
          } else if (isRenderedExpressionNode(node)) {
            record(node.text);
          }
        }

        if (ts.isIdentifier(node) && constStringValues.has(node.text)) {
          const attribute = nearestJsxAttribute(node);
          const visibleAttribute = attribute && USER_FACING_ATTRIBUTES.has(jsxAttributeName(attribute) ?? "");
          if (visibleAttribute || (!attribute && isRenderedExpressionNode(node))) {
            record(constStringValues.get(node.text)!);
          }
        }

        ts.forEachChild(node, visit);
      };

      visit(sourceFile);
    }

    const newHardcodedCopy = [...currentCounts.entries()]
      .filter(([key, count]) => count > (baselineCounts.get(key) ?? 0))
      .map(([key, count]) => `${key}\tcurrent=${count}\tbaseline=${baselineCounts.get(key) ?? 0}`);

    expect(newHardcodedCopy).toEqual([]);
  });
});

function baselineKey(file: string, copy: string): string {
  return `${file}\t${copy}`;
}

function readBaselineCounts(): Map<string, number> {
  const counts = new Map<string, number>();
  const source = fs.readFileSync(BASELINE_FILE, "utf8");
  for (const line of source.split("\n")) {
    const trimmed = line.trimEnd();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const [file, copy, countValue] = trimmed.split("\t");
    const count = Number(countValue);
    if (!file || !copy || !Number.isInteger(count) || count < 1) {
      throw new Error(`Invalid hardcoded copy baseline row: ${line}`);
    }
    counts.set(baselineKey(file, copy), count);
  }
  return counts;
}
