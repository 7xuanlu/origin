// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

type Theme = "dark" | "light";
type Color = { readonly r: number; readonly g: number; readonly b: number; readonly a: number };
type ContrastCase = { readonly background: Color; readonly foreground: Color; readonly label: string };

const sourceDirectory = dirname(fileURLToPath(import.meta.url));
const indexCss = readFileSync(resolve(sourceDirectory, "index.css"), "utf8");
const navigationCss = readFileSync(resolve(sourceDirectory, "components/memory/navigation/navigation-shell.css"), "utf8");
const reviewBadgeCss = readFileSync(resolve(sourceDirectory, "components/memory/navigation/review-environment-badge.css"), "utf8");
const spacesCss = readFileSync(resolve(sourceDirectory, "components/memory/spaces/spacesInventory.css"), "utf8");
const spaceDetailCss = readFileSync(resolve(sourceDirectory, "components/memory/space-detail/space-detail.css"), "utf8");
const spaceHeaderCss = readFileSync(resolve(sourceDirectory, "components/memory/space-detail/space-detail-header.css"), "utf8");
const entityCss = readFileSync(resolve(sourceDirectory, "components/memory/entity-detail/EntityDetail.css"), "utf8");

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function themeToken(theme: Theme, name: string): string {
  const selector = theme === "dark"
    ? /:root,\s*html\[data-theme="dark"\]\s*\{(?<body>[\s\S]*?)\n\}/g
    : /html\[data-theme="light"\]\s*\{(?<body>[\s\S]*?)\n\}/g;
  const blocks = indexCss.matchAll(selector);
  for (const block of blocks) {
    const value = block.groups?.body.match(new RegExp(`${escapeRegex(name)}:\\s*([^;]+);`))?.[1]?.trim();
    if (value) return value;
  }
  throw new Error(`Missing ${theme} token ${name}`);
}

function tokenWithFallback(theme: Theme, preferred: string, fallback: string): string {
  try {
    return themeToken(theme, preferred);
  } catch (error) {
    if (!(error instanceof Error)) throw error;
    return themeToken(theme, fallback);
  }
}

function parseColor(value: string): Color {
  const hex = value.match(/^#(?<r>[0-9A-F]{2})(?<g>[0-9A-F]{2})(?<b>[0-9A-F]{2})$/i);
  if (hex?.groups) {
    return {
      r: Number.parseInt(hex.groups.r, 16),
      g: Number.parseInt(hex.groups.g, 16),
      b: Number.parseInt(hex.groups.b, 16),
      a: 1,
    };
  }
  const rgba = value.match(/^rgba\(\s*(?<r>\d+)\s*,\s*(?<g>\d+)\s*,\s*(?<b>\d+)\s*,\s*(?<a>[\d.]+)\s*\)$/);
  if (!rgba?.groups) throw new Error(`Unsupported CSS color ${value}`);
  return {
    r: Number(rgba.groups.r),
    g: Number(rgba.groups.g),
    b: Number(rgba.groups.b),
    a: Number(rgba.groups.a),
  };
}

function composite(foreground: Color, background: Color): Color {
  return {
    r: foreground.r * foreground.a + background.r * (1 - foreground.a),
    g: foreground.g * foreground.a + background.g * (1 - foreground.a),
    b: foreground.b * foreground.a + background.b * (1 - foreground.a),
    a: 1,
  };
}

function mix(foreground: Color, background: Color, foregroundShare: number): Color {
  return composite({ ...foreground, a: foregroundShare }, background);
}

function luminance(color: Color): number {
  const channel = (value: number) => {
    const normalized = value / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
}

function contrast(foreground: Color, background: Color): number {
  const lighter = Math.max(luminance(foreground), luminance(background));
  const darker = Math.min(luminance(foreground), luminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

function ruleDeclaration(source: string, selector: string, property: string): string {
  let searchFrom = 0;
  const selectorPattern = new RegExp(`^\\s*${escapeRegex(selector)}(?=\\s*[,\\{])`, "gm");
  while (searchFrom < source.length) {
    selectorPattern.lastIndex = searchFrom;
    const selectorMatch = selectorPattern.exec(source);
    if (!selectorMatch || selectorMatch.index < 0) break;
    const selectorStart = selectorMatch.index;
    const bodyStart = source.indexOf("{", selectorStart);
    const bodyEnd = source.indexOf("}", bodyStart);
    if (bodyStart < 0 || bodyEnd < 0) break;
    const body = source.slice(bodyStart + 1, bodyEnd);
    const value = body.match(new RegExp(`${escapeRegex(property)}:\\s*([^;]+);`))?.[1]?.trim();
    if (value) return value;
    searchFrom = bodyEnd + 1;
  }
  throw new Error(`Missing ${property} in ${selector}`);
}

function resolveDeclarationColor(theme: Theme, declaration: string): Color {
  const colorMix = declaration.match(/color-mix\(in srgb,\s*var\((?<foreground>--[\w-]+)\)\s*(?<share>\d+)%\s*,\s*var\((?<background>--[\w-]+)\)\)/);
  if (colorMix?.groups) {
    return mix(
      parseColor(themeToken(theme, colorMix.groups.foreground)),
      parseColor(themeToken(theme, colorMix.groups.background)),
      Number(colorMix.groups.share) / 100,
    );
  }
  const token = declaration.match(/var\((?<name>--[\w-]+)\)/)?.groups?.name;
  if (!token) throw new Error(`Unsupported color declaration ${declaration}`);
  return parseColor(themeToken(theme, token));
}

function expectContrast(cases: readonly ContrastCase[], minimum: number): void {
  for (const testCase of cases) {
    expect(contrast(testCase.foreground, testCase.background), testCase.label).toBeGreaterThanOrEqual(minimum);
  }
}

describe("Spaces navigation redesign contrast", () => {
  it("keeps the Review proof stamp AA-readable in both themes", () => {
    for (const theme of ["dark", "light"] as const) {
      expectContrast([{
        label: `${theme} Review proof stamp`,
        foreground: resolveDeclarationColor(theme, ruleDeclaration(reviewBadgeCss, ".review-environment-stamp", "color")),
        background: parseColor(themeToken(theme, "--mem-sidebar")),
      }], 4.5);
    }
  });

  it("keeps metadata and sage identity text AA-readable in both themes", () => {
    for (const theme of ["dark", "light"] as const) {
      const canvas = parseColor(themeToken(theme, "--mem-bg"));
      const surface = parseColor(themeToken(theme, "--mem-detail-surface"));
      const sidebar = parseColor(themeToken(theme, "--mem-sidebar"));
      const raised = parseColor(themeToken(theme, "--mem-detail-surface-raised"));
      const hover = parseColor(themeToken(theme, "--mem-hover"));
      const hoverStrong = parseColor(themeToken(theme, "--mem-hover-strong"));
      const metadata = parseColor(tokenWithFallback(theme, "--mem-text-meta-aa", "--mem-text-tertiary"));
      const sage = parseColor(tokenWithFallback(theme, "--mem-accent-sage-text", "--mem-accent-sage"));
      const sageAccent = parseColor(themeToken(theme, "--mem-accent-sage"));
      expectContrast([
        { label: `${theme} metadata on canvas`, foreground: metadata, background: canvas },
        { label: `${theme} metadata on raised`, foreground: metadata, background: raised },
        { label: `${theme} metadata on row hover`, foreground: metadata, background: composite(hover, canvas) },
        { label: `${theme} metadata on selected recent`, foreground: metadata, background: composite(hoverStrong, sidebar) },
        { label: `${theme} metadata on graph hover`, foreground: metadata, background: composite(hoverStrong, surface) },
        { label: `${theme} sage action`, foreground: sage, background: canvas },
        { label: `${theme} sage seal`, foreground: sage, background: mix(sageAccent, canvas, 0.14) },
        { label: `${theme} sage graph hover`, foreground: sage, background: composite(hoverStrong, surface) },
      ], 4.5);
    }
  });

  it("keeps required controls and entity graphics distinguishable", () => {
    for (const theme of ["dark", "light"] as const) {
      const canvas = parseColor(themeToken(theme, "--mem-bg"));
      const surface = parseColor(themeToken(theme, "--mem-detail-surface"));
      const raised = parseColor(themeToken(theme, "--mem-detail-surface-raised"));
      const control = parseColor(tokenWithFallback(theme, "--mem-control-border", "--mem-border"));
      expectContrast([
        { label: `${theme} control on canvas`, foreground: control, background: canvas },
        { label: `${theme} control on surface`, foreground: control, background: surface },
        { label: `${theme} control on raised`, foreground: control, background: raised },
        { label: `${theme} entity seal`, foreground: resolveDeclarationColor(theme, ruleDeclaration(indexCss, ".entity-detail-seal", "border")), background: canvas },
        { label: `${theme} relationship edge`, foreground: resolveDeclarationColor(theme, ruleDeclaration(indexCss, ".entity-graph-edges line", "stroke")), background: surface },
        { label: `${theme} observation editor`, foreground: resolveDeclarationColor(theme, ruleDeclaration(indexCss, ".entity-obs-input", "border-bottom")), background: surface },
      ], 3);
    }
  });

  it("uses the scoped AA tokens only on the redesigned surfaces", () => {
    for (const [source, selector] of [
      [navigationCss, ".memory-sidebar"],
      [spacesCss, ".spaces-overview"],
      [spaceHeaderCss, ".space-dossier"],
      [entityCss, ".entity-detail-dossier"],
    ] as const) {
      expect(ruleDeclaration(source, selector, "--mem-text-tertiary")).toBe("var(--mem-text-meta-aa)");
    }
    for (const [source, selector] of [
      [spaceDetailCss, ".space-dossier-text-action"],
      [indexCss, ".entity-detail-seal"],
      [indexCss, ".entity-relation-name"],
    ] as const) {
      expect(ruleDeclaration(source, selector, "color")).toBe("var(--mem-accent-sage-text)");
    }
    expect(ruleDeclaration(indexCss, ".entity-graph-node-name", "color")).toBe(
      "var(--mem-text)",
    );
    for (const [source, selector, property] of [
      [spacesCss, ".spaces-filter input", "border"],
      [spaceHeaderCss, ".space-dossier-title-input", "border-bottom"],
      [spaceHeaderCss, ".space-dossier-description-editor", "border"],
    ] as const) {
      expect(ruleDeclaration(source, selector, property), selector).toContain("var(--mem-control-border)");
    }
    expect(ruleDeclaration(spacesCss, ".spaces-menu", "border")).toContain(
      "var(--mem-popover-border)",
    );
  });
});
