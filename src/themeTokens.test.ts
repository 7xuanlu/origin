// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const css = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "index.css"), "utf8");

function darkToken(name: string): string | null {
  const darkBlock = css.match(/:root,\s*html\[data-theme="dark"\]\s*\{(?<body>[\s\S]*?)\n\}/);
  const body = darkBlock?.groups?.body ?? "";
  return body.match(new RegExp(`${name}:\\s*([^;]+);`))?.[1].trim() ?? null;
}

function lightToken(name: string): string | null {
  const blocks = css.matchAll(/html\[data-theme="light"\]\s*\{(?<body>[\s\S]*?)\n\}/g);
  for (const block of blocks) {
    const value = block.groups?.body.match(new RegExp(`${name}:\\s*([^;]+);`))?.[1].trim();
    if (value) return value;
  }
  return null;
}

function rootToken(name: string): string | null {
  const rootBlock = css.match(/:root\s*\{(?<body>[\s\S]*?)\n\}/);
  const body = rootBlock?.groups?.body ?? "";
  return body.match(new RegExp(`${name}:\\s*([^;]+);`))?.[1].trim() ?? null;
}

function rgb(hex: string): { r: number; g: number; b: number } {
  const match = hex.match(/^#(?<r>[0-9A-F]{2})(?<g>[0-9A-F]{2})(?<b>[0-9A-F]{2})$/i);
  if (!match?.groups) {
    throw new Error(`Expected hex color, got ${hex}`);
  }
  return {
    r: Number.parseInt(match.groups.r, 16),
    g: Number.parseInt(match.groups.g, 16),
    b: Number.parseInt(match.groups.b, 16),
  };
}

function expectLogoInk(hex: string | null): void {
  expect(hex).not.toBeNull();
  const color = rgb(hex ?? "");
  const spread = Math.max(color.r, color.g, color.b) - Math.min(color.r, color.g, color.b);

  expect(color.b).toBeGreaterThan(color.r);
  expect(color.g).toBeGreaterThan(color.r);
  expect(spread).toBeGreaterThanOrEqual(8);
}

function luminance(hex: string): number {
  const channel = (value: number) => {
    const normalized = value / 255;
    return normalized <= 0.03928
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  };

  const color = rgb(hex);
  return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
}

function contrastRatio(foreground: string | null, background: string | null): number {
  expect(foreground).not.toBeNull();
  expect(background).not.toBeNull();
  const lighter = Math.max(luminance(foreground ?? ""), luminance(background ?? ""));
  const darker = Math.min(luminance(foreground ?? ""), luminance(background ?? ""));
  return (lighter + 0.05) / (darker + 0.05);
}

describe("dark theme brand tokens", () => {
  it("uses a deep logo-ink scale instead of orange or flat graphite gray", () => {
    expect(rootToken("--bg-primary")).toBe("#11131A");
    expect(rootToken("--bg-secondary")).toBe("#171A23");
    expect(rootToken("--border")).toBe("#2B3140");

    expect(darkToken("--mem-bg")).toBe("#11131A");
    expect(darkToken("--mem-surface")).toBe("#171A23");
    expect(darkToken("--mem-sidebar")).toBe("#0E1017");
    expect(darkToken("--mem-border")).toBe("#2B3140");
    expect(darkToken("--mem-brand-text")).toBe("#ECEAF3");
  });

  it("keeps dark structural surfaces logo-tinted ink, not neutral gray", () => {
    for (const token of [
      rootToken("--bg-primary"),
      rootToken("--bg-secondary"),
      darkToken("--mem-bg"),
      darkToken("--mem-surface"),
      darkToken("--mem-sidebar"),
    ]) {
      expectLogoInk(token);
    }
  });

  it("moves page emphasis to logo-cyan while keeping warm states muted", () => {
    expect(darkToken("--mem-accent-page")).toBe("#8FB3EA");
    expect(darkToken("--mem-accent-indigo")).toBe("#A9AEF2");
    expect(darkToken("--mem-accent-warm")).toBe("#BB8967");

    const page = rgb(darkToken("--mem-accent-page") ?? "");
    expect(page.g).toBeGreaterThan(page.r);
    expect(page.b).toBeGreaterThan(page.r);
    expect(page.b - page.r).toBeGreaterThanOrEqual(35);

    const warm = rgb(darkToken("--mem-accent-warm") ?? "");
    expect(warm.r - warm.g).toBeLessThanOrEqual(50);
    expect(warm.g - warm.b).toBeLessThanOrEqual(35);
  });

  it("keeps wiki page icons quieter than page emphasis colors", () => {
    expect(darkToken("--mem-page-icon")).toBe("#C19A7B");
    expect(darkToken("--mem-page-icon-hover")).toBe("#D1AF95");

    const icon = rgb(darkToken("--mem-page-icon") ?? "");
    const accent = rgb(darkToken("--mem-accent-page") ?? "");
    const warm = rgb(darkToken("--mem-accent-warm") ?? "");

    expect(icon.r).toBeGreaterThan(icon.g);
    expect(icon.g).toBeGreaterThan(icon.b);
    expect(warm.r - warm.g).toBeGreaterThan(icon.r - icon.g);
    expect(accent.b - accent.r).toBeGreaterThan(icon.b - icon.r);
  });
});

describe("light theme brand tokens", () => {
  it("keeps the light workspace slightly warm with a lighter sidebar", () => {
    expect(lightToken("--bg-primary")).toBe("#FCFCFB");
    expect(lightToken("--bg-secondary")).toBe("#FFFFFF");
    expect(lightToken("--bg-tertiary")).toBe("#FFFFFF");
    expect(lightToken("--border")).toBe("#E3E7EE");

    expect(lightToken("--mem-bg")).toBe("#FCFCFB");
    expect(lightToken("--mem-surface")).toBe("#FFFFFF");
    expect(lightToken("--mem-sidebar")).toBe("#FFFFFF");
    expect(lightToken("--mem-account-card")).toBe("#FCFCFB");
    expect(lightToken("--mem-popover")).toBe("#FFFFFF");
    expect(lightToken("--mem-border")).toBe("#E3E7EE");
    expect(lightToken("--mem-brand-text")).toBe("#1A1A2E");
    expect(lightToken("--mem-accent-page")).toBe("#5E58C8");
    expect(lightToken("--mem-accent-warm")).toBe("#B46A3A");
    expect(lightToken("--mem-page-icon")).toBe("#B46A3A");
    expect(lightToken("--mem-page-icon-hover")).toBe("#9F5C32");
  });

  it("keeps light structural surfaces out of the lavender family", () => {
    for (const token of [
      lightToken("--bg-primary"),
      lightToken("--bg-tertiary"),
      lightToken("--mem-bg"),
      lightToken("--mem-sidebar"),
    ]) {
      expect(token).not.toBeNull();
      const color = rgb(token ?? "");
      expect(color.g).toBeGreaterThanOrEqual(color.b);
      expect(color.r).toBeGreaterThanOrEqual(color.b);
    }
  });
});

describe("reading contrast", () => {
  it("keeps body and sidebar text readable in both themes", () => {
    for (const token of [darkToken, lightToken]) {
      expect(contrastRatio(token("--mem-text"), token("--mem-bg"))).toBeGreaterThan(12);
      expect(contrastRatio(token("--mem-text-secondary"), token("--mem-bg"))).toBeGreaterThan(4.5);
      expect(contrastRatio(token("--mem-text-tertiary"), token("--mem-bg"))).toBeGreaterThan(2.8);
      expect(contrastRatio(token("--mem-text-secondary"), token("--mem-sidebar"))).toBeGreaterThan(4.5);
    }
  });
});
