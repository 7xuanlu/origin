// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";

export type GraphSlot =
  | "project"
  | "tool"
  | "org"
  | "person"
  | "concept"
  | "neutral";

// Daemon entity vocabulary → validated 5-slot palette. place, event, and any
// unknown type take neutral: the 5-slot set clears the dataviz validator
// (ΔE 11.6 normal / 1.6 deutan), whereas 7-slot candidates that colored
// place/event failed it. Do not extend this to 7 without re-validating.
export function slotForEntityType(entityType: string): GraphSlot {
  switch (entityType) {
    case "project":
      return "project";
    case "technology":
      return "tool";
    case "organization":
      return "org";
    case "person":
      return "person";
    case "concept":
      return "concept";
    default:
      return "neutral";
  }
}

export interface GraphPalette {
  project: string;
  tool: string;
  org: string;
  person: string;
  concept: string;
  neutral: string;
  edge: string;
  edgeStrong: string;
  /** Label ink — reads the shared --mem-text token, not a --kg-* slot. */
  label: string;
  /** Muted ink for region names (--mem-text-tertiary). */
  labelMuted: string;
  /** Graph ground (--mem-surface) — what translucent node fills composite against. */
  surface: string;
  /** Community hull fill — rgba; drawn on the 2D underlay, not sigma. */
  hull: string;
  /** Community hull border — rgba; 2D underlay. */
  hullBorder: string;
  /** Dashed orientation rings — rgba; 2D underlay. */
  graticule: string;
  /** Cross-community edge ink — opaque (sigma-consumed). */
  bridge: string;
}

// Read the resolved --kg-* custom properties off <html>. getComputedStyle
// resolves them through the cascade (index.css theme blocks), so this returns
// the values for whichever theme is currently active.
function readPalette(): GraphPalette {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string) => style.getPropertyValue(name).trim();
  return {
    project: read("--kg-project"),
    tool: read("--kg-tool"),
    org: read("--kg-org"),
    person: read("--kg-person"),
    concept: read("--kg-concept"),
    neutral: read("--kg-neutral"),
    edge: read("--kg-edge"),
    edgeStrong: read("--kg-edge-strong"),
    label: read("--mem-text"),
    labelMuted: read("--mem-text-tertiary"),
    surface: read("--mem-surface"),
    hull: read("--kg-hull"),
    hullBorder: read("--kg-hull-border"),
    graticule: read("--kg-graticule"),
    bridge: read("--kg-bridge"),
  };
}

/**
 * Graph colors as React state, re-read whenever the theme flips. The theme
 * switch stamps data-theme on <html>; a MutationObserver on that attribute
 * triggers the re-read. Never read tokens inside a paint callback — read here,
 * pass the resolved values down.
 */
export function useGraphPalette(): GraphPalette {
  const [palette, setPalette] = useState<GraphPalette>(readPalette);
  useEffect(() => {
    const observer = new MutationObserver(() => setPalette(readPalette()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);
  return palette;
}

/** Slot color for an entity type, resolved against the current palette. */
export function colorForEntityType(entityType: string, palette: GraphPalette): string {
  return palette[slotForEntityType(entityType)];
}

const HEX6 = /^#[0-9a-f]{6}$/i;

/**
 * The opaque color a fill of `fg` at `alpha` over `bg` would produce. Sigma's
 * WebGL blend treats packed colors as non-premultiplied under ONE /
 * ONE_MINUS_SRC_ALPHA, so genuinely translucent fills additive-wash toward
 * white — pre-compositing in JS is how Atlas gets translucent-LOOKING nodes.
 * Non-hex inputs (jsdom's empty computed styles) pass `fg` through untouched.
 */
export function compositeOver(fg: string, bg: string, alpha: number): string {
  if (!HEX6.test(fg) || !HEX6.test(bg)) return fg;
  const channels = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const f = channels(fg);
  const b = channels(bg);
  const mixed = f.map((v, i) =>
    Math.round(v * alpha + b[i] * (1 - alpha))
      .toString(16)
      .padStart(2, "0"),
  );
  return `#${mixed.join("")}`;
}

/**
 * Stability-tiered node fill matching the old canvas graph's translucency:
 * confirmed entities at 0.9 alpha, everything else (unconfirmed, or
 * relation-derived neighbors whose status is unknown — confirmed: null) at
 * the airy 0.5. Composited over the surface, not real alpha (see
 * compositeOver).
 */
export function nodeFillFor(
  entityType: string,
  confirmed: boolean | null,
  palette: GraphPalette,
): string {
  const alpha = confirmed === true ? 0.9 : 0.5;
  return compositeOver(colorForEntityType(entityType, palette), palette.surface, alpha);
}
