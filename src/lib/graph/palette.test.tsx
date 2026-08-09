// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, waitFor, cleanup, act } from "@testing-library/react";
import {
  slotForEntityType,
  useGraphPalette,
  colorForEntityType,
  compositeOver,
  nodeFillFor,
  type GraphPalette,
} from "./palette";

const KG_TOKENS = ["project", "tool", "org", "person", "concept", "neutral", "edge", "edge-strong"];

function setToken(name: string, value: string) {
  document.documentElement.style.setProperty(name, value);
}

afterEach(() => {
  cleanup();
  document.documentElement.removeAttribute("data-theme");
  for (const token of KG_TOKENS) document.documentElement.style.removeProperty(`--kg-${token}`);
});

function Probe() {
  const palette = useGraphPalette();
  return <div data-testid="project">{palette.project}</div>;
}

describe("slotForEntityType", () => {
  it("maps the five validated daemon types to their slots", () => {
    expect(slotForEntityType("project")).toBe("project");
    expect(slotForEntityType("technology")).toBe("tool");
    expect(slotForEntityType("organization")).toBe("org");
    expect(slotForEntityType("person")).toBe("person");
    expect(slotForEntityType("concept")).toBe("concept");
  });

  it("maps place, event, and unknown types to neutral", () => {
    expect(slotForEntityType("place")).toBe("neutral");
    expect(slotForEntityType("event")).toBe("neutral");
    expect(slotForEntityType("wildcard")).toBe("neutral");
  });
});

describe("useGraphPalette", () => {
  it("reads the --kg-* tokens synchronously on mount", () => {
    setToken("--kg-project", "#111111");
    render(<Probe />);
    expect(screen.getByTestId("project")).toHaveTextContent("#111111");
  });

  it("re-reads the tokens when data-theme flips on <html>", async () => {
    setToken("--kg-project", "#111111");
    render(<Probe />);
    expect(screen.getByTestId("project")).toHaveTextContent("#111111");

    act(() => {
      setToken("--kg-project", "#222222");
      document.documentElement.setAttribute("data-theme", "light");
    });

    await waitFor(() =>
      expect(screen.getByTestId("project")).toHaveTextContent("#222222"),
    );
  });
});

describe("colorForEntityType", () => {
  it("resolves an entity type through its slot to a palette color", () => {
    const palette: GraphPalette = {
      project: "#proj",
      tool: "#tool",
      org: "#org",
      person: "#person",
      concept: "#concept",
      neutral: "#neutral",
      edge: "#edge",
      edgeStrong: "#edgeStrong",
      label: "#label",
      labelMuted: "#labelMuted",
      surface: "#surface",
      hull: "#hull",
      hullBorder: "#hullBorder",
      graticule: "#graticule",
      bridge: "#bridge",
    };
    expect(colorForEntityType("technology", palette)).toBe("#tool");
    expect(colorForEntityType("place", palette)).toBe("#neutral");
  });
});

describe("compositeOver", () => {
  it("composites a translucent fill over the background into one opaque hex", () => {
    // 0xff * 0.5 + 0x00 * 0.5 = 127.5 → 128 = 0x80.
    expect(compositeOver("#ffffff", "#000000", 0.5)).toBe("#808080");
    // Per channel: 0x40 * 0.9 + 0xf0 * 0.1 = 57.6 + 24 = 81.6 → 82 = 0x52.
    expect(compositeOver("#404040", "#f0f0f0", 0.9)).toBe("#525252");
  });

  it("passes the foreground through untouched when either color is not 6-digit hex (jsdom's empty tokens)", () => {
    expect(compositeOver("#123456", "", 0.5)).toBe("#123456");
    expect(compositeOver("blue", "#000000", 0.5)).toBe("blue");
  });
});

describe("nodeFillFor", () => {
  // Values chosen for clean math over a black surface: channel = slot * alpha.
  const palette: GraphPalette = {
    project: "#111111",
    tool: "#222222",
    org: "#333333",
    person: "#444444",
    concept: "#555555",
    neutral: "#666666",
    edge: "#777777",
    edgeStrong: "#888888",
    label: "#999999",
    labelMuted: "#aaaaaa",
    surface: "#000000",
    hull: "rgba(1,2,3,0.05)",
    hullBorder: "rgba(1,2,3,0.16)",
    graticule: "rgba(4,5,6,0.13)",
    bridge: "#bbbbbb",
  };

  it("fills confirmed entities at 0.9 alpha and everything else at 0.5", () => {
    // person #444444: 0x44 * 0.9 = 61.2 → 61 = 0x3d.
    expect(nodeFillFor("person", true, palette)).toBe("#3d3d3d");
    // 0x44 * 0.5 = 34 = 0x22 — unconfirmed and unknown (relation-derived) alike.
    expect(nodeFillFor("person", false, palette)).toBe("#222222");
    expect(nodeFillFor("person", null, palette)).toBe("#222222");
  });

  it("resolves the entity type through its palette slot before compositing", () => {
    // place → neutral #666666: 0x66 * 0.5 = 51 = 0x33.
    expect(nodeFillFor("place", null, palette)).toBe("#333333");
  });
});
