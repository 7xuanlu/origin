// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { buildSpine, displayLabel, layoutMap, nodeBoxSize } from "./tree";
import { radialPolar, type MapNodeInput } from "./radial";
import type { PageMapNode } from "../tauri";

function n(o: Partial<PageMapNode> & { id: string }): PageMapNode {
  return {
    id: o.id,
    parent_id: o.parent_id ?? null,
    rank: o.rank ?? 0,
    ref_kind: o.ref_kind ?? "memory",
    ref_id: o.ref_id ?? o.id,
    label: o.label ?? null,
    status: o.status ?? "active",
    pinned: o.pinned ?? false,
    placed: o.placed ?? false,
    collapsed: o.collapsed ?? false,
    x: o.x ?? null,
    y: o.y ?? null,
    width: o.width ?? null,
    height: o.height ?? null,
    ref_state: o.ref_state ?? "live",
  };
}

const anyLabel = () => "FB";
const childIds = (node: MapNodeInput) => (node.children ?? []).map((c) => c.id);
const allIds = (node: MapNodeInput): string[] => [
  node.id,
  ...(node.children ?? []).flatMap(allIds),
];

describe("buildSpine", () => {
  it("drops dismissed nodes before the walk", () => {
    const spine = buildSpine(
      [
        n({ id: "root" }),
        n({ id: "a", parent_id: "root", rank: 0 }),
        n({ id: "b", parent_id: "root", rank: 1, status: "dismissed" }),
      ],
      anyLabel,
    );
    expect(childIds(spine!.root)).toEqual(["a"]);
    expect(spine!.byId.has("b")).toBe(false);
  });

  it("orders siblings by rank, breaking ties by id", () => {
    // "z" is inserted before "m" on purpose: with a stable sort, insertion
    // order survives unless the id tiebreak actually runs.
    const spine = buildSpine(
      [
        n({ id: "root" }),
        n({ id: "c", parent_id: "root", rank: 2 }),
        n({ id: "a", parent_id: "root", rank: 1 }),
        n({ id: "z", parent_id: "root", rank: 0 }),
        n({ id: "m", parent_id: "root", rank: 0 }),
      ],
      anyLabel,
    );
    expect(childIds(spine!.root)).toEqual(["m", "z", "a", "c"]);
  });

  it("skips a node whose parent is absent from the payload", () => {
    const spine = buildSpine(
      [
        n({ id: "root" }),
        n({ id: "a", parent_id: "root" }),
        n({ id: "ghost", parent_id: "not_in_payload" }),
      ],
      anyLabel,
    );
    expect(childIds(spine!.root)).toEqual(["a"]);
    expect(allIds(spine!.root)).not.toContain("ghost");
  });

  it("terminates on a 2-cycle among non-root nodes and skips them", () => {
    const spine = buildSpine(
      [
        n({ id: "root" }),
        n({ id: "a", parent_id: "b" }),
        n({ id: "b", parent_id: "a" }),
      ],
      anyLabel,
    );
    expect(allIds(spine!.root)).toEqual(["root"]);
  });

  it("terminates when a duplicated id makes a node its own ancestor", () => {
    // parent_id is single-valued, so the only way a cycle becomes *reachable*
    // from the root is a duplicated id. This is the case the `seen` guard
    // exists for: without it the walk recurses forever.
    const nodes = [
      n({ id: "root" }),
      n({ id: "a", parent_id: "root" }),
      n({ id: "a", parent_id: "a", rank: 1 }),
    ];
    expect(() => buildSpine(nodes, anyLabel)).not.toThrow();
    expect(allIds(buildSpine(nodes, anyLabel)!.root)).toEqual(["root", "a"]);
  });

  it("returns null when the payload has no root", () => {
    expect(
      buildSpine(
        [n({ id: "a", parent_id: "b" }), n({ id: "b", parent_id: "a" })],
        anyLabel,
      ),
    ).toBeNull();
  });
});

describe("displayLabel", () => {
  it("prefers the map label, then the resolved override, then the fallback", () => {
    const overrides = new Map([["memory:m1", "Resolved title"]]);
    const at = (label: string | null, refId = "m1") =>
      displayLabel(
        n({ id: "x", ref_kind: "memory", ref_id: refId, label }),
        overrides,
        "FB",
      );

    expect(at("Map label")).toBe("Map label");
    expect(at("  Map label  ")).toBe("Map label");
    expect(at(null)).toBe("Resolved title");
    // whitespace-only label must fall through to the override, not render blank
    expect(at("   ")).toBe("Resolved title");
    expect(at(null, "m_unknown")).toBe("FB");
  });
});

describe("layoutMap", () => {
  it("uses stored coordinates for placed nodes and the layout slot otherwise", () => {
    const nodes = [
      n({ id: "root", ref_kind: "page", ref_id: "p1" }),
      n({ id: "pinned", parent_id: "root", rank: 0, placed: true, x: 900, y: -800 }),
      // stale coordinates that must be ignored while placed is false
      n({ id: "floating", parent_id: "root", rank: 1, placed: false, x: 500, y: 500 }),
    ];
    const views = layoutMap(nodes, new Map(), "FB");
    const pinned = views.find((v) => v.node.id === "pinned")!;
    const floating = views.find((v) => v.node.id === "floating")!;
    const slots = radialPolar(buildSpine(nodes, anyLabel)!.root);
    const floatingSlot = slots.find((s) => s.id === "floating")!;

    expect(pinned.x).toBe(900);
    expect(pinned.y).toBe(-800);
    expect(floating.x).not.toBe(500);
    expect(floating.y).not.toBe(500);
    expect(floating.x).toBeCloseTo(floatingSlot.x, 6);
    expect(floating.y).toBeCloseTo(floatingSlot.y, 6);
  });

  it("leaves a suggestion room for its accept and dismiss buttons", () => {
    // Sized from the label alone, the two buttons took their width out of the
    // text: "Vector search" rendered as "Vector …" on the one kind of box whose
    // whole job is to be read and judged.
    const label = "Vector search";
    const nodes = [
      n({ id: "root", ref_kind: "page", ref_id: "p1" }),
      n({ id: "live", parent_id: "root", rank: 0, ref_kind: "section", label }),
      n({
        id: "guess",
        parent_id: "root",
        rank: 1,
        ref_kind: "section",
        label,
        status: "suggested",
      }),
    ];
    const views = layoutMap(nodes, new Map(), "FB");
    const width = (id: string) => views.find((v) => v.node.id === id)!.width;

    expect(width("guess")).toBeGreaterThan(width("live"));
    expect(width("live")).toBe(nodeBoxSize(label).width);
  });
});
