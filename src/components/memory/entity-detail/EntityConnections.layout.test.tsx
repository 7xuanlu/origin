// SPDX-License-Identifier: AGPL-3.0-only
import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../../i18n";
import type { RelationWithEntity } from "../../../lib/tauri";
import { EntityConnections } from "./EntityConnections";

const relations: readonly RelationWithEntity[] = [
  {
    id: "relation-babbage",
    relation_type: "collaborated with",
    direction: "outgoing",
    entity_id: "entity-babbage",
    entity_name: "Charles Babbage",
    entity_type: "person",
    source_agent: "fixture",
    created_at: 1_700_000_200,
  },
  {
    id: "relation-hopper",
    relation_type: "inspired",
    direction: "incoming",
    entity_id: "entity-hopper",
    entity_name: "Grace Hopper",
    entity_type: "person",
    source_agent: "fixture",
    created_at: 1_700_000_300,
  },
  {
    id: "relation-turing",
    relation_type: "influenced",
    direction: "outgoing",
    entity_id: "entity-turing",
    entity_name: "Alan Turing",
    entity_type: "person",
    source_agent: "fixture",
    created_at: 1_700_000_400,
  },
];

function graphTops(): readonly number[] {
  const graph = screen.getByRole("group", { name: "Connection map for Ada Lovelace" });
  return within(graph)
    .getAllByRole("button")
    .map((button) => Number.parseFloat(button.style.top));
}

function graphLefts(): readonly number[] {
  const graph = screen.getByRole("group", { name: "Connection map for Ada Lovelace" });
  return within(graph)
    .getAllByRole("button")
    .map((button) => Number.parseFloat(button.style.left));
}

function graphVerbPositions(): readonly { readonly left: number; readonly top: number }[] {
  const graph = screen.getByRole("group", { name: "Connection map for Ada Lovelace" });
  return Array.from(graph.querySelectorAll<HTMLElement>(".entity-graph-verb")).map((verb) => ({
    left: Number.parseFloat(verb.style.left),
    top: Number.parseFloat(verb.style.top),
  }));
}

describe("EntityConnections graph layout", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps one, two, and three neighbors on deterministic readable lanes", () => {
    const { rerender } = render(
      <EntityConnections name="Ada Lovelace" relations={relations.slice(0, 1)} onEntityClick={() => {}} />,
    );
    expect(graphTops()).toEqual([50]);

    rerender(
      <EntityConnections name="Ada Lovelace" relations={relations.slice(0, 2)} onEntityClick={() => {}} />,
    );
    const twoTops = graphTops();
    expect(twoTops).toHaveLength(2);
    expect(new Set(twoTops).size).toBe(2);
    expect(Math.abs(twoTops[0] - twoTops[1])).toBeGreaterThanOrEqual(24);
    expect(graphLefts()).toEqual([28, 72]);
    expect(graphVerbPositions()).toEqual([
      { left: 50, top: 28 },
      { left: 50, top: 72 },
    ]);
    const graph = screen.getByRole("group", { name: "Connection map for Ada Lovelace" });
    expect(
      within(graph).getByRole("button", {
        name: "Grace Hopper (person) · incoming · inspired",
      }),
    ).toBeInTheDocument();
    expect(
      within(graph).getByRole("button", {
        name: "Charles Babbage (person) · outgoing · collaborated with",
      }),
    ).toBeInTheDocument();

    rerender(
      <EntityConnections name="Ada Lovelace" relations={relations} onEntityClick={() => {}} />,
    );
    expect(new Set(graphTops()).size).toBe(3);
  });

  it("stacks two named neighbors inside the compact graph bounds", () => {
    vi.stubGlobal("matchMedia", () => ({
      matches: true,
      media: "(max-width: 640px)",
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    render(
      <EntityConnections
        name="Ada Lovelace"
        relations={relations.slice(0, 2)}
        onEntityClick={() => {}}
      />,
    );

    const graph = screen.getByRole("group", { name: "Connection map for Ada Lovelace" });
    expect(graph).toHaveClass("is-compact");
    const nodes = within(graph).getAllByRole("button");
    expect(nodes.map((node) => Number.parseFloat(node.style.left))).toEqual([50, 50]);
    expect(nodes.map((node) => Number.parseFloat(node.style.top))).toEqual([20, 80]);
    expect(graphVerbPositions()).toEqual([
      { left: 50, top: 33.5 },
      { left: 50, top: 66.5 },
    ]);
    expect(nodes[0]).toHaveAccessibleName("Grace Hopper (person) · incoming · inspired");
    expect(nodes[1]).toHaveAccessibleName(
      "Charles Babbage (person) · outgoing · collaborated with",
    );
  });
});
