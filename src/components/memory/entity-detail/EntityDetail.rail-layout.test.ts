// SPDX-License-Identifier: AGPL-3.0-only
/// <reference types="node" />
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const stylesheet = readFileSync(
  path.resolve(process.cwd(), "src/components/memory/entity-detail/EntityDetail.css"),
  "utf8",
);

function declarationsFor(selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = stylesheet.match(new RegExp(`${escapedSelector}\\s*\\{([^}]*)\\}`));
  return match?.[1] ?? "";
}

describe("EntityDetail context rail layout", () => {
  it("stacks Details and Recall matches in one readable rail column", () => {
    const rail = declarationsFor(".entity-detail-dossier .memory-detail-rail");
    const sections = declarationsFor(
      ".entity-detail-dossier .memory-detail-rail > .memory-detail-rail-section",
    );

    expect(rail).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)/);
    expect(sections).toMatch(/grid-column:\s*1/);
    expect(sections).toMatch(/min-width:\s*0/);
    expect(sections).toMatch(/overflow-wrap:\s*normal/);
    expect(sections).toMatch(/word-break:\s*normal/);
  });

  it("keeps compact graph names and relation labels readable inside the graph", () => {
    const allVerbs = declarationsFor(".entity-detail-dossier .entity-graph-verb");
    const nodes = declarationsFor(
      ".entity-detail-dossier .entity-graph.is-compact .entity-graph-node",
    );
    const names = declarationsFor(
      ".entity-detail-dossier .entity-graph.is-compact .entity-graph-node-name",
    );
    const verbs = declarationsFor(
      ".entity-detail-dossier .entity-graph.is-compact .entity-graph-verb",
    );

    expect(nodes).toMatch(/max-width:\s*calc\(100%\s*-\s*24px\)/);
    expect(nodes).toMatch(/transform:\s*translate\(-50%,\s*-50%\)/);
    expect(names).toMatch(/white-space:\s*normal/);
    expect(names).toMatch(/text-overflow:\s*clip/);
    expect(allVerbs).toMatch(/max-width:\s*min\(28%,\s*144px\)/);
    expect(allVerbs).toMatch(/overflow-wrap:\s*anywhere/);
    expect(allVerbs).toMatch(/white-space:\s*normal/);
    expect(allVerbs).toMatch(/text-overflow:\s*clip/);
    expect(verbs).toMatch(/max-width:\s*calc\(100%\s*-\s*48px\)/);
  });
});
