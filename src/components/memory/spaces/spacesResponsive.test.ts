import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("Spaces overview responsive contract", () => {
  it("uses the final inventory cascade for compact row geometry without collapsing rename", () => {
    const component = readFileSync(resolve("src/components/memory/spaces/SpacesOverview.tsx"), "utf8");
    const baseImport = component.indexOf('import "./spaces.css"');
    const inventoryImport = component.indexOf('import "./spacesInventory.css"');
    const inventoryCss = readFileSync(resolve("src/components/memory/spaces/spacesInventory.css"), "utf8");
    const narrowRules = inventoryCss.slice(inventoryCss.indexOf("@media (max-width: 699px)"));

    expect(baseImport).toBeGreaterThanOrEqual(0);
    expect(inventoryImport).toBeGreaterThan(baseImport);
    expect(narrowRules).toMatch(
      /\.spaces-overview \.spaces-row:not\(\.spaces-row-suggested\):not\(\.spaces-row-edit\)\s*\{[^}]*grid-template-columns:\s*28px minmax\(0, 1fr\) 36px;/s,
    );
    expect(inventoryCss).toMatch(
      /\.spaces-overview \.spaces-row-edit\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s,
    );
    expect(inventoryCss).toMatch(
      /\.spaces-row-edit\s*>\s*\.spaces-editor\s*\{[^}]*grid-column:\s*1 \/ -1;[^}]*width:\s*100%/s,
    );
  });
});
