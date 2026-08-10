import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { readSourceText } from "../../../test/sourceText";

const memoryDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function stylesheet(path: string): string {
  return readSourceText(resolve(memoryDirectory, path));
}

function declaration(source: string, selector: string, property: string): string | undefined {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const body = source.match(new RegExp(`${escapedSelector}\\s*\\{(?<body>[^}]*)\\}`))?.groups?.body;
  return body?.match(new RegExp(`${property}:\\s*(?<value>[^;]+);`))?.groups?.value.trim();
}

describe("Wiki and Space editorial title scale", () => {
  it("uses the Fable-converged fixed destination scale for Wiki and Spaces", () => {
    expect(declaration(stylesheet("navigation/navigation-shell.css"), ".wiki-overview-header h1", "font-size"))
      .toBe("var(--mem-destination-title-size)");
    expect(declaration(stylesheet("spaces/spacesInventory.css"), ".spaces-overview h1", "font-size"))
      .toBe("var(--mem-destination-title-size)");
    expect(declaration(stylesheet("navigation/navigation-shell.css"), ".wiki-overview-header h1", "line-height"))
      .toBe("1.12");
    expect(declaration(stylesheet("spaces/spacesInventory.css"), ".spaces-overview h1", "line-height"))
      .toBe("1.12");
    expect(declaration(stylesheet("navigation/navigation-shell.css"), ".wiki-overview-header h1", "letter-spacing"))
      .toBe("-0.03em");
    expect(declaration(stylesheet("spaces/spacesInventory.css"), ".spaces-overview h1", "letter-spacing"))
      .toBe("-0.03em");
  });

  it("uses the Page and Entity detail scale for the Space title and its editor", () => {
    const source = stylesheet("space-detail/space-detail-header.css");
    expect(declaration(source, ".space-dossier-title-block h1,\n.space-dossier-title-input", "font-size"))
      .toBe("clamp(24px, 1.6vw + 14px, 30px)");
  });

  it("keeps Space section headings subordinate when the detail title reaches its minimum", () => {
    expect(stylesheet("space-detail/space-detail.css")).toMatch(
      /@media \(max-width: 639px\) \{[\s\S]*?\.space-dossier h2\s*\{[^}]*font-size:\s*20px;/,
    );
  });

  it("keeps the Space title and overflow action on one row at narrow widths", () => {
    expect(stylesheet("space-detail/space-detail.css")).toMatch(
      /@media \(max-width: 639px\) \{[\s\S]*?\.space-dossier-title-row\s*\{[^}]*align-items:\s*flex-start;[^}]*flex-direction:\s*row;/,
    );
  });
});
