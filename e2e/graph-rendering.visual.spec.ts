// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "@playwright/test";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

type CanvasEvidence = {
  coloredPixels: number;
  orangeCoverage: number;
  sampledPixels: number;
  uniqueColors: number;
};

test("renders Graph as a structured canvas instead of a flat orange field", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, {
    locale: "en",
    localStorage: { "wenlan-theme": "light" },
    rawActions: [],
  });
  await page.goto("/");

  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Graph", exact: true })
    .click();

  const graph = page.getByTestId("atlas-view");
  await expect(graph).toBeVisible();
  // Pages lead the line now, and the counts are over what is actually drawn:
  // seven wiki pages plus the three entities that have a connection.
  await expect(page.getByText(/^7 pages · 3 entities(?: · \d+ regions?)?$/)).toBeVisible();

  // The place-name overlay: region names in a muted ink with a ground-
  // coloured halo, drawn above the nodes. It is the ONLY 2D canvas on the
  // map — nothing is painted under the nodes (no terrain, wash or hull, so
  // no shadow or aura around a point) — and it is what can be read back; the
  // WebGL node layer is covered by the screenshot below.
  await expect(graph.locator("canvas[data-testid]")).toHaveCount(1);
  const canvas = graph.locator('canvas[data-testid="atlas-region-names"]');
  await expect(canvas).toHaveCount(1);
  await expect(canvas).toBeVisible();
  await expect(canvas).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");

  const readOverlay = (): Promise<CanvasEvidence> =>
    canvas.evaluate((node): CanvasEvidence => {
      if (!(node instanceof HTMLCanvasElement)) {
        return { coloredPixels: 0, orangeCoverage: 1, sampledPixels: 0, uniqueColors: 0 };
      }
      const context = node.getContext("2d", { willReadFrequently: true });
      if (!context) {
        return { coloredPixels: 0, orangeCoverage: 1, sampledPixels: 0, uniqueColors: 0 };
      }
      const pixels = context.getImageData(0, 0, node.width, node.height).data;
      const colors = new Set<string>();
      let coloredPixels = 0;
      let orangePixels = 0;
      let sampledPixels = 0;
      for (let y = 0; y < node.height; y += 2) {
        for (let x = 0; x < node.width; x += 2) {
          sampledPixels += 1;
          const offset = (y * node.width + x) * 4;
          const red = pixels[offset] ?? 0;
          const green = pixels[offset + 1] ?? 0;
          const blue = pixels[offset + 2] ?? 0;
          const alpha = pixels[offset + 3] ?? 0;
          if (alpha === 0) continue;
          coloredPixels += 1;
          // Raw alpha in the key: anti-aliased text shows many alpha steps
          // where a flat fill shows one.
          colors.add(`${red >> 4}:${green >> 4}:${blue >> 4}:${alpha}`);
          if (red > 170 && green > 55 && green < 175 && blue < 100) {
            orangePixels += 1;
          }
        }
      }
      return {
        coloredPixels,
        orangeCoverage: sampledPixels === 0 ? 1 : orangePixels / sampledPixels,
        sampledPixels,
        uniqueColors: colors.size,
      };
    });

  await expect(page).toHaveScreenshot("graph-1280x900-light.png", {
    animations: "disabled",
    fullPage: false,
    maxDiffPixelRatio: 0.002,
  });

  // At the default fit the fixture's one named region earns its place name:
  // some text pixels, anti-aliased through many alphas, none of them orange.
  // Text is all this canvas carries, so a painted wash would show up here as
  // a flood of colored pixels far beyond what a name can account for.
  let evidence: CanvasEvidence = { coloredPixels: 0, orangeCoverage: 1, sampledPixels: 0, uniqueColors: 0 };
  await expect
    .poll(
      async () => {
        evidence = await readOverlay();
        return evidence.coloredPixels;
      },
      { timeout: 10_000 },
    )
    .toBeGreaterThan(25);
  expect(evidence.sampledPixels).toBeGreaterThan(0);
  expect(evidence.coloredPixels / evidence.sampledPixels).toBeLessThan(0.02);
  expect(evidence.uniqueColors).toBeGreaterThan(8);
  expect(evidence.orangeCoverage).toBeLessThan(0.25);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
