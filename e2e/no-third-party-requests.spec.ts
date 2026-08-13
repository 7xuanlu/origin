// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "@playwright/test";
import { installTauriMock } from "./tauriMock";

// The app is local-first, and it should not reach the public internet just to
// paint. index.html used to link Google's font stylesheet directly, which meant
// every launch told Google someone had opened Wenlan, the window rendered with
// fallback faces offline, and the end-to-end suite failed whenever a font 404'd
// — on a different test each run, because any page load could be the unlucky
// one. The fonts now live in public/fonts.
//
// This is the gate that keeps it that way. It fails on the next external
// stylesheet, script, image, or beacon someone adds, whether or not the network
// happens to be up when CI runs.
test("loads the whole app without a single request leaving the origin", async ({ page, baseURL }) => {
  const origin = new URL(baseURL ?? "http://127.0.0.1").origin;
  const offOrigin: string[] = [];
  page.on("request", (request) => {
    const url = request.url();
    // data:, blob: and about: never leave the process.
    if (/^(data|blob|about):/.test(url)) {
      return;
    }
    if (!url.startsWith(origin)) {
      offOrigin.push(`${request.resourceType()} ${url}`);
    }
  });

  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await expect(page.locator("main")).toBeVisible();
  // Fonts are the reason this test exists, so wait for them specifically rather
  // than trusting that first paint already asked for everything.
  await page.evaluate(async () => {
    await document.fonts.ready;
  });

  // A couple of surfaces beyond the first paint, so a lazily loaded chunk or an
  // image that only one view asks for is covered too.
  const sidebar = page.getByRole("navigation", { name: "Primary navigation" });
  for (const destination of ["Wiki", "Spaces"]) {
    await sidebar.getByRole("button", { name: destination, exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: destination })).toBeVisible();
  }

  expect(offOrigin).toEqual([]);
});
