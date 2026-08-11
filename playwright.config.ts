// SPDX-License-Identifier: AGPL-3.0-only
import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.E2E_PORT ?? 14320);
const baseURL = `http://127.0.0.1:${port}`;

// Playwright's "Desktop Chrome" descriptor carries a Windows user agent whatever
// host it runs on, so a macOS run reports Windows to the page. The app reads the
// platform off the user agent to decide whether to leave room for the macOS
// traffic lights (src/lib/windowChrome.ts), and the approved screenshots are of
// the macOS app, so the descriptor's default renders a layout no user sees.
// Patched rather than spelled out so the Chrome version keeps tracking whatever
// Playwright ships — and asserted, because a silent no-op here would put the
// wrong layout back without failing anything.
const WINDOWS_UA_TOKEN = "Windows NT 10.0; Win64; x64";
const desktopChromeUserAgent = devices["Desktop Chrome"].userAgent;
if (!desktopChromeUserAgent?.includes(WINDOWS_UA_TOKEN)) {
  throw new Error(
    `Desktop Chrome's user agent no longer contains "${WINDOWS_UA_TOKEN}": ${desktopChromeUserAgent}. ` +
      "Point the chromium project at a macOS user agent some other way before running the suite.",
  );
}
const macOSUserAgent = desktopChromeUserAgent.replace(
  WINDOWS_UA_TOKEN,
  "Macintosh; Intel Mac OS X 10_15_7",
);

export default defineConfig({
  testDir: "./e2e",
  testIgnore: "**/*.review.spec.ts",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: {
    timeout: 10_000,
  },
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL,
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `VITE_DISABLE_REACT_DEVTOOLS=1 pnpm exec vite --host 127.0.0.1 --port ${port} --strictPort`,
    url: baseURL,
    reuseExistingServer: false,
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        timezoneId: "America/Los_Angeles",
        userAgent: macOSUserAgent,
      },
    },
    // The app ships inside WKWebView, and pointer-driven canvas behaviour is
    // where WebKit and Chromium are most likely to disagree — a connector drag
    // that holds in Chromium could still drop its box in the real window. Off by
    // default because CI installs Chromium only: run it with
    // `pnpm exec playwright install webkit && E2E_WEBKIT=1 pnpm test:e2e`.
    ...(process.env.E2E_WEBKIT
      ? [
          {
            name: "webkit",
            // Glob, not a regex: a project-level regex testMatch silently
            // matched nothing here, and the run reported "No tests found",
            // which reads a lot like the suite having passed.
            testMatch: "**/page-canvas-*.spec.ts",
            use: {
              ...devices["Desktop Safari"],
              timezoneId: "America/Los_Angeles",
            },
          },
        ]
      : []),
  ],
});
