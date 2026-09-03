// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import QuickCapture from "./QuickCapture";

// Real user agents from the two webviews that matter here. The quick-capture
// window is transparent and undecorated on both, but only macOS turns the
// native window shadow off (app/src/lib.rs, `setHasShadow: NO`), so only macOS
// has to paint one in CSS.
const MACOS_WKWEBVIEW =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15";
const WINDOWS_WEBVIEW2 =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0";
const LINUX_WEBKITGTK =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15";

const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");

/** Replace the ambient user agent, the way `needsCssWindowShadow` reads it. */
function setPlatform(userAgent: string | undefined): void {
  Object.defineProperty(globalThis, "navigator", {
    value: userAgent === undefined ? undefined : { userAgent },
    configurable: true,
  });
}

afterEach(() => {
  if (originalNavigator) {
    Object.defineProperty(globalThis, "navigator", originalNavigator);
  } else {
    Reflect.deleteProperty(globalThis, "navigator");
  }
});

/**
 * The card, found by IDENTITY rather than by presentation.
 *
 * This used to be `container.querySelector(".rounded-xl")`, and that is not a
 * selection, it is a coincidence that keeps working. `querySelector` returns
 * the FIRST match and never says how many there were, so a second element
 * picking up the same utility class -- a Tailwind class, present for its corner
 * radius and nothing else -- silently redirects every assertion below onto the
 * wrong node. And the assertions are all `expect(...).toBe("none")`: an element
 * with no inline shadow at all satisfies them for the wrong reason, so the test
 * would keep passing while the card it was written about lost its shadow. A
 * control that cannot fail is worse than no control.
 *
 * Preferred hook: `data-testid="quick-capture-card"`. It does NOT exist yet --
 * `src/components/QuickCapture.tsx` is owned by another agent in this
 * workstream and is not mine to edit, so adding it is a REQUIRED FOLLOW-UP and
 * this helper is written to use it the moment it lands.
 *
 * Until then the identity used is SEMANTIC and asserted UNIQUE: the card is the
 * one element whose inline style carries BOTH a `border-color` and a
 * `box-shadow`. That pair is exactly the decision under test -- `borderAccent`
 * plus the platform-scoped `boxShadow` -- and it distinguishes the card from
 * the Save button, which sets an inline `box-shadow` but no `border-color`.
 * `toHaveLength(1)` is what makes it able to fail: if the component changes so
 * that two elements match, or none does, this throws instead of quietly
 * measuring something else.
 */
function findCard(container: HTMLElement): HTMLElement {
  const tagged = container.querySelectorAll('[data-testid="quick-capture-card"]');
  if (tagged.length > 0) {
    expect(tagged).toHaveLength(1);
    return tagged[0] as HTMLElement;
  }
  const byIdentity = Array.from(container.querySelectorAll<HTMLElement>("[style]")).filter(
    (el) => el.style.borderColor !== "" && el.style.boxShadow !== "",
  );
  // Exactly one, or the identity this test relies on no longer holds and the
  // right outcome is a loud failure rather than a silent redirect.
  expect(byIdentity).toHaveLength(1);
  return byIdentity[0];
}

/**
 * Renders and hands back the two elements the platform scoping touches: the
 * standalone window's outer wrapper (which owns the transparent inset) and the
 * card itself (which owns the shadow).
 */
function renderCapture(standalone: boolean) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const { container } = render(
    <QueryClientProvider client={client}>
      <QuickCapture isOpen onClose={() => {}} standalone={standalone} />
    </QueryClientProvider>,
  );
  const wrapper = container.firstElementChild as HTMLElement;
  expect(wrapper).toBeTruthy();
  const card = findCard(container);
  // The card is INSIDE the wrapper whose inset is asserted alongside it. Both
  // halves of every test below are then known to be about one window rather
  // than about two elements that happen to be in the same container.
  expect(wrapper.contains(card)).toBe(true);
  expect(wrapper).not.toBe(card);
  return { wrapper, card };
}

/** Inset in px, whichever way React and jsdom chose to serialise a zero. */
function insetPx(el: HTMLElement): number {
  return parseInt(el.style.padding || "0", 10);
}

describe("QuickCapture standalone window shadow is platform-scoped", () => {
  it("keeps the CSS shadow on macOS, where the NSWindow draws none", () => {
    setPlatform(MACOS_WKWEBVIEW);
    const { wrapper, card } = renderCapture(true);

    // app/src/lib.rs sets setHasShadow:NO on this NSWindow, so removing the CSS
    // shadow would flatten the card against the desktop behind it.
    expect(card.style.boxShadow).not.toBe("none");
    expect(card.style.boxShadow).toContain("32px");
    // And the shadow needs somewhere to render: an outer shadow is clipped at
    // the viewport edge without a transparent margin.
    expect(insetPx(wrapper)).toBe(12);
  });

  it("drops the shadow and the inset on a Windows layered window", () => {
    setPlatform(WINDOWS_WEBVIEW2);
    const { wrapper, card } = renderCapture(true);

    // The reported bug: this shadow composites as a flat grey band, not a halo
    // that fades into the desktop.
    expect(card.style.boxShadow).toBe("none");
    expect(insetPx(wrapper)).toBe(0);
  });

  it("treats Linux like Windows -- layered windows composite the same way", () => {
    setPlatform(LINUX_WEBKITGTK);
    const { wrapper, card } = renderCapture(true);

    expect(card.style.boxShadow).toBe("none");
    expect(insetPx(wrapper)).toBe(0);
  });

  it("defaults an unknown platform to no shadow, which cannot look broken", () => {
    // No navigator at all is the real first-paint / unknown-host case. A card
    // missing its shadow reads as plain; a shadow the compositor cannot blend
    // reads as a rendering artifact, so the unknown side takes the former.
    setPlatform(undefined);
    const { wrapper, card } = renderCapture(true);

    expect(card.style.boxShadow).toBe("none");
    expect(insetPx(wrapper)).toBe(0);
  });

  it("leaves the modal's shadow alone on every platform", () => {
    // The modal floats over the app on an opaque backdrop with room around it.
    // Nothing about the host window applies, so it must not be scoped at all.
    for (const ua of [MACOS_WKWEBVIEW, WINDOWS_WEBVIEW2, LINUX_WEBKITGTK]) {
      setPlatform(ua);
      const { card } = renderCapture(false);
      expect(card.style.boxShadow).not.toBe("none");
      expect(card.style.boxShadow).toContain("32px");
    }
  });
});
