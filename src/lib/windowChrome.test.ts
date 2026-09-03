// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";

import * as windowChrome from "./windowChrome";
import {
  CSS_WINDOW_SHADOW_INSET,
  MACOS_TRAFFIC_LIGHT_INSET,
  NATIVE_TITLE_BAR_INSET,
  cssWindowShadowInset,
  dragStripHeight,
  hostPlatform,
  needsCssWindowShadow,
  topBarLeftInset,
} from "./windowChrome";

// Real user agents from the three webviews the app ships against.
const MACOS_WKWEBVIEW =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15";
const WINDOWS_WEBVIEW2 =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0";
const LINUX_WEBKITGTK =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15";

// A WebView whose UA the app or an embedder has customised past recognition.
// This is not a hypothetical: it is what a real Mac looks like to this module
// once anything rewrites the UA string, and it is the case every one of the
// three questions below used to answer as "not macOS".
const UNRECOGNISED_UA = "Wenlan/1.0";

/** Runs `body` with `globalThis.navigator` replaced, then puts it back. */
function withNavigator(replacement: { userAgent: string } | undefined, body: () => void): void {
  const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    value: replacement,
    configurable: true,
  });
  try {
    body();
  } finally {
    if (original) {
      Object.defineProperty(globalThis, "navigator", original);
    } else {
      Reflect.deleteProperty(globalThis, "navigator");
    }
  }
}

describe("window chrome", () => {
  // The three states, and the two that used to be one. "Windows" is a
  // measurement that says not-macOS; a missing or unplaceable user agent is
  // not a measurement at all.
  it("tells an unrecognised platform apart from a measured non-macOS one", () => {
    expect(hostPlatform(MACOS_WKWEBVIEW)).toBe("macos");
    expect(hostPlatform(WINDOWS_WEBVIEW2)).toBe("other");
    expect(hostPlatform(LINUX_WEBKITGTK)).toBe("other");

    expect(hostPlatform("")).toBe("unknown");
    expect(hostPlatform(UNRECOGNISED_UA)).toBe("unknown");
    // The real missing-user-agent path is no navigator at all. Passing
    // `undefined` explicitly would re-trigger the default parameter and read
    // the ambient user agent instead, so it would pass for the wrong reason.
    withNavigator(undefined, () => {
      expect(hostPlatform()).toBe("unknown");
    });

    expect(hostPlatform(UNRECOGNISED_UA)).not.toBe(hostPlatform(WINDOWS_WEBVIEW2));
  });

  // "Mac" alone appears in unrelated tokens; only the platform strings count.
  it("does not match a stray Mac substring", () => {
    expect(hostPlatform("Mozilla/5.0 (Windows NT 10.0) MacGyver/1.0")).toBe("other");
    expect(hostPlatform("MacGyver/1.0")).toBe("unknown");
  });

  it("measures the two platforms it can place", () => {
    expect(topBarLeftInset(MACOS_WKWEBVIEW)).toBe(82);
    expect(topBarLeftInset(WINDOWS_WEBVIEW2)).toBe(20);
    expect(topBarLeftInset(LINUX_WEBKITGTK)).toBe(20);

    expect(dragStripHeight(MACOS_WKWEBVIEW)).toBe(32);
    expect(dragStripHeight(WINDOWS_WEBVIEW2)).toBe(0);
    expect(dragStripHeight(LINUX_WEBKITGTK)).toBe(0);

    expect(needsCssWindowShadow(MACOS_WKWEBVIEW)).toBe(true);
    expect(needsCssWindowShadow(WINDOWS_WEBVIEW2)).toBe(false);
    expect(needsCssWindowShadow(LINUX_WEBKITGTK)).toBe(false);
  });

  // How the components actually call these: no argument, so the values come
  // from the webview the app is running in.
  it("reads the ambient user agent when called with no argument", () => {
    withNavigator({ userAgent: MACOS_WKWEBVIEW }, () => {
      expect(topBarLeftInset()).toBe(82);
      expect(dragStripHeight()).toBe(32);
      expect(needsCssWindowShadow()).toBe(true);
    });

    withNavigator({ userAgent: WINDOWS_WEBVIEW2 }, () => {
      expect(topBarLeftInset()).toBe(20);
      expect(dragStripHeight()).toBe(0);
      expect(needsCssWindowShadow()).toBe(false);
    });
  });

  // ---------------------------------------------------------------------
  // The three unknown-platform DECISIONS. Each is pinned to a literal, so an
  // edit that flips one fails here rather than shipping. They deliberately do
  // not all take the same side; the last assertion in this block is what makes
  // "three separate decisions" rather than "one shared boolean" testable.
  // ---------------------------------------------------------------------

  it("DECISION: an unmeasured platform gets the macOS top-bar inset, not the native one", () => {
    // Guessing "not macOS" on a real Mac paints the sidebar toggle under the
    // traffic lights, which sit on top of the webview and take the click --
    // aiming at the toggle presses Close. Guessing "macOS" off macOS indents a
    // full-width header by 62px. Destructive loses to cosmetic.
    expect(topBarLeftInset(UNRECOGNISED_UA)).toBe(82);
    expect(topBarLeftInset("")).toBe(82);
    withNavigator(undefined, () => {
      expect(topBarLeftInset()).toBe(82);
    });

    // The defect this replaces: `unknown` and a measured "Windows" produced the
    // same number, so the failed measurement was invisible.
    expect(topBarLeftInset(UNRECOGNISED_UA)).not.toBe(topBarLeftInset(WINDOWS_WEBVIEW2));
    expect(topBarLeftInset(UNRECOGNISED_UA)).toBe(MACOS_TRAFFIC_LIGHT_INSET);
    expect(topBarLeftInset(UNRECOGNISED_UA)).not.toBe(NATIVE_TITLE_BAR_INSET);
  });

  it("DECISION: an unmeasured platform gets the macOS drag strip, not a zero one", () => {
    // Guessing "not macOS" on a real Mac leaves the setup wizard -- no top bar
    // of its own, `titleBarStyle: "Overlay"` so no native bar either -- with no
    // drag target at all. The window cannot be moved. Guessing "macOS" off
    // macOS costs 32px of dead space above the content. Unmovable loses.
    expect(dragStripHeight(UNRECOGNISED_UA)).toBe(32);
    expect(dragStripHeight("")).toBe(32);
    withNavigator(undefined, () => {
      expect(dragStripHeight()).toBe(32);
    });

    expect(dragStripHeight(UNRECOGNISED_UA)).not.toBe(dragStripHeight(WINDOWS_WEBVIEW2));
    expect(dragStripHeight(UNRECOGNISED_UA)).toBe(dragStripHeight(MACOS_WKWEBVIEW));
  });

  it("DECISION: an unmeasured platform paints no CSS window shadow", () => {
    // The one question whose asymmetry runs the other way: both mistakes are
    // cosmetic, so the tie goes to the one that cannot render as an artifact. A
    // missing shadow looks plain; a shadow the compositor cannot blend is the
    // flat grey band that was actually reported.
    expect(needsCssWindowShadow(UNRECOGNISED_UA)).toBe(false);
    expect(needsCssWindowShadow("")).toBe(false);
    withNavigator(undefined, () => {
      expect(needsCssWindowShadow()).toBe(false);
    });

    expect(cssWindowShadowInset(UNRECOGNISED_UA)).toBe(0);
    expect(cssWindowShadowInset(MACOS_WKWEBVIEW)).toBe(CSS_WINDOW_SHADOW_INSET);
    expect(cssWindowShadowInset(MACOS_WKWEBVIEW)).toBe(12);
  });

  it("makes the three decisions separately -- they do not all take the macOS side", () => {
    // If a future edit reunifies these behind one boolean, whichever side that
    // boolean picks, this fails: geometry sides with macOS and the shadow does
    // not, so no single answer satisfies all three.
    expect(topBarLeftInset(UNRECOGNISED_UA)).toBe(topBarLeftInset(MACOS_WKWEBVIEW));
    expect(dragStripHeight(UNRECOGNISED_UA)).toBe(dragStripHeight(MACOS_WKWEBVIEW));
    expect(needsCssWindowShadow(UNRECOGNISED_UA)).not.toBe(needsCssWindowShadow(MACOS_WKWEBVIEW));
  });

  // `isMacOS` was the shared boolean all three questions used to route through,
  // and it collapsed `unknown` to `false` for every one of them without saying
  // so. Removing it is the fix; this keeps it removed, because a re-added
  // helper with that name is what a future caller would reach for.
  it("exports no boolean isMacOS for a caller to collapse unknown through", () => {
    expect(Object.keys(windowChrome)).not.toContain("isMacOS");
    expect(Object.keys(windowChrome)).toContain("hostPlatform");
  });
});
