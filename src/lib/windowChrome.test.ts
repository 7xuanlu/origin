// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";

import {
  MACOS_TRAFFIC_LIGHT_INSET,
  NATIVE_TITLE_BAR_INSET,
  dragStripHeight,
  isMacOS,
  topBarLeftInset,
} from "./windowChrome";

// Real user agents from the three webviews the app ships against.
const MACOS_WKWEBVIEW =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15";
const WINDOWS_WEBVIEW2 =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0";
const LINUX_WEBKITGTK =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15";

describe("window chrome", () => {
  it("recognises the macOS webview and nothing else", () => {
    expect(isMacOS(MACOS_WKWEBVIEW)).toBe(true);
    expect(isMacOS(WINDOWS_WEBVIEW2)).toBe(false);
    expect(isMacOS(LINUX_WEBKITGTK)).toBe(false);
  });

  it("treats a missing user agent as not macOS", () => {
    expect(isMacOS(undefined)).toBe(false);
    expect(isMacOS("")).toBe(false);
  });

  // "Mac" alone appears in unrelated tokens; only the platform strings count.
  it("does not match a stray Mac substring", () => {
    expect(isMacOS("Mozilla/5.0 (Windows NT 10.0) MacGyver/1.0")).toBe(false);
  });

  it("insets the top bar past the traffic lights only on macOS", () => {
    expect(topBarLeftInset(MACOS_WKWEBVIEW)).toBe(MACOS_TRAFFIC_LIGHT_INSET);
    expect(topBarLeftInset(WINDOWS_WEBVIEW2)).toBe(NATIVE_TITLE_BAR_INSET);
    expect(topBarLeftInset(LINUX_WEBKITGTK)).toBe(NATIVE_TITLE_BAR_INSET);
  });

  it("only reserves a drag strip where the title bar is overlaid", () => {
    expect(dragStripHeight(MACOS_WKWEBVIEW)).toBe(32);
    expect(dragStripHeight(WINDOWS_WEBVIEW2)).toBe(0);
  });
});
