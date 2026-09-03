// SPDX-License-Identifier: AGPL-3.0-only

/**
 * Where the window's own chrome sits, and how much room the app has to leave
 * for it.
 *
 * macOS draws the traffic lights on top of the webview, because
 * app/tauri.conf.json asks for `titleBarStyle: "Overlay"` and positions them at
 * x=16. Everything the app paints in that corner would land underneath them, so
 * the top bars inset their content past it. Windows and Linux draw a real title
 * bar above the webview instead, so the same inset is just dead space.
 *
 * Read from the user agent rather than @tauri-apps/plugin-os or the daemon's
 * get_system_info: both are async, and these values are needed for the first
 * paint. The webview's user agent is the platform's own, so it reports the host
 * rather than anything the app configured.
 *
 * `hostPlatform` is three-valued, because two values were being spent on three
 * states: a UA that says "Windows" and a UA that is missing or unrecognised are
 * different measurements, and a boolean `isMacOS` answered `false` to both.
 * Every caller here still wants a number or a boolean in the end -- these values
 * decide pixel offsets at first paint, so nothing may wait on an async platform
 * lookup -- but the collapse happens at a named site *per question*, because the
 * three questions do not have the same answer for `unknown`:
 *
 *   - "how much left inset?"   -> `UNKNOWN_PLATFORM_TOP_BAR_INSET`     (macOS's)
 *   - "how tall a drag strip?" -> `UNKNOWN_PLATFORM_DRAG_STRIP_HEIGHT` (macOS's)
 *   - "who paints the shadow?" -> `UNKNOWN_PLATFORM_PAINTS_SHADOW`     (not macOS's)
 *
 * Each is a DECISION about an unmeasured platform, justified where it is
 * declared, and each is pinned by a test so flipping one fails loudly. They do
 * not agree with each other, which is the whole reason they are three sites and
 * not one shared boolean.
 *
 * There is deliberately no exported `isMacOS`. It existed; all three of these
 * questions were answered by calling it; and every one of them therefore
 * inherited `unknown -> false` invisibly -- the exact
 * failed-measurement-as-a-negative this module was split up to remove. Anything
 * new that needs the platform asks `hostPlatform()` and handles all three
 * states, or calls a function below that has already decided.
 */
export type HostPlatform = "macos" | "other" | "unknown";

export function hostPlatform(
  ua: string | undefined = globalThis.navigator?.userAgent,
): HostPlatform {
  // No navigator, or an empty UA: nothing was measured. A WebView whose UA the
  // app or the host has customised lands here too.
  if (!ua) return "unknown";
  if (/\b(Macintosh|Mac OS X)\b/.test(ua)) return "macos";
  // A UA that names one of the other two platforms the app ships to is a
  // measured "not macOS". Anything else is a UA we cannot place, and saying
  // "not macOS" about it would be the same failed-measurement-as-negative.
  if (/\b(Windows|Linux|X11|CrOS|Android)\b/.test(ua)) return "other";
  return "unknown";
}

/** Room for the overlaid traffic lights, matched to their x=16 origin. */
export const MACOS_TRAFFIC_LIGHT_INSET = 82;

/** The inset used everywhere else, matched to the top bars' right padding. */
export const NATIVE_TITLE_BAR_INSET = 20;

/**
 * What an unmeasured platform gets for the top bar's left padding: the macOS
 * inset.
 *
 * This is a DECISION about an unmeasured platform, not a measurement of one.
 * The two ways of being wrong are not symmetrical.
 *
 * Wrong as "not macOS" on a real Mac (a WebView whose UA has been customised
 * past recognition): the header's leftmost control -- the sidebar toggle --
 * gets 20px of inset and paints underneath traffic lights that start at x=16.
 * Those buttons are composited *over* the webview and take the click, so the
 * user does not merely lose the toggle; aiming at it presses Close. Functional,
 * and destructive.
 *
 * Wrong as "macOS" on Windows or Linux: `MACOS_TRAFFIC_LIGHT_INSET -
 * NATIVE_TITLE_BAR_INSET` = 62px of extra padding at the left of a header that
 * spans the window anyway. Nothing is covered, nothing is unreachable, the row
 * is just indented. Cosmetic.
 *
 * Cosmetic beats destructive, so `unknown` takes the macOS side here. Note that
 * this is the opposite side from `UNKNOWN_PLATFORM_PAINTS_SHADOW`, whose
 * asymmetry genuinely runs the other way -- see there.
 */
const UNKNOWN_PLATFORM_TOP_BAR_INSET = MACOS_TRAFFIC_LIGHT_INSET;

/** Left padding for a top bar that shares its row with the window controls. */
export function topBarLeftInset(ua?: string): number {
  const platform = hostPlatform(ua);
  if (platform === "unknown") return UNKNOWN_PLATFORM_TOP_BAR_INSET;
  return platform === "macos" ? MACOS_TRAFFIC_LIGHT_INSET : NATIVE_TITLE_BAR_INSET;
}

/** Drag target macOS needs above content that has no top bar of its own. */
export const MACOS_DRAG_STRIP_HEIGHT = 32;

/** None needed where the platform draws a real, draggable title bar itself. */
export const NATIVE_DRAG_STRIP_HEIGHT = 0;

/**
 * What an unmeasured platform gets for the drag strip: the macOS height.
 *
 * This is a DECISION about an unmeasured platform, not a measurement of one,
 * and it is the same asymmetry as the inset above in a sharper form.
 *
 * Wrong as "not macOS" on a real Mac: the setup wizard gets a strip of zero. It
 * has no top bar of its own, and its window is `titleBarStyle: "Overlay"`, so
 * there is no native bar to grab either. The window cannot be moved. There is
 * no other drag handle on that screen, so the failure is not merely functional
 * but unrecoverable from inside the app.
 *
 * Wrong as "macOS" on Windows or Linux: a 32px transparent strip sits above the
 * wizard's content, under a native title bar that already drags, and the
 * content starts 32px lower. Cosmetic -- and the strip is, if anything, a
 * second working drag target.
 *
 * An unmovable window beats 32px of dead space, so `unknown` takes the macOS
 * side here too.
 */
const UNKNOWN_PLATFORM_DRAG_STRIP_HEIGHT = MACOS_DRAG_STRIP_HEIGHT;

/**
 * Height of the drag strip above content that has no top bar of its own, such
 * as the setup wizard. Off macOS the native title bar already provides both the
 * drag target and the clearance.
 */
export function dragStripHeight(ua?: string): number {
  const platform = hostPlatform(ua);
  if (platform === "unknown") return UNKNOWN_PLATFORM_DRAG_STRIP_HEIGHT;
  return platform === "macos" ? MACOS_DRAG_STRIP_HEIGHT : NATIVE_DRAG_STRIP_HEIGHT;
}

/**
 * Whether an undecorated, transparent window must paint its own drop shadow.
 *
 * macOS: app/src/lib.rs sets `setHasShadow: NO` on the quick-capture NSWindow,
 * alongside `setBackgroundColor: clearColor` and `setOpaque: NO`. The window
 * server therefore draws no shadow at all, so the CSS one is the only thing
 * separating the floating card from the desktop behind it. It also needs a
 * transparent inset to render into, because an outer shadow is clipped at the
 * viewport edge.
 *
 * Windows and Linux: there is no equivalent block. The window is a plain
 * layered `transparent(true)` surface, and a shadow drawn into its transparent
 * margin composites as a flat grey band around the card instead of fading into
 * the desktop -- the "weird semi-transparent outer layer" this scoping removes.
 *
 * Unknown platform (no navigator, an empty user agent, or a WebView whose UA
 * has been customised past recognition): `UNKNOWN_PLATFORM_PAINTS_SHADOW`
 * below.
 */
export function needsCssWindowShadow(ua?: string): boolean {
  const platform = hostPlatform(ua);
  if (platform === "unknown") return UNKNOWN_PLATFORM_PAINTS_SHADOW;
  return platform === "macos";
}

/**
 * What an unmeasured platform gets for the window shadow: `false`.
 *
 * This is a DECISION about an unmeasured platform, not a measurement of one,
 * and it is the one place where the asymmetry runs *against* guessing macOS.
 *
 * Wrong as "not macOS" on a real Mac: the card loses the only shadow it has and
 * reads as plain against the desktop. It still has its own opaque surface, its
 * border, and its accent, so it is legible and fully usable. Cosmetic.
 *
 * Wrong as "macOS" on Windows or Linux: the shadow is drawn into a transparent
 * margin the compositor cannot blend, and renders as the flat grey band around
 * the card -- the "weird semi-transparent outer layer" that was actually
 * reported. That looks broken rather than plain.
 *
 * Both mistakes are cosmetic here, so unlike the two decisions above there is
 * no functional failure to avoid; the tie is broken by which one cannot render
 * as an artifact. The residual is stated: a macOS WebView running a customised
 * UA loses its shadow.
 */
const UNKNOWN_PLATFORM_PAINTS_SHADOW = false;

/** Transparent margin an undecorated window leaves for its own CSS shadow. */
export const CSS_WINDOW_SHADOW_INSET = 12;

/** Room for that shadow, and none where the platform draws no shadow at all. */
export function cssWindowShadowInset(ua?: string): number {
  return needsCssWindowShadow(ua) ? CSS_WINDOW_SHADOW_INSET : 0;
}
