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
 */
export function isMacOS(ua: string | undefined = globalThis.navigator?.userAgent): boolean {
  return /\b(Macintosh|Mac OS X)\b/.test(ua ?? "");
}

/** Room for the overlaid traffic lights, matched to their x=16 origin. */
export const MACOS_TRAFFIC_LIGHT_INSET = 82;

/** The inset used everywhere else, matched to the top bars' right padding. */
export const NATIVE_TITLE_BAR_INSET = 20;

/** Left padding for a top bar that shares its row with the window controls. */
export function topBarLeftInset(ua?: string): number {
  return isMacOS(ua) ? MACOS_TRAFFIC_LIGHT_INSET : NATIVE_TITLE_BAR_INSET;
}

/**
 * Height of the drag strip above content that has no top bar of its own, such
 * as the setup wizard. Off macOS the native title bar already provides both the
 * drag target and the clearance.
 */
export function dragStripHeight(ua?: string): number {
  return isMacOS(ua) ? 32 : 0;
}
