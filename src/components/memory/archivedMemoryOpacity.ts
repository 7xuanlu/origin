// SPDX-License-Identifier: AGPL-3.0-only

/**
 * Resting opacity of a memory that an 'archive'-mode superseder has replaced:
 * kept visible but muted. Every surface that renders memory rows uses this
 * value, and so does the Playwright guard in e2e/archived-memory-opacity.spec.ts,
 * which asserts the browser's *computed* opacity against it — jsdom cannot,
 * because it runs no cascade and no keyframes.
 */
export const ARCHIVED_MEMORY_OPACITY = 0.55;
