// SPDX-License-Identifier: AGPL-3.0-only
/** `Reading` fixtures (see `Reading` in src/lib/tauri.ts).
 *
 *  These exist so a fixture cannot quietly spell a three-valued measurement as
 *  a boolean. `YES`/`NO` are the two MEASURED answers; `unreadable` is the
 *  third — a look that failed — and it has no boolean spelling at all, which
 *  is the point: a test that wants it has to say so. */
import type { Reading } from "../lib/reading";

export const YES: Reading = { kind: "yes" };
export const NO: Reading = { kind: "no" };
export const unreadable = (error: string): Reading => ({ kind: "unreadable", error });
