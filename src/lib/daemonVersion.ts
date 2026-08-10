// SPDX-License-Identifier: AGPL-3.0-only
import { gte, parse } from "semver";

/**
 * First released daemon proven to preserve exact page source while keeping the
 * v0.14.0 CAS/idempotency contract, so the first one page edits may target.
 *
 * The native bridge carries its own copy in `app/src/search.rs`, because a Rust
 * command cannot import this module. `daemonVersion.test.ts` reads that file and
 * fails if the two ever disagree — the UI must not offer an edit the backing
 * command will refuse.
 */
export const PAGE_EDIT_DAEMON_FLOOR = "0.14.1";

/**
 * Return whether a daemon version meets a stable SemVer floor.
 *
 * Rust's semver parser rejects loose forms such as `v1.2.3` and surrounding
 * whitespace. Keep every TypeScript surface equally strict so entry and save
 * gates cannot disagree with the native bridge.
 */
export function daemonMeetsFloor(
  version: string | null | undefined,
  floor = "0.10.0",
): boolean {
  if (!version) return false;

  const candidate = parse(version);
  const minimum = parse(floor);
  if (!candidate || !minimum) return false;
  const canonicalCandidate =
    candidate.version +
    (candidate.build.length > 0 ? `+${candidate.build.join(".")}` : "");
  const canonicalMinimum =
    minimum.version +
    (minimum.build.length > 0 ? `+${minimum.build.join(".")}` : "");
  if (canonicalCandidate !== version || canonicalMinimum !== floor) return false;
  if (candidate.prerelease.length > 0 || minimum.prerelease.length > 0) {
    return false;
  }

  return gte(candidate, minimum);
}
