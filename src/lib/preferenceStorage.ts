// SPDX-License-Identifier: AGPL-3.0-only

export function readPreference(key: string, legacyKey?: string): string | null {
  try {
    const current = localStorage.getItem(key);
    if (current !== null) return current;

    if (!legacyKey) return null;
    const legacy = localStorage.getItem(legacyKey);
    if (legacy === null) return null;

    try {
      localStorage.setItem(key, legacy);
    } catch {
      // Reading legacy state is still useful even if writing is unavailable.
    }
    return legacy;
  } catch {
    return null;
  }
}

export function writePreference(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Preferences are non-critical; disabled/full storage should not break UI.
  }
}
