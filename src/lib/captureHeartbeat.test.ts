// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from 'vitest';
import {
  recordCapture,
  getLastCapture,
  subscribe,
} from './captureHeartbeat';

describe('captureHeartbeat', () => {
  it('getLastCapture returns correct shape', () => {
    const result = getLastCapture();
    if (result !== null) {
      expect(result).toHaveProperty('source');
      expect(result).toHaveProperty('timestamp');
    }
  });

  it('recordCapture stores source and timestamp', () => {
    const before = Date.now();
    recordCapture('clipboard');
    const after = Date.now();

    const last = getLastCapture();
    expect(last).not.toBeNull();
    expect(last!.source).toBe('clipboard');
    expect(last!.timestamp).toBeGreaterThanOrEqual(before);
    expect(last!.timestamp).toBeLessThanOrEqual(after);
  });

  it('subscribe notifies on recordCapture', () => {
    const listener = vi.fn();
    const unsub = subscribe(listener);

    recordCapture('focus');
    expect(listener).toHaveBeenCalledTimes(1);

    unsub();
    recordCapture('ambient');
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
