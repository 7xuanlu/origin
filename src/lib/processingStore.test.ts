// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from 'vitest';
import {
  getSnapshot,
  markProcessing,
  clearProcessing,
  isProcessing,
  subscribe,
} from './processingStore';

describe('processingStore', () => {
  // Each test uses unique IDs and is self-contained — no order dependency.

  it('getSnapshot returns a number', () => {
    expect(typeof getSnapshot()).toBe('number');
  });

  it('markProcessing adds a source ID and bumps snapshot', () => {
    const before = getSnapshot();
    markProcessing('mark-test-1');
    expect(isProcessing('mark-test-1')).toBe(true);
    expect(getSnapshot()).toBeGreaterThan(before);
    clearProcessing('mark-test-1'); // cleanup
  });

  it('markProcessing is idempotent — no bump on duplicate', () => {
    markProcessing('idem-mark-1');
    const before = getSnapshot();
    markProcessing('idem-mark-1');
    expect(getSnapshot()).toBe(before);
    clearProcessing('idem-mark-1'); // cleanup
  });

  it('clearProcessing removes a source ID and bumps snapshot', () => {
    markProcessing('clear-test-1'); // setup: ensure it's in the set
    const before = getSnapshot();
    clearProcessing('clear-test-1');
    expect(isProcessing('clear-test-1')).toBe(false);
    expect(getSnapshot()).toBeGreaterThan(before);
  });

  it('clearProcessing is idempotent — no bump if not present', () => {
    const before = getSnapshot();
    clearProcessing('never-added-id');
    expect(getSnapshot()).toBe(before);
  });

  it('subscribe notifies on change and returns unsubscribe', () => {
    const listener = vi.fn();
    const unsub = subscribe(listener);

    markProcessing('sub-test-a');
    expect(listener).toHaveBeenCalledTimes(1);

    clearProcessing('sub-test-a');
    expect(listener).toHaveBeenCalledTimes(2);

    unsub();
    markProcessing('sub-test-b');
    expect(listener).toHaveBeenCalledTimes(2); // no more calls after unsub
    clearProcessing('sub-test-b'); // cleanup
  });

  it('ignores empty string source IDs', () => {
    const before = getSnapshot();
    markProcessing('');
    expect(getSnapshot()).toBe(before);
    expect(isProcessing('')).toBe(false);
  });
});
