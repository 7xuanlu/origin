// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { invoke } from '@tauri-apps/api/core';
import { useSearch } from './useSearch';
import React from 'react';

const mockInvoke = vi.mocked(invoke);

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
    },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe('useSearch', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    // shouldAdvanceTime: true lets waitFor's own polling intervals fire
    // while still giving us manual control with advanceTimersByTime
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns empty results for empty query', () => {
    const { result } = renderHook(() => useSearch(), { wrapper: createWrapper() });
    expect(result.current.query).toBe('');
    expect(result.current.results).toEqual([]);
    expect(result.current.isLoading).toBe(false);
  });

  it('debounces search by 300ms', async () => {
    mockInvoke.mockResolvedValue([{
      id: '1', content: 'match', source: 'test', source_id: 's1',
      title: 'T', url: null, chunk_index: 0, last_modified: 0, score: 1.0
    }]);

    const { result } = renderHook(() => useSearch(), { wrapper: createWrapper() });

    act(() => result.current.setQuery('rust'));

    // Before debounce — should not have fired
    expect(mockInvoke).not.toHaveBeenCalled();

    // Advance past the 300ms debounce window
    await act(async () => {
      vi.advanceTimersByTime(300);
    });

    await waitFor(
      () => {
        expect(mockInvoke).toHaveBeenCalledWith('search', expect.objectContaining({ query: 'rust' }));
      },
      { timeout: 3000 }
    );
  });

  it('clears results when query becomes empty', async () => {
    mockInvoke.mockResolvedValue([{
      id: '1', content: 'x', source: 't', source_id: 's',
      title: 'T', url: null, chunk_index: 0, last_modified: 0, score: 1
    }]);

    const { result } = renderHook(() => useSearch(), { wrapper: createWrapper() });

    act(() => result.current.setQuery('test'));
    await act(async () => { vi.advanceTimersByTime(300); });
    await waitFor(() => expect(result.current.results.length).toBeGreaterThan(0), { timeout: 3000 });

    act(() => result.current.setQuery(''));
    await act(async () => { vi.advanceTimersByTime(300); });
    expect(result.current.results).toEqual([]);
  });
});
