// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { search, getIndexStatus, type SearchResult } from "../lib/tauri";
import { useState, useEffect, useRef } from "react";

export function useSearch(sourceFilter?: string) {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [query]);

  const searchQuery = useQuery({
    queryKey: ["search", debouncedQuery, sourceFilter],
    queryFn: () => search(debouncedQuery, 10, sourceFilter),
    enabled: debouncedQuery.length > 0,
    placeholderData: (prev) => prev,
  });

  // When query is empty, return no results immediately (don't keep stale data)
  const results = debouncedQuery.length > 0
    ? (searchQuery.data ?? []) as SearchResult[]
    : [];

  return {
    query,
    setQuery,
    results,
    isLoading: searchQuery.isFetching && debouncedQuery.length > 0,
    error: searchQuery.error,
  };
}

export function useIndexStatus() {
  return useQuery({
    queryKey: ["indexStatus"],
    queryFn: getIndexStatus,
    refetchInterval: 5000,
  });
}
