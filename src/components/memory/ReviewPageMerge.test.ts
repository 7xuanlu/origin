// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { deriveMergeLedger } from "./ReviewPageMerge";
import type { PageSourceWithMemory } from "../../lib/tauri";

function source(pageId: string, memorySourceId: string): PageSourceWithMemory {
  return {
    source: {
      page_id: pageId,
      memory_source_id: memorySourceId,
      linked_at: 0,
    },
    memory: null,
  };
}

describe("deriveMergeLedger", () => {
  it("subset case: retiring page's sources are all shared — onlyRetire is empty", () => {
    const keep = [source("keep", "a"), source("keep", "b"), source("keep", "c")];
    const retire = [source("retire", "a"), source("retire", "b")];

    const ledger = deriveMergeLedger(keep, retire);

    expect(ledger.shared.map((entry) => entry.source.memory_source_id)).toEqual([
      "a",
      "b",
    ]);
    expect(ledger.onlyKeep.map((entry) => entry.source.memory_source_id)).toEqual([
      "c",
    ]);
    expect(ledger.onlyRetire).toEqual([]);
  });

  it("transfer case: retiring page has unique sources that move to the kept page", () => {
    const keep = [source("keep", "a"), source("keep", "b")];
    const retire = [source("retire", "b"), source("retire", "c"), source("retire", "d")];

    const ledger = deriveMergeLedger(keep, retire);

    expect(ledger.shared.map((entry) => entry.source.memory_source_id)).toEqual([
      "b",
    ]);
    expect(ledger.onlyKeep.map((entry) => entry.source.memory_source_id)).toEqual([
      "a",
    ]);
    expect(ledger.onlyRetire.map((entry) => entry.source.memory_source_id)).toEqual([
      "c",
      "d",
    ]);
  });

  it("disjoint case: no overlap between the two pages' sources", () => {
    const keep = [source("keep", "a"), source("keep", "b")];
    const retire = [source("retire", "x"), source("retire", "y")];

    const ledger = deriveMergeLedger(keep, retire);

    expect(ledger.shared).toEqual([]);
    expect(ledger.onlyKeep.map((entry) => entry.source.memory_source_id)).toEqual([
      "a",
      "b",
    ]);
    expect(ledger.onlyRetire.map((entry) => entry.source.memory_source_id)).toEqual([
      "x",
      "y",
    ]);
  });

  it("empty inputs on both sides produce empty groups", () => {
    expect(deriveMergeLedger([], [])).toEqual({
      shared: [],
      onlyKeep: [],
      onlyRetire: [],
    });
  });
});
