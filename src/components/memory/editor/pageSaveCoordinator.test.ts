import { describe, expect, it, vi } from "vitest";

import {
  beginPageSave,
  pageVersionChanged,
  pageDraftChanged,
  settlePageSave,
  type PageEditBaseline,
  type PageSaveCoordinatorState,
} from "./pageSaveCoordinator";

const baseline: PageEditBaseline = {
  pageId: "page-1",
  content: "# Original\n",
  version: 7,
};

function operationIds(...ids: string[]) {
  const factory = vi.fn();
  for (const id of ids) factory.mockReturnValueOnce(id);
  return factory;
}

describe("page save coordinator", () => {
  it("sends exact untrimmed source with CAS and stable caller identity", () => {
    const createOperationId = operationIds("operation-1");
    const content = "  # Edited  \n\nBody\t \n";

    const attempt = beginPageSave(
      { phase: "idle" },
      baseline,
      content,
      createOperationId,
    );

    expect(attempt).toEqual({
      kind: "request",
      input: {
        id: "page-1",
        content,
        expectedVersion: 7,
        callerId: "wenlan-app",
        operationId: "operation-1",
      },
      state: {
        phase: "pending",
        pending: {
          operationId: "operation-1",
          content,
          expectedVersion: 7,
        },
      },
    });
    expect(createOperationId).toHaveBeenCalledOnce();
  });

  it("blocks a rapid repeated save while preserving the pending snapshot", () => {
    const createOperationId = operationIds("operation-1", "must-not-be-used");
    const first = beginPageSave(
      { phase: "idle" },
      baseline,
      "# First pending draft\n",
      createOperationId,
    );
    if (first.kind !== "request") throw new Error("expected request");

    expect(
      beginPageSave(
        first.state,
        baseline,
        "# A second click must not replace this snapshot\n",
        createOperationId,
      ),
    ).toEqual({ kind: "blocked_pending", state: first.state });
    expect(createOperationId).toHaveBeenCalledOnce();
  });

  it("does not write unchanged or whitespace-only content", () => {
    const createOperationId = operationIds("unused");

    expect(
      beginPageSave(
        { phase: "idle" },
        baseline,
        baseline.content,
        createOperationId,
      ),
    ).toEqual({ kind: "unchanged", state: { phase: "idle" } });
    expect(
      beginPageSave(
        { phase: "idle" },
        baseline,
        " \n\t ",
        createOperationId,
      ),
    ).toEqual({ kind: "invalid", reason: "empty", state: { phase: "idle" } });
    expect(createOperationId).not.toHaveBeenCalled();
  });

  it("reuses an operation ID for an unchanged retryable snapshot", () => {
    const createOperationId = operationIds("operation-1", "must-not-be-used");
    const first = beginPageSave(
      { phase: "idle" },
      baseline,
      "# Edited\n",
      createOperationId,
    );
    if (first.kind !== "request") throw new Error("expected request");

    const failed = settlePageSave(first.state, {
      outcome: "failure",
      kind: "server",
      status: 503,
      message: "unavailable",
    });
    const retry = beginPageSave(
      failed,
      baseline,
      "# Edited\n",
      createOperationId,
    );

    expect(retry.kind).toBe("request");
    if (retry.kind !== "request") return;
    expect(retry.input.operationId).toBe("operation-1");
    expect(createOperationId).toHaveBeenCalledOnce();
  });

  it("creates a new operation ID after editing a failed snapshot", () => {
    const createOperationId = operationIds("operation-1", "operation-2");
    const first = beginPageSave(
      { phase: "idle" },
      baseline,
      "# Edited\n",
      createOperationId,
    );
    if (first.kind !== "request") throw new Error("expected request");
    const failed = settlePageSave(first.state, {
      outcome: "failure",
      kind: "rate_limited",
      status: 429,
      message: "slow down",
    });

    const afterEdit = pageDraftChanged(failed);
    const next = beginPageSave(
      afterEdit,
      baseline,
      "# Edited again\n",
      createOperationId,
    );

    expect(next.kind).toBe("request");
    if (next.kind !== "request") return;
    expect(next.input.operationId).toBe("operation-2");
    expect(createOperationId).toHaveBeenCalledTimes(2);
  });

  it("keeps conflict state and blocks repeated stale saves until reload", () => {
    const createOperationId = operationIds("operation-1", "must-not-be-used");
    const first = beginPageSave(
      { phase: "idle" },
      baseline,
      "# Local draft\n",
      createOperationId,
    );
    if (first.kind !== "request") throw new Error("expected request");
    const conflict = settlePageSave(first.state, {
      outcome: "conflict",
      message: "page changed",
    });

    expect(conflict).toEqual({
      phase: "conflict",
      pending: first.state.pending,
      message: "page changed",
    });
    expect(pageDraftChanged(conflict)).toEqual(conflict);
    expect(
      beginPageSave(
        conflict,
        baseline,
        "# Local draft plus reconciliation\n",
        createOperationId,
      ),
    ).toEqual({ kind: "blocked_conflict", state: conflict });
    expect(createOperationId).toHaveBeenCalledOnce();
  });

  it("enters conflict before saving when a dirty baseline version changes", () => {
    const createConflictId = operationIds("external-conflict-1");
    const conflict = pageVersionChanged(
      { phase: "idle" },
      baseline,
      " \n",
      8,
      createConflictId,
    );

    expect(conflict).toEqual({
      phase: "conflict",
      pending: {
        operationId: "external-conflict-1",
        content: " \n",
        expectedVersion: 7,
      },
      message: "Page version changed from 7 to 8.",
    });
    const createSaveId = operationIds("must-not-be-used");
    expect(
      beginPageSave(conflict, baseline, " \n", createSaveId),
    ).toEqual({ kind: "blocked_conflict", state: conflict });
    expect(createConflictId).toHaveBeenCalledOnce();
    expect(createSaveId).not.toHaveBeenCalled();
  });

  it.each(["not_found", "payload_too_large", "validation", "other"] as const)(
    "retains a definite %s failure without offering an implicit retry",
    (kind) => {
      const pendingState: PageSaveCoordinatorState = {
        phase: "pending",
        pending: {
          operationId: "operation-1",
          content: "# Local draft\n",
          expectedVersion: 7,
        },
      };
      const failed = settlePageSave(pendingState, {
        outcome: "failure",
        kind,
        status: 422,
        message: "daemon detail",
      });

      expect(failed.phase).toBe("failed");
      expect(pageDraftChanged(failed)).toEqual({ phase: "idle" });
    },
  );

  it("treats transport failure as retryable with the same snapshot", () => {
    const pendingState: PageSaveCoordinatorState = {
      phase: "pending",
      pending: {
        operationId: "operation-1",
        content: "# Local draft\n",
        expectedVersion: 7,
      },
    };

    expect(settlePageSave(pendingState, { outcome: "transport" })).toEqual({
      phase: "retryable",
      pending: pendingState.pending,
      failure: { outcome: "transport" },
    });
  });

  it("retains an upgrade-required snapshot without turning the save action into a retry", () => {
    const pendingState: PageSaveCoordinatorState = {
      phase: "pending",
      pending: {
        operationId: "operation-1",
        content: "# Local draft\n",
        expectedVersion: 7,
      },
    };

    const upgradeRequired = settlePageSave(pendingState, {
      outcome: "upgrade_required",
      reportedVersion: "0.14.0",
      requiredFloor: "0.14.1",
    });

    expect(upgradeRequired).toEqual({
      phase: "upgrade_required",
      pending: pendingState.pending,
      reportedVersion: "0.14.0",
      requiredFloor: "0.14.1",
    });
    expect(pageDraftChanged(upgradeRequired)).toEqual(upgradeRequired);

    const laterSave = beginPageSave(
      upgradeRequired,
      baseline,
      "# Local draft after daemon upgrade\n",
      operationIds("operation-2"),
    );
    expect(laterSave.kind).toBe("request");
    if (laterSave.kind !== "request") return;
    expect(laterSave.input.operationId).toBe("operation-2");
  });
});
