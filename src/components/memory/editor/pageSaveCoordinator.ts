import type {
  UpdatePageInput,
  UpdatePageOutcome,
} from "../../../lib/tauri";

export interface PageEditBaseline {
  pageId: string;
  content: string;
  version: number;
}

export interface PendingPageSave {
  operationId: string;
  content: string;
  expectedVersion: number;
}

export type PageSaveTransportFailure = { outcome: "transport" };
export type PageSaveSettlement = UpdatePageOutcome | PageSaveTransportFailure;
type UpdatePageFailure = Extract<
  UpdatePageOutcome,
  { outcome: "failure" }
>;
type RetryablePageSaveFailure =
  | PageSaveTransportFailure
  | (UpdatePageFailure & {
      kind: "auth_required" | "rate_limited" | "server";
    });

function isRetryableUpdatePageFailure(
  failure: UpdatePageFailure,
): failure is UpdatePageFailure & {
  kind: "auth_required" | "rate_limited" | "server";
} {
  return (
    failure.kind === "auth_required" ||
    failure.kind === "rate_limited" ||
    failure.kind === "server"
  );
}

export type PageSaveCoordinatorState =
  | { phase: "idle" }
  | { phase: "pending"; pending: PendingPageSave }
  | {
      phase: "retryable";
      pending: PendingPageSave;
      failure: RetryablePageSaveFailure;
    }
  | {
      phase: "failed";
      pending: PendingPageSave;
      failure: UpdatePageFailure;
    }
  | {
      phase: "upgrade_required";
      pending: PendingPageSave;
      reportedVersion: string;
      requiredFloor: string;
    }
  | { phase: "conflict"; pending: PendingPageSave; message: string };

export type BeginPageSaveResult =
  | {
      kind: "request";
      input: UpdatePageInput;
      state: Extract<PageSaveCoordinatorState, { phase: "pending" }>;
    }
  | { kind: "unchanged"; state: PageSaveCoordinatorState }
  | { kind: "invalid"; reason: "empty"; state: PageSaveCoordinatorState }
  | { kind: "blocked_pending"; state: PageSaveCoordinatorState }
  | { kind: "blocked_conflict"; state: PageSaveCoordinatorState };

export function beginPageSave(
  state: PageSaveCoordinatorState,
  baseline: PageEditBaseline,
  content: string,
  createOperationId: () => string,
): BeginPageSaveResult {
  if (state.phase === "conflict") {
    return { kind: "blocked_conflict", state };
  }
  if (state.phase === "pending") {
    return { kind: "blocked_pending", state };
  }
  if (!content.trim()) {
    return { kind: "invalid", reason: "empty", state };
  }
  if (content === baseline.content) {
    return { kind: "unchanged", state };
  }

  const operationId =
    state.phase === "retryable" &&
    state.pending.content === content &&
    state.pending.expectedVersion === baseline.version
      ? state.pending.operationId
      : createOperationId();
  const pending: PendingPageSave = {
    operationId,
    content,
    expectedVersion: baseline.version,
  };

  return {
    kind: "request",
    input: {
      id: baseline.pageId,
      content,
      expectedVersion: baseline.version,
      callerId: "wenlan-app",
      operationId,
    },
    state: { phase: "pending", pending },
  };
}

export function settlePageSave(
  state: PageSaveCoordinatorState,
  outcome: PageSaveSettlement,
): PageSaveCoordinatorState {
  if (state.phase !== "pending") return state;

  if (outcome.outcome === "saved") {
    return { phase: "idle" };
  }
  if (outcome.outcome === "conflict") {
    return {
      phase: "conflict",
      pending: state.pending,
      message: outcome.message,
    };
  }
  if (outcome.outcome === "upgrade_required") {
    return {
      phase: "upgrade_required",
      pending: state.pending,
      reportedVersion: outcome.reportedVersion,
      requiredFloor: outcome.requiredFloor,
    };
  }
  if (outcome.outcome === "transport") {
    return {
      phase: "retryable",
      pending: state.pending,
      failure: outcome,
    };
  }
  if (isRetryableUpdatePageFailure(outcome)) {
    return {
      phase: "retryable",
      pending: state.pending,
      failure: outcome,
    };
  }

  return {
    phase: "failed",
    pending: state.pending,
    failure: outcome,
  };
}

export function pageVersionChanged(
  state: PageSaveCoordinatorState,
  baseline: PageEditBaseline,
  content: string,
  latestVersion: number,
  createConflictId: () => string,
): PageSaveCoordinatorState {
  if (
    state.phase === "pending" ||
    state.phase === "conflict" ||
    latestVersion === baseline.version ||
    content === baseline.content
  ) {
    return state;
  }

  return {
    phase: "conflict",
    pending: {
      operationId: createConflictId(),
      content,
      expectedVersion: baseline.version,
    },
    message: `Page version changed from ${baseline.version} to ${latestVersion}.`,
  };
}

export function pageDraftChanged(
  state: PageSaveCoordinatorState,
): PageSaveCoordinatorState {
  if (
    state.phase === "conflict" ||
    state.phase === "pending" ||
    state.phase === "upgrade_required"
  ) {
    return state;
  }
  return { phase: "idle" };
}
