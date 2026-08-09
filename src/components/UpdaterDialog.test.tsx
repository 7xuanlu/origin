// SPDX-License-Identifier: AGPL-3.0-only
import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const emitMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/event", () => ({
  emit: emitMock,
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

import UpdaterDialog from "./UpdaterDialog";

describe("UpdaterDialog", () => {
  beforeEach(() => {
    emitMock.mockReset().mockResolvedValue(undefined);
  });

  it("announces that the updater UI is ready after mounting", async () => {
    render(<UpdaterDialog />);

    await waitFor(() => expect(emitMock).toHaveBeenCalledWith("updater://ui-ready"));
  });
});
