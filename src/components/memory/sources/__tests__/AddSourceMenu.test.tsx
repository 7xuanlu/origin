import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import AddSourceMenu from "../AddSourceMenu";

const mocks = vi.hoisted(() => ({
  getDaemonVersion: vi.fn(),
  uploadSourceFile: vi.fn(),
  openFile: vi.fn(),
  openDialog: vi.fn(),
}));

vi.mock("../../../../lib/tauri", () => ({
  getDaemonVersion: mocks.getDaemonVersion,
  daemonMeetsFloor: (v: string) => v.split(".").map(Number)[1] >= 10,
  uploadSourceFile: mocks.uploadSourceFile,
  openFile: mocks.openFile,
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: mocks.openDialog }));
vi.mock("sonner", () => ({ toast: vi.fn() }));

function wrap(ui: React.ReactNode) {
  return <QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>;
}

describe("AddSourceMenu", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows both entry points on a current daemon", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.11.0");
    render(wrap(<AddSourceMenu onClose={() => {}} />));
    await screen.findByText("Add a folder");
    expect(screen.getByText("Add files")).toBeInTheDocument();
    expect(screen.queryByText("Your daemon needs an update to index files.")).toBeNull();
  });

  it("shows the update-daemon notice on an old daemon", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.9.5");
    render(wrap(<AddSourceMenu onClose={() => {}} />));
    await screen.findByText("Your daemon needs an update to index files.");
  });

  it("picking Add files stages the chosen file", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.11.0");
    mocks.openDialog.mockResolvedValue("/Users/me/paper.pdf");
    mocks.uploadSourceFile.mockResolvedValue({ id: "directory-sources" });
    render(wrap(<AddSourceMenu onClose={() => {}} />));
    fireEvent.click(await screen.findByText("Add files"));
    await waitFor(() => expect(mocks.uploadSourceFile).toHaveBeenCalledWith("/Users/me/paper.pdf"));
  });
});
