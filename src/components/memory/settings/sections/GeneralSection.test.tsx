// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import GeneralSection from "./GeneralSection";
import { isRunAtLoginEnabled, setRunAtLogin } from "../../../../lib/tauri";

vi.mock("../../../../lib/tauri", () => ({
  getProfile: vi.fn(() =>
    Promise.resolve({
      id: "p1",
      name: "Lucian",
      display_name: "Lucian",
      email: null,
      bio: null,
      avatar_path: null,
      created_at: 0,
    }),
  ),
  updateProfile: vi.fn(() => Promise.resolve()),
  setAvatar: vi.fn(() => Promise.resolve()),
  removeAvatar: vi.fn(() => Promise.resolve()),
  setSetupCompleted: vi.fn(() => Promise.resolve()),
  isRunAtLoginEnabled: vi.fn(() => Promise.resolve(false)),
  setRunAtLogin: vi.fn(() => Promise.resolve()),
}));

vi.mock("../../../../lib/theme", () => ({
  useTheme: () => ["system", vi.fn()] as const,
}));

function renderGeneralSection() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <GeneralSection />
      </QueryClientProvider>,
    ),
    // Exposed so a test can drive a REAL refetch failure rather than
    // simulating the state it would produce.
    queryClient,
  };
}

/**
 * Wait for the run-at-login state to actually be MEASURED, then click.
 *
 * The row is inert until the query settles, because until then there is no
 * value to take the complement of — `!(undefined ?? false)` is `true` whatever
 * launchd holds. Tests that clicked without waiting used to pass only because
 * the old code was willing to act on that fiction; waiting is what makes the
 * click a real user action rather than a race.
 */
async function clickRunAtLoginOnce(): Promise<HTMLElement> {
  const toggle = await screen.findByLabelText("Run Wenlan in background at login");
  await waitFor(() => expect(toggle).not.toBeDisabled());
  fireEvent.click(toggle);
  return toggle;
}

// S6: run-at-login, theme, and language used to be three separate
// `Card padding="rows"` blocks. They now share one. `.divide-y` is the class
// `Card` only applies for padding="rows", so counting it proves the merge
// instead of just trusting the JSX shape. ProfileSettingsBlock also renders
// a `rows` Card (Photo + Display name), so scope to the card that contains
// "Language" — unique to the App section — rather than every `.divide-y`.
describe("GeneralSection app card merge", () => {
  it("renders run-at-login, theme, and language inside a single rows Card", async () => {
    renderGeneralSection();

    await screen.findByLabelText("Language");

    const rowsCards = Array.from(document.querySelectorAll(".divide-y")).filter((el) =>
      el.textContent?.includes("Language"),
    );
    expect(rowsCards).toHaveLength(1);

    const merged = rowsCards[0];
    expect(merged.textContent).toContain("Run Wenlan in background at login");
    expect(merged.textContent).toContain("Theme");
    expect(merged.textContent).toContain("Language");
  });
});

// `is_run_at_login_enabled` reports an unreadable launchctl as an error rather
// than as `false`. A rejected query must not leave the row looking like a
// measured "off": the user would read the feature as disabled while launchd
// may still be starting Wenlan every boot.
describe("GeneralSection run-at-login state", () => {
  it("says the state could not be read instead of silently showing it off", async () => {
    vi.mocked(isRunAtLoginEnabled).mockRejectedValueOnce(
      new Error("Could not read the Run at Login state: launchctl did not answer."),
    );

    renderGeneralSection();

    expect(
      await screen.findByText(/could not read whether this is on/i),
    ).toBeInTheDocument();
  });

  it("shows no such notice when the state was actually measured", async () => {
    renderGeneralSection();

    await screen.findByLabelText("Run Wenlan in background at login");
    expect(screen.queryByText(/could not read whether this is on/i)).toBeNull();
  });
});

// `set_run_at_login` refuses the handover when it could not confirm Wenlan's
// own daemon stopped — registering launchd against a port the old daemon
// still holds would leave two owners. The refusal names which of the two
// happened and what to do about it, and that sentence exists only in the
// rejected `invoke` promise. Without it rendered, the toggle simply fails to
// move and the user is told nothing about why.
describe("GeneralSection run-at-login refusal", () => {
  it("shows the reason the handover was refused", async () => {
    const refusal =
      "Run at Login was not changed: Wenlan's own daemon is still running and still holds " +
      "the port, so handing it to the system launcher would leave two owners. Quit and " +
      "reopen Wenlan, then try again.";
    // A Tauri command declared `Result<_, String>` rejects with the bare
    // string, not an Error — so this is the shape the component really sees.
    vi.mocked(setRunAtLogin).mockRejectedValueOnce(refusal);

    renderGeneralSection();
    await clickRunAtLoginOnce();

    expect(await screen.findByText(refusal)).toBeInTheDocument();
  });

  it("says nothing when the handover succeeded", async () => {
    renderGeneralSection();
    await clickRunAtLoginOnce();

    await waitFor(() => expect(vi.mocked(setRunAtLogin)).toHaveBeenCalled());
    expect(screen.queryByText(/was not changed/i)).toBeNull();
  });

  // A rejection that carries no readable sentence must leave the row silent
  // rather than printing `[object Object]` or an empty red line: a row that
  // claims to be explaining something, and is not, is worse than no row.
  // An unread state is not `false`. Painting the switch off claims launchd was
  // measured, and `!(undefined ?? false)` is `true` whatever launchd actually
  // holds — so a click on an unreadable row would send a value derived from
  // nothing. Both halves are asserted: the row must not claim, and must not act.
  it("neither paints nor acts on a run-at-login state it could not read", async () => {
    vi.mocked(isRunAtLoginEnabled).mockRejectedValueOnce(
      new Error("Could not read the Run at Login state: launchctl did not answer."),
    );

    renderGeneralSection();
    await screen.findByText(/could not read whether this is on/i);

    const toggle = screen.getByLabelText("Run Wenlan in background at login");
    expect(toggle).toBeDisabled();
    // Not `aria-pressed="false"` — that is a measurement claim we cannot make.
    expect(toggle).not.toHaveAttribute("aria-pressed");

    // Counted rather than `not.toHaveBeenCalled()`: the module mock is shared
    // across this file and is not cleared between tests, so an absolute
    // assertion here would be measuring earlier tests' calls, not this click.
    const callsBefore = vi.mocked(setRunAtLogin).mock.calls.length;
    fireEvent.click(toggle);
    expect(vi.mocked(setRunAtLogin).mock.calls.length).toBe(callsBefore);
  });

  // The two failures are independent and say different things: one reports the
  // value is unknown, the other names why a change was refused and what to do.
  // Ranking them drops the half carrying the remedy.
  it("shows the unreadable notice and the refusal together when both are live", async () => {
    const refusal =
      "Run at Login was not changed: Wenlan's own daemon is still running and still holds " +
      "the port, so handing it to the system launcher would leave two owners.";
    vi.mocked(setRunAtLogin).mockRejectedValueOnce(refusal);

    const { queryClient } = renderGeneralSection();
    await clickRunAtLoginOnce();
    await screen.findByText(new RegExp(refusal.slice(0, 40).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));

    // A real refetch that fails, not a simulated error state.
    vi.mocked(isRunAtLoginEnabled).mockRejectedValueOnce(new Error("launchctl went away"));
    await queryClient.invalidateQueries({ queryKey: ["runAtLogin"] });

    await waitFor(() => {
      const row = screen.getByLabelText("Run Wenlan in background at login").closest(".px-5");
      expect(row?.textContent).toMatch(/could not read whether this is on/i);
      expect(row?.textContent).toContain("still holds the port");
    });
  });

  it("stays silent when the rejection carries no message", async () => {
    vi.mocked(setRunAtLogin).mockRejectedValueOnce({ code: 42 });

    renderGeneralSection();
    await clickRunAtLoginOnce();

    await waitFor(() => expect(vi.mocked(setRunAtLogin)).toHaveBeenCalled());
    expect(screen.queryByText(/object Object/i)).toBeNull();
  });
});

// Round 5, D2. The guard used to be `runAtLoginQuery.data === undefined`, which
// answers "has any value ever been cached", not "did the current read succeed".
// React Query RETAINS the last `data` when a refetch fails, so the reachable
// state `data === false, isError === true` is a FAILED read still wearing an
// earlier read's answer — and the row rendered it as an enabled, measured
// switch beside the notice saying the state could not be read. Both retained
// values are staged, because they fail in opposite directions: a retained
// `false` sends `mutate(true)`, a retained `true` sends `mutate(false)`. Both
// of these fail against the old guard, where the toggle stays enabled.
describe("GeneralSection run-at-login value retained across a failed refresh", () => {
  const runAtLoginToggle = () =>
    screen.getByLabelText("Run Wenlan in background at login");

  async function measureThenFailRefresh(measured: boolean) {
    vi.mocked(isRunAtLoginEnabled).mockResolvedValueOnce(measured);
    const { queryClient } = renderGeneralSection();

    await screen.findByLabelText("Run Wenlan in background at login");
    await waitFor(() => expect(runAtLoginToggle()).not.toBeDisabled());
    expect(runAtLoginToggle()).toHaveAttribute("aria-pressed", String(measured));

    // A real refetch that fails, leaving `data` at `measured` and `isError`
    // true — not a simulated error state.
    vi.mocked(isRunAtLoginEnabled).mockRejectedValueOnce(
      new Error("Could not read the Run at Login state: launchctl did not answer."),
    );
    await queryClient.invalidateQueries({ queryKey: ["runAtLogin"] });
    await waitFor(() =>
      expect(screen.getByText(/could not read whether this is on/i)).toBeInTheDocument(),
    );
  }

  it.each([true, false])(
    "stops asserting a retained %s once the refresh that would confirm it failed",
    async (measured) => {
      await measureThenFailRefresh(measured);

      // The switch must stop claiming the stale reading is a measurement.
      await waitFor(() => expect(runAtLoginToggle()).toBeDisabled());
      // Not `aria-pressed="false"`/`"true"` — either is a claim nobody read.
      expect(runAtLoginToggle()).not.toHaveAttribute("aria-pressed");

      // And it must not ACT on it: the complement of a stale reading is a
      // write to launchd derived from an earlier instant.
      // Counted rather than `not.toHaveBeenCalled()` — the module mock is
      // shared across this file and never cleared.
      const callsBefore = vi.mocked(setRunAtLogin).mock.calls.length;
      fireEvent.click(runAtLoginToggle());
      expect(vi.mocked(setRunAtLogin).mock.calls.length).toBe(callsBefore);
    },
  );
});
