# Manual test plan — Tray + lifecycle (Path B)

Run before merging the `feature/tray-lsui-spike` branch. Requires a clean
test user account or willingness to clobber existing Origin install.

## Setup

```bash
pkill -9 -f Origin.app; pkill -9 -f origin-server
rm -f ~/Library/LaunchAgents/com.origin.{server,desktop}.plist
# Reset opt-out sentinel file if present:
rm -f ~/Library/Application\ Support/origin/auto_start_disabled.flag

# Build:
cd /Users/lucian/Repos/origin/.worktrees/tray-lsui-spike
CXXFLAGS="-std=c++17" pnpm release

# Install:
sudo rm -rf /Applications/Origin.app
sudo cp -r target/release/bundle/macos/Origin.app /Applications/
sudo xattr -cr /Applications/Origin.app
```

## Steps


| #   | Action                                                                                                                                                        | Expected                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Double-click `/Applications/Origin.app`                                                                                                                       | No window appears (LSUIElement); tray icon appears in menu bar within 2s; both plists registered (`launchctl list | grep com.origin`)             |
| 2   | Click tray icon                                                                                                                                               | Window appears, focused                                                                                                                           |
| 3   | Click red X on window                                                                                                                                         | Window hides; tray persists                                                                                                                       |
| 4   | Tray menu → "Quit Origin"                                                                                                                                     | Tray gone within 2s; daemon dead (`lsof -ti :7878` empty); plists REMOVED from `~/Library/LaunchAgents/` (full off — spec lifecycle invariant #4) |
| 4a  | Re-launch `/Applications/Origin.app`                                                                                                                          | Tray reappears; both plists re-installed in `~/Library/LaunchAgents/`; `auto_start_disabled.flag` absent                                          |
| 5   | Reboot Mac (or `launchctl bootout gui/$(id -u)/com.origin.server; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.origin.{server,desktop}.plist`) | Both plists auto-load; daemon up; tray icon visible at login                                                                                      |
| 6   | Settings → toggle "Run at login" off                                                                                                                          | App exits within 2s; daemon dies; plists removed from `~/Library/LaunchAgents/`                                                                   |
| 7   | Open Origin.app from /Applications (after Step 6)                                                                                                             | App runs; setup() detects `auto_start_disabled.flag` exists, skips silent install; daemon spawned as Tauri child (fallback mode)                  |
| 8   | Settings → toggle "Run at login" on                                                                                                                           | Daemon up; tray active; plists re-installed; `auto_start_disabled.flag` removed                                                                   |
| 9   | `pkill -9 -f origin-server` (do NOT use `kill $(lsof -ti :7878)` — `lsof` returns the app's keep-alive client too, so it kills both)                          | Tray icon dim ~5s, then active again (launchd respawn)                                                                                            |
| ~   | Force-quit Origin.app via Activity Monitor                                                                                                                    | Tray gone ~5s, then back (launchd respawn)                                                                                                        |
| 11  | Open Origin.app from Dock when LaunchAgent already started it                                                                                                 | Existing instance focused via single-instance plugin (no duplicate process; check `pgrep -c -f "MacOS/origin"`)                                   |
| 12  | First run with `chmod -w ~/Library/LaunchAgents/` (perms denied)                                                                                              | Toast: "Origin background mode unavailable…"; app works in fallback mode                                                                          |


## Cleanup

```bash
chmod +w ~/Library/LaunchAgents/  # restore if step 12 was tested
sudo rm -rf /Applications/Origin.app  # optional
```

