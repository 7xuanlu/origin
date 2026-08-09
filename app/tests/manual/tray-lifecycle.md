# Manual test plan — Wenlan tray + lifecycle

Run before merging lifecycle/runtime identity changes. Requires a clean test
user account or willingness to replace an existing Wenlan install. Legacy
Origin LaunchAgents may be present to verify bridge cleanup, but the current
install target is Wenlan.

## Setup

```bash
pkill -9 -f 'Wenlan.app|wenlan-app'
pkill -9 -f wenlan-server
for label in com.wenlan.desktop com.wenlan.server; do
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
done
rm -f ~/Library/LaunchAgents/com.wenlan.{server,desktop}.plist
# Leave legacy Origin plists alone unless they are fixtures created for step 10.
# Reset opt-out sentinel file if present:
rm -f ~/Library/Application\ Support/wenlan/auto_start_disabled.flag

# Build:
cd /path/to/current/wenlan-app-checkout
CXXFLAGS="-std=c++17" pnpm release

# Install:
sudo rm -rf /Applications/Wenlan.app
sudo cp -r target/release/bundle/macos/Wenlan.app /Applications/
sudo xattr -cr /Applications/Wenlan.app
```

## Steps


| #   | Action                                                                                                                                                        | Expected                                                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Double-click `/Applications/Wenlan.app`                                                                                                                         | Window appears and focuses; `launchctl list` shows both current `com.wenlan` plists                                                                  |
| 2   | Click red X on window                                                                                                                                           | Window hides; app process/tray health stays alive                                                                                                    |
| 3   | Re-open Wenlan from Dock or Finder                                                                                                                              | Existing instance focuses via single-instance plugin; no duplicate app process (`pgrep -c -f "MacOS/wenlan-app"`)                                    |
| 4   | Tray menu -> Quit Wenlan; do not use the macOS app-menu Quit for this step                                                                                       | App exits within 2s; daemon dead (`lsof -ti :7878` empty); current plists removed from `~/Library/LaunchAgents/`                                     |
| 4a  | Re-launch `/Applications/Wenlan.app`                                                                                                                            | App launches; current plists re-installed; `~/Library/Application Support/wenlan/auto_start_disabled.flag` absent                                    |
| 5   | Reboot Mac, or run the launchd fallback below                                                                                                                    | Current plists auto-load; daemon up; Wenlan app can be focused from Dock/Finder                                                                       |
| 6   | Settings -> toggle "Run at login" off                                                                                                                           | App remains usable; current plists are removed; `auto_start_disabled.flag` exists; this toggle is not the full-shutdown path                           |
| 7   | After Step 6, use tray menu -> Quit Wenlan, then open Wenlan.app from /Applications                                                                               | App runs; setup detects `auto_start_disabled.flag`, skips silent install, and spawns daemon as a Tauri child fallback                                 |
| 8   | Settings → toggle "Run at login" on                                                                                                                             | Daemon up; current plists re-installed; `auto_start_disabled.flag` removed                                                                            |
| 9   | `pkill -9 -f wenlan-server` (do NOT use `kill $(lsof -ti :7878)` because it can also kill the app keep-alive client)                                             | Health indicator goes down briefly, then active again after launchd respawn                                                                           |
| 10  | Create owned legacy `com.origin.{server,desktop}.plist`, then launch Wenlan                                                                                      | Owned legacy Origin plists are unloaded/removed only after current Wenlan replacements are installed; foreign legacy files are preserved              |
| 11  | First run with `chmod -w ~/Library/LaunchAgents/` (permissions denied)                                                                                          | App remains usable in fallback mode; no silent deletion of user/foreign LaunchAgent files                                                             |

Launchd fallback for Step 5:

```bash
for label in com.wenlan.desktop com.wenlan.server; do
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
done
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.wenlan.server.plist \
  ~/Library/LaunchAgents/com.wenlan.desktop.plist
```


## Cleanup

```bash
chmod +w ~/Library/LaunchAgents/  # restore if step 11 was tested
sudo rm -rf /Applications/Wenlan.app  # optional
```
