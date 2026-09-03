# Cross-platform support

Platform detail extracted from the root `AGENTS.md` so it loads only when you are
working on platform-conditional code, service registration, or the release matrix.

Supported builds and prebuilt releases cover macOS arm64, Linux x86_64/aarch64 with glibc, and Windows x86_64. macOS x86_64 is not a supported stock source-build target: the pinned ONNX Runtime dependency has no prebuilt Intel macOS binary, so a custom build must compile ONNX Runtime separately and provide it through `ORT_LIB_LOCATION`.

| OS | Data dir | Service registration |
|---|---|---|
| macOS | `~/Library/Application Support/wenlan/` | launchd via `~/Library/LaunchAgents/com.wenlan.server.plist` (user-level) |
| Linux | `~/.local/share/wenlan/` (or `$XDG_DATA_HOME/wenlan`) | systemd user unit at `~/.config/systemd/user/wenlan-server.service` (qualifier dropped per `ServiceLabel::to_script_name()`). Enable lingering with `loginctl enable-linger` if you want the service alive after logout. |
| Windows | `%LOCALAPPDATA%\wenlan\` | Per-user Task Scheduler ONLOGON task registered via `schtasks.exe /create /tn WenlanServer /sc ONLOGON /tr <exe> /f`. `wenlan background on` short-circuits before service-manager and drives schtasks directly (wenlan-server is a plain console app and would otherwise time out at 30s under sc.exe + the Windows Service Control Protocol). `wenlan background off` stops the running task with `schtasks /end /tn WenlanServer` while preserving its registration. |

`wenlan background on` / `wenlan background off` work on macOS, Linux, and Windows. macOS + Linux go through the `service-manager` crate (launchd / systemd-user); Windows takes the schtasks path described above so the daemon does not need a service dispatcher.

The Windows desktop app installs to `%LOCALAPPDATA%\Programs\Wenlan`:
`app/windows/installer-hooks.nsh` moves the NSIS per-user default off
`%LOCALAPPDATA%\Wenlan`, which is the CLI data root above on a case-insensitive
filesystem. A directory the user picks in the installer is kept, and so is the
directory of an install that predates the move when the in-app updater runs (that
path skips the old uninstaller and keeps the existing shortcuts); running the full
installer moves it.

In sidecar mode (no service registration) a quit or SIGTERM asks the daemon to shut
down over HTTP and kills it through the child handle if it has not released the port
within 3 s. On Windows the sidecar is also bound to a kill-on-close job object, so an
app crash takes it down too. That binding can fail (a restricted token, a job
assignment the OS refuses), and the outcome is recorded on the sidecar handle as
`daemon_start::JobBinding` rather than only logged: `sidecar_job_binding()` answers
`Bound`, `Unbound { reason }`, or `NotSupported`, and never assumes. Startup still
continues on `Unbound` — refusing to start would leave the user with no daemon at all
— but the quit path then verifies its kill against the recorded pid *and* process
start time and reissues it.

**Ending the daemon is best-effort, and which cases are covered is a measurement, not
a promise.** `daemon_start::stop_sidecar` returns a `SidecarStopOutcome` —
`Ended`, `StillRunning { reason }`, `CouldNotMeasure { reason }`, or `NoSidecar` —
and the last one is on the diagnostics wire as `daemon.last_sidecar_stop` (`null`
until the app has tried to stop one; `null` is not `ended`). Concretely:

| Case | Does the daemon end? |
|---|---|
| Windows, `JobBinding::Bound` | Yes, on every exit including a hard kill: the job object ends it when the app's last handle closes. This is the only categorical case. |
| Any platform, app code runs, the daemon answers the shutdown request | Yes, and it is measured: the port is released and the recorded process is confirmed gone. |
| Any platform, app code runs, the daemon must be killed | Usually — the child handle kill, plus a kill by pid when the binding did not take. The outcome says which. |
| `Unbound` **and** the sidecar's start time could not be captured | **No guarantee.** The process cannot be identified, so no kill by pid is issued (killing an unidentified pid is worse than not killing) and the result is `CouldNotMeasure`. |
| `Unbound` and the kill by pid failed | **No.** Reported as `StillRunning`; the next launch meets it as a held port. |
| Hard kill of the app (Task Manager, a crash, `Stop-Process`) while `Unbound`, or on macOS/Linux, which have no job object | **No.** No app code runs, so nothing ends the daemon. |

A launchd, systemd, or Task Scheduler daemon is service-owned and outlives the app by
design.

On macOS the app registers the launchd job first on a fresh install and starts its
own sidecar only when launchd does not end up with a loaded job for the selected
data root (registration skipped for an opted-out user, an unstable app path, or an
isolated data dir, or it failed), so the two never compete for the port. Every owner
decision is made and acted on under one lock, so startup and the "Start Wenlan"
button cannot both spawn a sidecar. Turning on "Run at Login" stops an app-owned
sidecar before it hands the daemon to launchd, and starts one again if the handover
fails.

`lifecycle::launchd_owns_server_daemon` answers `Owns` / `DoesNot` / `Unknown`, and
`Unknown` is a real third answer: `launchctl list` that will not run, exits nonzero,
or exits 0 without printing its `PID/Status/Label` table, or a server plist that
exists and cannot be read. Both owner decisions take a port-health measurement
first, so an unknown owner only ever spawns against a silent port — and when it does,
the app records it: `daemon.sidecar_spawned_on_unknown_owner` on the diagnostics wire
is `true` for the rest of that run.

## llama-cpp-2 backend

macOS builds use Metal, Windows x86_64 builds use Vulkan, and Linux builds remain CPU/OpenMP. The Windows CPU/OpenMP fallback was observed on 2026-07-25 on a machine with a working vendor ICD — forced CPU and an injected bad device index both reloaded the model with zero GPU layers and reported a `fallback_reason`. It has never been observed on a machine with **no** ICD, because that path sits downstream of `LlamaBackend::init()`; the Vulkan SDK is a build-time prerequisite only, but the end-user impact of a missing vendor driver is unproven in both directions. Windows setup, device selection, CI/release prerequisites, physical Qwen live-smoke commands, and the driverless-VM gate that would settle the end-user question are in [`windows-vulkan.md`](windows-vulkan.md).

## ORT (ONNX Runtime) on Windows

If you see `Failed to load onnxruntime.dll` or version-mismatch errors on Windows, set `ORT_DYLIB_PATH` to the bundled `onnxruntime.dll` inside the Wenlan install directory before starting the daemon. The bundled DLL ships in the Windows release zip.

## Manual Windows verification

The CI matrix includes `windows-2022` for Windows-affecting PRs, but hosted runners do not prove physical GPU inference. Follow [`windows-vulkan.md`](windows-vulkan.md) on a real Windows 11 GPU machine and run all three Qwen live-smoke legs: Vulkan/device assertion, forced CPU, and injected CPU fallback.

A fourth leg is defined but **has never been run**: the same install on a clean Windows VM or Windows Sandbox with *no* vendor ICD, from the real signed installer or release ZIP, loading a real GGUF and asserting `/api/status` reports `backend=cpu` and `gpu_layers=0`. The three legs above all ran on a machine with working drivers, so none of them covers it. Until that leg runs, the driverless end-user path is unproven — do not assert either that users are unaffected or that they are affected.

## Linux smoke from macOS

```bash
bash scripts/smoke-linux.sh
```

Builds the multi-arch daemon image (linux/arm64 for native Apple Silicon speed via OrbStack / Docker Desktop), starts a container, exercises the HTTP API, asserts responses, tears down. Runtime ~3 minutes after the first build.

