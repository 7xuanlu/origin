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

## llama-cpp-2 backend

macOS builds use Metal, Windows x86_64 builds use Vulkan with observable CPU/OpenMP fallback, and Linux builds remain CPU/OpenMP. Windows setup, device selection, CI/release prerequisites, and physical Qwen live-smoke commands are in [`windows-vulkan.md`](windows-vulkan.md).

## ORT (ONNX Runtime) on Windows

If you see `Failed to load onnxruntime.dll` or version-mismatch errors on Windows, set `ORT_DYLIB_PATH` to the bundled `onnxruntime.dll` inside the Wenlan install directory before starting the daemon. The bundled DLL ships in the Windows release zip.

## Manual Windows verification

The CI matrix includes `windows-2022` for Windows-affecting PRs, but hosted runners do not prove physical GPU inference. Follow [`windows-vulkan.md`](windows-vulkan.md) on a real Windows 11 GPU machine and run all three Qwen live-smoke legs: Vulkan/device assertion, forced CPU, and injected CPU fallback.

## Linux smoke from macOS

```bash
bash scripts/smoke-linux.sh
```

Builds the multi-arch daemon image (linux/arm64 for native Apple Silicon speed via OrbStack / Docker Desktop), starts a container, exercises the HTTP API, asserts responses, tears down. Runtime ~3 minutes after the first build.

