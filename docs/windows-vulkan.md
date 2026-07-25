# Windows Vulkan development and live verification

Windows x86_64 builds compile llama.cpp with Vulkan plus CPU/OpenMP. Vulkan is
the default accelerator because one release binary can use supported NVIDIA,
AMD, and Intel GPUs without requiring users to install the CUDA toolkit.
FastEmbed/ONNX Runtime embeddings and reranking remain CPU-backed; this document
is specifically about the Qwen GGUF inference path.

## Prerequisites

- Windows 11 x86_64 with a current vendor GPU driver and a working Vulkan
  runtime (`vulkaninfo.exe --summary` should list the adapter).
- Rust 1.95.0, matching `rust-toolchain.toml`.
- Visual Studio 2022 or 2019 Build Tools with **Desktop development with C++**
  and a Windows SDK.
- CMake, LLVM/libclang, vcpkg, and a complete Strawberry Perl distribution.
  The CI-compatible SQLite triplet is `sqlite3:x64-windows-static-md`. Git for
  Windows' trimmed Perl is not sufficient for the vendored OpenSSL build
  because it omits modules such as `Locale::Maketext::Simple`.
- LunarG Vulkan SDK 1.4.350.0. The repository setup script downloads the
  pinned official installer, verifies SHA-256, and uses LunarG's `copy_only=1`
  mode so it does not require Administrator access or write registry state.

From PowerShell:

```powershell
winget install --id Kitware.CMake --exact
winget install --id LLVM.LLVM --exact
winget install --id StrawberryPerl.StrawberryPerl --exact

git clone --depth 1 --branch 2026.06.24 https://github.com/microsoft/vcpkg.git "$env:LOCALAPPDATA\wenlan-build\vcpkg"
& "$env:LOCALAPPDATA\wenlan-build\vcpkg\bootstrap-vcpkg.bat" -disableMetrics
& "$env:LOCALAPPDATA\wenlan-build\vcpkg\vcpkg.exe" install sqlite3:x64-windows-static-md

& scripts\setup-vulkan-sdk-windows.ps1
$env:LIB = "$env:LOCALAPPDATA\wenlan-build\vcpkg\installed\x64-windows-static-md\lib;$env:LIB"
& scripts\setup-msvc-ninja-windows.ps1

# llama.cpp's nested Vulkan shader build can exceed legacy MAX_PATH when the
# checkout is deep. Keep Cargo output on a deliberately short local path.
$env:CARGO_TARGET_DIR = "C:\wl-target"

# MSVC's nested Vulkan shader probes can concurrently write the same PDB.
# Serialize release builds so cl.exe does not fail with C1041.
$env:CARGO_BUILD_JOBS = "1"
```

If libclang is not on its standard path, set `LIBCLANG_PATH` to the directory
containing `libclang.dll`. The MSVC setup script enters the x64 developer
environment, selects Visual Studio's bundled Ninja, preserves the vcpkg
`LIB` prefix, and serializes nested CMake builds. Use it with both Visual
Studio 2019 and 2022: llama.cpp's Vulkan ExternalProject/shader rules are not
reliably ordered by the Visual Studio generators. Verify `perl
-MLocale::Maketext::Simple -e "print qq(ok\n)"` before building the server.

## Build and test

```powershell
cargo fmt --check --all
cargo test -p wenlan-types
cargo test -p wenlan-core --lib engine::tests
cargo test -p wenlan-server status_reports_selected_vulkan_device
cargo build --release --jobs 1 -p wenlan-core --bin model_probe
cargo build --release --jobs 1 -p wenlan-server

& scripts\setup-vulkan-sdk-windows.test.ps1
& scripts\setup-msvc-ninja-windows.test.ps1
& scripts\smoke-windows-llm.test.ps1
```

The Windows CI and release jobs run the same pinned Vulkan SDK setup before any
Cargo build. The Vulkan SDK is a build-time prerequisite; end users need a
working Vulkan-capable GPU driver, not the SDK. The Vulkan-enabled Windows
executables have a process-start dependency on `vulkan-1.dll`; a missing
Vulkan loader fails before Rust can select CPU. Install a current vendor GPU
driver or Vulkan runtime even when using `WENLAN_LLM_DEVICE=cpu`.

## Device policy and observability

`WENLAN_LLM_DEVICE` controls llama.cpp device selection:

| Value | Behavior |
|---|---|
| unset or `auto` | Prefer a discrete GPU, then free memory, then the lower stable llama.cpp device index |
| `cpu` | Force CPU/OpenMP |
| `<index>` | Force the matching llama.cpp GPU device index |

An invalid index, GPU model-load failure, or GPU context-allocation failure
falls back to CPU. A model/context failure performs a real second model load
with zero GPU layers; it does not merely relabel the failed GPU instance.

`GET /api/status` exposes the effective result:

```json
{
  "on_device_inference": {
    "backend": "vulkan",
    "device": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "device_index": 1,
    "gpu_layers": 99
  }
}
```

When recovery occurs, `backend` is `cpu`, `gpu_layers` is `0`, and
`fallback_reason` records the GPU or selection failure.

## Physical Windows live smoke

Use the cached Qwen GGUF and the release `model_probe.exe`. These are live
smokes: each command loads the real model and requires it to return a valid
`preference` classification.

```powershell
$model = "$env:USERPROFILE\.cache\wenlan\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# Auto policy must select the discrete NVIDIA adapter on a mixed-GPU machine.
& scripts\smoke-windows-llm.ps1 `
  -ModelPath $model `
  -ProbePath target\release\model_probe.exe `
  -Device auto `
  -ExpectedBackend vulkan `
  -ExpectedDevicePattern "NVIDIA.*RTX 3060"

# CPU remains a supported, deterministic escape hatch.
& scripts\smoke-windows-llm.ps1 `
  -ModelPath $model `
  -ProbePath target\release\model_probe.exe `
  -Device cpu `
  -ExpectedBackend cpu

# Inject an unavailable device selection and prove visible CPU recovery.
& scripts\smoke-windows-llm.ps1 `
  -ModelPath $model `
  -ProbePath target\release\model_probe.exe `
  -Device 99 `
  -ExpectedBackend cpu `
  -ExpectedFallbackPattern "requested GPU device index 99 is unavailable"
```

Do not count a successful compile, a placeholder sidecar, or hardware inventory
as a live smoke. The pass marker is
`--- Verified classification: preference ---` together with the asserted
effective backend and device. CPU smoke additionally rejects any real
`VulkanN ... buffer size = ...` allocation; llama.cpp may still print device
inventory, use `Vulkan_Host` memory, and report a zero-byte Vulkan device
buffer during teardown.

### Verified physical result

The 2026-07-22 physical run used Windows 11, an Intel Iris Xe integrated GPU,
an NVIDIA GeForce RTX 3060 Laptop discrete GPU, Vulkan SDK 1.4.350.0, Visual
Studio Build Tools 2019, and
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2,497,281,120 bytes, SHA-256
`3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`).
The latest verification ran backend code commit
`b4677e277e70613585e37e99fa29721426b2a179`, after merging current
`origin/main`. The source-built `wenlan-server.exe` SHA-256 was
`a38b6c682ea6750ac1802ded0517c6ffe7a3bf2538c1b517e2c6bdeda02f03bf`.

| Leg | Observed result |
|---|---|
| `auto`, expected Vulkan | Selected llama.cpp device `1`, `NVIDIA GeForce RTX 3060 Laptop GPU`; offloaded `37/37` layers; allocated 576 MiB KV and 301.75 MiB compute on Vulkan1; valid classification |
| `cpu`, expected CPU | All 36 KV layers reported `dev = CPU`; graph splits `1`; Vulkan1 device compute allocation `0.0000 MiB`; valid classification in about 11.56 seconds |
| device `99`, expected fallback | Reported `requested GPU device index 99 is unavailable`; used the same CPU-only context contract; valid classification in about 11.95 seconds |
| warm Vulkan inference | Valid classification in about 1.16 seconds; an earlier cold Vulkan run also paid shader/pipeline setup and took about 20.56 seconds |
| status route | `routes::recent_endpoints_tests::status_reports_selected_vulkan_device` passed |
| app-owned daemon | `wenlan-app` native WebView2 smoke reported `vulkan`, device `1`, RTX 3060, and `gpu_layers=99`; all 35 evidence assertions passed and guarded quit removed both processes |

These timings are smoke evidence, not a benchmark. Compare warmed, repeated
runs before making performance claims.

The synchronized Windows build used a short `C:\wl-target-ninja` Cargo target,
Ninja, and one build job. It completed without Rust warnings after making
Unix-only projection parameters explicitly consumed on non-Unix targets. Keep
Windows warning-free because CI runs Clippy with warnings denied; a successful
link alone is not the release gate.

## Troubleshooting

- `could not find any instance of Visual Studio`: the selected generator lacks
  the C++ workload. Install it and run from a matching developer environment.
- `add_custom_command DEPFILE is not supported by this generator`, or a
  missing nested `cmake_install.cmake`: run
  `scripts\setup-msvc-ninja-windows.ps1`; do not use a Visual Studio generator
  for the Vulkan build.
- `cannot open input file 'sqlite3.lib'`: install the vcpkg triplet above and
  prepend its `lib` directory to `LIB`.
- A test executable exits with `0xc0000135` / `STATUS_DLL_NOT_FOUND` after a
  successful link: add the directory containing the staged `onnxruntime.dll`
  to `PATH`. `ORT_DYLIB_PATH` pins the build/runtime choice but cannot satisfy
  a Windows process-start DLL import by itself.
- `Command 'perl' not found` or `Can't locate Locale/Maketext/Simple.pm` while
  building `openssl-sys`: install full Strawberry Perl, put its `perl\bin`
  before Git's `usr\bin`, and run the module probe above.
- `Unable to find Vulkan`: run `scripts\setup-vulkan-sdk-windows.ps1` in the
  same PowerShell session and verify `$env:VULKAN_SDK`.
- `vulkan-1.dll was not found` at process start: install or update the vendor
  GPU driver/Vulkan runtime. CPU mode cannot recover because Windows resolves
  this DLL before `main`.
- `C1083: Cannot open compiler generated file: '': Invalid argument` inside
  `vulkan-shaders-gen`: the nested CMake path crossed legacy MAX_PATH. Set
  `CARGO_TARGET_DIR=C:\wl-target`, then rebuild; enabling Windows long paths
  does not make every older MSVC/CMake child tool long-path aware.
- `C1041: cannot open program database`: concurrent nested MSVC probes wrote
  the same PDB. Set `CARGO_BUILD_JOBS=1` and pass `--jobs 1` to the release
  build. The Windows CI and release jobs intentionally use this slower,
  deterministic path.
- Vulkan builds but the daemon reports CPU: inspect `fallback_reason`, update
  the GPU driver, run `vulkaninfo.exe --summary`, then retry the live smoke.
- A hybrid laptop picks the integrated GPU: inspect the device indexes printed
  by `model_probe.exe`, then temporarily set `WENLAN_LLM_DEVICE=<index>` and
  attach the full device inventory to the issue.
