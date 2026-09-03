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
- `cargo-nextest` 0.9.x for the repository's impacted-test planner. The
  official Windows package is signed; install it through winget so local
  planner tests exercise the same `cargo nextest list/run` boundary as CI.
- LunarG Vulkan SDK 1.4.350.0. The repository setup script downloads the
  pinned official installer, verifies SHA-256, and uses LunarG's `copy_only=1`
  mode so it does not require Administrator access or write registry state.

From PowerShell:

```powershell
$env:RUSTUP_TOOLCHAIN = "1.95.0"
winget install --id Kitware.CMake --exact
winget install --id LLVM.LLVM --exact
winget install --id StrawberryPerl.StrawberryPerl --exact
winget install --id nextest.cargo-nextest --exact

git clone --depth 1 --branch 2026.06.24 https://github.com/microsoft/vcpkg.git "$env:LOCALAPPDATA\wenlan-build\vcpkg"
& "$env:LOCALAPPDATA\wenlan-build\vcpkg\bootstrap-vcpkg.bat" -disableMetrics
& "$env:LOCALAPPDATA\wenlan-build\vcpkg\vcpkg.exe" install sqlite3:x64-windows-static-md

& scripts\setup-vulkan-sdk-windows.ps1
$env:LIB = "$env:LOCALAPPDATA\wenlan-build\vcpkg\installed\x64-windows-static-md\lib;$env:LIB"
& scripts\setup-msvc-ninja-windows.ps1

# Pin the linker before crossing into Git Bash. Git for Windows also ships a
# coreutils link.exe, which is not the MSVC linker.
$sysroot = (rustc --print sysroot).Trim()
$lld = Join-Path $sysroot `
  "lib\rustlib\x86_64-pc-windows-msvc\bin\rust-lld.exe"
Get-Item $lld
$env:RUSTFLAGS = "-C linker=$lld -C linker-flavor=lld-link"

$gitBash = Join-Path $env:ProgramFiles "Git\bin\bash.exe"
Get-Item $gitBash

# llama.cpp's nested Vulkan shader build can exceed legacy MAX_PATH when the
# checkout is deep. Keep Cargo output on a deliberately short, fresh local
# path. Do not reuse a target previously configured with a Visual Studio
# generator after setup-msvc-ninja-windows.ps1 selects Ninja.
$env:CARGO_TARGET_DIR = "C:\wl-target"

# Cargo can overlap two independent package builds. Keep nested CMake at one
# worker so MSVC Vulkan shader probes cannot race while writing the same PDB.
$env:CARGO_BUILD_JOBS = "2"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "1"
```

If libclang is not on its standard path, set `LIBCLANG_PATH` to the directory
containing `libclang.dll`. The MSVC setup script enters the x64 developer
environment, selects Visual Studio's bundled Ninja, preserves the vcpkg
`LIB` prefix, and serializes nested CMake builds. Use it with both Visual
Studio 2019 and 2022: llama.cpp's Vulkan ExternalProject/shader rules are not
reliably ordered by the Visual Studio generators. Verify `perl
-MLocale::Maketext::Simple -e "print qq(ok\n)"` before building the server.
If the Strawberry Perl MSI requires elevation or stalls under a non-elevated
winget session, use the official 64-bit portable ZIP instead, point
`OPENSSL_SRC_PERL` at its `perl\bin\perl.exe`, and run the same module probe.

## Build and test

```powershell
$target = "x86_64-pc-windows-msvc"
$testRuntimeDir = Join-Path $env:CARGO_TARGET_DIR "test-runtime"
$releaseDir = Join-Path $env:CARGO_TARGET_DIR "$target\release"
& scripts\stage-onnxruntime-windows.ps1 `
  -DestinationDirectory $testRuntimeDir
& scripts\stage-vulkan-loader-windows.ps1 `
  -DestinationDirectory $testRuntimeDir
$env:ORT_DYLIB_PATH = Join-Path $testRuntimeDir "onnxruntime.dll"
$env:PATH = "$testRuntimeDir;$env:PATH"

cargo fmt --check --all
cargo test -p wenlan-types
cargo test -p wenlan-core --lib engine::tests
cargo test -p wenlan-server status_reports_selected_vulkan_device
# The shared build script launches each .exe immediately after linking. Stage
# both runtime DLLs beside that future output path before invoking the script.
& scripts\stage-onnxruntime-windows.ps1 `
  -DestinationDirectory $releaseDir
& scripts\stage-vulkan-loader-windows.ps1 `
  -DestinationDirectory $releaseDir
& $gitBash scripts\build-release-binaries.sh $target
cargo build --release --target $target --jobs 2 `
  -p wenlan-core --bin model_probe

& scripts\setup-vulkan-sdk-windows.test.ps1
& scripts\stage-vulkan-loader-windows.test.ps1
& scripts\setup-msvc-ninja-windows.test.ps1
& scripts\smoke-windows-llm.test.ps1
python scripts\release_targets.test.py
& $gitBash scripts\build-release-binaries.test.sh
python scripts\ci_test_plan.test.py
```

The Windows CI and release jobs run the same pinned Vulkan SDK setup before any
Cargo build. **The Vulkan SDK is a build-time prerequisite, and that much is
established:** `llama-cpp-sys-2`'s build script panics without `VULKAN_SDK`, so
no daemon crate compiles without it, and nothing from the SDK is redistributed
except the loader. Vulkan-enabled Windows executables have a process-start
dependency on `vulkan-1.dll`, so the Windows release archive ships the verified
official loader beside `wenlan-server.exe` and includes `VulkanRT-License.txt`.

### What is and is not established about end-user impact

**Measured.** `first-run-gauntlet.yml` downloads the published
`wenlan-windows-x64.zip` and the published NSIS installer, runs them on a
`windows-2025` hosted runner with no vendor GPU driver, and gets
`/api/health` (with a version assertion), `wenlan.exe status`, and
`wenlan.exe doctor`. Process start and the HTTP surface therefore work without
an ICD. That is the whole of the measurement.

**Unmeasured.** Whether a machine that has the loader but no vendor ICD can
load a GGUF and complete an inference at all. The CPU plan in
`crates/wenlan-core/src/engine.rs` is built from a live llama.cpp device list,
and that list is only reached *after* `shared_backend()` calls
`LlamaBackend::init()` (`engine.rs:33`). If backend init itself fails for want
of an ICD, `shared_backend()` returns `WenlanError::Llm("backend init: …")` and
neither the device enumeration nor the zero-GPU-layer reload is ever reached.
No CI job loads a real GGUF on a driverless Windows runner, and `/api/health`
answers before any model is loaded, so it cannot stand in for one. The CPU
fallback on a driverless machine is supported by code reading, not by an
executed gate.

Do not write that users are unaffected, and do not write that they are
affected — neither has been measured.

**The gate that would settle it.** On a clean Windows VM or Windows Sandbox
with no vendor ICD installed: install from the real signed installer or the
release ZIP, load `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, complete one
inference, and assert `/api/status` reports `backend` = `cpu` and
`gpu_layers` = `0`. Until that run exists and its result is recorded in this
section, the driverless end-user path stays unproven. GPU execution separately
requires a working vendor ICD from the GPU driver; that part is not in
dispute.

### CI and release loader staging

GitHub's hosted Windows runner has no vendor GPU driver and therefore cannot be
assumed to have the loader. CI downloads LunarG's pinned
`vulkan-runtime-components.zip` for `1.4.350.0`, verifies archive SHA-256
`23ce69f32cef3e2799617e2b1776cd0c71030d23a91f8375821cc40d76b185b9`
and x64 loader SHA-256
`0419974f00e82a3d619077ba414da265a774f8db9d45ad93bc1843f44b2c2c1f`,
checks the loader's LunarG Authenticode signer, and copies its accompanying
license. Test jobs put that extracted directory on the job-scoped `PATH` and
also stage the verified loader beside `target\debug\wenlan-server.exe` for the
Task Scheduler round-trip; a scheduled process must not depend on the workflow
shell's SDK path. Release jobs stage the loader beside the executables and
include both files in the zip. Nothing writes `System32` or the registry. This
lets CPU-only tests and released binaries **start** on a vendorless machine —
process start is what was measured, not model load — and it is not GPU
evidence. The physical smoke below remains the Vulkan execution proof, and it
ran on a machine with working drivers; the driverless CPU-inference gate named
above is still unrun.

The implementation and CI contract were checked against these primary sources:

- [LunarG Windows SDK guide](https://vulkan.lunarg.com/doc/view/latest/windows/getting_started.html)
  for the SDK/driver/loader boundary;
- [Khronos Vulkan Loader](https://github.com/KhronosGroup/Vulkan-Loader)
  for the loader's role between an application and ICDs, and its
  [Apache-2.0 redistribution terms](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/LICENSE.txt);
- [Microsoft DLL search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)
  and [GitHub `GITHUB_PATH`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-system-path)
  for process-start lookup and job scoping;
- [Microsoft command-line MSVC setup](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line)
  and [CMake generators](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)
  for the imported x64 developer environment and Ninja;
- [llama.cpp Vulkan build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan)
  for `GGML_VULKAN` and GPU-layer verification; and
- [ONNX Runtime Windows deployment](https://onnxruntime.ai/docs/get-started/with-c.html#deployment)
  for keeping the verified runtime adjacent to shipped executables.

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

All three recoveries were measured on 2026-07-25 on a machine with a working
vendor ICD (see the verified physical result below). They live **downstream of
`LlamaBackend::init()`** and downstream of device enumeration, so they say
nothing about a machine with no ICD at all: if backend init fails, this policy
never runs and there is no `fallback_reason` to report. That case is unproven —
see the driverless gate under "What is and is not established about end-user
impact".

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
$target = "x86_64-pc-windows-msvc"
$releaseDir = Join-Path $env:CARGO_TARGET_DIR "$target\release"
$model = "$env:USERPROFILE\.cache\wenlan\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$probe = Join-Path $releaseDir "model_probe.exe"

# Auto policy must select the discrete NVIDIA adapter on a mixed-GPU machine.
& scripts\smoke-windows-llm.ps1 `
  -ModelPath $model `
  -ProbePath $probe `
  -Device auto `
  -ExpectedBackend vulkan `
  -ExpectedDevicePattern "NVIDIA.*RTX 3060"

# CPU remains a supported, deterministic escape hatch.
& scripts\smoke-windows-llm.ps1 `
  -ModelPath $model `
  -ProbePath $probe `
  -Device cpu `
  -ExpectedBackend cpu

# Inject an unavailable device selection and prove visible CPU recovery.
& scripts\smoke-windows-llm.ps1 `
  -ModelPath $model `
  -ProbePath $probe `
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

The 2026-07-25 physical run used Windows 11, an Intel Iris Xe integrated GPU,
an NVIDIA GeForce RTX 3060 Laptop discrete GPU, Vulkan SDK 1.4.350.0, Visual
Studio Build Tools, and
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2,497,281,120 bytes, SHA-256
`3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`).
The post-`origin/main` implementation baseline was backend code commit
`fc0e9ba` (before this documentation-only update). The source-built,
rust-lld-linked `wenlan-server.exe` SHA-256 was
`9391b98fa411ceb573fea0c495fbbf83083e0d979b050868e1d07d801d94e209`.
The adjacent staged loader was the pinned LunarG `vulkan-1.dll` with SHA-256
`0419974f00e82a3d619077ba414da265a774f8db9d45ad93bc1843f44b2c2c1f`;
the process module inventory resolved that exact path, not a system loader.

| Leg | Observed result |
|---|---|
| `auto`, expected Vulkan | Selected llama.cpp device `1`, `NVIDIA GeForce RTX 3060 Laptop GPU`; offloaded `37/37` layers; valid classification in 1.35 seconds; 14 `nvidia-smi` samples peaked at 95% utilization and 3272 MiB |
| `cpu`, expected CPU | Offloaded `0/37` layers; all KV layers reported `dev = CPU`; Vulkan1 device compute allocation `0.0000 MiB`; valid classification in 12.19 seconds |
| device `99`, expected fallback | Reported `requested GPU device index 99 is unavailable`; offloaded `0/37` layers and used the same CPU-only context contract; valid classification in 11.80 seconds |
| status route | `routes::recent_endpoints_tests::status_reports_selected_vulkan_device` passed |
| backend daemon smoke | Stored one source, returned a vector-only semantic-search hit, and loaded the exact adjacent ONNX Runtime and Vulkan loader modules |
| app-owned daemon | App-side changes are now PRs in this repository under `app/`; the app-owned daemon leg must record its branch-tip `result.json` from that same PR/branch. A historical app run or this model probe alone is not accepted as current app evidence |

These timings are smoke evidence, not a benchmark. Compare warmed, repeated
runs before making performance claims.

Treat this dated section as a reproducible baseline, not a substitute for the
current PR gate. Before merge, rebuild from the current backend tip and require
the companion app sidecar manifest, daemon health commit, executable hashes,
loaded-module paths, UI marker, and `result.json` to agree with that tip. Record
the exact final hashes in the PR evidence instead of rewriting this runbook
after every docs-only commit.

The synchronized Windows build used a short `C:\wl-target-ninja` Cargo target,
Ninja, and one build job. It completed without Rust warnings after making
Unix-only projection parameters explicitly consumed on non-Unix targets. Keep
Windows warning-free because CI runs Clippy with warnings denied; a successful
link alone is not the release gate.

## Troubleshooting

- `could not find any instance of Visual Studio`: the selected generator lacks
  the C++ workload. Install it and run from a matching developer environment.
- Plain `bash` resolves to `C:\Windows\System32\bash.exe`: that is the WSL
  launcher, not Git Bash. Invoke
  `C:\Program Files\Git\bin\bash.exe` explicitly for repository Bash scripts.
- Cargo reports `C:\Program Files\Git\usr\bin\link.exe` and coreutils says
  `missing operand`: Git Bash's `link.exe` shadowed the Windows linker. Export
  the `rust-lld` `RUSTFLAGS` shown above before starting the Bash build. CI and
  tag release use the same explicit linker.
- `add_custom_command DEPFILE is not supported by this generator`, or a
  missing nested `cmake_install.cmake`: run
  `scripts\setup-msvc-ninja-windows.ps1`; do not use a Visual Studio generator
  for the Vulkan build.
- `CMake project was already configured` followed by
  `MSB1009: Project file does not exist ... install.vcxproj`: the Cargo target
  contains a CMake cache created by a different generator. Point
  `CARGO_TARGET_DIR` at a new short directory before rebuilding (for example
  `C:\wl-target-ninja`). Do not mix Visual Studio and Ninja artifacts in one
  target directory.
- `cannot open input file 'sqlite3.lib'`: install the vcpkg triplet above and
  prepend its `lib` directory to `LIB`.
- A test executable exits with `0xc0000135` / `STATUS_DLL_NOT_FOUND` after a
  successful link: do not infer the missing DLL from the exit code. Run
  `dumpbin /DEPENDENTS <test.exe>`. The Vulkan-enabled test binary imports
  `vulkan-1.dll` before Rust starts; vendorless CI must run
  `scripts\stage-vulkan-loader-windows.ps1`. `ORT_DYLIB_PATH` separately pins
  the verified ONNX Runtime, whose directory is also job-scoped in CI. If a
  scheduled task reports decimal result `-1073741515`, stage both verified
  runtime DLLs beside the scheduled `wenlan-server.exe`; the task's `Start In`
  directory and inherited `PATH` are not runtime-distribution contracts. CI
  and tag release likewise stage both DLLs before the shared build script,
  because that script launches the shipped executables immediately after link.
- `ort ... is not compatible ... expected version >= '1.23.x'`: the process
  found a stale `onnxruntime.dll` (for example 1.17.1). Run
  `scripts\stage-onnxruntime-windows.ps1`, set `ORT_DYLIB_PATH` to that exact
  1.23.2 DLL, and put the staged directory first on the current process
  `PATH` before invoking Cargo. Keep the full MSVC/LLVM/Vulkan environment in
  that same shell because changing runtime environment inputs can make Cargo
  rerun native build scripts.
- `Command 'perl' not found` or `Can't locate Locale/Maketext/Simple.pm` while
  building `openssl-sys`: install full Strawberry Perl, put its `perl\bin`
  before Git's `usr\bin`, and run the module probe above.
- `Unable to find Vulkan`: run `scripts\setup-vulkan-sdk-windows.ps1` in the
  same PowerShell session and verify `$env:VULKAN_SDK`.
- `vulkan-1.dll was not found` at process start: a release installation is
  incomplete; re-extract the Windows zip and confirm `vulkan-1.dll` and
  `VulkanRT-License.txt` sit beside `wenlan-server.exe`. For a local build, run
  `scripts\stage-vulkan-loader-windows.ps1` against its release directory.
  Update the vendor driver separately when Vulkan GPU enumeration fails.
- `C1083: Cannot open compiler generated file: '': Invalid argument` inside
  `vulkan-shaders-gen`: the nested CMake path crossed legacy MAX_PATH. Set
  `CARGO_TARGET_DIR=C:\wl-target`, then rebuild; enabling Windows long paths
  does not make every older MSVC/CMake child tool long-path aware.
- `C1041: cannot open program database`: concurrent nested MSVC probes wrote
  the same PDB. Run `scripts\setup-msvc-ninja-windows.ps1` and keep
  `CMAKE_BUILD_PARALLEL_LEVEL=1`. Windows CI and release allow two outer Cargo
  jobs, but they deliberately do not lift this nested CMake/PDB protection.
- Vulkan builds but the daemon reports CPU: inspect `fallback_reason`, update
  the GPU driver, run `vulkaninfo.exe --summary`, then retry the live smoke.
- A hybrid laptop picks the integrated GPU: inspect the device indexes printed
  by `model_probe.exe`, then temporarily set `WENLAN_LLM_DEVICE=<index>` and
  attach the full device inventory to the issue.
