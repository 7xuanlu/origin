#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/scripts/profile-ambient-rb01.sh"

first_preflight_line="$(
  /usr/bin/awk '/^preflight$/ { print NR; exit }' "${TARGET}"
)"
post_build_admission_line="$(
  /usr/bin/awk '/^wait_for_post_build_admission$/ { print NR; exit }' "${TARGET}"
)"
build_line="$(
  /usr/bin/awk '/^[[:space:]]*rtk proxy cargo test --locked --offline -j 2 -p wenlan-server rb01_profile_admission --lib --no-run --message-format=json / { print NR }' \
    "${TARGET}"
)"
inference_line="$(
  /usr/bin/awk '/scheduler::tests::rb01_profile_real_on_device_slice/ { print NR; exit }' \
    "${TARGET}"
)"
freeze_test_line="$(
  /usr/bin/awk '/cp "\$\{test_binary\}" "\$\{frozen_test_binary\}"/ { print NR; exit }' \
    "${TARGET}"
)"
publish_test_line="$(
  /usr/bin/awk '/cp "\$\{frozen_test_binary\}" "\$\{test_binary\}"/ { print NR; exit }' \
    "${TARGET}"
)"
final_verify_line="$(
  /usr/bin/awk '/^verify_frozen_binaries$/ { latest=NR } END { print latest }' \
    "${TARGET}"
)"

if [[ -z "${first_preflight_line}" || -z "${post_build_admission_line}" ]]; then
  echo "expected an immediate initial preflight and bounded post-build admission" >&2
  exit 1
fi

if [[ -z "${build_line}" || -z "${inference_line}" ||
      -z "${freeze_test_line}" || -z "${publish_test_line}" ||
      -z "${final_verify_line}" ]]; then
  echo "could not locate the bounded build or real inference command" >&2
  exit 1
fi

lock_line="$(
  /usr/bin/awk '/^acquire_profile_lock$/ { print NR; exit }' "${TARGET}"
)"
if [[ -z "${lock_line}" ]] || ! (( lock_line < first_preflight_line )); then
  echo "expected an exclusive profile lock before the first real preflight" >&2
  exit 1
fi

measured_cooldown_line="$(
  /usr/bin/awk '/^  cooldown_from_report .* \|\| true$/ { print NR; exit }' "${TARGET}"
)"
if [[ -z "${measured_cooldown_line}" ]] ||
  ! /usr/bin/grep -Fq '"rb01_recovery_known"' "${TARGET}" ||
  ! /usr/bin/grep -Fq '"rb01_recovery_known"' \
    "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs"; then
  echo "expected success or an explicit completed-work marker to narrow the fail-safe cooldown" >&2
  exit 1
fi

if ! /usr/bin/grep -Fq 'wenlan_server::scheduler=debug' \
  "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs" ||
  [[ "$(/usr/bin/grep -Fc 'emit_recovery_report(' \
    "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs")" -lt 3 ]]; then
  echo "expected live timeout evidence to preserve admission reasons and measured recovery" >&2
  exit 1
fi

THERMAL_HELPER_SOURCE="${ROOT}/scripts/rb01-thermal-state.m"
if [[ ! -f "${THERMAL_HELPER_SOURCE}" ]] ||
  /usr/bin/grep -Fq '/usr/bin/swift' "${TARGET}" ||
  /usr/bin/grep -Fq '/usr/bin/swift' \
    "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs" ||
  ! /usr/bin/grep -Fq 'compile_thermal_helper' "${TARGET}" ||
  ! /usr/bin/grep -Fq 'WENLAN_RB01_THERMAL_HELPER' "${TARGET}" ||
  ! /usr/bin/grep -Fq 'WENLAN_RB01_THERMAL_HELPER' \
    "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs"; then
  echo "expected a frozen low-overhead thermal helper instead of repeated Swift JIT probes" >&2
  exit 1
fi

if ! /usr/bin/grep -q '"rss_peak_during_slice_bytes"' \
  "${ROOT}/crates/wenlan-server/src/scheduler.rs"; then
  echo "expected the real profile JSON to report observed peak slice RSS" >&2
  exit 1
fi

if ! (( first_preflight_line < build_line &&
        build_line < freeze_test_line &&
        freeze_test_line < publish_test_line &&
        publish_test_line < post_build_admission_line &&
        publish_test_line < final_verify_line &&
        post_build_admission_line < final_verify_line &&
        final_verify_line < inference_line &&
        build_line < post_build_admission_line &&
        post_build_admission_line < inference_line )); then
  echo "unsafe order: expected preflight -> build -> freeze -> publish -> bounded admission -> verify -> inference" >&2
  exit 1
fi

headroom_calls="$(
  /usr/bin/awk '/^  check_model_headroom( |$)/ { count += 1 } END { print count + 0 }' \
    "${TARGET}"
)"
if [[ "${headroom_calls}" -ne 1 ]]; then
  echo "expected preflight to enforce model working-set headroom exactly once" >&2
  exit 1
fi

if ! /usr/bin/grep -q '"report_elapsed_ms"' "${TARGET}" ||
  ! /usr/bin/grep -q 'slice_seconds \* 19' "${TARGET}" ||
  ! /usr/bin/grep -q 'cooldown_seconds < 120' "${TARGET}" ||
  ! /usr/bin/grep -q 'cooldown_seconds=120' "${TARGET}" ||
  ! /usr/bin/grep -q '^FAILSAFE_COOLDOWN_SECS=5700$' "${TARGET}" ||
  ! /usr/bin/grep -q '^POST_BUILD_ADMISSION_WAIT_SECS=1800$' "${TARGET}" ||
  ! /usr/bin/grep -q 'local cooldown_seconds="${FAILSAFE_COOLDOWN_SECS}"' "${TARGET}"; then
  echo "expected measured cooldown, fixed fail-safe, and bounded post-build admission" >&2
  exit 1
fi

if ! /usr/bin/grep -q '^trap - INT TERM EXIT$' "${TARGET}" ||
  ! /usr/bin/grep -q '^exit "${status}"$' "${TARGET}"; then
  echo "expected the real profile to finalize once and propagate the child status" >&2
  exit 1
fi

process_group_calls="$(
  /usr/bin/grep -Fc 'setpgrp(0, 0)' "${TARGET}" || true
)"
group_termination_calls="$(
  /usr/bin/grep -Ec '^[[:space:]]*terminate_child_group$' "${TARGET}" || true
)"
if [[ "${process_group_calls}" -ne 3 ||
      "${group_termination_calls}" -lt 2 ]] ||
  ! /usr/bin/grep -Fq '/bin/kill -TERM -- "-${child_pgid}"' "${TARGET}" ||
  ! /usr/bin/grep -Fq '/bin/kill -KILL -- "-${child_pgid}"' "${TARGET}" ||
  ! /usr/bin/grep -Fq 'process_group_cleanup_failed=1' "${TARGET}" ||
  ! /usr/bin/grep -Fq 'status=1' "${TARGET}"; then
  echo "expected timeout and interrupt cleanup to fail closed after terminating the isolated child process group" >&2
  exit 1
fi

for provenance_key in \
  source_scheduler_sha256 \
  source_db_sha256 \
  source_profiler_sha256 \
  source_live_daemon_harness_sha256 \
  source_backlog_audit_sha256 \
  source_backlog_result_sha256 \
  source_bundle_sha256 \
  cargo_lock_sha256 \
  test_binary_sha256 \
  daemon_binary_sha256 \
  model_blob_sha256 \
  fastembed_model_blob_sha256 \
  model_resolved_path \
  git_worktree_diff_sha256; do
  if ! /usr/bin/grep -q "^  echo \"${provenance_key}=" "${TARGET}"; then
    echo "expected manifest provenance key ${provenance_key}" >&2
    exit 1
  fi
done

if ! /usr/bin/grep -Fq 'WENLAN_RB01_RUN_DIR="${artifact_dir}/daemon-data"' \
  "${TARGET}" ||
  ! /usr/bin/grep -q 'std::env::var("WENLAN_RB01_RUN_DIR")' \
    "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs"; then
  echo "expected daemon evidence DB to live in the persistent external run directory" >&2
  exit 1
fi

if ! /usr/bin/grep -q '\.env_clear()' \
  "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs" ||
  ! /usr/bin/grep -q 'daemon_binary_sha256' "${TARGET}" ||
  ! /usr/bin/grep -Fq \
    'WENLAN_RB01_DAEMON_BINARY="${daemon_binary}"' "${TARGET}" ||
  ! /usr/bin/grep -Fq \
    'WENLAN_RB01_DAEMON_SHA256="${daemon_binary_sha256}"' "${TARGET}" ||
  ! /usr/bin/grep -q 'std::env::var("WENLAN_RB01_DAEMON_SHA256")' \
    "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs"; then
  echo "expected an environment-isolated daemon and revalidated frozen daemon provenance" >&2
  exit 1
fi

if ! /usr/bin/grep -q -- '--message-format=json' "${TARGET}" ||
  ! /usr/bin/grep -q 'rtk proxy "${test_binary}"' "${TARGET}"; then
  echo "expected the measured child to be the hashed Cargo test executable" >&2
  exit 1
fi

if /usr/bin/grep -q '/usr/bin/jq' "${TARGET}" ||
  ! /usr/bin/grep -q '^JQ_BIN="$(command -v jq || true)"$' "${TARGET}" ||
  ! /usr/bin/grep -q '^  "${JQ_BIN}" -r ' "${TARGET}"; then
  echo "expected jq to be discovered from PATH instead of a Homebrew-incompatible fixed path" >&2
  exit 1
fi

if ! /usr/bin/grep -q '^    rtk proxy git diff --binary HEAD$' "${TARGET}" ||
  ! /usr/bin/grep -q '^    rtk proxy git diff --binary --cached HEAD$' "${TARGET}"; then
  echo "expected provenance to hash exact unfiltered git diff bytes" >&2
  exit 1
fi

if ! /usr/bin/grep -Fq \
  'rtk proxy git ls-files --others --exclude-standard -z' \
  "${TARGET}" ||
  ! /usr/bin/grep -Fq '/bin/cat -- "${path}"' "${TARGET}" ||
  ! /usr/bin/grep -Fq 'source_bundle_sha256="$(worktree_source_bundle_sha256)"' \
    "${TARGET}"; then
  echo "expected a canonical source fingerprint that includes every untracked file byte" >&2
  exit 1
fi

if ! /usr/bin/grep -Fq \
  'artifact_dir="${PROFILE_DATA_ROOT}/wenlan/benchmarks/${git_head}-${source_bundle_sha256:0:12}/${stamp}-${lane}"' \
  "${TARGET}" ||
  ! /usr/bin/grep -Fq \
    '[[ "$(sha256_file "${test_binary}")" != "${test_binary_sha256}" ]]' \
    "${TARGET}" ||
  ! /usr/bin/grep -Fq \
    '[[ "$(sha256_file Cargo.lock)" != "${cargo_lock_sha256}" ]]' \
    "${TARGET}"; then
  echo "expected SHA-keyed artifacts and a final actual-exec/lockfile verification" >&2
  exit 1
fi

if ! /usr/bin/grep -q '^  document|entity|page-growth|reconcile|citation|daemon)$' "${TARGET}"; then
  echo "expected the real profiler CLI to include the five slice lanes and live daemon lane" >&2
  exit 1
fi

if /usr/bin/grep -q '/usr/bin/find -L "${MODEL_CACHE}"' "${TARGET}" ||
  ! /usr/bin/grep -q 'MODEL_REPO_ROOT=.*MODEL_CACHE%/snapshots' "${TARGET}" ||
  ! /usr/bin/grep -q 'MODEL_REF_FILE="${MODEL_REPO_ROOT}/refs/main"' "${TARGET}"; then
  echo "expected the profiler to require the exact hf-hub main ref cache hit" >&2
  exit 1
fi

if ! /usr/bin/grep -Fq \
  'FASTEMBED_CACHE="${HOME}/Library/Application Support/wenlan/memorydb/fastembed_cache"' \
  "${TARGET}" ||
  ! /usr/bin/grep -Fq \
    'FASTEMBED_REPO_ROOT="${FASTEMBED_CACHE}/models--Qdrant--bge-base-en-v1.5-onnx-Q"' \
    "${TARGET}" ||
  ! /usr/bin/grep -Fq \
    'FASTEMBED_REF_FILE="${FASTEMBED_REPO_ROOT}/refs/main"' \
    "${TARGET}" ||
  ! /usr/bin/grep -q '^  check_fastembed_cache || return 1$' "${TARGET}"; then
  echo "expected preflight to require the exact production FastEmbed cache snapshot" >&2
  exit 1
fi

fastembed_env_calls="$(
  /usr/bin/grep -Fc \
    'WENLAN_TEST_FASTEMBED_CACHE="${FASTEMBED_CACHE}"' \
    "${TARGET}"
)"
fail_closed_endpoint_calls="$(
  /usr/bin/grep -Fc \
    'HF_ENDPOINT="http://127.0.0.1:9"' \
    "${TARGET}"
)"
unset_hf_home_calls="$(
  /usr/bin/grep -Fc \
    '/usr/bin/env -u HF_HOME' \
    "${TARGET}"
)"
if [[ "${fastembed_env_calls}" -ne 2 ||
      "${fail_closed_endpoint_calls}" -ne 2 ||
      "${unset_hf_home_calls}" -ne 2 ]]; then
  echo "expected both real child lanes to pin FastEmbed and fail closed on cache miss" >&2
  exit 1
fi

if ! /usr/bin/grep -q \
  'std::env::var("WENLAN_TEST_FASTEMBED_CACHE")' \
  "${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs"; then
  echo "expected the live daemon harness to refuse an unpinned FastEmbed cache" >&2
  exit 1
fi

if ! /usr/bin/grep -q 'ambient_live_daemon::persistent_provider_respects_production_cooldown' \
  "${TARGET}"; then
  echo "expected daemon mode to invoke the ignored production-timing integration test" >&2
  exit 1
fi

if ! /usr/bin/grep -q 'calibration_no_inference_recovered' "${TARGET}" ||
  ! /usr/bin/grep -q 'rb01_calibration_no_inference' "${TARGET}" ||
  ! /usr/bin/grep -q 'rb01_calibration_skipped' "${TARGET}"; then
  echo "expected calibration runs with no inference to release the fail-safe only after resource recovery" >&2
  exit 1
fi

LIVE_HARNESS="${ROOT}/crates/wenlan-server/tests/ambient_live_daemon.rs"
safety_sample_calls="$(
  /usr/bin/grep -Ec '^[[:space:]]*sample_safety_if_due\(' "${LIVE_HARNESS}" || true
)"
if ! /usr/bin/grep -Fq \
  'const SAFETY_SAMPLE_INTERVAL: Duration = Duration::from_secs(30);' \
  "${LIVE_HARNESS}" ||
  [[ "${safety_sample_calls}" -lt 2 ]] ||
  ! /usr/bin/grep -Fq \
    'const MIN_MID_COOLDOWN_RESIDENCY_CHECKS: usize = 1;' \
    "${LIVE_HARNESS}" ||
  ! /usr/bin/grep -Fq \
    'mid_cooldown_residency_checks.len() >= MIN_MID_COOLDOWN_RESIDENCY_CHECKS' \
    "${LIVE_HARNESS}" ||
  ! /usr/bin/grep -Fq \
    'cooldown_gap >= MIN_PRODUCTION_COOLDOWN' \
    "${LIVE_HARNESS}"; then
  echo "expected the live daemon proof to fail closed throughout load/run and sample provider residency during the exact cooldown" >&2
  exit 1
fi

start_log_line="$(
  /usr/bin/awk 'index($0, "ambient turn started job=") { print NR; exit }' \
    "${ROOT}/crates/wenlan-server/src/scheduler.rs"
)"
ambient_call_line="$(
  /usr/bin/awk 'index($0, "let report = run_ambient_job_safe(") { print NR; exit }' \
    "${ROOT}/crates/wenlan-server/src/scheduler.rs"
)"
completion_log_line="$(
  /usr/bin/awk 'index($0, "ambient job={:?} selected=") { print NR; exit }' \
    "${ROOT}/crates/wenlan-server/src/scheduler.rs"
)"
if [[ -z "${start_log_line}" || -z "${ambient_call_line}" || -z "${completion_log_line}" ]] ||
  ! (( start_log_line < ambient_call_line &&
        ambient_call_line < completion_log_line )); then
  echo "expected ambient start -> awaited slice -> completion telemetry ordering" >&2
  exit 1
fi

# Positive control the exact final provenance verifier: all frozen executables
# pass first, then changing either helper or daemon must make it reject.
if ! (
  source "${TARGET}"
  lane="daemon"
  provenance_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-provenance.XXXXXX)"
  trap '/bin/rm -rf "${provenance_probe_dir}"' EXIT
  test_binary="${provenance_probe_dir}/test-binary"
  daemon_binary="${provenance_probe_dir}/daemon-binary"
  thermal_helper="${provenance_probe_dir}/thermal-helper"
  /bin/cp /usr/bin/true "${test_binary}"
  /bin/cp /usr/bin/true "${daemon_binary}"
  /bin/cp /usr/bin/true "${thermal_helper}"
  test_binary_sha256="$(sha256_file "${test_binary}")"
  daemon_binary_sha256="$(sha256_file "${daemon_binary}")"
  thermal_helper_sha256="$(sha256_file "${thermal_helper}")"
  cargo_lock_sha256="$(sha256_file Cargo.lock)"
  source_bundle_sha256="$(worktree_source_bundle_sha256)"
  source_scheduler_sha256="$(sha256_file crates/wenlan-server/src/scheduler.rs)"
  source_db_sha256="$(sha256_file crates/wenlan-core/src/db.rs)"
  source_profiler_sha256="$(sha256_file scripts/profile-ambient-rb01.sh)"
  source_live_daemon_harness_sha256="$(
    sha256_file crates/wenlan-server/tests/ambient_live_daemon.rs
  )"
  source_thermal_helper_sha256="$(sha256_file scripts/rb01-thermal-state.m)"
  source_backlog_audit_sha256="$(sha256_file scripts/audit-ambient-rb01-backlog.sh)"
  source_backlog_result_sha256="$(
    sha256_file docs/eval/results/ambient-rb01-backlog-2026-07-20.json
  )"
  verify_frozen_binaries
  /usr/bin/printf 'tampered' >>"${thermal_helper}"
  if verify_frozen_binaries 2>/dev/null; then
    exit 1
  fi
  /bin/cp /usr/bin/true "${thermal_helper}"
  thermal_helper_sha256="$(sha256_file "${thermal_helper}")"
  /usr/bin/printf 'tampered' >>"${daemon_binary}"
  if verify_frozen_binaries 2>/dev/null; then
    exit 1
  fi
); then
  echo "frozen-binary replacement positive control failed" >&2
  exit 1
fi

# Positive control the post-build admission policy. CPU contention is the only
# retryable condition; all other safety failures must remain immediate.
if ! (
  source "${TARGET}"
  admission_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-admission.XXXXXX)"
  trap '/bin/rm -rf "${admission_probe_dir}"' EXIT
  attempts_file="${admission_probe_dir}/attempts"
  echo 0 >"${attempts_file}"
  POST_BUILD_ADMISSION_WAIT_SECS=5
  run_preflight_attempt_until() {
    local attempts
    attempts="$(/bin/cat "${attempts_file}")"
    attempts=$((attempts + 1))
    echo "${attempts}" >"${attempts_file}"
    if (( attempts == 1 )); then
      preflight_attempt_output="refusing profile: aggregate CPU sample 1/2 was 21.0% (>20%)"
      return 1
    fi
    preflight_attempt_output="preflight_ok thermal_state=0 memory_free_percent=50 cpu_percent_samples=10,10"
  }
  admission_output="$(wait_for_post_build_admission 2>&1)"
  [[ "$(/bin/cat "${attempts_file}")" == "2" ]] || exit 1
  /usr/bin/grep -q 'waiting for post-build CPU admission' <<<"${admission_output}" || exit 1
  /usr/bin/grep -q '^preflight_ok ' <<<"${admission_output}" || exit 1
); then
  echo "post-build CPU retry positive control failed" >&2
  exit 1
fi

if ! (
  source "${TARGET}"
  admission_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-admission.XXXXXX)"
  trap '/bin/rm -rf "${admission_probe_dir}"' EXIT
  attempts_file="${admission_probe_dir}/attempts"
  echo 0 >"${attempts_file}"
  POST_BUILD_ADMISSION_WAIT_SECS=5
  run_preflight_attempt_until() {
    local attempts
    attempts="$(/bin/cat "${attempts_file}")"
    echo $((attempts + 1)) >"${attempts_file}"
    preflight_attempt_output="refusing profile: macOS thermal state is 1, expected 0 (nominal)"
    return 1
  }
  if wait_for_post_build_admission >/dev/null 2>&1; then
    exit 1
  fi
  [[ "$(/bin/cat "${attempts_file}")" == "1" ]] || exit 1
); then
  echo "post-build non-CPU fail-closed positive control failed" >&2
  exit 1
fi

if ! (
  source "${TARGET}"
  admission_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-admission.XXXXXX)"
  trap '/bin/rm -rf "${admission_probe_dir}"' EXIT
  preflight_state_file="${admission_probe_dir}/state"
  /usr/bin/printf '%s\n' \
    'verify_frozen_binaries() { return 0; }' \
    'preflight() { /bin/sleep 30; }' \
    >"${preflight_state_file}"
  POST_BUILD_ADMISSION_TERMINATION_GRACE_SECS=1
  started="$(date +%s)"
  deadline=$((started + 3))
  if run_sourced_preflight_attempt_until "${deadline}" "${preflight_state_file}"; then
    exit 1
  else
    status=$?
  fi
  finished="$(date +%s)"
  elapsed=$((finished - started))
  [[ "${status}" == "124" ]] || exit 1
  (( elapsed <= 4 )) || exit 1
  [[ "${preflight_attempt_output}" == *"post-build admission deadline reached"* ]] || exit 1
  [[ -z "${child_pid}" && -z "${child_pgid}" && -z "${preflight_watchdog_pid}" ]] || exit 1
); then
  echo "post-build hard-deadline positive control failed" >&2
  exit 1
fi

if ! (
  source "${TARGET}"
  admission_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-admission.XXXXXX)"
  trap '/bin/rm -rf "${admission_probe_dir}"' EXIT
  attempts_file="${admission_probe_dir}/attempts"
  echo 0 >"${attempts_file}"
  POST_BUILD_ADMISSION_WAIT_SECS=5
  run_preflight_attempt_until() {
    local attempts
    attempts="$(/bin/cat "${attempts_file}")"
    attempts=$((attempts + 1))
    echo "${attempts}" >"${attempts_file}"
    preflight_attempt_output="refusing profile: aggregate CPU sample 1/2 was 21.0% (>20%)"$'\n'"refusing profile: source bytes changed after the measured build"
    return 1
  }
  if wait_for_post_build_admission >"${admission_probe_dir}/output" 2>&1; then
    exit 1
  fi
  [[ "$(/bin/cat "${attempts_file}")" == "1" ]] || exit 1
  /usr/bin/grep -q 'source bytes changed after the measured build' \
    "${admission_probe_dir}/output" || exit 1
); then
  echo "post-build source-mutation positive control failed" >&2
  exit 1
fi

if ! (
  source "${TARGET}"
  admission_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-admission.XXXXXX)"
  runner_state="${admission_probe_dir}/interrupt-state"
  attempt_pid_file="${admission_probe_dir}/attempt-pid"
  if ! {
    printf 'attempt_pid_file=%q\n' "${attempt_pid_file}"
    /usr/bin/printf '%s\n' \
      'verify_frozen_binaries() { return 0; }' \
      'preflight() { echo "$$" >"${attempt_pid_file}"; /bin/sleep 30; }'
  } >"${runner_state}"; then
    exit 1
  fi
  /bin/bash -c '
    source "$1"
    preflight_state_file="$2"
    trap cleanup INT TERM EXIT
    run_sourced_preflight_attempt_until "$(($(date +%s) + 30))" "${preflight_state_file}"
  ' bash "${TARGET}" "${runner_state}" >/dev/null 2>&1 &
  runner_pid=$!
  attempt_pid=""
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if [[ -s "${attempt_pid_file}" ]]; then
      attempt_pid="$(/bin/cat "${attempt_pid_file}")"
      break
    fi
    /bin/sleep 0.05
  done
  [[ "${attempt_pid}" =~ ^[0-9]+$ ]] || exit 1
  /bin/kill -TERM "${runner_pid}" || exit 1
  wait "${runner_pid}" 2>/dev/null || true
  /bin/sleep 0.1
  if /bin/kill -0 -- "-${attempt_pid}" 2>/dev/null; then
    exit 1
  fi
  /bin/rm -f "${attempt_pid_file}" "${runner_state}"
  /bin/rmdir "${admission_probe_dir}" 2>/dev/null || true
); then
  echo "post-build interrupt cleanup positive control failed" >&2
  exit 1
fi

# Positive control the exact macOS process-group primitive used by the
# profiler. The child shell creates a grandchild so this proves group cleanup,
# not merely termination of one PID.
if ! /bin/bash -c '
  probe_pid=""
  cleanup_probe() {
    if [[ -n "${probe_pid}" ]]; then
      /bin/kill -KILL -- "-${probe_pid}" 2>/dev/null || true
      /bin/kill -KILL "${probe_pid}" 2>/dev/null || true
      wait "${probe_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup_probe EXIT
  (
    /usr/bin/perl -e \
      '"'"'defined(setpgrp(0, 0)) or die "setpgrp: $!"; exec @ARGV'"'"' \
      /bin/sh -c "/bin/sleep 30 & wait"
  ) &
  probe_pid=$!
  probe_group_ready=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if /bin/kill -0 -- "-${probe_pid}" 2>/dev/null; then
      probe_group_ready=1
      break
    fi
    /bin/sleep 0.01
  done
  (( probe_group_ready == 1 )) || exit 1
  /bin/kill -TERM -- "-${probe_pid}"
  wait "${probe_pid}" 2>/dev/null || true
  /bin/sleep 0.01
  ! /bin/kill -0 -- "-${probe_pid}" 2>/dev/null || exit 1
  probe_pid=""
  trap - EXIT
' >/dev/null 2>&1; then
  echo "process-group positive control failed" >&2
  exit 1
fi

echo "profile order ok: preflight -> build -> freeze -> publish -> bounded admission -> verify -> inference"
