#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_CACHE="${HOME}/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots"
MODEL_REPO_ROOT="${MODEL_CACHE%/snapshots}"
MODEL_REF_FILE="${MODEL_REPO_ROOT}/refs/main"
MODEL_FILE="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
FASTEMBED_CACHE="${HOME}/Library/Application Support/wenlan/memorydb/fastembed_cache"
FASTEMBED_REPO_ROOT="${FASTEMBED_CACHE}/models--Qdrant--bge-base-en-v1.5-onnx-Q"
FASTEMBED_REF_FILE="${FASTEMBED_REPO_ROOT}/refs/main"
FASTEMBED_REQUIRED_FILES=(
  "model_optimized.onnx"
  "tokenizer.json"
  "config.json"
  "special_tokens_map.json"
  "tokenizer_config.json"
)
THERMAL_HELPER_SOURCE="${ROOT}/scripts/rb01-thermal-state.m"
COOLDOWN_STATE="/private/tmp/wenlan-rb01-profile-next-allowed"
PROFILE_LOCK="/private/tmp/wenlan-rb01-profile.lock"
TIMEOUT_SECS=300
FAILSAFE_COOLDOWN_SECS=5700
POST_BUILD_ADMISSION_WAIT_SECS=1800
POST_BUILD_ADMISSION_TERMINATION_GRACE_SECS=2
PROFILE_DATA_ROOT="${REPO_DATA_ROOT:-${HOME}/.local/share/repo-data}"
JQ_BIN="$(command -v jq || true)"
child_pid=""
child_pgid=""
evidence_dir=""
artifact_dir=""
artifact_created=0
evidence_persisted=0
model_path=""
model_resolved_path=""
model_blob_sha256=""
fastembed_model_path=""
fastembed_model_resolved_path=""
fastembed_model_blob_sha256=""
build_metadata=""
frozen_dir=""
frozen_test_binary=""
frozen_daemon_binary=""
frozen_thermal_helper=""
test_binary=""
test_binary_sha256=""
daemon_binary=""
daemon_binary_sha256=""
thermal_helper=""
thermal_helper_sha256=""
compiled_thermal_helper=""
cargo_lock_sha256=""
git_worktree_diff_sha256=""
source_bundle_sha256=""
source_scheduler_sha256=""
source_db_sha256=""
source_profiler_sha256=""
source_live_daemon_harness_sha256=""
source_thermal_helper_sha256=""
source_backlog_audit_sha256=""
source_backlog_result_sha256=""
git_head=""
test_target_name=""
lock_held=0
process_group_cleanup_failed=0
preflight_watchdog_pid=""
preflight_probe_dir=""
preflight_state_file=""
preflight_attempt_output=""

usage() {
  echo "usage: scripts/profile-ambient-rb01.sh [--preflight-only|document|entity|page-growth|reconcile|citation|daemon]"
}

compile_thermal_helper() {
  if [[ ! -f "${THERMAL_HELPER_SOURCE}" ]]; then
    echo "refusing profile: thermal helper source is missing" >&2
    return 1
  fi
  compiled_thermal_helper="$(/usr/bin/mktemp /private/tmp/wenlan-rb01-thermal.XXXXXX)"
  if ! /usr/bin/clang -O2 -fobjc-arc -framework Foundation \
    "${THERMAL_HELPER_SOURCE}" -o "${compiled_thermal_helper}"; then
    /bin/rm -f "${compiled_thermal_helper}"
    compiled_thermal_helper=""
    echo "refusing profile: could not compile the low-overhead thermal helper" >&2
    return 1
  fi
  /bin/chmod 500 "${compiled_thermal_helper}"
  thermal_helper="${compiled_thermal_helper}"
  thermal_helper_sha256="$(sha256_file "${thermal_helper}")"
}

thermal_state() {
  if [[ -z "${thermal_helper}" || ! -x "${thermal_helper}" ]]; then
    echo "thermal helper unavailable" >&2
    return 1
  fi
  "${thermal_helper}"
}

memory_free_percent() {
  /usr/bin/memory_pressure -Q |
    /usr/bin/awk -F': ' '/System-wide memory free percentage/ {gsub("%", "", $2); print $2}'
}

sha256_file() {
  /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

worktree_source_bundle_sha256() {
  {
    /usr/bin/printf 'wenlan-rb01-source-bundle-v1\0'
    rtk proxy git diff --binary HEAD
    while IFS= read -r -d '' path; do
      /usr/bin/printf '\0untracked-path\0%s\0' "${path}"
      if [[ -L "${path}" ]]; then
        /usr/bin/printf 'symlink\0'
        /usr/bin/readlink "${path}"
      elif [[ -f "${path}" ]]; then
        /usr/bin/printf 'regular\0%s\0' "$(/usr/bin/stat -f '%z' "${path}")"
        /bin/cat -- "${path}"
      else
        echo "refusing profile: unsupported untracked source entry ${path}" >&2
        return 1
      fi
    done < <(
      rtk proxy git ls-files --others --exclude-standard -z |
        /usr/bin/perl -0ne 'push @paths, $_; END { print sort @paths }'
    )
  } | /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
}

verify_frozen_binaries() {
  if [[ ! -x "${thermal_helper}" ]]; then
    echo "refusing profile: frozen thermal helper is missing or not executable" >&2
    return 1
  fi
  if [[ "$(sha256_file "${thermal_helper}")" != "${thermal_helper_sha256}" ]]; then
    echo "refusing profile: frozen thermal helper changed before execution" >&2
    return 1
  fi
  if [[ ! -x "${test_binary}" ]]; then
    echo "refusing profile: frozen test binary is missing or not executable" >&2
    return 1
  fi
  if [[ "$(sha256_file "${test_binary}")" != "${test_binary_sha256}" ]]; then
    echo "refusing profile: frozen test binary changed before execution" >&2
    return 1
  fi
  if [[ "${lane}" == "daemon" ]]; then
    if [[ ! -x "${daemon_binary}" ]]; then
      echo "refusing profile: frozen daemon binary is missing or not executable" >&2
      return 1
    fi
    if [[ "$(sha256_file "${daemon_binary}")" != "${daemon_binary_sha256}" ]]; then
      echo "refusing profile: frozen daemon binary changed before execution" >&2
      return 1
    fi
  fi
  if [[ "$(sha256_file Cargo.lock)" != "${cargo_lock_sha256}" ]]; then
    echo "refusing profile: Cargo.lock changed after the measured build" >&2
    return 1
  fi
  if [[ "$(worktree_source_bundle_sha256)" != "${source_bundle_sha256}" ]]; then
    echo "refusing profile: source bytes changed after the measured build" >&2
    return 1
  fi
  if [[ "$(sha256_file crates/wenlan-server/src/scheduler.rs)" != "${source_scheduler_sha256}" ||
        "$(sha256_file crates/wenlan-core/src/db.rs)" != "${source_db_sha256}" ||
        "$(sha256_file scripts/profile-ambient-rb01.sh)" != "${source_profiler_sha256}" ||
        "$(sha256_file crates/wenlan-server/tests/ambient_live_daemon.rs)" != "${source_live_daemon_harness_sha256}" ||
        "$(sha256_file scripts/rb01-thermal-state.m)" != "${source_thermal_helper_sha256}" ||
        "$(sha256_file scripts/audit-ambient-rb01-backlog.sh)" != "${source_backlog_audit_sha256}" ||
        "$(sha256_file docs/eval/results/ambient-rb01-backlog-2026-07-20.json)" != "${source_backlog_result_sha256}" ]]; then
    echo "refusing profile: individually recorded source bytes changed after the measured build" >&2
    return 1
  fi
}

acquire_profile_lock() {
  if ! /bin/mkdir "${PROFILE_LOCK}" 2>/dev/null; then
    echo "refusing profile: another RB-01 profiler owns ${PROFILE_LOCK}" >&2
    return 1
  fi
  lock_held=1
  echo "$$" >"${PROFILE_LOCK}/owner_pid"
}

release_profile_lock() {
  if (( lock_held == 0 )); then
    return
  fi
  /bin/rm -f "${PROFILE_LOCK}/owner_pid"
  /bin/rmdir "${PROFILE_LOCK}" 2>/dev/null || true
  lock_held=0
}

check_cached_model() {
  local revision
  if [[ ! -f "${MODEL_REF_FILE}" ]]; then
    echo "refusing profile: hf-hub main ref is missing for ${MODEL_FILE}; this script never downloads models" >&2
    return 1
  fi
  revision="$(/bin/cat "${MODEL_REF_FILE}")"
  if [[ ! "${revision}" =~ ^[0-9a-f]{40,64}$ ]]; then
    echo "refusing profile: hf-hub main ref is not a content revision" >&2
    return 1
  fi
  model_path="${MODEL_CACHE}/${revision}/${MODEL_FILE}"
  if [[ ! -f "${model_path}" ]]; then
    echo "refusing profile: exact hf-hub main-ref cache entry for ${MODEL_FILE} was not found; this script never downloads models" >&2
    return 1
  fi
}

check_fastembed_cache() {
  local revision required_file snapshot
  if [[ ! -f "${FASTEMBED_REF_FILE}" ]]; then
    echo "refusing profile: FastEmbed hf-hub main ref is missing; this script never downloads models" >&2
    return 1
  fi
  revision="$(/bin/cat "${FASTEMBED_REF_FILE}")"
  if [[ ! "${revision}" =~ ^[0-9a-f]{40,64}$ ]]; then
    echo "refusing profile: FastEmbed hf-hub main ref is not a content revision" >&2
    return 1
  fi
  snapshot="${FASTEMBED_REPO_ROOT}/snapshots/${revision}"
  for required_file in "${FASTEMBED_REQUIRED_FILES[@]}"; do
    if [[ ! -f "${snapshot}/${required_file}" ]]; then
      echo "refusing profile: exact FastEmbed cache entry ${required_file} is missing; this script never downloads models" >&2
      return 1
    fi
  done
  fastembed_model_path="${snapshot}/model_optimized.onnx"
}

check_model_headroom() {
  local free_percent="${1}"
  local total_bytes model_bytes headroom
  total_bytes="$(
    /usr/bin/memory_pressure -Q |
      /usr/bin/awk '/^The system has [0-9]+/ { print $4; exit }'
  )"
  model_bytes="$(/usr/bin/stat -Lf '%z' "${model_path}" 2>/dev/null || true)"
  if [[ ! "${total_bytes}" =~ ^[0-9]+$ ]] ||
    [[ ! "${model_bytes}" =~ ^[0-9]+$ ]]; then
    echo "refusing profile: model working-set headroom is unavailable" >&2
    return 1
  fi

  if ! headroom="$(
    /usr/bin/awk \
      -v total="${total_bytes}" \
      -v free_percent="${free_percent}" \
      -v model="${model_bytes}" '
        BEGIN {
          policy_floor = total * 0.15
          if (policy_floor < 2147483648) {
            policy_floor = 2147483648
          }
          free_bytes = total * free_percent / 100
          required = model + policy_floor
          printf "%.0f %.0f", free_bytes, required
          exit !(free_bytes >= required)
        }
      '
  )"; then
    echo "refusing profile: estimated free/required bytes are ${headroom}; required includes cached model plus scheduler memory floor" >&2
    return 1
  fi
}

check_cooldown() {
  local now next remaining
  now="$(date +%s)"
  if [[ ! -f "${COOLDOWN_STATE}" ]]; then
    return 0
  fi
  next="$(/bin/cat "${COOLDOWN_STATE}")"
  if [[ ! "${next}" =~ ^[0-9]+$ ]] || (( now >= next )); then
    return 0
  fi
  remaining=$((next - now))
  echo "refusing profile: prior real slice still owns the cooldown for ${remaining}s" >&2
  return 1
}

arm_failsafe_cooldown() {
  local cooldown_seconds="${FAILSAFE_COOLDOWN_SECS}"
  echo $(( $(date +%s) + cooldown_seconds )) >"${COOLDOWN_STATE}"
  if [[ -n "${evidence_dir}" ]]; then
    {
      echo "cooldown_source=failsafe_timeout"
      echo "cooldown_seconds=${cooldown_seconds}"
    } >>"${evidence_dir}/manifest.txt"
  fi
}

cooldown_from_report() {
  local raw_log="${1}"
  local report_elapsed_ms slice_seconds cooldown_seconds
  report_elapsed_ms="$(
    /usr/bin/awk '
      match($0, /"report_elapsed_ms":[0-9]+/) {
        value = substr($0, RSTART, RLENGTH)
        sub(/^.*:/, "", value)
        latest = value
      }
      END {
        if (latest != "") {
          print latest
        }
      }
    ' "${raw_log}"
  )"
  if [[ ! "${report_elapsed_ms}" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  slice_seconds=$(( (report_elapsed_ms + 999) / 1000 ))
  cooldown_seconds=$((slice_seconds * 19))
  if (( cooldown_seconds < 120 )); then
    cooldown_seconds=120
  fi
  echo $(( $(date +%s) + cooldown_seconds )) >"${COOLDOWN_STATE}"
  {
    echo "cooldown_source=measured_job_elapsed"
    echo "report_elapsed_ms=${report_elapsed_ms}"
    echo "cooldown_seconds=${cooldown_seconds}"
  } >>"${evidence_dir}/manifest.txt"
}

calibration_no_inference_reported() {
  local raw_log="${1}"
  /usr/bin/grep -Fq '"event":"rb01_calibration_no_inference"' "${raw_log}" ||
    /usr/bin/grep -Fq '"event":"rb01_calibration_skipped"' "${raw_log}"
}

calibration_no_inference_recovered() {
  local raw_log="${1}"
  local thermal memory
  calibration_no_inference_reported "${raw_log}" || return 1
  thermal="$(thermal_state 2>/dev/null || true)"
  memory="$(memory_free_percent 2>/dev/null || true)"
  if [[ "${thermal}" != "0" || -z "${memory}" ]] ||
    ! /usr/bin/awk -v value="${memory}" 'BEGIN { exit !(value >= 15.0) }'; then
    return 1
  fi
  /bin/rm -f "${COOLDOWN_STATE}"
  {
    echo "cooldown_source=calibration_no_inference_recovered"
    echo "cooldown_seconds=0"
  } >>"${evidence_dir}/manifest.txt"
}

check_two_cpu_samples() {
  local samples count value
  samples="$(
    /usr/bin/top -l 3 -n 0 -s 30 |
      /usr/bin/awk '
        /CPU usage/ {
          for (i = 1; i <= NF; i++) {
            if (index($i, "idle") > 0) {
              idle = $(i - 1)
              gsub("%", "", idle)
              print 100 - idle
            }
          }
        }
      ' |
      /usr/bin/tail -n 2
  )"
  count=0
  while IFS= read -r value; do
    [[ -z "${value}" ]] && continue
    count=$((count + 1))
    if ! /usr/bin/awk -v value="${value}" 'BEGIN { exit !(value <= 20.0) }'; then
      echo "refusing profile: aggregate CPU sample ${count}/2 was ${value}% (>20%)" >&2
      return 1
    fi
  done <<<"${samples}"
  if (( count != 2 )); then
    echo "refusing profile: expected two aggregate CPU samples, observed ${count}" >&2
    return 1
  fi
  echo "${samples}"
}

preflight() {
  local thermal memory cpu_samples
  check_cached_model || return 1
  check_fastembed_cache || return 1
  check_cooldown || return 1

  if ! thermal="$(thermal_state)"; then
    thermal=""
  fi
  if [[ "${thermal}" != "0" ]]; then
    echo "refusing profile: macOS thermal state is ${thermal:-unavailable}, expected 0 (nominal)" >&2
    return 1
  fi

  if ! memory="$(memory_free_percent)"; then
    memory=""
  fi
  if [[ -z "${memory}" ]] ||
    ! /usr/bin/awk -v value="${memory}" 'BEGIN { exit !(value >= 15.0) }'; then
    echo "refusing profile: free memory is ${memory:-unavailable}%, expected at least 15%" >&2
    return 1
  fi
  check_model_headroom "${memory}" || return 1

  if ! cpu_samples="$(check_two_cpu_samples)"; then
    return 1
  fi
  echo "preflight_ok thermal_state=${thermal} memory_free_percent=${memory} cpu_percent_samples=$(echo "${cpu_samples}" | tr '\n' ',')"
}

stop_preflight_watchdog() {
  if [[ -z "${preflight_watchdog_pid}" ]]; then
    return
  fi
  /bin/kill "${preflight_watchdog_pid}" 2>/dev/null || true
  wait "${preflight_watchdog_pid}" 2>/dev/null || true
  preflight_watchdog_pid=""
}

remove_preflight_probe_files() {
  if [[ -n "${preflight_probe_dir}" ]]; then
    /bin/rm -f \
      "${preflight_probe_dir}/output" \
      "${preflight_probe_dir}/timed-out"
    /bin/rmdir "${preflight_probe_dir}" 2>/dev/null || true
    preflight_probe_dir=""
  fi
  if [[ -n "${preflight_state_file}" ]]; then
    /bin/rm -f "${preflight_state_file}"
    preflight_state_file=""
  fi
}

run_sourced_preflight_attempt_until() {
  local deadline="${1}"
  local state_file="${2}"
  local profiler_path now status output_file timeout_marker

  preflight_attempt_output=""
  now="$(date +%s)"
  if (( now >= deadline )); then
    preflight_attempt_output="refusing profile: post-build admission deadline reached"
    return 124
  fi

  preflight_probe_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-preflight.XXXXXX)"
  output_file="${preflight_probe_dir}/output"
  timeout_marker="${preflight_probe_dir}/timed-out"
  profiler_path="${BASH_SOURCE[0]}"

  /usr/bin/perl -e \
    'defined(setpgrp(0, 0)) or die "setpgrp: $!"; exec @ARGV' \
    /bin/bash -c '
      source "$1"
      source "$2"
      if ! verify_frozen_binaries; then
        exit 1
      fi
      if preflight; then
        status=0
      else
        status=$?
      fi
      verify_frozen_binaries || exit 1
      exit "${status}"
    ' bash "${profiler_path}" "${state_file}" \
    >"${output_file}" 2>&1 &
  child_pid=$!
  child_pgid="${child_pid}"

  /usr/bin/perl -MTime::HiRes=time,sleep -e '
    my ($deadline, $grace, $pgid, $marker) = @ARGV;
    my $soft_at = $deadline - $grace;
    my $delay = $soft_at - time();
    sleep($delay) if $delay > 0;
    open(my $fh, ">", $marker) or die "open timeout marker: $!";
    print {$fh} "1\n";
    close($fh) or die "close timeout marker: $!";
    kill "TERM", -$pgid;
    $delay = $deadline - time();
    sleep($delay) if $delay > 0;
    kill "KILL", -$pgid;
  ' "${deadline}" "${POST_BUILD_ADMISSION_TERMINATION_GRACE_SECS}" \
    "${child_pgid}" "${timeout_marker}" &
  preflight_watchdog_pid=$!

  if wait "${child_pid}" 2>/dev/null; then
    status=0
  else
    status=$?
  fi

  if [[ -s "${timeout_marker}" ]]; then
    wait "${preflight_watchdog_pid}" 2>/dev/null || true
    preflight_watchdog_pid=""
    preflight_attempt_output="$(/bin/cat "${output_file}")"
    if [[ -n "${preflight_attempt_output}" ]]; then
      preflight_attempt_output="${preflight_attempt_output}"$'\n'
    fi
    preflight_attempt_output="${preflight_attempt_output}refusing profile: post-build admission deadline reached"
    status=124
  else
    stop_preflight_watchdog
    preflight_attempt_output="$(/bin/cat "${output_file}")"
  fi

  child_pid=""
  child_pgid=""
  remove_preflight_probe_files
  return "${status}"
}

run_preflight_attempt_until() {
  local deadline="${1}"
  local status
  preflight_state_file="$(/usr/bin/mktemp /private/tmp/wenlan-rb01-preflight-state.XXXXXX)"
  /bin/chmod 600 "${preflight_state_file}"
  if ! {
    printf 'lane=%q\n' "${lane}"
    printf 'test_binary=%q\n' "${test_binary}"
    printf 'test_binary_sha256=%q\n' "${test_binary_sha256}"
    printf 'daemon_binary=%q\n' "${daemon_binary}"
    printf 'daemon_binary_sha256=%q\n' "${daemon_binary_sha256}"
    printf 'thermal_helper=%q\n' "${thermal_helper}"
    printf 'thermal_helper_sha256=%q\n' "${thermal_helper_sha256}"
    printf 'cargo_lock_sha256=%q\n' "${cargo_lock_sha256}"
    printf 'source_bundle_sha256=%q\n' "${source_bundle_sha256}"
    printf 'source_scheduler_sha256=%q\n' "${source_scheduler_sha256}"
    printf 'source_db_sha256=%q\n' "${source_db_sha256}"
    printf 'source_profiler_sha256=%q\n' "${source_profiler_sha256}"
    printf 'source_live_daemon_harness_sha256=%q\n' \
      "${source_live_daemon_harness_sha256}"
    printf 'source_thermal_helper_sha256=%q\n' \
      "${source_thermal_helper_sha256}"
    printf 'source_backlog_audit_sha256=%q\n' \
      "${source_backlog_audit_sha256}"
    printf 'source_backlog_result_sha256=%q\n' \
      "${source_backlog_result_sha256}"
  } >"${preflight_state_file}"; then
    remove_preflight_probe_files
    preflight_attempt_output="refusing profile: could not serialize bounded preflight state"
    return 1
  fi

  if run_sourced_preflight_attempt_until "${deadline}" "${preflight_state_file}"; then
    status=0
  else
    status=$?
  fi
  remove_preflight_probe_files
  return "${status}"
}

wait_for_post_build_admission() {
  local deadline output status now attempt
  deadline=$(( $(date +%s) + POST_BUILD_ADMISSION_WAIT_SECS ))
  attempt=0

  while true; do
    attempt=$((attempt + 1))
    output=""
    if run_preflight_attempt_until "${deadline}"; then
      output="${preflight_attempt_output}"
      echo "${output}"
      return 0
    else
      status=$?
      output="${preflight_attempt_output}"
    fi

    [[ -z "${output}" ]] || echo "${output}" >&2
    if ! /usr/bin/awk '
      NR == 1 {
        cpu_busy = ($0 ~ /^refusing profile: aggregate CPU sample [12]\/2 was [0-9.]+% \(>20%\)$/)
      }
      END { exit !(NR == 1 && cpu_busy) }
    ' <<<"${output}"; then
      return "${status}"
    fi

    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "refusing profile: post-build admission remained CPU-busy for ${POST_BUILD_ADMISSION_WAIT_SECS}s" >&2
      return 1
    fi
    echo "waiting for post-build CPU admission after attempt=${attempt}; deadline_epoch=${deadline}" >&2
  done
}

finish_manifest() {
  local status="${1:-130}"
  if [[ -n "${evidence_dir}" ]]; then
    {
      echo "exit_status=${status}"
      echo "finished_at_epoch=$(date +%s)"
      echo "thermal_state_after=$(thermal_state || true)"
      echo "memory_free_percent_after=$(memory_free_percent || true)"
    } >>"${evidence_dir}/manifest.txt"
  fi
}

persist_evidence() {
  local status="${1:-130}"
  local git_dirty hw_model total_memory_bytes
  if [[ -z "${evidence_dir}" || ! -d "${evidence_dir}" ]] ||
    (( evidence_persisted != 0 )); then
    return
  fi

  if [[ -z "${artifact_dir}" ]]; then
    artifact_dir="${PROFILE_DATA_ROOT}/wenlan/benchmarks/${git_head}-${source_bundle_sha256:0:12}/${stamp}-${lane}"
  fi
  /bin/mkdir -p "${artifact_dir}"
  /bin/cp -p "${evidence_dir}/manifest.txt" "${artifact_dir}/manifest.txt"
  /bin/cp -p "${evidence_dir}/raw.log" "${artifact_dir}/raw.log"
  git_dirty=false
  if [[ -n "$(rtk git status --porcelain)" ]]; then
    git_dirty=true
  fi
  hw_model="$(/usr/sbin/sysctl -n hw.model 2>/dev/null || true)"
  total_memory_bytes="$(
    /usr/bin/memory_pressure -Q |
      /usr/bin/awk '/^The system has [0-9]+/ { print $4; exit }'
  )"
  "${JQ_BIN}" -n \
    --arg git_sha "${git_head}" \
    --argjson dirty "${git_dirty}" \
    --arg diff_sha256 "${git_worktree_diff_sha256}" \
    --arg source_bundle_sha256 "${source_bundle_sha256}" \
    --arg lane "${lane}" \
    --arg model_id "qwen3-4b" \
    --arg model_blob_sha256 "${model_blob_sha256}" \
    --arg fastembed_model_blob_sha256 "${fastembed_model_blob_sha256}" \
    --arg test_binary_sha256 "${test_binary_sha256}" \
    --arg daemon_binary_sha256 "${daemon_binary_sha256}" \
    --arg thermal_helper_sha256 "${thermal_helper_sha256}" \
    --arg scheduler_sha256 "${source_scheduler_sha256}" \
    --arg db_sha256 "${source_db_sha256}" \
    --arg profiler_sha256 "${source_profiler_sha256}" \
    --arg live_daemon_harness_sha256 "${source_live_daemon_harness_sha256}" \
    --arg thermal_helper_source_sha256 "${source_thermal_helper_sha256}" \
    --arg backlog_audit_sha256 "${source_backlog_audit_sha256}" \
    --arg backlog_result_sha256 "${source_backlog_result_sha256}" \
    --arg arch "$(/usr/bin/uname -m)" \
    --arg hw_model "${hw_model}" \
    --arg total_memory_bytes "${total_memory_bytes:-0}" \
    --arg timeout_secs "${TIMEOUT_SECS}" \
    --arg exit_status "${status}" \
    '{
      git_sha: $git_sha,
      dirty: $dirty,
      git_worktree_diff_sha256: $diff_sha256,
      seed: null,
      params: {
        lane: $lane,
        model_id: $model_id,
        timeout_secs: ($timeout_secs | tonumber),
        production_timing: ($lane == "daemon")
      },
      hardware: {
        arch: $arch,
        model: $hw_model,
        total_memory_bytes: ($total_memory_bytes | tonumber)
      },
      provenance: {
        model_blob_sha256: $model_blob_sha256,
        fastembed_model_blob_sha256: $fastembed_model_blob_sha256,
        test_binary_sha256: $test_binary_sha256,
        daemon_binary_sha256: $daemon_binary_sha256,
        thermal_helper_sha256: $thermal_helper_sha256,
        scheduler_sha256: $scheduler_sha256,
        db_sha256: $db_sha256,
        profiler_sha256: $profiler_sha256,
        live_daemon_harness_sha256: $live_daemon_harness_sha256,
        thermal_helper_source_sha256: $thermal_helper_source_sha256,
        backlog_audit_sha256: $backlog_audit_sha256,
        backlog_result_sha256: $backlog_result_sha256,
        source_bundle_sha256: $source_bundle_sha256
      },
      exit_status: ($exit_status | tonumber)
    }' >"${artifact_dir}/manifest.json"
  evidence_persisted=1
}

terminate_child_group() {
  local attempt
  if [[ -z "${child_pgid}" ]]; then
    return
  fi

  # The test binary and its daemon child share a dedicated process group.
  # Kill the group even when SIGALRM already terminated the group leader;
  # otherwise the daemon grandchild can survive the profiler timeout.
  /bin/kill -TERM -- "-${child_pgid}" 2>/dev/null ||
    /bin/kill -TERM "${child_pid}" 2>/dev/null ||
    true
  for attempt in 1 2 3 4 5 6 7 8 9 10 \
    11 12 13 14 15 16 17 18 19 20 \
    21 22 23 24 25 26 27 28 29 30; do
    if ! /bin/kill -0 -- "-${child_pgid}" 2>/dev/null; then
      child_pgid=""
      return
    fi
    /bin/sleep 0.2
  done
  /bin/kill -KILL -- "-${child_pgid}" 2>/dev/null || true
  for attempt in 1 2 3 4 5; do
    if ! /bin/kill -0 -- "-${child_pgid}" 2>/dev/null; then
      child_pgid=""
      return
    fi
    /bin/sleep 0.2
  done
  echo "warning: isolated profile process group ${child_pgid} survived SIGKILL" >&2
  if [[ -n "${evidence_dir}" ]]; then
    echo "process_group_cleanup_failed=true" >>"${evidence_dir}/manifest.txt"
  fi
  process_group_cleanup_failed=1
  child_pgid=""
}

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  terminate_child_group
  stop_preflight_watchdog
  if (( process_group_cleanup_failed != 0 )); then
    status=1
  fi
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    wait "${child_pid}" 2>/dev/null || true
  fi
  remove_preflight_probe_files
  finish_manifest "${status}"
  persist_evidence "${status}" || true
  if [[ -n "${build_metadata}" ]]; then
    /bin/rm -f "${build_metadata}"
  fi
  if [[ -n "${frozen_dir}" && -d "${frozen_dir}" ]]; then
    /bin/rm -rf "${frozen_dir}"
  fi
  if [[ -n "${compiled_thermal_helper}" ]]; then
    /bin/rm -f "${compiled_thermal_helper}"
    compiled_thermal_helper=""
  fi
  if (( artifact_created != 0 )) &&
    [[ -z "${evidence_dir}" && -n "${artifact_dir}" && -d "${artifact_dir}" ]]; then
    /bin/rm -rf "${artifact_dir}"
  fi
  release_profile_lock
  exit "${status}"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

trap cleanup INT TERM EXIT

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  --help|-h)
    usage
    exit 0
    ;;
  --preflight-only)
    compile_thermal_helper
    preflight
    exit 0
    ;;
  document|entity|page-growth|reconcile|citation|daemon)
    lane="$1"
    if [[ "${lane}" == "daemon" ]]; then
      TIMEOUT_SECS=2400
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

cd "${ROOT}"
acquire_profile_lock
compile_thermal_helper
if [[ -z "${JQ_BIN}" ]]; then
  echo "refusing profile: jq is required to identify the exact Cargo test executable" >&2
  exit 1
fi
# Refuse before compiling so the profiler itself never adds build load to an
# already-busy foreground session.
preflight
source_bundle_sha256="$(worktree_source_bundle_sha256)"
cargo_lock_sha256="$(sha256_file Cargo.lock)"
build_metadata="$(/usr/bin/mktemp /private/tmp/wenlan-rb01-build.XXXXXX)"
if [[ "${lane}" == "daemon" ]]; then
  rtk proxy cargo test --locked --offline -j 2 -p wenlan-server --test ambient_live_daemon --no-run --message-format=json >"${build_metadata}"
  test_target_name="ambient_live_daemon"
else
  rtk proxy cargo test --locked --offline -j 2 -p wenlan-server rb01_profile_admission --lib --no-run --message-format=json >"${build_metadata}"
  test_target_name="wenlan_server"
fi
test_binary="$(
  "${JQ_BIN}" -r --arg target_name "${test_target_name}" '
    select(
      .reason == "compiler-artifact"
      and .target.name == $target_name
      and .profile.test == true
      and .executable != null
    )
    | .executable
  ' "${build_metadata}" |
    /usr/bin/tail -n 1
)"
if [[ -z "${test_binary}" || ! -x "${test_binary}" ]]; then
  echo "refusing profile: Cargo did not produce the expected wenlan-server test binary" >&2
  exit 1
fi
if [[ "${lane}" == "daemon" ]]; then
  daemon_binary="$(
    "${JQ_BIN}" -r '
      select(
        .reason == "compiler-artifact"
        and .target.name == "wenlan-server"
        and (.target.kind | index("bin"))
        and .executable != null
      )
      | .executable
    ' "${build_metadata}" |
      /usr/bin/tail -n 1
  )"
  if [[ -z "${daemon_binary}" || ! -x "${daemon_binary}" ]]; then
    echo "refusing profile: Cargo did not identify the exact wenlan-server daemon binary" >&2
    exit 1
  fi
fi
if [[ "$(sha256_file Cargo.lock)" != "${cargo_lock_sha256}" ||
      "$(worktree_source_bundle_sha256)" != "${source_bundle_sha256}" ]]; then
  echo "refusing profile: source bytes changed during the measured build" >&2
  exit 1
fi
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

frozen_dir="$(/usr/bin/mktemp -d /private/tmp/wenlan-rb01-binaries.XXXXXX)"
/bin/chmod 700 "${frozen_dir}"
frozen_test_binary="${frozen_dir}/${test_target_name}"
/bin/cp "${test_binary}" "${frozen_test_binary}"
/bin/chmod 500 "${frozen_test_binary}"
test_binary_sha256="$(sha256_file "${frozen_test_binary}")"
frozen_thermal_helper="${frozen_dir}/rb01-thermal-state"
/bin/cp "${thermal_helper}" "${frozen_thermal_helper}"
/bin/chmod 500 "${frozen_thermal_helper}"
thermal_helper_sha256="$(sha256_file "${frozen_thermal_helper}")"
if [[ "${lane}" == "daemon" ]]; then
  frozen_daemon_binary="${frozen_dir}/wenlan-server"
  /bin/cp "${daemon_binary}" "${frozen_daemon_binary}"
  /bin/chmod 500 "${frozen_daemon_binary}"
  daemon_binary_sha256="$(sha256_file "${frozen_daemon_binary}")"
fi
test_binary="${frozen_test_binary}"
daemon_binary="${frozen_daemon_binary}"
thermal_helper="${frozen_thermal_helper}"

model_resolved_path="$(
  /usr/bin/perl -MCwd=realpath -e 'print realpath($ARGV[0])' "${model_path}"
)"
model_blob_sha256="$(/usr/bin/basename "${model_resolved_path}")"
if [[ ! "${model_blob_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "refusing profile: cached model is not backed by a content-addressed blob" >&2
  exit 1
fi
fastembed_model_resolved_path="$(
  /usr/bin/perl -MCwd=realpath -e 'print realpath($ARGV[0])' "${fastembed_model_path}"
)"
fastembed_model_blob_sha256="$(/usr/bin/basename "${fastembed_model_resolved_path}")"
if [[ ! "${fastembed_model_blob_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "refusing profile: cached FastEmbed model is not backed by a content-addressed blob" >&2
  exit 1
fi
git_worktree_diff_sha256="$(
  {
    rtk proxy git diff --binary HEAD
    rtk proxy git diff --binary --cached HEAD
  } | /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
)"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
git_head="$(rtk git rev-parse HEAD)"
artifact_dir="${PROFILE_DATA_ROOT}/wenlan/benchmarks/${git_head}-${source_bundle_sha256:0:12}/${stamp}-${lane}"
if [[ -e "${artifact_dir}" ]]; then
  echo "refusing profile: run-specific artifact directory already exists" >&2
  exit 1
fi
/bin/mkdir -p "${artifact_dir}/binaries"
artifact_created=1
thermal_helper="${artifact_dir}/binaries/rb01-thermal-state"
/bin/cp "${frozen_thermal_helper}" "${thermal_helper}"
/bin/chmod 500 "${thermal_helper}"
test_binary="${artifact_dir}/binaries/${test_target_name}"
/bin/cp "${frozen_test_binary}" "${test_binary}"
/bin/chmod 500 "${test_binary}"
if [[ "${lane}" == "daemon" ]]; then
  daemon_binary="${artifact_dir}/binaries/wenlan-server"
  /bin/cp "${frozen_daemon_binary}" "${daemon_binary}"
  /bin/chmod 500 "${daemon_binary}"
fi
verify_frozen_binaries

# Compilation and publishing the frozen binaries can change both CPU and
# thermal state. Reacquire the full two-sample admission window only after all
# substantial pre-run writes are complete. Only CPU contention is retryable;
# every other safety refusal remains immediate and fail-closed.
wait_for_post_build_admission
verify_frozen_binaries

evidence_dir="/private/tmp/wenlan-rb01-profile-${stamp}-${lane}"
/bin/mkdir -p "${evidence_dir}"
{
  echo "lane=${lane}"
  echo "started_at_epoch=$(date +%s)"
  echo "git_head=${git_head}"
  echo "source_scheduler_sha256=${source_scheduler_sha256}"
  echo "source_db_sha256=${source_db_sha256}"
  echo "source_profiler_sha256=${source_profiler_sha256}"
  echo "source_live_daemon_harness_sha256=${source_live_daemon_harness_sha256}"
  echo "source_thermal_helper_sha256=${source_thermal_helper_sha256}"
  echo "source_backlog_audit_sha256=${source_backlog_audit_sha256}"
  echo "source_backlog_result_sha256=${source_backlog_result_sha256}"
  echo "source_bundle_sha256=${source_bundle_sha256}"
  echo "cargo_lock_sha256=${cargo_lock_sha256}"
  echo "test_binary_path=${test_binary}"
  echo "test_binary_sha256=${test_binary_sha256}"
  echo "daemon_binary_path=${daemon_binary}"
  echo "daemon_binary_sha256=${daemon_binary_sha256}"
  echo "thermal_helper_path=${thermal_helper}"
  echo "thermal_helper_sha256=${thermal_helper_sha256}"
  echo "model_resolved_path=${model_resolved_path}"
  echo "model_blob_sha256=${model_blob_sha256}"
  echo "fastembed_model_resolved_path=${fastembed_model_resolved_path}"
  echo "fastembed_model_blob_sha256=${fastembed_model_blob_sha256}"
  echo "git_worktree_diff_sha256=${git_worktree_diff_sha256}"
  echo "thermal_state_before=$(thermal_state)"
  echo "memory_free_percent_before=$(memory_free_percent)"
  echo "timeout_secs=${TIMEOUT_SECS}"
} >"${evidence_dir}/manifest.txt"

# Arm a conservative timeout-sized recovery window before the child can load
# the model. An interrupt or missing JSON result leaves this fail-safe intact.
verify_frozen_binaries
arm_failsafe_cooldown

set +e
if [[ "${lane}" == "daemon" ]]; then
  (
    /usr/bin/env -u HF_HOME \
      WENLAN_TEST_FASTEMBED_CACHE="${FASTEMBED_CACHE}" \
      WENLAN_RB01_RUN_DIR="${artifact_dir}/daemon-data" \
      WENLAN_RB01_DAEMON_BINARY="${daemon_binary}" \
      WENLAN_RB01_DAEMON_SHA256="${daemon_binary_sha256}" \
      WENLAN_RB01_THERMAL_HELPER="${thermal_helper}" \
      WENLAN_RB01_THERMAL_HELPER_SHA256="${thermal_helper_sha256}" \
      HF_ENDPOINT="http://127.0.0.1:9" \
      WENLAN_RB01_DAEMON_PROFILE=1 \
      /usr/bin/perl -e 'defined(setpgrp(0, 0)) or die "setpgrp: $!"; alarm shift @ARGV; exec @ARGV' \
      "${TIMEOUT_SECS}" \
      rtk proxy "${test_binary}" \
      ambient_live_daemon::persistent_provider_respects_production_cooldown \
      --ignored --exact --nocapture --test-threads=1
  ) >"${evidence_dir}/raw.log" 2>&1 &
else
  (
    /usr/bin/env -u HF_HOME \
      WENLAN_TEST_FASTEMBED_CACHE="${FASTEMBED_CACHE}" \
      WENLAN_RB01_THERMAL_HELPER="${thermal_helper}" \
      WENLAN_RB01_THERMAL_HELPER_SHA256="${thermal_helper_sha256}" \
      HF_ENDPOINT="http://127.0.0.1:9" \
      WENLAN_RB01_PROFILE=1 \
      WENLAN_RB01_LANE="${lane}" \
      /usr/bin/perl -e 'defined(setpgrp(0, 0)) or die "setpgrp: $!"; alarm shift @ARGV; exec @ARGV' \
      "${TIMEOUT_SECS}" \
      rtk proxy "${test_binary}" \
      scheduler::tests::rb01_profile_real_on_device_slice \
      --ignored --exact --nocapture --test-threads=1
  ) >"${evidence_dir}/raw.log" 2>&1 &
fi
child_pid=$!
child_pgid="${child_pid}"
wait "${child_pid}"
status=$?
terminate_child_group
if (( process_group_cleanup_failed != 0 )); then
  status=1
fi
child_pid=""
set -e

# A successful child or the daemon's explicit recovery-known marker narrows
# the fail-safe to the production thermal policy. The marker is emitted only
# after observed work has stopped without a safety violation. An interrupt,
# watchdog failure, or unbounded shutdown keeps the fail-safe intact.
completed_thermal_work=0
if [[ "${lane}" == "daemon" ]] &&
  /usr/bin/grep -Fq '"rb01_recovery_known"' "${evidence_dir}/raw.log"; then
  completed_thermal_work=1
fi
if (( status == 0 )) && calibration_no_inference_reported "${evidence_dir}/raw.log"; then
  # The model process has exited and no request was sent. Release the pre-armed
  # fail-safe only when the host has already returned to nominal thermal/RAM;
  # otherwise leave the conservative fail-safe intact.
  calibration_no_inference_recovered "${evidence_dir}/raw.log" || true
elif (( status == 0 || completed_thermal_work != 0 )); then
  cooldown_from_report "${evidence_dir}/raw.log" || true
fi
finish_manifest "${status}"
persist_evidence "${status}"
if [[ -n "${build_metadata}" ]]; then
  /bin/rm -f "${build_metadata}"
  build_metadata=""
fi
if [[ -n "${frozen_dir}" && -d "${frozen_dir}" ]]; then
  /bin/rm -rf "${frozen_dir}"
  frozen_dir=""
fi
release_profile_lock
trap - INT TERM EXIT

/bin/cat "${evidence_dir}/raw.log"
echo "evidence_dir=${evidence_dir}"
echo "artifact_dir=${artifact_dir}"
exit "${status}"
