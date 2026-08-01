#!/usr/bin/env python3
"""Build a deterministic receipt from read-only GitHub Actions metadata."""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_GITHUB_CLOCK_SKEW_MS = 5_000
CACHE_ALERT_RATIO_PPM = 1_150_000
CACHE_BUDGET_SOURCE = "repository_policy"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SLO_EXEMPT_JOB_PREFIXES = ("release-preflight (",)


def require_object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--jobs-pages", type=Path, required=True)
    parser.add_argument("--cache-usage", type=Path, required=True)
    parser.add_argument("--cache-budget-gb", required=True)
    parser.add_argument("--required-gate-target-minutes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValueError(f"{path} exceeds {MAX_INPUT_BYTES} bytes")
    return json.loads(path.read_text())


def parse_time(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string or null")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration_ms(started_at, completed_at, allow_negative=False):
    start = parse_time(started_at)
    end = parse_time(completed_at)
    if start is None or end is None:
        return None
    duration = int((end - start).total_seconds() * 1000)
    if duration < 0:
        if allow_negative or duration >= -MAX_GITHUB_CLOCK_SKEW_MS:
            return 0
        raise ValueError("completion timestamp precedes start timestamp")
    return duration


def validate_run(run):
    run = require_object(run, "workflow_run")
    if run.get("name") != "CI" or run.get("status") != "completed":
        raise ValueError("observer input is not a completed CI workflow run")
    if not isinstance(run.get("id"), int) or not isinstance(run.get("run_attempt"), int):
        raise ValueError("run id and attempt must be integers")
    if not FULL_SHA.fullmatch(run.get("head_sha", "")):
        raise ValueError("head SHA must be 40 lowercase hexadecimal characters")
    if not isinstance(run.get("event"), str) or not run["event"]:
        raise ValueError("workflow event must be a non-empty string")
    if run.get("head_branch") is not None and not isinstance(run["head_branch"], str):
        raise ValueError("head branch must be a string or null")
    duration_ms(run.get("run_started_at"), run.get("updated_at"))


def normalize_step(step):
    step = require_object(step, "job step")
    conclusion = step.get("conclusion")
    return {
        "number": step.get("number"),
        "name": step.get("name"),
        "started_at": step.get("started_at"),
        "completed_at": step.get("completed_at"),
        "duration_ms": duration_ms(
            step.get("started_at"),
            step.get("completed_at"),
            allow_negative=conclusion == "skipped",
        ),
        "conclusion": conclusion,
    }


def normalize_jobs(pages, run):
    if not isinstance(pages, list) or not pages:
        raise ValueError("jobs pages must be a non-empty JSON array")
    first_page = require_object(pages[0], "jobs page")
    expected_total = first_page.get("total_count")
    if not isinstance(expected_total, int):
        raise ValueError("jobs response has no integer total_count")

    jobs = []
    seen_ids = set()
    for page in pages:
        page = require_object(page, "jobs page")
        if page.get("total_count") != expected_total:
            raise ValueError("jobs pages disagree on total_count")
        page_jobs = page.get("jobs", [])
        if not isinstance(page_jobs, list):
            raise ValueError("jobs page jobs must be a JSON array")
        for job in page_jobs:
            job = require_object(job, "job")
            if job.get("run_id") != run["id"] or job.get("run_attempt") != run["run_attempt"]:
                raise ValueError("job belongs to a different run or attempt")
            job_id = job.get("id")
            if not isinstance(job_id, int) or job_id in seen_ids:
                raise ValueError("job ids must be unique integers")
            seen_ids.add(job_id)
            conclusion = job.get("conclusion")
            job_steps = job.get("steps", [])
            if not isinstance(job_steps, list):
                raise ValueError("job steps must be a JSON array")
            steps = sorted(
                (normalize_step(step) for step in job_steps),
                key=lambda step: (step["number"] or 0, step["name"] or ""),
            )
            jobs.append(
                {
                    "id": job_id,
                    "name": job.get("name"),
                    "started_at": job.get("started_at"),
                    "completed_at": job.get("completed_at"),
                    "duration_ms": duration_ms(
                        job.get("started_at"),
                        job.get("completed_at"),
                        allow_negative=conclusion == "skipped",
                    ),
                    "conclusion": conclusion,
                    "runner_name": job.get("runner_name"),
                    "runner_group_name": job.get("runner_group_name"),
                    "labels": sorted(job.get("labels", [])),
                    "steps": steps,
                }
            )
    if len(jobs) != expected_total:
        raise ValueError(
            f"jobs pagination returned {len(jobs)} jobs, expected {expected_total}"
        )
    return sorted(jobs, key=lambda job: (job["name"] or "", job["id"]))


def parse_positive_integer(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def parse_budget_gb(value):
    return parse_positive_integer(value, "cache budget in GB")


def require_nonnegative_integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def build_receipt(event_payload, pages, usage, budget_gb, gate_target_minutes):
    event_payload = require_object(event_payload, "event payload")
    usage = require_object(usage, "cache usage")
    run = event_payload.get("workflow_run", {})
    validate_run(run)
    jobs = normalize_jobs(pages, run)
    gates = [job for job in jobs if job["name"] == "conclusion"]
    if run.get("conclusion") == "cancelled":
        if len(gates) > 1:
            raise ValueError("expected at most one conclusion job for a cancelled run")
    elif len(gates) != 1:
        raise ValueError("expected exactly one required conclusion job")

    budget_bytes = budget_gb * 1_000_000_000
    usage_bytes = require_nonnegative_integer(
        usage.get("active_caches_size_in_bytes"), "active cache size"
    )
    active_count = require_nonnegative_integer(
        usage.get("active_caches_count"), "active cache count"
    )
    over_by = max(0, usage_bytes - budget_bytes)
    ratio_ppm = usage_bytes * 1_000_000 // budget_bytes
    alert = usage_bytes * 1_000_000 > budget_bytes * CACHE_ALERT_RATIO_PPM
    gate_elapsed_ms = (
        duration_ms(run.get("run_started_at"), gates[0]["completed_at"])
        if gates
        else None
    )
    gate_target_ms = gate_target_minutes * 60_000
    exempt_jobs = sorted(
        job["name"]
        for job in jobs
        if isinstance(job["name"], str)
        and job["conclusion"] != "skipped"
        and job["name"].startswith(SLO_EXEMPT_JOB_PREFIXES)
    )
    ordinary_pr = (
        run["event"] == "pull_request"
        and not (run.get("head_branch") or "").startswith(
            "release-please--branches--"
        )
        and not exempt_jobs
    )
    gate_over_by_ms = (
        max(0, gate_elapsed_ms - gate_target_ms)
        if gate_elapsed_ms is not None
        else None
    )

    return {
        "schema_version": 2,
        "observed_run": {
            "id": run["id"],
            "attempt": run["run_attempt"],
            "name": run["name"],
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "head_sha": run["head_sha"],
            "event": run["event"],
            "head_branch": run.get("head_branch"),
            "run_started_at": run.get("run_started_at"),
            "updated_at": run.get("updated_at"),
        },
        "required_gate_elapsed_ms": gate_elapsed_ms,
        "required_gate_target": {
            "minutes": gate_target_minutes,
            "milliseconds": gate_target_ms,
            "scope": "ordinary_pull_request",
            "applicable": ordinary_pr,
            "exempt_jobs": exempt_jobs,
            "status": (
                "not_applicable"
                if not ordinary_pr
                else "unavailable"
                if gate_elapsed_ms is None
                else "over_target"
                if gate_over_by_ms
                else "within_target"
            ),
            "over_by_ms": gate_over_by_ms,
        },
        "jobs": jobs,
        "cache": {
            "usage_bytes": usage_bytes,
            "active_count": active_count,
            "budget_gb": budget_gb,
            "budget_source": CACHE_BUDGET_SOURCE,
            "budget_bytes": budget_bytes,
            "ratio_ppm": ratio_ppm,
            "alert_ratio_ppm": CACHE_ALERT_RATIO_PPM,
            "alert": alert,
            "status": "over_budget" if over_by else "under_budget",
            "over_by_bytes": over_by,
        },
    }


def raw_input_ref(path):
    try:
        stat = path.stat()
    except OSError as error:
        return {
            "path": str(path),
            "exists": False,
            "error": str(error),
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
    }


def error_receipt(args, error):
    return {
        "schema_version": 2,
        "status": "invalid",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "cache_budget": {
            "configured_gb": args.cache_budget_gb,
            "source": CACHE_BUDGET_SOURCE,
        },
        "required_gate_target": {
            "configured_minutes": args.required_gate_target_minutes,
        },
        "raw_refs": {
            "event": raw_input_ref(args.event),
            "jobs_pages": raw_input_ref(args.jobs_pages),
            "cache_usage": raw_input_ref(args.cache_usage),
        },
    }


def write_receipt(output, receipt):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, output)


def warning(title, message):
    escaped = (
        str(message)
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )
    print(f"::warning title={title}::{escaped}")


def main():
    args = parse_args()
    try:
        budget_gb = parse_budget_gb(args.cache_budget_gb)
        gate_target_minutes = parse_positive_integer(
            args.required_gate_target_minutes, "required gate target minutes"
        )
        receipt = build_receipt(
            load_json(args.event),
            load_json(args.jobs_pages),
            load_json(args.cache_usage),
            budget_gb,
            gate_target_minutes,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ci-observer: invalid input: {error}", file=sys.stderr)
        write_receipt(args.output, error_receipt(args, error))
        warning("CI observer invalid input", error)
        raise SystemExit(0)

    write_receipt(args.output, receipt)
    if receipt["cache"]["status"] == "over_budget":
        warning(
            "CI cache budget exceeded",
            f"Cache usage is {receipt['cache']['usage_bytes']} bytes against the "
            f"{receipt['cache']['budget_gb']} GB repository budget.",
        )
    if receipt["required_gate_target"]["status"] == "over_target":
        warning(
            "CI required gate target exceeded",
            f"Required conclusion completed in {receipt['required_gate_elapsed_ms']} ms "
            f"against the {receipt['required_gate_target']['minutes']}-minute target.",
        )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
