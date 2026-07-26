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
REQUIRED_GATE_TARGET_MS = 20 * 60 * 1000
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def require_object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--jobs-pages", type=Path, required=True)
    parser.add_argument("--cache-usage", type=Path, required=True)
    parser.add_argument("--cache-limit", type=Path, required=True)
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


def build_receipt(event_payload, pages, usage, limit):
    event_payload = require_object(event_payload, "event payload")
    usage = require_object(usage, "cache usage")
    limit = require_object(limit, "cache limit")
    run = event_payload.get("workflow_run", {})
    validate_run(run)
    jobs = normalize_jobs(pages, run)
    gates = [job for job in jobs if job["name"] == "conclusion"]
    if len(gates) != 1:
        raise ValueError("expected exactly one required conclusion job")

    budget_bytes = int(limit["max_cache_size_gb"]) * 1_000_000_000
    usage_bytes = int(usage["active_caches_size_in_bytes"])
    active_count = int(usage["active_caches_count"])
    if budget_bytes <= 0 or usage_bytes < 0 or active_count < 0:
        raise ValueError("cache usage and limit must be non-negative")
    over_by = max(0, usage_bytes - budget_bytes)
    ratio_ppm = usage_bytes * 1_000_000 // budget_bytes
    alert = usage_bytes * 1_000_000 > budget_bytes * CACHE_ALERT_RATIO_PPM
    required_gate_elapsed_ms = duration_ms(
        run.get("run_started_at"), gates[0]["completed_at"]
    )

    return {
        "schema_version": 1,
        "observed_run": {
            "id": run["id"],
            "attempt": run["run_attempt"],
            "name": run["name"],
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "head_sha": run["head_sha"],
            "run_started_at": run.get("run_started_at"),
            "updated_at": run.get("updated_at"),
        },
        "required_gate_elapsed_ms": required_gate_elapsed_ms,
        "required_gate_target_ms": REQUIRED_GATE_TARGET_MS,
        "required_gate_target_met": required_gate_elapsed_ms is not None
        and required_gate_elapsed_ms < REQUIRED_GATE_TARGET_MS,
        "jobs": jobs,
        "cache": {
            "usage_bytes": usage_bytes,
            "active_count": active_count,
            "limit_gb": int(limit["max_cache_size_gb"]),
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
        "schema_version": 1,
        "status": "invalid",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "raw_refs": {
            "event": raw_input_ref(args.event),
            "jobs_pages": raw_input_ref(args.jobs_pages),
            "cache_usage": raw_input_ref(args.cache_usage),
            "cache_limit": raw_input_ref(args.cache_limit),
        },
    }


def write_receipt(output, receipt):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, output)


def main():
    args = parse_args()
    try:
        receipt = build_receipt(
            load_json(args.event),
            load_json(args.jobs_pages),
            load_json(args.cache_usage),
            load_json(args.cache_limit),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ci-observer: invalid input: {error}", file=sys.stderr)
        write_receipt(args.output, error_receipt(args, error))
        raise SystemExit(1)

    write_receipt(args.output, receipt)
    raise SystemExit(2 if receipt["cache"]["alert"] else 0)


if __name__ == "__main__":
    main()
