#!/usr/bin/env python3
"""Throwaway replica of the drift_guard teeth that can bite Track A's edit.

`cargo test -p wenlan-core` cannot build on this host: llama-cpp-sys-2's
build.rs refuses without a Vulkan SDK, and the guards live in the same crate.
So the teeth that actually read `.github/workflows/release.yml` were read out of
crates/wenlan-core/src/drift_guard.rs by hand and re-implemented here against
the same YAML.

What was read, and what it turned out to assert:

  * release_rust_cache_violations              (drift_guard.rs:615-644)
  * windows_ort_distribution_violations        (:1458-1513)
  * release_promotion_contract_violations       (:5928-6242, the release-side
    parts: the forbidden-string scan at :5939, the per-job timeout map at
    :5948-5961, the all-jobs checkout-ref loop at :5991-6019, the control-plane
    and release-SHA pinning at :6040-6058, the tag revalidation list at
    :6120-6138, and the promotion DAG at :6157-6167)
  * release_preflight_contract_violations's DAG half (:4970-4984)
  * release_version_sync_never_runs_package_lifecycle_scripts (:3606-3639)
  * workflow_action_pin_violations             (:9123-9178)

Two findings worth stating plainly, because they change what this replica can
claim:

  1. NO drift_guard tooth references `app-bundle-windows`'s STEPS, its
     timeout-minutes, or its permissions. The per-job timeout map at :5948
     covers prepare-release, publish-crates, publish-npm, update-homebrew,
     docker-manifest and finalize-release only; app-bundle-windows appears in
     drift_guard solely as a name in promote-app-assets's `needs` list
     (:4980, :6166). So there is no sealed step here to break, and no
     mapping-length check over this job.
  2. workflow_action_pin_violations, the only SHA-pin guard in the Rust, runs
     over ci.yml, main-canary.yml, coverage.yml, ci-observer.yml and
     ci-benchmark.yml -- NOT release.yml and NOT the new signpath-status.yml.
     The SignPath action's pin is therefore enforced by
     release-workflow-contract.test.py alone. It is applied to both files here
     anyway, labelled as stricter than the Rust, because it is free.

The real trap this replica exists to catch is the forbidden-substring scan:
three separate teeth read the WHOLE release.yml text for "cargo build",
"build-release-binaries", "Swatinem/rust-cache" and "sccache-action", comments
included. A new step or comment containing any of those fails the Rust suite in
CI with no local way to find out.

ROUND 3 (Codex Sol), FINDINGS N3 AND N5. Two defects in the controls at the
bottom, both of this workstream's signature shape -- a control credited for
something it did not cause:

  N3: the replacements were `text.replace(old, new, 1)` with NO count. An
      absent anchor mutates nothing, and the control was then scored against
      the UNMUTATED file; a DUPLICATE anchor mutates the first occurrence,
      which may not be the one the check reads. Both are now hard errors:
      every mutation goes through `once()` or `once_in_job()`, which count.
      `job_span()` refuses a duplicated job key for the same reason.

  N5: `release_violations` implemented sixteen checks and the controls
      exercised five of them. Deleting any of the other eleven left every
      control green. Rather than a hand-kept list that goes stale the moment
      someone adds a seventeenth, each check now carries an ID, the ID
      REGISTRY IS READ OUT OF THIS FILE'S OWN SOURCE, and a meta-control fails
      when any registered ID has no mutation that provokes it. A new check
      with a new ID therefore fails this file until it is controlled.

  And a third, implied by both: a control is now required to be measured
  against a GREEN BASELINE for its own check. If release.yml already violated
  the check a control targets, the control's "expected string appeared" would
  be satisfied by the pre-existing violation and the mutation would be
  credited for it -- exactly the shape N1 found in the POSIX harness.

Every check below is exercised by a negative control at the bottom; a check
that cannot fail is not a check, and a control that cannot be shown to have
CAUSED its check to fire is not a control.

What this file CANNOT tell you: whether the hand reading above is still the
whole list. The seals are literals in this source; nothing here opens
drift_guard.rs, so a tooth added upstream would leave this replica passing over
an inventory that had gone short. `a-drift-guard-inventory.py` beside this file
is the part that can be measured -- it enumerates every function in
drift_guard.rs that mentions release.yml and requires each to be accounted for
by name.

Run: python3 scripts/negative-controls/a-drift-guard-replica.py
"""

from __future__ import annotations

import atexit
import inspect
import re
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
RELEASE_PATH = ROOT / ".github/workflows/release.yml"
CI_PATH = ROOT / ".github/workflows/ci.yml"
BUMP_PATH = ROOT / "scripts/bump-version.sh"
SIGNPATH_STATUS_PATH = ROOT / ".github/workflows/signpath-status.yml"

#: The terminal completion marker. Printed as the LAST line, unconditionally,
#: and only after every control has been scored. A consumer that does not see
#: it did not see a finished run -- which is the difference between "zero
#: control failures" and "the harness was killed before it could fail".
MARKER = "NEGATIVE-CONTROL COMPLETE"
HARNESS = "a-drift-guard-replica.py"
_ABORT_STARTED = time.time()


#: ROUND 3 (Codex Sol), FINDING N7. Set immediately before the completion
#: marker is printed. Every OTHER way out of this file -- an early
#: `sys.exit("...")` refusal, an unhandled exception, a signal, a watchdog kill
#: -- leaves it False, and the handler below says so. A transcript that simply
#: stops is the one thing a reader cannot tell from a transcript that finished.
_COMPLETED = False


@atexit.register
def _abort_marker():
    if _COMPLETED:
        return
    # stderr first: an early `sys.exit("...")` writes its message there, and the
    # aggregate runner reads the LAST non-empty line of the two merged.
    sys.stderr.flush()
    print('NEGATIVE-CONTROL ABORTED %s elapsed=%.1fs'
          % (HARNESS, time.time() - _ABORT_STARTED))
    print('  This run did not reach its own summary. Nothing above it is a '
          'result about this harness.')
    sys.stdout.flush()


ACTION_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")

# drift_guard.rs:621-626 and :5939-5940 and :1482-1483.
FORBIDDEN_IN_RELEASE = [
    "cargo build",
    "build-release-binaries",
    "Swatinem/rust-cache",
    "sccache-action",
]

# drift_guard.rs:5948-5961. app-bundle-windows is deliberately absent; that is
# the finding, not an omission in this replica.
RELEASE_JOB_TIMEOUTS = {
    "prepare-release": 10,
    "publish-crates": 15,
    "publish-npm": 10,
    "update-homebrew": 20,
    "docker-manifest": 10,
    "finalize-release": 10,
}

# drift_guard.rs:4970-4984 and :6157-6167.
EXPECTED_NEEDS = {
    "bind-release-tag": ["resolve-promotion"],
    "prepare-release": ["resolve-promotion", "bind-release-tag"],
    "promote-assets": ["resolve-promotion", "bind-release-tag", "prepare-release"],
    "promote-app-assets": [
        "resolve-promotion",
        "bind-release-tag",
        "prepare-release",
        "app-bundle",
        "app-bundle-windows",
    ],
}

# drift_guard.rs:6040-6049 vs :6050-6058.
CONTROL_PLANE_JOBS = ["promote-assets", "docker"]
RELEASE_SHA_JOBS = ["prepare-release", "publish-crates", "publish-npm"]

# drift_guard.rs:6120-6138.
TAG_REVALIDATING_JOBS = [
    "promote-assets",
    "promote-app-assets",
    "docker",
    "docker-manifest",
    "publish-crates",
    "publish-npm",
    "update-homebrew",
    "finalize-release",
]


class Violation(NamedTuple):
    """One finding, tagged with the id of the check that produced it.

    The tag is what makes the meta-control possible: without it, "did this
    mutation provoke the check I aimed at, or some other one?" is a question
    about the wording of a message.
    """

    check: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.message


def job_needs(parsed: dict, job_name: str) -> list[str]:
    needs = parsed.get("jobs", {}).get(job_name, {}).get("needs")
    if isinstance(needs, str):
        return [needs]
    if isinstance(needs, list):
        return [item for item in needs if isinstance(item, str)]
    return []


def yaml_dump(value) -> str:
    return yaml.safe_dump(value, default_flow_style=False, sort_keys=False)


def release_violations(release_text: str, bump_text: str) -> list[Violation]:
    violations: list[Violation] = []

    def add(check: str, message: str) -> None:
        violations.append(Violation(check, message))

    # serde_yaml::from_str(...).expect("parse release.yml") -- a syntax error is
    # a panic in three separate teeth, not a soft failure.
    try:
        parsed = yaml.safe_load(release_text)
    except yaml.YAMLError as error:
        add("parse", f"release.yml does not parse: {error}")
        return violations
    if not isinstance(parsed, dict) or "jobs" not in parsed:
        add("jobs-mapping", "release.yml has no jobs mapping")
        return violations
    jobs = parsed["jobs"]

    # drift_guard.rs:618-620, :1481, :5938 -- the duplicate tag build matrix.
    if "release" in jobs:
        add(
            "duplicate-tag-matrix",
            "release workflow retains the duplicate tag build matrix",
        )

    # drift_guard.rs:621-632 -- whole-file text scan, comments included.
    for forbidden in FORBIDDEN_IN_RELEASE:
        if forbidden in release_text:
            line = next(
                (
                    index
                    for index, text in enumerate(release_text.splitlines(), start=1)
                    if forbidden in text
                ),
                0,
            )
            add(
                "forbidden-string",
                f"release workflow can rebuild or cache PR-validated binaries via "
                f"{forbidden!r} (first at line {line})",
            )

    # drift_guard.rs:633-643 and :1489-1499.
    promote_steps = jobs.get("promote-assets", {}).get("steps", []) or []
    download = next(
        (
            step.get("run", "")
            for step in promote_steps
            if step.get("name") == "Download exact validated wrapper once"
        ),
        "",
    )
    if "scripts/release-promotion.py download-assets" not in download:
        add(
            "receipt-bound-archive",
            "release workflow does not consume the receipt-bound archive bundle",
        )

    # drift_guard.rs:5948-5961.
    for job_name, timeout in RELEASE_JOB_TIMEOUTS.items():
        if jobs.get(job_name, {}).get("timeout-minutes") != timeout:
            add(
                "job-timeout",
                f"release job {job_name!r} does not keep its {timeout}-minute bound",
            )

    # drift_guard.rs:5991-6019 -- every checkout in every job, app-bundle-windows
    # included.
    checkout_count = 0
    for job_name, job in jobs.items():
        for step in job.get("steps", []) or []:
            uses = step.get("uses")
            if not isinstance(uses, str) or not uses.startswith("actions/checkout@"):
                continue
            checkout_count += 1
            ref = (step.get("with") or {}).get("ref")
            if ref not in ("${{ github.sha }}", "${{ env.RELEASE_SHA }}"):
                add(
                    "checkout-ref-pin",
                    "release checkout is not pinned to its immutable control or "
                    f"release SHA (job {job_name}, ref {ref!r})",
                )
    # Two separate failures, and they were one `or` with one message. A file
    # with no checkout at all and a file that checks out a mutable tag are
    # different defects; sharing an id would let one control stand in for both.
    if checkout_count == 0:
        add(
            "checkout-count-zero",
            "release workflow performs no checkout at all; the ref rule above "
            "scanned nothing and would report clean over any ref",
        )
    if "ref: refs/tags/${{ env.RELEASE_TAG }}" in release_text:
        add("mutable-tag-checkout", "tag release retains a mutable tag-ref checkout")

    # drift_guard.rs:6020-6031.
    resolver_text = yaml_dump(jobs.get("resolve-promotion", {}))
    resolver_permissions = jobs.get("resolve-promotion", {}).get("permissions", {})
    if (
        "ref: ${{ github.sha }}" not in resolver_text
        or "ref: ${{ env.RELEASE_SHA }}" in resolver_text
        or resolver_permissions.get("actions") != "read"
        or resolver_permissions.get("contents") != "read"
        or resolver_permissions.get("pull-requests") != "read"
    ):
        add(
            "resolver-control-plane",
            "release resolver is not pinned to its immutable read-only main "
            "control plane",
        )

    # drift_guard.rs:6040-6058.
    for job_name in CONTROL_PLANE_JOBS:
        job_text = yaml_dump(jobs.get(job_name, {}))
        if (
            "ref: ${{ github.sha }}" not in job_text
            or "ref: ${{ env.RELEASE_SHA }}" in job_text
        ):
            add(
                "control-plane-pin",
                f"release packaging job {job_name!r} is not pinned to its main "
                "control plane",
            )
    for job_name in RELEASE_SHA_JOBS:
        job_text = yaml_dump(jobs.get(job_name, {}))
        if (
            "ref: ${{ env.RELEASE_SHA }}" not in job_text
            or "ref: ${{ github.sha }}" in job_text
        ):
            add(
                "release-sha-pin",
                f"release source job {job_name!r} is not pinned to the resolved "
                "release SHA",
            )

    # drift_guard.rs:6120-6138.
    for job_name in TAG_REVALIDATING_JOBS:
        job_text = yaml_dump(jobs.get(job_name, {}))
        if "/git/ref/tags/$RELEASE_TAG" not in job_text or "RELEASE_SHA" not in job_text:
            add(
                "tag-revalidation",
                f"publication job {job_name!r} does not revalidate the immutable "
                "receipt-derived tag",
            )

    # drift_guard.rs:4970-4984 and :6157-6167.
    for job_name, expected in EXPECTED_NEEDS.items():
        actual = job_needs(parsed, job_name)
        if actual != expected:
            add(
                "dag-needs",
                f"tag release DAG drift: {job_name!r} needs {actual}, expected "
                f"{expected}",
            )

    # drift_guard.rs:3616-3634 -- an exact inventory of four, across two files.
    npm_version_lines = [
        line
        for line in (bump_text.splitlines() + release_text.splitlines())
        if line.strip().startswith("npm version ") or "&& npm version " in line
    ]
    if len(npm_version_lines) != 4:
        add(
            "npm-version-inventory",
            f"unexpected npm version command inventory: {len(npm_version_lines)}",
        )
    if not all("--ignore-scripts" in line for line in npm_version_lines):
        add(
            "npm-ignore-scripts",
            "release version sync must not execute candidate-controlled npm "
            "lifecycle scripts",
        )

    return violations


def action_pin_violations(label: str, text: str) -> list[Violation]:
    """drift_guard.rs:9123-9153, applied to files the Rust does not cover."""
    parsed = yaml.safe_load(text)
    violations: list[Violation] = []

    def add(check: str, message: str) -> None:
        violations.append(Violation(check, message))

    for job_name, job in (parsed.get("jobs") or {}).items():
        candidates = [(f"job {job_name}", job.get("uses"))]
        for index, step in enumerate(job.get("steps", []) or []):
            step_name = step.get("name") or f"#{index}"
            candidates.append((f"job {job_name} step {step_name}", step.get("uses")))
        for location, uses in candidates:
            if not isinstance(uses, str):
                continue
            if uses.startswith("./") or ACTION_PIN.match(uses):
                continue
            add(
                "action-sha-pin",
                f"{label} {location} uses action {uses!r} without an immutable "
                "SHA pin",
            )
    return violations


# --------------------------------------------------------------- the registry
#
# ROUND 3, FINDING N5. The list of checks is read out of the checking
# functions' own source, not maintained beside them. A hand-kept list is a
# claim about the code; this is a measurement of it. `add("<id>"` is the one
# spelling both functions use, and a check that appends any other way has no id
# -- which `registry_is_exhaustive()` below turns into a failure rather than a
# silent gap.
ADD_CALL = re.compile(r'\badd\(\s*"([a-z0-9-]+)"')
#: Every statement that puts something into `violations`. Anything here that is
#: not an `add("<id>", ...)` call is an untagged, and therefore uncontrollable,
#: check.
APPEND_ANY = re.compile(r"\bviolations\.append\(|\badd\(")
#: The one legitimate `violations.append` and the one `add(` that is a
#: definition rather than a call. It is matched EXACTLY, and its absence is
#: itself reported: if the helper is ever spelled differently, this scan would
#: otherwise start silently mis-counting in the direction that hides checks.
HELPER_DEF = (
    "    def add(check: str, message: str) -> None:\n"
    "        violations.append(Violation(check, message))\n"
)


def check_ids(func: Callable) -> list[str]:
    return ADD_CALL.findall(inspect.getsource(func))


def registry_is_exhaustive(func: Callable) -> list[str]:
    """Complaints about checks that could not be given an id.

    `add(...)` is defined once per function as the only way in, so the count of
    `add(` calls and the count of tagged ids must agree. A raw
    `violations.append(...)` in the body is a check with no id: it can fire,
    and no meta-control can ever notice that nothing provokes it.
    """
    source = inspect.getsource(func)
    problems = []
    if source.count(HELPER_DEF) != 1:
        return [
            f"{func.__name__}: the `add` helper is not spelled the way this scan "
            f"expects ({source.count(HELPER_DEF)} match(es)); every count below "
            "would be off, so nothing is reported rather than reported wrong"
        ]
    body = source.replace(HELPER_DEF, "")
    calls = [m.group(0) for m in APPEND_ANY.finditer(body)]
    raw = [c for c in calls if c.startswith("violations.append")]
    tagged = ADD_CALL.findall(body)
    untagged = len(calls) - len(raw) - len(tagged)
    if raw:
        problems.append(
            f"{func.__name__}: {len(raw)} check(s) append to `violations` "
            "directly, so they carry no id and no meta-control can require a "
            "mutation for them"
        )
    if untagged:
        problems.append(
            f"{func.__name__}: {untagged} `add(` call(s) do not name a literal "
            "id; the registry cannot see them"
        )
    return problems


# ------------------------------------------------------------ anchored edits
#
# ROUND 3, FINDING N3. Every mutation below goes through one of these. An
# anchor that matches zero times mutates nothing and the control is then scored
# against the shipped file; an anchor that matches twice mutates the first
# occurrence, which need not be the one the check reads. Both are refusals.


class StaleAnchor(Exception):
    """The mutation could not be applied where it was aimed."""


def once(text: str, old: str, new: str, where: str = "release.yml") -> str:
    hits = text.count(old)
    if hits != 1:
        raise StaleAnchor(
            f"{where}: anchor matched {hits} time(s), wanted exactly 1: "
            f"{old.strip()[:80]!r}"
        )
    return text.replace(old, new, 1)


def once_re(text: str, pattern: str, new: str, where: str = "release.yml") -> str:
    hits = list(re.finditer(pattern, text))
    if len(hits) != 1:
        raise StaleAnchor(
            f"{where}: pattern {pattern!r} matched {len(hits)} time(s), wanted "
            "exactly 1"
        )
    return text[: hits[0].start()] + new + text[hits[0].end() :]


def job_span(text: str, job: str) -> tuple[int, int]:
    """The raw-text span of one top-level job, refusing a duplicated key.

    A duplicated job key is the YAML analogue of N8's duplicate function
    definition: `yaml.safe_load` keeps the LAST one, and a text edit aimed at
    the first would mutate a block the checks never read.
    """
    starts = list(re.finditer(r"(?m)^  %s:[ \t]*$" % re.escape(job), text))
    if len(starts) != 1:
        raise StaleAnchor(
            f"release.yml: job {job!r} is declared {len(starts)} time(s), wanted "
            "exactly 1"
        )
    start = starts[0].start()
    following = re.search(r"(?m)^  \S", text[starts[0].end() :])
    end = starts[0].end() + following.start() if following else len(text)
    return start, end


def once_in_job(text: str, job: str, old: str, new: str) -> str:
    start, end = job_span(text, job)
    block = text[start:end]
    hits = block.count(old)
    if hits != 1:
        raise StaleAnchor(
            f"release.yml job {job!r}: anchor matched {hits} time(s), wanted "
            f"exactly 1: {old.strip()[:80]!r}"
        )
    return text[:start] + block.replace(old, new, 1) + text[end:]


# ----------------------------------------------------------------- controls


class Control(NamedTuple):
    label: str
    #: The id this mutation must provoke.
    check: str
    #: Ids this mutation is EXPECTED to provoke as well, named out loud. A
    #: mutation that reddens the whole replica has not localised anything, so
    #: anything outside `{check} | collateral` is a failure.
    collateral: tuple[str, ...]
    #: (release_text, bump_text) -> (release_text, bump_text).
    mutate: Callable[[str, str], tuple[str, str]]


CONTROLS: tuple[Control, ...] = (
    Control(
        "a document that does not parse",
        "parse",
        (),
        lambda r, b: (r + "\nstray: [unclosed\n", b),
    ),
    Control(
        "a document with no jobs mapping",
        "jobs-mapping",
        (),
        lambda r, b: ("name: release\non: push\n", b),
    ),
    Control(
        "the duplicate tag build matrix comes back",
        "duplicate-tag-matrix",
        (),
        lambda r, b: (
            once_re(
                r,
                r"(?m)^jobs:$",
                "jobs:\n  release:\n    runs-on: ubuntu-24.04\n    steps: []\n",
            ),
            b,
        ),
    ),
    Control(
        "a comment containing the forbidden string",
        "forbidden-string",
        (),
        lambda r, b: (
            once_re(r, r"(?m)^jobs:$", "# cargo build would go here\njobs:"),
            b,
        ),
    ),
    Control(
        "promote-assets stops consuming the receipt-bound bundle",
        "receipt-bound-archive",
        (),
        lambda r, b: (
            once_in_job(
                r,
                "promote-assets",
                "python3 scripts/release-promotion.py download-assets",
                "gh release download",
            ),
            b,
        ),
    ),
    Control(
        "prepare-release loses its ten-minute bound",
        "job-timeout",
        (),
        lambda r, b: (
            once_in_job(
                r, "prepare-release", "    timeout-minutes: 10\n", "    timeout-minutes: 11\n"
            ),
            b,
        ),
    ),
    Control(
        "the Windows job checks out an unpinned ref",
        "checkout-ref-pin",
        (),
        lambda r, b: (
            once_in_job(
                r,
                "app-bundle-windows",
                "ref: ${{ env.RELEASE_SHA }}",
                "ref: ${{ github.ref }}",
            ),
            b,
        ),
    ),
    Control(
        "the Windows job checks out the mutable tag",
        "mutable-tag-checkout",
        # The ref rule sees it too, and says so; that is the point of naming
        # collateral rather than scoring only the first message that matched.
        ("checkout-ref-pin",),
        lambda r, b: (
            once_in_job(
                r,
                "app-bundle-windows",
                "ref: ${{ env.RELEASE_SHA }}",
                "ref: refs/tags/${{ env.RELEASE_TAG }}",
            ),
            b,
        ),
    ),
    Control(
        "no job checks out at all, so the ref rule scans nothing",
        "checkout-count-zero",
        (),
        lambda r, b: (r.replace("uses: actions/checkout@", "uses: local/checkout@"), b),
    ),
    Control(
        "the resolver loses a read-only permission",
        "resolver-control-plane",
        (),
        lambda r, b: (
            once_in_job(r, "resolve-promotion", "      pull-requests: read\n", ""),
            b,
        ),
    ),
    Control(
        "docker leaves the immutable main control plane",
        "control-plane-pin",
        (),
        lambda r, b: (
            once_in_job(
                r, "docker", "ref: ${{ github.sha }}", "ref: ${{ env.RELEASE_SHA }}"
            ),
            b,
        ),
    ),
    Control(
        "publish-npm builds from the control plane, not the resolved release SHA",
        "release-sha-pin",
        (),
        lambda r, b: (
            once_in_job(
                r, "publish-npm", "ref: ${{ env.RELEASE_SHA }}", "ref: ${{ github.sha }}"
            ),
            b,
        ),
    ),
    Control(
        "update-homebrew stops revalidating the tag",
        "tag-revalidation",
        (),
        lambda r, b: (
            once_in_job(
                r,
                "update-homebrew",
                "/git/ref/tags/$RELEASE_TAG",
                "/git/refs/tags/$RELEASE_TAG",
            ),
            b,
        ),
    ),
    Control(
        "dropping app-bundle-windows from the promotion DAG",
        "dag-needs",
        (),
        lambda r, b: (
            once(
                r,
                "    needs: [resolve-promotion, bind-release-tag, prepare-release, "
                "app-bundle, app-bundle-windows]",
                "    needs: [resolve-promotion, bind-release-tag, prepare-release, "
                "app-bundle]",
            ),
            b,
        ),
    ),
    Control(
        "an extra npm version line",
        "npm-version-inventory",
        (),
        # Carrying --ignore-scripts on purpose: without it this mutation would
        # trip the lifecycle rule too and stand in for a control it is not.
        lambda r, b: (r, b + '\nnpm version "$VERSION" --ignore-scripts\n'),
    ),
    Control(
        "one npm version line loses --ignore-scripts",
        "npm-ignore-scripts",
        (),
        lambda r, b: (
            r,
            once(
                b,
                "(cd crates/wenlan-mcp/npm && npm version \"$NEW_VERSION\" "
                "--no-git-tag-version --allow-same-version --ignore-scripts "
                ">/dev/null)",
                "(cd crates/wenlan-mcp/npm && npm version \"$NEW_VERSION\" "
                "--no-git-tag-version --allow-same-version >/dev/null)",
                where="scripts/bump-version.sh",
            ),
        ),
    ),
)

PIN_CONTROLS: tuple[Control, ...] = (
    Control(
        "an unpinned SignPath reference",
        "action-sha-pin",
        (),
        lambda r, b: (
            once(
                r,
                "uses: SignPath/github-action-submit-signing-request@"
                "c92b958760219087e01f8d67a1669ed57afe2627",
                "uses: SignPath/github-action-submit-signing-request@v2",
            ),
            b,
        ),
    ),
)


def score(
    controls: tuple[Control, ...],
    run: Callable[[str, str], list[Violation]],
    release_text: str,
    bump_text: str,
    baseline: list[Violation],
) -> tuple[int, set[str]]:
    """Run each control; return (failures, the set of ids they provoked)."""
    failures = 0
    provoked: set[str] = set()
    baseline_ids = {v.check for v in baseline}
    for control in controls:
        # A GREEN BASELINE FOR THIS CHECK, first. If release.yml already
        # violates the check a control targets, "the expected violation is
        # present" is satisfied by the pre-existing one and the mutation is
        # credited for a defect it did not cause.
        if control.check in baseline_ids:
            print(
                f"  FAIL {control.label}: {control.check!r} is ALREADY violated by "
                "the shipped file, so this control cannot be credited for causing it"
            )
            failures += 1
            continue
        try:
            mutated_release, mutated_bump = control.mutate(release_text, bump_text)
        except StaleAnchor as error:
            print(f"  FAIL {control.label}: {error}")
            failures += 1
            continue
        if (mutated_release, mutated_bump) == (release_text, bump_text):
            print(
                f"  FAIL {control.label}: the mutation changed nothing; this "
                "control tested the shipped file"
            )
            failures += 1
            continue
        found = run(mutated_release, mutated_bump)
        found_ids = {v.check for v in found}
        new_ids = found_ids - baseline_ids
        provoked |= new_ids
        if control.check not in found_ids:
            print(
                f"  FAIL {control.label}: the mutation did not provoke "
                f"{control.check!r}; it provoked {sorted(new_ids) or 'nothing'}"
            )
            failures += 1
            continue
        unexpected = new_ids - {control.check} - set(control.collateral)
        if unexpected:
            print(
                f"  FAIL {control.label}: also provoked {sorted(unexpected)}; the "
                "control is not pinned to the check it names"
            )
            failures += 1
            continue
        hit = next(v for v in found if v.check == control.check)
        extra = (
            f" (+{sorted(new_ids - {control.check})} as declared)"
            if control.collateral
            else ""
        )
        print(f"  ok   {control.label} -> {control.check}{extra}")
        print(f"         {hit.message}")
    return failures, provoked


def main() -> int:
    started = time.time()
    release_text = RELEASE_PATH.read_text(encoding="utf-8")
    bump_text = BUMP_PATH.read_text(encoding="utf-8")
    signpath_text = SIGNPATH_STATUS_PATH.read_text(encoding="utf-8")

    print("== replicated drift_guard teeth, against the edited release.yml ==")
    violations = release_violations(release_text, bump_text)
    for item in violations:
        print(f"  VIOLATION [{item.check}] {item.message}")
    print(f"  {len(violations)} violation(s)")

    print()
    print("== SHA pins (stricter than the Rust: it does not cover these files) ==")
    all_pin_violations = action_pin_violations("release.yml", release_text)
    all_pin_violations += action_pin_violations("signpath-status.yml", signpath_text)
    # `dtolnay/rust-toolchain@stable` in publish-crates predates this change --
    # it is in HEAD's release.yml at line 1550 -- and no Rust or Python contract
    # covers release.yml's pins, so nothing enforces it today. Reported, not
    # fixed: release.yml's publish-crates job is outside Track A's file set.
    preexisting = [
        item
        for item in all_pin_violations
        if "dtolnay/rust-toolchain@stable" in item.message
    ]
    pin_violations = [item for item in all_pin_violations if item not in preexisting]
    for item in preexisting:
        print(f"  PRE-EXISTING (not Track A, not enforced by any contract) {item.message}")
    for item in pin_violations:
        print(f"  VIOLATION {item.message}")
    print(
        f"  {len(pin_violations)} violation(s) attributable to this change, "
        f"{len(preexisting)} pre-existing"
    )

    print()
    print("== negative controls: each check has to be able to fail ==")
    failures, provoked = score(
        CONTROLS, release_violations, release_text, bump_text, violations
    )

    pin_baseline = action_pin_violations("release.yml", release_text)
    # The SignPath pin control is scored against a baseline restricted to the
    # SignPath step, because release.yml carries one PRE-EXISTING unpinned
    # action (dtolnay/rust-toolchain@stable) that would otherwise make the
    # whole `action-sha-pin` id look already-red and disqualify its control.
    pin_baseline = [v for v in pin_baseline if "submit-signing-request" in v.message]
    pin_failures, pin_provoked = score(
        PIN_CONTROLS,
        lambda r, _b: [
            v
            for v in action_pin_violations("release.yml", r)
            if "submit-signing-request" in v.message
        ],
        release_text,
        bump_text,
        pin_baseline,
    )
    failures += pin_failures

    print()
    print("== meta-control: every check must have a mutation that provokes it ==")
    meta_failures = 0
    for func, provoked_ids, where in (
        (release_violations, provoked, "release_violations"),
        (action_pin_violations, pin_provoked, "action_pin_violations"),
    ):
        declared = set(check_ids(func))
        for problem in registry_is_exhaustive(func):
            print(f"  FAIL {problem}")
            meta_failures += 1
        uncontrolled = sorted(declared - provoked_ids)
        if uncontrolled:
            print(
                f"  FAIL {where}: {len(uncontrolled)} check(s) have no control that "
                f"provokes them: {uncontrolled}"
            )
            meta_failures += 1
        else:
            print(
                f"  ok   {where}: all {len(declared)} registered check(s) were "
                "provoked by a control"
            )
        orphans = sorted(provoked_ids - declared)
        if orphans:
            print(
                f"  FAIL {where}: control(s) provoked {orphans}, which the registry "
                "does not know about"
            )
            meta_failures += 1
    failures += meta_failures

    print()
    elapsed = time.time() - started
    if violations or pin_violations or failures:
        print("DRIFT REPLICA: NOT CLEAN")
        globals()["_COMPLETED"] = True
        print(
            f"{MARKER} {HARNESS} failures={failures + len(violations) + len(pin_violations)} "
            f"elapsed={elapsed:.1f}s"
        )
        return 1
    print(
        f"DRIFT REPLICA CLEAN: no replicated drift_guard tooth is broken by the "
        f"Track A edit; {len(CONTROLS) + len(PIN_CONTROLS)} negative controls "
        f"each provoked the one check they name, off a baseline that was green "
        f"for that check, and every registered check has a control"
    )
    globals()["_COMPLETED"] = True
    print(f"{MARKER} {HARNESS} failures=0 elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
