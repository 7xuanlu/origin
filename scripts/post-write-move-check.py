#!/usr/bin/env python3
"""Byte-oriented R6 post_write movement comparator.

The manifest points every moved syntactic item at its pre-R6 source and its
flow-owned destination.  This comparator deliberately knows only three
normalizations:

* CRLF -> LF;
* one uniform module-body indent for the externalized test body;
* one leading ``pub(super) `` on an explicitly allowlisted item.

Manifest-declared reflection-string substitutions are applied before, and kept
separate from, normalization.

Top-level imports are compared as their own sorted set.  They are never stripped
from a function/module body before hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts/post-write-move-manifest.json"


class CheckError(RuntimeError):
    pass


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def dedent_one_uniform_body(text: str) -> str:
    lines = text.splitlines(keepends=True)
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        return text
    prefixes = []
    for line in nonblank:
        match = re.match(r"^[ \t]*", line)
        assert match is not None
        prefixes.append(match.group(0))
    prefix = prefixes[0]
    if not prefix or any(not candidate.startswith(prefix) for candidate in prefixes):
        raise CheckError("test module body has no one uniform indent to remove")
    return "".join(
        line[len(prefix) :] if line.startswith(prefix) else line
        for line in lines
    )


def remove_allowlisted_pub_super(text: str, allowed: bool) -> str:
    pattern = re.compile(
        r"(?m)^(\s*(?:#\[[^\n]+\]\s*\n\s*)*)(pub\(super\)\s+)"
    )
    found = list(pattern.finditer(text))
    if not found:
        return text
    if not allowed:
        raise CheckError("non-allowlisted leading pub(super) widening")
    if len(found) != 1:
        raise CheckError("expected exactly one allowlisted leading pub(super)")
    return pattern.sub(r"\1", text, count=1)


def apply_substitutions(
    text: str, substitutions: list[dict[str, Any]], side: str, phase: str
) -> str:
    for substitution in substitutions:
        phases = substitution.get("phases")
        if substitution["side"] != side or (phases and phase not in phases):
            continue
        old = substitution["from"]
        count = text.count(old)
        expected = substitution.get("count", 1)
        if count != expected:
            raise CheckError(
                f"explicit {side} substitution {old!r} matched {count}, expected {expected}"
            )
        text = text.replace(old, substitution["to"], expected)
    return text


def mask_rust(text: str) -> str:
    """Mask comments and string/char contents while preserving str offsets."""
    data = list(text)
    out = data.copy()
    index = 0
    block_depth = 0
    string = False
    char = False
    escaped = False
    raw_end: str | None = None
    while index < len(data):
        if raw_end is not None:
            if text.startswith(raw_end, index):
                out[index : index + len(raw_end)] = [" "] * len(raw_end)
                index += len(raw_end)
                raw_end = None
            else:
                if data[index] != "\n":
                    out[index] = " "
                index += 1
            continue
        if block_depth:
            out[index] = " "
            if text.startswith("/*", index):
                block_depth += 1
                out[index : index + 2] = [" ", " "]
                index += 2
            elif text.startswith("*/", index):
                block_depth -= 1
                out[index : index + 2] = [" ", " "]
                index += 2
            else:
                index += 1
            continue
        if string or char:
            if data[index] != "\n":
                out[index] = " "
            if escaped:
                escaped = False
            elif data[index] == "\\":
                escaped = True
            elif string and data[index] == '"':
                string = False
            elif char and data[index] == "'":
                char = False
            index += 1
            continue
        if text.startswith("//", index):
            while index < len(data) and data[index] != "\n":
                out[index] = " "
                index += 1
            continue
        if text.startswith("/*", index):
            block_depth = 1
            out[index : index + 2] = [" ", " "]
            index += 2
            continue
        if data[index] == "r":
            cursor = index + 1
            while cursor < len(data) and data[cursor] == "#":
                cursor += 1
            if cursor < len(data) and data[cursor] == '"':
                hashes = cursor - index - 1
                out[index : cursor + 1] = [" "] * (cursor + 1 - index)
                raw_end = '"' + ("#" * hashes)
                index = cursor + 1
                continue
        if data[index] == '"':
            string = True
            out[index] = " "
        elif (
            data[index] == "'"
            and index + 2 < len(data)
            and data[index + 2] == "'"
        ):
            char = True
            out[index] = " "
        elif (
            data[index] == "'"
            and index + 3 < len(data)
            and data[index + 1] == "\\"
            and data[index + 3] == "'"
        ):
            char = True
            out[index] = " "
        index += 1
    return "".join(out)


def item_pattern(selector: dict[str, Any]) -> re.Pattern[str]:
    kind = selector["kind"]
    name = re.escape(selector["name"])
    if kind == "fn":
        head = rf"(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+{name}\b"
    elif kind in {"struct", "enum", "type", "const", "static"}:
        head = rf"(?:pub(?:\([^)]*\))?\s+)?{kind}\s+{name}\b"
    elif kind == "impl":
        head = rf"impl(?:<[^>]+>\s*)?\s*{name}(?:<[^>]+>)?\b"
    elif kind == "module_body":
        head = rf"mod\s+{name}\s*\{{"
    elif kind == "file":
        return re.compile(r"\A", re.M)
    else:
        raise CheckError(f"unsupported selector kind: {kind}")
    return re.compile(rf"(?m)^[ \t]*{head}")


def declaration_start(text: str, matched_start: int) -> int:
    line_start = text.rfind("\n", 0, matched_start) + 1
    start = line_start
    while start > 0:
        previous_end = start - 1
        previous_start = text.rfind("\n", 0, previous_end) + 1
        previous = text[previous_start:previous_end]
        stripped = previous.strip()
        if stripped.startswith(("///", "//!", "#[")):
            start = previous_start
            continue
        break
    return start


def extract_selector(text: str, selector: dict[str, Any]) -> str:
    text = normalize_line_endings(text)
    if selector["kind"] == "file":
        return text
    masked = mask_rust(text)
    matches = list(item_pattern(selector).finditer(masked))
    ordinal = selector.get("ordinal", 1)
    if ordinal < 1 or len(matches) < ordinal:
        raise CheckError(
            f"{selector['kind']} {selector['name']} ordinal {ordinal} not found"
        )
    match = matches[ordinal - 1]
    start = declaration_start(text, match.start())
    declaration = masked[match.start() :]
    brace = declaration.find("{")
    semicolon = declaration.find(";")
    if selector["kind"] == "module_body":
        if brace < 0:
            raise CheckError(f"module {selector['name']} has no body")
        body_start = match.start() + brace + 1
        depth = 1
        cursor = body_start
        while cursor < len(masked) and depth:
            depth += (masked[cursor] == "{") - (masked[cursor] == "}")
            cursor += 1
        if depth:
            raise CheckError(f"unclosed module body: {selector['name']}")
        return text[body_start : cursor - 1]
    if semicolon >= 0 and (brace < 0 or semicolon < brace):
        return text[start : match.start() + semicolon + 1]
    if brace < 0:
        raise CheckError(f"braced item not found: {selector}")
    cursor = match.start() + brace
    depth = 0
    while cursor < len(masked):
        depth += (masked[cursor] == "{") - (masked[cursor] == "}")
        cursor += 1
        if depth == 0:
            return text[start:cursor]
    raise CheckError(f"unclosed item: {selector}")


def read_git(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise CheckError(result.stderr.strip())
    return result.stdout


def normalize_item(
    text: str,
    *,
    dedent: bool,
    allow_pub_super: bool,
) -> str:
    text = normalize_line_endings(text)
    if dedent:
        text = dedent_one_uniform_body(text)
    text = remove_allowlisted_pub_super(text, allow_pub_super)
    return text


def top_level_imports(text: str) -> set[str]:
    text = normalize_line_endings(text)
    masked = mask_rust(text)
    imports: set[str] = set()
    depth = 0
    start = None
    for index, char in enumerate(masked):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and start is None:
            prefix = masked[index:]
            match = re.match(r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?use\s+", prefix)
            if match and (index == 0 or masked[index - 1] == "\n"):
                start = index + match.start()
        if start is not None and depth == 0 and char == ";":
            statement = re.sub(r"\s+", " ", text[start : index + 1].strip())
            imports.add(statement)
            start = None
    return imports


def current_phase(manifest: dict[str, Any]) -> str:
    phase_path = ROOT / manifest["phase_file"]
    return phase_path.read_text().strip()


def check_manifest(manifest: dict[str, Any]) -> list[str]:
    phase = current_phase(manifest)
    errors = []
    old_cache: dict[str, str] = {}
    new_cache: dict[str, str] = {}
    for entry in manifest["items"]:
        phases = entry.get("phases")
        if phases and phase not in phases:
            continue
        target = ROOT / entry["new"]["path"]
        if not target.is_file():
            errors.append(f"{entry['id']}: missing target {entry['new']['path']}")
            continue
        try:
            old_path = entry["old"]["path"]
            old_text = old_cache.setdefault(
                old_path, read_git(manifest["old_ref"], old_path)
            )
            new_path = entry["new"]["path"]
            new_text = new_cache.setdefault(new_path, target.read_text())
            old_item = extract_selector(old_text, entry["old"]["selector"])
            new_item = extract_selector(new_text, entry["new"]["selector"])
            substitutions = entry.get("substitutions", [])
            old_item = apply_substitutions(old_item, substitutions, "old", phase)
            new_item = apply_substitutions(new_item, substitutions, "new", phase)
            old_normalized = normalize_item(
                old_item,
                dedent=entry.get("dedent_old", False),
                allow_pub_super=entry.get("allow_pub_super_old", False),
            )
            new_normalized = normalize_item(
                new_item,
                dedent=entry.get("dedent_new", False),
                allow_pub_super=entry.get("allow_pub_super_new", False),
            )
            if old_normalized != new_normalized:
                old_hash = hashlib.sha256(old_normalized.encode()).hexdigest()
                new_hash = hashlib.sha256(new_normalized.encode()).hexdigest()
                errors.append(
                    f"{entry['id']}: body mismatch {old_hash} != {new_hash}"
                )
        except CheckError as error:
            errors.append(f"{entry['id']}: {error}")

    import_spec = manifest["imports"]
    old_imports = set()
    for path in import_spec["old_paths"]:
        old_imports |= top_level_imports(
            old_cache.setdefault(path, read_git(manifest["old_ref"], path))
        )
    new_imports = set()
    for path in import_spec["new_paths"]:
        current = ROOT / path
        if current.is_file():
            new_imports |= top_level_imports(
                new_cache.setdefault(path, current.read_text())
            )
    expected_new = (
        old_imports
        - set(import_spec.get("allow_removed", []))
        | set(import_spec.get("allow_added", []))
    )
    if new_imports != expected_new:
        errors.append(
            "top-level import set mismatch: "
            f"missing={sorted(expected_new - new_imports)!r} "
            f"unexpected={sorted(new_imports - expected_new)!r}"
        )
    return errors


def selftest() -> None:
    crlf = "fn moved() {\r\n    use crate::Thing;\r\n}\r\n"
    assert normalize_line_endings(crlf) == (
        "fn moved() {\n    use crate::Thing;\n}\n"
    )
    body = "    fn one() {\n        use crate::Thing;\n    }\n"
    assert dedent_one_uniform_body(body) == (
        "fn one() {\n    use crate::Thing;\n}\n"
    )
    widened = "pub(super) async fn moved() {}\n"
    assert remove_allowlisted_pub_super(widened, True) == "async fn moved() {}\n"
    try:
        remove_allowlisted_pub_super(widened, False)
    except CheckError:
        pass
    else:
        raise AssertionError("non-allowlisted pub(super) widening must fail")

    old = 'fn reflects() { let source = include_str!("post_write.rs"); }\n'
    substituted = apply_substitutions(
        old,
        [
            {
                "side": "old",
                "from": 'include_str!("post_write.rs")',
                "to": 'include_str!("../post_write.rs")',
                "count": 1,
            }
        ],
        "old",
        "tests-externalized",
    )
    assert '../post_write.rs' in substituted

    imports = top_level_imports(
        "use crate::{A, B};\nfn body() { use crate::Inside; }\nuse std::path::Path;\n"
    )
    assert imports == {"use crate::{A, B};", "use std::path::Path;"}
    assert normalize_item(
        "fn body() {\n    use crate::Inside;\n}\n",
        dedent=False,
        allow_pub_super=False,
    ) != normalize_item(
        "fn body() {\n}\n",
        dedent=False,
        allow_pub_super=False,
    ), "a body-local import must remain part of the item hash"

    original = "async fn stable() {\n    do_work(token);\n}"
    mutation = original.replace("token", "other_token")
    assert normalize_item(
        original, dedent=False, allow_pub_super=False
    ) != normalize_item(mutation, dedent=False, allow_pub_super=False), (
        "a non-allowlisted body/token mutation must fail comparison"
    )

    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    old_cache: dict[str, str] = {}
    checked = 0
    for entry in manifest["items"]:
        old_path = entry["old"]["path"]
        old_text = old_cache.setdefault(
            old_path, read_git(manifest["old_ref"], old_path)
        )
        selector = entry["old"]["selector"]
        selected = extract_selector(old_text, selector)
        if not selected.strip():
            raise AssertionError(f"{entry['id']}: baseline selector extracted no bytes")
        if selector["kind"] != "module_body":
            local_selector = dict(selector)
            local_selector["ordinal"] = 1
            if len(list(item_pattern(local_selector).finditer(mask_rust(selected)))) != 1:
                raise AssertionError(
                    f"{entry['id']}: baseline selector did not isolate one named item"
                )
        checked += 1

    for path in manifest["imports"]["old_paths"]:
        source = old_cache.setdefault(
            path, read_git(manifest["old_ref"], path)
        )
        for statement in top_level_imports(source):
            if not statement.startswith(("use ", "pub use ", "pub(crate) use ")):
                raise AssertionError(
                    f"{path}: malformed top-level import extraction: {statement!r}"
                )
            if not statement.endswith(";"):
                raise AssertionError(
                    f"{path}: unterminated top-level import extraction: {statement!r}"
                )

    print(
        f"post-write move comparator selftest: ok "
        f"({checked} baseline selectors)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return 0
    manifest = json.loads(args.manifest.read_text())
    errors = check_manifest(manifest)
    if errors:
        print("post_write movement comparison failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"post_write movement comparison: ok ({len(manifest['items'])} manifest items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
