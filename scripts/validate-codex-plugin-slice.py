#!/usr/bin/env python3
"""Validate the hand-authored Codex plugin surface.

This is intentionally narrower than the Codex plugin validator. It checks the
Wenlan-specific surface contract before any shared generator exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin-codex"
REQUIRED_SKILL_INTERFACE = {
    "brief": {
        "display_name": "Wenlan Brief",
        "short_description": "Read the Space Brief and optional related context",
    },
    "capture": {
        "display_name": "Wenlan Capture",
        "short_description": "Save a durable memory to Wenlan from the current conversation",
    },
    "curate": {
        "display_name": "Wenlan Curate",
        "short_description": "Review pending Wenlan captures or revisions from Codex",
    },
    "distill": {
        "display_name": "Wenlan Distill",
        "short_description": "Synthesize or refresh source-backed Wenlan pages",
    },
    "forget": {
        "display_name": "Wenlan Forget",
        "short_description": "Delete a Wenlan memory by exact id with confirmation",
    },
    "handoff": {
        "display_name": "Wenlan Handoff",
        "short_description": "End a Codex session with captures and a typed Brief update",
    },
    "help": {
        "display_name": "Wenlan Help",
        "short_description": "Show the Codex Wenlan command reference",
    },
    "lint": {
        "display_name": "Wenlan Lint",
        "short_description": "Diagnose and resolve every finding without hidden mutation",
    },
    "pages": {
        "display_name": "Wenlan Pages",
        "short_description": "List or open distilled Wenlan pages from Codex",
    },
    "recall": {
        "display_name": "Wenlan Recall",
        "short_description": "Search Wenlan memories from Codex",
    },
    "setup": {
        "display_name": "Wenlan Setup",
        "short_description": "Set up and verify the local Wenlan runtime and MCP bridge",
    },
}
SKILLS_WITHOUT_MCP_REFERENCE = {"handoff", "help", "pages"}
SKILLS_USING_RESOLVER = {"brief", "capture", "distill", "handoff", "lint", "recall"}
REQUIRED_GUARDRAILS = {
    "forget": [
        "cannot be undone",
        "delete <id>",
        "Always confirm with the user before calling forget",
    ],
    "distill": [
        "rebuild <page-id>",
        "force=true",
        "user-edited page prose is wiped",
    ],
    "curate": [
        "revision_source_id",
        "Perform no mutation until the user replies",
        "Ambiguous replies do not mutate",
    ],
    "lint": [
        "Population truncation is honest coverage metadata, not an automatic incomplete result.",
        "every packet candidate has exactly one accepted verdict",
        "trust the typed report's `complete` flag and preserve its denominator, evaluated, and truncation metadata",
        "Lint creates durable Review Items for choices that are not yet exact.",
        "turns exactly one Review Item into a separately approved manifest",
        "`lint_repair_review` generic accept remains rejected and non-mutating",
        "Never call `apply_lint_repair` in the same turn as `prepare_lint_repair`.",
        "prepare_lint_repair",
        "Lead repair output with exactly one compact typed-count funnel",
        "Never substitute check, family, or candidate counts for occurrence counts",
        "one or more exact approval lines",
        "contain only ready tuples from the same displayed plan",
        "no duplicates, blank lines, prose, or code fences",
        "A single line remains valid",
        "Validate the complete reply before the first apply",
        "one contiguous copy-pasteable block",
        "in the order they appear among ready tuples in the displayed plan",
        "Immediately rerun fresh General once",
        "Only after one manifest is `verified` may the next approved manifest begin",
        "`next_apply`",
        "bounded daemon handoff reservation",
        "Do not apply any later approved manifest",
        "`verified`, `applied_unverified`, `failed`, or `not_attempted`",
        "Use `applied_unverified` whenever an apply receipt exists but durable verification does not",
    ],
}
CLAUDE_ONLY_TOKENS = (
    "CLAUDE_PLUGIN_ROOT",
    ".claude-plugin",
    "mcp__plugin_wenlan_wenlan__",
)


def fail(message: str) -> None:
    print(f"codex plugin slice validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing {path.relative_to(ROOT)}")
    except json.JSONDecodeError as exc:
        fail(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path.relative_to(ROOT)} must contain a JSON object")
    return payload


def assert_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing {path.relative_to(ROOT)}")


def assert_no_claude_tokens(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for token in CLAUDE_ONLY_TOKENS:
        if token in text:
            fail(f"{path.relative_to(ROOT)} contains Claude-only token {token!r}")


def frontmatter_value(text: str, key: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return None
    prefix = f"{key}:"
    for line in lines[1:]:
        if line == "---":
            return None
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip('"')
    return None


def shared_codex_skills() -> tuple[str, ...]:
    contract = read_json(ROOT / "plugin-contract.json")
    skills: list[str] = []
    for item in contract.get("skills", []):
        if not isinstance(item, dict):
            fail("plugin-contract.json skills entries must be objects")
        if item.get("status") == "shared_now":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                fail("shared plugin-contract skill entries must have names")
            if item.get("codex_user_invocable") is not True:
                fail(f"shared skill {name} must set codex_user_invocable true")
            if item.get("codex_openai_interface") is not True:
                fail(f"shared skill {name} must set codex_openai_interface true")
            skills.append(name)
    return tuple(sorted(skills))


def validate_manifest() -> None:
    manifest = read_json(PLUGIN / ".codex-plugin" / "plugin.json")
    if manifest.get("name") != "wenlan":
        fail("plugin manifest name must stay `wenlan`")
    if "hooks" in manifest:
        fail("Codex plugin manifest must not declare hooks")
    if manifest.get("skills") != "./skills/":
        fail("plugin manifest must point skills at ./skills/")
    if manifest.get("mcpServers") != "./.mcp.json":
        fail("plugin manifest must point mcpServers at ./.mcp.json")
    interface = manifest.get("interface")
    if not isinstance(interface, dict):
        fail("plugin manifest must include interface metadata")
    prompts = interface.get("defaultPrompt")
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 3:
        fail("interface.defaultPrompt must contain 1-3 starter prompts")


def validate_mcp() -> None:
    mcp = read_json(PLUGIN / ".mcp.json")
    servers = mcp.get("mcpServers")
    if not isinstance(servers, dict):
        fail(".mcp.json must contain mcpServers object")
    wenlan = servers.get("wenlan")
    if not isinstance(wenlan, dict):
        fail(".mcp.json must define the wenlan MCP server")
    if wenlan.get("command") != "./bin/wenlan-mcp-runner.sh":
        fail("wenlan MCP server must run ./bin/wenlan-mcp-runner.sh")
    if wenlan.get("cwd") != ".":
        fail("wenlan MCP server must resolve its runner from plugin cwd")


def validate_runner() -> None:
    runner = PLUGIN / "bin" / "wenlan-mcp-runner.sh"
    assert_file(runner)
    text = runner.read_text(encoding="utf-8")
    if "--agent-name" not in text or "codex" not in text:
        fail("Codex MCP runner must pass an explicit codex agent name")


def validate_resolver() -> None:
    resolver = PLUGIN / "bin" / "resolve-space.sh"
    assert_file(resolver)
    assert_no_claude_tokens(resolver)


def validate_skills() -> None:
    required_skills = shared_codex_skills()
    if set(REQUIRED_SKILL_INTERFACE) != set(required_skills):
        fail(
            "REQUIRED_SKILL_INTERFACE must exactly match shared Codex skills: "
            f"{sorted(required_skills)}"
        )
    for skill in required_skills:
        path = PLUGIN / "skills" / skill / "SKILL.md"
        metadata_path = PLUGIN / "skills" / skill / "agents" / "openai.yaml"
        assert_file(path)
        assert_file(metadata_path)
        assert_no_claude_tokens(path)
        assert_no_claude_tokens(metadata_path)
        text = path.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        if f"name: {skill}" not in text:
            fail(f"{path.relative_to(ROOT)} frontmatter must name {skill}")
        if skill == "lint" and frontmatter_value(text, "argument-hint") != (
            "[deep|repair] [global|uncategorized|space:<name>]"
        ):
            fail(f"{path.relative_to(ROOT)} exposes the wrong lint argument grammar")
        if "user-invocable: true" not in text:
            fail(f"{path.relative_to(ROOT)} must be marked user-invocable for slash autocomplete")
        if skill not in SKILLS_WITHOUT_MCP_REFERENCE and "mcp__wenlan__" not in text:
            fail(f"{path.relative_to(ROOT)} must use Codex wenlan MCP tool names")
        if skill in SKILLS_USING_RESOLVER and "plugin-codex/bin/resolve-space.sh" not in text:
            fail(f"{path.relative_to(ROOT)} must use plugin-codex/bin/resolve-space.sh")
        for needle in REQUIRED_GUARDRAILS.get(skill, []):
            if " ".join(needle.split()) not in normalized_text:
                fail(f"{path.relative_to(ROOT)} must contain guardrail {needle!r}")
        metadata = metadata_path.read_text(encoding="utf-8")
        expected = REQUIRED_SKILL_INTERFACE[skill]
        if "interface:" not in metadata:
            fail(f"{metadata_path.relative_to(ROOT)} must declare Codex app interface metadata")
        for key, value in expected.items():
            expected_line = f'  {key}: "{value}"'
            if expected_line not in metadata:
                fail(f"{metadata_path.relative_to(ROOT)} must contain {expected_line}")


def validate_pages_skill() -> None:
    path = PLUGIN / "skills" / "pages" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    required = [
        "wenlan pages",
        "Never read a page body",
        "Do not use a picker",
        "If several pages match, print the CLI output",
        "If `wenlan` is not found, tell the user to run `/setup`",
    ]
    for needle in required:
        if needle not in text:
            fail(f"{path.relative_to(ROOT)} must contain {needle!r}")
    forbidden = [
        "AskUserQuestion",
        "native picker",
        "command sandbox DISABLED",
        "mcp__plugin_wenlan_wenlan__",
    ]
    for needle in forbidden:
        if needle in text:
            fail(f"{path.relative_to(ROOT)} must not contain {needle!r}")


def main() -> None:
    validate_manifest()
    validate_mcp()
    validate_runner()
    validate_resolver()
    validate_skills()
    validate_pages_skill()
    print("Codex plugin slice validation passed")


if __name__ == "__main__":
    main()
