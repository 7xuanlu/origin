#!/usr/bin/env python3
"""Validate the shared Wenlan Claude/Codex plugin contract.

The contract is deliberately small: it documents which skills are shared now,
which Claude skills are not ported yet, and the surface-specific MCP, runner,
and marketplace rules that should not drift silently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SKILL_STATUSES = {"shared_now", "claude_only_until_ported"}
CODEX_SKILLS_WITHOUT_MCP_REFERENCE = {"help", "pages"}
CODEX_SKILLS_USING_RESOLVER = {
    "brief",
    "capture",
    "distill",
    "handoff",
    "lint",
    "recall",
}
CODEX_REQUIRED_GUARDRAILS = {
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
}
LINT_SHARED_GUARDRAILS = [
    "/lint repair",
    "[deep|repair] [global|uncategorized|space:<name>]",
    "General uses exactly one lint MCP call",
    "Agent-assisted Deep uses exactly two lint MCP calls",
    "never evaluate records outside agent_work",
    "get_lint_agent_work_page",
    "submit verdicts exactly once",
    "authorized record refs (`evidence_refs` plus `counterevidence_refs`)",
    "Do not mechanically copy every evidence ref",
    "Population truncation is honest coverage metadata, not an automatic incomplete result.",
    "every packet candidate has exactly one accepted verdict",
    "trust the typed report's `complete` flag and preserve its denominator, evaluated, and truncation metadata",
    "If Deep is incomplete, its producer receipt differs from General, or its DB analysis digest differs from General, rerun fresh General exactly once after Deep before prepare; do not rerun Deep.",
    "Do not compare General and Deep Page digests across profiles because their Page scan coverage intentionally differs.",
    "Lead repair output with exactly one compact typed-count funnel",
    "Never substitute check, family, or candidate counts for occurrence counts",
    "CLI fallback: `wenlan lint --profile deep --agent_assist`",
    "global",
    "uncategorized",
    "Plain `/lint`, `/lint deep`, the lint MCP tool, and `/api/lint` are fully read-only",
    "Lint creates durable Review Items for choices that are not yet exact.",
    "turns exactly one Review Item into a separately approved manifest",
    "`lint_repair_review` generic accept remains rejected and non-mutating",
    "Never call `apply_lint_repair` in the same turn as `prepare_lint_repair`.",
    "prepare_lint_repair",
    "prepare_lint_repair_plan",
    "get_lint_repair_plan_entries",
    "every observed family",
    "`ready`",
    "`review`",
    "`system_action`",
    "`blocked`",
    "artifact_path",
    "canonical data is unchanged",
    "apply repair <manifest-id> <manifest-digest>",
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
    "Never call `apply_lint_repair` in the same turn as `prepare_lint_repair_plan`",
    "applied_unverified",
    "Match every line byte-for-byte",
    "the MCP repair-manifest tools have no CLI equivalent yet",
]
ENRICHMENT_CONSENT_GUARDRAILS = [
    "wenlan enrichment status",
    "wenlan enrichment configure --everyday <source> --synthesis <source>",
    "wenlan enrichment disable",
]


def fail(message: str) -> None:
    print(f"plugin contract validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing {rel(root, path)}")
    except json.JSONDecodeError as exc:
        fail(f"{rel(root, path)} is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        fail(f"{rel(root, path)} must contain a JSON object")
    return payload


def read_text(root: Path, path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"missing {rel(root, path)}")


def require_file(root: Path, path: Path) -> None:
    if not path.is_file():
        fail(f"missing {rel(root, path)}")


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        fail(f"{label} must be {expected!r}, got {actual!r}")


def load_contract(root: Path) -> dict[str, Any]:
    contract = read_json(root, root / "plugin-contract.json")
    require_equal("plugin-contract.json schema_version", contract.get("schema_version"), 1)
    surfaces = contract.get("surfaces")
    if not isinstance(surfaces, dict):
        fail("plugin-contract.json must contain surfaces object")
    for surface in ("claude", "codex"):
        if not isinstance(surfaces.get(surface), dict):
            fail(f"plugin-contract.json surfaces must include {surface}")
    skills = contract.get("skills")
    if not isinstance(skills, list) or not skills:
        fail("plugin-contract.json must contain non-empty skills array")
    return contract


def contract_skill_sets(contract: dict[str, Any]) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    shared_now: set[str] = set()
    claude_only: set[str] = set()
    for item in contract["skills"]:
        if not isinstance(item, dict):
            fail("each plugin-contract skill entry must be an object")
        name = item.get("name")
        status = item.get("status")
        if not isinstance(name, str) or not name:
            fail("each plugin-contract skill entry must have a name")
        if name in names:
            fail(f"duplicate plugin-contract skill entry {name!r}")
        names.add(name)
        if status not in ALLOWED_SKILL_STATUSES:
            fail(f"skill {name} has unsupported status {status!r}")
        if status == "shared_now":
            if item.get("codex_user_invocable") is not True:
                fail(f"shared skill {name} must set codex_user_invocable true")
            if item.get("codex_openai_interface") is not True:
                fail(f"shared skill {name} must set codex_openai_interface true")
            shared_now.add(name)
        elif status == "claude_only_until_ported":
            claude_only.add(name)
            if not item.get("reason"):
                fail(f"claude-only skill {name} must document why it is not ported yet")
    return shared_now, claude_only


def skill_names(root: Path, plugin_root: Path) -> set[str]:
    skills_root = plugin_root / "skills"
    if not skills_root.is_dir():
        fail(f"missing {rel(root, skills_root)}")
    return {
        path.parent.name
        for path in skills_root.glob("*/SKILL.md")
        if path.is_file()
    }


def parse_frontmatter(root: Path, path: Path) -> dict[str, str]:
    text = read_text(root, path)
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        fail(f"{rel(root, path)} must start with frontmatter")
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line == "---":
            return metadata
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match:
            metadata[match.group(1)] = match.group(2).strip().strip('"')
    fail(f"{rel(root, path)} frontmatter is not closed")


def validate_manifest(root: Path, surface: str, config: dict[str, Any]) -> None:
    manifest_path = root / config["manifest_path"]
    manifest = read_json(root, manifest_path)
    require_equal(f"{rel(root, manifest_path)} name", manifest.get("name"), config["manifest_name"])
    if "manifest_category" in config:
        require_equal(
            f"{rel(root, manifest_path)} category",
            manifest.get("category"),
            config["manifest_category"],
        )
    if surface == "codex":
        require_equal(
            f"{rel(root, manifest_path)} skills",
            manifest.get("skills"),
            config["manifest_skills"],
        )
        require_equal(
            f"{rel(root, manifest_path)} mcpServers",
            manifest.get("mcpServers"),
            config["manifest_mcp_servers"],
        )
        if "hooks" in manifest:
            fail(f"{rel(root, manifest_path)} must not declare hooks")


def validate_mcp_config(root: Path, surface: str, config: dict[str, Any]) -> None:
    path = root / config["mcp_config_path"]
    mcp = read_json(root, path)
    servers = mcp.get("mcpServers")
    if not isinstance(servers, dict):
        fail(f"{rel(root, path)} must contain mcpServers object")
    server_name = config["mcp_server_name"]
    server = servers.get(server_name)
    if not isinstance(server, dict):
        fail(f"{rel(root, path)} must define MCP server {server_name!r}")
    require_equal(
        f"{surface} MCP command",
        server.get("command"),
        config["mcp_command"],
    )
    if "mcp_cwd" in config:
        require_equal(
            f"{surface} MCP cwd",
            server.get("cwd"),
            config["mcp_cwd"],
        )


def validate_runner(root: Path, surface: str, config: dict[str, Any]) -> None:
    runner_path = root / config["runner_path"]
    runner = read_text(root, runner_path)
    agent = config["agent_name"]
    if agent["mode"] == "runner_argument_default":
        env = agent["env"]
        value = agent["value"]
        if f'{env}:-{value}' not in runner:
            fail(f"{rel(root, runner_path)} must default {env} to {value}")
        if '--agent-name "${agent_name}"' not in runner:
            fail(f"{rel(root, runner_path)} must pass --agent-name through the runner")
    elif agent["mode"] == "wenlan_mcp_stdio_default":
        if "--agent-name" in runner:
            fail(f"{rel(root, runner_path)} must rely on the wenlan-mcp stdio default agent")
        source = read_text(root, root / "crates" / "wenlan-mcp" / "src" / "main.rs")
        pattern = re.compile(
            r"if\s+serve_args\.is_some\(\)\s*\{.*?\"remote-mcp\"\.into\(\).*?\}"
            r"\s*else\s*\{.*?\"" + re.escape(agent["value"]) + r"\"\.into\(\).*?\}",
            re.DOTALL,
        )
        if not pattern.search(source):
            fail(f"wenlan-mcp stdio default agent must remain {agent['value']!r}")
    else:
        fail(f"{surface} has unsupported agent mode {agent['mode']!r}")


def validate_resolver_parity(root: Path) -> None:
    claude_resolver = read_text(root, root / "plugin" / "bin" / "resolve-space.sh")
    codex_resolver = read_text(root, root / "plugin-codex" / "bin" / "resolve-space.sh")
    if claude_resolver != codex_resolver:
        fail("plugin-codex/bin/resolve-space.sh must match plugin/bin/resolve-space.sh")


def validate_codex_readme(root: Path) -> None:
    path = root / "plugin-codex" / "README.md"
    text = read_text(root, path)
    if not re.search(r"(?<![A-Za-z0-9_-])/setup\b", text):
        fail(f"{rel(root, path)} must direct users to /setup")
    if re.search(r"(?<![A-Za-z0-9_-])/init\b", text):
        fail(f"{rel(root, path)} must not advertise the retired /init command")


def iter_matching_plugin(plugins: list[Any], plugin_name: str):
    for item in plugins:
        if isinstance(item, dict) and item.get("name") == plugin_name:
            yield item


def validate_marketplace(root: Path, surface: str, config: dict[str, Any]) -> None:
    marketplace_config = config.get("marketplace")
    if not isinstance(marketplace_config, dict):
        fail(f"{surface} contract must include marketplace object")
    path = root / marketplace_config["path"]
    marketplace = read_json(root, path)
    require_equal(f"{rel(root, path)} name", marketplace.get("name"), marketplace_config["name"])
    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        fail(f"{rel(root, path)} must contain plugins array")
    plugin = next(iter_matching_plugin(plugins, marketplace_config["plugin_name"]), None)
    if plugin is None:
        fail(f"{rel(root, path)} must contain plugin {marketplace_config['plugin_name']!r}")
    label = f"{surface.capitalize()} marketplace"
    source = plugin.get("source")
    if not isinstance(source, dict):
        fail(f"{rel(root, path)} plugin source must be an object")
    require_equal(f"{label} source", source.get("source"), marketplace_config["source"])
    require_equal(
        f"{label} source.path",
        source.get("path"),
        marketplace_config["source_path"],
    )
    if "source_url" in marketplace_config:
        require_equal(
            f"{label} source.url",
            source.get("url"),
            marketplace_config["source_url"],
        )
    if "policy_installation" in marketplace_config:
        policy = plugin.get("policy")
        if not isinstance(policy, dict):
            fail(f"{rel(root, path)} plugin policy must be an object")
        require_equal(
            f"{label} policy.installation",
            policy.get("installation"),
            marketplace_config["policy_installation"],
        )
        require_equal(
            f"{label} policy.authentication",
            policy.get("authentication"),
            marketplace_config["policy_authentication"],
        )
    require_equal(
        f"{label} category",
        plugin.get("category"),
        marketplace_config["category"],
    )
    # The storefront listing is not a second copy to keep in sync by hand: the
    # blurb and the discovery tags must be the manifest's own.
    manifest = read_json(root, root / config["manifest_path"])
    for field in ("description", "keywords"):
        if field in plugin:
            require_equal(
                f"{label} {field}",
                plugin.get(field),
                manifest.get(field),
            )


def validate_skill_surface(
    root: Path,
    surface: str,
    config: dict[str, Any],
    expected_names: set[str],
    shared_now: set[str],
    contract: dict[str, Any],
) -> None:
    plugin_root = root / config["plugin_root"]
    actual_names = skill_names(root, plugin_root)
    if "lint-repair" in actual_names or "lint-repair" in expected_names:
        fail(f"{surface} must expose repair only through the lint skill")
    if actual_names != expected_names:
        fail(
            f"{surface} skill inventory drift: expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}"
        )

    expected_prefix = config["skill_mcp_tool_prefix"]
    other_prefix = (
        contract["surfaces"]["codex"]["skill_mcp_tool_prefix"]
        if surface == "claude"
        else contract["surfaces"]["claude"]["skill_mcp_tool_prefix"]
    )

    for name in sorted(expected_names):
        skill_path = plugin_root / "skills" / name / "SKILL.md"
        require_file(root, skill_path)
        frontmatter = parse_frontmatter(root, skill_path)
        require_equal(f"{rel(root, skill_path)} frontmatter name", frontmatter.get("name"), name)
        text = read_text(root, skill_path)
        if other_prefix in text:
            fail(f"{rel(root, skill_path)} contains wrong MCP tool prefix {other_prefix!r}")
        mcp_tools = set(re.findall(r"mcp__[A-Za-z0-9_]+__[A-Za-z0-9_]+", text))
        wrong_tools = sorted(token for token in mcp_tools if not token.startswith(expected_prefix))
        if wrong_tools:
            fail(f"{rel(root, skill_path)} contains unexpected MCP tools: {wrong_tools}")
        if mcp_tools and not any(token.startswith(expected_prefix) for token in mcp_tools):
            fail(f"{rel(root, skill_path)} must use MCP prefix {expected_prefix!r}")
        if name == "lint":
            require_equal(
                f"{rel(root, skill_path)} argument-hint",
                frontmatter.get("argument-hint"),
                "[deep|repair] [global|uncategorized|space:<name>]",
            )
            for tool in (
                "lint",
                "get_lint_agent_work_page",
                "prepare_lint_repair",
                "prepare_lint_repair_plan",
                "get_lint_repair_plan_entries",
                "apply_lint_repair",
                "verify_lint_repair",
            ):
                expected_tool = f"{expected_prefix}{tool}"
                if expected_tool not in text:
                    fail(f"{rel(root, skill_path)} must call {expected_tool!r}")
            try:
                allowed_tools = json.loads(frontmatter.get("allowed-tools", ""))
            except json.JSONDecodeError:
                fail(f"{rel(root, skill_path)} allowed-tools must be a JSON array")
            expected_allowed_tools = {
                "Bash",
                *(f"{expected_prefix}{tool}" for tool in (
                    "lint",
                    "get_lint_agent_work_page",
                    "prepare_lint_repair",
                    "prepare_lint_repair_plan",
                    "get_lint_repair_plan_entries",
                    "apply_lint_repair",
                    "verify_lint_repair",
                )),
            }
            if not isinstance(allowed_tools, list) or set(allowed_tools) != expected_allowed_tools:
                fail(
                    f"{rel(root, skill_path)} allowed-tools must be exactly "
                    f"{sorted(expected_allowed_tools)!r}"
                )
            normalized_text = " ".join(text.split())
            for needle in LINT_SHARED_GUARDRAILS:
                if needle not in normalized_text:
                    fail(f"{rel(root, skill_path)} must contain guardrail {needle!r}")
            resolver = (
                "$CLAUDE_PLUGIN_ROOT/bin/resolve-space.sh"
                if surface == "claude"
                else "plugin-codex/bin/resolve-space.sh"
            )
            if resolver not in text:
                fail(f"{rel(root, skill_path)} must use {resolver}")
        if name == "help" and "/lint [deep|repair] [scope]" not in text:
            fail(f"{rel(root, skill_path)} must advertise the unified lint grammar")
        if name in {"help", "setup"}:
            normalized_text = " ".join(text.split())
            for needle in ENRICHMENT_CONSENT_GUARDRAILS:
                if needle not in normalized_text:
                    fail(
                        f"{rel(root, skill_path)} must delegate background consent through {needle!r}"
                    )

        if surface == "codex" and name in shared_now:
            require_equal(
                f"{rel(root, skill_path)} user-invocable",
                frontmatter.get("user-invocable"),
                "true",
            )
            if name not in CODEX_SKILLS_WITHOUT_MCP_REFERENCE and expected_prefix not in text:
                fail(f"{rel(root, skill_path)} must use MCP prefix {expected_prefix!r}")
            if name in CODEX_SKILLS_USING_RESOLVER and "plugin-codex/bin/resolve-space.sh" not in text:
                fail(f"{rel(root, skill_path)} must use plugin-codex/bin/resolve-space.sh")
            for needle in CODEX_REQUIRED_GUARDRAILS.get(name, []):
                if needle not in text:
                    fail(f"{rel(root, skill_path)} must contain guardrail {needle!r}")
            metadata_path = plugin_root / "skills" / name / "agents" / "openai.yaml"
            require_file(root, metadata_path)
            metadata = read_text(root, metadata_path)
            if "interface:" not in metadata:
                fail(f"{rel(root, metadata_path)} must declare interface metadata")


def validate_contract(root: Path) -> None:
    contract = load_contract(root)
    shared_now, claude_only = contract_skill_sets(contract)
    surfaces = contract["surfaces"]

    for surface, config in surfaces.items():
        validate_manifest(root, surface, config)
        validate_mcp_config(root, surface, config)
        validate_runner(root, surface, config)

    validate_resolver_parity(root)
    validate_codex_readme(root)
    validate_marketplace(root, "claude", surfaces["claude"])
    validate_marketplace(root, "codex", surfaces["codex"])

    validate_skill_surface(
        root,
        "claude",
        surfaces["claude"],
        shared_now | claude_only,
        shared_now,
        contract,
    )
    validate_skill_surface(
        root,
        "codex",
        surfaces["codex"],
        shared_now,
        shared_now,
        contract,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    validate_contract(args.root.resolve())
    print("Plugin contract validation passed")


if __name__ == "__main__":
    main()
