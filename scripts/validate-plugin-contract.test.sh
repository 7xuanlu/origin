#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VALIDATOR="$PWD/scripts/validate-plugin-contract.py"
TMPDIR_TEST=$(mktemp -d)
trap "rm -rf $TMPDIR_TEST" EXIT

copy_fixture() {
    rm -rf "$TMPDIR_TEST/root"
    mkdir -p "$TMPDIR_TEST/root/crates/wenlan-mcp/src"
    cp -R plugin "$TMPDIR_TEST/root/plugin"
    cp -R plugin-codex "$TMPDIR_TEST/root/plugin-codex"
    cp -R .agents "$TMPDIR_TEST/root/.agents"
    cp -R .claude-plugin "$TMPDIR_TEST/root/.claude-plugin"
    cp plugin-contract.json "$TMPDIR_TEST/root/plugin-contract.json"
    cp crates/wenlan-mcp/src/main.rs "$TMPDIR_TEST/root/crates/wenlan-mcp/src/main.rs"
}

assert_rejects() {
    local name="$1"
    shift
    copy_fixture
    "$@"
    if python3 "$VALIDATOR" --root "$TMPDIR_TEST/root" 2>/dev/null; then
        echo "FAIL $name: validator accepted drift"
        exit 1
    fi
    echo "PASS $name"
}

copy_fixture
python3 "$VALIDATOR" --root "$TMPDIR_TEST/root"
echo "PASS valid plugin contract"

assert_rejects "brief context adapter regression" \
    perl -0pi -e 's/mcp__wenlan__brief/mcp__wenlan__context/g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/brief/SKILL.md"

assert_rejects "brief mandatory session-start regression" \
    perl -0pi -e 's/It is not a mandatory\s+every-session boot step\./Call FIRST at session start./g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/brief/SKILL.md"

assert_rejects "brief Markdown authority regression" \
    perl -0pi -e 's/must\s+never be read as product state/must cat ~\/.wenlan\/sessions\/_status\/<project>.md/gi' \
    "$TMPDIR_TEST/root/plugin/skills/brief/SKILL.md"

assert_rejects "handoff missing Brief read" \
    perl -0pi -e 's/This read is mandatory before any Brief\s+delta is authored for a registered Space\./Compose deltas from conversation./g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/handoff/SKILL.md"

assert_rejects "handoff auto-demotion regression" \
    perl -0pi -e 's/Never auto-demote untouched Active work\./Auto-demote untouched Active items./g' \
    "$TMPDIR_TEST/root/plugin/skills/handoff/SKILL.md"

assert_rejects "handoff direct status rewrite regression" \
    perl -0pi -e 's/Never read, edit, or overwrite that receipt as\s+authority\./Overwrite `~\/.wenlan\/sessions\/_status\/<project>.md`./g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/handoff/SKILL.md"

assert_rejects "handoff missing repo-basename fallback" \
    perl -0pi -e 's/cwd-repo-new/unscoped/g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/handoff/SKILL.md"

assert_rejects "handoff guesses past Space absence probe" \
    perl -0pi -e 's/For `cwd-repo-new`, prove the Space is absent with\s+`spaces show` before composing deltas\./Treat any read failure as a first-handoff absence./g' \
    "$TMPDIR_TEST/root/plugin/skills/handoff/SKILL.md"

assert_rejects "handoff invents a non-repository Space" \
    perl -0pi -e 's/Outside a Git repository, do not derive a new Space from the directory\s+basename\./Outside a Git repository, use the directory basename as a new Space./g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/handoff/SKILL.md"

assert_rejects "handoff captures before first Space creation" \
    perl -0pi -e 's/Apply the Brief update before Space-scoped captures when this fallback is new./Store captures before the Brief update./g' \
    "$TMPDIR_TEST/root/plugin/skills/handoff/SKILL.md"

assert_rejects "handoff same-item snapshot version drift" \
    perl -0pi -e 's/Every delta for one existing item uses the same version from the pre-handoff Brief snapshot./Chain item versions inside the request./g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/handoff/SKILL.md"

python3 - "$TMPDIR_TEST/root/plugin-codex/.mcp.json" <<'PY'
import json
import sys

server = json.load(open(sys.argv[1], encoding="utf-8"))["mcpServers"]["wenlan"]
if server.get("cwd") != ".":
    raise SystemExit("FAIL codex MCP plugin-relative cwd: expected cwd='.'")
PY
echo "PASS codex MCP plugin-relative cwd"

assert_rejects "codex skill autocomplete drift" \
    perl -0pi -e 's/user-invocable: true/user-invocable: false/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/brief/SKILL.md"

assert_rejects "claude skill copied into codex" \
    bash -c 'rm -rf "$1"; cp -R "$2" "$1"' _ \
    "$TMPDIR_TEST/root/plugin-codex/skills/handoff" \
    "$TMPDIR_TEST/root/plugin/skills/handoff"

assert_rejects "missing destructive confirmation wording" \
    perl -0pi -e 's/cannot be undone/cannot be reversed/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/forget/SKILL.md"

assert_rejects "missing curate ambiguity guardrail" \
    perl -0pi -e 's/Ambiguous replies do not mutate/Ambiguous replies are clarified/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate refinements explicit-only drift" \
    perl -0pi -e 's/Use `\/curate refinements` only when the user explicitly asks to inspect or\s+review the daemon proposal\/refinement queue\./Inspect refinements whenever useful./' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate refinements explicit-only drift" \
    perl -0pi -e 's/Use `\/curate refinements` only when the user explicitly asks to inspect or\s+review the daemon proposal\/refinement queue\./Inspect refinements whenever useful./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate refinements mutation gate drift" \
    perl -0pi -e 's/Perform no mutation until the user gives an unambiguous item-level accept or\s+reject decision\./Apply likely decisions immediately./' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate refinements mutation gate drift" \
    perl -0pi -e 's/Perform no mutation until the user gives an unambiguous item-level accept or\s+reject decision\./Apply likely decisions immediately./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate refinements skip drift" \
    perl -0pi -e 's/Skip or cancel is a no-op\./Skip accepts the proposal./' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate refinements skip drift" \
    perl -0pi -e 's/Skip or cancel is a no-op\./Skip accepts the proposal./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate missing refinement accept permission" \
    perl -0pi -e 's/, "mcp__plugin_wenlan_wenlan__accept_refinement"//' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate missing refinement reject permission" \
    perl -0pi -e 's/, "mcp__wenlan__reject_refinement"//' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate missing existing capture permission" \
    perl -0pi -e 's/, "mcp__plugin_wenlan_wenlan__capture"//' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate missing existing list permission" \
    perl -0pi -e 's/, "mcp__wenlan__list_pending"//' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate added unintended mutation permission" \
    perl -0pi -e 's/"Bash",/"Bash", "mcp__plugin_wenlan_wenlan__delete_page",/' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate added unintended mutation permission" \
    perl -0pi -e 's/"Bash",/"Bash", "mcp__wenlan__delete_page",/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "claude curate lint repair review route drift" \
    perl -0pi -e 's/A generic accept does not apply\s+`lint_repair_review`; route that action through `\/lint repair` instead\./A generic accept applies `lint_repair_review` directly./' \
    "$TMPDIR_TEST/root/plugin/skills/curate/SKILL.md"

assert_rejects "codex curate lint repair review route drift" \
    perl -0pi -e 's/Do not generically accept `lint_repair_review`;\s+route it through `\/lint repair`\./Accept `lint_repair_review` directly./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/curate/SKILL.md"

assert_rejects "setup autocomplete drift" \
    perl -0pi -e 's/user-invocable: true/user-invocable: false/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/setup/SKILL.md"

assert_rejects "codex background-consent command drift" \
    perl -0pi -e 's/wenlan enrichment disable/wenlan background off/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/setup/SKILL.md"

assert_rejects "codex README setup command drift" \
    perl -0pi -e 's|/setup|/init|g' \
    "$TMPDIR_TEST/root/plugin-codex/README.md"

assert_rejects "codex resolver parity drift" \
    perl -0pi -e 's/locked-env/codex-locked-env/' \
        "$TMPDIR_TEST/root/plugin-codex/bin/resolve-space.sh"

assert_rejects "codex MCP plugin-relative cwd drift" \
    perl -0pi -e 's/"cwd": "\."/"cwd": ".."/' \
    "$TMPDIR_TEST/root/plugin-codex/.mcp.json"

assert_rejects "marketplace source drift" \
    perl -0pi -e 's|"path": "./plugin-codex"|"path": "./plugin"|' \
    "$TMPDIR_TEST/root/.agents/plugins/marketplace.json"

assert_rejects "claude marketplace category drift" \
    perl -0pi -e 's/"category": "productivity"/"category": "memory"/' \
    "$TMPDIR_TEST/root/.claude-plugin/marketplace.json"

assert_rejects "claude manifest category drift" \
    perl -0pi -e 's/"category": "productivity"/"category": "memory"/' \
    "$TMPDIR_TEST/root/plugin/.claude-plugin/plugin.json"

assert_rejects "claude marketplace description drift" \
    perl -0pi -e 's/"description": "A living knowledge base/"description": "A memory layer/' \
    "$TMPDIR_TEST/root/.claude-plugin/marketplace.json"

assert_rejects "claude marketplace keywords drift" \
    perl -0pi -e 's/"wiki",/"memory-layer",/' \
    "$TMPDIR_TEST/root/.claude-plugin/marketplace.json"

assert_rejects "claude stdio default drift" \
    perl -0pi -e 's/"claude-code"/"codex"/' \
    "$TMPDIR_TEST/root/crates/wenlan-mcp/src/main.rs"

assert_rejects "claude lint general-call drift" \
    perl -0pi -e 's/General uses exactly one lint MCP call/General uses one lint MCP call/g' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint agent-submit drift" \
    perl -0pi -e 's/submit verdicts exactly once/submit verdicts when useful/g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude capture relation order drift" \
    perl -0pi -e 's/for both named endpoints first/for both named endpoints after capture/' \
    "$TMPDIR_TEST/root/plugin/skills/capture/SKILL.md"

assert_rejects "codex capture relation order drift" \
    perl -0pi -e 's/for both named endpoints first/for both named endpoints after capture/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/capture/SKILL.md"

assert_rejects "claude capture explicit-relation policy drift" \
    perl -0pi -e 's/only when the user explicitly states a durable relation/whenever a relation seems useful/' \
    "$TMPDIR_TEST/root/plugin/skills/capture/SKILL.md"

assert_rejects "codex capture explicit-relation policy drift" \
    perl -0pi -e 's/only when the user explicitly states a durable relation/whenever a relation seems useful/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/capture/SKILL.md"

assert_rejects "claude capture ordinary entity policy drift" \
    perl -0pi -e 's/Do not call\s+`create_entity` for ordinary captures\./Call `create_entity` for ordinary captures./' \
    "$TMPDIR_TEST/root/plugin/skills/capture/SKILL.md"

assert_rejects "codex capture ordinary entity policy drift" \
    perl -0pi -e 's/Do not call\s+`mcp__wenlan__create_entity` for ordinary captures\./Call `mcp__wenlan__create_entity` for ordinary captures./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/capture/SKILL.md"

assert_rejects "claude capture inference policy drift" \
    perl -0pi -e 's/Never infer a\s+relation the\s+user did not state\./Infer likely relations./' \
    "$TMPDIR_TEST/root/plugin/skills/capture/SKILL.md"

assert_rejects "codex capture inference policy drift" \
    perl -0pi -e 's/Never infer a\s+relation the\s+user did not state\./Infer likely relations./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/capture/SKILL.md"

assert_rejects "claude lint counterevidence authorization drift" \
    perl -0pi -e 's/authorized\s+record refs \(`evidence_refs` plus `counterevidence_refs`\)/unbounded record refs/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint counterevidence authorization drift" \
    perl -0pi -e 's/authorized\s+record refs \(`evidence_refs` plus `counterevidence_refs`\)/unbounded record refs/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint bounded semantic completion drift" \
    perl -0pi -e 's/Population truncation is honest coverage metadata/Population truncation is always incomplete/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint bounded semantic completion drift" \
    perl -0pi -e 's/Population truncation is honest coverage metadata/Population truncation is always incomplete/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint outcome-funnel drift" \
    perl -0pi -e 's/Lead repair output with exactly one compact typed-count funnel/Render the repair totals/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint outcome-funnel drift" \
    perl -0pi -e 's/Lead repair output with exactly one compact typed-count funnel/Render the repair totals/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint multi-approval drift" \
    perl -0pi -e 's/one or more exact approval lines/one exact approval line/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint multi-approval drift" \
    perl -0pi -e 's/one or more exact approval lines/one exact approval line/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint prevalidation drift" \
    perl -0pi -e 's/Validate the complete reply before the first apply/Validate after applying the first manifest/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint prevalidation drift" \
    perl -0pi -e 's/Validate the complete reply before the first apply/Validate after applying the first manifest/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint cross-plan approval drift" \
    perl -0pi -e 's/ready tuples from the same displayed plan/ready tuples from any displayed plan/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint cross-plan approval drift" \
    perl -0pi -e 's/ready tuples from the same displayed plan/ready tuples from any displayed plan/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint ready-only approval drift" \
    perl -0pi -e 's/contain only ready tuples/contain ready tuples/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint ready-only approval drift" \
    perl -0pi -e 's/contain only ready tuples/contain ready tuples/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint mixed-content approval drift" \
    perl -0pi -e 's/no duplicates, blank\s+lines, prose, or code fences/duplicates and prose are accepted/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint mixed-content approval drift" \
    perl -0pi -e 's/no duplicates, blank\s+lines, prose, or code fences/duplicates and prose are accepted/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint single-line compatibility drift" \
    perl -0pi -e 's/A single\s+line remains valid/A single line is rejected/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint single-line compatibility drift" \
    perl -0pi -e 's/A single\s+line remains valid/A single line is rejected/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint fresh-per-manifest drift" \
    perl -0pi -e 's/Immediately rerun\s+fresh General once/Reuse the prepared lint reports/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint fresh-per-manifest drift" \
    perl -0pi -e 's/Immediately rerun\s+fresh General once/Reuse the prepared lint reports/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint verify-before-next drift" \
    perl -0pi -e 's/Only after one manifest is `verified` may the next approved\s+manifest begin/Start the next manifest before verification/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint verify-before-next drift" \
    perl -0pi -e 's/Only after one manifest is `verified` may the next approved\s+manifest begin/Start the next manifest before verification/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint result-state drift" \
    perl -0pi -e 's/`verified`,\s+`applied_unverified`,\s+`failed`, or `not_attempted`/`done`/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint result-state drift" \
    perl -0pi -e 's/`verified`,\s+`applied_unverified`,\s+`failed`, or\s+`not_attempted`/`done`/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint ready-order drift" \
    perl -0pi -e 's/in\s+the\s+order\s+they\s+appear among ready tuples in the displayed plan/in any order/g' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint ready-order drift" \
    perl -0pi -e 's/in\s+the\s+order\s+they\s+appear among ready tuples in the displayed plan/in any order/g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint copy-block drift" \
    perl -0pi -e 's/one contiguous\s+copy-pasteable block/individual scattered lines/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint copy-block drift" \
    perl -0pi -e 's/one contiguous\s+copy-pasteable block/individual scattered lines/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint stop-before-next drift" \
    perl -0pi -e 's/Do not\s+apply any later\s+approved manifest/Continue with later approved manifests/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint stop-before-next drift" \
    perl -0pi -e 's/Do not\s+apply any later\s+approved manifest/Continue with later approved manifests/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint daemon handoff drift" \
    perl -0pi -e 's/`next_apply`/`next_hint`/g' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint daemon handoff drift" \
    perl -0pi -e 's/`next_apply`/`next_hint`/g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint fallback drift" \
    perl -0pi -e 's/CLI fallback: `wenlan lint --profile deep --agent-assist`/There is no CLI fallback./g' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "claude pages revision-id resolver drift" \
    perl -0pi -e 's/--resolve-id/--show-id/g' \
    "$TMPDIR_TEST/root/plugin/skills/pages/SKILL.md"

assert_rejects "codex pages revision-id resolver drift" \
    perl -0pi -e 's/--resolve-id/--show-id/g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/pages/SKILL.md"

assert_rejects "codex lint repair mode drift" \
    perl -0pi -e 's|/lint repair|/lint-repair|g' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint exact approval drift" \
    perl -0pi -e 's/Match every\s+line byte-for-byte/Accept the later reply semantically/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint exact approval drift" \
    perl -0pi -e 's/Match every\s+line byte-for-byte/Accept the later reply semantically/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "codex lint read-only boundary drift" \
    perl -0pi -e 's|Plain `/lint`, `/lint deep`, the lint MCP tool, and `/api/lint` are fully\s+read-only|Plain `/lint` may write|' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "codex lint stale-general fallback drift" \
    perl -0pi -e 's/If Deep is incomplete, its producer receipt differs from General, or its DB\s+analysis digest differs from General, rerun fresh General exactly once after\s+Deep before prepare; do not rerun Deep\./Reuse the original General after Deep./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "codex lint cross-profile Page receipt drift" \
    perl -0pi -e 's/Do not compare General and Deep Page\s+digests across profiles because their Page scan coverage intentionally\s+differs\./Compare every General and Deep receipt./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint appended fix argument" \
    perl -0pi -e 's/(argument-hint: "[^"]+)"/$1 --fix"/' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint appended agent argument" \
    perl -0pi -e 's/(argument-hint: "[^"]+)"/$1 agent"/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude lint dropped repair permission" \
    perl -0pi -e 's/, "mcp__plugin_wenlan_wenlan__prepare_lint_repair_plan"//' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "claude lint dropped review-choice prepare permission" \
    perl -0pi -e 's/, "mcp__plugin_wenlan_wenlan__prepare_lint_repair"//' \
    "$TMPDIR_TEST/root/plugin/skills/lint/SKILL.md"

assert_rejects "codex lint review-choice flow drift" \
    perl -0pi -e 's/Lint creates durable Review Items for choices that are not yet exact\./Lint silently resolves every Review Item./' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "codex lint added undeclared permission" \
    perl -0pi -e 's/"Bash"/"Bash", "mcp__wenlan__unknown"/' \
    "$TMPDIR_TEST/root/plugin-codex/skills/lint/SKILL.md"

assert_rejects "claude help omitted lint" \
    perl -0pi -e 's|  /lint \[deep\|repair\] \[scope\].*\n||' \
    "$TMPDIR_TEST/root/plugin/skills/help/SKILL.md"

assert_rejects "codex help omitted lint" \
    perl -0pi -e 's|  /lint \[deep\|repair\] \[scope\].*\n||' \
    "$TMPDIR_TEST/root/plugin-codex/skills/help/SKILL.md"
