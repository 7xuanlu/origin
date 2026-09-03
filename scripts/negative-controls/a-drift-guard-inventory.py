#!/usr/bin/env python3
"""Which drift_guard teeth touch release.yml, and is each one accounted for.

a-drift-guard-replica.py re-implements, in Python, the teeth that read
`.github/workflows/release.yml`, because `cargo test -p wenlan-core` cannot
build on this host (llama-cpp-sys-2's build.rs refuses without a Vulkan SDK).
That replica was written from a HAND reading of drift_guard.rs. A hand reading
is a snapshot: a tooth added upstream tomorrow is not in it, and nothing here
would notice -- the replica would keep passing, over an inventory that had
silently gone short.

This script is the part that CAN be measured mechanically. It does NOT parse
the assertions; a parser that walked `release.contains("...")` would have to
follow the text through function parameters (`release_workflow`, `promote`,
`build`, ...), and a partial one reports zero literals for a file full of them
-- the same failed-measurement-reads-as-a-negative shape this workstream keeps
finding. Written and run: over the current drift_guard.rs it finds four
identifiers bound to release.yml by `let ... = std::fs::read_to_string(...)`
and ZERO `.contains` literals on them, because every assertion is reached
through a parameter of another function.

So the sound measurement is one level up: enumerate every function in
drift_guard.rs that mentions release.yml at all, follow its calls, and require
every function in that closure to carry an explicit disposition below.

ROUND 13b, FINDING 5. Three separate ways a tooth could die without this file
saying a word, all three now controlled at the bottom of main():

  1. The inventory listed only functions that MENTION release.yml -- the
     callers. `release_rust_cache_violations` and
     `release_promotion_contract_violations` hold the assertions and name no
     path, so deleting a check inside either changed nothing here. The scan now
     takes the transitive call closure (22 functions from 11 mention sites,
     enumerated because it is small enough to enumerate honestly).
  2. A span began at `fn`, so `#[test]` sat outside it. Adding `#[ignore]` left
     every digest identical. Spans now start at the attributes.
  3. The digest stripped every line before hashing. These teeth assert over
     YAML inside Rust string literals, where indentation IS the content, so a
     re-indented fixture hashed the same. The digest is now verbatim.

ROUND 13, FINDING 5. The first version of this compared only the SET of
function names, and said in its own docstring that "a new tooth then fails here
by name". That was false, and it was false in the likeliest direction: teeth
grow INSIDE existing functions, which leaves the name set identical. A
disposition is a statement about a body, so each entry now pins the body's
digest and any edit inside an inventoried function is reported as CHANGED until
someone re-reads it and re-dispositions it. The same round replaced the
backward line scan -- which had no idea about brace scope -- with a real
brace-matched span over a source masked for comments and string literals, so a
`{` inside a string cannot shift every attribution after it.

ROUND 3 (Codex Sol), FINDING N4. Two more ways the scan could go short without
saying so, and one way it could report two things as one:

  * THE SEED was the literal `workflows/release.yml`. drift_guard.rs mentions
    `release.yml` twenty-eight times and only eighteen carry the directory, so
    four functions -- and they are the four TEETH -- were reached only as
    callees. Measured: widening the seed to the bare file name takes the
    mention sites from 11 to 15 and leaves the closure at 30, because those
    four were already reached through their callers. That is the honest result:
    the widening found no missing tooth TODAY; it removes the way one goes
    missing tomorrow, when a caller is renamed and a tooth that names no path
    quietly drops out.
  * FILE-LOCAL was an assumption, not a measurement. drift_guard.rs declares
    two `#[path] mod`s whose bodies are other files. Those are now loaded as
    UNITS, cross-unit edges are followed where the source spells one, and the
    report names every unit it covered. Also measured: they add nothing to
    today's closure, because neither mentions release.yml.
  * A path that is ASSEMBLED, code spliced by `include!`, and a non-inert
    attribute macro on a function IN the closure are each a REFUSAL, on the
    argument `PASTES` already made: they remove the text the scan reads, so a
    shorter closure would be reported as a complete one.
  * And FINDING N8: the inventory was keyed by the BARE function name.
    drift_guard.rs holds three `record_cfg_test_node` and three `visit_expr`
    across three impl blocks each, and r4_test_support_test.rs holds two
    `scan_items` in two different functions. Each set collapsed into one row
    with one digest -- the concatenation -- and one disposition describing
    whichever body the reader opened. Names are now qualified by every
    enclosing `mod`, `impl`, `trait` AND function, and a duplicate that
    survives that is REPORTED rather than merged.

What this still does not prove, said plainly: that each REPLICATED entry's
Python twin asserts the same thing as its Rust original. That is a hand
reading, re-done when a digest changes. The inventory makes the re-reading
mandatory; it does not do the reading.

Run: python3 scripts/negative-controls/a-drift-guard-inventory.py
     python3 scripts/negative-controls/a-drift-guard-inventory.py --print-digests
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import io
import os
import re
import sys
import time
from typing import NamedTuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GUARD = os.path.join(ROOT, 'crates', 'wenlan-core', 'src', 'drift_guard.rs')

#: The terminal completion marker. Last line, unconditionally, and only after
#: every control has been scored. Without one, a harness killed by a watchdog
#: and a harness that found nothing produce the same transcript tail.
MARKER = 'NEGATIVE-CONTROL COMPLETE'
HARNESS = 'a-drift-guard-inventory.py'
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


#: ROUND 3 (Codex Sol), FINDING N4. The seed was the LITERAL string
#: `workflows/release.yml`. drift_guard.rs mentions `release.yml` twenty-eight
#: times and only eighteen of those carry the directory -- the other ten are
#: `.expect("read release.yml")` and prose -- so ten mention sites were outside
#: the seed set and any tooth whose only mention was one of them was outside
#: the closure. The seed is now the file name, wherever it appears.
#:
#: What a broader literal still cannot see is a path that is ASSEMBLED, code
#: spliced in by `include!`, an item rewritten by an attribute macro, or a
#: helper that lives in another FILE. Those are not made visible by widening a
#: literal; they are refused, below, the way identifier pasting already was.
MENTION = re.compile(r'release\.yml')
#: Round 13e, reopened finding 5. The qualifier list was `pub`/`pub(...)` and
#: `async`, so `const fn`, `unsafe fn` and `extern "C" fn` were not functions as
#: far as this file was concerned -- a tooth declared `const fn` would have no
#: span, no digest, and no place in the inventory, and the report would be short
#: without saying so. Rust allows these in one fixed order but there is no cost
#: to accepting any order, and every one of them is a function.
FN = re.compile(
    r'(?:^|\n)[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?'
    r'(?:(?:default|const|async|unsafe|extern(?:[ \t]+"[^"]*")?)[ \t]+)*'
    r'fn[ \t]+([A-Za-z_][A-Za-z0-9_]*)')

REPLICATED = 'replicated'
CONTROL = 'control'
OUT_OF_SCOPE = 'out of scope'
TOOTH = 'tooth'
HELPER = 'helper'
#: In the closure only because a parameter or local binding is spelled like a
#: file-local function. The widened edge relation (round 13c, finding 5) is an
#: over-approximation on purpose, and this kind is how that shows in the report
#: rather than being quietly filtered away.
SHADOW = 'name-shadow'


def mask(source: str) -> str:
    """A same-length copy with comments and string/char contents blanked.

    Brace matching over raw Rust counts the `{` in `"{}"` and in a `// {`
    comment. Blanking rather than deleting keeps every offset identical, so a
    span found here indexes straight back into the original text.
    """
    out = list(source)
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == '/' and i + 1 < n and source[i + 1] == '/':
            while i < n and source[i] != '\n':
                out[i] = ' '
                i += 1
            continue
        if c == '/' and i + 1 < n and source[i + 1] == '*':
            depth = 1
            out[i] = out[i + 1] = ' '
            i += 2
            while i < n and depth:
                if source.startswith('/*', i):
                    depth += 1
                    out[i] = out[i + 1] = ' '
                    i += 2
                    continue
                if source.startswith('*/', i):
                    depth -= 1
                    out[i] = out[i + 1] = ' '
                    i += 2
                    continue
                if source[i] != '\n':
                    out[i] = ' '
                i += 1
            continue
        if c == 'r' and i + 1 < n and source[i + 1] in '#"':
            j = i + 1
            hashes = 0
            while j < n and source[j] == '#':
                hashes += 1
                j += 1
            if j < n and source[j] == '"':
                close = '"' + '#' * hashes
                end = source.find(close, j + 1)
                end = n if end == -1 else end + len(close)
                for k in range(i, end):
                    if source[k] != '\n':
                        out[k] = ' '
                i = end
                continue
        if c == '"':
            out[i] = ' '
            i += 1
            while i < n:
                if source[i] == '\\':
                    out[i] = ' '
                    if i + 1 < n and source[i + 1] != '\n':
                        out[i + 1] = ' '
                    i += 2
                    continue
                if source[i] == '"':
                    out[i] = ' '
                    i += 1
                    break
                if source[i] != '\n':
                    out[i] = ' '
                i += 1
            continue
        if c == "'":
            # A lifetime ('a) is not a char literal; only blank when it closes.
            m = re.match(r"'(?:\\.|[^\\'])'", source[i:i + 4])
            if m:
                for k in range(i, i + m.end()):
                    out[k] = ' '
                i += m.end()
                continue
        i += 1
    return ''.join(out)


def with_attributes(source: str, start: int, masked: str | None = None) -> int:
    """Extend a span backwards over the attributes attached to the fn.

    ROUND 13b, FINDING 5. A span that begins at `fn` excludes `#[test]`, so
    turning a tooth into `#[ignore]` -- the cheapest way to disable one -- left
    the body digest identical and the inventory silent. The attribute IS part
    of the disposition: a function that no longer runs is not the function that
    was read.

    ROUND 13c, FINDING 5. That version matched attributes LINE BY LINE, and
    said so in a comment: "a multi-line one would stop the scan early". A
    comment is not a guard. `#[cfg(\\n    any()\\n)]` written above an existing
    `#[test]` compiles the tooth out entirely, and the line-wise scan stops at
    the `)]` line -- so the span excludes the attribute that killed the tooth,
    the digest is unchanged, and the inventory reports the function as read. The
    disabling edit hides in the one construct the scanner declined to parse,
    which is the whole defect class in miniature.

    So it matches BRACKETS, over the masked source, so a `]` inside a string or
    comment cannot end an attribute. The returned offset is still the newline
    before the attribute's first line, exactly as before, so digests recorded
    against the line-wise version stay valid.
    """
    m = masked if masked is not None else mask(source)
    pos = start
    while pos > 0:
        j = pos - 1
        while j >= 0 and m[j] in ' \t\r\n':
            j -= 1
        if j < 0 or m[j] != ']':
            break
        depth = 0
        k = j
        while k >= 0:
            if m[k] == ']':
                depth += 1
            elif m[k] == '[':
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        if k < 0 or depth != 0:
            break
        h = k - 1
        while h >= 0 and m[h] in ' \t':
            h -= 1
        if h >= 0 and m[h] == '!':      # #![inner]
            h -= 1
        if h < 0 or m[h] != '#':
            break
        line_start = m.rfind('\n', 0, h) + 1
        pos = max(line_start - 1, 0)
    return max(pos, 0)


#: A `mod`, `trait` or `impl` block, which is what makes two functions with the
#: same spelling two different functions.
SCOPE = re.compile(
    r'(?:^|\n)[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?'
    r'(?:(?:unsafe|default)[ \t]+)*'
    r'(mod|trait|impl)\b([^{;]*)')


def _close(masked: str, open_brace: int) -> int:
    """The offset one past the `}` that closes `masked[open_brace]`, or -1."""
    depth = 0
    i = open_brace
    while i < len(masked):
        if masked[i] == '{':
            depth += 1
        elif masked[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _impl_label(head: str) -> str:
    """A stable name for one `impl` header.

    `impl<'a> Visit<'a> for CfgTestSpanCollector` becomes
    `CfgTestSpanCollector#Visit`. It does not have to be Rust's own path
    spelling; it has to be different for two different blocks, which is the
    whole point -- drift_guard.rs holds three `record_cfg_test_node` and three
    `visit_expr`, in three impl blocks each, and a name-keyed inventory
    collapsed each trio into one row.
    """
    text = head.split(' where ')[0].split('\nwhere')[0]
    # Drop generic parameter lists, two levels of nesting deep, which is as far
    # as anything in this file goes.
    for _ in range(3):
        text = re.sub(r'<[^<>]*>', '', text)
    text = ' '.join(text.split())
    if ' for ' in text:
        trait, target = text.split(' for ', 1)
        return '%s#%s' % (target.strip(), trait.strip())
    return text.strip()


def scopes(source: str, masked: str | None = None) -> list[tuple[str, int, int]]:
    """(label, start, end) for every `mod`/`trait`/`impl` BLOCK."""
    m = masked if masked is not None else mask(source)
    found: list[tuple[str, int, int]] = []
    for match in SCOPE.finditer(m):
        kind, head = match.group(1), match.group(2)
        open_brace = m.find('{', match.end(1))
        if open_brace == -1:
            continue
        semi = m.find(';', match.end(1))
        if semi != -1 and semi < open_brace:
            continue  # `mod x;` -- a declaration, handled by load_units()
        end = _close(m, open_brace)
        if end == -1:
            continue
        if kind == 'impl':
            label = _impl_label(source[match.end(1):open_brace])
        else:
            label = (head.strip().split() or [''])[0]
        found.append((label or kind, match.start(), end))
    return found


@functools.lru_cache(maxsize=16)
def functions(source: str) -> list[tuple[str, int, int]]:
    """(qualified_name, start_offset, end_offset) for every fn.

    The span starts at the function's attributes, not at `fn`, and the name is
    qualified by every enclosing `mod`/`trait`/`impl` block.

    ROUND 3 (Codex Sol), FINDING N4, second half. This used to return the BARE
    name, and `reachable()` keyed spans by it. drift_guard.rs has three
    `record_cfg_test_node` and three `visit_expr`; each trio collapsed into one
    row whose digest was the concatenation of three unrelated bodies and whose
    single disposition described whichever one the reader happened to open.
    """
    masked = mask(source)
    scope_spans = scopes(source, masked)
    raw: list[tuple[str, int, int, int]] = []   # bare, decl, start, end
    for match in FN.finditer(masked):
        name = match.group(1)
        open_brace = masked.find('{', match.end())
        if open_brace == -1:
            continue
        semi = masked.find(';', match.end())
        if semi != -1 and semi < open_brace:
            continue  # a trait signature, not a body
        end = _close(masked, open_brace)
        if end == -1:
            continue
        raw.append((name, match.start(), with_attributes(source, match.start(),
                                                         masked), end))
    found: list[tuple[str, int, int]] = []
    for name, decl, start, end in raw:
        # An enclosing FUNCTION is a scope too. Two helpers with the same
        # spelling inside two different test functions is the commonest way a
        # name-keyed inventory collapses two bodies into one row, and it is what
        # r4_test_support_test.rs actually does with `scan_items`.
        enclosing = [(s[0], s[1]) for s in scope_spans if s[1] <= decl < s[2]]
        enclosing += [(o[0], o[1]) for o in raw
                      if o[2] <= decl < o[3] and o[1] != decl]
        enclosing.sort(key=lambda pair: pair[1])
        qualified = '::'.join([label for label, _ in enclosing] + [name])
        found.append((qualified, start, end))
    return found


def digest_of(body: str) -> str:
    """The body's exact bytes, line endings normalised and nothing else.

    ROUND 13b, FINDING 5. The previous version stripped every line before
    hashing, "so reindenting is not a false alarm". But these teeth assert over
    YAML held in Rust raw strings, where indentation is the meaning: a fixture
    re-indented from four spaces to two is a different fixture, and the digest
    said nothing had changed. Normalising away the one thing the subject treats
    as significant is not a convenience, it is a blind spot.

    So the digest is verbatim. A reformat now costs a re-read of eleven
    functions, which is the correct price: this file's entire job is to make
    re-reading mandatory when the body moves. \\r\\n is normalised because a
    checkout's line endings are not a property of the code.
    """
    return hashlib.sha256(
        body.replace('\r\n', '\n').encode('utf-8')).hexdigest()[:16]


# ROUND 13c, FINDING 5. This was `\b([a-z_][a-z0-9_]*)\s*\(` -- a call regex,
# and not a sound call graph. `let f = helper;` followed by `f(text)` is an edge
# with no `helper(` anywhere in the source, and a macro that pastes a name is
# another; either one leaves a tooth outside the closure and therefore outside
# the inventory, which is exactly the invisibility round 13b's finding 5 closed
# for a different reason. A precise call graph for Rust is not something a
# regex can be; an OVER-approximation is, and over-approximating is the safe
# direction here. Every extra name it drags in must be dispositioned, so the
# cost of a spurious edge is one line of inventory, while the cost of a missing
# one is a tooth nobody re-reads. So: any mention of a file-local function's
# name, as a whole identifier, in the masked source.
REFERENCE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')

# ROUND 13d, FINDING 5 REOPENED. `REFERENCE` follows whole identifiers found
# inside a function's own span. A call that exists only after macro expansion is
# not in that span: the identifier is in the MACRO's body, and the function
# contains only `the_macro!(...)`. So the closure could stop one step short of a
# tooth, silently, which is the same invisibility this file exists to remove.
#
# The fix is to treat a macro invocation as an edge to everything its definition
# mentions. That is sound for `macro_rules!`, because every identifier the
# expansion can produce is literally present in the macro body -- with one
# exception, below.
MACRO_RULES = re.compile(r'\bmacro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\{\(\[]')


class Unmeasurable(Exception):
    """The closure cannot be over-approximated, so no closure is reported.

    The third state, in the one place this harness had only two: a call graph
    it can bound, and a call graph it cannot. Returning a short closure would be
    a failed measurement wearing a measured one.
    """

# The exception, and the reason this is a refusal rather than a heuristic. These
# constructs ASSEMBLE an identifier that appears nowhere in the source, so no
# amount of scanning can over-approximate a call graph in their presence. If one
# ever appears in drift_guard.rs, this file must say it cannot measure rather
# than report a closure it cannot justify.
PASTES = re.compile(r'\b(?:concat_idents\s*!|paste\s*!|paste\s*::)')

#: Round 13e, reopened finding 5, the other half: `PASTES` matches the crate's
#: own spelling, and an import may rename it. `use paste::paste as p;` puts a
#: pasting macro in the file under a name this pattern has never heard of, and
#: `p! { fn [<tooth _ a>]() {} }` then assembles an identifier with the refusal
#: switched off. The import is the signal -- nobody renames `paste` without
#: using it -- so an aliased import of either crate is refused on sight, which
#: is the conservative direction and costs nothing while neither is imported.
PASTE_ALIAS = re.compile(
    r'\buse\b[^;]*\b(?:paste|concat_idents)\b[^;]*?\bas[ \t]+'
    r'([A-Za-z_][A-Za-z0-9_]*)[^;]*;')

# --- ROUND 3 (Codex Sol), FINDING N4 ---------------------------------------
#
# Three more constructs for which no widening of a literal seed helps, because
# the thing the seed is looking for is not in this file at all. Each is a
# REFUSAL, on the same argument `PASTES` already makes: a closure reported in
# their presence is a claim the scan cannot justify, and a claim that cannot be
# justified is exactly the failed-measurement-wearing-a-measurement shape this
# whole directory exists to remove.

#: `include!` splices Rust ITEMS from another file into this one. Everything
#: this harness knows about spans, digests and edges is about text it has read.
INCLUDE_MACRO = re.compile(r'\binclude\s*!\s*[\(\[\{]')

#: An attribute macro can replace the item it is written on with anything at
#: all, so a body digest taken from the source is a digest of code that will
#: never run. These are the attributes that CANNOT do that: built-ins and inert
#: helper attributes. Anything else ON A FUNCTION IN THE CLOSURE is treated as
#: a proc macro and refused.
#:
#: The scoping is deliberate and was measured. A first version refused on any
#: non-inert attribute anywhere in the crate, and `#[tokio::test]` in
#: drift_guard/r4_test_support_test.rs -- a file that never mentions
#: release.yml -- made the whole harness unmeasurable. An attribute macro can
#: only rewrite the item it is written on, so the ones that can invalidate a
#: digest in this inventory are the ones written on a function IN it. A
#: function-like proc macro is a different construct and is covered by the
#: macro-body walk and by `PASTES`.
#:
#: drift_guard.rs uses exactly three attributes today -- `test`, `cfg`, `path`
#: -- so the list is generous and still leaves the refusal reachable.
INERT_ATTRS = frozenset((
    'test', 'cfg', 'cfg_attr', 'path', 'allow', 'deny', 'warn', 'expect',
    'forbid', 'doc', 'derive', 'ignore', 'should_panic', 'inline', 'must_use',
    'non_exhaustive', 'automatically_derived', 'track_caller', 'repr', 'cold',
    'no_mangle', 'used', 'link', 'macro_export', 'macro_use', 'rustfmt',
    'clippy', 'serde', 'allow_internal_unstable', 'bench', 'global_allocator',
    'panic_handler', 'no_std', 'feature', 'deprecated',
))
#: The whole attribute PATH, not its first segment: `#[tokio::test]` and
#: `#[test]` are different attributes and only one of them is a built-in.
ATTR_NAME = re.compile(
    r'#!?\[\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)')

#: A workflow path the source never spells. `format!("{dir}/release.yml")` is
#: still seeded, because the file name is in the literal; `format!("{dir}/{}",
#: name)` where `name` is decided at runtime is not, and no scan of this file
#: can decide whether it names release.yml. Two shapes are refused: an
#: interpolated format/concat string that mentions a workflow directory or a
#: `.yml` suffix, and a `.join(...)` whose argument is not a string literal in
#: a statement that mentions the workflow directory.
ASSEMBLED_FORMAT = re.compile(
    r'\b(?:format|concat)\s*!\s*\(\s*(?:r#*)?"([^"]*)"')
ASSEMBLED_JOIN = re.compile(r'\.join\s*\(\s*')
#: The start of a Rust string literal, in the ORIGINAL source. The masked copy
#: cannot answer this question -- it blanks literals to spaces, so every
#: `.join("....")` looks like `.join(     )` and reads as a non-literal. The
#: first version of this rule did exactly that and refused the shipped file over
#: `repo_root().join(".github/workflows/ci.yml")`.
STRING_LITERAL_START = re.compile(r'(?:b?r#*)?"')
WORKFLOW_HINT = re.compile(r'\.github|workflows|\.yml|\.yaml')

#: `mod x;` -- a module whose body is in ANOTHER FILE.
MOD_DECL = re.compile(
    r'(?:^|\n)[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+'
    r'([A-Za-z_][A-Za-z0-9_]*)[ \t]*;')
PATH_ATTR = re.compile(r'#\[[ \t]*path[ \t]*=[ \t]*"([^"]*)"[ \t]*\]')
GLOB_USE = re.compile(r'\buse[ \t]+([A-Za-z_][A-Za-z0-9_]*)\s*::\s*\*\s*;')
QUALIFIED_REF = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]*)\s*::\s*([A-Za-z_][A-Za-z0-9_]*)\b')


class Unit(NamedTuple):
    """One SOURCE FILE in the module tree, with the module path that reaches it.

    ROUND 3, FINDING N4. `sites`/`reachable` were file-local by construction:
    everything they knew came from one string. drift_guard.rs declares two
    modules whose bodies live in `crates/wenlan-core/src/drift_guard/*.rs`, and
    a tooth in either was outside the closure with nothing saying so. The unit
    list is what makes "file-local" a measured statement instead of an
    assumption.
    """

    module: str   # '' for the root file
    path: str     # for the report; '<memory>' for a fixture
    source: str

    def qualify(self, name: str) -> str:
        return '%s::%s' % (self.module, name) if self.module else name


def as_units(subject) -> list[Unit]:
    """A str is one anonymous unit; a list of Units is itself."""
    if isinstance(subject, str):
        return [Unit('', '<memory>', subject)]
    return list(subject)


def load_units(main_path: str) -> list[Unit]:
    """The root file and every `mod x;` file it reaches, transitively.

    A declaration whose file cannot be found is `Unmeasurable`: the module
    exists, its contents are part of the crate, and reporting a closure without
    them would be reporting a short one silently.
    """
    units: list[Unit] = []
    pending = [('', main_path)]
    seen = set()
    while pending:
        module, path = pending.pop(0)
        real = os.path.abspath(path)
        if real in seen:
            continue
        seen.add(real)
        try:
            source = io.open(real, encoding='utf-8').read()
        except OSError as exc:
            raise Unmeasurable(
                'module %r declares a body in %s, which cannot be read (%s). '
                'Its functions are part of the crate and would be missing from '
                'the closure without a word.' % (module or '<root>', path, exc))
        units.append(Unit(module, real, source))
        masked = mask(source)
        directory = os.path.dirname(real)
        stem = os.path.splitext(os.path.basename(real))[0]
        for decl in MOD_DECL.finditer(masked):
            name = decl.group(1)
            # The `#[path = "..."]` attribute, if any, sits in the attribute
            # block this declaration owns -- found the same way a function's is.
            attr_start = with_attributes(source, decl.start(1), masked)
            override = PATH_ATTR.search(source[attr_start:decl.end()])
            child = '%s::%s' % (module, name) if module else name
            if override:
                candidates = [os.path.join(directory, override.group(1))]
            else:
                base = directory if stem in ('mod', 'lib', 'main') else \
                    os.path.join(directory, stem)
                candidates = [os.path.join(base, name + '.rs'),
                              os.path.join(base, name, 'mod.rs')]
            found = next((c for c in candidates if os.path.exists(c)), None)
            if found is None:
                raise Unmeasurable(
                    '%s declares `mod %s;` and none of %s exists. The module is '
                    'part of the crate; a closure that silently omits it is '
                    'short by however many teeth live there.'
                    % (os.path.relpath(real, ROOT), name,
                       [os.path.relpath(c, ROOT) for c in candidates]))
            pending.append((child, found))
    return units


def refuse_unmeasurable(units: list[Unit]) -> None:
    """Raise `Unmeasurable` for any construct that puts a tooth out of reach.

    Every rule here is a REFUSAL rather than a heuristic, for the reason
    `PASTES` gives: the construct removes the thing the scan reads, so a
    shorter closure would be reported as a complete one.
    """
    for unit in units:
        where = os.path.relpath(unit.path, ROOT) if unit.path != '<memory>' \
            else '<memory>'
        masked = mask(unit.source)

        def line_of(pos: int) -> int:
            return masked[:pos].count('\n') + 1

        paste = PASTES.search(masked)
        if paste:
            raise Unmeasurable(
                '%s line %d uses %r, which assembles an identifier that appears '
                'nowhere in the source. No scan of this file can over-'
                'approximate its call graph, so the closure below would be a '
                'claim this harness cannot justify. Resolve the expansion by '
                'hand and inventory it explicitly.'
                % (where, line_of(paste.start()), paste.group(0)))
        alias = PASTE_ALIAS.search(masked)
        if alias:
            raise Unmeasurable(
                '%s line %d imports a pasting macro under the name %r (%r). '
                'Everything the refusal above looks for is spelled differently '
                'from here on, so the closure below would be reported with the '
                'refusal switched off. Resolve the expansion by hand and '
                'inventory it explicitly.'
                % (where, line_of(alias.start()), alias.group(1),
                   alias.group(0).strip()))
        included = INCLUDE_MACRO.search(masked)
        if included:
            raise Unmeasurable(
                '%s line %d uses `include!`, which splices ITEMS from another '
                'file into this one. Those items are functions of this module '
                'that no span, digest or edge here has ever seen. Inline them '
                'or inventory the included file explicitly.'
                % (where, line_of(included.start())))
        for fmt in ASSEMBLED_FORMAT.finditer(unit.source):
            literal = fmt.group(1)
            if '{' not in literal or not WORKFLOW_HINT.search(literal):
                continue
            if MENTION.search(literal):
                continue  # the file name is still spelled out; still seeded
            raise Unmeasurable(
                '%s line %d builds a workflow path from %r, whose file name is '
                'decided at runtime. No literal seed can tell whether it names '
                'release.yml.'
                % (where, unit.source[:fmt.start()].count('\n') + 1, literal))
        for join in ASSEMBLED_JOIN.finditer(unit.source):
            if STRING_LITERAL_START.match(unit.source, join.end()):
                continue  # `.join("literal")` -- the name is in the source
            head = max(masked.rfind(';', 0, join.start()),
                       masked.rfind('{', 0, join.start())) + 1
            statement = unit.source[head:join.end()]
            if not WORKFLOW_HINT.search(statement):
                continue
            if MENTION.search(statement):
                continue
            raise Unmeasurable(
                '%s line %d joins a non-literal component onto a workflow path '
                '(%r). The resulting file name is not in this source.'
                % (where, unit.source[:join.start()].count('\n') + 1,
                   ' '.join(statement.split())[-90:]))


def macros(source: str) -> dict[str, list[tuple[int, int]]]:
    """(name -> body spans) for every `macro_rules!` definition in the file."""
    masked = mask(source)
    found: dict[str, list[tuple[int, int]]] = {}
    openers = {'{': '}', '(': ')', '[': ']'}
    for match in MACRO_RULES.finditer(masked):
        start = match.end() - 1
        opener = masked[start]
        closer = openers[opener]
        depth = 0
        i = start
        while i < len(masked):
            if masked[i] == opener:
                depth += 1
            elif masked[i] == closer:
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            continue
        found.setdefault(match.group(1), []).append((start, i + 1))
    return found


def reachable(source, seeds: list[str]) -> dict[str, str]:
    """Every fn reachable from `seeds`, mapped to its body digest.

    ROUND 13b, FINDING 5. The inventory used to enumerate functions that
    MENTION release.yml, and the assertions do not live there. `#[test] fn
    release_reuses_receipt_bound_archives_without_compiling` reads the file and
    hands the text to `release_rust_cache_violations`, which holds every
    forbidden-substring check -- and which names no path, so it was invisible
    here. The same for `release_promotion_contract_violations`. Deleting a
    check inside either changed nothing this file could see: an inventory of
    callers, pinning digests of functions that contain no teeth.

    References are read off the MASKED source, so an identifier inside a string
    or a comment is not an edge. The closure is small, which is why enumerating
    it is the honest option rather than a heuristic.

    ROUND 13c, FINDING 5. The edge relation is now every MENTION of a file-local
    function's name, not every `name(` call site, because a function pointer or
    a macro-pasted name is an edge a call regex cannot see -- see `REFERENCE`.
    It over-approximates: the closure may contain a function reached only by a
    same-named local. That is the safe direction, and it is why the inventory
    records a disposition per name rather than a count.

    ROUND 13d, FINDING 5 STILL OPEN. Round 13c's answer over-approximated calls
    but not EXPANSIONS: an edge that exists only inside a `macro_rules!` body is
    not in the calling function's span at all, so the closure could stop one
    step short of a tooth. Macro bodies are now followed as edges of every
    function that invokes them, which is sound for `macro_rules!` because every
    identifier the expansion can produce is literally in the body. Identifier
    PASTING is the one construct for which that is false, and it is refused
    rather than approximated -- see `PASTES`.

    ROUND 3 (Codex Sol), FINDING N4. The closure is now taken over a list of
    UNITS -- the root file and every `mod x;` file it declares -- and keyed by
    the scope-qualified name, so two functions that share a spelling are two
    rows. Cross-unit edges are followed only where the source spells one:
    a `module::name` path, or a glob import that puts one unit's names in
    another's scope. That is deliberately narrower than the within-unit
    relation, and the report says how many units it covered, so "file-local"
    stops being an unstated assumption about a crate with two other files in it.

    All four refusals -- identifier pasting, an aliased pasting import,
    `include!`, a non-inert attribute macro and an assembled workflow path --
    are taken first, by `refuse_unmeasurable`, for the same reason: each of them
    removes the text this walk reads, so a shorter closure would be reported as
    a complete one.
    """
    units = as_units(source)
    refuse_unmeasurable(units)

    masked = [mask(u.source) for u in units]
    #: qualified name -> [(unit index, start, end)]
    spans: dict[str, list[tuple[int, int, int]]] = {}
    #: unit index -> {bare name -> {qualified names in that unit}}
    local: list[dict[str, set[str]]] = [{} for _ in units]
    for index, unit in enumerate(units):
        for name, start, end in functions(unit.source):
            qualified = unit.qualify(name)
            spans.setdefault(qualified, []).append((index, start, end))
            local[index].setdefault(name.rsplit('::', 1)[-1], set()).add(qualified)

    # ROUND 3, FINDING N8. Two definitions with the SAME qualified name are two
    # bodies behind one digest and one disposition, and the later one is the one
    # Rust uses. There is nothing to choose between them here, so both are
    # reported rather than merged.
    global DUPLICATES
    DUPLICATES = []

    by_module = {u.module: i for i, u in enumerate(units) if u.module}
    #: unit index -> unit indices whose bare names it can see
    visible: list[set[int]] = [{i} for i in range(len(units))]
    for index, unit in enumerate(units):
        for glob in GLOB_USE.finditer(masked[index]):
            target = glob.group(1)
            if target == 'super' and unit.module:
                parent = unit.module.rsplit('::', 1)[0] if '::' in unit.module else ''
                if parent == '':
                    visible[index].add(0)
                elif parent in by_module:
                    visible[index].add(by_module[parent])
            elif target in by_module:
                visible[index].add(by_module[target])

    macro_spans = [macros(u.source) for u in units]
    seen: dict[str, str] = {}
    seen_macros: set[tuple[int, str]] = set()
    frontier = [s for s in seeds if s in spans]
    macro_frontier: list[tuple[int, str]] = []

    def walk(index: int, region: str, owner: str) -> None:
        for ref in REFERENCE.finditer(region):
            callee = ref.group(1)
            hit = False
            for other in visible[index]:
                for qualified in local[other].get(callee, ()):
                    hit = True
                    if qualified not in seen and qualified != owner:
                        frontier.append(qualified)
            if not hit and callee in macro_spans[index] \
                    and (index, callee) not in seen_macros:
                macro_frontier.append((index, callee))
        # `module::name` -- the one cross-unit edge the source spells out.
        for ref in QUALIFIED_REF.finditer(region):
            module, callee = ref.group(1), ref.group(2)
            other = by_module.get(module)
            if other is None:
                continue
            for qualified in local[other].get(callee, ()):
                if qualified not in seen and qualified != owner:
                    frontier.append(qualified)

    while frontier or macro_frontier:
        while macro_frontier:
            index, macro = macro_frontier.pop()
            if (index, macro) in seen_macros:
                continue
            seen_macros.add((index, macro))
            for start, end in macro_spans[index][macro]:
                walk(index, masked[index][start:end], macro)
        if not frontier:
            continue
        name = frontier.pop()
        if name in seen:
            continue
        seen[name] = digest_of(
            ''.join(units[i].source[a:b] for i, a, b in spans[name]))
        for index, start, end in spans[name]:
            walk(index, masked[index][start:end], name)

    # ROUND 3, FINDING N8. A qualified name in THIS INVENTORY that names two
    # bodies is a row whose digest is their concatenation and whose disposition
    # describes whichever one the reader opened. Scoped to the closure: two
    # same-named helpers in code nothing here reaches are not this file's
    # business, and reporting them would drown the ones that are.
    DUPLICATES.extend(sorted(n for n in seen if len(spans[n]) > 1))

    # ROUND 3, FINDING N4, the attribute half. Taken here rather than over the
    # whole crate: an attribute macro can only rewrite the item it is written
    # on, so the ones that make a digest meaningless are the ones on a function
    # that ended up IN this closure.
    for name in sorted(seen):
        for index, start, end in spans[name]:
            text = units[index].source[start:end]
            head = text[:text.index('fn ')] if 'fn ' in text else text
            for attr in ATTR_NAME.finditer(head):
                path = re.sub(r'\s+', '', attr.group(1))
                if path in INERT_ATTRS:
                    continue
                raise Unmeasurable(
                    '%s carries the attribute `#[%s]`, which is not a known '
                    'inert attribute. An attribute macro may replace the item '
                    'it is written on, so the body digested here need not be '
                    'the body that runs, and a digest that pins the wrong text '
                    'is worse than no digest. Add it to INERT_ATTRS once '
                    'someone has checked that it rewrites nothing.'
                    % (name, path))
    return seen


#: Filled by the last `reachable()` call; read by `discrepancies()`.
DUPLICATES: list[str] = []


def sites(source) -> dict[str, dict[str, object]]:
    """Every function that mentions release.yml, with its lines and digest.

    Attribution is by SPAN, so a mention inside a nested closure belongs to the
    function that encloses it, and the innermost enclosing fn wins. The name is
    scope-qualified and unit-qualified, so `tests::helper` in the root file and
    `r4_test_support_test::helper` in another are two rows.
    """
    units = as_units(source)
    found: dict[str, dict[str, object]] = {}
    for unit in units:
        spans = functions(unit.source)
        line_of = [0] * (len(unit.source) + 1)
        line = 1
        for i, ch in enumerate(unit.source):
            line_of[i] = line
            if ch == '\n':
                line += 1
        line_of[len(unit.source)] = line

        for match in MENTION.finditer(unit.source):
            pos = match.start()
            enclosing = [s for s in spans if s[1] <= pos < s[2]]
            if enclosing:
                # innermost = latest start
                name, start, end = max(enclosing, key=lambda s: s[1])
                name, body = unit.qualify(name), unit.source[start:end]
            else:
                name = unit.qualify('<file scope>') if unit.module \
                    else '<file scope>'
                body = ''
            entry = found.setdefault(
                name, {'lines': [], 'digest': digest_of(body)})
            entry['lines'].append(line_of[pos])  # type: ignore[union-attr]
    return found


# Every function this workstream's teeth depend on -- the ones that mention
# release.yml AND everything they call, transitively -- with what this
# workstream does about it. Read out of drift_guard.rs by hand; the digest pins
# the body that reading was done against, so an edit inside one of these has to
# be re-read rather than inherited. Refresh with --print-digests AFTER
# re-reading, never before.
#
# ROUND 13b, FINDING 5: the entries below the mention sites -- TOOTH and HELPER
# -- are new. Before them this file inventoried CALLERS: functions that name
# release.yml. The assertions do not live there. Deleting a check inside
# release_rust_cache_violations or release_promotion_contract_violations
# changed no digest here, because neither function names a path.
INVENTORY: dict[str, tuple[str, str, str]] = {
    'release_reuses_receipt_bound_archives_without_compiling': (
        REPLICATED,
        'f5682cf8a7defa35',
        'reads release.yml and hands it to release_rust_cache_violations, which '
        'holds the actual scan (inventoried below). This function is the file '
        'read and the assert!; the teeth are one call away.',
    ),
    'release_rust_cache_violations': (
        TOOTH,
        '6d1b2ae29c78d852',
        'THE TEETH. Forbidden substrings over the whole file text ("cargo '
        'build", "build-release-binaries", "Swatinem/rust-cache", '
        '"sccache-action"), the jobs.release check, and promote-assets '
        'consuming release-promotion.py download-assets. Replicated in '
        'a-drift-guard-replica.py. Track A hazard: the scan is over raw TEXT, '
        'so a SignPath step whose COMMENT contains "cargo build" fails the '
        'release. Named by round 13b as invisible to the old inventory.',
    ),
    'windows_ort_distribution_stages_packages_and_exercises_exact_dll': (
        REPLICATED,
        '0a7a0908d3b4ff27',
        'reads ci.yml, release.yml and the smoke script, and asserts '
        'windows_ort_distribution_violations is empty over the shipped files.',
    ),
    'windows_ort_distribution_violations': (
        TOOTH,
        '3876429f7014148d',
        'THE TEETH for the ORT/DLL contract, over ci.yml + release.yml + the '
        'smoke script. Replicated. Track A edits the same app-bundle-windows '
        'job this reads, next to "Verify the runtime DLLs ship inside the '
        'installer".',
    ),
    'windows_ort_distribution_contract_rejects_unscoped_archive_payloads': (
        CONTROL,
        '9d268894b5ca2241',
        'feeds a MUTATED release.yml (download-assets -> build-release-binaries) '
        'and asserts the tooth fires. An edit to the shipped file cannot make '
        'this pass wrongly; the tooth it controls is replicated above.',
    ),
    'release_version_sync_never_runs_package_lifecycle_scripts': (
        REPLICATED,
        '56e8cd788bf40721',
        'holds its own assertions inline -- no helper -- and is asserted clean '
        'over the shipped file.',
    ),
    'release_preflight_is_release_gated_and_non_publishing': (
        REPLICATED,
        'cd58292dca03e8fa',
        "release_preflight_contract_violations' DAG half, asserted clean over "
        'the shipped file.',
    ),
    'release_preflight_contract_violations': (
        TOOTH,
        'e434b4a7661315e4',
        'THE TEETH for release-preflight. Its release.yml dependence is the '
        'PARSE, not an assertion: it serde_yaml-parses release.yml with '
        '.expect() and then asserts almost entirely over ci.yml. So a '
        'release.yml edit reaches it only by making the YAML invalid, which is '
        'what the replica reproduces with yaml.safe_load.',
    ),
    'release_preflight_contract_rejects_drift_and_side_effects': (
        CONTROL,
        'c38ed072830cf536',
        'mutates ci.yml, reads release.yml unmutated, and asserts SOME violation '
        'exists. A release.yml edit can only break it by making the tooth panic '
        '- a YAML parse failure - which the replica reproduces by parsing the '
        'same file with yaml.safe_load.',
    ),
    'release_preflight_contract_rejects_overlapping_cache_roots': (
        CONTROL,
        'ba2c87cbdf436751',
        'same shape: the mutation is in ci.yml, the assertion is any-violation.',
    ),
    'release_promotion_reuses_exact_archives_and_fails_closed': (
        REPLICATED,
        'bd5528ef48c2d13f',
        'reads seven files and asserts release_promotion_contract_violations is '
        'empty. Release-side parts replicated.',
    ),
    'release_promotion_contract_violations': (
        TOOTH,
        '8cbb6c54f31a4a90',
        'THE TEETH for promotion, seven arguments wide; release.yml is the '
        'fifth. The other function round 13b named: it carries the assertions '
        'and no path literal, so the mention-based inventory could not see it '
        'and a deleted check here reported nothing.',
    ),
    'release_promotion_contract_rejects_rebuild_and_unbounded_receipts': (
        CONTROL,
        '5f4cbc5951a81095',
        'mutates release.yml (download-assets -> cargo build --release) and '
        'asserts the tooth fires.',
    ),
    'ci_routing_contract_violations': (
        OUT_OF_SCOPE,
        '44f14044d890217a',
        "release.yml appears only as a STRING in ci.yml's required path-filter "
        'lists; the tooth never reads release.yml\'s content. Track A adds '
        'signpath-status.yml, which no list here names.',
    ),
    'ci_manifest_and_lockfile_changes_get_focused_platform_compile_proof': (
        OUT_OF_SCOPE,
        'ff6d638b678238f5',
        'same: a required path-filter entry in ci.yml, not a read of release.yml.',
    ),
    # --- helpers: no assertions of their own, but every tooth above sees the
    # --- workflow through them, so a change here changes what all of them mean.
    'repo_root': (
        HELPER,
        '8f873066781b4481',
        'CARGO_MANIFEST_DIR/../.. canonicalized, with .expect. Every path any '
        'tooth opens starts here.',
    ),
    'job_step': (
        HELPER,
        '7b6d9ac47452ba67',
        'finds a step by EXACT name inside a named job. Every caller then does '
        '.and_then(run).unwrap_or_default(), so a step this cannot find yields '
        'an empty run: and the tooth reports "omits X" rather than "no such '
        'step". Fails closed, but names the wrong cause -- worth knowing when '
        'Track A renames a step.',
    ),
    'job_step_using': (
        HELPER,
        '96a5a2bf793c8a9f',
        'same, matched on a substring of the step\'s uses:, not its name.',
    ),
    'workflow_step_run': (
        HELPER,
        '3c22015ce92d8326',
        "a step's run: found by name across ALL jobs, not scoped to one. Two "
        'jobs with a same-named step resolve to whichever comes first in '
        'mapping order; release.yml\'s SignPath step names are unique today.',
    ),
    'job_needs': (
        HELPER,
        'f3e86cfcc43df6ce',
        'jobs.<name>.needs, tolerating both the scalar and the sequence '
        'spelling. Feeds the required-job closure the preflight tooth walks.',
    ),
    'detect_change_filter_paths': (
        HELPER,
        '74c12c6ec9c91d36',
        "ci.yml's detect-changes filter patterns for one named filter. Returns "
        'an EMPTY SET on every failure -- no detect-changes job, no filter '
        'step, unparseable filters -- which would make the routing assertions '
        'vacuous rather than failed. That is ci.yml\'s fail-open, in a tooth '
        'this workstream does not touch, and it is recorded here rather than '
        'fixed.',
    ),
    'filter_routes_path': (
        HELPER,
        'd88977e6220e837e',
        'pure glob matching over a pattern set (exact, *.ext at root, **/*.ext, '
        'prefix/**). No I/O, no release.yml.',
    ),

    # --- ROUND 13c, FINDING 5: what the widened edge relation dragged in -----
    #
    # These eight arrived when the closure stopped following `name(` and started
    # following every mention of a file-local function's name. Their whole
    # subtree hangs off TWO edges that are not calls at all: the release.yml
    # seed `ci_routing_contract_violations` takes parameters named
    # `platform_sensitive_paths` and `release_profile_sensitive_paths`, spelled
    # exactly like the two functions that compute what its callers pass in.
    #
    # They are kept, dispositioned and digest-pinned rather than filtered back
    # out. Filtering would mean deciding which identifiers are "really" the
    # function -- the same precision the call regex claimed and did not have,
    # and the thing that would have to be right about function pointers and
    # macros too. Over-approximating costs eight bodies to re-read when they
    # move; under-approximating costs a tooth nobody re-reads at all. And the
    # edge is not nonsense: what these compute IS the input the seed's teeth
    # judge, one call frame further out than the closure can see.
    'platform_sensitive_paths': (
        SHADOW,
        '7cd8179456c1dc82',
        'the platform-routing input: every native source git ls-files can find, '
        'mapped to the CI filter that must cover it. In the closure only because '
        'ci_routing_contract_violations names its parameter the same thing; the '
        'real call sites are in ci_routing tests that read ci.yml, not '
        'release.yml. Asserts drift_guard is still #[cfg(test)] before excluding '
        'itself, which is the fail-closed half.',
    ),
    'release_profile_sensitive_paths': (
        SHADOW,
        'b0f7239e712018b8',
        'the same shape for release-profile markers: crates/**/src/*.rs whose '
        'contents carry a production release marker. Same name-shadow edge, same '
        'drift_guard-is-test-only assertion.',
    ),
    'git_ls_files': (
        HELPER,
        '898f9d60de93c4d5',
        'git ls-files wrapper reached from both shadow roots. It asserts '
        'out.status.success(), so a git that failed is a panic rather than an '
        'empty file list -- the one thing that would make every scan above it '
        'vacuously green. Worth the inventory line on its own.',
    ),
    'native_platform_markers': (
        HELPER,
        '68df3ca88f347b43',
        'the regex table (_WIN32/_WIN64, __APPLE__/__MACH__, __linux__ and so '
        'on) that decides which platform a native source belongs to. Reached '
        'from platform_sensitive_paths.',
    ),
    'source_platform_routes': (
        HELPER,
        'b23d16e0220ee22a',
        'per-file platform routing: extension rules for C/ObjC sources plus '
        'cfg-expression scanning for Rust. Reached from platform_sensitive_paths.',
    ),
    'cfg_expression_has_word': (
        HELPER,
        'f9b1def27fe9880f',
        'whole-word test inside a cfg(...) expression, splitting on anything '
        'that is not alphanumeric or underscore. Five lines, no I/O.',
    ),
    'rust_cfg_expression_ranges': (
        HELPER,
        '976c969e988f4e89',
        'hand-written scanner for cfg(...) spans in Rust source, skipping '
        'identifiers that merely end in "cfg". The longest of the eight and the '
        'one most likely to be edited; that is the argument for pinning it.',
    ),
    'has_production_release_marker': (
        HELPER,
        '061b1dd3fd5bddc6',
        'whether a file carries a NON-test debug_assertions cfg. Reached from '
        'release_profile_sensitive_paths, and controlled in the guard itself by '
        'release_profile_marker_scan_is_fail_closed_after_test_modules.',
    ),
}


def audit(source) -> dict[str, dict[str, object]]:
    """Every function this workstream depends on: the mention sites AND the
    functions they call, transitively. Each with the digest of its own body."""
    mentions = sites(source)
    entries: dict[str, dict[str, object]] = {}
    for name, digest in reachable(source, sorted(mentions)).items():
        entries[name] = {
            'digest': digest,
            'lines': mentions.get(name, {}).get('lines', []),
            'mentions': name in mentions,
        }
    for name in mentions:
        if name not in entries:  # '<file scope>', or a mention with no fn body
            entries[name] = dict(mentions[name], mentions=True)
    return entries


def discrepancies(source, inventory: dict[str, tuple[str, str, str]]) -> list[str]:
    problems = []
    found = audit(source)
    if not sites(source):
        # A scan that finds nothing in a 12k-line file that certainly mentions
        # release.yml has not measured a clean inventory; it has failed.
        return ['no release.yml site found at all; the scan is broken, not the file clean']
    # ROUND 3, FINDING N8. A qualified name that names two bodies is a row whose
    # digest and disposition cover neither of them properly.
    for name in DUPLICATES:
        problems.append(
            'DUPLICATE    %s is defined more than once at the same scope; its '
            'digest is the concatenation and its disposition describes whichever '
            'body was read' % name)
    for name in sorted(found):
        lines = found[name]['lines']
        if name not in inventory:
            where = ('mentions release.yml at line%s %s'
                     % ('' if len(lines) == 1 else 's',  # type: ignore[arg-type]
                        ', '.join(str(n) for n in lines))  # type: ignore[union-attr]
                     ) if lines else 'is reached from a release.yml site'
            problems.append(
                'UNACCOUNTED  %s %s and has no disposition' % (name, where))
            continue
        want = inventory[name][1]
        got = found[name]['digest']
        if want != got:
            # The evasion the name-set version could not see: a tooth added
            # inside a function already on the list.
            problems.append(
                'CHANGED      %s body is %s, the disposition was written against %s; '
                're-read it and refresh with --print-digests' % (name, got, want))
    for name in sorted(inventory):
        if name not in found:
            problems.append(
                'STALE        %s has a disposition but is no longer reachable from '
                'a release.yml site' % name)
    return problems


def main() -> int:
    started = time.time()
    try:
        units = load_units(GUARD)
    except Unmeasurable as refusal:
        print('UNMEASURABLE: %s' % refusal)
        globals()['_COMPLETED'] = True
        print('%s %s failures=1 elapsed=%.1fs' % (MARKER, HARNESS,
                                                  time.time() - started))
        return 1
    source = units[0].source

    def with_root(text: str) -> list[Unit]:
        """The real module tree with the root file's text substituted.

        The controls below mutate the root file; keeping the other units means
        a control is scored against the same crate the report was."""
        return [units[0]._replace(source=text)] + units[1:]

    try:
        found = audit(units)
    except Unmeasurable as refusal:
        print('UNMEASURABLE: %s' % refusal)
        globals()['_COMPLETED'] = True
        print('%s %s failures=1 elapsed=%.1fs' % (MARKER, HARNESS,
                                                  time.time() - started))
        return 1

    if '--print-digests' in sys.argv:
        for name in sorted(found):
            print("    '%s': %s," % (name, found[name]['digest']))
        # ROUND 5 (Codex Sol). This mode DID reach its own summary -- the
        # digests are the whole of it -- and returning without saying so left
        # the atexit handler printing `NEGATIVE-CONTROL ABORTED` and "Nothing
        # above it is a result" over a complete listing, beside exit 0. A
        # transcript that contradicts its own status is the defect this file
        # audits, so it must not be the shape this file exits in.
        #
        # It does NOT print the completion MARKER: the aggregate runner reads
        # the last line and would score `failures=0` as a clean control run,
        # and this mode runs no controls at all. A plain terminal line says
        # what happened without minting a result.
        globals()['_COMPLETED'] = True
        print('--print-digests: %d digest(s) listed in %.1fs. This mode runs no '
              'controls; it is a listing, not a result about the guard.'
              % (len(found), time.time() - started))
        return 0

    mention_count = sum(1 for e in found.values() if e['mentions'])
    print('module tree: %d unit(s) -- %s'
          % (len(units),
             ', '.join('%s (%d lines)'
                       % (os.path.relpath(u.path, ROOT), u.source.count('\n') + 1)
                       for u in units)))
    print('drift_guard.rs: %d line(s), %d function(s) mention release.yml, '
          '%d reachable from them'
          % (len(source.split('\n')), mention_count, len(found)))
    print()
    for name in sorted(found):
        kind, _, why = INVENTORY.get(name, ('UNKNOWN', '', ''))
        print('  %-13s %s  [%s]' % (kind, name, found[name]['digest']))
        if found[name]['lines']:
            print('                lines %s'
                  % ', '.join(str(n) for n in found[name]['lines']))  # type: ignore[union-attr]
        else:
            print('                reached from a release.yml site; names no path itself')
        if why:
            for chunk in re.findall(r'.{1,72}(?:\s|$)', why):
                print('                %s' % chunk.strip())
        print()

    problems = discrepancies(units, INVENTORY)
    for problem in problems:
        print('  %s' % problem)

    # The check has to be able to fail, in every direction it claims to cover.
    controls = []

    injected = source + (
        '\n#[test]\nfn a_new_tooth_nobody_replicated() {\n'
        '    let _ = std::fs::read_to_string(root.join(".github/workflows/release.yml"));\n}\n')
    if not any('a_new_tooth_nobody_replicated' in p
               for p in discrepancies(with_root(injected), INVENTORY)):
        controls.append('a new release.yml-reading fn was NOT reported')

    # THE control this file exists for after round 13: a tooth grown inside a
    # function already on the inventory. The name set is unchanged.
    victim = 'release_promotion_reuses_exact_archives_and_fails_closed'
    spans = {n: (a, b) for n, a, b in functions(source)}
    if victim not in spans:
        controls.append('the in-place-growth control could not find %s' % victim)
    else:
        start, end = spans[victim]
        grown = (source[:end - 1]
                 + '\n    assert!(release.contains("a tooth added in place"));\n'
                 + source[end - 1:])
        reported = discrepancies(with_root(grown), INVENTORY)
        if not any(p.startswith('CHANGED') and victim in p for p in reported):
            controls.append(
                'an assertion added INSIDE %s was NOT reported (the name set is '
                'unchanged, which is how the first version of this file missed it)'
                % victim)

    shortened = dict(INVENTORY)
    shortened.pop('release_reuses_receipt_bound_archives_without_compiling')
    if not any('release_reuses_receipt_bound_archives_without_compiling' in p
               for p in discrepancies(units, shortened)):
        controls.append('a missing disposition was NOT reported')

    renamed = dict(INVENTORY)
    renamed['a_tooth_that_no_longer_exists'] = (CONTROL, '0' * 16, 'fixture')
    if not any('a_tooth_that_no_longer_exists' in p
               for p in discrepancies(units, renamed)):
        controls.append('a stale disposition was NOT reported')

    if not any('scan is broken' in p for p in discrepancies('fn nothing() {}', INVENTORY)):
        controls.append('an empty scan was NOT reported as broken')

    # --- round 13b, finding 5: three ways a tooth dies quietly ---------------

    # 1. A tooth deleted from a function that names no path. This is the whole
    #    reason the inventory now follows calls: before, this edit was invisible.
    tooth = 'release_rust_cache_violations'
    forbidden = '        "Swatinem/rust-cache",\n'
    if source.count(forbidden) != 1:
        controls.append('the tooth-deletion control is stale: %d matches for %r'
                        % (source.count(forbidden), forbidden))
    else:
        pulled = source.replace(forbidden, '', 1)
        if not any(p.startswith('CHANGED') and tooth in p
                   for p in discrepancies(with_root(pulled), INVENTORY)):
            controls.append(
                'deleting a forbidden-substring check inside %s was NOT reported; '
                'the inventory is back to pinning callers' % tooth)

    # 2. #[test] -> #[test] #[ignore]. The body is untouched; only the
    #    attribute says whether the tooth runs at all.
    disabled_victim = 'release_reuses_receipt_bound_archives_without_compiling'
    attr = '#[test]\nfn %s() {' % disabled_victim
    if source.count(attr) != 1:
        controls.append('the attribute control is stale: %d matches' % source.count(attr))
    else:
        disabled = source.replace(attr, '#[test]\n#[ignore]\nfn %s() {' % disabled_victim, 1)
        if not any(p.startswith('CHANGED') and disabled_victim in p
                   for p in discrepancies(with_root(disabled), INVENTORY)):
            controls.append(
                '#[ignore] on %s was NOT reported; a tooth that never runs read '
                'as the tooth that was inventoried' % disabled_victim)

    # 2b. ROUND 13c, FINDING 5: the same kill, written as a MULTI-LINE
    #     attribute. `#[cfg(\n    any()\n)]` compiles the tooth out completely,
    #     and the line-wise scanner stopped at the `)]` line -- so the span
    #     began below the attribute that killed it and the digest was unchanged.
    #     The version that shipped said so in a comment; a comment does not fail.
    multiline_attr = '#[cfg(\n    any()\n)]\n'
    if source.count(attr) != 1:
        controls.append('the multi-line attribute control is stale: %d matches'
                        % source.count(attr))
    else:
        cfg_out = source.replace(attr, multiline_attr + attr, 1)
        if not any(p.startswith('CHANGED') and disabled_victim in p
                   for p in discrepancies(with_root(cfg_out), INVENTORY)):
            controls.append(
                'a multi-line #[cfg(any())] above %s was NOT reported; the tooth is '
                'compiled out and the inventory still calls it read' % disabled_victim)
    # And the bracket walk must not run away on a `]` that is only text: an
    # attribute-looking string above a function is not an attribute.
    decoy = 'fn g() {\n    let _ = "#[ignore]";\n}\nfn h() {\n    let _ = 1;\n}\n'
    decoy_spans = {n: (a, b) for n, a, b in functions(decoy)}
    if 'h' in decoy_spans and 'fn g' in decoy[decoy_spans['h'][0]:decoy_spans['h'][1]]:
        controls.append(
            'the attribute walk swallowed the previous function on a "]" that was '
            'inside a string literal')

    # 2c. ROUND 13c, FINDING 5, the other half: an edge no CALL regex can see.
    #     `let f = tooth;` then `f("x")` -- the string "tooth(" appears nowhere
    #     -- so the old relation left it outside the closure, outside the
    #     inventory, and free to be deleted unnoticed. A macro that pastes the
    #     name is the same shape.
    pointer_fixture = (
        'fn tooth(text: &str) -> bool { text.contains("cargo build") }\n'
        'fn seed() {\n'
        '    let _ = ".github/workflows/release.yml";\n'
        '    let check = tooth;\n'
        '    assert!(check("cargo build"));\n'
        '}\n')
    pointer_seeds = list(sites(pointer_fixture))
    if pointer_seeds != ['seed']:
        controls.append('the function-pointer control found seeds %s, expected [seed]'
                        % pointer_seeds)
    elif 'tooth' not in reachable(pointer_fixture, pointer_seeds):
        controls.append(
            'a function reached only through a function POINTER is outside the '
            'closure; the edge relation is back to being a call regex')

    # 2d. ROUND 13d, FINDING 5, the half round 13c did not answer: an edge that
    #     exists only after MACRO EXPANSION. `seed` contains `check!()` and
    #     nothing else; the identifier `tooth` is in the macro's body, not in
    #     `seed`'s span, so a relation that reads only function spans stops one
    #     step short and the tooth is outside the inventory. This is the case
    #     AD17 claimed and did not have.
    macro_fixture = (
        'macro_rules! check {\n'
        '    () => { assert!(tooth("cargo build")); };\n'
        '}\n'
        'fn tooth(text: &str) -> bool { text.contains("cargo build") }\n'
        'fn seed() {\n'
        '    let _ = ".github/workflows/release.yml";\n'
        '    check!();\n'
        '}\n')
    macro_seeds = list(sites(macro_fixture))
    if macro_seeds != ['seed']:
        controls.append('the macro-expansion control found seeds %s, expected [seed]'
                        % macro_seeds)
    elif 'tooth' not in reachable(macro_fixture, macro_seeds):
        controls.append(
            'a function reached only through a MACRO EXPANSION is outside the '
            'closure; the edge relation reads function spans and stops there')

    # 2e. And the construct for which no scan can over-approximate: an
    #     identifier assembled at expansion time appears nowhere in the source.
    #     The honest answer is a refusal, not a shorter closure.
    for pasting in ('paste! { fn [<tooth _ a>]() {} }',
                    'concat_idents!(tooth, _a)()'):
        pasted = macro_fixture.replace('check!();', pasting)
        try:
            reachable(pasted, ['seed'])
        except Unmeasurable as refusal:
            # ROUND 4 FOLLOW-UP: the one-branch witness. Written `except
            # Unmeasurable: pass`, the exception IS the pass condition -- and
            # this file raises Unmeasurable at eight distinct sites. So the
            # control credited any of them: an attribute the inert list has not
            # heard of, a `mod` whose file would not open, a fixture that failed
            # to substitute. A witness reachable only from the exception path
            # ratifies "something refused here", never "this construct did".
            # Naming the rule is what turns it back into a measurement.
            if 'assembles an identifier' not in str(refusal):
                controls.append(
                    'identifier pasting (%r) was refused, but not by the rule '
                    'under test; the refusal reads: %s' % (pasting, refusal))
        else:
            controls.append(
                'identifier pasting (%r) produced a closure instead of a refusal; '
                'the call graph cannot be bounded in its presence' % pasting)

    # 2f. ROUND 13e, REOPENED FINDING 5. The refusal above matches the crate's
    #     own spelling, and an import may rename it.
    #
    #     Measured, not assumed: `use paste::paste as p;` is ALREADY refused,
    #     because renaming the macro does not rename the crate and `paste::` is
    #     still in the statement. `use std::concat_idents as ci;` is the one
    #     that got through -- `concat_idents` is matched only with a `!` after
    #     it, and an import has none -- so `ci!(tooth, _a)` assembled an
    #     identifier with the refusal switched off. Both are controlled, and the
    #     `covered_already` flag is what keeps this row honest about which rule
    #     is doing the work rather than claiming the new one for all three.
    for importing, invoking, covered_already in (
        ('use paste::paste as p;\n', 'p! { fn [<tooth _ a>]() {} }', True),
        ('use paste::{paste as pz};\n', 'pz! { fn [<tooth _ a>]() {} }', True),
        ('use std::concat_idents as ci;\n', 'ci!(tooth, _a)()', False),
    ):
        aliased = importing + macro_fixture.replace('check!();', invoking)
        if bool(PASTES.search(mask(aliased))) != covered_already:
            controls.append(
                'the aliased-paste control (%r) is not the case it claims: the '
                'crate-spelling rule %s match it' % (
                    importing.strip(), 'does' if covered_already else 'does not'))
            continue
        try:
            reachable(aliased, ['seed'])
        except Unmeasurable as refusal:
            # The same repair, and here the rule the row is entitled to differs
            # per row: the two crate-spelled aliases are still caught by PASTES
            # ("assembles an identifier"), while `concat_idents as ci` is caught
            # only by the import rule. `covered_already` already records which,
            # so the control can say which refusal would vindicate it.
            wanted = ('assembles an identifier' if covered_already
                      else 'imports a pasting macro')
            if wanted not in str(refusal):
                controls.append(
                    'the aliased-paste control (%r) was refused, but not by the '
                    'rule it tests (nothing naming %r in it); the refusal reads: '
                    '%s' % (importing.strip(), wanted, refusal))
        else:
            controls.append(
                'a pasting macro imported as %r produced a closure instead of a '
                'refusal; the refusal is spelling-bound' % importing.strip())

    # 2g. ROUND 13e, REOPENED FINDING 5, the first half: a tooth is still a
    #     function when it is declared `const`, `unsafe` or `extern`. Under the
    #     old qualifier list it had no span at all, so it was not in the
    #     inventory and nothing reported that it was missing.
    for qualifier in ('const', 'unsafe', 'pub const', 'pub(crate) unsafe',
                      'extern "C"', 'pub unsafe extern "C"', 'async'):
        qualified = (
            '%s fn tooth(text: &str) -> bool { text.contains("cargo build") }\n'
            'fn seed() {\n'
            '    let _ = ".github/workflows/release.yml";\n'
            '    assert!(tooth("cargo build"));\n'
            '}\n' % qualifier)
        names = {n for n, _, _ in functions(qualified)}
        if 'tooth' not in names:
            controls.append(
                'a function declared `%s fn` is not a function to this harness; '
                'it would have no span, no digest and no inventory row' % qualifier)
        elif 'tooth' not in reachable(qualified, ['seed']):
            controls.append(
                'a `%s fn` reached from a seed is outside the closure' % qualifier)

    # 3. Indentation INSIDE a string. These teeth assert over YAML held in Rust
    #    string literals, where two spaces versus four is the meaning; the old
    #    per-line strip hashed both to the same value.
    four = 'fn f() {\n    let y = r#"\n    jobs:\n      a: 1\n"#;\n}\n'
    two = 'fn f() {\n    let y = r#"\n    jobs:\n        a: 1\n"#;\n}\n'
    if digest_of(four) == digest_of(two):
        controls.append(
            'reindenting YAML inside a string literal did NOT change the digest; '
            'the normalisation is hiding the one thing these teeth read')

    # The masker earns its keep only if a brace inside a string cannot shift
    # every span after it. Attribution must survive one.
    tricky = ('fn decoy() {\n    let s = "{ unbalanced";\n    let c = \'}\';\n}\n'
              'fn holder() {\n'
              '    let _ = ".github/workflows/release.yml";\n}\n')
    tricky_sites = sites(tricky)
    if list(tricky_sites) != ['holder']:
        controls.append(
            'a brace inside a string literal broke attribution: got %s'
            % list(tricky_sites))

    # --- ROUND 3 (Codex Sol), FINDING N4 -----------------------------------
    #
    # Four more ways a tooth can be outside a literal-seeded, file-local scan,
    # and the fifth way two of them can be inside it and still collapse into one
    # row. Each control below has to make the harness REFUSE or REPORT; a
    # control that produces a shorter closure instead is the finding, not a pass.

    def refuses(subject, seeds=('seed',)) -> str | None:
        """The refusal text, or None if a closure came back instead.

        It returns the TEXT and not a bool on purpose: every caller below that
        requires a refusal has to check which rule produced it. `Unmeasurable`
        is raised at eight sites in this file, so `refuses(x) is not None` says
        only that something here declined -- the same one-branch-witness hole
        repaired in sections 2e, 2f and 4b above. Callers that require the
        ABSENCE of a refusal (`is not None` is the complaint) are sound as they
        stand: a refusal from any rule is a failure there whatever raised it.
        """
        try:
            reachable(subject, list(seeds))
        except Unmeasurable as refusal:
            return str(refusal)
        return None

    # 4a. THE SEED. A tooth whose only mention of the file is the bare name --
    #     `.expect("read release.yml")` is how ten of drift_guard.rs's twenty-
    #     eight mentions are spelled -- was outside a seed set that required the
    #     directory. Measured both ways so the control cannot silently become
    #     vacuous if MENTION widens again.
    bare_name = ('fn bare_seed() {\n'
                 '    let _ = std::fs::read_to_string(p).expect("read release.yml");\n'
                 '}\n')
    if list(sites(bare_name)) != ['bare_seed']:
        controls.append(
            'a function whose only mention is the bare file name is not a seed: '
            'got %s' % list(sites(bare_name)))
    if 'workflows/release.yml' in bare_name:
        controls.append('the bare-name seed fixture is not bare; it names a '
                        'directory, so it proves nothing about the seed')

    # 4b. ANOTHER FILE. `mod x;` whose body is elsewhere. The real crate has
    #     two; a declaration this harness cannot resolve is a refusal, because
    #     the alternative is a closure that is short by however many teeth live
    #     in the file it could not open.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root_rs = os.path.join(tmp, 'root.rs')
        # newline='' on both fixture writes below. On Windows the default text
        # mode turns every '\n' in these literals into '\r\n' on disk, so the
        # file this control feeds to load_units is not the file written here.
        # It is not a live defect -- load_units reads in text mode too, so the
        # round trip cancels -- but the cancellation is a coincidence of two
        # translations, and a fixture that is not the bytes its source says it
        # is cannot be reasoned about when either half changes.
        io.open(root_rs, 'w', encoding='utf-8', newline='').write(
            'mod helper;\nfn seed() {\n'
            '    let _ = ".github/workflows/release.yml";\n    helper::tooth();\n}\n')
        try:
            load_units(root_rs)
        except Unmeasurable as refusal:
            # One-branch witness again: this fixture must be refused BECAUSE the
            # module has no file, not because load_units tripped over something
            # else in a temp directory it has never seen before.
            if 'and none of' not in str(refusal):
                controls.append(
                    'the missing-`mod` control was refused for another reason, '
                    'so it shows nothing about missing modules: %s' % refusal)
        else:
            controls.append(
                'a `mod x;` whose file does not exist produced a unit list '
                'instead of a refusal; every function in it would be missing '
                'from the closure with nothing said')
        os.makedirs(os.path.join(tmp, 'root'))
        io.open(os.path.join(tmp, 'root', 'helper.rs'), 'w',
                encoding='utf-8', newline='').write(
            'pub fn tooth(text: &str) -> bool { text.contains("cargo build") }\n')
        try:
            resolved = load_units(root_rs)
        except Unmeasurable as refusal:
            controls.append('a resolvable `mod x;` was refused: %s' % refusal)
        else:
            if len(resolved) != 2:
                controls.append('a `mod x;` file was not loaded as a unit: %s'
                                % [u.module for u in resolved])
            elif 'helper::tooth' not in reachable(resolved, ['seed']):
                controls.append(
                    'a tooth in ANOTHER FILE, reached by `helper::tooth()`, is '
                    'outside the closure; the scan is still file-local and says '
                    'nothing about it')

    # 4c. `include!` splices items this harness has never read.
    included = ('fn seed() {\n'
                '    let _ = ".github/workflows/release.yml";\n'
                '    include!("teeth.rs");\n}\n')
    included_refusal = refuses(included)
    if included_refusal is None:
        controls.append(
            '`include!` produced a closure instead of a refusal; the items it '
            'splices in are functions of this module that no span here has seen')
    elif 'uses `include!`' not in included_refusal:
        controls.append(
            'the `include!` fixture was refused by some other rule, so it does '
            'not establish that `include!` is refused: %s' % included_refusal)

    # 4d. An attribute macro on a function IN the closure can replace its body,
    #     so the digest pins text that will never run.
    rewritten = ('#[wrap_it]\nfn seed() {\n'
                 '    let _ = ".github/workflows/release.yml";\n}\n')
    rewritten_refusal = refuses(rewritten)
    if rewritten_refusal is None:
        controls.append(
            'a non-inert attribute macro on a function in the closure produced a '
            'digest instead of a refusal; the body digested need not be the body '
            'that runs')
    elif 'is not a known inert attribute' not in rewritten_refusal:
        controls.append(
            'the attribute-macro fixture was refused by some other rule, so it '
            'does not establish that a rewriting attribute is refused: %s'
            % rewritten_refusal)
    inert = ('#[test]\n#[cfg(test)]\nfn seed() {\n'
             '    let _ = ".github/workflows/release.yml";\n}\n')
    if refuses(inert) is not None:
        controls.append(
            'the attribute refusal fired on `#[test]` / `#[cfg(test)]`; it is not '
            'discriminating and would make every real run unmeasurable')

    # 4e. An ASSEMBLED path: the file name is not in the source at all.
    # Each row carries the phrase its own rule uses: `format!` is caught by the
    # runtime-file-name rule, `.join(name)` by the non-literal-component rule,
    # and a control that accepted either refusal for either fixture would not
    # notice if one of the two rules stopped firing altogether.
    for assembled, naming in (
        ('fn seed() {\n'
         '    let p = format!("{}/workflows/{}.yml", dir, name);\n}\n',
         'decided at runtime'),
        ('fn seed() {\n'
         '    let p = root.join(".github/workflows").join(name);\n}\n',
         'joins a non-literal component'),
    ):
        assembled_refusal = refuses(assembled, seeds=())
        if assembled_refusal is None:
            controls.append(
                'an assembled workflow path produced a closure instead of a '
                'refusal: %r' % ' '.join(assembled.split())[:70])
        elif naming not in assembled_refusal:
            controls.append(
                'the assembled-path fixture %r was refused by some other rule '
                '(nothing naming %r in it): %s'
                % (' '.join(assembled.split())[:70], naming, assembled_refusal))
    # And it must not fire on the shape the real file uses, or the refusal is
    # not a measurement, it is a permanent outage.
    literal_path = ('fn seed() {\n'
                    '    let p = repo_root().join(".github/workflows/release.yml");\n}\n')
    if refuses(literal_path, seeds=('seed',)) is not None:
        controls.append(
            'the assembled-path refusal fired on a literal `.join(".github/'
            'workflows/release.yml")`, which is what drift_guard.rs writes')

    # 4f. FINDING N8: two bodies, one row. Two helpers spelled the same inside
    #     two different functions -- which is what r4_test_support_test.rs does
    #     with `scan_items` -- used to collapse into a single name-keyed entry
    #     whose digest was their concatenation.
    twins = ('fn seed() {\n'
             '    let _ = ".github/workflows/release.yml";\n'
             '    outer_a();\n    outer_b();\n}\n'
             'fn outer_a() {\n    fn inner() -> u8 { 1 }\n    inner();\n}\n'
             'fn outer_b() {\n    fn inner() -> u8 { 2 }\n    inner();\n}\n')
    twin_names = {n for n, _, _ in functions(twins)}
    if 'outer_a::inner' not in twin_names or 'outer_b::inner' not in twin_names:
        controls.append(
            'two same-named helpers in two different functions were not '
            'separated by scope: got %s' % sorted(twin_names))
    if 'inner' in twin_names:
        controls.append(
            'a nested helper is still keyed by its bare name, so two of them '
            'share one row, one digest and one disposition')
    # The same for two impl blocks, which is drift_guard.rs's own shape:
    # three `record_cfg_test_node` and three `visit_expr`.
    impls = ('struct A;\nstruct B;\n'
             'impl A {\n    fn probe(&self) -> u8 { 1 }\n}\n'
             'impl B {\n    fn probe(&self) -> u8 { 2 }\n}\n')
    impl_names = {n for n, _, _ in functions(impls)}
    if impl_names != {'A::probe', 'B::probe'}:
        controls.append(
            'two same-named methods in two impl blocks were not separated by '
            'scope: got %s' % sorted(impl_names))
    # ...and a duplicate that survives qualification must be REPORTED, not
    # merged. Nothing in Rust produces one at file scope, so it is produced here.
    forced = ('fn seed() {\n'
              '    let _ = ".github/workflows/release.yml";\n    twin();\n}\n'
              'fn twin() -> u8 { 1 }\n'
              'fn twin() -> u8 { 2 }\n')
    forced_units = as_units(forced)
    reachable(forced_units, ['seed'])
    if 'twin' not in DUPLICATES:
        controls.append(
            'a function defined twice at the same qualified name was not '
            'reported; its row would carry the concatenation of two bodies')

    for control in controls:
        print('  CONTROL FAILED: %s' % control)

    print()
    print('INVENTORY: %d function(s), %d discrepancy(ies), %d control failure(s)'
          % (len(found), len(problems), len(controls)))
    failures = len(problems) + len(controls)
    globals()['_COMPLETED'] = True
    print('%s %s failures=%d elapsed=%.1fs'
          % (MARKER, HARNESS, failures, time.time() - started))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
