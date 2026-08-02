#!/usr/bin/env python3
"""Enumerate internal page-prose readers for the M5 reader manifest.

The M5 Stage 0 reader manifest (docs/plans/2026-07-27-m5-reader-manifest-inventory.md)
carries a set of internal call sites that read page prose. Three review rounds
produced three different counts because the predicate lived in prose and each
implementation interpreted it differently. The predicate now lives here, and the
number is whatever this script says.

CI routes edits to this executable contract to its exact drift-guard test.

PREDICATE (exact, and the only one that counts):

  A function is an internal page-prose reader iff its brace-matched body
  contains a string literal that both

    (a) matches FROM/JOIN <whitespace> pages   (case-insensitive), and
    (b) contains one of the prose column names as a whole word:
        content, title, summary, excerpt, body, snippet

  scanned over crates/{wenlan-core,wenlan-server,wenlan-cli,wenlan-mcp}/src,
  skipping *_test.rs / *_tests.rs files and #[cfg(test)] blocks (brace-tracked,
  never first-match truncation).

Both halves must hold INSIDE THE SAME string literal. An earlier version tested
(a) and (b) anywhere in the function body, which counted `list_tags_scoped`
(selects tags, merely checks page existence) and `tally` (no page SQL at all).

Function bodies are brace-matched. An earlier version delimited a body by the
next `fn` match, which merged adjacent functions and inflated the set.

CALLERS: the SQL-bearing definitions are the *sources* of page prose, not the
readers the governing spec asks for. `get_page_inner` is private; the actual
reader is whatever calls `get_page` and does something with `page.content` —
e.g. the citation backfill at citations.rs:423. So the set is expanded to the
transitive caller closure of the SQL-bearing definitions, to a stated depth.
Depth 0 is the SQL layer, depth 1 the wrapper layer, depth 2+ the consumers.

The depth is DEPTH_TITLES, and it reaches 3 because a two-deep closure stopped
one hop short of the HTTP handler on several routes. `handle_review_page` calls
`review_page_with_presence` calls `review_in_txn` calls the `pages` SELECT: four
names, so at depth 2 the endpoint — the thing an external reader most wants to
find — was absent rather than listed. This is a parameter, not a claim that
nothing lies past it.

Measured once, when the cap was raised: depth 3 added 75 rows, 29 of them in
wenlan-server, and 23 of those 29 are handlers actually registered on a route
(cross-checked by name against
crates/wenlan-server/src/route_registry/handler_manifest.txt; the other six are
`run_daemon`, two page-map helpers, two scheduler functions, and a source-sync
function). A one-time measurement, not a maintained figure — re-derive it from
--json rather than trusting these numbers later.

EXPOSURE: a reader is an exposure path iff it is `pub` (unqualified — not
`pub(crate)` / `pub(super)`) AND called from outside wenlan-core. Caller edges
are resolved by NAME, which over-matches on generic names; rows whose name is
ambiguous are flagged rather than given a caller list. PR-B replaces this with
language-server resolution.

IDENTITY: committed rows use `short/path.rs::function[#ordinal]`; line numbers
remain in --json only. Caller identities are the uniquely enclosing
brace-matched function. A relevant name-resolvable call with no unique owner
fails closed. The exact generated list retains duplicate multiplicity.

Usage:
  python3 scripts/m5-reader-sweep.py [--json]
  python3 scripts/m5-reader-sweep.py --self-test
  python3 scripts/m5-reader-sweep.py --check
  python3 scripts/m5-reader-sweep.py --update-inventory
"""
import difflib
import json
import os
import re
import sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path

MAX_FN_LINES = 12000
# One entry per depth the closure walks, so the walk and the rendered sections
# cannot disagree about how far it went.
DEPTH_TITLES = (
    'SQL-bearing definitions',
    'wrapper layer',
    'consumers',
    'outer consumers — route handlers and orchestration',
)
MAX_DEPTH = len(DEPTH_TITLES) - 1
TRUNCATED = []
INVENTORY = 'docs/plans/2026-07-27-m5-reader-manifest-inventory.md'
INVENTORY_BEGIN = '<!-- m5-reader-sweep:begin -->'
INVENTORY_END = '<!-- m5-reader-sweep:end -->'
REPO_ROOT = Path(__file__).resolve().parents[1]

FN = re.compile(
    r'^(\s*)(pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+(\w+)'
)
PROSE = re.compile(r'\b(content|title|summary|excerpt|body|snippet)\b')
SQLBLK = re.compile(r'"[^"]*(?:FROM|JOIN)\s+pages\b[^"]*"', re.I | re.S)
RAW_STRING_START = re.compile(r'r(#{0,255})"')
CHAR_LITERAL = re.compile(
    r"""'(?:\\(?:[nrt0\\'"]|x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]{1,6}\})|[^'\\])'"""
)
CRATES = ['crates/wenlan-core/src', 'crates/wenlan-server/src',
          'crates/wenlan-cli/src', 'crates/wenlan-mcp/src']
CORE = 'crates/wenlan-core/src'


def rust_structure(lines):
    """Mask comments and literals while preserving structural Rust tokens."""
    out = []
    block_depth = 0
    normal_string = False
    escaped = False
    raw_end = None

    for line in lines:
        masked = [' '] * len(line)
        i = 0
        while i < len(line):
            if raw_end is not None:
                end = line.find(raw_end, i)
                if end < 0:
                    i = len(line)
                else:
                    i = end + len(raw_end)
                    raw_end = None
                continue

            if normal_string:
                ch = line[i]
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    normal_string = False
                i += 1
                continue

            if block_depth:
                if line.startswith('/*', i):
                    block_depth += 1
                    i += 2
                elif line.startswith('*/', i):
                    block_depth -= 1
                    i += 2
                else:
                    i += 1
                continue

            if line.startswith('//', i):
                break
            if line.startswith('/*', i):
                block_depth = 1
                i += 2
                continue

            raw = RAW_STRING_START.match(line, i)
            if raw:
                raw_end = '"' + raw.group(1)
                i = raw.end()
                continue

            if line[i] == '"':
                normal_string = True
                escaped = False
                i += 1
                continue

            if line[i] == "'":
                char = CHAR_LITERAL.match(line, i)
                if char:
                    i = char.end()
                    continue

            masked[i] = line[i]
            i += 1
        out.append(''.join(masked))

    return out


def strip_test(lines, include_structure=False):
    """Blank test blocks, optionally returning the already-computed structure."""
    out, kept_structure, skip, depth, armed = [], [], 0, 0, False
    structure = rust_structure(lines)
    for l, structural in zip(lines, structure):
        if not skip and re.match(r'\s*#\[cfg\(test\)\]', structural):
            armed = True; out.append(''); kept_structure.append(''); continue
        if armed and not skip:
            if '{' not in structural and structural.strip().endswith(';'):
                # `#[cfg(test)] mod foo;` / `use ...;` / a test-only const.
                # The attribute governs an item that never opens a brace, so
                # arming the brace scan here would blank forward until the NEXT
                # unrelated `{` and swallow real code -- 45% of db.rs, in the
                # version that shipped. Blank the item, disarm, move on.
                out.append(''); kept_structure.append(''); armed = False; continue
            if '{' in structural:
                skip, depth, armed = (
                    1,
                    structural.count('{') - structural.count('}'),
                    False,
                )
                out.append(''); kept_structure.append('')
                if depth <= 0: skip = 0
                continue
            out.append(''); kept_structure.append(''); continue
        if skip:
            depth += structural.count('{') - structural.count('}')
            out.append(''); kept_structure.append('')
            if depth <= 0: skip = 0
            continue
        out.append(l); kept_structure.append(structural)
    if include_structure:
        return out, kept_structure
    return out


def body_spans(lines, structure=None, offsets=None):
    """Yield brace-matched function spans without changing scanner semantics."""
    i, n = 0, len(lines)
    structure = structure or rust_structure(lines)
    offsets = offsets or _line_offsets(structure)
    while i < n:
        m = FN.match(structure[i])
        if not m:
            i += 1; continue
        name_hint = m.group(3)
        j, depth, started, closed_at = i, 0, False, None
        while j < n:
            for column, ch in enumerate(structure[j]):
                if ch == '{': depth += 1; started = True
                elif ch == '}': depth -= 1
                if started and depth <= 0:
                    closed_at = column + 1
                    break
            if closed_at is not None: break
            j += 1
            if j - i > MAX_FN_LINES:
                # run_migrations is a known multi-thousand-line function. Past
                # this cap the brace scan has lost sync (an unbalanced brace
                # inside a string or macro), so report it rather than silently
                # truncating a body. Avoid pinning its fast-moving line count in
                # this comment: the executable inventory checks the real extent.
                t = '%s:%d' % (name_hint, i + 1)
                if t not in TRUNCATED: TRUNCATED.append(t)
                break
        vis = (m.group(2) or '').strip() or 'private'
        end_index = min(j, n - 1)
        body_lines = lines[i:end_index + 1]
        if closed_at is not None:
            body_lines[-1] = body_lines[-1][:closed_at]
        start_offset = offsets[i] if closed_at is not None else None
        end_offset = (
            offsets[end_index] + closed_at
            if closed_at is not None else None
        )
        yield (
            i + 1,
            end_index + 1,
            m.group(3),
            vis,
            '\n'.join(body_lines),
            start_offset,
            end_offset,
        )
        i = max(j + 1, i + 1)


def bodies(lines):
    """Yield the historical body tuple used by structural regression tests."""
    for start, _, name, vis, body, _, _ in body_spans(lines):
        yield (start, name, vis, body)


def canonical_path(path):
    """Return one repository-relative spelling on POSIX and Windows."""
    return os.path.normpath(path).replace('\\', '/')


def rust_files():
    paths = []
    for base in CRATES:
        for root, dirs, fs in os.walk(REPO_ROOT / base):
            dirs.sort()
            for f in fs:
                if f.endswith('.rs') and not f.endswith(('_test.rs', '_tests.rs')):
                    paths.append(
                        (Path(root) / f).relative_to(REPO_ROOT).as_posix()
                    )
    yield from sorted(paths)


def short_path(path):
    path = canonical_path(path)
    prefixes = {
        'crates/wenlan-core/src/': 'core/',
        'crates/wenlan-server/src/': 'server/',
        'crates/wenlan-cli/src/': 'cli/',
        'crates/wenlan-mcp/src/': 'mcp/',
    }
    for prefix, short in prefixes.items():
        if path.startswith(prefix):
            return short + path[len(prefix):]
    return path


CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')


def _line_offsets(lines):
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1
    return offsets


def _owner_locators(functions, offset):
    return [
        function['locator']
        for function in functions
        if function['start_offset'] is not None
        and function['start_offset'] <= offset < function['end_offset']
    ]


def _validate_call_ownership(index, names):
    """Fail closed if a relevant, name-resolvable call has no unique owner."""
    for name in sorted(set(names)):
        if len(index['declared'].get(name, ())) != 1:
            continue
        unresolved = index['unresolved_by_name'].get(name, ())
        if unresolved:
            raise ValueError(
                'callsite ownership for %s is not unique: %s'
                % (name, ', '.join(unresolved))
            )


def build_index(sources):
    """Index functions and calls once; retain lines only for diagnostics."""
    normalized = {
        canonical_path(path): text
        for path, text in sources.items()
    }
    functions = []
    declared = {}
    per_file = {}

    for path in sorted(normalized):
        lines, structure = strip_test(
            normalized[path].split('\n'), include_structure=True
        )
        code_text = '\n'.join(structure)
        offsets = _line_offsets(structure)
        file_functions = []
        definition_offsets = set()

        for line_index, structural in enumerate(structure):
            match = FN.match(structural)
            if match:
                definition_offsets.add(offsets[line_index] + match.start(3))

        for match in re.finditer(r'\bfn\s+(\w+)', code_text):
            name = match.group(1)
            line = bisect_right(offsets, match.start())
            declared.setdefault(name, set()).add('%s:%d' % (path, line))

        for start, end, name, vis, body, start_offset, end_offset in body_spans(
                lines, structure, offsets):
            function = {
                'file': path,
                'line': start,
                'end_line': end,
                'fn': name,
                'vis': vis,
                'body': body,
                'start_offset': start_offset,
                'end_offset': end_offset,
            }
            file_functions.append(function)

        totals = Counter(function['fn'] for function in file_functions)
        seen = Counter()
        for function in file_functions:
            seen[function['fn']] += 1
            suffix = ''
            if totals[function['fn']] > 1:
                suffix = '#%d' % seen[function['fn']]
            function['locator'] = '%s::%s%s' % (
                short_path(path), function['fn'], suffix
            )
        functions.extend(file_functions)
        per_file[path] = {
            'code': code_text,
            'offsets': offsets,
            'definitions': definition_offsets,
            'functions': file_functions,
        }

    calls_by_locator = {}
    callers_by_name = {}
    unresolved_by_name = {}
    for path, file_index in per_file.items():
        for match in CALL.finditer(file_index['code']):
            if match.start(1) in file_index['definitions']:
                continue
            name = match.group(1)
            line = bisect_right(file_index['offsets'], match.start(1))
            owners = _owner_locators(file_index['functions'], match.start(1))
            if len(owners) != 1:
                detail = '%s:%d (%d owners)' % (path, line, len(owners))
                unresolved_by_name.setdefault(name, []).append(detail)
                continue
            owner = owners[0]
            calls_by_locator.setdefault(owner, set()).add(name)
            callers_by_name.setdefault(name, set()).add(owner)

    return {
        'functions': functions,
        'by_locator': {function['locator']: function for function in functions},
        'declared': declared,
        'calls_by_locator': calls_by_locator,
        'callers_by_name': callers_by_name,
        'unresolved_by_name': unresolved_by_name,
    }


def _public_row(function, depth, index, via=None):
    row = {
        key: function[key]
        for key in ('file', 'line', 'fn', 'vis', 'locator')
    }
    row['depth'] = depth
    if via is not None:
        row['via'] = sorted(via)
    row['ambiguous'] = len(index['declared'].get(row['fn'], ())) > 1
    if row['ambiguous']:
        row['ext'] = []
    else:
        _validate_call_ownership(index, (row['fn'],))
        row['ext'] = sorted(
            caller
            for caller in index['callers_by_name'].get(row['fn'], ())
            if not index['by_locator'][caller]['file'].startswith(CORE + '/')
            and index['by_locator'][caller]['file'] != CORE
        )
    row['exposure'] = row['vis'] == 'pub' and bool(row['ext'])
    return row


def _read_sources():
    return {
        path: (REPO_ROOT / path).read_text(encoding='utf-8')
        for path in rust_files()
    }


def sweep(sources=None, index=None):
    if index is None:
        index = build_index(_read_sources() if sources is None else sources)
    readers = []
    for function in index['functions']:
        sqls = SQLBLK.findall(function['body'])
        if sqls and any(PROSE.search(query) for query in sqls):
            readers.append(_public_row(function, 0, index))

    # Transitive caller closure. Name ambiguity remains explicit until a Rust
    # resolver replaces this scanner, but every accepted call has one owner.
    known = {reader['locator'] for reader in readers}
    frontier = {reader['fn']: reader['locator'] for reader in readers}
    out = list(readers)
    for depth in range(1, MAX_DEPTH + 1):
        eligible = {
            name for name in frontier
            if len(index['declared'].get(name, ())) == 1
        }
        _validate_call_ownership(index, eligible)
        for function in index['functions']:
            if function['locator'] in known:
                continue
            matches = sorted(
                index['calls_by_locator'].get(function['locator'], set()) & eligible
            )
            if not matches:
                continue
            row = _public_row(
                function,
                depth,
                index,
                via=[frontier[name] for name in matches],
            )
            out.append(row)
            known.add(function['locator'])
        frontier = {
            row['fn']: row['locator']
            for row in out if row['depth'] == depth
        }
    out.sort(key=lambda row: (row['depth'], row['locator']))
    return out


def render_inventory(rows):
    """Render the canonical, generated reader rows embedded in the M5 inventory."""
    sections = [INVENTORY_BEGIN]
    for depth, title in enumerate(DEPTH_TITLES):
        sections.extend([
            '',
            '### Depth %d — %s' % (depth, title),
            '',
            '| Reader | Visibility | Ambiguous | Exposure | External callers | Reaches prose via |',
            '|---|---|---|---|---|---|',
        ])
        for row in (r for r in rows if r['depth'] == depth):
            callers = ', '.join('`%s`' % caller for caller in row['ext']) or '—'
            via = ', '.join(
                '`%s`' % predecessor for predecessor in row.get('via', ())
            ) or '—'
            sections.append(
                '| `%s` | `%s` | %s | %s | %s | %s |'
                % (
                    row['locator'],
                    row['vis'],
                    'yes' if row['ambiguous'] else 'no',
                    '**yes**' if row['exposure'] else 'no',
                    callers,
                    via,
                )
            )
    sections.extend(['', INVENTORY_END])
    return '\n'.join(sections)


def inventory_block(document):
    start = document.find(INVENTORY_BEGIN)
    end = document.find(INVENTORY_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(
            '%s must contain one ordered %s / %s marker pair'
            % (INVENTORY, INVENTORY_BEGIN, INVENTORY_END)
        )
    if document.find(INVENTORY_BEGIN, start + 1) >= 0 \
            or document.find(INVENTORY_END, end + 1) >= 0:
        raise ValueError('%s contains duplicate reader inventory markers' % INVENTORY)
    return document[start:end + len(INVENTORY_END)]


def replace_inventory_block(document, generated):
    current = inventory_block(document)
    return document.replace(current, generated, 1)


def check_inventory(rows):
    generated = render_inventory(rows)
    document = (REPO_ROOT / INVENTORY).read_text(encoding='utf-8')
    current = inventory_block(document)
    if current != generated:
        diff = difflib.unified_diff(
            current.splitlines(),
            generated.splitlines(),
            fromfile=INVENTORY + ' (committed)',
            tofile=INVENTORY + ' (current tree)',
            lineterm='',
        )
        print('\n'.join(diff), file=sys.stderr)
        return False

    depths = Counter(r['depth'] for r in rows)
    exposure = sum(r['exposure'] for r in rows)
    print(
        'reader inventory check: ok '
        '(%d rows; depth %s; exposure %d)'
        % (
            len(rows),
            '/'.join(str(depths[d]) for d in range(MAX_DEPTH + 1)),
            exposure,
        )
    )
    return True


def update_inventory(rows):
    generated = render_inventory(rows)
    inventory_path = REPO_ROOT / INVENTORY
    document = inventory_path.read_text(encoding='utf-8')
    updated = replace_inventory_block(document, generated)
    with open(inventory_path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(updated)
    print('updated %s (%d rows)' % (INVENTORY, len(rows)))


def selftest(repository_index=None):
    """Assert the structural predicates that decide what this script can see.

    These bugs all shipped once. `strip_test` armed the brace scan on
    `#[cfg(test)] mod foo;` -- an attribute over an item with no brace -- and
    then blanked forward to the next unrelated `{`, hiding real code from the
    whole sweep; five HTTP route handlers went uncounted. The `ext` scan
    reported a function's own definition line as an external caller of itself,
    which reads as an exposure path in the output table. The body scanner also
    counted braces inside strings and comments, making `run_migrations` swallow
    later sibling methods and hiding a page-prose reader.
    """
    src = """#[cfg(test)]
mod claim_identity_test;

#[cfg(test)]
#[path = "other_test.rs"]
mod other_test;

pub async fn real_reader(&self) {
    let q = "SELECT content FROM pages WHERE id = ?1";
}

#[cfg(test)]
mod tests {
    fn hidden_by_design() {
        let q = "SELECT content FROM pages";
    }
}

pub async fn also_real(&self) {
    let q = "SELECT title FROM pages";
}"""
    kept = '\n'.join(strip_test(src.split('\n')))
    assert 'real_reader' in kept, 'a `#[cfg(test)] mod foo;` swallowed the code after it'
    assert 'also_real' in kept, 'a #[cfg(test)] block swallowed the code after it'
    assert 'hidden_by_design' not in kept, 'a real #[cfg(test)] block was not stripped'
    assert len(kept.split('\n')) == len(src.split('\n')), 'line numbers must survive stripping'

    brace_fixture = '''fn stringy() {
    let normal = "}";
    let raw = r##"{
        still a string
    }"##;
    /* a comment with a closing brace: } */
}

fn after_stringy() {}'''
    parsed = list(bodies(brace_fixture.split('\n')))
    assert [row[1] for row in parsed] == ['stringy', 'after_stringy'], \
        'braces inside Rust strings/comments changed function boundaries'
    assert 'let raw' in parsed[0][3], 'a brace in a string truncated the function body'

    if repository_index is None:
        db_lines = strip_test(
            (REPO_ROOT / 'crates/wenlan-core/src/db.rs')
            .read_text(encoding='utf-8')
            .split('\n')
        )
        db_bodies = {name: body for _, name, _, body in bodies(db_lines)}
    else:
        db_bodies = {
            function['fn']: function['body']
            for function in repository_index['functions']
            if function['file'] == 'crates/wenlan-core/src/db.rs'
        }
    assert 'migrate_80_page_scope_fold' in db_bodies, \
        'run_migrations swallowed the next sibling method'
    assert 'fn migrate_80_page_scope_fold' not in db_bodies['run_migrations'], \
        'run_migrations body overran its LSP-resolved end'

    core_path = 'crates/wenlan-core/src/db.rs'
    server_path = 'crates/wenlan-server/src/routes.rs'
    base = {
        core_path: '''fn read_one() {
    let query = "SELECT content FROM pages";
}

fn bridge() {
    read_one();
}
''',
        server_path: '''pub fn handle_one() {
    bridge();
}
''',
    }

    def rendered(sources):
        return render_inventory(sweep(sources))

    stable = rendered(base)
    shifted = dict(base)
    shifted[core_path] = '// line shift only\n\n' + shifted[core_path]
    shifted_rows = sweep(shifted)
    assert stable == rendered(shifted), 'pure line shifts changed the semantic contract'
    assert sweep(base)[0]['line'] != shifted_rows[0]['line'], \
        'diagnostic JSON lines must still report source movement'

    windows = {path.replace('/', '\\'): text for path, text in base.items()}
    assert stable == rendered(windows), 'Windows and POSIX paths produced different identities'
    assert all('\\' not in row['locator'] for row in sweep(windows)), \
        'a Windows path escaped canonical locator normalization'

    added = dict(base)
    added[core_path] += '''
fn read_two() {
    let query = "SELECT title FROM pages";
}
'''
    assert stable != rendered(added), 'an added reader must fail exact equality'
    deleted = {
        core_path: added[core_path].replace('''
fn read_two() {
    let query = "SELECT title FROM pages";
}
''', ''),
        server_path: base[server_path],
    }
    assert rendered(added) != rendered(deleted), \
        'a deleted reader must fail exact equality'
    assert stable == rendered(deleted), \
        'the delete-reader control did not restore its baseline'

    visible = dict(base)
    visible[core_path] = visible[core_path].replace('fn read_one()', 'pub fn read_one()')
    assert stable != rendered(visible), 'a visibility change must fail exact equality'

    direct = {
        core_path: base[core_path].split('\n\nfn bridge')[0] + '\n',
        server_path: '''pub fn handle_one() {
    read_one();
}

pub fn handle_two() {}
''',
    }
    no_caller = dict(direct)
    no_caller[server_path] = no_caller[server_path].replace('    read_one();\n', '')
    assert rendered(direct) != rendered(no_caller), \
        'adding or removing an external caller must fail exact equality'
    moved_caller = dict(direct)
    moved_caller[server_path] = '''pub fn handle_one() {}

pub fn handle_two() {
    read_one();
}
'''
    assert rendered(direct) != rendered(moved_caller), \
        'moving a call to another enclosing function must change its caller identity'
    shifted_caller = dict(direct)
    shifted_caller[server_path] = direct[server_path].replace(
        '    read_one();', '\n\n    read_one();'
    )
    assert rendered(direct) == rendered(shifted_caller), \
        'moving a call inside one enclosing function changed its identity'

    two_readers = {
        core_path: '''fn read_one() {
    let query = "SELECT content FROM pages";
}

fn read_two() {
    let query = "SELECT title FROM pages";
}

fn bridge() {
    read_one();
}
''',
    }
    via_two = dict(two_readers)
    via_two[core_path] = via_two[core_path].replace(
        'fn bridge() {\n    read_one();', 'fn bridge() {\n    read_two();'
    )
    assert rendered(two_readers) != rendered(via_two), \
        'changing the prose path (`via`) must fail exact equality'
    both_via = dict(two_readers)
    both_via[core_path] = both_via[core_path].replace(
        '    read_one();\n}', '    read_one();\n    read_two();\n}', 1
    )
    bridge = next(row for row in sweep(both_via) if row['fn'] == 'bridge')
    assert bridge['via'] == [
        'core/db.rs::read_one', 'core/db.rs::read_two'
    ], 'multiple shortest predecessors were not retained and sorted'
    assert rendered(both_via) != rendered(two_readers), \
        'removing a non-first predecessor must fail exact equality'

    indirect = dict(direct)
    indirect[core_path] += '''
fn bridge() {
    read_one();
}
'''
    indirect[server_path] = indirect[server_path].replace('read_one()', 'bridge()')
    assert rendered(direct) != rendered(indirect), \
        'changing closure depth must fail exact equality'

    duplicate = {
        core_path: '''impl One {
    fn duplicate() {
        let query = "SELECT content FROM pages";
    }
}

impl Two {
    fn duplicate() {
        let query = "SELECT title FROM pages";
    }
}
''',
    }
    duplicate_rows = sweep(duplicate)
    assert [row['locator'] for row in duplicate_rows] == [
        'core/db.rs::duplicate#1', 'core/db.rs::duplicate#2'
    ], 'same-file same-name readers lost their ordinal multiplicity'
    one_duplicate = dict(duplicate)
    one_duplicate[core_path] = duplicate[core_path][duplicate[core_path].index('impl Two'):]
    assert rendered(duplicate) != rendered(one_duplicate), \
        'deleting one duplicate reader must fail exact equality'
    duplicate_caller = dict(duplicate)
    duplicate_caller[core_path] += '''
fn wrapper() {
    duplicate();
}
'''
    assert all(row['fn'] != 'wrapper' for row in sweep(duplicate_caller)), \
        'a name-ambiguous callee produced a downstream edge'

    split_predicate = {
        core_path: '''fn not_a_reader() {
    let columns = "SELECT content";
    let table = "FROM pages";
}
''',
    }
    assert not sweep(split_predicate), \
        'SQL and prose in different string literals must not qualify'

    masked_calls = dict(direct)
    masked_calls[server_path] = '''pub fn handle_one() {
    let example = "read_one(";
    // read_one();
}
'''
    assert all(row['fn'] != 'handle_one' for row in sweep(masked_calls)), \
        'calls inside strings/comments entered the call index'

    recursive_index = build_index({
        core_path: '''fn recursive() {
    let query = "SELECT body FROM pages";
    recursive();
}
''',
    })
    assert recursive_index['callers_by_name']['recursive'] == {'core/db.rs::recursive'}, \
        'a declaration was counted as a call or recursion was dropped'

    unowned = {
        core_path: base[core_path].split('\n\nfn bridge')[0]
        + '\n\nconst _: () = { read_one(); };\n',
    }
    try:
        sweep(unowned)
        assert False, 'an unowned relevant callsite must fail closed'
    except ValueError as error:
        assert 'callsite ownership' in str(error)

    same_line_unowned = {
        core_path: base[core_path].split('\n\nfn bridge')[0]
        + '\n\nfn helper() {} const _: () = { read_one(); };\n',
    }
    try:
        sweep(same_line_unowned)
        assert False, 'a call after a same-line closing brace must fail closed'
    except ValueError as error:
        assert '(0 owners)' in str(error)

    synthetic = {
        'declared': {'read_one': {'core/db.rs:1'}},
        'unresolved_by_name': {'read_one': ['core/db.rs:9 (2 owners)']},
    }
    try:
        _validate_call_ownership(synthetic, {'read_one'})
        assert False, 'an ambiguously owned relevant callsite must fail closed'
    except ValueError as error:
        assert '2 owners' in str(error)

    fixture = 'before\n%s\nafter\n' % stable
    expected = inventory_block(fixture)
    assert replace_inventory_block(fixture, expected) == fixture, \
        'replacing an unchanged generated block must be stable'
    print('selftest: ok')


if __name__ == '__main__':
    if '--selftest' in sys.argv or '--self-test' in sys.argv:
        selftest()
        sys.exit(0)
    repository_index = build_index(_read_sources())
    rows = sweep(index=repository_index)
    if TRUNCATED:
        print(
            'BRACE SCAN LOST SYNC in: %s' % ', '.join(TRUNCATED),
            file=sys.stderr,
        )
        sys.exit(1)
    if '--check' in sys.argv:
        try:
            selftest(repository_index)
            ok = check_inventory(rows)
        except (OSError, ValueError) as error:
            print('reader inventory check failed: %s' % error, file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if ok else 1)
    if '--update-inventory' in sys.argv:
        try:
            update_inventory(rows)
        except (OSError, ValueError) as error:
            print('reader inventory update failed: %s' % error, file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if '--json' in sys.argv:
        json.dump(rows, sys.stdout, indent=1)
        sys.exit(0)
    exp = [r for r in rows if r['exposure']]
    amb = [r for r in rows if r['ambiguous']]
    d = Counter(r['depth'] for r in rows)
    print('internal page-prose readers: %d' % len(rows))
    for depth, title in enumerate(DEPTH_TITLES):
        print('  depth %d (%s): %d' % (depth, title, d[depth]))
    if TRUNCATED:
        print('  BRACE SCAN LOST SYNC in: %s' % ', '.join(TRUNCATED))
    print('  exposure paths (pub + caller outside wenlan-core): %d' % len(exp))
    print('  internal-only: %d' % (len(rows) - len(exp)))
    print('  name-ambiguous (caller edges unresolvable by name): %d' % len(amb))
    for f, c in Counter(r['file'] for r in rows).most_common():
        print('    %-52s %d' % (f, c))
