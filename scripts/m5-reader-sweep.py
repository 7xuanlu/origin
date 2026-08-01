#!/usr/bin/env python3
"""Enumerate internal page-prose readers for the M5 reader manifest.

The M5 Stage 0 reader manifest (docs/plans/2026-07-27-m5-reader-manifest-inventory.md)
carries a set of internal call sites that read page prose. Three review rounds
produced three different counts because the predicate lived in prose and each
implementation interpreted it differently. The predicate now lives here, and the
number is whatever this script says.

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
find — was absent rather than listed. Raising it to 3 surfaced 75 rows, 28 of
them route handlers. This is a parameter, not a claim that nothing lies past it.

EXPOSURE: a reader is an exposure path iff it is `pub` (unqualified — not
`pub(crate)` / `pub(super)`) AND called from outside wenlan-core. Caller edges
are resolved by NAME, which over-matches on generic names; rows whose name is
ambiguous are flagged rather than given a caller list. PR-B replaces this with
language-server resolution.

Usage:
  python3 scripts/m5-reader-sweep.py [--json]
  python3 scripts/m5-reader-sweep.py --check
  python3 scripts/m5-reader-sweep.py --update-inventory
"""
import difflib
import json
import os
import re
import sys
from collections import Counter

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


def strip_test(lines):
    """Blank out #[cfg(test)] blocks by brace tracking, preserving line numbers."""
    out, skip, depth, armed = [], 0, 0, False
    structure = rust_structure(lines)
    for l, structural in zip(lines, structure):
        if not skip and re.match(r'\s*#\[cfg\(test\)\]', structural):
            armed = True; out.append(''); continue
        if armed and not skip:
            if '{' not in structural and structural.strip().endswith(';'):
                # `#[cfg(test)] mod foo;` / `use ...;` / a test-only const.
                # The attribute governs an item that never opens a brace, so
                # arming the brace scan here would blank forward until the NEXT
                # unrelated `{` and swallow real code -- 45% of db.rs, in the
                # version that shipped. Blank the item, disarm, move on.
                out.append(''); armed = False; continue
            if '{' in structural:
                skip, depth, armed = (
                    1,
                    structural.count('{') - structural.count('}'),
                    False,
                )
                out.append('')
                if depth <= 0: skip = 0
                continue
            out.append(''); continue
        if skip:
            depth += structural.count('{') - structural.count('}')
            out.append('')
            if depth <= 0: skip = 0
            continue
        out.append(l)
    return out


def bodies(lines):
    """Yield (line_no, name, visibility, body) with brace-matched bodies."""
    i, n = 0, len(lines)
    structure = rust_structure(lines)
    while i < n:
        m = FN.match(structure[i])
        if not m:
            i += 1; continue
        name_hint = m.group(3)
        j, depth, started = i, 0, False
        while j < n:
            for ch in structure[j]:
                if ch == '{': depth += 1; started = True
                elif ch == '}': depth -= 1
            if started and depth <= 0: break
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
        yield (i + 1, m.group(3), vis, '\n'.join(lines[i:j + 1]))
        i = max(j + 1, i + 1)


def rust_files():
    for base in CRATES:
        for root, _, fs in os.walk(base):
            for f in fs:
                if f.endswith('.rs') and not f.endswith(('_test.rs', '_tests.rs')):
                    yield os.path.join(root, f)


def sweep():
    readers, texts, code_texts = [], {}, {}
    for p in rust_files():
        lines = strip_test(open(p, errors='ignore').read().split('\n'))
        texts[p] = '\n'.join(lines)
        code_texts[p] = '\n'.join(rust_structure(lines))
        for ln, name, vis, body in bodies(lines):
            sqls = SQLBLK.findall(body)
            if sqls and any(PROSE.search(q) for q in sqls):
                readers.append({'file': p, 'line': ln, 'fn': name, 'vis': vis})

    # A name is ambiguous when more than one distinct function in the tree
    # declares it; a name-keyed caller scan cannot attribute edges then.
    declared = {}
    for p, t in code_texts.items():
        for m in re.finditer(r'\bfn\s+(\w+)', t):
            declared.setdefault(m.group(1), set()).add(p + ':' + str(t[:m.start()].count('\n') + 1))

    for r in readers:
        r['ambiguous'] = len(declared.get(r['fn'], ())) > 1
        ext = []
        if not r['ambiguous']:
            pat = re.compile(r'\b' + re.escape(r['fn']) + r'\s*\(')
            for p, t in code_texts.items():
                if p.startswith(CORE): continue
                for m in pat.finditer(t):
                    ln = t[:m.start()].count('\n') + 1
                    if p == r['file'] and ln == r['line']: continue
                    ext.append('%s:%d' % (p, ln))
        r['ext'] = sorted(set(ext))
        r['exposure'] = (r['vis'] == 'pub') and bool(r['ext'])
    readers.sort(key=lambda r: (r['file'], r['line']))
    for r in readers:
        r['depth'] = 0

    # Transitive caller closure. Depth 1 is the wrapper layer (get_page over
    # get_page_inner), depth 2+ the consumers the governing spec asks for.
    all_fns = {}
    for p, t in texts.items():
        for ln, name, vis, body in bodies(t.split('\n')):
            all_fns.setdefault((p, ln), {'file': p, 'line': ln, 'fn': name,
                                         'vis': vis, 'body': body,
                                         'code': '\n'.join(
                                             rust_structure(body.split('\n'))
                                         )})
    known = {(r['file'], r['line']) for r in readers}
    frontier = {r['fn'] for r in readers}
    out = list(readers)
    for depth in range(1, MAX_DEPTH + 1):
        pats = [(n, re.compile(r'\b' + re.escape(n) + r'\s*\(')) for n in sorted(frontier)
                if len(declared.get(n, ())) == 1]
        nxt = set()
        for key, f in all_fns.items():
            if key in known: continue
            for n, pat in pats:
                if pat.search(f['code']):
                    f = dict(f)
                    f.pop('body', None)
                    f.pop('code', None)
                    f['depth'] = depth; f['via'] = n
                    f['ambiguous'] = len(declared.get(f['fn'], ())) > 1
                    f['ext'] = []
                    f['exposure'] = False
                    out.append(f); known.add(key); nxt.add(f['fn'])
                    break
        frontier = nxt
    for r in out:
        if r['depth'] and not r['ambiguous']:
            pat = re.compile(r'\b' + re.escape(r['fn']) + r'\s*\(')
            ext = []
            for p, t in code_texts.items():
                if p.startswith(CORE): continue
                for m in pat.finditer(t):
                    ln = t[:m.start()].count('\n') + 1
                    if p == r['file'] and ln == r['line']: continue
                    ext.append('%s:%d' % (p, ln))
            r['ext'] = sorted(set(ext))
            r['exposure'] = (r['vis'] == 'pub') and bool(r['ext'])
    out.sort(key=lambda r: (r['depth'], r['file'], r['line']))
    return out


def short_path(path):
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


def render_inventory(rows):
    """Render the canonical, generated reader rows embedded in the M5 inventory."""
    sections = [INVENTORY_BEGIN]
    for depth, title in enumerate(DEPTH_TITLES):
        last_column = 'Exposure' if depth == 0 else 'Reaches prose via'
        sections.extend([
            '',
            '### Depth %d — %s' % (depth, title),
            '',
            '| Address | Function | Visibility | %s |' % last_column,
            '|---|---|---|---|',
        ])
        for row in (r for r in rows if r['depth'] == depth):
            address = '%s:%d' % (short_path(row['file']), row['line'])
            if depth == 0:
                if row['ambiguous']:
                    tail = 'name-ambiguous'
                elif row['exposure']:
                    callers = ', '.join('`%s`' % short_path(c) for c in row['ext'])
                    tail = '**exposure** — ' + callers
                else:
                    tail = 'internal-only'
            else:
                tail = '`%s`' % row['via']
            sections.append(
                '| `%s` | `%s` | `%s` | %s |'
                % (address, row['fn'], row['vis'], tail)
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
    document = open(INVENTORY, encoding='utf-8').read()
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
    document = open(INVENTORY, encoding='utf-8').read()
    updated = replace_inventory_block(document, generated)
    with open(INVENTORY, 'w', encoding='utf-8') as f:
        f.write(updated)
    print('updated %s (%d rows)' % (INVENTORY, len(rows)))


def selftest():
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

    db_lines = strip_test(open('crates/wenlan-core/src/db.rs').read().split('\n'))
    db_bodies = {name: body for _, name, _, body in bodies(db_lines)}
    assert 'migrate_80_page_scope_fold' in db_bodies, \
        'run_migrations swallowed the next sibling method'
    assert 'fn migrate_80_page_scope_fold' not in db_bodies['run_migrations'], \
        'run_migrations body overran its LSP-resolved end'

    # A definition is not a call site of itself.
    for r in sweep():
        assert '%s:%d' % (r['file'], r['line']) not in r['ext'], \
            '%s counts its own definition as an external caller' % r['fn']

    # Positive control for exact set equality: either an added or a missing
    # reader must change the generated block. This is deliberately independent
    # of the current inventory so a stale snapshot cannot make the self-test
    # agree with itself.
    fixture_rows = [
        {
            'file': 'crates/wenlan-core/src/db.rs',
            'line': 10,
            'fn': 'read_one',
            'vis': 'pub',
            'depth': 0,
            'ambiguous': False,
            'exposure': False,
            'ext': [],
        },
        {
            'file': 'crates/wenlan-server/src/routes.rs',
            'line': 20,
            'fn': 'handle_one',
            'vis': 'pub',
            'depth': 1,
            'via': 'read_one',
            'ambiguous': False,
            'exposure': False,
            'ext': [],
        },
    ]
    fixture = 'before\n%s\nafter\n' % render_inventory(fixture_rows)
    expected = inventory_block(fixture)
    added = fixture_rows + [
        dict(fixture_rows[1], line=21, fn='handle_two')
    ]
    assert expected != render_inventory(added), 'an added reader must fail exact equality'
    assert expected != render_inventory(fixture_rows[:-1]), \
        'a missing reader must fail exact equality'
    assert replace_inventory_block(fixture, expected) == fixture, \
        'replacing an unchanged generated block must be stable'
    print('selftest: ok')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
        sys.exit(0)
    rows = sweep()
    if TRUNCATED:
        print(
            'BRACE SCAN LOST SYNC in: %s' % ', '.join(TRUNCATED),
            file=sys.stderr,
        )
        sys.exit(1)
    if '--check' in sys.argv:
        try:
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
