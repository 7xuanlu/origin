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

EXPOSURE: a reader is an exposure path iff it is `pub` (unqualified — not
`pub(crate)` / `pub(super)`) AND called from outside wenlan-core. Caller edges
are resolved by NAME, which over-matches on generic names; rows whose name is
ambiguous are flagged rather than given a caller list. PR-B replaces this with
language-server resolution.

Usage: python3 scripts/m5-reader-sweep.py [--json]
"""
import os, re, sys, json

MAX_FN_LINES = 12000
TRUNCATED = []

FN = re.compile(
    r'^(\s*)(pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+(\w+)'
)
PROSE = re.compile(r'\b(content|title|summary|excerpt|body|snippet)\b')
SQLBLK = re.compile(r'"[^"]*(?:FROM|JOIN)\s+pages\b[^"]*"', re.I | re.S)
CRATES = ['crates/wenlan-core/src', 'crates/wenlan-server/src',
          'crates/wenlan-cli/src', 'crates/wenlan-mcp/src']
CORE = 'crates/wenlan-core/src'


def strip_test(lines):
    """Blank out #[cfg(test)] blocks by brace tracking, preserving line numbers."""
    out, skip, depth, armed = [], 0, 0, False
    for l in lines:
        if not skip and re.match(r'\s*#\[cfg\(test\)\]', l):
            armed = True; out.append(''); continue
        if armed and not skip:
            if '{' in l:
                skip, depth, armed = 1, l.count('{') - l.count('}'), False
                out.append('')
                if depth <= 0: skip = 0
                continue
            out.append(''); continue
        if skip:
            depth += l.count('{') - l.count('}')
            out.append('')
            if depth <= 0: skip = 0
            continue
        out.append(l)
    return out


def bodies(lines):
    """Yield (line_no, name, visibility, body) with brace-matched bodies."""
    i, n = 0, len(lines)
    while i < n:
        m = FN.match(lines[i])
        if not m:
            i += 1; continue
        name_hint = m.group(3)
        j, depth, started = i, 0, False
        while j < n:
            for ch in lines[j]:
                if ch == '{': depth += 1; started = True
                elif ch == '}': depth -= 1
            if started and depth <= 0: break
            j += 1
            if j - i > MAX_FN_LINES:
                # The longest real function in this tree is db.rs::run_migrations
                # at 8819 lines. Past the cap the brace scan has lost sync (an
                # unbalanced brace inside a string or macro), so report it rather
                # than silently truncating a body.
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
    readers, texts = [], {}
    for p in rust_files():
        lines = strip_test(open(p, errors='ignore').read().split('\n'))
        texts[p] = '\n'.join(lines)
        for ln, name, vis, body in bodies(lines):
            sqls = SQLBLK.findall(body)
            if sqls and any(PROSE.search(q) for q in sqls):
                readers.append({'file': p, 'line': ln, 'fn': name, 'vis': vis})

    # A name is ambiguous when more than one distinct function in the tree
    # declares it; a name-keyed caller scan cannot attribute edges then.
    declared = {}
    for p, t in texts.items():
        for m in re.finditer(r'\bfn\s+(\w+)', t):
            declared.setdefault(m.group(1), set()).add(p + ':' + str(t[:m.start()].count('\n') + 1))

    for r in readers:
        r['ambiguous'] = len(declared.get(r['fn'], ())) > 1
        ext = []
        if not r['ambiguous']:
            pat = re.compile(r'\b' + re.escape(r['fn']) + r'\s*\(')
            for p, t in texts.items():
                if p.startswith(CORE): continue
                for m in pat.finditer(t):
                    ext.append('%s:%d' % (p, t[:m.start()].count('\n') + 1))
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
                                         'vis': vis, 'body': body})
    known = {(r['file'], r['line']) for r in readers}
    frontier = {r['fn'] for r in readers}
    out = list(readers)
    for depth in (1, 2):
        pats = [(n, re.compile(r'\b' + re.escape(n) + r'\s*\(')) for n in frontier
                if len(declared.get(n, ())) == 1]
        nxt = set()
        for key, f in all_fns.items():
            if key in known: continue
            for n, pat in pats:
                if pat.search(f['body']):
                    f = dict(f); f.pop('body', None)
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
            for p, t in texts.items():
                if p.startswith(CORE): continue
                for m in pat.finditer(t):
                    ext.append('%s:%d' % (p, t[:m.start()].count('\n') + 1))
            r['ext'] = sorted(set(ext))
            r['exposure'] = (r['vis'] == 'pub') and bool(r['ext'])
    out.sort(key=lambda r: (r['depth'], r['file'], r['line']))
    return out


if __name__ == '__main__':
    rows = sweep()
    if '--json' in sys.argv:
        json.dump(rows, sys.stdout, indent=1)
        sys.exit(0)
    exp = [r for r in rows if r['exposure']]
    amb = [r for r in rows if r['ambiguous']]
    from collections import Counter
    d = Counter(r['depth'] for r in rows)
    print('internal page-prose readers: %d' % len(rows))
    print('  depth 0 (SQL-bearing definitions): %d' % d[0])
    print('  depth 1 (wrapper layer):           %d' % d[1])
    print('  depth 2 (consumers):               %d' % d[2])
    if TRUNCATED:
        print('  BRACE SCAN LOST SYNC in: %s' % ', '.join(TRUNCATED))
    print('  exposure paths (pub + caller outside wenlan-core): %d' % len(exp))
    print('  internal-only: %d' % (len(rows) - len(exp)))
    print('  name-ambiguous (caller edges unresolvable by name): %d' % len(amb))
    for f, c in Counter(r['file'] for r in rows).most_common():
        print('    %-52s %d' % (f, c))
