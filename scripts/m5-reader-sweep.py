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
  skipping *_test.rs files and #[cfg(test)] blocks (brace-tracked, never
  first-match truncation).

Both halves must hold INSIDE THE SAME string literal. An earlier version tested
(a) and (b) anywhere in the function body, which counted `list_tags_scoped`
(selects tags, merely checks page existence) and `tally` (no page SQL at all).

Function bodies are brace-matched. An earlier version delimited a body by the
next `fn` match, which merged adjacent functions and inflated the set.

EXPOSURE: a reader is an exposure path iff it is `pub` (unqualified — not
`pub(crate)` / `pub(super)`) AND called from outside wenlan-core. Caller edges
are resolved by NAME, which over-matches on generic names; rows whose name is
ambiguous are flagged rather than given a caller list. PR-B replaces this with
language-server resolution.

Usage: python3 scripts/m5-reader-sweep.py [--json]
"""
import os, re, sys, json

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
        j, depth, started = i, 0, False
        while j < n:
            for ch in lines[j]:
                if ch == '{': depth += 1; started = True
                elif ch == '}': depth -= 1
            if started and depth <= 0: break
            j += 1
            if j - i > 200: break
        vis = (m.group(2) or '').strip() or 'private'
        yield (i + 1, m.group(3), vis, '\n'.join(lines[i:j + 1]))
        i = max(j + 1, i + 1)


def rust_files():
    for base in CRATES:
        for root, _, fs in os.walk(base):
            for f in fs:
                if f.endswith('.rs') and not f.endswith('_test.rs'):
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
    return readers


if __name__ == '__main__':
    rows = sweep()
    if '--json' in sys.argv:
        json.dump(rows, sys.stdout, indent=1)
        sys.exit(0)
    exp = [r for r in rows if r['exposure']]
    amb = [r for r in rows if r['ambiguous']]
    print('internal page-prose readers: %d' % len(rows))
    print('  exposure paths (pub + caller outside wenlan-core): %d' % len(exp))
    print('  internal-only: %d' % (len(rows) - len(exp)))
    print('  name-ambiguous (caller edges unresolvable by name): %d' % len(amb))
    from collections import Counter
    for f, c in Counter(r['file'] for r in rows).most_common():
        print('    %-52s %d' % (f, c))
