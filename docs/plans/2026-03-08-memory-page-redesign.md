# Memory Page Redesign: "The Gentle Stream"

## Context

Origin has pivoted from screen capture/focus tracking to being a **personal agent memory layer** — where AI agents write what they learn and humans curate. The current MemoryView is built around a capture-centric model (time-grouped files from screen, clipboard, focus). This redesign reimagines the memory page around the new primitives: profile, preferences, facts, identity, goals, relationships — the same concepts as the official MCP memory server, but with a human-first browsing experience.

## Design Principles

1. **Surfing, not reviewing** — Users browse memories organically, not forced into review queues. Lower cognitive overhead always.
2. **Gradual saturation** — Memories emerge, refine, and consolidate beautifully without info overload. It's a way of life.
3. **Warm humanity** — Differentiate from cold AI-efficiency aesthetics. The feel of Claude's warmth: liquid, minimal, alive, but never clinical.
4. **Silent refinement** — Consolidation happens quietly. You see the latest best version. A gentle "refined" glow lets you notice evolution without demanding attention.
5. **Subtle connections** — Knowledge graph powers navigation via hyperlinks, not graph visualizations. Like Wikipedia, not a node diagram.

## Layout: Two-Panel with Identity Hub

```
┌─────────────────────────────────────────────────────────┐
│  Origin                                    [search] [gear] │
├──────────────┬──────────────────────────────────────────┤
│  Sidebar     │  Main Stream                            │
│  (240px)     │  (remaining width)                      │
│              │                                          │
│  [Identity   │  Domain-grouped memory clusters         │
│   Card]      │  with confidence-based visual weight    │
│              │                                          │
│  Domain      │                                          │
│  Filters     │                                          │
│              │                                          │
│  Soft Stats  │                                          │
├──────────────┴──────────────────────────────────────────┤
│  [ambient status line]                                  │
└─────────────────────────────────────────────────────────┘
```

**Top bar**: App name, hybrid search, settings gear. Minimal.

**Sidebar** (fixed, ~240px):
- Identity card at top (see below)
- Domain filter pills: "Everything" (default), then existing domains
- Soft stats at bottom: total memories, new today, confirmed count

**Main stream** (scrollable):
- Domain-grouped clusters of memory cards
- Default "Everything" shows all domains as collapsible groups
- Selecting a domain in sidebar filters to just that domain

**Status bar** (bottom):
- Single quiet line showing consolidation activity
- Types in letter-by-letter, fades after 5s
- Not persistent — only appears when there's something to say

## Identity Hub

### Sidebar Card (Compact)

The identity card queries the knowledge graph for the self entity. Shows:
- Avatar area (warm gradient circle with initials, or photo)
- Name (largest entity of type "person" with self relation)
- Role/title (auto-derived from highest-confidence observations)
- Rotating quote: cycles through top observations, 0.6s crossfade every 30s
- Trait pills: top 3-4 confirmed preferences/traits

If no identity exists yet (fresh install): warm empty state with pulsing outline — "As agents learn about you, your profile will take shape here."

Clicking the identity card opens the Identity Detail view in the main panel. The sidebar card stays visible and in sync.

### Identity Detail View (Replaces Stream)

Three sections:
- **About**: All observations, editable inline. Confidence shown as dot (filled=high, hollow=low). Click to edit, trash to remove, "+" to add.
- **Connections**: Relations to other entities as navigable links. Click entity name to jump to its detail view.
- **Recent activity**: Quiet changelog — refinements, additions, confirmations. Fades older items.

This same detail view pattern works for **any entity** — click a person, project, or tool and get the same structure.

## Memory Stream

### Domain Clusters

Each domain is a soft-bordered group showing top 3 memories by confidence, with "show N more..." to expand.

### Memory Cards

Each card represents one memory (`source = "memory"` in chunks). Shows:
- Content (first ~2 lines)
- Source agent, confidence dot, status (confirmed / refined Xh ago / new)
- 3px left border colored by confidence (terracotta=high confirmed, amber=unconfirmed, transparent=low)

### Visual Confidence Gradient

No binary review queue. Instead, a natural gradient:
- **Confirmed + high confidence**: Full opacity, solid terracotta dot (●), solid left border
- **Unconfirmed**: Softer opacity, hollow amber dot (◌), amber left border
- **Low confidence**: Further faded, nearly invisible left border
- **Recently refined**: Subtle warm shimmer (box-shadow pulse every 8s, fades over 24h)
- **New**: Soft amber dot, persists until scrolled past (intersection observer)

### Ordering Within Clusters

1. Confirmed + high confidence (the essentials)
2. Recently refined (alive and evolving)
3. New unconfirmed (ambient arrivals)
4. Older unconfirmed low-confidence (fading naturally)

Best stuff always at top. Low-quality memories naturally sink.

### Connections

Entity names mentioned in memories appear as warm-underlined links. Click to navigate to that entity's detail view. Breadcrumb trail: `Everything > Work > Origin project`.

## Domain Scaffolding

Soft scaffolding approach: a few suggested domains to start (Identity, Work, Preferences, Relationships) that agents default to. New domains appear naturally when agents create them. Empty domains show: "Nothing here yet — agents will fill this in."

## Interactions

### Confirming
Single click on hollow dot (◌) → fills (●) with scale animation. No modal. Reversible.

### Editing
Click memory text → inline textarea, auto-focused. Click away or Cmd+Enter to save. Brief "saved" whisper that fades. New embedding computed server-side if content changes significantly.

### Deleting
Hover reveals trash icon. Click → card collapses (height+fade animation). Undo toast for 5 seconds. No confirmation modal for singles.

### Adding Manually
Soft "+" button at top of stream or within domain cluster. Inline card with:
- Content textarea ("What do you want to remember?")
- Domain dropdown (existing + "new domain")
- Auto-set: type=fact, confidence=1.0, confirmed=true

### Search
Hybrid search across all memories. Results replace stream temporarily, grouped by domain. Escape returns to stream.

### Keyboard Shortcuts
- `/` — focus search
- `Esc` — clear search, close editors
- `↑↓` — navigate between cards
- `Enter` — expand / start editing
- `Cmd+Enter` — save edit

## Visual Language

### Color Palette

```
Background:     #FEFCF9  (warm cream)
Surface:        #FFFDF8  (card backgrounds)
Sidebar bg:     #FAF6F0  (aged linen)

Text primary:   #2C2417  (warm near-black)
Text secondary: #8C7E6A  (warm clay gray)
Text tertiary:  #B8AD9E  (timestamps, agent names)

Accent warm:    #C4733B  (terracotta — confirmed, links)
Accent amber:   #D4943A  (unconfirmed, new)
Accent sage:    #7A9B7E  (connections, relations)
Accent glow:    #E8C97A  (refined shimmer)

Border:         #EDE7DD  (warm, barely visible)
```

Warm and light — like afternoon sunlight on paper. Deliberately not dark mode.

### Typography

- **Headings**: Fraunces — soft serif with optical sizing. Warm, humanist.
- **Body**: Instrument Sans — clean, softer geometry than Inter/Roboto.
- **Mono/metadata**: JetBrains Mono at lower opacity — agent names, confidence, timestamps.

### Motion

- **Card entrance**: Fade up + Y translate (0→8px), staggered 50ms/card, 400ms ease-out-cubic
- **Confirm click**: Dot scales 1→1.3→1 with fill transition, 300ms
- **Refined shimmer**: Subtle box-shadow pulse using accent-glow, every 8s, fades over 24h
- **Rotating identity quote**: Crossfade 600ms, every 30s
- **Delete collapse**: Height→0 + opacity→0, 300ms ease-in
- **Page transitions**: Crossfade 200ms (gentle focus shifts, not sliding)

### Card Anatomy

```
┌─ 1px solid border (warm, barely there) ─────────────┐
│                                                      │
│  ● Memory content text, up to two lines              │
│    with generous line-height (1.6)                   │
│                                                      │
│  claude-code  ·  0.95  ·  refined 2h ago             │
│                                                      │
└──────────────────────────────────────────────────────┘
```

3px left border colored by confidence. Trash icon on hover (fades in 150ms).

### Ambient Details

- Sidebar stats: soft count-up animation on value changes
- Status bar: letter-by-letter type-in (30ms/char), fades after 5s
- Domain headers: dotted leader line between name and count
- Empty states: single line in tertiary text

## Data Model Requirements

### Backend (new or modified)

**Knowledge graph retrieval** (not yet implemented):
- `get_entity(id)` — fetch entity with observations and relations
- `list_entities(entity_type?, domain?)` — list/filter entities
- `get_observations(entity_id)` — list observations for entity
- `get_relations(entity_id)` — list relations from/to entity
- `update_observation(id, content)` — edit observation text
- `delete_observation(id)` — remove observation
- `delete_entity(id)` — remove entity (cascades)

**Memory-specific**:
- `list_memories(domain?, memory_type?, confirmed?)` — richer filtering
- `update_memory(source_id, content?, domain?, confirmed?)` — edit memory fields
- `get_memory_stats()` — total, new today, confirmed count

**Self-entity convention**: An entity with `entity_type = "person"` and a relation `relation_type = "self"` pointing to itself (or a special marker) represents the user's identity.

### Frontend (new Tauri commands + invoke wrappers)

All new backend methods need corresponding:
1. Tauri `#[command]` functions in `search.rs`
2. TypeScript `invoke()` wrappers in `lib/tauri.ts`
3. React Query hooks where appropriate

### New Components

- `MemoryPage.tsx` — new top-level page (replaces MemoryView for memory-source content)
- `IdentityCard.tsx` — sidebar identity hub
- `IdentityDetail.tsx` — entity detail view (reusable for any entity)
- `MemoryStream.tsx` — domain-grouped memory list
- `MemoryCard.tsx` — individual memory card with interactions
- `DomainFilter.tsx` — sidebar domain pills
- `MemoryStats.tsx` — sidebar stats
- `AddMemoryForm.tsx` — inline memory creation
- `StatusBar.tsx` — ambient consolidation status

### Migration from MemoryView

The current MemoryView continues to work for non-memory sources (local files, clipboard, webpage captures). The new MemoryPage specifically renders `source = "memory"` content with the new design. Navigation: the default page becomes MemoryPage. A toggle or tab can switch to the "raw captures" view (current MemoryView) for users who still use file indexing/clipboard.
