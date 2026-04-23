// SPDX-License-Identifier: AGPL-3.0-only
//! Compiled-in default prompt strings for the intelligence pipeline.
//! These are the open-source defaults — proprietary overrides are loaded from files at runtime.

pub(crate) const CLASSIFY_MEMORY: &str = "\
Classify this memory. Respond with ONLY valid JSON:\n\
{\"memory_type\": \"...\", \"domain\": \"...\", \"tags\": [\"...\", \"...\"]}\n\n\
memory_type must be one of: identity, preference, decision, fact, goal\n\
- decision: a choice was made between alternatives, or a direction was chosen with rationale (e.g. \"switched from X to Y because...\", \"chose to use X over Y\")\n\
- fact: objective knowledge without a choice (e.g. \"X supports feature Y\", \"the API returns JSON\")\n\
domain is a short topic label (1-3 words, lowercase)\n\
tags are 2-4 semantic keywords (lowercase)";

pub(crate) const CLASSIFY_MEMORY_QUALITY: &str = "\
Classify this memory. Respond with ONLY valid JSON:\n\
{\"memory_type\": \"...\", \"domain\": \"...\", \"tags\": [\"...\", \"...\"], \"quality\": \"...\"}\n\n\
memory_type must be one of: identity, preference, decision, fact, goal\n\
- decision: a choice was made between alternatives, or a direction was chosen with rationale (e.g. \"switched from X to Y because...\", \"chose to use X over Y\")\n\
- fact: objective knowledge without a choice (e.g. \"X supports feature Y\", \"the API returns JSON\")\n\
domain is a short topic label (1-3 words, lowercase)\n\
tags are 2-4 semantic keywords (lowercase)\n\
quality is low (vague/trivial), medium (useful), or high (specific+actionable)";

pub(crate) const CLASSIFY_MEMORY_QUALITY_STRICT: &str = "\
Classify this memory. Respond with ONLY valid JSON:\n\
{\"memory_type\": \"...\", \"domain\": \"...\", \"tags\": [\"...\", \"...\"], \"quality\": \"...\"}\n\n\
memory_type must be one of: identity, preference, decision, fact, goal\n\
- decision: a choice was made between alternatives, or a direction was chosen with rationale (e.g. \"switched from X to Y because...\", \"chose to use X over Y\")\n\
- fact: objective knowledge without a choice (e.g. \"X supports feature Y\", \"the API returns JSON\")\n\
domain is a short topic label (1-3 words, lowercase)\n\
tags are 2-4 semantic keywords (lowercase)\n\
quality must be one of: low, medium, high (how specific and actionable is this memory?)";

// Used by llm_formatter in the app crate; referenced again once llm_formatter
// moves into origin-core in a later phase.
#[allow(dead_code)]
pub(crate) const CLASSIFY_PROFILE_SUBTYPE: &str = "\
Classify this profile memory into one of exactly 3 types. Respond with ONLY the type name.\n\n\
identity — who the user is (role, background, expertise, values)\n\
preference — how the user likes things done (tools, workflow, style)\n\
goal — what the user is working toward (objectives, targets, deadlines)\n\n\
Respond with one word: identity, preference, or goal";

pub(crate) const CLASSIFY_SCREEN: &str = "\
You classify screen capture text from a desktop application.\n\
Classify the content into exactly one space from: [{spaces_str}].\n\
Provide 2-4 semantic tags (lowercase single words or short phrases) describing the content.\n\
Optionally provide a short stream_name describing the work session (e.g. \"debugging auth flow\").\n\
Respond with ONLY valid JSON: {\"summary\": \"...\", \"space\": \"...\", \"tags\": [\"...\"], \"stream_name\": \"...\"}\n\
IMPORTANT: Inside JSON strings, escape newlines as \\n and quotes as \\\".\n\
The summary should be 1-2 sentences describing what the user was doing.";

pub(crate) const MERGE_MEMORIES: &str = "\
Combine these notes into one clean paragraph that states the key facts directly.\n\
\n\
Rules: 2-4 sentences. State facts directly — never start with 'The most recent memory' or 'This memory' or any meta-commentary. Write fresh — do not copy input sentences. If notes contradict, keep the most recent. Stop after the paragraph. No labels, headers, or multiple drafts.";

pub(crate) const DETECT_CONTRADICTION: &str = "\
Compare two memories. Respond with exactly one of:\n\
- CONSISTENT (if they agree or are unrelated)\n\
- CONTRADICTS: <brief explanation>\n\
- SUPERSEDES: <merged version combining both>";

pub(crate) const SUMMARIZE_DECISIONS: &str = "\
You summarize a set of decisions made by one person.\n\
State the key decisions as one concise sentence. If no coherent theme, respond: null";

pub(crate) const DETECT_PATTERN: &str = "\
You analyze memories belonging to one person in the domain '{domain_hint}'.\n\
Determine if they reveal a pattern: a preference, habit, identity trait, or recurring decision.\n\
If yes, state it as one concise sentence. If no clear pattern, respond with exactly: null";

pub(crate) const NARRATIVE: &str = "\
Write a 3-5 sentence portrait of this person in second person. \
Make it read like a colleague describing them — flow naturally between topics. \
Be specific, not generic. Do not list items. Do not number things. \
Just write a smooth paragraph.";

pub(crate) const BRIEFING_TOPIC: &str = "\
Write one casual sentence summarizing what this person has been doing. \
Use \"you\" (second person). Be specific — mention the actual topics. \
Keep it under 25 words. Do not list items. Do not repeat the input.";

pub(crate) const RERANK_RESULTS: &str = "\
Rate each result's relevance to the query on a scale of 0-10.\n\
Output ONLY a JSON array of integer scores, e.g. [8, 3, 7].";

pub(crate) const SUMMARIZE_ACTIVITY_SYSTEM: &str = "\
You summarize user activity logs into JSON. Always respond with exactly one JSON object, no markdown.";

pub(crate) const SUMMARIZE_ACTIVITY_USER: &str = "\
Summarize this activity session in 1-2 sentences and give 3-5 topic tags.\n\n\
Apps: {apps}\n\nLog:\n{log}\n\n\
Respond ONLY with JSON: {\"summary\": \"...\", \"tags\": [\"...\"]}";

pub(crate) const BATCH_CLASSIFY: &str = "\
Classify each memory. Return a JSON array.\n\
For each: {\"i\": <number>, \"type\": \"<identity|preference|fact|decision|goal>\", \"domain\": \"<work|personal|health|finance|technology|travel|food>\", \"tags\": [\"<tag>\"]}";

pub(crate) const EXTRACT_KNOWLEDGE_GRAPH: &str = "\
Extract entities and facts from each memory. Return a JSON array.\n\
For each: {\"i\": <number>, \"entities\": [{\"name\": \"...\", \"type\": \"person|project|technology|organization|place|concept\"}], \"observations\": [{\"entity\": \"...\", \"content\": \"...\"}], \"relations\": [{\"from\": \"...\", \"to\": \"...\", \"type\": \"...\"}]}\n\
Include \"user\" (person) when the memory is about the user.";

pub(crate) const EXTRACT_STRUCTURED_FIELDS: &str = "\
Extract structured fields from this {memory_type} memory. Respond with ONLY valid JSON:\n\
{{{fields_json},\n  \"retrieval_cue\": \"a question this memory answers\"\n}}\n\n\
Required fields: {required}\n\
Optional fields (include if inferable, omit if not): {optional}\n\
retrieval_cue: a natural question someone would ask to find this memory later\n\n\
Keep values concise. If a field can't be inferred, omit it.";

pub(crate) const CORRECT_MEMORY: &str = "\
You are correcting a memory based on user feedback. The user says something is wrong with the \
original memory and has described what should change.\n\n\
Original memory:\n\
{original}\n\n\
User's correction:\n\
{correction}\n\n\
Write the corrected memory. Keep the same style and length as the original. Only change what the \
user asked to fix. Respond with ONLY the corrected text, no explanation.";

pub(crate) const DISTILL_CONCEPT: &str = "\
Compile these memories into a wiki-style knowledge page.\n\
\n\
Format:\n\
Do NOT start with a title heading (# Title) -- the title is displayed separately by the UI.\n\
Start directly with a one-sentence TLDR summary.\n\
\n\
Then write the body organized with short topical headers (## Header) and prose paragraphs under each, \
like a Wikipedia article with sections. \
Weave in specific facts (names, numbers, versions) naturally. \
Use [[Topic Name]] wikilinks when referencing related topics. \
Use bullet lists only for genuinely enumerable things (steps, lists of tools, etc.).\n\
\n\
## Open Questions\n\
- List gaps, uncertainties, or contradictions between sources.\n\
\n\
## Sources\n\
- Attribute key claims to source memories (include memory ID).\n\
\n\
Rules:\n\
- Write prose with topical headers. Paragraphs that synthesize, with bullets only for lists.\n\
- Read like an encyclopedia entry — concise, informative, no filler or meta-commentary.\n\
- Preserve specifics — don't generalize away details like exact names, versions, or numbers.\n\
- If sources contradict, keep the most recent and note the contradiction in Open Questions.\n\
- 3-5 paragraphs total. Quality over quantity.";

pub(crate) const UPDATE_CONCEPT: &str = "\
You maintain a wiki-style knowledge page. Update it with new information.\n\
Integrate new facts into the existing prose naturally — don't just append bullets.\n\
If the new information contradicts existing content, note it in Open Questions.\n\
Do not remove existing content unless it is explicitly superseded.\n\
Do NOT include a title heading (# Title) -- the title is displayed separately by the UI.\n\
Output the complete updated page in the same format (TLDR, prose paragraphs, Open Questions, Sources).";

pub(crate) const ASSIGN_ORPHANS: &str = r#"You are a knowledge organization assistant. Given a list of unassigned memories and existing concepts, for each memory:
1. If it clearly belongs to an existing concept, assign it (return the concept_id)
2. If 3+ unassigned memories share a theme not covered by existing concepts, propose a new concept (return a title and the memory indices)
3. Skip memories that are too isolated to group

Return a JSON object with two arrays:
- "assignments": [{"memory_index": 0, "concept_id": "existing_concept_id"}]
- "proposals": [{"title": "Proposed Concept Title", "memory_indices": [1, 3, 5]}]

Only return valid JSON. No explanation text."#;

pub(crate) const GLOBAL_CONCEPT_REVIEW: &str = r#"You are reviewing a knowledge base for organization quality. Given all concept titles and summaries, identify:
1. Concepts that should merge (overlapping topics) — return pairs of concept_ids
2. Cross-cutting themes missing — return proposed titles with related concept_ids
3. Concepts that should split (too broad) — return concept_id with proposed sub-titles

Return a JSON object:
- "merges": [{"keep": "concept_id_1", "remove": "concept_id_2", "reason": "..."}]
- "missing": [{"title": "...", "related_concepts": ["id1", "id2"]}]
- "splits": [{"concept_id": "...", "sub_titles": ["...", "..."]}]

Be conservative. Only suggest changes with high confidence. Return empty arrays if nothing needs changing."#;

pub(crate) const REFINE_CLUSTERS: &str = r#"You are organizing memory clusters for wiki compilation. Each cluster will become a separate concept page.

Given clusters for an entity, decide for each:
- KEEP: cluster is a coherent topic, compile as-is
- MERGE [i,j]: clusters i and j should be one concept (same topic from different angles)
- SPLIT [i]: cluster i covers two distinct topics — provide two sub-topic titles
- RENAME [i]: better title for cluster i

Return a JSON array of actions, one per line:
[
  {"action": "keep", "cluster": 0},
  {"action": "merge", "clusters": [1, 3], "title": "Combined Topic"},
  {"action": "split", "cluster": 2, "titles": ["Sub-topic A", "Sub-topic B"]},
  {"action": "rename", "cluster": 4, "title": "Better Name"}
]

Rules:
- Default to KEEP unless you're confident about a change
- MERGE when two clusters are clearly the same topic from different angles
- SPLIT when a cluster mixes unrelated topics (e.g., licensing + architecture)
- Only return valid JSON"#;

pub(crate) const FORMAT_OCR_TEXT: &str = "\
You restructure raw OCR text from a single application window into clean, well-organized markdown.\n\
Rules:\n\
1. STRUCTURE: Add ### subsection headers to group related content. Each ### section should be a \
coherent topic (e.g., ### Chat History, ### Navigation, ### Email Preview, ### Code Editor). \
These headers will be used as chunk boundaries for semantic search — choose them carefully.\n\
2. CLEANUP: Fix broken words, join fragments that belong to the same sentence/paragraph. \
Remove OCR noise (garbled characters, isolated symbols, repeated lines) but preserve all \
meaningful content.\n\
3. FORMAT: Use code blocks for code, - for bullets, 1. for numbered lists. \
Leave one blank line before and after each header. Leave one blank line between paragraphs.\n\
4. CONTEXT: Sidebar/navigation items should be kept as a concise bullet list under a ### Navigation \
or ### Sidebar header. Do NOT discard them.\n\
5. BREVITY: Do not add commentary. Do not explain what you did. Output only the cleaned text.\n\
Also classify the content into exactly one space from: [{spaces}].\n\
Provide 2-4 semantic tags (lowercase single words or short phrases) describing the content.\n\
Optionally provide a short stream_name describing the work session (e.g. \"debugging auth flow\").\n\
Respond with ONLY valid JSON: {\"formatted_text\": \"...\", \"summary\": \"...\", \"space\": \"...\", \"tags\": [\"...\"], \"stream_name\": \"...\"}\n\
IMPORTANT: Inside JSON strings, escape newlines as \\n and quotes as \\\".\n\
The summary should be 1-2 sentences describing what the user was doing.";
