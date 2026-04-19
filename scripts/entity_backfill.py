#!/usr/bin/env python3
"""Entity backfill via Origin REST API."""
import json
import urllib.request
import sqlite3

DB_PATH = "/Users/lucian/Library/Application Support/origin/memorydb/origin_memory.db"
API = "http://127.0.0.1:7878"

ENTITY_RULES = [
    (["libsql", "libSQL"], "libSQL", "technology", "origin"),
    (["diskann", "DiskANN"], "DiskANN", "technology", "origin"),
    (["fts5", "FTS5"], "FTS5", "technology", "origin"),
    (["tauri 2", "tauri window", "#[tauri::command]", "Tauri"], "Tauri", "technology", "origin"),
    (["qwen3", "Qwen3"], "Qwen3-4B", "technology", "origin"),
    (["fastembed", "FastEmbed"], "FastEmbed", "technology", "origin"),
    (["bge-small", "BGE-Small"], "BGE-Small", "technology", "origin"),
    (["bge-base", "BGE-Base"], "BGE-Base", "technology", "origin"),
    (["gte-base", "GTE-Base"], "GTE-Base", "technology", "origin"),
    (["chromadb", "ChromaDB"], "ChromaDB", "technology", None),
    (["obsidian vault", "Obsidian"], "Obsidian", "technology", None),
    (["mem0", "Mem0"], "Mem0", "organization", "origin"),
    (["zep ", "graphiti", "Graphiti"], "Zep", "organization", "origin"),
    (["letta ", "memgpt", "MemGPT"], "Letta", "organization", "origin"),
    (["nowledge", "Nowledge"], "Nowledge", "organization", "origin"),
    (["mempalace", "MemPalace"], "MemPalace", "project", "origin"),
    (["supermemory", "Supermemory"], "Supermemory", "organization", "origin"),
    (["karpathy"], "Andrej Karpathy", "person", None),
    (["origin-mcp"], "origin-mcp", "project", "origin"),
    (["knowledge graph", "entity graph"], "Knowledge Graph", "concept", "origin"),
    (["quality gate"], "Quality Gate", "concept", "origin"),
    (["refinery sweep", "refinery phase"], "Refinery", "concept", "origin"),
    (["embedding model", "vector embedding", "embedding dimension"], "Embeddings", "concept", "origin"),
    (["concept distill", "distill_concepts"], "Concept Distillation", "concept", "origin"),
    (["entity extraction"], "Entity Extraction", "concept", "origin"),
    (["longmemeval", "LongMemEval"], "LongMemEval", "concept", "origin"),
    (["locomo", "LoCoMo"], "LoCoMo", "concept", "origin"),
    (["H-1B", "h-1b"], "H-1B Visa", "concept", "personal"),
    (["s-corp", "S-Corp"], "S-Corp", "concept", "personal"),
    (["clean room rewrite"], "Clean Room Rewrite", "concept", "origin"),
    (["ambient overlay", "ambient card"], "Ambient Overlay", "concept", "origin"),
    (["icon trigger", "selection trigger"], "Icon Trigger", "concept", "origin"),
    (["contradiction detection"], "Contradiction Detection", "concept", "origin"),
    (["memory decay", "recency boost", "decay sweep"], "Memory Decay", "concept", "origin"),
    (["hybrid search", "RRF", "reciprocal rank"], "Hybrid Search", "concept", "origin"),
    (["post-ingest", "post_ingest"], "Post-Ingest Pipeline", "concept", "origin"),
    (["NDCG", "ndcg"], "NDCG Metric", "concept", "origin"),
    (["MRR", "mrr"], "MRR Metric", "concept", "origin"),
    (["piaget", "bartlett", "schema formation"], "Cognitive Science", "concept", "origin"),
    (["macOS", "macos", "CoreGraphics", "NSScreen"], "macOS", "technology", "origin"),
    (["metal gpu", "Metal GPU"], "Metal GPU", "technology", "origin"),
    (["AGPL", "agpl"], "AGPL License", "concept", "origin"),
    (["turso", "Turso"], "Turso", "technology", "origin"),
]

def api_call(method, path, data=None):
    url = f"{API}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None

def find_entities(content):
    content_lower = content.lower()
    found = []
    seen = set()
    for keywords, name, etype, domain in ENTITY_RULES:
        if name in seen:
            continue
        for kw in keywords:
            if kw.lower() in content_lower:
                found.append((name, etype, domain))
                seen.add(name)
                break
    return found

def main():
    # Read memories from DB (SELECT doesn't trigger DiskANN)
    conn = sqlite3.connect(DB_PATH)
    memories = conn.execute("""
        SELECT source_id, content, domain
        FROM memories 
        WHERE source = 'memory' AND chunk_index = 0 
          AND entity_id IS NULL AND embedding IS NOT NULL
          AND is_recap = 0 AND source_id NOT LIKE 'merged_%'
    """).fetchall()
    conn.close()
    print(f"Unlinked memories: {len(memories)}")
    
    # Cache existing entities
    existing = {}
    # Try to get from DB directly
    conn = sqlite3.connect(DB_PATH)
    for row in conn.execute("SELECT id, name FROM entities"):
        existing[row[1].lower()] = row[0]
    conn.close()
    print(f"Existing entities: {len(existing)}")
    
    linked = 0
    created = 0
    
    for source_id, content, domain in memories:
        entities = find_entities(content)
        if not entities:
            continue
        
        first_eid = None
        for name, etype, edomain in entities:
            name_lower = name.lower()
            if name_lower in existing:
                eid = existing[name_lower]
            else:
                resp = api_call("POST", "/api/memory/entities", {
                    "name": name, "entity_type": etype, "domain": edomain
                })
                if resp and "id" in resp:
                    eid = resp["id"]
                    existing[name_lower] = eid
                    created += 1
                    print(f"  Created entity: {name} ({etype})")
                else:
                    continue
            
            if first_eid is None:
                first_eid = eid
        
        if first_eid:
            resp = api_call("POST", "/api/memory/link-entity", {
                "source_id": source_id, "entity_id": first_eid
            })
            if resp:
                linked += 1
    
    # Verify
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE source = 'memory' AND chunk_index = 0").fetchone()[0]
    with_entity = conn.execute("SELECT COUNT(*) FROM memories WHERE source = 'memory' AND chunk_index = 0 AND entity_id IS NOT NULL").fetchone()[0]
    conn.close()
    
    print(f"\nResults:")
    print(f"  Entities created: {created}")
    print(f"  Memories linked: {linked}")
    print(f"  Coverage: {with_entity}/{total} ({100*with_entity/total:.1f}%)")

if __name__ == "__main__":
    main()
