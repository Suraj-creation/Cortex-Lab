#!/usr/bin/env python3
"""
Cortex Lab — Vector Store Reindex Script
==========================================
Fixes the critical bug where DuckDB has 130+ memories but FAISS only has ~15 vectors.
This script:
1. Reads all memories from DuckDB
2. Filters out user queries (chat-sourced questions) that were incorrectly ingested
3. Embeds all real memories
4. Rebuilds the FAISS index
5. Saves the new vector state

Run: cd backend && python3 reindex_vectors.py

IMPORTANT: Stop the backend server before running this script (DuckDB lock).
"""

import sys
import os
import time
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 60)
    print("  Cortex Lab — Vector Store Reindex")
    print("=" * 60)

    # ── 1. Load embedding model ─────────────────────────────────────────
    print("\n[1/5] Loading embedding model (BGE-large)...")
    t0 = time.time()
    from src.models.embeddings import EmbeddingModel
    embedding_model = EmbeddingModel()
    print(f"  ✓ Embedding model loaded in {time.time()-t0:.1f}s")
    print(f"    Dimension: {embedding_model.dimension}, Device: {embedding_model.device}")

    # ── 2. Connect to DuckDB and load memories ─────────────────────────
    print("\n[2/5] Loading memories from DuckDB...")
    import duckdb
    db_path = os.path.join(os.path.dirname(__file__), "data", "cortex.duckdb")
    conn = duckdb.connect(db_path)

    # Get total counts
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    chat_queries = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE source='chat'"
    ).fetchone()[0]
    print(f"  Total memories: {total}")
    print(f"  Chat-sourced (potential queries): {chat_queries}")

    # Load all memories with their IDs and content
    rows = conn.execute(
        "SELECT id, content, source, memory_type, timestamp FROM memories ORDER BY timestamp"
    ).fetchall()

    # ── 3. Filter out query contamination ────────────────────────────────
    print("\n[3/5] Filtering out query contamination...")

    question_starters = (
        "what ", "what's ", "who ", "who's ", "where ", "where's ",
        "when ", "when's ", "how ", "how's ", "why ", "why's ",
        "which ", "whose ", "whom ",
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "will ", "would ", "should ", "shall ",
        "have ", "has ", "had ",
        "tell me", "list ", "describe ", "summarize ", "explain ",
        "show me", "give me", "find ", "search ",
    )

    real_memories = []
    query_memories = []  # Will be deleted
    for row in rows:
        mid, content, source, mtype, ts = row
        content_lower = content.strip().lower()

        is_query = False
        if source == "chat":
            # Check if it's a question (retrieval query)
            is_question = (
                content_lower.endswith("?")
                or any(content_lower.startswith(q) for q in question_starters)
            )
            if is_question and len(content.strip()) < 200:
                is_query = True

        if is_query:
            query_memories.append((mid, content[:80]))
        else:
            real_memories.append((mid, content, ts))

    print(f"  Real memories to index: {len(real_memories)}")
    print(f"  Query contamination to remove: {len(query_memories)}")

    if query_memories:
        print(f"\n  Contaminated queries (will be deleted from DuckDB):")
        for mid, preview in query_memories[:10]:
            print(f"    ❌ {preview}")
        if len(query_memories) > 10:
            print(f"    ... and {len(query_memories) - 10} more")

    # ── 3b. Delete contaminated queries from DuckDB ─────────────────────
    if query_memories:
        query_ids = [q[0] for q in query_memories]
        print(f"\n  Deleting {len(query_ids)} contaminated queries from DuckDB...")
        for qid in query_ids:
            conn.execute("DELETE FROM memories WHERE id = ?", [qid])
            # Also clean junction tables
            try:
                conn.execute("DELETE FROM memory_topics WHERE memory_id = ?", [qid])
            except Exception:
                pass
            try:
                conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", [qid])
            except Exception:
                pass
        print(f"  ✓ Deleted {len(query_ids)} contaminated memories")

    # ── 4. Embed all real memories ───────────────────────────────────────
    print(f"\n[4/5] Embedding {len(real_memories)} memories...")
    t0 = time.time()

    from src.storage.vector_store import VectorStore
    vector_dir = os.path.join(os.path.dirname(__file__), "data", "vectors")
    vector_store = VectorStore(dimension=embedding_model.dimension, data_dir=vector_dir)

    # Clear existing vectors
    vector_store.vectors.clear()
    vector_store.timestamps.clear()
    vector_store.hot_ids.clear()
    vector_store.warm_ids.clear()
    vector_store.cold_ids.clear()

    # Re-initialize FAISS index (fresh)
    vector_store._init_indices()

    # Batch embed for efficiency
    batch_size = 32
    total_embedded = 0
    for i in range(0, len(real_memories), batch_size):
        batch = real_memories[i:i + batch_size]
        contents = [content for _, content, _ in batch]

        # Embed batch
        embeddings = embedding_model.embed_batch(contents)

        for j, (mid, content, ts) in enumerate(batch):
            from datetime import datetime
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts)
                except Exception:
                    timestamp = datetime.now()
            else:
                timestamp = ts if ts else datetime.now()

            vector_store.add(mid, embeddings[j], timestamp)
            total_embedded += 1

        pct = (i + len(batch)) / len(real_memories) * 100
        print(f"  📊 Embedded {i + len(batch)}/{len(real_memories)} ({pct:.0f}%)", end="\r")

    elapsed = time.time() - t0
    print(f"\n  ✓ Embedded {total_embedded} memories in {elapsed:.1f}s")

    # ── 5. Save vector state ─────────────────────────────────────────────
    print("\n[5/5] Saving FAISS index and vector state...")
    vector_store.save()
    print(f"  ✓ Saved! Vectors: {vector_store.count()}")

    # ── Verification ──────────────────────────────────────────────────────
    remaining = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()

    print(f"\n{'='*60}")
    print(f"  ✅ Reindex Complete!")
    print(f"  📊 Memories in DuckDB: {remaining}")
    print(f"  📊 Vectors in FAISS: {vector_store.count()}")
    print(f"  📊 Coverage: {vector_store.count()/max(remaining,1):.0%}")
    print(f"  📊 Contamination removed: {len(query_memories)} queries")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
