"""
Cortex Lab — Bulk Personal Data Ingestion
Reads all MD files from raw_data/ and ingests them as memories
through the running FastAPI server's /api/memories/ingest endpoint.

This populates:
  - DuckDB metadata store
  - Vector store (embeddings)
  - Knowledge graph (entities & relationships)

Usage: python scripts/ingest_raw_data.py
"""

import os
import re
import sys
import time
import json
import urllib.request
import urllib.error

API_BASE = "http://localhost:8000"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data")
SESSION_ID = f"bulk-ingest-{int(time.time())}"


def switch_to_gemini():
    """Ensure Gemini is the active provider."""
    data = json.dumps({"provider": "gemini"}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/llm/provider",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"  Provider: {result.get('provider', 'unknown')}")
    except Exception as e:
        print(f"  ⚠ Could not switch provider: {e}")


def ingest_memory(content: str, source: str = "personal_file") -> dict:
    """Send a single memory to the ingestion endpoint."""
    data = json.dumps({
        "content": content,
        "source": source,
        "session_id": SESSION_ID,
    }).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/memories/ingest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def chunk_markdown(text: str, source_file: str) -> list:
    """
    Split a markdown file into meaningful memory chunks.
    Strategy:
      1. Split by H2/H3 sections
      2. If a section is too long (>800 chars), split by paragraphs
      3. Prefix each chunk with context from the section header
      4. Skip very short chunks (<50 chars)
    """
    chunks = []

    # Split by headers (## or ###)
    sections = re.split(r'\n(?=#{2,3}\s)', text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        # Extract header if present
        header_match = re.match(r'^(#{2,3})\s+(.+?)(?:\n|$)', section)
        header = header_match.group(2).strip() if header_match else ""
        body = section[header_match.end():].strip() if header_match else section

        # Clean up markdown artifacts
        body = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', body)  # Remove link URLs
        body = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', body)    # Remove images
        body = body.replace('**', '').replace('*', '')           # Remove bold/italic markers

        if len(body) < 40:
            continue

        # If section is short enough, keep as one chunk
        if len(body) <= 800:
            chunk = f"{header}: {body}" if header else body
            chunks.append(chunk.strip())
        else:
            # Split by paragraphs (double newline) or bullet groups
            paragraphs = re.split(r'\n\n+', body)
            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 20:
                    continue

                if len(current_chunk) + len(para) < 800:
                    current_chunk += "\n" + para if current_chunk else para
                else:
                    if current_chunk and len(current_chunk) >= 50:
                        prefix = f"{header}: " if header else ""
                        chunks.append(f"{prefix}{current_chunk}".strip())
                    current_chunk = para

            # Don't forget the last chunk
            if current_chunk and len(current_chunk) >= 50:
                prefix = f"{header}: " if header else ""
                chunks.append(f"{prefix}{current_chunk}".strip())

    return chunks


def read_and_chunk_file(filepath: str) -> list:
    """Read a markdown file and return memory chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    filename = os.path.basename(filepath)
    chunks = chunk_markdown(text, filename)

    # Add file context prefix to first chunk
    if chunks:
        chunks[0] = f"[Source: {filename}] {chunks[0]}"

    return chunks


def main():
    print("=" * 60)
    print("  Cortex Lab — Bulk Personal Data Ingestion")
    print("=" * 60)

    # Check if server is running
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/health", timeout=5) as resp:
            health = json.loads(resp.read())
            print(f"\n  Server status: {health['status']}")
            print(f"  Gemini available: {health['model_info'].get('gemini_available', False)}")
    except Exception as e:
        print(f"\n  ❌ Server not reachable: {e}")
        print(f"  Start the backend first: cd backend && python -m uvicorn server:app")
        sys.exit(1)

    # Switch to Gemini
    print("\n  Switching to Gemini provider...")
    switch_to_gemini()

    # Find all MD files
    md_files = sorted([
        os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)
        if f.endswith(".md")
    ])
    print(f"\n  Found {len(md_files)} markdown files in raw_data/")

    # Chunk all files
    all_chunks = []
    for filepath in md_files:
        filename = os.path.basename(filepath)
        chunks = read_and_chunk_file(filepath)
        all_chunks.extend([(filename, c) for c in chunks])
        print(f"    {filename}: {len(chunks)} chunks")

    print(f"\n  Total chunks to ingest: {len(all_chunks)}")
    print(f"  Session ID: {SESSION_ID}")
    print()

    # Ingest each chunk
    success = 0
    errors = 0
    t0 = time.time()

    for i, (filename, chunk) in enumerate(all_chunks):
        try:
            result = ingest_memory(chunk, source=f"file:{filename}")
            mem = result.get("memory", {})
            mem_id = mem.get("id", "???")
            mem_type = mem.get("memory_type", "?")
            entities = mem.get("entities", [])[:4]
            topics = mem.get("topics", [])[:3]

            preview = chunk[:60].replace("\n", " ")
            entity_str = ", ".join(entities) if entities else "—"
            topic_str = ", ".join(topics) if topics else "—"

            print(f"  [{i+1}/{len(all_chunks)}] ✓ {mem_id} [{mem_type}] "
                  f"entities=[{entity_str}] topics=[{topic_str}]")
            print(f"           {preview}...")
            success += 1

            # Small delay to avoid overwhelming the API
            time.sleep(0.3)

        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"  [{i+1}/{len(all_chunks)}] ✗ HTTP {e.code}: {body[:100]}")
            errors += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  [{i+1}/{len(all_chunks)}] ✗ {e}")
            errors += 1
            time.sleep(0.5)

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Ingestion Complete!")
    print(f"  ✓ {success} memories ingested")
    if errors:
        print(f"  ✗ {errors} errors")
    print(f"  ⏱ {elapsed:.1f}s total ({elapsed/max(success,1):.1f}s per memory)")
    print("=" * 60)

    # Get final stats
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/rag/stats", timeout=10) as resp:
            stats = json.loads(resp.read())
            mem_stats = stats.get("memories", {})
            vec_stats = stats.get("vectors", {})
            graph_stats = stats.get("graph", {})
            print(f"\n  📊 Final RAG Stats:")
            print(f"     Memories: {mem_stats.get('memories', 0)}")
            print(f"     Vectors: {vec_stats.get('total_vectors', 0)}")
            print(f"     Graph nodes: {graph_stats.get('nodes', 0)}")
            print(f"     Graph edges: {graph_stats.get('edges', 0)}")
    except Exception:
        pass

    print()


if __name__ == "__main__":
    main()
