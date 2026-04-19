#!/usr/bin/env python3
"""
Bulk Personal Data Ingestion Script for Cortex Lab
===================================================

Reads all markdown files from raw_data/, intelligently chunks them,
and ingests each chunk via the /api/memories/ingest endpoint.

Chunking Strategy:
- Splits on markdown headers (##, ###) for natural section boundaries
- Falls back to paragraph splitting for sections without headers
- Enforces max chunk size (~1500 chars) for optimal embedding quality
- Preserves context by prepending file source + section header

Usage:
    # Make sure backend is running on port 8000
    python ingest_personal_data.py

    # Dry run (preview chunks without ingesting):
    python ingest_personal_data.py --dry-run

    # Custom chunk size:
    python ingest_personal_data.py --max-chunk-size 2000
"""

import os
import re
import sys
import time
import argparse
import requests
from typing import Dict, List, Optional, Tuple

# ─── Configuration ───────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
INGEST_ENDPOINT = f"{API_BASE}/api/memories/ingest"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
DEFAULT_MAX_CHUNK_SIZE = 1500  # characters per chunk (optimal for BGE embeddings)
MIN_CHUNK_SIZE = 50  # skip trivially small chunks
SESSION_ID = "personal-data-bulk-ingest"

# File-to-source label mapping for better memory categorization
FILE_SOURCE_MAP = {
    "Suraj Kumar - Resume.md": "resume",
    "Master-Resume.md": "master-resume",
    "Projects_Repository.md": "projects-repository",
    "Educational_paradigm_shift- the Inevitable.md": "vision-education",
    "Stratup_Ideas- the inevitable.md": "startup-ideas",
    "Reimagining and redefining.md": "vision-institute",
    "Reinventing_educational_paradigm_shift- the Inevitable-Abstract_version.md": "sih2025-proposal",
}


def read_file(filepath: str) -> str:
    """Read a file with UTF-8 encoding."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_markdown(text: str, filename: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE) -> List[Tuple[str, str]]:
    """
    Split markdown text into meaningful chunks.
    
    Returns list of (chunk_text, section_header) tuples.
    
    Strategy:
    1. Split by ## or ### headers first
    2. If a section is too large, split by paragraphs (double newline)
    3. If a paragraph is still too large, split by sentences
    4. Prepend source context to each chunk
    """
    chunks = []
    
    # Split by markdown headers (## or ###)
    # Keep the header with its content
    header_pattern = r'(?=^#{1,3}\s+.+$)'
    sections = re.split(header_pattern, text, flags=re.MULTILINE)
    
    for section in sections:
        section = section.strip()
        if not section or len(section) < MIN_CHUNK_SIZE:
            continue
        
        # Extract section header if present
        header_match = re.match(r'^(#{1,3}\s+.+?)$', section, re.MULTILINE)
        header = header_match.group(1).strip().lstrip('#').strip() if header_match else ""
        
        if len(section) <= max_size:
            chunks.append((section, header))
        else:
            # Split large sections by paragraphs
            paragraphs = re.split(r'\n\s*\n', section)
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                if len(current_chunk) + len(para) + 2 <= max_size:
                    current_chunk += ("\n\n" + para if current_chunk else para)
                else:
                    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                        chunks.append((current_chunk, header))
                    
                    # If single paragraph exceeds max_size, split by sentences
                    if len(para) > max_size:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        sent_chunk = ""
                        for sent in sentences:
                            if len(sent_chunk) + len(sent) + 1 <= max_size:
                                sent_chunk += (" " + sent if sent_chunk else sent)
                            else:
                                if sent_chunk and len(sent_chunk) >= MIN_CHUNK_SIZE:
                                    chunks.append((sent_chunk, header))
                                sent_chunk = sent
                        if sent_chunk and len(sent_chunk) >= MIN_CHUNK_SIZE:
                            current_chunk = sent_chunk
                        else:
                            current_chunk = ""
                    else:
                        current_chunk = para
            
            if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append((current_chunk, header))
    
    # If no headers found, treat the whole text as one section and chunk by paragraphs
    if not chunks and len(text.strip()) >= MIN_CHUNK_SIZE:
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append((current_chunk, ""))
                current_chunk = para
        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append((current_chunk, ""))
    
    return chunks


def add_source_context(chunk_text: str, filename: str, section_header: str) -> str:
    """
    Prepend source context for better retrieval relevance.
    This helps the RAG system understand where this information came from.
    """
    source_label = FILE_SOURCE_MAP.get(filename, filename)
    context_parts = [f"[Source: {source_label}]"]
    if section_header:
        context_parts.append(f"[Section: {section_header}]")
    context = " ".join(context_parts)
    return f"{context}\n{chunk_text}"


def get_system_stats(timeout: int = 5) -> Optional[Dict[str, object]]:
    """Fetch normalized RAG counters from API.

    Prefers /api/rag/stats (authoritative). Falls back to /api/health
    for backward compatibility.
    """
    # Preferred endpoint: /api/rag/stats
    try:
        resp = requests.get(f"{API_BASE}/api/rag/stats", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json() or {}
            mem = data.get("memories", {}) or {}
            vec = data.get("vectors", {}) or {}
            graph = data.get("graph", {}) or {}
            return {
                "source": "rag/stats",
                "memories": int(mem.get("memories", 0) or 0),
                "vectors": int(vec.get("total_vectors", 0) or 0),
                "graph_nodes": int(graph.get("nodes", 0) or 0),
                "graph_edges": int(graph.get("edges", 0) or 0),
                "backend": str(mem.get("backend", "unknown") or "unknown"),
            }
    except Exception:
        pass

    # Fallback endpoint: /api/health
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json() or {}
            return {
                "source": "health",
                "memories": int(data.get("memories_count", 0) or 0),
                "vectors": int(data.get("vectors_count", 0) or 0),
                "graph_nodes": int(data.get("graph_nodes", 0) or 0),
                "graph_edges": int(data.get("graph_edges", 0) or 0),
                "backend": str(data.get("memories_backend", "unknown") or "unknown"),
            }
    except Exception:
        pass

    return None


def check_backend_health() -> bool:
    """Check if the backend is running and healthy."""
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✅ Backend is healthy")

            stats = get_system_stats(timeout=5)
            if stats:
                print(f"     Stats source: {stats['source']}")
                print(f"     Memories: {stats['memories']} (backend={stats['backend']})")
                print(f"     Vectors: {stats['vectors']}")
                print(f"     Graph nodes: {stats['graph_nodes']}")
                print(f"     Graph edges: {stats['graph_edges']}")
            else:
                print("     ⚠ Could not fetch RAG counters")
            return True
        else:
            print(f"  ❌ Backend returned status {resp.status_code}")
            return False
    except requests.ConnectionError:
        print(f"  ❌ Cannot connect to backend at {API_BASE}")
        print(f"     Please start the backend first: cd backend && python server.py")
        return False


def ingest_chunk(content: str, source: str, session_id: str = SESSION_ID) -> dict:
    """Send a single chunk to the ingest endpoint."""
    payload = {
        "content": content,
        "source": source,
        "session_id": session_id,
    }
    resp = requests.post(INGEST_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest personal data into Cortex Lab")
    parser.add_argument("--dry-run", action="store_true", help="Preview chunks without ingesting")
    parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE,
                        help=f"Max characters per chunk (default: {DEFAULT_MAX_CHUNK_SIZE})")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between ingestions in seconds (default: 0.5)")
    parser.add_argument("--file", type=str, default=None,
                        help="Ingest only a specific file (filename, not path)")
    args = parser.parse_args()

    print("=" * 70)
    print("🧠 Cortex Lab — Personal Data Bulk Ingestion")
    print("=" * 70)
    
    # Check raw_data directory
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"\n❌ raw_data/ directory not found at: {RAW_DATA_DIR}")
        sys.exit(1)
    
    # List files
    md_files = sorted([
        f for f in os.listdir(RAW_DATA_DIR)
        if f.endswith(".md") and os.path.isfile(os.path.join(RAW_DATA_DIR, f))
    ])
    
    if args.file:
        md_files = [f for f in md_files if f == args.file]
        if not md_files:
            print(f"\n❌ File '{args.file}' not found in raw_data/")
            sys.exit(1)
    
    print(f"\n📁 Found {len(md_files)} files in raw_data/:")
    for f in md_files:
        size = os.path.getsize(os.path.join(RAW_DATA_DIR, f))
        print(f"   • {f} ({size:,} bytes)")
    
    # Check backend (skip in dry run)
    if not args.dry_run:
        print(f"\n🔌 Checking backend connection...")
        if not check_backend_health():
            sys.exit(1)
    
    # Process each file
    all_chunks = []
    print(f"\n📝 Chunking files (max {args.max_chunk_size} chars/chunk)...")
    print("-" * 70)
    
    for filename in md_files:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        text = read_file(filepath)
        source = FILE_SOURCE_MAP.get(filename, "personal-data")
        
        chunks = chunk_markdown(text, filename, max_size=args.max_chunk_size)
        
        # Add source context to each chunk
        contextualized = []
        for chunk_text, section_header in chunks:
            full_text = add_source_context(chunk_text, filename, section_header)
            contextualized.append((full_text, source, filename, section_header))
        
        all_chunks.extend(contextualized)
        print(f"   📄 {filename}: {len(chunks)} chunks")
    
    print(f"\n   📊 Total chunks to ingest: {len(all_chunks)}")
    
    # Dry run - preview chunks
    if args.dry_run:
        print(f"\n🔍 DRY RUN — Previewing all {len(all_chunks)} chunks:")
        print("=" * 70)
        for i, (text, source, fname, header) in enumerate(all_chunks, 1):
            preview = text[:200].replace('\n', ' ')
            print(f"\n[{i}/{len(all_chunks)}] Source: {source} | File: {fname}")
            if header:
                print(f"  Section: {header}")
            print(f"  Length: {len(text)} chars")
            print(f"  Preview: {preview}...")
            print("-" * 50)
        
        print(f"\n✅ Dry run complete. {len(all_chunks)} chunks would be ingested.")
        print(f"   Run without --dry-run to actually ingest.")
        return
    
    # Actual ingestion
    print(f"\n🚀 Starting ingestion ({len(all_chunks)} chunks, {args.delay}s delay)...")
    print("=" * 70)

    baseline_stats = get_system_stats(timeout=8)
    if baseline_stats:
        print(
            "  📌 Baseline: "
            f"memories={baseline_stats['memories']}, "
            f"vectors={baseline_stats['vectors']}, "
            f"nodes={baseline_stats['graph_nodes']}, "
            f"edges={baseline_stats['graph_edges']}"
        )
    
    success = 0
    failed = 0
    errors = []
    start_time = time.time()
    
    for i, (text, source, fname, header) in enumerate(all_chunks, 1):
        try:
            preview = text[:80].replace('\n', ' ')
            print(f"  [{i}/{len(all_chunks)}] Ingesting from {fname}...", end="", flush=True)
            
            result = ingest_chunk(content=text, source=source)
            memory_payload = result.get("memory", result)

            memory_id = memory_payload.get("id", "unknown")
            memory_type = memory_payload.get("memory_type", "unknown")
            topics = memory_payload.get("topics", [])
            
            print(f" ✅ [{memory_type}] topics={topics[:3]}")
            success += 1
            
            # Delay between ingestions to avoid overwhelming the system
            if i < len(all_chunks):
                time.sleep(args.delay)
                
        except requests.exceptions.HTTPError as e:
            print(f" ❌ HTTP Error: {e}")
            errors.append((fname, str(e)))
            failed += 1
        except requests.exceptions.ConnectionError:
            print(f" ❌ Connection lost!")
            errors.append((fname, "Connection lost"))
            failed += 1
            print("  ⏸  Waiting 5s before retry...")
            time.sleep(5)
        except Exception as e:
            print(f" ❌ Error: {e}")
            errors.append((fname, str(e)))
            failed += 1
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 INGESTION SUMMARY")
    print("=" * 70)
    print(f"  ✅ Successfully ingested: {success}/{len(all_chunks)} chunks")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏱  Total time: {elapsed:.1f}s ({elapsed/max(success,1):.1f}s/chunk avg)")
    
    if errors:
        print(f"\n  ⚠️  Errors:")
        for fname, err in errors:
            print(f"     • {fname}: {err}")
    
    # Verify final state
    print(f"\n🔍 Verifying final state...")
    final_stats = get_system_stats(timeout=8)
    if final_stats:
        print(f"  📦 Total memories: {final_stats['memories']} (backend={final_stats['backend']})")
        print(f"  🔢 Total vectors: {final_stats['vectors']}")
        print(f"  🕸  Graph nodes: {final_stats['graph_nodes']}")
        print(f"  🔗 Graph edges: {final_stats['graph_edges']}")

        if baseline_stats:
            delta_memories = final_stats["memories"] - baseline_stats["memories"]
            delta_vectors = final_stats["vectors"] - baseline_stats["vectors"]
            delta_nodes = final_stats["graph_nodes"] - baseline_stats["graph_nodes"]
            delta_edges = final_stats["graph_edges"] - baseline_stats["graph_edges"]

            print("  Δ Change during this run:")
            print(f"     Memories: {delta_memories:+d}")
            print(f"     Vectors: {delta_vectors:+d}")
            print(f"     Graph nodes: {delta_nodes:+d}")
            print(f"     Graph edges: {delta_edges:+d}")

            if success > 0 and delta_vectors <= 0:
                print("  ⚠️  No new vectors added. This can happen when chunks deduplicate against existing memories.")
            if success > 0 and delta_nodes <= 0 and delta_edges <= 0:
                print("  ⚠️  No new graph structure added. This can happen when entities already exist or chunks deduplicate.")
    else:
        print("  ⚠️  Could not verify final state from /api/rag/stats or /api/health")
    
    print(f"\n✅ Done! Your personal data is now part of Cortex Lab's memory.")
    print(f"   Chat with your AI to ask questions about yourself, your projects, and visions.")


if __name__ == "__main__":
    main()
