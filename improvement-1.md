# Cortex-Lab Improvement Report v1

## Complete Fix Log, Architecture Analysis & Improvement Roadmap

> **Purpose:** This document captures every fix applied to the Cortex-Lab Agentic RAG system, explains root causes, maps the full architecture flow with its gaps, and provides a roadmap for remaining improvements. Use this to replicate all fixes on a fresh git-pull clone.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Bugs Fixed](#2-critical-bugs-fixed)
3. [File-by-File Change Log](#3-file-by-file-change-log)
4. [Architecture Flow Analysis](#4-architecture-flow-analysis)
5. [What Was Lagging & Why](#5-what-was-lagging--why)
6. [Remaining Gaps](#6-remaining-gaps)
7. [Improvement Roadmap](#7-improvement-roadmap)
8. [Hardcoded Values Inventory](#8-hardcoded-values-inventory)
9. [Quick-Apply Checklist](#9-quick-apply-checklist)

---

## 1. Executive Summary

### Before Fixes
- Answers truncated to **127 characters** despite correct evidence retrieval
- Knowledge graph **empty** (0 nodes, 0 edges) — lost on server kill
- Graph retrieval channel returning **0 results** on every query
- Server startup **hanging for 10+ minutes** trying to build proposition index
- Gemini provider **never activated** — all queries routed to LocalLLM (which had no model loaded)
- Evidence passed to LLM capped at **200-500 chars** across 6+ code locations
- Frontend showing "offline" because Gemini-only mode not handled

### After Fixes
- Answers: **2,000–4,000+ characters** with comprehensive details
- Knowledge graph: **314 nodes, 7,239 edges** rebuilt from 399 memories
- All 3 retrieval channels active: dense=16, sparse=16, graph=16 → 8 fused+reranked
- Server startup: **~7 seconds** (proposition index deferred to lazy build)
- Gemini provider auto-set on startup, provider switching logic fixed
- Evidence limits raised to **1,500 chars** across all agent/orchestrator paths
- Both `/api/chat` and `/api/rag/chat` fully operational

---

## 2. Critical Bugs Fixed

### Bug #1: LLM Provider Delegation (Root Cause of 127-char answers)

**Symptom:** Every RAG query produced exactly 127 characters regardless of evidence quality.

**Root Cause Chain:**
1. `server.py` → `_set_request_provider(req.llm_provider)` received `"local"` (the default in the request body)
2. `LLMProvider.set_provider("local")` switched `active_llm` to `LocalLLM`
3. `LocalLLM` had `model=None` (no local model loaded in Gemini-only mode)
4. `LocalLLM.raft_generate()` has its own implementation with `doc[:200]`, `max_tokens=400`, plus aggressive `_strip_hallucination_patterns()` and `_validate_or_extract()` post-processing
5. With `model=None`, LocalLLM returned a short fallback string → 127 chars

**Fix (server.py, `_set_request_provider`):**
```python
# BEFORE:
def _set_request_provider(provider: str):
    rag_engine.llm.set_provider(provider)

# AFTER:
def _set_request_provider(provider: str):
    if provider == "local" and (rag_engine.llm.local_llm is None
                                 or rag_engine.llm.local_llm.model is None):
        provider = "gemini" if rag_engine.llm.has_gemini else "local"
    rag_engine.llm.set_provider(provider)
```

**Fix (server.py, `lifespan` startup):**
```python
# After Gemini-only init, explicitly set provider:
rag_engine.llm.set_provider("gemini")
```

### Bug #2: Gemini 2.5 Flash Thinking Token Budget

**Symptom:** Second and subsequent queries truncated (176 chars). Direct API debug revealed `finish_reason: FinishReason.MAX_TOKENS` with `thoughts_token_count=1089`.

**Root Cause:** Gemini 2.5 Flash is a **thinking model** — internal Chain-of-Thought reasoning tokens count against `max_output_tokens`. With `max_output_tokens=1024`:
- ~1,000 tokens consumed by thinking
- Only ~24 tokens left for visible output
- Result: `MAX_TOKENS` finish reason, answer cut off mid-sentence

**Fix (gemini_llm.py, `_make_config`):**
```python
# BEFORE:
def _make_config(self, max_tokens=512, temperature=0.3, top_p=0.95):
    return self._types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        ...
    )

# AFTER:
def _make_config(self, max_tokens=512, temperature=0.3, top_p=0.95):
    effective_max = min(max_tokens * 8, 65536)
    return self._types.GenerateContentConfig(
        max_output_tokens=effective_max,
        ...
    )
```

**Why 8x:** At max_tokens=1024 → effective=8192. Gemini uses ~6,000-7,000 for thinking, leaving ~1,200-2,200 for visible output. Cap at 65536 (Gemini's absolute limit).

### Bug #3: Evidence Truncation at 6+ Code Locations

**Symptom:** Even when LLM was working, answers were shallow because evidence passed to the LLM was only 200-500 chars per document.

**Root Cause:** The original code was designed for a local 7B model with limited context windows. Evidence was aggressively truncated to fit. With Gemini (1M token context), this was unnecessary.

**Fixes Applied (6 locations):**

| File | Function | Before | After |
|------|----------|--------|-------|
| `specialized.py` | `_evidence_texts()` | `content[:500]` | `content[:1500]` |
| `specialized.py` | `PlanningAgent` oracle docs | `content[:250]`, 3 docs | `content[:1500]`, 5 docs |
| `specialized.py` | `PlanningAgent` distractors | `content[:250]`, 3 total | `content[:1500]`, 8 total |
| `specialized.py` | Sub-query evidence | `max_items=3` | `max_items=5` |
| `orchestrator.py` | `_handle_multi_step` | `content[:250]`, 5 pieces | `content[:1500]`, 8 pieces |
| `orchestrator.py` | Self-RAG critique | `content[:200]` | `content[:1000]` |
| `orchestrator.py` | Self-RAG revision | `answer[:300]`, `evidence[:3]`, `max_tokens=400` | `answer[:800]`, `evidence[:5]`, `max_tokens=1024` |
| `orchestrator.py` | No-retrieval generation | `max_tokens=300` | `max_tokens=1024` |
| `orchestrator.py` | Fallback synthesis | `max_tokens=500` | `max_tokens=1024` |

### Bug #4: Knowledge Graph Lost on Server Kill

**Symptom:** After server restart, `/api/rag/stats` showed 0 graph nodes despite 399 memories in DuckDB.

**Root Cause:** The knowledge graph is an in-memory NetworkX DiGraph. When the server process is killed (especially on Windows), the shutdown hook that serializes the graph to JSON doesn't fire. The graph file at `data/graph/knowledge_graph.json` was either empty or stale.

**Fix:** Created `rebuild_graph.py` script that:
1. Reads all 399 memories from DuckDB
2. Extracts entities and topics from each memory
3. Builds entity nodes and co-occurrence edges
4. Serializes to `data/graph/knowledge_graph.json`
5. Result: **314 nodes, 7,239 edges**

### Bug #5: Graph Retrieval Returns 0 Results

**Symptom:** After graph rebuild, `graph: 0` in retrieval stats while dense and sparse channels returned 16 each.

**Root Cause:** `PlanningAgent` creates sub-queries for multi-step decomposition. These sub-queries were constructed as new `MemoryQuery` objects but **without propagating `entities` and `topics`** from the parent query. The `_graph_retrieve()` method checks `query.entities` — if empty, it returns immediately with 0 results.

**Fix (specialized.py, PlanningAgent sub-query construction):**
```python
# BEFORE:
sub_q = MemoryQuery(
    raw_query=sq,
    intent=query.intent,
    ...
)

# AFTER:
sub_q = MemoryQuery(
    raw_query=sq,
    intent=query.intent,
    entities=query.entities,   # ← propagate from parent
    topics=query.topics,       # ← propagate from parent
    ...
)
```

### Bug #6: Server Startup Hang (10+ minutes)

**Symptom:** Server blocked on startup for 10+ minutes before accepting requests.

**Root Cause:** `engine.py` init was trying to pre-build the proposition index, which required embedding every proposition of every memory. With 399 memories × ~6.5 propositions each = ~2,533 Gemini embedding API calls at startup.

**Fix (engine.py):** Commented out proposition pre-build:
```python
# ── PRE-BUILD proposition index ──────
# Skip on startup — proposition channel has low weight (0.10) and
# will rebuild lazily on first query. Startup speed is more important.
```

Additionally, wrapped all 11 init steps in try/except for resilient initialization and ran init via `run_in_executor` for non-blocking startup.

### Bug #7: Proposition Channel API Overload

**Symptom:** When proposition retrieval channel is active with Gemini, each query triggers ~26 additional API calls (proposition extraction + embedding for each candidate).

**Fix (hybrid_retriever.py):** Conditionally disable proposition channel for Gemini backend:
```python
if getattr(self.embeddings, '_backend', 'stub') == 'local':
    # Only enable propositions when using local embeddings (instant)
    channels.append(("proposition", self._proposition_retrieve))
```

Channel weights were also rebalanced: dense=0.35, sparse=0.25, graph=0.20, temporal=0.10.

---

## 3. File-by-File Change Log

### Files MODIFIED

#### `backend/server.py` (~2,400 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Provider fix | `_set_request_provider()` ~L702 | Don't switch to "local" when no local model exists |
| Gemini auto-set | `lifespan()` ~L148 | Call `set_provider("gemini")` after Gemini-only init |
| model_loaded flag | `lifespan()` ~L155 | Set `model_loaded = True` in Gemini path |
| Non-blocking init | `lifespan()` ~L140 | Run `rag_engine.init()` via `loop.run_in_executor` |
| False-premise detection | `_check_no_info_streaming()` ~L351 | Prevent hallucinated answers for salary, PhD, marriage queries |
| Factual extraction | `_try_extract_factual()` ~L451 | Pre-generation bypass for name, email, phone, education |
| Pronoun fixing | `_fix_person_pronouns()` ~L430 | Convert first-person evidence to second-person for LLM |

#### `backend/src/engine.py` (~650 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Resilient init | `init()` ~L145 | All 11 subsystem inits wrapped in try/except |
| Proposition skip | ~L265 | Proposition pre-build commented out |
| Content filter | `_is_meaningful_content()` ~L418 | Prevents user queries from being stored as memories |
| Non-blocking init | Startup | Runs via `run_in_executor` |
| Auto-reindex | `_reindex_missing_vectors()` ~L390 | Bulk-embed memories below 80% vector coverage |

#### `backend/src/agents/orchestrator.py` (~1,050 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Evidence size | `_handle_multi_step` ~L650 | `content[:250]` → `content[:1500]`, 5 → 8 pieces |
| Self-RAG critique | `_self_rag_critique` ~L803 | `content[:200]` → `content[:1000]` |
| Self-RAG revision | ~L880 | `answer[:300]` → `[:800]`, `evidence[:3]` → `[:5]`, `max_tokens` 400 → 1024 |
| No-retrieval gen | `_handle_no_retrieval` ~L569 | `max_tokens` 300 → 1024 |
| Fallback synthesis | ~L595 | `max_tokens` 500 → 1024 |
| Self-RAG skip | ~L468 | Only runs if confidence < 0.55 (saves 3-6s) |
| FLARE skip | ~L611 | Only runs if confidence < 0.4 (most expensive step) |
| LLM routing skip | ~L274 | Only if 0.35 < complexity < 0.65 (saves 2-4s) |

#### `backend/src/agents/specialized.py` (~545 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Evidence filter | `_evidence_texts()` ~L40 | `content[:500]` → `content[:1500]` |
| Evidence quality | `_evidence_texts()` ~L40-75 | Full quality filtering (min length, question detection, trigram spam, query pattern rejection) |
| RAFT oracle docs | `PlanningAgent` ~L197 | `content[:250]` → `content[:1500]`, 3 → 5 docs |
| RAFT distractors | `PlanningAgent` ~L198 | `content[:250]` → `content[:1500]`, 3 → 8 total |
| Sub-query evidence | `PlanningAgent` ~L200 | `max_items` 3 → 5 |
| Entity propagation | `PlanningAgent` ~L195 | Sub-queries now inherit `entities` and `topics` from parent query |
| Debug logging | `PlanningAgent` ~L199 | Added print for RAFT generation tracing |

#### `backend/src/retrieval/hybrid_retriever.py` (~850 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Proposition gate | ~L128 | Only enable when `_backend == 'local'` |
| Channel weights | ~L52 | Rebalanced: dense=0.35, sparse=0.25, graph=0.20, temporal=0.10 |
| PageIndex bypass | ~L228 | PageIndex results bypass RRF, injected at top |
| Graph early return | `_graph_retrieve` ~L454 | Return empty if `query.entities` is empty |
| PageIndex strategy | `_pageindex_retrieve` ~L598 | Use `chat_retrieve()` (5-10s) instead of `retrieve_sections()` (30-40s) |

#### `backend/src/retrieval/query_engine.py` (~750 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Batched Gemini | `_transform_batched_gemini()` ~L419 | Single API call produces all query variants (3 calls → 1) |
| Relevance validation | `_generate_multi_queries()` ~L473 | Discard generated queries with zero content-word overlap |
| Entity-aware queries | Multi-query generation | Include detected entities in generated variants |

#### `backend/src/llm/gemini_llm.py` (~500 lines)
| Change | Location | Description |
|--------|----------|-------------|
| **Token budget** | `_make_config()` ~L49 | `max_output_tokens *= 8`, capped at 65536 |
| Evidence in faithful | `generate_faithful()` ~L310 | `e[:1500]`, 8 pieces, `max_tokens=1024` |
| Evidence in RAFT | `raft_generate()` ~L328 | `content[:1500]`, 5 oracle + 3 distractor, `max_tokens=1024` |
| False premise checks | `_validate_or_extract()` ~L255 | Detects false premises (PhD, marriage, salary) |

#### `backend/src/models/embeddings.py` (~200 lines)
| Change | Location | Description |
|--------|----------|-------------|
| Gemini fallback | `__init__` ~L37 | 3-tier: local → Gemini API → stub hash |
| Batch embedding | `_gemini_embed_batch()` ~L104 | 100/call, 3 retries, exponential backoff |
| `_backend` property | ~L60 | Exposed for conditional logic in retriever |
| Dimension: 3072 | API-derived | Gemini embedding-001 outputs 3072d vectors |

#### `frontend/src/lib/types.ts`
| Change | Description |
|--------|-------------|
| Gemini fields | Added Gemini-related type definitions |

#### `frontend/src/components/ChatPanel.tsx`
| Change | Description |
|--------|-------------|
| isOnline check | Handles Gemini-only mode (no local model = still online) |

#### `backend/src/storage/pageindex_store.py`
| Change | Description |
|--------|-------------|
| Connection test | Fixed initialization/connection validation |

### Files CREATED

| File | Purpose |
|------|---------|
| `backend/src/models/__init__.py` | 21 data classes for the entire system (CausalMemoryObject, EntityNode, GraphEdge, QueryIntent, RoutingStrategy, etc.) |
| `backend/src/models/embeddings.py` | Tiered embedding model (local → Gemini → stub) with 3072d output |
| `backend/src/cache/__init__.py` | 3-level caching (exact hash → semantic similarity → miss) |
| `backend/rebuild_graph.py` | Rebuild knowledge graph from DuckDB memories |
| `backend/scripts/ingest_raw_data.py` | Bulk ingestion from raw_data/ markdown files |

---

## 4. Architecture Flow Analysis

### Full Request Pipeline (How a RAG query flows)

```
User Query
    │
    ├─ /api/rag/chat (server.py L1190)
    │   ├─ _set_request_provider()     ← BUG #1 was here (provider switching)
    │   ├─ _check_no_info_streaming()  ← False premise detection
    │   └─ _try_extract_factual()      ← Direct extraction bypass
    │
    ├─ engine.rag_chat() (engine.py L526)
    │   ├─ Check semantic cache         ← 3-level cache lookup
    │   └─ orchestrator.process()
    │
    ├─ Orchestrator Pipeline (orchestrator.py L150)
    │   ├─ Step 1: Query Analysis
    │   │   └─ query_engine.analyze()   ← Intent, complexity, routing
    │   │
    │   ├─ Step 2: Query Transformation
    │   │   └─ query_engine.transform() ← Multi-query, HyDE, step-back, decomposition
    │   │   └─ _transform_batched_gemini() ← Single API call for all variants
    │   │
    │   ├─ Step 3: Routing Decision
    │   │   ├─ complexity < 0.3  → NO_RETRIEVAL
    │   │   ├─ complexity < 0.6  → SINGLE_STEP
    │   │   └─ complexity >= 0.6 → MULTI_STEP
    │   │   └─ LLM routing only if 0.35 < complexity < 0.65
    │   │
    │   ├─ Step 4: Retrieval
    │   │   └─ hybrid_retriever.retrieve() (6 channels, parallel)
    │   │       ├─ Dense (FAISS/NumPy cosine)  weight=0.35
    │   │       ├─ Sparse (BM25)               weight=0.25
    │   │       ├─ Graph (NetworkX traversal)   weight=0.20
    │   │       ├─ Temporal (time-filtered)     weight=0.10
    │   │       ├─ Proposition (disabled for Gemini)
    │   │       └─ PageIndex (cloud, bypasses RRF)
    │   │       └─ RRF Fusion → Cross-Encoder Rerank → top_k results
    │   │
    │   ├─ Step 5: Agent Execution
    │   │   ├─ TimelineAgent   → temporal queries
    │   │   ├─ CausalAgent     → cause-effect reasoning
    │   │   ├─ ReflectionAgent → belief evolution
    │   │   ├─ PlanningAgent   → complex multi-step (RAFT) ← Most queries land here
    │   │   └─ ArbitrationAgent → conflict resolution
    │   │
    │   ├─ Step 6: CRAG Evaluation
    │   │   └─ _crag_evaluate() → CORRECT/AMBIGUOUS/INCORRECT verdict
    │   │   └─ Quality = 0.40*avg + 0.20*max + 0.20*coverage + 0.20*entities
    │   │
    │   ├─ Step 7: Self-RAG Reflection (only if confidence < 0.55)
    │   │   └─ _self_rag_critique() → ISREL/ISSUP/ISUSE scoring
    │   │   └─ If fails → re-generate with more evidence
    │   │
    │   └─ Step 8: FLARE Active Retrieval (only if confidence < 0.4)
    │       └─ _flare_active_retrieval() → 2 additional retrieval rounds
    │
    └─ Response Assembly
        ├─ Cache result (semantic + exact)
        ├─ Background ingest (if meaningful, not a question)
        └─ Return with evidence, confidence, agent trace
```

### Where The Architecture Was Broken

| Pipeline Stage | Problem | Impact |
|----------------|---------|--------|
| **Provider Selection** | Always switched to "local" | 100% of queries used wrong LLM |
| **Token Budget** | Gemini thinking tokens not accounted for | 80%+ of output truncated |
| **Evidence Passing** | 200-500 char limits designed for 7B model | LLM couldn't synthesize meaningful answers |
| **Graph Retrieval** | Entities not propagated to sub-queries | Entire graph channel wasted |
| **Proposition Channel** | 26 API calls per query with Gemini | Startup blocked, queries slow |
| **Startup** | Synchronous proposition pre-build | 10+ minute server unavailability |
| **Graph Persistence** | In-memory only, no crash recovery | Data loss on server kill |

---

## 5. What Was Lagging & Why

### 5.1 Design Assumptions That Didn't Hold

The architecture was designed for a **fine-tuned local 7B/13B model** with:
- 4K-8K context window → aggressive evidence truncation was necessary
- Local inference → all LLM calls are cheap and fast
- GPU memory constraints → VRAMGuard, tiered vectors, etc.
- Sentence-transformers on GPU → embeddings are instant and local

With **Gemini 2.5 Flash** (1M context, thinking model, API-based):
- Evidence truncation is counterproductive
- Each LLM call costs API quota + network latency
- Thinking tokens are invisible but consume output budget
- Embeddings require batch API calls, not instant

### 5.2 Evidence Handling Was The Biggest Lag

The evidence flow has **3 layers** of truncation between retrieval and generation:

```
Retrieved Memories (full text, 500-5000 chars each)
    ↓ Layer 1: specialized.py _evidence_texts()     — was [:500]  → now [:1500]
    ↓ Layer 2: specialized.py PlanningAgent oracle   — was [:250]  → now [:1500]
    ↓ Layer 3: orchestrator.py _handle_multi_step    — was [:250]  → now [:1500]
    ↓
LLM receives evidence (was 200-500 chars, now 1500 chars per doc)
```

Each layer independently truncated, so by the time evidence reached the LLM, it was just fragments of fragments.

### 5.3 Provider System Never Tested for Gemini-Only

The `LLMProvider` class has two backends (`local_llm` and `gemini_llm`) with a `set_provider()` switcher. But:
- The startup path for Gemini-only never called `set_provider("gemini")`
- Every request's `_set_request_provider()` received `"local"` (default) and blindly switched
- No guard against switching to a non-existent backend

### 5.4 Knowledge Graph Has No Crash Recovery

The NetworkX graph lives entirely in memory. Serialization to JSON only happens in the `shutdown()` hook. On Windows, killing the server process (Ctrl+C, Task Manager) doesn't reliably trigger shutdown hooks → graph data is lost.

### 5.5 BM25 Index Stale After Ingestion

The BM25 sparse index is only rebuilt when `retrieve()` is called and detects staleness. New memories ingested between retrievals don't appear in sparse search until the next query triggers a rebuild. This means the first query after bulk ingestion may miss recent memories in the sparse channel.

### 5.6 Cross-Encoder Reranking Disabled

With `sentence-transformers` not installed, the `CrossEncoderReranker` falls back to a simplistic score: `1 / (rank + 1)`. This means the reranking step is essentially a no-op — it preserves the RRF fusion order rather than doing actual semantic re-scoring.

---

## 6. Remaining Gaps

### 6.1 Critical (Will Affect Answer Quality)

| # | Gap | File | Impact |
|---|-----|------|--------|
| 1 | **Cross-encoder reranking disabled** | `embeddings.py` | RRF fusion not re-scored; bad documents may rank higher than relevant ones |
| 2 | **Self-RAG skips high-confidence answers** | `orchestrator.py` L468 | If evidence is misleading but scores high, wrong answer passes unchecked |
| 3 | **Entity extraction is primitive** | `query_engine.py` L300+ | Only detects capitalized words → misses "python", "linux", "react", "aws" |
| 4 | **No max_tokens in Gemini streaming** | `gemini_llm.py` | Streaming paths may not apply 8x multiplier consistently |
| 5 | **LocalLLM still has 200-char limits** | `llm/__init__.py` | If anyone switches to local, old truncation returns |
| 6 | **No rate limiting on `/api/rag/chat`** | `server.py` | Single user can exhaust Gemini API quota |

### 6.2 Important (Architectural Weaknesses)

| # | Gap | File | Impact |
|---|-----|------|--------|
| 7 | **Sub-queries are sequential** | `specialized.py` | PlanningAgent runs sub-queries one-by-one; could be parallelized |
| 8 | **Proposition channel disabled** | `hybrid_retriever.py` | Atomic fact retrieval unavailable with Gemini backend |
| 9 | **PageIndex results bypass RRF** | `hybrid_retriever.py` | Cloud results injected at top with fixed scores, not fairly ranked |
| 10 | **BM25 not rebuilt after ingestion** | `hybrid_retriever.py` | New memories invisible to sparse search until next query |
| 11 | **FLARE implementation incomplete** | `orchestrator.py` | Referenced in pipeline but execution path partially implemented |
| 12 | **ArbitrationAgent underutilized** | `specialized.py` | Conflict resolution agent exists but routing rarely selects it |
| 13 | **Function calling (Stage 13) unused** | `gemini_llm.py` | `call_function()` registered but never invoked in pipeline |
| 14 | **No semantic cache enabled** | `cache/__init__.py` | Cache exists but not wired into engine for real queries |

### 6.3 Operational (Deployment & Reliability)

| # | Gap | File | Impact |
|---|-----|------|--------|
| 15 | **No graceful graph persistence** | `knowledge_graph.py` | Graph lost on crash/kill |
| 16 | **Ambient voice service untested** | `ambient/` | 8 files, never tested end-to-end |
| 17 | **DuckDB file lock on Windows** | `metadata_store.py` | Can't run rebuild_graph while server is running |
| 18 | **No health check for Gemini** | `server.py` | `/api/health` doesn't verify Gemini API connectivity |
| 19 | **Hardcoded CORS origins** | `server.py` | Only allows specific origins, breaks on different deployments |
| 20 | **Vector store dimension mismatch** | `vector_store.py` | Default `dimension=384` in code, actual Gemini = 3072 |
| 21 | **Zero test coverage** | (none) | No unit tests, no integration tests for any component |

---

## 7. Improvement Roadmap

### Phase 1: Immediate Fixes (Apply to Fresh Clone)

These are the minimum changes needed to get a fresh git-pull working with Gemini:

1. **Apply all evidence truncation fixes** (Section 2, Bug #3)
2. **Apply provider delegation fix** (Section 2, Bug #1)
3. **Apply Gemini token budget fix** (Section 2, Bug #2)
4. **Apply entity propagation fix** (Section 2, Bug #5)
5. **Disable proposition pre-build** (Section 2, Bug #6)
6. **Add `_set_request_provider` guard** (Section 2, Bug #1)
7. **Set Gemini provider on startup** (Section 2, Bug #1)
8. **Create `backend/src/models/embeddings.py`** with Gemini fallback
9. **Create `backend/src/models/__init__.py`** with data classes
10. **Run `rebuild_graph.py`** after ingesting data

### Phase 2: Quality & Reliability

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| HIGH | Install sentence-transformers for cross-encoder reranking | 1 hour | Significant quality boost |
| HIGH | Add periodic graph serialization (every 5 min + on new edges) | 2 hours | Prevents data loss |
| HIGH | Add entity extraction for lowercase tech terms | 2 hours | Better graph retrieval |
| MED | Parallelize PlanningAgent sub-queries with `asyncio.gather()` | 3 hours | 2-4x faster for complex queries |
| MED | Wire semantic cache into engine.rag_chat() | 2 hours | Skip LLM for repeated queries |
| MED | Add BM25 incremental update after ingestion | 2 hours | Sparse channel always current |
| LOW | Self-RAG on high-confidence answers (spot-check 10%) | 3 hours | Catch confident-but-wrong answers |

### Phase 3: Architecture Improvements

| Improvement | Description | Effort |
|-------------|-------------|--------|
| **Configuration system** | Extract all hardcoded values to YAML configs (see Section 8) | 1 day |
| **Proper error handling** | Replace all `print()` + `except: pass` with structured logging | 4 hours |
| **Rate limiting** | Add per-IP rate limiting on RAG endpoint (Gemini quota protection) | 2 hours |
| **Plugin system** | Replace hardcoded tool definitions with plugin registry | 1 day |
| **Test suite** | Unit tests for evidence filtering, routing, CRAG scoring | 2 days |
| **Vector tier persistence** | Save warm/cold FAISS indices, not just hot | 4 hours |
| **Streaming token budget** | Ensure 8x multiplier in all streaming paths | 2 hours |
| **Function calling pipeline** | Wire Stage 13 into orchestrator for tool-use queries | 1 day |
| **Graph-based reranking** | Use graph proximity as reranking signal | 4 hours |
| **Proposition lazy rebuild** | Build proposition index in background after first query | 3 hours |

### Phase 4: Advanced Features

| Feature | Description |
|---------|-------------|
| **Multi-model routing** | Use Flash for simple queries, Pro for complex ones |
| **Adaptive evidence sizing** | Dynamically set evidence limits based on query complexity |
| **Graph community summarization** | Pre-compute community summaries for faster graph retrieval |
| **Conversation memory** | Track multi-turn context for follow-up queries |
| **A/B testing framework** | Compare different retrieval/generation strategies |
| **Embedding fine-tuning** | Fine-tune Gemini embeddings on user's domain |

---

## 8. Hardcoded Values Inventory

All values that should be extracted to configuration files:

### `config/server.yaml`
```yaml
server:
  port: 8000
  cors_origins: ["http://localhost:3000", ...]
  max_concurrent_rag: 2      # semaphore limit
  streaming_timeout: 90       # seconds
  default_temperature: 0.6
  default_top_p: 0.95
  default_max_tokens: 2048
  max_context_length: 8000    # Gemini-only mode

llm:
  gemini_model: "gemini-2.5-flash"
  thinking_multiplier: 8      # for thinking models
  max_output_tokens_cap: 65536
  local_repetition_penalty: 1.15
  local_stop_patterns: ["\nUser:", ...]
```

### `config/retrieval.yaml`
```yaml
channels:
  dense:
    weight: 0.35
    variants_per_query: 3
  sparse:
    weight: 0.25
    bm25_k1: 1.5
    bm25_b: 0.75
  graph:
    weight: 0.20
    max_hops: 2
    causal_relations: ["caused", "led_to", "influenced", "resulted_in"]
  temporal:
    weight: 0.10
  proposition:
    weight: 0.10
    enabled_backends: ["local"]  # disable for "gemini"
    similarity_threshold: 0.4

rrf:
  k: 60

reranking:
  cross_encoder_model: "BAAI/bge-reranker-v2-m3"
  blend_weights: [0.70, 0.20, 0.10]  # ce, rrf, importance
  max_doc_length: 512

evidence:
  max_chars_per_doc: 1500
  max_oracle_docs: 5
  max_total_docs: 8
  max_items_filter: 8
```

### `config/agents.yaml`
```yaml
routing:
  complexity_llm_range: [0.35, 0.65]
  no_retrieval_threshold: 0.3
  single_step_threshold: 0.6

crag:
  quality_weights: [0.40, 0.20, 0.20, 0.20]
  correct_threshold: 0.55
  ambiguous_threshold: 0.30

self_rag:
  skip_confidence: 0.55
  isrel_threshold: 0.05
  issup_threshold: 0.20
  isuse_threshold: 0.15
  max_revisions: 5

flare:
  skip_confidence: 0.4
  max_retrievals: 2

evidence_filtering:
  min_content_length: 50
  max_trigram_repeats: 3
  question_length_threshold: 120
  query_pattern_length_threshold: 200
```

### `config/storage.yaml`
```yaml
vector_store:
  default_dimension: 3072      # Gemini embedding-001
  hot_tier_days: 30
  warm_tier_days: 365
  hnsw_ef_search: 64
  hnsw_ef_construction: 128

metadata:
  db_path: "data/cortex.duckdb"
  default_emotion: "neutral"
  default_importance: 0.5

knowledge_graph:
  data_dir: "data/graph"
  max_hops: 2
  fuzzy_match_enabled: true

embeddings:
  local_model: "BAAI/bge-large-en-v1.5"
  gemini_model: "gemini-embedding-001"
  batch_size: 100
  max_text_length: 2048
  retry_attempts: 3

cache:
  max_exact: 200
  max_semantic: 50
  semantic_threshold: 0.92
```

---

## 9. Quick-Apply Checklist

Use this checklist to apply all fixes to a fresh clone in order:

### Prerequisites
```bash
pip install google-genai python-dotenv duckdb networkx scikit-learn numpy
```

Create `backend/.env`:
```
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
SKIP_LOCAL_MODEL=true
```

### Step-by-Step

- [ ] **1. Create `backend/src/models/__init__.py`** — Copy from current working system (21 data classes)
- [ ] **2. Create `backend/src/models/embeddings.py`** — Copy from current (3-tier embedding with Gemini)
- [ ] **3. Create `backend/src/cache/__init__.py`** — Copy from current (3-level cache)
- [ ] **4. Edit `backend/src/llm/gemini_llm.py`**
  - In `_make_config()`: Change `max_output_tokens=max_tokens` → `max_output_tokens=min(max_tokens * 8, 65536)`
  - In `generate_faithful()`: Change evidence `[:500]` → `[:1500]`, pieces from 5 → 8, `max_tokens` → 1024
  - In `raft_generate()`: Change docs `[:250]` → `[:1500]`, oracle 3 → 5, total 6 → 8, `max_tokens` → 1024
- [ ] **5. Edit `backend/server.py`**
  - In `_set_request_provider()`: Add guard for non-existent local model
  - In `lifespan()`: Add `rag_engine.llm.set_provider("gemini")` after Gemini init
  - In `lifespan()`: Set `model_loaded = True` in Gemini path
- [ ] **6. Edit `backend/src/agents/specialized.py`**
  - In `_evidence_texts()`: Change `content[:500]` → `content[:1500]`
  - In `PlanningAgent.execute()`: Change oracle `[:250]` → `[:1500]`, 3 → 5 docs
  - In `PlanningAgent.execute()`: Change distractors `[:250]` → `[:1500]`, 3 → 8 total
  - In `PlanningAgent.execute()`: Add `entities=query.entities, topics=query.topics` to sub-queries
  - Change `max_items=3` → `max_items=5` for sub-query evidence
- [ ] **7. Edit `backend/src/agents/orchestrator.py`**
  - In `_handle_multi_step()`: `content[:250]` → `[:1500]`, 5 → 8 pieces
  - In `_self_rag_critique()`: `content[:200]` → `[:1000]`
  - In Self-RAG revision: `answer[:300]` → `[:800]`, `evidence[:3]` → `[:5]`, `max_tokens=400` → `1024`
  - In `_handle_no_retrieval()`: `max_tokens=300` → `1024`
  - Fallback synthesis: `max_tokens=500` → `1024`
- [ ] **8. Edit `backend/src/engine.py`**
  - Comment out proposition pre-build section
  - Wrap all init steps in try/except
- [ ] **9. Edit `backend/src/retrieval/hybrid_retriever.py`**
  - Disable proposition channel for non-local backends
  - Rebalance weights: dense=0.35, sparse=0.25, graph=0.20, temporal=0.10
  - Add early return in `_graph_retrieve()` for empty entities
- [ ] **10. Edit `backend/src/retrieval/query_engine.py`**
  - Add `_transform_batched_gemini()` for single-call Gemini variant generation
- [ ] **11. Ingest data**
  ```bash
  cd backend
  python scripts/ingest_raw_data.py
  ```
- [ ] **12. Rebuild knowledge graph**
  ```bash
  python rebuild_graph.py
  ```
- [ ] **13. Start server**
  ```bash
  cd backend
  SKIP_LOCAL_MODEL=true python server.py
  ```
- [ ] **14. Verify**
  ```bash
  curl -X POST http://localhost:8000/api/rag/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Tell me about my projects and skills"}'
  ```
  Expected: 2000+ char answer with project details, GitHub repos, tech stacks.

---

## Appendix A: System Specifications (Current Working State)

| Component | Value |
|-----------|-------|
| Python | 3.14 (64-bit, Windows) |
| LLM | gemini-2.5-flash via google.genai SDK |
| Embeddings | gemini-embedding-001, 3072 dimensions |
| Vector Store | NumPy brute-force (no FAISS), 399 × 3072d |
| Metadata | DuckDB at data/cortex.duckdb, 399 memories |
| Knowledge Graph | NetworkX DiGraph, 314 nodes, 7,239 edges |
| Sparse Index | BM25 (rank_bm25 library) |
| Cross-Encoder | Disabled (no sentence-transformers) |
| PageIndex | v0.2.6, API key configured, 1 document |
| Frontend | Next.js on port 3000 |
| Backend | FastAPI on port 8000 |

## Appendix B: Package Dependencies

### Installed
```
fastapi, uvicorn, pydantic, numpy, duckdb, networkx, scikit-learn,
google-genai, python-dotenv, transformers, python-multipart,
pageindex (v0.2.6), rank_bm25
```

### Missing (handled gracefully)
```
torch, faiss-cpu/faiss-gpu, sentence-transformers, accelerate,
bitsandbytes, sounddevice, faster-whisper, speechbrain
```

---

*Generated: Session improvement audit v1*
*Files analyzed: 22 backend files, 2 frontend files*
*Changes tracked: 10 modified files, 5 created files*
