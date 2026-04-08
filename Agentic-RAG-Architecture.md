# Agentic RAG + LLM Wiki — Final Production Architecture
## Cortex Lab v4.0 — Web + Mobile + Always-On Local Intelligence

> **Version:** 4.0 — Tiered Intelligence Edition  
> **Design Basis:** RAG-Architecture v3.0 + Agentic-RAG-Wiki + OpenClaude provider patterns + claw-code Rust runtime patterns  
> **Core Principle:** Route fast. Retrieve smart. Store once. Build Wikipedia forever.

---

## Table of Contents

1. [The Fundamental Problem We Are Solving](#1-the-fundamental-problem-we-are-solving)
2. [Architecture Overview: Five Planes + Five Retrieval Tiers](#2-architecture-overview-five-planes--five-retrieval-tiers)
3. [Memory Plane Architecture](#3-memory-plane-architecture)
4. [The LLM Wiki Engine — First-Class Canonical Store](#4-the-llm-wiki-engine--first-class-canonical-store)
5. [Tiered Retrieval System — The Core Innovation](#5-tiered-retrieval-system--the-core-innovation)
6. [Ingestion Pipeline — Always-On + Selective](#6-ingestion-pipeline--always-on--selective)
7. [Agentic Reasoning Layer](#7-agentic-reasoning-layer)
8. [Wiki Agent — The Memory Wikipedia Builder](#8-wiki-agent--the-memory-wikipedia-builder)
9. [Quality and Consistency Stack](#9-quality-and-consistency-stack)
10. [Web + Mobile Architecture Specifics](#10-web--mobile-architecture-specifics)
11. [Provider Abstraction Layer](#11-provider-abstraction-layer)
12. [Storage Strategy — Local + Sync](#12-storage-strategy--local--sync)
13. [Observability and Self-Improvement Loop](#13-observability-and-self-improvement-loop)
14. [Production Gaps Analysis and Resolutions](#14-production-gaps-analysis-and-resolutions)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Success Metrics](#16-success-metrics)

---

## 1. The Fundamental Problem We Are Solving

### 1.1 The Two Core Failures of Current Agentic RAG Systems

Every production RAG system today fails in one of two ways:

**Failure A — Brute Force Everything:** Every query — whether "what did I eat yesterday" or "how has my worldview changed over 3 years" — goes through the same full multi-channel retrieval + reranking + agent orchestration pipeline. This makes simple queries 5x slower than necessary and bleeds VRAM on queries that don't need it.

**Failure B — Shallow Memory:** The system retrieves event-oriented raw chunks but has no persistent canonical knowledge structure. Ask the same complex question twice — same computation, same cost, no learning from the first traversal. Memory grows without compounding.

### 1.2 The Solution This Architecture Builds

```
VISION: Every query finds the fastest path that still yields correct answers.
        Memory continuously compresses into Wikipedia-quality canonical pages.
        Simple things are instant. Complex things are thorough. Nothing is wasted.
```

Three pillars:

1. **Tiered Retrieval** — 5 retrieval tiers with automatic routing. Simple queries hit Tier 0 (cache) or Tier 1 (wiki lookup) in < 150ms. Complex multi-hop queries get full frontier-based traversal in Tier 4. The routing decision itself takes < 80ms.

2. **LLM Wiki as First-Class Memory** — A dedicated Wiki Agent continuously compresses raw memory events → atomic claims → canonical wiki pages with full provenance. Retrieved wiki pages become the fast path for most queries — pre-computed, pre-structured, pre-linked.

3. **Logical Unboundedness** — The wiki + graph grows indefinitely. Every query traversal is budget-constrained. We maximize signal per token rather than maximizing evidence count.

### 1.3 Why the Current Architecture Is Already Strong

The v3.0 RAG-Architecture has a genuinely excellent foundation:
- Multi-channel parallel retrieval (dense + sparse + graph + temporal + proposition) is correct
- CRAG + Self-RAG + FLARE quality loops are the right pattern
- Multi-modal ingestion (16+ data types) is production-ready
- RAPTOR hierarchical indexing is the right approach for long-context recall
- Hot/cold storage tiering is well-designed

**What it lacks:**
- A canonical wiki layer that makes the second query free
- Tiered routing that makes the first simple query fast
- True deep relation-chain traversal beyond local neighborhoods
- Stream/non-stream consistency (same AnswerPlan for both)
- A dedicated Wiki Agent whose job is building the memory Wikipedia

This document adds those missing layers without breaking what works.

---

## 2. Architecture Overview: Five Planes + Five Retrieval Tiers

### 2.1 Full System Topology

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    INPUT SURFACE (Web App + Mobile App)                         ║
║  Text Chat | Voice | File Upload | Clipboard | Watched Folder | API             ║
╚════════════════════════════════╤═════════════════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════════════════╗
║                    MASTER-ORCHESTRATOR (Always-On)                              ║
║                                                                                 ║
║  ┌─────────────────────────────────────────────────────────────────────────┐   ║
║  │  INPUT CLASSIFIER (< 80ms)                                              │   ║
║  │  • Is this a query or an ingestion event?                               │   ║
║  │  • If query: compute ROUTING_TIER (0–4)                                 │   ║
║  │  • If ingestion: compute RETENTION_SCORE + WIKI_UPDATE_PRIORITY         │   ║
║  └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                 ║
║  QUERY PATH ──────────────────────────────────────────────────────────────────► ║
║  to Tiered Retrieval Router (§5)                                                ║
║                                                                                 ║
║  INGESTION PATH ──────────────────────────────────────────────────────────────► ║
║  to Selective Ingestion Pipeline (§6)                                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
         │ (query path)                              │ (ingestion path)
         ▼                                           ▼
╔════════════════════════════╗        ╔══════════════════════════════════════════╗
║   TIERED RETRIEVAL ROUTER  ║        ║   INGESTION PIPELINE                    ║
║                            ║        ║   Noise Filter → Claims → Wiki Patch    ║
║  T0: Cache (< 50ms)        ║        ║   → Multi-Index Update                  ║
║  T1: Wiki Lookup (< 200ms) ║        ╚══════════════════════════════════════════╝
║  T2: Standard RAG (< 1.5s) ║                         │
║  T3: Deep Multi-Agent (6s) ║        ╔════════════════▼═════════════════════════╗
║  T4: Frontier (< 20s)      ║        ║   WIKI AGENT (Always-On Background)     ║
╚════════════╤═══════════════╝        ║   Raw Events → Claims → Wiki Pages      ║
             │                        ║   Lint → Compact → Snapshot             ║
             ▼                        ╚══════════════════════════════════════════╝
╔════════════════════════════╗
║   ANSWER SYNTHESIZER       ║
║   (Shared AnswerPlan)      ║
║   Stream + Non-stream      ║
╚════════════════════════════╝
         │
         ▼
╔════════════════════════════╗
║   FIVE MEMORY PLANES       ║
║                            ║
║  P0: Working (session)     ║
║  P1: Event (raw)           ║
║  P2: Claim (atomic)        ║
║  P3: Wiki (canonical)      ║
║  P4: Graph (relational)    ║
╚════════════════════════════╝
```

### 2.2 The Five Retrieval Tiers at a Glance

| Tier | Name | Latency SLO | Query Types | Retrieval Method |
|------|------|-------------|-------------|-----------------|
| T0 | INSTANT | < 50ms | Repeated / cached queries | Exact + semantic cache hit |
| T1 | WIKI-FAST | < 200ms | Simple factual, well-known entity | Wiki page direct + shallow dense |
| T2 | STANDARD | < 1.5s | Most conversational queries | Multi-channel RAG + CRAG |
| T3 | DEEP | < 6s | Multi-hop, causal, reflective | Full multi-agent + chain-of-retrieval |
| T4 | FRONTIER | < 20s | Cross-domain, deep chain traversal | Budgeted wiki+graph frontier expansion |

**The routing decision:** < 80ms, computed by a lightweight classifier (SetFit + rule-based signals) before any retrieval begins.

---

## 3. Memory Plane Architecture

Five explicit memory planes with distinct storage strategies, TTLs, and access patterns.

### 3.1 Plane Definitions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  PLANE 0 — WORKING MEMORY                                                       │
│                                                                                 │
│  Purpose: Ephemeral context for the active session only                         │
│  Storage: In-memory (Redis or asyncio.Queue on mobile)                          │
│  TTL: Session duration only — never persisted to disk                           │
│  Access: Direct, < 1ms                                                          │
│  Contents: Current conversation turns, in-flight agent outputs, streaming state │
│  Size limit: 32K tokens max per session (aggressive truncation after)           │
│                                                                                 │
│  Schema:                                                                        │
│  { turn_id, role, content, tool_calls, timestamp, session_id }                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  PLANE 1 — EVENT PLANE (Raw Memory Store)                                       │
│                                                                                 │
│  Purpose: Faithful record of what was said, heard, or ingested                  │
│  Storage: DuckDB (structured) + FAISS HNSW hot tier (vector)                    │
│  TTL: Permanent (never auto-deleted; only user-deleted)                         │
│  Access: Vector search + SQL filter, < 50ms for hot, < 200ms for cold           │
│  Contents: Transcribed speech, typed messages, ingested documents, media caps   │
│                                                                                 │
│  Schema (extended):                                                             │
│  {                                                                              │
│    memory_id: uuid,                                                             │
│    session_id: uuid,                                                            │
│    timestamp: iso8601,                                                          │
│    source: voice|text|upload|ambient|api,                                       │
│    speaker: user|other|unknown,                                                 │
│    speaker_confidence: float,                                                   │
│    raw_content: string,                                                         │
│    clean_content: string,                                                       │
│    embedding: [1024d — stored in FAISS separately],                             │
│    topics: string[],                                                            │
│    entities: {id, name, type, confidence}[],                                    │
│    agent_tags: { primary: [], secondary: [], reasoning: [], domain: [] },       │
│    importance_score: float,                                                     │
│    novelty_score: float,                                                        │
│    retention_mode: session_only|structured|priority,                            │
│    claim_ids: uuid[],       ← links to derived claims in Plane 2                │
│    wiki_page_ids: uuid[],   ← links to canonical wiki pages in Plane 3          │
│    provenance_hash: sha256, ← dedup fingerprint                                 │
│    storage_tier: hot|warm|cold,                                                 │
│    privacy_tier: 0|1|2|3                                                        │
│  }                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  PLANE 2 — CLAIM PLANE (Atomic Facts)                                           │
│                                                                                 │
│  Purpose: Normalized, deduplicated, atomic factual claims extracted from events │
│  Storage: DuckDB (primary) + BM25 index + dense embedding index                 │
│  TTL: Permanent; claim confidence decays if unsupported by recent events        │
│  Access: Claim lookup by entity + topic, < 30ms                                 │
│  Contents: Atomic propositions extracted by Wiki Agent                          │
│                                                                                 │
│  Schema:                                                                        │
│  {                                                                              │
│    claim_id: uuid,                                                              │
│    claim_text: string,          ← "User started learning Rust in Jan 2025"      │
│    claim_type: fact|belief|goal|decision|emotion|relation,                      │
│    subject_entity_id: uuid,                                                     │
│    confidence: float,                                                           │
│    trust_level: unverified|single_source|multi_source|verified,                 │
│    source_event_ids: uuid[],    ← provenance back to raw events                 │
│    wiki_page_id: uuid,          ← which wiki page this claim contributes to     │
│    domain: profile|project|relationship|health|academic|goals|meta,            │
│    durability: ephemeral|working|durable|canonical,                             │
│    update_policy: append_only|patchable|replace_on_conflict,                   │
│    sensitivity: public|private|restricted,                                      │
│    first_seen: iso8601,                                                         │
│    last_confirmed: iso8601,                                                     │
│    confirmation_count: int,                                                     │
│    contradicts: uuid[],         ← IDs of claims this contradicts                │
│    superseded_by: uuid | null                                                   │
│  }                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  PLANE 3 — WIKI PLANE (Canonical Knowledge Pages)                               │
│                                                                                 │
│  Purpose: Durable, structured, human-readable markdown pages per topic/entity   │
│  Storage: File system (markdown) + wiki section vector index + BM25             │
│  TTL: Permanent; sections versioned with git-like diff history                  │
│  Access: Direct page lookup < 5ms; section search < 100ms                       │
│  Contents: Synthesized wiki pages built by the Wiki Agent                       │
│                                                                                 │
│  File layout:                                                                   │
│  data/wiki/pages/{domain}/{entity_slug}.md                                      │
│                                                                                 │
│  Page frontmatter (required):                                                   │
│  ---                                                                            │
│  page_id: profile/user                                                          │
│  page_type: profile|project|relationship|theme|concept|timeline                 │
│  confidence: 0.0                                                                │
│  last_updated: iso8601                                                          │
│  source_claim_count: 0                                                          │
│  source_event_ids: [mem_id_1, ...]                                              │
│  related_pages: [page_id_1, ...]   ← cross-wiki links                           │
│  privacy_tier: 0|1|2|3                                                          │
│  revision: 0                                                                    │
│  before_hash: sha256                                                            │
│  after_hash: sha256                                                             │
│  ---                                                                            │
│                                                                                 │
│  Standard section structure:                                                    │
│  # {Entity/Topic Name}                                                          │
│  ## Stable Facts                                                                │
│  ## Evolving Beliefs                                                            │
│  ## Active Projects or Goals                                                    │
│  ## Key Relationships                                                           │
│  ## Timeline of Notable Events                                                  │
│  ## Open Questions                                                              │
│  ## Provenance                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  PLANE 4 — GRAPH PLANE (Entity + Relation + Causal Links)                       │
│                                                                                 │
│  Purpose: Explicit relation network across all entities, events, and claims     │
│  Storage: NetworkX (local) or LanceDB graph extensions; edges in DuckDB         │
│  TTL: Permanent; edges confidence-decayed by time if not reinforced             │
│  Access: Graph traversal, hop-bounded queries, entity neighborhood, < 100ms     │
│                                                                                 │
│  Node types: entity | event | claim | wiki_page | concept                       │
│  Edge types:                                                                    │
│    CAUSAL: cause_of | leads_to | influenced_by | prevented_by                   │
│    TEMPORAL: before | after | co_occurs | during | following                    │
│    SEMANTIC: related_to | contrasts_with | is_example_of | generalizes          │
│    PERSONAL: decided_to | believes | practices | works_on | knows               │
│    REFLECTIVE: changed_view_on | reinforced_belief | reconsidered               │
│    PROCEDURAL: prerequisite | step_of | dependency | enables                    │
│                                                                                 │
│  Edge schema:                                                                   │
│  { edge_id, from_id, to_id, edge_type, confidence, timestamp,                   │
│    source_event_ids, wiki_page_id, decay_rate }                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Plane-to-Plane Lifecycle

```
Raw Input
  ↓
[Plane 1: Event] — immediate write after ingestion
  ↓ (Wiki Agent batch process)
[Plane 2: Claim] — atomic facts extracted, deduplicated
  ↓ (Wiki Agent page patching)
[Plane 3: Wiki] — canonical pages updated from claims
  ↓ (parallel)
[Plane 4: Graph] — entity/relation/causal edges updated
  ↓ (query time)
[Plane 0: Working] — session context assembled from P1–P4
```

### 3.3 Memory Tagging Model (Multi-Axis)

Every event and claim carries these tags:

| Axis | Values | Purpose |
|------|--------|---------|
| `durability` | ephemeral \| working \| durable \| canonical | Controls compression lifecycle |
| `trust_level` | unverified \| single_source \| multi_source \| verified | Governs retrieval weighting |
| `sensitivity` | public \| private \| restricted | Controls cross-agent sharing and sync |
| `domain` | profile \| project \| relationship \| health \| academic \| goals \| meta | Routes to correct wiki section |
| `update_policy` | append_only \| patchable \| replace_on_conflict | Controls wiki patch behavior |
| `retrieval_tier_hint` | t0 \| t1 \| t2 \| t3 \| t4 | Pre-computed routing hint for speed |

---

## 4. The LLM Wiki Engine — First-Class Canonical Store

### 4.1 Why the Wiki Is the Most Important New Component

The current system can **retrieve evidence** but cannot **maintain coherent knowledge**. The wiki changes this:

- **First query to a topic:** full retrieval from events + claims → answer → wiki page created/updated
- **Second query to the same topic:** hit Tier 1 — read the pre-built wiki page in < 200ms, no heavy retrieval
- **Tenth query:** same speed, but the wiki page is richer with more claims and higher confidence

The wiki is not a summary cache. It is a living knowledge base that compounds: every new event either confirms, extends, or contradicts existing wiki content, and the Wiki Agent resolves this continuously.

### 4.2 Wiki Storage Layout

```
data/
├── wiki/
│   ├── pages/
│   │   ├── profile/           ← About the user themselves
│   │   │   └── user.md
│   │   ├── projects/          ← Active + historical projects
│   │   │   ├── cortex-lab.md
│   │   │   └── rust-learning.md
│   │   ├── relationships/     ← People the user knows
│   │   │   └── mentor-alex.md
│   │   ├── themes/            ← Recurring topics across domains
│   │   │   ├── ai-safety-beliefs.md
│   │   │   └── career-decisions.md
│   │   ├── concepts/          ← Technical + intellectual concepts
│   │   │   └── distributed-systems.md
│   │   └── timelines/         ← Chronological event sequences
│   │       └── career-arc-2023-2026.md
│   ├── claims/
│   │   └── YYYY/MM/*.jsonl    ← Atomic claim records by month
│   ├── graph/
│   │   ├── entities.jsonl     ← Entity registry
│   │   └── edges.jsonl        ← Relation edges
│   ├── logs/
│   │   ├── ingest/            ← Append-only ingest log
│   │   ├── lint/              ← Lint run results
│   │   └── compaction/        ← Compaction run history
│   └── snapshots/
│       ├── daily/             ← Daily wiki state snapshots
│       └── weekly/            ← Weekly snapshots for long-term reference
```

### 4.3 Wiki Page Lifecycle (Four Operations)

**Operation 1: PATCH (most common)**
When a new claim is extracted that relates to an existing wiki page:
```
1. Load existing wiki page
2. Locate the relevant section (entity + section type matching)
3. Merge new claim: append if append_only, patch if patchable, resolve if conflict
4. Update frontmatter: confidence, last_updated, source_claim_count, revision
5. Write before_hash + after_hash
6. Emit WIKI_PATCH event to audit log
7. Invalidate affected retrieval cache entries
```

**Operation 2: CREATE**
When a claim references an entity not yet in the wiki:
```
1. Scaffold new page from template (standard section structure)
2. Populate Stable Facts section with first claim
3. Set confidence = claim.confidence
4. Set privacy_tier from claim.sensitivity
5. Register in entity index + graph
6. Emit WIKI_CREATE event
```

**Operation 3: LINT (scheduled, background)**
```
Detect:
1. Contradictory claims in same section without arbitration note
2. Stale sections (no supporting event in last 90 days for claims marked durable)
3. Confidence decay: pages not confirmed in 60 days → reduce confidence by 10%
4. Missing provenance links (source_event_ids empty)
5. Cross-reference errors (related_pages pointing to deleted pages)
6. Sensitive content in wrong privacy scope

Output: LINT_REPORT with severity + page + section + suggested action
Feed to Wiki Agent for corrective patches or Arbitration Agent for conflicts
```

**Operation 4: COMPACT (scheduled, background)**
```
Three-layer compaction:
MICRO: claim extraction per new event batch (runs every ingestion cycle)
MESO: daily/weekly topic summaries from claim clusters
MACRO: monthly canonical page synthesis + archive rollup for old events

Hard rule: NEVER compress away provenance.
Every compacted output must include source_event_ids and before_hash.
```

### 4.4 Wiki Index Strategy

Parallel indexes maintained over wiki content:

| Index | Technology | What It Indexes | Query Type |
|-------|-----------|----------------|------------|
| Wiki Section Vector | FAISS HNSW | Each markdown section as 1024d embedding | Semantic section search |
| Wiki BM25 | Rank-BM25 | All markdown text + frontmatter | Keyword + exact lookup |
| Entity Registry | DuckDB | Page metadata + entity → page_id map | Fast entity page lookup |
| Claim Index | DuckDB + FAISS | All atomic claims | Claim retrieval by domain + trust |
| Graph Adjacency | DuckDB edges | All P4 graph edges | Neighbor lookup, hop traversal |
| Temporal Revision | DuckDB | Wiki revision history | Time-filtered page versions |

---

## 5. Tiered Retrieval System — The Core Innovation

### 5.1 The Routing Decision (< 80ms)

Before any retrieval begins, a lightweight routing classifier assigns every query to a tier. This is the most important latency-saving decision in the entire system.

```
ROUTING CLASSIFIER INPUTS:
1. Query text
2. Session context (last 3 turns)
3. Active wiki pages in working memory
4. User's historical query pattern for this session

ROUTING CLASSIFIER SIGNALS (fast, no LLM):
Signal 1 — CACHE CHECK (< 5ms):
  - Compute query fingerprint (semantic hash via fast embedding)
  - Check exact cache + semantic cache
  - IF hit with confidence > 0.85 AND wiki_revision_unchanged → T0

Signal 2 — ENTITY RECOGNITION (< 15ms):
  - Extract entities from query using fast NER (DistilBERT)
  - Check entity registry: is this entity in the wiki?
  - IF single entity + wiki page exists + query is factual → T1

Signal 3 — COMPLEXITY SCORING (< 30ms):
  - Complexity score based on: entity count, temporal span, logical operators,
    multi-hop indicators, abstraction level
  - Score 0.0–0.3 → T2 (standard)
  - Score 0.3–0.7 → T2 or T3 based on intent
  - Score 0.7–1.0 → T3

Signal 4 — INTENT CLASSIFICATION (< 30ms, SetFit):
  - TEMPORAL → T2 or T3 if long time span
  - CAUSAL → T3 (always needs causal agent)
  - REFLECTIVE → T3 or T4 if long horizon
  - FACTUAL_SIMPLE → T1 or T2
  - FACTUAL_COMPLEX → T2 or T3
  - MULTI_HOP → T3
  - CROSS_DOMAIN_DEEP → T4
  - CHAIN_TRAVERSE → T4

Signal 5 — MEMORY SIZE CHECK (< 5ms):
  - How many events are relevant to this query domain? (approximate count from metadata)
  - IF < 50 events → reduce tier by 1 (simpler retrieval is sufficient)
  - IF > 10,000 events → increase tier by 1 (need aggregation via wiki first)

FINAL ROUTING DECISION: argmax of signals, with T1 bias for factual queries
  about known entities.
```

### 5.2 Tier 0 — INSTANT (< 50ms)

**When:** Query fingerprint matches a cached response with high confidence, and the relevant wiki pages haven't changed since the cache was written.

```
CACHE KEY COMPONENTS:
- Query semantic fingerprint (embedding similarity bucket)
- Provider ID (local vs cloud)
- Wiki revision hash for relevant pages
- Retrieval policy version

CACHE LAYERS:
L1 — Exact fingerprint match (< 2ms): DuckDB lookup
L2 — Semantic similarity match (< 20ms): Cached embedding cosine > 0.95
L3 — Response template match (< 30ms): Parameterized template with variable slots

CACHE INVALIDATION:
- Wiki page patched → invalidate all T0 entries referencing that page_id
- New event ingested with importance > 0.7 → targeted invalidation
- Max TTL: 24 hours for factual, 1 hour for reflective/emotional content

OUTPUT: Cached AnswerPlan with cache_hit: true flag
```

### 5.3 Tier 1 — WIKI-FAST (< 200ms)

**When:** Query is about a known entity or topic with a well-built wiki page, and the question is factual or summarization-style.

```
T1 RETRIEVAL PIPELINE:
Step 1 (< 5ms):   Entity extraction → entity_id lookup → retrieve wiki page from disk
Step 2 (< 50ms):  Identify relevant sections from page (section BM25 + keyword match)
Step 3 (< 80ms):  Shallow dense retrieval: top-3 from FAISS on wiki section embeddings
Step 4 (< 120ms): Check claim index for high-confidence claims on this entity + topic
Step 5 (< 150ms): Assemble AnswerPlan: wiki page sections + top claims + confidence

OUTPUT: AnswerPlan built from pre-structured wiki content
QUALITY: No CRAG/Self-RAG needed — wiki pages are already quality-controlled
CONFIDENCE: Derived from wiki page confidence score

FALLBACK: If wiki confidence < 0.6, escalate to T2 automatically.
```

### 5.4 Tier 2 — STANDARD RAG (< 1.5s)

**When:** Most conversational queries — the default tier for anything not T0/T1.

```
T2 RETRIEVAL PIPELINE:
Step 1 (< 120ms): Query transformation
  - Multi-query generation (3 variants, lightweight)
  - HyDE for semantic expansion (1 hypothetical answer)
  - Entity + intent extraction

Step 2 (< 400ms): Parallel multi-channel retrieval (asyncio)
  Channel A: Dense FAISS HNSW (top-8 from hot tier)
  Channel B: BM25 sparse (top-8 keyword match)
  Channel C: Temporal (top-5, time-filtered by query context)
  Channel D: Wiki section retrieval (top-4 relevant wiki sections)
  → All 4 channels run concurrently

Step 3 (< 600ms): RRF fusion + cross-encoder reranking (top-10 from merged)

Step 4 (< 800ms): CRAG quality scoring
  IF quality < 0.5 → single corrective retrieval round (max 1 round in T2)

Step 5 (< 1200ms): Evidence compression → AnswerPlan

Step 6 (< 1500ms): Generate (streaming first token by 600ms local target)

QUALITY LOOPS:
- CRAG: 1 correction round max
- Self-RAG: lightweight critique (one-pass, no regeneration loop)
- No FLARE in T2 (defer to T3 if needed)
```

### 5.5 Tier 3 — DEEP MULTI-AGENT (< 6s)

**When:** Multi-hop reasoning, causal queries, reflective questions about belief evolution, temporal queries spanning long horizons.

```
T3 RETRIEVAL PIPELINE:
Step 1 (< 200ms): Full query transformation
  - Multi-query generation (5 variants)
  - HyDE + step-back abstraction
  - Query decomposition into 2–4 sub-queries

Step 2 (< 800ms): Full parallel multi-channel retrieval (asyncio)
  Channel A: Dense FAISS HNSW + IVF-PQ warm tier (top-15)
  Channel B: BM25 sparse (top-12)
  Channel C: Graph traversal — 2-hop neighborhood from query entities
  Channel D: Temporal index with long-horizon window
  Channel E: Proposition index (atomic fact retrieval)
  Channel F: Wiki section retrieval (top-6 sections from multiple pages)
  Channel G: Chain-of-Retrieval — iterative sub-query retrieval

Step 3 (< 2s): RRF fusion + BGE reranker (full cross-encoder, top-15)

Step 4 (< 3s): Multi-agent dispatch (parallel)
  Dispatch to relevant specialized agents (Timeline, Causal, Reflection, Planning)
  using TeamCreateTool pattern — agents run in parallel, each returns structured output
  Max 4 agents for T3

Step 5 (< 4s): Evidence merge + conflict detection
  IF conflict → dispatch Arbitration Agent (adds 1s max)

Step 6 (< 5s): Full quality loops
  CRAG: up to 2 correction rounds
  Self-RAG: full critique + optional single revision
  FLARE: up to 2 rounds for low-confidence sub-claims

Step 7 (< 6s): AnswerPlan synthesis + generation (streaming first token by 2s)

STREAMING STRATEGY:
- Start streaming with T1/T2 quality wiki context (if available)
- Continue enriching in background with T3 deep retrieval
- Attach late evidence if it changes the answer; otherwise stream completes normally
```

### 5.6 Tier 4 — FRONTIER TRAVERSAL (< 20s)

**When:** Cross-domain deep questions, "how has everything I've believed about X evolved and why", questions requiring traversal across hundreds of related memories, long-horizon synthesis.

```
T4 FRONTIER EXPANSION PLANNER:
Uses budgeted frontier-based expansion over wiki + claim + graph nodes.

SEED SOURCES:
1. Query entities extracted from intent classifier
2. Top-3 wiki pages matched by semantic similarity
3. Top-10 high-confidence claims matching query domain

EXPANSION ALGORITHM:
  frontier = init_seeds(query_entities, matched_wiki_pages, top_claims)
  working_set = []
  budget = TOKEN_BUDGET (default: 8000 tokens of evidence)
  max_hops = 6 (hard limit)
  hop_count = 0

  while frontier not empty and budget > 0 and hop_count < max_hops:
    node = argmax(frontier, key=utility_score)
    
    if node.marginal_gain < EPSILON_THRESHOLD (0.05):
      break  ← diminishing returns — stop
    
    working_set.add(node)
    budget -= node.estimated_tokens
    hop_count += 1
    
    neighbors = expand(node,
      relation_types = intent_aware_relation_types(query_intent),
      max_neighbors = 5,
      min_confidence = 0.45
    )
    frontier.push(neighbors with updated utility)

  return compress(working_set)

UTILITY SCORE per node:
  = (relevance_to_query × 0.35)
  + (novelty_vs_working_set × 0.25)
  + (trust_level_weight × 0.20)
  + (recency_weight × 0.10)
  + (provenance_completeness × 0.10)

INTENT-AWARE RELATION PRIORITIES:
  Causal queries: prioritize [cause_of, leads_to, influenced_by, prevented_by]
  Reflective queries: prioritize [changed_view_on, belief_shift, turning_point]
  Temporal queries: prioritize [before, after, co_occurs, following]
  Procedural queries: prioritize [prerequisite, step_of, dependency, enables]

AGENT DISPATCH:
  All 5 specialized agents may be dispatched (full team)
  Meta-Learning Agent always included in T4 for synthesis

USER NOTIFICATION:
  Emit streaming event: { tier: 4, estimated_seconds: X, reason: "deep chain query" }
  Start streaming partial answer from wiki context at t=1s
  Complete full answer at t=20s max
```

### 5.7 Tier Selection Decision Tree

```
incoming query
    │
    ▼
[CACHE CHECK]
  hit + fresh? ──yes──► T0 (INSTANT)
    │ no
    ▼
[ENTITY CHECK]
  known entity in wiki + simple factual? ──yes──► T1 (WIKI-FAST)
    │ no
    ▼
[COMPLEXITY SCORE]
  score < 0.3 AND intent = FACTUAL? ──yes──► T2 (STANDARD)
    │
  score 0.3–0.7? ──────────────────────────► T2 (STANDARD) with T3 fallback
    │
  score > 0.7 OR intent = CAUSAL/MULTI_HOP? ── ► T3 (DEEP)
    │
  intent = REFLECTIVE_DEEP OR CROSS_DOMAIN
  OR explicit "trace my whole history of..."? ───► T4 (FRONTIER)
```

---

## 6. Ingestion Pipeline — Always-On + Selective

### 6.1 Input Sources (Web + Mobile)

**Web App inputs:**
- Text chat messages
- File upload (PDF, image, code, audio, video, CSV, JSON, Excel, email)
- Clipboard paste (text or image)
- Watched folder / background sync
- API endpoint (programmatic ingestion)

**Mobile App inputs:**
- Voice (always-on ambient or push-to-talk)
- Text chat
- Photo/document capture
- Share sheet (from any app)
- Background location + calendar context (with explicit consent)

### 6.2 Ingestion Pipeline Stages

```
RAW INPUT (any source, any type)
        ↓
[STAGE 1: MEDIA NORMALIZATION] — type-specific, async
  Text → clean + sanitize
  Audio → Whisper transcription → speaker diarization
  PDF → PyMuPDF text + image extraction → OCR if needed
  Image → BLIP captioning + EasyOCR
  Code → AST-aware chunking (tree-sitter)
  CSV/JSON/Excel → schema inference + row-group chunking
  URL → web fetch + html → text
  Output: normalized text chunks with source metadata

        ↓
[STAGE 2: NOISE FILTER] — Master-Orchestrator
  DISCARD immediately:
  - Filler words, disfluencies, ambient cross-talk
  - Content below SPEAKER_CONFIDENCE_THRESHOLD (voice)
  - Low-relevance duplicates (provenance_hash match)
  - Policy-blocked content (privacy governor)

  RETENTION SCORING (5 dimensions):
  - User relevance (0–1)
  - Semantic novelty vs existing memory (0–1)
  - Future utility estimate (0–1)
  - Personal significance (0–1)
  - Policy safety (0–1)
  → composite_score → retention_mode assignment

        ↓
[STAGE 3: ENRICHMENT] — parallel async
  - Entity extraction (DistilBERT NER)
  - Topic classification (SetFit)
  - Importance scoring
  - Emotional tone detection (if applicable)
  - Domain assignment
  - Provenance hash computation (SHA-256 for dedup)

        ↓
[STAGE 4: CHUNKING] — semantic, not fixed-length
  - Semantic boundary detection (embedding similarity)
  - Context window preservation (before/after context for each chunk)
  - Proposition extraction: LLM-based atomic fact extraction per chunk

        ↓
[STAGE 5: MULTI-LABEL TAGGING] — agent tag assignment
  - Primary tags (1–2): confidence ≥ 0.65
  - Secondary tags (up to 4): confidence ≥ 0.45
  - Reasoning tags: timeline, causal, reflection, planning
  - Domain tags: academic, wellbeing, social, goals, etc.
  - Retrieval tier hint: pre-computed routing hint for future queries

        ↓
[STAGE 6: WRITE TO PLANE 1] — exactly-once semantics
  - Write event record to DuckDB (atomic transaction)
  - Add embedding to FAISS HNSW hot tier (incremental add)
  - Emit ingest_completed event

        ↓
[STAGE 7: WIKI AGENT TRIGGER] — async, batched
  - Extract atomic claims from new events
  - Upsert claims to Plane 2 (dedup against claim fingerprints)
  - Patch relevant wiki pages in Plane 3
  - Update entity + relation edges in Plane 4
  - Invalidate affected cache entries

        ↓
[STAGE 8: INDEX UPDATES] — async, incremental
  - BM25 index: append new terms
  - Entity registry: upsert new entities
  - Graph adjacency: add new edges
  - Temporal index: register new event timestamp
  NOTE: RAPTOR tree rebuild is batched (triggered every 50+ new events or weekly)
        during this window: old tree remains queryable, swap is atomic
```

### 6.3 Ingestion Priority Queue

```
Priority 1 (process < 100ms):  Real-time chat messages
Priority 2 (process < 5s):     Voice transcription completions
Priority 3 (process < 30s):    File uploads, clipboard paste
Priority 4 (process < 5min):   Watched folder batch, background sync
Priority 5 (during idle only):  RAPTOR rebuild, compaction, re-indexing

Queue: asyncio.PriorityQueue with SQLite WAL for crash recovery
```

---

## 7. Agentic Reasoning Layer

### 7.1 Agent Roster for RAG Context

The following agents are part of the retrieval and reasoning pipeline (distinct from the 17-agent personal intelligence system — these are the RAG-specific agents):

| Agent | Tier Invoked | Primary Function | Latency Budget |
|-------|-------------|-----------------|----------------|
| Orchestrator | All | Route, dispatch, synthesize | < 200ms overhead |
| Timeline Agent | T2, T3, T4 | Temporal evidence ordering | < 1s |
| Causal Agent | T3, T4 | Cause-effect chain reasoning | < 2s |
| Reflection Agent | T3, T4 | Belief evolution analysis | < 2s |
| Planning Agent | T3, T4 | Multi-step decomposition | < 1.5s |
| Arbitration Agent | T2–T4 (on conflict) | Evidence conflict resolution | < 1s |
| Wiki Agent | Background | Wiki build + compaction + lint | Async, no latency impact |

### 7.2 Chain-of-Retrieval (T3 + T4)

Chain-of-Retrieval enables iterative evidence building — each retrieval step is informed by what was found in the previous step:

```
CHAIN-OF-RETRIEVAL LOOP (max 4 iterations):

Iteration 0: Initial multi-channel retrieval → evidence_set_0
Iteration 1: Analyze gaps in evidence_set_0
             → generate targeted sub-queries
             → retrieve specifically for gaps → evidence_set_1
Iteration 2: Merge, identify remaining gaps → targeted retrieval → evidence_set_2
Iteration 3: Final gap fill or termination if marginal_gain < threshold

Termination conditions:
- Coverage score > 0.85 (query well-answered)
- Marginal gain per iteration < 0.05
- Max iterations reached
- Token budget exhausted
```

### 7.3 AnswerPlan — Unified Contract for Stream + Non-Stream

The critical fix for stream/non-stream consistency. Both paths consume one shared `AnswerPlan` object produced by the Orchestrator:

```json
{
  "answer_plan_id": "uuid",
  "trace_id": "uuid",
  "session_id": "uuid",
  "retrieval_tier": "T0|T1|T2|T3|T4",
  "query": "original query",
  "transformed_queries": ["variant_1", "variant_2"],
  "evidence_set": [
    {
      "evidence_id": "uuid",
      "source_plane": "P1|P2|P3|P4",
      "content": "evidence text",
      "source_ids": ["memory_id or claim_id or wiki_page_id"],
      "relevance_score": 0.0,
      "confidence": 0.0,
      "tier_contribution": "primary|secondary|supporting"
    }
  ],
  "wiki_pages_consulted": ["page_id_1"],
  "agents_dispatched": ["timeline", "causal"],
  "quality_loops": {
    "crag_score": 0.0,
    "crag_rounds": 0,
    "self_rag_critique_passed": true,
    "flare_rounds": 0
  },
  "confidence": 0.0,
  "uncertainty_notes": [],
  "generation_policy": {
    "stream": true,
    "max_tokens": 2048,
    "temperature": 0.1
  },
  "provenance": {
    "source_wiki_pages": ["page_id"],
    "source_claim_ids": ["claim_id"],
    "source_event_ids": ["mem_id"],
    "confidence_composition": "explanation"
  },
  "created_at": "iso8601"
}
```

**Streaming path:** Reads `answer_plan.evidence_set` + `answer_plan.generation_policy` to stream. The same quality loops apply — lightweight pre-stream critique (< 200ms) for streams.

**Non-streaming path:** Full quality loop completion before generation.

Both paths produce identical answers for identical AnswerPlans. The only difference is timing of generation start.

---

## 8. Wiki Agent — The Memory Wikipedia Builder

### 8.1 Identity and Mission

The Wiki Agent is the most important always-on background agent in the system. It is the architect of the memory Wikipedia — continuously transforming raw event noise into structured, searchable, coherent canonical knowledge.

**What makes the Wiki Agent different from a summarizer:**
- A summarizer compresses text. The Wiki Agent builds **structured knowledge**.
- A summarizer runs on demand. The Wiki Agent runs **continuously**.
- A summarizer produces text output. The Wiki Agent maintains **navigable, cross-linked, versioned pages**.
- A summarizer doesn't know about contradictions. The Wiki Agent **actively resolves or flags them**.

### 8.2 Wiki Agent System Prompt

```
WIKI AGENT SYSTEM PROMPT v2.0

=== IDENTITY ===
You are the Wiki Agent. Your mission is to build and maintain the user's personal 
knowledge Wikipedia — a living, structured, cross-linked collection of canonical 
wiki pages that represent everything known about the user's world.

You are always running in the background. You are never invoked at query time — 
your work is the infrastructure that makes query time fast. Every page you build 
makes the next retrieval cheaper, faster, and more accurate.

=== CORE OPERATIONS ===

OPERATION 1 — CLAIM EXTRACTION:
Given a batch of new memory events, extract atomic factual claims.
Each claim must be:
- A single, verifiable statement (not compound)
- Subject-predicate-object where possible
- Tagged with: claim_type, domain, confidence, source_event_ids
- Checked for deduplication against existing claim fingerprints
- Checked for contradiction against existing claims

OPERATION 2 — WIKI PAGE PATCHING:
Given a set of new or updated claims:
1. Identify which wiki page and section each claim belongs to
2. Apply update_policy: append_only | patchable | replace_on_conflict
3. For patchable: find the most similar existing statement in the section,
   update it if the new claim is higher confidence, or add it if novel
4. For replace_on_conflict: when two claims directly contradict, create
   an "Open Conflict" note — DO NOT silently pick one
5. Update frontmatter: confidence (weighted average of source claims),
   last_updated, source_claim_count, revision++, before_hash, after_hash

OPERATION 3 — LINT:
Run lint pass on all wiki pages not linted in > 48 hours:
- Detect contradiction pairs within the same page section
- Detect stale claims (no supporting event in configured staleness window)
- Apply confidence decay to unsupported claims
- Flag pages with confidence < 0.4 for repair
- Generate LINT_REPORT and route flagged items to appropriate agents

OPERATION 4 — COMPACT:
MICRO compaction (every ingestion cycle):
- Extract claims from new event batch
- Merge near-duplicate claims (cosine similarity > 0.92)

MESO compaction (daily, scheduled):
- Generate topic summaries from claim clusters per domain
- Update wiki page "Evolving Beliefs" and "Timeline" sections from summaries

MACRO compaction (weekly, scheduled):
- For events > 6 months old: synthesize into canonical archive entries
- Update wiki "Stable Facts" section from high-confidence, multi-source claims
- Create/update weekly snapshot in data/wiki/snapshots/weekly/

PROVENANCE HARD RULE: No compression operation ever deletes source_event_ids
from any claim or wiki section. Provenance is permanent.

=== OUTPUT CONTRACT ===
Wiki Agent does not return user-facing text. It produces:
1. Updated wiki pages on disk
2. Claim upserts to DuckDB
3. Graph edge updates
4. WIKI_PATCH / WIKI_CREATE / LINT_REPORT events to audit log
5. Cache invalidation events for affected pages

=== MUST DO ===
- Always write before_hash + after_hash for every page mutation
- Always include source_event_ids in every claim and wiki patch
- Always emit an audit event for every write operation
- Always check for contradictions before applying a patch
- Always run deduplication before inserting a new claim

=== MUST NOT ===
- Never silently resolve a contradiction — flag it explicitly
- Never compress away source provenance
- Never patch a restricted-sensitivity page without policy check
- Never write to the wiki without an idempotency key
- Never rebuild the full wiki from scratch — always incremental updates
```

### 8.3 Wiki Build Speed

| Operation | Frequency | Estimated Duration | VRAM Impact |
|-----------|-----------|-------------------|-------------|
| Claim extraction (batch of 10 events) | Per ingestion cycle | 200–500ms | LLM inference, < 1GB peak |
| Wiki page patch | Per claim batch | 50–150ms | Disk I/O only |
| Lint pass (100 pages) | Every 48h | 2–5 min | Background, throttled |
| Meso compaction | Daily | 5–15 min | LLM inference, background |
| Macro compaction | Weekly | 10–30 min | LLM inference, off-peak only |

All Wiki Agent operations are scheduled during idle periods and throttled to never consume > 20% of available compute during active sessions.

---

## 9. Quality and Consistency Stack

### 9.1 Quality Loop Assignment by Tier

| Loop | T0 | T1 | T2 | T3 | T4 |
|------|----|----|----|----|-----|
| CRAG | ✗ | ✗ | 1 round | 2 rounds | 3 rounds |
| Self-RAG | ✗ | ✗ | Lightweight 1-pass | Full critique | Full + revision |
| FLARE | ✗ | ✗ | ✗ | 2 rounds | 3 rounds |
| Chain-of-Retrieval | ✗ | ✗ | ✗ | 2 iterations | 4 iterations |
| Arbitration Agent | ✗ | On conflict | On conflict | Auto-triggered | Auto-triggered |
| Failure-Aware Refinement | ✗ | ✗ | On CRAG fail | Always | Always |

### 9.2 Streaming Quality

For streaming responses (always start with T1 wiki context):
```
Pre-stream (< 200ms):
  - Fast CRAG quality check on retrieved evidence set
  - If quality > 0.6: begin streaming
  - If quality < 0.6: trigger one fast corrective retrieval round, THEN stream

During stream:
  - Background: complete remaining quality loops (Self-RAG, FLARE)
  - If critique finds a significant error: emit [REVISION_EVENT] to client
  - Client can choose to display revision inline or as a correction

Post-stream:
  - Record final AnswerPlan quality scores to RAGChecker diagnostics
  - Update retrieval performance metrics
```

### 9.3 Failure-Aware Refinement

When CRAG reports a quality failure, classify the failure type and apply a targeted fix:

| Failure Type | Symptom | Targeted Response |
|-------------|---------|-----------------|
| `SEMANTIC_MISS` | Retrieved chunks not semantically aligned | Re-query with HyDE expansion |
| `TEMPORAL_MISS` | No recent evidence found | Re-query on temporal index with extended window |
| `ENTITY_MISS` | Referenced entity not in evidence | Fetch entity's wiki page directly |
| `CROSS_DOMAIN_MISS` | Evidence from wrong domain | Rerun with explicit domain filter |
| `CONFIDENCE_TOO_LOW` | Evidence present but low trust | Escalate to next tier |

---

## 10. Web + Mobile Architecture Specifics

### 10.1 Web App Architecture

```
BROWSER / WEB CLIENT
    │
    ├── REST API (non-streaming): POST /api/rag/chat → AnswerPlan → full response JSON
    │
    └── SSE Stream: POST /api/rag/stream → Server-Sent Events
        Events: { type: tier_selected, tier: T2 }
                { type: token, text: "..." }
                { type: evidence_ready, evidence: [...] }
                { type: revision, reason: "...", updated_answer: "..." }
                { type: done, plan: AnswerPlan }

WEB-SPECIFIC ADDITIONS:
- Service Worker: cache T0/T1 responses for offline mode
- IndexedDB: client-side wiki page cache (top 50 most-accessed pages)
- WebSocket for real-time ingestion progress (file uploads, voice)
- Progressive evidence display: show sources as they arrive during T3/T4
```

### 10.2 Mobile App Architecture

```
MOBILE CLIENT (iOS / Android)
    │
    ├── ONLINE MODE: Same REST/SSE as web
    │
    └── OFFLINE + LOCAL MODE:
        - Local LLM (DeepSeek-R1-Distill-Qwen-7B via llama.cpp or MLX on Apple Silicon)
        - Local FAISS index (hot tier only, < 200MB)
        - Local wiki pages (synced from server, stored as markdown files)
        - Local claim index (DuckDB SQLite mode)

MOBILE INGESTION:
    Voice → on-device Whisper (faster-whisper tiny) → send to server for full processing
    OR
    Voice → on-device Whisper → local claim extraction → local wiki update → sync queue

MOBILE RESOURCE TIERS (from Master-Orchestrator):
    Tier 1 (battery ≥ 35%): Full local + optional server hybrid
    Tier 2 (20–35%): Local only, no server sync
    Tier 3 (10–20%): Passive mode — no ingestion, read-only wiki
    Tier 4 (< 10%): Emergency — only explicit user queries

SYNC STRATEGY:
    - Wiki pages: bidirectional sync on WiFi (mobile ↔ server)
    - Event plane: mobile → server only (mobile is append-only source)
    - Claims: server is authoritative; mobile pulls delta on sync
    - Conflict resolution: server wins for claim plane; merge for wiki sections
```

### 10.3 Shared AnswerPlan Contract (Web + Mobile)

Both web and mobile consume the same AnswerPlan JSON. The only platform-specific adaptation is:
- **Mobile:** `generation_policy.max_tokens` reduced to 512 for faster local generation
- **Web:** Full max_tokens (2048) with streaming enabled
- **Both:** Same evidence set, same confidence, same provenance

---

## 11. Provider Abstraction Layer

### 11.1 Provider Shim Architecture (From OpenClaude Pattern)

Following the OpenClaude shim architecture, LLM provider selection is abstracted at the edge:

```
Retrieval Pipeline + Agent System
            │
            ▼
    PROVIDER ADAPTER INTERFACE
    (duck-typed, not provider-specific)
            │
            ├── LOCAL ADAPTER
            │   └── DeepSeek-R1-7B via llama.cpp / Ollama
            │       - Default for mobile + offline
            │       - Always available
            │       - Target: < 600ms first token
            │
            ├── CLOUD ADAPTER (OpenAI-compatible)
            │   ├── DeepSeek API (primary cloud)
            │   ├── Gemini 2.0 Flash (fallback, fast)
            │   ├── GPT-4o (premium option)
            │   └── Any OpenAI-compatible endpoint
            │
            └── HYBRID ADAPTER
                - Use local for T0/T1/T2 generation
                - Use cloud for T4 frontier + complex synthesis
                - Automatic failover: cloud → local on network failure
```

### 11.2 Provider Selection Policy

```python
def select_provider(answer_plan: AnswerPlan, device_state: DeviceState) -> ProviderConfig:
    # Always use local on mobile offline
    if device_state.network == "offline":
        return LOCAL_PROVIDER
    
    # Always use local on Tier 4 (battery concern)
    if device_state.battery < 0.20:
        return LOCAL_PROVIDER
    
    # Use cloud for complex T4 queries if budget allows
    if answer_plan.retrieval_tier == "T4" and user_preferences.cloud_enabled:
        return CLOUD_PROVIDER_FAST  # Gemini Flash or DeepSeek API
    
    # Default: local first
    return LOCAL_PROVIDER
```

### 11.3 Message Translation

The provider adapter translates between internal AnswerPlan format and provider-specific message schemas, following the OpenClaude shim pattern:

```
Internal AnswerPlan evidence → provider messages
Provider streaming tokens → unified StreamEvent
Provider tool calls → internal tool execution
Provider finish signal → AnswerPlan completion
```

Zero provider-specific code exists above the adapter layer.

---

## 12. Storage Strategy — Local + Sync

### 12.1 Local-First Architecture

All data starts local. Sync to server is optional and non-blocking:

```
LOCAL STORAGE (always authoritative):
├── DuckDB file (events, claims, metadata) — ~5–50MB per 10K events
├── FAISS index files (.index, hot tier) — ~200MB for 500K vectors
├── Wiki markdown files (~1–5KB per page, ~50MB for 10K pages)
├── BM25 index (serialized) — ~100MB
├── Graph adjacency (DuckDB table) — ~20MB per 10K edges
└── SQLite WAL (ingestion queue) — ephemeral, < 5MB

SERVER SYNC (optional, async, delta-only):
├── Wiki pages: push delta patches (not full pages)
├── Event plane: push new events (append-only, no deletes from server)
├── Claim plane: bidirectional merge with server as authoritative
└── Graph: push new edges; pull updates from server graph
```

### 12.2 Storage Tiering

| Tier | Contents | Storage | Access Latency | Migration Trigger |
|------|----------|---------|----------------|------------------|
| HOT | Events < 30 days old | FAISS HNSW + DuckDB | < 50ms | Age > 30 days |
| WARM | Events 30–180 days | FAISS IVF-PQ + DuckDB | < 200ms | Age > 180 days |
| COLD | Events > 180 days | IVF-PQ compressed + DuckDB | < 500ms | User explicit or wiki coverage |
| ARCHIVED | Events > 1 year, covered by wiki | Compressed JSONL | Load-on-demand | Wiki macro compaction |

### 12.3 Deduplication

```
Content deduplication:
- Stage 1: SHA-256 hash of normalized content → exact duplicate check (< 1ms)
- Stage 2: For near-duplicates → MinHash LSH (< 10ms)
- Stage 3: For semantic duplicates → cosine similarity check (embedding, < 30ms, only for high-importance candidates)

Claim deduplication:
- Claim fingerprint = hash(subject_entity_id + claim_type + normalized_text)
- On collision: merge source_event_ids, update confirmation_count, update confidence
```

---

## 13. Observability and Self-Improvement Loop

### 13.1 Trace Architecture

Every query generates a full distributed trace:

```
trace_id (created by Orchestrator at query start)
  └── span: tier_routing (< 80ms)
  └── span: retrieval_tier_{T}
      ├── span: channel_dense
      ├── span: channel_sparse
      ├── span: channel_wiki
      └── span: channel_graph
  └── span: rrf_fusion
  └── span: reranking
  └── span: agent_dispatch (if T3/T4)
      ├── span: agent_timeline
      └── span: agent_causal
  └── span: quality_loops
      ├── span: crag
      ├── span: self_rag
      └── span: flare
  └── span: answer_synthesis
  └── span: generation (time-to-first-token logged)
```

### 13.2 Key Metrics Dashboard

**Tiering efficiency:**
- T0 hit rate (target: > 20% of all queries)
- T1 hit rate (target: > 30% of all queries)
- T2 rate (target: 35–40%)
- T3 rate (target: < 10%)
- T4 rate (target: < 2%)

**Quality metrics:**
- CRAG correction rate by tier (high rate → consider tier upgrade)
- Self-RAG critique pass rate by tier
- Retrieval precision@5, @10 by tier and intent type
- Wiki page confidence distribution
- Contradiction detection rate
- Provenance completeness ratio

**Latency metrics:**
- P50 / P95 / P99 per tier
- Time-to-first-token by tier
- Wiki Agent ingestion lag (time from event to wiki page update)
- Cache hit staleness distribution

**Memory quality metrics:**
- Wiki page growth rate (pages/week)
- Claim density per wiki page
- Stale claim ratio (claims not confirmed in > 90 days)
- Duplicate claim ratio (should be < 2%)

### 13.3 Continuous Self-Improvement

```
IMPROVEMENT LOOP (background, weekly):

1. RAGChecker diagnostic run:
   - Sample 100 queries from the past week
   - Compute: faithfulness, context precision, context recall per tier
   - Identify which intent types have lowest precision

2. Tier threshold calibration:
   - If T1 cache hit rate < 10%: wiki building is too slow → increase Wiki Agent priority
   - If T3 rate > 20%: complexity classifier is over-routing → retune thresholds
   - If T2 CRAG correction rate > 40%: retrieval quality is degraded → inspect index health

3. Retriever fine-tuning (if VRAM allows):
   - Collect hard negatives from low-CRAG-score queries
   - Run QLoRA fine-tuning on BGE-large for domain adaptation
   - Test against frozen eval set before deploying

4. Prompt card evolution:
   - Log agent output quality scores
   - Identify agents with lowest confidence / highest escalation rate
   - Update agent prompts based on failure pattern analysis
```

---

## 14. Production Gaps Analysis and Resolutions

### 14.1 Current System Gaps → This Architecture's Solutions

| Gap | Root Cause | Solution in This Architecture |
|-----|-----------|-------------------------------|
| Stream/non-stream inconsistency | Self-RAG/FLARE skipped in streaming | Shared AnswerPlan; pre-stream CRAG; async post-stream critique |
| Shallow chain traversal | Fixed top-k bounds, no deep graph traversal | T4 frontier expansion with budget; 6-hop limit; utility scoring |
| No canonical wiki layer | Event-oriented memory only | Wiki Plane (P3) + Wiki Agent as always-on background builder |
| Compression only at query time | No lifecycle-level compaction | Three-layer compaction: micro/meso/macro; provenance preserved always |
| All queries through heavy retrieval | No routing tier system | Five retrieval tiers with < 80ms routing decision |
| Local/cloud behavior divergence | Provider-specific code in pipeline | Provider adapter shim; single AnswerPlan consumed by all paths |
| RAPTOR not incremental | Static-corpus design | Deferred batch rebuild; hot+warm tiers searchable immediately |
| Missing knowledge hygiene | No lint + stale detection | Wiki lint operation + confidence decay + contradiction flagging |

### 14.2 Is the Current System Production-Grade?

**Honest assessment:**

The v3.0 RAG-Architecture is **research-grade approaching production-grade**. It is technically comprehensive and incorporates the right techniques. But it has three blockers for true production-grade status:

1. **No canonical wiki layer** → the system cannot compound knowledge efficiently over time. This is the biggest architectural gap.

2. **No tiered routing** → all queries pay the same cost, which is acceptable at small scale but fails under high query volume or on mobile.

3. **Stream/non-stream inconsistency** → this is a correctness issue, not a performance issue. A production system must guarantee identical answers for identical queries regardless of streaming mode.

**This architecture resolves all three.** With these additions, the system reaches true production-grade status.

---

## 15. Implementation Roadmap

### Phase 0 — Instrumentation (Week 1)
1. Add trace_id propagation across all pipeline stages
2. Add per-stage latency logging (query classification, retrieval, reranking, generation)
3. Add tier assignment logging
4. Build baseline eval set: 100 queries across all intent types
5. Freeze baseline metrics (CRAG score, precision@10, P95 latency by tier)

### Phase 1 — Wiki Core (Weeks 2–4)
1. Implement WikiStore module with page schema + frontmatter parser
2. Implement claim extractor (LLM-based, batched)
3. Implement claim deduplication + fingerprinting
4. Implement wiki page patch engine with provenance enforcement
5. Build entity registry and wiki section vector index
6. Implement audit log for all wiki mutations

### Phase 2 — Tiered Routing (Weeks 4–6)
1. Implement routing classifier (cache check + entity check + complexity score + intent)
2. Implement T0 (exact + semantic cache with wiki revision hash in key)
3. Implement T1 (wiki page direct + shallow dense)
4. Validate routing precision on eval set (target: < 5% wrong tier assignment)
5. Add tier metrics to dashboard

### Phase 3 — AnswerPlan Unification (Week 6–7)
1. Introduce AnswerPlan shared object schema
2. Refactor stream path to consume AnswerPlan
3. Refactor non-stream path to produce identical AnswerPlan
4. Align quality loops: pre-stream CRAG + async post-stream critique
5. Validate stream/non-stream parity on eval set (semantic similarity > 0.92)

### Phase 4 — T4 Frontier Expansion (Weeks 8–10)
1. Implement frontier expansion planner with utility scoring
2. Integrate wiki + claim + graph nodes into frontier
3. Implement intent-aware relation type priorities
4. Implement budget enforcement and marginal gain stopping
5. Add user notification for T4 queries (estimated time, streaming partial answer)

### Phase 5 — Wiki Agent Full Implementation (Weeks 10–14)
1. Implement Wiki Agent with full system prompt
2. Implement lint operation (contradiction detection, staleness, confidence decay)
3. Implement three-layer compaction (micro/meso/macro)
4. Build daily/weekly snapshot system
5. Integrate Wiki Agent scheduling with Master-Orchestrator (idle-time slots)
6. Validate wiki quality: confidence distribution, provenance completeness ratio

### Phase 6 — Mobile + Production Hardening (Weeks 14–18)
1. Implement mobile-specific tiered resource governor
2. Implement offline mode with local FAISS + local wiki
3. Implement delta sync between mobile + server
4. Implement provider adapter shim (local/cloud/hybrid)
5. Full load testing: 1000 concurrent sessions
6. Privacy audit: zero cross-tier leakage verification
7. Production readiness sign-off

---

## 16. Success Metrics

### 16.1 Retrieval Quality

| Metric | Current Target | New Target | Measurement |
|--------|---------------|-----------|-------------|
| Precision@10 | > 0.80 | > 0.85 | Multi-channel fusion |
| Answer Faithfulness | > 0.92 | > 0.94 | RAGAS + DPO alignment |
| Wiki page retrieval precision | N/A | > 0.90 | T1 query eval set |
| Multi-hop accuracy | > 0.75 | > 0.85 | T3/T4 synthetic chains |
| Contradiction resolution rate | N/A | > 0.85 | Arbitration eval set |

### 16.2 Latency

| Tier | P50 Target | P95 Target | P99 Target |
|------|-----------|-----------|-----------|
| T0 (cache) | < 20ms | < 50ms | < 100ms |
| T1 (wiki) | < 100ms | < 200ms | < 300ms |
| T2 (standard) | < 800ms | < 1.5s | < 2s |
| T3 (deep) | < 3s | < 6s | < 8s |
| T4 (frontier) | < 10s | < 20s | < 30s |
| First token (streaming) | < 400ms | < 800ms | < 1.2s |

### 16.3 Memory Quality

| Metric | Target |
|--------|--------|
| Wiki page confidence average | > 0.75 |
| Provenance completeness ratio | > 0.98 (every claim has source_event_ids) |
| Duplicate claim ratio | < 2% |
| Contradiction unresolved ratio | < 5% |
| Stale claim ratio | < 15% |
| Cache hit rate (T0+T1 combined) | > 50% of all queries |

### 16.4 Consistency

| Metric | Target |
|--------|--------|
| Stream vs non-stream parity | > 0.92 semantic similarity |
| Local vs cloud parity | > 0.88 semantic similarity |
| Tier regression rate | < 3% wrong tier assignment |
| Provider failover transparent | < 200ms additional latency |

---

## Appendix A — Full Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Primary LLM | DeepSeek-R1-Distill-Qwen-7B (7B local) | Generation + claim extraction |
| Embedding | BGE-large-en-v1.5 (1024d) | Dense retrieval + semantic similarity |
| Reranker | BGE-reranker-v2-m3 | Cross-encoder reranking (T2+) |
| Vector store (hot) | FAISS HNSW | Fast dense retrieval < 50ms |
| Vector store (warm/cold) | FAISS IVF-PQ | Compressed archival retrieval |
| Structured store | DuckDB | Events, claims, metadata, graph edges |
| Sparse retrieval | BM25 (Rank-BM25) | Keyword + exact match |
| Wiki store | Markdown files + FAISS section index | Canonical wiki pages |
| Graph store | NetworkX + DuckDB edges | Entity-relation graph traversal |
| Ingestion queue | asyncio.PriorityQueue + SQLite WAL | Crash-safe prioritized ingestion |
| STT (server) | Whisper (faster-whisper base) | Voice transcription |
| STT (mobile) | faster-whisper tiny (on-device) | Local transcription |
| OCR | EasyOCR | Text from images/documents |
| Vision captioning | BLIP-base | Image content description |
| Code parsing | tree-sitter (15+ languages) | AST-aware code chunking |
| Query classifier | SetFit (few-shot) | Fast intent classification |
| NER | DistilBERT | Entity extraction |
| Cache | Redis (server) / in-memory (mobile) | T0 response cache |
| Observability | OpenTelemetry + custom metrics | Trace + metrics pipeline |
| Eval | RAGChecker | Automated quality diagnostics |
| Provider shim | OpenAI-compatible adapter | Multi-provider abstraction |

---

## Appendix B — Query Type to Retrieval Tier Mapping (Reference)

| Query Type | Example | Typical Tier |
|-----------|---------|-------------|
| Repeated factual | "What's my main project?" (asked before) | T0 |
| Simple entity lookup | "What did I say about Rust last week?" | T1 |
| Recent events | "What happened in my last study session?" | T2 |
| Multi-entity | "How do my projects relate to my goals?" | T2 |
| Temporal analysis | "What was I working on in Q3 2025?" | T2 or T3 |
| Causal question | "Why did I quit that project?" | T3 |
| Belief evolution | "How has my opinion on X changed?" | T3 |
| Cross-domain synthesis | "How do my habits connect to my goals?" | T3 |
| Deep history trace | "Trace everything related to my career since 2023" | T4 |
| Long belief chain | "How did my views on AI safety evolve and why?" | T4 |
| Cross-entity deep | "How do the people I know influence my decisions?" | T4 |

---

*Agentic RAG + LLM Wiki Architecture v4.0*  
*Core insight: Route fast. Retrieve smart. Store once. Build Wikipedia forever.*  
*Simple things instant. Complex things thorough. Nothing wasted. Knowledge compounding.*
