# Agentic RAG + LLM Wiki Reimagining

## 0) Document Intent

This document defines a grounded, implementation-aware path to evolve Cortex Lab into a persistent, compounding intelligence system that combines:

1. Agentic RAG for runtime reasoning and adaptive retrieval.
2. A persistent LLM Wiki layer for durable, cumulative memory.
3. Budgeted but effectively unbounded relation-chain traversal.
4. Multi-level memory compression with provenance and consistency guarantees.

The goal is not just better answers per query. The goal is a system that becomes more coherent over time without memory pollution.

---

## 1) Current System: What Is Actually Implemented

## 1.1 Runtime Entry Paths

### Non-streaming path

Request path:

`/api/rag/chat` -> `rag_engine.rag_chat(...)` -> `orchestrator.process(...)`

High-level method chain:

1. `backend/server.py`:
   - `rag_chat(req)`
   - `_set_request_provider(req.llm_provider)`
2. `backend/src/engine.py`:
   - `CortexRAGEngine.rag_chat(...)`
   - cache lookup via `MultiLevelCache.get(...)`
   - `self.orchestrator.process(user_message, session_context)`
3. `backend/src/agents/orchestrator.py`:
   - `analyzer.analyze(...)`
   - optional `_llm_route_query(...)`
   - `transformer.transform(...)`
   - optional `_try_function_calling(...)`
   - `_handle_no_retrieval(...)` or `_handle_single_step(...)` or `_handle_multi_step(...)`
   - `compressor.compress_evidence(...)`
   - `_crag_evaluate(...)`
   - optional `_self_rag_critique(...)`
   - optional `_flare_active_retrieval(...)`

### Streaming path

Request path:

`/api/rag/chat` with `stream=true` -> `rag_engine.rag_retrieve(...)` -> server-side streaming generation

High-level method chain:

1. `backend/server.py`:
   - `_stream_rag_generate(...)`
2. `backend/src/engine.py`:
   - `CortexRAGEngine.rag_retrieve(...)`
   - `self.orchestrator.retrieve_only(...)`
3. `backend/src/agents/orchestrator.py`:
   - analyze + transform + retrieve
   - context compression
   - CRAG
   - Self-RAG and FLARE skipped in retrieve-only mode
4. `backend/server.py`:
   - `select_prompt_evidence(...)`
   - optional direct factual extraction `_try_extract_factual(...)`
   - local/Gemini stream generation

## 1.2 Retrieval Topology

Current retriever is a parallel multi-channel design:

- Dense vector retrieval (`_dense_retrieve`)
- Sparse BM25 (`_sparse_retrieve`)
- Graph traversal (`_graph_retrieve`)
- Temporal retrieval (`_temporal_retrieve`)
- Proposition retrieval (`_proposition_retrieve`, conditional)
- Optional PageIndex channel (`_pageindex_retrieve`)

Fusion and ranking:

1. RRF fusion via `_rrf_fusion(...)`.
2. Cross-encoder reranking via `_cross_encoder_rerank(...)`.

## 1.3 Memory Ingestion Reality

Important behavior already present:

1. Chat queries are intentionally not auto-ingested in chat path (to avoid question pollution).
2. Durable ingestion comes from:
   - Manual/API memory ingestion.
   - Ambient voice ingestion pipeline.
3. Ingestion enriches each memory with:
   - type, emotion, entities, topics, importance
   - propositions
   - optional context prefix

## 1.4 Existing Quality and Safety Loops

Quality:

1. CRAG quality scoring and supplementary retrieval.
2. Self-RAG critique and optional revision.
3. FLARE low-confidence retrieval augmentation.

Safety/runtime controls:

1. Function calling with policy checks.
2. Approval-gated risky operations.
3. Runtime task manager and cancellation support.

---

## 2) Core Limitations and Root Causes

## 2.1 Consistency Split Between Stream and Non-stream

Root cause:

- Non-streaming path gets the full orchestration quality loop.
- Streaming path uses retrieve-only + server prompt generation and bypasses Self-RAG/FLARE.

Consequence:

- Different answer quality and behavior for the same query depending on stream mode.

## 2.2 Retrieval Horizon Is Bounded and Shallow for Deep Chains

Root cause:

1. Top-k bounded retrieval in each channel.
2. Graph traversal has fixed local constraints (for example, limited hop patterning and score heuristics).
3. FLARE retrieves only a small number of additional segments.

Consequence:

- Long relation chains are approximated, not deeply traversed.
- Cross-domain evidence graphs are under-explored for complex reflective queries.

## 2.3 Local-vs-cloud Query Transformation Drift

Root cause:

- Local path uses fast heuristic transformations.
- Heavier transform steps (HyDE, step-back, decomposition) are reduced or skipped for latency.

Consequence:

- Behavioral differences by provider/runtime mode.
- Lower recall on complex local-mode queries.

## 2.4 Missing Durable Canonical Memory Layer

Root cause:

- Current memory objects are rich but remain event-oriented.
- There is no first-class canonical markdown knowledge layer that compounds over time.

Consequence:

- The system can retrieve evidence, but cannot maintain coherent long-horizon narrative pages without repeated recomputation.

## 2.5 Compression Exists Mostly at Query Time, Not at Knowledge Lifecycle Level

Root cause:

- Context compression is applied during response generation.
- Consolidation exists but is not yet the central memory lifecycle backbone.

Consequence:

- Token savings happen per request, but long-term memory growth is not fully governed by canonical compaction rules.

## 2.6 Latency Hotspots

Root cause contributors:

1. Sequential steps after retrieval are still significant.
2. Reranking and synthesis are cost-heavy under large evidence sets.
3. Optional cloud retrieval/document channels can add tail latency.

Consequence:

- Tail latencies remain high for complex queries.
- Consistent sub-2s complex reasoning is not guaranteed.

---

## 3) Reimagined Target: Agentic RAG + LLM Wiki

## 3.1 Strategic Principle

Unbounded memory should be logical, not brute force.

Meaning:

1. The knowledge graph/wiki can grow indefinitely.
2. Each query traversal is budgeted and adaptive.
3. We maximize marginal utility per retrieved token.

## 3.2 New Memory Planes

Introduce five explicit planes:

1. Event Plane (raw memory events)
2. Claim Plane (atomic normalized claims)
3. Wiki Plane (canonical markdown pages)
4. Graph Plane (entities, claim relations, provenance edges)
5. Working Plane (ephemeral short-lived context for active sessions)

## 3.3 LLM Wiki as a First-class Store

The wiki is the durable semantic center, not just a byproduct.

Each wiki page should be markdown with stable sections and metadata frontmatter.

Example structure:

```md
---
page_id: profile/suraj
page_type: profile
confidence: 0.86
last_updated: 2026-04-06T10:12:00Z
source_claim_count: 48
source_memory_ids:
  - mem_abc
  - mem_xyz
privacy_tier: private
---

# Profile

## Stable Facts
...

## Evolving Beliefs
...

## Active Projects
...

## Open Questions
...

## Provenance
...
```

## 3.4 Selective Memory Tagging Model

Every claim/memory gets a multi-axis tag set:

1. `durability`: ephemeral | working | durable | canonical
2. `trust_level`: unverified | single_source | multi_source | verified
3. `sensitivity`: public | private | restricted
4. `domain`: profile | project | relationship | health | finance | meta
5. `update_policy`: append_only | patchable | replace_on_conflict

This enables selective retention and safer autonomous updates.

---

## 4) Operations Model (Ingest, Query, Lint, Compress)

## 4.1 Ingest Operation

Pipeline for any new input (chat/manual/ambient/doc):

1. Normalize and sanitize payload.
2. Extract claims and entities.
3. Deduplicate against claim fingerprints.
4. Score relevance and durability.
5. Write event record.
6. Upsert claim records.
7. Patch wiki pages.
8. Update retrieval indexes.
9. Emit ingest log event.

## 4.2 Query Operation

Two-stage retrieval and reasoning:

1. Retrieve wiki summaries first for fast global context.
2. Drill down into supporting claims/events for verification.

Then run budgeted chain expansion until stop criteria are met.

## 4.3 Lint Operation (Knowledge Hygiene)

A periodic wiki linter should detect:

1. Contradictory claims without arbitration notes.
2. Stale statements lacking recent support.
3. Pages with confidence decay.
4. Missing provenance links.
5. Sensitive content in wrong scope.

Lint output feeds arbitration/planning agents for corrective action.

## 4.4 Compression Operation

Three-layer compaction strategy:

1. Micro compression: claim extraction per event.
2. Meso compression: daily/weekly summaries by topic.
3. Macro compression: canonical wiki page synthesis and archive rollups.

Hard requirement: never compress away provenance.

---

## 5) Effectively Unbounded Relation-chain Retrieval

## 5.1 Current Gap

Current retrieval primarily returns high-scoring local neighborhoods. It is powerful, but not true chain-exploration across a large evolving graph.

## 5.2 Proposed Retrieval Planner

Use a frontier-based expansion planner over wiki + claim + graph nodes.

Seed sources:

1. Query entities and topics.
2. Matched wiki pages.
3. High-confidence claims.

Expansion policy:

1. Score each frontier node by relevance, novelty, trust, and recency.
2. Expand highest utility nodes first.
3. Stop when budget exhausted or marginal gain drops below threshold.

Pseudo-flow:

```text
frontier = init_seeds(query)
working_set = []
budget = token_budget

while frontier not empty and budget > 0:
    node = argmax(frontier.utility)
    if node.marginal_gain < epsilon:
        break

    working_set.add(node)
    budget -= node.estimated_tokens

    neighbors = expand(node, relation_types=query_intent_relations)
    frontier.push(neighbors with updated utility)

return compress(working_set)
```

This yields logical unboundedness with practical compute bounds.

## 5.3 Intent-aware Relation Expansion

Relation priorities by intent:

1. Causal: cause_of, leads_to, influenced_by
2. Reflective: belief_shift, preference_change, turning_point
3. Temporal: before, after, co_occurs
4. Procedural: prerequisite, step_of, dependency

---

## 6) Latency Re-architecture

## 6.1 Budgeted Latency Envelopes

Define explicit SLO budgets per stage:

1. Analysis + routing: <= 120ms
2. First-pass retrieval (wiki+claims): <= 350ms
3. Expansion planner: <= 350ms (adaptive)
4. Rerank + compression: <= 250ms
5. Generation first token: <= 600ms local target

## 6.2 Early-answer Strategy

For stream mode:

1. Start streaming with high-confidence wiki summary context.
2. Continue background expansion and attach late evidence if needed.
3. Preserve one answer contract across stream/non-stream modes.

## 6.3 Cache Architecture Evolution

Extend cache keys to include:

1. provider
2. query fingerprint
3. wiki revision hash
4. retrieval policy version

This avoids stale results when canonical knowledge changes.

---

## 7) Consistency Re-architecture

## 7.1 Single Answer Contract

Both stream and non-stream should consume one shared `AnswerPlan` output from orchestrator.

`AnswerPlan` should include:

1. selected evidence set
2. confidence and arbitration notes
3. generation policy
4. citation/provenance requirements

## 7.2 Unified Quality Loops

Self-RAG/FLARE should not be mode-dependent.

For streaming, use lightweight staged quality:

1. Pre-stream fast critique on retrieval quality.
2. Optional post-stream revision event if confidence drops below threshold.

## 7.3 Deterministic Provenance Envelope

Every final answer should carry internal provenance fields:

1. source wiki pages
2. source claim IDs
3. source event IDs
4. confidence composition

---

## 8) LLM Wiki Storage and Index Strategy

## 8.1 Storage Layout Proposal

```text
data/wiki/
  pages/
    profile/
    projects/
    relationships/
    themes/
  claims/
    YYYY/MM/*.jsonl
  logs/
    ingest/
    lint/
    compaction/
  snapshots/
    daily/
    weekly/
```

## 8.2 Indexes

Maintain parallel indexes:

1. Wiki-section vector index (semantic recall)
2. BM25/keyword index over markdown and claims
3. Graph index over entity-claim-page links
4. Temporal index over events and revisions

## 8.3 Log and Replay

Every mutation should emit an append-only log entry with:

1. operation_id
2. operation_type
3. target page/claim
4. before hash
5. after hash
6. actor agent
7. confidence
8. approval/policy status

This enables full replay and audit.

---

## 9) Migration Plan (Implementation-oriented)

## Phase 0: Instrumentation and Baseline

1. Add stage-level latency tracing for stream and non-stream parity.
2. Add retrieval depth metrics and evidence drift metrics.
3. Freeze baseline eval set for regression.

## Phase 1: Wiki Core

1. Add wiki store module and page schema.
2. Build claim extractor/upserter.
3. Add page patch engine with provenance enforcement.

Likely touchpoints:

- `backend/src/ingestion/__init__.py`
- `backend/src/engine.py`
- new `backend/src/wiki/*`

## Phase 2: Planner-based Retrieval

1. Add retrieval planner for frontier expansion.
2. Integrate wiki and claim indexes into retriever.
3. Add intent-aware relation expansion.

Likely touchpoints:

- `backend/src/retrieval/hybrid_retriever.py`
- `backend/src/retrieval/query_engine.py`
- new `backend/src/retrieval/wiki_planner.py`

## Phase 3: Answer Contract Unification

1. Introduce `AnswerPlan` shared object.
2. Refactor stream and non-stream to use same orchestrator plan output.
3. Align quality loops across modes.

Likely touchpoints:

- `backend/src/agents/orchestrator.py`
- `backend/server.py`
- `backend/src/engine.py`

## Phase 4: Lint + Compaction

1. Add scheduled lint jobs.
2. Add confidence decay and stale-page detectors.
3. Add macro compaction and snapshotting.

Likely touchpoints:

- `backend/src/runtime/*`
- `backend/src/wiki/lint.py`
- `backend/src/wiki/compactor.py`

## Phase 5: Agentic Governance

1. Add explicit wiki patch approval policy for restricted domains.
2. Add automated conflict arbitration workflow.
3. Add repair loops for low-confidence pages.

---

## 10) Success Metrics

## 10.1 Retrieval and Reasoning

1. Evidence relevance at top-10 and top-20
2. Multi-hop answer accuracy on synthetic chain benchmarks
3. Contradiction resolution accuracy

## 10.2 Latency

1. P50 and P95 first-token latency
2. P95 full-answer latency for complex queries
3. Planner expansion cost per query

## 10.3 Memory Quality

1. Wiki page confidence drift over time
2. Provenance completeness ratio
3. Duplicate claim ratio
4. Contradiction unresolved ratio

## 10.4 Consistency

1. Stream vs non-stream semantic parity score
2. Provider parity score (local vs Gemini)
3. Regression rate on personal fact queries

---

## 11) Risks and Mitigations

## Risk 1: Wiki bloat and over-fragmentation

Mitigation:

1. Enforce page schema and max section budgets.
2. Run weekly compaction and page merge heuristics.

## Risk 2: Incorrect autonomous page edits

Mitigation:

1. Enforce trust/sensitivity gates.
2. Route high-impact edits through arbitration + approval.

## Risk 3: Latency regression from deeper traversal

Mitigation:

1. Strict planner budgets and marginal gain stopping.
2. Early-answer streaming with progressive enrichment.

## Risk 4: Provider behavior divergence

Mitigation:

1. Unified answer contract and retrieval envelope.
2. Provider-specific adapters only at generation edge.

---

## 12) Immediate Next Steps (Practical)

1. Implement a minimal WikiStore and page schema with provenance fields.
2. Add claim extraction + patching into current ingestion path.
3. Build retrieval planner prototype over existing knowledge graph.
4. Introduce one shared `AnswerPlan` for stream/non-stream parity.
5. Add lint job for contradiction and stale confidence detection.

---

## 13) Final Design Position

Cortex Lab should evolve from memory-enhanced chat to a true persistent intelligence runtime.

The synthesis is:

1. Agentic RAG for adaptive, query-time reasoning.
2. LLM Wiki for durable, compounding, canonical knowledge.
3. Budgeted relation-chain traversal for deep relevance.
4. Compression with provenance to scale memory safely.

This design preserves privacy-first local intelligence while moving toward lifelong coherence and higher-trust answers.
