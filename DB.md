# Cortex Lab Data Management Blueprint (DB.md)

## 1) Purpose

This document is the canonical data-management plan for Cortex Lab.

It does four things:

1. Maps all currently implemented data stores, caches, and persistence flows.
2. Maps implemented behavior against Orchestrator and Agentic-RAG architecture targets.
3. Defines the target local-first, durable, and sync-ready data architecture.
4. Defines a phased execution plan with P0/P1/P2/P3 priorities.

Scope includes:

- Memory ingestion and retrieval storage.
- Wiki and claim layers.
- Ambient voice data and conversation persistence.
- Session/deep-app persistence (Session Forge, Life Chronicle, session tree JSONL).
- Cache layers and invalidation.
- Update/delete lifecycle, retention, archival, and sync.


## 2) Grounding Sources

This plan is grounded in:

- Orchestrator production spec (runtime governance, privacy tiering, audit expectations).
- Agentic-RAG architecture spec (5 memory planes, tiered retrieval, local+sync strategy).
- Current backend implementation in `backend/src` and `backend/server.py`.

Key architecture expectations (from docs):

- Local-first authoritative storage with optional async sync.
- 5 memory planes: working, event, claim, wiki, graph.
- Tiered retrieval with cache-first fast path.
- Auditability for write/update/delete actions.
- Retention and privacy-tier controls.


## 3) Current Implemented Data Inventory

## 3.1 Core Memory and Retrieval Stores

| Surface | Path | Format | Primary Owner | Notes |
|---|---|---|---|---|
| Metadata DB | `backend/data/cortex.duckdb` | DuckDB | `MetadataStore` | Core tables for memories, entities, edges, belief deltas, conversations, feedback |
| Vector state | `backend/data/vectors/` | FAISS + JSON + NPY | `VectorStore` | `hot.index`, `warm.index`, `cold.index`, `vectors.npy`, `vector_ids.json`, `vector_state.json` |
| Knowledge graph | `backend/data/graph/knowledge_graph.json` | JSON | `KnowledgeGraph` | NetworkX node-link export |
| PageIndex mapping | `backend/data/pageindex/doc_mapping.json`, `usage.json` | JSON | `PageIndexStore` | Local cloud-doc mapping and usage counters |

DuckDB tables currently created by `MetadataStore`:

- `memories`
- `belief_deltas`
- `entities`
- `graph_edges`
- `conversations`
- `memory_topics`
- `memory_entities`
- `feedback`


## 3.2 Wiki and Claim Plane

| Surface | Path | Format | Owner | Notes |
|---|---|---|---|---|
| Wiki pages | `backend/data/wiki/pages/*.md` | Markdown + frontmatter | `WikiStore` | Human-readable canonical pages |
| Wiki page metadata | `backend/data/wiki/pages/*.meta.json` | JSON | `WikiStore` | Version, topics, claim links |
| Wiki topic index | `backend/data/wiki/index.json` | JSON | `WikiStore` | Topic to page ID map |
| Claims | `backend/data/wiki/claims/*.json` | JSON per claim | `ClaimStore` | Atomic claims, confidence, contradictions |
| Lint reports | `backend/data/wiki/lint/*.json*` | JSON + JSONL | `WikiLinter` | Hygiene reports and summary history |
| Compaction reports | `backend/data/wiki/compaction/*.json*` | JSON + JSONL | `WikiCompactor` | Compaction run history and summaries |


## 3.3 Session and Deep-App Persistence

| Surface | Path | Format | Owner | Notes |
|---|---|---|---|---|
| Session tree | `backend/data/sessions/<agent_id>/<session_id>.jsonl` | JSONL | `SessionPersistence` | Append-only tree with compaction entries |
| Session Forge artifacts | `backend/data/deep_apps/session_forge/*.jsonl` | JSONL | `SessionMemoryForgeService` | Thought objects, decision records, open loops, gap signals, belief evolution, summaries, runs |
| Chronicle moments | `backend/data/chronicle/moments/YYYY/MM/DD/*.json` | JSON | `LifeChronicleService` | Saved passive moments |
| Chronicle timeline | `backend/data/chronicle/timeline/timeline.jsonl` | JSONL | `LifeChronicleService` | Append-only timeline |
| Chronicle people index | `backend/data/chronicle/people_appearances/*.json` | JSON | `LifeChronicleService` | Person to moments lookup |


## 3.4 Ambient and Voice Persistence

| Surface | Path | Format | Owner | Notes |
|---|---|---|---|---|
| Ambient config | `backend/data/ambient_config.json` | JSON | `AmbientService` | Provider and capture settings |
| Voiceprints | `backend/data/voiceprints/user.npy` | NPY | `SpeakerIdentifier` | User enrollment embedding |
| Speaker aliases | `backend/data/voiceprints/speaker_aliases.json` | JSON | `SpeakerIdentifier` | Label to name mapping |
| Local conversation files | `backend/data/conversations/conv_*.json` | JSON | `ConversationSegmenter` | Finalized ambient conversations |
| Ambient conversation tables | `backend/data/cortex.duckdb` | DuckDB | `ConversationSegmenter` | Raw transcript + turns audit tables |
| TTS models | `backend/data/tts_voices/*` | ONNX + JSON | `TextToSpeech` | Local voice model files |


## 3.5 Runtime and Cache State (Ephemeral)

| Surface | Storage | Owner | Persistence |
|---|---|---|---|
| Query response cache | In-memory dict/list | `MultiLevelCache` | No disk persistence |
| BM25 index | In-memory inverted index | `HybridRetriever` | Rebuilt on demand |
| Proposition embedding index | In-memory | `HybridRetriever` | Rebuilt on demand |
| Runtime task registry | In-memory | `RuntimeTaskManager` | No disk persistence |
| Permission queue | In-memory | `SafeToolRuntime` | No disk persistence |
| Runtime event bus | In-memory queues | `RuntimeEventBus` | No disk persistence |
| Scheduler run state | In-memory | `BackgroundScheduler` | No disk persistence |


## 4) Implemented Lifecycle: Create/Update/Delete/Archive

## 4.1 Create/Write Flows

Memory ingestion (`MemoryIngestionPipeline.ingest`):

1. Classify and enrich content.
2. Embed content.
3. Deduplicate check.
4. Write vector (`VectorStore.add`).
5. Write metadata (`MetadataStore.store_memory`).
6. Update graph and entity/edge sync.
7. Belief-delta detection and write.

Conversation/chat turns:

- Stored in DuckDB `conversations` via `MetadataStore.store_conversation_turn`.

Ambient conversations:

- Segment-level summary/structured memories may be ingested.
- Raw transcript and turns stored to DuckDB by `ConversationSegmenter`.
- Conversation record saved to `backend/data/conversations/*.json`.

Wiki/claim:

- Page create/patch in markdown + meta JSON.
- Claim upsert in per-claim JSON files.
- Linking between wiki pages and claim IDs.

Deep apps:

- Session Forge writes append-only JSONL artifacts.
- Life Chronicle writes moment JSON and timeline JSONL.


## 4.2 Update/Upsert Flows

- `MetadataStore.store_memory` is insert-or-replace.
- Claim upsert reinforces confidence and source IDs.
- Wiki patch increments page version and updated timestamp.
- Graph merge updates edge weights and memory IDs.
- Background jobs run wiki lint/compaction and Session Forge synthesis.


## 4.3 Delete Flows (Current)

- `delete_memory` removes from metadata `memories` and marks vector deleted.
- No guaranteed cascade to:
  - `memory_topics` / `memory_entities`
  - `entities`/`graph_edges` references
  - wiki claim links, claim sources, chronicle/deep-app references

Other surfaces:

- Wiki pages: no explicit delete API.
- Claims: soft deactivation exists; no full lifecycle delete pipeline.
- Chronicle and Session Forge: append-only; no purge/retention compaction policy.


## 4.4 Archive/Retention (Current)

- Vector has hot/warm/cold tiers and migration function.
- No implemented archived tier with deterministic lifecycle.
- No global retention executor by privacy tier/domain.
- No central tombstone registry or legal-hold mechanism.


## 5) Current Cache Model

Implemented:

- Query result cache:
  - Exact cache (provider-aware hash key).
  - Semantic cache (embedding similarity threshold 0.92).
  - Bounded sizes (`_max_exact=200`, `_max_semantic=50`).
- Retriever internal caches:
  - BM25 inverted index rebuilt when memory count changes.
  - Proposition embedding index rebuilt when memory count changes.

Gaps:

- No persistent cache snapshots.
- No TTL policy by data class.
- Invalidation mostly coarse and source-driven.
- No wiki revision/hash-aware cache keying.


## 6) Implemented vs Planned Architecture Mapping

## 6.1 Five Memory Planes Mapping

| Planned Plane | Planned Intent | Implemented Status |
|---|---|---|
| P0 Working Memory | Session-only ephemeral context | Partially implemented (in-memory history/context and runtime state) |
| P1 Event Plane | Durable raw events with metadata | Implemented via DuckDB `memories` + vector store |
| P2 Claim Plane | Atomic deduplicated claims | Implemented as JSON claim store (not yet relational/indexed as planned) |
| P3 Wiki Plane | Canonical markdown pages + indexes | Implemented core wiki store + lint + compaction |
| P4 Graph Plane | Entity/relation graph for traversal | Implemented via NetworkX + DuckDB edge sync |


## 6.2 Storage Strategy Mapping (Doc -> Code)

| Planned (Architecture) | Current |
|---|---|
| Local authoritative + async sync | Local-first yes; robust outbox sync not implemented |
| Tiered storage incl archive | Hot/warm/cold vector tiers partly implemented; archive policy missing |
| Dedup with hash + near-dup + semantic | Semantic dedup implemented; exact hash/LSH stages missing |
| Durable audit for writes/deletes/updates | Runtime audit mostly in-memory; durable audit log store missing |
| Privacy-tier retention rules | Policy concepts present; persistence-level enforcement incomplete |


## 7) Confirmed Risks and Defects (Code-Grounded)

1. Deduplication bug: ingestion calls `metadata.update_memory_timestamp(...)`, but method is not implemented in `MetadataStore`.
2. Multi-store write atomicity gap: vector, metadata, and graph writes are not transactional as a unit.
3. Delete lifecycle incompleteness: memory deletion does not cascade through graph/wiki/claim/index references.
4. Retrieval recall caps: BM25/proposition index rebuild reads capped subsets (`get_memory_texts(limit=5000)`, propositions `limit=2000`).
5. Potential table-name collision: both metadata and ambient layers use a `conversations` table name in the same DuckDB file with incompatible intended schemas.
6. Hardcoded PageIndex API key fallback in config file creates security and operational risk.
7. Metadata fallback risk: if DuckDB unavailable, `_fallback` in-memory map has no persistence.
8. Vector deletions are tombstone-based; no scheduled index compaction/rebuild pipeline.
9. Runtime audit, permission queue, event bus, and scheduler state are in-memory only (non-durable).
10. Relative-path drift risk: some services default to `data/...` relative paths, others use engine absolute data dir.


## 8) Target Data Architecture (Planned DB Strategy)

## 8.1 Core Principles

1. Local authoritative state.
2. Event-sourced durability for all mutating operations.
3. Async projectors for secondary indexes (vector, graph, wiki derivatives).
4. Explicit tombstones and deletion propagation.
5. Outbox-based cloud sync (idempotent, retry-safe).
6. Privacy-tier and retention as first-class DB policy.


## 8.2 Canonical Storage Roles

- DuckDB remains authoritative relational/event store.
- FAISS remains vector serving/index layer.
- Markdown wiki remains human-readable canonical narrative layer.
- JSONL artifacts remain append-only deep-app logs (with retention and compaction policy).


## 8.3 Proposed Durable Tables (Additive)

Add to DuckDB:

- `memory_events` (append-only event log)
  - `event_id`, `memory_id`, `event_type`, `payload_json`, `source`, `session_id`, `created_at`
- `memory_tombstones`
  - `memory_id`, `deleted_at`, `deleted_by`, `reason`, `cascade_status`
- `audit_log`
  - matches Orchestrator audit schema: action, agent, decision, permission mode, trace, session
- `sync_outbox`
  - `outbox_id`, `entity_type`, `entity_id`, `op`, `payload_json`, `status`, `retry_count`, `next_retry_at`
- `index_watermarks`
  - per-projector positions for replay and recovery
- `wiki_page_registry` (metadata mirror for markdown pages)
  - `page_id`, `path`, `version`, `updated_at`, `hash`


## 8.4 Write Path Contract (Unit of Work)

For every ingest/update/delete mutation:

1. Start transaction in DuckDB.
2. Write primary row changes.
3. Append `memory_events` entry.
4. Append `audit_log` entry.
5. Append `sync_outbox` entry when cloud sync eligible.
6. Commit.
7. Async projectors consume events and update FAISS/graph/wiki caches.

Effect: DB commit is source of truth; projectors become replayable.


## 8.5 Deletion Contract

Replace direct hard delete with two-stage deletion:

1. Soft delete (tombstone + `is_deleted=true` in primary table).
2. Async cascade workers:
   - vector removal/rebuild queue
   - graph edge and node relation cleanup
   - claim source-link cleanup
   - wiki link repair and lint trigger
   - cache invalidation by memory/topic/entity

Hard purge is optional maintenance operation after retention window.


## 8.6 Cache Contract

Define three cache classes:

- Response cache (T0): provider-aware + wiki revision hash + TTL.
- Retrieval index caches: persisted BM25/proposition snapshots with incremental updates.
- Runtime ephemeral caches: permission queue/event streams remain in-memory but backed by durable audit records.

Invalidation triggers:

- Memory ingest/update/delete event.
- Wiki page patch/create/lint conflict resolution.
- Claim upsert/deactivate/contradiction update.


## 8.7 Retention and Archival Contract

Retention policies by plane:

- P0 working: session-scoped, non-persistent except optional session tree.
- P1 event: retained by policy tier; archived when wiki/claim coverage threshold reached.
- P2 claim: confidence decay and supersession policy.
- P3 wiki: revisions retained; compaction retains provenance hashes.
- P4 graph: decay and pruning for weak stale edges.

Archive target:

- Compressed JSONL snapshots with manifest hashes.
- Recoverability tested by replay into clean environment.


## 8.8 Sync Strategy (Local + Cloud)

Use `sync_outbox` worker:

- Local commit never waits on cloud.
- Outbox records include deterministic idempotency key.
- Retry with exponential backoff and dead-letter state.
- Pull-side merge for claims/wiki uses conflict policy:
  - append-only events
  - claim-level arbitration metadata
  - page-level version and hash checks


## 9) Phased Execution Plan

## P0 (Immediate Stabilization, 1-2 weeks)

1. Implement `MetadataStore.update_memory_timestamp` or remove call from dedup path.
2. Resolve DuckDB `conversations` table naming collision between chat and ambient pipelines.
3. Remove hardcoded PageIndex API key fallback; require env-secret injection.
4. Expand delete path to clean junction tables and invalidate retriever caches consistently.
5. Add periodic autosave/rebuild scheduling for vector tombstone compaction.
6. Add startup path-normalization guard and explicit data root logs.

## P1 (Durability and Consistency, 2-4 weeks)

1. Add `memory_events`, `audit_log`, `memory_tombstones`, `sync_outbox`, `index_watermarks` tables.
2. Move writes to transactional unit-of-work pattern.
3. Introduce projector workers for vector/graph/wiki/index updates.
4. Persist BM25/proposition index snapshots and incremental update path.
5. Add dashboard checks for store divergence (metadata vs vector vs graph).

## P2 (Retention, Privacy, and Governance, 4-8 weeks)

1. Implement plane-aware retention policy executor.
2. Add privacy-tier enforcement hooks at persistence layer.
3. Implement archive snapshots + restore validation jobs.
4. Add full delete-cascade with status tracking and retry.
5. Add durable scheduler run ledger and audit replay tooling.

## P3 (Cloud Sync and Frontier Scale, 8+ weeks)

1. Deploy outbox-driven sync worker with conflict-resolution policies.
2. Add claim/wiki bidirectional merge metadata.
3. Add archive tier load-on-demand and cost-aware retrieval routing.
4. Integrate frontier retrieval needs with durable graph/wiki revision indexing.


## 10) Verification Checklist

After each phase, verify:

1. No write acknowledged without durable transaction commit.
2. Delete operation reaches terminal cascade status across all planes.
3. Replay from `memory_events` can fully rebuild secondary indexes.
4. Cache invalidation fires for all mutating operations.
5. Retention policy produces deterministic, auditable outputs.
6. Cloud sync retries are idempotent and non-destructive.
7. Audit log coverage for write/update/delete is 100%.


## 11) Immediate Action Summary

Highest-priority implementation fixes:

1. Dedup timestamp update bug.
2. Conversation table namespace collision.
3. Hardcoded API key removal.
4. Delete cascade correctness.
5. Index cap and persistence strategy for BM25/proposition.

These are prerequisites before deeper cloud-sync and archival rollout.
