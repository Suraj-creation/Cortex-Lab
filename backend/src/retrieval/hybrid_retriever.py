"""
Multi-Channel Hybrid Retrieval Engine for Cortex Lab
6 parallel retrieval channels with RRF fusion and real cross-encoder reranking.

Channels:
1. Dense (BGE-large-en-v1.5 + FAISS) — w: 0.30
2. Sparse (BM25 keyword) — w: 0.20
3. Graph (Knowledge Graph traversal) — w: 0.15
4. Temporal (SQL time filter) — w: 0.10
5. Proposition (Atomic fact matching) — w: 0.05
6. PageIndex (Cloud document reasoning) — w: 0.20

Reranking: BGE-reranker-v2-m3 cross-encoder (or embedding fallback)
"""

import asyncio
import re
import time
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import (
    CausalMemoryObject, MemoryQuery, QueryIntent, RetrievalResult,
    RetrievalChannelTrace
)
from src.models.embeddings import EmbeddingModel, CrossEncoderReranker
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph


class HybridRetriever:
    """
    Multi-channel hybrid retrieval with async parallel execution
    and real cross-encoder reranking.
    
    Channels execute simultaneously via asyncio:
    Sequential (naive): 100 + 50 + 80 + 30 + 90 + 200 = 550ms
    Parallel (async): max(100, 50, 80, 30, 90, 200) = ~200ms
    Latency savings: 64%
    """

    # Channel weights — includes backward-compatible legacy channels.
    # PageIndex bypasses RRF and is injected post-fusion.
    # Legacy channels (`raptor`, `community`) are compatibility placeholders
    # unless enabled by a downstream runtime extension.
    WEIGHTS = {
        "dense": 0.35,
        "sparse": 0.25,
        "graph": 0.20,
        "temporal": 0.10,
        "proposition": 0.05,
        "raptor": 0.03,
        "community": 0.02,
    }

    # Fallback weights when PageIndex is disabled / has no documents.
    # Keep backward-compatible channels represented to preserve old contracts.
    WEIGHTS_LOCAL_ONLY = {
        "dense": 0.40,
        "sparse": 0.25,
        "graph": 0.20,
        "temporal": 0.10,
        "proposition": 0.03,
        "raptor": 0.01,
        "community": 0.01,
    }

    # RRF constant
    RRF_K = 60

    def __init__(self, embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 metadata_store: MetadataStore,
                 knowledge_graph: KnowledgeGraph,
                 reranker: Optional[CrossEncoderReranker] = None,
                 pageindex_store=None):
        self.embeddings = embedding_model
        self.vectors = vector_store
        self.metadata = metadata_store
        self.graph = knowledge_graph
        self.reranker = reranker  # Real cross-encoder reranker
        self.pageindex_store = pageindex_store  # PageIndex cloud retrieval (optional)

        # BM25 inverted index (rebuilt on demand, invalidated on new ingestion)
        self._bm25_inverted: Dict[str, Dict[str, int]] = {}  # token -> {mid: tf}
        self._bm25_doc_lengths: Dict[str, int] = {}  # mid -> doc_length
        self._bm25_idf: Dict[str, float] = {}
        self._bm25_avg_dl: float = 0.0
        self._bm25_doc_count: int = 0
        self._bm25_last_count: int = 0  # Track memory count for invalidation

        # Proposition embedding index (pre-computed, updated on ingestion)
        self._prop_index: Dict[str, List[Tuple[str, np.ndarray]]] = {}  # memory_id -> [(prop_text, embedding)]
        self._prop_last_count: int = 0

    async def retrieve(self, query: MemoryQuery, top_k: int = 20) -> List[RetrievalResult]:
        """
        Execute all channels in parallel, fuse results, and optionally rerank.
        Dynamically includes PageIndex channel if documents are available.
        """
        t0 = time.time()

        # Determine if PageIndex should participate
        use_pageindex = (
            self.pageindex_store is not None
            and self.pageindex_store.is_connected
            and self.pageindex_store.has_documents
        )

        # Timed wrapper for per-channel observability
        async def _timed(name: str, coro):
            t = time.time()
            result = await coro
            duration = (time.time() - t) * 1000
            return name, result, duration

        # Build channel tasks — always run local channels
        # Proposition channel is disabled when using API embeddings (too slow).
        # It uses local embedding model detection to auto-enable when available.
        tasks = [
            _timed("dense", self._dense_retrieve(query, top_k * 2)),
            _timed("sparse", self._sparse_retrieve(query, top_k * 2)),
            _timed("graph", self._graph_retrieve(query, top_k * 2)),
            _timed("temporal", self._temporal_retrieve(query, top_k * 2)),
            _timed("raptor", self._raptor_retrieve(query, top_k * 2)),
            _timed("community", self._community_retrieve(query, top_k * 2)),
        ]
        # Only add proposition channel if local (not API) embeddings available
        if getattr(self.embeddings, '_backend', 'stub') == 'local':
            tasks.append(_timed("proposition", self._proposition_retrieve(query, top_k * 2)))

        # Add PageIndex channel if available
        if use_pageindex:
            tasks.append(
                _timed("pageindex", self._pageindex_retrieve(query, top_k * 2))
            )

        # Run all channels concurrently
        channel_tasks = await asyncio.gather(*tasks)

        # Unpack timed results into a dict
        channel_timings = {}
        all_channels = {}
        for name, results, duration in channel_tasks:
            channel_timings[name] = duration
            all_channels[name] = results

        # Use appropriate weights based on whether PageIndex participated
        # PageIndex results bypass RRF — they're already LLM-reasoned answers
        # and can't compete in multi-channel rank fusion.
        # Both WEIGHTS and WEIGHTS_LOCAL_ONLY cover the 5 local channels.
        # Use LOCAL_ONLY when no PageIndex (slightly different weight distribution).
        active_weights = self.WEIGHTS_LOCAL_ONLY

        # Extract PageIndex results before RRF (they'll be injected post-fusion)
        pageindex_raw = all_channels.pop("pageindex", [])

        fused = self._rrf_fusion(all_channels, top_k * 2, active_weights)  # Over-retrieve for reranking

        # Cross-encoder reranking on the fused local results only
        t_rerank = time.time()
        fused = self._cross_encoder_rerank(query, fused, top_k)
        rerank_ms = (time.time() - t_rerank) * 1000

        # Inject PageIndex results at the TOP of the final list
        # These bypass RRF and reranking because they're already
        # high-quality LLM-reasoned answers from the document tree
        if pageindex_raw:
            from src.models import CausalMemoryObject, MemoryType, EmotionLabel
            pageindex_results = []
            for rank, (memory_id, orig_score) in enumerate(pageindex_raw):
                if memory_id.startswith("pageindex:"):
                    parts = memory_id.split(":", 3)
                    page_num = parts[1] if len(parts) > 1 else "0"
                    pi_doc_id = parts[2] if len(parts) > 2 else ""
                    content_text = parts[3] if len(parts) > 3 else ""
                    if not content_text or len(content_text.strip()) < 10:
                        continue
                    synthetic_memory = CausalMemoryObject(
                        id=memory_id[:200],  # Keep ID short
                        content=content_text[:2000],  # Full document answer
                        memory_type=MemoryType.SEMANTIC,
                        source="pageindex",
                        metadata={"page": page_num, "pi_doc_id": pi_doc_id},
                    )
                    # Score high enough to always appear at the top
                    inject_score = max(0.70 - (rank * 0.05), 0.40)
                    pageindex_results.append(RetrievalResult(
                        memory=synthetic_memory,
                        score=inject_score,
                        channel="pageindex",
                        evidence_text=content_text[:500],
                    ))
            # Prepend PageIndex results, then local results
            fused = pageindex_results + fused
            fused = fused[:top_k]  # Trim to top_k
            # Re-add pageindex channel for trace logging
            all_channels["pageindex"] = pageindex_raw

        elapsed = (time.time() - t0) * 1000
        channel_counts = {k: len(v) for k, v in all_channels.items()}
        print(f"  🔎 Retrieved: {channel_counts} → {len(fused)} fused+reranked ({elapsed:.0f}ms)")

        # Build channel traces for observability (with real per-channel timing)
        self._last_channel_traces = []
        for ch_name, ch_results in all_channels.items():
            scores = [s for _, s in ch_results] if ch_results else []
            self._last_channel_traces.append(RetrievalChannelTrace(
                channel=ch_name,
                result_count=len(ch_results),
                top_score=round(max(scores), 4) if scores else 0.0,
                avg_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
                duration_ms=round(channel_timings.get(ch_name, 0.0), 1),
            ))
        self._last_retrieval_ms = elapsed
        self._last_rerank_ms = rerank_ms
        self._last_rerank_method = "cross_encoder" if (self.reranker and self.reranker.model is not None) else "embedding_fallback"
        self._last_fused_count = len(fused)

        return fused

    def get_last_retrieval_trace(self) -> dict:
        """Return the last retrieval operation's trace data for observability."""
        return {
            "channels": [c.to_dict() for c in getattr(self, '_last_channel_traces', [])],
            "total_ms": round(getattr(self, '_last_retrieval_ms', 0), 1),
            "rerank_ms": round(getattr(self, '_last_rerank_ms', 0), 1),
            "rerank_method": getattr(self, '_last_rerank_method', 'unknown'),
            "fused_count": getattr(self, '_last_fused_count', 0),
        }

    def retrieve_sync(self, query: MemoryQuery, top_k: int = 20) -> List[RetrievalResult]:
        """Synchronous wrapper for retrieve."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.retrieve(query, top_k))
        finally:
            loop.close()

    # ─── Channel 1: Dense Retrieval (FAISS) ─────────────────────────────

    async def _dense_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """Dense vector similarity search with batched embeddings (§3.3)."""
        if query.embedding is None:
            return []

        embedding = np.array(query.embedding, dtype=np.float32)
        results = self.vectors.search(
            embedding, top_k=top_k,
            time_start=query.time_start,
            time_end=query.time_end,
        )

        # Batch all variant texts for embedding in one call (§3.3)
        variant_texts = []
        variant_discounts = []
        if query.hyde_answer:
            variant_texts.append(query.hyde_answer)
            variant_discounts.append(0.8)
        if query.step_back_query:
            variant_texts.append(query.step_back_query)
            variant_discounts.append(0.75)
        for variant in query.multi_queries[:2]:
            variant_texts.append(variant)
            variant_discounts.append(0.85)

        if variant_texts:
            # Single batch embed call instead of N separate calls
            variant_embeddings = self.embeddings.embed_batch(variant_texts)
            variant_top_ks = [top_k // 2 if i == 0 else top_k // 3
                              for i in range(len(variant_texts))]

            for i, (var_emb, discount) in enumerate(zip(variant_embeddings, variant_discounts)):
                var_results = self.vectors.search(var_emb, top_k=variant_top_ks[i])
                seen = {mid for mid, _ in results}
                for mid, score in var_results:
                    if mid not in seen:
                        results.append((mid, score * discount))

        return results

    # ─── Channel 2: Sparse Retrieval (BM25) ─────────────────────────────

    async def _sparse_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """BM25 keyword-based retrieval using inverted index (O(q×avgDF))."""
        query_tokens = self._tokenize(query.raw_query)
        if not query_tokens:
            return []

        # Rebuild inverted index only when new memories are added
        current_count = self.metadata.count_memories()
        if current_count != self._bm25_last_count or not self._bm25_inverted:
            self._rebuild_bm25_index()

        if not self._bm25_inverted:
            return []

        # BM25 scoring via inverted index — only iterate docs matching query terms
        scores = defaultdict(float)
        k1, b = 1.5, 0.75

        for qt in query_tokens:
            if qt not in self._bm25_idf:
                continue
            idf = self._bm25_idf[qt]
            postings = self._bm25_inverted.get(qt, {})
            for mid, tf in postings.items():
                dl = self._bm25_doc_lengths.get(mid, 1)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / max(self._bm25_avg_dl, 1))
                scores[mid] += idf * numerator / denominator

        # Sort and return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Normalize scores
        if sorted_scores:
            max_score = sorted_scores[0][1]
            if max_score > 0:
                sorted_scores = [(mid, s / max_score) for mid, s in sorted_scores]

        return sorted_scores[:top_k]

    def _rebuild_bm25_index(self):
        """Rebuild the BM25 inverted index from all memories.
        Uses inverted posting lists for O(q×avgDF) retrieval (§4.6)."""
        # Build inverted index: token → {mid: term_frequency}
        self._bm25_inverted: Dict[str, Dict[str, int]] = {}
        self._bm25_doc_lengths: Dict[str, int] = {}

        batch_size = 2000
        offset = 0
        while True:
            memory_texts = self.metadata.get_memory_texts(limit=batch_size, offset=offset)
            if not memory_texts:
                break

            for mid, content in memory_texts:
                tokens = self._tokenize(content)
                self._bm25_doc_lengths[mid] = len(tokens)

                # Count term frequencies
                tf_map: Dict[str, int] = defaultdict(int)
                for t in tokens:
                    tf_map[t] += 1

                # Add to inverted index
                for token, freq in tf_map.items():
                    if token not in self._bm25_inverted:
                        self._bm25_inverted[token] = {}
                    self._bm25_inverted[token][mid] = freq

            if len(memory_texts) < batch_size:
                break
            offset += len(memory_texts)

        self._bm25_doc_count = len(self._bm25_doc_lengths)
        self._bm25_avg_dl = (
            sum(self._bm25_doc_lengths.values()) / max(self._bm25_doc_count, 1)
        )

        # Pre-compute IDF for all tokens in inverted index
        self._bm25_idf = {}
        for token, postings in self._bm25_inverted.items():
            df = len(postings)
            self._bm25_idf[token] = math.log(
                (self._bm25_doc_count - df + 0.5) / (df + 0.5) + 1
            )

        self._bm25_last_count = self.metadata.count_memories()

    # ─── Channel 3: Graph Retrieval ──────────────────────────────────────

    async def _graph_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """Knowledge graph traversal."""
        results = []
        graph_backend = self.graph

        if (
            not query.entities
            or graph_backend is None
            or not hasattr(graph_backend, "find_entity_by_name")
            or not hasattr(graph_backend, "get_entity_memories")
            or not hasattr(graph_backend, "get_neighbors")
            or not hasattr(graph_backend, "get_causal_chain")
        ):
            return results

        for entity_name in query.entities:
            entity_id = graph_backend.find_entity_by_name(entity_name)
            if entity_id:
                # Get entity's memories
                memory_ids = graph_backend.get_entity_memories(entity_id)
                for mid in memory_ids:
                    results.append((mid, 0.8))

                # Get neighbors' memories (2-hop)
                neighbors = graph_backend.get_neighbors(entity_id, max_hops=2)
                for neighbor in neighbors:
                    n_mids = neighbor.get("memory_ids", [])
                    hop_discount = 1.0 / (neighbor.get("hop_distance", 1) + 1)
                    for mid in n_mids:
                        results.append((mid, 0.5 * hop_discount))

        # For causal queries, also trace causal chains
        if query.intent == QueryIntent.CAUSAL:
            for entity_name in query.entities:
                entity_id = graph_backend.find_entity_by_name(entity_name)
                if entity_id:
                    chain = graph_backend.get_causal_chain(entity_id, direction="backward")
                    for node in chain:
                        for mid in node.get("memory_ids", []):
                            results.append((mid, 0.9))

        # Deduplicate
        seen = {}
        for mid, score in results:
            if mid not in seen or score > seen[mid]:
                seen[mid] = score

        return list(seen.items())[:top_k]

    # ─── Channel 4: Temporal Retrieval ───────────────────────────────────

    async def _temporal_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """Time-filtered retrieval."""
        if not query.time_start and not query.time_end:
            return []

        memories = self.metadata.search_by_time(
            start=query.time_start,
            end=query.time_end,
            limit=top_k,
        )

        # Also filter by topic/entity if available
        results = []
        for mem in memories:
            score = 0.7
            # Boost if topic matches
            if query.topics and any(t in mem.topics for t in query.topics):
                score += 0.2
            # Boost if entity matches
            if query.entities and any(e.lower() in [x.lower() for x in mem.entities] for e in query.entities):
                score += 0.2
            results.append((mem.id, min(score, 1.0)))

        return results

    # ─── Legacy Compatibility Channels ────────────────────────────────

    async def _raptor_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """Retrieve RAPTOR summary nodes and expand to their child memories.

        This channel scores hierarchical summary memories (`raptor_level > 0`) and
        projects relevance down to the summary's child IDs.
        """
        try:
            # Keep this bounded; RAPTOR summaries should be sparse compared to leaves.
            scan_limit = max(200, min(top_k * 30, 2000))
            candidates = self.metadata.get_all_memories(limit=scan_limit)
        except Exception:
            return []

        raptor_nodes = [
            m for m in candidates
            if m.raptor_level > 0 or str(m.source).lower() == "raptor" or bool(m.raptor_children)
        ]
        if not raptor_nodes:
            return []

        query_tokens = set(self._tokenize(query.raw_query))
        query_emb = None
        if query.embedding is not None:
            query_emb = np.array(query.embedding, dtype=np.float32)
            qnorm = float(np.linalg.norm(query_emb))
            if qnorm > 0:
                query_emb = query_emb / qnorm
            else:
                query_emb = None

        requested_entities = {str(entity).strip().lower() for entity in (query.entities or []) if entity}
        requested_topics = {str(topic).strip().lower() for topic in (query.topics or []) if topic}
        scored: Dict[str, float] = {}

        def _token_overlap(text: str) -> float:
            if not query_tokens:
                return 0.0
            text_tokens = set(self._tokenize(text))
            if not text_tokens:
                return 0.0
            return len(query_tokens & text_tokens) / max(len(query_tokens), 1)

        for summary in raptor_nodes:
            if query.time_start and summary.timestamp and summary.timestamp < query.time_start:
                continue
            if query.time_end and summary.timestamp and summary.timestamp > query.time_end:
                continue

            lexical = _token_overlap(summary.content)

            summary_entities = {str(entity).strip().lower() for entity in (summary.entities or []) if entity}
            entity_hits = 0.0
            if requested_entities:
                entity_hits = sum(
                    1
                    for entity in requested_entities
                    if entity in summary_entities or entity in summary.content.lower()
                ) / max(len(requested_entities), 1)

            summary_topics = {str(topic).strip().lower() for topic in (summary.topics or []) if topic}
            topic_hits = 0.0
            if requested_topics:
                topic_hits = sum(
                    1
                    for topic in requested_topics
                    if topic in summary_topics
                ) / max(len(requested_topics), 1)

            semantic = 0.0
            if query_emb is not None:
                summary_emb = None
                if summary.embedding is not None:
                    summary_emb = np.array(summary.embedding, dtype=np.float32)
                else:
                    summary_emb = self.vectors.vectors.get(summary.id)

                if summary_emb is not None:
                    summary_emb = np.array(summary_emb, dtype=np.float32)
                    snorm = float(np.linalg.norm(summary_emb))
                    if snorm > 0:
                        semantic = float(np.dot(query_emb, summary_emb / snorm))
                        semantic = max(semantic, 0.0)

            base_score = (
                0.15
                + (0.45 * semantic)
                + (0.25 * lexical)
                + (0.10 * float(summary.importance or 0.0))
                + (0.03 * topic_hits)
                + (0.02 * entity_hits)
            )

            if base_score > 0:
                scored[summary.id] = max(scored.get(summary.id, 0.0), base_score)

            for child_rank, child_id in enumerate((summary.raptor_children or [])[:10]):
                child = self.metadata.get_memory(child_id)
                if child is None:
                    continue
                if query.time_start and child.timestamp and child.timestamp < query.time_start:
                    continue
                if query.time_end and child.timestamp and child.timestamp > query.time_end:
                    continue

                child_lexical = _token_overlap(child.content)
                rank_discount = max(0.55, 0.88 - (child_rank * 0.04))
                child_score = (base_score * rank_discount) + (0.08 * child_lexical)
                if child_score > 0:
                    scored[child_id] = max(scored.get(child_id, 0.0), child_score)

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    async def _community_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """Retrieve memories via graph community summaries.

        Communities are coarse topical clusters. This channel maps the query to
        relevant clusters and returns member memory IDs with graded relevance.
        """
        graph_backend = self.graph
        if graph_backend is None or not hasattr(graph_backend, "get_community_summaries"):
            return []

        try:
            communities = graph_backend.get_community_summaries()
        except Exception:
            return []

        if not communities:
            return []

        query_tokens = set(self._tokenize(query.raw_query))
        requested_entities = {str(entity).strip().lower() for entity in (query.entities or []) if entity}
        requested_topics = {str(topic).strip().lower() for topic in (query.topics or []) if topic}

        scored: Dict[str, float] = {}

        for community in communities:
            members = [str(member) for member in (community.get("members", []) or []) if member]
            memory_ids = [str(memory_id) for memory_id in (community.get("memory_ids", []) or []) if memory_id]
            if not memory_ids:
                continue

            member_blob = " ".join(members)
            member_tokens = set(self._tokenize(member_blob))
            lexical = 0.0
            if query_tokens and member_tokens:
                lexical = len(query_tokens & member_tokens) / max(len(query_tokens), 1)

            entity_hit_score = 0.0
            if requested_entities:
                member_blob_lower = member_blob.lower()
                entity_hit_score = sum(
                    1
                    for entity in requested_entities
                    if entity in member_blob_lower
                ) / max(len(requested_entities), 1)

            size = float(community.get("size", 0) or 0)
            community_score = 0.12 + (0.35 * lexical) + (0.30 * entity_hit_score) + min(size / 25.0, 0.12)

            for rank, memory_id in enumerate(memory_ids[: max(top_k, 8)]):
                memory = self.metadata.get_memory(memory_id)
                if memory is None:
                    continue
                if query.time_start and memory.timestamp and memory.timestamp < query.time_start:
                    continue
                if query.time_end and memory.timestamp and memory.timestamp > query.time_end:
                    continue

                memory_tokens = set(self._tokenize(memory.content))
                overlap = 0.0
                if query_tokens and memory_tokens:
                    overlap = len(query_tokens & memory_tokens) / max(len(query_tokens), 1)

                topic_hits = 0.0
                if requested_topics:
                    memory_topics = {str(topic).strip().lower() for topic in (memory.topics or []) if topic}
                    topic_hits = sum(1 for topic in requested_topics if topic in memory_topics) / max(len(requested_topics), 1)

                rank_discount = 1.0 / (1.0 + (rank * 0.25))
                score = (community_score * rank_discount) + (0.25 * overlap) + (0.10 * topic_hits)
                if score > 0:
                    scored[memory_id] = max(scored.get(memory_id, 0.0), score)

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    # ─── Channel 5: Proposition Retrieval ────────────────────────────────

    async def _proposition_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """Atomic fact-level retrieval with pre-indexed proposition embeddings."""
        if query.embedding is None:
            return []

        query_emb = np.array(query.embedding, dtype=np.float32)

        # Rebuild proposition index if needed (only on new memories)
        current_count = self.metadata.count_memories()
        if current_count != self._prop_last_count or not self._prop_index:
            self._rebuild_proposition_index()

        # Score each pre-computed proposition embedding against query
        results = []
        for mid, prop_entries in self._prop_index.items():
            for prop_text, prop_emb in prop_entries:
                sim = float(np.dot(query_emb, prop_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(prop_emb) + 1e-10
                ))
                if sim > 0.4:
                    results.append((mid, sim))

        # Deduplicate (keep best score per memory)
        best = {}
        for mid, score in results:
            if mid not in best or score > best[mid]:
                best[mid] = score

        sorted_results = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _rebuild_proposition_index(self):
        """Pre-compute proposition embeddings for all memories.
        Uses batch embedding to minimize API calls (§4.6)."""
        self._prop_index = {}

        memory_batch_size = 1000
        embed_batch_size = 256
        offset = 0
        total_props = 0

        while True:
            memory_props = self.metadata.get_memory_propositions(
                limit=memory_batch_size,
                offset=offset,
            )
            if not memory_props:
                break

            flat_props = []  # [(mem_id, prop_text), ...]
            for mem_id, propositions in memory_props:
                for prop in propositions:
                    if isinstance(prop, str) and prop.strip():
                        flat_props.append((mem_id, prop.strip()))

            for i in range(0, len(flat_props), embed_batch_size):
                batch = flat_props[i:i + embed_batch_size]
                if not batch:
                    continue

                texts = [prop_text for _, prop_text in batch]
                embeddings = self.embeddings.embed_batch(texts)

                for (mem_id, prop_text), prop_emb in zip(batch, embeddings):
                    if mem_id not in self._prop_index:
                        self._prop_index[mem_id] = []
                    self._prop_index[mem_id].append((prop_text, prop_emb))

                total_props += len(batch)

            if len(memory_props) < memory_batch_size:
                break
            offset += len(memory_props)

        self._prop_last_count = self.metadata.count_memories()
        print(
            f"  📋 Proposition index rebuilt: {total_props} propositions "
            f"across {len(self._prop_index)} memories"
        )

    # ─── Channel 6: PageIndex Document Retrieval ──────────────────────

    async def _pageindex_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
        """
        PageIndex reasoning-based document retrieval (cloud API).
        Uses the Chat API for a direct answer from uploaded documents.
        Returns a single high-quality synthetic result containing the
        PageIndex answer with document context.

        Strategy: Use chat_retrieve() (non-streaming, fast) instead of
        retrieve_sections() (streaming JSON extraction, slow ~30-40s).
        This gets a direct LLM-reasoned answer from the document tree
        in ~5-10s instead of ~40s.

        Gracefully returns empty on any failure (network, timeout, budget).
        """
        if not self.pageindex_store or not self.pageindex_store.has_documents:
            return []

        try:
            # Get ready doc_ids only (skip still-processing docs)
            ready_ids = self.pageindex_store.get_doc_ids_for_query(query.raw_query)
            if not ready_ids:
                return []

            # Use fast non-streaming chat retrieval with a retrieval-focused prompt
            retrieval_query = (
                f"Answer the following question using ONLY the document content. "
                f"Be specific, include facts, numbers, and details from the document. "
                f"Do NOT add preamble or commentary.\n\n"
                f"Question: {query.raw_query}"
            )
            answer = self.pageindex_store.chat_retrieve(
                query=retrieval_query,
                doc_ids=ready_ids,
                stream=False,
            )

            if not answer or len(answer.strip()) < 10:
                return []

            # Build synthetic results — split the answer into chunks for
            # better RRF participation (each chunk = separate memory)
            results = []

            # Clean the answer — remove PageIndex API preamble and JSON metadata
            import re
            clean_answer = answer.strip()
            # Remove inline JSON metadata like {"doc_name": "..."} or {"doc_name": "...", "pages": "..."}
            clean_answer = re.sub(r'\{["\']doc_name["\'][^}]*\}', '', clean_answer)
            # Remove citation tags like <doc=...; page=N>
            clean_answer = re.sub(r'<doc=[^>]*>', '', clean_answer)
            # Remove preamble — find where substantial content starts
            # PageIndex often prepends "I'll retrieve..." before the real answer
            lines = clean_answer.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip preamble lines (short, conversational, no substance)
                if i < 3 and (
                    stripped.lower().startswith("i'll ") or
                    stripped.lower().startswith("let me ") or
                    stripped.lower().startswith("i will ") or
                    len(stripped) < 5
                ):
                    start_idx = i + 1
                else:
                    break
            clean_answer = '\n'.join(lines[start_idx:]).strip()
            # Clean up extra whitespace
            clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer).strip()

            if len(clean_answer) < 10:
                return []

            # Primary result: full answer (store full content in the ID —
            # the retriever extracts it via split(":", 3) to build the memory)
            doc_id = ready_ids[0] if ready_ids else ""
            synthetic_id = f"pageindex:0:{doc_id}:{clean_answer[:2000]}"
            results.append((synthetic_id, 1.0))

            # If answer is long enough, split into paragraph-level chunks
            # so multiple PageIndex results participate in RRF
            paragraphs = [p.strip() for p in clean_answer.split("\n") if len(p.strip()) > 30]
            for i, para in enumerate(paragraphs[:4]):  # Max 4 additional chunks
                chunk_id = f"pageindex:{i+1}:{doc_id}:{para[:800]}"
                if chunk_id != synthetic_id:  # Avoid duplicates
                    results.append((chunk_id, max(0.9 - (i * 0.15), 0.3)))

            return results

        except Exception as e:
            print(f"  ⚠ PageIndex channel failed: {e}")
            return []  # Graceful degradation — other local channels still work

    # ─── RRF Fusion ──────────────────────────────────────────────────────

    def _rrf_fusion(self, channels: Dict[str, List[Tuple[str, float]]],
                    top_k: int,
                    weights: Optional[Dict[str, float]] = None) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion across all channels."""
        if weights is None:
            weights = self.WEIGHTS
        fused_scores: Dict[str, float] = defaultdict(float)
        memory_channels: Dict[str, List[str]] = defaultdict(list)

        for channel_name, results in channels.items():
            weight = weights.get(channel_name, 0.1)
            for rank, (memory_id, _score) in enumerate(results):
                rrf_score = weight / (self.RRF_K + rank + 1)
                fused_scores[memory_id] += rrf_score
                memory_channels[memory_id].append(channel_name)

        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        # Build RetrievalResult objects
        results = []
        for memory_id in sorted_ids[:top_k]:
            # Handle PageIndex synthetic results (pageindex:page:doc_id:content_hash)
            if memory_id.startswith("pageindex:"):
                parts = memory_id.split(":", 3)
                page_num = parts[1] if len(parts) > 1 else "0"
                pi_doc_id = parts[2] if len(parts) > 2 else ""
                content_text = parts[3] if len(parts) > 3 else ""
                # Create a synthetic memory object for PageIndex results
                from src.models import CausalMemoryObject, MemoryType, EmotionLabel
                synthetic_memory = CausalMemoryObject(
                    id=memory_id,
                    content=content_text[:500] if content_text else f"[PageIndex document content from page {page_num}]",
                    memory_type=MemoryType.SEMANTIC,
                    source="pageindex",
                    metadata={"page": page_num, "pi_doc_id": pi_doc_id},
                )
                results.append(RetrievalResult(
                    memory=synthetic_memory,
                    score=fused_scores[memory_id],
                    channel=", ".join(memory_channels[memory_id]),
                    evidence_text=synthetic_memory.content[:200],
                ))
            else:
                memory = self.metadata.get_memory(memory_id)
                if memory:
                    results.append(RetrievalResult(
                        memory=memory,
                        score=fused_scores[memory_id],
                        channel=", ".join(memory_channels[memory_id]),
                        evidence_text=memory.content[:200],
                    ))

        return results

    # ─── Cross-Encoder Reranking ───────────────────────────────────────

    def _cross_encoder_rerank(self, query: MemoryQuery, results: List[RetrievalResult],
                               top_k: int) -> List[RetrievalResult]:
        """
        Real cross-encoder reranking using BGE-reranker-v2-m3.
        Falls back to embedding-based scoring if cross-encoder is unavailable.
        """
        if not results:
            return results[:top_k]

        # ── Strategy A: Real Cross-Encoder (if available) ──
        if self.reranker and self.reranker.model is not None:
            try:
                # Separate PageIndex results from local results —
                # cross-encoder wasn't trained on PageIndex document content
                # so it unfairly demotes them. Keep PageIndex scores from RRF.
                pageindex_results = [r for r in results if r.memory.source == "pageindex"]
                local_results = [r for r in results if r.memory.source != "pageindex"]

                documents = [r.memory.content[:512] for r in local_results]
                if documents:
                    reranked_indices = self.reranker.rerank(
                        query.raw_query, documents, top_k=top_k
                    )

                    reranked_local = []
                    for idx, score in reranked_indices:
                        r = local_results[idx]
                        # Blend cross-encoder score with original RRF score
                        blended = 0.70 * score + 0.20 * r.score + 0.10 * r.memory.importance
                        r.score = round(blended, 4)
                        reranked_local.append(r)
                else:
                    reranked_local = []

                # Merge: PageIndex results keep their RRF scores, then local reranked
                merged = pageindex_results + reranked_local
                merged.sort(key=lambda r: r.score, reverse=True)
                return merged[:top_k]
            except Exception as e:
                print(f"  ⚠ Cross-encoder rerank failed: {e}, falling back to embedding")

        # ── Strategy B: Embedding-based fallback ──
        if query.embedding is None:
            return results[:top_k]

        query_emb = np.array(query.embedding, dtype=np.float32)

        scored = []
        for r in results:
            # Compute embedding-based relevance
            if r.memory.embedding:
                mem_emb = np.array(r.memory.embedding, dtype=np.float32)
            else:
                mem_emb = self.embeddings.embed(r.memory.content)
            semantic_sim = float(np.dot(query_emb, mem_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem_emb) + 1e-10
            ))

            # Lexical overlap boost
            query_tokens = set(self._tokenize(query.raw_query))
            doc_tokens = set(self._tokenize(r.memory.content))
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)

            # Entity match boost
            entity_boost = 0.0
            if query.entities:
                for ent in query.entities:
                    if ent.lower() in r.memory.content.lower():
                        entity_boost += 0.05

            # Combined rerank score
            rerank_score = (
                0.50 * semantic_sim +
                0.25 * r.score +
                0.15 * overlap +
                0.10 * min(entity_boost, 0.2) +
                0.05 * r.memory.importance
            )
            scored.append((r, rerank_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for r, score in scored[:top_k]:
            r.score = round(score, 4)
            reranked.append(r)

        return reranked

    def invalidate_caches(self):
        """Mark BM25 and proposition caches as stale (lazy rebuild on next query)."""
        self._bm25_last_count = 0
        self._prop_last_count = 0

    def incremental_bm25_add(self, memory_id: str, content: str):
        """Add a single document to BM25 inverted index without full rebuild (§3.1).
        Called after ingestion to avoid O(n) rebuild on every message."""
        if not self._bm25_inverted:
            # Index hasn't been built yet; let it build on first query
            return
        tokens = self._tokenize(content)
        self._bm25_doc_lengths[memory_id] = len(tokens)

        # Count term frequencies for this doc
        tf_map: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1

        # Add to inverted index
        for token, freq in tf_map.items():
            if token not in self._bm25_inverted:
                self._bm25_inverted[token] = {}
            self._bm25_inverted[token][memory_id] = freq

        self._bm25_doc_count = len(self._bm25_doc_lengths)
        self._bm25_avg_dl = (
            sum(self._bm25_doc_lengths.values()) / max(self._bm25_doc_count, 1)
        )

        # Update IDF for affected tokens only
        for token in set(tokens):
            df = len(self._bm25_inverted.get(token, {}))
            self._bm25_idf[token] = math.log(
                (self._bm25_doc_count - df + 0.5) / (df + 0.5) + 1
            )
        self._bm25_last_count = self.metadata.count_memories()

    def incremental_proposition_add(self, memory_id: str, propositions: List[str]):
        """Add propositions for a single memory without full rebuild (§3.2)."""
        if not propositions or not self._prop_index and self._prop_last_count == 0:
            return  # Index not built yet
        prop_embeddings = self.embeddings.embed_batch(propositions)
        entries = list(zip(propositions, prop_embeddings))
        self._prop_index[memory_id] = entries
        self._prop_last_count = self.metadata.count_memories()

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _prefilter_by_metadata(self, query: MemoryQuery, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Compatibility helper to pre-filter candidate IDs by query metadata.

        Preserves older retriever contracts that apply time/entity/topic filters
        before rank fusion.
        """
        if not candidates:
            return []

        filtered: List[Tuple[str, float]] = []
        for memory_id, score in candidates:
            memory = self.metadata.get_memory(memory_id)
            if memory is None:
                continue

            if query.time_start and memory.timestamp and memory.timestamp < query.time_start:
                continue
            if query.time_end and memory.timestamp and memory.timestamp > query.time_end:
                continue

            if query.entities:
                memory_entities = {entity.lower() for entity in (memory.entities or [])}
                if not any(entity.lower() in memory_entities for entity in query.entities):
                    continue

            if query.topics:
                memory_topics = set(memory.topics or [])
                if not any(topic in memory_topics for topic in query.topics):
                    continue

            filtered.append((memory_id, score))

        return filtered

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "was", "are", "were", "be", "been",
                      "have", "has", "had", "do", "does", "did", "will", "would",
                      "could", "should", "may", "might", "to", "of", "in", "for",
                      "on", "with", "at", "by", "from", "it", "this", "that", "i",
                      "me", "my", "we", "our", "you", "your", "he", "she", "they"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def update_bm25_index(self):
        """Rebuild BM25 inverted index from all memories."""
        self._rebuild_bm25_index()
