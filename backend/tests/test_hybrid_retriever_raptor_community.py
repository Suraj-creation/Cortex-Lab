"""Tests for RAPTOR/community retrieval channels in HybridRetriever."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import CausalMemoryObject, MemoryQuery
from src.retrieval.hybrid_retriever import HybridRetriever


class _DummyEmbeddingModel:
    _backend = "stub"

    def embed(self, _text: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_batch(self, texts):
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]


class _DummyVectorStore:
    def __init__(self):
        self.vectors = {}

    def search(self, *_args, **_kwargs):
        return []


class _DummyMetadataStore:
    def __init__(self, memories):
        self._memories = {m.id: m for m in memories}

    def get_all_memories(self, limit=100, offset=0):
        ordered = list(self._memories.values())
        return ordered[offset : offset + limit]

    def get_memory(self, memory_id: str):
        return self._memories.get(memory_id)

    def count_memories(self):
        return len(self._memories)

    def get_memory_texts(self, limit=None, offset=0):
        rows = [(mid, mem.content) for mid, mem in self._memories.items()]
        if limit is None:
            return rows[offset:]
        return rows[offset : offset + limit]

    def get_memory_propositions(self, limit=None, offset=0):
        rows = [(mid, list(mem.propositions)) for mid, mem in self._memories.items() if mem.propositions]
        if limit is None:
            return rows[offset:]
        return rows[offset : offset + limit]

    def search_by_time(self, start=None, end=None, limit=50):
        return []


class _DummyGraph:
    def __init__(self, community_summaries=None):
        self._community_summaries = community_summaries or []

    def find_entity_by_name(self, _name: str):
        return None

    def get_entity_memories(self, _entity_id: str):
        return []

    def get_neighbors(self, _entity_id: str, max_hops=2):
        return []

    def get_causal_chain(self, _entity_id: str, direction="backward"):
        return []

    def get_community_summaries(self):
        return list(self._community_summaries)


class _MissingGraphMethods:
    """Simulates a retriever booted without a usable graph backend."""

    pass


def _build_memory(memory_id: str, content: str, **kwargs) -> CausalMemoryObject:
    return CausalMemoryObject(
        id=memory_id,
        content=content,
        timestamp=kwargs.pop("timestamp", datetime.now()),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_raptor_channel_returns_summary_and_children_when_relevant():
    summary = _build_memory(
        "raptor-summary-1",
        "Weekly summary about distributed systems reliability and graph retrieval.",
        raptor_level=1,
        raptor_children=["leaf-1", "leaf-2"],
        topics=["distributed systems"],
        entities=["Cortex"],
        importance=0.9,
        source="raptor",
    )
    leaf_1 = _build_memory("leaf-1", "We improved retrieval recall with better graph traversal.")
    leaf_2 = _build_memory("leaf-2", "Latency dropped after sparse and dense rank fusion tuning.")

    metadata = _DummyMetadataStore([summary, leaf_1, leaf_2])
    vectors = _DummyVectorStore()
    vectors.vectors["raptor-summary-1"] = np.array([1.0, 0.0], dtype=np.float32)

    retriever = HybridRetriever(
        _DummyEmbeddingModel(),
        vectors,
        metadata,
        _DummyGraph(),
    )

    query = MemoryQuery(
        raw_query="How did Cortex improve distributed retrieval?",
        embedding=[1.0, 0.0],
        entities=["Cortex"],
        topics=["distributed systems"],
    )

    results = await retriever._raptor_retrieve(query, top_k=5)
    result_ids = [memory_id for memory_id, _ in results]

    assert "raptor-summary-1" in result_ids
    assert "leaf-1" in result_ids
    assert any(score > 0.0 for _mid, score in results)


@pytest.mark.asyncio
async def test_community_channel_ranks_memories_from_matching_cluster():
    mem_a = _build_memory("community-m1", "Cortex retrieval improved for graph questions.")
    mem_b = _build_memory("community-m2", "Knowledge graph neighborhoods reduced misses.")
    mem_c = _build_memory("other-m3", "Completely unrelated cooking notes.")

    metadata = _DummyMetadataStore([mem_a, mem_b, mem_c])
    graph = _DummyGraph(
        community_summaries=[
            {
                "community_id": 0,
                "members": ["Cortex", "Retrieval"],
                "size": 2,
                "memory_ids": ["community-m1", "community-m2"],
            }
        ]
    )

    retriever = HybridRetriever(
        _DummyEmbeddingModel(),
        _DummyVectorStore(),
        metadata,
        graph,
    )

    query = MemoryQuery(
        raw_query="What changed in Cortex retrieval quality?",
        entities=["Cortex"],
    )

    results = await retriever._community_retrieve(query, top_k=3)
    result_ids = [memory_id for memory_id, _ in results]

    assert "community-m1" in result_ids
    assert "community-m2" in result_ids
    assert "other-m3" not in result_ids


@pytest.mark.asyncio
async def test_retrieve_pipeline_executes_raptor_and_community_channels(monkeypatch):
    memory = _build_memory("dense-1", "Fallback memory")
    metadata = _DummyMetadataStore([memory])

    retriever = HybridRetriever(
        _DummyEmbeddingModel(),
        _DummyVectorStore(),
        metadata,
        _DummyGraph(),
    )

    async def _empty(*_args, **_kwargs):
        return []

    async def _raptor(*_args, **_kwargs):
        return [("dense-1", 0.5)]

    async def _community(*_args, **_kwargs):
        return [("dense-1", 0.4)]

    monkeypatch.setattr(retriever, "_dense_retrieve", _empty)
    monkeypatch.setattr(retriever, "_sparse_retrieve", _empty)
    monkeypatch.setattr(retriever, "_graph_retrieve", _empty)
    monkeypatch.setattr(retriever, "_temporal_retrieve", _empty)
    monkeypatch.setattr(retriever, "_raptor_retrieve", _raptor)
    monkeypatch.setattr(retriever, "_community_retrieve", _community)

    seen_channels = {}

    def _capture_rrf(channels, top_k, weights=None):
        seen_channels.update(channels)
        return []

    monkeypatch.setattr(retriever, "_rrf_fusion", _capture_rrf)
    monkeypatch.setattr(retriever, "_cross_encoder_rerank", lambda *_args, **_kwargs: [])

    query = MemoryQuery(raw_query="probe", embedding=[1.0, 0.0], complexity=0.8)
    await retriever.retrieve(query, top_k=3)

    assert "raptor" in seen_channels
    assert "community" in seen_channels


@pytest.mark.asyncio
async def test_graph_channels_gracefully_skip_when_graph_backend_is_missing():
    memory = _build_memory("dense-1", "Fallback memory about retrieval stability.")
    retriever = HybridRetriever(
        _DummyEmbeddingModel(),
        _DummyVectorStore(),
        _DummyMetadataStore([memory]),
        _MissingGraphMethods(),
    )

    query = MemoryQuery(
        raw_query="What happened to Cortex retrieval quality?",
        entities=["Cortex"],
    )

    graph_results = await retriever._graph_retrieve(query, top_k=5)
    community_results = await retriever._community_retrieve(query, top_k=5)

    assert graph_results == []
    assert community_results == []
