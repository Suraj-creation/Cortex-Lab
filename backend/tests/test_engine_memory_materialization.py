from __future__ import annotations

import asyncio


class _DummyMemory:
    def __init__(self, source: str):
        self.source = source
        self.topics = ["engineering"]

    def to_dict(self):
        return {
            "id": "mem-001",
            "content": "Manual memory for wiki materialization",
            "source": self.source,
            "session_id": "session-001",
            "topics": list(self.topics),
        }


class _DummyIngestion:
    async def ingest(self, *, content: str, session_id: str, source: str):
        return _DummyMemory(source)


class _DummyCache:
    def __init__(self):
        self.invalidated_topic = None

    def invalidate_topic(self, topic: str):
        self.invalidated_topic = topic


class _DummyHybridRetriever:
    def __init__(self):
        self.invalidated = False

    def invalidate_caches(self):
        self.invalidated = True


class _DummyFlushable:
    def __init__(self):
        self.flushed = False
        self.saved = False

    def flush(self):
        self.flushed = True

    def save(self):
        self.saved = True


def _build_engine():
    from src.engine import CortexRAGEngine

    engine = CortexRAGEngine.__new__(CortexRAGEngine)
    engine.initialized = True
    engine.ingestion = _DummyIngestion()
    engine.cache = _DummyCache()
    engine.hybrid_retriever = _DummyHybridRetriever()
    engine.metadata_store = _DummyFlushable()
    engine.vector_store = _DummyFlushable()
    engine.knowledge_graph = _DummyFlushable()
    return engine


def test_ingest_memory_materializes_wiki_for_manual_sources(monkeypatch):
    calls: list[dict] = []

    def _capture(memory):
        payload = memory.to_dict() if hasattr(memory, "to_dict") else dict(memory)
        calls.append(payload)
        return {"claims_upserted": 1, "pages_created": 1}

    monkeypatch.setattr("src.wiki.materializer.materialize_memory_into_wiki", _capture)

    engine = _build_engine()
    result = asyncio.run(
        engine.ingest_memory(
            "Manual memory for wiki materialization",
            source="manual_memory",
            session_id="session-001",
        )
    )

    assert result["id"] == "mem-001"
    assert calls == [
        {
            "id": "mem-001",
            "content": "Manual memory for wiki materialization",
            "source": "manual_memory",
            "session_id": "session-001",
            "topics": ["engineering"],
        }
    ]
    assert engine.cache.invalidated_topic == "engineering"
    assert engine.hybrid_retriever.invalidated is True
    assert engine.metadata_store.flushed is True
    assert engine.vector_store.saved is True
    assert engine.knowledge_graph.saved is True


def test_ingest_memory_skips_wiki_materialization_for_chat_source(monkeypatch):
    calls: list[dict] = []

    def _capture(memory):
        payload = memory.to_dict() if hasattr(memory, "to_dict") else dict(memory)
        calls.append(payload)
        return {"claims_upserted": 1, "pages_created": 1}

    monkeypatch.setattr("src.wiki.materializer.materialize_memory_into_wiki", _capture)

    engine = _build_engine()
    result = asyncio.run(
        engine.ingest_memory(
            "Chat questions should not seed the wiki",
            source="chat",
            session_id="session-chat",
        )
    )

    assert result["source"] == "chat"
    assert calls == []
