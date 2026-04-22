"""Phase 3 API tests for exposing runtime task references in user-facing RAG responses."""

import os
import sys
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as backend_server


class _FakeLLMProvider:
    def __init__(self):
        self.provider = "local"
        self.has_gemini = False
        self.local_llm = type("_Local", (), {"model": object()})()

    def set_provider(self, provider: str):
        self.provider = provider


class _FakeRAGEngine:
    def __init__(self):
        self.initialized = True
        self.llm = _FakeLLMProvider()

    async def rag_chat(self, user_message: str, session_id: str = "", conversation_history=None):
        _ = (user_message, session_id, conversation_history)
        return {
            "answer": "ok",
            "thinking": "test",
            "evidence": [],
            "agents_used": ["planning"],
            "confidence": 0.77,
            "query_analysis": {"intent": "causal", "complexity": 0.82, "routing": "multi_step"},
            "processing_time_ms": 123.4,
            "cache_hit": False,
            "pipeline_trace": {
                "trace_id": "trace-api-123",
                "coordinator_plan": {"strategy": "parallel_multi_agent", "subagent_count": 1},
                "subagent_spawn_records": [
                    {
                        "parent_task_id": "coord-trace-api-123",
                        "task_id": "subagent-trace-api-123-00-planning",
                        "agent": "planning",
                        "role": "primary",
                        "spawned_at": "2026-04-05T10:00:00+00:00",
                    }
                ],
                "sidechain_transcript": [],
            },
        }


def test_rag_chat_response_includes_runtime_task_refs(monkeypatch):
    fake_engine = _FakeRAGEngine()

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    monkeypatch.setattr(backend_server.app.router, "lifespan_context", _no_lifespan)
    monkeypatch.setattr(backend_server, "rag_engine", fake_engine)

    client = TestClient(backend_server.app)
    with client:
        response = client.post(
            "/api/rag/chat",
            json={
                "messages": [{"role": "user", "content": "why did this happen"}],
                "stream": False,
                "llm_provider": "local",
            },
        )

    assert response.status_code == 200
    payload = response.json()

    runtime_tasks = payload.get("runtime_tasks")
    assert isinstance(runtime_tasks, dict)
    assert runtime_tasks["coordinator_task_id"] == "coord-trace-api-123"
    assert "subagent-trace-api-123-00-planning" in runtime_tasks["subagent_task_ids"]
    assert runtime_tasks["api"]["list"] == "/api/runtime/tasks"
