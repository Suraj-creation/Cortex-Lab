"""Tests for agent data-access tool exposure and LLM adapter fallback tool-calling."""

import os
import sys
from typing import Any

import pytest
from pydantic import BaseModel

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _DummyParams(BaseModel):
    query: str = ""


class _FakeBackend:
    def __init__(self):
        self.model = object()

    def call_function(self, _query: str, _tools: list[dict[str, Any]]) -> dict[str, Any]:
        # Simulate a backend returning a non-actionable "no tool" payload.
        return {"tool_name": "none", "response": "No tool call selected"}

    def generate(self, *_args, **_kwargs) -> str:
        return "fallback generation"


class _FakeProvider:
    def __init__(self):
        self.local_llm = _FakeBackend()
        self.gemini_llm = None
        self.provider = "local"


@pytest.mark.asyncio
async def test_llm_adapter_falls_back_to_query_personal_data_tool_when_no_tool_selected():
    from src.agents.llm_adapter import CortexLoopLLMAdapter
    from src.agents.tool_types import ToolDefinition, ToolResult

    async def _noop_execute(call_id: str, _params: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id=call_id, content="ok")

    tool = ToolDefinition(
        name="query_personal_data",
        label="Query Personal Data",
        description="Search all persisted data planes.",
        parameters_schema=_DummyParams,
        execute=_noop_execute,
    )

    adapter = CortexLoopLLMAdapter(llm_provider=_FakeProvider(), preferred_provider="local")
    response = await adapter(
        context=[{"role": "user", "content": "what did I decide about project deadlines?"}],
        tools=[tool],
    )

    assert response.get("tool_calls"), "adapter should issue a fallback retrieval tool call"
    assert response["tool_calls"][0].get("name") == "query_personal_data"
    assert "deadline" in str(response["tool_calls"][0].get("arguments", {}).get("query", "")).lower()


def test_query_personal_data_tool_is_registered():
    from src.agents.cortex_tools import query_personal_data_tool

    assert query_personal_data_tool.name == "query_personal_data"
    assert "data" in query_personal_data_tool.description.lower()
