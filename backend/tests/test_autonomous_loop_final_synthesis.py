"""Regression coverage for final-answer synthesis after tool execution."""

from __future__ import annotations

import os
import sys

import pytest
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.autonomous_loop import AgentConfig, CortexAgentLoop, SessionConfig
from src.agents.tool_types import ToolDefinition, ToolResult


class LookupMemoryParams(BaseModel):
    query: str
    metadata: dict | None = None


async def _lookup_memory(tool_call_id: str, params: dict) -> ToolResult:
    query = str(params.get("query", "") or "").strip()
    return ToolResult(
        tool_call_id=tool_call_id,
        content=f"Bob said the launch moved to Monday. Query: {query}",
    )


@pytest.mark.asyncio
async def test_agent_loop_forces_grounded_final_answer_after_tool_round():
    llm_calls: list[dict] = []

    async def fake_llm(context, tools):
        llm_calls.append(
            {
                "context": context,
                "tools": [tool.name for tool in tools],
            }
        )

        latest_user_text = ""
        for message in reversed(context):
            if str(message.get("role", "")) == "user":
                latest_user_text = str(message.get("content", "") or "")
                break

        if "Tool-call guard triggered" in latest_user_text:
            return {
                "text": "Bob said the launch moved to Monday, and there is no evidence of a further slip.",
                "tool_calls": [],
            }

        if any(str(message.get("role", "")) == "tool" for message in context):
            return {
                "text": (
                    "I've already looked into what Bob has been talking about and generated a plan "
                    "to answer your question. I will now provide you with the information based on "
                    "the retrieved memories."
                ),
                "tool_calls": [],
            }

        return {
            "text": "Let me look into Bob's updates.",
            "tool_calls": [
                {
                    "id": "tc-memory-1",
                    "name": "lookup_memory",
                    "arguments": {"query": "Bob launch update"},
                }
            ],
        }

    loop = CortexAgentLoop(
        AgentConfig(
            agent_id="test-agent",
            system_prompt="Answer from tools only.",
            tools=[
                ToolDefinition(
                    name="lookup_memory",
                    label="Lookup Memory",
                    description="Return the matching memory snippet.",
                    parameters_schema=LookupMemoryParams,
                    execute=_lookup_memory,
                )
            ],
            session_config=SessionConfig(persist=False),
        ),
        llm_fn=fake_llm,
    )

    result = await loop.prompt("What did Bob say about the launch?")

    assert (
        result["text"]
        == "Bob said the launch moved to Monday, and there is no evidence of a further slip."
    )
    assert len(llm_calls) == 3
    assert llm_calls[0]["tools"] == ["lookup_memory"]
    assert llm_calls[-1]["tools"] == []
