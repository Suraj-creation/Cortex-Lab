"""Phase 3 orchestrator coordination tests for task-manager wiring and sidechain tracing."""

import asyncio
import os
import sys

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeLLM:
    def __init__(self):
        self.model = object()

    def generate_faithful(self, _query, evidence_texts, _session_context="", session_context=""):
        return f"synthesized with {len(evidence_texts)} evidence chunks"

    def generate(self, _prompt, max_tokens=1024, temperature=0.3):
        return f"fallback answer ({max_tokens}, {temperature})"


class _FakeRetriever:
    pass


class _FakeAgent:
    def __init__(self, name: str, delay_seconds: float = 0.0):
        self.name = name
        self.delay_seconds = delay_seconds

    async def execute(self, query, context=""):
        from src.models import AgentResponse, CausalMemoryObject, MemoryType, RetrievalResult

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        memory = CausalMemoryObject(
            content=f"{self.name} evidence for {query.raw_query}",
            memory_type=MemoryType.SEMANTIC,
            source="unit-test",
        )
        return AgentResponse(
            agent_name=self.name,
            answer=f"{self.name} answer",
            evidence=[RetrievalResult(memory=memory, score=0.72, channel="dense")],
            confidence=0.68,
            reasoning_trace=f"{self.name} reasoning",
        )


@pytest.mark.asyncio
async def test_multi_step_orchestration_registers_subagent_tasks_and_sidechain_trace():
    from src.agents.orchestrator import AgentOrchestrator
    from src.models import MemoryQuery, PipelineTrace, QueryIntent
    from src.runtime.contracts import TaskState
    from src.runtime.task_manager import RuntimeTaskManager

    task_manager = RuntimeTaskManager()
    orchestrator = AgentOrchestrator(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        analyzer=object(),
        transformer=object(),
        runtime_task_manager=task_manager,
    )
    orchestrator.agents = {
        "causal": _FakeAgent("causal"),
        "timeline": _FakeAgent("timeline"),
        "planning": _FakeAgent("planning"),
        "reflection": _FakeAgent("reflection"),
        "arbitration": _FakeAgent("arbitration"),
    }

    query = MemoryQuery(
        raw_query="why did this happen",
        intent=QueryIntent.CAUSAL,
        complexity=0.9,
        sub_queries=["sub-question A"],
        embedding=[0.1, 0.2, 0.3],
    )
    trace = PipelineTrace(query=query.raw_query)

    response = await orchestrator._handle_multi_step(query, trace=trace)

    parent_task_id = f"coord-{trace.trace_id}"
    parent_task = task_manager.get_task(parent_task_id)
    assert parent_task.lifecycle.state == TaskState.COMPLETED
    assert trace.coordinator_task_id == parent_task_id
    assert len(parent_task.child_task_ids) == len(response.agents_used)

    for child_task_id in parent_task.child_task_ids:
        child_task = task_manager.get_task(child_task_id)
        assert child_task.parent_task_id == parent_task_id
        assert child_task.lifecycle.state == TaskState.COMPLETED

    assert trace.coordinator_plan.get("primary_agent") == "causal"
    assert len(trace.subagent_spawn_records) == len(response.agents_used)
    assert any(
        event.get("event") == "subagent_completed"
        for event in trace.sidechain_transcript
    )


@pytest.mark.asyncio
async def test_multi_step_orchestration_subagent_runs_are_cancellable_via_runtime_task_manager():
    from src.agents.orchestrator import AgentOrchestrator
    from src.models import MemoryQuery, PipelineTrace, QueryIntent
    from src.runtime.contracts import TaskState
    from src.runtime.task_manager import RuntimeTaskManager

    task_manager = RuntimeTaskManager()
    orchestrator = AgentOrchestrator(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        analyzer=object(),
        transformer=object(),
        runtime_task_manager=task_manager,
    )
    orchestrator.agents = {
        "causal": _FakeAgent("causal", delay_seconds=30),
        "timeline": _FakeAgent("timeline", delay_seconds=30),
        "planning": _FakeAgent("planning", delay_seconds=30),
        "reflection": _FakeAgent("reflection", delay_seconds=30),
        "arbitration": _FakeAgent("arbitration", delay_seconds=30),
    }

    query = MemoryQuery(
        raw_query="why did this happen",
        intent=QueryIntent.CAUSAL,
        complexity=0.9,
        sub_queries=["sub-question A"],
        embedding=[0.1, 0.2, 0.3],
    )
    trace = PipelineTrace(query=query.raw_query)

    orchestration_task = asyncio.create_task(orchestrator._handle_multi_step(query, trace=trace))
    parent_task_id = f"coord-{trace.trace_id}"

    parent_task = None
    for _ in range(200):
        try:
            candidate = task_manager.get_task(parent_task_id)
            if candidate.child_task_ids:
                parent_task = candidate
                break
        except KeyError:
            pass
        await asyncio.sleep(0.01)

    assert parent_task is not None

    cancelled_ids = task_manager.cancel_task(parent_task_id, reason="operator cancel", propagate=True)
    assert parent_task_id in cancelled_ids

    response = await orchestration_task

    assert task_manager.get_task(parent_task_id).lifecycle.state == TaskState.CANCELLED
    assert trace.coordinator_task_id == parent_task_id
    for child_task_id in parent_task.child_task_ids:
        assert task_manager.get_task(child_task_id).lifecycle.state == TaskState.CANCELLED

    assert "cancelled" in response.reasoning_trace.lower()
