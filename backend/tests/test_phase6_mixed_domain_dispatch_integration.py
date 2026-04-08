"""Phase 6 integration tests for analyzer-driven mixed-domain dispatch contracts."""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeLLM:
    def __init__(self):
        self.model = object()

    def generate_faithful(self, _query, _evidence_texts, session_context=""):
        _ = session_context
        return "faithful synthesis"

    def generate(self, _prompt, max_tokens=256, temperature=0.3):
        _ = (max_tokens, temperature)
        return "fallback synthesis"


class _FakeRetriever:
    pass


class _StaticAgent:
    def __init__(self, name: str, answer: str = ""):
        self.name = name
        self.answer = answer or f"{name} analysis output"

    async def execute(self, query, context=""):
        from src.models import AgentResponse

        _ = (query, context)
        return AgentResponse(
            agent_name=self.name,
            answer=self.answer,
            evidence=[],
            confidence=0.72,
            reasoning_trace=f"{self.name} reasoning",
        )


def _build_orchestrator():
    from src.agents.orchestrator import AgentOrchestrator
    from src.retrieval.query_engine import QueryAnalyzer

    return AgentOrchestrator(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        analyzer=QueryAnalyzer(),
        transformer=object(),
    )


def test_integration_real_analyzer_single_step_prefers_domain_specialist():
    from src.retrieval.query_engine import QueryAnalyzer
    from src.models import RoutingStrategy

    analyzer = QueryAnalyzer()
    query = analyzer.analyze("What is my stress level?")
    assert query.routing == RoutingStrategy.SINGLE_STEP

    orchestrator = _build_orchestrator()
    orchestrator.agents = {
        "planning": _StaticAgent("planning"),
        "wellbeing": _StaticAgent("wellbeing"),
    }

    response = asyncio.run(orchestrator._handle_single_step(query))

    assert response.agents_used == ["wellbeing"]


def test_integration_real_analyzer_multi_step_dispatches_mixed_domain_specialists():
    from src.retrieval.query_engine import QueryAnalyzer
    from src.models import RoutingStrategy

    analyzer = QueryAnalyzer()
    query = analyzer.analyze(
        "Compare how my exam preparation, stress, decision quality, and social communication "
        "have evolved over time and explain why."
    )
    assert query.routing == RoutingStrategy.MULTI_STEP

    orchestrator = _build_orchestrator()
    orchestrator.agents = {
        "causal": _StaticAgent("causal"),
        "timeline": _StaticAgent("timeline"),
        "planning": _StaticAgent("planning"),
        "academic": _StaticAgent("academic"),
        "wellbeing": _StaticAgent("wellbeing"),
        "decisions": _StaticAgent("decisions"),
        "social": _StaticAgent("social"),
        "arbitration": _StaticAgent("arbitration"),
    }

    response = asyncio.run(orchestrator._handle_multi_step(query))
    specialists = {"academic", "wellbeing", "decisions", "social"} & set(response.agents_used)

    assert len(response.agents_used) <= 5
    assert len(specialists) >= 2
    assert any(agent in response.agents_used for agent in {"causal", "timeline", "arbitration", "planning"})


def test_integration_conflict_path_invokes_arbitration_first_when_outputs_disagree():
    from src.models import PipelineTrace, RoutingStrategy
    from src.retrieval.query_engine import QueryAnalyzer

    analyzer = QueryAnalyzer()
    query = analyzer.analyze(
        "Why do my recent notes say stress is improving but also say stress is not improving over time?"
    )
    assert query.routing == RoutingStrategy.MULTI_STEP

    orchestrator = _build_orchestrator()
    orchestrator.agents = {
        "causal": _StaticAgent(
            "causal",
            answer="Stress is improving over time and sleep quality is getting better.",
        ),
        "timeline": _StaticAgent(
            "timeline",
            answer="Stress is not improving over time and sleep quality is not getting better.",
        ),
        "arbitration": _StaticAgent(
            "arbitration",
            answer="The latest evidence is mixed, but recent trends suggest partial recovery.",
        ),
        "planning": _StaticAgent("planning"),
    }

    trace = PipelineTrace(query=query.raw_query)
    response = asyncio.run(orchestrator._handle_multi_step(query, trace=trace))

    assert "arbitration" in response.agents_used
    assert trace.coordinator_plan.get("conflict_resolution_path") == "arbitration_first"
    assert trace.coordinator_plan.get("plan_mode", {}).get("metadata", {}).get("arbitration_invoked") is True


def test_integration_plan_mode_confirmation_gate_blocks_unconfirmed_execution():
    from src.models import PipelineTrace, RoutingStrategy
    from src.retrieval.query_engine import QueryAnalyzer

    analyzer = QueryAnalyzer()
    query = analyzer.analyze(
        "Compare how my exam preparation, stress, and decisions changed over time and explain why."
    )
    assert query.routing == RoutingStrategy.MULTI_STEP
    query.metadata["plan_confirmation_required"] = True

    orchestrator = _build_orchestrator()
    orchestrator.agents = {
        "causal": _StaticAgent("causal"),
        "timeline": _StaticAgent("timeline"),
        "planning": _StaticAgent("planning"),
        "academic": _StaticAgent("academic"),
        "wellbeing": _StaticAgent("wellbeing"),
    }

    trace = PipelineTrace(query=query.raw_query)
    response = asyncio.run(orchestrator._handle_multi_step(query, trace=trace))

    assert response.agents_used == ["plan_mode_confirmation_required"]
    assert trace.coordinator_plan.get("confirmation_gate", {}).get("required") is True
    assert trace.coordinator_plan.get("confirmation_gate", {}).get("status") == "pending"
