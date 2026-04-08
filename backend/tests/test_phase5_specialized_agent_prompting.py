"""Phase 5 tests for specialized-agent prompt layering and 15-agent orchestration contracts."""

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


class _FakeAnalyzer:
    pass


class _FakeTransformer:
    pass


class _FakeAgent:
    def __init__(self, name: str):
        self.name = name

    async def execute(self, query, context=""):
        from src.models import AgentResponse

        _ = (query, context)
        return AgentResponse(
            agent_name=self.name,
            answer=f"{self.name} answer",
            evidence=[],
            confidence=0.7,
            reasoning_trace=f"{self.name} reasoning",
        )


def test_prompt_layer_registry_has_all_15_profiles():
    from src.agents.prompt_layers import SPECIALIZED_AGENT_PROMPTS

    assert len(SPECIALIZED_AGENT_PROMPTS) == 15

    required = {
        "timeline",
        "causal",
        "reflection",
        "planning",
        "arbitration",
        "academic",
        "journaling",
        "wellbeing",
        "cognitive",
        "decisions",
        "emotional",
        "behavioral",
        "social",
        "goals",
        "meta_learning",
    }
    assert required.issubset(set(SPECIALIZED_AGENT_PROMPTS.keys()))

    for profile in SPECIALIZED_AGENT_PROMPTS.values():
        assert profile.identity
        assert profile.mission
        assert profile.must_do
        assert profile.must_not


def test_compose_specialized_system_prompt_contains_required_layers():
    from src.agents.prompt_layers import compose_specialized_system_prompt

    prompt = compose_specialized_system_prompt(
        agent_key="academic",
        query="How can I improve my exam preparation this month?",
        session_context="User has exams in 3 weeks and reported focus issues.",
    )

    assert "GLOBAL SAFETY LAYER" in prompt
    assert "RUNTIME ORCHESTRATOR LAYER" in prompt
    assert "SPECIALIZED AGENT LAYER" in prompt
    assert "MUST DO" in prompt
    assert "MUST NOT" in prompt
    assert "exam preparation" in prompt.lower()


def test_build_specialized_agents_registers_all_15_runtime_agents():
    from src.agents.specialized import build_specialized_agents

    agents = build_specialized_agents(_FakeLLM(), _FakeRetriever())

    assert len(agents) == 15
    assert "timeline" in agents
    assert "arbitration" in agents
    assert "academic" in agents
    assert "meta_learning" in agents


def test_orchestrator_select_domain_specialists_from_query_signals():
    from src.agents.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        analyzer=_FakeAnalyzer(),
        transformer=_FakeTransformer(),
    )

    specialists = orchestrator._select_domain_specialists(
        "Can you analyze my stress, sleep quality, and mood recovery patterns?"
    )

    assert "wellbeing" in specialists
    assert "emotional" in specialists


def test_orchestrator_single_step_prefers_domain_specialist_for_broad_intent():
    from src.agents.orchestrator import AgentOrchestrator
    from src.models import MemoryQuery, QueryIntent

    orchestrator = AgentOrchestrator(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        analyzer=_FakeAnalyzer(),
        transformer=_FakeTransformer(),
    )
    orchestrator.agents = {
        "planning": _FakeAgent("planning"),
        "wellbeing": _FakeAgent("wellbeing"),
    }

    query = MemoryQuery(
        raw_query="Help me understand stress and sleep recovery patterns",
        intent=QueryIntent.FACTUAL,
        complexity=0.45,
        embedding=[0.1, 0.2, 0.3],
    )

    response = asyncio.run(orchestrator._handle_single_step(query))

    assert response.agents_used == ["wellbeing"]


def test_orchestrator_multi_step_caps_agent_fanout_to_five():
    from src.agents.orchestrator import AgentOrchestrator
    from src.models import MemoryQuery, QueryIntent

    orchestrator = AgentOrchestrator(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        analyzer=_FakeAnalyzer(),
        transformer=_FakeTransformer(),
    )
    orchestrator.agents = {
        "planning": _FakeAgent("planning"),
        "academic": _FakeAgent("academic"),
        "wellbeing": _FakeAgent("wellbeing"),
        "cognitive": _FakeAgent("cognitive"),
        "decisions": _FakeAgent("decisions"),
        "emotional": _FakeAgent("emotional"),
        "social": _FakeAgent("social"),
        "goals": _FakeAgent("goals"),
    }

    query = MemoryQuery(
        raw_query=(
            "Create a plan for exam preparation, stress control, cognitive focus, "
            "decision quality, emotional stability, social communication, and goals"
        ),
        intent=QueryIntent.EXPLORATORY,
        complexity=0.9,
        embedding=[0.1, 0.2, 0.3],
    )

    response = asyncio.run(orchestrator._handle_multi_step(query))

    assert "planning" in response.agents_used
    assert len(response.agents_used) <= 5
