"""
AgentConfig Definitions — All Cortex agents as configurations.
Architecture: Orchestrator.md §21.4, Agentic-RAG-Architecture.md §17.3

Each agent is a CONFIGURATION of the same CortexAgentLoop.
The only things that vary: system_prompt, tools, extensions, session_config.
"""

from __future__ import annotations

from src.agents.autonomous_loop import (
    AgentConfig, SessionConfig, ScheduleConfig, RetryConfig,
)
from src.agents.cortex_tools import (
    retrieve_memory_tool, search_wiki_tool, search_claims_tool,
    query_graph_tool, search_by_time_tool, classify_query_tier_tool,
    analyze_query_intent_tool, spawn_agent_tool, collect_agent_results_tool,
    dissolve_team_tool, arbitrate_conflict_tool, compress_evidence_tool,
    generate_answer_plan_tool, build_event_timeline_tool,
    detect_temporal_gaps_tool, trace_causal_chain_tool,
    detect_belief_change_tool, analyze_pattern_tool, decompose_query_tool,
    extract_claims_tool, upsert_claim_tool, patch_wiki_page_tool,
    create_wiki_page_tool, lint_wiki_page_tool, compact_wiki_section_tool,
    ingest_memory_tool, update_graph_edge_tool, invalidate_cache_tool,
    score_importance_tool, assemble_context_tool, score_initiative_tool,
    detect_idle_tool, read_mood_signal_tool,
)
from src.agents.cortex_prompts import (
    L0_MASTER_PROMPT, L1_ORCHESTRATOR_PROMPT, TIMELINE_AGENT_PROMPT,
    CAUSAL_AGENT_PROMPT, REFLECTION_AGENT_PROMPT, PLANNING_AGENT_PROMPT,
    ARBITRATION_AGENT_PROMPT, ACADEMIC_AGENT_PROMPT, JOURNALING_AGENT_PROMPT,
    WELLBEING_AGENT_PROMPT, COGNITIVE_AGENT_PROMPT, DECISION_LOG_AGENT_PROMPT,
    EMOTIONAL_AGENT_PROMPT, BEHAVIORAL_AGENT_PROMPT, SOCIAL_AGENT_PROMPT,
    GOAL_AGENT_PROMPT, META_LEARNING_AGENT_PROMPT, WIKI_AGENT_PROMPT,
    PRESENCE_AGENT_PROMPT, SESSION_CRYSTALLIZER_PROMPT,
)


# ── L0 Master Orchestrator ────────────────────────────────────────────────────

L0_CONFIG = AgentConfig(
    agent_id="l0_master",
    system_prompt=L0_MASTER_PROMPT,
    tools=[],
    extensions=[],
    session_config=SessionConfig(persist=True, compact_threshold=0.8, max_age_hours=24),
    scheduling=ScheduleConfig(always_on=True),
    max_turns=100,
)


# ── L1 Runtime Orchestrator (per-query) ───────────────────────────────────────

L1_CONFIG = AgentConfig(
    agent_id="l1_orchestrator",
    system_prompt=L1_ORCHESTRATOR_PROMPT,
    tools=[
        retrieve_memory_tool, search_wiki_tool, search_claims_tool,
        query_graph_tool, classify_query_tier_tool, analyze_query_intent_tool,
        spawn_agent_tool, collect_agent_results_tool, dissolve_team_tool,
        arbitrate_conflict_tool, compress_evidence_tool, generate_answer_plan_tool,
    ],
    extensions=[],
    session_config=SessionConfig(persist=True),
    max_turns=50,
    max_tool_chain_depth=10,
)


# ── L2 Specialized Agent Configs ──────────────────────────────────────────────

TIMELINE_CONFIG = AgentConfig(
    agent_id="timeline",
    system_prompt=TIMELINE_AGENT_PROMPT,
    tools=[
        retrieve_memory_tool, search_by_time_tool, build_event_timeline_tool,
        detect_temporal_gaps_tool, analyze_pattern_tool,
    ],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

CAUSAL_CONFIG = AgentConfig(
    agent_id="causal",
    system_prompt=CAUSAL_AGENT_PROMPT,
    tools=[
        retrieve_memory_tool, trace_causal_chain_tool, query_graph_tool,
        analyze_pattern_tool, score_importance_tool,
    ],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

REFLECTION_CONFIG = AgentConfig(
    agent_id="reflection",
    system_prompt=REFLECTION_AGENT_PROMPT,
    tools=[
        retrieve_memory_tool, detect_belief_change_tool, search_claims_tool,
        query_graph_tool, analyze_pattern_tool,
    ],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

PLANNING_CONFIG = AgentConfig(
    agent_id="planning",
    system_prompt=PLANNING_AGENT_PROMPT,
    tools=[
        retrieve_memory_tool, decompose_query_tool, analyze_pattern_tool,
        score_importance_tool,
    ],
    session_config=SessionConfig(persist=True),
    max_turns=25,
)

ARBITRATION_CONFIG = AgentConfig(
    agent_id="arbitration",
    system_prompt=ARBITRATION_AGENT_PROMPT,
    tools=[
        retrieve_memory_tool, search_claims_tool, detect_belief_change_tool,
        score_importance_tool,
    ],
    session_config=SessionConfig(persist=True),
    max_turns=15,
)

ACADEMIC_CONFIG = AgentConfig(
    agent_id="academic",
    system_prompt=ACADEMIC_AGENT_PROMPT,
    tools=[retrieve_memory_tool, search_claims_tool, analyze_pattern_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

JOURNALING_CONFIG = AgentConfig(
    agent_id="journaling",
    system_prompt=JOURNALING_AGENT_PROMPT,
    tools=[retrieve_memory_tool, search_by_time_tool, analyze_pattern_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

WELLBEING_CONFIG = AgentConfig(
    agent_id="wellbeing",
    system_prompt=WELLBEING_AGENT_PROMPT,
    tools=[retrieve_memory_tool, analyze_pattern_tool, read_mood_signal_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

COGNITIVE_CONFIG = AgentConfig(
    agent_id="cognitive",
    system_prompt=COGNITIVE_AGENT_PROMPT,
    tools=[retrieve_memory_tool, analyze_pattern_tool, detect_belief_change_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

DECISION_LOG_CONFIG = AgentConfig(
    agent_id="decision_log",
    system_prompt=DECISION_LOG_AGENT_PROMPT,
    tools=[retrieve_memory_tool, search_by_time_tool, analyze_pattern_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

EMOTIONAL_CONFIG = AgentConfig(
    agent_id="emotional",
    system_prompt=EMOTIONAL_AGENT_PROMPT,
    tools=[retrieve_memory_tool, analyze_pattern_tool, read_mood_signal_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

BEHAVIORAL_CONFIG = AgentConfig(
    agent_id="behavioral",
    system_prompt=BEHAVIORAL_AGENT_PROMPT,
    tools=[retrieve_memory_tool, analyze_pattern_tool, detect_belief_change_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

SOCIAL_CONFIG = AgentConfig(
    agent_id="social",
    system_prompt=SOCIAL_AGENT_PROMPT,
    tools=[retrieve_memory_tool, query_graph_tool, analyze_pattern_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

GOAL_CONFIG = AgentConfig(
    agent_id="goal",
    system_prompt=GOAL_AGENT_PROMPT,
    tools=[retrieve_memory_tool, analyze_pattern_tool, detect_belief_change_tool],
    session_config=SessionConfig(persist=True),
    max_turns=20,
)

META_LEARNING_CONFIG = AgentConfig(
    agent_id="meta_learning",
    system_prompt=META_LEARNING_AGENT_PROMPT,
    tools=[retrieve_memory_tool, analyze_pattern_tool, search_claims_tool],
    session_config=SessionConfig(persist=True),
    max_turns=25,
)


# ── Background Agent Configs ──────────────────────────────────────────────────

WIKI_AGENT_CONFIG = AgentConfig(
    agent_id="wiki_agent",
    system_prompt=WIKI_AGENT_PROMPT,
    tools=[
        extract_claims_tool, upsert_claim_tool, patch_wiki_page_tool,
        create_wiki_page_tool, lint_wiki_page_tool, compact_wiki_section_tool,
    ],
    session_config=SessionConfig(persist=True, compact_threshold=0.7, max_age_hours=168),
    scheduling=ScheduleConfig(on_ingest=True, daily="02:00"),
    max_turns=200,
)

PRESENCE_CONFIG = AgentConfig(
    agent_id="presence",
    system_prompt=PRESENCE_AGENT_PROMPT,
    tools=[
        assemble_context_tool, score_initiative_tool, detect_idle_tool,
        retrieve_memory_tool, search_wiki_tool, read_mood_signal_tool,
    ],
    session_config=SessionConfig(persist=True, compact_threshold=0.7),
    scheduling=ScheduleConfig(continuous=True, interval_min=30),
    max_turns=50,
)

SESSION_CRYSTALLIZER_CONFIG = AgentConfig(
    agent_id="session_crystallizer",
    system_prompt=SESSION_CRYSTALLIZER_PROMPT,
    tools=[
        retrieve_memory_tool, extract_claims_tool, upsert_claim_tool,
        patch_wiki_page_tool, analyze_pattern_tool,
    ],
    session_config=SessionConfig(persist=True, compact_threshold=0.8),
    scheduling=ScheduleConfig(interval_min=15),
    max_turns=30,
)


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_AGENT_CONFIGS: dict[str, AgentConfig] = {
    "l0_master": L0_CONFIG,
    "l1_orchestrator": L1_CONFIG,
    "timeline": TIMELINE_CONFIG,
    "causal": CAUSAL_CONFIG,
    "reflection": REFLECTION_CONFIG,
    "planning": PLANNING_CONFIG,
    "arbitration": ARBITRATION_CONFIG,
    "academic": ACADEMIC_CONFIG,
    "journaling": JOURNALING_CONFIG,
    "wellbeing": WELLBEING_CONFIG,
    "cognitive": COGNITIVE_CONFIG,
    "decision_log": DECISION_LOG_CONFIG,
    "emotional": EMOTIONAL_CONFIG,
    "behavioral": BEHAVIORAL_CONFIG,
    "social": SOCIAL_CONFIG,
    "goal": GOAL_CONFIG,
    "meta_learning": META_LEARNING_CONFIG,
    "wiki_agent": WIKI_AGENT_CONFIG,
    "presence": PRESENCE_CONFIG,
    "session_crystallizer": SESSION_CRYSTALLIZER_CONFIG,
}

L2_AGENT_IDS = [
    "timeline", "causal", "reflection", "planning", "arbitration",
    "academic", "journaling", "wellbeing", "cognitive", "decision_log",
    "emotional", "behavioral", "social", "goal", "meta_learning",
]

BACKGROUND_AGENT_IDS = ["wiki_agent", "presence", "session_crystallizer"]
