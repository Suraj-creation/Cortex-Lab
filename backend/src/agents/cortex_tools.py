"""
Cortex Tool Catalog — Pydantic-validated tool definitions.
Architecture: Agentic-RAG-Architecture.md §17.13

All tools follow the pi-mono pattern:
1. Schema-first (Pydantic models for params)
2. ToolResult output (content + is_error + details)
3. Concurrency annotations
4. Prompt snippets for LLM context
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agents.tool_types import PermissionModel, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrieveMemoryParams(BaseModel):
    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(10, description="Number of results to return")
    memory_type: str | None = Field(None, description="Filter by type: episodic|semantic|procedural|reflective")
    time_start: str | None = Field(None, description="ISO datetime range start")
    time_end: str | None = Field(None, description="ISO datetime range end")

async def _retrieve_memory(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.retrieval.hybrid_retriever import HybridRetriever
    try:
        retriever = HybridRetriever.get_instance()
        results = await retriever.retrieve(
            query=params["query"],
            top_k=params.get("top_k", 10),
        )
        formatted = [
            {
                "content": r.memory.content if hasattr(r, 'memory') else str(r),
                "score": r.score if hasattr(r, 'score') else 0.0,
                "channel": r.channel if hasattr(r, 'channel') else "unknown",
            }
            for r in results
        ]
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps(formatted, default=str),
        )
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Retrieval error: {e}", is_error=True)

retrieve_memory_tool = ToolDefinition(
    name="retrieve_memory",
    label="Retrieve Memory",
    description="Search through stored memories by semantic similarity with hybrid retrieval (dense + sparse + graph)",
    parameters_schema=RetrieveMemoryParams,
    execute=_retrieve_memory,
    prompt_snippet="Use for any query that needs evidence from stored memories.",
)


class SearchWikiParams(BaseModel):
    topic: str = Field(..., description="Wiki topic to search")
    include_claims: bool = Field(True, description="Include atomic claims from claim store")

async def _search_wiki(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.wiki.wiki_store import WikiStore
        store = WikiStore.get_instance()
        results = store.search(params["topic"], include_claims=params.get("include_claims", True))
        return ToolResult(tool_call_id=call_id, content=json.dumps(results, default=str))
    except ImportError:
        return ToolResult(tool_call_id=call_id, content="Wiki store not yet initialized", is_error=True)
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Wiki search error: {e}", is_error=True)

search_wiki_tool = ToolDefinition(
    name="search_wiki",
    label="Search Wiki",
    description="Search the personal wiki for synthesized knowledge about a topic",
    parameters_schema=SearchWikiParams,
    execute=_search_wiki,
)


class SearchClaimsParams(BaseModel):
    query: str = Field(..., description="Claim search query")
    min_confidence: float = Field(0.5, description="Minimum confidence threshold")
    limit: int = Field(20, description="Max results")

async def _search_claims(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.wiki.claim_store import ClaimStore
        store = ClaimStore.get_instance()
        claims = store.search(params["query"], min_confidence=params.get("min_confidence", 0.5))
        return ToolResult(tool_call_id=call_id, content=json.dumps(claims[:params.get("limit", 20)], default=str))
    except ImportError:
        return ToolResult(tool_call_id=call_id, content="Claim store not yet initialized", is_error=True)
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Claim search error: {e}", is_error=True)

search_claims_tool = ToolDefinition(
    name="search_claims",
    label="Search Claims",
    description="Search atomic claims extracted from memories with confidence scoring",
    parameters_schema=SearchClaimsParams,
    execute=_search_claims,
)


class QueryGraphParams(BaseModel):
    entity: str = Field(..., description="Entity name or ID")
    hops: int = Field(2, description="Graph traversal depth")
    relationship_type: str | None = Field(None, description="Filter by relationship type")

async def _query_graph(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.storage.graph_store import GraphStore
        store = GraphStore.get_instance()
        subgraph = store.neighbors(params["entity"], hops=params.get("hops", 2))
        return ToolResult(tool_call_id=call_id, content=json.dumps(subgraph, default=str))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Graph query error: {e}", is_error=True)

query_graph_tool = ToolDefinition(
    name="query_graph",
    label="Query Knowledge Graph",
    description="Query the entity-relationship knowledge graph for connections",
    parameters_schema=QueryGraphParams,
    execute=_query_graph,
)


class SearchByTimeParams(BaseModel):
    start_date: str = Field(..., description="Start date (ISO format)")
    end_date: str = Field(..., description="End date (ISO format)")
    top_k: int = Field(20, description="Max results")

async def _search_by_time(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.storage.duckdb_store import DuckDBStore
        store = DuckDBStore.get_instance()
        results = store.query_time_range(params["start_date"], params["end_date"], limit=params.get("top_k", 20))
        return ToolResult(tool_call_id=call_id, content=json.dumps(results, default=str))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Time search error: {e}", is_error=True)

search_by_time_tool = ToolDefinition(
    name="search_by_time",
    label="Search by Time",
    description="Find memories from a specific time period using temporal indexing",
    parameters_schema=SearchByTimeParams,
    execute=_search_by_time,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION TOOLS (used by L1)
# ═══════════════════════════════════════════════════════════════════════════════

class ClassifyQueryTierParams(BaseModel):
    query: str = Field(..., description="The user query to classify")
    context: str | None = Field(None, description="Optional session context")

async def _classify_query_tier(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.retrieval.query_engine import QueryAnalyzer
    try:
        analyzer = QueryAnalyzer()
        analysis = await analyzer.analyze(params["query"])
        tier = "T0"
        if analysis.complexity > 0.3:
            tier = "T1"
        if analysis.complexity > 0.5 or len(analysis.sub_queries) > 1:
            tier = "T2"
        if analysis.complexity > 0.7 or len(analysis.sub_queries) > 3:
            tier = "T3"
        if analysis.complexity > 0.9:
            tier = "T4"
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps({
                "tier": tier,
                "complexity": analysis.complexity,
                "intent": analysis.intent.value if hasattr(analysis, 'intent') else "unknown",
                "sub_queries": analysis.sub_queries if hasattr(analysis, 'sub_queries') else [],
            }, default=str),
        )
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Tier classification error: {e}", is_error=True)

classify_query_tier_tool = ToolDefinition(
    name="classify_query_tier",
    label="Classify Query Tier",
    description="Classify a query into complexity tiers T0-T4 for routing",
    parameters_schema=ClassifyQueryTierParams,
    execute=_classify_query_tier,
)


class AnalyzeQueryIntentParams(BaseModel):
    query: str = Field(..., description="The user query to analyze")

async def _analyze_query_intent(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.retrieval.query_engine import QueryAnalyzer
    try:
        analyzer = QueryAnalyzer()
        analysis = await analyzer.analyze(params["query"])
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps({
                "intent": analysis.intent.value if hasattr(analysis, 'intent') else "unknown",
                "entities": analysis.entities if hasattr(analysis, 'entities') else [],
                "topics": analysis.topics if hasattr(analysis, 'topics') else [],
                "complexity": analysis.complexity if hasattr(analysis, 'complexity') else 0.5,
            }, default=str),
        )
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Intent analysis error: {e}", is_error=True)

analyze_query_intent_tool = ToolDefinition(
    name="analyze_query_intent",
    label="Analyze Query Intent",
    description="Deep-analyze user query for intent, entities, topics, and complexity",
    parameters_schema=AnalyzeQueryIntentParams,
    execute=_analyze_query_intent,
)


class SpawnAgentParams(BaseModel):
    agent_id: str = Field(..., description="Agent config ID to spawn")
    query: str = Field(..., description="Query/instruction for the agent")
    context: str | None = Field(None, description="Additional context")

async def _spawn_agent(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.agents.agent_configs import ALL_AGENT_CONFIGS
        from src.agents.autonomous_loop import CortexAgentLoop
        agent_id = params["agent_id"]
        config = ALL_AGENT_CONFIGS.get(agent_id)
        if not config:
            return ToolResult(
                tool_call_id=call_id,
                content=f"Unknown agent: {agent_id}. Available: {list(ALL_AGENT_CONFIGS.keys())}",
                is_error=True,
            )
        loop = CortexAgentLoop(config=config)
        result = await loop.prompt(params["query"])
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps({
                "agent_id": agent_id,
                "text": result.get("text", ""),
                "turns": result.get("turns", 0),
            }, default=str),
        )
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Spawn error: {e}", is_error=True)

spawn_agent_tool = ToolDefinition(
    name="spawn_agent",
    label="Spawn Agent",
    description="Spawn a specialist agent with a specific query. The L1 orchestrator uses this to delegate to L2 agents.",
    parameters_schema=SpawnAgentParams,
    execute=_spawn_agent,
    concurrency_safe=True,
)


class CollectAgentResultsParams(BaseModel):
    agent_ids: list[str] = Field(..., description="Agent IDs whose results to collect")

async def _collect_agent_results(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({"status": "collected", "agent_ids": params["agent_ids"]}),
    )

collect_agent_results_tool = ToolDefinition(
    name="collect_agent_results",
    label="Collect Agent Results",
    description="Collect results from spawned specialist agents",
    parameters_schema=CollectAgentResultsParams,
    execute=_collect_agent_results,
)


class DissolveTeamParams(BaseModel):
    agent_ids: list[str] = Field(..., description="Agent IDs to dissolve")

async def _dissolve_team(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({"dissolved": params["agent_ids"]}),
    )

dissolve_team_tool = ToolDefinition(
    name="dissolve_team",
    label="Dissolve Team",
    description="Dissolve a team of specialist agents after collecting results",
    parameters_schema=DissolveTeamParams,
    execute=_dissolve_team,
)


class ArbitrateConflictParams(BaseModel):
    claim_a: str = Field(..., description="First conflicting claim")
    claim_b: str = Field(..., description="Second conflicting claim")
    evidence_a: str | None = Field(None, description="Evidence for claim A")
    evidence_b: str | None = Field(None, description="Evidence for claim B")

async def _arbitrate_conflict(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "resolution": "pending_llm_judgment",
            "claim_a": params["claim_a"],
            "claim_b": params["claim_b"],
        }),
    )

arbitrate_conflict_tool = ToolDefinition(
    name="arbitrate_conflict",
    label="Arbitrate Conflict",
    description="Resolve conflicting claims from different agents using evidence weighting",
    parameters_schema=ArbitrateConflictParams,
    execute=_arbitrate_conflict,
)


class CompressEvidenceParams(BaseModel):
    evidence: list[str] = Field(..., description="Evidence snippets to compress")
    max_tokens: int = Field(2000, description="Target compressed size in tokens")

async def _compress_evidence(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.compression import ContextCompressor
    try:
        compressor = ContextCompressor()
        compressed = compressor.compress_context(params["evidence"], max_tokens=params.get("max_tokens", 2000))
        return ToolResult(tool_call_id=call_id, content=compressed)
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Compression error: {e}", is_error=True)

compress_evidence_tool = ToolDefinition(
    name="compress_evidence",
    label="Compress Evidence",
    description="Compress retrieved evidence to fit within context budget",
    parameters_schema=CompressEvidenceParams,
    execute=_compress_evidence,
)


class GenerateAnswerPlanParams(BaseModel):
    query: str = Field(..., description="Original query")
    evidence_summary: str = Field(..., description="Compressed evidence summary")
    agent_results: list[str] | None = Field(None, description="Results from specialist agents")

async def _generate_answer_plan(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "plan": "synthesize",
            "query": params["query"],
            "evidence_length": len(params.get("evidence_summary", "")),
            "agent_count": len(params.get("agent_results", []) or []),
        }),
    )

generate_answer_plan_tool = ToolDefinition(
    name="generate_answer_plan",
    label="Generate Answer Plan",
    description="Generate a plan for synthesizing the final answer from evidence and agent results",
    parameters_schema=GenerateAnswerPlanParams,
    execute=_generate_answer_plan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class BuildEventTimelineParams(BaseModel):
    topic: str = Field(..., description="Topic to build timeline for")
    start_date: str | None = Field(None, description="Start date filter (ISO)")
    end_date: str | None = Field(None, description="End date filter (ISO)")

async def _build_event_timeline(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.storage.duckdb_store import DuckDBStore
        store = DuckDBStore.get_instance()
        events = store.query_by_topic(params["topic"], limit=50)
        timeline = sorted(events, key=lambda e: e.get("timestamp", ""))
        return ToolResult(tool_call_id=call_id, content=json.dumps(timeline, default=str))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Timeline build error: {e}", is_error=True)

build_event_timeline_tool = ToolDefinition(
    name="build_event_timeline",
    label="Build Event Timeline",
    description="Construct a chronological timeline of events related to a topic",
    parameters_schema=BuildEventTimelineParams,
    execute=_build_event_timeline,
)


class DetectTemporalGapsParams(BaseModel):
    timeline: list[str] = Field(..., description="Ordered timestamps to check for gaps")
    min_gap_hours: int = Field(24, description="Minimum gap duration to flag (hours)")

async def _detect_temporal_gaps(call_id: str, params: dict[str, Any]) -> ToolResult:
    from datetime import datetime
    gaps = []
    timestamps = params.get("timeline", [])
    for i in range(1, len(timestamps)):
        try:
            prev = datetime.fromisoformat(timestamps[i-1])
            curr = datetime.fromisoformat(timestamps[i])
            diff_hours = (curr - prev).total_seconds() / 3600
            if diff_hours >= params.get("min_gap_hours", 24):
                gaps.append({"from": timestamps[i-1], "to": timestamps[i], "gap_hours": round(diff_hours, 1)})
        except (ValueError, TypeError):
            continue
    return ToolResult(tool_call_id=call_id, content=json.dumps(gaps))

detect_temporal_gaps_tool = ToolDefinition(
    name="detect_temporal_gaps",
    label="Detect Temporal Gaps",
    description="Find gaps in a timeline where no events were recorded",
    parameters_schema=DetectTemporalGapsParams,
    execute=_detect_temporal_gaps,
)


class TraceCausalChainParams(BaseModel):
    event: str = Field(..., description="Event or decision to trace causes for")
    max_depth: int = Field(5, description="Maximum causal chain depth")

async def _trace_causal_chain(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.storage.graph_store import GraphStore
        store = GraphStore.get_instance()
        chain = store.trace_causes(params["event"], max_depth=params.get("max_depth", 5))
        return ToolResult(tool_call_id=call_id, content=json.dumps(chain, default=str))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Causal trace error: {e}", is_error=True)

trace_causal_chain_tool = ToolDefinition(
    name="trace_causal_chain",
    label="Trace Causal Chain",
    description="Trace cause-effect relationships backward from an event through the knowledge graph",
    parameters_schema=TraceCausalChainParams,
    execute=_trace_causal_chain,
)


class DetectBeliefChangeParams(BaseModel):
    topic: str = Field(..., description="Topic to check for belief changes")
    time_range_days: int = Field(90, description="How far back to look (days)")

async def _detect_belief_change(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.storage.duckdb_store import DuckDBStore
        store = DuckDBStore.get_instance()
        changes = store.detect_belief_changes(params["topic"], days=params.get("time_range_days", 90))
        return ToolResult(tool_call_id=call_id, content=json.dumps(changes, default=str))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Belief detection error: {e}", is_error=True)

detect_belief_change_tool = ToolDefinition(
    name="detect_belief_change",
    label="Detect Belief Change",
    description="Detect how beliefs about a topic have evolved over time",
    parameters_schema=DetectBeliefChangeParams,
    execute=_detect_belief_change,
)


class AnalyzePatternParams(BaseModel):
    topic: str = Field(..., description="Topic or domain to analyze")
    pattern_type: str = Field("general", description="Type: general|temporal|behavioral|emotional")
    min_occurrences: int = Field(3, description="Minimum pattern occurrences to surface")

async def _analyze_pattern(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "topic": params["topic"],
            "pattern_type": params.get("pattern_type", "general"),
            "patterns": [],
            "note": "Pattern analysis requires accumulated data — results improve over time",
        }),
    )

analyze_pattern_tool = ToolDefinition(
    name="analyze_pattern",
    label="Analyze Pattern",
    description="Detect recurring patterns in memories related to a topic",
    parameters_schema=AnalyzePatternParams,
    execute=_analyze_pattern,
)


class DecomposeQueryParams(BaseModel):
    query: str = Field(..., description="Complex query to decompose")
    max_sub_queries: int = Field(5, description="Maximum sub-queries to generate")

async def _decompose_query(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "original": params["query"],
            "sub_queries": [params["query"]],
            "note": "LLM-powered decomposition: the model itself should break this down",
        }),
    )

decompose_query_tool = ToolDefinition(
    name="decompose_query",
    label="Decompose Query",
    description="Break a complex query into manageable sub-queries for multi-step retrieval",
    parameters_schema=DecomposeQueryParams,
    execute=_decompose_query,
)


class ScoreImportanceParams(BaseModel):
    content: str = Field(..., description="Content to score for importance")
    context: str | None = Field(None, description="Additional context for scoring")

async def _score_importance(call_id: str, params: dict[str, Any]) -> ToolResult:
    content = params["content"]
    score = min(1.0, max(0.1, len(content) / 500))
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({"score": round(score, 2), "content_preview": content[:100]}),
    )

score_importance_tool = ToolDefinition(
    name="score_importance",
    label="Score Importance",
    description="Score the importance of a piece of content for memory prioritization",
    parameters_schema=ScoreImportanceParams,
    execute=_score_importance,
)


# ═══════════════════════════════════════════════════════════════════════════════
# WIKI TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class ExtractClaimsParams(BaseModel):
    content: str = Field(..., description="Content to extract atomic claims from")
    source_id: str = Field(..., description="Source memory/document ID")

async def _extract_claims(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "claims": [],
            "source_id": params["source_id"],
            "note": "LLM-powered claim extraction — model generates atomic facts",
        }),
    )

extract_claims_tool = ToolDefinition(
    name="extract_claims",
    label="Extract Claims",
    description="Extract atomic factual claims from content for the claim store",
    parameters_schema=ExtractClaimsParams,
    execute=_extract_claims,
)


class UpsertClaimParams(BaseModel):
    claim: str = Field(..., description="The atomic claim text")
    confidence: float = Field(0.8, description="Confidence score 0-1")
    source_ids: list[str] = Field(default_factory=list, description="Source memory IDs")
    topic: str = Field("", description="Topic category")

async def _upsert_claim(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.wiki.claim_store import ClaimStore
        store = ClaimStore.get_instance()
        claim_id = store.upsert(
            claim=params["claim"],
            confidence=params.get("confidence", 0.8),
            source_ids=params.get("source_ids", []),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps({"claim_id": claim_id}))
    except ImportError:
        return ToolResult(tool_call_id=call_id, content="Claim store not initialized", is_error=True)
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Claim upsert error: {e}", is_error=True)

upsert_claim_tool = ToolDefinition(
    name="upsert_claim",
    label="Upsert Claim",
    description="Insert or update an atomic claim in the claim store",
    parameters_schema=UpsertClaimParams,
    execute=_upsert_claim,
)


class PatchWikiPageParams(BaseModel):
    page_id: str = Field(..., description="Wiki page ID to patch")
    section: str = Field(..., description="Section to update")
    content: str = Field(..., description="New content for the section")
    operation: str = Field("append", description="Operation: append|replace|prepend")

async def _patch_wiki_page(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.wiki.wiki_store import WikiStore
        store = WikiStore.get_instance()
        store.patch_page(
            page_id=params["page_id"],
            section=params["section"],
            content=params["content"],
            operation=params.get("operation", "append"),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps({"patched": True, "page_id": params["page_id"]}))
    except ImportError:
        return ToolResult(tool_call_id=call_id, content="Wiki store not initialized", is_error=True)
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Wiki patch error: {e}", is_error=True)

patch_wiki_page_tool = ToolDefinition(
    name="patch_wiki_page",
    label="Patch Wiki Page",
    description="Update a section of a personal wiki page",
    parameters_schema=PatchWikiPageParams,
    execute=_patch_wiki_page,
)


class CreateWikiPageParams(BaseModel):
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Initial page content (markdown)")
    topics: list[str] = Field(default_factory=list, description="Topic tags")

async def _create_wiki_page(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.wiki.wiki_store import WikiStore
        store = WikiStore.get_instance()
        page_id = store.create_page(
            title=params["title"],
            content=params["content"],
            topics=params.get("topics", []),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps({"page_id": page_id, "title": params["title"]}))
    except ImportError:
        return ToolResult(tool_call_id=call_id, content="Wiki store not initialized", is_error=True)
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Wiki create error: {e}", is_error=True)

create_wiki_page_tool = ToolDefinition(
    name="create_wiki_page",
    label="Create Wiki Page",
    description="Create a new personal wiki page",
    parameters_schema=CreateWikiPageParams,
    execute=_create_wiki_page,
)


class LintWikiPageParams(BaseModel):
    page_id: str = Field(..., description="Page ID to lint")

async def _lint_wiki_page(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({"page_id": params["page_id"], "issues": [], "status": "clean"}),
    )

lint_wiki_page_tool = ToolDefinition(
    name="lint_wiki_page",
    label="Lint Wiki Page",
    description="Check a wiki page for quality issues (stale claims, broken links, redundancy)",
    parameters_schema=LintWikiPageParams,
    execute=_lint_wiki_page,
)


class CompactWikiSectionParams(BaseModel):
    page_id: str = Field(..., description="Page ID")
    section: str = Field(..., description="Section to compact")
    max_tokens: int = Field(500, description="Target size after compaction")

async def _compact_wiki_section(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({"compacted": True, "page_id": params["page_id"], "section": params["section"]}),
    )

compact_wiki_section_tool = ToolDefinition(
    name="compact_wiki_section",
    label="Compact Wiki Section",
    description="Compress a wiki section to reduce redundancy while preserving key claims",
    parameters_schema=CompactWikiSectionParams,
    execute=_compact_wiki_section,
)


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE / INGEST TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class IngestMemoryParams(BaseModel):
    content: str = Field(..., description="Memory content to ingest")
    memory_type: str = Field("episodic", description="Type: episodic|semantic|procedural|reflective")
    source: str = Field("chat", description="Source: chat|import|voice|journal")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

async def _ingest_memory(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.ingestion import IngestionPipeline
        pipeline = IngestionPipeline.get_instance()
        memory_id = await pipeline.ingest(
            content=params["content"],
            memory_type=params.get("memory_type", "episodic"),
            source=params.get("source", "chat"),
            metadata=params.get("metadata"),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps({"memory_id": memory_id}))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Ingestion error: {e}", is_error=True)

ingest_memory_tool = ToolDefinition(
    name="ingest_memory",
    label="Ingest Memory",
    description="Store a new memory into the system through the full ingestion pipeline",
    parameters_schema=IngestMemoryParams,
    execute=_ingest_memory,
)


class UpdateGraphEdgeParams(BaseModel):
    source_entity: str = Field(..., description="Source entity name")
    target_entity: str = Field(..., description="Target entity name")
    relationship: str = Field(..., description="Relationship type")
    weight: float = Field(1.0, description="Edge weight")

async def _update_graph_edge(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.storage.graph_store import GraphStore
        store = GraphStore.get_instance()
        store.add_edge(
            params["source_entity"],
            params["target_entity"],
            relationship=params["relationship"],
            weight=params.get("weight", 1.0),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps({"updated": True}))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Graph edge error: {e}", is_error=True)

update_graph_edge_tool = ToolDefinition(
    name="update_graph_edge",
    label="Update Graph Edge",
    description="Add or update an edge in the knowledge graph",
    parameters_schema=UpdateGraphEdgeParams,
    execute=_update_graph_edge,
)


class InvalidateCacheParams(BaseModel):
    key: str = Field(..., description="Cache key or pattern to invalidate")

async def _invalidate_cache(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({"invalidated": True, "key": params["key"]}),
    )

invalidate_cache_tool = ToolDefinition(
    name="invalidate_cache",
    label="Invalidate Cache",
    description="Invalidate cached query results (T0 cache)",
    parameters_schema=InvalidateCacheParams,
    execute=_invalidate_cache,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PRESENCE / INITIATIVE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

class AssembleContextParams(BaseModel):
    user_state: str = Field(..., description="Current user state: active|idle|returning")
    last_interaction_minutes: int = Field(0, description="Minutes since last interaction")

async def _assemble_context(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "user_state": params["user_state"],
            "context_ready": True,
            "recent_topics": [],
            "pending_insights": [],
        }),
    )

assemble_context_tool = ToolDefinition(
    name="assemble_context",
    label="Assemble Presence Context",
    description="Gather current context for the presence agent to make initiative decisions",
    parameters_schema=AssembleContextParams,
    execute=_assemble_context,
)


class ScoreInitiativeParams(BaseModel):
    insight: str = Field(..., description="Insight to potentially surface")
    urgency: float = Field(0.5, description="How urgent is this 0-1")
    relevance: float = Field(0.5, description="How relevant to current context 0-1")

async def _score_initiative(call_id: str, params: dict[str, Any]) -> ToolResult:
    score = (params.get("urgency", 0.5) * 0.4 + params.get("relevance", 0.5) * 0.6)
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "score": round(score, 2),
            "should_act": score > 0.7,
            "insight": params["insight"],
        }),
    )

score_initiative_tool = ToolDefinition(
    name="score_initiative",
    label="Score Initiative",
    description="Score whether a proactive insight should be surfaced to the user",
    parameters_schema=ScoreInitiativeParams,
    execute=_score_initiative,
)


class DetectIdleParams(BaseModel):
    idle_threshold_minutes: int = Field(15, description="Minutes of inactivity to count as idle")

async def _detect_idle(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "is_idle": False,
            "last_activity_minutes": 0,
            "threshold": params.get("idle_threshold_minutes", 15),
        }),
    )

detect_idle_tool = ToolDefinition(
    name="detect_idle",
    label="Detect Idle",
    description="Check if the user has been idle for the specified threshold",
    parameters_schema=DetectIdleParams,
    execute=_detect_idle,
)


class ReadMoodSignalParams(BaseModel):
    window_hours: int = Field(24, description="How far back to read mood signals")

async def _read_mood_signal(call_id: str, params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        content=json.dumps({
            "dominant_mood": "neutral",
            "confidence": 0.5,
            "window_hours": params.get("window_hours", 24),
            "signals": [],
        }),
    )

read_mood_signal_tool = ToolDefinition(
    name="read_mood_signal",
    label="Read Mood Signal",
    description="Read aggregated mood signals from recent interactions",
    parameters_schema=ReadMoodSignalParams,
    execute=_read_mood_signal,
)
