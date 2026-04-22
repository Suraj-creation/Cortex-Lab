"""Core Cortex tool catalog for Phase 0 runtime contract baseline."""

from __future__ import annotations

from typing import List

from .contracts import ToolContract, ToolParameterSpec, ToolRiskTier


def build_core_tool_catalog() -> List[ToolContract]:
    """Build contract representations for current Cortex engine operations.

    Phase 0 gate requirement:
    all current core operations are representable as tool contracts.
    """

    return [
        ToolContract(
            name="rag_chat",
            description="Run full agentic RAG chat pipeline and return answer with evidence.",
            parameters=[
                ToolParameterSpec("user_message", "string", "User chat message.", required=True),
                ToolParameterSpec("session_id", "string", "Session identifier."),
                ToolParameterSpec("conversation_history", "array", "Conversation turns for context."),
            ],
            risk_tier=ToolRiskTier.MEDIUM,
            side_effect_free=False,
            capabilities=["chat", "retrieval", "generation"],
            output_schema={"type": "object", "required": ["answer", "evidence", "confidence"]},
        ),
        ToolContract(
            name="rag_retrieve",
            description="Run retrieval-only RAG stage for streaming answer generation.",
            parameters=[
                ToolParameterSpec("user_message", "string", "User query.", required=True),
                ToolParameterSpec("session_id", "string", "Session identifier."),
                ToolParameterSpec("conversation_history", "array", "Conversation turns for context."),
            ],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["retrieval", "streaming"],
            output_schema={"type": "object", "required": ["evidence", "confidence"]},
        ),
        ToolContract(
            name="ingest_memory",
            description="Create and index a new memory entry in storage and retrieval layers.",
            parameters=[
                ToolParameterSpec("content", "string", "Memory content.", required=True),
                ToolParameterSpec("source", "string", "Memory source tag.", default="manual"),
                ToolParameterSpec("session_id", "string", "Session identifier."),
            ],
            risk_tier=ToolRiskTier.HIGH,
            side_effect_free=False,
            capabilities=["memory", "indexing", "write"],
            output_schema={"type": "object", "required": ["id", "content", "timestamp"]},
        ),
        ToolContract(
            name="search_memories",
            description="Semantic memory search over indexed vectors.",
            parameters=[
                ToolParameterSpec("query", "string", "Search query.", required=True),
                ToolParameterSpec("top_k", "integer", "Maximum number of results.", default=10),
            ],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["memory", "search"],
            output_schema={"type": "array"},
        ),
        ToolContract(
            name="get_memories",
            description="List stored memories with pagination.",
            parameters=[
                ToolParameterSpec("limit", "integer", "Max memories returned.", default=50),
                ToolParameterSpec("offset", "integer", "Pagination offset.", default=0),
            ],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["memory", "read"],
            output_schema={"type": "array"},
        ),
        ToolContract(
            name="delete_memory",
            description="Delete memory from metadata and vector indexes.",
            parameters=[
                ToolParameterSpec("memory_id", "string", "Memory identifier.", required=True),
            ],
            risk_tier=ToolRiskTier.HIGH,
            side_effect_free=False,
            capabilities=["memory", "delete", "write"],
            output_schema={"type": "boolean"},
        ),
        ToolContract(
            name="get_graph_data",
            description="Return knowledge graph nodes and edges for visualization.",
            parameters=[],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["graph", "read"],
            output_schema={"type": "object", "required": ["nodes", "edges"]},
        ),
        ToolContract(
            name="get_entities",
            description="Return entity index entries from metadata store.",
            parameters=[
                ToolParameterSpec("limit", "integer", "Maximum entities returned.", default=100),
            ],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["graph", "entity", "read"],
            output_schema={"type": "array"},
        ),
        ToolContract(
            name="get_belief_deltas",
            description="Return tracked belief evolution events.",
            parameters=[
                ToolParameterSpec("limit", "integer", "Maximum belief deltas returned.", default=50),
            ],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["beliefs", "read"],
            output_schema={"type": "array"},
        ),
        ToolContract(
            name="get_community_summaries",
            description="Return GraphRAG-style community summaries.",
            parameters=[],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["graph", "community", "read"],
            output_schema={"type": "array"},
        ),
        ToolContract(
            name="get_rag_stats",
            description="Return aggregate runtime stats for memory, retrieval, graph, cache, and llm.",
            parameters=[],
            risk_tier=ToolRiskTier.LOW,
            side_effect_free=True,
            capabilities=["telemetry", "stats", "read"],
            output_schema={"type": "object", "required": ["status"]},
        ),
    ]


def build_core_tool_catalog_dicts() -> List[dict]:
    """Serialize core catalog contracts for API and diagnostics endpoints."""

    return [contract.to_dict() for contract in build_core_tool_catalog()]
