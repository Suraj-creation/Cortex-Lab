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
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agents.tool_types import PermissionModel, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


_PERSONAL_DATA_SOURCES = (
    "memories",
    "conversations",
    "wiki",
    "claims",
    "graph",
    "ambient",
    "session_forge",
    "chronicle",
)


def _get_metadata_store():
    from src.engine import rag_engine

    store = getattr(rag_engine, "metadata_store", None)
    if store is None:
        raise RuntimeError("Metadata store is not initialized")
    return store


def _get_knowledge_graph():
    from src.engine import rag_engine

    graph = getattr(rag_engine, "knowledge_graph", None)
    if graph is None:
        raise RuntimeError("Knowledge graph is not initialized")
    return graph


def _to_memory_dict(memory: Any) -> dict[str, Any]:
    if hasattr(memory, "to_dict"):
        raw = memory.to_dict()
        if isinstance(raw, dict):
            return raw
    if isinstance(memory, dict):
        return dict(memory)
    return {"content": str(memory)}


def _parse_iso_datetime(value: str) -> datetime:
    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _clip_text(value: Any, limit: int = 420) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _safe_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return list(parsed)
        except Exception:
            return []
    return []


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return dict(parsed)
        except Exception:
            return {}
    return {}


def _query_contains(query: str, *parts: Any) -> bool:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return True
    blob = " ".join(str(part or "") for part in parts).lower()
    return normalized in blob


def _normalize_sources(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(_PERSONAL_DATA_SOURCES)

    normalized: list[str] = []
    for item in requested:
        key = str(item or "").strip().lower()
        if key in _PERSONAL_DATA_SOURCES and key not in normalized:
            normalized.append(key)

    return normalized or list(_PERSONAL_DATA_SOURCES)


def _duckdb_table_exists(conn: Any, table_name: str) -> bool:
    try:
        rows = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1",
            [table_name],
        ).fetchall()
        return bool(rows)
    except Exception:
        pass

    try:
        conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchall()
        return True
    except Exception:
        return False


def _open_sqlite_ambient_connection() -> Any:
    try:
        import sqlite3
        from src.engine import rag_engine

        db_path = os.path.join(str(getattr(rag_engine, "data_dir", "data")), "cortex.sqlite3")
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


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
    from src.engine import rag_engine
    from src.retrieval.query_engine import QueryAnalyzer

    try:
        retriever = getattr(rag_engine, "hybrid_retriever", None)
        if retriever is None:
            return ToolResult(tool_call_id=call_id, content="Retriever not initialized", is_error=True)

        analyzer = QueryAnalyzer()
        query = analyzer.analyze(params["query"])

        if params.get("time_start"):
            query.time_start = _parse_iso_datetime(params["time_start"])
        if params.get("time_end"):
            query.time_end = _parse_iso_datetime(params["time_end"])

        raw_results = await retriever.retrieve(
            query=query,
            top_k=params.get("top_k", 10),
        )

        memory_type = str(params.get("memory_type") or "").strip().lower()

        formatted = [
            {
                "content": r.memory.content if hasattr(r, 'memory') else str(r),
                "score": r.score if hasattr(r, 'score') else 0.0,
                "channel": r.channel if hasattr(r, 'channel') else "unknown",
                "memory_type": (
                    r.memory.memory_type.value
                    if hasattr(r, 'memory') and hasattr(r.memory, 'memory_type')
                    else "unknown"
                ),
                "timestamp": (
                    r.memory.timestamp.isoformat()
                    if hasattr(r, 'memory') and hasattr(r.memory, 'timestamp')
                    else ""
                ),
            }
            for r in raw_results
            if not memory_type
            or (
                hasattr(r, 'memory')
                and hasattr(r.memory, 'memory_type')
                and str(r.memory.memory_type.value).lower() == memory_type
            )
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
        graph = _get_knowledge_graph()
        metadata = _get_metadata_store()

        requested_entity = str(params["entity"])
        entity_id = graph.find_entity_by_name(requested_entity) or requested_entity
        neighbors = graph.get_neighbors(entity_id, max_hops=params.get("hops", 2))

        edges = metadata.get_edges(entity_id=entity_id)
        relationship_type = params.get("relationship_type")
        if relationship_type:
            rel = str(relationship_type).strip().lower()
            edges = [e for e in edges if str(e.get("relation", "")).lower() == rel]

        payload = {
            "entity": requested_entity,
            "entity_id": entity_id,
            "neighbors": neighbors,
            "edges": edges,
        }
        return ToolResult(tool_call_id=call_id, content=json.dumps(payload, default=str))
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
        store = _get_metadata_store()
        start = _parse_iso_datetime(params["start_date"])
        end = _parse_iso_datetime(params["end_date"])
        rows = store.search_by_time(start=start, end=end, limit=params.get("top_k", 20))
        results = [_to_memory_dict(row) for row in rows]
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


class QueryPersonalDataParams(BaseModel):
    query: str = Field("", description="Search query across all persisted personal data")
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Optional source filters. Supported: "
            "memories|conversations|wiki|claims|graph|ambient|session_forge|chronicle"
        ),
    )
    session_id: str = Field("", description="Optional session_id focus for conversation/session-forge sources")
    limit_per_source: int = Field(8, ge=1, le=25, description="Maximum items returned per source")
    min_confidence: float = Field(0.35, ge=0.0, le=1.0, description="Confidence floor for claim retrieval")


async def _query_personal_data(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.engine import rag_engine

    query = str(params.get("query", "") or "").strip()
    session_id = str(params.get("session_id", "") or "").strip()
    limit = max(1, min(int(params.get("limit_per_source", 8) or 8), 25))
    min_confidence = float(params.get("min_confidence", 0.35) or 0.35)
    sources = _normalize_sources(params.get("sources"))

    payload: dict[str, Any] = {
        "query": query,
        "session_id": session_id,
        "sources": sources,
        "counts": {},
        "results": {source: [] for source in sources},
        "top_hits": [],
        "errors": {},
    }

    if "memories" in sources:
        try:
            if query:
                memory_rows = list(rag_engine.search_memories(query, top_k=limit) or [])
            else:
                memory_rows = list(rag_engine.get_memories(limit=limit) or [])

            memory_items = []
            for row in memory_rows[:limit]:
                data = _to_memory_dict(row)
                memory_items.append(
                    {
                        "id": data.get("id", ""),
                        "content": _clip_text(data.get("content", ""), 480),
                        "score": float(data.get("score", 0.0) or 0.0),
                        "timestamp": data.get("timestamp", ""),
                        "memory_type": data.get("memory_type", ""),
                        "source": data.get("source", ""),
                        "topics": list(data.get("topics", []) or [])[:6],
                        "entities": list(data.get("entities", []) or [])[:6],
                        "session_id": data.get("session_id", ""),
                    }
                )

            payload["results"]["memories"] = memory_items
        except Exception as e:
            payload["errors"]["memories"] = str(e)

    metadata_store = None
    if any(source in sources for source in ("conversations", "graph", "ambient")):
        try:
            metadata_store = _get_metadata_store()
        except Exception as e:
            payload["errors"]["metadata"] = str(e)

    if "conversations" in sources:
        try:
            turns: list[dict[str, Any]] = []
            if metadata_store is None:
                turns = []
            elif session_id:
                turns = list(metadata_store.get_conversation(session_id) or [])
            elif getattr(metadata_store, "conn", None) is not None and bool(getattr(metadata_store, "_use_duckdb", False)):
                conn = metadata_store.conn
                if query:
                    rows = conn.execute(
                        """
                        SELECT session_id, role, content, thinking, timestamp
                        FROM conversations
                        WHERE lower(content) LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        [f"%{query.lower()}%", limit],
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT session_id, role, content, thinking, timestamp
                        FROM conversations
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        [limit],
                    ).fetchall()

                turns = [
                    {
                        "session_id": str(row[0] or ""),
                        "role": str(row[1] or ""),
                        "content": str(row[2] or ""),
                        "thinking": str(row[3] or ""),
                        "timestamp": str(row[4] or ""),
                    }
                    for row in rows
                ]

            conversation_items: list[dict[str, Any]] = []
            for turn in turns:
                if not _query_contains(query, turn.get("content", ""), turn.get("thinking", "")):
                    continue
                conversation_items.append(
                    {
                        "session_id": str(turn.get("session_id", session_id)),
                        "role": str(turn.get("role", "")),
                        "content": _clip_text(turn.get("content", ""), 320),
                        "thinking": _clip_text(turn.get("thinking", ""), 220),
                        "timestamp": str(turn.get("timestamp", "")),
                    }
                )
                if len(conversation_items) >= limit:
                    break

            payload["results"]["conversations"] = conversation_items
        except Exception as e:
            payload["errors"]["conversations"] = str(e)

    if "wiki" in sources:
        try:
            from src.wiki.wiki_store import WikiStore

            wiki_store = WikiStore.get_instance()
            wiki_rows = (
                wiki_store.search(query, include_claims=True, limit=limit)
                if query
                else wiki_store.list_pages(limit=limit)
            )
            payload["results"]["wiki"] = [
                {
                    "id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "topics": list(row.get("topics", []) or [])[:6],
                    "claim_count": len(list(row.get("claim_ids", []) or [])),
                    "updated_at": row.get("updated_at", ""),
                    "content_preview": _clip_text(row.get("content", ""), 420),
                    "search_score": float(row.get("search_score", 0.0) or 0.0),
                }
                for row in wiki_rows[:limit]
            ]
        except Exception as e:
            payload["errors"]["wiki"] = str(e)

    if "claims" in sources:
        try:
            from src.wiki.claim_store import ClaimStore

            claim_store = ClaimStore.get_instance()
            claim_rows = claim_store.search(
                query,
                min_confidence=min_confidence,
                limit=limit,
            )
            payload["results"]["claims"] = [
                {
                    "id": row.get("id", ""),
                    "text": _clip_text(row.get("text", ""), 320),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "topic": row.get("topic", ""),
                    "source_ids": list(row.get("source_ids", []) or [])[:8],
                    "updated_at": row.get("updated_at", ""),
                    "is_active": bool(row.get("is_active", True)),
                }
                for row in claim_rows[:limit]
            ]
        except Exception as e:
            payload["errors"]["claims"] = str(e)

    if "graph" in sources:
        try:
            graph = _get_knowledge_graph()
            matched_entities: list[dict[str, Any]] = []
            matched_edges: list[dict[str, Any]] = []
            focus_entity_id = graph.find_entity_by_name(query) if query else None

            if metadata_store is not None:
                entities = list(metadata_store.get_entities(limit=max(limit * 4, limit)) or [])
                for entity in entities:
                    if query and not _query_contains(
                        query,
                        entity.get("canonical_name", ""),
                        " ".join(entity.get("aliases", []) or []),
                        entity.get("entity_type", ""),
                    ):
                        continue
                    matched_entities.append(
                        {
                            "id": entity.get("id", ""),
                            "canonical_name": entity.get("canonical_name", ""),
                            "entity_type": entity.get("entity_type", ""),
                            "aliases": list(entity.get("aliases", []) or [])[:6],
                            "memory_ref_count": len(list(entity.get("memory_ids", []) or [])),
                            "last_seen": entity.get("last_seen", ""),
                        }
                    )
                    if len(matched_entities) >= limit:
                        break

                if focus_entity_id and not any(str(item.get("id", "")) == str(focus_entity_id) for item in matched_entities):
                    focus_rows = [e for e in entities if str(e.get("id", "")) == str(focus_entity_id)]
                    if focus_rows:
                        entity = focus_rows[0]
                        matched_entities.insert(
                            0,
                            {
                                "id": entity.get("id", ""),
                                "canonical_name": entity.get("canonical_name", ""),
                                "entity_type": entity.get("entity_type", ""),
                                "aliases": list(entity.get("aliases", []) or [])[:6],
                                "memory_ref_count": len(list(entity.get("memory_ids", []) or [])),
                                "last_seen": entity.get("last_seen", ""),
                            },
                        )
                        matched_entities = matched_entities[:limit]

                edge_seen: set[tuple[str, str, str]] = set()
                edge_entity_ids = [focus_entity_id] if focus_entity_id else []
                edge_entity_ids.extend(item.get("id", "") for item in matched_entities[:3])
                for entity_id in edge_entity_ids:
                    if not entity_id:
                        continue
                    for edge in list(metadata_store.get_edges(entity_id=str(entity_id)) or []):
                        signature = (
                            str(edge.get("source_id", "")),
                            str(edge.get("target_id", "")),
                            str(edge.get("relation", "")),
                        )
                        if signature in edge_seen:
                            continue
                        edge_seen.add(signature)
                        matched_edges.append(
                            {
                                "source_id": signature[0],
                                "target_id": signature[1],
                                "relation": signature[2],
                                "weight": float(edge.get("weight", 0.0) or 0.0),
                                "memory_ref_count": len(list(edge.get("memory_ids", []) or [])),
                            }
                        )
                        if len(matched_edges) >= limit:
                            break
                    if len(matched_edges) >= limit:
                        break

            payload["results"]["graph"] = {
                "focus_entity_id": focus_entity_id or "",
                "matched_entities": matched_entities[:limit],
                "edges": matched_edges[:limit],
            }
        except Exception as e:
            payload["errors"]["graph"] = str(e)

    if "ambient" in sources:
        try:
            ambient_items: list[dict[str, Any]] = []
            db_used = "none"

            if metadata_store is not None and getattr(metadata_store, "conn", None) is not None and bool(getattr(metadata_store, "_use_duckdb", False)):
                conn = metadata_store.conn
                if _duckdb_table_exists(conn, "ambient_conversations"):
                    db_used = "duckdb"
                    if query:
                        conv_rows = conn.execute(
                            """
                            SELECT id, started_at, ended_at, duration_seconds,
                                   participants, turn_count, topic_labels,
                                   importance_score, raw_transcript
                            FROM ambient_conversations
                            WHERE lower(raw_transcript) LIKE ?
                            ORDER BY ended_at DESC
                            LIMIT ?
                            """,
                            [f"%{query.lower()}%", limit],
                        ).fetchall()
                    else:
                        conv_rows = conn.execute(
                            """
                            SELECT id, started_at, ended_at, duration_seconds,
                                   participants, turn_count, topic_labels,
                                   importance_score, raw_transcript
                            FROM ambient_conversations
                            ORDER BY ended_at DESC
                            LIMIT ?
                            """,
                            [limit],
                        ).fetchall()

                    for row in conv_rows:
                        participants = _safe_json_list(row[4]) if isinstance(row[4], str) else list(row[4] or [])
                        topics = _safe_json_list(row[6]) if isinstance(row[6], str) else list(row[6] or [])
                        ambient_items.append(
                            {
                                "record_type": "conversation",
                                "id": str(row[0] or ""),
                                "started_at": str(row[1] or ""),
                                "ended_at": str(row[2] or ""),
                                "duration_seconds": float(row[3] or 0.0),
                                "participants": participants[:6],
                                "turn_count": int(row[5] or 0),
                                "topic_labels": topics[:6],
                                "importance_score": float(row[7] or 0.0),
                                "transcript_preview": _clip_text(row[8] or "", 360),
                            }
                        )

                if len(ambient_items) < limit and _duckdb_table_exists(conn, "ambient_conversation_turns"):
                    if query:
                        turn_rows = conn.execute(
                            """
                            SELECT conversation_id, turn_index, speaker, speaker_name, text, timestamp_s
                            FROM ambient_conversation_turns
                            WHERE lower(text) LIKE ?
                            ORDER BY conversation_id DESC, turn_index DESC
                            LIMIT ?
                            """,
                            [f"%{query.lower()}%", max(limit - len(ambient_items), 1)],
                        ).fetchall()
                    else:
                        turn_rows = conn.execute(
                            """
                            SELECT conversation_id, turn_index, speaker, speaker_name, text, timestamp_s
                            FROM ambient_conversation_turns
                            ORDER BY conversation_id DESC, turn_index DESC
                            LIMIT ?
                            """,
                            [max(limit - len(ambient_items), 1)],
                        ).fetchall()

                    for row in turn_rows:
                        ambient_items.append(
                            {
                                "record_type": "turn",
                                "conversation_id": str(row[0] or ""),
                                "turn_index": int(row[1] or 0),
                                "speaker": str(row[2] or ""),
                                "speaker_name": str(row[3] or ""),
                                "text": _clip_text(row[4] or "", 280),
                                "timestamp_s": float(row[5] or 0.0),
                            }
                        )
                        if len(ambient_items) >= limit:
                            break

            if not ambient_items:
                sqlite_conn = _open_sqlite_ambient_connection()
                if sqlite_conn is not None:
                    try:
                        exists = sqlite_conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='ambient_conversations'"
                        ).fetchone()
                        if exists:
                            db_used = "sqlite"
                            if query:
                                conv_rows = sqlite_conn.execute(
                                    """
                                    SELECT id, started_at, ended_at, duration_seconds,
                                           participants, turn_count, topic_labels,
                                           importance_score, raw_transcript
                                    FROM ambient_conversations
                                    WHERE lower(raw_transcript) LIKE ?
                                    ORDER BY ended_at DESC
                                    LIMIT ?
                                    """,
                                    (f"%{query.lower()}%", limit),
                                ).fetchall()
                            else:
                                conv_rows = sqlite_conn.execute(
                                    """
                                    SELECT id, started_at, ended_at, duration_seconds,
                                           participants, turn_count, topic_labels,
                                           importance_score, raw_transcript
                                    FROM ambient_conversations
                                    ORDER BY ended_at DESC
                                    LIMIT ?
                                    """,
                                    (limit,),
                                ).fetchall()

                            for row in conv_rows:
                                participants = _safe_json_list(row[4])
                                topics = _safe_json_list(row[6])
                                ambient_items.append(
                                    {
                                        "record_type": "conversation",
                                        "id": str(row[0] or ""),
                                        "started_at": str(row[1] or ""),
                                        "ended_at": str(row[2] or ""),
                                        "duration_seconds": float(row[3] or 0.0),
                                        "participants": participants[:6],
                                        "turn_count": int(row[5] or 0),
                                        "topic_labels": topics[:6],
                                        "importance_score": float(row[7] or 0.0),
                                        "transcript_preview": _clip_text(row[8] or "", 360),
                                    }
                                )

                            if len(ambient_items) < limit:
                                turn_exists = sqlite_conn.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ambient_conversation_turns'"
                                ).fetchone()
                                if turn_exists:
                                    if query:
                                        turn_rows = sqlite_conn.execute(
                                            """
                                            SELECT conversation_id, turn_index, speaker, speaker_name, text, timestamp_s
                                            FROM ambient_conversation_turns
                                            WHERE lower(text) LIKE ?
                                            ORDER BY conversation_id DESC, turn_index DESC
                                            LIMIT ?
                                            """,
                                            (f"%{query.lower()}%", max(limit - len(ambient_items), 1)),
                                        ).fetchall()
                                    else:
                                        turn_rows = sqlite_conn.execute(
                                            """
                                            SELECT conversation_id, turn_index, speaker, speaker_name, text, timestamp_s
                                            FROM ambient_conversation_turns
                                            ORDER BY conversation_id DESC, turn_index DESC
                                            LIMIT ?
                                            """,
                                            (max(limit - len(ambient_items), 1),),
                                        ).fetchall()

                                    for row in turn_rows:
                                        ambient_items.append(
                                            {
                                                "record_type": "turn",
                                                "conversation_id": str(row[0] or ""),
                                                "turn_index": int(row[1] or 0),
                                                "speaker": str(row[2] or ""),
                                                "speaker_name": str(row[3] or ""),
                                                "text": _clip_text(row[4] or "", 280),
                                                "timestamp_s": float(row[5] or 0.0),
                                            }
                                        )
                                        if len(ambient_items) >= limit:
                                            break
                    finally:
                        sqlite_conn.close()

            payload["results"]["ambient"] = ambient_items[:limit]
            payload["results"]["ambient_db"] = db_used
        except Exception as e:
            payload["errors"]["ambient"] = str(e)

    if "session_forge" in sources:
        try:
            from src.applications import session_memory_forge_service

            artifact_types = [
                "thought_objects",
                "decision_records",
                "open_loops",
                "gap_signals",
                "belief_evolution",
                "structured_summaries",
            ]

            forge_items: list[dict[str, Any]] = []
            for artifact_type in artifact_types:
                rows = session_memory_forge_service.list_artifacts(
                    artifact_type,
                    limit=max(limit * 3, limit),
                    session_id=session_id,
                )
                for row in rows:
                    if query and not _query_contains(query, json.dumps(row, default=str)):
                        continue
                    forge_items.append(
                        {
                            "artifact_type": artifact_type,
                            "source_session": row.get("source_session", ""),
                            "timestamp": row.get("timestamp", row.get("created_at", "")),
                            "preview": _clip_text(
                                row.get("core_claim", "")
                                or row.get("decision_text", "")
                                or row.get("question", "")
                                or row.get("narrative_summary", "")
                                or json.dumps(row, default=str),
                                320,
                            ),
                            "payload": row,
                        }
                    )
                    if len(forge_items) >= limit:
                        break
                if len(forge_items) >= limit:
                    break

            payload["results"]["session_forge"] = forge_items[:limit]
        except Exception as e:
            payload["errors"]["session_forge"] = str(e)

    if "chronicle" in sources:
        try:
            from src.applications import life_chronicle_service

            moments = life_chronicle_service.list_moments(limit=max(limit * 4, limit), tag="")
            chronicle_items: list[dict[str, Any]] = []
            for moment in moments:
                if query and not _query_contains(
                    query,
                    moment.get("title", ""),
                    moment.get("narrative", ""),
                    moment.get("retrieval_hint", ""),
                    " ".join(moment.get("tags", []) or []),
                    " ".join(moment.get("people_present", []) or []),
                ):
                    continue

                chronicle_items.append(
                    {
                        "memory_id": moment.get("memory_id", ""),
                        "title": moment.get("title", ""),
                        "timestamp": moment.get("timestamp", ""),
                        "life_domain": moment.get("life_domain", ""),
                        "importance_score": float(moment.get("importance_score", 0.0) or 0.0),
                        "tags": list(moment.get("tags", []) or [])[:8],
                        "people_present": list(moment.get("people_present", []) or [])[:8],
                        "narrative_preview": _clip_text(moment.get("narrative", ""), 360),
                        "retrieval_hint": _clip_text(moment.get("retrieval_hint", ""), 140),
                    }
                )
                if len(chronicle_items) >= limit:
                    break

            payload["results"]["chronicle"] = chronicle_items
        except Exception as e:
            payload["errors"]["chronicle"] = str(e)

    for source in sources:
        value = payload["results"].get(source)
        if isinstance(value, list):
            payload["counts"][source] = len(value)
        elif isinstance(value, dict):
            payload["counts"][source] = (
                len(list(value.get("matched_entities", []) or []))
                + len(list(value.get("edges", []) or []))
            )
        else:
            payload["counts"][source] = 0

    for source in sources:
        source_result = payload["results"].get(source)
        if isinstance(source_result, list):
            for item in source_result[:2]:
                preview = (
                    item.get("content")
                    or item.get("text")
                    or item.get("title")
                    or item.get("preview")
                    or item.get("transcript_preview")
                    or item.get("narrative_preview")
                    or ""
                )
                if preview:
                    payload["top_hits"].append(
                        {
                            "source": source,
                            "preview": _clip_text(preview, 180),
                        }
                    )
        elif isinstance(source_result, dict):
            for entity in list(source_result.get("matched_entities", []) or [])[:2]:
                name = str(entity.get("canonical_name", "") or entity.get("id", "")).strip()
                if name:
                    payload["top_hits"].append(
                        {
                            "source": source,
                            "preview": _clip_text(name, 180),
                        }
                    )

    if not payload["top_hits"] and not payload["errors"]:
        payload["top_hits"] = [{"source": "none", "preview": "No matching records found"}]

    return ToolResult(tool_call_id=call_id, content=json.dumps(payload, default=str))


query_personal_data_tool = ToolDefinition(
    name="query_personal_data",
    label="Query Personal Data",
    description=(
        "Search all persisted personal data planes (memories, conversations, wiki, claims, "
        "graph, ambient archives, session forge artifacts, and chronicle moments)."
    ),
    parameters_schema=QueryPersonalDataParams,
    execute=_query_personal_data,
    prompt_snippet="Use this as the first retrieval step when the user asks about previously stored information.",
)


class GetPersonalDataStatsParams(BaseModel):
    include_samples: bool = Field(False, description="Include small sample identifiers per data source")


async def _get_personal_data_stats(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.engine import rag_engine

    include_samples = bool(params.get("include_samples", False))
    payload: dict[str, Any] = {
        "runtime_initialized": bool(getattr(rag_engine, "initialized", False)),
        "sources": {},
        "errors": {},
    }

    metadata_store = None
    try:
        metadata_store = _get_metadata_store()
        payload["sources"]["memories"] = metadata_store.get_stats()

        conn = getattr(metadata_store, "conn", None)
        if conn is not None and bool(getattr(metadata_store, "_use_duckdb", False)):
            ambient_stats = {
                "backend": "duckdb",
                "conversations": 0,
                "turns": 0,
            }
            if _duckdb_table_exists(conn, "ambient_conversations"):
                ambient_stats["conversations"] = int(
                    conn.execute("SELECT COUNT(*) FROM ambient_conversations").fetchone()[0]
                )
            if _duckdb_table_exists(conn, "ambient_conversation_turns"):
                ambient_stats["turns"] = int(
                    conn.execute("SELECT COUNT(*) FROM ambient_conversation_turns").fetchone()[0]
                )
            payload["sources"]["ambient"] = ambient_stats
    except Exception as e:
        payload["errors"]["memories"] = str(e)

    if "ambient" not in payload["sources"]:
        sqlite_conn = _open_sqlite_ambient_connection()
        if sqlite_conn is not None:
            try:
                ambient_stats = {
                    "backend": "sqlite",
                    "conversations": 0,
                    "turns": 0,
                }
                exists = sqlite_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ambient_conversations'"
                ).fetchone()
                if exists:
                    ambient_stats["conversations"] = int(
                        sqlite_conn.execute("SELECT COUNT(*) FROM ambient_conversations").fetchone()[0]
                    )
                turn_exists = sqlite_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ambient_conversation_turns'"
                ).fetchone()
                if turn_exists:
                    ambient_stats["turns"] = int(
                        sqlite_conn.execute("SELECT COUNT(*) FROM ambient_conversation_turns").fetchone()[0]
                    )
                payload["sources"]["ambient"] = ambient_stats
            except Exception as e:
                payload["errors"]["ambient"] = str(e)
            finally:
                sqlite_conn.close()

    try:
        from src.wiki.wiki_store import WikiStore

        wiki_store = WikiStore.get_instance()
        wiki_stats = wiki_store.stats()
        if include_samples:
            wiki_stats["sample_pages"] = [
                page.get("title", "")
                for page in wiki_store.list_pages(limit=3)
            ]
        payload["sources"]["wiki"] = wiki_stats
    except Exception as e:
        payload["errors"]["wiki"] = str(e)

    try:
        from src.wiki.claim_store import ClaimStore

        claim_store = ClaimStore.get_instance()
        claim_stats = claim_store.stats()
        if include_samples:
            claim_stats["sample_claims"] = [
                row.get("text", "")
                for row in claim_store.search("", min_confidence=0.0, limit=3)
            ]
        payload["sources"]["claims"] = claim_stats
    except Exception as e:
        payload["errors"]["claims"] = str(e)

    try:
        graph = _get_knowledge_graph()
        payload["sources"]["graph"] = graph.get_stats()
    except Exception as e:
        payload["errors"]["graph"] = str(e)

    try:
        from src.applications import session_memory_forge_service

        forge_status = session_memory_forge_service.status()
        if not include_samples:
            forge_status.pop("updated_at", None)
        payload["sources"]["session_forge"] = forge_status
    except Exception as e:
        payload["errors"]["session_forge"] = str(e)

    try:
        from src.applications import life_chronicle_service

        chronicle_status = life_chronicle_service.status()
        if not include_samples:
            chronicle_status.pop("updated_at", None)
        payload["sources"]["chronicle"] = chronicle_status
    except Exception as e:
        payload["errors"]["chronicle"] = str(e)

    payload["error_count"] = len(payload["errors"])
    return ToolResult(tool_call_id=call_id, content=json.dumps(payload, default=str))


get_personal_data_stats_tool = ToolDefinition(
    name="get_personal_data_stats",
    label="Get Personal Data Stats",
    description="Return availability and counts across all persisted personal data sources.",
    parameters_schema=GetPersonalDataStatsParams,
    execute=_get_personal_data_stats,
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
        analysis = analyzer.analyze(params["query"])
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
        analysis = analyzer.analyze(params["query"])
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
    metadata: dict[str, Any] | None = Field(None, description="Optional runtime metadata (trace, parent task, session)")

async def _spawn_agent(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.agents.agent_configs import ALL_AGENT_CONFIGS
        from src.agents.autonomous_loop import CortexAgentLoop
        from src.agents.llm_adapter import make_cortex_loop_llm_fn
        from src.engine import rag_engine
        agent_id = params["agent_id"]
        config = ALL_AGENT_CONFIGS.get(agent_id)
        if not config:
            return ToolResult(
                tool_call_id=call_id,
                content=f"Unknown agent: {agent_id}. Available: {list(ALL_AGENT_CONFIGS.keys())}",
                is_error=True,
            )
        llm_fn = make_cortex_loop_llm_fn(
            rag_engine.llm,
            preferred_provider=config.llm_provider,
        )
        metadata = dict(params.get("metadata") or {})
        parent_task_id = str(metadata.get("parent_task_id", "")).strip() or None
        loop = CortexAgentLoop(
            config=config,
            llm_fn=llm_fn,
            runtime_task_manager=getattr(rag_engine, "runtime_task_manager", None),
            parent_task_id=parent_task_id,
        )

        prompt = str(params["query"])
        context = str(params.get("context") or "").strip()
        if context:
            prompt = f"{prompt}\n\nAdditional context:\n{context[:4000]}"

        result = await loop.prompt(prompt)
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps({
                "agent_id": agent_id,
                "text": result.get("text", ""),
                "turns": result.get("turns", 0),
                "tool_results": result.get("tool_results", []),
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
    evidence_summary: str = Field("", description="Compressed evidence summary")
    agent_results: list[Any] | None = Field(None, description="Results from specialist agents")
    evidence_ids: list[str] | None = Field(None, description="Selected evidence IDs")
    confidence: float = Field(0.0, description="Current confidence score")
    arbitration_notes: list[str] | None = Field(None, description="Conflict/arbitration notes")
    generation_policy: str = Field("default", description="Generation policy label")
    source_wiki_pages: list[str] | None = Field(None, description="Source wiki page IDs")
    source_claim_ids: list[str] | None = Field(None, description="Source claim IDs")
    source_event_ids: list[str] | None = Field(None, description="Source event IDs")
    quality_loops: dict[str, Any] | None = Field(None, description="Quality loop status map")

async def _generate_answer_plan(call_id: str, params: dict[str, Any]) -> ToolResult:
    from src.models import AnswerPlan

    evidence_ids = [str(item) for item in list(params.get("evidence_ids", []) or []) if item]
    wiki_pages = [str(item) for item in list(params.get("source_wiki_pages", []) or []) if item]
    claim_ids = [str(item) for item in list(params.get("source_claim_ids", []) or []) if item]
    event_ids = [str(item) for item in list(params.get("source_event_ids", []) or []) if item]

    arbitration_notes = [
        str(item)
        for item in list(params.get("arbitration_notes", []) or [])
        if str(item).strip()
    ]

    agent_results = list(params.get("agent_results", []) or [])
    conflict_signals = 0
    for item in agent_results:
        text = str(item).lower()
        if any(token in text for token in ("conflict", "contradict", "inconsistent", "tradeoff")):
            conflict_signals += 1
    if conflict_signals and not arbitration_notes:
        arbitration_notes.append(f"Detected {conflict_signals} potential conflict signals in agent outputs.")

    evidence_summary = str(params.get("evidence_summary", "") or "")
    confidence = max(0.0, min(1.0, float(params.get("confidence", 0.0))))

    plan = AnswerPlan(
        selected_evidence_ids=evidence_ids,
        confidence=confidence,
        confidence_composition={
            "provided_confidence": confidence,
            "evidence_summary_density": min(len(evidence_summary) / 1200.0, 1.0),
            "agent_coverage": min(len(agent_results) / 5.0, 1.0),
        },
        arbitration_notes=arbitration_notes,
        generation_policy=str(params.get("generation_policy", "default") or "default"),
        citation_required=bool(evidence_ids or evidence_summary),
        source_wiki_pages=wiki_pages,
        source_claim_ids=claim_ids,
        source_event_ids=event_ids,
        quality_loops=dict(params.get("quality_loops", {}) or {}),
    )

    payload = plan.to_dict()
    payload["query"] = params.get("query", "")
    payload["agent_count"] = len(agent_results)
    payload["evidence_summary_length"] = len(evidence_summary)
    return ToolResult(tool_call_id=call_id, content=json.dumps(payload))

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
        store = _get_metadata_store()
        events = store.search_by_topic(params["topic"], limit=200)
        timeline = []
        for event in events:
            data = _to_memory_dict(event)
            timeline.append(
                {
                    "id": data.get("id", ""),
                    "timestamp": data.get("timestamp", ""),
                    "content": data.get("content", ""),
                    "topics": data.get("topics", []),
                    "entities": data.get("entities", []),
                }
            )
        timeline.sort(key=lambda e: str(e.get("timestamp", "")))
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
        graph = _get_knowledge_graph()
        requested = str(params["event"])
        entity_id = graph.find_entity_by_name(requested) or requested
        chain = graph.get_causal_chain(entity_id, direction="backward")
        if params.get("max_depth"):
            chain = chain[: max(int(params.get("max_depth", 5)), 1)]
        payload = {
            "event": requested,
            "entity_id": entity_id,
            "causal_chain": chain,
        }
        return ToolResult(tool_call_id=call_id, content=json.dumps(payload, default=str))
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
        store = _get_metadata_store()
        deltas = store.get_belief_deltas(topic=params["topic"], limit=200)
        days = int(params.get("time_range_days", 90) or 90)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(days, 1))

        changes = []
        for delta in deltas:
            detected_at = str(delta.get("detected_at", ""))
            try:
                dt = _parse_iso_datetime(detected_at)
            except Exception:
                continue
            if dt >= cutoff:
                changes.append(delta)

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
    try:
        from src.wiki.materializer import extract_claim_candidates

        claims = extract_claim_candidates(
            content=str(params.get("content", "")),
            max_claims=12,
        )
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps(
                {
                    "claims": claims,
                    "count": len(claims),
                    "source_id": params["source_id"],
                }
            ),
        )
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Claim extraction error: {e}", is_error=True)

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
            topic=params.get("topic", ""),
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
    stale_days: int = Field(90, description="Mark pages stale after N days")
    min_confidence: float = Field(0.45, description="Minimum claim confidence threshold")

async def _lint_wiki_page(call_id: str, params: dict[str, Any]) -> ToolResult:
    try:
        from src.wiki.lint import WikiLinter

        report = WikiLinter().lint_page(
            page_id=params["page_id"],
            stale_days=int(params.get("stale_days", 90)),
            min_confidence=float(params.get("min_confidence", 0.45)),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps(report))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Wiki lint error: {e}", is_error=True)

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
    try:
        from src.wiki.compactor import WikiCompactor

        result = WikiCompactor().compact_section(
            page_id=params["page_id"],
            section=params["section"],
            max_tokens=int(params.get("max_tokens", 500)),
        )
        return ToolResult(tool_call_id=call_id, content=json.dumps(result))
    except Exception as e:
        return ToolResult(tool_call_id=call_id, content=f"Wiki compaction error: {e}", is_error=True)

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
        from src.engine import rag_engine

        result = await rag_engine.ingest_memory(
            content=params["content"],
            source=params.get("source", "chat"),
            session_id=str((params.get("metadata") or {}).get("session_id", "")),
        )
        memory_id = str((result or {}).get("id", ""))
        return ToolResult(tool_call_id=call_id, content=json.dumps({"memory_id": memory_id, "memory": result}))
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
        from src.models import GraphEdge

        graph = _get_knowledge_graph()
        metadata = _get_metadata_store()

        edge = GraphEdge(
            source_id=params["source_entity"],
            target_id=params["target_entity"],
            relation=params["relationship"],
            weight=float(params.get("weight", 1.0)),
            memory_ids=[],
        )
        graph.add_edge(edge)
        try:
            metadata.store_edge(edge)
        except Exception:
            pass

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
