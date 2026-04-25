from __future__ import annotations

from typing import Any, Dict, Optional

from src.ambient.companion_logic import build_retention_trace
from src.runtime.session_manager import RuntimeSessionManager, runtime_session_manager


def build_manual_memory_retention_trace(
    content: str,
    *,
    session_id: str = "",
    platform: str = "mobile",
    source: str = "manual",
    force_keep: bool = True,
) -> Dict[str, Any]:
    text = str(content or "").strip()
    trace = build_retention_trace(
        text,
        session_id=session_id,
        platform=platform or "mobile",
        source=source or "manual",
    )

    if not text:
        return trace

    tags = [str(tag).strip() for tag in list(trace.get("tags", []) or []) if str(tag).strip()]
    if "manual_submission" not in tags:
        tags.append("manual_submission")
    if "document" in str(source or "").lower() and "document_ingest" not in tags:
        tags.append("document_ingest")

    if force_keep:
        memory_decision = str(trace.get("memory_decision") or "structured").strip().lower()
        if memory_decision in {"discard", "discarded", "session_only"}:
            memory_decision = "structured"
        trace["decision"] = "keep"
        trace["memory_decision"] = memory_decision
        trace["reason"] = "explicit_manual_memory_submission"
        trace["archive_policy"] = "session"

    trace["tags"] = list(dict.fromkeys(tags))
    trace["session_id"] = session_id
    trace["platform"] = platform or "mobile"
    trace["source"] = source or "manual"
    return trace


def prepare_manual_memory_session(
    content: str,
    *,
    session_id: str = "",
    platform: str = "mobile",
    source: str = "manual",
    metadata: Optional[Dict[str, Any]] = None,
    force_keep: bool = True,
    session_manager: RuntimeSessionManager = runtime_session_manager,
) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise ValueError("Memory content is required")

    session = session_manager.get_session(session_id) if session_id else None
    session_created = False

    base_metadata = {
        "platform": platform or "mobile",
        "source": source or "manual",
        "intake_type": "manual_memory",
        **dict(metadata or {}),
    }

    if session is None:
        session = session_manager.open_session(
            mode="manual_memory",
            metadata=base_metadata,
        )
        session_created = True
        session_id = session.session_id
    else:
        session_manager.update_metadata(session_id, base_metadata)

    trace = build_manual_memory_retention_trace(
        text,
        session_id=session_id,
        platform=platform,
        source=source,
        force_keep=force_keep,
    )

    retention_key = str(trace.get("memory_decision") or "structured").strip().lower()
    if retention_key not in {"discarded", "session_only", "structured", "priority"}:
        retention_key = "structured"

    session_manager.merge_retention_summary(session_id, {retention_key: 1})
    session_manager.append_agent_tags(session_id, list(trace.get("tags", []) or []))
    session_manager.update_metadata(
        session_id,
        {
            **base_metadata,
            "last_manual_memory_text": text,
            "last_retention_trace": trace,
        },
    )

    snapshot = session_manager.get_session(session_id)
    return {
        "session_id": session_id,
        "session_created": session_created,
        "retention_trace": trace,
        "session": snapshot.to_dict() if snapshot else None,
    }
