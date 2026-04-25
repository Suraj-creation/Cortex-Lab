"""Runtime session manager for voice/text/hybrid orchestration metadata."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeSession:
    session_id: str
    mode: str
    start_time: str
    end_time: Optional[str] = None
    user_detected: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    retention_summary: Dict[str, int] = field(
        default_factory=lambda: {
            "discarded": 0,
            "session_only": 0,
            "structured": 0,
            "priority": 0,
        }
    )
    agent_tags: List[str] = field(default_factory=list)

    def close(self, reason: str = "") -> None:
        self.end_time = _utc_now_iso()
        if reason:
            self.metadata["close_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "user_detected": self.user_detected,
            "metadata": dict(self.metadata),
            "retention_summary": dict(self.retention_summary),
            "agent_tags": list(self.agent_tags),
        }


class RuntimeSessionManager:
    """Thread-safe runtime session registry."""

    def __init__(self) -> None:
        self._sessions: Dict[str, RuntimeSession] = {}
        self._active_ids: List[str] = []
        self._lock = Lock()

    def open_session(
        self,
        *,
        mode: str,
        user_detected: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuntimeSession:
        session = RuntimeSession(
            session_id=f"session-{uuid.uuid4().hex[:12]}",
            mode=mode,
            start_time=_utc_now_iso(),
            user_detected=user_detected,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._active_ids.append(session.session_id)

        return session

    def close_session(
        self,
        session_id: str,
        *,
        reason: str = "",
        retention_summary: Optional[Dict[str, int]] = None,
        agent_tags: Optional[List[str]] = None,
    ) -> RuntimeSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session_not_found:{session_id}")

            session = self._sessions[session_id]
            session.close(reason=reason)
            if retention_summary is not None:
                session.retention_summary = dict(retention_summary)
            if agent_tags is not None:
                session.agent_tags = list(agent_tags)
            if session_id in self._active_ids:
                self._active_ids.remove(session_id)

        return session

    def get_session(self, session_id: str) -> Optional[RuntimeSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def update_metadata(self, session_id: str, metadata: Dict[str, Any]) -> RuntimeSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session_not_found:{session_id}")
            session = self._sessions[session_id]
            session.metadata.update(dict(metadata or {}))
            return session

    def merge_retention_summary(self, session_id: str, updates: Dict[str, int]) -> RuntimeSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session_not_found:{session_id}")
            session = self._sessions[session_id]
            for key, value in dict(updates or {}).items():
                try:
                    session.retention_summary[key] = int(session.retention_summary.get(key, 0)) + int(value)
                except Exception:
                    continue
            return session

    def append_agent_tags(self, session_id: str, tags: List[str]) -> RuntimeSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session_not_found:{session_id}")
            session = self._sessions[session_id]
            merged = list(session.agent_tags)
            for tag in list(tags or []):
                normalized = str(tag or "").strip()
                if normalized and normalized not in merged:
                    merged.append(normalized)
            session.agent_tags = merged
            return session

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())

        sessions.sort(key=lambda item: item.start_time, reverse=True)
        return [session.to_dict() for session in sessions[: max(limit, 1)]]

    def active_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            active = [self._sessions[sid] for sid in self._active_ids if sid in self._sessions]
        return [session.to_dict() for session in active]


runtime_session_manager = RuntimeSessionManager()
