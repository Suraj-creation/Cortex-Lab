"""Runtime trace helpers and in-memory trace buffer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex}"


def new_message_id() -> str:
    return f"msg-{uuid.uuid4().hex[:12]}"


def new_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class RuntimeTraceEvent:
    message_id: str
    trace_id: str
    session_id: str
    plane: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: str
    schema_version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "plane": self.plane,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }


class RuntimeTraceBuffer:
    """Bounded trace event store for diagnostics and runtime replay."""

    def __init__(self, max_events: int = 5000) -> None:
        self.max_events = max(max_events, 100)
        self._events: List[RuntimeTraceEvent] = []
        self._lock = Lock()

    def append(self, event: RuntimeTraceEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                overflow = len(self._events) - self.max_events
                if overflow > 0:
                    del self._events[:overflow]

    def record(
        self,
        *,
        plane: str,
        event_type: str,
        payload: Dict[str, Any] | None = None,
        trace_id: str = "",
        session_id: str = "",
    ) -> RuntimeTraceEvent:
        event = RuntimeTraceEvent(
            message_id=new_message_id(),
            trace_id=trace_id or new_trace_id(),
            session_id=session_id,
            plane=plane,
            event_type=event_type,
            payload=dict(payload or {}),
            timestamp=_utc_now_iso(),
        )
        self.append(event)
        return event

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self._events[-max(limit, 1) :])
        return [event.to_dict() for event in reversed(events)]

    def by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            events = [event for event in self._events if event.trace_id == trace_id]
        return [event.to_dict() for event in events]


runtime_trace_buffer = RuntimeTraceBuffer()
