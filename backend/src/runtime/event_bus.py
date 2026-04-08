"""Runtime event bus contracts for CONTROL/DATA/EVENT plane signaling."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict


_RUNTIME_PLANES = ("CONTROL_BUS", "DATA_BUS", "EVENT_BUS")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RuntimeEvent:
    """Cross-plane event envelope used by runtime orchestration surfaces."""

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

    def to_sse(self) -> str:
        import json

        return f"data: {json.dumps(self.to_dict())}\n\n"


class RuntimeEventBus:
    """In-memory pub/sub bus for runtime event envelopes."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._lock = Lock()
        self._published_count = 0

    def subscribe(self, subscriber_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers[subscriber_id] = queue
        return queue

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, event: RuntimeEvent) -> int:
        with self._lock:
            queues = list(self._subscribers.values())
            self._published_count += 1

        delivered = 0
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                continue

        return delivered

    def emit(
        self,
        *,
        plane: str,
        event_type: str,
        payload: Dict[str, Any] | None = None,
        trace_id: str = "",
        session_id: str = "",
    ) -> RuntimeEvent:
        normalized_plane = (plane or "EVENT_BUS").strip().upper()
        if normalized_plane not in _RUNTIME_PLANES:
            raise ValueError(f"Unsupported runtime event plane: {plane}")

        event = RuntimeEvent(
            message_id=f"evt-{uuid.uuid4().hex[:12]}",
            trace_id=trace_id or f"trace-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            plane=normalized_plane,
            event_type=event_type,
            payload=dict(payload or {}),
            timestamp=_utc_now_iso(),
        )
        self.publish(event)
        return event

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "subscribers": len(self._subscribers),
                "published": self._published_count,
            }


runtime_events = RuntimeEventBus()
