"""
Cortex Lab — Real-Time Pipeline Observability
Provides live SSE event broadcasting for pipeline step tracking.
Each pipeline step emits events as it starts, progresses, and completes,
allowing the frontend to visualize the RAG pipeline in real-time.
"""

import asyncio
import time
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Deque, Dict, List, Optional


@dataclass
class PipelineEvent:
    """A single real-time pipeline event emitted during query processing."""
    event_type: str  # step_start, step_complete, step_skip, step_error, metric, pipeline_start, pipeline_complete
    step_name: str = ""
    step_type: str = ""  # query_analysis, routing, query_transform, retrieval, reranking, agent_execution, crag, self_rag, flare, generation, cache_check, compression
    status: str = ""  # running, completed, skipped, error
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""

    def to_sse(self) -> str:
        """Serialize to SSE data line."""
        payload = {
            "event_type": self.event_type,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "details": self.details,
            "timestamp": round(self.timestamp, 3),
            "trace_id": self.trace_id,
        }
        return f"data: {json.dumps(payload)}\n\n"


class PipelineEventBus:
    """
    Event bus for broadcasting pipeline step events to SSE listeners.
    Supports multiple concurrent listeners (one per active request).
    Thread-safe via asyncio.Queue per subscriber.
    """

    def __init__(self):
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._global_subscribers: Dict[str, asyncio.Queue] = {}
        # Keep last N events for late-joining subscribers
        self._recent_events: Deque[PipelineEvent] = deque(maxlen=50)
        # Aggregate metrics
        self._metrics = {
            "total_queries": 0,
            "total_steps_executed": 0,
            "avg_pipeline_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "reranker_invocations": 0,
            "compression_invocations": 0,
            "importance_boosts_applied": 0,
        }

    def subscribe(self, trace_id: str) -> asyncio.Queue:
        """Subscribe to events for a specific trace_id. Returns an asyncio.Queue."""
        queue = asyncio.Queue(maxsize=100)
        self._subscribers[trace_id] = queue
        return queue

    def unsubscribe(self, trace_id: str):
        """Remove subscriber."""
        self._subscribers.pop(trace_id, None)

    def subscribe_global(self, subscriber_id: str) -> asyncio.Queue:
        """Subscribe to ALL pipeline events (broadcast). Returns an asyncio.Queue."""
        queue = asyncio.Queue(maxsize=200)
        self._global_subscribers[subscriber_id] = queue
        return queue

    def unsubscribe_global(self, subscriber_id: str):
        """Remove global subscriber."""
        self._global_subscribers.pop(subscriber_id, None)

    async def emit(self, event: PipelineEvent):
        """Emit an event to the subscriber for this trace_id AND all global subscribers."""
        self._recent_events.append(event)
        self._metrics["total_steps_executed"] += 1

        # Per-trace subscriber
        queue = self._subscribers.get(event.trace_id)
        if queue:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        # Global broadcast subscribers
        for q in self._global_subscribers.values():
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def emit_step_start(self, trace_id: str, step_name: str, step_type: str,
                               details: Dict = None):
        """Convenience: emit a step_start event."""
        await self.emit(PipelineEvent(
            event_type="step_start",
            step_name=step_name,
            step_type=step_type,
            status="running",
            details=details or {},
            trace_id=trace_id,
        ))

    async def emit_step_complete(self, trace_id: str, step_name: str, step_type: str,
                                  duration_ms: float, details: Dict = None):
        """Convenience: emit a step_complete event."""
        await self.emit(PipelineEvent(
            event_type="step_complete",
            step_name=step_name,
            step_type=step_type,
            status="completed",
            duration_ms=duration_ms,
            details=details or {},
            trace_id=trace_id,
        ))

    async def emit_step_skip(self, trace_id: str, step_name: str, step_type: str,
                              reason: str = ""):
        """Convenience: emit a step_skip event."""
        await self.emit(PipelineEvent(
            event_type="step_skip",
            step_name=step_name,
            step_type=step_type,
            status="skipped",
            details={"reason": reason},
            trace_id=trace_id,
        ))

    async def emit_pipeline_start(self, trace_id: str, query: str):
        """Emit pipeline_start event."""
        self._metrics["total_queries"] += 1
        await self.emit(PipelineEvent(
            event_type="pipeline_start",
            step_name="Pipeline",
            status="running",
            details={"query": query[:200]},
            trace_id=trace_id,
        ))

    async def emit_pipeline_complete(self, trace_id: str, total_ms: float,
                                      details: Dict = None):
        """Emit pipeline_complete event."""
        # Update running avg
        n = self._metrics["total_queries"]
        old_avg = self._metrics["avg_pipeline_ms"]
        self._metrics["avg_pipeline_ms"] = old_avg + (total_ms - old_avg) / max(n, 1)

        await self.emit(PipelineEvent(
            event_type="pipeline_complete",
            step_name="Pipeline",
            status="completed",
            duration_ms=total_ms,
            details=details or {},
            trace_id=trace_id,
        ))

    async def emit_metric(self, trace_id: str, metric_name: str, value: Any):
        """Emit a metric event (e.g. cache_hit, compression_ratio)."""
        if metric_name in self._metrics and isinstance(value, (int, float)):
            self._metrics[metric_name] += value
        await self.emit(PipelineEvent(
            event_type="metric",
            step_name=metric_name,
            details={"value": value},
            trace_id=trace_id,
        ))

    def get_metrics(self) -> Dict:
        """Return aggregate observability metrics."""
        return dict(self._metrics)

    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """Return recent events for dashboard."""
        events = list(self._recent_events)[-limit:]
        return [{"event_type": e.event_type, "step_name": e.step_name,
                 "step_type": e.step_type, "status": e.status,
                 "duration_ms": round(e.duration_ms, 1),
                 "details": e.details, "trace_id": e.trace_id}
                for e in events]


# Global event bus singleton
pipeline_events = PipelineEventBus()
