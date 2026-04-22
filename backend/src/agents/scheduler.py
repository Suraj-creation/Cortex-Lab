"""
Background Agent Scheduler — Manages always-on and periodic agents.
Architecture: Orchestrator.md §21.4 (ScheduleConfig)

Runs:
- L0 Master on heartbeat (every 60s)
- Wiki Agent on memory ingest events + daily schedule
- Presence Agent continuously (every 30 min)
- Session Crystallizer every 15 min
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    agent_id: str
    interval_seconds: int
    last_run: float = 0.0
    last_reason: str = ""
    last_event: str = ""
    is_running: bool = False
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""
    enabled: bool = True


class BackgroundScheduler:
    """
    Manages periodic execution of background agents.
    Each task is a (agent_id, coroutine_factory) pair with an interval.
    """

    def __init__(self):
        self._tasks: dict[str, ScheduledTask] = {}
        self._factories: dict[str, Callable[..., Any]] = {}
        self._running = False
        self._tick_interval = 5.0
        self._event_handlers: dict[str, list[str]] = {}

    def register(
        self,
        agent_id: str,
        factory: Callable[..., Any],
        interval_seconds: int = 60,
        events: list[str] | None = None,
    ) -> None:
        self._tasks[agent_id] = ScheduledTask(
            agent_id=agent_id,
            interval_seconds=interval_seconds,
        )
        self._factories[agent_id] = factory

        for event_name in (events or []):
            self._event_handlers.setdefault(event_name, []).append(agent_id)

    def unregister(self, agent_id: str) -> None:
        self._tasks.pop(agent_id, None)
        self._factories.pop(agent_id, None)
        for handlers in self._event_handlers.values():
            if agent_id in handlers:
                handlers.remove(agent_id)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info("Background scheduler started with %d tasks", len(self._tasks))
        asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        self._running = False
        logger.info("Background scheduler stopped")

    async def trigger_event(self, event_name: str, payload: dict[str, Any] | None = None) -> list[str]:
        """Trigger all agents registered for an event."""
        triggered = []
        agent_ids = self._event_handlers.get(event_name, [])
        for agent_id in agent_ids:
            task = self._tasks.get(agent_id)
            if task and task.enabled and not task.is_running:
                asyncio.create_task(self._run_task(agent_id, reason="event", event_name=event_name, payload=payload))
                triggered.append(agent_id)
        return triggered

    def enable(self, agent_id: str) -> None:
        if agent_id in self._tasks:
            self._tasks[agent_id].enabled = True

    def disable(self, agent_id: str) -> None:
        if agent_id in self._tasks:
            self._tasks[agent_id].enabled = False

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "event_handlers": {
                event: list(agent_ids)
                for event, agent_ids in self._event_handlers.items()
            },
            "tasks": {
                tid: {
                    "interval_seconds": t.interval_seconds,
                    "last_run": t.last_run,
                    "last_reason": t.last_reason,
                    "last_event": t.last_event,
                    "is_running": t.is_running,
                    "run_count": t.run_count,
                    "error_count": t.error_count,
                    "enabled": t.enabled,
                }
                for tid, t in self._tasks.items()
            },
        }

    async def _tick_loop(self) -> None:
        while self._running:
            now = time.time()
            for agent_id, task in self._tasks.items():
                if not task.enabled or task.is_running:
                    continue
                elapsed = now - task.last_run
                if elapsed >= task.interval_seconds:
                    asyncio.create_task(self._run_task(agent_id, reason="interval"))
            await asyncio.sleep(self._tick_interval)

    async def _run_task(
        self,
        agent_id: str,
        *,
        reason: str,
        event_name: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        task = self._tasks.get(agent_id)
        if not task:
            return

        task.is_running = True
        task.last_run = time.time()
        task.last_reason = reason
        task.last_event = event_name
        task.run_count += 1

        try:
            factory = self._factories.get(agent_id)
            if factory:
                try:
                    result = factory(payload=payload, reason=reason, event_name=event_name)
                except TypeError:
                    result = factory()
                if asyncio.iscoroutine(result):
                    await result
        except Exception as e:
            task.error_count += 1
            task.last_error = str(e)
            logger.error("Scheduler task %s failed: %s", agent_id, e)
        finally:
            task.is_running = False


background_scheduler = BackgroundScheduler()
