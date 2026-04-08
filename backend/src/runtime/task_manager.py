"""Phase 3 runtime task manager for subagent isolation and cancellation control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import uuid

from .contracts import TaskLifecycle, TaskState


@dataclass
class RuntimeTask:
    """One runtime task with lifecycle, permission scope, and parent-child links."""

    task_id: str
    lifecycle: TaskLifecycle
    parent_task_id: Optional[str] = None
    permission_scope: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    child_task_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attached_asyncio_task: Optional[asyncio.Task] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "state": self.lifecycle.state.value,
            "permission_scope": sorted(self.permission_scope) if self.permission_scope is not None else None,
            "child_task_ids": list(self.child_task_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.lifecycle.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeTaskEvent:
    """Structured runtime task lifecycle event for SSE subscribers."""

    event_id: str
    sequence: int
    event_type: str
    timestamp: str
    task: Dict[str, Any]
    previous_state: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "task": dict(self.task),
            "previous_state": self.previous_state,
            "state": self.task.get("state", ""),
            "note": self.note,
        }


class RuntimeTaskManager:
    """Tracks foreground/background tasks and enforces subagent permission scoping."""

    def __init__(self):
        self._tasks: Dict[str, RuntimeTask] = {}
        self._subscribers: Dict[str, asyncio.Queue[Dict[str, Any]]] = {}
        self._event_sequence: int = 0

    def create_task(
        self,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        permission_scope: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuntimeTask:
        resolved_task_id = task_id or str(uuid.uuid4())
        if resolved_task_id in self._tasks:
            raise ValueError(f"Task already exists: {resolved_task_id}")

        parent = self._tasks.get(parent_task_id) if parent_task_id else None
        if parent_task_id and parent is None:
            raise KeyError(f"Parent task not found: {parent_task_id}")

        resolved_scope = self._resolve_permission_scope(
            parent_scope=parent.permission_scope if parent else None,
            requested_scope=permission_scope,
        )

        lifecycle = TaskLifecycle(task_id=resolved_task_id)
        task = RuntimeTask(
            task_id=resolved_task_id,
            lifecycle=lifecycle,
            parent_task_id=parent_task_id,
            permission_scope=resolved_scope,
            metadata=dict(metadata or {}),
        )

        self._tasks[resolved_task_id] = task
        if parent is not None:
            parent.child_task_ids.append(resolved_task_id)

        self._publish_event("task_created", task)

        return task

    @staticmethod
    def _resolve_permission_scope(
        parent_scope: Optional[Set[str]],
        requested_scope: Optional[Set[str]],
    ) -> Optional[Set[str]]:
        if parent_scope is None:
            return set(requested_scope) if requested_scope is not None else None

        if requested_scope is None:
            return set(parent_scope)

        resolved = set(requested_scope)
        if not resolved.issubset(parent_scope):
            forbidden = sorted(resolved - parent_scope)
            raise ValueError(
                "Subagent permission scope exceeds parent scope: "
                + ", ".join(forbidden)
            )
        return resolved

    def get_task(self, task_id: str) -> RuntimeTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def list_tasks(self) -> List[RuntimeTask]:
        return list(self._tasks.values())

    def can_use_tool(self, task_id: str, tool_name: str) -> bool:
        scope = self.get_task(task_id).permission_scope
        if scope is None:
            return True
        return tool_name in scope

    def subscribe(self, subscriber_id: str) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers[subscriber_id] = queue
        return queue

    def unsubscribe(self, subscriber_id: str) -> None:
        self._subscribers.pop(subscriber_id, None)

    def _publish_event(
        self,
        event_type: str,
        task: RuntimeTask,
        previous_state: Optional[str] = None,
        note: str = "",
    ) -> None:
        if not self._subscribers:
            return

        self._event_sequence += 1
        event = RuntimeTaskEvent(
            event_id=f"task-evt-{uuid.uuid4().hex[:12]}",
            sequence=self._event_sequence,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=task.to_dict(),
            previous_state=previous_state,
            note=note,
        ).to_dict()

        for queue in list(self._subscribers.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    def _transition_task(self, task_id: str, target_state: TaskState, note: str = "") -> None:
        task = self.get_task(task_id)
        previous_state = task.lifecycle.state.value
        task.lifecycle.transition_to(target_state, note=note)
        self._publish_event(
            event_type="task_transition",
            task=task,
            previous_state=previous_state,
            note=note,
        )

    def attach_asyncio_task(self, task_id: str, task: asyncio.Task) -> None:
        runtime_task = self.get_task(task_id)
        runtime_task.attached_asyncio_task = task
        self._publish_event(
            event_type="task_attached",
            task=runtime_task,
            note="asyncio task attached",
        )

    def mark_task_running(self, task_id: str, note: str = "") -> None:
        self._transition_task(task_id, TaskState.RUNNING, note=note)

    def mark_task_waiting_approval(self, task_id: str, note: str = "") -> None:
        self._transition_task(task_id, TaskState.WAITING_APPROVAL, note=note)

    def mark_task_blocked(self, task_id: str, note: str = "") -> None:
        self._transition_task(task_id, TaskState.BLOCKED, note=note)

    def mark_task_completed(self, task_id: str, note: str = "") -> None:
        self._transition_task(task_id, TaskState.COMPLETED, note=note)

    def mark_task_failed(self, task_id: str, note: str = "") -> None:
        self._transition_task(task_id, TaskState.FAILED, note=note)

    def cancel_task(self, task_id: str, reason: str = "", propagate: bool = True) -> List[str]:
        cancelled: List[str] = []
        for current_task_id in self._cancellation_order(task_id, propagate=propagate):
            task = self.get_task(current_task_id)

            if task.lifecycle.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
                continue

            if task.lifecycle.can_transition_to(TaskState.CANCELLED):
                previous_state = task.lifecycle.state.value
                task.lifecycle.transition_to(TaskState.CANCELLED, note=reason)
                cancelled.append(current_task_id)
                self._publish_event(
                    event_type="task_transition",
                    task=task,
                    previous_state=previous_state,
                    note=reason,
                )

            attached = task.attached_asyncio_task
            if attached is not None and not attached.done():
                attached.cancel()

        return cancelled

    def _cancellation_order(self, task_id: str, propagate: bool) -> List[str]:
        root = self.get_task(task_id)
        ordered: List[str] = [root.task_id]
        if not propagate:
            return ordered

        for child_id in root.child_task_ids:
            ordered.extend(self._cancellation_order(child_id, propagate=True))

        return ordered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self._tasks),
            "tasks": [task.to_dict() for task in self._tasks.values()],
        }
