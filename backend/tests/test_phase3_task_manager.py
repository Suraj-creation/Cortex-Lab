"""Phase 3 task manager tests for subagent isolation and cancellation propagation."""

import asyncio
import os
import sys

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_subagent_permission_scope_must_be_subset_of_parent_scope():
    from src.runtime.task_manager import RuntimeTaskManager

    manager = RuntimeTaskManager()
    parent = manager.create_task(
        task_id="parent",
        permission_scope={"search_memories", "rag_chat"},
    )

    child = manager.create_task(
        task_id="child-safe",
        parent_task_id=parent.task_id,
        permission_scope={"search_memories"},
    )

    assert child.permission_scope == {"search_memories"}

    with pytest.raises(ValueError):
        manager.create_task(
            task_id="child-escalation",
            parent_task_id=parent.task_id,
            permission_scope={"delete_memory"},
        )


def test_subagent_inherits_parent_scope_when_scope_is_not_provided():
    from src.runtime.task_manager import RuntimeTaskManager

    manager = RuntimeTaskManager()
    parent = manager.create_task(
        task_id="parent",
        permission_scope={"search_memories", "get_memories"},
    )

    child = manager.create_task(
        task_id="child-inherited",
        parent_task_id=parent.task_id,
    )

    assert child.permission_scope == {"search_memories", "get_memories"}


def test_task_scope_enforcement_blocks_disallowed_tool_use():
    from src.runtime.task_manager import RuntimeTaskManager

    manager = RuntimeTaskManager()
    task = manager.create_task(
        task_id="task-scope",
        permission_scope={"search_memories"},
    )

    assert manager.can_use_tool(task.task_id, "search_memories") is True
    assert manager.can_use_tool(task.task_id, "delete_memory") is False


@pytest.mark.asyncio
async def test_parent_cancellation_propagates_to_children_and_attached_async_tasks():
    from src.runtime.contracts import TaskState
    from src.runtime.task_manager import RuntimeTaskManager

    manager = RuntimeTaskManager()
    parent = manager.create_task(task_id="parent", permission_scope={"search_memories"})
    child_a = manager.create_task(task_id="child-a", parent_task_id=parent.task_id)
    child_b = manager.create_task(task_id="child-b", parent_task_id=parent.task_id)

    manager.mark_task_running(parent.task_id)
    manager.mark_task_running(child_a.task_id)
    manager.mark_task_running(child_b.task_id)

    started = asyncio.Event()

    async def _long_running():
        started.set()
        await asyncio.sleep(60)

    background = asyncio.create_task(_long_running())
    await started.wait()
    manager.attach_asyncio_task(child_a.task_id, background)

    cancelled = manager.cancel_task(parent.task_id, reason="operator cancel")

    assert cancelled == ["parent", "child-a", "child-b"]
    assert manager.get_task("parent").lifecycle.state == TaskState.CANCELLED
    assert manager.get_task("child-a").lifecycle.state == TaskState.CANCELLED
    assert manager.get_task("child-b").lifecycle.state == TaskState.CANCELLED
    assert background.cancelling() > 0

    with pytest.raises(asyncio.CancelledError):
        await background


@pytest.mark.asyncio
async def test_cancellation_without_propagation_keeps_child_running():
    from src.runtime.contracts import TaskState
    from src.runtime.task_manager import RuntimeTaskManager

    manager = RuntimeTaskManager()
    parent = manager.create_task(task_id="parent", permission_scope={"search_memories"})
    child = manager.create_task(task_id="child", parent_task_id=parent.task_id)

    manager.mark_task_running(parent.task_id)
    manager.mark_task_running(child.task_id)

    cancelled = manager.cancel_task(parent.task_id, propagate=False)

    assert cancelled == ["parent"]
    assert manager.get_task("parent").lifecycle.state == TaskState.CANCELLED
    assert manager.get_task("child").lifecycle.state == TaskState.RUNNING
