"""Phase 3 API integration tests for runtime task lifecycle endpoints."""

import os
import sys
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as backend_server
from src.runtime.task_manager import RuntimeTaskManager


def _client_with_task_manager(monkeypatch):
    task_manager = RuntimeTaskManager()

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    monkeypatch.setattr(backend_server.app.router, "lifespan_context", _no_lifespan)
    monkeypatch.setattr(backend_server, "_runtime_task_manager", task_manager)

    client = TestClient(backend_server.app)
    return client


def test_runtime_tasks_create_and_list(monkeypatch):
    client = _client_with_task_manager(monkeypatch)

    with client:
        parent_resp = client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "parent",
                "permission_scope": ["search_memories", "rag_chat"],
            },
        )
        child_resp = client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "child",
                "parent_task_id": "parent",
                "permission_scope": ["search_memories"],
            },
        )
        list_resp = client.get("/api/runtime/tasks")

    assert parent_resp.status_code == 200
    assert child_resp.status_code == 200
    assert list_resp.status_code == 200

    list_payload = list_resp.json()
    assert list_payload["count"] == 2

    by_id = {task["task_id"]: task for task in list_payload["tasks"]}
    assert by_id["parent"]["state"] == "queued"
    assert by_id["child"]["parent_task_id"] == "parent"
    assert by_id["child"]["permission_scope"] == ["search_memories"]


def test_runtime_tasks_reject_subagent_scope_escalation(monkeypatch):
    client = _client_with_task_manager(monkeypatch)

    with client:
        parent_resp = client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "parent",
                "permission_scope": ["search_memories"],
            },
        )
        child_resp = client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "child-escalation",
                "parent_task_id": "parent",
                "permission_scope": ["delete_memory"],
            },
        )

    assert parent_resp.status_code == 200
    assert child_resp.status_code == 400
    assert "exceeds parent scope" in child_resp.json()["detail"]


def test_runtime_tasks_cancel_propagates_to_children(monkeypatch):
    client = _client_with_task_manager(monkeypatch)

    with client:
        client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "parent",
                "permission_scope": ["search_memories"],
            },
        )
        client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "child",
                "parent_task_id": "parent",
            },
        )

        cancel_resp = client.post(
            "/api/runtime/tasks/parent/cancel",
            json={
                "reason": "operator cancel",
                "propagate": True,
            },
        )
        child_resp = client.get("/api/runtime/tasks/child")

    assert cancel_resp.status_code == 200
    cancel_payload = cancel_resp.json()
    assert cancel_payload["cancelled_task_ids"] == ["parent", "child"]

    assert child_resp.status_code == 200
    assert child_resp.json()["task"]["state"] == "cancelled"


def test_runtime_tasks_cancel_without_propagation_keeps_child_queued(monkeypatch):
    client = _client_with_task_manager(monkeypatch)

    with client:
        client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "parent",
                "permission_scope": ["search_memories"],
            },
        )
        client.post(
            "/api/runtime/tasks",
            json={
                "task_id": "child",
                "parent_task_id": "parent",
            },
        )

        cancel_resp = client.post(
            "/api/runtime/tasks/parent/cancel",
            json={
                "reason": "operator cancel",
                "propagate": False,
            },
        )
        child_resp = client.get("/api/runtime/tasks/child")

    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["cancelled_task_ids"] == ["parent"]

    assert child_resp.status_code == 200
    assert child_resp.json()["task"]["state"] == "queued"


@pytest.mark.asyncio
async def test_runtime_task_events_stream_uses_sse_media_type(monkeypatch):
    _client_with_task_manager(monkeypatch)

    class _FakeRequest:
        async def is_disconnected(self):
            return True

    response = await backend_server.runtime_task_events(_FakeRequest())
    assert response.media_type == "text/event-stream"


def test_runtime_task_manager_emits_lifecycle_events_for_sse_subscribers(monkeypatch):
    _client_with_task_manager(monkeypatch)
    subscriber_id = "test-subscriber"
    queue = backend_server._runtime_task_manager.subscribe(subscriber_id)

    try:
        backend_server._runtime_task_manager.create_task(
            task_id="event-task",
            permission_scope={"search_memories"},
        )
        backend_server._runtime_task_manager.cancel_task(
            "event-task",
            reason="operator cancel",
            propagate=True,
        )

        created_event = queue.get_nowait()
        transition_event = queue.get_nowait()
    finally:
        backend_server._runtime_task_manager.unsubscribe(subscriber_id)

    assert created_event["event_type"] == "task_created"
    assert created_event["task"]["task_id"] == "event-task"

    assert transition_event["event_type"] == "task_transition"
    assert transition_event["task"]["task_id"] == "event-task"
    assert transition_event["state"] == "cancelled"
