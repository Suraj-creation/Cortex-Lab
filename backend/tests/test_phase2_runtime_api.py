"""Phase 2 API integration tests for approval worker runtime endpoints."""

import os
import sys
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as backend_server


class _FakeApprovalWorker:
    def __init__(self):
        self.run_once_calls = 0

    def get_status(self):
        return {
            "running": True,
            "poll_interval_seconds": 2.0,
            "execution_timeout_seconds": 60.0,
            "max_attempts": 2,
            "summary": {
                "approved_total": 3,
                "pending_total": 1,
                "running": 1,
                "completed": 2,
                "failed": 0,
                "unsupported": 0,
                "idle": 0,
            },
        }

    async def run_once(self):
        self.run_once_calls += 1


class _FakeEngine:
    def expire_permission_requests(self):
        return []

    def get_pending_permissions(self):
        return []

    def resolve_permission_request(self, permission_id: str, approve: bool, actor: str, note: str = ""):
        return {
            "permission_id": permission_id,
            "status": "approved" if approve else "denied",
            "decided_by": actor,
            "decision_note": note,
        }


def _client_with_fakes(monkeypatch):
    fake_engine = _FakeEngine()
    fake_worker = _FakeApprovalWorker()

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    monkeypatch.setattr(backend_server.app.router, "lifespan_context", _no_lifespan)
    monkeypatch.setattr(backend_server, "rag_engine", fake_engine)
    monkeypatch.setattr(backend_server, "_approval_execution_worker", fake_worker)

    client = TestClient(backend_server.app)
    return client, fake_worker


def test_runtime_safety_executor_endpoint_returns_worker_status(monkeypatch):
    client, _worker = _client_with_fakes(monkeypatch)
    with client:
        response = client.get("/api/runtime/safety/executor")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["running"] is True
    assert payload["summary"]["approved_total"] == 3
    assert payload["summary"]["running"] == 1


def test_runtime_safety_resolve_approve_triggers_worker_execution(monkeypatch):
    client, worker = _client_with_fakes(monkeypatch)

    with client:
        response = client.post(
            "/api/runtime/safety/permissions/perm-123/resolve",
            json={
                "approve": True,
                "actor": "api-test",
                "note": "approve from test",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved"]["permission_id"] == "perm-123"
    assert payload["resolved"]["status"] == "approved"
    assert worker.run_once_calls == 1


def test_runtime_safety_resolve_deny_skips_worker_execution(monkeypatch):
    client, worker = _client_with_fakes(monkeypatch)

    with client:
        response = client.post(
            "/api/runtime/safety/permissions/perm-456/resolve",
            json={
                "approve": False,
                "actor": "api-test",
                "note": "deny from test",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved"]["permission_id"] == "perm-456"
    assert payload["resolved"]["status"] == "denied"
    assert worker.run_once_calls == 0
