from src.runtime.manual_memory_intake import (
    build_manual_memory_retention_trace,
    prepare_manual_memory_session,
)
from src.runtime.session_manager import RuntimeSessionManager


def test_manual_memory_retention_promotes_explicit_submission():
    trace = build_manual_memory_retention_trace(
        "passport",
        session_id="session-demo",
        platform="mobile",
        source="manual_memory",
        force_keep=True,
    )

    assert trace["decision"] == "keep"
    assert trace["memory_decision"] == "structured"
    assert "manual_submission" in trace["tags"]
    assert trace["session_id"] == "session-demo"


def test_prepare_manual_memory_session_creates_session_and_updates_summary():
    manager = RuntimeSessionManager()

    result = prepare_manual_memory_session(
        "Need to review the architecture doc tonight.",
        platform="mobile",
        source="manual_memory",
        session_manager=manager,
    )

    assert result["session_created"] is True
    assert result["session_id"].startswith("session-")
    assert result["session"]["mode"] == "manual_memory"
    assert result["session"]["retention_summary"]["priority"] == 1
    assert "manual_submission" in result["session"]["agent_tags"]
