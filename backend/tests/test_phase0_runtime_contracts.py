"""
Phase 0 contract tests for autonomous runtime interfaces.

These tests lock the baseline interfaces promised in newBranch-Plan.md:
1. Tool contract schema
2. Runtime loop interface
3. Policy interface
4. Task state model
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_tool_contract_generates_json_schema():
    from src.runtime.contracts import ToolContract, ToolParameterSpec, ToolRiskTier

    contract = ToolContract(
        name="search_memories",
        description="Search user memories by semantic similarity.",
        parameters=[
            ToolParameterSpec(
                name="query",
                param_type="string",
                description="User query text.",
                required=True,
            ),
            ToolParameterSpec(
                name="top_k",
                param_type="integer",
                description="Maximum number of results.",
                required=False,
                default=10,
            ),
        ],
        risk_tier=ToolRiskTier.LOW,
        side_effect_free=True,
    )

    schema = contract.to_json_schema()

    assert schema["type"] == "object"
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["top_k"]["type"] == "integer"
    assert "query" in schema["required"]
    assert "top_k" not in schema["required"]


def test_runtime_loop_budget_rejects_non_positive_limits():
    from src.runtime.contracts import RuntimeLoopBudget

    with pytest.raises(ValueError):
        RuntimeLoopBudget(max_iterations=0)

    with pytest.raises(ValueError):
        RuntimeLoopBudget(max_tool_calls_per_window=-1)


def test_stop_reason_contains_guardrail_reasons():
    from src.runtime.contracts import StopReason

    names = {reason.value for reason in StopReason}
    assert "completed" in names
    assert "max_iterations" in names
    assert "max_tool_calls" in names
    assert "max_tokens" in names
    assert "time_budget_exceeded" in names
    assert "policy_denied" in names


def test_policy_decision_flags_human_approval_correctly():
    from src.runtime.contracts import PolicyDecision, PolicyEffect

    allow = PolicyDecision(effect=PolicyEffect.ALLOW, reason="safe read")
    review = PolicyDecision(effect=PolicyEffect.REQUIRE_APPROVAL, reason="filesystem write")

    assert allow.requires_human_approval is False
    assert review.requires_human_approval is True


def test_task_lifecycle_allows_phase0_transitions():
    from src.runtime.contracts import TaskLifecycle, TaskState

    lifecycle = TaskLifecycle(task_id="task-1")
    assert lifecycle.state == TaskState.QUEUED

    lifecycle.transition_to(TaskState.RUNNING, note="worker started")
    lifecycle.transition_to(TaskState.WAITING_APPROVAL, note="awaiting permission")
    lifecycle.transition_to(TaskState.RUNNING, note="approved and resumed")
    lifecycle.transition_to(TaskState.COMPLETED, note="done")

    assert lifecycle.state == TaskState.COMPLETED
    assert len(lifecycle.history) >= 4


def test_task_lifecycle_rejects_invalid_transition():
    from src.runtime.contracts import LifecycleTransitionError, TaskLifecycle, TaskState

    lifecycle = TaskLifecycle(task_id="task-2")
    lifecycle.transition_to(TaskState.RUNNING)
    lifecycle.transition_to(TaskState.COMPLETED)

    with pytest.raises(LifecycleTransitionError):
        lifecycle.transition_to(TaskState.RUNNING)


def test_core_tool_catalog_covers_current_engine_operations():
    from src.runtime.tool_catalog import build_core_tool_catalog

    contracts = build_core_tool_catalog()
    names = {tool.name for tool in contracts}

    expected = {
        "rag_chat",
        "rag_retrieve",
        "ingest_memory",
        "search_memories",
        "get_memories",
        "delete_memory",
        "get_graph_data",
        "get_entities",
        "get_belief_deltas",
        "get_community_summaries",
        "get_rag_stats",
    }

    assert expected.issubset(names)


def test_catalog_entries_are_versioned_and_have_risk_tiers():
    from src.runtime.tool_catalog import build_core_tool_catalog

    contracts = build_core_tool_catalog()

    for contract in contracts:
        assert contract.version.count(".") == 2
        assert contract.risk_tier.value in {"low", "medium", "high", "critical"}
        assert isinstance(contract.description, str) and contract.description.strip()


def test_engine_exposes_tool_contracts_interface():
    from src.engine import CortexRAGEngine

    engine = CortexRAGEngine(data_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
    contracts = engine.get_tool_contracts()

    assert isinstance(contracts, list)
    assert any(c.get("name") == "rag_chat" for c in contracts)


def test_policy_rules_respect_priority_and_deny_precedence():
    from src.runtime.contracts import PolicyEffect, PolicyInterface, PolicyRule

    rules = [
        PolicyRule(
            rule_id="allow_default",
            tool_name_pattern="*",
            effect=PolicyEffect.ALLOW,
            reason="default allow",
            priority=100,
        ),
        PolicyRule(
            rule_id="deny_delete",
            tool_name_pattern="delete_*",
            effect=PolicyEffect.DENY,
            reason="deletes are blocked",
            priority=10,
        ),
    ]

    policy = PolicyInterface(rules)
    decision = policy.evaluate("delete_memory")

    assert decision.effect == PolicyEffect.DENY
    assert decision.rule_id == "deny_delete"


def test_policy_escalates_dangerous_signal_even_if_allow_rule_matches():
    from src.runtime.contracts import (
        DangerousCommandSignal,
        PolicyEffect,
        PolicyInterface,
        PolicyRule,
        ToolRiskTier,
    )

    rules = [
        PolicyRule(
            rule_id="allow_all",
            tool_name_pattern="*",
            effect=PolicyEffect.ALLOW,
            reason="default allow",
            priority=100,
        )
    ]

    signal = DangerousCommandSignal(
        tool_name="shell_exec",
        command_text="rm -rf /",
        matched_pattern=r"rm\s+-rf",
        severity=ToolRiskTier.CRITICAL,
    )

    decision = PolicyInterface(rules).evaluate(
        tool_name="shell_exec",
        command_text="rm -rf /",
        dangerous_signals=[signal],
    )

    assert decision.effect == PolicyEffect.REQUIRE_APPROVAL
    assert decision.requires_human_approval is True


def test_runtime_loop_state_enforces_deterministic_tool_dispatch_window():
    from src.runtime.contracts import RuntimeLoopBudget, RuntimeLoopState, RuntimeRequestEnvelope, StopReason

    budget = RuntimeLoopBudget(max_tool_calls_per_window=2, window_seconds=30)
    envelope = RuntimeRequestEnvelope(query="dispatch test", budget=budget)
    state = RuntimeLoopState(envelope=envelope)

    t0 = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)

    assert state.try_register_tool_dispatch(now=t0) is True
    assert state.try_register_tool_dispatch(now=t0 + timedelta(seconds=10)) is True
    assert state.try_register_tool_dispatch(now=t0 + timedelta(seconds=20)) is False
    assert state.stop_reason == StopReason.RATE_LIMITED

    snapshot = state.to_dict()
    assert snapshot["tool_window"]["calls_in_window"] == 2
    assert snapshot["tool_window"]["window_seconds"] == 30


def test_runtime_loop_state_allows_dispatch_again_after_window_rollover():
    from src.runtime.contracts import RuntimeLoopBudget, RuntimeLoopState, RuntimeRequestEnvelope

    budget = RuntimeLoopBudget(max_tool_calls_per_window=1, window_seconds=5)
    envelope = RuntimeRequestEnvelope(query="dispatch rollover", budget=budget)
    state = RuntimeLoopState(envelope=envelope)

    t0 = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)

    assert state.try_register_tool_dispatch(now=t0) is True
    assert state.try_register_tool_dispatch(now=t0 + timedelta(seconds=2)) is False
    assert state.try_register_tool_dispatch(now=t0 + timedelta(seconds=7)) is True
