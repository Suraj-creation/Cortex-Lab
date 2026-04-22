"""Phase 6 contract tests for L1 plan-mode and L0 master-orchestrator scaffolding."""

import os
import sys

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_l1_execution_plan_exposes_confirmation_gate_and_arbitration_path():
    from src.runtime.contracts import (
        ConflictResolutionPath,
        L1ExecutionPlan,
        PlanConfirmationGate,
        PlanConfirmationStatus,
        RuntimeExecutionMode,
    )

    plan = L1ExecutionPlan(
        query="Compare my decision quality and stress trends over time.",
        intent="causal",
        complexity=0.86,
        primary_agent="causal",
        selected_agents=["causal", "timeline", "decisions", "wellbeing"],
        execution_mode=RuntimeExecutionMode.PLAN_MODE,
        conflict_resolution_path=ConflictResolutionPath.ARBITRATION_FIRST,
        confirmation_gate=PlanConfirmationGate(
            required=True,
            reasons=["High-stakes multi-agent dispatch."],
            status=PlanConfirmationStatus.PENDING,
        ),
    )

    assert plan.execution_mode == RuntimeExecutionMode.PLAN_MODE
    assert plan.requires_confirmation is True
    assert plan.conflict_resolution_path == ConflictResolutionPath.ARBITRATION_FIRST

    plan.mark_confirmed(actor="test-user", note="approved in unit test")

    assert plan.requires_confirmation is False
    serialized = plan.to_dict()
    assert serialized["confirmation_gate"]["status"] == PlanConfirmationStatus.CONFIRMED.value
    assert serialized["conflict_resolution_path"] == "arbitration_first"
    assert serialized["steps"]


def test_master_orchestrator_state_machine_applies_tier_escalation_and_recovery_rules():
    from src.runtime.contracts import (
        MasterLifecycleTransitionError,
        MasterOrchestratorState,
        MasterOrchestratorStateMachine,
        ResourceTier,
    )

    machine = MasterOrchestratorStateMachine()

    machine.transition_to(MasterOrchestratorState.PASSIVE_MONITORING, trigger="wake")
    machine.transition_to(MasterOrchestratorState.ACTIVE_LISTENING, trigger="user_detected")
    machine.transition_to(MasterOrchestratorState.ACTIVE_PROCESSING, trigger="query_received")

    tier = machine.update_health(battery_pct=18, thermal_state="normal")
    assert tier == ResourceTier.TIER_3_MINIMUM
    assert machine.state == MasterOrchestratorState.DEGRADED_MODE

    tier = machine.update_health(battery_pct=8, thermal_state="critical")
    assert tier == ResourceTier.TIER_4_EMERGENCY
    assert machine.state == MasterOrchestratorState.EMERGENCY_MUTE

    with pytest.raises(MasterLifecycleTransitionError):
        machine.transition_to(MasterOrchestratorState.ACTIVE_PROCESSING, trigger="invalid_resume")

    machine.transition_to(MasterOrchestratorState.PASSIVE_MONITORING, trigger="manual_resume")
    assert machine.state == MasterOrchestratorState.PASSIVE_MONITORING


def test_master_orchestrator_service_handles_wake_process_and_emergency_clear_flow():
    from src.runtime.contracts import MasterOrchestratorState, ResourceTier
    from src.runtime.master_orchestrator import MasterOrchestratorService

    service = MasterOrchestratorService()

    assert service.wake() is True
    assert service.state_machine.state == MasterOrchestratorState.PASSIVE_MONITORING

    assert service.begin_listening() is True
    assert service.begin_processing() is True
    assert service.state_machine.state == MasterOrchestratorState.ACTIVE_PROCESSING

    tier = service.apply_health_snapshot(battery_pct=9, thermal_state="hot")
    assert tier == ResourceTier.TIER_4_EMERGENCY
    assert service.state_machine.state == MasterOrchestratorState.EMERGENCY_MUTE

    assert service.clear_emergency_mute(reason="battery recovered") is True
    assert service.state_machine.state == MasterOrchestratorState.PASSIVE_MONITORING

    status = service.get_status()
    assert status["state"] == MasterOrchestratorState.PASSIVE_MONITORING.value
    assert "state_machine" in status
