"""L0 master-orchestrator runtime service scaffolding.

This service intentionally keeps behavior lightweight while exposing explicit
entry points for wake/sleep control, active processing transitions, and
resource-tier enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .contracts import (
    MasterLifecycleTransitionError,
    MasterOrchestratorState,
    MasterOrchestratorStateMachine,
    ResourceTier,
)


@dataclass
class MasterOrchestratorService:
    """Service wrapper for the L0 master-orchestrator runtime state machine."""

    state_machine: MasterOrchestratorStateMachine = field(default_factory=MasterOrchestratorStateMachine)
    last_transition_reason: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def _transition(self, target: MasterOrchestratorState, trigger: str, note: str = "") -> bool:
        try:
            self.state_machine.transition_to(target, trigger=trigger, note=note)
        except MasterLifecycleTransitionError:
            return False

        self.last_transition_reason = note or trigger
        self.updated_at = datetime.now(timezone.utc)
        return True

    def wake(self, trigger: str = "explicit_wake") -> bool:
        if self.state_machine.state == MasterOrchestratorState.EMERGENCY_MUTE:
            return False
        if self.state_machine.state == MasterOrchestratorState.SLEEPING:
            return self._transition(
                MasterOrchestratorState.PASSIVE_MONITORING,
                trigger=trigger,
                note="wake to passive monitoring",
            )
        return True

    def sleep(self, reason: str = "idle_timeout") -> bool:
        if self.state_machine.state == MasterOrchestratorState.SLEEPING:
            return True
        return self._transition(
            MasterOrchestratorState.SLEEPING,
            trigger="sleep",
            note=reason,
        )

    def begin_listening(self, trigger: str = "user_detected") -> bool:
        if self.state_machine.state == MasterOrchestratorState.PASSIVE_MONITORING:
            return self._transition(
                MasterOrchestratorState.ACTIVE_LISTENING,
                trigger=trigger,
                note="session opened",
            )
        if self.state_machine.state == MasterOrchestratorState.ACTIVE_LISTENING:
            return True
        return False

    def begin_processing(self, trigger: str = "query_received") -> bool:
        if self.state_machine.state == MasterOrchestratorState.SLEEPING:
            woke = self.wake(trigger="scheduled_processing")
            if not woke:
                return False

        if self.state_machine.state in {
            MasterOrchestratorState.PASSIVE_MONITORING,
            MasterOrchestratorState.ACTIVE_LISTENING,
            MasterOrchestratorState.DEGRADED_MODE,
        }:
            return self._transition(
                MasterOrchestratorState.ACTIVE_PROCESSING,
                trigger=trigger,
                note="processing started",
            )
        if self.state_machine.state == MasterOrchestratorState.ACTIVE_PROCESSING:
            return True
        return False

    def complete_processing(self) -> bool:
        if self.state_machine.state == MasterOrchestratorState.ACTIVE_PROCESSING:
            return self._transition(
                MasterOrchestratorState.ACTIVE_LISTENING,
                trigger="processing_complete",
                note="return to listening",
            )
        return False

    def apply_health_snapshot(
        self,
        battery_pct: Optional[float],
        thermal_state: str,
        network_state: str = "good",
        charging: bool = False,
    ) -> ResourceTier:
        tier = self.state_machine.update_health(
            battery_pct=battery_pct,
            thermal_state=thermal_state,
            network_state=network_state,
            charging=charging,
        )
        self.updated_at = datetime.now(timezone.utc)
        return tier

    def force_emergency_mute(self, reason: str = "manual_emergency_mute") -> bool:
        self.state_machine.resource_tier = ResourceTier.TIER_4_EMERGENCY
        if self.state_machine.state == MasterOrchestratorState.EMERGENCY_MUTE:
            self.last_transition_reason = reason
            self.updated_at = datetime.now(timezone.utc)
            return True
        return self._transition(
            MasterOrchestratorState.EMERGENCY_MUTE,
            trigger="emergency_mute",
            note=reason,
        )

    def clear_emergency_mute(self, reason: str = "manual_resume") -> bool:
        if self.state_machine.state != MasterOrchestratorState.EMERGENCY_MUTE:
            return False
        self.state_machine.resource_tier = ResourceTier.TIER_2_CONSERVATIVE
        return self._transition(
            MasterOrchestratorState.PASSIVE_MONITORING,
            trigger="resume",
            note=reason,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state_machine.state.value,
            "resource_tier": self.state_machine.resource_tier.value,
            "last_transition_reason": self.last_transition_reason,
            "updated_at": self.updated_at.isoformat(),
            "state_machine": self.state_machine.to_dict(),
        }
