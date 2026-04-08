"""Phase 0 runtime contracts for Cortex autonomous agent infrastructure.

These contracts define the baseline interfaces for:
1. Tool contracts and schemas
2. Runtime loop budgets and stop reasons
3. Policy decisions and auditing
4. Task lifecycle state transitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional
import re
import uuid


_JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "object", "array"}
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class ToolRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StopReason(str, Enum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOKENS = "max_tokens"
    TIME_BUDGET_EXCEEDED = "time_budget_exceeded"
    RATE_LIMITED = "rate_limited"
    POLICY_DENIED = "policy_denied"
    CANCELLED = "cancelled"
    ERROR = "error"


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RuntimeExecutionMode(str, Enum):
    NO_RETRIEVAL = "NO_RETRIEVAL"
    SINGLE_STEP = "SINGLE_STEP"
    MULTI_STEP_SEQUENTIAL = "MULTI_STEP_SEQUENTIAL"
    MULTI_STEP_PARALLEL = "MULTI_STEP_PARALLEL"
    PLAN_MODE = "PLAN_MODE"


class ConflictResolutionPath(str, Enum):
    ARBITRATION_FIRST = "arbitration_first"
    SYNTHESIS_FIRST = "synthesis_first"


class PlanConfirmationStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DENIED = "denied"


class ResourceTier(str, Enum):
    TIER_1_FULL = "tier_1_full"
    TIER_2_CONSERVATIVE = "tier_2_conservative"
    TIER_3_MINIMUM = "tier_3_minimum"
    TIER_4_EMERGENCY = "tier_4_emergency"


class MasterOrchestratorState(str, Enum):
    SLEEPING = "sleeping"
    PASSIVE_MONITORING = "passive_monitoring"
    ACTIVE_LISTENING = "active_listening"
    ACTIVE_PROCESSING = "active_processing"
    DEGRADED_MODE = "degraded_mode"
    EMERGENCY_MUTE = "emergency_mute"


class LifecycleTransitionError(ValueError):
    """Raised when a task state transition is invalid."""


class MasterLifecycleTransitionError(ValueError):
    """Raised when a master-orchestrator state transition is invalid."""


@dataclass
class PlanConfirmationGate:
    """Confirmation gate contract for plan-mode execution in L1."""

    required: bool = False
    reasons: List[str] = field(default_factory=list)
    status: PlanConfirmationStatus = PlanConfirmationStatus.NOT_REQUIRED
    confirmed_by: str = ""
    confirmed_at: Optional[datetime] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.required and self.status == PlanConfirmationStatus.NOT_REQUIRED:
            self.status = PlanConfirmationStatus.PENDING
        if not self.required:
            self.status = PlanConfirmationStatus.NOT_REQUIRED

    @property
    def is_confirmed(self) -> bool:
        return self.status == PlanConfirmationStatus.CONFIRMED

    def mark_confirmed(self, actor: str, note: str = "") -> None:
        self.status = PlanConfirmationStatus.CONFIRMED
        self.confirmed_by = actor
        self.note = note
        self.confirmed_at = datetime.now(timezone.utc)

    def mark_denied(self, actor: str, note: str = "") -> None:
        self.status = PlanConfirmationStatus.DENIED
        self.confirmed_by = actor
        self.note = note
        self.confirmed_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "reasons": list(self.reasons),
            "status": self.status.value,
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "note": self.note,
        }


@dataclass
class ExecutionPlanStep:
    """One scoped sub-task inside an L1 execution plan."""

    agent_name: str
    role: str = "support"
    rationale: str = ""
    expected_output: str = ""
    depends_on: List[str] = field(default_factory=list)
    time_budget_ms: int = 8000
    step_id: str = field(default_factory=lambda: f"step-{uuid.uuid4().hex[:10]}")

    def __post_init__(self) -> None:
        if not self.agent_name or not _NAME_PATTERN.match(self.agent_name):
            raise ValueError(f"Invalid agent_name in execution plan step: {self.agent_name}")
        if self.time_budget_ms <= 0:
            raise ValueError(f"time_budget_ms must be positive, got {self.time_budget_ms}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_name": self.agent_name,
            "role": self.role,
            "rationale": self.rationale,
            "expected_output": self.expected_output,
            "depends_on": list(self.depends_on),
            "time_budget_ms": self.time_budget_ms,
        }


@dataclass
class L1ExecutionPlan:
    """Explicit plan-mode contract used by the L1 runtime orchestrator."""

    query: str
    intent: str
    complexity: float
    primary_agent: str
    selected_agents: List[str] = field(default_factory=list)
    execution_mode: RuntimeExecutionMode = RuntimeExecutionMode.MULTI_STEP_PARALLEL
    conflict_resolution_path: ConflictResolutionPath = ConflictResolutionPath.ARBITRATION_FIRST
    potential_conflicts: List[str] = field(default_factory=list)
    steps: List[ExecutionPlanStep] = field(default_factory=list)
    confirmation_gate: PlanConfirmationGate = field(default_factory=PlanConfirmationGate)
    trace_id: str = ""
    plan_id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.complexity = min(max(float(self.complexity), 0.0), 1.0)

        if self.primary_agent and self.primary_agent not in self.selected_agents:
            self.selected_agents = [self.primary_agent, *self.selected_agents]

        deduped_agents: List[str] = []
        seen = set()
        for agent_name in self.selected_agents:
            if agent_name in seen:
                continue
            seen.add(agent_name)
            deduped_agents.append(agent_name)
        self.selected_agents = deduped_agents

        if not self.steps and self.selected_agents:
            self.steps = [
                ExecutionPlanStep(
                    agent_name=agent_name,
                    role="primary" if agent_name == self.primary_agent else "support",
                    rationale=(
                        "primary intent handler"
                        if agent_name == self.primary_agent
                        else "intent-supporting decomposition"
                    ),
                    expected_output="Domain-scoped evidence and grounded synthesis",
                )
                for agent_name in self.selected_agents
            ]

    @property
    def requires_confirmation(self) -> bool:
        if not self.confirmation_gate.required:
            return False
        return self.confirmation_gate.status != PlanConfirmationStatus.CONFIRMED

    def mark_confirmed(self, actor: str, note: str = "") -> None:
        self.confirmation_gate.mark_confirmed(actor=actor, note=note)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "trace_id": self.trace_id,
            "query": self.query,
            "intent": self.intent,
            "complexity": round(self.complexity, 3),
            "execution_mode": self.execution_mode.value,
            "primary_agent": self.primary_agent,
            "selected_agents": list(self.selected_agents),
            "steps": [step.to_dict() for step in self.steps],
            "potential_conflicts": list(self.potential_conflicts),
            "conflict_resolution_path": self.conflict_resolution_path.value,
            "confirmation_gate": self.confirmation_gate.to_dict(),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MasterStateTransitionRecord:
    """One state transition record for the L0 master-orchestrator."""

    from_state: MasterOrchestratorState
    to_state: MasterOrchestratorState
    timestamp: datetime
    trigger: str = ""
    note: str = ""


@dataclass
class RuntimeHealthSnapshot:
    """Health snapshot used by L0 tiering and wake/sleep decisions."""

    battery_pct: Optional[float] = None
    thermal_state: str = "normal"
    network_state: str = "good"
    charging: bool = False
    sampled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "thermal_state": self.thermal_state,
            "network_state": self.network_state,
            "charging": self.charging,
            "sampled_at": self.sampled_at.isoformat(),
        }


_ALLOWED_MASTER_TRANSITIONS = {
    MasterOrchestratorState.SLEEPING: {
        MasterOrchestratorState.PASSIVE_MONITORING,
        MasterOrchestratorState.ACTIVE_PROCESSING,
        MasterOrchestratorState.EMERGENCY_MUTE,
    },
    MasterOrchestratorState.PASSIVE_MONITORING: {
        MasterOrchestratorState.SLEEPING,
        MasterOrchestratorState.ACTIVE_LISTENING,
        MasterOrchestratorState.DEGRADED_MODE,
        MasterOrchestratorState.EMERGENCY_MUTE,
    },
    MasterOrchestratorState.ACTIVE_LISTENING: {
        MasterOrchestratorState.PASSIVE_MONITORING,
        MasterOrchestratorState.ACTIVE_PROCESSING,
        MasterOrchestratorState.DEGRADED_MODE,
        MasterOrchestratorState.EMERGENCY_MUTE,
    },
    MasterOrchestratorState.ACTIVE_PROCESSING: {
        MasterOrchestratorState.ACTIVE_LISTENING,
        MasterOrchestratorState.DEGRADED_MODE,
        MasterOrchestratorState.EMERGENCY_MUTE,
    },
    MasterOrchestratorState.DEGRADED_MODE: {
        MasterOrchestratorState.ACTIVE_PROCESSING,
        MasterOrchestratorState.ACTIVE_LISTENING,
        MasterOrchestratorState.EMERGENCY_MUTE,
    },
    MasterOrchestratorState.EMERGENCY_MUTE: {
        MasterOrchestratorState.PASSIVE_MONITORING,
    },
}


@dataclass
class MasterOrchestratorStateMachine:
    """L0 lifecycle/resource scaffolding for wake/sleep and tier transitions."""

    state: MasterOrchestratorState = MasterOrchestratorState.SLEEPING
    resource_tier: ResourceTier = ResourceTier.TIER_1_FULL
    history: List[MasterStateTransitionRecord] = field(default_factory=list)
    last_health_snapshot: Optional[RuntimeHealthSnapshot] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def allowed_transitions() -> Dict[str, List[str]]:
        return {
            state.value: sorted(target.value for target in targets)
            for state, targets in _ALLOWED_MASTER_TRANSITIONS.items()
        }

    def can_transition_to(self, new_state: MasterOrchestratorState) -> bool:
        return new_state in _ALLOWED_MASTER_TRANSITIONS[self.state]

    def transition_to(self, new_state: MasterOrchestratorState, trigger: str = "", note: str = "") -> None:
        if not self.can_transition_to(new_state):
            raise MasterLifecycleTransitionError(
                f"Invalid master state transition: {self.state.value} -> {new_state.value}"
            )

        now = datetime.now(timezone.utc)
        self.history.append(
            MasterStateTransitionRecord(
                from_state=self.state,
                to_state=new_state,
                timestamp=now,
                trigger=trigger,
                note=note,
            )
        )
        self.state = new_state
        self.updated_at = now

    @staticmethod
    def _derive_resource_tier(battery_pct: Optional[float], thermal_state: str) -> ResourceTier:
        thermal = (thermal_state or "normal").strip().lower()

        if thermal == "critical" or (battery_pct is not None and battery_pct < 10):
            return ResourceTier.TIER_4_EMERGENCY
        if thermal == "hot" or (battery_pct is not None and battery_pct < 20):
            return ResourceTier.TIER_3_MINIMUM
        if thermal == "warm" or (battery_pct is not None and battery_pct < 35):
            return ResourceTier.TIER_2_CONSERVATIVE
        return ResourceTier.TIER_1_FULL

    def apply_resource_tier(self, tier: ResourceTier, trigger: str = "resource_governor") -> ResourceTier:
        self.resource_tier = tier

        if tier == ResourceTier.TIER_4_EMERGENCY:
            if self.state != MasterOrchestratorState.EMERGENCY_MUTE:
                self.transition_to(
                    MasterOrchestratorState.EMERGENCY_MUTE,
                    trigger=trigger,
                    note="resource tier escalation",
                )
            return tier

        if tier in {ResourceTier.TIER_2_CONSERVATIVE, ResourceTier.TIER_3_MINIMUM}:
            if self.state in {
                MasterOrchestratorState.PASSIVE_MONITORING,
                MasterOrchestratorState.ACTIVE_LISTENING,
                MasterOrchestratorState.ACTIVE_PROCESSING,
            } and self.state != MasterOrchestratorState.DEGRADED_MODE:
                self.transition_to(
                    MasterOrchestratorState.DEGRADED_MODE,
                    trigger=trigger,
                    note=f"tier={tier.value}",
                )
            return tier

        if tier == ResourceTier.TIER_1_FULL and self.state == MasterOrchestratorState.DEGRADED_MODE:
            target = MasterOrchestratorState.ACTIVE_PROCESSING
            if not self.can_transition_to(target):
                target = MasterOrchestratorState.ACTIVE_LISTENING
            self.transition_to(target, trigger=trigger, note="resource recovery")

        return tier

    def update_health(
        self,
        battery_pct: Optional[float],
        thermal_state: str,
        network_state: str = "good",
        charging: bool = False,
        trigger: str = "health_sample",
    ) -> ResourceTier:
        self.last_health_snapshot = RuntimeHealthSnapshot(
            battery_pct=battery_pct,
            thermal_state=thermal_state,
            network_state=network_state,
            charging=charging,
        )
        tier = self._derive_resource_tier(
            battery_pct=self.last_health_snapshot.battery_pct,
            thermal_state=self.last_health_snapshot.thermal_state,
        )
        self.apply_resource_tier(tier=tier, trigger=trigger)
        return tier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "resource_tier": self.resource_tier.value,
            "updated_at": self.updated_at.isoformat(),
            "last_health_snapshot": (
                self.last_health_snapshot.to_dict()
                if self.last_health_snapshot is not None
                else None
            ),
            "history": [
                {
                    "from_state": record.from_state.value,
                    "to_state": record.to_state.value,
                    "timestamp": record.timestamp.isoformat(),
                    "trigger": record.trigger,
                    "note": record.note,
                }
                for record in self.history
            ],
        }


@dataclass(frozen=True)
class ToolParameterSpec:
    """Declarative schema for one tool input parameter."""

    name: str
    param_type: str
    description: str
    required: bool = False
    default: Any = None
    enum_values: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(f"Invalid parameter name: {self.name}")
        if self.param_type not in _JSON_SCHEMA_TYPES:
            raise ValueError(f"Invalid JSON schema type: {self.param_type}")
        if self.enum_values is not None and len(self.enum_values) == 0:
            raise ValueError("enum_values must be non-empty when provided")

    def to_json_schema_property(self) -> Dict[str, Any]:
        prop: Dict[str, Any] = {
            "type": self.param_type,
            "description": self.description,
        }
        if self.default is not None:
            prop["default"] = self.default
        if self.enum_values:
            prop["enum"] = list(self.enum_values)
        return prop


@dataclass(frozen=True)
class ToolContract:
    """Versioned runtime contract for a Cortex tool operation."""

    name: str
    description: str
    parameters: List[ToolParameterSpec] = field(default_factory=list)
    risk_tier: ToolRiskTier = ToolRiskTier.LOW
    side_effect_free: bool = True
    capabilities: List[str] = field(default_factory=list)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(f"Invalid tool name: {self.name}")
        if not _SEMVER_PATTERN.match(self.version):
            raise ValueError(f"Tool contract version must be semver (x.y.z): {self.version}")
        seen = set()
        for spec in self.parameters:
            if spec.name in seen:
                raise ValueError(f"Duplicate parameter name in tool contract: {spec.name}")
            seen.add(spec.name)

    def to_json_schema(self) -> Dict[str, Any]:
        properties = {
            spec.name: spec.to_json_schema_property()
            for spec in self.parameters
        }
        required = [spec.name for spec in self.parameters if spec.required]
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "risk_tier": self.risk_tier.value,
            "side_effect_free": self.side_effect_free,
            "capabilities": list(self.capabilities),
            "input_schema": self.to_json_schema(),
            "output_schema": dict(self.output_schema),
        }


@dataclass
class RuntimeLoopBudget:
    """Deterministic guardrails for one runtime query loop."""

    max_iterations: int = 8
    max_tool_calls_per_window: int = 16
    window_seconds: int = 300
    max_input_tokens: int = 8192
    max_output_tokens: int = 2048
    max_wall_time_seconds: int = 120

    def __post_init__(self) -> None:
        checks = {
            "max_iterations": self.max_iterations,
            "max_tool_calls_per_window": self.max_tool_calls_per_window,
            "window_seconds": self.window_seconds,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_wall_time_seconds": self.max_wall_time_seconds,
        }
        for name, value in checks.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def to_dict(self) -> Dict[str, int]:
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls_per_window": self.max_tool_calls_per_window,
            "window_seconds": self.window_seconds,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_wall_time_seconds": self.max_wall_time_seconds,
        }


@dataclass
class RuntimeRequestEnvelope:
    """Normalized query envelope for runtime loop execution."""

    query: str
    session_id: str = ""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    budget: RuntimeLoopBudget = field(default_factory=RuntimeLoopBudget)
    allowed_tools: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "query": self.query,
            "created_at": self.created_at.isoformat(),
            "budget": self.budget.to_dict(),
            "allowed_tools": self.allowed_tools,
            "metadata": dict(self.metadata),
        }


@dataclass
class RuntimeLoopState:
    """Mutable state snapshot for runtime loop accounting and stop reasons."""

    envelope: RuntimeRequestEnvelope
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    iterations_executed: int = 0
    tool_calls_executed: int = 0
    tool_dispatch_timestamps: List[datetime] = field(default_factory=list)
    stop_reason: Optional[StopReason] = None
    stop_note: str = ""

    def register_iteration(self) -> None:
        self.iterations_executed += 1
        if self.iterations_executed >= self.envelope.budget.max_iterations:
            self.stop_reason = StopReason.MAX_ITERATIONS

    def register_tool_call(self) -> None:
        self.tool_calls_executed += 1
        if self.tool_calls_executed > self.envelope.budget.max_tool_calls_per_window:
            self.stop_reason = StopReason.MAX_TOOL_CALLS

    def _prune_dispatch_window(self, now: datetime) -> None:
        window_start = now - timedelta(seconds=self.envelope.budget.window_seconds)
        self.tool_dispatch_timestamps = [
            ts for ts in self.tool_dispatch_timestamps
            if ts > window_start
        ]

    def try_register_tool_dispatch(self, now: Optional[datetime] = None) -> bool:
        ts = now or datetime.now(timezone.utc)
        self._prune_dispatch_window(ts)

        window_budget = self.envelope.budget.max_tool_calls_per_window
        calls_in_window = len(self.tool_dispatch_timestamps)
        if calls_in_window >= window_budget:
            self.stop_reason = StopReason.RATE_LIMITED
            self.stop_note = (
                f"Tool dispatch rate-limited: {calls_in_window}/{window_budget} "
                f"within {self.envelope.budget.window_seconds}s"
            )
            return False

        self.tool_dispatch_timestamps.append(ts)
        self.register_tool_call()
        return True

    def mark_stop(self, reason: StopReason, note: str = "") -> None:
        self.stop_reason = reason
        self.stop_note = note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.envelope.request_id,
            "session_id": self.envelope.session_id,
            "started_at": self.started_at.isoformat(),
            "budget": self.envelope.budget.to_dict(),
            "iterations_executed": self.iterations_executed,
            "tool_calls_executed": self.tool_calls_executed,
            "tool_window": {
                "window_seconds": self.envelope.budget.window_seconds,
                "calls_in_window": len(self.tool_dispatch_timestamps),
            },
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "stop_note": self.stop_note,
        }


@dataclass(frozen=True)
class DangerousCommandSignal:
    """Classifier output describing a potentially dangerous command pattern."""

    tool_name: str
    command_text: str
    matched_pattern: str
    severity: ToolRiskTier = ToolRiskTier.HIGH


@dataclass(frozen=True)
class PolicyRule:
    """Policy rule that maps tool actions to allow/approval/deny effects."""

    rule_id: str
    tool_name_pattern: str
    effect: PolicyEffect
    reason: str
    command_pattern: Optional[str] = None
    priority: int = 100

    def matches(self, tool_name: str, command_text: str = "") -> bool:
        if not fnmatch(tool_name, self.tool_name_pattern):
            return False
        if self.command_pattern is None:
            return True
        if not command_text:
            return False
        return re.search(self.command_pattern, command_text, flags=re.IGNORECASE) is not None


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of evaluating a tool invocation against policy rules."""

    effect: PolicyEffect
    reason: str
    rule_id: Optional[str] = None
    matched_signals: List[DangerousCommandSignal] = field(default_factory=list)

    @property
    def requires_human_approval(self) -> bool:
        return self.effect == PolicyEffect.REQUIRE_APPROVAL


@dataclass
class PolicyAuditEvent:
    """Immutable audit record for policy decisions."""

    request_id: str
    tool_name: str
    decision: PolicyDecision
    decision_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "effect": self.decision.effect.value,
            "rule_id": self.decision.rule_id,
            "reason": self.decision.reason,
            "decision_source": self.decision_source,
            "signal_count": len(self.decision.matched_signals),
            "metadata": dict(self.metadata),
        }


class PolicyInterface:
    """Minimal policy evaluator for Phase 0 runtime integration."""

    def __init__(self, rules: Optional[List[PolicyRule]] = None):
        self.rules = sorted(rules or [], key=lambda r: r.priority)

    def evaluate(
        self,
        tool_name: str,
        command_text: str = "",
        dangerous_signals: Optional[List[DangerousCommandSignal]] = None,
    ) -> PolicyDecision:
        matched_allow_rule: Optional[PolicyRule] = None

        for rule in self.rules:
            if rule.matches(tool_name, command_text):
                if rule.effect == PolicyEffect.DENY:
                    return PolicyDecision(
                        effect=PolicyEffect.DENY,
                        reason=rule.reason,
                        rule_id=rule.rule_id,
                        matched_signals=list(dangerous_signals or []),
                    )
                if rule.effect == PolicyEffect.REQUIRE_APPROVAL:
                    return PolicyDecision(
                        effect=PolicyEffect.REQUIRE_APPROVAL,
                        reason=rule.reason,
                        rule_id=rule.rule_id,
                        matched_signals=list(dangerous_signals or []),
                    )

                matched_allow_rule = rule

        if dangerous_signals:
            return PolicyDecision(
                effect=PolicyEffect.REQUIRE_APPROVAL,
                reason="Dangerous command signal requires approval",
                rule_id=matched_allow_rule.rule_id if matched_allow_rule else None,
                matched_signals=list(dangerous_signals),
            )

        if matched_allow_rule is not None:
            return PolicyDecision(
                effect=PolicyEffect.ALLOW,
                reason=matched_allow_rule.reason,
                rule_id=matched_allow_rule.rule_id,
            )

        return PolicyDecision(
            effect=PolicyEffect.ALLOW,
            reason="No matching restrictive policy rule",
        )


@dataclass(frozen=True)
class TaskTransitionRecord:
    """One state transition record for task lifecycle auditing."""

    from_state: TaskState
    to_state: TaskState
    timestamp: datetime
    note: str = ""


_ALLOWED_TASK_TRANSITIONS = {
    TaskState.QUEUED: {TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED},
    TaskState.RUNNING: {
        TaskState.WAITING_APPROVAL,
        TaskState.BLOCKED,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.WAITING_APPROVAL: {TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED},
    TaskState.BLOCKED: {TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELLED: set(),
}


@dataclass
class TaskLifecycle:
    """Task lifecycle state model for autonomous/background runtime operations."""

    task_id: str
    state: TaskState = TaskState.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[TaskTransitionRecord] = field(default_factory=list)
    error_message: str = ""

    def can_transition_to(self, new_state: TaskState) -> bool:
        return new_state in _ALLOWED_TASK_TRANSITIONS[self.state]

    def transition_to(self, new_state: TaskState, note: str = "") -> None:
        if not self.can_transition_to(new_state):
            raise LifecycleTransitionError(
                f"Invalid task transition: {self.state.value} -> {new_state.value}"
            )

        record = TaskTransitionRecord(
            from_state=self.state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc),
            note=note,
        )
        self.history.append(record)
        self.state = new_state
        self.updated_at = record.timestamp

    @staticmethod
    def allowed_transitions() -> Dict[str, List[str]]:
        return {
            state.value: sorted(target.value for target in targets)
            for state, targets in _ALLOWED_TASK_TRANSITIONS.items()
        }


def phase0_interface_snapshot() -> Dict[str, Any]:
    """Export a machine-readable summary of Phase 0 runtime interfaces."""

    return {
        "stop_reasons": [s.value for s in StopReason],
        "policy_effects": [e.value for e in PolicyEffect],
        "task_states": [s.value for s in TaskState],
        "task_transitions": TaskLifecycle.allowed_transitions(),
        "default_loop_budget": RuntimeLoopBudget().to_dict(),
        "l1_execution_modes": [m.value for m in RuntimeExecutionMode],
        "l1_plan_confirmation_statuses": [s.value for s in PlanConfirmationStatus],
        "l1_conflict_resolution_paths": [p.value for p in ConflictResolutionPath],
        "master_orchestrator_states": [s.value for s in MasterOrchestratorState],
        "master_state_transitions": MasterOrchestratorStateMachine.allowed_transitions(),
        "resource_tiers": [tier.value for tier in ResourceTier],
    }
