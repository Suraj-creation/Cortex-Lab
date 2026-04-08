"""Phase 1 safe-tool runtime primitives: classifier, permission queue, policy bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Sequence
import re
import uuid

from .contracts import (
    DangerousCommandSignal,
    PolicyAuditEvent,
    PolicyDecision,
    PolicyEffect,
    PolicyInterface,
    PolicyRule,
    StopReason,
    ToolRiskTier,
)


@dataclass(frozen=True)
class DangerousCommandPattern:
    """One dangerous-command regex rule with tool scoping and severity."""

    pattern: str
    reason: str
    severity: ToolRiskTier = ToolRiskTier.HIGH
    tool_name_patterns: Sequence[str] = ("*",)

    def matches_tool(self, tool_name: str) -> bool:
        return any(fnmatch(tool_name, tool_pattern) for tool_pattern in self.tool_name_patterns)


@dataclass(frozen=True)
class _CompiledDangerousPattern:
    pattern: re.Pattern[str]
    rule: DangerousCommandPattern


class DangerousCommandClassifier:
    """Regex-based baseline classifier for potentially dangerous tool commands."""

    def __init__(self, patterns: Sequence[DangerousCommandPattern]):
        self._compiled_patterns: List[_CompiledDangerousPattern] = []
        for rule in patterns:
            self._compiled_patterns.append(
                _CompiledDangerousPattern(
                    pattern=re.compile(rule.pattern, flags=re.IGNORECASE),
                    rule=rule,
                )
            )

    @classmethod
    def default(cls) -> "DangerousCommandClassifier":
        tool_patterns = ("*shell*", "*terminal*", "*powershell*", "*exec*")
        patterns = [
            DangerousCommandPattern(
                pattern=r"\brm\s+-rf\b",
                reason="Recursive force delete",
                severity=ToolRiskTier.CRITICAL,
                tool_name_patterns=tool_patterns,
            ),
            DangerousCommandPattern(
                pattern=r"\bremove-item\b[^\n]*-(recurse|force)",
                reason="PowerShell recursive or force delete",
                severity=ToolRiskTier.CRITICAL,
                tool_name_patterns=tool_patterns,
            ),
            DangerousCommandPattern(
                pattern=r"\bdel\s+/f\s+/s\s+/q\b",
                reason="Windows recursive force delete",
                severity=ToolRiskTier.CRITICAL,
                tool_name_patterns=tool_patterns,
            ),
            DangerousCommandPattern(
                pattern=r"(curl|wget)[^|\n]{0,300}\|\s*(sh|bash|pwsh|powershell)",
                reason="Remote script piped directly into shell",
                severity=ToolRiskTier.CRITICAL,
                tool_name_patterns=tool_patterns,
            ),
            DangerousCommandPattern(
                pattern=r"\b(format|mkfs)\b",
                reason="Disk formatting command",
                severity=ToolRiskTier.CRITICAL,
                tool_name_patterns=tool_patterns,
            ),
            DangerousCommandPattern(
                pattern=r"\b(chmod\s+777|icacls\b.*\beveryone\b)",
                reason="Overly permissive filesystem permission change",
                severity=ToolRiskTier.HIGH,
                tool_name_patterns=tool_patterns,
            ),
        ]
        return cls(patterns)

    def classify(self, tool_name: str, command_text: str) -> List[DangerousCommandSignal]:
        if not command_text or not command_text.strip():
            return []

        normalized = command_text.strip()
        signals: List[DangerousCommandSignal] = []

        for compiled in self._compiled_patterns:
            if not compiled.rule.matches_tool(tool_name):
                continue
            match = compiled.pattern.search(normalized)
            if match:
                signals.append(
                    DangerousCommandSignal(
                        tool_name=tool_name,
                        command_text=normalized,
                        matched_pattern=match.group(0),
                        severity=compiled.rule.severity,
                    )
                )

        return signals


class PermissionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class PermissionRequest:
    """One human-approval request for a risky tool operation."""

    permission_id: str
    request_id: str
    tool_name: str
    command_text: str
    reason: str
    status: PermissionStatus = PermissionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    decided_at: Optional[datetime] = None
    decided_by: str = ""
    decision_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission_id": self.permission_id,
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "command_text": self.command_text,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
            "decision_note": self.decision_note,
            "metadata": dict(self.metadata),
        }


class PermissionQueue:
    """In-memory permission queue with timeout and resolution behavior."""

    def __init__(self, default_timeout_seconds: int = 120):
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        self.default_timeout_seconds = default_timeout_seconds
        self._requests: Dict[str, PermissionRequest] = {}

    def enqueue(
        self,
        request_id: str,
        tool_name: str,
        command_text: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> PermissionRequest:
        timeout = timeout_seconds or self.default_timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        created_at = datetime.now(timezone.utc)
        permission = PermissionRequest(
            permission_id=f"perm-{uuid.uuid4().hex[:12]}",
            request_id=request_id,
            tool_name=tool_name,
            command_text=command_text,
            reason=reason,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=timeout),
            metadata=dict(metadata or {}),
        )
        self._requests[permission.permission_id] = permission
        return permission

    def get(self, permission_id: str) -> Optional[PermissionRequest]:
        return self._requests.get(permission_id)

    def list_pending(self) -> List[PermissionRequest]:
        pending = [r for r in self._requests.values() if r.status == PermissionStatus.PENDING]
        return sorted(pending, key=lambda r: r.created_at)

    def list_by_status(self, status: PermissionStatus) -> List[PermissionRequest]:
        matches = [r for r in self._requests.values() if r.status == status]
        return sorted(matches, key=lambda r: r.created_at)

    def resolve(self, permission_id: str, approve: bool, actor: str, note: str = "") -> PermissionRequest:
        req = self._requests.get(permission_id)
        if req is None:
            raise KeyError(f"Permission request not found: {permission_id}")
        if req.status != PermissionStatus.PENDING:
            raise ValueError(f"Permission request is not pending: {permission_id}")

        req.status = PermissionStatus.APPROVED if approve else PermissionStatus.DENIED
        req.decided_at = datetime.now(timezone.utc)
        req.decided_by = actor
        req.decision_note = note
        return req

    def expire_requests(self, now: Optional[datetime] = None) -> List[PermissionRequest]:
        ts = now or datetime.now(timezone.utc)
        expired: List[PermissionRequest] = []

        for req in self._requests.values():
            if req.status == PermissionStatus.PENDING and ts >= req.expires_at:
                req.status = PermissionStatus.EXPIRED
                req.decided_at = ts
                req.decision_note = "Timed out waiting for human approval"
                expired.append(req)

        return sorted(expired, key=lambda r: r.created_at)


@dataclass
class SafeToolRuntimeResult:
    """Structured result for one safe runtime tool-policy evaluation."""

    decision: PolicyDecision
    dangerous_signals: List[DangerousCommandSignal]
    permission_request: Optional[PermissionRequest]
    stop_reason: Optional[StopReason]
    audit_event: PolicyAuditEvent


class SafeToolRuntime:
    """Phase 1 baseline glue: classifier + policy + permission queue + audit."""

    def __init__(
        self,
        policy: Optional[PolicyInterface] = None,
        classifier: Optional[DangerousCommandClassifier] = None,
        permission_queue: Optional[PermissionQueue] = None,
    ):
        self.policy = policy or self._default_policy()
        self.classifier = classifier or DangerousCommandClassifier.default()
        self.permission_queue = permission_queue or PermissionQueue()
        self._audit_events: List[PolicyAuditEvent] = []

    @staticmethod
    def _default_policy() -> PolicyInterface:
        return PolicyInterface(
            rules=[
                PolicyRule(
                    rule_id="require_approval_delete_memory_default",
                    tool_name_pattern="delete_memory",
                    effect=PolicyEffect.REQUIRE_APPROVAL,
                    reason="delete_memory requires explicit approval",
                    priority=20,
                ),
                PolicyRule(
                    rule_id="allow_default",
                    tool_name_pattern="*",
                    effect=PolicyEffect.ALLOW,
                    reason="default allow",
                    priority=100,
                ),
            ]
        )

    @classmethod
    def default(cls) -> "SafeToolRuntime":
        return cls()

    def evaluate_tool_operation(
        self,
        request_id: str,
        tool_name: str,
        command_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SafeToolRuntimeResult:
        dangerous_signals = self.classifier.classify(tool_name=tool_name, command_text=command_text)
        decision = self.policy.evaluate(
            tool_name=tool_name,
            command_text=command_text,
            dangerous_signals=dangerous_signals,
        )

        permission_request = None
        stop_reason: Optional[StopReason] = None

        if decision.effect == PolicyEffect.REQUIRE_APPROVAL:
            permission_request = self.permission_queue.enqueue(
                request_id=request_id,
                tool_name=tool_name,
                command_text=command_text,
                reason=decision.reason,
                metadata=metadata,
            )
        elif decision.effect == PolicyEffect.DENY:
            stop_reason = StopReason.POLICY_DENIED

        audit_metadata = dict(metadata or {})
        if dangerous_signals:
            audit_metadata["dangerous_patterns"] = [s.matched_pattern for s in dangerous_signals]
        if permission_request:
            audit_metadata["permission_id"] = permission_request.permission_id

        audit_event = PolicyAuditEvent(
            request_id=request_id,
            tool_name=tool_name,
            decision=decision,
            decision_source="safe_tool_runtime",
            metadata=audit_metadata,
        )
        self._audit_events.append(audit_event)

        return SafeToolRuntimeResult(
            decision=decision,
            dangerous_signals=dangerous_signals,
            permission_request=permission_request,
            stop_reason=stop_reason,
            audit_event=audit_event,
        )

    def resolve_permission_request(
        self,
        permission_id: str,
        approve: bool,
        actor: str,
        note: str = "",
    ) -> PermissionRequest:
        request = self.permission_queue.resolve(
            permission_id=permission_id,
            approve=approve,
            actor=actor,
            note=note,
        )

        decision = PolicyDecision(
            effect=PolicyEffect.ALLOW if approve else PolicyEffect.DENY,
            reason="Approved by human operator" if approve else "Denied by human operator",
            rule_id="human_permission_resolution",
        )
        audit_event = PolicyAuditEvent(
            request_id=request.request_id,
            tool_name=request.tool_name,
            decision=decision,
            decision_source="human_approval",
            metadata={
                "permission_id": request.permission_id,
                "approve": approve,
                "actor": actor,
                "note": note,
            },
        )
        self._audit_events.append(audit_event)
        return request

    def expire_permission_requests(self) -> List[PermissionRequest]:
        return self.permission_queue.expire_requests()

    def list_pending_permissions(self) -> List[PermissionRequest]:
        return self.permission_queue.list_pending()

    def list_permissions_by_status(self, status: PermissionStatus) -> List[PermissionRequest]:
        return self.permission_queue.list_by_status(status)

    def record_permission_execution(
        self,
        permission_id: str,
        execution_status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[PolicyAuditEvent]:
        request = self.permission_queue.get(permission_id)
        if request is None:
            return None

        normalized = (execution_status or "").strip().lower()
        if normalized == "failed":
            effect = PolicyEffect.DENY
            reason = "Approved operation execution failed"
        elif normalized == "unsupported":
            effect = PolicyEffect.DENY
            reason = "Approved operation has no execution handler"
        elif normalized == "running":
            effect = PolicyEffect.ALLOW
            reason = "Approved operation execution started"
        else:
            effect = PolicyEffect.ALLOW
            reason = "Approved operation executed"

        event = PolicyAuditEvent(
            request_id=request.request_id,
            tool_name=request.tool_name,
            decision=PolicyDecision(
                effect=effect,
                reason=reason,
                rule_id="approval_executor",
            ),
            decision_source="approval_executor",
            metadata={
                "permission_id": permission_id,
                "execution_status": normalized,
                **dict(metadata or {}),
            },
        )
        self._audit_events.append(event)
        return event

    def list_audit_events(self, limit: int = 200) -> List[PolicyAuditEvent]:
        if limit <= 0:
            return []
        return self._audit_events[-limit:]
