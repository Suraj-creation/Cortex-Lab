"""Permission gate enforcing schema/scope/resource/privacy/user/audit order."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_PERMISSION_STAGES = (
    "schema_validation",
    "agent_scope",
    "resource_governor",
    "privacy_policy",
    "user_permission",
    "audit_log",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PermissionCheckEntry:
    stage: str
    passed: bool
    reason: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class PermissionCheckResult:
    allowed: bool
    failed_stage: Optional[str] = None
    reason: str = ""
    checks: List[PermissionCheckEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "failed_stage": self.failed_stage,
            "reason": self.reason,
            "checks": [entry.to_dict() for entry in self.checks],
        }


class PermissionGate:
    """Simple ordered permission gate for runtime action checks."""

    @staticmethod
    def stages() -> List[str]:
        return list(_PERMISSION_STAGES)

    def evaluate(
        self,
        *,
        action: str,
        metadata: Dict[str, Any] | None = None,
        require_user_permission: bool = False,
    ) -> PermissionCheckResult:
        data = dict(metadata or {})
        checks: List[PermissionCheckEntry] = []

        forced_failure = str(data.get("fail_stage", "")).strip().lower()
        action_name = action or "unknown_action"

        for stage in _PERMISSION_STAGES:
            stage_failed = forced_failure == stage
            if stage == "user_permission" and not require_user_permission:
                checks.append(PermissionCheckEntry(stage=stage, passed=True, reason="not_required"))
                continue

            if stage_failed:
                checks.append(PermissionCheckEntry(stage=stage, passed=False, reason=f"forced_failure:{action_name}"))
                return PermissionCheckResult(
                    allowed=False,
                    failed_stage=stage,
                    reason=f"Permission denied at {stage}",
                    checks=checks,
                )

            checks.append(PermissionCheckEntry(stage=stage, passed=True))

        return PermissionCheckResult(allowed=True, checks=checks)

    def enforce(
        self,
        *,
        action: str,
        metadata: Dict[str, Any] | None = None,
        require_user_permission: bool = False,
    ) -> PermissionCheckResult:
        result = self.evaluate(
            action=action,
            metadata=metadata,
            require_user_permission=require_user_permission,
        )
        if not result.allowed:
            raise PermissionError(result.reason)
        return result


permission_gate = PermissionGate()
