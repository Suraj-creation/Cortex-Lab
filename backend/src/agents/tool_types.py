"""
Cortex Agent Tool System — Pydantic-validated tool definitions.
Maps to pi-mono's ToolDefinition interface (tool-definition-wrapper.ts).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, Optional, Type

from pydantic import BaseModel, Field


class PermissionModel(str, Enum):
    AUTO = "auto"
    USER_CONFIRM = "user_confirm"
    PLAN_MODE_ONLY = "plan_mode_only"


class ToolCallAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    DEFER = "defer"


class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False
    details: dict[str, Any] | None = None


class ToolCallDecision(BaseModel):
    action: ToolCallAction = ToolCallAction.ALLOW
    reason: str = ""
    params: dict[str, Any] | None = None


@dataclass
class ToolDefinition:
    """
    Universal tool definition. Maps to pi-mono's ToolDefinition in
    tool-definition-wrapper.ts with Typebox → Pydantic migration.
    """
    name: str
    label: str
    description: str
    parameters_schema: Type[BaseModel]
    execute: Callable[..., Awaitable[ToolResult]]
    permission_model: PermissionModel = PermissionModel.AUTO
    prompt_snippet: str = ""
    prompt_guidelines: str = ""
    concurrency_safe: bool = True
    timeout_seconds: float = 30.0
    truncate_output: int = 50_000

    def validate_params(self, raw_params: dict[str, Any]) -> BaseModel:
        return self.parameters_schema.model_validate(raw_params)


# ── Event types matching pi-mono AgentSessionEvent (agent-session.ts:112-129) ─

class CortexEventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"
    QUEUE_UPDATE = "queue_update"
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"
    AUTO_RETRY_START = "auto_retry_start"
    AUTO_RETRY_END = "auto_retry_end"
    # Cortex-specific
    TIER_SELECTED = "tier_selected"
    RETRIEVAL_CHANNEL_COMPLETE = "retrieval_channel_complete"
    EVIDENCE_READY = "evidence_ready"
    QUALITY_LOOP = "quality_loop"
    WIKI_UPDATE = "wiki_update"
    BELIEF_SHIFT = "belief_shift"
    GAP_SIGNAL = "gap_signal"
    PRESENCE_INITIATIVE = "presence_initiative"


@dataclass
class CortexEvent:
    """Universal event envelope for all agent communication."""
    type: CortexEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""
    agent_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "trace_id": self.trace_id,
        }

    def to_sse(self) -> str:
        import json
        return f"data: {json.dumps(self.to_dict())}\n\n"
