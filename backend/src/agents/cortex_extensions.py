"""
Built-in Cortex Extensions — Pluggable cross-cutting concerns.
Architecture: Agentic-RAG-Architecture.md §17.10

These implement the Extension base class with specific Cortex behaviors:
- WikiUpdateExtension: triggers wiki agent on new memory ingest
- SafetyGateExtension: blocks dangerous tool calls, PII detection
- ObservabilityExtension: logs all events for tracing/debugging
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.extension_runner import (
    Extension, InputResult, BeforeAgentStartResult,
    ToolCallGateResult, ToolResultRewrite,
)
from src.agents.tool_types import CortexEvent, CortexEventType

logger = logging.getLogger(__name__)


class WikiUpdateExtension(Extension):
    """
    Triggers wiki agent when new memories are ingested.
    Listens for ingest_memory tool results and queues claim extraction.
    """

    @property
    def name(self) -> str:
        return "WikiUpdate"

    async def on_tool_result(
        self, tool_name: str, content: str, is_error: bool
    ) -> ToolResultRewrite:
        if tool_name == "ingest_memory" and not is_error:
            try:
                from src.agents.scheduler import background_scheduler
                import asyncio
                asyncio.create_task(
                    background_scheduler.trigger_event("memory_ingested", {"content": content})
                )
            except Exception as e:
                logger.warning("WikiUpdate trigger failed: %s", e)
        return ToolResultRewrite()

    async def on_agent_end(self, messages: list[dict]) -> None:
        pass


class SafetyGateExtension(Extension):
    """
    Safety gate: blocks dangerous operations and detects PII.
    Runs before every tool call.
    """

    BLOCKED_PATTERNS = [
        "delete_all", "drop_table", "truncate",
        "rm -rf", "format", "destroy",
    ]

    PII_PATTERNS = [
        "ssn", "social security", "credit card",
        "password", "secret key", "api_key",
    ]

    @property
    def name(self) -> str:
        return "SafetyGate"

    async def on_tool_call(
        self, tool_name: str, params: dict[str, Any]
    ) -> ToolCallGateResult:
        params_str = json.dumps(params).lower()

        for pattern in self.BLOCKED_PATTERNS:
            if pattern in params_str:
                return ToolCallGateResult(
                    blocked=True,
                    reason=f"Safety gate: blocked pattern '{pattern}' in tool call {tool_name}",
                )

        for pattern in self.PII_PATTERNS:
            if pattern in params_str:
                logger.warning("PII detected in %s params: %s", tool_name, pattern)

        return ToolCallGateResult(params=params)


class ObservabilityExtension(Extension):
    """
    Logs all lifecycle events for tracing and debugging.
    Stores events in a structured trace log.
    """

    def __init__(self, trace_log: list[dict[str, Any]] | None = None):
        self._trace_log = trace_log if trace_log is not None else []
        self._start_time = time.time()

    @property
    def name(self) -> str:
        return "Observability"

    def _log(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {
            "type": event_type,
            "elapsed_ms": round((time.time() - self._start_time) * 1000),
            **data,
        }
        self._trace_log.append(entry)
        logger.debug("TRACE: %s", json.dumps(entry, default=str))

    async def on_input(self, text: str, options: Any = None) -> InputResult:
        self._log("input", {"text": text[:200]})
        return InputResult()

    async def on_before_agent_start(
        self, text: str, system_prompt: str
    ) -> BeforeAgentStartResult:
        self._log("agent_start", {"text": text[:100], "prompt_len": len(system_prompt)})
        return BeforeAgentStartResult()

    async def on_tool_call(
        self, tool_name: str, params: dict[str, Any]
    ) -> ToolCallGateResult:
        self._log("tool_call", {"tool": tool_name, "params_keys": list(params.keys())})
        return ToolCallGateResult(params=params)

    async def on_tool_result(
        self, tool_name: str, content: str, is_error: bool
    ) -> ToolResultRewrite:
        self._log("tool_result", {
            "tool": tool_name,
            "is_error": is_error,
            "content_len": len(content),
        })
        return ToolResultRewrite()

    async def on_agent_end(self, messages: list[dict]) -> None:
        self._log("agent_end", {"message_count": len(messages)})

    async def on_message_start(self, message: dict) -> None:
        self._log("message_start", {"role": message.get("role", "unknown")})

    async def on_message_end(self, message: dict) -> None:
        self._log("message_end", {"role": message.get("role", "unknown")})

    def get_trace(self) -> list[dict[str, Any]]:
        return list(self._trace_log)


class CRAGQualityExtension(Extension):
    """
    CRAG (Corrective RAG) quality evaluation.
    After retrieval tool results, evaluate quality and trigger re-retrieval if needed.
    """

    @property
    def name(self) -> str:
        return "CRAGQuality"

    async def on_tool_result(
        self, tool_name: str, content: str, is_error: bool
    ) -> ToolResultRewrite:
        if tool_name != "retrieve_memory" or is_error:
            return ToolResultRewrite()

        try:
            results = json.loads(content)
            if isinstance(results, list):
                high_quality = [r for r in results if r.get("score", 0) > 0.6]
                if not high_quality and results:
                    quality_note = (
                        "\n\n[CRAG: Low retrieval quality detected. "
                        f"Best score: {max(r.get('score', 0) for r in results):.2f}. "
                        "Consider reformulating query or trying different search terms.]"
                    )
                    return ToolResultRewrite(content=content + quality_note)
        except (json.JSONDecodeError, TypeError):
            pass

        return ToolResultRewrite()


DEFAULT_EXTENSIONS = [
    SafetyGateExtension(),
    ObservabilityExtension(),
    WikiUpdateExtension(),
    CRAGQualityExtension(),
]
