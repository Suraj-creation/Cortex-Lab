"""
Extension Runner — Pi-Mono Lifecycle Hook System.
Source: pi-mono/packages/coding-agent/src/core/extensions/runner.ts

Provides 10 lifecycle hooks for pluggable cross-cutting concerns.
Extensions can: transform input, inject messages, modify system prompts,
gate/modify tool calls, rewrite tool results, observe streaming, post-process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InputResult:
    handled: bool = False
    transform: bool = False
    text: str = ""
    images: list[Any] = field(default_factory=list)


@dataclass
class BeforeAgentStartResult:
    custom_messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt_override: str | None = None


@dataclass
class ToolCallGateResult:
    blocked: bool = False
    reason: str = ""
    params: dict[str, Any] | None = None


@dataclass
class ToolResultRewrite:
    content: str | None = None
    is_error: bool | None = None
    details: dict[str, Any] | None = None


class Extension(ABC):
    """
    Base extension class. Override only the hooks you need.
    Maps to pi-mono's extension handler functions registered via ExtensionRunner.
    
    Hook order (from runner.ts emit flow):
    1. on_input
    2. on_before_agent_start
    3. on_context
    4. on_before_provider_request
    5. on_tool_call (can BLOCK)
    6. on_tool_result (can REWRITE)
    7. on_message_start / on_message_delta / on_message_end
    8. on_agent_end
    9. on_session_before_compact
    10. on_session_compact
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def on_input(self, text: str, options: Any = None) -> InputResult:
        return InputResult()

    async def on_before_agent_start(
        self, text: str, system_prompt: str
    ) -> BeforeAgentStartResult:
        return BeforeAgentStartResult()

    async def on_context(self, messages: list[dict]) -> list[dict]:
        return messages

    async def on_before_provider_request(self, payload: dict) -> dict:
        return payload

    async def on_tool_call(
        self, tool_name: str, params: dict[str, Any]
    ) -> ToolCallGateResult:
        return ToolCallGateResult(params=params)

    async def on_tool_result(
        self, tool_name: str, result_content: str, is_error: bool
    ) -> ToolResultRewrite:
        return ToolResultRewrite()

    async def on_message_start(self, message: dict) -> None:
        pass

    async def on_message_delta(self, delta: str) -> None:
        pass

    async def on_message_end(self, message: dict) -> None:
        pass

    async def on_agent_end(self, messages: list[dict]) -> None:
        pass

    async def on_session_before_compact(self, entries: list[dict]) -> str | None:
        """Return a custom summary or None to use default compaction."""
        return None

    async def on_session_compact(self, summary: str) -> None:
        pass


class ExtensionRunner:
    """
    Runs extension hooks in order. Maps to pi-mono's ExtensionRunner.
    
    Key behaviors from runner.ts:
    - emit() iterates extensions' handlers
    - session_before_* merges last non-cancel result
    - cancel short-circuits
    - emitToolCall can BLOCK
    - emitToolResult can REWRITE content/details/isError
    """

    def __init__(self, extensions: list[Extension] | None = None):
        self._extensions = list(extensions or [])

    def add(self, ext: Extension) -> None:
        self._extensions.append(ext)

    def remove(self, ext_name: str) -> None:
        self._extensions = [e for e in self._extensions if e.name != ext_name]

    @property
    def extensions(self) -> list[Extension]:
        return list(self._extensions)

    async def emit_input(self, text: str, options: Any = None) -> InputResult:
        for ext in self._extensions:
            result = await ext.on_input(text, options)
            if result.handled:
                return result
            if result.transform:
                return result
        return InputResult()

    async def emit_before_agent_start(
        self, text: str, system_prompt: str
    ) -> BeforeAgentStartResult:
        combined = BeforeAgentStartResult()
        for ext in self._extensions:
            result = await ext.on_before_agent_start(text, system_prompt)
            if result.custom_messages:
                combined.custom_messages.extend(result.custom_messages)
            if result.system_prompt_override is not None:
                combined.system_prompt_override = result.system_prompt_override
        return combined

    async def emit_tool_call(
        self, tool_name: str, params: dict[str, Any]
    ) -> ToolCallGateResult:
        current_params = dict(params)
        for ext in self._extensions:
            result = await ext.on_tool_call(tool_name, current_params)
            if result.blocked:
                return result
            if result.params is not None:
                current_params = result.params
        return ToolCallGateResult(params=current_params)

    async def emit_tool_result(
        self, tool_name: str, content: str, is_error: bool
    ) -> tuple[str, bool]:
        current_content = content
        current_error = is_error
        for ext in self._extensions:
            rewrite = await ext.on_tool_result(tool_name, current_content, current_error)
            if rewrite.content is not None:
                current_content = rewrite.content
            if rewrite.is_error is not None:
                current_error = rewrite.is_error
        return current_content, current_error

    async def emit_agent_end(self, messages: list[dict]) -> None:
        for ext in self._extensions:
            await ext.on_agent_end(messages)

    async def emit_session_before_compact(self, entries: list[dict]) -> str | None:
        last_summary = None
        for ext in self._extensions:
            result = await ext.on_session_before_compact(entries)
            if result is not None:
                last_summary = result
        return last_summary

    async def emit_session_compact(self, summary: str) -> None:
        for ext in self._extensions:
            await ext.on_session_compact(summary)
