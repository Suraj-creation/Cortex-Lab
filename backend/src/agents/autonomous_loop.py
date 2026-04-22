"""
CortexAgentLoop — Universal Autonomous Agent Runtime.
Source: pi-mono/packages/coding-agent/src/core/agent-session.ts

This is the ONE runtime class. All agents (L0, L1, 15 L2 specialists,
background agents) are configurations of this single class.
The LLM IS the planner — it decides which tools to call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.agents.tool_types import (
    CortexEvent, CortexEventType, ToolCallAction, ToolCallDecision,
    ToolDefinition, ToolResult,
)
from src.agents.session_persistence import SessionPersistence, CompactionResult
from src.agents.extension_runner import ExtensionRunner, Extension
from src.runtime.event_bus import RuntimeEventBus, runtime_events
from src.runtime.retry import RetryMatrix
from src.runtime.task_manager import RuntimeTaskManager

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    persist: bool = True
    compact_threshold: float = 0.8
    max_age_hours: int = 0
    auto_compaction: bool = True
    base_dir: str = "data/sessions"


@dataclass
class ScheduleConfig:
    always_on: bool = False
    continuous: bool = False
    on_ingest: bool = False
    interval_min: int = 0
    daily: str = ""
    weekly: str = ""
    cron: str = ""


@dataclass
class RetryConfig:
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 2000


@dataclass
class AgentConfig:
    """
    One per agent type. The ONLY thing that varies between agents.
    The CortexAgentLoop is the same for all.
    """
    agent_id: str
    system_prompt: str
    tools: list[ToolDefinition] = field(default_factory=list)
    extensions: list[Extension] = field(default_factory=list)
    session_config: SessionConfig = field(default_factory=SessionConfig)
    scheduling: ScheduleConfig | None = None
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    resource_tier_minimum: int = 1
    max_turns: int = 50
    max_tool_chain_depth: int = 10
    context_window: int = 32000
    llm_provider: str = "local"


# ── Steering + Follow-Up ──────────────────────────────────────────────────────

class SteeringManager:
    """
    Pi-mono: _steeringMessages[], _followUpMessages[]
    agent-session.ts lines 244-271, 484-505
    
    STEERING: delivered between tool rounds (after tools, before next LLM call)
    FOLLOW-UP: delivered when agent is fully idle
    """

    def __init__(self):
        self._steering: list[str] = []
        self._follow_up: list[str] = []
        self._pending_next_turn: list[dict[str, Any]] = []
        self._mode_steering: str = "one-at-a-time"
        self._mode_follow_up: str = "one-at-a-time"

    def queue_steer(self, text: str) -> None:
        self._steering.append(text)

    def queue_follow_up(self, text: str) -> None:
        self._follow_up.append(text)

    def add_pending_next_turn(self, message: dict[str, Any]) -> None:
        self._pending_next_turn.append(message)

    def drain_steering(self) -> list[dict[str, Any]]:
        if not self._steering:
            return []
        if self._mode_steering == "one-at-a-time":
            text = self._steering.pop(0)
            return [{"role": "user", "content": text}]
        else:
            msgs = [{"role": "user", "content": t} for t in self._steering]
            self._steering.clear()
            return msgs

    def drain_follow_up(self) -> list[dict[str, Any]]:
        if not self._follow_up:
            return []
        if self._mode_follow_up == "one-at-a-time":
            text = self._follow_up.pop(0)
            return [{"role": "user", "content": text}]
        else:
            msgs = [{"role": "user", "content": t} for t in self._follow_up]
            self._follow_up.clear()
            return msgs

    def drain_pending_next_turn(self) -> list[dict[str, Any]]:
        msgs = list(self._pending_next_turn)
        self._pending_next_turn.clear()
        return msgs

    def on_message_start_user(self, text: str) -> None:
        """Pi-mono: remove from display queue when message starts processing."""
        if text in self._steering:
            self._steering.remove(text)
        elif text in self._follow_up:
            self._follow_up.remove(text)

    def get_queue_state(self) -> dict[str, list[str]]:
        return {
            "steering": list(self._steering),
            "followUp": list(self._follow_up),
        }

    @property
    def has_pending(self) -> bool:
        return bool(self._steering or self._follow_up)

    def clear(self) -> None:
        self._steering.clear()
        self._follow_up.clear()
        self._pending_next_turn.clear()


# ── The Core Agent Loop ──────────────────────────────────────────────────────

class CortexAgentLoop:
    """
    Universal agent runtime. Maps 1:1 to pi-mono's AgentSession.
    
    Source references:
    - prompt() flow: agent-session.ts lines 929-1066
    - Event handling: _handleAgentEvent / _processAgentEvent
    - Retry: _handleRetryableError, _retryPromise coordination
    - Compaction: _checkCompaction, _runAutoCompaction
    - Queues: _queueSteer, _queueFollowUp
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_fn: Callable[..., Any] | None = None,
        event_bus: RuntimeEventBus | None = None,
        runtime_task_manager: RuntimeTaskManager | None = None,
        parent_task_id: str | None = None,
    ):
        self.config = config
        self._llm_fn = llm_fn
        self._event_bus = event_bus or runtime_events
        self._extension_runner = ExtensionRunner(config.extensions)
        self._steering = SteeringManager()
        self._retry_matrix = RetryMatrix()
        self._tools: dict[str, ToolDefinition] = {t.name: t for t in config.tools}
        self._listeners: list[Callable[[CortexEvent], Any]] = []

        # State (from agent-session.ts lines 244-271)
        self._is_streaming = False
        self._is_running = False
        self._overflow_recovery_attempted = False
        self._retry_attempt = 0
        self._turn_count = 0
        self._abort_requested = False
        self._trace_id = ""
        self._system_prompt = config.system_prompt
        self._runtime_task_manager = runtime_task_manager
        self._parent_task_id = parent_task_id
        self._active_task_id: str | None = None

        # Session
        self.session = SessionPersistence.create(
            agent_id=config.agent_id,
            base_dir=config.session_config.base_dir,
        ) if config.session_config.persist else None

    # ── Public API ──────────────────────────────────────────────────────────

    async def prompt(self, text: str, images: list[Any] | None = None) -> dict[str, Any]:
        """
        Main entry point. Maps to agent-session.ts prompt() lines 929-1066.
        """
        self._trace_id = f"trace-{uuid.uuid4().hex[:12]}"
        self._retry_attempt = 0
        self._abort_requested = False
        self._overflow_recovery_attempted = False

        # STEP 1: Extension command check
        if text.startswith("/"):
            handled = await self._try_extension_command(text)
            if handled:
                return {"handled": True, "type": "extension_command"}

        # STEP 2: Input hook (emitInput)
        input_result = await self._extension_runner.emit_input(text)
        if input_result.handled:
            return {"handled": True, "type": "input_hook"}
        if input_result.transform:
            text = input_result.text

        # STEP 3: If streaming, queue instead (lines 971-981)
        if self._is_streaming:
            self._steering.queue_steer(text)
            self._emit(CortexEventType.QUEUE_UPDATE, self._steering.get_queue_state())
            return {"queued": True, "type": "steering"}

        # STEP 4: Pre-prompt compaction check (lines 1010-1013)
        await self._check_compaction_pre_prompt()

        # STEP 5: Build messages (lines 1017-1039)
        messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
        messages.extend(self._steering.drain_pending_next_turn())

        # STEP 6: before_agent_start hook (lines 1041-1058)
        ext_result = await self._extension_runner.emit_before_agent_start(
            text, self._system_prompt
        )
        if ext_result.custom_messages:
            messages.extend(ext_result.custom_messages)
        if ext_result.system_prompt_override:
            self._system_prompt = ext_result.system_prompt_override

        # Persist user message
        if self.session:
            self.session.append_message(role="user", content=text)

        runtime_task_id = self._start_runtime_task(text)
        self._active_task_id = runtime_task_id

        # STEP 7: Run the core agent loop
        try:
            self._emit(CortexEventType.AGENT_START, {"agent_id": self.config.agent_id})
            self._is_running = True
            result = await self._run_agent_loop(messages)

            # STEP 8: Post-processing
            final_answer = str(result.get("text", "") or "")
            self._emit(CortexEventType.AGENT_END, {
                "agent_id": self.config.agent_id,
                "turns": self._turn_count,
                "answer": final_answer,
                "session_id": self.session.session_id if self.session else "",
            })
            await self._extension_runner.emit_agent_end(result.get("all_messages", []))

            # STEP 9: Retry check
            await self._handle_retry_if_needed(result)

            # STEP 10: Process follow-up queue
            await self._process_follow_ups()

            if runtime_task_id:
                if self._abort_requested:
                    self._runtime_task_manager.cancel_task(runtime_task_id, reason="agent aborted", propagate=True)
                elif result.get("error"):
                    self._runtime_task_manager.mark_task_failed(runtime_task_id, note=str(result.get("error", ""))[:240])
                else:
                    self._runtime_task_manager.mark_task_completed(runtime_task_id, note="agent prompt completed")

            return result
        except asyncio.CancelledError:
            if runtime_task_id:
                self._runtime_task_manager.cancel_task(runtime_task_id, reason="agent task cancelled", propagate=True)
            raise
        except Exception as e:
            if runtime_task_id:
                self._runtime_task_manager.mark_task_failed(runtime_task_id, note=str(e)[:240])
            raise
        finally:
            self._is_running = False
            self._active_task_id = None

    async def steer(self, text: str) -> None:
        self._steering.queue_steer(text)
        self._emit(CortexEventType.QUEUE_UPDATE, self._steering.get_queue_state())

    async def follow_up(self, text: str) -> None:
        self._steering.queue_follow_up(text)
        self._emit(CortexEventType.QUEUE_UPDATE, self._steering.get_queue_state())

    def abort(self) -> None:
        self._abort_requested = True
        self._is_streaming = False
        if self._runtime_task_manager and self._active_task_id:
            self._runtime_task_manager.cancel_task(self._active_task_id, reason="abort requested", propagate=True)

    def subscribe(self, listener: Callable[[CortexEvent], Any]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def steering_state(self) -> dict[str, list[str]]:
        return self._steering.get_queue_state()

    # ── Core Loop ───────────────────────────────────────────────────────────

    async def _run_agent_loop(self, initial_messages: list[dict]) -> dict[str, Any]:
        """
        The autonomous loop: model → tools → results → model.
        Continues until model stops calling tools or limits are reached.
        """
        self._turn_count = 0
        all_messages = list(initial_messages)
        last_response: dict[str, Any] = {}
        total_tool_calls_executed = 0
        no_progress_rounds = 0
        duplicate_rounds = 0
        previous_round_signatures: list[str] = []
        last_tool_result_by_signature: dict[str, str] = {}

        while self._turn_count < self.config.max_turns and not self._abort_requested:
            self._turn_count += 1
            self._emit(CortexEventType.TURN_START, {
                "turn": self._turn_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Drain steering between turns
            steering_msgs = self._steering.drain_steering()
            if steering_msgs:
                all_messages.extend(steering_msgs)
                for sm in steering_msgs:
                    self._steering.on_message_start_user(sm.get("content", ""))

            # Build LLM context
            context = self._build_context(all_messages)

            # Call LLM
            self._is_streaming = True
            try:
                response = await self._call_llm(context)
            except Exception as e:
                self._is_streaming = False
                logger.error("LLM call failed: %s", e)
                last_response = {"error": str(e), "all_messages": all_messages}
                break
            self._is_streaming = False

            assistant_text = response.get("text", "")
            tool_calls = response.get("tool_calls", [])

            # Persist assistant message
            if self.session and assistant_text:
                self.session.append_message(
                    role="assistant",
                    content=assistant_text,
                    tool_use=tool_calls if tool_calls else None,
                )

            self._emit(CortexEventType.TURN_END, {
                "turn": self._turn_count,
                "has_tool_calls": bool(tool_calls),
            })

            # No tool calls → agent is done
            if not tool_calls:
                last_response = {
                    "text": assistant_text,
                    "turns": self._turn_count,
                    "all_messages": all_messages,
                }
                break

            current_round_signatures = [self._tool_call_signature(tc) for tc in tool_calls]
            if current_round_signatures and current_round_signatures == previous_round_signatures and no_progress_rounds >= 1:
                final_text = await self._force_final_answer_from_context(
                    all_messages,
                    reason="duplicate_tool_round_no_progress",
                )
                last_response = {
                    "text": final_text,
                    "turns": self._turn_count,
                    "all_messages": all_messages,
                    "stop_reason": "duplicate_tool_round_no_progress",
                    "no_progress_rounds": no_progress_rounds,
                }
                break

            remaining_tool_budget = max(self.config.max_tool_chain_depth - total_tool_calls_executed, 0)
            if remaining_tool_budget <= 0:
                final_text = await self._force_final_answer_from_context(
                    all_messages,
                    reason="max_tool_chain_depth_reached",
                )
                last_response = {
                    "text": final_text,
                    "turns": self._turn_count,
                    "all_messages": all_messages,
                    "stop_reason": "max_tool_chain_depth_reached",
                }
                break

            if len(tool_calls) > remaining_tool_budget:
                tool_calls = tool_calls[:remaining_tool_budget]

            # Execute tool calls
            tool_results = await self._execute_tools(tool_calls)
            total_tool_calls_executed += len(tool_calls)

            round_signatures = current_round_signatures
            round_has_progress = False
            for tc, tr in zip(tool_calls, tool_results):
                if tr.is_error:
                    continue
                signature = self._tool_call_signature(tc)
                normalized_content = self._normalize_tool_result_for_progress(tr.content)
                if not normalized_content:
                    continue
                previous_content = last_tool_result_by_signature.get(signature, "")
                if normalized_content != previous_content:
                    round_has_progress = True
                last_tool_result_by_signature[signature] = normalized_content

            if round_signatures and round_signatures == previous_round_signatures:
                duplicate_rounds += 1
            else:
                duplicate_rounds = 0
            previous_round_signatures = round_signatures

            if round_has_progress:
                no_progress_rounds = 0
            else:
                no_progress_rounds += 1

            if no_progress_rounds >= 3 or duplicate_rounds >= 2:
                reason = "tool_round_stagnation"
                final_text = await self._force_final_answer_from_context(all_messages, reason=reason)
                last_response = {
                    "text": final_text,
                    "turns": self._turn_count,
                    "all_messages": all_messages,
                    "tool_results": [r.model_dump() for r in tool_results],
                    "stop_reason": reason,
                    "no_progress_rounds": no_progress_rounds,
                }
                break

            # Add to conversation
            all_messages.append({"role": "assistant", "content": assistant_text, "tool_calls": tool_calls})
            for tr in tool_results:
                msg = {
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                }
                all_messages.append(msg)
                if self.session:
                    self.session.append_message(
                        role="toolResult",
                        content=tr.content,
                        tool_call_id=tr.tool_call_id,
                        is_error=tr.is_error,
                    )

            last_response = {
                "text": assistant_text,
                "tool_results": [r.model_dump() for r in tool_results],
                "turns": self._turn_count,
                "all_messages": all_messages,
            }

        return last_response

    @staticmethod
    def _tool_call_signature(tool_call: dict[str, Any]) -> str:
        tool_name = str(tool_call.get("name", tool_call.get("function", {}).get("name", "")) or "")
        raw_params = tool_call.get("arguments", tool_call.get("function", {}).get("arguments", {}))
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params)
            except Exception:
                raw_params = {"raw": raw_params}
        if not isinstance(raw_params, dict):
            raw_params = {"raw": str(raw_params)}
        try:
            stable_args = json.dumps(raw_params, sort_keys=True, separators=(",", ":"))
        except Exception:
            stable_args = str(raw_params)
        return f"{tool_name}|{stable_args}"

    @staticmethod
    def _normalize_tool_result_for_progress(content: str) -> str:
        normalized = " ".join(str(content or "").split())
        if not normalized:
            return ""
        lowered = normalized.lower()
        empty_markers = {"[]", "{}", "null", "none", "[empty result]", "\"\""}
        if lowered in empty_markers:
            return ""
        return lowered[:2000]

    async def _force_final_answer_from_context(self, messages: list[dict[str, Any]], reason: str) -> str:
        guard_instruction = {
            "role": "user",
            "content": (
                "Tool-call guard triggered ("
                + reason
                + "). Stop calling tools. Provide your best final answer using only the evidence already in this conversation. "
                "If evidence is incomplete, explicitly say what is uncertain."
            ),
        }
        context = self._build_context(messages + [guard_instruction], available_tools=[])
        response = await self._call_llm(context, available_tools=[])
        text = str(response.get("text", "") or "").strip()
        if text:
            return text
        return "I gathered available evidence but could not converge on additional new information. Please refine the query for a more specific answer."

    async def _execute_tools(self, tool_calls: list[dict]) -> list[ToolResult]:
        """Execute tools with lifecycle hooks. Pi-mono: _execute_tools pattern."""
        results: list[ToolResult] = []
        parallel_calls = []
        sequential_calls = []

        for tc in tool_calls:
            tool_name = tc.get("name", tc.get("function", {}).get("name", ""))
            tool_def = self._tools.get(tool_name)
            if tool_def and tool_def.concurrency_safe:
                parallel_calls.append(tc)
            else:
                sequential_calls.append(tc)

        if parallel_calls:
            parallel_results = await asyncio.gather(
                *[self._execute_single_tool(tc) for tc in parallel_calls],
                return_exceptions=True,
            )
            for r in parallel_results:
                if isinstance(r, Exception):
                    results.append(ToolResult(
                        tool_call_id="unknown",
                        content=f"Tool execution error: {r}",
                        is_error=True,
                    ))
                else:
                    results.append(r)

        for tc in sequential_calls:
            result = await self._execute_single_tool(tc)
            results.append(result)

        return results

    async def _execute_single_tool(self, tool_call: dict) -> ToolResult:
        """Execute one tool with before/after hooks. Errors are INFORMATION."""
        tool_call_id = tool_call.get("id", str(uuid.uuid4()))
        tool_name = tool_call.get("name", tool_call.get("function", {}).get("name", ""))
        raw_params = tool_call.get("arguments", tool_call.get("function", {}).get("arguments", {}))
        if isinstance(raw_params, str):
            import json
            try:
                raw_params = json.loads(raw_params)
            except json.JSONDecodeError:
                raw_params = {"raw": raw_params}

        tool_task_id: str | None = None
        if self._runtime_task_manager and self._active_task_id:
            tool_task_id = f"{self._active_task_id}:tool:{tool_name}:{uuid.uuid4().hex[:8]}"
            try:
                self._runtime_task_manager.create_task(
                    task_id=tool_task_id,
                    parent_task_id=self._active_task_id,
                    metadata={
                        "agent_id": self.config.agent_id,
                        "tool_name": tool_name,
                        "trace_id": self._trace_id,
                        "tool_call_id": tool_call_id,
                    },
                )
            except ValueError:
                tool_task_id = None
            if tool_task_id:
                self._runtime_task_manager.mark_task_running(tool_task_id, note="tool execution started")

        def _finish_tool_task(is_error: bool, note: str) -> None:
            if not tool_task_id:
                return
            if is_error:
                self._runtime_task_manager.mark_task_failed(tool_task_id, note=note[:240])
            else:
                self._runtime_task_manager.mark_task_completed(tool_task_id, note=note[:240])

        self._emit(CortexEventType.TOOL_EXECUTION_START, {
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": raw_params,
        })

        tool_def = self._tools.get(tool_name)
        if not tool_def:
            result = ToolResult(
                tool_call_id=tool_call_id,
                content=f"Tool not found: {tool_name}",
                is_error=True,
            )
            self._emit(CortexEventType.TOOL_EXECUTION_END, {
                "toolCallId": tool_call_id,
                "result": result.content,
                "isError": True,
            })
            _finish_tool_task(True, result.content)
            return result

        # Before hook — can BLOCK
        gate = await self._extension_runner.emit_tool_call(tool_name, raw_params)
        if gate.blocked:
            result = ToolResult(
                tool_call_id=tool_call_id,
                content=f"Tool blocked: {gate.reason}",
                is_error=True,
            )
            self._emit(CortexEventType.TOOL_EXECUTION_END, {
                "toolCallId": tool_call_id,
                "result": result.content,
                "isError": True,
            })
            _finish_tool_task(True, result.content)
            return result

        actual_params = gate.params or raw_params
        if isinstance(actual_params, dict):
            metadata = dict(actual_params.get("metadata") or {})
            metadata.setdefault("trace_id", self._trace_id)
            metadata.setdefault("session_id", self.session.session_id if self.session else "")
            metadata.setdefault("agent_id", self.config.agent_id)
            metadata.setdefault("parent_task_id", self._active_task_id or "")
            actual_params["metadata"] = metadata

        # Execute with timeout
        try:
            validated = tool_def.validate_params(actual_params)
            result = await asyncio.wait_for(
                tool_def.execute(tool_call_id, validated.model_dump()),
                timeout=tool_def.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                tool_call_id=tool_call_id,
                content=f"Tool timeout after {tool_def.timeout_seconds}s: {tool_name}",
                is_error=True,
            )
        except Exception as e:
            result = ToolResult(
                tool_call_id=tool_call_id,
                content=f"Tool error: {type(e).__name__}: {e}",
                is_error=True,
            )

        # Truncate (pi-mono: ~50KB)
        if len(result.content) > tool_def.truncate_output:
            result.content = result.content[:tool_def.truncate_output] + "\n... [truncated]"

        # After hook — can REWRITE
        new_content, new_error = await self._extension_runner.emit_tool_result(
            tool_name, result.content, result.is_error,
        )
        result.content = new_content
        result.is_error = new_error

        self._emit(CortexEventType.TOOL_EXECUTION_END, {
            "toolCallId": tool_call_id,
            "result": result.content[:500],
            "isError": result.is_error,
        })
        _finish_tool_task(result.is_error, "tool execution completed")
        return result

    # ── Runtime Task Tracking ──────────────────────────────────────────────

    def _start_runtime_task(self, text: str) -> str | None:
        if self._runtime_task_manager is None:
            return None

        task_id = f"loop-{self.config.agent_id}-{uuid.uuid4().hex[:12]}"
        metadata = {
            "agent_id": self.config.agent_id,
            "trace_id": self._trace_id,
            "query_preview": str(text)[:240],
            "mode": "autonomous_loop",
        }

        try:
            self._runtime_task_manager.create_task(
                task_id=task_id,
                parent_task_id=self._parent_task_id,
                metadata=metadata,
            )
            self._runtime_task_manager.mark_task_running(task_id, note="agent prompt started")
            current = asyncio.current_task()
            if current is not None:
                self._runtime_task_manager.attach_asyncio_task(task_id, current)
            return task_id
        except Exception:
            return None

    # ── LLM Integration ─────────────────────────────────────────────────────

    def _build_context(
        self,
        messages: list[dict],
        available_tools: Optional[list[ToolDefinition]] = None,
    ) -> list[dict]:
        """Build context for LLM call, incorporating session history if available."""
        context: list[dict] = []
        if self._system_prompt:
            context.append({"role": "system", "content": self._system_prompt})

        if self.session:
            session_ctx = self.session.build_session_context()
            context.extend(session_ctx)

        for msg in messages:
            if msg.get("role") != "system":
                context.append(msg)

        tools = list(self._tools.values()) if available_tools is None else list(available_tools)
        tool_descriptions = self._build_tool_descriptions(tools)
        if tool_descriptions and context:
            sys_msg = context[0] if context[0].get("role") == "system" else None
            if sys_msg:
                sys_msg["content"] += f"\n\n{tool_descriptions}"

        return context

    def _build_tool_descriptions(self, tools: list[ToolDefinition]) -> str:
        if not tools:
            return ""
        lines = ["## Available Tools\n"]
        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            if tool.prompt_snippet:
                lines.append(f"{tool.prompt_snippet}")
            lines.append("")
        return "\n".join(lines)

    async def _call_llm(
        self,
        context: list[dict],
        available_tools: Optional[list[ToolDefinition]] = None,
    ) -> dict[str, Any]:
        """Call the LLM. Override _llm_fn for custom providers."""
        if self._llm_fn:
            tools = list(self._tools.values()) if available_tools is None else list(available_tools)
            return await self._llm_fn(context, tools)
        return {"text": "[No LLM provider configured]", "tool_calls": []}

    # ── Compaction ──────────────────────────────────────────────────────────

    async def _check_compaction_pre_prompt(self) -> None:
        if not self.session or not self.config.session_config.auto_compaction:
            return
        msg_count = self.session.message_count
        if msg_count > 100:
            estimated_tokens = msg_count * 150
            threshold = int(self.config.context_window * self.config.session_config.compact_threshold)
            if estimated_tokens > threshold:
                await self._run_compaction("threshold")

    async def _run_compaction(self, reason: str) -> None:
        if not self.session:
            return
        self._emit(CortexEventType.COMPACTION_START, {"reason": reason})

        ext_summary = await self._extension_runner.emit_session_before_compact(
            self.session.get_messages_before_boundary()
        )

        if ext_summary:
            summary = ext_summary
        else:
            messages = self.session.get_messages_before_boundary()
            summary_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')[:200]}"
                for m in messages[-30:]
            )
            summary = f"Session summary ({len(messages)} messages compacted):\n{summary_text}"

        result = self.session.append_compaction(summary)
        await self._extension_runner.emit_session_compact(summary)

        self._emit(CortexEventType.COMPACTION_END, {
            "reason": reason,
            "compacted_count": result.compacted_message_count,
            "aborted": False,
            "willRetry": reason == "overflow",
        })

    # ── Retry ───────────────────────────────────────────────────────────────

    RETRYABLE_SIGNALS = [
        "rate_limit", "429", "500", "502", "503", "529",
        "overloaded", "connection reset", "timeout", "fetch failed",
    ]

    async def _handle_retry_if_needed(self, result: dict) -> None:
        error = result.get("error", "")
        if not error or not self.config.retry_config.enabled:
            return

        if not any(sig in error.lower() for sig in self.RETRYABLE_SIGNALS):
            return

        for attempt in range(1, self.config.retry_config.max_retries + 1):
            self._retry_attempt = attempt
            delay = self.config.retry_config.base_delay_ms * (2 ** (attempt - 1))

            self._emit(CortexEventType.AUTO_RETRY_START, {
                "attempt": attempt,
                "maxAttempts": self.config.retry_config.max_retries,
                "delayMs": delay,
                "errorMessage": error,
            })

            if self.session:
                self.session.strip_last_assistant()

            await asyncio.sleep(delay / 1000)

            if self._abort_requested:
                break

            try:
                await self._run_agent_loop([])
                self._emit(CortexEventType.AUTO_RETRY_END, {
                    "success": True, "attempt": attempt,
                })
                return
            except Exception as e:
                error = str(e)
                continue

        self._emit(CortexEventType.AUTO_RETRY_END, {
            "success": False,
            "attempt": self._retry_attempt,
            "finalError": error,
        })

    # ── Follow-ups ──────────────────────────────────────────────────────────

    async def _process_follow_ups(self) -> None:
        while self._steering.has_pending and not self._abort_requested:
            follow_ups = self._steering.drain_follow_up()
            if not follow_ups:
                break
            for fu in follow_ups:
                text = fu.get("content", "")
                if text:
                    await self.prompt(text)

    # ── Extension Commands ──────────────────────────────────────────────────

    async def _try_extension_command(self, text: str) -> bool:
        return False

    # ── Event Emission ──────────────────────────────────────────────────────

    def _emit(self, event_type: CortexEventType, data: dict[str, Any] | None = None) -> None:
        event = CortexEvent(
            type=event_type,
            data=data or {},
            session_id=self.session.session_id if self.session else "",
            agent_id=self.config.agent_id,
            trace_id=self._trace_id,
        )
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

        payload = {
            **(data or {}),
            "agent_id": self.config.agent_id,
            "event_id": event.event_id,
        }

        self._event_bus.emit(
            plane="EVENT_BUS",
            event_type=event_type.value,
            payload=payload,
            trace_id=self._trace_id,
            session_id=self.session.session_id if self.session else "",
        )

    # ── Stats ───────────────────────────────────────────────────────────────

    def get_session_stats(self) -> dict[str, Any]:
        return {
            "agent_id": self.config.agent_id,
            "session_id": self.session.session_id if self.session else None,
            "message_count": self.session.message_count if self.session else 0,
            "is_streaming": self._is_streaming,
            "is_running": self._is_running,
            "turn_count": self._turn_count,
            "retry_attempt": self._retry_attempt,
            "steering": self._steering.get_queue_state(),
        }
