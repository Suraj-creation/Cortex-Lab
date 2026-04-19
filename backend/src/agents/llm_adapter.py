"""LLM adapter for CortexAgentLoop tool-calling rounds.

This adapter bridges the loop runtime with the existing LocalLLM/Gemini providers
without changing the core orchestrator stack.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from src.agents.tool_types import ToolDefinition
from src.llm import LLMProvider


def _normalize_provider(provider: str) -> str:
    requested = str(provider or "local").strip().lower()
    if requested == "gemma_local":
        return "local"
    return requested


def _safe_json_parse(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


class CortexLoopLLMAdapter:
    """Adapter callable used by CortexAgentLoop as llm_fn."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        preferred_provider: str = "local",
    ) -> None:
        self._provider = llm_provider
        self._preferred = _normalize_provider(preferred_provider)

    def _resolve_backend(self):
        local = self._provider.local_llm
        gemini = self._provider.gemini_llm

        local_ready = local is not None and getattr(local, "model", None) is not None
        gemini_ready = gemini is not None and getattr(gemini, "model", None) is not None
        active_provider = str(getattr(self._provider, "provider", "") or "").strip().lower()

        # Honor requested backend only when it is actually available.
        if self._preferred == "gemini" and gemini_ready:
            return gemini
        if self._preferred == "local" and local_ready:
            return local

        # Then honor currently active provider selected at runtime.
        if active_provider == "gemini" and gemini_ready:
            return gemini
        if active_provider in ("local", "gemma_local") and local_ready:
            return local

        # Fallback to any usable backend.
        if gemini_ready:
            return gemini
        if local_ready:
            return local

        # Last resort for downstream error handling paths.
        if gemini is not None:
            return gemini
        if local is not None:
            return local
        return None

    def _tool_descriptors(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        for tool in tools:
            try:
                params_schema = tool.parameters_schema.model_json_schema()
            except Exception:
                params_schema = {"type": "object", "properties": {}}

            descriptors.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params_schema,
                }
            )
        return descriptors

    def _normalize_role(self, raw_role: str) -> str:
        role = str(raw_role or "user").strip().lower()
        if role in {"tool", "toolresult"}:
            return "tool_result"
        if not role:
            return "user"
        return role

    def _summarize_content(self, role: str, content: Any) -> str:
        text = _content_to_text(content).strip()
        if not text:
            return ""

        compact = " ".join(text.split())
        if role == "tool_result":
            lowered = compact.lower()
            if lowered in {"[]", "{}", "null", "none", ""}:
                return "[empty result]"
            if len(compact) > 500:
                return compact[:500] + "..."
            return compact

        if len(compact) > 1200:
            return compact[:1200] + "..."
        return compact

    def _build_query(self, context: list[dict[str, Any]]) -> str:
        recent = context[-20:]
        lines: list[str] = []
        for msg in recent:
            role = self._normalize_role(str(msg.get("role", "user")))
            if role == "system":
                # Keep session compaction notes but avoid resending full system prompts.
                summary = self._summarize_content(role, msg.get("content", ""))
                if summary.startswith("[Session Summary]"):
                    line = f"context_summary: {summary[:700]}"
                    if not lines or lines[-1] != line:
                        lines.append(line)
                continue

            content = self._summarize_content(role, msg.get("content", ""))
            if not content:
                continue
            line = f"{role}: {content}"
            if lines and lines[-1] == line:
                continue
            lines.append(line)
        return "\n".join(lines)[-6000:]

    def _build_function_call_query(self, context: list[dict[str, Any]]) -> str:
        latest_user = ""
        for msg in reversed(context):
            role = self._normalize_role(str(msg.get("role", "user")))
            if role != "user":
                continue
            latest_user = self._summarize_content("user", msg.get("content", ""))
            if latest_user:
                break

        recent_context = self._build_query(context)
        if latest_user and recent_context:
            combined = (
                "Current user request:\n"
                f"{latest_user}\n\n"
                "Recent conversation context (tool_result lines are prior tool outputs):\n"
                f"{recent_context}"
            )
            return combined[-6000:]
        if latest_user:
            return latest_user
        return recent_context

    def _build_generation_prompt(self, context: list[dict[str, Any]]) -> str:
        query = self._build_function_call_query(context)
        return (
            "You are Cortex autonomous agent runtime. "
            "Provide a concise next assistant response based on this conversation.\n\n"
            f"{query}\n\n"
            "Assistant:"
        )

    def _extract_latest_user_query(self, context: list[dict[str, Any]]) -> str:
        for msg in reversed(context):
            role = self._normalize_role(str(msg.get("role", "user")))
            if role != "user":
                continue
            content = self._summarize_content("user", msg.get("content", ""))
            if content:
                return content
        return ""

    def _context_has_tool_results(self, context: list[dict[str, Any]]) -> bool:
        for msg in reversed(context):
            role = self._normalize_role(str(msg.get("role", "")))
            if role == "tool_result":
                return True
        return False

    def _looks_like_information_request(self, text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False

        # Wake-only trigger utterances should not force retrieval.
        if normalized in {"sia", "see ya", "cya", "s i a", "see-ya"}:
            return False

        if "?" in normalized:
            return True

        info_markers = {
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "which",
            "remember",
            "recall",
            "find",
            "show",
            "tell",
            "did",
            "about",
            "history",
            "timeline",
        }

        tokens = set(normalized.split())
        if tokens & info_markers:
            return True

        return len(tokens) >= 6

    def _recover_tool_calls_from_text(
        self,
        text: Any,
        allowed_tool_names: set[str],
    ) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            return []

        raw = text.strip()
        if not raw:
            return []

        parsed = _safe_json_parse(raw)
        if not parsed:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                parsed = _safe_json_parse(raw[start:end + 1])

        if not parsed:
            return []

        return self._extract_tool_calls(parsed, allowed_tool_names)

    def _build_fallback_tool_call(
        self,
        context: list[dict[str, Any]],
        tools: list[ToolDefinition],
    ) -> dict[str, Any] | None:
        if not tools:
            return None

        if self._context_has_tool_results(context):
            return None

        latest_user = self._extract_latest_user_query(context)
        if not self._looks_like_information_request(latest_user):
            return None

        names = {tool.name for tool in tools}
        call_id = f"call-{uuid.uuid4().hex[:12]}"

        if "query_personal_data" in names:
            return {
                "id": call_id,
                "name": "query_personal_data",
                "arguments": {
                    "query": latest_user,
                    "limit_per_source": 8,
                },
            }

        if "retrieve_memory" in names:
            return {
                "id": call_id,
                "name": "retrieve_memory",
                "arguments": {
                    "query": latest_user,
                    "top_k": 8,
                },
            }

        if "search_wiki" in names:
            return {
                "id": call_id,
                "name": "search_wiki",
                "arguments": {
                    "topic": latest_user,
                    "include_claims": True,
                },
            }

        return None

    def _extract_tool_calls(
        self,
        fc_result: dict[str, Any],
        allowed_tool_names: set[str],
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []

        raw_calls = fc_result.get("tool_calls")
        if isinstance(raw_calls, list):
            for raw in raw_calls:
                payload = _safe_json_parse(raw)
                name = (
                    payload.get("name")
                    or _safe_json_parse(payload.get("function", {})).get("name")
                )
                if not isinstance(name, str):
                    continue
                normalized = name.strip()
                if normalized not in allowed_tool_names:
                    continue

                args = payload.get("arguments")
                if args is None:
                    args = _safe_json_parse(payload.get("function", {})).get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {}

                tool_calls.append(
                    {
                        "id": payload.get("id") or f"call-{uuid.uuid4().hex[:12]}",
                        "name": normalized,
                        "arguments": args,
                    }
                )

        if tool_calls:
            return tool_calls

        single_name = (
            fc_result.get("tool")
            or fc_result.get("tool_name")
            or fc_result.get("name")
        )
        if not isinstance(single_name, str):
            return []

        normalized = single_name.strip()
        if not normalized or normalized.lower() in {"none", "null", "no_tool"}:
            return []
        if normalized not in allowed_tool_names:
            return []

        args = fc_result.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        return [
            {
                "id": f"call-{uuid.uuid4().hex[:12]}",
                "name": normalized,
                "arguments": args,
            }
        ]

    async def __call__(
        self,
        context: list[dict[str, Any]],
        tools: list[ToolDefinition],
    ) -> dict[str, Any]:
        backend = self._resolve_backend()
        if backend is None:
            return {"text": "[No LLM backend available]", "tool_calls": []}

        allowed = {t.name for t in tools}
        descriptors = self._tool_descriptors(tools)
        query = self._build_function_call_query(context)

        fc_result: dict[str, Any] = {}
        if hasattr(backend, "call_function"):
            try:
                raw_fc = await asyncio.to_thread(backend.call_function, query, descriptors)
                fc_result = _safe_json_parse(raw_fc)
            except Exception:
                fc_result = {}

        tool_calls = self._extract_tool_calls(fc_result, allowed)
        if not tool_calls and isinstance(fc_result, dict):
            recovered = self._recover_tool_calls_from_text(
                fc_result.get("response") or fc_result.get("text"),
                allowed,
            )
            if recovered:
                tool_calls = recovered

        if not tool_calls:
            fallback_call = self._build_fallback_tool_call(context, tools)
            if fallback_call is not None:
                tool_calls = [fallback_call]

        if tool_calls:
            reasoning = str(fc_result.get("reasoning", "")).strip()
            return {"text": reasoning, "tool_calls": tool_calls}

        answer = str(fc_result.get("response", "")).strip() if isinstance(fc_result, dict) else ""
        if not answer and hasattr(backend, "generate"):
            prompt = self._build_generation_prompt(context)
            try:
                answer = await asyncio.to_thread(backend.generate, prompt, 700, 0.2, 0.95)
            except Exception:
                answer = ""

        return {
            "text": answer.strip() or "I could not complete that request.",
            "tool_calls": [],
        }


def make_cortex_loop_llm_fn(
    llm_provider: LLMProvider,
    preferred_provider: str = "local",
):
    """Create a bound async llm_fn for CortexAgentLoop."""
    return CortexLoopLLMAdapter(llm_provider=llm_provider, preferred_provider=preferred_provider)
