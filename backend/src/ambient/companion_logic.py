"""Shared heuristics for client-driven ambient companion sessions."""

from __future__ import annotations

import re
import time
from typing import Any


DEFAULT_ASSISTANT_ALIASES = (
    "eva",
    "ava",
    "eve",
    "cortex",
    "assistant",
)

RETRIEVAL_PATTERNS = (
    "remember",
    "what did i say",
    "what do you know",
    "find",
    "search",
    "look up",
    "wiki",
    "knowledge graph",
    "memory",
    "memories",
    "retrieve",
    "recall",
)

ACTION_PATTERNS = (
    "todo",
    "to-do",
    "need to",
    "must",
    "deadline",
    "by ",
    "follow up",
    "remind me",
)

DECISION_PATTERNS = (
    "i decided",
    "we decided",
    "the plan is",
    "next step",
    "i will",
    "we will",
)

PREFERENCE_PATTERNS = (
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "favorite",
)

TECHNICAL_PATTERNS = (
    "agent",
    "api",
    "architecture",
    "backend",
    "frontend",
    "graph",
    "memory",
    "rag",
    "retrieval",
    "session",
    "prompt",
    "pipeline",
    "mobile",
)


def build_assistant_aliases(
    assistant_name: str = "",
    raw_aliases: list[str] | str | None = None,
) -> list[str]:
    aliases: list[str] = []

    if assistant_name:
        aliases.append(str(assistant_name).strip().lower())

    if isinstance(raw_aliases, str):
        aliases.extend(part.strip().lower() for part in raw_aliases.split(","))
    elif isinstance(raw_aliases, list):
        aliases.extend(str(part).strip().lower() for part in raw_aliases)

    aliases.extend(DEFAULT_ASSISTANT_ALIASES)

    seen: set[str] = set()
    normalized: list[str] = []
    for alias in aliases:
        cleaned = re.sub(r"[^a-z0-9]+", " ", alias or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def analyze_client_turn(
    text: str,
    *,
    assistant_aliases: list[str],
    engaged_until: float = 0.0,
    now_ts: float | None = None,
) -> dict[str, Any]:
    now_value = float(now_ts if now_ts is not None else time.time())
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return {
            "direct_address": False,
            "retrieval_intent": False,
            "reply_expected": False,
            "followup_active": False,
            "assistant_alias": "",
            "query_text": "",
        }

    assistant_alias = ""
    direct_address = False
    for alias in assistant_aliases:
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, normalized):
            assistant_alias = alias
            direct_address = True
            break

    query_text = normalized
    if assistant_alias:
        query_text = re.sub(rf"\b{re.escape(assistant_alias)}\b", " ", query_text)
        query_text = re.sub(r"^(hey|hi|hello)\s+", "", query_text).strip(" ,.:;!?-")

    retrieval_intent = any(pattern in normalized for pattern in RETRIEVAL_PATTERNS)
    question_like = "?" in text or bool(
        re.search(r"\b(what|when|where|why|how|who|can|could|would|should|tell|show)\b", normalized)
    )
    followup_active = engaged_until > now_value
    reply_expected = bool(direct_address or followup_active)
    if retrieval_intent and (direct_address or followup_active):
        reply_expected = True
    elif question_like and direct_address:
        reply_expected = True

    return {
        "direct_address": direct_address,
        "retrieval_intent": retrieval_intent,
        "reply_expected": reply_expected,
        "followup_active": followup_active,
        "assistant_alias": assistant_alias,
        "query_text": query_text,
    }


def build_retention_trace(
    text: str,
    *,
    session_id: str = "",
    direct_address: bool = False,
    retrieval_intent: bool = False,
    reply_expected: bool = False,
    platform: str = "web",
    source: str = "client_companion",
) -> dict[str, Any]:
    raw = str(text or "").strip()
    lowered = raw.lower()
    words = raw.split()
    word_count = len(words)

    tags: list[str] = []
    score = 0.0

    score += min(word_count / 18.0, 0.45)
    if any(pattern in lowered for pattern in ACTION_PATTERNS):
        tags.append("action_item")
        score += 0.30
    if any(pattern in lowered for pattern in DECISION_PATTERNS):
        tags.append("decision")
        score += 0.24
    if any(pattern in lowered for pattern in PREFERENCE_PATTERNS):
        tags.append("preference")
        score += 0.18
    if any(pattern in lowered for pattern in TECHNICAL_PATTERNS):
        tags.append("technical")
        score += 0.14
    if re.search(r"\b(i|my|we|our)\b", lowered):
        tags.append("personal_context")
        score += 0.10
    if any(char.isdigit() for char in raw):
        tags.append("specific_detail")
        score += 0.12
    if "?" in raw:
        tags.append("question")
        score += 0.10
    if direct_address:
        tags.append("companion_invocation")
        score += 0.18
    if retrieval_intent:
        tags.append("retrieval_query")
        score += 0.22
    if reply_expected:
        tags.append("spoken_dialogue")
        score += 0.08

    if word_count <= 2:
        score -= 0.15

    memory_decision = "session_only"
    if score >= 0.85 or {"action_item", "decision"} & set(tags):
        memory_decision = "priority"
    elif score >= 0.48 or {"technical", "preference", "specific_detail", "retrieval_query"} & set(tags):
        memory_decision = "structured"

    reason = {
        "session_only": "kept_for_session_context",
        "structured": "retained_for_structured_memory",
        "priority": "retained_as_high_priority_memory",
    }[memory_decision]

    if not tags:
        tags.append("conversation")

    return {
        "decision": "keep" if raw else "discard",
        "memory_decision": memory_decision if raw else "discard",
        "archive_policy": "session",
        "reason": reason if raw else "empty_transcript",
        "score": round(max(score, 0.0), 3),
        "tags": list(dict.fromkeys(tags)),
        "direct_address": bool(direct_address),
        "retrieval_intent": bool(retrieval_intent),
        "reply_expected": bool(reply_expected),
        "source": source,
        "platform": platform,
        "session_id": session_id,
    }
