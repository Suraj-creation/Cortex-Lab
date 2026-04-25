"""Shared heuristics for client-driven ambient companion sessions."""

from __future__ import annotations

import re
import time
from typing import Any

from .speech_cleanup import clean_transcript


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

GREETING_PREFIXES = ("hey", "hi", "hello", "yo")

KNOWN_HALLUCINATED_TRANSCRIPTS = {
    "quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
    "i m not sure if i m going to be able to make it to the meeting",
    "im not sure if im going to be able to make it to the meeting",
    "i'm not sure if i'm going to be able to make it to the meeting",
}

LOW_SIGNAL_TRANSCRIPTS = {
    "she had a",
    "she had",
}


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


def normalize_spoken_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def extract_assistant_trigger(
    text: str,
    *,
    assistant_aliases: list[str],
) -> tuple[bool, str, str]:
    raw = str(text or "").strip()
    if not raw:
        return False, "", ""

    for alias in assistant_aliases:
        pattern = rf"^\s*(?:(?:{'|'.join(GREETING_PREFIXES)})\s+)?{re.escape(alias)}\b[\s,.:;!?-]*(.*)$"
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        query_after_trigger = str(match.group(1) or "").strip()
        return True, query_after_trigger, alias

    match = re.search(
        r"\b(?:see[\s\-]*ya|cya|s[\s\.\-]*i[\s\.\-]*a)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if not match:
        return False, raw, ""

    query_after_trigger = str(raw[match.end():] or "").strip(" \t,:;-")
    return True, query_after_trigger, "retrieve_trigger"


def sanitize_client_transcript(
    text: str,
    *,
    previous_text: str = "",
    previous_turn_ts: float = 0.0,
    now_ts: float | None = None,
    confidence: float = 0.0,
    estimated_duration_s: float = 0.0,
) -> str:
    cleaned = clean_transcript(str(text or ""), confidence=0.0) or ""
    if not cleaned:
        return ""

    normalized = normalize_spoken_text(cleaned)
    if not normalized:
        return ""

    if normalized in KNOWN_HALLUCINATED_TRANSCRIPTS or normalized in LOW_SIGNAL_TRANSCRIPTS:
        return ""

    if _looks_like_looped_hallucination(normalized):
        return ""

    duration_s = max(float(estimated_duration_s or 0.0), 0.0)
    if confidence > 0 and confidence < 0.33 and duration_s >= 1.0:
        return ""

    word_count = len(normalized.split())
    if duration_s >= 2.0 and word_count <= 2:
        return ""

    now_value = float(now_ts if now_ts is not None else time.time())
    previous_normalized = normalize_spoken_text(previous_text)
    if (
        previous_normalized
        and previous_normalized == normalized
        and len(normalized.split()) >= 4
        and previous_turn_ts > 0
        and now_value - previous_turn_ts <= 18
    ):
        return ""

    return cleaned


def _looks_like_looped_hallucination(normalized: str) -> bool:
    words = [part for part in normalized.split() if part]
    if len(words) < 8:
        return False

    if len(set(words)) <= max(2, len(words) // 5):
        return True

    for size in range(4, min(9, len(words) // 2 + 1)):
        segment = words[:size]
        segment_text = " ".join(segment)
        if not segment_text:
            continue
        if " ".join(words).count(segment_text) >= 3:
            return True

    sentences = [part.strip() for part in re.split(r"[.!?]+", normalized) if part.strip()]
    if len(sentences) >= 2 and len(set(sentences)) == 1:
        return True

    return False


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

    direct_address, query_text, assistant_alias = extract_assistant_trigger(
        text,
        assistant_aliases=assistant_aliases,
    )

    if not direct_address:
        assistant_alias = ""
        query_text = normalized

    if assistant_alias:
        query_text = re.sub(rf"\b{re.escape(assistant_alias)}\b", " ", query_text, flags=re.IGNORECASE)
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
