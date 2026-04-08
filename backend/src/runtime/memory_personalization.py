"""Phase 4 memory personalization helpers.

This module centralizes:
1. Bounded memory extraction profiles with anti-dup + safety guards.
2. Prompt-evidence selection with strict budgets and relevance filters.
3. Ambient-context term construction for optional retrieval augmentation.
4. Lightweight personal-memory quality evaluation metrics.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


_CHATML_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|im_sep|>",
    "<think>",
    "</think>",
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"\+\d{1,3}[\s-]?\d[\d\s-]{8,14}\d")
_NAME_PHRASE_RE = re.compile(
    r"(?:my\s+name\s+is|name\s*[:=-])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
)
_PROJECT_RE = re.compile(r"(?:project\s+name\s*[:=-])\s*([^\n.]{3,120})", re.IGNORECASE)

_SECRET_HINT_RE = re.compile(r"(api[_-]?key|token|secret|password)", re.IGNORECASE)
_SECRET_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")

_CONTEXT_DEPENDENT_TRIGGERS = (
    "that",
    "this",
    "those",
    "these",
    "as i said",
    "as we discussed",
    "what we discussed",
    "earlier",
    "previously",
    "above",
    "last thing",
    "last topic",
    "continue",
    "follow up",
)

_HINT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "were",
    "have",
    "has",
    "had",
    "your",
    "about",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "there",
    "their",
    "them",
    "they",
    "into",
    "over",
    "under",
    "during",
    "after",
    "before",
}


def sanitize_prompt_text(text: str) -> str:
    """Sanitize text before using it for extraction or prompt context."""
    value = text or ""
    for marker in _CHATML_MARKERS:
        value = value.replace(marker, " ")
    value = _CONTROL_RE.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _safe_fact_value(value: str) -> str:
    cleaned = sanitize_prompt_text(value)
    if not cleaned:
        return ""
    if _SECRET_HINT_RE.search(cleaned) or _SECRET_TOKEN_RE.search(cleaned):
        return ""
    return cleaned[:160]


def build_memory_extraction_profile(
    content: str,
    *,
    max_summary_chars: int = 240,
    max_hint_terms: int = 12,
    max_facts: int = 12,
) -> Dict[str, Any]:
    """Build a bounded extraction profile for one memory content string.

    The profile is intentionally compact and safe to store in memory metadata.
    """
    source = sanitize_prompt_text(content)
    summary = source[:max_summary_chars]

    facts: List[Dict[str, str]] = []
    seen = set()

    def add_fact(fact_type: str, raw_value: str) -> None:
        if len(facts) >= max_facts:
            return
        value = _safe_fact_value(raw_value)
        if not value:
            return
        key = f"{fact_type}:{value.lower()}"
        if key in seen:
            return
        seen.add(key)
        facts.append({"type": fact_type, "value": value})

    for email in _EMAIL_RE.findall(source):
        add_fact("email", email)

    for phone in _PHONE_RE.findall(source):
        add_fact("phone", phone)

    for match in _NAME_PHRASE_RE.findall(source):
        add_fact("name", match)

    # Keep one concise education signal if present.
    edu_match = re.search(
        r"((?:B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|Ph\.?D|Bachelor|Master(?:'?s))[^.\n]{0,120})",
        source,
        re.IGNORECASE,
    )
    if edu_match:
        add_fact("education", edu_match.group(1))

    for project in _PROJECT_RE.findall(source):
        add_fact("project", project)

    tokens = re.findall(r"\b[a-z][a-z0-9+.#-]{2,}\b", source.lower())
    filtered = [t for t in tokens if t not in _HINT_STOPWORDS]
    counts = Counter(filtered)
    hint_terms = [term for term, _ in counts.most_common(max_hint_terms)]

    return {
        "version": "v1",
        "summary": summary,
        "hint_terms": hint_terms,
        "facts": facts,
        "source_chars": len(source),
        "truncated": len(source) > max_summary_chars,
    }


def is_context_dependent_query(query: str) -> bool:
    """Heuristic check for queries that depend on immediate conversation context."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(trigger in q for trigger in _CONTEXT_DEPENDENT_TRIGGERS):
        return True
    # Short pronoun-heavy prompts often depend on recent context.
    pronouns = re.findall(r"\b(this|that|it|they|them|he|she)\b", q)
    return len(pronouns) >= 1 and len(q.split()) <= 10


def build_ambient_terms(
    current_turns: Sequence[Dict[str, Any]],
    recent_conversations: Sequence[Dict[str, Any]],
    *,
    max_terms: int = 12,
    max_chars: int = 180,
) -> str:
    """Build compact ambient context terms from live/recent conversation signals."""
    candidates: List[str] = []

    for turn in list(current_turns or [])[-5:]:
        text = sanitize_prompt_text(str(turn.get("text", "")))
        if text:
            candidates.extend(
                t
                for t in re.findall(r"\b[a-z][a-z0-9+.#-]{2,}\b", text.lower())
                if t not in _HINT_STOPWORDS
            )
        speaker = sanitize_prompt_text(str(turn.get("speaker_name", "")))
        if speaker and speaker.lower() not in {"speaker", "user", "unknown"}:
            candidates.append(speaker.lower())

    for conv in list(recent_conversations or [])[:3]:
        for topic in conv.get("topic_labels", []) or []:
            topic_clean = sanitize_prompt_text(str(topic).lower())
            if topic_clean:
                candidates.append(topic_clean)

    unique_terms: List[str] = []
    seen = set()
    for term in candidates:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
        if len(unique_terms) >= max_terms:
            break

    joined = " ".join(unique_terms)
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip()
    return joined


def _looks_like_low_quality_context(content: str) -> bool:
    text = sanitize_prompt_text(content)
    lower = text.lower()
    if len(text) < 40:
        personal_signals = (
            "my name is",
            "email",
            "phone",
            "project name",
            "b.tech",
            "iiit",
        )
        if any(sig in lower for sig in personal_signals):
            return False
        if _EMAIL_RE.search(text) or _PHONE_RE.search(text):
            return False
        return True

    if lower.endswith("?") and len(lower) < 120:
        return True

    words = lower.split()
    if len(words) > 10:
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if trigrams and max(Counter(trigrams).values()) > 3:
            return True

    if re.match(r"^(tell me|what is|what are|who is|where is|how is|list|describe|explain)\b", lower):
        if len(text) < 180 and "[source:" not in lower:
            return True

    return False


def select_prompt_evidence(
    evidence: Sequence[Dict[str, Any]],
    *,
    query_analysis: Optional[Dict[str, Any]],
    is_local_model: bool,
) -> Dict[str, Any]:
    """Select bounded, deduplicated prompt evidence with explicit budgets."""
    qa = query_analysis or {}
    intent = str(qa.get("intent", "") or "").lower()
    complexity = float(qa.get("complexity", 0.5) or 0.5)
    is_synthesis = complexity >= 0.6 or intent in {"reflective", "comparative", "causal"}

    if is_local_model:
        item_budget = 8 if is_synthesis else 5
        item_chars = 700 if is_synthesis else 360
        total_chars = 2600 if is_synthesis else 1400
        min_score = 0.20 if is_synthesis else 0.28
    else:
        item_budget = 12 if is_synthesis else 7
        item_chars = 1500 if is_synthesis else 600
        total_chars = 5200 if is_synthesis else 2600
        min_score = 0.16 if is_synthesis else 0.22

    selected_texts: List[str] = []
    selected_entries: List[Dict[str, Any]] = []
    seen_fingerprints = set()

    metrics = {
        "candidate_count": len(evidence or []),
        "selected_count": 0,
        "pageindex_selected": 0,
        "local_selected": 0,
        "filtered_low_score": 0,
        "filtered_low_quality": 0,
        "filtered_duplicate": 0,
        "item_budget": item_budget,
        "item_char_budget": item_chars,
        "total_char_budget": total_chars,
        "is_synthesis": is_synthesis,
        "is_local_model": is_local_model,
    }

    def add_entry(content: str, score: float, channel: str, memory_type: str) -> None:
        if len(selected_entries) >= item_budget:
            return
        if sum(len(x) for x in selected_texts) >= total_chars:
            return

        cleaned = sanitize_prompt_text(content)
        max_chars = 2000 if "pageindex" in channel else item_chars
        clipped = cleaned[:max_chars]
        fingerprint = re.sub(r"\W+", "", clipped.lower())[:120]
        if fingerprint in seen_fingerprints:
            metrics["filtered_duplicate"] += 1
            return

        if "pageindex" not in channel and _looks_like_low_quality_context(clipped):
            metrics["filtered_low_quality"] += 1
            return

        current_total = sum(len(x) for x in selected_texts)
        remaining = max(total_chars - current_total, 0)
        if remaining <= 0:
            return
        if len(clipped) > remaining:
            clipped = clipped[:remaining].rstrip()
        if len(clipped) < 20:
            return

        seen_fingerprints.add(fingerprint)
        selected_texts.append(clipped)
        selected_entries.append(
            {
                "content": clipped,
                "score": round(float(score), 4),
                "channel": channel,
                "memory_type": memory_type,
            }
        )
        if "pageindex" in channel:
            metrics["pageindex_selected"] += 1
        else:
            metrics["local_selected"] += 1

    candidates = list(evidence or [])

    # Pass 1: prioritize PageIndex chunks for document evidence.
    for item in candidates[:20]:
        channel = str(item.get("channel", "") or "")
        if "pageindex" not in channel:
            continue
        content = str(item.get("content", "") or "")
        score = float(item.get("score", 0.0) or 0.0)
        memory_type = str(item.get("memory_type", "semantic") or "semantic")
        add_entry(content, score, channel, memory_type)
        if metrics["pageindex_selected"] >= 3:
            break

    # Pass 2: fill with local evidence under score + quality gates.
    for item in candidates[:24]:
        if len(selected_entries) >= item_budget:
            break
        if sum(len(x) for x in selected_texts) >= total_chars:
            break

        channel = str(item.get("channel", "") or "")
        if "pageindex" in channel:
            continue

        score = float(item.get("score", 0.0) or 0.0)
        if score < min_score:
            metrics["filtered_low_score"] += 1
            continue

        content = str(item.get("content", "") or "")
        memory_type = str(item.get("memory_type", "semantic") or "semantic")
        add_entry(content, score, channel, memory_type)

    metrics["selected_count"] = len(selected_entries)
    metrics["selected_chars"] = sum(len(x) for x in selected_texts)
    metrics["has_pageindex_evidence"] = metrics["pageindex_selected"] > 0

    return {
        "texts": selected_texts,
        "entries": selected_entries,
        "metrics": metrics,
        "has_pageindex_evidence": metrics["has_pageindex_evidence"],
        "pageindex_evidence_count": metrics["pageindex_selected"],
    }


def _infer_personal_facet(query: str) -> str:
    q = (query or "").lower()
    if any(x in q for x in ("name", "who am i")):
        return "name"
    if any(x in q for x in ("email", "e-mail", "mail")):
        return "email"
    if any(x in q for x in ("phone", "number", "mobile", "contact")):
        return "phone"
    if any(x in q for x in ("university", "college", "degree", "education", "study")):
        return "education"
    if any(x in q for x in ("project", "portfolio", "built", "developed")):
        return "projects"
    return "general"


def _patterns_for_facet(facet: str) -> List[re.Pattern[str]]:
    if facet == "name":
        return [
            re.compile(r"\bmy\s+name\s+is\b", re.IGNORECASE),
            re.compile(r"\bname\s*[:=-]\s*[A-Z][a-z]+\s+[A-Z][a-z]+"),
        ]
    if facet == "email":
        return [re.compile(_EMAIL_RE.pattern, re.IGNORECASE)]
    if facet == "phone":
        return [re.compile(_PHONE_RE.pattern)]
    if facet == "education":
        return [
            re.compile(r"\b(university|college|degree|b\.?tech|m\.?tech|iiit|iit|nit)\b", re.IGNORECASE)
        ]
    if facet == "projects":
        return [re.compile(r"\b(project|built|developed|portfolio)\b", re.IGNORECASE)]
    return [re.compile(r"\b(name|email|phone|education|project)\b", re.IGNORECASE)]


def evaluate_personal_memory_quality(
    query: str,
    evidence_texts: Sequence[str],
    extracted_answer: str = "",
) -> Dict[str, Any]:
    """Compute lightweight precision/recall-style quality metrics for one query."""
    facet = _infer_personal_facet(query)
    patterns = _patterns_for_facet(facet)

    relevant_count = 0
    for text in evidence_texts:
        if any(p.search(text or "") for p in patterns):
            relevant_count += 1

    evidence_count = len(evidence_texts)
    precision_at_k = (relevant_count / evidence_count) if evidence_count else 0.0
    recall_proxy = 1.0 if relevant_count > 0 else 0.0
    extraction_hit = bool(extracted_answer and any(p.search(extracted_answer) for p in patterns))

    return {
        "query": query,
        "facet": facet,
        "precision_at_k": round(precision_at_k, 3),
        "recall_proxy": round(recall_proxy, 3),
        "evidence_count": evidence_count,
        "relevant_count": relevant_count,
        "extraction_hit": extraction_hit,
    }
