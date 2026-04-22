"""
Deterministic memory -> wiki materialization pipeline.

This module keeps wiki pages and atomic claims in sync with ingested memories
without relying exclusively on autonomous agent runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.wiki.claim_store import ClaimStore
from src.wiki.wiki_store import WikiStore


_BRACKET_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
_IMPORTANCE_SUFFIX_RE = re.compile(r"\s*\(importance\s*:\s*[0-9.]+\)\s*$", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBERED_SPLIT_RE = re.compile(r"(?:^|\n)\s*\d+[).]\s+")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _memory_to_dict(memory: Any) -> dict[str, Any]:
    if memory is None:
        return {}
    if isinstance(memory, dict):
        return dict(memory)

    to_dict = getattr(memory, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass

    return {
        "id": str(getattr(memory, "id", "")),
        "content": str(getattr(memory, "content", "")),
        "source": str(getattr(memory, "source", "")),
        "session_id": str(getattr(memory, "session_id", "")),
        "topics": list(getattr(memory, "topics", []) or []),
        "entities": list(getattr(memory, "entities", []) or []),
        "propositions": list(getattr(memory, "propositions", []) or []),
        "importance": float(getattr(memory, "importance", 0.6) or 0.6),
    }


def normalize_topic(raw_topic: str) -> str:
    topic = re.sub(r"\s+", " ", str(raw_topic or "").strip().lower())
    topic = _NORMALIZE_RE.sub(" ", topic).strip()
    if not topic:
        return "general"
    return topic.replace(" ", "_")[:64]


def _topic_to_title(topic: str) -> str:
    if topic == "general":
        return "General Knowledge"
    return topic.replace("_", " ").title()


def _normalize_claim_key(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def _clean_claim_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    text = text.lstrip("-*\u2022 ").strip()
    text = _BRACKET_PREFIX_RE.sub("", text)
    text = _IMPORTANCE_SUFFIX_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    low = text.lower()
    if low.startswith(("summary:", "participants:", "topics:")):
        return ""
    if text.endswith("?"):
        return ""
    if len(text) < 14 or len(text) > 320:
        return ""

    return text


def extract_claim_candidates(
    content: str,
    propositions: list[str] | None = None,
    max_claims: int = 10,
) -> list[str]:
    """Extract deterministic claim candidates from memory text/propositions."""

    raw_candidates: list[str]
    props = [str(p).strip() for p in list(propositions or []) if str(p).strip()]

    if props:
        raw_candidates = props
    else:
        text = str(content or "").strip()
        if not text:
            return []

        numbered = _NUMBERED_SPLIT_RE.split(text)
        if len(numbered) > 2:
            raw_candidates = [item.strip() for item in numbered if item.strip()]
        else:
            raw_candidates = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]

    deduped: list[str] = []
    seen: set[str] = set()

    for candidate in raw_candidates:
        cleaned = _clean_claim_text(candidate)
        if not cleaned:
            continue

        key = _normalize_claim_key(cleaned)
        if key in seen:
            continue

        seen.add(key)
        deduped.append(cleaned)
        if len(deduped) >= max_claims:
            break

    return deduped


def _estimate_claim_confidence(memory: dict[str, Any], claim: str) -> float:
    try:
        importance = float(memory.get("importance", 0.6) or 0.6)
    except Exception:
        importance = 0.6

    importance = min(max(importance, 0.0), 1.0)
    confidence = 0.6 + (importance * 0.3)

    source = str(memory.get("source", "") or "").lower()
    if source == "voice":
        confidence -= 0.05

    low = claim.lower()
    if any(token in low for token in ("maybe", "might", "probably", "possibly")):
        confidence -= 0.1

    return round(min(max(confidence, 0.35), 0.95), 3)


def _topic_candidates(memory: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    for raw in list(memory.get("topics", []) or []):
        topic = normalize_topic(str(raw))
        if topic and topic not in candidates:
            candidates.append(topic)

    for raw in list(memory.get("entities", []) or []):
        topic = normalize_topic(str(raw))
        if topic and topic not in candidates:
            candidates.append(topic)

    if not candidates:
        candidates.append("general")

    return candidates


def _pick_topic(memory: dict[str, Any], claim: str) -> str:
    del claim
    return _topic_candidates(memory)[0]


def _initial_page_content(title: str) -> str:
    return (
        f"# {title}\n\n"
        "## Summary\n"
        "Auto-generated topic page synthesized from ingested memories.\n\n"
        "## Stable Facts\n\n"
        "## Open Questions\n\n"
        "## Provenance\n"
        "- Created automatically by the deterministic wiki materializer.\n"
    )


def _find_or_create_page(
    store: WikiStore,
    *,
    topic: str,
) -> tuple[str, bool]:
    by_topic = store.search_by_topic(topic)
    if by_topic:
        return str(by_topic[0]["id"]), False

    title = _topic_to_title(topic)
    page_id = store.create_page(
        title=title,
        content=_initial_page_content(title),
        topics=[topic],
    )
    return page_id, True


def _build_claim_line(
    *,
    claim_id: str,
    claim_text: str,
    confidence: float,
    memory_id: str,
    source: str,
    session_id: str,
) -> str:
    meta = [f"confidence {confidence:.2f}"]
    if memory_id:
        meta.append(f"source {memory_id}")
    if source:
        meta.append(f"origin {source}")
    if session_id:
        meta.append(f"session {session_id}")
    return f"- [{claim_id}] {claim_text} ({'; '.join(meta)})"


def _append_log(wiki_data_dir: str, payload: dict[str, Any]) -> None:
    logs_dir = Path(wiki_data_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "materializer.log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def materialize_memory_into_wiki(
    memory: dict[str, Any] | Any,
    *,
    wiki_data_dir: str = "data/wiki",
    claims_data_dir: str = "data/wiki/claims",
    max_claims: int = 10,
) -> dict[str, Any]:
    """Create/update claims and wiki pages for a single ingested memory."""

    memory_data = _memory_to_dict(memory)
    memory_id = str(memory_data.get("id", "") or "")
    content = str(memory_data.get("content", "") or "")
    source = str(memory_data.get("source", "") or "")
    session_id = str(memory_data.get("session_id", "") or "")

    summary = {
        "memory_id": memory_id,
        "claims_extracted": 0,
        "claims_upserted": 0,
        "claims_linked": 0,
        "pages_created": 0,
        "pages_touched": [],
        "contradictions_flagged": 0,
        "source": source,
    }

    if not content.strip():
        return summary

    claim_store = ClaimStore.get_instance(data_dir=claims_data_dir)
    wiki_store = WikiStore.get_instance(data_dir=wiki_data_dir)

    claims = extract_claim_candidates(
        content,
        propositions=list(memory_data.get("propositions", []) or []),
        max_claims=max_claims,
    )
    summary["claims_extracted"] = len(claims)

    touched_pages: set[str] = set()

    for claim_text in claims:
        topic = _pick_topic(memory_data, claim_text)
        confidence = _estimate_claim_confidence(memory_data, claim_text)

        claim_id = claim_store.upsert(
            claim=claim_text,
            confidence=confidence,
            source_ids=[memory_id] if memory_id else [],
            topic=topic,
        )
        summary["claims_upserted"] += 1

        claim_obj = claim_store.get_claim(claim_id)
        if claim_obj and claim_obj.contradiction_ids:
            summary["contradictions_flagged"] += 1

        page_id, created = _find_or_create_page(wiki_store, topic=topic)
        if created:
            summary["pages_created"] += 1

        page = wiki_store.get_page(page_id)
        if page and claim_id not in page.claim_ids:
            line = _build_claim_line(
                claim_id=claim_id,
                claim_text=claim_text,
                confidence=confidence,
                memory_id=memory_id,
                source=source,
                session_id=session_id,
            )
            wiki_store.patch_page(
                page_id=page_id,
                section="Stable Facts",
                content=line,
                operation="append",
            )
            wiki_store.link_claim(page_id, claim_id)
            summary["claims_linked"] += 1

        touched_pages.add(page_id)

    summary["pages_touched"] = sorted(touched_pages)

    _append_log(
        wiki_data_dir,
        {
            "ts": memory_data.get("timestamp", ""),
            **summary,
        },
    )

    return summary


def materialize_memories_into_wiki(
    memories: Iterable[dict[str, Any] | Any],
    *,
    wiki_data_dir: str = "data/wiki",
    claims_data_dir: str = "data/wiki/claims",
    max_memories: int | None = None,
    max_claims_per_memory: int = 10,
) -> dict[str, Any]:
    """Backfill wiki state from a batch of stored memories."""

    memory_list = list(memories or [])
    if max_memories is not None:
        memory_list = memory_list[: max(0, int(max_memories))]

    # Stored memories are usually newest-first; process oldest-first for stable pages.
    ordered = list(reversed(memory_list))

    totals = {
        "processed": 0,
        "claims_extracted": 0,
        "claims_upserted": 0,
        "claims_linked": 0,
        "pages_created": 0,
        "contradictions_flagged": 0,
    }
    page_ids: set[str] = set()

    for memory in ordered:
        summary = materialize_memory_into_wiki(
            memory,
            wiki_data_dir=wiki_data_dir,
            claims_data_dir=claims_data_dir,
            max_claims=max_claims_per_memory,
        )
        totals["processed"] += 1
        totals["claims_extracted"] += int(summary.get("claims_extracted", 0) or 0)
        totals["claims_upserted"] += int(summary.get("claims_upserted", 0) or 0)
        totals["claims_linked"] += int(summary.get("claims_linked", 0) or 0)
        totals["pages_created"] += int(summary.get("pages_created", 0) or 0)
        totals["contradictions_flagged"] += int(summary.get("contradictions_flagged", 0) or 0)
        page_ids.update(str(pid) for pid in list(summary.get("pages_touched", []) or []) if pid)

    return {
        **totals,
        "pages_touched": sorted(page_ids),
    }
