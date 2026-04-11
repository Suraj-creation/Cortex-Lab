"""
Claim Store — Atomic fact management with confidence scoring.
Architecture: Agentic-RAG-Architecture.md §5.3 (Claim Memory Plane)

Claims are atomic facts extracted from memories:
  "Sarah works at Google" (confidence: 0.9, sources: [mem_123, mem_456])
  
Claims can be:
- Inserted (new fact)
- Reinforced (same fact from new source → confidence ↑)
- Contradicted (conflicting fact → flag for arbitration)
- Decayed (confidence drops if no reinforcement over time)
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional


@dataclass
class Claim:
    id: str
    text: str
    confidence: float
    source_ids: list[str]
    topic: str
    created_at: str
    updated_at: str
    reinforcement_count: int = 0
    contradiction_ids: list[str] = field(default_factory=list)
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "confidence": self.confidence,
            "source_ids": self.source_ids,
            "topic": self.topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reinforcement_count": self.reinforcement_count,
            "contradiction_ids": self.contradiction_ids,
            "is_active": self.is_active,
        }


class ClaimStore:
    """
    Local-first claim store backed by JSONL + in-memory index.
    Production: could be backed by DuckDB for full-text search.
    """

    _instance: Optional["ClaimStore"] = None

    def __init__(self, data_dir: str = "data/wiki/claims"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._claims: dict[str, Claim] = {}
        self._topic_index: dict[str, list[str]] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def get_instance(cls, data_dir: str = "data/wiki/claims") -> "ClaimStore":
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def upsert(
        self,
        claim: str,
        confidence: float = 0.8,
        source_ids: list[str] | None = None,
        topic: str = "",
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        source_ids = source_ids or []

        with self._lock:
            existing = self._find_similar(claim)
            if existing:
                existing.source_ids = list(set(existing.source_ids + source_ids))
                existing.reinforcement_count += 1
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.updated_at = now
                self._persist_claim(existing)
                return existing.id

            claim_id = f"claim-{uuid.uuid4().hex[:12]}"
            new_claim = Claim(
                id=claim_id,
                text=claim,
                confidence=confidence,
                source_ids=source_ids,
                topic=topic,
                created_at=now,
                updated_at=now,
            )
            self._claims[claim_id] = new_claim
            if topic:
                self._topic_index.setdefault(topic, []).append(claim_id)
            self._persist_claim(new_claim)
            return claim_id

    def search(
        self,
        query: str,
        min_confidence: float = 0.5,
        topic: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query_lower = query.lower()
        with self._lock:
            candidates = list(self._claims.values())

        if topic:
            topic_ids = self._topic_index.get(topic, [])
            candidates = [c for c in candidates if c.id in topic_ids]

        results = []
        for c in candidates:
            if not c.is_active or c.confidence < min_confidence:
                continue
            if query_lower in c.text.lower():
                results.append(c.to_dict())

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:limit]

    def get_claim(self, claim_id: str) -> Claim | None:
        with self._lock:
            return self._claims.get(claim_id)

    def flag_contradiction(self, claim_id: str, contradicting_claim_id: str) -> None:
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim:
                claim.contradiction_ids.append(contradicting_claim_id)
                claim.confidence = max(0.1, claim.confidence - 0.1)
                claim.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist_claim(claim)

    def deactivate(self, claim_id: str) -> None:
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim:
                claim.is_active = False
                claim.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist_claim(claim)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = sum(1 for c in self._claims.values() if c.is_active)
            return {
                "total": len(self._claims),
                "active": active,
                "topics": len(self._topic_index),
            }

    def _find_similar(self, text: str) -> Claim | None:
        text_lower = text.lower().strip()
        for claim in self._claims.values():
            if claim.text.lower().strip() == text_lower:
                return claim
        return None

    def _persist_claim(self, claim: Claim) -> None:
        file_path = self._data_dir / f"{claim.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(claim.to_dict(), f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        for file_path in self._data_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                claim = Claim(
                    id=data["id"],
                    text=data["text"],
                    confidence=data["confidence"],
                    source_ids=data.get("source_ids", []),
                    topic=data.get("topic", ""),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    reinforcement_count=data.get("reinforcement_count", 0),
                    contradiction_ids=data.get("contradiction_ids", []),
                    is_active=data.get("is_active", True),
                )
                self._claims[claim.id] = claim
                if claim.topic:
                    self._topic_index.setdefault(claim.topic, []).append(claim.id)
            except (json.JSONDecodeError, KeyError):
                continue
