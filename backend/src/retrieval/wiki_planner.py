"""Frontier-based wiki/claim retrieval planner.

Provides lightweight planning metadata that can be consumed by orchestrator
routing and provenance envelopes without changing retriever contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models import MemoryQuery


@dataclass
class FrontierCandidate:
    node_type: str
    node_id: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "score": round(max(0.0, min(1.0, float(self.score))), 3),
            "reason": self.reason,
        }


class WikiRetrievalPlanner:
    """Build a deterministic retrieval frontier over wiki/claims/graph."""

    def __init__(self):
        self._wiki = None
        self._claims = None
        self._graph = None

    def _ensure_backends(self) -> None:
        if self._wiki is None:
            from src.wiki.wiki_store import WikiStore

            self._wiki = WikiStore.get_instance()
        if self._claims is None:
            from src.wiki.claim_store import ClaimStore

            self._claims = ClaimStore.get_instance()
        if self._graph is None:
            from src.engine import rag_engine

            self._graph = getattr(rag_engine, "knowledge_graph", None)

    @staticmethod
    def _dedupe_preserve(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def build_frontier(self, query: MemoryQuery, *, max_frontier: int = 24) -> dict[str, Any]:
        self._ensure_backends()

        query_text = str(getattr(query, "raw_query", "") or "")
        entities = list(getattr(query, "entities", []) or [])
        topics = list(getattr(query, "topics", []) or [])

        candidates: list[FrontierCandidate] = []
        wiki_page_ids: list[str] = []
        claim_ids: list[str] = []
        graph_entity_ids: list[str] = []

        try:
            wiki_hits = self._wiki.search(query_text, include_claims=True, limit=max_frontier)
        except Exception:
            wiki_hits = []

        for hit in wiki_hits:
            page_id = str(hit.get("id", "")).strip()
            if not page_id:
                continue
            score = float(hit.get("search_score", 0.0) or 0.0)
            wiki_page_ids.append(page_id)
            candidates.append(
                FrontierCandidate(
                    node_type="wiki_page",
                    node_id=page_id,
                    score=min(1.0, 0.5 + score),
                    reason="wiki lexical/semantic match",
                )
            )
            for claim_id in list(hit.get("claim_ids", []) or []):
                claim_ids.append(str(claim_id))

        try:
            claim_hits = self._claims.search(query_text, min_confidence=0.35, limit=max_frontier)
        except Exception:
            claim_hits = []

        for claim in claim_hits:
            cid = str(claim.get("id", "")).strip()
            if not cid:
                continue
            cscore = float(claim.get("confidence", 0.0) or 0.0)
            claim_ids.append(cid)
            candidates.append(
                FrontierCandidate(
                    node_type="claim",
                    node_id=cid,
                    score=min(1.0, 0.45 + cscore * 0.55),
                    reason="claim text/confidence match",
                )
            )

        if self._graph is not None:
            for entity in entities:
                entity_id = self._graph.find_entity_by_name(str(entity))
                if entity_id:
                    graph_entity_ids.append(str(entity_id))
                    candidates.append(
                        FrontierCandidate(
                            node_type="graph_entity",
                            node_id=str(entity_id),
                            score=0.72,
                            reason="entity direct graph hit",
                        )
                    )

        for topic in topics:
            topic_pages = self._wiki.search_by_topic(str(topic))
            for page in topic_pages[:4]:
                page_id = str(page.get("id", "")).strip()
                if not page_id:
                    continue
                wiki_page_ids.append(page_id)
                candidates.append(
                    FrontierCandidate(
                        node_type="wiki_page",
                        node_id=page_id,
                        score=0.6,
                        reason=f"topic expansion: {topic}",
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        trimmed = candidates[: max(int(max_frontier), 1)]

        wiki_page_ids = self._dedupe_preserve(wiki_page_ids)
        claim_ids = self._dedupe_preserve(claim_ids)
        graph_entity_ids = self._dedupe_preserve(graph_entity_ids)

        return {
            "frontier": [candidate.to_dict() for candidate in trimmed],
            "selected_wiki_pages": wiki_page_ids[:max_frontier],
            "selected_claim_ids": claim_ids[:max_frontier],
            "selected_graph_entities": graph_entity_ids[:max_frontier],
            "seed_entities": entities,
            "seed_topics": topics,
            "expansion_depth": 2,
        }
