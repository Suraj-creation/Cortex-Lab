"""
Tiered Query Router — T0 through T4 routing logic.
Architecture: Agentic-RAG-Architecture.md §7 (5 Retrieval Tiers)

T0: Cache hit (sub-100ms) — exact/near-exact match in response cache
T1: Single retrieval (1 agent, <2s) — simple factual lookup
T2: Multi-agent (2-4 agents, <5s) — requires coordination
T3: Deep research (5+ agents, <30s) — multi-hop reasoning
T4: Creative synthesis (open-ended, <60s) — novel connections
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TierClassification:
    tier: str
    complexity: float
    intent: str
    entities: list[str]
    topics: list[str]
    sub_queries: list[str]
    confidence: float
    cache_key: str = ""
    recommended_agents: list[str] = field(default_factory=list)
    estimated_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "complexity": self.complexity,
            "intent": self.intent,
            "entities": self.entities,
            "topics": self.topics,
            "sub_queries": self.sub_queries,
            "confidence": self.confidence,
            "cache_key": self.cache_key,
            "recommended_agents": self.recommended_agents,
            "estimated_latency_ms": self.estimated_latency_ms,
        }


class ResponseCache:
    """T0 cache — exact and near-exact query match cache."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, query: str) -> dict[str, Any] | None:
        key = self._make_key(query)
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self._misses += 1
                return None
            age = time.time() - entry.get("_cached_at", 0)
            if age > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.get("response")

    def put(self, query: str, response: dict[str, Any]) -> None:
        key = self._make_key(query)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].get("_cached_at", 0))
                del self._cache[oldest_key]
            self._cache[key] = {"response": response, "_cached_at": time.time()}

    def invalidate(self, pattern: str) -> int:
        pattern_lower = pattern.lower()
        with self._lock:
            to_remove = [k for k, v in self._cache.items() if pattern_lower in str(v).lower()]
            for k in to_remove:
                del self._cache[k]
            return len(to_remove)

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
        }

    @staticmethod
    def _make_key(query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:24]


# Agent recommendations per tier
_TIER_AGENTS = {
    "T0": [],
    "T1": ["timeline"],
    "T2": ["timeline", "causal", "reflection"],
    "T3": ["timeline", "causal", "reflection", "planning", "arbitration"],
    "T4": ["timeline", "causal", "reflection", "planning", "arbitration",
           "cognitive", "emotional", "meta_learning"],
}

_TIER_LATENCY = {
    "T0": 50,
    "T1": 1500,
    "T2": 4000,
    "T3": 15000,
    "T4": 45000,
}


class TierRouter:
    """
    Routes queries to the appropriate processing tier.
    Uses complexity heuristics + optional LLM classification.
    """

    def __init__(self, cache: ResponseCache | None = None):
        self._cache = cache or ResponseCache()

    def classify(self, query: str, context: str = "") -> TierClassification:
        cache_key = ResponseCache._make_key(query)

        cached = self._cache.get(query)
        if cached is not None:
            return TierClassification(
                tier="T0",
                complexity=0.0,
                intent="cached",
                entities=[],
                topics=[],
                sub_queries=[],
                confidence=1.0,
                cache_key=cache_key,
                recommended_agents=[],
                estimated_latency_ms=_TIER_LATENCY["T0"],
            )

        complexity = self._estimate_complexity(query)
        intent = self._classify_intent(query)
        entities = self._extract_entities(query)
        topics = self._extract_topics(query)

        if complexity <= 0.2:
            tier = "T1"
        elif complexity <= 0.45:
            tier = "T2"
        elif complexity <= 0.7:
            tier = "T3"
        else:
            tier = "T4"

        recommended = list(_TIER_AGENTS.get(tier, []))

        if intent == "temporal":
            if "timeline" not in recommended:
                recommended.insert(0, "timeline")
        elif intent == "causal":
            if "causal" not in recommended:
                recommended.insert(0, "causal")
        elif intent == "reflective":
            if "reflection" not in recommended:
                recommended.insert(0, "reflection")

        return TierClassification(
            tier=tier,
            complexity=complexity,
            intent=intent,
            entities=entities,
            topics=topics,
            sub_queries=self._decompose_if_needed(query, complexity),
            confidence=0.8,
            cache_key=cache_key,
            recommended_agents=recommended,
            estimated_latency_ms=_TIER_LATENCY.get(tier, 5000),
        )

    def cache_response(self, query: str, response: dict[str, Any]) -> None:
        self._cache.put(query, response)

    def invalidate(self, pattern: str) -> int:
        return self._cache.invalidate(pattern)

    @property
    def cache_stats(self) -> dict[str, Any]:
        return self._cache.stats()

    def _estimate_complexity(self, query: str) -> float:
        score = 0.0
        word_count = len(query.split())

        if word_count > 20:
            score += 0.2
        if word_count > 40:
            score += 0.2

        complex_markers = [
            "why", "how has", "compare", "what changed", "evolution",
            "relationship between", "pattern", "over time", "trajectory",
            "connection", "impact", "analyze", "synthesize",
        ]
        for marker in complex_markers:
            if marker in query.lower():
                score += 0.15

        multi_hop = ["and also", "and then", "as well as", "in addition",
                     "furthermore", "combined with"]
        for marker in multi_hop:
            if marker in query.lower():
                score += 0.1

        if "?" in query and query.count("?") > 1:
            score += 0.15

        return min(1.0, score)

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["when", "timeline", "history", "chronolog", "date"]):
            return "temporal"
        if any(w in q for w in ["why", "cause", "because", "led to", "result"]):
            return "causal"
        if any(w in q for w in ["reflect", "believe", "think about", "changed my mind", "evolution"]):
            return "reflective"
        if any(w in q for w in ["what is", "define", "explain", "who is"]):
            return "factual"
        if any(w in q for w in ["how to", "how do", "steps", "process", "procedure"]):
            return "procedural"
        if any(w in q for w in ["compare", "versus", "difference", "similar"]):
            return "comparative"
        return "exploratory"

    def _extract_entities(self, query: str) -> list[str]:
        words = query.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 1] if words else []
        return list(set(entities))[:10]

    def _extract_topics(self, query: str) -> list[str]:
        return []

    def _decompose_if_needed(self, query: str, complexity: float) -> list[str]:
        if complexity <= 0.4:
            return [query]
        parts = [s.strip() for s in query.replace("?", "?\n").split("\n") if s.strip()]
        return parts if len(parts) > 1 else [query]
