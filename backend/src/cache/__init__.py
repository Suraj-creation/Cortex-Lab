"""
Cortex Lab — Multi-Level Cache
3-level caching: exact match → semantic similarity → miss.
"""

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time


class MultiLevelCache:
    """
    3-level response cache:
      Level 1: Exact string match (hash-based)
      Level 2: Semantic similarity (embedding-based, if embeddings available)
      Level 3: Miss
    """

    def __init__(self, embedding_model=None):
        self._embedding_model = embedding_model
        self._exact_cache: Dict[str, Dict] = {}
        self._semantic_cache: List[Dict] = []
        self._hits = {"exact": 0, "semantic": 0, "miss": 0}
        self._max_exact = 200
        self._max_semantic = 50
        self._semantic_threshold = 0.92

    def _hash_key(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Tuple[Optional[Dict], str]:
        """Look up a cached result. Returns (result, cache_level)."""
        key = self._hash_key(query)

        # Level 1: exact match
        if key in self._exact_cache:
            self._hits["exact"] += 1
            return self._exact_cache[key], "exact"

        # Level 2: semantic similarity (only if embedding model available)
        if self._embedding_model and hasattr(self._embedding_model, '_available') and self._embedding_model._available:
            try:
                import numpy as np
                q_emb = self._embedding_model.embed(query)
                best_score = 0.0
                best_result = None
                for entry in self._semantic_cache:
                    sim = float(np.dot(q_emb, entry["embedding"]))
                    if sim > best_score:
                        best_score = sim
                        best_result = entry["result"]
                if best_score >= self._semantic_threshold and best_result:
                    self._hits["semantic"] += 1
                    return best_result, "semantic"
            except Exception:
                pass

        # Level 3: miss
        self._hits["miss"] += 1
        return None, "miss"

    def set(self, query: str, result: Dict):
        """Store a result in both exact and semantic caches."""
        key = self._hash_key(query)

        # Exact cache
        if len(self._exact_cache) >= self._max_exact:
            oldest = next(iter(self._exact_cache))
            del self._exact_cache[oldest]
        self._exact_cache[key] = result

        # Semantic cache
        if self._embedding_model and hasattr(self._embedding_model, '_available') and self._embedding_model._available:
            try:
                embedding = self._embedding_model.embed(query)
                if len(self._semantic_cache) >= self._max_semantic:
                    self._semantic_cache.pop(0)
                self._semantic_cache.append({
                    "query": query,
                    "embedding": embedding,
                    "result": result,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    def set_exact(self, query: str, result: Dict):
        """Store in exact cache only."""
        key = self._hash_key(query)
        self._exact_cache[key] = result

    def invalidate_topic(self, topic: str):
        """Remove cached entries related to a topic."""
        if not topic:
            return
        topic_lower = topic.lower()
        keys_to_remove = []
        for k, v in self._exact_cache.items():
            cached_answer = v.get("answer", "")
            if topic_lower in cached_answer.lower():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self._exact_cache[k]

        self._semantic_cache = [
            e for e in self._semantic_cache
            if topic_lower not in e.get("query", "").lower()
        ]

    def get_stats(self) -> Dict:
        total = sum(self._hits.values())
        return {
            "exact_cache_size": len(self._exact_cache),
            "semantic_cache_size": len(self._semantic_cache),
            "hits": dict(self._hits),
            "hit_rate": round(
                (self._hits["exact"] + self._hits["semantic"]) / max(total, 1), 3
            ),
        }
