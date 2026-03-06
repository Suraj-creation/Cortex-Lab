"""
Embedding Model for Cortex Lab — Upgraded Architecture
Primary: BAAI/bge-large-en-v1.5 (1024d) — SOTA dense retrieval
Reranker: BAAI/bge-reranker-v2-m3 — cross-encoder reranking
Fallback: sentence-transformers/all-MiniLM-L6-v2 (384d)
Hash fallback if no model available.
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict
import hashlib


class EmbeddingModel:
    """
    Sentence embedding model with priority-based loading.
    BGE-large-en-v1.5 (1024d) → MiniLM-L6-v2 (384d) → hash fallback.
    Supports separate query/passage embedding for asymmetric retrieval.
    """

    MODEL_PRIORITY = [
        ("BAAI/bge-large-en-v1.5", 1024),
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ]

    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.dimension = 1024
        self._is_bge = False
        # LRU embedding cache for deduplication (§3.6)
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._embed_cache_max = 4096
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_model()

    def _load_model(self):
        """Try loading models in priority order."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("  ⚠ sentence-transformers not installed, using hash embeddings")
            self.model = None
            self.dimension = 384
            self.model_name = "hash-fallback"
            return

        if self.model_name:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                self._is_bge = "bge" in self.model_name.lower()
                print(f"  ✓ Embedding model loaded: {self.model_name} ({self.dimension}d on {self.device})")
                return
            except Exception as e:
                print(f"  ⚠ Failed to load {self.model_name}: {e}")

        for mname, expected_dim in self.MODEL_PRIORITY:
            try:
                self.model = SentenceTransformer(mname, device=self.device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.model_name = mname
                self._is_bge = "bge" in mname.lower()
                print(f"  ✓ Embedding model loaded: {mname} ({self.dimension}d on {self.device})")
                return
            except Exception as e:
                print(f"  ⚠ {mname} unavailable: {e}")

        print("  ⚠ All embedding models failed, using hash embeddings (384d)")
        self.model = None
        self.dimension = 384
        self.model_name = "hash-fallback"

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text with LRU cache (§3.6)."""
        # Check cache first
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._embed_cache:
            self._embed_cache.move_to_end(cache_key)
            self._cache_hits += 1
            return self._embed_cache[cache_key]

        self._cache_misses += 1
        if self.model is not None:
            if self._is_bge:
                text_input = f"Represent this sentence for retrieval: {text}"
            else:
                text_input = text
            emb = self.model.encode(text_input, normalize_embeddings=True, show_progress_bar=False)
            result = np.array(emb, dtype=np.float32)
        else:
            result = self._fallback_embed(text)

        # Store in cache
        self._embed_cache[cache_key] = result
        if len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)

        return result

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with instruction prefix for BGE asymmetric retrieval."""
        if self.model is not None:
            if self._is_bge:
                text = f"Represent this sentence for retrieval: {text}"
            emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return np.array(emb, dtype=np.float32)
        return self._fallback_embed(text)

    def embed_passage(self, text: str) -> np.ndarray:
        """Embed a passage/document (no instruction prefix)."""
        if self.model is not None:
            emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return np.array(emb, dtype=np.float32)
        return self._fallback_embed(text)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        if self.model is not None:
            embs = self.model.encode(texts, normalize_embeddings=True,
                                     batch_size=batch_size, show_progress_bar=False)
            return np.array(embs, dtype=np.float32)
        return np.array([self._fallback_embed(t) for t in texts], dtype=np.float32)

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding fallback."""
        h = hashlib.sha512(text.encode("utf-8")).digest()
        extended = h
        while len(extended) < self.dimension * 4:
            extended += hashlib.sha512(extended).digest()
        arr = np.frombuffer(extended[: self.dimension * 4], dtype=np.float32).copy()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))


class CrossEncoderReranker:
    """
    Cross-encoder reranker for fine-grained query-document relevance scoring.
    Primary: BAAI/bge-reranker-v2-m3
    Fallback: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            print(f"  ✓ Cross-encoder reranker loaded: {self.model_name}")
        except Exception as e:
            print(f"  ⚠ Cross-encoder {self.model_name} unavailable: {e}")
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)
                self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                print(f"  ✓ Fallback cross-encoder loaded: {self.model_name}")
            except Exception:
                self.model = None
                print("  ⚠ No cross-encoder available, using embedding-based reranking")

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """Rerank documents. Returns List[(original_index, score)] sorted by score."""
        if self.model is None or not documents:
            return [(i, 0.5) for i in range(len(documents))]
        try:
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)
            if not hasattr(scores, '__len__'):
                scores = [scores]
            indexed = list(enumerate(float(s) for s in scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return indexed[:top_k] if top_k else indexed
        except Exception as e:
            print(f"  ⚠ Cross-encoder reranking failed: {e}")
            return [(i, 0.5) for i in range(len(documents))]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        if self.model is None:
            return 0.5
        try:
            score = self.model.predict([(query, document)], show_progress_bar=False)
            return float(score[0]) if hasattr(score, '__len__') else float(score)
        except Exception:
            return 0.5
