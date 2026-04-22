"""
Vector Store for Cortex Lab
FAISS-based vector storage with tiered indexing (Hot/Warm/Cold).
L2-normalized vectors for cosine similarity via inner-product search.
"""

import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (or batch) so cosine sim == inner product."""
    vec = np.array(vec, dtype=np.float32)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-10)
    else:
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vec / norms


class VectorStore:
    """
    FAISS-based vector store with hot/warm/cold tiering.
    - Hot (HNSW): Recent memories (<30 days), ~5ms, ~98% recall
    - Warm (IVFFlat): 30 days-1 year, ~15ms, ~95% recall
    - Cold (IVFFlat): >1 year, ~25ms, ~90% recall
    
    Falls back to brute-force flat index if FAISS is unavailable.
    """

    def __init__(self, dimension: int = 384, data_dir: str = "data/vectors"):
        self.dimension = dimension
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.faiss = None
        self._use_faiss = False
        self._load_faiss()

        # In-memory storage
        self.hot_index = None   # HNSW for recent
        self.warm_index = None  # IVFFlat for medium
        self.cold_index = None  # IVFFlat for old

        # ID mappings: faiss_idx -> memory_id
        self.hot_ids: List[str] = []
        self.warm_ids: List[str] = []
        self.cold_ids: List[str] = []

        # Deletion tracking — FAISS doesn't support in-place deletion
        self._deleted_ids: Set[str] = set()

        # Simple flat store as fallback / primary
        self.vectors: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, datetime] = {}

        self._init_indices()
        self._load_state()

    def _load_faiss(self):
        try:
            import faiss
            self.faiss = faiss
            self._use_faiss = True
            print("  ✓ FAISS loaded successfully")
        except ImportError:
            print("  ⚠ FAISS not available, using NumPy fallback")
            self._use_faiss = False

    def _init_indices(self):
        if self._use_faiss:
            # Hot tier: HNSW (fast, high recall for recent memories)
            self.hot_index = self.faiss.IndexHNSWFlat(self.dimension, 32)
            self.hot_index.hnsw.efSearch = 64
            self.hot_index.hnsw.efConstruction = 128

            # Warm tier: IVFFlat with scalar quantizer (4x compression)
            # Will be trained once we have enough warm vectors
            self.warm_quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.warm_index = None  # Initialized when first warm vectors added

            # Cold tier: IVFFlat with PQ (8-16x compression)
            self.cold_quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.cold_index = None  # Initialized when first cold vectors added

    def _ensure_warm_index(self, training_vectors: np.ndarray = None):
        """Initialize warm tier IVF-SQ8 index if not already created."""
        if self._use_faiss and self.warm_index is None:
            nlist = max(4, min(int(len(self.warm_ids) ** 0.5), 64))
            # IVF with scalar quantizer (SQ8) — 4x compression
            self.warm_index = self.faiss.IndexIVFScalarQuantizer(
                self.warm_quantizer, self.dimension, nlist,
                self.faiss.ScalarQuantizer.QT_8bit
            )
            if training_vectors is not None and len(training_vectors) >= nlist:
                self.warm_index.train(training_vectors)
                self.warm_index.nprobe = max(2, nlist // 4)

    def _ensure_cold_index(self, training_vectors: np.ndarray = None):
        """Initialize cold tier IVF-PQ index if not already created."""
        if self._use_faiss and self.cold_index is None:
            nlist = max(4, min(int(len(self.cold_ids) ** 0.5), 32))
            # IVF with product quantizer (PQ) — 8-16x compression
            m = min(48, self.dimension)  # Number of sub-quantizers (must divide dimension)
            while self.dimension % m != 0 and m > 1:
                m -= 1
            self.cold_index = self.faiss.IndexIVFPQ(
                self.cold_quantizer, self.dimension, nlist, m, 8
            )
            if training_vectors is not None and len(training_vectors) >= nlist:
                self.cold_index.train(training_vectors)
                self.cold_index.nprobe = max(2, nlist // 4)

    def migrate_tiers(self):
        """
        Migrate vectors between hot/warm/cold tiers based on age.
        - Hot: < 30 days
        - Warm: 30 days - 1 year
        - Cold: > 1 year
        Called periodically (e.g., daily or on startup).
        """
        if not self._use_faiss:
            return

        now = datetime.now()
        hot_cutoff = now - timedelta(days=30)
        warm_cutoff = now - timedelta(days=365)

        to_warm = []
        to_cold = []

        for mid, ts in list(self.timestamps.items()):
            if ts < warm_cutoff and mid not in self.cold_ids:
                to_cold.append(mid)
            elif ts < hot_cutoff and mid not in self.warm_ids and mid not in self.cold_ids:
                to_warm.append(mid)

        if to_warm:
            warm_vecs = np.array([self.vectors[mid] for mid in to_warm if mid in self.vectors], dtype=np.float32)
            if len(warm_vecs) >= 4:
                self._ensure_warm_index(warm_vecs)
                if self.warm_index is not None and self.warm_index.is_trained:
                    self.warm_index.add(warm_vecs)
                    self.warm_ids.extend(to_warm[:len(warm_vecs)])
                    print(f"  📦 Migrated {len(warm_vecs)} vectors to warm tier")

        if to_cold:
            cold_vecs = np.array([self.vectors[mid] for mid in to_cold if mid in self.vectors], dtype=np.float32)
            if len(cold_vecs) >= 4:
                self._ensure_cold_index(cold_vecs)
                if self.cold_index is not None and self.cold_index.is_trained:
                    self.cold_index.add(cold_vecs)
                    self.cold_ids.extend(to_cold[:len(cold_vecs)])
                    print(f"  🧊 Migrated {len(cold_vecs)} vectors to cold tier")

    def add(self, memory_id: str, embedding: np.ndarray, timestamp: Optional[datetime] = None):
        """Add a vector to the store (L2-normalized for cosine similarity)."""
        if timestamp is None:
            timestamp = datetime.now()

        embedding = np.array(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # L2-normalize so FAISS L2 distance ≈ cosine distance
        embedding = _l2_normalize(embedding)

        self.vectors[memory_id] = embedding.flatten()
        self.timestamps[memory_id] = timestamp

        # Remove from deleted set if re-added
        self._deleted_ids.discard(memory_id)

        if self._use_faiss and self.hot_index is not None:
            self.hot_index.add(embedding)
            self.hot_ids.append(memory_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 20,
               time_start: Optional[datetime] = None,
               time_end: Optional[datetime] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors across all tiers. Returns list of (memory_id, score).
        Filters out deleted IDs. Query is L2-normalized for consistency."""
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = _l2_normalize(query_embedding)

        results = []

        if self._use_faiss:
            # Search hot tier (HNSW — highest recall, lowest latency)
            if self.hot_index is not None and self.hot_index.ntotal > 0:
                k = min(top_k * 2, self.hot_index.ntotal)
                distances, indices = self.hot_index.search(query_embedding, k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.hot_ids):
                        continue
                    mid = self.hot_ids[idx]
                    score = 1.0 / (1.0 + float(dist))
                    results.append((mid, score))

            # Search warm tier (IVF-SQ8 — moderate recall)
            if self.warm_index is not None and self.warm_index.ntotal > 0:
                k_warm = min(top_k, self.warm_index.ntotal)
                try:
                    distances, indices = self.warm_index.search(query_embedding, k_warm)
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx < 0 or idx >= len(self.warm_ids):
                            continue
                        mid = self.warm_ids[idx]
                        score = 1.0 / (1.0 + float(dist)) * 0.95  # Slight discount for warm
                        results.append((mid, score))
                except Exception:
                    pass

            # Search cold tier (IVF-PQ — archive recall)
            if self.cold_index is not None and self.cold_index.ntotal > 0:
                k_cold = min(top_k // 2, self.cold_index.ntotal)
                try:
                    distances, indices = self.cold_index.search(query_embedding, k_cold)
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx < 0 or idx >= len(self.cold_ids):
                            continue
                        mid = self.cold_ids[idx]
                        score = 1.0 / (1.0 + float(dist)) * 0.90  # Slight discount for cold
                        results.append((mid, score))
                except Exception:
                    pass
        else:
            # NumPy fallback: brute-force cosine similarity
            for mid, vec in self.vectors.items():
                sim = float(np.dot(query_embedding.flatten(), vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-10
                ))
                results.append((mid, sim))

        # Time filtering
        if time_start or time_end:
            filtered = []
            for mid, score in results:
                ts = self.timestamps.get(mid)
                if ts:
                    if time_start and ts < time_start:
                        continue
                    if time_end and ts > time_end:
                        continue
                filtered.append((mid, score))
            results = filtered

        # Filter out deleted IDs (FAISS cannot delete in-place)
        if self._deleted_ids:
            results = [(mid, s) for mid, s in results if mid not in self._deleted_ids]

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, memory_id: str):
        """Mark a vector as deleted. FAISS doesn't support in-place removal,
        so we track deleted IDs and filter them from search results.
        The flat dict is also cleaned so count() stays accurate."""
        self._deleted_ids.add(memory_id)
        self.vectors.pop(memory_id, None)
        self.timestamps.pop(memory_id, None)

    def count(self) -> int:
        """Return number of live (non-deleted) vectors."""
        return len(self.vectors)

    def tombstone_stats(self) -> Dict[str, float]:
        """Return tombstone/deletion pressure stats for maintenance decisions."""
        live_count = len(self.vectors)
        deleted_count = len(self._deleted_ids)
        total_observed = live_count + deleted_count
        deleted_ratio = (deleted_count / total_observed) if total_observed > 0 else 0.0
        return {
            "live_count": live_count,
            "deleted_count": deleted_count,
            "deleted_ratio": round(float(deleted_ratio), 4),
            "total_observed": total_observed,
        }

    def compact_tombstones(
        self,
        force: bool = False,
        min_deleted: int = 100,
        min_deleted_ratio: float = 0.15,
    ) -> Dict[str, float]:
        """Compact tombstones by rebuilding live index mappings.

        FAISS does not support efficient in-place deletion. This method removes
        accumulated tombstones by rebuilding the searchable tier indexes from
        live vectors only.
        """
        before = self.tombstone_stats()
        deleted_count = int(before["deleted_count"])
        deleted_ratio = float(before["deleted_ratio"])

        if deleted_count == 0:
            return {"compacted": False, "reason": "no_tombstones", **before}

        if not force and deleted_count < max(int(min_deleted), 0) and deleted_ratio < float(min_deleted_ratio):
            return {
                "compacted": False,
                "reason": "below_threshold",
                **before,
            }

        live_items = list(self.vectors.items())

        # Reset FAISS tier indexes and ID maps from live vectors only.
        self.hot_ids = []
        self.warm_ids = []
        self.cold_ids = []

        if self._use_faiss:
            self.hot_index = None
            self.warm_index = None
            self.cold_index = None
            self._init_indices()

            if live_items and self.hot_index is not None:
                ids = [mid for mid, _ in live_items]
                vectors = np.array([vec for _, vec in live_items], dtype=np.float32)
                vectors = _l2_normalize(vectors)
                self.hot_index.add(vectors)
                self.hot_ids = ids

            # Re-populate warm/cold tiers based on timestamps.
            try:
                self.migrate_tiers()
            except Exception:
                pass

        self._deleted_ids.clear()
        after = self.tombstone_stats()
        removed = deleted_count - int(after["deleted_count"])

        return {
            "compacted": True,
            "deleted_compacted": max(removed, 0),
            "live_count": after["live_count"],
            "deleted_count": after["deleted_count"],
            "deleted_ratio": after["deleted_ratio"],
            "total_observed": after["total_observed"],
        }

    def save(self):
        """Persist state to disk (all tiers + deleted IDs)."""
        state = {
            "hot_ids": self.hot_ids,
            "warm_ids": self.warm_ids,
            "cold_ids": self.cold_ids,
            "deleted_ids": list(self._deleted_ids),
            "timestamps": {k: v.isoformat() for k, v in self.timestamps.items()},
        }
        with open(os.path.join(self.data_dir, "vector_state.json"), "w") as f:
            json.dump(state, f)

        # Save vectors as numpy
        if self.vectors:
            ids = list(self.vectors.keys())
            vecs = np.array([self.vectors[i] for i in ids], dtype=np.float32)
            np.save(os.path.join(self.data_dir, "vectors.npy"), vecs)
            with open(os.path.join(self.data_dir, "vector_ids.json"), "w") as f:
                json.dump(ids, f)

        # Save FAISS hot index
        if self._use_faiss and self.hot_index is not None and self.hot_index.ntotal > 0:
            self.faiss.write_index(self.hot_index, os.path.join(self.data_dir, "hot.index"))

        # Save warm/cold FAISS indices
        if self._use_faiss:
            if self.warm_index is not None and self.warm_index.ntotal > 0:
                self.faiss.write_index(self.warm_index, os.path.join(self.data_dir, "warm.index"))
            if self.cold_index is not None and self.cold_index.ntotal > 0:
                self.faiss.write_index(self.cold_index, os.path.join(self.data_dir, "cold.index"))

        print(f"  ✓ Vector store saved ({self.count()} vectors)")

    def _load_state(self):
        """Load state from disk (all tiers + deleted IDs)."""
        state_path = os.path.join(self.data_dir, "vector_state.json")
        ids_path = os.path.join(self.data_dir, "vector_ids.json")
        vecs_path = os.path.join(self.data_dir, "vectors.npy")

        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.hot_ids = state.get("hot_ids", [])
            self.warm_ids = state.get("warm_ids", [])
            self.cold_ids = state.get("cold_ids", [])
            self._deleted_ids = set(state.get("deleted_ids", []))
            self.timestamps = {
                k: datetime.fromisoformat(v)
                for k, v in state.get("timestamps", {}).items()
            }

        if os.path.exists(ids_path) and os.path.exists(vecs_path):
            with open(ids_path) as f:
                ids = json.load(f)
            vecs = np.load(vecs_path)
            for i, mid in enumerate(ids):
                self.vectors[mid] = vecs[i]

            # Try loading saved FAISS hot index from disk
            hot_path = os.path.join(self.data_dir, "hot.index")
            if self._use_faiss and os.path.exists(hot_path):
                try:
                    self.hot_index = self.faiss.read_index(hot_path)
                    self.hot_ids = ids[:]
                    print(f"  ✓ FAISS hot index loaded from disk ({self.hot_index.ntotal} vectors)")
                except Exception as e:
                    print(f"  ⚠ Failed to load FAISS index, rebuilding: {e}")
                    if self.hot_index is not None and len(vecs) > 0:
                        self.hot_index.add(vecs)
                        self.hot_ids = ids[:]
            elif self._use_faiss and self.hot_index is not None and len(vecs) > 0:
                # Fallback: rebuild FAISS from vectors
                self.hot_index.add(vecs)
                self.hot_ids = ids[:]

            # Load warm index from disk
            warm_path = os.path.join(self.data_dir, "warm.index")
            if self._use_faiss and os.path.exists(warm_path):
                try:
                    self.warm_index = self.faiss.read_index(warm_path)
                    print(f"  ✓ FAISS warm index loaded from disk ({self.warm_index.ntotal} vectors)")
                except Exception as e:
                    print(f"  ⚠ Failed to load warm index: {e}")

            # Load cold index from disk
            cold_path = os.path.join(self.data_dir, "cold.index")
            if self._use_faiss and os.path.exists(cold_path):
                try:
                    self.cold_index = self.faiss.read_index(cold_path)
                    print(f"  ✓ FAISS cold index loaded from disk ({self.cold_index.ntotal} vectors)")
                except Exception as e:
                    print(f"  ⚠ Failed to load cold index: {e}")

            print(f"  ✓ Vector store loaded ({len(ids)} vectors)")

    def get_stats(self) -> Dict:
        # When FAISS is available, use actual tier lists.
        # Otherwise classify vectors by timestamp for accurate stats.
        if self._use_faiss:
            hot = len(self.hot_ids)
            warm = len(self.warm_ids)
            cold = len(self.cold_ids)
        else:
            now = datetime.now()
            hot_cutoff = now - timedelta(days=30)
            warm_cutoff = now - timedelta(days=365)
            hot = warm = cold = 0
            for mid in self.vectors:
                ts = self.timestamps.get(mid)
                if not ts:
                    hot += 1  # No timestamp → treat as recent
                elif ts >= hot_cutoff:
                    hot += 1
                elif ts >= warm_cutoff:
                    warm += 1
                else:
                    cold += 1

        return {
            "total_vectors": self.count(),
            "hot_count": hot,
            "warm_count": warm,
            "cold_count": cold,
            "deleted_tombstones": len(self._deleted_ids),
            "using_faiss": self._use_faiss,
            "dimension": self.dimension,
        }
