"""
Tier 2: Speaker Identification — Resemblyzer (GE2E)
Extracts 256-dimensional speaker embeddings and verifies against enrolled voiceprint.
Cost: ~30 ms per segment, CPU only, ~10 MB model.

Replaces SpeechBrain ECAPA-TDNN which is incompatible with Python 3.14.

Speaker labels:
  - "USER" if cosine similarity >= 0.70 against enrolled voiceprint
  - "SPEAKER_A", "SPEAKER_B", etc. for unknown speakers (online clustering)
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class SpeakerIdentifier:
    SIMILARITY_THRESHOLD = 0.70
    CLUSTER_MERGE_THRESHOLD = 0.75  # Merge clusters if similarity > this
    VOICEPRINT_DIR = "data/voiceprints"

    def __init__(self, data_dir: str = "data"):
        self.VOICEPRINT_DIR = f"{data_dir}/voiceprints"
        Path(self.VOICEPRINT_DIR).mkdir(parents=True, exist_ok=True)

        # Shim webrtcvad if missing (Python 3.14 — C extension won't compile).
        # Resemblyzer uses it only in trim_long_silences(); we already have Silero VAD
        # upstream so the silence-trimming can safely become a no-op.
        import importlib
        if importlib.util.find_spec("webrtcvad") is None:
            import types
            _stub = types.ModuleType("webrtcvad")
            class _FakeVad:
                def __init__(self, mode=3): pass
                def is_speech(self, buf, sample_rate): return True
            _stub.Vad = _FakeVad
            import sys
            sys.modules["webrtcvad"] = _stub

        from resemblyzer import VoiceEncoder

        print("  👤 Loading Resemblyzer GE2E speaker encoder...")
        t0 = time.time()
        self.encoder = VoiceEncoder("cpu")
        elapsed = time.time() - t0
        print(f"  ✅ Resemblyzer loaded in {elapsed:.1f}s (256-dim embeddings)")

        # User voiceprint
        self.user_voiceprint: Optional[np.ndarray] = None
        self._load_voiceprint()

        # Session speaker clusters: {label: centroid_embedding}
        self._speaker_clusters: Dict[str, np.ndarray] = {}
        self._next_speaker_idx = 0

        # Aliases: {"SPEAKER_A": "Sarah"} — user-defined names
        self._aliases: Dict[str, str] = {}
        self._load_aliases()

    # ── Embedding Extraction ─────────────────────────────────────────────

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract a 256-dimensional speaker embedding from audio (int16, 16 kHz).
        Returns: np.ndarray of shape (256,)
        """
        from resemblyzer import preprocess_wav

        # Convert int16 → float32 normalized [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        # Resemblyzer expects float32 at 16kHz
        wav = preprocess_wav(audio_f32, source_sr=16000)

        embedding = self.encoder.embed_utterance(wav)
        return embedding  # (256,)

    # ── Identification ───────────────────────────────────────────────────

    def identify(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Identify the speaker of an audio segment.

        Returns:
            (label, confidence)
            label: "USER" or "SPEAKER_A", "SPEAKER_B", etc.
            confidence: cosine similarity score
        """
        if len(audio) < 4800:  # < 0.3s at 16 kHz — too short for reliable ID
            return ("UNKNOWN", 0.0)

        embedding = self.extract_embedding(audio)

        # Check against user voiceprint first
        if self.user_voiceprint is not None:
            similarity = self._cosine_similarity(embedding, self.user_voiceprint)
            if similarity >= self.SIMILARITY_THRESHOLD:
                return ("USER", float(similarity))

        # Cluster into speaker groups
        label = self._cluster_speaker(embedding)
        # Get confidence as similarity to cluster centroid
        if label in self._speaker_clusters:
            confidence = float(self._cosine_similarity(
                embedding, self._speaker_clusters[label]
            ))
        else:
            confidence = 1.0  # New cluster, first sample

        return (label, confidence)

    def _cluster_speaker(self, embedding: np.ndarray) -> str:
        """Assign a non-user speaker to a cluster (online k-means style)."""
        best_label = None
        best_sim = -1.0

        for label, centroid in self._speaker_clusters.items():
            sim = self._cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_sim >= self.CLUSTER_MERGE_THRESHOLD and best_label:
            # Update centroid (running average)
            old = self._speaker_clusters[best_label]
            self._speaker_clusters[best_label] = (old + embedding) / 2.0
            return best_label

        # Create new cluster
        label = f"SPEAKER_{chr(65 + self._next_speaker_idx)}"  # A, B, C, ...
        self._next_speaker_idx += 1
        self._speaker_clusters[label] = embedding
        return label

    # ── Enrollment ───────────────────────────────────────────────────────

    def enroll_user(self, audio_samples: List[np.ndarray]) -> Dict:
        """
        Enroll user voiceprint from multiple audio samples.
        Each sample should be 3-10 seconds of clean speech.

        Returns enrollment result dict.
        """
        if not audio_samples:
            return {"success": False, "error": "No audio samples provided"}

        embeddings = []
        for sample in audio_samples:
            if len(sample) < 4800:  # < 0.3s
                continue
            emb = self.extract_embedding(sample)
            embeddings.append(emb)

        if len(embeddings) < 2:
            return {"success": False, "error": "Need at least 2 valid speech samples"}

        # Average the embeddings to create the voiceprint
        self.user_voiceprint = np.mean(embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(self.user_voiceprint)
        if norm > 0:
            self.user_voiceprint = self.user_voiceprint / norm

        # Save to disk
        self._save_voiceprint()

        # Compute inter-sample consistency
        similarities = []
        for emb in embeddings:
            sim = self._cosine_similarity(emb, self.user_voiceprint)
            similarities.append(float(sim))

        return {
            "success": True,
            "samples_used": len(embeddings),
            "consistency": round(float(np.mean(similarities)), 3),
            "min_similarity": round(float(np.min(similarities)), 3),
        }

    def is_enrolled(self) -> bool:
        """Check if user has enrolled their voice."""
        return self.user_voiceprint is not None

    # ── Alias Management ─────────────────────────────────────────────────

    def set_alias(self, speaker_label: str, name: str):
        """Set a human-readable name for a speaker cluster."""
        self._aliases[speaker_label] = name
        self._save_aliases()

    def get_display_name(self, speaker_label: str) -> str:
        """Get display name for a speaker label."""
        if speaker_label == "USER":
            return "You"
        return self._aliases.get(speaker_label, speaker_label)

    # ── Session Management ───────────────────────────────────────────────

    def reset_session_clusters(self):
        """Reset speaker clusters for a new session/conversation."""
        self._speaker_clusters.clear()
        self._next_speaker_idx = 0

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_voiceprint(self):
        if self.user_voiceprint is not None:
            path = Path(self.VOICEPRINT_DIR) / "user.npy"
            np.save(str(path), self.user_voiceprint)
            print(f"  💾 Voiceprint saved to {path}")

    def _load_voiceprint(self):
        path = Path(self.VOICEPRINT_DIR) / "user.npy"
        if path.exists():
            vp = np.load(str(path))
            # Handle dimension mismatch (old ECAPA 192-dim vs new Resemblyzer 256-dim)
            if vp.shape[0] == 256:
                self.user_voiceprint = vp
                print(f"  ✅ Voiceprint loaded ({vp.shape[0]}-dim)")
            else:
                print(f"  ⚠ Voiceprint dimension mismatch ({vp.shape[0]}-dim, expected 256) "
                      f"— re-enrollment required")
                self.user_voiceprint = None
        else:
            print("  ⚠ No voiceprint found — enrollment required")

    def _save_aliases(self):
        path = Path(self.VOICEPRINT_DIR) / "speaker_aliases.json"
        with open(path, "w") as f:
            json.dump(self._aliases, f, indent=2)

    def _load_aliases(self):
        path = Path(self.VOICEPRINT_DIR) / "speaker_aliases.json"
        if path.exists():
            with open(path) as f:
                self._aliases = json.load(f)

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def get_stats(self) -> dict:
        return {
            "enrolled": self.is_enrolled(),
            "active_clusters": len(self._speaker_clusters),
            "cluster_labels": list(self._speaker_clusters.keys()),
            "aliases": self._aliases,
        }
