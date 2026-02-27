"""
Tier 3: Speech Transcription — faster-whisper (CTranslate2)
Batch-transcribes speaker-labeled audio segments into text with timestamps.
Cost: ~500 MB VRAM (GPU) or CPU-only fallback (~3x slower).

Uses VRAMGuard to coordinate GPU access with the LLM.
"""

import numpy as np
import asyncio
import time
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor


class Transcriber:
    """
    Speech-to-text using faster-whisper (CTranslate2 backend).
    Supports GPU (float16) and CPU (int8) modes.
    """

    def __init__(self, model_size: str = "small", device: str = "auto",
                 vram_guard=None):
        """
        Args:
            model_size: "tiny", "base", "small", "medium"
            device: "auto" (GPU if available), "cuda", or "cpu"
            vram_guard: VRAMGuard instance for GPU mutual exclusion
        """
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.vram_guard = vram_guard

        compute_type = "float16" if self.device == "cuda" else "int8"
        cpu_threads = 4 if self.device == "cpu" else 1

        print(f"  🗣️  Loading faster-whisper '{model_size}' on {self.device} "
              f"(compute={compute_type})...")
        t0 = time.time()
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        elapsed = time.time() - t0
        print(f"  ✅ Whisper '{model_size}' loaded in {elapsed:.1f}s")

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

        # Stats
        self._total_transcriptions = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0

    def _resolve_device(self, device: str) -> str:
        """Determine best device for Whisper."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # Check free VRAM — need at least 600 MB for Whisper small
                    free_mem = (torch.cuda.get_device_properties(0).total_mem
                                - torch.cuda.memory_allocated(0))
                    if free_mem > 600 * 1024 * 1024:  # 600 MB
                        return "cuda"
                    print(f"  ⚠ Only {free_mem / 1e6:.0f}MB VRAM free, Whisper falling back to CPU")
                    return "cpu"
                return "cpu"
            except (ImportError, Exception):
                return "cpu"
        return device

    async def transcribe(self, audio: np.ndarray,
                         language: Optional[str] = None) -> Dict:
        """
        Transcribe an audio segment (int16, 16 kHz mono).

        Returns:
            {
                "text": str,          # Full transcript
                "language": str,      # Detected language
                "segments": [...],    # Word-level segments with timestamps
                "duration": float,    # Audio duration in seconds
                "confidence": float,  # Average log probability
            }
        """
        if len(audio) == 0:
            return {"text": "", "language": "", "segments": [], "duration": 0, "confidence": 0}

        # Convert int16 → float32 normalized [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0
        duration = len(audio_f32) / 16000.0

        # Transcribe with VRAM guard (if using GPU)
        if self.vram_guard and self.device == "cuda":
            async with self.vram_guard.acquire("whisper"):
                result = await self._transcribe_threaded(audio_f32, language)
        else:
            result = await self._transcribe_threaded(audio_f32, language)

        result["duration"] = round(duration, 2)

        # Update stats
        self._total_transcriptions += 1
        self._total_audio_seconds += duration

        return result

    async def _transcribe_threaded(self, audio_f32: np.ndarray,
                                    language: Optional[str]) -> Dict:
        """Run transcription in a thread to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio_f32,
            language,
        )

    def _transcribe_sync(self, audio_f32: np.ndarray,
                          language: Optional[str]) -> Dict:
        """Synchronous transcription (runs in thread pool)."""
        t0 = time.time()

        kwargs = {
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": False,  # We already ran Silero VAD
        }
        if language:
            kwargs["language"] = language

        segments_iter, info = self.model.transcribe(audio_f32, **kwargs)

        # Collect segments
        segments = []
        full_text_parts = []
        total_log_prob = 0.0
        segment_count = 0

        for seg in segments_iter:
            full_text_parts.append(seg.text.strip())
            total_log_prob += seg.avg_log_prob
            segment_count += 1

            seg_data = {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "avg_log_prob": round(seg.avg_log_prob, 3),
            }

            # Add word-level timestamps if available
            if seg.words:
                seg_data["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3),
                    }
                    for w in seg.words
                ]

            segments.append(seg_data)

        full_text = " ".join(full_text_parts).strip()
        avg_confidence = (total_log_prob / segment_count) if segment_count > 0 else 0.0

        elapsed = time.time() - t0
        self._total_processing_seconds += elapsed

        return {
            "text": full_text,
            "language": info.language if info else "",
            "language_probability": round(info.language_probability, 3) if info else 0.0,
            "segments": segments,
            "confidence": round(avg_confidence, 3),
            "processing_time_s": round(elapsed, 2),
        }

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        rtf = (self._total_processing_seconds / self._total_audio_seconds
               if self._total_audio_seconds > 0 else 0)
        return {
            "model_size": self.model_size,
            "device": self.device,
            "total_transcriptions": self._total_transcriptions,
            "total_audio_seconds": round(self._total_audio_seconds, 1),
            "total_processing_seconds": round(self._total_processing_seconds, 1),
            "real_time_factor": round(rtf, 2),
        }
