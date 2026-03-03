"""
Text-to-Speech — Piper TTS (ONNX, CPU-only)
Neural TTS with near-natural quality, real-time on CPU.
Cost: 0 VRAM, ~50 MB RAM, faster-than-realtime on i9-14900K.

Uses Piper's VITS-based models for high-quality synthesis.
Supports full synthesis and streaming chunk generation.
"""

import io
import wave
import numpy as np
import time
from pathlib import Path
from typing import Optional, Generator


class TextToSpeech:
    VOICES_DIR = "data/tts_voices"
    DEFAULT_VOICE = "en_US-lessac-medium"
    SAMPLE_RATE = 22050  # Piper default output rate

    def __init__(self, voice: str = None, data_dir: str = "data"):
        self.voice_name = voice or self.DEFAULT_VOICE
        self.VOICES_DIR = f"{data_dir}/tts_voices"
        Path(self.VOICES_DIR).mkdir(parents=True, exist_ok=True)

        self._model = None
        self._available = False

        # Stats
        self._total_syntheses = 0
        self._total_chars = 0
        self._total_audio_seconds = 0.0

        self._load_model()

    def _load_model(self):
        """Load the Piper ONNX voice model."""
        model_path = Path(self.VOICES_DIR) / f"{self.voice_name}.onnx"
        config_path = Path(self.VOICES_DIR) / f"{self.voice_name}.onnx.json"

        if not model_path.exists() or not config_path.exists():
            print(f"  ⚠ TTS voice model not found: {model_path}")
            print(f"  ℹ  Download with:")
            print(f"     wget -O {model_path} "
                  f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                  f"en/en_US/lessac/medium/en_US-lessac-medium.onnx")
            print(f"     wget -O {config_path} "
                  f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                  f"en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")
            self._available = False
            return

        try:
            from piper import PiperVoice

            print(f"  🔈 Loading Piper TTS voice: {self.voice_name}...")
            t0 = time.time()
            self._model = PiperVoice.load(str(model_path), str(config_path))
            # Use the model's actual sample rate (may differ from default)
            self.SAMPLE_RATE = self._model.config.sample_rate
            elapsed = time.time() - t0
            self._available = True
            print(f"  ✅ Piper TTS loaded in {elapsed:.1f}s (voice={self.voice_name}, "
                  f"rate={self.SAMPLE_RATE}Hz)")
        except Exception as e:
            print(f"  ⚠ TTS model load error: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._model is not None

    # ── Synthesis ────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text to audio array (int16, 22050 Hz).

        Returns:
            np.ndarray of int16 audio samples, or None if TTS unavailable.
        """
        if not self.is_available or not text.strip():
            return None

        t0 = time.time()

        # Piper v1.4+: synthesize() returns an iterable of AudioChunk objects
        all_audio = []
        for chunk in self._model.synthesize(text):
            arr = chunk.audio_int16_array
            if arr is not None and len(arr) > 0:
                all_audio.append(np.array(arr, dtype=np.int16))

        if not all_audio:
            return None

        audio = np.concatenate(all_audio)

        elapsed = time.time() - t0
        duration = len(audio) / self.SAMPLE_RATE

        # Stats
        self._total_syntheses += 1
        self._total_chars += len(text)
        self._total_audio_seconds += duration

        return audio

    def synthesize_to_wav(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to WAV bytes (ready for HTTP response).

        Returns:
            WAV file bytes or None.
        """
        if not self.is_available or not text.strip():
            return None

        # Piper v1.4+: synthesize_wav() expects a wave.Wave_write object
        wav_buffer = io.BytesIO()
        wav_file = wave.open(wav_buffer, "wb")
        self._model.synthesize_wav(text, wav_file)
        wav_file.close()

        self._total_syntheses += 1
        self._total_chars += len(text)

        return wav_buffer.getvalue()

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Stream raw PCM int16 bytes chunk-by-chunk as they're generated.
        Yields raw PCM int16 bytes (no WAV header — caller wraps if needed).

        Useful for low-latency playback: frontend can start playing before
        synthesis is complete.
        """
        if not self.is_available or not text.strip():
            return

        # Piper v1.4+: synthesize() yields AudioChunk objects
        for chunk in self._model.synthesize(text):
            pcm_bytes = chunk.audio_int16_bytes
            if pcm_bytes and len(pcm_bytes) > 0:
                yield pcm_bytes

        self._total_syntheses += 1
        self._total_chars += len(text)

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for streaming synthesis."""
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def get_stats(self) -> dict:
        return {
            "available": self.is_available,
            "voice": self.voice_name,
            "sample_rate": self.SAMPLE_RATE,
            "total_syntheses": self._total_syntheses,
            "total_chars": self._total_chars,
            "total_audio_seconds": round(self._total_audio_seconds, 1),
        }
