"""
Tier 1: Voice Activity Detection — Silero VAD v5
Processes 30 ms audio frames, emits complete speech segments when silence is detected.
Cost: <1 ms per frame, CPU only, ~2 MB model (auto-downloaded via torch.hub).

Flow:
  AudioCapture → (30 ms frame) → VAD.process_frame()
                                   │
                                   ├── silence → discard (95% of frames)
                                   └── speech  → accumulate → on silence gap → emit segment
"""

import numpy as np
import torch
import time
from typing import Callable, Optional, List


class VoiceActivityDetector:
    THRESHOLD = 0.5            # Speech probability threshold (0.0 – 1.0)
    MIN_SPEECH_MS = 250        # Minimum speech duration to emit (filter noise bursts)
    MIN_SILENCE_MS = 300       # Silence duration to finalize a speech segment
    SPEECH_PAD_MS = 100        # Padding added before / after speech boundaries
    SAMPLE_RATE = 16000        # Must match AudioCapture.SAMPLE_RATE
    FRAME_MS = 32              # Must match AudioCapture.FRAME_MS (512 samples for Silero VAD v5)

    def __init__(self, threshold: float = 0.5):
        self.THRESHOLD = threshold

        # Load Silero VAD v5 (ONNX for maximum speed on CPU)
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True,
        )
        # Reset model state
        self.model.reset_states()

        # Internal state
        self._speech_active = False
        self._speech_frames: List[np.ndarray] = []
        self._silence_count = 0  # consecutive silent frames
        self._speech_start_time: float = 0.0
        self._frame_count = 0

        # Callbacks
        self._on_speech_segment: Optional[Callable] = None
        self._on_vad_activity: Optional[Callable] = None  # For UI live indicator

        # Stats
        self._total_speech_segments = 0
        self._total_speech_seconds = 0.0

        print(f"  🔊 Silero VAD loaded (threshold={self.THRESHOLD})")

    # ── Core Processing ──────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """
        Process a single 30 ms audio frame (int16, 480 samples at 16 kHz).
        Called by AudioCapture's frame callback.

        When a complete speech segment is detected (speech → silence gap),
        the accumulated audio + timestamps are emitted via the speech callback.
        """
        self._frame_count += 1
        timestamp = self._frame_count * self.FRAME_MS / 1000.0

        # Convert int16 → float32 normalized (Silero expects [-1, 1])
        audio_f32 = frame.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_f32)

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.SAMPLE_RATE).item()

        # Notify UI of VAD activity
        if self._on_vad_activity:
            try:
                self._on_vad_activity(speech_prob, timestamp)
            except Exception:
                pass

        is_speech = speech_prob >= self.THRESHOLD

        if is_speech:
            if not self._speech_active:
                # Speech just started
                self._speech_active = True
                self._speech_start_time = timestamp
                self._silence_count = 0
                self._speech_frames = []

            self._speech_frames.append(frame.copy())
            self._silence_count = 0

        else:
            if self._speech_active:
                self._silence_count += 1
                # Keep accumulating during short silences (< MIN_SILENCE_MS)
                silence_duration_ms = self._silence_count * self.FRAME_MS

                if silence_duration_ms < self.MIN_SILENCE_MS:
                    # Short silence — might be a pause in speech
                    self._speech_frames.append(frame.copy())
                else:
                    # Silence exceeded threshold — finalize the speech segment
                    self._emit_segment(timestamp)

    def _emit_segment(self, end_timestamp: float):
        """Finalize and emit a completed speech segment."""
        if not self._speech_frames:
            self._speech_active = False
            return

        # Check minimum duration
        duration_ms = len(self._speech_frames) * self.FRAME_MS
        if duration_ms < self.MIN_SPEECH_MS:
            # Too short — likely a noise burst, discard
            self._speech_active = False
            self._speech_frames = []
            self._silence_count = 0
            return

        # Concatenate all speech frames into one contiguous array
        audio_segment = np.concatenate(self._speech_frames)
        duration_s = len(audio_segment) / self.SAMPLE_RATE

        # Update stats
        self._total_speech_segments += 1
        self._total_speech_seconds += duration_s

        # Emit via callback
        if self._on_speech_segment:
            try:
                self._on_speech_segment(
                    audio_segment,
                    self._speech_start_time,
                    end_timestamp,
                )
            except Exception as e:
                print(f"  ⚠ VAD speech callback error: {e}")

        # Reset state
        self._speech_active = False
        self._speech_frames = []
        self._silence_count = 0

        # Reset Silero model state between segments for clean decoding
        self.model.reset_states()

    # ── Callbacks ────────────────────────────────────────────────────────

    def set_speech_callback(self, callback: Callable[[np.ndarray, float, float], None]):
        """
        Register callback for complete speech segments.
        Args to callback: (audio_int16, start_time_s, end_time_s)
        """
        self._on_speech_segment = callback

    def set_activity_callback(self, callback: Callable[[float, float], None]):
        """
        Register callback for live VAD activity (for UI).
        Args to callback: (speech_probability, timestamp_s)
        """
        self._on_vad_activity = callback

    # ── Control ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset VAD state (e.g. when stopping/starting ambient)."""
        self._speech_active = False
        self._speech_frames = []
        self._silence_count = 0
        self._frame_count = 0
        self.model.reset_states()

    def set_threshold(self, threshold: float):
        """Update VAD threshold. Lower = more sensitive."""
        self.THRESHOLD = max(0.1, min(0.95, threshold))

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "threshold": self.THRESHOLD,
            "speech_active": self._speech_active,
            "total_segments": self._total_speech_segments,
            "total_speech_seconds": round(self._total_speech_seconds, 1),
            "frames_processed": self._frame_count,
        }
