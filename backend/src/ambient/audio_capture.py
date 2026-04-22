"""
Tier 0: Continuous Audio Capture with Ring Buffer
Always-on 16 kHz mono PCM capture into a 60-second circular buffer.
Cost: ~0.1% CPU, ~2 MB RAM.

Uses sounddevice for low-latency PortAudio-backed capture.
The ring buffer ensures we never miss the onset of speech — the VAD can look
back up to 60 seconds to retrieve pre-speech audio.
"""

import numpy as np
import sounddevice as sd
import threading
from collections import deque
from typing import Callable, Optional


class AudioCapture:
    SAMPLE_RATE = 16000        # 16 kHz — Whisper / Silero VAD standard
    CHANNELS = 1               # Mono
    DTYPE = np.int16           # 16-bit PCM
    FRAME_MS = 32              # 32 ms frames → 512 samples (Silero VAD v5 ONNX requires exactly 512)
    BUFFER_SECONDS = 60        # 60-second ring buffer

    def __init__(self, device: Optional[int] = None):
        """
        Args:
            device: PortAudio device index. None = system default microphone.
        """
        self.device = device
        self.frame_size = int(self.SAMPLE_RATE * self.FRAME_MS / 1000)  # 480 samples per 30 ms
        self.ring_buffer: deque = deque(
            maxlen=int(self.BUFFER_SECONDS * 1000 / self.FRAME_MS)  # 2000 frames
        )
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_callbacks: list[Callable] = []
        self._audio_level: float = 0.0  # RMS level for UI meter

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        """Start audio capture in a background thread."""
        if self._running:
            return

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="int16",
            blocksize=self.frame_size,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True
        print(f"  🎙️  Audio capture started (device={self.device or 'default'}, "
              f"{self.SAMPLE_RATE}Hz, {self.FRAME_MS}ms frames)")

    def stop(self):
        """Stop audio capture and release resources."""
        if not self._running:
            return
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("  🔇 Audio capture stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Audio Callback (runs in PortAudio C thread) ──────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            pass  # Silently ignore buffer overflows in ambient mode

        # Copy the audio frame (int16 mono)
        frame = indata[:, 0].copy()

        # Update RMS audio level (for UI meter)
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        self._audio_level = float(rms)

        # Store in ring buffer
        with self._lock:
            self.ring_buffer.append(frame)

        # Forward to all registered callbacks (e.g. VAD)
        for cb in self._frame_callbacks:
            try:
                cb(frame)
            except Exception:
                pass  # Never crash the audio thread

    # ── Public API ───────────────────────────────────────────────────────

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Get the last N seconds from the ring buffer as a contiguous int16 array."""
        frames_needed = int(seconds * 1000 / self.FRAME_MS)
        with self._lock:
            available = list(self.ring_buffer)[-frames_needed:]
        if not available:
            return np.array([], dtype=np.int16)
        return np.concatenate(available)

    def get_audio_level(self) -> float:
        """Return current RMS audio level (0.0-32768.0 for int16)."""
        return self._audio_level

    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Register a callback that receives each 30 ms audio frame (int16)."""
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable):
        """Remove a previously registered frame callback."""
        try:
            self._frame_callbacks.remove(callback)
        except ValueError:
            pass

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append({
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sample_rate": d["default_samplerate"],
                })
        return result
