"""
Wake Word Detection — "Cortex" Activation
Always-on lightweight wake word detection that triggers AmbientService.start().

Uses openwakeword (ONNX-based, ~5MB models):
  - Runs continuously on a separate lightweight audio stream (low CPU)
  - Listens for "hey cortex" or similar wake phrases
  - On detection → triggers callback to start full ambient pipeline
  - After configurable silence timeout → auto-pause back to wake-word-only mode

Modes:
  - always_on: Wake word detector runs 24/7, full pipeline only when activated
  - manual: User clicks Start in UI (no wake word)
  - hybrid: Wake word OR manual start
"""

import numpy as np
import time
import threading
from typing import Optional, Callable


class WakeWordDetector:
    """
    Lightweight wake word detection using openwakeword.
    Sits between VAD and the full ambient pipeline.
    """

    # Confidence threshold for wake word activation
    ACTIVATION_THRESHOLD = 0.5
    # Cooldown after activation to prevent double-triggers
    COOLDOWN_S = 3.0
    # Audio frame size expected by openwakeword (16kHz, ~80ms chunks = 1280 samples)
    CHUNK_SIZE = 1280

    def __init__(self, wake_word: str = "hey_jarvis",
                 threshold: float = 0.5,
                 on_wake: Optional[Callable] = None):
        """
        Args:
            wake_word: Wake word model name. openwakeword includes pre-trained models:
                       "hey_jarvis", "alexa", "hey_mycroft", "timer", etc.
                       For custom "cortex", train with openwakeword training pipeline.
            threshold: Activation confidence threshold (0-1)
            on_wake: Callback when wake word is detected
        """
        self.wake_word = wake_word
        self.threshold = threshold
        self._on_wake = on_wake
        self._running = False
        self._last_trigger_time = 0.0
        self._model = None
        self._buffer = np.array([], dtype=np.int16)

        self._load_model()

    def _load_model(self):
        """Load the openwakeword model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            # Download default models if not present
            openwakeword.utils.download_models()

            print(f"  🎯 Loading wake word model: {self.wake_word}")
            t0 = time.time()
            self._model = Model(
                wakeword_models=[self.wake_word],
                inference_framework="onnx",
            )
            elapsed = time.time() - t0
            print(f"  ✅ Wake word detector loaded in {elapsed:.1f}s "
                  f"(model: {self.wake_word}, threshold: {self.threshold})")
        except ImportError:
            print("  ⚠ openwakeword not installed — wake word detection disabled")
            print("  ↳ Install with: pip install openwakeword")
            self._model = None
        except Exception as e:
            print(f"  ⚠ Wake word model load failed: {e}")
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start wake word detection."""
        if not self._model:
            return
        self._running = True
        self._buffer = np.array([], dtype=np.int16)
        print("  🎯 Wake word detection STARTED")

    def stop(self):
        """Stop wake word detection."""
        self._running = False
        self._buffer = np.array([], dtype=np.int16)
        if self._model:
            self._model.reset()
        print("  🎯 Wake word detection STOPPED")

    def process_frame(self, frame: np.ndarray):
        """
        Process an audio frame from AudioCapture.
        Called from the audio callback thread.

        Args:
            frame: int16 PCM audio, 16kHz, mono (typically 512 samples / 32ms)
        """
        if not self._running or not self._model:
            return

        # Accumulate frames until we have enough for openwakeword (1280 samples)
        self._buffer = np.concatenate([self._buffer, frame])

        while len(self._buffer) >= self.CHUNK_SIZE:
            chunk = self._buffer[:self.CHUNK_SIZE]
            self._buffer = self._buffer[self.CHUNK_SIZE:]

            # Run inference
            prediction = self._model.predict(chunk)

            # Check if any model exceeds threshold
            for model_name, score in prediction.items():
                if score >= self.threshold:
                    now = time.time()
                    # Cooldown check
                    if (now - self._last_trigger_time) < self.COOLDOWN_S:
                        continue

                    self._last_trigger_time = now
                    print(f"  🎯 Wake word detected: '{model_name}' "
                          f"(confidence: {score:.2f})")

                    if self._on_wake:
                        # Fire callback in a separate thread to avoid blocking audio
                        threading.Thread(
                            target=self._on_wake,
                            daemon=True,
                            name="wake-trigger",
                        ).start()

    def set_wake_callback(self, callback: Callable):
        """Set the callback for wake word detection."""
        self._on_wake = callback

    def set_threshold(self, threshold: float):
        """Update detection threshold."""
        self.threshold = max(0.1, min(1.0, threshold))

    def get_stats(self) -> dict:
        return {
            "available": self.is_available,
            "running": self._running,
            "wake_word": self.wake_word,
            "threshold": self.threshold,
            "last_trigger": self._last_trigger_time,
        }
