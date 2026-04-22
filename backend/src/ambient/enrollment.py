"""
Voice Enrollment — Record and save user voiceprint.
One-time setup: user records 15-30s of voice → averaged 192-dim embedding.
Voiceprint saved locally to data/voiceprints/user.npy
"""

import numpy as np
import asyncio
import time
from typing import Dict, Optional


class VoiceEnrollment:
    """Manages the user voice enrollment flow."""

    MIN_DURATION_S = 10    # Minimum recording duration
    MAX_DURATION_S = 30    # Maximum recording duration
    SEGMENT_SIZE_S = 3     # Split recording into 3s segments for embedding

    def __init__(self, audio_capture, speaker_id):
        """
        Args:
            audio_capture: AudioCapture instance (for recording)
            speaker_id: SpeakerIdentifier instance (for embedding + saving)
        """
        self.capture = audio_capture
        self.speaker = speaker_id
        self._enrolling = False
        self._enrollment_audio: list = []

    async def start_enrollment(self, duration_seconds: int = 20) -> Dict:
        """
        Record enrollment audio and create voiceprint.

        The user should speak naturally for the specified duration.
        Audio is split into 3-second segments, embeddings are averaged.

        Returns:
            {
                "success": bool,
                "samples_used": int,
                "consistency": float,
                "message": str,
            }
        """
        if self._enrolling:
            return {"success": False, "error": "Enrollment already in progress"}

        duration_seconds = max(self.MIN_DURATION_S,
                               min(self.MAX_DURATION_S, duration_seconds))

        self._enrolling = True
        self._enrollment_audio = []

        # Ensure audio capture is running
        was_running = self.capture.is_running
        if not was_running:
            self.capture.start()

        # Collect audio frames for the specified duration
        frames_needed = int(duration_seconds * 1000 / self.capture.FRAME_MS)
        collected_frames = []

        def collect_frame(frame: np.ndarray):
            if self._enrolling:
                collected_frames.append(frame.copy())

        self.capture.add_frame_callback(collect_frame)

        try:
            # Wait for recording to complete
            print(f"  🎤 Enrollment recording started ({duration_seconds}s)...")
            await asyncio.sleep(duration_seconds)
            print(f"  ✅ Enrollment recording complete ({len(collected_frames)} frames)")
        finally:
            self.capture.remove_frame_callback(collect_frame)
            if not was_running:
                self.capture.stop()
            self._enrolling = False

        if not collected_frames:
            return {"success": False, "error": "No audio captured"}

        # Concatenate all frames
        full_audio = np.concatenate(collected_frames)

        # Split into 3-second segments
        segment_samples = self.SEGMENT_SIZE_S * self.capture.SAMPLE_RATE
        audio_samples = []

        for i in range(0, len(full_audio) - segment_samples, segment_samples):
            segment = full_audio[i:i + segment_samples]
            # Only use segments with actual speech (simple energy check)
            energy = np.sqrt(np.mean(segment.astype(np.float32) ** 2))
            if energy > 200:  # Above silence threshold for int16
                audio_samples.append(segment)

        if len(audio_samples) < 2:
            return {
                "success": False,
                "error": "Not enough speech detected. Please speak clearly for the full duration.",
            }

        # Enroll with the speaker identifier
        result = self.speaker.enroll_user(audio_samples)

        if result["success"]:
            result["message"] = (
                f"Voice enrolled successfully! Used {result['samples_used']} samples "
                f"with {result['consistency']:.1%} consistency."
            )
        else:
            result["message"] = result.get("error", "Enrollment failed")

        return result

    def is_enrolled(self) -> bool:
        """Check if user has enrolled their voice."""
        return self.speaker.is_enrolled()

    @property
    def is_enrolling(self) -> bool:
        return self._enrolling
