"""
AmbientService — Orchestrates all voice tiers into a single start/stop pipeline.

Tier 0: AudioCapture (Microphone + Ring Buffer)
Tier 1: VoiceActivityDetector (Silero VAD v5)
Tier 2: SpeakerIdentifier (ECAPA-TDNN)
Tier 3: Transcriber (faster-whisper)
Tier 4: TextToSpeech (Piper TTS)
  +     ConversationSegmenter → MemoryIngestionPipeline (existing RAG bridge)

Flow:
  Mic → VAD → [speech segment] → SpeakerID + Whisper → ConversationSegmenter → RAG Ingest

All models run locally. Zero cloud. Zero API keys.
"""

import asyncio
import time
from enum import Enum
from typing import Optional, Dict, Callable, Any

from .config import AmbientConfig, load_config, save_config


class AmbientStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LISTENING = "listening"
    SPEECH_DETECTED = "speech_detected"
    TRANSCRIBING = "transcribing"
    PAUSED = "paused"
    ERROR = "error"


class AmbientService:
    """
    Top-level orchestrator for ambient voice listening.
    Provides a single start/stop interface for the entire voice pipeline.
    """

    def __init__(self, ingestion_pipeline=None, data_dir: str = "data"):
        self.data_dir = data_dir
        self.config = load_config(data_dir)

        # Status
        self._status = AmbientStatus.IDLE
        self._error_message = ""
        self._started_at: Optional[float] = None

        # Components (lazy-initialized on start)
        self.audio_capture = None
        self.vad = None
        self.speaker_id = None
        self.transcriber = None
        self.tts = None
        self.conversation = None
        self.enrollment = None

        # Store pipeline reference for deferred init
        self._ingestion_pipeline = ingestion_pipeline

        # VRAMGuard singleton
        from .vram_guard import vram_guard
        self.vram_guard = vram_guard

        # WebSocket broadcast callback (set by server.py)
        self._ws_broadcast: Optional[Callable] = None

        # Stats
        self._speech_segments_processed = 0
        self._transcriptions_completed = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> Dict[str, Any]:
        """
        Start the ambient listening pipeline.
        Initializes all tiers and begins capturing audio.
        """
        if self._status in (AmbientStatus.LISTENING, AmbientStatus.SPEECH_DETECTED,
                            AmbientStatus.TRANSCRIBING):
            return {"success": False, "error": "Already running",
                    "status": self._status.value}

        self._status = AmbientStatus.LOADING
        self._error_message = ""

        try:
            # Initialize components (lazy — only load on first start)
            await self._init_components()

            # Wire the pipeline:
            # AudioCapture → VAD → _on_speech_segment → SpeakerID + Whisper → Conversation
            self.audio_capture.add_frame_callback(self.vad.process_frame)
            self.vad.set_speech_callback(self._on_speech_segment)
            self.vad.set_activity_callback(self._on_vad_activity)

            # Start audio capture
            self.audio_capture.start()

            self._status = AmbientStatus.LISTENING
            self._started_at = time.time()

            print("\n  🎧 Ambient listening STARTED")
            return {"success": True, "status": self._status.value}

        except Exception as e:
            self._status = AmbientStatus.ERROR
            self._error_message = str(e)
            print(f"  ❌ Ambient start error: {e}")
            return {"success": False, "error": str(e), "status": self._status.value}

    async def stop(self) -> Dict[str, Any]:
        """Stop the ambient listening pipeline."""
        if self._status == AmbientStatus.IDLE:
            return {"success": False, "error": "Not running", "status": "idle"}

        # Finalize any in-progress conversation
        if self.conversation:
            await self.conversation.force_finalize()

        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop()

        # Reset VAD state
        if self.vad:
            self.vad.reset()

        # Reset speaker clusters for next session
        if self.speaker_id:
            self.speaker_id.reset_session_clusters()

        self._status = AmbientStatus.IDLE
        self._started_at = None

        print("  🛑 Ambient listening STOPPED")
        return {"success": True, "status": self._status.value}

    async def pause(self) -> Dict:
        """Pause listening (keeps models loaded)."""
        if self._status not in (AmbientStatus.LISTENING, AmbientStatus.SPEECH_DETECTED):
            return {"success": False, "error": "Not in a pauseable state"}

        if self.audio_capture:
            self.audio_capture.stop()
        self._status = AmbientStatus.PAUSED
        return {"success": True, "status": "paused"}

    async def resume(self) -> Dict:
        """Resume from pause."""
        if self._status != AmbientStatus.PAUSED:
            return {"success": False, "error": "Not paused"}

        if self.audio_capture:
            self.audio_capture.start()
        self._status = AmbientStatus.LISTENING
        return {"success": True, "status": "listening"}

    # ── Component Initialization ─────────────────────────────────────────

    async def _init_components(self):
        """Lazy-initialize all voice components."""
        if self.audio_capture is not None:
            return  # Already initialized

        print("\n  🔧 Initializing ambient voice components...")
        t0 = time.time()

        # Tier 0: Audio Capture
        from .audio_capture import AudioCapture
        self.audio_capture = AudioCapture(device=self.config.audio_device)

        # Tier 1: VAD
        from .vad import VoiceActivityDetector
        self.vad = VoiceActivityDetector(threshold=self.config.vad_threshold)

        # Tier 2: Speaker ID
        from .speaker_id import SpeakerIdentifier
        self.speaker_id = SpeakerIdentifier(data_dir=self.data_dir)

        # Tier 3: Transcriber
        from .transcription import Transcriber
        self.transcriber = Transcriber(
            model_size=self.config.whisper_model_size,
            device=self.config.whisper_device,
            vram_guard=self.vram_guard,
        )

        # Tier 4: TTS
        from .tts import TextToSpeech
        self.tts = TextToSpeech(
            voice=self.config.tts_voice,
            data_dir=self.data_dir,
        )

        # Conversation Segmenter
        from .conversation import ConversationSegmenter
        self.conversation = ConversationSegmenter(
            ingestion_pipeline=self._ingestion_pipeline,
            auto_ingest=self.config.auto_ingest,
            data_dir=self.data_dir,
        )

        # Voice Enrollment
        from .enrollment import VoiceEnrollment
        self.enrollment = VoiceEnrollment(self.audio_capture, self.speaker_id)

        elapsed = time.time() - t0
        print(f"  ✅ All ambient components initialized in {elapsed:.1f}s\n")

    # ── Speech Processing Pipeline ───────────────────────────────────────

    def _on_speech_segment(self, audio: 'np.ndarray', start_time: float,
                           end_time: float):
        """
        Called by VAD when a complete speech segment is detected.
        Runs speaker ID + transcription asynchronously.
        Note: This is called from the audio callback thread, NOT the event loop.
        """
        self._speech_segments_processed += 1
        self._status = AmbientStatus.SPEECH_DETECTED

        # Schedule the async processing on the main event loop
        import threading

        def _run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._process_speech(audio, start_time, end_time)
                )
            except Exception as e:
                print(f"  ⚠ Speech processing thread error: {e}")
            finally:
                loop.close()

        threading.Thread(target=_run_in_thread, daemon=True).start()

    async def _process_speech(self, audio: 'np.ndarray', start_time: float,
                               end_time: float):
        """
        Full speech processing pipeline:
        1. Speaker Identification (CPU, ~50ms)
        2. Transcription (GPU/CPU, ~1-4s)
        3. Add to conversation segmenter
        """
        import numpy as np

        try:
            self._status = AmbientStatus.TRANSCRIBING

            # 1. Speaker ID (runs on CPU, fast)
            speaker_label, speaker_confidence = self.speaker_id.identify(audio)
            speaker_name = self.speaker_id.get_display_name(speaker_label)

            # 2. Transcription (may wait for VRAM guard)
            result = await self.transcriber.transcribe(
                audio, language=self.config.whisper_language
            )

            text = result.get("text", "").strip()
            stt_confidence = result.get("confidence", 0.0)

            if not text:
                self._status = AmbientStatus.LISTENING
                return

            self._transcriptions_completed += 1

            # 3. Add to conversation segmenter (may trigger auto-ingest)
            await self.conversation.add_turn(
                speaker_label=speaker_label,
                speaker_name=speaker_name,
                text=text,
                timestamp=start_time,
                confidence=stt_confidence,
            )

            # 4. Broadcast to WebSocket clients
            if self._ws_broadcast:
                try:
                    await self._ws_broadcast({
                        "type": "transcript",
                        "speaker_label": speaker_label,
                        "speaker_name": speaker_name,
                        "text": text,
                        "timestamp": start_time,
                        "confidence": stt_confidence,
                        "speaker_confidence": speaker_confidence,
                    })
                except Exception:
                    pass

            self._status = AmbientStatus.LISTENING

        except Exception as e:
            print(f"  ⚠ Speech processing error: {e}")
            self._status = AmbientStatus.LISTENING

    def _on_vad_activity(self, speech_prob: float, timestamp: float):
        """Called by VAD for each frame — used for UI live indicator.
        Note: Called from audio thread. We only broadcast occasionally."""
        # Only broadcast every 5th frame to reduce traffic (~6 updates/sec)
        frame_idx = int(timestamp * 1000 / 30)
        if frame_idx % 5 != 0 or not self._ws_broadcast:
            return

        # Fire-and-forget broadcast (don't block audio thread)
        import threading

        def _send():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._ws_broadcast({
                    "type": "vad_activity",
                    "speech_prob": round(speech_prob, 2),
                    "timestamp": round(timestamp, 2),
                }))
            except Exception:
                pass
            finally:
                loop.close()

        threading.Thread(target=_send, daemon=True).start()

    # ── Configuration ────────────────────────────────────────────────────

    def update_config(self, updates: Dict) -> AmbientConfig:
        """Update ambient config and save."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Apply live updates
        if self.vad and "vad_threshold" in updates:
            self.vad.set_threshold(updates["vad_threshold"])

        save_config(self.config, self.data_dir)
        return self.config

    def get_config(self) -> Dict:
        return self.config.to_dict()

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get comprehensive ambient service status."""
        uptime = (time.time() - self._started_at) if self._started_at else 0

        status = {
            "status": self._status.value,
            "uptime_seconds": round(uptime, 1),
            "error": self._error_message or None,
            "enrolled": self.speaker_id.is_enrolled() if self.speaker_id else False,
            "tts_available": self.tts.is_available if self.tts else False,
            "audio_level": self.audio_capture.get_audio_level() if self.audio_capture else 0,
            "speech_segments": self._speech_segments_processed,
            "transcriptions": self._transcriptions_completed,
        }

        # Component stats
        if self._status != AmbientStatus.IDLE:
            if self.vad:
                status["vad"] = self.vad.get_stats()
            if self.speaker_id:
                status["speaker_id"] = self.speaker_id.get_stats()
            if self.transcriber:
                status["transcriber"] = self.transcriber.get_stats()
            if self.conversation:
                status["conversation"] = self.conversation.get_stats()
            if self.tts:
                status["tts"] = self.tts.get_stats()
            status["vram_guard"] = self.vram_guard.get_stats()

        return status

    # ── WebSocket ────────────────────────────────────────────────────────

    def set_ws_broadcast(self, callback: Callable):
        """Set WebSocket broadcast callback (called by server.py)."""
        self._ws_broadcast = callback
