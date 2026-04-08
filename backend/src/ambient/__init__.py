"""
AmbientService — Orchestrates all voice tiers into a single start/stop pipeline.

Tier 0: AudioCapture (Microphone + Ring Buffer)
Tier 1: VoiceActivityDetector (Silero VAD v5)
Tier 2: SpeakerIdentifier (ECAPA-TDNN)
Tier 3: Transcriber — provider routing:
    - Traditional: faster-whisper (CTranslate2), local GPU/CPU
    - Local alias: maps to the traditional local stack
    - Gemini: Google Gemini API multimodal audio transcription
Tier 4: TextToSpeech — provider routing:
    - Traditional: Piper TTS (ONNX), local CPU
    - Local alias: maps to the traditional local stack
    - Gemini: Google Gemini API audio output generation

Flow:
  Mic → VAD → [speech segment] → SpeakerID + STT → ConversationSegmenter → RAG Ingest

Provider selection configured via stt_provider / tts_provider in AmbientConfig.
Accepted values: "traditional" | "local" | "gemini".
"""

import asyncio
import time
import threading
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
    Supports dual STT/TTS providers: traditional (local) and Gemini (cloud).
    """

    def __init__(self, ingestion_pipeline=None, data_dir: str = "data",
                 gemini_api_key: str = None):
        self.data_dir = data_dir
        self.config = load_config(data_dir)
        self._gemini_api_key = gemini_api_key

        # Status
        self._status = AmbientStatus.IDLE
        self._error_message = ""
        self._started_at: Optional[float] = None

        # Components (lazy-initialized on start)
        self.audio_capture = None
        self.vad = None
        self.speaker_id = None
        self.conversation = None
        self.enrollment = None
        self.wake_word = None          # Phase 4: Wake word detector

        # Dual STT providers
        self._traditional_stt = None   # faster-whisper Transcriber
        self._gemini_stt = None        # GeminiSTT

        # Dual TTS providers
        self._traditional_tts = None   # Piper TextToSpeech
        self._gemini_tts = None        # GeminiTTS

        # Store pipeline reference for deferred init
        self._ingestion_pipeline = ingestion_pipeline

        # VRAMGuard singleton
        from .vram_guard import vram_guard
        self.vram_guard = vram_guard

        # WebSocket broadcast callback (set by server.py)
        self._ws_broadcast: Optional[Callable] = None

        # Persistent worker event loop (§8.1) — eliminates per-segment thread/loop creation
        self._worker_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._worker_thread = threading.Thread(
            target=self._worker_loop.run_forever,
            daemon=True,
            name="ambient-worker",
        )
        self._worker_thread.start()

        # Stats
        self._speech_segments_processed = 0
        self._transcriptions_completed = 0

        # Tracks whether _init_components completed fully
        self._components_initialized = False

    # ── Provider Properties ──────────────────────────────────────────────

    @property
    def transcriber(self):
        """Active STT provider based on config."""
        if self.config.stt_provider == "gemini" and self._gemini_stt:
            return self._gemini_stt
        return self._traditional_stt

    @property
    def tts(self):
        """Active TTS provider based on config."""
        if self.config.tts_provider == "gemini" and self._gemini_tts:
            return self._gemini_tts
        return self._traditional_tts

    def get_stt_provider(self) -> str:
        """Return current active STT provider name."""
        return self.config.stt_provider

    def get_tts_provider(self) -> str:
        """Return current active TTS provider name."""
        return self.config.tts_provider

    def set_stt_provider(self, provider: str) -> Dict[str, Any]:
        """Switch STT provider. Returns status."""
        if provider not in ("traditional", "local", "gemini"):
            return {"success": False, "error": f"Unknown STT provider: {provider}"}
        backend_provider = "traditional" if provider == "local" else provider
        if backend_provider == "gemini" and not self._gemini_api_key:
            return {"success": False, "error": "Gemini API key not configured"}
        if backend_provider == "gemini" and not self._gemini_stt:
            self._init_gemini_stt()
        self.config.stt_provider = provider
        save_config(self.config, self.data_dir)
        return {
            "success": True,
            "stt_provider": provider,
            "active_backend": backend_provider,
        }

    def set_tts_provider(self, provider: str) -> Dict[str, Any]:
        """Switch TTS provider. Returns status."""
        if provider not in ("traditional", "local", "gemini"):
            return {"success": False, "error": f"Unknown TTS provider: {provider}"}
        backend_provider = "traditional" if provider == "local" else provider
        if backend_provider == "gemini" and not self._gemini_api_key:
            return {"success": False, "error": "Gemini API key not configured"}
        if backend_provider == "gemini" and not self._gemini_tts:
            self._init_gemini_tts()
        self.config.tts_provider = provider
        save_config(self.config, self.data_dir)
        return {
            "success": True,
            "tts_provider": provider,
            "active_backend": backend_provider,
        }

    def _init_gemini_stt(self):
        """Initialize Gemini STT if not already done."""
        if self._gemini_stt or not self._gemini_api_key:
            return
        from .gemini_voice import GeminiSTT
        self._gemini_stt = GeminiSTT(api_key=self._gemini_api_key)
        print("  ✅ Gemini STT initialized")

    def _init_gemini_tts(self):
        """Initialize Gemini TTS if not already done."""
        if self._gemini_tts or not self._gemini_api_key:
            return
        from .gemini_voice import GeminiTTS
        self._gemini_tts = GeminiTTS(
            api_key=self._gemini_api_key,
            voice=self.config.gemini_tts_voice,
        )
        print("  ✅ Gemini TTS initialized")

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
        if self._components_initialized:
            return  # Already initialized

        print("\n  🔧 Initializing ambient voice components...")
        t0 = time.time()

        # Tier 0: Audio Capture
        from .audio_capture import AudioCapture
        self.audio_capture = AudioCapture(device=self.config.audio_device)

        # Tier 1: VAD
        from .vad import VoiceActivityDetector
        self.vad = VoiceActivityDetector(threshold=self.config.vad_threshold)

        # Tier 2: Speaker ID (optional — may fail on some Python versions)
        try:
            from .speaker_id import SpeakerIdentifier
            self.speaker_id = SpeakerIdentifier(data_dir=self.data_dir)
        except Exception as e:
            print(f"  ⚠ Speaker ID init failed (non-critical): {e}")
            print("  ↳ Continuing without speaker identification — all speakers labelled UNKNOWN")
            self.speaker_id = None

        # Tier 3: STT — initialize based on selected provider
        if self.config.stt_provider == "gemini" and self._gemini_api_key:
            self._init_gemini_stt()
        else:
            # Traditional: faster-whisper
            try:
                from .transcription import Transcriber
                self._traditional_stt = Transcriber(
                    model_size=self.config.whisper_model_size,
                    device=self.config.whisper_device,
                    vram_guard=self.vram_guard,
                )
            except Exception as e:
                print(f"  ⚠ Traditional STT (faster-whisper) init failed: {e}")
                # Fall back to Gemini if available
                if self._gemini_api_key:
                    print("  ↳ Falling back to Gemini STT")
                    self._init_gemini_stt()
                    self.config.stt_provider = "gemini"

        # Also init Gemini STT if API key is present (for easy switching)
        if self._gemini_api_key and not self._gemini_stt:
            self._init_gemini_stt()

        # Tier 4: TTS — initialize based on selected provider
        if self.config.tts_provider == "gemini" and self._gemini_api_key:
            self._init_gemini_tts()
        else:
            # Traditional: Piper TTS
            try:
                from .tts import TextToSpeech
                self._traditional_tts = TextToSpeech(
                    voice=self.config.tts_voice,
                    data_dir=self.data_dir,
                )
            except Exception as e:
                print(f"  ⚠ Traditional TTS (Piper) init failed: {e}")
                if self._gemini_api_key:
                    print("  ↳ Falling back to Gemini TTS")
                    self._init_gemini_tts()
                    self.config.tts_provider = "gemini"

        # Also init Gemini TTS if API key is present
        if self._gemini_api_key and not self._gemini_tts:
            self._init_gemini_tts()

        # Conversation Segmenter (Phase 2 + 5 + 6: Gemini summarizer + topic seg + dual storage)
        from .conversation import ConversationSegmenter
        self.conversation = ConversationSegmenter(
            ingestion_pipeline=self._ingestion_pipeline,
            auto_ingest=self.config.auto_ingest,
            data_dir=self.data_dir,
            gemini_api_key=self._gemini_api_key,
        )

        # Voice Enrollment
        from .enrollment import VoiceEnrollment
        self.enrollment = VoiceEnrollment(self.audio_capture, self.speaker_id)

        # Phase 4: Wake Word Detection
        if self.config.wake_word_enabled:
            try:
                from .wake_word import WakeWordDetector
                self.wake_word = WakeWordDetector(
                    wake_word=self.config.wake_word_model,
                    threshold=self.config.wake_word_threshold,
                )
            except Exception as e:
                print(f"  ⚠ Wake word init failed (non-critical): {e}")
                self.wake_word = None

        self._components_initialized = True

        elapsed = time.time() - t0
        stt_name = self.config.stt_provider
        tts_name = self.config.tts_provider
        spk = "yes" if self.speaker_id else "disabled"
        print(f"  ✅ All ambient components initialized in {elapsed:.1f}s")
        print(f"     STT: {stt_name} | TTS: {tts_name} | SpeakerID: {spk}\n")

    # ── Speech Processing Pipeline ───────────────────────────────────────

    def _on_speech_segment(self, audio: 'np.ndarray', start_time: float,
                           end_time: float):
        """
        Called by VAD when a complete speech segment is detected.
        Runs speaker ID + transcription on the persistent worker event loop (§8.1).
        Note: This is called from the audio callback thread, NOT the event loop.
        """
        self._speech_segments_processed += 1
        self._status = AmbientStatus.SPEECH_DETECTED

        # Schedule on the persistent worker loop — no new threads/loops created
        asyncio.run_coroutine_threadsafe(
            self._process_speech(audio, start_time, end_time),
            self._worker_loop,
        )

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

            # 1. Speaker ID (runs on CPU, fast) — optional
            if self.speaker_id:
                speaker_label, speaker_confidence = self.speaker_id.identify(audio)
                speaker_name = self.speaker_id.get_display_name(speaker_label)
            else:
                speaker_label, speaker_confidence = "UNKNOWN", 0.0
                speaker_name = "Speaker"

            # 2. Transcription (may wait for VRAM guard)
            result = await self.transcriber.transcribe(
                audio, language=self.config.whisper_language
            )

            text = result.get("text", "").strip()
            stt_confidence = result.get("confidence", 0.0)

            if not text:
                self._status = AmbientStatus.LISTENING
                return

            # 2.5 Phase 1: Speech cleanup — remove fillers, disfluencies, low-conf junk
            from .speech_cleanup import clean_transcript, get_word_confidences_from_segments
            word_confs = get_word_confidences_from_segments(result.get("segments", []))
            cleaned_text = clean_transcript(text, stt_confidence, word_confs)
            if not cleaned_text:
                self._status = AmbientStatus.LISTENING
                return
            text = cleaned_text

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
        Note: Called from audio thread. We only broadcast occasionally.
        Uses persistent worker loop instead of spawning threads (§8.1, §8.2)."""
        # Only broadcast every 5th frame to reduce traffic (~6 updates/sec)
        frame_idx = int(timestamp * 1000 / 30)
        if frame_idx % 5 != 0 or not self._ws_broadcast:
            return

        # Fire-and-forget on the persistent worker loop — no new threads
        asyncio.run_coroutine_threadsafe(
            self._broadcast_vad(speech_prob, timestamp),
            self._worker_loop,
        )

    async def _broadcast_vad(self, speech_prob: float, timestamp: float):
        """Send VAD activity to WebSocket clients (runs on worker loop)."""
        try:
            await self._ws_broadcast({
                "type": "vad_activity",
                "speech_prob": round(speech_prob, 2),
                "timestamp": round(timestamp, 2),
            })
        except Exception:
            pass

    # ── Configuration ────────────────────────────────────────────────────

    def update_config(self, updates: Dict) -> AmbientConfig:
        """Update ambient config and save."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Apply live updates
        if self.vad and "vad_threshold" in updates:
            self.vad.set_threshold(updates["vad_threshold"])

        # Handle provider switching
        if "stt_provider" in updates:
            self.set_stt_provider(updates["stt_provider"])
        if "tts_provider" in updates:
            self.set_tts_provider(updates["tts_provider"])
        if "gemini_tts_voice" in updates and self._gemini_tts:
            self._gemini_tts.set_voice(updates["gemini_tts_voice"])
        if "wake_word_enabled" in updates:
            if updates["wake_word_enabled"] and not self.wake_word:
                try:
                    from .wake_word import WakeWordDetector
                    self.wake_word = WakeWordDetector(
                        wake_word=self.config.wake_word_model,
                        threshold=self.config.wake_word_threshold,
                    )
                except Exception:
                    pass
            if self.wake_word:
                if updates["wake_word_enabled"]:
                    self.wake_word.start()
                else:
                    self.wake_word.stop()
        if "wake_word_threshold" in updates and self.wake_word:
            self.wake_word.set_threshold(updates["wake_word_threshold"])

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
            "stt_provider": self.config.stt_provider,
            "tts_provider": self.config.tts_provider,
            "gemini_available": self._gemini_api_key is not None,
            "wake_word_enabled": self.config.wake_word_enabled,
            "wake_word_active": self.wake_word.is_running if self.wake_word else False,
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
            if self.wake_word:
                status["wake_word"] = self.wake_word.get_stats()
            status["vram_guard"] = self.vram_guard.get_stats()

        return status

    # ── WebSocket ────────────────────────────────────────────────────────

    def set_ws_broadcast(self, callback: Callable):
        """Set WebSocket broadcast callback (called by server.py)."""
        self._ws_broadcast = callback
