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
import base64
import time
import threading
import uuid
from enum import Enum
from typing import Optional, Dict, Callable, Any, Awaitable

import numpy as np

from .config import AmbientConfig, load_config, save_config
from .companion_logic import (
    analyze_client_turn,
    build_assistant_aliases,
    build_retention_trace,
    sanitize_client_transcript,
)
from src.agents.scheduler import background_scheduler
from src.runtime.session_manager import runtime_session_manager


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

        # Gemini-first defaults for cloud voice mode.
        self._apply_gemini_defaults_if_available()

        # Status
        self._status = AmbientStatus.IDLE
        self._error_message = ""
        self._started_at: Optional[float] = None
        self._client_active_session_id = ""
        self._client_followup_until = 0.0

        # Components (lazy-initialized on start)
        self.audio_capture = None
        self.vad = None
        self.speaker_id = None
        self.conversation = None
        self.enrollment = None
        self.wake_word = None          # Phase 4: Wake word detector
        self.live_orchestrator = None
        self._live_assistant_reply_cb: Optional[Callable[[str, str], Awaitable[str]]] = None
        self._live_retrieve_reply_cb: Optional[Callable[[str, str], Awaitable[str]]] = None

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
        self._live_components_initialized = False

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
        if backend_provider != "gemini" and not self._traditional_stt:
            try:
                from .transcription import Transcriber
                self._traditional_stt = Transcriber(
                    model_size=self.config.whisper_model_size,
                    device=self.config.whisper_device,
                    vram_guard=self.vram_guard,
                )
            except Exception as e:
                return {"success": False, "error": f"Traditional STT init failed: {e}"}
        self.config.stt_provider = provider
        if backend_provider != "gemini":
            self.config.live_mode = "classic"
        elif self._gemini_api_key and self.config.tts_provider == "gemini":
            self.config.live_mode = "gemini_live"
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
        if backend_provider != "gemini" and not self._traditional_tts:
            try:
                from .tts import TextToSpeech
                self._traditional_tts = TextToSpeech(
                    voice=self.config.tts_voice,
                    data_dir=self.data_dir,
                )
            except Exception as e:
                return {"success": False, "error": f"Traditional TTS init failed: {e}"}
        self.config.tts_provider = provider
        if backend_provider != "gemini":
            self.config.live_mode = "classic"
        elif self._gemini_api_key and self.config.stt_provider == "gemini":
            self.config.live_mode = "gemini_live"
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

    def _apply_gemini_defaults_if_available(self):
        """Prefer Gemini providers when API key is configured.

        This keeps startup cloud-first and avoids local model initialization unless
        the user explicitly switches providers.
        """
        if not self._gemini_api_key:
            return

        changed = False
        if self.config.stt_provider != "gemini":
            self.config.stt_provider = "gemini"
            changed = True
        if self.config.tts_provider != "gemini":
            self.config.tts_provider = "gemini"
            changed = True
        if getattr(self.config, "live_mode", "classic") != "gemini_live":
            self.config.live_mode = "gemini_live"
            changed = True

        if changed:
            save_config(self.config, self.data_dir)

    def _should_use_gemini_live(self) -> bool:
        return bool(
            self._gemini_api_key
            and getattr(self.config, "live_mode", "classic") == "gemini_live"
            and self.config.stt_provider == "gemini"
            and self.config.tts_provider == "gemini"
        )

    def set_live_assistant_callback(self, callback: Optional[Callable[[str, str], Awaitable[str]]]):
        """Set async callback used to generate assistant replies for live mode."""
        self._live_assistant_reply_cb = callback
        if self.live_orchestrator:
            self.live_orchestrator.set_assistant_reply_callback(callback)

    def set_live_retrieve_callback(self, callback: Optional[Callable[[str, str], Awaitable[str]]]):
        """Set async callback used for retrieve-only wake phrase responses in live mode."""
        self._live_retrieve_reply_cb = callback
        if self.live_orchestrator:
            self.live_orchestrator.set_retrieve_reply_callback(callback)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> Dict[str, Any]:
        """
        Start the ambient listening pipeline.
        Initializes all tiers and begins capturing audio.
        """
        if self._should_use_gemini_live():
            return await self.start_live()

        # If we previously initialized in live-only mode, force classic init.
        if self.vad is None:
            self._components_initialized = False

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
            if self.live_orchestrator:
                self.audio_capture.remove_frame_callback(self.live_orchestrator.process_audio_frame)
            self.audio_capture.remove_frame_callback(self.vad.process_frame)
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
        if self.live_orchestrator and self.live_orchestrator.running:
            return await self.stop_live()

        if self._status == AmbientStatus.IDLE:
            return {"success": False, "error": "Not running", "status": "idle"}

        # Finalize any in-progress conversation
        if self.conversation:
            await self.conversation.force_finalize()

        # Stop audio capture
        if self.audio_capture:
            if self.vad:
                self.audio_capture.remove_frame_callback(self.vad.process_frame)
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
        if self.live_orchestrator:
            self.live_orchestrator.set_paused(True)
        self._status = AmbientStatus.PAUSED
        return {"success": True, "status": "paused"}

    async def resume(self) -> Dict:
        """Resume from pause."""
        if self._status != AmbientStatus.PAUSED:
            return {"success": False, "error": "Not paused"}

        if self.audio_capture:
            self.audio_capture.start()
        if self.live_orchestrator:
            self.live_orchestrator.set_paused(False)
        self._status = AmbientStatus.LISTENING
        return {"success": True, "status": "listening"}

    async def start_live(self) -> Dict[str, Any]:
        """Start Gemini Live ambient mode (energy gate + Gemini STT/TTS)."""
        if not self._gemini_api_key:
            return {
                "success": False,
                "error": "Gemini API key not configured",
                "status": AmbientStatus.ERROR.value,
            }

        if self.live_orchestrator and self.live_orchestrator.running:
            return {
                "success": False,
                "error": "Live mode already running",
                "status": self._status.value,
            }

        if self._status in (AmbientStatus.LISTENING, AmbientStatus.SPEECH_DETECTED, AmbientStatus.TRANSCRIBING):
            return {
                "success": False,
                "error": "Classic ambient session is active. Stop it before starting live mode.",
                "status": self._status.value,
            }

        self._status = AmbientStatus.LOADING
        self._error_message = ""

        try:
            await self._init_live_components()

            if not self._gemini_stt or not self._gemini_tts:
                raise RuntimeError("Gemini STT/TTS initialization failed for live mode")

            if not self.live_orchestrator:
                raise RuntimeError("Gemini live orchestrator unavailable")

            if self.vad:
                self.audio_capture.remove_frame_callback(self.vad.process_frame)
            self.audio_capture.remove_frame_callback(self.live_orchestrator.process_audio_frame)
            self.audio_capture.add_frame_callback(self.live_orchestrator.process_audio_frame)
            self.audio_capture.start()

            live_result = await self.live_orchestrator.start()
            if not live_result.get("success"):
                raise RuntimeError(live_result.get("error", "Failed to start Gemini Live"))

            self._status = AmbientStatus.LISTENING
            self._started_at = time.time()

            return {
                "success": True,
                "status": self._status.value,
                "mode": "gemini_live",
                "live": self.live_orchestrator.get_status(),
            }

        except Exception as e:
            self._status = AmbientStatus.ERROR
            self._error_message = str(e)
            return {"success": False, "error": str(e), "status": self._status.value}

    async def stop_live(self) -> Dict[str, Any]:
        """Stop Gemini Live ambient mode."""
        if not self.live_orchestrator or not self.live_orchestrator.running:
            return {
                "success": False,
                "error": "Live mode is not running",
                "status": self._status.value,
            }

        if self.audio_capture:
            self.audio_capture.remove_frame_callback(self.live_orchestrator.process_audio_frame)
            self.audio_capture.stop()

        try:
            await self.live_orchestrator.stop()
        except Exception as e:
            self._error_message = str(e)

        if self.conversation:
            await self.conversation.force_finalize()

        if self.speaker_id:
            self.speaker_id.reset_session_clusters()

        self._status = AmbientStatus.IDLE
        self._started_at = None
        return {
            "success": True,
            "status": self._status.value,
            "mode": "gemini_live",
        }

    def get_live_status(self) -> Dict[str, Any]:
        """Get Gemini Live mode status and diagnostics."""
        if self.live_orchestrator:
            return self.live_orchestrator.get_status()
        return {
            "enabled": bool(self._gemini_api_key),
            "running": False,
            "state": "idle_listening",
            "native_live_connected": False,
            "native_live_error": None,
        }

    async def _init_client_companion_components(self):
        """Initialize only the components needed for client-captured companion audio."""
        if self.conversation is None:
            from .conversation import ConversationSegmenter

            self.conversation = ConversationSegmenter(
                ingestion_pipeline=self._ingestion_pipeline,
                auto_ingest=self.config.auto_ingest,
                data_dir=self.data_dir,
                gemini_api_key=self._gemini_api_key,
            )

        backend_stt = "traditional" if self.config.stt_provider == "local" else self.config.stt_provider
        backend_tts = "traditional" if self.config.tts_provider == "local" else self.config.tts_provider

        if backend_stt == "gemini":
            self._init_gemini_stt()
        elif self._traditional_stt is None:
            try:
                from .transcription import Transcriber

                self._traditional_stt = Transcriber(
                    model_size=self.config.whisper_model_size,
                    device=self.config.whisper_device,
                    vram_guard=self.vram_guard,
                )
            except Exception:
                if self._gemini_api_key:
                    self._init_gemini_stt()
                    self.config.stt_provider = "gemini"
                else:
                    raise

        if backend_tts == "gemini":
            self._init_gemini_tts()
        elif self._traditional_tts is None:
            try:
                from .tts import TextToSpeech

                self._traditional_tts = TextToSpeech(
                    voice=self.config.tts_voice,
                    data_dir=self.data_dir,
                )
            except Exception:
                if self._gemini_api_key:
                    self._init_gemini_tts()
                    self.config.tts_provider = "gemini"
                else:
                    raise

    def _estimate_audio_duration_seconds(
        self,
        audio_bytes: bytes,
        mime_type: str,
        estimated_duration_s: float = 0.0,
    ) -> float:
        if estimated_duration_s and estimated_duration_s > 0:
            return round(float(estimated_duration_s), 3)

        if not audio_bytes:
            return 0.0

        normalized_mime = str(mime_type or "").lower()
        if "wav" in normalized_mime and len(audio_bytes) > 44:
            try:
                import io
                import wave

                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 16000
                    return round(frames / rate, 3)
            except Exception:
                pass

        # Best-effort fallback: assume 16-bit mono PCM-equivalent density.
        return round(max(len(audio_bytes) / 32000.0, 0.0), 3)

    async def _synthesize_response_wav(self, text: str) -> bytes:
        if not text.strip() or not self.config.tts_enabled:
            return b""

        tts = self.tts
        if not tts or not getattr(tts, "is_available", False):
            return b""

        try:
            if hasattr(tts, "synthesize_to_wav_async"):
                return (await tts.synthesize_to_wav_async(text)) or b""
            if hasattr(tts, "synthesize_to_wav"):
                return tts.synthesize_to_wav(text) or b""
        except Exception as e:
            self._error_message = str(e)
        return b""

    def _get_metadata_store(self):
        try:
            from src.engine import rag_engine

            return getattr(rag_engine, "metadata_store", None)
        except Exception:
            return None

    def _store_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        thinking: str = "",
        memory_id: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata_store = self._get_metadata_store()
        if not metadata_store or not hasattr(metadata_store, "store_conversation_turn"):
            return

        try:
            metadata_store.store_conversation_turn(
                session_id,
                role,
                content,
                thinking=thinking,
                memory_id=memory_id,
                metadata=metadata or {},
            )
        except TypeError:
            metadata_store.store_conversation_turn(
                session_id,
                role,
                content,
                thinking=thinking,
                memory_id=memory_id,
            )
        except Exception:
            pass

    async def start_client_session(
        self,
        *,
        platform: str = "web",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self._init_client_companion_components()

        session_metadata = {
            "platform": platform,
            "assistant_name": self.config.assistant_name,
            "assistant_aliases": build_assistant_aliases(
                self.config.assistant_name,
                self.config.assistant_aliases,
            ),
            "companion_followup_window_s": int(
                getattr(self.config, "companion_followup_window_s", 45) or 45
            ),
            "companion_followup_until": 0.0,
            "turn_counts": {
                "user": 0,
                "assistant": 0,
            },
            **dict(metadata or {}),
        }
        session = runtime_session_manager.open_session(
            mode="ambient_client",
            metadata=session_metadata,
        )

        self._client_active_session_id = session.session_id
        self._client_followup_until = 0.0
        self._status = AmbientStatus.LISTENING
        self._started_at = time.time()

        if self._ws_broadcast:
            try:
                await self._ws_broadcast(
                    {
                        "type": "session_state",
                        "session_id": session.session_id,
                        "status": self._status.value,
                        "platform": platform,
                    }
                )
            except Exception:
                pass

        return {
            "success": True,
            "session_id": session.session_id,
            "platform": platform,
            "metadata": dict(session.metadata),
            "session": session.to_dict(),
        }

    async def stop_client_session(
        self,
        session_id: str,
        reason: str = "user_request",
    ) -> Dict[str, Any]:
        session = runtime_session_manager.get_session(session_id)
        if session is None:
            return {
                "success": False,
                "error": f"Unknown session: {session_id}",
            }

        if self.conversation:
            await self.conversation.force_finalize()

        closed = runtime_session_manager.close_session(
            session_id,
            reason=reason,
            retention_summary=session.retention_summary,
            agent_tags=session.agent_tags,
        )

        if self._client_active_session_id == session_id:
            self._client_active_session_id = ""
            self._client_followup_until = 0.0

        remaining_active = [
            item
            for item in runtime_session_manager.active_sessions()
            if str(item.get("mode", "")) == "ambient_client"
        ]
        if not remaining_active:
            self._status = AmbientStatus.IDLE
            self._started_at = None

        triggered_agents: list[str] = []
        try:
            triggered_agents = await background_scheduler.trigger_event(
                "ambient_session_closed",
                {
                    "session_id": session_id,
                    "reason": reason,
                    "mode": "ambient_client",
                },
            )
        except Exception:
            triggered_agents = []

        if self._ws_broadcast:
            try:
                await self._ws_broadcast(
                    {
                        "type": "session_state",
                        "session_id": session_id,
                        "status": self._status.value,
                        "reason": reason,
                    }
                )
            except Exception:
                pass

        return {
            "success": True,
            "session": closed.to_dict(),
            "triggered_agents": triggered_agents,
        }

    def get_client_sessions(self) -> Dict[str, Any]:
        sessions = [
            session
            for session in runtime_session_manager.list_sessions(limit=100)
            if str(session.get("mode", "")) == "ambient_client"
        ]
        active = [
            session
            for session in runtime_session_manager.active_sessions()
            if str(session.get("mode", "")) == "ambient_client"
        ]
        return {
            "active_session_id": self._client_active_session_id or (
                active[0]["session_id"] if active else ""
            ),
            "followup_until": self._client_followup_until,
            "active_sessions": active,
            "sessions": sessions,
        }

    async def process_client_audio(
        self,
        *,
        session_id: str,
        audio_bytes: bytes,
        mime_type: str,
        platform: str = "web",
        language: Optional[str] = None,
        estimated_duration_s: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not audio_bytes:
            return {
                "success": False,
                "error": "Audio payload is empty",
                "session_id": session_id,
            }

        await self._init_client_companion_components()

        session = runtime_session_manager.get_session(session_id) if session_id else None
        if session is None:
            started = await self.start_client_session(
                platform=platform,
                metadata=metadata,
            )
            session_id = str(started.get("session_id", "") or "")
            session = runtime_session_manager.get_session(session_id)
        elif metadata:
            runtime_session_manager.update_metadata(
                session_id,
                {
                    "platform": platform,
                    **dict(metadata or {}),
                },
            )

        self._client_active_session_id = session_id
        self._status = AmbientStatus.TRANSCRIBING

        duration_s = self._estimate_audio_duration_seconds(
            audio_bytes,
            mime_type,
            estimated_duration_s=estimated_duration_s,
        )
        transcriber = self.transcriber
        if not transcriber:
            raise RuntimeError("No transcription provider is available for client companion mode")

        if hasattr(transcriber, "transcribe_bytes"):
            result = await transcriber.transcribe_bytes(
                audio_bytes,
                mime_type=mime_type,
                language=language or self.config.whisper_language,
                estimated_duration_s=duration_s,
            )
        else:
            raise RuntimeError("Selected transcription provider does not support encoded audio input")

        text = str(result.get("text", "") or "").strip()
        session_snapshot = runtime_session_manager.get_session(session_id)
        previous_text = ""
        previous_turn_ts = 0.0
        if session_snapshot and isinstance(session_snapshot.metadata, dict):
            previous_text = str(session_snapshot.metadata.get("last_user_text", "") or "")
            try:
                previous_turn_ts = float(session_snapshot.metadata.get("last_user_turn_at", 0.0) or 0.0)
            except (TypeError, ValueError):
                previous_turn_ts = 0.0

        text = sanitize_client_transcript(
            text,
            previous_text=previous_text,
            previous_turn_ts=previous_turn_ts,
            now_ts=time.time(),
        )
        if not text:
            self._status = AmbientStatus.LISTENING
            return {
                "success": True,
                "session_id": session_id,
                "transcript": "",
                "analysis": analyze_client_turn(
                    "",
                    assistant_aliases=build_assistant_aliases(
                        self.config.assistant_name,
                        self.config.assistant_aliases,
                    ),
                    engaged_until=self._client_followup_until,
                ),
                "retention_trace": build_retention_trace(
                    "",
                    session_id=session_id,
                    platform=platform,
                ),
                "assistant_text": "",
                "assistant_audio_base64": "",
            }

        now_ts = time.time()
        analysis = analyze_client_turn(
            text,
            assistant_aliases=build_assistant_aliases(
                self.config.assistant_name,
                self.config.assistant_aliases,
            ),
            engaged_until=self._client_followup_until,
            now_ts=now_ts,
        )
        retention_trace = build_retention_trace(
            text,
            session_id=session_id,
            direct_address=bool(analysis.get("direct_address")),
            retrieval_intent=bool(analysis.get("retrieval_intent")),
            reply_expected=bool(analysis.get("reply_expected")),
            platform=platform,
        )

        if analysis.get("reply_expected"):
            self._client_followup_until = now_ts + float(
                getattr(self.config, "companion_followup_window_s", 45) or 45
            )

        user_turn_id = f"client_{uuid.uuid4().hex[:10]}"
        if self.conversation:
            await self.conversation.add_turn(
                speaker_label="USER",
                speaker_name="You",
                text=text,
                timestamp=now_ts,
                confidence=float(result.get("confidence", 0.0) or 0.0),
                speaker_confidence=1.0,
                live_turn_id=user_turn_id,
                session_id=session_id,
                source_platform=platform,
                retention_trace=retention_trace,
            )

        user_turn_metadata = {
            "platform": platform,
            "source": "client_companion",
            "analysis": analysis,
            "retention_trace": retention_trace,
            "mime_type": mime_type,
            "estimated_duration_s": duration_s,
            "language": str(result.get("language", language or "en") or "en"),
            "stt_confidence": float(result.get("confidence", 0.0) or 0.0),
            **dict(metadata or {}),
        }
        self._store_conversation_turn(
            session_id,
            "user",
            text,
            metadata=user_turn_metadata,
        )

        retention_key = str(retention_trace.get("memory_decision") or "discarded")
        if retention_key not in {"discarded", "session_only", "structured", "priority"}:
            retention_key = "discarded"
        runtime_session_manager.merge_retention_summary(session_id, {retention_key: 1})
        runtime_session_manager.append_agent_tags(
            session_id,
            list(retention_trace.get("tags", []) or []),
        )
        session_snapshot = runtime_session_manager.get_session(session_id)
        turn_counts = dict(
            (session_snapshot.metadata.get("turn_counts") or {})
            if session_snapshot and isinstance(session_snapshot.metadata, dict)
            else {}
        )
        turn_counts["user"] = int(turn_counts.get("user", 0) or 0) + 1
        runtime_session_manager.update_metadata(
            session_id,
            {
                "platform": platform,
                "assistant_name": self.config.assistant_name,
                "last_user_turn_at": now_ts,
                "last_user_text": text,
                "last_analysis": analysis,
                "last_retention_trace": retention_trace,
                "companion_followup_until": self._client_followup_until,
                "turn_counts": turn_counts,
                **dict(metadata or {}),
            },
        )

        if self._ws_broadcast:
            try:
                await self._ws_broadcast(
                    {
                        "type": "live_final_turn",
                        "speaker_label": "USER",
                        "speaker_name": "You",
                        "text": text,
                        "timestamp": now_ts,
                        "confidence": float(result.get("confidence", 0.0) or 0.0),
                        "speaker_confidence": 1.0,
                        "live_turn_id": user_turn_id,
                        "session_id": session_id,
                        "source_platform": platform,
                        "retention_trace": retention_trace,
                    }
                )
            except Exception:
                pass

        assistant_text = ""
        assistant_audio_base64 = ""
        if analysis.get("reply_expected"):
            prompt_text = str(analysis.get("query_text") or text).strip() or text
            reply_callback = None
            if analysis.get("retrieval_intent") and self._live_retrieve_reply_cb:
                reply_callback = self._live_retrieve_reply_cb
            elif self._live_assistant_reply_cb:
                reply_callback = self._live_assistant_reply_cb

            if reply_callback:
                try:
                    assistant_text = str(
                        (await reply_callback(prompt_text, session_id)) or ""
                    ).strip()
                except Exception as e:
                    self._error_message = str(e)
                    assistant_text = ""

            if assistant_text:
                assistant_turn_id = f"assistant_{uuid.uuid4().hex[:10]}"
                assistant_trace = build_retention_trace(
                    assistant_text,
                    session_id=session_id,
                    platform=platform,
                    source="assistant_companion",
                )
                if self.conversation:
                    await self.conversation.add_turn(
                        speaker_label="ASSISTANT",
                        speaker_name=self.config.assistant_name or "Eva",
                        text=assistant_text,
                        timestamp=time.time(),
                        confidence=1.0,
                        speaker_confidence=1.0,
                        live_turn_id=assistant_turn_id,
                        session_id=session_id,
                        source_platform=platform,
                        retention_trace=assistant_trace,
                    )

                assistant_turn_metadata = {
                    "platform": platform,
                    "source": "assistant_companion",
                    "assistant_name": self.config.assistant_name or "Eva",
                    "retention_trace": assistant_trace,
                    "reply_to_turn_id": user_turn_id,
                }
                self._store_conversation_turn(
                    session_id,
                    "assistant",
                    assistant_text,
                    metadata=assistant_turn_metadata,
                )

                session_snapshot = runtime_session_manager.get_session(session_id)
                turn_counts = dict(
                    (session_snapshot.metadata.get("turn_counts") or {})
                    if session_snapshot and isinstance(session_snapshot.metadata, dict)
                    else {}
                )
                turn_counts["assistant"] = int(turn_counts.get("assistant", 0) or 0) + 1
                runtime_session_manager.update_metadata(
                    session_id,
                    {
                        "last_assistant_turn_at": time.time(),
                        "last_assistant_text": assistant_text,
                        "last_assistant_retention_trace": assistant_trace,
                        "turn_counts": turn_counts,
                    },
                )
                assistant_wav = await self._synthesize_response_wav(assistant_text)
                if assistant_wav:
                    assistant_audio_base64 = base64.b64encode(assistant_wav).decode("ascii")

                if self._ws_broadcast:
                    try:
                        await self._ws_broadcast(
                            {
                                "type": "live_final_turn",
                                "speaker_label": "ASSISTANT",
                                "speaker_name": self.config.assistant_name or "Eva",
                                "text": assistant_text,
                                "timestamp": time.time(),
                                "confidence": 1.0,
                                "speaker_confidence": 1.0,
                                "live_turn_id": assistant_turn_id,
                                "session_id": session_id,
                                "source_platform": platform,
                                "retention_trace": assistant_trace,
                            }
                        )
                    except Exception:
                        pass

        self._transcriptions_completed += 1
        self._status = AmbientStatus.LISTENING

        session_snapshot = runtime_session_manager.get_session(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "transcript": text,
            "analysis": analysis,
            "retention_trace": retention_trace,
            "assistant_text": assistant_text,
            "assistant_audio_base64": assistant_audio_base64,
            "assistant_name": self.config.assistant_name,
            "stt_confidence": float(result.get("confidence", 0.0) or 0.0),
            "language": str(result.get("language", language or "en") or "en"),
            "session": session_snapshot.to_dict() if session_snapshot else None,
        }

    # ── Component Initialization ─────────────────────────────────────────

    async def _init_live_components(self):
        """Initialize components required for Gemini Live mode."""
        if self._live_components_initialized and self.live_orchestrator:
            return

        if not self.audio_capture:
            from .audio_capture import AudioCapture
            self.audio_capture = AudioCapture(device=self.config.audio_device)

        if self.speaker_id is None:
            try:
                from .speaker_id import SpeakerIdentifier
                self.speaker_id = SpeakerIdentifier(data_dir=self.data_dir)
            except Exception as e:
                print(f"  ⚠ Speaker ID init failed (live mode, non-critical): {e}")
                self.speaker_id = None

        if not self._gemini_stt:
            self._init_gemini_stt()
        if not self._gemini_tts:
            self._init_gemini_tts()

        if self.conversation is None:
            from .conversation import ConversationSegmenter
            self.conversation = ConversationSegmenter(
                ingestion_pipeline=self._ingestion_pipeline,
                auto_ingest=self.config.auto_ingest,
                data_dir=self.data_dir,
                gemini_api_key=self._gemini_api_key,
            )

        if self.enrollment is None:
            from .enrollment import VoiceEnrollment
            self.enrollment = VoiceEnrollment(self.audio_capture, self.speaker_id)

        if self.live_orchestrator is None:
            from .gemini_live import GeminiLiveSessionOrchestrator

            self.live_orchestrator = GeminiLiveSessionOrchestrator(
                api_key=self._gemini_api_key,
                worker_loop=self._worker_loop,
                gemini_stt=self._gemini_stt,
                gemini_tts=self._gemini_tts,
                conversation=self.conversation,
                audio_capture=self.audio_capture,
                speaker_id=self.speaker_id,
                ws_broadcast=self._ws_broadcast,
                assistant_reply_fn=self._live_assistant_reply_cb,
                retrieve_reply_fn=self._live_retrieve_reply_cb,
                language=self.config.whisper_language,
                energy_threshold=float(getattr(self.config, "energy_gate_threshold", 700.0)),
                min_speech_ms=int(getattr(self.config, "energy_min_speech_ms", 320)),
                silence_ms=int(getattr(self.config, "energy_silence_ms", 420)),
                assistant_aliases=build_assistant_aliases(
                    self.config.assistant_name,
                    self.config.assistant_aliases,
                ),
            )
        else:
            self.live_orchestrator.set_broadcast_callback(self._ws_broadcast)
            self.live_orchestrator.set_assistant_reply_callback(self._live_assistant_reply_cb)
            self.live_orchestrator.set_retrieve_reply_callback(self._live_retrieve_reply_cb)
            self.live_orchestrator.set_assistant_aliases(
                build_assistant_aliases(
                    self.config.assistant_name,
                    self.config.assistant_aliases,
                )
            )

        self._live_components_initialized = True

    async def _init_components(self):
        """Lazy-initialize all voice components."""
        if self._components_initialized:
            return  # Already initialized

        print("\n  🔧 Initializing ambient voice components...")
        t0 = time.time()

        # Tier 0: Audio Capture
        from .audio_capture import AudioCapture
        self.audio_capture = AudioCapture(device=self.config.audio_device)

        use_live_mode = self._should_use_gemini_live()

        # Tier 1: VAD (classic mode only)
        if not use_live_mode:
            from .vad import VoiceActivityDetector
            self.vad = VoiceActivityDetector(threshold=self.config.vad_threshold)
        else:
            self.vad = None

        # Tier 2: Speaker ID (optional — may fail on some Python versions)
        try:
            from .speaker_id import SpeakerIdentifier
            self.speaker_id = SpeakerIdentifier(data_dir=self.data_dir)
        except Exception as e:
            print(f"  ⚠ Speaker ID init failed (non-critical): {e}")
            print("  ↳ Continuing without speaker identification — all speakers labelled UNKNOWN")
            self.speaker_id = None

        # Tier 3: STT — initialize only the selected provider.
        if self.config.stt_provider == "gemini" and self._gemini_api_key:
            self._init_gemini_stt()
        elif self.config.stt_provider in ("traditional", "local"):
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

        # Tier 4: TTS — initialize only the selected provider.
        if self.config.tts_provider == "gemini" and self._gemini_api_key:
            self._init_gemini_tts()
        elif self.config.tts_provider in ("traditional", "local"):
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

            retention_trace = {
                "decision": "keep" if len(text.split()) >= 2 else "discard",
                "reason": "classic_pipeline",
                "score": round(min(max(stt_confidence, 0.0), 1.0), 3),
                "tags": ["ambient", "classic"],
                "source": "classic_ambient",
            }

            # 3. Add to conversation segmenter (may trigger auto-ingest)
            await self.conversation.add_turn(
                speaker_label=speaker_label,
                speaker_name=speaker_name,
                text=text,
                timestamp=start_time,
                confidence=stt_confidence,
                speaker_confidence=speaker_confidence,
                live_turn_id=f"classic_{int(time.time() * 1000)}",
                retention_trace=retention_trace,
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
        if ("stt_provider" in updates or "tts_provider" in updates) and (
            self.config.stt_provider in ("traditional", "local")
            or self.config.tts_provider in ("traditional", "local")
        ):
            self.config.live_mode = "classic"
        if (
            "live_mode" in updates
            and self.config.live_mode == "gemini_live"
            and not self._gemini_api_key
        ):
            self.config.live_mode = "classic"
        if (
            self.config.live_mode != "classic"
            and self.config.live_mode != "gemini_live"
        ):
            self.config.live_mode = "classic"
        if (
            self._gemini_api_key
            and self.config.stt_provider == "gemini"
            and self.config.tts_provider == "gemini"
            and self.config.live_mode == "gemini_live"
        ):
            self._live_components_initialized = False
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
        if self.live_orchestrator and ("assistant_name" in updates or "assistant_aliases" in updates):
            self.live_orchestrator.set_assistant_aliases(
                build_assistant_aliases(
                    self.config.assistant_name,
                    self.config.assistant_aliases,
                )
            )

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
            "operating_mode": "gemini_live" if self._should_use_gemini_live() else "classic",
            "wake_word_enabled": self.config.wake_word_enabled,
            "wake_word_active": self.wake_word.is_running if self.wake_word else False,
            "no_local_model_policy_enforced": self.config.stt_provider == "gemini" and self.config.tts_provider == "gemini",
            "local_models_initialized": {
                "vad": self.vad is not None,
                "traditional_stt": self._traditional_stt is not None,
                "traditional_tts": self._traditional_tts is not None,
            },
            "live": self.get_live_status(),
            "client_session": self.get_client_sessions(),
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
        if self.live_orchestrator:
            self.live_orchestrator.set_broadcast_callback(callback)
