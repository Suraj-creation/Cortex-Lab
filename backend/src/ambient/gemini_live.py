"""
Gemini Live ambient orchestration.

Provides a cloud-first, always-listening runtime that avoids local model downloads
by default. It uses an energy gate for speech detection, optional speaker
verification, Gemini STT/TTS, and emits real-time websocket events.

This module intentionally keeps a fallback path:
- If native Gemini Live transport is available in the installed SDK, audio/control
  events are forwarded through that channel.
- If native transport is unavailable, the orchestrator still operates by using
  Gemini STT + Gemini TTS wrappers.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import math
import re
import time
import uuid
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

import numpy as np


LiveBroadcast = Callable[[Dict[str, Any]], Awaitable[None]]
AssistantReplyFn = Callable[[str, str], Awaitable[str]]
RetrieveReplyFn = Callable[[str, str], Awaitable[str]]


class LiveSessionState:
    IDLE_LISTENING = "idle_listening"
    USER_DETECTED = "user_detected"
    LIVE_STREAMING = "live_streaming"
    ASSISTANT_RESPONDING = "assistant_responding"
    BACKGROUND_PROCESSING = "background_processing"
    DEGRADED = "degraded"


class LiveInteractionMode:
    CAPTURE = "capture"
    RETRIEVE = "retrieve"


class GeminiLiveClient:
    """
    Thin Gemini Live transport wrapper with capability detection.

    The Google SDK has evolved across versions, so this wrapper probes for
    available methods at runtime and degrades gracefully.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-live-preview"):
        self.api_key = api_key
        self.model_name = model_name

        self._client = None
        self._session_cm = None
        self._session = None
        self._connected = False
        self._last_error: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self._connected and self._session is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    async def connect(self) -> bool:
        if self.connected:
            return True
        if not self.api_key:
            self._last_error = "Missing Gemini API key"
            return False

        try:
            from google import genai
        except Exception as exc:
            self._last_error = f"google-genai unavailable: {exc}"
            return False

        try:
            self._client = genai.Client(api_key=self.api_key)
            aio_client = getattr(self._client, "aio", None)
            live_api = getattr(aio_client, "live", None) if aio_client else None
            connect_fn = getattr(live_api, "connect", None) if live_api else None
            if not callable(connect_fn):
                self._last_error = "Gemini SDK does not expose aio.live.connect"
                await self.close()
                return False

            connect_variants = [
                {
                    "model": self.model_name,
                    "config": {
                        "response_modalities": ["TEXT", "AUDIO"],
                    },
                },
                {
                    "model": self.model_name,
                },
            ]

            last_exc: Optional[Exception] = None
            for kwargs in connect_variants:
                try:
                    self._session_cm = connect_fn(**kwargs)
                    self._session = await self._session_cm.__aenter__()
                    self._connected = True
                    self._last_error = None
                    return True
                except Exception as exc:  # pragma: no cover - capability probing
                    last_exc = exc
                    self._session_cm = None
                    self._session = None

            self._last_error = f"Live connect failed: {last_exc}"
            await self.close()
            return False

        except Exception as exc:
            self._last_error = f"Live init failed: {exc}"
            await self.close()
            return False

    async def close(self) -> None:
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._session_cm = None
        self._session = None
        self._connected = False

    async def send_audio_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Send a PCM16 chunk through whatever live transport signature is supported."""
        if not self.connected:
            return False

        pcm = np.asarray(audio, dtype=np.int16).tobytes()
        mime_type = f"audio/pcm;rate={sample_rate}"

        # Variant A: send_realtime_input(...)
        fn = getattr(self._session, "send_realtime_input", None)
        if callable(fn):
            variants = [
                {"audio": pcm, "mime_type": mime_type},
                {"input": {"data": pcm, "mime_type": mime_type}},
                {"audio": {"data": pcm, "mime_type": mime_type}},
            ]
            for kwargs in variants:
                try:
                    await fn(**kwargs)
                    return True
                except TypeError:
                    continue
                except Exception as exc:
                    self._last_error = f"send_realtime_input failed: {exc}"
                    break

        # Variant B: send(payload)
        fn = getattr(self._session, "send", None)
        if callable(fn):
            payload_variants = [
                {
                    "mime_type": mime_type,
                    "data": base64.b64encode(pcm).decode("ascii"),
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(pcm).decode("ascii"),
                    }
                },
            ]
            for payload in payload_variants:
                try:
                    await fn(payload)
                    return True
                except TypeError:
                    continue
                except Exception as exc:
                    self._last_error = f"send failed: {exc}"
                    break

        self._last_error = self._last_error or "No compatible live send method"
        return False

    async def send_control_event(self, event: Dict[str, Any]) -> bool:
        """Send a control event when supported by the live transport."""
        if not self.connected:
            return False

        fn = getattr(self._session, "send_client_content", None)
        if callable(fn):
            for payload in ({"content": event}, event):
                try:
                    await fn(payload)
                    return True
                except TypeError:
                    continue
                except Exception as exc:
                    self._last_error = f"send_client_content failed: {exc}"
                    break

        fn = getattr(self._session, "send", None)
        if callable(fn):
            try:
                await fn({"control": event})
                return True
            except Exception as exc:
                self._last_error = f"send control failed: {exc}"

        return False

    async def receive_events(self, *, max_events: int = 8) -> List[Dict[str, Any]]:
        """Best-effort receive for supported live session implementations."""
        if not self.connected:
            return []

        fn = getattr(self._session, "receive", None)
        if not callable(fn):
            return []

        events: List[Dict[str, Any]] = []
        try:
            received = fn()

            if hasattr(received, "__aiter__"):
                count = 0
                async for event in received:
                    events.append(self._coerce_event(event))
                    count += 1
                    if count >= max_events:
                        break
                return events

            if inspect.isawaitable(received):
                event = await asyncio.wait_for(received, timeout=0.05)
                events.append(self._coerce_event(event))
                return events

            if received is not None:
                events.append(self._coerce_event(received))
                return events

        except asyncio.TimeoutError:
            return []
        except Exception as exc:
            self._last_error = f"receive failed: {exc}"

        return []

    @staticmethod
    def _coerce_event(event: Any) -> Dict[str, Any]:
        if isinstance(event, dict):
            return event

        payload: Dict[str, Any] = {}
        for attr in (
            "type",
            "event_type",
            "text",
            "partial_text",
            "turn_complete",
            "mime_type",
            "audio",
            "audio_data",
        ):
            if hasattr(event, attr):
                payload[attr] = getattr(event, attr)

        if not payload:
            payload["raw"] = str(event)
        return payload


class GeminiLiveSessionOrchestrator:
    """Cloud-first full-duplex ambient session manager."""

    def __init__(
        self,
        *,
        api_key: str,
        worker_loop: asyncio.AbstractEventLoop,
        gemini_stt: Any,
        gemini_tts: Any,
        conversation: Any,
        audio_capture: Any,
        speaker_id: Any = None,
        ws_broadcast: Optional[LiveBroadcast] = None,
        assistant_reply_fn: Optional[AssistantReplyFn] = None,
        retrieve_reply_fn: Optional[RetrieveReplyFn] = None,
        language: Optional[str] = None,
        energy_threshold: float = 700.0,
        min_speech_ms: int = 320,
        silence_ms: int = 420,
    ):
        self._api_key = api_key
        self._worker_loop = worker_loop
        self._gemini_stt = gemini_stt
        self._gemini_tts = gemini_tts
        self._conversation = conversation
        self._audio_capture = audio_capture
        self._speaker_id = speaker_id
        self._ws_broadcast = ws_broadcast
        self._assistant_reply_fn = assistant_reply_fn
        self._retrieve_reply_fn = retrieve_reply_fn
        self._language = language

        self._frame_ms = int(getattr(audio_capture, "FRAME_MS", 32) or 32)
        self._sample_rate = int(getattr(audio_capture, "SAMPLE_RATE", 16000) or 16000)

        self._energy_threshold = float(max(150.0, energy_threshold))
        self._min_speech_frames = max(3, int(min_speech_ms / self._frame_ms))
        self._silence_frames = max(3, int(silence_ms / self._frame_ms))

        self._prebuffer: Deque[np.ndarray] = deque(maxlen=max(4, int(220 / self._frame_ms)))
        self._speech_frames: List[np.ndarray] = []
        self._speech_active = False
        self._silence_count = 0

        self._running = False
        self._paused = False
        self._state = LiveSessionState.IDLE_LISTENING
        self._state_updated_at = time.time()
        self._last_error = ""
        self._interaction_mode = LiveInteractionMode.CAPTURE
        self._retrieve_mode_armed = False
        self._retrieve_mode_updated_at = time.time()

        self._session_id = ""
        self._session_started_at: Optional[float] = None

        self._stats = {
            "segments_detected": 0,
            "user_turns": 0,
            "assistant_turns": 0,
            "memory_jobs": 0,
            "native_events_seen": 0,
            "audio_frames": 0,
            "last_audio_level": 0.0,
        }

        self._memory_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=256)
        self._memory_worker_task: Optional[asyncio.Task] = None
        self._native_events_task: Optional[asyncio.Task] = None

        self._live_client = GeminiLiveClient(api_key=api_key)
        self._native_live_connected = False

    @property
    def running(self) -> bool:
        return self._running

    def set_broadcast_callback(self, callback: Optional[LiveBroadcast]) -> None:
        self._ws_broadcast = callback

    def set_assistant_reply_callback(self, callback: Optional[AssistantReplyFn]) -> None:
        self._assistant_reply_fn = callback

    def set_retrieve_reply_callback(self, callback: Optional[RetrieveReplyFn]) -> None:
        self._retrieve_reply_fn = callback

    @staticmethod
    def _extract_retrieve_trigger(text: str) -> tuple[bool, str]:
        """Detect 'see ya'/'sia' wake trigger and return the post-trigger query."""
        raw = str(text or "").strip()
        if not raw:
            return False, ""

        match = re.search(
            r"\b(?:see[\s\-]*ya|cya|s[\s\.\-]*i[\s\.\-]*a)\b",
            raw,
            flags=re.IGNORECASE,
        )
        if not match:
            return False, raw

        query_after_trigger = str(raw[match.end():] or "").strip(" \t,:;-")
        return True, query_after_trigger

    async def _set_interaction_mode(self, mode: str, *, reason: str = "") -> None:
        mode = str(mode or LiveInteractionMode.CAPTURE).strip().lower()
        if mode not in {LiveInteractionMode.CAPTURE, LiveInteractionMode.RETRIEVE}:
            mode = LiveInteractionMode.CAPTURE

        if self._interaction_mode == mode:
            return

        self._interaction_mode = mode
        self._retrieve_mode_updated_at = time.time()
        await self._broadcast(
            {
                "type": "interaction_mode",
                "session_id": self._session_id or None,
                "mode": self._interaction_mode,
                "reason": reason,
                "timestamp": round(time.time(), 3),
            }
        )

    async def start(self) -> Dict[str, Any]:
        if self._running:
            return {
                "success": False,
                "error": "Gemini Live session already running",
                "session_state": self._state,
            }

        self._running = True
        self._paused = False
        self._session_id = f"live_{uuid.uuid4().hex[:12]}"
        self._session_started_at = time.time()
        self._state = LiveSessionState.IDLE_LISTENING
        self._state_updated_at = time.time()
        self._last_error = ""
        self._interaction_mode = LiveInteractionMode.CAPTURE
        self._retrieve_mode_armed = False
        self._retrieve_mode_updated_at = time.time()

        self._memory_worker_task = asyncio.create_task(self._memory_worker(), name="ambient-live-memory")

        self._native_live_connected = await self._live_client.connect()
        if self._native_live_connected:
            self._native_events_task = asyncio.create_task(
                self._pump_native_live_events(),
                name="ambient-live-native-events",
            )

        await self._emit_session_state(reason="session_started")

        return {
            "success": True,
            "session_id": self._session_id,
            "session_state": self._state,
            "native_live_connected": self._native_live_connected,
            "native_live_error": self._live_client.last_error,
        }

    async def stop(self) -> Dict[str, Any]:
        if not self._running:
            return {
                "success": False,
                "error": "Gemini Live session is not running",
                "session_state": self._state,
            }

        self._running = False
        self._paused = False

        if self._speech_active and len(self._speech_frames) >= self._min_speech_frames:
            try:
                segment = np.concatenate(self._speech_frames).astype(np.int16, copy=False)
                await self._handle_speech_segment(segment)
            except Exception:
                pass

        self._speech_active = False
        self._speech_frames = []
        self._silence_count = 0
        self._prebuffer.clear()

        try:
            self._memory_queue.put_nowait({"type": "shutdown"})
        except asyncio.QueueFull:
            pass

        if self._memory_worker_task:
            try:
                await self._memory_worker_task
            except Exception:
                pass
            self._memory_worker_task = None

        if self._native_events_task:
            self._native_events_task.cancel()
            try:
                await self._native_events_task
            except Exception:
                pass
            self._native_events_task = None

        await self._live_client.close()
        self._native_live_connected = False

        await self._emit_session_state(reason="session_stopped")

        self._session_id = ""
        self._session_started_at = None
        self._state = LiveSessionState.IDLE_LISTENING
        self._state_updated_at = time.time()
        self._interaction_mode = LiveInteractionMode.CAPTURE
        self._retrieve_mode_armed = False
        self._retrieve_mode_updated_at = time.time()

        return {
            "success": True,
            "session_state": self._state,
        }

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def process_audio_frame(self, frame: np.ndarray) -> None:
        """Called from the audio callback thread."""
        if not self._running or self._paused:
            return

        try:
            audio = np.asarray(frame, dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        except Exception:
            return

        self._stats["audio_frames"] += 1
        self._stats["last_audio_level"] = rms
        self._prebuffer.append(audio.copy())

        is_speech = rms >= self._energy_threshold

        if is_speech:
            if not self._speech_active:
                self._speech_active = True
                self._silence_count = 0
                self._speech_frames = list(self._prebuffer)
                self._set_state_from_thread(LiveSessionState.USER_DETECTED, reason="energy_gate")

            self._speech_frames.append(audio.copy())
            self._silence_count = 0
            self._forward_native_audio(audio)
            return

        if not self._speech_active:
            return

        self._silence_count += 1
        self._speech_frames.append(audio.copy())

        if self._silence_count < self._silence_frames:
            return

        buffered = self._speech_frames
        self._speech_active = False
        self._speech_frames = []
        self._silence_count = 0

        if len(buffered) < self._min_speech_frames:
            return

        try:
            segment = np.concatenate(buffered).astype(np.int16, copy=False)
        except Exception:
            return

        asyncio.run_coroutine_threadsafe(
            self._handle_speech_segment(segment),
            self._worker_loop,
        )

    def get_status(self) -> Dict[str, Any]:
        uptime = (time.time() - self._session_started_at) if self._session_started_at else 0.0
        return {
            "enabled": True,
            "running": self._running,
            "paused": self._paused,
            "state": self._state,
            "state_updated_at": self._state_updated_at,
            "session_id": self._session_id or None,
            "uptime_seconds": round(uptime, 1),
            "native_live_connected": self._native_live_connected,
            "native_live_error": self._live_client.last_error,
            "energy_threshold": self._energy_threshold,
            "interaction_mode": self._interaction_mode,
            "retrieve_mode_armed": self._retrieve_mode_armed,
            **self._stats,
            "last_error": self._last_error or None,
        }

    async def _handle_speech_segment(self, audio: np.ndarray) -> None:
        if not self._running:
            return

        self._stats["segments_detected"] += 1
        await self._set_state(LiveSessionState.LIVE_STREAMING, reason="speech_segment")

        speaker_label, speaker_name, speaker_conf = self._resolve_speaker(audio)

        transcript = await self._gemini_stt.transcribe(audio, language=self._language)
        text = str(transcript.get("text", "") or "").strip()
        confidence = float(transcript.get("confidence", 0.0) or 0.0)

        if not text:
            await self._set_state(LiveSessionState.IDLE_LISTENING, reason="empty_transcript")
            return

        wake_triggered, triggered_query = self._extract_retrieve_trigger(text)
        user_query = str(triggered_query or "").strip() if wake_triggered else text

        if wake_triggered:
            self._retrieve_mode_armed = True
            await self._set_interaction_mode(LiveInteractionMode.RETRIEVE, reason="wake_phrase")

        live_turn_id = f"lt_{uuid.uuid4().hex[:10]}"
        retention = self._build_retention_trace(
            text,
            speaker_label,
            speaker_conf,
            interaction_mode=self._interaction_mode,
            wake_triggered=wake_triggered,
        )

        await self._emit_live_partial(
            text=text,
            speaker_label=speaker_label,
            speaker_name=speaker_name,
            confidence=confidence,
            speaker_confidence=speaker_conf,
            live_turn_id=live_turn_id,
        )

        await self._emit_live_final_turn(
            text=text,
            speaker_label=speaker_label,
            speaker_name=speaker_name,
            confidence=confidence,
            speaker_confidence=speaker_conf,
            live_turn_id=live_turn_id,
            retention_trace=retention,
        )

        # Keep the original transcript event for backwards compatibility.
        await self._broadcast(
            {
                "type": "transcript",
                "speaker_label": speaker_label,
                "speaker_name": speaker_name,
                "text": text,
                "timestamp": round(time.time(), 3),
                "confidence": confidence,
                "speaker_confidence": speaker_conf,
                "live_turn_id": live_turn_id,
                "retention_trace": retention,
            }
        )

        if retention.get("decision") == "keep" and self._conversation:
            await self._conversation.add_turn(
                speaker_label=speaker_label,
                speaker_name=speaker_name,
                text=text,
                timestamp=time.time(),
                confidence=confidence,
                speaker_confidence=speaker_conf,
                live_turn_id=live_turn_id,
                retention_trace=retention,
            )
            self._stats["user_turns"] += 1
            self._queue_memory_job(
                {
                    "type": "turn",
                    "session_id": self._session_id,
                    "live_turn_id": live_turn_id,
                    "speaker_label": speaker_label,
                    "speaker_name": speaker_name,
                    "text": text,
                    "retention_trace": retention,
                }
            )

        should_reply = self._should_generate_reply(speaker_label)
        if self._interaction_mode == LiveInteractionMode.RETRIEVE:
            if not user_query:
                # Wake phrase with no immediate query: acknowledge and keep mode armed.
                await self._generate_assistant_turn(
                    user_text="",
                    retrieve_only=False,
                    override_text="How can I help you?",
                )
            elif should_reply:
                await self._generate_assistant_turn(user_text=user_query, retrieve_only=True)
                self._retrieve_mode_armed = False
                await self._set_interaction_mode(LiveInteractionMode.CAPTURE, reason="retrieve_cycle_complete")
        elif should_reply:
            await self._generate_assistant_turn(user_text=text, retrieve_only=False)

        await self._set_state(LiveSessionState.BACKGROUND_PROCESSING, reason="turn_complete")
        await self._set_state(LiveSessionState.IDLE_LISTENING, reason="ready")

    def _resolve_speaker(self, audio: np.ndarray) -> tuple[str, str, float]:
        if not self._speaker_id:
            return "USER", "You", 1.0

        try:
            speaker_label, confidence = self._speaker_id.identify(audio)
            speaker_name = self._speaker_id.get_display_name(speaker_label)
            return speaker_label, speaker_name, float(confidence)
        except Exception:
            return "UNKNOWN", "Speaker", 0.0

    def _build_retention_trace(
        self,
        text: str,
        speaker_label: str,
        speaker_confidence: float,
        *,
        interaction_mode: str = LiveInteractionMode.CAPTURE,
        wake_triggered: bool = False,
    ) -> Dict[str, Any]:
        words = text.split()
        word_count = len(words)
        lowered = text.lower()

        score = 0.0
        score += min(word_count / 18.0, 0.45)
        if speaker_label == "USER":
            score += 0.25
        if speaker_confidence >= 0.75:
            score += 0.20
        if any(k in lowered for k in ("remember", "decide", "plan", "todo", "deadline", "meeting")):
            score += 0.25
        if any(ch.isdigit() for ch in text):
            score += 0.10
        if word_count < 3:
            score -= 0.35

        # Conversation mode and question-form utterances should be preserved more often.
        if interaction_mode == LiveInteractionMode.RETRIEVE:
            score += 0.25
        if wake_triggered:
            score += 0.35
        if re.search(r"\b(what|who|when|where|why|how|which|show|find|tell)\b", lowered):
            score += 0.16
        if str(speaker_label).upper().startswith("SPEAKER_") and speaker_confidence >= 0.6:
            score += 0.12

        tags: List[str] = []
        if any(k in lowered for k in ("todo", "need to", "must", "by ")):
            tags.append("action_item")
        if "?" in text:
            tags.append("question")
        if any(k in lowered for k in ("prefer", "like", "love", "hate")):
            tags.append("preference")
        if any(k in lowered for k in ("project", "build", "architecture", "api")):
            tags.append("technical")
        if interaction_mode == LiveInteractionMode.RETRIEVE:
            tags.append("retrieval_query")
        if wake_triggered:
            tags.append("retrieve_wake")
        if not tags:
            tags.append("conversation")

        decision = "keep" if score >= 0.35 else "discard"
        if wake_triggered:
            decision = "keep"
        reason = "retained_for_memory" if decision == "keep" else "low_signal"

        return {
            "decision": decision,
            "reason": reason,
            "score": round(score, 3),
            "tags": list(dict.fromkeys(tags)),
            "source": "gemini_live",
            "speaker_label": speaker_label,
            "speaker_confidence": round(float(speaker_confidence), 3),
            "interaction_mode": interaction_mode,
            "wake_triggered": bool(wake_triggered),
        }

    async def _emit_live_partial(
        self,
        *,
        text: str,
        speaker_label: str,
        speaker_name: str,
        confidence: float,
        speaker_confidence: float,
        live_turn_id: str,
    ) -> None:
        words = text.split()
        if not words:
            return

        step = max(1, int(math.ceil(len(words) / 3)))
        seen = set()
        for i in range(step, len(words) + step, step):
            partial = " ".join(words[:i]).strip()
            if not partial or partial in seen:
                continue
            seen.add(partial)
            await self._broadcast(
                {
                    "type": "live_partial",
                    "session_id": self._session_id,
                    "live_turn_id": live_turn_id,
                    "speaker_label": speaker_label,
                    "speaker_name": speaker_name,
                    "text": partial,
                    "confidence": confidence,
                    "speaker_confidence": speaker_confidence,
                    "timestamp": round(time.time(), 3),
                }
            )

    async def _emit_live_final_turn(
        self,
        *,
        text: str,
        speaker_label: str,
        speaker_name: str,
        confidence: float,
        speaker_confidence: float,
        live_turn_id: str,
        retention_trace: Dict[str, Any],
    ) -> None:
        await self._broadcast(
            {
                "type": "live_final_turn",
                "session_id": self._session_id,
                "live_turn_id": live_turn_id,
                "speaker_label": speaker_label,
                "speaker_name": speaker_name,
                "text": text,
                "confidence": confidence,
                "speaker_confidence": speaker_confidence,
                "retention_trace": retention_trace,
                "timestamp": round(time.time(), 3),
            }
        )

    async def _generate_assistant_turn(
        self,
        *,
        user_text: str,
        retrieve_only: bool = False,
        override_text: str = "",
    ) -> None:
        await self._set_state(LiveSessionState.ASSISTANT_RESPONDING, reason="assistant_reply")

        assistant_text = ""
        try:
            if override_text:
                assistant_text = str(override_text).strip()
            elif retrieve_only and self._retrieve_reply_fn is not None:
                assistant_text = (await self._retrieve_reply_fn(user_text, self._session_id)).strip()
            elif self._assistant_reply_fn is not None:
                assistant_text = (await self._assistant_reply_fn(user_text, self._session_id)).strip()
        except Exception as exc:
            self._last_error = f"assistant callback failed: {exc}"

        if not assistant_text and retrieve_only:
            assistant_text = (
                "I checked your memory but could not find a strong match yet. "
                "Please rephrase with one more detail and I will retrieve again."
            )

        if not assistant_text:
            assistant_text = (
                "I heard you. I have captured that turn and will keep listening. "
                "You can continue speaking."
            )

        assistant_turn_id = f"at_{uuid.uuid4().hex[:10]}"
        retention = {
            "decision": "ephemeral",
            "reason": "assistant_response",
            "score": 1.0,
            "tags": ["assistant"],
            "source": "gemini_live",
        }

        if self._conversation:
            await self._conversation.add_turn(
                speaker_label="ASSISTANT",
                speaker_name="Cortex",
                text=assistant_text,
                timestamp=time.time(),
                confidence=1.0,
                speaker_confidence=1.0,
                live_turn_id=assistant_turn_id,
                retention_trace=retention,
            )

        self._stats["assistant_turns"] += 1

        await self._emit_live_final_turn(
            text=assistant_text,
            speaker_label="ASSISTANT",
            speaker_name="Cortex",
            confidence=1.0,
            speaker_confidence=1.0,
            live_turn_id=assistant_turn_id,
            retention_trace=retention,
        )

        await self._broadcast(
            {
                "type": "transcript",
                "speaker_label": "ASSISTANT",
                "speaker_name": "Cortex",
                "text": assistant_text,
                "timestamp": round(time.time(), 3),
                "confidence": 1.0,
                "speaker_confidence": 1.0,
                "live_turn_id": assistant_turn_id,
                "retention_trace": retention,
            }
        )

        await self._emit_assistant_audio_chunks(assistant_text, assistant_turn_id)

    async def _emit_assistant_audio_chunks(self, text: str, assistant_turn_id: str) -> None:
        if not self._gemini_tts or not getattr(self._gemini_tts, "is_available", False):
            return

        wav_bytes = None
        try:
            if hasattr(self._gemini_tts, "synthesize_to_wav_async"):
                wav_bytes = await self._gemini_tts.synthesize_to_wav_async(text)
            else:
                loop = asyncio.get_running_loop()
                wav_bytes = await loop.run_in_executor(None, self._gemini_tts.synthesize_to_wav, text)
        except Exception as exc:
            self._last_error = f"assistant tts failed: {exc}"
            return

        if not wav_bytes:
            return

        encoded = base64.b64encode(wav_bytes).decode("ascii")
        chunk_size = 12000
        chunks = [encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)]

        for idx, chunk in enumerate(chunks):
            await self._broadcast(
                {
                    "type": "assistant_audio_chunk",
                    "session_id": self._session_id,
                    "live_turn_id": assistant_turn_id,
                    "mime_type": "audio/wav",
                    "chunk_index": idx,
                    "chunk_total": len(chunks),
                    "audio_base64_chunk": chunk,
                    "is_last": idx == len(chunks) - 1,
                    "timestamp": round(time.time(), 3),
                }
            )

    async def _memory_worker(self) -> None:
        while True:
            job = await self._memory_queue.get()
            if job.get("type") == "shutdown":
                return

            self._stats["memory_jobs"] += 1
            try:
                await self._broadcast(
                    {
                        "type": "memory_tagging",
                        "session_id": job.get("session_id"),
                        "live_turn_id": job.get("live_turn_id"),
                        "speaker_label": job.get("speaker_label"),
                        "tags": (job.get("retention_trace") or {}).get("tags", []),
                        "decision": (job.get("retention_trace") or {}).get("decision", "keep"),
                        "timestamp": round(time.time(), 3),
                    }
                )
            except Exception:
                pass

    async def _pump_native_live_events(self) -> None:
        while self._running and self._native_live_connected:
            events = await self._live_client.receive_events(max_events=12)
            if not events:
                await asyncio.sleep(0.05)
                continue

            for event in events:
                self._stats["native_events_seen"] += 1
                await self._broadcast(
                    {
                        "type": "session_state",
                        "session_id": self._session_id,
                        "state": self._state,
                        "native_event": event,
                        "timestamp": round(time.time(), 3),
                    }
                )

    def _should_generate_reply(self, speaker_label: str) -> bool:
        if self._interaction_mode == LiveInteractionMode.RETRIEVE:
            return True

        if speaker_label == "USER":
            return True

        # In real-world rooms, USER can occasionally be clustered as SPEAKER_*.
        # Keep companion mode responsive for those turns.
        if str(speaker_label).upper().startswith("SPEAKER_"):
            return True

        # If no enrollment exists yet, keep companion mode conversational.
        if self._speaker_id and hasattr(self._speaker_id, "is_enrolled"):
            try:
                if not self._speaker_id.is_enrolled():
                    return True
            except Exception:
                pass

        return False

    async def _set_state(self, state: str, *, reason: str = "") -> None:
        if self._state == state:
            return
        self._state = state
        self._state_updated_at = time.time()
        await self._emit_session_state(reason=reason)

    def _set_state_from_thread(self, state: str, *, reason: str = "") -> None:
        if self._state == state:
            return
        self._state = state
        self._state_updated_at = time.time()
        asyncio.run_coroutine_threadsafe(
            self._emit_session_state(reason=reason),
            self._worker_loop,
        )

    async def _emit_session_state(self, *, reason: str = "") -> None:
        await self._broadcast(
            {
                "type": "session_state",
                "session_id": self._session_id or None,
                "state": self._state,
                "interaction_mode": self._interaction_mode,
                "retrieve_mode_armed": self._retrieve_mode_armed,
                "reason": reason,
                "running": self._running,
                "paused": self._paused,
                "native_live_connected": self._native_live_connected,
                "timestamp": round(time.time(), 3),
            }
        )

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._ws_broadcast:
            return
        try:
            await self._ws_broadcast(payload)
        except Exception:
            pass

    def _queue_memory_job(self, job: Dict[str, Any]) -> None:
        try:
            self._memory_queue.put_nowait(job)
        except asyncio.QueueFull:
            # Drop oldest intent under sustained load; listening must stay real-time.
            try:
                _ = self._memory_queue.get_nowait()
            except Exception:
                pass
            try:
                self._memory_queue.put_nowait(job)
            except Exception:
                pass

    def _forward_native_audio(self, frame: np.ndarray) -> None:
        if not self._native_live_connected:
            return
        asyncio.run_coroutine_threadsafe(
            self._live_client.send_audio_chunk(frame, sample_rate=self._sample_rate),
            self._worker_loop,
        )
