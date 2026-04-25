"""Regression tests for the client-driven ambient companion flow."""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as backend_server


class _FakeConversation:
    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self.force_finalize_calls = 0

    async def add_turn(self, **kwargs) -> None:
        self.turns.append(dict(kwargs))

    async def force_finalize(self) -> None:
        self.force_finalize_calls += 1

    def get_current_turns(self) -> list[dict[str, Any]]:
        return list(self.turns)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_conversations": 0,
            "current_turns": len(self.turns),
            "total_ingested": 0,
        }


class _FakeMetadataStore:
    def __init__(self) -> None:
        self.turns: list[dict[str, str]] = []

    def store_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        thinking: str = "",
        memory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.turns.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "thinking": thinking,
                "memory_id": memory_id or "",
                "metadata": dict(metadata or {}),
            }
        )


class _FakeSTT:
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None = None,
        quality_mode: bool = False,
        estimated_duration_s: float = 0.0,
    ) -> dict[str, Any]:
        assert audio_bytes
        assert mime_type == "audio/webm"
        return {
            "text": "Eva remember that I decided to ship the ambient companion this week.",
            "confidence": 0.94,
            "language": language or "en",
            "segments": [{"start": 0.0, "end": 1.2, "text": "test"}],
        }


class _FakeTTS:
    is_available = True

    async def synthesize_to_wav_async(self, text: str) -> bytes:
        assert "ambient companion" in text.lower()
        return b"RIFFfakewav"

    def get_stats(self) -> dict[str, Any]:
        return {
            "available": True,
            "voice": "Kore",
            "total_syntheses": 1,
        }


async def _assistant_reply(text: str, session_id: str) -> str:
    assert session_id.startswith("session-")
    return (
        "I've stored that ambient companion decision and will keep refining it "
        "for retrieval."
    )


def test_client_companion_processing_persists_session_turns_and_reply(tmp_path, monkeypatch):
    from src.ambient import AmbientService
    from src.runtime.session_manager import runtime_session_manager

    service = AmbientService(
        ingestion_pipeline=None,
        data_dir=str(tmp_path),
        gemini_api_key="test-gemini-key",
    )
    service._gemini_stt = _FakeSTT()
    service._gemini_tts = _FakeTTS()
    service.conversation = _FakeConversation()
    service.set_live_assistant_callback(_assistant_reply)

    fake_store = _FakeMetadataStore()
    fake_engine = type("FakeEngine", (), {"metadata_store": fake_store})()
    monkeypatch.setattr("src.engine.rag_engine", fake_engine, raising=False)

    session = asyncio.run(
        service.start_client_session(
            platform="web",
            metadata={"surface": "ambient-panel"},
        )
    )
    session_id = str(session["session_id"])

    result = asyncio.run(
        service.process_client_audio(
            session_id=session_id,
            audio_bytes=b"fake-webm-audio",
            mime_type="audio/webm",
            platform="web",
        )
    )

    assert result["session_id"] == session_id
    assert result["analysis"]["direct_address"] is True
    assert result["analysis"]["reply_expected"] is True
    assert result["assistant_text"]
    assert result["assistant_audio_base64"]
    assert result["retention_trace"]["memory_decision"] in {"structured", "priority"}
    assert len(service.conversation.turns) == 2
    assert [turn["role"] for turn in fake_store.turns] == ["user", "assistant"]
    assert fake_store.turns[0]["metadata"]["platform"] == "web"
    assert fake_store.turns[0]["metadata"]["retention_trace"]["memory_decision"] in {"structured", "priority"}
    assert fake_store.turns[1]["metadata"]["source"] == "assistant_companion"

    session_snapshot = runtime_session_manager.get_session(session_id)
    assert session_snapshot is not None
    assert session_snapshot.metadata["platform"] == "web"
    assert session_snapshot.metadata["turn_counts"]["user"] == 1
    assert session_snapshot.metadata["turn_counts"]["assistant"] == 1
    assert session_snapshot.metadata["last_retention_trace"]["memory_decision"] in {"structured", "priority"}
    assert float(session_snapshot.metadata["companion_followup_until"]) > 0
    assert "decision" in session_snapshot.agent_tags
    assert session_snapshot.retention_summary["structured"] >= 0
    assert session_snapshot.retention_summary["priority"] >= 0

    stop_result = asyncio.run(
        service.stop_client_session(session_id, reason="test_complete")
    )
    assert stop_result["session"]["end_time"] is not None
    assert service._client_active_session_id == ""


class _FakeHallucinatedSTT:
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None = None,
        quality_mode: bool = False,
        estimated_duration_s: float = 0.0,
    ) -> dict[str, Any]:
        assert audio_bytes
        return {
            "text": "Quick brown fox jumps over the lazy dog",
            "confidence": 0.92,
            "language": language or "en",
            "segments": [{"start": 0.0, "end": estimated_duration_s or 1.2, "text": "pangram"}],
        }


class _FakeMeetingHallucinationSTT:
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None = None,
        quality_mode: bool = False,
        estimated_duration_s: float = 0.0,
    ) -> dict[str, Any]:
        assert audio_bytes
        return {
            "text": "I'm not sure if I'm going to be able to make it to the meeting.",
            "confidence": 0.91,
            "language": language or "en",
            "segments": [{"start": 0.0, "end": estimated_duration_s or 2.0, "text": "meeting"}],
        }


class _TrackingSTT:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None = None,
        quality_mode: bool = False,
        estimated_duration_s: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "text": "Eva save this note.",
            "confidence": 0.91,
            "language": language or "en",
            "segments": [{"start": 0.0, "end": estimated_duration_s or 1.8, "text": "note"}],
        }


def test_client_companion_discards_known_hallucinated_pangram(tmp_path, monkeypatch):
    from src.ambient import AmbientService
    from src.runtime.session_manager import runtime_session_manager

    service = AmbientService(
        ingestion_pipeline=None,
        data_dir=str(tmp_path),
        gemini_api_key="test-gemini-key",
    )
    service._gemini_stt = _FakeHallucinatedSTT()
    service._gemini_tts = _FakeTTS()
    service.conversation = _FakeConversation()
    service.set_live_assistant_callback(_assistant_reply)

    fake_store = _FakeMetadataStore()
    fake_engine = type("FakeEngine", (), {"metadata_store": fake_store})()
    monkeypatch.setattr("src.engine.rag_engine", fake_engine, raising=False)

    session = asyncio.run(service.start_client_session(platform="web"))
    session_id = str(session["session_id"])

    result = asyncio.run(
        service.process_client_audio(
            session_id=session_id,
            audio_bytes=b"hallucinated-audio",
            mime_type="audio/webm",
            platform="web",
            estimated_duration_s=1.1,
        )
    )

    assert result["success"] is True
    assert result["transcript"] == ""
    assert result["assistant_text"] == ""
    assert service.conversation.turns == []
    assert fake_store.turns == []

    session_snapshot = runtime_session_manager.get_session(session_id)
    assert session_snapshot is not None
    assert session_snapshot.metadata.get("last_user_text") in (None, "")


def test_client_companion_discards_known_meeting_hallucination(tmp_path, monkeypatch):
    from src.ambient import AmbientService

    service = AmbientService(
        ingestion_pipeline=None,
        data_dir=str(tmp_path),
        gemini_api_key="test-gemini-key",
    )
    service._gemini_stt = _FakeMeetingHallucinationSTT()
    service._gemini_tts = _FakeTTS()
    service.conversation = _FakeConversation()
    service.set_live_assistant_callback(_assistant_reply)

    fake_store = _FakeMetadataStore()
    fake_engine = type("FakeEngine", (), {"metadata_store": fake_store})()
    monkeypatch.setattr("src.engine.rag_engine", fake_engine, raising=False)

    session = asyncio.run(service.start_client_session(platform="web"))
    session_id = str(session["session_id"])

    result = asyncio.run(
        service.process_client_audio(
            session_id=session_id,
            audio_bytes=b"hallucinated-meeting-audio",
            mime_type="audio/webm",
            platform="web",
            estimated_duration_s=2.3,
        )
    )

    assert result["success"] is True
    assert result["transcript"] == ""
    assert result["assistant_text"] == ""
    assert service.conversation.turns == []
    assert fake_store.turns == []


def test_client_companion_skips_silent_chunk_before_stt(tmp_path, monkeypatch):
    from src.ambient import AmbientService

    tracking_stt = _TrackingSTT()
    service = AmbientService(
        ingestion_pipeline=None,
        data_dir=str(tmp_path),
        gemini_api_key="test-gemini-key",
    )
    service._gemini_stt = tracking_stt
    service._gemini_tts = _FakeTTS()
    service.conversation = _FakeConversation()
    service.set_live_assistant_callback(_assistant_reply)

    fake_store = _FakeMetadataStore()
    fake_engine = type("FakeEngine", (), {"metadata_store": fake_store})()
    monkeypatch.setattr("src.engine.rag_engine", fake_engine, raising=False)

    session = asyncio.run(service.start_client_session(platform="web"))
    session_id = str(session["session_id"])

    result = asyncio.run(
        service.process_client_audio(
            session_id=session_id,
            audio_bytes=b"mostly-silent-audio",
            mime_type="audio/webm",
            platform="web",
            estimated_duration_s=1.8,
            metadata={
                "audio_peak_db": -59.0,
                "audio_avg_db": -64.0,
            },
        )
    )

    assert result["success"] is True
    assert result["transcript"] == ""
    assert result["assistant_text"] == ""
    assert tracking_stt.calls == 0
    assert service.conversation.turns == []
    assert fake_store.turns == []


def test_gemini_live_wake_trigger_accepts_assistant_aliases():
    from src.ambient.gemini_live import GeminiLiveSessionOrchestrator

    orchestrator = GeminiLiveSessionOrchestrator.__new__(GeminiLiveSessionOrchestrator)
    orchestrator._assistant_aliases = ["eva", "cortex", "assistant"]

    triggered, query = orchestrator._extract_retrieve_trigger("Eva what did I say about the graph?")
    assert triggered is True
    assert query.lower() == "what did i say about the graph?"

    triggered, query = orchestrator._extract_retrieve_trigger("Hey Cortex, summarize the latest session.")
    assert triggered is True
    assert query.lower() == "summarize the latest session."


class _FakeAmbientService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.config_updates: dict[str, Any] = {}

    def set_live_assistant_callback(self, callback) -> None:
        self.calls.append(("set_live_assistant_callback", {"callback": callback is not None}))

    def set_live_retrieve_callback(self, callback) -> None:
        self.calls.append(("set_live_retrieve_callback", {"callback": callback is not None}))

    async def start_client_session(self, platform: str = "web", metadata: dict[str, Any] | None = None):
        self.calls.append(("start_client_session", {"platform": platform, "metadata": metadata or {}}))
        return {
            "success": True,
            "session_id": "session-test123",
            "platform": platform,
            "metadata": metadata or {},
        }

    async def stop_client_session(self, session_id: str, reason: str = "user_request"):
        self.calls.append(("stop_client_session", {"session_id": session_id, "reason": reason}))
        return {
            "success": True,
            "session": {
                "session_id": session_id,
                "end_time": "2026-04-25T00:00:00Z",
            },
        }

    def get_client_sessions(self):
        self.calls.append(("get_client_sessions", {}))
        return {
            "active_session_id": "session-test123",
            "sessions": [{"session_id": "session-test123", "mode": "ambient_client"}],
        }

    async def process_client_audio(
        self,
        *,
        session_id: str,
        audio_bytes: bytes,
        mime_type: str,
        platform: str = "web",
        language: str | None = None,
        estimated_duration_s: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.calls.append(
            (
                "process_client_audio",
                {
                    "session_id": session_id,
                    "mime_type": mime_type,
                    "platform": platform,
                    "language": language,
                    "estimated_duration_s": estimated_duration_s,
                    "metadata": metadata or {},
                    "audio_size": len(audio_bytes),
                },
            )
        )
        return {
            "success": True,
            "session_id": session_id,
            "transcript": "hello eva",
            "assistant_text": "hello back",
            "assistant_audio_base64": "UklGRg==",
            "analysis": {"direct_address": True, "reply_expected": True},
            "retention_trace": {"memory_decision": "structured", "tags": ["spoken_dialogue"]},
        }

    def get_status(self):
        return {
            "status": "idle",
            "uptime_seconds": 0,
            "error": None,
            "enrolled": False,
            "tts_available": True,
            "audio_level": 0,
            "speech_segments": 0,
            "transcriptions": 0,
            "stt_provider": "gemini",
            "tts_provider": "gemini",
            "gemini_available": True,
        }


class _FakeRagEngine:
    initialized = True

    def __init__(self, ambient_service: _FakeAmbientService):
        self.ambient_service = ambient_service


def _ambient_client_test_client(monkeypatch):
    fake_ambient = _FakeAmbientService()

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    monkeypatch.setattr(backend_server.app.router, "lifespan_context", _no_lifespan)
    monkeypatch.setattr(backend_server, "rag_engine", _FakeRagEngine(fake_ambient))

    client = TestClient(backend_server.app)
    return client, fake_ambient


def test_ambient_client_routes_delegate_to_service(monkeypatch):
    client, fake_ambient = _ambient_client_test_client(monkeypatch)

    with client:
        start_response = client.post(
            "/api/ambient/client/session/start",
            json={"platform": "web", "metadata": {"surface": "ambient-panel"}},
        )
        list_response = client.get("/api/ambient/client/sessions")
        audio_response = client.post(
            "/api/ambient/client/process-audio",
            json={
                "session_id": "session-test123",
                "audio_base64": base64.b64encode(b"audio-bytes").decode("ascii"),
                "mime_type": "audio/webm",
                "platform": "web",
                "language": "en",
                "estimated_duration_s": 1.4,
                "metadata": {"surface": "ambient-panel"},
            },
        )
        stop_response = client.post(
            "/api/ambient/client/session/stop",
            json={"session_id": "session-test123", "reason": "user_request"},
        )

    assert start_response.status_code == 200
    assert list_response.status_code == 200
    assert audio_response.status_code == 200
    assert stop_response.status_code == 200

    assert start_response.json()["session_id"] == "session-test123"
    assert audio_response.json()["analysis"]["direct_address"] is True
    assert stop_response.json()["session"]["session_id"] == "session-test123"

    call_names = [name for name, _payload in fake_ambient.calls]
    assert "start_client_session" in call_names
    assert "get_client_sessions" in call_names
    assert "process_client_audio" in call_names
    assert "stop_client_session" in call_names
