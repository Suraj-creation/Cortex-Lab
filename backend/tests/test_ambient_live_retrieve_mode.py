"""Ambient live orchestrator behavior tests for retrieve-mode and persistence."""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeAudioCapture:
    FRAME_MS = 32
    SAMPLE_RATE = 16000


class _FakeSTT:
    def __init__(self, text: str):
        self._text = text

    async def transcribe(self, _audio: np.ndarray, language=None):
        return {
            "text": self._text,
            "confidence": 0.92,
            "language": language or "en",
        }


class _FakeTTS:
    is_available = False


class _FakeConversation:
    def __init__(self):
        self.turns: List[Dict[str, Any]] = []

    async def add_turn(self, **kwargs):
        self.turns.append(dict(kwargs))


class _FakeSpeakerId:
    def __init__(self, label: str = "SPEAKER_A", confidence: float = 0.86, enrolled: bool = True):
        self._label = label
        self._confidence = confidence
        self._enrolled = enrolled

    def identify(self, _audio: np.ndarray):
        return self._label, self._confidence

    def get_display_name(self, speaker_label: str):
        return speaker_label

    def is_enrolled(self) -> bool:
        return self._enrolled


async def _new_orchestrator(
    text: str,
    *,
    speaker_label: str,
    speaker_confidence: float,
    speaker_enrolled: bool,
):
    from src.ambient.gemini_live import GeminiLiveSessionOrchestrator

    conversation = _FakeConversation()
    speaker = _FakeSpeakerId(
        label=speaker_label,
        confidence=speaker_confidence,
        enrolled=speaker_enrolled,
    )

    ws_events: List[Dict[str, Any]] = []

    async def _broadcast(payload: Dict[str, Any]) -> None:
        ws_events.append(dict(payload))

    orchestrator = GeminiLiveSessionOrchestrator(
        api_key="test-key",
        worker_loop=asyncio.get_running_loop(),
        gemini_stt=_FakeSTT(text),
        gemini_tts=_FakeTTS(),
        conversation=conversation,
        audio_capture=_FakeAudioCapture(),
        speaker_id=speaker,
        ws_broadcast=_broadcast,
        assistant_reply_fn=None,
    )
    orchestrator._running = True
    orchestrator._session_id = "live_test"
    orchestrator._session_started_at = time.time()

    return orchestrator, conversation, ws_events


@pytest.mark.asyncio
async def test_enrolled_unknown_speaker_still_generates_reply_and_assistant_turn():
    """When speaker-id misses USER, live mode should still answer conversationally."""
    orchestrator, conversation, _events = await _new_orchestrator(
        "What are my goals for this week?",
        speaker_label="SPEAKER_A",
        speaker_confidence=0.88,
        speaker_enrolled=True,
    )

    callback_calls: List[str] = []

    async def _assistant_reply(text: str, _session_id: str) -> str:
        callback_calls.append(text)
        return "Here is your weekly goals summary."

    orchestrator.set_assistant_reply_callback(_assistant_reply)

    audio = np.ones(6400, dtype=np.int16)
    await orchestrator._handle_speech_segment(audio)

    # Desired behavior: unknown-but-confident speaker should not suppress replies.
    assert callback_calls, "assistant callback should be called for conversational turns"
    assert any(
        str(t.get("speaker_label", "")).upper() == "ASSISTANT"
        for t in conversation.turns
    ), "assistant turn should be captured for downstream persistence"


@pytest.mark.asyncio
async def test_wake_phrase_routes_to_retrieve_mode_and_uses_retrieve_callback():
    """SIA/see ya wake phrase should switch to retrieve mode and answer from retrieval flow."""
    orchestrator, _conversation, _events = await _new_orchestrator(
        "See ya what did I decide about project deadlines?",
        speaker_label="USER",
        speaker_confidence=0.95,
        speaker_enrolled=True,
    )

    assistant_calls: List[str] = []
    retrieve_calls: List[str] = []

    async def _assistant_reply(text: str, _session_id: str) -> str:
        assistant_calls.append(text)
        return "default assistant path"

    async def _retrieve_reply(text: str, _session_id: str) -> str:
        retrieve_calls.append(text)
        return "retrieved answer"

    orchestrator.set_assistant_reply_callback(_assistant_reply)
    orchestrator.set_retrieve_reply_callback(_retrieve_reply)

    audio = np.ones(6400, dtype=np.int16)
    await orchestrator._handle_speech_segment(audio)

    assert retrieve_calls, "retrieve callback should be used when wake phrase is present"
    assert "see ya" not in retrieve_calls[0].lower(), "wake phrase should be stripped from retrieval query"
    assert not assistant_calls, "default assistant callback should be bypassed in retrieve mode"


@pytest.mark.asyncio
async def test_wake_phrase_without_query_returns_quick_ack_and_keeps_mode_armed():
    """Wake phrase without payload should acknowledge quickly and keep retrieve mode armed."""
    orchestrator, conversation, _events = await _new_orchestrator(
        "SIA",
        speaker_label="USER",
        speaker_confidence=0.95,
        speaker_enrolled=True,
    )

    audio = np.ones(6400, dtype=np.int16)
    await orchestrator._handle_speech_segment(audio)

    assistant_turns = [
        t for t in conversation.turns
        if str(t.get("speaker_label", "")).upper() == "ASSISTANT"
    ]
    assert assistant_turns, "assistant should acknowledge wake-only utterances"
    assert assistant_turns[-1].get("text") == "How can I help you?"
    assert orchestrator._retrieve_mode_armed is True


def test_retrieve_trigger_parser_accepts_spaced_sia_variant():
    """Wake parser should handle common STT tokenization variants like 's i a'."""
    from src.ambient.gemini_live import GeminiLiveSessionOrchestrator

    triggered, query = GeminiLiveSessionOrchestrator._extract_retrieve_trigger(
        "s i a what did I decide about deadlines"
    )

    assert triggered is True
    assert query.lower() == "what did i decide about deadlines"


@pytest.mark.asyncio
async def test_single_retained_turn_is_persisted_on_finalize(tmp_path):
    """A single retained turn should still be saved as a conversation record."""
    from src.ambient.conversation import ConversationSegmenter

    segmenter = ConversationSegmenter(
        ingestion_pipeline=None,
        auto_ingest=False,
        data_dir=str(tmp_path),
        gemini_api_key=None,
    )

    assert getattr(segmenter, "_db_backend", "none") in {"duckdb", "sqlite"}

    await segmenter.add_turn(
        speaker_label="USER",
        speaker_name="You",
        text="Remember that I committed to finalize the proposal today.",
        timestamp=time.time(),
        confidence=0.91,
        speaker_confidence=0.89,
        live_turn_id="lt_single",
        retention_trace={"decision": "keep", "score": 0.9, "tags": ["action_item"]},
    )

    await segmenter.force_finalize()
    conversations = segmenter.get_conversations(limit=10, offset=0)

    assert len(conversations) == 1, "single-turn retained conversation should be persisted"
    assert len(conversations[0].get("turns", [])) == 1
