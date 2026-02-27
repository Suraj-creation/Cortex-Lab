"""
Conversation Segmenter + Auto-Ingestion Bridge
Groups speaker-labeled transcript turns into ConversationRecord objects.
Detects conversation boundaries (>2 min silence = end of conversation).
Automatically feeds completed conversations into MemoryIngestionPipeline.

This is THE bridge between the voice stack and the existing Agentic RAG system.
"""

import time
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    speaker_label: str        # "USER", "SPEAKER_A", etc.
    speaker_name: str = ""    # Resolved name: "Suraj", "Sarah"
    text: str = ""
    timestamp: float = 0.0    # Unix-like timestamp (seconds since ambient start)
    confidence: float = 0.0   # STT confidence


@dataclass
class ConversationRecord:
    id: str = field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    turns: List[ConversationTurn] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_ids: List[str] = field(default_factory=list)
    auto_ingested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "turns": [
                {
                    "speaker_label": t.speaker_label,
                    "speaker_name": t.speaker_name,
                    "text": t.text,
                    "timestamp": t.timestamp,
                    "confidence": t.confidence,
                }
                for t in self.turns
            ],
            "participants": self.participants,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 1),
            "memory_ids": self.memory_ids,
            "auto_ingested": self.auto_ingested,
        }


class ConversationSegmenter:
    """
    Groups transcript turns into conversations, detects boundaries,
    and feeds completed conversations into the existing MemoryIngestionPipeline.
    """

    SILENCE_THRESHOLD_S = 120   # 2 min silence = conversation end
    MIN_TURNS = 2               # Minimum turns to count as a conversation
    MERGE_THRESHOLD_S = 5       # Merge turns from same speaker within 5s

    def __init__(self, ingestion_pipeline=None, auto_ingest: bool = True,
                 data_dir: str = "data"):
        """
        Args:
            ingestion_pipeline: MemoryIngestionPipeline instance (for auto-ingestion)
            auto_ingest: Whether to automatically ingest completed conversations
            data_dir: Base data directory for saving conversation records
        """
        self.pipeline = ingestion_pipeline
        self.auto_ingest = auto_ingest
        self.data_dir = data_dir
        Path(f"{data_dir}/conversations").mkdir(parents=True, exist_ok=True)

        # Current in-progress conversation
        self._current_turns: List[ConversationTurn] = []
        self._last_turn_time: float = 0.0

        # Completed conversations
        self._conversations: List[ConversationRecord] = []

        # Callback for live transcript updates
        self._on_turn: Optional[Callable] = None
        self._on_conversation_end: Optional[Callable] = None

        # Load saved conversations
        self._load_conversations()

    # ── Core Processing ──────────────────────────────────────────────────

    async def add_turn(self, speaker_label: str, speaker_name: str,
                       text: str, timestamp: float, confidence: float = 0.0):
        """
        Add a transcribed turn. Automatically segments and ingests.

        Called by AmbientService after: VAD → SpeakerID → Whisper → here.
        """
        if not text.strip():
            return

        now = time.time()

        # Check for conversation boundary (silence > threshold)
        if (self._current_turns and
                self._last_turn_time > 0 and
                (now - self._last_turn_time) > self.SILENCE_THRESHOLD_S):
            await self._finalize_conversation()

        turn = ConversationTurn(
            speaker_label=speaker_label,
            speaker_name=speaker_name or speaker_label,
            text=text.strip(),
            timestamp=timestamp,
            confidence=confidence,
        )

        # Merge with previous turn if same speaker and close in time
        if (self._current_turns and
                self._current_turns[-1].speaker_label == speaker_label and
                (timestamp - self._current_turns[-1].timestamp) < self.MERGE_THRESHOLD_S):
            self._current_turns[-1].text += " " + text.strip()
            self._current_turns[-1].confidence = max(
                self._current_turns[-1].confidence, confidence
            )
        else:
            self._current_turns.append(turn)

        self._last_turn_time = now

        # Notify live transcript listeners
        if self._on_turn:
            try:
                self._on_turn(turn)
            except Exception:
                pass

    async def force_finalize(self):
        """Force-finalize the current conversation (e.g. when stopping ambient)."""
        if self._current_turns:
            await self._finalize_conversation()

    # ── Finalization ─────────────────────────────────────────────────────

    async def _finalize_conversation(self):
        """End current conversation, format, and ingest into RAG."""
        if len(self._current_turns) < self.MIN_TURNS:
            self._current_turns = []
            return

        # Build conversation record
        participants = list(set(t.speaker_label for t in self._current_turns))
        start_time = datetime.now()  # Approximate — could use first turn timestamp
        end_time = datetime.now()

        record = ConversationRecord(
            turns=list(self._current_turns),
            participants=participants,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=sum(
                max(0.5, len(t.text.split()) * 0.3) for t in self._current_turns
            ),
        )

        # Auto-ingest into RAG pipeline
        if self.auto_ingest and self.pipeline:
            try:
                memory_ids = await self._ingest_conversation(record)
                record.memory_ids = memory_ids
                record.auto_ingested = True
                print(f"  📝 Conversation {record.id} ingested "
                      f"({len(record.turns)} turns → {len(memory_ids)} memories)")
            except Exception as e:
                print(f"  ⚠ Conversation ingestion failed: {e}")

        # Store the record
        self._conversations.append(record)
        self._save_conversation(record)

        # Notify listeners
        if self._on_conversation_end:
            try:
                self._on_conversation_end(record)
            except Exception:
                pass

        # Reset current conversation
        self._current_turns = []
        self._last_turn_time = 0.0

    async def _ingest_conversation(self, record: ConversationRecord) -> List[str]:
        """
        Bridge to existing MemoryIngestionPipeline.
        Formats the conversation and calls:
            await self.pipeline.ingest(content=..., source="voice", ...)
        """
        memory_ids = []

        # Format the full conversation transcript
        parts = []
        participant_names = ", ".join(
            t.speaker_name for t in record.turns
            if t.speaker_name not in [p for p in parts]  # unique
        )
        unique_participants = list(set(t.speaker_name for t in record.turns))

        header = (
            f"[Voice Conversation with {', '.join(unique_participants)}]"
            f"\n[Duration: ~{record.duration_seconds:.0f}s, "
            f"Turns: {len(record.turns)}]"
        )

        transcript_lines = []
        for turn in record.turns:
            transcript_lines.append(f"{turn.speaker_name}: {turn.text}")

        full_transcript = header + "\n" + "\n".join(transcript_lines)

        # Ingest the full conversation as one memory
        try:
            memory = await self.pipeline.ingest(
                content=full_transcript,
                source="voice",
                session_context=f"Voice conversation captured via ambient listening. "
                                f"Participants: {', '.join(unique_participants)}. "
                                f"Duration: {record.duration_seconds:.0f} seconds.",
            )
            memory_ids.append(memory.id)
        except Exception as e:
            print(f"  ⚠ Full conversation ingestion error: {e}")

        # Also ingest USER turns individually for richer personal memories
        user_turns = [t for t in record.turns if t.speaker_label == "USER"]
        if user_turns and len(user_turns) >= 2:
            user_content = "\n".join(
                f"I said: {t.text}" for t in user_turns
            )
            user_context = (
                f"My own words from a conversation with "
                f"{', '.join(p for p in unique_participants if p != 'You')}."
            )
            try:
                mem = await self.pipeline.ingest(
                    content=user_content,
                    source="voice",
                    session_context=user_context,
                )
                memory_ids.append(mem.id)
            except Exception:
                pass

        return memory_ids

    # ── Callbacks ────────────────────────────────────────────────────────

    def set_turn_callback(self, callback: Callable):
        """Register callback for each new transcript turn (for live UI)."""
        self._on_turn = callback

    def set_conversation_end_callback(self, callback: Callable):
        """Register callback for when a conversation is finalized."""
        self._on_conversation_end = callback

    # ── Queries ──────────────────────────────────────────────────────────

    def get_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get conversation records with pagination."""
        sorted_convs = sorted(
            self._conversations,
            key=lambda c: c.start_time or datetime.min,
            reverse=True,
        )
        return [c.to_dict() for c in sorted_convs[offset:offset + limit]]

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        """Get a specific conversation by ID."""
        for c in self._conversations:
            if c.id == conv_id:
                return c.to_dict()
        return None

    def get_current_turns(self) -> List[Dict]:
        """Get turns from the current (in-progress) conversation."""
        return [
            {
                "speaker_label": t.speaker_label,
                "speaker_name": t.speaker_name,
                "text": t.text,
                "timestamp": t.timestamp,
                "confidence": t.confidence,
            }
            for t in self._current_turns
        ]

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_conversation(self, record: ConversationRecord):
        """Save a conversation record to disk."""
        path = Path(f"{self.data_dir}/conversations/{record.id}.json")
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _load_conversations(self):
        """Load saved conversation records from disk."""
        conv_dir = Path(f"{self.data_dir}/conversations")
        if not conv_dir.exists():
            return
        for f in sorted(conv_dir.glob("conv_*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                turns = [
                    ConversationTurn(**t)
                    for t in data.get("turns", [])
                ]
                record = ConversationRecord(
                    id=data["id"],
                    turns=turns,
                    participants=data.get("participants", []),
                    start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
                    end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
                    duration_seconds=data.get("duration_seconds", 0),
                    memory_ids=data.get("memory_ids", []),
                    auto_ingested=data.get("auto_ingested", False),
                )
                self._conversations.append(record)
            except Exception:
                pass

        if self._conversations:
            print(f"  📂 Loaded {len(self._conversations)} saved conversations")

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_conversations": len(self._conversations),
            "current_turns": len(self._current_turns),
            "total_ingested": sum(1 for c in self._conversations if c.auto_ingested),
        }
