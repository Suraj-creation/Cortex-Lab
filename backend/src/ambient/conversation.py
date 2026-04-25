"""
Conversation Segmenter + Gemini Summarizer + Dual Storage
Groups speaker-labeled transcript turns into ConversationRecord objects.
Detects conversation boundaries via:
  - Hard boundary: >2 min silence = end of conversation
  - Soft boundary: Gemini topic-shift detection within continuous speech

On finalization:
  1. Gemini extracts structured knowledge (facts, decisions, action items, opinions)
  2. Structured summaries → vector store (for RAG retrieval, importance >= 5 only)
  3. Raw cleaned transcripts → DuckDB (for audit trail + analytics)

This is THE bridge between the voice stack and the existing Agentic RAG system.
"""

import time
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ConversationTurn:
    speaker_label: str        # "USER", "SPEAKER_A", etc.
    speaker_name: str = ""    # Resolved name: "Suraj", "Sarah"
    text: str = ""
    timestamp: float = 0.0    # Unix-like timestamp (seconds since ambient start)
    confidence: float = 0.0   # STT confidence
    speaker_confidence: float = 0.0  # Speaker-match confidence (voice ID)
    live_turn_id: str = ""          # Live session turn identifier
    session_id: str = ""
    source_platform: str = "ambient"
    retention_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationRecord:
    id: str = field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    turns: List[ConversationTurn] = field(default_factory=list)
    session_id: str = ""
    source_platform: str = "ambient"
    participants: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_ids: List[str] = field(default_factory=list)
    auto_ingested: bool = False
    # Phase 2: Gemini summary
    gemini_summary: Optional[Dict[str, Any]] = None
    topic_labels: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    # Phase 5: Topic segments
    topic_segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "source_platform": self.source_platform,
            "turns": [
                {
                    "speaker_label": t.speaker_label,
                    "speaker_name": t.speaker_name,
                    "text": t.text,
                    "timestamp": t.timestamp,
                    "confidence": t.confidence,
                    "speaker_confidence": t.speaker_confidence,
                    "live_turn_id": t.live_turn_id,
                    "session_id": t.session_id,
                    "source_platform": t.source_platform,
                    "retention_trace": t.retention_trace,
                }
                for t in self.turns
            ],
            "participants": self.participants,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 1),
            "memory_ids": self.memory_ids,
            "auto_ingested": self.auto_ingested,
            "gemini_summary": self.gemini_summary,
            "topic_labels": self.topic_labels,
            "importance_score": self.importance_score,
            "topic_segments": self.topic_segments,
        }


class ConversationSegmenter:
    """
    Groups transcript turns into conversations, detects boundaries,
    and feeds completed conversations into the existing MemoryIngestionPipeline.

    Phase 2: Gemini-powered conversation summarizer extracts structured knowledge.
    Phase 5: Topic-based segmentation detects topic shifts within conversations.
    Phase 6: Dual storage — summaries→vectors, raw→DuckDB.
    """

    SILENCE_THRESHOLD_S = 120   # 2 min silence = conversation end
    MIN_TURNS = 2               # Minimum turns to count as a conversation
    MERGE_THRESHOLD_S = 5       # Merge turns from same speaker within 5s
    TOPIC_CHECK_INTERVAL = 10   # Check for topic shift every N turns
    IMPORTANCE_THRESHOLD = 5    # Only ingest items with importance >= this

    def __init__(self, ingestion_pipeline=None, auto_ingest: bool = True,
                 data_dir: str = "data", gemini_api_key: str = None):
        """
        Args:
            ingestion_pipeline: MemoryIngestionPipeline instance (for auto-ingestion)
            auto_ingest: Whether to automatically ingest completed conversations
            data_dir: Base data directory for saving conversation records
            gemini_api_key: Gemini API key for conversation summarization + topic detection
        """
        self.pipeline = ingestion_pipeline
        self.auto_ingest = auto_ingest
        self.data_dir = data_dir
        Path(f"{data_dir}/conversations").mkdir(parents=True, exist_ok=True)

        # Gemini client for Phase 2 (summarizer) and Phase 5 (topic segmentation)
        self._gemini_client = None
        self._gemini_types = None
        if gemini_api_key:
            try:
                from google import genai
                from google.genai import types
                self._gemini_client = genai.Client(api_key=gemini_api_key)
                self._gemini_types = types
                print("  🧠 Gemini conversation summarizer initialized")
            except Exception as e:
                print(f"  ⚠ Gemini summarizer init failed: {e}")

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conv-summarizer")

        # DuckDB for Phase 6 (raw transcript audit trail)
        self._db = None
        self._db_backend = "none"
        self._init_duckdb()

        # Current in-progress conversation
        self._current_turns: List[ConversationTurn] = []
        self._last_turn_time: float = 0.0
        self._turns_since_topic_check: int = 0

        # Completed conversations
        self._conversations: List[ConversationRecord] = []

        # Callback for live transcript updates
        self._on_turn: Optional[Callable] = None
        self._on_conversation_end: Optional[Callable] = None

        # Load saved conversations
        self._load_conversations()

    # ── DuckDB Initialization (Phase 6) ─────────────────────────────────

    def _init_duckdb(self):
        """Initialize DuckDB tables for raw transcript audit trail."""
        try:
            import duckdb
            db_path = f"{self.data_dir}/cortex.duckdb"
            self._db = duckdb.connect(db_path)
            self._db_backend = "duckdb"
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS ambient_conversations (
                    id VARCHAR PRIMARY KEY,
                    session_id VARCHAR,
                    source_platform VARCHAR,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    duration_seconds FLOAT,
                    participants VARCHAR[],
                    turn_count INTEGER,
                    topic_labels VARCHAR[],
                    importance_score FLOAT DEFAULT 0,
                    gemini_summary JSON,
                    raw_transcript TEXT,
                    ingested BOOLEAN DEFAULT FALSE
                )
            """)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS ambient_conversation_turns (
                    id VARCHAR PRIMARY KEY,
                    conversation_id VARCHAR,
                    session_id VARCHAR,
                    source_platform VARCHAR,
                    turn_index INTEGER,
                    speaker VARCHAR,
                    speaker_name VARCHAR,
                    timestamp_s FLOAT,
                    text TEXT,
                    confidence FLOAT,
                    speaker_confidence FLOAT DEFAULT 0,
                    live_turn_id VARCHAR,
                    retention_trace JSON,
                    FOREIGN KEY (conversation_id) REFERENCES ambient_conversations(id)
                )
            """)

            # Ensure newer metadata columns exist for older databases.
            conv_schema = {
                row[1]
                for row in self._db.execute(
                    "PRAGMA table_info('ambient_conversations')"
                ).fetchall()
            }
            if "session_id" not in conv_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversations ADD COLUMN session_id VARCHAR"
                )
            if "source_platform" not in conv_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversations ADD COLUMN source_platform VARCHAR"
                )

            turn_schema = {
                row[1]
                for row in self._db.execute(
                    "PRAGMA table_info('ambient_conversation_turns')"
                ).fetchall()
            }
            if "session_id" not in turn_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversation_turns ADD COLUMN session_id VARCHAR"
                )
            if "source_platform" not in turn_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversation_turns ADD COLUMN source_platform VARCHAR"
                )
            if "speaker_confidence" not in turn_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversation_turns ADD COLUMN speaker_confidence FLOAT DEFAULT 0"
                )
            if "live_turn_id" not in turn_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversation_turns ADD COLUMN live_turn_id VARCHAR"
                )
            if "retention_trace" not in turn_schema:
                self._db.execute(
                    "ALTER TABLE ambient_conversation_turns ADD COLUMN retention_trace JSON"
                )

            # Best-effort migration from legacy ambient table names.
            # Only runs when legacy schemas are detected.
            try:
                conv_cols = {
                    row[1]
                    for row in self._db.execute("PRAGMA table_info('conversations')").fetchall()
                }
                if {"started_at", "raw_transcript", "turn_count"}.issubset(conv_cols):
                    self._db.execute("""
                        INSERT OR IGNORE INTO ambient_conversations
                        (id, session_id, source_platform, started_at, ended_at, duration_seconds, participants,
                         turn_count, topic_labels, importance_score, gemini_summary,
                         raw_transcript, ingested)
                        SELECT id, '', 'ambient', started_at, ended_at, duration_seconds, participants,
                               turn_count, topic_labels, importance_score, gemini_summary,
                               raw_transcript, ingested
                        FROM conversations
                    """)

                turn_cols = {
                    row[1]
                    for row in self._db.execute("PRAGMA table_info('conversation_turns')").fetchall()
                }
                if {"conversation_id", "turn_index", "speaker", "text"}.issubset(turn_cols):
                    self._db.execute("""
                        INSERT OR IGNORE INTO ambient_conversation_turns
                        (id, conversation_id, session_id, source_platform, turn_index, speaker, speaker_name,
                         timestamp_s, text, confidence)
                        SELECT id, conversation_id, '', 'ambient', turn_index, speaker, speaker_name,
                               timestamp_s, text, confidence
                        FROM conversation_turns
                    """)
            except Exception:
                pass
        except Exception as e:
            print(f"  ⚠ DuckDB init for conversations failed: {e}")
            self._init_sqlite_fallback()

    def _init_sqlite_fallback(self):
        """Fallback local DB when DuckDB is unavailable in the runtime environment."""
        try:
            import sqlite3

            db_path = f"{self.data_dir}/cortex.sqlite3"
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db_backend = "sqlite"

            self._db.execute("""
                CREATE TABLE IF NOT EXISTS ambient_conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    source_platform TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    duration_seconds REAL,
                    participants TEXT,
                    turn_count INTEGER,
                    topic_labels TEXT,
                    importance_score REAL DEFAULT 0,
                    gemini_summary TEXT,
                    raw_transcript TEXT,
                    ingested INTEGER DEFAULT 0
                )
            """)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS ambient_conversation_turns (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    session_id TEXT,
                    source_platform TEXT,
                    turn_index INTEGER,
                    speaker TEXT,
                    speaker_name TEXT,
                    timestamp_s REAL,
                    text TEXT,
                    confidence REAL,
                    speaker_confidence REAL DEFAULT 0,
                    live_turn_id TEXT,
                    retention_trace TEXT
                )
            """)
            conv_schema = {
                row[1]
                for row in self._db.execute("PRAGMA table_info('ambient_conversations')").fetchall()
            }
            if "session_id" not in conv_schema:
                self._db.execute("ALTER TABLE ambient_conversations ADD COLUMN session_id TEXT")
            if "source_platform" not in conv_schema:
                self._db.execute("ALTER TABLE ambient_conversations ADD COLUMN source_platform TEXT")

            turn_schema = {
                row[1]
                for row in self._db.execute("PRAGMA table_info('ambient_conversation_turns')").fetchall()
            }
            if "session_id" not in turn_schema:
                self._db.execute("ALTER TABLE ambient_conversation_turns ADD COLUMN session_id TEXT")
            if "source_platform" not in turn_schema:
                self._db.execute("ALTER TABLE ambient_conversation_turns ADD COLUMN source_platform TEXT")
            self._db.commit()
            print("  💾 SQLite fallback initialized for ambient conversations")
        except Exception as sqlite_exc:
            print(f"  ⚠ SQLite fallback init failed: {sqlite_exc}")
            self._db = None
            self._db_backend = "none"

    # ── Core Processing ──────────────────────────────────────────────────

    async def add_turn(self, speaker_label: str, speaker_name: str,
                       text: str, timestamp: float, confidence: float = 0.0,
                       speaker_confidence: float = 0.0,
                       live_turn_id: str = "",
                       session_id: str = "",
                       source_platform: str = "ambient",
                       retention_trace: Optional[Dict[str, Any]] = None):
        """
        Add a transcribed turn. Automatically segments and ingests.

        Called by AmbientService after: VAD → SpeakerID → Whisper → cleanup → here.
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
            speaker_confidence=speaker_confidence,
            live_turn_id=live_turn_id,
            session_id=session_id,
            source_platform=source_platform,
            retention_trace=dict(retention_trace or {}),
        )

        # Merge with previous turn if same speaker and close in time
        if (self._current_turns and
                self._current_turns[-1].speaker_label == speaker_label and
                self._current_turns[-1].session_id == session_id and
                (timestamp - self._current_turns[-1].timestamp) < self.MERGE_THRESHOLD_S):
            self._current_turns[-1].text += " " + text.strip()
            self._current_turns[-1].confidence = max(
                self._current_turns[-1].confidence, confidence
            )
            self._current_turns[-1].speaker_confidence = max(
                self._current_turns[-1].speaker_confidence, speaker_confidence
            )
            if live_turn_id:
                self._current_turns[-1].live_turn_id = live_turn_id
            if retention_trace:
                self._current_turns[-1].retention_trace = dict(retention_trace)
        else:
            self._current_turns.append(turn)

        self._last_turn_time = now
        self._turns_since_topic_check += 1

        # Phase 5: Check for topic shift every N turns (Gemini-powered)
        if (self._gemini_client and
                self._turns_since_topic_check >= self.TOPIC_CHECK_INTERVAL and
                len(self._current_turns) > self.TOPIC_CHECK_INTERVAL):
            self._turns_since_topic_check = 0
            topic_shifted = await self._check_topic_shift()
            if topic_shifted:
                # Soft boundary — finalize current segment as a topic block
                await self._finalize_conversation()

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
        """
        End current conversation. Pipeline:
        1. Build ConversationRecord
        2. Phase 2: Gemini extracts structured knowledge
        3. Phase 5: Topic segmentation
        4. Phase 6: Store raw in DuckDB
        5. Ingest structured summaries into vector store (importance >= 5)
        """
        if len(self._current_turns) < self.MIN_TURNS:
            # Preserve high-signal single turns (for wake-phrase retrieval queries,
            # commitments, and short but meaningful utterances).
            if len(self._current_turns) == 1:
                only_turn = self._current_turns[0]
                trace = dict(only_turn.retention_trace or {})
                decision = str(trace.get("decision", "")).strip().lower()
                score = float(trace.get("score", 0.0) or 0.0)
                tags = [str(t).strip().lower() for t in list(trace.get("tags", []) or [])]

                high_signal = (
                    decision == "keep"
                    and (
                        score >= 0.55
                        or any(tag in {"action_item", "question", "retrieval_query", "retrieve_wake", "technical"} for tag in tags)
                    )
                )
                if not high_signal:
                    self._current_turns = []
                    return
            else:
                self._current_turns = []
                return

        # Build conversation record
        participants = list(set(t.speaker_label for t in self._current_turns))
        start_time = datetime.now()
        end_time = datetime.now()

        record = ConversationRecord(
            turns=list(self._current_turns),
            session_id=self._current_turns[0].session_id if self._current_turns else "",
            source_platform=self._current_turns[0].source_platform if self._current_turns else "ambient",
            participants=participants,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=sum(
                max(0.5, len(t.text.split()) * 0.3) for t in self._current_turns
            ),
        )

        # Build raw transcript for processing
        raw_transcript = self._format_transcript(record)

        # Phase 2: Gemini summarization (extract structured knowledge)
        if self._gemini_client:
            try:
                summary = await self._gemini_summarize(record, raw_transcript)
                record.gemini_summary = summary
                record.topic_labels = summary.get("topics", [])
                record.importance_score = summary.get("overall_importance", 0)
                print(f"  🧠 Gemini summary: {len(summary.get('items', []))} items extracted, "
                      f"importance={record.importance_score}")
            except Exception as e:
                print(f"  ⚠ Gemini summarization failed: {e}")

        # Phase 5: Topic segmentation for long conversations
        if self._gemini_client and len(self._current_turns) > 15:
            try:
                segments = await self._segment_topics(record, raw_transcript)
                record.topic_segments = segments
            except Exception as e:
                print(f"  ⚠ Topic segmentation failed: {e}")

        # Phase 6: Store raw transcript in DuckDB (audit trail)
        self._store_in_duckdb(record, raw_transcript)

        # Smart ingestion: ingest structured summaries, not raw transcripts
        if self.auto_ingest and self.pipeline:
            try:
                memory_ids = await self._ingest_structured(record, raw_transcript)
                record.memory_ids = memory_ids
                record.auto_ingested = True
                print(f"  📝 Conversation {record.id} ingested "
                      f"({len(record.turns)} turns → {len(memory_ids)} memories)")
            except Exception as e:
                print(f"  ⚠ Conversation ingestion failed: {e}")

        # Store the record (JSON file)
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
        self._turns_since_topic_check = 0

    def _format_transcript(self, record: ConversationRecord) -> str:
        """Format conversation into readable transcript string."""
        unique_participants = list(set(t.speaker_name for t in record.turns))
        now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        header = (
            f"[Conversation: {now_str} | "
            f"Participants: {', '.join(unique_participants)} | "
            f"Duration: ~{record.duration_seconds:.0f}s, {len(record.turns)} turns]"
        )
        lines = [f"{t.speaker_name}: {t.text}" for t in record.turns]
        return header + "\n" + "\n".join(lines)

    # ── Phase 2: Gemini Conversation Summarizer ──────────────────────────

    async def _gemini_summarize(self, record: ConversationRecord,
                                 transcript: str) -> Dict[str, Any]:
        """
        Send transcript to Gemini and extract structured knowledge.
        Returns JSON with facts, decisions, action items, opinions, etc.
        """
        prompt = f"""You are analyzing a voice conversation transcript. Extract ONLY information worth remembering long-term.

TRANSCRIPT:
{transcript}

Extract the following categories. For EACH item, assign an importance_score from 1-10:
  10 = critical life decision, major action item
  7-9 = important fact, decision, or preference
  5-6 = moderately useful information
  1-4 = small talk, greetings, logistics noise

Categories to extract:
1. FACTS: Concrete facts mentioned (names, dates, numbers, places, events, technical details)
2. DECISIONS: Choices made or agreed upon ("we decided to...", "I'm going to...")
3. ACTION_ITEMS: Tasks assigned or committed to ("I need to...", "remind me to...", "by Friday")
4. OPINIONS: Beliefs or preferences expressed ("I think...", "I prefer...", "I believe...")
5. PERSONAL_INFO: Information about any participant (relationships, preferences, habits)
6. EMOTIONAL_CONTEXT: Notable emotional states (frustration, excitement, concern, enthusiasm)
7. KEY_QUOTES: Exact notable quotes worth preserving verbatim

Also provide:
- topics: List of 1-3 word topic labels for this conversation
- overall_importance: Single 1-10 score for the whole conversation
- summary: 1-2 sentence summary of the conversation

IGNORE: greetings, small talk, weather chat, filler, "okay bye", repetition.

Return ONLY valid JSON in this exact format:
{{
  "summary": "Brief 1-2 sentence summary",
  "topics": ["topic1", "topic2"],
  "overall_importance": 7,
  "items": [
    {{"category": "FACT", "content": "...", "speaker": "...", "importance_score": 8}},
    {{"category": "DECISION", "content": "...", "speaker": "...", "importance_score": 9}},
    {{"category": "ACTION_ITEM", "content": "...", "speaker": "...", "importance_score": 7}}
  ]
}}"""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, self._call_gemini, prompt)
        return result

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Synchronous Gemini call for summarization/topic detection."""
        config = self._gemini_types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.1,  # Low temp for factual extraction
            response_mime_type="application/json",
        )

        response = self._gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=config,
        )

        if not response or not response.text:
            return {"items": [], "topics": [], "overall_importance": 0, "summary": ""}

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"items": [], "topics": [], "overall_importance": 0,
                        "summary": text[:200]}

    # ── Phase 5: Topic-Based Segmentation ────────────────────────────────

    async def _check_topic_shift(self) -> bool:
        """
        Ask Gemini if the recent turns indicate a topic shift.
        Returns True if topic has changed.
        """
        if not self._gemini_client or len(self._current_turns) < self.TOPIC_CHECK_INTERVAL:
            return False

        # Get the last N turns for analysis
        recent = self._current_turns[-self.TOPIC_CHECK_INTERVAL:]
        earlier = self._current_turns[:-self.TOPIC_CHECK_INTERVAL]

        if len(earlier) < 3:
            return False

        recent_text = "\n".join(f"{t.speaker_name}: {t.text}" for t in recent)
        earlier_text = "\n".join(f"{t.speaker_name}: {t.text}" for t in earlier[-5:])

        prompt = f"""Has the conversation topic changed between these two segments?

EARLIER:
{earlier_text}

RECENT:
{recent_text}

Respond with ONLY valid JSON:
{{"topic_shifted": true/false, "old_topic": "...", "new_topic": "..."}}"""

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, self._call_gemini, prompt)
            return result.get("topic_shifted", False)
        except Exception:
            return False

    async def _segment_topics(self, record: ConversationRecord,
                               transcript: str) -> List[Dict[str, Any]]:
        """
        Segment a long conversation into topical chunks.
        Each segment gets its own topic label and importance score.
        """
        prompt = f"""Split this conversation transcript into topical segments.
Each segment should cover a distinct topic of discussion.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON:
{{
  "segments": [
    {{
      "topic": "Brief topic label (2-5 words)",
      "start_turn_index": 0,
      "end_turn_index": 5,
      "importance": 7,
      "summary": "1-sentence summary of this segment"
    }}
  ]
}}"""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, self._call_gemini, prompt)
        return result.get("segments", [])

    # ── Phase 6: Dual Storage ────────────────────────────────────────────

    def _store_in_duckdb(self, record: ConversationRecord, raw_transcript: str):
        """Store raw conversation data in DuckDB for audit trail."""
        if not self._db:
            return

        try:
            if self._db_backend == "sqlite":
                self._db.execute(
                    """
                    INSERT OR REPLACE INTO ambient_conversations
                    (id, session_id, source_platform, started_at, ended_at, duration_seconds, participants,
                     turn_count, topic_labels, importance_score, gemini_summary,
                     raw_transcript, ingested)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.session_id,
                        record.source_platform,
                        record.start_time.isoformat() if record.start_time else None,
                        record.end_time.isoformat() if record.end_time else None,
                        record.duration_seconds,
                        json.dumps(record.participants or []),
                        len(record.turns),
                        json.dumps(record.topic_labels or []),
                        record.importance_score,
                        json.dumps(record.gemini_summary) if record.gemini_summary else None,
                        raw_transcript,
                        0,
                    ),
                )

                for i, turn in enumerate(record.turns):
                    turn_id = f"{record.id}_t{i:03d}"
                    self._db.execute(
                        """
                        INSERT OR REPLACE INTO ambient_conversation_turns
                        (id, conversation_id, session_id, source_platform, turn_index, speaker, speaker_name,
                         timestamp_s, text, confidence, speaker_confidence,
                         live_turn_id, retention_trace)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            turn_id,
                            record.id,
                            turn.session_id or record.session_id,
                            turn.source_platform or record.source_platform,
                            i,
                            turn.speaker_label,
                            turn.speaker_name,
                            turn.timestamp,
                            turn.text,
                            turn.confidence,
                            turn.speaker_confidence,
                            turn.live_turn_id,
                            json.dumps(turn.retention_trace or {}),
                        ),
                    )

                self._db.commit()
                return

            self._db.execute("""
                INSERT OR REPLACE INTO ambient_conversations
                (id, session_id, source_platform, started_at, ended_at, duration_seconds, participants,
                 turn_count, topic_labels, importance_score, gemini_summary,
                 raw_transcript, ingested)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record.id,
                record.session_id,
                record.source_platform,
                record.start_time,
                record.end_time,
                record.duration_seconds,
                record.participants,
                len(record.turns),
                record.topic_labels,
                record.importance_score,
                json.dumps(record.gemini_summary) if record.gemini_summary else None,
                raw_transcript,
                False,
            ])

            for i, turn in enumerate(record.turns):
                turn_id = f"{record.id}_t{i:03d}"
                self._db.execute("""
                    INSERT OR REPLACE INTO ambient_conversation_turns
                    (id, conversation_id, session_id, source_platform, turn_index, speaker, speaker_name,
                     timestamp_s, text, confidence, speaker_confidence,
                     live_turn_id, retention_trace)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    turn_id,
                    record.id,
                    turn.session_id or record.session_id,
                    turn.source_platform or record.source_platform,
                    i,
                    turn.speaker_label,
                    turn.speaker_name, turn.timestamp, turn.text,
                    turn.confidence,
                    turn.speaker_confidence,
                    turn.live_turn_id,
                    json.dumps(turn.retention_trace or {}),
                ])

        except Exception as e:
            print(f"  ⚠ DuckDB store failed: {e}")

    # ── Smart Ingestion (Phase 2 + 6 combined) ──────────────────────────

    async def _ingest_structured(self, record: ConversationRecord,
                                  raw_transcript: str) -> List[str]:
        """
        Ingest STRUCTURED knowledge (not raw transcript) into vector store.
        Only items with importance >= IMPORTANCE_THRESHOLD get vectorized.
        Falls back to raw transcript ingestion if Gemini summary unavailable.
        """
        memory_ids = []
        unique_participants = list(set(t.speaker_name for t in record.turns))
        now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")

        # If we have a Gemini summary, ingest structured items
        if record.gemini_summary and record.gemini_summary.get("items"):
            items = record.gemini_summary["items"]
            important_items = [
                item for item in items
                if item.get("importance_score", 0) >= self.IMPORTANCE_THRESHOLD
            ]

            if important_items:
                # Group items into a structured knowledge chunk
                context_prefix = (
                    f"[Conversation: {now_str} | "
                    f"Participants: {', '.join(unique_participants)} | "
                    f"Topics: {', '.join(record.topic_labels)}]"
                )

                # Build structured content for ingestion
                content_parts = [context_prefix, ""]

                if record.gemini_summary.get("summary"):
                    content_parts.append(f"Summary: {record.gemini_summary['summary']}")
                    content_parts.append("")

                for item in important_items:
                    cat = item.get("category", "INFO")
                    content = item.get("content", "")
                    speaker = item.get("speaker", "")
                    score = item.get("importance_score", 0)
                    if speaker:
                        content_parts.append(f"[{cat}] {speaker}: {content} (importance: {score})")
                    else:
                        content_parts.append(f"[{cat}] {content} (importance: {score})")

                structured_content = "\n".join(content_parts)

                try:
                    memory = await self.pipeline.ingest(
                        content=structured_content,
                        session_id=record.session_id,
                        source="voice",
                        session_context=(
                            f"Structured knowledge extracted from voice conversation. "
                            f"Participants: {', '.join(unique_participants)}. "
                            f"Topics: {', '.join(record.topic_labels)}."
                        ),
                    )
                    memory_ids.append(memory.id)
                except Exception as e:
                    print(f"  ⚠ Structured ingestion error: {e}")

            # Also ingest high-importance topic segments individually
            if record.topic_segments:
                for seg in record.topic_segments:
                    if seg.get("importance", 0) >= self.IMPORTANCE_THRESHOLD:
                        seg_content = (
                            f"[{now_str} | Topic: {seg.get('topic', 'Unknown')}]\n"
                            f"{seg.get('summary', '')}"
                        )
                        try:
                            mem = await self.pipeline.ingest(
                                content=seg_content,
                                session_id=record.session_id,
                                source="voice",
                                session_context=f"Topic segment from conversation: {seg.get('topic', '')}",
                            )
                            memory_ids.append(mem.id)
                        except Exception:
                            pass

            # Mark as ingested in DuckDB
            if self._db and memory_ids:
                try:
                    if self._db_backend == "sqlite":
                        self._db.execute(
                            "UPDATE ambient_conversations SET ingested = 1 WHERE id = ?",
                            (record.id,),
                        )
                        self._db.commit()
                    else:
                        self._db.execute(
                            "UPDATE ambient_conversations SET ingested = TRUE WHERE id = ?",
                            [record.id]
                        )
                except Exception:
                    pass

            return memory_ids

        # Fallback: no Gemini summary available — ingest raw transcript
        return await self._ingest_raw_fallback(record, raw_transcript, unique_participants)

    async def _ingest_raw_fallback(self, record: ConversationRecord,
                                    raw_transcript: str,
                                    unique_participants: List[str]) -> List[str]:
        """Fallback ingestion when Gemini summarizer is unavailable."""
        memory_ids = []

        try:
            memory = await self.pipeline.ingest(
                content=raw_transcript,
                session_id=record.session_id,
                source="voice",
                session_context=f"Voice conversation captured via ambient listening. "
                                f"Participants: {', '.join(unique_participants)}. "
                                f"Duration: {record.duration_seconds:.0f} seconds.",
            )
            memory_ids.append(memory.id)
        except Exception as e:
            print(f"  ⚠ Raw conversation ingestion error: {e}")

        # Also ingest USER turns individually
        user_turns = [t for t in record.turns if t.speaker_label == "USER"]
        if user_turns and len(user_turns) >= 2:
            user_content = "\n".join(f"I said: {t.text}" for t in user_turns)
            user_context = (
                f"My own words from a conversation with "
                f"{', '.join(p for p in unique_participants if p != 'You')}."
            )
            try:
                mem = await self.pipeline.ingest(
                    content=user_content,
                    session_id=record.session_id,
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
                "speaker_confidence": t.speaker_confidence,
                "live_turn_id": t.live_turn_id,
                "session_id": t.session_id,
                "source_platform": t.source_platform,
                "retention_trace": t.retention_trace,
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
                    session_id=data.get("session_id", ""),
                    source_platform=data.get("source_platform", "ambient"),
                    participants=data.get("participants", []),
                    start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
                    end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
                    duration_seconds=data.get("duration_seconds", 0),
                    memory_ids=data.get("memory_ids", []),
                    auto_ingested=data.get("auto_ingested", False),
                    gemini_summary=data.get("gemini_summary"),
                    topic_labels=data.get("topic_labels", []),
                    importance_score=data.get("importance_score", 0.0),
                    topic_segments=data.get("topic_segments", []),
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
