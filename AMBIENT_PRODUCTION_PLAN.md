# Cortex Lab — Ambient Intelligence Production Plan

## Current State

- **STT**: faster-whisper (traditional) + GeminiSTT (cloud) — both working
- **TTS**: Piper (traditional) + GeminiTTS (`gemini-2.5-flash-preview-tts`) — both working
- **Speaker ID**: ECAPA-TDNN via SpeechBrain — **FAILING on Python 3.14**, currently `None` with graceful fallback
- **VAD**: Silero VAD v5 (ONNX) — working
- **Conversation Segmenter**: 2-min silence boundary, 5s same-speaker merge — working but naive
- **Ingestion Pipeline**: 13 stages — working but ingests raw noisy transcripts
- **Wake Word**: Not implemented
- **Noise/Filler Filtering**: Not implemented

---

## Phase 1 — Speech Cleanup Filter

**Goal**: Remove filler words, disfluencies, and low-confidence junk before ingestion.

**New file**: `backend/src/ambient/speech_cleanup.py`

**What it does**:
- **Filler word removal**: Strip "um", "uh", "like", "you know", "basically", "right", "I mean", "so", "well", "actually" when used as fillers (not meaningful content)
- **Disfluency removal**: Repeated words ("I I I think"), false starts ("I was going — I went to")
- **Confidence gating**: Drop segments where faster-whisper's word-level confidence < 0.3 (likely noise, coughs, background chatter)
- **STT error normalization**: Common ASR mistakes ("gonna" → "going to", "wanna" → "want to")
- **Minimum content gate**: Drop turns that are just filler with no substance (e.g., "um yeah uh")

**Integration point**: Insert between transcription and `conversation.add_turn()` in `_process_speech()`

**Estimated complexity**: Low — regex + word-level confidence from faster-whisper

---

## Phase 2 — Gemini Conversation Summarizer

**Goal**: Extract structured knowledge from conversations instead of ingesting raw transcripts.

**Enhancement**: Modify `_finalize_conversation()` in `backend/src/ambient/conversation.py`

**What it does**:
- When a conversation finalizes (2-min silence or manual stop), send the cleaned transcript to Gemini with a structured extraction prompt
- **Extraction targets**:
  - **Facts & Information**: Concrete facts mentioned ("Python 3.14 breaks SpeechBrain")
  - **Decisions Made**: Choices agreed upon ("We'll use resemblyzer instead")
  - **Action Items**: Tasks assigned ("Need to implement wake word by Friday")
  - **Opinions & Preferences**: Expressed views ("I prefer Gemini over local models")
  - **Personal Information**: Names, relationships, preferences mentioned
  - **Emotional Context**: Sentiment of the conversation (frustrated, excited, neutral)
  - **Key Quotes**: Exact notable quotes worth preserving
- Returns structured JSON that gets ingested as high-quality knowledge chunks
- **Importance scoring**: Each extracted item gets a 1-10 importance score; only items ≥ 5 get vector-indexed
- **Filter out**: Small talk, greetings, filler conversations, logistics noise

**Why this matters**: Instead of storing "um so I think we should uh maybe use the uh resemblyzer thing because like speechbrain doesn't work", we store: `{"decision": "Replace SpeechBrain with resemblyzer for speaker identification", "reason": "SpeechBrain incompatible with Python 3.14", "importance": 9}`

---

## Phase 3 — Speaker Recognition with Resemblyzer

**Goal**: Replace broken SpeechBrain ECAPA-TDNN with a Python 3.14-compatible speaker ID system.

**Modification**: `backend/src/ambient/speaker_id.py`

**Why resemblyzer**:
- Uses GE2E (Generalized End-to-End) model — lightweight, battle-tested
- 256-dimensional embeddings (vs ECAPA-TDNN's 192)
- Pure Python + NumPy + librosa — no PyTorch/TorchAudio dependency issues
- ~10MB model size, CPU-only, ~30ms per embedding
- Works on Python 3.14 (no C extension compatibility issues)
- `pip install resemblyzer` — single dependency

**What changes**:
- Replace `from speechbrain.inference.speaker import EncoderClassifier` with `from resemblyzer import VoiceEncoder, preprocess_wav`
- `VoiceEncoder("cpu")` replaces `EncoderClassifier.from_hparams()`
- `encoder.embed_utterance(wav)` replaces `encode_batch()`
- Keep existing enrollment flow: record 10-30s → segment into 3s chunks → average embeddings → save voiceprint
- Keep existing identification logic: cosine similarity ≥ 0.70 → "USER", else online clustering
- Keep alias system and voiceprint storage in `data/voiceprints/`

**Enrollment endpoint**: `/api/ambient/enroll` will work again (currently returns 503)

---

## Phase 4 — Wake Word Detection ("Cortex")

**Goal**: Always-on lightweight wake word detection that activates ambient listening.

**New file**: `backend/src/ambient/wake_word.py`

**Technology**: openWakeWord (ONNX-based, ~5MB models)

**How it works**:
1. Wake word detector runs continuously on a separate lightweight audio stream (low CPU)
2. Listens for "Cortex" (or "Hey Cortex")
3. On detection → triggers `AmbientService.start()` to begin full pipeline (VAD → STT → Speaker ID → Segmenter)
4. After configurable silence timeout (e.g., 2 min no speech) → auto-pause back to wake-word-only mode
5. Optional: play a subtle acknowledgment sound on wake word detection

**Modes**:
- **Always-on**: Wake word detector runs 24/7, full pipeline only when activated
- **Manual**: User clicks Start in UI (current behavior, preserved as option)
- **Hybrid**: Wake word OR manual start

**Integration**:
- New config option: `wake_word_enabled: bool = False` in AmbientConfig
- New endpoint: `/api/ambient/wake-word/enable|disable`
- Frontend: Toggle in Settings tab of AmbientPanel

**Training custom wake word**: openWakeWord supports fine-tuning with ~50 positive samples. Can record "Cortex" samples during enrollment.

---

## Phase 5 — Topic-Based Conversation Segmentation

**Goal**: Replace naive 2-min silence boundary with intelligent topic-based segmentation.

**Enhancement**: `backend/src/ambient/conversation.py`

**Current problem**: A 30-minute conversation gets split only on 2-min silences. If someone talks continuously for 30 minutes about 5 different topics, it's stored as one monolithic blob.

**New approach**:
1. **Sliding window analysis**: Every N turns (e.g., 10), send the recent window to Gemini with prompt: "Has the topic changed? If so, where's the boundary?"
2. **Topic labels**: Each segment gets a topic label ("Project Architecture Discussion", "Weekend Plans", "Bug Triage")
3. **Dual boundary detection**:
   - **Silence-based**: Keep 2-min silence as a hard boundary (conversation clearly ended)
   - **Topic-based**: Gemini detects topic shifts within continuous speech → soft boundary → finalize previous topic segment, start new one
4. **Importance per topic**: Each topic segment gets scored 1-10. "Weekend Plans" = 2 (skip ingestion), "Architecture Decision" = 9 (ingest immediately)
5. **Context carryover**: If a topic continues after a brief tangent, merge the segments

**Conversation chunking for long conversations (5-30 min)**:
- Max chunk size: ~500 words per topic segment
- If a single topic runs > 500 words, split at natural paragraph boundaries
- Each chunk retains: topic label, participants, timestamp range, previous topic context

---

## Phase 6 — Dual Storage Strategy

**Goal**: Store processed summaries in vectors for retrieval, raw transcripts in DuckDB for audit.

**What gets stored where**:

### Vector Store (for RAG retrieval)
- Gemini-extracted structured summaries (Phase 2 output)
- Key facts, decisions, action items
- Each with importance score, topic label, timestamp, participants
- Only items with importance ≥ 5 get embedded and indexed

### DuckDB (for audit trail and replay)
- Raw cleaned transcripts (Phase 1 output — fillers removed but otherwise verbatim)
- Full conversation metadata: start/end time, duration, participants, turn count
- Per-turn data: speaker, timestamp, raw text, confidence scores
- Conversation-level Gemini analysis JSON
- Queryable via SQL for analytics ("How many conversations this week?", "Average conversation length?")

**New table schema**:
```sql
CREATE TABLE conversations (
    id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    duration_seconds FLOAT,
    participants TEXT[],
    turn_count INTEGER,
    topic_labels TEXT[],
    importance_score FLOAT,
    gemini_summary JSON,
    raw_transcript TEXT,
    ingested BOOLEAN DEFAULT FALSE
);

CREATE TABLE conversation_turns (
    id VARCHAR PRIMARY KEY,
    conversation_id VARCHAR REFERENCES conversations(id),
    turn_index INTEGER,
    speaker VARCHAR,
    timestamp TIMESTAMP,
    text TEXT,
    confidence FLOAT,
    duration_seconds FLOAT
);
```

---

## Implementation Priority

### Recommended: Start with Phase 1 + 2 + 3 together

These three phases are **synergistic**:
- Phase 1 (cleanup) improves Phase 2 (summarizer gets cleaner input)
- Phase 3 (speaker ID) improves Phase 2 (summarizer knows WHO said what)
- All three are required for quality ingestion

### Sequence:
1. **Phase 3** first — get speaker ID working again (unblocks enrollment, speaker labels)
2. **Phase 1** next — clean up transcripts (unblocks quality ingestion)
3. **Phase 2** next — intelligent summarization (transforms what gets stored)
4. **Phase 6** alongside Phase 2 — dual storage for summaries + raw
5. **Phase 5** — topic segmentation (enhances long conversations)
6. **Phase 4** last — wake word (convenience feature, not core quality)

### Dependencies:
```
Phase 3 (Speaker ID) ──→ Phase 1 (Cleanup) ──→ Phase 2 (Summarizer) ──→ Phase 5 (Topics)
                                                       │
                                                       ▼
                                                Phase 6 (Dual Storage)

Phase 4 (Wake Word) — independent, can be done anytime
```

---

## Technical Requirements

| Phase | New Dependencies | Estimated Files Changed |
|-------|-----------------|------------------------|
| 1 | None (regex + existing whisper confidence) | `speech_cleanup.py` (new), `__init__.py` |
| 2 | None (uses existing Gemini SDK) | `conversation.py`, `__init__.py` |
| 3 | `resemblyzer` (~10MB) | `speaker_id.py`, `enrollment.py`, `requirements.txt` |
| 4 | `openwakeword` (~5MB ONNX models) | `wake_word.py` (new), `__init__.py`, `config.py`, `server.py`, frontend |
| 5 | None (uses existing Gemini SDK) | `conversation.py` |
| 6 | None (uses existing DuckDB) | `conversation.py`, storage layer |

---

## Gemini API Limitations (Confirmed)

- **No voice cloning**: Cannot replicate a user's voice for TTS
- **No speaker embeddings**: Cannot extract voiceprints for identification
- **No streaming STT**: Gemini STT is request-response, not real-time streaming
- **Speaker diarization**: Not available via Gemini API — must be done locally
- **TTS model**: Must use `gemini-2.5-flash-preview-tts` specifically (not `gemini-2.5-flash`)
