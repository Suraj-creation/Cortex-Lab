# Cortex Lab: Speech-to-Text & Text-to-Speech Integration Plan
## Complete STT/TTS Architecture Aligned with 9-Layer Agentic RAG

---

## 📋 Table of Contents
S
1. [Why We're Building This](#1-why-were-building-this)
2. [Current System State — What Already Exists](#2-current-system-state)
3. [Hardware Budget — What We Have to Work With](#3-hardware-budget)
4. [Architecture Decision: Final Model Choices](#4-architecture-decision-final-model-choices)
5. [TIER 0: Microphone + Ring Buffer](#5-tier-0-microphone--ring-buffer)
6. [TIER 1: Voice Activity Detection](#6-tier-1-voice-activity-detection)
7. [TIER 2: Speaker Identification](#7-tier-2-speaker-identification)
8. [TIER 3: Transcription (STT)](#8-tier-3-transcription-stt)
9. [TIER 4: Text-to-Speech (TTS)](#9-tier-4-text-to-speech-tts)
10. [Conversation Segmenter + Auto-Ingestion Bridge](#10-conversation-segmenter--auto-ingestion-bridge)
11. [VRAM Coordination — Mutual Exclusion](#11-vram-coordination)
12. [Backend API Endpoints](#12-backend-api-endpoints)
13. [Frontend Components](#13-frontend-components)
14. [Data Models](#14-data-models)
15. [File Structure](#15-file-structure)
16. [Dependencies](#16-dependencies)
17. [Resource Budget Summary](#17-resource-budget-summary)
18. [Privacy & Ethics](#18-privacy--ethics)
19. [Implementation Order](#19-implementation-order)

---

## 1. Why We're Building This

### 1.1 The Gap in Our Vision

Cortex Lab's Vision-Plan.md states:

> *"Cortex Lab is your personal cognitive operating system — I remember every conversation we've had."*

Right now, Cortex Lab only remembers **typed** conversations. But human cognition doesn't work through keyboards — most of our thinking, decisions, and meaningful interactions happen through **speech**. If Cortex Lab only captures text input, it misses:

- **Verbal conversations** with colleagues, friends, family — the richest source of episodic memories
- **Spoken reflections** — "thinking out loud" moments that reveal beliefs and emotional states
- **Meetings and discussions** — where career decisions, project pivots, and relationship dynamics unfold
- **Quick voice notes** — ideas captured while walking, driving, or away from the keyboard

Without voice, Cortex Lab is a **text-only brain** in a **speech-first world**.

### 1.2 Why STT (Speech-to-Text)?

STT transforms Cortex Lab from a **typed journal** into a **continuous cognitive capture system**:

| Without STT | With STT |
|---|---|
| User must type every memory | Conversations auto-captured as memories |
| Misses 90%+ of daily interactions | Captures verbal meetings, calls, discussions |
| Keyboard-only input | Voice queries while hands are busy |
| No ambient awareness | 24/7 passive listening with cascading tiers |
| Single-user text chat | Multi-speaker conversation tracking |

**How STT connects to our Agentic RAG:**
- Voice transcripts → `MemoryIngestionPipeline.ingest(source="voice")` → same 11-stage pipeline
- Same DuckDB + FAISS + Knowledge Graph storage
- Same 5-channel hybrid retrieval finds voice memories alongside typed ones
- Same 5 specialized agents reason over ALL memories regardless of source
- Entity extraction catches names spoken in conversation → Knowledge Graph grows richer
- Emotion detection works on transcribed speech content → more natural emotional data
- Belief evolution tracking catches verbal opinion changes → richer belief deltas

### 1.3 Why TTS (Text-to-Speech)?

TTS transforms Cortex Lab from a **text display** into a **conversational partner**:

| Without TTS | With TTS |
|---|---|
| Read answers on screen | Hear answers spoken naturally |
| Screen-bound interaction | Eyes-free, hands-free interaction |
| Feels like a search engine | Feels like talking to a real assistant |
| Can't interact while cooking/driving | Full interaction while multitasking |
| No voice personality | Consistent voice identity for Cortex |

**How TTS connects to our Agentic RAG:**
- Orchestrator generates answer → TTS speaks it while text streams on screen
- Evidence cards still display visually, but the answer is also spoken
- Thinking traces shown on screen, final answer spoken — best of both worlds
- Wake word "Hey Cortex" triggers active mode → STT captures query → RAG processes → TTS speaks answer
- Natural conversation loop: Speak → Transcribe → RAG → Generate → Speak back

### 1.4 The Complete Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE CORTEX LAB VOICE LOOP                            │
│                                                                         │
│  AMBIENT MODE (passive, 24/7):                                          │
│  Microphone → VAD → Speaker ID → Whisper STT → Conversation Segmenter  │
│       → MemoryIngestionPipeline.ingest(source="voice")                  │
│       → DuckDB + FAISS + Knowledge Graph (same as typed memories)       │
│                                                                         │
│  ACTIVE MODE (query, on-demand):                                        │
│  Wake Word "Hey Cortex" → STT captures query                            │
│       → RAG Engine (Orchestrator → 5 Agents → Hybrid Retrieval)         │
│       → LLM generates answer with evidence                              │
│       → TTS speaks the answer aloud                                     │
│       → Evidence cards display on screen                                │
│                                                                         │
│  TYPED MODE (existing, unchanged):                                      │
│  Keyboard → Chat Panel → RAG Engine → LLM → Text response              │
│       (optionally TTS speaks the response too)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current System State

### 2.1 What Already Exists (Production-Ready Back Half)

These components are built and tested. Voice data will plug directly into them:

| Component | File | Status | How Voice Connects |
|---|---|---|---|
| **Memory Ingestion Pipeline** | `backend/src/ingestion/__init__.py` (553 lines) | ✅ Built | `ingest(content=transcript, source="voice")` |
| **CausalMemoryObject** | `backend/src/models/__init__.py` | ✅ Built | `source: "voice"` already a valid value (line 106) |
| **DuckDB Metadata Store** | `backend/src/storage/metadata_store.py` | ✅ Built | Stores voice memories with same schema |
| **FAISS Vector Store** | `backend/src/storage/vector_store.py` | ✅ Built | Embeddings from transcripts stored identically |
| **Knowledge Graph** | `backend/src/storage/knowledge_graph.py` | ✅ Built | Entities from spoken conversations auto-added |
| **BGE-large Embeddings** | `backend/src/models/embeddings.py` | ✅ Built | Same 1024d embeddings for voice transcripts |
| **CrossEncoder Reranker** | `backend/src/models/embeddings.py` | ✅ Built | Reranks voice memories same as typed ones |
| **5-Channel Hybrid Retriever** | `backend/src/retrieval/hybrid_retriever.py` | ✅ Built | Dense+sparse+graph+temporal+proposition search |
| **5 Specialized Agents** | `backend/src/agents/specialized.py` | ✅ Built | Timeline, Causal, Reflection, Planning, Arbitration |
| **Agent Orchestrator** | `backend/src/agents/orchestrator.py` | ✅ Built | Routes queries, Self-RAG, FLARE, CRAG |
| **Fine-Tuned 7B LLM** | `fine_tuned/stage15_spin/merged/` | ✅ Built | 15-stage fine-tuned, runs 4-bit on GPU |
| **3-Level Cache** | `backend/src/cache/__init__.py` | ✅ Built | Caches voice queries identically |
| **FastAPI Server** | `backend/server.py` (762 lines) | ✅ Built | Add new ambient endpoints alongside existing |
| **Next.js Frontend** | `frontend/src/` (9 components) | ✅ Built | Add new Ambient panel alongside existing views |
| **`/api/memories/ingest` endpoint** | `backend/server.py:663` | ✅ Built | Already accepts `source` parameter |

### 2.2 Critical Integration Points

**The ingestion bridge is the key connection.** The existing `MemoryIngestionPipeline.ingest()` method signature:

```python
async def ingest(self, content: str, session_id: str = "",
                 source: str = "chat", session_context: str = "") -> CausalMemoryObject
```

Voice transcripts just call this with `source="voice"` and `session_context` set to the conversation context. The entire 11-stage pipeline (classify → emotion → entities → topics → importance → propositions → context prefix → embedding → store → belief detection) runs identically on voice transcripts as on typed text.

**The RAG engine in `engine.py` already exposes:**
- `rag_chat()` — full RAG with generation (for active voice queries)
- `rag_retrieve()` — retrieval only for streaming (for streaming voice answers)
- `ingest_memory()` — manual ingestion (for voice conversation records)

No changes needed to these methods. Voice is just a new **input source**, not a new processing pipeline.

---

## 3. Hardware Budget

### 3.1 Current Hardware (Verified)

```
GPU:    NVIDIA RTX 4000 Ada Generation — 20,475 MiB (20 GB) VRAM
CPU:    Intel Core i9-14900K — 32 threads (8P + 16E cores)
RAM:    32 GB DDR5
Disk:   SSD (fast enough for all operations)
Audio:  HDA Intel PCH — ALC222 Analog (capture card 0, device 0)
Audio:  PipeWire 1.0.5 (audio system)
OS:     Linux (Ubuntu)
```

### 3.2 VRAM Budget When Server Is Running

| Component | VRAM | Notes |
|---|---|---|
| DeepSeek-R1-7B (4-bit) | ~14,300 MiB | Primary LLM, always loaded |
| BGE-large-en-v1.5 (1024d) | ~0 MiB | Runs on CPU |
| BGE-reranker-v2-m3 | ~0 MiB | Runs on CPU |
| System/Xorg/Desktop | ~500 MiB | Fixed overhead |
| **Used** | **~14,800 MiB** | |
| **Free VRAM** | **~5,600 MiB** | Available for STT/TTS |

### 3.3 CPU Budget

With i9-14900K (32 threads), the existing system uses <10% CPU at idle. We have **abundant CPU headroom** for:
- Ring buffer + audio thread: <0.1% CPU
- Silero VAD: <0.1% CPU (always-on)
- ECAPA-TDNN speaker ID: ~1% CPU (only during speech)
- Conversation segmentation: negligible

### 3.4 RAM Budget

32 GB total, ~8.3 GB used. **~22 GB available.** More than enough for all STT/TTS models on CPU.

### 3.5 Budget Constraint Summary

```
HARD CONSTRAINT: 5.6 GB free VRAM — STT and TTS must fit within this
SOFT CONSTRAINT: 22 GB free RAM — generous, no concern  
SOFT CONSTRAINT: 32 CPU threads — generous, no concern
RULE: STT GPU and LLM inference MUST NOT run simultaneously
      (mutual exclusion via asyncio Lock — see Section 11)
```

---

## 4. Architecture Decision: Final Model Choices

### 4.1 STT: faster-whisper `small` (CTranslate2)

| Property | Value |
|---|---|
| **Model** | `faster-whisper` with `small` variant |
| **Backend** | CTranslate2 (optimized inference engine) |
| **VRAM** | ~500 MiB on GPU, or CPU-only (3x slower but works) |
| **Accuracy** | WER ~7.6% on LibriSpeech (excellent for conversational speech) |
| **Speed** | ~1-2s per 10s of speech (GPU), ~4-6s per 10s (CPU) |
| **Languages** | 99 languages, auto-detect per segment |
| **Word timestamps** | ✅ Available |
| **License** | MIT |
| **Disk** | ~500 MB |

**Why faster-whisper over alternatives:**

| Model | VRAM | Speed | Accuracy | Edge-Ready | Verdict |
|---|---|---|---|---|---|
| **faster-whisper small** | 500 MB | 1-2s/10s GPU | WER ~7.6% | ✅ | **BEST PICK** |
| whisper.cpp (small) | 0 (CPU) | 4-6s/10s | WER ~7.6% | ✅ | Backup if VRAM tight |
| OpenAI Whisper (small) | ~1 GB | 3-4s/10s GPU | WER ~7.6% | ⚠️ Heavier | Original, slower |
| Whisper large-v3 | ~3 GB | 4-6s/10s GPU | WER ~4% | ❌ Too much VRAM | Overkill |
| Deepgram Nova-3 | Cloud | Real-time | WER ~6% | ❌ Cloud, not local | Violates privacy |
| Vosk | 0 (CPU) | Real-time | WER ~12% | ✅ | Much worse accuracy |

**Decision:** `faster-whisper small` on GPU (500 MB VRAM) with CPU fallback. Fits within our 5.6 GB free VRAM budget with room to spare. CTranslate2 is 4x faster than standard Whisper with identical accuracy.

### 4.2 VAD: Silero VAD v5

| Property | Value |
|---|---|
| **Model** | Silero VAD v5 |
| **Source** | `torch.hub.load('snakers4/silero-vad')` |
| **VRAM** | 0 (CPU only) |
| **RAM** | ~10 MB |
| **CPU** | <1 ms per 30 ms frame |
| **Accuracy** | 97%+ speech detection, <2% false positive |
| **License** | MIT |
| **Disk** | 2 MB |

**Why Silero VAD:** There is no alternative that comes close. It's the undisputed best VAD for edge deployment. 2 MB model, sub-millisecond latency, 97%+ accuracy. Nothing else to consider.

### 4.3 Speaker ID: ECAPA-TDNN (SpeechBrain)

| Property | Value |
|---|---|
| **Model** | ECAPA-TDNN |
| **Source** | SpeechBrain (`speechbrain.inference.SpeakerRecognition`) |
| **VRAM** | 0 (CPU only) |
| **RAM** | ~50 MB |
| **CPU** | ~50 ms per 3-second segment |
| **Embedding** | 192-dimensional speaker fingerprint |
| **Accuracy** | EER ~0.69% on VoxCeleb1 (99.3% speaker identification) |
| **Enrollment** | Record 15-30s of voice once → averaged embedding = voiceprint |
| **Verification** | Extract embedding → cosine similarity vs voiceprint → threshold 0.70 |
| **License** | Apache 2.0 |
| **Disk** | ~25 MB |

**Why ECAPA-TDNN over alternatives:**

| Model | Size | Accuracy (EER) | Speed | Edge-Ready | Verdict |
|---|---|---|---|---|---|
| **ECAPA-TDNN (SpeechBrain)** | 25 MB | 0.69% | 50ms CPU | ✅ | **BEST PICK** |
| Resemblyzer (GE2E) | 5 MB | ~3-5% | 30ms CPU | ✅ | Lighter but way less accurate |
| TitaNet-Large (NeMo) | 90 MB | 0.66% | 80ms CPU | ⚠️ | Marginally better, 4x larger |
| pyannote embedding | 17 MB | ~1.5% | 60ms CPU | ✅ | Good but ECAPA is better |

**Decision:** ECAPA-TDNN handles both speaker **verification** ("Is this me?") and speaker **diarization** ("Which cluster is this?") using the same 192-dim embeddings + online cosine clustering. No extra model needed for diarization.

### 4.4 TTS: Piper TTS (ONNX, Local)

| Property | Value |
|---|---|
| **Model** | Piper TTS |
| **Source** | `piper-tts` Python package (ONNX Runtime backend) |
| **VRAM** | 0 (CPU + ONNX Runtime, no GPU needed) |
| **RAM** | ~50-100 MB (depends on voice model) |
| **Speed** | Real-time (1s of audio generated in <0.5s on CPU) |
| **Quality** | Near-natural, VITS-based neural TTS |
| **Voices** | 100+ voices, multiple languages, male/female |
| **License** | MIT |
| **Disk** | ~30-60 MB per voice model |
| **Streaming** | ✅ Can generate in chunks for low-latency playback |

**Why Piper TTS over alternatives:**

| Model | VRAM | Quality | Speed | Local | License | Verdict |
|---|---|---|---|---|---|---|
| **Piper TTS** | 0 | ★★★★☆ | Real-time CPU | ✅ | MIT | **BEST PICK** |
| Coqui XTTS v2 | ~2 GB | ★★★★★ | 2-3x slower | ✅ | MPL-2.0 | Too much VRAM |
| Bark (Suno) | ~4 GB | ★★★★★ | Very slow | ✅ | MIT | Way too much VRAM |
| edge-tts | 0 | ★★★★☆ | Fast | ❌ Cloud | Free | Not local |
| espeak-ng | 0 | ★★☆☆☆ | Instant | ✅ | GPL-3.0 | Robotic quality |
| Kokoro TTS | ~500 MB | ★★★★★ | Fast GPU | ✅ | Apache 2.0 | Needs GPU VRAM |
| gTTS | 0 | ★★★☆☆ | OK | ❌ Cloud | MIT | Google API needed |

**Decision:** Piper TTS. It's the only option that delivers near-natural quality, runs entirely on CPU with ONNX Runtime (zero VRAM), is MIT-licensed, and supports real-time streaming. Our i9-14900K can synthesize faster-than-realtime with ease.

**Recommended voice:** `en_US-lessac-medium` (high quality, natural American English, ~40 MB)

### 4.5 Optional: Wake Word — OpenWakeWord

| Property | Value |
|---|---|
| **Model** | OpenWakeWord |
| **Wake Phrase** | Custom "Hey Cortex" |
| **VRAM** | 0 (CPU only) |
| **RAM** | ~5 MB |
| **CPU** | <1 ms per frame |
| **License** | Apache 2.0 |
| **Disk** | ~2 MB per wake word model |

**Why:** Enables hands-free mode switching between passive ambient listening and active query mode. Low-priority — implement after core STT/TTS works.

### 4.6 Final Stack Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CORTEX LAB VOICE STACK — ALL LOCAL, ZERO CLOUD                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Audio Capture       sounddevice (PyAudio alternative)     CPU, <1 MB   │
│  Ring Buffer         numpy circular buffer (60s)           CPU, ~2 MB   │
│  VAD                 Silero VAD v5                          CPU, 10 MB   │
│  Speaker ID          ECAPA-TDNN (SpeechBrain)              CPU, 50 MB   │
│  STT                 faster-whisper (small, CTranslate2)   GPU 500 MB   │
│  TTS                 Piper TTS (ONNX)                      CPU, 50 MB   │
│  Wake Word (opt.)    OpenWakeWord                          CPU, 5 MB    │
│                                                                         │
│  ALL storage is LOCAL:                                                  │
│  • Voiceprints → JSON file on disk                                      │
│  • Transcripts → existing DuckDB (CausalMemoryObject, source="voice")  │
│  • Embeddings  → existing FAISS vector store                            │
│  • Entities    → existing NetworkX knowledge graph                      │
│  • Audio raw   → optional WAV archival on disk                          │
│  • Config      → JSON file on disk                                      │
│                                                                         │
│  ZERO CLOUD DEPENDENCIES. ZERO API KEYS. 100% OFFLINE.                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. TIER 0: Microphone + Ring Buffer

### 5.1 What It Does

Continuously captures system microphone audio at 16 kHz mono (16-bit PCM) into a 60-second circular ring buffer. This is the always-on foundation — it costs essentially nothing (~0.1% CPU, ~2 MB RAM) but ensures we never miss the beginning of a conversation.

### 5.2 Why We Need It

Without a ring buffer, we'd miss the first few seconds of speech while the VAD wakes up. The buffer ensures we can look back 60 seconds and capture the complete start of any conversation.

### 5.3 Implementation

**File:** `backend/src/ambient/audio_capture.py`

```python
"""
Tier 0: Continuous Audio Capture with Ring Buffer
Always-on 16kHz mono PCM capture into 60-second circular buffer.
Cost: ~0.1% CPU, ~2 MB RAM.
"""

import numpy as np
import sounddevice as sd
import threading
from collections import deque

class AudioCapture:
    SAMPLE_RATE = 16000       # 16 kHz
    CHANNELS = 1              # Mono
    DTYPE = np.int16          # 16-bit PCM
    FRAME_MS = 30             # 30ms frames (Silero VAD expects this)
    BUFFER_SECONDS = 60       # 60-second ring buffer
    
    def __init__(self, device=None):
        self.device = device   # None = system default mic
        self.frame_size = int(self.SAMPLE_RATE * self.FRAME_MS / 1000)  # 480 samples
        self.ring_buffer = deque(maxlen=int(self.BUFFER_SECONDS * 1000 / self.FRAME_MS))
        self._stream = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_callback = None  # Set by VAD to receive frames
        
    def start(self):
        """Start audio capture in background thread."""
        ...
    
    def stop(self):
        """Stop audio capture."""
        ...
    
    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Get last N seconds from ring buffer as contiguous array."""
        ...
    
    def set_frame_callback(self, callback):
        """Register callback for each new 30ms frame (used by VAD)."""
        ...
```

### 5.4 Integration

- VAD (Tier 1) registers a `frame_callback` to receive each 30 ms frame
- Ring buffer allows looking back to capture speech onset
- `AudioCapture` runs in its own daemon thread, managed by `AmbientService`

---

## 6. TIER 1: Voice Activity Detection

### 6.1 What It Does

Silero VAD processes each 30 ms audio frame and outputs a speech probability (0.0-1.0). When probability exceeds threshold (default 0.5), it marks `speech_start`. When it drops below threshold for >300 ms, it marks `speech_end`. This filters out ~95% of audio (silence, background noise, ambient sounds) and only passes actual speech segments forward.

### 6.2 Why We Need It

Without VAD, we'd be running Whisper transcription 24/7 on silence — wasting 100% of GPU time on nothing. VAD ensures Whisper only processes actual speech, reducing GPU usage from 100% to ~5% of the day.

### 6.3 Implementation

**File:** `backend/src/ambient/vad.py`

```python
"""
Tier 1: Voice Activity Detection — Silero VAD v5
Processes 30ms frames, emits speech segments.
Cost: <1ms per frame, CPU only, 2MB model.
"""

import torch
import numpy as np
from typing import Callable, Optional

class VoiceActivityDetector:
    THRESHOLD = 0.5           # Speech probability threshold
    MIN_SPEECH_MS = 250       # Minimum speech duration to emit
    MIN_SILENCE_MS = 300      # Silence duration to end segment
    SPEECH_PAD_MS = 100       # Padding before/after speech
    
    def __init__(self):
        self.model, self.utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', onnx=True
        )
        self._speech_active = False
        self._speech_frames = []
        self._silence_count = 0
        self._on_speech_segment = None  # Callback: (np.ndarray, float, float) -> None
    
    def process_frame(self, frame: np.ndarray, timestamp: float):
        """Process a single 30ms frame. Emits speech segments via callback."""
        ...
    
    def set_speech_callback(self, callback):
        """Register callback for complete speech segments."""
        ...
```

### 6.4 Integration

- Receives 30 ms frames from `AudioCapture.frame_callback`
- Emits complete speech segments (np.ndarray + start/end timestamps) to Speaker ID (Tier 2)
- 95% of frames are discarded (silence) — only ~5% produce speech segments

---

## 7. TIER 2: Speaker Identification

### 7.1 What It Does

For each speech segment from VAD, ECAPA-TDNN extracts a 192-dimensional speaker embedding. This embedding is compared to the user's enrolled voiceprint via cosine similarity:
- **Score ≥ 0.70** → label as `"USER"` (you)
- **Score < 0.70** → cluster into `"SPEAKER_A"`, `"SPEAKER_B"`, etc.

### 7.2 Why We Need It

When Cortex Lab captures a meeting or dinner conversation, it needs to know **who said what**. The RAG system stores memories with speaker attribution. Later, when you ask "What did my colleague say about the project deadline?", the system can retrieve only OTHER speakers' turns. Your own speech gets labeled as first-person episodic memories.

### 7.3 Implementation

**File:** `backend/src/ambient/speaker_id.py`

```python
"""
Tier 2: Speaker Identification — ECAPA-TDNN (SpeechBrain)
Verifies user voice, clusters other speakers.
Cost: ~50ms per segment, CPU only, 25MB model.
"""

import numpy as np
from pathlib import Path

class SpeakerIdentifier:
    SIMILARITY_THRESHOLD = 0.70
    VOICEPRINT_PATH = "data/voiceprints/"  # Local storage
    
    def __init__(self):
        from speechbrain.inference import SpeakerRecognition
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        self.user_voiceprint = None  # Loaded from disk on init
        self._speaker_clusters = {}  # session speaker centroids
        self._load_voiceprint()
    
    def enroll_user(self, audio_samples: list[np.ndarray]) -> bool:
        """Enroll user voiceprint from 15-30s of speech samples."""
        ...
    
    def identify(self, audio_segment: np.ndarray) -> tuple[str, float]:
        """Identify speaker. Returns (label, confidence)."""
        ...
    
    def _cluster_speaker(self, embedding: np.ndarray) -> str:
        """Assign non-user speech to a speaker cluster."""
        ...
```

**File:** `backend/src/ambient/enrollment.py`

```python
"""
Voice Enrollment — Record and save user voiceprint.
One-time setup: user records 15-30s of voice.
Voiceprint saved locally to data/voiceprints/user.npy
"""

class VoiceEnrollment:
    def __init__(self, audio_capture, speaker_id):
        self.capture = audio_capture
        self.speaker = speaker_id
    
    async def start_enrollment(self, duration_seconds: int = 20) -> dict:
        """Record enrollment audio and create voiceprint."""
        ...
    
    def is_enrolled(self) -> bool:
        """Check if user has enrolled their voice."""
        ...
```

### 7.4 Voiceprint Storage

Voiceprints are stored **locally** as numpy arrays:
```
backend/data/voiceprints/
├── user.npy              # 192-dim averaged embedding
├── user_samples.npy      # Raw enrollment samples (for re-enrollment)
└── speaker_aliases.json  # {"SPEAKER_A": "Sarah (colleague)", ...}
```

This aligns with our 100% local storage principle. No cloud, no API.

---

## 8. TIER 3: Transcription (STT)

### 8.1 What It Does

`faster-whisper` (CTranslate2 backend) transcribes speaker-labeled speech segments into text with word-level timestamps. Runs on GPU (500 MB VRAM) when available, falls back to CPU.

### 8.2 Why We Need It

This is the actual Speech-to-Text conversion. Without it, we have audio segments but no text to feed into the Agentic RAG pipeline. Whisper is the bridge between spoken words and the entire memory/retrieval/reasoning system.

### 8.3 Implementation

**File:** `backend/src/ambient/transcription.py`

```python
"""
Tier 3: Speech Transcription — faster-whisper (CTranslate2)
Batch transcribe speaker-labeled segments.
Cost: ~500MB VRAM (GPU) or CPU-only fallback.
"""

from faster_whisper import WhisperModel
import numpy as np
from typing import Optional
import asyncio

class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "auto"):
        """
        Args:
            model_size: "tiny", "base", "small", "medium" 
            device: "auto" (GPU if available), "cuda", or "cpu"
        """
        self.device = self._resolve_device(device)
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            model_size, device=self.device, compute_type=compute_type
        )
    
    async def transcribe(self, audio: np.ndarray, 
                          language: Optional[str] = None) -> dict:
        """
        Transcribe audio segment.
        Returns: {"text": str, "language": str, "segments": [...], "duration": float}
        """
        # Run in thread to avoid blocking asyncio event loop
        ...
    
    def _resolve_device(self, device: str) -> str:
        """Determine best device, respecting VRAM coordination."""
        ...
```

### 8.4 VRAM Coordination with LLM

**Critical:** faster-whisper (500 MB VRAM) and the fine-tuned 7B LLM (~14.3 GB VRAM) must not compete for GPU memory. See Section 11 for the mutual exclusion lock design.

---

## 9. TIER 4: Text-to-Speech (TTS)

### 9.1 What It Does

Piper TTS converts the LLM's text response into natural-sounding speech. Runs entirely on CPU via ONNX Runtime — zero VRAM required. Supports streaming (generate audio chunks as text is generated) for low-latency playback.

### 9.2 Why We Need It

TTS completes the voice interaction loop. Without it, users must read the screen even when they asked a question by voice. With TTS, Cortex Lab becomes a true conversational partner — you speak, it speaks back.

### 9.3 Implementation

**File:** `backend/src/ambient/tts.py`

```python
"""
Text-to-Speech — Piper TTS (ONNX, CPU-only)
Neural TTS with near-natural quality, real-time on CPU.
Cost: 0 VRAM, ~50MB RAM, faster-than-realtime on i9-14900K.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Generator
import wave
import io

class TextToSpeech:
    VOICES_DIR = "data/tts_voices/"
    DEFAULT_VOICE = "en_US-lessac-medium"
    SAMPLE_RATE = 22050  # Piper default output rate
    
    def __init__(self, voice: str = None):
        self.voice = voice or self.DEFAULT_VOICE
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load Piper ONNX voice model."""
        import piper
        model_path = Path(self.VOICES_DIR) / f"{self.voice}.onnx"
        config_path = Path(self.VOICES_DIR) / f"{self.voice}.onnx.json"
        self._model = piper.PiperVoice.load(str(model_path), str(config_path))
    
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize full text to audio array."""
        ...
    
    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """Stream audio chunks as they're generated (for low-latency playback)."""
        ...
    
    def synthesize_to_wav(self, text: str) -> bytes:
        """Synthesize to WAV bytes (for API response)."""
        ...
```

### 9.4 Frontend Playback

The frontend receives audio via:
1. **REST endpoint** → returns WAV bytes, frontend plays via Web Audio API
2. **WebSocket** → streams audio chunks for real-time playback during generation

---

## 10. Conversation Segmenter + Auto-Ingestion Bridge

### 10.1 What It Does

Groups speaker-labeled transcript turns into `ConversationRecord` objects. Detects conversation boundaries (>2 min silence = end of conversation). Automatically feeds completed conversations into the existing `MemoryIngestionPipeline`.

### 10.2 Why We Need It

Raw transcription produces individual utterances. But memories are about **conversations** — coherent exchanges with context, participants, and duration. The segmenter transforms a stream of "Speaker A said X at time T" into structured conversation records that the RAG system can reason over.

### 10.3 Implementation

**File:** `backend/src/ambient/conversation.py`

```python
"""
Conversation Segmenter + Auto-Ingestion Bridge
Groups turns into conversations, feeds into existing RAG pipeline.
"""

import time
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class ConversationTurn:
    speaker_label: str       # "USER", "SPEAKER_A", etc.
    text: str
    timestamp: float         # Unix timestamp
    confidence: float = 0.0  # STT confidence

@dataclass
class ConversationRecord:
    id: str
    turns: List[ConversationTurn]
    participants: List[str]
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    auto_ingested: bool = False

class ConversationSegmenter:
    SILENCE_THRESHOLD_S = 120  # 2 min silence = conversation end
    MIN_TURNS = 2              # Minimum turns to count as conversation
    
    def __init__(self, ingestion_pipeline):
        self.pipeline = ingestion_pipeline
        self._current_turns: List[ConversationTurn] = []
        self._last_turn_time: float = 0
        self._conversations: List[ConversationRecord] = []
    
    async def add_turn(self, speaker: str, text: str, timestamp: float):
        """Add a transcribed turn. Auto-segments and ingests."""
        ...
    
    async def _finalize_conversation(self):
        """End current conversation, format, and ingest into RAG."""
        ...
    
    async def _ingest_conversation(self, record: ConversationRecord):
        """
        Bridge to existing MemoryIngestionPipeline.
        Formats conversation into memory content and calls:
        
        await self.pipeline.ingest(
            content=formatted_transcript,
            source="voice",
            session_context=conversation_context
        )
        """
        ...
```

### 10.4 The Critical Bridge

The ingestion call is:
```python
await self.pipeline.ingest(
    content="[Conversation with Sarah, 14:30-14:45]\n"
            "USER: I think we should pivot the project...\n"
            "SARAH: That makes sense, what about the timeline?\n"
            "USER: Good point, let me think about it...",
    source="voice",
    session_context="Voice conversation captured via ambient listening"
)
```

This single call triggers the **entire existing 11-stage pipeline**: classification → emotion → entities ("Sarah", "project") → topics ("work", "decisions") → importance → propositions → contextual prefix → BGE-large embedding → FAISS + DuckDB + Knowledge Graph storage → belief evolution detection.

**No new processing pipeline needed.** Voice just feeds into the existing one.

---

## 11. VRAM Coordination

### 11.1 The Problem

The fine-tuned 7B LLM uses ~14.3 GB VRAM. faster-whisper needs ~500 MB VRAM. Together they fit (14.8 GB < 20 GB), but they should not run GPU-intensive operations simultaneously to avoid CUDA out-of-memory spikes during peak allocation.

### 11.2 The Solution: asyncio Lock

**File:** `backend/src/ambient/vram_guard.py`

```python
"""
VRAM Coordination — Mutual Exclusion between LLM and Whisper GPU usage.
Uses asyncio.Lock to ensure only one GPU-intensive task runs at a time.
"""

import asyncio

class VRAMGuard:
    """
    Ensures LLM inference and Whisper transcription don't overlap on GPU.
    
    Usage:
        async with vram_guard.acquire("whisper"):
            result = await transcriber.transcribe(audio)
        
        async with vram_guard.acquire("llm"):
            result = await llm.generate(prompt)
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_holder = None
    
    def acquire(self, holder: str):
        """Context manager for GPU-exclusive access."""
        ...
```

### 11.3 How It Works in Practice

**During ambient listening (no user query):**
- Whisper acquires lock → transcribes speech segment → releases lock
- LLM is idle, no conflict

**During active chat (user asks a question):**
- LLM acquires lock → generates response → releases lock  
- If speech arrives during LLM generation, Whisper waits for lock release
- Speech is buffered (ring buffer holds 60s), nothing is lost
- After LLM finishes, Whisper catches up on buffered speech

**During voice query (speak → RAG → speak back):**
1. Whisper acquires lock → transcribes user query → releases lock
2. LLM acquires lock → RAG pipeline → generates answer → releases lock
3. Piper TTS speaks answer (CPU only, no lock needed)
4. Whisper resumes ambient listening

**Worst case latency impact:** Whisper waits 2-5 seconds during LLM generation. Acceptable — the ring buffer ensures no audio is lost.

---

## 12. Backend API Endpoints

### 12.1 New Endpoints to Add to `backend/server.py`

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/ambient/start` | Start ambient listening service |
| `POST` | `/api/ambient/stop` | Stop ambient listening service |
| `GET` | `/api/ambient/status` | Get current listening state + stats |
| `POST` | `/api/ambient/enroll` | Start voice enrollment (15-30s recording) |
| `GET` | `/api/ambient/enrollment-status` | Check if user is enrolled |
| `GET` | `/api/ambient/conversations` | List captured conversations |
| `GET` | `/api/ambient/conversations/{id}` | Get specific conversation details |
| `POST` | `/api/ambient/config` | Update ambient config (VAD threshold, auto-ingest, etc.) |
| `GET` | `/api/ambient/config` | Get current ambient config |
| `POST` | `/api/tts/synthesize` | Synthesize text to speech (returns WAV audio) |
| `POST` | `/api/voice/query` | Voice query: upload audio → STT → RAG → return text + TTS audio |
| `WebSocket` | `/ws/ambient` | Live transcript streaming + audio level + VAD activity |

### 12.2 Integration with Existing Endpoints

No changes needed to existing endpoints. Voice features are additive:
- `/api/health` — add `ambient_status` field to response
- `/api/rag/stats` — add `voice_memories_count` to stats
- `/api/memories` — voice memories appear alongside typed ones (distinguishable by `source: "voice"`)

---

## 13. Frontend Components

### 13.1 New Components

| Component | Purpose |
|---|---|
| `AmbientPanel.tsx` | Main ambient listening dashboard — start/stop, live waveform, conversation list |
| `VoiceEnrollment.tsx` | Voice enrollment UI — record button, progress bar, confirmation |
| `LiveTranscript.tsx` | Real-time transcript display via WebSocket — speaker labels, timestamps |
| `ConversationHistory.tsx` | Browse past ambient conversations — expandable, searchable |
| `VoiceQueryButton.tsx` | Mic button in ChatPanel — hold to record voice query |
| `TTSPlayback.tsx` | Audio playback controls for TTS responses — play/pause/speed |

### 13.2 Sidebar Integration

Add a new navigation tab to the existing Sidebar:

```tsx
// In Sidebar.tsx, add to navItems array:
{ view: "ambient", icon: Mic, label: "Ambient Listening" }
```

### 13.3 ChatPanel Integration

Add voice capabilities to the existing chat:
- **Mic button** next to the text input — tap to speak a query
- **Speaker icon** on assistant messages — tap to hear TTS playback
- **"Voice" badge** on messages that originated from voice input

---

## 14. Data Models

### 14.1 New Data Models

Add to `backend/src/models/__init__.py`:

```python
# ─── Voice/Ambient Models ────────────────────────────────────────────────

class AmbientStatus(str, Enum):
    IDLE = "idle"                       # Service not running
    LISTENING = "listening"             # Mic active, waiting for speech
    SPEECH_DETECTED = "speech_detected" # VAD triggered, capturing speech
    TRANSCRIBING = "transcribing"       # Whisper processing speech segment
    PAUSED = "paused"                   # Manually paused by user

@dataclass
class ConversationTurn:
    speaker_label: str          # "USER", "SPEAKER_A", etc.
    speaker_name: str           # Resolved name: "Suraj", "Sarah", etc.
    text: str
    timestamp: datetime
    confidence: float = 0.0     # STT confidence

@dataclass 
class ConversationRecord:
    id: str
    turns: List[ConversationTurn]
    participants: List[str]
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_ids: List[str] = field(default_factory=list)  # Linked ingested memories
    auto_ingested: bool = False

@dataclass
class SpeakerProfile:
    id: str
    name: str                           # Display name
    voiceprint_path: str                # Path to .npy file
    is_user: bool = False
    first_heard: datetime = field(default_factory=datetime.now)
    conversation_count: int = 0

@dataclass
class AmbientConfig:
    enabled: bool = False
    vad_threshold: float = 0.5
    auto_ingest: bool = True            # Auto-feed to RAG pipeline
    silence_timeout_s: int = 120        # Conversation end detection
    min_speech_ms: int = 250            # Minimum speech to process
    tts_enabled: bool = True
    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0
    pause_schedule: List[Dict] = field(default_factory=list)  # Time-based pauses
    record_raw_audio: bool = False      # Archive WAV files (disk-heavy)
```

### 14.2 Existing Model Update

The existing `CausalMemoryObject.source` field already supports `"voice"`. No schema change needed. Voice conversation metadata can go into the existing `metadata: Dict[str, Any]` field:

```python
memory.metadata = {
    "conversation_id": "conv_abc123",
    "speaker": "USER",
    "turn_index": 3,
    "participants": ["USER", "SPEAKER_A"],
    "stt_confidence": 0.92,
    "duration_seconds": 45.2,
}
```

---

## 15. File Structure

### 15.1 New Files to Create

```
backend/src/ambient/                    # NEW PACKAGE
├── __init__.py                         # AmbientService (orchestrates all tiers)
├── audio_capture.py                    # Tier 0: Microphone + Ring Buffer
├── vad.py                              # Tier 1: Silero VAD v5
├── speaker_id.py                       # Tier 2: ECAPA-TDNN speaker verification
├── enrollment.py                       # Voice enrollment flow
├── transcription.py                    # Tier 3: faster-whisper STT
├── tts.py                              # Tier 4: Piper TTS
├── conversation.py                     # Conversation segmenter + auto-ingestion
├── vram_guard.py                       # VRAM mutual exclusion lock
└── config.py                           # AmbientConfig load/save

backend/data/voiceprints/               # NEW DIRECTORY (local storage)
├── user.npy                            # User voiceprint (192-dim)
└── speaker_aliases.json                # Named speaker mappings

backend/data/tts_voices/                # NEW DIRECTORY (local storage)
├── en_US-lessac-medium.onnx            # Piper voice model
└── en_US-lessac-medium.onnx.json       # Voice config

frontend/src/components/                # NEW COMPONENTS
├── AmbientPanel.tsx                    # Ambient listening dashboard
├── VoiceEnrollment.tsx                 # Voice enrollment UI
├── LiveTranscript.tsx                  # Real-time transcript (WebSocket)
├── ConversationHistory.tsx             # Past conversations browser
├── VoiceQueryButton.tsx                # Mic button for ChatPanel
└── TTSPlayback.tsx                     # TTS audio playback controls
```

### 15.2 Existing Files to Modify

| File | Change |
|---|---|
| `backend/server.py` | Add ambient + TTS API endpoints |
| `backend/src/engine.py` | Add `ambient_service` to RAG engine init |
| `backend/src/models/__init__.py` | Add voice data models |
| `backend/requirements.txt` | Add new dependencies |
| `frontend/src/components/Sidebar.tsx` | Add "Ambient" nav tab |
| `frontend/src/components/ChatPanel.tsx` | Add mic button + TTS playback |
| `frontend/src/lib/types.ts` | Add ambient TypeScript types |
| `frontend/src/lib/api.ts` | Add ambient + TTS API functions |
| `frontend/src/app/page.tsx` | Add AmbientPanel route/view |

---

## 16. Dependencies

### 16.1 New Python Packages

Add to `backend/requirements.txt`:

```
# Voice / Ambient Listening
sounddevice>=0.4.6          # Audio capture (PortAudio wrapper)
faster-whisper>=1.0.0       # STT — CTranslate2 Whisper (MIT)
speechbrain>=1.0.0          # Speaker ID — ECAPA-TDNN (Apache 2.0)
piper-tts>=1.2.0            # TTS — Neural TTS, ONNX (MIT)
onnxruntime>=1.17.0         # ONNX Runtime for Piper TTS
webrtcvad>=2.0.10           # Optional: Google WebRTC VAD backup
```

**Note:** Silero VAD is loaded via `torch.hub.load()` — no separate pip package needed. PyTorch is already installed.

### 16.2 System Dependencies

```bash
# PortAudio (required by sounddevice)
sudo apt-get install libportaudio2 portaudio19-dev

# PipeWire (already installed on this system)
# No action needed — PipeWire 1.0.5 detected
```

### 16.3 Model Downloads (One-Time)

```bash
# Silero VAD — auto-downloads via torch.hub (~2 MB)
# ECAPA-TDNN — auto-downloads via SpeechBrain (~25 MB)
# faster-whisper small — auto-downloads via CTranslate2 (~500 MB)
# Piper voice — manual download (~40 MB per voice)
wget -O data/tts_voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O data/tts_voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## 17. Resource Budget Summary

### 17.1 Idle State (Ambient Listening, No Speech)

| Component | VRAM | RAM | CPU | Disk |
|---|---|---|---|---|
| DeepSeek-R1-7B (4-bit) | 14,300 MB | — | ~0% | — |
| BGE-large + Reranker | 0 | ~800 MB | ~0% | — |
| Silero VAD | 0 | 10 MB | <0.1% | 2 MB |
| Ring buffer + audio | 0 | 2 MB | <0.1% | 0 |
| **Total (idle)** | **14,300 MB** | **~812 MB** | **<0.5%** | **2 MB** |
| **Free VRAM** | **6,175 MB** | | | |

### 17.2 Active Speech (Conversation Detected)

| Component | VRAM | RAM | CPU | Disk |
|---|---|---|---|---|
| Everything from idle | 14,300 MB | 812 MB | <0.5% | — |
| ECAPA-TDNN Speaker ID | 0 | 50 MB | ~1% | 25 MB |
| faster-whisper (small) | 500 MB | 200 MB | ~5% | 500 MB |
| Conversation segmenter | 0 | 10 MB | <0.1% | 0 |
| **Total (active)** | **14,800 MB** | **~1,072 MB** | **~6%** | **527 MB** |
| **Free VRAM** | **5,675 MB** | | | |

### 17.3 Voice Query (RAG + TTS Response)

| Component | VRAM | RAM | CPU | Disk |
|---|---|---|---|---|
| Everything from active | 14,800 MB | 1,072 MB | ~6% | — |
| LLM generation (peak) | +0 (already loaded) | — | ~80% GPU | — |
| Piper TTS | 0 | 50 MB | ~10% | 40 MB |
| **Total (query)** | **14,800 MB** | **~1,122 MB** | ~80% GPU + 10% CPU | **567 MB** |
| **Free VRAM** | **5,675 MB** | | | |

### 17.4 Verdict

✅ **Everything fits comfortably within our 20 GB VRAM + 32 GB RAM + 32 CPU threads.**

The worst case (Whisper + LLM both loaded) uses 14.8 GB of 20 GB VRAM = **74% utilization**. The VRAM guard ensures they don't run GPU-intensive kernels simultaneously, preventing CUDA OOM during peak allocation.

---

## 18. Privacy & Ethics

### 18.1 Core Principle: 100% Local

All processing happens on your machine. Zero audio or text is sent to any cloud:
- ✅ Silero VAD — local ONNX model
- ✅ ECAPA-TDNN — local SpeechBrain model
- ✅ faster-whisper — local CTranslate2 model
- ✅ Piper TTS — local ONNX model
- ✅ All storage — local DuckDB, FAISS, NetworkX, JSON files
- ✅ LLM — local fine-tuned DeepSeek-R1-7B

### 18.2 Recording Other People

Since ambient listening records **other people's conversations**:

| Concern | Mitigation Built Into System |
|---|---|
| **Legal (wiretapping)** | Recording laws vary by jurisdiction. System includes prominently accessible pause button and configurable schedules. Users are responsible for local compliance. |
| **Ethical** | Ambient mode is **opt-in** (not on by default). Start/stop clearly visible in UI. |
| **Other speakers' data** | Option to store only YOUR turns, summarize/anonymize others |
| **Sensitive spaces** | Configurable pause zones/schedules (bathroom, bedroom, night hours) |
| **Data deletion** | Full conversation deletion via API + UI |
| **Transparency** | Visual indicator in UI when ambient listening is active |

### 18.3 Default Configuration

Ambient listening is **OFF by default**. User must:
1. Explicitly enable it in Settings
2. Complete voice enrollment first
3. Acknowledge the privacy notice

---

## 19. Implementation Order

### 19.1 Recommended Build Sequence

Build in this order to have testable milestones at each step:

```
PHASE 1: Core STT (Days 1-3)
─────────────────────────────
  1. Install dependencies (sounddevice, faster-whisper, etc.)
  2. AudioCapture (Tier 0) — test mic input
  3. Silero VAD (Tier 1) — test speech detection
  4. faster-whisper Transcription (Tier 3) — test STT
  5. VRAMGuard — test mutual exclusion
  ★ MILESTONE: Speak into mic → see transcript in terminal

PHASE 2: Speaker ID + Conversations (Days 4-5)
─────────────────────────────────────────────────
  6. ECAPA-TDNN SpeakerIdentifier (Tier 2) — test speaker labeling
  7. VoiceEnrollment — test enrollment flow
  8. ConversationSegmenter — test grouping turns
  9. Auto-ingestion bridge to MemoryIngestionPipeline
  ★ MILESTONE: Conversations auto-ingested as voice memories

PHASE 3: TTS (Day 6)
─────────────────────
  10. Piper TTS — test text→speech
  11. Audio streaming endpoint
  ★ MILESTONE: POST text → receive WAV audio

PHASE 4: AmbientService Orchestrator (Day 7)
─────────────────────────────────────────────
  12. AmbientService __init__.py — wire all tiers together
  13. Config load/save
  14. Data models in models/__init__.py
  ★ MILESTONE: Single start/stop controls entire pipeline

PHASE 5: Backend API (Days 8-9)
──────────────────────────────
  15. Ambient API endpoints in server.py
  16. TTS endpoint in server.py
  17. Voice query endpoint
  18. WebSocket for live transcript
  ★ MILESTONE: curl can start/stop ambient, get conversations

PHASE 6: Frontend UI (Days 10-12)
─────────────────────────────────
  19. AmbientPanel.tsx
  20. VoiceEnrollment.tsx
  21. LiveTranscript.tsx
  22. ConversationHistory.tsx
  23. VoiceQueryButton.tsx in ChatPanel
  24. TTSPlayback.tsx for assistant messages
  25. Sidebar "Ambient" tab
  ★ MILESTONE: Full UI for ambient + voice chat + TTS

PHASE 7: Polish + Wake Word (Days 13-14)
─────────────────────────────────────────
  26. OpenWakeWord "Hey Cortex" (optional)
  27. Pause schedule UI
  28. Speaker alias naming UI
  29. Testing, edge cases, error handling
  ★ MILESTONE: Production-ready voice system
```

### 19.2 Dependencies Between Phases

```
Phase 1 (STT) ← standalone, no dependencies
Phase 2 (Speaker ID) ← depends on Phase 1 (needs audio segments)
Phase 3 (TTS) ← standalone, no dependencies  
Phase 4 (Orchestrator) ← depends on Phases 1 + 2 + 3
Phase 5 (Backend API) ← depends on Phase 4
Phase 6 (Frontend) ← depends on Phase 5
Phase 7 (Polish) ← depends on all above
```

**Phases 1 and 3 can be built in parallel** since they have no dependencies on each other.

---

## Summary

This document describes a **7-phase implementation plan for 52 new components** that add full voice capabilities to Cortex Lab:

- **STT** via faster-whisper (small) — 500 MB VRAM, MIT license
- **TTS** via Piper TTS (ONNX) — 0 VRAM (CPU only), MIT license
- **VAD** via Silero VAD v5 — 0 VRAM, MIT license
- **Speaker ID** via ECAPA-TDNN — 0 VRAM, Apache 2.0 license
- **Wake Word** via OpenWakeWord — 0 VRAM, Apache 2.0 (optional)

**All models run locally.** Zero cloud. Zero API keys. Everything stores in the existing local DuckDB + FAISS + Knowledge Graph via the existing `MemoryIngestionPipeline.ingest(source="voice")` method.

**Total additional resource cost:**
- VRAM: +500 MB (during speech only) → 74% utilization, well within 20 GB budget
- RAM: +300 MB → well within 32 GB budget
- CPU: +6% during speech, <0.5% at idle → i9-14900K handles this effortlessly
- Disk: ~570 MB for models

The voice system transforms Cortex Lab from a **typed chat interface** into a **true cognitive operating system** that captures your entire verbal life — meetings, calls, thoughts, reflections — and makes them searchable, retrievable, and reasoning-capable through the same 9-Layer Agentic RAG architecture.

---

*Document created: February 26, 2026*
*Aligned with: Vision-Plan.md, RAG-Architecture.md*
*Hardware: NVIDIA RTX 4000 Ada (20GB) + Intel i9-14900K + 32GB RAM*
*All storage: 100% local (DuckDB, FAISS, NetworkX, JSON, WAV)*
