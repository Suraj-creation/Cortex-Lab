# Cortex Lab — Complete Architecture Deep Dive

## A Comprehensive Guide to How This Agentic RAG System Actually Works

> **Written for clarity.** Every layer, every decision, every code path — explained as if teaching someone who has never seen this codebase before.
>
> **Based on:** Full audit of every Python and TypeScript source file in the project (March 2026).
>
> **Critical distinction:** This document explains **what is actually implemented in code**, not what is described in RAG-Architecture.md. Where the design document and the code diverge, this document says so explicitly.

---

## Table of Contents

1. [What Is Cortex Lab?](#1-what-is-cortex-lab)
2. [The Big Picture — How a Question Becomes an Answer](#2-the-big-picture--how-a-question-becomes-an-answer)
3. [Hardware & Model Stack](#3-hardware--model-stack)
4. [Layer 0 — The Server (FastAPI Gateway)](#4-layer-0--the-server-fastapi-gateway)
5. [Layer 1 — System Prompting & Prompt Construction](#5-layer-1--system-prompting--prompt-construction)
6. [Layer 2 — The RAG Engine (Central Orchestration)](#6-layer-2--the-rag-engine-central-orchestration)
7. [Layer 3 — Query Intelligence (Intent Detection & Transformation)](#7-layer-3--query-intelligence-intent-detection--transformation)
8. [Layer 4 — Agent Orchestrator (The Brain)](#8-layer-4--agent-orchestrator-the-brain)
9. [Layer 5 — Specialized Agents (The Workers)](#9-layer-5--specialized-agents-the-workers)
10. [Layer 6 — Hybrid Retrieval Engine (6-Channel Memory Search)](#10-layer-6--hybrid-retrieval-engine-6-channel-memory-search)
11. [Layer 7 — Quality Assurance (CRAG, Self-RAG, FLARE)](#11-layer-7--quality-assurance-crag-self-rag-flare)
12. [Layer 8 — The LLM Interface (Fine-Tuned Model Methods)](#12-layer-8--the-llm-interface-fine-tuned-model-methods)
13. [Layer 9 — Memory Ingestion Pipeline (How Memories Are Born)](#13-layer-9--memory-ingestion-pipeline-how-memories-are-born)
14. [Layer 10 — Storage Stack (Where Everything Lives)](#14-layer-10--storage-stack-where-everything-lives)
15. [Layer 11 — Streaming & Token Delivery](#15-layer-11--streaming--token-delivery)
16. [Layer 12 — Hallucination Defense System](#16-layer-12--hallucination-defense-system)
17. [Layer 13 — Ambient Voice Pipeline (STT/TTS)](#17-layer-13--ambient-voice-pipeline-stttts)
18. [Layer 14 — PageIndex (Cloud Document Retrieval)](#18-layer-14--pageindex-cloud-document-retrieval)
19. [Layer 15 — Observability & Pipeline Tracing](#19-layer-15--observability--pipeline-tracing)
20. [Layer 16 — Frontend (Next.js UI)](#20-layer-16--frontend-nextjs-ui)
21. [Data Model Reference](#21-data-model-reference)
22. [Complete Request Lifecycle — Step by Step](#22-complete-request-lifecycle--step-by-step)
23. [What's in RAG-Architecture.md but NOT in Code](#23-whats-in-rag-architecturemd-but-not-in-code)
24. [File Map — Every Source File and Its Purpose](#24-file-map--every-source-file-and-its-purpose)

---

## 1. What Is Cortex Lab?

Cortex Lab is a **personal AI memory and reasoning engine**. Think of it as a second brain that:

- **Remembers everything you tell it** — every conversation, fact, opinion, project detail, and life event gets stored as a "memory."
- **Understands what kind of question you're asking** — is this about a timeline? A cause-and-effect chain? A reflection on how your beliefs changed?
- **Searches your memories intelligently** — using 6 different search strategies simultaneously, then combining their results.
- **Reasons about your memories** — using 5 specialized AI agents (Timeline, Causal, Reflection, Planning, Arbitration), each expert at a different type of thinking.
- **Checks its own work** — using techniques called CRAG, Self-RAG, and FLARE to catch mistakes before you see them.
- **Generates human-quality answers** — using a fine-tuned 7-billion-parameter language model (DeepSeek-R1-7B) that was trained in 15 stages specifically for this system.
- **Runs entirely on your machine** — nothing is sent to the cloud (except optional PageIndex document processing). Your data stays local.

### The Core Idea: "Agentic RAG"

**RAG** stands for **R**etrieval-**A**ugmented **G**eneration. It means: before the AI answers your question, it first searches for relevant information (retrieval), then uses that information to craft a grounded answer (generation).

**Agentic** means the system doesn't just do one dumb search-and-answer. Instead, it has multiple intelligent **agents** that can:
- Decide HOW to search (simple lookup vs. multi-step reasoning)
- Choose WHICH agent is best for your question type
- Evaluate WHETHER the search results are good enough
- Retry with different strategies if the first attempt fails

---

## 2. The Big Picture — How a Question Becomes an Answer

Here is the complete journey of a user query through the system, from the moment you press Enter to when you see the response:

```
YOU TYPE: "What projects have I built?"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 0: FastAPI Server (server.py)                        │
│  • Receives HTTP POST to /api/rag/chat                      │
│  • Validates request, checks model is loaded                │
│  • Routes to streaming or non-streaming path                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER "PRE-FILTER": Extraction Bypass (server.py)          │
│  • For simple factual queries (name, email, projects),      │
│    tries to answer directly from evidence using regex.      │
│  • If successful, SKIPS the entire LLM generation.          │
│  • Why? Because the fine-tuned model hallucinates on        │
│    simple factual queries, so regex extraction is           │
│    more reliable.                                           │
└────────────────────────┬────────────────────────────────────┘
                         │ (if extraction fails)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: RAG Engine (engine.py)                            │
│  • Checks cache (exact match → semantic match → miss)       │
│  • Ingests your message as a memory (in background)         │
│  • Calls the Agent Orchestrator                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Query Intelligence (query_engine.py)              │
│  • Detects intent: TEMPORAL, CAUSAL, REFLECTIVE, etc.       │
│  • Scores complexity: 0.0 (trivial) to 1.0 (complex)       │
│  • Generates query variants (multi-query, HyDE, step-back)  │
│                                                             │
│  For "What projects have I built?":                         │
│    Intent: FACTUAL                                          │
│    Complexity: ~0.35 (moderate — needs retrieval)            │
│    Routing: SINGLE_STEP                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: Agent Orchestrator (orchestrator.py)              │
│  • Routes to the appropriate specialized agent              │
│  • For FACTUAL → PlanningAgent                              │
│  • For TEMPORAL → TimelineAgent                             │
│  • For CAUSAL → CausalAgent                                 │
│  • For REFLECTIVE → ReflectionAgent                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: Specialized Agent (specialized.py)                │
│  • PlanningAgent calls the Hybrid Retriever                 │
│  • Gets back ranked, scored evidence from your memories     │
│  • Generates an answer using the LLM with evidence context  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 6: Hybrid Retrieval (hybrid_retriever.py)            │
│  • Runs 6 search channels IN PARALLEL:                      │
│    1. Dense (FAISS vector similarity)                       │
│    2. Sparse (BM25 keyword matching)                        │
│    3. Graph (entity relationship traversal)                 │
│    4. Temporal (time-range filtered search)                 │
│    5. Proposition (atomic fact search)                      │
│    6. PageIndex (cloud document search)                     │
│  • Fuses results using Reciprocal Rank Fusion (RRF)         │
│  • Re-ranks top results using a cross-encoder model         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 7: Quality Assurance (orchestrator.py)               │
│  • CRAG: Evaluates retrieval quality (CORRECT/AMBIGUOUS/    │
│    INCORRECT). If AMBIGUOUS, fetches more evidence.         │
│  • Self-RAG: Critiques the answer using ISREL/ISSUP/ISUSE   │
│    scores. If poor, regenerates. (Only when confidence <0.55)│
│  • FLARE: If confidence is still < 0.4, identifies weak     │
│    sentences and retrieves more evidence for them.          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 11: Streaming (server.py)                            │
│  • Builds a prompt with evidence + system instructions      │
│  • Streams the LLM's response token-by-token via SSE        │
│  • Filters out <think> tags, hallucination patterns,        │
│    and stop patterns in real-time                           │
│  • If hallucination detected mid-stream, replaces           │
│    the response with a regex-extracted answer               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              YOU SEE THE ANSWER
```

**Two paths exist for answering questions:**

| Path | When Used | What Happens |
|------|-----------|--------------|
| **Full Pipeline** (non-streaming) | `stream: false` | Orchestrator does everything: retrieval + agent reasoning + CRAG + Self-RAG + FLARE + LLM generation. Returns complete answer. |
| **Retrieve-then-Stream** (streaming) | `stream: true` (default) | Orchestrator only does retrieval + CRAG (no LLM generation). Then server.py builds its own prompt with evidence and streams the LLM token-by-token. Self-RAG and FLARE are SKIPPED in this path. |

**Important:** The streaming path (which the frontend uses) **skips Self-RAG and FLARE** quality checks. This means the quality assurance layers described in RAG-Architecture.md only activate in non-streaming mode.

---

## 3. Hardware & Model Stack

### What's Running on Your GPU

| Component | Model | Size on Disk | VRAM Usage | Purpose |
|-----------|-------|-------------|------------|---------|
| **Main LLM** | DeepSeek-R1-Distill-Qwen-7B (Fine-Tuned, 15 stages) | ~14GB | ~4.2GB (4-bit quantized) | Generates all text responses, does reasoning, classification, critique |
| **Embedding Model** | BGE-large-en-v1.5 | ~1.3GB | ~1.3GB | Converts text → 1024-dimensional vectors for semantic search |
| **Cross-Encoder Reranker** | BGE-reranker-v2-m3 | ~1.1GB | ~1.1GB | Re-scores search results for better ranking |
| **Total** | — | ~16.4GB | **~6.6GB** | Leaves ~13.4GB of your 20GB GPU free |

### How the LLM Loads

```python
# From server.py — the model loading sequence:
# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 2. Load model in 4-bit quantization (NF4 format)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,     # Compute in bfloat16 for speed
    bnb_4bit_use_double_quant=True,            # Double quantize for memory savings
    bnb_4bit_quant_type="nf4",                 # NormalFloat4 quantization
)

# 3. Memory budget: 17GB GPU + 30GB CPU offload
max_memory = {0: "17GB", "cpu": "30GB"}

# 4. Attention: SDPA (unless Flash Attention 2 is installed)
attn_implementation = "flash_attention_2"  # or "sdpa" as fallback

# 5. Repetition penalty: 1.15 (prevents repetitive output)
```

### Fine-Tuning: 15 Training Stages

The base model was fine-tuned through 15 progressive stages, each teaching a specific skill:

| Stage | Name | What It Teaches |
|-------|------|-----------------|
| 1 | Faithfulness | Answer only from provided evidence, never fabricate |
| 2 | Agentic Routing | Output structured JSON to classify query intent |
| 3 | Causal Reasoning | Trace cause-effect chains across memories |
| 4 | Self-RAG Critique | Score answers on relevance, support, usefulness (ISREL/ISSUP/ISUSE) |
| 5 | Belief Evolution | Detect when the user changed their mind about something |
| 6 | Summarization | Condense long text preserving key facts |
| 7 | Dialogue Coherence | Maintain context across multi-turn conversations |
| 8 | Long Context | Handle 8K+ token contexts accurately |
| 9 | DPO Alignment | Prefer high-quality answers over low-quality ones |
| 10 | User Style | Adapt tone and language to match the user |
| 11 | ORPO Alignment | Combined SFT + preference optimization |
| 12 | RAFT | Identify relevant vs. distractor documents |
| 13 | Function Calling | Parse user intent into tool/function invocations |
| 14 | Rejection Fine-Tuning | Know when to say "I don't know" |
| 15 | SPIN | Self-play iterative improvement |

---

## 4. Layer 0 — The Server (FastAPI Gateway)

**File:** `backend/server.py` (2028 lines)

The server is the entry point for everything. It's a FastAPI application that:

### Startup Sequence (what happens when you run `python server.py`):

1. **Load Tokenizer** — DeepSeek-R1's tokenizer with special tokens (`<think>`, `</think>`, `<|im_start|>`, etc.)
2. **Load Model** — 7B parameters in 4-bit quantization, with CPU offloading enabled
3. **Initialize RAG Engine** — Calls `rag_engine.init(model, tokenizer)` which initializes ALL 11 subsystems (see Layer 2)
4. **Start Uvicorn server** — Listens on port 8000

### API Endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check — is the model loaded? |
| `/api/system/gpu` | GET | GPU memory usage monitoring |
| `/api/chat` | POST | Direct LLM chat (no RAG, no memories) |
| `/api/rag/chat` | POST | **Main endpoint** — RAG-enhanced chat with memory search |
| `/api/memories` | GET | List all stored memories (paginated) |
| `/api/memories/ingest` | POST | Manually add a memory |
| `/api/memories/search` | POST | Semantic search across memories |
| `/api/memories/{id}` | DELETE | Delete a specific memory |
| `/api/graph` | GET | Knowledge graph data for visualization |
| `/api/entities` | GET | All entities in the knowledge graph |
| `/api/beliefs` | GET | Detected belief evolution events |
| `/api/communities` | GET | GraphRAG community clusters |
| `/api/rag/stats` | GET | RAG system statistics |
| `/api/rag/traces` | GET | Pipeline trace history (observability) |
| `/api/documents/upload` | POST | Upload PDF for PageIndex indexing |
| `/api/documents` | GET | List indexed documents |
| `/api/ambient/*` | Various | Voice pipeline: start/stop/config/enroll |
| `/api/tts/synthesize` | POST | Text-to-speech synthesis |
| `/api/voice/query` | POST | Full voice pipeline: audio → STT → RAG → TTS |
| `/ws/ambient` | WebSocket | Live ambient transcript streaming |

### Concurrency & Safety:

```python
# Only 2 concurrent RAG requests allowed (prevents GPU OOM)
_inference_semaphore = asyncio.Semaphore(2)

# Hard timeout of 90 seconds per request
_REQUEST_TIMEOUT = 90.0
```

---

## 5. Layer 1 — System Prompting & Prompt Construction

There are **multiple different system prompts** used in different situations. This is a crucial layer to understand because the system prompt dramatically affects behavior.

### The Main System Prompt (for `/api/chat` — no RAG):

```
You are Cortex Lab, a personal AI memory and reasoning assistant.
You help the user by answering their questions thoughtfully and concisely.
If the user asks about personal information (their name, preferences, etc.)
that you don't actually know, honestly say you don't have that information yet
and suggest they can teach you by telling you.
Never fabricate personal details about the user.
Keep responses focused and do NOT generate follow-up questions or continue
the conversation on behalf of the user.
```

### The RAG Prompt (for `/api/rag/chat` streaming — with evidence):

```
You are Cortex Lab, an intelligent personal AI assistant who knows the user well.
You have access to the user's stored memories below. Use them to answer naturally.

PERSONALITY:
- Speak warmly and conversationally, like a knowledgeable friend
- Give direct, confident answers — never say "Based on your stored memories"
  or "According to evidence"
- For simple questions (name, email, location), answer in ONE short sentence
- For broader questions (skills, projects, background), write a flowing natural
  paragraph — NOT bullet lists
- Always use "you/your" when referring to the user, NEVER "I/my"
- NEVER add citations like [1] [2] — just speak naturally
- NEVER generate "Confidence:", "Evidence:", "Answer:" labels
- NEVER say "belief evolution", "emotion timeline", "key insight",
  "clarity of scope", or similar generic phrases

If the evidence doesn't answer the question, simply say
"I don't have that information yet — feel free to tell me and I'll remember it!"
```

### The Greeting Prompt (for casual messages — no evidence):

```
You are Cortex Lab, a friendly and warm personal AI assistant.
The user is greeting you or making casual conversation.
Respond naturally and briefly. Be cheerful, helpful, and personable.
Keep it to 1-2 sentences.
Do NOT reference memories, evidence, or past conversations.
Do NOT generate philosophical content, analysis, or "key insights".
```

### The Document Prompt (for PageIndex document queries):

```
You are Cortex Lab, an intelligent AI assistant with access to the user's
uploaded documents and memories. The user is asking about content from their
uploaded documents.

RULES:
- Answer ONLY based on the document content provided below
- Be thorough and detailed
- Organize your answer clearly with bullet points or sections
- If the documents don't contain the answer, say so
- NEVER make up information
- NEVER add citations — just speak naturally
```

### The Prompt Template Format:

All prompts use the ChatML template format (`<|im_start|>...<|im_end|>`):

```
<|im_start|>system
{system prompt}
<|im_end|>
<|im_start|>user
{user question}

Here is what I know about you:
[1] {evidence chunk 1}
[2] {evidence chunk 2}
...
<|im_end|>
<|im_start|>assistant
```

### How Evidence Gets Into the Prompt:

1. **PageIndex evidence** gets priority (up to 3 chunks, 2000 chars each)
2. **Local memory evidence** fills remaining slots (up to 5 chunks, 250 chars each)
3. Evidence is filtered:
   - Skip chunks shorter than 50 characters
   - Skip chunks that look like questions (end with `?`, short)
   - Skip chunks with repetitive trigrams (spam detection)
   - Skip stored user queries (prevents "what is my name?" appearing as evidence)

---

## 6. Layer 2 — The RAG Engine (Central Orchestration)

**File:** `backend/src/engine.py` (696 lines)

The `CortexRAGEngine` class is the **central integration point** that wires together all 11 subsystems. It's instantiated as a global singleton:

```python
rag_engine = CortexRAGEngine()  # Line 696 — bottom of engine.py
```

### Initialization Sequence (11 Steps):

When `rag_engine.init(model, tokenizer)` is called during server startup:

```
Step  1/11: EmbeddingModel         — BGE-large-en-v1.5 (1024d vectors, loads onto GPU)
Step  2/11: CrossEncoderReranker   — BGE-reranker-v2-m3 (loads onto GPU)
Step  3/11: VectorStore            — FAISS with Hot/Warm/Cold tiers
Step  4/11: MetadataStore          — DuckDB (6 tables: memories, belief_deltas, entities, graph_edges, conversations, junction tables)
Step  5/11: KnowledgeGraph         — NetworkX directed graph
Step  6/11: LocalLLM               — Interface to the fine-tuned 7B model
Step  7/11: QueryAnalyzer          — Intent detection + complexity scoring
            QueryTransformer       — Multi-query, HyDE, step-back, decomposition
Step  8/11: HybridRetriever        — 6-channel parallel retrieval + RRF fusion + cross-encoder reranking
Step  9/11: AgentOrchestrator      — 5 specialized agents + Adaptive-RAG routing + CRAG + Self-RAG + FLARE
Step 10/11: IngestionPipeline      — 11-stage memory processing
            MultiLevelCache        — 3-level response caching
Step 11/11: PageIndexStore         — Cloud document retrieval (optional, checks config)
```

After initialization, it also:
- Runs **tier migration** (moves old vectors from hot → warm → cold FAISS indexes)
- Migrates **junction tables** (for fast topic/entity lookups)
- Checks **vector coverage** — if < 80% of DuckDB memories have FAISS vectors, automatically re-embeds the missing ones
- Initializes **Ambient Voice Service** (lazy — models load on first use)

### The Two Main Methods:

**`rag_chat()`** — Full pipeline (non-streaming):
1. Check if user message is meaningful → ingest as memory in background
2. Check 3-level cache (exact → semantic → miss)
3. Call `orchestrator.process()` (full pipeline with LLM generation)
4. Cache the result
5. Store conversation turn in DuckDB

**`rag_retrieve()`** — Retrieval only (streaming):
1. Same ingestion + cache check
2. Call `orchestrator.retrieve_only()` (no LLM generation — much faster)
3. Return evidence + metadata → server.py streams the answer itself

### Background Ingestion:

When you send a message, the engine checks if it's "meaningful content" (not a question, not a greeting). If it is, it ingests it as a memory **in the background** without blocking your chat:

```python
if self._is_meaningful_content(user_message):
    asyncio.create_task(self._background_ingest(user_message, session_id))
```

The `_is_meaningful_content()` filter is aggressive — it rejects:
- Greetings ("hi", "hello", "how are you")
- Questions ("What is my name?", "List my projects")
- Very short messages (< 8 chars, < 3 words)
- Messages that start with question words (what, who, where, etc.)
- Messages under 150 characters that end with `?`

This is important because without this filter, your questions would pollute the vector store and get retrieved as "evidence" for future queries.

---

## 7. Layer 3 — Query Intelligence (Intent Detection & Transformation)

**File:** `backend/src/retrieval/query_engine.py` (529 lines)

This layer has two components:

### 7.1 QueryAnalyzer — Understanding What You're Asking

The analyzer classifies your query into:

**7 Intent Types:**

| Intent | Trigger Keywords | Example Query |
|--------|-----------------|--------------|
| TEMPORAL | "when", "timeline", "last month", "before", dates | "When did I start learning ML?" |
| CAUSAL | "why", "because", "caused", "reason", "led to" | "Why did I switch to Python?" |
| REFLECTIVE | "changed mind", "believe", "think about", "opinion" | "How has my view on AI changed?" |
| PROCEDURAL | "how to", "steps", "process", "method" | "How do I deploy the project?" |
| FACTUAL | "what is", "who", "name", "email", specific facts | "What's my phone number?" |
| COMPARATIVE | "vs", "compare", "difference", "better" | "React vs Vue — what do I prefer?" |
| EXPLORATORY | Catch-all for broad, open-ended queries | "Tell me about my career journey" |

**How detection works:** Pure keyword matching against the query text. No LLM call needed. Each intent has a list of trigger words, and the intent with the most matches wins.

**Complexity Scoring (0.0 to 1.0):**

```
Base score: 0.3
+ 0.1 if word count > 10
+ 0.1 if word count > 20
+ 0.15 if complexity indicators present (analyze, trace, evolve, compare, etc.)
+ 0.1 if multiple question marks
+ 0.1 if CAUSAL or REFLECTIVE intent
+ 0.1 if COMPARATIVE or EXPLORATORY intent
```

**Routing Decision:**
- Complexity < 0.3 → `NO_RETRIEVAL` (answer directly, no memory search)
- Complexity 0.3–0.6 → `SINGLE_STEP` (one agent, one retrieval pass)
- Complexity > 0.6 → `MULTI_STEP` (multiple agents work in parallel)

**Temporal Extraction:** The analyzer also extracts time references from your query:
- Relative: "last week", "yesterday", "2 months ago"
- Absolute: "January 2025", "March 2024"
- These are used to filter retrieval results by date range.

### 7.2 QueryTransformer — Making Better Queries

After analysis, the transformer generates **multiple variants** of your query to improve retrieval coverage. This runs in parallel using `ThreadPoolExecutor(max_workers=4)`:

| Technique | What It Does | When It Runs |
|-----------|-------------|--------------|
| **Multi-Query** | Generates 3 paraphrased versions of your query | Always (complexity > 0.3) |
| **HyDE** | Generates a hypothetical answer, then embeds THAT to find similar real answers | Complexity > 0.45 |
| **Step-Back** | Generates a broader, more abstract version of your question | Complexity > 0.5 AND intent is CAUSAL or REFLECTIVE |
| **Decomposition** | Breaks a complex question into simpler sub-questions | Complexity > 0.7 |

**Example:**

Your query: "Why did I switch from Java to Python for my projects?"

Transformer generates:
- Multi-query 1: "What made me switch programming languages?"
- Multi-query 2: "Reasons for choosing Python over Java"
- Multi-query 3: "My programming language transitions"
- HyDE: "I switched from Java to Python because Python has better ML libraries..."
- Step-back: "What factors influenced my technology choices over time?"

All of these are embedded and searched against the vector store, dramatically increasing the chance of finding relevant memories.

---

## 8. Layer 4 — Agent Orchestrator (The Brain)

**File:** `backend/src/agents/orchestrator.py` (983 lines)

The orchestrator is the decision-maker. It receives the analyzed query and executes the full pipeline.

### The Pipeline (7 Steps):

```
Step 1: Query Analysis          ← Keyword heuristics (fast, ~5ms, no LLM)
Step 2: LLM Routing             ← Only when 0.35 < complexity < 0.65 (ambiguous)
Step 3: Query Transformation    ← Parallel: multi-query + HyDE + step-back + decompose
Step 4: Agent Execution         ← Routes to specialized agent(s)
Step 5: CRAG Evaluation         ← Multi-signal quality check (no LLM call)
Step 6: Self-RAG Critique       ← ONLY if confidence < 0.55 (uses LLM)
Step 7: FLARE Active Retrieval  ← ONLY if confidence < 0.40 (uses LLM + retriever)
```

### Routing Decision:

```
Intent → Agent Mapping:
  TEMPORAL    → TimelineAgent
  CAUSAL      → CausalAgent
  REFLECTIVE  → ReflectionAgent
  COMPARATIVE → ArbitrationAgent
  FACTUAL     → PlanningAgent
  PROCEDURAL  → PlanningAgent
  EXPLORATORY → PlanningAgent
```

### Three Execution Modes:

**NO_RETRIEVAL (complexity < 0.3):**
- Skips all retrieval
- Generates answer directly from the LLM
- Used for greetings, simple factual questions the model can answer from training

**SINGLE_STEP (complexity 0.3–0.6):**
- One specialized agent handles the query
- Agent does retrieval + generates answer
- Most common path for everyday questions

**MULTI_STEP (complexity > 0.6):**
- Multiple agents execute **in parallel** using `asyncio.gather()`
- Primary agent + 1–2 secondary agents based on intent
- Results from all agents are combined, evidence deduplicated
- Final answer synthesized using `generate_faithful()` (Stage 1)

### LLM Routing (Stage 2):

Only triggered when keyword analysis is ambiguous (complexity between 0.35 and 0.65). The model outputs a JSON like:

```json
{
  "intent": "causal",
  "complexity": 0.6,
  "agents": ["causal", "timeline"],
  "needs_retrieval": true,
  "reasoning": "User asking about cause-effect chain"
}
```

If the LLM's complexity score differs from the keyword score by > 0.2, they're averaged. This prevents runaway complexity inflation.

---

## 9. Layer 5 — Specialized Agents (The Workers)

**File:** `backend/src/agents/specialized.py` (500 lines)

Each agent inherits from `BaseAgent` and implements an `execute(query)` method. All agents follow the same pattern:

1. Call `self.retriever.retrieve(query)` to get evidence
2. Filter evidence (remove short, question-like, repetitive content)
3. Build a specialized prompt with the filtered evidence
4. Call `self.llm.generate_faithful()` or `self.llm.generate()` to get an answer
5. Run `self.llm._validate_or_extract()` to catch hallucinations
6. Return `AgentResponse(answer, evidence, confidence, reasoning_trace)`

### The 5 Agents:

**TimelineAgent** — For "when" questions
- Sorts retrieved memories chronologically
- Builds a narrative of events in time order
- Prompt instructs the LLM to create a timeline narrative

**CausalAgent** — For "why" questions
- Calls `self.llm.causal_reason()` (Stage 3 fine-tuning)
- Traces cause-effect chains: "Event A caused Event B because..."
- Also traverses the knowledge graph for explicit causal edges

**ReflectionAgent** — For belief change questions
- Retrieves earliest AND latest memories on the topic separately
- Calls `self.llm.detect_belief_change()` (Stage 5 fine-tuning)
- Identifies contradictions, refinements, and expansions in beliefs

**PlanningAgent** — For complex and factual queries (default agent)
- Most versatile; handles factual, procedural, and exploratory queries
- Uses RAFT generation (Stage 12) — mixes real evidence with distractors to test grounding
- Calls `self.llm.raft_generate()` for distractor-aware answers

**ArbitrationAgent** — For comparative/contradictory queries
- Groups evidence into "perspectives" (pro/con, option A/B)
- Asks the LLM to evaluate each perspective and reach a conclusion
- Used when memories contain contradictory information

### Evidence Filtering (All Agents):

Before using evidence, every agent filters it through `_evidence_texts()`:

```python
# Skip evidence that is:
# - Too short (< 30 chars)
# - Question-like (ends with ?, short)
# - Repetitive (trigram frequency > 3)
# - Stored user queries ("tell me", "what is", etc.)
```

This is critical because without it, stored questions like "What is my name?" would appear as top evidence.

---

## 10. Layer 6 — Hybrid Retrieval Engine (6-Channel Memory Search)

**File:** `backend/src/retrieval/hybrid_retriever.py` (760 lines)

This is where the actual memory search happens. The retriever runs **6 independent search channels in parallel**, then combines their results.

### The 6 Channels:

| # | Channel | How It Works | Weight in RRF |
|---|---------|-------------|---------------|
| 1 | **Dense** (FAISS) | Embeds your query → cosine similarity search against all memory embeddings. Best for semantic meaning. | 0.30 |
| 2 | **Sparse** (BM25) | Classic keyword search — counts term frequency, inverse document frequency. Best for exact word matches. | 0.20 |
| 3 | **Graph** (NetworkX) | Finds entities in your query → traverses the knowledge graph to find connected memories (2-hop). | 0.15 |
| 4 | **Temporal** | If your query has a time reference ("last week"), searches only within that date range. | 0.10 |
| 5 | **Proposition** | Searches against decomposed atomic facts rather than full memory text. Higher precision for specific facts. | 0.05 |
| 6 | **PageIndex** | Queries uploaded documents via the PageIndex cloud API. | Special (bypasses RRF) |

### How Channels Execute:

All channels run simultaneously using `asyncio.gather()`:

```python
results = await asyncio.gather(
    self._dense_retrieve(query),
    self._sparse_retrieve(query),
    self._graph_retrieve(query),
    self._temporal_retrieve(query),
    self._proposition_retrieve(query),
    self._pageindex_retrieve(query),
    return_exceptions=True,
)
```

### Reciprocal Rank Fusion (RRF):

After all channels return results, RRF combines them. The formula:

```
RRF_score(d) = Σ  (weight_channel × 1 / (k + rank_in_channel))
```

Where `k = 60` (constant that prevents high-ranked results from dominating too much).

**Example:** If memory M123 is:
- Ranked #1 in Dense (weight 0.30): 0.30 × 1/(60+1) = 0.00492
- Ranked #3 in Sparse (weight 0.20): 0.20 × 1/(60+3) = 0.00317
- Not found in Graph, Temporal, Proposition
- **Total RRF score**: 0.00809

### Cross-Encoder Reranking:

After RRF fusion, the top results are re-scored using the BGE-reranker-v2-m3 cross-encoder. This is the most accurate relevance scoring method because it processes the query and document **together** (as opposed to FAISS which processes them independently).

The final score blends:
```
final_score = 0.70 × cross_encoder_score + 0.20 × rrf_score + 0.10 × importance_score
```

### BM25 and Proposition Index Rebuilding:

These indexes are rebuilt **lazily** — only when the memory count changes:

```python
# BM25 index is rebuilt from scratch when memory count changes
if current_mem_count != self._bm25_last_count:
    self._rebuild_bm25_index()

# Proposition index is rebuilt similarly
if current_mem_count != self._prop_last_count:
    self._rebuild_proposition_index()
```

### PageIndex Channel:

PageIndex results bypass RRF entirely. If PageIndex returns answers, they're injected at the top of the results list directly. This is because PageIndex returns fully-formed answers from a cloud LLM, not raw text chunks.

---

## 11. Layer 7 — Quality Assurance (CRAG, Self-RAG, FLARE)

**File:** `backend/src/agents/orchestrator.py` (within the `process()` method)

Three progressive quality checks, each more expensive and triggered only when the previous one indicates low quality.

### 11.1 CRAG (Corrective Retrieval-Augmented Generation)

**What it does:** Evaluates the quality of retrieved evidence. No LLM call — pure scoring.

**How it scores:**

```
quality_score = 0.40 × avg_evidence_score
             + 0.20 × max_evidence_score
             + 0.20 × min(evidence_count / 5, 1.0)
             + 0.20 × entity_coverage
```

**Three verdicts:**

| Verdict | Quality Score | Action |
|---------|-------------|--------|
| CORRECT | > 0.55 | Use evidence as-is |
| AMBIGUOUS | 0.30 – 0.55 | Reduce confidence by 15%, try supplementary retrieval with step-back query |
| INCORRECT | < 0.30 | Reduce confidence by 45%, add caveat |

### 11.2 Self-RAG (Self-Reflective RAG)

**When triggered:** Only when confidence < 0.55 AND evidence exists AND answer > 20 chars

**What it does:** Asks the fine-tuned model (Stage 4) to critique the answer on three dimensions:

| Token | Full Name | What It Measures | Scale |
|-------|-----------|-----------------|-------|
| ISREL | Is Relevant | Does the evidence address the query? | 1-10 |
| ISSUP | Is Supported | Is the answer grounded in evidence? | 1-10 |
| ISUSE | Is Useful | Is the answer complete and helpful? | 1-10 |

**Decision based on average score:**

| Avg Score | Action |
|-----------|--------|
| ≥ 7.0 | ACCEPT — boost confidence by 0.1 |
| 5.0 – 7.0 | REVISE — find the weakest dimension, regenerate focusing on it |
| < 5.0 | LOW QUALITY — reduce confidence by 0.15 |

When revising, it identifies the weakest area (relevance, faithfulness, or completeness) and generates a new prompt specifically asking the model to improve that dimension.

### 11.3 FLARE (Forward-Looking Active Retrieval)

**When triggered:** Only when confidence < 0.4 AND evidence exists AND answer > 20 chars

**What it does:**
1. Splits the answer into sentences
2. Identifies "uncertain" sentences (containing words like "might", "possibly", "unclear", "?")
3. For each uncertain sentence (max 2), embeds and retrieves additional evidence
4. If new evidence found, regenerates the answer using `generate_faithful()` with the augmented evidence
5. Boosts confidence by 0.1

**Important note for streaming mode:** FLARE and Self-RAG are **skipped** in `retrieve_only()` mode (which is what the frontend uses). They only run in the full `process()` pipeline.

---

## 12. Layer 8 — The LLM Interface (Fine-Tuned Model Methods)

**File:** `backend/src/llm/__init__.py` (1279 lines)

The `LocalLLM` class wraps ALL interactions with the fine-tuned DeepSeek-R1-7B model. This is the single interface through which every part of the system talks to the LLM.

### Core Methods:

| Method | Stage | Purpose | Token Budget | Temp |
|--------|-------|---------|-------------|------|
| `generate()` | — | General text generation | 512 | 0.3 |
| `classify()` | — | Quick classification from options list | 20 | 0.1 |
| `extract_json()` | — | Generate and parse JSON | 256 | 0.1 |
| `summarize()` | 6 | Concise summarization | 200 | 0.2 |
| `route_query()` | 2 | Structured JSON intent routing | 200 | 0.1 |
| `self_rag_critique()` | 4 | ISREL/ISSUP/ISUSE scoring | 200 | 0.1 |
| `causal_reason()` | 3 | Cause-effect chain analysis | 500 | 0.3 |
| `detect_belief_change()` | 5 | Belief evolution detection | 200 | 0.1 |
| `generate_faithful()` | 1 | Grounded generation with evidence | 500 | 0.1 |
| `raft_generate()` | 12 | Distractor-aware generation | 400 | 0.1 |
| `call_function()` | 13 | Tool/function calling | 200 | 0.1 |

### How `generate()` Works Internally:

```python
def generate(self, prompt, max_tokens=512, temperature=0.3, ...):
    # 1. Tokenize with dynamic context budget
    context_budget = min(3072 + max_tokens, 4096)
    inputs = tokenizer(prompt, truncation=True, max_length=context_budget)

    # 2. Build stop tokens (eos + "User:" + "<|im_end|>" + "<|endoftext|>")

    # 3. Generate with model.generate()
    outputs = model.generate(
        max_new_tokens=min(max_tokens, 2048),
        temperature=max(temperature, 0.01),
        repetition_penalty=1.15,  # or 1.0 for structured/JSON output
    )

    # 4. Strip <think>...</think> tags (THINKING IS REMOVED)
    if "<think>" in generated:
        generated = generated[after_think_end:]  # Only visible content returned

    # 5. Truncate at hallucinated conversation turns
    generated = _truncate_at_stop(generated)  # Stops at "\nUser:", "\nHuman:", etc.

    # 6. Strip leaked special tokens
    # 7. Periodic VRAM defragmentation (every 100 calls)
```

### Critical Behavior — `<think>` Tags Are Stripped:

The DeepSeek-R1 model uses `<think>...</think>` tags for chain-of-thought reasoning. However, in the `generate()` method, **thinking content is always stripped before returning**:

```python
if "<think>" in generated:
    think_end = generated.find("</think>")
    if think_end > -1:
        generated = generated[think_end + len("</think>"):].strip()
```

This means that when agents call methods like `generate_faithful()` or `causal_reason()` (which internally call `generate()`), the thinking trace is **never passed back to the agent**. The thinking is only preserved in two places:
1. The **streaming path** in server.py (thinking tokens are collected separately)
2. The **non-streaming `/api/chat`** endpoint (thinking is split and returned)

### The Validation Pipeline:

After every LLM generation for RAG, the output goes through `_validate_or_extract()`:

```
                LLM generates answer
                        │
                        ▼
            ┌──────────────────────┐
            │ Check for no-info    │ ← Does query ask about something
            │ (false premises)     │   NOT in evidence? (salary, PhD, etc.)
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Check length         │ ← Answer < 30 chars? → try extraction
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Hallucination detect │ ← Match against 60+ known garbage phrases
            │ (pattern matching)   │   Even ONE match → try extraction
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Relevance check      │ ← < 30% query content words in response?
            │ (word overlap)       │   → try extraction
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Evidence dump detect │ ← Response contains "[Document 1:]"?
            │                      │   → try extraction
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Factual query check  │ ← Query matches simple patterns?
            │                      │   (name, email, projects, skills...)
            │                      │   → always try extraction first
            └──────────┬───────────┘
                       │
                       ▼
              Return result (or extraction)
```

### Regex Extraction Fallback:

When the LLM hallucinates, `_extract_answer_from_evidence()` takes over. This is a ~350-line function with regex patterns for:

| Query Type | Pattern It Looks For |
|-----------|---------------------|
| Name | `**Name** format`, "My name is X Y", "Name: X Y" |
| Email | Standard email regex |
| Phone | International phone format |
| University | "Institute of", "IIIT/IIT/NIT + City" |
| Skills | Section headers + programming language names |
| Projects | "📌 Project Name:", bold titles, "Developed/Built..." |
| Location | City names, "from/lives in" patterns |
| Achievements | "Award", "Winner", "Hackathon", emoji markers |
| LinkedIn/GitHub | URL extraction |
| Class 10/12 marks | Percentage patterns |

---

## 13. Layer 9 — Memory Ingestion Pipeline (How Memories Are Born)

**File:** `backend/src/ingestion/__init__.py` (612 lines)

When something worth remembering reaches the system, it goes through an **11-stage enrichment pipeline**:

### The 11 Stages:

```
Raw text arrives
     │
     ▼
 1. VALIDATION
     │  ← Strip null bytes, remove prompt injection markers (<|im_start|> etc.)
     │  ← Truncate if > 10,000 chars
     │  ← Reject if < 2 chars
     ▼
 2. CLASSIFICATION (Memory Type)
     │  ← Keyword matching: "went to" → EPISODIC, "learned" → SEMANTIC, etc.
     │  ← LLM fallback for ambiguous cases
     │  ← Types: episodic, semantic, procedural, reflective
     ▼
 3. EMOTION DETECTION
     │  ← Keyword scoring: "happy/joy/great" → HAPPY, "frustrated/stuck" → FRUSTRATED
     │  ← 8 emotions: happy, sad, angry, anxious, excited, confused, hopeful, frustrated
     │  ← Falls back to NEUTRAL with 0.5 confidence
     ▼
 4. ENTITY EXTRACTION
     │  ← Capitalized words (proper nouns), quoted strings
     │  ← Max 10 entities per memory
     ▼
 5. TOPIC EXTRACTION
     │  ← Keyword matching: "code/programming" → technology, "feel/think" → personal
     │  ← 8 topics: work, health, relationships, learning, technology, finance, personal, creative
     ▼
 6. IMPORTANCE SCORING
     │  ← Base: 0.5
     │  ← +0.1 for > 50 words, +0.1 for > 100 words
     │  ← +0.1 for non-neutral emotion, +0.1 for high emotion confidence
     │  ← +0.15 for reflective type
     │  ← +0.1 for > 2 entities
     │  ← +0.15 for decision keywords ("decided", "chose", "plan to")
     ▼
 7. PROPOSITION DECOMPOSITION (Atomic Facts)
     │  ← LLM-based: "Decompose text into independent atomic facts"
     │  ← Fallback: sentence + clause splitting
     │  ← Max 12 propositions per memory
     ▼
 8. CONTEXTUAL PREFIX
     │  ← If session context available, LLM generates 1-2 sentence prefix
     │  ← Example: "During a career discussion on March 5, 2026"
     ▼
 9. EMBEDDING GENERATION
     │  ← embed_passage(context_prefix + content)
     │  ← BGE-large asymmetric embedding (1024 dimensions)
     │  ← The passage prefix improves embedding quality vs naked text
     ▼
10. STORAGE
     │  ← Vector: Add to FAISS hot index + in-memory dict
     │  ← Metadata: Insert into DuckDB memories table
     │  ← Graph: Add entities as nodes, co-occurring entities get edges
     ▼
11. BELIEF EVOLUTION DETECTION
      ← Find semantically similar old memories (>0.75 similarity)
      ← Skip same-session memories
      ← Require > 1 day time gap
      ← LLM detect_belief_change() (Stage 5) OR keyword stance detection
      ← Classify: CONTRADICTION, REFINEMENT, REINFORCEMENT, NEW_BELIEF
      ← Store as BeliefDelta in DuckDB
```

### Example:

Input: *"I used to think React was the best framework, but now I'm convinced that Next.js is the way to go for production apps."*

After ingestion:
- **Type:** REFLECTIVE (matches "i think", "changed my mind")
- **Emotion:** NEUTRAL (no strong emotion keywords)
- **Entities:** ["React", "Next.js"]
- **Topics:** ["technology"]
- **Importance:** 0.75 (reflective type + 2 entities)
- **Propositions:** ["I used to think React was the best framework", "Now I'm convinced Next.js is the way to go", "Next.js is better for production apps"]
- **Belief Evolution:** CONTRADICTION detected with earlier memory: "React is the best frontend framework" → confidence 0.82

---

## 14. Layer 10 — Storage Stack (Where Everything Lives)

### 14.1 Vector Store (FAISS)

**File:** `backend/src/storage/vector_store.py` (310 lines)

Stores the actual embedding vectors for similarity search.

**Three tiers:**

| Tier | Index Type | Age | Recall | Latency | Score Discount |
|------|-----------|-----|--------|---------|----------------|
| **Hot** | HNSW (M=32, ef=64) | < 30 days | ~98% | ~5ms | None |
| **Warm** | IVF-SQ8 (scalar quantizer) | 30 days – 1 year | ~95% | ~15ms | 5% |
| **Cold** | IVF-PQ (product quantizer) | > 1 year | ~90% | ~25ms | 10% |

- New vectors always go into the **hot** tier
- Monthly migration job moves aging vectors to warm/cold
- If FAISS isn't installed, falls back to numpy brute-force cosine similarity
- Vectors saved to disk as `.npy` files + JSON state

### 14.2 Metadata Store (DuckDB)

**File:** `backend/src/storage/metadata_store.py` (500+ lines)

Relational database storing all structured information. 

**Tables:**

| Table | Purpose | Key Columns |
|-------|---------|------------|
| `memories` | All memory objects | id, content, memory_type, timestamp, emotion, importance, topics, entities, propositions, session_id, source |
| `belief_deltas` | Belief evolution events | topic, old_belief_id, new_belief_id, change_type, confidence |
| `entities` | Knowledge graph entities | canonical_name, aliases, entity_type, memory_ids |
| `graph_edges` | Entity relationships | source_id, target_id, relation, weight |
| `conversations` | Chat history | session_id, role, content, thinking |
| `memory_topics` | Junction table for fast topic lookup | memory_id, topic |
| `memory_entities` | Junction table for fast entity lookup | memory_id, entity |

- Falls back to in-memory dict if DuckDB isn't installed
- Has indexed junction tables for O(1) topic/entity lookups

### 14.3 Knowledge Graph (NetworkX)

**File:** `backend/src/storage/knowledge_graph.py` (313 lines)

A directed graph where:
- **Nodes** = entities (people, projects, places, concepts)
- **Edges** = relationships (co_mentioned, caused, works_with, discussed)

**Features:**
- Multi-hop traversal (explore 2 hops from any entity)
- Shortest path finding
- Causal chain tracing (follow "caused" edges backward)
- Community detection (greedy modularity clustering)
- O(1) entity lookup via inverted name/alias index
- Entity merging (combine duplicate nodes)
- Exports to JSON for frontend visualization

**How entities get into the graph:**
- During ingestion, capitalized words and quoted strings are extracted
- Each entity becomes a node (if not already present)
- Entities co-occurring in the same memory get edges between them
- Entity types are inferred from context (person/place/project/concept)

---

## 15. Layer 11 — Streaming & Token Delivery

**File:** `backend/server.py` (the `_stream_rag_generate()` function, ~200 lines)

When streaming is enabled (default in the frontend), the response delivery works in two phases:

### Phase 1: Evidence Retrieval (Fast)

```python
# Calls the retrieve-only pipeline (no LLM generation)
rag_result = await rag_engine.rag_retrieve(user_message, ...)

# Sends metadata as the FIRST SSE event:
{
  "id": "rag-abc123",
  "delta": "",
  "rag_meta": {
    "evidence": [...],
    "agents_used": ["planning"],
    "confidence": 0.72,
    "query_analysis": {"intent": "factual", "complexity": 0.4},
    "thinking": "...",
    "pipeline_trace": {...}
  }
}
```

### Phase 2: Token Streaming (Slower)

Before streaming even starts, three bypass checks run:

1. **False premise detection** (`_check_no_info_streaming()`) — if asking about salary/PhD/family that doesn't exist in evidence, immediately return "I don't have that info"
2. **Factual extraction** (`_try_extract_factual()`) — for name/email/phone/skills/projects, try regex extraction from evidence. If successful, bypass the LLM entirely.
3. **Greeting detection** — if the query is just "hi" or "how are you", use a greeting-only prompt with no evidence.

If none of these bypass, the LLM generates the response token-by-token using `TextIteratorStreamer`:

```python
# Token-by-token streaming with real-time filtering:
for token_text in streamer:
    accumulated += token_text

    # 1. Suppress <think>...</think> blocks (don't stream thinking)
    if in_think_block:
        continue

    # 2. Clean special tokens (<|im_end|>, etc.)

    # 3. Check for hallucination mid-stream
    for halluc_phrase in _STREAMING_HALLUC_PHRASES:
        if halluc_phrase in accumulated.lower():
            halluc_detected = True  # STOP streaming

    # 4. Check for stop patterns ("\nUser:", etc.)
    for pattern in _STOP_PATTERNS:
        if pattern in accumulated:
            should_stop = True

    # 5. Send clean token to client
    yield f"data: {json.dumps({'id': msg_id, 'delta': clean_token})}\n\n"
```

### Mid-Stream Hallucination Recovery:

If a hallucination phrase is detected mid-stream, the system:
1. Stops streaming immediately
2. Tries `_try_extract_factual()` on the evidence
3. If extraction succeeds, sends a `replace` event to overwrite the partial hallucinated text
4. If extraction fails, finds the most relevant evidence chunk and returns it directly

---

## 16. Layer 12 — Hallucination Defense System

The system has **5 layers of defense** against hallucination, reflecting how frequently the fine-tuned model generates garbage:

### Defense Layer 1: Pre-Generation Bypass

**Files:** `server.py` — `_check_no_info_streaming()` and `_try_extract_factual()`

Before the LLM even runs:
- **False premise detection**: If you ask about salary/PhD/family/companies and the evidence doesn't contain that info, return "I don't have that" immediately
- **Factual extraction**: For simple queries (name, email, projects), extract the answer from evidence using regex. This **bypasses the LLM entirely** for these query types.

### Defense Layer 2: Hallucination Pattern Stripping

**File:** `llm/__init__.py` — `_strip_hallucination_patterns()` (~150 lines)

After LLM generates, known garbage phrases are stripped:
- "Belief evolution", "emotion timeline", "clarity of scope", "key insight"
- "Tracing causal chains across your thinking journey"
- "The relationship is more complex than people say"
- Fake confidence claims: "Confidence: High — based on 5 memories"
- Format leaks: "**Answer:**", "**Evidence:**"
- Robotic prefixes: "Based on your stored memories:"
- Placeholder tokens: "[Name]", "[Email]"
- Inline citations: "[1]", "[2]"
- Raw model tokens that leaked through

### Defense Layer 3: Aggressive Validation

**File:** `llm/__init__.py` — `_validate_or_extract()` (~350 lines)

Checks whether the LLM output is:
1. Long enough (> 30 chars)
2. Free of hallucination indicators (60+ patterns checked)
3. Relevant (> 30% query content words appear in response)
4. Not an evidence dump ("[Document 1:]", "[Memory 1]")
5. Not off-topic

If ANY check fails → tries regex extraction from evidence.

### Defense Layer 4: Streaming Hallucination Filter

**File:** `server.py` — within `_stream_rag_generate()`

~50 phrases in `_STREAMING_HALLUC_PHRASES` are checked against the accumulated streaming text in real-time. If detected, streaming stops and the response is replaced.

### Defense Layer 5: Stop Pattern Truncation

**File:** `server.py` and `llm/__init__.py`

The model sometimes "hallucinates" new conversation turns:
```
Your name is John.

User: What is my email?
Assistant: Your email is...
```

Stop patterns (`\nUser:`, `\nHuman:`, `\nQ:`, etc.) catch this and truncate the response.

---

## 17. Layer 13 — Ambient Voice Pipeline (STT/TTS)

**Files:** `backend/src/ambient/` (10 files)

The ambient voice system enables:
- **Continuous listening** — VAD (Voice Activity Detection) detects when someone is speaking
- **Speech-to-text** — faster-whisper transcribes speech
- **Speaker identification** — identifies WHO is speaking (after enrollment)
- **Auto-ingestion** — transcribed conversations automatically become memories
- **Text-to-speech** — reads responses aloud using a voice model
- **VRAM guarding** — monitors GPU memory and unloads voice models when the LLM needs space

### Key Components:

| File | Purpose |
|------|---------|
| `audio_capture.py` | Records audio from microphone in a background thread |
| `vad.py` | Voice Activity Detection — detects speech vs silence |
| `transcription.py` | Whisper-based speech-to-text |
| `speaker_id.py` | Speaker identification using voice embeddings |
| `enrollment.py` | Records a voice sample to learn "owner" vs "other" |
| `tts.py` | Text-to-speech synthesis |
| `conversation.py` | Groups speech segments into conversations |
| `vram_guard.py` | Monitors GPU memory, unloads models when needed |
| `config.py` | Configuration dataclass |

### Voice Query Pipeline:

```
Audio → Decode base64 → Transcribe (Whisper) → RAG Chat → TTS → Audio response
```

---

## 18. Layer 14 — PageIndex (Cloud Document Retrieval)

**File:** `backend/src/storage/pageindex_store.py`

PageIndex is an optional cloud service for document-level retrieval:

1. **Upload** a PDF via `/api/documents/upload`
2. PageIndex processes it into a reasoning tree structure
3. When queries are about "documents" or "files", the PageIndex channel is activated
4. PageIndex returns pre-generated answers from its cloud LLM

**Special handling in streaming:**
- If PageIndex evidence is present AND the query mentions documents/PDFs:
  - The answer is returned **directly** from PageIndex, bypassing the local LLM entirely
- If PageIndex evidence is present BUT the query is personal (not about documents):
  - PageIndex evidence is **stripped out**, only local memories are used

---

## 19. Layer 15 — Observability & Pipeline Tracing

**File:** `backend/src/agents/orchestrator.py` (PipelineTrace class)

Every query execution generates a detailed trace that records:

```python
PipelineTrace:
    trace_id: str              # Unique identifier
    query: str                 # Original query
    timestamp: str             # When it started
    
    # Analysis results
    query_analysis: dict       # intent, complexity, routing, entities, topics
    query_transform: dict      # Multi-queries, HyDE, step-back, sub-queries
    routing_decision: str      # NO_RETRIEVAL, SINGLE_STEP, MULTI_STEP
    
    # Retrieval details
    retrieval_channels: list   # Per-channel: result count, duration
    reranking: dict            # Method, duration, input count
    
    # Quality evaluation
    crag_evaluation: dict      # Quality score, verdict, entity coverage
    self_rag_critique: dict    # ISREL, ISSUP, ISUSE scores, verdict
    flare_trace: dict          # Uncertain sentences, new evidence count
    
    # Pipeline steps
    steps: list                # Each step: name, type, status, duration, details
    
    # Final stats
    total_duration_ms: float
    final_confidence: float
    evidence_count: int
    token_usage: dict
```

These traces are stored in an in-memory ring buffer (last 100) and exposed via `/api/rag/traces`.

The frontend has an **Observability** view that displays these traces with analytics:
- Average duration, confidence, evidence count
- Channel usage breakdown
- CRAG/Self-RAG/FLARE activation rates
- Cache hit rate

---

## 20. Layer 16 — Frontend (Next.js UI)

**Files:** `frontend/src/app/page.tsx`, `frontend/src/components/ChatPanel.tsx`

The frontend is a Next.js application with multiple views:

| View | Purpose |
|------|---------|
| **Chat** | Main conversation interface with streaming responses |
| **Memories** | Browse, search, and delete stored memories |
| **Graph** | Interactive knowledge graph visualization |
| **Dashboard** | System stats, GPU usage, model info |
| **Observability** | Pipeline trace viewer and analytics |
| **Ambient** | Voice pipeline controls and live transcript |
| **Documents** | Upload and manage PageIndex documents |

### Chat Panel Streaming:

The ChatPanel uses SSE (Server-Sent Events) with a batched token buffer:

```typescript
// Tokens are buffered for 50ms before rendering
const TOKEN_BUFFER_INTERVAL = 50; // ms

// On receiving SSE data:
if (data.rag_meta) {
    // First event: metadata (evidence, agents, thinking, trace)
    setCurrentEvidence(data.rag_meta.evidence);
    setCurrentThinking(data.rag_meta.thinking);
}
if (data.delta) {
    // Subsequent events: append token to buffer
    tokenBuffer.current += data.delta;
}
if (data.replace) {
    // Hallucination recovery: replace accumulated text
    message.content = data.replace;
}
if (data.done) {
    // Final event: flush buffer, finalize message
}
```

Messages are persisted to `localStorage` for cross-session continuity.

---

## 21. Data Model Reference

The core data models are defined in `backend/src/models/` (gitignored — exists on disk but not in git):

### CausalMemoryObject

The fundamental unit of storage. Every memory is one of these:

```
CausalMemoryObject:
    id: str                 # UUID
    content: str            # The actual text
    memory_type: MemoryType # episodic, semantic, procedural, reflective
    timestamp: datetime     # When it was created
    emotion: EmotionLabel   # happy, sad, angry, anxious, excited, confused, hopeful, frustrated, neutral
    emotion_confidence: float
    importance: float       # 0.0 to 1.0
    topics: list[str]       # work, health, technology, etc.
    entities: list[str]     # Extracted proper nouns
    entity_ids: list[str]   # Linked graph node IDs
    causes: list[str]       # What caused this event
    effects: list[str]      # What this event caused
    causal_description: str
    context_prefix: str     # Anthropic-style contextual prefix
    propositions: list[str] # Decomposed atomic facts
    raptor_level: int       # 0 = raw (note: RAPTOR not implemented)
    raptor_children: list   # (not used)
    session_id: str
    source: str             # chat, manual, ambient, etc.
    metadata: dict
    embedding: list[float]  # 1024-d BGE vector
```

### Other Key Models:

| Model | Purpose |
|-------|---------|
| `MemoryQuery` | Analyzed query with intent, complexity, routing, variants |
| `QueryIntent` | Enum: TEMPORAL, CAUSAL, REFLECTIVE, PROCEDURAL, FACTUAL, COMPARATIVE, EXPLORATORY |
| `RoutingStrategy` | Enum: NO_RETRIEVAL, SINGLE_STEP, MULTI_STEP |
| `RetrievalResult` | memory + score + channel name |
| `OrchestratorResponse` | answer + thinking + evidence + agents_used + confidence + trace |
| `AgentResponse` | Per-agent result |
| `BeliefDelta` | Detected belief change event |
| `EntityNode` | Knowledge graph entity with aliases, type, memory_ids |
| `GraphEdge` | Relationship between entities with type and weight |
| `PipelineTrace` | Full observability trace for one query |
| `PipelineStep` | Individual step within a trace |
| `CRAGEvaluation` | CRAG quality verdict and scores |
| `SelfRAGCritique` | Self-RAG ISREL/ISSUP/ISUSE scores |
| `FLARETrace` | FLARE active retrieval details |

---

## 22. Complete Request Lifecycle — Step by Step

Here is EXACTLY what happens when you send a message through the frontend (streaming mode):

```
1. Frontend sends POST /api/rag/chat {messages: [...], stream: true}

2. server.py: rag_chat() handler
   → Validates request, extracts user_message and history
   → Returns StreamingResponse(_stream_rag_generate(...))

3. _stream_rag_generate() starts:
   
   3a. RETRIEVAL PHASE:
       → Calls rag_engine.rag_retrieve(user_message, ...)
       → engine.py: 
         → Checks _is_meaningful_content() → if yes, fires background ingestion task
         → Checks cache (exact match → semantic match)
         → Calls orchestrator.retrieve_only(user_message, ...)
         → orchestrator.py:
           → QueryAnalyzer.analyze() — keyword-based intent + complexity (no LLM)
           → If ambiguous (0.35 < complexity < 0.65): LLM routing (Stage 2)
           → QueryTransformer.transform() — multi-query + HyDE + step-back
           → HybridRetriever.retrieve(query):
             → Parallel: Dense + Sparse + Graph + Temporal + Proposition + PageIndex
             → RRF fusion of all channels
             → Cross-encoder reranking (BGE-reranker-v2-m3)
           → Multi-signal confidence scoring
           → CRAG evaluation (no LLM, just scoring)
           → Returns evidence + confidence + trace

   3b. METADATA SSE EVENT:
       → Sends rag_meta: {evidence, agents_used, confidence, thinking, trace}

   3c. PRE-GENERATION CHECKS:
       → _check_no_info_streaming() — false premise detection
       → _try_extract_factual() — regex extraction for simple facts
       → Greeting detection
       → If any bypass succeeds → send answer, done

   3d. PROMPT CONSTRUCTION:
       → Selects prompt template (RAG / Document / Greeting)
       → Injects filtered evidence into prompt
       → Evidence selection: PageIndex first, then top local memories

   3e. TOKEN STREAMING:
       → model.generate() with TextIteratorStreamer
       → For each token:
         → Suppress <think>...</think> blocks
         → Clean special tokens
         → Check hallucination phrases (50+ patterns)
         → Check stop patterns (User:, Human:, etc.)
         → Send clean token as SSE event

   3f. POST-STREAM:
       → If hallucination detected: try extraction, send replace event
       → Send done event
       → Store assistant turn in conversation history
```

---

## 23. What's in RAG-Architecture.md but NOT in Code

The RAG-Architecture.md document describes a comprehensive system with 25+ techniques. Here's what's **actually implemented** vs. what remains **design-document only**:

### Fully Implemented ✅

| Feature | Code Location |
|---------|--------------|
| Multi-channel hybrid retrieval (Dense + Sparse + Graph + Temporal + Proposition) | `hybrid_retriever.py` |
| RRF fusion | `hybrid_retriever.py` |
| Cross-encoder reranking (BGE-reranker-v2-m3) | `hybrid_retriever.py` |
| 5 specialized agents (Timeline, Causal, Reflection, Planning, Arbitration) | `specialized.py` |
| Agent orchestrator with Adaptive-RAG routing | `orchestrator.py` |
| CRAG quality evaluation | `orchestrator.py` |
| Self-RAG ISREL/ISSUP/ISUSE critique | `orchestrator.py` |
| FLARE active retrieval | `orchestrator.py` |
| Query analysis (intent + complexity) | `query_engine.py` |
| Query transformation (multi-query, HyDE, step-back, decomposition) | `query_engine.py` |
| Belief evolution detection | `ingestion/__init__.py`, `llm/__init__.py` |
| Knowledge graph with community detection | `knowledge_graph.py` |
| FAISS vector store with hot/warm/cold tiers | `vector_store.py` |
| DuckDB metadata storage | `metadata_store.py` |
| Multi-level caching | `src/cache/` (gitignored) |
| Proposition decomposition | `ingestion/__init__.py` |
| Contextual chunking (Anthropic-style) | `ingestion/__init__.py` |
| 15-stage fine-tuning training pipeline | `training_data/`, `train_model.py` |
| Pipeline observability traces | `orchestrator.py` |
| PageIndex document retrieval (6th channel) | `pageindex_store.py` |
| Ambient voice pipeline (STT/TTS/VAD/Speaker ID) | `ambient/` |

### NOT Implemented (Design Only) ❌

| Feature | Claimed In | Status |
|---------|-----------|--------|
| **RAPTOR Tree Indexing** | RAG-Architecture.md §4.1 | DuckDB has `raptor_level` and `raptor_children` columns, but NO RAPTOR code exists. No clustering, no hierarchical summarization. |
| **ColBERTv2 Late Interaction** | RAG-Architecture.md §1.2 | Not implemented. Retrieval uses BGE-large dense + BM25 sparse. |
| **SPLADE Sparse Retrieval** | RAG-Architecture.md §1.2 | Not implemented. Uses simple BM25 via `rank_bm25` library. |
| **SetFit Classifiers** | RAG-Architecture.md §5.1 | Not implemented. Intent detection uses keyword matching, not trained classifiers. |
| **Semantic Chunking** | RAG-Architecture.md §13.2 | Not implemented as a standalone component. Ingestion uses sentence splitting, not embedding-similarity boundary detection. |
| **RAGChecker Evaluation** | RAG-Architecture.md §13.8 | Not implemented. No automated evaluation framework exists in code. |
| **Token Efficiency Optimization** | RAG-Architecture.md §13.11 | Not implemented as a distinct system. Some implicit efficiency (skipping LLM calls when confidence is high). |
| **Retriever Fine-tuning Pipeline** | RAG-Architecture.md §13.12 | Not implemented. Embedding model (BGE) is used as-is, not fine-tuned on user data. |
| **Self-Improvement Loop** | RAG-Architecture.md §13.13 | Not implemented. No automated feedback/improvement loop. |
| **Multi-Modal Pipeline** (OCR, vision, audio processing) | RAG-Architecture.md §1.2 | Not implemented. System only handles text. |
| **Memory Consolidation** | RAG-Architecture.md §9.6 | Not implemented. Old memories don't get summarized or consolidated. |
| **TreeRAG** | RAG-Architecture.md §1.2 | Not implemented. |
| **Chain-of-Retrieval** | RAG-Architecture.md §1.2 | Not implemented as described. Multi-step queries are handled by parallel agent execution, not chained retrieval. |
| **Failure-Aware Query Refinement** | RAG-Architecture.md §13.9 | Not implemented. Queries don't get refined based on failure type classification. |
| **Function Calling in Pipeline** | RAG-Architecture.md, AVAILABLE_TOOLS in orchestrator.py | `call_function()` method exists in LLM but is **never invoked** anywhere in the pipeline. Tools are defined but unused. |
| **Data-at-Rest Encryption** | RAG-Architecture.md §12.2 | Not implemented. DuckDB and FAISS files are stored unencrypted. |
| **API Key Authentication** | RAG-Architecture.md §12.2 | Not implemented. Server binds to 0.0.0.0 with no authentication (CORS only). |

---

## 24. File Map — Every Source File and Its Purpose

### Backend Core (`backend/`)

| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | 2028 | FastAPI server: all API endpoints, streaming, prompts, hallucination filters |
| `src/engine.py` | 696 | Central RAG engine: wires all 11 subsystems together |
| `src/llm/__init__.py` | 1279 | LLM interface: all model methods (generate, route, critique, etc.) |
| `src/agents/orchestrator.py` | 983 | Pipeline orchestrator: routing, CRAG, Self-RAG, FLARE |
| `src/agents/specialized.py` | 500 | 5 specialized agents: Timeline, Causal, Reflection, Planning, Arbitration |
| `src/retrieval/hybrid_retriever.py` | 760 | 6-channel hybrid retrieval with RRF fusion and cross-encoder reranking |
| `src/retrieval/query_engine.py` | 529 | Intent detection, complexity scoring, query transformation |
| `src/ingestion/__init__.py` | 612 | 11-stage memory ingestion pipeline |
| `src/storage/vector_store.py` | 310 | FAISS vector store with hot/warm/cold tiers |
| `src/storage/metadata_store.py` | 500+ | DuckDB metadata storage (6 tables) |
| `src/storage/knowledge_graph.py` | 313 | NetworkX knowledge graph with community detection |
| `src/storage/pageindex_store.py` | ~300 | PageIndex cloud document retrieval integration |
| `src/models/` | (gitignored) | Data models: CausalMemoryObject, MemoryQuery, enums, etc. |
| `src/cache/` | (gitignored) | MultiLevelCache: exact + semantic + embedding caching |

### Backend Ambient Voice (`backend/src/ambient/`)

| File | Purpose |
|------|---------|
| `audio_capture.py` | Microphone recording in background thread |
| `vad.py` | Voice Activity Detection |
| `transcription.py` | Whisper-based speech-to-text |
| `speaker_id.py` | Speaker identification via voice embeddings |
| `enrollment.py` | Voice enrollment recording |
| `tts.py` | Text-to-speech synthesis |
| `conversation.py` | Groups speech into conversations |
| `config.py` | Ambient configuration dataclass |
| `vram_guard.py` | GPU memory monitoring and model offloading |

### Frontend (`frontend/src/`)

| File | Purpose |
|------|---------|
| `app/page.tsx` | Main application layout with sidebar and views |
| `app/layout.tsx` | Root layout with metadata |
| `components/ChatPanel.tsx` | Chat interface with SSE streaming |
| (plus other component files for each view) |

### Configuration & Scripts

| File | Purpose |
|------|---------|
| `config/training_config.py` | 15-stage fine-tuning configuration |
| `config/pageindex_config.py` | PageIndex API settings |
| `scripts/fine_tune_cortex.py` | QLoRA fine-tuning script |
| `scripts/generate_datasets.py` | Training data generation |
| `scripts/ingest_user_files.py` | Bulk file ingestion |
| `training_data/*.json` | 15 training stage datasets + user memories |
| `raw_data/*.md` | Source documents for ingestion |

---

*This document was generated by exhaustively reading every source file in the Cortex Lab codebase. It reflects the actual implementation as of March 2026.*
