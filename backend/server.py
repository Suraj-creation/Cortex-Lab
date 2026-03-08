"""
FastAPI Backend Server for Cortex Lab — Fine-Tuned DeepSeek-R1-7B
Serves the 15-stage curriculum fine-tuned model via REST API + Server-Sent Events (streaming).
Includes full Agentic RAG system with memory, retrieval, and multi-agent reasoning.

Model: DeepSeek-R1-Distill-Qwen-7B fine-tuned across 15 stages:
  Faithfulness → Agentic → Causal → Self-RAG → Belief → Summarization →
  Dialogue → LongContext → DPO → UserStyle → ORPO → RAFT → FunctionCalling →
  RFT → SPIN
"""

import os
import sys
import time
import json
import re
import asyncio
import uuid
import traceback
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    TextIteratorStreamer = None
from threading import Thread

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv not installed — use system env vars

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Add backend dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine import rag_engine
from src.prompts import PromptBuilder

# ── Configuration ────────────────────────────────────────────────────────────

# Fine-tuned model path — auto-detect latest merged stage
FINE_TUNED_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fine_tuned")

def _find_latest_merged_model():
    """Find the latest merged model from our fine-tuning pipeline."""
    # Check stages in reverse order (15 → 1) for the latest merged model
    stage_names = [
        "stage15_spin", "stage14_rft", "stage13_function_calling",
        "stage12_raft", "stage11_orpo", "stage10_user_style",
        "stage9_dpo", "stage8_longcontext", "stage7_dialogue",
        "stage6_summarization", "stage5_belief", "stage4_selfrag",
        "stage3_causal", "stage2_agentic", "stage1_faithfulness",
    ]
    for stage in stage_names:
        merged_path = os.path.join(FINE_TUNED_BASE, stage, "merged")
        if os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
            print(f"  🎯 Found fine-tuned model: {stage}/merged")
            return merged_path
    return None

_fine_tuned_path = _find_latest_merged_model()
MODEL_NAME = os.environ.get("MODEL_NAME",
    _fine_tuned_path if _fine_tuned_path else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

def _count_completed_stages():
    """Count how many training stages have completed."""
    count = 0
    for i in range(1, 16):
        stage_dirs = [d for d in os.listdir(FINE_TUNED_BASE) if d.startswith(f"stage{i}_")]
        for sd in stage_dirs:
            meta = os.path.join(FINE_TUNED_BASE, sd, "training_meta.json")
            if os.path.exists(meta):
                count += 1
    return count

USE_4BIT   = os.environ.get("USE_4BIT", "true").lower() == "true"   # Default ON for 7B
USE_8BIT   = os.environ.get("USE_8BIT", "false").lower() == "true"
HOST       = os.environ.get("HOST", "0.0.0.0")
PORT       = int(os.environ.get("PORT", "8000"))

# ── Global state ─────────────────────────────────────────────────────────────

model = None
tokenizer = None
model_loaded = False
model_info = {}

# ── Concurrency & Timeout Guards (§9.1, §9.2) ───────────────────────────────
_inference_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent RAG/chat requests
_REQUEST_TIMEOUT = 180.0  # Hard timeout in seconds for any LLM request

# ── Lifespan – loads model once on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, model_loaded, model_info

    print("\n" + "=" * 64)
    print("  Cortex Lab  ·  Fine-Tuned DeepSeek-R1-7B  ·  FastAPI Backend")
    print("=" * 64 + "\n")

    # Check if local model loading should be skipped (Gemini-only mode)
    skip_local = os.environ.get("SKIP_LOCAL_MODEL", "false").lower() == "true"

    if skip_local:
        print("  ⚡ SKIP_LOCAL_MODEL=true — running in Gemini-only mode")
        print("  ⚡ Local model will NOT be loaded. Only Gemini API available.\n")
        model_loaded = False
        model_info = {
            "name": "Gemini-Only Mode (no local model)",
            "parameters": "",
            "quantization": "",
            "device": "API",
            "gpu_memory": "",
            "max_context": 1048576,
            "load_time_seconds": 0,
            "fine_tuned": False,
            "training_stages_completed": 0,
            "model_path": "",
            "base_model": "gemini-2.5-flash",
        }

        # Still initialize RAG engine without a local model
        # Run in thread so the event loop stays responsive during init
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: rag_engine.init(model=None, tokenizer=None)
            )
        except Exception as e:
            print(f"  ⚠ RAG Engine initialization error: {e}")
            print("  ⚠ RAG features may be limited in Gemini-only mode.")

        # Set provider to Gemini since we have no local model
        if rag_engine.initialized and hasattr(rag_engine.llm, "set_provider"):
            rag_engine.llm.set_provider("gemini")

        model_loaded = True  # Mark ready so health check returns "ok"
        print(f"\n  Server ready → http://{HOST}:{PORT}\n")

        yield
        rag_engine.shutdown()
        return

    # ── Tokenizer ────────────────────────────────────────────────────────
    print(f"[1/2] Loading tokenizer from: {MODEL_NAME[:80]}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer ready")

    # ── Model ────────────────────────────────────────────────────────────
    # Help PyTorch manage GPU memory more efficiently
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_kwargs = {"trust_remote_code": True}

    if USE_4BIT:
        print("[2/2] Loading model in 4-bit with CPU offloading …")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # Enable Flash Attention 2 for 30-50% faster inference (§1.1)
        try:
            import importlib
            if importlib.util.find_spec("flash_attn") is not None:
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("  ✓ Flash Attention 2 enabled")
            else:
                print("  ⚠ flash-attn not installed, using sdpa attention")
                load_kwargs["attn_implementation"] = "sdpa"
        except Exception:
            load_kwargs["attn_implementation"] = "sdpa"
            print("  ⚠ Using SDPA attention (Flash Attention 2 unavailable)")
        # Set max memory to leave some GPU memory free
        max_memory = {0: "17GB", "cpu": "30GB"}
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["offload_folder"] = "offload"
    elif USE_8BIT:
        print("[2/2] Loading model in 8-bit …")
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        print("[2/2] Loading model in full precision …")
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    elapsed = time.time() - t0

    if not (USE_4BIT or USE_8BIT):
        model.eval()

    model_loaded = True
    quant = "4-bit" if USE_4BIT else ("8-bit" if USE_8BIT else "fp16/fp32")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem  = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"

    completed_stages = _count_completed_stages()
    model_display_name = "DeepSeek-R1-7B (Fine-Tuned)" if _fine_tuned_path else "DeepSeek-R1-Distill-Qwen-7B"

    model_info = {
        "name": model_display_name,
        "parameters": "7B",
        "quantization": quant,
        "device": gpu_name,
        "gpu_memory": gpu_mem,
        "max_context": 32768,
        "load_time_seconds": round(elapsed, 1),
        "fine_tuned": _fine_tuned_path is not None,
        "training_stages_completed": completed_stages,
        "model_path": MODEL_NAME[:80],
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }

    print(f"  ✓ Model loaded in {elapsed:.1f}s  ({quant} on {gpu_name})")
    print(f"  ✓ Fine-tuned: {_fine_tuned_path is not None} ({completed_stages}/15 stages)")
    print(f"\n  Server ready → http://{HOST}:{PORT}\n")

    # ── Initialize RAG Engine ────────────────────────────────────────────
    try:
        rag_engine.init(model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"  ⚠ RAG Engine initialization error: {e}")
        print("  ⚠ RAG features will be unavailable, basic chat still works.")

    yield  # ← app runs here

    # cleanup
    rag_engine.shutdown()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cortex Lab — Fine-Tuned DeepSeek-R1-7B Agentic RAG API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://192.168.3.169:3000",
        "https://*.trycloudflare.com",
    ],
    allow_origin_regex=r"https?://.*\.(trycloudflare\.com|localhost|192\.168\.\d+\.\d+)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for large responses (§9.6)
app.add_middleware(GZipMiddleware, minimum_size=500)

# ── API Key Authentication Middleware (§Gap 7) ───────────────────────────────

_CORTEX_API_KEY = os.environ.get("CORTEX_API_KEY", "")
_AUTH_ENABLED = bool(_CORTEX_API_KEY)  # Auto-enable when env var is set

# Paths that don't require authentication
_PUBLIC_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token authentication middleware.
    Enable by setting CORTEX_API_KEY environment variable.
    All /api/* endpoints require: Authorization: Bearer <key>
    """
    async def dispatch(self, request: Request, call_next):
        if not _AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path
        # Skip auth for public endpoints and WebSocket upgrades
        if path in _PUBLIC_PATHS or request.scope.get("type") == "websocket":
            return await call_next(request)

        # Check for Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                content='{"detail":"Missing or invalid Authorization header. Use: Bearer <API_KEY>"}',
                status_code=401,
                media_type="application/json",
            )

        token = auth_header[7:].strip()
        if token != _CORTEX_API_KEY:
            return Response(
                content='{"detail":"Invalid API key"}',
                status_code=403,
                media_type="application/json",
            )

        return await call_next(request)


if _AUTH_ENABLED:
    app.add_middleware(APIKeyAuthMiddleware)
    print(f"  🔒 API key authentication ENABLED (set via CORTEX_API_KEY)")
else:
    print(f"  ⚠ API key authentication DISABLED — set CORTEX_API_KEY env var to enable")

# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(4096, ge=1, le=32768)
    stream: bool = False
    llm_provider: str = Field("local", description="'local' for fine-tuned DeepSeek or 'gemini' for Gemini API")

class ChatResponse(BaseModel):
    id: str
    model: str
    created: int
    content: str
    thinking: Optional[str] = None
    usage: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict

# ── RAG Schemas ──────────────────────────────────────────────────────────────

class RAGChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(4096, ge=1, le=32768)
    stream: bool = False
    use_rag: bool = True  # Enable/disable RAG enhancement
    session_id: str = ""
    llm_provider: str = Field("local", description="'local' for fine-tuned DeepSeek or 'gemini' for Gemini API")

class MemoryIngestRequest(BaseModel):
    content: str
    source: str = "manual"
    session_id: str = ""

class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ModelInfoResponse(BaseModel):
    """Detailed model information for the frontend."""
    name: str
    parameters: str
    quantization: str
    device: str
    gpu_memory: str
    max_context: int
    load_time_seconds: float
    fine_tuned: bool
    training_stages_completed: int
    base_model: str

# ── Helpers ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Cortex Lab, a personal AI memory and reasoning assistant. "
    "You help the user by answering their questions thoughtfully and concisely. "
    "If the user asks about personal information (their name, preferences, etc.) "
    "that you don't actually know, honestly say you don't have that information yet "
    "and suggest they can teach you by telling you. "
    "Never fabricate personal details about the user. "
    "Keep responses focused and do NOT generate follow-up questions or continue "
    "the conversation on behalf of the user."
)

# Stop patterns: if the model starts generating these, it's hallucinating a new turn
_STOP_PATTERNS = ["\nUser:", "\nuser:", "\nHuman:", "\nhuman:", "\nQ:", "\nA:", "\n\nUser "]

# ── Streaming hallucination filters ─────────────────────────────────────────
# These are known patterns the fine-tuned model generates instead of real answers.
# Used to detect and suppress hallucination in the streaming path.
_STREAMING_HALLUC_PHRASES = [
    "belief evolution", "the key insight is", "clarity of scope",
    "clarity requires constraints", "systems matter more than goals",
    "the intersection of deep work", "personal growth and modern technology",
    "deep work patterns", "sporadic bursts", "cumulative insight",
    "emotional trajectory", "emotion timeline", "deliberate practice",
    "the bottleneck has shifted", "the timeline for meaningful",
    "tracing causal chains", "your thinking journey",
    "comprehensive answer to your question",
    "here's a comprehensive answer", "here's the revised answer",
    "here's what your beliefs", "revised answer focused on",
    "according to research on personal growth",
    "reflecting on my relationship with my mentor",
    "here's a decomposed analysis",
    "you strongly believe that",
    "the relationship is more complex than people say",
    "lived experiences",
    "synthesizing", "based on strong empirical evidence",
    # Self-RAG format leaks
    "**answer:**", "**evidence:**", "**confidence:**",
    "**relevance:**", "**sources:**",
    "answer:\n", "evidence:\n", "confidence: high",
    "confidence: medium", "confidence: low",
    # Robotic database-dump prefixes
    "based on your stored memories:",
    "according to the evidence provided:",
    "from your stored memories:",
]


def _check_no_info_streaming(query: str, evidence_texts: list) -> str:
    """
    Pre-generation check: if the query asks about data that doesn't exist
    in evidence, return a polite 'no info' message instead of generating.
    This prevents the model from hallucinating answers about nonexistent data.
    Uses word-boundary regex to avoid false positives (e.g., "earn" in "learning").
    """
    import re as _re
    q = query.lower().strip()
    all_ev = "\n".join(e.lower() for e in evidence_texts) if evidence_texts else ""

    # If no evidence at all, don't trigger false "no info" checks — let the
    # LLM handle it. This prevents false positives when evidence was stripped.
    if not evidence_texts:
        return ""

    def _ev_has_word(keywords):
        for k in keywords:
            if _re.search(r'\b' + _re.escape(k) + r'\b', all_ev):
                return True
        return False

    def _q_has_phrase(triggers):
        """Check if query contains trigger phrases using word-boundary matching
        to avoid false positives like 'earning' matching in 'learning'."""
        for t in triggers:
            # Multi-word phrases: use simple 'in' (already specific enough)
            if ' ' in t:
                if t in q:
                    return True
            else:
                # Single words: use word-boundary regex to prevent partial matches
                if _re.search(r'\b' + _re.escape(t) + r'\b', q):
                    return True
        return False

    checks = [
        # Salary / compensation — strict whole-word evidence matching
        (["salary", "compensation", "how much do i earn", "how much do i make",
          "my income", "my pay", "how much does"],
         ["salary", "compensation", "annual income", "monthly pay",
          "ctc", "lpa", "stipend", "remuneration"],
         "I don't have any salary or compensation details yet. Feel free to share that info and I'll remember it!"),

        # PhD / Doctoral
        (["phd", "doctoral", "dissertation", "phd thesis"],
         ["phd", "doctoral", "dissertation"],
         "I don't have any PhD or doctoral information. If that's part of your journey, let me know!"),

        # Marriage / family — broader triggers
        (["wife", "husband", "spouse", "children", "kids",
          "son", "daughter", "married", "wedding",
          "have a wife", "have a husband", "have children",
          "is he married", "is she married", "marital status",
          "family members"],
         ["wife", "husband", "spouse", "married", "wedding",
          "children names", "son named", "daughter named"],
         "I don't have any family or marriage details yet. You can share that with me anytime!"),

        # Published papers
        (["published research", "research paper", "published paper",
          "my publications", "published papers", "research papers"],
         ["published paper", "publication in", "journal paper",
          "ieee", "arxiv", "conference paper"],
         "I don't have any research publication records yet. If you've published papers, tell me about them!"),
    ]

    for query_triggers, evidence_keywords, msg in checks:
        if _q_has_phrase(query_triggers):
            if not _ev_has_word(evidence_keywords):
                return msg

    # Employment at specific companies (false premise)
    companies = ["google", "microsoft", "amazon", "meta", "apple", "tesla",
                 "netflix", "uber", "airbnb", "spotify"]
    for company in companies:
        if f"work at {company}" in q or f"{company} job" in q or f"employed at {company}" in q:
            if not _re.search(r'\b' + _re.escape(company) + r'\b', all_ev):
                return f"I don't have any information about working at {company.title()}. If that's part of your experience, tell me about it!"

    # Stanford / MIT / Harvard etc. (false premise universities)
    false_unis = ["stanford", "mit ", "harvard", "oxford", "cambridge", "princeton", "yale"]
    for uni in false_unis:
        if uni in q and uni.strip() not in all_ev:
            return f"I don't have any details about {uni.strip().title()} in what I know about you."

    return ""


def _fix_person_pronouns(text: str) -> str:
    """Convert first-person evidence text to second-person for natural responses.
    'My name is X' → 'Your name is X', 'I have' → 'You have', etc."""
    import re as _re
    # Only fix at sentence starts or after punctuation
    replacements = [
        (r'\bMy\b', 'Your'),
        (r'\bmy\b', 'your'),
        (r'\bI am\b', 'You are'),
        (r'\bI\'m\b', "You're"),
        (r'\bI have\b', 'You have'),
        (r'\bI was\b', 'You were'),
        (r'\bI do\b', 'You do'),
        (r'\bI also\b', 'You also'),
        (r'^I\b', 'You'),
        (r'\. I\b', '. You'),
    ]
    for pattern, repl in replacements:
        text = _re.sub(pattern, repl, text)
    return text


def _try_extract_factual(query: str, evidence_texts: list) -> str:
    """
    Pre-generation extraction: for simple factual queries, try to extract
    the answer directly from evidence using regex patterns.
    This bypasses the LLM entirely for queries it consistently hallucinates on.
    Returns conversational, natural-sounding answers.
    """
    import re
    q = query.lower().strip()

    # Remove greetings
    for prefix in ["hey ", "hi ", "hello ", "hey, ", "hi, "]:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    # ── Guard: Skip extraction for complex synthesis/philosophical queries ──
    # These need a comprehensive LLM-generated answer, not a regex snippet.
    _synthesis_indicators = [
        "vision", "philosophy", "paradigm", "ideology", "worldview",
        "core belief", "fundamental", "reimagining", "redefining",
        "changing", "transforming", "revolutionizing", "rethinking",
        "system", "framework", "approach to", "perspective on",
        "dream about", "aspiration", "what drives",
        "how do you see", "what do you think about",
        "comprehensive", "in detail", "elaborate", "tell me everything",
        "all about", "deep dive", "summarize everything",
    ]
    if any(ind in q for ind in _synthesis_indicators):
        # Exception: allow simple factual queries that happen to contain these words
        # e.g., "what is my email" should still extract even if query has "system" elsewhere
        _simple_factual = ["my name", "who am i", "email", "phone", "number", "university", "college"]
        if not any(sf in q for sf in _simple_factual):
            return ""  # Skip extraction — let LLM handle synthesis queries

    # ── Detect pure greetings (no factual question after prefix removal) ──
    q_clean = q.rstrip("?!.,")
    greeting_only = q_clean in ["", "how are you", "how are you doing",
                          "what's up", "whats up", "sup",
                          "good morning", "good evening", "good afternoon",
                          "how's it going", "hows it going", "hey there",
                          "hi there", "hello there", "yo", "howdy",
                          "hi", "hey", "hello", "hii", "hiii",
                          "thanks", "thank you", "bye", "goodbye",
                          "good night", "hey there", "yo"]
    if greeting_only:
        return "Hey! I'm doing great, thanks for asking! 😊 How can I help you today?"

    all_ev = "\n".join(evidence_texts) if evidence_texts else ""
    if not all_ev:
        return ""

    # ── Combined personal info extraction ──
    # Detect which facts the user is asking for, extract all, and combine.
    wants_name = any(w in q for w in ["my name", "who am i", "full name", "what's my name", "whats my name"])
    wants_email = any(w in q for w in ["email", "e-mail", "mail address", "gmail"])
    wants_phone = any(w in q for w in ["phone", "number", "contact number", "mobile"])

    extracted_parts = []

    # ── Name extraction ──
    if wants_name:
        name = None
        # Try bold pattern
        m = re.search(r'\*\*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\*\*', all_ev)
        if m:
            name = m.group(1)
        if not name:
            m = re.search(r'[Mm]y name is ([A-Z][a-z]+ [A-Z][a-z]+)', all_ev)
            if m:
                name = m.group(1)
        if not name:
            m = re.search(r'[Nn]ame[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', all_ev)
            if m:
                name = m.group(1)
        if not name:
            # Source doc pattern: [Source: ...] Name Name
            m = re.search(r'\[Source:[^\]]*\]\s*([A-Z][a-z]+ [A-Z][a-z]+)', all_ev)
            if m:
                name = m.group(1)
        if not name:
            m = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', all_ev)
            if m:
                name = m.group(1)
        if name:
            extracted_parts.append(f"Your name is **{name}**")

    # ── Email extraction ──
    if wants_email:
        m = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', all_ev)
        if m:
            email = m.group(0).rstrip('.')
            extracted_parts.append(f"your email is **{email}**")

    # ── Phone extraction (for combined queries) ──
    if wants_phone:
        # Prefer +country-code format, then 10+ digit sequences
        m = re.search(r'\+\d{1,3}[\s-]?\d[\d\s-]{8,14}\d', all_ev)
        if not m:
            # Fallback: 10+ consecutive digits possibly with spaces/hyphens
            m = re.search(r'(?<!\d)\d{10,13}(?!\d)', all_ev.replace(' ', '').replace('-', ''))
        if m:
            extracted_parts.append(f"your phone number is **{m.group(0).strip()}**")

    # If we extracted any facts, return combined answer
    if extracted_parts:
        if len(extracted_parts) == 1:
            return extracted_parts[0] + "!"
        # Capitalize first part, join with 'and'
        first = extracted_parts[0]
        rest = " and ".join(extracted_parts[1:])
        return f"{first}, and {rest}."

    # Single-intent fallback for name-only (in case no name found above)
    if wants_name and not wants_email and not wants_phone:
        return ""
    if wants_email:
        return ""
    if wants_phone:
        return ""

    # ── Phone ──
    if any(w in q for w in ["phone", "number", "contact number", "mobile"]):
        m = re.search(r'\+\d{1,3}[\s-]?\d[\d\s-]{8,14}\d', all_ev)
        if m:
            return f"Your phone number is **{m.group(0).strip()}**."

    # ── University / Education / Studying ──
    if any(w in q for w in ["university", "college", "where do i study",
                             "where am i studying", "my institution",
                             "education", "my degree", "studying", "b.tech", "btech",
                             "education background"]):
        # Try to find degree + institution combo
        # Note: "Master" alone matches "Master-Resume.md" so require "Master's" or "Masters"
        degree_match = re.search(
            r'(B\.?Tech|M\.?Tech|Btech|B\.?Sc|M\.?Sc|MBA|Ph\.?D|Bachelor|Master(?:\'?s))[^\n]{0,120}',
            all_ev, re.IGNORECASE
        )
        uni_match = re.search(
            r'((?:Indian\s+)?(?:Institute|University|College)\s+of\s+[^\n,|]{5,60}|IIIT\s+[A-Za-z]+|IIT\s+[A-Za-z]+|NIT\s+[A-Za-z]+)',
            all_ev, re.IGNORECASE
        )
        # Also try to find IIIT/IIT/NIT patterns
        if not uni_match:
            uni_match = re.search(r'(IIIT|IIT|NIT|BITS)\s+[A-Z][a-z]+', all_ev)

        parts = []
        if degree_match:
            deg = degree_match.group(0).strip().rstrip(',|*')
            parts.append(f"pursuing **{deg}**")
        if uni_match:
            uni = uni_match.group(0).strip().rstrip(',|*')
            parts.append(f"at **{uni}**")

        if parts:
            return "You're " + " ".join(parts) + "."

        # Broader education fallback
        edu_match = re.search(r'(?:EDUCATION|Education)[:\s]*\n?(.{20,200})', all_ev, re.IGNORECASE)
        if edu_match:
            return edu_match.group(0).strip()[:250]

    # ── Skills / Programming / Tech Stack ──
    if any(w in q for w in ["skill", "language", "programming", "tech stack",
                             "technologies", "what do i know", "coding", "frameworks"]):
        prog_langs = ["python", "java", "javascript", "typescript", "c++", "c#",
                      "go", "rust", "sql", "ruby", "swift", "kotlin", " c,", " r,"]
        # Find the best evidence chunk with actual skill/language names
        best_skills = ""
        best_count = 0
        for ev in evidence_texts:
            ev_lower = ev.lower()
            count = sum(1 for lang in prog_langs if lang in ev_lower)
            if count > best_count:
                best_count = count
                # Try to extract the skills section
                skills_match = re.search(
                    r'(?:\*?\*?Skills?\*?\*?|\*?\*?Programming\*?\*?|\*?\*?Technical\*?\*?|\*?\*?Specialized\*?\*?)[:\s|*]*(.{20,400})',
                    ev, re.IGNORECASE
                )
                if skills_match:
                    best_skills = skills_match.group(0).strip()[:400]
                elif count >= 2:
                    best_skills = ev.strip()[:400]
        if best_skills:
            return f"Here are your technical skills:\n\n{_fix_person_pronouns(best_skills)}"

    # ── Projects ──
    if any(w in q for w in ["project", "built", "portfolio", "developed", "my work", "created"]):
        # For filtered/exploratory queries (e.g., "projects related to deep learning"),
        # skip extraction and let the LLM do proper filtering + description
        _filter_words = ["related to", "involving", "about", "using", "with",
                         "in the field", "regarding", "deep learning", "machine learning",
                         "ai ", "web", "mobile", "data", "all project", "list all",
                         "describe", "explain", "detail"]
        if any(fw in q for fw in _filter_words):
            return ""  # Let the LLM generate a proper filtered response
        projects = []
        # Find project names from evidence
        for ev in evidence_texts:
            # "📌 Project Name: X" pattern
            found = re.findall(r'📌\s*Project\s*Name:\s*([^\n.]{3,80})', ev)
            projects.extend(found)
            # Bold project titles
            bold = re.findall(r'\*\*([A-Z][^*\n]{4,60})\*\*', ev)
            for b in bold:
                b_lower = b.lower()
                skip = ["name", "email", "phone", "skills", "education", "source",
                        "university", "experience", "summary", "contact"]
                if not any(s in b_lower for s in skip) and len(b) > 6:
                    projects.append(b.strip())
        # Deduplicate
        seen = set()
        unique = []
        for p in projects:
            p_clean = p.strip().rstrip('*#').strip()
            if p_clean.lower() not in seen and len(p_clean) >= 4:
                seen.add(p_clean.lower())
                unique.append(p_clean)
        if unique:
            project_list = "\n".join(f"• **{p}**" for p in unique[:10])
            return f"Here are the projects you've built:\n\n{project_list}"

    # ── Location / Hometown ──
    if any(w in q for w in ["where do i live", "my location", "my city", "my hometown",
                             "where am i from", "my address", "where i stay"]):
        # Try specific city names first
        loc_match = re.search(
            r'(Patna|Bihar|Bangalore|Karnataka|Mumbai|Delhi|Hyderabad|Chennai|Kolkata)[,\s]*(?:[A-Z][a-z]+)?(?:[,\s]*India)?',
            all_ev
        )
        if not loc_match:
            loc_match = re.search(
                r'(?:from|located in|lives in|hometown)[:\s]+([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)*)',
                all_ev, re.IGNORECASE
            )
        if loc_match:
            location = loc_match.group(0).strip()
            # Remove "from " prefix if captured
            if location.lower().startswith("from "):
                location = location[5:]
            return f"You're from **{location}**."

    return ""  # Don't extract — let the model generate


def _build_prompt(messages: list[ChatMessage]) -> str:
    """
    Build a prompt using the tokenizer's chat template when available.
    Falls back to a structured format with system prompt and stop boundaries.
    """
    # Try to use the model's native chat template (best for DeepSeek-R1)
    if tokenizer is not None:
        try:
            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in messages:
                chat_messages.append({"role": m.role, "content": m.content})
            prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            return prompt
        except Exception:
            pass  # Fall back to manual format

    # Fallback: structured prompt with clear boundaries
    parts: list[str] = [f"System: {SYSTEM_PROMPT}"]
    for m in messages:
        if m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _truncate_at_stop_patterns(text: str) -> str:
    """
    Truncate generated text at the first occurrence of any stop pattern.
    This prevents the model from hallucinating new conversation turns.
    """
    earliest_pos = len(text)
    for pattern in _STOP_PATTERNS:
        pos = text.find(pattern)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    return text[:earliest_pos].strip()


def _split_thinking(text: str):
    """
    Separate <think>…</think> reasoning from visible answer.
    Works with both raw special-token output and clean text.
    DeepSeek-R1 format: generation starts with <think>\n...reasoning...</think>answer
    """
    thinking = None
    content  = text

    # The generation prompt already contains <think>\n so output starts with thinking content
    if "<think>" in text:
        start = text.index("<think>") + len("<think>")
        if "</think>" in text:
            end = text.index("</think>")
            thinking = text[start:end].strip()
            content  = text[end + len("</think>"):].strip()
        else:
            # Model never closed the think tag — everything is thinking, no content
            thinking = text[start:].strip()
            content  = ""
    elif "</think>" in text:
        # Generation started inside <think> (prompt already had <think>\n)
        end = text.index("</think>")
        thinking = text[:end].strip()
        content  = text[end + len("</think>"):].strip()

    # Truncate hallucinated continuations from the visible content
    if content:
        content = _truncate_at_stop_patterns(content)
    return thinking, content


# ── LLM Provider Helpers ─────────────────────────────────────────────────────

def _set_request_provider(provider: str):
    """Set the LLM provider for this request. Raises ValueError if the
    requested provider is genuinely unavailable (no silent fallback)."""
    if rag_engine.initialized and hasattr(rag_engine.llm, "set_provider"):
        local_available = (rag_engine.llm.local_llm is not None
                           and rag_engine.llm.local_llm.model is not None)
        gemini_available = rag_engine.llm.has_gemini

        if provider == "local" and not local_available:
            raise ValueError("local_unavailable")

        if provider == "gemini" and not gemini_available:
            raise ValueError("gemini_unavailable")

        rag_engine.llm.set_provider(provider)


def _is_gemini_active() -> bool:
    """Check if Gemini is the currently active LLM provider."""
    return (
        rag_engine.initialized
        and hasattr(rag_engine.llm, "provider")
        and rag_engine.llm.provider == "gemini"
        and rag_engine.llm.has_gemini
    )


async def _stream_gemini_generate(prompt: str, req):
    """Stream Gemini API response as SSE events (for /api/chat Gemini path)."""
    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    gemini = rag_engine.llm.gemini_llm

    try:
        for chunk_text in gemini.generate_stream(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        ):
            if chunk_text:
                yield f"data: {json.dumps({'id': msg_id, 'delta': chunk_text})}\n\n"
                await asyncio.sleep(0)
    except Exception as e:
        print(f"  ⚠ Gemini stream error: {e}")

    yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"


async def _stream_gemini_rag_generate(
    user_message: str, history: list, req, evidence_texts: list,
    has_meaningful_evidence: bool, has_pageindex_evidence: bool,
    is_document_query: bool, msg_id: str, is_synthesis: bool = False
):
    """Stream Gemini API response with RAG evidence context."""
    gemini = rag_engine.llm.gemini_llm

    evidence_block = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence_texts))

    if has_meaningful_evidence:
        if has_pageindex_evidence and is_document_query:
            system = (
                "You are Cortex Lab, an AI assistant with access to the user's uploaded documents. "
                "Answer ONLY based on the document content provided. Be thorough and detailed. "
                "NEVER make up information. NEVER add citations like [1] [2]."
            )
            context_label = "Relevant document content"
        elif is_synthesis:
            system = (
                "You are Cortex Lab, an intelligent personal AI assistant who deeply understands the user. "
                "The user is asking a synthesis question that requires a comprehensive, thoughtful answer. "
                "Write a THOROUGH, multi-paragraph response that covers ALL relevant aspects from the evidence. "
                "Weave together themes, ideas, and details into a cohesive narrative. "
                "Be specific — reference concrete projects, ideas, writings, and experiences. "
                "Connect different pieces of evidence to paint a complete picture. "
                "Write at least 3-5 paragraphs for complex questions about vision, philosophy, or worldview. "
                "Speak warmly and conversationally. Use 'you/your' when referring to the user. "
                "NEVER truncate your answer — finish every thought completely. "
                "NEVER say 'Based on stored memories' or cite evidence numbers. "
                "NEVER generate labels like 'Confidence:', 'Evidence:', 'Answer:'."
            )
            context_label = "Here is what I know about you"
        else:
            system = (
                "You are Cortex Lab, an intelligent personal AI assistant who knows the user well. "
                "Use the user's stored memories below to answer naturally. "
                "Speak warmly and conversationally. Give direct, confident answers. "
                "Use 'you/your' when referring to the user. "
                "NEVER say 'Based on stored memories' or cite evidence numbers. "
                "If the evidence doesn't answer the question, say so honestly."
            )
            context_label = "What I know about you"
    else:
        system = (
            "You are Cortex Lab, a friendly personal AI assistant. "
            "Respond naturally and briefly. Be cheerful and helpful."
        )
        context_label = None

    if context_label and evidence_block:
        prompt = f"{system}\n\nUser: {user_message}\n\n{context_label}:\n{evidence_block}\n\nAssistant:"
    else:
        prompt = f"{system}\n\nUser: {user_message}\n\nAssistant:"

    try:
        for chunk_text in gemini.generate_stream(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        ):
            if chunk_text:
                yield f"data: {json.dumps({'id': msg_id, 'delta': chunk_text})}\n\n"
                await asyncio.sleep(0)
    except Exception as e:
        print(f"  ⚠ Gemini RAG stream error: {e}")

    yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    ambient_status = None
    if rag_engine.ambient_service:
        try:
            ambient_status = rag_engine.ambient_service.get_status()
        except Exception:
            pass
    return HealthResponse(
        status="ok" if model_loaded else "loading",
        model_loaded=model_loaded,
        model_info={
            **model_info,
            "llm_provider": rag_engine.llm.provider if rag_engine.initialized and hasattr(rag_engine.llm, "provider") else "local",
            "gemini_available": rag_engine.llm.has_gemini if rag_engine.initialized and hasattr(rag_engine.llm, "has_gemini") else False,
        },
    )


@app.get("/api/system/gpu")
async def gpu_status():
    """GPU memory monitoring endpoint (§6.2)."""
    if torch is None or not torch.cuda.is_available():
        return {"gpu_available": False}
    try:
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        max_allocated = torch.cuda.max_memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_mem
        utilization = allocated / total if total > 0 else 0
        return {
            "gpu_available": True,
            "device_name": torch.cuda.get_device_properties(0).name,
            "total_mb": round(total / 1e6, 1),
            "allocated_mb": round(allocated / 1e6, 1),
            "reserved_mb": round(reserved / 1e6, 1),
            "max_allocated_mb": round(max_allocated / 1e6, 1),
            "free_mb": round((total - allocated) / 1e6, 1),
            "utilization_pct": round(utilization * 100, 1),
        }
    except Exception as e:
        return {"gpu_available": True, "error": str(e)}


# ── LLM Provider Toggle ─────────────────────────────────────────────────────

@app.get("/api/llm/provider")
async def get_llm_provider():
    """Get current LLM provider and available providers."""
    has_gemini = (
        rag_engine.initialized
        and hasattr(rag_engine, "llm")
        and hasattr(rag_engine.llm, "has_gemini")
        and rag_engine.llm.has_gemini
    )
    local_available = (
        rag_engine.initialized
        and rag_engine.llm.local_llm is not None
        and rag_engine.llm.local_llm.model is not None
    )
    current = "local"
    if rag_engine.initialized and hasattr(rag_engine.llm, "provider"):
        current = rag_engine.llm.provider
    available = []
    if local_available:
        available.append("local")
    if has_gemini:
        available.append("gemini")
    return {
        "provider": current,
        "available": available,
        "gemini_configured": has_gemini,
        "local_model_loaded": local_available,
    }


@app.post("/api/llm/provider")
async def set_llm_provider(body: dict):
    """Switch the active LLM provider ('local' or 'gemini'). Strict isolation."""
    provider = body.get("provider", "local")
    if provider not in ("local", "gemini"):
        raise HTTPException(400, "Provider must be 'local' or 'gemini'")
    if provider == "gemini":
        if not (rag_engine.initialized and rag_engine.llm.has_gemini):
            raise HTTPException(400, "Gemini is not configured. Set GOOGLE_API_KEY in backend/.env")
    if provider == "local":
        local_available = (rag_engine.initialized
                           and rag_engine.llm.local_llm is not None
                           and rag_engine.llm.local_llm.model is not None)
        if not local_available:
            raise HTTPException(400, "Local model is not loaded. Start with SKIP_LOCAL_MODEL=false or load the model first.")
    rag_engine.llm.set_provider(provider)
    return {"provider": provider, "status": "switched"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Apply per-request LLM provider switch
    try:
        _set_request_provider(req.llm_provider)
    except ValueError as e:
        if "local_unavailable" in str(e):
            raise HTTPException(
                503,
                "Local model is not loaded. Switch to Gemini in settings, or start the server with a local model."
            )
        raise HTTPException(503, f"LLM provider '{req.llm_provider}' is not available.")

    # ── Gemini path (no local model needed) ──────────────────────────────
    if _is_gemini_active():
        gemini = rag_engine.llm.gemini_llm
        # Build a clean prompt (no ChatML tokens)
        system = (
            "You are Cortex Lab, a personal AI memory and reasoning assistant. "
            "You help the user by answering their questions thoughtfully and concisely. "
            "Never fabricate personal details about the user. "
            "Keep responses focused and do NOT generate follow-up questions."
        )
        user_text = req.messages[-1].content if req.messages else ""
        history_text = ""
        for m in req.messages[:-1]:
            history_text += f"{m.role.capitalize()}: {m.content}\n"
        full_prompt = f"{system}\n\n{history_text}User: {user_text}\nAssistant:"

        if req.stream:
            return StreamingResponse(
                _stream_gemini_generate(full_prompt, req),
                media_type="text/event-stream",
            )
        content = gemini.generate(
            full_prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return ChatResponse(
            id=f"msg-{uuid.uuid4().hex[:12]}",
            model="gemini-2.5-flash",
            created=int(time.time()),
            content=content or "I'm not sure how to respond to that.",
            thinking=None,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # ── Local model path ─────────────────────────────────────────────────
    prompt = _build_prompt(req.messages)

    # ── Streaming ────────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_generate(prompt, req),
            media_type="text/event-stream",
        )

    # ── Non-streaming ────────────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    # Build stop token IDs to prevent runaway generation
    stop_token_ids = [tokenizer.eos_token_id]
    # Try to add common stop tokens
    for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
        try:
            ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[0])
        except Exception:
            pass

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=min(req.max_tokens, 8192),
            temperature=max(req.temperature, 0.01),
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
            repetition_penalty=1.15,
        )

    # Decode with special tokens to extract <think>...</think>
    raw_output = tokenizer.decode(out[0][input_len:], skip_special_tokens=False).strip()
    thinking, content = _split_thinking(raw_output)

    # Clean up special tokens from content
    if content:
        # Remove any remaining special tokens
        for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "<｜User｜>", "<｜Assistant｜>"]:
            content = content.replace(tok, "")
        content = _truncate_at_stop_patterns(content.strip())
    if thinking:
        for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "<｜User｜>", "<｜Assistant｜>"]:
            thinking = thinking.replace(tok, "")
        thinking = thinking.strip()

    return ChatResponse(
        id=f"msg-{uuid.uuid4().hex[:12]}",
        model=model_info.get("name", MODEL_NAME),
        created=int(time.time()),
        content=content or "I'm not sure how to respond to that.",
        thinking=thinking,
        usage={
            "prompt_tokens": input_len,
            "completion_tokens": out.shape[-1] - input_len,
            "total_tokens": out.shape[-1],
        },
    )


async def _stream_generate(prompt: str, req: ChatRequest):
    """Yield Server-Sent Events token by token with stop-pattern detection.
    Suppresses <think>…</think> blocks so only the visible answer is streamed."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Use skip_special_tokens=False so we can detect <think> tags ourselves
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    # Build stop token IDs
    stop_token_ids = [tokenizer.eos_token_id]
    for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
        try:
            ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[0])
        except Exception:
            pass

    gen_kwargs = {
        **inputs,
        "max_new_tokens": min(req.max_tokens, 8192),
        "temperature": max(req.temperature, 0.01),
        "top_p": req.top_p,
        "do_sample": req.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": stop_token_ids,
        "repetition_penalty": 1.15,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    msg_id = f"msg-{uuid.uuid4().hex[:12]}"
    accumulated = ""  # Track full text to detect stop patterns mid-stream
    in_think = False  # Track <think> block to suppress from stream
    thinking_text = ""  # Collect thinking for optional later use

    for token_text in streamer:
        accumulated += token_text

        # ── Handle <think>…</think> blocks — suppress from visible output ──
        if "<think>" in accumulated and not in_think:
            in_think = True
        if in_think:
            if "</think>" in accumulated:
                # Think block complete — extract and continue with answer
                think_end = accumulated.index("</think>") + len("</think>")
                thinking_text = accumulated[:think_end]
                accumulated = accumulated[think_end:]
                in_think = False
                # If there's text after </think>, process it below
                if not accumulated:
                    continue
            else:
                continue  # Don't stream thinking tokens

        # ── Clean special tokens from the visible output ──
        clean_token = token_text
        for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>",
                     "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"]:
            clean_token = clean_token.replace(tok, "")
        if not clean_token:
            continue

        # ── Check for stop patterns ──
        should_stop = False
        for pattern in _STOP_PATTERNS:
            if pattern in accumulated:
                # Send only the part before the stop pattern
                safe_part = accumulated[:accumulated.index(pattern)]
                leftover = safe_part[len(accumulated) - len(token_text):]
                if leftover:
                    chunk = {"id": msg_id, "delta": leftover}
                    yield f"data: {json.dumps(chunk)}\n\n"
                should_stop = True
                break
        if should_stop:
            break
        chunk = {"id": msg_id, "delta": clean_token}
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
    thread.join()


# ── RAG-Enhanced Chat ────────────────────────────────────────────────────────

@app.post("/api/rag/chat")
async def rag_chat(req: RAGChatRequest):
    """RAG-enhanced chat: uses memory retrieval + multi-agent reasoning.
    Supports both streaming and non-streaming modes."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine is still initializing.")

    # Apply per-request LLM provider switch
    try:
        _set_request_provider(req.llm_provider)
    except ValueError as e:
        if "local_unavailable" in str(e):
            raise HTTPException(
                503,
                "Local model is not loaded. Switch to Gemini in settings, or start the server with a local model."
            )
        raise HTTPException(503, f"LLM provider '{req.llm_provider}' is not available.")

    user_message = req.messages[-1].content if req.messages else ""
    if not user_message:
        raise HTTPException(400, "No message provided.")

    history = [{"role": m.role, "content": m.content} for m in req.messages[:-1]]

    # ── Streaming RAG ────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_rag_generate(user_message, history, req),
            media_type="text/event-stream",
        )

    # ── Non-streaming RAG (with concurrency limit + timeout) ─────────────
    async with _inference_semaphore:
        try:
            result = await asyncio.wait_for(
                rag_engine.rag_chat(
                    user_message=user_message,
                    session_id=req.session_id,
                    conversation_history=history,
                ),
                timeout=_REQUEST_TIMEOUT,
            )

            # Store trace for observability history
            _store_trace(result.get("pipeline_trace"))

            return {
                "id": f"rag-{uuid.uuid4().hex[:12]}",
                "model": model_info.get("name", MODEL_NAME),
                "created": int(time.time()),
                "content": result.get("answer", ""),
                "thinking": result.get("thinking", ""),
                "evidence": result.get("evidence", []),
                "agents_used": result.get("agents_used", []),
                "confidence": result.get("confidence", 0),
                "query_analysis": result.get("query_analysis", {}),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "cache_hit": result.get("cache_hit", False),
                "pipeline_trace": result.get("pipeline_trace", None),
            }
        except asyncio.TimeoutError:
            raise HTTPException(504, "Request timed out after 180 seconds")
        except Exception as e:
            print(f"  ❌ RAG error: {e}")
            traceback.print_exc()
            raise HTTPException(500, f"RAG processing error: {str(e)}")


async def _stream_rag_generate(user_message: str, history: list, req: RAGChatRequest):
    """
    Stream RAG-enhanced chat.
    1. Run RAG pipeline to get evidence + thinking (non-streamed)
    2. Stream the final answer generation token by token with evidence context
    """
    msg_id = f"rag-{uuid.uuid4().hex[:12]}"

    try:
        # Step 1: Run RAG pipeline for evidence retrieval (fast, no generation)
        rag_result = await rag_engine.rag_retrieve(
            user_message=user_message,
            session_id=req.session_id,
            conversation_history=history,
        )

        evidence = rag_result.get("evidence", [])
        agents_used = rag_result.get("agents_used", [])
        confidence = rag_result.get("confidence", 0)
        query_analysis = rag_result.get("query_analysis", {})
        thinking = rag_result.get("thinking", "")

        # Send metadata first
        meta_chunk = {
            "id": msg_id,
            "delta": "",
            "rag_meta": {
                "evidence": evidence,
                "agents_used": agents_used,
                "confidence": confidence,
                "query_analysis": query_analysis,
                "thinking": thinking,
                "pipeline_trace": rag_result.get("pipeline_trace", None),
            }
        }
        yield f"data: {json.dumps(meta_chunk)}\n\n"

        # Store trace for observability history
        _store_trace(rag_result.get("pipeline_trace"))

        # Step 2: Build prompt with evidence context for streaming generation

        # ── Supplement: Direct memory search for personal info queries ──
        # The hybrid retriever's RRF fusion can miss high-quality results.
        # For simple personal-info queries, do a direct vector search and
        # inject any highly-relevant results into the evidence list.
        _personal_triggers = [
            "my name", "who am i", "my email", "e-mail", "my phone",
            "my number", "contact", "my education", "where do i study",
            "my university", "my college", "my skills", "my resume",
            "my experience", "my projects", "what do i do",
            "my location", "my hometown", "my address", "about me",
            "my degree", "my profile", "my background",
        ]
        _q_for_supplement = user_message.lower()
        if any(t in _q_for_supplement for t in _personal_triggers):
            try:
                # Use augmented search terms that match resume/personal data content
                # rather than the raw question (which has low semantic similarity).
                # All personal-info queries benefit from "resume contact summary"
                # base terms since personal data lives in the resume memories.
                _base = "resume contact information summary"
                _search_augments = {
                    "my name": f"{_base} name email phone B.Tech",
                    "who am i": f"{_base} name email phone B.Tech",
                    "my email": f"{_base} name email phone B.Tech",
                    "e-mail": f"{_base} name email phone B.Tech",
                    "my phone": f"{_base} name email phone B.Tech",
                    "my number": f"{_base} name email phone B.Tech",
                    "contact": f"{_base} name email phone B.Tech",
                    "my education": f"{_base} education university degree B.Tech",
                    "where do i study": f"{_base} education university degree B.Tech",
                    "my university": f"{_base} education university degree B.Tech",
                    "my college": f"{_base} education university degree B.Tech",
                    "my skills": f"{_base} skills programming technical tools frameworks",
                    "my resume": f"{_base} name email phone education skills B.Tech",
                    "my experience": f"{_base} experience internship work projects",
                    "my projects": f"{_base} projects portfolio built developed",
                    "about me": f"{_base} name email phone education skills B.Tech",
                    "my degree": f"{_base} education university degree B.Tech",
                    "my profile": f"{_base} name email phone education skills B.Tech",
                    "my background": f"{_base} education experience skills B.Tech",
                }
                # Pick the best augmented query based on which triggers matched
                matched_aug = None
                for trigger, aug in _search_augments.items():
                    if trigger in _q_for_supplement:
                        matched_aug = aug
                        break
                augmented_query = matched_aug or f"{_base} name email phone B.Tech"

                supplement = rag_engine.search_memories(augmented_query, top_k=3)
                existing_ids = {e.get("content", "")[:50] for e in evidence}
                for mem in supplement:
                    score = mem.get("score", 0)
                    if score > 0.45:
                        preview = mem.get("content", "")[:50]
                        if preview not in existing_ids:
                            evidence.insert(0, {
                                "content": mem.get("content", "")[:600],
                                "score": score,
                                "channel": "direct_supplement",
                                "memory_type": mem.get("memory_type", "semantic"),
                            })
            except Exception as exc:
                pass  # Supplement is best-effort

        # PageIndex evidence gets priority — document content should always appear
        # when the user is asking about uploaded documents
        evidence_texts = []
        has_pageindex_evidence = False
        pageindex_evidence_count = 0

        # ── Query-adaptive evidence limits ──
        # Complex/synthesis queries need more context for comprehensive answers.
        _query_complexity = query_analysis.get("complexity", 0.5) if isinstance(query_analysis, dict) else 0.5
        _query_intent = query_analysis.get("intent", "") if isinstance(query_analysis, dict) else ""
        _is_synthesis = _query_complexity >= 0.6 or _query_intent in ("reflective", "comparative", "causal")
        _max_evidence_chars = 1500 if _is_synthesis else 500
        _max_evidence_items = 12 if _is_synthesis else 7

        # First pass: collect PageIndex evidence (from uploaded documents)
        for e in evidence[:20]:
            channel = e.get("channel", "")
            if "pageindex" in channel:
                content = e.get("content", "").strip()
                if len(content) >= 20:
                    evidence_texts.append(content[:2000])  # Full doc answer
                    has_pageindex_evidence = True
                    pageindex_evidence_count += 1
                    if pageindex_evidence_count >= 3:  # Max 3 PageIndex chunks
                        break

        # Second pass: fill remaining slots with local memory evidence
        for e in evidence[:16]:
            channel = e.get("channel", "")
            if "pageindex" in channel:
                continue  # Already handled above
            content = e.get("content", "").strip()
            lower = content.lower()
            # Skip short question-like evidence
            if len(content) < 50:
                continue
            if (lower.endswith("?") and len(content) < 120
                    and not any(k in lower for k in ["[source:", "**", "project", "built"])):
                continue
            # Skip repetitive/spam content
            words = lower.split()
            if len(words) > 10:
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                from collections import Counter
                trigram_counts = Counter(trigrams)
                if trigram_counts and max(trigram_counts.values()) > 3:
                    continue
            # Skip stored user queries (but allow long source docs)
            if re.match(r'^(tell me|what is|what are|who is|where is|how is|list|describe|explain)\b', lower):
                if len(content) < 200 and "[source:" not in lower:
                    continue
            evidence_texts.append(content[:_max_evidence_chars])
            if len(evidence_texts) >= _max_evidence_items:
                break
        evidence_block = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence_texts))

        # Check if evidence is actually meaningful (not just greetings)
        has_meaningful_evidence = any(
            len(e.strip()) > 10 and e.strip().lower() not in {"hi", "hello", "hey", "hii", "hiii"}
            for e in evidence_texts
        )

        # ── Detect casual / greeting queries — force no-evidence path ──
        _q_lower = user_message.lower().strip().rstrip("?!.")
        _casual_queries = {
            "hi", "hello", "hey", "hii", "hiii", "yo", "sup", "howdy",
            "hey there", "hi there", "hello there",
            "how are you", "how are you doing", "how's it going",
            "hows it going", "what's up", "whats up",
            "good morning", "good evening", "good afternoon",
            "good night", "thanks", "thank you", "bye", "goodbye",
        }
        if _q_lower in _casual_queries:
            has_meaningful_evidence = False

        # ── Document intent detection ──
        # Determine if the user is asking about uploaded documents vs personal data
        _doc_keywords = [
            "document", "pdf", "uploaded", "file", "paper",
            "report", "article", "thesis", "book", "chapter",
            "page", "section", "paragraph", "content of",
            "in the document", "from the document", "in the pdf",
            "from the pdf", "uploaded file", "my document",
        ]
        _q_doc = user_message.lower()
        is_document_query = any(kw in _q_doc for kw in _doc_keywords)

        # ── PageIndex Direct Answer Path ──
        # When PageIndex evidence is present AND the query is about documents,
        # bypass the local LLM entirely and return PageIndex's cloud LLM answer.
        if has_pageindex_evidence and has_meaningful_evidence and is_document_query:
            pi_answer = evidence_texts[0] if evidence_texts else ""
            if pi_answer and len(pi_answer.strip()) > 20:
                formatted = f"From your uploaded documents:\n\n{pi_answer}"
                chunk = {"id": msg_id, "delta": formatted}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
                return

        # ── For non-document queries, prefer local memories but keep PageIndex as fallback ──
        # Only strip PageIndex evidence if there's enough non-PageIndex evidence to use.
        # If PageIndex is the only/best evidence (e.g., user asked about projects
        # that happen to be in an uploaded doc), keep it.
        if has_pageindex_evidence and not is_document_query:
            non_pi_evidence = evidence_texts[pageindex_evidence_count:]
            if non_pi_evidence and any(len(e.strip()) > 30 for e in non_pi_evidence):
                # Good local evidence exists — prefer it over PageIndex
                evidence_texts = non_pi_evidence
                has_pageindex_evidence = False
            # else: keep PageIndex evidence — it's all we have and may be relevant
            has_meaningful_evidence = any(
                len(e.strip()) > 10 and e.strip().lower() not in {"hi", "hello", "hey"}
                for e in evidence_texts
            )
            evidence_block = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence_texts))

        # ── Pre-generation: Check for false premises / no-data queries ──
        no_info_response = _check_no_info_streaming(user_message, evidence_texts)
        if no_info_response:
            chunk = {"id": msg_id, "delta": no_info_response}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
            return

        # ── Pre-generation: For simple factual queries, try direct extraction ──
        extracted_answer = _try_extract_factual(user_message, evidence_texts)
        if extracted_answer:
            chunk = {"id": msg_id, "delta": extracted_answer}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
            return

        if has_meaningful_evidence:
            if has_pageindex_evidence:
                # Document-aware prompt — when PageIndex evidence is present,
                # the user is asking about their uploaded documents
                rag_prompt = PromptBuilder.pageindex_generation(user_message, evidence_block)
            elif _is_synthesis:
                # Complex synthesis/vision/philosophical query needs comprehensive prompt
                rag_prompt = PromptBuilder.synthesis_rag_generation(user_message, evidence_block)
            else:
                rag_prompt = PromptBuilder.streaming_rag_generation(user_message, evidence_block)
        else:
            # Greeting / casual conversation — no evidence needed
            rag_prompt = PromptBuilder.greeting_response(user_message)

        # ── Gemini streaming path — bypass local model entirely ──────────
        if _is_gemini_active():
            async for event in _stream_gemini_rag_generate(
                user_message, history, req, evidence_texts,
                has_meaningful_evidence, has_pageindex_evidence,
                is_document_query, msg_id, _is_synthesis,
            ):
                yield event
            return

        inputs = tokenizer(rag_prompt, return_tensors="pt", truncation=True, max_length=4096)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        stop_token_ids = [tokenizer.eos_token_id]
        for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = tokenizer.encode(stop_str, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids[0])
            except Exception:
                pass

        gen_kwargs = {
            **inputs,
            "max_new_tokens": min(req.max_tokens, 8192),
            "temperature": max(req.temperature, 0.01),
            "top_p": req.top_p,
            "do_sample": req.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": stop_token_ids,
            "repetition_penalty": 1.15,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        streamed_text = ""  # Track what we've actually sent to client
        in_think = False  # Track <think> block to suppress from stream
        halluc_detected = False  # Track if hallucination was detected mid-stream
        for token_text in streamer:
            accumulated += token_text

            # Handle <think>...</think> blocks — don't stream them
            if "<think>" in accumulated and not in_think:
                in_think = True
            if in_think:
                if "</think>" in accumulated:
                    # Think block complete — extract thinking and continue streaming the rest
                    think_end = accumulated.index("</think>") + len("</think>")
                    accumulated = accumulated[think_end:]
                    in_think = False
                continue  # Don't stream thinking tokens

            # Clean special tokens from the visible output
            clean_token = token_text
            for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>",
                         "<｜User｜>", "<｜Assistant｜>"]:
                clean_token = clean_token.replace(tok, "")
            if not clean_token:
                continue

            # ── Hallucination detection in streaming ──
            # Check accumulated text for known hallucination patterns
            acc_lower = accumulated.lower()
            for halluc_phrase in _STREAMING_HALLUC_PHRASES:
                if halluc_phrase in acc_lower:
                    halluc_detected = True
                    break
            if halluc_detected:
                break  # Stop streaming — will try extraction fallback below

            should_stop = False
            for pattern in _STOP_PATTERNS:
                if pattern in accumulated:
                    safe_part = accumulated[:accumulated.index(pattern)]
                    leftover = safe_part[len(accumulated) - len(token_text):]
                    if leftover:
                        chunk = {"id": msg_id, "delta": leftover}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        streamed_text += leftover
                    should_stop = True
                    break
            if should_stop:
                break
            chunk = {"id": msg_id, "delta": clean_token}
            yield f"data: {json.dumps(chunk)}\n\n"
            streamed_text += clean_token
            await asyncio.sleep(0)

        # ── Post-stream: If hallucination was detected, replace with extraction ──
        if halluc_detected and evidence_texts:
            # Try to extract a factual answer from evidence
            extracted = _try_extract_factual(user_message, evidence_texts)
            if not extracted:
                # Generic extraction: find best matching evidence
                import re as _re
                q_words = set(_re.findall(r'\b[a-z]{3,}\b', user_message.lower()))
                filler = {"what", "who", "where", "when", "how", "why", "the",
                          "and", "for", "are", "tell", "about", "your", "you",
                          "please", "give", "show", "list", "does", "did"}
                content_words = q_words - filler
                best_ev, best_score = "", 0
                for ev in evidence_texts:
                    score = sum(1 for w in content_words if w in ev.lower())
                    if score > best_score and len(ev) > 50:
                        best_score = score
                        best_ev = ev
                if best_ev:
                    extracted = best_ev[:400]
                else:
                    extracted = "I don't have specific information about that yet — feel free to tell me and I'll remember it!"

            # REPLACE any partial hallucinated text — send correction that overwrites
            # Use a special separator to signal the frontend to replace content
            if streamed_text.strip():
                # Send a "replace" signal — the frontend should clear previous text
                replace_chunk = {"id": msg_id, "delta": "", "replace": extracted}
            else:
                replace_chunk = {"id": msg_id, "delta": extracted}
            yield f"data: {json.dumps(replace_chunk)}\n\n"

        yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
        thread.join()

        # Store assistant response in conversation history
        # Clean accumulated text of any leftover special tokens
        clean_accumulated = accumulated
        for tok in ["<think>", "</think>", "<｜end▁of▁sentence｜>",
                     "<|im_end|>", "<|endoftext|>", "<｜User｜>", "<｜Assistant｜>"]:
            clean_accumulated = clean_accumulated.replace(tok, "")
        clean_accumulated = clean_accumulated.strip()
        if clean_accumulated and rag_engine and rag_engine.initialized:
            try:
                rag_engine.metadata_store.store_conversation_turn(
                    session_id=req.session_id or f"session-{int(time.time())}",
                    role="assistant",
                    content=clean_accumulated,
                )
            except Exception as store_err:
                print(f"  ⚠️ Failed to store assistant turn: {store_err}")

    except Exception as e:
        error_chunk = {"id": msg_id, "delta": f"\n\n⚠️ RAG streaming error: {str(e)}", "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ── Memory Management Endpoints ─────────────────────────────────────────────

@app.post("/api/memories/ingest")
async def ingest_memory(req: MemoryIngestRequest):
    """Manually ingest a memory into the RAG system."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    try:
        result = await rag_engine.ingest_memory(
            content=req.content, source=req.source, session_id=req.session_id
        )
        return {"status": "ok", "memory": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/memories")
async def get_memories(limit: int = Query(50, ge=1, le=500),
                       offset: int = Query(0, ge=0)):
    """Get stored memories with pagination."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    memories = rag_engine.get_memories(limit=limit, offset=offset)
    total = rag_engine.metadata_store.count_memories()
    return {"memories": memories, "total": total, "limit": limit, "offset": offset}


@app.post("/api/memories/search")
async def search_memories(req: MemorySearchRequest):
    """Search memories by semantic similarity."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    results = rag_engine.search_memories(query=req.query, top_k=req.top_k)
    return {"results": results, "count": len(results)}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    success = rag_engine.delete_memory(memory_id)
    return {"status": "ok" if success else "not_found"}


# ── Feedback & Self-Improvement (§7) ────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(body: dict):
    """Store user feedback on a RAG response for the self-improvement loop."""
    query = body.get("query", "")
    answer = body.get("answer", "")
    rating = body.get("rating")
    comment = body.get("comment", "")
    session_id = body.get("session_id", "")

    if rating is None or not isinstance(rating, int) or not (1 <= rating <= 5):
        raise HTTPException(400, "rating must be an integer 1-5")
    if not query or not answer:
        raise HTTPException(400, "query and answer are required")

    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")

    feedback_id = rag_engine.metadata_store.store_feedback(
        query=query, answer=answer, rating=rating,
        comment=comment, session_id=session_id,
    )
    return {"status": "ok", "feedback_id": feedback_id}


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get aggregate feedback statistics."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return rag_engine.metadata_store.get_feedback_stats()


@app.post("/api/memories/consolidate")
async def consolidate_memories(body: dict = None):
    """Consolidate old memories into topic-based summaries."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    if not rag_engine.ingestion:
        raise HTTPException(503, "Ingestion pipeline not available.")

    max_age = (body or {}).get("max_age_days", 180)
    count = await rag_engine.ingestion.consolidate_old_memories(max_age_days=max_age)
    return {"status": "ok", "memories_consolidated": count}


# ── Knowledge Graph Endpoints ────────────────────────────────────────────────

@app.get("/api/graph")
async def get_graph():
    """Get knowledge graph data for visualization."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return rag_engine.get_graph_data()


@app.get("/api/entities")
async def get_entities(limit: int = Query(100, ge=1, le=1000)):
    """Get all entities in the knowledge graph."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return {"entities": rag_engine.get_entities(limit=limit)}


@app.get("/api/beliefs")
async def get_belief_deltas(limit: int = Query(50, ge=1, le=200)):
    """Get detected belief evolution events."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return {"beliefs": rag_engine.get_belief_deltas(limit=limit)}


@app.get("/api/communities")
async def get_communities():
    """Get GraphRAG community summaries."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    return {"communities": rag_engine.get_community_summaries()}


# ── RAG System Stats ────────────────────────────────────────────────────────

@app.get("/api/rag/stats")
async def rag_stats():
    """Get comprehensive RAG system statistics."""
    return rag_engine.get_rag_stats()


@app.get("/api/rag/health")
async def rag_health():
    """RAG system health check."""
    return {
        "rag_initialized": rag_engine.initialized,
        "stats": rag_engine.get_rag_stats() if rag_engine.initialized else {},
    }


# ── Pipeline Trace History (Observability) ──────────────────────────────────

# In-memory ring buffer for recent pipeline traces (last 100)
_trace_history: List[dict] = []
_TRACE_MAX_HISTORY = 100

def _sanitize_floats(obj):
    """Recursively replace NaN/Inf floats with 0 so JSON serialization won't fail."""
    import math
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj

def _store_trace(trace_dict: dict):
    """Store a pipeline trace in the history ring buffer (deduplicated)."""
    if trace_dict:
        # Sanitize any NaN/Inf values that would break JSON serialization
        trace_dict = _sanitize_floats(trace_dict)
        # Deduplicate: don't store if trace_id already exists
        trace_id = trace_dict.get("trace_id", "")
        if trace_id and any(t.get("trace_id") == trace_id for t in _trace_history):
            return
        _trace_history.append(trace_dict)
        if len(_trace_history) > _TRACE_MAX_HISTORY:
            _trace_history.pop(0)


@app.get("/api/rag/traces")
async def get_pipeline_traces(limit: int = Query(20, ge=1, le=100)):
    """Get recent pipeline traces for observability dashboard."""
    traces = _trace_history[-limit:]
    traces.reverse()  # Most recent first

    # Compute aggregate analytics
    if traces:
        total_durations = [t.get("total_duration_ms", 0) for t in traces]
        avg_duration = sum(total_durations) / len(total_durations)
        avg_confidence = sum(t.get("final_confidence", 0) for t in traces) / len(traces)
        avg_evidence = sum(t.get("evidence_count", 0) for t in traces) / len(traces)

        # Channel usage breakdown across all traces
        channel_totals: dict = {}
        for t in traces:
            for ch in t.get("retrieval_channels", []):
                name = ch.get("channel", "unknown")
                if name not in channel_totals:
                    channel_totals[name] = {"total_results": 0, "total_duration_ms": 0.0, "usage_count": 0}
                channel_totals[name]["total_results"] += ch.get("result_count", 0)
                channel_totals[name]["total_duration_ms"] += ch.get("duration_ms", 0)
                channel_totals[name]["usage_count"] += 1 if ch.get("result_count", 0) > 0 else 0

        # Step frequency analysis
        step_stats: dict = {}
        for t in traces:
            for step in t.get("steps", []):
                stype = step.get("step_type", "unknown")
                if stype not in step_stats:
                    step_stats[stype] = {"completed": 0, "skipped": 0, "total_duration_ms": 0.0}
                if step.get("status") == "completed":
                    step_stats[stype]["completed"] += 1
                    step_stats[stype]["total_duration_ms"] += step.get("duration_ms", 0)
                elif step.get("status") == "skipped":
                    step_stats[stype]["skipped"] += 1

        # CRAG/Self-RAG/FLARE activation rates
        crag_activated = sum(1 for t in traces if t.get("crag_evaluation") is not None)
        selfrag_activated = sum(1 for t in traces if t.get("self_rag_critique") is not None)
        flare_activated = sum(1 for t in traces if t.get("flare_trace") is not None)
        cache_hits = sum(1 for t in traces if t.get("cache_status", {}).get("hit", False))

        analytics = {
            "total_traces": len(_trace_history),
            "showing": len(traces),
            "avg_duration_ms": round(avg_duration, 1),
            "avg_confidence": round(avg_confidence, 3),
            "avg_evidence_count": round(avg_evidence, 1),
            "channel_usage": channel_totals,
            "step_stats": step_stats,
            "crag_activation_rate": round(crag_activated / len(traces), 3) if traces else 0,
            "selfrag_activation_rate": round(selfrag_activated / len(traces), 3) if traces else 0,
            "flare_activation_rate": round(flare_activated / len(traces), 3) if traces else 0,
            "cache_hit_rate": round(cache_hits / len(traces), 3) if traces else 0,
        }
    else:
        analytics = {
            "total_traces": 0, "showing": 0,
            "avg_duration_ms": 0, "avg_confidence": 0, "avg_evidence_count": 0,
            "channel_usage": {}, "step_stats": {},
            "crag_activation_rate": 0, "selfrag_activation_rate": 0,
            "flare_activation_rate": 0, "cache_hit_rate": 0,
        }

    return {"traces": traces, "analytics": analytics}


@app.get("/api/rag/traces/{trace_id}")
async def get_pipeline_trace(trace_id: str):
    """Get a specific pipeline trace by ID."""
    for t in _trace_history:
        if t.get("trace_id") == trace_id:
            return t
    raise HTTPException(404, f"Trace {trace_id} not found")


# ── Live Pipeline Events (Real-Time SSE) ────────────────────────────────────

from src.observability import pipeline_events

@app.get("/api/rag/pipeline-events")
async def live_pipeline_events(request: Request):
    """
    SSE endpoint for real-time pipeline step events.
    Frontend subscribes here before sending a chat request to see
    each pipeline step light up in real-time as it executes.
    Uses a global broadcast — events include trace_id so clients
    can filter to their own request.
    """
    subscriber_id = f"sub-{uuid.uuid4().hex[:8]}"

    async def event_generator():
        queue = pipeline_events.subscribe_global(subscriber_id)
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            pipeline_events.unsubscribe_global(subscriber_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/rag/observability/metrics")
async def observability_metrics():
    """Get aggregate observability metrics from the pipeline event bus."""
    metrics = pipeline_events.get_metrics()

    # Add compression stats if available
    try:
        from src.compression import ContextCompressor
        comp = ContextCompressor()
        metrics["compression"] = comp.get_stats()
    except Exception:
        metrics["compression"] = {}

    # Add cache stats
    if rag_engine.initialized and rag_engine.cache:
        metrics["cache"] = rag_engine.cache.get_stats()

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  AMBIENT VOICE SERVICE — STT / TTS / Speaker ID
# ══════════════════════════════════════════════════════════════════════════════

class AmbientConfigUpdate(BaseModel):
    vad_threshold: Optional[float] = None
    auto_ingest: Optional[bool] = None
    silence_timeout_s: Optional[int] = None
    min_speech_ms: Optional[int] = None
    tts_enabled: Optional[bool] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    whisper_model_size: Optional[str] = None
    whisper_device: Optional[str] = None
    whisper_language: Optional[str] = None
    record_raw_audio: Optional[bool] = None
    stt_provider: Optional[str] = None       # "traditional" or "gemini"
    tts_provider: Optional[str] = None       # "traditional" or "gemini"
    gemini_tts_voice: Optional[str] = None   # Gemini voice name
    wake_word_enabled: Optional[bool] = None  # Enable wake word detection
    wake_word_threshold: Optional[float] = None  # Wake word confidence threshold
    wake_word_mode: Optional[str] = None     # "always_on", "manual", "hybrid"

class TTSSynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None

class VoiceQueryRequest(BaseModel):
    audio_base64: str  # Base64-encoded int16 PCM audio at 16kHz
    settings: Optional[dict] = None

class EnrollmentRequest(BaseModel):
    duration_seconds: int = Field(20, ge=10, le=30)

class SpeakerAliasRequest(BaseModel):
    speaker_label: str
    name: str


def _get_ambient():
    """Helper to get ambient service or raise 503."""
    if not rag_engine.initialized or not rag_engine.ambient_service:
        raise HTTPException(503, "Ambient service not available.")
    return rag_engine.ambient_service


# ── Ambient Lifecycle ────────────────────────────────────────────────────────

@app.post("/api/ambient/start")
async def ambient_start():
    """Start ambient listening pipeline."""
    ambient = _get_ambient()
    result = await ambient.start()
    return result


@app.post("/api/ambient/stop")
async def ambient_stop():
    """Stop ambient listening pipeline."""
    ambient = _get_ambient()
    result = await ambient.stop()
    return result


@app.post("/api/ambient/pause")
async def ambient_pause():
    """Pause ambient listening (keeps models loaded)."""
    ambient = _get_ambient()
    return await ambient.pause()


@app.post("/api/ambient/resume")
async def ambient_resume():
    """Resume ambient listening from pause."""
    ambient = _get_ambient()
    return await ambient.resume()


@app.get("/api/ambient/status")
async def ambient_status():
    """Get current ambient listening status + stats."""
    ambient = _get_ambient()
    return ambient.get_status()


# ── Ambient Configuration ────────────────────────────────────────────────────

@app.get("/api/ambient/config")
async def get_ambient_config():
    """Get current ambient configuration."""
    ambient = _get_ambient()
    return ambient.get_config()


@app.post("/api/ambient/config")
async def update_ambient_config(req: AmbientConfigUpdate):
    """Update ambient configuration."""
    ambient = _get_ambient()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    config = ambient.update_config(updates)
    return config.to_dict()


# ── Voice Provider Management ────────────────────────────────────────────────

class VoiceProviderRequest(BaseModel):
    provider: str  # "traditional" or "gemini"

@app.get("/api/ambient/voice-providers")
async def get_voice_providers():
    """Get available voice providers and current selection."""
    ambient = _get_ambient()
    return {
        "stt_provider": ambient.get_stt_provider(),
        "tts_provider": ambient.get_tts_provider(),
        "gemini_available": ambient._gemini_api_key is not None,
        "traditional_stt_available": ambient._traditional_stt is not None,
        "traditional_tts_available": (ambient._traditional_tts is not None
                                      and ambient._traditional_tts.is_available),
        "gemini_stt_available": ambient._gemini_stt is not None,
        "gemini_tts_available": ambient._gemini_tts is not None,
        "gemini_tts_voices": ["Aoede", "Charon", "Fenrir", "Kore",
                              "Puck", "Leda", "Orus", "Zephyr"],
    }

@app.post("/api/ambient/wake-word/enable")
async def enable_wake_word():
    """Enable wake word detection."""
    ambient = _get_ambient()
    ambient.update_config({"wake_word_enabled": True})
    return {"success": True, "wake_word_enabled": True}

@app.post("/api/ambient/wake-word/disable")
async def disable_wake_word():
    """Disable wake word detection."""
    ambient = _get_ambient()
    ambient.update_config({"wake_word_enabled": False})
    return {"success": True, "wake_word_enabled": False}

@app.get("/api/ambient/wake-word/status")
async def get_wake_word_status():
    """Get wake word detector status."""
    ambient = _get_ambient()
    if ambient.wake_word:
        return ambient.wake_word.get_stats()
    return {"available": False, "running": False, "wake_word": None}

@app.post("/api/ambient/stt-provider")
async def set_stt_provider(req: VoiceProviderRequest):
    """Switch STT provider between traditional and Gemini."""
    ambient = _get_ambient()
    result = ambient.set_stt_provider(req.provider)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result

@app.post("/api/ambient/tts-provider")
async def set_tts_provider(req: VoiceProviderRequest):
    """Switch TTS provider between traditional and Gemini."""
    ambient = _get_ambient()
    result = ambient.set_tts_provider(req.provider)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result


# ── Voice Enrollment ─────────────────────────────────────────────────────────

@app.post("/api/ambient/enroll")
async def start_enrollment(req: EnrollmentRequest):
    """Start voice enrollment recording. User speaks for specified duration."""
    ambient = _get_ambient()
    if ambient.enrollment is None:
        # Initialize components if not yet done
        await ambient._init_components()
    if ambient.speaker_id is None:
        raise HTTPException(
            503,
            "Speaker identification is unavailable on this system. "
            "Voice enrollment requires SpeechBrain (ECAPA-TDNN). "
            "Ambient listening still works without enrollment."
        )
    result = await ambient.enrollment.start_enrollment(req.duration_seconds)
    return result


@app.get("/api/ambient/enrollment-status")
async def enrollment_status():
    """Check if user has enrolled their voice."""
    ambient = _get_ambient()
    enrolled = False
    if ambient.speaker_id:
        enrolled = ambient.speaker_id.is_enrolled()
    return {"enrolled": enrolled, "speaker_id_available": ambient.speaker_id is not None}


@app.post("/api/ambient/speaker-alias")
async def set_speaker_alias(req: SpeakerAliasRequest):
    """Set a human-readable name for a detected speaker."""
    ambient = _get_ambient()
    if not ambient.speaker_id:
        raise HTTPException(503, "Speaker ID not initialized.")
    ambient.speaker_id.set_alias(req.speaker_label, req.name)
    return {"success": True, "label": req.speaker_label, "name": req.name}


# ── Conversations ────────────────────────────────────────────────────────────

@app.get("/api/ambient/conversations")
async def get_conversations(limit: int = Query(50, ge=1, le=200),
                             offset: int = Query(0, ge=0)):
    """List captured ambient conversations."""
    ambient = _get_ambient()
    if not ambient.conversation:
        return {"conversations": [], "total": 0}
    convs = ambient.conversation.get_conversations(limit=limit, offset=offset)
    return {"conversations": convs, "total": len(convs)}


@app.get("/api/ambient/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a specific conversation by ID."""
    ambient = _get_ambient()
    if not ambient.conversation:
        raise HTTPException(404, "No conversations available.")
    conv = ambient.conversation.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, f"Conversation {conv_id} not found.")
    return conv


@app.get("/api/ambient/live-transcript")
async def get_live_transcript():
    """Get current in-progress conversation turns."""
    ambient = _get_ambient()
    if not ambient.conversation:
        return {"turns": []}
    return {"turns": ambient.conversation.get_current_turns()}


# ── Text-to-Speech ───────────────────────────────────────────────────────────

@app.post("/api/tts/synthesize")
async def tts_synthesize(req: TTSSynthesizeRequest):
    """Synthesize text to speech. Returns WAV audio.
    Uses the currently configured TTS provider (traditional or Gemini)."""
    ambient = _get_ambient()

    # Ensure components are initialized
    if not ambient.tts:
        await ambient._init_components()

    tts = ambient.tts
    if not tts or not tts.is_available:
        raise HTTPException(503, "TTS not available. Check provider config or model download.")

    # Use async version if available (Gemini), otherwise sync
    if hasattr(tts, 'synthesize_to_wav_async'):
        wav_bytes = await tts.synthesize_to_wav_async(req.text)
    else:
        wav_bytes = tts.synthesize_to_wav(req.text)

    if not wav_bytes:
        raise HTTPException(500, "TTS synthesis failed.")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@app.get("/api/tts/status")
async def tts_status():
    """Get TTS availability status."""
    ambient = _get_ambient()
    if ambient.tts:
        return ambient.tts.get_stats()
    return {"available": False, "voice": None}


# ── Voice Query (Speak → STT → RAG → TTS) ───────────────────────────────────

@app.post("/api/voice/query")
async def voice_query(req: VoiceQueryRequest):
    """
    Voice query pipeline:
    1. Decode base64 audio
    2. Transcribe with Whisper (STT)
    3. Run through RAG pipeline
    4. Synthesize response with TTS
    5. Return text + audio
    """
    import base64
    import numpy as np

    ambient = _get_ambient()
    if not ambient.transcriber:
        await ambient._init_components()

    # 1. Decode audio
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
    except Exception as e:
        raise HTTPException(400, f"Invalid audio data: {e}")

    if len(audio) < 1600:  # < 0.1s at 16kHz
        raise HTTPException(400, "Audio too short.")

    # 2. Transcribe
    transcript = await ambient.transcriber.transcribe(audio)
    user_text = transcript.get("text", "").strip()
    if not user_text:
        return {"transcript": "", "answer": "I couldn't hear anything.", "audio": None}

    # 3. RAG pipeline
    answer_text = ""
    evidence = []
    try:
        if rag_engine.initialized:
            rag_result = await rag_engine.rag_chat(
                user_message=user_text,
                session_id="voice",
                conversation_history=[],
            )
            answer_text = rag_result.get("answer", "")
            evidence = rag_result.get("evidence", [])
    except Exception as e:
        answer_text = f"I heard you say: '{user_text}', but had trouble processing: {str(e)}"

    if not answer_text:
        answer_text = f"I heard: '{user_text}'. I don't have enough context to answer that yet."

    # 4. TTS (optional) — uses active provider (traditional or Gemini)
    audio_base64_response = None
    if ambient.tts and ambient.tts.is_available and ambient.config.tts_enabled:
        if hasattr(ambient.tts, 'synthesize_to_wav_async'):
            wav_bytes = await ambient.tts.synthesize_to_wav_async(answer_text)
        else:
            wav_bytes = ambient.tts.synthesize_to_wav(answer_text)
        if wav_bytes:
            audio_base64_response = base64.b64encode(wav_bytes).decode("utf-8")

    return {
        "transcript": user_text,
        "answer": answer_text,
        "evidence": evidence,
        "audio_base64": audio_base64_response,
        "language": transcript.get("language", ""),
        "stt_confidence": transcript.get("confidence", 0),
        "stt_provider": ambient.config.stt_provider,
        "tts_provider": ambient.config.tts_provider,
    }


# ── WebSocket: Live Ambient Transcript Stream ───────────────────────────────

_ws_clients: set[WebSocket] = set()


async def _broadcast_to_ws_clients(data: dict):
    """Broadcast ambient events to all connected WebSocket clients."""
    if not _ws_clients:
        return
    msg = json.dumps(data)
    disconnected = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    _ws_clients -= disconnected


@app.websocket("/ws/ambient")
async def websocket_ambient(ws: WebSocket):
    """
    WebSocket for live ambient transcript streaming.
    Sends:
      - {"type": "transcript", "speaker_label": ..., "text": ..., ...}
      - {"type": "vad_activity", "speech_prob": ..., "timestamp": ...}
      - {"type": "status", "status": "listening"|"transcribing"|...}
    """
    await ws.accept()
    _ws_clients.add(ws)

    # Register broadcast callback with ambient service
    ambient = rag_engine.ambient_service
    if ambient:
        ambient.set_ws_broadcast(_broadcast_to_ws_clients)

    try:
        # Send initial status
        if ambient:
            await ws.send_text(json.dumps({
                "type": "status",
                **ambient.get_status(),
            }))

        # Keep connection alive, listen for client messages
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Handle client commands via WebSocket
                data = json.loads(msg)
                if data.get("command") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await ws.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)


# ── PageIndex Document Management ────────────────────────────────────────────

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF to PageIndex for tree-based document retrieval.
    The document is indexed by PageIndex's reasoning engine and becomes
    available as the 6th retrieval channel.
    """
    if not rag_engine.pageindex_store:
        raise HTTPException(
            status_code=503,
            detail="PageIndex is not enabled. Check config/pageindex_config.py"
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for document indexing."
        )

    # Save uploaded file to temp location
    upload_dir = os.path.join(
        rag_engine.data_dir, "pageindex", "uploads"
    )
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        result = rag_engine.pageindex_store.upload_document(
            file_path=temp_path,
            filename=file.filename,
        )

        return {
            "status": "success",
            "doc_id": result["doc_id"],
            "filename": result["filename"],
            "processing_status": result["status"],
            "already_indexed": result.get("already_indexed", False),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/api/documents")
async def list_documents():
    """List all PageIndex-indexed documents."""
    if not rag_engine.pageindex_store:
        return {"documents": [], "pageindex_enabled": False}

    try:
        docs = rag_engine.pageindex_store.list_documents()
        return {
            "documents": docs,
            "total": len(docs),
            "pageindex_enabled": True,
        }
    except Exception as e:
        print(f"⚠ list_documents error: {e}")
        return {
            "documents": [],
            "total": 0,
            "pageindex_enabled": True,
            "error": str(e),
        }


@app.get("/api/documents/usage")
async def get_pageindex_usage():
    """Get PageIndex API usage stats for the current month."""
    if not rag_engine.pageindex_store:
        return {"enabled": False, "usage": {}, "connected": False}

    try:
        return {
            "enabled": True,
            "connected": rag_engine.pageindex_store.is_connected,
            "usage": rag_engine.pageindex_store.get_usage(),
            "stats": rag_engine.pageindex_store.get_stats(),
        }
    except Exception as e:
        print(f"⚠ get_pageindex_usage error: {e}")
        return {
            "enabled": True,
            "connected": False,
            "usage": {},
            "stats": {},
            "error": str(e),
        }


@app.get("/api/documents/{doc_id}")
async def get_document_info(doc_id: str):
    """Get status and info for a specific document."""
    if not rag_engine.pageindex_store:
        raise HTTPException(status_code=503, detail="PageIndex not enabled")

    info = rag_engine.pageindex_store.get_document_info(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="Document not found")

    # Also check live status
    status = rag_engine.pageindex_store.check_status(doc_id)
    info["live_status"] = status.get("status", "unknown")
    return info


@app.get("/api/documents/{doc_id}/tree")
async def get_document_tree(doc_id: str):
    """Get the hierarchical tree structure of a document."""
    if not rag_engine.pageindex_store:
        raise HTTPException(status_code=503, detail="PageIndex not enabled")

    try:
        tree = rag_engine.pageindex_store.get_tree(doc_id)
        if tree is None:
            raise HTTPException(status_code=404, detail="Tree not available")
        return {"doc_id": doc_id, "tree": tree}
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠ get_document_tree error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from PageIndex."""
    if not rag_engine.pageindex_store:
        raise HTTPException(status_code=503, detail="PageIndex not enabled")

    removed = rag_engine.pageindex_store.delete_document(doc_id)
    return {
        "status": "deleted" if removed else "not_found",
        "doc_id": doc_id,
    }


@app.post("/api/documents/query")
async def query_documents(request: MemorySearchRequest):
    """
    Query uploaded documents using PageIndex reasoning retrieval.
    Returns structured sections with page numbers.
    """
    if not rag_engine.pageindex_store:
        raise HTTPException(status_code=503, detail="PageIndex not enabled")

    if not rag_engine.pageindex_store.has_documents:
        return {
            "answer": "",
            "sections": [],
            "message": "No documents uploaded yet.",
        }

    # Use chat retrieval for a natural answer
    answer = rag_engine.pageindex_store.chat_retrieve(
        query=request.query,
        stream=False,
    )

    # Also get structured sections
    sections = rag_engine.pageindex_store.retrieve_sections(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "answer": answer,
        "sections": sections,
        "doc_count": len(rag_engine.pageindex_store.get_all_doc_ids()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
        timeout_keep_alive=120,   # Keep proxy connections alive longer
    )
