"""
FastAPI Backend Server for Cortex Lab — Qwen3.5-9B-Opus Reasoning
Serves the Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled model via REST API + Server-Sent Events (streaming).
Includes full Agentic RAG system with memory, retrieval, and multi-agent reasoning.

Model: Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled (local 4-bit)
"""

import os
import sys
import hashlib

# Windows consoles often use cp1252; emoji in log lines would raise UnicodeEncodeError.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
import json
import re
import asyncio
import uuid
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List
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
from fastapi.responses import StreamingResponse, Response, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Add backend dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine import rag_engine
from src.prompts import PromptBuilder
from src.runtime.approval_executor import ApprovalExecutionWorker
from src.runtime.memory_personalization import (
    build_ambient_terms,
    build_memory_extraction_profile,
    evaluate_personal_memory_quality,
    is_context_dependent_query,
    select_prompt_evidence,
)
from src.runtime.task_manager import RuntimeTaskManager

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

# ── Model Selection ──────────────────────────────────────────────────────
# Primary: Qwen3.5-9B-Opus (local download)
# Fallback: fine-tuned model from pipeline, then HuggingFace
_QWEN_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen35-9b-opus")
if os.path.exists(os.path.join(_QWEN_LOCAL, "config.json")):
    _default_model = _QWEN_LOCAL
elif _fine_tuned_path:
    _default_model = _fine_tuned_path
else:
    _default_model = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled"

MODEL_NAME = os.environ.get("MODEL_NAME", _default_model)

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
_approval_execution_worker: Optional[ApprovalExecutionWorker] = None
_runtime_task_manager = RuntimeTaskManager()
rag_engine.set_runtime_task_manager(_runtime_task_manager)

SUPPORTED_RUNTIME_MODES = ("cloud", "hybrid", "local_offline")
SUPPORTED_LLM_PROVIDERS = ("local", "gemini", "gemma_local")
SUPPORTED_VOICE_PROVIDERS = ("traditional", "gemini", "local")
_LLM_PROVIDER_ALIAS_TO_BACKEND = {
    "local": "local",
    "gemini": "gemini",
    "gemma_local": "local",
}
_VOICE_PROVIDER_ALIAS_TO_BACKEND = {
    "traditional": "traditional",
    "local": "local",
    "gemini": "gemini",
}

_runtime_selection: Dict[str, Any] = {
    "mode": "cloud",
    "llm_provider": "local",
    "stt_provider": "traditional",
    "tts_provider": "traditional",
    "allow_cloud_fallback": True,
    "updated_at": datetime.utcnow().isoformat() + "Z",
}

_MODELPACK_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "infra",
    "modelpacks",
    "release-manifest.json",
)
_MODELPACK_DOCS_URL = (
    "https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs%2Fapi%2Fkotlin%2Fgetting_started.md"
)


def _default_modelpacks_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "generated_at": _utc_now_iso(),
        "signature_required": True,
        "channels": ["stable", "candidate", "canary"],
        "docs_url": _MODELPACK_DOCS_URL,
        "source": "builtin-default",
        "packs": [
            {
                "id": "gemma-4-e4b-it-litert-lm",
                "display_name": "Gemma 4 E4B IT (LiteRT-LM)",
                "version": "2026.04.0",
                "target": "android-web",
                "family": "gemma-4",
                "quantization": "E4B",
                "availability": "available",
                "summary": "Higher-quality Gemma 4 local model for capable devices.",
                "download_url": "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm",
                "cta_label": "Download from Hugging Face",
                "requires": ["litertlm-runtime", "gemma-runtime-bridge"],
                "files": [],
            },
            {
                "id": "gemma-4-e2b-it-litert-lm",
                "display_name": "Gemma 4 E2B IT (LiteRT-LM)",
                "version": "2026.04.0",
                "target": "android-web",
                "family": "gemma-4",
                "quantization": "E2B",
                "availability": "available",
                "summary": "Lean Gemma 4 local model for faster installs and mid-range devices.",
                "download_url": "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm",
                "cta_label": "Download from Hugging Face",
                "requires": ["litertlm-runtime", "gemma-runtime-bridge"],
                "files": [],
            },
            {
                "id": "gemma-3-5-ft-local",
                "display_name": "Gemma 3.5 Fine-Tuned (Planned)",
                "version": "planned",
                "target": "android-web",
                "family": "gemma-3.5",
                "quantization": "tbd",
                "availability": "coming_soon",
                "summary": "Reserved slot for the upcoming fine-tuned local model.",
                "cta_label": "Coming Soon",
                "requires": ["litertlm-runtime", "finetuned-pack-release"],
                "files": [],
            },
        ],
    }

# ── Concurrency & Timeout Guards (§9.1, §9.2) ───────────────────────────────
_inference_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent RAG/chat requests
_REQUEST_TIMEOUT = 180.0  # Hard timeout in seconds for any LLM request


def _extract_memory_id(metadata: Dict[str, Any]) -> str:
    memory_id = str(metadata.get("memory_id", "")).strip()
    if memory_id:
        return memory_id

    args = metadata.get("arguments")
    if isinstance(args, dict):
        memory_id = str(args.get("memory_id", "")).strip()
        if memory_id:
            return memory_id

    return ""


def _approval_worker_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _approval_worker_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _build_approval_execution_handlers() -> Dict[str, Any]:
    async def _execute_delete_memory(permission_request):
        metadata = dict(permission_request.metadata or {})
        memory_id = _extract_memory_id(metadata)
        if not memory_id:
            raise ValueError("approved delete_memory request missing memory_id")

        success = rag_engine.delete_memory(memory_id)
        return {
            "status": "ok" if success else "not_found",
            "memory_id": memory_id,
        }

    async def _execute_ingest_memory(permission_request):
        metadata = dict(permission_request.metadata or {})
        content = str(metadata.get("content", "")).strip()
        if not content:
            raise ValueError("approved ingest_memory request missing content")

        source = str(metadata.get("source", "manual") or "manual")
        session_id = str(metadata.get("session_id", "") or "")

        memory = await rag_engine.ingest_memory(
            content=content,
            source=source,
            session_id=session_id,
        )
        memory_id = ""
        if isinstance(memory, dict):
            memory_id = str(memory.get("id", ""))

        return {
            "status": "ok",
            "memory_id": memory_id,
            "source": source,
        }

    return {
        "delete_memory": _execute_delete_memory,
        "ingest_memory": _execute_ingest_memory,
    }


async def _start_approval_execution_worker() -> None:
    global _approval_execution_worker

    safe_runtime = getattr(rag_engine, "safe_tool_runtime", None)
    if safe_runtime is None:
        _approval_execution_worker = None
        print("  ⚠ Approval execution worker disabled: safe runtime unavailable")
        return

    if _approval_execution_worker and _approval_execution_worker.is_running():
        return

    _approval_execution_worker = ApprovalExecutionWorker(
        safe_tool_runtime=safe_runtime,
        handlers=_build_approval_execution_handlers(),
        poll_interval_seconds=_approval_worker_float_env("APPROVAL_WORKER_POLL_SECONDS", 2.0),
        execution_timeout_seconds=_approval_worker_float_env("APPROVAL_WORKER_TIMEOUT_SECONDS", 60.0),
        max_attempts=_approval_worker_int_env("APPROVAL_WORKER_MAX_ATTEMPTS", 2),
    )
    await _approval_execution_worker.start()
    print("  ✓ Approval execution worker started")


async def _stop_approval_execution_worker() -> None:
    global _approval_execution_worker

    if _approval_execution_worker is None:
        return

    await _approval_execution_worker.stop()
    _approval_execution_worker = None

# ── Lifespan – loads model once on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, model_loaded, model_info

    print("\n" + "=" * 64)
    print("  Cortex Lab  ·  Qwen3.5-9B-Opus Reasoning  ·  FastAPI Backend")
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

        try:
            await _start_approval_execution_worker()
        except Exception as e:
            print(f"  ⚠ Approval execution worker start failed: {e}")

        model_loaded = True  # Mark ready so health check returns "ok"
        print(f"\n  Server ready → http://{HOST}:{PORT}\n")

        yield
        await _stop_approval_execution_worker()
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
    _is_qwen = "qwen" in MODEL_NAME.lower() or "opus" in MODEL_NAME.lower()
    model_display_name = "Qwen3.5-9B-Opus-Reasoning" if _is_qwen else (
        "DeepSeek-R1-7B (Fine-Tuned)" if _fine_tuned_path else "DeepSeek-R1-Distill-Qwen-7B"
    )

    model_info = {
        "name": model_display_name,
        "parameters": "9B" if _is_qwen else "7B",
        "quantization": quant,
        "device": gpu_name,
        "gpu_memory": gpu_mem,
        "max_context": 32768,
        "load_time_seconds": round(elapsed, 1),
        "fine_tuned": _fine_tuned_path is not None,
        "training_stages_completed": completed_stages,
        "model_path": MODEL_NAME[:80],
        "base_model": "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled" if _is_qwen else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
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

    try:
        await _start_approval_execution_worker()
    except Exception as e:
        print(f"  ⚠ Approval execution worker start failed: {e}")

    yield  # ← app runs here

    # cleanup
    await _stop_approval_execution_worker()
    rag_engine.shutdown()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cortex Lab — Qwen3.5-9B-Opus Agentic RAG API",
    version="2.0.0",
    lifespan=lifespan,
)

_default_allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
    "http://localhost:19006",
    "http://127.0.0.1:19006",
    "http://192.168.3.169:3000",
]

_allowed_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "")
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or _default_allowed_origins

_default_origin_regex = (
    r"^https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$"
    r"|^https://.*\.trycloudflare\.com$"
    r"|^https://.*\.up\.railway\.app$"
    r"|^https://.*\.railway\.app$"
)
_allow_origin_regex = os.environ.get("CORS_ALLOW_ORIGIN_REGEX", _default_origin_regex)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=_allow_origin_regex,
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
    llm_provider: str = Field(
        "local",
        description="'local'/'gemma_local' for on-device model or 'gemini' for Gemini API",
    )
    thinking_mode: bool = Field(True, description="When True, stream <think> reasoning blocks; when False, suppress for faster responses")

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
    llm_provider: str = Field(
        "local",
        description="'local'/'gemma_local' for on-device model or 'gemini' for Gemini API",
    )
    thinking_mode: bool = Field(True, description="When True, stream <think> reasoning blocks; when False, suppress for faster responses")


class RuntimeModeRequest(BaseModel):
    mode: str = Field(
        "cloud",
        description="Runtime mode: cloud | hybrid | local_offline",
    )
    allow_cloud_fallback: Optional[bool] = True


class RuntimeProvidersRequest(BaseModel):
    llm_provider: Optional[str] = None
    stt_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    allow_cloud_fallback: Optional[bool] = None


class ModelpackVerifyRequest(BaseModel):
    file_path: str
    expected_sha256: str = Field(..., min_length=64, max_length=64)

class MemoryIngestRequest(BaseModel):
    content: str
    source: str = "manual"
    session_id: str = ""

class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 10


class RuntimeToolOperationRequest(BaseModel):
    request_id: str = ""
    tool_name: str
    command_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PermissionResolveRequest(BaseModel):
    approve: bool
    actor: str = "operator"
    note: str = ""


class RuntimeTaskCreateRequest(BaseModel):
    task_id: str = ""
    parent_task_id: str = ""
    permission_scope: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuntimeTaskCancelRequest(BaseModel):
    reason: str = ""
    propagate: bool = True


class MemoryExtractionJobRequest(BaseModel):
    limit: int = Field(500, ge=1, le=5000)
    offset: int = Field(0, ge=0)
    dry_run: bool = True


class MemoryQualityEvaluateRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)
    top_k: int = Field(5, ge=1, le=20)

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
# Only catch truly robotic/format-leak patterns that indicate the model is
# dumping structured output instead of natural language.
# NOTE: Generic phrases like "synthesizing", "lived experiences", "belief evolution"
# were REMOVED — they caused false positives with Qwen3.5 which uses these
# legitimately in thoughtful answers, causing truncation.
_STREAMING_HALLUC_PHRASES = [
    # Self-RAG format leaks (model outputting structured format instead of prose)
    "**answer:**", "**evidence:**", "**confidence:**",
    "**relevance:**", "**sources:**",
    "answer:\n", "evidence:\n", "confidence: high",
    "confidence: medium", "confidence: low",
    # Robotic database-dump prefixes
    "based on your stored memories:",
    "according to the evidence provided:",
    "from your stored memories:",
    # Meta-answers (model talking about itself instead of answering)
    "here's the revised answer",
    "revised answer focused on",
    "comprehensive answer to your question",
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


# Personal-info supplementation config used by both streaming and non-streaming RAG.
_PERSONAL_QUERY_TRIGGERS = [
    "my name", "who am i", "my email", "e-mail", "my phone",
    "my number", "contact", "my education", "where do i study",
    "my university", "my college", "my skills", "my resume",
    "my experience", "my projects", "what do i do",
    "my location", "my hometown", "my address", "about me",
    "my degree", "my profile", "my background",
]

_PERSONAL_QUERY_BASE = "resume contact information summary"
_PERSONAL_QUERY_AUGMENTS = {
    "my name": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "who am i": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "my email": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "e-mail": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "my phone": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "my number": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "contact": f"{_PERSONAL_QUERY_BASE} name email phone B.Tech",
    "my education": f"{_PERSONAL_QUERY_BASE} education university degree B.Tech",
    "where do i study": f"{_PERSONAL_QUERY_BASE} education university degree B.Tech",
    "my university": f"{_PERSONAL_QUERY_BASE} education university degree B.Tech",
    "my college": f"{_PERSONAL_QUERY_BASE} education university degree B.Tech",
    "my skills": f"{_PERSONAL_QUERY_BASE} skills programming technical tools frameworks",
    "my resume": f"{_PERSONAL_QUERY_BASE} name email phone education skills B.Tech",
    "my experience": f"{_PERSONAL_QUERY_BASE} experience internship work projects",
    "my projects": f"{_PERSONAL_QUERY_BASE} projects portfolio built developed",
    "about me": f"{_PERSONAL_QUERY_BASE} name email phone education skills B.Tech",
    "my degree": f"{_PERSONAL_QUERY_BASE} education university degree B.Tech",
    "my profile": f"{_PERSONAL_QUERY_BASE} name email phone education skills B.Tech",
    "my background": f"{_PERSONAL_QUERY_BASE} education experience skills B.Tech",
}

_UNCERTAIN_ANSWER_MARKERS = [
    "i don't have",
    "i do not have",
    "i don't know",
    "cannot determine",
    "can't determine",
    "not enough",
    "no information",
    "unable to",
    "unknown",
    "can't find",
    "cannot find",
]

_SIMPLE_PERSONAL_FACT_TRIGGERS = [
    "my name", "who am i", "full name", "what's my name", "whats my name",
    "my email", "e-mail", "mail address", "gmail",
    "my phone", "my number", "contact number", "mobile",
    "where do i study", "my university", "my college", "my degree",
]

_PERSONAL_QUALITY_DEFAULT_QUERIES = [
    "What is my name?",
    "What is my email?",
    "What is my phone number?",
    "Where do I study?",
    "What projects have I built?",
]


def _is_personal_info_query(query: str) -> bool:
    q = (query or "").lower()
    return any(trigger in q for trigger in _PERSONAL_QUERY_TRIGGERS)


def _is_simple_personal_fact_query(query: str) -> bool:
    q = (query or "").lower()
    return any(trigger in q for trigger in _SIMPLE_PERSONAL_FACT_TRIGGERS)


def _build_personal_augmented_query(query: str) -> str:
    q = (query or "").lower()
    for trigger, augmented in _PERSONAL_QUERY_AUGMENTS.items():
        if trigger in q:
            return augmented
    return f"{_PERSONAL_QUERY_BASE} name email phone B.Tech"


def _collect_ambient_terms() -> str:
    """Collect compact ambient conversation terms for optional retrieval augmentation."""
    try:
        ambient = getattr(rag_engine, "ambient_service", None)
        if not ambient or not getattr(ambient, "conversation", None):
            return ""

        current_turns = ambient.conversation.get_current_turns() or []
        recent_conversations = ambient.conversation.get_conversations(limit=3, offset=0) or []
        return build_ambient_terms(current_turns, recent_conversations)
    except Exception:
        return ""


def _supplement_ambient_evidence(
    user_message: str,
    evidence: list,
    search_fn=None,
    min_score: float = 0.42,
) -> list:
    """Optionally augment evidence for context-dependent queries using ambient signals."""
    if not is_context_dependent_query(user_message):
        return evidence

    ambient_terms = _collect_ambient_terms()
    if not ambient_terms:
        return evidence

    search_fn = search_fn or rag_engine.search_memories
    merged = list(evidence) if isinstance(evidence, list) else []
    existing_previews = {
        (item.get("content", "")[:80])
        for item in merged
        if isinstance(item, dict)
    }

    try:
        supplements = search_fn(f"{user_message} {ambient_terms}".strip(), top_k=3) or []
    except Exception:
        return merged

    for mem in supplements:
        score = float(mem.get("score", 0) or 0)
        content = (mem.get("content", "") or "").strip()
        if not content or score <= min_score:
            continue

        preview = content[:80]
        if preview in existing_previews:
            continue

        merged.insert(0, {
            "content": content[:600],
            "score": score,
            "channel": "ambient_supplement",
            "memory_type": mem.get("memory_type", "semantic"),
        })
        existing_previews.add(preview)

    return merged


def _supplement_personal_evidence(
    user_message: str,
    evidence: list,
    search_fn=None,
    min_score: float = 0.45,
    ambient_terms: str = "",
) -> list:
    """Best-effort direct supplement for personal-info queries.

    Uses augmented query terms that better match resume-style personal memories.
    """
    if not _is_personal_info_query(user_message):
        return evidence

    search_fn = search_fn or rag_engine.search_memories
    merged = list(evidence) if isinstance(evidence, list) else []
    existing_previews = {
        (item.get("content", "")[:80])
        for item in merged
        if isinstance(item, dict)
    }

    try:
        augmented_query = _build_personal_augmented_query(user_message)
        if ambient_terms:
            augmented_query = f"{augmented_query} {ambient_terms}"[:320]
        supplements = search_fn(augmented_query, top_k=3) or []
    except Exception:
        return merged

    for mem in supplements:
        score = float(mem.get("score", 0) or 0)
        content = (mem.get("content", "") or "").strip()
        if not content or score <= min_score:
            continue

        preview = content[:80]
        if preview in existing_previews:
            continue

        merged.insert(0, {
            "content": content[:600],
            "score": score,
            "channel": "direct_supplement",
            "memory_type": mem.get("memory_type", "semantic"),
        })
        existing_previews.add(preview)

    return merged


def _is_uncertain_answer(answer: str) -> bool:
    clean = (answer or "").strip().lower()
    if len(clean) < 8:
        return True
    return any(marker in clean for marker in _UNCERTAIN_ANSWER_MARKERS)


def _answer_contains_extracted_fact(answer: str, extracted: str) -> bool:
    """Check whether the generated answer actually contains extracted fact values."""
    import re as _re

    haystack = (answer or "").lower()
    if not haystack:
        return False

    tokens = []

    # Prefer explicit fact values wrapped in markdown bold.
    for value in _re.findall(r'\*\*([^*]{2,80})\*\*', extracted or ""):
        v = value.strip().lower()
        if v:
            tokens.append(v)

    # Also capture direct email/phone values.
    for value in _re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', extracted or ""):
        tokens.append(value.strip().lower())
    for value in _re.findall(r'\+\d{1,3}[\s-]?\d[\d\s-]{8,14}\d', extracted or ""):
        tokens.append(value.strip().lower())

    # Deduplicate while preserving order.
    unique_tokens = []
    seen = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    return any(token in haystack for token in unique_tokens)


def _postprocess_non_stream_result(user_message: str, result: dict, search_fn=None) -> dict:
    """Align non-stream /api/rag/chat behavior with streaming factual safeguards."""
    if not isinstance(result, dict) or not _is_personal_info_query(user_message):
        return result

    ambient_terms = _collect_ambient_terms()
    evidence = result.get("evidence", [])
    evidence = _supplement_personal_evidence(
        user_message,
        evidence,
        search_fn=search_fn,
        ambient_terms=ambient_terms,
    )
    evidence = _supplement_ambient_evidence(user_message, evidence, search_fn=search_fn)
    result["evidence"] = evidence

    evidence_texts = [
        e.get("content", "").strip()
        for e in evidence
        if isinstance(e, dict) and e.get("content")
    ]
    if not evidence_texts:
        return result

    extracted = _try_extract_factual(user_message, evidence_texts)
    if not extracted:
        return result

    answer = result.get("answer", "")
    try:
        confidence = float(result.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0

    missing_fact_in_answer = (
        _is_simple_personal_fact_query(user_message)
        and not _answer_contains_extracted_fact(answer, extracted)
    )

    if missing_fact_in_answer or _is_uncertain_answer(answer) or confidence < 0.65:
        result["answer"] = extracted
        result["confidence"] = round(max(confidence, 0.88), 3)
        note = "Direct factual extraction applied for personal info query."
        thinking = (result.get("thinking") or "").strip()
        if note not in thinking:
            result["thinking"] = f"{thinking}\n{note}".strip() if thinking else note

    return result


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
    # Try to use the model's native chat template (best for Qwen/ChatML models)
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
    Qwen3.5/DeepSeek format: generation starts with <think>\n...reasoning...</think>answer
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

def _normalize_llm_provider(provider: str) -> str:
    requested = str(provider or "local").strip().lower()
    return _LLM_PROVIDER_ALIAS_TO_BACKEND.get(requested, requested)


def _normalize_voice_provider(provider: str) -> str:
    requested = str(provider or "traditional").strip().lower()
    return _VOICE_PROVIDER_ALIAS_TO_BACKEND.get(requested, requested)


def _llm_provider_availability() -> Dict[str, bool]:
    local_available = (
        rag_engine.initialized
        and rag_engine.llm.local_llm is not None
        and rag_engine.llm.local_llm.model is not None
    )
    gemini_available = (
        rag_engine.initialized
        and hasattr(rag_engine.llm, "has_gemini")
        and rag_engine.llm.has_gemini
    )
    return {
        "local": bool(local_available),
        "gemma_local": bool(local_available),
        "gemini": bool(gemini_available),
    }


def _runtime_provider_availability() -> Dict[str, Any]:
    llm = _llm_provider_availability()
    ambient = rag_engine.ambient_service if rag_engine.initialized else None

    traditional_stt = bool(ambient is not None and ambient._traditional_stt is not None)
    traditional_tts = bool(
        ambient is not None
        and ambient._traditional_tts is not None
        and ambient._traditional_tts.is_available
    )
    gemini_stt = bool(ambient is not None and ambient._gemini_stt is not None)
    gemini_tts = bool(ambient is not None and ambient._gemini_tts is not None)

    return {
        "llm": llm,
        "stt": {
            "traditional": traditional_stt,
            "local": traditional_stt,
            "gemini": gemini_stt,
        },
        "tts": {
            "traditional": traditional_tts,
            "local": traditional_tts,
            "gemini": gemini_tts,
        },
        "ambient_available": ambient is not None,
    }


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _runtime_selection_snapshot() -> Dict[str, Any]:
    return {
        "mode": _runtime_selection["mode"],
        "llm_provider": _runtime_selection["llm_provider"],
        "stt_provider": _runtime_selection["stt_provider"],
        "tts_provider": _runtime_selection["tts_provider"],
        "allow_cloud_fallback": bool(_runtime_selection["allow_cloud_fallback"]),
        "updated_at": _runtime_selection["updated_at"],
    }


def _set_runtime_selection(
    *,
    mode: Optional[str] = None,
    llm_provider: Optional[str] = None,
    stt_provider: Optional[str] = None,
    tts_provider: Optional[str] = None,
    allow_cloud_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    if mode is not None:
        _runtime_selection["mode"] = mode
    if llm_provider is not None:
        _runtime_selection["llm_provider"] = llm_provider
    if stt_provider is not None:
        _runtime_selection["stt_provider"] = stt_provider
    if tts_provider is not None:
        _runtime_selection["tts_provider"] = tts_provider
    if allow_cloud_fallback is not None:
        _runtime_selection["allow_cloud_fallback"] = bool(allow_cloud_fallback)
    _runtime_selection["updated_at"] = _utc_now_iso()
    return _runtime_selection_snapshot()


def _apply_llm_provider_selection(provider: str) -> None:
    requested = str(provider or "local").strip().lower()
    if requested not in SUPPORTED_LLM_PROVIDERS:
        raise HTTPException(400, f"Unsupported llm_provider '{requested}'")

    availability = _llm_provider_availability()
    if requested in ("local", "gemma_local") and not availability["local"]:
        raise HTTPException(409, "Local LLM backend is unavailable for this provider selection.")
    if requested == "gemini" and not availability["gemini"]:
        raise HTTPException(409, "Gemini LLM backend is unavailable for this provider selection.")

    if rag_engine.initialized and hasattr(rag_engine.llm, "set_provider"):
        rag_engine.llm.set_provider(_normalize_llm_provider(requested))


def _apply_voice_provider_selection(kind: str, provider: str) -> None:
    requested = str(provider or "traditional").strip().lower()
    if requested not in SUPPORTED_VOICE_PROVIDERS:
        raise HTTPException(400, f"Unsupported {kind}_provider '{requested}'")

    ambient = _get_ambient()
    backend_provider = _normalize_voice_provider(requested)

    if kind == "stt":
        result = ambient.set_stt_provider(backend_provider)
    elif kind == "tts":
        result = ambient.set_tts_provider(backend_provider)
    else:
        raise HTTPException(500, f"Unknown voice provider kind '{kind}'")

    if not result.get("success"):
        raise HTTPException(409, result.get("error", f"Unable to set {kind}_provider"))


def _set_request_provider(provider: str):
    """Set the LLM provider for this request. Raises ValueError if the
    requested provider is unavailable (no silent fallback)."""
    requested = str(provider or "local").strip().lower()
    normalized = _normalize_llm_provider(requested)

    if requested not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError("unsupported_provider")

    if rag_engine.initialized and hasattr(rag_engine.llm, "set_provider"):
        availability = _llm_provider_availability()

        if requested in ("local", "gemma_local") and not availability["local"]:
            raise ValueError("local_unavailable")

        if requested == "gemini" and not availability["gemini"]:
            raise ValueError("gemini_unavailable")

        rag_engine.llm.set_provider(normalized)


def _is_gemini_active() -> bool:
    """Check if Gemini is the currently active LLM provider."""
    return (
        rag_engine.initialized
        and hasattr(rag_engine.llm, "provider")
        and rag_engine.llm.provider == "gemini"
        and rag_engine.llm.has_gemini
    )


def _effective_max_tokens(req, local_default: int = 8192) -> int:
    """Apply token caps only for paid providers; local runtime uses full local budget."""
    requested = int(getattr(req, "max_tokens", local_default) or local_default)
    provider = str(getattr(req, "llm_provider", "local") or "local").lower()
    if provider in ("local", "gemma_local"):
        return local_default
    return max(1, min(requested, local_default))


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
            "runtime_llm_provider": _runtime_selection.get("llm_provider", "local"),
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
    availability = _llm_provider_availability()
    selected_provider = _runtime_selection.get("llm_provider", "local")
    current_backend = "local"
    if rag_engine.initialized and hasattr(rag_engine.llm, "provider"):
        current_backend = rag_engine.llm.provider

    available = [
        provider
        for provider in SUPPORTED_LLM_PROVIDERS
        if availability.get(provider, False)
    ]

    return {
        "provider": selected_provider,
        "active_backend": current_backend,
        "available": available,
        "gemini_configured": availability["gemini"],
        "local_model_loaded": availability["local"],
    }


@app.post("/api/llm/provider")
async def set_llm_provider(body: dict):
    """Switch the active LLM provider (local | gemini | gemma_local)."""
    provider = str(body.get("provider", "local") or "local").strip().lower()
    _apply_llm_provider_selection(provider)
    selection = _set_runtime_selection(llm_provider=provider)
    return {
        "provider": selection["llm_provider"],
        "active_backend": _normalize_llm_provider(selection["llm_provider"]),
        "status": "switched",
    }


@app.get("/api/runtime/mode")
async def get_runtime_mode():
    """Get runtime execution mode and fallback policy."""
    return {
        **_runtime_selection_snapshot(),
        "supported_modes": list(SUPPORTED_RUNTIME_MODES),
    }


@app.post("/api/runtime/mode")
async def set_runtime_mode(req: RuntimeModeRequest):
    """Set runtime mode (cloud | hybrid | local_offline)."""
    mode = str(req.mode or "cloud").strip().lower()
    if mode not in SUPPORTED_RUNTIME_MODES:
        raise HTTPException(400, f"Unsupported runtime mode '{mode}'")

    allow_cloud_fallback = (
        bool(req.allow_cloud_fallback)
        if req.allow_cloud_fallback is not None
        else bool(_runtime_selection["allow_cloud_fallback"])
    )

    if mode == "local_offline":
        allow_cloud_fallback = False
        availability = _runtime_provider_availability()
        if not availability["llm"]["local"]:
            raise HTTPException(
                409,
                "local_offline mode requires a local LLM backend (local/gemma_local).",
            )

        if _runtime_selection["llm_provider"] == "gemini":
            _apply_llm_provider_selection("gemma_local")
            _runtime_selection["llm_provider"] = "gemma_local"

        if not availability["ambient_available"]:
            _runtime_selection["stt_provider"] = "local"
            _runtime_selection["tts_provider"] = "local"

        if availability["ambient_available"]:
            if _runtime_selection["stt_provider"] == "traditional" and availability["stt"]["local"]:
                _apply_voice_provider_selection("stt", "local")
                _runtime_selection["stt_provider"] = "local"

            if _runtime_selection["tts_provider"] == "traditional" and availability["tts"]["local"]:
                _apply_voice_provider_selection("tts", "local")
                _runtime_selection["tts_provider"] = "local"

            if _runtime_selection["stt_provider"] == "gemini":
                if not availability["stt"]["local"]:
                    raise HTTPException(409, "local_offline mode requires local STT availability.")
                _apply_voice_provider_selection("stt", "local")
                _runtime_selection["stt_provider"] = "local"

            if _runtime_selection["tts_provider"] == "gemini":
                if not availability["tts"]["local"]:
                    raise HTTPException(409, "local_offline mode requires local TTS availability.")
                _apply_voice_provider_selection("tts", "local")
                _runtime_selection["tts_provider"] = "local"

    selection = _set_runtime_selection(
        mode=mode,
        allow_cloud_fallback=allow_cloud_fallback,
    )
    return {
        **selection,
        "supported_modes": list(SUPPORTED_RUNTIME_MODES),
    }


@app.get("/api/runtime/providers")
async def get_runtime_providers():
    """Get runtime provider selection and availability matrix."""
    availability = _runtime_provider_availability()
    available = {
        "llm": [p for p in SUPPORTED_LLM_PROVIDERS if availability["llm"].get(p, False)],
        "stt": [p for p in SUPPORTED_VOICE_PROVIDERS if availability["stt"].get(p, False)],
        "tts": [p for p in SUPPORTED_VOICE_PROVIDERS if availability["tts"].get(p, False)],
    }

    return {
        "selection": _runtime_selection_snapshot(),
        "available": available,
        "availability": availability,
        "supported": {
            "llm": list(SUPPORTED_LLM_PROVIDERS),
            "stt": list(SUPPORTED_VOICE_PROVIDERS),
            "tts": list(SUPPORTED_VOICE_PROVIDERS),
        },
    }


@app.post("/api/runtime/providers")
async def set_runtime_providers(req: RuntimeProvidersRequest):
    """Set runtime provider selection with mode-aware fallback policy."""
    mode = _runtime_selection["mode"]
    stt_requested = req.stt_provider is not None
    tts_requested = req.tts_provider is not None
    allow_cloud_fallback = (
        bool(req.allow_cloud_fallback)
        if req.allow_cloud_fallback is not None
        else bool(_runtime_selection["allow_cloud_fallback"])
    )

    if mode == "local_offline":
        allow_cloud_fallback = False

    next_llm = str(req.llm_provider or _runtime_selection["llm_provider"]).strip().lower()
    next_stt = str(req.stt_provider or _runtime_selection["stt_provider"]).strip().lower()
    next_tts = str(req.tts_provider or _runtime_selection["tts_provider"]).strip().lower()

    if next_llm not in SUPPORTED_LLM_PROVIDERS:
        raise HTTPException(400, f"Unsupported llm_provider '{next_llm}'")
    if next_stt not in SUPPORTED_VOICE_PROVIDERS:
        raise HTTPException(400, f"Unsupported stt_provider '{next_stt}'")
    if next_tts not in SUPPORTED_VOICE_PROVIDERS:
        raise HTTPException(400, f"Unsupported tts_provider '{next_tts}'")

    if mode == "local_offline":
        if next_llm == "gemini":
            raise HTTPException(409, "local_offline mode cannot use Gemini LLM provider.")
        if next_stt == "gemini":
            raise HTTPException(409, "local_offline mode cannot use Gemini STT provider.")
        if next_tts == "gemini":
            raise HTTPException(409, "local_offline mode cannot use Gemini TTS provider.")

    availability = _runtime_provider_availability()
    fallback_applied: List[Dict[str, str]] = []

    if not availability["llm"].get(next_llm, False):
        if mode == "hybrid" and allow_cloud_fallback and next_llm in ("local", "gemma_local") and availability["llm"]["gemini"]:
            fallback_applied.append({"target": "llm_provider", "from": next_llm, "to": "gemini"})
            next_llm = "gemini"
        else:
            raise HTTPException(409, f"Requested LLM provider '{next_llm}' is unavailable.")

    if stt_requested or mode == "local_offline":
        if not availability["stt"].get(next_stt, False):
            if mode == "hybrid" and allow_cloud_fallback and next_stt in ("traditional", "local") and availability["stt"]["gemini"]:
                fallback_applied.append({"target": "stt_provider", "from": next_stt, "to": "gemini"})
                next_stt = "gemini"
            else:
                raise HTTPException(409, f"Requested STT provider '{next_stt}' is unavailable.")

    if tts_requested or mode == "local_offline":
        if not availability["tts"].get(next_tts, False):
            if mode == "hybrid" and allow_cloud_fallback and next_tts in ("traditional", "local") and availability["tts"]["gemini"]:
                fallback_applied.append({"target": "tts_provider", "from": next_tts, "to": "gemini"})
                next_tts = "gemini"
            else:
                raise HTTPException(409, f"Requested TTS provider '{next_tts}' is unavailable.")

    _apply_llm_provider_selection(next_llm)

    if stt_requested:
        _apply_voice_provider_selection("stt", next_stt)
    if tts_requested:
        _apply_voice_provider_selection("tts", next_tts)

    selection = _set_runtime_selection(
        llm_provider=next_llm,
        stt_provider=next_stt if stt_requested else None,
        tts_provider=next_tts if tts_requested else None,
        allow_cloud_fallback=allow_cloud_fallback,
    )

    return {
        "selection": selection,
        "fallback_applied": fallback_applied,
    }


@app.get("/api/runtime/health")
async def get_runtime_health():
    """Runtime mode/provider health snapshot for orchestration surfaces."""
    availability = _runtime_provider_availability()
    task_snapshot = _runtime_task_manager.to_dict()
    return {
        "status": "ok",
        "selection": _runtime_selection_snapshot(),
        "provider_availability": availability,
        "active_llm_backend": (
            rag_engine.llm.provider
            if rag_engine.initialized and hasattr(rag_engine.llm, "provider")
            else "local"
        ),
        "model_loaded": model_loaded,
        "runtime_tasks": task_snapshot.get("summary", {}),
        "timestamp": _utc_now_iso(),
    }


@app.get("/api/modelpacks/manifest")
async def get_modelpacks_manifest():
    """Return model pack manifest with robust defaults for settings downloads."""
    manifest = _default_modelpacks_manifest()
    source = "builtin-default"

    if os.path.exists(_MODELPACK_MANIFEST_PATH):
        with open(_MODELPACK_MANIFEST_PATH, "r", encoding="utf-8") as handle:
            from_file = json.load(handle)

        if isinstance(from_file, dict):
            source = "infra/modelpacks/release-manifest.json"
            if isinstance(from_file.get("schema_version"), str):
                manifest["schema_version"] = from_file["schema_version"]
            if isinstance(from_file.get("generated_at"), str):
                manifest["generated_at"] = from_file["generated_at"]
            if isinstance(from_file.get("signature_required"), bool):
                manifest["signature_required"] = from_file["signature_required"]
            if isinstance(from_file.get("channels"), list):
                manifest["channels"] = from_file["channels"]
            if isinstance(from_file.get("docs_url"), str) and from_file["docs_url"].strip():
                manifest["docs_url"] = from_file["docs_url"].strip()

            packs = from_file.get("packs")
            if isinstance(packs, list) and packs:
                manifest["packs"] = packs

    manifest["source"] = source
    return manifest


@app.post("/api/modelpacks/verify")
async def verify_modelpack(req: ModelpackVerifyRequest):
    """Verify a model-pack artifact with SHA-256 digest checking."""
    file_path = os.path.abspath(os.path.expanduser(req.file_path))
    if not os.path.exists(file_path):
        raise HTTPException(404, f"Modelpack file not found: {file_path}")
    if not os.path.isfile(file_path):
        raise HTTPException(400, f"Modelpack path must be a file: {file_path}")

    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    actual = digest.hexdigest().lower()
    expected = req.expected_sha256.lower()

    return {
        "verified": actual == expected,
        "algorithm": "sha256",
        "file_path": file_path,
        "file_size_bytes": os.path.getsize(file_path),
        "expected_sha256": expected,
        "actual_sha256": actual,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Apply per-request LLM provider switch
    try:
        _set_request_provider(req.llm_provider)
    except ValueError as e:
        if "unsupported_provider" in str(e):
            raise HTTPException(
                400,
                f"Unsupported LLM provider '{req.llm_provider}'. Supported providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}",
            )
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
            max_new_tokens=_effective_max_tokens(req),
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
        "max_new_tokens": _effective_max_tokens(req),
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
    thinking_text = ""  # Collect thinking for optional later use
    suppress_thinking = not getattr(req, "thinking_mode", True)
    sent_think_open = False  # Whether we already streamed <think> tag

    # Qwen3.5-Opus: chat_template ends with <|im_start|>assistant\n<think>\n
    # so generation starts INSIDE a think block — no <think> tag in output.
    # Detect this: if prompt already contains the generation-prompt think tag.
    _prompt_ends_in_think = prompt.rstrip().endswith("<think>") or prompt.rstrip().endswith("<think>\n")
    in_think = _prompt_ends_in_think  # Start inside think block if template injected it

    if in_think and not suppress_thinking and not sent_think_open:
        chunk = {"id": msg_id, "delta": "<think>"}
        yield f"data: {json.dumps(chunk)}\n\n"
        sent_think_open = True

    for token_text in streamer:
        accumulated += token_text

        # ── Handle <think>…</think> blocks ──
        if "<think>" in accumulated and not in_think:
            in_think = True
            if not suppress_thinking and not sent_think_open:
                chunk = {"id": msg_id, "delta": "<think>"}
                yield f"data: {json.dumps(chunk)}\n\n"
                sent_think_open = True
        if in_think:
            if "</think>" in accumulated:
                # Think block complete — extract thinking and continue with answer
                think_end = accumulated.index("</think>") + len("</think>")
                thinking_text = accumulated[:think_end]
                if not suppress_thinking:
                    # Only send the closing tag — individual tokens were already
                    # streamed in the else branch below, so content is already in
                    # the frontend's message.content after <think>.
                    chunk = {"id": msg_id, "delta": "</think>"}
                    yield f"data: {json.dumps(chunk)}\n\n"
                accumulated = accumulated[think_end:]
                in_think = False
                if not accumulated:
                    continue
            else:
                if not suppress_thinking:
                    # Stream thinking tokens as they arrive (stripped of tags)
                    tok = token_text
                    for tag in ["<think>", "</think>"]:
                        tok = tok.replace(tag, "")
                    if tok:
                        chunk = {"id": msg_id, "delta": tok}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0)
                continue

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
        if "unsupported_provider" in str(e):
            raise HTTPException(
                400,
                f"Unsupported LLM provider '{req.llm_provider}'. Supported providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}",
            )
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

            # Keep non-stream behavior aligned with streaming safeguards for
            # personal-profile factual queries (name/email/phone/education).
            result = _postprocess_non_stream_result(user_message, result)

            # Store trace for observability history
            _store_trace(result.get("pipeline_trace"))
            runtime_tasks = _build_runtime_task_refs(result.get("pipeline_trace"))

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
                "runtime_tasks": runtime_tasks,
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
        runtime_tasks = _build_runtime_task_refs(rag_result.get("pipeline_trace"))

        # Metadata chunk is emitted after prompt-evidence shaping so it can
        # include memory budget and quality instrumentation.

        # Step 2: Build prompt with evidence context for streaming generation

        _query_complexity = query_analysis.get("complexity", 0.5) if isinstance(query_analysis, dict) else 0.5
        _query_intent = query_analysis.get("intent", "") if isinstance(query_analysis, dict) else ""
        _is_synthesis = _query_complexity >= 0.6 or _query_intent in ("reflective", "comparative", "causal")
        _is_local_model = not _is_gemini_active()

        ambient_terms = _collect_ambient_terms()

        # Personal and ambient supplements run before bounded evidence shaping.
        evidence = _supplement_personal_evidence(
            user_message,
            evidence,
            ambient_terms=ambient_terms,
        )
        evidence = _supplement_ambient_evidence(user_message, evidence)

        prompt_selection = select_prompt_evidence(
            evidence,
            query_analysis=query_analysis if isinstance(query_analysis, dict) else {},
            is_local_model=_is_local_model,
        )
        evidence_texts = prompt_selection.get("texts", [])
        has_pageindex_evidence = bool(prompt_selection.get("has_pageindex_evidence", False))
        pageindex_evidence_count = int(prompt_selection.get("pageindex_evidence_count", 0) or 0)
        prompt_metrics = dict(prompt_selection.get("metrics", {}))

        if ambient_terms:
            prompt_metrics["ambient_terms_used"] = ambient_terms

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

        # Personal-context quality instrumentation for Phase 4 memory evaluation.
        personal_quality = None
        if _is_personal_info_query(user_message):
            extracted = _try_extract_factual(user_message, evidence_texts)
            personal_quality = evaluate_personal_memory_quality(
                query=user_message,
                evidence_texts=evidence_texts,
                extracted_answer=extracted,
            )
            personal_quality["source"] = "live_stream"
            _store_memory_quality_snapshot(personal_quality)

        # Attach memory prompt and quality metrics into trace generation details.
        trace_payload = rag_result.get("pipeline_trace")
        if isinstance(trace_payload, dict):
            generation_details = trace_payload.setdefault("generation_details", {})
            generation_details["memory_prompt"] = prompt_metrics
            if personal_quality:
                generation_details["personal_memory_quality"] = personal_quality
            rag_result["pipeline_trace"] = trace_payload

        # Send metadata after prompt shaping so diagnostics are included.
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
                "runtime_tasks": runtime_tasks,
                "memory_prompt": prompt_metrics,
                "personal_memory_quality": personal_quality,
            },
        }
        yield f"data: {json.dumps(meta_chunk)}\n\n"

        # Store trace for observability history
        _store_trace(rag_result.get("pipeline_trace"))

        # ── Pre-generation: Check for false premises / no-data queries ──
        no_info_response = _check_no_info_streaming(user_message, evidence_texts)
        if no_info_response:
            chunk = {"id": msg_id, "delta": no_info_response}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id': msg_id, 'delta': '', 'done': True})}\n\n"
            return

        # ── Pre-generation: For simple factual queries, try direct extraction ──
        # Skip when thinking_mode is ON — user wants to see the model reason
        _thinking_on = getattr(req, "thinking_mode", True)
        if not _thinking_on:
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

        # ── Force thinking mode for Qwen3.5 ─────────────────────────────
        # Qwen3.5-Opus chat_template ends with <|im_start|>assistant\n<think>\n
        # but _format_chat only produces <|im_start|>assistant\n — append <think>
        # so the model enters its reasoning mode (matches chat_template behavior).
        # When thinking_mode is OFF we still inject it (model needs it) but
        # suppress the thinking tokens in the streaming loop below.
        if not rag_prompt.rstrip().endswith("<think>"):
            rag_prompt = rag_prompt.rstrip() + "<think>\n"

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
            "max_new_tokens": _effective_max_tokens(req),
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
        halluc_detected = False  # Track if hallucination was detected mid-stream
        suppress_thinking = not getattr(req, "thinking_mode", True)
        sent_think_open = False

        # Qwen3.5-Opus: if prompt ends with <think>, generation starts inside think block
        _prompt_ends_in_think = rag_prompt.rstrip().endswith("<think>") or rag_prompt.rstrip().endswith("<think>\n")
        in_think = _prompt_ends_in_think

        if in_think and not suppress_thinking and not sent_think_open:
            chunk = {"id": msg_id, "delta": "<think>"}
            yield f"data: {json.dumps(chunk)}\n\n"
            sent_think_open = True

        for token_text in streamer:
            accumulated += token_text

            # Handle <think>...</think> blocks
            if "<think>" in accumulated and not in_think:
                in_think = True
                if not suppress_thinking and not sent_think_open:
                    chunk = {"id": msg_id, "delta": "<think>"}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    sent_think_open = True
            if in_think:
                if "</think>" in accumulated:
                    # Think block complete
                    think_end = accumulated.index("</think>") + len("</think>")
                    if not suppress_thinking:
                        # Only send closing tag — tokens already streamed below
                        chunk = {"id": msg_id, "delta": "</think>"}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    accumulated = accumulated[think_end:]
                    in_think = False
                else:
                    if not suppress_thinking:
                        tok = token_text
                        for tag in ["<think>", "</think>"]:
                            tok = tok.replace(tag, "")
                        if tok:
                            chunk = {"id": msg_id, "delta": tok}
                            yield f"data: {json.dumps(chunk)}\n\n"
                            await asyncio.sleep(0)
                    continue

            # Clean special tokens from the visible output
            clean_token = token_text
            for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>",
                         "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"]:
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

        # Store conversation turns (lightweight history — NOT as memories)
        # Clean accumulated text of any leftover special tokens
        clean_accumulated = accumulated
        for tok in ["<think>", "</think>", "<｜end▁of▁sentence｜>",
                     "<|im_end|>", "<|endoftext|>", "<｜User｜>", "<｜Assistant｜>"]:
            clean_accumulated = clean_accumulated.replace(tok, "")
        clean_accumulated = clean_accumulated.strip()
        if rag_engine and rag_engine.initialized:
            try:
                _sid = req.session_id or f"session-{int(time.time())}"
                # Store user turn
                rag_engine.metadata_store.store_conversation_turn(
                    session_id=_sid,
                    role="user",
                    content=user_message,
                )
                # Store assistant turn
                if clean_accumulated:
                    rag_engine.metadata_store.store_conversation_turn(
                        session_id=_sid,
                        role="assistant",
                        content=clean_accumulated,
                    )
            except Exception as store_err:
                print(f"  ⚠️ Failed to store conversation turn: {store_err}")

    except Exception as e:
        error_chunk = {"id": msg_id, "delta": f"\n\n⚠️ RAG streaming error: {str(e)}", "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ── Memory Management Endpoints ─────────────────────────────────────────────

@app.post("/api/memories/ingest")
async def ingest_memory(req: MemoryIngestRequest):
    """Manually ingest a memory into the RAG system."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")

    request_id = f"ingest-{uuid.uuid4().hex[:12]}"
    try:
        evaluation = rag_engine.evaluate_tool_operation(
            request_id=request_id,
            tool_name="ingest_memory",
            command_text=req.content[:300],
            metadata={
                "content": req.content,
                "source": req.source,
                "session_id": req.session_id,
                "entrypoint": "api.memories.ingest",
            },
        )
    except RuntimeError as e:
        if "safe_tool_runtime_unavailable" in str(e):
            raise HTTPException(503, "Safe tool runtime is not available.")
        raise

    effect = ((evaluation.get("decision") or {}).get("effect") or "").lower()
    if effect == "deny":
        reason = ((evaluation.get("decision") or {}).get("reason") or "Blocked by policy")
        raise HTTPException(403, reason)

    if effect == "require_approval":
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending_approval",
                "request_id": request_id,
                "decision": evaluation.get("decision", {}),
                "permission_request": evaluation.get("permission_request"),
                "next": "Approve via /api/runtime/safety/permissions/{permission_id}/resolve. The approval worker executes approved requests automatically.",
            },
        )

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


@app.delete("/api/memories/purge/chat-queries")
async def purge_chat_query_memories():
    """Purge all memories that were auto-ingested from chat queries (source='chat').
    These pollute the vector store with questions instead of actual knowledge.
    Only memories added via manual/ambient/API ingestion should be kept."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")
    try:
        if rag_engine.metadata_store._use_duckdb:
            # Get all chat-source memory IDs
            rows = rag_engine.metadata_store.conn.execute(
                "SELECT id FROM memories WHERE source = 'chat'"
            ).fetchall()
            chat_ids = [r[0] for r in rows]
            deleted = 0
            for mid in chat_ids:
                rag_engine.delete_memory(mid)
                deleted += 1
            return {"status": "ok", "deleted": deleted, "message": f"Purged {deleted} chat-query memories"}
        return {"status": "error", "message": "DuckDB not available"}
    except Exception as e:
        raise HTTPException(500, f"Purge failed: {str(e)}")


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str, permission_id: str = ""):
    """Delete a memory with SafeToolRuntime approval enforcement."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")

    if permission_id:
        try:
            permission = rag_engine.get_permission_request(permission_id)
        except RuntimeError as e:
            if "safe_tool_runtime_unavailable" in str(e):
                raise HTTPException(503, "Safe tool runtime is not available.")
            raise

        if not permission:
            raise HTTPException(404, f"Permission request not found: {permission_id}")
        if permission.get("tool_name") != "delete_memory":
            raise HTTPException(400, "Permission is not valid for delete_memory")
        if permission.get("status") != "approved":
            raise HTTPException(409, "Permission request is not approved")

        expected_memory = str((permission.get("metadata") or {}).get("memory_id", "")).strip()
        if expected_memory and expected_memory != memory_id:
            raise HTTPException(400, "Permission request does not match memory_id")

        execution = (permission.get("metadata") or {}).get("_execution")
        execution_status = ""
        if isinstance(execution, dict):
            execution_status = str(execution.get("status", "")).strip().lower()

        if execution_status == "completed":
            result = execution.get("result") if isinstance(execution, dict) else {}
            result_status = "ok"
            if isinstance(result, dict) and str(result.get("status", "")).lower() == "not_found":
                result_status = "not_found"
            return {
                "status": result_status,
                "approved_execution": True,
                "permission_id": permission_id,
                "already_executed": True,
            }

        if execution_status == "running":
            return JSONResponse(
                status_code=202,
                content={
                    "status": "pending_execution",
                    "permission_id": permission_id,
                    "next": "Approved operation is executing in the background worker.",
                },
            )

        if execution_status == "failed":
            raise HTTPException(409, "Approved execution failed in background worker; inspect runtime safety audit.")

        success = rag_engine.delete_memory(memory_id)
        return {
            "status": "ok" if success else "not_found",
            "approved_execution": True,
            "permission_id": permission_id,
        }

    request_id = f"delete-{memory_id}-{uuid.uuid4().hex[:8]}"
    try:
        evaluation = rag_engine.evaluate_tool_operation(
            request_id=request_id,
            tool_name="delete_memory",
            command_text=f"memory_id={memory_id}",
            metadata={
                "memory_id": memory_id,
                "entrypoint": "api.memories.delete",
            },
        )
    except RuntimeError as e:
        if "safe_tool_runtime_unavailable" in str(e):
            raise HTTPException(503, "Safe tool runtime is not available.")
        raise

    effect = ((evaluation.get("decision") or {}).get("effect") or "").lower()
    if effect == "deny":
        reason = ((evaluation.get("decision") or {}).get("reason") or "Blocked by policy")
        raise HTTPException(403, reason)

    if effect == "require_approval":
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending_approval",
                "request_id": request_id,
                "decision": evaluation.get("decision", {}),
                "permission_request": evaluation.get("permission_request"),
                "next": "Approve via /api/runtime/safety/permissions/{permission_id}/resolve. The approval worker executes approved deletes automatically.",
            },
        )

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


@app.post("/api/runtime/memory-extraction/jobs")
async def create_memory_extraction_job(req: MemoryExtractionJobRequest):
    """Create a bounded background job to extract compact memory profiles."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")

    job_id = f"memext-{uuid.uuid4().hex[:10]}"
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "finished_at": None,
        "limit": req.limit,
        "offset": req.offset,
        "dry_run": req.dry_run,
        "total": 0,
        "processed": 0,
        "updated": 0,
        "failures": 0,
        "duration_ms": 0.0,
        "error": "",
    }
    _memory_extraction_jobs[job_id] = job

    # Keep registry bounded to avoid unbounded in-memory growth.
    if len(_memory_extraction_jobs) > 200:
        ordered = sorted(
            _memory_extraction_jobs.items(),
            key=lambda item: item[1].get("created_at", ""),
        )
        for stale_id, _stale_job in ordered[: max(len(_memory_extraction_jobs) - 200, 0)]:
            _memory_extraction_jobs.pop(stale_id, None)

    asyncio.create_task(
        _run_memory_extraction_job(
            job_id,
            limit=req.limit,
            offset=req.offset,
            dry_run=req.dry_run,
        )
    )

    return JSONResponse(status_code=202, content=job)


@app.get("/api/runtime/memory-extraction/jobs")
async def list_memory_extraction_jobs(limit: int = Query(20, ge=1, le=200)):
    """List recent memory extraction jobs."""
    jobs = list(_memory_extraction_jobs.values())
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {
        "count": len(jobs),
        "jobs": jobs[:limit],
    }


@app.get("/api/runtime/memory-extraction/jobs/{job_id}")
async def get_memory_extraction_job(job_id: str):
    """Get memory extraction job status by ID."""
    job = _memory_extraction_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Memory extraction job not found: {job_id}")
    return job


@app.post("/api/runtime/memory-quality/evaluate")
async def evaluate_memory_quality(req: MemoryQualityEvaluateRequest):
    """Evaluate personal-context memory precision/recall-style quality metrics."""
    if not rag_engine.initialized:
        raise HTTPException(503, "RAG engine not ready.")

    queries = [q for q in req.queries if q and q.strip()]
    if not queries:
        queries = list(_PERSONAL_QUALITY_DEFAULT_QUERIES)

    report = _evaluate_memory_quality_batch(queries=queries, top_k=req.top_k)
    report["top_k"] = req.top_k
    _store_memory_quality_snapshot(
        {
            "source": "manual_eval",
            "top_k": req.top_k,
            "avg_precision_at_k": report.get("avg_precision_at_k", 0.0),
            "recall_proxy_rate": report.get("recall_proxy_rate", 0.0),
            "extraction_hit_rate": report.get("extraction_hit_rate", 0.0),
            "sample_count": report.get("sample_count", 0),
        }
    )
    return report


@app.get("/api/runtime/memory-quality/history")
async def get_memory_quality_history(limit: int = Query(50, ge=1, le=500)):
    """Return memory-quality snapshot history for observability dashboards."""
    history = list(_memory_quality_history[-limit:])
    history.reverse()
    return {
        "count": len(history),
        "history": history,
    }


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


@app.get("/api/runtime/tool-contracts")
async def runtime_tool_contracts():
    """Get Phase 0 core tool contracts for autonomous runtime governance."""
    contracts = rag_engine.get_tool_contracts()
    return {
        "count": len(contracts),
        "contracts": contracts,
    }


@app.get("/api/runtime/interfaces")
async def runtime_interfaces():
    """Get machine-readable runtime interface snapshot (Phase 0 baseline)."""
    from src.runtime.contracts import phase0_interface_snapshot

    return phase0_interface_snapshot()


def _normalize_permission_scope(scope_values: Optional[List[str]]) -> Optional[set[str]]:
    if scope_values is None:
        return None
    normalized = {value.strip() for value in scope_values if value and value.strip()}
    return normalized


def _build_runtime_task_refs(pipeline_trace: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Derive coordinator/subagent task references from a pipeline trace payload."""
    if not isinstance(pipeline_trace, dict):
        return None

    coordinator_task_id = str(pipeline_trace.get("coordinator_task_id", "") or "").strip()
    subagent_task_ids: List[str] = []

    spawn_records = pipeline_trace.get("subagent_spawn_records")
    if isinstance(spawn_records, list):
        for record in spawn_records:
            if not isinstance(record, dict):
                continue

            task_id = str(record.get("task_id", "") or "").strip()
            if task_id and task_id not in subagent_task_ids:
                subagent_task_ids.append(task_id)

            if not coordinator_task_id:
                parent_task_id = str(record.get("parent_task_id", "") or "").strip()
                if parent_task_id:
                    coordinator_task_id = parent_task_id

    trace_id = str(pipeline_trace.get("trace_id", "") or "").strip()
    coordinator_plan = pipeline_trace.get("coordinator_plan")
    has_coordination = bool(subagent_task_ids) or (isinstance(coordinator_plan, dict) and len(coordinator_plan) > 0)
    if not coordinator_task_id and trace_id and has_coordination:
        coordinator_task_id = f"coord-{trace_id}"

    if not coordinator_task_id and not subagent_task_ids:
        return None

    all_task_ids: List[str] = []
    if coordinator_task_id:
        all_task_ids.append(coordinator_task_id)
    for task_id in subagent_task_ids:
        if task_id not in all_task_ids:
            all_task_ids.append(task_id)

    api: Dict[str, Any] = {
        "list": "/api/runtime/tasks",
        "subagents": [
            {
                "task_id": task_id,
                "get": f"/api/runtime/tasks/{task_id}",
                "cancel": f"/api/runtime/tasks/{task_id}/cancel",
            }
            for task_id in subagent_task_ids
        ],
    }
    if coordinator_task_id:
        api.update(
            {
                "coordinator": f"/api/runtime/tasks/{coordinator_task_id}",
                "cancel_coordinator": f"/api/runtime/tasks/{coordinator_task_id}/cancel",
            }
        )

    return {
        "trace_id": trace_id,
        "coordinator_task_id": coordinator_task_id,
        "subagent_task_ids": subagent_task_ids,
        "all_task_ids": all_task_ids,
        "api": api,
    }


@app.get("/api/runtime/tasks")
async def runtime_tasks_list():
    """List runtime tasks for Phase 3 coordinator/subagent lifecycle tracking."""
    return _runtime_task_manager.to_dict()


@app.get("/api/runtime/tasks/events")
async def runtime_task_events(request: Request):
    """SSE stream for runtime task lifecycle transitions."""
    subscriber_id = f"runtime-task-sub-{uuid.uuid4().hex[:8]}"

    async def event_generator():
        queue = _runtime_task_manager.subscribe(subscriber_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _runtime_task_manager.unsubscribe(subscriber_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runtime/tasks/{task_id}")
async def runtime_task_get(task_id: str):
    """Fetch one runtime task snapshot by task id."""
    try:
        task = _runtime_task_manager.get_task(task_id)
        return {"task": task.to_dict()}
    except KeyError:
        raise HTTPException(404, f"Task not found: {task_id}")


@app.post("/api/runtime/tasks")
async def runtime_task_create(req: RuntimeTaskCreateRequest):
    """Create a runtime task with optional parent link and scoped permissions."""
    try:
        task = _runtime_task_manager.create_task(
            task_id=req.task_id.strip() or None,
            parent_task_id=req.parent_task_id.strip() or None,
            permission_scope=_normalize_permission_scope(req.permission_scope),
            metadata=req.metadata,
        )
        return {"task": task.to_dict()}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/runtime/tasks/{task_id}/cancel")
async def runtime_task_cancel(task_id: str, req: RuntimeTaskCancelRequest):
    """Cancel one runtime task and optionally propagate cancellation to its descendants."""
    try:
        cancelled = _runtime_task_manager.cancel_task(
            task_id=task_id,
            reason=req.reason,
            propagate=req.propagate,
        )
        return {"cancelled_task_ids": cancelled}
    except KeyError:
        raise HTTPException(404, f"Task not found: {task_id}")


@app.post("/api/runtime/safety/evaluate")
async def runtime_safety_evaluate(req: RuntimeToolOperationRequest):
    """Evaluate one tool operation via Phase 1 safe runtime baseline."""
    request_id = req.request_id or f"tool-{uuid.uuid4().hex[:12]}"

    try:
        return rag_engine.evaluate_tool_operation(
            request_id=request_id,
            tool_name=req.tool_name,
            command_text=req.command_text,
            metadata=req.metadata,
        )
    except RuntimeError as e:
        if "safe_tool_runtime_unavailable" in str(e):
            raise HTTPException(503, "Safe tool runtime is not available.")
        raise


@app.get("/api/runtime/safety/permissions")
async def runtime_safety_permissions():
    """List pending permission requests for risky tool operations."""
    try:
        expired = rag_engine.expire_permission_requests()
        pending = rag_engine.get_pending_permissions()
        return {
            "count": len(pending),
            "pending": pending,
            "expired_count": len(expired),
        }
    except RuntimeError as e:
        if "safe_tool_runtime_unavailable" in str(e):
            raise HTTPException(503, "Safe tool runtime is not available.")
        raise


@app.get("/api/runtime/safety/executor")
async def runtime_safety_executor_status():
    """Expose runtime approval-worker health and execution summary."""
    if _approval_execution_worker is None:
        return {
            "enabled": False,
            "running": False,
            "summary": {
                "approved_total": 0,
                "pending_total": 0,
                "running": 0,
                "waiting_retry": 0,
                "completed": 0,
                "failed": 0,
                "unsupported": 0,
                "idle": 0,
            },
        }

    status = _approval_execution_worker.get_status()
    return {"enabled": True, **status}


@app.post("/api/runtime/safety/permissions/{permission_id}/resolve")
async def runtime_safety_resolve_permission(permission_id: str, req: PermissionResolveRequest):
    """Resolve a pending approval request by approving or denying it."""
    try:
        resolved = rag_engine.resolve_permission_request(
            permission_id=permission_id,
            approve=req.approve,
            actor=req.actor,
            note=req.note,
        )

        if req.approve and _approval_execution_worker is not None:
            await _approval_execution_worker.run_once()

        return {"resolved": resolved}
    except RuntimeError as e:
        if "safe_tool_runtime_unavailable" in str(e):
            raise HTTPException(503, "Safe tool runtime is not available.")
        raise
    except KeyError:
        raise HTTPException(404, f"Permission request not found: {permission_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/runtime/safety/audit")
async def runtime_safety_audit(limit: int = Query(100, ge=1, le=500)):
    """Return recent safe-runtime policy audit events."""
    try:
        events = rag_engine.get_policy_audit_events(limit=limit)
        return {"count": len(events), "events": events}
    except RuntimeError as e:
        if "safe_tool_runtime_unavailable" in str(e):
            raise HTTPException(503, "Safe tool runtime is not available.")
        raise


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

# In-memory registries for Phase 4 memory personalization telemetry.
_memory_extraction_jobs: Dict[str, Dict[str, Any]] = {}
_memory_quality_history: List[Dict[str, Any]] = []
_MEMORY_QUALITY_MAX_HISTORY = 200


def _store_memory_quality_snapshot(snapshot: Optional[Dict[str, Any]]) -> None:
    """Store one memory-quality metric snapshot in a bounded in-memory history."""
    if not snapshot:
        return
    payload = dict(snapshot)
    payload["timestamp"] = datetime.now().isoformat()
    _memory_quality_history.append(payload)
    if len(_memory_quality_history) > _MEMORY_QUALITY_MAX_HISTORY:
        _memory_quality_history.pop(0)


def _evaluate_memory_quality_batch(queries: List[str], top_k: int) -> Dict[str, Any]:
    """Run lightweight personal-context precision/recall-style evaluation."""
    per_query: List[Dict[str, Any]] = []

    for query in queries:
        try:
            base_evidence = rag_engine.search_memories(query, top_k=top_k)
            ambient_terms = _collect_ambient_terms()
            merged = _supplement_personal_evidence(
                query,
                base_evidence,
                ambient_terms=ambient_terms,
                min_score=0.0,
            )
            merged = _supplement_ambient_evidence(query, merged, min_score=0.0)

            selected = select_prompt_evidence(
                merged,
                query_analysis={"intent": "factual", "complexity": 0.35},
                is_local_model=not _is_gemini_active(),
            )
            evidence_texts = selected.get("texts", [])
            extracted = _try_extract_factual(query, evidence_texts)

            metrics = evaluate_personal_memory_quality(
                query=query,
                evidence_texts=evidence_texts,
                extracted_answer=extracted,
            )
            metrics["selected_count"] = len(evidence_texts)
            metrics["memory_prompt"] = selected.get("metrics", {})
            per_query.append(metrics)
        except Exception as e:
            per_query.append(
                {
                    "query": query,
                    "facet": "error",
                    "precision_at_k": 0.0,
                    "recall_proxy": 0.0,
                    "evidence_count": 0,
                    "relevant_count": 0,
                    "extraction_hit": False,
                    "error": str(e),
                }
            )

    avg_precision = (
        sum(float(item.get("precision_at_k", 0.0) or 0.0) for item in per_query) / max(len(per_query), 1)
    )
    recall_rate = (
        sum(float(item.get("recall_proxy", 0.0) or 0.0) for item in per_query) / max(len(per_query), 1)
    )
    extraction_rate = (
        sum(1 for item in per_query if item.get("extraction_hit")) / max(len(per_query), 1)
    )

    return {
        "sample_count": len(per_query),
        "avg_precision_at_k": round(avg_precision, 3),
        "recall_proxy_rate": round(recall_rate, 3),
        "extraction_hit_rate": round(extraction_rate, 3),
        "queries": per_query,
    }


async def _run_memory_extraction_job(
    job_id: str,
    *,
    limit: int,
    offset: int,
    dry_run: bool,
) -> None:
    """Background extraction job for bounded memory profiles (Phase 4)."""
    job = _memory_extraction_jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    started_at = time.time()

    processed = 0
    updated = 0
    failures = 0

    try:
        if not rag_engine.metadata_store:
            raise RuntimeError("metadata_store_unavailable")

        memories = rag_engine.metadata_store.get_all_memories(limit=limit, offset=offset)
        job["total"] = len(memories)

        for memory in memories:
            processed += 1
            try:
                profile = build_memory_extraction_profile(memory.content)
                if not dry_run:
                    metadata_update = {
                        "memory_extraction_profile": profile,
                        "memory_extraction_updated_at": datetime.now().isoformat(),
                    }
                    success = rag_engine.metadata_store.update_memory_metadata(
                        memory.id,
                        metadata_update,
                        merge=True,
                    )
                    if success:
                        updated += 1
                else:
                    updated += 1
            except Exception:
                failures += 1

            job["processed"] = processed
            job["updated"] = updated
            job["failures"] = failures

            # Yield control periodically for cooperative scheduling under load.
            if processed % 50 == 0:
                await asyncio.sleep(0)

        job["status"] = "completed"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["duration_ms"] = round((time.time() - started_at) * 1000, 1)
        job["finished_at"] = datetime.now().isoformat()

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

    from src.runtime.trace_analytics import build_trace_analytics

    analytics = build_trace_analytics(traces=traces, total_history_count=len(_trace_history))

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

    # Add Phase 4 memory-quality snapshots
    if _memory_quality_history:
        latest = _memory_quality_history[-1]
        window = _memory_quality_history[-20:]
        metrics["memory_quality"] = {
            "latest": latest,
            "samples": len(_memory_quality_history),
            "rolling_avg_precision_at_k": round(
                sum(
                    float(
                        x.get("avg_precision_at_k", x.get("precision_at_k", 0.0))
                        or 0.0
                    )
                    for x in window
                )
                / max(len(window), 1),
                3,
            ),
            "rolling_recall_proxy_rate": round(
                sum(
                    float(
                        x.get("recall_proxy_rate", x.get("recall_proxy", 0.0))
                        or 0.0
                    )
                    for x in window
                )
                / max(len(window), 1),
                3,
            ),
        }
    else:
        metrics["memory_quality"] = {
            "latest": None,
            "samples": 0,
            "rolling_avg_precision_at_k": 0.0,
            "rolling_recall_proxy_rate": 0.0,
        }

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
    stt_provider: Optional[str] = None       # "traditional" | "local" | "gemini"
    tts_provider: Optional[str] = None       # "traditional" | "local" | "gemini"
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
    provider: str  # "traditional" | "local" | "gemini"

@app.get("/api/ambient/voice-providers")
async def get_voice_providers():
    """Get available voice providers and current selection."""
    ambient = _get_ambient()
    traditional_stt_available = ambient._traditional_stt is not None
    traditional_tts_available = (
        ambient._traditional_tts is not None
        and ambient._traditional_tts.is_available
    )
    return {
        "stt_provider": ambient.get_stt_provider(),
        "tts_provider": ambient.get_tts_provider(),
        "gemini_available": ambient._gemini_api_key is not None,
        "traditional_stt_available": traditional_stt_available,
        "traditional_tts_available": traditional_tts_available,
        "local_stt_available": traditional_stt_available,
        "local_tts_available": traditional_tts_available,
        "gemini_stt_available": ambient._gemini_stt is not None,
        "gemini_tts_available": ambient._gemini_tts is not None,
        "supported_stt_providers": list(SUPPORTED_VOICE_PROVIDERS),
        "supported_tts_providers": list(SUPPORTED_VOICE_PROVIDERS),
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
    """Switch STT provider between traditional/local and Gemini."""
    ambient = _get_ambient()
    provider = str(req.provider or "").strip().lower()
    if provider not in SUPPORTED_VOICE_PROVIDERS:
        raise HTTPException(400, f"Provider must be one of: {', '.join(SUPPORTED_VOICE_PROVIDERS)}")

    result = ambient.set_stt_provider(provider)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result

@app.post("/api/ambient/tts-provider")
async def set_tts_provider(req: VoiceProviderRequest):
    """Switch TTS provider between traditional/local and Gemini."""
    ambient = _get_ambient()
    provider = str(req.provider or "").strip().lower()
    if provider not in SUPPORTED_VOICE_PROVIDERS:
        raise HTTPException(400, f"Provider must be one of: {', '.join(SUPPORTED_VOICE_PROVIDERS)}")

    result = ambient.set_tts_provider(provider)
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
