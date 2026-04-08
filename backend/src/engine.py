"""
Cortex Lab RAG Engine — Central Integration Point
Fine-Tuned DeepSeek-R1-7B Agentic RAG with BGE-large-1024d + CrossEncoder reranking.
Ties together all components: LLM, Embeddings, Reranker, Storage, Agents, Cache, Ingestion.
Provides a single interface for the FastAPI server.
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

from src.models import (
    CausalMemoryObject, MemoryQuery, OrchestratorResponse,
    RetrievalResult, BeliefDelta
)
from src.models.embeddings import EmbeddingModel, CrossEncoderReranker
from src.llm import LocalLLM, LLMProvider
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph
from src.retrieval.query_engine import QueryAnalyzer, QueryTransformer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.orchestrator import AgentOrchestrator
from src.ingestion import MemoryIngestionPipeline
from src.cache import MultiLevelCache
from src.runtime.task_manager import RuntimeTaskManager


class CortexRAGEngine:
    """
    The central Agentic RAG engine for Cortex Lab.
    
    This is the single entry point that the FastAPI server uses.
    It initializes and wires together all subsystems:
    - EmbeddingModel (BGE-large-en-v1.5, 1024d)
    - CrossEncoderReranker (BGE-reranker-v2-m3)
    - VectorStore (FAISS with hot/warm/cold tiers)
    - MetadataStore (DuckDB)
    - KnowledgeGraph (NetworkX)
    - LocalLLM (Fine-Tuned DeepSeek-R1-7B interface)
    - QueryAnalyzer + QueryTransformer
    - HybridRetriever (5-channel + cross-encoder reranking)
    - AgentOrchestrator (5 specialized agents + LLM routing + Self-RAG + FLARE)
    - MemoryIngestionPipeline
    - MultiLevelCache (3-level)
    """

    def __init__(self, data_dir: Optional[str] = None):
        # Resolve data path from env or backend-root default so startup is stable
        # regardless of the process working directory.
        if data_dir is None:
            data_dir = os.environ.get("CORTEX_DATA_DIR")
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.initialized = False

        # Components (initialized in init())
        self.embedding_model: Optional[EmbeddingModel] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.vector_store: Optional[VectorStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.llm: Optional[LLMProvider] = None
        self.query_analyzer: Optional[QueryAnalyzer] = None
        self.query_transformer: Optional[QueryTransformer] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.ingestion: Optional[MemoryIngestionPipeline] = None
        self.cache: Optional[MultiLevelCache] = None
        self.pageindex_store = None  # PageIndex cloud document retrieval (optional)
        self.safe_tool_runtime = None
        self.runtime_task_manager: Optional[RuntimeTaskManager] = None

        # Ambient Voice Service (lazy-initialized after RAG init)
        self.ambient_service = None

        # Session tracking
        self._current_session_id = ""
        self._session_context = ""

        # Phase 1 safe tool runtime baseline (policy + classifier + approvals)
        try:
            from src.runtime.safety import SafeToolRuntime

            self.safe_tool_runtime = SafeToolRuntime.default()
        except Exception as e:
            print(f"  ⚠ Safe tool runtime init skipped: {e}")
            self.safe_tool_runtime = None

    def init(self, model=None, tokenizer=None):
        """
        Initialize all RAG components.
        Called during server startup after the LLM model loads.
        """
        t0 = time.time()
        print("\n" + "=" * 60)
        print("  🧠 Initializing Cortex Lab RAG Engine v2.1")
        print("  📦 BGE-large-1024d + CrossEncoder + Fine-Tuned 7B + PageIndex")
        print("=" * 60)

        _has_cuda = torch is not None and torch.cuda.is_available()

        # 1. Embedding Model (BGE-large-en-v1.5, 1024d)
        print("\n[1/11] Embedding Model (BGE-large-en-v1.5)...")
        try:
            _embed_device = "cuda" if _has_cuda else "cpu"
            self.embedding_model = EmbeddingModel(device=_embed_device)
            print(f"  → {self.embedding_model.dimension}d embeddings on {_embed_device}")
        except Exception as e:
            print(f"  ⚠ Embedding model failed: {e}")

        # 2. Cross-Encoder Reranker (BGE-reranker-v2-m3)
        print("[2/11] Cross-Encoder Reranker...")
        try:
            _rerank_device = "cuda" if _has_cuda else "cpu"
            self.reranker = CrossEncoderReranker(device=_rerank_device)
        except Exception as e:
            print(f"  ⚠ Reranker failed: {e}")

        # 3. Vector Store
        print("[3/11] Vector Store...")
        try:
            dim = self.embedding_model.dimension if self.embedding_model else 1024
            self.vector_store = VectorStore(
                dimension=dim,
                data_dir=f"{self.data_dir}/vectors"
            )
        except Exception as e:
            print(f"  ⚠ Vector store failed: {e}")

        # 4. Metadata Store
        print("[4/11] Metadata Store...")
        try:
            self.metadata_store = MetadataStore(db_path=f"{self.data_dir}/cortex.duckdb")
        except Exception as e:
            print(f"  ⚠ Metadata store failed: {e}")

        # 5. Knowledge Graph
        print("[5/11] Knowledge Graph...")
        try:
            self.knowledge_graph = KnowledgeGraph(data_dir=f"{self.data_dir}/graph")
        except Exception as e:
            print(f"  ⚠ Knowledge graph failed: {e}")

        # 6. LLM Interface — Provider wraps Local + optional Gemini
        print("[6/11] LLM Interface (Fine-Tuned 7B + Gemini Provider)...")
        self.llm = LLMProvider()
        try:
            self.llm.local_llm = LocalLLM(model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"  ⚠ Local LLM init failed: {e}")

        # Try to initialize Gemini if API key is available
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".env"))
            gemini_key = os.environ.get("GOOGLE_API_KEY", "")
            if gemini_key:
                from src.llm.gemini_llm import GeminiLLM
                self.llm.gemini_llm = GeminiLLM(api_key=gemini_key)
                print("  ✓ Gemini API configured (gemini-2.5-flash)")
            else:
                print("  ℹ No GOOGLE_API_KEY — Gemini unavailable")
        except Exception as e:
            print(f"  ⚠ Gemini init skipped: {e}")

        # 7. Query Engine
        print("[7/11] Query Engine...")
        try:
            self.query_analyzer = QueryAnalyzer()
            self.query_transformer = QueryTransformer(self.llm, self.embedding_model)
        except Exception as e:
            print(f"  ⚠ Query engine failed: {e}")

        # 8. Hybrid Retriever (with cross-encoder reranker)
        print("[8/11] Hybrid Retriever (6-channel + CrossEncoder)...")
        try:
            self.hybrid_retriever = HybridRetriever(
                self.embedding_model, self.vector_store,
                self.metadata_store, self.knowledge_graph,
                reranker=self.reranker,
            )
        except Exception as e:
            print(f"  ⚠ Hybrid retriever failed: {e}")

        # 9. Agent Orchestrator (LLM routing + Self-RAG + FLARE)
        print("[9/11] Agent Orchestrator (Adaptive-RAG + Self-RAG + FLARE)...")
        try:
            self.orchestrator = AgentOrchestrator(
                self.llm, self.hybrid_retriever,
                self.query_analyzer, self.query_transformer,
                safe_tool_runtime=self.safe_tool_runtime,
                runtime_task_manager=self.runtime_task_manager,
            )
        except Exception as e:
            print(f"  ⚠ Orchestrator failed: {e}")

        # 10. Ingestion Pipeline + Cache
        print("[10/11] Ingestion Pipeline + Cache...")
        try:
            self.ingestion = MemoryIngestionPipeline(
                self.llm, self.embedding_model,
                self.vector_store, self.metadata_store,
                self.knowledge_graph
            )
        except Exception as e:
            print(f"  ⚠ Ingestion pipeline failed: {e}")
        self.cache = MultiLevelCache(self.embedding_model)

        # 11. PageIndex Store (optional cloud-based document retrieval)
        print("[11/11] PageIndex Document Store...")
        try:
            from config.pageindex_config import PAGEINDEX_CONFIG
            if PAGEINDEX_CONFIG.get("enabled", False):
                from src.storage.pageindex_store import PageIndexStore
                self.pageindex_store = PageIndexStore(
                    api_key=PAGEINDEX_CONFIG["api_key"],
                    data_dir=f"{self.data_dir}/pageindex",
                    config=PAGEINDEX_CONFIG,
                )
                # Inject into hybrid retriever as 6th channel
                self.hybrid_retriever.pageindex_store = self.pageindex_store
                # Sync any processing documents
                self.pageindex_store.sync_statuses()
                pi_stats = self.pageindex_store.get_stats()
                print(f"  📄 PageIndex enabled: {pi_stats['documents']} docs "
                      f"({pi_stats['ready_documents']} ready)")
            else:
                print("  ℹ PageIndex disabled in config")
        except ImportError as e:
            print(f"  ⚠ PageIndex SDK not installed: {e}")
            print("    → Install with: pip install pageindex")
        except Exception as e:
            print(f"  ⚠ PageIndex init failed: {e}")
            print("    → Document retrieval will use local-only channels")

        # Run tier migration on startup
        try:
            if self.vector_store:
                self.vector_store.migrate_tiers()
        except Exception as e:
            print(f"  ⚠ Tier migration skipped: {e}")

        # RAPTOR tree indexing — cluster memories at thresholds
        try:
            self._maybe_build_raptor()
        except Exception as e:
            print(f"  ⚠ RAPTOR clustering skipped: {e}")

        # Migrate junction tables for indexed topic/entity lookups (§4.1)
        try:
            if self.metadata_store:
                self.metadata_store.migrate_junction_tables()
        except Exception as e:
            print(f"  ⚠ Junction table migration skipped: {e}")

        # ── AUTO-REINDEX: ensure all memories have vectors ──────────────
        # If vector store coverage is below 80%, bulk-embed missing memories
        try:
            if self.metadata_store and self.vector_store:
                mem_count = self.metadata_store.count_memories()
                vec_count = self.vector_store.count()
                if mem_count > 0 and vec_count / mem_count < 0.8:
                    print(f"\n  ⚠ Vector coverage low: {vec_count}/{mem_count} ({vec_count/mem_count:.0%})")
                    print(f"  🔄 Auto-reindexing missing memories...")
                    self._reindex_missing_vectors()
        except Exception as e:
            print(f"  ⚠ Auto-reindex skipped: {e}")

        # ── PRE-BUILD proposition index to avoid first-query delay ──────
        # Skip on startup — proposition channel has low weight (0.10) and
        # will rebuild lazily on first query. Startup speed is more important.
        # try:
        #     if self.hybrid_retriever and self.metadata_store:
        #         self.hybrid_retriever._rebuild_proposition_index()
        # except Exception as e:
        #     print(f"  ⚠ Proposition index pre-build skipped: {e}")
        print("  ℹ Proposition index will build lazily on first query")

        self.initialized = True
        elapsed = time.time() - t0
        print(f"\n  ✅ RAG Engine v2.1 ready in {elapsed:.1f}s")

        # Print stats safely
        try:
            pi_status = "enabled" if self.pageindex_store else "disabled"
            mem_c = self.metadata_store.count_memories() if self.metadata_store else 0
            vec_c = self.vector_store.count() if self.vector_store else 0
            graph_s = self.knowledge_graph.get_stats() if self.knowledge_graph else {}
            print(f"  📊 Memories: {mem_c} | "
                  f"Vectors: {vec_c} | "
                  f"Graph: {graph_s} | "
                  f"PageIndex: {pi_status}")
        except Exception as e:
            print(f"  📊 Stats unavailable: {e}")
        print("=" * 60 + "\n")

        # Initialize Ambient Voice Service (lazy — models load on first start)
        try:
            import os
            gemini_key_for_voice = os.environ.get("GOOGLE_API_KEY", "") or None
            from src.ambient import AmbientService
            self.ambient_service = AmbientService(
                ingestion_pipeline=self.ingestion,
                data_dir=self.data_dir,
                gemini_api_key=gemini_key_for_voice,
            )
            print("  🎙️  Ambient voice service initialized (idle, ready to start)")
            if gemini_key_for_voice:
                print("     Gemini STT/TTS available as alternative provider")
        except Exception as e:
            print(f"  ⚠ Ambient service init skipped: {e}")
            self.ambient_service = None

    def set_model(self, model, tokenizer):
        """Update LLM reference (called when model finishes loading)."""
        if self.llm:
            self.llm.set_model(model, tokenizer)

    # ─── Helpers ───────────────────────────────────────────────────────

    def _reindex_missing_vectors(self):
        """Bulk-embed memories that are in DuckDB but not in the vector store."""
        import numpy as np
        all_memory_texts = self.metadata_store.get_memory_texts(limit=5000)
        existing_ids = set(self.vector_store.vectors.keys())

        missing = [(mid, content) for mid, content in all_memory_texts
                    if mid not in existing_ids and len(content.strip()) > 10]

        if not missing:
            print(f"  ✓ All memories already vectorized")
            return

        print(f"  📊 Vectorizing {len(missing)} missing memories...")

        batch_size = 32
        added = 0
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            contents = [content for _, content in batch]

            embeddings = self.embedding_model.embed_batch(contents)

            for j, (mid, content) in enumerate(batch):
                from datetime import datetime
                self.vector_store.add(mid, embeddings[j], datetime.now())
                added += 1

            pct = min((i + len(batch)) / len(missing) * 100, 100)
            print(f"    Indexed {i + len(batch)}/{len(missing)} ({pct:.0f}%)", end="\r")

        print(f"\n  ✓ Indexed {added} new vectors (total: {self.vector_store.count()})")
        self.vector_store.save()

    # RAPTOR thresholds — build clusters when memory count crosses these
    _RAPTOR_THRESHOLDS = [50, 200, 1000]
    _raptor_last_built_at = 0

    def _maybe_build_raptor(self):
        """Trigger RAPTOR clustering when memory count crosses thresholds.
        
        Skips if RAPTOR summaries already exist for the current threshold
        (avoids re-generating on every server restart).
        Runs in a background thread to avoid blocking server startup.
        """
        mem_count = self.metadata_store.count_memories()

        # Check if RAPTOR summaries already exist
        all_memories = self.metadata_store.get_all_memories(limit=5000)
        existing_raptor_count = sum(1 for m in all_memories if m.raptor_level > 0)

        for threshold in self._RAPTOR_THRESHOLDS:
            if mem_count >= threshold > self._raptor_last_built_at:
                # Skip if we already have enough RAPTOR summaries for this tier
                expected_min = max(2, threshold // 25)
                if existing_raptor_count >= expected_min:
                    print(f"\n  🌳 RAPTOR: {existing_raptor_count} summaries already exist, skipping rebuild")
                    self._raptor_last_built_at = mem_count
                    break

                print(f"\n  🌳 RAPTOR threshold {threshold} reached ({mem_count} memories)")
                print(f"  🌳 Building clusters in background thread...")

                import threading
                def _build():
                    try:
                        self.ingestion.build_raptor_clusters(
                            min_cluster_size=max(3, threshold // 20),
                            max_clusters=min(30, threshold // 5),
                        )
                    except Exception as e:
                        print(f"  ⚠ RAPTOR background build failed: {e}")

                t = threading.Thread(target=_build, daemon=True)
                t.start()
                self._raptor_last_built_at = mem_count
                break

    @staticmethod
    def _is_meaningful_content(text: str) -> bool:
        """Check if content is substantial enough to store as a memory.
        Uses factual density scoring: only net-positive content is ingested.

        Rejected: greetings, pure questions, very short filler.
        Accepted: informational statements worth remembering.
        """
        import re

        stripped = text.strip()
        lower = stripped.lower().rstrip("!?.,")
        words = lower.split()

        # ── Immediate rejects ──
        if len(lower) < 8 or len(words) < 3:
            return False

        TRIVIAL = {
            "hi", "hey", "hello", "hii", "hiii", "yo", "sup", "howdy",
            "good morning", "good evening", "good night", "good afternoon",
            "thanks", "thank you", "ok", "okay", "bye", "goodbye",
            "yes", "no", "yeah", "nah", "sure", "hmm", "hm", "hmmmm",
            "what", "who", "why", "how", "when", "where",
        }
        if lower in TRIVIAL:
            return False

        # ── Strip greeting prefix ──
        core = lower
        for gp in ("hey ", "hi ", "hello ", "hii ", "yo ", "sup ",
                    "hey, ", "hi, ", "hello, ", "okay ", "ok ",
                    "hey there ", "hi there ", "hello there ",
                    "good morning ", "good evening ", "good afternoon "):
            if core.startswith(gp):
                core = core[len(gp):].strip()
                break

        # ── Factual density scoring ──
        score = 0.0

        # +1.5 max for word count (diminishing returns)
        score += min(len(words) / 15.0, 1.5)

        # +2 for strong informational signals
        INFO_SIGNALS = [
            "i learned", "i built", "i created", "i worked on", "i made",
            "i developed", "i designed", "i implemented", "i think",
            "i believe", "i decided", "i realized", "i discovered",
            "my project", "my experience", "my work", "we built",
            "we created", "we developed", "today i", "yesterday i",
        ]
        if any(sig in lower for sig in INFO_SIGNALS):
            score += 2.0

        # +0.5 for personal pronouns (first person = factual content)
        if re.search(r'\b(i|my|we|our)\b', lower):
            score += 0.5

        # -3 for question patterns (strong negative signal)
        QUESTION_STARTS = (
            "what ", "what's ", "who ", "who's ", "where ", "where's ",
            "when ", "when's ", "how ", "how's ", "why ", "why's ",
            "which ", "whose ", "whom ", "is ", "are ", "was ", "were ",
            "do ", "does ", "did ", "can ", "could ", "will ", "would ",
            "should ", "shall ", "have ", "has ", "had ",
            "tell me", "list ", "describe ", "summarize ", "explain ",
            "show me", "give me", "find ", "search ",
        )
        is_question = (
            any(core.startswith(q) for q in QUESTION_STARTS)
            or any(lower.startswith(q) for q in QUESTION_STARTS)
            or lower.rstrip().endswith("?")
            or bool(re.search(
                r'\b(what|who|where|when|how|why|which)\b.{0,50}'
                r'\b(is|are|was|were|do|does|did|my|the|about)\b',
                core
            ))
        )
        if is_question:
            score -= 3.0

        # -1 for very short content (<60 chars) without info signals
        if len(stripped) < 60 and score < 1.0:
            score -= 1.0

        return score >= 0.0

    async def _background_ingest(self, content: str, session_id: str,
                                  session_context: str = ""):
        """Background ingestion task (§5.1) — enriches memory without blocking chat."""
        try:
            memory = await self.ingestion.ingest(
                content=content,
                session_id=session_id,
                source="chat",
                session_context=session_context,
            )
            # Invalidate retriever caches after new ingestion
            self.hybrid_retriever.invalidate_caches()
            # Store user conversation turn
            self.metadata_store.store_conversation_turn(
                session_id=session_id,
                role="user",
                content=content,
                memory_id=memory.id if memory else None,
            )
        except Exception as e:
            print(f"  ⚠ Background ingestion error: {e}")

    # ─── RAG-Enhanced Chat ───────────────────────────────────────────────

    async def rag_chat(self, user_message: str, session_id: str = "",
                        conversation_history: List[Dict] = None) -> Dict:
        """
        Main RAG-enhanced chat endpoint.
        1. Ingest user message as memory
        2. Check cache
        3. Run agentic RAG pipeline
        4. Return enhanced response with evidence
        """
        if not self.initialized:
            return {"answer": "RAG system is still initializing...", "evidence": []}

        t0 = time.time()

        # Set session
        if not session_id:
            session_id = f"session-{int(time.time())}"
        self._current_session_id = session_id

        # Build session context from history
        # Dynamic sizing: Gemini has 1M token context → send more history
        session_context = ""
        if conversation_history:
            is_gemini = (hasattr(self.llm, 'provider') and self.llm.provider == 'gemini')
            history_limit = 30 if is_gemini else 6   # 15 vs 3 exchanges
            char_limit = 2000 if is_gemini else 200   # More per message for Gemini
            recent = conversation_history[-history_limit:]
            session_context = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')[:char_limit]}"
                for m in recent
            )

        # NOTE: Chat queries are NOT ingested as memories (§5.1 revised).
        # Memories should only be created through:
        #   1. Manual "Add Memory" button in Memory Browser
        #   2. Ambient STT pipeline (voice-captured knowledge)
        #   3. /api/memories/ingest endpoint (programmatic)
        # Ingesting every chat query pollutes the vector store with questions
        # that later surface as low-quality evidence in retrieval.

        # 2. Check cache (provider-aware: Gemini vs Local have separate caches)
        _provider = getattr(self.llm, 'provider', 'local') if self.llm else 'local'
        cached, cache_level = self.cache.get(user_message, provider=_provider)
        if cached:
            print(f"  ⚡ Cache hit ({cache_level}, provider={_provider})")
            cached["cache_hit"] = True
            cached["cache_level"] = cache_level
            return cached

        # 3. Run orchestrator
        if not self.orchestrator:
            return {
                "answer": "",
                "thinking": "RAG retrieval subsystems are not available. No memories or evidence found in the database.",
                "evidence": [],
                "agents_used": [],
                "confidence": 0.0,
                "reasoning_trace": "Orchestrator unavailable — subsystems not initialized",
                "query_analysis": {"intent": "unknown", "complexity": 0, "routing": "unknown"},
                "processing_time_ms": round((time.time() - t0) * 1000, 1),
                "cache_hit": False,
                "pipeline_trace": None,
            }

        response = await self.orchestrator.process(user_message, session_context)

        # 4. Format result
        result = {
            "answer": response.answer,
            "thinking": response.thinking,
            "evidence": [
                {
                    # PageIndex evidence gets more space (document answers are longer)
                    "content": e.memory.content[:2000] if e.memory.source == "pageindex" else e.memory.content[:600],
                    "score": round(e.score, 3),
                    "channel": e.channel,
                    "timestamp": e.memory.timestamp.isoformat(),
                    "memory_type": e.memory.memory_type.value,
                    "emotion": e.memory.emotion.value,
                    "entities": e.memory.entities[:5],
                }
                for e in response.evidence[:10]
            ],
            "agents_used": response.agents_used,
            "confidence": round(response.confidence, 3),
            "reasoning_trace": response.reasoning_trace,
            "query_analysis": {
                "intent": response.query_analysis.intent.value if response.query_analysis else "unknown",
                "complexity": round(response.query_analysis.complexity, 2) if response.query_analysis else 0,
                "routing": response.query_analysis.routing.value if response.query_analysis else "unknown",
            },
            "processing_time_ms": round(response.processing_time_ms, 1),
            "cache_hit": False,
            "pipeline_trace": response.pipeline_trace.to_dict() if response.pipeline_trace else None,
        }

        # 5. Cache result (provider-aware)
        self.cache.set(user_message, result, provider=_provider)

        # 6. Store conversation turns (lightweight — just for history, NOT as memories)
        self.metadata_store.store_conversation_turn(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        self.metadata_store.store_conversation_turn(
            session_id=session_id,
            role="assistant",
            content=response.answer,
            thinking=response.thinking,
        )

        return result

    # ─── Memory Management ───────────────────────────────────────────────

    async def rag_retrieve(self, user_message: str, session_id: str = "",
                            conversation_history: List[Dict] = None) -> Dict:
        """
        RAG retrieval-only pipeline (no final generation).
        Used for streaming mode: retrieves evidence + thinking, then lets server stream.
        Uses orchestrator.retrieve_only() to skip the expensive LLM generation step —
        the server will stream the final answer token-by-token.
        """
        if not self.initialized:
            return {"answer": "", "evidence": [], "thinking": "RAG system initializing..."}

        t0 = time.time()

        if not session_id:
            session_id = f"session-{int(time.time())}"
        self._current_session_id = session_id

        session_context = ""
        if conversation_history:
            is_gemini = (hasattr(self.llm, 'provider') and self.llm.provider == 'gemini')
            history_limit = 30 if is_gemini else 6
            char_limit = 2000 if is_gemini else 200
            recent = conversation_history[-history_limit:]
            session_context = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')[:char_limit]}"
                for m in recent
            )

        # NOTE: Chat queries are NOT ingested as memories (§5.1 revised).
        # Only manual/ambient/API ingestion creates memories.
        # See rag_chat() comment for rationale.

        # Check cache (provider-aware)
        _provider = getattr(self.llm, 'provider', 'local') if self.llm else 'local'
        cached, cache_level = self.cache.get(user_message, provider=_provider)
        if cached:
            cached["cache_hit"] = True
            cached["cache_level"] = cache_level
            return cached

        # Run the retrieve-only orchestrator (no LLM generation — much faster)
        if not self.orchestrator:
            return {
                "answer": "",
                "thinking": "No memories or evidence found in the database. The RAG retrieval subsystems are not initialized.",
                "evidence": [],
                "agents_used": [],
                "confidence": 0.0,
                "reasoning_trace": "Orchestrator unavailable — subsystems not initialized",
                "query_analysis": {"intent": "unknown", "complexity": 0, "routing": "unknown"},
                "processing_time_ms": round((time.time() - t0) * 1000, 1),
                "cache_hit": False,
                "pipeline_trace": None,
            }

        response = await self.orchestrator.retrieve_only(user_message, session_context)

        # Format evidence (no final answer — caller will stream it)
        # Send up to 20 evidence items so the streaming code has enough context
        result = {
            "answer": "",  # Empty — will be streamed by server
            "thinking": response.thinking,
            "evidence": [
                {
                    # PageIndex evidence gets more space (document answers are longer)
                    "content": e.memory.content[:2000] if e.memory.source == "pageindex" else e.memory.content[:600],
                    "score": round(e.score, 3),
                    "channel": e.channel,
                    "timestamp": e.memory.timestamp.isoformat(),
                    "memory_type": e.memory.memory_type.value,
                    "emotion": e.memory.emotion.value,
                    "entities": e.memory.entities[:5],
                }
                for e in response.evidence[:20]
            ],
            "agents_used": response.agents_used,
            "confidence": round(response.confidence, 3),
            "reasoning_trace": response.reasoning_trace,
            "query_analysis": {
                "intent": response.query_analysis.intent.value if response.query_analysis else "unknown",
                "complexity": round(response.query_analysis.complexity, 2) if response.query_analysis else 0,
                "routing": response.query_analysis.routing.value if response.query_analysis else "unknown",
            },
            "processing_time_ms": round(response.processing_time_ms, 1),
            "cache_hit": False,
            "pipeline_trace": response.pipeline_trace.to_dict() if response.pipeline_trace else None,
        }

        # User turn is stored in background ingestion; assistant turn stored after streaming

        return result

    # ─── Memory Management (continued) ───────────────────────────────────

    async def ingest_memory(self, content: str, source: str = "manual",
                             session_id: str = "") -> Dict:
        """Manually ingest a memory."""
        if not self.initialized:
            return {"error": "RAG system not initialized"}
        if not self.ingestion:
            return {"error": "Ingestion pipeline not available"}

        memory = await self.ingestion.ingest(
            content=content, session_id=session_id, source=source
        )

        # Invalidate caches (new memory might change future answers)
        self.cache.invalidate_topic(memory.topics[0] if memory.topics else "")
        if self.hybrid_retriever:
            self.hybrid_retriever.invalidate_caches()

        return memory.to_dict()

    def get_memories(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get all memories with pagination."""
        if not self.initialized:
            return []
        memories = self.metadata_store.get_all_memories(limit=limit, offset=offset)
        return [m.to_dict() for m in memories]

    def search_memories(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search memories by content similarity."""
        if not self.initialized:
            return []
        embedding = self.embedding_model.embed(query)
        results = self.vector_store.search(embedding, top_k=top_k)

        memories = []
        for mid, score in results:
            mem = self.metadata_store.get_memory(mid)
            if mem:
                d = mem.to_dict()
                d["score"] = round(score, 3)
                memories.append(d)
        return memories

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self.initialized:
            return False
        self.metadata_store.delete_memory(memory_id)
        self.vector_store.delete(memory_id)
        return True

    # ─── Knowledge Graph ─────────────────────────────────────────────────

    def get_graph_data(self) -> Dict:
        """Get graph data for visualization."""
        if not self.initialized or not self.knowledge_graph:
            return {"nodes": [], "edges": []}
        return self.knowledge_graph.get_graph_data()

    def get_entities(self, limit: int = 100) -> List[Dict]:
        """Get all entities."""
        if not self.initialized or not self.metadata_store:
            return []
        return self.metadata_store.get_entities(limit=limit)

    def get_belief_deltas(self, limit: int = 50) -> List[Dict]:
        """Get detected belief evolution events."""
        if not self.initialized:
            return []
        try:
            return self.metadata_store.get_belief_deltas(limit=limit)
        except Exception as e:
            print(f"  ⚠ Belief deltas retrieval error: {e}")
            return []

    def get_community_summaries(self) -> List[Dict]:
        """Get GraphRAG community summaries."""
        if not self.initialized:
            return []
        return self.knowledge_graph.get_community_summaries()

    # ─── System Stats ────────────────────────────────────────────────────

    def get_rag_stats(self) -> Dict:
        """Get comprehensive RAG system statistics."""
        if not self.initialized:
            return {"status": "not_initialized"}

        return {
            "status": "ready",
            "memories": self.metadata_store.get_stats() if self.metadata_store else {},
            "vectors": self.vector_store.get_stats() if self.vector_store else {},
            "graph": self.knowledge_graph.get_stats() if self.knowledge_graph else {},
            "cache": self.cache.get_stats() if self.cache else {},
            "llm": self.llm.get_stats() if self.llm else {},
        }

    def get_tool_contracts(self) -> List[Dict]:
        """Return Phase 0 core tool contracts for runtime governance.

        This accessor allows runtime and API layers to inspect which existing
        engine operations are represented as typed tool contracts.
        """
        from src.runtime.tool_catalog import build_core_tool_catalog_dicts

        return build_core_tool_catalog_dicts()

    def evaluate_tool_operation(
        self,
        request_id: str,
        tool_name: str,
        command_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a tool operation against safe runtime policy and queue rules."""
        if self.safe_tool_runtime is None:
            raise RuntimeError("safe_tool_runtime_unavailable")

        result = self.safe_tool_runtime.evaluate_tool_operation(
            request_id=request_id,
            tool_name=tool_name,
            command_text=command_text,
            metadata=metadata,
        )

        return {
            "request_id": request_id,
            "tool_name": tool_name,
            "decision": {
                "effect": result.decision.effect.value,
                "reason": result.decision.reason,
                "rule_id": result.decision.rule_id,
                "requires_human_approval": result.decision.requires_human_approval,
            },
            "stop_reason": result.stop_reason.value if result.stop_reason else None,
            "dangerous_signals": [
                {
                    "tool_name": signal.tool_name,
                    "matched_pattern": signal.matched_pattern,
                    "severity": signal.severity.value,
                }
                for signal in result.dangerous_signals
            ],
            "permission_request": result.permission_request.to_dict() if result.permission_request else None,
            "audit_event": result.audit_event.to_dict(),
        }

    def get_pending_permissions(self) -> List[Dict[str, Any]]:
        if self.safe_tool_runtime is None:
            raise RuntimeError("safe_tool_runtime_unavailable")
        return [
            request.to_dict()
            for request in self.safe_tool_runtime.list_pending_permissions()
        ]

    def get_permission_request(self, permission_id: str) -> Optional[Dict[str, Any]]:
        if self.safe_tool_runtime is None:
            raise RuntimeError("safe_tool_runtime_unavailable")
        request = self.safe_tool_runtime.permission_queue.get(permission_id)
        return request.to_dict() if request else None

    def expire_permission_requests(self) -> List[Dict[str, Any]]:
        if self.safe_tool_runtime is None:
            raise RuntimeError("safe_tool_runtime_unavailable")
        return [
            request.to_dict()
            for request in self.safe_tool_runtime.expire_permission_requests()
        ]

    def resolve_permission_request(
        self,
        permission_id: str,
        approve: bool,
        actor: str,
        note: str = "",
    ) -> Dict[str, Any]:
        if self.safe_tool_runtime is None:
            raise RuntimeError("safe_tool_runtime_unavailable")
        request = self.safe_tool_runtime.resolve_permission_request(
            permission_id=permission_id,
            approve=approve,
            actor=actor,
            note=note,
        )
        return request.to_dict()

    def get_policy_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.safe_tool_runtime is None:
            raise RuntimeError("safe_tool_runtime_unavailable")
        return [
            event.to_dict()
            for event in self.safe_tool_runtime.list_audit_events(limit=limit)
        ]

    def set_runtime_task_manager(self, runtime_task_manager: RuntimeTaskManager) -> None:
        """Attach shared runtime task manager used for coordinator/subagent tracking."""
        self.runtime_task_manager = runtime_task_manager
        if self.orchestrator is not None:
            self.orchestrator.runtime_task_manager = runtime_task_manager

    # ─── Persistence ─────────────────────────────────────────────────────

    def save_all(self):
        """Persist all data to disk."""
        if not self.initialized:
            return
        print("\n💾 Saving all data...")
        if self.vector_store:
            self.vector_store.save()
        if self.knowledge_graph:
            self.knowledge_graph.save()
        print("  ✅ All data saved\n")

    def shutdown(self):
        """Graceful shutdown."""
        # Stop ambient voice service
        if self.ambient_service:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.ambient_service.stop())
                else:
                    asyncio.run(self.ambient_service.stop())
            except Exception as e:
                print(f"  ⚠ Ambient stop error: {e}")

        self.save_all()
        if self.metadata_store:
            self.metadata_store.close()


# Singleton instance
rag_engine = CortexRAGEngine()
