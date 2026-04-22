"""
Centralized RAG Configuration for Cortex Lab
All magic numbers, thresholds, and tunable parameters in one place.

Usage:
    from config.rag_config import RAG_CONFIG
    threshold = RAG_CONFIG["routing"]["complexity_low"]
"""

# ─── Routing & Query Intelligence ────────────────────────────────────────────

ROUTING = {
    "complexity_low": 0.30,          # Below → NO_RETRIEVAL
    "complexity_medium": 0.60,       # Below → SINGLE_STEP, above → MULTI_STEP
    "llm_routing_min": 0.35,         # LLM routing only when complexity in [min, max]
    "llm_routing_max": 0.65,
}

# ─── CRAG Quality Evaluation ────────────────────────────────────────────────

CRAG = {
    "correct_threshold": 0.55,       # Above → CORRECT verdict
    "ambiguous_threshold": 0.30,     # Above → AMBIGUOUS, below → INCORRECT
    "ambiguous_confidence_penalty": 0.85,  # Multiply confidence by this for AMBIGUOUS
    "incorrect_confidence_penalty": 0.55,  # Multiply confidence by this for INCORRECT
}

# ─── Self-RAG ───────────────────────────────────────────────────────────────

SELF_RAG = {
    "trigger_confidence": 0.55,      # Only run if confidence < this
    "min_answer_length": 20,         # Skip if answer shorter than this
    "accept_threshold": 7.0,         # ISREL+ISSUP+ISUSE avg >= this → ACCEPT
    "revise_threshold": 5.0,         # avg >= this → REVISE, below → REJECT
}

# ─── FLARE Active Retrieval ─────────────────────────────────────────────────

FLARE = {
    "trigger_confidence": 0.40,      # Only run if confidence < this
    "min_answer_length": 20,
    "max_retrieval_iterations": 2,   # Max FLARE retrieval rounds
    "max_new_evidence": 3,           # Max new evidence items to add
}

# ─── Retrieval ──────────────────────────────────────────────────────────────

RETRIEVAL = {
    "default_top_k": 20,             # Default top-K for retrieval
    "rrf_k": 60,                     # RRF constant K
    "dense_weight": 0.35,
    "sparse_weight": 0.25,
    "graph_weight": 0.20,
    "temporal_weight": 0.10,
    # Weights when no PageIndex documents exist
    "dense_weight_local": 0.40,
    "sparse_weight_local": 0.30,
    "graph_weight_local": 0.20,
    "temporal_weight_local": 0.10,
}

# ─── Ingestion ──────────────────────────────────────────────────────────────

INGESTION = {
    "max_memory_length": 10000,      # Max chars per memory
    "dedup_threshold": 0.95,         # Cosine similarity threshold for deduplication
    "max_entities": 15,              # Max entities extracted per memory
    "max_propositions": 12,          # Max atomic propositions per memory
    "max_topics": 5,                 # Max topics per memory
}

# ─── Belief Evolution ───────────────────────────────────────────────────────

BELIEF = {
    "similarity_threshold": 0.75,    # Min cosine similarity for related memories
    "min_time_gap_seconds": 86400,   # 1 day minimum between old and new belief
}

# ─── BM25 Sparse Retrieval ─────────────────────────────────────────────────

BM25 = {
    "k1": 1.5,                       # Term frequency saturation
    "b": 0.75,                       # Length normalization
}

# ─── LLM Generation ────────────────────────────────────────────────────────

LLM = {
    "max_context_budget": 4096,      # Max input tokens
    "default_max_tokens": 512,
    "default_temperature": 0.3,
    "default_top_p": 0.9,
    "repetition_penalty": 1.15,
    "vram_defrag_interval": 100,     # Clear CUDA cache every N calls
}

# ─── Evidence & Streaming ──────────────────────────────────────────────────

EVIDENCE = {
    "max_evidence_items": 20,        # Max evidence items passed to streaming
    "max_evidence_text_chars": 1500, # Max chars per evidence text
    "min_evidence_length": 50,       # Skip evidence shorter than this
}

# ─── Session Context ───────────────────────────────────────────────────────

SESSION = {
    "default_history_messages": 6,   # Last N messages for local model context
    "gemini_history_messages": 30,   # Last N messages for Gemini (larger context)
    "message_truncate_chars": 200,   # Max chars per message in context (local)
    "gemini_message_truncate_chars": 2000,  # Max chars per message (Gemini)
}

# ─── API Authentication ────────────────────────────────────────────────────

AUTH = {
    "enabled": False,                # Set True to require bearer token
    "token_env_var": "CORTEX_API_KEY",  # Env var name for the API key
}

# ─── RAPTOR Clustering ──────────────────────────────────────────────────────

RAPTOR = {
    "min_cluster_size": 5,
    "max_clusters": 20,
    "summary_raptor_level": 1,
    "summary_importance": 0.8,
}

# ─── Aggregate config dict for easy import ──────────────────────────────────

RAG_CONFIG = {
    "routing": ROUTING,
    "crag": CRAG,
    "self_rag": SELF_RAG,
    "flare": FLARE,
    "retrieval": RETRIEVAL,
    "ingestion": INGESTION,
    "belief": BELIEF,
    "bm25": BM25,
    "llm": LLM,
    "evidence": EVIDENCE,
    "session": SESSION,
    "auth": AUTH,
    "raptor": RAPTOR,
}
