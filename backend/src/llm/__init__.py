"""
Local LLM Interface for Cortex Lab — Fine-Tuned Model Integration
Interfaces with the fine-tuned DeepSeek-R1-7B model.
Leverages all 15 training stages:
  Stage 1: Faithfulness (grounded generation)
  Stage 2: Agentic routing (structured JSON intent classification)
  Stage 3: Causal reasoning
  Stage 4: Self-RAG critique tokens (ISREL/ISSUP/ISUSE)
  Stage 5: Belief evolution tracking
  Stage 6: Summarization
  Stage 7: Dialogue coherence
  Stage 8: Long-context handling
  Stage 9: DPO preference alignment
  Stage 10: User-style adaptation
  Stage 11: ORPO alignment
  Stage 12: RAFT distractor-aware generation
  Stage 13: Function calling / tool use
  Stage 14: Rejection fine-tuning (knowing limits)
  Stage 15: SPIN self-play improvement
"""

import asyncio
import time
import re
import json
from typing import Optional, Dict, List, Any
from src.prompts import PromptBuilder, sanitize
try:
    import torch
except ImportError:
    torch = None

# Stop patterns that indicate the model is hallucinating new turns
_LLM_STOP_PATTERNS = [
    "\nUser:", "\nuser:", "\nHuman:", "\nhuman:",
    "\nQ:", "\nA:", "\n\nUser ", "\nQuestion:",
]

# Unified hallucination pattern set (merged from _validate_or_extract + _strip)
_HALLUC_PATTERNS = frozenset([
    "life purpose", "deep work", "stay focused in meetings",
    "rest is an input", "difficult lesson", "performance, not a reward",
    "consistent over years", "communication skills", "deliberate practice",
    "my view on", "improved answer", "what led me to",
    "## summary of", "### key findings", "### key insight",
    "limited relevant memories", "partial information",
    "emotional trajectory", "belief evolution",
    "had a difficult moment", "key lesson from",
    "moving cities", "city-building", "modern technology",
    "clarity of scope", "clarity requires constraints", "scope creep",
    "systems matter more than goals", "the relationship is more complex than",
    "the intersection of", "deep work patterns", "the bottleneck has shifted",
    "the timeline for meaningful", "sporadic bursts", "cumulative insight",
    "your thinking journey", "lived experiences", "key moments",
    "strong empirical evidence", "belief about relationships evolution",
    "reflecting on my relationship with my mentor",
    "emotional resilience", "emotion evolution", "personal growth",
    "according to research on personal growth",
    "excited \u2014 anxious \u2014 drained", "emotion timeline:",
    "emotion timeline", "emotional trajectory",
    "drained by the lack of follow-through", "still processing this",
    "confidence: high", "confidence: medium",
    "based on strong empirical", "comprehensive answer to your question",
    "here's a comprehensive answer", "here is a comprehensive answer",
    "here's the revised answer", "here's what your beliefs",
    "revised answer focused on",
    "[name]", "s-repository]", "\u2022 s-repository",
    "technical documentation of all projects",
    "causal link:", "step 1 [memory", "\u2192 step 2", "\u2192 step 3", "key findings:",
    "had an unexpected complication", "strongly motivated thr",
    "here's your answer:", "here is your answer:",
    "here's a decomposed analysis",
])


def _truncate_at_stop(text: str) -> str:
    """Truncate at the first occurrence of any stop pattern."""
    earliest = len(text)
    for pattern in _LLM_STOP_PATTERNS:
        pos = text.find(pattern)
        if pos != -1 and pos < earliest:
            earliest = pos
    return text[:earliest].strip()


class LocalLLM:
    """
    Interface to the fine-tuned DeepSeek-R1-7B model.
    Provides structured LLM calls leveraging fine-tuned capabilities:

    Core methods:
    - generate(): General text generation
    - classify(): Quick classification with constrained output
    - extract_json(): Parse JSON from model output
    - summarize(): Concise summarization (Stage 6)

    Fine-tuning-aware methods:
    - route_query(): Structured JSON routing (Stage 2 Agentic)
    - self_rag_critique(): ISREL/ISSUP/ISUSE token generation (Stage 4)
    - causal_reason(): Cause-effect chain analysis (Stage 3)
    - detect_belief_change(): Belief evolution tracking (Stage 5)
    - generate_faithful(): Grounded generation with citations (Stage 1)
    - call_function(): Tool/function calling (Stage 13)
    - raft_generate(): Distractor-aware generation (Stage 12)
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._call_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0
        self._cache_clear_interval = 100  # Clear VRAM fragmentation every N calls (§6.4)

    def set_model(self, model, tokenizer):
        """Set model reference (called after model loads in server.py)."""
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.3, top_p: float = 0.9,
                 stop_sequences: Optional[list] = None,
                 structured: bool = False) -> str:
        """Generate text from the model with stop-pattern safety.
        Args:
            structured: If True, disables repetition_penalty for JSON/classification outputs (§1.8)
        """
        if self.model is None or self.tokenizer is None:
            return self._fallback_generate(prompt)

        t0 = time.time()
        # Dynamic max_length based on max_tokens (§1.3)
        context_budget = min(3072 + max_tokens, 4096)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_budget)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        # Build stop token IDs
        eos_ids = [self.tokenizer.eos_token_id]
        for stop_str in ["User:", "<|im_end|>", "<|endoftext|>"]:
            try:
                ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if ids:
                    eos_ids.append(ids[0])
            except Exception:
                pass

        # Disable repetition penalty for structured outputs (§1.8)
        rep_penalty = 1.0 if structured else 1.15

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 2048),
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                repetition_penalty=rep_penalty,
            )

        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        # Strip thinking tags if present
        if "<think>" in generated:
            think_end = generated.find("</think>")
            if think_end > -1:
                generated = generated[think_end + len("</think>"):].strip()

        # Apply custom stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated:
                    generated = generated[:generated.index(seq)]

        # Always truncate at hallucinated conversation continuations
        generated = _truncate_at_stop(generated)

        # Strip any raw model tokens that leaked through
        for token in ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                       "<|im_sep|>", "<｜end▁of▁sentence｜>",
                       "<｜User｜>", "<｜Assistant｜>"]:
            generated = generated.replace(token, "")
        # Also catch partial/malformed ChatML tokens like <|im_start|user|...>
        generated = re.sub(r'<\|im_start\|[^>]*>', '', generated)
        generated = re.sub(r'<\|im_end\|[^>]*>', '', generated)
        generated = generated.strip()

        elapsed_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_tokens += input_len + len(self.tokenizer.encode(generated))
        self._total_time_ms += elapsed_ms

        # Periodic VRAM defragmentation (§6.4)
        if self._call_count % self._cache_clear_interval == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated

    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async wrapper for generate() — unblocks the event loop (§11.4).
        Use this from all async handlers (FastAPI routes, agents, orchestrator)."""
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    def generate_with_retry(self, prompt: str, max_retries: int = 2, min_length: int = 5, **kwargs) -> str:
        """Generate with retry on empty/garbage output (§13.2).
        Retries up to max_retries times if output is too short or empty."""
        for attempt in range(max_retries + 1):
            result = self.generate(prompt, **kwargs)
            if result.strip() and len(result.strip()) >= min_length:
                return result
            if attempt < max_retries:
                # Increase temperature slightly on retry to escape local minimum
                kwargs["temperature"] = min((kwargs.get("temperature", 0.3)) + 0.1, 0.8)
        # All retries failed — return last result or fallback
        return result if result.strip() else self._fallback_generate(prompt)

    def classify(self, prompt: str, options: list, default: str = "") -> str:
        """Quick classification with constrained output."""
        full_prompt = f"{prompt}\n\nChoose EXACTLY ONE from: {', '.join(options)}\nAnswer:"
        result = self.generate(full_prompt, max_tokens=20, temperature=0.1)
        result_lower = result.strip().lower()

        for opt in options:
            if opt.lower() in result_lower:
                return opt
        return default or options[0]

    def extract_json(self, prompt: str, max_tokens: int = 256) -> dict:
        """Generate and parse JSON output."""
        result = self.generate(prompt + "\n\nOutput valid JSON only:",
                               max_tokens=max_tokens, temperature=0.1, structured=True)
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except json.JSONDecodeError:
            pass
        return {}

    def summarize(self, text: str, max_length: int = 100) -> str:
        """Generate a concise summary (leverages Stage 6 training)."""
        prompt = f"""Summarize the following text concisely, preserving key entities, dates, and causal relationships.
Keep it under {max_length} words.

Text: {text}

Summary:"""
        return self.generate(prompt, max_tokens=max_length * 2, temperature=0.2)

    # ─── Fine-Tuning-Aware Methods ────────────────────────────────────────

    def route_query(self, query: str, session_context: str = "") -> Dict[str, Any]:
        """
        Stage 2 (Agentic Routing): Structured JSON intent classification.
        Returns: {intent, complexity, agents, reasoning, needs_retrieval}
        """
        prompt = PromptBuilder.route_query(query, session_context)
        result = self.extract_json(prompt, max_tokens=200)

        # Validate and provide defaults
        defaults = {
            "intent": "exploratory",
            "complexity": 0.5,
            "agents": ["planning"],
            "needs_retrieval": True,
            "reasoning": "Default routing",
        }
        for k, v in defaults.items():
            if k not in result:
                result[k] = v

        return result

    def self_rag_critique(self, query: str, answer: str,
                          evidence: List[str]) -> Dict[str, Any]:
        """
        Stage 4 (Self-RAG): Generate ISREL/ISSUP/ISUSE critique tokens.
        Returns structured critique with scores.
        """
        evidence_text = "\n".join(f"[{i+1}] {e[:200]}" for i, e in enumerate(evidence[:5]))
        prompt = PromptBuilder.self_rag_critique(query, answer[:300], evidence_text)
        result = self.extract_json(prompt, max_tokens=200)

        # Parse scores with fallbacks
        isrel = min(max(result.get("ISREL", result.get("isrel", result.get("relevance", 5))), 1), 10)
        issup = min(max(result.get("ISSUP", result.get("issup", result.get("support", 5))), 1), 10)
        isuse = min(max(result.get("ISUSE", result.get("isuse", result.get("usefulness", 5))), 1), 10)

        return {
            "ISREL": isrel,
            "ISSUP": issup,
            "ISUSE": isuse,
            "avg_score": (isrel + issup + isuse) / 3.0,
            "verdict": "ACCEPT" if (isrel + issup + isuse) / 3.0 >= 6.0 else "REVISE",
            "justification": result.get("justification", result.get("reasoning", "")),
        }

    def causal_reason(self, query: str, memories: List[str]) -> str:
        """
        Stage 3 (Causal Reasoning): Trace cause-effect chains from memories.
        """
        mem_text = "\n".join(f"[{i+1}] {m[:200]}" for i, m in enumerate(memories[:8]))
        prompt = PromptBuilder.causal_reasoning(query, mem_text)
        result = self.generate(prompt, max_tokens=500, temperature=0.3)
        result = self._strip_hallucination_patterns(result)
        result = self._validate_or_extract(query, result, memories)

        if not result or len(result.strip()) < 15:
            result = "I don't have enough stored memories to trace a causal chain for this question."

        return result

    def detect_belief_change(self, old_text: str, new_text: str, topic: str = "") -> Dict[str, Any]:
        """
        Stage 5 (Belief Evolution): Detect stance changes between two memories.
        Returns: {change_type, old_stance, new_stance, confidence, explanation}
        """
        prompt = PromptBuilder.belief_change(old_text[:300], new_text[:300], topic or "general")
        result = self.extract_json(prompt, max_tokens=200)
        defaults = {
            "change_type": "NONE",
            "old_stance": "",
            "new_stance": "",
            "confidence": 0.5,
            "explanation": "",
        }
        for k, v in defaults.items():
            if k not in result:
                result[k] = v
        return result

    def generate_faithful(self, query: str, evidence: List[str],
                          session_context: str = "") -> str:
        """
        Stage 1 (Faithfulness): Generate a grounded answer with inline citations.
        References evidence as [1], [2], etc.
        If the model hallucinates, falls back to evidence-based extraction.
        """
        evidence_text = "\n".join(f"[{i+1}] {e[:250]}" for i, e in enumerate(evidence[:5]))
        prompt = PromptBuilder.faithful_generation(query, evidence_text, session_context)
        result = self.generate(prompt, max_tokens=500, temperature=0.1)
        result = self._strip_hallucination_patterns(result)

        # Check if the result is actually relevant to the query
        # The model has a tendency to generate completely off-topic content
        result = self._validate_or_extract(query, result, evidence)

        # Final safety: never return empty or near-empty
        if not result or len(result.strip()) < 15:
            result = "I don't have that information yet — feel free to tell me and I'll remember it!"

        return result

    def raft_generate(self, query: str, oracle_docs: List[str],
                      distractor_docs: List[str]) -> str:
        """
        Stage 12 (RAFT): Generate answers while ignoring distractor documents.
        Trained to identify and use only relevant docs from mixed context.
        """
        # Interleave oracle and distractor documents
        all_docs = []
        for i, doc in enumerate(oracle_docs[:3]):
            all_docs.append(f"[Doc {len(all_docs)+1}] {doc[:200]}")
        for i, doc in enumerate(distractor_docs[:3]):
            all_docs.append(f"[Doc {len(all_docs)+1}] {doc[:200]}")

        # Shuffle to test distractor resistance (like RAFT training)
        import random
        random.shuffle(all_docs)
        docs_text = "\n".join(all_docs)

        prompt = PromptBuilder.raft_generation(query, docs_text)
        result = self.generate(prompt, max_tokens=400, temperature=0.1)
        result = self._strip_hallucination_patterns(result)

        # Validate relevance or fall back to evidence extraction
        result = self._validate_or_extract(query, result, oracle_docs)

        # Final safety: never return empty or near-empty
        if not result or len(result.strip()) < 15:
            result = "I don't have that information yet — feel free to tell me and I'll remember it!"

        return result

    def call_function(self, query: str, available_tools: List[Dict]) -> Dict[str, Any]:
        """
        Stage 13 (Function Calling): Parse user intent into tool calls.
        Returns: {tool_name, arguments, reasoning}
        """
        tools_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')} | params: {json.dumps(t.get('parameters', {}))}"
            for t in available_tools
        )
        prompt = PromptBuilder.function_calling(query, tools_desc)
        result = self.extract_json(prompt, max_tokens=200)
        if "tool_name" not in result:
            result["tool_name"] = "none"
        return result

    # ─── Fallback & Stats ─────────────────────────────────────────────────

    def _validate_or_extract(self, query: str, result: str, evidence: List[str]) -> str:
        """
        Aggressively check if the LLM result is actually relevant and faithful.
        If it's hallucinated/off-topic/generic, extract answer from evidence instead.
        """
        import re

        query_lower = query.lower().strip()
        result_lower = result.lower()

        # Strip greeting prefix from query for analysis
        for prefix in ["hey ", "hi ", "hello ", "hey, ", "hi, "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):].strip()
                break

        # ── 0. Check if query asks about data NOT in evidence ──
        no_info = self._detect_no_info(query_lower, evidence)
        if no_info:
            return no_info

        # ── 1. Short/empty results → try extraction ──
        if len(result.strip()) < 30 or "couldn't find" in result_lower:
            extracted = self._extract_answer_from_evidence(query, evidence)
            return extracted if extracted else result

        # ── 2. Detect hallucination patterns (unified set) ──
        halluc_count = sum(1 for p in _HALLUC_PATTERNS if p in result_lower)

        if halluc_count >= 1:
            # Even ONE hallucination indicator = try extraction
            extracted = self._extract_answer_from_evidence(query, evidence)
            if extracted:
                return extracted
            # If extraction fails AND multiple indicators, return "no info"
            if halluc_count >= 2:
                return "I don't have that information yet — feel free to tell me and I'll remember it!"

        # ── 3. Check relevance: query content words vs result ──
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        stopwords = {"what", "who", "where", "when", "how", "why", "which",
                      "the", "and", "for", "are", "was", "were", "been",
                      "have", "has", "had", "does", "did", "will", "would",
                      "can", "could", "should", "shall", "may", "might",
                      "that", "this", "with", "from", "about", "your",
                      "you", "tell", "list", "describe", "give", "show",
                      "all", "know", "just", "them", "please", "also",
                      "there", "their", "they", "some", "any", "more",
                      "much", "many", "very", "really", "quite"}
        query_content_words = query_words - stopwords

        if query_content_words:
            result_words = set(re.findall(r'\b[a-z]{3,}\b', result_lower))
            overlap = query_content_words & result_words
            relevance = len(overlap) / max(len(query_content_words), 1)

            # If less than 30% query words appear in response → likely off-topic
            if relevance < 0.3:
                extracted = self._extract_answer_from_evidence(query, evidence)
                if extracted:
                    return extracted

        # ── 4. Detect raw evidence dump ──
        evidence_dump_markers = [
            "[Document ", "[Memory 20", "• [Memory", "• [Document",
            "[Source:", "[Memory 1]", "[Memory 2]",
            "Document 1:", "Document 2:", "Document 3:",
            "Key Point:", "key points:",
        ]
        dump_count = sum(1 for m in evidence_dump_markers if m in result)
        if dump_count >= 2:
            extracted = self._extract_answer_from_evidence(query, evidence)
            if extracted:
                return extracted

        # ── 5. For factual queries, always try extraction FIRST ──
        simple_factual_patterns = [
            "my name", "who am i", "full name", "my email", "e-mail",
            "my phone", "my number", "my contact",
            "my university", "my college", "where do i study",
            "my degree", "what am i studying",
            "my gpa", "my grade", "my marks", "my percentage",
            "my address", "where do i live", "where am i from",
            "my age", "how old",
            "project", "built", "worked on", "portfolio", "developed",
            "skill", "language", "programming", "tech stack",
            "achievement", "award", "hackathon",
            "linkedin", "github",
            "about me", "summary", "overview",
            "class 10", "class 12", "10th", "12th",
            "chatbot", "hope", "cortex",
        ]
        for pat in simple_factual_patterns:
            if pat in query_lower:
                extracted = self._extract_answer_from_evidence(query, evidence)
                if extracted:
                    return extracted
                break

        return result

    def _detect_no_info(self, query_lower: str, evidence: List[str]) -> str:
        """
        Detect queries about information that does NOT exist in evidence.
        Returns a polite 'no info' message, or empty string if info might exist.
        Uses word-boundary regex matching to avoid substring false positives
        (e.g., "earn" matching "learning", "pay" matching "display").
        """
        import re
        all_evidence_lower = "\n".join(e.lower() for e in evidence) if evidence else ""

        def _evidence_has_word(keywords: List[str]) -> bool:
            """Check if any keyword appears as a whole word in evidence."""
            for k in keywords:
                if re.search(r'\b' + re.escape(k) + r'\b', all_evidence_lower):
                    return True
            return False

        # ── False premise detection ──
        # These are topics NOT in the user's data — the model must NOT fabricate answers
        false_premise_checks = [
            # Employment at specific companies
            (["work at google", "google job", "google employee", "employed at google",
              "work at microsoft", "microsoft job", "work at amazon", "amazon job",
              "work at meta", "meta job", "work at apple", "apple job",
              "work at tesla", "tesla job"],
             ["google employee", "microsoft employee", "amazon employee",
              "work at google", "work at microsoft", "work at amazon"],
             "I don't have any information about working at that company. If that's part of your experience, tell me about it!"),

            # PhD / Masters at specific universities
            (["phd at", "phd thesis", "phd from", "doctoral", "dissertation",
              "masters at", "masters from", "graduate school",
              "stanford", "mit ", "harvard", "oxford", "cambridge"],
             ["phd", "doctoral", "dissertation", "masters degree"],
             "I don't have any PhD or Masters information. If that's part of your journey, let me know!"),

            # Salary / compensation — use STRICT whole-word matching
            (["salary", "compensation", "how much do i earn", "how much do i make",
              "my income", "my pay", "annual salary", "monthly salary",
              "how much does", "what does he earn", "earning"],
             ["salary", "compensation", "annual income", "monthly pay",
              "ctc", "lpa", "stipend", "remuneration"],
             "I don't have any salary or compensation details yet. Feel free to share that info!"),

            # Marriage / family details — expanded triggers for 3rd person
            (["wife", "husband", "spouse", "children", "kids",
              "son", "daughter", "married", "wedding", "family members",
              "have a wife", "have a husband", "have children",
              "is he married", "is she married", "marital status"],
             ["wife", "husband", "spouse", "married", "wedding",
              "children names", "son named", "daughter named"],
             "I don't have any family or marriage details yet. You can share that with me anytime!"),

            # Published papers (if not actually published)
            (["published research", "research paper", "published paper",
              "my publications", "my paper", "my journal",
              "research papers", "published papers"],
             ["published paper", "publication in", "journal paper",
              "ieee", "arxiv", "conference paper"],
             "I don't have any research publication records yet. If you've published papers, tell me about them!"),
        ]

        for query_triggers, evidence_keywords, no_info_msg in false_premise_checks:
            # Check if query matches any trigger
            query_matches = any(t in query_lower for t in query_triggers)
            if query_matches:
                # Check if evidence actually contains relevant info (whole-word match)
                evidence_has_info = _evidence_has_word(evidence_keywords)
                if not evidence_has_info:
                    return no_info_msg

        # ── GPA check (special — evidence might have percentage but not GPA) ──
        if any(w in query_lower for w in ["my gpa", "gpa score", "grade point"]):
            if "gpa" not in all_evidence_lower and "grade point" not in all_evidence_lower:
                pct_match = re.search(r'(\d{1,3})%', "\n".join(evidence))
                if pct_match:
                    return f"I don't have a GPA score in your memories, but I found percentage-based marks. Would you like to know about those?"
                return "I don't have GPA information stored in your memories."

        return ""  # No false premise detected

    @staticmethod
    def _strip_hallucination_patterns(text: str) -> str:
        """Post-process model output to remove known hallucination patterns
        and raw model tokens that shouldn't appear in user-facing output.
        
        AGGRESSIVE: The fine-tuned model has severe hallucination from 15 training
        stages. This method catches and removes all known garbage patterns.
        """
        import re

        # Strip raw model tokens
        for token in ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                       "<think>", "</think>", "<|im_sep|>",
                       "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"]:
            text = text.replace(token, "")
        # Catch partial/malformed ChatML tokens like <|im_start|user|entity|...>
        text = re.sub(r'<\|im_start\|[^>]*>', '', text)
        text = re.sub(r'<\|im_end\|[^>]*>', '', text)

        # ── Phase 1: Remove hallucination phrases (unified set) ──
        text_lower = text.lower()
        for phrase in _HALLUC_PATTERNS:
            if phrase in text_lower:
                pos = text_lower.find(phrase)
                if pos < 150:  # Hallucination starts early → mostly garbage
                    text = text[:pos].strip()
                    text_lower = text.lower()
                    if len(text) < 20:
                        text = ""
                        break
                else:
                    text = text[:pos].rstrip(" \n\t,;:-")
                    text_lower = text.lower()

        # ── Phase 2: Remove fake confidence/synthesis claims ──
        # The model generates "Confidence: High — based on N memories" even when
        # the answer is completely fabricated
        fake_confidence_patterns = [
            r'\*?\*?Confidence:?\*?\*?\s*:?\s*(High|Medium|Low)\s*[—–-]\s*based on.*$',
            r'Synthesizing\s+\d+\s+memories?\s+about\b.*$',
            r'Based on\s+\d+\s+memories?\s+about\b.*$',
            r'\*?\*?Confidence:?\*?\*?\s*:?\s*(High|Medium|Low).*$',
            # Self-RAG format leaks: "Answer:", "Evidence:", etc.
            r'^\*?\*?Answer:?\*?\*?\s*',
            r'\*?\*?Evidence:?\*?\*?\s*:?.*$',
            r'\*?\*?Relevance:?\*?\*?\s*:?.*$',
            r'\*?\*?Sources?:?\*?\*?\s*:?\s*\[?\d.*$',
        ]
        for pattern in fake_confidence_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE).strip()

        # ── Phase 2b: Remove robotic prefixes ──
        robotic_prefixes = [
            "Based on your stored memories:",
            "Based on your memories:",
            "Based on the evidence provided:",
            "Based on the provided evidence:",
            "According to your stored memories:",
            "According to the evidence:",
            "From your stored memories:",
        ]
        for prefix in robotic_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # ── Phase 2c: Remove inline citations [1], [2], etc. ──
        text = re.sub(r'\s*\[\d+\]\.?', '', text).strip()

        # ── Phase 3: Remove placeholder tokens ──
        # The model generates [Name], [Email], etc. instead of actual data
        placeholder_patterns = [
            r'\[Name\]', r'\[Email\]', r'\[Phone\]', r'\[University\]',
            r'\[Location\]', r'\[Project\]', r'\[Skills\]',
        ]
        for pat in placeholder_patterns:
            text = re.sub(pat, '', text, flags=re.IGNORECASE).strip()

        # ── Phase 4: Remove garbled source references ──
        # Model produces "• s-repository]" type garbage
        text = re.sub(r'[•\-]\s*s-repository\]', '', text).strip()
        text = re.sub(r'\[?s-repository\]?', '', text).strip()

        # ── Phase 5: Remove "Document N:" format dumps ──
        # Model sometimes lists "Document 1: ... Key Point: ..."
        if re.search(r'Document\s+\d+:', text) and text.count("Document") >= 2:
            # Try to extract just the useful content
            lines = text.split('\n')
            useful = [l for l in lines if not re.match(r'^\s*Document\s+\d+:', l)]
            if useful:
                text = '\n'.join(useful).strip()

        # ── Phase 6: If text is now empty or very short, signal for extraction ──
        if len(text.strip()) < 10:
            text = ""

        return text.strip()

    @staticmethod
    def _fix_person(text: str) -> str:
        """Convert first-person text to second-person for natural responses."""
        import re as _re
        replacements = [
            (r'\bMy\b', 'Your'), (r'\bmy\b', 'your'),
            (r'\bI am\b', 'You are'), (r'\bI\'m\b', "You're"),
            (r'\bI have\b', 'You have'), (r'\bI was\b', 'You were'),
            (r'\bI do\b', 'You do'), (r'\bI also\b', 'You also'),
            (r'^I\b', 'You'), (r'\. I\b', '. You'),
        ]
        for pattern, repl in replacements:
            text = _re.sub(pattern, repl, text)
        return text

    @staticmethod
    def _extract_answer_from_evidence(query: str, evidence: List[str]) -> str:
        """
        When the LLM hallucinates, extract a factual answer directly from evidence.
        This is a comprehensive rule-based fallback that covers all common query types.
        
        The fine-tuned model frequently hallucinates, so this extraction must be
        robust and cover: name, email, phone, skills, projects, education,
        location, achievements, certifications, and general factual queries.
        """
        import re
        query_lower = query.lower().strip()

        # Remove greeting prefixes
        for prefix in ["hey ", "hi ", "hello ", "hey, ", "hi, ", "hey,", "hi,", "hello,"]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):].strip()
                break

        # Join all evidence for searching — strip [Source: ...] markers first
        # Also filter out spam entries (repeated phrases, stored queries)
        def _is_spam(text: str) -> bool:
            """Detect spam evidence: repeated phrases or stored user queries."""
            t = text.lower().strip()
            # Stored user query patterns
            if re.match(r'^tell me\b|^what is\b|^what are\b|^who is\b|^list\b|^give me\b', t):
                return True
            # Trigram repetition check
            words = t.split()
            if len(words) >= 9:
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                from collections import Counter
                counts = Counter(trigrams)
                if counts and counts.most_common(1)[0][1] > 3:
                    return True
            return False

        evidence = [ev for ev in evidence if not _is_spam(ev)]
        all_evidence = "\n".join(evidence) if evidence else ""
        all_evidence = re.sub(r'\[Source:\s*[^\]]*\]\s*', '', all_evidence)
        all_evidence = re.sub(r's-repository\]', '', all_evidence)  # cleanup partial markers
        all_evidence_lower = all_evidence.lower()

        # ─── Compound query detection: extract ALL matching facts ───
        # Check which facts are being asked about
        asks_name = any(w in query_lower for w in ["my name", "who am i", "full name", "what's my name", "whats my name"])
        asks_email = any(w in query_lower for w in ["email", "e-mail", "mail address", "gmail", "my mail"])
        asks_phone = any(w in query_lower for w in ["phone", "number", "contact number", "mobile", "call"])

        # Count how many different facts are requested
        compound_count = sum([asks_name, asks_email, asks_phone])

        if compound_count >= 2:
            # Compound query — extract multiple facts
            parts = []
            name_match = re.search(r'\*\*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\*\*', all_evidence)
            if not name_match:
                name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', all_evidence)
            email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', all_evidence)
            phone_match = re.search(r'\+?\d[\d\s-]{8,15}', all_evidence)

            if asks_name and name_match:
                parts.append(f"**Name:** {name_match.group(1) if '(' not in name_match.group(0) else name_match.group(0)}")
            if asks_email and email_match:
                parts.append(f"**Email:** {email_match.group(0)}")
            if asks_phone and phone_match:
                parts.append(f"**Phone:** {phone_match.group(0).strip()}")

            if parts:
                return "Here's what I have for you:\n\n" + "\n".join(f"• {p}" for p in parts)

        # ─── Name ───
        if asks_name:
            # Look for bold name patterns (resume format)
            name_match = re.search(r'\*\*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\*\*', all_evidence)
            if name_match:
                return f"Your name is **{name_match.group(1)}**!"
            # "My name is X Y" pattern
            name_match = re.search(r'[Mm]y name is ([A-Z][a-z]+ [A-Z][a-z]+)', all_evidence)
            if name_match:
                return f"Your name is **{name_match.group(1)}**!"
            # "Name: X Y" pattern
            name_match = re.search(r'[Nn]ame[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', all_evidence)
            if name_match:
                return f"Your name is **{name_match.group(1)}**!"
            # Plain name at start of evidence
            name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', all_evidence)
            if name_match:
                return f"Your name is **{name_match.group(1)}**!"

        # ─── Email ───
        if asks_email:
            email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', all_evidence)
            if email_match:
                return f"Your email address is **{email_match.group(0)}**."

        # ─── Phone ───
        if asks_phone:
            phone_match = re.search(r'\+?\d[\d\s-]{8,15}', all_evidence)
            if phone_match:
                return f"Your phone number is **{phone_match.group(0).strip()}**."

        # ─── University / College / Education ───
        if any(w in query_lower for w in ["university", "college", "where do i study",
                                           "where am i studying", "my school", "institution",
                                           "education", "my degree", "studying", "b.tech", "btech",
                                           "education background"]):
            # Look for university/institution name — try specific patterns first
            uni_match = re.search(
                r'((?:Indian\s+)?(?:Institute|University|College)\s+of\s+[^\n,|]{5,60}|IIIT\s+[A-Za-z]+|IIT\s+[A-Za-z]+|NIT\s+[A-Za-z]+)',
                all_evidence, re.IGNORECASE
            )
            if not uni_match:
                uni_match = re.search(r'(IIIT|IIT|NIT|BITS)\s+[A-Z][a-z]+', all_evidence)
            if not uni_match:
                uni_match = re.search(r'(?:University|College|Institute)[:\s]*([^\n,|]{5,60})', all_evidence, re.IGNORECASE)

            # Also get degree info
            degree_match = re.search(
                r'(B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|Ph\.?D|Bachelor|Master)[^\n]{0,100}',
                all_evidence, re.IGNORECASE
            )

            if uni_match or degree_match:
                parts = []
                if degree_match:
                    deg = degree_match.group(0).strip().rstrip(',|*')
                    parts.append(f"pursuing **{deg}**")
                if uni_match:
                    uni = uni_match.group(0).strip().rstrip(',|*')
                    parts.append(f"at **{uni}**")
                if parts:
                    return "You're " + " ".join(parts) + "."

            # Try broader education section
            edu_match = re.search(
                r'(?:EDUCATION|Education)[:\s]*\n(.{20,300})',
                all_evidence, re.IGNORECASE
            )
            if edu_match:
                return edu_match.group(0).strip()[:300]

        # ─── Skills / Programming Languages / Tech Stack ───
        if any(w in query_lower for w in ["skill", "language", "programming", "tech stack",
                                           "technologies", "frameworks", "tools i use",
                                           "what do i know", "coding"]):
            # Strategy: search each evidence item separately and find the best match
            # Prioritize evidence that actually contains programming language names
            prog_langs = ["python", "java", "javascript", "typescript", "c++", "c#",
                          "go", "rust", "sql", "ruby", "swift", "kotlin", "scala",
                          " c,", " c ", " r,", " r "]

            # First pass: find evidence with actual language/skill names
            best_skills = ""
            best_lang_count = 0
            for ev in evidence:
                ev_lower = ev.lower()
                lang_count = sum(1 for lang in prog_langs if lang in ev_lower)
                if lang_count > best_lang_count:
                    best_lang_count = lang_count
                    # Extract the skills portion
                    skills_match = re.search(
                        r'(?:\*?\*?Skills\*?\*?|\*?\*?Programming\*?\*?|\*?\*?Technical\*?\*?)[:\s|*]*(.{20,500})',
                        ev, re.IGNORECASE
                    )
                    if skills_match:
                        best_skills = skills_match.group(0).strip()[:400]
                    elif lang_count >= 2:
                        best_skills = ev.strip()[:400]

            if best_skills:
                return f"Here are your technical skills:\n\n{LocalLLM._fix_person(best_skills)}"

            # Fallback: look for "skilled in" pattern
            skills_match = re.search(r'skilled in \*?\*?([^.]{20,300})', all_evidence, re.IGNORECASE)
            if skills_match:
                return f"Your technical skills include:\n\n{LocalLLM._fix_person(skills_match.group(0).strip()[:400])}"

        # ─── Projects ───
        if any(w in query_lower for w in ["project", "built", "worked on", "portfolio",
                                           "developed", "created", "my work", "my apps"]):
            # Look for specific project names from various evidence formats
            project_patterns = [
                r'📌\s*Project\s*Name:\s*([^\n]{3,80})',                   # "📌 Project Name: Sysmind-CLI"
                r'Section:\s*\*?\*?([^|*\]]{5,80}?)(?:\s*\||\*\*|\])',    # "[Section: **Project Title | ...]"
                r'(?:♥️|🏥|🏫|🔬|🤖|📊)\s*([^*\n]{5,80}?)(?:\s*[-–—]|\*\*)',  # Emoji + title
                r'\*\*📌\s*([^*]{3,60})\*\*',                              # "**📌 ChatGPT Clone**"
            ]
            projects = []
            for pattern in project_patterns:
                found = re.findall(pattern, all_evidence, re.IGNORECASE | re.MULTILINE)
                projects.extend(found)

            # From resume/evidence: bold project-like entries
            bold_names = re.findall(r'\*\*([A-Z][^*\n]{4,60})\*\*', all_evidence)
            for name in bold_names:
                name_clean = name.strip()
                name_lower = name_clean.lower()
                # Skip non-project bold text
                skip_words = ["name", "email", "phone", "university", "education",
                    "skills", "source", "repository", "experience", "summary",
                    "objective", "address", "contact", "reference", "language",
                    "framework", "tools", "tech stack", "deployment", "features",
                    "description", "domain", "key features", "undertaken", "section",
                    "engineering", "btech", "computer science", "production url",
                    "local development", "browser support"]
                if any(s in name_lower for s in skip_words):
                    continue
                # Keep if it looks like a project name
                project_keywords = ["autofill", "chatbot", "bot", "cli", "dashboard",
                    "platform", "system", "app", "lab", "tool", "engine", "api",
                    "clone", "detection", "prediction", "analysis", "optimizer",
                    "canvas", "generator", "foundation", "website", "panel",
                    "resume", "portfolio", "finance", "healthcare", "ai-powered",
                    "machine", "deep learning", "course", "note", "assistant",
                    "segmentation", "captioning", "eda", "exploratory", "sparc"]
                if any(k in name_lower for k in project_keywords):
                    projects.append(name_clean)

            # "Designed and developed an X" pattern from resume descriptions
            dev_found = re.findall(
                r'(?:Designed and developed|Developed|Built|Created)\s+(?:an?\s+)?(?:the\s+)?\*?\*?([^*\n,]{5,80}?)(?:\*\*|\s+(?:platform|system|using|with|for|that|to|enabling)\b)',
                all_evidence, re.IGNORECASE
            )
            projects.extend(dev_found)

            # Also look for standalone bold titles (2+ words, starts uppercase)
            bold_titles = re.findall(r'\*\*([A-Z][a-z]+(?:[-\s][A-Za-z]+){1,5})\*\*', all_evidence)
            for title in bold_titles:
                title_lower = title.lower()
                skip_generic = ["name", "email", "phone", "university", "education",
                    "skills", "source", "repository", "experience", "summary",
                    "objective", "address", "contact", "reference", "language",
                    "framework", "tools", "tech stack", "deployment", "features",
                    "key features", "undertaken", "section", "engineering",
                    "btech", "computer science", "projects undertaken"]
                if any(s in title_lower for s in skip_generic):
                    continue
                if len(title) > 8:
                    projects.append(title)

            if projects:
                # Deduplicate and filter garbage
                seen = set()
                unique = []
                garbage_words = ["repository", "source", "s-repository", "]", "["]
                for p in projects:
                    p_clean = p.strip().rstrip('*#').strip()
                    p_lower = p_clean.lower()
                    # Skip garbage, duplicates, too-short entries
                    if p_lower in seen or len(p_clean) < 4:
                        continue
                    if any(g in p_lower for g in garbage_words):
                        continue
                    seen.add(p_lower)
                    unique.append(p_clean)
                if unique:
                    project_list = "\n".join(f"• **{p}**" for p in unique[:10])
                    return f"Here are the projects you've built:\n\n{project_list}"

            # Project-specific fallback: if no project names found but evidence has project data
            # Summarize the best project-related evidence
            for ev in evidence:
                ev_lower = ev.lower()
                if any(k in ev_lower for k in ["project", "built", "developed", "designed and developed", "platform", "application"]):
                    clean_ev = re.sub(r'\[Source:\s*[^\]]*\]\s*', '', ev).strip()
                    clean_ev = re.sub(r'\[?s-repository\]?', '', clean_ev).strip()
                    # Skip tech-stack-only or tools-only evidence  
                    if clean_ev.startswith("🛠") or clean_ev.startswith("🔧") or clean_ev.startswith("🚀"):
                        continue
                    if len(clean_ev) > 50:
                        return f"Here's what I know about your projects:\n\n{clean_ev[:500]}"

        # ─── Location / Hometown ───
        if any(w in query_lower for w in ["where do i live", "my location", "my city",
                                           "my hometown", "where am i from", "my address",
                                           "where i stay", "home town"]):
            loc_patterns = [
                r'(?:Patna|Bihar|Bangalore|Karnataka|Mumbai|Delhi|Hyderabad|Chennai|Kolkata)[,\s]*(?:Patna|Bihar|Bangalore|Karnataka|India)?',
                r'(?:from|located in|lives in|staying in|hometown)\s*:?\s*([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)*)',
            ]
            for pattern in loc_patterns:
                match = re.search(pattern, all_evidence, re.IGNORECASE)
                if match:
                    return f"You're from **{match.group(0).strip()}**."

        # ─── Achievements / Awards ───
        if any(w in query_lower for w in ["achievement", "award", "honor", "recognition",
                                           "scholarship", "competition", "hackathon", "won"]):
            achieve_patterns = [
                r'(?:Times|Scholar|Award|Winner|Hackathon|Top\s+\d|Competition)[^.\n]{10,200}',
                r'(?:🏆|🥇|🏅)[^.\n]{10,200}',
            ]
            achievements = []
            for pattern in achieve_patterns:
                found = re.findall(pattern, all_evidence, re.IGNORECASE)
                achievements.extend(found)
            if achievements:
                ach_list = "\n".join(f"• {a.strip()}" for a in achievements[:5])
                return f"Here are your achievements:\n\n{ach_list}"

        # ─── Specific project query (e.g., "tell me about Hope chatbot") ───
        # Extract specific entities from query
        query_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        for entity in query_entities:
            entity_lower = entity.lower()
            if entity_lower in ["the", "what", "tell", "about", "how", "who",
                                "does", "suraj", "kumar", "list", "give"]:
                continue
            if entity_lower in all_evidence_lower:
                # Find the paragraph containing this entity
                for ev in evidence:
                    if entity_lower in ev.lower():
                        # Clean evidence: remove [Source: ...] markers
                        clean_ev = re.sub(r'\[Source:\s*[^\]]+\]\s*', '', ev).strip()
                        clean_ev = re.sub(r'\[?s-repository\]?', '', clean_ev).strip()
                        if len(clean_ev) > 30:
                            return f"Here's what I know about **{entity}**:\n\n{LocalLLM._fix_person(clean_ev[:400])}"

        # ─── LinkedIn / GitHub / Social Links ───
        if any(w in query_lower for w in ["linkedin", "github", "social", "profile", "link"]):
            link_match = re.search(r'(https?://[^\s\|]+)', all_evidence)
            if link_match:
                links = re.findall(r'(https?://[^\s\|]+)', all_evidence)
                link_list = "\n".join(f"• {l.strip()}" for l in links[:5])
                return f"Here are your profile links:\n\n{link_list}"

        # ─── Summary / Overview / About Me ───
        if any(w in query_lower for w in ["about me", "summary", "overview", "introduce",
                                           "who am i", "tell me about myself", "about myself"]):
            # Find the summary/bio section
            summary_match = re.search(
                r'(?:Summary|About|Bio)[:\s*]*\n?(.{30,500})',
                all_evidence, re.IGNORECASE
            )
            if summary_match:
                return summary_match.group(0).strip()[:400]

        # ─── Class 10th / 12th marks ───
        if any(w in query_lower for w in ["class 10", "class 12", "10th", "12th",
                                           "board exam", "school marks", "percentage"]):
            marks_match = re.search(r'Class\s+1[02](?:th)?\s*[:\s]*(\d{1,3})%', all_evidence, re.IGNORECASE)
            if marks_match:
                # Get both 10th and 12th
                all_marks = re.findall(r'Class\s+(1[02])(?:th)?\s*[:\s]*(\d{1,3})%', all_evidence, re.IGNORECASE)
                if all_marks:
                    marks_list = ", ".join(f"Class {cls}th: {pct}%" for cls, pct in all_marks)
                    return f"Your board exam scores: {marks_list}."

        # ─── Generic: Find the most relevant evidence piece by query word matching ───
        if evidence:
            best_ev = ""
            best_score = 0
            query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
            filler = {"what", "who", "where", "when", "how", "why", "which",
                      "the", "and", "for", "are", "was", "tell", "about",
                      "your", "you", "please", "give", "show", "list",
                      "does", "did", "has", "have", "been", "will"}
            content_words = query_words - filler

            for ev in evidence:
                ev_lower = ev.lower()
                score = sum(1 for w in content_words if w in ev_lower)
                if score > best_score and len(ev) > 50:
                    best_score = score
                    best_ev = ev

            if best_ev and best_score >= 1:
                # Clean evidence text
                clean = re.sub(r'\[Source:\s*[^\]]+\]\s*', '', best_ev).strip()
                clean = re.sub(r'\[?s-repository\]?', '', clean).strip()
                if len(clean) > 30:
                    return f"{LocalLLM._fix_person(clean[:400])}"

        return ""  # No extraction possible

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when model is not loaded."""
        return "[Model not loaded - cannot generate response]"

    def get_stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_latency_ms": round(self._total_time_ms / max(self._call_count, 1), 1),
            "model_loaded": self.model is not None,
        }

    def reset_stats(self):
        self._call_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0


class LLMProvider:
    """
    Proxy that transparently delegates LLM calls to either the local
    fine-tuned model (LocalLLM) or Google Gemini (GeminiLLM).

    All downstream code (orchestrator, agents, ingestion, query engine)
    receives this provider and calls .generate(), .route_query(), etc.
    without knowing which backend is active.
    """

    def __init__(self):
        self.local_llm: Optional[LocalLLM] = None
        self.gemini_llm = None          # GeminiLLM instance (or None)
        self.provider: str = "local"    # "local" | "gemini"

    @property
    def active_llm(self):
        """Return whichever LLM is currently selected."""
        if self.provider == "gemini" and self.gemini_llm is not None:
            return self.gemini_llm
        return self.local_llm

    @property
    def has_gemini(self) -> bool:
        return self.gemini_llm is not None

    def set_provider(self, name: str):
        if name in ("local", "gemini"):
            self.provider = name

    # ── Delegate everything else to the active LLM ───────────────────────
    def __getattr__(self, name):
        # Called only for attributes not found on this instance.
        # 'local_llm', 'gemini_llm', 'provider', 'active_llm',
        # 'has_gemini', 'set_provider' are all found normally.
        active = self.active_llm
        if active is None:
            raise AttributeError(
                f"LLMProvider has no active LLM (provider={self.provider!r})"
            )
        return getattr(active, name)
