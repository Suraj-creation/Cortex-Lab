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
import torch

# Stop patterns that indicate the model is hallucinating new turns
_LLM_STOP_PATTERNS = [
    "\nUser:", "\nuser:", "\nHuman:", "\nhuman:",
    "\nQ:", "\nA:", "\n\nUser ", "\nQuestion:",
]


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
        prompt = f"""<|im_start|>system
You are Cortex Lab's query router. Analyze the user's query and output a JSON routing decision.

Available intents: temporal, causal, reflective, factual, procedural, comparative, exploratory
Available agents: timeline, causal, reflection, planning, arbitration
Complexity: low (0.0-0.3), medium (0.3-0.6), high (0.6-1.0)
<|im_end|>
<|im_start|>user
{f"Session context: {session_context[:200]}" if session_context else ""}
Query: {query}

Output routing decision as JSON:
<|im_end|>
<|im_start|>assistant
"""
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

        prompt = f"""<|im_start|>system
You are a retrieval quality evaluator. For the given query, answer, and evidence,
evaluate three criteria and provide scores:

ISREL (Is Relevant): Does the evidence address the query? Score 1-10.
ISSUP (Is Supported): Is the answer grounded in the evidence? Score 1-10.
ISUSE (Is Useful): Is the answer useful and complete for the user? Score 1-10.

Output JSON with scores and brief justifications.
<|im_end|>
<|im_start|>user
Query: {query}
Answer: {answer[:300]}

Evidence:
{evidence_text}
<|im_end|>
<|im_start|>assistant
"""
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

        prompt = f"""<|im_start|>system
You are Cortex Lab's causal reasoning engine. Analyze the user's memories to trace
cause-effect relationships. Structure your response as:
1. Identify the causal chain (what caused what)
2. Note any contributing factors
3. Describe the effects/outcomes
Be grounded in the evidence — never fabricate causal links.
<|im_end|>
<|im_start|>user
Query: {query}

Memories:
{mem_text}
<|im_end|>
<|im_start|>assistant
"""
        return self.generate(prompt, max_tokens=500, temperature=0.3)

    def detect_belief_change(self, old_text: str, new_text: str, topic: str = "") -> Dict[str, Any]:
        """
        Stage 5 (Belief Evolution): Detect stance changes between two memories.
        Returns: {change_type, old_stance, new_stance, confidence, explanation}
        """
        prompt = f"""<|im_start|>system
You are a belief evolution detector. Compare two memories about the same topic
and classify the change. Types: CONTRADICTION, REFINEMENT, EXPANSION, REINFORCEMENT, NONE.
Output JSON with: change_type, old_stance, new_stance, confidence (0-1), explanation.
<|im_end|>
<|im_start|>user
Topic: {topic or "general"}
Earlier memory: {old_text[:300]}
Later memory: {new_text[:300]}
<|im_end|>
<|im_start|>assistant
"""
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

        prompt = f"""<|im_start|>system
You are Cortex Lab, a personal AI memory assistant. Your task is to answer the
user's question based STRICTLY on the evidence provided below.

RULES:
1. ONLY use information explicitly stated in the evidence. NEVER fabricate.
2. Use inline citations [1], [2], etc. to reference specific evidence.
3. If the evidence contains the direct answer (name, email, list, etc.), state it clearly.
4. If the evidence is insufficient, say "I don't have this information in your memories."
5. DO NOT generate generic patterns like "belief evolution", "emotion timeline", 
   "key insight", or "clarity of scope" unless the evidence explicitly discusses these.
6. For factual questions (name, email, skills, projects), give a direct factual answer.
7. Keep your answer concise and directly relevant to the question.
{f"Session context: {session_context[:200]}" if session_context else ""}
<|im_end|>
<|im_start|>user
Question: {query}

Evidence from your memories:
{evidence_text}

Based ONLY on this evidence, answer the question directly:
<|im_end|>
<|im_start|>assistant
"""
        result = self.generate(prompt, max_tokens=500, temperature=0.1)
        result = self._strip_hallucination_patterns(result)

        # Check if the result is actually relevant to the query
        # The model has a tendency to generate completely off-topic content
        result = self._validate_or_extract(query, result, evidence)

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

        prompt = f"""<|im_start|>system
You are Cortex Lab, a personal AI memory assistant.
Answer the question using ONLY the relevant documents below.
Some documents may be distractors (irrelevant) — ignore them completely.
Cite relevant documents with [Doc N].

RULES:
1. Extract and state factual information directly from the documents.
2. DO NOT generate generic patterns, emotion timelines, or belief evolutions
   unless the documents explicitly discuss them.
3. For factual questions, give a direct, concise answer.
4. If no document answers the question, say "I don't have this information."
<|im_end|>
<|im_start|>user
Question: {query}

Documents:
{docs_text}

Answer based ONLY on relevant documents:
<|im_end|>
<|im_start|>assistant
"""
        result = self.generate(prompt, max_tokens=400, temperature=0.1)
        result = self._strip_hallucination_patterns(result)

        # Validate relevance or fall back to evidence extraction
        result = self._validate_or_extract(query, result, oracle_docs)

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

        prompt = f"""<|im_start|>system
You are a function calling assistant. Given the user's request and available tools,
decide which tool to call and with what arguments.
Output JSON: {{"tool_name": "...", "arguments": {{...}}, "reasoning": "..."}}
If no tool is needed, set tool_name to "none".

Available tools:
{tools_desc}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        result = self.extract_json(prompt, max_tokens=200)
        if "tool_name" not in result:
            result["tool_name"] = "none"
        return result

    # ─── Fallback & Stats ─────────────────────────────────────────────────

    def _validate_or_extract(self, query: str, result: str, evidence: List[str]) -> str:
        """
        Check if the LLM result is actually relevant to the query.
        If it's hallucinated/off-topic, extract answer from evidence instead.
        """
        import re

        # Short/empty results → extract
        if len(result.strip()) < 30 or "couldn't find" in result.lower():
            extracted = self._extract_answer_from_evidence(query, evidence)
            return extracted if extracted else result

        query_lower = query.lower().strip()
        result_lower = result.lower()

        # Strip greeting prefix from query for analysis
        for prefix in ["hey ", "hi ", "hello ", "hey, ", "hi, "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):].strip()
                break

        # Extract key terms from query (nouns/content words)
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        stopwords = {"what", "who", "where", "when", "how", "why", "which",
                      "the", "and", "for", "are", "was", "were", "been",
                      "have", "has", "had", "does", "did", "will", "would",
                      "can", "could", "should", "shall", "may", "might",
                      "that", "this", "with", "from", "about", "your",
                      "you", "tell", "list", "describe", "give", "show",
                      "all", "know", "just", "them", "please", "also",
                      "there", "their", "they", "some", "any", "more"}
        query_content_words = query_words - stopwords

        # Check: does the response mention ANY of the query's content words?
        if query_content_words:
            result_words = set(re.findall(r'\b[a-z]{3,}\b', result_lower))
            overlap = query_content_words & result_words
            relevance = len(overlap) / max(len(query_content_words), 1)

            # If less than 30% query words appear in response → likely off-topic
            if relevance < 0.3:
                # Check if evidence has better content
                extracted = self._extract_answer_from_evidence(query, evidence)
                if extracted:
                    return extracted

        # Detect generic hallucination indicators
        halluc_indicators = [
            "life purpose", "deep work", "stay focused in meetings",
            "rest is an input", "difficult lesson", "performance, not a reward",
            "avoid deep work", "consistent over years",
            "communication skills", "deliberate practice",
            "My View on", "Improved Answer", "What led me to",
            "## Summary of", "### Key Findings",
            "Limited relevant memories", "partial information",
            "Emotional trajectory", "belief evolution",
            "Had a difficult moment", "Key lesson from",
            "moving cities", "city-building",
        ]
        for indicator in halluc_indicators:
            if indicator.lower() in result_lower:
                extracted = self._extract_answer_from_evidence(query, evidence)
                if extracted:
                    return extracted
                break

        # Detect raw evidence dump format (model just copies evidence verbatim)
        evidence_dump_markers = [
            "[Document ", "[Memory 20", "• [Memory", "• [Document",
            "[Source:", "[Memory 1]", "[Memory 2]",
        ]
        dump_count = sum(1 for m in evidence_dump_markers if m in result)
        if dump_count >= 2:
            extracted = self._extract_answer_from_evidence(query, evidence)
            if extracted:
                return extracted

        # For simple factual queries, always try extraction first
        simple_factual_patterns = [
            "my name", "who am i", "my email", "e-mail",
            "my phone", "my number", "my contact",
        ]
        for pat in simple_factual_patterns:
            if pat in query_lower:
                extracted = self._extract_answer_from_evidence(query, evidence)
                if extracted:
                    return extracted
                break

        return result

    @staticmethod
    def _strip_hallucination_patterns(text: str) -> str:
        """Post-process model output to remove known hallucination patterns
        and raw model tokens that shouldn't appear in user-facing output."""
        # Strip raw model tokens
        for token in ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                       "<think>", "</think>", "<|im_sep|>"]:
            text = text.replace(token, "")

        # Strip the model's tendency to generate generic reflective patterns
        # when it doesn't have a real answer. These are artifacts of the
        # fine-tuning stages being over-represented.
        halluc_phrases = [
            "Your belief evolution can be traced across",
            "The key insight is that clarity of scope prevents scope creep",
            "This comes from watching someone you respect make it happen",
            "The key driver behind my most impactful work has been clarity of scope",
            "I do my best learning during transitions",
            "I'm more motivated when I'm tired than when I'm excited",
            "small consistent actions beat sporadic bursts",
            "The bottleneck has shifted from resources to knowledge integration",
            "The timeline for meaningful impact has accelerated",
            "Tracing causal chains across your thinking journey",
            "Tracing causal chains across",
            "chain of cumulative insight",
            "Each step built on the previous",
            "Your thinking journey reveals",
            "Excited — Anxious — Drained",
            "Emotion Timeline:",
            "Emotion Timeline",
            "Emotion evolution",
            "Emotional resilience",
            "had an unexpected complication",
            "strongly motivated thr",
            "sporadic bursts",
            "personal growth and modern technology",
            "key moments",
            "cumulative insight",
            "deliberate practice",
            "The core challenge has always been",
            "This evolved naturally from years",
            "The timeline for meaningful change",
            "Causal link:",
            "Step 1 [Memory",
            "→ Step 2",
            "→ Step 3",
            "Drained by the lack of follow-through",
            "Still processing this",
            "Key Findings:",  # Only hallucinated when appearing at start with no context
        ]
        for phrase in halluc_phrases:
            if phrase in text:
                # If the text is mostly hallucination, indicate insufficient data
                phrase_pos = text.find(phrase)
                if phrase_pos < 150:  # Hallucination starts early → mostly garbage
                    text = text[:phrase_pos].strip()
                    if len(text) < 20:
                        text = "Based on your stored memories, I couldn't find a specific answer to this question. The relevant evidence may be limited."

        return text.strip()

    @staticmethod
    def _extract_answer_from_evidence(query: str, evidence: List[str]) -> str:
        """
        When the LLM hallucinates, extract a factual answer directly from evidence.
        This is a rule-based fallback that looks for common patterns in the stored data.
        """
        import re
        query_lower = query.lower().strip()

        # Remove greeting prefixes
        for prefix in ["hey ", "hi ", "hello ", "hey, ", "hi, "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):].strip()
                break

        # Join all evidence for searching
        all_evidence = "\n".join(evidence)

        # Pattern matching for common factual queries
        # ─── Name ───
        if any(w in query_lower for w in ["my name", "who am i"]):
            # Look for name patterns in evidence
            name_match = re.search(r'\*\*([A-Z][a-z]+ [A-Z][a-z]+)\*\*', all_evidence)
            if name_match:
                return f"Based on your stored memories, your name is **{name_match.group(1)}**."
            name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)\s*\n', all_evidence)
            if name_match:
                return f"Based on your stored memories, your name is **{name_match.group(1)}**."

        # ─── Email ───
        if any(w in query_lower for w in ["email", "e-mail", "mail"]):
            email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', all_evidence)
            if email_match:
                return f"Your email address is **{email_match.group(0)}**."

        # ─── Phone ───
        if any(w in query_lower for w in ["phone", "number", "contact"]):
            phone_match = re.search(r'\+?\d[\d\s-]{8,}', all_evidence)
            if phone_match:
                return f"Your phone number is **{phone_match.group(0).strip()}**."

        # ─── Skills/Languages ───
        if any(w in query_lower for w in ["skill", "language", "programming", "tech"]):
            # Look for skills section
            skills_match = re.search(
                r'(?:Skills|Programming|Technical)[:\s*]*(.{20,300})',
                all_evidence, re.IGNORECASE
            )
            if skills_match:
                return f"Based on your resume and memories:\n\n{skills_match.group(0)[:300]}"

        # ─── Projects ───
        if any(w in query_lower for w in ["project", "built", "worked on", "portfolio"]):
            # Look for project headers
            projects = re.findall(
                r'(?:📌|Project|###)\s*(?:Project\s*\d*[:\s]*)?(.{10,100})',
                all_evidence, re.IGNORECASE
            )
            if projects:
                project_list = "\n".join(f"• {p.strip()}" for p in projects[:8])
                return f"Based on your stored memories, here are your projects:\n\n{project_list}"

        # ─── Generic: Return the most relevant evidence snippet ───
        # Find the longest, most informative evidence piece
        best_evidence = ""
        for e in evidence:
            if len(e) > len(best_evidence) and "[Source:" in e:
                best_evidence = e

        if best_evidence and len(best_evidence) > 50:
            return f"Based on your stored memories:\n\n{best_evidence[:400]}"

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
