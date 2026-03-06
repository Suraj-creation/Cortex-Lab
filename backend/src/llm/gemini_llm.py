"""
Gemini LLM Interface for Cortex Lab — Google Gemini API Integration
Mirrors the LocalLLM interface so that the entire RAG pipeline can
transparently switch between the fine-tuned local model and Gemini.

Uses the new google.genai SDK with gemini-2.5-flash by default.
"""

import asyncio
import time
import re
import json
from typing import Optional, Dict, List, Any


class GeminiLLM:
    """
    Interface to Google Gemini API.
    Implements the same method signatures as LocalLLM so the
    rest of the pipeline (orchestrator, agents, ingestion) can use
    either one without code changes.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model_name = model_name

        # Tracking (same attributes as LocalLLM for stats compatibility)
        self._call_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0

        # Set model to a truthy sentinel so `self.llm.model is not None`
        # checks pass (code uses this to decide if LLM calls are available).
        # tokenizer stays None — Gemini doesn't use a local tokenizer.
        self.model = "gemini-api"
        self.tokenizer = None

    # ── Core generation ──────────────────────────────────────────────────

    def _make_config(self, max_tokens: int = 512, temperature: float = 0.3,
                     top_p: float = 0.95):
        # Gemini 2.5 Flash is a thinking model — internal reasoning tokens count
        # against max_output_tokens. Multiply by 8 to leave room for thinking.
        effective_max = min(max_tokens * 8, 65536)
        return self._types.GenerateContentConfig(
            max_output_tokens=effective_max,
            temperature=max(temperature, 0.0),
            top_p=top_p,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        structured: bool = False,
    ) -> str:
        """General text generation via Gemini API."""
        t0 = time.time()
        self._call_count += 1

        config = self._make_config(max_tokens, temperature, top_p)

        try:
            response = self._client.models.generate_content(
                model=self._model_name, contents=prompt, config=config
            )
            text = response.text.strip() if response.text else ""
        except Exception as e:
            print(f"  ⚠ Gemini generate error: {e}")
            text = ""

        elapsed_ms = (time.time() - t0) * 1000
        self._total_time_ms += elapsed_ms

        # Estimate token usage
        self._total_tokens += len(text.split())
        return text

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
    ) -> str:
        """Async generation via Gemini API."""
        t0 = time.time()
        self._call_count += 1
        config = self._make_config(max_tokens, temperature, top_p)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_name, contents=prompt, config=config
            )
            text = response.text.strip() if response.text else ""
        except Exception as e:
            print(f"  ⚠ Gemini async generate error: {e}")
            text = ""

        elapsed_ms = (time.time() - t0) * 1000
        self._total_time_ms += elapsed_ms
        self._total_tokens += len(text.split())
        return text

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.95,
    ):
        """Yields text chunks for streaming responses."""
        self._call_count += 1
        config = self._make_config(max_tokens, temperature, top_p)
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model_name, contents=prompt, config=config
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"  ⚠ Gemini stream error: {e}")
            yield ""

    async def generate_stream_async(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.95,
    ):
        """Async generator that yields text chunks for streaming."""
        self._call_count += 1
        config = self._make_config(max_tokens, temperature, top_p)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model_name, contents=prompt, config=config
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"  ⚠ Gemini async stream error: {e}")

    # ── Classification & JSON ────────────────────────────────────────────

    def classify(self, prompt: str, options: list, default: str = "") -> str:
        """Quick classification with constrained output."""
        cats = ", ".join(options)
        full_prompt = (
            f"{prompt}\n\n"
            f"Choose EXACTLY ONE from: {cats}\n"
            f"Respond with ONLY the category name, nothing else."
        )

        result = self.generate(full_prompt, max_tokens=20, temperature=0.1)
        # Find the best matching option
        result_lower = result.strip().lower()
        for opt in options:
            if opt.lower() in result_lower:
                return opt
        return default or options[0]

    def extract_json(self, prompt: str, max_tokens: int = 256) -> dict:
        """Generate and parse JSON from model output."""
        full_prompt = prompt + "\n\nRespond with valid JSON only, no markdown fences."
        result = self.generate(full_prompt, max_tokens=max_tokens, temperature=0.1)
        # Strip markdown code fences if present
        result = re.sub(r"^```(?:json)?\s*", "", result.strip())
        result = re.sub(r"\s*```$", "", result.strip())
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Try to find JSON in the output
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Concise summarization."""
        prompt = (
            f"Summarize the following text concisely in {max_length} characters or fewer. "
            f"Preserve key facts and entities.\n\nText: {text}\n\nSummary:"
        )
        return self.generate(prompt, max_tokens=200, temperature=0.2)

    # ── Fine-tuning-aware methods (Stage-equivalent prompts) ─────────────

    def route_query(self, query: str, session_context: str = "") -> dict:
        """Structured JSON routing — equivalent to Stage 2 Agentic training."""
        prompt = f"""Analyze this query and output a JSON object with the following fields:
- "intent": one of [temporal, causal, reflective, procedural, factual, comparative, exploratory]
- "complexity": float 0.0-1.0 (how complex is the query)
- "agents": list of agents needed from [timeline, causal, reflection, planning, arbitration]
- "needs_retrieval": boolean (does it need memory/document retrieval)
- "reasoning": brief explanation

Query: {query}
{('Context: ' + session_context) if session_context else ''}

Respond with valid JSON only."""

        return self.extract_json(prompt, max_tokens=256)

    def self_rag_critique(
        self, query: str, answer: str, evidence: List[str]
    ) -> dict:
        """ISREL/ISSUP/ISUSE scoring — equivalent to Stage 4 Self-RAG."""
        evidence_text = "\n".join(f"[{i+1}] {e[:200]}" for i, e in enumerate(evidence[:5]))
        prompt = f"""Evaluate the quality of this answer. Score each dimension 1-10:

ISREL (Relevance): Does the retrieved evidence address the query?
ISSUP (Support): Is the answer grounded in the evidence?
ISUSE (Usefulness): Is the answer complete and helpful?

Query: {query}
Evidence: {evidence_text}
Answer: {answer[:500]}

Output JSON:
{{"isrel": <1-10>, "issup": <1-10>, "isuse": <1-10>, "verdict": "ACCEPT or REVISE", "reasoning": "brief explanation"}}"""

        result = self.extract_json(prompt, max_tokens=200)
        # Ensure required keys with defaults
        return {
            "isrel": result.get("isrel", 7),
            "issup": result.get("issup", 7),
            "isuse": result.get("isuse", 7),
            "verdict": result.get("verdict", "ACCEPT"),
            "reasoning": result.get("reasoning", ""),
        }

    def causal_reason(self, query: str, memories: List[str]) -> str:
        """Cause-effect chain analysis — equivalent to Stage 3 Causal."""
        mem_text = "\n".join(f"[{i+1}] {m[:200]}" for i, m in enumerate(memories[:8]))
        prompt = f"""Analyze causal relationships in the user's memories to answer their question.
Trace cause-and-effect chains clearly.

Question: {query}

Relevant memories:
{mem_text}

Provide a clear, conversational explanation of the causal chain. Use "you/your" to refer to the user."""

        return self.generate(prompt, max_tokens=500, temperature=0.3)

    def detect_belief_change(self, old_text: str, new_text: str, topic: str = "") -> dict:
        """Belief evolution detection — equivalent to Stage 5 Belief."""
        prompt = f"""Compare these two memories and detect if the user's belief or stance has changed.

Topic: {topic or 'general'}
Earlier memory: {old_text[:300]}
Later memory: {new_text[:300]}

Output JSON:
{{"change_type": "CONTRADICTION|REFINEMENT|EXPANSION|REINFORCEMENT|NONE", "confidence": 0.0-1.0, "explanation": "brief description of the change"}}"""

        result = self.extract_json(prompt, max_tokens=200)
        return {
            "change_type": result.get("change_type", "NONE"),
            "confidence": result.get("confidence", 0.5),
            "explanation": result.get("explanation", ""),
        }

    def generate_faithful(self, query: str, evidence: List[str],
                          session_context: str = "") -> str:
        """Grounded generation with evidence — equivalent to Stage 1 Faithfulness."""
        # Use up to 8 evidence pieces, 1500 chars each (Gemini has 1M token context)
        evidence_text = "\n\n".join(f"[{i+1}] {e[:1500]}" for i, e in enumerate(evidence[:8]))

        system = (
            "You are Cortex Lab, a personal AI memory assistant for Suraj Kumar. "
            "Answer based on the provided evidence — be comprehensive and detailed. "
            "Use 'you/your' when referring to the user. "
            "Be conversational and natural. Never say 'Based on stored memories' or cite evidence numbers. "
            "If the evidence doesn't contain the answer, say so honestly."
        )

        prompt = f"""{system}
{f'Session context: {session_context[:200]}' if session_context else ''}

User question: {query}

Evidence:
{evidence_text}

Answer comprehensively based on the evidence:"""

        result = self.generate(prompt, max_tokens=1024, temperature=0.1)
        return result

    def raft_generate(self, query: str, oracle_docs: List[str],
                      distractor_docs: List[str]) -> str:
        """Distractor-aware generation — equivalent to Stage 12 RAFT."""
        # Mix oracle with distractors
        all_docs = oracle_docs[:5] + distractor_docs[:3]
        import random
        random.shuffle(all_docs)
        docs_text = "\n---\n".join(f"[Doc {i+1}] {d[:1500]}" for i, d in enumerate(all_docs))

        prompt = f"""Answer the question based ONLY on the relevant documents below.
Some documents may be irrelevant distractors — ignore them. Be comprehensive and detailed.

Question: {query}

Documents:
{docs_text}

Answer (use only relevant documents, ignore distractors):"""

        return self.generate(prompt, max_tokens=1024, temperature=0.1)

    def call_function(self, query: str, available_tools: list) -> dict:
        """Tool/function calling — equivalent to Stage 13."""
        tools_desc = json.dumps(available_tools, indent=2)
        prompt = f"""Based on the user's query, determine if a tool/function should be called.

Available tools:
{tools_desc}

Query: {query}

If a tool should be called, output JSON: {{"tool": "tool_name", "arguments": {{...}}}}
If no tool is needed, output: {{"tool": null, "response": "direct answer"}}"""

        return self.extract_json(prompt, max_tokens=200)

    # ── Validation helpers (lightweight for Gemini) ──────────────────────

    def _validate_or_extract(
        self, query: str, evidence_texts: list, generated: str
    ) -> str:
        """Lightweight validation for Gemini output.

        Gemini follows instructions much better than the local 7B model,
        so most of the aggressive hallucination detection from LocalLLM
        is unnecessary.  We still check for empty/very-short answers and
        false-premise queries.
        """
        if not generated or len(generated.strip()) < 5:
            # Try extraction if generation is empty
            extracted = self._extract_answer_from_evidence(query, evidence_texts)
            return extracted or "I don't have that information yet — feel free to tell me and I'll remember it!"

        # Check for false premises
        no_info = self._detect_no_info(query, evidence_texts)
        if no_info:
            return no_info

        return generated

    def _detect_no_info(self, query: str, evidence_texts: list) -> str:
        """Detect queries about info that isn't in evidence."""
        q_lower = query.lower()
        evidence_joined = " ".join(evidence_texts).lower() if evidence_texts else ""

        checks = [
            (["salary", "compensation", "pay", "income", "ctc", "package"],
             "I don't have information about your salary or compensation."),
            (["phd", "doctorate", "doctoral"],
             "I don't have information about a PhD or doctorate."),
            (["married", "wife", "husband", "spouse", "wedding"],
             "I don't have information about your marital status."),
            (["publication", "published paper", "research paper"],
             "I don't have information about your publications."),
        ]

        for keywords, response in checks:
            if any(kw in q_lower for kw in keywords):
                if not any(kw in evidence_joined for kw in keywords):
                    return response
        return ""

    def _extract_answer_from_evidence(
        self, query: str, evidence_texts: list
    ) -> str:
        """Simple extraction — find the best matching evidence chunk."""
        if not evidence_texts:
            return ""

        q_lower = query.lower()
        q_words = set(re.findall(r"\b[a-z]{3,}\b", q_lower))
        filler = {
            "what", "who", "where", "when", "how", "why", "the",
            "and", "for", "are", "tell", "about", "your", "you",
            "please", "give", "show", "list", "does", "did",
        }
        content_words = q_words - filler

        best_evidence = ""
        best_score = 0
        for ev in evidence_texts:
            score = sum(1 for w in content_words if w in ev.lower())
            if score > best_score and len(ev) > 30:
                best_score = score
                best_evidence = ev

        return best_evidence[:400] if best_evidence else ""

    def _strip_hallucination_patterns(self, text: str) -> str:
        """Minimal cleanup — Gemini rarely produces the same garbage patterns."""
        # Only strip the most universal artifacts
        text = re.sub(r"\*\*(?:Answer|Evidence|Confidence|Source):\*\*.*", "", text)
        text = re.sub(r"\[(\d+)\]", "", text)  # Remove inline citations
        return text.strip()

    # ── Stats (compatibility with LocalLLM) ──────────────────────────────

    def get_stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_latency_ms": round(
                self._total_time_ms / max(self._call_count, 1), 1
            ),
            "model_loaded": True,
            "provider": "gemini",
            "model_name": self._model_name,
        }

    def reset_stats(self):
        self._call_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0
