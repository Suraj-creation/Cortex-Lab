"""Specialized agents for Cortex Lab Agentic RAG.

This module now supports the full 15-agent orchestration design while preserving
compatibility with the current 5-agent runtime paths.
"""

import re
import time
from collections import Counter
from typing import Dict, List, Optional

from src.agents.prompt_layers import compose_specialized_system_prompt
from src.llm import LocalLLM
from src.models import AgentResponse, MemoryQuery, RetrievalResult
from src.retrieval.hybrid_retriever import HybridRetriever


SPECIALIZED_AGENT_ORDER = [
    "timeline",
    "causal",
    "reflection",
    "planning",
    "arbitration",
    "academic",
    "journaling",
    "wellbeing",
    "cognitive",
    "decisions",
    "emotional",
    "behavioral",
    "social",
    "goals",
    "meta_learning",
]


class BaseAgent:
    """Base class for all specialized agents."""

    def __init__(
        self,
        name: str,
        llm: LocalLLM,
        retriever: HybridRetriever,
        prompt_key: Optional[str] = None,
        retrieval_top_k: int = 12,
        evidence_top_k: int = 8,
    ):
        self.name = name
        self.llm = llm
        self.retriever = retriever
        self.prompt_key = prompt_key or name
        self.retrieval_top_k = retrieval_top_k
        self.evidence_top_k = evidence_top_k

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        raise NotImplementedError

    def _compose_layered_context(
        self,
        query: MemoryQuery,
        context: str = "",
        extra_instructions: str = "",
    ) -> str:
        runtime_kwargs = self._runtime_prompt_kwargs(query)
        return compose_specialized_system_prompt(
            agent_key=self.prompt_key,
            query=query.raw_query,
            session_context=context,
            extra_instructions=extra_instructions,
            event_type=runtime_kwargs["event_type"],
            runtime_mode=runtime_kwargs["runtime_mode"],
            llm_provider=runtime_kwargs["llm_provider"],
            trace_id=runtime_kwargs["trace_id"],
            execution_mode=runtime_kwargs["execution_mode"],
            conflict_resolution=runtime_kwargs["conflict_resolution"],
            permission_chain=runtime_kwargs["permission_chain"],
            privacy_tier=runtime_kwargs["privacy_tier"],
        )

    def _runtime_prompt_kwargs(self, query: MemoryQuery) -> Dict[str, str]:
        metadata = dict(getattr(query, "metadata", {}) or {})
        return {
            "event_type": str(metadata.get("runtime_event", "query_flow")),
            "runtime_mode": str(metadata.get("runtime_mode", "cloud")),
            "llm_provider": str(metadata.get("llm_provider", self.llm.__class__.__name__)),
            "trace_id": str(metadata.get("trace_id", "")),
            "execution_mode": str(metadata.get("execution_mode", "")),
            "conflict_resolution": str(metadata.get("conflict_resolution", "")),
            "permission_chain": str(
                metadata.get(
                    "permission_chain",
                    "schema->scope->resource->privacy->user_permission->audit",
                )
            ),
            "privacy_tier": str(metadata.get("privacy_tier", "default")),
        }

    def _format_evidence(self, results: List[RetrievalResult], max_items: int = 5) -> str:
        """Format retrieval results into a text block for diagnostics."""
        parts = []
        for i, result in enumerate(results[:max_items]):
            ts = result.memory.timestamp.strftime("%Y-%m-%d %H:%M") if result.memory.timestamp else "Unknown"
            parts.append(
                f"[{i + 1}] ({ts}, {result.memory.memory_type.value}, score: {result.score:.2f})\n"
                f"{result.memory.content}"
            )
        return "\n\n".join(parts) if parts else "No relevant memories found."

    def _evidence_texts(self, results: List[RetrievalResult], max_items: int = 5) -> List[str]:
        """Extract high-signal evidence snippets for LLM calls."""
        texts: List[str] = []

        for result in results[: max_items + 8]:
            content = result.memory.content.strip()
            lower = content.lower()

            if len(content) < 50:
                continue

            if lower.endswith("?") and len(content) < 120:
                if not any(key in lower for key in ["[source:", "project", "built", "created"]):
                    continue

            words = lower.split()
            if len(words) > 10:
                trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
                trigram_counts = Counter(trigrams)
                if trigram_counts and max(trigram_counts.values()) > 3:
                    continue

            if re.match(r"^(tell me|what is|what are|who is|where is|how is|list|describe|explain)\b", lower):
                if len(content) < 200 and "[source:" not in lower:
                    continue

            texts.append(content[:1500])
            if len(texts) >= max_items:
                break

        return texts

    def _fallback_no_info_answer(self, query: MemoryQuery) -> str:
        return self.llm.generate(
            (
                "You are Cortex Lab. The user asked for information that is not available in stored memories. "
                "Respond honestly and concisely without fabrication.\n\n"
                f"User: {query.raw_query}\n\n"
                "Assistant:"
            ),
            max_tokens=200,
            temperature=0.2,
        )


class DomainSpecializedAgent(BaseAgent):
    """Generic agent for domain-specialized retrieval + grounded synthesis."""

    def __init__(
        self,
        name: str,
        llm: LocalLLM,
        retriever: HybridRetriever,
        extra_instructions: str,
        retrieval_top_k: int = 12,
        evidence_top_k: int = 8,
    ):
        super().__init__(
            name=name,
            llm=llm,
            retriever=retriever,
            prompt_key=name,
            retrieval_top_k=retrieval_top_k,
            evidence_top_k=evidence_top_k,
        )
        self.extra_instructions = extra_instructions

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()
        results = await self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        evidence = self._evidence_texts(results, max_items=self.evidence_top_k)

        if evidence:
            answer = self.llm.generate_faithful(
                query.raw_query,
                evidence,
                session_context=self._compose_layered_context(
                    query,
                    context,
                    extra_instructions=self.extra_instructions,
                ),
            )
        else:
            answer = self._fallback_no_info_answer(query)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.45 + len(results) * 0.04, 0.88),
            reasoning_trace=(
                f"{self.name} agent: retrieved {len(results)} memories, "
                f"layered prompt synthesis"
            ),
            processing_time_ms=elapsed,
        )


class TimelineAgent(BaseAgent):
    """Temporal/chronological reasoning with layered prompting."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="timeline",
            llm=llm,
            retriever=retriever,
            prompt_key="timeline",
            retrieval_top_k=15,
            evidence_top_k=8,
        )

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()
        results = await self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        results.sort(key=lambda item: item.memory.timestamp.isoformat() if item.memory.timestamp else "")

        evidence = self._evidence_texts(results, max_items=self.evidence_top_k)
        if evidence:
            answer = self.llm.generate_faithful(
                query.raw_query,
                evidence,
                session_context=self._compose_layered_context(
                    query,
                    context,
                    extra_instructions="Focus on sequence, chronology, and temporal anchors.",
                ),
            )
        else:
            answer = self._fallback_no_info_answer(query)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.5 + len(results) * 0.05, 0.95),
            reasoning_trace=(
                f"Timeline agent: retrieved {len(results)} memories, "
                "sorted chronologically, layered prompt generation"
            ),
            processing_time_ms=elapsed,
        )


class CausalAgent(BaseAgent):
    """Cause-effect reasoning using causal model path + layered fallback."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="causal",
            llm=llm,
            retriever=retriever,
            prompt_key="causal",
            retrieval_top_k=15,
            evidence_top_k=8,
        )

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()
        results = await self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        evidence = self._evidence_texts(results, max_items=self.evidence_top_k)

        if evidence:
            answer = self.llm.causal_reason(query.raw_query, evidence)
        else:
            answer = "I don't have enough stored memories to trace a causal chain for this question."

        if len(answer.strip()) < 50 and evidence:
            supplement = self.llm.generate_faithful(
                query.raw_query,
                evidence,
                session_context=self._compose_layered_context(
                    query,
                    context,
                    extra_instructions="Prioritize causal chain clarity and confidence.",
                ),
            )
            if len(supplement.strip()) > len(answer.strip()):
                answer = supplement

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.4 + len(results) * 0.06, 0.9),
            reasoning_trace=(
                f"Causal agent: analyzed {len(results)} memories with causal reasoning "
                "and layered fallback generation"
            ),
            processing_time_ms=elapsed,
        )


class ReflectionAgent(BaseAgent):
    """Belief and perspective evolution analysis with layered prompts."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="reflection",
            llm=llm,
            retriever=retriever,
            prompt_key="reflection",
            retrieval_top_k=20,
            evidence_top_k=10,
        )

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()
        results = await self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        results.sort(key=lambda item: item.memory.timestamp.isoformat() if item.memory.timestamp else "")
        evidence = self._evidence_texts(results, max_items=self.evidence_top_k)

        belief_analysis = ""
        if len(results) >= 2:
            earliest = results[0].memory.content[:250]
            latest = results[-1].memory.content[:250]
            topic = query.topics[0] if query.topics else query.raw_query[:80]
            try:
                delta = self.llm.detect_belief_change(earliest, latest, topic)
                change_type = delta.get("change_type", "unknown")
                explanation = delta.get("explanation", "")
                if explanation:
                    belief_analysis = f"\n\nBelief shift ({change_type}): {explanation}"
            except Exception:
                pass

        if evidence:
            answer = self.llm.generate_faithful(
                query.raw_query,
                evidence,
                session_context=self._compose_layered_context(
                    query,
                    context,
                    extra_instructions="Explain directional shifts and turning points over time.",
                ),
            )
            answer += belief_analysis
        else:
            answer = "I don't have enough stored memories to identify meaningful reflection patterns for this query."

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.4 + len(results) * 0.04, 0.85),
            reasoning_trace=(
                f"Reflection agent: analyzed {len(results)} memories with belief-change checks "
                "and layered synthesis"
            ),
            processing_time_ms=elapsed,
        )


class PlanningAgent(BaseAgent):
    """Complex multi-step synthesis with RAFT and layered planning context."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="planning",
            llm=llm,
            retriever=retriever,
            prompt_key="planning",
            retrieval_top_k=12,
            evidence_top_k=8,
        )

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()
        all_results: List[RetrievalResult] = []
        sub_answers: List[str] = []

        sub_queries = query.sub_queries if query.sub_queries else [query.raw_query]
        for sub_query_text in sub_queries:
            sub_query = MemoryQuery(
                raw_query=sub_query_text,
                intent=query.intent,
                complexity=0.4,
                embedding=self.retriever.embeddings.embed(sub_query_text).tolist(),
                entities=query.entities,
                topics=query.topics,
                metadata={
                    **dict(getattr(query, "metadata", {}) or {}),
                    "runtime_event": "query_flow",
                    "sub_query": sub_query_text,
                },
            )
            results = await self.retriever.retrieve(sub_query, top_k=8)
            all_results.extend(results)

            if results:
                evidence = self._evidence_texts(results, max_items=5)
                if evidence:
                    sub_answer = self.llm.generate_faithful(
                        sub_query_text,
                        evidence,
                        session_context=self._compose_layered_context(
                            sub_query,
                            context,
                            extra_instructions="Generate concise sub-answer for multi-step synthesis.",
                        ),
                    )
                    sub_answers.append(f"Q: {sub_query_text}\nA: {sub_answer}")

        seen_ids = set()
        unique_results: List[RetrievalResult] = []
        for result in all_results:
            if result.memory.id in seen_ids:
                continue
            seen_ids.add(result.memory.id)
            unique_results.append(result)

        if len(unique_results) >= 3:
            sorted_results = sorted(unique_results, key=lambda item: item.score, reverse=True)
            oracle_docs = [item.memory.content[:1500] for item in sorted_results[:5]]
            distractor_docs = [item.memory.content[:1500] for item in sorted_results[5:8]]
            final_answer = self.llm.raft_generate(query.raw_query, oracle_docs, distractor_docs)
        elif sub_answers:
            evidence = self._evidence_texts(unique_results, max_items=5)
            final_answer = self.llm.generate_faithful(
                query.raw_query,
                evidence,
                session_context=self._compose_layered_context(
                    query,
                    context,
                    extra_instructions="Synthesize all sub-answers into one coherent plan response.\n"
                    + "\n\n".join(sub_answers[:4]),
                ),
            )
        else:
            final_answer = "I don't have enough stored memories to answer this complex planning question yet."

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=final_answer,
            evidence=unique_results[:5],
            confidence=min(0.5 + len(unique_results) * 0.03, 0.9),
            reasoning_trace=(
                f"Planning agent: decomposed into {len(sub_queries)} sub-queries, "
                f"{len(unique_results)} unique memories, layered synthesis"
            ),
            sub_queries_used=sub_queries,
            processing_time_ms=elapsed,
        )


class ArbitrationAgent(BaseAgent):
    """Conflict-resolution agent with contradiction-aware synthesis."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="arbitration",
            llm=llm,
            retriever=retriever,
            prompt_key="arbitration",
            retrieval_top_k=15,
            evidence_top_k=8,
        )

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()
        results = await self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        evidence = self._evidence_texts(results, max_items=self.evidence_top_k)

        if evidence:
            answer = self.llm.generate_faithful(
                query.raw_query,
                evidence,
                session_context=self._compose_layered_context(
                    query,
                    context,
                    extra_instructions=(
                        "Resolve contradictions by ranking evidence quality, recency, and consistency. "
                        "State unresolved conflicts explicitly if they remain."
                    ),
                ),
            )
        else:
            answer = "I don't have enough stored memories to compare and resolve this conflict yet."

        if len(results) >= 2:
            sorted_results = sorted(results, key=lambda item: item.memory.timestamp.isoformat() if item.memory.timestamp else "")
            earliest = sorted_results[0].memory.content[:250]
            latest = sorted_results[-1].memory.content[:250]
            topic = query.topics[0] if query.topics else query.raw_query[:80]
            try:
                delta = self.llm.detect_belief_change(earliest, latest, topic)
                change_type = delta.get("change_type", "")
                explanation = delta.get("explanation", "")
                if explanation and change_type in ("contradiction", "refinement"):
                    answer += f"\n\nBelief change ({change_type}): {explanation}"
            except Exception:
                pass

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=0.7,
            reasoning_trace=(
                f"Arbitration agent: analyzed {len(results)} memories with layered conflict synthesis"
            ),
            processing_time_ms=elapsed,
        )


class AcademicIntelligenceAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="academic",
            llm=llm,
            retriever=retriever,
            extra_instructions="Prioritize subjects, deadlines, gaps, and exam readiness signals.",
            retrieval_top_k=14,
        )


class PersonalJournalingAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="journaling",
            llm=llm,
            retriever=retriever,
            extra_instructions="Preserve first-person reflection fidelity and privacy-sensitive framing.",
            retrieval_top_k=10,
        )


class PersonalWellbeingAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="wellbeing",
            llm=llm,
            retriever=retriever,
            extra_instructions="Focus on stress, sleep, energy, and safe non-medical wellbeing patterns.",
            retrieval_top_k=14,
        )


class CognitivePatternsAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="cognitive",
            llm=llm,
            retriever=retriever,
            extra_instructions="Identify repeat reasoning patterns using evidence-backed examples only.",
            retrieval_top_k=12,
        )


class DecisionLogAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="decisions",
            llm=llm,
            retriever=retriever,
            extra_instructions="Track options, rationale, and expected versus observed outcomes.",
            retrieval_top_k=12,
        )


class EmotionalIntelligenceAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="emotional",
            llm=llm,
            retriever=retriever,
            extra_instructions="Map mood episodes, trigger confidence, and recovery signatures.",
            retrieval_top_k=14,
        )


class BehavioralHabitsAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="behavioral",
            llm=llm,
            retriever=retriever,
            extra_instructions="Measure adherence, streaks, deviations, and intent-action gaps.",
            retrieval_top_k=12,
        )


class SocialIntelligenceAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="social",
            llm=llm,
            retriever=retriever,
            extra_instructions="Evaluate relationship dynamics, communication tone, and friction patterns.",
            retrieval_top_k=12,
        )


class GoalVisionAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="goals",
            llm=llm,
            retriever=retriever,
            extra_instructions="Track goal hierarchy, drift, blockers, and focus recommendations.",
            retrieval_top_k=14,
        )


class MetaLearningAgent(DomainSpecializedAgent):
    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__(
            name="meta_learning",
            llm=llm,
            retriever=retriever,
            extra_instructions="Synthesize cross-domain lessons with at least two supporting episodes.",
            retrieval_top_k=14,
        )


def build_specialized_agents(llm: LocalLLM, retriever: HybridRetriever) -> Dict[str, BaseAgent]:
    """Build the full 15-agent registry used by orchestrator routing."""

    return {
        "timeline": TimelineAgent(llm, retriever),
        "causal": CausalAgent(llm, retriever),
        "reflection": ReflectionAgent(llm, retriever),
        "planning": PlanningAgent(llm, retriever),
        "arbitration": ArbitrationAgent(llm, retriever),
        "academic": AcademicIntelligenceAgent(llm, retriever),
        "journaling": PersonalJournalingAgent(llm, retriever),
        "wellbeing": PersonalWellbeingAgent(llm, retriever),
        "cognitive": CognitivePatternsAgent(llm, retriever),
        "decisions": DecisionLogAgent(llm, retriever),
        "emotional": EmotionalIntelligenceAgent(llm, retriever),
        "behavioral": BehavioralHabitsAgent(llm, retriever),
        "social": SocialIntelligenceAgent(llm, retriever),
        "goals": GoalVisionAgent(llm, retriever),
        "meta_learning": MetaLearningAgent(llm, retriever),
    }
