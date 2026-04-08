"""
Agent Orchestrator for Cortex Lab — Fine-Tuned Model Integration
Routes queries to specialized agents based on intent and complexity.
Implements:
  - Adaptive-RAG routing with LLM-based structured JSON routing (Stage 2)
  - CRAG quality evaluation with multi-signal assessment
  - Self-RAG reflection with ISREL/ISSUP/ISUSE critique tokens (Stage 4)
  - FLARE: Forward-Looking Active Retrieval (EMNLP 2023)
  - RAFT: Distractor-aware generation (Stage 12)
  - Chain-of-Retrieval for complex multi-hop queries
  - Function calling integration (Stage 13)
"""

import asyncio
import hashlib
import json
import time
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.models import (
    AgentResponse, MemoryQuery, OrchestratorResponse, QueryIntent,
    RetrievalQuality, RetrievalResult, RoutingStrategy,
    PipelineTrace, PipelineStep, QueryTransformTrace,
    CRAGEvaluation, SelfRAGCritique, FLARETrace
)
from src.llm import LocalLLM
from src.retrieval.query_engine import QueryAnalyzer, QueryTransformer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.specialized import (
    SPECIALIZED_AGENT_ORDER,
    build_specialized_agents,
)
from src.observability import pipeline_events
from src.compression import ContextCompressor
from src.runtime.contracts import (
    ConflictResolutionPath,
    L1ExecutionPlan,
    PlanConfirmationGate,
    PlanConfirmationStatus,
    RuntimeExecutionMode,
    RuntimeLoopState,
    RuntimeRequestEnvelope,
    StopReason,
    TaskState,
)
from src.runtime.safety import PermissionStatus, SafeToolRuntime
from src.runtime.task_manager import RuntimeTaskManager


# ─── Tool Registry for Function Calling (Stage 13) ──────────────────────────

AVAILABLE_TOOLS = [
    {
        "name": "search_memories",
        "description": "Search through stored memories by semantic similarity",
        "parameters": {"query": "str", "top_k": "int (default 10)"},
    },
    {
        "name": "search_by_time",
        "description": "Find memories from a specific time period",
        "parameters": {"start_date": "str (ISO format)", "end_date": "str (ISO format)"},
    },
    {
        "name": "find_entity",
        "description": "Look up information about a specific entity (person, place, project)",
        "parameters": {"entity_name": "str"},
    },
    {
        "name": "trace_causal_chain",
        "description": "Trace cause-effect relationships for an event or decision",
        "parameters": {"event": "str"},
    },
    {
        "name": "detect_belief_evolution",
        "description": "Check how beliefs about a topic have changed over time",
        "parameters": {"topic": "str"},
    },
    {
        "name": "summarize_topic",
        "description": "Get a summary of all memories related to a topic",
        "parameters": {"topic": "str"},
    },
    {
        "name": "delete_memory",
        "description": "Delete one memory by ID (high-risk, approval required).",
        "parameters": {
            "memory_id": "str",
            "permission_id": "str (required after approval)",
        },
    },
]


class AgentOrchestrator:
    """
    Central orchestrator implementing Adaptive-RAG with fine-tuned model integration.
    
    Pipeline:
    1. LLM-based structured routing (Stage 2) with keyword fallback
    2. Query transformation (multi-query, HyDE, step-back, decomposition)
    3. Agent execution (single or multi-agent)
    4. CRAG quality evaluation (multi-signal)
    5. Self-RAG ISREL/ISSUP/ISUSE reflection (Stage 4)
    6. FLARE: Forward-looking active retrieval on low-confidence segments
    7. RAFT: Distractor-aware final generation (Stage 12)
    """

    def __init__(
        self,
        llm: LocalLLM,
        retriever: HybridRetriever,
        analyzer: QueryAnalyzer,
        transformer: QueryTransformer,
        safe_tool_runtime: Optional[SafeToolRuntime] = None,
        runtime_task_manager: Optional[RuntimeTaskManager] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.analyzer = analyzer
        self.transformer = transformer
        self.safe_tool_runtime = safe_tool_runtime
        self.runtime_task_manager = runtime_task_manager
        self.max_multi_agent_dispatch = 5
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_entries = 256

        # Context compression for evidence (Gap 2.3: LLMLingua-style)
        self.compressor = ContextCompressor(target_ratio=0.5, min_sentences=2)

        # Initialize specialized agents
        self.agents = build_specialized_agents(llm, retriever)

        # Domain-specialist keyword signals (used to select secondary agents)
        self.domain_signal_map: Dict[str, List[str]] = {
            "academic": ["exam", "study", "course", "assignment", "syllabus", "grade", "learning"],
            "journaling": ["journal", "diary", "reflection", "write", "entry", "day log"],
            "wellbeing": ["stress", "sleep", "wellbeing", "well-being", "burnout", "energy", "health"],
            "cognitive": ["thinking", "bias", "focus", "attention", "mental model", "cognitive"],
            "decisions": ["decision", "decide", "choice", "tradeoff", "option", "outcome"],
            "emotional": ["emotion", "emotional", "mood", "feeling", "anxiety", "recovery", "trigger"],
            "behavioral": ["habit", "routine", "pattern", "discipline", "consistency", "adherence"],
            "social": ["team", "friend", "family", "relationship", "communication", "conflict", "social"],
            "goals": ["goal", "vision", "target", "milestone", "objective", "roadmap"],
            "meta_learning": ["meta", "learn how", "improve learning", "strategy", "feedback loop", "iteration"],
        }

        # Intent → Agent mapping
        self.intent_to_agent = {
            QueryIntent.TEMPORAL: "timeline",
            QueryIntent.CAUSAL: "causal",
            QueryIntent.REFLECTIVE: "reflection",
            QueryIntent.COMPARATIVE: "arbitration",
            QueryIntent.FACTUAL: "planning",
            QueryIntent.PROCEDURAL: "planning",
            QueryIntent.EXPLORATORY: "planning",
        }

    def _new_runtime_loop_state(self, raw_query: str, session_context: str, trace_id: str) -> RuntimeLoopState:
        envelope = RuntimeRequestEnvelope(
            query=raw_query,
            request_id=trace_id,
            metadata={"session_context_chars": len(session_context or "")},
        )
        return RuntimeLoopState(envelope=envelope)

    @staticmethod
    def _format_tool_command(tool_name: str, arguments: Dict) -> str:
        try:
            args = json.dumps(arguments, sort_keys=True)
        except Exception:
            args = str(arguments)
        return f"{tool_name}({args})"[:1000]

    def _authorize_tool_execution(
        self,
        query: MemoryQuery,
        trace: PipelineTrace,
        tool_name: str,
        arguments: Dict,
    ) -> Optional[OrchestratorResponse]:
        if self.safe_tool_runtime is None:
            return None

        permission_id = str(arguments.get("permission_id", "")).strip()
        if permission_id:
            approved = self.safe_tool_runtime.permission_queue.get(permission_id)
            if approved and approved.status == PermissionStatus.APPROVED:
                if approved.tool_name == tool_name:
                    expected_memory = str((approved.metadata or {}).get("memory_id", "")).strip()
                    provided_memory = str(arguments.get("memory_id", "")).strip()
                    if not expected_memory or expected_memory == provided_memory:
                        trace.add_step(PipelineStep(
                            step_name=f"Tool Safety ({tool_name})",
                            step_type="tool_policy",
                            status="completed",
                            duration_ms=0,
                            details={
                                "decision": "approved",
                                "permission_id": permission_id,
                                "source": "human_approval",
                            },
                        ))
                        return None

        command_text = self._format_tool_command(tool_name, arguments)
        evaluation = self.safe_tool_runtime.evaluate_tool_operation(
            request_id=f"{trace.trace_id}:{tool_name}",
            tool_name=tool_name,
            command_text=command_text,
            metadata={
                "query": query.raw_query,
                "tool_name": tool_name,
                "arguments": arguments,
                "memory_id": arguments.get("memory_id", ""),
            },
        )

        effect = evaluation.decision.effect.value
        details = {
            "decision": effect,
            "rule_id": evaluation.decision.rule_id,
            "reason": evaluation.decision.reason,
            "signal_count": len(evaluation.dangerous_signals),
            "permission_id": evaluation.permission_request.permission_id if evaluation.permission_request else None,
        }
        trace.add_step(PipelineStep(
            step_name=f"Tool Safety ({tool_name})",
            step_type="tool_policy",
            status="completed" if effect == "allow" else "error",
            duration_ms=0,
            details=details,
        ))

        if effect == "allow":
            return None

        if effect == "require_approval" and evaluation.permission_request is not None:
            permission = evaluation.permission_request
            return OrchestratorResponse(
                answer=(
                    "This action is queued for approval before execution. "
                    f"Permission ID: {permission.permission_id}."
                ),
                thinking="Tool execution paused by SafeToolRuntime pending explicit approval.",
                agents_used=["function_calling", "safe_runtime_block"],
                confidence=0.25,
                reasoning_trace=(
                    f"Function call blocked pending approval: {tool_name} "
                    f"(permission_id={permission.permission_id})"
                ),
            )

        return OrchestratorResponse(
            answer=(
                "I blocked that action due to runtime safety policy. "
                f"Reason: {evaluation.decision.reason}"
            ),
            thinking="Tool execution denied by SafeToolRuntime policy.",
            agents_used=["function_calling", "safe_runtime_block"],
            confidence=0.2,
            reasoning_trace=f"Function call denied by policy: {tool_name}",
        )

    @staticmethod
    def _finalize_runtime_loop(
        trace: PipelineTrace,
        runtime_loop: RuntimeLoopState,
        reason: StopReason = StopReason.COMPLETED,
        note: str = "pipeline finished",
    ) -> None:
        if runtime_loop.stop_reason is None:
            runtime_loop.mark_stop(reason, note)
        trace.runtime_loop_state = runtime_loop.to_dict()
        trace.stop_reason = runtime_loop.stop_reason.value if runtime_loop.stop_reason else reason.value

    @staticmethod
    def _coordinator_task_id(trace_id: str) -> str:
        return f"coord-{trace_id}"

    @staticmethod
    def _subagent_task_id(trace_id: str, index: int, agent_name: str) -> str:
        safe_agent = re.sub(r"[^a-z0-9_-]", "_", agent_name.lower())
        return f"subagent-{trace_id}-{index:02d}-{safe_agent}"

    def _safe_transition_task(self, task_id: str, target_state: TaskState, note: str = "") -> None:
        if self.runtime_task_manager is None:
            return
        try:
            task = self.runtime_task_manager.get_task(task_id)
        except KeyError:
            return

        current = task.lifecycle.state
        if current == target_state:
            return
        if current in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
            return
        if task.lifecycle.can_transition_to(target_state):
            task.lifecycle.transition_to(target_state, note=note)

    @staticmethod
    def _tokenize_claim_text(text: str) -> set:
        tokens = re.findall(r"[a-z]{3,}", (text or "").lower())
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "about",
            "your", "have", "has", "had", "were", "was", "are", "not", "but", "can",
            "will", "should", "would", "could", "there", "their", "them", "they", "because",
        }
        return {token for token in tokens if token not in stopwords}

    @staticmethod
    def _contains_negation(text: str) -> bool:
        normalized = f" {(text or '').lower()} "
        markers = (" not ", " no ", " never ", " cannot ", " can't ", " won't ", "n\'t")
        return any(marker in normalized for marker in markers)

    def _detect_inter_agent_conflicts(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        if len(payloads) < 2:
            return conflicts

        for left_index in range(len(payloads) - 1):
            left = payloads[left_index]
            left_text = str(getattr(left.get("response"), "answer", "") or "")
            left_tokens = self._tokenize_claim_text(left_text)
            if not left_tokens:
                continue

            for right in payloads[left_index + 1 :]:
                right_text = str(getattr(right.get("response"), "answer", "") or "")
                right_tokens = self._tokenize_claim_text(right_text)
                if not right_tokens:
                    continue

                union = left_tokens | right_tokens
                overlap = (len(left_tokens & right_tokens) / len(union)) if union else 0.0
                polarity_mismatch = self._contains_negation(left_text) != self._contains_negation(right_text)
                explicit_conflict_signal = any(
                    marker in (left_text + " " + right_text).lower()
                    for marker in ("contradict", "conflict", "opposite", "inconsistent")
                )

                if overlap >= 0.24 and (polarity_mismatch or explicit_conflict_signal):
                    conflicts.append(
                        {
                            "agent_a": left.get("agent", ""),
                            "agent_b": right.get("agent", ""),
                            "overlap": round(overlap, 3),
                            "reason": "polarity_mismatch" if polarity_mismatch else "explicit_conflict_signal",
                            "answer_a_preview": left_text[:220],
                            "answer_b_preview": right_text[:220],
                        }
                    )

        return conflicts

    def _should_require_plan_confirmation(self, query: MemoryQuery, agent_names: List[str]) -> bool:
        metadata = dict(getattr(query, "metadata", {}) or {})
        if "plan_confirmation_required" in metadata:
            return bool(metadata.get("plan_confirmation_required"))

        high_risk_phrases = (
            "delete memory",
            "erase memory",
            "remove memory",
            "overwrite memory",
            "bulk update",
            "rewrite my profile",
            "store this permanently",
        )
        query_lower = (query.raw_query or "").lower()
        if any(phrase in query_lower for phrase in high_risk_phrases):
            return True

        if metadata.get("high_stakes_inference") and len(agent_names) >= 3:
            return True

        return False

    def _build_plan_confirmation_gate(self, query: MemoryQuery, agent_names: List[str]) -> PlanConfirmationGate:
        metadata = dict(getattr(query, "metadata", {}) or {})
        requires_confirmation = self._should_require_plan_confirmation(query, agent_names)

        reasons: List[str] = []
        if requires_confirmation:
            reasons.append("Plan includes memory-impacting or high-stakes operations.")
            if len(agent_names) >= 4:
                reasons.append("Plan fanout is high; explicit user confirmation is safer.")

        gate = PlanConfirmationGate(required=requires_confirmation, reasons=reasons)

        if requires_confirmation and bool(metadata.get("plan_confirmed")):
            actor = str(metadata.get("plan_confirmed_by", "runtime"))
            note = str(metadata.get("plan_confirmation_note", ""))
            gate.mark_confirmed(actor=actor, note=note)
        elif requires_confirmation and bool(metadata.get("plan_denied")):
            actor = str(metadata.get("plan_denied_by", "runtime"))
            note = str(metadata.get("plan_denial_note", ""))
            gate.mark_denied(actor=actor, note=note)

        return gate

    def _build_l1_execution_plan(self, query: MemoryQuery, agent_names: List[str], primary_agent_name: str) -> L1ExecutionPlan:
        intent_label = query.intent.value if hasattr(query.intent, "value") else str(query.intent)
        metadata = dict(getattr(query, "metadata", {}) or {})

        if len(agent_names) <= 1:
            execution_mode = RuntimeExecutionMode.SINGLE_STEP
        elif query.sub_queries:
            execution_mode = RuntimeExecutionMode.MULTI_STEP_SEQUENTIAL
        else:
            execution_mode = RuntimeExecutionMode.MULTI_STEP_PARALLEL

        if len(agent_names) >= 3 or query.complexity >= 0.75:
            execution_mode = RuntimeExecutionMode.PLAN_MODE

        potential_conflicts: List[str] = []
        if query.intent == QueryIntent.COMPARATIVE:
            potential_conflicts.append("Comparative intent often yields contradictory evidence candidates.")

        domain_specialists = [
            name
            for name in agent_names
            if name not in {"timeline", "causal", "reflection", "planning", "arbitration"}
        ]
        if len(domain_specialists) >= 2:
            potential_conflicts.append("Mixed-domain specialist synthesis may require arbitration.")

        q_lower = (query.raw_query or "").lower()
        if any(token in q_lower for token in ("compare", "versus", "vs", "conflict", "contradict", "tradeoff")):
            potential_conflicts.append("Query phrasing indicates explicit conflict or tradeoff analysis.")

        conflict_path = ConflictResolutionPath.ARBITRATION_FIRST
        if str(metadata.get("conflict_resolution_path", "")).lower() == "synthesis_first":
            conflict_path = ConflictResolutionPath.SYNTHESIS_FIRST

        confirmation_gate = self._build_plan_confirmation_gate(query, agent_names)

        return L1ExecutionPlan(
            query=query.raw_query,
            intent=intent_label,
            complexity=query.complexity,
            primary_agent=primary_agent_name,
            selected_agents=list(agent_names),
            execution_mode=execution_mode,
            conflict_resolution_path=conflict_path,
            potential_conflicts=potential_conflicts,
            confirmation_gate=confirmation_gate,
            metadata={
                "routing": query.routing.value if hasattr(query.routing, "value") else str(query.routing),
                "sub_query_count": len(query.sub_queries),
            },
        )

    def _build_coordinator_plan(
        self,
        query: MemoryQuery,
        agent_names: List[str],
        primary_agent_name: str,
        execution_plan: Optional[L1ExecutionPlan] = None,
    ) -> Dict[str, Any]:
        plan = execution_plan or self._build_l1_execution_plan(query, agent_names, primary_agent_name)
        return {
            "strategy": "parallel_multi_agent",
            "query_intent": query.intent.value,
            "query_complexity": round(query.complexity, 3),
            "primary_agent": primary_agent_name,
            "subagent_count": len(agent_names),
            "execution_mode": plan.execution_mode.value,
            "conflict_resolution_path": plan.conflict_resolution_path.value,
            "potential_conflicts": list(plan.potential_conflicts),
            "confirmation_gate": plan.confirmation_gate.to_dict(),
            "subagents": [
                {
                    "agent": name,
                    "role": "primary" if name == primary_agent_name else "support",
                    "reason": (
                        "primary intent handler"
                        if name == primary_agent_name
                        else "intent-supporting decomposition"
                    ),
                }
                for name in agent_names
            ],
            "plan_mode": plan.to_dict(),
        }

    def _select_domain_specialists(self, raw_query: str) -> List[str]:
        """Select domain specialists based on lexical signals in the user query."""
        q = (raw_query or "").lower()
        selected: List[str] = []

        for agent_name in SPECIALIZED_AGENT_ORDER:
            if agent_name in {"timeline", "causal", "reflection", "planning", "arbitration"}:
                continue
            keywords = self.domain_signal_map.get(agent_name, [])
            if any(keyword in q for keyword in keywords):
                selected.append(agent_name)

        return selected

    def _make_cache_key(self, raw_query: str, session_context: str = "") -> str:
        """Build a stable, session-aware cache key (legacy compatibility API)."""
        query_norm = (raw_query or "").strip().lower()
        ctx = (session_context or "").strip()
        session_hash = hashlib.md5(ctx.encode("utf-8")).hexdigest()[:8] if ctx else "no_ctx"
        return f"{query_norm}|{session_hash}"

    def _cache_get(self, raw_query: str, session_context: str = "") -> Optional[Dict[str, Any]]:
        """Get a cached payload by normalized query+context key (legacy compatibility API)."""
        return self._response_cache.get(self._make_cache_key(raw_query, session_context))

    def _cache_put(
        self,
        raw_query: str,
        session_context: str,
        payload: Dict[str, Any],
    ) -> str:
        """Store payload in the lightweight orchestrator response cache (legacy compatibility API)."""
        key = self._make_cache_key(raw_query, session_context)
        if len(self._response_cache) >= self._cache_max_entries:
            oldest_key = next(iter(self._response_cache))
            self._response_cache.pop(oldest_key, None)
        self._response_cache[key] = payload
        return key

    def _select_agents(self, query: MemoryQuery, force_multi_step: bool = False) -> List[str]:
        """Select agents for the current query (legacy-compatible helper)."""
        primary_agent_name = self.intent_to_agent.get(query.intent, "planning")
        agent_names: List[str] = [primary_agent_name]

        if not force_multi_step and query.routing == RoutingStrategy.SINGLE_STEP:
            if query.intent in {QueryIntent.FACTUAL, QueryIntent.EXPLORATORY, QueryIntent.PROCEDURAL}:
                specialists = self._select_domain_specialists(query.raw_query)
                if specialists:
                    candidate = specialists[0]
                    if candidate in self.agents:
                        return [candidate]
            return agent_names

        if force_multi_step or query.routing == RoutingStrategy.MULTI_STEP:
            if query.intent == QueryIntent.CAUSAL:
                agent_names.append("timeline")
            elif query.intent == QueryIntent.REFLECTIVE:
                agent_names.append("causal")
            elif query.intent == QueryIntent.TEMPORAL:
                agent_names.append("reflection")

            if "planning" not in agent_names and query.sub_queries:
                agent_names.append("planning")

            for specialist in self._select_domain_specialists(query.raw_query):
                if len(agent_names) >= self.max_multi_agent_dispatch:
                    break
                if specialist in self.agents and specialist not in agent_names:
                    agent_names.append(specialist)

        return agent_names[: self.max_multi_agent_dispatch]

    async def process(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """
        Full orchestration pipeline with fine-tuned model integration.
        Optimized to minimize LLM calls for latency reduction.
        Includes full pipeline observability trace.
        """
        t0 = time.time()
        trace = PipelineTrace(query=raw_query)
        trace_id = trace.trace_id  # Alias for event emissions
        runtime_loop = self._new_runtime_loop_state(raw_query, session_context, trace_id)

        print(f"\n{'='*60}")
        print(f"  🧠 Orchestrator: Processing query")
        print(f"  📝 Query: {raw_query[:80]}...")
        print(f"{'='*60}")

        # Emit pipeline start event for real-time visualization
        await pipeline_events.emit_pipeline_start(trace_id, raw_query)

        # 1. Analyze query (keyword heuristics — fast, no LLM call)
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "Query Analysis", "query_analysis")
        t_step = time.time()
        query = self.analyzer.analyze(raw_query)
        analysis_ms = (time.time() - t_step) * 1000
        await pipeline_events.emit_step_complete(trace_id, "Query Analysis", "query_analysis", analysis_ms, {
            "intent": query.intent.value, "complexity": round(query.complexity, 2), "routing": query.routing.value,
            "entities_found": len(query.entities),
        })

        trace.query_analysis = {
            "intent": query.intent.value,
            "complexity": round(query.complexity, 2),
            "routing": query.routing.value,
            "entities": query.entities,
            "topics": query.topics,
            "time_start": query.time_start.isoformat() if query.time_start else None,
            "time_end": query.time_end.isoformat() if query.time_end else None,
        }
        query_metadata = dict(getattr(query, "metadata", {}) or {})
        query_metadata.setdefault("runtime_event", "query_flow")
        query_metadata.setdefault("runtime_mode", "cloud")
        query_metadata.setdefault("llm_provider", self.llm.__class__.__name__)
        query_metadata.setdefault(
            "permission_chain",
            "schema->scope->resource->privacy->user_permission->audit",
        )
        query_metadata.setdefault("privacy_tier", "default")
        query_metadata["trace_id"] = trace_id
        query_metadata["routing_strategy"] = query.routing.value
        query.metadata = query_metadata
        trace.add_step(PipelineStep(
            step_name="Query Analysis",
            step_type="query_analysis",
            status="completed",
            duration_ms=analysis_ms,
            details={
                "intent": query.intent.value,
                "complexity": round(query.complexity, 2),
                "routing": query.routing.value,
                "entities_found": len(query.entities),
                "method": "keyword_heuristics",
            }
        ))

        # 1b. Only use LLM routing when keyword analysis is truly ambiguous
        # Skip LLM routing for clear-cut queries to save 2-4s
        llm_routed = False
        if 0.35 < query.complexity < 0.65 and self.llm.model is not None:
            await pipeline_events.emit_step_start(trace_id, "LLM Routing", "routing")
            t_step = time.time()
            query = await self._llm_route_query(query, session_context)
            llm_route_ms = (time.time() - t_step) * 1000
            llm_routed = True
            await pipeline_events.emit_step_complete(trace_id, "LLM Routing", "routing", llm_route_ms, {
                "refined_intent": query.intent.value, "refined_complexity": round(query.complexity, 2),
            })
            trace.add_step(PipelineStep(
                step_name="LLM Routing",
                step_type="routing",
                status="completed",
                duration_ms=llm_route_ms,
                details={
                    "reason": "keyword_ambiguous",
                    "refined_intent": query.intent.value,
                    "refined_complexity": round(query.complexity, 2),
                }
            ))
        else:
            await pipeline_events.emit_step_skip(trace_id, "LLM Routing", "routing", "keyword_confident")
            trace.add_step(PipelineStep(
                step_name="LLM Routing",
                step_type="routing",
                status="skipped",
                duration_ms=0,
                details={"reason": "keyword_confident" if query.complexity <= 0.35 or query.complexity >= 0.65 else "no_llm"}
            ))

        # 2. Transform query (add multi-query, HyDE, etc.)
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "Query Transformation", "query_transform")
        t_step = time.time()
        query = self.transformer.transform(query)
        transform_ms = (time.time() - t_step) * 1000

        total_variants = len(query.multi_queries) + (1 if query.hyde_answer else 0) + (1 if query.step_back_query else 0) + len(query.sub_queries)
        await pipeline_events.emit_step_complete(trace_id, "Query Transformation", "query_transform", transform_ms, {
            "total_variants": total_variants,
            "multi_queries": len(query.multi_queries),
            "hyde_generated": bool(query.hyde_answer),
        })

        trace.query_transform = QueryTransformTrace(
            original_query=raw_query,
            multi_queries=query.multi_queries,
            hyde_answer=query.hyde_answer,
            step_back_query=query.step_back_query,
            sub_queries=query.sub_queries,
            total_variants=total_variants,
            duration_ms=transform_ms,
        )
        trace.add_step(PipelineStep(
            step_name="Query Transformation",
            step_type="query_transform",
            status="completed",
            duration_ms=transform_ms,
            details={
                "multi_queries": len(query.multi_queries),
                "hyde_generated": bool(query.hyde_answer),
                "step_back_generated": bool(query.step_back_query),
                "sub_queries": len(query.sub_queries),
                "total_variants": trace.query_transform.total_variants,
            }
        ))

        trace.routing_decision = query.routing.value

        # 2b. Function calling check (Stage 13) — try tool use before agent execution
        tool_response = None
        if self.llm.model is not None and query.routing != RoutingStrategy.NO_RETRIEVAL:
            t_step = time.time()
            tool_response = await self._try_function_calling(query, trace, runtime_loop=runtime_loop)
            fc_ms = (time.time() - t_step) * 1000
            if tool_response:
                trace.add_step(PipelineStep(
                    step_name="Function Calling",
                    step_type="function_calling",
                    status="completed",
                    duration_ms=fc_ms,
                    details={"tool_used": True}
                ))
            else:
                trace.add_step(PipelineStep(
                    step_name="Function Calling",
                    step_type="function_calling",
                    status="skipped",
                    duration_ms=fc_ms,
                    details={"reason": "no_tool_needed"}
                ))

        # 3. Route based on complexity (skip if function calling handled it)
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "Agent Execution", "agent_execution")
        t_step = time.time()
        if tool_response:
            response = tool_response
            if "tool_rate_limited" in response.agents_used:
                route_label = "FUNCTION_CALL_RATE_LIMITED"
                runtime_loop.mark_stop(StopReason.RATE_LIMITED, "Tool dispatch blocked by deterministic rate-limit window")
            elif "safe_runtime_block" in response.agents_used:
                route_label = "FUNCTION_CALL_BLOCKED"
                runtime_loop.mark_stop(StopReason.POLICY_DENIED, "Tool execution blocked by SafeToolRuntime")
            else:
                route_label = "FUNCTION_CALL"
        elif query.routing == RoutingStrategy.NO_RETRIEVAL:
            response = await self._handle_no_retrieval(query)
            route_label = "NO_RETRIEVAL"
        elif query.routing == RoutingStrategy.SINGLE_STEP:
            response = await self._handle_single_step(query)
            route_label = "SINGLE_STEP"
        else:
            response = await self._handle_multi_step(query, trace=trace)
            if "coordinator_cancelled" in response.agents_used:
                route_label = "MULTI_STEP_CANCELLED"
                runtime_loop.mark_stop(StopReason.CANCELLED, "Multi-agent subagent execution cancelled")
            else:
                route_label = "MULTI_STEP"
        route_ms = (time.time() - t_step) * 1000
        await pipeline_events.emit_step_complete(trace_id, f"Agent Execution ({route_label})", "agent_execution", route_ms, {
            "routing": route_label,
            "agents": response.agents_used,
            "evidence_count": len(response.evidence),
            "initial_confidence": round(response.confidence, 3),
        })

        # 3b. Context Compression — reduce evidence noise before LLM generation
        compression_metrics = None
        if response.evidence and len(response.evidence) > 2:
            await pipeline_events.emit_step_start(trace_id, "Context Compression", "compression")
            t_comp = time.time()
            evidence_texts = [r.memory.content for r in response.evidence]
            compressed_texts, compression_metrics = self.compressor.compress_evidence(
                query.raw_query, evidence_texts,
                entities=query.entities,
                max_total_chars=4000 if query.complexity >= 0.6 else 2500,
            )
            # Update evidence with compressed content
            for i, r in enumerate(response.evidence):
                if i < len(compressed_texts):
                    r.evidence_text = compressed_texts[i]
            comp_ms = (time.time() - t_comp) * 1000
            await pipeline_events.emit_step_complete(trace_id, "Context Compression", "compression", comp_ms, compression_metrics or {})
            await pipeline_events.emit_metric(trace_id, "compression_invocations", 1)

            trace.add_step(PipelineStep(
                step_name="Context Compression",
                step_type="compression",
                status="completed",
                duration_ms=comp_ms,
                details=compression_metrics or {},
            ))

        # 3c. Memory Importance Boost — boost retrieval scores by importance
        if response.evidence:
            for r in response.evidence:
                importance = getattr(r.memory, 'importance', 0.5)
                if importance > 0.6:
                    boost = 0.05 * (importance - 0.5)
                    r.score = min(r.score + boost, 1.0)
            # Re-sort by boosted score
            response.evidence.sort(key=lambda r: r.score, reverse=True)
            await pipeline_events.emit_metric(trace_id, "importance_boosts_applied", 1)

        # Collect retrieval channel traces
        if hasattr(self.retriever, '_last_channel_traces'):
            trace.retrieval_channels = getattr(self.retriever, '_last_channel_traces', [])
            retrieval_trace = self.retriever.get_last_retrieval_trace()
            trace.reranking = {
                "method": retrieval_trace.get("rerank_method", "unknown"),
                "duration_ms": retrieval_trace.get("rerank_ms", 0),
                "input_count": retrieval_trace.get("fused_count", 0),
            }

        # Agent invocation trace
        agent_name = self.intent_to_agent.get(query.intent, "planning")
        trace.agents_invoked = [{
            "agent": a,
            "is_primary": a == agent_name,
        } for a in response.agents_used]

        trace.add_step(PipelineStep(
            step_name=f"Agent Execution ({route_label})",
            step_type="agent_execution",
            status="completed",
            duration_ms=route_ms,
            details={
                "routing": route_label,
                "agents": response.agents_used,
                "evidence_count": len(response.evidence),
                "initial_confidence": round(response.confidence, 3),
            }
        ))

        # Attach trace BEFORE CRAG/Self-RAG/FLARE so they can write their evaluations
        response.pipeline_trace = trace

        # 4. CRAG quality evaluation (fast — no LLM call, just scoring)
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "CRAG Quality Evaluation", "crag")
        t_step = time.time()
        pre_crag_confidence = response.confidence
        response = await self._crag_evaluate(query, response)
        crag_ms = (time.time() - t_step) * 1000

        crag_eval = trace.crag_evaluation
        crag_details = {
            "verdict": crag_eval.verdict if crag_eval else "no_evidence",
            "quality_score": round(crag_eval.quality_score, 3) if crag_eval else 0,
            "avg_score": round(crag_eval.avg_evidence_score, 3) if crag_eval else 0,
            "max_score": round(crag_eval.max_evidence_score, 3) if crag_eval else 0,
            "entity_coverage": round(crag_eval.entity_coverage, 3) if crag_eval else 0,
            "evidence_count": crag_eval.evidence_count if crag_eval else 0,
            "confidence_delta": round(response.confidence - pre_crag_confidence, 3),
        }
        trace.add_step(PipelineStep(
            step_name="CRAG Quality Evaluation",
            step_type="crag",
            status="completed",
            duration_ms=crag_ms,
            details=crag_details,
        ))
        await pipeline_events.emit_step_complete(trace_id, "CRAG Quality Evaluation", "crag", crag_ms, crag_details)

        # 5. Self-RAG reflection — trigger for moderate confidence answers
        # Threshold 0.80: most single-agent responses land at 0.70-0.85
        if (response.evidence and response.confidence < 0.80
                and len(response.answer.strip()) > 20):
            runtime_loop.register_iteration()
            await pipeline_events.emit_step_start(trace_id, "Self-RAG Critique", "self_rag")
            t_step = time.time()
            pre_selfrag_confidence = response.confidence
            response = await self._self_rag_critique(query, response)
            selfrag_ms = (time.time() - t_step) * 1000

            selfrag_details = {
                "verdict": trace.self_rag_critique.verdict if trace.self_rag_critique else "unknown",
                "confidence_delta": round(response.confidence - pre_selfrag_confidence, 3),
                "revision_applied": trace.self_rag_critique.revision_applied if trace.self_rag_critique else False,
            }
            trace.add_step(PipelineStep(
                step_name="Self-RAG Critique",
                step_type="self_rag",
                status="completed",
                duration_ms=selfrag_ms,
                details=selfrag_details,
            ))
            await pipeline_events.emit_step_complete(trace_id, "Self-RAG Critique", "self_rag", selfrag_ms, selfrag_details)
        else:
            skip_reason = "no_evidence" if not response.evidence else (
                "confidence_sufficient" if response.confidence >= 0.80 else "answer_too_short"
            )
            trace.add_step(PipelineStep(
                step_name="Self-RAG Critique",
                step_type="self_rag",
                status="skipped",
                duration_ms=0,
                details={"reason": skip_reason, "confidence": round(response.confidence, 3)}
            ))
            await pipeline_events.emit_step_skip(trace_id, "Self-RAG Critique", "self_rag", skip_reason)

        # 6. FLARE — for low confidence answers after Self-RAG
        # Triggers when confidence remains below 0.55 after Self-RAG
        if (response.confidence < 0.55 and response.evidence
                and len(response.answer.strip()) > 20):
            runtime_loop.register_iteration()
            await pipeline_events.emit_step_start(trace_id, "FLARE Active Retrieval", "flare")
            t_step = time.time()
            pre_flare_confidence = response.confidence
            response = await self._flare_active_retrieval(query, response)
            flare_ms = (time.time() - t_step) * 1000

            flare_details = {
                "new_evidence": trace.flare_trace.new_evidence_count if trace.flare_trace else 0,
                "answer_revised": trace.flare_trace.answer_revised if trace.flare_trace else False,
                "confidence_delta": round(response.confidence - pre_flare_confidence, 3),
            }
            trace.add_step(PipelineStep(
                step_name="FLARE Active Retrieval",
                step_type="flare",
                status="completed",
                duration_ms=flare_ms,
                details=flare_details,
            ))
            await pipeline_events.emit_step_complete(trace_id, "FLARE Active Retrieval", "flare", flare_ms, flare_details)
        else:
            skip_reason = "confidence_sufficient" if response.confidence >= 0.55 else (
                "no_evidence" if not response.evidence else "answer_too_short"
            )
            trace.add_step(PipelineStep(
                step_name="FLARE Active Retrieval",
                step_type="flare",
                status="skipped",
                duration_ms=0,
                details={"reason": skip_reason}
            ))
            await pipeline_events.emit_step_skip(trace_id, "FLARE Active Retrieval", "flare", skip_reason)

        response.query_analysis = query
        response.processing_time_ms = (time.time() - t0) * 1000

        # Track token usage
        response.token_usage = self.llm.get_stats()

        # Finalize trace
        trace.total_duration_ms = response.processing_time_ms
        trace.final_confidence = response.confidence
        trace.evidence_count = len(response.evidence)
        trace.token_usage = response.token_usage
        trace.generation_details = {
            "model": "DeepSeek-R1-7B (Fine-Tuned)",
            "quantization": "4-bit",
        }
        trace.cache_status = {"hit": False, "level": None}
        self._finalize_runtime_loop(trace, runtime_loop)
        response.pipeline_trace = trace

        # Emit pipeline complete event
        await pipeline_events.emit_pipeline_complete(trace_id, response.processing_time_ms, {
            "confidence": round(response.confidence, 3),
            "evidence_count": len(response.evidence),
            "agents_used": response.agents_used,
            "steps_total": len(trace.steps),
            "stop_reason": trace.stop_reason,
            "runtime_iterations": trace.runtime_loop_state.get("iterations_executed", 0),
            "runtime_tool_calls": trace.runtime_loop_state.get("tool_calls_executed", 0),
        })

        print(f"\n  ✅ Response ready: confidence={response.confidence:.2f}, "
              f"agents={response.agents_used}, time={response.processing_time_ms:.0f}ms\n")

        return response

    async def retrieve_only(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """
        Retrieval-only pipeline: routes query, retrieves evidence, evaluates quality
        but does NOT generate a final LLM answer. Used for streaming mode where the
        server will stream the answer token-by-token after receiving evidence.
        Includes pipeline trace for observability.
        """
        t0 = time.time()
        trace = PipelineTrace(query=raw_query)
        trace_id = raw_query  # Use query as trace identifier for events
        runtime_loop = self._new_runtime_loop_state(raw_query, session_context, trace.trace_id)

        print(f"\n{'='*60}")
        print(f"  🔍 Orchestrator: Retrieve-only mode")
        print(f"  📝 Query: {raw_query[:80]}...")
        print(f"{'='*60}")

        await pipeline_events.emit_pipeline_start(trace_id, raw_query)

        # 1. Analyze query
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "Query Analysis", "query_analysis")
        t_step = time.time()
        query = self.analyzer.analyze(raw_query)
        analysis_ms = (time.time() - t_step) * 1000

        analysis_details = {
            "intent": query.intent.value,
            "complexity": round(query.complexity, 2),
            "routing": query.routing.value,
            "entities": query.entities,
            "topics": query.topics,
        }
        trace.query_analysis = analysis_details
        trace.add_step(PipelineStep(
            step_name="Query Analysis",
            step_type="query_analysis",
            status="completed",
            duration_ms=analysis_ms,
            details={"intent": query.intent.value, "complexity": round(query.complexity, 2), "routing": query.routing.value}
        ))
        await pipeline_events.emit_step_complete(trace_id, "Query Analysis", "query_analysis", analysis_ms, analysis_details)

        # 1b. LLM routing only for ambiguous queries
        if 0.35 < query.complexity < 0.65 and self.llm.model is not None:
            await pipeline_events.emit_step_start(trace_id, "LLM Routing", "routing")
            t_step = time.time()
            query = await self._llm_route_query(query, session_context)
            routing_ms = (time.time() - t_step) * 1000
            trace.add_step(PipelineStep(
                step_name="LLM Routing", step_type="routing", status="completed",
                duration_ms=routing_ms,
                details={"refined_intent": query.intent.value}
            ))
            await pipeline_events.emit_step_complete(trace_id, "LLM Routing", "routing", routing_ms, {"refined_intent": query.intent.value})
        else:
            trace.add_step(PipelineStep(step_name="LLM Routing", step_type="routing", status="skipped"))
            await pipeline_events.emit_step_skip(trace_id, "LLM Routing", "routing", "complexity_outside_range")

        # 2. Transform query
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "Query Transformation", "query_transform")
        t_step = time.time()
        query = self.transformer.transform(query)
        transform_ms = (time.time() - t_step) * 1000

        trace.query_transform = QueryTransformTrace(
            original_query=raw_query,
            multi_queries=query.multi_queries,
            hyde_answer=query.hyde_answer,
            step_back_query=query.step_back_query,
            sub_queries=query.sub_queries,
            total_variants=len(query.multi_queries) + (1 if query.hyde_answer else 0) + (1 if query.step_back_query else 0) + len(query.sub_queries),
            duration_ms=transform_ms,
        )
        trace.add_step(PipelineStep(
            step_name="Query Transformation", step_type="query_transform", status="completed",
            duration_ms=transform_ms,
            details={"total_variants": trace.query_transform.total_variants}
        ))
        await pipeline_events.emit_step_complete(trace_id, "Query Transformation", "query_transform", transform_ms, {"total_variants": trace.query_transform.total_variants})

        trace.routing_decision = query.routing.value

        # 3. Retrieve evidence (no LLM generation)
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "Evidence Retrieval", "agent_execution")
        t_step = time.time()
        if query.routing == RoutingStrategy.NO_RETRIEVAL:
            response = OrchestratorResponse(
                answer="",
                thinking="Simple query — no memory retrieval needed.",
                agents_used=["direct"],
                confidence=0.8,
                reasoning_trace="Direct answer (retrieve-only, no evidence needed)",
            )
        else:
            # Use the retriever directly to get evidence without agent LLM calls
            results = await self.retriever.retrieve(query)
            agent_name = self.intent_to_agent.get(query.intent, "planning")

            thinking = (
                f"Intent: {query.intent.value} (complexity: {query.complexity:.2f})\n"
                f"Agent: {agent_name}\n"
                f"Evidence: {len(results)} memories retrieved (retrieve-only)"
            )

            # Multi-signal confidence (not just avg_score)
            # For broad/exploratory queries, evidence count matters more than individual scores
            if results:
                avg_score = sum(r.score for r in results) / len(results)
                max_score = max(r.score for r in results)
                evidence_count = len(results)

                # Entity coverage: check if query entities appear in evidence
                entity_coverage = 0.0
                if query.entities:
                    matched = sum(
                        1 for ent in query.entities
                        if any(ent.lower() in r.memory.content.lower() for r in results)
                    )
                    entity_coverage = matched / len(query.entities)

                # Multi-signal confidence formula (mirrors CRAG scoring)
                confidence = (
                    0.30 * min(avg_score * 1.5, 1.0) +   # Average relevance (boosted)
                    0.20 * min(max_score * 1.2, 1.0) +   # Best match quality
                    0.25 * min(evidence_count / 5.0, 1.0) +  # Evidence breadth
                    0.25 * entity_coverage                 # Entity coverage
                )
                confidence = min(max(confidence, 0.15), 0.95)

                # Exploratory queries with good evidence breadth get a confidence floor
                if query.intent == QueryIntent.EXPLORATORY and evidence_count >= 3:
                    confidence = max(confidence, 0.45)
            else:
                avg_score = 0.0
                confidence = 0.3

            response = OrchestratorResponse(
                answer="",  # No answer — will be streamed by the server
                thinking=thinking,
                evidence=results,
                agents_used=[agent_name],
                confidence=confidence,
                reasoning_trace=f"Retrieve-only: {len(results)} results via {agent_name}",
            )

        retrieve_ms = (time.time() - t_step) * 1000

        # Collect retrieval channel traces
        if hasattr(self.retriever, '_last_channel_traces'):
            trace.retrieval_channels = getattr(self.retriever, '_last_channel_traces', [])
            retrieval_trace = self.retriever.get_last_retrieval_trace()
            trace.reranking = {
                "method": retrieval_trace.get("rerank_method", "unknown"),
                "duration_ms": retrieval_trace.get("rerank_ms", 0),
            }

        retrieval_details = {"evidence_count": len(response.evidence), "mode": "retrieve_only"}
        trace.add_step(PipelineStep(
            step_name="Evidence Retrieval", step_type="agent_execution", status="completed",
            duration_ms=retrieve_ms,
            details=retrieval_details,
        ))
        await pipeline_events.emit_step_complete(trace_id, "Evidence Retrieval", "agent_execution", retrieve_ms, retrieval_details)

        # 3b. Context Compression (retrieve-only mode)
        if response.evidence:
            await pipeline_events.emit_step_start(trace_id, "Context Compression", "compression")
            t_comp = time.time()
            evidence_texts = [r.memory.content for r in response.evidence]
            entities = query.entities if hasattr(query, 'entities') else []
            compressed_texts, comp_metrics = self.compressor.compress_evidence(
                query.raw_query, evidence_texts, entities
            )
            for i, r in enumerate(response.evidence):
                if i < len(compressed_texts):
                    r.memory.content = compressed_texts[i]
            comp_ms = (time.time() - t_comp) * 1000
            trace.add_step(PipelineStep(
                step_name="Context Compression", step_type="compression", status="completed",
                duration_ms=comp_ms,
                details=comp_metrics,
            ))
            await pipeline_events.emit_step_complete(trace_id, "Context Compression", "compression", comp_ms, comp_metrics)
            await pipeline_events.emit_metric(trace_id, "compression_ratio", comp_metrics.get("compression_ratio", 1.0))
        else:
            await pipeline_events.emit_step_skip(trace_id, "Context Compression", "compression", "no_evidence")

        # 4. CRAG quality evaluation (fast — no LLM call)
        runtime_loop.register_iteration()
        await pipeline_events.emit_step_start(trace_id, "CRAG Quality Evaluation", "crag")
        t_step = time.time()
        response = await self._crag_evaluate(query, response)
        crag_ms = (time.time() - t_step) * 1000

        crag_details = {"verdict": trace.crag_evaluation.verdict if trace.crag_evaluation else "no_evidence"}
        trace.add_step(PipelineStep(
            step_name="CRAG Quality Evaluation", step_type="crag", status="completed",
            duration_ms=crag_ms,
            details=crag_details,
        ))
        await pipeline_events.emit_step_complete(trace_id, "CRAG Quality Evaluation", "crag", crag_ms, crag_details)

        # Self-RAG and FLARE are skipped in retrieve-only mode
        trace.add_step(PipelineStep(step_name="Self-RAG Critique", step_type="self_rag", status="skipped",
                                     details={"reason": "retrieve_only_mode"}))
        await pipeline_events.emit_step_skip(trace_id, "Self-RAG Critique", "self_rag", "retrieve_only_mode")
        trace.add_step(PipelineStep(step_name="FLARE Active Retrieval", step_type="flare", status="skipped",
                                     details={"reason": "retrieve_only_mode"}))
        await pipeline_events.emit_step_skip(trace_id, "FLARE Active Retrieval", "flare", "retrieve_only_mode")

        response.query_analysis = query
        response.processing_time_ms = (time.time() - t0) * 1000

        # Finalize trace
        trace.total_duration_ms = response.processing_time_ms
        trace.final_confidence = response.confidence
        trace.evidence_count = len(response.evidence)
        trace.cache_status = {"hit": False, "level": None}
        self._finalize_runtime_loop(trace, runtime_loop)
        response.pipeline_trace = trace

        # Emit pipeline complete
        await pipeline_events.emit_pipeline_complete(trace_id, response.processing_time_ms, {
            "confidence": round(response.confidence, 3),
            "evidence_count": len(response.evidence),
            "mode": "retrieve_only",
            "steps_total": len(trace.steps),
            "stop_reason": trace.stop_reason,
            "runtime_iterations": trace.runtime_loop_state.get("iterations_executed", 0),
            "runtime_tool_calls": trace.runtime_loop_state.get("tool_calls_executed", 0),
        })

        print(f"\n  ✅ Retrieval ready: confidence={response.confidence:.2f}, "
              f"evidence={len(response.evidence)}, time={response.processing_time_ms:.0f}ms\n")

        return response

    async def _llm_route_query(self, query: MemoryQuery, session_context: str) -> MemoryQuery:
        """Use fine-tuned LLM (Stage 2) for structured routing when keyword analysis is uncertain."""
        if self.llm.model is None:
            return query

        # Only use LLM routing if keyword confidence is low
        if query.complexity < 0.3 or query.complexity > 0.7:
            return query  # High/low confidence — keyword routing is sufficient

        try:
            routing = self.llm.route_query(query.raw_query, session_context)
            heuristic_intent = query.intent

            # Map LLM intent to our enum
            intent_map = {
                "temporal": QueryIntent.TEMPORAL,
                "causal": QueryIntent.CAUSAL,
                "reflective": QueryIntent.REFLECTIVE,
                "factual": QueryIntent.FACTUAL,
                "procedural": QueryIntent.PROCEDURAL,
                "comparative": QueryIntent.COMPARATIVE,
                "exploratory": QueryIntent.EXPLORATORY,
            }
            llm_intent = routing.get("intent", "").lower()
            if llm_intent in intent_map:
                candidate_intent = intent_map[llm_intent]

                # Guardrail: keep explicit "why/causal" queries on the causal path
                # when heuristic routing already identified causality.
                raw_query_lower = query.raw_query.lower()
                strong_causal_signal = any(
                    token in raw_query_lower
                    for token in ("why", "how come", "what caused", "because", "reason")
                )
                if (
                    heuristic_intent == QueryIntent.CAUSAL
                    and candidate_intent == QueryIntent.REFLECTIVE
                    and strong_causal_signal
                ):
                    candidate_intent = QueryIntent.CAUSAL

                query.intent = candidate_intent

            # Use LLM complexity if it disagrees significantly
            llm_complexity = float(routing.get("complexity", query.complexity))
            if abs(llm_complexity - query.complexity) > 0.2:
                query.complexity = (query.complexity + llm_complexity) / 2.0

            # Re-evaluate routing
            if query.complexity < 0.3:
                query.routing = RoutingStrategy.NO_RETRIEVAL
            elif query.complexity < 0.6:
                query.routing = RoutingStrategy.SINGLE_STEP
            else:
                query.routing = RoutingStrategy.MULTI_STEP

            print(f"  🎯 LLM routing: intent={query.intent.value}, complexity={query.complexity:.2f}")
        except Exception as e:
            print(f"  ⚠ LLM routing failed: {e}, using keyword routing")

        return query

    async def _handle_no_retrieval(self, query: MemoryQuery) -> OrchestratorResponse:
        """Handle simple queries that don't need memory retrieval."""
        print("  ⚡ Routing: NO_RETRIEVAL (simple query)")

        answer = self.llm.generate(
            f"""You are Cortex Lab, a personal AI memory and reasoning assistant.
If this is a personal question about the user and you don't have stored memories
about it, honestly say you don't have that information yet.
Never fabricate personal details.

User: {query.raw_query}

Assistant:""",
            max_tokens=1024, temperature=0.3
        )

        return OrchestratorResponse(
            answer=answer,
            thinking="Simple query — no memory retrieval needed.",
            agents_used=["direct"],
            confidence=0.8,
            reasoning_trace="Direct LLM answer (no retrieval needed)",
        )

    async def _handle_single_step(self, query: MemoryQuery) -> OrchestratorResponse:
        """Handle moderate queries with a single agent."""
        selected_agents = self._select_agents(query)
        agent_name = selected_agents[0] if selected_agents else self.intent_to_agent.get(query.intent, "planning")
        query_metadata = dict(getattr(query, "metadata", {}) or {})
        query_metadata["execution_mode"] = RuntimeExecutionMode.SINGLE_STEP.value
        query_metadata["selected_agents"] = [agent_name]
        query_metadata.setdefault("conflict_resolution", ConflictResolutionPath.ARBITRATION_FIRST.value)
        query.metadata = query_metadata

        agent = self.agents.get(agent_name, self.agents["planning"])

        print(f"  🔀 Routing: SINGLE_STEP → {agent_name} agent")

        agent_response = await agent.execute(query)

        thinking = (
            f"Intent: {query.intent.value} (complexity: {query.complexity:.2f})\n"
            f"Agent: {agent_name}\n"
            f"Evidence: {len(agent_response.evidence)} memories retrieved\n"
            f"Reasoning: {agent_response.reasoning_trace}"
        )

        return OrchestratorResponse(
            answer=agent_response.answer,
            thinking=thinking,
            evidence=agent_response.evidence,
            agents_used=[agent_name],
            confidence=agent_response.confidence,
            reasoning_trace=agent_response.reasoning_trace,
        )

    async def _handle_multi_step(self, query: MemoryQuery, trace: Optional[PipelineTrace] = None) -> OrchestratorResponse:
        """Handle complex queries with multiple agents + Chain-of-Retrieval."""
        primary_agent_name = self.intent_to_agent.get(query.intent, "planning")
        agent_names = self._select_agents(query, force_multi_step=True)
        if not agent_names:
            agent_names = [primary_agent_name]

        execution_plan = self._build_l1_execution_plan(query, agent_names, primary_agent_name)
        query_metadata = dict(getattr(query, "metadata", {}) or {})
        query_metadata["execution_mode"] = execution_plan.execution_mode.value
        query_metadata["selected_agents"] = list(agent_names)
        query_metadata["conflict_resolution"] = execution_plan.conflict_resolution_path.value
        query.metadata = query_metadata

        if execution_plan.requires_confirmation:
            if trace is not None:
                trace.coordinator_plan = self._build_coordinator_plan(
                    query,
                    agent_names,
                    primary_agent_name,
                    execution_plan=execution_plan,
                )

            return OrchestratorResponse(
                answer=(
                    "This multi-agent plan requires your confirmation before execution. "
                    "Confirm the proposed dispatch plan to continue."
                ),
                thinking="Plan-mode confirmation gate is active for this request.",
                evidence=[],
                agents_used=["plan_mode_confirmation_required"],
                confidence=0.2,
                reasoning_trace=(
                    f"Execution paused pending plan confirmation (plan_id={execution_plan.plan_id})."
                ),
            )

        print(f"  🔀 Routing: MULTI_STEP → {agent_names}")

        coordinator_plan = self._build_coordinator_plan(
            query,
            agent_names,
            primary_agent_name,
            execution_plan=execution_plan,
        )
        if trace is not None:
            trace.coordinator_plan = coordinator_plan

        parent_task_id = None
        spawn_records: List[Dict[str, Any]] = []
        sidechain_events: List[Dict[str, Any]] = []
        if self.runtime_task_manager is not None and trace is not None:
            parent_task_id = self._coordinator_task_id(trace.trace_id)
            trace.coordinator_task_id = parent_task_id
            try:
                self.runtime_task_manager.create_task(
                    task_id=parent_task_id,
                    metadata={
                        "trace_id": trace.trace_id,
                        "query": query.raw_query,
                        "plan": coordinator_plan,
                    },
                )
            except ValueError:
                pass
            self._safe_transition_task(parent_task_id, TaskState.RUNNING, note="coordinator started")

        # Execute agents in parallel with task-manager registration.
        running_entries: List[Dict[str, Any]] = []
        for index, name in enumerate(agent_names):
            agent = self.agents.get(name, self.agents["planning"])
            subagent_task_id = None

            if self.runtime_task_manager is not None and parent_task_id is not None and trace is not None:
                subagent_task_id = self._subagent_task_id(trace.trace_id, index, name)
                metadata = {
                    "trace_id": trace.trace_id,
                    "agent": name,
                    "role": "primary" if name == primary_agent_name else "support",
                    "query": query.raw_query,
                }
                try:
                    self.runtime_task_manager.create_task(
                        task_id=subagent_task_id,
                        parent_task_id=parent_task_id,
                        metadata=metadata,
                    )
                except ValueError:
                    pass
                self._safe_transition_task(subagent_task_id, TaskState.RUNNING, note="subagent dispatched")

                spawn_record = {
                    "parent_task_id": parent_task_id,
                    "task_id": subagent_task_id,
                    "agent": name,
                    "role": metadata["role"],
                    "spawned_at": datetime.now(timezone.utc).isoformat(),
                }
                spawn_records.append(spawn_record)
                sidechain_events.append(
                    {
                        "event": "subagent_spawned",
                        "agent": name,
                        "task_id": subagent_task_id,
                        "trace_id": trace.trace_id,
                        "timestamp": spawn_record["spawned_at"],
                    }
                )

            task = asyncio.create_task(agent.execute(query), name=f"subagent:{name}")
            if self.runtime_task_manager is not None and subagent_task_id is not None:
                self.runtime_task_manager.attach_asyncio_task(subagent_task_id, task)
            running_entries.append({"agent": name, "task_id": subagent_task_id, "task": task})

        agent_responses = await asyncio.gather(*[entry["task"] for entry in running_entries], return_exceptions=True)

        # Combine all evidence and answers
        combined_answers = []
        all_evidence = []
        all_traces = []
        successful_payloads: List[Dict[str, Any]] = []
        successful_responses = 0
        cancelled_responses = 0

        for entry, outcome in zip(running_entries, agent_responses):
            name = entry["agent"]
            task_id = entry["task_id"]
            ts = datetime.now(timezone.utc).isoformat()

            if isinstance(outcome, asyncio.CancelledError):
                cancelled_responses += 1
                if task_id is not None:
                    self._safe_transition_task(task_id, TaskState.CANCELLED, note="subagent cancelled")
                sidechain_events.append(
                    {
                        "event": "subagent_cancelled",
                        "agent": name,
                        "task_id": task_id,
                        "trace_id": trace.trace_id if trace else "",
                        "timestamp": ts,
                    }
                )
                continue
            if isinstance(outcome, Exception):
                if task_id is not None:
                    self._safe_transition_task(task_id, TaskState.FAILED, note=str(outcome)[:240])
                sidechain_events.append(
                    {
                        "event": "subagent_failed",
                        "agent": name,
                        "task_id": task_id,
                        "trace_id": trace.trace_id if trace else "",
                        "timestamp": ts,
                        "error": str(outcome),
                    }
                )
                continue

            successful_responses += 1
            resp = outcome
            if task_id is not None:
                self._safe_transition_task(task_id, TaskState.COMPLETED, note="subagent completed")
            successful_payloads.append({"agent": name, "response": resp})
            combined_answers.append(f"[{name.title()} Agent]: {resp.answer}")
            all_evidence.extend(resp.evidence)
            all_traces.append(f"{name}: {resp.reasoning_trace}")
            sidechain_events.append(
                {
                    "event": "subagent_completed",
                    "agent": name,
                    "task_id": task_id,
                    "trace_id": trace.trace_id if trace else "",
                    "timestamp": ts,
                    "confidence": round(resp.confidence, 3),
                    "answer_preview": resp.answer[:220],
                }
            )

        if trace is not None:
            trace.subagent_spawn_records.extend(spawn_records)
            trace.sidechain_transcript.extend(sidechain_events)

        if cancelled_responses == len(running_entries):
            if parent_task_id is not None:
                self._safe_transition_task(parent_task_id, TaskState.CANCELLED, note="all subagents cancelled")
            if parent_task_id is not None and trace is not None:
                trace.sidechain_transcript.append(
                    {
                        "event": "coordinator_cancelled",
                        "task_id": parent_task_id,
                        "trace_id": trace.trace_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            return OrchestratorResponse(
                answer="Multi-agent execution was cancelled before completion.",
                thinking="Coordinator cancelled all delegated subagent tasks.",
                evidence=[],
                agents_used=agent_names + ["coordinator_cancelled"],
                confidence=0.0,
                reasoning_trace="Multi-agent orchestration cancelled by runtime task manager.",
            )

        if successful_responses == 0:
            if parent_task_id is not None:
                self._safe_transition_task(parent_task_id, TaskState.FAILED, note="no successful subagent responses")
            return OrchestratorResponse(
                answer="I couldn't complete this request because delegated subagents failed.",
                thinking="Coordinator observed failures across all delegated subagents.",
                evidence=[],
                agents_used=agent_names + ["coordinator_failed"],
                confidence=0.1,
                reasoning_trace="Multi-agent orchestration failed: no successful subagent responses.",
            )

        detected_conflicts = self._detect_inter_agent_conflicts(successful_payloads)
        arbitration_invoked = False
        if detected_conflicts:
            execution_plan.metadata["detected_conflicts"] = detected_conflicts

        if (
            detected_conflicts
            and execution_plan.conflict_resolution_path == ConflictResolutionPath.ARBITRATION_FIRST
            and "arbitration" in self.agents
            and all(payload.get("agent") != "arbitration" for payload in successful_payloads)
        ):
            conflict_context = "Detected conflicts:\n" + json.dumps(detected_conflicts[:5], indent=2)
            arbitration_agent = self.agents["arbitration"]
            try:
                arbitration_response = await arbitration_agent.execute(query, context=conflict_context)
                arbitration_invoked = True
                successful_payloads.append({"agent": "arbitration", "response": arbitration_response})
                combined_answers.append(f"[Arbitration Agent]: {arbitration_response.answer}")
                all_evidence.extend(arbitration_response.evidence)
                all_traces.append(f"arbitration: {arbitration_response.reasoning_trace}")

                arbitration_event = {
                    "event": "arbitration_invoked",
                    "agent": "arbitration",
                    "trace_id": trace.trace_id if trace else "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "conflict_count": len(detected_conflicts),
                    "resolution_confidence": round(arbitration_response.confidence, 3),
                }
                sidechain_events.append(arbitration_event)
                if trace is not None:
                    trace.sidechain_transcript.append(arbitration_event)
            except Exception as exc:
                arbitration_error_event = {
                    "event": "arbitration_failed",
                    "agent": "arbitration",
                    "trace_id": trace.trace_id if trace else "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                }
                sidechain_events.append(arbitration_error_event)
                if trace is not None:
                    trace.sidechain_transcript.append(arbitration_error_event)

        execution_plan.metadata["arbitration_invoked"] = arbitration_invoked
        execution_plan.metadata["detected_conflict_count"] = len(detected_conflicts)
        if trace is not None:
            trace.coordinator_plan = self._build_coordinator_plan(
                query,
                agent_names,
                primary_agent_name,
                execution_plan=execution_plan,
            )

        # Deduplicate evidence
        seen = set()
        unique_evidence = []
        for e in all_evidence:
            if e.memory.id not in seen:
                seen.add(e.memory.id)
                unique_evidence.append(e)

        # Use faithful generation (Stage 1) for synthesis with citations
        evidence_texts = [e.memory.content[:1500] for e in unique_evidence[:8]]
        if evidence_texts:
            final_answer = self.llm.generate_faithful(
                query.raw_query, evidence_texts,
                session_context="\n".join(combined_answers[:3])
            )
        else:
            # Fallback synthesis
            synthesis_prompt = f"""You are Cortex Lab, synthesizing multi-agent analysis of the user's memories.
Be concise but thorough. If no relevant memories exist, say so honestly.

User: {query.raw_query}

Agent Analyses:
{chr(10).join(combined_answers)}

Synthesized answer:"""
            final_answer = self.llm.generate(synthesis_prompt, max_tokens=1024, temperature=0.3)

        successful_confidence_values = [
            payload["response"].confidence
            for payload in successful_payloads
            if isinstance(payload.get("response"), AgentResponse)
        ]
        if successful_confidence_values:
            avg_confidence = sum(successful_confidence_values) / len(successful_confidence_values)
        else:
            # Guard against contract drift in mocked/non-standard agent payloads.
            avg_confidence = min(0.4 + 0.1 * successful_responses, 0.75)

        thinking = (
            f"Intent: {query.intent.value} (complexity: {query.complexity:.2f})\n"
            f"Agents: {', '.join(agent_names)}\n"
            f"Total evidence: {len(unique_evidence)} unique memories\n"
            f"Traces:\n" + "\n".join(f"  - {t}" for t in all_traces)
        )

        if parent_task_id is not None:
            self._safe_transition_task(parent_task_id, TaskState.COMPLETED, note="multi-agent synthesis completed")

        agents_used = list(agent_names)
        if arbitration_invoked and "arbitration" not in agents_used:
            agents_used.append("arbitration")

        return OrchestratorResponse(
            answer=final_answer,
            thinking=thinking,
            evidence=unique_evidence[:10],
            agents_used=agents_used,
            confidence=avg_confidence,
            reasoning_trace=f"Multi-agent synthesis from {len(agents_used)} agents",
        )

    async def _crag_evaluate(self, query: MemoryQuery, response: OrchestratorResponse) -> OrchestratorResponse:
        """
        CRAG (Corrective RAG): Multi-signal retrieval quality evaluation.
        → CORRECT: Use as is
        → AMBIGUOUS: Supplement with more retrieval
        → INCORRECT: Refine and caveat
        Populates pipeline trace with CRAG evaluation details.
        """
        if not response.evidence:
            # No evidence — populate trace and return
            if response.pipeline_trace:
                response.pipeline_trace.crag_evaluation = CRAGEvaluation(
                    quality_score=0.0, verdict="NO_EVIDENCE", evidence_count=0,
                )
            return response

        # Multi-signal quality evaluation
        avg_score = sum(r.score for r in response.evidence) / len(response.evidence)
        max_score = max(r.score for r in response.evidence)
        evidence_count = len(response.evidence)

        # Entity coverage check
        entity_coverage = 0.0
        if query.entities:
            matched = 0
            for ent in query.entities:
                for r in response.evidence:
                    if ent.lower() in r.memory.content.lower():
                        matched += 1
                        break
            entity_coverage = matched / len(query.entities)

        # Combined quality score
        quality_score = (
            0.40 * avg_score +
            0.20 * max_score +
            0.20 * min(evidence_count / 5.0, 1.0) +
            0.20 * entity_coverage
        )

        supplementary_count = 0

        if quality_score > 0.55:
            verdict = "CORRECT"
            response.reasoning_trace += f" | CRAG: CORRECT (q={quality_score:.2f})"
        elif quality_score > 0.30:
            verdict = "AMBIGUOUS"
            response.reasoning_trace += f" | CRAG: AMBIGUOUS (q={quality_score:.2f})"
            response.confidence *= 0.85

            # Supplementary retrieval with step-back query
            if query.step_back_query:
                try:
                    sb_query = MemoryQuery(
                        raw_query=query.step_back_query,
                        intent=query.intent,
                        complexity=0.4,
                        embedding=self.retriever.embeddings.embed(query.step_back_query).tolist(),
                    )
                    extra_results = await self.retriever.retrieve(sb_query, top_k=5)
                    existing_ids = {r.memory.id for r in response.evidence}
                    added = 0
                    for r in extra_results:
                        if r.memory.id not in existing_ids:
                            response.evidence.append(r)
                            existing_ids.add(r.memory.id)
                            added += 1
                    supplementary_count = added
                    response.reasoning_trace += f" → +{added} from step-back"
                except Exception:
                    pass
        else:
            verdict = "INCORRECT"
            response.reasoning_trace += f" | CRAG: INCORRECT (q={quality_score:.2f})"
            response.confidence *= 0.55
            # Don't prefix with warning — let the response speak for itself

        # Populate trace
        if response.pipeline_trace:
            response.pipeline_trace.crag_evaluation = CRAGEvaluation(
                quality_score=quality_score,
                verdict=verdict,
                avg_evidence_score=avg_score,
                max_evidence_score=max_score,
                evidence_count=evidence_count,
                entity_coverage=entity_coverage,
                supplementary_retrieved=supplementary_count,
            )

        return response

    async def _self_rag_critique(self, query: MemoryQuery,
                                  response: OrchestratorResponse) -> OrchestratorResponse:
        """
        Self-RAG with ISREL/ISSUP/ISUSE critique tokens (Stage 4 fine-tuning).
        Generate → Critique → Revise loop (max 2 iterations).
        Populates pipeline trace with critique details.
        """
        if self.llm.model is None:
            return response

        evidence_texts = [r.memory.content[:1000] for r in response.evidence[:5]]
        if not evidence_texts:
            return response

        critique_trace = SelfRAGCritique()

        try:
            # Use fine-tuned ISREL/ISSUP/ISUSE critique
            critique = self.llm.self_rag_critique(
                query.raw_query, response.answer, evidence_texts
            )

            isrel = critique.get("ISREL", 5)
            issup = critique.get("ISSUP", 5)
            isuse = critique.get("ISUSE", 5)
            avg = critique.get("avg_score", 5.0)
            verdict = critique.get("verdict", "REVISE")

            critique_trace.isrel = isrel
            critique_trace.issup = issup
            critique_trace.isuse = isuse
            critique_trace.avg_score = avg
            critique_trace.verdict = verdict

            response.reasoning_trace += f" | Self-RAG: R={isrel}/S={issup}/U={isuse} ({verdict})"

            if verdict == "ACCEPT" or avg >= 7.0:
                response.confidence = min(response.confidence + 0.1, 0.95)
            elif avg >= 5.0:
                # Identify weakest area and revise
                weak = "relevance" if isrel <= issup and isrel <= isuse else (
                    "faithfulness" if issup <= isuse else "completeness"
                )
                critique_trace.revision_focus = weak

                revision_prompt = f"""Revise this answer to improve {weak}. Be grounded in the evidence. Provide a comprehensive answer.

Question: {query.raw_query}
Original answer: {response.answer[:800]}
Evidence: {chr(10).join(f"[{i+1}] {e}" for i, e in enumerate(evidence_texts[:5]))}

Improved answer (focus on {weak}):"""
                revised = self.llm.generate(revision_prompt, max_tokens=1024, temperature=0.3)
                if len(revised.strip()) > 20:
                    response.answer = revised.strip()
                    response.reasoning_trace += f" → revised ({weak})"
                    response.confidence = min(response.confidence + 0.05, 0.85)
                    critique_trace.revision_applied = True
            else:
                response.confidence = max(response.confidence - 0.15, 0.25)
                response.reasoning_trace += " (low quality)"

        except Exception as e:
            response.reasoning_trace += f" | Self-RAG error: {str(e)[:50]}"
            critique_trace.verdict = f"ERROR: {str(e)[:50]}"

        # Populate trace
        if response.pipeline_trace:
            response.pipeline_trace.self_rag_critique = critique_trace

        return response

    async def _flare_active_retrieval(self, query: MemoryQuery,
                                       response: OrchestratorResponse) -> OrchestratorResponse:
        """
        FLARE: Forward-Looking Active Retrieval (EMNLP 2023).
        Identifies low-confidence segments in the answer and retrieves
        additional evidence to fill gaps.
        Populates pipeline trace with FLARE details.
        """
        if self.llm.model is None or not response.answer:
            return response

        flare = FLARETrace(triggered=True)
        pre_confidence = response.confidence

        try:
            # Split answer into sentences
            sentences = re.split(r'(?<=[.!?])\s+', response.answer)
            if len(sentences) < 2:
                flare.triggered = False
                if response.pipeline_trace:
                    response.pipeline_trace.flare_trace = flare
                return response

            # Identify sentences that might need more evidence
            uncertain_markers = [
                "might", "possibly", "perhaps", "unclear", "not sure",
                "limited", "insufficient", "partial", "?", "may have",
            ]

            sentences_to_verify = []
            for i, sent in enumerate(sentences):
                if any(marker in sent.lower() for marker in uncertain_markers):
                    sentences_to_verify.append((i, sent))

            flare.uncertain_sentences = len(sentences_to_verify)

            if not sentences_to_verify:
                flare.triggered = False
                if response.pipeline_trace:
                    response.pipeline_trace.flare_trace = flare
                return response

            # Retrieve additional evidence for uncertain segments
            additional_evidence = []
            for idx, sent in sentences_to_verify[:2]:  # Max 2 FLARE retrievals
                flare_query = MemoryQuery(
                    raw_query=sent,
                    intent=query.intent,
                    complexity=0.4,
                    embedding=self.retriever.embeddings.embed(sent).tolist(),
                )
                results = await self.retriever.retrieve(flare_query, top_k=3)
                additional_evidence.extend(results)
                flare.retrieval_iterations += 1

            if additional_evidence:
                # Deduplicate
                existing_ids = {r.memory.id for r in response.evidence}
                new_evidence = [r for r in additional_evidence if r.memory.id not in existing_ids]

                if new_evidence:
                    response.evidence.extend(new_evidence[:3])
                    flare.new_evidence_count = len(new_evidence[:3])

                    # Re-generate with augmented evidence
                    all_evidence_texts = (
                        [r.memory.content[:200] for r in response.evidence[:5]]
                    )
                    revised = self.llm.generate_faithful(
                        query.raw_query, all_evidence_texts
                    )
                    if len(revised.strip()) > 20:
                        response.answer = revised.strip()
                        response.confidence = min(response.confidence + 0.1, 0.85)
                        response.reasoning_trace += f" | FLARE: +{len(new_evidence)} evidence"
                        flare.answer_revised = True

        except Exception as e:
            response.reasoning_trace += f" | FLARE error: {str(e)[:50]}"

        flare.confidence_delta = response.confidence - pre_confidence

        # Populate trace
        if response.pipeline_trace:
            response.pipeline_trace.flare_trace = flare

        return response

    async def _try_function_calling(
        self,
        query: MemoryQuery,
        trace: PipelineTrace,
        runtime_loop: Optional[RuntimeLoopState] = None,
        now: Optional[datetime] = None,
    ) -> Optional[OrchestratorResponse]:
        """
        Stage 13 (Function Calling): Check if query should be handled by a tool.
        Returns OrchestratorResponse if a tool was successfully executed, else None.
        
        Tool execution:
        - search_memories → HybridRetriever.retrieve()
        - search_by_time → MetadataStore.search_by_time()
        - find_entity → KnowledgeGraph.find_entity_by_name()
        - trace_causal_chain → CausalAgent
        - detect_belief_evolution → LLM detect_belief_change
        - summarize_topic → LLM summarize
        - delete_memory → metadata + vector deletion (approval-gated)
        """
        try:
            # Ask the LLM which tool to use
            fc_result = self.llm.call_function(query.raw_query, AVAILABLE_TOOLS)
            tool_name = fc_result.get("tool_name", "none")
            arguments = fc_result.get("arguments", {})

            if tool_name == "none" or not tool_name:
                return None

            print(f"  🔧 Function call: {tool_name}({arguments})")

            if runtime_loop is not None:
                dispatch_ts = now or datetime.now(timezone.utc)
                allowed = runtime_loop.try_register_tool_dispatch(now=dispatch_ts)
                if not allowed:
                    trace.add_step(PipelineStep(
                        step_name=f"Tool Dispatch Rate Limit ({tool_name})",
                        step_type="tool_rate_limit",
                        status="error",
                        duration_ms=0,
                        details={
                            "decision": "rate_limited",
                            "window_seconds": runtime_loop.envelope.budget.window_seconds,
                            "max_tool_calls_per_window": runtime_loop.envelope.budget.max_tool_calls_per_window,
                            "tool_calls_executed": runtime_loop.tool_calls_executed,
                            "reason": runtime_loop.stop_note,
                        },
                    ))
                    return OrchestratorResponse(
                        answer=(
                            "Tool dispatch rate limit reached for this query window. "
                            "Please retry after the current window resets."
                        ),
                        thinking="Function call blocked by deterministic tool dispatch window guardrail.",
                        agents_used=["function_calling", "tool_rate_limited"],
                        confidence=0.2,
                        reasoning_trace=f"Function call rate-limited: {tool_name}",
                    )

            safety_block = self._authorize_tool_execution(query, trace, tool_name, arguments)
            if safety_block is not None:
                return safety_block

            # Execute the tool
            if tool_name == "search_memories":
                search_query = arguments.get("query", query.raw_query)
                top_k = int(arguments.get("top_k", 10))
                search_mq = MemoryQuery(
                    raw_query=search_query,
                    intent=query.intent,
                    complexity=0.4,
                    embedding=self.retriever.embeddings.embed(search_query).tolist(),
                )
                results = await self.retriever.retrieve(search_mq, top_k=top_k)
                if results:
                    evidence_texts = [r.memory.content[:1500] for r in results[:5]]
                    answer = self.llm.generate_faithful(query.raw_query, evidence_texts)
                    return OrchestratorResponse(
                        answer=answer, evidence=results[:10],
                        agents_used=["function_calling"],
                        confidence=min(0.5 + len(results) * 0.05, 0.90),
                        reasoning_trace=f"Function call: search_memories('{search_query[:50]}') → {len(results)} results",
                    )

            elif tool_name == "find_entity":
                entity_name = arguments.get("entity_name", "")
                if entity_name:
                    entity_id = self.retriever.graph.find_entity_by_name(entity_name)
                    if entity_id and self.retriever.graph.graph:
                        # Get all memories linked to this entity
                        node_data = self.retriever.graph.graph.nodes.get(entity_id, {})
                        memory_ids = node_data.get("memory_ids", [])
                        memories = []
                        for mid in memory_ids[:10]:
                            mem = self.retriever.metadata.get_memory(mid)
                            if mem:
                                memories.append(mem.content[:300])
                        if memories:
                            answer = self.llm.generate_faithful(query.raw_query, memories)
                            return OrchestratorResponse(
                                answer=answer, agents_used=["function_calling"],
                                confidence=0.75,
                                reasoning_trace=f"Function call: find_entity('{entity_name}') → {len(memories)} memories",
                            )

            elif tool_name == "summarize_topic":
                topic = arguments.get("topic", query.raw_query)
                summary_mq = MemoryQuery(
                    raw_query=topic,
                    intent=QueryIntent.EXPLORATORY,
                    complexity=0.4,
                    embedding=self.retriever.embeddings.embed(topic).tolist(),
                )
                results = await self.retriever.retrieve(summary_mq, top_k=15)
                if results:
                    texts = [r.memory.content[:300] for r in results[:10]]
                    combined = "\n".join(texts)
                    summary = self.llm.summarize(combined, max_length=200)
                    if summary and len(summary.strip()) > 20:
                        return OrchestratorResponse(
                            answer=summary, evidence=results[:5],
                            agents_used=["function_calling"],
                            confidence=0.70,
                            reasoning_trace=f"Function call: summarize_topic('{topic[:50]}') → summary from {len(results)} memories",
                        )

            elif tool_name == "delete_memory":
                memory_id = str(arguments.get("memory_id", "")).strip()
                if memory_id:
                    self.retriever.metadata.delete_memory(memory_id)
                    self.retriever.vectors.delete(memory_id)
                    return OrchestratorResponse(
                        answer=f"Deleted memory {memory_id}.",
                        agents_used=["function_calling"],
                        confidence=0.7,
                        reasoning_trace=f"Function call: delete_memory('{memory_id}') → deleted",
                    )

            # Tool not handled or didn't produce results → fall back to normal pipeline
            return None

        except Exception as e:
            print(f"  ⚠ Function calling failed: {e}")
            return None

    def process_sync(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """Synchronous wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.process(raw_query, session_context))
        finally:
            loop.close()
