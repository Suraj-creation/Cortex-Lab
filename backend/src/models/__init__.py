"""
Core Data Models for Cortex Lab Agentic RAG System
Defines memory objects, queries, agent responses, and all shared types.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid
import json


# ─── Enums ───────────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    EPISODIC = "episodic"       # "Had coffee with Sarah"
    SEMANTIC = "semantic"       # "Transformers use self-attention"
    PROCEDURAL = "procedural"   # "My code review process"
    REFLECTIVE = "reflective"   # "I realized I avoid difficult conversations"


class EmotionLabel(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CONFUSED = "confused"
    HOPEFUL = "hopeful"
    FRUSTRATED = "frustrated"


class QueryIntent(str, Enum):
    TEMPORAL = "temporal"       # "When did X happen?"
    CAUSAL = "causal"           # "Why did I do X?"
    REFLECTIVE = "reflective"   # "How did my thinking change?"
    FACTUAL = "factual"         # "What is X?"
    PROCEDURAL = "procedural"   # "How do I do X?"
    COMPARATIVE = "comparative" # "Compare X and Y"
    EXPLORATORY = "exploratory" # "Tell me about X"


class RoutingStrategy(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"


class RetrievalQuality(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class BeliefChangeType(str, Enum):
    CONTRADICTION = "contradiction"
    REFINEMENT = "refinement"
    REINFORCEMENT = "reinforcement"
    NEW_BELIEF = "new_belief"


# ─── Core Data Models ────────────────────────────────────────────────────────

@dataclass
class CausalMemoryObject:
    """
    A rich memory event with full metadata.
    Central data model for all memory operations.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    timestamp: datetime = field(default_factory=datetime.now)

    # Classification
    emotion: EmotionLabel = EmotionLabel.NEUTRAL
    emotion_confidence: float = 0.0
    importance: float = 0.5  # 0.0 - 1.0
    topics: List[str] = field(default_factory=list)

    # Entity extraction
    entities: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)

    # Causal links
    causes: List[str] = field(default_factory=list)  # IDs of memories that caused this
    effects: List[str] = field(default_factory=list)  # IDs of memories caused by this
    causal_description: str = ""

    # Embeddings
    embedding: Optional[List[float]] = None
    context_prefix: str = ""  # Contextual chunking prefix

    # Propositions (atomic facts)
    propositions: List[str] = field(default_factory=list)

    # RAPTOR
    raptor_level: int = 0  # 0=raw, 1=daily, 2=weekly, 3=monthly, 4=yearly
    raptor_children: List[str] = field(default_factory=list)

    # Metadata
    session_id: str = ""
    source: str = "chat"  # chat, import, voice
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "emotion": self.emotion.value,
            "emotion_confidence": self.emotion_confidence,
            "importance": self.importance,
            "topics": self.topics,
            "entities": self.entities,
            "entity_ids": self.entity_ids,
            "causes": self.causes,
            "effects": self.effects,
            "causal_description": self.causal_description,
            "context_prefix": self.context_prefix,
            "propositions": self.propositions,
            "raptor_level": self.raptor_level,
            "raptor_children": self.raptor_children,
            "session_id": self.session_id,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalMemoryObject":
        data = dict(data)
        if "memory_type" in data and isinstance(data["memory_type"], str):
            data["memory_type"] = MemoryType(data["memory_type"])
        if "emotion" in data and isinstance(data["emotion"], str):
            data["emotion"] = EmotionLabel(data["emotion"])
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        # Remove embedding from dict (stored separately)
        data.pop("embedding", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryQuery:
    """Query object with analysis metadata."""
    raw_query: str = ""
    intent: QueryIntent = QueryIntent.FACTUAL
    complexity: float = 0.5  # 0.0 - 1.0
    routing: RoutingStrategy = RoutingStrategy.SINGLE_STEP

    # Temporal
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Extracted info
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # Transformed queries
    multi_queries: List[str] = field(default_factory=list)
    hyde_answer: str = ""
    step_back_query: str = ""
    sub_queries: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    confidence: float = 0.0


@dataclass
class RetrievalResult:
    """A single retrieval result with score and provenance."""
    memory: CausalMemoryObject
    score: float = 0.0
    channel: str = ""  # dense, sparse, graph, temporal, proposition
    evidence_text: str = ""


@dataclass
class AgentResponse:
    """Response from a specialized agent."""
    agent_name: str = ""
    answer: str = ""
    evidence: List[RetrievalResult] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: str = ""
    sub_queries_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    retrieval_quality: RetrievalQuality = RetrievalQuality.CORRECT


@dataclass
class OrchestratorResponse:
    """Final response from the orchestrator combining all agent outputs."""
    answer: str = ""
    thinking: str = ""
    evidence: List[RetrievalResult] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: str = ""
    query_analysis: Optional[MemoryQuery] = None
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    token_usage: Dict[str, int] = field(default_factory=dict)
    pipeline_trace: Optional["PipelineTrace"] = None


@dataclass
class BeliefDelta:
    """Tracks a change in user belief over time."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    old_belief_id: str = ""
    new_belief_id: str = ""
    old_belief_text: str = ""
    new_belief_text: str = ""
    change_type: BeliefChangeType = BeliefChangeType.NEW_BELIEF
    confidence: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)
    evidence_chain: List[str] = field(default_factory=list)


@dataclass
class EntityNode:
    """An entity in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    canonical_name: str = ""
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "unknown"  # person, place, project, concept, etc.
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    memory_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A relationship between two entities."""
    source_id: str = ""
    target_id: str = ""
    relation: str = ""  # works_with, caused, discussed, etc.
    weight: float = 1.0
    memory_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Multi-level cache entry."""
    key: str = ""
    value: Any = None
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: int = 3600


# ─── Pipeline Observability Models ──────────────────────────────────────────

@dataclass
class PipelineStep:
    """A single step in the RAG pipeline with timing and metadata."""
    step_name: str = ""
    step_type: str = ""  # query_analysis, query_transform, retrieval, crag, self_rag, flare, generation, routing
    status: str = "completed"  # pending, running, completed, skipped, error
    duration_ms: float = 0.0
    start_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    sub_steps: List["PipelineStep"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "step_type": self.step_type,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "details": self.details,
            "sub_steps": [s.to_dict() for s in self.sub_steps],
        }


@dataclass
class RetrievalChannelTrace:
    """Trace of a single retrieval channel's results."""
    channel: str = ""  # dense, sparse, graph, temporal, proposition
    result_count: int = 0
    top_score: float = 0.0
    avg_score: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "result_count": self.result_count,
            "top_score": round(self.top_score, 4),
            "avg_score": round(self.avg_score, 4),
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class CRAGEvaluation:
    """CRAG quality evaluation results."""
    quality_score: float = 0.0
    verdict: str = ""  # CORRECT, AMBIGUOUS, INCORRECT
    avg_evidence_score: float = 0.0
    max_evidence_score: float = 0.0
    evidence_count: int = 0
    entity_coverage: float = 0.0
    supplementary_retrieved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": round(self.quality_score, 3),
            "verdict": self.verdict,
            "avg_evidence_score": round(self.avg_evidence_score, 3),
            "max_evidence_score": round(self.max_evidence_score, 3),
            "evidence_count": self.evidence_count,
            "entity_coverage": round(self.entity_coverage, 3),
            "supplementary_retrieved": self.supplementary_retrieved,
        }


@dataclass
class SelfRAGCritique:
    """Self-RAG critique token evaluation."""
    isrel: int = 0  # Relevance score 1-10
    issup: int = 0  # Support/faithfulness score 1-10
    isuse: int = 0  # Usefulness score 1-10
    avg_score: float = 0.0
    verdict: str = ""  # ACCEPT, REVISE, REJECT
    revision_applied: bool = False
    revision_focus: str = ""  # relevance, faithfulness, completeness

    def to_dict(self) -> Dict[str, Any]:
        return {
            "isrel": self.isrel,
            "issup": self.issup,
            "isuse": self.isuse,
            "avg_score": round(self.avg_score, 1),
            "verdict": self.verdict,
            "revision_applied": self.revision_applied,
            "revision_focus": self.revision_focus,
        }


@dataclass
class FLARETrace:
    """FLARE active retrieval trace."""
    triggered: bool = False
    uncertain_sentences: int = 0
    retrieval_iterations: int = 0
    new_evidence_count: int = 0
    answer_revised: bool = False
    confidence_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "uncertain_sentences": self.uncertain_sentences,
            "retrieval_iterations": self.retrieval_iterations,
            "new_evidence_count": self.new_evidence_count,
            "answer_revised": self.answer_revised,
            "confidence_delta": round(self.confidence_delta, 3),
        }


@dataclass
class QueryTransformTrace:
    """Trace of query transformations applied."""
    original_query: str = ""
    multi_queries: List[str] = field(default_factory=list)
    hyde_answer: str = ""
    step_back_query: str = ""
    sub_queries: List[str] = field(default_factory=list)
    total_variants: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "multi_queries": self.multi_queries,
            "hyde_answer": self.hyde_answer[:200] if self.hyde_answer else "",
            "step_back_query": self.step_back_query,
            "sub_queries": self.sub_queries,
            "total_variants": self.total_variants,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class PipelineTrace:
    """Complete pipeline observability trace for a single RAG request."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    query: str = ""
    total_duration_ms: float = 0.0

    # Pipeline steps (ordered)
    steps: List[PipelineStep] = field(default_factory=list)

    # Detailed traces
    query_analysis: Dict[str, Any] = field(default_factory=dict)
    query_transform: Optional[QueryTransformTrace] = None
    retrieval_channels: List[RetrievalChannelTrace] = field(default_factory=list)
    reranking: Dict[str, Any] = field(default_factory=dict)
    crag_evaluation: Optional[CRAGEvaluation] = None
    self_rag_critique: Optional[SelfRAGCritique] = None
    flare_trace: Optional[FLARETrace] = None

    # Agent routing
    routing_decision: str = ""  # no_retrieval, single_step, multi_step
    agents_invoked: List[Dict[str, Any]] = field(default_factory=list)

    # Generation
    generation_details: Dict[str, Any] = field(default_factory=dict)

    # Cache
    cache_status: Dict[str, Any] = field(default_factory=dict)

    # Final metrics
    final_confidence: float = 0.0
    evidence_count: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "steps": [s.to_dict() for s in self.steps],
            "query_analysis": self.query_analysis,
            "query_transform": self.query_transform.to_dict() if self.query_transform else None,
            "retrieval_channels": [c.to_dict() for c in self.retrieval_channels],
            "reranking": self.reranking,
            "crag_evaluation": self.crag_evaluation.to_dict() if self.crag_evaluation else None,
            "self_rag_critique": self.self_rag_critique.to_dict() if self.self_rag_critique else None,
            "flare_trace": self.flare_trace.to_dict() if self.flare_trace else None,
            "routing_decision": self.routing_decision,
            "agents_invoked": self.agents_invoked,
            "generation_details": self.generation_details,
            "cache_status": self.cache_status,
            "final_confidence": round(self.final_confidence, 3),
            "evidence_count": self.evidence_count,
            "token_usage": self.token_usage,
        }
