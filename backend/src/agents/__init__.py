"""
Agents layer — combines legacy orchestrator with new autonomous runtime.

Legacy (preserved): AgentOrchestrator, specialized agents
New (pi-mono pattern): CortexAgentLoop, AgentConfig, Extensions, Tools
"""

# Legacy — still used by existing /api/rag/chat endpoint
from src.agents.orchestrator import AgentOrchestrator
from src.agents.specialized import (
    AcademicIntelligenceAgent,
    ArbitrationAgent,
    BaseAgent,
    BehavioralHabitsAgent,
    build_specialized_agents,
    CausalAgent,
    CognitivePatternsAgent,
    DecisionLogAgent,
    DomainSpecializedAgent,
    EmotionalIntelligenceAgent,
    GoalVisionAgent,
    MetaLearningAgent,
    PersonalJournalingAgent,
    PersonalWellbeingAgent,
    PlanningAgent,
    ReflectionAgent,
    SocialIntelligenceAgent,
    TimelineAgent,
)

# New autonomous runtime (pi-mono pattern)
from src.agents.autonomous_loop import CortexAgentLoop, AgentConfig, SteeringManager
from src.agents.session_persistence import SessionPersistence
from src.agents.extension_runner import ExtensionRunner, Extension
from src.agents.tool_types import ToolDefinition, ToolResult, CortexEvent, CortexEventType
from src.agents.tier_router import TierRouter
from src.agents.scheduler import BackgroundScheduler, background_scheduler

__all__ = [
    # Legacy
    "AgentOrchestrator",
    "BaseAgent",
    "DomainSpecializedAgent",
    "TimelineAgent",
    "CausalAgent",
    "ReflectionAgent",
    "PlanningAgent",
    "ArbitrationAgent",
    "AcademicIntelligenceAgent",
    "PersonalJournalingAgent",
    "PersonalWellbeingAgent",
    "CognitivePatternsAgent",
    "DecisionLogAgent",
    "EmotionalIntelligenceAgent",
    "BehavioralHabitsAgent",
    "SocialIntelligenceAgent",
    "GoalVisionAgent",
    "MetaLearningAgent",
    "build_specialized_agents",
    # New autonomous runtime
    "CortexAgentLoop",
    "AgentConfig",
    "SteeringManager",
    "SessionPersistence",
    "ExtensionRunner",
    "Extension",
    "ToolDefinition",
    "ToolResult",
    "CortexEvent",
    "CortexEventType",
    "TierRouter",
    "BackgroundScheduler",
    "background_scheduler",
]
