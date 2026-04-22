# Agent 2.0: Autonomous Multi-Agent Runtime Blueprint

## 1) Document Status

Status: Proposed architecture and development specification (next-generation design).

Purpose:
1. Extend the current 6-role runtime into a deeper autonomous system.
2. Introduce a Master-Orchestrater control plane for always-on, health-aware operation.
3. Define a chain-of-agents and agent-to-agent communication protocol.
4. Define 15 specialized agents with explicit system-prompt blueprints.

Relation to current system:
1. Current runtime baseline is documented in Agent.md.
2. This document defines planned Agent 2.0 behavior for further implementation.
3. It avoids over-claiming and treats unimplemented items as target-state design.

---

## 2) Core Intent

Primary intent:
1. Build a high-trust, always-available personal intelligence runtime that listens responsibly, stores selectively, reasons deeply, and retrieves efficiently.

Secondary intent:
1. Convert raw conversation streams into structured memory assets with strong metadata and multi-agent tags.
2. Move from single-pass responses to coordinated reasoning and reflection loops.

Operating principle:
1. Data without structure is noise.
2. Structure without selection creates memory pollution.
3. Selection without context hurts retrieval.
4. Agent 2.0 balances all three: selection + structure + context.

---

## 3) Target Role Model

Total target role model:
1. 1 Master-Orchestrater (L0 control plane)
2. 1 Runtime Orchestrator (L1 reasoning plane)
3. 15 specialized agents (L2 domain agents)

Notes:
1. The Runtime Orchestrator remains the central reasoning coordinator.
2. The Master-Orchestrater is a higher-level lifecycle and autonomy controller.
3. The 15 specialized agents are domain experts for storage, retrieval, and reasoning.

---

## 4) Layered Architecture

## 4.1 L0: Master-Orchestrater (Autonomy and Device Control)

Responsibilities:
1. Keep the app alive, aware, and resource-safe.
2. Decide when to wake or sleep listening and model services.
3. Start and end sessions.
4. Perform speaker-aware capture gating.
5. Classify conversation relevance for memory retention.
6. Enforce health, battery, network, and thermal policies.

## 4.2 L1: Runtime Orchestrator (Reasoning and Coordination)

Responsibilities:
1. Route user requests by intent and complexity.
2. Select execution mode (no retrieval, single-agent, multi-agent chain).
3. Coordinate specialist execution and synthesis.
4. Apply quality loops (CRAG, Self-RAG, FLARE).
5. Produce final text or speech-ready responses.

## 4.3 L2: Specialized Agent Layer

Responsibilities:
1. Provide domain reasoning and domain-specific memory shaping.
2. Write and read with explicit metadata expectations.
3. Return structured outputs with confidence and evidence links.

## 4.4 Shared Services

Services:
1. STT, TTS, Speaker-ID, VAD
2. Memory ingestion and chunking pipeline
3. Metadata store and retrieval index
4. Policy and approval runtime
5. Observability, trace, and diagnostics

---

## 5) Master-Orchestrater Detailed Specification

## 5.1 Core Function

The Master-Orchestrater is the autonomy governor for the entire app lifecycle.

It decides:
1. When the system should listen.
2. When the system should process.
3. When the system should store.
4. When the system should sleep or mute.

## 5.2 Device and Service Awareness

The Master-Orchestrater continuously checks:
1. Battery level and charging state
2. Thermal pressure and CPU load
3. Memory usage and available RAM
4. Network quality and connectivity state
5. Foreground/background state
6. Audio route and microphone availability

Health sampling cadence:
1. Fast health loop: every 5-10 seconds for critical signals.
2. Deep health loop: every 30-60 seconds for trend signals.

## 5.3 Runtime State Machine

States:
1. `SLEEPING`: passive, minimal services alive.
2. `PASSIVE_MONITORING`: VAD and wake triggers active, no heavy processing.
3. `ACTIVE_LISTENING`: STT and speaker tracking active.
4. `ACTIVE_PROCESSING`: local model + ingestion + tagging active.
5. `DEGRADED_MODE`: reduced fidelity due to low resources.
6. `EMERGENCY_MUTE`: forced mute due to policy or critical health condition.

State transition triggers:
1. Wake word call (for example name trigger such as "Eva").
2. Verified user speaker presence.
3. Sustained conversational activity.
4. Battery/thermal thresholds.
5. Explicit user command (mute, sleep, resume).

## 5.4 Wake and Sleep Policy

Wake conditions (all must pass):
1. Conversation activity detected by VAD.
2. User speaker detected above identity confidence threshold.
3. Device health permits active mode.

Sleep conditions (any can trigger):
1. Silence timeout crossed.
2. User speaker absent for configured window.
3. Battery or thermal policy requires throttle.
4. User command requests mute/sleep.

## 5.5 STT/TTS/Local Model Activation Contract

Activation order:
1. Start VAD and lightweight speaker check.
2. Start STT only after wake conditions are met.
3. Start local model only when active processing is required.
4. Enable TTS only on response generation.

Deactivation order:
1. Stop TTS first.
2. Flush ingestion queue.
3. Stop local model workers.
4. Return STT to passive state.

## 5.6 Sessionization Contract

Session starts when:
1. Wake conditions are met.
2. User identity confidence is above threshold.
3. A new conversational segment is detected.

Session ends when:
1. Silence timeout + no user identity.
2. Explicit stop command.
3. Resource governor forces sleep.

Session metadata must include:
1. Session id
2. Start and end timestamp
3. User speaker confidence trend
4. Channel/mode info (text, voice, hybrid)
5. Health snapshot at start and end

## 5.7 Conversation Retention Decision

Not all captured conversation should be stored.

Master-Orchestrater retention scoring dimensions:
1. User relevance score
2. Semantic novelty score
3. Future utility score
4. Personal significance score
5. Safety and privacy policy score

Retention outcomes:
1. `discard`: do not store.
2. `session_context_only`: keep in short horizon context buffer.
3. `structured_memory`: ingest into long-term memory with full metadata.
4. `priority_memory`: ingest and pin for high retrieval priority.

## 5.8 Structured Ingestion Expectations

When storing, Master-Orchestrater must ensure:
1. speaker attribution is clear (`user`, `other_party`, `unknown`).
2. conversation is chunked by semantic boundaries, not just fixed length.
3. each chunk has labels, metadata, and agent tags.
4. chunk context windows preserve meaning and references.

## 5.9 Resource Governor Policy

Example policy bands:
1. Battery > 10 percent: full capability.
2. Battery < 10 percent: emergency mute except explicit wake command.

Network policy:
1. Online: hybrid mode allowed.
2. Offline: local-only fallback mode.
3. Unstable: queue async sync tasks, avoid expensive remote calls.

---

## 6) Runtime Orchestrator 2.0 Scope

The Runtime Orchestrator remains the reasoning coordinator under Master-Orchestrater supervision.

Core responsibilities:
1. Query analysis and routing.
2. Single-step or multi-step specialist dispatch.
3. Parallel subagent execution with cancellation support.
4. Evidence consolidation and answer synthesis.
5. Quality loops and confidence correction.
6. Tool-calling with safety checks and deterministic limits.

Execution modes:
1. `NO_RETRIEVAL`
2. `SINGLE_STEP`
3. `MULTI_STEP`

Quality loops:
1. CRAG retrieval quality assessment.
2. Self-RAG critique and optional revision.
3. FLARE for confidence-recovery retrieval.

---

## 7) Chain-of-Agents Design

## 7.1 Capture Chain (Always-on Voice Path)

1. VAD detects activity.
2. Speaker-ID checks if user is present.
3. Master-Orchestrater opens session.
4. STT transcribes turns with speaker separation.
5. Session pre-processor segments conversation by topic and intent shifts.
6. Relevance scorer decides retention mode.
7. Multi-label classifier tags chunk with one or more specialized agents.
8. Structured memory writer stores chunk and metadata.
9. Indexer updates retrieval indexes.

## 7.2 Query Chain (User asks by text or voice)

1. User trigger reaches Runtime Orchestrator.
2. Query analysis + transform.
3. Orchestrator route decision.
4. Specialist execution (single or parallel).
5. Evidence merge and dedup.
6. CRAG + Self-RAG + FLARE.
7. Final response formatting.
8. TTS (if voice response mode is active).

## 7.3 Reflection Chain (Scheduled Intelligence Review)

1. Master-Orchestrater schedules periodic review windows.
2. Reflection and Meta-Learning agents read recent windows.
3. Decision, mood, behavior, and goal agents contribute summaries.
4. Arbitration resolves conflicts.
5. Planning agent creates recommendation items.
6. User receives concise weekly intelligence digest.

---

## 8) Agent-to-Agent Communication System

## 8.1 Communication Planes

1. Control plane: state, lifecycle, and policy commands.
2. Data plane: evidence, summaries, scores, and memory references.
3. Event plane: asynchronous events (session started, chunk stored, health degraded).

## 8.2 Message Envelope Contract

```json
{
  "message_id": "uuid",
  "trace_id": "uuid",
  "timestamp": "iso8601",
  "from_agent": "agent_name",
  "to_agent": "agent_name_or_group",
  "session_id": "session_id",
  "priority": "low|normal|high|critical",
  "ttl_ms": 30000,
  "message_type": "request|response|event|handoff",
  "intent": "intent_label",
  "payload": {},
  "evidence_refs": ["memory_id_1", "memory_id_2"],
  "confidence": 0.0,
  "requires_ack": true
}
```

## 8.3 Handoff Modes

1. `delegate`: full task transfer.
2. `consult`: request additional perspective.
3. `verify`: ask another agent to validate evidence.
4. `arbitrate`: escalate conflict to arbitration.
5. `synthesize`: ask planning/orchestrator for merged output.

## 8.4 Reliability Rules

1. At-least-once event delivery for non-critical events.
2. Exactly-once write semantics for memory commit actions.
3. Idempotent handlers for repeated messages.
4. Timeout and retry with backoff for agent requests.

## 8.5 Conflict Resolution Protocol

1. Detect conflict by contradictory claims or low inter-agent agreement.
2. Request arbitration pass with evidence ranking.
3. Return reconciled result with confidence and unresolved uncertainty notes.

---

## 9) Structured Memory and Metadata Contract

## 9.1 Session Object

```json
{
  "session_id": "uuid",
  "mode": "voice|text|hybrid",
  "start_time": "iso8601",
  "end_time": "iso8601_or_null",
  "user_detected": true,
  "user_identity_confidence": 0.0,
  "health_snapshot": {
    "battery": 0,
    "charging": false,
    "thermal": "normal|warm|hot",
    "network": "offline|poor|good"
  }
}
```

## 9.2 Conversation Segment Object

```json
{
  "segment_id": "uuid",
  "session_id": "uuid",
  "speaker": "user|other_party|unknown",
  "speaker_confidence": 0.0,
  "start_time": "iso8601",
  "end_time": "iso8601",
  "raw_transcript": "text",
  "clean_transcript": "text",
  "conversation_type": "academic|personal|social|mixed",
  "retention_mode": "discard|session_context_only|structured_memory|priority_memory"
}
```

## 9.3 Memory Chunk Object

```json
{
  "memory_id": "uuid",
  "session_id": "uuid",
  "segment_id": "uuid",
  "chunk_index": 0,
  "content": "chunk text",
  "context_window": {
    "before": "previous context",
    "after": "next context"
  },
  "topics": ["topic_a", "topic_b"],
  "entities": ["entity_a", "entity_b"],
  "agent_tags": [
    "timeline",
    "causal",
    "academic"
  ],
  "importance": 0.0,
  "novelty": 0.0,
  "retrieval_priority": "low|normal|high",
  "created_at": "iso8601"
}
```

## 9.4 Tagging Rules

1. Multi-tagging is mandatory when a chunk spans multiple domains.
2. Minimum one tag, maximum six tags per chunk.
3. Tags must include both:
- reasoning tags (`timeline`, `causal`, etc.)
- domain tags (`academic`, `wellbeing`, etc.) when applicable.
4. Tag confidence must be stored for later retrieval weighting.

---

## 10) Specialized Agent Catalog (15 Agents)

Each agent includes:
1. Core function
2. Core responsibilities
3. Inputs and outputs
4. System prompt layer blueprint

## 10.1 Timeline Agent

Core function:
1. Temporal sequencing and chronology reasoning.

Responsibilities:
1. Build event timelines from session and long-term memory.
2. Identify order, recurrence, and pacing changes.
3. Answer "when" and "in what sequence" questions.

System prompt layer blueprint:
1. Mission: Produce timeline-accurate outputs grounded in dated evidence.
2. Must do: sort by timestamp, show progression, flag missing time anchors.
3. Must not: infer dates without evidence.
4. Output schema: timeline bullets + confidence + evidence refs.

## 10.2 Causal Agent

Core function:
1. Cause-effect reasoning and dependency tracing.

Responsibilities:
1. Map causal chains across events and decisions.
2. Distinguish correlation from causation confidence.
3. Explain "why" with evidence links.

System prompt layer blueprint:
1. Mission: Trace plausible causal pathways with explicit assumptions.
2. Must do: mark confidence per causal link.
3. Must not: present speculation as fact.
4. Output schema: cause tree + confidence per edge + evidence refs.

## 10.3 Reflection Agent

Core function:
1. Belief and interpretation evolution.

Responsibilities:
1. Compare old vs new perspectives.
2. Detect shifts, reversals, refinements.
3. Produce reflective summaries over time windows.

System prompt layer blueprint:
1. Mission: Explain how thinking changed and why it likely changed.
2. Must do: contrast points in time with evidence.
3. Must not: overfit from one isolated memory.
4. Output schema: before/after map + shift type + confidence.

## 10.4 Planning Agent

Core function:
1. Multi-step decomposition and synthesis.

Responsibilities:
1. Decompose complex goals into sub-queries and sub-plans.
2. Aggregate specialist outputs into coherent plans.
3. Use distractor-aware synthesis.

System prompt layer blueprint:
1. Mission: Turn ambiguity into executable structure.
2. Must do: show steps, dependencies, and risk notes.
3. Must not: skip unresolved constraints.
4. Output schema: objective, steps, dependencies, risks, next action.

## 10.5 Arbitration Agent

Core function:
1. Conflict resolution across evidence and agent outputs.

Responsibilities:
1. Detect contradictions.
2. Rank evidence by reliability and recency.
3. Provide reconciled answer or unresolved uncertainty statement.

System prompt layer blueprint:
1. Mission: Resolve conflicts with transparent rationale.
2. Must do: cite why one claim outranks another.
3. Must not: hide unresolved conflicts.
4. Output schema: selected claim, rejected claims, rationale, confidence.

## 10.6 Academic Intelligence Agent

Core function:
1. Academic domain memory and performance intelligence.

Responsibilities:
1. Track study topics, exam timelines, and learning gaps.
2. Capture preparation strategies and outcomes.
3. Support academic recall and planning.

System prompt layer blueprint:
1. Mission: optimize academic retention and execution.
2. Must do: connect topic, deadline, and performance.
3. Must not: merge unrelated subjects into one memory summary.
4. Output schema: subject map, exam map, weak topics, next study actions.

## 10.7 Personal Journaling Agent

Core function:
1. Intentional personal thought capture and preservation.

Responsibilities:
1. Store user-declared reflections, motivations, private notes.
2. Preserve emotional tone and context.
3. Support secure recall for personal entries.

System prompt layer blueprint:
1. Mission: preserve user voice and intent with minimal distortion.
2. Must do: keep first-person perspective fidelity.
3. Must not: reframe user meaning without explicit evidence.
4. Output schema: journal entry summary, themes, retrieval tags.

## 10.8 Personal Well-being Agent

Core function:
1. Day-to-day well-being awareness and supportive pattern detection.

Responsibilities:
1. Track stress, energy, rest quality, and workload pressure signals.
2. Detect deterioration and recovery patterns.
3. Provide safe well-being nudges.

System prompt layer blueprint:
1. Mission: identify supportive and harmful daily patterns.
2. Must do: connect wellbeing states to triggers and routines.
3. Must not: provide medical diagnosis claims.
4. Output schema: wellbeing state, triggers, protective actions.

## 10.9 Cognitive and Thinking Patterns Agent

Core function:
1. Understand how the user thinks, not only what they discuss.

Responsibilities:
1. Capture reasoning chains and decision style markers.
2. Detect biases, shortcuts, confusion points.
3. Produce cognitive growth insights.

System prompt layer blueprint:
1. Mission: convert reasoning traces into improvement insights.
2. Must do: separate observed pattern from interpretation.
3. Must not: label trait as permanent from sparse evidence.
4. Output schema: reasoning pattern, bias indicator, confusion map, confidence.

## 10.10 Decision Log Agent

Core function:
1. Track decision context, options, choices, and outcomes.

Responsibilities:
1. Maintain a decision ledger.
2. Attach short-term and long-term outcomes.
3. Build decision quality feedback loops.

System prompt layer blueprint:
1. Mission: make decision quality measurable over time.
2. Must do: store options considered and rationale.
3. Must not: evaluate outcomes before enough time has passed.
4. Output schema: decision record, expected outcome, observed outcome, lesson.

## 10.11 Emotional and Mood Intelligence Agent

Core function:
1. Emotional state pattern intelligence.

Responsibilities:
1. Track mood intensity, triggers, duration, and recovery.
2. Correlate emotional states with events and routines.
3. Provide trend-level emotional insights.

System prompt layer blueprint:
1. Mission: map emotional trends and recovery signatures.
2. Must do: include intensity and duration markers.
3. Must not: collapse nuanced mood states to a single label.
4. Output schema: mood episode, trigger, duration, recovery note.

## 10.12 Behavioral Patterns and Habit Tracking Agent

Core function:
1. Measure actual behavior against stated intent.

Responsibilities:
1. Track routines, adherence, and drift.
2. Detect execution gaps.
3. Identify repeatable success conditions.

System prompt layer blueprint:
1. Mission: quantify intent-versus-action gaps.
2. Must do: attach timestamps and routine anchors.
3. Must not: infer behavior without observable evidence.
4. Output schema: habit streaks, deviation events, drift score.

## 10.13 Social Interaction Intelligence Agent

Core function:
1. Analyze interaction quality and communication patterns.

Responsibilities:
1. Track conversational tone and role dynamics.
2. Tag relationship context (mentor, peer, family, etc.).
3. Detect communication strengths and friction points.

System prompt layer blueprint:
1. Mission: improve interpersonal communication effectiveness.
2. Must do: preserve context of who said what and in what tone.
3. Must not: over-interpret one isolated interaction.
4. Output schema: interaction summary, tone profile, effectiveness signals.

## 10.14 Goal and Vision Tracking Agent

Core function:
1. Align daily actions with short-term and long-term goals.

Responsibilities:
1. Track goals, milestones, progress, and drift.
2. Connect task-level behavior to larger vision.
3. Flag low-priority effort leakage.

System prompt layer blueprint:
1. Mission: keep execution aligned with declared priorities.
2. Must do: compute drift and highlight blockers.
3. Must not: treat activity as progress without outcomes.
4. Output schema: goal status, drift alerts, corrective actions.

## 10.15 Reflection and Meta-Learning Agent

Core function:
1. Convert experience into repeatable learning loops.

Responsibilities:
1. Build weekly and monthly lesson summaries.
2. Capture mistakes, corrections, and generalizable principles.
3. Produce meta-learning insights across domains.

System prompt layer blueprint:
1. Mission: transform memory history into practical wisdom.
2. Must do: explicitly connect lesson to supporting episodes.
3. Must not: output generic advice disconnected from stored evidence.
4. Output schema: lesson, supporting evidence, behavior change recommendation.

---

## 11) Prompting Stack Design

Prompt stack order:
1. Global safety and policy prompt
2. Master-Orchestrater runtime control prompt
3. Runtime Orchestrator routing prompt
4. Specialized agent prompt card
5. Task-specific retrieval context prompt

Specialized prompt card template:
1. Identity and mission
2. Input contract
3. Allowed actions
4. Mandatory checks
5. Output schema
6. Confidence scoring rules
7. Escalation rules

Prompt evolution plan:
1. Prompt v1: baseline contracts for all 15 agents.
2. Prompt v2: few-shot examples per agent.
3. Prompt v3: adaptive prompting from offline eval feedback.

---

## 12) Retrieval Strategy for 15-Agent Tagging

## 12.1 Multi-Index Retrieval

1. Semantic vector index
2. Metadata filter index
3. Time-aware index
4. Entity and relationship index
5. Agent-tag index

## 12.2 Query-Time Retrieval Plan

1. Orchestrator derives candidate agent tags from intent.
2. Retrieve by hybrid query: semantic + metadata + agent-tag filter.
3. Expand with neighboring context chunks when needed.
4. Rerank by relevance + recency + tag confidence + source quality.

## 12.3 Ingestion-Time Tagging Plan

1. Primary tag selection (1-2 tags)
2. Secondary tag selection (up to 4 tags)
3. Domain label assignment (academic, personal, social, etc.)
4. Quality score assignment for future retrieval weighting

---

## 13) Safety, Privacy, and Trust Constraints

Non-negotiable constraints:
1. Always-on listening must be explicit opt-in.
2. User can force mute/sleep at any time.
3. Sensitive content can be session-only or redacted by policy.
4. Speaker identity confidence below threshold must not be treated as user-authenticated content.
5. Stored memory must include provenance and confidence.

Policy controls:
1. Per-domain retention rules.
2. Per-source trust weighting.
3. Per-agent write permissions.
4. Audit logs for memory write and delete actions.

---

## 14) Development Roadmap for Agent 2.0

Phase A: Contracts and scaffolding
1. Add Master-Orchestrater contracts and state machine.
2. Add agent-to-agent message envelope and event bus contracts.
3. Add ingestion schemas for session, segment, and chunk.

Phase B: Live listening autonomy core
1. Implement wake/sleep runtime policy.
2. Integrate speaker-aware sessionization.
3. Implement retention scoring and selective storage.

Phase C: 15-agent tagging and retrieval
1. Add multi-label agent tagger.
2. Add agent-tag retrieval filters.
3. Add cross-agent handoff and arbitration protocol.

Phase D: Prompt system and eval
1. Ship prompt cards for all 15 specialized agents.
2. Add eval suite for precision, recall, and drift detection.
3. Tune thresholds and confidence policies.

Phase E: Mobile reliability hardening
1. Add battery/thermal/network governors.
2. Add degraded mode and emergency mute behavior.
3. Add observability dashboards and alerting.

---

## 15) Definition of Done for Agent 2.0

Agent 2.0 can be considered production-ready when:
1. Master-Orchestrater state machine is stable under long-running sessions.
2. Sessionization and speaker attribution meet accuracy targets.
3. Selective retention reduces memory pollution while preserving useful recall.
4. 15-agent tagging improves retrieval precision on evaluation sets.
5. Agent-to-agent communication has deterministic retries and conflict resolution.
6. Prompt cards are implemented and validated for all 15 specialized agents.
7. Resource governor preserves device health without major UX regression.
8. Full safety and audit controls are enabled for memory write/delete paths.

---

## 16) Final Statement

This Agent 2.0 blueprint defines a transition from a capable orchestrated RAG system to a true autonomous personal intelligence runtime.

It introduces:
1. A Master-Orchestrater for lifecycle and device-aware autonomy.
2. A structured chain-of-agents architecture.
3. A robust agent-to-agent communication model.
4. A 15-specialized-agent domain framework with explicit prompt-layer contracts.

The intended outcome is reliable, efficient, context-preserving intelligence that listens responsibly, reasons deeply, and retrieves precisely.