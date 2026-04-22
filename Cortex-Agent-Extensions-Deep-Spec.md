# Cortex Agent — Comprehensive Extension Architecture
## All Ways to Extend the Pi-Mono Agent Loop for the Personal Intelligence Runtime

> **Core Objective:** Build a personal autonomous intelligence runtime — a second mind that listens, ingests, reasons, and builds structured knowledge about one person over years. The foundation is the `CortexAgentLoop` (derived from pi-mono's `AgentSession`), extended through tools, extensions, skills, sessions, providers, and 10 deep applications.
>
> **Core Insight (pi-mono derived):** The agent is just a CONFIGURATION. The runtime is ONE class. The LLM IS the planner. Extensions are how you expand capability without forking the core. Every extension category below is a way to make the agent do more, know more, or act more robustly — without changing the loop itself.
>
> **This document lists every extension category, mechanism, and specific implementation possibility — tied directly to what the core objective requires.**

---

## Table of Contents

0. [Runtime-Aligned Naming and Status (Authoritative)](#0-runtime-aligned-naming-and-status-authoritative-for-implementation)
1. [The Four-Piece Agent Model and Why It Is Infinitely Extensible](#1-the-four-piece-model)
2. [Extension Category 1 — Tool Calling: The Endless Loop](#2-tool-calling-extensions)
3. [Extension Category 2 — The Extension Hook System (20+ Lifecycle Events)](#3-extension-hook-system)
4. [Extension Category 3 — Skill System Extensions](#4-skill-system-extensions)
5. [Extension Category 4 — Session Management & History Tree Extensions](#5-session-management-extensions)
6. [Extension Category 5 — Provider & Model Extensions](#6-provider-model-extensions)
7. [Extension Category 6 — Background Agent Scheduling Extensions](#7-background-scheduling-extensions)
8. [Extension Category 7 — Memory Plane Extensions (5-Plane Architecture)](#8-memory-plane-extensions)
9. [Extension Category 8 — Retrieval Pipeline Extensions (5-Tier RAG)](#9-retrieval-pipeline-extensions)
10. [Extension Category 9 — Frontend / RPC Event Protocol Extensions](#10-frontend-rpc-extensions)
11. [Extension Category 10 — Multi-Agent Orchestration Extensions](#11-multi-agent-orchestration)
12. [Extension Category 11 — Security, Permission & Privacy Extensions](#12-security-permission-extensions)
13. [Extension Category 12 — Observability & Self-Improvement Extensions](#13-observability-extensions)
14. [Extension Category 13 — Domain-Specific Agent Configuration Extensions](#14-domain-agent-configs)
15. [Extension Category 14 — Deep Application Service Extensions](#15-deep-application-extensions)
16. [Extension Category 15 — Novel Extension Frontiers (Not Yet Designed)](#16-novel-extension-frontiers)
17. [Extension Interaction Map — How Everything Connects](#17-interaction-map)
18. [Priority Order for Implementation](#18-priority-order)
19. [Frontend Web Application Deep Observability Backlog](#19-frontend-web-application-deep-observability-backlog)

---

## 0. Runtime-Aligned Naming and Status (Authoritative for Implementation)

This section is the implementation authority for naming, route contracts, and runtime status. If any older section uses different names, this section wins.

### 0.1 Canonical Agent IDs (Current Runtime)

- `l0_master`
- `l1_orchestrator`
- `timeline`
- `causal`
- `reflection`
- `planning`
- `arbitration`
- `academic`
- `journaling`
- `wellbeing`
- `cognitive`
- `decision_log` (legacy alias: `decisions`)
- `emotional`
- `behavioral`
- `social`
- `goal` (legacy alias: `goals`)
- `meta_learning`
- `wiki_agent`
- `presence`
- `session_crystallizer`
- `structured_summary_forge`

### 0.2 Canonical Service and Runtime Names

- Session forge service: `session_memory_forge_service`
- Chronicle service: `life_chronicle_service`
- Task manager: `RuntimeTaskManager`
- Background scheduler: `background_scheduler`
- Orchestration loop class: `CortexAgentLoop`

### 0.3 Canonical API Routes (Current Runtime)

- Health: `/api/health`
- Runtime tasks list: `/api/runtime/tasks`
- Runtime task events (SSE): `/api/runtime/tasks/events`
- Agent events (SSE): `/api/agent/events`
- Scheduler status: `/api/agent/scheduler/status`
- Deep app Session Forge: `/api/deep/session-forge/*`
- Deep app Chronicle passive mode: `/api/chronicle/passive/*`
- Wiki governance: `/api/wiki/lint/*`, `/api/wiki/compaction/*`

### 0.4 Canonical Extension Hook Signatures

The implemented extension hooks are:

- `on_input(text, options=None) -> InputResult`
- `on_before_agent_start(text, system_prompt) -> BeforeAgentStartResult`
- `on_tool_call(tool_name, params) -> ToolCallGateResult`
- `on_tool_result(tool_name, result_content, is_error) -> ToolResultRewrite`
- `on_agent_end(messages) -> None`
- `on_session_before_compact(entries) -> str | None`
- `on_session_compact(summary) -> None`

Planned-but-not-fully-emitted hooks in loop flow:

- `on_context(messages)`
- `on_before_provider_request(payload)`
- `on_message_start(message)`
- `on_message_delta(delta)`
- `on_message_end(message)`

### 0.5 Implementation Status Tags

- `READY`: Implemented and running in current backend.
- `PARTIAL`: Implemented with placeholders or reduced behavior.
- `PLANNED`: Not implemented yet.
- `RESEARCH`: Deferred frontier scope.

## 1. The Four-Piece Model and Why It Is Infinitely Extensible

Pi-mono's architecture, which Cortex is built upon, can be understood as four interacting primitives:

```
┌──────────────────────────────────────────────────────────────────────┐
│  PIECE 1: TOOLS                                                       │
│  Typed, schema-validated, permission-gated functions the LLM calls.  │
│  The LLM is the planner. It calls tools to act on the world.         │
│  → Extensible: add any tool. The loop never changes.                 │
├──────────────────────────────────────────────────────────────────────┤
│  PIECE 2: EXTENSIONS                                                  │
│  TypeScript/Python modules hooking 20+ lifecycle events.             │
│  They can BLOCK tool calls, REWRITE tool results, inject context,    │
│  add commands, add UI widgets, write custom JSONL entries.           │
│  → Extensible: add any cross-cutting concern without touching core.  │
├──────────────────────────────────────────────────────────────────────┤
│  PIECE 3: SESSIONS                                                    │
│  JSONL append-only tree. Persist every message, tool call, result.  │
│  Branch, fork, compact. The model always has correct context.        │
│  → Extensible: add custom entry types, compaction strategies,        │
│    branching triggers, serialization formats.                        │
├──────────────────────────────────────────────────────────────────────┤
│  PIECE 4: CONFIGURATIONS (AgentConfig)                               │
│  System prompt + tool list + extension list + session config.        │
│  Different configurations = different "agents". Same runtime.        │
│  → Extensible: add new AgentConfig instances for new agent types.    │
│    17+ configurations already defined. Infinite more possible.       │
└──────────────────────────────────────────────────────────────────────┘
```

**The power:** Any capability you want to add fits into one or more of these four pieces. You never need to fork the runtime. You never need to write custom agent orchestration logic. Every extension category below maps onto this four-piece model.

---

## 2. Tool Calling Extensions — The Endless Loop

### 2.1 What Makes the Tool Loop "Endless"

The agent loop in pi-mono is intrinsically iterative: the LLM calls tools, sees results, decides what to call next, and loops until it emits `end_turn` with no tool calls. There is no fixed step count. The loop can run for 200+ tool calls if the task warrants it. This is the foundation for all autonomous behavior.

**The extension mechanism:** add new `ToolDefinition` instances to any `AgentConfig`. The LLM will discover and use them without any other code change.

### 2.2 Core Tool Categories to Build

#### Retrieval Tools (all agents use these)

| Tool Name | Purpose | Implementation Priority |
|-----------|---------|------------------------|
| `retrieve_memory` | Dense + sparse + temporal search over event plane | ✅ EXISTS (extend) |
| `search_wiki` | Wiki page and section lookup by entity/topic | ✅ EXISTS (extend) |
| `search_claims` | Claim plane search by entity + domain + confidence | ✅ EXISTS (extend) |
| `query_graph` | Hop-bounded graph traversal from seed entities | ✅ EXISTS (extend) |
| `search_by_time` | Temporal index query — find events in time window | ✅ EXISTS (extend) |
| `retrieve_chronicle` | Search Life Chronicle entries by people/emotion/time | 🟡 BUILD PHASE 3 |
| `search_decisions` | Retrieve from decision ledger with outcome data | 🟡 BUILD PHASE 3 |

#### Analysis Tools (specialized agents use these)

| Tool Name | Agent Owner | What It Does | Returns |
|-----------|-------------|-------------|---------|
| `build_event_timeline` | Timeline | Constructs ordered event list from memory refs | `{ timeline: [...], gaps: [...], patterns: [...] }` |
| `detect_temporal_gaps` | Timeline | Finds periods with no stored events | `{ gaps: [{ from, to, severity }] }` |
| `trace_causal_chain` | Causal | Follows cause→effect edges N hops | `{ causal_tree: [...], confidence_per_edge: [...] }` |
| `detect_belief_change` | Reflection | Compares belief states across time windows | `{ shifts: [...], shift_type: [...] }` |
| `analyze_pattern` | Behavioral/General | Computes recurring patterns and drift signals | `{ patterns: [...], pattern_type: string }` |
| `score_importance` | Orchestrator | Evaluates significance of evidence chunk | `{ importance: float, rationale: string }` |
| `decompose_query` | Orchestrator | Splits complex query into sub-queries | `{ sub_queries: [...], dependencies: [...] }` |
| `classify_query_tier` + `analyze_query_intent` | Orchestrator | Identifies complexity tier and intent/domain attributes | `{ tier, complexity, intent, entities, topics }` |
| `read_mood_signal` | Emotional | Reads aggregated mood signals from recent interactions | `{ dominant_mood, confidence, signals }` |
| `map_relationship_health` | Social | Assesses health of specific relationship | `{ health: string, drift: bool, evidence: [...] }` |
| `score_goal_drift` | Goal | Computes divergence between stated goals and behavior | `{ drift_score: float, alerts: [...] }` |

#### Orchestration Tools (L1 Orchestrator only)

```python
# These tools give the orchestrator the ability to spawn and manage sub-agents
# Following pi-mono's pattern: no custom orchestration logic — just tool calls

spawn_agent_tool = ToolDefinition(
    name="spawn_agent",
    description="Start a CortexAgentLoop with a given AgentConfig to handle a sub-task",
    parameters=SpawnAgentParams,  # { agent_id, query, context, metadata }
    permission_model="auto",
    execute=lambda p, ctx: agent_factory.spawn(p.agent_id, p.query, p.context),
)

collect_agent_results_tool = ToolDefinition(
    name="collect_agent_results",
    description="Gather completed results from all spawned agents",
    parameters=CollectAgentResultsParams,    # { agent_ids }
    execute=lambda p, ctx: agent_factory.collect(p.agent_ids),
)

dissolve_team_tool = ToolDefinition(
    name="dissolve_team",
    description="Clean up and deallocate all spawned agents for this trace",
    parameters=DissolveParams,
    execute=lambda p, ctx: agent_factory.dissolve(p.agent_ids),
)
```

#### Wiki Tools (Wiki Agent only)

| Tool | Trigger Condition | Core Logic |
|------|------------------|------------|
| `extract_claims` | After each ingestion batch | LLM call with atomic claim extraction prompt |
| `upsert_claim` | After extraction | DuckDB upsert with dedup fingerprint check |
| `patch_wiki_page` | After claim upsert | PATCH/CREATE/LINT/COMPACT operations on markdown |
| `create_wiki_page` | New entity detected | Scaffold from template with standard section structure |
| `lint_wiki_page` | Background hourly | Contradiction detection, staleness check, confidence decay |
| `compact_wiki_section` | Weekly batch | Macro-compaction of old claims into canonical text |
| `resolve_contradiction` | On lint trigger | Route to Arbitration Agent + update with resolution |

#### Write Tools (any agent that modifies state)

```python
# All writes go through the permission gate — user_confirm or plan_mode_only

ingest_memory_tool = ToolDefinition(
    name="ingest_memory",
    description="Write a processed memory event to Plane 1 (event plane)",
    permission_model="user_confirm",      # prompts user before committing
    audit_required=True,                  # every call logged to audit log
)

update_graph_edge_tool = ToolDefinition(
    name="update_graph_edge",
    description="Add or update a relation edge in the knowledge graph (Plane 4)",
    permission_model="user_confirm",
)

invalidate_cache_tool = ToolDefinition(
    name="invalidate_cache",
    description="Clear T0/T1 cache entries for specified wiki page IDs",
    permission_model="auto",              # safe, no data mutation
)
```

### 2.3 Extending the Tool Loop with Meta-Tools

Meta-tools are tools that operate on other tools or on the agent's own reasoning. These represent a major extension frontier:

**Self-Diagnosis Tool:**
```python
# Tool that lets the agent assess its own confidence before committing an answer
diagnose_confidence_tool = ToolDefinition(
    name="self_diagnose_confidence",
    description="Before finalizing a response, assess whether retrieved evidence is sufficient",
    execute=lambda p, ctx: {
        "evidence_count": p.evidence_count,
        "coverage_score": compute_coverage(p.query, p.evidence),
        "recommendation": "proceed" if coverage > 0.7 else "retrieve_more",
    }
)
```

**Uncertainty Flagging Tool:**
```python
# When the agent is about to make a low-confidence claim, it can use this tool
# to explicitly mark it as uncertain before including it in the response
flag_uncertainty_tool = ToolDefinition(
    name="flag_uncertain_claim",
    description="Explicitly tag a claim as uncertain with confidence score and what would resolve it",
    permission_model="auto",
)
```

**Chain-of-Thought Inspection Tool:**
```python
# Lets the agent externalize its reasoning at any step, making it auditable
externalise_reasoning_tool = ToolDefinition(
    name="externalise_reasoning_step",
    description="Write current reasoning step to audit log for observability",
    permission_model="auto",
)
```

---

## 3. Extension Hook System — 20+ Lifecycle Events

This is the most powerful extension mechanism in pi-mono. Unlike tools (which the LLM calls), extensions fire automatically at lifecycle points. They run transparently from the model's perspective.

### 3.1 The 10 Core Hook Points (from pi-mono `ExtensionRunner`)

```
1. input                    → called when user sends text
2. before_agent_start       → called just before the LLM loop begins
3. context                  → called to filter/augment the context window sent to LLM
4. before_provider_request  → called with the raw API payload before sending
5. tool_call                → called for each tool the LLM tries to invoke (can BLOCK)
6. tool_result              → called with tool output (can REWRITE content and is_error)
7. message_start/delta/end  → called during streaming (can observe or modify)
8. agent_end                → called when the LLM loop completes one full turn
9. session_before_compact   → called before compaction runs (can provide custom compactor)
10. session_compact         → called after compaction completes
```

### 3.2 All Extensions Cortex Needs — Built Around These Hooks

#### `WikiUpdateExtension`

```python
class WikiUpdateExtension(Extension):
    """
    After every agent_end, check whether any tool results contained
    new memory writes, and if so trigger the Wiki Agent claim extraction pipeline.
    """
    async def on_agent_end(self, messages: list[AgentMessage]) -> None:
        writes = [m for m in messages if m.type == "tool_result"
                  and m.tool_name in ["ingest_memory", "write_memory_chunk"]]
        if writes:
            await wiki_agent.follow_up(f"Process {len(writes)} new events for claim extraction")
```

#### `SafetyGateExtension`

```python
class SafetyGateExtension(Extension):
    """
    On every tool_call hook, enforce safety policy via SafeToolRuntime.
    This BLOCKS any tool call that violates current policy.
    Maps directly to the existing backend/src/runtime/safety.py.
    """
    async def on_tool_call(self, event: ToolCallEvent) -> ToolCallDecision:
        if not self.safety_runtime.allows(event.tool_name, event.params):
            return ToolCallDecision.BLOCK(reason="policy_violation", details=event)
        return ToolCallDecision.ALLOW(params=event.params)
```

#### `CacheInvalidationExtension`

```python
class CacheInvalidationExtension(Extension):
    """
    After any tool that writes to wiki, memory, or claims, invalidate
    the affected T0/T1 cache entries. Ensures cache freshness automatically.
    """
    WRITE_TOOLS = {"patch_wiki_page", "create_wiki_page", "upsert_claim", "ingest_memory"}

    async def on_tool_result(self, event: ToolResultEvent) -> ToolResultEvent:
        if event.tool_name in self.WRITE_TOOLS:
            affected_ids = event.details.get("affected_wiki_page_ids", [])
            await cache.invalidate(affected_ids)
        return event
```

#### `ObservabilityExtension`

```python
class ObservabilityExtension(Extension):
    """
    Emit OpenTelemetry spans on every lifecycle event.
    Maps to existing backend/src/runtime/trace.py — just wire the hooks.
    """
    async def on_tool_call(self, event: ToolCallEvent) -> ToolCallDecision:
        self.tracer.start_span(f"tool.{event.tool_name}", trace_id=event.trace_id)
        return ToolCallDecision.ALLOW(params=event.params)

    async def on_tool_result(self, event: ToolResultEvent) -> ToolResultEvent:
        self.tracer.end_span(f"tool.{event.tool_name}", status="error" if event.is_error else "ok")
        return event
```

#### `SessionMemoryExtension`

```python
class SessionMemoryExtension(Extension):
    """
    On agent_end, extract structured 'thought objects' from the completed turn
    and queue them for the Session Crystallizer agent to process.
    This is Application 1's pipeline trigger.
    """
    async def on_agent_end(self, messages: list[AgentMessage]) -> None:
        crystallizer_queue.enqueue(SessionBatch(
            session_id=self.session.id,
            messages=messages,
            closed_at=datetime.now(),
        ))
```

#### `PresenceContextExtension`

```python
class PresenceContextExtension(Extension):
    """
    On before_agent_start, inject the CURRENT_CONTEXT_SUMMARY from the Presence Agent
    into the system prompt. The agent always has full context without extra tool calls.
    """
    async def on_before_agent_start(self, text: str, system_prompt: str) -> BeforeAgentResult:
        ctx = await presence_context_assembler.get_current_summary()
        enriched_prompt = f"{system_prompt}\n\n=== CURRENT USER CONTEXT ===\n{ctx}"
        return BeforeAgentResult(system_prompt_override=enriched_prompt)
```

#### `ThrottleExtension`

```python
class ThrottleExtension(Extension):
    """
    For background agents, enforce that total compute never exceeds
    max_compute_pct (default 20%) of available resources.
    Pauses tool_call execution if threshold is exceeded.
    """
    async def on_tool_call(self, event: ToolCallEvent) -> ToolCallDecision:
        current_load = resource_governor.get_cpu_pct()
        if current_load > self.max_compute_pct:
            await asyncio.sleep(2.0)  # back off, let active session take priority
        return ToolCallDecision.ALLOW(params=event.params)
```

#### `TruncationExtension`

```python
class TruncationExtension(Extension):
    """
    Pi-mono pattern: truncate tool results exceeding ~50KB to prevent context overflow.
    Applied on every on_tool_result call. Logs truncation events.
    """
    MAX_CHARS = 50_000

    async def on_tool_result(self, event: ToolResultEvent) -> ToolResultEvent:
        if len(event.content) > self.MAX_CHARS:
            event.content = event.content[:self.MAX_CHARS] + "\n...[truncated]"
            event.details["was_truncated"] = True
        return event
```

#### `DomainScopingExtension`

```python
class DomainScopingExtension(Extension):
    """
    For L2 specialized agents — scope their retrieval tool calls to only
    their domain. A Wellbeing Agent should never retrieve academic data.
    Enforces the agent write permission matrix from Orchestrator.md §10.2.
    """
    def __init__(self, allowed_domains: list[str]):
        self.allowed_domains = allowed_domains

    async def on_tool_call(self, event: ToolCallEvent) -> ToolCallDecision:
        if event.tool_name == "retrieve_memory":
            # Inject domain filter into params
            event.params["agent_tag_filter"] = self.allowed_domains
        return ToolCallDecision.ALLOW(params=event.params)
```

#### `HeartbeatExtension`

```python
class HeartbeatExtension(Extension):
    """
    For background agents scheduled by Paperclip-style heartbeat.
    Injects the current heartbeat context (what triggered this run, goal ancestry)
    into the agent's system prompt before each scheduled execution.
    """
    async def on_before_agent_start(self, text: str, system_prompt: str) -> BeforeAgentResult:
        hb = self.scheduler.get_current_heartbeat()
        ancestry = f"\n\n=== GOAL ANCESTRY ===\n{hb.to_prompt_string()}"
        return BeforeAgentResult(system_prompt_override=system_prompt + ancestry)
```

#### `CustomCompactionExtension`

```python
class CustomCompactionExtension(Extension):
    """
    Override pi-mono's default summarization compaction for specialized agents.
    Wiki Agent uses structured compaction (claims → wiki pages) instead of prose summary.
    """
    async def on_session_before_compact(self, messages: list) -> CompactionResult | None:
        if self.agent_id == "wiki_agent":
            # Extract claims from messages before compacting
            claims = await claim_extractor.extract_from_messages(messages)
            summary = wiki_store.build_compaction_summary(claims)
            return CompactionResult(summary=summary, claims_extracted=claims)
        return None  # let default compaction handle other agents
```

### 3.3 New Extension Types Not Yet in Pi-Mono (Cortex Innovations)

These extend beyond what pi-mono natively supports, using the extension API as the integration point:

#### `ContextWindowOptimizerExtension`
Monitors token usage per turn. When the context approaches the compaction threshold, it proactively summarizes low-importance earlier messages (not just old ones) to maximize useful context headroom.

#### `PrivacyGovernorExtension`
On every `before_provider_request` hook, scans the full payload for privacy-tier-1 content (journaling, private intentions). If found and the provider is cloud, BLOCKS the request and forces local-only execution. Privacy tiers are respected at the network layer, not just the logic layer.

#### `CrossSessionLearningExtension`
On `session_compact`, analyzes the compacted content for patterns that should inform the agent's system prompt evolution. Over weeks, the agent prompt itself becomes more personalized — the agent learns what kinds of reasoning this specific user values.

#### `EvidenceProvenanceExtension`
Wraps every tool result that contains memory content to inject `source_event_ids` and `wiki_page_ids` into the result before the LLM sees it. This ensures the LLM's claims are always traceable back to actual memory — a RAG faithfulness enforcement at the extension layer.

---

## 4. Skill System Extensions

### 4.1 What Skills Are in Pi-Mono

Skills are named, reusable reasoning workflows encoded as text files (markdown with tool usage examples). The agent can invoke a skill with `/skill_name` or the skill is injected into the system prompt at activation. They encode "how to do X" without being code.

From pi-mono's coding-agent docs: "Build CLI tools with READMEs (see Skills)." Skills are essentially structured prompts that teach the agent repeatable patterns.

### 4.2 Cortex Skills to Build

#### `weekly_reflection_skill`
```markdown
## SKILL: Weekly Intelligence Digest

TRIGGER: Sunday evening, automated by Heartbeat scheduler
AGENTS INVOLVED: Meta-Learning (15), Reflection (03), Emotional (11), Goals (14), Decisions (10)

EXECUTION PROTOCOL:
1. Retrieve all sessions from the past 7 days
2. Call build_event_timeline for each domain (timeline agent)
3. Call detect_belief_change for all topics active this week (reflection agent)
4. Call map_relationship_health for all active relationships (social agent)
5. Call score_goal_drift for each active goal (goals agent)
6. Synthesize into digest sections: [THIS WEEK IN REVIEW] [SHIFTS DETECTED] [DRIFT ALERTS] [DECISIONS MADE]
7. Store digest as wiki page in timelines/weekly/
8. Emit weekly_digest_ready event to frontend

SUCCESS CRITERIA: Digest covers >80% of session volume, all drift alerts have evidence refs
```

#### `daily_brief_skill`
```markdown
## SKILL: Morning Intelligence Brief

TRIGGER: User-configured wake time (default 07:30)
AGENTS INVOLVED: Goals (14), Behavioral (12), Wellbeing (08), Social (13), Presence (04)

EXECUTION PROTOCOL:
1. Retrieve yesterday's session summaries (Session Crystallizer output)
2. Check open loops from last 3 sessions (Gap Mapper output)
3. Check upcoming calendar events (Android device command: calendar.get)
4. Pull relationship drift alerts (Social agent output)
5. Pull top 1 goal drift alert (Goals agent output)
6. Compose brief: [MOOD SNAPSHOT] [TODAY'S FOCUS] [RELATIONSHIP NUDGE] [OPEN LOOP TO CLOSE]
7. Deliver via TTS (Presence Agent) or push notification

TONE: Brief, specific, evidence-grounded. Never generic. Always personalized.
```

#### `decision_capture_skill`
```markdown
## SKILL: Capture a Decision

TRIGGER: User says "I've decided to..." or "I'm going to go with..."
AGENT INVOLVED: Decision Log (10)

EXECUTION PROTOCOL:
1. Confirm the decision is significant (not trivial preference)
2. Retrieve context: what was the situation? what options were visible?
3. Extract stated rationale (verbatim from user's words)
4. Extract expected outcome and timeframe
5. Write to decision ledger with full schema
6. Set outcome_check triggers: 2-week, 1-month, 3-month
7. Ask: "Is there anything else about this decision I should remember?"

DO NOT: Evaluate the decision. DO: Preserve it faithfully with full context.
```

#### `pre_meeting_context_skill`
```markdown
## SKILL: Pre-Meeting Context Brief

TRIGGER: User says "I'm about to talk to [name]" or calendar event detected
AGENT INVOLVED: Social (13), Timeline (01), Wellbeing (08), Goals (14)

EXECUTION PROTOCOL:
1. Retrieve all memory chunks tagged with [name] entity
2. Get last 3 interaction summaries with this person
3. Get any open commitments to this person (from decision ledger)
4. Get any things this person told the user they were dealing with
5. Get emotional history of recent interactions (Emotional agent)
6. Compose context brief: [WHO THEY ARE] [LAST TIME] [WHAT THEY WERE DEALING WITH] [YOUR OPEN ITEMS] [RELATIONSHIP HEALTH]

DELIVERY: Voice brief (TTS) or notification, depending on current mode.
```

#### `capture_moment_skill`
```markdown
## SKILL: Life Chronicle Moment Capture

TRIGGER: User says "capture this moment" or explicit chronicle command

EXECUTION PROTOCOL:
1. Invoke node.invoke: camera.clip (60s recording)
2. Invoke node.invoke: location.get (parallel)
3. Invoke audio.transcribe (continuous STT during clip)
4. Run scene understanding: vision model + speaker detection
5. Retrieve wiki context: who are these people? what event is this?
6. Compose polished narrative (Chronicle Agent style)
7. Store to chronicle store: video, thumbnails, transcript, narrative

NARRATIVE STYLE: First-person, emotionally faithful, sensory-rich.
NOT: Reporter style. YES: Diary style.
```

### 4.3 Dynamic Skill Injection

A major extension: skills should be dynamically selected and injected based on what the agent needs in the current session. The `PresenceContextExtension` can pre-load the most relevant 2-3 skills based on the user's current context before the agent starts — so the agent already knows how to handle what's coming.

---

## 5. Session Management Extensions

### 5.1 The JSONL Tree — What It Enables

Pi-mono's session storage is an append-only JSONL file with an ID/parentID tree structure and a movable `leafId` pointer. This is not just storage — it is a branching, compactable, replayable history of every agent interaction. Each node can carry:

- `type: "message"` — standard agent message (user/assistant/toolResult)
- `type: "compaction"` — context summary with `firstKeptEntryId` boundary
- `type: "custom"` — application-specific metadata (Cortex extension point)
- `type: "customMessage"` — injected LLM context (invisible to user but in model context)

### 5.2 Custom Entry Types for Cortex

#### `thought_object_entry` — Session Crystallizer output
```json
{
  "type": "custom",
  "subtype": "thought_object",
  "thought_id": "uuid",
  "category": "self_insight",
  "domain": "creative_projects",
  "core_claim": "Fear of imperfect output is blocking book project, not time scarcity",
  "confidence": 0.92,
  "follow_up_flag": true,
  "parentId": "message_entry_uuid"
}
```

#### `wiki_patch_event_entry` — What changed in the wiki during this session
```json
{
  "type": "custom",
  "subtype": "wiki_patch_event",
  "page_id": "projects/book-project",
  "operation": "PATCH",
  "sections_updated": ["Evolving Beliefs"],
  "confidence_before": 0.71,
  "confidence_after": 0.85,
  "parentId": "tool_result_uuid"
}
```

#### `gap_signal_entry` — Recorded gaps for future retrieval
```json
{
  "type": "custom",
  "subtype": "gap_signal",
  "gap_type": "stated_priority_vs_attention",
  "entity": "app_project",
  "severity": "high",
  "days_gap": 7,
  "route_to": "presence_agent_idle_queue"
}
```

#### `decision_checkpoint_entry` — Decision outcome check trigger
```json
{
  "type": "custom",
  "subtype": "decision_checkpoint",
  "decision_id": "uuid",
  "check_at": "2025-08-01T07:00:00Z",
  "check_type": "2_week",
  "current_outcome_status": "PENDING"
}
```

### 5.3 Session Branching for Hypothetical Reasoning

Pi-mono supports branching the session tree — creating a new branch from any historical node. Cortex can use this for:

**Decision Oracle "What If" Mode:** When the user asks "what if I had chosen differently in March?", the Decision Oracle creates a branch from the session node where that decision was recorded and simulates the alternative timeline using the Causal Agent. The branch is kept for comparison but never merged back into main.

**Mirror Calibration Sessions:** The Deep Self Mirror can branch from a current session to run a "challenge" sub-session — the agent plays devil's advocate against its own self-model. The branch output informs confidence scoring.

### 5.4 Compaction Strategy Extensions

Default pi-mono compaction = LLM summarizes old messages. For Cortex, four specialized compaction strategies:

| Agent | Compaction Strategy | Implementation |
|-------|--------------------|-|
| Wiki Agent | Structured compaction: extract claims, update wiki pages, summary = "wiki updated" | `CustomCompactionExtension` |
| Presence Agent | Context compaction: preserve only live context signals, discard resolved items | Override `session_before_compact` |
| Mirror Agents | Pattern compaction: extract observed patterns with instance counts, compress examples | Custom compaction with minimum-3-instance filtering |
| Decision Log | Outcome compaction: for closed decisions, compress to `[DECISION_ID]: [OUTCOME_SUMMARY]` | Structured JSON compaction |

---

## 6. Provider & Model Extensions

### 6.1 The Provider Adapter Shim (OpenClaude Pattern)

Pi-mono provides a **unified multi-provider LLM API** (`@mariozechner/pi-ai`) that handles OpenAI, Anthropic, Google, and any OpenAI-compatible endpoint through a single interface. This is the provider adapter shim referenced throughout Cortex's architecture.

**Current working providers in pi-mono:**
- Anthropic (Claude models including claude-sonnet-4-6, claude-opus-4-6)
- OpenAI (GPT-4o, o1, etc.)
- Google Gemini (2.0 Flash, etc.)
- Any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, etc.)
- Thinking/reasoning level support across providers

### 6.2 Extension: Local Model Router

```python
class LocalModelRouterExtension(Extension):
    """
    On before_provider_request, decide whether this request should go
    to local (Ollama/LM Studio/vLLM) or cloud (Anthropic/OpenAI/Gemini)
    based on: query tier, device state, user preference, privacy level.

    This is the Provider Adapter Policy (Architecture §11.2) as an extension.
    """
    async def on_before_provider_request(self, payload: ProviderPayload) -> ProviderPayload:
        device_state = resource_governor.get_state()
        session_ctx = self.session.get_current_context()

        # Always local when offline
        if device_state.network == "offline":
            payload.provider = "ollama"
            payload.model = "deepseek-r1:7b"
            return payload

        # Always local for privacy-tier-1 content
        if session_ctx.has_privacy_tier_1_content:
            payload.provider = "ollama"
            return payload

        # T4 frontier queries — use faster cloud for synthesis
        if session_ctx.retrieval_tier == "T4" and user_prefs.cloud_enabled:
            payload.provider = "gemini-flash"
            return payload

        # Default: local first
        payload.provider = user_prefs.preferred_local_provider
        return payload
```

### 6.3 Extension: Thinking Level Control

Pi-mono supports configurable thinking/reasoning levels per model (documented in pi-mono 2.8). Cortex can vary thinking depth by query tier:

```python
class ThinkingLevelExtension(Extension):
    """
    Match model thinking level to retrieval tier.
    T0/T1 = no extended thinking (speed)
    T2 = default thinking
    T3/T4 = max extended thinking (quality)
    """
    TIER_TO_THINKING = {
        "T0": "none",
        "T1": "none",
        "T2": "default",
        "T3": "extended",
        "T4": "max",
    }

    async def on_before_provider_request(self, payload: ProviderPayload) -> ProviderPayload:
        tier = self.session.get_current_tier()
        payload.thinking_level = self.TIER_TO_THINKING.get(tier, "default")
        return payload
```

### 6.4 Extension: Model Fallback Chain

When a model fails (rate limit, timeout, network error), the extension handles failover transparently:

```python
class ModelFallbackExtension(Extension):
    """
    When a provider request fails with a retryable error,
    switch to the fallback provider before the retry state machine fires.
    Pi-mono already handles retry; this extension changes WHICH model retries.
    """
    FALLBACK_CHAIN = [
        "claude-sonnet-4-6",     # primary
        "gpt-4o",                # first fallback
        "gemini-flash",          # second fallback
        "ollama/deepseek-r1:7b", # local fallback (always available)
    ]

    async def on_tool_result(self, event: ToolResultEvent) -> ToolResultEvent:
        if event.is_error and "provider_failure" in event.content:
            self.advance_provider_chain()
        return event
```

### 6.5 Extension: Multi-Model Ensemble for High-Stakes Answers

For T4 frontier queries and Mirror Agent reports, run the same synthesis prompt through 2 different models and produce a consensus answer with confidence bounds:

```python
class EnsembleExtension(Extension):
    """
    For T4 tier or Mirror reports: run synthesis through two models,
    compare outputs semantically, and flag where they diverge.
    Divergence = genuine uncertainty the user should know about.
    """
    async def on_agent_end(self, messages: list[AgentMessage]) -> None:
        if self.session.get_tier() == "T4" and self.config.ensemble_enabled:
            primary_answer = messages[-1].content
            ensemble_answer = await second_model.synthesize(self.session.get_evidence())
            divergence = semantic_diff(primary_answer, ensemble_answer)
            if divergence > 0.15:
                await self.session.append_custom({
                    "type": "ensemble_divergence",
                    "score": divergence,
                    "note": "Models diverged on this synthesis — treat with additional uncertainty"
                })
```

---

## 7. Background Agent Scheduling Extensions

### 7.1 The Heartbeat Pattern (Paperclip-Derived)

Background agents in Cortex follow Paperclip's heartbeat scheduling: agents wake on schedule, check what work needs doing (via their tool loop), act, and sleep. Every task carries goal ancestry metadata. This is implemented as a combination of the `HeartbeatExtension` + `ScheduleConfig` in `AgentConfig`.

### 7.2 All Background Agent Schedules

| Agent | Schedule | Trigger Type | Goal Ancestry |
|-------|----------|-------------|---------------|
| Wiki Agent | On every ingestion event | Event-driven | "Build memory Wikipedia" |
| Session Crystallizer | 15-20min after session close | Event-driven | "Structure raw session knowledge" |
| Gap Mapper | Daily 02:00 | Cron | "Identify attention divergence from stated goals" |
| Belief Update Detector | Sunday 00:00 | Cron weekly | "Track intellectual evolution" |
| Summary Forge | Every 72 hours | Interval | "Build structured arc summaries" |
| Wiki Lint | Every 48 hours | Interval | "Maintain wiki quality and correctness" |
| Wiki Meso Compact | Daily 03:00 | Cron | "Compress claims into canonical wiki sections" |
| Wiki Macro Compact | Monthly first Sunday | Cron monthly | "Archive old events into canonical form" |
| Daily Brief Generator | 07:00 (configurable) | Cron daily | "Surface daily intelligence" |
| Goal Drift Alert | Sunday 18:00 | Cron weekly | "Maintain goal alignment visibility" |
| Decision Outcome Check | Per-decision TTL | Timer-per-record | "Close decision feedback loops" |
| Mirror Agents | Every 72 hours (data collection) | Interval | "Build honest self-model" |
| Mirror Report | Biweekly Saturday 08:00 | Cron biweekly | "Synthesize psychological profile" |
| Weekly Synthesis | Sunday 20:00 | Cron weekly | "Weekly intelligence review" |
| Relationship Drift Detector | Daily 09:00 | Cron daily | "Maintain relationship health awareness" |
| Presence Context Assembler | Every 30 min during active hours | Interval | "Keep context current for proactive intelligence" |
| Knowledge Amplifier | Monthly first Monday | Cron monthly | "Identify highest-leverage knowledge gaps" |
| Dream Capture Session | Daily at wake time | Cron daily | "Capture subconscious pattern data" |

### 7.3 Scheduler Implementation Extensions

**Idle-Priority Scheduling:** Not all background tasks should run on schedule. Low-priority ones (wiki macro compaction, knowledge amplifier) should only run when the system is truly idle — no active user session, battery > 50%, device not in meeting. The scheduler checks resource governor state before launching any background agent.

**Adaptive Scheduling:** The scheduler tracks execution time per agent and dynamically spreads background work across the day. If Monday morning's tasks ran overtime, it delays the next batch. Prevents background agents from competing with active use.

**Goal-Ancestry Injection:** Every scheduled task, before launch, receives a `goal_ancestry` dict that is injected into its context. This ensures every agent always knows: why am I running, what is my mission, what higher goal does this serve.

---

## 8. Memory Plane Extensions

### 8.1 Five-Plane Architecture — What's Already Designed

The five memory planes (from Agentic-RAG-Architecture.md) are the core of the persistent intelligence system:

```
P0: Working (session RAM)     → asyncio.Queue, ephemeral
P1: Event (raw facts)         → DuckDB + FAISS HNSW hot tier
P2: Claim (atomic facts)      → DuckDB + BM25 + dense embedding index
P3: Wiki (canonical pages)    → Markdown files + FAISS section index
P4: Graph (relations)         → NetworkX + DuckDB edges
```

### 8.2 Extensions to Each Plane

#### Plane 1 Extensions

**Provenance-Chain Indexing:** Every event in P1 should carry a full provenance chain: what session it came from, what ingestion pipeline version processed it, what noise filter score it received, what retention tier was assigned. An additional DuckDB table tracks this chain for every `memory_id`. This enables the future capability of "explain why this memory was kept."

**Multi-Modal Event Enrichment:** Extend P1 to store not just text content but vision captions (from Chronicle), audio intensity metadata, location context, and calendar context as first-class fields in the event schema — not just in `details`. These become filterable facets in retrieval.

**Ephemeral Session Buffer:** A separate in-memory P0.5 layer — a rolling 3-minute buffer of unprocessed audio/video (the "dashcam for life"). Never written to P1 unless the user explicitly says "save the last 3 minutes." Encrypted in-memory, auto-purged.

#### Plane 2 Extensions

**Claim Confidence Decay Model:** Claims in P2 lose confidence over time if not reinforced by new events. A background job (daily) applies a decay function: `confidence *= (1 - decay_rate * days_since_last_confirmed)`. Claims that fall below 0.3 are flagged for the Wiki Agent to review. This prevents stale beliefs from dominating retrieval.

**Cross-Claim Contradiction Index:** A DuckDB table that tracks all pairs of claims that semantically contradict each other. Built by the Arbitration Agent over time. Retrieval can use this index to proactively warn when a query touches a known controversy in the user's belief history.

**Claim Lineage Graph:** Track how each claim was derived: from which events, through which version of which agent, at which confidence level. When a claim is superseded, the lineage shows the evolution. This is the "intellectual history" layer — you can ask "when did I first believe this?"

#### Plane 3 Extensions

**Wiki Confidence Heatmap:** A live dashboard view that shows which wiki pages are high-confidence (green), medium (yellow), or degraded (red). Red pages trigger proactive user prompts: "It's been 90 days since you mentioned [topic] — do you want to update your wiki page for it?"

**Version-Diffed Wiki History:** Every wiki page update stores a diff (before_hash → after_hash, plus the actual diff content). Users can "time travel" through their wiki's evolution. The question "what did I believe about X in March?" becomes a direct wiki lookup.

**Wiki Cross-Link Discovery:** The Wiki Agent automatically discovers semantic links between wiki pages (using embedding similarity) and adds `related_pages` references. Over time, the wiki becomes a genuine hyperlinked knowledge base, not just a flat collection.

#### Plane 4 Extensions

**Temporal Edge Confidence Decay:** Like P2 claims, graph edges should decay in confidence if not reinforced. A relationship edge with `relationship_type: knows` that has no supporting event in 12 months loses confidence weight. This prevents stale social graph connections from distorting the current picture.

**Counterfactual Edge Type:** A new edge type `COUNTERFACTUAL` — edges that represent "what might have happened" from the Decision Oracle's "what if" analysis. These are explicitly marked as counterfactual, never mixed with observed edges, but navigable for exploration.

**Causal Cluster Detection:** A graph analytics extension that detects clusters of nodes with dense causal connections. These clusters represent "life domains with tight coupling" — areas where changes in one thing reliably trigger changes in another. Surfaces as insight in the Meta-Learning Agent.

---

## 9. Retrieval Pipeline Extensions

### 9.1 The Five Retrieval Tiers — What Needs to Be Built

The tiered retrieval system (T0-T4) is the core performance innovation. The routing classifier (< 80ms decision) determines which tier handles each query.

### 9.2 Tier-by-Tier Extension Points

#### T0 (Instant, < 50ms) Extensions

**Semantic Cache with Wiki Hash:** The T0 cache key includes a `wiki_revision_hash` for all relevant pages. When a wiki page is patched, the `CacheInvalidationExtension` fires and clears all T0 entries that included that page's hash. Cache is never stale by design.

**Parameterized Response Templates:** For highly frequent queries like "what's my main project?", build response templates with variable slots. `cache.get_template("main_project")` → `"Your main project is {PROJECT_NAME}, last active {LAST_ACTIVE}."` — instant, personalized, zero LLM call.

**Query Fingerprinting:** Fast semantic hashing (not exact text matching) for the T0 lookup. Two semantically identical queries ("what am I working on?" vs "what's my current project?") should hit the same cache entry. Use a lightweight embedding + locality-sensitive hashing for sub-5ms fingerprinting.

#### T1 (Wiki-Fast, < 200ms) Extensions

**Wiki Health Gating:** T1 only returns a response if the target wiki page has `confidence >= 0.6`. Below that threshold, it automatically escalates to T2. This prevents the "fast but wrong" failure mode where T1 returns an outdated or low-confidence answer.

**Section-Level Retrieval:** T1 doesn't just return the whole wiki page — it identifies the most relevant sections using BM25 + keyword matching and returns only those sections. This keeps the evidence set compact and focused.

**Wiki Freshness Scoring:** Each T1 result carries a `freshness_score` — how recently the relevant sections were updated. If freshness < 7 days, full confidence. If freshness > 30 days, inject a caveat: "Note: this information was last confirmed 47 days ago."

#### T2 (Standard RAG, < 1.5s) Extensions

**Adaptive Channel Weighting:** The four retrieval channels (dense, sparse, temporal, wiki section) have fixed weights in the base design. An adaptive extension learns per-user, per-intent-type weights from feedback signals. If temporal queries consistently get better feedback when temporal channel weight is higher, the weights self-adjust.

**Query Expansion via HyDE:** For T2 queries where initial retrieval quality is poor (CRAG score < 0.5), generate a Hypothetical Document Embedding — a synthetic ideal answer — and use its embedding to retrieve more relevant evidence. This is the HyDE extension point in the T2 pipeline.

**Proposition Extraction Cache:** Pre-compute proposition embeddings for all memory events and store them in a dedicated proposition index. T2 queries then retrieve at the proposition level (atomic facts) rather than the chunk level, improving precision for factual queries.

#### T3 (Deep Multi-Agent, < 6s) Extensions

**Parallel Agent Fan-Out:** The T3 dispatch uses pi-mono's team pattern — spawn multiple specialized agents in parallel using `spawn_agent` + `collect_agent_results`. The extension point is the team composition algorithm: which agents get dispatched depends on the detected query intents (timeline + causal for "why did X happen after Y?").

**Chain-of-Retrieval Iteration:** T3's iterative retrieval loop (up to 4 iterations) can be extended with a "gap analysis step" between iterations: after each retrieval round, the agent explicitly identifies what's still missing from the evidence set before generating the next targeted sub-query.

**Streaming Partial Answers:** While T3 deep retrieval is running, the frontend receives a partial answer based on T1/T2-quality context within the first 1-2 seconds. The `tier_selected` event tells the frontend to show a "Searching deeper..." indicator. When deep retrieval completes, a `revision` event updates the answer if it meaningfully changes.

#### T4 (Frontier Traversal, < 20s) Extensions

**Utility-Scored Frontier Expansion:** The T4 frontier expansion algorithm scores each candidate node by a utility function: `(relevance × 0.35) + (novelty × 0.25) + (trust × 0.20) + (recency × 0.10) + (provenance_completeness × 0.10)`. The extension point is the utility function itself — users can emphasize different dimensions through configuration.

**Domain-Aware Relation Priority:** T4's graph traversal prioritizes different relation types based on query intent. Extension: let users configure which relation types they consider most important for their primary query patterns. Someone who frequently asks causal questions can upweight `[cause_of, leads_to, influenced_by]`.

**Frontier Expansion Visualization:** Emit real-time `frontier_expansion_event` events as T4 traverses nodes. The frontend can show a live expanding graph visualization — making the "thinking" process visible and building user trust in deep queries.

---

## 10. Frontend / RPC Event Protocol Extensions

### 10.1 Pi-Mono's Three Operational Modes

Pi-mono's coding agent has three distinct modes (from DeepWiki analysis):
1. **Interactive** — Full TUI with real-time streaming, message queuing, status widgets
2. **RPC** — JSON protocol over stdin/stdout for programmatic control (IDE integration)
3. **Print/JSON** — Batch mode for scripts and pipelines

Cortex maps these to: Interactive → full web/mobile chat UI; RPC → background agent control; Print/JSON → automated daily brief generation.

### 10.2 The Full CortexEvent Stream

Every event the frontend needs to render the experience in real time:

```typescript
type CortexEvent =
  // Pi-mono core agent events
  | { type: "agent_start" }
  | { type: "agent_end"; messages: AgentMessage[] }
  | { type: "turn_start"; turnIndex: number }
  | { type: "turn_end"; message: AssistantMessage; toolResults: ToolResult[] }
  | { type: "message_start"; message: AgentMessage }
  | { type: "message_update"; delta: TextDelta | ThinkingDelta }
  | { type: "message_end"; message: AgentMessage }
  | { type: "tool_execution_start"; toolCallId: string; toolName: string; args: any }
  | { type: "tool_execution_update"; toolCallId: string; content: string }
  | { type: "tool_execution_end"; toolCallId: string; result: string; isError: boolean }
  | { type: "queue_update"; steering: string[]; followUp: string[] }
  | { type: "compaction_start"; reason: "threshold" | "overflow" | "manual" }
  | { type: "compaction_end"; result: CompactionResult; willRetry: boolean }
  | { type: "auto_retry_start"; attempt: number; maxAttempts: number; delayMs: number }
  | { type: "auto_retry_end"; success: boolean }

  // Cortex RAG events
  | { type: "tier_selected"; tier: "T0"|"T1"|"T2"|"T3"|"T4"; reason: string; estimated_ms: number }
  | { type: "retrieval_channel_start"; channel: "dense"|"sparse"|"wiki"|"graph"|"temporal" }
  | { type: "retrieval_channel_end"; channel: string; results: number; quality: number }
  | { type: "evidence_ready"; evidence: Evidence[]; fusion_method: "RRF"|"weighted" }
  | { type: "quality_loop"; loop: "CRAG"|"Self-RAG"|"FLARE"; score: number; rounds: number }
  | { type: "answer_plan_ready"; plan: AnswerPlan }
  | { type: "revision"; reason: string; changes: "minor"|"major" }

  // Cortex wiki / background events
  | { type: "wiki_update"; pageId: string; operation: "PATCH"|"CREATE"|"LINT"|"COMPACT"; sections: string[] }
  | { type: "belief_shift"; entity: string; shiftType: "REFINEMENT"|"REVERSAL"|"EXPANSION" }
  | { type: "gap_signal"; gapType: string; entity: string; severity: "low"|"medium"|"high" }
  | { type: "decision_checkpoint"; decisionId: string; checkType: "2_week"|"1_month"|"3_month" }

  // Cortex deep application events
  | { type: "presence_initiative"; message: string; priority: number; source: string }
  | { type: "daily_brief_ready"; briefId: string }
  | { type: "weekly_digest_ready"; digestId: string }
  | { type: "mirror_report_ready"; reportId: string; confidence: number }
  | { type: "chronicle_moment_captured"; momentId: string; narrativePreview: string }
  | { type: "relationship_drift_alert"; entityId: string; daysSinceContact: number }

  // Cortex frontier-specific events
  | { type: "frontier_expansion_step"; nodeId: string; nodeType: string; utility: number; hopNumber: number }
  | { type: "frontier_budget_consumed"; tokensUsed: number; hopsCompleted: number }
  | { type: "frontier_complete"; totalNodes: number; workingSetSize: number }
```

### 10.3 The Steering Bar — Real-Time Mid-Query Refinement

This is a direct extension of pi-mono's `_steeringMessages` + `_followUpMessages` queue system:

```
STEERING: injected BETWEEN tool rounds (during active loop)
          → Model sees the guidance before calling its next tool
          → Response redirects without restarting

FOLLOW-UP: injected only when agent is fully idle
           → Perfect for "now that you've answered that, also look at X"

USER INTERFACE:
┌─────────────────────────────────────────────────────────────────────┐
│  T3 DEEP SEARCH — timeline + causal agents running (est. 4s)       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Focus on events after January 2025 specifically...          │ ← │
│  └─────────────────────────────────────────────────────────────┘   │
│  [Send Steering]  [Queue as Follow-Up]  [Let Finish]  [Abort]      │
└─────────────────────────────────────────────────────────────────────┘

POST /api/sessions/{id}/steer   → _queue_steer(text)
POST /api/sessions/{id}/followup → _queue_follow_up(text)
```

### 10.4 The Pipeline Visualizer Extension

During T2-T4 queries, the frontend should render a live pipeline visualization:

```
[Query: "Why did I stop exercising in October?"]
  ↓
[T3 Selected: multi-hop causal + temporal]         ← tier_selected event
  ↓
[Retrieval Channels Running in Parallel]           ← channel_start × 7
  ├── Dense:    ████████░░ 8 results               ← channel_end events
  ├── Sparse:   ███████░░░ 6 results
  ├── Timeline: █████░░░░░ 4 results
  └── Graph:    ███░░░░░░░ 3 results  (2-hop from 'exercise' entity)
  ↓
[Evidence Merged via RRF → 12 unique pieces]        ← evidence_ready event
  ↓
[Agents Dispatched]
  ├── Timeline Agent: ordering events...            ← tool_execution events
  └── Causal Agent:   tracing cause chain...
  ↓
[CRAG Quality: 0.81 — proceeding]                   ← quality_loop event
  ↓
[Generating response...]                             ← message_update stream
```

---

## 11. Multi-Agent Orchestration Extensions

### 11.1 Agent-as-Config: Why This Is the Right Model

The fundamental insight from pi-mono's architecture is that there is no special "orchestrator class." The orchestrator is just an agent with `spawn_agent` and `collect_agent_results` in its tool set. The L1 Runtime Orchestrator dispatches work to L2 agents by calling tools — and the LLM inside the L1 decides which agents to spawn based on its system prompt and the current query.

This means: **every improvement to multi-agent coordination is either a new tool, a new extension, or a new AgentConfig.** There is no separate orchestration framework to maintain.

### 11.2 All 17 AgentConfig Extensions

The following configurations represent the full agent registry. Each is a `CortexAgentLoop` instance — same runtime, different system prompt + tool set:

```
L0 Layer:
    l0_master                     → noise filter + session governance + resource tiers

L1 Layer:
    l1_orchestrator               → RAG routing + team dispatch + CRAG/Self-RAG/FLARE

L2 Specialized Agents:
    timeline                      → temporal queries, event ordering, pattern detection
    causal                        → cause-effect chains, dependency mapping
    reflection                    → belief evolution, intellectual growth tracking
    planning                      → multi-step decomposition, goal execution
    arbitration                   → conflict resolution, evidence ranking
    academic                      → study patterns, knowledge gaps, exam tracking
    journaling                    → private reflections, highest privacy tier
    wellbeing                     → health signals, pattern intelligence (non-clinical)
    cognitive                     → reasoning patterns, bias detection
    decision_log                  → decision tracking, outcome analysis
    emotional                     → mood episodes, trigger correlations
    behavioral                    → habit adherence, intent-action gap
    social                        → relationship dynamics, communication patterns
    goal                          → goal drift, milestone tracking
    meta_learning                 → cross-domain lesson extraction, wisdom synthesis

Background Agents:
    wiki_agent                    → always-on wiki builder (claim extract → page patch)
    presence                      → always-on companion with proactive intelligence
    session_crystallizer          → post-session thought object extraction
    structured_summary_forge      → 72h structured arc summarization
    session_forge_gap_mapper      → implemented as service/scheduler task (not AgentConfig)
    session_forge_belief_detector → implemented as service/scheduler task (not AgentConfig)
    life_chronicle_passive        → implemented as service/scheduler task (not AgentConfig)
    mirror/relationship/oracle/amplifier/dream agents → planned
```

### 11.3 Team Communication Extensions

**Timeout-Resilient Team Pattern:**
```python
# If an agent in the team times out, the orchestrator proceeds with available outputs
# and marks missing outputs explicitly in the synthesis
async def collect_with_timeout(trace_id, timeout_ms=8000):
    results = {}
    async with asyncio.timeout(timeout_ms / 1000):
        results = await agent_factory.collect(trace_id)
    missing = [a for a in team_members if a not in results]
    if missing:
        for agent_id in missing:
            results[agent_id] = {"status": "TIMEOUT", "confidence": 0.0}
    return results
```

**Priority-Weighted Evidence Merge:**
After collecting from multiple agents, the merge step can weight each agent's contribution by its domain confidence × query relevance. Timeline Agent output gets more weight on temporal queries. Causal Agent output gets more weight on "why" queries.

**Arbitration as Default:**
Automatically route all team outputs through the Arbitration Agent before synthesis, not just when explicit contradictions are detected. This catches subtle inconsistencies — two agents saying similar things with opposite emotional valence, or different confidence levels on the same claim.

---

## 12. Security, Permission & Privacy Extensions

### 12.1 Permission Model Extensions

Beyond pi-mono's `auto | user_confirm | plan_mode_only` permission modes, Cortex adds:

**`privacy_tiered` mode:** Before executing, check the target memory's privacy tier. Tier 1 content (journaling, private intentions) requires user confirmation even for reads (not just writes). Tier 3 content (health, financial) requires elevated confirmation and is always logged.

**`batch_approval` mode:** For bulk operations (wiki macro compaction, archive migration), present the full plan as a human-readable summary and request a single approval for the batch. No action is taken until approved.

**`audit_only` mode:** For highly sensitive operations, execute but generate a detailed audit entry that the user can review later. Used for cross-agent data sharing between privacy-tier-2 and higher-tier agents.

### 12.2 Privacy Governor Extension

```python
class PrivacyGovernorExtension(Extension):
    """
    Enforce privacy tiers at the network and context layer.
    Tier 1 (journaling) never leaves the device.
    Tier 3 (health/financial) requires explicit per-use consent.
    """
    async def on_before_provider_request(self, payload: ProviderPayload) -> ProviderPayload:
        # Scan payload for privacy-tier-1 content markers
        for message in payload.messages:
            if contains_privacy_tier_1(message.content):
                if payload.provider in CLOUD_PROVIDERS:
                    # BLOCK cloud request, force local
                    payload.provider = "ollama"
                    payload.model = user_prefs.local_model

        # Scan for tier-3 content
        for message in payload.messages:
            if contains_privacy_tier_3(message.content):
                if not await user_has_granted_tier3_consent(self.session.id):
                    raise PrivacyViolation("Tier-3 content requires explicit consent for this session")

        return payload
```

### 12.3 Speaker Attribution Extension

For voice capture sessions, speaker attribution must be validated before any content is committed to long-term memory:

```python
class SpeakerAttributionExtension(Extension):
    """
    Ensure only user-attributed content (confidence >= threshold) is
    written to personal memory. Third-party speech in ambient sessions
    is either discarded or stored as 'other_party' attribution.
    """
    CONFIDENCE_THRESHOLD = 0.75

    async def on_tool_call(self, event: ToolCallEvent) -> ToolCallDecision:
        if event.tool_name == "ingest_memory":
            chunk = event.params.get("chunk", {})
            if chunk.get("speaker_confidence", 0) < self.CONFIDENCE_THRESHOLD:
                if chunk.get("speaker") == "user":
                    # Low confidence user attribution — hold for review, don't auto-commit
                    return ToolCallDecision.BLOCK(reason="low_speaker_confidence")
        return ToolCallDecision.ALLOW(params=event.params)
```

### 12.4 Sandboxed Execution Extension (NemoClaw Pattern)

For agents that invoke device capabilities (camera, microphone, location), wrap all `node.invoke` calls in policy-enforced sandbox checks:

```python
class DeviceSandboxExtension(Extension):
    """
    Before any device capability invocation (camera, audio, location),
    verify: user has granted consent for this capability, current privacy
    policy allows it, and the call is logged to the audit trail.
    """
    DEVICE_TOOLS = {"camera_snap", "camera_clip", "audio_capture", "location_get"}

    async def on_tool_call(self, event: ToolCallEvent) -> ToolCallDecision:
        if event.tool_name in self.DEVICE_TOOLS:
            if not consent_registry.has_consent(event.tool_name):
                return ToolCallDecision.BLOCK(reason="no_device_consent")
            audit_log.record_device_access(event)
        return ToolCallDecision.ALLOW(params=event.params)
```

---

## 13. Observability & Self-Improvement Extensions

### 13.1 Trace Architecture Extension

Every query should generate a full distributed trace. The `ObservabilityExtension` covers per-tool-call tracing. Additionally:

**Per-Tier Performance Tracking:** After every query completion, record actual vs. target latency per tier. Build a rolling 7-day latency distribution. Alert when P95 degrades above target.

**Wiki Health Dashboard:** Track `wiki_page_confidence_avg`, `contradiction_ratio`, `stale_claim_ratio`, `provenance_completeness` over time. Visualize as a health graph. Surface degradation before it affects query quality.

**Retrieval Quality Diagnostics (RAGChecker):** Weekly automated evaluation on 100 sampled queries. Compute faithfulness, context precision, context recall per tier and per intent type. Feed results back to the routing classifier to tune tier thresholds.

### 13.2 Self-Improvement Loop Extensions

**Routing Calibration:** If T3 queries consistently have high CRAG correction rates, the routing classifier is underestimating complexity. A weekly calibration job adjusts the routing thresholds based on observed quality metrics.

**Wiki Agent Priority Tuning:** If T1 cache hit rate falls below 10%, the wiki is not being built fast enough. The scheduler increases Wiki Agent invocation priority and batch size.

**Prompt Card Evolution (v2, v3):** Each agent's system prompt is not static. A `PromptEvolutionExtension` tracks agent output quality scores (confidence distributions, escalation rates, CRAG correction rates per agent) and flags prompts that need revision. Agents with high escalation rates need more specific processing protocols.

**Hard-Negative Mining for Retrieval:** Collect queries where CRAG score was low. Extract the retrieval failure pattern (semantic miss? temporal miss? entity miss?). Feed as training signal for BGE embedding fine-tuning with QLoRA.

---

## 14. Domain-Specific Agent Configuration Extensions

### 14.1 Making Each of the 17 Agents More Powerful

Every agent in the Cortex system is an `AgentConfig`. The system prompt is the primary lever for making each agent more capable. Here are the specific extension vectors for each:

**Timeline Agent extensions:**
- Add a `temporal_uncertainty_quantification` tool that explicitly models how uncertain a timestamp inferred from context actually is (vs. a directly stated date)
- Add a `recurrence_prediction` tool that, given detected patterns, predicts when the pattern will next occur
- Few-shot examples in system prompt for the 5 most common query types it handles

**Causal Agent extensions:**
- Add a `counterfactual_exploration` tool: "if X had not happened, would Y have occurred?" — uses the graph's COUNTERFACTUAL edge type
- Add `confounding_detector` tool: identifies third variables that explain apparent correlations without true causation
- Extend to track multi-year causal chains, not just recent events

**Reflection Agent extensions:**
- Add a `belief_confidence_map` tool: shows not just what the user believes but how strongly and how consistently across different framings of the same question
- Add a `perspective_shift_detector`: notices when the user's framing of an issue changes even when their stated position doesn't

**Presence Agent extensions:**
- Configurable persona system: the user can define the agent's name, personality traits, communication style, and relationship dynamic. The persona is stored as a wiki page that the agent reads at every context assembly.
- Proactive engagement scheduling intelligence: learns when the user is most receptive to different types of engagement (factual questions vs. emotional check-ins vs. planning discussions) and adapts timing accordingly

**Meta-Learning Agent extensions:**
- Add `principle_violation_tracker`: when the agent detects the user acting against a previously crystallized lesson, flag it with evidence — this is the "accountability" feature no other system provides
- Add `transferable_lesson_scorer`: assess how broadly applicable each extracted lesson is (specific to one context vs. general life principle) and weight the wisdom digest accordingly

---

## 15. Deep Application Service Extensions

### 15.1 Application 1 — Session Memory Forge

**Emotion Arc Detection (Session Crystallizer v2):** Beyond extracting thought objects, detect the emotional arc of each session. Did the user start anxious and end calm? Did a topic shift their energy? Store as `emotional_arc_record` in the session — feeds the Mirror Agents with longitudinal emotional data.

**Conversation Quality Scoring:** Score each session for depth of thinking (surface talk vs. genuine exploration), decision density (how many decisions or commitments were made), and intellectual progress (new claims added vs. old claims reinforced). Surfaces as a session quality metric in the Life OS Dashboard.

### 15.2 Application 3 — Deep Self Mirror

**The Contradiction Report (Mirror Agent D extension):** Beyond "here is the gap between stated values and behavior," produce an explicit contradiction map: "You said X on March 3rd. Your behavior from March 4th to March 20th contradicts X. Confidence: 0.84." This is not judgment — it is the most honest intelligence the system produces, and no human can do it reliably.

**Cognitive Style Trajectory:** Track how the user's thinking style evolves over quarters. Are they becoming more analytical? More emotionally expressive? More or less confident? The Thought Archaeologist tracks this as a time series.

### 15.3 Application 4 — Presence Agent

**Multi-Modal Presence:** The Presence Agent should know which interaction mode is appropriate for the current context. When the user is driving: voice only. When in a meeting: silent mode, notifications only. When at home in the evening: voice or text, longer interactions okay. This context awareness is derived from calendar, location, and motion data.

**Emotional Resonance Calibration:** Over time, the Presence Agent learns which topics, framings, and tones resonate most positively with the user. It calibrates its communication style to match — not to be obsequious, but to be genuinely effective.

### 15.4 Application 9 — Decision Oracle

**Past-Future Bridge:** When consulting the oracle on a current decision, the system can show a direct comparison: "Here is how you thought about this type of decision 1 year ago, and here is how you think about it now. What changed?" This makes intellectual growth visible in the moment it matters most.

**Decision Pattern Library:** After enough decisions accumulate, extract a personal "decision signature" — the user's characteristic way of making decisions (what they optimize for, what they fear, what they under-weight). This library informs new decision consultations automatically.

### 15.5 Application 10 — Knowledge Amplifier

**The Socratic Extension:** When the knowledge graph identifies a structural gap (a highly connected concept the user doesn't understand yet), the system generates Socratic questions designed to help the user reason toward that understanding themselves — not just to tell them the answer.

**Cross-Domain Connection Discovery:** The knowledge amplifier's highest-value output is not "you don't know X" but "your knowledge of X directly connects to something you know deeply in Y, and the connection unlocks Z." These serendipitous connections are what true knowledge amplification feels like.

---

## 16. Novel Extension Frontiers (Not Yet in the Architecture)

These represent truly new capabilities not yet designed in any of the three documents — but directly enabled by the four-piece model:

### 16.1 Multi-Agent Debate Extension

For complex queries where the evidence genuinely supports multiple reasonable conclusions, spawn two specialized agents with opposing priors and have them argue:

```python
# Agent A: argue for "reason was primarily emotional"
# Agent B: argue for "reason was primarily situational"
# Arbitration Agent: evaluate the arguments, produce calibrated conclusion
```

This produces far better-calibrated answers than single-agent synthesis for genuinely ambiguous questions.

### 16.2 Longitudinal Personality Drift Tracker

A new agent that measures not momentary state but multi-year identity trajectory. "Who am I becoming?" answered with trend analysis, not snapshots. Tracks: vocabulary evolution (are you using more technical terms? more emotional terms?), value emphasis drift (are health topics appearing more or less?), social complexity evolution (are your relationships becoming more or less complex?).

### 16.3 Pre-Commitment Device Integration

When a decision is logged, the Decision Log Agent can (with user permission) create calendar events and reminders at the 2-week, 1-month, and 3-month outcome check points. These become part of the person's actual calendar — not just internal system events. The agent sends itself a scheduled message at each checkpoint.

### 16.4 Dream Pattern Cross-Reference Engine

Dream content is captured separately (Application 8) but currently analyzed in isolation. A cross-reference extension that finds semantic links between dream imagery and current life stressors, open loops, and major decisions — not to claim mystical meaning, but to surface when the pattern "unusual dream frequency about [topic]" correlates with documented life stress in that domain.

### 16.5 Adaptive System Prompt Evolution

The most ambitious extension: over months, the CortexAgentLoop learns what framing, emphasis, and level of detail works best for this specific user. The agent prompt itself becomes personalized. Not through explicit prompt editing, but through a `PromptEvolutionExtension` that learns from session quality scores, user feedback signals, and output calibration data — and periodically proposes prompt updates for user approval.

### 16.6 Secure Third-Party Intelligence Sharing

With explicit, fine-grained, revocable user consent: share specific wiki page summaries (not raw events) with trusted third parties. Example: share a "professional summary wiki page" with a therapist who can see your documented patterns without accessing raw transcripts. End-to-end encrypted, time-limited, audited.

### 16.7 Cross-Device Intelligence Sync Protocol

Today's architecture is single-device local-first. Extension: a differential sync protocol that syncs only deltas between devices (phone, tablet, laptop) without syncing raw events. Wiki pages sync. Claims sync. Raw events stay local. The intelligence is ubiquitous; the raw personal data stays on the device that captured it.

### 16.8 Ambient Environment Intelligence

Beyond capturing what the user says, capture what they're doing: what apps are open, what documents are being written, what websites are visited (with explicit permission). An `AmbientContextExtension` that enriches memory events with this behavioral context — "wrote 800 words of the book chapter" becomes as storable as "talked about the book." This closes the gap between what the user says they do and what they actually do.

---

## 17. Extension Interaction Map

How all extensions interact with each other:

```
USER INPUT
    │
    ▼
PrivacyGovernorExtension (input hook: scan for tier-1/3 content)
    │
    ▼
SpeakerAttributionExtension (input hook: validate voice attribution)
    │
    ▼
PresenceContextExtension (before_agent_start: inject context summary)
HeartbeatExtension (before_agent_start: inject goal ancestry)
    │
    ▼
[LLM CALLS TOOLS]
    │
    ├── SafetyGateExtension (tool_call: block policy violations)
    ├── DomainScopingExtension (tool_call: inject domain filters)
    ├── DeviceSandboxExtension (tool_call: gate device capability access)
    ├── ThrottleExtension (tool_call: enforce compute budget for background agents)
    │
    ├── [TOOL EXECUTES]
    │
    ├── TruncationExtension (tool_result: enforce ~50KB limit)
    ├── CacheInvalidationExtension (tool_result: invalidate on writes)
    ├── EvidenceProvenanceExtension (tool_result: inject source_event_ids)
    ├── ObservabilityExtension (tool_result: emit trace spans)
    │
    ▼
[LLM LOOP CONTINUES — calls more tools as needed]
    │
    ▼
SessionMemoryExtension (agent_end: queue for Session Crystallizer)
WikiUpdateExtension (agent_end: trigger Wiki Agent for new events)
CrossSessionLearningExtension (agent_end: check for prompt evolution signals)
    │
    ▼
CustomCompactionExtension (session_before_compact: specialized strategies per agent)
    │
    ▼
USER RESPONSE + EVENTS TO FRONTEND
(CortexEvent stream: tier_selected, evidence_ready, wiki_update, etc.)
```

---

## 18. Priority Order for Implementation

Based on impact on core objective vs. implementation effort:

### Phase 0 — The Loop Itself (Weeks 1-3)
**Impact:** Everything else depends on this.
1. `CortexAgentLoop` — universal pi-mono-based runtime
2. `CortexSessionManager` — JSONL persistence with compaction
3. `RetryStateMachine` — auto-retry with exponential backoff
4. `SteeringFollowUpManager` — steering + follow-up queues
5. `ExtensionRunner` — 10 lifecycle hooks
6. `ObservabilityExtension` + `SafetyGateExtension` — baseline cross-cutting concerns

### Phase 1 — Memory Compounding (Weeks 3-7)
**Impact:** Makes the second query free. Knowledge compounds.
1. `WikiStore` — markdown CRUD + frontmatter + version history
2. `ClaimExtractor` — LLM-based atomic claim extraction
3. `WikiAgent` AgentConfig — always-on background builder
4. `WikiUpdateExtension` — triggers wiki agent after every ingestion
5. `CacheInvalidationExtension` — keeps T0/T1 fresh
6. T0 cache + T1 wiki-fast retrieval tier

### Phase 2 — Session Intelligence (Weeks 7-12)
**Impact:** Turns raw sessions into structured knowledge. Enables Applications 3-10.
1. `SessionCrystallizer` AgentConfig — post-session thought extraction
2. `GapMapper` AgentConfig — daily attention gap detection
3. `HeartbeatExtension` + Scheduler — background agent orchestration
4. `SessionMemoryExtension` — automatic crystallizer queue population
5. Custom JSONL entry types for thought objects and wiki events

### Phase 3 — Deep Retrieval + Answer Quality (Weeks 12-16)
**Impact:** T2-T4 pipeline quality. Answers become trustworthy.
1. T2 standard RAG (all 4 channels)
2. T3 deep multi-agent dispatch (team pattern)
3. T4 frontier expansion (budgeted utility-scored traversal)
4. `AnswerPlan` unified contract for stream/non-stream parity
5. `SteeringBar` + Runtime Operations Center frontend components

### Phase 4 — Presence + Mirror + Gaps (Weeks 16-24)
**Impact:** The psychological and relational intelligence layer. Applications 3, 4, 5 live.
1. `PresenceAgent` AgentConfig — always-on companion
2. `MirrorArchaeologist` × 4 + biweekly report
3. `GapIntelligence` × 3 detectors + weekly digest
4. `PresenceContextExtension` — real-time context injection

### Phase 5 — Relationships + Life OS + Oracle (Weeks 24-32)
**Impact:** Applications 6, 7, 9, 10. Synthesis and wisdom layer.
1. `RelationshipMemoryEngine` service
2. `LifeOSDashboard` — daily/weekly/monthly generators
3. `DecisionOracle` — past-self consultation
4. `KnowledgeAmplifier` — personal knowledge graph gaps

### Phase 6 — Novel Frontiers (Weeks 32+)
**Impact:** Beyond what any existing personal intelligence system offers.
1. Multi-agent debate extension
2. Longitudinal personality drift tracker
3. Adaptive system prompt evolution
4. Ambient environment intelligence
5. Cross-device sync protocol

## 19. Frontend Web Application Deep Observability Backlog

This section is a concrete TODO and execution backlog for web UI observability and long-running backend work visibility.

### 19.1 Primary UX Objectives

- Show exactly which agents are active right now.
- Show task lifecycle for queued/running/waiting/blocking/completed/failed/cancelled work.
- Show parent-child agent/task graph and trace ancestry.
- Show that backend work continues even if user navigates away or refreshes.
- Keep user trust high during long T3/T4 and background processing windows.

### 19.2 Runtime Data Sources (Use Existing APIs First)

- `/api/runtime/tasks`
- `/api/runtime/tasks/events` (SSE)
- `/api/agent/events` (SSE)
- `/api/agent/scheduler/status`
- `/api/runtime/health`
- `/api/runtime/memory-quality/history`

### 19.3 Required Frontend Surfaces

1. Runtime Operations Center page
    - global status card (runtime mode, provider health, scheduler health)
    - active agent count and active task count
    - queue depth, blocked count, approval-waiting count
2. Live Agent Graph panel
    - node per agent/task
    - edge for spawn/parent lineage
    - color by state and severity
3. Task Queue and History board
    - tabs: queued, running, waiting approval, blocked, completed, failed
    - filters by session_id, agent_id, trace_id
4. Trace Timeline drawer
    - ordered event timeline with durations
    - replay mode for postmortem
5. Long-Running Work persistence strip
    - sticky bar showing continuing background work
    - reconnect/resubscribe UX after refresh

### 19.4 Backend Enhancements Needed for Deep UI Observability

- Normalize event schema across runtime and agent streams:
  - `event_id`, `event_type`, `timestamp`, `trace_id`, `session_id`, `agent_id`, `task_id`, `parent_task_id`, `state`, `note`.
- Add summarized counters endpoint for low-cost polling fallback.
- Persist runtime task history to durable store (not memory-only) for restart-safe observability.
- Add task SLA metadata (`started_at`, `eta_ms`, `duration_ms`, `last_heartbeat_at`).
- Add explicit queue metrics endpoint (`queued`, `running`, `blocked`, `approval_waiting`).

### 19.5 Frontend TODO (Execution Order)

1. Build SSE store for `/api/runtime/tasks/events` and `/api/agent/events` with auto-reconnect.
2. Build Runtime Operations Center with state counters and health widgets.
3. Build Task Queue board with filters and search.
4. Build Agent Graph visualization with spawn/child links.
5. Build Trace Timeline drawer and raw event inspector.
6. Add session-resume behavior to restore active long-running tasks after reload.
7. Add degraded-mode polling fallback when SSE disconnects repeatedly.
8. Add notifications for state transitions (`blocked`, `waiting_approval`, `failed`).

### 19.6 KPI Targets

- Time-to-first-visible-status under 500 ms after page load.
- SSE reconnect success over 99% in active sessions.
- Missing-event rate under 0.1% per 10,000 task events.
- User-visible "system is working" confidence score over 4.5/5 in testing.

---

*Cortex Agent Extension Architecture v1.0*
*Derived from: pi-mono source (badlogic/pi-mono), Agentic-RAG-Architecture.md v6.0, Orchestrator.md Agent 4.0, Cortex-Deep-Applications.md v1.0*
*Core insight: Tools call tools. Extensions gate everything. Sessions preserve everything. Configs define everything. The loop runs forever.*
