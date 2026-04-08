# Cortex Lab Agent Runtime (Current Implementation)

## 1) Runtime Snapshot

Current implemented and operational agent role count: **6**

1. **1 orchestration agent**
2. **5 specialized execution agents**

Roles:
1. Orchestrator agent
2. Timeline agent
3. Causal agent
4. Reflection agent
5. Planning agent
6. Arbitration agent

Notes:
1. `BaseAgent` exists as an abstract base class and is not counted as an operational role.
2. The same 6 roles run with either LocalLLM or GeminiLLM through the provider proxy.

---

## 2) Core Intent

The runtime is designed to provide **high-trust, memory-grounded reasoning** using explicit specialization, rather than relying on one-pass generic response generation.

This intent is enforced by:
1. Intent-aware routing and complexity-aware execution paths.
2. Specialized agents with domain-specific reasoning behavior.
3. Retrieval quality checks and post-retrieval correction loops.
4. Tool calling under policy constraints and deterministic loop controls.

---

## 3) End-to-End Operation Model

A query is processed in this order:

1. **Query analysis**
- Detect intent, complexity, entities, and topics.

2. **Optional LLM routing refinement**
- Runs only for ambiguous complexity windows.

3. **Query transformation**
- Builds multi-query, HyDE, step-back query, and sub-queries.

4. **Optional function-calling check**
- If tool use is selected, execute through safety and rate-limit guards.

5. **Execution path selection**
- `NO_RETRIEVAL` for very simple queries.
- `SINGLE_STEP` for moderate complexity.
- `MULTI_STEP` for complex queries with parallel specialist execution.

6. **Evidence shaping and scoring improvements**
- Context compression and importance-based evidence score boosting.

7. **Quality guardrails**
- CRAG quality evaluation.
- Self-RAG critique and optional revision.
- FLARE active retrieval for low-confidence outputs.

8. **Synthesis and response assembly**
- Unified answer, reasoning trace, evidence bundle, confidence, and pipeline trace.

---

## 4) Orchestrator Agent

### Core function
Controls the full reasoning pipeline and determines how specialist agents are used.

### Responsibilities (currently implemented)
1. Maintains intent-to-agent map and route strategy.
2. Selects `NO_RETRIEVAL`, `SINGLE_STEP`, or `MULTI_STEP` based on complexity.
3. Executes optional function-calling before specialist execution.
4. Dispatches specialist agents (single or parallel multi-agent).
5. Aggregates specialist outputs and deduplicates evidence.
6. Applies CRAG, Self-RAG, and FLARE quality loops.
7. Integrates task manager state transitions for coordinator/subagents.
8. Integrates tool safety checks (allow, require approval, deny) and rate-limit stop behavior.

### Intent-to-agent primary mapping
1. `temporal -> timeline`
2. `causal -> causal`
3. `reflective -> reflection`
4. `comparative -> arbitration`
5. `factual -> planning`
6. `procedural -> planning`
7. `exploratory -> planning`

### Complexity-to-route thresholds
1. `< 0.30 -> NO_RETRIEVAL`
2. `0.30 to < 0.60 -> SINGLE_STEP`
3. `>= 0.60 -> MULTI_STEP`

### Multi-step agent expansion rules
1. Causal intent adds timeline support.
2. Reflective intent adds causal support.
3. Temporal intent adds reflection support.
4. Planning is appended if sub-queries exist and planning is not already selected.

---

## 5) Specialized Execution Agents

## 5.1 Timeline Agent

### Core function
Temporal and chronological reasoning.

### Current behavior
1. Retrieves memories with temporal focus (`top_k=15`).
2. Sorts evidence by timestamp.
3. Generates timeline-grounded output via faithful generation.
4. Uses explicit fallback response when evidence is absent.

### Confidence behavior
1. Confidence heuristic: `min(0.5 + 0.05 * result_count, 0.95)`.

---

## 5.2 Causal Agent

### Core function
Cause-and-effect reasoning.

### Current behavior
1. Retrieves evidence (`top_k=15`).
2. Runs `causal_reason(...)` on retrieved evidence.
3. If causal output is too thin, supplements with faithful generation.
4. Returns explicit insufficiency message when evidence is absent.

### Confidence behavior
1. Confidence heuristic: `min(0.4 + 0.06 * result_count, 0.90)`.

---

## 5.3 Reflection Agent

### Core function
Belief and pattern evolution over time.

### Current behavior
1. Retrieves broader evidence (`top_k=20`) and sorts by timestamp.
2. Attempts belief change detection using earliest vs latest evidence.
3. Generates reflection-focused faithful response.
4. Appends belief-evolution note when detect_belief_change succeeds.
5. Returns explicit insufficiency message when evidence is absent.

### Confidence behavior
1. Confidence heuristic: `min(0.4 + 0.04 * result_count, 0.85)`.

---

## 5.4 Planning Agent

### Core function
Complex multi-step decomposition and synthesis.

### Current behavior
1. Uses provided sub-queries; if none, uses original query as single unit.
2. For each sub-query:
- Builds query embedding.
- Retrieves evidence (`top_k=8`).
- Produces sub-answer using faithful generation.
3. Deduplicates combined evidence by memory id.
4. Uses RAFT-style synthesis when enough unique evidence exists:
- Top evidence treated as oracle docs.
- Lower-ranked evidence treated as distractors.
- Calls `raft_generate(...)` for distractor-aware synthesis.
5. Falls back to faithful synthesis over sub-answers when evidence is smaller.
6. Returns insufficiency message when evidence is not enough.

### Confidence behavior
1. Confidence heuristic: `min(0.5 + 0.03 * unique_result_count, 0.90)`.

---

## 5.5 Arbitration Agent

### Core function
Contradiction and conflict resolution.

### Current behavior
1. Retrieves evidence (`top_k=15`).
2. Runs faithful generation with explicit conflict-resolution instructions.
3. Requests reconciliation framing (recency/context/conflict explanation) inside generation prompt.
4. Optionally runs belief-change detection between earliest and latest memories.
5. Appends belief-change explanation for contradiction/refinement outcomes.
6. Returns insufficiency message when evidence is absent.

### Confidence behavior
1. Fixed confidence baseline: `0.7`.

---

## 6) How All 6 Roles Work Together

1. Query arrives and is analyzed for intent and complexity.
2. Orchestrator selects execution path:
- Direct no-retrieval for simple requests.
- Single specialist for moderate complexity.
- Parallel multi-agent for complex requests.
3. Specialists retrieve and reason in their domain.
4. Orchestrator merges specialist outputs into one unified answer.
5. Post-processing guardrails run:
- CRAG quality evaluation.
- Self-RAG critique and optional revision.
- FLARE additional retrieval when confidence remains low.
6. Tool calling can short-circuit or enrich the flow when selected, but only through safety policy and rate-limit constraints.

---

## 7) Quality and Safety Guardrails (Current)

1. **CRAG evaluation**
- Computes quality from evidence signals and can reduce confidence or trigger supplementary retrieval behavior.

2. **Self-RAG critique**
- Uses ISREL/ISSUP/ISUSE style evaluation.
- Revises answers when quality is moderate and confidence needs improvement.

3. **FLARE active retrieval**
- Triggered for low-confidence responses after Self-RAG.
- Retrieves extra evidence for uncertain answer segments and may regenerate.

4. **Tool safety pipeline**
- Function calls are checked against policy.
- Supports allow, require-approval, and deny decisions.
- Includes deterministic tool dispatch window limits.

5. **Task lifecycle integration**
- Coordinator and subagents are tracked with explicit task states.
- Supports cancellation and failure propagation in multi-agent mode.

---

## 8) Provider Compatibility and Operational Notes

1. The orchestrator and specialist logic operate through `LLMProvider`.
2. LocalLLM and GeminiLLM expose the same core method surface used by agents.
3. In Gemini-only runtime mode, the same agent roles remain operational.
4. Function-calling stage remains available through the provider interface and safety runtime checks.

---

## 9) Summary

The current runtime is a **6-role explicit agent system**:
1. One orchestration brain for routing, coordination, quality control, and safety.
2. Five specialists for temporal, causal, reflective, planning, and arbitration reasoning.

Its core intent is to produce grounded, auditable, and higher-trust memory reasoning by combining specialization with quality guardrails, rather than producing single-pass generic answers.
