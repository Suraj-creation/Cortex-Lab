# Cortex Lab New Branch Plan

Branch: feature/autonomous-agent-roadmap
Date: 2026-04-05
Owner: Cortex core engineering
Status: Planning baseline approved for execution handoff

---

## 1) Mission and Outcome

Build a truly autonomous, reliable, and secure Cortex agent runtime that matches the product vision while staying grounded in what is currently implemented.

Primary outcome for this branch:
1. Create a reality-based roadmap that closes the gap between design docs and code.
2. Import proven reliability patterns from `claude-code` into Cortex architecture decisions.
3. Define a phased implementation contract with measurable gates and clear ownership boundaries.
4. Prepare the next execution sessions to implement without architecture churn.

Non-goal for this branch:
1. Large feature implementation in the same planning commit.
2. Rewriting the full backend/mobile stack before contracts and interfaces are frozen.

---

## 2) Source-of-Truth Inputs

This plan is synthesized from:
1. `ARCHITECTURE_DEEP_DIVE.md` (actual implemented behavior and current divergences).
2. `Vision-Plan.md` (product ambition and user experience goals).
3. `RAG-Architecture.md` (advanced research backlog and production optimization ideas).
4. `Gemma-4-implementation.md` (mobile offline runtime and deployment reality).
5. `../claude-code` (reference patterns for robust autonomous agent runtime).

Reality rule for all future execution:
1. Current-state decisions must start from `ARCHITECTURE_DEEP_DIVE.md`.
2. Aspirational capabilities from `RAG-Architecture.md` are introduced only with explicit phase gates.
3. Mobile offline strategy follows the runtime-mode contract from `Gemma-4-implementation.md`.

---

## 3) Critical Gap Map to Close

## 3.1 High-priority reliability and autonomy gaps

1. Streaming path quality asymmetry.
- Current stream path skips Self-RAG and FLARE, reducing quality protections in the default user flow.
- Risk: user-visible inconsistency between streaming and non-streaming behavior.

2. Tooling and agent contracts are not yet first-class runtime contracts.
- Need typed tool contracts, execution envelopes, policy integration, and deterministic loop controls.
- Risk: fragile orchestration and hard-to-debug agent behavior under load.

3. Missing governance primitives.
- Need explicit permission queue, dangerous-command classifier, and policy-limit enforcement.
- Risk: unsafe tool execution and poor enterprise readiness.

4. Design-only advanced features still unimplemented.
- RAPTOR, RAGChecker, semantic chunking boundary detection, retriever fine-tuning, failure-aware refinement, stronger security controls.
- Risk: roadmap drift and overclaiming capability without production quality.

5. Mobile offline runtime is not yet first-class.
- Current app is backend-URL driven; local inference adapter and model-pack lifecycle are pending.
- Risk: inability to satisfy true offline expectations.

## 3.2 Robustness patterns to port from claude-code

Target references for implementation patterns:
1. Runtime loop hardening: `../claude-code/src/QueryEngine.ts:130`.
2. Permission pipeline hardening: `../claude-code/src/hooks/toolPermission/PermissionContext.ts:222`.
3. Subagent orchestration behavior: `../claude-code/src/tools/AgentTool/runAgent.ts:748`.
4. Bounded memory extraction and replay: `../claude-code/src/services/extractMemories/extractMemories.ts:598`.
5. Context construction discipline: `../claude-code/src/context.ts:116`.
6. MCP client connection and timeout controls: `../claude-code/src/services/mcp/client.ts:595`.
7. Internal MCP server contract shape: `../claude-code/mcp-server/src/server.ts:257`.

---

## 4) Target Architecture Baseline

## 4.1 Core runtime layers for Cortex

1. Agent Runtime Kernel.
- Deterministic query loop, bounded iterations, stop reasons, tool rate limits, retry matrix.

2. Tool Fabric and Contract Layer.
- Typed tool schema, validation, execution envelopes, side-effect auditing, compatibility versioning.

3. Safety and Policy Layer.
- Permission queue, auto-approval classifier, dangerous-command classifier, policy-limit sync.

4. Multi-agent Coordination Layer.
- Coordinator mode, scoped subagents, isolation modes, background task lifecycle, cancellation.

5. Memory and Context Layer.
- Bounded memory extraction, memory load budget, context assembly invariants, personal-info precision paths.

6. Retrieval and Quality Layer.
- Multi-channel retrieval, CRAG path parity, streaming-safe quality controls, refinement loop.

7. MCP Federation Layer.
- Internal MCP tools, external MCP clients, auth/timeout/retry policy, schema versioning.

8. Mobile Offline Inference Layer.
- Mode-aware routing (`local_offline`, `hybrid`, `cloud`), model-pack lifecycle, local retrieval store.

9. Observability and Evaluation Layer.
- Pipeline traces, stop-reason analytics, latency SLO dashboards, red-team safety metrics.

## 4.2 Primary execution surfaces in this repo

Backend:
1. `backend/server.py`
2. `backend/src/engine.py`
3. `backend/src/agents/orchestrator.py`
4. `backend/src/agents/specialized.py`
5. `backend/src/retrieval/hybrid_retriever.py`
6. `backend/src/retrieval/query_engine.py`
7. `backend/src/ingestion/__init__.py`
8. `backend/src/storage/*`
9. `backend/src/llm/*`
10. `backend/src/observability.py`

Frontend:
1. `frontend/src/app/*`
2. `frontend/src/components/*`

Mobile:
1. `mobile/App.tsx`
2. `mobile/shared/core/types.ts`
3. `mobile/shared/core/api/client.ts`
4. `mobile/shared/core/storage.ts`
5. `mobile/shared/core/inference/*` (new)

Infra/ops:
1. `requirements.txt`, `backend/requirements.txt`
2. `scripts/*`
3. new model-pack manifest/signing scripts from Gemma plan

---

## 5) Phased Implementation Roadmap (12 Weeks)

This section follows the required phase structure and adds concrete execution scope.

## Phase 0 (Week 1): Architecture Baseline and Interfaces

Deliver:
1. Tool contract schema.
2. Runtime loop interface.
3. Policy interface.
4. Task state model.

Execution scope:
1. Define canonical tool contract schema for Cortex backend tools and future MCP tools.
2. Define runtime loop interface (request envelope, tool loop budget, stop-reason enum, telemetry fields).
3. Define policy interface (permission decision model, audit event model, dangerous-command rule model).
4. Define task lifecycle model (queued, running, waiting_approval, blocked, completed, failed, cancelled).
5. Produce architecture decision records (ADRs) for loop, tool, and policy contracts.

Gate:
1. All current Cortex operations can be represented as tools without behavior loss.
2. Contract review completed by backend, frontend, and mobile owners.

---

## Phase 1 (Weeks 2-3): Safe Tool Runtime

Deliver:
1. Tool registry.
2. Permission queue.
3. Dangerous-command classifier.
4. Approval UX.

Execution scope:
1. Build central tool registry with capability tags and policy requirements.
2. Add permission queue with explicit approval states and timeout behavior.
3. Add dangerous-command classifier inspired by `claude-code` permission setup patterns.
4. Implement frontend approval surfaces for pending risky actions.
5. Add full audit log for denied/approved/executed actions.

Gate:
1. Zero unapproved high-risk actions in red-team tests.
2. 100 percent of tool invocations produce structured audit records.

---

## Phase 2 (Weeks 4-5): Deterministic Agent Loop

Deliver:
1. Bounded loop.
2. Rate-limited execution.
3. Retry matrix.
4. Stop-reason analytics.

Execution scope:
1. Add max iterations, max tool calls per window, and token/time budgets.
2. Add source-aware retry matrix (capacity, timeout, network, model transient).
3. Add stop reason taxonomy and dashboard counters.
4. Align streaming and non-streaming quality control paths.
5. Add graceful fallback path for max-token and capacity pressure scenarios.

Gate:
1. No runaway iterations in stress tests.
2. Graceful fallback is verified for token and capacity pressure.
3. P95 loop latency and retry behavior meet defined SLOs.

---

## Phase 3 (Weeks 6-7): Multi-Agent and Isolation

Deliver:
1. Coordinator role.
2. Subagent spawning.
3. Worktree isolation.
4. Background task lifecycle.

Execution scope:
1. Add coordinator planner agent for decomposition and delegation.
2. Add subagent execution context with scoped permissions and cancellation propagation.
3. Add optional worktree isolation for risky code-edit agents.
4. Add unified task manager for foreground/background jobs with notifications.
5. Add sidechain transcript capture for subagent traceability.

Gate:
1. Parallel task throughput improves without cross-task side effects.
2. Cancellation and failure propagation are deterministic.
3. Scoped permissions prevent parent-level escalation by subagents.

---

## Phase 4 (Weeks 8-9): Memory and Ambient Personalization

Deliver:
1. Memory extraction jobs.
2. Bounded prompt loading.
3. Retrieval augmentation for personal and ambient context.

Execution scope:
1. Build bounded memory extraction with anti-duplication and safety constraints.
2. Add strict memory prompt budgets and relevance filtering.
3. Improve personal info precision pathways (name, email, phone, education, project entities).
4. Integrate ambient signals as optional retrieval context features.
5. Add memory quality evaluation (precision/recall for personal context prompts).

Gate:
1. Measurable gain in personal-context response accuracy.
2. Reduced irrelevant recall in sampled eval set.
3. Memory extraction jobs remain stable under sustained ingestion load.

---

## Phase 5 (Weeks 10-11): MCP Federation and External Integrations

Deliver:
1. MCP client manager.
2. Internal MCP server tools.
3. Auth and timeout hardening.

Execution scope:
1. Build MCP client manager with connection caching, timeout defaults, auth error pathways.
2. Expose selected Cortex capabilities via internal MCP server tools.
3. Add external MCP integrations behind policy controls and capability gating.
4. Add standardized schema validation for MCP tool request/response payloads.
5. Add retry and circuit-breaker protections for external MCP failures.

Gate:
1. External tool calls meet reliability and latency SLOs.
2. Unauthorized or malformed MCP calls fail closed.
3. Internal MCP tool contract tests pass in CI.

---

## Phase 6 (Week 12): Production Hardening

Deliver:
1. Policy-limit sync.
2. Remote-managed settings.
3. Chaos/retry drills.
4. Audit dashboards.

Execution scope:
1. Add policy-limit synchronization and cached fallback behavior.
2. Add remote-managed settings for runtime knobs and emergency controls.
3. Run chaos scenarios (model outage, retrieval failure, MCP timeout, partial storage corruption).
4. Build operational dashboards: stop reasons, retries, permission outcomes, latency, memory quality.
5. Complete release runbook and rollback playbook.

Gate:
1. Release checklist passes for reliability, safety, observability.
2. Operational drill results documented and accepted.

---

## 6) Mobile Offline Track (Parallel to Phases 1-6)

The mobile runtime plan is integrated, not separate from autonomy goals.

## 6.1 Required runtime modes

1. `local_offline`: local model + local retrieval, no inference network calls.
2. `hybrid`: local-first inference, optional cloud enrichment.
3. `cloud`: backend-first fallback for low-end devices or skipped model packs.

## 6.2 Sequencing contract

1. Freeze mode and profile contracts before native runtime work.
2. Implement inference router before local model-pack distribution.
3. Implement capability gating before default profile assignment.
4. Implement integrity checks before activation of any model pack.
5. Keep existing backend-required features isolated from offline core.

## 6.3 Mobile quality gates

1. Airplane-mode chat works after model install.
2. Unsigned/corrupted model packs are never loaded.
3. Rollback path works after failed update.
4. Thermal and memory safety limits are enforced per device tier.

---

## 7) Advanced Optimization and Refactor Backlog (Prioritized)

P0 (must implement for robustness baseline):
1. Streaming path quality parity with non-streaming safeguards.
2. Permission queue and dangerous-command classifier.
3. Bounded loop and stop-reason analytics.
4. Task lifecycle manager with cancellation and notifications.

P1 (high value, post-baseline):
1. Failure-aware query refinement.
2. Semantic chunking boundaries.
3. RAGChecker-style diagnostics and regression dashboards.
4. Retriever fine-tuning pipeline for user-memory domain.

P2 (strategic, schedule after stability):
1. RAPTOR hierarchical indexing.
2. Memory consolidation pipelines.
3. Continuous self-improvement loops with safety guardrails.

Security and compliance refactors:
1. API authentication for sensitive surfaces.
2. Data-at-rest protection strategy for local stores.
3. Tool execution audit retention and integrity checks.

---

## 8) Learning Plan for Building Autonomous Agents

This is a required competency track while implementing phases.

1. Runtime Engineering.
- Study loop control, budgets, and stop-reason handling in `../claude-code/src/QueryEngine.ts:130`.
- Output artifact: Cortex loop spec and retry matrix.

2. Safety Systems.
- Reproduce permission pipeline behavior from `../claude-code/src/hooks/toolPermission/PermissionContext.ts:222`.
- Output artifact: permission queue state machine and policy event schema.

3. Multi-Agent Orchestration.
- Implement minimal coordinator and subagent prototype from `../claude-code/src/tools/AgentTool/runAgent.ts:748`.
- Output artifact: subagent lifecycle contract and scoped-context API.

4. Memory and Context.
- Practice bounded memory extraction and replay from `../claude-code/src/services/extractMemories/extractMemories.ts:598` and `../claude-code/src/context.ts:116`.
- Output artifact: memory budget policy and context assembly invariants.

5. Interop and Operations.
- Build one internal MCP tool and one external MCP integration using `../claude-code/src/services/mcp/client.ts:595` and `../claude-code/mcp-server/src/server.ts:257`.
- Output artifact: MCP reliability checklist and integration test harness.

---

## 9) Engineering Governance for Execution

## 9.1 Done criteria per phase

1. Contracts and tests merged.
2. Telemetry and dashboards updated.
3. Failure modes validated.
4. Documentation updated.
5. Rollback path documented.

## 9.2 Testing matrix

1. Unit tests for contracts, policies, and routing logic.
2. Integration tests for retrieval, agent loop, MCP, and task lifecycle.
3. Safety tests for dangerous-command handling and approval bypass attempts.
4. Load and chaos tests for retries, timeouts, partial failure, and recovery.
5. Mobile device-matrix tests for profile gating, thermal pressure, and offline behavior.

## 9.3 Metrics to track continuously

1. Retrieval precision, faithfulness, hallucination fallback rate.
2. Stop-reason distribution and retry outcomes.
3. Permission approval/deny rates and unsafe-action prevention count.
4. Task throughput, cancellation success rate, and cross-task isolation incidents.
5. Mobile TTFT, tokens/sec, crash-free sessions, offline success rate.

---

## 10) Immediate Next Session Plan (Execution Kickoff)

1. Freeze Phase 0 interfaces and publish ADRs.
2. Create scaffold modules for tool contracts, policy contracts, and task state model.
3. Add red-team baseline tests for high-risk tool actions.
4. Add deterministic loop budget fields and stop-reason enums to request tracing.
5. Start mobile inference router scaffold in parallel with backend runtime contract work.

---

## 11) Branch Workflow

Current branch for this planning baseline:
1. `feature/autonomous-agent-roadmap`

Execution branch strategy for follow-up implementation:
1. Keep this branch as roadmap source.
2. Create phase-specific branches from this branch for each implementation phase.
3. Merge phase branches only after phase gate evidence is attached.

---

## 12) Final Note

This plan intentionally prioritizes robustness and verifiable behavior over broad feature count. The guiding principle is: stabilize the autonomous runtime first, then scale capabilities.

---

## 13) Continuation Intent (What We Actually Want)

This section turns the remaining phases into explicit product intent so execution does not drift.

## 13.1 Phase 2 core intent: predictable reliability under pressure

1. The system should fail safely and predictably, not mysteriously.
2. Every retry should be explainable by source and bounded by policy.
3. Tool dispatch should be deterministic and protected by windowed limits.
4. Operators should see stop-reason SLO health in one glance.

Success signal:
1. When incidents happen, we can answer "why did this stop" and "what retried" with structured telemetry, not guesswork.

## 13.2 Phase 3 core intent: autonomy with containment

1. Multi-agent throughput should increase without cross-task contamination.
2. Subagents must never inherit broader privileges than intended.
3. Cancellation should be immediate, complete, and auditable.

Success signal:
1. Parallel work improves speed while isolation guarantees remain intact.

## 13.3 Phase 4 core intent: personalization without memory pollution

1. Memory extraction should improve relevance and factual precision, not just memory volume.
2. Prompt memory loading must stay budgeted and query-relevant.
3. Ambient signals should be optional enrichment, never a reliability dependency.

Success signal:
1. Personal-context accuracy improves while irrelevant recall decreases.

## 13.4 Phase 5 core intent: extensibility without trust erosion

1. MCP federation should broaden capability while staying fail-closed by default.
2. External integrations must be timeout-bounded, schema-validated, and policy-gated.
3. Internal MCP tools should be contract-tested like first-party APIs.

Success signal:
1. New integrations increase capability without creating a new outage surface.

## 13.5 Phase 6 core intent: operational confidence at release time

1. Runtime limits and policy knobs should be remotely governable with safe defaults.
2. Chaos drills must prove recovery behavior, not just document expectations.
3. Audit dashboards should answer reliability and safety posture in near real time.

Success signal:
1. Release readiness is evidence-driven and reversible, not confidence-driven.

## 13.6 Execution discipline for all remaining phases

1. Every phase change must ship with tests, telemetry, rollback notes, and operator-visible diagnostics.
2. No capability is considered complete unless failure mode behavior is validated.
3. Feature growth is secondary to deterministic behavior, auditability, and safety.
