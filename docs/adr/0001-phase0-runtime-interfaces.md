# ADR 0001: Phase 0 Runtime Interfaces

Date: 2026-04-05
Status: Accepted
Owners: Cortex core engineering

## Context

Cortex requires a stable interface baseline before deeper autonomous runtime work.
Current behavior is implemented across engine, orchestrator, retrieval, and server routes,
but lacks explicit contracts for tool metadata, runtime loop budgets, policy decisions,
and task lifecycle transitions.

Without a frozen interface layer, later phases (safe tool runtime, deterministic loop,
subagent isolation, MCP federation) risk churn and inconsistent behavior.

## Decision

Adopt a Phase 0 runtime interface package in backend source:

1. Tool contracts:
- `ToolParameterSpec`
- `ToolContract`
- `ToolRiskTier`

2. Runtime loop interface:
- `RuntimeLoopBudget`
- `RuntimeRequestEnvelope`
- `RuntimeLoopState`
- `StopReason`

3. Policy interface:
- `PolicyRule`
- `PolicyDecision`
- `PolicyEffect`
- `DangerousCommandSignal`
- `PolicyAuditEvent`
- `PolicyInterface`

4. Task state model:
- `TaskState`
- `TaskLifecycle`
- `TaskTransitionRecord`
- `LifecycleTransitionError`

5. Core operation representability:
- Add `build_core_tool_catalog()` for current engine operations.
- Add engine accessor `get_tool_contracts()`.
- Add API endpoints:
  - `/api/runtime/tool-contracts`
  - `/api/runtime/interfaces`

## Rationale

1. Contracts-first de-risks later implementation phases.
2. Versioned tool contracts support compatibility and governance.
3. Explicit stop reasons and budgets are required for deterministic execution.
4. Policy effects and audit events provide a safe path for approval and denial workflows.
5. Task lifecycle transitions prevent invalid runtime state mutations.

## Consequences

Positive:
1. Baseline interfaces are now test-locked and discoverable via API.
2. Current operations can be reasoned about as tool contracts.
3. Future phases can integrate against stable models rather than implicit behavior.

Trade-offs:
1. This introduces new model types before full runtime execution plumbing.
2. Some catalog risk tiers are initial defaults and may need tuning with red-team evidence.

## Validation

Automated tests:
1. `backend/tests/test_phase0_runtime_contracts.py`
- Contract schema generation
- Budget validation guardrails
- Stop reason coverage
- Policy approval semantics
- Task transition validity and invalid transition protection
- Core operation catalog coverage
- Engine-level contract accessor

## Follow-up

1. Phase 1: connect policy interface to real permission queue and approvals.
2. Phase 2: wire runtime loop state and stop reasons into orchestrator loop telemetry.
3. Phase 3: bind task lifecycle model to background/multi-agent execution manager.
