# Gemma 4 Orchestrator-Aligned Speech-to-Speech Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a production-grade, orchestrator-aligned Gemma integration across mobile and web, with full speech-to-speech architecture and local-first runtime contracts, while keeping the current Gemini-based path fully operational until Gemma model packs are downloaded later.

**Architecture:** Preserve the existing backend and ambient voice system, then add a cross-platform inference routing layer that supports `cloud`, `hybrid`, and `local_offline` modes. Mobile is the primary always-on speech-to-speech runtime ("Eva" wake flow); web gets voice parity within browser constraints. L0/L1/L2 orchestrator contracts from `Orchestrator.md` are treated as hard requirements.

**Tech Stack:** FastAPI backend, existing ambient voice pipeline, React/Next.js web app, React Native mobile app (Expo EAS Dev Build), native Gemma runtime bridge, model-pack manifest/signing pipeline, local SQLite/vector store for offline RAG, and strict audit/permission gates.

---

## 1. Executive Direction (What We Implement Now vs Later)

### 1.1 Implement Now (Required)

- End-to-end integration contracts for STT -> Orchestrator -> RAG -> TTS.
- Provider and mode routing across backend, mobile, and web.
- Full mobile always-on "Eva" orchestration path.
- Web voice interaction path aligned to the same orchestrator contracts.
- Local runtime adapters, model registry, downloader, verifier, and install-state machine.
- Local memory/RAG storage contracts and retention/compaction policy.
- Quality loops (CRAG, Self-RAG, FLARE), trace IDs, audit, and permission checks.

### 1.2 Defer Until Later (Explicitly Deferred)

- Downloading and activating large Gemma model packs on this machine right now.
- Final per-device Gemma latency/thermal tuning that requires target devices and real packs.

### 1.3 Current Default Runtime (Today)

- Gemini remains the default testing/runtime provider path.
- Existing traditional and Gemini voice providers remain intact.
- Existing APIs and UI paths continue to work while Gemma-local pieces are integrated behind feature flags.

---

## 2. Current Baseline (Repo-Grounded)

### 2.1 Existing Orchestrator and Runtime Contracts

- Layer model is already defined in `Orchestrator.md`: L0 Master-Orchestrator, L1 Runtime Orchestrator, L2 Specialized Agents.
- Flow contracts already defined and must be preserved:
  - CAPTURE FLOW (always-on voice path)
  - QUERY FLOW (agent dispatch and synthesis)
  - REFLECTION FLOW (scheduled)
- Bus contracts exist and must be mirrored in implementation design:
  - `CONTROL_BUS`
  - `DATA_BUS`
  - `EVENT_BUS`

### 2.2 Existing Runtime in This Repository

- LLM provider switching is currently `local` and `gemini` in:
  - `backend/server.py`
  - `backend/src/llm/__init__.py`
- Ambient voice pipeline is already substantial in:
  - `backend/src/ambient/__init__.py`
  - `backend/src/ambient/config.py`
  - `backend/src/ambient/vad.py`
  - `backend/src/ambient/transcription.py`
  - `backend/src/ambient/tts.py`
  - `backend/src/ambient/gemini_voice.py`
  - `backend/src/ambient/wake_word.py`
- Mobile already has ambient UI/controls:
  - `mobile/src/screens/AmbientVoiceScreen.tsx`
  - `mobile/shared/core/api/client.ts`
  - `mobile/shared/core/types.ts`
- Web already has ambient UI/controls:
  - `frontend/src/components/AmbientPanel.tsx`
  - `frontend/src/lib/api.ts`
  - `frontend/src/lib/types.ts`
- Mobile Gemma inference module tree from the previous plan is **not yet present** under `mobile/shared/core/inference/` and must be created.

### 2.3 Constraint to Preserve

- Current working behavior must not regress:
  - chat path
  - ambient voice controls
  - provider switching
  - memory ingestion

---

## 3. Scope and Non-Goals

### 3.1 In Scope

- Full architecture and integration for Gemma-ready speech-to-speech runtime.
- Mobile always-active Eva invocation path using wake-word and orchestrator gating.
- Web speech path aligned with same orchestration and memory contracts.
- Local/offline-capable routing and storage contracts.

### 3.2 Out of Scope for This Immediate Cycle

- Shipping all model weights inside app binaries.
- Removing Gemini or traditional providers.
- Requiring offline-only behavior before model packs are available.

---

## 4. Target Runtime Modes and Fallback Strategy

### 4.1 Runtime Modes

| Mode | Platform | LLM Path | STT Path | TTS Path | Internet Dependency |
|---|---|---|---|---|---|
| `cloud` | mobile + web | backend (`gemini` or existing `local`) | existing backend provider | existing backend provider | required |
| `hybrid` | mobile + web | prefer local gemma when ready, fallback cloud | local or cloud | local or cloud | optional |
| `local_offline` | mobile primary, web optional limited | on-device Gemma | on-device STT | on-device TTS | none at runtime |

### 4.2 Provider Contract Expansion

Current contracts are narrow and must be upgraded while preserving backward compatibility.

```ts
// Target cross-platform contract
type InferenceMode = "cloud" | "hybrid" | "local_offline";
type LLMProvider = "local" | "gemini" | "gemma_local";
type VoiceProvider = "traditional" | "gemini" | "local";

interface RuntimeSelection {
  mode: InferenceMode;
  llmProvider: LLMProvider;
  sttProvider: VoiceProvider;
  ttsProvider: VoiceProvider;
  allowCloudFallback: boolean;
}
```

### 4.3 Fallback Resolution Rules

1. If `mode=local_offline`, never call cloud providers.
2. If local model or local voice component is unavailable in `local_offline`, return explicit offline-unavailable error (no silent cloud fallback).
3. If `mode=hybrid`, local path is first attempt; cloud fallback allowed only if `allowCloudFallback=true`.
4. If `mode=cloud`, use existing backend provider logic unchanged.

---

## 5. End-to-End Speech-to-Speech Architecture

### 5.1 Mobile Always-On Eva Flow (Primary)

Mobile is the canonical always-active speech-to-speech runtime.

```text
Mic -> VAD -> Wake Word ("Eva") -> Speaker ID -> STT ->
L0 Master-Orchestrator (noise filter + relevance + health gates) ->
L1 Runtime Orchestrator (intent + agent routing + evidence merge) ->
RAG (local/offline or hybrid) ->
Response synthesis (CRAG + Self-RAG + FLARE) ->
TTS -> audio playback -> memory/audit write
```

### 5.2 Web Voice Flow (Parity)

Web supports the same logical flow with browser/platform constraints:

- Session-active voice (click-to-start, optional wake strategy where supported).
- Same L0/L1 contracts and quality loops.
- Same response synthesis contract and trace propagation.
- Same memory write/audit constraints.

### 5.3 Activation/Deactivation Contract (Hard Requirement)

Adopt the orchestrator sequence as implementation rule:

Activation order:
1. Start VAD.
2. Start lightweight speaker check.
3. Start STT after wake/session conditions are met.
4. Start model workers only when active processing begins.
5. Enable TTS only when response is ready.

Deactivation order:
1. Stop TTS first.
2. Flush ingestion queue.
3. Stop model workers.
4. Return STT to passive VAD state.

### 5.4 Barge-In and Interrupt Contract

- User speech while TTS is playing immediately triggers barge-in:
  - pause/stop TTS
  - preserve partial response in trace
  - open new query turn
- This applies in all modes (`cloud`, `hybrid`, `local_offline`).

---

## 6. Orchestrator Alignment Matrix (Must Match `Orchestrator.md`)

| Orchestrator Requirement | Gemma Plan Requirement | Primary Implementation Areas |
|---|---|---|
| L0 is gatekeeper | Add L0-compatible noise/relevance/privacy gate before memory commit | `backend/src/agents/orchestrator.py`, `backend/src/ambient/__init__.py` |
| Capture Flow | Keep STT ingestion path aligned to VAD -> filter -> chunk -> embed -> write -> index | `backend/src/ambient/*`, `backend/src/ingestion/__init__.py` |
| Query Flow | Route all voice/text queries through L1 decision logic and quality loops | `backend/src/agents/orchestrator.py`, inference routers |
| CRAG + Self-RAG + FLARE | Enforce in non-trivial responses across cloud/hybrid/local | L1 runtime + adapter layer |
| CONTROL/DATA/EVENT buses | Introduce explicit event envelope and trace propagation in mobile/web/backend | new runtime bus/contracts files |
| Sessionization contract | Enforce open/close rules and metadata object for voice sessions | ambient service + conversation store |
| Permission order | Schema -> scope -> resource -> privacy -> user permission -> audit | backend gate + mobile/web write actions |
| Auditability | Every retention/provider/mode decision logged with reason | backend audit + telemetry modules |

---

## 7. STT/TTS and Gemma Capability Mapping

Gemma is the reasoning model path. Speech IO must still be handled by STT/TTS modules.

### 7.1 Current Working Providers (Keep)

- STT: `traditional`, `gemini`
- TTS: `traditional`, `gemini`
- LLM: `gemini` (and existing backend `local` contract)

### 7.2 Target Added Providers

- STT: add `local` on-device path for offline mode.
- TTS: add `local` on-device path for offline mode.
- LLM: add `gemma_local` provider path.

### 7.3 Provider Policy

- Do not remove existing providers.
- Add Gemma-local capabilities as additive, not replacing current operations.
- Use explicit provider availability checks; no silent fallback in `local_offline`.

---

## 8. Memory, RAG, and Retention Contract for Voice Sessions

### 8.1 Mandatory Ingestion Rules

- Apply noise filtering before persistence.
- Apply relevance scoring dimensions:
  - user relevance
  - semantic novelty
  - future utility
  - personal significance
  - policy safety
- Apply retention tiers:
  - `<0.30` discard
  - `0.30-0.49` session-only
  - `0.50-0.74` structured memory
  - `>=0.75` priority memory

### 8.2 Memory Object Minimum Fields

- `session_id`
- `timestamp`
- `speaker_id` and confidence
- `mode` (`voice`, `text`, `hybrid`)
- `source` (`ambient`, `chat`, `local_offline`, etc.)
- `trace_id`
- `agent_tags`

### 8.3 Local RAG for Offline Mode

- Local conversation store + retrieval corpus on device (SQLite + vector index).
- Prompt builder must consume local context first in `local_offline`.
- Sync/merge queue when returning online in `hybrid` mode.

### 8.4 Storage Control on Device

- Retention + compaction mandatory for offline data:
  - hot: recent full-fidelity turns
  - warm: compressed summaries + selective turns
  - cold: highly compressed summaries and pinned facts

---

## 9. Security, Privacy, and Permission Enforcement

### 9.1 Permission Check Order (Required)

1. Schema validation
2. Agent scope check
3. Resource governor check
4. Privacy policy check
5. User permission check
6. Audit log write

### 9.2 Privacy Rules

- Journaling/private tiers cannot be cross-exposed without explicit user policy.
- Sensitive domains require hold/review policy handling.
- No long-term write for low-confidence unknown speaker attribution.

### 9.3 Integrity Rules for Model Packs

- Manifest signature validation.
- Chunk hash and full file hash validation.
- Never activate unverifiable model artifacts.

---

## 10. Repository Change Plan (Exact Touchpoints)

### 10.1 Existing Files to Modify

Backend:
- `backend/server.py`
- `backend/src/llm/__init__.py`
- `backend/src/agents/orchestrator.py`
- `backend/src/ambient/__init__.py`
- `backend/src/ambient/config.py`
- `backend/src/ambient/gemini_voice.py`
- `backend/src/ambient/transcription.py`
- `backend/src/ambient/tts.py`
- `backend/src/ambient/wake_word.py`
- `backend/src/ingestion/__init__.py`
- `backend/src/storage/vector_store.py`

Mobile:
- `mobile/App.tsx`
- `mobile/shared/core/types.ts`
- `mobile/shared/core/api/client.ts`
- `mobile/src/screens/AmbientVoiceScreen.tsx`
- `mobile/src/screens/SettingsScreen.tsx`
- `mobile/src/modals/SettingsSheet.tsx`
- `mobile/app.json`
- `mobile/package.json`

Web:
- `frontend/src/lib/types.ts`
- `frontend/src/lib/api.ts`
- `frontend/src/components/AmbientPanel.tsx`
- `frontend/src/components/LiveTranscript.tsx`
- `frontend/src/app/page.tsx`

Infra/Build:
- `backend/requirements.txt` (only if backend hosts new manifest helpers)

### 10.2 New Mobile Files

- `mobile/shared/core/inference/types.ts`
- `mobile/shared/core/inference/router.ts`
- `mobile/shared/core/inference/adapters/localOfflineAdapter.ts`
- `mobile/shared/core/inference/adapters/cloudAdapter.ts`
- `mobile/shared/core/inference/capability/deviceProfiler.ts`
- `mobile/shared/core/inference/capability/profileSelector.ts`
- `mobile/shared/core/inference/modelpacks/manifest.ts`
- `mobile/shared/core/inference/modelpacks/downloader.ts`
- `mobile/shared/core/inference/modelpacks/verifier.ts`
- `mobile/shared/core/inference/modelpacks/registry.ts`
- `mobile/shared/core/inference/localrag/store.ts`
- `mobile/shared/core/inference/localrag/retriever.ts`
- `mobile/shared/core/inference/localrag/ingestion.ts`
- `mobile/src/components/modelpacks/ModelRecommendationCard.tsx`
- `mobile/src/components/modelpacks/ModelDownloadManager.tsx`
- `mobile/src/components/modelpacks/OfflineReadinessBadge.tsx`

### 10.3 New Web Files

- `frontend/src/lib/inference/types.ts`
- `frontend/src/lib/inference/router.ts`
- `frontend/src/lib/inference/adapters/cloudAdapter.ts`
- `frontend/src/lib/inference/adapters/localAdapter.ts`
- `frontend/src/components/voice/WebVoiceRuntimePanel.tsx`
- `frontend/src/components/modelpacks/WebOfflineReadinessBadge.tsx`

### 10.4 New Backend Runtime Files

- `backend/src/runtime/contracts.py`
- `backend/src/runtime/event_bus.py`
- `backend/src/runtime/permission_gate.py`
- `backend/src/runtime/health_governor.py`
- `backend/src/runtime/session_manager.py`
- `backend/src/runtime/trace.py`

### 10.5 Native Integration and Infra Files

- `mobile/plugins/withGemmaRuntime.js`
- `mobile/modules/gemma-runtime/` (native bridge)
- `infra/modelpacks/manifest.schema.json`
- `infra/modelpacks/release-manifest.json` (generated)
- `scripts/modelpacks/build_modelpack.py`
- `scripts/modelpacks/sign_manifest.py`

---

## 11. API and Event Contracts

### 11.1 API Surfaces (Additive)

- `GET /api/runtime/mode`
- `POST /api/runtime/mode`
- `GET /api/runtime/providers`
- `POST /api/runtime/providers`
- `GET /api/modelpacks/manifest`
- `POST /api/modelpacks/verify`
- `GET /api/runtime/health`

Existing ambient endpoints remain and are extended, not replaced.

### 11.2 Event Envelope (Cross-Platform)

```json
{
  "message_id": "uuid-v4",
  "trace_id": "uuid-v4",
  "session_id": "uuid-v4",
  "plane": "CONTROL_BUS | DATA_BUS | EVENT_BUS",
  "event_type": "session_started",
  "payload": {},
  "timestamp": "iso8601",
  "schema_version": "2.0"
}
```

### 11.3 Required Runtime Events

- `session_started`
- `session_ended`
- `eva_wake_triggered`
- `chunk_stored`
- `health_degraded`
- `conflict_detected`
- `response_spoken`
- `fallback_applied`

---

## 12. Implementation Phases (Production Track)

### Phase P0: Contract Freeze and No-Regression Rules

**Outcome:** all runtime/type contracts frozen and compatibility guarantees documented.

- [ ] Freeze `InferenceMode`, provider, and event envelope contracts.
- [ ] Document backward compatibility for existing `local/gemini` behavior.
- [ ] Lock no-regression acceptance list for ambient, chat, and memory APIs.

### Phase P1: Cross-Platform Provider/Mode Type Expansion

**Outcome:** backend, mobile, and web can represent Gemma-local capabilities without breaking current paths.

- [ ] Expand provider enums in backend/mobile/web type contracts.
- [ ] Add mode selection state (`cloud`, `hybrid`, `local_offline`).
- [ ] Add strict availability checks and explicit error codes.

### Phase P2: Inference Routing Layer (Mobile + Web)

**Outcome:** single inference entrypoint per platform with deterministic routing and fallback rules.

- [ ] Implement platform router and adapters.
- [ ] Route existing chat calls through router abstraction.
- [ ] Keep non-chat API modules untouched initially to reduce migration risk.

### Phase P3: L0/L1 Orchestrator Contract Integration

**Outcome:** voice and text requests follow orchestrator-compatible gate, routing, and synthesis flow.

- [ ] Enforce noise/relevance/privacy gate before memory commit.
- [ ] Enforce sessionization metadata for all voice sessions.
- [ ] Enforce L1 query analysis and execution mode logging.

### Phase P4: Mobile Always-On Eva Speech Runtime

**Outcome:** mobile can run continuous listening and speech-to-speech orchestration with wake-word invocation.

- [ ] Connect wake-word trigger to orchestrator session open event.
- [ ] Implement barge-in interrupt behavior.
- [ ] Enforce activation/deactivation order contract.

### Phase P5: Web Speech-to-Speech Parity

**Outcome:** web supports voice interactions through the same orchestrator logic.

- [ ] Add web voice runtime panel and controls.
- [ ] Reuse event envelope and trace propagation.
- [ ] Implement web-specific fallback behavior under browser constraints.

### Phase P6: Local RAG Store and Ingestion Pipeline

**Outcome:** offline reasoning uses local memory and retrieval, not pure completion.

- [ ] Create local memory store and retriever contracts.
- [ ] Add ingestion from user turns and optional local docs.
- [ ] Add retention/compaction pipeline for device storage limits.

### Phase P7: Model Pack Control Plane

**Outcome:** artifact lifecycle can be managed safely before model activation.

- [ ] Define manifest schema and signing workflow.
- [ ] Define channels (`stable`, `candidate`, `canary`).
- [ ] Add rollback and revocation rules.

### Phase P8: Downloader and Verifier

**Outcome:** resilient and secure install lifecycle exists even before final large packs are used.

- [ ] Implement resumable chunked download.
- [ ] Implement chunk and full-package hash checks.
- [ ] Implement signature verification and install states.

### Phase P9: Native Gemma Runtime Bridge (Integration-Ready)

**Outcome:** app can call local runtime interface even if no model pack is active yet.

- [ ] Expose `loadModel`, `unloadModel`, `generate`, `health` bridge methods.
- [ ] Add streaming callback API for token stream compatibility.
- [ ] Integrate with EAS Development Build and plugin config.

### Phase P10: Hybrid/Cloud Coexistence and Policy

**Outcome:** existing production behavior remains stable while local mode is introduced.

- [ ] Keep existing backend cloud provider flows unchanged.
- [ ] Add explicit fallback policy (only when allowed).
- [ ] Add runtime indicators for mode, provider, and offline readiness.

### Phase P11: Quality, Safety, and Permission Hardening

**Outcome:** production safeguards enforced consistently in all modes.

- [ ] Enforce CRAG + Self-RAG and FLARE ceilings.
- [ ] Enforce permission check order on write/destructive actions.
- [ ] Enforce full audit logging and trace continuity.

### Phase P12: Observability and Telemetry

**Outcome:** operational visibility across mobile/web/backend runtime paths.

- [ ] Emit mode/provider/fallback metrics.
- [ ] Emit latency metrics (STT, orchestration, LLM, TTS).
- [ ] Emit error class metrics by component and mode.

### Phase P13: Validation and Release Gates

**Outcome:** deployment confidence before broad rollout.

- [ ] Run airplane-mode offline tests on mobile after pack install.
- [ ] Run correctness suite across cloud/hybrid/local_offline.
- [ ] Run performance and thermal gates on device matrix.

### Phase P14: Deferred Gemma Model Download and Activation (Later)

**Outcome:** complete local-offline runtime enabled on target capable devices.

- [ ] Download approved model pack on target device class.
- [ ] Verify and activate model pack.
- [ ] Run post-activation benchmark and quality checks.
- [ ] Switch selected users/devices from `hybrid` to `local_offline`.

---

## 13. Quality Gates (Must Pass)

- Existing Gemini/traditional behavior remains operational (no regression).
- Mobile Eva path supports continuous speech-to-speech sessions.
- Web voice path follows same orchestrator and trace contracts.
- Offline mode performs end-to-end without network after model installation.
- No unsigned/corrupt model can be activated.
- Memory retention and compaction prevent uncontrolled local growth.
- Quality loops and permission checks are enforced in runtime, not just documented.

---

## 14. Deferred Model Download Activation Runbook

### 14.1 Before Download

- Confirm profile recommendation (`int4`, `int8`, `fp16`) from device profiler.
- Confirm storage and thermal thresholds.
- Confirm manifest signature chain and release channel.

### 14.2 Download and Verify

- Download in resumable chunks.
- Verify chunk hashes during transfer.
- Verify full hash and signature before install state changes to `ready`.

### 14.3 Activate and Validate

- Load model through native bridge.
- Run health checks (`load`, `generate`, `memory`, `thermal`).
- Run fixed prompt suite and compare against baseline quality thresholds.

### 14.4 Rollback

- On health/quality failure, rollback to previous known-good model pack.
- If no local pack remains valid, fallback by policy to `hybrid` or `cloud` mode.

---

## 15. Risk Register and Mitigation

- Thermal throttling on mid-tier devices:
  - enforce profile gating
  - enforce runtime backpressure
- Storage pressure during updates:
  - staged unpack
  - rollback reserve checks
- Silent mode/provider fallback causing policy drift:
  - explicit fallback events + user policy checks
- Regression risk to existing Gemini runtime:
  - additive integration
  - compatibility gate in CI
- Browser limitations for always-on web audio:
  - explicit UX and capability-based behavior contracts

---

## 16. Immediate Next Actions (Start This Week)

- [ ] Freeze cross-platform runtime contracts (`mode`, `provider`, event envelope).
- [ ] Implement provider/mode type expansion in backend/mobile/web.
- [ ] Add inference routers with Gemini-first default and Gemma-local placeholders.
- [ ] Wire mobile Eva wake-word trigger path to orchestrator session events.
- [ ] Add model-pack manifest schema and verifier scaffolding.
- [ ] Add local RAG store interfaces and retention/compaction contract.
- [ ] Define and run no-regression test suite for existing ambient and chat paths.

---

## 17. Final Notes for Implementation Teams

- This plan is intentionally integration-first: build all production architecture now.
- Keep Gemini operational as the current runtime while Gemma-local readiness is built.
- Treat model download/activation as a controlled final switch, not a prerequisite for integration work.
- Mobile is the flagship always-on speech-to-speech surface; web receives orchestrator-aligned parity with platform-appropriate constraints.
