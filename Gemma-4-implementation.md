# Gemma 4 Offline Mobile Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a cross-platform mobile app that can run Gemma 4 inference locally with no internet dependency at runtime, while preserving optional cloud fallback and production safety.

**Architecture:** Keep the current backend-centric Cortex system for cloud and sync features, then add an on-device inference plane (MediaPipe/AI Edge runtime + local model packs + local retrieval store). Route chat/inference through a mode-aware adapter (`local_offline`, `hybrid`, `cloud`) selected by device capability and user preference.

**Tech Stack:** Expo/EAS development builds, React Native, native modules for MediaPipe AI Edge runtime, model pack CDN/object storage, signed manifests, resumable downloader, AsyncStorage/SQLite local state, existing FastAPI backend for optional services.

---

## 1. What This Plan Solves

- True offline inference after model pack install.
- Cross-platform packaging strategy for Android and iOS.
- Device-aware profile recommendation (INT4 default, INT8 mid, FP16 premium).
- Production-grade model delivery (resume, integrity check, rollback).
- Seamless coexistence with the existing backend APIs.

## 2. Current System Baseline (Repo-Grounded)

- Mobile is API-first and depends on backend endpoints for chat, RAG, memories, graph, ambient voice, and documents.
- Primary integration surface is [mobile/App.tsx](mobile/App.tsx), which calls the shared API client methods.
- API client base URL and network fallback behavior live in [mobile/shared/core/api/client.ts](mobile/shared/core/api/client.ts).
- Chat settings and provider type currently support only `"local" | "gemini"` in [mobile/shared/core/types.ts](mobile/shared/core/types.ts).
- Backend provider switching is implemented in [backend/server.py](backend/server.py) and [backend/src/llm/__init__.py](backend/src/llm/__init__.py).
- Backend currently exposes extensive `/api/*` surfaces; mobile can remain functional online while offline mode is incrementally added.

## 3. Critical Constraints and Decisions

- FP16 E2B (~9.6 GB weights) must be optional, not default.
- Consumer app-store first-launch with zero internet and FP16 bundled is not practical.
- Offline-first for mass users means model pack download once, then run without internet.
- "No internet ever" requires enterprise preload/sideload/OEM path.
- Expo Go is not sufficient for custom native inference runtime; use EAS Development Build and config plugins.

## 4. Target Runtime Modes

- `local_offline`: on-device model + local retrieval/data store, no network calls for inference.
- `hybrid`: local inference primary, backend optional for enrichment (docs/cloud sync) when online.
- `cloud`: existing backend/Gemini/local-server mode for low-end devices or users who skip model packs.

## 5. Model Profile Strategy

| Profile | Intended Users | Storage Budget (Practical) | Runtime Characteristics | Default |
|---|---|---:|---|---|
| `e2b-int4` | Most users | 4-6 GB | Fastest startup, lowest quality among tiers | Yes |
| `e2b-int8` | Mid/high devices | 7-10 GB | Better quality, moderate speed | Recommended when device qualifies |
| `e2b-fp16` | Flagship devices only | 12-16 GB+ | Highest quality, highest memory/thermal pressure | Optional premium |

### Capability Gating Inputs

- Total RAM.
- Available disk/storage.
- Device model class and SoC generation.
- Thermal trend and startup benchmark score.
- User consent for large download and Wi-Fi-only policy.

### Capability Gating Outcome

- Recommend one profile by default.
- Show optional higher/lower profiles with tradeoff summary.
- Block profiles that fail hard constraints.

## 6. Offline Product Behavior Contract

### What Works Fully Offline (Target)

- Local chat and reasoning.
- Local conversation memory and retrieval from device data.
- Local model profile switching among downloaded packs.

### What Remains Online-Optional

- Cloud documents pipeline and PageIndex features.
- Backend observability dashboards and server-side traces.
- Cloud sync and remote backup.

### First-Launch Internet Policy

- Consumer path: one-time download needed for model pack.
- Enterprise path: preload model pack in managed deployment image.
- Sideload path: import signed offline pack via file transfer.

## 7. Production Architecture

### 7.1 Control Plane

- Model manifest service (versioned JSON, signatures, checksums, profile metadata).
- Release channels (`stable`, `candidate`, `canary`).
- Revocation list for compromised model signatures.

### 7.2 Data Plane

- Chunked model artifact delivery from object storage + CDN.
- HTTP range requests for resume support.
- SHA-256 verification per chunk and whole package.
- Signature verification before activation.

### 7.3 On-Device Runtime

- Native MediaPipe/AI Edge bridge module.
- Local inference adapter in JS/TS layer.
- Download manager with pause/resume/cancel.
- Local state store for model registry and active profile.

### 7.4 Security

- Signed manifests and signed model package metadata.
- Integrity-first activation (never run unverified model files).
- Optional at-rest encryption for model artifacts and local vector DB.

## 8. Repository Change Plan (Exact Touchpoints)

### Existing Files To Modify

- [mobile/shared/core/types.ts](mobile/shared/core/types.ts)
- [mobile/shared/core/api/client.ts](mobile/shared/core/api/client.ts)
- [mobile/App.tsx](mobile/App.tsx)
- [mobile/app.json](mobile/app.json)
- [mobile/package.json](mobile/package.json)
- [backend/server.py](backend/server.py) (only for optional hybrid telemetry and manifest proxy endpoints)
- [backend/requirements.txt](backend/requirements.txt) (only if backend model pack metadata generation is hosted here)

### New Mobile Files To Create

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

### New Native Integration Files (Expo Development Build Path)

- `mobile/plugins/withGemmaRuntime.js`
- `mobile/modules/gemma-runtime/` (native bridge implementation)
- `mobile/android/` and `mobile/ios/` generated via `expo prebuild` for native module linking

### New Backend/Infra Planning Files

- `infra/modelpacks/manifest.schema.json`
- `infra/modelpacks/release-manifest.json` (generated per release)
- `scripts/modelpacks/build_modelpack.py`
- `scripts/modelpacks/sign_manifest.py`

## 9. Implementation Phases (Production Track)

### Phase P0: Architecture and Contract Freeze

**Outcome:** frozen interfaces and rollout contract before native work.

- [ ] Define inference mode contract (`local_offline`, `hybrid`, `cloud`) in `mobile/shared/core/inference/types.ts`.
- [ ] Define model profile metadata contract (size, min specs, checksum, signature, release channel).
- [ ] Define compatibility matrix for Android/iOS minimum versions.
- [ ] Write failure-state matrix (download failure, checksum mismatch, thermal throttle, low storage).
- [ ] Approve product policy for first-launch offline expectations.

### Phase P1: Mobile Inference Routing Layer

**Outcome:** one abstraction entry point for all chat/inference calls.

- [ ] Add router in `mobile/shared/core/inference/router.ts`.
- [ ] Implement cloud adapter that wraps existing API client behavior.
- [ ] Implement local adapter interface with placeholder native calls.
- [ ] Refactor [mobile/App.tsx](mobile/App.tsx) chat send/stream paths to call router instead of API directly.
- [ ] Keep existing API endpoints for non-chat modules to avoid full rewrite at once.

### Phase P2: Device Profiling and Profile Recommendation

**Outcome:** deterministic model recommendation based on device capability.

- [ ] Implement device profiler for RAM, free disk, model class, benchmark score.
- [ ] Add profile selector algorithm with hard constraints and safety margins.
- [ ] Add user consent UI for download policy (Wi-Fi-only default).
- [ ] Persist accepted profile and user overrides in local storage.
- [ ] Add telemetry counters for recommendation acceptance and failure reasons.

### Phase P3: Model Pack Pipeline (Build, Sign, Publish)

**Outcome:** production artifact pipeline for INT4/INT8/FP16 packs.

- [ ] Build conversion pipeline from Hugging Face weights to AI Edge runtime format (`.task` / runtime-compatible binaries).
- [ ] Generate three profile artifacts: `e2b-int4`, `e2b-int8`, `e2b-fp16`.
- [ ] Create signed release manifest with per-artifact SHA-256.
- [ ] Publish artifacts to object storage + CDN with range request enabled.
- [ ] Define rollback procedure to previous manifest version.

### Phase P4: Download Manager and Integrity Verification

**Outcome:** resilient download/install lifecycle for large model packs.

- [ ] Implement chunked resumable download with retries and backoff.
- [ ] Verify chunk hash and final package hash.
- [ ] Verify signature before model activation.
- [ ] Keep previous working model as rollback target.
- [ ] Add explicit install states: `not_installed`, `downloading`, `verifying`, `ready`, `failed`.

### Phase P5: Native Runtime Integration (MediaPipe/AI Edge)

**Outcome:** on-device inference available through React Native bridge.

- [ ] Add custom native module and expose `loadModel`, `unloadModel`, `generate`, `health` bridge methods.
- [ ] Wire plugin/config in [mobile/app.json](mobile/app.json) and EAS build profile.
- [ ] Move development/testing from Expo Go to EAS Development Build.
- [ ] Implement streaming token callback API for UI typing effect compatibility.
- [ ] Validate local inference for all profiles on representative devices.

### Phase P6: Local Retrieval and Memory Layer

**Outcome:** offline reasoning with local data, not only pure LLM completion.

- [ ] Create local memory store (SQLite-based) for conversations and retrieval corpus.
- [ ] Add ingestion path from user messages and optional imported docs.
- [ ] Add lightweight local retriever and context builder.
- [ ] Route chat prompt assembly through local retrieval in `local_offline` mode.
- [ ] Add retention and compaction strategy for device storage limits.

### Phase P7: Hybrid and Cloud Coexistence

**Outcome:** seamless fallback without breaking current backend product surface.

- [ ] Keep existing backend provider flows (`local`/`gemini`) unchanged for cloud mode.
- [ ] Add `offline_ready` and `active_model_profile` metadata in mobile status surfaces.
- [ ] Add explicit fallback policy (local fail -> cloud only when user allows internet).
- [ ] Keep backend-required modules (PageIndex, ambient cloud options) isolated from offline core.
- [ ] Add UI indicators for current mode and internet requirement.

### Phase P8: Test Matrix, Performance Gates, and Release

**Outcome:** production confidence before broad rollout.

- [ ] Build device matrix by RAM/storage tiers and OS versions.
- [ ] Run correctness tests against a fixed prompt suite across INT4/INT8/FP16.
- [ ] Run performance gates (TTFT, tokens/sec, peak RAM, thermal throttling, crash-free session).
- [ ] Run network-off test cases (airplane mode) for full local inference flows.
- [ ] Roll out with staged channels and kill-switch for problematic profile packs.

## 10. Quality Gates (Must Pass)

- Offline inference works in airplane mode after model install.
- No unsigned or corrupted model can be loaded.
- App startup remains stable if model missing or partially downloaded.
- Profile recommendation never suggests pack that violates hard constraints.
- Upgrade and rollback paths are deterministic and user-safe.

## 11. Deployment and Cost Guidance

- Do not rely on free GPU hosting for sustained production inference.
- Use cloud hosting only for model pack delivery (CDN/object storage), not required for runtime inference.
- Keep cloud inference as optional fallback, not a hard dependency.

## 12. "No Internet at All" Scenarios

### Consumer App-Store Path

- Realistic approach: first-time online model download, then fully offline runtime.

### Enterprise/OEM Path

- Preload model packs during device provisioning.
- Sideload signed model packs from offline media.
- Disable network requirement completely for inference mode.

## 13. Risk Register and Mitigation

- Thermal throttling on mid devices: enforce profile limits and runtime backpressure.
- Disk pressure during updates: staged unpack + rollback reserve checks.
- Conversion/runtime mismatch: version lock runtime + pack format in manifest.
- UX abandonment from large download: honest size/time estimates and resume support.
- Fragmentation across devices: capability database and staged rollout channels.

## 14. Execution Order Recommendation

1. P0 and P1 first to prevent architecture churn.
2. P2 and P4 next so user/device safety exists before native runtime scale-up.
3. P3 and P5 in parallel after contracts freeze.
4. P6 after local runtime is stable.
5. P7 and P8 for release hardening.

## 15. Immediate Next Actions (Start This Week)

- [ ] Freeze interfaces for inference routing and model profile metadata.
- [ ] Stand up model pack manifest format and signing pipeline.
- [ ] Add mobile capability profiling + recommendation UI skeleton.
- [ ] Create native runtime proof-of-concept with one quantized pack.
- [ ] Validate airplane mode offline chat on one Android and one iOS test device.
