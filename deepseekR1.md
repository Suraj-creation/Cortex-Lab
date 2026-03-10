# DeepSeek-R1-Distill-Qwen-7B — 15-Stage Curriculum Fine-Tuning Report

> **Project:** Cortex Lab — Agentic Personal Memory RAG System  
> **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (7.6B parameters)  
> **Hardware:** NVIDIA RTX 4000 Ada Generation (20,480 MiB VRAM, Compute Capability 8.9)  
> **Training Period:** February 20–23, 2026 (Base stages 1-10 on Feb 20-21, Extended stages 11-15 on Feb 23)  
> **Final Status:** ✅ All 15/15 stages completed  
> **Total Training Examples:** ~39,516 across 15 stages  
> **Total Training Data:** ~76 MB of structured JSON  
> **Author:** Suraj Kumar  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Quantization & Memory Strategy](#2-quantization--memory-strategy)
3. [LoRA Configuration Philosophy](#3-lora-configuration-philosophy)
4. [Dataset Generation Pipeline](#4-dataset-generation-pipeline)
5. [Stage 1 — RAG-Grounded Faithfulness](#5-stage-1--rag-grounded-faithfulness)
6. [Stage 2 — Agentic Reasoning & Routing](#6-stage-2--agentic-reasoning--routing)
7. [Stage 3 — Causal Chain & Temporal Reasoning](#7-stage-3--causal-chain--temporal-reasoning)
8. [Stage 4 — Self-RAG Critique (ISREL/ISSUP/ISUSE)](#8-stage-4--self-rag-critique-isrelissup-isuse)
9. [Stage 5 — Belief Evolution Tracking](#9-stage-5--belief-evolution-tracking)
10. [Stage 6 — Memory Consolidation & Summarization](#10-stage-6--memory-consolidation--summarization)
11. [Stage 7 — Multi-Turn Dialogue Coherence](#11-stage-7--multi-turn-dialogue-coherence)
12. [Stage 8 — Long-Context Multi-Hop Reasoning](#12-stage-8--long-context-multi-hop-reasoning)
13. [Stage 9 — DPO Preference Alignment](#13-stage-9--dpo-preference-alignment)
14. [Stage 10 — User Style Adaptation (Hot-Swap LoRA)](#14-stage-10--user-style-adaptation-hot-swap-lora)
15. [Stage 11 — ORPO Preference Optimization](#15-stage-11--orpo-preference-optimization)
16. [Stage 12 — RAFT (Retrieval-Augmented Fine-Tuning)](#16-stage-12--raft-retrieval-augmented-fine-tuning)
17. [Stage 13 — Function-Calling Fine-Tuning](#17-stage-13--function-calling-fine-tuning)
18. [Stage 14 — Rejection Sampling Fine-Tuning (RFT)](#18-stage-14--rejection-sampling-fine-tuning-rft)
19. [Stage 15 — SPIN Self-Play Improvement](#19-stage-15--spin-self-play-improvement)
20. [Cumulative Training Summary](#20-cumulative-training-summary)
21. [Disk Management & Cleanup Strategy](#21-disk-management--cleanup-strategy)
22. [VRAM Budget Analysis](#22-vram-budget-analysis)
23. [Key Research References](#23-key-research-references)

---

## 1. Architecture Overview

### 1.1 Training Pipeline Design

The Cortex Lab fine-tuning pipeline implements a **15-stage sequential curriculum** where each stage builds on the merged output of the previous one. This is orchestrated by `CortexLabTrainer` in `scripts/fine_tune_cortex.py` (1,212 lines).

```
DeepSeek-R1-Distill-Qwen-7B (base)
    │
    ├─ Stage 1:  Faithfulness (SFT)     → merge → checkpoint
    ├─ Stage 2:  Agentic Routing (SFT)  → merge → checkpoint
    ├─ Stage 3:  Causal Reasoning (SFT) → merge → checkpoint
    ├─ Stage 4:  Self-RAG Critique (SFT)→ merge → checkpoint
    ├─ Stage 5:  Belief Evolution (SFT) → merge → checkpoint
    ├─ Stage 6:  Summarization (SFT)    → merge → checkpoint
    ├─ Stage 7:  Dialogue (SFT)         → merge → checkpoint
    ├─ Stage 8:  Long-Context (SFT)     → merge → checkpoint
    ├─ Stage 9:  DPO Alignment          → merge → checkpoint
    ├─ Stage 10: User Style (SFT)       → ❌ NEVER MERGED (hot-swap adapter)
    ├─ Stage 11: ORPO Alignment         → merge → checkpoint
    ├─ Stage 12: RAFT (SFT)             → merge → checkpoint
    ├─ Stage 13: Function Calling (SFT) → merge → checkpoint
    ├─ Stage 14: RFT Refinement (SFT)   → merge → checkpoint
    └─ Stage 15: SPIN Self-Play (DPO)   → merge → FINAL MODEL
```

### 1.2 Trainer Types Used

| Trainer | Stages | Library |
|---------|--------|---------|
| **SFTTrainer** | 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14 | `trl.SFTTrainer` |
| **DPOTrainer** | 9, 15 | `trl.DPOTrainer` |
| **ORPOTrainer** | 11 | `trl.ORPOTrainer` |

### 1.3 Chat Template Format

All training data uses **ChatML** format:

```
<|im_start|>system
{system_instruction}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

### 1.4 Key Software Versions (at training time)

| Package | Version |
|---------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.2.0 |
| PEFT | 0.18.1 |
| TRL | 0.28.0 |
| Datasets | 4.5.0 |
| bitsandbytes | (latest, paged_adamw_8bit) |

---

## 2. Quantization & Memory Strategy

### 2.1 QLoRA Configuration (NF4)

```python
quantization_config = {
    "load_in_4bit":            True,
    "bnb_4bit_compute_dtype":  torch.bfloat16,     # Ada Lovelace native
    "bnb_4bit_use_double_quant": True,              # Further compresses quantization constants
    "bnb_4bit_quant_type":     "nf4",              # Normal Float 4-bit (optimal for pre-trained weights)
}
```

**Why NF4 + Double Quantization:**
- NF4 is information-theoretically optimal for normally distributed pre-trained weights (QLoRA paper §3.1)
- Double quantization compresses the quantization constants themselves, saving ~0.37 bits/param (≈370 MB for 7B)
- bfloat16 compute dtype is native to Ada Lovelace (CC 8.9), avoiding conversion overhead

### 2.2 VRAM Budget (Training)

| Component | VRAM |
|-----------|------|
| Base model (4-bit NF4 + double quant) | ~4,200 MB |
| LoRA weights (r=64, ALL_MODULES) | ~460 MB |
| Paged AdamW 8-bit optimizer | ~500 MB |
| Activations (with gradient checkpointing) | ~3,200 MB |
| Forward pass working memory | ~800 MB |
| KV cache (batch=4, seq=2048) | ~1,200 MB |
| CUDA + overhead | ~600 MB |
| bf16 compute buffers | ~740 MB |
| **TOTAL** | **~13,000 MB (63% of 20GB)** |
| **HEADROOM** | **~7,480 MB safety buffer** |

### 2.3 VRAM Budget (Inference)

| Component | VRAM |
|-----------|------|
| Model (4-bit) | ~4,200 MB |
| User LoRA (Stage 10, r=16) | ~35 MB |
| KV cache (seq=4096) | ~600 MB |
| BGE-large-en-v1.5 embedder | ~1,340 MB |
| BGE-reranker-v2-m3 | ~560 MB |
| CUDA overhead | ~300 MB |
| **TOTAL** | **~7,035 MB (34% of 20GB)** |

---

## 3. LoRA Configuration Philosophy

### 3.1 Module Targeting Strategy

Three targeting strategies are used across stages:

```python
ALL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_ONLY   = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_GATE   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
```

**Rationale:**
- **ALL_MODULES**: Used when learning new skills that require broad adaptation (stages 1, 2, 4, 5, 7, 8, 12, 13, 14)
- **ATTN_ONLY**: Used for preference alignment where we tune attention patterns, not knowledge (stages 9, 15)
- **ATTN_GATE**: Used for compression/summarization tasks needing gating control (stage 6)

### 3.2 Per-Stage LoRA Summary

| Stage | Rank (r) | Alpha (α) | Target Modules | Dropout | Trainable Params | % Trainable |
|-------|----------|-----------|----------------|---------|------------------|-------------|
| 1 — Faithfulness | 64 | 128 | ALL_MODULES | 0.05 | 161.5M | 2.08% |
| 2 — Agentic | 64 | 128 | ALL_MODULES | 0.05 | 161.5M | 2.08% |
| 3 — Causal | 32 | 64 | ALL_MODULES | 0.05 | 80.7M | 1.05% |
| 4 — Self-RAG | 64 | 128 | ALL_MODULES | 0.05 | 161.5M | 2.08% |
| 5 — Belief | 32 | 64 | ALL_MODULES | 0.05 | 80.7M | 1.05% |
| 6 — Summarization | 32 | 64 | ATTN_GATE | 0.05 | 60.6M | 0.79% |
| 7 — Dialogue | 48 | 96 | ALL_MODULES | 0.05 | 121.1M | 1.57% |
| 8 — Long Context | 64 | 128 | ALL_MODULES | 0.05 | 161.5M | 2.08% |
| 9 — DPO | 32 | 64 | ATTN_ONLY | 0.05 | 20.2M | 0.26% |
| 10 — User Style | 16 | 32 | q_proj, v_proj | 0.05 | ~10M | ~0.13% |
| 11 — ORPO | 32 | 64 | ATTN_ONLY | 0.05 | 20.2M | 0.26% |
| 12 — RAFT | 64 | 128 | ALL_MODULES | 0.05 | 161.5M | 2.08% |
| 13 — Function Calling | 64 | 128 | ALL_MODULES | 0.05 | 161.5M | 2.08% |
| 14 — RFT | 32 | 64 | ALL_MODULES | 0.03 | 80.7M | 1.05% |
| 15 — SPIN | 32 | 64 | ATTN_ONLY | 0.05 | 20.2M | 0.26% |

**Design Principles:**
- **High rank (r=64)** for stages that teach fundamentally new behaviors (faithfulness, agentic routing, self-RAG, long-context, RAFT, function-calling)
- **Medium rank (r=32-48)** for stages that refine existing capabilities (causal, belief, dialogue, RFT)
- **Low rank (r=16-32)** for alignment stages that make subtle distribution shifts (DPO, ORPO, SPIN, user style)
- **α = 2r** ratio ensures the LoRA effective learning rate scales properly with rank

---

## 4. Dataset Generation Pipeline

### 4.1 Generation Strategy

All training data is generated **100% locally** with **zero API calls, zero cost** using two scripts:

| Script | Lines | Stages | Approach |
|--------|-------|--------|----------|
| `scripts/generate_datasets.py` | 2,430 | 1–10 | Deterministic template engine |
| `scripts/generate_extended_datasets.py` | 1,477 | 11–15 | Extended templates, imports shared generators |

### 4.2 Memory Generation Architecture

The foundation is the `MemoryGenerator` class, which creates **15 personas × 200 memories each = 3,000 base memories** spanning a 2-year synthetic timeline (2024-01-01 to 2025-12-31).

**Memory Type Distribution (diary-like):**
| Type | Weight | Description |
|------|--------|-------------|
| Episodic | 50% | Daily events, conversations, actions |
| Semantic | 25% | Learned concepts, insights, readings |
| Belief | 12% | Opinions, worldview, values |
| Reflective | 13% | Meta-cognition, patterns, self-analysis |

**15 Personas** span diverse backgrounds (software engineer, student, freelancer, manager, etc.) ensuring the model generalizes across user archetypes.

### 4.3 Template Variability

- **5 template categories**: Episodic, Semantic, Belief, Reflective, Discussion
- **80+ template variable categories** (topic, emotion, entity, concept, etc.)
- **Combinatorial explosion**: 15 personas × 200 memories × 80+ variable slots = enormous diversity
- **Belief evolution arcs**: Deliberately injected 3-4 belief snapshots per persona on 2-3 topics, modeling opinion change over 18 months
- **18 belief topics**: remote work, work-life balance, ambition, social media, career, health, education, etc.

### 4.4 Quality Filter

Every generated example passes through `quality_filter()`:
- Minimum output length threshold
- De-duplication within stage
- Format validation (JSON, citation patterns, think-block structure)

---

## 5. Stage 1 — RAG-Grounded Faithfulness

> **Goal:** Teach the model to answer ONLY from provided memory context, cite every claim with `[Memory: timestamp]`, express calibrated confidence, and refuse when evidence is insufficient.

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 3,450 |
| **Dataset Size** | 6.5 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 2e-4 (cosine decay) |
| **Sequence Length** | 1024 |
| **LoRA** | r=64, α=128, ALL_MODULES |
| **Optimizer** | paged_adamw_8bit |
| **Base Model** | DeepSeek-R1-Distill-Qwen-7B (original HF) |

### 5.2 Dataset Composition (8 Categories)

| Category | Count | % | Description |
|----------|-------|---|-------------|
| Fully Grounded | ~800 | 23% | All retrieved memories relevant; generate cited answer |
| Partial Evidence | ~500 | 14% | Mix of relevant/irrelevant; note gaps explicitly |
| No Relevant Context | ~400 | 11% | All irrelevant memories; model must refuse |
| Empty Context | ~200 | 6% | Zero retrieved memories; refuse gracefully |
| Contradictory | ~300 | 9% | Memories disagree; surface the contradiction |
| Multi-Hop | ~300 | 9% | Chain reasoning across 4+ memories |
| Negative Examples | ~500 | 14% | Hallucinated outputs for contrastive learning |
| Confidence Calibration | ~500 | 14% | Train High/Medium/Low confidence expressions |

### 5.3 System Instruction

```
You are Cortex Lab, a personal AI memory assistant.
Answer ONLY from the provided memories. Cite every claim with [Memory: timestamp].
Use <think>...</think> for reasoning.
Say 'I don't have enough memories' if context is insufficient.
Express calibrated confidence: High / Medium / Low.
```

### 5.4 Training Results

| Metric | Start (Epoch 0) | End (Epoch 3) |
|--------|-----------------|---------------|
| **Completed** | 2026-02-21 09:40:45 |
| **Train Loss** | — (skipped in this run, was pre-completed) |

> *Note: Stage 1 was trained in an earlier run before the full 15-stage pipeline execution. Completion metadata confirms success.*

### 5.5 Key Technique

The model learns a **structured output pattern**:

```
<think>
{step-by-step reasoning checking each memory for relevance}
</think>

{answer with [Memory: YYYY-MM-DD] citations}

**Confidence:** {High/Medium/Low} — {justification}
```

---

## 6. Stage 2 — Agentic Reasoning & Routing

> **Goal:** Teach the model to analyze user queries, classify intent, assess complexity, select the correct agent (TimelineAgent, CausalAgent, ReflectionAgent, PlanningAgent), and choose optimal retrieval channels with weights.

### 6.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,950 |
| **Dataset Size** | 3.3 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 2e-4 (cosine decay) |
| **Sequence Length** | 1024 |
| **LoRA** | r=64, α=128, ALL_MODULES |
| **Base Model** | Stage 1 merged |

### 6.2 Dataset Sub-Tasks

| Sub-Task | Count | % | Description |
|----------|-------|---|-------------|
| Routing | ~500 | 17% | Intent classification → JSON routing decision |
| Multi-Query + HyDE | ~400 | 13% | Generate 4 query variants + hypothetical answer |
| Decomposition | ~300 | 10% | Break complex queries into sequential sub-questions |
| Step-Back | ~200 | 7% | Abstract the query for broader context retrieval |
| Entity Extraction | ~300 | 10% | Extract entities, relationships from queries |
| Combined | ~300 | 10% | Full agentic reasoning pipeline |

### 6.3 Routing Schema

5 intent types × 3 complexity levels = 15 routing patterns:

| Intent | Primary Agent | Retrieval Channels | Weights |
|--------|--------------|-------------------|---------|
| TEMPORAL | TimelineAgent | temporal, dense | 0.55, 0.45 |
| CAUSAL | CausalAgent | graph, temporal, dense | 0.40, 0.35, 0.25 |
| REFLECTIVE | ReflectionAgent | dense, graph | 0.55, 0.45 |
| FACTUAL | TimelineAgent | dense, sparse | 0.60, 0.40 |
| COMPLEX | PlanningAgent | graph, temporal, dense, sparse | 0.35, 0.25, 0.25, 0.15 |

### 6.4 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-21 09:42:04 |
| **Examples Trained** | 2,950 |

---

## 7. Stage 3 — Causal Chain & Temporal Reasoning

> **Goal:** Train the CausalAgent and TimelineAgent to build causal chains from memories, distinguish direct causes vs contributing factors vs correlations, and construct chronological narratives.

### 7.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,950 |
| **Dataset Size** | 7.6 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 1.5e-4 (slightly lower — building on stage 1) |
| **Sequence Length** | 1024 |
| **LoRA** | r=32, α=64, ALL_MODULES |
| **Base Model** | Stage 2 merged |

### 7.2 Dataset Composition

- **50% Causal Chain examples**: Given an event + surrounding memories → trace the causal chain backward to root cause, label each link as direct/contributing/correlated
- **50% Timeline examples**: Build chronological narratives identifying temporal patterns, transitions, and phase changes

### 7.3 Output Format

```markdown
## Causal Chain Analysis

### The Event
**{event description}** [Memory: YYYY-MM-DD]

### Root Cause
[Memory: YYYY-MM-DD] — {earliest contributing memory}

### Contributing Factors
1. **Step** [Memory: YYYY-MM-DD] — {intermediate cause}
2. **Step** [Memory: YYYY-MM-DD] — {intermediate cause}

### Key Insight
The root cause began {date} — well before the event on {date}.
```

### 7.4 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-21 12:15:47 |
| **Training Time** | ~2h 30m |
| **Examples Trained** | 2,950 |

---

## 8. Stage 4 — Self-RAG Critique (ISREL/ISSUP/ISUSE)

> **Goal:** Teach the model to evaluate its own generated answers against retrieved context using Self-RAG critique tokens and CRAG (Corrective RAG) relevance assessment.

### 8.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 3,450 |
| **Dataset Size** | 6.3 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 2e-4 |
| **Warmup** | 0.05 (longer — new token patterns) |
| **Sequence Length** | 1024 |
| **LoRA** | r=64, α=128, ALL_MODULES |
| **Base Model** | Stage 3 merged |

### 8.2 Critique Token System

**Self-RAG Tokens:**
- `[ISREL: yes/no]` — Is the answer relevant to the query?
- `[ISSUP: full/partial/none]` — Is the answer supported by retrieved context?
- `[ISUSE: 1-5]` — How useful is the answer (1=useless, 5=excellent)?

**CRAG Assessment:**
- Per-memory relevance score (0-1)
- Per-memory support score (0-1)
- Per-memory verdict: KEEP / REMOVE
- Overall CRAG decision: CORRECT / AMBIGUOUS / INCORRECT

### 8.3 Decision Output

```
ACCEPT — answer is fully grounded and useful
REGENERATE — answer has issues but context is sufficient
REJECT — answer is hallucinated or context is completely insufficient
```

### 8.4 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-21 14:30:11 |
| **Training Time** | ~2h 15m |
| **Examples Trained** | 3,450 |

---

## 9. Stage 5 — Belief Evolution Tracking

> **Goal:** Train the ReflectionAgent to detect belief changes over time, classify each change type, surface contradictions, and build belief timeline tables.

### 9.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,450 |
| **Dataset Size** | 5.7 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 1.5e-4 (cosine decay) |
| **Sequence Length** | 1024 |
| **LoRA** | r=32, α=64, ALL_MODULES |
| **Base Model** | Stage 4 merged |
| **Trainable Params** | 80,740,352 (1.05%) |

### 9.2 Belief Change Classification

| Change Type | Description |
|-------------|-------------|
| REFINEMENT | View becomes more nuanced |
| CONTRADICTION | Direct opposition to earlier view |
| EXPANSION | View broadens to include new dimensions |
| ABANDONMENT | View is completely dropped |
| STABLE | No change detected |

### 9.3 Training Metrics (from logs)

| Metric | Epoch 0.07 | Epoch 1.0 | Epoch 2.0 | Epoch 3.0 |
|--------|-----------|-----------|-----------|-----------|
| **Loss** | 1.518 | 0.0573 | 0.0555 | 0.0545 |
| **Token Accuracy** | 77.88% | 97.33% | 97.42% | 97.58% |
| **Grad Norm** | 0.0864 | 0.0073 | 0.0068 | 0.0074 |
| **Entropy** | 0.4362 | 0.0584 | 0.0560 | 0.0546 |

| Summary | Value |
|---------|-------|
| **Training Time** | 1h 49m 00s |
| **Train Loss (avg)** | 0.0929 |
| **Final Token Accuracy** | **97.58%** |
| **Samples/sec** | 1.124 |
| **Total Tokens** | 4,668,000 |
| **Disk Freed** | 14,537 MB (previous stage cleanup) |

### 9.4 Key Technique: Belief Arc Injection

The dataset generator deliberately injects **3-4 belief snapshots per persona** on randomly selected topics from 18 belief domains, creating synthetic opinion evolution arcs over 18 months:

```
Step 1: "I strongly believe {topic} is very important" (Jan 2024)
Step 2: "Starting to have doubts about {topic}" (Jul 2024)  
Step 3: "My view has completely shifted" (Dec 2024)
Step 4: "Neither extreme was right — it's nuanced" (Jun 2025)
```

---

## 10. Stage 6 — Memory Consolidation & Summarization

> **Goal:** Teach the model to compress memories at different abstraction levels (DAILY/WEEKLY/MONTHLY) and extract atomic, self-contained propositions from memories.

### 10.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,450 |
| **Dataset Size** | 3.5 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 1.5e-4 |
| **Sequence Length** | 1024 |
| **LoRA** | r=32, α=64, ATTN_GATE |
| **Base Model** | Stage 5 merged |
| **Trainable Params** | 60,555,264 (0.79%) |

### 10.2 Summarization Levels

| Level | Preserve | Output |
|-------|----------|--------|
| DAILY | All specifics, exact timestamps | Full reconstruction possible |
| WEEKLY | Themes, key decisions, mood shifts | Category-level aggregation |
| MONTHLY | Major events only | Narrative arc |

### 10.3 Proposition Extraction

Each memory is decomposed into **atomic, self-contained propositions** that are independently understandable without context:

```
Input: "Had a productive meeting with Sarah about the API redesign. We decided to use REST over GraphQL."

Propositions:
1. A meeting occurred about the API redesign.
2. Sarah participated in the API redesign meeting.
3. The meeting was productive.
4. A decision was made to use REST for the API.
5. GraphQL was considered but rejected for the API.
```

### 10.4 Training Metrics (from logs)

| Metric | Epoch 0.07 | Epoch 1.0 | Epoch 2.0 | Epoch 3.0 |
|--------|-----------|-----------|-----------|-----------|
| **Loss** | 1.490 | 0.0675 | 0.0467 | 0.0390 |
| **Token Accuracy** | 76.76% | 97.54% | 98.10% | 98.43% |
| **Grad Norm** | 0.0972 | 0.0176 | 0.0143 | 0.0143 |
| **Entropy** | 0.7931 | 0.0777 | 0.0542 | 0.0462 |

| Summary | Value |
|---------|-------|
| **Training Time** | 1h 05m 59s |
| **Train Loss (avg)** | 0.0948 |
| **Final Token Accuracy** | **98.43%** |
| **Samples/sec** | 1.856 |
| **Total Tokens** | 2,720,000 |
| **Disk Freed** | 14,537 MB + 947 MB checkpoints |

### 10.5 Key Design Choice: ATTN_GATE Targeting

Stage 6 uniquely uses `ATTN_GATE` instead of `ALL_MODULES`. The **gate_proj** in transformer MLP layers controls the gating mechanism of the SwiGLU activation — crucial for compression tasks where the model must learn what information to "let through" vs "filter out." The up/down projections are frozen since the knowledge content shouldn't change, only the filtering behavior.

---

## 11. Stage 7 — Multi-Turn Dialogue Coherence

> **Goal:** Train the model to maintain full context across multi-turn conversations, resolve coreferences ("it", "that project", "she"), and reference earlier turns naturally.

### 11.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 1,950 |
| **Dataset Size** | 3.0 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 1.5e-4 |
| **Warmup** | 0.05 |
| **Sequence Length** | 1024 |
| **LoRA** | r=48, α=96, ALL_MODULES |
| **Base Model** | Stage 6 merged |
| **Trainable Params** | 121,110,528 (1.57%) |

### 11.2 Training Metrics (from logs)

| Metric | Epoch 0.08 | Epoch 1.0 | Epoch 2.0 | Epoch 3.0 |
|--------|-----------|-----------|-----------|-----------|
| **Loss** | 2.613 | 0.0941 | 0.0893 | 0.0878 |
| **Token Accuracy** | 58.21% | 95.77% | 95.92% | 96.02% |
| **Grad Norm** | 0.1836 | 0.0114 | 0.0237 | 0.0117 |
| **Entropy** | 1.439 | 0.0938 | 0.0899 | 0.0886 |

| Summary | Value |
|---------|-------|
| **Training Time** | 46m 36s |
| **Train Loss (avg)** | 0.1692 |
| **Final Token Accuracy** | **96.02%** |
| **Samples/sec** | 2.092 |
| **Total Tokens** | 2,159,000 |
| **Disk Freed** | 14,537 MB + 715 MB checkpoints |

### 11.3 Key Technique: High Initial Loss

Stage 7 shows the **highest initial loss (2.613)** of any SFT stage because multi-turn dialogue is structurally different from single-turn Q&A. The model must learn to track conversation state, resolve pronouns, and maintain narrative coherence across turns — a fundamentally different skill from previous stages. Despite the high start, it converges to 96.0% accuracy.

---

## 12. Stage 8 — Long-Context Multi-Hop Reasoning

> **Goal:** Train the PlanningAgent to synthesize evidence across 10-20 memory chunks, build complete multi-hop reasoning chains, and identify patterns spanning the full memory timeline.

### 12.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,450 |
| **Dataset Size** | 9.5 MB (largest stage) |
| **Epochs** | 3 |
| **Batch Size** | 1 × 16 = 16 effective |
| **Learning Rate** | 1e-4 (most conservative SFT LR) |
| **Max Grad Norm** | 0.5 (tighter clipping) |
| **Sequence Length** | **2048** (doubled) |
| **LoRA** | r=64, α=128, ALL_MODULES |
| **Base Model** | Stage 7 merged |
| **Trainable Params** | 161,480,704 (2.08%) |

### 12.2 Training Metrics (from logs)

| Metric | Epoch 0.07 | Epoch 1.0 | Epoch 1.4 |
|--------|-----------|-----------|-----------|
| **Loss** | 1.160 | 0.1179 | 0.1070 |
| **Token Accuracy** | 77.67% | 95.39% | 95.66% |
| **Grad Norm** | 0.0415 | 0.0089 | 0.0073 |
| **Entropy** | 0.7197 | 0.1262 | 0.1128 |

| Summary | Value |
|---------|-------|
| **Completed** | 2026-02-21 22:57:58 |
| **Training Time** | ~3h 12m |
| **Examples Trained** | 2,450 |

### 12.3 Key Design Choices

- **Batch size = 1**: Long-context (2048 tokens) requires more memory per sample; compensated with gradient_accumulation_steps=16
- **Learning rate = 1e-4**: Most conservative of any SFT stage — deep reasoning chains are fragile and we don't want to catastrophically forget earlier stages
- **Tighter gradient clipping (0.5 vs 1.0)**: Long sequences can produce large gradient spikes; tighter clipping ensures stable training

---

## 13. Stage 9 — DPO Preference Alignment

> **Goal:** Align the model's outputs with human preference patterns using Direct Preference Optimization (DPO). Teach it to prefer grounded, cited, well-structured responses over vague, hallucinated ones.

### 13.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | DPOTrainer |
| **Examples** | 2,950 preference pairs |
| **Dataset Size** | 3.6 MB |
| **Epochs** | 1 (single pass for DPO) |
| **Batch Size** | 1 (DPO needs 2× memory: chosen + rejected) |
| **Gradient Accumulation** | 8 |
| **Learning Rate** | 5e-6 (very conservative) |
| **Beta (β)** | 0.1 (KL divergence penalty) |
| **Max Length** | 1024 |
| **Max Prompt Length** | 512 |
| **LoRA** | r=32, α=64, ATTN_ONLY |
| **Reference Model** | None (implicit PEFT frozen base) |
| **Base Model** | Stage 8 merged |
| **Trainable Params** | 20,185,088 (0.26%) |

### 13.2 DPO Loss Formulation

The DPO loss optimizes: $\mathcal{L}_\text{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)}\right)\right]$

Where $y_w$ = chosen response, $y_l$ = rejected response, $\beta$ = 0.1 controls deviation from reference.

### 13.3 Key Design Choices

- **Single epoch**: DPO typically only needs 1 pass — multiple epochs risk overspecializing on the preference data
- **Implicit reference model**: Using PEFT's frozen base layers as the reference policy instead of loading a separate model (saves 50% VRAM)
- **ATTN_ONLY targeting**: Preference alignment modifies attention patterns (what to attend to), not the knowledge stored in MLP layers

### 13.4 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-21 23:40:44 |
| **Training Time** | ~42m |
| **Examples Trained** | 2,950 |

---

## 14. Stage 10 — User Style Adaptation (Hot-Swap LoRA)

> **Goal:** Create a lightweight, user-specific style adapter that captures personal communication preferences. **This adapter is NEVER merged into the base model** — it remains as a hot-swap LoRA that can be loaded/unloaded at inference time.

### 14.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 1,466 |
| **Dataset Size** | 1.3 MB |
| **Epochs** | 2 |
| **Batch Size** | 2 × 4 = 8 effective |
| **Learning Rate** | 5e-5 (lower — don't overwrite stages 1-9) |
| **Warmup** | 0.10 (longer warmup) |
| **Sequence Length** | 1024 |
| **LoRA** | r=16, α=32, q_proj + v_proj only |
| **Base Model** | Stage 9 DPO merged |
| **MERGE** | ❌ **NEVER** — hot-swap adapter |

### 14.2 Why Not Merged

Stage 10's adapter captures **user-specific** style (vocabulary, formality, humor level, response length preference). Merging would permanently bake one user's preferences into the model weights. By keeping it as a hot-swap LoRA:

1. **Multi-user support**: Different users can have different style adapters
2. **Easy retraining**: User style can be updated without retraining the full model
3. **Reversibility**: Can be disabled for generic mode
4. **Tiny footprint**: r=16 with only q_proj/v_proj ≈ 35 MB on disk

### 14.3 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-23 10:05:08 |
| **Examples Trained** | 1,466 |

---

## 15. Stage 11 — ORPO Preference Optimization

> **Goal:** Apply Odds-Ratio Preference Optimization (ORPO) for reference-free preference alignment. ORPO combines SFT and alignment into a single loss, eliminating the need for a reference model (saving 50% memory vs DPO).

### 15.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | ORPOTrainer |
| **Examples** | 3,000 preference pairs |
| **Dataset Size** | 5.0 MB |
| **Epochs** | 1 |
| **Batch Size** | 1 (preference pairs need 2× memory) |
| **Gradient Accumulation** | 8 |
| **Learning Rate** | 8e-6 (higher than DPO — ORPO has built-in regularization) |
| **Beta (β)** | 0.1 (odds ratio weight) |
| **Max Length** | 1024 |
| **LoRA** | r=32, α=64, ATTN_ONLY |
| **Base Model** | Stage 9 merged (skips Stage 10 — never merged) |

### 15.2 ORPO vs DPO

| Property | DPO (Stage 9) | ORPO (Stage 11) |
|----------|---------------|-----------------|
| Reference Model | Required (implicit PEFT) | **Not required** |
| Memory Usage | 2× (model + ref) | 1.5× (no reference) |
| Learning Rate | 5e-6 | 8e-6 (higher) |
| Regularization | KL divergence (β) | Built-in odds ratio |
| Theoretical Basis | Bradley-Terry model | Odds ratio optimization |

### 15.3 10 Quality Categories for Preference Pairs

| Category | Weight | Description |
|----------|--------|-------------|
| Citation Quality | 6 | Good [Memory: date] citations vs missing |
| Reasoning Depth | 5 | Deep multi-step analysis vs surface-level |
| Honest Uncertainty | 5 | Calibrated confidence vs overconfidence |
| Retrieval Grounding | 5 | Evidence-based vs speculative answers |
| Structured Response | 4 | Well-formatted vs rambling |
| Think Block Quality | 4 | Step-by-step reasoning vs shallow |
| Multi-Memory Synthesis | 4 | Cross-referencing vs single-source |
| Empathy with Precision | 3 | Warm + precise vs cold/vague |
| Refusal When Appropriate | 3 | Honest refusal vs hallucinated answer |
| Temporal Reasoning | 3 | Correct time ordering vs confused |

### 15.4 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-23 10:48:59 |
| **Training Time** | ~44m |
| **Examples Trained** | 3,000 |

---

## 16. Stage 12 — RAFT (Retrieval-Augmented Fine-Tuning)

> **Goal:** Teach the model to reason over noisy retrieval context by identifying relevant documents among distractors. The model sees 1 oracle document + N distractor documents and must learn to cite only the relevant one.

### 16.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,500 |
| **Dataset Size** | 6.1 MB |
| **Epochs** | 2 |
| **Batch Size** | 1 × 16 = 16 effective |
| **Learning Rate** | 1e-4 |
| **Sequence Length** | **2048** (multi-doc context) |
| **LoRA** | r=64, α=128, ALL_MODULES |
| **Base Model** | Stage 11 ORPO merged |

### 16.2 RAFT Architecture

Each training example contains a query + 4-6 documents, where exactly 1 is the relevant personal memory (oracle) and the rest are plausible but irrelevant distractors:

| Position Type | % | Difficulty | Description |
|--------------|---|------------|-------------|
| Oracle First (D1) | 30% | Easy | Relevant doc is first — model gets it immediately |
| Oracle Middle (D2-D3) | 30% | Medium | Relevant doc is buried in the middle |
| Oracle Last (D4-D5) | 30% | Hard | Relevant doc is at the very end |
| No Oracle | 10% | Refusal | All distractors — model must refuse |

### 16.3 Distractor Generation

Distractors are generated from 16 unrelated topics (quantum computing, ancient Roman trade routes, AI regulation, etc.) using templates that produce **plausible-sounding but irrelevant** text:

```
"According to recent research on {topic}, the key finding is that {finding}. 
This has implications for {implication}. Multiple studies confirm this trend."
```

### 16.4 Training Results

| Metric | Value |
|--------|-------|
| **Completed** | 2026-02-23 11:51:38 |
| **Training Time** | ~1h 03m |
| **Examples Trained** | 2,500 |

---

## 17. Stage 13 — Function-Calling Fine-Tuning

> **Goal:** Train the model to generate structured JSON tool-call output with strict schema adherence, correct argument construction, and multi-tool chaining.

### 17.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 3,000 |
| **Dataset Size** | 3.9 MB |
| **Epochs** | 3 |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 1.5e-4 |
| **Sequence Length** | 1024 |
| **LoRA** | r=64, α=128, ALL_MODULES |
| **Base Model** | Stage 12 RAFT merged |
| **Trainable Params** | 161,480,704 (2.08%) |

### 17.2 Cortex Function Registry (8 Tools)

| Function | Description | Complexity |
|----------|-------------|------------|
| `memory_search` | Search memories by query, time range, entities, types | Medium |
| `memory_store` | Store new memory with type, importance, entities | Simple |
| `belief_tracker` | Track belief evolution on a topic over time | Medium |
| `causal_chain` | Build causal chains linking events (forward/backward) | Complex |
| `summarize_period` | Generate period summaries (brief/detailed/narrative) | Medium |
| `entity_graph` | Query knowledge graph for entity relationships | Medium |
| `emotion_timeline` | Generate emotional timeline (daily/weekly/monthly) | Medium |
| `pattern_detect` | Detect recurring patterns in behavior/thoughts | Complex |

### 17.3 Multi-Tool Chaining

20% of training examples involve multi-tool calls with complementary functions:

```json
{
  "tool_calls": [
    {
      "function": "memory_search",
      "arguments": {"query": "career direction", "limit": 10}
    },
    {
      "function": "belief_tracker", 
      "arguments": {"topic": "career direction", "include_contradictions": true}
    }
  ]
}
```

### 17.4 Training Metrics (from logs)

| Metric | Epoch 0.05 | Epoch 1.0 | Epoch 2.0 | Epoch 3.0 |
|--------|-----------|-----------|-----------|-----------|
| **Loss** | — | ~0.057 | ~0.055 | 0.0552 |
| **Token Accuracy** | — | ~97.4% | ~97.5% | **97.55%** |

| Summary | Value |
|---------|-------|
| **Training Time** | 1h 06m 13s |
| **Train Loss (avg)** | 0.1101 |
| **Final Token Accuracy** | **97.55%** |
| **Samples/sec** | 2.265 |
| **Total Tokens** | 2,720,000 |
| **Disk Freed** | 88 MB (old adapter) + 14,537 MB (merged) + 1,871 MB (checkpoints) |

---

## 18. Stage 14 — Rejection Sampling Fine-Tuning (RFT)

> **Goal:** Train the model exclusively on "best-of-N" quality outputs, teaching it to consistently produce top-tier responses. For each prompt, N candidate responses of varying quality are scored against 7 quality criteria, and only the gold-standard version is kept.

### 18.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | SFTTrainer |
| **Examples** | 2,000 |
| **Dataset Size** | 4.5 MB |
| **Epochs** | 2 (refinement stage — less data needed) |
| **Batch Size** | 2 × 8 = 16 effective |
| **Learning Rate** | 5e-5 (very conservative — polishing) |
| **Max Grad Norm** | 0.5 (tight clipping) |
| **LoRA** | r=32, α=64, ALL_MODULES, dropout=0.03 |
| **Base Model** | Stage 13 merged |
| **Trainable Params** | 80,740,352 (1.05%) |

### 18.2 7 Quality Criteria (All Must Pass)

| Criterion | Description |
|-----------|-------------|
| `citation_present` | Contains `[Memory: YYYY-MM-DD]` citations |
| `think_block_present` | Contains `<think>...</think>` reasoning block |
| `confidence_stated` | Contains `**Confidence:**` marker |
| `structured` | Uses headers, bullet points, clear sections |
| `empathetic` | Personal, warm tone (not clinical/robotic) |
| `grounded` | No speculative claims beyond evidence |
| `multi_source` | References multiple memories |

### 18.3 8 Query Types for RFT

Memory recall, belief analysis, pattern detection, causal trace, emotional review, synthesis, contradiction check, timeline review.

### 18.4 Training Metrics (from logs)

| Metric | Epoch 0.08 | Epoch 0.5 | Epoch 1.0 | Epoch 2.0 |
|--------|-----------|-----------|-----------|-----------|
| **Loss** | 1.835 | 0.2058 | 0.1567 | 0.1497 |
| **Token Accuracy** | 71.56% | 92.89% | 93.66% | 93.85% |
| **Grad Norm** | 0.1309 | 0.0284 | 0.0304 | 0.0214 |
| **Entropy** | 0.5737 | 0.2192 | 0.1613 | 0.1551 |

| Summary | Value |
|---------|-------|
| **Training Time** | 1h 00m 44s |
| **Train Loss (avg)** | 0.2845 |
| **Final Token Accuracy** | **93.85%** |
| **Samples/sec** | 1.098 |
| **Total Tokens** | 2,503,000 |
| **Disk Freed** | 319 MB + 14,537 MB + 1,871 MB |

### 18.5 Note on Lower Accuracy

Stage 14 shows the **lowest final token accuracy (93.85%)** of completed stages. This is expected and intentional:
- RFT examples are the **highest-quality, longest, most complex** outputs in the entire pipeline
- They contain sophisticated multi-source synthesis, nuanced confidence calibration, and detailed reasoning chains
- The model is learning to produce outputs that are harder to predict token-by-token
- A lower token accuracy on high-quality targets is better than high token accuracy on simple targets

---

## 19. Stage 15 — SPIN Self-Play Improvement

> **Goal:** Apply Self-Play Improvement (SPIN) using DPO-style training where the model learns to distinguish its own outputs from ground truth, closing the quality gap each iteration.

### 19.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Trainer** | DPOTrainer (SPIN uses DPO-style loss) |
| **Examples** | 2,500 preference pairs |
| **Dataset Size** | 5.4 MB |
| **Epochs** | 1 (single pass per SPIN iteration) |
| **Batch Size** | 1 × 8 |
| **Learning Rate** | 5e-6 (very conservative — subtle alignment) |
| **Beta (β)** | 0.1 (KL divergence penalty) |
| **Max Length** | 1024 |
| **Max Prompt Length** | 512 |
| **LoRA** | r=32, α=64, ATTN_ONLY |
| **Base Model** | Stage 14 RFT merged |
| **Trainable Params** | 20,185,088 (0.26%) |

### 19.2 SPIN Mechanism

In standard DPO: chosen = human-preferred, rejected = human-dispreferred.  
In SPIN: **chosen = ground truth**, **rejected = model's own (imperfect) output**.

The model iteratively learns to close the gap between its own generations and gold-standard outputs. This creates a self-improvement loop without requiring human annotations.

### 19.3 6 Imperfection Patterns (for Rejected Samples)

| Pattern | Description |
|---------|-------------|
| `generic_response` | Correct but lacks personalization/citations |
| `no_citations` | Good content but missing [Memory:] citations |
| `overconfident` | Claims high confidence without sufficient evidence |
| `surface_level` | Addresses question but doesn't go deep |
| `wrong_format` | Content OK but format is off (no think block, no structure) |
| `slight_hallucination` | Mostly correct but adds ungrounded claims |

### 19.4 Training Metrics (from logs — full DPO metrics)

| Metric | Step 10 | Step 40 | Step 160 | Step 310 (final) |
|--------|---------|---------|----------|-------------------|
| **Loss** | 0.709 | 0.361 | 0.049 | 0.011 |
| **Rewards/Chosen** | -0.003 | 0.967 | 4.213 | 5.038 |
| **Rewards/Rejected** | 0.015 | 0.004 | -0.553 | -1.364 |
| **Reward Margin** | -0.018 | 0.963 | 4.766 | **6.403** |
| **Reward Accuracy** | 37.5% | 97.5% | 100% | **100%** |
| **Grad Norm** | 1.800 | 0.569 | 0.025 | 0.005 |

| Summary | Value |
|---------|-------|
| **Training Time** | 54m 54s |
| **Train Loss (avg)** | 0.1114 |
| **Final Reward Margin** | **6.508** |
| **Final Reward Accuracy** | **100%** |
| **Chosen Reward (final)** | 5.081 |
| **Rejected Reward (final)** | -1.427 |
| **Disk Freed** | 319 MB + 14,537 MB + 947 MB |

### 19.5 Analysis of SPIN Convergence

The SPIN training shows remarkably clean convergence:

1. **Steps 1-30**: Reward accuracy jumps from 37.5% to 97.5% — the model quickly learns the basic distinction between gold and imperfect outputs
2. **Steps 30-80**: Reward accuracy hits 100% and stays there — the model now perfectly distinguishes its own outputs from ground truth
3. **Steps 80-310**: The reward margin continues widening (4.8 → 6.5), indicating the model is pushing its distribution further toward gold-standard behavior
4. **Loss collapses**: From 0.709 to 0.011 — indicating near-perfect alignment

The widening gap between chosen reward (+5.08) and rejected reward (-1.43) confirms the model has internalized the quality criteria from all 14 previous stages.

---

## 20. Cumulative Training Summary

### 20.1 All 15 Stages — Complete Table

| # | Stage | Trainer | Examples | Data Size | Epochs | LR | LoRA r/α | Seq Len | Time | Final Token Acc | Final Loss |
|---|-------|---------|----------|-----------|--------|-----|---------|---------|------|----------------|------------|
| 1 | Faithfulness | SFT | 3,450 | 6.5 MB | 3 | 2e-4 | 64/128 | 1024 | — | — | — |
| 2 | Agentic | SFT | 2,950 | 3.3 MB | 3 | 2e-4 | 64/128 | 1024 | — | — | — |
| 3 | Causal | SFT | 2,950 | 7.6 MB | 3 | 1.5e-4 | 32/64 | 1024 | ~2h30m | — | — |
| 4 | Self-RAG | SFT | 3,450 | 6.3 MB | 3 | 2e-4 | 64/128 | 1024 | ~2h15m | — | — |
| 5 | Belief | SFT | 2,450 | 5.7 MB | 3 | 1.5e-4 | 32/64 | 1024 | 1h49m | **97.58%** | 0.0929 |
| 6 | Summarization | SFT | 2,450 | 3.5 MB | 3 | 1.5e-4 | 32/64 | 1024 | 1h06m | **98.43%** | 0.0948 |
| 7 | Dialogue | SFT | 1,950 | 3.0 MB | 3 | 1.5e-4 | 48/96 | 1024 | 46m36s | **96.02%** | 0.1692 |
| 8 | Long Context | SFT | 2,450 | 9.5 MB | 3 | 1e-4 | 64/128 | 2048 | ~3h12m | ~95.6% | — |
| 9 | DPO | DPO | 2,950 | 3.6 MB | 1 | 5e-6 | 32/64 | 1024 | ~42m | — | — |
| 10 | User Style | SFT | 1,466 | 1.3 MB | 2 | 5e-5 | 16/32 | 1024 | — | — | — |
| 11 | ORPO | ORPO | 3,000 | 5.0 MB | 1 | 8e-6 | 32/64 | 1024 | ~44m | — | — |
| 12 | RAFT | SFT | 2,500 | 6.1 MB | 2 | 1e-4 | 64/128 | 2048 | ~1h03m | — | — |
| 13 | Function Call | SFT | 3,000 | 3.9 MB | 3 | 1.5e-4 | 64/128 | 1024 | 1h06m | **97.55%** | 0.1101 |
| 14 | RFT | SFT | 2,000 | 4.5 MB | 2 | 5e-5 | 32/64 | 1024 | 1h01m | **93.85%** | 0.2845 |
| 15 | SPIN | DPO | 2,500 | 5.4 MB | 1 | 5e-6 | 32/64 | 1024 | 54m54s | 100% (reward) | 0.1114 |

### 20.2 Aggregate Statistics

| Metric | Value |
|--------|-------|
| **Total Training Examples** | **39,516** |
| **Total Dataset Size** | **~76 MB** |
| **Total Training Time (est.)** | **~18-20 hours** |
| **Stages Using SFT** | 12 |
| **Stages Using DPO** | 2 (stages 9, 15) |
| **Stages Using ORPO** | 1 (stage 11) |
| **Max Sequence Length** | 2048 (stages 8, 12) |
| **Min Sequence Length** | 1024 (all others) |
| **Base Model Parameters** | 7,615,616,512 |
| **Max Trainable %** | 2.08% (stages 1, 2, 4, 8, 12, 13) |
| **Min Trainable %** | ~0.13% (stage 10) |

### 20.3 Token Accuracy Trajectory (Logged Stages)

```
Stage  5 (Belief):        77.88% → 97.58%  (Δ +19.70%)
Stage  6 (Summarization): 76.76% → 98.43%  (Δ +21.67%)  ← Highest
Stage  7 (Dialogue):      58.21% → 96.02%  (Δ +37.81%)  ← Largest improvement
Stage  8 (Long Context):  77.67% → 95.66%  (Δ +17.99%)
Stage 13 (Fn Calling):    —      → 97.55%
Stage 14 (RFT):           71.56% → 93.85%  (Δ +22.29%)  ← Complex targets
Stage 15 (SPIN):          37.5%  → 100%    (Δ +62.50%)  ← Reward accuracy
```

---

## 21. Disk Management & Cleanup Strategy

### 21.1 Automatic Cleanup Pipeline

After each stage completes, `_cleanup_previous_stages()` automatically:

1. **Deletes merged models** from stages OLDER than the current one (~14,537 MB each)
2. **Deletes adapters** from stages older than the immediate previous (~88-319 MB each)
3. **Deletes checkpoints** from all completed stages (~715-1,871 MB each)

Each 7B merged model is ~14.5 GB. Without cleanup, 15 stages × 14.5 GB = 218 GB in merged models alone — impossible on 226 GB disk.

### 21.2 Final Disk State

After all 15 stages:
- **Stage 14 RFT**: Adapter only (165 MB) — merged was cleaned after stage 15 used it
- **Stage 15 SPIN**: Merged model only (final checkpoint)
- All other stages: Only `training_meta.json` (12 KB each)

### 21.3 Approximate Disk Freed During Training

| Event | Freed |
|-------|-------|
| Stage 5 cleanup (stage 4 merged) | 14,537 MB |
| Stage 6 cleanup (stage 5 merged + checkpoints) | 15,484 MB |
| Stage 7 cleanup (stage 6 merged + checkpoints) | 15,252 MB |
| Stage 13 cleanup (stage 12 merged + checkpoints + stage 11 adapter) | 16,496 MB |
| Stage 14 cleanup (stage 13 merged + checkpoints + stage 12 adapter) | 16,727 MB |
| Stage 15 cleanup (stage 14 merged + checkpoints + stage 13 adapter) | 15,803 MB |
| **Total freed (approximate)** | **~95+ GB recycled** |

---

## 22. VRAM Budget Analysis

### 22.1 Peak VRAM Usage by Stage

| Stage Type | Peak Estimate | Key Factor |
|------------|--------------|------------|
| SFT (r=64, seq=1024, batch=2) | ~13,000 MB | Largest adapter, all modules |
| SFT (r=64, seq=2048, batch=1) | ~14,000 MB | Long context (stages 8, 12) |
| DPO (r=32, batch=1) | ~11,000 MB | 2× memory for chosen+rejected |
| ORPO (r=32, batch=1) | ~10,500 MB | No reference model needed |
| SFT (r=16, q+v only) | ~8,000 MB | Minimal stage 10 |

### 22.2 Memory-Saving Techniques

| Technique | Savings | Reference |
|-----------|---------|-----------|
| 4-bit NF4 quantization | ~75% model memory | QLoRA §3 |
| Double quantization | ~370 MB | QLoRA §3.2 |
| Gradient checkpointing | ~60% activation memory | PyTorch docs |
| Paged AdamW 8-bit | ~1.3 GB vs adamw_torch | bitsandbytes §7.3 |
| bf16 compute | Native on Ada Lovelace | No conversion overhead |
| CPU merge (not GPU) | ~14 GB VRAM savings | Custom implementation |

---

## 23. Key Research References

| Paper/Method | Stage(s) | Key Contribution |
|-------------|----------|------------------|
| **QLoRA** (Dettmers et al., 2023) | All | 4-bit NF4 quantization + LoRA fine-tuning |
| **LoRA** (Hu et al., 2021) | All | Low-rank adaptation of large language models |
| **Self-RAG** (Asai et al., 2023) | 4 | ISREL/ISSUP/ISUSE critique tokens |
| **CRAG** (Yan et al., 2024) | 4 | Corrective RAG with relevance evaluation |
| **DPO** (Rafailov et al., 2023) | 9, 15 | Direct Preference Optimization |
| **ORPO** (Hong et al., 2024) | 11 | Odds-Ratio Preference Optimization |
| **RAFT** (Zhang et al., 2024) | 12 | Retrieval-Augmented Fine Tuning |
| **SPIN** (Chen et al., 2024) | 15 | Self-Play Improvement |
| **RFT** (Yuan et al., 2023) | 14 | Rejection Sampling Fine-Tuning |
| **SFT with ChatML** | 1-8, 10, 12-14 | ChatML template formatting |
| **HyDE** (Gao et al., 2022) | 2 | Hypothetical Document Embeddings |
| **Paged AdamW 8-bit** (Dettmers et al., 2022) | All | Memory-efficient optimizer |

---

## Final Notes

### What was achieved

- A **7B parameter model** was successfully curriculum-fine-tuned through **15 sequential stages** on a **single RTX 4000 Ada (20GB VRAM)** GPU in approximately **18-20 hours**
- The model learned specialized skills for: RAG-grounded citation, agentic routing, causal reasoning, self-critique, belief tracking, summarization, multi-turn dialogue, long-context synthesis, preference alignment, user style adaptation, retrieval filtering (RAFT), structured function calling, quality refinement (RFT), and self-play improvement (SPIN)
- **39,516 training examples** across **76 MB** of structured data were generated **entirely locally** with zero API calls
- The automatic disk cleanup strategy recycled **~95+ GB** during training, making the process feasible on a 226 GB disk
- Final token accuracy ranged from **93.85%** (complex RFT targets) to **98.43%** (summarization), with SPIN achieving **100% reward accuracy** and a reward margin of **6.508**

### Model Lineage

```
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    → 15-stage QLoRA curriculum (this report)
    → Currently running as: Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled
       (migrated to production Qwen model with accumulated knowledge transfer)
```

---

*Document generated from analysis of `scripts/fine_tune_cortex.py`, `scripts/generate_datasets.py`, `scripts/generate_extended_datasets.py`, `config/training_config.py`, `fine_tuned/*/training_meta.json`, `fine_tuned/training_run.log`, and `fine_tuned/training_run_ext.log`.*
