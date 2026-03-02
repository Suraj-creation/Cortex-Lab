# PageIndex Integration: Vectorless Reasoning-Based RAG for Cortex Lab
## Comprehensive Analysis, Architecture Comparison & Implementation Guide

> **Last Updated:** February 28, 2026  
> **Version:** 1.0  
> **Author:** Cortex Lab Architecture Team  
> **Status:** Research & Integration Planning  
> **PageIndex Version:** SDK v1.x (Vectify AI, Sep 2025)  
> **Cortex Lab Version:** v2.0 (Agentic RAG with BGE-large-1024d + Fine-Tuned DeepSeek-R1-7B)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is PageIndex?](#2-what-is-pageindex)
3. [PageIndex Architecture Deep Dive](#3-pageindex-architecture-deep-dive)
4. [Cortex Lab Current Architecture (Recap)](#4-cortex-lab-current-architecture-recap)
5. [Head-to-Head Comparison: Agentic RAG vs PageIndex](#5-head-to-head-comparison-agentic-rag-vs-pageindex)
6. [Impact Analysis on Cortex Lab Codebase](#6-impact-analysis-on-cortex-lab-codebase)
7. [Recommendation: Hybrid Architecture (Keep Both)](#7-recommendation-hybrid-architecture-keep-both)
8. [Implementation Plan: PageIndex Integration](#8-implementation-plan-pageindex-integration)
9. [Pricing Analysis](#9-pricing-analysis)
10. [MCP Integration Guide](#10-mcp-integration-guide)
11. [Code Changes Required](#11-code-changes-required)
12. [Risk Assessment & Mitigation](#12-risk-assessment--mitigation)
13. [Decision Matrix & Final Verdict](#13-decision-matrix--final-verdict)

---

## 1. Executive Summary

### The Question

Should Cortex Lab adopt **PageIndex** — a vectorless, reasoning-based RAG framework — to replace, augment, or run alongside its existing **Agentic RAG** architecture (FAISS + BGE-large-1024d + 5-channel hybrid retrieval + Self-RAG + FLARE)?

### The Short Answer

**Keep both. Use them for what each does best.**

| Concern | Verdict |
|---------|---------|
| Should we replace Agentic RAG entirely? | **No** — PageIndex is document-centric (PDFs, reports); Cortex Lab's memory system handles conversational memories, beliefs, causal chains, emotions — things PageIndex was NOT designed for |
| Should we add PageIndex as a new retrieval channel? | **Yes** — for document-heavy queries (ingested PDFs, research papers, uploaded files), PageIndex's reasoning-based tree search is superior to chunked vector retrieval |
| Should we keep the existing 5-channel retrieval? | **Yes** — BM25, dense, graph, temporal, proposition channels are essential for the memory-centric use case |
| Is PageIndex free? | Free tier: 200 tree gen pages + 100 chat queries. Pay-as-you-go: $0.01/page tree gen, $0.02/chat query. Scale plan: $50/mo with credits |
| Does it require code changes? | **Moderate** — new retrieval channel module, orchestrator routing update, MCP config, PageIndex SDK integration |
| Privacy concerns? | **Yes** — PageIndex is a cloud API. Documents are uploaded to Vectify AI servers. This conflicts with Cortex Lab's "all data stays local" principle for sensitive data. Use PageIndex ONLY for non-sensitive documents, or wait for their Enterprise on-premise option |

### Architecture Decision Record (ADR)

```
ADR-007: PageIndex Integration for Document Retrieval
─────────────────────────────────────────────────────
Status:   APPROVED (conditional)
Decision: Integrate PageIndex as a 6th retrieval channel specifically for 
          uploaded documents (PDFs, research papers, reports). Keep all 
          existing 5 channels for memory-based retrieval.
Rationale: PageIndex's tree-based reasoning retrieval excels at structured
           document navigation (following cross-references, ToC-based 
           search, maintaining section context). Our vector-based chunking 
           fragments documents and loses structural context.
Conditions:
  1. Only non-sensitive documents are sent to PageIndex cloud API
  2. Memories, personal data, beliefs stay 100% local (existing pipeline)
  3. PageIndex results are integrated via the existing RRF fusion layer
  4. Fallback to local-only retrieval if PageIndex API is unavailable
```

---

## 2. What is PageIndex?

### 2.1 Core Philosophy

PageIndex is a **vectorless, reasoning-based RAG framework** built by [Vectify AI](https://vectify.ai/) (Sep 2025). It fundamentally rejects the traditional RAG pipeline of:

```
Document → Chunk → Embed → Vector DB → Top-K → LLM → Answer
```

Instead, it implements:

```
Document → Hierarchical Tree Index (ToC) → LLM Reasoning-Based Navigation → 
Targeted Section Retrieval → Answer
```

### 2.2 Core Insight

> "Vector-based RAG searches for *similar* text. Reasoning-based RAG *thinks* about where to look and *why*."  
> — Mingtian Zhang & Yu Tang, PageIndex Team

The key insight is that **semantic similarity ≠ relevance**. When you ask "What were the deferred asset values?", the most semantically similar chunks might discuss assets in general, but the actual answer is in Appendix G, Table 5.3, referenced from page 77 — something a vector search cannot follow.

### 2.3 How It Works (Step by Step)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  PageIndex: REASONING-BASED RETRIEVAL FLOW                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INDEXING PHASE (one-time, via API):                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  1. Upload PDF to PageIndex API                                         │   │
│  │  2. PageIndex generates hierarchical "Table of Contents" tree           │   │
│  │     - Each node has: node_id, title, summary, page_index, text         │   │
│  │     - Nodes have sub_nodes (recursive nesting)                         │   │
│  │  3. Tree structure is stored server-side                               │   │
│  │  4. Returns doc_id for subsequent queries                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  RETRIEVAL PHASE (per query):                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  1. READ the Tree (ToC) — understand document structure                │   │
│  │  2. SELECT relevant sections via LLM reasoning                         │   │
│  │     - LLM reads node titles + summaries                                │   │
│  │     - Decides which nodes are relevant based on reasoning              │   │
│  │     - Returns node_ids with reasoning trace                            │   │
│  │  3. EXTRACT content from selected nodes                                │   │
│  │  4. CHECK if information is sufficient                                  │   │
│  │     - Yes → Generate answer                                            │   │
│  │     - No → Return to step 1, explore more sections                     │   │
│  │  5. FOLLOW cross-references (e.g., "see Appendix G")                   │   │
│  │     - LLM navigates tree to referenced section                         │   │
│  │     - Retrieves additional context                                     │   │
│  │  6. ANSWER with full context from multiple coherent sections           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  KEY DIFFERENCE FROM VECTOR RAG:                                                │
│  • No embeddings computed                                                       │
│  • No vector similarity search                                                 │
│  • No arbitrary chunking that breaks context                                   │
│  • LLM actively REASONS about where information is located                     │
│  • Document structure is preserved, not destroyed by chunking                  │
│  • Cross-references are followed naturally                                     │
│  • Chat history informs iterative retrieval                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 The Tree Index Structure

PageIndex generates a hierarchical JSON tree — an LLM-optimized "Table of Contents":

```json
{
  "node_id": "0006",
  "title": "Financial Stability",
  "start_index": 21,
  "end_index": 22,
  "summary": "The Federal Reserve maintains financial stability through...",
  "sub_nodes": [
    {
      "node_id": "0007",
      "title": "Monitoring Financial Vulnerabilities",
      "start_index": 22,
      "end_index": 28,
      "summary": "The Federal Reserve's monitoring focuses on..."
    },
    {
      "node_id": "0008",
      "title": "Domestic and International Cooperation",
      "start_index": 28,
      "end_index": 31,
      "summary": "In 2023, the Federal Reserve collaborated..."
    }
  ]
}
```

This tree serves as an **in-context index** — the LLM can directly reference, navigate, and reason over it during inference. Unlike a vector database (external, static), this index resides within the LLM's active reasoning context.

### 2.5 Search Methods

PageIndex supports three search strategies:

| Method | How It Works | Speed | Accuracy |
|--------|-------------|-------|----------|
| **LLM Tree Search** | LLM reads tree structure + summaries, reasons about which nodes are relevant, returns node_ids | Slower (~2-5s) | Highest — full reasoning |
| **Value-based Tree Search** | Chunks → embedding similarity → node scoring via aggregated chunk scores | Fast (~100ms) | Good — may miss context |
| **Hybrid Tree Search** (default) | Runs both in parallel, merges results via a queue, LLM agent evaluates sufficiency | Balanced | Best of both worlds |

The **Hybrid approach** is inspired by AlphaGo's Monte Carlo Tree Search (MCTS) — combining the speed of value predictions with the depth of LLM reasoning.

### 2.6 Key Differentiators

| Feature | Traditional Vector RAG | PageIndex |
|---------|----------------------|-----------|
| Indexing | Embed chunks into vectors | Generate hierarchical tree |
| Search | Cosine similarity (approximate) | LLM reasoning over structure |
| Chunking | Fixed-size (512/1000 tokens) | Natural document sections |
| Cross-references | Cannot follow | Navigates via tree structure |
| Chat history | Each query isolated | Multi-turn reasoning with context |
| Transparency | Black-box similarity scores | Full reasoning trace visible |
| Expert knowledge | Requires embedding fine-tuning | Add to prompt (zero-shot) |
| Infrastructure | Vector DB required | No vector DB needed |

---

## 3. PageIndex Architecture Deep Dive

### 3.1 The Two-Phase Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PageIndex: COMPLETE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: DOCUMENT PROCESSING (async, one-time per document)           │   │
│  │                                                                         │   │
│  │  PDF Upload → OCR Engine → Markdown Extraction → Tree Generation       │   │
│  │                                                                         │   │
│  │  Input:  Raw PDF (any format, scanned or digital)                      │   │
│  │  OCR:    Structure-preserving markdown conversion                      │   │
│  │  Tree:   Hierarchical ToC with node_id, title, summary, page_index    │   │
│  │  Output: doc_id + searchable tree structure                            │   │
│  │                                                                         │   │
│  │  API:    POST https://api.pageindex.ai/doc/                            │   │
│  │  Cost:   $0.01 per page                                                │   │
│  │  Speed:  ~30-60s for a 100-page document                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: REASONING-BASED RETRIEVAL (per query)                        │   │
│  │                                                                         │   │
│  │  Option A: Chat API (fully managed agentic retrieval)                  │   │
│  │  ─────────────────────────────────────────────────────                 │   │
│  │  POST https://api.pageindex.ai/chat/completions                        │   │
│  │  - Sends query + doc_id(s)                                             │   │
│  │  - PageIndex LLM navigates tree, retrieves content, generates answer   │   │
│  │  - Supports streaming, multi-doc, multi-turn conversations             │   │
│  │  - Cost: $0.02 per query                                               │   │
│  │                                                                         │   │
│  │  Option B: SDK Tree Access (self-directed retrieval)                   │   │
│  │  ─────────────────────────────────────────────────────                 │   │
│  │  get_tree(doc_id) → Full tree structure with node text                 │   │
│  │  - You parse the tree with YOUR LLM (e.g., DeepSeek-R1-7B)            │   │
│  │  - You implement tree search yourself                                  │   │
│  │  - Cost: $0.01/page (tree gen only, no per-query cost)                 │   │
│  │  - More control, but more work                                         │   │
│  │                                                                         │   │
│  │  Option C: MCP Integration (for agent frameworks)                      │   │
│  │  ─────────────────────────────────────────────────────                 │   │
│  │  MCP Server: https://api.pageindex.ai/mcp                              │   │
│  │  - Works with Claude, Vercel AI SDK, LangChain, OpenAI Agents          │   │
│  │  - Same API key, same documents, same plan                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Hybrid Tree Search (Default Algorithm)

PageIndex uses a hybrid approach inspired by AlphaGo's MCTS:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID TREE SEARCH PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                         ┌──────────┐                                            │
│                         │  Query   │                                            │
│                         └────┬─────┘                                            │
│                              │                                                  │
│              ┌───────────────┼───────────────┐                                  │
│              ▼                               ▼                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐                        │
│  │  Value-based Search  │      │  LLM Tree Search     │                        │
│  │  (Fast, ~100ms)      │      │  (Deep, ~2-5s)       │                        │
│  │                      │      │                      │                        │
│  │  1. Chunk each node  │      │  1. Read tree ToC    │                        │
│  │  2. Embed chunks     │      │  2. Reason about     │                        │
│  │  3. Similarity search│      │     which sections   │                        │
│  │  4. Aggregate scores │      │     are relevant     │                        │
│  │     per node         │      │  3. Return node_ids  │                        │
│  │                      │      │     with reasoning   │                        │
│  │  NodeScore =         │      │     trace            │                        │
│  │  Σ ChunkScore(n)     │      │                      │                        │
│  │  ─────────────────   │      │                      │                        │
│  │  √(N+1)              │      │                      │                        │
│  └──────────┬───────────┘      └──────────┬───────────┘                        │
│              │                              │                                   │
│              └──────────────┬───────────────┘                                   │
│                             ▼                                                   │
│                   ┌──────────────────┐                                          │
│                   │  Unified Queue   │                                          │
│                   │  (deduplicated   │                                          │
│                   │   node results)  │                                          │
│                   └────────┬─────────┘                                          │
│                            ▼                                                    │
│                   ┌──────────────────┐                                          │
│                   │  Node Consumer   │                                          │
│                   │  Extract/summarize│                                          │
│                   │  relevant content│                                          │
│                   └────────┬─────────┘                                          │
│                            ▼                                                    │
│                   ┌──────────────────┐                                          │
│                   │  LLM Agent       │                                          │
│                   │  "Enough info?"  │──── No ──→ Search more nodes             │
│                   │  Yes → Answer    │                                          │
│                   └──────────────────┘                                          │
│                                                                                 │
│  Advantages:                                                                    │
│  • Fast initial results from value-based search                                │
│  • Deep reasoning from LLM search catches what embeddings miss                 │
│  • Higher recall than either alone                                             │
│  • Early termination when sufficient info gathered                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Five Limitations of Vector RAG That PageIndex Solves

| # | Problem | Vector RAG Behavior | PageIndex Solution |
|---|---------|--------------------|--------------------|
| 1 | **Query-Knowledge Mismatch** | Matches surface-level semantics; often misses true context | Uses LLM inference to identify relevant sections structurally |
| 2 | **Similarity ≠ Relevance** | Retrieves semantically similar but irrelevant chunks (especially in domain docs) | Retrieves contextually relevant content via reasoning |
| 3 | **Hard Chunking** | Fixed-length chunks (512 tokens) fragment meaning and context | Retrieves coherent, natural document sections dynamically |
| 4 | **No Chat Context** | Each query treated independently; retriever ignores conversation history | Multi-turn reasoning considers prior context |
| 5 | **Cross-References** | "See Appendix G" — vector search cannot follow internal links | LLM navigates tree to referenced sections naturally |

### 3.4 Expert Knowledge Integration (Zero-Shot)

A unique PageIndex advantage: you can inject domain expertise directly into the search prompt without any model fine-tuning:

```python
prompt = f"""
You are given a query and a tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {PageIndex_Tree}

Expert Knowledge of relevant sections: {Preference}

Reply in JSON format:
{{
  "thinking": <reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```

Example expert preference:
> *"If the query mentions EBITDA adjustments, prioritize Item 7 (MD&A) and footnotes in Item 8 (Financial Statements) in 10-K reports."*

In traditional RAG, integrating such knowledge requires fine-tuning the embedding model. In PageIndex, it's just a prompt addition.

---

## 4. Cortex Lab Current Architecture (Recap)

For context, here is a concise summary of Cortex Lab's existing Agentic RAG system:

### 4.1 Core Components (from `engine.py`)

```
CortexRAGEngine
├── EmbeddingModel         (BGE-large-en-v1.5, 1024d)
├── CrossEncoderReranker   (BGE-reranker-v2-m3)
├── VectorStore            (FAISS: HNSW hot + IVF-SQ8 warm + IVF-PQ cold)
├── MetadataStore          (DuckDB — SQL queries, temporal filters)
├── KnowledgeGraph         (NetworkX — entity-relation traversal)
├── LocalLLM               (Fine-Tuned DeepSeek-R1-7B)
├── QueryAnalyzer          (Intent + complexity scoring)
├── QueryTransformer       (Multi-query, HyDE, step-back, decomposition)
├── HybridRetriever        (5-channel + RRF fusion + cross-encoder reranking)
├── AgentOrchestrator      (5 agents + Adaptive-RAG + Self-RAG + FLARE)
├── MemoryIngestionPipeline (11-stage enrichment)
├── MultiLevelCache        (3-level: exact + semantic + embedding)
└── AmbientService         (Voice: VAD + STT + TTS + Speaker ID)
```

### 4.2 The 5 Retrieval Channels (from `hybrid_retriever.py`)

| Channel | Weight | What It Does |
|---------|--------|-------------|
| Dense (BGE-large + FAISS) | 0.35 | Semantic vector similarity search |
| Sparse (BM25 keywords) | 0.25 | Keyword matching (exact terms, names) |
| Graph (Knowledge Graph) | 0.20 | Entity-relation traversal via NetworkX |
| Temporal (SQL time filter) | 0.10 | Time-based memory retrieval via DuckDB |
| Proposition (Atomic facts) | 0.10 | Fine-grained fact-level matching |

These are fused via **Reciprocal Rank Fusion (RRF)** with `k=60`, then reranked with the **BGE-reranker-v2-m3** cross-encoder.

### 4.3 What Cortex Lab Does That PageIndex Cannot

| Capability | Cortex Lab | PageIndex |
|------------|-----------|-----------|
| Personal memory storage | Yes (DuckDB + FAISS + KG) | No |
| Belief evolution tracking | Yes (Stage 5 fine-tuning) | No |
| Emotion detection | Yes (keyword + LLM) | No |
| Causal chain tracing | Yes (CausalAgent + KG) | No |
| Entity resolution | Yes (fuzzy matching + coreference) | No |
| Multi-turn memory-aware chat | Yes (session context) | Partial (Chat API supports multi-turn for docs) |
| Temporal queries ("last month") | Yes (SQL + temporal channel) | No |
| Voice interaction (ambient) | Yes (VAD + STT + TTS) | No |
| Fine-tuned local model | Yes (10-stage QLoRA + DPO) | No (cloud LLM) |
| Works offline / fully local | Yes | No (requires API) |
| Privacy (no data leaves device) | Yes | No (docs uploaded to cloud) |

---

## 5. Head-to-Head Comparison: Agentic RAG vs PageIndex

### 5.1 Fundamental Design Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              ARCHITECTURAL PHILOSOPHY COMPARISON                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  CORTEX LAB (Agentic RAG):                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ MEMORY   │→ │ EMBED    │→ │ MULTIPLE │→ │ RERANK   │→ │ REFLECT  │        │
│  │ INGESTION│  │ (BGE)    │  │ CHANNELS │  │ (Cross-  │  │ (Self-RAG│        │
│  │ (11-step)│  │ 1024d    │  │ (5 fused)│  │ Encoder) │  │ + FLARE) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                                 │
│  Strengths:                                                                     │
│  • Deep personal memory system (beliefs, emotions, causality)                  │
│  • Multi-channel retrieval with RRF fusion                                     │
│  • Self-reflective generation (critique + correction)                          │
│  • Fully local — no cloud dependency                                           │
│  • Fine-tuned model adapted to user's style                                    │
│                                                                                 │
│  Weaknesses:                                                                    │
│  • Fixed chunking fragments documents                                          │
│  • Cannot follow cross-references in documents                                 │
│  • Loses document structure after chunking                                     │
│  • Embedding similarity != true relevance for domain docs                      │
│                                                                                 │
│  ────────────────────────────────────────────────────────────────────           │
│                                                                                 │
│  PageIndex (Vectorless RAG):                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ DOCUMENT │→ │ TREE     │→ │ LLM      │→ │ TARGETED │                       │
│  │ UPLOAD   │  │ INDEX    │  │ REASONING│  │ SECTION  │                       │
│  │ (PDF)    │  │ (ToC)    │  │ SEARCH   │  │ RETRIEVAL│                       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                       │
│                                                                                 │
│  Strengths:                                                                     │
│  • Preserves document structure (no chunking)                                  │
│  • Follows cross-references naturally                                          │
│  • Transparent reasoning trace                                                 │
│  • Expert knowledge via prompt (no fine-tuning needed)                         │
│  • Superior for structured documents (reports, papers, filings)                │
│                                                                                 │
│  Weaknesses:                                                                    │
│  • Cloud-only (data leaves device)                                             │
│  • No personal memory / belief system                                          │
│  • No multi-channel retrieval fusion                                           │
│  • No self-reflective generation                                               │
│  • PDF-only (no code, images, audio, video ingestion)                          │
│  • Pay-per-use pricing (ongoing cost)                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Feature Matrix

| Feature | Cortex Lab Agentic RAG | PageIndex | Winner |
|---------|----------------------|-----------|--------|
| **Document Understanding** | Chunks lose context | Preserves full structure | **PageIndex** |
| **Cross-reference Following** | Cannot | Natural via tree navigation | **PageIndex** |
| **Search Transparency** | Agent reasoning chains visible | Tree search reasoning trace | **Tie** |
| **Personal Memories** | Full system (DuckDB + KG + emotions) | Not designed for this | **Cortex Lab** |
| **Belief Tracking** | Multi-stage contradiction detection | None | **Cortex Lab** |
| **Multi-channel Retrieval** | 5 channels + RRF + reranking | Single tree search | **Cortex Lab** |
| **Self-Correction** | Self-RAG + CRAG + FLARE | None (single-pass) | **Cortex Lab** |
| **Offline Capability** | Fully local, no internet needed | Requires API connection | **Cortex Lab** |
| **Privacy** | All data local, no telemetry | Cloud API, docs uploaded | **Cortex Lab** |
| **Infrastructure Cost** | GPU only (one-time hardware) | Pay-per-use API ($0.01-0.02/op) | **Cortex Lab** |
| **Setup Complexity** | High (12+ components) | Low (pip install + API key) | **PageIndex** |
| **Multi-document Query** | Via KG cross-entity | Native multi-doc chat | **PageIndex** |
| **File Format Support** | 16+ types (PDF, code, audio, video) | PDF only (markdown beta) | **Cortex Lab** |
| **Query Latency** | 1.5-8s (local GPU) | 2-10s (network + cloud LLM) | **Tie** |
| **Expert Knowledge** | Requires fine-tuning / prompting | Zero-shot prompt injection | **PageIndex** |

### 5.3 When to Use Which

| Query Type | Best Choice | Why |
|------------|------------|-----|
| "What did I say about AI ethics last month?" | **Cortex Lab** | Personal memory, temporal, belief tracking |
| "Summarize Section 5 of this research paper" | **PageIndex** | Document structure, section-level retrieval |
| "Why did I change my mind about project X?" | **Cortex Lab** | Belief evolution, causal chain tracing |
| "Compare findings across these 3 uploaded PDFs" | **PageIndex** | Multi-document reasoning, preserves structure |
| "What's the total deferred asset value in the annual report?" | **PageIndex** | Cross-reference following (Appendix G → Table 5.3) |
| "How am I feeling about work lately?" | **Cortex Lab** | Emotion detection, temporal aggregation |
| "What code changes were made in the authentication module?" | **Cortex Lab** | Code-aware chunking, AST parsing |
| "What does page 47 of my uploaded thesis say about methodology?" | **PageIndex** | Page-level retrieval, structure preservation |

---

## 6. Impact Analysis on Cortex Lab Codebase

### 6.1 Files That Need Changes

| File | Change Type | Impact | Effort |
|------|------------|--------|--------|
| `backend/src/engine.py` | **Modify** | Add PageIndexClient initialization, new retrieval method | Medium |
| `backend/src/retrieval/hybrid_retriever.py` | **Modify** | Add 6th channel: `_pageindex_retrieve()` | Medium |
| `backend/src/agents/orchestrator.py` | **Modify** | Route document queries to PageIndex channel | Low |
| `backend/src/ingestion/__init__.py` | **Modify** | Upload PDFs to PageIndex during ingestion | Medium |
| `backend/src/storage/pageindex_store.py` | **Create** | PageIndex SDK wrapper + doc_id management | Medium |
| `backend/requirements.txt` | **Modify** | Add `pageindex` SDK dependency | Trivial |
| `backend/server.py` | **Modify** | Add PageIndex config endpoints | Low |
| `config/pageindex_config.py` | **Create** | API key, enabled/disabled flag, privacy settings | Low |
| `frontend/src/lib/api.ts` | **Modify** | Add PageIndex document management API calls | Low |
| `frontend/src/components/SettingsPanel.tsx` | **Modify** | PageIndex API key config UI | Low |

### 6.2 Files That Stay Unchanged

| File | Why |
|------|-----|
| `backend/src/storage/vector_store.py` | Vector store still needed for memories — no change |
| `backend/src/storage/metadata_store.py` | DuckDB still the primary metadata store |
| `backend/src/storage/knowledge_graph.py` | KG still needed for entity/causal queries |
| `backend/src/retrieval/query_engine.py` | Query analysis/transformation still needed |
| `backend/src/agents/specialized.py` | All 5 agents still needed for memory queries |
| `backend/src/models/` | Embedding model + reranker still needed |
| `backend/src/cache/` | Caching still needed (extends to PageIndex results too) |
| `backend/src/ambient/` | Voice system independent of retrieval backend |
| `frontend/src/components/ChatPanel.tsx` | Chat UI doesn't change (backend handles routing) |
| Training pipeline (all `training_data/`, `scripts/`) | Fine-tuning is for the local model |

### 6.3 Architecture Integration Point

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               HYBRID ARCHITECTURE: CORTEX LAB + PageIndex                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                         ┌─────────────────────┐                                 │
│                         │   USER QUERY         │                                 │
│                         └──────────┬──────────┘                                 │
│                                    │                                            │
│                         ┌──────────▼──────────┐                                 │
│                         │  QUERY ANALYZER      │                                 │
│                         │  Intent + Complexity  │                                 │
│                         └──────────┬──────────┘                                 │
│                                    │                                            │
│                    ┌───────────────┼───────────────┐                             │
│                    │               │               │                             │
│                    ▼               ▼               ▼                             │
│          ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│          │ MEMORY QUERY │  │ DOC QUERY    │  │ HYBRID QUERY │                   │
│          │ (personal)   │  │ (uploaded    │  │ (both needed)│                   │
│          │              │  │  documents)  │  │              │                   │
│          └──────┬──────┘  └──────┬───────┘  └──────┬───────┘                   │
│                 │                │                   │                           │
│                 ▼                ▼                   ▼                           │
│  ┌──────────────────┐  ┌───────────────┐  ┌────────────────────┐               │
│  │ 5-CHANNEL LOCAL  │  │ PageIndex     │  │ BOTH CHANNELS      │               │
│  │ RETRIEVAL        │  │ Chat API      │  │ (merge via RRF)    │               │
│  │                  │  │ or Tree Search│  │                    │               │
│  │ • Dense (BGE)    │  │               │  │ Local → memories   │               │
│  │ • Sparse (BM25)  │  │ • Tree-based  │  │ PageIndex → docs   │               │
│  │ • Graph (KG)     │  │ • Reasoning   │  │ RRF Fusion → top-K │               │
│  │ • Temporal (SQL)  │  │ • Multi-doc   │  │ Rerank → answer    │               │
│  │ • Proposition    │  │               │  │                    │               │
│  └──────────────────┘  └───────────────┘  └────────────────────┘               │
│                                                                                 │
│  ROUTING LOGIC (in AgentOrchestrator):                                         │
│  ─────────────────────────────────────                                         │
│  if query references uploaded documents:                                       │
│      → PageIndex channel (primary) + sparse memory search (secondary)          │
│  elif query is about personal memories/beliefs/emotions:                       │
│      → 5-channel local retrieval (existing pipeline)                           │
│  elif query could benefit from both:                                           │
│      → Run both, fuse with RRF, rerank with cross-encoder                     │
│  else (no-retrieval, casual chat):                                             │
│      → Direct LLM generation (existing path)                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Recommendation: Hybrid Architecture (Keep Both)

### 7.1 Why Not Replace Agentic RAG?

**PageIndex is NOT a replacement for Cortex Lab's core system.** Here's why:

1. **Different data models:** Cortex Lab stores enriched memories (emotions, beliefs, entities, causal links). PageIndex stores documents. These are fundamentally different data types.

2. **Privacy:** Cortex Lab is designed as a local-first personal AI. PageIndex is a cloud API. Uploading personal memories to a third-party server violates the core privacy promise.

3. **No memory system:** PageIndex has no concept of belief evolution, causal chains, temporal memory queries, or emotion tracking.

4. **Single retrieval method:** PageIndex does one thing well (document tree search). Cortex Lab needs 5+ retrieval strategies for the memory use case.

5. **No self-correction:** No Self-RAG, no CRAG, no FLARE. PageIndex's retrieval is single-pass.

6. **Format limitation:** PageIndex currently only supports PDFs (markdown in beta). Cortex Lab ingests 16+ data types.

### 7.2 Why Add PageIndex?

PageIndex fills a genuine gap in Cortex Lab's current architecture:

1. **Document structure preservation:** When a user uploads a 200-page PDF, Cortex Lab chunks it into 512-token pieces, destroying headings, table boundaries, and cross-references. PageIndex preserves the full structure.

2. **Cross-reference following:** If the PDF says "see Appendix G", Cortex Lab's vector search cannot follow that reference. PageIndex can.

3. **Multi-document reasoning:** "Compare the methodology sections of these 3 papers" — PageIndex can navigate each paper's tree structure and compare targeted sections. Cortex Lab would retrieve scattered chunks from all 3 papers.

4. **Reduced self-hosting burden for documents:** PageIndex handles OCR, tree generation, and document-level reasoning in the cloud, freeing GPU resources for the local LLM.

5. **Expert knowledge injection:** For domain-specific document queries, adding expert knowledge to PageIndex's search prompt is zero-shot — no embedding fine-tuning needed.

### 7.3 The Hybrid Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               RECOMMENDED HYBRID ARCHITECTURE                                   │
│               "Best of Both Worlds"                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  DATA LAYER:                                                                    │
│  ┌─────────────────────────┐     ┌─────────────────────────┐                   │
│  │  LOCAL (Cortex Lab)     │     │  CLOUD (PageIndex)       │                   │
│  │  ─────────────────────  │     │  ─────────────────────   │                   │
│  │  • Personal memories    │     │  • Uploaded PDFs         │                   │
│  │  • Beliefs, emotions    │     │  • Research papers       │                   │
│  │  • Chat history         │     │  • Reports, filings     │                   │
│  │  • Entity graph         │     │  • Reference documents   │                   │
│  │  • Causal chains        │     │  • Technical manuals     │                   │
│  │  • Code files           │     │                          │                   │
│  │  • Voice transcripts    │     │  Privacy: NON-SENSITIVE  │                   │
│  │                         │     │  documents only           │                   │
│  │  Privacy: ALL DATA      │     │                          │                   │
│  │  STAYS LOCAL            │     │  Encrypted in transit    │                   │
│  └─────────────────────────┘     └─────────────────────────┘                   │
│                                                                                 │
│  QUERY ROUTING:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  if is_document_query(query) AND has_pageindex_docs():                  │   │
│  │      pageindex_results = pageindex_retrieve(query, doc_ids)             │   │
│  │      local_results = hybrid_retrieve(query)  # may still have memories │   │
│  │      merged = rrf_fuse(pageindex_results, local_results)                │   │
│  │  elif is_personal_query(query):                                         │   │
│  │      results = hybrid_retrieve(query)  # 5-channel local               │   │
│  │  else:                                                                   │   │
│  │      # Run both, let reranker decide                                    │   │
│  │      pageindex_results = pageindex_retrieve_if_available(query)          │   │
│  │      local_results = hybrid_retrieve(query)                             │   │
│  │      merged = rrf_fuse(pageindex_results, local_results)                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Plan: PageIndex Integration

### 8.1 Phase 1: SDK Setup & Document Management (Week 1)

```python
# config/pageindex_config.py

"""
PageIndex integration configuration.
Controls when and how PageIndex is used alongside local retrieval.
"""

PAGEINDEX_CONFIG = {
    # API Configuration
    "api_key": "8aa9ad8830aa438c926efc748b5489a9",  # From dash.pageindex.ai
    "enabled": True,                                   # Master switch
    "mcp_url": "https://api.pageindex.ai/mcp",        # MCP endpoint

    # Privacy Controls
    "allow_cloud_upload": True,              # Set False to disable PageIndex entirely
    "sensitive_data_filter": True,           # Auto-detect & block sensitive content
    "allowed_sources": ["pdf_upload", "research_paper"],  # Only these source types go to PageIndex
    "blocked_sources": ["chat", "voice", "personal_note"],  # Never send these to cloud

    # Retrieval Configuration
    "channel_weight": 0.25,                  # Weight in RRF fusion (vs 0.35 dense, 0.25 sparse, etc.)
    "enable_chat_api": True,                 # Use Chat API (managed reasoning) vs Tree API (self-directed)
    "enable_streaming": True,                # Stream responses for faster UX
    "fallback_to_local": True,               # If PageIndex API fails, use local-only retrieval
    "timeout_seconds": 15,                   # Max wait for PageIndex response

    # Cost Controls
    "max_monthly_queries": 500,              # Budget cap: queries per month
    "max_monthly_pages": 2000,               # Budget cap: pages processed per month
    "track_usage": True,                     # Log usage for cost monitoring
}
```

### 8.2 Phase 2: PageIndex Store Module (Week 1-2)

```python
# backend/src/storage/pageindex_store.py

"""
PageIndex Document Store — manages documents indexed via PageIndex API.
Maps local file uploads to PageIndex doc_ids.
Handles upload, status tracking, tree retrieval, and cleanup.
"""

from typing import Dict, List, Optional
from pageindex import PageIndexClient
import json, os, time

class PageIndexStore:
    """Interface between Cortex Lab and PageIndex cloud API."""

    def __init__(self, api_key: str, data_dir: str = "data/pageindex"):
        self.client = PageIndexClient(api_key=api_key)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Local mapping: file_hash -> doc_id
        self.doc_mapping: Dict[str, str] = {}  # {file_hash: pi-doc_id}
        self._load_mapping()

    def upload_document(self, file_path: str) -> str:
        """Upload a PDF to PageIndex, return doc_id."""
        result = self.client.submit_document(file_path, mode="mcp")
        doc_id = result["doc_id"]
        # Store mapping
        file_hash = self._hash_file(file_path)
        self.doc_mapping[file_hash] = doc_id
        self._save_mapping()
        return doc_id

    def is_ready(self, doc_id: str) -> bool:
        """Check if document processing is complete."""
        return self.client.is_retrieval_ready(doc_id)

    def get_tree(self, doc_id: str, include_summaries: bool = True) -> dict:
        """Get the PageIndex tree structure for a document."""
        return self.client.get_tree(doc_id, node_summary=include_summaries)

    def chat_retrieve(self, query: str, doc_ids: List[str],
                       stream: bool = False) -> str:
        """
        Use PageIndex Chat API for managed agentic retrieval.
        Returns the retrieved + generated answer.
        """
        messages = [{"role": "user", "content": query}]
        
        if stream:
            result = ""
            for chunk in self.client.chat_completions(
                messages=messages,
                doc_id=doc_ids if len(doc_ids) > 1 else doc_ids[0],
                stream=True
            ):
                result += chunk
            return result
        else:
            response = self.client.chat_completions(
                messages=messages,
                doc_id=doc_ids if len(doc_ids) > 1 else doc_ids[0],
                stream=False
            )
            return response["choices"][0]["message"]["content"]

    def agentic_retrieve(self, query: str, doc_ids: List[str]) -> List[dict]:
        """
        Use PageIndex Chat API in retrieval mode.
        Returns structured JSON with page numbers and content.
        """
        retrieval_prompt = f"""
Your job is to retrieve the raw relevant content from the document based on the user's query.

Query: {query}

Return in JSON format:
```json
[
  {{
    "page": <number>,
    "content": "<raw text>"
  }},
  ...
]
```
"""
        messages = [{"role": "user", "content": retrieval_prompt}]
        
        full_response = ""
        for chunk in self.client.chat_completions(
            messages=messages,
            doc_id=doc_ids if len(doc_ids) > 1 else doc_ids[0],
            stream=True
        ):
            full_response += chunk
        
        # Parse JSON from response
        return self._extract_json(full_response)

    def list_documents(self) -> List[dict]:
        """List all documents in PageIndex."""
        result = self.client.list_documents(limit=100)
        return result.get("documents", [])

    def delete_document(self, doc_id: str):
        """Delete a document from PageIndex."""
        self.client.delete_document(doc_id)
        # Remove from local mapping
        self.doc_mapping = {k: v for k, v in self.doc_mapping.items() if v != doc_id}
        self._save_mapping()

    def get_doc_ids_for_query(self, query: str) -> List[str]:
        """
        Determine which PageIndex documents are relevant to a query.
        For now, returns all doc_ids. Future: semantic doc search.
        """
        return list(self.doc_mapping.values())

    # ─── Private ─────────────────────────────────────────────────────

    def _load_mapping(self):
        path = os.path.join(self.data_dir, "doc_mapping.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.doc_mapping = json.load(f)

    def _save_mapping(self):
        path = os.path.join(self.data_dir, "doc_mapping.json")
        with open(path, "w") as f:
            json.dump(self.doc_mapping, f, indent=2)

    def _hash_file(self, file_path: str) -> str:
        import hashlib
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _extract_json(self, content: str) -> list:
        """Extract JSON array from LLM response."""
        import re
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
        return []
```

### 8.3 Phase 3: Hybrid Retriever Integration (Week 2)

Add a 6th channel to `HybridRetriever`:

```python
# Addition to backend/src/retrieval/hybrid_retriever.py

# Updated channel weights (renormalized to sum to 1.0)
WEIGHTS = {
    "dense": 0.30,        # Was 0.35
    "sparse": 0.20,        # Was 0.25
    "graph": 0.15,         # Was 0.20
    "temporal": 0.10,      # Same
    "proposition": 0.05,   # Was 0.10
    "pageindex": 0.20,     # NEW — document-level reasoning retrieval
}

async def _pageindex_retrieve(self, query: MemoryQuery, top_k: int) -> List[Tuple[str, float]]:
    """
    Channel 6: PageIndex reasoning-based document retrieval.
    Uses Chat API for managed retrieval over uploaded documents.
    
    Returns: List of (memory_id, score) tuples
    """
    if not self.pageindex_store or not self.pageindex_store.doc_mapping:
        return []
    
    try:
        doc_ids = self.pageindex_store.get_doc_ids_for_query(query.query_text)
        if not doc_ids:
            return []
        
        results = self.pageindex_store.agentic_retrieve(
            query=query.query_text,
            doc_ids=doc_ids
        )
        
        # Convert PageIndex results to (memory_id, score) format
        scored_results = []
        for i, result in enumerate(results[:top_k]):
            # Create a synthetic memory_id linked to the PageIndex result
            memory_id = f"pageindex:{result.get('page', 0)}:{doc_ids[0]}"
            score = 1.0 - (i * 0.05)  # Rank-based scoring
            scored_results.append((memory_id, score))
        
        return scored_results
    except Exception as e:
        print(f"  ⚠ PageIndex retrieval failed: {e}")
        return []  # Graceful degradation
```

### 8.4 Phase 4: Orchestrator Routing Update (Week 2-3)

```python
# Addition to backend/src/agents/orchestrator.py

def _is_document_query(self, query: str, analysis) -> bool:
    """
    Detect if the query is about uploaded documents vs personal memories.
    Indicators:
    - References to "the paper", "this document", "the report"
    - Asks about specific sections, pages, or chapters
    - Mentions uploaded file names
    - Academic/professional query patterns
    """
    doc_indicators = [
        "paper", "document", "report", "section", "chapter", "page",
        "figure", "table", "appendix", "abstract", "methodology",
        "findings", "conclusion", "pdf", "uploaded", "the file",
        "this article", "the study", "the analysis"
    ]
    query_lower = query.lower()
    return any(ind in query_lower for ind in doc_indicators)

async def process(self, query: str, session_id: str = "",
                   conversation_history: List[Dict] = None) -> OrchestratorResponse:
    """Enhanced routing with PageIndex awareness."""
    
    # ... existing analysis ...
    analysis = self.analyzer.analyze(query)
    
    # NEW: Check if this is a document query
    if self._is_document_query(query, analysis):
        # Route to PageIndex-enhanced retrieval
        return await self._process_with_pageindex(query, analysis, session_id)
    
    # ... existing routing logic ...
```

---

## 9. Pricing Analysis

### 9.1 PageIndex Developer Plans

| Plan | Cost | Tree Gen | Chat API | MCP | Support |
|------|------|----------|----------|-----|---------|
| **Free Trial** | $0 | 200 pages | 100 queries | Yes | Community |
| **Starter** | Pay-as-you-go | $0.01/page | 100 free queries | Standard | Email |
| **Scale** (Popular) | $50/mo (incl. credits) | $0.01/page | $0.02/query | Unlimited | 24/7 priority |
| **Enterprise** | Custom | Custom | Custom | Custom | Dedicated |

### 9.2 Cost Projections for Cortex Lab

Assuming typical usage patterns for a personal knowledge assistant:

| Usage Scenario | Monthly Volume | Monthly Cost (Starter) | Monthly Cost (Scale) |
|---------------|---------------|----------------------|---------------------|
| **Light user** (5 docs, 50 queries) | 250 pages + 50 queries | $2.50 + $0 = **$2.50** | **$50** (overkill) |
| **Medium user** (20 docs, 200 queries) | 1,000 pages + 200 queries | $10 + $2 = **$12** | **$50** (within credits) |
| **Heavy user** (50 docs, 500 queries) | 2,500 pages + 500 queries | $25 + $8 = **$33** | **$50** (within credits) |
| **Power user** (100 docs, 1000 queries) | 5,000 pages + 1000 queries | $50 + $18 = **$68** | $50 + $10 overage = **$60** |

### 9.3 Cost vs Benefit Analysis

| Factor | Local-Only (Current) | With PageIndex |
|--------|---------------------|---------------|
| Hardware cost | RTX 4000 Ada (~$1,000 one-time) | Same + $12-60/mo API |
| Electricity | ~$5-10/mo (GPU inference) | Same (GPU still needed) |
| Document understanding | Chunked, lossy | Structure-preserving |
| Cross-reference resolution | Cannot | Can follow |
| Setup effort | 12+ components | +1 SDK install + API key |
| Ongoing maintenance | Weekly ~30min | Same (PageIndex is managed) |

**Verdict:** For users who work with documents regularly (researchers, professionals), the $12-50/mo cost is justified by the quality improvement in document retrieval. For personal-memory-only users, PageIndex adds cost without significant benefit.

### 9.4 Cost Optimization Strategies

1. **Use Tree API (Option B) instead of Chat API (Option A):** Tree generation is $0.01/page (one-time). Then use YOUR local DeepSeek-R1-7B to do the tree search — no per-query cost. Trade-off: more implementation work, uses local GPU.

2. **Cache PageIndex trees locally:** After generating a tree, cache the full JSON tree structure locally. Subsequent queries use cached tree + local LLM reasoning — zero API cost.

3. **Selective upload:** Only upload documents the user explicitly marks as "index with PageIndex." Don't auto-upload everything.

4. **Batch tree generation:** Upload multiple documents at once during off-peak hours to minimize API calls.

---

## 10. MCP Integration Guide

### 10.1 MCP Server Configuration

PageIndex provides a ready-made MCP server. Add this to your MCP configuration:

```json
{
  "mcpServers": {
    "pageindex": {
      "type": "http",
      "url": "https://api.pageindex.ai/mcp",
      "headers": {
        "Authorization": "Bearer 8aa9ad8830aa438c926efc748b5489a9"
      }
    }
  }
}
```

### 10.2 What the MCP Server Provides

The PageIndex MCP server exposes the same API endpoints as the REST API:

| MCP Tool | REST Equivalent | What It Does |
|----------|----------------|-------------|
| `submit_document` | `POST /doc/` | Upload PDF for processing |
| `get_document` | `GET /doc/{id}/metadata` | Check processing status |
| `get_tree` | `GET /doc/{id}/?type=tree` | Get tree structure |
| `get_ocr` | `GET /doc/{id}/?type=ocr` | Get OCR results |
| `chat_completions` | `POST /chat/completions` | Chat with documents |
| `list_documents` | `GET /docs` | List all documents |
| `delete_document` | `DELETE /doc/{id}/` | Delete a document |

### 10.3 MCP vs SDK: When to Use Which

| Scenario | Use MCP | Use SDK |
|----------|---------|---------|
| Claude-based agent orchestration | Yes | No |
| Vercel AI SDK integration | Yes | No |
| Direct Python backend integration | No | Yes (recommended) |
| Cortex Lab (Python + FastAPI backend) | No | **Yes** — SDK is cleaner |
| LangChain/LangGraph agent chains | Yes | Also works |
| Multiple AI agent frameworks | Yes | Harder to share |

**For Cortex Lab: Use the Python SDK directly** in the backend, not MCP. MCP is designed for multi-framework agent orchestration (Claude, Vercel, LangChain). Since Cortex Lab has its own orchestrator, the SDK is more direct and efficient.

However, **keep MCP config available** for users who want to use Cortex Lab's PageIndex documents with external tools (e.g., Claude Desktop, Cursor).

---

## 11. Code Changes Required

### 11.1 Summary of Changes

```
CHANGES OVERVIEW:

backend/
  requirements.txt                  +1 line (pageindex SDK)
  src/
    engine.py                       +40 lines (PageIndex init + routing)
    storage/
      pageindex_store.py            NEW FILE (~200 lines)
    retrieval/
      hybrid_retriever.py           +60 lines (6th channel)
    agents/
      orchestrator.py               +30 lines (document query routing)
    ingestion/
      __init__.py                   +20 lines (PDF upload to PageIndex)

config/
  pageindex_config.py               NEW FILE (~30 lines)

frontend/
  src/lib/api.ts                    +20 lines (PageIndex doc management)
  src/components/SettingsPanel.tsx   +30 lines (PageIndex config UI)

TOTAL: ~2 new files, ~6 modified files, ~430 lines of new code
```

### 11.2 Dependency Addition

```
# backend/requirements.txt — add:
pageindex>=1.0.0
```

### 11.3 Engine.py Changes

```python
# In CortexRAGEngine.__init__():
self.pageindex_store: Optional[PageIndexStore] = None

# In CortexRAGEngine.init():
# 11. PageIndex Store (optional, cloud-based document retrieval)
from config.pageindex_config import PAGEINDEX_CONFIG
if PAGEINDEX_CONFIG.get("enabled", False):
    try:
        from src.storage.pageindex_store import PageIndexStore
        self.pageindex_store = PageIndexStore(
            api_key=PAGEINDEX_CONFIG["api_key"],
            data_dir=f"{self.data_dir}/pageindex"
        )
        # Inject into hybrid retriever
        self.hybrid_retriever.pageindex_store = self.pageindex_store
        print("  📄 PageIndex integration enabled")
    except Exception as e:
        print(f"  ⚠ PageIndex init failed: {e}")
```

### 11.4 Privacy Guard

```python
# In MemoryIngestionPipeline — when ingesting a PDF:

def _should_upload_to_pageindex(self, source: str, content: str) -> bool:
    """
    Privacy guard: only upload non-sensitive documents to PageIndex cloud.
    Personal memories, voice transcripts, and chat never leave the device.
    """
    from config.pageindex_config import PAGEINDEX_CONFIG
    
    if not PAGEINDEX_CONFIG.get("enabled", False):
        return False
    if not PAGEINDEX_CONFIG.get("allow_cloud_upload", False):
        return False
    if source in PAGEINDEX_CONFIG.get("blocked_sources", []):
        return False
    if source not in PAGEINDEX_CONFIG.get("allowed_sources", []):
        return False
    
    # Sensitive content detection (basic)
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',       # SSN
        r'\b\d{16}\b',                     # Credit card
        r'password\s*[:=]',                # Passwords
        r'(private|secret|confidential)',  # Explicit markers
    ]
    import re
    for pattern in sensitive_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False
    
    return True
```

---

## 12. Risk Assessment & Mitigation

### 12.1 Risk Matrix

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|-----------|--------|-----------|
| **Privacy breach** — sensitive data sent to cloud | Critical | Low (with guards) | High | Sensitive content filter + source-based blocking + user consent UI |
| **API downtime** — PageIndex unavailable | Medium | Medium | Low | Fallback to local-only retrieval (existing 5 channels) |
| **Cost overrun** — unexpected API charges | Low | Medium | Low | Monthly usage caps + tracking + alerts |
| **Vendor lock-in** — dependence on Vectify AI | Medium | Low | Medium | Cache trees locally; tree format is portable JSON |
| **Latency increase** — network round-trip | Low | High (inherent) | Low | Async parallel execution; cache PageIndex results |
| **Model inconsistency** — PageIndex LLM vs local DeepSeek | Low | Medium | Low | Use PageIndex for retrieval only; final answer from local LLM |
| **Data inconsistency** — doc updated locally but not in PageIndex | Medium | Medium | Low | Re-upload on file change; hash-based change detection |

### 12.2 Graceful Degradation with PageIndex

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEGRADATION TIERS (Updated for PageIndex)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Tier 1 (FULL): All components + PageIndex API → Full capability               │
│                 Document reasoning + memory retrieval + voice                   │
│                                                                                 │
│  Tier 2 (NO PAGEINDEX): PageIndex API unavailable → Local-only                 │
│                          All 5 channels work; document queries                  │
│                          fall back to chunked vector retrieval                  │
│                          (degraded document quality, but functional)            │
│                                                                                 │
│  Tier 3 (DEGRADED): LLM + BGE + FAISS + DuckDB → Core RAG works              │
│                      No PageIndex, no graph, no BM25                           │
│                                                                                 │
│  Tier 4 (MINIMAL): LLM + DuckDB only → Direct SQL + LLM generation            │
│                                                                                 │
│  Tier 5 (OFFLINE): DuckDB only → Browse/search memories without LLM           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 12.3 Privacy Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PRIVACY-FIRST DATA ROUTING                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  USER INPUT                                                                     │
│      │                                                                          │
│      ▼                                                                          │
│  ┌──────────────────────────────────┐                                          │
│  │  PRIVACY CLASSIFIER              │                                          │
│  │                                   │                                          │
│  │  Source type?                      │                                          │
│  │  ├── chat message → LOCAL ONLY    │                                          │
│  │  ├── voice transcript → LOCAL ONLY│                                          │
│  │  ├── personal note → LOCAL ONLY   │                                          │
│  │  ├── uploaded PDF → EVALUATE      │                                          │
│  │  └── research paper → EVALUATE    │                                          │
│  │                                   │                                          │
│  │  Content scan?                    │                                          │
│  │  ├── SSN detected → LOCAL ONLY    │                                          │
│  │  ├── Credit card → LOCAL ONLY     │                                          │
│  │  ├── "confidential" → LOCAL ONLY  │                                          │
│  │  └── Clean → PAGEINDEX OK         │                                          │
│  │                                   │                                          │
│  │  User consent?                    │                                          │
│  │  ├── "Index with PageIndex" ✓     │                                          │
│  │  └── No consent → LOCAL ONLY      │                                          │
│  └──────────────────────────────────┘                                          │
│              │                    │                                              │
│              ▼                    ▼                                              │
│    ┌──────────────┐    ┌──────────────┐                                        │
│    │ LOCAL STORAGE │    │ PageIndex    │                                        │
│    │ (FAISS+DuckDB│    │ Cloud API    │                                        │
│    │  +KG)        │    │ (tree gen +  │                                        │
│    │              │    │  retrieval)  │                                        │
│    │ 100% on-device│    │ Encrypted    │                                        │
│    └──────────────┘    │ in transit   │                                        │
│                         └──────────────┘                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Decision Matrix & Final Verdict

### 13.1 Decision Matrix

| Criterion | Weight | Agentic RAG Only | PageIndex Only | **Hybrid (Both)** |
|-----------|--------|-----------------|---------------|-------------------|
| Personal memory system | 25% | 10/10 | 1/10 | **10/10** |
| Document understanding | 20% | 5/10 | 9/10 | **9/10** |
| Privacy | 15% | 10/10 | 3/10 | **8/10** |
| Multi-channel retrieval | 10% | 9/10 | 4/10 | **9/10** |
| Self-correction | 10% | 9/10 | 3/10 | **9/10** |
| Setup simplicity | 5% | 4/10 | 9/10 | **4/10** |
| Ongoing cost | 5% | 9/10 | 5/10 | **7/10** |
| Cross-reference handling | 5% | 2/10 | 9/10 | **9/10** |
| Offline capability | 5% | 10/10 | 0/10 | **8/10** |
| **Weighted Score** | | **7.85** | **4.20** | **8.60** |

### 13.2 Final Verdict

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FINAL RECOMMENDATION                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ✅ ADOPT HYBRID ARCHITECTURE                                                   │
│                                                                                 │
│  1. KEEP all existing Agentic RAG components                                   │
│     - 5-channel hybrid retrieval (dense, sparse, graph, temporal, proposition) │
│     - Self-RAG + CRAG + FLARE self-correction                                  │
│     - Agent orchestration (5 specialized agents)                               │
│     - Memory system (beliefs, emotions, causal chains)                         │
│     - Local LLM (Fine-Tuned DeepSeek-R1-7B)                                   │
│     - Voice system (ambient STT/TTS)                                           │
│                                                                                 │
│  2. ADD PageIndex as a 6th retrieval channel                                   │
│     - For uploaded documents (PDFs, research papers, reports)                  │
│     - Reasoning-based tree search (superior document understanding)            │
│     - Managed cloud API (offloads document processing from local GPU)          │
│     - Cross-reference following (unique capability)                            │
│                                                                                 │
│  3. ENFORCE privacy boundaries                                                  │
│     - Personal memories NEVER leave the device                                 │
│     - Only user-consented, non-sensitive documents go to PageIndex             │
│     - Sensitive content auto-detection blocks cloud upload                     │
│     - Fallback to local-only if PageIndex unavailable                          │
│                                                                                 │
│  4. USE the Python SDK (not MCP) for backend integration                       │
│     - More direct, lower latency, better error handling                        │
│     - MCP config available for external tool integration                       │
│                                                                                 │
│  5. START with the Starter plan ($0.01/page, 100 free queries)                 │
│     - Upgrade to Scale ($50/mo) if usage exceeds light tier                    │
│     - Cache trees locally to minimize per-query costs                          │
│                                                                                 │
│  6. CONSIDER future: self-hosted tree generation                               │
│     - PageIndex is open-source on GitHub: github.com/VectifyAI/PageIndex       │
│     - When stable, run tree generation locally for full privacy                │
│     - Eliminates cloud dependency entirely                                     │
│     - Uses local DeepSeek-R1-7B for tree search (already available)            │
│                                                                                 │
│  ESTIMATED EFFORT: 2-3 weeks, ~430 lines of new code                          │
│  ESTIMATED COST:   $12-50/mo (depends on document volume)                      │
│  RISK LEVEL:       LOW (additive change, no existing functionality removed)    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 13.3 What Cortex Lab Gains from PageIndex

| Before (Agentic RAG Only) | After (Hybrid: Agentic RAG + PageIndex) |
|---------------------------|----------------------------------------|
| Documents chunked into 512-token fragments | Full document structure preserved via tree index |
| Cross-references like "see Appendix G" are lost | LLM navigates tree to follow references naturally |
| All processing on local GPU (20GB VRAM constraint) | Document processing offloaded to cloud; local GPU focuses on memory + LLM |
| Expert knowledge requires embedding fine-tuning | Zero-shot prompt injection into tree search |
| 5 retrieval channels | 6 retrieval channels (+ PageIndex reasoning channel) |
| Document questions answered from scattered chunks | Document questions answered from coherent sections |

### 13.4 What Stays The Same

- **All personal data stays local** — memories, beliefs, emotions, causal chains, voice
- **All 5 existing retrieval channels** — unchanged
- **Self-RAG + CRAG + FLARE** — unchanged
- **Agent orchestrator** — enhanced routing only
- **Fine-tuned DeepSeek-R1-7B** — still the primary LLM
- **Voice system (ambient)** — completely independent
- **Training pipeline** — unchanged
- **Frontend** — minimal changes (settings panel for PageIndex config)

---

## Appendix A: Quick Setup Checklist

```bash
# 1. Install PageIndex SDK
pip install -U pageindex

# 2. Get API key from https://dash.pageindex.ai/api-keys
# Already have: 8aa9ad8830aa438c926efc748b5489a9

# 3. Test connection
python -c "
from pageindex import PageIndexClient
client = PageIndexClient(api_key='8aa9ad8830aa438c926efc748b5489a9')
docs = client.list_documents()
print(f'Connected! {len(docs.get(\"documents\", []))} documents')
"

# 4. Upload a test document
python -c "
from pageindex import PageIndexClient
client = PageIndexClient(api_key='8aa9ad8830aa438c926efc748b5489a9')
result = client.submit_document('./test.pdf')
print(f'Uploaded: {result[\"doc_id\"]}')
"

# 5. Add MCP config (for external tools like Claude Desktop)
# See Section 10.1
```

## Appendix B: PageIndex API Reference (Quick)

| Action | Method | Endpoint | Cost |
|--------|--------|----------|------|
| Upload PDF | `POST` | `/doc/` | $0.01/page |
| Check status | `GET` | `/doc/{id}/metadata` | Free |
| Get tree | `GET` | `/doc/{id}/?type=tree` | Free |
| Get OCR | `GET` | `/doc/{id}/?type=ocr` | Free |
| Chat with doc | `POST` | `/chat/completions` | $0.02/query |
| List all docs | `GET` | `/docs` | Free |
| Delete doc | `DELETE` | `/doc/{id}/` | Free |

## Appendix C: Related Architecture Documents

- [RAG-Architecture.md](RAG-Architecture.md) — Full Agentic RAG system design (this document provides the PageIndex integration layer on top)
- [Advanced_RAG_Architecture_Guide.md](Advanced_RAG_Architecture_Guide.md) — Research foundations and advanced techniques
- [RAG_Literature_Survey.md](RAG_Literature_Survey.md) — Academic paper survey underpinning the architecture
- [Fine-Tuning.md](Fine-Tuning.md) — 10-stage training curriculum for the local model
- [DEPLOYMENT.md](DEPLOYMENT.md) — Deployment guide
- [STT-&-TTS.md](STT-&-TTS.md) — Voice system architecture

## Appendix D: Citation

```bibtex
@article{zhang2025pageindex,
  author = {Mingtian Zhang and Yu Tang and PageIndex Team},
  title = {PageIndex: Next-Generation Vectorless, Reasoning-based RAG},
  journal = {PageIndex Blog},
  year = {2025},
  month = {September},
  note = {https://pageindex.ai/blog/pageindex-intro},
}
```

---

> *"The best retrieval system is not one that finds similar text — it's one that understands where to look and why."*
> — PageIndex Philosophy, adapted for Cortex Lab
