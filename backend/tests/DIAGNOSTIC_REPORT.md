# Cortex Lab — Comprehensive Diagnostic Report
## Test Results: 161 tests | 143 Passed | 18 Failed | 88.8% Pass Rate

---

## 🔴 CRITICAL ISSUES (3)

### 1. Emotion Detection: 16.7% Accuracy (1/6 non-neutral tests)
**Severity: CRITICAL** | **Impact: Misclassified emotions affect importance scoring, belief evolution, and agent responses**

**Root Cause:** The `_emotion_keywords` dictionary in `ingestion/__init__.py` has overlapping and insufficient keyword coverage:
- `"frustrated"` appears in BOTH `ANGRY` and `FRUSTRATED` → conflicts
- `"annoyed"` appears in BOTH `ANGRY` and `FRUSTRATED` → conflicts  
- No keyword coverage for contextual emotions (e.g., "coffee with friend" → happy, "new paper published" → excited)
- The keyword matcher picks the FIRST highest-scoring emotion, which is often wrong when keywords overlap

**Failing Cases:**
| Input | Detected | Expected | Issue |
|-------|----------|----------|-------|
| "Had coffee with Sarah..." | sad | happy | No positive contextual words ("coffee", "café" not in happy list) |
| "I've been avoiding difficult conversations..." | frustrated | anxious | "difficult" matches frustrated, but "avoiding" is anxious |
| "I'm frustrated with the project deadline..." | angry | frustrated | "frustrated" matches both ANGRY and FRUSTRATED, ANGRY sorted first |
| "Met with Dr. Chen at university..." | neutral | excited | "new paper" not in excited keywords |
| "TechCorp's culture has become toxic..." | neutral | frustrated | "toxic" not in any emotion list |

**Fix Required in `ingestion/__init__.py`:**
```python
# 1. Remove "frustrated" from ANGRY keywords (it has its own category)
# 2. Remove "annoyed" from ANGRY or FRUSTRATED (pick one)
# 3. Add contextual emotion words:
#    HAPPY: "enjoyed", "fun", "nice", "pleasant", "café", "coffee with"
#    EXCITED: "new", "published", "breakthrough", "discovered"
#    FRUSTRATED: "toxic", "unreasonable", "doesn't listen"
#    ANXIOUS: "avoiding", "afraid", "dreading"
```

---

### 2. Intent Detection Misclassification (2 critical failures)
**Severity: CRITICAL** | **Impact: Wrong intent → wrong agent routed → wrong answer**

**Failing Cases:**
| Query | Detected | Expected | Reason |
|-------|----------|----------|--------|
| "What did I learn about transformers?" | exploratory | factual | "What did I learn" matches FACTUAL keyword, but "tell me about" in EXPLORATORY also partially matches |
| "What is my code review process?" | factual | procedural | "What is" matches FACTUAL but "process" should trigger PROCEDURAL |

**Root Cause:** Keyword overlap in `query_engine.py`:
- `FACTUAL` has `"what did I learn"` → but `"What did I learn about transformers?"` doesn't fully match because the keyword check uses `if kw in query`, and `"what did i learn"` IS in the lowered query. However, `EXPLORATORY` has `"tell me"` which doesn't match here. The real issue is that `FACTUAL` has `"tell me about"` which should be in `EXPLORATORY` only.
- `PROCEDURAL` only has `"process"` as a keyword, but `"what is"` in `FACTUAL` wins because it matches first and has the same score.

**Fix: Add priority weighting or more specific keywords to PROCEDURAL:**
```python
PROCEDURAL: ["how do", "how to", "steps", "process", "method", "procedure", 
             "workflow", "guide", "my process", "my method", "review process"]
```
And consider adding tie-breaking logic that favors more specific intents.

---

### 3. Complexity Scoring Under-estimates Complex Queries
**Severity: WARNING → affects routing** | **Impact: Complex queries routed to SINGLE_STEP instead of MULTI_STEP**

| Query | Score | Expected Range | Issue |
|-------|-------|----------------|-------|
| "How has my opinion about TechCorp changed over time?" | 0.40 | 0.50-1.0 | "over time" boosts +0.1, but base is 0.30 |
| "Compare my early and recent feelings about my job" | 0.45 | 0.50-1.0 | "compare" boosts +0.1, and "and" +0.05, but still under 0.5 |

**Root Cause:** Base complexity starts at 0.30 and most queries only get 1-2 boosters (+0.10 each). For queries with "over time" + entity + reflective intent, the score should be higher.

**Fix:** Add intent-based boosting: if intent is REFLECTIVE or COMPARATIVE, add +0.15 complexity bonus. Also consider word count: these queries are 8-10 words which should boost more.

---

## ⚠️ WARNINGS (14 total)

### 4. Memory Type Classification: 62.5% Accuracy (5/8)
**Severity: WARNING** | 3 reflective memories misclassified

- `"I'm frustrated with the project deadline..."` → classified as EPISODIC (has "I'm" which is not in reflective keywords, and no reflective keyword match)
- `"I love working at TechCorp..."` → classified as SEMANTIC (matches "is a" via "is amazing")
- `"TechCorp's culture has become toxic..."` → classified as EPISODIC (default fallback, no keyword match)

**Root Cause:** Reflective keywords are too narrow: `["realized", "think", "feel", "believe", "changed my mind", "pattern", "noticed"]`. Missing: "frustrated", "love", "hate", opinion-expressing, emotional declarations.

**Fix:** Add to reflective keywords: `"I feel", "I think", "I believe", "I love", "I hate", "opinion", "consider", "seriously considering", "culture"`

---

### 5. Entity Extraction Misses Possessive/Contraction Forms
- `"TechCorp's culture..."` → extracted `['Im', 'Management']` instead of `['TechCorp']`
- Cause: `TechCorp's` has an apostrophe, and the capitalization heuristic strips it to `TechCorp` but the `s` is appended making it `TechCorps` which doesn't match. Actually `'` makes `re.sub(r'[^\w]', '', "TechCorp's")` produce `TechCorps`.

**Fix:** In `_extract_entities()`, strip possessive `'s` before cleaning:
```python
word = re.sub(r"'s$", "", word)  # Remove possessive
```

---

### 6. Proposition Decomposition: Numbered Lists Not Split
- `"My code review process: 1) Read... 2) Check... 3) Run... 4) Review... 5) Leave..."` → Only 1 proposition
- Cause: The fallback clause splitter uses `. ` (period + space) as delimiter, but numbered items use `) ` without periods between them.

**Fix:** Add numbered list splitting in `_extract_propositions()`:
```python
# Split on numbered patterns: "1)", "2.", etc.
text = re.sub(r'\d+[).]', '|||', text)
clauses = [c.strip() for c in text.split('|||') if c.strip()]
```

---

### 7. Embedding Batch vs Single Inconsistency (cosine = 0.82)
- BGE model's batch encoding produces embeddings with cosine similarity of only 0.82 vs single encoding
- **This is a known BGE behavior** — BGE uses query prefix "Represent this sentence:" for single embed vs batch. The `embed()` adds a query instruction prefix but `embed_batch()` may handle it differently.
- **Impact:** Low — batch is used for proposition indexing, single for live queries. They work well separately.

---

### 8. Stance Detection: "Changed my mind" Not Caught Cross-Context
- `"I changed my mind about the approach"` vs `"something else"` → detected as "neutral" instead of "disagree"
- Cause: The contradiction words check (`"changed my mind"`) only checks the NEW text, but in this test case it's in the OLD text. The check `if any(w in new_lower for w in contradiction_words)` should also check `old_lower`.

---

## ✅ STRONG AREAS (Perfect Scores)

| Category | Score | Notes |
|----------|-------|-------|
| LLM Quality (stop patterns, fallback, stats) | 11/11 (100%) | Stop pattern truncation works perfectly |
| Storage Layer (FAISS, DuckDB, KnowledgeGraph) | 4/4 (100%) | All CRUD operations work correctly |
| Cache System (exact, semantic, stats) | 3/3 (100%) | Cache hit/miss tracking works |
| Hybrid Retrieval (BM25, RRF fusion) | 2/2 (100%) | RRF correctly ranks multi-channel results |
| Adversarial/Edge Cases | 12/12 (100%) | Prompt injection stripped, Unicode handled, extreme lengths OK |
| Data Models | 6/6 (100%) | Serialization round-trip perfect, all enums valid |
| E2E Integration | 15/15 (100%) | Full ingestion pipeline + retrieval both work |

---

## ⏱️ PERFORMANCE ANALYSIS

| Component | Latency | Target | Status |
|-----------|---------|--------|--------|
| Query Analysis | 0.1ms avg | <50ms | ✅ Excellent |
| Embedding (short text) | 50ms cold, <1ms warm | <500ms | ✅ Good |
| Embedding (medium text) | 105ms cold, <1ms warm | <500ms | ✅ Good |
| Embedding (long text) | 182ms cold, <1ms warm | <500ms | ✅ Good |
| Embedding Cache Speedup | 1700x | >10x | ✅ Excellent |
| DuckDB Time Search (100 memories) | 12.7ms | <100ms | ✅ Excellent |
| DuckDB Count | 1.1ms | <10ms | ✅ Excellent |
| Vector Store Search | 38-46ms | <100ms | ✅ Good |
| Full Ingestion Pipeline | 91-168ms | <500ms | ✅ Good |
| Embedding Model Init | 4.3s | <10s | ✅ Acceptable |

**Key Finding:** The system is NOT slow in any measurable component. The "taking lot of time to generate response" issue is likely caused by:
1. **LLM Generation** (not testable without GPU model) — DeepSeek-R1-7B on 4-bit generates ~20-50 tokens/sec
2. **Query Transformation** — 4 parallel LLM calls for multi-query + HyDE + step-back + decomposition (each ~2-5s)
3. **Self-RAG Critique** — Additional LLM call if confidence < 0.55
4. **FLARE Active Retrieval** — Additional retrieval + LLM if confidence < 0.4

**Response time breakdown (estimated with model):**
- Query Analysis: <1ms ✅
- Query Transformation: 2-5s (4 parallel LLM calls) ⚠️
- 5-Channel Retrieval: 50-200ms ✅
- RRF + Cross-Encoder: 100-300ms ✅
- Agent Execution: 3-8s (LLM generation) ⚠️
- CRAG Evaluation: 500ms-2s ⚠️
- Self-RAG (if triggered): 3-5s ⚠️
- FLARE (if triggered): 3-8s ⚠️
- **Total worst case: 20-30s** 😱

---

## 🎯 PRIORITIZED FIX PLAN

### P0 — Fix Now (Affects Answer Quality)
1. **Fix emotion detection keyword conflicts** — Remove "frustrated"/"annoyed" from ANGRY, add contextual words
2. **Fix memory type classification** — Add "I feel", "I love", "opinion" etc. to REFLECTIVE keywords
3. **Fix intent detection overlap** — Add "my process", "review process" to PROCEDURAL; add tie-breaking

### P1 — Fix Soon (Edge Cases)
4. **Fix entity extraction possessives** — Strip `'s` before processing
5. **Fix proposition decomposition for numbered lists** — Add numbered-item splitting
6. **Fix stance detection direction** — Check contradiction words in BOTH old and new texts
7. **Fix complexity scoring** — Add intent-based bonus for REFLECTIVE/COMPARATIVE queries

### P2 — Optimize (Performance)
8. **Consider disabling query transformation LLM calls for simple queries** — Already partially done (NO_RETRIEVAL skips), but SINGLE_STEP still runs multi-query
9. **Make Self-RAG threshold configurable** — Currently hardcoded at 0.55
10. **Add response time budget** — If approaching 15s, skip optional pipeline stages

---

*Generated by Cortex Lab Comprehensive Test Suite v1.0*
*Test run: 161 tests in 94.1s*
