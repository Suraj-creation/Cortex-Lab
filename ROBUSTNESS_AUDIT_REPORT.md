# Cortex Lab — Agentic RAG Robustness Audit Report

## Executive Summary

Deep analysis of all Agentic RAG components revealed **12 robustness issues** across 6 subsystems.
The most critical finding: the education vision query ("what is my core vision about changing education system") 
returns only 209 characters truncated mid-word because `_try_extract_factual()` falsely triggers on the word 
"education" and returns a raw evidence snippet instead of letting the LLM generate a comprehensive answer.

---

## Issue #1 — CRITICAL: `_try_extract_factual` False Positive on Complex Queries

**File:** `backend/server.py` → `_try_extract_factual()` (line ~670)  
**Severity:** CRITICAL — Completely breaks synthesis/vision queries  
**Root Cause:** The education extraction check matches on `"education"` keyword without considering query intent.  
The query "what is my core vision about changing education system" contains "education" → triggers the education 
background extractor → regex `(?:EDUCATION|Education)[:\s]*\n?(.{20,200})` grabs a raw snippet from evidence → 
returns 209 chars truncated mid-word (`"should b"`), bypassing the LLM entirely.

**Fix:** Add synthesis/philosophical query guard. If the query contains words like "vision", "philosophy", 
"system", "paradigm", "changing", "core belief" etc., skip the factual extraction entirely and let the LLM 
generate a comprehensive answer.

---

## Issue #2 — HIGH: No Vision/Synthesis Intent Classification

**File:** `backend/src/retrieval/query_engine.py` → `INTENT_KEYWORDS` (line ~34)  
**Severity:** HIGH — Vision queries get lowest-priority EXPLORATORY intent  
**Root Cause:** No keywords exist for educational philosophy, vision, synthesis, or worldview queries.  
"vision", "dream", "philosophy", "aspiration", "paradigm" are missing from all intent categories.

**Impact:**
- "what is my core vision about changing education system" → EXPLORATORY (catch-all, -0.01 priority)
- EXPLORATORY maps to `planning` agent (not ideal — should be `reflection`)
- Complexity scores only ~0.35 → SINGLE_STEP routing (too shallow)

**Fix:** Add vision/synthesis keywords to REFLECTIVE intent. Add complexity boosters for these keywords.

---

## Issue #3 — HIGH: Evidence Capped at 500 chars for ALL Query Types

**File:** `backend/server.py` → evidence collection loop (line ~1500)  
**Severity:** HIGH — Synthesis queries get insufficient context  
**Root Cause:** `evidence_texts.append(content[:500])` and `if len(evidence_texts) >= 7: break` are hardcoded constants regardless of query complexity.

**Impact:** A vision/synthesis query about the user's educational philosophy gets only 500 chars × 7 items = 3500 chars of evidence — far too little for a comprehensive answer.

**Fix:** Make evidence limits adaptive:
- Simple queries: 500 chars × 7 items (current)
- Complex/synthesis queries: 1500 chars × 12 items

---

## Issue #4 — MEDIUM: Gemini Streaming Path Uses Inline Prompt, Ignores PromptBuilder

**File:** `backend/server.py` → `_stream_gemini_rag_generate()` (line ~913)  
**Severity:** MEDIUM — Two inconsistent prompt systems  
**Root Cause:** The Gemini streaming path builds its own system prompt inline:
```python
system = "You are Cortex Lab... Speak warmly and conversationally..."
```
Meanwhile, the local model path uses `PromptBuilder.streaming_rag_generation()` which has more detailed 
instructions. The two are inconsistent, and neither has synthesis-specific guidance.

**Fix:** Make the Gemini streaming path aware of query complexity and use appropriate prompt variants. 
Add a synthesis prompt for complex queries.

---

## Issue #5 — MEDIUM: No Synthesis/Comprehensive Prompt Variant

**File:** `backend/src/prompts.py` → `streaming_rag_generation()` (line ~370)  
**Severity:** MEDIUM — LLM not guided to produce comprehensive answers  
**Root Cause:** `streaming_rag_generation()` only has two modes:
- "For simple questions (name, email, location), answer in ONE short sentence"
- "For broader questions (skills, projects, background), write a flowing natural paragraph"

There's no guidance for synthesis/vision/philosophical queries that need multi-paragraph comprehensive responses.

**Fix:** Add `synthesis_rag_generation()` prompt that instructs: comprehensive multi-paragraph response, 
cover all aspects from evidence, weave themes together, don't truncate.

---

## Issue #6 — MEDIUM: Complexity Boosters Missing Vision/Synthesis Keywords

**File:** `backend/src/retrieval/query_engine.py` → `COMPLEXITY_BOOSTERS` (line ~81)  
**Severity:** MEDIUM — Vision queries score too low on complexity  
**Root Cause:** Missing boosters: "vision", "dream", "philosophy", "values", "worldview", "aspiration", 
"core belief", "paradigm", "redefining", "reimagining", "fundamental", "ideology"

**Impact:** "what is my core vision about changing education system" → complexity ~0.35 → SINGLE_STEP 
(just barely above the 0.3 NO_RETRIEVAL threshold)

**Fix:** Add these keywords to COMPLEXITY_BOOSTERS.

---

## Issue #7 — LOW: EXPLORATORY Intent Maps to Planning Agent (Not Reflection)

**File:** `backend/src/agents/orchestrator.py` → `intent_to_agent` mapping  
**Severity:** LOW (in streaming mode, agent is used for routing context only, not generation)  
**Root Cause:** EXPLORATORY → 'planning' agent. For vision/philosophical queries that fall into 
EXPLORATORY, the 'reflection' agent would be more appropriate.

**Fix:** With Issue #2 fixed (vision queries → REFLECTIVE), this becomes moot. REFLECTIVE → 'reflection' agent.

---

## Issue #8 — LOW: Inconsistent Evidence Size Limits Across Paths

**File:** Multiple files  
**Severity:** LOW — Functional but inconsistent  
**Locations:**
- `server.py` streaming evidence: `content[:500]`, max 7 items
- `specialized.py` agent evidence: `content[:1500]`, max 5-10 items  
- `server.py` PageIndex evidence: `content[:2000]`, max 3 items
- `gemini_llm.py` generate_faithful: `evidence[:200]` chars in prompt

---

## Issue #9 — LOW: No Query-Type Awareness in Evidence Collection

**File:** `backend/server.py` → evidence collection (line ~1480)  
**Severity:** LOW — All queries get same evidence treatment  
**Root Cause:** Evidence collection doesn't receive the query analysis (intent, complexity, routing) 
from the orchestrator. It treats factual "what is my name" the same as historical "trace my career evolution".

**Fix:** Pass query complexity to evidence collection and adjust limits accordingly.

---

## Issue #10 — INFO: Agent-Level Tests Missing

**Current State:** No unit tests for individual agent execution (Timeline, Causal, Reflection, Planning, Arbitration).
Tests exist in `backend/tests/` but focus on integration, not individual agent quality.

**Recommendation:** Create `backend/tests/test_agents.py` with tests for each agent, verifying:
- Correct evidence retrieval
- Appropriate response length for query type
- Confidence scoring sanity
- Agent selection based on intent

---

## Issue #11 — INFO: Query Router Not Tested for Edge Cases

**Current State:** No tests for query classifier edge cases like:
- "vision" queries being misclassified as EXPLORATORY
- "education system" being confused with "education background"
- Multi-intent queries (e.g., "compare my vision about education with my actual projects")

---

## Issue #12 — INFO: Streaming RAG Has No Minimum Response Length Guard

**Current State:** No check that the streamed response exceeds a minimum length for complex queries.
A vision query should produce 500+ characters, but the system happily returns 209 chars.

**Recommendation:** After streaming completes, if the query was complex and the response is too short, 
consider a retry with more evidence or a different prompt strategy.

---

## Implementation Priority

| Priority | Issue | Component | Impact |
|----------|-------|-----------|--------|
| P0 | #1 | `_try_extract_factual` | Blocks all synthesis queries |
| P0 | #2 | Query intent detection | Wrong routing for vision queries |
| P1 | #3 | Evidence sizing | Insufficient context for synthesis |
| P1 | #4 | Gemini prompt path | Inconsistent prompt quality |
| P1 | #5 | PromptBuilder | No synthesis prompt |
| P1 | #6 | Complexity boosters | Too-low complexity for vision |
| P2 | #7-9 | Various | Consistency improvements |
| P3 | #10-12 | Testing | Quality assurance |

---

## Fixes Implemented

All P0 and P1 issues are fixed in this commit. See individual file changes for details.
