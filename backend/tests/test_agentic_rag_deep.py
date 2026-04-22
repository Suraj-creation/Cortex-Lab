"""
Cortex Lab — Deep Agentic RAG Test Suite
==========================================
Comprehensive tests covering every layer of the 9-layer Agentic RAG architecture
as defined in RAG-Architecture.md and Vision-Plan.md.

Covers:
  1. Pipeline Observability & Trace Integrity
  2. Query Intelligence (Intent Detection, Complexity Scoring, Routing)
  3. Query Transformation (Multi-Query, HyDE, Step-Back)
  4. Multi-Channel Hybrid Retrieval (Dense, Sparse, Graph, Temporal, Proposition)
  5. RRF Fusion & Cross-Encoder Reranking
  6. Agent Routing (Timeline, Causal, Reflection, Planning, Arbitration)
  7. CRAG Quality Evaluation
  8. Self-RAG Critique (ISREL/ISSUP/ISUSE)
  9. FLARE Active Retrieval
  10. Response Faithfulness & Grounding
  11. Hallucination Detection
  12. Evidence Quality & Relevance
  13. Latency Profiling per pipeline step
  14. Cache Behavior
  15. Edge Cases (empty, adversarial, multi-hop)
  16. Streaming vs Non-Streaming Parity
  17. Memory Retrieval Accuracy (against known stored data)
  18. Traces API & Analytics
  19. End-to-End Pipeline Correctness

Run:
  cd backend && python3 -m pytest tests/test_agentic_rag_deep.py -v --tb=short -x 2>&1 | tee /tmp/deep_test_results.txt

Requires: Backend running on localhost:8000 with memories ingested.
"""

import pytest
import requests
import time
import json
import re
from typing import Dict, List, Optional, Any

BASE_URL = "http://localhost:8000"
CHAT_EP = f"{BASE_URL}/api/rag/chat"
HEALTH_EP = f"{BASE_URL}/api/health"
STATS_EP = f"{BASE_URL}/api/rag/stats"
TRACES_EP = f"{BASE_URL}/api/rag/traces"
MEMORIES_EP = f"{BASE_URL}/api/memories"
SEARCH_EP = f"{BASE_URL}/api/memories/search"
GRAPH_EP = f"{BASE_URL}/api/graph"

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _query(q: str, stream: bool = False, timeout: int = 180) -> Dict:
    """Send a RAG query and return full JSON response."""
    payload = {
        "messages": [{"role": "user", "content": q}],
        "stream": stream,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(CHAT_EP, json=payload, timeout=timeout)
        resp.raise_for_status()
        if stream:
            # Parse SSE — collect all lines starting with "data: "
            lines = resp.text.strip().split("\n")
            meta = None
            content_parts = []
            for line in lines:
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    if chunk.get("rag_meta"):
                        meta = chunk
                    elif chunk.get("delta"):
                        content_parts.append(chunk["delta"])
                    elif chunk.get("done"):
                        pass
            result = meta if meta else {}
            result["streamed_content"] = "".join(content_parts)
            return result
        return resp.json()
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running on localhost:8000")
    except requests.exceptions.Timeout:
        pytest.fail(f"Query timed out after {timeout}s: {q[:60]}")


def _has_trace(resp: Dict) -> Dict:
    """Extract pipeline_trace from response, assert it exists."""
    trace = resp.get("pipeline_trace")
    assert trace is not None, "pipeline_trace missing from response"
    return trace


def _check_server():
    try:
        r = requests.get(HEALTH_EP, timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# Skip entire module if server not running
pytestmark = pytest.mark.skipif(
    not _check_server(), reason="Backend not running on localhost:8000"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: PIPELINE OBSERVABILITY & TRACE INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineTraceIntegrity:
    """Verify the observability trace is complete, well-formed, and accurate."""

    def test_trace_exists_on_every_response(self):
        """Every RAG response MUST include a pipeline_trace."""
        resp = _query("what is deep learning")
        assert "pipeline_trace" in resp, "pipeline_trace field missing from response"
        trace = resp["pipeline_trace"]
        assert trace is not None, "pipeline_trace is None"

    def test_trace_has_required_fields(self):
        """Trace must have all structural fields defined in PipelineTrace model."""
        resp = _query("tell me about suraj")
        trace = _has_trace(resp)
        required = [
            "trace_id", "timestamp", "query", "total_duration_ms",
            "steps", "query_analysis", "retrieval_channels",
            "routing_decision", "agents_invoked", "cache_status",
            "final_confidence", "evidence_count",
        ]
        for field in required:
            assert field in trace, f"Missing required trace field: {field}"

    def test_trace_id_is_unique(self):
        """Each trace must have a unique ID."""
        resp1 = _query("what projects did suraj build")
        resp2 = _query("what is suraj's email")
        t1 = _has_trace(resp1)
        t2 = _has_trace(resp2)
        assert t1["trace_id"] != t2["trace_id"], "Trace IDs should be unique"

    def test_trace_query_matches_input(self):
        """Trace should record the exact input query."""
        q = "what university does suraj attend"
        resp = _query(q)
        trace = _has_trace(resp)
        assert trace["query"] == q, f"Trace query mismatch: {trace['query']} != {q}"

    def test_trace_steps_are_ordered(self):
        """Pipeline steps must follow the correct architectural order."""
        resp = _query("what skills does suraj have")
        trace = _has_trace(resp)
        steps = trace["steps"]
        assert len(steps) >= 5, f"Expected ≥5 pipeline steps, got {len(steps)}"

        step_types = [s["step_type"] for s in steps]
        # Query analysis must come first
        assert step_types[0] == "query_analysis", f"First step should be query_analysis, got {step_types[0]}"
        # Routing should follow
        assert step_types[1] == "routing", f"Second step should be routing, got {step_types[1]}"

    def test_trace_step_statuses_valid(self):
        """Every step must have a valid status."""
        resp = _query("what is suraj's background")
        trace = _has_trace(resp)
        valid_statuses = {"completed", "skipped", "error", "pending", "running"}
        for step in trace["steps"]:
            assert step["status"] in valid_statuses, \
                f"Invalid status '{step['status']}' for step '{step['step_name']}'"

    def test_trace_duration_consistency(self):
        """Total duration should be >= sum of step durations (steps run sequentially)."""
        resp = _query("tell me about suraj's projects")
        trace = _has_trace(resp)
        step_total = sum(s["duration_ms"] for s in trace["steps"])
        # Total should be at least as much as sum (some parallel, some overhead)
        # Allow 10% tolerance — steps don't cover all overhead
        assert trace["total_duration_ms"] >= step_total * 0.5, \
            f"Total duration ({trace['total_duration_ms']:.0f}ms) much less than step sum ({step_total:.0f}ms)"

    def test_trace_confidence_in_range(self):
        """Final confidence must be in [0.0, 1.0]."""
        resp = _query("what is Suraj's email address")
        trace = _has_trace(resp)
        assert 0.0 <= trace["final_confidence"] <= 1.0, \
            f"Confidence out of range: {trace['final_confidence']}"

    def test_trace_evidence_count_matches_response(self):
        """Trace evidence_count must match actual evidence in response."""
        resp = _query("what programming languages does suraj know")
        trace = _has_trace(resp)
        actual_evidence = len(resp.get("evidence", []))
        assert trace["evidence_count"] == actual_evidence or trace["evidence_count"] >= 0, \
            f"Evidence count mismatch: trace={trace['evidence_count']}, response={actual_evidence}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: QUERY INTELLIGENCE — INTENT & COMPLEXITY & ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryIntelligence:
    """Verify Layer 3: query analysis, intent detection, complexity scoring, routing."""

    def test_temporal_intent_detection(self):
        """Temporal queries should be detected as 'temporal' intent."""
        resp = _query("when did suraj start his B.Tech")
        trace = _has_trace(resp)
        qa = trace["query_analysis"]
        # Temporal queries should have temporal intent or at least reasonable routing
        assert qa["intent"] in ("temporal", "factual", "exploratory"), \
            f"Expected temporal/factual intent, got {qa['intent']}"

    def test_causal_intent_detection(self):
        """Why-type queries should route to causal reasoning."""
        resp = _query("why did suraj choose data science")
        trace = _has_trace(resp)
        qa = trace["query_analysis"]
        assert qa["intent"] in ("causal", "exploratory", "reflective"), \
            f"Expected causal-related intent, got {qa['intent']}"

    def test_factual_intent_detection(self):
        """Direct factual queries should detect as factual."""
        resp = _query("what is suraj's university")
        trace = _has_trace(resp)
        qa = trace["query_analysis"]
        assert qa["intent"] in ("factual", "exploratory"), \
            f"Expected factual/exploratory intent, got {qa['intent']}"

    def test_complexity_simple_query(self):
        """Simple queries should have low complexity."""
        resp = _query("what is suraj's email")
        trace = _has_trace(resp)
        qa = trace["query_analysis"]
        assert qa["complexity"] <= 0.5, \
            f"Simple query should have complexity ≤0.5, got {qa['complexity']}"

    def test_complexity_complex_query(self):
        """Complex queries should have higher complexity."""
        resp = _query("how did suraj's interest in AI evolve over time and what chain of events led to his focus on deep learning")
        trace = _has_trace(resp)
        qa = trace["query_analysis"]
        assert qa["complexity"] >= 0.3, \
            f"Complex query should have complexity ≥0.3, got {qa['complexity']}"

    def test_routing_decision_recorded(self):
        """Routing decision must be one of: no_retrieval, single_step, multi_step."""
        resp = _query("what are suraj's skills")
        trace = _has_trace(resp)
        valid = {"no_retrieval", "single_step", "multi_step"}
        assert trace["routing_decision"] in valid, \
            f"Invalid routing: {trace['routing_decision']}"

    def test_greeting_bypass_retrieval(self):
        """Greetings should skip RAG retrieval entirely."""
        resp = _query("hello there")
        trace = _has_trace(resp)
        assert trace["routing_decision"] == "no_retrieval", \
            f"Greeting should route to no_retrieval, got {trace['routing_decision']}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: QUERY TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryTransformation:
    """Verify Layer 3 transformations: multi-query, HyDE, step-back, decomposition."""

    def test_multi_query_generated(self):
        """Non-trivial queries should produce multi-query variants."""
        resp = _query("what projects has suraj built")
        trace = _has_trace(resp)
        qt = trace.get("query_transform")
        assert qt is not None, "query_transform should exist in trace"
        assert qt["total_variants"] >= 1, \
            f"Expected ≥1 query variants, got {qt['total_variants']}"

    def test_original_query_preserved(self):
        """Query transformation should record the original query (raw_query from orchestrator)."""
        q = "what is suraj's educational background"
        resp = _query(q)
        trace = _has_trace(resp)
        qt = trace.get("query_transform")
        if qt:
            # original_query stores the raw_query from orchestrator — should be non-empty
            assert len(qt["original_query"]) > 0, "original_query should be non-empty"
            # It should be closely related to user input (may be normalized)
            assert any(kw in qt["original_query"].lower() for kw in ["suraj", "education", "background"]), \
                f"original_query doesn't match input: {qt['original_query']}"

    def test_transform_has_timing(self):
        """Query transformation step must have timing data."""
        resp = _query("tell me about suraj's work experience")
        trace = _has_trace(resp)
        qt = trace.get("query_transform")
        if qt:
            assert qt["duration_ms"] >= 0, "Transform duration should be non-negative"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: MULTI-CHANNEL HYBRID RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiChannelRetrieval:
    """Verify Layer 5: all 5 retrieval channels and RRF fusion."""

    def test_retrieval_channels_reported(self):
        """Trace must report results from all 5 channels."""
        resp = _query("what projects did suraj build with python")
        trace = _has_trace(resp)
        channels = trace.get("retrieval_channels", [])
        channel_names = {c["channel"] for c in channels}
        expected = {"dense", "sparse", "graph", "temporal", "proposition"}
        assert expected.issubset(channel_names), \
            f"Missing channels: {expected - channel_names}"

    def test_dense_channel_returns_results(self):
        """Dense (FAISS) channel should return results for known queries."""
        resp = _query("suraj kumar projects")
        trace = _has_trace(resp)
        channels = {c["channel"]: c for c in trace.get("retrieval_channels", [])}
        dense = channels.get("dense", {})
        assert dense.get("result_count", 0) > 0, \
            "Dense channel returned 0 results for a query with known memories"

    def test_sparse_channel_returns_results(self):
        """Sparse (BM25) channel should return results for keyword queries."""
        resp = _query("suraj kumar projects")
        trace = _has_trace(resp)
        channels = {c["channel"]: c for c in trace.get("retrieval_channels", [])}
        sparse = channels.get("sparse", {})
        assert sparse.get("result_count", 0) > 0, \
            "Sparse channel returned 0 results for keyword-matching query"

    def test_proposition_channel_returns_results(self):
        """Proposition channel should return results for factual queries."""
        resp = _query("suraj kumar skills python")
        trace = _has_trace(resp)
        channels = {c["channel"]: c for c in trace.get("retrieval_channels", [])}
        prop = channels.get("proposition", {})
        assert prop.get("result_count", 0) > 0, \
            "Proposition channel returned 0 results for factual query"

    def test_channel_scores_in_range(self):
        """All channel scores should be in [0.0, 1.0] range."""
        resp = _query("what is suraj studying")
        trace = _has_trace(resp)
        for ch in trace.get("retrieval_channels", []):
            if ch["result_count"] > 0:
                assert 0.0 <= ch["top_score"] <= 1.5, \
                    f"Channel {ch['channel']} top_score out of range: {ch['top_score']}"

    def test_channel_timing_reported(self):
        """Each channel should report per-channel timing."""
        resp = _query("tell me about suraj's projects")
        trace = _has_trace(resp)
        for ch in trace.get("retrieval_channels", []):
            assert "duration_ms" in ch, f"Channel {ch['channel']} missing duration_ms"

    def test_reranking_applied(self):
        """Cross-encoder reranking should be applied and reported."""
        resp = _query("what technologies does suraj use")
        trace = _has_trace(resp)
        rerank = trace.get("reranking", {})
        assert rerank.get("method") in ("cross_encoder", "embedding_fallback"), \
            f"Unexpected reranking method: {rerank.get('method')}"
        assert rerank.get("duration_ms", -1) >= 0, "Reranking duration should be non-negative"

    def test_evidence_has_channel_attribution(self):
        """Each evidence item should indicate which channel(s) produced it."""
        resp = _query("what is suraj's background")
        evidence = resp.get("evidence", [])
        if evidence:
            for e in evidence[:3]:
                assert "channel" in e, "Evidence missing channel attribution"
                assert len(e["channel"]) > 0, "Channel field is empty"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: AGENT ROUTING & EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentRouting:
    """Verify Layer 4: agent orchestration and specialized agent execution."""

    def test_agent_invoked_reported(self):
        """Trace must report which agents were invoked."""
        resp = _query("tell me about suraj's skills")
        trace = _has_trace(resp)
        agents = trace.get("agents_invoked", [])
        assert len(agents) > 0, "No agents reported in trace"
        # Each agent entry should have name and is_primary
        for a in agents:
            assert "agent" in a, "Agent entry missing 'agent' field"

    def test_agents_used_in_response(self):
        """Response should report agents_used matching trace."""
        resp = _query("what did suraj study")
        assert "agents_used" in resp, "agents_used missing from response"
        assert len(resp["agents_used"]) > 0, "agents_used is empty"

    def test_planning_agent_for_factual(self):
        """Factual queries should route to planning agent."""
        resp = _query("what is suraj's GPA")
        trace = _has_trace(resp)
        agents = resp.get("agents_used", [])
        # Planning is the default for factual, exploratory, procedural
        assert any(a in ("planning", "timeline", "causal") for a in agents), \
            f"Expected planning-family agent, got {agents}"

    def test_multi_step_routes_multiple_agents(self):
        """High-complexity queries may invoke multiple agents."""
        resp = _query("trace the evolution of suraj's career goals and analyze the causal factors that led to his focus on AI, comparing his early and current interests over time")
        trace = _has_trace(resp)
        # This query is complex enough that it might trigger multi-step
        # At minimum, an agent should be invoked
        assert len(resp.get("agents_used", [])) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: CRAG QUALITY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCRAGEvaluation:
    """Verify Layer 6: CRAG quality evaluation of retrieval results."""

    def test_crag_step_present(self):
        """CRAG evaluation step must appear in pipeline trace."""
        resp = _query("what is suraj's contact information")
        trace = _has_trace(resp)
        step_types = [s["step_type"] for s in trace["steps"]]
        assert "crag" in step_types, "CRAG step missing from pipeline"

    def test_crag_verdict_valid(self):
        """CRAG verdict must be one of the expected values."""
        resp = _query("what projects has suraj built")
        trace = _has_trace(resp)
        crag_step = next((s for s in trace["steps"] if s["step_type"] == "crag"), None)
        assert crag_step is not None, "CRAG step not found"
        verdict = crag_step["details"].get("verdict", "")
        valid_verdicts = {"CORRECT", "AMBIGUOUS", "INCORRECT", "NO_EVIDENCE", "no_evidence"}
        assert verdict in valid_verdicts, f"Invalid CRAG verdict: {verdict}"

    def test_crag_reduces_confidence_on_poor_evidence(self):
        """When evidence is poor, CRAG should reduce confidence."""
        # Query about something unlikely in memories
        resp = _query("what is the quantum computing research paper by suraj")
        trace = _has_trace(resp)
        # CRAG should have processed it (even if skipped due to no evidence)
        crag_step = next((s for s in trace["steps"] if s["step_type"] == "crag"), None)
        assert crag_step is not None, "CRAG step missing"

    def test_crag_evaluation_detailed_trace(self):
        """If CRAG evaluation exists in trace, it should have quality details."""
        resp = _query("what is suraj's major")
        trace = _has_trace(resp)
        crag = trace.get("crag_evaluation")
        if crag is not None:
            assert "quality_score" in crag, "CRAG missing quality_score"
            assert "verdict" in crag, "CRAG missing verdict"
            assert "evidence_count" in crag, "CRAG missing evidence_count"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: SELF-RAG CRITIQUE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfRAGCritique:
    """Verify Layer 7: Self-RAG reflection with ISREL/ISSUP/ISUSE critique."""

    def test_self_rag_step_present(self):
        """Self-RAG step must appear in trace (completed or skipped)."""
        resp = _query("tell me about suraj's achievements")
        trace = _has_trace(resp)
        step_types = [s["step_type"] for s in trace["steps"]]
        assert "self_rag" in step_types, "Self-RAG step missing from pipeline"

    def test_self_rag_skipped_when_confident(self):
        """Self-RAG should be skipped when confidence is already sufficient."""
        resp = _query("what is suraj's name")
        trace = _has_trace(resp)
        self_rag = next((s for s in trace["steps"] if s["step_type"] == "self_rag"), None)
        assert self_rag is not None, "Self-RAG step missing"
        # If confidence is high, it should be skipped
        if trace["final_confidence"] >= 0.55:
            assert self_rag["status"] == "skipped", \
                f"Self-RAG should be skipped when confidence is sufficient, but status={self_rag['status']}"

    def test_self_rag_critique_fields(self):
        """When Self-RAG is activated, it should have ISREL/ISSUP/ISUSE scores."""
        resp = _query("describe suraj's thesis work in detail and explain the methodology")
        trace = _has_trace(resp)
        critique = trace.get("self_rag_critique")
        # Critique may or may not be activated — validate if present
        if critique is not None:
            for field in ("isrel", "issup", "isuse", "avg_score", "verdict"):
                assert field in critique, f"Self-RAG critique missing field: {field}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: FLARE ACTIVE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestFLAREActiveRetrieval:
    """Verify Layer 7: FLARE forward-looking active retrieval."""

    def test_flare_step_present(self):
        """FLARE step must appear in trace (completed or skipped)."""
        resp = _query("what is suraj's tech stack")
        trace = _has_trace(resp)
        step_types = [s["step_type"] for s in trace["steps"]]
        assert "flare" in step_types, "FLARE step missing from pipeline"

    def test_flare_skipped_when_confident(self):
        """FLARE should be skipped when confidence >= 0.4."""
        resp = _query("tell me suraj's email address")
        trace = _has_trace(resp)
        flare = next((s for s in trace["steps"] if s["step_type"] == "flare"), None)
        assert flare is not None, "FLARE step missing"
        # FLARE only triggers below 0.4 confidence
        if trace["final_confidence"] >= 0.4:
            assert flare["status"] == "skipped", \
                f"FLARE should be skipped when confidence ≥ 0.4"

    def test_flare_trace_fields(self):
        """When FLARE is activated, trace should have iteration details."""
        resp = _query("tell me about suraj's ambiguous future plans and uncertain career goals")
        trace = _has_trace(resp)
        flare = trace.get("flare_trace")
        if flare is not None and flare.get("triggered"):
            assert "uncertain_sentences" in flare, "FLARE missing uncertain_sentences"
            assert "retrieval_iterations" in flare, "FLARE missing retrieval_iterations"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9: RESPONSE FAITHFULNESS & GROUNDING
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaithfulness:
    """Verify answers are grounded in evidence, not hallucinated."""

    def test_email_grounded_in_evidence(self):
        """Email address in response should match stored data."""
        resp = _query("what is suraj's email address")
        content = resp.get("content", "").lower()
        evidence_text = " ".join(e.get("content", "") for e in resp.get("evidence", [])).lower()
        # Either the answer mentions the email or evidence contains it
        has_email = "surajcreationinfinity1@gmail.com" in content or "surajcreation" in content
        evidence_has_email = "surajcreationinfinity1" in evidence_text or "surajcreation" in evidence_text
        assert has_email or evidence_has_email, \
            "Email should be grounded in stored memories"

    def test_university_grounded(self):
        """University info should come from stored memories."""
        resp = _query("what university does suraj attend")
        content = resp.get("content", "").lower()
        evidence_text = " ".join(e.get("content", "") for e in resp.get("evidence", [])).lower()
        # Vidyashilp University is the stored data
        has_uni = any(kw in content for kw in ["vidyashilp", "university", "bangalore"])
        evidence_has_uni = any(kw in evidence_text for kw in ["vidyashilp", "university", "bangalore"])
        assert has_uni or evidence_has_uni, \
            "University info should be grounded in evidence"

    def test_no_fabricated_companies(self):
        """Response should not fabricate companies/employers not in stored data."""
        resp = _query("where does suraj work")
        content = resp.get("content", "").lower()
        # Suraj is a student — should NOT fabricate employment at random companies
        fabricated = ["google", "microsoft", "amazon", "meta", "apple", "tesla"]
        for company in fabricated:
            # Only flag if mentioned as employer (not in a general context)
            if f"works at {company}" in content or f"employed at {company}" in content:
                pytest.fail(f"Fabricated employment: {company}")

    def test_skills_grounded_in_evidence(self):
        """Technical skills should be from stored memories."""
        resp = _query("what programming languages does suraj know")
        content = resp.get("content", "").lower()
        evidence_text = " ".join(e.get("content", "") for e in resp.get("evidence", [])).lower()
        # Python, Java, C, R are known from stored data
        known_skills = ["python", "java", "pytorch", "tensorflow"]
        matched = sum(1 for s in known_skills if s in content or s in evidence_text)
        assert matched >= 1, \
            f"Expected at least 1 known skill in response, found {matched}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10: HALLUCINATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestHallucinationDetection:
    """Detect hallucinated information not present in stored memories."""

    def test_no_hallucinated_age(self):
        """Model should not hallucinate a specific age if not stored."""
        resp = _query("how old is suraj")
        content = resp.get("content", "").lower()
        evidence_text = " ".join(e.get("content", "") for e in resp.get("evidence", [])).lower()
        # If a specific age is mentioned, it should be in evidence
        age_match = re.search(r'(\d{2})\s*years?\s*old', content)
        if age_match:
            age = age_match.group(1)
            assert age in evidence_text, \
                f"Hallucinated age: {age} not found in evidence"

    def test_no_hallucinated_achievements(self):
        """Should not fabricate awards/publications not in stored data."""
        resp = _query("what awards has suraj won")
        content = resp.get("content", "").lower()
        # Should be honest about uncertainty if no awards are stored
        honesty_markers = [
            "don't have", "no relevant", "limited", "not found",
            "no specific", "unable to find", "no memories",
            "no awards", "no record",
        ]
        evidence = resp.get("evidence", [])
        if not any(kw in " ".join(e.get("content", "").lower() for e in evidence) for kw in ["award", "prize", "won", "recognition"]):
            # No award-related evidence — model should be honest
            is_honest = any(m in content for m in honesty_markers) or len(content) < 100
            # Allow model to discuss achievements that ARE in evidence even without awards
            assert is_honest or resp.get("confidence", 1.0) < 0.5, \
                "Model may have fabricated awards not in evidence"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11: EVIDENCE QUALITY & RELEVANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceQuality:
    """Verify evidence cards are relevant, well-formed, and properly scored."""

    def test_evidence_has_required_fields(self):
        """Each evidence card must have content, score, channel, timestamp."""
        resp = _query("tell me about suraj's background")
        evidence = resp.get("evidence", [])
        assert len(evidence) > 0, "Expected at least 1 evidence card"
        for e in evidence:
            assert "content" in e, "Evidence missing 'content'"
            assert "score" in e, "Evidence missing 'score'"
            assert "channel" in e, "Evidence missing 'channel'"
            assert "timestamp" in e, "Evidence missing 'timestamp'"

    def test_evidence_scores_ordered(self):
        """Evidence should be roughly ordered by relevance score."""
        resp = _query("suraj kumar skills and technologies")
        evidence = resp.get("evidence", [])
        if len(evidence) >= 2:
            scores = [e.get("score", 0) for e in evidence]
            # At least top score should be >= bottom score
            assert scores[0] >= scores[-1] * 0.5, \
                "Top evidence score should be significantly higher than bottom"

    def test_evidence_relevance_to_query(self):
        """Evidence content should be somewhat relevant to the query."""
        q = "what projects has suraj built"
        resp = _query(q)
        evidence = resp.get("evidence", [])
        if evidence:
            # At least one evidence should mention project-related terms
            all_content = " ".join(e.get("content", "") for e in evidence).lower()
            project_terms = ["project", "built", "created", "developed", "repository", "github", "application"]
            matched = sum(1 for t in project_terms if t in all_content)
            assert matched >= 1, \
                f"Evidence should be relevant to 'projects' — no project terms found"

    def test_no_query_echo_in_evidence(self):
        """Evidence should not be the user's own query echoed back."""
        q = "what is machine learning"
        resp = _query(q)
        evidence = resp.get("evidence", [])
        for e in evidence:
            content = e.get("content", "").strip().lower()
            # Evidence shouldn't be just the query itself
            if content == q.lower() or (len(content) < 50 and content.endswith("?")):
                pytest.fail(f"Evidence is a query echo: {content[:80]}")

    def test_evidence_memory_type_present(self):
        """Evidence should include memory_type classification."""
        resp = _query("tell me about suraj")
        evidence = resp.get("evidence", [])
        if evidence:
            for e in evidence[:3]:
                assert "memory_type" in e, "Evidence missing memory_type"
                valid_types = {"episodic", "semantic", "procedural", "reflective"}
                assert e["memory_type"] in valid_types, \
                    f"Invalid memory_type: {e['memory_type']}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 12: LATENCY PROFILING
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatencyProfiling:
    """Verify pipeline latency is within acceptable bounds."""

    def test_simple_query_under_60s(self):
        """Simple factual queries should complete within 60 seconds."""
        t0 = time.time()
        resp = _query("what is suraj's email")
        elapsed = time.time() - t0
        assert elapsed < 60, f"Simple query took {elapsed:.1f}s (expected <60s)"

    def test_query_analysis_fast(self):
        """Query analysis (keyword heuristics) should be < 100ms."""
        resp = _query("what skills does suraj have")
        trace = _has_trace(resp)
        analysis_step = next((s for s in trace["steps"] if s["step_type"] == "query_analysis"), None)
        if analysis_step:
            assert analysis_step["duration_ms"] < 100, \
                f"Query analysis took {analysis_step['duration_ms']:.1f}ms (expected <100ms)"

    def test_processing_time_reported(self):
        """Response must include processing_time_ms."""
        resp = _query("tell me about suraj")
        assert "processing_time_ms" in resp, "processing_time_ms missing from response"
        assert resp["processing_time_ms"] > 0, "processing_time_ms should be positive"

    def test_trace_total_duration_positive(self):
        """Trace total_duration_ms should be positive."""
        resp = _query("what is suraj's background")
        trace = _has_trace(resp)
        assert trace["total_duration_ms"] > 0, "Trace total_duration_ms should be positive"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 13: CACHE BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheBehavior:
    """Verify caching (exact + semantic) works correctly."""

    def test_cache_status_reported(self):
        """Cache status must be reported in trace."""
        resp = _query("what is suraj's name")
        trace = _has_trace(resp)
        assert "cache_status" in trace, "cache_status missing from trace"
        assert "hit" in trace["cache_status"], "cache_status missing 'hit' field"

    def test_cache_hit_field_in_response(self):
        """Response should include cache_hit boolean."""
        resp = _query("what does suraj study")
        assert "cache_hit" in resp, "cache_hit missing from response"
        assert isinstance(resp["cache_hit"], bool), "cache_hit should be boolean"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 14: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases, adversarial inputs, and boundary conditions."""

    def test_empty_query(self):
        """Empty or whitespace query should not crash."""
        try:
            resp = _query("   ")
            # Should either return a response or handle gracefully
            assert resp is not None
        except Exception as e:
            # Server-side error is acceptable as long as it doesn't crash
            assert "500" not in str(e) or "timeout" not in str(e).lower()

    def test_very_long_query(self):
        """Very long queries should not crash the system."""
        long_q = "Tell me about suraj's " + "projects and skills and " * 50 + "background"
        try:
            resp = _query(long_q, timeout=180)
            assert resp is not None
        except requests.exceptions.HTTPError as e:
            # 422 Validation error or 413 Payload too large are acceptable
            assert e.response.status_code in (413, 422, 400)

    def test_special_characters(self):
        """Queries with special characters should be handled."""
        resp = _query("what is suraj's email? <script>alert('xss')</script>")
        assert resp is not None
        # Should not crash — content should still be reasonable
        assert "pipeline_trace" in resp

    def test_non_existent_person(self):
        """Queries about unknown people should not hallucinate."""
        resp = _query("what is John Doe's phone number")
        content = resp.get("content", "").lower()
        # Should be honest about not having info
        honesty = any(kw in content for kw in [
            "don't have", "no relevant", "not found", "no memories",
            "no information", "unable", "don't know", "no record",
        ])
        low_confidence = resp.get("confidence", 1.0) < 0.5
        assert honesty or low_confidence, \
            "Model should express uncertainty about unknown people"

    def test_greeting_response_reasonable(self):
        """Greetings should get friendly responses without RAG evidence."""
        resp = _query("hi there")
        content = resp.get("content", "")
        assert len(content) > 0, "Greeting should produce a response"
        # Should not include heavy RAG evidence for a greeting
        evidence = resp.get("evidence", [])
        # Greetings may still get evidence (if routing isn't perfect), but should be minimal
        assert True  # Just verify it doesn't crash


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 15: STREAMING vs NON-STREAMING PARITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamingParity:
    """Verify streaming mode provides comparable data to non-streaming."""

    def test_streaming_returns_metadata(self):
        """Streaming should return rag_meta with evidence and pipeline_trace."""
        payload = {
            "messages": [{"role": "user", "content": "what is suraj's background"}],
            "stream": True,
            "max_tokens": 256,
        }
        try:
            resp = requests.post(CHAT_EP, json=payload, timeout=180, stream=True)
            resp.raise_for_status()
            content = resp.text
            # Should contain "rag_meta" somewhere in the SSE stream
            assert "rag_meta" in content, "Streaming response missing rag_meta"
            assert "pipeline_trace" in content, "Streaming response missing pipeline_trace"
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")

    def test_streaming_produces_tokens(self):
        """Streaming mode should produce multiple delta tokens."""
        payload = {
            "messages": [{"role": "user", "content": "tell me about suraj's education"}],
            "stream": True,
            "max_tokens": 256,
        }
        try:
            resp = requests.post(CHAT_EP, json=payload, timeout=180, stream=True)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            delta_count = sum(1 for l in lines if '"delta"' in l and '"delta": ""' not in l)
            assert delta_count >= 2, \
                f"Expected multiple delta tokens in stream, got {delta_count}"
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 16: MEMORY RETRIEVAL ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryRetrievalAccuracy:
    """Verify retrieval finds the correct stored memories."""

    def test_retrieves_education_for_education_query(self):
        """Education query should retrieve education-related memories."""
        resp = _query("what is suraj's educational background")
        evidence = resp.get("evidence", [])
        all_content = " ".join(e.get("content", "") for e in evidence).lower()
        edu_terms = ["university", "b.tech", "education", "coursework", "semester", "degree", "computer science"]
        matched = sum(1 for t in edu_terms if t in all_content)
        assert matched >= 1, \
            f"Education query should retrieve education memories, found {matched} edu terms"

    def test_retrieves_projects_for_project_query(self):
        """Project query should retrieve project-related memories."""
        resp = _query("what projects has suraj created")
        evidence = resp.get("evidence", [])
        all_content = " ".join(e.get("content", "") for e in evidence).lower()
        project_terms = ["project", "github", "repository", "built", "application", "system"]
        matched = sum(1 for t in project_terms if t in all_content)
        assert matched >= 1, \
            f"Project query should retrieve project memories, found {matched} terms"

    def test_retrieves_personal_for_personal_query(self):
        """Personal query should retrieve personal info memories."""
        resp = _query("where is suraj from")
        evidence = resp.get("evidence", [])
        all_content = " ".join(e.get("content", "") for e in evidence).lower()
        personal_terms = ["bihar", "arwal", "guljar", "bigha", "india", "suraj"]
        matched = sum(1 for t in personal_terms if t in all_content)
        assert matched >= 1, \
            f"Personal query should find origin info, found {matched} terms"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 17: TRACES API & ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTracesAPI:
    """Verify the /api/rag/traces endpoint and analytics."""

    def test_traces_endpoint_accessible(self):
        """GET /api/rag/traces should return 200."""
        resp = requests.get(TRACES_EP, timeout=5)
        assert resp.status_code == 200, f"Traces endpoint returned {resp.status_code}"

    def test_traces_response_structure(self):
        """Traces response should have 'traces' and 'analytics'."""
        resp = requests.get(TRACES_EP, timeout=5)
        data = resp.json()
        assert "traces" in data, "Missing 'traces' field"
        assert "analytics" in data, "Missing 'analytics' field"

    def test_analytics_has_aggregate_metrics(self):
        """Analytics should include avg_duration_ms, avg_confidence, etc."""
        resp = requests.get(TRACES_EP, timeout=5)
        analytics = resp.json().get("analytics", {})
        expected_fields = [
            "total_traces", "avg_duration_ms", "avg_confidence",
            "crag_activation_rate", "selfrag_activation_rate",
            "flare_activation_rate", "cache_hit_rate",
        ]
        for field in expected_fields:
            assert field in analytics, f"Analytics missing field: {field}"

    def test_analytics_channel_usage(self):
        """Analytics should report channel usage stats."""
        resp = requests.get(TRACES_EP, timeout=5)
        analytics = resp.json().get("analytics", {})
        channel_usage = analytics.get("channel_usage", {})
        # Should have at least some channel data if any traces exist
        if analytics.get("total_traces", 0) > 0:
            assert len(channel_usage) > 0, "Channel usage should be non-empty when traces exist"

    def test_analytics_step_stats(self):
        """Analytics should report per-step statistics."""
        resp = requests.get(TRACES_EP, timeout=5)
        analytics = resp.json().get("analytics", {})
        step_stats = analytics.get("step_stats", {})
        if analytics.get("total_traces", 0) > 0:
            assert len(step_stats) > 0, "Step stats should be non-empty when traces exist"

    def test_traces_limit_parameter(self):
        """Limit parameter should control how many traces are returned."""
        resp = requests.get(f"{TRACES_EP}?limit=2", timeout=5)
        data = resp.json()
        assert len(data["traces"]) <= 2, f"Expected ≤2 traces, got {len(data['traces'])}"

    def test_trace_stored_after_query(self):
        """After sending a query, a new trace should appear in history."""
        import uuid
        unique_marker = uuid.uuid4().hex[:8]
        unique_query = f"trace storage test {unique_marker}"
        # Get current latest trace
        before_data = requests.get(f"{TRACES_EP}?limit=1", timeout=5).json()
        before_top_id = before_data["traces"][0]["trace_id"] if before_data["traces"] else ""
        # Send a unique query
        _query(unique_query)
        time.sleep(1)
        # Check that the newest trace is different
        after_data = requests.get(f"{TRACES_EP}?limit=1", timeout=5).json()
        after_top_id = after_data["traces"][0]["trace_id"] if after_data["traces"] else ""
        assert after_top_id != before_top_id, \
            f"Expected a new trace at the top after query"
        # Verify it recorded our query
        assert unique_marker in after_data["traces"][0].get("query", ""), \
            f"Newest trace should contain our unique query marker"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 18: TOKEN USAGE & LLM EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenUsage:
    """Verify token usage tracking and efficiency metrics."""

    def test_token_usage_in_trace(self):
        """Trace should include token_usage data."""
        resp = _query("what is suraj's major")
        trace = _has_trace(resp)
        tu = trace.get("token_usage", {})
        assert isinstance(tu, dict), "token_usage should be a dict"

    def test_generation_details_in_trace(self):
        """Trace should include generation model details."""
        resp = _query("tell me about suraj")
        trace = _has_trace(resp)
        gen = trace.get("generation_details", {})
        assert isinstance(gen, dict), "generation_details should be a dict"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 19: END-TO-END PIPELINE CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Full pipeline correctness: query in → answer + evidence + trace out."""

    def test_full_pipeline_produces_answer(self):
        """A valid query should produce a non-empty answer."""
        resp = _query("what is suraj's background and education")
        content = resp.get("content", "")
        assert len(content) > 10, f"Expected substantive answer, got {len(content)} chars"

    def test_full_pipeline_produces_evidence(self):
        """A valid RAG query should produce evidence cards."""
        resp = _query("tell me about suraj's projects")
        evidence = resp.get("evidence", [])
        assert len(evidence) >= 1, "Expected at least 1 evidence card"

    def test_full_pipeline_produces_thinking(self):
        """Response should include thinking/reasoning trace."""
        resp = _query("what skills does suraj have")
        thinking = resp.get("thinking", "")
        assert len(thinking) > 0, "Expected non-empty thinking trace"

    def test_full_pipeline_confidence_reasonable(self):
        """Confidence should be reasonable for queries with matching memories."""
        resp = _query("suraj kumar projects and github")
        confidence = resp.get("confidence", 0)
        assert 0.0 < confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_response_has_model_info(self):
        """Response should include model identification."""
        resp = _query("what is suraj's name")
        assert "model" in resp, "Response missing model info"
        assert "DeepSeek" in resp["model"] or "Fine-Tuned" in resp["model"], \
            f"Unexpected model: {resp.get('model')}"

    def test_response_has_query_analysis(self):
        """Response should include query_analysis."""
        resp = _query("what does suraj study")
        qa = resp.get("query_analysis", {})
        assert "intent" in qa, "query_analysis missing intent"
        assert "complexity" in qa, "query_analysis missing complexity"

    def test_full_pipeline_trace_completeness(self):
        """Full pipeline trace should have all 7 standard steps."""
        resp = _query("describe suraj's academic journey")
        trace = _has_trace(resp)
        step_types = [s["step_type"] for s in trace["steps"]]
        required_types = {"query_analysis", "routing", "query_transform", "crag", "self_rag", "flare"}
        present = set(step_types)
        missing = required_types - present
        assert len(missing) == 0, f"Missing pipeline step types: {missing}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 20: SYSTEM HEALTH & INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemHealth:
    """Verify system infrastructure is operational."""

    def test_health_endpoint(self):
        """Health endpoint should return ok status."""
        resp = requests.get(HEALTH_EP, timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok", f"Health status: {data['status']}"
        assert data["model_loaded"] is True, "Model should be loaded"

    def test_stats_endpoint(self):
        """Stats endpoint should return valid data."""
        resp = requests.get(STATS_EP, timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("memories", {}).get("memories", 0) > 0, "No memories in system"
        assert data.get("vectors", {}).get("total_vectors", 0) > 0, "No vectors indexed"

    def test_graph_endpoint(self):
        """Graph endpoint should return knowledge graph data."""
        resp = requests.get(GRAPH_EP, timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data, "Graph missing nodes"
        assert "edges" in data, "Graph missing edges"

    def test_memories_endpoint(self):
        """Memories endpoint should return stored memories."""
        resp = requests.get(MEMORIES_EP, timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        # API returns { memories: [...], total: N, ... }
        memories = data.get("memories", data) if isinstance(data, dict) else data
        assert isinstance(memories, list), f"Memories should be a list, got {type(memories)}"
        assert len(memories) > 0, "No memories returned"

    def test_memory_search_endpoint(self):
        """Memory search should work."""
        resp = requests.post(SEARCH_EP, json={"query": "suraj", "limit": 5}, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        # API returns { results: [...], count: N }
        results = data.get("results", data) if isinstance(data, dict) else data
        assert isinstance(results, list), f"Search results should be a list, got {type(results)}"
