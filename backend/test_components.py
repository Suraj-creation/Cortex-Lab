"""
Comprehensive Agentic RAG Component Tests
Tests each component individually:
1. Query Classification (intent detection, complexity scoring, routing)
2. Evidence Collection (sizing, quality, filtering)
3. Factual Extraction (guard against false triggers)
4. Prompt Selection (standard vs synthesis)
5. End-to-End Streaming (multiple query types)
"""
import requests
import json
import sys
import time
import re

BASE = "http://localhost:8000"

def _rag_query(query, timeout=120):
    """Send a streaming RAG query and return (text, meta)."""
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/api/rag/chat", json={
                "messages": [{"role": "user", "content": query}],
                "stream": True,
                "llm_provider": "gemini"
            }, stream=True, timeout=timeout)
            break
        except requests.ConnectionError:
            if attempt < 2:
                time.sleep(3)
            else:
                return None, None

    text = ""
    meta = None
    for line in r.iter_lines():
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "rag_meta" in data:
                meta = data["rag_meta"]
            elif not data.get("done"):
                text += data.get("delta", "")
    return text, meta


def _check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}")
    if detail and not condition:
        print(f"           {detail}")
    return condition


# ═══════════════════════════════════════════════════════════════════════
#  Test 1: Query Classification
# ═══════════════════════════════════════════════════════════════════════
def test_query_classification():
    """Test that queries are classified with correct intent and complexity."""
    print("\n" + "=" * 70)
    print("  TEST 1: Query Classification")
    print("=" * 70)
    passed = 0
    total = 0

    # Test cases: (query, expected_intent, min_complexity)
    test_cases = [
        ("what is my core vision about changing education system", "reflective", 0.5),
        ("What is my name?", "factual", 0.2),
        ("Hey, how are you?", None, 0.0),  # greeting → NO_RETRIEVAL
        ("Why did I decide to build Cortex Lab?", "causal", 0.3),
        ("How has my thinking about AI evolved over time?", "reflective", 0.5),
        ("Compare my skills in Python vs JavaScript", "comparative", 0.5),
        ("When did I start working on deep learning?", "temporal", 0.3),
        ("Tell me about my philosophy on education", "reflective", 0.5),
        ("What is my dream about reimagining education?", "reflective", 0.5),
    ]

    for query, expected_intent, min_complexity in test_cases:
        total += 1
        text, meta = _rag_query(query)
        qa = meta.get("query_analysis", {}) if meta else {}
        intent = qa.get("intent", "unknown")
        complexity = qa.get("complexity", 0)

        if expected_intent is None:
            # Greeting — just check it responds
            ok = text is not None and len(text) > 5
            passed += _check(
                f"Greeting: '{query[:40]}...'",
                ok,
                f"Got: intent={intent}, complexity={complexity}"
            )
        else:
            intent_ok = intent == expected_intent
            complexity_ok = complexity >= min_complexity
            ok = intent_ok and complexity_ok
            passed += _check(
                f"'{query[:50]}...' → intent={intent} (expect {expected_intent}), complexity={complexity:.2f} (min {min_complexity})",
                ok,
                f"Intent match: {intent_ok}, Complexity match: {complexity_ok}"
            )
        time.sleep(1.5)

    print(f"\n  Classification: {passed}/{total} passed")
    return passed, total


# ═══════════════════════════════════════════════════════════════════════
#  Test 2: Evidence Quality
# ═══════════════════════════════════════════════════════════════════════
def test_evidence_quality():
    """Test that evidence collection is appropriate for query type."""
    print("\n" + "=" * 70)
    print("  TEST 2: Evidence Quality")
    print("=" * 70)
    passed = 0
    total = 0

    # Simple factual query → should have evidence
    total += 1
    text, meta = _rag_query("What is my name?")
    ev = meta.get("evidence", []) if meta else []
    passed += _check(
        f"Factual query evidence count: {len(ev)}",
        len(ev) >= 1,
        "Expected at least 1 evidence item for factual query"
    )
    time.sleep(1.5)

    # Complex synthesis query → should have MORE evidence
    total += 1
    text, meta = _rag_query("what is my core vision about changing education system")
    ev = meta.get("evidence", []) if meta else []
    passed += _check(
        f"Synthesis query evidence count: {len(ev)}",
        len(ev) >= 3,
        f"Expected at least 3 evidence items for synthesis query, got {len(ev)}"
    )
    time.sleep(1.5)

    # Greeting → should have no meaningful evidence
    total += 1
    text, meta = _rag_query("Hey there!")
    ev = meta.get("evidence", []) if meta else []
    conf = meta.get("confidence", 0) if meta else 0
    passed += _check(
        f"Greeting evidence: {len(ev)} items, response present: {bool(text)}",
        text is not None and len(text.strip()) > 5,
        "Greeting should still produce a response"
    )

    print(f"\n  Evidence Quality: {passed}/{total} passed")
    return passed, total


# ═══════════════════════════════════════════════════════════════════════
#  Test 3: Factual Extraction Guard
# ═══════════════════════════════════════════════════════════════════════
def test_factual_extraction_guard():
    """Test that _try_extract_factual doesn't false-trigger on synthesis queries."""
    print("\n" + "=" * 70)
    print("  TEST 3: Factual Extraction Guard")
    print("=" * 70)
    passed = 0
    total = 0

    # These should NOT trigger factual extraction (synthesis queries)
    synthesis_queries = [
        ("what is my core vision about changing education system", 300),
        ("tell me about my philosophy on transforming education", 200),
        ("what is my dream about reimagining the education paradigm", 200),
        ("how do I see the future of education system changing", 200),
    ]

    for query, min_len in synthesis_queries:
        total += 1
        text, meta = _rag_query(query)
        text_len = len(text) if text else 0
        # Synthesis answers should be longer than 300 chars (not a raw snippet)
        # Also check it doesn't start mid-sentence (no lowercase start without pronoun)
        starts_clean = text and (text[0].isupper() or text[0] in '"\'(')
        passed += _check(
            f"'{query[:50]}...' → {text_len} chars, starts_clean={starts_clean}",
            text_len >= min_len and starts_clean,
            f"Expected ≥{min_len} chars & clean start. Got {text_len} chars. Start: '{text[:50]}...'" if text else "No response"
        )
        time.sleep(2)

    # These SHOULD trigger factual extraction (simple queries)
    factual_queries = [
        ("What is my name?", ["suraj", "name"]),
        ("What is my email?", ["@", "email"]),
    ]

    for query, keywords in factual_queries:
        total += 1
        text, meta = _rag_query(query)
        found = any(kw.lower() in (text or "").lower() for kw in keywords)
        passed += _check(
            f"'{query}' → contains {keywords[0]}: {found}",
            found,
            f"Expected one of {keywords} in response. Got: {text[:100]}" if text else "No response"
        )
        time.sleep(1.5)

    print(f"\n  Factual Extraction Guard: {passed}/{total} passed")
    return passed, total


# ═══════════════════════════════════════════════════════════════════════
#  Test 4: Agent Routing
# ═══════════════════════════════════════════════════════════════════════
def test_agent_routing():
    """Test that queries are routed to the correct agents."""
    print("\n" + "=" * 70)
    print("  TEST 4: Agent Routing")
    print("=" * 70)
    passed = 0
    total = 0

    # (query, expected_agent_in_list)
    routing_tests = [
        ("what is my core vision about changing education system", "reflection"),
        ("Why did I build Cortex Lab?", "causal"),
        ("When did I start learning deep learning?", "timeline"),
        ("How has my thinking evolved over time?", "reflection"),
        ("Compare my skills then and now", "arbitration"),
    ]

    for query, expected_agent in routing_tests:
        total += 1
        text, meta = _rag_query(query)
        agents = meta.get("agents_used", []) if meta else []
        # Check if expected agent is in the agents list
        agent_ok = expected_agent in agents
        passed += _check(
            f"'{query[:50]}...' → agents={agents} (expect '{expected_agent}')",
            agent_ok,
            f"Expected '{expected_agent}' in agents list"
        )
        time.sleep(1.5)

    print(f"\n  Agent Routing: {passed}/{total} passed")
    return passed, total


# ═══════════════════════════════════════════════════════════════════════
#  Test 5: Response Robustness
# ═══════════════════════════════════════════════════════════════════════
def test_response_robustness():
    """Test that responses are complete, not truncated, and well-formed."""
    print("\n" + "=" * 70)
    print("  TEST 5: Response Robustness")
    print("=" * 70)
    passed = 0
    total = 0

    robustness_queries = [
        ("what is my core vision about changing education system", 300, ["education"]),
        ("Tell me about my projects related to deep learning", 100, ["project"]),
        ("What are my skills and technologies?", 50, ["python", "skill"]),
        ("What is my name and email?", 20, ["suraj", "@"]),
        ("Hey, how are you?", 10, []),
    ]

    for query, min_len, keywords in robustness_queries:
        total += 1
        text, meta = _rag_query(query)
        text = text or ""
        text_len = len(text)

        # Check for truncation indicators (ends mid-word)
        truncated = text.rstrip().endswith((" b", " t", " a", " th", " wh", " an"))
        # Check min length
        length_ok = text_len >= min_len
        # Check keywords
        kw_ok = not keywords or any(kw.lower() in text.lower() for kw in keywords)
        # Check not truncated
        all_ok = length_ok and not truncated and kw_ok

        trunc_flag = " [TRUNCATED!]" if truncated else ""
        passed += _check(
            f"'{query[:45]}...' → {text_len} chars, truncated={truncated}, kw_match={kw_ok}{trunc_flag}",
            all_ok,
            f"Expected ≥{min_len} chars, no truncation, keywords {keywords}. Start: '{text[:80]}...'"
        )
        time.sleep(2)

    print(f"\n  Response Robustness: {passed}/{total} passed")
    return passed, total


# ═══════════════════════════════════════════════════════════════════════
#  Test 6: Personal Info Regression
# ═══════════════════════════════════════════════════════════════════════
def test_personal_info_regression():
    """Regression tests for personal info extraction (previously fixed)."""
    print("\n" + "=" * 70)
    print("  TEST 6: Personal Info Regression")
    print("=" * 70)
    passed = 0
    total = 0

    personal_tests = [
        ("What is my name?", ["suraj"]),
        ("What is my email?", ["@"]),
        ("What is my phone number?", ["620", "972"]),
        ("What is my education?", ["B.Tech", "B.tech", "Computer", "Data"]),
        ("What are my skills?", ["python", "Python", "AI"]),
        ("Tell me about my projects", ["project", "built", "develop"]),
    ]

    for query, keywords in personal_tests:
        total += 1
        text, meta = _rag_query(query)
        text = text or ""
        found = any(kw.lower() in text.lower() for kw in keywords)
        passed += _check(
            f"'{query}' → found keywords: {found} ({len(text)} chars)",
            found,
            f"Expected one of {keywords}. Got: {text[:150]}"
        )
        time.sleep(1.5)

    print(f"\n  Personal Info Regression: {passed}/{total} passed")
    return passed, total


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CORTEX LAB — AGENTIC RAG COMPONENT TEST SUITE")
    print("=" * 70)

    # Check server health first
    try:
        r = requests.get(f"{BASE}/api/health", timeout=5)
        if r.status_code != 200:
            print("  ❌ Server not healthy!")
            sys.exit(1)
        print("  ✅ Server healthy")
    except Exception:
        print("  ❌ Server not reachable!")
        sys.exit(1)

    total_passed = 0
    total_tests = 0
    results = {}

    # Run all test suites
    suites = [
        ("Query Classification", test_query_classification),
        ("Evidence Quality", test_evidence_quality),
        ("Factual Extraction Guard", test_factual_extraction_guard),
        ("Agent Routing", test_agent_routing),
        ("Response Robustness", test_response_robustness),
        ("Personal Info Regression", test_personal_info_regression),
    ]

    for name, test_fn in suites:
        p, t = test_fn()
        total_passed += p
        total_tests += t
        results[name] = (p, t)
        time.sleep(2)

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for name, (p, t) in results.items():
        status = "✅" if p == t else "❌"
        print(f"  {status} {name}: {p}/{t}")
    print(f"\n  TOTAL: {total_passed}/{total_tests} passed")
    print("=" * 70)

    sys.exit(0 if total_passed == total_tests else 1)
