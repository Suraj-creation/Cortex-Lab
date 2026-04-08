#!/usr/bin/env python3
"""
Deep Agentic RAG Pipeline Live Tests
=====================================
Tests the FULL end-to-end RAG pipeline against the running server.
Each test validates a real query against known data in the database.

Run: python tests/test_rag_pipeline_live.py
Requires: server running on http://localhost:8000
"""

import requests
import json
import time
import sys

import pytest

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0
TESTS = []


def _pytest_server_ready() -> bool:
    try:
        response = requests.get(f"{BASE}/api/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pytest_server_ready(),
    reason="Live RAG pipeline tests require backend server at localhost:8000",
)


def check_server():
    """Verify server is running before tests. Wait up to 120s."""
    print("â³ Waiting for server...")
    for i in range(60):
        try:
            r = requests.get(f"{BASE}/api/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(f"âœ… Server ready after {i*2}s")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_case(name, query, checks):
    """Run a single RAG pipeline test."""
    global PASS, FAIL
    print(f'\n{"="*70}')
    print(f"TEST: {name}")
    print(f"QUERY: {query}")
    print(f'{"="*70}')

    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/api/rag/chat",
            json={
                "messages": [{"role": "user", "content": query}],
                "session_id": f"deep_test_{int(time.time())}",
                "stream": False,
            },
            timeout=150,
        )
        elapsed = time.time() - t0

        if r.status_code != 200:
            print(f"  âŒ HTTP {r.status_code}: {r.text[:200]}")
            FAIL += 1
            TESTS.append((name, "FAIL", f"HTTP {r.status_code}"))
            return

        data = r.json()
        answer = data.get("content", "")
        evidence = data.get("evidence", [])
        agents = data.get("agents_used", [])
        qa = data.get("query_analysis", {})
        conf = data.get("confidence", 0)
        proc_ms = data.get("processing_time_ms", 0)
        cache_hit = data.get("cache_hit", False)

        print(f'  â±  Time: {elapsed:.1f}s (pipeline: {proc_ms:.0f}ms)')
        print(f'  ðŸ” Intent: {qa.get("intent","?")} | Routing: {qa.get("routing","?")}')
        print(f"  ðŸ¤– Agents: {agents}")
        print(f"  ðŸ“„ Evidence: {len(evidence)} pieces | Confidence: {conf}")
        print(f"  ðŸ’¾ Cache hit: {cache_hit}")
        print(f"  ðŸ“ ANSWER ({len(answer)} chars):")
        print(f"     {answer[:600]}")

        # Show evidence
        print(f"\n  ðŸ“‹ Evidence preview:")
        for i, e in enumerate(evidence[:3]):
            c = e.get("content", str(e))[:100] if isinstance(e, dict) else str(e)[:100]
            print(f"     [{i+1}] {c}")

        # Run checks
        failures = []
        for ck_name, ck_fn in checks.items():
            try:
                ok = ck_fn(data)
                print(f'  {"âœ…" if ok else "âŒ"} {ck_name}')
                if not ok:
                    failures.append(ck_name)
            except Exception as exc:
                print(f"  âŒ {ck_name}: {exc}")
                failures.append(ck_name)

        if failures:
            FAIL += 1
            TESTS.append((name, "FAIL", ", ".join(failures)))
        else:
            PASS += 1
            TESTS.append((name, "PASS", f"{elapsed:.1f}s"))

    except Exception as e:
        elapsed = time.time() - t0
        FAIL += 1
        TESTS.append((name, "FAIL", str(e)[:80]))
        print(f"  âŒ Exception: {e}")


def main():
    global PASS, FAIL

    # Pre-flight check
    if not check_server():
        print("âŒ Server not running at http://localhost:8000")
        print("   Start it first: python server.py")
        sys.exit(1)

    print("\n" + "ðŸ§ª " * 20)
    print("  DEEP AGENTIC RAG PIPELINE TESTS")
    print("  Testing real queries against real stored memories")
    print("ðŸ§ª " * 20)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 1: Personal Identity (Factual)
    # DB memory: "My name is Suraj Kumar. I am from Patna, Bihar, India."
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "1. Factual: Personal Identity",
        "What is my name and where am I from?",
        {
            "answer_not_empty": lambda d: len(d["content"]) > 15,
            "mentions_suraj": lambda d: "suraj" in d["content"].lower(),
            "mentions_location": lambda d: any(
                w in d["content"].lower() for w in ["patna", "bihar", "india"]
            ),
            "has_evidence": lambda d: len(d.get("evidence", [])) > 0,
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 2: Education (Factual)
    # DB: "B.Tech CSE at Vidyashilp University, Bangalore"
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "2. Factual: Education Background",
        "Where am I studying and what is my degree?",
        {
            "answer_not_empty": lambda d: len(d["content"]) > 15,
            "mentions_university": lambda d: any(
                w in d["content"].lower()
                for w in ["vidyashilp", "university", "bangalore"]
            ),
            "mentions_degree": lambda d: any(
                w in d["content"].lower()
                for w in ["b.tech", "btech", "computer science", "cse", "data science"]
            ),
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 3: Contact Info (Factual Compound)
    # DB: "surajcreationinfinity1@gmail.com", "+91 6204153972"
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "3. Factual: Contact Information",
        "What is my email address and phone number?",
        {
            "answer_not_empty": lambda d: len(d["content"]) > 15,
            "has_email": lambda d: "@" in d["content"] and "gmail" in d["content"].lower(),
            "has_phone": lambda d: "6204" in d["content"] or "153972" in d["content"],
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 4: Skills (Factual)
    # DB: "Python, Java, C... PyTorch, TensorFlow... NLP, CV, RAG"
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "4. Factual: Technical Skills",
        "What are my programming languages and AI skills?",
        {
            "answer_not_empty": lambda d: len(d["content"]) > 30,
            "mentions_python": lambda d: "python" in d["content"].lower(),
            "mentions_ai": lambda d: any(
                w in d["content"].lower()
                for w in ["pytorch", "tensorflow", "nlp", "deep learning", "rag", "machine learning"]
            ),
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 5: Projects (Factual - listing)
    # DB: 20+ projects including Cortex Lab, Jarurat Care, Echo Chamber, etc.
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "5. Factual: List Projects",
        "List my top projects that I have built",
        {
            "answer_not_empty": lambda d: len(d["content"]) > 50,
            "mentions_project": lambda d: any(
                w in d["content"].lower()
                for w in [
                    "cortex", "jarurat", "echo chamber", "healthcare",
                    "snake", "note", "sysmind", "portfolio", "classroom",
                    "finance", "resume", "captioning",
                ]
            ),
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 6: False Premise Rejection (Hallucination Guard)
    # Suraj has NO PhD â€” system must reject, not fabricate
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "6. Hallucination Guard: False Premise (PhD)",
        "Tell me about my PhD research at Stanford",
        {
            "rejects_premise": lambda d: any(
                w in d["content"].lower()
                for w in ["don't have", "no phd", "no information", "not", "haven't"]
            ),
            "no_fabrication": lambda d: "stanford" not in d["content"].lower()
            or "don't" in d["content"].lower()
            or "no " in d["content"].lower(),
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEST 7: Philosophy & Vision (Reflective)
    # DB: education manifesto, "redefining learning", startup visions
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    run_case(
        "7. Reflective: Education Vision",
        "What is my vision for transforming education?",
        {
            "answer_not_empty": lambda d: len(d["content"]) > 40,
            "mentions_education": lambda d: "education" in d["content"].lower()
            or "learn" in d["content"].lower(),
            "mentions_vision_concepts": lambda d: any(
                w in d["content"].lower()
                for w in ["reinvent", "redefin", "transform", "innovat", "purpose", "ai", "agentic"]
            ),
        },
    )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # SUMMARY
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print(f"\n\n{'#'*70}")
    print(f"  DEEP RAG PIPELINE TEST RESULTS: {PASS} PASSED, {FAIL} FAILED / {PASS+FAIL} total")
    print(f"{'#'*70}")
    for name, status, detail in TESTS:
        icon = "âœ…" if status == "PASS" else "âŒ"
        print(f"  {icon} {name}: {detail}")
    print()

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()

