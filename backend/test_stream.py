"""Comprehensive RAG streaming test — tests multiple query types."""
import requests
import json
import sys
import time

BASE = "http://localhost:8000"

queries = [
    ("what is my core vision about changing education system", ["education", "vision"]),
    ("What is my name and email?", ["Suraj", "gmail"]),
    ("What is my phone number?", ["620", "972"]),
    ("What is my education?", ["B.Tech", "B.tech", "Computer Science", "Data Science"]),
    ("What are my skills?", ["Python", "AI", "ML"]),
    ("Tell me about my projects", ["project", "built", "develop"]),
    ("Hey, how are you?", ["Hey", "help", "doing"]),
    ("Why did I decide to build Cortex Lab?", ["Cortex", "memory", "AI", "build"]),
    ("What happened in my recent conversations?", ["conversation", "talk", "discuss"]),
    ("How has my thinking about AI evolved?", ["AI", "evolve", "change", "think"]),
]

def test_query(query, expected_keywords, verbose=False):
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/api/rag/chat", json={
                "messages": [{"role": "user", "content": query}],
                "stream": True,
                "llm_provider": "gemini"
            }, stream=True, timeout=120)
            break
        except requests.ConnectionError:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  [FAIL] {query}")
                print(f"         Connection refused after 3 attempts")
                return False

    all_text = ""
    meta = None
    for line in r.iter_lines():
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "rag_meta" in data:
                meta = data["rag_meta"]
            elif not data.get("done"):
                all_text += data.get("delta", "")

    found = [kw for kw in expected_keywords if kw.lower() in all_text.lower()]
    passed = len(found) > 0
    status = "PASS" if passed else "FAIL"
    conf = meta.get("confidence", 0) if meta else 0
    agents = meta.get("agents_used", []) if meta else []
    ev_count = len(meta.get("evidence", [])) if meta else 0

    # Check for truncation indicators
    truncated = all_text.rstrip().endswith(("—", "-", ",", " b", " t", " a", " th"))
    trunc_flag = " [TRUNCATED!]" if truncated else ""

    print(f"  [{status}] {query}")
    print(f"         conf={conf:.2f} agents={agents} evidence={ev_count} answer_len={len(all_text)}{trunc_flag}")
    if verbose or not passed:
        print(f"         answer: {all_text[:300]}")
        if len(all_text) > 300:
            print(f"         ...tail: ...{all_text[-100:]}")
    if not passed:
        print(f"         Expected one of: {expected_keywords}")
    return passed

print("=" * 70)
print("  COMPREHENSIVE RAG STREAMING TEST")
print("=" * 70)

passed = 0
total = len(queries)
for i, (q, kw) in enumerate(queries):
    # First query (education vision) gets verbose output
    if test_query(q, kw, verbose=(i == 0)):
        passed += 1
    print()
    if i < total - 1:
        time.sleep(2)  # Brief pause between queries

print("=" * 70)
print(f"  RESULT: {passed}/{total} passed")
print("=" * 70)
sys.exit(0 if passed == total else 1)
