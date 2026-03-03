#!/usr/bin/env python3
"""Test all 7 queries against the RAG chat endpoint."""
import requests, json, time

BASE = "http://localhost:8000/api/rag/chat"

queries = [
    ("Simple Name", "What is my name?"),
    ("Education", "Where am I studying?"),
    ("Greeting", "Hey, how are you?"),
    ("Skills", "What are my skills?"),
    ("Projects", "What projects have I built?"),
    ("Education Background", "What is my education background?"),
    ("Everything About", "Tell me everything about Suraj"),
]

for label, query in queries:
    print(f"\n{'━'*50}")
    print(f"TEST: {label}")
    print(f"QUERY: {query}")
    
    try:
        r = requests.post(BASE, json={
            "messages": [{"role": "user", "content": query}],
            "session_id": f"test-{int(time.time())}",
            "stream": True,
        }, stream=True, timeout=120)
        
        answer = ""
        replaced = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
                if d.get("replace"):
                    replaced = d["replace"]
                if d.get("delta") and not d.get("done"):
                    answer += d["delta"]
            except:
                pass
        
        final = replaced if replaced else answer
        print(f"ANSWER: {final[:500]}")
        
        # Quality checks
        issues = []
        lower = final.lower()
        if "based on your stored memories" in lower:
            issues.append("❌ Robotic prefix 'Based on your stored memories'")
        if "[1]" in final or "[2]" in final:
            issues.append("❌ Has inline citations [1]/[2]")
        if "belief evolution" in lower or "emotion timeline" in lower:
            issues.append("❌ Hallucination pattern detected")
        if "confidence:" in lower or "evidence:" in lower:
            issues.append("❌ Self-RAG format leak")
        if lower.startswith("i ") or ". i " in lower or " i am " in lower or " i have " in lower:
            if "i don't" not in lower and "i'm doing" not in lower:
                issues.append("⚠️ First-person pronoun detected")
        if not final.strip():
            issues.append("❌ Empty response")
            
        if issues:
            for i in issues:
                print(f"  {i}")
        else:
            print("  ✅ Quality OK")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    time.sleep(2)

print(f"\n{'━'*50}")
print("All tests complete!")
