#!/bin/bash
# Test script for natural response quality
BASE="http://localhost:8000/api/rag/chat"

test_query() {
    local label="$1"
    local query="$2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: $label"
    echo "QUERY: $query"
    echo ""

    response=$(curl -s -N -X POST "$BASE" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$query\"}],\"session_id\":\"test-$(date +%s)\",\"stream\":true}" 2>/dev/null)

    # Extract all delta values and concatenate
    answer=$(echo "$response" | grep '^data: ' | while IFS= read -r line; do
        payload="${line#data: }"
        echo "$payload" | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.read().strip())
    delta=d.get('delta','')
    if delta and not d.get('done'):
        print(delta,end='')
except: pass
" 2>/dev/null
    done)

    echo "ANSWER: $answer"
    echo ""
}

test_query "Simple Name" "What is my name?"
sleep 2
test_query "Education" "Where am I studying?"
sleep 2
test_query "Skills" "What are my skills?"
sleep 2
test_query "Projects" "What projects have I built?"
sleep 2
test_query "Greeting" "Hey, how are you?"
sleep 2
test_query "Education Background" "What is my education background?"
sleep 2
test_query "Everything About" "Tell me everything about Suraj"
