#!/usr/bin/env python3
"""
Cortex Lab — Generate Cortex-Specific Orchestrator Training Data
Produces ~10,000 examples covering L0/L1/L2 agents, wiki ops, Deep Apps, SIA wake.

Usage:
    python scripts/generate_cortex_orchestrator_data.py
    python scripts/generate_cortex_orchestrator_data.py --count 5000
"""
import json, random, hashlib, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("CortexDataGen")

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "training_data" / "gemma4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEMMA4_BOS = "<bos>"

def fmt(system, user, assistant):
    return (f"{GEMMA4_BOS}<start_of_turn>system\n{system.strip()}<end_of_turn>\n"
            f"<start_of_turn>user\n{user.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n{assistant.strip()}<end_of_turn>")

# ── System prompts ────────────────────────────────────────────────────────────
L0_SYS = """You are the L0 Master-Orchestrator of Cortex Lab.
Role: classify input, filter noise, score retention, emit routing events.
NEVER generate user-facing text. ONLY output structured JSON events.
Output JSON with: event_type, trace_id, routing_decision, retention_score (0-1), agent_target, rationale."""

L1_SYS = """You are the L1 Runtime Orchestrator of Cortex Lab.
Role: receive classified query, dispatch L2 agents, run CRAG/Self-RAG/FLARE, synthesize evidence.
Always use <think>...</think> before output. Cite every claim with [Memory: timestamp]."""

AGENT_SYS = {
    "TimelineAgent":    "You are the Timeline Agent. Build chronological narratives from memories. Detect patterns and transitions. Use <think>...</think>. Cite [Memory: timestamp].",
    "CausalAgent":      "You are the Causal Agent. Trace causal chains backward/forward from events. Distinguish direct causes, contributing factors, correlations. Use <think>...</think>.",
    "ReflectionAgent":  "You are the Reflection Agent. Detect belief evolution, contradictions, growth patterns. Classify changes: REFINEMENT/CONTRADICTION/EXPANSION/ABANDONMENT/STABLE.",
    "PlanningAgent":    "You are the Planning Agent. Synthesize cross-domain evidence. Build multi-hop reasoning chains (5-7 hops). Identify cross-domain patterns.",
    "ArbitrationAgent": "You are the Arbitration Agent. Resolve conflicts between agent outputs. Weigh evidence quality, recency, and confidence. Output a final arbitrated answer.",
    "AcademicAgent":    "You are the Academic Agent. Ground responses in learning, research, and knowledge-building memories. Identify knowledge gaps and learning trajectories.",
    "WellbeingAgent":   "You are the Wellbeing Agent. Analyze health, mood, stress, and energy patterns. Apply privacy gate — never expose sensitive data without explicit recall.",
    "DecisionAgent":    "You are the Decision Agent. Analyze past decisions, outcomes, and patterns. Build decision trees from historical evidence. Identify recurring decision styles.",
    "GoalAgent":        "You are the Goal Agent. Track goal setting, progress, completions, and failures. Identify patterns in goal-pursuit behavior. Surface open loops.",
    "RelationshipAgent":"You are the Relationship Agent. Track relationship dynamics, recurring topics, emotional patterns. Apply privacy gate for sensitive interpersonal data.",
}

# ── Sample memory bank ────────────────────────────────────────────────────────
def make_memory(days_ago: int, content: str) -> dict:
    ts = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {"timestamp": ts, "content": content}

MEMORY_BANK = [
    make_memory(3,   "Completed the auth module in Rust. Team gave positive feedback."),
    make_memory(10,  "Had a difficult 1:1 with my manager about promotion timeline. Q2 is the target."),
    make_memory(15,  "Felt burnt out. Decided to take a full weekend off digital screens."),
    make_memory(22,  "Started learning TypeScript. Already see patterns from Python."),
    make_memory(30,  "Argued with Sarah about work-life balance. She feels I'm always distracted."),
    make_memory(45,  "Read 'Atomic Habits'. Key insight: systems beat goals. Identity-based change."),
    make_memory(60,  "Team restructure announced. My role is now lead for the infra squad."),
    make_memory(75,  "Sleep tracking started. Average 6.2h. Target is 7.5h minimum."),
    make_memory(90,  "Presentation to board went well. Got direct praise from CTO."),
    make_memory(120, "Old startup idea resurfaced. AI tutoring for K-12. Talked to John about it."),
    make_memory(150, "Promotion denied. Feedback: need more cross-functional impact."),
    make_memory(180, "Joined gym. Committed to 3x/week. First week done."),
]

def fmt_memories(mems: List[dict], n: int = 6) -> str:
    return "\n".join(f"{i+1}. [{m['timestamp']}] {m['content']}" for i, m in enumerate(mems[:n]))

# ── L0 examples ──────────────────────────────────────────────────────────────
L0_SCENARIOS = [
    ("User is speaking: 'I need to check the weather in London tomorrow.'",
     {"event_type":"ROUTING_DECISION","trace_id":"tr_001","routing_decision":"DISCARD","retention_score":0.05,"agent_target":"none","rationale":"Weather query — no personal memory relevance. Below retention threshold 0.30. Noise filter applied."}),
    ("User: 'Remember when I talked to John about the startup idea?'",
     {"event_type":"ROUTING_DECISION","trace_id":"tr_002","routing_decision":"ROUTE_TO_L1","retention_score":0.82,"agent_target":"TimelineAgent","rationale":"Personal episodic recall query. High retention score — named entity 'John', topic 'startup idea'. Route to Timeline retrieval."}),
    ("Background noise: ambient television audio detected, no speech.",
     {"event_type":"NOISE_FILTER","trace_id":"tr_003","routing_decision":"DISCARD","retention_score":0.0,"agent_target":"none","rationale":"Non-speech audio classified as ambient noise. VAD confidence below threshold. Discard."}),
    ("User: 'Why am I always tired after team meetings?'",
     {"event_type":"ROUTING_DECISION","trace_id":"tr_004","routing_decision":"ROUTE_TO_L1","retention_score":0.75,"agent_target":"CausalAgent","rationale":"Causal introspective query. Pattern detection across sessions needed. Route to CausalAgent for energy/meeting correlation analysis."}),
    ("SIA. [Wake phrase detected, user says: 'What did I decide about the gym?']",
     {"event_type":"WAKE_RETRIEVE","trace_id":"tr_005","routing_decision":"ROUTE_TO_L1","retention_score":0.88,"agent_target":"TimelineAgent","rationale":"SIA wake phrase confirmed. Factual recall query about specific decision. High priority retrieve mode activated."}),
    ("User: 'I had pizza for lunch.'",
     {"event_type":"RETENTION_DECISION","trace_id":"tr_006","routing_decision":"SESSION_ONLY","retention_score":0.32,"agent_target":"EventPlane","rationale":"Low-novelty episodic event. Retention score 0.32 — above discard threshold (0.30) but below structured memory (0.50). Store as session-only memory."}),
    ("User: 'I just got promoted to Staff Engineer!'",
     {"event_type":"RETENTION_DECISION","trace_id":"tr_007","routing_decision":"PRIORITY_MEMORY","retention_score":0.97,"agent_target":"EventPlane+ClaimPlane","rationale":"High-significance life event. Retention score 0.97. Store to both Event Plane (P1) and Claim Plane (P2). Trigger wiki update for Career page."}),
]

def gen_l0_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        scenario, expected_json = random.choice(L0_SCENARIOS)
        # Add slight variation
        think = f"<think>\nClassifying input: '{scenario[:80]}...'\nApplying noise filter → relevance check → retention scoring...\n</think>\n\n"
        output = think + json.dumps(expected_json, indent=2)
        examples.append(fmt(L0_SYS, scenario, output))
    return examples

# ── L1 Orchestrator examples ──────────────────────────────────────────────────
L1_QUERIES = [
    ("Why have I been feeling more stressed lately?",
     "CausalAgent", "STRESS_PATTERN",
     lambda mems: f"<think>\nAnalyzing stress indicators across {len(mems)} memories...\nKey events: meeting with manager [promo denied], burnout feeling, argument with Sarah.\nDispatch CausalAgent for pattern analysis.\n</think>\n\n**Dispatch: CausalAgent**\n```json\n{{\"agent\":\"CausalAgent\",\"query\":\"stress causation pattern last 90 days\",\"retrieval_channels\":[\"temporal\",\"dense\"],\"trace_id\":\"tr_l1_001\"}}\n```\n\n**Synthesis after CausalAgent response:**\nBased on the memory analysis:\n1. Promotion denial [Memory: {mems[4]['timestamp']}] — significant stressor\n2. Increased workload as infra lead [Memory: {mems[2]['timestamp']}]\n3. Relationship friction [Memory: {mems[1]['timestamp']}]\n\nPattern: Role expansion without recognition → compounding stress.\n**Confidence:** High — 3 corroborating memories within 60 days."),
    ("What have I learned about myself this year?",
     "ReflectionAgent", "SELF_REFLECTION",
     lambda mems: f"<think>\nReflection query spanning full year. Dispatch ReflectionAgent.\nWill need: belief evolution, growth patterns, recurring challenges.\n</think>\n\n**Dispatch: ReflectionAgent + TimelineAgent (parallel)**\n\n**Synthesis:**\nThis year reveals three key themes:\n1. **Systems over goals** [Memory: {mems[1]['timestamp']}] — reading Atomic Habits shifted your approach from outcome-focus to identity-focus.\n2. **Recovery as strategy** [Memory: {mems[2]['timestamp']}] — burnout triggered a digital detox, suggesting growing self-awareness.\n3. **Recognition gap** [Memory: {mems[4]['timestamp']}] — promotion denial is an open loop requiring attention.\n\n**Confidence:** High — 6 memories spanning the year with clear pattern."),
]

def gen_l1_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        q, agent, intent, response_fn = random.choice(L1_QUERIES)
        mems = random.sample(MEMORY_BANK, 6)
        mem_ctx = fmt_memories(mems)
        user = f"Query: {q}\n\nRetrieved Memories:\n{mem_ctx}"
        output = response_fn(mems)
        examples.append(fmt(L1_SYS, user, output))
    return examples

# ── L2 Agent examples ─────────────────────────────────────────────────────────
def gen_agent_examples(n: int) -> List[str]:
    examples = []
    agents = list(AGENT_SYS.keys())
    queries = {
        "TimelineAgent": "Build a timeline of my career progress over the past 6 months.",
        "CausalAgent": "Why did I decide to join the gym when I did?",
        "ReflectionAgent": "How has my view on work-life balance changed?",
        "PlanningAgent": "What cross-domain patterns connect my learning, career, and health this year?",
        "ArbitrationAgent": "Two memories conflict: one says I committed to gym 3x/week, another says I skipped consistently. What's the truth?",
        "AcademicAgent": "What have I been learning recently and what knowledge gaps remain?",
        "WellbeingAgent": "What patterns do I see in my energy and sleep?",
        "DecisionAgent": "What types of decisions do I tend to delay?",
        "GoalAgent": "What goals have I set but not completed?",
        "RelationshipAgent": "How has my relationship with Sarah been evolving?",
    }
    for _ in range(n):
        agent = random.choice(agents)
        q = queries[agent]
        mems = random.sample(MEMORY_BANK, min(5, len(MEMORY_BANK)))
        mem_ctx = fmt_memories(mems)
        user = f"Query: {q}\n\nRetrieved Memories:\n{mem_ctx}"

        m1, m2 = mems[0], mems[1]
        think = f"<think>\nAnalyzing {len(mems)} memories for: {q[:60]}...\nKey memory: [{m1['timestamp']}] {m1['content'][:50]}...\n</think>\n\n"
        answer = (f"{think}Based on your memories:\n\n"
                  f"**Key finding** [Memory: {m1['timestamp']}]: {m1['content']}\n\n"
                  f"**Supporting context** [Memory: {m2['timestamp']}]: {m2['content']}\n\n"
                  f"**Confidence:** High — {len(mems)} relevant memories analyzed.")
        examples.append(fmt(AGENT_SYS[agent], user, answer))
    return examples

# ── Wiki Operations ───────────────────────────────────────────────────────────
WIKI_SYS = """You are the Wiki Agent in Cortex Lab.
Operations: PATCH (merge new claim into page), CREATE (scaffold new page), LINT (detect stale/contradictory content), COMPACT (compress event log to claims).
Output structured wiki operations as JSON."""

WIKI_OPS = [
    ("PATCH the Career wiki page with this new memory: 'Got promoted to Staff Engineer on 2026-04-01.'",
     {"operation":"PATCH","page":"Career","section":"Current_Role","action":"UPDATE","old_claim":"Software Engineer at TechCorp since 2024","new_claim":"Staff Engineer at TechCorp since April 2026","evidence":"[Memory: 2026-04-01]","confidence":0.99}),
    ("CREATE a new wiki page for the AI tutoring startup idea.",
     {"operation":"CREATE","page":"StartupIdea_AITutoring","sections":{"overview":"AI tutoring startup targeting K-12 market","co_founders":["John"],"status":"idea phase","key_decisions":["Target K-12 first","Build MVP in 6 weeks"],"open_loops":["funding","technical stack","go-to-market"]},"created_from":"[Memory: 2026-01-10]"}),
    ("LINT the Health wiki page for stale content.",
     {"operation":"LINT","page":"Health","issues":[{"section":"Exercise","issue":"Gym commitment recorded as 3x/week but no recent entries confirm adherence","severity":"HIGH","recommendation":"Update with current status"},{"section":"Sleep","issue":"Sleep target of 7.5h may be outdated — no recent tracking memories","severity":"MEDIUM","recommendation":"Add recent sleep data"}]}),
    ("COMPACT the last 30 days of career event logs into wiki claims.",
     {"operation":"COMPACT","source":"EventPlane_Career_Last30Days","output_claims":[{"claim":"Led infrastructure squad through Q1 milestone delivery","confidence":0.9,"evidence":["Memory: 2026-03-15"]},{"claim":"Received promotion to Staff Engineer","confidence":0.99,"evidence":["Memory: 2026-04-01"]}],"events_compacted":12,"claims_produced":2}),
]

def gen_wiki_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        user, expected = random.choice(WIKI_OPS)
        think = f"<think>\nWiki operation: {user[:60]}...\nValidating against existing page content...\nGenerating structured operation...\n</think>\n\n"
        output = think + json.dumps(expected, indent=2)
        examples.append(fmt(WIKI_SYS, user, output))
    return examples

# ── Deep Application workflows ────────────────────────────────────────────────
DEEP_APP_SYS = "You are Cortex, running a Deep Application workflow. Use <think>...</think>. Cite memories. Produce rich, structured output."

DEEP_APP_SCENARIOS = [
    ("Session Memory Forge: Crystallize today's session into structured memory objects.",
     lambda mems: f"<think>\nCrystallizing session: extracting claims, detecting open loops, updating beliefs...\n</think>\n\n**Session Memory Forge Report**\n\n**Claims Extracted:**\n- [{mems[0]['timestamp']}] {mems[0]['content']}\n\n**Open Loops Detected:**\n- Promotion timeline is unresolved — Q2 target mentioned but not confirmed\n\n**Belief Updates:**\n- Work-life balance view showing tension [Memory: {mems[1]['timestamp']}]\n\n**Retention Scores:**\n- Career event: 0.89 → Priority Memory\n- Daily log: 0.45 → Session Only"),
    ("Deep Self Mirror: What are my top 3 recurring patterns this quarter?",
     lambda mems: f"<think>\nAnalyzing patterns across {len(mems)} memories for the quarter...\nLooking for: behavioral patterns, emotional cycles, decision styles.\n</think>\n\n**Deep Self Mirror — Q1 Patterns**\n\n1. **Recovery after overcommitment** [Memory: {mems[0]['timestamp']}] — You take on too much, burn out, then self-correct. This cycle repeats every 6-8 weeks.\n\n2. **Learning during transition** [Memory: {mems[1]['timestamp']}] — Major skills acquired during role changes, not during stability.\n\n3. **Recognition as motivation** [Memory: {mems[2]['timestamp']}] — External validation (promotion, praise) strongly correlates with energy spikes.\n\n**Key Insight:** You're a burst-learner, not a consistent-grinder. Design your systems accordingly."),
    ("Gap Intelligence: What knowledge and experience gaps should I address next?",
     lambda mems: f"<think>\nScanning memories for gap signals: unresolved questions, incomplete projects, stated ignorance...\n</think>\n\n**Gap Intelligence Report**\n\n**Skill Gaps:**\n- Cross-functional leadership experience [Memory: {mems[0]['timestamp']}] — Promotion feedback cited this explicitly\n- TypeScript depth [Memory: {mems[1]['timestamp']}] — Started but not completed\n\n**Knowledge Gaps:**\n- Startup go-to-market strategy — AI tutoring idea lacks business model depth\n\n**Experience Gaps:**\n- Managing reports — current role is IC, next step requires people management\n\n**Recommended Priority:** Cross-functional impact (directly tied to promotion goal)"),
]

def gen_deep_app_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        user_q, resp_fn = random.choice(DEEP_APP_SCENARIOS)
        mems = random.sample(MEMORY_BANK, 4)
        mem_ctx = fmt_memories(mems)
        user = f"{user_q}\n\nAvailable Memories:\n{mem_ctx}"
        output = resp_fn(mems)
        examples.append(fmt(DEEP_APP_SYS, user, output))
    return examples

# ── SIA Wake phrase examples ──────────────────────────────────────────────────
SIA_SYS = """You are Cortex in SIA retrieve mode (activated by wake phrase).
Respond immediately with direct, relevant memory retrieval. Skip preamble.
Cite [Memory: timestamp] for every fact. Be concise — the user is in voice mode."""

SIA_QUERIES = [
    ("What did I decide about the gym last month?",
     lambda mems: f"[Memory: {mems[5]['timestamp']}] You committed to going to the gym 3x per week. First week completed successfully."),
    ("When did I last talk to John?",
     lambda mems: f"[Memory: {mems[3]['timestamp']}] You talked to John about the AI tutoring startup idea targeting the K-12 market."),
    ("What was my manager's feedback about the promotion?",
     lambda mems: f"[Memory: {mems[4]['timestamp']}] Your manager said you need more cross-functional impact. Q2 is the current target timeline."),
]

def gen_sia_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        q, resp_fn = random.choice(SIA_QUERIES)
        output = resp_fn(MEMORY_BANK)
        examples.append(fmt(SIA_SYS, f"SIA. {q}", output))
    return examples

# ── Memory plane lifecycle ────────────────────────────────────────────────────
PLANE_SYS = "You are the Memory Plane Manager in Cortex Lab. Route and transform memories through the 5-plane system: P0 Working, P1 Event, P2 Claim, P3 Wiki, P4 Graph."

def gen_plane_examples(n: int) -> List[str]:
    examples = []
    raw_events = [
        "User said: 'I got promoted to Staff Engineer today!'",
        "User said: 'Just finished the Rust auth module — took 3 hours.'",
        "User said: 'Sarah and I had a fight about my work hours.'",
    ]
    for _ in range(n):
        event = random.choice(raw_events)
        think = "<think>\nProcessing raw event through memory plane lifecycle...\nP1 Event: store raw → P2 Claim: extract atomic facts → P3 Wiki: check if page update needed → P4 Graph: update entity relations.\n</think>\n\n"
        output = think + json.dumps({
            "P1_Event": {"stored": True, "event_id": f"ev_{random.randint(1000,9999)}", "raw": event},
            "P2_Claims": [{"claim": event.replace("User said: '","").rstrip("'"), "confidence": 0.95, "type": "episodic"}],
            "P3_Wiki": {"operation": "PATCH", "page": "Career", "update_needed": True},
            "P4_Graph": {"edges_added": [{"from": "self", "relation": "achieved", "to": "Staff_Engineer_Role"}]},
        }, indent=2)
        examples.append(fmt(PLANE_SYS, f"Process this raw event through all memory planes:\n{event}", output))
    return examples

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--out",   type=str, default=str(OUT_DIR / "cortex_orchestrator_data.jsonl"))
    args = parser.parse_args()

    n = args.count
    out_path = Path(args.out)

    log.info(f"Generating {n:,} Cortex-specific training examples...")

    # Distribution across categories
    dist = {
        "L0 Master-Orchestrator":    int(n * 0.15),
        "L1 Runtime Orchestrator":   int(n * 0.15),
        "L2 Specialized Agents":     int(n * 0.30),
        "Wiki Operations":           int(n * 0.10),
        "Deep Applications":         int(n * 0.20),
        "SIA Wake Mode":             int(n * 0.05),
        "Memory Plane Lifecycle":    int(n * 0.05),
    }

    all_examples = []
    all_examples.extend(gen_l0_examples(dist["L0 Master-Orchestrator"]))
    all_examples.extend(gen_l1_examples(dist["L1 Runtime Orchestrator"]))
    all_examples.extend(gen_agent_examples(dist["L2 Specialized Agents"]))
    all_examples.extend(gen_wiki_examples(dist["Wiki Operations"]))
    all_examples.extend(gen_deep_app_examples(dist["Deep Applications"]))
    all_examples.extend(gen_sia_examples(dist["SIA Wake Mode"]))
    all_examples.extend(gen_plane_examples(dist["Memory Plane Lifecycle"]))

    random.shuffle(all_examples)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps({"text": ex}, ensure_ascii=False) + "\n")

    log.info(f"\nGenerated: {len(all_examples):,} examples")
    log.info(f"Saved to:  {out_path}")
    log.info(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")
    for cat, cnt in dist.items():
        log.info(f"  {cat:<35} {cnt:>6,}")
    log.info("\nNext step: python scripts/prepare_gemma4_datasets.py --stage phaseJ")

if __name__ == "__main__":
    main()
