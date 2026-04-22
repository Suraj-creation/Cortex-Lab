#!/usr/bin/env python3
"""
Cortex Lab — Gemma-4 Post-Training Evaluation
Evaluates fine-tuned model across all critical capabilities.

Usage:
    python scripts/evaluate_gemma4.py
    python scripts/evaluate_gemma4.py --stage stageC_faithfulness
    python scripts/evaluate_gemma4.py --quick   # 50 examples per test
"""
import os, sys, json, argparse, logging, time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Eval")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from config.gemma4_training_config import OUTPUT_DIR, FULL_STAGE_ORDER, BASE_MODEL, HF_TOKEN, ALL_STAGES

# ─── Test Cases ───────────────────────────────────────────────────────────────
EVAL_SUITES = {
    "faithfulness": {
        "description": "RAG grounded answer fidelity",
        "tests": [
            {"query": "What did I learn about productivity last week?",
             "memories": ["[2026-04-15] Learned about Pomodoro technique from a blog post. Key insight: 25 min blocks with 5 min breaks."],
             "expected_behavior": ["cite [Memory: 2026-04-15]", "mention Pomodoro", "NOT hallucinate"]},
            {"query": "What is my favorite restaurant?",
             "memories": ["[2026-04-10] Had a meeting with my colleague about the API"],
             "expected_behavior": ["refuse gracefully", "say insufficient memories", "NOT make up restaurant"]},
        ],
    },
    "routing": {
        "description": "L1 agent routing JSON accuracy",
        "tests": [
            {"query": "Why am I feeling stressed lately?",
             "expected_behavior": ["route to CausalAgent", "output valid JSON", "include trace_id"]},
            {"query": "What happened last Tuesday?",
             "expected_behavior": ["route to TimelineAgent", "output valid JSON"]},
        ],
    },
    "self_rag": {
        "description": "ISREL/ISSUP/ISUSE critique tokens",
        "tests": [
            {"query": "Evaluate this answer for factual grounding",
             "expected_behavior": ["output ISREL tag", "output ISSUP tag", "output ISUSE tag", "provide verdict"]},
        ],
    },
    "belief_evolution": {
        "description": "Contradiction detection and belief tracking",
        "tests": [
            {"query": "How has my view on remote work changed?",
             "memories": [
                 "[2024-01-15] I believe remote work is the future. Productivity is higher.",
                 "[2024-06-20] Starting to miss office collaboration. Maybe hybrid is better.",
                 "[2025-01-10] Fully convinced now: hybrid is the sweet spot."
             ],
             "expected_behavior": ["detect 3 stages", "classify as REFINEMENT", "show timeline"]},
        ],
    },
}


def find_latest_merged() -> str:
    """Find the most recent merged model."""
    for stage in reversed(FULL_STAGE_ORDER):
        merged = OUTPUT_DIR / stage / "merged"
        if merged.exists() and (merged / "config.json").exists():
            return str(merged)
    return BASE_MODEL


def run_inference(model, tokenizer, system: str, user: str, max_tokens: int = 512) -> str:
    """Run a single inference."""
    prompt = (f"<bos><start_of_turn>system\n{system}<end_of_turn>\n"
              f"<start_of_turn>user\n{user}<end_of_turn>\n"
              f"<start_of_turn>model\n")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def check_behavior(response: str, expected: List[str]) -> Dict:
    """Check if response matches expected behaviors."""
    results = {}
    for behavior in expected:
        if behavior.startswith("NOT "):
            # Negative check
            forbidden = behavior[4:].lower()
            results[behavior] = forbidden not in response.lower()
        else:
            results[behavior] = behavior.lower() in response.lower() or any(
                w in response.lower() for w in behavior.lower().split()
            )
    passed = sum(1 for v in results.values() if v)
    return {"checks": results, "passed": passed, "total": len(results), "score": passed / max(len(results), 1)}


def evaluate_suite(model, tokenizer, suite_name: str, suite: Dict, quick: bool = False) -> Dict:
    """Run evaluation for a single suite."""
    log.info(f"\n  Evaluating: {suite_name} — {suite['description']}")
    results = []

    system = "You are Cortex, a personal AI with persistent memory. Use <think>...</think> for reasoning. Cite [Memory: timestamp]."

    for test in suite["tests"]:
        query = test["query"]
        memories = test.get("memories", [])
        mem_ctx = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(memories))
        user_input = f"Query: {query}"
        if mem_ctx:
            user_input += f"\n\nRetrieved Memories:\n{mem_ctx}"

        response = run_inference(model, tokenizer, system, user_input)
        check = check_behavior(response, test["expected_behavior"])
        results.append({"query": query, "response": response[:300], **check})
        log.info(f"    {query[:50]}... → {check['passed']}/{check['total']} checks passed")

    avg_score = sum(r["score"] for r in results) / max(len(results), 1)
    return {"suite": suite_name, "results": results, "avg_score": avg_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model (default: latest merged)")
    parser.add_argument("--suite", type=str, choices=list(EVAL_SUITES.keys()), help="Run specific suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()

    model_path = args.model or find_latest_merged()
    log.info(f"Evaluating model: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=HF_TOKEN)

    suites = {args.suite: EVAL_SUITES[args.suite]} if args.suite else EVAL_SUITES
    all_results = {}

    log.info(f"\n{'='*60}")
    log.info(f"CORTEX LAB — Gemma-4 Evaluation Report")
    log.info(f"{'='*60}")

    for name, suite in suites.items():
        result = evaluate_suite(model, tokenizer, name, suite, args.quick)
        all_results[name] = result

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY")
    log.info(f"{'='*60}")
    for name, r in all_results.items():
        score = r["avg_score"] * 100
        status = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
        log.info(f"  {status} {name:<30} {score:.0f}%")

    # Save report
    report_path = OUTPUT_DIR / "evaluation_report.json"
    report = {"model": model_path, "evaluated_at": datetime.now().isoformat(), "results": all_results}
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
