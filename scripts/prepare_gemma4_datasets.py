#!/usr/bin/env python3
"""
Cortex Lab — Dataset Preparation for Gemma-4-E2B-it
=====================================================
Converts all training datasets into Gemma-4's native chat template format.

Gemma-4 Chat Template:
  <bos><start_of_turn>system
  {system_prompt}<end_of_turn>
  <start_of_turn>user
  {user_message}<end_of_turn>
  <start_of_turn>model
  {model_response}<end_of_turn>

Key behaviors:
  - Maps <think>...</think> tags in existing data to Gemma-4's thinking format
  - Merges stage2_agentic + stage13_function_calling into phaseD_agentic_combined
  - Merges stage7_dialogue + stage8_longcontext into phaseI_dialogue_longctx
  - Merges stage14_rft + stage15_spin into phaseO_rft_spin
  - Converts Opus-4.6 reasoning format → Gemma-4 chat template
  - Converts Edge-Agent WebSearch format → Gemma-4 agentic format

Usage:
    python scripts/prepare_gemma4_datasets.py
    python scripts/prepare_gemma4_datasets.py --stage phaseA  # single phase
    python scripts/prepare_gemma4_datasets.py --validate      # validate only
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DataPrep")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TRAINING_DATA = ROOT / "training_data"
RAW_DIR       = TRAINING_DATA / "gemma4" / "raw"
OUT_DIR       = TRAINING_DATA / "gemma4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Gemma-4 system prompts per role ─────────────────────────────────────────
CORTEX_L0_SYSTEM = """You are the L0 Master-Orchestrator of the Cortex Lab personal intelligence system.
Your role: classify all incoming input (speech/text), apply noise filtering, make retention decisions, and emit routing events.
You NEVER generate user-facing answers. You ONLY output structured JSON events.
Apply the Four Laws: (1) Data without structure is noise. (2) Structure without selection creates pollution. (3) Selection without context destroys retrieval. (4) Retrieval without confidence is hallucination risk.
Output format: JSON routing event with fields: event_type, session_id, trace_id, routing_decision, retention_score, agent_target, rationale."""

CORTEX_L1_SYSTEM = """You are the L1 Runtime Orchestrator of the Cortex Lab personal intelligence system.
Your role: receive classified queries from L0, execute query analysis, dispatch to specialized L2 agents via TeamCreateTool, run CRAG/Self-RAG/FLARE quality loops, and synthesize evidence into final responses.
Always use <think>...</think> for reasoning before producing structured output.
Citation format: [Memory: YYYY-MM-DD] or [Memory: event_id]
Confidence levels: High (3+ memories, <30 days) | Medium (1-2 memories, 30-90 days) | Low (tangential, >90 days) | Insufficient (no relevant memories)"""

CORTEX_AGENT_SYSTEM = """You are a specialized L2 agent in the Cortex Lab personal intelligence system.
Always use <think>...</think> for reasoning.
Always cite evidence with [Memory: timestamp] for every claim.
Express calibrated confidence. Refuse gracefully when evidence is insufficient.
Produce structured JSON output for all routing/dispatch decisions."""

CORTEX_DEFAULT_SYSTEM = """You are Cortex, a personal AI intelligence system with persistent memory.
Answer ONLY from provided memories. Cite every claim with [Memory: timestamp].
Use <think>...</think> for reasoning. Express calibrated confidence: High / Medium / Low / Insufficient.
Never hallucinate facts not present in the retrieved memories."""


# ─── Gemma-4 chat template formatter ─────────────────────────────────────────
def to_gemma4_format(
    system: str,
    user: str,
    assistant: str,
    bos_token: str = "<bos>",
) -> str:
    """Format a single example into Gemma-4 chat template."""
    # Normalize thinking tags: <think> → embedded in response as-is
    # Gemma-4 uses <|channel>thought\n[thinking]<channel|> internally,
    # but during SFT we train on the raw <think> format in text.
    text = (
        f"{bos_token}"
        f"<start_of_turn>system\n{system.strip()}<end_of_turn>\n"
        f"<start_of_turn>user\n{user.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant.strip()}<end_of_turn>"
    )
    return text


def fmt_sft_example(ex: dict, system_override: Optional[str] = None) -> Optional[str]:
    """Convert an SFT example dict to Gemma-4 format."""
    instruction = ex.get("instruction", "").strip()
    inp         = ex.get("input", "").strip()
    output      = ex.get("output", "").strip()

    if not output or len(output) < 20:
        return None

    system = system_override or instruction or CORTEX_DEFAULT_SYSTEM
    user   = inp if inp else instruction

    if not user:
        return None

    return to_gemma4_format(system, user, output)


def fmt_dpo_example(ex: dict) -> Optional[dict]:
    """Convert a DPO/ORPO example to Gemma-4 format."""
    prompt   = ex.get("prompt", "").strip()
    chosen   = ex.get("chosen", "").strip()
    rejected = ex.get("rejected", "").strip()

    if not (prompt and chosen and rejected) or chosen == rejected:
        return None

    # For DPO, we format prompt as the user turn
    sys_prompt = CORTEX_DEFAULT_SYSTEM
    formatted_prompt = (
        f"<bos>"
        f"<start_of_turn>system\n{sys_prompt}<end_of_turn>\n"
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return {
        "prompt":   formatted_prompt,
        "chosen":   chosen + "<end_of_turn>",
        "rejected": rejected + "<end_of_turn>",
    }


# ─── Opus-4.6 Dataset Converter ──────────────────────────────────────────────
def detect_opus_format(sample: dict) -> dict:
    """Detect the field names in an Opus-4.6 dataset."""
    fields = set(sample.keys())
    # Common Opus field patterns
    if "conversations" in fields:
        return {"type": "conversations"}
    if "messages" in fields:
        return {"type": "messages"}
    if "prompt" in fields and "completion" in fields:
        return {"type": "prompt_completion"}
    if "instruction" in fields and "output" in fields:
        return {"type": "instruction_output"}
    if "question" in fields and "answer" in fields:
        return {"type": "qa"}
    if "input" in fields and "output" in fields:
        return {"type": "input_output"}
    return {"type": "unknown", "fields": list(fields)}


def convert_opus_example(ex: dict, fmt: dict) -> Optional[str]:
    """Convert a single Opus example to Gemma-4 format."""
    try:
        t = fmt["type"]

        if t == "conversations":
            convs = ex["conversations"]
            if len(convs) < 2:
                return None
            # Multi-turn: flatten to system + user + model
            system = CORTEX_AGENT_SYSTEM
            user_parts, assistant_parts = [], []
            for turn in convs:
                role = turn.get("role") or turn.get("from", "")
                content = turn.get("content") or turn.get("value", "")
                if role in ("system",):
                    system = content
                elif role in ("human", "user"):
                    user_parts.append(content)
                elif role in ("gpt", "assistant", "model"):
                    assistant_parts.append(content)
            if not user_parts or not assistant_parts:
                return None
            return to_gemma4_format(system, "\n\n".join(user_parts), "\n\n".join(assistant_parts))

        elif t == "messages":
            msgs = ex["messages"]
            system = CORTEX_AGENT_SYSTEM
            user, assistant = "", ""
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "system":
                    system = content
                elif role in ("user", "human"):
                    user += content + "\n"
                elif role in ("assistant", "model"):
                    assistant += content + "\n"
            if not user.strip() or not assistant.strip():
                return None
            return to_gemma4_format(system, user.strip(), assistant.strip())

        elif t == "prompt_completion":
            prompt = ex.get("prompt", "")
            completion = ex.get("completion", "")
            if not prompt or not completion:
                return None
            return to_gemma4_format(CORTEX_AGENT_SYSTEM, prompt, completion)

        elif t == "instruction_output":
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            output = ex.get("output", "")
            if not output:
                return None
            user = f"{instruction}\n\n{inp}".strip() if inp else instruction
            return to_gemma4_format(CORTEX_AGENT_SYSTEM, user, output)

        elif t == "qa":
            q = ex.get("question", "")
            a = ex.get("answer", "")
            if not q or not a:
                return None
            return to_gemma4_format(CORTEX_AGENT_SYSTEM, q, a)

        elif t == "input_output":
            inp = ex.get("input", "")
            out = ex.get("output", "")
            if not inp or not out:
                return None
            return to_gemma4_format(CORTEX_AGENT_SYSTEM, inp, out)

        else:
            # Generic fallback: concatenate all string fields
            text_fields = {k: v for k, v in ex.items() if isinstance(v, str) and len(v) > 20}
            if len(text_fields) < 2:
                return None
            items = list(text_fields.values())
            return to_gemma4_format(CORTEX_AGENT_SYSTEM, items[0], items[1])

    except Exception:
        return None


# ─── Edge-Agent Converter ─────────────────────────────────────────────────────
def convert_edge_agent_example(ex: dict) -> Optional[str]:
    """Convert Edge-Agent WebSearch example to Gemma-4 agentic format."""
    try:
        # Edge-Agent examples typically have query + multi-step search traces
        query = (
            ex.get("query") or ex.get("question") or
            ex.get("input") or ex.get("instruction") or ""
        ).strip()

        response = (
            ex.get("response") or ex.get("output") or
            ex.get("answer") or ex.get("completion") or ""
        ).strip()

        if not query or not response:
            # Try conversations format
            if "conversations" in ex:
                return convert_opus_example(ex, {"type": "conversations"})
            if "messages" in ex:
                return convert_opus_example(ex, {"type": "messages"})
            return None

        # Enrich with any tool_calls / search_results if present
        context_parts = []
        for field in ["tool_calls", "search_results", "steps", "reasoning", "context"]:
            val = ex.get(field)
            if val:
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False)[:2000]
                context_parts.append(f"[{field.upper()}]\n{val}")

        full_query = query
        if context_parts:
            full_query += "\n\n" + "\n\n".join(context_parts)

        # Add <think> wrapper if response doesn't already have reasoning
        if "<think>" not in response and len(response) > 100:
            # Wrap in think tag to match our training format
            response = f"<think>\nAnalyzing the query and available search context...\n</think>\n\n{response}"

        return to_gemma4_format(CORTEX_L1_SYSTEM, full_query, response)

    except Exception:
        return None


# ─── File processors ──────────────────────────────────────────────────────────
def process_jsonl(in_path: Path, converter_fn, out_path: Path, max_examples: Optional[int] = None):
    """Process a JSONL file through a converter and write output."""
    total, written, skipped = 0, 0, 0

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            if max_examples and total > max_examples:
                break
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = converter_fn(ex)
            if result is None:
                skipped += 1
                continue

            if isinstance(result, dict):
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps({"text": result}, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"  {in_path.name}: {total:,} total → {written:,} written ({skipped:,} skipped)")
    return written


def process_json_list(in_path: Path, converter_fn, out_path: Path, is_dpo: bool = False):
    """Process a JSON array file through a converter."""
    with in_path.open(encoding="utf-8") as f:
        data = json.load(f)

    total, written, skipped = len(data), 0, 0
    with out_path.open("w", encoding="utf-8") as fout:
        for ex in data:
            result = converter_fn(ex)
            if result is None:
                skipped += 1
                continue
            if isinstance(result, dict):
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps({"text": result}, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"  {in_path.name}: {total:,} total → {written:,} written ({skipped:,} skipped)")
    return written


def merge_jsonl_files(sources: list, out_path: Path, shuffle: bool = True):
    """Merge multiple JSONL files into one, optionally shuffling."""
    all_lines = []
    for src in sources:
        if Path(src).exists():
            with open(src, encoding="utf-8") as f:
                all_lines.extend(f.readlines())
    if shuffle:
        random.shuffle(all_lines)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(all_lines)
    log.info(f"  Merged {len(sources)} files → {len(all_lines):,} examples → {out_path.name}")


# ─── Phase processors ─────────────────────────────────────────────────────────
def prepare_phaseA(force: bool = False):
    """Phase A: Opus-4.6 reasoning datasets → Gemma-4 format."""
    out = OUT_DIR / "phaseA_opus_reasoning.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase A] Already exists: {out}")
        return

    log.info("\n[Phase A] Preparing Opus-4.6 reasoning datasets...")
    tmp_paths = []

    for name in ["opus_filtered", "opus_extended"]:
        raw = RAW_DIR / f"{name}.jsonl"
        if not raw.exists():
            log.warning(f"  {raw} not found — run download_gemma4_datasets.py first")
            continue

        # Detect format from first line
        with raw.open() as f:
            first = json.loads(f.readline())
        fmt = detect_opus_format(first)
        log.info(f"  {name}: detected format = {fmt['type']}")

        tmp = OUT_DIR / f"tmp_{name}.jsonl"
        process_jsonl(raw, lambda ex: convert_opus_example(ex, fmt), tmp)
        tmp_paths.append(tmp)

    if tmp_paths:
        merge_jsonl_files(tmp_paths, out)
        for t in tmp_paths:
            t.unlink(missing_ok=True)

    count = sum(1 for _ in out.open()) if out.exists() else 0
    log.info(f"[Phase A] Done: {count:,} examples → {out}")


def prepare_phaseB(force: bool = False):
    """Phase B: Edge-Agent WebSearch → Gemma-4 agentic format."""
    out = OUT_DIR / "phaseB_edge_agent.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase B] Already exists: {out}")
        return

    log.info("\n[Phase B] Preparing Edge-Agent WebSearch dataset...")
    raw = RAW_DIR / "edge_agent.jsonl"
    if not raw.exists():
        log.warning(f"  {raw} not found — run download_gemma4_datasets.py first")
        return

    process_jsonl(raw, convert_edge_agent_example, out)
    count = sum(1 for _ in out.open()) if out.exists() else 0
    log.info(f"[Phase B] Done: {count:,} examples → {out}")


def prepare_phaseD(force: bool = False):
    """Phase D: Agentic routing + function calling combined."""
    out = OUT_DIR / "phaseD_agentic_combined.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase D] Already exists: {out}")
        return

    log.info("\n[Phase D] Preparing agentic routing + function calling...")
    tmp_paths = []

    for stage_file in ["stage2_agentic.json", "stage13_function_calling.json"]:
        src = TRAINING_DATA / stage_file
        if not src.exists():
            log.warning(f"  {src} not found")
            continue
        tmp = OUT_DIR / f"tmp_{stage_file}.jsonl"
        process_json_list(src, fmt_sft_example, tmp)
        tmp_paths.append(tmp)

    if tmp_paths:
        merge_jsonl_files(tmp_paths, out)
        for t in tmp_paths:
            t.unlink(missing_ok=True)

    count = sum(1 for _ in out.open()) if out.exists() else 0
    log.info(f"[Phase D] Done: {count:,} examples → {out}")


def prepare_phaseH(force: bool = False):
    """Phase H: Wiki + summarization → combined memory operations."""
    out = OUT_DIR / "phaseH_wiki_memory.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase H] Already exists: {out}")
        return

    log.info("\n[Phase H] Preparing wiki + summarization data...")
    tmp_paths = []
    for stage_file in ["stage6_summarization.json"]:
        src = TRAINING_DATA / stage_file
        if not src.exists():
            continue
        tmp = OUT_DIR / f"tmp_{stage_file}.jsonl"
        process_json_list(src, fmt_sft_example, tmp)
        tmp_paths.append(tmp)

    # Also include Cortex-specific wiki ops if generated
    cortex_wiki = OUT_DIR / "cortex_wiki_ops.jsonl"
    if cortex_wiki.exists():
        tmp_paths.append(cortex_wiki)

    if tmp_paths:
        merge_jsonl_files(tmp_paths, out)
        for t in tmp_paths:
            if "tmp_" in t.name:
                t.unlink(missing_ok=True)

    count = sum(1 for _ in out.open()) if out.exists() else 0
    log.info(f"[Phase H] Done: {count:,} examples → {out}")


def prepare_phaseI(force: bool = False):
    """Phase I: Multi-turn dialogue + long-context combined."""
    out = OUT_DIR / "phaseI_dialogue_longctx.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase I] Already exists: {out}")
        return

    log.info("\n[Phase I] Preparing dialogue + long-context data...")
    tmp_paths = []
    for stage_file in ["stage7_dialogue.json", "stage8_longcontext.json"]:
        src = TRAINING_DATA / stage_file
        if not src.exists():
            continue
        tmp = OUT_DIR / f"tmp_{stage_file}.jsonl"
        process_json_list(src, fmt_sft_example, tmp)
        tmp_paths.append(tmp)

    if tmp_paths:
        merge_jsonl_files(tmp_paths, out)
        for t in tmp_paths:
            t.unlink(missing_ok=True)

    count = sum(1 for _ in out.open()) if out.exists() else 0
    log.info(f"[Phase I] Done: {count:,} examples → {out}")


def prepare_phaseJ(force: bool = False):
    """Phase J: Cortex-specific orchestrator data (generated separately)."""
    out = OUT_DIR / "phaseJ_orchestrator.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase J] Already exists: {out}")
        return

    log.info("\n[Phase J] Looking for Cortex orchestrator data...")
    cortex_src = OUT_DIR / "cortex_orchestrator_data.jsonl"
    if cortex_src.exists():
        import shutil
        shutil.copy(cortex_src, out)
        count = sum(1 for _ in out.open())
        log.info(f"[Phase J] Done: {count:,} examples → {out}")
    else:
        log.warning(f"[Phase J] Cortex orchestrator data not found.")
        log.warning("  Run: python scripts/generate_cortex_orchestrator_data.py first")


def prepare_phaseO(force: bool = False):
    """Phase O: RFT + SPIN combined."""
    out = OUT_DIR / "phaseO_rft_spin.jsonl"
    if out.exists() and not force:
        log.info(f"[Phase O] Already exists: {out}")
        return

    log.info("\n[Phase O] Preparing RFT + SPIN data...")
    tmp_paths = []
    for stage_file in ["stage14_rft.json", "stage15_spin.json"]:
        src = TRAINING_DATA / stage_file
        if not src.exists():
            continue
        tmp = OUT_DIR / f"tmp_{stage_file}.jsonl"
        process_json_list(src, fmt_sft_example, tmp)
        tmp_paths.append(tmp)

    if tmp_paths:
        merge_jsonl_files(tmp_paths, out)
        for t in tmp_paths:
            t.unlink(missing_ok=True)

    count = sum(1 for _ in out.open()) if out.exists() else 0
    log.info(f"[Phase O] Done: {count:,} examples → {out}")


def prepare_standard_stages(force: bool = False):
    """Prepare standard single-file stages (C, E, F, G, K, L, M, N)."""
    stage_map = {
        "stageC_faithfulness":   ("stage1_faithfulness.json",  False),
        "stageE_causal_temporal":("stage3_causal.json",         False),
        "stageF_selfrag_crag":   ("stage4_selfrag.json",        False),
        "stageG_belief_evolution":("stage5_belief.json",        False),
        "stageK_dpo":            ("stage9_dpo.json",            True),
        "stageL_orpo":           ("stage11_orpo.json",          True),
        "stageM_raft":           ("stage12_raft.json",          False),
        "stageN_function_calling":("stage13_function_calling.json", False),
    }

    for stage_name, (src_file, is_dpo) in stage_map.items():
        out = OUT_DIR / f"{stage_name}.jsonl"
        if out.exists() and not force:
            log.info(f"[{stage_name}] Already exists: {out}")
            continue

        src = TRAINING_DATA / src_file
        if not src.exists():
            log.warning(f"[{stage_name}] Source not found: {src}")
            continue

        log.info(f"\n[{stage_name}] Processing {src_file}...")
        converter = fmt_dpo_example if is_dpo else fmt_sft_example
        process_json_list(src, converter, out, is_dpo=is_dpo)


def validate_all():
    """Validate all output files exist and have reasonable content."""
    log.info("\n" + "="*60)
    log.info("VALIDATION REPORT")
    log.info("="*60)

    expected = [
        "phaseA_opus_reasoning.jsonl",
        "phaseB_edge_agent.jsonl",
        "stageC_faithfulness.jsonl",
        "phaseD_agentic_combined.jsonl",
        "stageE_causal_temporal.jsonl",
        "stageF_selfrag_crag.jsonl",
        "stageG_belief_evolution.jsonl",
        "phaseH_wiki_memory.jsonl",
        "phaseI_dialogue_longctx.jsonl",
        "phaseJ_orchestrator.jsonl",
        "stageK_dpo.jsonl",
        "stageL_orpo.jsonl",
        "stageM_raft.jsonl",
        "stageN_function_calling.jsonl",
        "phaseO_rft_spin.jsonl",
    ]

    total_examples = 0
    for fname in expected:
        path = OUT_DIR / fname
        if path.exists():
            count = sum(1 for _ in path.open())
            size_mb = path.stat().st_size / 1e6
            total_examples += count
            status = "✅" if count > 100 else "⚠️"
            log.info(f"  {status} {fname:<45} {count:>8,} examples  ({size_mb:.1f} MB)")
        else:
            log.info(f"  ❌ {fname:<45} MISSING")

    log.info(f"\n  Total prepared examples: {total_examples:,}")
    log.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Prepare Gemma-4 training datasets")
    parser.add_argument("--stage",    type=str, help="Prepare specific phase only")
    parser.add_argument("--force",    action="store_true", help="Re-process even if files exist")
    parser.add_argument("--validate", action="store_true", help="Validate existing files only")
    args = parser.parse_args()

    if args.validate:
        validate_all()
        return

    log.info("Cortex Lab — Gemma-4 Dataset Preparation")
    log.info(f"Output: {OUT_DIR}")

    if args.stage:
        fn_map = {
            "phaseA": prepare_phaseA,
            "phaseB": prepare_phaseB,
            "phaseD": prepare_phaseD,
            "phaseH": prepare_phaseH,
            "phaseI": prepare_phaseI,
            "phaseJ": prepare_phaseJ,
            "phaseO": prepare_phaseO,
        }
        if args.stage in fn_map:
            fn_map[args.stage](force=args.force)
        else:
            log.error(f"Unknown stage: {args.stage}")
        return

    # Full preparation
    prepare_phaseA(force=args.force)
    prepare_phaseB(force=args.force)
    prepare_standard_stages(force=args.force)
    prepare_phaseD(force=args.force)
    prepare_phaseH(force=args.force)
    prepare_phaseI(force=args.force)
    prepare_phaseJ(force=args.force)
    prepare_phaseO(force=args.force)

    validate_all()
    log.info("\nNext step:")
    log.info("  python scripts/generate_cortex_orchestrator_data.py")
    log.info("  python train_gemma4.py --phase 1")


if __name__ == "__main__":
    main()
