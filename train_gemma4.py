#!/usr/bin/env python3
"""
Cortex Lab — Gemma-4-E2B-it Multi-Phase Training Pipeline
==========================================================
4-Phase curriculum fine-tuning on RTX 6000 Ada (24GB VRAM).

Usage:
  python train_gemma4.py --phase 1          # Foundation reasoning
  python train_gemma4.py --phase 2          # Cortex core
  python train_gemma4.py --phase 3          # Intelligence layer
  python train_gemma4.py --phase 4          # Alignment
  python train_gemma4.py --all              # Run all phases
  python train_gemma4.py --all --resume     # Resume from last checkpoint
  python train_gemma4.py --status           # Show training status
"""
import os, sys, json, time, shutil, argparse, logging, gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Gemma4Train")

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.gemma4_training_config import (
    BASE_MODEL, HF_TOKEN, QUANT_CONFIG, OUTPUT_DIR, TRAINING_DATA, GEMMA4_DATA,
    ALL_STAGES, FULL_STAGE_ORDER, PHASE1_ORDER, PHASE2_ORDER, PHASE3_ORDER, PHASE4_ORDER,
    DPO_STAGES, ORPO_STAGES, SFT_STAGES, PHASE_BOUNDARIES, FINAL_STAGE,
)

# ─── Lazy imports ─────────────────────────────────────────────────────────────
def _import_deps():
    global torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    global LoraConfig, get_peft_model, PeftModel
    global SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, ORPOTrainer, ORPOConfig
    global Dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, ORPOTrainer, ORPOConfig
    from datasets import Dataset
    log.info(f"torch={torch.__version__} | CUDA={torch.cuda.is_available()}")

# ─── VRAM Check ───────────────────────────────────────────────────────────────
def check_gpu():
    if not torch.cuda.is_available():
        log.error("CUDA not available!"); sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    log.info(f"GPU: {props.name} | VRAM: {total:.1f} GB | CC: {props.major}.{props.minor}")
    if total < 20:
        log.warning(f"Only {total:.1f}GB VRAM — may OOM on large stages")
    return total

# ─── Dataset Loading ──────────────────────────────────────────────────────────
def load_sft_dataset(stage: str) -> "Dataset":
    cfg = ALL_STAGES[stage]
    data_file = cfg["data_file"]
    # Try gemma4/ subdir first, then root training_data/
    path = TRAINING_DATA / data_file
    if not path.exists():
        path = GEMMA4_DATA / Path(data_file).name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    if path.suffix == ".jsonl":
        examples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    text = ex.get("text", "")
                    if len(text) >= 100:
                        examples.append({"text": text})
    else:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        examples = []
        for ex in raw:
            instr = ex.get("instruction", "").strip()
            inp = ex.get("input", "").strip()
            out = ex.get("output", "").strip()
            if not out or len(out) < 20:
                continue
            system = instr or "You are Cortex, a personal AI with persistent memory."
            user = inp if inp else instr
            text = (f"<bos><start_of_turn>system\n{system}<end_of_turn>\n"
                    f"<start_of_turn>user\n{user}<end_of_turn>\n"
                    f"<start_of_turn>model\n{out}<end_of_turn>")
            examples.append({"text": text})

    log.info(f"Loaded {len(examples):,} examples from {path.name}")
    return Dataset.from_list(examples)

def load_dpo_dataset(stage: str) -> "Dataset":
    cfg = ALL_STAGES[stage]
    path = TRAINING_DATA / cfg["data_file"]
    if not path.exists():
        raise FileNotFoundError(f"DPO dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f) if path.suffix == ".json" else [json.loads(l) for l in f if l.strip()]
    examples = []
    sys_p = "You are Cortex, a personal AI with persistent memory."
    for ex in raw:
        p = ex.get("prompt", "").strip()
        c = ex.get("chosen", "").strip()
        r = ex.get("rejected", "").strip()
        if not (p and c and r) or c == r:
            continue
        prompt = (f"<bos><start_of_turn>system\n{sys_p}<end_of_turn>\n"
                  f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n")
        examples.append({"prompt": prompt, "chosen": c + "<end_of_turn>", "rejected": r + "<end_of_turn>"})
    log.info(f"Loaded {len(examples):,} DPO pairs from {path.name}")
    return Dataset.from_list(examples)

# ─── Model Loading ────────────────────────────────────────────────────────────
def get_base_path(stage: str) -> str:
    idx = FULL_STAGE_ORDER.index(stage)
    if idx == 0:
        return BASE_MODEL
    for prev_idx in range(idx - 1, -1, -1):
        prev = FULL_STAGE_ORDER[prev_idx]
        merged = OUTPUT_DIR / prev / "merged"
        if merged.exists() and (merged / "config.json").exists():
            log.info(f"Using merged model from: {prev}")
            return str(merged)
        adapter = OUTPUT_DIR / prev / "adapter"
        if adapter.exists() and (adapter / "adapter_config.json").exists():
            log.info(f"Re-merging {prev} adapter...")
            remerge(prev)
            if merged.exists():
                return str(merged)
    log.warning(f"No previous merged model found for {stage}, using base")
    return BASE_MODEL

def remerge(stage: str):
    adapter_dir = OUTPUT_DIR / stage / "adapter"
    merged_dir = OUTPUT_DIR / stage / "merged"
    idx = FULL_STAGE_ORDER.index(stage)
    base = BASE_MODEL
    for i in range(idx - 1, -1, -1):
        p = OUTPUT_DIR / FULL_STAGE_ORDER[i] / "merged"
        if p.exists() and (p / "config.json").exists():
            base = str(p); break
    log.info(f"Re-merging {stage} from {base}")
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    pm = PeftModel.from_pretrained(m, str(adapter_dir), is_trainable=False)
    merged = pm.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tok.save_pretrained(str(merged_dir))
    del merged, pm, m; gc.collect()

def load_model(model_path: str, stage: str, is_dpo=False, is_orpo=False):
    bnb = BitsAndBytesConfig(
        load_in_4bit=QUANT_CONFIG["load_in_4bit"],
        bnb_4bit_compute_dtype=QUANT_CONFIG["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=QUANT_CONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=QUANT_CONFIG["bnb_4bit_quant_type"],
    )
    log.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", token=HF_TOKEN,
    )
    tok_path = model_path
    if not Path(model_path).exists() or not (Path(model_path) / "tokenizer.json").exists():
        tok_path = BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, padding_side="right", token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_cfg_dict = ALL_STAGES[stage]["lora"]
    lora_config = LoraConfig(**lora_cfg_dict)

    if not is_dpo and not is_orpo:
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    return model, tokenizer, lora_config

# ─── Training Functions ──────────────────────────────────────────────────────
def train_sft(stage, model, tokenizer, dataset, out_dir, cfg, resume=None):
    sft_cfg = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing") else None,
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        optim=cfg.get("optim", "adamw_torch"),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=cfg.get("bf16", True),
        max_seq_length=cfg.get("max_seq_length", 2048),
        dataset_text_field="text",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset, processing_class=tokenizer)
    log.info(f"SFT: {stage} | {len(dataset):,} examples | {cfg['num_train_epochs']} epochs | lr={cfg['learning_rate']}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    log.info(f"SFT {stage} done in {timedelta(seconds=int(time.time()-t0))}")
    return trainer

def train_dpo(stage, model, tokenizer, lora_config, dataset, out_dir, cfg, resume=None):
    dpo_cfg = DPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=cfg["learning_rate"], num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        warmup_ratio=cfg.get("warmup_ratio", 0.1), beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 2048), max_prompt_length=cfg.get("max_prompt_length", 1024),
        bf16=cfg.get("bf16", True), logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"), save_total_limit=cfg.get("save_total_limit", 2),
        report_to="none", remove_unused_columns=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainer = DPOTrainer(model=model, ref_model=None, args=dpo_cfg, train_dataset=dataset, processing_class=tokenizer)
    log.info(f"DPO: {stage} | {len(dataset):,} pairs | beta={cfg.get('beta', 0.1)}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    log.info(f"DPO {stage} done in {timedelta(seconds=int(time.time()-t0))}")
    return trainer

def train_orpo(stage, model, tokenizer, lora_config, dataset, out_dir, cfg, resume=None):
    orpo_cfg = ORPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=cfg["learning_rate"], num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing") else None,
        warmup_ratio=cfg.get("warmup_ratio", 0.1), beta=cfg.get("beta", 0.1),
        max_length=cfg.get("max_length", 1024), bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10), save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2), report_to="none", remove_unused_columns=False,
        optim=cfg.get("optim", "adamw_torch"), weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    trainer = ORPOTrainer(model=model, args=orpo_cfg, train_dataset=dataset, processing_class=tokenizer)
    log.info(f"ORPO: {stage} | {len(dataset):,} pairs | reference-free")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    log.info(f"ORPO {stage} done in {timedelta(seconds=int(time.time()-t0))}")
    return trainer

# ─── Save / Merge / Cleanup ──────────────────────────────────────────────────
def save_adapter(trainer, out_dir, stage):
    ad = out_dir / "adapter"
    ad.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ad))
    trainer.processing_class.save_pretrained(str(ad))
    log.info(f"Adapter saved → {ad}")

def merge_adapter(out_dir, stage):
    ad = out_dir / "adapter"
    md = out_dir / "merged"
    if not ad.exists():
        log.error(f"No adapter found at {ad}"); return
    idx = FULL_STAGE_ORDER.index(stage)
    base = BASE_MODEL
    for i in range(idx - 1, -1, -1):
        p = OUTPUT_DIR / FULL_STAGE_ORDER[i] / "merged"
        if p.exists() and (p / "config.json").exists():
            base = str(p); break
    log.info(f"Merging {stage} | base={base}")
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True, token=HF_TOKEN)
    pm = PeftModel.from_pretrained(m, str(ad))
    merged = pm.merge_and_unload()
    md.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(md))
    tok = AutoTokenizer.from_pretrained(str(ad), trust_remote_code=True)
    tok.save_pretrained(str(md))
    (md / "merge_meta.json").write_text(json.dumps({"stage": stage, "merged_at": datetime.now().isoformat(), "base": base}, indent=2))
    log.info(f"Merged model saved → {md}")
    del merged, pm, m; gc.collect()

def save_meta(out_dir, stage, elapsed, examples, loss=None):
    meta = {"stage": stage, "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": int(elapsed), "examples": examples,
            "final_loss": loss, "gpu": torch.cuda.get_device_name(0)}
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

def cleanup_prev_merged(stage: str):
    """Remove previous stage's merged model to save disk. Keeps adapter for rollback."""
    idx = FULL_STAGE_ORDER.index(stage)
    if idx < 2:
        return  # Keep first two stages' merged models
    # Clean up the stage TWO positions back (keep N-1 for rollback)
    cleanup_idx = idx - 2
    if cleanup_idx >= 0:
        prev = FULL_STAGE_ORDER[cleanup_idx]
        prev_merged = OUTPUT_DIR / prev / "merged"
        prev_adapter = OUTPUT_DIR / prev / "adapter"
        if prev_merged.exists() and prev_adapter.exists():
            log.info(f"Disk cleanup: removing merged model from {prev} (adapter preserved)")
            shutil.rmtree(prev_merged)

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ─── Stage Runner ─────────────────────────────────────────────────────────────
def run_stage(stage: str, resume: bool = False):
    cfg = ALL_STAGES[stage]
    out_dir = OUTPUT_DIR / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    if (out_dir / "training_meta.json").exists() and not resume:
        log.info(f"[{stage}] Already completed — skipping")
        return

    log.info(f"\n{'='*60}")
    log.info(f"STAGE: {stage}")
    log.info(f"  {cfg['description']}")
    log.info(f"  Trainer: {cfg['trainer'].upper()}")
    log.info(f"  Epochs:  {cfg['train']['num_train_epochs']}")
    log.info(f"{'='*60}")

    # Determine base model
    base_path = get_base_path(stage)

    # Check for resume checkpoint
    resume_ckpt = None
    if resume:
        ckpt_dir = out_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and "checkpoint" in d.name])
            if ckpts:
                resume_ckpt = str(ckpts[-1])
                log.info(f"Resuming from: {resume_ckpt}")

    is_dpo = stage in DPO_STAGES
    is_orpo = stage in ORPO_STAGES
    trainer_type = cfg["trainer"]

    # Load model
    model, tokenizer, lora_config = load_model(base_path, stage, is_dpo=is_dpo, is_orpo=is_orpo)

    # Load dataset
    if is_dpo or is_orpo:
        dataset = load_dpo_dataset(stage)
    else:
        dataset = load_sft_dataset(stage)

    # Train
    t0 = time.time()
    if trainer_type == "sft":
        trainer = train_sft(stage, model, tokenizer, dataset, out_dir, cfg["train"], resume_ckpt)
    elif trainer_type == "dpo":
        trainer = train_dpo(stage, model, tokenizer, lora_config, dataset, out_dir, cfg["train"], resume_ckpt)
    elif trainer_type == "orpo":
        trainer = train_orpo(stage, model, tokenizer, lora_config, dataset, out_dir, cfg["train"], resume_ckpt)
    else:
        raise ValueError(f"Unknown trainer: {trainer_type}")
    elapsed = time.time() - t0

    # Save adapter
    save_adapter(trainer, out_dir, stage)

    # Get final loss
    final_loss = None
    if trainer.state.log_history:
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            final_loss = losses[-1]

    # Save metadata
    save_meta(out_dir, stage, elapsed, len(dataset), final_loss)
    log.info(f"Stage {stage} complete | loss={final_loss} | time={timedelta(seconds=int(elapsed))}")

    # Free VRAM before merge
    del trainer, model, tokenizer, dataset
    free_vram()

    # Merge adapter into weights
    merge_adapter(out_dir, stage)

    # Cleanup old merged models to save disk
    cleanup_prev_merged(stage)

    # Final VRAM cleanup
    free_vram()

# ─── Phase Runner ─────────────────────────────────────────────────────────────
def run_phase(phase_num: int, resume: bool = False):
    phase_map = {1: PHASE1_ORDER, 2: PHASE2_ORDER, 3: PHASE3_ORDER, 4: PHASE4_ORDER}
    stages = phase_map.get(phase_num)
    if not stages:
        log.error(f"Invalid phase: {phase_num}"); return

    log.info(f"\n{'#'*60}")
    log.info(f"# PHASE {phase_num} — {len(stages)} stages")
    log.info(f"{'#'*60}")

    for stage in stages:
        run_stage(stage, resume=resume)

    log.info(f"\n{'#'*60}")
    log.info(f"# PHASE {phase_num} COMPLETE")
    log.info(f"{'#'*60}")

# ─── Status ───────────────────────────────────────────────────────────────────
def print_status():
    print(f"\n{'='*70}")
    print(f"  CORTEX LAB — Gemma-4-E2B-it Training Status")
    print(f"  Base: {BASE_MODEL}")
    print(f"{'='*70}")
    done = 0
    for phase_num, (label, stages) in enumerate({
        "PHASE 1 — FOUNDATION": PHASE1_ORDER,
        "PHASE 2 — CORTEX CORE": PHASE2_ORDER,
        "PHASE 3 — INTELLIGENCE": PHASE3_ORDER,
        "PHASE 4 — ALIGNMENT": PHASE4_ORDER,
    }.items(), 1):
        print(f"\n  ─── {label} ───")
        for s in stages:
            out = OUTPUT_DIR / s
            meta = out / "training_meta.json"
            merged = out / "merged"
            adapter = out / "adapter"
            if meta.exists():
                m = json.loads(meta.read_text())
                loss = m.get("final_loss", "?")
                status = f"✅ done (loss={loss})"
                done += 1
            elif merged.exists():
                status = "✅ merged"; done += 1
            elif adapter.exists():
                status = "⚡ adapter only"
            elif (out / "checkpoints").exists():
                status = "🔄 in-progress"
            else:
                status = "⬜ not started"
            trainer_tag = ALL_STAGES[s]["trainer"].upper()
            print(f"    {s:<35} {status:<30} [{trainer_tag}]")
    print(f"\n  Completed: {done}/{len(FULL_STAGE_ORDER)} stages")
    print(f"{'='*70}\n")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Gemma-4-E2B-it Training Pipeline")
    parser.add_argument("--phase",  type=int, choices=[1,2,3,4], help="Run specific phase")
    parser.add_argument("--stage",  type=str, help="Run specific stage")
    parser.add_argument("--all",    action="store_true", help="Run all 4 phases")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    if args.status:
        print_status(); return

    # Import heavy deps
    _import_deps()
    check_gpu()

    # HF auth
    from huggingface_hub import login
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)

    if args.stage:
        if args.stage not in ALL_STAGES:
            log.error(f"Unknown stage: {args.stage}"); sys.exit(1)
        run_stage(args.stage, resume=args.resume)
    elif args.phase:
        run_phase(args.phase, resume=args.resume)
    elif args.all:
        for p in [1, 2, 3, 4]:
            run_phase(p, resume=args.resume)
        log.info("\n🎉 ALL PHASES COMPLETE — Model ready for deployment!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
