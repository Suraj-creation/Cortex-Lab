"""
Cortex Lab — Gemma-4-E2B-it Training Configuration
Hardware: NVIDIA RTX 6000 Ada Generation (24GB VRAM)
Model:    google/gemma-4-E2B-it
Method:   QLoRA 4-bit NF4 + BF16 + SFTTrainer + DPOTrainer + ORPOTrainer

4-Phase Curriculum:
  Phase 1 — FOUNDATION REASONING  (Opus-4.6 + Edge-Agent datasets)
  Phase 2 — CORTEX CORE           (faithfulness, routing, causal, self-RAG)
  Phase 3 — INTELLIGENCE LAYER    (belief, wiki, dialogue, orchestrator)
  Phase 4 — ALIGNMENT             (DPO, ORPO, RAFT, function-calling, SPIN)
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TRAINING_DATA  = PROJECT_ROOT / "training_data"
GEMMA4_DATA    = TRAINING_DATA / "gemma4"
OUTPUT_DIR     = PROJECT_ROOT / "fine_tuned_gemma4"
LOG_DIR        = OUTPUT_DIR / "logs"

for d in [GEMMA4_DATA, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Model & Auth ────────────────────────────────────────────────────────────
BASE_MODEL = os.getenv("MODEL_ID", "google/gemma-4-E2B-it")
HF_TOKEN   = os.getenv("HF_TOKEN", "")

# ─── Quantization (4-bit NF4 QLoRA) ─────────────────────────────────────────
QUANT_CONFIG = {
    "load_in_4bit":            True,
    "bnb_4bit_compute_dtype":  torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type":     "nf4",
}

# ─── Target modules — ALL transformer projection layers ──────────────────────
ALL_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — FOUNDATION REASONING
#  Trains the base model on deep Opus-level reasoning + agentic web-search.
#  Highest epochs (6) because these datasets are the foundation of all
#  subsequent behavior — they must be deeply internalized.
# ─────────────────────────────────────────────────────────────────────────────
PHASE1_STAGES = {

    "stageA_opus_reasoning": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseA_opus_reasoning.jsonl",
        "description": "Opus-4.6 deep chain-of-thought reasoning foundation",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            6,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,   # effective batch = 16
            "learning_rate":               2e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              4096,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageB_edge_agent": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseB_edge_agent.jsonl",
        "description": "Edge-Agent WebSearch agentic tool-use (15K filtered subset)",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            6,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               1.5e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              4096,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               20,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — CORTEX CORE
#  Builds the core Cortex Lab behaviors on top of the reasoning foundation.
#  Faithfulness → Routing → Causal → Self-RAG.
#  5 epochs each — smaller datasets need more passes to compete with
#  the much larger Phase 1 pre-training signal.
# ─────────────────────────────────────────────────────────────────────────────
PHASE2_STAGES = {

    "stageC_faithfulness": {
        "trainer":    "sft",
        "data_file":  "stage1_faithfulness.json",
        "description": "RAG-grounded faithfulness, evidence citation, calibrated refusal",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               2e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageD_agentic_routing": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseD_agentic_combined.jsonl",
        "description": "L1 routing JSON, intent classification, multi-query, function calling",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               2e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageE_causal_temporal": {
        "trainer":    "sft",
        "data_file":  "stage3_causal.json",
        "description": "Causal chain tracing, temporal narrative, Timeline + Causal agents",
        "lora": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               1.5e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageF_selfrag_crag": {
        "trainer":    "sft",
        "data_file":  "stage4_selfrag.json",
        "description": "ISREL/ISSUP/ISUSE critique tokens, CRAG relevance evaluation",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               2e-4,
            "warmup_ratio":                0.08,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — INTELLIGENCE LAYER
#  Higher-order intelligence: belief tracking, wiki operations, multi-turn
#  dialogue, orchestrator-specific L0/L1/L2 behavior, Deep Applications.
#  Uses longer sequences (up to 8192) to leverage Gemma-4's 128K window.
# ─────────────────────────────────────────────────────────────────────────────
PHASE3_STAGES = {

    "stageG_belief_evolution": {
        "trainer":    "sft",
        "data_file":  "stage5_belief.json",
        "description": "Contradiction detection, belief shifts, Reflection Agent",
        "lora": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               1.5e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageH_wiki_memory": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseH_wiki_memory.jsonl",
        "description": "Wiki PATCH/CREATE/LINT/COMPACT ops + memory plane lifecycle P1→P4",
        "lora": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               1.5e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageI_dialogue_longctx": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseI_dialogue_longctx.jsonl",
        "description": "Multi-turn coherence + long-context multi-hop reasoning (8K tokens)",
        "lora": {
            "r": 48, "lora_alpha": 96,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,  # effective batch = 16
            "learning_rate":               1.5e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               0.5,  # tighter clipping for long ctx
            "max_seq_length":              8192,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageJ_orchestrator": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseJ_orchestrator.jsonl",
        "description": "L0/L1/L2 orchestrator behaviors, 15 agents, Deep Applications, SIA wake",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            6,   # More epochs — core orchestration behavior
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               2e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              4096,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — ALIGNMENT & ADVANCED
#  DPO → ORPO → RAFT → Function-Calling → RFT → SPIN.
#  These alignment stages must run AFTER all SFT stages to avoid
#  corrupting the factual/behavioral capabilities.
# ─────────────────────────────────────────────────────────────────────────────
PHASE4_STAGES = {

    "stageK_dpo": {
        "trainer":    "dpo",
        "data_file":  "stage9_dpo.json",
        "description": "DPO: prefer grounded/empathetic/calibrated answers",
        "lora": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ATTN_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               5e-6,
            "warmup_ratio":                0.1,
            "beta":                        0.1,
            "max_length":                  2048,
            "max_prompt_length":           1024,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
        },
    },

    "stageL_orpo": {
        "trainer":    "orpo",
        "data_file":  "stage11_orpo.json",
        "description": "ORPO: reference-free preference optimization",
        "lora": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ATTN_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               8e-6,
            "warmup_ratio":                0.1,
            "beta":                        0.1,
            "max_length":                  1024,
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
        },
    },

    "stageM_raft": {
        "trainer":    "sft",
        "data_file":  "stage12_raft.json",
        "description": "RAFT: retrieval-augmented fine-tuning, document-grounded generation",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate":               1e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               0.5,
            "max_seq_length":              4096,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageN_function_calling": {
        "trainer":    "sft",
        "data_file":  "stage13_function_calling.json",
        "description": "Native Gemma-4 function calling + Cortex tool contracts",
        "lora": {
            "r": 64, "lora_alpha": 128,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               1.5e-4,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },

    "stageO_rft_spin": {
        "trainer":    "sft",
        "data_file":  "gemma4/phaseO_rft_spin.jsonl",
        "description": "Rejection sampling FT + SPIN self-play (merged dataset)",
        "lora": {
            "r": 32, "lora_alpha": 64,
            "target_modules": ALL_MODULES,
            "lora_dropout": 0.03, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "train": {
            "num_train_epochs":            5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate":               8e-5,
            "warmup_ratio":                0.05,
            "lr_scheduler_type":           "cosine",
            "optim":                       "adamw_torch",
            "weight_decay":                0.01,
            "max_grad_norm":               1.0,
            "max_seq_length":              2048,
            "bf16":                        True,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "epoch",
            "save_total_limit":            2,
            "dataloader_num_workers":      2,
        },
    },
}

# ─── Stage order for dependency chain ────────────────────────────────────────
PHASE1_ORDER = list(PHASE1_STAGES.keys())
PHASE2_ORDER = list(PHASE2_STAGES.keys())
PHASE3_ORDER = list(PHASE3_STAGES.keys())
PHASE4_ORDER = list(PHASE4_STAGES.keys())

ALL_STAGES = {
    **PHASE1_STAGES,
    **PHASE2_STAGES,
    **PHASE3_STAGES,
    **PHASE4_STAGES,
}

FULL_STAGE_ORDER = PHASE1_ORDER + PHASE2_ORDER + PHASE3_ORDER + PHASE4_ORDER

# Stages that are DPO (need DPOTrainer)
DPO_STAGES  = {"stageK_dpo"}
# Stages that are ORPO (need ORPOTrainer, reference-free)
ORPO_STAGES = {"stageL_orpo"}
# Stages that use SFT
SFT_STAGES  = set(FULL_STAGE_ORDER) - DPO_STAGES - ORPO_STAGES

# Phase boundaries — used for disk cleanup logic
PHASE_BOUNDARIES = {
    "phase1": PHASE1_ORDER,
    "phase2": PHASE2_ORDER,
    "phase3": PHASE3_ORDER,
    "phase4": PHASE4_ORDER,
}

# Final phase — never clean its merged model (it's the production model)
FINAL_PHASE = "phase4"
FINAL_STAGE = PHASE4_ORDER[-1]
