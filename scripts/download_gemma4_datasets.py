#!/usr/bin/env python3
"""
Cortex Lab — HuggingFace Dataset Downloader for Gemma-4 Training
=================================================================
Downloads:
  1. google/gemma-4-E2B-it          (model weights cache)
  2. nohurry/Opus-4.6-Reasoning-3000x-filtered
  3. Crownelius/Opus-4.6-Reasoning-3300x
  4. yatin-superintelligence/Edge-Agent-Reasoning-WebSearch-260K

Usage:
    python scripts/download_gemma4_datasets.py
    python scripts/download_gemma4_datasets.py --inspect   # show schema only
    python scripts/download_gemma4_datasets.py --model-only
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Gemma4Download")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

HF_TOKEN    = os.getenv("HF_TOKEN", "")
MODEL_ID    = os.getenv("MODEL_ID", "google/gemma-4-E2B-it")
DATA_OUT    = ROOT / "training_data" / "gemma4" / "raw"
DATA_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "opus_filtered": {
        "id":   "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "split": "train",
        "desc": "Opus-4.6 Reasoning 3000x filtered (high-quality reasoning chains)",
    },
    "opus_extended": {
        "id":   "Crownelius/Opus-4.6-Reasoning-3300x",
        "split": "train",
        "desc": "Opus-4.6 Reasoning 3300x (extended reasoning coverage)",
    },
    "edge_agent": {
        "id":   "yatin-superintelligence/Edge-Agent-Reasoning-WebSearch-260K",
        "split": "train",
        "desc": "Edge-Agent Reasoning WebSearch 260K (agentic tool-use traces)",
    },
}

# ─── Edge-Agent quality filtering ────────────────────────────────────────────
EDGE_AGENT_QUALITY_FILTERS = {
    "min_length_chars": 200,      # Skip very short examples
    "max_length_chars": 32000,    # Skip pathologically long examples
    "target_count":     15000,    # Keep top 15K by quality score
    "prefer_fields":    ["tool_calls", "search_results", "reasoning", "steps"],
}


def authenticate_hf():
    """Login to HuggingFace Hub with the stored token."""
    from huggingface_hub import login, whoami
    if not HF_TOKEN:
        log.error("HF_TOKEN not set. Check your .env file.")
        sys.exit(1)
    login(token=HF_TOKEN, add_to_git_credential=False)
    info = whoami()
    log.info(f"Authenticated as: {info['name']} ({info.get('email', 'no email')})")


def inspect_dataset(dataset_key: str):
    """Print schema and sample rows for a dataset without saving."""
    from datasets import load_dataset
    cfg = DATASETS[dataset_key]
    log.info(f"\n{'='*60}")
    log.info(f"Inspecting: {cfg['id']}")
    log.info(f"{'='*60}")
    ds = load_dataset(cfg["id"], split=cfg["split"], token=HF_TOKEN, streaming=True)
    sample = next(iter(ds))
    log.info(f"Fields: {list(sample.keys())}")
    log.info(f"Sample:\n{json.dumps({k: str(v)[:200] for k, v in sample.items()}, indent=2)}")


def quality_score_edge_agent(example: dict) -> float:
    """Score an Edge-Agent example for quality. Higher = better."""
    score = 0.0
    text = str(example)

    # Prefer examples with explicit tool calls / search steps
    for field in EDGE_AGENT_QUALITY_FILTERS["prefer_fields"]:
        if field in example and example[field]:
            score += 2.0

    # Prefer longer, richer examples
    length = len(text)
    if length > 1000:
        score += 1.0
    if length > 3000:
        score += 1.0
    if length > 6000:
        score += 0.5

    # Penalize examples that are mostly whitespace or repeated content
    unique_chars = len(set(text))
    if unique_chars < 50:
        score -= 5.0

    # Prefer examples that have explicit reasoning steps
    reasoning_markers = ["because", "therefore", "first,", "step", "reasoning:", "thought:"]
    for marker in reasoning_markers:
        if marker.lower() in text.lower():
            score += 0.3

    return score


def download_dataset(dataset_key: str, force: bool = False) -> Path:
    """Download a dataset and save as JSONL. Returns output path."""
    from datasets import load_dataset

    cfg = DATASETS[dataset_key]
    out_path = DATA_OUT / f"{dataset_key}.jsonl"

    if out_path.exists() and not force:
        count = sum(1 for _ in out_path.open(encoding='utf-8'))
        log.info(f"[{dataset_key}] Already downloaded: {count:,} examples → {out_path}")
        return out_path

    log.info(f"\n[{dataset_key}] Downloading: {cfg['id']}")
    log.info(f"  Description: {cfg['desc']}")

    ds = load_dataset(cfg["id"], split=cfg["split"], token=HF_TOKEN)
    log.info(f"  Loaded: {len(ds):,} total examples")
    log.info(f"  Fields: {ds.column_names}")

    # ── Special handling for Edge-Agent (260K → 15K quality subset) ──────────
    if dataset_key == "edge_agent":
        log.info(f"  Filtering Edge-Agent to top {EDGE_AGENT_QUALITY_FILTERS['target_count']:,}...")

        # Length filter
        min_len = EDGE_AGENT_QUALITY_FILTERS["min_length_chars"]
        max_len = EDGE_AGENT_QUALITY_FILTERS["max_length_chars"]
        filtered = [ex for ex in ds if min_len <= len(str(ex)) <= max_len]
        log.info(f"  After length filter: {len(filtered):,}")

        # Quality score + sort
        scored = [(quality_score_edge_agent(ex), ex) for ex in filtered]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top N
        target = EDGE_AGENT_QUALITY_FILTERS["target_count"]
        selected = [ex for _, ex in scored[:target]]
        log.info(f"  After quality filter: {len(selected):,} examples kept")

        with out_path.open("w", encoding="utf-8") as f:
            for ex in selected:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    else:
        # Full dataset — write all examples
        with out_path.open("w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    count = sum(1 for _ in out_path.open(encoding='utf-8'))
    log.info(f"  Saved: {count:,} examples → {out_path}")
    log.info(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


def prefetch_model():
    """Prefetch/cache Gemma-4-E2B-it model weights locally."""
    from huggingface_hub import snapshot_download
    log.info(f"\nPrefetching model: {MODEL_ID}")
    log.info("This will download ~4-8 GB depending on dtype...")
    cache_dir = snapshot_download(
        repo_id=MODEL_ID,
        token=HF_TOKEN,
        ignore_patterns=["*.pt", "*.ot"],   # Skip non-safetensor files
    )
    log.info(f"Model cached at: {cache_dir}")
    return cache_dir


def write_download_manifest(results: dict):
    """Write a manifest file recording what was downloaded and when."""
    import datetime
    manifest = {
        "downloaded_at": datetime.datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "datasets": results,
    }
    manifest_path = DATA_OUT / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"\nManifest written → {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Gemma-4 training datasets")
    parser.add_argument("--inspect",    action="store_true", help="Inspect schemas only, no download")
    parser.add_argument("--model-only", action="store_true", help="Only prefetch model weights")
    parser.add_argument("--data-only",  action="store_true", help="Only download datasets (skip model)")
    parser.add_argument("--force",      action="store_true", help="Re-download even if files exist")
    parser.add_argument("--dataset",    type=str, choices=list(DATASETS.keys()),
                        help="Download a specific dataset only")
    args = parser.parse_args()

    authenticate_hf()

    if args.inspect:
        for key in ([args.dataset] if args.dataset else DATASETS.keys()):
            inspect_dataset(key)
        return

    results = {}

    if not args.data_only:
        prefetch_model()

    if not args.model_only:
        targets = [args.dataset] if args.dataset else list(DATASETS.keys())
        for key in targets:
            path = download_dataset(key, force=args.force)
            results[key] = {
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1e6, 1),
                "examples": sum(1 for _ in path.open(encoding='utf-8')),
            }
        write_download_manifest(results)

    log.info("\n" + "="*60)
    log.info("DOWNLOAD COMPLETE")
    log.info("="*60)
    for k, v in results.items():
        log.info(f"  {k}: {v['examples']:,} examples ({v['size_mb']} MB)")
    log.info("\nNext step:")
    log.info("  python scripts/prepare_gemma4_datasets.py")


if __name__ == "__main__":
    main()
