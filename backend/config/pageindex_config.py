"""
PageIndex Integration Configuration for Cortex Lab
Controls when and how PageIndex cloud-based document retrieval
is used alongside local memory retrieval.

Privacy principle: Personal memories NEVER leave the device.
Only user-consented, non-sensitive documents go to PageIndex.
"""

import os

PAGEINDEX_CONFIG = {
    # ── API Configuration ─────────────────────────────────────────
    "api_key": os.environ.get(
        "PAGEINDEX_API_KEY", "8aa9ad8830aa438c926efc748b5489a9"
    ),
    "enabled": True,  # Master on/off switch

    # ── Privacy Controls ──────────────────────────────────────────
    "allow_cloud_upload": True,
    "sensitive_data_filter": True,  # Auto-detect & block PII
    "allowed_sources": [
        "pdf_upload", "research_paper", "document", "report",
    ],
    "blocked_sources": [
        "chat", "voice", "personal_note", "manual",
    ],

    # ── Retrieval Configuration ───────────────────────────────────
    "channel_weight": 0.20,      # Weight in RRF fusion (6th channel)
    "enable_streaming": True,
    "fallback_to_local": True,   # If PageIndex API fails, local-only retrieval
    "timeout_seconds": 15,       # Max wait for PageIndex API response

    # ── Cost Controls ─────────────────────────────────────────────
    "max_monthly_queries": 500,
    "max_monthly_pages": 2000,
    "track_usage": True,
}
