"""
PageIndex Integration Configuration for Cortex Lab
Controls when and how PageIndex cloud-based document retrieval
is used alongside local memory retrieval.

Privacy principle: Personal memories NEVER leave the device.
Only user-consented, non-sensitive documents go to PageIndex.
"""

import os

_PAGEINDEX_API_KEY = os.environ.get("PAGEINDEX_API_KEY", "").strip()
_PAGEINDEX_ENABLED_ENV = os.environ.get("PAGEINDEX_ENABLED", "").strip().lower()
if _PAGEINDEX_ENABLED_ENV in {"1", "true", "yes", "on"}:
    _PAGEINDEX_ENABLED = True
elif _PAGEINDEX_ENABLED_ENV in {"0", "false", "no", "off"}:
    _PAGEINDEX_ENABLED = False
else:
    _PAGEINDEX_ENABLED = bool(_PAGEINDEX_API_KEY)

PAGEINDEX_CONFIG = {
    # ── API Configuration ─────────────────────────────────────────
    "api_key": _PAGEINDEX_API_KEY,
    "enabled": _PAGEINDEX_ENABLED,  # Master on/off switch

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
