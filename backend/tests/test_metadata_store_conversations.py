"""Regression tests for conversation-turn persistence in MetadataStore."""

from __future__ import annotations

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_metadata_store_persists_conversation_metadata_in_fallback(tmp_path, monkeypatch):
    import src.storage.metadata_store as metadata_store_module

    monkeypatch.setattr(metadata_store_module, "HAS_DUCKDB", False)

    store = metadata_store_module.MetadataStore(
        db_path=str(tmp_path / "fallback-cortex.duckdb")
    )
    store.store_conversation_turn(
        "session-123",
        "user",
        "Eva remember that I decided to ship the mobile companion.",
        metadata={
            "platform": "mobile",
            "analysis": {"direct_address": True, "reply_expected": True},
            "retention_trace": {
                "memory_decision": "priority",
                "tags": ["decision", "spoken_dialogue"],
            },
        },
    )
    store.store_conversation_turn(
        "session-123",
        "assistant",
        "I've saved that decision for the session forge.",
        metadata={
            "platform": "mobile",
            "source": "assistant_companion",
        },
    )
    store.flush()

    restored = metadata_store_module.MetadataStore(
        db_path=str(tmp_path / "fallback-cortex.duckdb")
    )
    turns = restored.get_conversation("session-123")

    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[0]["metadata"]["platform"] == "mobile"
    assert turns[0]["metadata"]["retention_trace"]["memory_decision"] == "priority"
    assert turns[1]["metadata"]["source"] == "assistant_companion"
