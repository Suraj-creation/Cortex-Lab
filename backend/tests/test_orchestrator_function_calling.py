"""Regression tests for orchestrator function-call normalization."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.orchestrator import normalize_function_call_result


def test_normalize_function_call_result_accepts_gemini_tool_shape():
    normalized = normalize_function_call_result(
        {
            "tool": "find_entity",
            "arguments": {"entity_name": "Eva"},
        }
    )

    assert normalized["tool_name"] == "find_entity"
    assert normalized["arguments"] == {"entity_name": "Eva"}


def test_normalize_function_call_result_maps_null_tool_to_none():
    normalized = normalize_function_call_result({"tool": None, "arguments": []})

    assert normalized["tool_name"] == "none"
    assert normalized["arguments"] == {}
