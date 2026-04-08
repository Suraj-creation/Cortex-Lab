"""Trace analytics helpers for /api/rag/traces payload generation."""

from __future__ import annotations

from typing import Any, Dict, List


def build_trace_analytics(traces: List[Dict[str, Any]], total_history_count: int) -> Dict[str, Any]:
    """Build aggregate analytics for a list of pipeline traces."""
    if not traces:
        return {
            "total_traces": total_history_count,
            "showing": 0,
            "avg_duration_ms": 0,
            "avg_confidence": 0,
            "avg_evidence_count": 0,
            "channel_usage": {},
            "step_stats": {},
            "crag_activation_rate": 0,
            "selfrag_activation_rate": 0,
            "flare_activation_rate": 0,
            "cache_hit_rate": 0,
            "stop_reason_distribution": {},
            "runtime_loop": {
                "avg_iterations": 0,
                "avg_tool_calls": 0,
            },
            "memory_quality": {
                "sample_count": 0,
                "avg_precision_at_k": 0,
                "avg_recall_proxy": 0,
                "extraction_hit_rate": 0,
            },
        }

    total_durations = [float(t.get("total_duration_ms", 0) or 0) for t in traces]
    avg_duration = sum(total_durations) / len(total_durations)
    avg_confidence = sum(float(t.get("final_confidence", 0) or 0) for t in traces) / len(traces)
    avg_evidence = sum(float(t.get("evidence_count", 0) or 0) for t in traces) / len(traces)

    channel_totals: Dict[str, Dict[str, float]] = {}
    for trace in traces:
        for channel in trace.get("retrieval_channels", []) or []:
            name = channel.get("channel", "unknown")
            if name not in channel_totals:
                channel_totals[name] = {
                    "total_results": 0,
                    "total_duration_ms": 0.0,
                    "usage_count": 0,
                }
            channel_totals[name]["total_results"] += channel.get("result_count", 0)
            channel_totals[name]["total_duration_ms"] += channel.get("duration_ms", 0)
            channel_totals[name]["usage_count"] += 1 if channel.get("result_count", 0) > 0 else 0

    step_stats: Dict[str, Dict[str, float]] = {}
    for trace in traces:
        for step in trace.get("steps", []) or []:
            step_type = step.get("step_type", "unknown")
            if step_type not in step_stats:
                step_stats[step_type] = {
                    "completed": 0,
                    "skipped": 0,
                    "total_duration_ms": 0.0,
                }
            status = step.get("status")
            if status == "completed":
                step_stats[step_type]["completed"] += 1
                step_stats[step_type]["total_duration_ms"] += step.get("duration_ms", 0)
            elif status == "skipped":
                step_stats[step_type]["skipped"] += 1

    crag_activated = sum(1 for t in traces if t.get("crag_evaluation") is not None)
    selfrag_activated = sum(1 for t in traces if t.get("self_rag_critique") is not None)
    flare_activated = sum(1 for t in traces if t.get("flare_trace") is not None)
    cache_hits = sum(1 for t in traces if t.get("cache_status", {}).get("hit", False))

    stop_reason_distribution: Dict[str, int] = {}
    for trace in traces:
        reason = (trace.get("stop_reason") or "unknown").strip() or "unknown"
        stop_reason_distribution[reason] = stop_reason_distribution.get(reason, 0) + 1

    iterations = [
        float((trace.get("runtime_loop_state") or {}).get("iterations_executed", 0) or 0)
        for trace in traces
    ]
    tool_calls = [
        float((trace.get("runtime_loop_state") or {}).get("tool_calls_executed", 0) or 0)
        for trace in traces
    ]

    quality_entries = []
    for trace in traces:
        generation_details = trace.get("generation_details") or {}
        quality = generation_details.get("personal_memory_quality")
        if isinstance(quality, dict):
            quality_entries.append(quality)

    avg_precision = 0.0
    avg_recall = 0.0
    extraction_rate = 0.0
    if quality_entries:
        avg_precision = sum(float(x.get("precision_at_k", 0) or 0) for x in quality_entries) / len(quality_entries)
        avg_recall = sum(float(x.get("recall_proxy", 0) or 0) for x in quality_entries) / len(quality_entries)
        extraction_rate = sum(1 for x in quality_entries if x.get("extraction_hit")) / len(quality_entries)

    return {
        "total_traces": total_history_count,
        "showing": len(traces),
        "avg_duration_ms": round(avg_duration, 1),
        "avg_confidence": round(avg_confidence, 3),
        "avg_evidence_count": round(avg_evidence, 1),
        "channel_usage": channel_totals,
        "step_stats": step_stats,
        "crag_activation_rate": round(crag_activated / len(traces), 3),
        "selfrag_activation_rate": round(selfrag_activated / len(traces), 3),
        "flare_activation_rate": round(flare_activated / len(traces), 3),
        "cache_hit_rate": round(cache_hits / len(traces), 3),
        "stop_reason_distribution": stop_reason_distribution,
        "runtime_loop": {
            "avg_iterations": round(sum(iterations) / len(iterations), 2),
            "avg_tool_calls": round(sum(tool_calls) / len(tool_calls), 2),
        },
        "memory_quality": {
            "sample_count": len(quality_entries),
            "avg_precision_at_k": round(avg_precision, 3),
            "avg_recall_proxy": round(avg_recall, 3),
            "extraction_hit_rate": round(extraction_rate, 3),
        },
    }
