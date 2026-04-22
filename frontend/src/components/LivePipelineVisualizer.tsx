"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Activity,
  Search,
  Brain,
  Shield,
  Zap,
  Eye,
  RefreshCw,
  GitBranch,
  Shrink,
  TrendingUp,
  CheckCircle2,
  SkipForward,
  Loader2,
  Radio,
} from "lucide-react";
import { subscribePipelineEvents } from "@/lib/api";
import type { LivePipelineEvent } from "@/lib/types";

// Steps in expected order for the visualization pipeline
const PIPELINE_STEPS = [
  { key: "query_analysis", label: "Query Analysis", icon: Search, color: "blue" },
  { key: "routing", label: "LLM Routing", icon: GitBranch, color: "purple" },
  { key: "query_transform", label: "Query Transform", icon: RefreshCw, color: "cyan" },
  { key: "agent_execution", label: "Agent Execution", icon: Brain, color: "indigo" },
  { key: "compression", label: "Compression", icon: Shrink, color: "teal" },
  { key: "importance_boost", label: "Importance Boost", icon: TrendingUp, color: "pink" },
  { key: "crag", label: "CRAG Evaluation", icon: Shield, color: "amber" },
  { key: "self_rag", label: "Self-RAG Critique", icon: Eye, color: "emerald" },
  { key: "flare", label: "FLARE Retrieval", icon: Zap, color: "orange" },
] as const;

type StepStatus = "idle" | "running" | "completed" | "skipped";

interface StepState {
  status: StepStatus;
  duration_ms: number;
  details: Record<string, unknown>;
}

const colorMap: Record<string, { bg: string; border: string; text: string; glow: string; ring: string }> = {
  blue: { bg: "bg-blue-50", border: "border-blue-300", text: "text-blue-600", glow: "shadow-blue-200/50", ring: "ring-blue-400" },
  purple: { bg: "bg-purple-50", border: "border-purple-300", text: "text-purple-600", glow: "shadow-purple-200/50", ring: "ring-purple-400" },
  cyan: { bg: "bg-cyan-50", border: "border-cyan-300", text: "text-cyan-600", glow: "shadow-cyan-200/50", ring: "ring-cyan-400" },
  indigo: { bg: "bg-indigo-50", border: "border-indigo-300", text: "text-indigo-600", glow: "shadow-indigo-200/50", ring: "ring-indigo-400" },
  teal: { bg: "bg-teal-50", border: "border-teal-300", text: "text-teal-600", glow: "shadow-teal-200/50", ring: "ring-teal-400" },
  pink: { bg: "bg-pink-50", border: "border-pink-300", text: "text-pink-600", glow: "shadow-pink-200/50", ring: "ring-pink-400" },
  amber: { bg: "bg-amber-50", border: "border-amber-300", text: "text-amber-600", glow: "shadow-amber-200/50", ring: "ring-amber-400" },
  emerald: { bg: "bg-emerald-50", border: "border-emerald-300", text: "text-emerald-600", glow: "shadow-emerald-200/50", ring: "ring-emerald-400" },
  orange: { bg: "bg-orange-50", border: "border-orange-300", text: "text-orange-600", glow: "shadow-orange-200/50", ring: "ring-orange-400" },
};

export function LivePipelineVisualizer({ isActive }: { isActive: boolean }) {
  const [steps, setSteps] = useState<Record<string, StepState>>({});
  const [pipelineActive, setPipelineActive] = useState(false);
  const [totalMs, setTotalMs] = useState(0);
  const [connected, setConnected] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  const handleEvent = useCallback((event: LivePipelineEvent) => {
    switch (event.event_type) {
      case "pipeline_start":
        // Reset all steps for new pipeline run
        setSteps({});
        setPipelineActive(true);
        setTotalMs(0);
        break;

      case "step_start":
        setSteps((prev) => ({
          ...prev,
          [event.step_type]: {
            status: "running",
            duration_ms: 0,
            details: event.details,
          },
        }));
        break;

      case "step_complete":
        setSteps((prev) => ({
          ...prev,
          [event.step_type]: {
            status: "completed",
            duration_ms: event.duration_ms,
            details: event.details,
          },
        }));
        break;

      case "step_skip":
        setSteps((prev) => ({
          ...prev,
          [event.step_type]: {
            status: "skipped",
            duration_ms: 0,
            details: event.details,
          },
        }));
        break;

      case "pipeline_complete":
        setPipelineActive(false);
        setTotalMs(event.duration_ms);
        break;
    }
  }, []);

  // Connect/disconnect SSE based on isActive
  useEffect(() => {
    if (!isActive) {
      controllerRef.current?.abort();
      controllerRef.current = null;
      setConnected(false);
      return;
    }

    const ctrl = subscribePipelineEvents(
      (event) => {
        setConnected(true);
        handleEvent(event);
      },
      () => {
        setConnected(false);
        // Auto-reconnect after 3s
        setTimeout(() => {
          if (controllerRef.current) {
            controllerRef.current.abort();
            controllerRef.current = subscribePipelineEvents(handleEvent);
          }
        }, 3000);
      },
    );
    controllerRef.current = ctrl;
    setConnected(true);

    return () => {
      ctrl.abort();
      controllerRef.current = null;
    };
  }, [isActive, handleEvent]);

  const completedCount = Object.values(steps).filter((s) => s.status === "completed").length;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 backdrop-blur-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 py-2.5 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-white">
        <div className="relative">
          <Activity size={14} className="text-indigo-500" />
          {pipelineActive && (
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-emerald-400 rounded-full animate-ping" />
          )}
        </div>
        <span className="text-[11px] font-semibold text-slate-600">
          Live Pipeline
        </span>

        {/* Connection indicator */}
        <div className="flex items-center gap-1 ml-1">
          <Radio size={9} className={connected ? "text-emerald-500" : "text-slate-300"} />
          <span className={`text-[8px] ${connected ? "text-emerald-500" : "text-slate-400"}`}>
            {connected ? "LIVE" : "OFF"}
          </span>
        </div>

        <div className="flex items-center gap-1.5 ml-auto">
          {pipelineActive && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-indigo-50 text-indigo-600 border border-indigo-200 animate-pulse">
              Processing...
            </span>
          )}
          {completedCount > 0 && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-emerald-50 text-emerald-600 border border-emerald-200 font-mono">
              {completedCount} done
            </span>
          )}
          {totalMs > 0 && !pipelineActive && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-slate-50 text-slate-500 border border-slate-200 font-mono">
              {Math.round(totalMs)}ms
            </span>
          )}
        </div>
      </div>

      {/* Pipeline steps visualization */}
      <div className="px-3 py-2.5 space-y-1">
        {PIPELINE_STEPS.map(({ key, label, icon: Icon, color }) => {
          const step = steps[key];
          const status: StepStatus = step?.status ?? "idle";
          const c = colorMap[color];

          return (
            <div
              key={key}
              className={`
                flex items-center gap-2 py-1.5 px-2.5 rounded-xl transition-all duration-300
                ${status === "running"
                  ? `${c.bg} ${c.border} border shadow-md ${c.glow} ring-1 ${c.ring} ring-opacity-30`
                  : status === "completed"
                  ? `${c.bg} border ${c.border} border-opacity-50`
                  : status === "skipped"
                  ? "bg-slate-25 border border-slate-100 opacity-50"
                  : "bg-white border border-transparent"
                }
              `}
            >
              {/* Icon */}
              <div className={`flex-shrink-0 ${status === "idle" ? "text-slate-300" : c.text}`}>
                {status === "running" ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Icon size={13} />
                )}
              </div>

              {/* Label */}
              <span className={`text-[10px] font-medium min-w-[110px] ${
                status === "idle" ? "text-slate-300" : "text-slate-600"
              }`}>
                {label}
              </span>

              {/* Progress / Status */}
              <div className="flex-1 flex items-center justify-end gap-1.5">
                {status === "running" && (
                  <div className="flex-1 max-w-[80px] h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full bg-gradient-to-r ${
                      color === "blue" ? "from-blue-400 to-blue-500"
                      : color === "purple" ? "from-purple-400 to-purple-500"
                      : color === "indigo" ? "from-indigo-400 to-indigo-500"
                      : color === "amber" ? "from-amber-400 to-amber-500"
                      : color === "emerald" ? "from-emerald-400 to-emerald-500"
                      : color === "orange" ? "from-orange-400 to-orange-500"
                      : color === "cyan" ? "from-cyan-400 to-cyan-500"
                      : color === "teal" ? "from-teal-400 to-teal-500"
                      : "from-pink-400 to-pink-500"
                    } animate-pulse`}
                      style={{ width: "60%" }}
                    />
                  </div>
                )}

                {status === "completed" && step && (
                  <>
                    {/* Key detail chip */}
                    {step.details && Object.keys(step.details).length > 0 && (
                      <span className="text-[8px] px-1 py-0.5 rounded bg-white/70 text-slate-500 font-mono truncate max-w-[120px]">
                        {formatKeyDetail(key, step.details)}
                      </span>
                    )}
                    <span className="text-[9px] font-mono text-slate-400">
                      {step.duration_ms < 1 ? "<1" : Math.round(step.duration_ms)}ms
                    </span>
                    <CheckCircle2 size={10} className={c.text} />
                  </>
                )}

                {status === "skipped" && (
                  <>
                    <span className="text-[8px] text-slate-400 italic">
                      {(step?.details?.reason as string) || "skipped"}
                    </span>
                    <SkipForward size={9} className="text-slate-300" />
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Format the most important detail for each step type */
function formatKeyDetail(stepKey: string, details: Record<string, unknown>): string {
  switch (stepKey) {
    case "query_analysis": {
      const intent = details.intent as string || "";
      const complexity = details.complexity as number;
      return complexity != null ? `${intent} (${(complexity * 100).toFixed(0)}%)` : intent;
    }
    case "routing":
      return (details.refined_intent as string) || "";
    case "query_transform":
      return `${details.total_variants || 0} variants`;
    case "agent_execution":
      return `${details.evidence_count || 0} evidence`;
    case "compression":
      return details.compression_ratio != null
        ? `${((details.compression_ratio as number) * 100).toFixed(0)}% ratio`
        : "";
    case "importance_boost":
      return `${details.boosted_count || 0} boosted`;
    case "crag":
      return (details.verdict as string) || "";
    case "self_rag":
      return details.verdict
        ? `${details.verdict}${details.revision_applied ? " (revised)" : ""}`
        : "";
    case "flare":
      return `+${details.new_evidence || 0} evidence`;
    default:
      return "";
  }
}
