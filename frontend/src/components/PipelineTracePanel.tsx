"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Activity,
  Search,
  Brain,
  Shield,
  Zap,
  RefreshCw,
  CheckCircle2,
  XCircle,
  SkipForward,
  Clock,
  Layers,
  GitBranch,
  BarChart3,
  Eye,
  Target,
  Loader2,
  X,
} from "lucide-react";
import type { PipelineTrace, PipelineStep } from "@/lib/types";
import { cancelRuntimeTask } from "@/lib/api";

interface Props {
  trace: PipelineTrace;
}

export function PipelineTracePanel({ trace }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const [cancellingCoordinator, setCancellingCoordinator] = useState(false);
  const [runtimeTaskActionMessage, setRuntimeTaskActionMessage] = useState("");

  const toggleStep = (idx: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const stepIcon = (step: PipelineStep) => {
    const iconClass = "flex-shrink-0";
    switch (step.step_type) {
      case "query_analysis":
        return <Search size={12} className={`${iconClass} text-blue-500`} />;
      case "routing":
        return <GitBranch size={12} className={`${iconClass} text-purple-500`} />;
      case "query_transform":
        return <RefreshCw size={12} className={`${iconClass} text-cyan-500`} />;
      case "agent_execution":
        return <Brain size={12} className={`${iconClass} text-indigo-500`} />;
      case "crag":
        return <Shield size={12} className={`${iconClass} text-amber-500`} />;
      case "self_rag":
        return <Eye size={12} className={`${iconClass} text-emerald-500`} />;
      case "flare":
        return <Zap size={12} className={`${iconClass} text-orange-500`} />;
      default:
        return <Activity size={12} className={`${iconClass} text-slate-400`} />;
    }
  };

  const statusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200">
            <CheckCircle2 size={8} />
            done
          </span>
        );
      case "skipped":
        return (
          <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full bg-slate-50 text-slate-400 border border-slate-200">
            <SkipForward size={8} />
            skip
          </span>
        );
      case "error":
        return (
          <span className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-full bg-red-50 text-red-500 border border-red-200">
            <XCircle size={8} />
            err
          </span>
        );
      default:
        return null;
    }
  };

  const cragVerdictColor = (verdict: string) => {
    switch (verdict) {
      case "CORRECT":
        return "text-emerald-600 bg-emerald-50 border-emerald-200";
      case "AMBIGUOUS":
        return "text-amber-600 bg-amber-50 border-amber-200";
      case "INCORRECT":
        return "text-red-500 bg-red-50 border-red-200";
      default:
        return "text-slate-500 bg-slate-50 border-slate-200";
    }
  };

  // Calculate timing percentages for the waterfall
  const totalMs = trace.total_duration_ms || 1;
  const completedSteps = trace.steps.filter((s) => s.status === "completed");
  const coordinatorTaskId =
    trace.coordinator_task_id
    || trace.subagent_spawn_records?.[0]?.parent_task_id
    || "";
  const subagentTaskIds = Array.from(new Set((trace.subagent_spawn_records || [])
    .map((record) => record.task_id)
    .filter((taskId) => Boolean(taskId))));

  const handleCancelCoordinator = async () => {
    if (!coordinatorTaskId) return;
    setCancellingCoordinator(true);
    try {
      await cancelRuntimeTask(coordinatorTaskId, "Cancelled from pipeline trace panel", true);
      setRuntimeTaskActionMessage("Coordinator cancellation submitted.");
    } catch (err) {
      setRuntimeTaskActionMessage(err instanceof Error ? err.message : "Failed to cancel coordinator task");
    } finally {
      setCancellingCoordinator(false);
    }
  };

  return (
    <div className="mt-2 rounded-2xl border border-slate-200 bg-white overflow-hidden transition-all duration-300">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left hover:bg-slate-50/50 transition-colors"
      >
        <Activity size={13} className="text-indigo-500 flex-shrink-0" />
        <span className="text-[11px] font-semibold text-slate-600">
          Pipeline Trace
        </span>
        <span className="text-[9px] text-slate-400 font-mono">
          {trace.trace_id}
        </span>

        {/* Summary badges */}
        <div className="flex items-center gap-1.5 ml-auto mr-2">
          <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-indigo-50 text-indigo-600 border border-indigo-200 font-medium">
            {completedSteps.length}/{trace.steps.length} steps
          </span>
          <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-slate-50 text-slate-500 border border-slate-200 font-mono flex items-center gap-1">
            <Clock size={8} />
            {Math.round(trace.total_duration_ms)}ms
          </span>
          {trace.crag_evaluation && (
            <span
              className={`text-[9px] px-1.5 py-0.5 rounded-md border font-semibold ${cragVerdictColor(
                trace.crag_evaluation.verdict
              )}`}
            >
              CRAG: {trace.crag_evaluation.verdict}
            </span>
          )}
        </div>

        {expanded ? (
          <ChevronDown size={14} className="text-slate-400 flex-shrink-0" />
        ) : (
          <ChevronRight size={14} className="text-slate-400 flex-shrink-0" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-slate-100 px-4 py-3 space-y-3">
          {/* ── Pipeline Waterfall ────────────────────────────── */}
          <div>
            <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Layers size={10} />
              Pipeline Steps
            </h4>
            <div className="space-y-1">
              {trace.steps.map((step, idx) => {
                const pct =
                  step.status === "completed"
                    ? Math.max((step.duration_ms / totalMs) * 100, 2)
                    : 0;
                const isExpanded = expandedSteps.has(idx);

                return (
                  <div key={idx}>
                    <button
                      onClick={() => toggleStep(idx)}
                      className="w-full flex items-center gap-2 py-1.5 px-2 rounded-lg hover:bg-slate-50 transition-colors group"
                    >
                      {stepIcon(step)}
                      <span className="text-[10px] font-medium text-slate-600 min-w-[140px] text-left">
                        {step.step_name}
                      </span>

                      {/* Timing bar */}
                      <div className="flex-1 h-3.5 bg-slate-50 rounded-full border border-slate-100 overflow-hidden relative">
                        {step.status === "completed" && (
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              step.step_type === "agent_execution"
                                ? "bg-gradient-to-r from-indigo-400 to-indigo-500"
                                : step.step_type === "crag"
                                ? "bg-gradient-to-r from-amber-400 to-amber-500"
                                : step.step_type === "self_rag"
                                ? "bg-gradient-to-r from-emerald-400 to-emerald-500"
                                : step.step_type === "flare"
                                ? "bg-gradient-to-r from-orange-400 to-orange-500"
                                : "bg-gradient-to-r from-blue-400 to-blue-500"
                            }`}
                            style={{ width: `${Math.min(pct, 100)}%` }}
                          />
                        )}
                        {step.status === "skipped" && (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="h-px w-full bg-slate-200 mx-2" style={{ backgroundImage: "repeating-linear-gradient(90deg, transparent, transparent 4px, #e2e8f0 4px, #e2e8f0 8px)" }} />
                          </div>
                        )}
                      </div>

                      <span className="text-[9px] text-slate-400 font-mono min-w-[45px] text-right">
                        {step.status === "completed"
                          ? `${Math.round(step.duration_ms)}ms`
                          : "—"}
                      </span>
                      {statusBadge(step.status)}
                      {isExpanded ? (
                        <ChevronDown size={10} className="text-slate-300" />
                      ) : (
                        <ChevronRight size={10} className="text-slate-300" />
                      )}
                    </button>

                    {/* Expanded step details */}
                    {isExpanded && step.details && Object.keys(step.details).length > 0 && (
                      <div className="ml-7 mt-1 mb-2 p-2.5 rounded-xl bg-slate-50/80 border border-slate-100">
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1.5">
                          {Object.entries(step.details).map(([key, value]) => (
                            <div key={key} className="flex items-start gap-1.5">
                              <span className="text-[9px] text-slate-400 font-medium whitespace-nowrap">
                                {key.replace(/_/g, " ")}:
                              </span>
                              <span className="text-[9px] text-slate-600 font-mono break-all">
                                {typeof value === "boolean"
                                  ? value
                                    ? "✓"
                                    : "✗"
                                  : typeof value === "number"
                                  ? Math.round(value * 1000) / 1000
                                  : Array.isArray(value)
                                  ? value.join(", ") || "—"
                                  : String(value ?? "—")}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── Runtime Task Links ─────────────────────────── */}
          {(coordinatorTaskId || subagentTaskIds.length > 0) && (
            <div>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Activity size={10} />
                Runtime Task Actions
              </h4>
              <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-2.5 space-y-1.5">
                {coordinatorTaskId ? (
                  <div className="text-[10px] text-slate-600 break-all">
                    <span className="font-semibold text-slate-700">Coordinator:</span>{" "}
                    <span className="font-mono text-slate-700">{coordinatorTaskId}</span>
                    <a
                      href={`/api/runtime/tasks/${encodeURIComponent(coordinatorTaskId)}`}
                      target="_blank"
                      rel="noreferrer"
                      className="ml-2 text-indigo-600 hover:text-indigo-700 underline"
                    >
                      view api
                    </a>
                    <button
                      onClick={handleCancelCoordinator}
                      disabled={cancellingCoordinator}
                      className="ml-2 inline-flex items-center gap-1 rounded-md border border-red-300 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700 hover:bg-red-100 disabled:opacity-60"
                    >
                      {cancellingCoordinator ? <Loader2 size={9} className="animate-spin" /> : <X size={9} />}
                      cancel
                    </button>
                  </div>
                ) : null}

                {subagentTaskIds.length > 0 ? (
                  <div className="text-[10px] text-slate-600 break-all">
                    <span className="font-semibold text-slate-700">Subagents:</span>{" "}
                    {subagentTaskIds.map((taskId) => (
                      <span key={taskId} className="mr-2">
                        <a
                          href={`/api/runtime/tasks/${encodeURIComponent(taskId)}`}
                          target="_blank"
                          rel="noreferrer"
                          className="font-mono text-indigo-600 hover:text-indigo-700 underline"
                        >
                          {taskId}
                        </a>
                      </span>
                    ))}
                  </div>
                ) : null}

                {runtimeTaskActionMessage ? (
                  <div className="rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1 text-[10px] text-indigo-700">
                    {runtimeTaskActionMessage}
                  </div>
                ) : null}
              </div>
            </div>
          )}

          {/* ── Retrieval Channels Breakdown ──────────────────── */}
          {trace.retrieval_channels.length > 0 && (
            <div>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Search size={10} />
                Retrieval Channels
              </h4>
              <div className="grid grid-cols-5 gap-1.5">
                {trace.retrieval_channels.map((ch) => {
                  const maxResults = Math.max(
                    ...trace.retrieval_channels.map((c) => c.result_count),
                    1
                  );
                  const barPct = (ch.result_count / maxResults) * 100;
                  const channelColors: Record<string, string> = {
                    dense: "from-blue-400 to-blue-500",
                    sparse: "from-purple-400 to-purple-500",
                    graph: "from-emerald-400 to-emerald-500",
                    temporal: "from-amber-400 to-amber-500",
                    proposition: "from-rose-400 to-rose-500",
                  };

                  return (
                    <div
                      key={ch.channel}
                      className="rounded-xl border border-slate-100 bg-slate-50/50 p-2 text-center"
                    >
                      <div className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
                        {ch.channel}
                      </div>
                      <div className="text-lg font-bold text-slate-700">
                        {ch.result_count}
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-full mt-1 overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-gradient-to-r ${
                            channelColors[ch.channel] || "from-slate-300 to-slate-400"
                          }`}
                          style={{ width: `${barPct}%` }}
                        />
                      </div>
                      <div className="text-[8px] text-slate-400 mt-1">
                        top: {(ch.top_score * 100).toFixed(0)}%
                      </div>
                    </div>
                  );
                })}
              </div>
              {trace.reranking && (
                <div className="mt-1.5 flex items-center gap-2 text-[9px] text-slate-400">
                  <Target size={9} />
                  <span>
                    Reranking: {trace.reranking.method} ({Math.round(trace.reranking.duration_ms)}ms)
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ── CRAG Evaluation ───────────────────────────────── */}
          {trace.crag_evaluation && (
            <div>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Shield size={10} />
                CRAG Quality Evaluation
              </h4>
              <div className="flex items-center gap-3 p-2.5 rounded-xl bg-slate-50/80 border border-slate-100">
                <div
                  className={`px-2.5 py-1 rounded-lg border text-[10px] font-bold tracking-wide ${cragVerdictColor(
                    trace.crag_evaluation.verdict
                  )}`}
                >
                  {trace.crag_evaluation.verdict}
                </div>
                <div className="flex-1 grid grid-cols-4 gap-2">
                  <div className="text-center">
                    <div className="text-[9px] text-slate-400">Quality</div>
                    <div className="text-xs font-bold text-slate-700">
                      {(trace.crag_evaluation.quality_score * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-[9px] text-slate-400">Avg Score</div>
                    <div className="text-xs font-bold text-slate-700">
                      {(trace.crag_evaluation.avg_evidence_score * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-[9px] text-slate-400">Entity Cov.</div>
                    <div className="text-xs font-bold text-slate-700">
                      {(trace.crag_evaluation.entity_coverage * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-[9px] text-slate-400">Evidence</div>
                    <div className="text-xs font-bold text-slate-700">
                      {trace.crag_evaluation.evidence_count}
                    </div>
                  </div>
                </div>
                {trace.crag_evaluation.supplementary_retrieved > 0 && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-blue-50 text-blue-500 border border-blue-200">
                    +{trace.crag_evaluation.supplementary_retrieved} supplementary
                  </span>
                )}
              </div>
            </div>
          )}

          {/* ── Self-RAG Critique ─────────────────────────────── */}
          {trace.self_rag_critique && (
            <div>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Eye size={10} />
                Self-RAG Critique Tokens
              </h4>
              <div className="p-2.5 rounded-xl bg-slate-50/80 border border-slate-100">
                <div className="grid grid-cols-3 gap-3 mb-2">
                  {[
                    { label: "ISREL", value: trace.self_rag_critique.isrel, desc: "Relevance", color: "text-blue-600" },
                    { label: "ISSUP", value: trace.self_rag_critique.issup, desc: "Faithfulness", color: "text-emerald-600" },
                    { label: "ISUSE", value: trace.self_rag_critique.isuse, desc: "Usefulness", color: "text-purple-600" },
                  ].map((item) => (
                    <div key={item.label} className="text-center">
                      <div className="text-[9px] text-slate-400">{item.desc}</div>
                      <div className={`text-lg font-bold ${item.color}`}>
                        {item.value}
                        <span className="text-[9px] text-slate-400">/10</span>
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-full mt-1 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            item.value >= 7
                              ? "bg-emerald-400"
                              : item.value >= 5
                              ? "bg-amber-400"
                              : "bg-red-400"
                          }`}
                          style={{ width: `${item.value * 10}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 text-[9px]">
                  <span
                    className={`px-1.5 py-0.5 rounded-md border font-semibold ${
                      trace.self_rag_critique.verdict === "ACCEPT"
                        ? "bg-emerald-50 text-emerald-600 border-emerald-200"
                        : trace.self_rag_critique.verdict === "REVISE"
                        ? "bg-amber-50 text-amber-600 border-amber-200"
                        : "bg-red-50 text-red-500 border-red-200"
                    }`}
                  >
                    {trace.self_rag_critique.verdict}
                  </span>
                  {trace.self_rag_critique.revision_applied && (
                    <span className="text-indigo-500 flex items-center gap-1">
                      <RefreshCw size={8} />
                      Revised ({trace.self_rag_critique.revision_focus})
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── FLARE Trace ───────────────────────────────────── */}
          {trace.flare_trace && trace.flare_trace.triggered && (
            <div>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Zap size={10} />
                FLARE Active Retrieval
              </h4>
              <div className="p-2.5 rounded-xl bg-slate-50/80 border border-slate-100">
                <div className="grid grid-cols-4 gap-2 text-center text-[9px]">
                  <div>
                    <div className="text-slate-400">Uncertain</div>
                    <div className="text-sm font-bold text-orange-600">
                      {trace.flare_trace.uncertain_sentences}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-400">Iterations</div>
                    <div className="text-sm font-bold text-orange-600">
                      {trace.flare_trace.retrieval_iterations}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-400">New Evidence</div>
                    <div className="text-sm font-bold text-orange-600">
                      +{trace.flare_trace.new_evidence_count}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-400">Δ Confidence</div>
                    <div
                      className={`text-sm font-bold ${
                        trace.flare_trace.confidence_delta > 0
                          ? "text-emerald-600"
                          : "text-red-500"
                      }`}
                    >
                      {trace.flare_trace.confidence_delta > 0 ? "+" : ""}
                      {(trace.flare_trace.confidence_delta * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
                {trace.flare_trace.answer_revised && (
                  <div className="mt-2 text-[9px] text-indigo-500 flex items-center gap-1">
                    <RefreshCw size={8} />
                    Answer was revised with augmented evidence
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Query Transformations ─────────────────────────── */}
          {trace.query_transform && trace.query_transform.total_variants > 0 && (
            <div>
              <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <RefreshCw size={10} />
                Query Transformations ({trace.query_transform.total_variants} variants)
              </h4>
              <div className="space-y-1.5">
                {trace.query_transform.multi_queries.length > 0 && (
                  <div className="p-2 rounded-lg bg-slate-50/80 border border-slate-100">
                    <div className="text-[9px] font-medium text-blue-500 mb-1">
                      RAG-Fusion Multi-Queries
                    </div>
                    {trace.query_transform.multi_queries.map((q, i) => (
                      <div
                        key={i}
                        className="text-[9px] text-slate-500 pl-2 border-l-2 border-blue-200 mb-0.5"
                      >
                        {q}
                      </div>
                    ))}
                  </div>
                )}
                {trace.query_transform.hyde_answer && (
                  <div className="p-2 rounded-lg bg-slate-50/80 border border-slate-100">
                    <div className="text-[9px] font-medium text-cyan-500 mb-1">
                      HyDE (Hypothetical Document)
                    </div>
                    <div className="text-[9px] text-slate-500 pl-2 border-l-2 border-cyan-200 line-clamp-2">
                      {trace.query_transform.hyde_answer}
                    </div>
                  </div>
                )}
                {trace.query_transform.step_back_query && (
                  <div className="p-2 rounded-lg bg-slate-50/80 border border-slate-100">
                    <div className="text-[9px] font-medium text-purple-500 mb-1">
                      Step-Back Query
                    </div>
                    <div className="text-[9px] text-slate-500 pl-2 border-l-2 border-purple-200">
                      {trace.query_transform.step_back_query}
                    </div>
                  </div>
                )}
                {trace.query_transform.sub_queries.length > 0 && (
                  <div className="p-2 rounded-lg bg-slate-50/80 border border-slate-100">
                    <div className="text-[9px] font-medium text-orange-500 mb-1">
                      Decomposed Sub-Queries
                    </div>
                    {trace.query_transform.sub_queries.map((q, i) => (
                      <div
                        key={i}
                        className="text-[9px] text-slate-500 pl-2 border-l-2 border-orange-200 mb-0.5"
                      >
                        {q}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Agent Routing & Summary ───────────────────────── */}
          <div className="flex items-center gap-3 pt-1 border-t border-slate-100 text-[9px] text-slate-400">
            <span className="flex items-center gap-1">
              <BarChart3 size={9} />
              Final: {(trace.final_confidence * 100).toFixed(0)}% confidence
            </span>
            <span className="flex items-center gap-1">
              <Activity size={9} />
              {trace.evidence_count} evidence
            </span>
            {trace.routing_decision && (
              <span className="flex items-center gap-1">
                <GitBranch size={9} />
                {trace.routing_decision}
              </span>
            )}
            {trace.agents_invoked.length > 0 && (
              <span className="flex items-center gap-1">
                <Brain size={9} />
                {trace.agents_invoked.map((a) => a.agent).join(", ")}
              </span>
            )}
            {trace.cache_status?.hit && (
              <span className="flex items-center gap-1 text-amber-500">
                <Zap size={9} />
                Cache {trace.cache_status.level}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
