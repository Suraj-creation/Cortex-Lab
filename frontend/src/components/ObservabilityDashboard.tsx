"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ArrowLeft,
  Loader2,
  RefreshCw,
  Activity,
  Clock,
  Target,
  Zap,
  Search,
  GitBranch,
  ShieldCheck,
  Eye,
  BarChart3,
  TrendingUp,
  ChevronDown,
  ChevronRight,
  Database,
  Cpu,
} from "lucide-react";
import { PipelineTrace, TracesResponse } from "@/lib/types";
import { getPipelineTraces } from "@/lib/api";

/* ─── Helpers ────────────────────────────────────────────────────── */

function fmtMs(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-emerald-500",
  skipped: "bg-slate-300",
  error: "bg-red-500",
  pending: "bg-amber-400",
  running: "bg-blue-500",
};

const STATUS_TEXT: Record<string, string> = {
  completed: "text-emerald-600",
  skipped: "text-slate-400",
  error: "text-red-600",
  pending: "text-amber-600",
  running: "text-blue-600",
};

/* ─── Component ──────────────────────────────────────────────────── */

export function ObservabilityDashboard({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<TracesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedTrace, setExpandedTrace] = useState<string | null>(null);
  const [traceLimit, setTraceLimit] = useState(20);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getPipelineTraces(traceLimit);
      setData(res);
    } catch (err) {
      console.error("Failed to load traces:", err);
    } finally {
      setLoading(false);
    }
  }, [traceLimit]);

  useEffect(() => {
    load();
    const interval = setInterval(load, 10000);
    return () => clearInterval(interval);
  }, [load]);

  const analytics = data?.analytics;
  const traces = data?.traces ?? [];

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="rounded-lg p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h1 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
              <Activity size={22} className="text-indigo-500" />
              Pipeline Observability
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Real-time Agentic RAG pipeline monitoring &amp; analytics
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={traceLimit}
            onChange={(e) => setTraceLimit(Number(e.target.value))}
            className="text-xs border border-slate-200 rounded-lg px-2 py-1.5 text-slate-600 bg-white"
          >
            <option value={10}>Last 10</option>
            <option value={20}>Last 20</option>
            <option value={50}>Last 50</option>
            <option value={100}>Last 100</option>
          </select>
          <button
            onClick={load}
            disabled={loading}
            className="rounded-lg p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all disabled:opacity-50"
          >
            <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {loading && !data ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="animate-spin text-indigo-400" size={32} />
        </div>
      ) : (
        <>
          {/* ── Aggregate Analytics Cards ──────────────────────────── */}
          {analytics && analytics.total_traces > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              <MetricCard
                icon={<Database size={16} className="text-indigo-500" />}
                label="Total Queries"
                value={analytics.total_traces.toString()}
              />
              <MetricCard
                icon={<Clock size={16} className="text-amber-500" />}
                label="Avg Latency"
                value={fmtMs(analytics.avg_duration_ms)}
              />
              <MetricCard
                icon={<Target size={16} className="text-emerald-500" />}
                label="Avg Confidence"
                value={pct(analytics.avg_confidence)}
              />
              <MetricCard
                icon={<Search size={16} className="text-blue-500" />}
                label="Avg Evidence"
                value={analytics.avg_evidence_count.toFixed(1)}
              />
              <MetricCard
                icon={<Zap size={16} className="text-violet-500" />}
                label="Cache Hit Rate"
                value={pct(analytics.cache_hit_rate)}
              />
              <MetricCard
                icon={<ShieldCheck size={16} className="text-rose-500" />}
                label="CRAG Active"
                value={pct(analytics.crag_activation_rate)}
              />
            </div>
          )}

          {/* ── Advanced Quality Gates ─────────────────────────────── */}
          {analytics && analytics.total_traces > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* CRAG Panel */}
              <QualityGateCard
                title="CRAG Evaluation"
                subtitle="Corrective RAG quality gate"
                icon={<ShieldCheck size={16} />}
                rate={analytics.crag_activation_rate}
                color="rose"
                stats={analytics.step_stats["crag"]}
              />
              {/* Self-RAG Panel */}
              <QualityGateCard
                title="Self-RAG Critique"
                subtitle="Self-reflective token assessment"
                icon={<Eye size={16} />}
                rate={analytics.selfrag_activation_rate}
                color="violet"
                stats={analytics.step_stats["self_rag"]}
              />
              {/* FLARE Panel */}
              <QualityGateCard
                title="FLARE Active Retrieval"
                subtitle="Forward-looking active retrieval"
                icon={<TrendingUp size={16} />}
                rate={analytics.flare_activation_rate}
                color="blue"
                stats={analytics.step_stats["flare"]}
              />
            </div>
          )}

          {/* ── Retrieval Channel Usage ───────────────────────────── */}
          {analytics && Object.keys(analytics.channel_usage).length > 0 && (
            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <BarChart3 size={16} className="text-indigo-500" />
                Retrieval Channel Usage (Aggregate)
              </h3>
              <div className="space-y-3">
                {Object.entries(analytics.channel_usage).map(([ch, stat]) => {
                  const maxResults = Math.max(
                    ...Object.values(analytics.channel_usage).map((s) => s.total_results),
                    1
                  );
                  const barPct = (stat.total_results / maxResults) * 100;
                  const channelColors: Record<string, string> = {
                    dense: "bg-blue-500",
                    sparse: "bg-amber-500",
                    graph: "bg-emerald-500",
                    temporal: "bg-violet-500",
                    proposition: "bg-rose-500",
                  };
                  return (
                    <div key={ch} className="flex items-center gap-3">
                      <span className="text-xs font-medium text-slate-600 w-24 capitalize">
                        {ch}
                      </span>
                      <div className="flex-1 h-5 bg-slate-100 rounded-full overflow-hidden relative">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${channelColors[ch] ?? "bg-slate-400"}`}
                          style={{ width: `${barPct}%` }}
                        />
                        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold text-slate-700">
                          {stat.total_results} results · {stat.usage_count} queries · {fmtMs(stat.total_duration_ms)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Pipeline Step Breakdown ───────────────────────────── */}
          {analytics && Object.keys(analytics.step_stats).length > 0 && (
            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
                <GitBranch size={16} className="text-indigo-500" />
                Pipeline Step Breakdown
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-100">
                      <th className="text-left py-2 px-3 text-slate-500 font-medium">Step</th>
                      <th className="text-center py-2 px-3 text-slate-500 font-medium">Completed</th>
                      <th className="text-center py-2 px-3 text-slate-500 font-medium">Skipped</th>
                      <th className="text-center py-2 px-3 text-slate-500 font-medium">Run Rate</th>
                      <th className="text-right py-2 px-3 text-slate-500 font-medium">Total Time</th>
                      <th className="text-right py-2 px-3 text-slate-500 font-medium">Avg Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analytics.step_stats).map(([step, stat]) => {
                      const total = stat.completed + stat.skipped;
                      const runRate = total > 0 ? stat.completed / total : 0;
                      const avgTime = stat.completed > 0 ? stat.total_duration_ms / stat.completed : 0;
                      return (
                        <tr key={step} className="border-b border-slate-50 hover:bg-slate-50/50">
                          <td className="py-2 px-3 font-medium text-slate-700 capitalize">
                            {step.replace(/_/g, " ")}
                          </td>
                          <td className="py-2 px-3 text-center">
                            <span className="inline-flex items-center gap-1 text-emerald-600">
                              {stat.completed}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-center text-slate-400">{stat.skipped}</td>
                          <td className="py-2 px-3 text-center">
                            <span className={`font-semibold ${runRate > 0.8 ? "text-emerald-600" : runRate > 0.3 ? "text-amber-600" : "text-slate-400"}`}>
                              {pct(runRate)}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-right text-slate-600">{fmtMs(stat.total_duration_ms)}</td>
                          <td className="py-2 px-3 text-right text-slate-600">{fmtMs(avgTime)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Individual Trace Timeline ─────────────────────────── */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-700 mb-4 flex items-center gap-2">
              <Activity size={16} className="text-indigo-500" />
              Recent Pipeline Traces
              <span className="ml-auto text-[10px] text-slate-400 font-normal">
                {traces.length} trace{traces.length !== 1 ? "s" : ""}
              </span>
            </h3>

            {traces.length === 0 ? (
              <p className="text-sm text-slate-400 text-center py-8">
                No pipeline traces yet. Send a RAG query to start collecting traces.
              </p>
            ) : (
              <div className="space-y-2">
                {traces.map((trace, idx) => (
                  <TraceRow
                    key={`${trace.trace_id}-${idx}`}
                    trace={trace}
                    expanded={expandedTrace === trace.trace_id}
                    onToggle={() =>
                      setExpandedTrace((prev) =>
                        prev === trace.trace_id ? null : trace.trace_id
                      )
                    }
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

/* ─── Metric Card ────────────────────────────────────────────────── */

function MetricCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3.5 flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-xl font-bold text-slate-800">{value}</span>
    </div>
  );
}

/* ─── Quality Gate Card ──────────────────────────────────────────── */

function QualityGateCard({
  title,
  subtitle,
  icon,
  rate,
  color,
  stats,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  rate: number;
  color: string;
  stats?: { completed: number; skipped: number; total_duration_ms: number };
}) {
  const colorMap: Record<string, { bg: string; text: string; ring: string; bar: string }> = {
    rose: { bg: "bg-rose-50", text: "text-rose-600", ring: "ring-rose-200", bar: "bg-rose-500" },
    violet: { bg: "bg-violet-50", text: "text-violet-600", ring: "ring-violet-200", bar: "bg-violet-500" },
    blue: { bg: "bg-blue-50", text: "text-blue-600", ring: "ring-blue-200", bar: "bg-blue-500" },
  };
  const c = colorMap[color] ?? colorMap.blue;

  return (
    <div className={`rounded-2xl border border-slate-200 ${c.bg} p-4`}>
      <div className="flex items-center gap-2 mb-3">
        <span className={c.text}>{icon}</span>
        <div>
          <h4 className={`text-sm font-semibold ${c.text}`}>{title}</h4>
          <p className="text-[10px] text-slate-400">{subtitle}</p>
        </div>
      </div>
      {/* Activation rate bar */}
      <div className="mb-2">
        <div className="flex justify-between text-[10px] mb-1">
          <span className="text-slate-500">Activation Rate</span>
          <span className={`font-bold ${c.text}`}>{pct(rate)}</span>
        </div>
        <div className="h-2 bg-white/60 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${c.bar} transition-all duration-700`}
            style={{ width: `${Math.max(rate * 100, 1)}%` }}
          />
        </div>
      </div>
      {stats && (
        <div className="grid grid-cols-3 gap-2 mt-3 text-center">
          <div>
            <span className="text-sm font-bold text-slate-800">{stats.completed}</span>
            <p className="text-[9px] text-slate-400">Activated</p>
          </div>
          <div>
            <span className="text-sm font-bold text-slate-400">{stats.skipped}</span>
            <p className="text-[9px] text-slate-400">Skipped</p>
          </div>
          <div>
            <span className="text-sm font-bold text-slate-600">{fmtMs(stats.total_duration_ms)}</span>
            <p className="text-[9px] text-slate-400">Total Time</p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Individual Trace Row ───────────────────────────────────────── */

function TraceRow({
  trace,
  expanded,
  onToggle,
}: {
  trace: PipelineTrace;
  expanded: boolean;
  onToggle: () => void;
}) {
  const completedSteps = trace.steps?.filter((s) => s.status === "completed").length ?? 0;
  const totalSteps = trace.steps?.length ?? 0;
  const ts = trace.timestamp ? new Date(trace.timestamp).toLocaleTimeString() : "";
  const confidenceColor =
    (trace.final_confidence ?? 0) >= 0.7
      ? "text-emerald-600"
      : (trace.final_confidence ?? 0) >= 0.4
        ? "text-amber-600"
        : "text-red-500";

  return (
    <div className="border border-slate-100 rounded-xl overflow-hidden hover:border-slate-200 transition-all">
      {/* Summary Row */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-50/50 transition-colors"
      >
        {expanded ? (
          <ChevronDown size={14} className="text-slate-400 flex-shrink-0" />
        ) : (
          <ChevronRight size={14} className="text-slate-400 flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-slate-700 truncate">
            &quot;{trace.query}&quot;
          </p>
          <p className="text-[10px] text-slate-400 mt-0.5">
            {ts} · {trace.trace_id}
          </p>
        </div>
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="text-center">
            <p className="text-xs font-bold text-slate-700">{fmtMs(trace.total_duration_ms ?? 0)}</p>
            <p className="text-[9px] text-slate-400">Latency</p>
          </div>
          <div className="text-center">
            <p className={`text-xs font-bold ${confidenceColor}`}>
              {pct(trace.final_confidence ?? 0)}
            </p>
            <p className="text-[9px] text-slate-400">Confidence</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-bold text-slate-700">{trace.evidence_count ?? 0}</p>
            <p className="text-[9px] text-slate-400">Evidence</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-bold text-indigo-600">
              {completedSteps}/{totalSteps}
            </p>
            <p className="text-[9px] text-slate-400">Steps</p>
          </div>
          {trace.cache_status?.hit && (
            <span className="text-[9px] bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded-full font-medium">
              CACHED
            </span>
          )}
        </div>
      </button>

      {/* Expanded Detail */}
      {expanded && (
        <div className="border-t border-slate-100 bg-slate-50/30 px-4 py-4 space-y-4">
          {/* Step Timeline */}
          <div>
            <h4 className="text-[11px] font-semibold text-slate-600 mb-2 flex items-center gap-1.5">
              <GitBranch size={12} /> Pipeline Steps
            </h4>
            <div className="space-y-1">
              {trace.steps?.map((step, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${STATUS_COLORS[step.status] ?? "bg-slate-300"}`} />
                  <span className={`text-[11px] w-48 ${STATUS_TEXT[step.status] ?? "text-slate-500"}`}>
                    {step.step_name}
                  </span>
                  <span className="text-[10px] text-slate-400 w-16 text-right">
                    {step.duration_ms > 0 ? fmtMs(step.duration_ms) : "—"}
                  </span>
                  {step.details && Object.keys(step.details).length > 0 && (
                    <span className="text-[9px] text-slate-400 truncate max-w-[300px]">
                      {Object.entries(step.details)
                        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                        .join(" · ")}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Retrieval Channels */}
          {trace.retrieval_channels && trace.retrieval_channels.length > 0 && (
            <div>
              <h4 className="text-[11px] font-semibold text-slate-600 mb-2 flex items-center gap-1.5">
                <Search size={12} /> Retrieval Channels
              </h4>
              <div className="grid grid-cols-5 gap-2">
                {trace.retrieval_channels.map((ch) => (
                  <div
                    key={ch.channel}
                    className={`rounded-lg p-2 text-center ${
                      ch.result_count > 0 ? "bg-white border border-slate-200" : "bg-slate-50 border border-slate-100"
                    }`}
                  >
                    <p className="text-[10px] text-slate-500 capitalize font-medium">{ch.channel}</p>
                    <p className={`text-sm font-bold ${ch.result_count > 0 ? "text-slate-800" : "text-slate-300"}`}>
                      {ch.result_count}
                    </p>
                    {ch.result_count > 0 && (
                      <>
                        <p className="text-[9px] text-slate-400">
                          top: {ch.top_score?.toFixed(3)} · {fmtMs(ch.duration_ms ?? 0)}
                        </p>
                      </>
                    )}
                  </div>
                ))}
              </div>
              {trace.reranking && (
                <p className="text-[10px] text-slate-400 mt-1.5">
                  Reranking: {trace.reranking.method} · {fmtMs(trace.reranking.duration_ms ?? 0)}
                  {trace.reranking.input_count ? ` · ${trace.reranking.input_count} candidates` : ""}
                </p>
              )}
            </div>
          )}

          {/* Query Transform */}
          {trace.query_transform && (
            <div>
              <h4 className="text-[11px] font-semibold text-slate-600 mb-2 flex items-center gap-1.5">
                <Cpu size={12} /> Query Transformation
              </h4>
              <div className="bg-white border border-slate-200 rounded-lg p-3 text-[11px] space-y-1">
                <p className="text-slate-600">
                  <span className="font-medium text-slate-700">Original:</span> {trace.query_transform.original_query}
                </p>
                {trace.query_transform.multi_queries && trace.query_transform.multi_queries.length > 0 && (
                  <div>
                    <span className="font-medium text-slate-700">Multi-Queries ({trace.query_transform.multi_queries.length}):</span>
                    <ul className="mt-1 ml-4 list-disc text-slate-500">
                      {trace.query_transform.multi_queries.map((q, i) => (
                        <li key={i}>{q}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {trace.query_transform.hyde_answer && (
                  <p className="text-slate-500">
                    <span className="font-medium text-slate-700">HyDE:</span>{" "}
                    {trace.query_transform.hyde_answer.substring(0, 200)}...
                  </p>
                )}
                {trace.query_transform.step_back_query && (
                  <p className="text-slate-500">
                    <span className="font-medium text-slate-700">Step-Back:</span>{" "}
                    {trace.query_transform.step_back_query}
                  </p>
                )}
                <p className="text-slate-400">
                  {trace.query_transform.total_variants} variants · {fmtMs(trace.query_transform.duration_ms ?? 0)}
                </p>
              </div>
            </div>
          )}

          {/* Token Usage */}
          {trace.token_usage && Object.keys(trace.token_usage).length > 0 && (
            <div className="flex items-center gap-4 text-[10px] text-slate-400 bg-white border border-slate-200 rounded-lg px-3 py-2">
              <span className="flex items-center gap-1">
                <Cpu size={10} />
                {trace.token_usage.call_count ?? 0} LLM calls
              </span>
              <span>{trace.token_usage.total_tokens ?? 0} tokens</span>
              <span>{fmtMs(trace.token_usage.total_time_ms ?? 0)} LLM time</span>
              {trace.token_usage.avg_latency_ms && (
                <span>{fmtMs(trace.token_usage.avg_latency_ms)} avg/call</span>
              )}
              {trace.generation_details && "model" in trace.generation_details && (
                <span className="ml-auto text-slate-500">{String(trace.generation_details.model)}</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
