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
  Check,
  X,
  ShieldAlert,
} from "lucide-react";
import { PipelineTrace, RuntimePermissionRequest, RuntimeTaskSnapshot, TracesResponse } from "@/lib/types";
import {
  cancelRuntimeTask,
  getObservabilityMetrics,
  getPipelineTraces,
  getRuntimeSafetyExecutorStatus,
  getRuntimeSafetyPermissions,
  getRuntimeTasks,
  resolveRuntimeSafetyPermission,
  subscribeRuntimeTaskEvents,
} from "@/lib/api";

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

function sloPass(rate: number, target: number, direction: "min" | "max"): boolean {
  return direction === "min" ? rate >= target : rate <= target;
}

/* ─── Component ──────────────────────────────────────────────────── */

export function ObservabilityDashboard({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<TracesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedTrace, setExpandedTrace] = useState<string | null>(null);
  const [traceLimit, setTraceLimit] = useState(20);
  const [liveMetrics, setLiveMetrics] = useState<Record<string, unknown> | null>(null);
  const [pendingPermissions, setPendingPermissions] = useState<RuntimePermissionRequest[]>([]);
  const [expiredCount, setExpiredCount] = useState(0);
  const [executorStatus, setExecutorStatus] = useState<{
    enabled: boolean;
    running: boolean;
    summary: {
      approved_total: number;
      pending_total: number;
      running: number;
      waiting_retry: number;
      completed: number;
      failed: number;
      unsupported: number;
      idle: number;
    };
  } | null>(null);
  const [resolvingIds, setResolvingIds] = useState<Record<string, boolean>>({});
  const [approvalMessage, setApprovalMessage] = useState<string>("");
  const [runtimeTasks, setRuntimeTasks] = useState<RuntimeTaskSnapshot[]>([]);
  const [runtimeTaskMessage, setRuntimeTaskMessage] = useState<string>("");
  const [taskStreamConnected, setTaskStreamConnected] = useState(false);
  const [cancellingTaskIds, setCancellingTaskIds] = useState<Record<string, boolean>>({});
  const [taskNotifications, setTaskNotifications] = useState<Array<{
    id: string;
    taskId: string;
    fromState: string;
    toState: string;
    timestamp: string;
  }>>([]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [res, metrics, permissions, executor, tasksPayload] = await Promise.all([
        getPipelineTraces(traceLimit),
        getObservabilityMetrics().catch(() => null),
        getRuntimeSafetyPermissions().catch(() => ({ count: 0, pending: [], expired_count: 0 })),
        getRuntimeSafetyExecutorStatus().catch(() => null),
        getRuntimeTasks().catch(() => ({ count: 0, tasks: [] })),
      ]);
      setData(res);
      if (metrics) setLiveMetrics(metrics);
      setPendingPermissions(permissions.pending || []);
      setExpiredCount(permissions.expired_count || 0);
      setExecutorStatus(executor);
      setRuntimeTasks(tasksPayload.tasks || []);
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

  useEffect(() => {
    const controller = subscribeRuntimeTaskEvents(
      (event) => {
        setTaskStreamConnected(true);
        setRuntimeTasks((prev) => {
          const idx = prev.findIndex((task) => task.task_id === event.task.task_id);
          if (idx < 0) {
            return [event.task, ...prev].slice(0, 128);
          }
          const next = [...prev];
          next[idx] = event.task;
          return next;
        });

        if (event.event_type === "task_transition") {
          const fromState = event.previous_state || "unknown";
          const toState = event.state;
          const timestamp = event.timestamp || new Date().toISOString();
          setTaskNotifications((prev) => [
            {
              id: `${event.event_id}:${timestamp}`,
              taskId: event.task.task_id,
              fromState,
              toState,
              timestamp,
            },
            ...prev,
          ].slice(0, 12));
        }
      },
      () => {
        setTaskStreamConnected(false);
      },
    );

    return () => {
      controller.abort();
    };
  }, []);

  const handleResolvePermission = useCallback(
    async (permissionId: string, approve: boolean) => {
      setResolvingIds((prev) => ({ ...prev, [permissionId]: true }));
      try {
        await resolveRuntimeSafetyPermission(
          permissionId,
          approve,
          "observability-ui",
          approve ? "Approved from observability panel" : "Denied from observability panel",
        );
        setApprovalMessage(
          approve
            ? `Approved permission ${permissionId}`
            : `Denied permission ${permissionId}`,
        );
        await load();
      } catch (err) {
        setApprovalMessage(err instanceof Error ? err.message : "Failed to resolve permission");
      } finally {
        setResolvingIds((prev) => ({ ...prev, [permissionId]: false }));
      }
    },
    [load],
  );

  const handleCancelTask = useCallback(
    async (taskId: string) => {
      setCancellingTaskIds((prev) => ({ ...prev, [taskId]: true }));
      try {
        const result = await cancelRuntimeTask(
          taskId,
          "Cancelled from observability panel",
          true,
        );
        setRuntimeTaskMessage(
          result.cancelled_task_ids.length > 0
            ? `Cancelled task scope: ${result.cancelled_task_ids.join(", ")}`
            : `Task ${taskId} was already terminal.`,
        );
        await load();
      } catch (err) {
        setRuntimeTaskMessage(err instanceof Error ? err.message : "Failed to cancel runtime task");
      } finally {
        setCancellingTaskIds((prev) => ({ ...prev, [taskId]: false }));
      }
    },
    [load],
  );

  const analytics = data?.analytics;
  const traces = data?.traces ?? [];
  const stopReasonDistribution = analytics?.stop_reason_distribution || {};
  const stopReasonTotal = Object.values(stopReasonDistribution).reduce((sum, count) => sum + count, 0);
  const activeTaskStates = new Set(["queued", "running", "waiting_approval", "blocked"]);
  const activeRuntimeTasks = runtimeTasks.filter((task) => activeTaskStates.has(task.state));
  const taskStateTotals = runtimeTasks.reduce(
    (acc, task) => {
      acc[task.state] = (acc[task.state] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );
  const stopReasonRate = (reason: string): number => {
    if (stopReasonTotal <= 0) return 0;
    return (stopReasonDistribution[reason] || 0) / stopReasonTotal;
  };

  const stopReasonSLOs = [
    {
      key: "completed",
      label: "Completed",
      target: 0.9,
      direction: "min" as const,
      description: "Most queries should end cleanly.",
    },
    {
      key: "rate_limited",
      label: "Rate Limited",
      target: 0.03,
      direction: "max" as const,
      description: "Dispatch throttling should remain rare.",
    },
    {
      key: "max_iterations",
      label: "Max Iterations",
      target: 0.03,
      direction: "max" as const,
      description: "Runaway loops should be tightly bounded.",
    },
    {
      key: "policy_denied",
      label: "Policy Denied",
      target: 0.08,
      direction: "max" as const,
      description: "Denied actions should stay controlled.",
    },
  ];

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
            <h1 className="text-xl font-semibold text-slate-900 flex items-center gap-2">
              <Activity size={22} className="text-indigo-500" />
              Pipeline Observability
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Real-time Agentic RAG pipeline monitoring &amp; analytics
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={traceLimit}
            onChange={(e) => setTraceLimit(Number(e.target.value))}
            className="text-xs border border-slate-300 rounded-lg px-2 py-1.5 text-slate-700 bg-white"
          >
            <option value={10}>Last 10</option>
            <option value={20}>Last 20</option>
            <option value={50}>Last 50</option>
            <option value={100}>Last 100</option>
          </select>
          <button
            onClick={load}
            disabled={loading}
            className="rounded-lg p-2 text-slate-500 hover:text-indigo-600 hover:bg-indigo-50 transition-all disabled:opacity-50"
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
          {/* ── Runtime Approval Queue ───────────────────────────── */}
          <div className="rounded-2xl border border-amber-200 bg-amber-50/40 p-5">
            <div className="flex items-center gap-2 mb-3">
              <ShieldAlert size={16} className="text-amber-600" />
              <h3 className="text-sm font-semibold text-slate-800">
                Runtime Approval Queue
              </h3>
              <span className="ml-auto text-[11px] rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-amber-700">
                {pendingPermissions.length} pending
              </span>
            </div>

            <div className="text-xs text-slate-600 mb-3">
              Risky operations are blocked before execution and wait here for explicit approval.
              {expiredCount > 0 ? ` ${expiredCount} request(s) expired since last refresh.` : ""}
            </div>

            {executorStatus && (
              <div className="mb-3 grid grid-cols-2 md:grid-cols-5 gap-2 text-[11px]">
                <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                  <div className="text-slate-500">Worker</div>
                  <div className={`font-semibold ${executorStatus.running ? "text-emerald-600" : "text-slate-600"}`}>
                    {executorStatus.running ? "running" : "stopped"}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                  <div className="text-slate-500">Approved</div>
                  <div className="font-semibold text-slate-700">{executorStatus.summary.approved_total}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                  <div className="text-slate-500">Executing</div>
                  <div className="font-semibold text-blue-700">{executorStatus.summary.running}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                  <div className="text-slate-500">Waiting Retry</div>
                  <div className="font-semibold text-indigo-700">{executorStatus.summary.waiting_retry}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                  <div className="text-slate-500">Failed</div>
                  <div className="font-semibold text-red-700">{executorStatus.summary.failed}</div>
                </div>
              </div>
            )}

            {approvalMessage && (
              <div className="mb-3 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs text-indigo-700">
                {approvalMessage}
              </div>
            )}

            {pendingPermissions.length === 0 ? (
              <div className="rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs text-slate-500">
                No pending approval requests.
              </div>
            ) : (
              <div className="space-y-2">
                {pendingPermissions.map((request) => {
                  const resolving = !!resolvingIds[request.permission_id];
                  return (
                    <div
                      key={request.permission_id}
                      className="rounded-xl border border-slate-200 bg-white px-3 py-3"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-xs font-semibold text-slate-800 break-all">
                            {request.tool_name}
                          </div>
                          <div className="text-[11px] text-slate-500 mt-0.5 break-all">
                            {request.reason}
                          </div>
                          <div className="text-[11px] text-slate-500 mt-0.5 break-all">
                            {request.command_text}
                          </div>
                          <div className="text-[10px] text-slate-400 mt-1">
                            ID: {request.permission_id} · Expires: {new Date(request.expires_at).toLocaleString()}
                          </div>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <button
                            onClick={() => handleResolvePermission(request.permission_id, true)}
                            disabled={resolving}
                            className="inline-flex items-center gap-1 rounded-lg border border-emerald-300 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
                          >
                            {resolving ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                            Approve
                          </button>
                          <button
                            onClick={() => handleResolvePermission(request.permission_id, false)}
                            disabled={resolving}
                            className="inline-flex items-center gap-1 rounded-lg border border-red-300 bg-red-50 px-2.5 py-1 text-[11px] font-medium text-red-700 hover:bg-red-100 disabled:opacity-50"
                          >
                            {resolving ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                            Deny
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ── Runtime Task Lifecycle ─────────────────────────── */}
          <div
            data-testid="observability-runtime-task-panel"
            className="rounded-2xl border border-indigo-200 bg-indigo-50/30 p-5"
          >
            <div className="flex items-center gap-2 mb-3">
              <Activity size={16} className="text-indigo-600" />
              <h3 className="text-sm font-semibold text-slate-800">Active Background Jobs</h3>
              <div className="ml-auto flex items-center gap-2">
                <span className={`text-[10px] rounded-full border px-2 py-0.5 font-semibold ${
                  taskStreamConnected
                    ? "border-emerald-300 bg-emerald-100 text-emerald-700"
                    : "border-slate-300 bg-slate-100 text-slate-600"
                }`} data-testid="observability-task-stream-status">
                  {taskStreamConnected ? "LIVE" : "POLL"}
                </span>
                <span className="text-[11px] rounded-full border border-indigo-300 bg-indigo-100 px-2 py-0.5 text-indigo-700">
                  {activeRuntimeTasks.length} active
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] mb-3">
              <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                <div className="text-slate-500">Queued</div>
                <div className="font-semibold text-slate-700">{taskStateTotals.queued || 0}</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                <div className="text-slate-500">Running</div>
                <div className="font-semibold text-blue-700">{taskStateTotals.running || 0}</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                <div className="text-slate-500">Waiting Approval</div>
                <div className="font-semibold text-amber-700">{taskStateTotals.waiting_approval || 0}</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
                <div className="text-slate-500">Blocked</div>
                <div className="font-semibold text-violet-700">{taskStateTotals.blocked || 0}</div>
              </div>
            </div>

            {runtimeTaskMessage && (
              <div className="mb-3 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs text-indigo-700">
                {runtimeTaskMessage}
              </div>
            )}

            {activeRuntimeTasks.length === 0 ? (
              <div className="rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs text-slate-500">
                No active background jobs.
              </div>
            ) : (
              <div className="space-y-2 mb-3">
                {activeRuntimeTasks.slice(0, 8).map((task) => {
                  const cancelling = !!cancellingTaskIds[task.task_id];
                  return (
                    <div id={`runtime-task-${task.task_id}`} key={task.task_id} className="rounded-xl border border-slate-200 bg-white px-3 py-2.5">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-xs font-semibold text-slate-800 break-all">{task.task_id}</div>
                          <div className="text-[11px] text-slate-500 mt-0.5 break-all">
                            state={task.state} · parent={task.parent_task_id || "none"} · children={task.child_task_ids.length}
                          </div>
                        </div>
                        <button
                          onClick={() => handleCancelTask(task.task_id)}
                          disabled={cancelling}
                          className="inline-flex items-center gap-1 rounded-lg border border-red-300 bg-red-50 px-2.5 py-1 text-[11px] font-medium text-red-700 hover:bg-red-100 disabled:opacity-50"
                        >
                          {cancelling ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                          Cancel
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {taskNotifications.length > 0 && (
              <div className="rounded-xl border border-slate-200 bg-white px-3 py-2.5">
                <div className="text-[11px] font-semibold text-slate-700 mb-2">Recent Task State Notifications</div>
                <div className="space-y-1.5">
                  {taskNotifications.slice(0, 6).map((note) => (
                    <div
                      key={note.id}
                      data-testid="observability-task-notification"
                      className="text-[11px] text-slate-600 break-all"
                    >
                      <span className="font-medium text-slate-700">{note.taskId}</span>
                      {` ${note.fromState} -> ${note.toState} `}
                      <span className="text-slate-400">({new Date(note.timestamp).toLocaleTimeString()})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

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

          {/* ── Stop-Reason SLO Panel ───────────────────────────── */}
          {analytics && analytics.total_traces > 0 && (
            <div className="rounded-2xl border border-slate-300 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <Target size={16} className="text-indigo-500" />
                Stop-Reason SLOs
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {stopReasonSLOs.map((slo) => {
                  const rate = stopReasonRate(slo.key);
                  const pass = sloPass(rate, slo.target, slo.direction);
                  return (
                    <div
                      key={slo.key}
                      className={`rounded-xl border px-3.5 py-3 ${pass ? "border-emerald-200 bg-emerald-50/50" : "border-red-200 bg-red-50/50"}`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-xs font-semibold text-slate-800">{slo.label}</div>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${pass ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}
                        >
                          {pass ? "SLO pass" : "SLO miss"}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[11px]">
                        <span className="text-slate-500">Observed: {pct(rate)}</span>
                        <span className="text-slate-500">
                          Target: {slo.direction === "min" ? "≥" : "≤"} {pct(slo.target)}
                        </span>
                      </div>
                      <p className="mt-1 text-[11px] text-slate-500">{slo.description}</p>
                    </div>
                  );
                })}
              </div>

              <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-3 text-[11px] text-slate-600 flex flex-wrap gap-4">
                <span>Avg loop iterations: {analytics.runtime_loop.avg_iterations.toFixed(2)}</span>
                <span>Avg tool dispatches: {analytics.runtime_loop.avg_tool_calls.toFixed(2)}</span>
                <span>Total stop-reason samples: {stopReasonTotal}</span>
              </div>
            </div>
          )}

          {/* ── Live Pipeline Metrics ─────────────────────────────── */}
          {liveMetrics && (
            <div className="rounded-2xl border border-indigo-100 bg-gradient-to-r from-indigo-50/50 to-white p-5">
              <h3 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
                <Cpu size={16} className="text-indigo-500" />
                Live Pipeline Metrics
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-500 border border-emerald-200 ml-auto">
                  LIVE
                </span>
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="rounded-lg bg-white border border-slate-200 p-3">
                  <div className="text-slate-500 mb-1">Total Queries (Bus)</div>
                  <div className="text-lg font-semibold text-slate-800">{(liveMetrics.total_queries as number) ?? 0}</div>
                </div>
                <div className="rounded-lg bg-white border border-slate-200 p-3">
                  <div className="text-slate-500 mb-1">Avg Pipeline</div>
                  <div className="text-lg font-semibold text-slate-800">{fmtMs((liveMetrics.avg_pipeline_ms as number) ?? 0)}</div>
                </div>
                <div className="rounded-lg bg-white border border-slate-200 p-3">
                  <div className="text-slate-500 mb-1">Steps Executed</div>
                  <div className="text-lg font-semibold text-slate-800">{(liveMetrics.total_steps_executed as number) ?? 0}</div>
                </div>
                <div className="rounded-lg bg-white border border-slate-200 p-3">
                  <div className="text-slate-500 mb-1">Compressions</div>
                  <div className="text-lg font-semibold text-slate-800">{(liveMetrics.compression_invocations as number) ?? 0}</div>
                </div>
              </div>
              {/* Cache sub-section */}
              {liveMetrics.cache != null && typeof liveMetrics.cache === "object" ? (() => {
                const cache = liveMetrics.cache as Record<string, unknown>;
                return (
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                    <div className="rounded-lg bg-white border border-slate-200 p-2">
                      <div className="text-slate-500 text-[11px] mb-0.5">Cache Hits</div>
                      <div className="font-semibold text-emerald-600">{String((cache.total_hits as number) ?? 0)}</div>
                    </div>
                    <div className="rounded-lg bg-white border border-slate-200 p-2">
                      <div className="text-slate-500 text-[11px] mb-0.5">Cache Queries</div>
                      <div className="font-semibold text-slate-700">{String((cache.total_queries as number) ?? 0)}</div>
                    </div>
                    <div className="rounded-lg bg-white border border-slate-200 p-2">
                      <div className="text-slate-500 text-[11px] mb-0.5">Hit Rate</div>
                      <div className="font-semibold text-indigo-600">{pct((cache.hit_rate as number) ?? 0)}</div>
                    </div>
                  </div>
                );
              })() : null}
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
            <div className="rounded-2xl border border-slate-300 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
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
                      <span className="text-xs font-medium text-slate-700 w-24 capitalize">
                        {ch}
                      </span>
                      <div className="flex-1 h-5 bg-slate-100 rounded-full overflow-hidden relative">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${channelColors[ch] ?? "bg-slate-400"}`}
                          style={{ width: `${barPct}%` }}
                        />
                        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold text-slate-800">
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
            <div className="rounded-2xl border border-slate-300 bg-white p-5">
              <h3 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <GitBranch size={16} className="text-indigo-500" />
                Pipeline Step Breakdown
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="text-left py-2 px-3 text-slate-600 font-semibold">Step</th>
                      <th className="text-center py-2 px-3 text-slate-600 font-semibold">Completed</th>
                      <th className="text-center py-2 px-3 text-slate-600 font-semibold">Skipped</th>
                      <th className="text-center py-2 px-3 text-slate-600 font-semibold">Run Rate</th>
                      <th className="text-right py-2 px-3 text-slate-600 font-semibold">Total Time</th>
                      <th className="text-right py-2 px-3 text-slate-600 font-semibold">Avg Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analytics.step_stats).map(([step, stat]) => {
                      const total = stat.completed + stat.skipped;
                      const runRate = total > 0 ? stat.completed / total : 0;
                      const avgTime = stat.completed > 0 ? stat.total_duration_ms / stat.completed : 0;
                      return (
                        <tr key={step} className="border-b border-slate-100 hover:bg-slate-50/50">
                          <td className="py-2 px-3 font-medium text-slate-800 capitalize">
                            {step.replace(/_/g, " ")}
                          </td>
                          <td className="py-2 px-3 text-center">
                            <span className="inline-flex items-center gap-1 text-emerald-600">
                              {stat.completed}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-center text-slate-500">{stat.skipped}</td>
                          <td className="py-2 px-3 text-center">
                            <span className={`font-semibold ${runRate > 0.8 ? "text-emerald-600" : runRate > 0.3 ? "text-amber-600" : "text-slate-400"}`}>
                              {pct(runRate)}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-right text-slate-700">{fmtMs(stat.total_duration_ms)}</td>
                          <td className="py-2 px-3 text-right text-slate-700">{fmtMs(avgTime)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── Individual Trace Timeline ─────────────────────────── */}
          <div className="rounded-2xl border border-slate-300 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <Activity size={16} className="text-indigo-500" />
              Recent Pipeline Traces
              <span className="ml-auto text-[11px] text-slate-500 font-normal">
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
                    onCancelTask={handleCancelTask}
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
    <div className="rounded-xl border border-slate-300 bg-white p-3.5 flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-[11px] text-slate-500 font-medium uppercase tracking-wider">{label}</span>
      </div>
      <span className="text-xl font-bold text-slate-900">{value}</span>
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
    <div className={`rounded-2xl border border-slate-300 ${c.bg} p-4`}>
      <div className="flex items-center gap-2 mb-3">
        <span className={c.text}>{icon}</span>
        <div>
          <h4 className={`text-sm font-semibold ${c.text}`}>{title}</h4>
          <p className="text-[11px] text-slate-500">{subtitle}</p>
        </div>
      </div>
      {/* Activation rate bar */}
      <div className="mb-2">
        <div className="flex justify-between text-[11px] mb-1">
          <span className="text-slate-600">Activation Rate</span>
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
            <span className="text-sm font-bold text-slate-900">{stats.completed}</span>
            <p className="text-[10px] text-slate-500">Activated</p>
          </div>
          <div>
            <span className="text-sm font-bold text-slate-500">{stats.skipped}</span>
            <p className="text-[10px] text-slate-500">Skipped</p>
          </div>
          <div>
            <span className="text-sm font-bold text-slate-700">{fmtMs(stats.total_duration_ms)}</span>
            <p className="text-[10px] text-slate-500">Total Time</p>
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
  onCancelTask,
  onToggle,
}: {
  trace: PipelineTrace;
  expanded: boolean;
  onCancelTask: (taskId: string) => void;
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
  const coordinatorTaskId =
    trace.coordinator_task_id
    || trace.subagent_spawn_records?.[0]?.parent_task_id
    || "";
  const coordinatorTaskApiPath = coordinatorTaskId
    ? `/api/runtime/tasks/${encodeURIComponent(coordinatorTaskId)}`
    : "";

  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden hover:border-slate-300 transition-all">
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
          <p className="text-xs font-medium text-slate-800 truncate">
            &quot;{trace.query}&quot;
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">
            {ts} · {trace.trace_id}
          </p>
        </div>
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="text-center">
            <p className="text-xs font-bold text-slate-800">{fmtMs(trace.total_duration_ms ?? 0)}</p>
            <p className="text-[10px] text-slate-500">Latency</p>
          </div>
          <div className="text-center">
            <p className={`text-xs font-bold ${confidenceColor}`}>
              {pct(trace.final_confidence ?? 0)}
            </p>
            <p className="text-[10px] text-slate-500">Confidence</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-bold text-slate-800">{trace.evidence_count ?? 0}</p>
            <p className="text-[10px] text-slate-500">Evidence</p>
          </div>
          <div className="text-center">
            <p className="text-xs font-bold text-indigo-600">
              {completedSteps}/{totalSteps}
            </p>
            <p className="text-[10px] text-slate-500">Steps</p>
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
        <div className="border-t border-slate-200 bg-slate-50/30 px-4 py-4 space-y-4">
          {/* Step Timeline */}
          <div>
            <h4 className="text-[11px] font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
              <GitBranch size={12} /> Pipeline Steps
            </h4>
            <div className="space-y-1">
              {trace.steps?.map((step, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${STATUS_COLORS[step.status] ?? "bg-slate-300"}`} />
                  <span className={`text-[11px] w-48 ${STATUS_TEXT[step.status] ?? "text-slate-500"}`}>
                    {step.step_name}
                  </span>
                  <span className="text-[11px] text-slate-500 w-16 text-right">
                    {step.duration_ms > 0 ? fmtMs(step.duration_ms) : "—"}
                  </span>
                  {step.details && Object.keys(step.details).length > 0 && (
                    <span className="text-[10px] text-slate-500 truncate max-w-[300px]">
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
              <h4 className="text-[11px] font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
                <Search size={12} /> Retrieval Channels
              </h4>
              <div className="grid grid-cols-5 gap-2">
                {trace.retrieval_channels.map((ch) => (
                  <div
                    key={ch.channel}
                    className={`rounded-lg p-2 text-center ${
                      ch.result_count > 0 ? "bg-white border border-slate-300" : "bg-slate-50 border border-slate-200"
                    }`}
                  >
                    <p className="text-[11px] text-slate-600 capitalize font-medium">{ch.channel}</p>
                    <p className={`text-sm font-bold ${ch.result_count > 0 ? "text-slate-900" : "text-slate-300"}`}>
                      {ch.result_count}
                    </p>
                    {ch.result_count > 0 && (
                      <>
                        <p className="text-[10px] text-slate-500">
                          top: {ch.top_score?.toFixed(3)} · {fmtMs(ch.duration_ms ?? 0)}
                        </p>
                      </>
                    )}
                  </div>
                ))}
              </div>
              {trace.reranking && (
                <p className="text-[11px] text-slate-500 mt-1.5">
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
                    <ul className="mt-1 ml-4 list-disc text-slate-600">
                      {trace.query_transform.multi_queries.map((q, i) => (
                        <li key={i}>{q}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {trace.query_transform.hyde_answer && (
                  <p className="text-slate-600">
                    <span className="font-medium text-slate-800">HyDE:</span>{" "}
                    {trace.query_transform.hyde_answer.substring(0, 200)}...
                  </p>
                )}
                {trace.query_transform.step_back_query && (
                  <p className="text-slate-600">
                    <span className="font-medium text-slate-800">Step-Back:</span>{" "}
                    {trace.query_transform.step_back_query}
                  </p>
                )}
                <p className="text-slate-500">
                  {trace.query_transform.total_variants} variants · {fmtMs(trace.query_transform.duration_ms ?? 0)}
                </p>
              </div>
            </div>
          )}

          {/* Coordinator Plan + Sidechain */}
          {((trace.coordinator_plan && Object.keys(trace.coordinator_plan).length > 0)
            || Boolean(coordinatorTaskId)
            || (trace.subagent_spawn_records && trace.subagent_spawn_records.length > 0)
            || (trace.sidechain_transcript && trace.sidechain_transcript.length > 0)) && (
            <div>
              <h4 className="text-[11px] font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
                <GitBranch size={12} /> Coordinator &amp; Subagent Sidechain
              </h4>
              <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
                {coordinatorTaskId ? (
                  <div className="text-[11px] text-slate-600 break-all">
                    <span className="font-medium text-slate-700">Coordinator Task:</span>{" "}
                    <span className="font-mono text-slate-700">{coordinatorTaskId}</span>
                    <a
                      href={coordinatorTaskApiPath}
                      target="_blank"
                      rel="noreferrer"
                      className="ml-2 text-indigo-600 hover:text-indigo-700 underline"
                    >
                      view api
                    </a>
                    <button
                      onClick={() => onCancelTask(coordinatorTaskId)}
                      className="ml-2 rounded-md border border-red-300 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700 hover:bg-red-100"
                    >
                      cancel
                    </button>
                  </div>
                ) : null}

                {trace.coordinator_plan && Object.keys(trace.coordinator_plan).length > 0 && (
                  <div className="text-[11px] text-slate-600">
                    <span className="font-medium text-slate-700">Plan:</span>{" "}
                    strategy={String(trace.coordinator_plan.strategy || "unknown")} ·
                    primary={String(trace.coordinator_plan.primary_agent || "unknown")} ·
                    subagents={String(trace.coordinator_plan.subagent_count || 0)}
                  </div>
                )}

                {trace.subagent_spawn_records && trace.subagent_spawn_records.length > 0 && (
                  <div>
                    <div className="text-[11px] font-medium text-slate-700">Spawn Records</div>
                    <div className="mt-1 space-y-1">
                      {trace.subagent_spawn_records.slice(0, 6).map((record) => (
                        <div key={record.task_id} className="text-[11px] text-slate-600 break-all">
                          <a href={`#runtime-task-${record.task_id}`} className="font-mono text-indigo-600 hover:text-indigo-700 underline">
                            {record.task_id}
                          </a>
                          {` · role=${record.role} · agent=${record.agent}`}
                          <a
                            href={`/api/runtime/tasks/${encodeURIComponent(record.task_id)}`}
                            target="_blank"
                            rel="noreferrer"
                            className="ml-2 text-indigo-600 hover:text-indigo-700 underline"
                          >
                            api
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {trace.sidechain_transcript && trace.sidechain_transcript.length > 0 && (
                  <div>
                    <div className="text-[11px] font-medium text-slate-700">Sidechain Transcript</div>
                    <div className="mt-1 space-y-1">
                      {trace.sidechain_transcript.slice(0, 8).map((event, idx) => (
                        <div key={`${event.event}-${event.timestamp}-${idx}`} className="text-[11px] text-slate-600 break-all">
                          {event.event}
                          {event.agent ? ` · agent=${event.agent}` : ""}
                          {event.task_id ? ` · task=${event.task_id}` : ""}
                          {event.error ? ` · error=${event.error}` : ""}
                          {event.timestamp ? ` · ${new Date(event.timestamp).toLocaleTimeString()}` : ""}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Token Usage */}
          {trace.token_usage && Object.keys(trace.token_usage).length > 0 && (
            <div className="flex items-center gap-4 text-[11px] text-slate-500 bg-white border border-slate-300 rounded-lg px-3 py-2">
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
