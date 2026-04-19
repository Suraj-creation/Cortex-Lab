"use client";

import { useCallback, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  Clock3,
  Filter,
  RefreshCw,
  Search,
  Server,
  ShieldAlert,
  Waves,
} from "lucide-react";
import { cancelRuntimeTask } from "@/lib/api";
import type { RuntimeTaskSnapshot, RuntimeTaskState } from "@/lib/types";
import {
  type OpsEvent,
  useRuntimeOperationsCenter,
} from "@/lib/observability/useRuntimeOperationsCenter";

type TaskTab = "all" | RuntimeTaskState;

const TERMINAL_STATES = new Set<RuntimeTaskState>([
  "completed",
  "failed",
  "cancelled",
]);

function parseIsoMs(value: string): number {
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? 0 : ms;
}

function elapsedFrom(isoTime: string): string {
  const delta = Math.max(0, Date.now() - parseIsoMs(isoTime));
  if (delta < 60_000) return `${Math.round(delta / 1000)}s`;
  if (delta < 3_600_000) return `${Math.round(delta / 60_000)}m`;
  return `${(delta / 3_600_000).toFixed(1)}h`;
}

function compactTime(isoTime: string): string {
  const parsed = parseIsoMs(isoTime);
  if (!parsed) return "n/a";
  return new Date(parsed).toLocaleTimeString();
}

function stateBadgeClass(state: RuntimeTaskState): string {
  switch (state) {
    case "queued":
      return "border-slate-200 bg-slate-100 text-slate-700";
    case "running":
      return "border-blue-200 bg-blue-50 text-blue-700";
    case "waiting_approval":
      return "border-amber-200 bg-amber-50 text-amber-700";
    case "blocked":
      return "border-violet-200 bg-violet-50 text-violet-700";
    case "completed":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "failed":
      return "border-red-200 bg-red-50 text-red-700";
    case "cancelled":
      return "border-rose-200 bg-rose-50 text-rose-700";
    default:
      return "border-slate-200 bg-slate-50 text-slate-700";
  }
}

function readMetadataString(task: RuntimeTaskSnapshot, key: string): string {
  const value = task.metadata?.[key];
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function summarizeEvent(event: OpsEvent): string {
  const core = event.note || event.state || "-";
  if (event.source === "task") return core;

  switch (event.eventType) {
    case "queue_update":
      return "steering queue updated";
    case "tool_execution_start":
      return "tool execution started";
    case "tool_execution_end":
      return "tool execution finished";
    case "agent_start":
      return "agent turn started";
    case "agent_end":
      return "agent turn completed";
    default:
      return core;
  }
}

function streamBadgeClass(status: string): string {
  switch (status) {
    case "live":
      return "border-emerald-300 bg-emerald-100 text-emerald-700";
    case "reconnecting":
      return "border-amber-300 bg-amber-100 text-amber-700";
    case "polling":
      return "border-violet-300 bg-violet-100 text-violet-700";
    default:
      return "border-slate-300 bg-slate-100 text-slate-700";
  }
}

function outcomeNote(task: RuntimeTaskSnapshot, event?: OpsEvent): string {
  if (event?.note) return event.note;
  return (
    readMetadataString(task, "failure_reason") ||
    readMetadataString(task, "error") ||
    readMetadataString(task, "note") ||
    "no additional details"
  );
}

export function RuntimeOperationsCenter() {
  const {
    isLoading,
    runtimeTasks,
    longRunningTasks,
    opsEvents,
    taskStream,
    taskCounts,
    runtimeHealth,
    scheduler,
    schedulerTaskCount,
    schedulerRunningCount,
    failedTaskCount,
    activeAgents,
    agentStreamConnected,
    refreshTaskSnapshot,
    refreshControlPlane,
  } = useRuntimeOperationsCenter();

  const [tab, setTab] = useState<TaskTab>("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [sessionFilter, setSessionFilter] = useState("");
  const [agentFilter, setAgentFilter] = useState("");
  const [traceFilter, setTraceFilter] = useState("");
  const [cancellingTaskIds, setCancellingTaskIds] = useState<Record<string, boolean>>({});
  const [actionMessage, setActionMessage] = useState<string>("");

  const tabs = useMemo(
    () => [
      { id: "all" as TaskTab, label: "All", count: taskCounts.total },
      { id: "queued" as TaskTab, label: "Queued", count: taskCounts.queued },
      { id: "running" as TaskTab, label: "Running", count: taskCounts.running },
      {
        id: "waiting_approval" as TaskTab,
        label: "Waiting Approval",
        count: taskCounts.waitingApproval,
      },
      { id: "blocked" as TaskTab, label: "Blocked", count: taskCounts.blocked },
      { id: "completed" as TaskTab, label: "Completed", count: taskCounts.completed },
      { id: "failed" as TaskTab, label: "Failed", count: taskCounts.failed },
      { id: "cancelled" as TaskTab, label: "Cancelled", count: taskCounts.cancelled },
    ],
    [taskCounts],
  );

  const filteredTasks = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    const sessionQuery = sessionFilter.trim().toLowerCase();
    const agentQuery = agentFilter.trim().toLowerCase();
    const traceQuery = traceFilter.trim().toLowerCase();

    return runtimeTasks.filter((task) => {
      if (tab !== "all" && task.state !== tab) return false;

      const sessionId = readMetadataString(task, "session_id");
      const agentId = readMetadataString(task, "agent_id") || readMetadataString(task, "agent");
      const traceId = readMetadataString(task, "trace_id");

      if (sessionQuery && !sessionId.toLowerCase().includes(sessionQuery)) return false;
      if (agentQuery && !agentId.toLowerCase().includes(agentQuery)) return false;
      if (traceQuery && !traceId.toLowerCase().includes(traceQuery)) return false;

      if (!query) return true;

      const haystack = [
        task.task_id,
        task.state,
        task.parent_task_id || "",
        sessionId,
        agentId,
        traceId,
      ]
        .join(" ")
        .toLowerCase();

      return haystack.includes(query);
    });
  }, [agentFilter, runtimeTasks, searchTerm, sessionFilter, tab, traceFilter]);

  const recentHistoryTasks = useMemo(
    () =>
      runtimeTasks
        .filter((task) => TERMINAL_STATES.has(task.state))
        .sort((a, b) => parseIsoMs(b.updated_at) - parseIsoMs(a.updated_at))
        .slice(0, 14),
    [runtimeTasks],
  );

  const taskOutcomeEvents = useMemo(() => {
    const byTaskId = new Map<string, OpsEvent>();
    for (const event of opsEvents) {
      if (event.source !== "task") continue;
      if (!event.taskId) continue;
      if (byTaskId.has(event.taskId)) continue;
      if (!event.note && !event.state) continue;
      byTaskId.set(event.taskId, event);
    }
    return byTaskId;
  }, [opsEvents]);

  const handleRefresh = useCallback(() => {
    refreshTaskSnapshot().catch(() => {
      // Manual refresh is best effort.
    });
    refreshControlPlane().catch(() => {
      // Keep stale control-plane snapshot when refresh fails.
    });
  }, [refreshControlPlane, refreshTaskSnapshot]);

  const handleCancelTask = useCallback(
    async (taskId: string) => {
      setCancellingTaskIds((prev) => ({ ...prev, [taskId]: true }));
      try {
        const result = await cancelRuntimeTask(
          taskId,
          "Cancelled from runtime operations center",
          true,
        );
        if (result.cancelled_task_ids.length > 0) {
          setActionMessage(`Cancelled: ${result.cancelled_task_ids.join(", ")}`);
        } else {
          setActionMessage(`Task ${taskId} was already terminal.`);
        }
        await refreshTaskSnapshot();
      } catch (err) {
        setActionMessage(err instanceof Error ? err.message : "Failed to cancel task");
      } finally {
        setCancellingTaskIds((prev) => ({ ...prev, [taskId]: false }));
      }
    },
    [refreshTaskSnapshot],
  );

  const runtimeMode = runtimeHealth?.selection?.mode || "unknown";
  const activeBackend = runtimeHealth?.active_llm_backend || "unknown";
  const schedulerRunning = scheduler?.running ?? false;

  return (
    <section className="rounded-2xl border border-indigo-200 bg-indigo-50/25 p-5">
      <div className="flex flex-wrap items-start gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
            <Server size={16} className="text-indigo-600" />
            Runtime Operations Center
          </h2>
          <p className="mt-1 text-xs text-slate-600">
            Unified live view of runtime tasks, agent events, scheduler activity, and queue health.
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${streamBadgeClass(taskStream.status)}`}>
            TASK STREAM {taskStream.status.toUpperCase()}
          </span>
          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
            agentStreamConnected
              ? "border-emerald-300 bg-emerald-100 text-emerald-700"
              : "border-slate-300 bg-slate-100 text-slate-600"
          }`}>
            AGENT STREAM {agentStreamConnected ? "LIVE" : "OFFLINE"}
          </span>
          <button
            onClick={handleRefresh}
            className="rounded-lg border border-slate-300 bg-white p-1.5 text-slate-600 hover:text-indigo-700 hover:border-indigo-300 transition-colors"
            title="Refresh operations center"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2 text-[11px]">
        <MetricMini label="Active Tasks" value={taskCounts.active} tone="blue" />
        <MetricMini label="Queued" value={taskCounts.queued} tone="slate" />
        <MetricMini label="Waiting Approval" value={taskCounts.waitingApproval} tone="amber" />
        <MetricMini label="Blocked" value={taskCounts.blocked} tone="violet" />
        <MetricMini label="Failed" value={failedTaskCount} tone="red" />
        <MetricMini label="Active Agents" value={activeAgents.length} tone="indigo" />
        <MetricMini label="Scheduled Tasks" value={schedulerTaskCount} tone="emerald" />
        <MetricMini label="Scheduler Running" value={schedulerRunning ? "yes" : "no"} tone={schedulerRunning ? "emerald" : "slate"} />
      </div>

      <div className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-[11px] text-slate-600 flex flex-wrap gap-x-4 gap-y-1">
        <span className="inline-flex items-center gap-1">
          <Waves size={11} className="text-indigo-500" />
          mode={runtimeMode}
        </span>
        <span className="inline-flex items-center gap-1">
          <Bot size={11} className="text-blue-500" />
          active_backend={activeBackend}
        </span>
        <span className="inline-flex items-center gap-1">
          <Clock3 size={11} className="text-slate-500" />
          scheduler_running={schedulerRunningCount}
        </span>
        {taskStream.error ? (
          <span className="inline-flex items-center gap-1 text-amber-700">
            <AlertTriangle size={11} />
            {taskStream.error}
          </span>
        ) : null}
        {taskStream.lastEventAt ? (
          <span className="ml-auto text-slate-500">last_task_event={compactTime(taskStream.lastEventAt)}</span>
        ) : null}
      </div>

      {longRunningTasks.length > 0 ? (
        <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-3">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-amber-800">
            <ShieldAlert size={12} />
            Long-running work still active
          </div>
          <div className="mt-1.5 space-y-1 text-[11px]">
            {longRunningTasks.slice(0, 5).map((task) => {
              const agentId = readMetadataString(task, "agent_id") || readMetadataString(task, "agent");
              return (
                <div key={task.task_id} className="text-amber-900 break-all">
                  <span className="font-mono">{task.task_id}</span>
                  {` · ${task.state} · running_for=${elapsedFrom(task.created_at)}`}
                  {agentId ? ` · agent=${agentId}` : ""}
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
            <Activity size={13} className="text-indigo-600" />
            Task Queue and History Board
          </h3>
          <span className="ml-auto text-[11px] text-slate-500">{filteredTasks.length} visible</span>
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {tabs.map((item) => (
            <button
              key={item.id}
              onClick={() => setTab(item.id)}
              className={`rounded-lg border px-2 py-1 text-[11px] font-medium transition-colors ${
                tab === item.id
                  ? "border-indigo-300 bg-indigo-50 text-indigo-700"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
              }`}
            >
              {item.label} ({item.count})
            </button>
          ))}
        </div>

        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
          <label className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600 flex items-center gap-1.5">
            <Search size={12} />
            <input
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="search task/state/trace"
              className="w-full bg-transparent outline-none text-slate-700 placeholder:text-slate-400"
            />
          </label>
          <label className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600 flex items-center gap-1.5">
            <Filter size={12} />
            <input
              value={sessionFilter}
              onChange={(e) => setSessionFilter(e.target.value)}
              placeholder="session_id"
              className="w-full bg-transparent outline-none text-slate-700 placeholder:text-slate-400"
            />
          </label>
          <label className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600 flex items-center gap-1.5">
            <Filter size={12} />
            <input
              value={agentFilter}
              onChange={(e) => setAgentFilter(e.target.value)}
              placeholder="agent_id"
              className="w-full bg-transparent outline-none text-slate-700 placeholder:text-slate-400"
            />
          </label>
          <label className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] text-slate-600 flex items-center gap-1.5">
            <Filter size={12} />
            <input
              value={traceFilter}
              onChange={(e) => setTraceFilter(e.target.value)}
              placeholder="trace_id"
              className="w-full bg-transparent outline-none text-slate-700 placeholder:text-slate-400"
            />
          </label>
        </div>

        {actionMessage ? (
          <div className="mt-3 rounded-lg border border-indigo-200 bg-indigo-50 px-2.5 py-2 text-[11px] text-indigo-700">
            {actionMessage}
          </div>
        ) : null}

        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[760px] text-[11px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="px-2 py-1.5 text-left font-semibold">Task</th>
                <th className="px-2 py-1.5 text-left font-semibold">State</th>
                <th className="px-2 py-1.5 text-left font-semibold">Agent</th>
                <th className="px-2 py-1.5 text-left font-semibold">Session</th>
                <th className="px-2 py-1.5 text-left font-semibold">Trace</th>
                <th className="px-2 py-1.5 text-left font-semibold">Age</th>
                <th className="px-2 py-1.5 text-left font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTasks.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-2 py-4 text-center text-slate-400">
                    {isLoading ? "Loading runtime tasks..." : "No tasks match current filters."}
                  </td>
                </tr>
              ) : (
                filteredTasks.slice(0, 120).map((task) => {
                  const agentId = readMetadataString(task, "agent_id") || readMetadataString(task, "agent");
                  const sessionId = readMetadataString(task, "session_id");
                  const traceId = readMetadataString(task, "trace_id");
                  const cancelling = !!cancellingTaskIds[task.task_id];
                  const isTerminal = TERMINAL_STATES.has(task.state);

                  return (
                    <tr key={task.task_id} className="border-b border-slate-100 align-top">
                      <td className="px-2 py-1.5">
                        <div className="font-mono text-slate-700 break-all">{task.task_id}</div>
                        {task.parent_task_id ? (
                          <div className="mt-0.5 text-[10px] text-slate-400 break-all">
                            parent={task.parent_task_id}
                          </div>
                        ) : null}
                      </td>
                      <td className="px-2 py-1.5">
                        <span className={`inline-flex rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${stateBadgeClass(task.state)}`}>
                          {task.state}
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-slate-700 break-all">{agentId || "-"}</td>
                      <td className="px-2 py-1.5 text-slate-600 break-all">{sessionId || "-"}</td>
                      <td className="px-2 py-1.5 text-slate-600 break-all">{traceId || "-"}</td>
                      <td className="px-2 py-1.5 text-slate-600">
                        <div>{elapsedFrom(task.created_at)}</div>
                        <div className="text-[10px] text-slate-400">upd {compactTime(task.updated_at)}</div>
                      </td>
                      <td className="px-2 py-1.5">
                        <div className="flex items-center gap-1.5">
                          <a
                            href={`/api/runtime/tasks/${encodeURIComponent(task.task_id)}`}
                            target="_blank"
                            rel="noreferrer"
                            className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600 hover:border-slate-300"
                          >
                            view
                          </a>
                          <button
                            onClick={() => handleCancelTask(task.task_id)}
                            disabled={isTerminal || cancelling}
                            className="rounded-md border border-red-300 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700 hover:bg-red-100 disabled:opacity-50"
                          >
                            {cancelling ? "..." : "cancel"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3">
        <h3 className="text-xs font-semibold text-slate-800">Merged Runtime Event Timeline</h3>
        <p className="mt-1 text-[11px] text-slate-500">
          Combined feed of runtime task transitions and global agent events.
        </p>
        <div className="mt-2 max-h-56 overflow-y-auto divide-y divide-slate-100 border border-slate-100 rounded-lg">
          {opsEvents.length === 0 ? (
            <div className="px-3 py-4 text-[11px] text-slate-400">No runtime events observed yet.</div>
          ) : (
            opsEvents.slice(0, 60).map((event) => (
              <div key={event.id} className="px-3 py-2 text-[11px]">
                <div className="flex items-center gap-2">
                  <span className={`rounded-full border px-1.5 py-0.5 text-[10px] font-semibold ${
                    event.source === "task"
                      ? "border-indigo-200 bg-indigo-50 text-indigo-700"
                      : "border-emerald-200 bg-emerald-50 text-emerald-700"
                  }`}>
                    {event.source}
                  </span>
                  <span className="font-semibold text-slate-700">{event.eventType}</span>
                  <span className="ml-auto text-slate-400">{compactTime(event.timestamp)}</span>
                </div>
                <div className="mt-0.5 text-slate-600 break-all">
                  {event.taskId ? `task=${event.taskId} · ` : ""}
                  {event.agentId ? `agent=${event.agentId} · ` : ""}
                  {event.traceId ? `trace=${event.traceId} · ` : ""}
                  {summarizeEvent(event)}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3">
        <h3 className="text-xs font-semibold text-slate-800">Recent Task Outcomes</h3>
        <p className="mt-1 text-[11px] text-slate-500">
          Terminal queue history for completed, failed, and cancelled work.
        </p>
        <div className="mt-2 max-h-56 overflow-y-auto divide-y divide-slate-100 border border-slate-100 rounded-lg">
          {recentHistoryTasks.length === 0 ? (
            <div className="px-3 py-4 text-[11px] text-slate-400">No terminal tasks yet.</div>
          ) : (
            recentHistoryTasks.map((task) => {
              const event = taskOutcomeEvents.get(task.task_id);
              const agentId = readMetadataString(task, "agent_id") || readMetadataString(task, "agent");
              const traceId = readMetadataString(task, "trace_id");

              return (
                <div key={task.task_id} className="px-3 py-2 text-[11px]">
                  <div className="flex items-center gap-2">
                    <span className={`inline-flex rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${stateBadgeClass(task.state)}`}>
                      {task.state}
                    </span>
                    <span className="font-mono text-slate-700 break-all">{task.task_id}</span>
                    <span className="ml-auto text-slate-400">{compactTime(task.updated_at)}</span>
                  </div>
                  <div className="mt-0.5 text-slate-600 break-all">
                    {agentId ? `agent=${agentId} · ` : ""}
                    {traceId ? `trace=${traceId} · ` : ""}
                    {outcomeNote(task, event)}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
}

function MetricMini({
  label,
  value,
  tone,
}: {
  label: string;
  value: string | number;
  tone: "slate" | "blue" | "amber" | "violet" | "red" | "indigo" | "emerald";
}) {
  const toneClass: Record<string, string> = {
    slate: "text-slate-700",
    blue: "text-blue-700",
    amber: "text-amber-700",
    violet: "text-violet-700",
    red: "text-red-700",
    indigo: "text-indigo-700",
    emerald: "text-emerald-700",
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white px-2.5 py-2">
      <div className="text-slate-500">{label}</div>
      <div className={`text-sm font-semibold ${toneClass[tone]}`}>{value}</div>
    </div>
  );
}
