"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { getSchedulerStatus } from "@/lib/agent/api";
import { useAgentStore } from "@/lib/agent/store";
import {
  getRuntimeHealth,
  getRuntimeTasks,
  subscribeRuntimeTaskEvents,
} from "@/lib/api";
import type {
  CortexEvent,
  RuntimeTaskEvent,
  RuntimeTaskSnapshot,
  RuntimeTaskState,
} from "@/lib/types";

const ACTIVE_TASK_STATES = new Set<RuntimeTaskState>([
  "queued",
  "running",
  "waiting_approval",
  "blocked",
]);

const TERMINAL_TASK_STATES = new Set<RuntimeTaskState>([
  "completed",
  "failed",
  "cancelled",
]);

const MAX_TASK_EVENTS = 200;
const MAX_OPS_EVENTS = 160;

export type TaskStreamStatus = "connecting" | "live" | "reconnecting" | "polling";

export interface TaskStreamSnapshot {
  status: TaskStreamStatus;
  reconnectAttempts: number;
  error: string | null;
  lastEventAt: string | null;
}

export interface SchedulerTaskSnapshot {
  interval_seconds: number;
  last_run: number;
  is_running: boolean;
  run_count: number;
  error_count: number;
  enabled: boolean;
}

export interface SchedulerSnapshot {
  running: boolean;
  tasks: Record<string, SchedulerTaskSnapshot>;
}

export interface RuntimeHealthSnapshot {
  status?: string;
  selection?: {
    mode?: string;
    llm_provider?: string;
    stt_provider?: string;
    tts_provider?: string;
    allow_cloud_fallback?: boolean;
    updated_at?: string;
  };
  provider_availability?: Record<string, unknown>;
  active_llm_backend?: string;
  model_loaded?: boolean;
  runtime_tasks?: Record<string, unknown>;
  timestamp?: string;
}

export type OpsEventSource = "task" | "agent";

export interface OpsEvent {
  id: string;
  source: OpsEventSource;
  eventType: string;
  timestamp: string;
  sessionId: string;
  agentId: string;
  traceId: string;
  taskId: string;
  parentTaskId: string;
  state: string;
  note: string;
}

interface RuntimeTaskCounts {
  total: number;
  queued: number;
  running: number;
  waitingApproval: number;
  blocked: number;
  completed: number;
  failed: number;
  cancelled: number;
  active: number;
}

function parseIsoMs(value: string): number {
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? 0 : ms;
}

function sortTasks(tasks: RuntimeTaskSnapshot[]): RuntimeTaskSnapshot[] {
  return [...tasks].sort((a, b) => {
    const delta = parseIsoMs(b.updated_at) - parseIsoMs(a.updated_at);
    if (delta !== 0) return delta;
    return b.task_id.localeCompare(a.task_id);
  });
}

function readMetadataString(task: RuntimeTaskSnapshot, key: string): string {
  const value = task.metadata?.[key];
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function mapTaskEvent(event: RuntimeTaskEvent): OpsEvent {
  return {
    id: `task-${event.event_id}`,
    source: "task",
    eventType: event.event_type,
    timestamp: event.timestamp,
    sessionId: readMetadataString(event.task, "session_id"),
    agentId: readMetadataString(event.task, "agent_id") || readMetadataString(event.task, "agent"),
    traceId: readMetadataString(event.task, "trace_id"),
    taskId: event.task.task_id,
    parentTaskId: event.task.parent_task_id || "",
    state: event.state,
    note: event.note || "",
  };
}

function mapAgentEvent(event: CortexEvent, index: number): OpsEvent {
  const data = event.data || {};
  const taskIdRaw = data.task_id;
  const parentTaskRaw = data.parent_task_id;
  const stateRaw = data.state;
  const noteRaw = data.note;

  return {
    id: `agent-${event.timestamp}-${event.trace_id}-${index}`,
    source: "agent",
    eventType: event.type,
    timestamp: event.timestamp,
    sessionId: event.session_id || "",
    agentId: event.agent_id || "",
    traceId: event.trace_id || "",
    taskId: typeof taskIdRaw === "string" ? taskIdRaw : "",
    parentTaskId: typeof parentTaskRaw === "string" ? parentTaskRaw : "",
    state: typeof stateRaw === "string" ? stateRaw : "",
    note: typeof noteRaw === "string" ? noteRaw : "",
  };
}

export function useRuntimeOperationsCenter() {
  const globalEvents = useAgentStore((s) => s.globalEvents);
  const agentStreamConnected = useAgentStore((s) => s.isConnected);

  const [runtimeTasks, setRuntimeTasks] = useState<RuntimeTaskSnapshot[]>([]);
  const [runtimeTaskEvents, setRuntimeTaskEvents] = useState<RuntimeTaskEvent[]>([]);
  const [runtimeHealth, setRuntimeHealth] = useState<RuntimeHealthSnapshot | null>(null);
  const [scheduler, setScheduler] = useState<SchedulerSnapshot | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [taskStream, setTaskStream] = useState<TaskStreamSnapshot>({
    status: "connecting",
    reconnectAttempts: 0,
    error: null,
    lastEventAt: null,
  });

  const upsertTask = useCallback((task: RuntimeTaskSnapshot) => {
    setRuntimeTasks((prev) => {
      const idx = prev.findIndex((existing) => existing.task_id === task.task_id);
      if (idx < 0) {
        return sortTasks([task, ...prev]);
      }
      const next = [...prev];
      next[idx] = task;
      return sortTasks(next);
    });
  }, []);

  const refreshTaskSnapshot = useCallback(async () => {
    const payload = await getRuntimeTasks();
    setRuntimeTasks(sortTasks(payload.tasks || []));
  }, []);

  const refreshControlPlane = useCallback(async () => {
    const [health, schedulerSnapshot] = await Promise.all([
      getRuntimeHealth().catch(() => null),
      getSchedulerStatus().catch(() => null),
    ]);

    if (health) {
      setRuntimeHealth(health as RuntimeHealthSnapshot);
    }
    if (schedulerSnapshot) {
      setScheduler(schedulerSnapshot as SchedulerSnapshot);
    }
  }, []);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        await Promise.all([refreshTaskSnapshot(), refreshControlPlane()]);
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    })();

    return () => {
      mounted = false;
    };
  }, [refreshControlPlane, refreshTaskSnapshot]);

  useEffect(() => {
    const pollId = setInterval(() => {
      refreshControlPlane().catch(() => {
        // Keep stale values when control-plane endpoints are temporarily unavailable.
      });
    }, 12000);

    return () => clearInterval(pollId);
  }, [refreshControlPlane]);

  useEffect(() => {
    const pollMs = taskStream.status === "polling" ? 8000 : 15000;
    const pollId = setInterval(() => {
      refreshTaskSnapshot().catch(() => {
        // Task polling is best-effort fallback when SSE is unstable.
      });
    }, pollMs);

    return () => clearInterval(pollId);
  }, [refreshTaskSnapshot, taskStream.status]);

  useEffect(() => {
    let active = true;
    let reconnectAttempts = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let controller: AbortController | null = null;

    const connect = () => {
      if (!active) return;

      controller?.abort();
      setTaskStream((prev) => ({
        ...prev,
        status: reconnectAttempts > 0 ? "reconnecting" : "connecting",
      }));

      controller = subscribeRuntimeTaskEvents(
        (event) => {
          if (!active) return;

          reconnectAttempts = 0;
          setTaskStream({
            status: "live",
            reconnectAttempts: 0,
            error: null,
            lastEventAt: event.timestamp || new Date().toISOString(),
          });

          setRuntimeTaskEvents((prev) => [event, ...prev].slice(0, MAX_TASK_EVENTS));
          upsertTask(event.task);
        },
        (err) => {
          if (!active) return;

          reconnectAttempts += 1;
          const nextStatus = reconnectAttempts >= 3 ? "polling" : "reconnecting";
          setTaskStream((prev) => ({
            ...prev,
            status: nextStatus,
            reconnectAttempts,
            error: err.message,
          }));

          if (reconnectAttempts >= 3) {
            // Keep probing in polling mode so SSE can recover without a page refresh.
            reconnectTimer = setTimeout(() => {
              reconnectAttempts = 0;
              connect();
            }, 30000);
            return;
          }

          const delayMs = Math.min(1000 * 2 ** (reconnectAttempts - 1), 15000);
          reconnectTimer = setTimeout(connect, delayMs);
        },
      );
    };

    connect();

    return () => {
      active = false;
      controller?.abort();
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [upsertTask]);

  const taskCounts = useMemo<RuntimeTaskCounts>(() => {
    let queued = 0;
    let running = 0;
    let waitingApproval = 0;
    let blocked = 0;
    let completed = 0;
    let failed = 0;
    let cancelled = 0;

    for (const task of runtimeTasks) {
      switch (task.state) {
        case "queued":
          queued += 1;
          break;
        case "running":
          running += 1;
          break;
        case "waiting_approval":
          waitingApproval += 1;
          break;
        case "blocked":
          blocked += 1;
          break;
        case "completed":
          completed += 1;
          break;
        case "failed":
          failed += 1;
          break;
        case "cancelled":
          cancelled += 1;
          break;
      }
    }

    return {
      total: runtimeTasks.length,
      queued,
      running,
      waitingApproval,
      blocked,
      completed,
      failed,
      cancelled,
      active: queued + running + waitingApproval + blocked,
    };
  }, [runtimeTasks]);

  const activeTasks = useMemo(
    () => runtimeTasks.filter((task) => ACTIVE_TASK_STATES.has(task.state)),
    [runtimeTasks],
  );

  const longRunningTasks = useMemo(() => {
    const now = Date.now();
    return activeTasks
      .filter((task) => now - parseIsoMs(task.created_at) >= 2 * 60 * 1000)
      .sort((a, b) => parseIsoMs(a.created_at) - parseIsoMs(b.created_at));
  }, [activeTasks]);

  const activeAgents = useMemo(() => {
    const fromTasks = activeTasks
      .map((task) => readMetadataString(task, "agent_id") || readMetadataString(task, "agent"))
      .filter((value) => value.length > 0);

    const fromRecentEvents = globalEvents
      .slice(-80)
      .map((event) => (event.agent_id || "").trim())
      .filter((value) => value.length > 0);

    return Array.from(new Set([...fromTasks, ...fromRecentEvents])).sort();
  }, [activeTasks, globalEvents]);

  const opsEvents = useMemo(() => {
    const taskEvents = runtimeTaskEvents.map(mapTaskEvent);
    const agentEvents = globalEvents
      .slice(-MAX_OPS_EVENTS)
      .map((event, index) => mapAgentEvent(event, index));

    return [...taskEvents, ...agentEvents]
      .sort((a, b) => parseIsoMs(b.timestamp) - parseIsoMs(a.timestamp))
      .slice(0, MAX_OPS_EVENTS);
  }, [globalEvents, runtimeTaskEvents]);

  const schedulerTaskCount = useMemo(() => {
    if (!scheduler) return 0;
    return Object.values(scheduler.tasks).filter((task) => task.enabled).length;
  }, [scheduler]);

  const schedulerRunningCount = useMemo(() => {
    if (!scheduler) return 0;
    return Object.values(scheduler.tasks).filter((task) => task.is_running).length;
  }, [scheduler]);

  const failedTaskCount = useMemo(
    () => runtimeTasks.filter((task) => task.state === "failed").length,
    [runtimeTasks],
  );

  const terminalTasks = useMemo(
    () => runtimeTasks.filter((task) => TERMINAL_TASK_STATES.has(task.state)),
    [runtimeTasks],
  );

  return {
    isLoading,
    runtimeTasks,
    activeTasks,
    terminalTasks,
    longRunningTasks,
    runtimeTaskEvents,
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
  };
}
