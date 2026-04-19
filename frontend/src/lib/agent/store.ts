/**
 * Zustand store for autonomous agent state.
 * Architecture: Orchestrator.md §22.2, Agentic-RAG-Architecture.md §18.4
 *
 * Single source of truth for:
 * - Active agent sessions
 * - CortexEvent stream
 * - Tier classifications
 * - Tool executions in flight
 * - Steering queue state
 */

import { create } from "zustand";
import type {
  CortexEvent,
  TierClassification,
  ToolExecutionEvent,
} from "@/lib/types";

interface ActiveToolExecution {
  toolCallId: string;
  toolName: string;
  occurrence: number;
  startTime: number;
  args?: Record<string, unknown>;
}

interface AgentSessionState {
  sessionId: string | null;
  agentId: string;
  isStreaming: boolean;
  isRunning: boolean;
  turnCount: number;
  currentTier: TierClassification | null;
  events: CortexEvent[];
  activeTools: ActiveToolExecution[];
  completedTools: ToolExecutionEvent[];
  steeringQueue: string[];
  followUpQueue: string[];
  answer: string;
  error: string | null;
}

interface AgentStore {
  sessions: Record<string, AgentSessionState>;
  activeSessionId: string | null;
  globalEvents: CortexEvent[];
  isConnected: boolean;

  createSession: (sessionId: string, agentId: string) => void;
  setActiveSession: (sessionId: string | null) => void;
  removeSession: (sessionId: string) => void;

  handleEvent: (event: CortexEvent) => void;
  setAnswer: (sessionId: string, answer: string) => void;
  setError: (sessionId: string, error: string) => void;
  setStreaming: (sessionId: string, streaming: boolean) => void;
  setConnected: (connected: boolean) => void;

  addSteeringMessage: (sessionId: string, text: string) => void;
  addFollowUpMessage: (sessionId: string, text: string) => void;
  clearQueues: (sessionId: string) => void;

  getActiveSession: () => AgentSessionState | null;
  reset: () => void;
}

function stableSerialize(value: unknown): string {
  if (value === null || value === undefined) return String(value);
  if (typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableSerialize(item)).join(",")}]`;
  }

  const entries = Object.entries(value as Record<string, unknown>).sort(
    ([a], [b]) => a.localeCompare(b),
  );
  return `{${entries
    .map(([key, item]) => `${JSON.stringify(key)}:${stableSerialize(item)}`)
    .join(",")}}`;
}

function getEventId(event: CortexEvent): string | null {
  if (typeof event.event_id === "string" && event.event_id.trim()) {
    return event.event_id;
  }

  if (!event.data || typeof event.data !== "object") {
    return null;
  }

  const payload = event.data as Record<string, unknown>;
  const fromData = payload.event_id;
  if (typeof fromData === "string" && fromData.trim()) {
    return fromData;
  }

  const messageId = payload.message_id;
  if (typeof messageId === "string" && messageId.trim()) {
    return messageId;
  }

  return null;
}

function eventFingerprint(event: CortexEvent): string {
  const eventId = getEventId(event);
  if (eventId) {
    return `id:${eventId}`;
  }

  if (event.data && typeof event.data === "object") {
    const payload = event.data as Record<string, unknown>;
    const toolCallId = payload.toolCallId;
    if (
      (event.type === "tool_execution_start" || event.type === "tool_execution_end") &&
      typeof toolCallId === "string" &&
      toolCallId.trim()
    ) {
      const resultText =
        event.type === "tool_execution_end"
          ? String(payload.result ?? "")
          : "";
      const isErrorText =
        event.type === "tool_execution_end"
          ? String(Boolean(payload.isError))
          : "";
      return [
        "tool",
        event.type,
        event.session_id || "",
        event.trace_id || "",
        toolCallId,
        resultText,
        isErrorText,
      ].join("|");
    }
  }

  return [
    event.type,
    event.session_id || "",
    event.trace_id || "",
    event.timestamp || "",
    stableSerialize(event.data || {}),
  ].join("|");
}

function isDuplicateEvent(history: CortexEvent[], incoming: CortexEvent): boolean {
  const incomingKey = eventFingerprint(incoming);
  const recent = history.slice(-80);
  return recent.some((evt) => eventFingerprint(evt) === incomingKey);
}

function appendGlobalEvent(history: CortexEvent[], incoming: CortexEvent): CortexEvent[] {
  if (isDuplicateEvent(history, incoming)) {
    return history;
  }
  return [...history.slice(-200), incoming];
}

const defaultSessionState = (
  sessionId: string,
  agentId: string,
): AgentSessionState => ({
  sessionId,
  agentId,
  isStreaming: false,
  isRunning: false,
  turnCount: 0,
  currentTier: null,
  events: [],
  activeTools: [],
  completedTools: [],
  steeringQueue: [],
  followUpQueue: [],
  answer: "",
  error: null,
});

export const useAgentStore = create<AgentStore>((set, get) => ({
  sessions: {},
  activeSessionId: null,
  globalEvents: [],
  isConnected: false,

  createSession: (sessionId, agentId) =>
    set((state) => ({
      sessions: {
        ...state.sessions,
        [sessionId]: defaultSessionState(sessionId, agentId),
      },
      activeSessionId: sessionId,
    })),

  setActiveSession: (sessionId) => set({ activeSessionId: sessionId }),

  removeSession: (sessionId) =>
    set((state) => {
      const rest = { ...state.sessions };
      delete rest[sessionId];
      return {
        sessions: rest,
        activeSessionId:
          state.activeSessionId === sessionId ? null : state.activeSessionId,
      };
    }),

  handleEvent: (event) =>
    set((state) => {
      const sid = event.session_id || state.activeSessionId;
      const session = sid ? state.sessions[sid] : null;
      if (!session) {
        const nextGlobal = appendGlobalEvent(state.globalEvents, event);
        return nextGlobal === state.globalEvents
          ? state
          : { globalEvents: nextGlobal };
      }

      const nextGlobalEvents = appendGlobalEvent(state.globalEvents, event);
      if (isDuplicateEvent(session.events, event)) {
        return nextGlobalEvents === state.globalEvents
          ? state
          : { globalEvents: nextGlobalEvents };
      }

      const updated = { ...session, events: [...session.events.slice(-500), event] };

      switch (event.type) {
        case "agent_start":
          updated.isRunning = true;
          updated.isStreaming = true;
          updated.error = null;
          break;

        case "agent_end":
          updated.isRunning = false;
          updated.isStreaming = false;
          if (event.data?.answer) {
            updated.answer = event.data.answer as string;
          }
          break;

        case "turn_start":
          updated.turnCount = (event.data?.turn as number) || updated.turnCount + 1;
          break;

        case "tier_selected":
          updated.currentTier = event.data as unknown as TierClassification;
          break;

        case "tool_execution_start":
          {
            const tcId = (event.data?.toolCallId as string) || "";
            if (!tcId) break;
            if (updated.activeTools.some((t) => t.toolCallId === tcId)) break;

            const previousOccurrences =
              updated.activeTools.filter((t) => t.toolCallId === tcId).length +
              updated.completedTools.filter((t) => t.toolCallId === tcId).length;

            updated.activeTools = [
              ...updated.activeTools,
              {
                toolCallId: tcId,
                toolName: (event.data?.toolName as string) || "",
                occurrence: previousOccurrences + 1,
                startTime: Date.now(),
                args: event.data?.args as Record<string, unknown>,
              },
            ];
          }
          break;

        case "tool_execution_end": {
          const tcId = event.data?.toolCallId as string;
          if (!tcId) break;

          const activeCopy = [...updated.activeTools];
          const activeIndex = activeCopy.findIndex((t) => t.toolCallId === tcId);

          let toolName = "";
          let occurrence = 1;

          if (activeIndex >= 0) {
            const [activeTool] = activeCopy.splice(activeIndex, 1);
            toolName = activeTool.toolName || "";
            occurrence = activeTool.occurrence;
          } else {
            const previous = [...updated.completedTools]
              .reverse()
              .find((t) => t.toolCallId === tcId);
            if (previous) {
              toolName = previous.toolName || "";
              occurrence = (previous.occurrence || 1) + 1;
            }
          }

          const resultText = (event.data?.result as string) || "";
          const isError = Boolean(event.data?.isError as boolean);

          const isDuplicateCompletion = updated.completedTools.some(
            (tool) =>
              tool.toolCallId === tcId &&
              (tool.occurrence || 1) === occurrence &&
              (tool.result || "") === resultText &&
              Boolean(tool.isError) === isError,
          );

          updated.activeTools = activeCopy;
          if (!isDuplicateCompletion) {
            updated.completedTools = [
              ...updated.completedTools,
              {
                toolCallId: tcId,
                toolName,
                occurrence,
                result: resultText,
                isError,
              },
            ];
          }
          break;
        }

        case "queue_update":
          updated.steeringQueue = (event.data?.steering as string[]) || [];
          updated.followUpQueue = (event.data?.followUp as string[]) || [];
          break;

        case "auto_retry_start":
          updated.error = `Retrying (attempt ${event.data?.attempt}/${event.data?.maxAttempts})...`;
          break;

        case "auto_retry_end":
          if (event.data?.success) {
            updated.error = null;
          } else {
            updated.error = `Retry failed: ${event.data?.finalError || "Unknown error"}`;
          }
          break;
      }

      return {
        sessions: { ...state.sessions, [sid!]: updated },
        globalEvents: nextGlobalEvents,
      };
    }),

  setAnswer: (sessionId, answer) =>
    set((state) => {
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, answer },
        },
      };
    }),

  setError: (sessionId, error) =>
    set((state) => {
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, error },
        },
      };
    }),

  setStreaming: (sessionId, streaming) =>
    set((state) => {
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...session, isStreaming: streaming },
        },
      };
    }),

  setConnected: (connected) => set({ isConnected: connected }),

  addSteeringMessage: (sessionId, text) =>
    set((state) => {
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            steeringQueue: [...session.steeringQueue, text],
          },
        },
      };
    }),

  addFollowUpMessage: (sessionId, text) =>
    set((state) => {
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            followUpQueue: [...session.followUpQueue, text],
          },
        },
      };
    }),

  clearQueues: (sessionId) =>
    set((state) => {
      const session = state.sessions[sessionId];
      if (!session) return state;
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            steeringQueue: [],
            followUpQueue: [],
          },
        },
      };
    }),

  getActiveSession: () => {
    const state = get();
    return state.activeSessionId
      ? state.sessions[state.activeSessionId] ?? null
      : null;
  },

  reset: () =>
    set({
      sessions: {},
      activeSessionId: null,
      globalEvents: [],
      isConnected: false,
    }),
}));
