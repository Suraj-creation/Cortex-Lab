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
  CortexEventType,
  TierClassification,
  AgentSessionInfo,
  ToolExecutionEvent,
  AgentTurnInfo,
} from "@/lib/types";

interface ActiveToolExecution {
  toolCallId: string;
  toolName: string;
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
      const { [sessionId]: _, ...rest } = state.sessions;
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
        return { globalEvents: [...state.globalEvents.slice(-200), event] };
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
          updated.activeTools = [
            ...updated.activeTools,
            {
              toolCallId: (event.data?.toolCallId as string) || "",
              toolName: (event.data?.toolName as string) || "",
              startTime: Date.now(),
              args: event.data?.args as Record<string, unknown>,
            },
          ];
          break;

        case "tool_execution_end": {
          const tcId = event.data?.toolCallId as string;
          updated.activeTools = updated.activeTools.filter(
            (t) => t.toolCallId !== tcId,
          );
          updated.completedTools = [
            ...updated.completedTools,
            {
              toolCallId: tcId,
              toolName: updated.activeTools.find((t) => t.toolCallId === tcId)
                ?.toolName || "",
              result: event.data?.result as string,
              isError: event.data?.isError as boolean,
            },
          ];
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
        globalEvents: [...state.globalEvents.slice(-200), event],
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
