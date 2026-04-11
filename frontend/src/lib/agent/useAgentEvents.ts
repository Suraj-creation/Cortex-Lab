/**
 * SSE hook for CortexEvent streaming.
 * Architecture: Orchestrator.md §22.2 (Event-Driven UI Architecture)
 *
 * Connects to /api/agent/events (global) or /api/agent/query/stream (per-query).
 * Parses CortexEvent objects and dispatches to the Zustand store.
 */

"use client";

import { useEffect, useRef, useCallback } from "react";
import { useAgentStore } from "./store";
import type { CortexEvent } from "@/lib/types";

const BACKEND_BASE = (() => {
  if (typeof window !== "undefined") {
    const envBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
    if (envBase) return envBase.replace(/\/$/, "");
    const protocol = window.location.protocol;
    return `${protocol}//${window.location.hostname}:8000/api`;
  }
  return "http://localhost:8000/api";
})();

export function useGlobalAgentEvents() {
  const handleEvent = useAgentStore((s) => s.handleEvent);
  const setConnected = useAgentStore((s) => s.setConnected);
  const sourceRef = useRef<EventSource | null>(null);
  const retryCountRef = useRef(0);

  useEffect(() => {
    function connect() {
      const source = new EventSource(`${BACKEND_BASE}/agent/events`);
      sourceRef.current = source;

      source.onopen = () => {
        setConnected(true);
        retryCountRef.current = 0;
      };

      source.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data) as CortexEvent;
          if (parsed.type === "keepalive") return;
          handleEvent(parsed);
        } catch {
          // ignore parse errors
        }
      };

      source.onerror = () => {
        setConnected(false);
        source.close();
        const delay = Math.min(1000 * 2 ** retryCountRef.current, 30000);
        retryCountRef.current++;
        setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      sourceRef.current?.close();
      setConnected(false);
    };
  }, [handleEvent, setConnected]);
}

export function useAgentQueryStream() {
  const handleEvent = useAgentStore((s) => s.handleEvent);
  const setStreaming = useAgentStore((s) => s.setStreaming);
  const setAnswer = useAgentStore((s) => s.setAnswer);
  const setError = useAgentStore((s) => s.setError);

  const sendQuery = useCallback(
    async (
      query: string,
      sessionId?: string | null,
    ): Promise<string | null> => {
      const abortController = new AbortController();

      try {
        const res = await fetch(`${BACKEND_BASE}/agent/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            session_id: sessionId || null,
          }),
          signal: abortController.signal,
        });

        if (!res.ok) {
          throw new Error(`Agent query failed: ${res.status}`);
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";
        let capturedSessionId: string | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr) as CortexEvent;
              handleEvent(event);

              if (event.type === "agent_end" && event.data?.session_id) {
                capturedSessionId = event.data.session_id as string;
                if (event.data.answer) {
                  setAnswer(capturedSessionId, event.data.answer as string);
                }
              }

              if (event.session_id && !capturedSessionId) {
                capturedSessionId = event.session_id;
              }
            } catch {
              // ignore parse errors for individual events
            }
          }
        }

        return capturedSessionId;
      } catch (err) {
        const errMsg =
          err instanceof Error ? err.message : "Unknown stream error";
        if (sessionId) {
          setError(sessionId, errMsg);
          setStreaming(sessionId, false);
        }
        return null;
      }
    },
    [handleEvent, setStreaming, setAnswer, setError],
  );

  return { sendQuery };
}
