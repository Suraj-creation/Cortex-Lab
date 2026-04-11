"use client";

import { useState, useCallback } from "react";
import { useAgentStore } from "@/lib/agent/store";
import { steerAgent, followUpAgent, abortAgent } from "@/lib/agent/api";

export function SteeringBar({ sessionId }: { sessionId: string }) {
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"steer" | "followup">("steer");
  const session = useAgentStore((s) => s.sessions[sessionId]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || !sessionId) return;
    try {
      if (mode === "steer") {
        await steerAgent(sessionId, input.trim());
      } else {
        await followUpAgent(sessionId, input.trim());
      }
      setInput("");
    } catch (err) {
      console.error("Steering error:", err);
    }
  }, [input, mode, sessionId]);

  const handleAbort = useCallback(async () => {
    if (!sessionId) return;
    try {
      await abortAgent(sessionId);
    } catch (err) {
      console.error("Abort error:", err);
    }
  }, [sessionId]);

  if (!session?.isRunning && !session?.steeringQueue.length && !session?.followUpQueue.length) {
    return null;
  }

  return (
    <div className="border-t border-zinc-800 bg-zinc-900/80 backdrop-blur-sm">
      {(session?.steeringQueue.length > 0 || session?.followUpQueue.length > 0) && (
        <div className="px-4 py-2 flex gap-2 flex-wrap">
          {session.steeringQueue.map((msg, i) => (
            <span
              key={`s-${i}`}
              className="text-[10px] px-2 py-0.5 bg-amber-500/10 border border-amber-500/30 text-amber-400 rounded"
            >
              Steering: {msg.slice(0, 50)}
            </span>
          ))}
          {session.followUpQueue.map((msg, i) => (
            <span
              key={`f-${i}`}
              className="text-[10px] px-2 py-0.5 bg-blue-500/10 border border-blue-500/30 text-blue-400 rounded"
            >
              Follow-up: {msg.slice(0, 50)}
            </span>
          ))}
        </div>
      )}

      {session?.isRunning && (
        <div className="px-4 py-2 flex items-center gap-2">
          <div className="flex bg-zinc-800 rounded-md overflow-hidden text-xs">
            <button
              onClick={() => setMode("steer")}
              className={`px-3 py-1 transition-colors ${
                mode === "steer"
                  ? "bg-amber-500/20 text-amber-400"
                  : "text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              Steer
            </button>
            <button
              onClick={() => setMode("followup")}
              className={`px-3 py-1 transition-colors ${
                mode === "followup"
                  ? "bg-blue-500/20 text-blue-400"
                  : "text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              Follow-up
            </button>
          </div>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={
              mode === "steer"
                ? "Redirect the agent mid-query..."
                : "Queue for after current query..."
            }
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-3 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-zinc-600"
          />

          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="px-3 py-1.5 bg-zinc-700 text-zinc-200 rounded-md text-xs hover:bg-zinc-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>

          <button
            onClick={handleAbort}
            className="px-3 py-1.5 bg-red-500/10 text-red-400 border border-red-500/30 rounded-md text-xs hover:bg-red-500/20 transition-colors"
          >
            Abort
          </button>
        </div>
      )}
    </div>
  );
}
