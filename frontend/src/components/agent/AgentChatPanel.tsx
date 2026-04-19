"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useAgentStore } from "@/lib/agent/store";
import { useAgentQueryStream } from "@/lib/agent/useAgentEvents";
import { createAgentSession } from "@/lib/agent/api";
import { TierBadge, TierDetail } from "./TierBadge";
import { SteeringBar } from "./SteeringBar";
import { ToolExecutionPanel } from "./ToolExecutionPanel";
import { AgentEventLog } from "./AgentEventLog";
import ReactMarkdown from "react-markdown";
import type { TierClassification } from "@/lib/types";

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  tier?: TierClassification | null;
  turnCount?: number;
  isStreaming?: boolean;
}

export function AgentChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showEventLog, setShowEventLog] = useState(false);
  const [showTierDetail, setShowTierDetail] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { sendQuery } = useAgentQueryStream();
  const createSession = useAgentStore((s) => s.createSession);
  const session = useAgentStore((s) =>
    sessionId ? s.sessions[sessionId] : null,
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!input.trim()) return;
    const query = input.trim();
    setInput("");

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);

    const assistantMsg: Message = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    let activeSession = sessionId;
    if (!activeSession) {
      try {
        const res = await createAgentSession("l1_orchestrator");
        activeSession = res.session_id;
        setSessionId(activeSession);
        createSession(activeSession, "l1_orchestrator");
      } catch {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: "Failed to create session. Is the backend running?", isStreaming: false }
              : m,
          ),
        );
        return;
      }
    }

    const resultSessionId = await sendQuery(query, activeSession);

    if (resultSessionId && resultSessionId !== activeSession) {
      setSessionId(resultSessionId);
      createSession(resultSessionId, "l1_orchestrator");
    }

    const finalSession = useAgentStore.getState().sessions[activeSession || resultSessionId || ""];
    const answer = finalSession?.answer || "[No response from agent]";
    const tier = finalSession?.currentTier || null;
    const turnCount = finalSession?.turnCount || 0;

    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantMsg.id
          ? { ...m, content: answer, isStreaming: false, tier, turnCount }
          : m,
      ),
    );
  }, [input, sessionId, sendQuery, createSession]);

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 bg-zinc-900/50">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-zinc-200">Cortex Agent</h2>
          {session?.currentTier && (
            <button onClick={() => setShowTierDetail(!showTierDetail)}>
              <TierBadge tier={session.currentTier} />
            </button>
          )}
          {session?.isRunning && (
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              <span className="text-[10px] text-amber-400">
                Turn {session.turnCount}
                {session.activeTools.length > 0 &&
                  ` / ${session.activeTools.length} tool${session.activeTools.length !== 1 ? "s" : ""}`}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowEventLog(!showEventLog)}
            className={`px-2 py-1 text-[10px] rounded border transition-colors ${
              showEventLog
                ? "bg-zinc-700 border-zinc-600 text-zinc-200"
                : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
            }`}
          >
            Events
          </button>
          {sessionId && (
            <span className="text-[10px] text-zinc-600 font-mono">
              {sessionId.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      {/* Tier detail dropdown */}
      {showTierDetail && session?.currentTier && (
        <div className="px-4 py-2 border-b border-zinc-800">
          <TierDetail tier={session.currentTier} />
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-zinc-600 text-sm mb-2">Cortex Autonomous Agent</div>
            <div className="text-zinc-700 text-xs max-w-md">
              Ask anything about your memories, experiences, and knowledge.
              The agent will classify your query (T0-T4), select specialist agents,
              retrieve evidence, and synthesize a grounded answer.
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-600/20 border border-blue-500/20 text-zinc-200"
                  : "bg-zinc-900 border border-zinc-800 text-zinc-300"
              }`}
            >
              {msg.isStreaming ? (
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                  <span className="text-xs text-zinc-500">Agent thinking...</span>
                </div>
              ) : (
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              )}

              {!msg.isStreaming && msg.role === "assistant" && msg.tier && (
                <div className="mt-2 pt-2 border-t border-zinc-800">
                  <TierBadge tier={msg.tier} />
                  {msg.turnCount != null && msg.turnCount > 0 && (
                    <span className="text-[10px] text-zinc-600 ml-2">
                      {msg.turnCount} turn{msg.turnCount !== 1 ? "s" : ""}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Tool execution panel */}
      {sessionId && (session?.activeTools.length || session?.completedTools.length) ? (
        <div className="px-4 py-2 border-t border-zinc-800">
          <ToolExecutionPanel sessionId={sessionId} />
        </div>
      ) : null}

      {/* Event log */}
      {showEventLog && (
        <div className="px-4 py-2 border-t border-zinc-800 max-h-64">
          <AgentEventLog sessionId={sessionId || undefined} />
        </div>
      )}

      {/* Steering bar */}
      {sessionId && <SteeringBar sessionId={sessionId} />}

      {/* Input */}
      <div className="px-4 py-3 border-t border-zinc-800 bg-zinc-900/50">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Ask Cortex anything..."
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-zinc-500 transition-colors"
            disabled={session?.isStreaming}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || session?.isStreaming}
            className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-xl text-sm font-medium transition-colors disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
