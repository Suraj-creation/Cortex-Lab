"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Radio, User, Users } from "lucide-react";
import { getLiveTranscript } from "@/lib/api";
import { AmbientState, ConversationTurn } from "@/lib/types";

interface Props {
  status: AmbientState | null;
}

const SPEAKER_COLORS: Record<string, string> = {
  USER: "text-indigo-600 bg-indigo-50 border-indigo-200",
  SPEAKER_A: "text-emerald-600 bg-emerald-50 border-emerald-200",
  SPEAKER_B: "text-amber-600 bg-amber-50 border-amber-200",
  SPEAKER_C: "text-violet-600 bg-violet-50 border-violet-200",
  SPEAKER_D: "text-rose-600 bg-rose-50 border-rose-200",
  UNKNOWN: "text-slate-500 bg-slate-50 border-slate-200",
};

function getSpeakerColor(label: string): string {
  return SPEAKER_COLORS[label] || SPEAKER_COLORS.UNKNOWN;
}

export function LiveTranscript({ status }: Props) {
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [partialTurns, setPartialTurns] = useState<Record<string, ConversationTurn>>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const seenTurnIdsRef = useRef<Set<string>>(new Set());

  const isActive = useCallback(() => {
    return (
      status?.status === "listening" ||
      status?.status === "speech_detected" ||
      status?.status === "transcribing" ||
      status?.live?.running
    );
  }, [status?.status, status?.live?.running]);

  // Poll live transcript as fallback
  useEffect(() => {
    if (!isActive()) return;

    const poll = async () => {
      try {
        const data = await getLiveTranscript();
        setTurns(data.turns);
      } catch {
        // ignore
      }
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [isActive]);

  // WebSocket for real-time updates with reconnection
  useEffect(() => {
    mountedRef.current = true;

    const connectWs = () => {
      if (!isActive() || !mountedRef.current) return;

      // Don't reconnect if already open
      if (wsRef.current?.readyState === WebSocket.OPEN) return;

      const wsUrl = `ws://${window.location.hostname}:8000/ws/ambient`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        // Clear any pending reconnect
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current);
          reconnectTimer.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "live_partial") {
            const turnId = String(data.live_turn_id || "");
            if (!turnId) return;
            setPartialTurns((prev) => ({
              ...prev,
              [turnId]: {
                speaker_label: data.speaker_label || "USER",
                speaker_name: data.speaker_name || data.speaker_label || "Speaker",
                text: data.text || "",
                timestamp: Number(data.timestamp || 0),
                confidence: Number(data.confidence || 0),
                speaker_confidence: Number(data.speaker_confidence || 0),
                live_turn_id: turnId,
              },
            }));
            return;
          }

          if (data.type === "live_final_turn" || data.type === "transcript") {
            const turnId = String(data.live_turn_id || "");
            if (turnId) {
              if (seenTurnIdsRef.current.has(turnId)) {
                if (data.type === "live_final_turn") {
                  setPartialTurns((prev) => {
                    const next = { ...prev };
                    delete next[turnId];
                    return next;
                  });
                }
                return;
              }
              seenTurnIdsRef.current.add(turnId);
            }

            if (data.type === "live_final_turn" && turnId) {
              setPartialTurns((prev) => {
                const next = { ...prev };
                delete next[turnId];
                return next;
              });
            }

            setTurns((prev) => [
              ...prev,
              {
                speaker_label: data.speaker_label,
                speaker_name: data.speaker_name,
                text: data.text,
                timestamp: Number(data.timestamp || 0),
                confidence: Number(data.confidence || 0),
                speaker_confidence: Number(data.speaker_confidence || 0),
                live_turn_id: turnId || undefined,
                retention_trace: data.retention_trace || undefined,
              },
            ]);
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {};

      ws.onclose = () => {
        wsRef.current = null;
        // Auto-reconnect after 3s if still active
        if (mountedRef.current && isActive()) {
          reconnectTimer.current = setTimeout(connectWs, 3000);
        }
      };
    };

    if (isActive()) {
      connectWs();
    }

    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isActive]);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [turns]);

  const active = isActive();

  if (!active && turns.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-12">
        <div className="text-center">
          <div className="mx-auto h-16 w-16 rounded-2xl bg-slate-100 flex items-center justify-center mb-4">
            <Radio size={28} className="text-slate-400" />
          </div>
          <h3 className="text-sm font-medium text-slate-600 mb-1">
            Not Listening
          </h3>
          <p className="text-xs text-slate-400 max-w-xs">
            Start ambient listening to see live transcripts appear here in
            real-time.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Live indicator */}
      {active && (
        <div className="px-6 py-2 bg-emerald-50 border-b border-emerald-100 flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] font-medium text-emerald-600 uppercase tracking-wider">
            Live Transcript
          </span>
          {wsRef.current?.readyState === WebSocket.OPEN && (
            <span className="text-[9px] text-emerald-400 ml-1">● WS</span>
          )}
          {status?.conversation?.current_turns !== undefined && (
            <span className="text-[10px] text-emerald-500 ml-auto">
              {status.conversation.current_turns} turns
            </span>
          )}
        </div>
      )}

      {/* Transcript */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-3">
        {turns.map((turn, i) => (
          <div key={i} className="flex gap-3">
            <div
              className={`h-7 w-7 rounded-lg flex items-center justify-center flex-shrink-0 border ${getSpeakerColor(
                turn.speaker_label
              )}`}
            >
              {turn.speaker_label === "USER" ? (
                <User size={12} />
              ) : (
                <Users size={12} />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-xs font-medium text-slate-700">
                  {turn.speaker_name || turn.speaker_label}
                </span>
                <span className="text-[10px] text-slate-400">
                  {turn.timestamp > 0 ? `${turn.timestamp.toFixed(1)}s` : ""}
                </span>
                {turn.confidence > 0 && (
                  <span className="text-[10px] text-slate-300">
                    ({(turn.confidence * 100).toFixed(0)}%)
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-600 leading-relaxed">
                {turn.text}
              </p>
            </div>
          </div>
        ))}
        {Object.values(partialTurns).map((turn) => (
          <div key={`partial-${turn.live_turn_id || turn.timestamp}`} className="flex gap-3 opacity-75">
            <div
              className={`h-7 w-7 rounded-lg flex items-center justify-center flex-shrink-0 border ${getSpeakerColor(
                turn.speaker_label
              )}`}
            >
              {turn.speaker_label === "USER" ? (
                <User size={12} />
              ) : (
                <Users size={12} />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-xs font-medium text-slate-500">
                  {turn.speaker_name || turn.speaker_label}
                </span>
                <span className="text-[10px] text-violet-400 uppercase tracking-wider">
                  partial
                </span>
              </div>
              <p className="text-sm text-slate-500 leading-relaxed italic">
                {turn.text}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
