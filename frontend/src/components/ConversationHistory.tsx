"use client";

import { useState } from "react";
import {
  MessageSquare,
  Clock,
  Users,
  ChevronDown,
  ChevronRight,
  Brain,
  RefreshCw,
} from "lucide-react";
import { ConversationRecord } from "@/lib/types";

interface Props {
  conversations: ConversationRecord[];
  onRefresh: () => void;
}

export function ConversationHistory({ conversations, onRefresh }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (conversations.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-12">
        <div className="text-center">
          <div className="mx-auto h-16 w-16 rounded-2xl bg-slate-100 flex items-center justify-center mb-4">
            <MessageSquare size={28} className="text-slate-400" />
          </div>
          <h3 className="text-sm font-medium text-slate-600 mb-1">
            No Conversations Yet
          </h3>
          <p className="text-xs text-slate-400 max-w-xs">
            Conversations captured by ambient listening will appear here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-700">
          {conversations.length} Conversation{conversations.length !== 1 ? "s" : ""}
        </h3>
        <button
          onClick={onRefresh}
          className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {conversations.map((conv) => {
        const expanded = expandedId === conv.id;
        return (
          <div
            key={conv.id}
            className="rounded-xl border border-slate-200 bg-white overflow-hidden"
          >
            {/* Header */}
            <button
              onClick={() => setExpandedId(expanded ? null : conv.id)}
              className="w-full px-4 py-3 flex items-center gap-3 hover:bg-slate-50 transition-all"
            >
              <div className="text-slate-400">
                {expanded ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </div>
              <div className="flex-1 text-left">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-slate-700">
                    {conv.participants.join(", ")}
                  </span>
                  {conv.auto_ingested && (
                    <span className="flex items-center gap-0.5 rounded-full bg-indigo-50 border border-indigo-200 px-1.5 py-0.5 text-[9px] text-indigo-600">
                      <Brain size={9} />
                      Ingested
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 mt-0.5">
                  <span className="flex items-center gap-1 text-[10px] text-slate-400">
                    <Clock size={10} />
                    {conv.start_time
                      ? new Date(conv.start_time).toLocaleString()
                      : "Unknown"}
                  </span>
                  <span className="flex items-center gap-1 text-[10px] text-slate-400">
                    <MessageSquare size={10} />
                    {conv.turns.length} turns
                  </span>
                  <span className="flex items-center gap-1 text-[10px] text-slate-400">
                    <Users size={10} />
                    {conv.participants.length} participants
                  </span>
                  <span className="text-[10px] text-slate-400">
                    {formatDuration(conv.duration_seconds)}
                  </span>
                </div>
              </div>
            </button>

            {/* Expanded Turns */}
            {expanded && (
              <div className="border-t border-slate-100 px-4 py-3 space-y-2.5 bg-slate-50/50">
                {conv.turns.map((turn, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-xs font-medium text-slate-500 w-24 flex-shrink-0 text-right">
                      {turn.speaker_name || turn.speaker_label}:
                    </span>
                    <p className="text-xs text-slate-600 leading-relaxed flex-1">
                      {turn.text}
                    </p>
                  </div>
                ))}
                {conv.memory_ids.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-slate-200">
                    <span className="text-[10px] text-slate-400">
                      Linked memories: {conv.memory_ids.join(", ")}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}
