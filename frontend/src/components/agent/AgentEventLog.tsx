"use client";

import { useAgentStore } from "@/lib/agent/store";
import type { CortexEvent } from "@/lib/types";

const EVENT_COLORS: Partial<Record<string, string>> = {
  agent_start: "text-green-400",
  agent_end: "text-green-300",
  turn_start: "text-blue-400",
  turn_end: "text-blue-300",
  tool_execution_start: "text-amber-400",
  tool_execution_end: "text-amber-300",
  tier_selected: "text-purple-400",
  quality_loop: "text-cyan-400",
  wiki_update: "text-emerald-400",
  belief_shift: "text-rose-400",
  auto_retry_start: "text-orange-400",
  auto_retry_end: "text-orange-300",
  compaction_start: "text-zinc-400",
  compaction_end: "text-zinc-300",
  evidence_ready: "text-teal-400",
  presence_initiative: "text-indigo-400",
};

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}

function EventRow({ event }: { event: CortexEvent }) {
  const color = EVENT_COLORS[event.type] || "text-zinc-400";

  return (
    <div className="flex items-start gap-2 px-3 py-1 hover:bg-zinc-800/50 text-[11px] font-mono">
      <span className="text-zinc-600 flex-shrink-0 w-16">{formatTime(event.timestamp)}</span>
      <span className={`flex-shrink-0 w-40 ${color}`}>{event.type}</span>
      <span className="text-zinc-500 truncate">
        {event.agent_id && <span className="text-zinc-600 mr-1">[{event.agent_id}]</span>}
        {summarizeData(event)}
      </span>
    </div>
  );
}

function summarizeData(event: CortexEvent): string {
  const d = event.data;
  if (!d || Object.keys(d).length === 0) return "";

  switch (event.type) {
    case "tier_selected":
      return `${d.tier} (${d.intent}, complexity=${d.complexity})`;
    case "tool_execution_start":
      return `${d.toolName}(${JSON.stringify(d.args || {}).slice(0, 60)})`;
    case "tool_execution_end":
      return `${d.isError ? "ERROR" : "OK"}: ${String(d.result || "").slice(0, 60)}`;
    case "turn_start":
    case "turn_end":
      return `turn ${d.turn}`;
    case "agent_end":
      return `${d.turns} turns`;
    case "auto_retry_start":
      return `attempt ${d.attempt}/${d.maxAttempts} (${d.delayMs}ms delay)`;
    default:
      return JSON.stringify(d).slice(0, 80);
  }
}

export function AgentEventLog({ sessionId }: { sessionId?: string }) {
  const session = useAgentStore((s) =>
    sessionId ? s.sessions[sessionId] || null : null,
  );
  const globalEvents = useAgentStore((s) => s.globalEvents);

  const events = session?.events || globalEvents;
  const recent = events.slice(-100);

  if (recent.length === 0) {
    return (
      <div className="text-center text-zinc-500 text-xs py-8">
        No events yet. Send a query to see agent activity.
      </div>
    );
  }

  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden">
      <div className="px-3 py-2 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-300">Agent Event Log</span>
        <span className="text-[10px] text-zinc-500">{recent.length} events</span>
      </div>
      <div className="max-h-64 overflow-y-auto divide-y divide-zinc-800/30">
        {recent.map((event, i) => (
          <EventRow key={`${event.timestamp}-${i}`} event={event} />
        ))}
      </div>
    </div>
  );
}
