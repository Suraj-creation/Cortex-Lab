"use client";

import { useAgentStore } from "@/lib/agent/store";

export function ToolExecutionPanel({ sessionId }: { sessionId: string }) {
  const session = useAgentStore((s) => s.sessions[sessionId]);

  if (!session) return null;

  const { activeTools, completedTools } = session;
  const hasActivity = activeTools.length > 0 || completedTools.length > 0;
  if (!hasActivity) return null;

  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden">
      <div className="px-3 py-2 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-300">Tool Executions</span>
        <span className="text-[10px] text-zinc-500">
          {activeTools.length} active / {completedTools.length} complete
        </span>
      </div>

      <div className="max-h-48 overflow-y-auto">
        {activeTools.map((tool) => (
          <div
            key={`active-${tool.toolCallId}-${tool.occurrence}`}
            className="px-3 py-2 border-b border-zinc-800/50 flex items-center gap-2"
          >
            <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-xs font-mono text-amber-400">{tool.toolName || "tool"}</span>
            <span className="text-[10px] text-zinc-500 ml-auto">
              {((Date.now() - tool.startTime) / 1000).toFixed(1)}s
            </span>
          </div>
        ))}

        {completedTools.slice(-10).reverse().map((tool, index) => (
          <div
            key={`done-${tool.toolCallId}-${tool.occurrence || 1}-${index}`}
            className="px-3 py-2 border-b border-zinc-800/50 flex items-center gap-2"
          >
            <div
              className={`w-2 h-2 rounded-full ${
                tool.isError ? "bg-red-400" : "bg-green-400"
              }`}
            />
            <span
              className={`text-xs font-mono ${
                tool.isError ? "text-red-400" : "text-green-400"
              }`}
            >
              {tool.toolName || "tool"}
            </span>
            {tool.result && (
              <span className="text-[10px] text-zinc-500 truncate max-w-[200px] ml-auto">
                {tool.result.slice(0, 80)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
