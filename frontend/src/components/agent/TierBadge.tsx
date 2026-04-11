"use client";

import type { TierClassification } from "@/lib/types";

const TIER_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  T0: { bg: "bg-green-500/10", text: "text-green-400", border: "border-green-500/30" },
  T1: { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/30" },
  T2: { bg: "bg-amber-500/10", text: "text-amber-400", border: "border-amber-500/30" },
  T3: { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/30" },
  T4: { bg: "bg-red-500/10", text: "text-red-400", border: "border-red-500/30" },
};

const TIER_LABELS: Record<string, string> = {
  T0: "Cache Hit",
  T1: "Single Retrieval",
  T2: "Multi-Agent",
  T3: "Deep Research",
  T4: "Creative Synthesis",
};

export function TierBadge({ tier }: { tier: TierClassification | null }) {
  if (!tier) return null;

  const colors = TIER_COLORS[tier.tier] || TIER_COLORS.T1;
  const label = TIER_LABELS[tier.tier] || tier.tier;

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${colors.bg} ${colors.border}`}>
      <span className={`text-xs font-bold ${colors.text}`}>{tier.tier}</span>
      <span className="text-xs text-zinc-400">{label}</span>
      {tier.estimated_latency_ms > 0 && (
        <span className="text-[10px] text-zinc-500">
          ~{tier.estimated_latency_ms < 1000
            ? `${tier.estimated_latency_ms}ms`
            : `${(tier.estimated_latency_ms / 1000).toFixed(1)}s`}
        </span>
      )}
      {tier.recommended_agents.length > 0 && (
        <span className="text-[10px] text-zinc-500">
          {tier.recommended_agents.length} agent{tier.recommended_agents.length !== 1 ? "s" : ""}
        </span>
      )}
    </div>
  );
}

export function TierDetail({ tier }: { tier: TierClassification }) {
  const colors = TIER_COLORS[tier.tier] || TIER_COLORS.T1;

  return (
    <div className={`p-3 rounded-lg border ${colors.bg} ${colors.border} space-y-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-bold ${colors.text}`}>{tier.tier}</span>
          <span className="text-xs text-zinc-400">{TIER_LABELS[tier.tier]}</span>
        </div>
        <span className="text-xs text-zinc-500">
          Complexity: {(tier.complexity * 100).toFixed(0)}%
        </span>
      </div>

      <div className="flex flex-wrap gap-1">
        <span className="text-[10px] px-1.5 py-0.5 bg-zinc-800 rounded text-zinc-400">
          {tier.intent}
        </span>
        {tier.entities.map((e) => (
          <span key={e} className="text-[10px] px-1.5 py-0.5 bg-zinc-800 rounded text-zinc-400">
            {e}
          </span>
        ))}
      </div>

      {tier.recommended_agents.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tier.recommended_agents.map((a) => (
            <span
              key={a}
              className={`text-[10px] px-1.5 py-0.5 rounded ${colors.bg} ${colors.text} border ${colors.border}`}
            >
              {a}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
