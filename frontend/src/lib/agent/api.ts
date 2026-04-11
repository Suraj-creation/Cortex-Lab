/**
 * Agent API client — typed wrappers for all autonomous agent endpoints.
 * Architecture: Orchestrator.md §19.3, Agentic-RAG-Architecture.md §19.3
 */

import type {
  AgentSessionInfo,
  AgentQueryResponse,
  AgentConfigInfo,
  TierClassification,
  WikiPageInfo,
} from "@/lib/types";

const BACKEND_BASE = (() => {
  if (typeof window !== "undefined") {
    const envBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
    if (envBase) return envBase.replace(/\/$/, "");
    const protocol = window.location.protocol;
    return `${protocol}//${window.location.hostname}:8000/api`;
  }
  return "http://localhost:8000/api";
})();

async function agentFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "");
    throw new Error(`Agent API error ${res.status}: ${errorBody}`);
  }
  return res.json();
}

// ── Sessions ────────────────────────────────────────────────────────────────

export async function createAgentSession(
  agentId = "l1_orchestrator",
  title?: string,
): Promise<{ session_id: string; agent_id: string; status: string }> {
  return agentFetch("/agent/sessions", {
    method: "POST",
    body: JSON.stringify({ agent_id: agentId, title }),
  });
}

export async function listAgentSessions(): Promise<{
  sessions: AgentSessionInfo[];
  count: number;
}> {
  return agentFetch("/agent/sessions");
}

export async function getAgentSession(
  sessionId: string,
): Promise<AgentSessionInfo> {
  return agentFetch(`/agent/sessions/${sessionId}`);
}

export async function closeAgentSession(
  sessionId: string,
): Promise<{ session_id: string; status: string }> {
  return agentFetch(`/agent/sessions/${sessionId}`, { method: "DELETE" });
}

// ── Query ───────────────────────────────────────────────────────────────────

export async function agentQuery(
  query: string,
  sessionId?: string | null,
  tierOverride?: string,
): Promise<AgentQueryResponse> {
  return agentFetch("/agent/query", {
    method: "POST",
    body: JSON.stringify({
      query,
      session_id: sessionId || null,
      tier_override: tierOverride,
    }),
  });
}

export async function classifyQueryTier(
  query: string,
): Promise<TierClassification> {
  return agentFetch("/agent/classify", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

// ── Steering & Follow-Up ────────────────────────────────────────────────────

export async function steerAgent(
  sessionId: string,
  text: string,
): Promise<{ status: string; queue: { steering: string[]; followUp: string[] } }> {
  return agentFetch(`/agent/sessions/${sessionId}/steer`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export async function followUpAgent(
  sessionId: string,
  text: string,
): Promise<{ status: string; queue: { steering: string[]; followUp: string[] } }> {
  return agentFetch(`/agent/sessions/${sessionId}/follow-up`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export async function abortAgent(
  sessionId: string,
): Promise<{ status: string; session_id: string }> {
  return agentFetch(`/agent/sessions/${sessionId}/abort`, { method: "POST" });
}

// ── Agent Configs ───────────────────────────────────────────────────────────

export async function listAgentConfigs(): Promise<{
  agents: AgentConfigInfo[];
  count: number;
}> {
  return agentFetch("/agent/configs");
}

export async function getAgentConfig(
  agentId: string,
): Promise<{
  agent_id: string;
  system_prompt: string;
  tools: Array<{ name: string; description: string }>;
  max_turns: number;
  context_window: number;
}> {
  return agentFetch(`/agent/configs/${agentId}`);
}

// ── Cache ───────────────────────────────────────────────────────────────────

export async function getCacheStats(): Promise<{
  size: number;
  hits: number;
  misses: number;
  hit_rate: number;
}> {
  return agentFetch("/agent/cache/stats");
}

// ── Scheduler ───────────────────────────────────────────────────────────────

export async function getSchedulerStatus(): Promise<{
  running: boolean;
  tasks: Record<
    string,
    {
      interval_seconds: number;
      last_run: number;
      is_running: boolean;
      run_count: number;
      error_count: number;
      enabled: boolean;
    }
  >;
}> {
  return agentFetch("/agent/scheduler/status");
}

export async function enableScheduledAgent(
  agentId: string,
): Promise<{ agent_id: string; enabled: boolean }> {
  return agentFetch(`/agent/scheduler/${agentId}/enable`, { method: "POST" });
}

export async function disableScheduledAgent(
  agentId: string,
): Promise<{ agent_id: string; enabled: boolean }> {
  return agentFetch(`/agent/scheduler/${agentId}/disable`, { method: "POST" });
}

// ── Wiki & Claims ───────────────────────────────────────────────────────────

export async function listWikiPages(): Promise<{
  pages: WikiPageInfo[];
  stats: { total_pages: number; total_topics: number; total_linked_claims: number };
}> {
  return agentFetch("/wiki/pages");
}

export async function getWikiPage(pageId: string): Promise<WikiPageInfo> {
  return agentFetch(`/wiki/pages/${pageId}`);
}

export async function searchWiki(
  query: string,
  includeClaims = true,
): Promise<{ results: WikiPageInfo[] }> {
  return agentFetch("/wiki/search", {
    method: "POST",
    body: JSON.stringify({ query, include_claims: includeClaims }),
  });
}

export async function getClaimStats(): Promise<{
  total: number;
  active: number;
  topics: number;
}> {
  return agentFetch("/wiki/claims");
}

export async function searchClaims(
  query: string,
  minConfidence = 0.5,
): Promise<{
  claims: Array<{
    id: string;
    text: string;
    confidence: number;
    source_ids: string[];
    topic: string;
  }>;
}> {
  return agentFetch("/wiki/claims/search", {
    method: "POST",
    body: JSON.stringify({ query, min_confidence: minConfidence }),
  });
}
