import { ChatSettings, DEFAULT_SETTINGS, MemoryObject, GraphData, RAGStats, EvidenceCard, AmbientState, AmbientConfig, ConversationRecord, VoiceQueryResult, ConversationTurn, PipelineTrace, TracesResponse } from "./types";

const API_BASE = "/api";

// ── Non-streaming chat ──────────────────────────────────────────

export async function sendMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
): Promise<{
  content: string;
  thinking?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: settings.maxTokens,
      stream: false,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

// ── Streaming chat ──────────────────────────────────────────────

export async function streamMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        stream: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;

        const json = trimmed.slice(6);
        try {
          const data = JSON.parse(json);
          if (data.done) {
            onDone();
            return;
          }
          if (data.delta) {
            onToken(data.delta);
          }
        } catch {
          // skip malformed JSON
        }
      }
    }

    onDone();
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

// ── RAG-Enhanced Chat ───────────────────────────────────────────

export async function ragChat(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  sessionId: string = "",
): Promise<{
  content: string;
  thinking?: string;
  evidence?: EvidenceCard[];
  agents_used?: string[];
  confidence?: number;
  query_analysis?: { intent: string; complexity: number; routing: string };
  processing_time_ms?: number;
  cache_hit?: boolean;
  pipeline_trace?: PipelineTrace;
}> {
  const res = await fetch(`${API_BASE}/rag/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: settings.maxTokens,
      stream: false,
      use_rag: true,
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

// ── RAG Streaming Chat ──────────────────────────────────────────

export interface RAGStreamMeta {
  evidence?: EvidenceCard[];
  agents_used?: string[];
  confidence?: number;
  query_analysis?: { intent: string; complexity: number; routing: string };
  thinking?: string;
  pipeline_trace?: PipelineTrace;
}

export async function streamRAGMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  sessionId: string = "",
  onMeta: (meta: RAGStreamMeta) => void,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/rag/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        stream: true,
        use_rag: true,
        session_id: sessionId,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;

        const jsonStr = trimmed.slice(6);
        try {
          const data = JSON.parse(jsonStr);
          // Check for RAG metadata chunk
          if (data.rag_meta) {
            onMeta(data.rag_meta);
          }
          if (data.done) {
            onDone();
            return;
          }
          if (data.delta) {
            onToken(data.delta);
          }
        } catch {
          // skip malformed JSON
        }
      }
    }

    onDone();
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

// ── Memory Management ───────────────────────────────────────────

export async function getMemories(
  limit: number = 50,
  offset: number = 0,
): Promise<{ memories: MemoryObject[]; total: number }> {
  const res = await fetch(`${API_BASE}/memories?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch memories: ${res.status}`);
  return res.json();
}

export async function ingestMemory(
  content: string,
  source: string = "manual",
): Promise<{ status: string; memory: MemoryObject }> {
  const res = await fetch(`${API_BASE}/memories/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, source }),
  });
  if (!res.ok) throw new Error(`Failed to ingest memory: ${res.status}`);
  return res.json();
}

export async function searchMemories(
  query: string,
  topK: number = 10,
): Promise<{ results: MemoryObject[]; count: number }> {
  const res = await fetch(`${API_BASE}/memories/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Failed to search memories: ${res.status}`);
  return res.json();
}

export async function deleteMemory(memoryId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/memories/${memoryId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete memory: ${res.status}`);
}

// ── Knowledge Graph ─────────────────────────────────────────────

export async function getGraphData(): Promise<GraphData> {
  const res = await fetch(`${API_BASE}/graph`);
  if (!res.ok) throw new Error(`Failed to fetch graph: ${res.status}`);
  return res.json();
}

export async function getEntities(): Promise<{ entities: Record<string, unknown>[] }> {
  const res = await fetch(`${API_BASE}/entities`);
  if (!res.ok) throw new Error(`Failed to fetch entities: ${res.status}`);
  return res.json();
}

// ── Belief Evolution ────────────────────────────────────────────

export async function getBeliefDeltas(
  limit: number = 50,
): Promise<{
  beliefs: {
    id: string;
    topic: string;
    old_belief_text: string;
    new_belief_text: string;
    change_type: string;
    confidence: number;
    detected_at: string;
  }[];
}> {
  const res = await fetch(`${API_BASE}/beliefs?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch beliefs: ${res.status}`);
  return res.json();
}

// ── GraphRAG Communities ────────────────────────────────────────

export async function getCommunities(): Promise<{
  communities: {
    community_id: number;
    members: string[];
    size: number;
    memory_count: number;
  }[];
}> {
  const res = await fetch(`${API_BASE}/communities`);
  if (!res.ok) throw new Error(`Failed to fetch communities: ${res.status}`);
  return res.json();
}

// ── RAG Stats ───────────────────────────────────────────────────

export async function getRAGStats(): Promise<RAGStats> {
  const res = await fetch(`${API_BASE}/rag/stats`);
  if (!res.ok) throw new Error(`Failed to fetch stats: ${res.status}`);
  return res.json();
}

export async function getRAGHealth(): Promise<{
  rag_initialized: boolean;
  stats: RAGStats;
}> {
  const res = await fetch(`${API_BASE}/rag/health`);
  if (!res.ok) throw new Error(`Failed to fetch RAG health: ${res.status}`);
  return res.json();
}

// ── Ambient Voice Service ───────────────────────────────────────

export async function startAmbient(): Promise<{ success: boolean; error?: string; status: string }> {
  const res = await fetch(`${API_BASE}/ambient/start`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to start ambient: ${res.status}`);
  return res.json();
}

export async function stopAmbient(): Promise<{ success: boolean; status: string }> {
  const res = await fetch(`${API_BASE}/ambient/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to stop ambient: ${res.status}`);
  return res.json();
}

export async function pauseAmbient(): Promise<{ success: boolean; status: string }> {
  const res = await fetch(`${API_BASE}/ambient/pause`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to pause ambient: ${res.status}`);
  return res.json();
}

export async function resumeAmbient(): Promise<{ success: boolean; status: string }> {
  const res = await fetch(`${API_BASE}/ambient/resume`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to resume ambient: ${res.status}`);
  return res.json();
}

export async function getAmbientStatus(): Promise<AmbientState> {
  const res = await fetch(`${API_BASE}/ambient/status`);
  if (!res.ok) throw new Error(`Failed to fetch ambient status: ${res.status}`);
  return res.json();
}

export async function getAmbientConfig(): Promise<AmbientConfig> {
  const res = await fetch(`${API_BASE}/ambient/config`);
  if (!res.ok) throw new Error(`Failed to fetch ambient config: ${res.status}`);
  return res.json();
}

export async function updateAmbientConfig(updates: Partial<AmbientConfig>): Promise<AmbientConfig> {
  const res = await fetch(`${API_BASE}/ambient/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error(`Failed to update ambient config: ${res.status}`);
  return res.json();
}

export async function startEnrollment(durationSeconds: number = 20): Promise<{
  success: boolean;
  message: string;
  samples_used?: number;
  consistency?: number;
}> {
  const res = await fetch(`${API_BASE}/ambient/enroll`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ duration_seconds: durationSeconds }),
  });
  if (!res.ok) throw new Error(`Enrollment failed: ${res.status}`);
  return res.json();
}

export async function getEnrollmentStatus(): Promise<{ enrolled: boolean }> {
  const res = await fetch(`${API_BASE}/ambient/enrollment-status`);
  if (!res.ok) throw new Error(`Failed to check enrollment: ${res.status}`);
  return res.json();
}

export async function setSpeakerAlias(speakerLabel: string, name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/ambient/speaker-alias`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker_label: speakerLabel, name }),
  });
  if (!res.ok) throw new Error(`Failed to set alias: ${res.status}`);
}

export async function getConversations(
  limit: number = 50,
  offset: number = 0
): Promise<{ conversations: ConversationRecord[]; total: number }> {
  const res = await fetch(`${API_BASE}/ambient/conversations?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch conversations: ${res.status}`);
  return res.json();
}

export async function getConversation(convId: string): Promise<ConversationRecord> {
  const res = await fetch(`${API_BASE}/ambient/conversations/${convId}`);
  if (!res.ok) throw new Error(`Failed to fetch conversation: ${res.status}`);
  return res.json();
}

export async function getLiveTranscript(): Promise<{ turns: ConversationTurn[] }> {
  const res = await fetch(`${API_BASE}/ambient/live-transcript`);
  if (!res.ok) throw new Error(`Failed to fetch live transcript: ${res.status}`);
  return res.json();
}

// ── Text-to-Speech ──────────────────────────────────────────────

export async function synthesizeSpeech(text: string): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}/tts/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`TTS synthesis failed: ${res.status}`);
  return res.arrayBuffer();
}

export async function getTTSStatus(): Promise<{
  available: boolean;
  voice: string | null;
  total_syntheses: number;
}> {
  const res = await fetch(`${API_BASE}/tts/status`);
  if (!res.ok) throw new Error(`Failed to fetch TTS status: ${res.status}`);
  return res.json();
}

// ── Voice Query ─────────────────────────────────────────────────

export async function voiceQuery(audioBase64: string): Promise<VoiceQueryResult> {
  const res = await fetch(`${API_BASE}/voice/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_base64: audioBase64 }),
  });
  if (!res.ok) throw new Error(`Voice query failed: ${res.status}`);
  return res.json();
}

// ── Pipeline Observability ──────────────────────────────────────

export async function getPipelineTraces(limit = 20): Promise<TracesResponse> {
  const res = await fetch(`${API_BASE}/rag/traces?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch traces: ${res.status}`);
  return res.json();
}

export async function getPipelineTraceById(traceId: string): Promise<PipelineTrace> {
  const res = await fetch(`${API_BASE}/rag/traces/${traceId}`);
  if (!res.ok) throw new Error(`Failed to fetch trace: ${res.status}`);
  return res.json();
}
