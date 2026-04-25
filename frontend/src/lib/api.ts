import {
  ChatSettings,
  DEFAULT_SETTINGS,
  InferenceMode,
  LLMProviderType,
  MemoryObject,
  GraphData,
  RAGStats,
  EvidenceCard,
  ModelpackManifest,
  AmbientState,
  AmbientLiveStatus,
  AmbientConfig,
  AmbientClientAudioResponse,
  AmbientClientSessionInfo,
  ConversationRecord,
  VoiceQueryResult,
  ConversationTurn,
  PipelineTrace,
  TracesResponse,
  VoiceProviders,
  VoiceProviderType,
  LivePipelineEvent,
  RuntimePermissionRequest,
  RuntimeTaskEvent,
  RuntimeTaskListResponse,
  RuntimeTaskReferences,
  RuntimeTaskSnapshot,
  RuntimeSelection,
} from "./types";

const API_BASE = "/api";

function normalizeBaseUrl(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

// Direct backend URL for long-running SSE connections that may exceed
// the Next.js proxy timeout (RAG retrieval can take 30-60s before streaming).
const BACKEND_DIRECT = (() => {
  const envBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (envBase) {
    return normalizeBaseUrl(envBase);
  }

  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    return `${protocol}//${window.location.hostname}:8000/api`;
  }

  return "http://localhost:8000/api";
})();

export function getAmbientWebSocketUrl(): string {
  try {
    const direct = new URL(BACKEND_DIRECT, typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
    direct.protocol = direct.protocol === "https:" ? "wss:" : "ws:";
    const pathname = direct.pathname.replace(/\/+$/, "");
    direct.pathname = pathname.endsWith("/api")
      ? `${pathname.slice(0, -4)}/ws/ambient`
      : `${pathname}/ws/ambient`;
    direct.search = "";
    return direct.toString();
  } catch {
    if (typeof window !== "undefined") {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${window.location.host}/ws/ambient`;
    }
    return "ws://localhost:8000/ws/ambient";
  }
}

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
      llm_provider: settings.llmProvider || "local",
      inference_mode: settings.inferenceMode,
      allow_cloud_fallback: settings.allowCloudFallback,
      thinking_mode: settings.thinkingMode ?? true,
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
        llm_provider: settings.llmProvider || "local",
        inference_mode: settings.inferenceMode,
        allow_cloud_fallback: settings.allowCloudFallback,
        thinking_mode: settings.thinkingMode ?? true,
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
  runtime_tasks?: RuntimeTaskReferences;
}> {
  // Use direct backend URL to bypass Next.js proxy timeout
  const res = await fetch(`${BACKEND_DIRECT}/rag/chat`, {
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
      llm_provider: settings.llmProvider || "local",
      inference_mode: settings.inferenceMode,
      allow_cloud_fallback: settings.allowCloudFallback,
      thinking_mode: settings.thinkingMode ?? true,
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
  runtime_tasks?: RuntimeTaskReferences;
}

export async function streamRAGMessage(
  messages: { role: string; content: string }[],
  settings: ChatSettings = DEFAULT_SETTINGS,
  sessionId: string = "",
  onMeta: (meta: RAGStreamMeta) => void,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void,
  onReplace?: (text: string) => void,
): Promise<void> {
  try {
    // Use direct backend URL to bypass Next.js proxy timeout for long SSE streams.
    // RAG retrieval can take 30-60s before the first token is streamed.
    const res = await fetch(`${BACKEND_DIRECT}/rag/chat`, {
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
        llm_provider: settings.llmProvider || "local",
        inference_mode: settings.inferenceMode,
        allow_cloud_fallback: settings.allowCloudFallback,
        thinking_mode: settings.thinkingMode ?? true,
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
          // Handle hallucination replacement — server sends corrected content
          if (data.replace && onReplace) {
            onReplace(data.replace);
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

// ── LLM Provider Toggle ────────────────────────────────────────

export async function getLLMProvider(): Promise<{
  provider: LLMProviderType;
  active_backend?: "local" | "gemini";
  available: string[];
  gemini_configured: boolean;
  local_model_loaded: boolean;
}> {
  const res = await fetch(`${API_BASE}/llm/provider`);
  if (!res.ok) throw new Error(`Failed to fetch LLM provider: ${res.status}`);
  return res.json();
}

export async function setLLMProvider(
  provider: LLMProviderType,
): Promise<{ provider: string; status: string }> {
  const res = await fetch(`${API_BASE}/llm/provider`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to switch provider: ${res.status}`);
  }
  return res.json();
}

export async function getRuntimeMode(): Promise<{
  mode: InferenceMode;
  allow_cloud_fallback: boolean;
  supported_modes: InferenceMode[];
}> {
  const res = await fetch(`${API_BASE}/runtime/mode`);
  if (!res.ok) throw new Error(`Failed to fetch runtime mode: ${res.status}`);
  return res.json();
}

export async function setRuntimeMode(
  mode: InferenceMode,
  allowCloudFallback?: boolean,
): Promise<{
  mode: InferenceMode;
  allow_cloud_fallback: boolean;
  supported_modes: InferenceMode[];
}> {
  const res = await fetch(`${API_BASE}/runtime/mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, allow_cloud_fallback: allowCloudFallback }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to set runtime mode: ${res.status}`);
  }
  return res.json();
}

export async function getRuntimeProviders(): Promise<{
  selection: {
    mode: InferenceMode;
    llm_provider: LLMProviderType;
    stt_provider: VoiceProviderType;
    tts_provider: VoiceProviderType;
    allow_cloud_fallback: boolean;
    updated_at: string;
  };
  available: {
    llm: LLMProviderType[];
    stt: VoiceProviderType[];
    tts: VoiceProviderType[];
  };
  availability: {
    llm: Record<string, boolean>;
    stt: Record<string, boolean>;
    tts: Record<string, boolean>;
    ambient_available: boolean;
  };
}> {
  const res = await fetch(`${API_BASE}/runtime/providers`);
  if (!res.ok) throw new Error(`Failed to fetch runtime providers: ${res.status}`);
  return res.json();
}

export async function setRuntimeProviders(
  selection: Partial<RuntimeSelection>,
): Promise<{
  selection: {
    mode: InferenceMode;
    llm_provider: LLMProviderType;
    stt_provider: VoiceProviderType;
    tts_provider: VoiceProviderType;
    allow_cloud_fallback: boolean;
    updated_at: string;
  };
  fallback_applied: Array<{ target: string; from: string; to: string }>;
}> {
  const res = await fetch(`${API_BASE}/runtime/providers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      llm_provider: selection.llmProvider,
      stt_provider: selection.sttProvider,
      tts_provider: selection.ttsProvider,
      allow_cloud_fallback: selection.allowCloudFallback,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to set runtime providers: ${res.status}`);
  }
  return res.json();
}

export async function getRuntimeHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/runtime/health`);
  if (!res.ok) throw new Error(`Failed to fetch runtime health: ${res.status}`);
  return res.json();
}

export async function getModelpackManifest(): Promise<ModelpackManifest> {
  const res = await fetch(`${API_BASE}/modelpacks/manifest`);
  if (!res.ok) throw new Error(`Failed to fetch modelpack manifest: ${res.status}`);
  return res.json();
}

export async function verifyModelpack(
  filePath: string,
  expectedSha256: string,
): Promise<{
  verified: boolean;
  algorithm: string;
  file_path: string;
  file_size_bytes: number;
  expected_sha256: string;
  actual_sha256: string;
}> {
  const res = await fetch(`${API_BASE}/modelpacks/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_path: filePath, expected_sha256: expectedSha256 }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to verify modelpack: ${res.status}`);
  }
  return res.json();
}

// ── Memory Management (continued) ──────────────────────────────

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
): Promise<
  | { status: "ok"; memory: MemoryObject }
  | {
      status: "pending_approval";
      request_id: string;
      decision: Record<string, unknown>;
      permission_request: RuntimePermissionRequest;
      next: string;
    }
> {
  const res = await fetch(`${API_BASE}/memories/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, source }),
  });

  if (res.status === 202) {
    return res.json();
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to ingest memory: ${res.status}`);
  }

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

export async function deleteMemory(
  memoryId: string,
  permissionId?: string,
): Promise<
  | { status: "ok" | "not_found"; approved_execution?: boolean; permission_id?: string }
  | { status: "pending_execution"; permission_id: string; next: string }
  | {
      status: "pending_approval";
      request_id: string;
      decision: Record<string, unknown>;
      permission_request: RuntimePermissionRequest;
      next: string;
    }
> {
  const query = permissionId ? `?permission_id=${encodeURIComponent(permissionId)}` : "";
  const res = await fetch(`${API_BASE}/memories/${memoryId}${query}`, { method: "DELETE" });

  if (res.status === 202) {
    return res.json();
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to delete memory: ${res.status}`);
  }

  return res.json();
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

export async function startAmbientLive(): Promise<{
  success: boolean;
  status: string;
  mode?: string;
  live?: AmbientLiveStatus;
  error?: string;
}> {
  const res = await fetch(`${API_BASE}/ambient/live/start`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to start ambient live mode: ${res.status}`);
  return res.json();
}

export async function stopAmbient(): Promise<{ success: boolean; status: string }> {
  const res = await fetch(`${API_BASE}/ambient/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to stop ambient: ${res.status}`);
  return res.json();
}

export async function stopAmbientLive(): Promise<{ success: boolean; status: string; mode?: string }> {
  const res = await fetch(`${API_BASE}/ambient/live/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to stop ambient live mode: ${res.status}`);
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

export async function getAmbientLiveStatus(): Promise<AmbientLiveStatus> {
  const res = await fetch(`${API_BASE}/ambient/live/status`);
  if (!res.ok) throw new Error(`Failed to fetch ambient live status: ${res.status}`);
  return res.json();
}

export async function startAmbientClientSession(body?: {
  platform?: string;
  metadata?: Record<string, unknown>;
}): Promise<{
  success: boolean;
  session_id: string;
  platform: string;
  metadata: Record<string, unknown>;
  session: AmbientClientSessionInfo;
}> {
  const res = await fetch(`${API_BASE}/ambient/client/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      platform: body?.platform || "web",
      metadata: body?.metadata || {},
    }),
  });
  if (!res.ok) throw new Error(`Failed to start client companion session: ${res.status}`);
  return res.json();
}

export async function stopAmbientClientSession(body: {
  sessionId: string;
  reason?: string;
}): Promise<{
  success: boolean;
  session: AmbientClientSessionInfo;
  triggered_agents?: string[];
}> {
  const res = await fetch(`${API_BASE}/ambient/client/session/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: body.sessionId,
      reason: body.reason || "user_request",
    }),
  });
  if (!res.ok) throw new Error(`Failed to stop client companion session: ${res.status}`);
  return res.json();
}

export async function getAmbientClientSessions(): Promise<{
  active_session_id: string;
  followup_until: number;
  active_sessions: AmbientClientSessionInfo[];
  sessions: AmbientClientSessionInfo[];
}> {
  const res = await fetch(`${API_BASE}/ambient/client/sessions`);
  if (!res.ok) throw new Error(`Failed to fetch client companion sessions: ${res.status}`);
  return res.json();
}

export async function processAmbientClientAudio(body: {
  sessionId?: string;
  audioBase64: string;
  mimeType: string;
  platform?: string;
  language?: string;
  estimatedDurationS?: number;
  metadata?: Record<string, unknown>;
}): Promise<AmbientClientAudioResponse> {
  const res = await fetch(`${API_BASE}/ambient/client/process-audio`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: body.sessionId || "",
      audio_base64: body.audioBase64,
      mime_type: body.mimeType,
      platform: body.platform || "web",
      language: body.language,
      estimated_duration_s: body.estimatedDurationS || 0,
      metadata: body.metadata || {},
    }),
  });
  if (!res.ok) throw new Error(`Failed to process client companion audio: ${res.status}`);
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

// ── Voice Provider Management ───────────────────────────────────

export async function getVoiceProviders(): Promise<VoiceProviders> {
  const res = await fetch(`${API_BASE}/ambient/voice-providers`);
  if (!res.ok) throw new Error(`Failed to fetch voice providers: ${res.status}`);
  return res.json();
}

export async function setSTTProvider(provider: VoiceProviderType): Promise<{ success: boolean; stt_provider: string }> {
  const res = await fetch(`${API_BASE}/ambient/stt-provider`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `Failed to set STT provider: ${res.status}`);
  }
  return res.json();
}

export async function setTTSProvider(provider: VoiceProviderType): Promise<{ success: boolean; tts_provider: string }> {
  const res = await fetch(`${API_BASE}/ambient/tts-provider`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `Failed to set TTS provider: ${res.status}`);
  }
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

/**
 * Subscribe to real-time pipeline events via SSE.
 * Returns an AbortController — call .abort() to stop.
 */
export function subscribePipelineEvents(
  onEvent: (event: LivePipelineEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BACKEND_DIRECT}/rag/pipeline-events`, {
        signal: controller.signal,
        headers: { Accept: "text/event-stream" },
      });
      if (!res.ok || !res.body) {
        onError?.(new Error(`SSE connect failed: ${res.status}`));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.slice(6));
              onEvent(payload as LivePipelineEvent);
            } catch { /* skip malformed */ }
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  })();

  return controller;
}

/**
 * Fetch aggregate observability metrics.
 */
export async function getObservabilityMetrics(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BACKEND_DIRECT}/rag/observability/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
  return res.json();
}

// ── Runtime Safety / Approval Queue ───────────────────────────

export async function getRuntimeSafetyPermissions(): Promise<{
  count: number;
  pending: RuntimePermissionRequest[];
  expired_count?: number;
}> {
  const res = await fetch(`${API_BASE}/runtime/safety/permissions`);
  if (!res.ok) throw new Error(`Failed to fetch runtime permissions: ${res.status}`);
  return res.json();
}

export async function getRuntimeSafetyExecutorStatus(): Promise<{
  enabled: boolean;
  running: boolean;
  poll_interval_seconds?: number;
  execution_timeout_seconds?: number;
  max_attempts?: number;
  summary: {
    approved_total: number;
    pending_total: number;
    running: number;
    waiting_retry: number;
    completed: number;
    failed: number;
    unsupported: number;
    idle: number;
  };
}> {
  const res = await fetch(`${API_BASE}/runtime/safety/executor`);
  if (!res.ok) throw new Error(`Failed to fetch runtime executor status: ${res.status}`);
  return res.json();
}

export async function resolveRuntimeSafetyPermission(
  permissionId: string,
  approve: boolean,
  actor = "frontend-operator",
  note = "",
): Promise<{ resolved: RuntimePermissionRequest }> {
  const res = await fetch(`${API_BASE}/runtime/safety/permissions/${encodeURIComponent(permissionId)}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approve, actor, note }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to resolve permission: ${res.status}`);
  }

  return res.json();
}

export async function getRuntimeTasks(): Promise<RuntimeTaskListResponse> {
  const res = await fetch(`${API_BASE}/runtime/tasks`);
  if (!res.ok) throw new Error(`Failed to fetch runtime tasks: ${res.status}`);
  return res.json();
}

export async function getRuntimeTask(taskId: string): Promise<{ task: RuntimeTaskSnapshot }> {
  const res = await fetch(`${API_BASE}/runtime/tasks/${encodeURIComponent(taskId)}`);
  if (!res.ok) throw new Error(`Failed to fetch runtime task: ${res.status}`);
  return res.json();
}

export async function cancelRuntimeTask(
  taskId: string,
  reason = "Cancelled from observability dashboard",
  propagate = true,
): Promise<{ cancelled_task_ids: string[] }> {
  const res = await fetch(`${API_BASE}/runtime/tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason, propagate }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to cancel runtime task: ${res.status}`);
  }

  return res.json();
}

export function subscribeRuntimeTaskEvents(
  onEvent: (event: RuntimeTaskEvent) => void,
  onError?: (err: Error) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BACKEND_DIRECT}/runtime/tasks/events`, {
        signal: controller.signal,
        headers: { Accept: "text/event-stream" },
      });

      if (!res.ok || !res.body) {
        onError?.(new Error(`Runtime task SSE connect failed: ${res.status}`));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            onEvent(payload as RuntimeTaskEvent);
          } catch {
            // Skip malformed events.
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  })();

  return controller;
}

// ── PageIndex Document Management ───────────────────────────────

export interface PageIndexDocument {
  doc_id: string;
  filename: string;
  uploaded_at: string;
  status: string;
  estimated_pages: number;
  file_hash: string;
}

export interface PageIndexUsage {
  queries_used: number;
  pages_used: number;
  queries_limit: number;
  pages_limit: number;
  month: string;
}

export async function uploadDocument(file: File): Promise<{
  status: string;
  doc_id: string;
  filename: string;
  processing_status: string;
  already_indexed: boolean;
}> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function listDocuments(): Promise<{
  documents: PageIndexDocument[];
  total: number;
  pageindex_enabled: boolean;
}> {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error(`Failed to list documents: ${res.status}`);
  return res.json();
}

export async function getDocumentInfo(docId: string): Promise<PageIndexDocument & { live_status: string }> {
  const res = await fetch(`${API_BASE}/documents/${docId}`);
  if (!res.ok) throw new Error(`Failed to get document: ${res.status}`);
  return res.json();
}

export async function getDocumentTree(docId: string): Promise<{ doc_id: string; tree: unknown }> {
  const res = await fetch(`${API_BASE}/documents/${docId}/tree`);
  if (!res.ok) throw new Error(`Failed to get tree: ${res.status}`);
  return res.json();
}

export async function deleteDocument(docId: string): Promise<{ status: string; doc_id: string }> {
  const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete document: ${res.status}`);
  return res.json();
}

export async function queryDocuments(
  query: string,
  topK: number = 5,
): Promise<{
  answer: string;
  sections: { page: number; content: string; doc_id: string; score: number }[];
  doc_count: number;
}> {
  const res = await fetch(`${API_BASE}/documents/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Document query failed: ${res.status}`);
  return res.json();
}

export async function getPageIndexUsage(): Promise<{
  enabled: boolean;
  connected?: boolean;
  usage: PageIndexUsage;
  stats?: {
    connected: boolean;
    documents: number;
    ready_documents: number;
  };
}> {
  const res = await fetch(`${API_BASE}/documents/usage`);
  if (!res.ok) throw new Error(`Failed to get usage: ${res.status}`);
  return res.json();
}
