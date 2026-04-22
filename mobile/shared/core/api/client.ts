import {
  AgentConfigInfo,
  AgentQueryResponse,
  AgentSessionInfo,
  AmbientConfig,
  AmbientLiveStatus,
  AmbientState,
  ChatSettings,
  ConversationRecord,
  ConversationTurn,
  CortexEvent,
  DEFAULT_SETTINGS,
  EvidenceCard,
  GraphData,
  InferenceMode,
  RuntimeExecutorStatus,
  RuntimePermissionRequest,
  RuntimeTaskEvent,
  RuntimeTaskListResponse,
  RuntimeTaskSnapshot,
  LLMProviderType,
  LivePipelineEvent,
  MemoryObject,
  ModelStatus,
  ModelpackManifest,
  PipelineTrace,
  QueryAnalysis,
  RAGStats,
  RuntimeSelection,
  TierClassification,
  TracesResponse,
  WikiCompactionSummary,
  WikiLintSummary,
  WikiPageInfo,
  VoiceQueryResult,
  VoiceProviders,
  VoiceProviderType,
} from "../types";

export interface ApiConfig {
  baseUrl: string;
}

export interface ChatCompletion {
  content: string;
  thinking?: string;
  evidence?: EvidenceCard[];
  agents_used?: string[];
  confidence?: number;
  query_analysis?: QueryAnalysis;
  processing_time_ms?: number;
  cache_hit?: boolean;
  pipeline_trace?: PipelineTrace | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface RAGStreamMeta {
  evidence?: EvidenceCard[];
  agents_used?: string[];
  confidence?: number;
  query_analysis?: QueryAnalysis;
  thinking?: string;
  pipeline_trace?: PipelineTrace | null;
}

export interface SendMessagePayload {
  messages: { role: string; content: string }[];
  settings?: ChatSettings;
}

export interface StreamHandlers {
  onToken: (token: string) => void;
  onMeta?: (meta: RAGStreamMeta) => void;
  onReplace?: (content: string) => void;
  onDone: () => void;
  onError: (error: Error) => void;
}

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

interface LLMProviderResponse {
  provider: LLMProviderType;
  active_backend?: "local" | "gemini";
  available: string[];
  gemini_configured: boolean;
  local_model_loaded: boolean;
}

export interface RuntimeModeResponse {
  mode: InferenceMode;
  allow_cloud_fallback: boolean;
  supported_modes: InferenceMode[];
}

export interface RuntimeProvidersResponse {
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
}

export interface TTSStatus {
  available: boolean;
  voice: string | null;
  total_syntheses?: number;
}

export interface LiveTranscriptResponse {
  turns: ConversationTurn[];
}

export interface AgentStreamRequest {
  query: string;
  sessionId?: string | null;
  tierOverride?: string;
}

export interface AgentStreamResult {
  sessionId: string | null;
  answer: string;
}

export interface AgentStreamHandlers {
  onEvent?: (event: CortexEvent) => void;
  onDone?: (result: AgentStreamResult) => void;
  onError?: (error: Error) => void;
}

export interface AgentEventStreamHandlers {
  onEvent: (event: CortexEvent) => void;
  onError?: (error: Error) => void;
}

export interface RuntimePermissionsResponse {
  count: number;
  pending: RuntimePermissionRequest[];
  expired_count?: number;
}

export interface RuntimeResolvePermissionResponse {
  resolved: RuntimePermissionRequest;
}

export interface RuntimeTaskResponse {
  task: RuntimeTaskSnapshot;
}

export interface RuntimeCancelTaskResponse {
  cancelled_task_ids: string[];
}

export interface RuntimeTaskStreamHandlers {
  onEvent: (event: RuntimeTaskEvent) => void;
  onError?: (error: Error) => void;
}

function normalizeBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;
}

function ensureApiPath(rawBaseUrl: string): string {
  const normalized = normalizeBaseUrl(rawBaseUrl.trim());

  // Prefer URL parsing when the value is absolute; fallback keeps relative values working.
  try {
    const parsed = new URL(normalized);
    const pathname = parsed.pathname.replace(/\/+$/, "");
    if (pathname === "" || pathname === "/") {
      parsed.pathname = "/api";
    } else if (!pathname.endsWith("/api")) {
      parsed.pathname = `${pathname}/api`;
    } else {
      parsed.pathname = pathname;
    }
    return normalizeBaseUrl(parsed.toString());
  } catch {
    return normalized.endsWith("/api") ? normalized : `${normalized}/api`;
  }
}

function extractHostname(rawUrl: string): string | null {
  try {
    return new URL(rawUrl).hostname;
  } catch {
    return null;
  }
}

function isAndroidEmulatorHost(rawUrl: string): boolean {
  return extractHostname(rawUrl) === "10.0.2.2";
}

async function parseError(res: Response): Promise<Error> {
  const detail = await res
    .json()
    .then((data) => (typeof data?.detail === "string" ? data.detail : undefined))
    .catch(() => undefined);
  return new Error(detail || `Request failed with status ${res.status}`);
}

async function consumeSSE(
  res: Response,
  onData: (payload: unknown) => void,
): Promise<void> {
  if (!res.ok || !res.body) {
    throw await parseError(res);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      return;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line.startsWith("data: ")) {
        continue;
      }

      const jsonPayload = line.slice(6).trim();
      if (!jsonPayload) {
        continue;
      }

      try {
        onData(JSON.parse(jsonPayload));
      } catch {
        // Skip malformed payloads.
      }
    }
  }
}

export function createApiClient(config: ApiConfig) {
  const baseUrl = ensureApiPath(config.baseUrl);

  async function getModelStatus(): Promise<ModelStatus> {
    const res = await fetch(`${baseUrl}/health`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function sendMessage(payload: SendMessagePayload): Promise<ChatCompletion> {
    const settings = payload.settings || DEFAULT_SETTINGS;
    const endpoint = settings.useRAG ? `${baseUrl}/rag/chat` : `${baseUrl}/chat`;

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: payload.messages,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        stream: false,
        use_rag: settings.useRAG,
        llm_provider: settings.llmProvider,
        inference_mode: settings.inferenceMode,
        allow_cloud_fallback: settings.allowCloudFallback,
      }),
    });

    if (!res.ok) {
      throw await parseError(res);
    }

    const data = await res.json();
    return {
      content: data.content || "",
      thinking: data.thinking,
      evidence: Array.isArray(data.evidence) ? data.evidence : undefined,
      agents_used: Array.isArray(data.agents_used) ? data.agents_used : undefined,
      confidence: typeof data.confidence === "number" ? data.confidence : undefined,
      query_analysis: data.query_analysis,
      processing_time_ms:
        typeof data.processing_time_ms === "number" ? data.processing_time_ms : undefined,
      cache_hit: typeof data.cache_hit === "boolean" ? data.cache_hit : undefined,
      pipeline_trace: data.pipeline_trace || null,
      usage: data.usage,
    };
  }

  async function streamMessage(
    payload: SendMessagePayload,
    handlers: StreamHandlers,
  ): Promise<void> {
    const settings = payload.settings || DEFAULT_SETTINGS;
    const endpoint = settings.useRAG ? `${baseUrl}/rag/chat` : `${baseUrl}/chat`;

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: payload.messages,
          temperature: settings.temperature,
          top_p: settings.topP,
          max_tokens: settings.maxTokens,
          stream: true,
          use_rag: settings.useRAG,
          llm_provider: settings.llmProvider,
          inference_mode: settings.inferenceMode,
          allow_cloud_fallback: settings.allowCloudFallback,
        }),
      });

      if (!res.ok || !res.body) {
        throw await parseError(res);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          handlers.onDone();
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) {
            continue;
          }

          try {
            const data = JSON.parse(trimmed.slice(6));

            if (data.rag_meta && handlers.onMeta) {
              handlers.onMeta(data.rag_meta as RAGStreamMeta);
            }

            if (typeof data.replace === "string") {
              handlers.onReplace?.(data.replace);
            }

            if (data.done) {
              handlers.onDone();
              return;
            }

            if (typeof data.delta === "string" && data.delta.length > 0) {
              handlers.onToken(data.delta);
            }
          } catch {
            // Skip malformed event payloads.
          }
        }
      }
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error(String(error)));
    }
  }

  async function getLLMProvider(): Promise<LLMProviderResponse> {
    const res = await fetch(`${baseUrl}/llm/provider`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function setLLMProvider(
    provider: LLMProviderType,
  ): Promise<{ provider: string; status: string }> {
    const res = await fetch(`${baseUrl}/llm/provider`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeMode(): Promise<RuntimeModeResponse> {
    const res = await fetch(`${baseUrl}/runtime/mode`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function setRuntimeMode(
    mode: InferenceMode,
    allowCloudFallback?: boolean,
  ): Promise<RuntimeModeResponse> {
    const res = await fetch(`${baseUrl}/runtime/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        allow_cloud_fallback: allowCloudFallback,
      }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeProviders(): Promise<RuntimeProvidersResponse> {
    const res = await fetch(`${baseUrl}/runtime/providers`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function setRuntimeProviders(
    selection: Partial<RuntimeSelection>,
  ): Promise<{ selection: RuntimeProvidersResponse["selection"]; fallback_applied: Array<{ target: string; from: string; to: string }> }> {
    const res = await fetch(`${baseUrl}/runtime/providers`, {
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
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeHealth(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/health`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeSafetyPermissions(): Promise<RuntimePermissionsResponse> {
    const res = await fetch(`${baseUrl}/runtime/safety/permissions`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeSafetyExecutorStatus(): Promise<RuntimeExecutorStatus> {
    const res = await fetch(`${baseUrl}/runtime/safety/executor`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function resolveRuntimeSafetyPermission(
    permissionId: string,
    approve: boolean,
    actor: string = "mobile-operator",
    note: string = "",
  ): Promise<RuntimeResolvePermissionResponse> {
    const res = await fetch(
      `${baseUrl}/runtime/safety/permissions/${encodeURIComponent(permissionId)}/resolve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approve, actor, note }),
      },
    );
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeTasks(): Promise<RuntimeTaskListResponse> {
    const res = await fetch(`${baseUrl}/runtime/tasks`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeTask(taskId: string): Promise<RuntimeTaskResponse> {
    const res = await fetch(`${baseUrl}/runtime/tasks/${encodeURIComponent(taskId)}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function cancelRuntimeTask(
    taskId: string,
    reason: string = "Cancelled from mobile runtime center",
    propagate: boolean = true,
  ): Promise<RuntimeCancelTaskResponse> {
    const res = await fetch(`${baseUrl}/runtime/tasks/${encodeURIComponent(taskId)}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason, propagate }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  function subscribeRuntimeTaskEvents(
    handlers: RuntimeTaskStreamHandlers,
  ): AbortController {
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${baseUrl}/runtime/tasks/events`, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        });

        await consumeSSE(res, (payload) => {
          handlers.onEvent(payload as RuntimeTaskEvent);
        });
      } catch (error) {
        const abortLike =
          typeof error === "object" &&
          error !== null &&
          "name" in error &&
          (error as { name?: string }).name === "AbortError";
        if (abortLike) {
          return;
        }
        handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    })();

    return controller;
  }

  async function getModelpackManifest(): Promise<ModelpackManifest> {
    const res = await fetch(`${baseUrl}/modelpacks/manifest`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function verifyModelpack(
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
    const res = await fetch(`${baseUrl}/modelpacks/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_path: filePath, expected_sha256: expectedSha256 }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getMemories(
    limit: number = 50,
    offset: number = 0,
  ): Promise<{ memories: MemoryObject[]; total: number }> {
    const res = await fetch(`${baseUrl}/memories?limit=${limit}&offset=${offset}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function searchMemories(
    query: string,
    topK: number = 10,
  ): Promise<{ results: MemoryObject[]; count: number }> {
    const res = await fetch(`${baseUrl}/memories/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function ingestMemory(
    content: string,
    source: string = "manual",
  ): Promise<{ status: string; memory: MemoryObject }> {
    const res = await fetch(`${baseUrl}/memories/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, source }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function deleteMemory(memoryId: string): Promise<void> {
    const res = await fetch(`${baseUrl}/memories/${memoryId}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      throw await parseError(res);
    }
  }

  async function getGraphData(): Promise<GraphData> {
    const res = await fetch(`${baseUrl}/graph`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function listDocuments(): Promise<{
    documents: PageIndexDocument[];
    total: number;
    pageindex_enabled: boolean;
  }> {
    const res = await fetch(`${baseUrl}/documents`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function uploadDocument(formData: FormData): Promise<{
    status: string;
    doc_id: string;
    filename: string;
    processing_status: string;
    already_indexed: boolean;
  }> {
    const res = await fetch(`${baseUrl}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRAGStats(): Promise<RAGStats> {
    const res = await fetch(`${baseUrl}/rag/stats`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRAGHealth(): Promise<{ rag_initialized: boolean; stats: RAGStats }> {
    const res = await fetch(`${baseUrl}/rag/health`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getPipelineTraces(limit: number = 20): Promise<TracesResponse> {
    const res = await fetch(`${baseUrl}/rag/traces?limit=${limit}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getPipelineTraceById(traceId: string): Promise<PipelineTrace> {
    const res = await fetch(`${baseUrl}/rag/traces/${traceId}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  function subscribePipelineEvents(
    onEvent: (event: LivePipelineEvent) => void,
    onError?: (error: Error) => void,
  ): AbortController {
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${baseUrl}/rag/pipeline-events`, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        });

        await consumeSSE(res, (payload) => {
          onEvent(payload as LivePipelineEvent);
        });
      } catch (error) {
        const abortLike =
          typeof error === "object" &&
          error !== null &&
          "name" in error &&
          (error as { name?: string }).name === "AbortError";
        if (abortLike) {
          return;
        }
        onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    })();

    return controller;
  }

  async function getAmbientStatus(): Promise<AmbientState> {
    const res = await fetch(`${baseUrl}/ambient/status`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function ambientAction(action: "start" | "stop" | "pause" | "resume") {
    const res = await fetch(`${baseUrl}/ambient/${action}`, { method: "POST" });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function startAmbientLive(): Promise<{
    success: boolean;
    status: string;
    mode?: string;
    live?: AmbientLiveStatus;
    error?: string;
  }> {
    const res = await fetch(`${baseUrl}/ambient/live/start`, { method: "POST" });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function stopAmbientLive(): Promise<{ success: boolean; status: string; mode?: string }> {
    const res = await fetch(`${baseUrl}/ambient/live/stop`, { method: "POST" });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getAmbientLiveStatus(): Promise<AmbientLiveStatus> {
    const res = await fetch(`${baseUrl}/ambient/live/status`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function injectAmbientLiveAudio(
    audioBase64: string,
    sampleRateHz: number = 16000,
  ): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/ambient/live/inject-audio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_base64: audioBase64, sample_rate: sampleRateHz }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function setSpeakerAlias(speakerLabel: string, name: string): Promise<void> {
    const res = await fetch(`${baseUrl}/ambient/speaker-alias`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker_label: speakerLabel, name }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
  }

  async function getAmbientConfig(): Promise<AmbientConfig> {
    const res = await fetch(`${baseUrl}/ambient/config`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function updateAmbientConfig(
    updates: Partial<AmbientConfig>,
  ): Promise<AmbientConfig> {
    const res = await fetch(`${baseUrl}/ambient/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getVoiceProviders(): Promise<VoiceProviders> {
    const res = await fetch(`${baseUrl}/ambient/voice-providers`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function setSTTProvider(
    provider: VoiceProviderType,
  ): Promise<{ success: boolean; stt_provider: string }> {
    const res = await fetch(`${baseUrl}/ambient/stt-provider`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function setTTSProvider(
    provider: VoiceProviderType,
  ): Promise<{ success: boolean; tts_provider: string }> {
    const res = await fetch(`${baseUrl}/ambient/tts-provider`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getConversations(
    limit: number = 20,
    offset: number = 0,
  ): Promise<{ conversations: ConversationRecord[]; total: number }> {
    const res = await fetch(`${baseUrl}/ambient/conversations?limit=${limit}&offset=${offset}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getConversation(conversationId: string): Promise<ConversationRecord> {
    const res = await fetch(`${baseUrl}/ambient/conversations/${conversationId}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getLiveTranscript(): Promise<LiveTranscriptResponse> {
    const res = await fetch(`${baseUrl}/ambient/live-transcript`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getEnrollmentStatus(): Promise<{ enrolled: boolean; speaker_id_available?: boolean }> {
    const res = await fetch(`${baseUrl}/ambient/enrollment-status`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function startEnrollment(durationSeconds: number = 20): Promise<{
    success: boolean;
    message?: string;
    error?: string;
    samples_used?: number;
    consistency?: number;
  }> {
    const res = await fetch(`${baseUrl}/ambient/enroll`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ duration_seconds: durationSeconds }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getTTSStatus(): Promise<TTSStatus> {
    const res = await fetch(`${baseUrl}/tts/status`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function synthesizeSpeech(text: string, voice?: string): Promise<ArrayBuffer> {
    const res = await fetch(`${baseUrl}/tts/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.arrayBuffer();
  }

  async function voiceQuery(
    audioBase64: string,
    settings?: Record<string, unknown>,
  ): Promise<VoiceQueryResult> {
    const res = await fetch(`${baseUrl}/voice/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_base64: audioBase64, settings }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getObservabilityMetrics(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/rag/observability/metrics`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function createAgentSession(
    agentId: string = "l1_orchestrator",
    title?: string,
  ): Promise<{ session_id: string; agent_id: string; status: string }> {
    const res = await fetch(`${baseUrl}/agent/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agentId, title }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function listAgentSessions(): Promise<{ sessions: AgentSessionInfo[]; count: number }> {
    const res = await fetch(`${baseUrl}/agent/sessions`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getAgentSession(sessionId: string): Promise<AgentSessionInfo> {
    const res = await fetch(`${baseUrl}/agent/sessions/${encodeURIComponent(sessionId)}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function closeAgentSession(
    sessionId: string,
  ): Promise<{ session_id: string; status: string }> {
    const res = await fetch(`${baseUrl}/agent/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function agentQuery(
    query: string,
    sessionId?: string | null,
    tierOverride?: string,
  ): Promise<AgentQueryResponse> {
    const res = await fetch(`${baseUrl}/agent/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: sessionId || null,
        tier_override: tierOverride,
      }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function streamAgentQuery(
    request: AgentStreamRequest,
    handlers: AgentStreamHandlers,
  ): Promise<AgentStreamResult> {
    const res = await fetch(`${baseUrl}/agent/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: request.query,
        session_id: request.sessionId || null,
        tier_override: request.tierOverride,
      }),
    });

    let sessionId: string | null = request.sessionId || null;
    let answer = "";

    try {
      await consumeSSE(res, (payload) => {
        const event = payload as CortexEvent;
        if (event.session_id && !sessionId) {
          sessionId = event.session_id;
        }
        if (
          event.type === "agent_end" &&
          typeof event.data?.answer === "string"
        ) {
          answer = String(event.data.answer);
        }
        handlers.onEvent?.(event);
      });
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      handlers.onError?.(err);
      throw err;
    }

    const result = { sessionId, answer };
    handlers.onDone?.(result);
    return result;
  }

  function subscribeAgentEvents(handlers: AgentEventStreamHandlers): AbortController {
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${baseUrl}/agent/events`, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        });

        await consumeSSE(res, (payload) => {
          handlers.onEvent(payload as CortexEvent);
        });
      } catch (error) {
        const abortLike =
          typeof error === "object" &&
          error !== null &&
          "name" in error &&
          (error as { name?: string }).name === "AbortError";
        if (abortLike) {
          return;
        }
        handlers.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    })();

    return controller;
  }

  async function classifyQueryTier(query: string): Promise<TierClassification> {
    const res = await fetch(`${baseUrl}/agent/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function steerAgent(
    sessionId: string,
    text: string,
  ): Promise<{ status: string; queue: { steering: string[]; followUp: string[] } }> {
    const res = await fetch(`${baseUrl}/agent/sessions/${encodeURIComponent(sessionId)}/steer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function followUpAgent(
    sessionId: string,
    text: string,
  ): Promise<{ status: string; queue: { steering: string[]; followUp: string[] } }> {
    const res = await fetch(`${baseUrl}/agent/sessions/${encodeURIComponent(sessionId)}/follow-up`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function abortAgent(
    sessionId: string,
  ): Promise<{ status: string; session_id: string }> {
    const res = await fetch(`${baseUrl}/agent/sessions/${encodeURIComponent(sessionId)}/abort`, {
      method: "POST",
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function listAgentConfigs(): Promise<{ agents: AgentConfigInfo[]; count: number }> {
    const res = await fetch(`${baseUrl}/agent/configs`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function listWikiPages(): Promise<{
    pages: WikiPageInfo[];
    stats: { total_pages: number; total_topics: number; total_linked_claims: number };
    error?: string;
  }> {
    const res = await fetch(`${baseUrl}/wiki/pages`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getWikiPage(pageId: string): Promise<WikiPageInfo> {
    const res = await fetch(`${baseUrl}/wiki/pages/${encodeURIComponent(pageId)}`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function searchWiki(
    query: string,
    includeClaims: boolean = true,
  ): Promise<{ results: WikiPageInfo[] }> {
    const res = await fetch(`${baseUrl}/wiki/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, include_claims: includeClaims }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getClaimStats(): Promise<{ total: number; active: number; topics: number }> {
    const res = await fetch(`${baseUrl}/wiki/claims`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function searchClaims(
    query: string,
    minConfidence: number = 0.5,
  ): Promise<{
    claims: Array<{
      id: string;
      text: string;
      confidence: number;
      source_ids: string[];
      topic: string;
    }>;
  }> {
    const res = await fetch(`${baseUrl}/wiki/claims/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, min_confidence: minConfidence }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function rebuildWikiFromMemories(
    limit: number = 300,
    maxClaimsPerMemory: number = 10,
  ): Promise<{
    status: string;
    scanned: number;
    processed: number;
    pages_created: number;
    claims_linked: number;
  }> {
    const res = await fetch(`${baseUrl}/wiki/rebuild`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        limit,
        max_claims_per_memory: maxClaimsPerMemory,
      }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getWikiLintLatest(): Promise<WikiLintSummary> {
    const res = await fetch(`${baseUrl}/wiki/lint/latest`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function runWikiLint(body?: {
    page_id?: string;
    stale_days?: number;
    min_confidence?: number;
    limit?: number;
  }): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/wiki/lint/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getWikiCompactionLatest(): Promise<WikiCompactionSummary> {
    const res = await fetch(`${baseUrl}/wiki/compaction/latest`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function runWikiCompaction(body?: {
    page_id?: string;
    section?: string;
    max_tokens?: number;
    section_limit?: number;
    limit?: number;
  }): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/wiki/compaction/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function deleteDocument(docId: string): Promise<{ status: string; doc_id: string }> {
    const res = await fetch(`${baseUrl}/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getDocumentTree(docId: string): Promise<{ doc_id: string; tree: unknown }> {
    const res = await fetch(`${baseUrl}/documents/${docId}/tree`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function queryDocuments(
    query: string,
    topK: number = 5,
  ): Promise<{
    answer: string;
    sections: { page: number; content: string; doc_id: string; score: number }[];
    doc_count: number;
  }> {
    const res = await fetch(`${baseUrl}/documents/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getPageIndexUsage(): Promise<{
    enabled: boolean;
    connected?: boolean;
    usage: PageIndexUsage;
    stats?: {
      connected: boolean;
      documents: number;
      ready_documents: number;
    };
  }> {
    const res = await fetch(`${baseUrl}/documents/usage`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  return {
    getModelStatus,
    sendMessage,
    streamMessage,
    getLLMProvider,
    setLLMProvider,
    getRuntimeMode,
    setRuntimeMode,
    getRuntimeProviders,
    setRuntimeProviders,
    getRuntimeHealth,
    getRuntimeSafetyPermissions,
    getRuntimeSafetyExecutorStatus,
    resolveRuntimeSafetyPermission,
    getRuntimeTasks,
    getRuntimeTask,
    cancelRuntimeTask,
    subscribeRuntimeTaskEvents,
    getModelpackManifest,
    verifyModelpack,
    getMemories,
    searchMemories,
    ingestMemory,
    deleteMemory,
    getGraphData,
    listDocuments,
    uploadDocument,
    deleteDocument,
    getDocumentTree,
    queryDocuments,
    getPageIndexUsage,
    getRAGStats,
    getRAGHealth,
    getPipelineTraces,
    getPipelineTraceById,
    subscribePipelineEvents,
    getObservabilityMetrics,
    getAmbientStatus,
    ambientAction,
    startAmbientLive,
    stopAmbientLive,
    getAmbientLiveStatus,
    injectAmbientLiveAudio,
    getAmbientConfig,
    updateAmbientConfig,
    getVoiceProviders,
    setSTTProvider,
    setTTSProvider,
    setSpeakerAlias,
    getConversations,
    getConversation,
    getLiveTranscript,
    getEnrollmentStatus,
    startEnrollment,
    getTTSStatus,
    synthesizeSpeech,
    voiceQuery,
    createAgentSession,
    listAgentSessions,
    getAgentSession,
    closeAgentSession,
    agentQuery,
    streamAgentQuery,
    subscribeAgentEvents,
    classifyQueryTier,
    steerAgent,
    followUpAgent,
    abortAgent,
    listAgentConfigs,
    listWikiPages,
    getWikiPage,
    searchWiki,
    getClaimStats,
    searchClaims,
    rebuildWikiFromMemories,
    getWikiLintLatest,
    runWikiLint,
    getWikiCompactionLatest,
    runWikiCompaction,
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;

export function getDefaultApiBaseUrl(): string {
  const envBase =
    typeof process !== "undefined"
      ? process.env.EXPO_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL
      : undefined;

  const normalizedEnvBase = envBase?.trim() ? ensureApiPath(envBase.trim()) : undefined;

  // In web builds, prefer same-host backend because 10.0.2.2 is Android-emulator only.
  if (typeof window !== "undefined" && window.location?.hostname) {
    if (normalizedEnvBase && !isAndroidEmulatorHost(normalizedEnvBase)) {
      return normalizedEnvBase;
    }

    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    const host = window.location.hostname;
    return ensureApiPath(`${protocol}//${host}:8000`);
  }

  if (normalizedEnvBase) {
    return normalizedEnvBase;
  }

  // Android emulator: use 10.0.2.2. iOS simulator can use localhost.
  // For physical devices, set EXPO_PUBLIC_API_BASE_URL to your machine LAN IP.
  return ensureApiPath("http://10.0.2.2:8000");
}
