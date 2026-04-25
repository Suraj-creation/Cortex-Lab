import {
  AgentConfigInfo,
  AgentQueryResponse,
  AgentSessionInfo,
  AmbientConfig,
  AmbientClientAudioResponse,
  AmbientClientSessionInfo,
  AmbientLiveStatus,
  AmbientRetentionTrace,
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

export type ApiBaseSource =
  | "env"
  | "canonical"
  | "persisted"
  | "same-origin"
  | "local-dev";

export interface ApiBaseCandidate {
  url: string;
  source: ApiBaseSource;
}

export interface ApiBaseResolution {
  baseUrl: string;
  source: ApiBaseSource;
  reachable: boolean;
  checkedCandidates: ApiBaseCandidate[];
  status?: string;
  error?: string;
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

export interface MemoryIngestResponse {
  status: string;
  memory: MemoryObject;
  session?: AmbientClientSessionInfo | null;
  session_id?: string;
  retention_trace?: AmbientRetentionTrace;
  session_created?: boolean;
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

export interface RuntimeTaskCreatePayload {
  taskId?: string;
  parentTaskId?: string;
  permissionScope?: string[];
  metadata?: Record<string, unknown>;
}

export interface RuntimeSafetyEvaluatePayload {
  requestId?: string;
  toolName: string;
  commandText?: string;
  metadata?: Record<string, unknown>;
}

export interface SessionForgeRunPayload {
  sessionId?: string;
  lookbackDays?: number;
}

export interface SessionForgeSummaryPayload {
  sessionId?: string;
  windowDays?: number;
}

export interface ChronicleObservationPayload {
  note?: string;
  location?: Record<string, unknown>;
  peoplePresent?: string[];
  tags?: string[];
  mediaRef?: string;
  source?: string;
  emotionHint?: string;
}

export interface ChronicleSavePayload {
  title?: string;
  windowSeconds?: number;
  retrievalHint?: string;
  lifeDomain?: string;
}

export interface FeedbackPayload {
  query?: string;
  answer?: string;
  rating?: number;
  comment?: string;
  session_id?: string;
  type?: string;
  content?: string;
  context?: Record<string, unknown>;
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

function isLoopbackHost(rawUrl: string): boolean {
  const hostname = extractHostname(rawUrl);
  return hostname === "localhost" || hostname === "127.0.0.1";
}

function isLocalDevHost(rawUrl: string): boolean {
  return isAndroidEmulatorHost(rawUrl) || isLoopbackHost(rawUrl);
}

const CANONICAL_PRODUCTION_API_BASE = ensureApiPath(
  "https://cortex-backend-dbcv.onrender.com",
);

function pushUniqueCandidate(
  candidates: ApiBaseCandidate[],
  seen: Set<string>,
  source: ApiBaseSource,
  rawUrl: string | null | undefined,
) {
  const trimmed = rawUrl?.trim();
  if (!trimmed) {
    return;
  }

  const normalized = ensureApiPath(trimmed);
  if (seen.has(normalized)) {
    return;
  }

  seen.add(normalized);
  candidates.push({ url: normalized, source });
}

export function getApiBaseCandidates(
  persistedUrl?: string | null,
): ApiBaseCandidate[] {
  const candidates: ApiBaseCandidate[] = [];
  const seen = new Set<string>();
  const envBase =
    typeof process !== "undefined"
      ? process.env.EXPO_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL
      : undefined;

  pushUniqueCandidate(candidates, seen, "env", envBase);
  pushUniqueCandidate(candidates, seen, "canonical", CANONICAL_PRODUCTION_API_BASE);

  if (persistedUrl?.trim() && !isLocalDevHost(persistedUrl)) {
    pushUniqueCandidate(candidates, seen, "persisted", persistedUrl);
  }

  if (typeof window !== "undefined" && window.location?.origin) {
    pushUniqueCandidate(candidates, seen, "same-origin", window.location.origin);
  }

  if (persistedUrl?.trim() && isLocalDevHost(persistedUrl)) {
    pushUniqueCandidate(candidates, seen, "persisted", persistedUrl);
  }

  pushUniqueCandidate(candidates, seen, "local-dev", "http://10.0.2.2:8000");
  pushUniqueCandidate(candidates, seen, "local-dev", "http://localhost:8000");
  pushUniqueCandidate(candidates, seen, "local-dev", "http://127.0.0.1:8000");

  return candidates;
}

export async function probeApiBaseUrl(
  baseUrl: string,
  timeoutMs: number = 10000,
): Promise<{ reachable: boolean; status?: string; error?: string }> {
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timeout =
    controller !== null
      ? setTimeout(() => {
          controller.abort();
        }, timeoutMs)
      : null;

  try {
    const res = await fetch(`${ensureApiPath(baseUrl)}/health`, {
      signal: controller?.signal,
    });

    if (!res.ok) {
      return {
        reachable: false,
        error: `Health check failed with status ${res.status}`,
      };
    }

    const payload = await res.json().catch(() => ({}));
    return {
      reachable: true,
      status: typeof payload?.status === "string" ? payload.status : "ok",
    };
  } catch (error) {
    return {
      reachable: false,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    if (timeout !== null) {
      clearTimeout(timeout);
    }
  }
}

export async function resolveHealthyApiBaseUrl(
  persistedUrl?: string | null,
  timeoutMs: number = 10000,
): Promise<ApiBaseResolution> {
  const candidates = getApiBaseCandidates(persistedUrl);
  let lastError: string | undefined;

  for (const candidate of candidates) {
    const probe = await probeApiBaseUrl(candidate.url, timeoutMs);
    if (probe.reachable) {
      return {
        baseUrl: candidate.url,
        source: candidate.source,
        reachable: true,
        checkedCandidates: candidates,
        status: probe.status,
      };
    }
    lastError = probe.error;
  }

  const fallback = candidates[0] ?? {
    url: CANONICAL_PRODUCTION_API_BASE,
    source: "canonical" as const,
  };

  return {
    baseUrl: fallback.url,
    source: fallback.source,
    reachable: false,
    checkedCandidates: candidates,
    error: lastError,
  };
}

async function parseError(res: Response): Promise<Error> {
  let detail: string | undefined;

  try {
    const raw = await res.text();
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        detail =
          typeof parsed?.detail === "string"
            ? parsed.detail
            : typeof parsed?.error === "string"
              ? parsed.error
              : undefined;
      } catch {
        detail = raw.trim();
      }
    }
  } catch {
    detail = undefined;
  }

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

      if (!res.ok) {
        throw await parseError(res);
      }

      if (!res.body || typeof res.body.getReader !== "function") {
        const fallback = await sendMessage({
          ...payload,
          settings: {
            ...settings,
            stream: false,
          },
        });
        handlers.onMeta?.({
          evidence: fallback.evidence,
          agents_used: fallback.agents_used,
          confidence: fallback.confidence,
          query_analysis: fallback.query_analysis,
          thinking: fallback.thinking,
          pipeline_trace: fallback.pipeline_trace || null,
        });
        handlers.onReplace?.(fallback.content);
        handlers.onDone();
        return;
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

  async function getRuntimeToolContracts(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/tool-contracts`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeInterfaces(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/interfaces`);
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

  async function createRuntimeTask(payload: RuntimeTaskCreatePayload): Promise<RuntimeTaskResponse> {
    const res = await fetch(`${baseUrl}/runtime/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_id: payload.taskId || "",
        parent_task_id: payload.parentTaskId || "",
        permission_scope: payload.permissionScope,
        metadata: payload.metadata || {},
      }),
    });
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

  async function evaluateRuntimeSafetyOperation(
    payload: RuntimeSafetyEvaluatePayload,
  ): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/safety/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: payload.requestId || "",
        tool_name: payload.toolName,
        command_text: payload.commandText || "",
        metadata: payload.metadata || {},
      }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getRuntimeSafetyAudit(limit: number = 100): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/safety/audit?limit=${limit}`);
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
    options?: {
      sessionId?: string;
      platform?: string;
      forceKeep?: boolean;
    },
  ): Promise<MemoryIngestResponse> {
    const res = await fetch(`${baseUrl}/memories/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content,
        source,
        session_id: options?.sessionId || "",
        platform: options?.platform || "mobile",
        force_keep: options?.forceKeep !== false,
      }),
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

  async function purgeChatQueryMemories(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/memories/purge/chat-queries`, {
      method: "DELETE",
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
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

  async function startAmbientClientSession(body?: {
    platform?: string;
    metadata?: Record<string, unknown>;
  }): Promise<{
    success: boolean;
    session_id: string;
    platform: string;
    metadata: Record<string, unknown>;
    session: AmbientClientSessionInfo;
  }> {
    const res = await fetch(`${baseUrl}/ambient/client/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        platform: body?.platform || "mobile",
        metadata: body?.metadata || {},
      }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function stopAmbientClientSession(body: {
    sessionId: string;
    reason?: string;
  }): Promise<{
    success: boolean;
    session: AmbientClientSessionInfo;
    triggered_agents?: string[];
  }> {
    const res = await fetch(`${baseUrl}/ambient/client/session/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: body.sessionId,
        reason: body.reason || "user_request",
      }),
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getAmbientClientSessions(): Promise<{
    active_session_id: string;
    followup_until: number;
    active_sessions: AmbientClientSessionInfo[];
    sessions: AmbientClientSessionInfo[];
  }> {
    const res = await fetch(`${baseUrl}/ambient/client/sessions`);
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function processAmbientClientAudio(body: {
    sessionId?: string;
    audioBase64: string;
    mimeType: string;
    platform?: string;
    language?: string;
    estimatedDurationS?: number;
    metadata?: Record<string, unknown>;
  }): Promise<AmbientClientAudioResponse> {
    const res = await fetch(`${baseUrl}/ambient/client/process-audio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: body.sessionId || "",
        audio_base64: body.audioBase64,
        mime_type: body.mimeType,
        platform: body.platform || "mobile",
        language: body.language,
        estimated_duration_s: body.estimatedDurationS || 0,
        metadata: body.metadata || {},
      }),
    });
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

  async function enableWakeWord(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/ambient/wake-word/enable`, {
      method: "POST",
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function disableWakeWord(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/ambient/wake-word/disable`, {
      method: "POST",
    });
    if (!res.ok) {
      throw await parseError(res);
    }
    return res.json();
  }

  async function getWakeWordStatus(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/ambient/wake-word/status`);
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

  async function getClaimStats(): Promise<{
    total: number;
    active: number;
    topics: number;
    claims: Array<{
      id: string;
      text: string;
      confidence: number;
      source_ids: string[];
      topic: string;
      created_at: string;
      updated_at: string;
      reinforcement_count?: number;
      contradiction_ids?: string[];
      is_active?: boolean;
    }>;
  }> {
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

  // Backward-compatible aliases used by existing screens.
  const getWikiPages = listWikiPages;
  const rebuildWiki = rebuildWikiFromMemories;

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

  // ── Deep Applications: Session Memory Forge ──────────────────────────────
  async function getSessionForgeStatus(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/deep/session-forge/status`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function triggerCrystallize(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/deep/session-forge/crystallize`, { method: "POST" });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function triggerSummaryForge(payload?: SessionForgeSummaryPayload | string): Promise<Record<string, unknown>> {
    const body =
      typeof payload === "string"
        ? { session_id: payload }
        : {
            session_id: payload?.sessionId,
            window_days: payload?.windowDays,
          };

    const res = await fetch(`${baseUrl}/deep/session-forge/summary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function triggerGapMapper(payload?: SessionForgeRunPayload | string): Promise<Record<string, unknown>> {
    const body =
      typeof payload === "string"
        ? { session_id: payload }
        : {
            session_id: payload?.sessionId,
            lookback_days: payload?.lookbackDays,
          };

    const res = await fetch(`${baseUrl}/deep/session-forge/gaps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function triggerBeliefDetector(payload?: SessionForgeRunPayload | string): Promise<Record<string, unknown>> {
    const body =
      typeof payload === "string"
        ? { session_id: payload }
        : {
            session_id: payload?.sessionId,
            lookback_days: payload?.lookbackDays,
          };

    const res = await fetch(`${baseUrl}/deep/session-forge/beliefs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function getForgeArtifacts(artifactType: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/deep/session-forge/artifacts/${encodeURIComponent(artifactType)}`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  // ── Deep Applications: Life Chronicle ──────────────────────────────────
  async function enableChroniclePassive(
    consent: boolean = true,
    consentActor: string = "mobile-user",
  ): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/chronicle/passive/enable`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ consent, consent_actor: consentActor }),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function disableChroniclePassive(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/chronicle/passive/disable`, { method: "POST" });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function getChroniclePassiveStatus(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/chronicle/passive/status`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function addChronicleObservation(
    observation: string | ChronicleObservationPayload,
  ): Promise<Record<string, unknown>> {
    const body =
      typeof observation === "string"
        ? { note: observation }
        : {
            note: observation.note || "",
            location: observation.location || {},
            people_present: observation.peoplePresent || [],
            tags: observation.tags || [],
            media_ref: observation.mediaRef || "",
            source: observation.source || "passive_notification",
            emotion_hint: observation.emotionHint || "",
          };

    const res = await fetch(`${baseUrl}/chronicle/passive/observe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function saveChronicleWindow(payload?: ChronicleSavePayload): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/chronicle/passive/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: payload?.title || "",
        window_seconds: payload?.windowSeconds,
        retrieval_hint: payload?.retrievalHint || "",
        life_domain: payload?.lifeDomain || "everyday",
      }),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function listChronicleMoments(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/chronicle/moments`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function getChronicleMoment(memoryId: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/chronicle/moments/${encodeURIComponent(memoryId)}`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  // ── Scheduler ──────────────────────────────────────────────────────────
  async function getSchedulerStatus(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/agent/scheduler/status`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function enableScheduledAgent(agentId: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/agent/scheduler/${encodeURIComponent(agentId)}/enable`, { method: "POST" });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function disableScheduledAgent(agentId: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/agent/scheduler/${encodeURIComponent(agentId)}/disable`, { method: "POST" });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  // ── Cache Stats ────────────────────────────────────────────────────────
  async function getCacheStats(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/agent/cache/stats`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  // ── Memory Operations ──────────────────────────────────────────────────
  async function consolidateMemories(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/memories/consolidate`, { method: "POST" });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function createMemoryExtractionJob(payload?: {
    limit?: number;
    offset?: number;
    dryRun?: boolean;
  }): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/memory-extraction/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        limit: payload?.limit ?? 500,
        offset: payload?.offset ?? 0,
        dry_run: payload?.dryRun ?? true,
      }),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function listMemoryExtractionJobs(limit: number = 20): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/memory-extraction/jobs?limit=${limit}`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function getMemoryExtractionJob(jobId: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/memory-extraction/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function evaluateMemoryQuality(payload?: {
    queries?: string[];
    topK?: number;
  }): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/memory-quality/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        queries: payload?.queries || [],
        top_k: payload?.topK ?? 5,
      }),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function getMemoryQualityHistory(limit: number = 50): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/runtime/memory-quality/history?limit=${limit}`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  // ── Feedback ───────────────────────────────────────────────────────────
  async function submitFeedback(feedback: FeedbackPayload): Promise<Record<string, unknown>> {
    const context = feedback.context || {};
    const query = feedback.query || String(context.query || "");
    const answer = feedback.answer || String(context.answer || "");
    const rating =
      typeof feedback.rating === "number"
        ? feedback.rating
        : Number(context.rating || 0);
    const comment =
      feedback.comment || feedback.content || String(context.comment || "");

    const res = await fetch(`${baseUrl}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        answer,
        rating,
        comment,
        session_id: feedback.session_id || String(context.session_id || ""),
      }),
    });
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  async function getFeedbackStats(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/feedback/stats`);
    if (!res.ok) throw await parseError(res);
    return res.json();
  }

  // ── System ─────────────────────────────────────────────────────────────
  async function getGPUInfo(): Promise<Record<string, unknown>> {
    const res = await fetch(`${baseUrl}/system/gpu`);
    if (!res.ok) throw await parseError(res);
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
    getRuntimeToolContracts,
    getRuntimeInterfaces,
    getRuntimeSafetyPermissions,
    getRuntimeSafetyExecutorStatus,
    getRuntimeSafetyAudit,
    evaluateRuntimeSafetyOperation,
    resolveRuntimeSafetyPermission,
    getRuntimeTasks,
    createRuntimeTask,
    getRuntimeTask,
    cancelRuntimeTask,
    subscribeRuntimeTaskEvents,
    getModelpackManifest,
    verifyModelpack,
    getMemories,
    searchMemories,
    ingestMemory,
    deleteMemory,
    purgeChatQueryMemories,
    consolidateMemories,
    createMemoryExtractionJob,
    listMemoryExtractionJobs,
    getMemoryExtractionJob,
    evaluateMemoryQuality,
    getMemoryQualityHistory,
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
    startAmbientClientSession,
    stopAmbientClientSession,
    getAmbientClientSessions,
    processAmbientClientAudio,
    injectAmbientLiveAudio,
    getAmbientConfig,
    updateAmbientConfig,
    getVoiceProviders,
    enableWakeWord,
    disableWakeWord,
    getWakeWordStatus,
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
    getSchedulerStatus,
    enableScheduledAgent,
    disableScheduledAgent,
    getCacheStats,
    getWikiPages,
    listWikiPages,
    getWikiPage,
    searchWiki,
    getClaimStats,
    searchClaims,
    rebuildWiki,
    rebuildWikiFromMemories,
    getWikiLintLatest,
    runWikiLint,
    getWikiCompactionLatest,
    runWikiCompaction,
    // Deep Applications
    getSessionForgeStatus,
    triggerCrystallize,
    triggerSummaryForge,
    triggerGapMapper,
    triggerBeliefDetector,
    getForgeArtifacts,
    enableChroniclePassive,
    disableChroniclePassive,
    getChroniclePassiveStatus,
    addChronicleObservation,
    saveChronicleWindow,
    listChronicleMoments,
    getChronicleMoment,
    // System / Feedback
    submitFeedback,
    getFeedbackStats,
    getGPUInfo,
  };
}

export type ApiClient = ReturnType<typeof createApiClient>;

export function getDefaultApiBaseUrl(): string {
  return getApiBaseCandidates()[0]?.url || CANONICAL_PRODUCTION_API_BASE;
}
