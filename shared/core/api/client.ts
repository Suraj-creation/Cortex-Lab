import {
  AmbientConfig,
  AmbientState,
  ChatSettings,
  ConversationRecord,
  ConversationTurn,
  DEFAULT_SETTINGS,
  EvidenceCard,
  GraphData,
  LivePipelineEvent,
  MemoryObject,
  ModelStatus,
  PipelineTrace,
  QueryAnalysis,
  RAGStats,
  TracesResponse,
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
  provider: "local" | "gemini";
  available: string[];
  gemini_configured: boolean;
  local_model_loaded: boolean;
}

export interface TTSStatus {
  available: boolean;
  voice: string | null;
  total_syntheses?: number;
}

export interface LiveTranscriptResponse {
  turns: ConversationTurn[];
}

function normalizeBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;
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

export function createApiClient(config: ApiConfig) {
  const baseUrl = normalizeBaseUrl(config.baseUrl);

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
    provider: "local" | "gemini",
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

          for (const line of lines) {
            if (!line.startsWith("data: ")) {
              continue;
            }
            try {
              const payload = JSON.parse(line.slice(6)) as LivePipelineEvent;
              onEvent(payload);
            } catch {
              // Skip malformed payloads.
            }
          }
        }
      } catch (error) {
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
    getAmbientConfig,
    updateAmbientConfig,
    getVoiceProviders,
    setSTTProvider,
    setTTSProvider,
    getConversations,
    getConversation,
    getLiveTranscript,
    getEnrollmentStatus,
    startEnrollment,
    getTTSStatus,
    synthesizeSpeech,
    voiceQuery,
  };
}

export function getDefaultApiBaseUrl(): string {
  const envBase =
    typeof process !== "undefined"
      ? process.env.EXPO_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL
      : undefined;

  const normalizedEnvBase = envBase?.trim() ? normalizeBaseUrl(envBase.trim()) : undefined;

  // In web builds, prefer same-host backend because 10.0.2.2 is Android-emulator only.
  if (typeof window !== "undefined" && window.location?.hostname) {
    if (normalizedEnvBase && !isAndroidEmulatorHost(normalizedEnvBase)) {
      return normalizedEnvBase;
    }

    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    const host = window.location.hostname;
    return normalizeBaseUrl(`${protocol}//${host}:8000/api`);
  }

  if (normalizedEnvBase) {
    return normalizedEnvBase;
  }

  // Android emulator: use 10.0.2.2. iOS simulator can use localhost.
  // For physical devices, set EXPO_PUBLIC_API_BASE_URL to your machine LAN IP.
  return "http://10.0.2.2:8000/api";
}
