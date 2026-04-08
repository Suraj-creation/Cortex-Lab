import type {
  InferenceAdapter,
  InferenceCapabilities,
  InferenceRequest,
  InferenceResponse,
  RuntimeSelection,
} from "../types";

function normalizeBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;
}

export class CloudAdapter implements InferenceAdapter {
  readonly id = "cloud";
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
  }

  canHandle(_selection: RuntimeSelection, caps: InferenceCapabilities): boolean {
    return caps.cloudAvailable;
  }

  async run(request: InferenceRequest, selection: RuntimeSelection): Promise<InferenceResponse> {
    const endpoint = request.settings.useRAG ? "/rag/chat" : "/chat";
    const res = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: request.messages,
        temperature: request.settings.temperature,
        top_p: request.settings.topP,
        max_tokens: request.settings.maxTokens,
        stream: false,
        use_rag: request.settings.useRAG,
        session_id: request.sessionId || "",
        llm_provider: selection.llmProvider,
        inference_mode: selection.mode,
        allow_cloud_fallback: selection.allowCloudFallback,
      }),
    });

    if (!res.ok) {
      const detail = await res
        .json()
        .then((data) => data?.detail as string | undefined)
        .catch(() => undefined);
      throw new Error(detail || `Cloud inference failed (${res.status})`);
    }

    const data = await res.json();
    return {
      content: data.content || data.answer || "",
      thinking: data.thinking,
      usage: data.usage,
      provider: selection.llmProvider,
      mode: selection.mode,
      traceId: data.pipeline_trace?.trace_id,
    };
  }
}
