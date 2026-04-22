import type {
  ChatSettings,
  InferenceMode,
  LLMProviderType,
  RuntimeSelection,
  VoiceProviderType,
} from "../types";

export type { InferenceMode, LLMProviderType, RuntimeSelection, VoiceProviderType };

export interface InferenceRequest {
  messages: { role: string; content: string }[];
  settings: ChatSettings;
  sessionId?: string;
}

export interface InferenceResponse {
  content: string;
  thinking?: string;
  provider: LLMProviderType;
  mode: InferenceMode;
  traceId?: string;
}

export interface RuntimeCapabilities {
  cloudAvailable: boolean;
  localLlmReady: boolean;
  localVoiceReady: boolean;
}

export interface InferenceAdapter {
  readonly id: string;
  canHandle(selection: RuntimeSelection, caps: RuntimeCapabilities): boolean;
  run(request: InferenceRequest, selection: RuntimeSelection): Promise<InferenceResponse>;
}
