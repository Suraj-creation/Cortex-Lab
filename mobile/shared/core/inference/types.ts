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
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  provider: LLMProviderType;
  mode: InferenceMode;
  traceId?: string;
}

export interface InferenceCapabilities {
  cloudAvailable: boolean;
  localLlmReady: boolean;
  localSttReady: boolean;
  localTtsReady: boolean;
}

export interface InferenceAdapter {
  readonly id: string;
  canHandle(selection: RuntimeSelection, caps: InferenceCapabilities): boolean;
  run(request: InferenceRequest, selection: RuntimeSelection): Promise<InferenceResponse>;
}
