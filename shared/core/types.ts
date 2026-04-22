// Shared cross-platform domain types for Cortex Lab clients.

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  timestamp: number;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  isStreaming?: boolean;
  evidence?: EvidenceCard[];
  agentsUsed?: string[];
  confidence?: number;
  queryAnalysis?: QueryAnalysis;
  processingTimeMs?: number;
  cacheHit?: boolean;
  pipelineTrace?: PipelineTrace;
}

export interface EvidenceCard {
  content: string;
  score: number;
  channel: string;
  timestamp: string;
  memory_type: string;
  emotion: string;
  entities: string[];
}

export interface QueryAnalysis {
  intent: string;
  complexity: number;
  routing: string;
}

export interface LivePipelineEvent {
  event_type:
    | "pipeline_start"
    | "step_start"
    | "step_complete"
    | "step_skip"
    | "pipeline_complete"
    | "metric";
  step_name: string;
  step_type: string;
  status: "running" | "completed" | "skipped" | "error";
  duration_ms: number;
  details: Record<string, unknown>;
  timestamp: number;
  trace_id: string;
}

export interface PipelineStep {
  step_name: string;
  step_type: string;
  status: "completed" | "skipped" | "error" | "pending" | "running";
  duration_ms: number;
  details: Record<string, unknown>;
  sub_steps?: PipelineStep[];
}

export interface RetrievalChannelTrace {
  channel: string;
  result_count: number;
  top_score: number;
  avg_score: number;
  duration_ms: number;
}

export interface CRAGEvaluation {
  quality_score: number;
  verdict: string;
  avg_evidence_score: number;
  max_evidence_score: number;
  evidence_count: number;
  entity_coverage: number;
  supplementary_retrieved: number;
}

export interface SelfRAGCritique {
  isrel: number;
  issup: number;
  isuse: number;
  avg_score: number;
  verdict: string;
  revision_applied: boolean;
  revision_focus: string;
}

export interface FLARETrace {
  triggered: boolean;
  uncertain_sentences: number;
  retrieval_iterations: number;
  new_evidence_count: number;
  answer_revised: boolean;
  confidence_delta: number;
}

export interface QueryTransformTrace {
  original_query: string;
  multi_queries: string[];
  hyde_answer: string;
  step_back_query: string;
  sub_queries: string[];
  total_variants: number;
  duration_ms: number;
}

export interface PipelineTrace {
  trace_id: string;
  timestamp: string;
  query: string;
  total_duration_ms: number;
  steps: PipelineStep[];
  query_analysis: {
    intent: string;
    complexity: number;
    routing: string;
    entities?: string[];
    topics?: string[];
    time_start?: string | null;
    time_end?: string | null;
  };
  query_transform?: QueryTransformTrace | null;
  retrieval_channels: RetrievalChannelTrace[];
  reranking: {
    method: string;
    duration_ms: number;
    input_count?: number;
  };
  crag_evaluation?: CRAGEvaluation | null;
  self_rag_critique?: SelfRAGCritique | null;
  flare_trace?: FLARETrace | null;
  routing_decision: string;
  agents_invoked: { agent: string; is_primary: boolean }[];
  generation_details: Record<string, unknown>;
  cache_status: { hit: boolean; level: string | null };
  final_confidence: number;
  evidence_count: number;
  token_usage: Record<string, number>;
}

export interface MemoryObject {
  id: string;
  content: string;
  memory_type: string;
  timestamp: string;
  emotion: string;
  emotion_confidence: number;
  importance: number;
  topics: string[];
  entities: string[];
  propositions: string[];
  source: string;
  score?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  memory_count: number;
  mentions?: number;
  firstSeen?: string;
  lastSeen?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface RAGStats {
  status: string;
  memories: {
    memories: number;
    entities: number;
    edges: number;
    belief_deltas: number;
    conversations: number;
    backend: string;
  };
  vectors: {
    total_vectors: number;
    hot_count: number;
    warm_count: number;
    cold_count: number;
    using_faiss: boolean;
    dimension: number;
  };
  graph: {
    nodes: number;
    edges: number;
    density?: number;
  };
  cache: {
    exact_hits: number;
    exact_misses: number;
    semantic_hits: number;
    semantic_misses: number;
    embedding_hits: number;
    embedding_misses: number;
    total_queries: number;
    total_hits: number;
    hit_rate: number;
    exact_cache_size: number;
    semantic_cache_size: number;
    embedding_cache_size: number;
  };
  llm: {
    call_count: number;
    total_tokens: number;
    model_loaded: boolean;
  };
}

export interface ModelStatus {
  status: string;
  model_loaded: boolean;
  model_info: {
    name?: string;
    parameters?: string;
    quantization?: string;
    device?: string;
    gpu_memory?: string;
    max_context?: number;
    load_time_seconds?: number;
    fine_tuned?: boolean;
    training_stages_completed?: number;
    base_model?: string;
    llm_provider?: string;
    gemini_available?: boolean;
  };
}

export interface ChatSettings {
  temperature: number;
  topP: number;
  maxTokens: number;
  stream: boolean;
  useRAG: boolean;
  llmProvider: "local" | "gemini";
}

export const DEFAULT_SETTINGS: ChatSettings = {
  temperature: 0.6,
  topP: 0.95,
  maxTokens: 4096,
  stream: true,
  useRAG: true,
  llmProvider: "local",
};

export type AmbientStatusType =
  | "idle"
  | "loading"
  | "listening"
  | "speech_detected"
  | "transcribing"
  | "paused"
  | "error";

export type VoiceProviderType = "traditional" | "gemini";

export interface VoiceProviders {
  stt_provider: VoiceProviderType;
  tts_provider: VoiceProviderType;
  gemini_available: boolean;
  traditional_stt_available: boolean;
  traditional_tts_available: boolean;
  gemini_stt_available: boolean;
  gemini_tts_available: boolean;
  gemini_tts_voices: string[];
}

export interface AmbientState {
  status: AmbientStatusType;
  uptime_seconds: number;
  error: string | null;
  enrolled: boolean;
  tts_available: boolean;
  audio_level: number;
  speech_segments: number;
  transcriptions: number;
  stt_provider: VoiceProviderType;
  tts_provider: VoiceProviderType;
  gemini_available: boolean;
  vad?: {
    threshold: number;
    speech_active: boolean;
    total_segments: number;
    total_speech_seconds: number;
  };
  speaker_id?: {
    enrolled: boolean;
    active_clusters: number;
    cluster_labels: string[];
    aliases: Record<string, string>;
  };
  transcriber?: {
    model_size: string;
    device: string;
    total_transcriptions: number;
    total_audio_seconds: number;
    real_time_factor: number;
  };
  conversation?: {
    total_conversations: number;
    current_turns: number;
    total_ingested: number;
  };
  tts?: {
    available: boolean;
    voice: string;
    total_syntheses: number;
  };
}

export interface AmbientConfig {
  enabled: boolean;
  vad_threshold: number;
  auto_ingest: boolean;
  silence_timeout_s: number;
  min_speech_ms: number;
  stt_provider: VoiceProviderType;
  tts_provider: VoiceProviderType;
  tts_enabled: boolean;
  tts_voice: string;
  tts_speed: number;
  whisper_model_size: string;
  whisper_device: string;
  whisper_language: string | null;
  record_raw_audio: boolean;
  gemini_tts_voice: string;
}

export interface ConversationTurn {
  speaker_label: string;
  speaker_name: string;
  text: string;
  timestamp: number;
  confidence: number;
}

export interface ConversationRecord {
  id: string;
  turns: ConversationTurn[];
  participants: string[];
  start_time: string | null;
  end_time: string | null;
  duration_seconds: number;
  memory_ids: string[];
  auto_ingested: boolean;
}

export interface VoiceQueryResult {
  transcript: string;
  answer: string;
  evidence?: EvidenceCard[];
  audio_base64: string | null;
  language: string;
  stt_confidence: number;
  stt_provider?: VoiceProviderType;
  tts_provider?: VoiceProviderType;
}

export interface ChannelUsageStat {
  total_results: number;
  total_duration_ms: number;
  usage_count: number;
}

export interface StepStat {
  completed: number;
  skipped: number;
  total_duration_ms: number;
}

export interface TraceAnalytics {
  total_traces: number;
  showing: number;
  avg_duration_ms: number;
  avg_confidence: number;
  avg_evidence_count: number;
  channel_usage: Record<string, ChannelUsageStat>;
  step_stats: Record<string, StepStat>;
  crag_activation_rate: number;
  selfrag_activation_rate: number;
  flare_activation_rate: number;
  cache_hit_rate: number;
}

export interface TracesResponse {
  traces: PipelineTrace[];
  analytics: TraceAnalytics;
}
