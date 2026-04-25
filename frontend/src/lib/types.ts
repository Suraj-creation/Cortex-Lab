// ── Shared types ────────────────────────────────────────────────

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
  isRAG?: boolean;
  // RAG-enhanced fields
  evidence?: EvidenceCard[];
  agentsUsed?: string[];
  confidence?: number;
  queryAnalysis?: QueryAnalysis;
  processingTimeMs?: number;
  cacheHit?: boolean;
  // Pipeline observability
  pipelineTrace?: PipelineTrace;
  runtimeTasks?: RuntimeTaskReferences;
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

// ── Pipeline Observability Types ────────────────────────────────

export interface LivePipelineEvent {
  event_type: "pipeline_start" | "step_start" | "step_complete" | "step_skip" | "pipeline_complete" | "metric";
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
  coordinator_task_id?: string;
  coordinator_plan?: Record<string, unknown>;
  subagent_spawn_records?: Array<{
    parent_task_id: string;
    task_id: string;
    agent: string;
    role: string;
    spawned_at: string;
  }>;
  sidechain_transcript?: Array<{
    event: string;
    agent?: string;
    task_id?: string;
    trace_id?: string;
    timestamp: string;
    confidence?: number;
    answer_preview?: string;
    error?: string;
  }>;
  generation_details: Record<string, unknown>;
  cache_status: { hit: boolean; level: string | null };
  final_confidence: number;
  evidence_count: number;
  token_usage: Record<string, number>;
  runtime_loop_state?: {
    request_id?: string;
    session_id?: string;
    started_at?: string;
    budget?: {
      max_iterations: number;
      max_tool_calls_per_window: number;
      window_seconds: number;
      max_input_tokens: number;
      max_output_tokens: number;
      max_wall_time_seconds: number;
    };
    iterations_executed: number;
    tool_calls_executed: number;
    stop_reason?: string | null;
    stop_note?: string;
  };
  stop_reason?: string;
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

export type InferenceMode = "cloud" | "hybrid" | "local_offline";
export type LLMProviderType = "local" | "gemini" | "gemma_local";

export interface ChatSettings {
  temperature: number;
  topP: number;
  maxTokens: number;
  stream: boolean;
  useRAG: boolean;
  llmProvider: LLMProviderType;
  inferenceMode?: InferenceMode;
  allowCloudFallback?: boolean;
  thinkingMode: boolean;
}

export const DEFAULT_SETTINGS: ChatSettings = {
  temperature: 0.6,
  topP: 0.95,
  maxTokens: 4096,
  stream: true,
  useRAG: true,
  llmProvider: "local",
  inferenceMode: "cloud",
  allowCloudFallback: true,
  thinkingMode: true,
};

export interface RuntimeSelection {
  mode: InferenceMode;
  llmProvider: LLMProviderType;
  sttProvider: VoiceProviderType;
  ttsProvider: VoiceProviderType;
  allowCloudFallback: boolean;
}

export type ModelpackAvailability = "available" | "coming_soon";

export interface ModelpackFileEntry {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface ModelpackEntry {
  id: string;
  display_name: string;
  version: string;
  target?: string;
  family?: string;
  quantization?: string;
  summary?: string;
  availability?: ModelpackAvailability;
  download_url?: string;
  cta_label?: string;
  docs_url?: string;
  requires?: string[];
  files: ModelpackFileEntry[];
}

export interface ModelpackManifest {
  schema_version: string;
  generated_at: string;
  signature_required: boolean;
  source?: string;
  docs_url?: string;
  channels?: string[];
  packs: ModelpackEntry[];
}

// ── Ambient Voice Types ─────────────────────────────────────────

export type AmbientStatusType =
  | "idle"
  | "loading"
  | "listening"
  | "speech_detected"
  | "transcribing"
  | "paused"
  | "error";

export type VoiceProviderType = "traditional" | "gemini" | "local";

export type AmbientLiveSessionState =
  | "idle_listening"
  | "user_detected"
  | "live_streaming"
  | "assistant_responding"
  | "background_processing"
  | "degraded";

export interface AmbientLiveStatus {
  enabled: boolean;
  running: boolean;
  paused?: boolean;
  state: AmbientLiveSessionState;
  session_id?: string | null;
  uptime_seconds?: number;
  native_live_connected?: boolean;
  native_live_error?: string | null;
  energy_threshold?: number;
  segments_detected?: number;
  user_turns?: number;
  assistant_turns?: number;
  memory_jobs?: number;
  audio_frames?: number;
  last_audio_level?: number;
  last_error?: string | null;
}

export interface AmbientClientSessionInfo {
  session_id: string;
  mode: string;
  start_time: string;
  end_time?: string | null;
  user_detected?: boolean;
  metadata: Record<string, unknown>;
  retention_summary: {
    discarded: number;
    session_only: number;
    structured: number;
    priority: number;
  };
  agent_tags: string[];
}

export interface AmbientRetentionTrace {
  decision?: string;
  memory_decision?: string;
  archive_policy?: string;
  reason?: string;
  score?: number;
  tags?: string[];
  direct_address?: boolean;
  retrieval_intent?: boolean;
  reply_expected?: boolean;
  source?: string;
  platform?: string;
  session_id?: string;
}

export interface VoiceProviders {
  stt_provider: VoiceProviderType;
  tts_provider: VoiceProviderType;
  gemini_available: boolean;
  traditional_stt_available: boolean;
  traditional_tts_available: boolean;
  local_stt_available?: boolean;
  local_tts_available?: boolean;
  gemini_stt_available: boolean;
  gemini_tts_available: boolean;
  supported_stt_providers?: VoiceProviderType[];
  supported_tts_providers?: VoiceProviderType[];
  gemini_tts_voices: string[];
  live_mode?: "classic" | "gemini_live";
  live_status?: AmbientLiveStatus;
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
  operating_mode?: "classic" | "gemini_live";
  no_local_model_policy_enforced?: boolean;
  local_models_initialized?: {
    vad: boolean;
    traditional_stt: boolean;
    traditional_tts: boolean;
  };
  live?: AmbientLiveStatus;
  client_session?: {
    active_session_id: string;
    followup_until: number;
    active_sessions: AmbientClientSessionInfo[];
    sessions: AmbientClientSessionInfo[];
  };
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
  live_mode?: "classic" | "gemini_live";
  energy_gate_threshold?: number;
  energy_min_speech_ms?: number;
  energy_silence_ms?: number;
  assistant_name?: string;
  assistant_aliases?: string[];
  companion_followup_window_s?: number;
}

export interface ConversationTurn {
  speaker_label: string;
  speaker_name: string;
  text: string;
  timestamp: number;
  confidence: number;
  speaker_confidence?: number;
  live_turn_id?: string;
  session_id?: string;
  source_platform?: string;
  retention_trace?: AmbientRetentionTrace;
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

export interface AmbientClientAudioResponse {
  success: boolean;
  session_id: string;
  transcript: string;
  analysis: {
    direct_address: boolean;
    retrieval_intent: boolean;
    reply_expected: boolean;
    followup_active?: boolean;
    assistant_alias?: string;
    query_text?: string;
  };
  retention_trace: {
    decision: string;
    memory_decision: string;
    archive_policy?: string;
    reason?: string;
    score?: number;
    tags: string[];
    direct_address?: boolean;
    retrieval_intent?: boolean;
    reply_expected?: boolean;
    source?: string;
    platform?: string;
    session_id?: string;
  };
  assistant_text: string;
  assistant_audio_base64: string;
  assistant_name?: string;
  stt_confidence?: number;
  language?: string;
  session?: AmbientClientSessionInfo | null;
}

// ── Observability Analytics Types ───────────────────────────────

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
  stop_reason_distribution: Record<string, number>;
  runtime_loop: {
    avg_iterations: number;
    avg_tool_calls: number;
  };
}

export interface TracesResponse {
  traces: PipelineTrace[];
  analytics: TraceAnalytics;
}

export type RuntimePermissionStatus = "pending" | "approved" | "denied" | "expired";

export type RuntimeTaskState =
  | "queued"
  | "running"
  | "waiting_approval"
  | "blocked"
  | "completed"
  | "failed"
  | "cancelled";

export interface RuntimeTaskSnapshot {
  task_id: string;
  parent_task_id: string | null;
  state: RuntimeTaskState;
  permission_scope: string[] | null;
  child_task_ids: string[];
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface RuntimeTaskListResponse {
  count: number;
  tasks: RuntimeTaskSnapshot[];
}

export interface RuntimeTaskReferences {
  trace_id?: string;
  coordinator_task_id?: string;
  subagent_task_ids: string[];
  all_task_ids: string[];
  api: {
    list: string;
    coordinator?: string;
    cancel_coordinator?: string;
    subagents: Array<{
      task_id: string;
      get: string;
      cancel: string;
    }>;
  };
}

export type RuntimeTaskEventType =
  | "task_created"
  | "task_transition"
  | "task_attached";

export interface RuntimeTaskEvent {
  event_id: string;
  sequence: number;
  event_type: RuntimeTaskEventType;
  timestamp: string;
  task: RuntimeTaskSnapshot;
  previous_state?: RuntimeTaskState | null;
  state: RuntimeTaskState;
  note?: string;
}

export interface RuntimePermissionRequest {
  permission_id: string;
  request_id: string;
  tool_name: string;
  command_text: string;
  reason: string;
  status: RuntimePermissionStatus;
  created_at: string;
  expires_at: string;
  decided_at: string | null;
  decided_by: string;
  decision_note: string;
  metadata: Record<string, unknown>;
}

export interface RuntimeApprovalSummary {
  pending: number;
  expired: number;
  approved_total: number;
  running: number;
  waiting_retry: number;
  failed: number;
  completed: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Autonomous Agent Runtime Types (Pi-Mono Integration)
// ═══════════════════════════════════════════════════════════════════════════════

export type CortexEventType =
  | "agent_start"
  | "agent_end"
  | "turn_start"
  | "turn_end"
  | "message_start"
  | "message_update"
  | "message_end"
  | "tool_execution_start"
  | "tool_execution_update"
  | "tool_execution_end"
  | "queue_update"
  | "compaction_start"
  | "compaction_end"
  | "auto_retry_start"
  | "auto_retry_end"
  | "tier_selected"
  | "retrieval_channel_complete"
  | "evidence_ready"
  | "quality_loop"
  | "wiki_update"
  | "belief_shift"
  | "gap_signal"
  | "presence_initiative"
  | "keepalive";

export interface CortexEvent {
  event_id?: string;
  type: CortexEventType;
  data: Record<string, unknown>;
  timestamp: string;
  session_id: string;
  agent_id: string;
  trace_id: string;
}

export interface TierClassification {
  tier: "T0" | "T1" | "T2" | "T3" | "T4";
  complexity: number;
  intent: string;
  entities: string[];
  topics: string[];
  sub_queries: string[];
  confidence: number;
  cache_key: string;
  recommended_agents: string[];
  estimated_latency_ms: number;
}

export interface AgentSessionInfo {
  session_id: string;
  agent_id: string;
  is_running: boolean;
  is_streaming: boolean;
  message_count: number;
  turn_count: number;
  steering: {
    steering: string[];
    followUp: string[];
  };
}

export interface AgentQueryResponse {
  answer: string;
  tier: TierClassification;
  turns: number;
  session_id: string | null;
  tool_results: Array<{
    tool_call_id: string;
    content: string;
    is_error: boolean;
  }>;
}

export interface AgentConfigInfo {
  agent_id: string;
  tool_count: number;
  max_turns: number;
  scheduling: {
    always_on: boolean;
    continuous: boolean;
    on_ingest: boolean;
    interval_min: number;
  } | null;
}

export interface WikiPageInfo {
  id: string;
  title: string;
  content: string;
  topics: string[];
  claim_ids: string[];
  created_at: string;
  updated_at: string;
  version: number;
}

export interface ToolExecutionEvent {
  toolCallId: string;
  toolName: string;
  occurrence?: number;
  args?: Record<string, unknown>;
  result?: string;
  isError?: boolean;
}

export interface AgentTurnInfo {
  turn: number;
  has_tool_calls: boolean;
  timestamp: string;
}
