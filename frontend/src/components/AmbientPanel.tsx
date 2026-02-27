"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Mic,
  Play,
  Pause,
  Square,
  Settings2,
  ArrowLeft,
  Activity,
  Users,
  MessageSquare,
  Volume2,
  Fingerprint,
  Radio,
  Save,
  RotateCcw,
} from "lucide-react";
import {
  startAmbient,
  stopAmbient,
  pauseAmbient,
  resumeAmbient,
  getAmbientStatus,
  getConversations,
  getAmbientConfig,
  updateAmbientConfig,
} from "@/lib/api";
import { AmbientState, AmbientStatusType, AmbientConfig, ConversationRecord } from "@/lib/types";
import { LiveTranscript } from "./LiveTranscript";
import { ConversationHistory } from "./ConversationHistory";
import { VoiceEnrollment } from "./VoiceEnrollment";

interface Props {
  onBack: () => void;
}

type AmbientTab = "live" | "conversations" | "enrollment" | "settings";

const STATUS_COLORS: Record<AmbientStatusType, string> = {
  idle: "bg-slate-400",
  loading: "bg-amber-400 animate-pulse",
  listening: "bg-emerald-500 animate-glow-pulse",
  speech_detected: "bg-indigo-500 animate-pulse",
  transcribing: "bg-violet-500 animate-pulse",
  paused: "bg-amber-500",
  error: "bg-red-500",
};

const STATUS_LABELS: Record<AmbientStatusType, string> = {
  idle: "Idle",
  loading: "Loading Models...",
  listening: "Listening",
  speech_detected: "Speech Detected",
  transcribing: "Transcribing...",
  paused: "Paused",
  error: "Error",
};

export function AmbientPanel({ onBack }: Props) {
  const [status, setStatus] = useState<AmbientState | null>(null);
  const [activeTab, setActiveTab] = useState<AmbientTab>("live");
  const [conversations, setConversations] = useState<ConversationRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll ambient status
  const fetchStatus = useCallback(async () => {
    try {
      const data = await getAmbientStatus();
      setStatus(data);
      setError(null);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to fetch status";
      // Don't show 503 as errors — ambient service just not initialized yet
      if (!msg.includes("503")) setError(msg);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Load conversations when tab is active
  useEffect(() => {
    if (activeTab === "conversations") {
      getConversations(50, 0)
        .then((data) => setConversations(data.conversations))
        .catch(() => {});
    }
  }, [activeTab]);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await startAmbient();
      if (!result.success) setError(result.error || "Failed to start");
      await fetchStatus();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Start failed");
    }
    setLoading(false);
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopAmbient();
      await fetchStatus();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Stop failed");
    }
    setLoading(false);
  };

  const handlePauseResume = async () => {
    setLoading(true);
    try {
      if (status?.status === "paused") {
        await resumeAmbient();
      } else {
        await pauseAmbient();
      }
      await fetchStatus();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Action failed");
    }
    setLoading(false);
  };

  const isActive =
    status?.status === "listening" ||
    status?.status === "speech_detected" ||
    status?.status === "transcribing";
  const isPaused = status?.status === "paused";

  const tabs: { id: AmbientTab; label: string; icon: typeof Mic }[] = [
    { id: "live", label: "Live", icon: Radio },
    { id: "conversations", label: "Conversations", icon: MessageSquare },
    { id: "enrollment", label: "Voice ID", icon: Fingerprint },
    { id: "settings", label: "Settings", icon: Settings2 },
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b border-slate-200 bg-white/80 backdrop-blur-sm px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
            >
              <ArrowLeft size={18} />
            </button>
            <div className="flex items-center gap-2.5">
              <div className="h-8 w-8 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
                <Mic size={16} className="text-white" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-800">
                  Ambient Listening
                </h2>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div
                    className={`h-2 w-2 rounded-full ${
                      STATUS_COLORS[status?.status || "idle"]
                    }`}
                  />
                  <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">
                    {STATUS_LABELS[status?.status || "idle"]}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center gap-2">
            {(isActive || isPaused) && (
              <button
                onClick={handlePauseResume}
                disabled={loading}
                className="flex items-center gap-1.5 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-700 hover:bg-amber-100 transition-all disabled:opacity-50"
              >
                {isPaused ? <Play size={14} /> : <Pause size={14} />}
                {isPaused ? "Resume" : "Pause"}
              </button>
            )}
            {isActive || isPaused ? (
              <button
                onClick={handleStop}
                disabled={loading}
                className="flex items-center gap-1.5 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 hover:bg-red-100 transition-all disabled:opacity-50"
              >
                <Square size={14} />
                Stop
              </button>
            ) : (
              <button
                onClick={handleStart}
                disabled={loading || status?.status === "loading"}
                className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-2 text-xs font-medium text-white hover:from-indigo-600 hover:to-violet-600 transition-all disabled:opacity-50 shadow-sm"
              >
                <Mic size={14} />
                {loading || status?.status === "loading"
                  ? "Starting..."
                  : "Start Listening"}
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        {/* Stats Bar */}
        {status && status.status !== "idle" && (
          <div className="flex items-center gap-6 mt-3 text-xs text-slate-500">
            <div className="flex items-center gap-1.5">
              <Activity size={12} className="text-emerald-500" />
              <span>{status.vad?.total_segments || 0} segments</span>
            </div>
            <div className="flex items-center gap-1.5">
              <MessageSquare size={12} className="text-indigo-500" />
              <span>{status.transcriptions || 0} transcribed</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Users size={12} className="text-violet-500" />
              <span>{status.speaker_id?.active_clusters || 0} speakers</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Volume2 size={12} className="text-amber-500" />
              <span>{Math.round(status.audio_level || 0)} dB</span>
            </div>
            {status.uptime_seconds > 0 && (
              <span className="text-slate-400">
                Uptime: {formatUptime(status.uptime_seconds)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-slate-200 bg-white/60 px-6">
        <div className="flex gap-1">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium border-b-2 transition-all ${
                activeTab === id
                  ? "border-indigo-500 text-indigo-700"
                  : "border-transparent text-slate-400 hover:text-slate-600"
              }`}
            >
              <Icon size={13} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === "live" && <LiveTranscript status={status} />}
        {activeTab === "conversations" && (
          <ConversationHistory
            conversations={conversations}
            onRefresh={() =>
              getConversations(50, 0)
                .then((data) => setConversations(data.conversations))
                .catch(() => {})
            }
          />
        )}
        {activeTab === "enrollment" && <VoiceEnrollment status={status} />}
        {activeTab === "settings" && <AmbientSettings status={status} />}
      </div>
    </div>
  );
}

// ── Inline Settings Sub-Component ───────────────────────────────

const DEFAULT_CONFIG: AmbientConfig = {
  enabled: false,
  vad_threshold: 0.5,
  auto_ingest: true,
  silence_timeout_s: 120,
  min_speech_ms: 500,
  tts_enabled: true,
  tts_voice: "en_US-lessac-medium",
  tts_speed: 1.0,
  whisper_model_size: "small",
  whisper_device: "auto",
  whisper_language: null,
  record_raw_audio: false,
};

function AmbientSettings({ status }: { status: AmbientState | null }) {
  const [config, setConfig] = useState<AmbientConfig>(DEFAULT_CONFIG);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // Load config on mount
  useEffect(() => {
    getAmbientConfig()
      .then((c) => {
        setConfig(c);
        setDirty(false);
      })
      .catch(() => {
        // Use defaults if ambient not initialized yet
      });
  }, []);

  const updateField = <K extends keyof AmbientConfig>(
    key: K,
    value: AmbientConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
    setSaveMsg(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      const updated = await updateAmbientConfig(config);
      setConfig(updated);
      setDirty(false);
      setSaveMsg("Settings saved");
      setTimeout(() => setSaveMsg(null), 2000);
    } catch (e: unknown) {
      setSaveMsg(e instanceof Error ? e.message : "Save failed");
    }
    setSaving(false);
  };

  const handleReset = () => {
    setConfig(DEFAULT_CONFIG);
    setDirty(true);
    setSaveMsg(null);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      {/* Save bar */}
      {(dirty || saveMsg) && (
        <div className="sticky top-0 z-10 flex items-center justify-between rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-2.5">
          <span className="text-xs text-indigo-600">
            {saveMsg || "You have unsaved changes"}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-600 hover:bg-slate-50 transition-all"
            >
              <RotateCcw size={11} />
              Reset
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-1 rounded-lg bg-indigo-600 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-indigo-700 transition-all disabled:opacity-50"
            >
              <Save size={11} />
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      )}

      {/* VAD Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Voice Activity Detection
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">VAD Threshold</span>
              <span className="text-xs font-mono text-slate-700">
                {config.vad_threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={config.vad_threshold}
              onChange={(e) =>
                updateField("vad_threshold", parseFloat(e.target.value))
              }
              className="w-full"
            />
            <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
              <span>Sensitive (0.1)</span>
              <span>Strict (0.9)</span>
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">
                Min Speech Duration (ms)
              </span>
              <span className="text-xs font-mono text-slate-700">
                {config.min_speech_ms}
              </span>
            </div>
            <input
              type="range"
              min={200}
              max={2000}
              step={100}
              value={config.min_speech_ms}
              onChange={(e) =>
                updateField("min_speech_ms", parseInt(e.target.value))
              }
              className="w-full"
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">
                Silence Timeout (s)
              </span>
              <span className="text-xs font-mono text-slate-700">
                {config.silence_timeout_s}
              </span>
            </div>
            <input
              type="range"
              min={30}
              max={300}
              step={10}
              value={config.silence_timeout_s}
              onChange={(e) =>
                updateField("silence_timeout_s", parseInt(e.target.value))
              }
              className="w-full"
            />
            <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
              <span>30s</span>
              <span>5 min</span>
            </div>
          </div>
          {/* Live VAD stats */}
          <div className="border-t border-slate-100 pt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Total Segments
              </span>
              <span className="text-xs font-mono text-slate-600">
                {status?.vad?.total_segments || 0}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Total Speech Duration
              </span>
              <span className="text-xs font-mono text-slate-600">
                {formatUptime(status?.vad?.total_speech_seconds || 0)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Transcription Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Transcription (Whisper)
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">Model Size</span>
            </div>
            <div className="flex gap-1.5">
              {["tiny", "base", "small", "medium"].map((size) => (
                <button
                  key={size}
                  onClick={() => updateField("whisper_model_size", size)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                    config.whisper_model_size === size
                      ? "bg-indigo-100 text-indigo-700 border border-indigo-200"
                      : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                  }`}
                >
                  {size}
                </button>
              ))}
            </div>
            <p className="text-[9px] text-slate-400 mt-1.5">
              {config.whisper_model_size === "tiny" &&
                "Fastest, ~39M params, WER ~12%"}
              {config.whisper_model_size === "base" &&
                "Fast, ~74M params, WER ~10%"}
              {config.whisper_model_size === "small" &&
                "Balanced, ~244M params, WER ~7.6% (recommended)"}
              {config.whisper_model_size === "medium" &&
                "Accurate, ~769M params, WER ~5.4%, uses more VRAM"}
            </p>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">Device</span>
            </div>
            <div className="flex gap-1.5">
              {["auto", "cuda", "cpu"].map((dev) => (
                <button
                  key={dev}
                  onClick={() => updateField("whisper_device", dev)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                    config.whisper_device === dev
                      ? "bg-indigo-100 text-indigo-700 border border-indigo-200"
                      : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                  }`}
                >
                  {dev}
                </button>
              ))}
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">Language</span>
            </div>
            <select
              value={config.whisper_language || "auto"}
              onChange={(e) =>
                updateField(
                  "whisper_language",
                  e.target.value === "auto" ? null : e.target.value
                )
              }
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            >
              <option value="auto">Auto-detect</option>
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="hi">Hindi</option>
              <option value="ja">Japanese</option>
              <option value="zh">Chinese</option>
              <option value="ko">Korean</option>
              <option value="pt">Portuguese</option>
              <option value="ru">Russian</option>
            </select>
          </div>
          {/* Live transcription stats */}
          <div className="border-t border-slate-100 pt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">Device Active</span>
              <span className="text-xs font-mono text-slate-600">
                {status?.transcriber?.device || "—"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Real-Time Factor
              </span>
              <span className="text-xs font-mono text-slate-600">
                {status?.transcriber?.real_time_factor || 0}x
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Total Audio Processed
              </span>
              <span className="text-xs font-mono text-slate-600">
                {formatUptime(status?.transcriber?.total_audio_seconds || 0)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* TTS Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Text-to-Speech (Piper)
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">TTS Enabled</span>
            <button
              onClick={() => updateField("tts_enabled", !config.tts_enabled)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                config.tts_enabled ? "bg-indigo-500" : "bg-slate-300"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow-sm transition-transform ${
                  config.tts_enabled ? "translate-x-4.5" : "translate-x-0.5"
                }`}
              />
            </button>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-500">TTS Speed</span>
              <span className="text-xs font-mono text-slate-700">
                {config.tts_speed.toFixed(1)}x
              </span>
            </div>
            <input
              type="range"
              min={0.5}
              max={2.0}
              step={0.1}
              value={config.tts_speed}
              onChange={(e) =>
                updateField("tts_speed", parseFloat(e.target.value))
              }
              className="w-full"
            />
            <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
              <span>Slow (0.5x)</span>
              <span>Normal (1.0x)</span>
              <span>Fast (2.0x)</span>
            </div>
          </div>
          {/* Live TTS stats */}
          <div className="border-t border-slate-100 pt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">Available</span>
              <span
                className={`text-xs font-medium ${
                  status?.tts?.available ? "text-emerald-600" : "text-red-500"
                }`}
              >
                {status?.tts?.available ? "Yes" : "No"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">Voice</span>
              <span className="text-xs font-mono text-slate-600">
                {status?.tts?.voice || "—"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Total Syntheses
              </span>
              <span className="text-xs font-mono text-slate-600">
                {status?.tts?.total_syntheses || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Conversation / Ingestion Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Conversation & Ingestion
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-600 font-medium">
                Auto-Ingest to RAG
              </span>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Automatically ingest completed conversations into the memory
                pipeline
              </p>
            </div>
            <button
              onClick={() => updateField("auto_ingest", !config.auto_ingest)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                config.auto_ingest ? "bg-indigo-500" : "bg-slate-300"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow-sm transition-transform ${
                  config.auto_ingest ? "translate-x-4.5" : "translate-x-0.5"
                }`}
              />
            </button>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-600 font-medium">
                Record Raw Audio
              </span>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Archive WAV files on disk (uses more storage)
              </p>
            </div>
            <button
              onClick={() =>
                updateField("record_raw_audio", !config.record_raw_audio)
              }
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                config.record_raw_audio ? "bg-indigo-500" : "bg-slate-300"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow-sm transition-transform ${
                  config.record_raw_audio
                    ? "translate-x-4.5"
                    : "translate-x-0.5"
                }`}
              />
            </button>
          </div>
          {/* Live conversation stats */}
          <div className="border-t border-slate-100 pt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Total Captured
              </span>
              <span className="text-xs font-mono text-slate-600">
                {status?.conversation?.total_conversations || 0}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Auto-Ingested
              </span>
              <span className="text-xs font-mono text-slate-600">
                {status?.conversation?.total_ingested || 0}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">
                Current Turns
              </span>
              <span className="text-xs font-mono text-slate-600">
                {status?.conversation?.current_turns || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Speaker Identification Status */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Speaker Identification
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Enrolled</span>
            <span
              className={`text-xs font-medium ${
                status?.speaker_id?.enrolled
                  ? "text-emerald-600"
                  : "text-amber-500"
              }`}
            >
              {status?.speaker_id?.enrolled ? "Yes" : "No"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Active Speakers</span>
            <span className="text-xs font-mono text-slate-700">
              {status?.speaker_id?.active_clusters || 0}
            </span>
          </div>
          {status?.speaker_id?.cluster_labels &&
            status.speaker_id.cluster_labels.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {status.speaker_id.cluster_labels.map((label) => (
                  <span
                    key={label}
                    className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-600"
                  >
                    {status.speaker_id?.aliases?.[label] || label}
                  </span>
                ))}
              </div>
            )}
        </div>
      </div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600)
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
