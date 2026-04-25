"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Mic,
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
  Cloud,
  Cpu,
} from "lucide-react";
import {
  getAmbientStatus,
  getConversations,
  getAmbientConfig,
  updateAmbientConfig,
  getVoiceProviders,
  setSTTProvider,
  setTTSProvider,
} from "@/lib/api";
import { AmbientState, AmbientStatusType, AmbientConfig, ConversationRecord, VoiceProviders, VoiceProviderType } from "@/lib/types";
import { LiveTranscript } from "./LiveTranscript";
import { ConversationHistory } from "./ConversationHistory";
import { VoiceEnrollment } from "./VoiceEnrollment";
import { ClientAmbientCompanion } from "./voice/ClientAmbientCompanion";

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
  const [ambientConfig, setAmbientConfig] = useState<AmbientConfig | null>(null);
  const [activeTab, setActiveTab] = useState<AmbientTab>("live");
  const [conversations, setConversations] = useState<ConversationRecord[]>([]);
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

  useEffect(() => {
    getAmbientConfig()
      .then((config) => setAmbientConfig(config))
      .catch(() => {});
  }, []);

  // Load conversations when tab is active
  useEffect(() => {
    if (activeTab === "conversations") {
      getConversations(50, 0)
        .then((data) => setConversations(data.conversations))
        .catch(() => {});
    }
  }, [activeTab]);
  const isLiveMode = status?.operating_mode === "gemini_live";

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
                  {isLiveMode && (
                    <span className="rounded-full border border-violet-200 bg-violet-50 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-violet-600">
                      Gemini Live
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-right shadow-sm">
              <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-400">
                Companion
              </div>
              <div className="mt-1 text-xs font-medium text-slate-700">
                {status?.client_session?.active_session_id
                  ? "Client listening active"
                  : "Ready for browser capture"}
              </div>
            </div>
          </div>
        </div>

        {error && (
          <div className="mt-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        {/* Stats Bar */}
        {status && (
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
            {status.stt_provider && (
              <span className={`text-[10px] font-medium ${
                status.stt_provider === "gemini" ? "text-violet-500" : "text-indigo-500"
              }`}>
                STT: {status.stt_provider === "gemini" ? "Gemini" : "Whisper"}
              </span>
            )}
            {status.tts_provider && (
              <span className={`text-[10px] font-medium ${
                status.tts_provider === "gemini" ? "text-violet-500" : "text-indigo-500"
              }`}>
                TTS: {status.tts_provider === "gemini" ? "Gemini" : "Piper"}
              </span>
            )}
            {status.live?.running && (
              <span className="text-[10px] font-medium text-violet-600">
                Live: {status.live.state}
              </span>
            )}
            {status.live?.running && typeof status.live.user_turns === "number" && (
              <span className="text-[10px] text-slate-400">
                Turns: {status.live.user_turns}/{status.live.assistant_turns || 0}
              </span>
            )}
            {status.client_session?.active_session_id && (
              <span className="text-[10px] font-medium text-slate-500">
                Session: {status.client_session.active_session_id.slice(0, 18)}
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
        {activeTab === "live" && (
          <div className="space-y-6 p-6">
            <ClientAmbientCompanion
              ambientState={status}
              ambientConfig={ambientConfig}
              onSessionUpdate={fetchStatus}
            />
            <div className="overflow-hidden rounded-[28px] border border-slate-200 bg-white shadow-[0_18px_48px_-38px_rgba(15,23,42,0.45)]">
              <LiveTranscript status={status} />
            </div>
          </div>
        )}
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
        {activeTab === "settings" && (
          <AmbientSettings
            status={status}
            ambientConfig={ambientConfig}
            onConfigSaved={setAmbientConfig}
          />
        )}
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
  stt_provider: "gemini",
  tts_provider: "gemini",
  tts_enabled: true,
  tts_voice: "en_US-lessac-medium",
  tts_speed: 1.0,
  whisper_model_size: "small",
  whisper_device: "auto",
  whisper_language: null,
  record_raw_audio: false,
  gemini_tts_voice: "Kore",
  live_mode: "gemini_live",
  energy_gate_threshold: 700,
  energy_min_speech_ms: 320,
  energy_silence_ms: 420,
  assistant_name: "Eva",
  assistant_aliases: ["eva", "ava", "cortex", "assistant"],
  companion_followup_window_s: 45,
};

function AmbientSettings({
  status,
  ambientConfig,
  onConfigSaved,
}: {
  status: AmbientState | null;
  ambientConfig: AmbientConfig | null;
  onConfigSaved?: (config: AmbientConfig) => void;
}) {
  const [config, setConfig] = useState<AmbientConfig>(DEFAULT_CONFIG);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [providers, setProviders] = useState<VoiceProviders | null>(null);
  const [switching, setSwitching] = useState(false);
  const isLocalStt = config.stt_provider === "traditional" || config.stt_provider === "local";
  const isLocalTts = config.tts_provider === "traditional" || config.tts_provider === "local";

  // Load config on mount
  useEffect(() => {
    getVoiceProviders()
      .then(setProviders)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!ambientConfig || dirty) {
      return;
    }
    setConfig(ambientConfig);
  }, [ambientConfig, dirty]);

  useEffect(() => {
    if (ambientConfig) {
      setConfig(ambientConfig);
      return;
    }
    getAmbientConfig()
      .then((c) => {
        setConfig(c);
        setDirty(false);
        onConfigSaved?.(c);
      })
      .catch(() => {
        // Use defaults if ambient not initialized yet
      });
  }, [ambientConfig, onConfigSaved]);

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
      onConfigSaved?.(updated);
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

  const handleSwitchSTT = async (provider: VoiceProviderType) => {
    setSwitching(true);
    try {
      const res = await setSTTProvider(provider);
      if (res.success) {
        const nextConfig = { ...config, stt_provider: provider };
        setConfig(nextConfig);
        onConfigSaved?.(nextConfig);
        setProviders((prev) => prev ? { ...prev, stt_provider: provider } : prev);
      }
    } catch {
      setSaveMsg("Failed to switch STT provider");
    }
    setSwitching(false);
  };

  const handleSwitchTTS = async (provider: VoiceProviderType) => {
    setSwitching(true);
    try {
      const res = await setTTSProvider(provider);
      if (res.success) {
        const nextConfig = { ...config, tts_provider: provider };
        setConfig(nextConfig);
        onConfigSaved?.(nextConfig);
        setProviders((prev) => prev ? { ...prev, tts_provider: provider } : prev);
      }
    } catch {
      setSaveMsg("Failed to switch TTS provider");
    }
    setSwitching(false);
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

      {/* Voice Providers */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Voice Providers
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <div>
            <span className="text-xs text-slate-600 font-medium">Runtime Path</span>
            <p className="text-[10px] text-slate-400 mt-0.5">
              Gemini Live keeps always-on cloud listening with full-duplex assistant responses.
            </p>
            <div className="flex gap-1.5 mt-2">
              <button
                onClick={() => updateField("live_mode", "gemini_live")}
                disabled={!providers?.gemini_available}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  (config.live_mode || "classic") === "gemini_live"
                    ? "bg-violet-100 text-violet-700 border border-violet-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                } disabled:opacity-40`}
              >
                Gemini Live
              </button>
              <button
                onClick={() => updateField("live_mode", "classic")}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  (config.live_mode || "classic") === "classic"
                    ? "bg-indigo-100 text-indigo-700 border border-indigo-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                }`}
              >
                Classic
              </button>
            </div>
          </div>

          {/* STT Provider */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-xs text-slate-600 font-medium">Speech-to-Text Engine</span>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {config.stt_provider === "gemini"
                    ? "Gemini AI — cloud-based, multilingual, high accuracy"
                    : "Whisper (local) — on-device, private, configurable model size"}
                </p>
              </div>
            </div>
            <div className="flex gap-1.5">
              <button
                onClick={() => handleSwitchSTT("local")}
                disabled={switching || (!providers?.traditional_stt_available && !providers?.local_stt_available)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  isLocalStt
                    ? "bg-indigo-100 text-indigo-700 border border-indigo-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                } disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                <Cpu size={12} />
                Local (Whisper)
              </button>
              <button
                onClick={() => handleSwitchSTT("gemini")}
                disabled={switching || !providers?.gemini_available}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  config.stt_provider === "gemini"
                    ? "bg-violet-100 text-violet-700 border border-violet-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                } disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                <Cloud size={12} />
                Gemini AI
              </button>
            </div>
            {!providers?.gemini_available && (
              <p className="text-[9px] text-amber-500 mt-1">
                Gemini requires GOOGLE_API_KEY in backend/.env
              </p>
            )}
          </div>

          {/* TTS Provider */}
          <div className="border-t border-slate-100 pt-4">
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-xs text-slate-600 font-medium">Text-to-Speech Engine</span>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {config.tts_provider === "gemini"
                    ? "Gemini AI — natural voices, cloud-based, multiple voice options"
                    : "Piper (local) — on-device ONNX, fast, private"}
                </p>
              </div>
            </div>
            <div className="flex gap-1.5">
              <button
                onClick={() => handleSwitchTTS("local")}
                disabled={switching || (!providers?.traditional_tts_available && !providers?.local_tts_available)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  isLocalTts
                    ? "bg-indigo-100 text-indigo-700 border border-indigo-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                } disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                <Cpu size={12} />
                Local (Piper)
              </button>
              <button
                onClick={() => handleSwitchTTS("gemini")}
                disabled={switching || !providers?.gemini_available}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
                  config.tts_provider === "gemini"
                    ? "bg-violet-100 text-violet-700 border border-violet-200"
                    : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                } disabled:opacity-40 disabled:cursor-not-allowed`}
              >
                <Cloud size={12} />
                Gemini AI
              </button>
            </div>
          </div>

          {/* Gemini TTS Voice Selector */}
          {config.tts_provider === "gemini" && providers?.gemini_tts_voices && (
            <div className="border-t border-slate-100 pt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-600 font-medium">Gemini Voice</span>
                <span className="text-[10px] text-violet-500 font-medium">{config.gemini_tts_voice}</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {providers.gemini_tts_voices.map((voice) => (
                  <button
                    key={voice}
                    onClick={() => updateField("gemini_tts_voice", voice)}
                    className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition-all ${
                      config.gemini_tts_voice === voice
                        ? "bg-violet-100 text-violet-700 border border-violet-200"
                        : "bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100"
                    }`}
                  >
                    {voice}
                  </button>
                ))}
              </div>
              <p className="text-[9px] text-slate-400 mt-1.5">
                Select a Gemini TTS voice. Changes apply immediately after saving.
              </p>
            </div>
          )}

          {/* Provider status badges */}
          <div className="border-t border-slate-100 pt-3 flex flex-wrap gap-2">
            <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
              config.stt_provider === "gemini"
                ? "bg-violet-50 text-violet-600 border border-violet-200"
                : "bg-indigo-50 text-indigo-600 border border-indigo-200"
            }`}>
              <Mic size={9} />
              STT: {config.stt_provider === "gemini" ? "Gemini" : "Whisper"}
            </span>
            <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
              config.tts_provider === "gemini"
                ? "bg-violet-50 text-violet-600 border border-violet-200"
                : "bg-indigo-50 text-indigo-600 border border-indigo-200"
            }`}>
              <Volume2 size={9} />
              TTS: {config.tts_provider === "gemini" ? "Gemini" : "Piper"}
            </span>
            {providers?.gemini_available && (
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200 px-2 py-0.5 text-[10px] font-medium">
                <Cloud size={9} />
                Gemini Ready
              </span>
            )}
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Companion Identity
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="block">
              <span className="text-xs text-slate-600 font-medium">
                Assistant Name
              </span>
              <input
                type="text"
                value={config.assistant_name || ""}
                onChange={(e) => updateField("assistant_name", e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700 outline-none transition focus:border-indigo-300 focus:bg-white"
                placeholder="Eva"
              />
            </label>
            <label className="block">
              <span className="text-xs text-slate-600 font-medium">
                Wake Aliases
              </span>
              <input
                type="text"
                value={(config.assistant_aliases || []).join(", ")}
                onChange={(e) =>
                  updateField(
                    "assistant_aliases",
                    e.target.value
                      .split(",")
                      .map((value) => value.trim().toLowerCase())
                      .filter(Boolean)
                  )
                }
                className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700 outline-none transition focus:border-indigo-300 focus:bg-white"
                placeholder="eva, ava, cortex"
              />
            </label>
          </div>
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-xs text-slate-500">
                Follow-up Reply Window
              </span>
              <span className="text-xs font-mono text-slate-700">
                {config.companion_followup_window_s || 45}s
              </span>
            </div>
            <input
              type="range"
              min={10}
              max={120}
              step={5}
              value={config.companion_followup_window_s || 45}
              onChange={(e) =>
                updateField(
                  "companion_followup_window_s",
                  parseInt(e.target.value, 10)
                )
              }
              className="w-full"
            />
            <p className="mt-1.5 text-[10px] text-slate-400">
              After you say the assistant name, turns inside this window stay
              in spoken conversation mode without repeating the wake word.
            </p>
          </div>
        </div>
      </div>

      {/* VAD Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Voice Activity Detection
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          {(config.live_mode || "classic") === "gemini_live" ? (
            <>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-slate-500">Energy Gate Threshold</span>
                  <span className="text-xs font-mono text-slate-700">
                    {Math.round(config.energy_gate_threshold || 700)}
                  </span>
                </div>
                <input
                  type="range"
                  min={200}
                  max={2500}
                  step={50}
                  value={config.energy_gate_threshold || 700}
                  onChange={(e) =>
                    updateField("energy_gate_threshold", parseInt(e.target.value))
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
                  <span>More Sensitive</span>
                  <span>More Strict</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-slate-500">
                    Min Speech Window (ms)
                  </span>
                  <span className="text-xs font-mono text-slate-700">
                    {config.energy_min_speech_ms || 320}
                  </span>
                </div>
                <input
                  type="range"
                  min={160}
                  max={1200}
                  step={40}
                  value={config.energy_min_speech_ms || 320}
                  onChange={(e) =>
                    updateField("energy_min_speech_ms", parseInt(e.target.value))
                  }
                  className="w-full"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-slate-500">
                    End-of-Turn Silence (ms)
                  </span>
                  <span className="text-xs font-mono text-slate-700">
                    {config.energy_silence_ms || 420}
                  </span>
                </div>
                <input
                  type="range"
                  min={200}
                  max={1400}
                  step={40}
                  value={config.energy_silence_ms || 420}
                  onChange={(e) =>
                    updateField("energy_silence_ms", parseInt(e.target.value))
                  }
                  className="w-full"
                />
              </div>
            </>
          ) : (
            <>
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
            </>
          )}
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
          Transcription {config.stt_provider === "gemini" ? "(Gemini AI)" : "(Whisper)"}
        </h3>
        <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4">
          {config.stt_provider === "gemini" ? (
            <div className="text-center py-4">
              <Cloud size={24} className="mx-auto text-violet-400 mb-2" />
              <p className="text-xs text-slate-500">
                Gemini AI handles transcription in the cloud.
              </p>
              <p className="text-[10px] text-slate-400 mt-1">
                No local model configuration needed. Supports multilingual transcription automatically.
              </p>
            </div>
          ) : (
          <>
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
          </>
          )}
        </div>
      </div>

      {/* TTS Settings */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Text-to-Speech {config.tts_provider === "gemini" ? "(Gemini AI)" : "(Piper)"}
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
          {isLocalTts && (
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
          )}
          {config.tts_provider === "gemini" && (
            <div className="text-center py-2">
              <p className="text-[10px] text-slate-400">
                Voice: <span className="font-medium text-violet-500">{config.gemini_tts_voice}</span> · Change in Voice Providers above
              </p>
            </div>
          )}
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
