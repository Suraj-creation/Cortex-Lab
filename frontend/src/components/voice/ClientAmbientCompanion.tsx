"use client";

import { useEffect, useRef, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Ear,
  Mic,
  MicOff,
  Sparkles,
  Volume2,
  Waves,
} from "lucide-react";
import {
  processAmbientClientAudio,
  startAmbientClientSession,
  stopAmbientClientSession,
} from "@/lib/api";
import { AmbientConfig, AmbientState, AmbientRetentionTrace } from "@/lib/types";

type CompanionMode =
  | "idle"
  | "requesting"
  | "listening"
  | "processing"
  | "speaking"
  | "error";

const MEDIA_TIMESLICE_MS = 320;
const MIN_UTTERANCE_MS = 700;
const SILENCE_END_MS = 700;
const PRE_ROLL_CHUNKS = 2;
const SPEECH_START_THRESHOLD = 0.028;
const SPEECH_END_THRESHOLD = 0.018;

function chooseRecorderMimeType(): string {
  if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
    return "";
  }

  const preferred = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];
  return preferred.find((value) => MediaRecorder.isTypeSupported(value)) || "";
}

function bufferToBase64(buffer: ArrayBuffer): string {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    const slice = bytes.subarray(index, index + chunkSize);
    binary += String.fromCharCode(...slice);
  }
  return window.btoa(binary);
}

function rmsToDb(rms: number): number {
  if (!Number.isFinite(rms) || rms <= 0) {
    return -160;
  }
  return 20 * Math.log10(rms);
}

export function ClientAmbientCompanion({
  ambientState,
  ambientConfig,
  onSessionUpdate,
}: {
  ambientState: AmbientState | null;
  ambientConfig: AmbientConfig | null;
  onSessionUpdate?: () => void | Promise<void>;
}) {
  const [mode, setMode] = useState<CompanionMode>("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>("");
  const [speechDetected, setSpeechDetected] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [pendingUploads, setPendingUploads] = useState(0);
  const [lastTranscript, setLastTranscript] = useState("");
  const [lastAssistantReply, setLastAssistantReply] = useState("");
  const [lastRetention, setLastRetention] = useState<AmbientRetentionTrace | null>(null);

  const sessionIdRef = useRef("");
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const mimeTypeRef = useRef("");
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const uploadQueueRef = useRef<Promise<void>>(Promise.resolve());
  const shutdownRef = useRef<(notifyBackend: boolean) => Promise<void>>(async () => {});
  const utteranceActiveRef = useRef(false);
  const utteranceStartedAtRef = useRef(0);
  const lastSpeechAtRef = useRef(0);
  const activeChunksRef = useRef<Blob[]>([]);
  const preRollChunksRef = useRef<Blob[]>([]);
  const stoppingRef = useRef(false);
  const utterancePeakRmsRef = useRef(0);
  const utteranceRmsSumRef = useRef(0);
  const utteranceRmsCountRef = useRef(0);

  const assistantName = ambientConfig?.assistant_name?.trim() || "Eva";
  const assistantAliases = (ambientConfig?.assistant_aliases || [])
    .map((alias) => String(alias || "").trim())
    .filter(Boolean);
  const followupWindowSeconds = ambientConfig?.companion_followup_window_s || 45;
  const retentionTags = lastRetention?.tags || [];
  const retainedAs =
    lastRetention?.memory_decision === "priority"
      ? "priority memory"
      : lastRetention?.memory_decision === "structured"
        ? "structured memory"
        : lastRetention?.memory_decision === "session_only"
          ? "session context"
          : "discarded";

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    return () => {
      void shutdownRef.current(false);
    };
  }, []);

  async function notifySessionUpdate() {
    if (!onSessionUpdate) {
      return;
    }
    await onSessionUpdate();
  }

  function stopAnalyserLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }

  function teardownMedia() {
    stopAnalyserLoop();
    try {
      recorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    } catch {}
    try {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    } catch {}
    try {
      sourceNodeRef.current?.disconnect();
    } catch {}
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close().catch(() => {});
    }
    recorderRef.current = null;
    streamRef.current = null;
    sourceNodeRef.current = null;
    analyserRef.current = null;
    audioContextRef.current = null;
    utteranceActiveRef.current = false;
    utterancePeakRmsRef.current = 0;
    utteranceRmsSumRef.current = 0;
    utteranceRmsCountRef.current = 0;
    activeChunksRef.current = [];
    preRollChunksRef.current = [];
    setSpeechDetected(false);
    setAudioLevel(0);
  }

  async function playAssistantAudio(audioBase64: string) {
    if (!audioBase64) {
      return;
    }

    const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
    audioPlayerRef.current = audio;
    audio.onended = () => {
      setMode(stoppingRef.current || !sessionIdRef.current ? "idle" : "listening");
    };
    audio.onerror = () => {
      setMode(stoppingRef.current || !sessionIdRef.current ? "idle" : "listening");
    };

    try {
      setMode("speaking");
      await audio.play();
    } catch {
      setMode("listening");
    }
  }

  async function uploadUtterance(
    blob: Blob,
    durationMs: number,
    audioStats?: { peakDb?: number; avgDb?: number }
  ) {
    const activeSessionId = sessionIdRef.current;
    if (!activeSessionId || blob.size === 0) {
      return;
    }

    const buffer = await blob.arrayBuffer();
    const response = await processAmbientClientAudio({
      sessionId: activeSessionId,
      audioBase64: bufferToBase64(buffer),
      mimeType: blob.type || mimeTypeRef.current || "audio/webm",
      platform: "web",
      estimatedDurationS: durationMs / 1000,
      metadata: {
        surface: "ambient-panel",
        ...(audioStats?.peakDb !== undefined ? { audio_peak_db: audioStats.peakDb } : {}),
        ...(audioStats?.avgDb !== undefined ? { audio_avg_db: audioStats.avgDb } : {}),
      },
    });

    if (stoppingRef.current || sessionIdRef.current !== activeSessionId) {
      return;
    }

    if (response.transcript?.trim()) {
      setLastTranscript(response.transcript.trim());
    }
    if (response.assistant_text?.trim()) {
      setLastAssistantReply(response.assistant_text.trim());
    }
    setLastRetention(response.retention_trace || null);
    if (response.session_id && response.session_id !== activeSessionId) {
      setSessionId(response.session_id);
    }
    await notifySessionUpdate();

    if (response.assistant_audio_base64) {
      await playAssistantAudio(response.assistant_audio_base64);
    } else {
      setMode("listening");
    }
  }

  async function finalizeUtterance(force: boolean = false) {
    if (!utteranceActiveRef.current) {
      return;
    }

    const chunks = [...activeChunksRef.current];
    const durationMs = performance.now() - utteranceStartedAtRef.current;
    const peakRms = utterancePeakRmsRef.current;
    const avgRms =
      utteranceRmsCountRef.current > 0 ? utteranceRmsSumRef.current / utteranceRmsCountRef.current : 0;

    utteranceActiveRef.current = false;
    utterancePeakRmsRef.current = 0;
    utteranceRmsSumRef.current = 0;
    utteranceRmsCountRef.current = 0;
    activeChunksRef.current = [];
    preRollChunksRef.current = [];

    if (!force && durationMs < MIN_UTTERANCE_MS) {
      return;
    }

    const mimeType = mimeTypeRef.current || chooseRecorderMimeType() || "audio/webm";
    const blob = new Blob(chunks, { type: mimeType });
    if (blob.size < 512) {
      return;
    }

    setPendingUploads((value) => value + 1);
    uploadQueueRef.current = uploadQueueRef.current
      .then(async () => {
        setMode("processing");
        await uploadUtterance(blob, durationMs, {
          peakDb: rmsToDb(peakRms),
          avgDb: rmsToDb(avgRms),
        });
      })
      .catch((uploadError) => {
        setMode("error");
        setError(uploadError instanceof Error ? uploadError.message : String(uploadError));
      })
      .finally(() => {
        setPendingUploads((value) => Math.max(0, value - 1));
        if (!stoppingRef.current) {
          setMode("listening");
        }
      });
  }

  function startAnalyserLoop() {
    const analyser = analyserRef.current;
    if (!analyser) {
      return;
    }

    const samples = new Uint8Array(analyser.fftSize);

    const tick = () => {
      analyser.getByteTimeDomainData(samples);
      let sum = 0;
      for (const value of samples) {
        const centered = value / 128 - 1;
        sum += centered * centered;
      }
      const rms = Math.sqrt(sum / samples.length);
      setAudioLevel(rms);

      const now = performance.now();
      if (rms >= SPEECH_START_THRESHOLD) {
        setSpeechDetected(true);
        lastSpeechAtRef.current = now;
        if (!utteranceActiveRef.current) {
          utteranceActiveRef.current = true;
          utteranceStartedAtRef.current = now;
          utterancePeakRmsRef.current = rms;
          utteranceRmsSumRef.current = rms;
          utteranceRmsCountRef.current = 1;
          activeChunksRef.current = [...preRollChunksRef.current];
          preRollChunksRef.current = [];
        } else {
          utterancePeakRmsRef.current = Math.max(utterancePeakRmsRef.current, rms);
          utteranceRmsSumRef.current += rms;
          utteranceRmsCountRef.current += 1;
        }
      } else if (utteranceActiveRef.current) {
        utterancePeakRmsRef.current = Math.max(utterancePeakRmsRef.current, rms);
        utteranceRmsSumRef.current += rms;
        utteranceRmsCountRef.current += 1;
        if (lastSpeechAtRef.current === 0) {
          lastSpeechAtRef.current = now;
        }
        if (rms <= SPEECH_END_THRESHOLD && now - lastSpeechAtRef.current >= SILENCE_END_MS) {
          setSpeechDetected(false);
          void finalizeUtterance(false);
        }
      } else {
        setSpeechDetected(false);
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
  }

  async function startCompanion() {
    if (mode === "requesting" || mode === "listening" || mode === "processing" || mode === "speaking") {
      return;
    }

    if (typeof window === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setMode("error");
      setError("This browser does not support microphone capture for the ambient companion.");
      return;
    }

    setMode("requesting");
    setError(null);

    try {
      const session = await startAmbientClientSession({
        platform: "web",
        metadata: {
          surface: "ambient-panel",
        },
      });
      sessionIdRef.current = session.session_id;
      setSessionId(session.session_id);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const AudioContextCtor =
        window.AudioContext ||
        (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextCtor) {
        throw new Error("This browser cannot create an audio analysis context for continuous listening.");
      }
      const audioContext = new AudioContextCtor();
      const sourceNode = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.82;
      sourceNode.connect(analyser);

      const mimeType = chooseRecorderMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      mimeTypeRef.current = mimeType || recorder.mimeType || "audio/webm";
      recorder.ondataavailable = (event) => {
        if (!event.data || event.data.size === 0) {
          return;
        }

        if (utteranceActiveRef.current) {
          activeChunksRef.current.push(event.data);
          return;
        }

        preRollChunksRef.current = [
          ...preRollChunksRef.current,
          event.data,
        ].slice(-PRE_ROLL_CHUNKS);
      };

      recorder.start(MEDIA_TIMESLICE_MS);

      recorderRef.current = recorder;
      streamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceNodeRef.current = sourceNode;
      analyserRef.current = analyser;
      setMode("listening");
      await notifySessionUpdate();
      startAnalyserLoop();
    } catch (startError) {
      if (sessionIdRef.current) {
        try {
          await stopAmbientClientSession({
            sessionId: sessionIdRef.current,
            reason: "startup_failed",
          });
        } catch {}
      }
      sessionIdRef.current = "";
      setSessionId("");
      teardownMedia();
      setMode("error");
      setError(startError instanceof Error ? startError.message : String(startError));
    }
  }

  async function shutdownCompanion(notifyBackend: boolean) {
    if (stoppingRef.current) {
      return;
    }
    stoppingRef.current = true;

    const activeSessionId = sessionIdRef.current;
    try {
      try {
        audioPlayerRef.current?.pause();
        audioPlayerRef.current = null;
      } catch {}
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        await new Promise<void>((resolve) => {
          const handleStop = () => resolve();
          recorder.addEventListener("stop", handleStop, { once: true });
          recorder.stop();
        });
      }

      await finalizeUtterance(true);
      await uploadQueueRef.current;
      teardownMedia();

      if (notifyBackend && activeSessionId) {
        await stopAmbientClientSession({
          sessionId: activeSessionId,
          reason: "user_request",
        });
        await notifySessionUpdate();
      }
    } catch (stopError) {
      setMode("error");
      setError(stopError instanceof Error ? stopError.message : String(stopError));
    } finally {
      setSessionId("");
      sessionIdRef.current = "";
      stoppingRef.current = false;
      setMode((currentMode) => (currentMode === "error" ? currentMode : "idle"));
    }
  }

  shutdownRef.current = shutdownCompanion;

  async function stopCompanion() {
    await shutdownCompanion(true);
  }

  const isActive =
    mode === "listening" ||
    mode === "processing" ||
    mode === "speaking" ||
    mode === "requesting";

  return (
    <div className="rounded-[28px] border border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(129,140,248,0.14),_transparent_36%),linear-gradient(145deg,_#ffffff,_#f8fafc)] p-5 shadow-[0_24px_70px_-42px_rgba(15,23,42,0.38)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-2xl">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.32em] text-slate-400">
            <Sparkles size={12} />
            Eva Companion
          </div>
          <h3 className="mt-2 text-xl font-semibold text-slate-900">
            Continuous client-side listening with backend memory refinement
          </h3>
          <p className="mt-2 text-sm leading-6 text-slate-500">
            Say <span className="font-semibold text-slate-700">{assistantName}</span> to wake the
            assistant. Every utterance stays in the active session, and the backend tags
            high-value turns so other agents can retrieve them later.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1">
              Wake aliases: {assistantAliases.length ? assistantAliases.join(", ") : assistantName.toLowerCase()}
            </span>
            <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1">
              Follow-up window: {followupWindowSeconds}s
            </span>
          </div>
        </div>

        <div className="flex w-full flex-col gap-3 lg:max-w-sm">
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={startCompanion}
              disabled={isActive}
              className="flex items-center justify-center gap-2 rounded-2xl bg-slate-900 px-4 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              <Mic size={15} />
              {mode === "requesting" ? "Preparing..." : "Start Companion"}
            </button>
            <button
              onClick={stopCompanion}
              disabled={!isActive}
              className="flex items-center justify-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
            >
              <MicOff size={15} />
              Stop
            </button>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white/80 p-4">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-400">
                Session
              </span>
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.22em] ${
                  mode === "listening"
                    ? "bg-emerald-50 text-emerald-600"
                    : mode === "processing"
                      ? "bg-indigo-50 text-indigo-600"
                      : mode === "speaking"
                        ? "bg-amber-50 text-amber-600"
                        : mode === "error"
                          ? "bg-red-50 text-red-600"
                          : "bg-slate-100 text-slate-500"
                }`}
              >
                {mode}
              </span>
            </div>
            <div className="mt-3 space-y-2 text-xs text-slate-500">
              <div className="flex items-center justify-between gap-4">
                <span>Backend session</span>
                <span className="font-mono text-[11px] text-slate-700">
                  {sessionId || ambientState?.client_session?.active_session_id || "not started"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Speech activity</span>
                <span className={speechDetected ? "text-emerald-600" : "text-slate-400"}>
                  {speechDetected ? "capturing utterance" : "waiting"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Queued turns</span>
                <span className="text-slate-700">{pendingUploads}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[1.25fr_0.95fr]">
        <div className="rounded-3xl border border-slate-200 bg-slate-950 p-5 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-400">
              <Ear size={12} />
              Voice Loop
            </div>
            <div className="flex items-center gap-2 text-[11px] text-slate-400">
              <Activity size={12} />
              level {(audioLevel * 100).toFixed(1)}
            </div>
          </div>

          <div className="mt-4 h-20 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
            <div className="flex h-full items-end gap-1.5">
              {Array.from({ length: 22 }).map((_, index) => {
                const variance = 0.22 + ((index % 5) * 0.12);
                const height = Math.max(10, Math.min(100, audioLevel * 700 * variance));
                return (
                  <div
                    key={index}
                    className={`flex-1 rounded-full transition-all ${
                      speechDetected ? "bg-emerald-400/90" : "bg-white/20"
                    }`}
                    style={{ height: `${height}%` }}
                  />
                );
              })}
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-slate-400">
                <Waves size={12} />
                Capture
              </div>
              <p className="mt-2 text-sm text-slate-100">
                Browser audio is chunked locally, VAD-trimmed, then sent to Gemini STT.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-slate-400">
                <BrainCircuit size={12} />
                Retention
              </div>
              <p className="mt-2 text-sm text-slate-100">
                Decisions, technical notes, and retrieval cues are tagged for downstream agents.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-slate-400">
                <Volume2 size={12} />
                Reply
              </div>
              <p className="mt-2 text-sm text-slate-100">
                Spoken answers come back from backend TTS without requiring a manual URL setup.
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_18px_48px_-38px_rgba(15,23,42,0.45)]">
            <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-400">
              Latest Transcript
            </div>
            <p className="mt-3 min-h-[72px] text-sm leading-6 text-slate-700">
              {lastTranscript || "Waiting for the first spoken turn..."}
            </p>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_18px_48px_-38px_rgba(15,23,42,0.45)]">
            <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-400">
              Latest Reply
            </div>
            <p className="mt-3 min-h-[72px] text-sm leading-6 text-slate-700">
              {lastAssistantReply || `${assistantName} will answer here when a reply is needed.`}
            </p>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_18px_48px_-38px_rgba(15,23,42,0.45)]">
            <div className="flex items-center justify-between">
              <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-400">
                Retrieval Tagging
              </div>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                {retainedAs}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {retentionTags.length > 0 ? (
                retentionTags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-indigo-600"
                  >
                    {tag.replace(/_/g, " ")}
                  </span>
                ))
              ) : (
                <span className="text-sm text-slate-400">
                  Tags will appear here after the backend evaluates a spoken turn.
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
          {error}
        </div>
      )}
    </div>
  );
}
