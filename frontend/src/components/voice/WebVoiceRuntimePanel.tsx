"use client";

import type { RuntimeSelection } from "@/lib/types";

interface WebVoiceRuntimePanelProps {
  selection: RuntimeSelection;
  onChange: (selection: RuntimeSelection) => void;
  availability?: {
    llm?: Record<string, boolean>;
    stt?: Record<string, boolean>;
    tts?: Record<string, boolean>;
  };
}

export function WebVoiceRuntimePanel({
  selection,
  onChange,
  availability,
}: WebVoiceRuntimePanelProps) {
  const setMode = (mode: RuntimeSelection["mode"]) => {
    onChange({
      ...selection,
      mode,
      allowCloudFallback: mode === "local_offline" ? false : selection.allowCloudFallback,
    });
  };

  const setVoiceProvider = (kind: "sttProvider" | "ttsProvider", value: RuntimeSelection["sttProvider"]) => {
    onChange({
      ...selection,
      [kind]: value,
    });
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-slate-700">Voice Runtime</h3>
        <p className="text-xs text-slate-500 mt-1">
          Choose cloud, hybrid, or offline execution and keep voice providers aligned with that mode.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {(["cloud", "hybrid", "local_offline"] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setMode(mode)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium border ${
              selection.mode === mode
                ? "bg-indigo-50 text-indigo-700 border-indigo-200"
                : "bg-slate-50 text-slate-500 border-slate-200 hover:bg-slate-100"
            }`}
          >
            {mode}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <label className="text-xs text-slate-600">
          STT Provider
          <select
            value={selection.sttProvider}
            onChange={(e) => setVoiceProvider("sttProvider", e.target.value as RuntimeSelection["sttProvider"])}
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs"
          >
            {(["traditional", "local", "gemini"] as const).map((item) => (
              <option key={item} value={item}>
                {item}
                {availability?.stt && availability.stt[item] === false ? " (unavailable)" : ""}
              </option>
            ))}
          </select>
        </label>

        <label className="text-xs text-slate-600">
          TTS Provider
          <select
            value={selection.ttsProvider}
            onChange={(e) => setVoiceProvider("ttsProvider", e.target.value as RuntimeSelection["ttsProvider"])}
            className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs"
          >
            {(["traditional", "local", "gemini"] as const).map((item) => (
              <option key={item} value={item}>
                {item}
                {availability?.tts && availability.tts[item] === false ? " (unavailable)" : ""}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
