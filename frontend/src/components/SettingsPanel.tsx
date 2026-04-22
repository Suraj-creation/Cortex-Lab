"use client";

import { useEffect, useMemo, useState } from "react";

import { getModelpackManifest } from "@/lib/api";
import { ChatSettings, ModelpackEntry, ModelpackManifest } from "@/lib/types";
import { Download, ExternalLink, X, RotateCcw, Brain, Zap } from "lucide-react";

import { WebOfflineReadinessBadge } from "./modelpacks/WebOfflineReadinessBadge";

interface Props {
  settings: ChatSettings;
  onUpdate: (settings: ChatSettings) => void;
  onClose: () => void;
}

const MODELPACK_DOCS_FALLBACK =
  "https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs%2Fapi%2Fkotlin%2Fgetting_started.md";

const FALLBACK_MODELPACKS: ModelpackEntry[] = [
  {
    id: "gemma-4-e4b-it-litert-lm",
    display_name: "Gemma 4 E4B IT (LiteRT-LM)",
    version: "2026.04.0",
    target: "android-web",
    family: "gemma-4",
    quantization: "E4B",
    summary: "Higher-quality Gemma 4 local model for capable devices.",
    availability: "available",
    download_url: "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm",
    cta_label: "Download from Hugging Face",
    files: [],
  },
  {
    id: "gemma-4-e2b-it-litert-lm",
    display_name: "Gemma 4 E2B IT (LiteRT-LM)",
    version: "2026.04.0",
    target: "android-web",
    family: "gemma-4",
    quantization: "E2B",
    summary: "Lean Gemma 4 local model for faster installs and mid-range devices.",
    availability: "available",
    download_url: "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm",
    cta_label: "Download from Hugging Face",
    files: [],
  },
  {
    id: "gemma-3-5-ft-local",
    display_name: "Gemma 3.5 Fine-Tuned (Planned)",
    version: "planned",
    target: "android-web",
    family: "gemma-3.5",
    quantization: "tbd",
    summary: "Reserved slot for your upcoming fine-tuned local model integration.",
    availability: "coming_soon",
    cta_label: "Coming Soon",
    files: [],
  },
];

function normalizeModelpackManifest(input: unknown): ModelpackManifest {
  const raw = input && typeof input === "object" ? (input as Record<string, unknown>) : {};
  const rawPacks = Array.isArray(raw.packs) ? raw.packs : [];

  const packs: ModelpackEntry[] = rawPacks
    .filter((pack): pack is Record<string, unknown> => Boolean(pack && typeof pack === "object"))
    .map((pack, idx) => {
      const rawFiles = Array.isArray(pack.files) ? pack.files : [];
      return {
        id: typeof pack.id === "string" && pack.id.trim() ? pack.id.trim() : `pack-${idx + 1}`,
        display_name:
          typeof pack.display_name === "string" && pack.display_name.trim()
            ? pack.display_name.trim()
            : `Model Pack ${idx + 1}`,
        version: typeof pack.version === "string" && pack.version.trim() ? pack.version.trim() : "unknown",
        target: typeof pack.target === "string" ? pack.target : undefined,
        family: typeof pack.family === "string" ? pack.family : undefined,
        quantization: typeof pack.quantization === "string" ? pack.quantization : undefined,
        summary: typeof pack.summary === "string" ? pack.summary : undefined,
        availability: pack.availability === "coming_soon" ? "coming_soon" : "available",
        download_url:
          typeof pack.download_url === "string" && pack.download_url.trim()
            ? pack.download_url.trim()
            : undefined,
        cta_label: typeof pack.cta_label === "string" ? pack.cta_label : undefined,
        docs_url: typeof pack.docs_url === "string" ? pack.docs_url : undefined,
        requires: Array.isArray(pack.requires)
          ? pack.requires.filter((item): item is string => typeof item === "string")
          : undefined,
        files: rawFiles
          .filter((file): file is Record<string, unknown> => Boolean(file && typeof file === "object"))
          .map((file) => ({
            path: typeof file.path === "string" ? file.path : "",
            size_bytes: typeof file.size_bytes === "number" ? file.size_bytes : 0,
            sha256: typeof file.sha256 === "string" ? file.sha256 : "",
          })),
      };
    });

  return {
    schema_version: typeof raw.schema_version === "string" ? raw.schema_version : "1.1",
    generated_at: typeof raw.generated_at === "string" ? raw.generated_at : new Date().toISOString(),
    signature_required: raw.signature_required !== false,
    source: typeof raw.source === "string" ? raw.source : "ui-fallback",
    docs_url: typeof raw.docs_url === "string" ? raw.docs_url : MODELPACK_DOCS_FALLBACK,
    channels: Array.isArray(raw.channels)
      ? raw.channels.filter((item): item is string => typeof item === "string")
      : undefined,
    packs: packs.length > 0 ? packs : FALLBACK_MODELPACKS,
  };
}

export function SettingsPanel({ settings, onUpdate, onClose }: Props) {
  const isLocalProvider = settings.llmProvider === "local" || settings.llmProvider === "gemma_local";
  const [modelpacks, setModelpacks] = useState<ModelpackManifest>(() => normalizeModelpackManifest(null));
  const [modelpackLoading, setModelpackLoading] = useState(false);
  const [modelpackError, setModelpackError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadModelpacks = async () => {
      setModelpackLoading(true);
      try {
        const manifest = await getModelpackManifest();
        if (cancelled) {
          return;
        }
        setModelpacks(normalizeModelpackManifest(manifest));
        setModelpackError(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setModelpacks(normalizeModelpackManifest(null));
        setModelpackError(error instanceof Error ? error.message : "Failed to load modelpack catalog.");
      } finally {
        if (!cancelled) {
          setModelpackLoading(false);
        }
      }
    };

    void loadModelpacks();
    return () => {
      cancelled = true;
    };
  }, []);

  const downloadableNow = useMemo(
    () => modelpacks.packs.filter((pack) => pack.availability !== "coming_soon" && Boolean(pack.download_url)).length,
    [modelpacks.packs],
  );

  const cycleProvider = () => {
    const order: ChatSettings["llmProvider"][] = ["local", "gemma_local", "gemini"];
    const idx = order.indexOf(settings.llmProvider);
    const next = order[(idx >= 0 ? idx + 1 : 0) % order.length];
    onUpdate({ ...settings, llmProvider: next });
  };

  const handleReset = () => {
    onUpdate({
      temperature: 0.6,
      topP: 0.95,
      maxTokens: 4096,
      stream: true,
      useRAG: true,
      llmProvider: "local",
      thinkingMode: true,
    });
  };

  return (
    <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-40 w-full max-w-lg fade-in">
      <div className="rounded-2xl bg-white border border-slate-200 backdrop-blur-2xl shadow-2xl shadow-slate-200/60 overflow-hidden max-h-[82vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-100">
          <h3 className="text-sm font-semibold text-slate-700">
            Generation Settings
          </h3>
          <div className="flex items-center gap-1">
            <button
              onClick={handleReset}
              className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
              title="Reset to defaults"
            >
              <RotateCcw size={14} />
            </button>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all duration-200"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="px-5 py-4 space-y-5 max-h-[70vh] overflow-y-auto">
          {/* Temperature */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-medium text-slate-600">
                Temperature
              </label>
              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-md">
                {settings.temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={settings.temperature}
              onChange={(e) =>
                onUpdate({ ...settings, temperature: parseFloat(e.target.value) })
              }
              className="w-full"
            />
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              <span>Precise (0)</span>
              <span className="text-indigo-500">Recommended: 0.6</span>
              <span>Creative (2)</span>
            </div>
          </div>

          {/* Top-P */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-medium text-slate-600">
                Top-P (Nucleus Sampling)
              </label>
              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-md">
                {settings.topP.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={settings.topP}
              onChange={(e) =>
                onUpdate({ ...settings, topP: parseFloat(e.target.value) })
              }
              className="w-full"
            />
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              <span>Focused (0)</span>
              <span>Diverse (1)</span>
            </div>
          </div>

          {/* Max Tokens */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <label className="text-xs font-medium text-slate-600">
                Max Tokens (Paid Models)
              </label>
              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded-md">
                {isLocalProvider ? "local-unlimited" : settings.maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="64"
              max="8192"
              step="64"
              value={settings.maxTokens}
              disabled={isLocalProvider}
              onChange={(e) =>
                onUpdate({
                  ...settings,
                  maxTokens: parseInt(e.target.value),
                })
              }
              className={`w-full ${isLocalProvider ? "opacity-50 cursor-not-allowed" : ""}`}
            />
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              {isLocalProvider ? (
                <span>Local model ignores max-token cap to favor full reasoning depth</span>
              ) : (
                <>
                  <span>64</span>
                  <span>8192</span>
                </>
              )}
            </div>
          </div>

          {/* Streaming toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <label className="text-xs font-medium text-slate-600">
                Streaming
              </label>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Show tokens as they are generated
              </p>
            </div>
            <button
              onClick={() =>
                onUpdate({ ...settings, stream: !settings.stream })
              }
              className={`relative h-6 w-11 rounded-full transition-all duration-300 ${
                settings.stream
                  ? "bg-indigo-600 shadow-sm shadow-indigo-200"
                  : "bg-slate-300"
              }`}
            >
              <span
                className={`absolute top-1 left-1 h-4 w-4 rounded-full bg-white transition-transform duration-200 shadow-sm ${
                  settings.stream ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* RAG Memory toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <label className="text-xs font-medium text-slate-600 flex items-center gap-1.5">
                <Brain size={12} className="text-indigo-500" />
                RAG Memory
              </label>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Enable retrieval-augmented generation with memory
              </p>
            </div>
            <button
              onClick={() =>
                onUpdate({ ...settings, useRAG: !settings.useRAG })
              }
              className={`relative h-6 w-11 rounded-full transition-all duration-300 ${
                settings.useRAG
                  ? "bg-emerald-500 shadow-sm shadow-emerald-200"
                  : "bg-slate-300"
              }`}
            >
              <span
                className={`absolute top-1 left-1 h-4 w-4 rounded-full bg-white transition-transform duration-200 shadow-sm ${
                  settings.useRAG ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* LLM Provider toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <label className="text-xs font-medium text-slate-600 flex items-center gap-1.5">
                <Zap size={12} className={settings.llmProvider === "gemini" ? "text-blue-500" : settings.llmProvider === "gemma_local" ? "text-emerald-500" : "text-violet-500"} />
                LLM Provider
              </label>
              <p className="text-[10px] text-slate-400 mt-0.5">
                {settings.llmProvider === "gemini"
                  ? "Using Google Gemini 2.5 Flash API"
                  : settings.llmProvider === "gemma_local"
                    ? "Using Gemma Local runtime path"
                    : "Using local Qwen3.5-9B-Opus reasoning model"}
              </p>
            </div>
            <button
              onClick={cycleProvider}
              className={`relative h-6 w-11 rounded-full transition-all duration-300 ${
                settings.llmProvider === "gemini"
                  ? "bg-blue-500 shadow-sm shadow-blue-200"
                  : settings.llmProvider === "gemma_local"
                    ? "bg-emerald-500 shadow-sm shadow-emerald-200"
                    : "bg-violet-500 shadow-sm shadow-violet-200"
              }`}
            >
              <span
                className={`absolute top-1 left-1 h-4 w-4 rounded-full bg-white transition-transform duration-200 shadow-sm ${
                  settings.llmProvider === "gemini"
                    ? "translate-x-5"
                    : settings.llmProvider === "gemma_local"
                      ? "translate-x-2.5"
                      : "translate-x-0"
                }`}
              />
            </button>
          </div>

          {/* Local modelpack download catalog */}
          <div className="pt-4 border-t border-slate-100">
            <div className="flex items-start justify-between gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600 flex items-center gap-1.5">
                  <Download size={12} className="text-emerald-600" />
                  Local Model Packs
                </label>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  Direct download links for LiteRT Gemma packs used by Gemma Local mode.
                </p>
              </div>
              <WebOfflineReadinessBadge ready={false} details={`${downloadableNow} downloadable now`} />
            </div>

            <div className="mt-3 space-y-2.5">
              {modelpacks.packs.map((pack) => {
                const available = pack.availability !== "coming_soon";
                return (
                  <div key={pack.id} className="rounded-xl border border-slate-200 bg-slate-50/80 p-3">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-xs font-semibold text-slate-700">{pack.display_name}</p>
                        <p className="text-[10px] text-slate-500 mt-0.5">
                          {pack.summary || "Model pack prepared for local runtime."}
                        </p>
                      </div>
                      <span
                        className={`text-[10px] font-semibold rounded-full px-2 py-0.5 border ${
                          available
                            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                            : "bg-amber-50 text-amber-700 border-amber-200"
                        }`}
                      >
                        {available ? "Ready" : "Coming Soon"}
                      </span>
                    </div>

                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      {available && pack.download_url ? (
                        <a
                          href={pack.download_url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-2.5 py-1 text-[10px] font-semibold text-white hover:bg-emerald-700 transition-colors"
                        >
                          <Download size={10} />
                          {pack.cta_label || "Download"}
                        </a>
                      ) : (
                        <button
                          type="button"
                          disabled
                          className="inline-flex items-center gap-1 rounded-lg bg-slate-200 px-2.5 py-1 text-[10px] font-semibold text-slate-500 cursor-not-allowed"
                        >
                          {pack.cta_label || "Coming Soon"}
                        </button>
                      )}

                      <span className="text-[10px] text-slate-500">v{pack.version}</span>
                      {pack.quantization ? (
                        <span className="text-[10px] text-slate-500 rounded-md border border-slate-300 px-1.5 py-0.5">
                          {pack.quantization}
                        </span>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>

            <a
              href={modelpacks.docs_url || MODELPACK_DOCS_FALLBACK}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1 text-[10px] text-indigo-600 hover:text-indigo-700"
            >
              LiteRT-LM Kotlin integration guide
              <ExternalLink size={11} />
            </a>

            {modelpackLoading ? (
              <p className="text-[10px] text-slate-400 mt-1">Refreshing modelpack catalog...</p>
            ) : null}

            {modelpackError ? (
              <p className="text-[10px] text-amber-600 mt-1">
                Catalog API unavailable, using fallback links.
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
