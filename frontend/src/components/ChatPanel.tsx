"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Square, Settings2, Sparkles, Brain, Zap, Lightbulb, Activity, Loader2, X } from "lucide-react";
import {
  ChatMessage as ChatMessageType,
  ModelStatus,
  ChatSettings,
  DEFAULT_SETTINGS,
  VoiceQueryResult,
  RuntimeTaskReferences,
  RuntimeTaskSnapshot,
} from "@/lib/types";
import {
  cancelRuntimeTask,
  getLLMProvider,
  getRuntimeTasks,
  ragChat,
  RAGStreamMeta,
  sendMessage,
  streamMessage,
  streamRAGMessage,
  subscribeRuntimeTaskEvents,
} from "@/lib/api";
import { MessageBubble } from "./MessageBubble";
import { SettingsPanel } from "./SettingsPanel";
import { EmptyState } from "./EmptyState";
import { VoiceQueryButton } from "./VoiceQueryButton";
import { LivePipelineVisualizer } from "./LivePipelineVisualizer";

interface Props {
  modelStatus: ModelStatus;
  conversationId: string;
  onTitleUpdate: (title: string) => void;
}

export function ChatPanel({ modelStatus, conversationId, onTitleUpdate }: Props) {
  const [messages, setMessages] = useState<ChatMessageType[]>(() => {
    // Restore messages from localStorage on mount
    try {
      const saved = localStorage.getItem(`cortex-messages-${conversationId}`);
      if (saved) return JSON.parse(saved);
    } catch { /* ignore */ }
    return [];
  });
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [error, setError] = useState<string | null>(null);
  const [localModelAvailable, setLocalModelAvailable] = useState(true);
  // Track whether the live pipeline visualizer should be visible.
  // The SSE connection is kept alive whenever RAG is enabled (via pipelineConnected),
  // but we only SHOW it when a request is in flight.
  const [showLivePipeline, setShowLivePipeline] = useState(false);
  // Keep SSE pre-connected when RAG is on so events are never missed
  const [pipelineConnected, setPipelineConnected] = useState(false);
  const [runtimeTasks, setRuntimeTasks] = useState<RuntimeTaskSnapshot[]>([]);
  const [runtimeTaskRefs, setRuntimeTaskRefs] = useState<RuntimeTaskReferences | null>(null);
  const [runtimeTaskStreamConnected, setRuntimeTaskStreamConnected] = useState(false);
  const [runtimeTaskError, setRuntimeTaskError] = useState<string | null>(null);
  const [cancellingTaskIds, setCancellingTaskIds] = useState<Record<string, boolean>>({});
  const isLocalProvider = settings.llmProvider === "local" || settings.llmProvider === "gemma_local";

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef(false);
  // Track if user has manually scrolled up — if so, don't force auto-scroll
  const userScrolledUpRef = useRef(false);

  // Auto-detect available LLM providers on mount
  useEffect(() => {
    getLLMProvider()
      .then((info) => {
        setLocalModelAvailable(info.local_model_loaded);
        if (!info.local_model_loaded && info.gemini_configured) {
          setSettings((prev) => ({ ...prev, llmProvider: "gemini" }));
        }
      })
      .catch(() => {});
  }, []);

  // Keep pipeline SSE pre-connected when RAG is enabled so we never miss events.
  // The LivePipelineVisualizer internally subscribes when isActive=true.
  useEffect(() => {
    setPipelineConnected(settings.useRAG);
  }, [settings.useRAG]);

  const upsertRuntimeTask = useCallback((task: RuntimeTaskSnapshot) => {
    setRuntimeTasks((prev) => {
      const idx = prev.findIndex((existing) => existing.task_id === task.task_id);
      if (idx < 0) {
        return [task, ...prev].slice(0, 64);
      }
      const next = [...prev];
      next[idx] = task;
      return next;
    });
  }, []);

  useEffect(() => {
    let mounted = true;

    const hydrateRuntimeTasks = async () => {
      try {
        const snapshot = await getRuntimeTasks();
        if (!mounted) return;
        setRuntimeTasks(snapshot.tasks || []);
      } catch {
        // Ignore hydration errors — SSE stream may still connect.
      }
    };

    hydrateRuntimeTasks();

    const controller = subscribeRuntimeTaskEvents(
      (event) => {
        if (!mounted) return;
        setRuntimeTaskStreamConnected(true);
        setRuntimeTaskError(null);
        upsertRuntimeTask(event.task);
      },
      (err) => {
        if (!mounted) return;
        setRuntimeTaskStreamConnected(false);
        setRuntimeTaskError(err.message);
      },
    );

    const fallbackPoll = setInterval(() => {
      getRuntimeTasks()
        .then((snapshot) => {
          if (!mounted) return;
          setRuntimeTasks(snapshot.tasks || []);
        })
        .catch(() => {
          // Keep SSE as the primary path; polling is best-effort fallback.
        });
    }, 15000);

    return () => {
      mounted = false;
      controller.abort();
      clearInterval(fallbackPoll);
    };
  }, [upsertRuntimeTask]);

  const handleCancelRuntimeTask = useCallback(async (taskId: string) => {
    setCancellingTaskIds((prev) => ({ ...prev, [taskId]: true }));
    try {
      await cancelRuntimeTask(taskId, "Cancelled from main chat runtime strip", true);
      setRuntimeTaskError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to cancel runtime task";
      setRuntimeTaskError(message);
    } finally {
      setCancellingTaskIds((prev) => ({ ...prev, [taskId]: false }));
    }
  }, []);

  // ── Batched streaming buffer (§10.1) ──
  // Accumulate tokens in a ref and flush to state every 50ms to reduce re-renders
  const streamBufferRef = useRef<Record<string, string>>({});
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flushStreamBuffer = useCallback(() => {
    const buffered = { ...streamBufferRef.current };
    streamBufferRef.current = {};
    flushTimerRef.current = null;
    if (Object.keys(buffered).length === 0) return;
    setMessages((prev) =>
      prev.map((m) =>
        buffered[m.id] !== undefined
          ? { ...m, content: m.content + buffered[m.id] }
          : m,
      ),
    );
  }, []);

  const appendToken = useCallback((assistantId: string, token: string) => {
    streamBufferRef.current[assistantId] =
      (streamBufferRef.current[assistantId] || "") + token;
    if (!flushTimerRef.current) {
      flushTimerRef.current = setTimeout(flushStreamBuffer, 50);
    }
  }, [flushStreamBuffer]);

  // Smart auto-scroll — only scroll down if user is near the bottom.
  // If user scrolled up to read thinking/pipeline, don't yank them down.
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Persist messages to localStorage (skip while streaming to avoid noise)
  useEffect(() => {
    const anyStreaming = messages.some((m) => m.isStreaming);
    if (!anyStreaming && messages.length > 0) {
      localStorage.setItem(
        `cortex-messages-${conversationId}`,
        JSON.stringify(messages),
      );
    }
  }, [messages, conversationId]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isGenerating) return;

    setError(null);
    abortRef.current = false;
    setRuntimeTaskRefs(null);

    // Add user message
    const userMsg: ChatMessageType = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsGenerating(true);
    // Reset scroll lock — user just sent a message, they want to see the response
    userScrolledUpRef.current = false;

    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    // Update conversation title from first message
    if (messages.length === 0) {
      onTitleUpdate(text.slice(0, 50) + (text.length > 50 ? "…" : ""));
    }

    // Prepare history
    const history = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Create assistant placeholder
    const assistantId = `assistant-${Date.now()}`;

    // ── RAG-Enhanced Mode ───────────────────────────────────────
    if (settings.useRAG) {
      // Show live pipeline visualizer (SSE already pre-connected via pipelineConnected)
      setShowLivePipeline(true);

      if (settings.stream) {
        // ── RAG + Streaming ─────────────────────────────────────
        const assistantMsg: ChatMessageType = {
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: Date.now(),
          isStreaming: true,
          isRAG: true,       // Flag to show live pipeline inside bubble
        };
        setMessages((prev) => [...prev, assistantMsg]);

        await streamRAGMessage(
          history,
          settings,
          conversationId, // session_id
          (meta: RAGStreamMeta) => {
            // Update message with RAG metadata (evidence, agents, etc.)
            if (meta.runtime_tasks) {
              setRuntimeTaskRefs(meta.runtime_tasks);
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      thinking: meta.thinking || m.thinking,
                      evidence: meta.evidence || m.evidence,
                      agentsUsed: meta.agents_used || m.agentsUsed,
                      confidence: meta.confidence ?? m.confidence,
                      queryAnalysis: meta.query_analysis || m.queryAnalysis,
                      pipelineTrace: meta.pipeline_trace || m.pipelineTrace,
                      runtimeTasks: meta.runtime_tasks || m.runtimeTasks,
                    }
                  : m,
              ),
            );
          },
          (token) => {
            if (abortRef.current) return;
            appendToken(assistantId, token);
          },
          () => {
            // Flush remaining tokens before marking stream complete
            if (flushTimerRef.current) {
              clearTimeout(flushTimerRef.current);
              flushTimerRef.current = null;
            }
            const remaining = streamBufferRef.current[assistantId] || "";
            streamBufferRef.current = {};
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== assistantId) return m;
                const finalContent = m.content + remaining;
                // Preserve the LLM's deep reasoning in message.thinking so it
                // persists across re-renders and localStorage restore. Without this,
                // message.thinking falls back to the orchestrator's brief note.
                let persistedThinking = m.thinking;
                if (finalContent.includes("<think>")) {
                  const tStart = finalContent.indexOf("<think>") + 7;
                  const tEnd = finalContent.indexOf("</think>");
                  if (tEnd > tStart) {
                    persistedThinking = finalContent.slice(tStart, tEnd).trim();
                  }
                }
                return { ...m, content: finalContent, isStreaming: false, thinking: persistedThinking };
              }),
            );
            setIsGenerating(false);
            // Keep live pipeline visible briefly after completion so user sees final state
            setTimeout(() => setShowLivePipeline(false), 2000);
          },
          (err) => {
            setError(err.message);
            setIsGenerating(false);
            setShowLivePipeline(false);
            setMessages((prev) =>
              prev.filter((m) => m.id !== assistantId || m.content.length > 0),
            );
          },
          // onReplace — server detected hallucination and sends corrected text
          // Preserve thinking content that was already streamed before the replacement
          (replacedText) => {
            streamBufferRef.current = {};
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== assistantId) return m;
                // Extract thinking from existing content before replacing
                let thinkingToKeep = m.thinking;
                if (m.content.includes("<think>")) {
                  const tStart = m.content.indexOf("<think>") + 7;
                  const tEnd = m.content.indexOf("</think>");
                  if (tEnd > tStart) {
                    thinkingToKeep = m.content.slice(tStart, tEnd).trim();
                  }
                }
                return { ...m, content: replacedText, thinking: thinkingToKeep };
              }),
            );
          },
        );
      } else {
        // ── RAG Non-streaming ───────────────────────────────────
        try {
          const res = await ragChat(history, settings, conversationId);
          const assistantMsg: ChatMessageType = {
            id: assistantId,
            role: "assistant",
            content: res.content,
            thinking: res.thinking || undefined,
            timestamp: Date.now(),
            isRAG: true,
            evidence: res.evidence,
            agentsUsed: res.agents_used,
            confidence: res.confidence,
            queryAnalysis: res.query_analysis,
            processingTimeMs: res.processing_time_ms,
            cacheHit: res.cache_hit,
            pipelineTrace: res.pipeline_trace,
            runtimeTasks: res.runtime_tasks,
          };
          setMessages((prev) => [...prev, assistantMsg]);
          if (res.runtime_tasks) {
            setRuntimeTaskRefs(res.runtime_tasks);
          }
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : "Unknown error";
          setError(message);
        } finally {
          setIsGenerating(false);
          setTimeout(() => setShowLivePipeline(false), 2000);
        }
      }
      return;
    }

    if (settings.stream) {
      // ── Streaming ─────────────────────────────────────────────
      const assistantMsg: ChatMessageType = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        isStreaming: true,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      await streamMessage(
        history,
        settings,
        (token) => {
          if (abortRef.current) return;
          appendToken(assistantId, token);
        },
        () => {
          // Flush remaining tokens before marking stream complete
          if (flushTimerRef.current) {
            clearTimeout(flushTimerRef.current);
            flushTimerRef.current = null;
          }
          const remaining = streamBufferRef.current[assistantId] || "";
          streamBufferRef.current = {};
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantId) return m;
              const finalContent = m.content + remaining;
              // Persist thinking from <think> tags so it survives re-renders
              let persistedThinking = m.thinking;
              if (finalContent.includes("<think>")) {
                const tStart = finalContent.indexOf("<think>") + 7;
                const tEnd = finalContent.indexOf("</think>");
                if (tEnd > tStart) {
                  persistedThinking = finalContent.slice(tStart, tEnd).trim();
                }
              }
              return { ...m, content: finalContent, isStreaming: false, thinking: persistedThinking };
            }),
          );
          setIsGenerating(false);
        },
        (err) => {
          setError(err.message);
          setIsGenerating(false);
          // Remove empty assistant message on error
          setMessages((prev) =>
            prev.filter((m) => m.id !== assistantId || m.content.length > 0),
          );
        },
      );
    } else {
      // ── Non-streaming ─────────────────────────────────────────
      try {
        const res = await sendMessage(history, settings);
        const assistantMsg: ChatMessageType = {
          id: assistantId,
          role: "assistant",
          content: res.content,
          thinking: res.thinking || undefined,
          timestamp: Date.now(),
          usage: res.usage,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
      } finally {
        setIsGenerating(false);
      }
    }
  }, [input, isGenerating, messages, settings, onTitleUpdate, conversationId, appendToken]);

  const handleStop = () => {
    abortRef.current = true;
    setIsGenerating(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleVoiceResult = useCallback((result: VoiceQueryResult) => {
    // Add user message (transcript)
    const userMsg: ChatMessageType = {
      id: `voice-user-${Date.now()}`,
      role: "user",
      content: `🎤 ${result.transcript}`,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);

    // Add assistant response
    const assistantMsg: ChatMessageType = {
      id: `voice-assistant-${Date.now()}`,
      role: "assistant",
      content: result.answer,
      timestamp: Date.now(),
      evidence: result.evidence,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    // Auto-play TTS if available
    if (result.audio_base64) {
      try {
        const binaryStr = atob(result.audio_base64);
        const bytes = new Uint8Array(binaryStr.length);
        for (let i = 0; i < binaryStr.length; i++) {
          bytes[i] = binaryStr.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        audio.onerror = () => URL.revokeObjectURL(url);
        audio.play().catch(() => URL.revokeObjectURL(url));
      } catch {
        // ignore playback errors
      }
    }

    // Update title
    if (messages.length === 0) {
      onTitleUpdate(
        "🎤 " + result.transcript.slice(0, 30) + (result.transcript.length > 30 ? "…" : "")
      );
    }
  }, [messages.length, onTitleUpdate]);

  const isOnline = modelStatus.model_loaded || modelStatus.model_info?.gemini_available === true;
  const activeTaskStates = new Set(["queued", "running", "waiting_approval", "blocked"]);
  const runtimeTaskById = new Map(runtimeTasks.map((task) => [task.task_id, task]));
  const referencedTaskIds = runtimeTaskRefs?.all_task_ids || [];
  const referencedTasks = referencedTaskIds
    .map((taskId) => runtimeTaskById.get(taskId))
    .filter((task): task is RuntimeTaskSnapshot => task !== undefined);
  const activeReferencedTasks = referencedTasks.filter((task) => activeTaskStates.has(task.state));
  const fallbackActiveTasks = runtimeTasks.filter((task) => activeTaskStates.has(task.state));
  const visibleRuntimeTasks = activeReferencedTasks.length > 0
    ? activeReferencedTasks
    : (isGenerating ? fallbackActiveTasks.slice(0, 4) : []);

  const runtimeTaskStateClass = (state: string): string => {
    switch (state) {
      case "running":
        return "border-blue-200 bg-blue-50 text-blue-700";
      case "queued":
        return "border-slate-200 bg-slate-100 text-slate-600";
      case "waiting_approval":
        return "border-amber-200 bg-amber-50 text-amber-700";
      case "blocked":
        return "border-violet-200 bg-violet-50 text-violet-700";
      case "completed":
        return "border-emerald-200 bg-emerald-50 text-emerald-700";
      case "failed":
        return "border-red-200 bg-red-50 text-red-700";
      case "cancelled":
        return "border-rose-200 bg-rose-50 text-rose-700";
      default:
        return "border-slate-200 bg-slate-50 text-slate-600";
    }
  };

  const showRuntimeTaskStrip = settings.useRAG && (
    visibleRuntimeTasks.length > 0
    || (Boolean(runtimeTaskRefs?.coordinator_task_id) && isGenerating)
    || runtimeTaskError !== null
  );

  return (
    <div className="flex flex-1 flex-col overflow-hidden relative">
      {/* Messages area */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-4 py-6"
        onScroll={() => {
          const el = scrollContainerRef.current;
          if (!el) return;
          // User is "near bottom" if within 150px of the bottom
          const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 150;
          userScrolledUpRef.current = !nearBottom;
        }}
      >
        <div className="mx-auto max-w-3xl space-y-1">
          {messages.length === 0 ? (
            <EmptyState
              isOnline={isOnline}
              onSuggestion={(text: string) => {
                setInput(text);
                inputRef.current?.focus();
              }}
            />
          ) : (
            <>
              {messages.map((msg, idx) => (
                <div key={msg.id}>
                  <MessageBubble message={msg} />
                  {/* Live Pipeline Visualizer — render right after the LAST message
                      (the streaming assistant bubble) so it's near the thinking panel,
                      not pushed below 20 evidence cards. SSE stays pre-connected. */}
                  {idx === messages.length - 1 && msg.role === "assistant" && pipelineConnected && (
                    <div className={`mt-2 mb-3 mx-auto max-w-2xl transition-all duration-500 ${
                      showLivePipeline ? 'opacity-100 max-h-[600px]' : 'opacity-0 max-h-0 overflow-hidden pointer-events-none'
                    }`}>
                      <LivePipelineVisualizer isActive={pipelineConnected} />
                    </div>
                  )}
                </div>
              ))}
            </>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error toast */}
      {error && (
        <div className="absolute bottom-28 left-1/2 -translate-x-1/2 z-50 fade-in">
          <div className="rounded-xl bg-red-500/[0.07] border border-red-500/20 px-4 py-2.5 text-sm text-red-400 backdrop-blur-xl flex items-center gap-2 shadow-2xl shadow-red-500/10">
            <span>⚠️ {error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-400/60 hover:text-red-300 ml-2 transition-colors"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onUpdate={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* Input area */}
      <div className="border-t border-slate-200/80 bg-white/90 backdrop-blur-2xl px-4 py-4">
        <div className="mx-auto max-w-3xl">
          {showRuntimeTaskStrip ? (
            <div
              data-testid="chat-runtime-task-strip"
              className="mb-2 rounded-xl border border-indigo-200 bg-indigo-50/60 px-3 py-2.5"
            >
              <div className="flex items-center gap-2 text-[11px] text-indigo-700">
                <Activity size={12} />
                <span className="font-semibold">Runtime Task Control</span>
                <span className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                  runtimeTaskStreamConnected
                    ? "bg-emerald-100 text-emerald-700 border border-emerald-200"
                    : "bg-slate-100 text-slate-600 border border-slate-200"
                }`}>
                  {runtimeTaskStreamConnected ? "LIVE" : "SYNC"}
                </span>
              </div>

              {runtimeTaskRefs?.coordinator_task_id ? (
                <div className="mt-1.5 text-[11px] text-slate-600 break-all">
                  Coordinator: <span className="font-mono text-slate-700">{runtimeTaskRefs.coordinator_task_id}</span>
                  <a
                    href={`/api/runtime/tasks/${encodeURIComponent(runtimeTaskRefs.coordinator_task_id)}`}
                    target="_blank"
                    rel="noreferrer"
                    className="ml-2 text-indigo-600 hover:text-indigo-700 underline"
                  >
                    open
                  </a>
                </div>
              ) : null}

              {visibleRuntimeTasks.length > 0 ? (
                <div className="mt-2 space-y-1.5">
                  {visibleRuntimeTasks.map((task) => {
                    const cancelling = !!cancellingTaskIds[task.task_id];
                    return (
                      <div
                        key={task.task_id}
                        data-testid={`chat-runtime-task-row-${task.task_id}`}
                        className="rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-[11px]"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-slate-700 break-all">{task.task_id}</span>
                          <span className={`ml-auto rounded-md border px-1.5 py-0.5 text-[10px] font-semibold ${runtimeTaskStateClass(task.state)}`}>
                            {task.state}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500">
                          <a
                            href={`/api/runtime/tasks/${encodeURIComponent(task.task_id)}`}
                            target="_blank"
                            rel="noreferrer"
                            className="text-indigo-600 hover:text-indigo-700 underline"
                          >
                            view task
                          </a>
                          <button
                            onClick={() => handleCancelRuntimeTask(task.task_id)}
                            disabled={cancelling}
                            data-testid={`chat-runtime-task-cancel-${task.task_id}`}
                            className="inline-flex items-center gap-1 rounded-md border border-red-300 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-700 hover:bg-red-100 disabled:opacity-60"
                          >
                            {cancelling ? <Loader2 size={10} className="animate-spin" /> : <X size={10} />}
                            cancel
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}

              {runtimeTaskError ? (
                <div className="mt-2 rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] text-amber-700">
                  {runtimeTaskError}
                </div>
              ) : null}
            </div>
          ) : null}

          <div className="glow-border rounded-2xl bg-white transition-all duration-300 border border-slate-200">
            <div className="flex items-end gap-2 p-3">
              {/* Settings button */}
              <button
                onClick={() => setShowSettings((p) => !p)}
                className={`flex-shrink-0 rounded-xl p-2.5 transition-all duration-200 ${
                  showSettings
                    ? "bg-indigo-50 text-indigo-600 shadow-sm shadow-indigo-100"
                    : "text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                }`}
                title="Settings"
              >
                <Settings2 size={18} />
              </button>

              {/* Voice query button */}
              <VoiceQueryButton
                onResult={handleVoiceResult}
                onError={(err) => setError(err)}
                disabled={!isOnline || isGenerating}
              />

              {/* Input */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  isOnline
                    ? "Ask Cortex Lab anything…"
                    : "Model is loading — please wait…"
                }
                disabled={!isOnline}
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-slate-800 placeholder:text-slate-400 outline-none disabled:opacity-40 py-2.5 leading-relaxed"
              />

              {/* Send / Stop */}
              {isGenerating ? (
                <button
                  onClick={handleStop}
                  className="flex-shrink-0 rounded-xl bg-red-50 p-2.5 text-red-500 hover:bg-red-100 border border-red-200 transition-all duration-200"
                  title="Stop generating"
                >
                  <Square size={18} />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || !isOnline}
                  className="flex-shrink-0 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 p-2.5 text-white transition-all duration-200 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-20 disabled:hover:from-indigo-600 disabled:hover:to-indigo-500 shadow-lg shadow-indigo-200/50"
                  title="Send message"
                >
                  <Send size={18} />
                </button>
              )}
            </div>

            {/* Bottom bar */}
            <div className="flex items-center justify-between border-t border-slate-100 px-4 py-2 text-[11px] text-slate-400">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <Sparkles
                    size={11}
                    className={
                      settings.llmProvider === "gemini"
                        ? "text-blue-500/60"
                        : settings.llmProvider === "gemma_local"
                          ? "text-emerald-500/60"
                          : "text-indigo-500/60"
                    }
                  />
                  <span>
                    {settings.llmProvider === "gemini"
                      ? "Gemini 2.5 Flash"
                      : settings.llmProvider === "gemma_local"
                        ? "Gemma Local"
                        : "Qwen3.5-9B Opus"}
                  </span>
                </div>
                <button
                  onClick={() => {
                    if (!localModelAvailable && settings.llmProvider === "gemini") return;
                    setSettings((prev) => {
                      const order: ChatSettings["llmProvider"][] = localModelAvailable
                        ? ["local", "gemma_local", "gemini"]
                        : ["gemini"];
                      const idx = order.indexOf(prev.llmProvider);
                      const next = order[(idx >= 0 ? idx + 1 : 0) % order.length];
                      return {
                        ...prev,
                        llmProvider: next,
                      };
                    });
                  }}
                  title={!localModelAvailable ? "Local model not loaded — Gemini only" : "Toggle LLM provider"}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-md transition-all text-[10px] font-medium ${
                    settings.llmProvider === "gemini"
                      ? "bg-blue-50 text-blue-600 border border-blue-200"
                      : settings.llmProvider === "gemma_local"
                        ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                        : "bg-violet-50 text-violet-600 border border-violet-200"
                  } ${!localModelAvailable ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  <Zap size={10} />
                  {settings.llmProvider === "gemini"
                    ? "GEMINI"
                    : settings.llmProvider === "gemma_local"
                      ? "GEMMA"
                      : "LOCAL"}
                </button>
                <button
                  onClick={() => setSettings((prev) => ({ ...prev, useRAG: !prev.useRAG }))}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-md transition-all text-[10px] font-medium ${
                    settings.useRAG
                      ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                      : "bg-slate-50 text-slate-400 border border-slate-200"
                  }`}
                >
                  <Brain size={10} />
                  {settings.useRAG ? "RAG ON" : "RAG OFF"}
                </button>
                <button
                  onClick={() => setSettings((prev) => ({ ...prev, thinkingMode: !prev.thinkingMode }))}
                  title={settings.thinkingMode ? "Thinking mode ON — model shows reasoning" : "Thinking mode OFF — faster responses"}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-md transition-all text-[10px] font-medium ${
                    settings.thinkingMode
                      ? "bg-amber-50 text-amber-600 border border-amber-200"
                      : "bg-slate-50 text-slate-400 border border-slate-200"
                  }`}
                >
                  <Lightbulb size={10} />
                  {settings.thinkingMode ? "THINK" : "FAST"}
                </button>
              </div>
              <span className="text-slate-400">
                Temp {settings.temperature} · Top-P {settings.topP} · Max{" "}
                {isLocalProvider ? "local-unlimited" : settings.maxTokens}
              </span>
            </div>
          </div>

          <p className="mt-2.5 text-center text-[10px] text-slate-400">
            Shift+Enter for new line · Enter to send · Model may produce inaccurate responses
          </p>
        </div>
      </div>
    </div>
  );
}
