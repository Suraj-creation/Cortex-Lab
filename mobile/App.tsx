import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  View,
  ScrollView,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import * as DocumentPicker from "expo-document-picker";

import {
  createApiClient,
  getDefaultApiBaseUrl,
  type PageIndexDocument,
  type PageIndexUsage,
  type RAGStreamMeta,
  type TTSStatus,
} from "./shared/core/api";
import type {
  AmbientConfig,
  ChatMessage,
  ChatSettings,
  ConversationRecord,
  ConversationTurn,
  ModelStatus,
  MemoryObject,
  GraphData,
  RAGStats,
  AmbientState,
  LivePipelineEvent,
  VoiceProviders,
} from "./shared/core/types";
import { DEFAULT_SETTINGS } from "./shared/core/types";
import {
  getAllConversations,
  getConversation,
  getCurrentConversationId,
  initializeStorage,
  loadChatSettings,
  saveChatSettings,
  saveConversation,
  saveCurrentConversationId,
} from "./shared/core/storage";

import { Header } from "./src/components/ui/Header";
import { BottomNav } from "./src/components/ui/BottomNav";
import { Button } from "./src/components/ui/Button";
import { Card } from "./src/components/ui/Card";
import { Badge } from "./src/components/ui/Badge";
import { TextInput } from "./src/components/ui/TextInput";
import { MessageBubble } from "./src/components/MessageBubble";
import PipelineTracesList from "./src/components/PipelineTracesList";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from "./src/theme/colors";

type ActiveView = "chat" | "memories" | "graph" | "dashboard" | "observability" | "ambient" | "documents";

interface ConversationSummary {
  id: string;
  title: string;
  timestamp: number;
}

const NAV_ITEMS = [
  { key: "chat" as const, label: "Chat" },
  { key: "memories" as const, label: "Memory" },
  { key: "graph" as const, label: "Graph" },
  { key: "dashboard" as const, label: "Dashboard" },
  { key: "observability" as const, label: "Observe" },
  { key: "ambient" as const, label: "Ambient" },
  { key: "documents" as const, label: "Docs" },
];

const STARTER_TEXT = "How can I help you today?";

const QUICK_PROMPTS = [
  "Summarize my latest project milestones",
  "What has changed in my beliefs this month?",
  "Show top memory themes from recent conversations",
  "What should I focus on next week?",
];

function formatRelativeTime(timestampMs: number | null): string {
  if (!timestampMs) return "Never";
  const normalizedTs = timestampMs < 1_000_000_000_000 ? timestampMs * 1000 : timestampMs;
  const diff = Date.now() - normalizedTs;
  if (diff < 5000) return "Just now";
  if (diff < 60000) return `${Math.round(diff / 1000)}s ago`;
  if (diff < 3600000) return `${Math.round(diff / 60000)}m ago`;
  return `${Math.round(diff / 3600000)}h ago`;
}

function toPercent(value: number | undefined): string {
  if (typeof value !== "number") return "0%";
  return `${Math.round(value * 100)}%`;
}

function shortNumber(value: number | undefined): string {
  if (typeof value !== "number") return "0";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return `${value}`;
}

function inferTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return "New Chat";
  const raw = firstUser.content.trim();
  return raw.length > 40 ? `${raw.slice(0, 40)}…` : raw;
}

function AppContent() {
  const apiBase = getDefaultApiBaseUrl();
  const api = useMemo(() => createApiClient({ baseUrl: apiBase }), [apiBase]);
  const apiHost = useMemo(() => {
    try {
      return new URL(apiBase).host;
    } catch {
      return apiBase;
    }
  }, [apiBase]);

  // ─── View & Navigation ──────────────────────────────────────────────
  const [activeView, setActiveView] = useState<ActiveView>("chat");
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    status: "loading",
    model_loaded: false,
    model_info: {},
  });
  const [globalError, setGlobalError] = useState<string>("");

  // ─── Chat State ─────────────────────────────────────────────────────
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [localModelAvailable, setLocalModelAvailable] = useState(true);
  const [providerBusy, setProviderBusy] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const messagesRef = useRef<ChatMessage[]>(messages);

  // ─── Other Views State ──────────────────────────────────────────────
  const [memories, setMemories] = useState<MemoryObject[]>([]);
  const [memorySearch, setMemorySearch] = useState("");
  const [memoryDraft, setMemoryDraft] = useState("");
  const [memoryBusy, setMemoryBusy] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [ragStats, setRagStats] = useState<RAGStats | null>(null);
  const [observabilityMetrics, setObservabilityMetrics] = useState<Record<string, unknown> | null>(null);
  const [pipelineEvents, setPipelineEvents] = useState<LivePipelineEvent[]>([]);
  const [ambientState, setAmbientState] = useState<AmbientState | null>(null);
  const [ambientConfig, setAmbientConfig] = useState<AmbientConfig | null>(null);
  const [ambientEnrollment, setAmbientEnrollment] = useState<{ enrolled: boolean; speaker_id_available?: boolean } | null>(null);
  const [ambientProviders, setAmbientProviders] = useState<VoiceProviders | null>(null);
  const [ambientTurns, setAmbientTurns] = useState<ConversationTurn[]>([]);
  const [ambientConversations, setAmbientConversations] = useState<ConversationRecord[]>([]);
  const [ambientTTSStatus, setAmbientTTSStatus] = useState<TTSStatus | null>(null);
  const [ttsDraft, setTtsDraft] = useState("Hey, this is Cortex speaking. Voice synthesis is online.");
  const [ttsBusy, setTtsBusy] = useState(false);
  const [ttsLastBytes, setTtsLastBytes] = useState<number | null>(null);
  const [ambientLastUpdatedAt, setAmbientLastUpdatedAt] = useState<number | null>(null);
  const [ambientBusy, setAmbientBusy] = useState(false);
  const [documents, setDocuments] = useState<PageIndexDocument[]>([]);
  const [pageIndexUsage, setPageIndexUsage] = useState<PageIndexUsage | null>(null);
  const [pageIndexEnabled, setPageIndexEnabled] = useState<boolean | null>(null);
  const [documentQuery, setDocumentQuery] = useState("");
  const [documentQueryBusy, setDocumentQueryBusy] = useState(false);
  const [documentAnswer, setDocumentAnswer] = useState("");
  const [documentSections, setDocumentSections] = useState<
    { page: number; content: string; doc_id: string; score: number }[]
  >([]);
  const [documentTreeDocId, setDocumentTreeDocId] = useState<string | null>(null);
  const [documentTreePreview, setDocumentTreePreview] = useState<string[]>([]);
  const [documentsBusy, setDocumentsBusy] = useState(false);
  const [loadingView, setLoadingView] = useState(false);

  // ─── Setup ──────────────────────────────────────────────────────────
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const refreshModelStatus = useCallback(async () => {
    try {
      const status = await api.getModelStatus();
      setModelStatus(status);
    } catch {
      setModelStatus({
        status: "offline",
        model_loaded: false,
        model_info: {},
      });
    }
  }, [api]);

  const syncLLMProvider = useCallback(async () => {
    try {
      const provider = await api.getLLMProvider();
      setLocalModelAvailable(provider.local_model_loaded);
      setSettings((prev) => {
        if (prev.llmProvider === provider.provider) {
          if (provider.provider === "local" && !provider.local_model_loaded && provider.gemini_configured) {
            return { ...prev, llmProvider: "gemini" };
          }
          return prev;
        }

        if (provider.provider === "local" && !provider.local_model_loaded && provider.gemini_configured) {
          return { ...prev, llmProvider: "gemini" };
        }

        return { ...prev, llmProvider: provider.provider };
      });
    } catch {
      // Keep existing provider when backend provider endpoint is unavailable.
    }
  }, [api]);

  const loadConversations = useCallback(async () => {
    try {
      const all = await getAllConversations();
      const sorted = [...all].sort((a, b) => b.timestamp - a.timestamp);
      const summary = sorted.map((c) => ({
        id: c.id,
        title: c.title || inferTitle(c.messages),
        timestamp: c.timestamp,
      }));
      setConversations(summary);
      return summary;
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
      return [];
    }
  }, []);

  const ensureActiveConversation = useCallback(async () => {
    const summary = await loadConversations();
    const saved = await getCurrentConversationId();

    if (saved && summary.some((c) => c.id === saved)) {
      setActiveConversationId(saved);
      return;
    }

    if (summary.length > 0) {
      setActiveConversationId(summary[0].id);
      await saveCurrentConversationId(summary[0].id);
      return;
    }

    const id = `conv_${Date.now()}`;
    const starter: ChatMessage[] = [
      {
        id: "starter",
        role: "assistant",
        content: STARTER_TEXT,
        timestamp: Date.now(),
      },
    ];
    await saveConversation(starter, id, "New Chat");
    await saveCurrentConversationId(id);
    setConversations([{ id, title: "New Chat", timestamp: Date.now() }]);
    setActiveConversationId(id);
  }, [loadConversations]);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        await initializeStorage();
        const loadedSettings = await loadChatSettings();
        if (mounted) setSettings(loadedSettings);
        await Promise.all([refreshModelStatus(), syncLLMProvider(), ensureActiveConversation()]);
      } catch (error) {
        if (mounted) setGlobalError(error instanceof Error ? error.message : String(error));
      }
    })();

    const interval = setInterval(() => void refreshModelStatus(), 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [refreshModelStatus, syncLLMProvider, ensureActiveConversation]);

  useEffect(() => {
    void saveChatSettings(settings);
  }, [settings]);

  useEffect(() => {
    if (!activeConversationId) return;

    let mounted = true;
    (async () => {
      try {
        const conv = await getConversation(activeConversationId);
        if (!mounted) return;
        if (conv?.messages?.length) {
          setMessages(conv.messages);
        } else {
          setMessages([
            {
              id: "starter",
              role: "assistant",
              content: STARTER_TEXT,
              timestamp: Date.now(),
            },
          ]);
        }
      } catch (error) {
        if (mounted) {
          setGlobalError(error instanceof Error ? error.message : String(error));
        }
      }
    })();

    return () => {
      mounted = false;
    };
  }, [activeConversationId]);

  // ─── Chat Operations ────────────────────────────────────────────────
  const saveActiveConversation = useCallback(
    async (nextMessages: ChatMessage[]) => {
      if (!activeConversationId) return;
      const title = inferTitle(nextMessages);
      await saveConversation(nextMessages, activeConversationId, title);
      setConversations((prev) => {
        const existing = prev.find((c) => c.id === activeConversationId);
        const next: ConversationSummary = {
          id: activeConversationId,
          title,
          timestamp: Date.now(),
        };
        if (!existing) return [next, ...prev];
        return [next, ...prev.filter((c) => c.id !== activeConversationId)];
      });
    },
    [activeConversationId],
  );

  const createNewConversation = useCallback(async () => {
    const id = `conv_${Date.now()}`;
    const starter: ChatMessage[] = [
      {
        id: "starter",
        role: "assistant",
        content: STARTER_TEXT,
        timestamp: Date.now(),
      },
    ];
    await saveConversation(starter, id, "New Chat");
    await saveCurrentConversationId(id);
    setConversations((prev) => [{ id, title: "New Chat", timestamp: Date.now() }, ...prev]);
    setActiveConversationId(id);
    setMessages(starter);
  }, []);

  const toggleProvider = useCallback(async () => {
    const nextProvider = settings.llmProvider === "local" ? "gemini" : "local";
    if (nextProvider === "local" && !localModelAvailable) {
      setGlobalError("Local model is unavailable. Switch back after local model loads.");
      return;
    }

    setProviderBusy(true);
    try {
      await api.setLLMProvider(nextProvider);
      setSettings((prev) => ({ ...prev, llmProvider: nextProvider }));
      setGlobalError("");
      await refreshModelStatus();
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setProviderBusy(false);
    }
  }, [api, localModelAvailable, refreshModelStatus, settings.llmProvider]);

  const toggleRAG = useCallback(() => {
    setSettings((prev) => ({ ...prev, useRAG: !prev.useRAG }));
  }, []);

  const sendChat = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !activeConversationId) return;

    setGlobalError("");
    setSending(true);

    const userMessage: ChatMessage = {
      id: `u_${Date.now()}`,
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    const assistantId = `a_${Date.now()}`;
    const baseMessages = [...messagesRef.current, userMessage];
    setInput("");
    setMessages([...baseMessages]);

    try {
      if (settings.stream) {
        let assistantContent = "";
        let latestMeta: RAGStreamMeta | null = null;

        setStreamingMessageId(assistantId);
        setMessages([
          ...baseMessages,
          {
            id: assistantId,
            role: "assistant",
            content: "",
            timestamp: Date.now(),
            isStreaming: true,
          },
        ]);

        await api.streamMessage(
          {
            messages: baseMessages.map((m) => ({ role: m.role, content: m.content })),
            settings,
          },
          {
            onToken: (token) => {
              assistantContent += token;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: assistantContent, isStreaming: true } : m,
                ),
              );
            },
            onMeta: (meta) => {
              latestMeta = { ...latestMeta, ...meta };
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        thinking: meta.thinking ?? m.thinking,
                        evidence: meta.evidence ?? m.evidence,
                        agentsUsed: meta.agents_used ?? m.agentsUsed,
                        confidence: typeof meta.confidence === "number" ? meta.confidence : m.confidence,
                        queryAnalysis: meta.query_analysis ?? m.queryAnalysis,
                        pipelineTrace: meta.pipeline_trace ?? m.pipelineTrace,
                      }
                    : m,
                ),
              );
            },
            onReplace: (replacement) => {
              assistantContent = replacement;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: replacement,
                        isStreaming: true,
                      }
                    : m,
                ),
              );
            },
            onDone: () => {
              setMessages((prev) => {
                const finalized = prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: assistantContent,
                        isStreaming: false,
                        thinking: latestMeta?.thinking ?? m.thinking,
                        evidence: latestMeta?.evidence ?? m.evidence,
                        agentsUsed: latestMeta?.agents_used ?? m.agentsUsed,
                        confidence:
                          typeof latestMeta?.confidence === "number"
                            ? latestMeta.confidence
                            : m.confidence,
                        queryAnalysis: latestMeta?.query_analysis ?? m.queryAnalysis,
                        pipelineTrace: latestMeta?.pipeline_trace ?? m.pipelineTrace,
                      }
                    : m,
                );
                void saveActiveConversation(finalized);
                return finalized;
              });
            },
            onError: (error) => {
              setGlobalError(error.message);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, isStreaming: false } : m,
                ),
              );
            },
          },
        );
      } else {
        const completion = await api.sendMessage({
          messages: baseMessages.map((m) => ({ role: m.role, content: m.content })),
          settings,
        });

        const assistantMessage: ChatMessage = {
          id: assistantId,
          role: "assistant",
          content: completion.content,
          thinking: completion.thinking,
          evidence: completion.evidence,
          agentsUsed: completion.agents_used,
          confidence: completion.confidence,
          queryAnalysis: completion.query_analysis,
          processingTimeMs: completion.processing_time_ms,
          cacheHit: completion.cache_hit,
          pipelineTrace: completion.pipeline_trace || undefined,
          timestamp: Date.now(),
        };
        const finalized = [...baseMessages, assistantMessage];
        setMessages(finalized);
        await saveActiveConversation(finalized);
      }
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setStreamingMessageId(null);
      setSending(false);
    }
  }, [activeConversationId, api, input, saveActiveConversation, sending, settings]);

  // ─── Data Loading ───────────────────────────────────────────────────
  const loadMemories = useCallback(async () => {
    setLoadingView(true);
    try {
      const data = await api.getMemories(50, 0);
      setMemories(data.memories || []);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingView(false);
    }
  }, [api]);

  const searchMemories = useCallback(async () => {
    if (!memorySearch.trim()) {
      await loadMemories();
      return;
    }

    setMemoryBusy(true);
    try {
      const data = await api.searchMemories(memorySearch.trim(), 20);
      setMemories(data.results || []);
      setGlobalError("");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setMemoryBusy(false);
    }
  }, [api, loadMemories, memorySearch]);

  const addMemory = useCallback(async () => {
    if (!memoryDraft.trim()) return;

    setMemoryBusy(true);
    try {
      await api.ingestMemory(memoryDraft.trim(), "mobile");
      setMemoryDraft("");
      await loadMemories();
      setGlobalError("");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setMemoryBusy(false);
    }
  }, [api, loadMemories, memoryDraft]);

  const removeMemory = useCallback(
    async (memoryId: string) => {
      setMemoryBusy(true);
      try {
        await api.deleteMemory(memoryId);
        setMemories((prev) => prev.filter((mem) => mem.id !== memoryId));
        setGlobalError("");
      } catch (error) {
        setGlobalError(error instanceof Error ? error.message : String(error));
      } finally {
        setMemoryBusy(false);
      }
    },
    [api],
  );

  const loadGraph = useCallback(async () => {
    setLoadingView(true);
    try {
      const graph = await api.getGraphData();
      setGraphData(graph);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingView(false);
    }
  }, [api]);

  const loadDashboard = useCallback(async () => {
    setLoadingView(true);
    try {
      const stats = await api.getRAGStats();
      setRagStats(stats);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingView(false);
    }
  }, [api]);

  const loadObservability = useCallback(async (silent: boolean = false) => {
    if (!silent) {
      setLoadingView(true);
    }
    try {
      const metrics = await api.getObservabilityMetrics();
      setObservabilityMetrics(metrics);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      if (!silent) {
        setLoadingView(false);
      }
    }
  }, [api]);

  const loadAmbient = useCallback(async (silent: boolean = false) => {
    if (!silent) {
      setLoadingView(true);
    }
    try {
      const [
        statusResult,
        configResult,
        providersResult,
        enrollmentResult,
        transcriptResult,
        conversationsResult,
        ttsStatusResult,
      ] = await Promise.allSettled([
        api.getAmbientStatus(),
        api.getAmbientConfig(),
        api.getVoiceProviders(),
        api.getEnrollmentStatus(),
        api.getLiveTranscript(),
        api.getConversations(20, 0),
        api.getTTSStatus(),
      ]);

      if (statusResult.status === "fulfilled") {
        setAmbientState(statusResult.value);
      } else {
        throw statusResult.reason;
      }

      setAmbientConfig(configResult.status === "fulfilled" ? configResult.value : null);
      setAmbientProviders(providersResult.status === "fulfilled" ? providersResult.value : null);
      setAmbientEnrollment(enrollmentResult.status === "fulfilled" ? enrollmentResult.value : null);
      setAmbientTurns(transcriptResult.status === "fulfilled" ? transcriptResult.value.turns || [] : []);
      setAmbientConversations(
        conversationsResult.status === "fulfilled" ? conversationsResult.value.conversations || [] : [],
      );
      setAmbientTTSStatus(ttsStatusResult.status === "fulfilled" ? ttsStatusResult.value : null);
      setAmbientLastUpdatedAt(Date.now());
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      if (!silent) {
        setLoadingView(false);
      }
    }
  }, [api]);

  const runAmbientAction = useCallback(
    async (action: "start" | "stop" | "pause" | "resume") => {
      setAmbientBusy(true);
      try {
        await api.ambientAction(action);
        await loadAmbient(true);
      } catch (error) {
        setGlobalError(error instanceof Error ? error.message : String(error));
      } finally {
        setAmbientBusy(false);
      }
    },
    [api, loadAmbient],
  );

  const setAmbientProvider = useCallback(
    async (kind: "stt" | "tts", provider: "traditional" | "gemini") => {
      setAmbientBusy(true);
      try {
        if (kind === "stt") {
          await api.setSTTProvider(provider);
        } else {
          await api.setTTSProvider(provider);
        }
        await loadAmbient(true);
        setGlobalError("");
      } catch (error) {
        setGlobalError(error instanceof Error ? error.message : String(error));
      } finally {
        setAmbientBusy(false);
      }
    },
    [api, loadAmbient],
  );

  const startAmbientEnrollment = useCallback(async () => {
    setAmbientBusy(true);
    try {
      await api.startEnrollment(20);
      await loadAmbient(true);
      setGlobalError("");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setAmbientBusy(false);
    }
  }, [api, loadAmbient]);

  const toggleAmbientAutoIngest = useCallback(async () => {
    if (!ambientConfig) return;

    setAmbientBusy(true);
    try {
      const updated = await api.updateAmbientConfig({
        auto_ingest: !ambientConfig.auto_ingest,
      });
      setAmbientConfig(updated);
      await loadAmbient(true);
      setGlobalError("");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setAmbientBusy(false);
    }
  }, [ambientConfig, api, loadAmbient]);

  const runTTSHealthCheck = useCallback(async () => {
    const text = ttsDraft.trim();
    if (!text) return;

    setTtsBusy(true);
    try {
      const audioBytes = await api.synthesizeSpeech(text);
      setTtsLastBytes(audioBytes.byteLength);
      setGlobalError("");
      await loadAmbient(true);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setTtsBusy(false);
    }
  }, [api, loadAmbient, ttsDraft]);

  const loadDocuments = useCallback(async () => {
    setLoadingView(true);
    try {
      const [docs, usage] = await Promise.all([
        api.listDocuments(),
        api.getPageIndexUsage(),
      ]);
      setDocuments(docs.documents || []);
      setPageIndexEnabled(Boolean(docs.pageindex_enabled));
      setPageIndexUsage(usage.usage || null);
      if (documentTreeDocId && !(docs.documents || []).some((doc) => doc.doc_id === documentTreeDocId)) {
        setDocumentTreeDocId(null);
        setDocumentTreePreview([]);
      }
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoadingView(false);
    }
  }, [api, documentTreeDocId]);

  const uploadDocument = useCallback(async () => {
    try {
      setDocumentsBusy(true);

      const picked = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
        copyToCacheDirectory: true,
        multiple: false,
      });

      if (picked.canceled || picked.assets.length === 0) {
        return;
      }

      const asset = picked.assets[0] as DocumentPicker.DocumentPickerAsset & { file?: Blob };
      const form = new FormData();

      if (asset.file) {
        form.append("file", asset.file, asset.name || "document.pdf");
      } else {
        form.append(
          "file",
          {
            uri: asset.uri,
            name: asset.name || "document.pdf",
            type: "application/pdf",
          } as unknown as Blob,
        );
      }

      await api.uploadDocument(form);
      await loadDocuments();
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setDocumentsBusy(false);
    }
  }, [api, loadDocuments]);

  const deleteDocument = useCallback(
    async (docId: string) => {
      try {
        setDocumentsBusy(true);
        await api.deleteDocument(docId);
        if (documentTreeDocId === docId) {
          setDocumentTreeDocId(null);
          setDocumentTreePreview([]);
        }
        await loadDocuments();
      } catch (error) {
        setGlobalError(error instanceof Error ? error.message : String(error));
      } finally {
        setDocumentsBusy(false);
      }
    },
    [api, documentTreeDocId, loadDocuments],
  );

  const toggleDocumentTree = useCallback(
    async (docId: string) => {
      if (documentTreeDocId === docId) {
        setDocumentTreeDocId(null);
        setDocumentTreePreview([]);
        return;
      }

      setDocumentsBusy(true);
      try {
        const treeResponse = await api.getDocumentTree(docId);
        const treePayload = treeResponse.tree as { result?: Array<Record<string, unknown>> } | null;
        const preview = (treePayload?.result || [])
          .slice(0, 10)
          .map((node, index) => {
            const title = typeof node.title === "string" ? node.title : `Section ${index + 1}`;
            const summary = typeof node.summary === "string"
              ? node.summary
              : typeof node.text === "string"
                ? node.text
                : "";
            const page = typeof node.page_index === "number" ? `p.${node.page_index}` : "";
            return `${page ? `${page} · ` : ""}${title}${summary ? ` — ${summary.slice(0, 100)}` : ""}`;
          });
        setDocumentTreeDocId(docId);
        setDocumentTreePreview(preview);
        setGlobalError("");
      } catch (error) {
        setGlobalError(error instanceof Error ? error.message : String(error));
      } finally {
        setDocumentsBusy(false);
      }
    },
    [api, documentTreeDocId],
  );

  const runDocumentQuery = useCallback(async () => {
    const query = documentQuery.trim();
    if (!query) return;

    setDocumentQueryBusy(true);
    try {
      const result = await api.queryDocuments(query, 5);
      setDocumentAnswer(result.answer || "");
      setDocumentSections(result.sections || []);
      setGlobalError("");
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      setDocumentQueryBusy(false);
    }
  }, [api, documentQuery]);

  useEffect(() => {
    if (activeView === "memories") void loadMemories();
    else if (activeView === "graph") void loadGraph();
    else if (activeView === "dashboard") void loadDashboard();
    else if (activeView === "observability") void loadObservability();
    else if (activeView === "ambient") void loadAmbient();
    else if (activeView === "documents") void loadDocuments();
  }, [
    activeView,
    loadMemories,
    loadGraph,
    loadDashboard,
    loadObservability,
    loadAmbient,
    loadDocuments,
  ]);

  useEffect(() => {
    if (activeView !== "observability") return;

    setPipelineEvents([]);

    if (Platform.OS !== "web") {
      const nativeInterval = setInterval(() => {
        void loadObservability(true);
      }, 4000);
      return () => clearInterval(nativeInterval);
    }

    const controller = api.subscribePipelineEvents(
      (event) => {
        setPipelineEvents((prev) => [event, ...prev].slice(0, 60));
      },
      (error) => {
        setGlobalError(error.message);
      },
    );

    const interval = setInterval(() => {
      void loadObservability(true);
    }, 6000);

    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, [activeView, api, loadObservability]);

  useEffect(() => {
    if (activeView !== "ambient") return;

    const interval = setInterval(() => {
      void loadAmbient(true);
    }, 2500);

    return () => clearInterval(interval);
  }, [activeView, loadAmbient]);

  // ─── Render Views ───────────────────────────────────────────────────
  const renderChatView = () => (
    <View style={styles.viewContainer}>
      <View style={styles.chatTopShell}>
        <View style={styles.conversationBar}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.conversationList}
          >
            <Button
              label="New"
              variant="primary"
              size="sm"
              onPress={createNewConversation}
              style={styles.newChatButton}
            />
            {conversations.map((conv) => (
              <Pressable
                key={conv.id}
                onPress={() => {
                  setActiveConversationId(conv.id);
                  void saveCurrentConversationId(conv.id);
                }}
                style={[
                  styles.convChip,
                  conv.id === activeConversationId && styles.convChipActive,
                ]}
              >
                <Text
                  style={[
                    styles.convChipText,
                    conv.id === activeConversationId && styles.convChipTextActive,
                  ]}
                  numberOfLines={1}
                >
                  {conv.title}
                </Text>
              </Pressable>
            ))}
          </ScrollView>
        </View>

        <View style={styles.chatControlRow}>
          <Pressable
            onPress={() => void toggleProvider()}
            disabled={providerBusy || (settings.llmProvider === "gemini" && !localModelAvailable)}
            style={[
              styles.controlPill,
              settings.llmProvider === "gemini" ? styles.controlPillBlue : styles.controlPillViolet,
              (providerBusy || (settings.llmProvider === "gemini" && !localModelAvailable)) && styles.controlPillDisabled,
            ]}
          >
            <Text style={styles.controlPillText}>
              {providerBusy
                ? "Switching…"
                : settings.llmProvider === "gemini"
                  ? "Gemini"
                  : "Local"}
            </Text>
          </Pressable>

          <Pressable
            onPress={toggleRAG}
            style={[
              styles.controlPill,
              settings.useRAG ? styles.controlPillGreen : styles.controlPillNeutral,
            ]}
          >
            <Text style={styles.controlPillText}>{settings.useRAG ? "RAG On" : "RAG Off"}</Text>
          </Pressable>

          <Pressable
            onPress={() => {
              setSettings((prev) => ({ ...prev, stream: !prev.stream }));
            }}
            style={[
              styles.controlPill,
              settings.stream ? styles.controlPillAmber : styles.controlPillNeutral,
            ]}
          >
            <Text style={styles.controlPillText}>{settings.stream ? "Stream On" : "Batch"}</Text>
          </Pressable>
        </View>

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.quickPromptList}
        >
          {QUICK_PROMPTS.map((prompt) => (
            <Pressable
              key={prompt}
              onPress={() => setInput(prompt)}
              style={styles.quickPromptChip}
            >
              <Text numberOfLines={1} style={styles.quickPromptText}>{prompt}</Text>
            </Pressable>
          ))}
        </ScrollView>
      </View>

      <FlatList
        data={messages}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.messagesList}
        renderItem={({ item }) => (
          <MessageBubble
            role={item.role}
            content={item.content}
            timestamp={item.timestamp}
            isStreaming={Boolean(item.isStreaming)}
            thinking={item.thinking}
            confidence={item.confidence}
            agentsUsed={item.agentsUsed}
            evidence={item.evidence}
            queryAnalysis={item.queryAnalysis}
          />
        )}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>Start a conversation</Text>
            <Text style={styles.emptyBody}>Ask Cortex Lab anything…</Text>
          </View>
        }
      />

      {globalError ? (
        <Card variant="outlined" style={styles.errorBanner}>
          <Text style={styles.errorText}>{globalError}</Text>
        </Card>
      ) : null}

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        keyboardVerticalOffset={SPACING.lg}
      >
        <View style={styles.inputArea}>
          <TextInput
            placeholder="Message Cortex Lab…"
            value={input}
            onChangeText={setInput}
            multiline
            editable={!sending}
          />
          <Button
            label={sending ? (streamingMessageId ? "Live" : "Sending") : "Send"}
            variant="primary"
            size="md"
            disabled={!input.trim() || sending}
            onPress={sendChat}
            style={styles.sendButton}
          />
        </View>
        <View style={styles.inputMetaRow}>
          <Text style={styles.inputMetaText}>
            {settings.llmProvider.toUpperCase()} · {settings.useRAG ? "RAG" : "No RAG"} · {settings.stream ? "Stream" : "Batch"} · Temp {settings.temperature} · Top-P {settings.topP} · Max {settings.maxTokens}
          </Text>
        </View>
      </KeyboardAvoidingView>
    </View>
  );

  const renderMemoriesView = () => (
    <ScrollView style={styles.viewContainer} contentContainerStyle={[styles.viewPadding, styles.viewBottomSpace]}>
      <Text style={styles.viewTitle}>Memory Browser</Text>
      <Text style={styles.viewSubtitle}>{memories.length} memories stored</Text>

      <View style={styles.memoryToolbar}>
        <TextInput
          placeholder="Search memories…"
          value={memorySearch}
          onChangeText={setMemorySearch}
          style={styles.memorySearchInput}
        />
        <Button
          label={memoryBusy ? "…" : "Search"}
          size="sm"
          variant="outline"
          onPress={() => void searchMemories()}
          disabled={memoryBusy}
        />
      </View>

      <Card variant="outlined" style={styles.memoryComposerCard}>
        <Text style={styles.memoryComposerTitle}>Add a Memory</Text>
        <TextInput
          placeholder="Store an important thought, decision, or event…"
          value={memoryDraft}
          onChangeText={setMemoryDraft}
          multiline
          style={styles.memoryComposerInput}
        />
        <View style={styles.memoryComposerActions}>
          <Button
            label={memoryBusy ? "Saving…" : "Save Memory"}
            size="sm"
            onPress={() => void addMemory()}
            disabled={memoryBusy || !memoryDraft.trim()}
          />
          <Button
            label="Reload"
            size="sm"
            variant="secondary"
            onPress={() => void loadMemories()}
            disabled={memoryBusy}
          />
        </View>
      </Card>

      {loadingView ? (
        <ActivityIndicator color={COLORS.primary[500]} size="large" style={styles.loader} />
      ) : memories.length > 0 ? (
        <View style={styles.memoriesList}>
          {memories.map((mem) => (
            <Card key={mem.id} variant="outlined" style={styles.memoryCard}>
              <View style={styles.memoryMetaRow}>
                <Badge label={mem.memory_type || "episodic"} variant="primary" small />
                <Badge label={mem.emotion || "neutral"} variant="info" small />
                <Text style={styles.memoryScoreText}>imp {(mem.importance * 100).toFixed(0)}%</Text>
              </View>
              <Text style={styles.memoryText}>{mem.content}</Text>
              <View style={styles.memoryFooterRow}>
                <Badge
                  label={new Date(mem.timestamp).toLocaleDateString()}
                  variant="info"
                  small
                  style={styles.memoryDate}
                />
                <Pressable
                  onPress={() => void removeMemory(mem.id)}
                  style={styles.memoryDeleteButton}
                >
                  <Text style={styles.memoryDeleteText}>Delete</Text>
                </Pressable>
              </View>
            </Card>
          ))}
        </View>
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No memories yet</Text>
          <Text style={styles.emptyBody}>Memories will appear here as you build them.</Text>
        </View>
      )}
    </ScrollView>
  );

  const renderGraphView = () => (
    <ScrollView style={styles.viewContainer} contentContainerStyle={[styles.viewPadding, styles.viewBottomSpace]}>
      <Text style={styles.viewTitle}>Knowledge Graph</Text>

      {loadingView ? (
        <ActivityIndicator color={COLORS.primary[500]} size="large" style={styles.loader} />
      ) : graphData ? (
        (() => {
          const typeCounts = Object.entries(
            graphData.nodes.reduce<Record<string, number>>((acc, node) => {
              const key = node.type || "unknown";
              acc[key] = (acc[key] || 0) + 1;
              return acc;
            }, {}),
          ).sort((a, b) => b[1] - a[1]);

          const strongestEdges = [...graphData.edges]
            .sort((a, b) => (b.weight || 0) - (a.weight || 0))
            .slice(0, 10);

          return (
        <>
          <View style={styles.metricGrid}>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Nodes</Text>
              <Text style={styles.metricValue}>{graphData.nodes.length}</Text>
            </Card>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Edges</Text>
              <Text style={styles.metricValue}>{graphData.edges.length}</Text>
            </Card>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Node Types</Text>
              <Text style={styles.metricValue}>{typeCounts.length}</Text>
            </Card>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Density</Text>
              <Text style={styles.metricValue}>
                {graphData.nodes.length > 1
                  ? `${((graphData.edges.length * 2) / (graphData.nodes.length * (graphData.nodes.length - 1)) * 100).toFixed(1)}%`
                  : "0%"}
              </Text>
            </Card>
          </View>

          <Card variant="outlined" style={styles.graphSectionCard}>
            <Text style={styles.sectionCardTitle}>Node Type Distribution</Text>
            {typeCounts.slice(0, 8).map(([type, count]) => (
              <View key={type} style={styles.graphRow}>
                <Text style={styles.graphPrimaryText}>{type}</Text>
                <Text style={styles.graphSecondaryText}>{count}</Text>
              </View>
            ))}
          </Card>

          <Card variant="outlined" style={styles.graphSectionCard}>
            <Text style={styles.sectionCardTitle}>Top Nodes</Text>
            {graphData.nodes.slice(0, 10).map((node) => (
              <View key={node.id} style={styles.graphRow}>
                <Text style={styles.graphPrimaryText}>{node.label}</Text>
                <Text style={styles.graphSecondaryText}>{shortNumber(node.memory_count || node.mentions || 0)}</Text>
              </View>
            ))}
          </Card>

          <Card variant="outlined" style={styles.graphSectionCard}>
            <Text style={styles.sectionCardTitle}>Strongest Relationships</Text>
            {strongestEdges.map((edge, idx) => (
              <View key={`${edge.source}-${edge.target}-${idx}`} style={styles.graphRow}>
                <Text style={styles.graphPrimaryText} numberOfLines={1}>
                  {edge.source} → {edge.target}
                </Text>
                <Text style={styles.graphSecondaryText}>
                  {(edge.weight || 0).toFixed(2)} · {edge.relation}
                </Text>
              </View>
            ))}
          </Card>
        </>
          );
        })()
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>Graph not ready</Text>
          <Text style={styles.emptyBody}>Build connections through conversations.</Text>
        </View>
      )}
    </ScrollView>
  );

  const renderDashboardView = () => (
    <ScrollView style={styles.viewContainer} contentContainerStyle={[styles.viewPadding, styles.viewBottomSpace]}>
      <Text style={styles.viewTitle}>RAG Dashboard</Text>

      {loadingView ? (
        <ActivityIndicator color={COLORS.primary[500]} size="large" style={styles.loader} />
      ) : ragStats ? (
        <>
          <View style={styles.metricGrid}>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Memories</Text>
              <Text style={styles.metricValue}>{ragStats.memories.memories}</Text>
            </Card>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Entities</Text>
              <Text style={styles.metricValue}>{ragStats.memories.entities}</Text>
            </Card>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Vectors</Text>
              <Text style={styles.metricValue}>{ragStats.vectors.total_vectors}</Text>
            </Card>
            <Card variant="outlined" style={styles.metricCard}>
              <Text style={styles.metricLabel}>Graph Nodes</Text>
              <Text style={styles.metricValue}>{ragStats.graph.nodes}</Text>
            </Card>
          </View>

          <Card variant="outlined" style={styles.dashboardBlockCard}>
            <Text style={styles.sectionCardTitle}>Cache Performance</Text>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Hit Rate</Text>
              <Text style={styles.dashboardValue}>{(ragStats.cache.hit_rate * 100).toFixed(1)}%</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Total Queries</Text>
              <Text style={styles.dashboardValue}>{ragStats.cache.total_queries}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Total Hits</Text>
              <Text style={styles.dashboardValue}>{ragStats.cache.total_hits}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Cached Entries</Text>
              <Text style={styles.dashboardValue}>
                {ragStats.cache.exact_cache_size + ragStats.cache.semantic_cache_size + ragStats.cache.embedding_cache_size}
              </Text>
            </View>
          </Card>

          <Card variant="outlined" style={styles.dashboardBlockCard}>
            <Text style={styles.sectionCardTitle}>LLM Usage</Text>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Total Calls</Text>
              <Text style={styles.dashboardValue}>{ragStats.llm.call_count}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Total Tokens</Text>
              <Text style={styles.dashboardValue}>{ragStats.llm.total_tokens}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Model Loaded</Text>
              <Text style={styles.dashboardValue}>{ragStats.llm.model_loaded ? "Yes" : "No"}</Text>
            </View>
          </Card>
        </>
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>Dashboard loading…</Text>
        </View>
      )}
    </ScrollView>
  );

  const renderObservabilityView = () => (
    <View style={styles.viewContainer}>
      <View style={styles.viewPadding}>
        <Text style={styles.viewTitle}>Pipeline Observability</Text>
        <Text style={styles.viewSubtitle}>
          Live metrics and traces from the RAG pipeline · updated {formatRelativeTime(pipelineEvents[0]?.timestamp ?? null)}
        </Text>

        {loadingView ? (
          <ActivityIndicator color={COLORS.primary[500]} size="large" style={styles.loader} />
        ) : observabilityMetrics ? (
          <Card variant="outlined" style={styles.metricsCard}>
            <Text style={styles.sectionCardTitle}>Live Metrics</Text>
            {Object.entries(observabilityMetrics)
              .slice(0, 12)
              .map(([key, value]) => (
                <View key={key} style={styles.dashboardRow}>
                  <Text style={styles.dashboardLabel}>{key.replace(/_/g, " ")}</Text>
                  <Text style={styles.dashboardValue} numberOfLines={1}>
                    {typeof value === "number"
                      ? Number.isInteger(value)
                        ? value.toString()
                        : value.toFixed(3)
                      : typeof value === "string"
                        ? value
                        : JSON.stringify(value)}
                  </Text>
                </View>
              ))}
          </Card>
        ) : (
          <Card variant="outlined" style={styles.metricsCard}>
            <Text style={styles.emptyBody}>No observability metrics available yet.</Text>
          </Card>
        )}

        <Card variant="outlined" style={styles.metricsCard}>
          <View style={styles.liveEventHeaderRow}>
            <Text style={styles.sectionCardTitle}>Realtime Pipeline Events</Text>
            <Badge label={`${pipelineEvents.length} live`} variant="success" small />
          </View>

          {pipelineEvents.length > 0 ? (
            pipelineEvents.slice(0, 8).map((event, index) => (
              <View key={`${event.trace_id}-${event.timestamp}-${index}`} style={styles.liveEventRow}>
                <Text style={styles.liveEventPrimary} numberOfLines={1}>
                  {event.step_name.replace(/_/g, " ")}
                </Text>
                <Text style={styles.liveEventMeta} numberOfLines={1}>
                  {event.event_type} · {event.status} · {Math.round(event.duration_ms)}ms
                </Text>
              </View>
            ))
          ) : (
            <Text style={styles.emptyBody}>Waiting for live pipeline events…</Text>
          )}
        </Card>
      </View>

      <View style={styles.pipelineContainer}>
        <PipelineTracesList refreshInterval={4000} maxTraces={30} />
      </View>
    </View>
  );

  const renderAmbientView = () => (
    <ScrollView style={styles.viewContainer} contentContainerStyle={[styles.viewPadding, styles.viewBottomSpace]}>
      <Text style={styles.viewTitle}>Ambient Listening</Text>
      <Text style={styles.viewSubtitle}>
        STT/TTS control plane · updated {formatRelativeTime(ambientLastUpdatedAt)}
      </Text>

      {loadingView ? (
        <ActivityIndicator color={COLORS.primary[500]} size="large" style={styles.loader} />
      ) : ambientState ? (
        <>
          <Card variant="outlined" style={styles.ambientCard}>
            <View style={styles.ambientRow}>
              <Badge label={`Status: ${ambientState.status}`} variant="primary" />
              <Badge label={`STT: ${ambientState.stt_provider}`} variant="info" />
              <Badge label={`TTS: ${ambientState.tts_provider}`} variant="info" />
              <Badge
                label={ambientEnrollment?.enrolled ? "Voice ID: Enrolled" : "Voice ID: Missing"}
                variant={ambientEnrollment?.enrolled ? "success" : "warning"}
              />
            </View>

            <View style={styles.metricGrid}>
              <Card variant="default" padding="md" style={styles.metricCardCompact}>
                <Text style={styles.metricLabel}>Uptime</Text>
                <Text style={styles.metricValue}>{Math.round(ambientState.uptime_seconds)}s</Text>
              </Card>
              <Card variant="default" padding="md" style={styles.metricCardCompact}>
                <Text style={styles.metricLabel}>Segments</Text>
                <Text style={styles.metricValue}>{ambientState.speech_segments}</Text>
              </Card>
              <Card variant="default" padding="md" style={styles.metricCardCompact}>
                <Text style={styles.metricLabel}>Transcriptions</Text>
                <Text style={styles.metricValue}>{ambientState.transcriptions}</Text>
              </Card>
              <Card variant="default" padding="md" style={styles.metricCardCompact}>
                <Text style={styles.metricLabel}>Audio Level</Text>
                <Text style={styles.metricValue}>{Math.round(ambientState.audio_level || 0)} dB</Text>
              </Card>
            </View>

            <View style={styles.ambientActionRow}>
              <Button label="Start" size="sm" onPress={() => void runAmbientAction("start")} disabled={ambientBusy} />
              <Button label="Pause" size="sm" variant="secondary" onPress={() => void runAmbientAction("pause")} disabled={ambientBusy} />
              <Button label="Resume" size="sm" variant="outline" onPress={() => void runAmbientAction("resume")} disabled={ambientBusy} />
              <Button label="Stop" size="sm" variant="error" onPress={() => void runAmbientAction("stop")} disabled={ambientBusy} />
            </View>

            {ambientState.error ? <Text style={styles.errorText}>{ambientState.error}</Text> : null}
          </Card>

          <Card variant="outlined" style={styles.ambientCard}>
            <Text style={styles.sectionCardTitle}>Voice Provider Routing</Text>
            <Text style={styles.ambientHintText}>
              Traditional = Whisper/Piper. Gemini = API speech stack.
            </Text>

            <View style={styles.ambientActionRow}>
              <Button
                label="STT Traditional"
                size="sm"
                variant={ambientState.stt_provider === "traditional" ? "primary" : "outline"}
                onPress={() => void setAmbientProvider("stt", "traditional")}
                disabled={ambientBusy || ambientProviders?.traditional_stt_available === false}
              />
              <Button
                label="STT Gemini"
                size="sm"
                variant={ambientState.stt_provider === "gemini" ? "primary" : "outline"}
                onPress={() => void setAmbientProvider("stt", "gemini")}
                disabled={ambientBusy || ambientProviders?.gemini_stt_available === false}
              />
            </View>

            <View style={styles.ambientActionRow}>
              <Button
                label="TTS Traditional"
                size="sm"
                variant={ambientState.tts_provider === "traditional" ? "primary" : "outline"}
                onPress={() => void setAmbientProvider("tts", "traditional")}
                disabled={ambientBusy || ambientProviders?.traditional_tts_available === false}
              />
              <Button
                label="TTS Gemini"
                size="sm"
                variant={ambientState.tts_provider === "gemini" ? "primary" : "outline"}
                onPress={() => void setAmbientProvider("tts", "gemini")}
                disabled={ambientBusy || ambientProviders?.gemini_tts_available === false}
              />
            </View>

            {ambientProviders ? (
              <Text style={styles.ambientSmallText}>
                Gemini voices: {(ambientProviders.gemini_tts_voices || []).join(", ") || "none"}
              </Text>
            ) : null}
          </Card>

          <Card variant="outlined" style={styles.ambientCard}>
            <Text style={styles.sectionCardTitle}>Enrollment & Runtime Config</Text>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Voice enrolled</Text>
              <Text style={styles.dashboardValue}>{ambientEnrollment?.enrolled ? "Yes" : "No"}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Auto ingest</Text>
              <Text style={styles.dashboardValue}>{ambientConfig?.auto_ingest ? "Enabled" : "Disabled"}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>VAD threshold</Text>
              <Text style={styles.dashboardValue}>{ambientConfig?.vad_threshold ?? "-"}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Silence timeout</Text>
              <Text style={styles.dashboardValue}>{ambientConfig?.silence_timeout_s ?? "-"}s</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Whisper model</Text>
              <Text style={styles.dashboardValue}>{ambientConfig?.whisper_model_size || "-"}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>TTS voice</Text>
              <Text style={styles.dashboardValue}>{ambientConfig?.tts_voice || ambientConfig?.gemini_tts_voice || "-"}</Text>
            </View>

            <View style={styles.ambientActionRow}>
              <Button
                label={ambientBusy ? "Working…" : "Start Enrollment"}
                size="sm"
                variant="outline"
                onPress={() => void startAmbientEnrollment()}
                disabled={ambientBusy || ambientEnrollment?.speaker_id_available === false}
              />
              <Button
                label={ambientConfig?.auto_ingest ? "Disable Auto-Ingest" : "Enable Auto-Ingest"}
                size="sm"
                variant="secondary"
                onPress={() => void toggleAmbientAutoIngest()}
                disabled={ambientBusy || !ambientConfig}
              />
            </View>
          </Card>

          <Card variant="outlined" style={styles.ambientCard}>
            <Text style={styles.sectionCardTitle}>Live Transcript</Text>
            {ambientTurns.length > 0 ? (
              ambientTurns
                .slice(-8)
                .reverse()
                .map((turn, idx) => (
                  <View key={`${turn.timestamp}-${idx}`} style={styles.ambientTranscriptRow}>
                    <View style={styles.ambientTranscriptMeta}>
                      <Text style={styles.ambientTranscriptSpeaker}>{turn.speaker_name || turn.speaker_label}</Text>
                      <Text style={styles.ambientTranscriptScore}>{toPercent(turn.confidence)}</Text>
                    </View>
                    <Text style={styles.ambientTranscriptText}>{turn.text}</Text>
                  </View>
                ))
            ) : (
              <Text style={styles.emptyBody}>No live transcript turns yet.</Text>
            )}
          </Card>

          <Card variant="outlined" style={styles.ambientCard}>
            <Text style={styles.sectionCardTitle}>Recent Ambient Conversations</Text>
            {ambientConversations.length > 0 ? (
              ambientConversations.slice(0, 6).map((conversation) => (
                <View key={conversation.id} style={styles.ambientConversationRow}>
                  <View style={styles.ambientConversationMain}>
                    <Text style={styles.ambientConversationTitle} numberOfLines={1}>
                      {conversation.participants.join(", ") || "Unknown participants"}
                    </Text>
                    <Text style={styles.ambientConversationMeta}>
                      {conversation.turns.length} turns · {Math.round(conversation.duration_seconds)}s
                    </Text>
                  </View>
                  <Badge
                    label={conversation.auto_ingested ? "Ingested" : "Pending"}
                    variant={conversation.auto_ingested ? "success" : "warning"}
                    small
                  />
                </View>
              ))
            ) : (
              <Text style={styles.emptyBody}>No completed ambient conversations yet.</Text>
            )}
          </Card>

          <Card variant="outlined" style={styles.ambientCard}>
            <Text style={styles.sectionCardTitle}>TTS Health Check</Text>
            <TextInput
              placeholder="Type sample speech text for synthesis validation…"
              value={ttsDraft}
              onChangeText={setTtsDraft}
              multiline
              style={styles.memoryComposerInput}
            />
            <View style={styles.ambientActionRow}>
              <Button
                label={ttsBusy ? "Synthesizing…" : "Synthesize WAV"}
                size="sm"
                onPress={() => void runTTSHealthCheck()}
                disabled={ttsBusy || !ttsDraft.trim()}
              />
              <Button
                label="Refresh Status"
                size="sm"
                variant="outline"
                onPress={() => void loadAmbient(true)}
                disabled={ttsBusy}
              />
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>TTS available</Text>
              <Text style={styles.dashboardValue}>{ambientTTSStatus?.available ? "Yes" : "No"}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Voice</Text>
              <Text style={styles.dashboardValue}>{ambientTTSStatus?.voice || "-"}</Text>
            </View>
            <View style={styles.dashboardRow}>
              <Text style={styles.dashboardLabel}>Last synthesized</Text>
              <Text style={styles.dashboardValue}>{ttsLastBytes ? `${Math.round(ttsLastBytes / 1024)} KB` : "Not run"}</Text>
            </View>
          </Card>
        </>
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>Ambient status unavailable</Text>
          <Text style={styles.emptyBody}>Start backend ambient components and refresh this tab.</Text>
        </View>
      )}
    </ScrollView>
  );

  const renderDocumentsView = () => (
    <ScrollView style={styles.viewContainer} contentContainerStyle={[styles.viewPadding, styles.viewBottomSpace]}>
      <Text style={styles.viewTitle}>PageIndex Documents</Text>
      <Text style={styles.viewSubtitle}>Upload, inspect, and manage indexed PDF documents</Text>

      <View style={styles.documentsActionsRow}>
        <Button
          label={documentsBusy ? "Uploading…" : "Upload PDF"}
          size="sm"
          onPress={() => void uploadDocument()}
          disabled={documentsBusy}
        />
        <Button
          label="Refresh"
          size="sm"
          variant="outline"
          onPress={() => void loadDocuments()}
          disabled={documentsBusy}
        />
      </View>

      <Card variant="outlined" style={styles.documentQueryCard}>
        <Text style={styles.memoryComposerTitle}>Ask Your Documents</Text>
        <TextInput
          placeholder="Ask a question across indexed documents…"
          value={documentQuery}
          onChangeText={setDocumentQuery}
          multiline
          style={styles.memoryComposerInput}
        />
        <View style={styles.memoryComposerActions}>
          <Button
            label={documentQueryBusy ? "Querying…" : "Run Query"}
            size="sm"
            onPress={() => void runDocumentQuery()}
            disabled={documentQueryBusy || !documentQuery.trim()}
          />
          <Button
            label="Clear"
            size="sm"
            variant="secondary"
            onPress={() => {
              setDocumentAnswer("");
              setDocumentSections([]);
              setDocumentQuery("");
            }}
            disabled={documentQueryBusy}
          />
        </View>
      </Card>

      {pageIndexUsage ? (
        <View style={styles.metricGrid}>
          <Card variant="outlined" style={styles.metricCard}>
            <Text style={styles.metricLabel}>Month</Text>
            <Text style={styles.metricValue}>{pageIndexUsage.month}</Text>
          </Card>
          <Card variant="outlined" style={styles.metricCard}>
            <Text style={styles.metricLabel}>Queries</Text>
            <Text style={styles.metricValue}>
              {pageIndexUsage.queries_used}/{pageIndexUsage.queries_limit}
            </Text>
          </Card>
          <Card variant="outlined" style={styles.metricCard}>
            <Text style={styles.metricLabel}>Pages</Text>
            <Text style={styles.metricValue}>
              {pageIndexUsage.pages_used}/{pageIndexUsage.pages_limit}
            </Text>
          </Card>
          <Card variant="outlined" style={styles.metricCard}>
            <Text style={styles.metricLabel}>Enabled</Text>
            <Text style={styles.metricValue}>{pageIndexEnabled ? "Yes" : "No"}</Text>
          </Card>
        </View>
      ) : null}

      {documentAnswer ? (
        <Card variant="outlined" style={styles.documentAnswerCard}>
          <Text style={styles.sectionCardTitle}>Answer</Text>
          <Text style={styles.memoryText}>{documentAnswer}</Text>
          {documentSections.length > 0 ? (
            <View style={styles.documentSectionsList}>
              {documentSections.slice(0, 5).map((section, idx) => (
                <View key={`${section.doc_id}-${section.page}-${idx}`} style={styles.documentSectionRow}>
                  <Text style={styles.documentSectionMeta}>p.{section.page} · {(section.score * 100).toFixed(0)}%</Text>
                  <Text style={styles.documentSectionText} numberOfLines={3}>{section.content}</Text>
                </View>
              ))}
            </View>
          ) : null}
        </Card>
      ) : null}

      {loadingView ? (
        <ActivityIndicator color={COLORS.primary[500]} size="large" style={styles.loader} />
      ) : documents.length > 0 ? (
        <View style={styles.documentsList}>
          {documents.map((doc) => (
            <Card key={doc.doc_id} variant="outlined" style={styles.documentCard}>
              <Text style={styles.documentTitle}>{doc.filename}</Text>
              <View style={styles.documentMetaRow}>
                <Badge label={doc.status} variant={doc.status === "ready" ? "success" : "warning"} small />
                <Badge label={`${doc.estimated_pages} pages`} variant="info" small />
              </View>
              <View style={styles.documentActionsRow}>
                <Button
                  label={documentTreeDocId === doc.doc_id ? "Hide Tree" : "View Tree"}
                  size="sm"
                  variant="outline"
                  onPress={() => void toggleDocumentTree(doc.doc_id)}
                  disabled={documentsBusy}
                />
                <Button
                  label="Delete"
                  size="sm"
                  variant="error"
                  onPress={() => void deleteDocument(doc.doc_id)}
                  disabled={documentsBusy}
                />
              </View>
            </Card>
          ))}

          {documentTreeDocId && documentTreePreview.length > 0 ? (
            <Card variant="outlined" style={styles.documentAnswerCard}>
              <Text style={styles.sectionCardTitle}>Document Tree Preview</Text>
              <Text style={styles.documentSectionMeta}>{documentTreeDocId}</Text>
              <View style={styles.documentSectionsList}>
                {documentTreePreview.map((line, idx) => (
                  <View key={`${documentTreeDocId}-${idx}`} style={styles.documentSectionRow}>
                    <Text style={styles.documentSectionText} numberOfLines={3}>{line}</Text>
                  </View>
                ))}
              </View>
            </Card>
          ) : null}
        </View>
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No documents uploaded</Text>
          <Text style={styles.emptyBody}>Upload a PDF to enable PageIndex-backed document retrieval.</Text>
        </View>
      )}
    </ScrollView>
  );

  // ─── Main Render ────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
      <StatusBar style="dark" />

      <View pointerEvents="none" style={styles.backgroundLayer}>
        <View style={[styles.backgroundOrb, styles.backgroundOrbPrimary]} />
        <View style={[styles.backgroundOrb, styles.backgroundOrbSecondary]} />
      </View>

      {/* Header */}
      <Header
        modelStatus={modelStatus}
        title="Cortex Lab"
        subtitle={activeView !== "chat" ? `${NAV_ITEMS.find((n) => n.key === activeView)?.label || ""} • API ${apiHost}` : `API ${apiHost}`}
      />

      {/* Content */}
      {activeView === "chat" ? renderChatView() : null}
      {activeView === "memories" ? renderMemoriesView() : null}
      {activeView === "graph" ? renderGraphView() : null}
      {activeView === "dashboard" ? renderDashboardView() : null}
      {activeView === "observability" ? renderObservabilityView() : null}
      {activeView === "ambient" ? renderAmbientView() : null}
      {activeView === "documents" ? renderDocumentsView() : null}

      {/* Bottom Navigation */}
      <BottomNav
        items={NAV_ITEMS}
        activeKey={activeView}
        onSelect={setActiveView}
      />
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

// ════════════════════════════════════════════════════════════════════════
// STYLES
// ════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: SEMANTIC_COLORS.bgCanvas,
  },
  backgroundLayer: {
    ...StyleSheet.absoluteFillObject,
  },
  backgroundOrb: {
    position: "absolute",
    borderRadius: BORDER_RADIUS.full,
    opacity: 0.15,
  },
  backgroundOrbPrimary: {
    width: 300,
    height: 300,
    top: -120,
    left: -90,
    backgroundColor: COLORS.primary[300],
  },
  backgroundOrbSecondary: {
    width: 320,
    height: 320,
    bottom: -150,
    right: -80,
    backgroundColor: COLORS.cyan[500],
  },

  // ─── Views ──────────────────────────────────────────────────────────
  viewContainer: {
    flex: 1,
    backgroundColor: SEMANTIC_COLORS.bgCanvas,
  },
  viewPadding: {
    paddingHorizontal: SPACING.xl,
    paddingTop: SPACING.lg,
    gap: SPACING.lg,
  },
  viewBottomSpace: {
    paddingBottom: SPACING["4xl"],
  },
  viewTitle: {
    fontSize: TYPOGRAPHY.fontSize["2xl"],
    fontWeight: TYPOGRAPHY.fontWeight.bold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.xs,
  },
  viewSubtitle: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.md,
  },

  // ─── Chat View ──────────────────────────────────────────────────────
  chatTopShell: {
    backgroundColor: SEMANTIC_COLORS.glassOverlay,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
    paddingTop: SPACING.sm,
  },
  conversationBar: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.sm,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
  },
  conversationList: {
    gap: SPACING.sm,
    paddingHorizontal: SPACING.sm,
  },
  newChatButton: {
    minWidth: 60,
  },
  chatControlRow: {
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: SPACING.sm,
    paddingHorizontal: SPACING.xl,
    paddingBottom: SPACING.md,
  },
  quickPromptList: {
    gap: SPACING.sm,
    paddingHorizontal: SPACING.xl,
    paddingRight: SPACING["3xl"],
    paddingBottom: SPACING.md,
  },
  quickPromptChip: {
    maxWidth: 280,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.full,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  quickPromptText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  controlPill: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
  },
  controlPillBlue: {
    backgroundColor: COLORS.info[100],
    borderColor: COLORS.info[300],
  },
  controlPillViolet: {
    backgroundColor: COLORS.primary[100],
    borderColor: COLORS.primary[300],
  },
  controlPillGreen: {
    backgroundColor: COLORS.success[100],
    borderColor: COLORS.success[300],
  },
  controlPillNeutral: {
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  controlPillAmber: {
    backgroundColor: COLORS.warning[100],
    borderColor: COLORS.warning[300],
  },
  controlPillDisabled: {
    opacity: 0.6,
  },
  controlPillText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
  },
  convChip: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  convChipActive: {
    backgroundColor: COLORS.primary[600],
    borderColor: COLORS.primary[600],
  },
  convChipText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
  },
  convChipTextActive: {
    color: COLORS.white,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },

  messagesList: {
    paddingVertical: SPACING.lg,
    paddingBottom: SPACING["5xl"],
  },
  emptyState: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING["5xl"],
  },
  emptyTitle: {
    fontSize: TYPOGRAPHY.fontSize.lg,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.sm,
  },
  emptyBody: {
    fontSize: TYPOGRAPHY.fontSize.md,
    color: SEMANTIC_COLORS.textSecondary,
    textAlign: "center",
  },

  errorBanner: {
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.lg,
    borderColor: COLORS.error[300],
  },
  errorText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: COLORS.error[600],
  },

  inputArea: {
    flexDirection: "row",
    gap: SPACING.md,
    alignItems: "flex-end",
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.glassOverlay,
    borderTopWidth: 1,
    borderTopColor: SEMANTIC_COLORS.borderPrimary,
  },
  sendButton: {
    minWidth: 82,
  },
  inputMetaRow: {
    backgroundColor: SEMANTIC_COLORS.glassOverlay,
    paddingHorizontal: SPACING.xl,
    paddingBottom: SPACING.md,
    borderTopWidth: 1,
    borderTopColor: SEMANTIC_COLORS.borderPrimary,
  },
  inputMetaText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    textAlign: "right",
  },

  // ─── Memories View ──────────────────────────────────────────────────
  memoryToolbar: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.md,
    marginBottom: SPACING.md,
  },
  memorySearchInput: {
    flex: 1,
  },
  memoryComposerCard: {
    marginBottom: SPACING.md,
  },
  memoryComposerTitle: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.sm,
  },
  memoryComposerInput: {
    marginBottom: SPACING.md,
  },
  memoryComposerActions: {
    flexDirection: "row",
    gap: SPACING.sm,
  },
  memoriesList: {
    gap: SPACING.lg,
  },
  memoryCard: {
    marginBottom: SPACING.md,
  },
  memoryMetaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  memoryScoreText: {
    marginLeft: "auto",
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
  },
  memoryText: {
    fontSize: TYPOGRAPHY.fontSize.md,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.md,
    lineHeight: TYPOGRAPHY.lineHeight.relaxed * TYPOGRAPHY.fontSize.md,
  },
  memoryFooterRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  memoryDate: {
    alignSelf: "flex-start",
  },
  memoryDeleteButton: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.error[50],
    borderWidth: 1,
    borderColor: COLORS.error[200],
  },
  memoryDeleteText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.error[700],
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },

  // ─── Other Views ────────────────────────────────────────────────────
  loader: {
    marginVertical: SPACING["4xl"],
  },
  metricGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.md,
    marginBottom: SPACING.md,
  },
  metricCard: {
    flexBasis: "47%",
    flexGrow: 1,
  },
  metricCardCompact: {
    flexBasis: "47%",
    flexGrow: 1,
  },
  metricLabel: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  metricValue: {
    fontSize: TYPOGRAPHY.fontSize.lg,
    color: SEMANTIC_COLORS.textPrimary,
    fontWeight: TYPOGRAPHY.fontWeight.bold,
  },
  sectionCardTitle: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.sm,
  },
  graphSectionCard: {
    marginBottom: SPACING.md,
  },
  graphRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
  },
  graphPrimaryText: {
    flex: 1,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
  },
  graphSecondaryText: {
    marginLeft: SPACING.md,
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    textTransform: "capitalize",
  },
  dashboardBlockCard: {
    marginBottom: SPACING.md,
  },
  dashboardRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
  },
  dashboardLabel: {
    flex: 1,
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
  },
  dashboardValue: {
    maxWidth: "55%",
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    textAlign: "right",
  },
  metricsCard: {
    marginBottom: SPACING.lg,
  },
  liveEventHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: SPACING.sm,
  },
  liveEventRow: {
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  liveEventPrimary: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    marginBottom: SPACING.xs,
    textTransform: "capitalize",
  },
  liveEventMeta: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
  },
  pipelineContainer: {
    flex: 1,
    borderTopWidth: 1,
    borderTopColor: SEMANTIC_COLORS.borderPrimary,
  },
  ambientCard: {
    gap: SPACING.md,
  },
  ambientRow: {
    flexDirection: "row",
    gap: SPACING.sm,
    flexWrap: "wrap",
  },
  ambientText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
  },
  ambientActionRow: {
    flexDirection: "row",
    gap: SPACING.sm,
    flexWrap: "wrap",
    marginTop: SPACING.sm,
  },
  ambientHintText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.sm,
  },
  ambientSmallText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    marginTop: SPACING.sm,
  },
  ambientTranscriptRow: {
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  ambientTranscriptMeta: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: SPACING.xs,
  },
  ambientTranscriptSpeaker: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
  ambientTranscriptScore: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.primary[700],
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
  ambientTranscriptText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    lineHeight: TYPOGRAPHY.fontSize.sm * TYPOGRAPHY.lineHeight.normal,
  },
  ambientConversationRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.md,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    marginBottom: SPACING.sm,
    gap: SPACING.sm,
  },
  ambientConversationMain: {
    flex: 1,
  },
  ambientConversationTitle: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    marginBottom: SPACING.xs,
  },
  ambientConversationMeta: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
  },
  documentsActionsRow: {
    flexDirection: "row",
    gap: SPACING.md,
    marginBottom: SPACING.lg,
  },
  usageCard: {
    marginBottom: SPACING.lg,
  },
  documentQueryCard: {
    marginBottom: SPACING.md,
  },
  documentAnswerCard: {
    marginBottom: SPACING.md,
  },
  documentSectionsList: {
    marginTop: SPACING.md,
    gap: SPACING.sm,
  },
  documentSectionRow: {
    padding: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  documentSectionMeta: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    marginBottom: SPACING.xs,
  },
  documentSectionText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    lineHeight: TYPOGRAPHY.fontSize.sm * TYPOGRAPHY.lineHeight.normal,
  },
  documentsList: {
    gap: SPACING.md,
  },
  documentCard: {
    marginBottom: SPACING.md,
  },
  documentTitle: {
    fontSize: TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.xs,
  },
  documentMeta: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  documentMetaRow: {
    flexDirection: "row",
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  documentActionsRow: {
    marginTop: SPACING.sm,
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: SPACING.sm,
  },
});
