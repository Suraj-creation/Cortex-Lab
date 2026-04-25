/**
 * Cortex Lab Mobile — App.tsx Orchestrator
 * Cortex Aurora Light design system — premium mobile experience
 * All backend logic preserved from original implementation
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  StyleSheet,
  Platform,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import * as DocumentPicker from 'expo-document-picker';
import {
  createDownloadResumable,
  documentDirectory,
  makeDirectoryAsync,
} from 'expo-file-system/legacy';

import {
  type ApiClient,
  createApiClient,
  getDefaultApiBaseUrl,
  resolveHealthyApiBaseUrl,
  type PageIndexDocument,
  type PageIndexUsage,
  type RAGStreamMeta,
  type TTSStatus,
} from './shared/core/api';
import type {
  AmbientConfig,
  AmbientClientSessionInfo,
  AmbientLiveStatus,
  AmbientRetentionTrace,
  ChatMessage,
  ChatSettings,
  ConversationRecord,
  ConversationTurn,
  ModelStatus,
  ModelpackEntry,
  ModelpackInstallState,
  ModelpackManifest,
  MemoryObject,
  GraphData,
  RAGStats,
  AmbientState,
  LivePipelineEvent,
  LLMProviderType,
  VoiceProviders,
} from './shared/core/types';
import { DEFAULT_SETTINGS } from './shared/core/types';
import {
  getAllConversations,
  getConversation,
  getCurrentConversationId,
  getLastApiUrl,
  loadModelpackInstalls,
  getOnboardingCompleted,
  initializeStorage,
  loadChatSettings,
  saveModelpackInstalls,
  saveChatSettings,
  saveConversation,
  saveCurrentConversationId,
  saveLastApiUrl,
  saveOnboardingCompleted,
} from './shared/core/storage';

// ── UI Components ─────────────────────────────────────────────────────────────
import { Header } from './src/components/ui/Header';
import { BottomNav, type NavKey } from './src/components/ui/BottomNav';

// ── Screen Components ─────────────────────────────────────────────────────────
import { ChatScreen } from './src/screens/ChatScreen';
import { MemoryScreen } from './src/screens/MemoryScreen';
import { DashboardScreen } from './src/screens/DashboardScreen';
import { GraphScreen } from './src/screens/GraphScreen';
import { ObservabilityScreen } from './src/screens/ObservabilityScreen';
import { AgentScreen } from './src/screens/AgentScreen';
import { WikiScreen } from './src/screens/WikiScreen';
import { SessionForgeScreen } from './src/screens/SessionForgeScreen';
import { ChronicleScreen } from './src/screens/ChronicleScreen';
import { AmbientVoiceScreen } from './src/screens/AmbientVoiceScreen';
import { DocumentsScreen } from './src/screens/DocumentsScreen';
import { OnboardingScreen } from './src/screens/OnboardingScreen';
import { ErrorBoundary } from './src/components/ErrorBoundary';
import { NetworkStatusBanner } from './src/components/NetworkStatusBanner';
import { NetworkProvider } from './src/providers/NetworkProvider';

// ── Modals ────────────────────────────────────────────────────────────────────
import { ConversationDrawer } from './src/modals/ConversationDrawer';

// ── Theme ─────────────────────────────────────────────────────────────────────
import { NEURAL } from './src/theme/colors';

// ─── Nav items with icons ─────────────────────────────────────────────────────
const NAV_ITEMS = [
  { key: 'chat'          as NavKey, label: 'Chat',      iconName: 'chat-processing-outline' as const },
  { key: 'agent'         as NavKey, label: 'Agent',     iconName: 'robot-outline' as const },
  { key: 'memories'      as NavKey, label: 'Memory',    iconName: 'brain' as const },
  { key: 'observability' as NavKey, label: 'Observe',   iconName: 'chart-timeline-variant' as const },
  { key: 'wiki'          as NavKey, label: 'Wiki',      iconName: 'book-open-page-variant-outline' as const },
  { key: 'session-forge' as NavKey, label: 'Forge',     iconName: 'atom-variant' as const },
  { key: 'chronicle'     as NavKey, label: 'Chronicle', iconName: 'timeline-text-outline' as const },
  { key: 'graph'         as NavKey, label: 'Graph',     iconName: 'graph-outline' as const },
  { key: 'dashboard'     as NavKey, label: 'Hub',       iconName: 'view-dashboard-outline' as const },
  { key: 'ambient'       as NavKey, label: 'Voice',     iconName: 'microphone-outline' as const },
  { key: 'documents'     as NavKey, label: 'Docs',      iconName: 'file-document-outline' as const },
];

const STARTER_TEXT = 'How can I help you today?';

function inferTitle(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === 'user');
  if (!first) return 'New Chat';
  const raw = first.content.trim();
  return raw.length > 40 ? `${raw.slice(0, 40)}…` : raw;
}

interface ConvSummary { id: string; title: string; timestamp: number; }

interface SettingsPageProps {
  settings: ChatSettings;
  onUpdateSettings: (s: Partial<ChatSettings>) => void;
  onSelectLLMProvider: (provider: LLMProviderType) => void;
  onBack: () => void;
  onReconnect: () => void;
  reconnecting: boolean;
  connectionStatus: string;
  backendUrlLabel: string;
  localModelAvailable: boolean;
  modelpackManifest: ModelpackManifest | null;
  modelpackInstalls: Record<string, ModelpackInstallState>;
  modelpackCapabilityMessage: string;
  modelpackError: string;
  onRefreshModelpacks: () => void;
  onInstallModelpack: (pack: ModelpackEntry) => void;
}

const SettingsPage = require('./src/screens/SettingsScreen').SettingsScreen as React.ComponentType<SettingsPageProps>;

function formatApiEndpointLabel(rawUrl: string): string {
  try {
    return new URL(rawUrl).host;
  } catch {
    return rawUrl;
  }
}

function describeApiSource(source: string): string {
  switch (source) {
    case 'env':
      return 'build configuration';
    case 'canonical':
      return 'production fallback';
    case 'persisted':
      return 'saved remote endpoint';
    case 'same-origin':
      return 'same-origin proxy';
    case 'local-dev':
      return 'local development host';
    default:
      return 'automatic discovery';
  }
}

function inferModelpackFiles(pack: ModelpackEntry) {
  if (Array.isArray(pack.files) && pack.files.length > 0) {
    return pack.files.filter((file) => file.path.trim().length > 0);
  }

  switch (pack.id) {
    case 'gemma-4-e4b-it-litert-lm':
      return [
        { path: 'gemma-4-E4B-it.litertlm', size_bytes: 0, sha256: '' },
        { path: 'gemma-4-E4B-it-web.task', size_bytes: 0, sha256: '' },
      ];
    case 'gemma-4-e2b-it-litert-lm':
      return [
        { path: 'gemma-4-E2B-it.litertlm', size_bytes: 0, sha256: '' },
        { path: 'gemma-4-E2B-it-web.task', size_bytes: 0, sha256: '' },
      ];
    default:
      return [];
  }
}

function buildModelpackArtifactUrl(pack: ModelpackEntry, relativePath: string): string {
  const base = (pack.download_url || '').replace(/\/+$/, '');
  const encodedPath = relativePath
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
  return `${base}/resolve/main/${encodedPath}?download=1`;
}

// ─── Main App Content ─────────────────────────────────────────────────────────
function AppContent() {
  const [apiBase, setApiBase] = useState<string>(getDefaultApiBaseUrl());
  const api = useMemo(() => createApiClient({ baseUrl: apiBase }), [apiBase]);
  const apiRef = useRef<ApiClient>(api);

  // ── Navigation ──────────────────────────────────────────────────────────────
  const [activeView, setActiveView] = useState<NavKey>('chat');
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Auto-connecting to Cortex backend…');
  const [backendUrlLabel, setBackendUrlLabel] = useState(formatApiEndpointLabel(apiBase));
  const [modelpackManifest, setModelpackManifest] = useState<ModelpackManifest | null>(null);
  const [modelpackError, setModelpackError] = useState('');
  const [modelpackInstalls, setModelpackInstalls] = useState<Record<string, ModelpackInstallState>>({});

  // ── Global state ────────────────────────────────────────────────────────────
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ status: 'loading', model_loaded: false, model_info: {} });
  const [globalError, setGlobalError] = useState('');
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [loadingView, setLoadingView] = useState(false);

  // ── Chat ─────────────────────────────────────────────────────────────────────
  const [conversations, setConversations] = useState<ConvSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [localModelAvailable, setLocalModelAvailable] = useState(true);
  const [providerBusy, setProviderBusy] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const messagesRef = useRef<ChatMessage[]>(messages);

  // ── Memory ───────────────────────────────────────────────────────────────────
  const [memories, setMemories] = useState<MemoryObject[]>([]);
  const [memorySearch, setMemorySearch] = useState('');
  const [memoryDraft, setMemoryDraft] = useState('');
  const [memoryBusy, setMemoryBusy] = useState(false);
  const [lastMemoryRetention, setLastMemoryRetention] = useState<AmbientRetentionTrace | null>(null);
  const [lastMemorySession, setLastMemorySession] = useState<AmbientClientSessionInfo | null>(null);
  const [lastMemoryDocument, setLastMemoryDocument] = useState('');

  // ── Graph / Dashboard ─────────────────────────────────────────────────────────
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [ragStats, setRagStats] = useState<RAGStats | null>(null);

  // ── Observability ─────────────────────────────────────────────────────────────
  const [observabilityMetrics, setObservabilityMetrics] = useState<Record<string, unknown> | null>(null);
  const [pipelineEvents, setPipelineEvents] = useState<LivePipelineEvent[]>([]);

  // ── Ambient ───────────────────────────────────────────────────────────────────
  const [ambientState, setAmbientState] = useState<AmbientState | null>(null);
  const [ambientLiveStatus, setAmbientLiveStatus] = useState<AmbientLiveStatus | null>(null);
  const [ambientConfig, setAmbientConfig] = useState<AmbientConfig | null>(null);
  const [ambientEnrollment, setAmbientEnrollment] = useState<{ enrolled: boolean; speaker_id_available?: boolean } | null>(null);
  const [ambientProviders, setAmbientProviders] = useState<VoiceProviders | null>(null);
  const [ambientTurns, setAmbientTurns] = useState<ConversationTurn[]>([]);
  const [ambientConversations, setAmbientConversations] = useState<ConversationRecord[]>([]);
  const [ambientTTSStatus, setAmbientTTSStatus] = useState<TTSStatus | null>(null);
  const [ttsDraft, setTtsDraft] = useState('Hey, this is Cortex speaking. Voice synthesis is online.');
  const [ttsBusy, setTtsBusy] = useState(false);
  const [ttsLastBytes, setTtsLastBytes] = useState<number | null>(null);
  const [ambientBusy, setAmbientBusy] = useState(false);

  // ── Documents ─────────────────────────────────────────────────────────────────
  const [documents, setDocuments] = useState<PageIndexDocument[]>([]);
  const [pageIndexUsage, setPageIndexUsage] = useState<PageIndexUsage | null>(null);
  const [pageIndexEnabled, setPageIndexEnabled] = useState<boolean | null>(null);
  const [documentQuery, setDocumentQuery] = useState('');
  const [documentQueryBusy, setDocumentQueryBusy] = useState(false);
  const [documentAnswer, setDocumentAnswer] = useState('');
  const [documentSections, setDocumentSections] = useState<{ page: number; content: string; doc_id: string; score: number }[]>([]);
  const [documentTreeDocId, setDocumentTreeDocId] = useState<string | null>(null);
  const [documentTreePreview, setDocumentTreePreview] = useState<string[]>([]);
  const [documentsBusy, setDocumentsBusy] = useState(false);

  // ─── Backend helpers ────────────────────────────────────────────────────────
  useEffect(() => {
    apiRef.current = api;
  }, [api]);

  useEffect(() => { messagesRef.current = messages; }, [messages]);

  const refreshModelStatus = useCallback(async (client: ApiClient = apiRef.current) => {
    try {
      const status = await client.getModelStatus();
      setModelStatus(status);
    } catch {
      setModelStatus({ status: 'offline', model_loaded: false, model_info: {} });
    }
  }, []);

  const syncLLMProvider = useCallback(async (client: ApiClient = apiRef.current) => {
    try {
      const provider = await client.getLLMProvider();
      setLocalModelAvailable(provider.local_model_loaded);
      setSettings((prev) => {
        if (provider.provider === 'local' && !provider.local_model_loaded && provider.gemini_configured) {
          return { ...prev, llmProvider: 'gemini' };
        }
        return { ...prev, llmProvider: provider.provider };
      });
    } catch {}
  }, []);

  const loadModelpackManifest = useCallback(async (client: ApiClient = apiRef.current) => {
    try {
      const manifest = await client.getModelpackManifest();
      setModelpackManifest(manifest);
      setModelpackError('');
    } catch (e) {
      setModelpackError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const connectBackend = useCallback(async (persistedUrl?: string | null) => {
    setTestingConnection(true);
    setConnectionStatus('Auto-connecting to Cortex backend…');

    try {
      const resolution = await resolveHealthyApiBaseUrl(persistedUrl);
      const nextBase = resolution.baseUrl;
      const nextClient = createApiClient({ baseUrl: nextBase });

      setApiBase(nextBase);
      setBackendUrlLabel(formatApiEndpointLabel(nextBase));

      if (resolution.reachable) {
        setConnectionStatus(`Connected automatically via ${describeApiSource(resolution.source)}.`);
        await saveLastApiUrl(nextBase);
      } else {
        setConnectionStatus('Preferred backend selected, but the live service is still warming up.');
      }

      await Promise.allSettled([
        refreshModelStatus(nextClient),
        syncLLMProvider(nextClient),
        loadModelpackManifest(nextClient),
      ]);

      return nextClient;
    } finally {
      setTestingConnection(false);
    }
  }, [loadModelpackManifest, refreshModelStatus, syncLLMProvider]);

  const loadConversations = useCallback(async () => {
    try {
      const all = await getAllConversations();
      const sorted = [...all].sort((a, b) => b.timestamp - a.timestamp);
      const summary = sorted.map((c) => ({ id: c.id, title: c.title || inferTitle(c.messages), timestamp: c.timestamp }));
      setConversations(summary);
      return summary;
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : String(e));
      return [];
    }
  }, []);

  const ensureActiveConversation = useCallback(async () => {
    const summary = await loadConversations();
    const saved = await getCurrentConversationId();
    if (saved && summary.some((c) => c.id === saved)) { setActiveConversationId(saved); return; }
    if (summary.length > 0) { setActiveConversationId(summary[0].id); await saveCurrentConversationId(summary[0].id); return; }
    const id = `conv_${Date.now()}`;
    const starter: ChatMessage[] = [{ id: 'starter', role: 'assistant', content: STARTER_TEXT, timestamp: Date.now() }];
    await saveConversation(starter, id, 'New Chat');
    await saveCurrentConversationId(id);
    setConversations([{ id, title: 'New Chat', timestamp: Date.now() }]);
    setActiveConversationId(id);
  }, [loadConversations]);

  // Initial setup
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await initializeStorage();
        const ls = await loadChatSettings();
        const savedModelpacks = await loadModelpackInstalls();
        if (mounted) {
          setSettings(ls);
          setModelpackInstalls(savedModelpacks);
        }

        const onboardingCompleted = await getOnboardingCompleted();
        if (mounted) {
          setShowOnboarding(!onboardingCompleted);
        }

        const lastApiUrl = await getLastApiUrl();
        if (mounted) {
          await connectBackend(lastApiUrl);
          await ensureActiveConversation();
        }
      } catch (e) {
        if (mounted) setGlobalError(e instanceof Error ? e.message : String(e));
      }
    })();
    const interval = setInterval(() => void refreshModelStatus(), 15000);
    return () => { mounted = false; clearInterval(interval); };
  }, [connectBackend, ensureActiveConversation, refreshModelStatus]);

  useEffect(() => {
    if (!settingsVisible) {
      return;
    }
    if (modelpackManifest === null) {
      void loadModelpackManifest();
    }
  }, [settingsVisible, modelpackManifest, loadModelpackManifest]);

  useEffect(() => { void saveChatSettings(settings); }, [settings]);
  useEffect(() => { void saveModelpackInstalls(modelpackInstalls); }, [modelpackInstalls]);

  // Load active conversation messages
  useEffect(() => {
    if (!activeConversationId) return;
    let mounted = true;
    (async () => {
      try {
        const conv = await getConversation(activeConversationId);
        if (!mounted) return;
        if (conv?.messages?.length) setMessages(conv.messages);
        else setMessages([{ id: 'starter', role: 'assistant', content: STARTER_TEXT, timestamp: Date.now() }]);
      } catch (e) {
        if (mounted) setGlobalError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { mounted = false; };
  }, [activeConversationId]);

  // ─── Chat actions ────────────────────────────────────────────────────────────
  const saveActiveConversation = useCallback(async (nextMessages: ChatMessage[]) => {
    if (!activeConversationId) return;
    const title = inferTitle(nextMessages);
    await saveConversation(nextMessages, activeConversationId, title);
    setConversations((prev) => {
      const next: ConvSummary = { id: activeConversationId, title, timestamp: Date.now() };
      return [next, ...prev.filter((c) => c.id !== activeConversationId)];
    });
  }, [activeConversationId]);

  const createNewConversation = useCallback(async () => {
    const id = `conv_${Date.now()}`;
    const starter: ChatMessage[] = [{ id: 'starter', role: 'assistant', content: STARTER_TEXT, timestamp: Date.now() }];
    await saveConversation(starter, id, 'New Chat');
    await saveCurrentConversationId(id);
    setConversations((prev) => [{ id, title: 'New Chat', timestamp: Date.now() }, ...prev]);
    setActiveConversationId(id);
    setMessages(starter);
  }, []);

  const selectProvider = useCallback(async (provider: LLMProviderType) => {
    if ((provider === 'local' || provider === 'gemma_local') && !localModelAvailable) {
      setGlobalError('Local model unavailable.');
      return;
    }

    setProviderBusy(true);
    try {
      await api.setLLMProvider(provider);
      setSettings((prev) => ({ ...prev, llmProvider: provider }));
      setGlobalError('');
      await Promise.all([
        refreshModelStatus(),
        syncLLMProvider(),
      ]);
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setProviderBusy(false);
    }
  }, [api, localModelAvailable, refreshModelStatus, syncLLMProvider]);

  const toggleProvider = useCallback(async () => {
    const providerCycle: LLMProviderType[] = localModelAvailable
      ? ['local', 'gemma_local', 'gemini']
      : ['gemini'];
    const currentIndex = providerCycle.indexOf(settings.llmProvider);
    const safeIndex = currentIndex >= 0 ? currentIndex : providerCycle.length - 1;
    const next = providerCycle[(safeIndex + 1) % providerCycle.length];
    await selectProvider(next);
  }, [localModelAvailable, selectProvider, settings.llmProvider]);

  const sendChat = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !activeConversationId) return;
    setGlobalError('');
    setSending(true);
    const userMessage: ChatMessage = { id: `u_${Date.now()}`, role: 'user', content: text, timestamp: Date.now() };
    const assistantId = `a_${Date.now()}`;
    const baseMessages = [...messagesRef.current, userMessage];
    setInput('');
    setMessages([...baseMessages]);

    try {
      if (settings.stream) {
        let assistantContent = '';
        let latestMeta: RAGStreamMeta | null = null;
        setStreamingMessageId(assistantId);
        setMessages([...baseMessages, { id: assistantId, role: 'assistant', content: '', timestamp: Date.now(), isStreaming: true }]);

        await api.streamMessage(
          { messages: baseMessages.map((m) => ({ role: m.role, content: m.content })), settings },
          {
            onToken: (token) => {
              assistantContent += token;
              setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, content: assistantContent, isStreaming: true } : m));
            },
            onMeta: (meta) => {
              latestMeta = { ...latestMeta, ...meta };
              setMessages((prev) => prev.map((m) => m.id === assistantId
                ? { ...m, thinking: meta.thinking ?? m.thinking, evidence: meta.evidence ?? m.evidence, agentsUsed: meta.agents_used ?? m.agentsUsed, confidence: typeof meta.confidence === 'number' ? meta.confidence : m.confidence, queryAnalysis: meta.query_analysis ?? m.queryAnalysis }
                : m));
            },
            onReplace: (r) => {
              assistantContent = r;
              setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, content: r, isStreaming: true } : m));
            },
            onDone: () => {
              setMessages((prev) => {
                const finalized = prev.map((m) => m.id === assistantId
                  ? { ...m, content: assistantContent, isStreaming: false, thinking: latestMeta?.thinking ?? m.thinking, evidence: latestMeta?.evidence ?? m.evidence, agentsUsed: latestMeta?.agents_used ?? m.agentsUsed, confidence: typeof latestMeta?.confidence === 'number' ? latestMeta.confidence : m.confidence, queryAnalysis: latestMeta?.query_analysis ?? m.queryAnalysis }
                  : m);
                void saveActiveConversation(finalized);
                return finalized;
              });
            },
            onError: (error) => {
              setGlobalError(error.message);
              setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, isStreaming: false } : m));
            },
          }
        );
      } else {
        const completion = await api.sendMessage({ messages: baseMessages.map((m) => ({ role: m.role, content: m.content })), settings });
        const assistantMessage: ChatMessage = {
          id: assistantId, role: 'assistant', content: completion.content,
          thinking: completion.thinking, evidence: completion.evidence,
          agentsUsed: completion.agents_used, confidence: completion.confidence,
          queryAnalysis: completion.query_analysis, processingTimeMs: completion.processing_time_ms,
          cacheHit: completion.cache_hit, pipelineTrace: completion.pipeline_trace || undefined,
          timestamp: Date.now(),
        };
        const finalized = [...baseMessages, assistantMessage];
        setMessages(finalized);
        await saveActiveConversation(finalized);
      }
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setStreamingMessageId(null);
      setSending(false);
    }
  }, [activeConversationId, api, input, saveActiveConversation, sending, settings]);

  // ─── Data loaders ────────────────────────────────────────────────────────────
  const loadMemories = useCallback(async () => {
    setLoadingView(true);
    try { const data = await api.getMemories(50, 0); setMemories(data.memories || []); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setLoadingView(false); }
  }, [api]);

  const searchMemories = useCallback(async () => {
    if (!memorySearch.trim()) { await loadMemories(); return; }
    setMemoryBusy(true);
    try { const data = await api.searchMemories(memorySearch.trim(), 20); setMemories(data.results || []); setGlobalError(''); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setMemoryBusy(false); }
  }, [api, loadMemories, memorySearch]);

  const addMemory = useCallback(async () => {
    if (!memoryDraft.trim()) return;
    setMemoryBusy(true);
    try {
      const result = await api.ingestMemory(
        memoryDraft.trim(),
        'manual_memory',
        {
          platform: Platform.OS,
          forceKeep: true,
        },
      );
      setLastMemoryRetention(result.retention_trace || null);
      setLastMemorySession(result.session || null);
      setLastMemoryDocument('');
      setMemoryDraft('');
      await loadMemories();
      setGlobalError('');
    }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setMemoryBusy(false); }
  }, [api, loadMemories, memoryDraft]);

  const removeMemory = useCallback(async (id: string) => {
    setMemoryBusy(true);
    try { await api.deleteMemory(id); setMemories((prev) => prev.filter((m) => m.id !== id)); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setMemoryBusy(false); }
  }, [api]);

  const loadGraph = useCallback(async () => {
    setLoadingView(true);
    try { setGraphData(await api.getGraphData()); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setLoadingView(false); }
  }, [api]);

  const loadDashboard = useCallback(async () => {
    setLoadingView(true);
    try {
      const [statsR, graphR, docsR] = await Promise.allSettled([
        api.getRAGStats(),
        api.getGraphData(),
        api.listDocuments(),
      ]);

      if (statsR.status === 'fulfilled') {
        setRagStats(statsR.value);
      } else {
        throw statsR.reason;
      }

      if (graphR.status === 'fulfilled') {
        setGraphData(graphR.value);
      }

      if (docsR.status === 'fulfilled') {
        setDocuments(docsR.value.documents || []);
        setPageIndexEnabled(Boolean(docsR.value.pageindex_enabled));
      }
    }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setLoadingView(false); }
  }, [api]);

  const loadObservability = useCallback(async (silent = false) => {
    if (!silent) setLoadingView(true);
    try { setObservabilityMetrics(await api.getObservabilityMetrics()); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { if (!silent) setLoadingView(false); }
  }, [api]);

  const loadAmbient = useCallback(async (silent = false) => {
    if (!silent) setLoadingView(true);
    try {
      const [statusR, liveStatusR, configR, providersR, enrollmentR, transcriptR, conversationsR, ttsR] = await Promise.allSettled([
        api.getAmbientStatus(), api.getAmbientLiveStatus(), api.getAmbientConfig(), api.getVoiceProviders(),
        api.getEnrollmentStatus(), api.getLiveTranscript(), api.getConversations(20, 0), api.getTTSStatus(),
      ]);
      if (statusR.status === 'fulfilled') setAmbientState(statusR.value); else throw statusR.reason;
      if (liveStatusR.status === 'fulfilled') {
        setAmbientLiveStatus(liveStatusR.value);
      } else if (statusR.status === 'fulfilled') {
        setAmbientLiveStatus(statusR.value.live || null);
      }
      setAmbientConfig(configR.status === 'fulfilled' ? configR.value : null);
      setAmbientProviders(providersR.status === 'fulfilled' ? providersR.value : null);
      setAmbientEnrollment(enrollmentR.status === 'fulfilled' ? enrollmentR.value : null);
      setAmbientTurns(transcriptR.status === 'fulfilled' ? transcriptR.value.turns || [] : []);
      setAmbientConversations(conversationsR.status === 'fulfilled' ? conversationsR.value.conversations || [] : []);
      setAmbientTTSStatus(ttsR.status === 'fulfilled' ? ttsR.value : null);
    } catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { if (!silent) setLoadingView(false); }
  }, [api]);

  const runAmbientAction = useCallback(async (action: 'start' | 'stop' | 'pause' | 'resume') => {
    setAmbientBusy(true);
    try { await api.ambientAction(action); await loadAmbient(true); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setAmbientBusy(false); }
  }, [api, loadAmbient]);

  const runAmbientLiveAction = useCallback(async (action: 'start' | 'stop') => {
    setAmbientBusy(true);
    try {
      if (action === 'start') {
        await api.startAmbientLive();
      } else {
        await api.stopAmbientLive();
      }
      await loadAmbient(true);
      setGlobalError('');
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setAmbientBusy(false);
    }
  }, [api, loadAmbient]);

  const setAmbientProvider = useCallback(async (kind: 'stt' | 'tts', provider: 'traditional' | 'local' | 'gemini') => {
    setAmbientBusy(true);
    try {
      if (kind === 'stt') await api.setSTTProvider(provider); else await api.setTTSProvider(provider);
      await loadAmbient(true); setGlobalError('');
    } catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setAmbientBusy(false); }
  }, [api, loadAmbient]);

  const startAmbientEnrollment = useCallback(async () => {
    setAmbientBusy(true);
    try { await api.startEnrollment(20); await loadAmbient(true); setGlobalError(''); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setAmbientBusy(false); }
  }, [api, loadAmbient]);

  const toggleAmbientAutoIngest = useCallback(async () => {
    if (!ambientConfig) return;
    setAmbientBusy(true);
    try { const updated = await api.updateAmbientConfig({ auto_ingest: !ambientConfig.auto_ingest }); setAmbientConfig(updated); await loadAmbient(true); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setAmbientBusy(false); }
  }, [ambientConfig, api, loadAmbient]);

  const runTTSHealthCheck = useCallback(async () => {
    const text = ttsDraft.trim();
    if (!text) return;
    setTtsBusy(true);
    try { const bytes = await api.synthesizeSpeech(text); setTtsLastBytes(bytes.byteLength); setGlobalError(''); await loadAmbient(true); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setTtsBusy(false); }
  }, [api, loadAmbient, ttsDraft]);

  const loadDocuments = useCallback(async () => {
    setLoadingView(true);
    try {
      const [docs, usage] = await Promise.all([api.listDocuments(), api.getPageIndexUsage()]);
      setDocuments(docs.documents || []);
      setPageIndexEnabled(Boolean(docs.pageindex_enabled));
      setPageIndexUsage(usage.usage || null);
    } catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setLoadingView(false); }
  }, [api]);

  const uploadDocument = useCallback(async (surface: 'documents' | 'memory' = 'documents') => {
    try {
      setDocumentsBusy(true);
      const picked = await DocumentPicker.getDocumentAsync({ type: 'application/pdf', copyToCacheDirectory: true, multiple: false });
      if (picked.canceled || picked.assets.length === 0) return;
      const asset = picked.assets[0] as DocumentPicker.DocumentPickerAsset & { file?: Blob };
      const form = new FormData();
      if (asset.file) form.append('file', asset.file, asset.name || 'document.pdf');
      else form.append('file', { uri: asset.uri, name: asset.name || 'document.pdf', type: 'application/pdf' } as unknown as Blob);
      const uploaded = await api.uploadDocument(form);

      if (surface === 'memory') {
        const intake = await api.ingestMemory(
          `Uploaded document "${uploaded.filename || asset.name || 'document.pdf'}" into PageIndex for long-term retrieval and memory refinement.`,
          'pageindex_document',
          {
            platform: Platform.OS,
            forceKeep: true,
          },
        );
        setLastMemoryRetention(intake.retention_trace || null);
        setLastMemorySession(intake.session || null);
        setLastMemoryDocument(uploaded.filename || asset.name || 'document.pdf');
        await loadMemories();
      }

      await loadDocuments();
      setGlobalError('');
    } catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setDocumentsBusy(false); }
  }, [api, loadDocuments, loadMemories]);

  const installModelpack = useCallback(async (pack: ModelpackEntry) => {
    if (Platform.OS === 'web') {
      setGlobalError('In-app model downloads require an installed iOS or Android build.');
      return;
    }
    if (!documentDirectory) {
      setGlobalError('App storage is unavailable on this build, so model packs cannot be downloaded.');
      return;
    }

    const files = inferModelpackFiles(pack);
    if (!pack.download_url || files.length === 0) {
      setGlobalError('This model pack is missing direct artifact files, so it cannot be downloaded in-app yet.');
      return;
    }

    const updateInstallState = (patch: Partial<ModelpackInstallState>) => {
      setModelpackInstalls((prev) => ({
        ...prev,
        [pack.id]: {
          packId: pack.id,
          status: patch.status || prev[pack.id]?.status || 'queued',
          progress: typeof patch.progress === 'number' ? patch.progress : prev[pack.id]?.progress || 0,
          updatedAt: new Date().toISOString(),
          artifactPaths: patch.artifactPaths ?? prev[pack.id]?.artifactPaths,
          activeFile: patch.activeFile ?? prev[pack.id]?.activeFile,
          error: patch.error ?? prev[pack.id]?.error,
        },
      }));
    };

    try {
      setGlobalError('');
      updateInstallState({ status: 'queued', progress: 0, error: '' });

      const packDir = `${documentDirectory}modelpacks/${pack.id}/`;
      await makeDirectoryAsync(packDir, { intermediates: true });

      const artifactPaths: string[] = [];
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        const filename = file.path.split('/').pop() || `${pack.id}-${index + 1}.bin`;
        const destination = `${packDir}${filename}`;
        const artifactUrl = buildModelpackArtifactUrl(pack, file.path);

        updateInstallState({
          status: 'downloading',
          activeFile: filename,
          progress: (index / files.length) * 100,
        });

        const download = createDownloadResumable(
          artifactUrl,
          destination,
          {},
          (progressEvent) => {
            const total = progressEvent.totalBytesExpectedToWrite || 0;
            const current = total > 0 ? progressEvent.totalBytesWritten / total : 0;
            updateInstallState({
              status: 'downloading',
              activeFile: filename,
              progress: ((index + current) / files.length) * 100,
            });
          },
        );

        const result = await download.downloadAsync();
        if (!result?.uri) {
          throw new Error(`Download did not produce a local file for ${filename}.`);
        }

        if (file.sha256) {
          const verified = await api.verifyModelpack(result.uri, file.sha256);
          if (!verified.verified) {
            throw new Error(`Checksum verification failed for ${filename}.`);
          }
        }

        artifactPaths.push(result.uri);
      }

      updateInstallState({
        status: 'installed',
        progress: 100,
        artifactPaths,
        activeFile: '',
        error: '',
      });

      if (!localModelAvailable) {
        setGlobalError(
          'Model pack downloaded to this device. The current deployed backend is Gemini-only, so chat will stay on Gemini until a native local runtime or local backend is available.',
        );
      }
    } catch (e) {
      updateInstallState({
        status: 'error',
        error: e instanceof Error ? e.message : String(e),
      });
      setGlobalError(e instanceof Error ? e.message : String(e));
    }
  }, [api, localModelAvailable]);

  const deleteDocument = useCallback(async (docId: string) => {
    setDocumentsBusy(true);
    try { await api.deleteDocument(docId); if (documentTreeDocId === docId) { setDocumentTreeDocId(null); setDocumentTreePreview([]); } await loadDocuments(); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setDocumentsBusy(false); }
  }, [api, documentTreeDocId, loadDocuments]);

  const toggleDocumentTree = useCallback(async (docId: string) => {
    if (documentTreeDocId === docId) { setDocumentTreeDocId(null); setDocumentTreePreview([]); return; }
    setDocumentsBusy(true);
    try {
      const treeResponse = await api.getDocumentTree(docId);
      const treePayload = treeResponse.tree as { result?: Array<Record<string, unknown>> } | null;
      const preview = (treePayload?.result || []).slice(0, 10).map((node, i) => {
        const title = typeof node.title === 'string' ? node.title : `Section ${i + 1}`;
        const summary = typeof node.summary === 'string' ? node.summary : typeof node.text === 'string' ? node.text : '';
        const page = typeof node.page_index === 'number' ? `p.${node.page_index}` : '';
        return `${page ? `${page} · ` : ''}${title}${summary ? ` — ${summary.slice(0, 100)}` : ''}`;
      });
      setDocumentTreeDocId(docId); setDocumentTreePreview(preview); setGlobalError('');
    } catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setDocumentsBusy(false); }
  }, [api, documentTreeDocId]);

  const runDocumentQuery = useCallback(async () => {
    const query = documentQuery.trim();
    if (!query) return;
    setDocumentQueryBusy(true);
    try { const result = await api.queryDocuments(query, 5); setDocumentAnswer(result.answer || ''); setDocumentSections(result.sections || []); }
    catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setDocumentQueryBusy(false); }
  }, [api, documentQuery]);

  const reconnectBackend = useCallback(async () => {
    try {
      const persistedUrl = await getLastApiUrl();
      await connectBackend(persistedUrl);
      setGlobalError('');
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : String(e));
      setConnectionStatus('Automatic backend reconnection failed.');
      setTestingConnection(false);
    }
  }, [connectBackend]);

  const completeOnboarding = useCallback(async () => {
    await saveOnboardingCompleted(true);
    setShowOnboarding(false);
  }, []);

  // View-based data loading
  useEffect(() => {
    if (activeView === 'memories')      void loadMemories();
    else if (activeView === 'graph')    void loadGraph();
    else if (activeView === 'dashboard') void loadDashboard();
    else if (activeView === 'observability') void loadObservability();
    else if (activeView === 'ambient')  void loadAmbient();
    else if (activeView === 'documents') void loadDocuments();
  }, [activeView, loadMemories, loadGraph, loadDashboard, loadObservability, loadAmbient, loadDocuments]);

  // Observability live polling
  useEffect(() => {
    if (activeView !== 'observability') return;
    setPipelineEvents([]);
    if (Platform.OS !== 'web') {
      const interval = setInterval(() => void loadObservability(true), 4000);
      return () => clearInterval(interval);
    }
    const controller = api.subscribePipelineEvents(
      (event) => setPipelineEvents((prev) => [event, ...prev].slice(0, 60)),
      (error) => setGlobalError(error.message),
    );
    const interval = setInterval(() => void loadObservability(true), 6000);
    return () => { controller.abort(); clearInterval(interval); };
  }, [activeView, api, loadObservability]);

  // Ambient auto-refresh
  useEffect(() => {
    if (activeView !== 'ambient') return;
    const interval = setInterval(() => void loadAmbient(true), 2500);
    return () => clearInterval(interval);
  }, [activeView, loadAmbient]);

  const isOnline = modelStatus.status !== 'offline';

  // ─── Render ──────────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <StatusBar style="dark" />

      {settingsVisible ? (
        <SettingsPage
          settings={settings}
          onUpdateSettings={(patch: Partial<ChatSettings>) => setSettings((prev) => ({ ...prev, ...patch }))}
          onSelectLLMProvider={(provider: LLMProviderType) => void selectProvider(provider)}
          onBack={() => setSettingsVisible(false)}
          onReconnect={() => void reconnectBackend()}
          reconnecting={testingConnection}
          connectionStatus={connectionStatus}
          backendUrlLabel={backendUrlLabel}
          localModelAvailable={localModelAvailable}
          modelpackManifest={modelpackManifest}
          modelpackInstalls={modelpackInstalls}
          modelpackCapabilityMessage={
            localModelAvailable
              ? 'A local backend/runtime is available. Pick Local or Gemma Local for the next chat request.'
              : 'The deployed backend is currently Gemini-only. Downloaded model packs stay on-device for the upcoming native runtime bridge, while live chat continues through Gemini.'
          }
          modelpackError={modelpackError}
          onRefreshModelpacks={() => void loadModelpackManifest()}
          onInstallModelpack={(pack: ModelpackEntry) => void installModelpack(pack)}
        />
      ) : (
        <>
          <Header
            modelStatus={modelStatus}
            title="Cortex Lab"
            subtitle={activeView !== 'chat' ? NAV_ITEMS.find((n) => n.key === activeView)?.label : undefined}
            onMenuPress={() => setDrawerVisible(true)}
            onSettingsPress={() => setSettingsVisible(true)}
          />

          <NetworkStatusBanner />

          {/* Screen content */}
          {activeView === 'chat' && (
            <ChatScreen
              messages={messages}
              input={input}
              setInput={setInput}
              sending={sending}
              streamingMessageId={streamingMessageId}
              settings={settings}
              modelStatus={modelStatus}
              globalError={globalError}
              onSend={sendChat}
              onToggleProvider={toggleProvider}
              onToggleRAG={() => setSettings((p) => ({ ...p, useRAG: !p.useRAG }))}
              onToggleStream={() => setSettings((p) => ({ ...p, stream: !p.stream }))}
              providerBusy={providerBusy}
              localModelAvailable={localModelAvailable}
            />
          )}
          {activeView === 'memories' && (
            <MemoryScreen
              memories={memories}
              memorySearch={memorySearch}
              setMemorySearch={setMemorySearch}
              memoryDraft={memoryDraft}
              setMemoryDraft={setMemoryDraft}
              memoryBusy={memoryBusy}
              loadingView={loadingView}
              lastRetentionTrace={lastMemoryRetention}
              lastSession={lastMemorySession}
              lastUploadedDocument={lastMemoryDocument}
              onSearch={searchMemories}
              onAddMemory={addMemory}
              onUploadDocument={() => void uploadDocument('memory')}
              onDeleteMemory={removeMemory}
              onLoadMore={() => void loadMemories()}
            />
          )}
          {activeView === 'graph' && <GraphScreen graphData={graphData} loadingView={loadingView} />}
          {activeView === 'dashboard' && (
            <DashboardScreen
              ragStats={ragStats}
              graphData={graphData}
              documentCount={documents.length}
              apiBaseUrl={apiBase}
              modelStatus={modelStatus}
              loadingView={loadingView}
              onRefresh={loadDashboard}
              onOpenView={setActiveView}
            />
          )}
          {activeView === 'observability' && (
            <ObservabilityScreen
              observabilityMetrics={observabilityMetrics}
              pipelineEvents={pipelineEvents}
              loadingView={loadingView}
              apiBaseUrl={apiBase}
              api={api}
            />
          )}
          {activeView === 'agent' && <AgentScreen api={api} />}
          {activeView === 'wiki' && <WikiScreen api={api} />}
          {activeView === 'session-forge' && <SessionForgeScreen api={api} />}
          {activeView === 'chronicle' && <ChronicleScreen api={api} />}
          {activeView === 'ambient' && (
            <AmbientVoiceScreen
              ambientState={ambientState}
              ambientLiveStatus={ambientLiveStatus}
              ambientConfig={ambientConfig}
              voiceProviders={ambientProviders}
              onStartListening={() => runAmbientLiveAction('start')}
              onStopListening={() => runAmbientLiveAction('stop')}
              onPauseAmbient={() => runAmbientAction('pause')}
              onResumeAmbient={() => runAmbientAction('resume')}
              api={api}
            />
          )}
          {activeView === 'documents' && (
            <DocumentsScreen
              documents={documents}
              documentUsage={pageIndexUsage}
              onLoadDocuments={loadDocuments}
              onDeleteDocument={deleteDocument}
              onUploadDocument={() => void uploadDocument('documents')}
              loadingView={loadingView}
              api={api}
            />
          )}

          {/* Bottom navigation */}
          <BottomNav
            items={NAV_ITEMS}
            activeKey={activeView}
            onSelect={setActiveView}
          />

          {/* Modals */}
          <ConversationDrawer
            visible={drawerVisible}
            onClose={() => setDrawerVisible(false)}
            conversations={conversations}
            activeConversationId={activeConversationId}
            onSelectConversation={(id) => { setActiveConversationId(id); void saveCurrentConversationId(id); }}
            onNewChat={createNewConversation}
            isOnline={isOnline}
            onOpenSettings={() => setSettingsVisible(true)}
          />
        </>
      )}

      {showOnboarding && (
        <OnboardingScreen onContinue={() => void completeOnboarding()} />
      )}
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <NetworkProvider>
        <ErrorBoundary>
          <AppContent />
        </ErrorBoundary>
      </NetworkProvider>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
});
