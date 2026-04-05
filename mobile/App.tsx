/**
 * Cortex Lab Mobile — App.tsx Orchestrator
 * Neural Dark design system — clean screen routing
 * Stitch ref (App Shell Overview): 440ef4a948904807bfc5a55b0b027242
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
  createApiClient,
  getDefaultApiBaseUrl,
  type PageIndexDocument,
  type PageIndexUsage,
  type RAGStreamMeta,
  type TTSStatus,
} from './shared/core/api';
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
} from './shared/core/types';
import { DEFAULT_SETTINGS } from './shared/core/types';
import {
  getAllConversations,
  getConversation,
  getCurrentConversationId,
  getLastApiUrl,
  getOnboardingCompleted,
  initializeStorage,
  loadChatSettings,
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
import { AmbientVoiceScreen } from './src/screens/AmbientVoiceScreen';
import { DocumentsScreen } from './src/screens/DocumentsScreen';
import { OnboardingScreen } from './src/screens/OnboardingScreen';

// ── Modals ────────────────────────────────────────────────────────────────────
import { ConversationDrawer } from './src/modals/ConversationDrawer';

// ── Theme ─────────────────────────────────────────────────────────────────────
import { NEURAL } from './src/theme/colors';

// ─── Nav items with icons ─────────────────────────────────────────────────────
const NAV_ITEMS = [
  { key: 'chat'          as NavKey, label: 'Chat',    iconName: 'chat-processing-outline' as const },
  { key: 'memories'      as NavKey, label: 'Memory',  iconName: 'brain' as const },
  { key: 'graph'         as NavKey, label: 'Graph',   iconName: 'graph-outline' as const },
  { key: 'dashboard'     as NavKey, label: 'RAG',     iconName: 'view-dashboard-outline' as const },
  { key: 'observability' as NavKey, label: 'Observe', iconName: 'chart-timeline-variant' as const },
  { key: 'ambient'       as NavKey, label: 'Voice',   iconName: 'microphone-outline' as const },
  { key: 'documents'     as NavKey, label: 'Docs',    iconName: 'file-document-outline' as const },
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
  onBack: () => void;
  onSave: (backendUrl: string) => void;
  onTestConnection: (backendUrl: string) => void;
  testingConnection: boolean;
  connectionStatus: string;
  backendUrl: string;
}

const SettingsPage = require('./src/screens/SettingsScreen').SettingsScreen as React.ComponentType<SettingsPageProps>;

// ─── Main App Content ─────────────────────────────────────────────────────────
function AppContent() {
  const [apiBase, setApiBase] = useState<string>(getDefaultApiBaseUrl());
  const api = useMemo(() => createApiClient({ baseUrl: apiBase }), [apiBase]);

  // ── Navigation ──────────────────────────────────────────────────────────────
  const [activeView, setActiveView] = useState<NavKey>('chat');
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('');
  const [backendUrl, setBackendUrl] = useState(apiBase);

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

  // ── Graph / Dashboard ─────────────────────────────────────────────────────────
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [ragStats, setRagStats] = useState<RAGStats | null>(null);

  // ── Observability ─────────────────────────────────────────────────────────────
  const [observabilityMetrics, setObservabilityMetrics] = useState<Record<string, unknown> | null>(null);
  const [pipelineEvents, setPipelineEvents] = useState<LivePipelineEvent[]>([]);

  // ── Ambient ───────────────────────────────────────────────────────────────────
  const [ambientState, setAmbientState] = useState<AmbientState | null>(null);
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
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  const refreshModelStatus = useCallback(async () => {
    try {
      const status = await api.getModelStatus();
      setModelStatus(status);
    } catch {
      setModelStatus({ status: 'offline', model_loaded: false, model_info: {} });
    }
  }, [api]);

  const syncLLMProvider = useCallback(async () => {
    try {
      const provider = await api.getLLMProvider();
      setLocalModelAvailable(provider.local_model_loaded);
      setSettings((prev) => {
        if (provider.provider === 'local' && !provider.local_model_loaded && provider.gemini_configured) {
          return { ...prev, llmProvider: 'gemini' };
        }
        return { ...prev, llmProvider: provider.provider };
      });
    } catch {}
  }, [api]);

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
        if (mounted) {
          setSettings(ls);
        }

        const onboardingCompleted = await getOnboardingCompleted();
        if (mounted) {
          setShowOnboarding(!onboardingCompleted);
        }

        const lastApiUrl = await getLastApiUrl();
        if (mounted && lastApiUrl?.trim()) {
          const persistedUrl = lastApiUrl.trim();
          setApiBase(persistedUrl);
          setBackendUrl(persistedUrl);
        }

        await Promise.all([refreshModelStatus(), syncLLMProvider(), ensureActiveConversation()]);
      } catch (e) {
        if (mounted) setGlobalError(e instanceof Error ? e.message : String(e));
      }
    })();
    const interval = setInterval(() => void refreshModelStatus(), 15000);
    return () => { mounted = false; clearInterval(interval); };
  }, [refreshModelStatus, syncLLMProvider, ensureActiveConversation]);

  useEffect(() => { void saveChatSettings(settings); }, [settings]);

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

  const toggleProvider = useCallback(async () => {
    const next = settings.llmProvider === 'local' ? 'gemini' : 'local';
    if (next === 'local' && !localModelAvailable) { setGlobalError('Local model unavailable.'); return; }
    setProviderBusy(true);
    try {
      await api.setLLMProvider(next);
      setSettings((prev) => ({ ...prev, llmProvider: next }));
      setGlobalError('');
      await refreshModelStatus();
    } catch (e) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    } finally {
      setProviderBusy(false);
    }
  }, [api, localModelAvailable, refreshModelStatus, settings.llmProvider]);

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
    try { await api.ingestMemory(memoryDraft.trim(), 'mobile'); setMemoryDraft(''); await loadMemories(); setGlobalError(''); }
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
    try { setRagStats(await api.getRAGStats()); }
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
      const [statusR, configR, providersR, enrollmentR, transcriptR, conversationsR, ttsR] = await Promise.allSettled([
        api.getAmbientStatus(), api.getAmbientConfig(), api.getVoiceProviders(),
        api.getEnrollmentStatus(), api.getLiveTranscript(), api.getConversations(20, 0), api.getTTSStatus(),
      ]);
      if (statusR.status === 'fulfilled') setAmbientState(statusR.value); else throw statusR.reason;
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

  const setAmbientProvider = useCallback(async (kind: 'stt' | 'tts', provider: 'traditional' | 'gemini') => {
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

  const uploadDocument = useCallback(async () => {
    try {
      setDocumentsBusy(true);
      const picked = await DocumentPicker.getDocumentAsync({ type: 'application/pdf', copyToCacheDirectory: true, multiple: false });
      if (picked.canceled || picked.assets.length === 0) return;
      const asset = picked.assets[0] as DocumentPicker.DocumentPickerAsset & { file?: Blob };
      const form = new FormData();
      if (asset.file) form.append('file', asset.file, asset.name || 'document.pdf');
      else form.append('file', { uri: asset.uri, name: asset.name || 'document.pdf', type: 'application/pdf' } as unknown as Blob);
      await api.uploadDocument(form);
      await loadDocuments();
    } catch (e) { setGlobalError(e instanceof Error ? e.message : String(e)); }
    finally { setDocumentsBusy(false); }
  }, [api, loadDocuments]);

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

  const normalizeBackendUrlInput = useCallback((rawUrl: string) => {
    const fallback = apiBase.trim();
    const trimmed = rawUrl.trim();
    const candidate = trimmed || fallback;
    const withoutTrailingSlash = candidate.endsWith('/') ? candidate.slice(0, -1) : candidate;
    return withoutTrailingSlash.endsWith('/api') ? withoutTrailingSlash : `${withoutTrailingSlash}/api`;
  }, [apiBase]);

  // Connection test
  const testConnection = useCallback(async (urlDraft: string) => {
    setTestingConnection(true);
    try {
      const targetUrl = normalizeBackendUrlInput(urlDraft);
      const testApi = createApiClient({ baseUrl: targetUrl });
      const status = await testApi.getModelStatus();
      setConnectionStatus(`Connected · ${status.status}`);
      setModelStatus(status);
      setApiBase(targetUrl);
      setBackendUrl(targetUrl);
      setGlobalError('');
      await saveLastApiUrl(targetUrl);
    } catch {
      setConnectionStatus('Connection failed');
    } finally {
      setTestingConnection(false);
    }
  }, [normalizeBackendUrlInput]);

  const handleSaveSettings = useCallback(async (urlDraft: string) => {
    const targetUrl = normalizeBackendUrlInput(urlDraft);
    setApiBase(targetUrl);
    setBackendUrl(targetUrl);
    await saveLastApiUrl(targetUrl);
    setConnectionStatus('');
  }, [normalizeBackendUrlInput]);

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

  const isOnline = modelStatus.status === 'ready' || modelStatus.status === 'gemini' || modelStatus.model_loaded;

  // ─── Render ──────────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <StatusBar style="light" />

      {settingsVisible ? (
        <SettingsPage
          settings={settings}
          onUpdateSettings={(patch: Partial<ChatSettings>) => setSettings((prev) => ({ ...prev, ...patch }))}
          onBack={() => setSettingsVisible(false)}
          onSave={(url: string) => void handleSaveSettings(url)}
          onTestConnection={(url: string) => void testConnection(url)}
          testingConnection={testingConnection}
          connectionStatus={connectionStatus}
          backendUrl={backendUrl}
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
              onSearch={searchMemories}
              onAddMemory={addMemory}
              onDeleteMemory={removeMemory}
              onLoadMore={() => void loadMemories()}
            />
          )}
          {activeView === 'graph' && <GraphScreen graphData={graphData} loadingView={loadingView} />}
          {activeView === 'dashboard' && <DashboardScreen ragStats={ragStats} loadingView={loadingView} onRefresh={loadDashboard} />}
          {activeView === 'observability' && (
            <ObservabilityScreen
              observabilityMetrics={observabilityMetrics}
              pipelineEvents={pipelineEvents}
              loadingView={loadingView}
              apiBaseUrl={apiBase}
            />
          )}
          {activeView === 'ambient' && (
            <AmbientVoiceScreen
              ambientState={ambientState}
              ambientConfig={ambientConfig}
              ambientEnrollment={ambientEnrollment}
              ambientProviders={ambientProviders}
              ambientTurns={ambientTurns}
              ambientConversations={ambientConversations}
              ambientTTSStatus={ambientTTSStatus}
              ambientBusy={ambientBusy}
              loadingView={loadingView}
              ttsDraft={ttsDraft}
              setTtsDraft={setTtsDraft}
              ttsBusy={ttsBusy}
              ttsLastBytes={ttsLastBytes}
              onAmbientAction={runAmbientAction}
              onSetProvider={setAmbientProvider}
              onStartEnrollment={startAmbientEnrollment}
              onToggleAutoIngest={toggleAmbientAutoIngest}
              onRunTTS={runTTSHealthCheck}
            />
          )}
          {activeView === 'documents' && (
            <DocumentsScreen
              documents={documents}
              pageIndexUsage={pageIndexUsage}
              pageIndexEnabled={pageIndexEnabled}
              documentQuery={documentQuery}
              setDocumentQuery={setDocumentQuery}
              documentAnswer={documentAnswer}
              documentSections={documentSections}
              documentTreeDocId={documentTreeDocId}
              documentTreePreview={documentTreePreview}
              documentsBusy={documentsBusy}
              documentQueryBusy={documentQueryBusy}
              loadingView={loadingView}
              onUpload={uploadDocument}
              onDeleteDocument={deleteDocument}
              onToggleTree={toggleDocumentTree}
              onRunQuery={runDocumentQuery}
              onClearAnswer={() => { setDocumentAnswer(''); setDocumentSections([]); setDocumentQuery(''); }}
              onRefresh={loadDocuments}
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
      <AppContent />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: NEURAL.background,
  },
});
