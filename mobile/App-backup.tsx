import * as DocumentPicker from "expo-document-picker";
import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { createApiClient, getDefaultApiBaseUrl } from "./shared/core/api";
import {
  ChatSettings,
  DEFAULT_SETTINGS,
  MemoryObject,
  RAGStats,
} from "./shared/core/types";
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
import { MessageBubble } from "./src/components/MessageBubble";

interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ConversationSummary {
  id: string;
  title: string;
  timestamp: number;
}

type AppTab =
  | "chat"
  | "history"
  | "memories"
  | "documents"
  | "stats"
  | "settings"
  | "ambient";

const initialAssistantMessage: UiMessage = {
  id: "intro-assistant",
  role: "assistant",
  content:
    "Cortex Mobile is live. Streaming chat, settings, memory/documents, and ambient controls are now in migration baseline.",
};

export default function App() {
  const [activeTab, setActiveTab] = useState<AppTab>("chat");
  const [messages, setMessages] = useState<UiMessage[]>([initialAssistantMessage]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [settings, setSettings] = useState<ChatSettings>({
    ...DEFAULT_SETTINGS,
    stream: true,
    useRAG: true,
  });

  const [modelStatus, setModelStatus] = useState("checking");
  const [conversationId, setConversationId] = useState(`conv_${Date.now()}`);
  const [conversationHistory, setConversationHistory] = useState<ConversationSummary[]>([]);

  const [memories, setMemories] = useState<MemoryObject[]>([]);
  const [memoryQuery, setMemoryQuery] = useState("");

  const [documents, setDocuments] = useState<
    Array<{ doc_id: string; filename: string; status: string; uploaded_at: string }>
  >([]);
  const [uploadingDoc, setUploadingDoc] = useState(false);

  const [stats, setStats] = useState<RAGStats | null>(null);
  const [ambientStatus, setAmbientStatus] = useState<string>("idle");

  const [apiBase] = useState<string>(getDefaultApiBaseUrl());
  const api = useMemo(() => createApiClient({ baseUrl: apiBase }), [apiBase]);
  const messagesRef = useRef<UiMessage[]>(messages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const hydrateApp = useCallback(async () => {
    try {
      await initializeStorage();
      const loadedSettings = await loadChatSettings();
      setSettings(loadedSettings);

      const currentId = await getCurrentConversationId();
      if (currentId) {
        const conv = await getConversation(currentId);
        if (conv && conv.messages.length > 0) {
          setConversationId(currentId);
          setMessages(
            conv.messages.map((m) => ({
              id: m.id,
              role: m.role,
              content: m.content,
            }))
          );
        }
      }
    } catch (e) {
      console.error("❌ hydrateApp error:", e);
      setError(`Init error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  const refreshModelStatus = useCallback(async () => {
    try {
      setModelStatus("checking");
      const status = await api.getModelStatus();
      const provider = status?.model_info?.llm_provider;
      const loaded = status?.model_loaded;
      setModelStatus(loaded ? `online (${provider || "local"})` : "offline");
    } catch (e) {
      console.error("❌ refreshModelStatus error:", e);
      setModelStatus("offline (api unreachable)");
    }
  }, [api]);

  const loadTabData = useCallback(async () => {
    try {
      setError(null);
      if (activeTab === "history") {
        const all = await getAllConversations();
        const mapped = all
          .map((c) => ({
            id: c.id,
            title: c.title || "Untitled",
            timestamp: c.timestamp,
          }))
          .sort((a, b) => b.timestamp - a.timestamp);
        setConversationHistory(mapped);
      }

      if (activeTab === "memories") {
        if (memoryQuery.trim()) {
          const res = await api.searchMemories(memoryQuery.trim(), 25);
          setMemories(res.results || []);
        } else {
          const res = await api.getMemories(50, 0);
          setMemories(res.memories || []);
        }
      }

      if (activeTab === "documents") {
        const res = await api.listDocuments();
        setDocuments(
          (res.documents || []).map((d) => ({
            doc_id: d.doc_id,
            filename: d.filename,
            status: d.status,
            uploaded_at: d.uploaded_at,
          }))
        );
      }

      if (activeTab === "stats") {
        const res = await api.getRAGStats();
        setStats(res);
      }

      if (activeTab === "ambient") {
        const res = await api.getAmbientStatus();
        setAmbientStatus(res.status || "idle");
      }
    } catch (e) {
      console.error("❌ loadTabData error on tab", activeTab, ":", e);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [activeTab, api, memoryQuery]);

  useEffect(() => {
    void hydrateApp();
    void refreshModelStatus();
  }, [hydrateApp, refreshModelStatus]);

  useEffect(() => {
    void loadTabData();
  }, [loadTabData]);

  useEffect(() => {
    void saveChatSettings(settings);
  }, [settings]);

  const persistMessages = useCallback(
    async (nextMessages: UiMessage[]) => {
      const timestamp = Date.now();
      const chatMessages = nextMessages.map((m, idx) => ({
        id: m.id || `${timestamp}_${idx}`,
        role: m.role,
        content: m.content,
        timestamp,
      }));

      const userFirstMessage = nextMessages.find((m) => m.role === "user")?.content || "Chat";
      const nextId = await saveConversation(chatMessages, conversationId, userFirstMessage.slice(0, 40));
      setConversationId(nextId);
      await saveCurrentConversationId(nextId);
    },
    [conversationId]
  );

  const onSend = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || sending) {
      return;
    }

    setSending(true);
    setError(null);

    const userId = `u_${Date.now()}`;
    const assistantId = `a_${Date.now()}_pending`;

    const baseMessages: UiMessage[] = [...messagesRef.current, { id: userId, role: "user", content: trimmed }];
    setMessages([...baseMessages, { id: assistantId, role: "assistant", content: "" }]);
    setInput("");

    try {
      if (settings.stream) {
        let accumulatedResponse = "";
        await api.streamMessage(
          {
            messages: baseMessages.map((m) => ({ role: m.role, content: m.content })),
            settings,
          },
          {
            onToken: (token) => {
              accumulatedResponse += token;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, content: accumulatedResponse } : m))
              );
            },
            onDone: () => {
              setMessages((prev) => {
                const finalized = prev.map((m) =>
                  m.id === assistantId ? { ...m, content: accumulatedResponse || "(empty response)" } : m
                );
                void persistMessages(finalized);
                return finalized;
              });
              setSending(false);
            },
            onError: (streamError) => {
              setError(streamError.message);
              setSending(false);
            },
          }
        );
      } else {
        const res = await api.sendMessage({
          messages: baseMessages.map((m) => ({ role: m.role, content: m.content })),
          settings,
        });

        const finalized: UiMessage[] = [
          ...baseMessages,
          { id: `a_${Date.now()}`, role: "assistant", content: res.content || "" },
        ];
        setMessages(finalized);
        await persistMessages(finalized);
        setSending(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSending(false);
    }
  }, [api, input, persistMessages, sending, settings]);

  const onNewConversation = useCallback(() => {
    const nextId = `conv_${Date.now()}`;
    setConversationId(nextId);
    setMessages([initialAssistantMessage]);
    void saveCurrentConversationId(nextId);
    setActiveTab("chat");
  }, []);

  const onSwitchConversation = useCallback(async (id: string) => {
    const conv = await getConversation(id);
    if (!conv) {
      return;
    }

    setConversationId(id);
    await saveCurrentConversationId(id);
    setMessages(
      conv.messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
      }))
    );
    setActiveTab("chat");
  }, []);

  const onPickAndUploadDocument = useCallback(async () => {
    try {
      setUploadingDoc(true);
      const picked = await DocumentPicker.getDocumentAsync({
        multiple: false,
        copyToCacheDirectory: true,
      });

      if (picked.canceled || !picked.assets?.length) {
        setUploadingDoc(false);
        return;
      }

      const file = picked.assets[0];
      const formData = new FormData();
      formData.append("file", {
        uri: file.uri,
        name: file.name,
        type: file.mimeType || "application/octet-stream",
      } as unknown as Blob);

      await api.uploadDocument(formData);
      await loadTabData();
      setUploadingDoc(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setUploadingDoc(false);
    }
  }, [api, loadTabData]);

  const renderMainPanel = () => {
    if (activeTab === "chat") {
      return (
        <>
          <FlatList
            data={messages}
            keyExtractor={(m) => m.id}
            contentContainerStyle={styles.listContent}
            renderItem={({ item }) => <MessageBubble role={item.role} content={item.content} />}
          />
          <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined}>
            <View style={styles.composer}>
              <TextInput
                value={input}
                onChangeText={setInput}
                placeholder="Ask Cortex..."
                placeholderTextColor="#7b90b8"
                style={styles.input}
                multiline
              />
              <Pressable style={[styles.sendButton, sending && styles.sendButtonDisabled]} disabled={sending} onPress={onSend}>
                <Text style={styles.sendText}>{sending ? "..." : "Send"}</Text>
              </Pressable>
            </View>
          </KeyboardAvoidingView>
        </>
      );
    }

    if (activeTab === "history") {
      return (
        <ScrollView contentContainerStyle={styles.panelContent}>
          <Pressable style={styles.smallButton} onPress={onNewConversation}>
            <Text style={styles.smallButtonText}>New Conversation</Text>
          </Pressable>
          {conversationHistory.map((c) => (
            <View key={c.id} style={styles.card}>
              <Text style={styles.cardTitle}>{c.title}</Text>
              <Text style={styles.cardMeta}>{new Date(c.timestamp).toLocaleString()}</Text>
              <Pressable style={styles.smallButton} onPress={() => void onSwitchConversation(c.id)}>
                <Text style={styles.smallButtonText}>Open</Text>
              </Pressable>
            </View>
          ))}
        </ScrollView>
      );
    }

    if (activeTab === "memories") {
      return (
        <ScrollView contentContainerStyle={styles.panelContent}>
          <View style={styles.rowInputWrap}>
            <TextInput
              value={memoryQuery}
              onChangeText={setMemoryQuery}
              placeholder="Search memories"
              placeholderTextColor="#7b90b8"
              style={styles.rowInput}
            />
            <Pressable style={styles.smallButton} onPress={() => void loadTabData()}>
              <Text style={styles.smallButtonText}>Search</Text>
            </Pressable>
          </View>
          {memories.map((m) => (
            <View key={m.id} style={styles.card}>
              <Text style={styles.cardTitle}>{m.memory_type}</Text>
              <Text style={styles.cardBody}>{m.content}</Text>
              <Text style={styles.cardMeta}>{new Date(m.timestamp).toLocaleString()}</Text>
            </View>
          ))}
        </ScrollView>
      );
    }

    if (activeTab === "documents") {
      return (
        <ScrollView contentContainerStyle={styles.panelContent}>
          <Pressable
            style={[styles.smallButton, uploadingDoc && styles.sendButtonDisabled]}
            onPress={() => void onPickAndUploadDocument()}
            disabled={uploadingDoc}
          >
            <Text style={styles.smallButtonText}>{uploadingDoc ? "Uploading..." : "Upload Document"}</Text>
          </Pressable>
          {documents.map((d) => (
            <View key={d.doc_id} style={styles.card}>
              <Text style={styles.cardTitle}>{d.filename}</Text>
              <Text style={styles.cardBody}>Status: {d.status}</Text>
              <Text style={styles.cardMeta}>{new Date(d.uploaded_at).toLocaleString()}</Text>
            </View>
          ))}
        </ScrollView>
      );
    }

    if (activeTab === "stats") {
      return (
        <View style={styles.flex}>
          <ScrollView horizontal contentContainerStyle={styles.statsHeaderContent} style={styles.statsHeader}>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>Memories</Text>
              <Text style={styles.statValue}>{stats?.memories.memories ?? 0}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>Entities</Text>
              <Text style={styles.statValue}>{stats?.memories.entities ?? 0}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>Edges</Text>
              <Text style={styles.statValue}>{stats?.memories.edges ?? 0}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>Vectors</Text>
              <Text style={styles.statValue}>{stats?.vectors.total_vectors ?? 0}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>Dim</Text>
              <Text style={styles.statValue}>{stats?.vectors.dimension ?? 0}</Text>
            </View>
          </ScrollView>
          <View style={styles.flex}>
            <ScrollView contentContainerStyle={styles.panelContent}>
              <Text style={styles.cardTitle}>Pipeline Traces</Text>
              <Text style={styles.cardBody}>Traces will appear here as queries are processed...</Text>
            </ScrollView>
          </View>
        </View>
      );
    }

    if (activeTab === "settings") {
      return (
        <ScrollView contentContainerStyle={styles.panelContent}>
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Model Provider</Text>
            <View style={styles.buttonRow}>
              <Pressable
                style={[styles.smallButton, settings.llmProvider === "local" && styles.tabChipActive]}
                onPress={() => setSettings((prev) => ({ ...prev, llmProvider: "local" }))}
              >
                <Text style={styles.smallButtonText}>Local</Text>
              </Pressable>
              <Pressable
                style={[styles.smallButton, settings.llmProvider === "gemini" && styles.tabChipActive]}
                onPress={() => setSettings((prev) => ({ ...prev, llmProvider: "gemini" }))}
              >
                <Text style={styles.smallButtonText}>Gemini</Text>
              </Pressable>
            </View>
          </View>
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Temperature</Text>
            <TextInput
              style={styles.rowInput}
              keyboardType="decimal-pad"
              value={settings.temperature.toString()}
              onChangeText={(v) => setSettings((prev) => ({ ...prev, temperature: Number(v) || 0.6 }))}
            />
          </View>
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Top P</Text>
            <TextInput
              style={styles.rowInput}
              keyboardType="decimal-pad"
              value={settings.topP.toString()}
              onChangeText={(v) => setSettings((prev) => ({ ...prev, topP: Number(v) || 0.95 }))}
            />
          </View>
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Max Tokens</Text>
            <TextInput
              style={styles.rowInput}
              keyboardType="number-pad"
              value={settings.maxTokens.toString()}
              onChangeText={(v) => setSettings((prev) => ({ ...prev, maxTokens: Number(v) || 4096 }))}
            />
          </View>
        </ScrollView>
      );
    }

    return (
      <ScrollView contentContainerStyle={styles.panelContent}>
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Ambient Status</Text>
          <Text style={styles.cardBody}>{ambientStatus}</Text>
          <View style={styles.buttonRow}>
            <Pressable style={styles.smallButton} onPress={() => void api.ambientAction("start").then(loadTabData)}>
              <Text style={styles.smallButtonText}>Start</Text>
            </Pressable>
            <Pressable style={styles.smallButton} onPress={() => void api.ambientAction("pause").then(loadTabData)}>
              <Text style={styles.smallButtonText}>Pause</Text>
            </Pressable>
            <Pressable style={styles.smallButton} onPress={() => void api.ambientAction("resume").then(loadTabData)}>
              <Text style={styles.smallButtonText}>Resume</Text>
            </Pressable>
            <Pressable style={styles.smallButtonDanger} onPress={() => void api.ambientAction("stop").then(loadTabData)}>
              <Text style={styles.smallButtonText}>Stop</Text>
            </Pressable>
          </View>
        </View>
      </ScrollView>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Cortex Mobile</Text>
        <Text style={styles.subtitle}>API: {apiBase}</Text>
        <View style={styles.statusRow}>
          <View style={[styles.statusDot, modelStatus.startsWith("online") ? styles.dotOnline : styles.dotOffline]} />
          <Text style={styles.statusText}>Model: {modelStatus}</Text>
          <Pressable style={styles.refreshButton} onPress={() => void refreshModelStatus()}>
            <Text style={styles.refreshText}>Refresh</Text>
          </Pressable>
        </View>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.tabRow}>
          {[
            ["chat", "Chat"],
            ["history", "History"],
            ["memories", "Memories"],
            ["documents", "Docs"],
            ["stats", "RAG Stats"],
            ["settings", "Settings"],
            ["ambient", "Ambient"],
          ].map(([value, label]) => {
            const selected = activeTab === value;
            return (
              <Pressable
                key={value}
                style={[styles.tabChip, selected && styles.tabChipActive]}
                onPress={() => setActiveTab(value as AppTab)}
              >
                <Text style={[styles.tabText, selected && styles.tabTextActive]}>{label}</Text>
              </Pressable>
            );
          })}
        </ScrollView>
      </View>

      {error ? (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}

      <View style={styles.flex}>{renderMainPanel()}</View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#071122",
  },
  flex: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#223452",
  },
  title: {
    color: "#f4f8ff",
    fontSize: 24,
    fontWeight: "700",
  },
  subtitle: {
    color: "#a8bade",
    marginTop: 2,
  },
  statusRow: {
    marginTop: 10,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 99,
  },
  dotOnline: {
    backgroundColor: "#22c55e",
  },
  dotOffline: {
    backgroundColor: "#ef4444",
  },
  statusText: {
    color: "#d3e0ff",
    fontSize: 13,
    flex: 1,
  },
  refreshButton: {
    backgroundColor: "#1f3760",
    borderWidth: 1,
    borderColor: "#335384",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  refreshText: {
    color: "#dbe8ff",
    fontWeight: "600",
    fontSize: 12,
  },
  tabRow: {
    marginTop: 10,
    gap: 8,
    paddingBottom: 2,
  },
  tabChip: {
    borderWidth: 1,
    borderColor: "#324a70",
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "#102036",
  },
  tabChipActive: {
    backgroundColor: "#244880",
    borderColor: "#3d68a4",
  },
  tabText: {
    color: "#a9bde3",
    fontSize: 12,
    fontWeight: "600",
  },
  tabTextActive: {
    color: "#f3f7ff",
  },
  listContent: {
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  panelContent: {
    paddingHorizontal: 14,
    paddingVertical: 14,
    gap: 10,
  },
  card: {
    borderWidth: 1,
    borderColor: "#2f4568",
    backgroundColor: "#0f1f38",
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
  },
  cardTitle: {
    color: "#f4f8ff",
    fontWeight: "700",
    marginBottom: 6,
    fontSize: 14,
  },
  cardBody: {
    color: "#c8d8fb",
    fontSize: 13,
    lineHeight: 19,
  },
  cardMeta: {
    marginTop: 8,
    color: "#8ea5cf",
    fontSize: 12,
  },
  rowInputWrap: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 10,
  },
  rowInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#2f466d",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    color: "#f4f8ff",
    backgroundColor: "#0f1f3a",
  },
  buttonRow: {
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
    marginTop: 8,
  },
  smallButton: {
    borderWidth: 1,
    borderColor: "#3a5f98",
    backgroundColor: "#244f8f",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    alignSelf: "flex-start",
  },
  smallButtonDanger: {
    borderWidth: 1,
    borderColor: "#91465d",
    backgroundColor: "#6e2f42",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
    alignSelf: "flex-start",
  },
  smallButtonText: {
    color: "#edf3ff",
    fontSize: 12,
    fontWeight: "600",
  },
  errorBox: {
    marginHorizontal: 14,
    marginBottom: 10,
    backgroundColor: "#5a1f2f",
    borderWidth: 1,
    borderColor: "#8d2d45",
    borderRadius: 10,
    padding: 10,
  },
  errorText: {
    color: "#ffd9e1",
    fontSize: 13,
  },
  composer: {
    borderTopWidth: 1,
    borderTopColor: "#223452",
    padding: 12,
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
  },
  input: {
    flex: 1,
    minHeight: 44,
    maxHeight: 130,
    color: "#f4f8ff",
    borderWidth: 1,
    borderColor: "#2f466d",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: "#0f1f3a",
  },
  sendButton: {
    minHeight: 44,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "#2f63c6",
    alignItems: "center",
    justifyContent: "center",
  },
  sendButtonDisabled: {
    backgroundColor: "#38507f",
  },
  sendText: {
    color: "#ffffff",
    fontWeight: "700",
    fontSize: 14,
  },
  statsHeader: {
    maxHeight: 80,
    backgroundColor: "#0f1f38",
    borderBottomWidth: 1,
    borderBottomColor: "#223452",
  },
  statsHeaderContent: {
    paddingHorizontal: 10,
    paddingVertical: 10,
    gap: 8,
  },
  statCard: {
    minWidth: 92,
    borderWidth: 1,
    borderColor: "#2f4568",
    backgroundColor: "#102a4b",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  statLabel: {
    color: "#8ea5cf",
    fontSize: 11,
  },
  statValue: {
    color: "#f4f8ff",
    fontWeight: "700",
    fontSize: 16,
    marginTop: 3,
  },
});
