/**
 * ChatScreen — Cortex Aurora AI Chat
 * Features: Provider pills, quick prompts, message list, streaming input bar
 */
import React, { useCallback, useRef, useEffect } from 'react';
import {
  View,
  FlatList,
  Text,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TextInput as RNTextInput,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { MessageBubble } from '../components/MessageBubble';
import { NeuralPulse } from '../components/ui/NeuralPulse';
import { AppIcon } from '../components/ui/AppIcon';
import type { ChatMessage, ChatSettings, ModelStatus } from '../../shared/core/types';

const QUICK_PROMPTS = [
  'Summarize my latest project milestones',
  'What has changed in my beliefs this month?',
  'Show top memory themes from conversations',
  'What should I focus on next week?',
];

interface ChatScreenProps {
  messages: ChatMessage[];
  input: string;
  setInput: (v: string) => void;
  sending: boolean;
  streamingMessageId: string | null;
  settings: ChatSettings;
  modelStatus: ModelStatus;
  globalError: string;
  onSend: () => void;
  onToggleProvider: () => void;
  onToggleRAG: () => void;
  onToggleStream: () => void;
  providerBusy: boolean;
  localModelAvailable: boolean;
}

export function ChatScreen({
  messages,
  input,
  setInput,
  sending,
  streamingMessageId,
  settings,
  modelStatus,
  globalError,
  onSend,
  onToggleProvider,
  onToggleRAG,
  onToggleStream,
  providerBusy,
  localModelAvailable,
}: ChatScreenProps) {
  const flatRef = useRef<FlatList>(null);
  const inputRef = useRef<RNTextInput>(null);

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => flatRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }, [messages.length]);

  const providerLabel = providerBusy
    ? 'Switching…'
    : settings.llmProvider === 'gemini'
    ? 'Gemini'
    : 'Local';

  return (
    <View style={styles.container}>
      {/* ── Control bar ─ */}
      <View style={styles.topBar}>
        {/* Provider + RAG + Stream pills */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.pillRow}
        >
          <TouchableOpacity
            onPress={onToggleProvider}
            disabled={providerBusy}
            style={[
              styles.pill,
              settings.llmProvider === 'gemini' ? styles.pillBlue : styles.pillViolet,
            ]}
          >
            <AppIcon
              name={settings.llmProvider === 'gemini' ? 'cloud-outline' : 'chip'}
              size={12}
              color={settings.llmProvider === 'gemini' ? '#3b82f6' : '#8b5cf6'}
            />
            <Text style={[
              styles.pillText,
              { color: settings.llmProvider === 'gemini' ? '#2563eb' : '#7c3aed' },
            ]}>
              {providerLabel}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onToggleRAG}
            style={[styles.pill, settings.useRAG ? styles.pillGreen : styles.pillNeutral]}
          >
            <View style={[styles.pillDot, { backgroundColor: settings.useRAG ? '#10b981' : '#94a3b8' }]} />
            <Text style={[
              styles.pillText,
              { color: settings.useRAG ? '#065f46' : '#64748b' },
            ]}>
              {settings.useRAG ? 'RAG On' : 'RAG Off'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onToggleStream}
            style={[styles.pill, settings.stream ? styles.pillAmber : styles.pillNeutral]}
          >
            <Text style={[
              styles.pillText,
              { color: settings.stream ? '#92400e' : '#64748b' },
            ]}>
              {settings.stream ? '⚡ Stream' : 'Batch'}
            </Text>
          </TouchableOpacity>
        </ScrollView>

        {/* Quick prompts */}
        {messages.length <= 1 && (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.quickRow}
          >
            {QUICK_PROMPTS.map((p) => (
              <TouchableOpacity key={p} onPress={() => setInput(p)} style={styles.quickChip}>
                <AppIcon name="lightning-bolt-outline" size={12} color="#6366f1" />
                <Text style={styles.quickText} numberOfLines={1}>{p}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )}
      </View>

      {/* ── Messages ─ */}
      <FlatList
        ref={flatRef}
        data={messages}
        keyExtractor={(m) => m.id}
        contentContainerStyle={styles.msgList}
        showsVerticalScrollIndicator={false}
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
          <View style={styles.empty}>
            <View style={styles.emptyIconContainer}>
              <AppIcon name="chat-processing-outline" size={36} color="#a5b4fc" />
            </View>
            <Text style={styles.emptyTitle}>Start a conversation</Text>
            <Text style={styles.emptyBody}>Ask Cortex Lab anything about your memories, knowledge, and experiences…</Text>
          </View>
        }
      />

      {/* ── Error banner ─ */}
      {globalError ? (
        <View style={styles.errorBanner}>
          <AppIcon name="alert-circle-outline" size={14} color="#e11d48" />
          <Text style={styles.errorText} numberOfLines={2}>{globalError}</Text>
        </View>
      ) : null}

      {/* ── Input area ─ */}
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={16}>
        <View style={styles.inputBar}>
          <View style={styles.inputWrap}>
            <AppIcon name="microphone-outline" size={16} color="#94a3b8" style={{ paddingBottom: 2 }} />
            <RNTextInput
              ref={inputRef}
              style={styles.input}
              placeholder="Message Cortex Lab…"
              placeholderTextColor="#94a3b8"
              value={input}
              onChangeText={setInput}
              multiline
              maxLength={4000}
              editable={!sending}
              selectionColor="#6366f1"
            />
            <AppIcon name="paperclip" size={16} color="#94a3b8" style={{ paddingBottom: 2 }} />
          </View>

          {/* Send button */}
          <TouchableOpacity
            onPress={onSend}
            disabled={!input.trim() || sending}
            style={[styles.sendBtn, (!input.trim() || sending) && styles.sendBtnDisabled]}
          >
            {sending && streamingMessageId ? (
              <View style={styles.sendBtnLoading}>
                <NeuralPulse active size={8} color="#6366f1" />
              </View>
            ) : (
              <LinearGradient
                colors={['#6366f1', '#4f46e5']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.sendBtnGradient}
              >
                <AppIcon name="arrow-up" size={18} color="#ffffff" />
              </LinearGradient>
            )}
          </TouchableOpacity>
        </View>

        {/* Meta row */}
        <View style={styles.metaRow}>
          <Text style={styles.metaText}>
            {settings.llmProvider.toUpperCase()} · {settings.useRAG ? 'RAG' : 'No RAG'} · {settings.stream ? 'Stream' : 'Batch'} · T {settings.temperature} · P {settings.topP}
          </Text>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },

  // Top control bar
  topBar: {
    backgroundColor: '#ffffff',
    paddingTop: SPACING.sm,
    paddingBottom: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  pillRow: {
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
    paddingBottom: SPACING.sm,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACING.md,
    paddingVertical: 5,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    gap: 4,
  },
  pillDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  pillBlue:    { backgroundColor: '#eff6ff',   borderColor: '#bfdbfe' },
  pillViolet:  { backgroundColor: '#f5f3ff',   borderColor: '#ddd6fe' },
  pillGreen:   { backgroundColor: '#f0fdf4',   borderColor: '#bbf7d0' },
  pillAmber:   { backgroundColor: '#fffbeb',   borderColor: '#fde68a' },
  pillNeutral: { backgroundColor: '#f1f5f9',   borderColor: '#e2e8f0' },
  pillText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#475569',
  },

  // Quick prompts
  quickRow: {
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  quickChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#eef2ff',
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: '#c7d2fe',
    paddingHorizontal: SPACING.md,
    paddingVertical: 6,
    maxWidth: 240,
    gap: 4,
  },
  quickText: {
    fontSize: FONT_SIZE.xs,
    color: '#4338ca',
    fontWeight: FONT_WEIGHT.medium,
  },

  // Messages
  msgList: {
    paddingVertical: SPACING.md,
    paddingBottom: SPACING['4xl'],
  },
  empty: {
    alignItems: 'center',
    paddingVertical: SPACING['5xl'],
    paddingHorizontal: SPACING['2xl'],
  },
  emptyIconContainer: {
    width: 72,
    height: 72,
    borderRadius: RADIUS['2xl'],
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.lg,
  },
  emptyTitle: {
    fontSize: FONT_SIZE.xl,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
    marginBottom: SPACING.sm,
  },
  emptyBody: {
    fontSize: FONT_SIZE.base,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 20,
  },

  // Error
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.sm,
    backgroundColor: '#fff1f2',
    borderWidth: 1,
    borderColor: '#fecdd3',
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    gap: SPACING.sm,
  },
  errorText: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    color: '#e11d48',
  },

  // Input
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.sm,
    backgroundColor: '#ffffff',
    borderTopWidth: 1,
    borderTopColor: '#f1f5f9',
  },
  inputWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS['2xl'],
    borderWidth: 1,
    borderColor: '#e2e8f0',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    minHeight: 46,
  },
  input: {
    flex: 1,
    fontSize: FONT_SIZE.base,
    color: '#0f172a',
    maxHeight: 120,
    padding: 0,
    margin: 0,
    paddingVertical: 2,
  },
  sendBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  sendBtnDisabled: { opacity: 0.4 },
  sendBtnLoading: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnGradient: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.glow,
  },

  // Meta row
  metaRow: {
    backgroundColor: '#ffffff',
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.sm,
  },
  metaText: {
    fontSize: 10,
    color: '#94a3b8',
    textAlign: 'right',
    letterSpacing: 0.3,
  },
});
