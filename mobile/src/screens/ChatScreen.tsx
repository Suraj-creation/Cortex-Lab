/**
 * ChatScreen — Neural Dark AI Chat
 * Stitch ref: a7edacc10a204e7dae4dcd52d1057123
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
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
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
            style={[styles.pill, settings.llmProvider === 'gemini' ? styles.pillBlue : styles.pillViolet]}
          >
            <Text style={styles.pillText}>{providerLabel}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onToggleRAG}
            style={[styles.pill, settings.useRAG ? styles.pillGreen : styles.pillNeutral]}
          >
            <Text style={styles.pillText}>{settings.useRAG ? 'RAG On' : 'RAG Off'}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onToggleStream}
            style={[styles.pill, settings.stream ? styles.pillAmber : styles.pillNeutral]}
          >
            <Text style={styles.pillText}>{settings.stream ? 'Stream' : 'Batch'}</Text>
          </TouchableOpacity>
        </ScrollView>

        {/* Quick prompts */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.quickRow}
        >
          {QUICK_PROMPTS.map((p) => (
            <TouchableOpacity key={p} onPress={() => setInput(p.replace(/^[^\s]+\s/, ''))} style={styles.quickChip}>
              <Text style={styles.quickText} numberOfLines={1}>{p}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
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
            <Text style={styles.emptyTitle}>Start a conversation</Text>
            <Text style={styles.emptyBody}>Ask Cortex Lab anything…</Text>
          </View>
        }
      />

      {/* ── Error banner ─ */}
      {globalError ? (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{globalError}</Text>
        </View>
      ) : null}

      {/* ── Input area ─ */}
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={16}>
        <View style={styles.inputBar}>
          <View style={styles.inputWrap}>
            {/* Mic icon */}
            <View style={styles.inputIcon}>
              <AppIcon name="microphone-outline" size={16} color={NEURAL.onSurfaceVariant} />
            </View>

            <RNTextInput
              ref={inputRef}
              style={styles.input}
              placeholder="Message Cortex Lab…"
              placeholderTextColor={NEURAL.outline}
              value={input}
              onChangeText={setInput}
              multiline
              maxLength={4000}
              editable={!sending}
              selectionColor={NEURAL.primary}
            />

            {/* Attach icon */}
            <View style={styles.inputIcon}>
              <AppIcon name="paperclip" size={16} color={NEURAL.onSurfaceVariant} />
            </View>
          </View>

          {/* Send button */}
          <TouchableOpacity
            onPress={onSend}
            disabled={!input.trim() || sending}
            style={[styles.sendBtn, (!input.trim() || sending) && styles.sendBtnDisabled]}
          >
            {sending && streamingMessageId ? (
              <NeuralPulse active size={8} color="#ffffff" />
            ) : (
              <LinearGradient
                colors={[NEURAL.primary, NEURAL.primaryDim]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.sendBtnGradient}
              >
                <AppIcon name="arrow-up" size={16} color="#ffffff" style={styles.sendIcon} />
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
  container: { flex: 1, backgroundColor: NEURAL.background },

  // Top control bar
  topBar: {
    backgroundColor: NEURAL.surfaceContainerLow,
    paddingTop: SPACING.sm,
    paddingBottom: SPACING.sm,
  },
  pillRow: {
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
    paddingBottom: SPACING.sm,
  },
  pill: {
    paddingHorizontal: SPACING.md,
    paddingVertical: 5,
    borderRadius: RADIUS.full,
    borderWidth: 1,
  },
  pillBlue:    { backgroundColor: `${NEURAL.primary}22`,    borderColor: `${NEURAL.primary}60` },
  pillViolet:  { backgroundColor: `${NEURAL.secondary}22`,  borderColor: `${NEURAL.secondary}60` },
  pillGreen:   { backgroundColor: `${NEURAL.tertiary}22`,   borderColor: `${NEURAL.tertiary}60` },
  pillAmber:   { backgroundColor: '#f59e0b22',               borderColor: '#f59e0b60' },
  pillNeutral: { backgroundColor: NEURAL.surfaceContainerHigh, borderColor: NEURAL.outlineVariant },
  pillText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurfaceVariant,
  },

  // Quick prompts
  quickRow: {
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  quickChip: {
    backgroundColor: NEURAL.surfaceContainerHighest,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    paddingHorizontal: SPACING.md,
    paddingVertical: 5,
    maxWidth: 220,
  },
  quickText: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.onSurfaceVariant,
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
    paddingHorizontal: SPACING.lg,
  },
  emptyTitle: {
    fontSize: FONT_SIZE.xl,
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurface,
    marginBottom: SPACING.sm,
  },
  emptyBody: {
    fontSize: FONT_SIZE.base,
    color: NEURAL.onSurfaceVariant,
    textAlign: 'center',
  },

  // Error
  errorBanner: {
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.sm,
    backgroundColor: `${NEURAL.error}22`,
    borderWidth: 1,
    borderColor: `${NEURAL.error}60`,
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
  },
  errorText: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.error,
  },

  // Input
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.md,
    backgroundColor: NEURAL.surfaceContainerLow,
  },
  inputWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: NEURAL.surfaceContainerHighest,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    minHeight: 46,
  },
  inputIcon: { paddingBottom: 2 },
  input: {
    flex: 1,
    fontSize: FONT_SIZE.base,
    color: NEURAL.onSurface,
    maxHeight: 120,
    padding: 0,
    margin: 0,
    paddingVertical: 2,
  },
  sendBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  sendBtnDisabled: { opacity: 0.45 },
  sendBtnGradient: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendIcon: { marginBottom: 1 },

  // Meta row
  metaRow: {
    backgroundColor: NEURAL.surfaceContainerLow,
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.sm,
  },
  metaText: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.outline,
    textAlign: 'right',
  },
});
