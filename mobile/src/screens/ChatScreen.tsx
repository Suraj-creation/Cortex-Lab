/**
 * ChatScreen — mobile chat surface with keyboard-safe composer.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  FlatList,
  Text,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Keyboard,
  TextInput as RNTextInput,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
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
  globalError,
  onSend,
  onToggleProvider,
  onToggleRAG,
  onToggleStream,
  providerBusy,
}: ChatScreenProps) {
  const flatRef = useRef<FlatList>(null);
  const inputRef = useRef<RNTextInput>(null);
  const insets = useSafeAreaInsets();
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const composerLift = Platform.OS === 'android' ? Math.max(0, keyboardHeight - insets.bottom) : 0;
  const listBottomPadding = 152 + (composerLift > 0 ? composerLift : insets.bottom);

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => flatRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }, [messages.length]);

  useEffect(() => {
    const showEvent = Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow';
    const hideEvent = Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide';

    const showSub = Keyboard.addListener(showEvent, (event) => {
      setKeyboardHeight(event.endCoordinates?.height || 0);
    });
    const hideSub = Keyboard.addListener(hideEvent, () => {
      setKeyboardHeight(0);
    });

    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  const providerLabel = providerBusy
    ? 'Switching...'
    : settings.llmProvider === 'gemini'
      ? 'Gemini'
      : 'Local';

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? insets.top + 6 : 0}
    >
      <View style={styles.topBar}>
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
              size={13}
              color={settings.llmProvider === 'gemini' ? '#2b6cf6' : '#6f51f2'}
            />
            <Text
              style={[
                styles.pillText,
                { color: settings.llmProvider === 'gemini' ? '#2357c8' : '#5f46d3' },
              ]}
            >
              {providerLabel}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onToggleRAG}
            style={[styles.pill, settings.useRAG ? styles.pillGreen : styles.pillNeutral]}
          >
            <View style={[styles.pillDot, { backgroundColor: settings.useRAG ? '#10b981' : '#94a3b8' }]} />
            <Text
              style={[
                styles.pillText,
                { color: settings.useRAG ? '#0a7e58' : '#64748b' },
              ]}
            >
              {settings.useRAG ? 'RAG On' : 'RAG Off'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={onToggleStream}
            style={[styles.pill, settings.stream ? styles.pillAmber : styles.pillNeutral]}
          >
            <Text
              style={[
                styles.pillText,
                { color: settings.stream ? '#b26a00' : '#64748b' },
              ]}
            >
              {settings.stream ? 'Live Stream' : 'Batch'}
            </Text>
          </TouchableOpacity>
        </ScrollView>

        {messages.length <= 1 ? (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.quickRow}
          >
            {QUICK_PROMPTS.map((prompt) => (
              <TouchableOpacity key={prompt} onPress={() => setInput(prompt)} style={styles.quickChip}>
                <AppIcon name="lightning-bolt-outline" size={12} color="#5164e8" />
                <Text style={styles.quickText} numberOfLines={1}>{prompt}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        ) : null}
      </View>

      <FlatList
        ref={flatRef}
        data={messages}
        keyExtractor={(m) => m.id}
        contentContainerStyle={[styles.msgList, { paddingBottom: listBottomPadding }]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        ListFooterComponent={<View style={styles.listFooter} />}
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
              <AppIcon name="chat-processing-outline" size={36} color="#9aa7ff" />
            </View>
            <Text style={styles.emptyTitle}>Start a conversation</Text>
            <Text style={styles.emptyBody}>
              Ask Cortex Lab about your memories, knowledge graph, documents, or active projects.
            </Text>
          </View>
        }
      />

      {globalError ? (
        <View style={styles.errorBanner}>
          <AppIcon name="alert-circle-outline" size={14} color="#e11d48" />
          <Text style={styles.errorText} numberOfLines={2}>{globalError}</Text>
        </View>
      ) : null}

      <View
        style={[
          styles.composerShell,
          { paddingBottom: Math.max(insets.bottom, SPACING.sm) },
          composerLift ? { marginBottom: composerLift } : null,
        ]}
      >
        <View style={styles.inputBar}>
          <View style={styles.inputWrap}>
            <AppIcon name="microphone-outline" size={16} color="#8a94ab" style={styles.leadingIcon} />
            <RNTextInput
              ref={inputRef}
              style={styles.input}
              placeholder="Message Cortex Lab..."
              placeholderTextColor="#94a3b8"
              value={input}
              onChangeText={setInput}
              multiline
              maxLength={4000}
              editable={!sending}
              selectionColor="#6366f1"
            />
            <AppIcon name="paperclip" size={16} color="#8a94ab" style={styles.leadingIcon} />
          </View>

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
                colors={['#6c7dff', '#5262df']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.sendBtnGradient}
              >
                <AppIcon name="arrow-up" size={18} color="#ffffff" />
              </LinearGradient>
            )}
          </TouchableOpacity>
        </View>

        <View style={styles.metaRow}>
          <Text style={styles.metaText}>
            {`${settings.llmProvider.toUpperCase()} | ${settings.useRAG ? 'RAG' : 'No RAG'} | ${settings.stream ? 'Stream' : 'Batch'} | T ${settings.temperature} | P ${settings.topP}`}
          </Text>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#e9eef8',
  },
  topBar: {
    backgroundColor: '#e9eef8',
    paddingTop: SPACING.md,
    paddingBottom: SPACING.sm,
    gap: SPACING.sm,
  },
  pillRow: {
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
    paddingBottom: SPACING.xs,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACING.md,
    paddingVertical: 8,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    gap: 5,
    backgroundColor: '#edf2fb',
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
  pillDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  pillBlue: { backgroundColor: '#edf5ff', borderColor: '#ffffff' },
  pillViolet: { backgroundColor: '#f1eeff', borderColor: '#ffffff' },
  pillGreen: { backgroundColor: '#eefbf5', borderColor: '#ffffff' },
  pillAmber: { backgroundColor: '#fff8ea', borderColor: '#ffffff' },
  pillNeutral: { backgroundColor: '#edf2fb', borderColor: '#ffffff' },
  pillText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
  },
  quickRow: {
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  quickChip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#edf2fb',
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: '#ffffff',
    paddingHorizontal: SPACING.md,
    paddingVertical: 8,
    maxWidth: 250,
    gap: 4,
    ...SHADOWS.md,
  },
  quickText: {
    fontSize: FONT_SIZE.xs,
    color: '#4557d6',
    fontWeight: FONT_WEIGHT.medium,
  },
  msgList: {
    paddingTop: SPACING.sm,
  },
  listFooter: {
    height: SPACING.md,
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
    backgroundColor: '#eef3fb',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.lg,
    ...SHADOWS.md,
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
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.sm,
    backgroundColor: '#fff4f5',
    borderWidth: 1,
    borderColor: '#ffffff',
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    gap: SPACING.sm,
    ...SHADOWS.md,
  },
  errorText: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    color: '#e11d48',
  },
  composerShell: {
    backgroundColor: '#e9eef8',
  },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.sm,
    backgroundColor: '#e9eef8',
  },
  inputWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#edf2fb',
    borderRadius: RADIUS['2xl'],
    borderWidth: 1,
    borderColor: '#ffffff',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    minHeight: 48,
    ...SHADOWS.md,
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
  leadingIcon: {
    paddingBottom: 2,
  },
  sendBtn: {
    width: 46,
    height: 46,
    borderRadius: 23,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  sendBtnDisabled: { opacity: 0.4 },
  sendBtnLoading: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: '#edf2fb',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
  sendBtnGradient: {
    width: 46,
    height: 46,
    borderRadius: 23,
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.glow,
  },
  metaRow: {
    backgroundColor: '#e9eef8',
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.sm,
  },
  metaText: {
    fontSize: 10,
    color: '#8390aa',
    textAlign: 'right',
    letterSpacing: 0.3,
  },
});
