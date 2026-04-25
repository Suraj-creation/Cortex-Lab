/**
 * AmbientVoiceScreen — mobile client companion for continuous session capture.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  FlatList,
} from 'react-native';
import {
  createAudioPlayer,
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';
import { File, Paths } from 'expo-file-system';

import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Button } from '../components/ui/Button';
import { EmptyState } from '../components/ui/EmptyState';
import { TextInput } from '../components/ui/TextInput';
import type {
  AmbientClientAudioResponse,
  AmbientConfig,
  AmbientLiveStatus,
  AmbientRetentionTrace,
  AmbientState,
  ConversationRecord,
  ConversationTurn,
  VoiceProviders,
} from '../../shared/core/types';

type AmbientTab = 'live' | 'conversations' | 'settings';
type CompanionMode = 'idle' | 'requesting' | 'listening' | 'processing' | 'speaking' | 'error';

const MOBILE_CAPTURE_MS = 4200;

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function inferMimeType(uri: string): string {
  const normalized = uri.toLowerCase();
  if (normalized.endsWith('.wav')) return 'audio/wav';
  if (normalized.endsWith('.webm')) return 'audio/webm';
  if (normalized.endsWith('.3gp')) return 'audio/3gpp';
  return 'audio/mp4';
}

interface AmbientVoiceScreenProps {
  ambientState: AmbientState | null;
  ambientConfig: AmbientConfig | null;
  ambientLiveStatus: AmbientLiveStatus | null;
  voiceProviders: VoiceProviders | null;
  onStartListening: () => void;
  onStopListening: () => void;
  onPauseAmbient: () => void;
  onResumeAmbient: () => void;
  api: any;
}

export function AmbientVoiceScreen({
  ambientState,
  ambientConfig,
  ambientLiveStatus,
  voiceProviders,
  onStartListening: _onStartListening,
  onStopListening: _onStopListening,
  onPauseAmbient: _onPauseAmbient,
  onResumeAmbient: _onResumeAmbient,
  api,
}: AmbientVoiceScreenProps) {
  const [activeTab, setActiveTab] = useState<AmbientTab>('live');
  const [conversations, setConversations] = useState<ConversationRecord[]>([]);
  const [transcript, setTranscript] = useState<ConversationTurn[]>([]);
  const [companionMode, setCompanionMode] = useState<CompanionMode>('idle');
  const [companionError, setCompanionError] = useState<string>('');
  const [clientSessionId, setClientSessionId] = useState('');
  const [capturedChunks, setCapturedChunks] = useState(0);
  const [lastTranscript, setLastTranscript] = useState('');
  const [lastAssistantReply, setLastAssistantReply] = useState('');
  const [lastRetention, setLastRetention] = useState<AmbientRetentionTrace | null>(null);
  const [assistantNameDraft, setAssistantNameDraft] = useState('');
  const [assistantAliasesDraft, setAssistantAliasesDraft] = useState('');
  const [followupWindowDraft, setFollowupWindowDraft] = useState('45');
  const [savingCompanionSettings, setSavingCompanionSettings] = useState(false);
  const [companionSettingsMessage, setCompanionSettingsMessage] = useState('');

  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder, 250);
  const loopActiveRef = useRef(false);
  const loopPromiseRef = useRef<Promise<void> | null>(null);
  const playerRef = useRef<ReturnType<typeof createAudioPlayer> | null>(null);
  const sessionIdRef = useRef('');

  const assistantName = ambientConfig?.assistant_name?.trim() || 'Eva';
  const isClientActive =
    companionMode === 'requesting' ||
    companionMode === 'listening' ||
    companionMode === 'processing' ||
    companionMode === 'speaking';

  const loadConversations = useCallback(async () => {
    try {
      const res = await api.getConversations?.(20, 0);
      setConversations(res?.conversations || []);
    } catch {}
  }, [api]);

  const loadTranscript = useCallback(async () => {
    try {
      const res = await api.getLiveTranscript?.();
      setTranscript(res?.turns || []);
    } catch {}
  }, [api]);

  const tabs: { key: AmbientTab; label: string; icon: string }[] = [
    { key: 'live', label: 'Live', icon: 'microphone-outline' },
    { key: 'conversations', label: 'History', icon: 'history' },
    { key: 'settings', label: 'Settings', icon: 'cog-outline' },
  ];

  useEffect(() => {
    sessionIdRef.current = clientSessionId;
  }, [clientSessionId]);

  useEffect(() => {
    setAssistantNameDraft(ambientConfig?.assistant_name?.trim() || 'Eva');
    setAssistantAliasesDraft((ambientConfig?.assistant_aliases || ['eva', 'cortex']).join(', '));
    setFollowupWindowDraft(String(ambientConfig?.companion_followup_window_s || 45));
  }, [
    ambientConfig?.assistant_aliases,
    ambientConfig?.assistant_name,
    ambientConfig?.companion_followup_window_s,
  ]);

  useEffect(() => {
    return () => {
      loopActiveRef.current = false;
      playerRef.current?.remove();
      if (recorderState.isRecording) {
        void recorder.stop().catch(() => {});
      }
    };
  }, [recorder, recorderState.isRecording]);

  const refreshAmbientViews = useCallback(async () => {
    await Promise.allSettled([loadTranscript(), loadConversations()]);
  }, [loadConversations, loadTranscript]);

  const playAssistantAudio = useCallback(async (audioBase64: string) => {
    if (!audioBase64) return;

    try {
      const file = new File(Paths.cache, `ambient-reply-${Date.now()}.wav`);
      file.write(audioBase64, { encoding: 'base64' });

      playerRef.current?.remove();
      const player = createAudioPlayer({ uri: file.uri }, { updateInterval: 200 });
      playerRef.current = player;
      setCompanionMode('speaking');
      player.play();

      await new Promise<void>((resolve) => {
        const startedAt = Date.now();
        const interval = setInterval(() => {
          if (!player.playing && player.currentTime > 0) {
            clearInterval(interval);
            resolve();
            return;
          }
          if (Date.now() - startedAt > 20000) {
            clearInterval(interval);
            resolve();
          }
        }, 180);
      });

      player.remove();
      playerRef.current = null;
    } catch {}
  }, []);

  const processRecordedChunk = useCallback(async (recordingUri: string, durationMs: number) => {
    const activeSessionId = sessionIdRef.current;
    if (!activeSessionId || !recordingUri) return;

    const file = new File(recordingUri);
    const audioBase64 = await file.base64();
    if (!audioBase64) return;

    const response: AmbientClientAudioResponse = await api.processAmbientClientAudio({
      sessionId: activeSessionId,
      audioBase64,
      mimeType: inferMimeType(recordingUri),
      platform: 'mobile',
      estimatedDurationS: durationMs / 1000,
      metadata: {
        surface: 'ambient-screen',
      },
    });

    setCapturedChunks((count) => count + 1);
    setLastTranscript(response.transcript || '');
    setLastAssistantReply(response.assistant_text || '');
    setLastRetention(response.retention_trace || null);
    if (response.session_id && response.session_id !== activeSessionId) {
      setClientSessionId(response.session_id);
    }

    await refreshAmbientViews();

    if (response.assistant_audio_base64) {
      await playAssistantAudio(response.assistant_audio_base64);
    }
  }, [api, playAssistantAudio, refreshAmbientViews]);

  const recordingLoop = useCallback(async () => {
    while (loopActiveRef.current) {
      try {
        setCompanionMode('listening');
        await recorder.prepareToRecordAsync();
        recorder.record();

        const startedAt = Date.now();
        await wait(MOBILE_CAPTURE_MS);
        if (!recorderState.isRecording) {
          await wait(120);
        }
        if (recorder.getStatus().isRecording) {
          await recorder.stop();
        }

        const durationMs = Math.max(Date.now() - startedAt, 1);
        const recordedUri = recorder.uri || recorder.getStatus().url || '';
        if (recordedUri && loopActiveRef.current) {
          setCompanionMode('processing');
          await processRecordedChunk(recordedUri, durationMs);
        }
      } catch (error) {
        setCompanionMode('error');
        setCompanionError(error instanceof Error ? error.message : String(error));
        break;
      }
    }

    if (loopActiveRef.current) {
      setCompanionMode('listening');
    }
  }, [processRecordedChunk, recorder, recorderState.isRecording]);

  const startCompanion = useCallback(async () => {
    if (isClientActive) return;

    setCompanionError('');
    setCompanionMode('requesting');

    try {
      const permission = await requestRecordingPermissionsAsync();
      if (!permission.granted) {
        throw new Error('Microphone permission is required for continuous companion capture.');
      }

      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });

      const session = await api.startAmbientClientSession({
        platform: 'mobile',
        metadata: {
          surface: 'ambient-screen',
        },
      });

      setClientSessionId(session.session_id);
      sessionIdRef.current = session.session_id;
      loopActiveRef.current = true;
      loopPromiseRef.current = recordingLoop();
      await refreshAmbientViews();
    } catch (error) {
      setCompanionMode('error');
      setCompanionError(error instanceof Error ? error.message : String(error));
    }
  }, [api, isClientActive, recordingLoop, refreshAmbientViews]);

  const stopCompanion = useCallback(async () => {
    loopActiveRef.current = false;

    try {
      if (recorder.getStatus().isRecording) {
        await recorder.stop();
      }
      await loopPromiseRef.current;
      loopPromiseRef.current = null;

      const activeSessionId = sessionIdRef.current;
      if (activeSessionId) {
        await api.stopAmbientClientSession({
          sessionId: activeSessionId,
          reason: 'user_request',
        });
      }

      playerRef.current?.remove();
      playerRef.current = null;
      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
      setCompanionMode('idle');
      setClientSessionId('');
      sessionIdRef.current = '';
      await refreshAmbientViews();
    } catch (error) {
      setCompanionMode('error');
      setCompanionError(error instanceof Error ? error.message : String(error));
    }
  }, [api, recorder, refreshAmbientViews]);

  const saveCompanionSettings = useCallback(async () => {
    const followupWindow = Math.max(10, Math.min(120, parseInt(followupWindowDraft, 10) || 45));
    setSavingCompanionSettings(true);
    setCompanionSettingsMessage('');

    try {
      await api.updateAmbientConfig({
        assistant_name: assistantNameDraft.trim() || 'Eva',
        assistant_aliases: assistantAliasesDraft
          .split(',')
          .map((value: string) => value.trim().toLowerCase())
          .filter(Boolean),
        companion_followup_window_s: followupWindow,
      });
      setFollowupWindowDraft(String(followupWindow));
      setCompanionSettingsMessage('Companion settings saved.');
    } catch (error) {
      setCompanionSettingsMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setSavingCompanionSettings(false);
    }
  }, [api, assistantAliasesDraft, assistantNameDraft, followupWindowDraft]);

  const retainedAs =
    lastRetention?.memory_decision === 'priority'
      ? 'Priority memory'
      : lastRetention?.memory_decision === 'structured'
        ? 'Structured memory'
        : lastRetention?.memory_decision === 'session_only'
          ? 'Session context'
          : 'Awaiting analysis';

  const providerBadge =
    ambientConfig?.stt_provider || ambientState?.stt_provider || voiceProviders?.stt_provider || 'gemini';

  return (
    <View style={styles.container}>
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => {
              setActiveTab(tab.key);
              if (tab.key === 'conversations') void loadConversations();
              if (tab.key === 'live') void loadTranscript();
            }}
            activeOpacity={0.7}
          >
            <AppIcon name={tab.icon as any} size={16} color={activeTab === tab.key ? '#6366f1' : '#94a3b8'} />
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelActive]}>{tab.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {activeTab === 'live' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          <Card variant="accent" padding="lg">
            <View style={styles.heroHeader}>
              <View style={{ flex: 1 }}>
                <Text style={styles.heroEyebrow}>EVA COMPANION</Text>
                <Text style={styles.heroTitle}>Continuous mobile session capture</Text>
                <Text style={styles.heroBody}>
                  Say {assistantName} naturally. The app keeps a live session open, ships audio
                  chunks to Gemini STT, tags useful memories, and plays spoken answers back.
                </Text>
              </View>
              <Badge
                label={companionMode.toUpperCase()}
                variant={
                  companionMode === 'listening'
                    ? 'success'
                    : companionMode === 'processing'
                      ? 'info'
                      : companionMode === 'speaking'
                        ? 'warning'
                        : companionMode === 'error'
                          ? 'error'
                          : 'default'
                }
                size="sm"
                dot={companionMode === 'listening'}
              />
            </View>

            <View style={styles.controlButtons}>
              {!isClientActive ? (
                <Button
                  label="Start Companion"
                  onPress={startCompanion}
                  icon={<AppIcon name="microphone" size={16} color="#ffffff" />}
                  style={styles.controlBtn}
                />
              ) : (
                <Button
                  label="Stop Session"
                  variant="error"
                  onPress={stopCompanion}
                  icon={<AppIcon name="stop" size={16} color="#e11d48" />}
                  style={styles.controlBtn}
                />
              )}
            </View>

            {companionError ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{companionError}</Text>
              </View>
            ) : null}

            <View style={styles.liveStats}>
              <Text style={styles.liveStatText}>Session: {clientSessionId || ambientState?.client_session?.active_session_id || 'not started'}</Text>
              <Text style={styles.liveStatText}>Recorder: {recorderState.isRecording ? 'capturing' : 'waiting'}</Text>
              <Text style={styles.liveStatText}>Chunks: {capturedChunks}</Text>
            </View>
          </Card>

          <View style={styles.metricsGrid}>
            <MetricCard
              label="Voice Runtime"
              value={providerBadge}
              tone="indigo"
              compact
              style={styles.metricHalf}
            />
            <MetricCard
              label="Chunk Timer"
              value={`${Math.round((recorderState.durationMillis || 0) / 1000)}s`}
              tone="violet"
              compact
              style={styles.metricHalf}
            />
          </View>

          <View style={styles.metricsGrid}>
            <MetricCard
              label="Transcriptions"
              value={String(ambientState?.transcriptions ?? transcript.length)}
              tone="emerald"
              compact
              style={styles.metricHalf}
            />
            <MetricCard
              label="Memory Tags"
              value={String(lastRetention?.tags?.length ?? 0)}
              tone="blue"
              compact
              style={styles.metricHalf}
            />
          </View>

          <Card variant="default" padding="lg">
            <SectionHeader title="Latest Transcript" />
            <Text style={styles.panelText}>
              {lastTranscript || 'The next spoken turn will appear here once the first chunk is processed.'}
            </Text>
          </Card>

          <Card variant="default" padding="lg">
            <SectionHeader title="Latest Reply" />
            <Text style={styles.panelText}>
              {lastAssistantReply || `${assistantName} will answer here when the backend decides a spoken reply is needed.`}
            </Text>
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Retention Decision" />
            <View style={styles.retentionHeader}>
              <Badge label={retainedAs} variant="primary" size="sm" />
            </View>
            {lastRetention?.tags && lastRetention.tags.length > 0 ? (
              <View style={styles.tagWrap}>
                {lastRetention.tags.map((tag) => (
                  <Badge key={tag} label={tag.replace(/_/g, ' ')} variant="default" size="sm" />
                ))}
              </View>
            ) : (
              <Text style={styles.noData}>Tags will appear here after the backend evaluates a captured turn.</Text>
            )}
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Live Transcript" action={{ label: 'Refresh', onPress: loadTranscript }} />
            {transcript.length === 0 ? (
              <Text style={styles.noData}>No transcript segments available</Text>
            ) : (
              transcript.slice(-20).map((seg, index) => (
                <View key={`${seg.live_turn_id || index}-${seg.timestamp}`} style={styles.transcriptRow}>
                  <Badge label={seg.speaker_name || seg.speaker_label} variant="primary" size="sm" />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.transcriptText}>{seg.text}</Text>
                    {seg.retention_trace?.tags && seg.retention_trace.tags.length > 0 ? (
                      <View style={styles.tagWrap}>
                        {seg.retention_trace.tags.slice(0, 4).map((tag) => (
                          <Badge key={`${seg.live_turn_id || index}-${tag}`} label={tag.replace(/_/g, ' ')} variant="default" size="sm" />
                        ))}
                      </View>
                    ) : null}
                  </View>
                </View>
              ))
            )}
          </Card>
        </ScrollView>
      )}

      {activeTab === 'conversations' && (
        <FlatList
          data={conversations}
          keyExtractor={(c) => c.id}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={<EmptyState icon="history" title="No Conversations" message="Ambient conversations will appear here after listening sessions." />}
          renderItem={({ item }) => (
            <Card variant="default" padding="md">
              <Text style={styles.convTitle}>Conversation {item.id.slice(0, 8)}</Text>
              <View style={styles.convMeta}>
                <Text style={styles.convMetaText}>Turns: {item.turns?.length ?? 0}</Text>
                <Text style={styles.convMetaText}>Participants: {item.participants?.join(', ')}</Text>
                <Text style={styles.convMetaText}>Duration: {Math.round(item.duration_seconds)}s</Text>
              </View>
              {item.start_time ? (
                <Text style={styles.convTime}>{new Date(item.start_time).toLocaleString()}</Text>
              ) : null}
              <View style={styles.convBadges}>
                {item.auto_ingested ? <Badge label="Auto-ingested" variant="success" size="sm" /> : null}
                {item.memory_ids?.length > 0 ? (
                  <Badge label={`${item.memory_ids.length} memories`} variant="primary" size="sm" />
                ) : null}
              </View>
            </Card>
          )}
        />
      )}

      {activeTab === 'settings' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          <SectionHeader title="Voice Configuration" />

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Status" icon={<AppIcon name="information-outline" size={16} color="#6366f1" />} />
            <View style={styles.configRow}>
              <Text style={styles.configLabel}>Companion Session</Text>
              <Badge label={clientSessionId || ambientState?.client_session?.active_session_id ? 'Active' : 'Idle'} variant={clientSessionId || ambientState?.client_session?.active_session_id ? 'success' : 'default'} size="sm" />
            </View>
            <View style={styles.configRow}>
              <Text style={styles.configLabel}>Gemini Available</Text>
              <Badge label={ambientState?.gemini_available ? 'Yes' : 'No'} variant={ambientState?.gemini_available ? 'info' : 'default'} size="sm" />
            </View>
            <View style={styles.configRow}>
              <Text style={styles.configLabel}>Operating Mode</Text>
              <Badge label={ambientState?.operating_mode || 'classic'} variant="violet" size="sm" />
            </View>
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Companion Identity" icon={<AppIcon name="robot-outline" size={16} color="#8b5cf6" />} />
            <View style={styles.configRow}>
              <Text style={styles.configLabel}>Assistant Name</Text>
              <Text style={styles.configValue}>{assistantName}</Text>
            </View>
            <View style={styles.configRow}>
              <Text style={styles.configLabel}>Aliases</Text>
              <Text style={styles.configValue}>
                {(ambientConfig?.assistant_aliases || ['eva', 'cortex']).join(', ')}
              </Text>
            </View>
            <View style={styles.configRow}>
              <Text style={styles.configLabel}>Follow-up Window</Text>
              <Text style={styles.configValue}>{ambientConfig?.companion_followup_window_s || 45}s</Text>
            </View>
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Edit Wake Settings" icon={<AppIcon name="tune-variant" size={16} color="#6366f1" />} />
            <TextInput
              label="Assistant Name"
              value={assistantNameDraft}
              onChangeText={setAssistantNameDraft}
              placeholder="Eva"
              style={styles.fieldSpacing}
            />
            <TextInput
              label="Wake Aliases"
              value={assistantAliasesDraft}
              onChangeText={setAssistantAliasesDraft}
              placeholder="eva, cortex, assistant"
              style={styles.fieldSpacing}
            />
            <TextInput
              label="Follow-up Window (seconds)"
              value={followupWindowDraft}
              onChangeText={setFollowupWindowDraft}
              placeholder="45"
              keyboardType="number-pad"
              style={styles.fieldSpacing}
            />
            <Button
              label={savingCompanionSettings ? 'Saving...' : 'Save Companion Settings'}
              onPress={saveCompanionSettings}
              loading={savingCompanionSettings}
              fullWidth
            />
            {companionSettingsMessage ? (
              <Text style={styles.settingsMessage}>{companionSettingsMessage}</Text>
            ) : null}
          </Card>

          {voiceProviders ? (
            <Card variant="outlined" padding="lg">
              <SectionHeader title="Providers" icon={<AppIcon name="server-network" size={16} color="#8b5cf6" />} />
              <View style={styles.configRow}>
                <Text style={styles.configLabel}>STT Provider</Text>
                <Badge label={voiceProviders.stt_provider} variant="primary" size="sm" />
              </View>
              <View style={styles.configRow}>
                <Text style={styles.configLabel}>TTS Provider</Text>
                <Badge label={voiceProviders.tts_provider} variant="violet" size="sm" />
              </View>
              <View style={styles.configRow}>
                <Text style={styles.configLabel}>Live Mode</Text>
                <Badge label={voiceProviders.live_mode || 'classic'} variant="default" size="sm" />
              </View>
            </Card>
          ) : null}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },
  scrollView: { flex: 1 },
  scrollContent: { padding: SPACING.lg, paddingBottom: SPACING['5xl'], gap: SPACING.md },

  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACING.xs,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.lg,
  },
  tabActive: { backgroundColor: '#eef2ff' },
  tabLabel: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.medium, color: '#94a3b8' },
  tabLabelActive: { color: '#6366f1', fontWeight: FONT_WEIGHT.semibold },

  heroHeader: {
    flexDirection: 'row',
    gap: SPACING.md,
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  heroEyebrow: {
    fontSize: 10,
    letterSpacing: 1.6,
    color: '#6366f1',
    fontWeight: FONT_WEIGHT.bold,
    marginBottom: 8,
  },
  heroTitle: {
    fontSize: FONT_SIZE['2xl'],
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
    marginBottom: 8,
  },
  heroBody: {
    fontSize: FONT_SIZE.sm,
    lineHeight: 20,
    color: '#475569',
  },
  controlButtons: { flexDirection: 'row', gap: SPACING.sm, marginTop: SPACING.lg },
  controlBtn: { flex: 1 },
  liveStats: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.md, marginTop: SPACING.md },
  liveStatText: { fontSize: FONT_SIZE.xs, color: '#64748b' },
  errorBox: {
    marginTop: SPACING.md,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: '#fecaca',
    backgroundColor: '#fef2f2',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  errorText: { fontSize: FONT_SIZE.sm, color: '#dc2626' },
  fieldSpacing: { marginBottom: SPACING.md },
  settingsMessage: {
    marginTop: SPACING.sm,
    fontSize: FONT_SIZE.xs,
    color: '#6366f1',
  },

  metricsGrid: { flexDirection: 'row', gap: SPACING.sm },
  metricHalf: { flex: 1 },

  panelText: {
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    lineHeight: 20,
  },

  noData: {
    fontSize: FONT_SIZE.sm,
    color: '#94a3b8',
    paddingVertical: SPACING.md,
    textAlign: 'center',
  },
  transcriptRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
    alignItems: 'flex-start',
  },
  transcriptText: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    lineHeight: 18,
  },
  retentionHeader: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    marginBottom: SPACING.sm,
  },
  tagWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.xs,
    marginTop: SPACING.sm,
  },

  convTitle: { fontSize: FONT_SIZE.md, fontWeight: FONT_WEIGHT.semibold, color: '#0f172a', marginBottom: SPACING.xs },
  convMeta: { flexDirection: 'row', gap: SPACING.md, marginBottom: SPACING.xs, flexWrap: 'wrap' },
  convMetaText: { fontSize: 10, color: '#94a3b8' },
  convTime: { fontSize: 10, color: '#cbd5e1', marginBottom: SPACING.xs },
  convBadges: { flexDirection: 'row', gap: SPACING.sm, flexWrap: 'wrap' },

  configRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
    gap: SPACING.md,
  },
  configLabel: { fontSize: FONT_SIZE.sm, color: '#475569', fontWeight: FONT_WEIGHT.medium, flex: 1 },
  configValue: { fontSize: FONT_SIZE.sm, color: '#0f172a', fontWeight: FONT_WEIGHT.semibold, flex: 1, textAlign: 'right' },
});
