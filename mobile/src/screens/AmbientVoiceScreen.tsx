/**
 * AmbientVoiceScreen — Neural Dark Ambient Listening
 * Stitch ref: fe72ef75335741a59461ffbf6011f261
 * Hero: pulsing microphone circle, STT/TTS controls, live transcript
 */
import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Animated,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { NeuralPulse } from '../components/ui/NeuralPulse';
import { AppIcon } from '../components/ui/AppIcon';
import type {
  AmbientState, AmbientConfig, VoiceProviders,
  ConversationTurn, ConversationRecord,
} from '../../shared/core/types';
import type { TTSStatus } from '../../shared/core/api';

function toPercent(v?: number): string {
  if (typeof v !== 'number') return '0%';
  return `${Math.round(v * 100)}%`;
}
function fmtDuration(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

interface AmbientVoiceScreenProps {
  ambientState: AmbientState | null;
  ambientConfig: AmbientConfig | null;
  ambientEnrollment: { enrolled: boolean; speaker_id_available?: boolean } | null;
  ambientProviders: VoiceProviders | null;
  ambientTurns: ConversationTurn[];
  ambientConversations: ConversationRecord[];
  ambientTTSStatus: TTSStatus | null;
  ambientBusy: boolean;
  loadingView: boolean;
  ttsDraft: string;
  setTtsDraft: (v: string) => void;
  ttsBusy: boolean;
  ttsLastBytes: number | null;
  onAmbientAction: (a: 'start' | 'stop' | 'pause' | 'resume') => void;
  onSetProvider: (kind: 'stt' | 'tts', provider: 'traditional' | 'gemini') => void;
  onStartEnrollment: () => void;
  onToggleAutoIngest: () => void;
  onRunTTS: () => void;
}

/** Pulsing microphone hero circle */
function MicHero({ isListening }: { isListening: boolean }) {
  const scale   = useRef(new Animated.Value(1)).current;
  const opacity = useRef(new Animated.Value(0.4)).current;

  useEffect(() => {
    if (!isListening) { scale.setValue(1); opacity.setValue(0.15); return; }
    const loop = Animated.loop(
      Animated.parallel([
        Animated.sequence([
          Animated.timing(scale,   { toValue: 1.35, duration: 900, useNativeDriver: true }),
          Animated.timing(scale,   { toValue: 1,    duration: 900, useNativeDriver: true }),
        ]),
        Animated.sequence([
          Animated.timing(opacity, { toValue: 0,   duration: 900, useNativeDriver: true }),
          Animated.timing(opacity, { toValue: 0.35,duration: 900, useNativeDriver: true }),
        ]),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [isListening]);

  return (
    <View style={hero.wrap}>
      {/* Outer pulse ring */}
      <Animated.View style={[hero.ring, { transform: [{ scale }], opacity }]} />
      {/* Core gradient circle */}
      <LinearGradient
        colors={isListening ? [NEURAL.primary, NEURAL.secondary] : [NEURAL.surfaceContainerHigh, NEURAL.surfaceContainerHighest]}
        start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
        style={hero.core}
      >
        <AppIcon name="microphone" size={38} color={NEURAL.onSurface} style={hero.icon} />
      </LinearGradient>
    </View>
  );
}

const hero = StyleSheet.create({
  wrap: { alignItems: 'center', justifyContent: 'center', paddingVertical: SPACING['3xl'], position: 'relative' },
  ring: {
    position: 'absolute',
    width: 130, height: 130, borderRadius: 65,
    borderWidth: 2, borderColor: NEURAL.primary,
  },
  core: {
    width: 96, height: 96, borderRadius: 48,
    alignItems: 'center', justifyContent: 'center',
  },
  icon: { marginTop: 1 },
});

export function AmbientVoiceScreen({
  ambientState, ambientConfig, ambientEnrollment, ambientProviders,
  ambientTurns, ambientConversations, ambientTTSStatus,
  ambientBusy, loadingView, ttsDraft, setTtsDraft, ttsBusy, ttsLastBytes,
  onAmbientAction, onSetProvider, onStartEnrollment, onToggleAutoIngest, onRunTTS,
}: AmbientVoiceScreenProps) {
  const isListening = ambientState?.status === 'listening' || ambientState?.status === 'speech_detected' || ambientState?.status === 'transcribing';

  if (loadingView) {
    return <View style={[s.container, s.center]}><ActivityIndicator color={NEURAL.primary} size="large" /></View>;
  }

  if (!ambientState) {
    return (
      <View style={[s.container, s.center]}>
        <Text style={s.emptyTitle}>Ambient unavailable</Text>
        <Text style={s.emptyBody}>Start backend ambient components and refresh.</Text>
      </View>
    );
  }

  return (
    <View style={s.container}>
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Title */}
        <View style={s.header}>
          <Text style={s.title}>Ambient Voice</Text>
        </View>

        {/* Status badges top row */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.statusRow}>
          <Badge label={`STT: ${ambientState.stt_provider}`} variant="info" />
          <Badge label={`TTS: ${ambientState.tts_provider}`} variant="secondary" />
          <Badge
            label={ambientEnrollment?.enrolled ? 'Voice ID: Enrolled' : 'Voice ID: Missing'}
            variant={ambientEnrollment?.enrolled ? 'success' : 'warning'}
          />
          <Badge
            label={`${ambientState.status}`}
            variant={isListening ? 'success' : 'ghost'}
            dot={isListening}
          />
        </ScrollView>

        {/* Hero mic */}
        <MicHero isListening={isListening} />

        {/* 4 metric tiles */}
        <View style={s.metricGrid}>
          {[
            { label: 'Uptime',         value: fmtDuration(ambientState.uptime_seconds) },
            { label: 'Segments',       value: `${ambientState.speech_segments}` },
            { label: 'Transcriptions', value: `${ambientState.transcriptions}` },
            { label: 'Audio Level',    value: `${Math.round(ambientState.audio_level || 0)} dB` },
          ].map((m) => (
            <Card key={m.label} variant="elevated" style={s.metricTile}>
              <Text style={s.metricValue}>{m.value}</Text>
              <Text style={s.metricLabel}>{m.label}</Text>
            </Card>
          ))}
        </View>

        {/* Control buttons */}
        <View style={s.controlRow}>
          <Button label="Start"  size="sm" onPress={() => onAmbientAction('start')}  disabled={ambientBusy || isListening || ambientState?.status === 'paused'} />
          <Button label="Pause"  size="sm" variant="secondary" onPress={() => onAmbientAction('pause')}  disabled={ambientBusy || !isListening} />
          <Button label="Resume" size="sm" variant="outline"   onPress={() => onAmbientAction('resume')} disabled={ambientBusy || ambientState?.status !== 'paused'} />
          <Button label="Stop"   size="sm" variant="error"     onPress={() => onAmbientAction('stop')}   disabled={ambientBusy || (ambientState?.status === 'idle' && !isListening)} />
        </View>

        {/* STT Provider */}
        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Speech-to-Text Provider</Text>
          <View style={s.segmentRow}>
            {(['traditional', 'gemini'] as const).map((p) => (
              <TouchableOpacity
                key={p}
                onPress={() => onSetProvider('stt', p)}
                disabled={ambientBusy || (p === 'traditional' && ambientProviders?.traditional_stt_available === false)}
                style={[s.segmentBtn, ambientState.stt_provider === p && s.segmentBtnActive]}
              >
                <Text style={[s.segmentText, ambientState.stt_provider === p && s.segmentTextActive]}>
                  {p === 'traditional' ? 'Whisper' : 'Gemini'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card>

        {/* TTS Provider */}
        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Text-to-Speech Provider</Text>
          <View style={s.segmentRow}>
            {(['traditional', 'gemini'] as const).map((p) => (
              <TouchableOpacity
                key={p}
                onPress={() => onSetProvider('tts', p)}
                disabled={ambientBusy || (p === 'traditional' && ambientProviders?.traditional_tts_available === false)}
                style={[s.segmentBtn, ambientState.tts_provider === p && s.segmentBtnActive]}
              >
                <Text style={[s.segmentText, ambientState.tts_provider === p && s.segmentTextActive]}>
                  {p === 'traditional' ? 'Piper' : 'Gemini'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          {ambientProviders?.gemini_tts_voices?.length ? (
            <Text style={s.hintText}>Voices: {ambientProviders.gemini_tts_voices.slice(0, 4).join(', ')}</Text>
          ) : null}
        </Card>

        {/* Enrollment */}
        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Voice Enrollment</Text>
          <View style={s.statRow}>
            <Text style={s.statLabel}>Voice enrolled</Text>
            <Text style={s.statValue}>{ambientEnrollment?.enrolled ? 'Yes' : 'No'}</Text>
          </View>
          <View style={s.statRow}>
            <Text style={s.statLabel}>Auto ingest</Text>
            <Text style={s.statValue}>{ambientConfig?.auto_ingest ? 'Enabled' : 'Disabled'}</Text>
          </View>
          <View style={s.btnRow}>
            <Button label={ambientBusy ? 'Working…' : 'Start Enrollment'} size="sm" variant="outline" onPress={onStartEnrollment} disabled={ambientBusy || ambientEnrollment?.speaker_id_available === false} />
            <Button label={ambientConfig?.auto_ingest ? 'Disable Auto-Ingest' : 'Enable Auto-Ingest'} size="sm" variant="secondary" onPress={onToggleAutoIngest} disabled={ambientBusy || !ambientConfig} />
          </View>
        </Card>

        {/* Live Transcript */}
        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Live Transcript</Text>
          {ambientTurns.length > 0 ? (
            [...ambientTurns].reverse().slice(0, 8).map((turn, i) => (
              <View key={`${turn.timestamp}-${i}`} style={s.turnRow}>
                <View style={s.turnMeta}>
                  <View style={s.speakerPill}>
                    <Text style={s.speakerText}>{turn.speaker_name || turn.speaker_label || 'Speaker'}</Text>
                  </View>
                  <Text style={s.turnConf}>{toPercent(turn.confidence)}</Text>
                </View>
                <Text style={s.turnText}>{turn.text}</Text>
              </View>
            ))
          ) : (
            <Text style={s.emptyBody}>No live transcript turns yet.</Text>
          )}
        </Card>

        {/* TTS Test Panel */}
        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>TTS Test Panel</Text>
          <TextInput
            placeholder="Type text to synthesize…"
            value={ttsDraft}
            onChangeText={setTtsDraft}
            multiline
            style={{ marginBottom: SPACING.sm }}
          />
          <View style={s.btnRow}>
            <Button label={ttsBusy ? 'Synthesizing…' : 'Synthesize WAV'} size="sm" onPress={onRunTTS} disabled={ttsBusy || !ttsDraft.trim()} loading={ttsBusy} />
          </View>
          {ttsLastBytes != null && (
            <Text style={s.hintText}>Last: {Math.round(ttsLastBytes / 1024)} KB synthesized</Text>
          )}
          <View style={s.statRow}>
            <Text style={s.statLabel}>TTS available</Text>
            <Text style={s.statValue}>{ambientTTSStatus?.available ? 'Yes' : 'No'}</Text>
          </View>
          <View style={s.statRow}>
            <Text style={s.statLabel}>Voice</Text>
            <Text style={s.statValue}>{ambientTTSStatus?.voice || '—'}</Text>
          </View>
        </Card>

        {/* Past Sessions */}
        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Past Sessions</Text>
          {ambientConversations.length > 0 ? (
            ambientConversations.slice(0, 6).map((conv) => (
              <View key={conv.id} style={s.sessionRow}>
                <View style={s.sessionInfo}>
                  <Text style={s.sessionParticipants} numberOfLines={1}>
                    {conv.participants.join(', ') || 'Unknown'}
                  </Text>
                  <Text style={s.sessionMeta}>
                    {conv.turns.length} turns · {fmtDuration(conv.duration_seconds)}
                  </Text>
                </View>
                <Badge
                  label={conv.auto_ingested ? 'Ingested' : 'Pending'}
                  variant={conv.auto_ingested ? 'success' : 'warning'}
                  small
                />
              </View>
            ))
          ) : (
            <Text style={s.emptyBody}>No ambient sessions yet.</Text>
          )}
        </Card>
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  center: { alignItems: 'center', justifyContent: 'center' },
  scroll: { paddingBottom: SPACING['5xl'] },
  header: { paddingHorizontal: SPACING.lg, paddingTop: SPACING.lg, paddingBottom: SPACING.sm },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  statusRow: { paddingHorizontal: SPACING.lg, gap: SPACING.sm, marginBottom: SPACING.sm },

  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    marginBottom: SPACING.md,
  },
  metricTile: { width: '47%', flexGrow: 1, alignItems: 'center', paddingVertical: SPACING.md, gap: 3 },
  metricValue: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.primary },
  metricLabel: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  controlRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    marginBottom: SPACING.md,
    flexWrap: 'wrap',
  },

  card: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  cardTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },

  segmentRow: { flexDirection: 'row', gap: SPACING.sm },
  segmentBtn: {
    flex: 1,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.lg,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    alignItems: 'center',
  },
  segmentBtnActive: {
    backgroundColor: `${NEURAL.primary}26`,
    borderColor: `${NEURAL.primary}60`,
  },
  segmentText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, fontWeight: FONT_WEIGHT.medium },
  segmentTextActive: { color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },

  hintText: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, marginTop: 2 },
  statRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: SPACING.xs, borderTopWidth: 1, borderTopColor: `${NEURAL.outlineVariant}40` },
  statLabel: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
  statValue: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold },
  btnRow: { flexDirection: 'row', gap: SPACING.sm, flexWrap: 'wrap' },

  turnRow: { paddingVertical: SPACING.sm, borderTopWidth: 1, borderTopColor: `${NEURAL.outlineVariant}40`, gap: 4 },
  turnMeta: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  speakerPill: {
    backgroundColor: `${NEURAL.primary}22`,
    borderRadius: RADIUS.full,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderWidth: 1,
    borderColor: `${NEURAL.primary}40`,
  },
  speakerText: { fontSize: FONT_SIZE.xs, color: NEURAL.primary, fontWeight: FONT_WEIGHT.semibold },
  turnConf: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },
  turnText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, lineHeight: FONT_SIZE.sm * 1.5 },

  sessionRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: SPACING.sm, borderTopWidth: 1, borderTopColor: `${NEURAL.outlineVariant}40`, gap: SPACING.sm },
  sessionInfo: { flex: 1 },
  sessionParticipants: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.medium },
  sessionMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  emptyTitle: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, marginBottom: SPACING.sm },
  emptyBody: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, textAlign: 'center' },
});
