/**
 * AgentScreen — Neural Dark autonomous agent operations.
 * Includes session control, query streaming, steering, and live event feed.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge, type BadgeVariant } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { NeuralPulse } from '../components/ui/NeuralPulse';
import { AppIcon } from '../components/ui/AppIcon';
import type { ApiClient } from '../../shared/core/api';
import type {
  AgentConfigInfo,
  AgentSessionInfo,
  CortexEvent,
  TierClassification,
} from '../../shared/core/types';

interface AgentScreenProps {
  api: ApiClient;
}

const TIER_VARIANT: Record<string, BadgeVariant> = {
  T0: 'success',
  T1: 'tertiary',
  T2: 'primary',
  T3: 'warning',
  T4: 'error',
};

function formatIsoTime(iso: string | undefined): string {
  if (!iso) return 'now';
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return iso;
  const delta = Date.now() - ts;
  if (delta < 8000) return 'just now';
  if (delta < 60000) return `${Math.round(delta / 1000)}s ago`;
  if (delta < 3600000) return `${Math.round(delta / 60000)}m ago`;
  return `${Math.round(delta / 3600000)}h ago`;
}

function safeString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function extractTier(event: CortexEvent): TierClassification | null {
  if (event.type !== 'tier_selected') return null;
  const data = event.data;
  const tier = safeString(data.tier);
  if (!tier) return null;
  return {
    tier: tier as TierClassification['tier'],
    complexity: typeof data.complexity === 'number' ? data.complexity : 0,
    intent: safeString(data.intent) || 'unknown',
    entities: Array.isArray(data.entities) ? data.entities.filter((x): x is string => typeof x === 'string') : [],
    topics: Array.isArray(data.topics) ? data.topics.filter((x): x is string => typeof x === 'string') : [],
    sub_queries: Array.isArray(data.sub_queries) ? data.sub_queries.filter((x): x is string => typeof x === 'string') : [],
    confidence: typeof data.confidence === 'number' ? data.confidence : 0,
    cache_key: safeString(data.cache_key),
    recommended_agents: Array.isArray(data.recommended_agents)
      ? data.recommended_agents.filter((x): x is string => typeof x === 'string')
      : [],
    estimated_latency_ms: typeof data.estimated_latency_ms === 'number' ? data.estimated_latency_ms : 0,
  };
}

function extractAnswer(event: CortexEvent): string {
  if (event.type !== 'agent_end') return '';
  return safeString(event.data.answer);
}

function prependUniqueEvent(prev: CortexEvent[], event: CortexEvent): CortexEvent[] {
  const identity = event.event_id || `${event.type}:${event.timestamp}:${event.session_id}`;
  const exists = prev.some((existing) => {
    const existingIdentity = existing.event_id || `${existing.type}:${existing.timestamp}:${existing.session_id}`;
    return existingIdentity === identity;
  });
  if (exists) {
    return prev;
  }
  return [event, ...prev].slice(0, 80);
}

export function AgentScreen({ api }: AgentScreenProps) {
  const [sessions, setSessions] = useState<AgentSessionInfo[]>([]);
  const [configs, setConfigs] = useState<AgentConfigInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const [queryDraft, setQueryDraft] = useState('');
  const [steerDraft, setSteerDraft] = useState('');
  const [followUpDraft, setFollowUpDraft] = useState('');

  const [answer, setAnswer] = useState('');
  const [tier, setTier] = useState<TierClassification | null>(null);
  const [events, setEvents] = useState<CortexEvent[]>([]);

  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState('');

  const activeSession = useMemo(
    () => sessions.find((s) => s.session_id === activeSessionId) || null,
    [sessions, activeSessionId],
  );

  const loadOverview = useCallback(async () => {
    setLoading(true);
    try {
      const [sessionsResponse, configResponse] = await Promise.all([
        api.listAgentSessions(),
        api.listAgentConfigs(),
      ]);
      setSessions(sessionsResponse.sessions || []);
      setConfigs(configResponse.agents || []);
      setActiveSessionId((prev) => {
        if (prev && sessionsResponse.sessions.some((s) => s.session_id === prev)) {
          return prev;
        }
        return sessionsResponse.sessions[0]?.session_id || null;
      });
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    void loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    if (Platform.OS !== 'web') {
      return;
    }

    const controller = api.subscribeAgentEvents({
      onEvent: (event) => {
        setEvents((prev) => prependUniqueEvent(prev, event));
      },
      onError: (streamError) => {
        setError(streamError.message);
      },
    });

    return () => {
      controller.abort();
    };
  }, [api]);

  const createSession = useCallback(async () => {
    setBusy(true);
    try {
      const created = await api.createAgentSession();
      await loadOverview();
      setActiveSessionId(created.session_id);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api, loadOverview]);

  const closeSession = useCallback(
    async (sessionId: string) => {
      setBusy(true);
      try {
        await api.closeAgentSession(sessionId);
        await loadOverview();
        if (activeSessionId === sessionId) {
          setActiveSessionId(null);
        }
        setError('');
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [activeSessionId, api, loadOverview],
  );

  const runQuery = useCallback(async () => {
    const prompt = queryDraft.trim();
    if (!prompt || busy) return;

    setBusy(true);
    setStreaming(true);
    setAnswer('');
    setEvents([]);

    try {
      const streamResult = await api.streamAgentQuery(
        { query: prompt, sessionId: activeSessionId },
        {
          onEvent: (event) => {
            setEvents((prev) => prependUniqueEvent(prev, event));

            const maybeTier = extractTier(event);
            if (maybeTier) {
              setTier(maybeTier);
            }

            const maybeAnswer = extractAnswer(event);
            if (maybeAnswer) {
              setAnswer(maybeAnswer);
            }
          },
          onError: (streamError) => {
            setError(streamError.message);
          },
        },
      );

      if (streamResult.answer) {
        setAnswer(streamResult.answer);
      }
      if (streamResult.sessionId) {
        setActiveSessionId(streamResult.sessionId);
      }
      setQueryDraft('');
      setError('');
    } catch {
      try {
        const fallback = await api.agentQuery(prompt, activeSessionId);
        setAnswer(fallback.answer);
        setTier(fallback.tier);
        if (fallback.session_id) {
          setActiveSessionId(fallback.session_id);
        }
        setQueryDraft('');
        setError('');
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setStreaming(false);
      setBusy(false);
      await loadOverview();
    }
  }, [activeSessionId, api, busy, loadOverview, queryDraft]);

  const runSteer = useCallback(async () => {
    const text = steerDraft.trim();
    if (!activeSessionId || !text || busy) return;
    setBusy(true);
    try {
      await api.steerAgent(activeSessionId, text);
      setSteerDraft('');
      await loadOverview();
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [activeSessionId, api, busy, loadOverview, steerDraft]);

  const runFollowUp = useCallback(async () => {
    const text = followUpDraft.trim();
    if (!activeSessionId || !text || busy) return;
    setBusy(true);
    try {
      await api.followUpAgent(activeSessionId, text);
      setFollowUpDraft('');
      await loadOverview();
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [activeSessionId, api, busy, followUpDraft, loadOverview]);

  const runAbort = useCallback(async () => {
    if (!activeSessionId || busy) return;
    setBusy(true);
    try {
      await api.abortAgent(activeSessionId);
      await loadOverview();
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setStreaming(false);
    }
  }, [activeSessionId, api, busy, loadOverview]);

  return (
    <View style={s.container}>
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.header}>
          <View>
            <Text style={s.title}>Agent Ops</Text>
            <Text style={s.subtitle}>Autonomous sessions, steering, and streaming traces</Text>
          </View>
          <View style={s.headerStatus}>
            {streaming ? <NeuralPulse active size={6} color={NEURAL.tertiary} /> : null}
            <Badge label={streaming ? 'Streaming' : 'Idle'} variant={streaming ? 'success' : 'ghost'} small dot={streaming} />
          </View>
        </View>

        {error ? (
          <Card variant="outlined" style={s.errorCard}>
            <Text style={s.errorText}>{error}</Text>
          </Card>
        ) : null}

        <Card variant="outlined" style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardTitle}>Session Control</Text>
            <View style={s.rowBtns}>
              <Button label="Refresh" variant="ghost" size="xs" onPress={() => void loadOverview()} disabled={busy || loading} />
              <Button label="New Session" size="xs" onPress={() => void createSession()} disabled={busy} />
            </View>
          </View>

          <View style={s.badgeRow}>
            <Badge label={`Sessions: ${sessions.length}`} variant="info" small />
            <Badge label={`Agents: ${configs.length}`} variant="secondary" small />
            {activeSessionId ? <Badge label={`Active: ${activeSessionId.slice(0, 8)}`} variant="primary" small /> : null}
          </View>

          {loading ? (
            <ActivityIndicator color={NEURAL.primary} />
          ) : sessions.length > 0 ? (
            <View style={s.sessionList}>
              {sessions.slice(0, 6).map((session) => {
                const selected = session.session_id === activeSessionId;
                return (
                  <TouchableOpacity
                    key={session.session_id}
                    style={[s.sessionRow, selected && s.sessionRowActive]}
                    onPress={() => setActiveSessionId(session.session_id)}
                    disabled={busy}
                    activeOpacity={0.82}
                  >
                    <View style={s.sessionInfo}>
                      <Text style={s.sessionId} numberOfLines={1}>{session.session_id}</Text>
                      <Text style={s.sessionMeta}>
                        {session.agent_id} · {session.turn_count} turns · {session.message_count} msgs
                      </Text>
                    </View>
                    <Badge label={session.is_running ? 'Running' : 'Idle'} variant={session.is_running ? 'success' : 'ghost'} small />
                  </TouchableOpacity>
                );
              })}
            </View>
          ) : (
            <Text style={s.empty}>No active sessions yet.</Text>
          )}
        </Card>

        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Agent Query</Text>
          <TextInput
            placeholder="Ask the orchestrator a complex query..."
            value={queryDraft}
            onChangeText={setQueryDraft}
            multiline
            style={s.queryInput}
          />
          <View style={s.rowBtns}>
            <Button
              label={streaming ? 'Streaming…' : 'Run Query'}
              onPress={() => void runQuery()}
              loading={streaming}
              disabled={busy || !queryDraft.trim()}
              size="sm"
            />
            <Button
              label="Abort"
              variant="error"
              size="sm"
              onPress={() => void runAbort()}
              disabled={busy || !activeSessionId}
            />
          </View>
        </Card>

        {tier ? (
          <Card variant="elevated" style={s.card} leftAccent leftAccentColor={NEURAL.secondary}>
            <View style={s.cardHeader}>
              <Text style={s.cardTitle}>Tier Routing</Text>
              <Badge label={tier.tier} variant={TIER_VARIANT[tier.tier] || 'primary'} />
            </View>
            <Text style={s.bodyText}>Intent: {tier.intent}</Text>
            <Text style={s.bodyText}>Complexity: {tier.complexity.toFixed(2)} · Confidence: {Math.round(tier.confidence * 100)}%</Text>
            <View style={s.badgeRow}>
              {(tier.recommended_agents || []).slice(0, 4).map((agentId) => (
                <Badge key={agentId} label={agentId} variant="tertiary" small />
              ))}
            </View>
          </Card>
        ) : null}

        {answer ? (
          <Card variant="elevated" style={s.card} leftAccent leftAccentColor={NEURAL.tertiary}>
            <Text style={s.answerLabel}>Agent Answer</Text>
            <Text style={s.answerText}>{answer}</Text>
          </Card>
        ) : null}

        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Steering Queue</Text>
          <TextInput
            placeholder="Steering instruction for current session"
            value={steerDraft}
            onChangeText={setSteerDraft}
            multiline
            style={s.queryInput}
          />
          <Button
            label="Inject Steering"
            size="sm"
            variant="secondary"
            onPress={() => void runSteer()}
            disabled={busy || !activeSessionId || !steerDraft.trim()}
          />

          <TextInput
            placeholder="Follow-up to run when idle"
            value={followUpDraft}
            onChangeText={setFollowUpDraft}
            multiline
            style={s.followInput}
          />
          <Button
            label="Queue Follow-Up"
            size="sm"
            variant="outline"
            onPress={() => void runFollowUp()}
            disabled={busy || !activeSessionId || !followUpDraft.trim()}
          />

          {activeSession ? (
            <View style={s.queueState}>
              <Text style={s.queueTitle}>Current Queue</Text>
              <Text style={s.queueText} numberOfLines={2}>
                Steer: {activeSession.steering.steering.length} · Follow-up: {activeSession.steering.followUp.length}
              </Text>
              <Button
                label="Close Session"
                size="xs"
                variant="ghost"
                onPress={() => void closeSession(activeSession.session_id)}
                disabled={busy}
              />
            </View>
          ) : null}
        </Card>

        <Card variant="outlined" style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardTitle}>Event Feed</Text>
            <Badge label={`${events.length} events`} variant="info" small />
          </View>
          {events.length > 0 ? (
            events.slice(0, 14).map((event, index) => (
              <View key={`${event.type}-${event.timestamp}-${index}`} style={s.eventRow}>
                <AppIcon name="timeline-clock-outline" size={16} color={NEURAL.onSurfaceVariant} style={s.eventIcon} />
                <View style={s.eventBody}>
                  <Text style={s.eventType}>{event.type}</Text>
                  <Text style={s.eventMeta}>
                    {event.agent_id || 'agent'} · {formatIsoTime(event.timestamp)}
                  </Text>
                </View>
              </View>
            ))
          ) : (
            <Text style={s.empty}>No events yet. Run a query to populate live traces.</Text>
          )}
        </Card>
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  scroll: { paddingBottom: SPACING['5xl'] },
  header: {
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.md,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: SPACING.sm,
  },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  subtitle: { marginTop: 4, fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
  headerStatus: { flexDirection: 'row', alignItems: 'center', gap: 6 },

  card: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: SPACING.sm },
  cardTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },

  badgeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.xs },
  rowBtns: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.xs },

  sessionList: { gap: SPACING.xs },
  sessionRow: {
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}70`,
    backgroundColor: NEURAL.surfaceContainerLow,
    padding: SPACING.sm,
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  sessionRowActive: {
    borderColor: `${NEURAL.primary}80`,
    backgroundColor: `${NEURAL.primary}18`,
  },
  sessionInfo: { flex: 1, gap: 2 },
  sessionId: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface },
  sessionMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  queryInput: { marginBottom: 2 },
  followInput: { marginTop: SPACING.sm, marginBottom: 2 },

  answerLabel: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.bold,
    textTransform: 'uppercase',
    color: NEURAL.tertiary,
    letterSpacing: 0.4,
  },
  answerText: { fontSize: FONT_SIZE.base, color: NEURAL.onSurface, lineHeight: FONT_SIZE.base * 1.6 },
  bodyText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },

  queueState: {
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
    paddingTop: SPACING.sm,
    gap: SPACING.xs,
  },
  queueTitle: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold },
  queueText: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  eventRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}30`,
    paddingTop: SPACING.sm,
  },
  eventIcon: { marginTop: 1 },
  eventBody: { flex: 1 },
  eventType: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.medium },
  eventMeta: { marginTop: 2, fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  errorCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md },
  errorText: { fontSize: FONT_SIZE.sm, color: NEURAL.error },
  empty: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
});
