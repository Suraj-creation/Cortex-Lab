/**
 * AgentScreen — Cortex Aurora Autonomous Agent Chat
 * Full agent chat with tier classification, steering, tool execution, sessions
 */
import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Keyboard,
  TextInput as RNTextInput,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { EmptyState } from '../components/ui/EmptyState';
import { Button } from '../components/ui/Button';

type AgentTab = 'chat' | 'sessions' | 'configs' | 'scheduler';

interface AgentMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  tier?: any;
  turnCount?: number;
  isThinking?: boolean;
}

interface AgentScreenProps {
  api: any;
}

const TIER_COLORS: Record<string, { bg: string; border: string; text: string; label: string }> = {
  T0: { bg: '#f1f5f9', border: '#e2e8f0', text: '#475569', label: 'T0 · Instant' },
  T1: { bg: '#eff6ff', border: '#bfdbfe', text: '#1e40af', label: 'T1 · Fast' },
  T2: { bg: '#eef2ff', border: '#c7d2fe', text: '#4338ca', label: 'T2 · Standard' },
  T3: { bg: '#f5f3ff', border: '#ddd6fe', text: '#5b21b6', label: 'T3 · Deep' },
  T4: { bg: '#fdf4ff', border: '#f5d0fe', text: '#86198f', label: 'T4 · Research' },
};

export function AgentScreen({ api }: AgentScreenProps) {
  const insets = useSafeAreaInsets();
  const [activeTab, setActiveTab] = useState<AgentTab>('chat');
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [currentTier, setCurrentTier] = useState<string | null>(null);

  const [steerDraft, setSteerDraft] = useState('');
  const [followUpDraft, setFollowUpDraft] = useState('');
  const [controlBusy, setControlBusy] = useState(false);
  const [agentEvents, setAgentEvents] = useState<any[]>([]);
  const [eventError, setEventError] = useState('');

  // Sessions / configs / scheduler state
  const [sessions, setSessions] = useState<any[]>([]);
  const [configs, setConfigs] = useState<any[]>([]);
  const [scheduler, setScheduler] = useState<any>(null);
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);

  const flatRef = useRef<FlatList>(null);
  const composerLift = Platform.OS === 'android' ? Math.max(0, keyboardHeight - insets.bottom) : 0;
  const listBottomPadding = 208 + (composerLift > 0 ? composerLift : insets.bottom);

  const scrollToEnd = useCallback(() => {
    setTimeout(() => flatRef.current?.scrollToEnd({ animated: true }), 100);
  }, []);

  // Create session if needed
  const ensureSession = useCallback(async (): Promise<string> => {
    if (sessionId) return sessionId;
    try {
      const res = await api.createAgentSession?.('l1_orchestrator');
      const sid = res?.session_id;
      if (sid) setSessionId(sid);
      return sid || '';
    } catch (e) {
      Alert.alert('Error', 'Failed to create agent session');
      return '';
    }
  }, [sessionId, api]);

  // Send query
  const handleSend = useCallback(async () => {
    if (!input.trim() || sending) return;
    const query = input.trim();
    setInput('');
    setSending(true);

    const userMsg: AgentMessage = { id: `u-${Date.now()}`, role: 'user', content: query, timestamp: Date.now() };
    const thinkMsg: AgentMessage = { id: `t-${Date.now()}`, role: 'assistant', content: '', timestamp: Date.now(), isThinking: true };
    setMessages((prev) => [...prev, userMsg, thinkMsg]);
    scrollToEnd();

    try {
      const sid = await ensureSession();
      const res = await api.agentQuery?.(query, sid);
      const answer = res?.answer || res?.response || res?.content || '[No response]';
      const tier = res?.tier_classification || res?.tier || null;
      const turns = res?.turn_count || res?.turns || 0;

      if (tier?.tier) setCurrentTier(tier.tier);

      setMessages((prev) =>
        prev.map((m) =>
          m.id === thinkMsg.id
            ? { ...m, content: answer, isThinking: false, tier, turnCount: turns }
            : m,
        ),
      );
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === thinkMsg.id
            ? { ...m, content: `Error: ${e instanceof Error ? e.message : 'Failed to query agent'}`, isThinking: false }
            : m,
        ),
      );
    } finally {
      setSending(false);
      scrollToEnd();
    }
  }, [input, sending, ensureSession, api, scrollToEnd]);

  // Load sessions/configs/scheduler
  const loadSessions = useCallback(async () => {
    try {
      const res = await api.listAgentSessions?.();
      setSessions(res?.sessions || []);
    } catch {}
  }, [api]);

  const loadConfigs = useCallback(async () => {
    try {
      const res = await api.listAgentConfigs?.();
      setConfigs(res?.agents || []);
    } catch {}
  }, [api]);

  const loadScheduler = useCallback(async () => {
    try {
      const [sched, cache] = await Promise.allSettled([
        api.getSchedulerStatus?.(),
        api.getCacheStats?.(),
      ]);
      if (sched.status === 'fulfilled') setScheduler(sched.value);
      if (cache.status === 'fulfilled') setCacheStats(cache.value);
    } catch {}
  }, [api]);

  useEffect(() => {
    if (activeTab === 'sessions') loadSessions();
    if (activeTab === 'configs') loadConfigs();
    if (activeTab === 'scheduler') loadScheduler();
  }, [activeTab, loadSessions, loadConfigs, loadScheduler]);

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
  }, [insets.bottom]);

  const tabs: { key: AgentTab; label: string; icon: string }[] = [
    { key: 'chat', label: 'Agent Chat', icon: 'robot-outline' },
    { key: 'sessions', label: 'Sessions', icon: 'format-list-bulleted' },
    { key: 'configs', label: 'Agents', icon: 'cog-outline' },
    { key: 'scheduler', label: 'Scheduler', icon: 'clock-outline' },
  ];


  const handleSteer = useCallback(async () => {
    const text = steerDraft.trim();
    if (!text || !sessionId) return;
    setControlBusy(true);
    try {
      await api.steerAgent?.(sessionId, text);
      setSteerDraft('');
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-steer-${Date.now()}`,
          role: 'system',
          content: `Steering queued: ${text}`,
          timestamp: Date.now(),
        },
      ]);
    } catch (e) {
      Alert.alert('Steering Failed', e instanceof Error ? e.message : 'Unable to queue steering instruction.');
    }
    setControlBusy(false);
  }, [api, sessionId, steerDraft]);

  const handleFollowUp = useCallback(async () => {
    const text = followUpDraft.trim();
    if (!text || !sessionId) return;
    setControlBusy(true);
    try {
      await api.followUpAgent?.(sessionId, text);
      setFollowUpDraft('');
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-follow-${Date.now()}`,
          role: 'system',
          content: `Follow-up queued: ${text}`,
          timestamp: Date.now(),
        },
      ]);
    } catch (e) {
      Alert.alert('Follow-up Failed', e instanceof Error ? e.message : 'Unable to queue follow-up message.');
    }
    setControlBusy(false);
  }, [api, followUpDraft, sessionId]);

  const handleAbort = useCallback(async () => {
    if (!sessionId) return;
    setControlBusy(true);
    try {
      await api.abortAgent?.(sessionId);
      setMessages((prev) => [
        ...prev,
        {
          id: `sys-abort-${Date.now()}`,
          role: 'system',
          content: 'Abort signal sent to active session.',
          timestamp: Date.now(),
        },
      ]);
    } catch (e) {
      Alert.alert('Abort Failed', e instanceof Error ? e.message : 'Unable to abort current run.');
    }
    setControlBusy(false);
  }, [api, sessionId]);

  const renderTierBadge = (tier: any) => {
    if (!tier) return null;
    const tierKey = typeof tier === 'string' ? tier : tier.tier || 'T2';
    const cfg = TIER_COLORS[tierKey] || TIER_COLORS.T2;
    return (
      <View style={[styles.tierBadge, { backgroundColor: cfg.bg, borderColor: cfg.border }]}>
        <Text style={[styles.tierText, { color: cfg.text }]}>{cfg.label}</Text>
      </View>
    );
  };

  useEffect(() => {
    const controller = api.subscribeAgentEvents?.({
      onEvent: (event: any) => {
        setAgentEvents((prev) => [event, ...prev].slice(0, 80));
        if (event?.type === 'tier_selected' && event?.data?.tier) {
          setCurrentTier(String(event.data.tier));
        }
      },
      onError: (error: Error) => {
        setEventError(error.message);
      },
    });

    return () => {
      controller?.abort?.();
    };
  }, [api]);

  const toolEvents = agentEvents
    .filter((event) => String(event?.type || '').startsWith('tool_execution'))
    .slice(0, 8);

  const recentEvents = agentEvents.slice(0, 10);

  return (
    <View style={styles.container}>
      {/* Tab bar */}
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => setActiveTab(tab.key)}
            activeOpacity={0.7}
          >
            <AppIcon name={tab.icon as any} size={14} color={activeTab === tab.key ? '#6366f1' : '#94a3b8'} />
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* CHAT TAB */}
      {activeTab === 'chat' && (
        <View style={styles.chatContainer}>
          {/* Session + tier info */}
          {sessionId && (
            <View style={styles.sessionBar}>
              <Text style={styles.sessionId}>Session: {sessionId.slice(0, 12)}...</Text>
              {currentTier && renderTierBadge(currentTier)}
            </View>
          )}

          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.controlPanelsRow}>
            <Card variant="outlined" padding="md" style={styles.controlPanel}>
              <SectionHeader title="Steering" />
              {sessionId ? (
                <>
                  <RNTextInput
                    style={styles.controlInput}
                    placeholder="Steer the agent (priority, style, constraints)"
                    placeholderTextColor="#94a3b8"
                    value={steerDraft}
                    onChangeText={setSteerDraft}
                    editable={!controlBusy}
                  />
                  <Button
                    label="Queue Steering"
                    size="xs"
                    onPress={handleSteer}
                    loading={controlBusy}
                    disabled={!steerDraft.trim()}
                    style={styles.controlAction}
                  />

                  <RNTextInput
                    style={styles.controlInput}
                    placeholder="Queue follow-up message"
                    placeholderTextColor="#94a3b8"
                    value={followUpDraft}
                    onChangeText={setFollowUpDraft}
                    editable={!controlBusy}
                  />
                  <View style={styles.controlActionRow}>
                    <Button
                      label="Queue Follow-up"
                      size="xs"
                      variant="secondary"
                      onPress={handleFollowUp}
                      loading={controlBusy}
                      disabled={!followUpDraft.trim()}
                    />
                    <Button
                      label="Abort"
                      size="xs"
                      variant="error"
                      onPress={handleAbort}
                      loading={controlBusy}
                    />
                  </View>
                </>
              ) : (
                <Text style={styles.controlHint}>Start a chat turn to create a session and unlock steering controls.</Text>
              )}
            </Card>

            <Card variant="outlined" padding="md" style={styles.controlPanel}>
              <SectionHeader title="Tool Activity" />
              {toolEvents.length === 0 ? (
                <Text style={styles.controlHint}>No tool executions captured yet.</Text>
              ) : (
                toolEvents.map((event, index) => (
                  <View key={index} style={styles.eventRow}>
                    <View style={styles.eventHead}>
                      <Badge label={String(event.type || 'tool_execution')} variant="violet" size="sm" />
                      <Text style={styles.eventTime}>{new Date(String(event.timestamp || Date.now())).toLocaleTimeString()}</Text>
                    </View>
                    <Text style={styles.eventDetail} numberOfLines={2}>
                      {JSON.stringify(event.data || {}).slice(0, 180)}
                    </Text>
                  </View>
                ))
              )}
            </Card>

            <Card variant="outlined" padding="md" style={styles.controlPanel}>
              <SectionHeader title="Event Feed" />
              {eventError ? <Badge label={eventError} variant="error" size="sm" /> : null}
              {recentEvents.length === 0 ? (
                <Text style={styles.controlHint}>Live runtime events will appear while the agent is active.</Text>
              ) : (
                recentEvents.map((event, index) => (
                  <View key={index} style={styles.eventRow}>
                    <View style={styles.eventHead}>
                      <Badge label={String(event.type || 'event')} variant="primary" size="sm" />
                      <Text style={styles.eventTime}>{new Date(String(event.timestamp || Date.now())).toLocaleTimeString()}</Text>
                    </View>
                    <Text style={styles.eventDetail} numberOfLines={2}>
                      {JSON.stringify(event.data || {}).slice(0, 180)}
                    </Text>
                  </View>
                ))
              )}
            </Card>
          </ScrollView>

          {/* Messages */}
          <FlatList
            ref={flatRef}
            data={messages}
            keyExtractor={(m) => m.id}
            contentContainerStyle={[styles.msgList, { paddingBottom: listBottomPadding }]}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            renderItem={({ item }) => {
              if (item.role === 'user') {
                return (
                  <View style={styles.userMsgRow}>
                    <LinearGradient colors={['#6366f1', '#8b5cf6']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.userBubble}>
                      <Text style={styles.userMsgText}>{item.content}</Text>
                    </LinearGradient>
                  </View>
                );
              }
              if (item.role === 'system') {
                return (
                  <View style={styles.systemMsgRow}>
                    <Badge label="System" variant="info" size="sm" />
                    <Text style={styles.systemMsgText}>{item.content}</Text>
                  </View>
                );
              }
              return (
                <View style={styles.assistantMsgRow}>
                  <View style={styles.agentAvatar}>
                    <AppIcon name="robot-outline" size={14} color="#6366f1" />
                  </View>
                  <View style={styles.assistantBubble}>
                    {item.isThinking ? (
                      <View style={styles.thinkingRow}>
                        <LoadingSpinner size={16} />
                        <Text style={styles.thinkingText}>Agent processing...</Text>
                      </View>
                    ) : (
                      <>
                        <Text style={styles.assistantMsgText}>{item.content}</Text>
                        {(item.tier || item.turnCount) && (
                          <View style={styles.msgMeta}>
                            {item.tier && renderTierBadge(item.tier)}
                            {item.turnCount ? (
                              <Text style={styles.turnCount}>{item.turnCount} turn{item.turnCount !== 1 ? 's' : ''}</Text>
                            ) : null}
                          </View>
                        )}
                      </>
                    )}
                  </View>
                </View>
              );
            }}
            ListEmptyComponent={
              <EmptyState
                icon="robot-outline"
                title="Cortex Autonomous Agent"
                message="Ask anything. The agent classifies queries (T0-T4), selects specialist agents, retrieves evidence, and synthesizes grounded answers."
              />
            }
          />

          {/* Input */}
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            keyboardVerticalOffset={Platform.OS === 'ios' ? insets.top + 6 : 0}
            style={{ marginBottom: composerLift }}
          >
            <View style={[styles.inputBar, { paddingBottom: Math.max(SPACING.md, insets.bottom + 6) }]}>
              <RNTextInput
                style={styles.chatInput}
                placeholder="Ask Cortex Agent..."
                placeholderTextColor="#94a3b8"
                value={input}
                onChangeText={setInput}
                onSubmitEditing={handleSend}
                editable={!sending}
                selectionColor="#6366f1"
              />
              <TouchableOpacity
                onPress={handleSend}
                disabled={!input.trim() || sending}
                style={[styles.sendBtn, (!input.trim() || sending) && { opacity: 0.4 }]}
              >
                <LinearGradient colors={['#6366f1', '#4f46e5']} style={styles.sendBtnGrad}>
                  <AppIcon name="arrow-up" size={16} color="#ffffff" />
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </KeyboardAvoidingView>
        </View>
      )}

      {/* SESSIONS TAB */}
      {activeTab === 'sessions' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          <SectionHeader
            title="Agent Sessions"
            subtitle={`${sessions.length} sessions`}
            action={{ label: 'Refresh', onPress: loadSessions }}
          />
          {sessions.length === 0 ? (
            <EmptyState icon="format-list-bulleted" title="No Sessions" message="Start a conversation in the Agent Chat to create a session." />
          ) : (
            sessions.map((s, i) => (
              <Card key={s.session_id || i} variant="default" padding="md" style={styles.sessionCard}>
                <View style={styles.sessionHeader}>
                  <Text style={styles.sessionIdText} numberOfLines={1}>{s.session_id}</Text>
                  <Badge label={s.status || 'active'} variant={s.status === 'active' ? 'success' : 'default'} size="sm" />
                </View>
                <Text style={styles.sessionAgent}>{s.agent_id || 'l1_orchestrator'}</Text>
                {s.created_at && <Text style={styles.sessionTime}>{new Date(s.created_at).toLocaleString()}</Text>}
              </Card>
            ))
          )}
        </ScrollView>
      )}

      {/* CONFIGS TAB */}
      {activeTab === 'configs' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          <SectionHeader title="Registered Agents" subtitle={`${configs.length} agents`} action={{ label: 'Refresh', onPress: loadConfigs }} />
          {configs.length === 0 ? (
            <EmptyState icon="cog-outline" title="No Agent Configs" message="Agent configurations will appear when loaded from the backend." />
          ) : (
            configs.map((cfg, i) => (
              <Card key={cfg.agent_id || i} variant="default" padding="md" style={styles.configCard}>
                <View style={styles.configHeader}>
                  <AppIcon name="robot-outline" size={16} color="#6366f1" />
                  <Text style={styles.configId}>{cfg.agent_id}</Text>
                </View>
                {cfg.tools && (
                  <View style={styles.toolChips}>
                    {(cfg.tools || []).slice(0, 6).map((t: any, j: number) => (
                      <Badge key={j} label={typeof t === 'string' ? t : t.name || `tool-${j}`} variant="primary" size="sm" />
                    ))}
                    {cfg.tools.length > 6 && <Badge label={`+${cfg.tools.length - 6}`} variant="default" size="sm" />}
                  </View>
                )}
                <View style={styles.configMeta}>
                  {cfg.max_turns && <Text style={styles.configMetaText}>Max turns: {cfg.max_turns}</Text>}
                  {cfg.context_window && <Text style={styles.configMetaText}>Context: {cfg.context_window}</Text>}
                </View>
              </Card>
            ))
          )}
        </ScrollView>
      )}

      {/* SCHEDULER TAB */}
      {activeTab === 'scheduler' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          <SectionHeader title="Agent Scheduler" action={{ label: 'Refresh', onPress: loadScheduler }} />

          <View style={styles.metricsGrid}>
            <MetricCard
              label="Status"
              value={scheduler?.running ? 'Running' : 'Stopped'}
              tone={scheduler?.running ? 'emerald' : 'rose'}
              compact
              style={styles.metricHalf}
            />
            <MetricCard
              label="Cache Hit Rate"
              value={cacheStats?.hit_rate ? `${(cacheStats.hit_rate * 100).toFixed(1)}%` : 'N/A'}
              tone="indigo"
              compact
              style={styles.metricHalf}
            />
          </View>

          {scheduler?.tasks && Object.keys(scheduler.tasks).length > 0 ? (
            <Card variant="outlined" padding="lg">
              <SectionHeader title="Scheduled Tasks" />
              {Object.entries(scheduler.tasks).map(([taskId, task]: [string, any]) => (
                <View key={taskId} style={styles.schedTask}>
                  <View style={styles.schedTaskHeader}>
                    <Text style={styles.schedTaskId}>{taskId}</Text>
                    <Badge
                      label={task.enabled ? 'Enabled' : 'Disabled'}
                      variant={task.enabled ? 'success' : 'default'}
                      size="sm"
                    />
                  </View>
                  <View style={styles.schedTaskMeta}>
                    <Text style={styles.schedMetaText}>Interval: {task.interval_seconds}s</Text>
                    <Text style={styles.schedMetaText}>Runs: {task.run_count}</Text>
                    <Text style={styles.schedMetaText}>Errors: {task.error_count}</Text>
                  </View>
                </View>
              ))}
            </Card>
          ) : (
            <EmptyState icon="clock-outline" title="No Scheduled Tasks" message="Scheduled agent tasks will appear here." />
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },

  // Tab bar
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
    gap: SPACING.xs,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.lg,
  },
  tabActive: {
    backgroundColor: '#eef2ff',
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: FONT_WEIGHT.medium,
    color: '#94a3b8',
  },
  tabLabelActive: {
    color: '#6366f1',
    fontWeight: FONT_WEIGHT.semibold,
  },

  // Chat
  chatContainer: { flex: 1 },
  sessionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.sm,
    backgroundColor: '#fefce8',
    borderBottomWidth: 1,
    borderBottomColor: '#fef08a',
  },
  controlPanelsRow: {
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    gap: SPACING.sm,
  },
  controlPanel: {
    width: 320,
  },
  controlInput: {
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: RADIUS.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    fontSize: FONT_SIZE.sm,
    color: '#0f172a',
    marginBottom: SPACING.xs,
  },
  controlAction: {
    alignSelf: 'flex-start',
    marginBottom: SPACING.sm,
  },
  controlActionRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    marginTop: SPACING.xs,
  },
  controlHint: {
    fontSize: FONT_SIZE.xs,
    color: '#94a3b8',
    lineHeight: 16,
  },
  eventRow: {
    paddingVertical: SPACING.xs,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
    gap: 4,
  },
  eventHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  eventTime: {
    fontSize: 10,
    color: '#94a3b8',
  },
  eventDetail: {
    fontSize: 10,
    color: '#475569',
    lineHeight: 14,
  },
  sessionId: {
    fontSize: 10,
    color: '#92400e',
    fontFamily: 'monospace',
    fontWeight: FONT_WEIGHT.medium,
  },
  tierBadge: {
    borderRadius: RADIUS.full,
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  tierText: {
    fontSize: 10,
    fontWeight: FONT_WEIGHT.bold,
  },
  msgList: {
    padding: SPACING.lg,
    paddingBottom: SPACING['3xl'],
    gap: SPACING.sm,
  },
  userMsgRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  systemMsgRow: {
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.md,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  systemMsgText: {
    fontSize: FONT_SIZE.xs,
    color: '#475569',
    maxWidth: 260,
  },
  userBubble: {
    maxWidth: '80%',
    borderRadius: RADIUS.xl,
    borderBottomRightRadius: RADIUS.sm,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    ...SHADOWS.md,
  },
  userMsgText: {
    fontSize: FONT_SIZE.base,
    color: '#ffffff',
    lineHeight: 20,
  },
  assistantMsgRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
  agentAvatar: {
    width: 28,
    height: 28,
    borderRadius: RADIUS.md,
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 2,
  },
  assistantBubble: {
    flex: 1,
    maxWidth: '85%',
    backgroundColor: '#ffffff',
    borderRadius: RADIUS.xl,
    borderTopLeftRadius: RADIUS.sm,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderWidth: 1,
    borderColor: '#f1f5f9',
    ...SHADOWS.sm,
  },
  assistantMsgText: {
    fontSize: FONT_SIZE.base,
    color: '#1e293b',
    lineHeight: 22,
  },
  thinkingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    paddingVertical: SPACING.xs,
  },
  thinkingText: {
    fontSize: FONT_SIZE.sm,
    color: '#94a3b8',
  },
  msgMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    marginTop: SPACING.sm,
    paddingTop: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: '#f8fafc',
  },
  turnCount: {
    fontSize: 10,
    color: '#94a3b8',
    fontWeight: FONT_WEIGHT.medium,
  },

  // Input
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    backgroundColor: '#ffffff',
    borderTopWidth: 1,
    borderTopColor: '#f1f5f9',
  },
  chatInput: {
    flex: 1,
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS.xl,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    fontSize: FONT_SIZE.base,
    color: '#0f172a',
  },
  sendBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    overflow: 'hidden',
  },
  sendBtnGrad: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },

  // Scrollable tabs
  scrollView: { flex: 1 },
  scrollContent: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.md,
  },
  metricsGrid: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
  metricHalf: { flex: 1 },

  // Sessions
  sessionCard: { marginBottom: 0 },
  sessionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.xs,
  },
  sessionIdText: {
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    fontFamily: 'monospace',
    flex: 1,
    marginRight: SPACING.sm,
  },
  sessionAgent: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
  },
  sessionTime: {
    fontSize: 10,
    color: '#94a3b8',
    marginTop: 2,
  },

  // Configs
  configCard: { marginBottom: 0 },
  configHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  configId: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },
  toolChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginBottom: SPACING.sm,
  },
  configMeta: {
    flexDirection: 'row',
    gap: SPACING.lg,
  },
  configMetaText: {
    fontSize: 10,
    color: '#94a3b8',
  },

  // Scheduler
  schedTask: {
    paddingVertical: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  schedTaskHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.xs,
  },
  schedTaskId: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
    flex: 1,
  },
  schedTaskMeta: {
    flexDirection: 'row',
    gap: SPACING.md,
  },
  schedMetaText: {
    fontSize: 10,
    color: '#64748b',
  },
});
