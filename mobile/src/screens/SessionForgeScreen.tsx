/**
 * SessionForgeScreen — Deep Session Memory Forge control panel
 * Provides run controls, artifact browsing, and quick diagnostics.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  TextInput,
  FlatList,
  RefreshControl,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { SearchBar } from '../components/ui/SearchBar';
import { Button } from '../components/ui/Button';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { EmptyState } from '../components/ui/EmptyState';

type ForgeTab = 'overview' | 'artifacts';

interface SessionForgeScreenProps {
  api: any;
}

const ARTIFACT_TYPES = [
  'thought_objects',
  'decision_records',
  'open_loops',
  'gap_signals',
  'belief_evolution',
  'structured_summaries',
] as const;

function compact(value: unknown, maxLen: number = 220): string {
  if (typeof value === 'string') {
    return value.length > maxLen ? `${value.slice(0, maxLen)}...` : value;
  }
  try {
    const text = JSON.stringify(value);
    if (!text) return 'No details';
    return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
  } catch {
    return 'Unserializable artifact payload';
  }
}

function toInt(value: string, fallback: number): number {
  const parsed = parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return parsed;
}

export function SessionForgeScreen({ api }: SessionForgeScreenProps) {
  const [activeTab, setActiveTab] = useState<ForgeTab>('overview');
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [status, setStatus] = useState<Record<string, any> | null>(null);
  const [recentRuns, setRecentRuns] = useState<any[]>([]);

  const [actionBusy, setActionBusy] = useState(false);
  const [lastActionResult, setLastActionResult] = useState<Record<string, unknown> | null>(null);

  const [selectedArtifactType, setSelectedArtifactType] = useState<(typeof ARTIFACT_TYPES)[number]>('thought_objects');
  const [artifactSearch, setArtifactSearch] = useState('');
  const [artifactsLoading, setArtifactsLoading] = useState(false);
  const [artifacts, setArtifacts] = useState<any[]>([]);

  const [sessionIdDraft, setSessionIdDraft] = useState('');
  const [lookbackDaysDraft, setLookbackDaysDraft] = useState('7');
  const [windowDaysDraft, setWindowDaysDraft] = useState('14');

  const loadStatus = useCallback(async () => {
    setLoadingStatus(true);
    try {
      const res = await api.getSessionForgeStatus?.();
      setStatus((res?.status || null) as Record<string, any> | null);
      setRecentRuns(Array.isArray(res?.recent_runs) ? res.recent_runs : Array.isArray(res?.recentRuns) ? res.recentRuns : []);
    } catch {
      setStatus(null);
      setRecentRuns([]);
    }
    setLoadingStatus(false);
  }, [api]);

  const loadArtifacts = useCallback(async () => {
    setArtifactsLoading(true);
    try {
      const res = await api.getForgeArtifacts?.(selectedArtifactType);
      setArtifacts(Array.isArray(res?.items) ? res.items : []);
    } catch {
      setArtifacts([]);
    }
    setArtifactsLoading(false);
  }, [api, selectedArtifactType]);

  useEffect(() => {
    void loadStatus();
  }, [loadStatus]);

  useEffect(() => {
    if (activeTab === 'artifacts') {
      void loadArtifacts();
    }
  }, [activeTab, selectedArtifactType, loadArtifacts]);

  const runAction = useCallback(
    async (kind: 'crystallize' | 'summary' | 'gaps' | 'beliefs') => {
      setActionBusy(true);
      const sessionId = sessionIdDraft.trim();
      const lookbackDays = Math.max(1, Math.min(180, toInt(lookbackDaysDraft, 7)));
      const windowDays = Math.max(1, Math.min(180, toInt(windowDaysDraft, 14)));

      try {
        let result: Record<string, unknown> = {};
        if (kind === 'crystallize') {
          result = (await api.triggerCrystallize?.()) || {};
        } else if (kind === 'summary') {
          result =
            (await api.triggerSummaryForge?.({
              sessionId: sessionId || undefined,
              windowDays,
            })) || {};
        } else if (kind === 'gaps') {
          result =
            (await api.triggerGapMapper?.({
              sessionId: sessionId || undefined,
              lookbackDays,
            })) || {};
        } else {
          result =
            (await api.triggerBeliefDetector?.({
              sessionId: sessionId || undefined,
              lookbackDays,
            })) || {};
        }

        setLastActionResult(result);
        await loadStatus();
        if (activeTab === 'artifacts') {
          await loadArtifacts();
        }
      } catch {
        setLastActionResult({ status: 'error', message: `Failed to run ${kind}` });
      }

      setActionBusy(false);
    },
    [activeTab, api, loadArtifacts, loadStatus, lookbackDaysDraft, sessionIdDraft, windowDaysDraft],
  );

  const filteredArtifacts = useMemo(() => {
    const q = artifactSearch.trim().toLowerCase();
    if (!q) return artifacts;
    return artifacts.filter((item) => compact(item, 2000).toLowerCase().includes(q));
  }, [artifactSearch, artifacts]);

  const artifactStats = (status?.artifacts || {}) as Record<string, number>;

  const tabs: { key: ForgeTab; label: string; icon: string }[] = [
    { key: 'overview', label: 'Overview', icon: 'atom-variant' },
    { key: 'artifacts', label: 'Artifacts', icon: 'database-search' },
  ];

  return (
    <View style={styles.container}>
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => setActiveTab(tab.key)}
            activeOpacity={0.7}
          >
            <AppIcon name={tab.icon as any} size={16} color={activeTab === tab.key ? '#6366f1' : '#94a3b8'} />
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelActive]}>{tab.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {activeTab === 'overview' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          {loadingStatus && !status ? (
            <LoadingSpinner message="Loading Session Forge status..." />
          ) : (
            <>
              <SectionHeader
                title="Session Memory Forge"
                subtitle="Crystallize thoughts, decisions, loops, and belief shifts"
                action={{ label: 'Refresh', onPress: () => void loadStatus() }}
              />

              <View style={styles.metricsGrid}>
                <MetricCard
                  label="Thought Objects"
                  value={String(artifactStats.thought_objects || 0)}
                  tone="indigo"
                  compact
                  style={styles.metricHalf}
                />
                <MetricCard
                  label="Decision Records"
                  value={String(artifactStats.decision_records || 0)}
                  tone="violet"
                  compact
                  style={styles.metricHalf}
                />
              </View>

              <View style={styles.metricsGrid}>
                <MetricCard
                  label="Open Loops"
                  value={String(artifactStats.open_loops || 0)}
                  tone="amber"
                  compact
                  style={styles.metricHalf}
                />
                <MetricCard
                  label="Summary Artifacts"
                  value={String(artifactStats.structured_summaries || 0)}
                  tone="emerald"
                  compact
                  style={styles.metricHalf}
                />
              </View>

              <Card variant="outlined" padding="lg">
                <SectionHeader title="Run Controls" icon={<AppIcon name="play-circle-outline" size={16} color="#6366f1" />} />

                <Text style={styles.inputLabel}>Session ID (optional)</Text>
                <TextInput
                  value={sessionIdDraft}
                  onChangeText={setSessionIdDraft}
                  placeholder="session id or leave blank for auto"
                  placeholderTextColor="#94a3b8"
                  style={styles.textInput}
                />

                <View style={styles.inputRow}>
                  <View style={styles.inputCol}>
                    <Text style={styles.inputLabel}>Lookback Days</Text>
                    <TextInput
                      value={lookbackDaysDraft}
                      onChangeText={(value) => setLookbackDaysDraft(value.replace(/[^0-9]/g, ''))}
                      keyboardType="numeric"
                      style={styles.textInput}
                    />
                  </View>

                  <View style={styles.inputCol}>
                    <Text style={styles.inputLabel}>Summary Window</Text>
                    <TextInput
                      value={windowDaysDraft}
                      onChangeText={(value) => setWindowDaysDraft(value.replace(/[^0-9]/g, ''))}
                      keyboardType="numeric"
                      style={styles.textInput}
                    />
                  </View>
                </View>

                <View style={styles.actionGrid}>
                  <Button
                    label="Run Crystallizer"
                    size="sm"
                    onPress={() => void runAction('crystallize')}
                    loading={actionBusy}
                    style={styles.actionButton}
                  />
                  <Button
                    label="Forge Summary"
                    size="sm"
                    variant="secondary"
                    onPress={() => void runAction('summary')}
                    loading={actionBusy}
                    style={styles.actionButton}
                  />
                  <Button
                    label="Map Gaps"
                    size="sm"
                    variant="outline"
                    onPress={() => void runAction('gaps')}
                    loading={actionBusy}
                    style={styles.actionButton}
                  />
                  <Button
                    label="Detect Beliefs"
                    size="sm"
                    variant="outline"
                    onPress={() => void runAction('beliefs')}
                    loading={actionBusy}
                    style={styles.actionButton}
                  />
                </View>

                {lastActionResult && (
                  <View style={styles.lastRunBox}>
                    <Badge
                      label={String(lastActionResult.status || 'completed')}
                      variant={String(lastActionResult.status || '').toLowerCase() === 'error' ? 'error' : 'success'}
                      size="sm"
                    />
                    <Text style={styles.lastRunText}>{compact(lastActionResult)}</Text>
                  </View>
                )}
              </Card>

              <Card variant="default" padding="lg">
                <SectionHeader title="Recent Forge Runs" subtitle={`${recentRuns.length} recent`} />
                {recentRuns.length === 0 ? (
                  <Text style={styles.noData}>No recorded runs yet.</Text>
                ) : (
                  recentRuns.slice(0, 12).map((run, index) => (
                    <View key={index} style={styles.runRow}>
                      <View style={styles.runHead}>
                        <Text style={styles.runType}>{String(run.run_type || run.type || 'run')}</Text>
                        <Badge
                          label={String(run.status || 'done')}
                          variant={String(run.status || '').toLowerCase() === 'error' ? 'error' : 'primary'}
                          size="sm"
                        />
                      </View>
                      <Text style={styles.runMeta}>
                        {String(run.session_id || run.sessionId || 'multi-session')} • {String(run.created_at || run.timestamp || 'n/a')}
                      </Text>
                      <Text style={styles.runResult}>{compact(run.result || run)}</Text>
                    </View>
                  ))
                )}
              </Card>
            </>
          )}
        </ScrollView>
      )}

      {activeTab === 'artifacts' && (
        <FlatList
          data={filteredArtifacts}
          keyExtractor={(_, index) => `${selectedArtifactType}-${index}`}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={artifactsLoading}
              onRefresh={() => void loadArtifacts()}
              tintColor="#6366f1"
              colors={['#6366f1']}
            />
          }
          ListHeaderComponent={
            <View style={styles.artifactHeaderWrap}>
              <SectionHeader
                title="Forge Artifacts"
                subtitle={selectedArtifactType}
                action={{ label: 'Refresh', onPress: () => void loadArtifacts() }}
              />
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.typeChips}>
                {ARTIFACT_TYPES.map((type) => (
                  <TouchableOpacity
                    key={type}
                    style={[styles.typeChip, selectedArtifactType === type && styles.typeChipActive]}
                    onPress={() => setSelectedArtifactType(type)}
                    activeOpacity={0.7}
                  >
                    <Text style={[styles.typeChipText, selectedArtifactType === type && styles.typeChipTextActive]}>
                      {type}
                    </Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
              <SearchBar
                value={artifactSearch}
                onChangeText={setArtifactSearch}
                placeholder="Filter artifact content"
              />
            </View>
          }
          renderItem={({ item }) => (
            <Card variant="default" padding="md" style={styles.artifactCard}>
              <View style={styles.artifactTop}>
                <Badge label={selectedArtifactType} variant="violet" size="sm" />
                <Text style={styles.artifactTime}>{String(item.timestamp || item.created_at || item.createdAt || 'n/a')}</Text>
              </View>
              {item.source_session && <Text style={styles.artifactSession}>Session: {String(item.source_session)}</Text>}
              <Text style={styles.artifactBody}>{compact(item)}</Text>
            </Card>
          )}
          ListEmptyComponent={
            artifactsLoading ? (
              <LoadingSpinner message="Loading artifacts..." />
            ) : (
              <EmptyState
                icon="database-search"
                title="No Artifacts"
                message="Run Session Forge agents to populate this artifact stream."
              />
            )
          }
        />
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
    backgroundColor: '#f8fafc',
  },
  tabActive: { backgroundColor: '#eef2ff' },
  tabLabel: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.medium, color: '#94a3b8' },
  tabLabelActive: { color: '#6366f1', fontWeight: FONT_WEIGHT.semibold },

  metricsGrid: { flexDirection: 'row', gap: SPACING.sm },
  metricHalf: { flex: 1 },

  inputLabel: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.semibold,
    marginTop: SPACING.sm,
    marginBottom: 4,
  },
  textInput: {
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: RADIUS.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    fontSize: FONT_SIZE.sm,
    color: '#0f172a',
  },
  inputRow: { flexDirection: 'row', gap: SPACING.sm },
  inputCol: { flex: 1 },

  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
    marginTop: SPACING.md,
  },
  actionButton: {
    minWidth: 140,
  },

  lastRunBox: {
    marginTop: SPACING.md,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#ffffff',
    padding: SPACING.sm,
    gap: SPACING.xs,
  },
  lastRunText: {
    fontSize: FONT_SIZE.xs,
    color: '#475569',
    lineHeight: 16,
  },

  noData: {
    fontSize: FONT_SIZE.sm,
    color: '#94a3b8',
    textAlign: 'center',
    paddingVertical: SPACING.md,
  },

  runRow: {
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  runHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  runType: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
  },
  runMeta: {
    fontSize: 10,
    color: '#94a3b8',
    marginBottom: 4,
  },
  runResult: {
    fontSize: FONT_SIZE.xs,
    color: '#475569',
    lineHeight: 16,
  },

  artifactHeaderWrap: {
    gap: SPACING.sm,
  },
  typeChips: {
    gap: SPACING.sm,
    paddingBottom: SPACING.xs,
  },
  typeChip: {
    paddingHorizontal: SPACING.md,
    paddingVertical: 6,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#ffffff',
  },
  typeChipActive: {
    borderColor: '#c7d2fe',
    backgroundColor: '#eef2ff',
  },
  typeChipText: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.medium,
  },
  typeChipTextActive: {
    color: '#4338ca',
    fontWeight: FONT_WEIGHT.semibold,
  },

  artifactCard: { marginBottom: 0 },
  artifactTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.xs,
  },
  artifactTime: {
    fontSize: 10,
    color: '#94a3b8',
  },
  artifactSession: {
    fontSize: 10,
    color: '#64748b',
    marginBottom: SPACING.xs,
  },
  artifactBody: {
    fontSize: FONT_SIZE.xs,
    color: '#334155',
    lineHeight: 17,
  },
});
