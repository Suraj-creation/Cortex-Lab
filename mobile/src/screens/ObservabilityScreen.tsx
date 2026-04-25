/**
 * ObservabilityScreen — Cortex Aurora Pipeline Observability + Runtime Operations
 * Full metrics dashboard, pipeline traces, runtime tasks, and agent events
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  FlatList,
  Alert,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { SectionHeader } from '../components/ui/SectionHeader';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { EmptyState } from '../components/ui/EmptyState';
import type { LivePipelineEvent, RAGStats, PipelineTrace, TracesResponse } from '../../shared/core/types';

type ObsTab = 'metrics' | 'traces' | 'events' | 'runtime';

interface ObservabilityScreenProps {
  observabilityMetrics: Record<string, unknown> | null;
  pipelineEvents: LivePipelineEvent[];
  loadingView: boolean;
  apiBaseUrl: string;
  api: any;
}

function compact(value: unknown, maxLen: number = 200): string {
  if (typeof value === 'string') {
    return value.length > maxLen ? `${value.slice(0, maxLen)}...` : value;
  }
  try {
    const text = JSON.stringify(value);
    return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
  } catch {
    return 'No details';
  }
}

export function ObservabilityScreen({
  observabilityMetrics,
  pipelineEvents,
  loadingView,
  apiBaseUrl,
  api,
}: ObservabilityScreenProps) {
  const [activeTab, setActiveTab] = useState<ObsTab>('metrics');
  const [traces, setTraces] = useState<PipelineTrace[]>([]);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [selectedTrace, setSelectedTrace] = useState<PipelineTrace | null>(null);

  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [runtimeHealth, setRuntimeHealth] = useState<Record<string, any> | null>(null);
  const [runtimePermissions, setRuntimePermissions] = useState<any[]>([]);
  const [runtimeTasks, setRuntimeTasks] = useState<any[]>([]);
  const [runtimeAudit, setRuntimeAudit] = useState<any[]>([]);
  const [runtimeExecutor, setRuntimeExecutor] = useState<Record<string, any> | null>(null);
  const [runtimeContracts, setRuntimeContracts] = useState<Record<string, any> | null>(null);
  const [runtimeInterfaces, setRuntimeInterfaces] = useState<Record<string, any> | null>(null);
  const [memoryQualityHistory, setMemoryQualityHistory] = useState<any[]>([]);
  const [memoryEvalBusy, setMemoryEvalBusy] = useState(false);

  const loadTraces = useCallback(async () => {
    setTracesLoading(true);
    try {
      const data: TracesResponse = await api.getPipelineTraces?.(30);
      setTraces(data?.traces || []);
    } catch {}
    setTracesLoading(false);
  }, [api]);

  const loadTraceDetail = useCallback(async (traceId: string) => {
    try {
      const detail: PipelineTrace = await api.getPipelineTraceById?.(traceId);
      setSelectedTrace(detail);
    } catch (e) {
      Alert.alert('Error', 'Failed to load trace details');
    }
  }, [api]);

  const loadRuntimeCenter = useCallback(async () => {
    setRuntimeLoading(true);
    try {
      const [
        healthRes,
        permsRes,
        tasksRes,
        auditRes,
        executorRes,
        contractsRes,
        interfacesRes,
        memoryHistoryRes,
      ] = await Promise.allSettled([
        api.getRuntimeHealth?.(),
        api.getRuntimeSafetyPermissions?.(),
        api.getRuntimeTasks?.(),
        api.getRuntimeSafetyAudit?.(80),
        api.getRuntimeSafetyExecutorStatus?.(),
        api.getRuntimeToolContracts?.(),
        api.getRuntimeInterfaces?.(),
        api.getMemoryQualityHistory?.(20),
      ]);

      setRuntimeHealth(healthRes.status === 'fulfilled' ? (healthRes.value || null) : null);
      setRuntimePermissions(
        permsRes.status === 'fulfilled' && Array.isArray(permsRes.value?.pending)
          ? permsRes.value.pending
          : [],
      );
      setRuntimeTasks(
        tasksRes.status === 'fulfilled' && Array.isArray(tasksRes.value?.tasks)
          ? tasksRes.value.tasks
          : [],
      );
      setRuntimeAudit(
        auditRes.status === 'fulfilled' && Array.isArray(auditRes.value?.events)
          ? auditRes.value.events
          : [],
      );
      setRuntimeExecutor(executorRes.status === 'fulfilled' ? (executorRes.value || null) : null);
      setRuntimeContracts(contractsRes.status === 'fulfilled' ? (contractsRes.value || null) : null);
      setRuntimeInterfaces(interfacesRes.status === 'fulfilled' ? (interfacesRes.value || null) : null);
      setMemoryQualityHistory(
        memoryHistoryRes.status === 'fulfilled' && Array.isArray(memoryHistoryRes.value?.history)
          ? memoryHistoryRes.value.history
          : [],
      );
    } catch {
      setRuntimeHealth(null);
      setRuntimePermissions([]);
      setRuntimeTasks([]);
      setRuntimeAudit([]);
      setRuntimeExecutor(null);
      setRuntimeContracts(null);
      setRuntimeInterfaces(null);
      setMemoryQualityHistory([]);
    }
    setRuntimeLoading(false);
  }, [api]);

  const resolvePermission = useCallback(
    async (permissionId: string, approve: boolean) => {
      try {
        await api.resolveRuntimeSafetyPermission?.(
          permissionId,
          approve,
          'mobile-operator',
          approve ? 'Approved from runtime operations center' : 'Denied from runtime operations center',
        );
        await loadRuntimeCenter();
      } catch {
        Alert.alert('Action Failed', 'Unable to resolve runtime permission request.');
      }
    },
    [api, loadRuntimeCenter],
  );

  const cancelTask = useCallback(
    async (taskId: string) => {
      try {
        await api.cancelRuntimeTask?.(taskId, 'Cancelled from mobile runtime operations center', true);
        await loadRuntimeCenter();
      } catch {
        Alert.alert('Action Failed', 'Unable to cancel runtime task.');
      }
    },
    [api, loadRuntimeCenter],
  );

  const runMemoryQualityEval = useCallback(async () => {
    setMemoryEvalBusy(true);
    try {
      await api.evaluateMemoryQuality?.({ topK: 5 });
      await loadRuntimeCenter();
    } catch {
      Alert.alert('Evaluation Failed', 'Unable to run memory quality evaluation.');
    }
    setMemoryEvalBusy(false);
  }, [api, loadRuntimeCenter]);

  const metrics = observabilityMetrics;

  const tabs: { key: ObsTab; label: string; icon: string }[] = [
    { key: 'metrics', label: 'Metrics', icon: 'chart-bar' },
    { key: 'traces', label: 'Traces', icon: 'timeline-clock-outline' },
    { key: 'events', label: 'Live Events', icon: 'lightning-bolt' },
    { key: 'runtime', label: 'Runtime', icon: 'shield-check-outline' },
  ];

  // Helpers for extracting nested RAG stats
  const metricsAny = metrics as Record<string, any> | null;

  return (
    <View style={styles.container}>
      {/* Tab bar */}
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => {
              setActiveTab(tab.key);
              if (tab.key === 'traces' && traces.length === 0) loadTraces();
              if (tab.key === 'runtime' && !runtimeHealth) void loadRuntimeCenter();
            }}
            activeOpacity={0.7}
          >
            <AppIcon
              name={tab.icon as any}
              size={16}
              color={activeTab === tab.key ? '#6366f1' : '#94a3b8'}
            />
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {activeTab === 'metrics' && (
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {loadingView && !metrics ? (
            <LoadingSpinner message="Loading metrics..." />
          ) : !metrics ? (
            <EmptyState icon="chart-timeline-variant" title="No Metrics" message="Metrics will appear when the backend is connected." />
          ) : (
            <>
              {/* Primary metrics grid */}
              <View style={styles.metricsGrid}>
                <MetricCard
                  label="Total Requests"
                  value={String(metricsAny?.total_requests ?? metricsAny?.request_count ?? 0)}
                  tone="indigo"
                  compact
                  style={styles.metricHalf}
                />
                <MetricCard
                  label="Avg Latency"
                  value={metricsAny?.avg_latency_ms ? `${Number(metricsAny.avg_latency_ms).toFixed(0)}ms` : 'N/A'}
                  tone="blue"
                  compact
                  style={styles.metricHalf}
                />
              </View>

              <View style={styles.metricsGrid}>
                <MetricCard
                  label="Cache Hit Rate"
                  value={metricsAny?.cache_hit_rate ? `${(Number(metricsAny.cache_hit_rate) * 100).toFixed(1)}%` : 'N/A'}
                  tone="emerald"
                  compact
                  style={styles.metricHalf}
                />
                <MetricCard
                  label="Error Rate"
                  value={metricsAny?.error_rate ? `${(Number(metricsAny.error_rate) * 100).toFixed(1)}%` : '0%'}
                  tone="rose"
                  compact
                  style={styles.metricHalf}
                />
              </View>

              {/* Pipeline stage breakdown */}
              <Card variant="outlined" padding="lg" style={styles.section}>
                <SectionHeader title="Pipeline Stages" icon={<AppIcon name="pipe" size={16} color="#6366f1" />} />
                {Array.isArray(metricsAny?.stages) && metricsAny.stages.length > 0 ? (
                  metricsAny.stages.map((stage: any, i: number) => (
                    <View key={i} style={styles.stageRow}>
                      <View style={[styles.stageDot, { backgroundColor: stage.status === 'healthy' ? '#10b981' : '#f59e0b' }]} />
                      <Text style={styles.stageName}>{stage.name || stage.stage || `Stage ${i + 1}`}</Text>
                      <Badge label={stage.status || 'active'} variant={stage.status === 'healthy' ? 'success' : 'warning'} size="sm" />
                    </View>
                  ))
                ) : (
                  <Text style={styles.noData}>Pipeline stages will appear during active processing</Text>
                )}
              </Card>

              {/* All metrics dump */}
              <Card variant="outlined" padding="lg" style={styles.section}>
                <SectionHeader title="All Metrics" />
                {metricsAny && Object.entries(metricsAny).slice(0, 20).map(([key, val]) => {
                  if (typeof val === 'object') return null;
                  return (
                    <View key={key} style={styles.metricRow}>
                      <Text style={styles.metricKey}>{key.replace(/_/g, ' ')}</Text>
                      <Text style={styles.metricVal}>{String(val)}</Text>
                    </View>
                  );
                })}
              </Card>
            </>
          )}
        </ScrollView>
      )}

      {activeTab === 'traces' && (
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl refreshing={tracesLoading} onRefresh={loadTraces} tintColor="#6366f1" colors={['#6366f1']} />
          }
        >
          {tracesLoading && traces.length === 0 ? (
            <LoadingSpinner message="Loading traces..." />
          ) : traces.length === 0 ? (
            <EmptyState icon="timeline-clock-outline" title="No Traces" message="Pipeline traces will appear after queries are processed." />
          ) : (
            traces.slice(0, 50).map((trace, i) => (
              <TouchableOpacity
                key={trace.trace_id || String(i)}
                onPress={() => loadTraceDetail(trace.trace_id)}
                activeOpacity={0.7}
              >
                <Card variant="default" padding="md" style={styles.traceCard}>
                  <View style={styles.traceHeader}>
                    <Badge
                      label={trace.routing_decision || 'standard'}
                      variant="primary"
                      size="sm"
                    />
                    <Text style={styles.traceTime}>
                      {trace.timestamp ? new Date(trace.timestamp).toLocaleTimeString() : ''}
                    </Text>
                  </View>
                  <Text style={styles.traceQuery} numberOfLines={2}>
                    {trace.query || 'No query'}
                  </Text>
                  <View style={styles.traceMetaRow}>
                    <Text style={styles.traceMeta}>⏱ {trace.total_duration_ms}ms</Text>
                    <Text style={styles.traceMeta}>
                      {trace.agents_invoked?.length ?? 0} agent{(trace.agents_invoked?.length ?? 0) !== 1 ? 's' : ''}
                    </Text>
                    <Text style={styles.traceMeta}>
                      {trace.evidence_count} evidence
                    </Text>
                    <Text style={styles.traceMeta}>
                      {(trace.final_confidence * 100).toFixed(0)}%
                    </Text>
                  </View>
                </Card>
              </TouchableOpacity>
            ))
          )}
        </ScrollView>
      )}

      {activeTab === 'events' && (
        <FlatList
          data={pipelineEvents}
          keyExtractor={(_, i) => `event-${i}`}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <EmptyState icon="lightning-bolt" title="No Live Events" message="Pipeline events will stream here during active processing." />
          }
          renderItem={({ item, index }) => (
            <Card variant="default" padding="sm" style={styles.eventCard}>
              <View style={styles.eventHeader}>
                <Badge
                  label={item.event_type}
                  variant={
                    item.status === 'error' ? 'error' :
                    item.status === 'completed' ? 'success' :
                    item.status === 'running' ? 'primary' : 'default'
                  }
                  size="sm"
                />
                <Text style={styles.eventTime}>
                  {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : `#${pipelineEvents.length - index}`}
                </Text>
              </View>
              <View style={styles.eventBody}>
                <Text style={styles.eventStep}>{item.step_name}</Text>
                <Text style={styles.eventType}>{item.step_type}</Text>
              </View>
              {item.duration_ms > 0 && (
                <Text style={styles.eventDuration}>{item.duration_ms}ms</Text>
              )}
              {item.details && Object.keys(item.details).length > 0 && (
                <Text style={styles.eventContent} numberOfLines={2}>
                  {JSON.stringify(item.details).slice(0, 200)}
                </Text>
              )}
            </Card>
          )}
        />
      )}

      {activeTab === 'runtime' && (
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          refreshControl={
            <RefreshControl
              refreshing={runtimeLoading}
              onRefresh={() => void loadRuntimeCenter()}
              tintColor="#6366f1"
              colors={['#6366f1']}
            />
          }
        >
          <SectionHeader
            title="Runtime Operations Center"
            subtitle="Approvals, task controls, governance audit, and quality signals"
            action={{ label: 'Refresh', onPress: () => void loadRuntimeCenter() }}
          />

          <View style={styles.metricsGrid}>
            <MetricCard
              label="Pending Approvals"
              value={String(runtimePermissions.length)}
              tone={runtimePermissions.length > 0 ? 'amber' : 'emerald'}
              compact
              style={styles.metricHalf}
            />
            <MetricCard
              label="Runtime Tasks"
              value={String(runtimeTasks.length)}
              tone="indigo"
              compact
              style={styles.metricHalf}
            />
          </View>

          <View style={styles.metricsGrid}>
            <MetricCard
              label="Executor"
              value={runtimeExecutor?.running ? 'Running' : 'Idle'}
              tone={runtimeExecutor?.running ? 'emerald' : 'default'}
              compact
              style={styles.metricHalf}
            />
            <MetricCard
              label="Audit Events"
              value={String(runtimeAudit.length)}
              tone="violet"
              compact
              style={styles.metricHalf}
            />
          </View>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Approval Queue" />
            {runtimePermissions.length === 0 ? (
              <Text style={styles.noData}>No pending permissions. Risky operations are currently clear.</Text>
            ) : (
              runtimePermissions.slice(0, 12).map((permission, index) => (
                <View key={permission.permission_id || index} style={styles.runtimeRow}>
                  <View style={styles.runtimeRowTop}>
                    <Text style={styles.runtimeTitle}>{String(permission.tool_name || 'tool')}</Text>
                    <Badge label={String(permission.status || 'pending')} variant="warning" size="sm" />
                  </View>
                  <Text style={styles.runtimeMeta}>{String(permission.reason || permission.command_text || 'No reason provided')}</Text>
                  <Text style={styles.runtimeCode}>{String(permission.command_text || '')}</Text>
                  <View style={styles.runtimeActions}>
                    <Button
                      label="Approve"
                      size="xs"
                      variant="success"
                      onPress={() => void resolvePermission(String(permission.permission_id || ''), true)}
                    />
                    <Button
                      label="Deny"
                      size="xs"
                      variant="error"
                      onPress={() => void resolvePermission(String(permission.permission_id || ''), false)}
                    />
                  </View>
                </View>
              ))
            )}
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Runtime Tasks" />
            {runtimeTasks.length === 0 ? (
              <Text style={styles.noData}>No runtime tasks currently tracked.</Text>
            ) : (
              runtimeTasks.slice(0, 20).map((task, index) => {
                const state = String(task.state || 'unknown');
                const variant =
                  state === 'completed' ? 'success' :
                  state === 'failed' || state === 'cancelled' ? 'error' :
                  state === 'running' ? 'primary' :
                  state === 'waiting_approval' ? 'warning' : 'default';

                return (
                  <View key={task.task_id || index} style={styles.runtimeRow}>
                    <View style={styles.runtimeRowTop}>
                      <Text style={styles.runtimeTitle}>{String(task.task_id || 'task')}</Text>
                      <Badge label={state} variant={variant as any} size="sm" />
                    </View>
                    <Text style={styles.runtimeMeta}>Updated: {String(task.updated_at || task.created_at || 'n/a')}</Text>
                    {Array.isArray(task.permission_scope) && task.permission_scope.length > 0 && (
                      <Text style={styles.runtimeMeta}>Scope: {task.permission_scope.join(', ')}</Text>
                    )}
                    {(state === 'queued' || state === 'running' || state === 'waiting_approval' || state === 'blocked') && (
                      <View style={styles.runtimeActions}>
                        <Button
                          label="Cancel Task"
                          size="xs"
                          variant="error"
                          onPress={() => void cancelTask(String(task.task_id || ''))}
                        />
                      </View>
                    )}
                  </View>
                );
              })
            )}
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Safety Audit" />
            {runtimeAudit.length === 0 ? (
              <Text style={styles.noData}>No recent audit events found.</Text>
            ) : (
              runtimeAudit.slice(0, 15).map((event, index) => (
                <View key={index} style={styles.auditRow}>
                  <Badge label={String(event.effect || event.decision || 'event')} variant="violet" size="sm" />
                  <Text style={styles.auditText} numberOfLines={3}>{compact(event, 220)}</Text>
                </View>
              ))
            )}
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Interfaces & Contracts" />
            <View style={styles.runtimeKvRow}>
              <Text style={styles.runtimeKvKey}>Tool Contracts</Text>
              <Text style={styles.runtimeKvValue}>{String(runtimeContracts?.count ?? 0)}</Text>
            </View>
            <View style={styles.runtimeKvRow}>
              <Text style={styles.runtimeKvKey}>Interface Snapshot</Text>
              <Text style={styles.runtimeKvValue}>{runtimeInterfaces ? 'Loaded' : 'Unavailable'}</Text>
            </View>
            {runtimeHealth && (
              <View style={styles.runtimePreviewBox}>
                <Text style={styles.runtimePreviewText}>{compact(runtimeHealth, 320)}</Text>
              </View>
            )}
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader
              title="Memory Quality"
              action={{ label: memoryEvalBusy ? 'Running...' : 'Evaluate', onPress: () => void runMemoryQualityEval() }}
            />
            {memoryQualityHistory.length === 0 ? (
              <Text style={styles.noData}>No memory quality snapshots available yet.</Text>
            ) : (
              memoryQualityHistory.slice(0, 10).map((item, index) => (
                <View key={index} style={styles.runtimeRow}>
                  <View style={styles.runtimeRowTop}>
                    <Text style={styles.runtimeTitle}>{String(item.source || 'snapshot')}</Text>
                    <Text style={styles.runtimeTime}>{String(item.timestamp || item.created_at || '')}</Text>
                  </View>
                  <Text style={styles.runtimeMeta}>
                    P@k: {Number(item.avg_precision_at_k || 0).toFixed(2)} • Recall: {Number(item.recall_proxy_rate || 0).toFixed(2)} • Extraction hit: {Number(item.extraction_hit_rate || 0).toFixed(2)}
                  </Text>
                </View>
              ))
            )}
          </Card>
        </ScrollView>
      )}

      {/* Trace detail modal */}
      {selectedTrace && (
        <View style={styles.traceOverlay}>
          <View style={styles.traceDetailSheet}>
            <View style={styles.traceDetailHeader}>
              <View>
                <Text style={styles.traceDetailTitle}>Trace Details</Text>
                <Text style={styles.traceDetailId}>{selectedTrace.trace_id}</Text>
              </View>
              <TouchableOpacity onPress={() => setSelectedTrace(null)} style={styles.traceCloseBtn}>
                <AppIcon name="close" size={18} color="#64748b" />
              </TouchableOpacity>
            </View>
            <ScrollView style={styles.traceDetailContent} showsVerticalScrollIndicator={false}>
              {/* Query */}
              <SectionHeader title="Query" />
              <Text style={styles.traceDetailQuery}>{selectedTrace.query}</Text>

              {/* Quick stats */}
              <View style={styles.traceStatsRow}>
                <View style={styles.traceStat}>
                  <Text style={styles.traceStatVal}>{selectedTrace.total_duration_ms}ms</Text>
                  <Text style={styles.traceStatLabel}>Duration</Text>
                </View>
                <View style={styles.traceStat}>
                  <Text style={styles.traceStatVal}>{(selectedTrace.final_confidence * 100).toFixed(0)}%</Text>
                  <Text style={styles.traceStatLabel}>Confidence</Text>
                </View>
                <View style={styles.traceStat}>
                  <Text style={styles.traceStatVal}>{selectedTrace.evidence_count}</Text>
                  <Text style={styles.traceStatLabel}>Evidence</Text>
                </View>
              </View>

              {/* Steps */}
              <SectionHeader title="Pipeline Steps" subtitle={`${selectedTrace.steps.length} steps`} />
              {selectedTrace.steps.map((step, i) => (
                <View key={i} style={styles.stepRow}>
                  <View style={[styles.stepDot, {
                    backgroundColor: step.status === 'completed' ? '#10b981' : step.status === 'skipped' ? '#94a3b8' : '#f43f5e'
                  }]} />
                  <View style={styles.stepContent}>
                    <Text style={styles.stepName}>{step.step_name}</Text>
                    <Text style={styles.stepMeta}>{step.step_type} · {step.duration_ms}ms · {step.status}</Text>
                  </View>
                </View>
              ))}

              {/* Retrieval channels */}
              {selectedTrace.retrieval_channels.length > 0 && (
                <>
                  <SectionHeader title="Retrieval Channels" />
                  {selectedTrace.retrieval_channels.map((ch, i) => (
                    <View key={i} style={styles.channelRow}>
                      <Text style={styles.channelName}>{ch.channel}</Text>
                      <View style={styles.channelStats}>
                        <Badge label={`${ch.result_count} results`} variant="primary" size="sm" />
                        <Badge label={`${ch.top_score.toFixed(2)} top`} variant="success" size="sm" />
                        <Text style={styles.channelDuration}>{ch.duration_ms}ms</Text>
                      </View>
                    </View>
                  ))}
                </>
              )}

              {/* Agents invoked */}
              {selectedTrace.agents_invoked.length > 0 && (
                <>
                  <SectionHeader title="Agents Invoked" />
                  <View style={styles.agentChips}>
                    {selectedTrace.agents_invoked.map((a, i) => (
                      <Badge key={i} label={`${a.agent}${a.is_primary ? ' ★' : ''}`} variant={a.is_primary ? 'violet' : 'default'} size="sm" />
                    ))}
                  </View>
                </>
              )}

              {/* Cache status */}
              <SectionHeader title="Cache" />
              <View style={styles.cacheRow}>
                <Badge
                  label={selectedTrace.cache_status.hit ? 'Cache Hit' : 'Cache Miss'}
                  variant={selectedTrace.cache_status.hit ? 'success' : 'default'}
                  dot
                  size="md"
                />
                {selectedTrace.cache_status.level && (
                  <Text style={styles.cacheLevel}>Level: {selectedTrace.cache_status.level}</Text>
                )}
              </View>

              {/* Raw JSON fallback */}
              <SectionHeader title="Raw Trace" />
              <Text style={styles.traceDetailJson}>
                {JSON.stringify(selectedTrace, null, 2)}
              </Text>
            </ScrollView>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
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
  tabActive: {
    backgroundColor: '#eef2ff',
  },
  tabLabel: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.medium,
    color: '#94a3b8',
  },
  tabLabelActive: {
    color: '#6366f1',
    fontWeight: FONT_WEIGHT.semibold,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.md,
  },
  metricsGrid: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
  metricHalf: {
    flex: 1,
  },
  section: {
    marginTop: SPACING.xs,
  },
  stageRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  stageDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  stageName: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    fontWeight: FONT_WEIGHT.medium,
    textTransform: 'capitalize',
  },
  noData: {
    fontSize: FONT_SIZE.sm,
    color: '#94a3b8',
    paddingVertical: SPACING.md,
    textAlign: 'center',
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: SPACING.xs,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  metricKey: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
    flex: 1,
    textTransform: 'capitalize',
  },
  metricVal: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },

  // Traces
  traceCard: { marginBottom: 0 },
  traceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.xs,
  },
  traceTime: { fontSize: 10, color: '#94a3b8' },
  traceQuery: {
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    lineHeight: 18,
  },
  traceMetaRow: {
    flexDirection: 'row',
    gap: SPACING.md,
    marginTop: SPACING.sm,
  },
  traceMeta: {
    fontSize: 10,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.medium,
  },

  // Events
  eventCard: { marginBottom: 0 },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.xs,
  },
  eventTime: { fontSize: 10, color: '#94a3b8' },
  eventBody: {
    flexDirection: 'row',
    gap: SPACING.sm,
    alignItems: 'center',
  },
  eventStep: {
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    fontWeight: FONT_WEIGHT.semibold,
  },
  eventType: {
    fontSize: FONT_SIZE.xs,
    color: '#94a3b8',
  },
  eventDuration: {
    fontSize: 10,
    color: '#64748b',
    marginTop: 2,
  },
  eventContent: {
    fontSize: FONT_SIZE.xs,
    color: '#475569',
    lineHeight: 16,
    marginTop: SPACING.xs,
  },

  // Trace detail overlay
  traceOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(15, 23, 42, 0.4)',
    justifyContent: 'flex-end',
  },
  traceDetailSheet: {
    backgroundColor: '#ffffff',
    borderTopLeftRadius: RADIUS['3xl'],
    borderTopRightRadius: RADIUS['3xl'],
    maxHeight: '85%',
    ...SHADOWS.xl,
  },
  traceDetailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: SPACING.xl,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  traceDetailTitle: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
  },
  traceDetailId: {
    fontSize: 10,
    color: '#94a3b8',
    fontFamily: 'monospace',
    marginTop: 2,
  },
  traceCloseBtn: {
    padding: SPACING.sm,
    backgroundColor: '#f1f5f9',
    borderRadius: RADIUS.lg,
  },
  traceDetailContent: {
    padding: SPACING.xl,
  },
  traceDetailQuery: {
    fontSize: FONT_SIZE.base,
    color: '#334155',
    lineHeight: 22,
    marginBottom: SPACING.lg,
  },
  traceStatsRow: {
    flexDirection: 'row',
    marginBottom: SPACING.lg,
  },
  traceStat: {
    flex: 1,
    alignItems: 'center',
  },
  traceStatVal: {
    fontSize: FONT_SIZE.xl,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
  },
  traceStatLabel: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    marginTop: 2,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  stepDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 5,
  },
  stepContent: { flex: 1 },
  stepName: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
  },
  stepMeta: {
    fontSize: 10,
    color: '#94a3b8',
    marginTop: 1,
  },
  channelRow: {
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  channelName: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
    marginBottom: SPACING.xs,
    textTransform: 'capitalize',
  },
  channelStats: {
    flexDirection: 'row',
    gap: SPACING.sm,
    alignItems: 'center',
  },
  channelDuration: {
    fontSize: 10,
    color: '#94a3b8',
  },
  agentChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
    marginBottom: SPACING.lg,
  },
  cacheRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
    marginBottom: SPACING.lg,
  },
  cacheLevel: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
  },
  traceDetailJson: {
    fontSize: 10,
    color: '#475569',
    fontFamily: 'monospace',
    lineHeight: 14,
  },

  // Runtime center
  runtimeRow: {
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
    gap: 4,
  },
  runtimeRowTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  runtimeTitle: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
  },
  runtimeMeta: {
    fontSize: 10,
    color: '#64748b',
    lineHeight: 15,
  },
  runtimeCode: {
    fontSize: 10,
    color: '#475569',
    fontFamily: 'monospace',
    lineHeight: 14,
  },
  runtimeActions: {
    flexDirection: 'row',
    gap: SPACING.sm,
    marginTop: SPACING.xs,
  },
  auditRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    paddingVertical: SPACING.xs,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  auditText: {
    flex: 1,
    fontSize: FONT_SIZE.xs,
    color: '#475569',
    lineHeight: 16,
  },
  runtimeKvRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  runtimeKvKey: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
  },
  runtimeKvValue: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },
  runtimePreviewBox: {
    marginTop: SPACING.md,
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: RADIUS.lg,
    padding: SPACING.sm,
  },
  runtimePreviewText: {
    fontSize: 10,
    color: '#475569',
    lineHeight: 14,
    fontFamily: 'monospace',
  },
  runtimeTime: {
    fontSize: 10,
    color: '#94a3b8',
  },
});
