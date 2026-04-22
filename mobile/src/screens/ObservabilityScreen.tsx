/**
 * ObservabilityScreen — Neural Dark pipeline + runtime operations center.
 */
import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge, type BadgeVariant } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { NeuralPulse } from '../components/ui/NeuralPulse';
import { AppIcon, type AppIconName } from '../components/ui/AppIcon';
import PipelineTracesList from '../components/PipelineTracesList';
import type { ApiClient } from '../../shared/core/api';
import type {
  LivePipelineEvent,
  RuntimeExecutorStatus,
  RuntimePermissionRequest,
  RuntimeTaskEvent,
  RuntimeTaskSnapshot,
} from '../../shared/core/types';

function shortNum(v: unknown): string {
  if (typeof v !== 'number') return String(v ?? '-');
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return Number.isInteger(v) ? `${v}` : v.toFixed(2);
}

function formatRelativeTime(ts: number | null): string {
  if (!ts) return 'never';
  const d = Date.now() - ts;
  if (d < 5000) return 'just now';
  if (d < 60000) return `${Math.round(d / 1000)}s ago`;
  if (d < 3600000) return `${Math.round(d / 60000)}m ago`;
  return `${Math.round(d / 3600000)}h ago`;
}

function formatIsoRelative(iso: string): string {
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return iso;
  return formatRelativeTime(ts);
}

const STATUS_CONFIG: Record<string, { color: string; pulse: boolean }> = {
  running: { color: NEURAL.primary, pulse: true },
  complete: { color: NEURAL.tertiary, pulse: false },
  completed: { color: NEURAL.tertiary, pulse: false },
  success: { color: NEURAL.tertiary, pulse: false },
  error: { color: NEURAL.error, pulse: false },
  failed: { color: NEURAL.error, pulse: false },
};

const TASK_VARIANT: Record<string, BadgeVariant> = {
  queued: 'info',
  running: 'primary',
  waiting_approval: 'warning',
  blocked: 'error',
  completed: 'success',
  failed: 'error',
  cancelled: 'ghost',
};

const PERMISSION_VARIANT: Record<string, BadgeVariant> = {
  pending: 'warning',
  approved: 'success',
  denied: 'error',
  expired: 'ghost',
};

interface ObservabilityScreenProps {
  observabilityMetrics: Record<string, unknown> | null;
  pipelineEvents: LivePipelineEvent[];
  loadingView: boolean;
  apiBaseUrl: string;
  api: ApiClient;
}

const METRIC_KEYS = [
  { key: 'total_queries', label: 'Queries', iconName: 'database-search-outline' as AppIconName },
  { key: 'cache_hits', label: 'Cache Hits', iconName: 'lightning-bolt-outline' as AppIconName },
  { key: 'total_embeddings', label: 'Embeddings', iconName: 'vector-link' as AppIconName },
  { key: 'total_memories', label: 'Memories', iconName: 'brain' as AppIconName },
  { key: 'avg_response_ms', label: 'Avg Latency', iconName: 'timer-outline' as AppIconName },
  { key: 'error_count', label: 'Errors', iconName: 'alert-circle-outline' as AppIconName },
];

function sortTasks(tasks: RuntimeTaskSnapshot[]): RuntimeTaskSnapshot[] {
  return [...tasks].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
}

export function ObservabilityScreen({
  observabilityMetrics,
  pipelineEvents,
  loadingView,
  apiBaseUrl,
  api,
}: ObservabilityScreenProps) {
  const [expandedMetrics, setExpandedMetrics] = useState(false);

  const [runtimePermissions, setRuntimePermissions] = useState<RuntimePermissionRequest[]>([]);
  const [runtimeExecutorStatus, setRuntimeExecutorStatus] = useState<RuntimeExecutorStatus | null>(null);
  const [runtimeTasks, setRuntimeTasks] = useState<RuntimeTaskSnapshot[]>([]);
  const [runtimeTaskEvents, setRuntimeTaskEvents] = useState<RuntimeTaskEvent[]>([]);

  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [runtimeError, setRuntimeError] = useState('');
  const [actionBusyId, setActionBusyId] = useState<string | null>(null);

  const loadRuntimeOps = useCallback(async (silent = false) => {
    if (!silent) setRuntimeBusy(true);
    try {
      const [permissionsResponse, executorResponse, tasksResponse] = await Promise.all([
        api.getRuntimeSafetyPermissions(),
        api.getRuntimeSafetyExecutorStatus(),
        api.getRuntimeTasks(),
      ]);
      setRuntimePermissions(permissionsResponse.pending || []);
      setRuntimeExecutorStatus(executorResponse);
      setRuntimeTasks(sortTasks(tasksResponse.tasks || []).slice(0, 40));
      setRuntimeError('');
    } catch (e) {
      setRuntimeError(e instanceof Error ? e.message : String(e));
    } finally {
      if (!silent) setRuntimeBusy(false);
    }
  }, [api]);

  const resolvePermission = useCallback(
    async (permissionId: string, approve: boolean) => {
      setActionBusyId(permissionId);
      try {
        await api.resolveRuntimeSafetyPermission(permissionId, approve);
        await loadRuntimeOps(true);
        setRuntimeError('');
      } catch (e) {
        setRuntimeError(e instanceof Error ? e.message : String(e));
      } finally {
        setActionBusyId(null);
      }
    },
    [api, loadRuntimeOps],
  );

  const cancelTask = useCallback(
    async (taskId: string) => {
      setActionBusyId(taskId);
      try {
        await api.cancelRuntimeTask(taskId);
        await loadRuntimeOps(true);
        setRuntimeError('');
      } catch (e) {
        setRuntimeError(e instanceof Error ? e.message : String(e));
      } finally {
        setActionBusyId(null);
      }
    },
    [api, loadRuntimeOps],
  );

  useEffect(() => {
    void loadRuntimeOps();

    if (Platform.OS !== 'web') {
      const interval = setInterval(() => void loadRuntimeOps(true), 4500);
      return () => clearInterval(interval);
    }

    const controller = api.subscribeRuntimeTaskEvents({
      onEvent: (event) => {
        setRuntimeTaskEvents((prev) => [event, ...prev].slice(0, 60));
        setRuntimeTasks((prev) => {
          const filtered = prev.filter((task) => task.task_id !== event.task.task_id);
          return sortTasks([event.task, ...filtered]).slice(0, 40);
        });
      },
      onError: (e) => {
        setRuntimeError(e.message);
      },
    });

    const interval = setInterval(() => void loadRuntimeOps(true), 7000);
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, [api, loadRuntimeOps]);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Text style={styles.title}>Pipeline Observability</Text>
          <View style={styles.liveBadge}>
            <NeuralPulse active size={6} color={NEURAL.tertiary} />
            <Text style={styles.liveText}>LIVE</Text>
          </View>
        </View>
        <Text style={styles.subtitle}>Updated {formatRelativeTime(pipelineEvents[0]?.timestamp ?? null)}</Text>

        {loadingView ? (
          <ActivityIndicator color={NEURAL.primary} size="large" style={styles.loader} />
        ) : observabilityMetrics ? (
          <>
            <View style={styles.metricGrid}>
              {METRIC_KEYS.map((mk) => {
                const val = observabilityMetrics[mk.key];
                return (
                  <Card key={mk.key} variant="elevated" style={styles.metricTile}>
                    <AppIcon name={mk.iconName} size={18} color={NEURAL.onSurfaceVariant} style={styles.metricTileIcon} />
                    <Text style={styles.metricTileValue}>{shortNum(val)}</Text>
                    <Text style={styles.metricTileLabel}>{mk.label}</Text>
                  </Card>
                );
              })}
            </View>

            <TouchableOpacity onPress={() => setExpandedMetrics((p) => !p)} style={styles.expandBtn}>
              <Text style={styles.expandText}>{expandedMetrics ? 'Hide all metrics' : 'Show all metrics'}</Text>
            </TouchableOpacity>

            {expandedMetrics && (
              <Card variant="outlined" style={styles.metricsCard}>
                {Object.entries(observabilityMetrics)
                  .slice(0, 20)
                  .map(([key, value]) => (
                    <View key={key} style={styles.metricRow}>
                      <Text style={styles.metricKey}>{key.replace(/_/g, ' ')}</Text>
                      <Text style={styles.metricVal} numberOfLines={1}>
                        {typeof value === 'object' ? JSON.stringify(value).slice(0, 30) : shortNum(value)}
                      </Text>
                    </View>
                  ))}
              </Card>
            )}
          </>
        ) : (
          <Card variant="outlined" style={styles.metricsCard}>
            <Text style={styles.emptyBody}>No metrics available yet. Ensure backend is running.</Text>
          </Card>
        )}

        <Card variant="outlined" style={styles.eventsCard}>
          <View style={styles.eventsHeader}>
            <Text style={styles.sectionTitle}>Realtime Pipeline Events</Text>
            <Badge label={`${pipelineEvents.length} live`} variant="success" small dot />
          </View>

          {pipelineEvents.length > 0 ? (
            pipelineEvents.slice(0, 10).map((event, i) => {
              const cfg = STATUS_CONFIG[event.status?.toLowerCase() ?? 'running'] || STATUS_CONFIG.running;
              return (
                <View key={`${event.trace_id}-${i}`} style={styles.eventRow}>
                  {cfg.pulse ? (
                    <NeuralPulse active size={5} color={cfg.color} />
                  ) : (
                    <View style={[styles.eventDot, { backgroundColor: cfg.color }]} />
                  )}
                  <View style={styles.eventContent}>
                    <Text style={styles.eventName} numberOfLines={1}>{event.step_name.replace(/_/g, ' ')}</Text>
                    <Text style={styles.eventMeta}>{event.event_type} · {event.status} · {Math.round(event.duration_ms)}ms</Text>
                  </View>
                </View>
              );
            })
          ) : (
            <Text style={styles.emptyBody}>Waiting for pipeline events...</Text>
          )}
        </Card>

        <Card variant="outlined" style={styles.eventsCard}>
          <View style={styles.eventsHeader}>
            <Text style={styles.sectionTitle}>Runtime Safety Queue</Text>
            <View style={styles.headerActions}>
              <Badge label={`${runtimePermissions.length} pending`} variant={runtimePermissions.length > 0 ? 'warning' : 'ghost'} small />
              <Button label="Refresh" variant="ghost" size="xs" onPress={() => void loadRuntimeOps()} disabled={runtimeBusy} />
            </View>
          </View>

          {runtimeBusy && runtimePermissions.length === 0 ? (
            <ActivityIndicator color={NEURAL.primary} />
          ) : runtimePermissions.length > 0 ? (
            runtimePermissions.slice(0, 6).map((permission) => (
              <View key={permission.permission_id} style={styles.permissionRow}>
                <View style={styles.permissionTop}>
                  <Text style={styles.permissionTool} numberOfLines={1}>{permission.tool_name}</Text>
                  <Badge label={permission.status} variant={PERMISSION_VARIANT[permission.status] || 'warning'} small />
                </View>
                <Text style={styles.permissionBody} numberOfLines={2}>{permission.reason || permission.command_text}</Text>
                <Text style={styles.permissionMeta}>{formatIsoRelative(permission.created_at)} · {permission.request_id}</Text>
                <View style={styles.rowActions}>
                  <Button
                    label="Approve"
                    size="xs"
                    variant="success"
                    onPress={() => void resolvePermission(permission.permission_id, true)}
                    disabled={actionBusyId === permission.permission_id}
                  />
                  <Button
                    label="Deny"
                    size="xs"
                    variant="error"
                    onPress={() => void resolvePermission(permission.permission_id, false)}
                    disabled={actionBusyId === permission.permission_id}
                  />
                </View>
              </View>
            ))
          ) : (
            <Text style={styles.emptyBody}>No pending runtime permissions.</Text>
          )}
        </Card>

        <Card variant="outlined" style={styles.eventsCard}>
          <View style={styles.eventsHeader}>
            <Text style={styles.sectionTitle}>Runtime Executor</Text>
            <Badge label={runtimeExecutorStatus?.running ? 'Running' : 'Idle'} variant={runtimeExecutorStatus?.running ? 'success' : 'ghost'} small />
          </View>
          {runtimeExecutorStatus ? (
            <View style={styles.executorGrid}>
              <ExecutorMetric label="Approved" value={runtimeExecutorStatus.summary.approved_total} />
              <ExecutorMetric label="Pending" value={runtimeExecutorStatus.summary.pending_total} />
              <ExecutorMetric label="Running" value={runtimeExecutorStatus.summary.running} />
              <ExecutorMetric label="Failed" value={runtimeExecutorStatus.summary.failed} />
            </View>
          ) : (
            <Text style={styles.emptyBody}>Executor status unavailable.</Text>
          )}
        </Card>

        <Card variant="outlined" style={styles.eventsCard}>
          <View style={styles.eventsHeader}>
            <Text style={styles.sectionTitle}>Runtime Task Board</Text>
            <Badge label={`${runtimeTasks.length} tasks`} variant="info" small />
          </View>
          {runtimeTasks.length > 0 ? (
            runtimeTasks.slice(0, 10).map((task) => {
              const canCancel = !['completed', 'failed', 'cancelled'].includes(task.state);
              return (
                <View key={task.task_id} style={styles.taskRow}>
                  <View style={styles.taskHead}>
                    <Text style={styles.taskId} numberOfLines={1}>{task.task_id}</Text>
                    <Badge label={task.state} variant={TASK_VARIANT[task.state] || 'info'} small />
                  </View>
                  <Text style={styles.taskMeta}>Updated {formatIsoRelative(task.updated_at)}</Text>
                  {task.permission_scope?.length ? (
                    <Text style={styles.taskMeta} numberOfLines={1}>Scope: {task.permission_scope.join(', ')}</Text>
                  ) : null}
                  <View style={styles.rowActions}>
                    <Button
                      label="Refresh"
                      size="xs"
                      variant="ghost"
                      onPress={async () => {
                        try {
                          const response = await api.getRuntimeTask(task.task_id);
                          setRuntimeTasks((prev) => {
                            const next = prev.filter((t) => t.task_id !== task.task_id);
                            return sortTasks([response.task, ...next]).slice(0, 40);
                          });
                        } catch (e) {
                          setRuntimeError(e instanceof Error ? e.message : String(e));
                        }
                      }}
                      disabled={actionBusyId === task.task_id}
                    />
                    <Button
                      label="Cancel"
                      size="xs"
                      variant="error"
                      onPress={() => void cancelTask(task.task_id)}
                      disabled={!canCancel || actionBusyId === task.task_id}
                    />
                  </View>
                </View>
              );
            })
          ) : (
            <Text style={styles.emptyBody}>No runtime tasks yet.</Text>
          )}
        </Card>

        <Card variant="outlined" style={styles.eventsCard}>
          <View style={styles.eventsHeader}>
            <Text style={styles.sectionTitle}>Runtime Task Events</Text>
            <Badge label={Platform.OS === 'web' ? 'SSE live' : 'Polling mode'} variant="secondary" small />
          </View>
          {runtimeTaskEvents.length > 0 ? (
            runtimeTaskEvents.slice(0, 12).map((event) => (
              <View key={event.event_id} style={styles.eventRow}>
                <View style={[styles.eventDot, { backgroundColor: NEURAL.secondary }]} />
                <View style={styles.eventContent}>
                  <Text style={styles.eventName}>{event.event_type.replace(/_/g, ' ')}</Text>
                  <Text style={styles.eventMeta}>{event.task.task_id} · {formatIsoRelative(event.timestamp)}</Text>
                </View>
              </View>
            ))
          ) : (
            <Text style={styles.emptyBody}>
              {Platform.OS === 'web'
                ? 'Waiting for task SSE events...'
                : 'Task event stream unavailable on native fetch. Polling snapshots only.'}
            </Text>
          )}
        </Card>

        {runtimeError ? (
          <Card variant="outlined" style={styles.metricsCard}>
            <Text style={styles.runtimeError}>{runtimeError}</Text>
          </Card>
        ) : null}

        <View style={styles.tracesWrap}>
          <PipelineTracesList baseUrl={apiBaseUrl} refreshInterval={4000} maxTraces={30} />
        </View>
      </ScrollView>
    </View>
  );
}

function ExecutorMetric({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.executorTile}>
      <Text style={styles.executorValue}>{shortNum(value)}</Text>
      <Text style={styles.executorLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  scroll: { paddingBottom: SPACING['5xl'] },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
  },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${NEURAL.tertiary}22`,
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.sm,
    paddingVertical: 4,
    gap: 4,
    borderWidth: 1,
    borderColor: `${NEURAL.tertiary}60`,
  },
  liveText: { fontSize: FONT_SIZE.xs, fontWeight: FONT_WEIGHT.bold, color: NEURAL.tertiary },
  subtitle: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    paddingHorizontal: SPACING.lg,
    marginBottom: SPACING.lg,
    marginTop: 4,
  },

  loader: { marginVertical: SPACING['4xl'] },

  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    marginBottom: SPACING.sm,
  },
  metricTile: {
    width: '30%',
    flexGrow: 1,
    alignItems: 'center',
    paddingVertical: SPACING.md,
    gap: 3,
  },
  metricTileIcon: { marginBottom: 1 },
  metricTileValue: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.primary },
  metricTileLabel: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  expandBtn: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, alignItems: 'center' },
  expandText: { fontSize: FONT_SIZE.sm, color: NEURAL.primary, fontWeight: FONT_WEIGHT.medium },

  metricsCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: SPACING.xs + 2,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
  },
  metricKey: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, textTransform: 'capitalize' },
  metricVal: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurface,
    fontWeight: FONT_WEIGHT.semibold,
    maxWidth: '50%',
    textAlign: 'right',
  },

  eventsCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  eventsHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: SPACING.sm },
  sectionTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },

  eventRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}30`,
  },
  eventDot: { width: 8, height: 8, borderRadius: 4 },
  eventContent: { flex: 1 },
  eventName: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurface,
    fontWeight: FONT_WEIGHT.medium,
    textTransform: 'capitalize',
  },
  eventMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, marginTop: 2 },

  headerActions: { flexDirection: 'row', alignItems: 'center', gap: SPACING.xs },

  permissionRow: {
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}35`,
    paddingTop: SPACING.sm,
    gap: 4,
  },
  permissionTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: SPACING.sm },
  permissionTool: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold },
  permissionBody: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
  permissionMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  rowActions: { flexDirection: 'row', gap: SPACING.xs, flexWrap: 'wrap' },

  executorGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.sm },
  executorTile: {
    width: '47%',
    flexGrow: 1,
    borderRadius: RADIUS.md,
    backgroundColor: NEURAL.surfaceContainerLow,
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}70`,
    paddingVertical: SPACING.sm,
    alignItems: 'center',
    gap: 2,
  },
  executorValue: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.bold, color: NEURAL.primary },
  executorLabel: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  taskRow: {
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}35`,
    paddingTop: SPACING.sm,
    gap: 4,
  },
  taskHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: SPACING.sm },
  taskId: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold },
  taskMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  runtimeError: { fontSize: FONT_SIZE.sm, color: NEURAL.error },

  tracesWrap: { minHeight: 300 },
  emptyBody: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, textAlign: 'center', padding: SPACING.md },
});
