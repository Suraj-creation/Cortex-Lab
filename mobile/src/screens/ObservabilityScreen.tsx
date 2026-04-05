/**
 * ObservabilityScreen — Neural Dark Pipeline Observability
 * Stitch ref: abca14fc346f4e91b5fa6252d9b52eee
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
} from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { NeuralPulse } from '../components/ui/NeuralPulse';
import { AppIcon, type AppIconName } from '../components/ui/AppIcon';
import PipelineTracesList from '../components/PipelineTracesList';
import type { LivePipelineEvent } from '../../shared/core/types';

function shortNum(v: unknown): string {
  if (typeof v !== 'number') return String(v ?? '—');
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

const STATUS_CONFIG: Record<string, { color: string; pulse: boolean }> = {
  running:  { color: NEURAL.primary,  pulse: true },
  complete: { color: NEURAL.tertiary, pulse: false },
  success:  { color: NEURAL.tertiary, pulse: false },
  error:    { color: NEURAL.error,    pulse: false },
  failed:   { color: NEURAL.error,    pulse: false },
};

interface ObservabilityScreenProps {
  observabilityMetrics: Record<string, unknown> | null;
  pipelineEvents: LivePipelineEvent[];
  loadingView: boolean;
  apiBaseUrl: string;
}

// Key metrics to show prominently
const METRIC_KEYS = [
  { key: 'total_queries',      label: 'Queries',      iconName: 'database-search-outline' as AppIconName },
  { key: 'cache_hits',         label: 'Cache Hits',   iconName: 'lightning-bolt-outline' as AppIconName },
  { key: 'total_embeddings',   label: 'Embeddings',   iconName: 'vector-link' as AppIconName },
  { key: 'total_memories',     label: 'Memories',     iconName: 'brain' as AppIconName },
  { key: 'avg_response_ms',    label: 'Avg Latency',  iconName: 'timer-outline' as AppIconName },
  { key: 'error_count',        label: 'Errors',       iconName: 'alert-circle-outline' as AppIconName },
];

export function ObservabilityScreen({
  observabilityMetrics,
  pipelineEvents,
  loadingView,
  apiBaseUrl,
}: ObservabilityScreenProps) {
  const [expandedMetrics, setExpandedMetrics] = useState(false);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Pipeline Observability</Text>
          <View style={styles.liveBadge}>
            <NeuralPulse active size={6} color={NEURAL.tertiary} />
            <Text style={styles.liveText}>LIVE</Text>
          </View>
        </View>
        <Text style={styles.subtitle}>
          Updated {formatRelativeTime(pipelineEvents[0]?.timestamp ?? null)}
        </Text>

        {/* 6-tile live metrics grid */}
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

            {/* Extended metrics */}
            <TouchableOpacity
              onPress={() => setExpandedMetrics((p) => !p)}
              style={styles.expandBtn}
            >
              <Text style={styles.expandText}>
                {expandedMetrics ? 'Hide all metrics' : 'Show all metrics'}
              </Text>
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

        {/* Realtime Pipeline Events feed */}
        <Card variant="outlined" style={styles.eventsCard}>
          <View style={styles.eventsHeader}>
            <Text style={styles.sectionTitle}>Realtime Pipeline Events</Text>
            <Badge
              label={`${pipelineEvents.length} live`}
              variant="success"
              small
              dot
            />
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
                    <Text style={styles.eventName} numberOfLines={1}>
                      {event.step_name.replace(/_/g, ' ')}
                    </Text>
                    <Text style={styles.eventMeta}>
                      {event.event_type} · {event.status} · {Math.round(event.duration_ms)}ms
                    </Text>
                  </View>
                </View>
              );
            })
          ) : (
            <Text style={styles.emptyBody}>Waiting for pipeline events…</Text>
          )}
        </Card>

        {/* Recent Traces (full component with dark theme) */}
        <View style={styles.tracesWrap}>
          <PipelineTracesList baseUrl={apiBaseUrl} refreshInterval={4000} maxTraces={30} />
        </View>
      </ScrollView>
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
  subtitle: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, paddingHorizontal: SPACING.lg, marginBottom: SPACING.lg, marginTop: 4 },

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
  metricVal: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold, maxWidth: '50%', textAlign: 'right' },

  eventsCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  eventsHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
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
  eventName: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.medium, textTransform: 'capitalize' },
  eventMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, marginTop: 2 },

  tracesWrap: { minHeight: 300 },
  emptyBody: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, textAlign: 'center', padding: SPACING.md },
});
