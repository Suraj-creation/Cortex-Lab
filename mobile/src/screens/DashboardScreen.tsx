/**
 * DashboardScreen — Neural Dark RAG Dashboard
 * Stitch ref: 53094837d331410794d2137fb52c803c
 */
import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ProgressBar } from '../components/ui/ProgressBar';
import { AppIcon, type AppIconName } from '../components/ui/AppIcon';
import type { RAGStats } from '../../shared/core/types';

function shortNum(v: number | undefined): string {
  if (typeof v !== 'number') return '0';
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return `${v}`;
}

function toPercent(v: number): string {
  return `${Math.round(v * 100)}%`;
}

interface DashboardScreenProps {
  ragStats: RAGStats | null;
  loadingView: boolean;
  onRefresh: () => void;
}

export function DashboardScreen({ ragStats, loadingView, onRefresh }: DashboardScreenProps) {
  if (loadingView) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator color={NEURAL.primary} size="large" />
      </View>
    );
  }

  if (!ragStats) {
    return (
      <View style={[styles.container, styles.center]}>
        <Text style={styles.emptyTitle}>Dashboard loading…</Text>
        <Text style={styles.emptyBody}>Connect to backend to view RAG stats.</Text>
      </View>
    );
  }

  const hitRate = ragStats.cache.hit_rate ?? 0;
  const vectorHot   = (ragStats.vectors as any)?.hot_count ?? 0;
  const vectorWarm  = (ragStats.vectors as any)?.warm_count ?? 0;
  const vectorCold  = (ragStats.vectors as any)?.cold_count ?? (ragStats.vectors?.total_vectors ?? 0);
  const totalVectors = ragStats.vectors?.total_vectors ?? 0;

  const heroMetrics: Array<{ label: string; value: string; iconName: AppIconName; color: string }> = [
    { label: 'Memories', value: shortNum(ragStats.memories?.memories), iconName: 'brain', color: NEURAL.primary },
    { label: 'Entities', value: shortNum(ragStats.memories?.entities), iconName: 'hexagon-outline', color: NEURAL.secondary },
    { label: 'Vectors', value: shortNum(ragStats.vectors?.total_vectors), iconName: 'vector-link', color: NEURAL.tertiary },
    { label: 'Graph Nodes', value: shortNum(ragStats.graph?.nodes), iconName: 'graph-outline', color: '#60a5fa' },
  ];

  const vectorTiers: Array<{ iconName: AppIconName; label: string; count: number; color: string }> = [
    { iconName: 'fire', label: 'Hot', count: vectorHot, color: NEURAL.error },
    { iconName: 'weather-sunny', label: 'Warm', count: vectorWarm, color: '#f59e0b' },
    { iconName: 'snowflake', label: 'Cold', count: vectorCold, color: '#60a5fa' },
  ];

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>RAG Dashboard</Text>
          <Badge label="Live" variant="success" dot />
        </View>

        {/* 2×2 Hero metrics */}
        <View style={styles.metricGrid}>
          {heroMetrics.map((m) => (
            <Card key={m.label} variant="elevated" style={styles.metricCard}>
              <AppIcon name={m.iconName} size={22} color={m.color} style={styles.metricIcon} />
              <Text style={[styles.metricValue, { color: m.color }]}>{m.value}</Text>
              <Text style={styles.metricLabel}>{m.label}</Text>
            </Card>
          ))}
        </View>

        {/* Cache Performance */}
        <Card variant="outlined" style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>Cache Performance</Text>
          <ProgressBar
            value={hitRate}
            label="Hit Rate"
            style={styles.progressBar}
          />
          {[
            { label: 'Total Queries',  value: ragStats.cache.total_queries },
            { label: 'Total Hits',     value: ragStats.cache.total_hits },
            { label: 'Exact Cache',    value: ragStats.cache.exact_cache_size },
            { label: 'Semantic Cache', value: ragStats.cache.semantic_cache_size },
            { label: 'Embedding Cache',value: ragStats.cache.embedding_cache_size },
          ].map((row) => (
            <View key={row.label} style={styles.statRow}>
              <Text style={styles.statLabel}>{row.label}</Text>
              <Text style={styles.statValue}>{shortNum(row.value)}</Text>
            </View>
          ))}
        </Card>

        {/* LLM Usage */}
        <Card variant="outlined" style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>LLM Usage</Text>
          {[
            { label: 'Total Calls',   value: shortNum(ragStats.llm?.call_count) },
            { label: 'Total Tokens',  value: shortNum(ragStats.llm?.total_tokens) },
            { label: 'Model Loaded',  value: ragStats.llm?.model_loaded ? 'Yes' : 'No' },
          ].map((row) => (
            <View key={row.label} style={styles.statRow}>
              <Text style={styles.statLabel}>{row.label}</Text>
              <Text style={styles.statValue}>{row.value}</Text>
            </View>
          ))}
        </Card>

        {/* Vector Store Tier Breakdown */}
        <Card variant="outlined" style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>Vector Store Tiers</Text>
          {vectorTiers.map((tier) => (
            <View key={tier.label} style={styles.tierRow}>
              <AppIcon name={tier.iconName} size={16} color={tier.color} style={styles.tierIcon} />
              <Text style={styles.tierLabel}>{tier.label}</Text>
              <View style={[styles.tierBadge, { borderColor: `${tier.color}60`, backgroundColor: `${tier.color}18` }]}>
                <Text style={[styles.tierCount, { color: tier.color }]}>{shortNum(tier.count)}</Text>
              </View>
              {totalVectors > 0 && (
                <Text style={styles.tierPct}>{toPercent(tier.count / totalVectors)}</Text>
              )}
            </View>
          ))}
        </Card>

        {/* Belief Deltas */}
        {(ragStats as any).beliefs && (
          <Card variant="outlined" style={styles.sectionCard}>
            <Text style={styles.sectionTitle}>Belief Deltas</Text>
            <Text style={styles.emptyBody}>No recent belief changes recorded.</Text>
          </Card>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  center: { alignItems: 'center', justifyContent: 'center' },
  scroll: { paddingBottom: SPACING['5xl'] },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.md,
  },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, letterSpacing: -0.5 },

  // Metric grid
  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    marginBottom: SPACING.lg,
  },
  metricCard: {
    width: '47%',
    flexGrow: 1,
    alignItems: 'center',
    paddingVertical: SPACING.xl,
    gap: SPACING.xs,
  },
  metricIcon: { marginBottom: 1 },
  metricValue: { fontSize: FONT_SIZE['3xl'], fontWeight: FONT_WEIGHT.extrabold },
  metricLabel: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },

  // Section card
  sectionCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  sectionTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, marginBottom: SPACING.xs },
  progressBar: { marginVertical: SPACING.sm },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: SPACING.xs + 2,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
  },
  statLabel: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
  statValue: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold },

  // Tier row
  tierRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
  },
  tierIcon: { width: 24 },
  tierLabel: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.medium },
  tierBadge: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: 2,
    borderRadius: RADIUS.full,
    borderWidth: 1,
  },
  tierCount: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.bold },
  tierPct: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, width: 36, textAlign: 'right' },

  emptyTitle: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface, marginBottom: SPACING.sm },
  emptyBody: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, textAlign: 'center' },
});
