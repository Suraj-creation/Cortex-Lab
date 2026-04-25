/**
 * DashboardScreen — Cortex Lab mobile hub for deep applications.
 * Presents the core surfaces from the web product in a mobile-first launchpad.
 */
import React, { useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
  TouchableOpacity,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon, type AppIconName } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Badge } from '../components/ui/Badge';
import type { NavKey } from '../components/ui/BottomNav';
import type { GraphData, ModelStatus, RAGStats } from '../../shared/core/types';

interface DashboardScreenProps {
  ragStats: RAGStats | null;
  graphData: GraphData | null;
  documentCount: number;
  apiBaseUrl: string;
  modelStatus: ModelStatus;
  loadingView: boolean;
  onRefresh: () => void;
  onOpenView: (view: NavKey) => void;
}

interface FeatureTile {
  key: NavKey;
  title: string;
  description: string;
  icon: AppIconName;
  badge: string;
  badgeVariant: 'primary' | 'success' | 'info' | 'violet' | 'warning';
  metric?: string;
}

function formatEndpoint(rawUrl: string): string {
  try {
    return new URL(rawUrl).host;
  } catch {
    return rawUrl;
  }
}

export function DashboardScreen({
  ragStats,
  graphData,
  documentCount,
  apiBaseUrl,
  modelStatus,
  loadingView,
  onRefresh,
  onOpenView,
}: DashboardScreenProps) {
  const isConnected = modelStatus.status !== 'offline';
  const endpoint = formatEndpoint(apiBaseUrl);

  const featureTiles = useMemo<FeatureTile[]>(() => {
    const graphNodes = graphData?.nodes?.length ?? ragStats?.graph?.nodes ?? 0;
    const graphEdges = graphData?.edges?.length ?? ragStats?.graph?.edges ?? 0;
    const memories = ragStats?.memories?.memories ?? 0;
    const runtimeTraces = ragStats?.cache?.total_queries ?? 0;

    return [
      {
        key: 'chat',
        title: 'RAG System',
        description: 'Memory-grounded chat, fast recall, and streaming answers.',
        icon: 'chat-processing-outline',
        badge: 'Core',
        badgeVariant: 'primary',
        metric: `${memories} memories`,
      },
      {
        key: 'agent',
        title: 'Agent Chat',
        description: 'Tiered orchestration, steering, sessions, and runtime control.',
        icon: 'robot-outline',
        badge: 'Agentic',
        badgeVariant: 'violet',
      },
      {
        key: 'wiki',
        title: 'Personal Wiki',
        description: 'Canonical memory pages, claims, linting, and compaction.',
        icon: 'book-open-page-variant-outline',
        badge: 'Knowledge',
        badgeVariant: 'info',
      },
      {
        key: 'memories',
        title: 'Memory Browser',
        description: 'Search, ingest, inspect, and manage long-term memory objects.',
        icon: 'brain',
        badge: 'Recall',
        badgeVariant: 'success',
        metric: `${memories} stored`,
      },
      {
        key: 'graph',
        title: 'Knowledge Graph',
        description: 'Entity nodes, relations, and graph-driven context traversal.',
        icon: 'graph-outline',
        badge: 'Graph',
        badgeVariant: 'info',
        metric: `${graphNodes} nodes · ${graphEdges} edges`,
      },
      {
        key: 'dashboard',
        title: 'RAG Dashboard',
        description: 'Pipeline health, cache behavior, vector stats, and load signals.',
        icon: 'view-dashboard-outline',
        badge: 'Metrics',
        badgeVariant: 'primary',
        metric: `${runtimeTraces} queries`,
      },
      {
        key: 'observability',
        title: 'Pipeline Observability',
        description: 'Traces, live events, runtime safety queue, and task execution.',
        icon: 'chart-timeline-variant',
        badge: 'Ops',
        badgeVariant: 'warning',
      },
      {
        key: 'ambient',
        title: 'Ambient Listening',
        description: 'Live transcript, continuous context, and Gemini voice runtime.',
        icon: 'microphone-outline',
        badge: 'Live',
        badgeVariant: 'success',
      },
      {
        key: 'documents',
        title: 'PageIndex Documents',
        description: 'Upload, query, and inspect document trees and grounded answers.',
        icon: 'file-document-outline',
        badge: 'Docs',
        badgeVariant: 'info',
        metric: `${documentCount} indexed`,
      },
      {
        key: 'session-forge',
        title: 'Session Forge',
        description: 'Crystallization, summaries, gap mapping, and belief detection.',
        icon: 'atom-variant',
        badge: 'Deep App',
        badgeVariant: 'violet',
      },
      {
        key: 'chronicle',
        title: 'Life Chronicle',
        description: 'Moment capture, observation logging, and timeline snapshots.',
        icon: 'timeline-text-outline',
        badge: 'Chronicle',
        badgeVariant: 'warning',
      },
    ];
  }, [documentCount, graphData, ragStats]);

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
      refreshControl={
        <RefreshControl
          refreshing={loadingView}
          onRefresh={onRefresh}
          tintColor="#6366f1"
          colors={['#6366f1']}
        />
      }
    >
      <LinearGradient
        colors={['#f8fbff', '#e7eeff', '#dce5ff']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.hero}
      >
        <View style={styles.heroTopRow}>
          <View style={styles.heroTitleWrap}>
            <Text style={styles.heroEyebrow}>Cortex Deep Applications</Text>
            <Text style={styles.heroTitle}>Production mobile command center</Text>
            <Text style={styles.heroBody}>
              Agent chat, personal wiki, graph memory, observability, ambient listening,
              and document intelligence are all available from the same mobile shell now.
            </Text>
          </View>
          <View style={styles.heroOrb}>
            <AppIcon name="brain" size={24} color="#ffffff" />
          </View>
        </View>

        <View style={styles.heroBadgeRow}>
          <Badge
            label={isConnected ? 'Backend Connected' : 'Backend Offline'}
            variant={isConnected ? 'success' : 'error'}
            size="md"
            dot
          />
          <Badge
            label={modelStatus.model_info?.llm_provider?.toUpperCase() || 'Runtime'}
            variant="violet"
            size="md"
          />
        </View>

        <View style={styles.heroInfoCard}>
          <View style={styles.heroInfoRow}>
            <Text style={styles.heroInfoLabel}>Live backend</Text>
            <Text style={styles.heroInfoValue}>{endpoint}</Text>
          </View>
          <View style={styles.heroInfoRow}>
            <Text style={styles.heroInfoLabel}>Knowledge graph</Text>
            <Text style={styles.heroInfoValue}>
              {(graphData?.nodes?.length ?? ragStats?.graph?.nodes ?? 0)} nodes
            </Text>
          </View>
          <View style={styles.heroInfoRow}>
            <Text style={styles.heroInfoLabel}>Indexed documents</Text>
            <Text style={styles.heroInfoValue}>{documentCount}</Text>
          </View>
        </View>
      </LinearGradient>

      <View style={styles.metricsRow}>
        <MetricCard
          label="Memories"
          value={String(ragStats?.memories?.memories ?? 0)}
          tone="indigo"
          compact
          style={styles.metricCard}
        />
        <MetricCard
          label="Graph"
          value={String(graphData?.nodes?.length ?? ragStats?.graph?.nodes ?? 0)}
          tone="blue"
          compact
          style={styles.metricCard}
        />
        <MetricCard
          label="Queries"
          value={String(ragStats?.cache?.total_queries ?? 0)}
          tone="emerald"
          compact
          style={styles.metricCard}
        />
      </View>

      <SectionHeader
        title="Core Surfaces"
        subtitle="Everything critical from the web app, optimized for mobile"
        icon={<AppIcon name="view-dashboard-outline" size={18} color="#6366f1" />}
      />

      <View style={styles.featureGrid}>
        {featureTiles.map((feature) => (
          <TouchableOpacity
            key={feature.key}
            style={styles.featureTouch}
            activeOpacity={0.82}
            onPress={() => onOpenView(feature.key)}
          >
            <Card variant="outlined" padding="lg" style={styles.featureCard}>
              <View style={styles.featureHeader}>
                <View style={styles.featureIconWrap}>
                  <AppIcon name={feature.icon} size={18} color="#4338ca" />
                </View>
                <Badge label={feature.badge} variant={feature.badgeVariant} size="sm" />
              </View>

              <Text style={styles.featureTitle}>{feature.title}</Text>
              <Text style={styles.featureDescription}>{feature.description}</Text>

              <View style={styles.featureFooter}>
                <Text style={styles.featureMetric}>{feature.metric || 'Open surface'}</Text>
                <AppIcon name="arrow-right" size={16} color="#94a3b8" />
              </View>
            </Card>
          </TouchableOpacity>
        ))}
      </View>

      <Card variant="outlined" padding="lg" style={styles.pipelineCard}>
        <SectionHeader
          title="Pipeline Snapshot"
          subtitle="Live retrieval system telemetry"
          icon={<AppIcon name="chart-bar" size={16} color="#6366f1" />}
        />
        <View style={styles.pipelineRows}>
          <View style={styles.pipelineRow}>
            <Text style={styles.pipelineLabel}>Cache hit rate</Text>
            <Text style={styles.pipelineValue}>
              {ragStats?.cache?.hit_rate != null
                ? `${(ragStats.cache.hit_rate * 100).toFixed(1)}%`
                : 'Waiting for telemetry'}
            </Text>
          </View>
          <View style={styles.pipelineRow}>
            <Text style={styles.pipelineLabel}>Vector store</Text>
            <Text style={styles.pipelineValue}>
              {ragStats?.vectors?.total_vectors != null
                ? `${ragStats.vectors.total_vectors} vectors`
                : 'Unavailable'}
            </Text>
          </View>
          <View style={styles.pipelineRow}>
            <Text style={styles.pipelineLabel}>Active backend</Text>
            <Text style={styles.pipelineValue}>
              {ragStats?.memories?.backend || endpoint}
            </Text>
          </View>
          <View style={styles.pipelineRow}>
            <Text style={styles.pipelineLabel}>Model status</Text>
            <Text style={styles.pipelineValue}>
              {modelStatus.model_loaded ? 'Local model ready' : isConnected ? 'Remote runtime connected' : 'Offline'}
            </Text>
          </View>
        </View>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#e9eef8',
  },
  content: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.lg,
  },
  hero: {
    borderRadius: RADIUS['2xl'],
    padding: SPACING.xl,
    gap: SPACING.lg,
    ...SHADOWS.lg,
  },
  heroTopRow: {
    flexDirection: 'row',
    gap: SPACING.md,
    alignItems: 'flex-start',
  },
  heroTitleWrap: {
    flex: 1,
    gap: SPACING.xs,
  },
  heroEyebrow: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#5b67d8',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  heroTitle: {
    fontSize: FONT_SIZE['2xl'],
    fontWeight: FONT_WEIGHT.extrabold,
    color: '#111b32',
    lineHeight: 28,
  },
  heroBody: {
    fontSize: FONT_SIZE.sm,
    lineHeight: 20,
    color: '#475569',
  },
  heroOrb: {
    width: 52,
    height: 52,
    borderRadius: 18,
    backgroundColor: '#f8fbff',
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.md,
  },
  heroBadgeRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    flexWrap: 'wrap',
  },
  heroInfoCard: {
    backgroundColor: 'rgba(255,255,255,0.55)',
    borderRadius: RADIUS.xl,
    padding: SPACING.md,
    gap: SPACING.sm,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.92)',
  },
  heroInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: SPACING.md,
  },
  heroInfoLabel: {
    flex: 1,
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
  },
  heroInfoValue: {
    flex: 1,
    textAlign: 'right',
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#111b32',
  },
  metricsRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
  metricCard: {
    flex: 1,
  },
  featureGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.md,
  },
  featureTouch: {
    width: '48%',
  },
  featureCard: {
    minHeight: 178,
    justifyContent: 'space-between',
  },
  featureHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: SPACING.md,
  },
  featureIconWrap: {
    width: 42,
    height: 42,
    borderRadius: RADIUS.xl,
    backgroundColor: '#f5f8ff',
    borderWidth: 1,
    borderColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.md,
  },
  featureTitle: {
    fontSize: FONT_SIZE.md,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
    marginBottom: SPACING.xs,
  },
  featureDescription: {
    fontSize: FONT_SIZE.sm,
    lineHeight: 18,
    color: '#64748b',
    flex: 1,
  },
  featureFooter: {
    marginTop: SPACING.lg,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#f1f5f9',
    paddingTop: SPACING.sm,
  },
  featureMetric: {
    flex: 1,
    fontSize: FONT_SIZE.xs,
    color: '#4338ca',
    fontWeight: FONT_WEIGHT.semibold,
  },
  pipelineCard: {
    gap: SPACING.sm,
  },
  pipelineRows: {
    gap: SPACING.sm,
  },
  pipelineRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: SPACING.md,
    paddingVertical: SPACING.xs,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  pipelineLabel: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
  },
  pipelineValue: {
    flex: 1,
    textAlign: 'right',
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },
});
