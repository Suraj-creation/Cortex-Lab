/**
 * GraphScreen — Cortex Aurora Knowledge Graph
 * SVG-based visualization with light theme
 */
import React, { useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Dimensions,
} from 'react-native';
import Svg, { Circle, Line, Text as SvgText } from 'react-native-svg';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Badge } from '../components/ui/Badge';
import { AppIcon } from '../components/ui/AppIcon';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { EmptyState } from '../components/ui/EmptyState';
import type { GraphData } from '../../shared/core/types';

interface GraphScreenProps {
  graphData: GraphData | null;
  loadingView: boolean;
}

const GRAPH_COLORS = ['#6366f1', '#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#f43f5e', '#06b6d4', '#ec4899'];

export function GraphScreen({ graphData, loadingView }: GraphScreenProps) {
  const { width: screenWidth } = Dimensions.get('window');
  const graphSize = screenWidth - SPACING.lg * 2;

  const nodes = useMemo(() => {
    if (!graphData?.nodes?.length) return [];
    const angleStep = (2 * Math.PI) / graphData.nodes.length;
    const cx = graphSize / 2;
    const cy = graphSize / 2;
    const radius = Math.min(graphSize / 2 - 50, 160);

    return graphData.nodes.map((node, i) => ({
      ...node,
      x: cx + radius * Math.cos(angleStep * i - Math.PI / 2),
      y: cy + radius * Math.sin(angleStep * i - Math.PI / 2),
      color: GRAPH_COLORS[i % GRAPH_COLORS.length],
      radius: Math.max(8, Math.min(18, 7 + (node.mentions ?? node.memory_count ?? 1) * 0.6)),
    }));
  }, [graphData, graphSize]);

  const edges = useMemo(() => {
    if (!graphData?.edges?.length || !nodes.length) return [];
    return graphData.edges.map((edge) => {
      const source = nodes.find((n) => n.id === edge.source);
      const target = nodes.find((n) => n.id === edge.target);
      if (!source || !target) return null;
      return { ...edge, x1: source.x, y1: source.y, x2: target.x, y2: target.y };
    }).filter(Boolean);
  }, [graphData, nodes]);

  const relationshipRows = useMemo(() => {
    return edges.slice(0, 12).map((edge) => {
      const source = nodes.find((node) => node.id === edge?.source);
      const target = nodes.find((node) => node.id === edge?.target);
      return {
        key: `${edge?.source}-${edge?.target}-${edge?.relation}`,
        label: `${source?.label || edge?.source} → ${target?.label || edge?.target}`,
        relation: edge?.relation || 'related',
      };
    });
  }, [edges, nodes]);

  if (loadingView && !graphData) {
    return <LoadingSpinner fullScreen message="Loading knowledge graph..." />;
  }

  if (!graphData || !nodes.length) {
    return (
      <View style={styles.container}>
        <EmptyState
          icon="graph-outline"
          title="No Graph Data"
          message="Ingest memories to build the knowledge graph."
        />
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      <SectionHeader
        title="Knowledge Graph"
        subtitle={`${nodes.length} entities · ${edges.length} relationships`}
        icon={<AppIcon name="graph-outline" size={18} color="#6366f1" />}
      />

      {/* Stats */}
      <View style={styles.statsRow}>
        <MetricCard
          label="Entities"
          value={nodes.length}
          tone="indigo"
          compact
          style={styles.statCard}
        />
        <MetricCard
          label="Edges"
          value={edges.length}
          tone="violet"
          compact
          style={styles.statCard}
        />
        <MetricCard
          label="Clusters"
          value={new Set(graphData.nodes.map(n => n.type)).size}
          tone="emerald"
          compact
          style={styles.statCard}
        />
      </View>

      {/* Graph visualization */}
      <Card variant="elevated" padding="sm" style={styles.graphCard}>
        <Svg width={graphSize} height={graphSize}>
          {/* Edges */}
          {edges.map((edge, i) => edge && (
            <Line
              key={`e-${i}`}
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              stroke="#e2e8f0"
              strokeWidth={1.5}
              opacity={0.6}
            />
          ))}
          {/* Nodes */}
          {nodes.map((node, i) => (
            <React.Fragment key={`n-${i}`}>
              {/* Outer glow */}
              <Circle
                cx={node.x}
                cy={node.y}
                r={node.radius + 7}
                fill={node.color}
                opacity={0.15}
              />
              {/* Core */}
              <Circle
                cx={node.x}
                cy={node.y}
                r={node.radius}
                fill={node.color}
                stroke="#ffffff"
                strokeWidth={2}
              />
              {/* Label */}
              <SvgText
                x={node.x}
                y={node.y + 20}
                textAnchor="middle"
                fontSize={9}
                fontWeight="600"
                fill="#475569"
              >
                {(node.label || node.id || '').slice(0, 14)}
              </SvgText>
            </React.Fragment>
          ))}
        </Svg>
      </Card>

      <Card variant="outlined" padding="lg" style={styles.entitySection}>
        <SectionHeader title="Relationship Preview" subtitle={`${edges.length} total`} />
        {relationshipRows.length > 0 ? relationshipRows.map((row) => (
          <View key={row.key} style={styles.relationRow}>
            <View style={styles.relationTextWrap}>
              <Text style={styles.relationLabel} numberOfLines={1}>{row.label}</Text>
              <Text style={styles.relationMeta}>{row.relation}</Text>
            </View>
            <Badge label={row.relation} variant="violet" size="sm" />
          </View>
        )) : (
          <Text style={styles.emptyCopy}>Relationships appear once linked memories are available.</Text>
        )}
      </Card>

      {/* Entity list */}
      <Card variant="outlined" padding="lg" style={styles.entitySection}>
        <SectionHeader title="Entities" subtitle={`${nodes.length} total`} />
        {nodes.slice(0, 20).map((node, i) => (
          <View key={i} style={styles.entityRow}>
            <View style={[styles.entityDot, { backgroundColor: node.color }]} />
            <Text style={styles.entityName} numberOfLines={1}>{node.label || node.id}</Text>
            {node.type && <Badge label={node.type} variant="default" size="sm" />}
          </View>
        ))}
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  content: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.md,
  },
  statsRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
  statCard: {
    flex: 1,
  },
  graphCard: {
    alignItems: 'center',
    backgroundColor: '#ffffff',
  },
  entitySection: {
    gap: SPACING.sm,
  },
  relationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  relationTextWrap: {
    flex: 1,
    gap: 2,
  },
  relationLabel: {
    fontSize: FONT_SIZE.sm,
    color: '#0f172a',
    fontWeight: FONT_WEIGHT.semibold,
  },
  relationMeta: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
  },
  entityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  entityDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  entityName: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    fontWeight: FONT_WEIGHT.medium,
  },
  emptyCopy: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
  },
});
