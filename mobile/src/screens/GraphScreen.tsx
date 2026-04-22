/**
 * GraphScreen — Neural Dark Knowledge Graph
 * Stitch ref: de992097927847ada2034b4f3cfda1d0
 * SVG force-directed visualization with glowing nodes
 */
import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ActivityIndicator,
  Modal,
} from 'react-native';
import Svg, { Circle, Line, Text as SvgText, Defs, Filter, FeGaussianBlur, FeMerge, FeMergeNode } from 'react-native-svg';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { AppIcon } from '../components/ui/AppIcon';
import type { GraphData } from '../../shared/core/types';

const NODE_COLORS = [
  NEURAL.primary, NEURAL.secondary, NEURAL.tertiary, '#60a5fa', '#f59e0b', '#ff6e84',
];

function shortNum(v: number | undefined): string {
  if (typeof v !== 'number') return '0';
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return `${v}`;
}

/** Simple deterministic layout using circular positioning + spring approximation */
function computePositions(nodes: GraphData['nodes'], width: number, height: number) {
  const cx = width / 2;
  const cy = height / 2;
  const r  = Math.min(cx, cy) * 0.72;
  const positions: Record<string, { x: number; y: number; radius: number; color: string }> = {};
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / nodes.length;
    const rank  = node.memory_count || node.mentions || 1;
    const size  = Math.min(22, Math.max(8, Math.log(rank + 1) * 4));
    positions[node.id] = {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      radius: size,
      color: NODE_COLORS[i % NODE_COLORS.length],
    };
  });
  return positions;
}

const SCREEN_W = Dimensions.get('window').width;
const GRAPH_H  = 260;

interface GraphScreenProps {
  graphData: GraphData | null;
  loadingView: boolean;
}

export function GraphScreen({ graphData, loadingView }: GraphScreenProps) {
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [nodeTypeFilter, setNodeTypeFilter] = useState<string>('All');

  const typeCounts = useMemo(() => {
    if (!graphData) return [];
    const counts: Record<string, number> = {};
    graphData.nodes.forEach((n) => {
      const t = n.type || 'Entity';
      counts[t] = (counts[t] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [graphData]);

  const nodeTypes = ['All', ...typeCounts.map(([t]) => t)];

  const visNodes = useMemo(() => {
    if (!graphData) return [];
    return nodeTypeFilter === 'All'
      ? graphData.nodes.slice(0, 40)
      : graphData.nodes.filter((n) => (n.type || 'Entity') === nodeTypeFilter).slice(0, 40);
  }, [graphData, nodeTypeFilter]);

  const positions = useMemo(() => computePositions(visNodes, SCREEN_W - 32, GRAPH_H), [visNodes]);

  const visEdges = useMemo(() => {
    if (!graphData) return [];
    const nodeIds = new Set(visNodes.map((n) => n.id));
    return graphData.edges
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .sort((a, b) => (b.weight || 0) - (a.weight || 0))
      .slice(0, 60);
  }, [graphData, visNodes]);

  const strongestEdges = useMemo(() => {
    if (!graphData) return [];
    return [...graphData.edges]
      .sort((a, b) => (b.weight || 0) - (a.weight || 0))
      .slice(0, 8);
  }, [graphData]);

  const topEntities = useMemo(() => {
    if (!graphData) return [];
    return [...graphData.nodes]
      .sort((a, b) => ((b.memory_count || b.mentions || 0) - (a.memory_count || a.mentions || 0)))
      .slice(0, 8);
  }, [graphData]);

  const activeNode = graphData?.nodes.find((n) => n.id === activeNodeId);

  if (loadingView) {
    return <View style={[styles.container, styles.center]}><ActivityIndicator color={NEURAL.primary} size="large" /></View>;
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <View style={[styles.container, styles.center]}>
        <Text style={styles.emptyTitle}>Graph not ready</Text>
        <Text style={styles.emptyBody}>Build connections through conversations.</Text>
      </View>
    );
  }

  const density = graphData.nodes.length > 1
    ? ((graphData.edges.length * 2) / (graphData.nodes.length * (graphData.nodes.length - 1)) * 100).toFixed(1) + '%'
    : '0%';

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Knowledge Graph</Text>
        </View>

        {/* Stat pills */}
        <View style={styles.statPills}>
          {[
            { label: 'Nodes',  value: shortNum(graphData.nodes.length) },
            { label: 'Edges',  value: shortNum(graphData.edges.length) },
            { label: 'Types',  value: `${typeCounts.length}` },
            { label: 'Density', value: density },
          ].map((s) => (
            <View key={s.label} style={styles.statPill}>
              <Text style={styles.statPillValue}>{s.value}</Text>
              <Text style={styles.statPillLabel}>{s.label}</Text>
            </View>
          ))}
        </View>

        {/* SVG Graph Canvas */}
        <View style={styles.canvasWrap}>
          <Svg width={SCREEN_W - 32} height={GRAPH_H}>
            <Defs>
              <Filter id="glow">
                <FeGaussianBlur stdDeviation="3" result="blur" />
                <FeMerge>
                  <FeMergeNode in="blur" />
                  <FeMergeNode in="SourceGraphic" />
                </FeMerge>
              </Filter>
            </Defs>

            {/* Edges */}
            {visEdges.map((edge, i) => {
              const s = positions[edge.source];
              const t = positions[edge.target];
              if (!s || !t) return null;
              const opacity = Math.min(0.7, Math.max(0.15, (edge.weight || 0.5)));
              return (
                <Line
                  key={`e-${i}`}
                  x1={s.x} y1={s.y}
                  x2={t.x} y2={t.y}
                  stroke={NEURAL.outlineVariant}
                  strokeWidth={1}
                  strokeOpacity={opacity}
                />
              );
            })}

            {/* Nodes */}
            {visNodes.map((node) => {
              const p = positions[node.id];
              if (!p) return null;
              const isActive = node.id === activeNodeId;
              return (
                <React.Fragment key={node.id}>
                  {/* Glow halo */}
                  <Circle
                    cx={p.x} cy={p.y}
                    r={p.radius + 5}
                    fill={p.color}
                    opacity={0.15}
                    filter="url(#glow)"
                  />
                  {/* Core */}
                  <Circle
                    cx={p.x} cy={p.y}
                    r={p.radius}
                    fill={p.color}
                    opacity={isActive ? 1 : 0.85}
                    onPress={() => setActiveNodeId(isActive ? null : node.id)}
                  />
                  {p.radius > 12 && (
                    <SvgText
                      x={p.x} y={p.y + 4}
                      textAnchor="middle"
                      fontSize={8}
                      fill="#ffffff"
                      fontWeight="bold"
                    >
                      {(node.label || '').slice(0, 8)}
                    </SvgText>
                  )}
                </React.Fragment>
              );
            })}
          </Svg>
        </View>

        {/* Node type filter */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterRow}>
          {nodeTypes.slice(0, 8).map((t) => (
            <TouchableOpacity
              key={t}
              onPress={() => setNodeTypeFilter(t)}
              style={[styles.filterChip, nodeTypeFilter === t && styles.filterChipActive]}
            >
              <Text style={[styles.filterText, nodeTypeFilter === t && styles.filterTextActive]}>{t}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Node Type Distribution */}
        <Card variant="outlined" style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>Node Type Distribution</Text>
          {typeCounts.slice(0, 8).map(([type, count]) => (
            <View key={type} style={styles.distRow}>
              <Text style={styles.distLabel}>{type}</Text>
              <View style={styles.distBarWrap}>
                <View style={[styles.distBar, { width: `${(count / (typeCounts[0]?.[1] || 1)) * 100}%` }]} />
              </View>
              <Text style={styles.distCount}>{count}</Text>
            </View>
          ))}
        </Card>

        {/* Top Entities */}
        <Card variant="outlined" style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>Top Entities</Text>
          {topEntities.map((node, i) => (
            <View key={node.id} style={styles.entityRow}>
              <Text style={styles.entityRank}>#{i + 1}</Text>
              <Text style={styles.entityName} numberOfLines={1}>{node.label}</Text>
              <Badge label={node.type || 'Entity'} variant="primary" small />
              <Text style={styles.entityCount}>{shortNum(node.memory_count || node.mentions)}</Text>
            </View>
          ))}
        </Card>

        {/* Strongest Relationships */}
        <Card variant="outlined" style={styles.sectionCard}>
          <Text style={styles.sectionTitle}>Strongest Relationships</Text>
          {strongestEdges.map((edge, i) => (
            <View key={i} style={styles.edgeRow}>
              <Text style={styles.edgeSource} numberOfLines={1}>{edge.source}</Text>
              <AppIcon name="arrow-right" size={16} color={NEURAL.onSurfaceVariant} style={styles.edgeArrow} />
              <Text style={styles.edgeTarget} numberOfLines={1}>{edge.target}</Text>
              <Badge label={`${((edge.weight || 0) * 100).toFixed(0)}%`} variant="info" small />
            </View>
          ))}
        </Card>

        {/* Node detail bottom sheet (inline when a node is selected) */}
        {activeNode && (
          <Card variant="elevated" style={styles.nodeDetail}>
            <View style={styles.nodeDetailHeader}>
              <Text style={styles.nodeDetailName}>{activeNode.label}</Text>
              <TouchableOpacity onPress={() => setActiveNodeId(null)}>
                <AppIcon name="close" size={18} color={NEURAL.onSurfaceVariant} />
              </TouchableOpacity>
            </View>
            <Badge label={activeNode.type || 'Entity'} variant="primary" />
            {(activeNode.memory_count || activeNode.mentions) ? (
              <Text style={styles.nodeDetailStat}>
                {shortNum(activeNode.memory_count || activeNode.mentions)} mentions
              </Text>
            ) : null}
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
  header: { paddingHorizontal: SPACING.lg, paddingTop: SPACING.lg, paddingBottom: SPACING.md },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },

  // Stat pills
  statPills: {
    flexDirection: 'row',
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
    marginBottom: SPACING.md,
  },
  statPill: {
    flex: 1,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderRadius: RADIUS.xl,
    padding: SPACING.sm,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },
  statPillValue: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.bold, color: NEURAL.primary },
  statPillLabel: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, marginTop: 2 },

  // Graph canvas
  canvasWrap: {
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.md,
    backgroundColor: NEURAL.surfaceContainerLow,
    borderRadius: RADIUS.xl,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    height: GRAPH_H,
  },

  // Filter
  filterRow: { paddingHorizontal: SPACING.lg, gap: SPACING.sm, marginBottom: SPACING.md },
  filterChip: {
    paddingHorizontal: SPACING.md,
    paddingVertical: 5,
    borderRadius: RADIUS.full,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },
  filterChipActive: { backgroundColor: `${NEURAL.primary}26`, borderColor: `${NEURAL.primary}60` },
  filterText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, fontWeight: FONT_WEIGHT.medium },
  filterTextActive: { color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },

  // Section cards
  sectionCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.xs },
  sectionTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, marginBottom: SPACING.sm },

  // Distribution
  distRow: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, paddingVertical: SPACING.xs },
  distLabel: { width: 80, fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
  distBarWrap: { flex: 1, height: 4, backgroundColor: NEURAL.outlineVariant, borderRadius: 2, overflow: 'hidden' },
  distBar: { height: 4, backgroundColor: NEURAL.primary, borderRadius: 2 },
  distCount: { width: 32, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, textAlign: 'right', fontWeight: FONT_WEIGHT.semibold },

  // Entities
  entityRow: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, paddingVertical: SPACING.xs, borderTopWidth: 1, borderTopColor: `${NEURAL.outlineVariant}40` },
  entityRank: { width: 22, fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },
  entityName: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.medium },
  entityCount: { fontSize: FONT_SIZE.sm, color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },

  // Edges
  edgeRow: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, paddingVertical: SPACING.xs, borderTopWidth: 1, borderTopColor: `${NEURAL.outlineVariant}40` },
  edgeSource: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface },
  edgeArrow: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
  edgeTarget: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface },

  // Node detail
  nodeDetail: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  nodeDetailHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  nodeDetailName: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  nodeDetailStat: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },

  emptyTitle: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, marginBottom: SPACING.sm },
  emptyBody: { fontSize: FONT_SIZE.base, color: NEURAL.onSurfaceVariant, textAlign: 'center' },
});
