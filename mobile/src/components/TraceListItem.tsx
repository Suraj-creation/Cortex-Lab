import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { PipelineTrace } from "../../shared/core/types";
import { NEURAL, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from "../theme/colors";

interface TraceListItemProps {
  trace: PipelineTrace;
}

export default function TraceListItem({ trace }: TraceListItemProps) {
  const formattedTime = new Date(trace.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const queryPreview =
    trace.query.length > 60 ? trace.query.substring(0, 57) + "..." : trace.query;

  const completedSteps = trace.steps.filter((s) => s.status === "completed")
    .length;

  // Check if quality gates were applied
  const qualityGates = [];
  if (trace.crag_evaluation) qualityGates.push("CRAG");
  if (trace.self_rag_critique) qualityGates.push("SelfRAG");
  if (trace.flare_trace?.triggered) qualityGates.push("FLARE");

  // Cache status badge
  const cacheHit = trace.cache_status?.hit ?? false;

  return (
    <View style={styles.container}>
      {/* Header row: time + duration + cache */}
      <View style={styles.header}>
        <Text style={styles.timestamp}>{formattedTime}</Text>
        <View style={styles.badges}>
          {cacheHit ? (
            <View style={[styles.badge, styles.cacheBadge]}>
              <Text style={styles.badgeText}>Cache</Text>
            </View>
          ) : null}
          <View style={styles.badge}>
            <Text style={styles.badgeText}>{Math.round(trace.total_duration_ms)}ms</Text>
          </View>
          <View style={[styles.badge, styles.confidenceBadge]}>
            <Text style={styles.badgeText}>
              {Math.round(trace.final_confidence * 100)}%
            </Text>
          </View>
        </View>
      </View>

      {/* Query preview */}
      <Text style={styles.queryText} numberOfLines={2}>
        {queryPreview}
      </Text>

      {/* Step breakdown and channels */}
      <View style={styles.footer}>
        <View style={styles.stepInfo}>
          <Text style={styles.stepCount}>
            {completedSteps}/{trace.steps.length} steps
          </Text>
          {trace.retrieval_channels.length > 0 ? (
            <Text style={styles.channelInfo}>
              · {trace.retrieval_channels.length} channels
            </Text>
          ) : null}
        </View>

        {/* Quality gates row */}
        {qualityGates.length > 0 ? (
          <View style={styles.qualityGates}>
            {qualityGates.map((gate, idx) => (
              <View key={idx} style={styles.qgBadge}>
                <Text style={styles.qgText}>{gate}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </View>

      {/* Evidence and entities preview */}
      <View style={styles.metaRow}>
        <Text style={styles.metaText}>
          {trace.evidence_count} evidence
        </Text>
        {trace.query_analysis?.entities?.length ? (
          <Text style={styles.metaText}>
            · {trace.query_analysis.entities.length} entities
          </Text>
        ) : null}
        {trace.token_usage?.total_tokens ? (
          <Text style={styles.metaText}>
            · {trace.token_usage.total_tokens} tokens
          </Text>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.xl,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.lg,
    marginBottom: SPACING.md,
    ...{},
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SPACING.md,
  },
  timestamp: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    fontFamily: "monospace",
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  badges: {
    flexDirection: "row",
    gap: SPACING.sm,
  },
  badge: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
    borderRadius: BORDER_RADIUS.full,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderWidth: 0.5,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  cacheBadge: {
    backgroundColor: `${NEURAL.tertiary}22`,
    borderColor: `${NEURAL.tertiary}60`,
  },
  confidenceBadge: {
    backgroundColor: '#f59e0b18',
    borderColor: '#f59e0b60',
  },
  badgeText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
  },
  queryText: {
    fontSize: TYPOGRAPHY.fontSize.md,
    color: SEMANTIC_COLORS.textPrimary,
    lineHeight: TYPOGRAPHY.fontSize.md * TYPOGRAPHY.lineHeight.normal,
    marginBottom: SPACING.md,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SPACING.sm,
  },
  stepInfo: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
  },
  stepCount: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  channelInfo: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
  },
  qualityGates: {
    flexDirection: "row",
    gap: SPACING.xs,
  },
  qgBadge: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
    borderRadius: BORDER_RADIUS.sm,
    backgroundColor: '#f59e0b18',
    borderWidth: 0.5,
    borderColor: '#f59e0b60',
  },
  qgText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: '#d97706',
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.md,
  },
  metaText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
});
