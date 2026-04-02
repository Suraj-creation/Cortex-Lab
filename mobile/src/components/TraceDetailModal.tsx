import React, { useState } from "react";
import {
  Modal,
  View,
  Text,
  Pressable,
  ScrollView,
  StyleSheet,
} from "react-native";
import { PipelineTrace, PipelineStep } from "../../shared/core/types";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from "../theme/colors";

interface TraceDetailModalProps {
  trace: PipelineTrace;
  visible: boolean;
  onClose: () => void;
}

export default function TraceDetailModal({
  trace,
  visible,
  onClose,
}: TraceDetailModalProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  const toggleStep = (idx: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const renderStepIcon = (stepType: string) => {
    const iconMap: Record<string, string> = {
      query_analysis: "QRY",
      routing: "RTE",
      query_transform: "TRN",
      agent_execution: "AGN",
      crag: "CRG",
      self_rag: "SRG",
      flare: "FLR",
      compression: "CMP",
      reranking: "RNK",
    };
    return iconMap[stepType] || "STEP";
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return COLORS.success[500];
      case "skipped":
        return COLORS.surface[400];
      case "error":
        return COLORS.error[500];
      default:
        return COLORS.surface[600];
    }
  };

  return (
    <Modal
      transparent
      visible={visible}
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        <View style={styles.container}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>Trace Details</Text>
            <Pressable onPress={onClose} style={styles.closeButton}>
              <Text style={styles.closeButtonText}>X</Text>
            </Pressable>
          </View>

          <ScrollView style={styles.content}>
            {/* Trace ID and Timestamp */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Overview</Text>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Trace ID</Text>
                <Text style={styles.value} numberOfLines={1}>
                  {trace.trace_id}
                </Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Timestamp</Text>
                <Text style={styles.value}>
                  {new Date(trace.timestamp).toLocaleString()}
                </Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Duration</Text>
                <Text style={styles.value}>{trace.total_duration_ms}ms</Text>
              </View>
            </View>

            {/* Query Analysis */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Query Analysis</Text>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Query</Text>
                <Text style={styles.value}>{trace.query}</Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Intent</Text>
                <Text style={styles.value}>
                  {trace.query_analysis?.intent || "N/A"}
                </Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Complexity</Text>
                <Text style={styles.value}>
                  {trace.query_analysis?.complexity?.toFixed(2) || "N/A"}
                </Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Routing</Text>
                <Text style={styles.value}>
                  {trace.query_analysis?.routing || trace.routing_decision}
                </Text>
              </View>
            </View>

            {/* Agents */}
            {trace.agents_invoked?.length > 0 ? (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Agents Invoked</Text>
                <View style={styles.agentsList}>
                  {trace.agents_invoked.map((agent, idx) => (
                    <View key={idx} style={styles.agentBadge}>
                      <Text style={styles.agentName}>
                        {agent.agent}
                        {agent.is_primary ? " (primary)" : ""}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            ) : null}

            {/* Pipeline Steps Waterfall */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Pipeline Steps</Text>
              <View style={styles.stepsContainer}>
                {trace.steps.map((step, idx) => (
                  <StepRow
                    key={idx}
                    step={step}
                    index={idx}
                    isExpanded={expandedSteps.has(idx)}
                    onToggle={() => toggleStep(idx)}
                    icon={renderStepIcon(step.step_type)}
                    statusColor={getStatusColor(step.status)}
                  />
                ))}
              </View>
            </View>

            {/* Retrieval Channels */}
            {trace.retrieval_channels?.length > 0 ? (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Retrieval Channels</Text>
                <View style={styles.channelsGrid}>
                  {trace.retrieval_channels.map((ch, idx) => (
                    <View key={idx} style={styles.channelCard}>
                      <Text style={styles.channelName}>{ch.channel}</Text>
                      <Text style={styles.channelValue}>{ch.result_count}</Text>
                      <Text style={styles.channelMeta}>
                        score: {(ch.top_score * 100).toFixed(0)}%
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            ) : null}

            {/* Quality Gates */}
            {trace.crag_evaluation ? (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>CRAG Evaluation</Text>
                <View style={styles.qgCard}>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Verdict</Text>
                    <Text style={styles.value}>
                      {trace.crag_evaluation.verdict}
                    </Text>
                  </View>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Quality Score</Text>
                    <Text style={styles.value}>
                      {(trace.crag_evaluation.quality_score * 100).toFixed(1)}%
                    </Text>
                  </View>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Evidence Count</Text>
                    <Text style={styles.value}>
                      {trace.crag_evaluation.evidence_count}
                    </Text>
                  </View>
                </View>
              </View>
            ) : null}

            {trace.self_rag_critique ? (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Self-RAG Critique</Text>
                <View style={styles.qgCard}>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Verdict</Text>
                    <Text style={styles.value}>
                      {trace.self_rag_critique.verdict}
                    </Text>
                  </View>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Relevance</Text>
                    <Text style={styles.value}>
                      {(trace.self_rag_critique.isrel * 100).toFixed(0)}%
                    </Text>
                  </View>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Supported</Text>
                    <Text style={styles.value}>
                      {(trace.self_rag_critique.issup * 100).toFixed(0)}%
                    </Text>
                  </View>
                  <View style={styles.qgRow}>
                    <Text style={styles.label}>Useful</Text>
                    <Text style={styles.value}>
                      {(trace.self_rag_critique.isuse * 100).toFixed(0)}%
                    </Text>
                  </View>
                </View>
              </View>
            ) : null}

            {/* Summary Metrics */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Summary</Text>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Final Confidence</Text>
                <Text style={styles.value}>
                  {(trace.final_confidence * 100).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Evidence Count</Text>
                <Text style={styles.value}>{trace.evidence_count}</Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.label}>Cache Hit</Text>
                <Text style={styles.value}>
                  {trace.cache_status?.hit ? "Yes" : "No"}
                </Text>
              </View>
              {trace.token_usage?.total_tokens ? (
                <View style={styles.infoRow}>
                  <Text style={styles.label}>Tokens Used</Text>
                  <Text style={styles.value}>
                    {trace.token_usage.total_tokens}
                  </Text>
                </View>
              ) : null}
            </View>

            <View style={styles.spacer} />
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

interface StepRowProps {
  step: PipelineStep;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
  icon: string;
  statusColor: string;
}

function StepRow({
  step,
  index,
  isExpanded,
  onToggle,
  icon,
  statusColor,
}: StepRowProps) {
  const statusText =
    step.status === "completed"
      ? `${step.duration_ms}ms`
      : step.status === "skipped"
      ? "skipped"
      : "error";

  return (
    <View>
      <Pressable
        onPress={onToggle}
        style={[
          styles.stepRow,
          {
            borderLeftColor: statusColor,
            borderLeftWidth: 3,
          },
        ]}
      >
        <Text style={styles.stepIcon}>{icon}</Text>
        <View style={styles.stepInfo}>
          <Text style={styles.stepName}>{step.step_name}</Text>
          <Text style={styles.stepStatus}>{statusText}</Text>
        </View>
        <Text style={styles.expandIndicator}>{isExpanded ? "v" : ">"}</Text>
      </Pressable>

      {isExpanded && step.details && Object.keys(step.details).length > 0 ? (
        <View style={styles.stepDetails}>
          {Object.entries(step.details).map(([key, value], idx) => (
            <View key={idx} style={styles.detailRow}>
              <Text style={styles.detailLabel}>
                {key.replace(/_/g, " ")}:
              </Text>
              <Text style={styles.detailValue} numberOfLines={2}>
                {typeof value === "boolean"
                  ? value
                    ? "true"
                    : "false"
                  : typeof value === "number"
                  ? (Math.round(value * 1000) / 1000).toString()
                  : Array.isArray(value)
                  ? value.join(", ") || "N/A"
                  : String(value ?? "N/A")}
              </Text>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: SEMANTIC_COLORS.bgOverlay,
    justifyContent: "flex-end",
  },
  container: {
    backgroundColor: SEMANTIC_COLORS.bgCanvas,
    borderTopLeftRadius: BORDER_RADIUS["2xl"],
    borderTopRightRadius: BORDER_RADIUS["2xl"],
    height: "90%",
    overflow: "hidden",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.lg,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
  },
  title: {
    fontSize: TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
  },
  closeButton: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
  },
  closeButtonText: {
    fontSize: TYPOGRAPHY.fontSize.xl,
    color: SEMANTIC_COLORS.textTertiary,
  },
  content: {
    flex: 1,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.lg,
  },
  section: {
    marginBottom: SPACING.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.lg,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  sectionTitle: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.bold,
    color: SEMANTIC_COLORS.textSecondary,
    textTransform: "uppercase",
    marginBottom: SPACING.sm,
    letterSpacing: 0.5,
  },
  infoRow: {
    marginBottom: SPACING.sm,
  },
  label: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  value: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  agentsList: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  agentBadge: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    backgroundColor: COLORS.info[50],
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.info[200],
  },
  agentName: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: COLORS.info[700],
  },
  stepsContainer: {
    gap: 8,
  },
  stepRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  stepIcon: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.bold,
    color: SEMANTIC_COLORS.textSecondary,
    marginRight: SPACING.sm,
  },
  stepInfo: {
    flex: 1,
  },
  stepName: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textPrimary,
    marginBottom: SPACING.xs,
  },
  stepStatus: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
  },
  expandIndicator: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.surface[300],
  },
  stepDetails: {
    marginTop: -2,
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderBottomLeftRadius: BORDER_RADIUS.md,
    borderBottomRightRadius: BORDER_RADIUS.md,
  },
  detailRow: {
    marginBottom: 6,
  },
  detailLabel: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  detailValue: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    fontFamily: "monospace",
  },
  channelsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  channelCard: {
    flex: 1,
    minWidth: "45%",
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    alignItems: "center",
  },
  channelName: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
    textTransform: "capitalize",
    marginBottom: SPACING.xs,
  },
  channelValue: {
    fontSize: TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.bold,
    color: COLORS.primary[600],
    marginBottom: SPACING.xs,
  },
  channelMeta: {
    fontSize: 9,
    color: SEMANTIC_COLORS.textTertiary,
  },
  qgCard: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.md,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  qgRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SPACING.sm,
    paddingBottom: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surface[100],
  },
  spacer: {
    height: 24,
  },
});
