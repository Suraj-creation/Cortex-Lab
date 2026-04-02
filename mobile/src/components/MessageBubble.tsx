import { StyleSheet, Text, View } from "react-native";
import type { EvidenceCard, QueryAnalysis } from "../../shared/core/types";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from "../theme/colors";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: number;
  isStreaming?: boolean;
  thinking?: string;
  confidence?: number;
  agentsUsed?: string[];
  evidence?: EvidenceCard[];
  queryAnalysis?: QueryAnalysis;
}

export function MessageBubble({
  role,
  content,
  timestamp,
  isStreaming = false,
  thinking,
  confidence,
  agentsUsed,
  evidence,
  queryAnalysis,
}: MessageBubbleProps) {
  const isUser = role === "user";
  const timeText = timestamp ? new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  const safeContent = content.trim();
  const evidencePreview = (evidence || []).slice(0, 2);

  return (
    <View style={[styles.row, isUser ? styles.userRow : styles.assistantRow]}>
      <View style={styles.metaRow}>
        <Text style={styles.roleText}>{isUser ? "You" : "Cortex"}</Text>
        {timeText ? <Text style={styles.timeText}>{timeText}</Text> : null}
        {!isUser && typeof confidence === "number" ? (
          <View style={styles.confidenceBadge}>
            <Text style={styles.confidenceText}>{Math.round(confidence * 100)}%</Text>
          </View>
        ) : null}
      </View>

      {!isUser && thinking ? (
        <View style={styles.reasoningCard}>
          <Text style={styles.reasoningLabel}>Reasoning</Text>
          <Text style={styles.reasoningText} numberOfLines={4}>
            {thinking}
          </Text>
        </View>
      ) : null}

      <View
        style={[
          styles.bubble,
          isUser ? styles.userBubble : styles.assistantBubble,
        ]}
      >
        <Text
          style={[
            styles.text,
            isUser ? styles.userText : styles.assistantText,
          ]}
        >
          {safeContent || (isStreaming && !isUser ? "…" : "No response generated.")}
        </Text>
        {isStreaming && !isUser ? <Text style={styles.streamingText}>Generating…</Text> : null}
      </View>

      {!isUser && agentsUsed && agentsUsed.length > 0 ? (
        <View style={styles.agentRow}>
          {agentsUsed.slice(0, 4).map((agent, idx) => (
            <View key={`${agent}-${idx}`} style={styles.agentChip}>
              <Text style={styles.agentChipText}>{agent}</Text>
            </View>
          ))}
        </View>
      ) : null}

      {!isUser && queryAnalysis ? (
        <View style={styles.analysisRow}>
          <Text style={styles.analysisChip}>Intent {queryAnalysis.intent}</Text>
          <Text style={styles.analysisChip}>Route {queryAnalysis.routing}</Text>
          <Text style={styles.analysisChip}>Complexity {(queryAnalysis.complexity * 100).toFixed(0)}%</Text>
        </View>
      ) : null}

      {!isUser && evidencePreview.length > 0 ? (
        <View style={styles.evidenceList}>
          {evidencePreview.map((item, idx) => (
            <View key={`${item.channel}-${idx}`} style={styles.evidenceCard}>
              <View style={styles.evidenceMetaRow}>
                <Text style={styles.evidenceChannel}>{item.channel || "context"}</Text>
                <Text style={styles.evidenceScore}>{Math.round((item.score || 0) * 100)}%</Text>
              </View>
              <Text style={styles.evidenceText} numberOfLines={3}>
                {item.content}
              </Text>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    width: "100%",
    marginBottom: SPACING["2xl"],
    paddingHorizontal: SPACING.xl,
  },
  userRow: {
    alignItems: "flex-end",
  },
  assistantRow: {
    alignItems: "flex-start",
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
    paddingHorizontal: SPACING.xs,
  },
  roleText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
  timeText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
  },
  confidenceBadge: {
    marginLeft: "auto",
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
    borderColor: COLORS.success[200],
    backgroundColor: COLORS.success[50],
  },
  confidenceText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.success[700],
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
  reasoningCard: {
    maxWidth: "85%",
    marginBottom: SPACING.sm,
    borderRadius: BORDER_RADIUS.xl,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderAccent,
    backgroundColor: SEMANTIC_COLORS.bgHighlight,
  },
  reasoningLabel: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.primary[700],
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    marginBottom: SPACING.xs,
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  reasoningText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textSecondary,
    lineHeight: TYPOGRAPHY.fontSize.sm * TYPOGRAPHY.lineHeight.normal,
  },
  bubble: {
    maxWidth: "85%",
    borderRadius: BORDER_RADIUS["2xl"],
    paddingVertical: SPACING.lg,
    paddingHorizontal: SPACING.xl,
  },
  userBubble: {
    backgroundColor: COLORS.primary[700],
    borderWidth: 1,
    borderColor: COLORS.primary[800],
    ...SHADOWS.md,
  },
  assistantBubble: {
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    ...SHADOWS.sm,
  },
  text: {
    fontSize: TYPOGRAPHY.fontSize.md,
    lineHeight: TYPOGRAPHY.fontSize.md * TYPOGRAPHY.lineHeight.normal,
  },
  userText: {
    color: COLORS.white,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  assistantText: {
    color: SEMANTIC_COLORS.textPrimary,
  },
  streamingText: {
    marginTop: SPACING.sm,
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  agentRow: {
    marginTop: SPACING.sm,
    maxWidth: "85%",
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.xs,
  },
  agentChip: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderAccent,
    backgroundColor: SEMANTIC_COLORS.bgHighlight,
  },
  agentChipText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.primary[700],
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
  evidenceList: {
    marginTop: SPACING.sm,
    maxWidth: "85%",
    gap: SPACING.sm,
  },
  analysisRow: {
    marginTop: SPACING.sm,
    maxWidth: "85%",
    flexDirection: "row",
    flexWrap: "wrap",
    gap: SPACING.xs,
  },
  analysisChip: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderRadius: BORDER_RADIUS.full,
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
  },
  evidenceCard: {
    borderRadius: BORDER_RADIUS.lg,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    ...SHADOWS.sm,
  },
  evidenceMetaRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: SPACING.xs,
  },
  evidenceChannel: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    textTransform: "capitalize",
  },
  evidenceScore: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.primary[700],
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
  evidenceText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    color: SEMANTIC_COLORS.textPrimary,
    lineHeight: TYPOGRAPHY.fontSize.sm * TYPOGRAPHY.lineHeight.normal,
  },
});
