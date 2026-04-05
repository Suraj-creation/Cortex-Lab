/**
 * MessageBubble — Neural Dark Chat Message
 * Stitch design: user=indigo gradient bubble, assistant=dark card with reasoning block
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NEURAL, RADIUS, FONT_SIZE, FONT_WEIGHT, SPACING } from '../theme/colors';
import { Badge } from './ui/Badge';
import { NeuralPulse } from './ui/NeuralPulse';

interface Evidence {
  content?: string;
  source?: string;
  score?: number;
}

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
  isStreaming?: boolean;
  thinking?: string;
  confidence?: number;
  agentsUsed?: string[];
  evidence?: Evidence[];
  queryAnalysis?: string;
}

function formatTs(ts?: number) {
  if (!ts) return '';
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function toPercent(v?: number) {
  if (typeof v !== 'number') return null;
  return `${Math.round(v * 100)}%`;
}

export function MessageBubble({
  role,
  content,
  timestamp,
  isStreaming,
  thinking,
  confidence,
  agentsUsed,
  evidence,
  queryAnalysis,
}: MessageBubbleProps) {
  const [showThinking, setShowThinking] = useState(false);

  if (role === 'user') {
    return (
      <View style={styles.userRow}>
        <LinearGradient
          colors={[NEURAL.primary, NEURAL.primaryDim]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.userBubble}
        >
          <Text style={styles.userText}>{content}</Text>
          {timestamp ? (
            <Text style={styles.userTime}>{formatTs(timestamp)}</Text>
          ) : null}
        </LinearGradient>
      </View>
    );
  }

  // Assistant message
  return (
    <View style={styles.assistantRow}>
      {/* Avatar dot */}
      <View style={styles.avatarWrap}>
        <LinearGradient
          colors={[NEURAL.secondary, NEURAL.primaryDim]}
          style={styles.avatar}
        >
          <Text style={styles.avatarText}>C</Text>
        </LinearGradient>
      </View>

      <View style={styles.assistantContent}>
        {/* Reasoning block */}
        {thinking ? (
          <View style={styles.reasoningBlock}>
            <Pressable
              onPress={() => setShowThinking((prev) => !prev)}
              style={styles.reasoningHeader}
            >
              <Text style={styles.reasoningLabel}>Reasoning</Text>
              {confidence != null && (
                <Badge
                  label={`${Math.round(confidence * 100)}% confident`}
                  variant="tertiary"
                  small
                />
              )}
            </Pressable>
            {showThinking && (
              <Text style={styles.reasoningText}>{thinking}</Text>
            )}
          </View>
        ) : null}

        {/* Main response card */}
        <View style={styles.assistantCard}>
          {isStreaming && !content ? (
            <View style={styles.streamingRow}>
              <NeuralPulse active size={6} color={NEURAL.primary} />
              <Text style={styles.streamingLabel}>Thinking...</Text>
            </View>
          ) : (
            <Text style={styles.assistantText}>
              {content}
              {isStreaming ? (
                <Text style={{ color: NEURAL.primary }}> |</Text>
              ) : null}
            </Text>
          )}

          {/* Confidence badge (bottom right of card) */}
          {!isStreaming && confidence != null && (
            <View style={styles.confidenceRow}>
              <Badge
                label={`${Math.round(confidence * 100)}% confidence`}
                variant="tertiary"
                small
              />
            </View>
          )}
        </View>

        {/* Agent chips */}
        {agentsUsed && agentsUsed.length > 0 && !isStreaming ? (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.agentChipList}
          >
            {agentsUsed.map((agent, i) => (
              <View key={i} style={styles.agentChip}>
                <Text style={styles.agentChipText}>{agent.replace(/_/g, ' ')}</Text>
              </View>
            ))}
          </ScrollView>
        ) : null}

        {/* Evidence cards */}
        {evidence && evidence.length > 0 && !isStreaming ? (
          <View style={styles.evidenceSection}>
            <Text style={styles.evidenceLabel}>Evidence</Text>
            {evidence.slice(0, 3).map((ev, i) => (
              <View key={i} style={styles.evidenceCard}>
                <View style={styles.evidenceCardHeader}>
                  {ev.source ? (
                    <Text style={styles.evidenceSource} numberOfLines={1}>{ev.source}</Text>
                  ) : null}
                  {ev.score != null ? (
                    <Badge label={`${Math.round(ev.score * 100)}%`} variant="info" small />
                  ) : null}
                </View>
                {ev.content ? (
                  <Text style={styles.evidenceText} numberOfLines={3}>{ev.content}</Text>
                ) : null}
              </View>
            ))}
          </View>
        ) : null}

        {/* Query analysis */}
        {queryAnalysis && !isStreaming ? (
          <View style={styles.queryAnalysisChip}>
            <Text style={styles.queryAnalysisText} numberOfLines={1}>
              {typeof queryAnalysis === 'string' ? queryAnalysis : (queryAnalysis as { intent?: string; routing?: string }).intent || (queryAnalysis as { routing?: string }).routing || 'Query analyzed'}
            </Text>
          </View>
        ) : null}

        {timestamp ? (
          <Text style={styles.assistantTime}>{formatTs(timestamp)}</Text>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  // ─── User ─────────────────────────────────────────────────────────────
  userRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginVertical: SPACING.sm,
    paddingHorizontal: SPACING.lg,
  },
  userBubble: {
    maxWidth: '80%',
    borderRadius: RADIUS['2xl'],
    borderBottomRightRadius: RADIUS.sm,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm + 2,
  },
  userText: {
    fontSize: FONT_SIZE.base,
    color: '#ffffff',
    lineHeight: FONT_SIZE.base * 1.55,
    fontWeight: FONT_WEIGHT.medium,
  },
  userTime: {
    fontSize: FONT_SIZE.xs,
    color: 'rgba(255,255,255,0.65)',
    textAlign: 'right',
    marginTop: 4,
  },

  // ─── Assistant ─────────────────────────────────────────────────────────
  assistantRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginVertical: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    gap: SPACING.sm,
  },
  avatarWrap: {},
  avatar: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 4,
  },
  avatarText: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.bold,
    color: '#ffffff',
  },
  assistantContent: {
    flex: 1,
    gap: SPACING.sm,
  },

  // Reasoning block
  reasoningBlock: {
    backgroundColor: NEURAL.surfaceContainerLow,
    borderRadius: RADIUS.lg,
    borderLeftWidth: 2,
    borderLeftColor: NEURAL.secondary,
    overflow: 'hidden',
  },
  reasoningHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  reasoningLabel: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.secondary,
  },
  reasoningText: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    paddingHorizontal: SPACING.md,
    paddingBottom: SPACING.md,
    lineHeight: FONT_SIZE.sm * 1.6,
  },

  // Main card
  assistantCard: {
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderRadius: RADIUS.xl,
    borderBottomLeftRadius: RADIUS.sm,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.md,
  },
  streamingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  streamingLabel: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.primary,
    fontWeight: FONT_WEIGHT.medium,
  },
  assistantText: {
    fontSize: FONT_SIZE.base,
    color: NEURAL.onSurface,
    lineHeight: FONT_SIZE.base * 1.65,
  },
  confidenceRow: {
    marginTop: SPACING.sm,
    alignItems: 'flex-end',
  },

  // Agent chips
  agentChipList: {
    gap: SPACING.sm,
  },
  agentChip: {
    backgroundColor: `${NEURAL.secondary}20`,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: `${NEURAL.secondary}40`,
    paddingHorizontal: SPACING.sm + 2,
    paddingVertical: 3,
  },
  agentChipText: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.secondary,
    fontWeight: FONT_WEIGHT.semibold,
    textTransform: 'capitalize',
  },

  // Evidence
  evidenceSection: { gap: SPACING.sm },
  evidenceLabel: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurfaceVariant,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  evidenceCard: {
    backgroundColor: NEURAL.surfaceContainer,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    padding: SPACING.sm + 2,
  },
  evidenceCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  evidenceSource: {
    flex: 1,
    fontSize: FONT_SIZE.xs,
    color: NEURAL.primary,
    fontWeight: FONT_WEIGHT.medium,
    marginRight: 6,
  },
  evidenceText: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    lineHeight: FONT_SIZE.sm * 1.5,
  },

  // Query analysis
  queryAnalysisChip: {
    backgroundColor: `${NEURAL.primary}18`,
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.md,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: `${NEURAL.primary}30`,
    alignSelf: 'flex-start',
  },
  queryAnalysisText: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.primary,
    fontWeight: FONT_WEIGHT.medium,
  },
  assistantTime: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.outline,
    marginTop: 2,
  },
});
