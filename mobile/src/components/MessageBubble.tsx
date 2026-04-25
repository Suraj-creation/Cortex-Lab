/**
 * MessageBubble — Cortex Aurora chat message
 * User: indigo gradient, right-aligned. Assistant: white card, left-aligned.
 * Expandable metadata: thinking, evidence, agents, query analysis
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Clipboard,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from './ui/AppIcon';
import { Badge } from './ui/Badge';

interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
  isStreaming?: boolean;
  thinking?: string;
  confidence?: number;
  agentsUsed?: string[];
  evidence?: string[];
  queryAnalysis?: Record<string, unknown>;
  processingTimeMs?: number;
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
  processingTimeMs,
}: MessageBubbleProps) {
  const [showMeta, setShowMeta] = useState(false);
  const isUser = role === 'user';
  const hasMetadata = !isUser && (thinking || (evidence && evidence.length > 0) || (agentsUsed && agentsUsed.length > 0) || queryAnalysis);

  const handleCopy = () => {
    if (Platform.OS === 'web') {
      navigator.clipboard?.writeText(content);
    } else {
      Clipboard.setString(content);
    }
  };

  const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

  if (isUser) {
    return (
      <View style={styles.userRow}>
        <LinearGradient
          colors={['#6366f1', '#8b5cf6']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.userBubble}
        >
          <Text style={styles.userText}>{content}</Text>
          {timeStr ? <Text style={styles.userTime}>{timeStr}</Text> : null}
        </LinearGradient>
      </View>
    );
  }

  return (
    <View style={styles.assistantRow}>
      {/* Avatar */}
      <View style={styles.avatar}>
        <AppIcon name="robot-outline" size={16} color="#6366f1" />
      </View>

      <View style={styles.assistantContent}>
        <View style={[styles.assistantBubble, isStreaming && styles.streamingBubble]}>
          {isStreaming && !content ? (
            <View style={styles.typingIndicator}>
              <View style={styles.typingDot} />
              <View style={[styles.typingDot, { opacity: 0.7 }]} />
              <View style={[styles.typingDot, { opacity: 0.4 }]} />
              <Text style={styles.typingText}>Thinking...</Text>
            </View>
          ) : (
            <Text style={styles.assistantText}>{content}</Text>
          )}

          {/* Confidence + Agents row */}
          {!isStreaming && (confidence !== undefined || (agentsUsed && agentsUsed.length > 0)) && (
            <View style={styles.badgeRow}>
              {confidence !== undefined && (
                <Badge
                  label={`${Math.round(confidence * 100)}% confidence`}
                  variant={confidence > 0.7 ? 'success' : confidence > 0.4 ? 'warning' : 'error'}
                  size="sm"
                />
              )}
              {agentsUsed && agentsUsed.length > 0 && (
                <Badge
                  label={`${agentsUsed.length} agent${agentsUsed.length !== 1 ? 's' : ''}`}
                  variant="violet"
                  size="sm"
                />
              )}
              {processingTimeMs && (
                <Badge
                  label={`${processingTimeMs}ms`}
                  variant="default"
                  size="sm"
                />
              )}
            </View>
          )}
        </View>

        {/* Action bar */}
        {!isStreaming && content && (
          <View style={styles.actionBar}>
            <TouchableOpacity onPress={handleCopy} style={styles.actionBtn}>
              <AppIcon name="content-copy" size={13} color="#94a3b8" />
            </TouchableOpacity>
            {hasMetadata && (
              <TouchableOpacity onPress={() => setShowMeta(!showMeta)} style={styles.actionBtn}>
                <AppIcon name={showMeta ? 'chevron-up' : 'chevron-down'} size={13} color="#94a3b8" />
                <Text style={styles.actionText}>Details</Text>
              </TouchableOpacity>
            )}
            {timeStr ? <Text style={styles.timeText}>{timeStr}</Text> : null}
          </View>
        )}

        {/* Expandable metadata */}
        {showMeta && (
          <View style={styles.metaContainer}>
            {thinking && (
              <View style={styles.metaSection}>
                <Text style={styles.metaLabel}>💭 Thinking</Text>
                <Text style={styles.metaContent}>{thinking}</Text>
              </View>
            )}
            {evidence && evidence.length > 0 && (
              <View style={styles.metaSection}>
                <Text style={styles.metaLabel}>📚 Evidence ({evidence.length})</Text>
                {evidence.slice(0, 5).map((e, i) => (
                  <Text key={i} style={styles.metaEvidence}>• {typeof e === 'string' ? e.slice(0, 200) : JSON.stringify(e).slice(0, 200)}</Text>
                ))}
              </View>
            )}
            {agentsUsed && agentsUsed.length > 0 && (
              <View style={styles.metaSection}>
                <Text style={styles.metaLabel}>🤖 Agents Used</Text>
                <View style={styles.agentChips}>
                  {agentsUsed.map((a, i) => (
                    <Badge key={i} label={a} variant="primary" size="sm" />
                  ))}
                </View>
              </View>
            )}
            {queryAnalysis && (
              <View style={styles.metaSection}>
                <Text style={styles.metaLabel}>🔍 Query Analysis</Text>
                <Text style={styles.metaContent}>{JSON.stringify(queryAnalysis, null, 2)}</Text>
              </View>
            )}
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  // User message
  userRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.xs,
  },
  userBubble: {
    maxWidth: '80%',
    borderRadius: RADIUS.xl,
    borderBottomRightRadius: RADIUS.sm,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    ...SHADOWS.md,
  },
  userText: {
    fontSize: FONT_SIZE.base,
    color: '#ffffff',
    lineHeight: 20,
  },
  userTime: {
    fontSize: 10,
    color: 'rgba(255, 255, 255, 0.6)',
    marginTop: 4,
    textAlign: 'right',
  },

  // Assistant message
  assistantRow: {
    flexDirection: 'row',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.xs,
    gap: SPACING.sm,
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: RADIUS.lg,
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 2,
  },
  assistantContent: {
    flex: 1,
    maxWidth: '85%',
  },
  assistantBubble: {
    backgroundColor: '#ffffff',
    borderRadius: RADIUS.xl,
    borderTopLeftRadius: RADIUS.sm,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderWidth: 1,
    borderColor: '#f1f5f9',
    ...SHADOWS.sm,
  },
  streamingBubble: {
    borderColor: '#c7d2fe',
  },
  assistantText: {
    fontSize: FONT_SIZE.base,
    color: '#1e293b',
    lineHeight: 22,
  },

  // Typing indicator
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingVertical: 4,
  },
  typingDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#a5b4fc',
  },
  typingText: {
    fontSize: FONT_SIZE.xs,
    color: '#94a3b8',
    marginLeft: 4,
  },

  // Badges row
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginTop: SPACING.sm,
  },

  // Action bar
  actionBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    gap: SPACING.sm,
    paddingLeft: SPACING.xs,
  },
  actionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    paddingVertical: 2,
    paddingHorizontal: 4,
    borderRadius: RADIUS.sm,
  },
  actionText: {
    fontSize: 10,
    color: '#94a3b8',
    fontWeight: FONT_WEIGHT.medium,
  },
  timeText: {
    fontSize: 10,
    color: '#cbd5e1',
    marginLeft: 'auto',
  },

  // Metadata
  metaContainer: {
    marginTop: SPACING.sm,
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    gap: SPACING.md,
  },
  metaSection: {
    gap: 4,
  },
  metaLabel: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#475569',
  },
  metaContent: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    lineHeight: 16,
  },
  metaEvidence: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    lineHeight: 16,
    paddingLeft: 8,
  },
  agentChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginTop: 2,
  },
});
