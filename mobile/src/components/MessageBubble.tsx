/**
 * MessageBubble - Cortex Aurora chat message.
 */
import React, { useMemo, useState } from 'react';
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
import { AppIcon, type AppIconName } from './ui/AppIcon';
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

interface DetailCardData {
  title: string;
  body: string;
  caption?: string;
  icon: AppIconName;
}

function titleize(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function compactValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => compactValue(item))
      .filter(Boolean)
      .join(', ');
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function truncate(value: string, maxLength: number = 220): string {
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 1).trim()}...`;
}

function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return trimmed;
  }
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return trimmed;
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

function normalizeEvidence(items: unknown[] | undefined): DetailCardData[] {
  if (!items || items.length === 0) {
    return [];
  }

  return items.slice(0, 6).map((item, index) => {
    const parsed = parseMaybeJson(item);

    if (typeof parsed === 'string') {
      return {
        title: `Evidence ${index + 1}`,
        body: truncate(parsed),
        icon: 'file-document-outline',
      };
    }

    if (Array.isArray(parsed)) {
      return {
        title: `Evidence ${index + 1}`,
        body: truncate(compactValue(parsed)),
        icon: 'file-document-outline',
      };
    }

    if (parsed && typeof parsed === 'object') {
      const record = parsed as Record<string, unknown>;
      const title =
        compactValue(record.title) ||
        compactValue(record.source) ||
        compactValue(record.entity) ||
        compactValue(record.id) ||
        `Evidence ${index + 1}`;
      const body =
        compactValue(record.snippet) ||
        compactValue(record.text) ||
        compactValue(record.summary) ||
        compactValue(record.content) ||
        compactValue(record.body) ||
        compactValue(record.payload);
      const captionParts = [
        compactValue(record.type),
        compactValue(record.relation),
        compactValue(record.memory_id || record.memoryId),
      ].filter(Boolean);

      return {
        title: truncate(title, 72),
        body: truncate(body || compactValue(record)),
        caption: captionParts.length ? truncate(captionParts.join(' | '), 80) : undefined,
        icon: 'file-document-outline',
      };
    }

    return {
      title: `Evidence ${index + 1}`,
      body: truncate(compactValue(parsed)),
      icon: 'file-document-outline',
    };
  });
}

function normalizeAnalysisRows(queryAnalysis?: Record<string, unknown>): Array<{ label: string; value: string }> {
  if (!queryAnalysis) {
    return [];
  }

  return Object.entries(queryAnalysis)
    .map(([key, value]) => ({
      label: titleize(key),
      value: truncate(compactValue(parseMaybeJson(value)), 160),
    }))
    .filter((item) => item.value.length > 0);
}

function normalizeThinkingCards(thinking?: string): DetailCardData[] {
  if (!thinking?.trim()) {
    return [];
  }

  return [
    {
      title: 'Reasoning Trace',
      body: truncate(thinking, 260),
      icon: 'brain',
    },
  ];
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
  const evidenceCards = useMemo(() => normalizeEvidence(evidence as unknown[] | undefined), [evidence]);
  const thinkingCards = useMemo(() => normalizeThinkingCards(thinking), [thinking]);
  const analysisRows = useMemo(() => normalizeAnalysisRows(queryAnalysis), [queryAnalysis]);
  const hasMetadata =
    !isUser &&
    (thinkingCards.length > 0 || evidenceCards.length > 0 || (agentsUsed && agentsUsed.length > 0) || analysisRows.length > 0);

  const handleCopy = () => {
    if (Platform.OS === 'web') {
      navigator.clipboard?.writeText(content);
    } else {
      Clipboard.setString(content);
    }
  };

  const timeStr = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

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

          {!isStreaming && (confidence !== undefined || (agentsUsed && agentsUsed.length > 0) || processingTimeMs) && (
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
              {processingTimeMs ? <Badge label={`${processingTimeMs}ms`} variant="default" size="sm" /> : null}
            </View>
          )}
        </View>

        {!isStreaming && content && (
          <View style={styles.actionBar}>
            <TouchableOpacity onPress={handleCopy} style={styles.actionBtn}>
              <AppIcon name="content-copy" size={13} color="#94a3b8" />
            </TouchableOpacity>
            {hasMetadata ? (
              <TouchableOpacity onPress={() => setShowMeta((value) => !value)} style={styles.actionBtn}>
                <AppIcon name={showMeta ? 'chevron-up' : 'chevron-down'} size={13} color="#94a3b8" />
                <Text style={styles.actionText}>Details</Text>
              </TouchableOpacity>
            ) : null}
            {timeStr ? <Text style={styles.timeText}>{timeStr}</Text> : null}
          </View>
        )}

        {showMeta && (
          <View style={styles.metaContainer}>
            <View style={styles.metaHeader}>
              <View style={styles.metaHeaderTitle}>
                <AppIcon name="text-search" size={14} color="#475569" />
                <Text style={styles.metaHeaderText}>Response Details</Text>
              </View>
              <Badge label="Grounding" variant="default" size="sm" />
            </View>

            {thinkingCards.length > 0 ? (
              <View style={styles.metaSection}>
                <View style={styles.metaTitleRow}>
                  <AppIcon name="brain" size={14} color="#6366f1" />
                  <Text style={styles.metaSectionTitle}>Reasoning</Text>
                </View>
                {thinkingCards.map((card, index) => (
                  <View key={`thinking-${index}`} style={styles.detailCard}>
                    <View style={styles.detailCardIcon}>
                      <AppIcon name={card.icon} size={14} color="#6366f1" />
                    </View>
                    <View style={styles.detailCardBody}>
                      <Text style={styles.detailCardTitle}>{card.title}</Text>
                      <Text style={styles.detailCardText}>{card.body}</Text>
                      {card.caption ? <Text style={styles.detailCardCaption}>{card.caption}</Text> : null}
                    </View>
                  </View>
                ))}
              </View>
            ) : null}

            {evidenceCards.length > 0 ? (
              <View style={styles.metaSection}>
                <View style={styles.metaTitleRow}>
                  <AppIcon name="file-document-outline" size={14} color="#6366f1" />
                  <Text style={styles.metaSectionTitle}>Evidence</Text>
                  <Badge label={`${evidenceCards.length}`} variant="primary" size="sm" />
                </View>
                {evidenceCards.map((card, index) => (
                  <View key={`evidence-${index}`} style={styles.detailCard}>
                    <View style={styles.detailCardIcon}>
                      <AppIcon name={card.icon} size={14} color="#6366f1" />
                    </View>
                    <View style={styles.detailCardBody}>
                      <Text style={styles.detailCardTitle}>{card.title}</Text>
                      <Text style={styles.detailCardText}>{card.body}</Text>
                      {card.caption ? <Text style={styles.detailCardCaption}>{card.caption}</Text> : null}
                    </View>
                  </View>
                ))}
              </View>
            ) : null}

            {agentsUsed && agentsUsed.length > 0 ? (
              <View style={styles.metaSection}>
                <View style={styles.metaTitleRow}>
                  <AppIcon name="robot-outline" size={14} color="#6366f1" />
                  <Text style={styles.metaSectionTitle}>Agents</Text>
                </View>
                <View style={styles.agentChips}>
                  {agentsUsed.map((agent, index) => (
                    <Badge key={`${agent}-${index}`} label={titleize(agent)} variant="primary" size="sm" />
                  ))}
                </View>
              </View>
            ) : null}

            {analysisRows.length > 0 ? (
              <View style={styles.metaSection}>
                <View style={styles.metaTitleRow}>
                  <AppIcon name="magnify" size={14} color="#6366f1" />
                  <Text style={styles.metaSectionTitle}>Query Analysis</Text>
                </View>
                <View style={styles.analysisGrid}>
                  {analysisRows.map((row) => (
                    <View key={row.label} style={styles.analysisRow}>
                      <Text style={styles.analysisLabel}>{row.label}</Text>
                      <Text style={styles.analysisValue}>{row.value}</Text>
                    </View>
                  ))}
                </View>
              </View>
            ) : null}
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
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
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    marginTop: SPACING.sm,
  },
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
  metaContainer: {
    marginTop: SPACING.sm,
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS.lg,
    padding: SPACING.md,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    gap: SPACING.md,
  },
  metaHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: SPACING.sm,
  },
  metaHeaderTitle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaHeaderText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
    letterSpacing: 0.2,
  },
  metaSection: {
    gap: SPACING.sm,
  },
  metaTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaSectionTitle: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#475569',
  },
  detailCard: {
    flexDirection: 'row',
    gap: SPACING.sm,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#ffffff',
    padding: SPACING.sm,
  },
  detailCardIcon: {
    width: 28,
    height: 28,
    borderRadius: RADIUS.md,
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  detailCardBody: {
    flex: 1,
    gap: 3,
  },
  detailCardTitle: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#334155',
  },
  detailCardText: {
    fontSize: FONT_SIZE.xs,
    lineHeight: 18,
    color: '#64748b',
  },
  detailCardCaption: {
    fontSize: 10,
    color: '#94a3b8',
  },
  agentChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  analysisGrid: {
    gap: 8,
  },
  analysisRow: {
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#ffffff',
    paddingHorizontal: SPACING.sm,
    paddingVertical: 10,
    gap: 4,
  },
  analysisLabel: {
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    color: '#94a3b8',
    fontWeight: FONT_WEIGHT.semibold,
  },
  analysisValue: {
    fontSize: FONT_SIZE.xs,
    lineHeight: 18,
    color: '#475569',
  },
});
