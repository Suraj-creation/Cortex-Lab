/**
 * OnboardingScreen — First-launch product introduction
 * Stitch ref: 49def0b9aed64e78bf4c6daad93eb821
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { AppIcon, type AppIconName } from '../components/ui/AppIcon';

interface OnboardingScreenProps {
  onContinue: () => void;
}

const HIGHLIGHTS: Array<{ iconName: AppIconName; title: string; description: string }> = [
  {
    iconName: 'chat-processing-outline',
    title: 'AI Chat',
    description: 'Switch providers and use streaming responses with RAG context.',
  },
  {
    iconName: 'brain',
    title: 'Memory Browser',
    description: 'Search, save, and manage long-term memory records.',
  },
  {
    iconName: 'graph-outline',
    title: 'Knowledge Graph',
    description: 'Inspect entities, relationships, and graph distribution.',
  },
  {
    iconName: 'microphone-outline',
    title: 'Ambient Voice',
    description: 'Run always-on listening, enrollment, and TTS checks.',
  },
];

export function OnboardingScreen({ onContinue }: OnboardingScreenProps) {
  return (
    <View style={styles.backdrop}>
      <Card variant="elevated" style={styles.card}>
        <View style={styles.heroRow}>
          <View style={styles.heroIconWrap}>
            <AppIcon name="brain" size={20} color={NEURAL.primary} />
          </View>
          <View style={styles.heroTextWrap}>
            <Text style={styles.title}>Welcome to Cortex Lab</Text>
            <Text style={styles.subtitle}>Production mobile workspace for RAG, memory, graph, and voice.</Text>
          </View>
        </View>

        <View style={styles.featureList}>
          {HIGHLIGHTS.map((item) => (
            <View key={item.title} style={styles.featureRow}>
              <View style={styles.featureIconWrap}>
                <AppIcon name={item.iconName} size={16} color={NEURAL.secondary} />
              </View>
              <View style={styles.featureTextWrap}>
                <Text style={styles.featureTitle}>{item.title}</Text>
                <Text style={styles.featureBody}>{item.description}</Text>
              </View>
            </View>
          ))}
        </View>

        <Button label="Start Exploring" onPress={onContinue} fullWidth />
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    zIndex: 40,
    justifyContent: 'center',
    padding: SPACING.lg,
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
  },
  card: {
    gap: SPACING.lg,
    borderRadius: RADIUS['2xl'],
  },
  heroRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
  },
  heroIconWrap: {
    width: 44,
    height: 44,
    borderRadius: RADIUS.full,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: `${NEURAL.primary}20`,
    borderWidth: 1,
    borderColor: `${NEURAL.primary}60`,
  },
  heroTextWrap: { flex: 1 },
  title: {
    fontSize: FONT_SIZE['2xl'],
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurface,
  },
  subtitle: {
    marginTop: 4,
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    lineHeight: FONT_SIZE.sm * 1.5,
  },
  featureList: {
    gap: SPACING.sm,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    backgroundColor: `${NEURAL.surfaceContainerHigh}80`,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}55`,
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.sm,
  },
  featureIconWrap: {
    width: 28,
    alignItems: 'center',
    paddingTop: 2,
  },
  featureTextWrap: { flex: 1 },
  featureTitle: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurface,
  },
  featureBody: {
    marginTop: 2,
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    lineHeight: FONT_SIZE.sm * 1.45,
  },
});
