/**
 * Header — Neural Dark top bar
 * Stitch Chat Screen: model status, title, hamburger → drawer, gear → settings
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { NEURAL, RADIUS, FONT_SIZE, FONT_WEIGHT, SPACING } from '../../theme/colors';
import { NeuralPulse } from './NeuralPulse';
import { AppIcon } from './AppIcon';
import type { ModelStatus } from '../../../shared/core/types';

interface HeaderProps {
  title?: string;
  subtitle?: string;
  modelStatus?: ModelStatus;
  onMenuPress?: () => void;
  onSettingsPress?: () => void;
}

export function Header({
  title = 'Cortex Lab',
  subtitle,
  modelStatus,
  onMenuPress,
  onSettingsPress,
}: HeaderProps) {
  const isOnline =
    modelStatus?.status === 'ready' ||
    modelStatus?.status === 'gemini' ||
    (modelStatus?.model_loaded ?? false);
  const isLoading = modelStatus?.status === 'loading';

  return (
    <View style={styles.container}>
      {/* Left: hamburger */}
      <TouchableOpacity
        style={styles.iconButton}
        onPress={onMenuPress}
        activeOpacity={0.7}
        accessibilityLabel="Open conversation history"
      >
        <View style={styles.hamburger}>
          <View style={styles.hamburgerLine} />
          <View style={[styles.hamburgerLine, { width: 16 }]} />
          <View style={styles.hamburgerLine} />
        </View>
      </TouchableOpacity>

      {/* Center: title + subtitle */}
      <View style={styles.center}>
        <Text style={styles.title} numberOfLines={1}>{title}</Text>
        {subtitle ? (
          <Text style={styles.subtitle} numberOfLines={1}>{subtitle}</Text>
        ) : null}
      </View>

      {/* Right: status + settings */}
      <View style={styles.right}>
        <View style={styles.statusPill}>
          <NeuralPulse
            active={!isLoading}
            color={isOnline ? NEURAL.tertiary : NEURAL.onSurfaceVariant}
            size={6}
          />
          <Text style={[styles.statusText, { color: isOnline ? NEURAL.tertiary : NEURAL.onSurfaceVariant }]}>
            {isLoading ? 'Loading' : isOnline ? 'Online' : 'Offline'}
          </Text>
        </View>
        <TouchableOpacity
          style={styles.iconButton}
          onPress={onSettingsPress}
          activeOpacity={0.7}
          accessibilityLabel="Open settings"
        >
          <AppIcon name="cog-outline" size={18} color={NEURAL.onSurfaceVariant} style={styles.settingsIcon} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: NEURAL.surfaceContainerLow,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderBottomWidth: 0, // No-line rule — color shift defines edge
  },
  iconButton: {
    padding: SPACING.sm,
    borderRadius: RADIUS.md,
  },
  hamburger: {
    gap: 4,
    alignItems: 'flex-start',
  },
  hamburgerLine: {
    width: 20,
    height: 2,
    backgroundColor: NEURAL.onSurfaceVariant,
    borderRadius: 1,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: SPACING.sm,
  },
  title: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurface,
    letterSpacing: -0.3,
  },
  subtitle: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.onSurfaceVariant,
    marginTop: 1,
  },
  right: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: NEURAL.surfaceContainerHighest,
    borderRadius: RADIUS.full,
    paddingHorizontal: 8,
    paddingVertical: 4,
    gap: 2,
  },
  statusText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
  },
  settingsIcon: {
    marginVertical: 1,
  },
});
