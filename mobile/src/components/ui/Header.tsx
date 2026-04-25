/**
 * Header — Cortex Aurora light top bar
 * Clean white glass header with subtle shadow, indigo accents
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { NEURAL, RADIUS, FONT_SIZE, FONT_WEIGHT, SPACING, SHADOWS } from '../../theme/colors';
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
  const isLoading = modelStatus?.status === 'loading';
  const isOnline = Boolean(modelStatus?.status) && modelStatus?.status !== 'offline';
  const statusLabel = isLoading ? 'Syncing' : isOnline ? 'Connected' : 'Offline';
  const statusColor = isLoading ? '#6366f1' : isOnline ? '#059669' : '#64748b';

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

      {/* Center: logo + title */}
      <View style={styles.center}>
        <View style={styles.titleRow}>
          <View style={styles.logoContainer}>
            <View style={styles.logoDot} />
          </View>
          <Text style={styles.title} numberOfLines={1}>{title}</Text>
        </View>
        {subtitle ? (
          <Text style={styles.subtitle} numberOfLines={1}>{subtitle}</Text>
        ) : null}
      </View>

      {/* Right: status + settings */}
      <View style={styles.right}>
        <View style={[
          styles.statusPill,
          isOnline && !isLoading && styles.statusPillOnline,
          !isOnline && !isLoading && styles.statusPillOffline,
        ]}>
          <NeuralPulse
            active={!isLoading}
            color={isLoading ? '#6366f1' : isOnline ? '#10b981' : '#94a3b8'}
            size={5}
          />
          <Text style={[
            styles.statusText,
            { color: statusColor },
          ]}>
            {statusLabel}
          </Text>
        </View>
        <TouchableOpacity
          style={styles.settingsButton}
          onPress={onSettingsPress}
          activeOpacity={0.7}
          accessibilityLabel="Open settings"
        >
          <AppIcon name="cog-outline" size={18} color="#475569" />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e9eef8',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    gap: SPACING.sm,
  },
  iconButton: {
    padding: SPACING.sm,
    borderRadius: RADIUS.lg,
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
  hamburger: {
    gap: 4,
    alignItems: 'flex-start',
  },
  hamburgerLine: {
    width: 20,
    height: 2,
    backgroundColor: '#334155',
    borderRadius: 1,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: SPACING.sm,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  logoContainer: {
    width: 22,
    height: 22,
    borderRadius: 8,
    backgroundColor: '#6b79ff',
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.glow,
  },
  logoDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ffffff',
  },
  title: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
    letterSpacing: -0.3,
  },
  subtitle: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    marginTop: 1,
    fontWeight: FONT_WEIGHT.medium,
  },
  right: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#edf2fb',
    borderRadius: RADIUS.full,
    paddingHorizontal: 10,
    paddingVertical: 5,
    gap: 4,
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
  statusPillOnline: {
    backgroundColor: '#f0fdf4',
    borderColor: '#bbf7d0',
  },
  statusPillOffline: {
    backgroundColor: '#f8fafc',
    borderColor: '#e2e8f0',
  },
  statusText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
  },
  settingsButton: {
    padding: SPACING.sm,
    borderRadius: RADIUS.lg,
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
});
