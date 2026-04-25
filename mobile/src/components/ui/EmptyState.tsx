/**
 * EmptyState — Cortex Aurora illustrated empty placeholder
 */
import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { FONT_SIZE, FONT_WEIGHT, SPACING, RADIUS } from '../../theme/colors';
import { AppIcon, type AppIconName } from './AppIcon';

interface EmptyStateProps {
  icon?: AppIconName;
  title: string;
  message?: string;
  action?: React.ReactNode;
  style?: ViewStyle;
}

export function EmptyState({
  icon = 'information-outline',
  title,
  message,
  action,
  style,
}: EmptyStateProps) {
  return (
    <View style={[styles.container, style]}>
      <View style={styles.iconContainer}>
        <AppIcon name={icon} size={32} color="#a5b4fc" />
      </View>
      <Text style={styles.title}>{title}</Text>
      {message && <Text style={styles.message}>{message}</Text>}
      {action && <View style={styles.action}>{action}</View>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: SPACING['5xl'],
    paddingHorizontal: SPACING['3xl'],
  },
  iconContainer: {
    width: 72,
    height: 72,
    borderRadius: RADIUS['2xl'],
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.lg,
  },
  title: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
    textAlign: 'center',
    marginBottom: SPACING.sm,
  },
  message: {
    fontSize: FONT_SIZE.base,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 20,
    maxWidth: 280,
  },
  action: {
    marginTop: SPACING.xl,
  },
});
