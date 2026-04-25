/**
 * Badge — Cortex Aurora status / label badges
 */
import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { FONT_SIZE, FONT_WEIGHT, RADIUS } from '../../theme/colors';

export type BadgeVariant = 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info' | 'violet';

interface BadgeProps {
  label: string;
  variant?: BadgeVariant;
  size?: 'sm' | 'md';
  style?: ViewStyle;
  dot?: boolean;
}

const BADGE_CONFIG: Record<BadgeVariant, { bg: string; text: string; border: string; dot: string }> = {
  default: { bg: '#f1f5f9', text: '#475569', border: '#e2e8f0', dot: '#94a3b8' },
  primary: { bg: '#eef2ff', text: '#4338ca', border: '#c7d2fe', dot: '#6366f1' },
  success: { bg: '#f0fdf4', text: '#065f46', border: '#bbf7d0', dot: '#10b981' },
  warning: { bg: '#fffbeb', text: '#92400e', border: '#fde68a', dot: '#f59e0b' },
  error:   { bg: '#fff1f2', text: '#9f1239', border: '#fecdd3', dot: '#f43f5e' },
  info:    { bg: '#eff6ff', text: '#1e40af', border: '#bfdbfe', dot: '#3b82f6' },
  violet:  { bg: '#f5f3ff', text: '#5b21b6', border: '#ddd6fe', dot: '#8b5cf6' },
};

export function Badge({ label, variant = 'default', size = 'sm', style, dot = false }: BadgeProps) {
  const cfg = BADGE_CONFIG[variant];
  const isSmall = size === 'sm';

  return (
    <View style={[
      styles.badge,
      {
        backgroundColor: cfg.bg,
        borderColor: cfg.border,
        paddingHorizontal: isSmall ? 6 : 10,
        paddingVertical: isSmall ? 2 : 4,
      },
      style,
    ]}>
      {dot && <View style={[styles.dot, { backgroundColor: cfg.dot }]} />}
      <Text style={[
        styles.text,
        { color: cfg.text, fontSize: isSmall ? 10 : FONT_SIZE.xs },
      ]}>
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: RADIUS.full,
    borderWidth: 1,
    gap: 4,
  },
  dot: {
    width: 5,
    height: 5,
    borderRadius: 2.5,
  },
  text: {
    fontWeight: FONT_WEIGHT.semibold,
  },
});
