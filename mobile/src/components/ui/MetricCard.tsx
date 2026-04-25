/**
 * MetricCard — Stat display card with label, value, icon support
 */
import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { RADIUS, FONT_SIZE, FONT_WEIGHT, SPACING, SHADOWS } from '../../theme/colors';

type MetricTone = 'default' | 'indigo' | 'emerald' | 'amber' | 'rose' | 'violet' | 'blue';

interface MetricCardProps {
  label: string;
  value: string | number;
  tone?: MetricTone;
  icon?: React.ReactNode;
  subtitle?: string;
  style?: ViewStyle;
  compact?: boolean;
}

const TONE_CONFIG: Record<MetricTone, { bg: string; text: string; iconBg: string }> = {
  default: { bg: '#ffffff', text: '#0f172a', iconBg: '#f1f5f9' },
  indigo:  { bg: '#eef2ff', text: '#4338ca', iconBg: '#e0e7ff' },
  emerald: { bg: '#f0fdf4', text: '#065f46', iconBg: '#d1fae5' },
  amber:   { bg: '#fffbeb', text: '#92400e', iconBg: '#fef3c7' },
  rose:    { bg: '#fff1f2', text: '#9f1239', iconBg: '#ffe4e6' },
  violet:  { bg: '#f5f3ff', text: '#5b21b6', iconBg: '#ede9fe' },
  blue:    { bg: '#eff6ff', text: '#1e40af', iconBg: '#dbeafe' },
};

export function MetricCard({
  label,
  value,
  tone = 'default',
  icon,
  subtitle,
  style,
  compact = false,
}: MetricCardProps) {
  const cfg = TONE_CONFIG[tone];

  return (
    <View style={[
      styles.container,
      { backgroundColor: cfg.bg },
      compact && styles.compact,
      style,
    ]}>
      {icon && (
        <View style={[styles.iconContainer, { backgroundColor: cfg.iconBg }]}>
          {icon}
        </View>
      )}
      <View style={styles.content}>
        <Text style={[styles.label]}>{label}</Text>
        <Text style={[styles.value, { color: cfg.text }]}>{value}</Text>
        {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    borderWidth: 1,
    borderColor: '#f1f5f9',
    ...SHADOWS.sm,
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
  },
  compact: {
    padding: SPACING.md,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: RADIUS.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    flex: 1,
  },
  label: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.medium,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  value: {
    fontSize: FONT_SIZE.xl,
    fontWeight: FONT_WEIGHT.bold,
    marginTop: 2,
  },
  subtitle: {
    fontSize: FONT_SIZE.xs,
    color: '#94a3b8',
    marginTop: 2,
  },
});
