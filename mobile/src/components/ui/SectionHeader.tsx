/**
 * SectionHeader — Reusable section header with optional action
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ViewStyle } from 'react-native';
import { FONT_SIZE, FONT_WEIGHT, SPACING } from '../../theme/colors';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  action?: { label: string; onPress: () => void };
  icon?: React.ReactNode;
  style?: ViewStyle;
}

export function SectionHeader({ title, subtitle, action, icon, style }: SectionHeaderProps) {
  return (
    <View style={[styles.container, style]}>
      <View style={styles.left}>
        {icon && <View style={styles.icon}>{icon}</View>}
        <View>
          <Text style={styles.title}>{title}</Text>
          {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
        </View>
      </View>
      {action && (
        <TouchableOpacity onPress={action.onPress} activeOpacity={0.7}>
          <Text style={styles.actionText}>{action.label}</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: SPACING.sm,
  },
  left: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    flex: 1,
  },
  icon: {},
  title: {
    fontSize: FONT_SIZE.md,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },
  subtitle: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    marginTop: 1,
  },
  actionText: {
    fontSize: FONT_SIZE.sm,
    color: '#6366f1',
    fontWeight: FONT_WEIGHT.semibold,
  },
});
