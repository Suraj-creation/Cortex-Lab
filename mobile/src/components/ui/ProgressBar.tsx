/**
 * ProgressBar — Neural Dark style
 * Rounded pill with gradient fill using expo-linear-gradient
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NEURAL, SEMANTIC, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../../theme/colors';

interface ProgressBarProps {
  value: number; // 0–1
  total?: number;
  label?: string;
  showLabel?: boolean;
  height?: number;
  color?: [string, string];
  style?: object;
}

export function ProgressBar({
  value,
  total,
  label,
  showLabel = true,
  height = 6,
  color = [NEURAL.primary, NEURAL.primaryDim],
  style,
}: ProgressBarProps) {
  const pct = Math.min(1, Math.max(0, value));

  return (
    <View style={[styles.container, style]}>
      {(label || showLabel) && (
        <View style={styles.labelRow}>
          {label && <Text style={styles.label}>{label}</Text>}
          {showLabel && (
            <Text style={styles.pct}>
              {total != null
                ? `${Math.round(pct * (total ?? 0))} / ${total}`
                : `${Math.round(pct * 100)}%`}
            </Text>
          )}
        </View>
      )}
      <View style={[styles.track, { height }]}>
        <LinearGradient
          colors={color}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={[styles.fill, { width: `${pct * 100}%`, height, borderRadius: RADIUS.full }]}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: '100%' },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  label: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    fontWeight: FONT_WEIGHT.medium,
  },
  pct: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.primary,
    fontWeight: FONT_WEIGHT.semibold,
  },
  track: {
    backgroundColor: NEURAL.outlineVariant,
    borderRadius: RADIUS.full,
    overflow: 'hidden',
  },
  fill: {
    borderRadius: RADIUS.full,
  },
});
