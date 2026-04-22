/**
 * Badge — Neural Dark chip/badge component
 * Used for: type labels, status indicators, confidence chips
 */
import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { NEURAL, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../../theme/colors';

export type BadgeVariant =
  | 'primary'
  | 'secondary'
  | 'success'
  | 'warning'
  | 'error'
  | 'info'
  | 'ghost'
  | 'tertiary';

interface BadgeProps {
  label: string;
  variant?: BadgeVariant;
  small?: boolean;
  dot?: boolean;
  style?: ViewStyle;
}

const BG: Record<BadgeVariant, string> = {
  primary:   `${NEURAL.primary}26`,    // 15% opacity
  secondary: `${NEURAL.secondary}26`,
  success:   `${NEURAL.tertiary}26`,
  warning:   '#f59e0b26',
  error:     `${NEURAL.error}26`,
  info:      `${NEURAL.primary}1a`,
  ghost:     NEURAL.outlineVariant + '22',
  tertiary:  `${NEURAL.tertiary}26`,
};

const FG: Record<BadgeVariant, string> = {
  primary:   NEURAL.primary,
  secondary: NEURAL.secondary,
  success:   NEURAL.tertiary,
  warning:   '#f59e0b',
  error:     NEURAL.error,
  info:      NEURAL.primary,
  ghost:     NEURAL.onSurfaceVariant,
  tertiary:  NEURAL.tertiary,
};

const BORDER: Record<BadgeVariant, string> = {
  primary:   `${NEURAL.primary}40`,
  secondary: `${NEURAL.secondary}40`,
  success:   `${NEURAL.tertiary}40`,
  warning:   '#f59e0b40',
  error:     `${NEURAL.error}40`,
  info:      `${NEURAL.primary}30`,
  ghost:     NEURAL.outlineVariant,
  tertiary:  `${NEURAL.tertiaryDim}60`,
};

export function Badge({ label, variant = 'primary', small = false, dot = false, style }: BadgeProps) {
  const fs = small ? FONT_SIZE.xs : FONT_SIZE.sm;
  const px = small ? 8 : 10;
  const py = small ? 2 : 4;

  return (
    <View
      style={[
        styles.base,
        {
          backgroundColor: BG[variant],
          borderColor: BORDER[variant],
          paddingHorizontal: px,
          paddingVertical: py,
        },
        style,
      ]}
    >
      {dot && (
        <View
          style={[styles.dot, { backgroundColor: FG[variant] }]}
        />
      )}
      <Text style={[styles.label, { fontSize: fs, color: FG[variant] }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: RADIUS.full,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 5,
  },
  label: {
    fontWeight: FONT_WEIGHT.semibold,
    letterSpacing: 0.2,
  },
});
