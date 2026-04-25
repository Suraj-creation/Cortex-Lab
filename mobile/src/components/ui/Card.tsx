/**
 * Card — Cortex Aurora light surface component
 * White cards with subtle borders, shadows, and variants
 */
import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { RADIUS, SHADOWS } from '../../theme/colors';

export type CardVariant = 'default' | 'elevated' | 'outlined' | 'glass' | 'accent';
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

interface CardProps {
  children: React.ReactNode;
  variant?: CardVariant;
  padding?: CardPadding;
  style?: ViewStyle | ViewStyle[];
  leftAccent?: boolean;
  leftAccentColor?: string;
}

const CARD_PADDING = { none: 0, sm: 10, md: 14, lg: 18 };

export function Card({
  children,
  variant = 'default',
  padding = 'md',
  style,
  leftAccent = false,
  leftAccentColor = '#6366f1',
}: CardProps) {
  return (
    <View
      style={[
        styles.base,
        styles[variant],
        { padding: CARD_PADDING[padding] },
        leftAccent && styles.leftAccentBase,
        leftAccent && { borderLeftColor: leftAccentColor, borderLeftWidth: 3 },
        style as ViewStyle,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  base: {
    borderRadius: RADIUS.xl,
    overflow: 'hidden',
  },
  default: {
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.lg,
  },
  elevated: {
    backgroundColor: '#eef3fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.xl,
  },
  outlined: {
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.94)',
    ...SHADOWS.md,
  },
  glass: {
    backgroundColor: 'rgba(245, 248, 255, 0.82)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.92)',
    ...SHADOWS.md,
  },
  accent: {
    backgroundColor: '#e7edff',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.lg,
  },
  leftAccentBase: {
    paddingLeft: 14,
  },
});
