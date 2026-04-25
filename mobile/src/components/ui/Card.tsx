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
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#f1f5f9',
    ...SHADOWS.sm,
  },
  elevated: {
    backgroundColor: '#ffffff',
    ...SHADOWS.lg,
  },
  outlined: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  glass: {
    backgroundColor: 'rgba(255, 255, 255, 0.75)',
    borderWidth: 1,
    borderColor: 'rgba(241, 245, 249, 0.8)',
    ...SHADOWS.sm,
  },
  accent: {
    backgroundColor: '#eef2ff',
    borderWidth: 1,
    borderColor: '#c7d2fe',
  },
  leftAccentBase: {
    paddingLeft: 14,
  },
});
