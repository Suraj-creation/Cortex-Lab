/**
 * Card — Neural Dark surface component
 * No borders — uses background color shifts for depth
 * Design: "Tonal Layering" from Cortex Neural Dark
 */
import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { NEURAL, RADIUS, FONT_SIZE } from '../../theme/colors';

export type CardVariant = 'default' | 'elevated' | 'outlined' | 'glass';
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
  leftAccentColor = NEURAL.primary,
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
    backgroundColor: NEURAL.surfaceContainer,  // #0f1930
  },
  elevated: {
    backgroundColor: NEURAL.surfaceContainerHigh,    // #141f38
    shadowColor: NEURAL.primary,
    shadowOpacity: 0.10,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 4 },
    elevation: 5,
  },
  outlined: {
    backgroundColor: NEURAL.surfaceContainer,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,  // #40485d ghost border
  },
  glass: {
    backgroundColor: `${NEURAL.surfaceVariant}99`, // 60% opacity
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}26`,     // 15% opacity
  },
  leftAccentBase: {
    paddingLeft: 14,
  },
});
