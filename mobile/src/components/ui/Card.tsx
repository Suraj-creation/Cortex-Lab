import React from "react";
import { View, StyleSheet, ViewStyle } from "react-native";
import { COLORS, SEMANTIC_COLORS, SPACING, BORDER_RADIUS, SHADOWS } from "../../theme/colors";

interface CardProps {
  children: React.ReactNode;
  variant?: "default" | "outlined" | "elevated";
  padding?: "sm" | "md" | "lg";
  style?: ViewStyle;
}

export function Card({
  children,
  variant = "default",
  padding = "lg",
  style,
}: CardProps) {
  const paddingValue = padding === "sm" ? SPACING.md : padding === "lg" ? SPACING.xl : SPACING.lg;

  const variantStyle: ViewStyle =
    variant === "outlined"
      ? {
          borderWidth: 1,
          borderColor: SEMANTIC_COLORS.borderPrimary,
          backgroundColor: SEMANTIC_COLORS.bgPrimary,
          ...SHADOWS.none,
        }
      : variant === "elevated"
        ? {
            backgroundColor: SEMANTIC_COLORS.bgElevated,
          borderWidth: 1,
          borderColor: SEMANTIC_COLORS.borderPrimary,
          ...SHADOWS.lg,
          }
        : {
          backgroundColor: SEMANTIC_COLORS.bgElevated,
          borderWidth: 1,
          borderColor: SEMANTIC_COLORS.borderPrimary,
          ...SHADOWS.sm,
          };

  return (
    <View
      style={[
        {
          borderRadius: BORDER_RADIUS.xl,
          padding: paddingValue,
          ...variantStyle,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}
