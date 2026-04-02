import React from "react";
import { View, Text, StyleSheet, TextStyle, ViewStyle } from "react-native";
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from "../../theme/colors";

interface BadgeProps {
  label: string;
  variant?: "default" | "success" | "warning" | "error" | "info" | "primary";
  small?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const variantStyles = {
  default: { bg: COLORS.surface[100], text: COLORS.surface[700] },
  success: { bg: COLORS.success[50], text: COLORS.success[700] },
  warning: { bg: COLORS.warning[50], text: COLORS.warning[700] },
  error: { bg: COLORS.error[50], text: COLORS.error[700] },
  info: { bg: COLORS.info[50], text: COLORS.info[700] },
  primary: { bg: COLORS.primary[50], text: COLORS.primary[700] },
};

export function Badge({
  label,
  variant = "default",
  small = false,
  style,
  textStyle,
}: BadgeProps) {
  const variantStyle = variantStyles[variant];

  return (
    <View
      style={[
        styles.badge,
        small && styles.badgeSmall,
        { backgroundColor: variantStyle.bg },
        style,
      ]}
    >
      <Text
        style={[
          styles.text,
          small && styles.textSmall,
          { color: variantStyle.text },
          textStyle,
        ]}
      >
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.08)",
    alignSelf: "flex-start",
  },
  badgeSmall: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
    borderRadius: BORDER_RADIUS.full,
  },
  text: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    letterSpacing: 0.2,
  },
  textSmall: {
    fontSize: 10,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
});
