import React from "react";
import { Pressable, Text, StyleSheet, ViewStyle, TextStyle } from "react-native";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from "../../theme/colors";

interface ButtonProps {
  onPress: () => void;
  label: string;
  variant?: "primary" | "secondary" | "outline" | "error";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  style?: ViewStyle;
}

const createButtonStyles = (variant: string, size: string, disabled: boolean) => {
  const buttonStyle: ViewStyle = {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.sm,
    borderRadius: BORDER_RADIUS.xl,
    paddingHorizontal: size === "sm" ? SPACING.md : size === "lg" ? SPACING["2xl"] : SPACING.xl,
    paddingVertical: size === "sm" ? SPACING.sm : size === "lg" ? SPACING.lg : SPACING.md,
    minHeight: size === "sm" ? 36 : size === "lg" ? 50 : 42,
  };

  const textStyle: TextStyle = {
    fontSize: size === "sm" ? TYPOGRAPHY.fontSize.sm : size === "lg" ? TYPOGRAPHY.fontSize.lg : TYPOGRAPHY.fontSize.md,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  };

  if (variant === "primary") {
    return {
      buttonStyle: {
        ...buttonStyle,
        backgroundColor: disabled ? SEMANTIC_COLORS.buttonPrimaryDisabled : SEMANTIC_COLORS.buttonPrimary,
        borderWidth: 1,
        borderColor: disabled ? COLORS.surface[300] : COLORS.primary[700],
        ...SHADOWS.md,
      },
      textStyle: {
        ...textStyle,
        color: SEMANTIC_COLORS.textOnAccent,
      },
    };
  }

  if (variant === "secondary") {
    return {
      buttonStyle: {
        ...buttonStyle,
        backgroundColor: disabled ? COLORS.surface[100] : SEMANTIC_COLORS.buttonSecondary,
        borderWidth: 1,
        borderColor: SEMANTIC_COLORS.borderPrimary,
      },
      textStyle: {
        ...textStyle,
        color: SEMANTIC_COLORS.buttonSecondaryText,
      },
    };
  }

  if (variant === "outline") {
    return {
      buttonStyle: {
        ...buttonStyle,
        backgroundColor: COLORS.transparent,
        borderWidth: 1,
        borderColor: SEMANTIC_COLORS.borderAccent,
      },
      textStyle: {
        ...textStyle,
        color: COLORS.primary[700],
      },
    };
  }

  if (variant === "error") {
    return {
      buttonStyle: {
        ...buttonStyle,
        backgroundColor: disabled ? SEMANTIC_COLORS.buttonPrimaryDisabled : COLORS.error[500],
        borderWidth: 1,
        borderColor: COLORS.error[600],
        ...SHADOWS.sm,
      },
      textStyle: {
        ...textStyle,
        color: COLORS.white,
      },
    };
  }

  return { buttonStyle, textStyle };
};

export function Button({
  onPress,
  label,
  variant = "primary",
  size = "md",
  disabled = false,
  loading = false,
  icon,
  style,
}: ButtonProps) {
  const { buttonStyle, textStyle } = createButtonStyles(variant, size, disabled);

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        buttonStyle,
        style,
        pressed && !disabled && { opacity: 0.9, transform: [{ scale: 0.985 }] },
        disabled && { opacity: 0.6 },
      ]}
    >
      {icon ? <>{icon}</> : null}
      <Text style={[textStyle, styles.label]}>{loading ? "…" : label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  label: {
    textAlign: "center",
  },
});
