import React from "react";
import { View, ScrollView, StyleSheet, ViewStyle } from "react-native";
import { COLORS, SPACING } from "../../theme/colors";

interface ScreenProps {
  children: React.ReactNode;
  scrollable?: boolean;
  paddingHorizontal?: number;
  paddingVertical?: number;
  backgroundColor?: string;
  style?: ViewStyle;
}

/**
 * Standard screen wrapper with consistent padding and styling
 * Matches frontend layout consistency
 */
export function Screen({
  children,
  scrollable = false,
  paddingHorizontal = SPACING.lg,
  paddingVertical = SPACING.lg,
  backgroundColor = COLORS.surface[50],
  style,
}: ScreenProps) {
  const containerStyle = {
    flex: 1,
    paddingHorizontal,
    paddingVertical,
    backgroundColor,
  };

  if (scrollable) {
    return (
      <ScrollView
        style={[containerStyle, style]}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {children}
      </ScrollView>
    );
  }

  return (
    <View style={[containerStyle, style]}>
      {children}
    </View>
  );
}

/**
 * Vertical spacing utility
 */
export function Spacer({ size = "md" }: { size?: "xs" | "sm" | "md" | "lg" | "xl" }) {
  const sizeMap = {
    xs: SPACING.xs,
    sm: SPACING.sm,
    md: SPACING.md,
    lg: SPACING.lg,
    xl: SPACING.xl,
  };

  return <View style={{ height: sizeMap[size] }} />;
}

/**
 * Container for consistent section styling
 */
export function Section({
  children,
  gap = "lg",
  style,
}: {
  children: React.ReactNode;
  gap?: "xs" | "sm" | "md" | "lg" | "xl";
  style?: ViewStyle;
}) {
  const gapMap = {
    xs: SPACING.xs,
    sm: SPACING.sm,
    md: SPACING.md,
    lg: SPACING.lg,
    xl: SPACING.xl,
  };

  return (
    <View style={[styles.section, { gap: gapMap[gap] }, style]}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
  },
  section: {
    flex: 0,
  },
});
