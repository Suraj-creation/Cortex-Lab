import React from "react";
import { View, Text, StyleSheet, Pressable, ViewStyle } from "react-native";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from "../../theme/colors";
import { ModelStatus } from "../../../shared/core/types";

interface HeaderProps {
  modelStatus: ModelStatus;
  title?: string;
  subtitle?: string;
  onSettingsPress?: () => void;
  style?: ViewStyle;
}

export function Header({
  modelStatus,
  title = "Cortex Lab",
  subtitle,
  onSettingsPress,
  style,
}: HeaderProps) {
  const isLoaded = modelStatus.model_loaded;
  const isLoading = modelStatus.status === "loading";

  const statusColor = isLoaded
    ? COLORS.success[500]
    : isLoading
      ? COLORS.warning[500]
      : COLORS.error[500];

  const statusBgColor = isLoaded
    ? COLORS.success[50]
    : isLoading
      ? COLORS.warning[50]
      : COLORS.error[50];

  const statusText = isLoaded ? "Online" : isLoading ? "Loading…" : "Offline";

  return (
    <View
      style={[
        styles.container,
        style,
      ]}
    >
      <View style={styles.leftSection}>
        <Text style={styles.eyebrow}>CORTEX MOBILE</Text>
        <Text style={styles.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>

      <View style={styles.rightSection}>
        <View
          style={[
            styles.statusBadge,
            { backgroundColor: statusBgColor },
          ]}
        >
          <View
            style={[
              styles.statusDot,
              { backgroundColor: statusColor },
            ]}
          />
          <Text style={[styles.statusText, { color: statusColor }]}>
            {statusText}
          </Text>
        </View>

        {onSettingsPress ? (
          <Pressable onPress={onSettingsPress} style={styles.settingsButton}>
            <Text style={styles.settingsButtonText}>Tune</Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    paddingHorizontal: SPACING.xl,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.sm,
    backgroundColor: SEMANTIC_COLORS.bgPrimary,
    borderBottomWidth: 1,
    borderBottomColor: SEMANTIC_COLORS.borderPrimary,
    ...SHADOWS.sm,
  },
  leftSection: {
    flex: 1,
    paddingRight: SPACING.sm,
  },
  eyebrow: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textTertiary,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    letterSpacing: 0.9,
    marginBottom: SPACING.xs,
  },
  title: {
    fontSize: TYPOGRAPHY.fontSize.xl,
    fontWeight: TYPOGRAPHY.fontWeight.bold,
    color: SEMANTIC_COLORS.textPrimary,
  },
  subtitle: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: SEMANTIC_COLORS.textSecondary,
    marginTop: SPACING.xs,
  },
  rightSection: {
    alignItems: "flex-end",
    gap: SPACING.sm,
    marginLeft: SPACING.sm,
  },
  statusBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: BORDER_RADIUS.full,
  },
  statusText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
  settingsButton: {
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderAccent,
    backgroundColor: SEMANTIC_COLORS.bgHighlight,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.full,
  },
  settingsButtonText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    color: COLORS.primary[700],
    fontWeight: TYPOGRAPHY.fontWeight.medium,
  },
});
