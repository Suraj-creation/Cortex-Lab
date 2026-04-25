import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";

import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from "../../theme/colors";

interface ModelDownloadManagerProps {
  packName: string;
  status: "not_installed" | "queued" | "downloading" | "installed" | "error";
  progress?: number;
  detail?: string;
  onInstall?: () => void;
  onRetry?: () => void;
  actionLabel?: string;
  actionDisabled?: boolean;
}

export function ModelDownloadManager({
  packName,
  status,
  progress = 0,
  detail,
  onInstall,
  onRetry,
  actionLabel,
  actionDisabled = false,
}: ModelDownloadManagerProps) {
  const computedActionLabel = actionLabel || (status === "error" ? "Retry" : "Install");
  const action = status === "error" ? onRetry : onInstall;
  const statusLabel = status.replace(/_/g, " ");

  return (
    <View style={styles.card}>
      <Text style={styles.title}>{packName}</Text>
      <Text style={styles.status}>Status: {statusLabel}</Text>
      {detail ? <Text style={styles.detail}>{detail}</Text> : null}
      {status === "queued" ? (
        <Text style={styles.progressText}>Preparing secure download…</Text>
      ) : null}
      {status === "downloading" ? (
        <>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${Math.max(0, Math.min(progress, 100))}%` }]} />
          </View>
          <Text style={styles.progressText}>{Math.round(progress)}%</Text>
        </>
      ) : null}
      {status === "installed" ? (
        <View style={styles.installedBadge}>
          <Text style={styles.installedBadgeText}>Installed on device</Text>
        </View>
      ) : null}
      {(status === "not_installed" || status === "error") && (action || actionDisabled) ? (
        <TouchableOpacity
          style={[styles.button, actionDisabled && styles.buttonDisabled]}
          onPress={action}
          disabled={actionDisabled || !action}
        >
          <Text style={[styles.buttonText, actionDisabled && styles.buttonTextDisabled]}>{computedActionLabel}</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: RADIUS.xl,
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}66`,
    backgroundColor: NEURAL.surfaceContainerLow,
    padding: SPACING.md,
    gap: SPACING.sm,
  },
  title: {
    color: NEURAL.onSurface,
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.bold,
  },
  status: {
    color: NEURAL.onSurfaceVariant,
    fontSize: FONT_SIZE.sm,
  },
  detail: {
    color: NEURAL.onSurfaceVariant,
    fontSize: FONT_SIZE.xs,
  },
  progressTrack: {
    height: 8,
    borderRadius: 4,
    backgroundColor: `${NEURAL.outlineVariant}40`,
    overflow: "hidden",
  },
  progressFill: {
    height: 8,
    borderRadius: 4,
    backgroundColor: NEURAL.primary,
  },
  progressText: {
    color: NEURAL.onSurfaceVariant,
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.medium,
  },
  installedBadge: {
    alignSelf: "flex-start",
    borderRadius: RADIUS.full,
    backgroundColor: `${NEURAL.tertiary}20`,
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.xs,
  },
  installedBadgeText: {
    color: NEURAL.tertiary,
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
  },
  button: {
    alignSelf: "flex-start",
    borderRadius: RADIUS.lg,
    backgroundColor: NEURAL.primary,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  buttonDisabled: {
    backgroundColor: `${NEURAL.outlineVariant}80`,
  },
  buttonText: {
    color: NEURAL.onPrimary,
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
  },
  buttonTextDisabled: {
    color: NEURAL.onSurfaceVariant,
  },
});
