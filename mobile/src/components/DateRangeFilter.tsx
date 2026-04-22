import React, { useState } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
} from "react-native";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from "../theme/colors";

interface DateRangeFilterProps {
  onApply: (startTime: number | null, endTime: number | null) => void;
  onCancel: () => void;
}

export default function DateRangeFilter({
  onApply,
  onCancel,
}: DateRangeFilterProps) {
  const [preset, setPreset] = useState<"last1h" | "last6h" | "last24h" | "all">(
    "last1h"
  );

  const getTimeRange = (
    preset: "last1h" | "last6h" | "last24h" | "all"
  ): [number | null, number | null] => {
    const now = Date.now();

    switch (preset) {
      case "last1h":
        return [now - 60 * 60 * 1000, now];
      case "last6h":
        return [now - 6 * 60 * 60 * 1000, now];
      case "last24h":
        return [now - 24 * 60 * 60 * 1000, now];
      case "all":
        return [null, null];
      default:
        return [null, null];
    }
  };

  const handleApply = () => {
    const [start, end] = getTimeRange(preset);
    onApply(start, end);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Time Range</Text>
      <View style={styles.presets}>
        {(["last1h", "last6h", "last24h", "all"] as const).map((p) => (
          <Pressable
            key={p}
            onPress={() => setPreset(p)}
            style={({ pressed }) => [
              styles.presetButton,
              preset === p && styles.presetButtonActive,
              pressed && styles.buttonPressed,
            ]}
          >
            <Text
              style={[
                styles.presetButtonText,
                preset === p && styles.presetButtonTextActive,
              ]}
            >
              {p === "last1h" ? "Last Hour" : p === "last6h" ? "Last 6h" : p === "last24h" ? "Last 24h" : "All"}
            </Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.actions}>
        <Pressable
          onPress={onCancel}
          style={({ pressed }) => [styles.cancelButton, pressed && styles.buttonPressed]}
        >
          <Text style={styles.cancelButtonText}>Cancel</Text>
        </Pressable>
        <Pressable
          onPress={handleApply}
          style={({ pressed }) => [styles.applyButton, pressed && styles.buttonPressed]}
        >
          <Text style={styles.applyButtonText}>Apply Filter</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.md,
  },
  label: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
    marginBottom: SPACING.sm,
  },
  presets: {
    flexDirection: "row",
    gap: SPACING.sm,
    marginBottom: SPACING.md,
    flexWrap: "wrap",
  },
  presetButton: {
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.sm,
    borderRadius: BORDER_RADIUS.full,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
  },
  presetButtonActive: {
    backgroundColor: COLORS.primary[600],
    borderColor: COLORS.primary[600],
  },
  presetButtonText: {
    fontSize: TYPOGRAPHY.fontSize.xs,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
  },
  presetButtonTextActive: {
    color: COLORS.white,
  },
  actions: {
    flexDirection: "row",
    gap: SPACING.sm,
  },
  cancelButton: {
    flex: 1,
    paddingVertical: SPACING.md,
    borderRadius: BORDER_RADIUS.xl,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    alignItems: "center",
  },
  cancelButtonText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: SEMANTIC_COLORS.textSecondary,
  },
  applyButton: {
    flex: 1,
    paddingVertical: SPACING.md,
    borderRadius: BORDER_RADIUS.xl,
    backgroundColor: COLORS.primary[600],
    borderWidth: 1,
    borderColor: COLORS.primary[600],
    alignItems: "center",
  },
  applyButtonText: {
    fontSize: TYPOGRAPHY.fontSize.sm,
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
    color: COLORS.white,
  },
  buttonPressed: {
    opacity: 0.85,
  },
});
