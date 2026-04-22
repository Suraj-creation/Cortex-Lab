import React from "react";
import { View, Text, StyleSheet } from "react-native";

import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from "../../theme/colors";

interface OfflineReadinessBadgeProps {
  ready: boolean;
  details?: string;
}

export function OfflineReadinessBadge({ ready, details }: OfflineReadinessBadgeProps) {
  return (
    <View style={[styles.badge, ready ? styles.ready : styles.notReady]}>
      <Text style={[styles.label, ready ? styles.labelReady : styles.labelNotReady]}>
        {ready ? "Offline Ready" : "Offline Incomplete"}
      </Text>
      {details ? <Text style={styles.details}>{details}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.xs,
    alignSelf: "flex-start",
  },
  ready: {
    backgroundColor: `${NEURAL.tertiary}22`,
    borderWidth: 1,
    borderColor: `${NEURAL.tertiary}66`,
  },
  notReady: {
    backgroundColor: "#f59e0b22",
    borderWidth: 1,
    borderColor: "#f59e0b66",
  },
  label: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.bold,
  },
  labelReady: {
    color: NEURAL.tertiary,
  },
  labelNotReady: {
    color: "#f59e0b",
  },
  details: {
    marginTop: 2,
    color: NEURAL.onSurfaceVariant,
    fontSize: FONT_SIZE.xs,
  },
});
