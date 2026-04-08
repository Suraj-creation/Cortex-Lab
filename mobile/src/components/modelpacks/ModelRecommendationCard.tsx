import React from "react";
import { View, Text, StyleSheet } from "react-native";

import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from "../../theme/colors";

interface ModelRecommendationCardProps {
  title: string;
  reason: string;
  recommended: boolean;
}

export function ModelRecommendationCard({
  title,
  reason,
  recommended,
}: ModelRecommendationCardProps) {
  return (
    <View style={[styles.card, recommended && styles.cardRecommended]}>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.reason}>{reason}</Text>
      <Text style={[styles.badge, recommended ? styles.badgeOn : styles.badgeOff]}>
        {recommended ? "Recommended" : "Optional"}
      </Text>
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
    gap: SPACING.xs,
  },
  cardRecommended: {
    borderColor: `${NEURAL.primary}66`,
    backgroundColor: `${NEURAL.primary}10`,
  },
  title: {
    color: NEURAL.onSurface,
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.bold,
  },
  reason: {
    color: NEURAL.onSurfaceVariant,
    fontSize: FONT_SIZE.sm,
    lineHeight: FONT_SIZE.sm * 1.4,
  },
  badge: {
    marginTop: SPACING.xs,
    alignSelf: "flex-start",
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.sm,
    paddingVertical: 2,
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
  },
  badgeOn: {
    color: NEURAL.primary,
    backgroundColor: `${NEURAL.primary}22`,
  },
  badgeOff: {
    color: NEURAL.onSurfaceVariant,
    backgroundColor: `${NEURAL.outlineVariant}22`,
  },
});
