import React from "react";
import { View, Pressable, Text, StyleSheet, ViewStyle, ScrollView } from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { COLORS, SEMANTIC_COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS, SHADOWS } from "../../theme/colors";

type ActiveView = "chat" | "memories" | "graph" | "dashboard" | "observability" | "ambient" | "documents";

interface NavItem {
  key: ActiveView;
  label: string;
  icon?: React.ReactNode;
}

interface BottomNavProps {
  items: NavItem[];
  activeKey: ActiveView;
  onSelect: (key: ActiveView) => void;
  style?: ViewStyle;
}

const NAV_ICONS: Record<ActiveView, keyof typeof MaterialCommunityIcons.glyphMap> = {
  chat: "chat-outline",
  memories: "head-snowflake-outline",
  graph: "graph-outline",
  dashboard: "view-dashboard-outline",
  observability: "pulse",
  ambient: "microphone-outline",
  documents: "file-document-outline",
};

export function BottomNav({
  items,
  activeKey,
  onSelect,
  style,
}: BottomNavProps) {
  return (
    <View style={[styles.container, style]}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.navContent}
        style={styles.navScroll}
      >
        {items.map((item) => {
          const isActive = item.key === activeKey;
          return (
            <Pressable
              key={item.key}
              onPress={() => onSelect(item.key)}
              style={({ pressed }) => [
                styles.navItem,
                isActive && styles.navItemActive,
                pressed && styles.navItemPressed,
              ]}
            >
              <View style={[styles.iconContainer, isActive && styles.iconContainerActive]}>
                {item.icon ? (
                  item.icon
                ) : (
                  <MaterialCommunityIcons
                    name={NAV_ICONS[item.key]}
                    size={15}
                    color={isActive ? COLORS.primary[700] : SEMANTIC_COLORS.textSecondary}
                  />
                )}
              </View>
              <Text
                style={[
                  styles.label,
                  isActive && styles.labelActive,
                ]}
                numberOfLines={1}
              >
                {item.label}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: SEMANTIC_COLORS.navBackground,
    borderTopWidth: 1,
    borderTopColor: SEMANTIC_COLORS.borderPrimary,
    paddingHorizontal: SPACING.sm,
    paddingVertical: SPACING.sm,
    ...SHADOWS.md,
  },
  navScroll: {
    flexGrow: 0,
  },
  navContent: {
    gap: SPACING.sm,
    paddingHorizontal: SPACING.sm,
  },
  navItem: {
    minWidth: 78,
    maxWidth: 90,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: SPACING.sm,
    paddingHorizontal: SPACING.sm,
    borderRadius: BORDER_RADIUS.xl,
    borderWidth: 1,
    borderColor: "transparent",
    gap: SPACING.xs,
  },
  navItemActive: {
    backgroundColor: SEMANTIC_COLORS.bgHighlight,
    borderColor: SEMANTIC_COLORS.borderAccent,
  },
  navItemPressed: {
    opacity: 0.8,
  },
  iconContainer: {
    width: 30,
    height: 30,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: BORDER_RADIUS.full,
    borderWidth: 1,
    borderColor: SEMANTIC_COLORS.borderPrimary,
    backgroundColor: SEMANTIC_COLORS.bgSecondary,
  },
  iconContainerActive: {
    backgroundColor: COLORS.primary[100],
    borderColor: COLORS.primary[300],
  },
  label: {
    fontSize: 11,
    fontWeight: TYPOGRAPHY.fontWeight.medium,
    color: SEMANTIC_COLORS.textSecondary,
    textAlign: "center",
  },
  labelActive: {
    color: COLORS.primary[700],
    fontWeight: TYPOGRAPHY.fontWeight.semibold,
  },
});
