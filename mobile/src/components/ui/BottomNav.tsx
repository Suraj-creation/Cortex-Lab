/**
 * BottomNav — Neural Dark 7-tab navigation
 * Background: surfaceContainer (#0f1930), active: primary with underline
 * No border — color shift creates separation from content
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { NEURAL, FONT_SIZE, FONT_WEIGHT, SPACING } from '../../theme/colors';
import { AppIcon, type AppIconName } from './AppIcon';

export type NavKey = 'chat' | 'memories' | 'graph' | 'dashboard' | 'observability' | 'ambient' | 'documents';

interface NavItem {
  key: NavKey;
  label: string;
  iconName: AppIconName;
}

interface BottomNavProps {
  items: NavItem[];
  activeKey: NavKey;
  onSelect: (key: NavKey) => void;
}

export function BottomNav({ items, activeKey, onSelect }: BottomNavProps) {
  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
        bounces={false}
      >
        {items.map((item) => {
          const isActive = item.key === activeKey;
          return (
            <TouchableOpacity
              key={item.key}
              style={[styles.tab, isActive && styles.tabActive]}
              onPress={() => onSelect(item.key)}
              activeOpacity={0.75}
              accessibilityRole="button"
              accessibilityLabel={item.label}
            >
              <AppIcon
                name={item.iconName}
                size={18}
                color={isActive ? NEURAL.primary : NEURAL.onSurfaceVariant}
                style={styles.icon}
              />
              <Text style={[styles.label, isActive && styles.labelActive]}>
                {item.label}
              </Text>
              {isActive && <View style={styles.indicator} />}
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: NEURAL.surfaceContainerLow,
    paddingBottom: SPACING.sm,
    paddingTop: SPACING.xs,
  },
  scrollContent: {
    paddingHorizontal: SPACING.sm,
    gap: 2,
  },
  tab: {
    alignItems: 'center',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: 12,
    minWidth: 60,
    position: 'relative',
  },
  tabActive: {
    backgroundColor: `${NEURAL.primary}18`,
  },
  icon: {
    marginBottom: 2,
  },
  label: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.onSurfaceVariant,
    fontWeight: FONT_WEIGHT.medium,
  },
  labelActive: {
    color: NEURAL.primary,
    fontWeight: FONT_WEIGHT.bold,
  },
  indicator: {
    position: 'absolute',
    top: 0,
    left: '20%',
    right: '20%',
    height: 2,
    backgroundColor: NEURAL.primary,
    borderBottomLeftRadius: 2,
    borderBottomRightRadius: 2,
  },
});
