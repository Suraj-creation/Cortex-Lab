/**
 * BottomNav — Cortex Aurora 5-tab navigation with "More" overflow
 * White background, indigo active pill, clean typography
 */
import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  ScrollView,
  Pressable,
  Platform,
} from 'react-native';
import { NEURAL, FONT_SIZE, FONT_WEIGHT, SPACING, RADIUS, SHADOWS } from '../../theme/colors';
import { AppIcon, type AppIconName } from './AppIcon';

export type NavKey =
  | 'chat' | 'memories' | 'graph' | 'dashboard' | 'observability'
  | 'agent' | 'wiki' | 'ambient' | 'documents'
  | 'session-forge' | 'chronicle';

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

// Primary launch surfaces + More
const PRIMARY_KEYS: NavKey[] = ['dashboard', 'chat', 'agent', 'memories'];

export function BottomNav({ items, activeKey, onSelect }: BottomNavProps) {
  const [moreVisible, setMoreVisible] = useState(false);

  const primaryItems = PRIMARY_KEYS
    .map((key) => items.find((item) => item.key === key))
    .filter((item): item is NavItem => Boolean(item));
  const moreItems = items.filter((i) => !PRIMARY_KEYS.includes(i.key));
  const isMoreActive = moreItems.some((i) => i.key === activeKey);

  const handleMoreSelect = useCallback((key: NavKey) => {
    setMoreVisible(false);
    onSelect(key);
  }, [onSelect]);

  return (
    <>
      <View style={styles.container}>
        {primaryItems.map((item) => {
          const isActive = item.key === activeKey;
          return (
            <TouchableOpacity
              key={item.key}
              style={[styles.tab, isActive && styles.tabActive]}
              onPress={() => onSelect(item.key)}
              activeOpacity={0.7}
              accessibilityRole="button"
              accessibilityLabel={item.label}
            >
              <View style={[styles.tabIconWrap, isActive && styles.tabIconWrapActive]}>
                <AppIcon
                  name={item.iconName}
                  size={20}
                  color={isActive ? '#4f5fe2' : '#7f8aa4'}
                />
              </View>
              <Text style={[styles.label, isActive && styles.labelActive]}>
                {item.label}
              </Text>
              {isActive && <View style={styles.indicator} />}
            </TouchableOpacity>
          );
        })}

        {/* More button */}
        <TouchableOpacity
          style={[styles.tab, isMoreActive && styles.tabActive]}
          onPress={() => setMoreVisible(true)}
          activeOpacity={0.7}
          accessibilityRole="button"
          accessibilityLabel="More options"
        >
          <View style={[styles.tabIconWrap, isMoreActive && styles.tabIconWrapActive]}>
            <AppIcon
              name="dots-horizontal"
              size={20}
              color={isMoreActive ? '#4f5fe2' : '#7f8aa4'}
            />
          </View>
          <Text style={[styles.label, isMoreActive && styles.labelActive]}>
            More
          </Text>
          {isMoreActive && <View style={styles.indicator} />}
        </TouchableOpacity>
      </View>

      {/* More drawer */}
      <Modal
        visible={moreVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setMoreVisible(false)}
      >
        <Pressable style={styles.overlay} onPress={() => setMoreVisible(false)}>
          <Pressable style={styles.moreSheet} onPress={(e) => e.stopPropagation()}>
            <View style={styles.sheetHandle} />
            <Text style={styles.sheetTitle}>All Screens</Text>
            <ScrollView style={styles.moreList} showsVerticalScrollIndicator={false}>
              {items.map((item) => {
                const isActive = item.key === activeKey;
                return (
                  <TouchableOpacity
                    key={item.key}
                    style={[styles.moreItem, isActive && styles.moreItemActive]}
                    onPress={() => handleMoreSelect(item.key)}
                    activeOpacity={0.7}
                  >
                    <View style={[
                      styles.moreIconContainer,
                      isActive && styles.moreIconContainerActive,
                    ]}>
                      <AppIcon
                        name={item.iconName}
                        size={20}
                        color={isActive ? '#6366f1' : '#64748b'}
                      />
                    </View>
                    <Text style={[styles.moreLabel, isActive && styles.moreLabelActive]}>
                      {item.label}
                    </Text>
                    {isActive && (
                      <View style={styles.moreActiveDot} />
                    )}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#e9eef8',
    paddingHorizontal: SPACING.md,
    paddingBottom: Platform.OS === 'ios' ? SPACING.xs : SPACING.sm,
    paddingTop: SPACING.sm,
    gap: SPACING.xs,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS['2xl'],
    position: 'relative',
  },
  tabActive: {
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    ...SHADOWS.md,
  },
  tabIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 14,
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.md,
  },
  tabIconWrapActive: {
    backgroundColor: '#eef1ff',
  },
  label: {
    fontSize: 10,
    color: '#7f8aa4',
    fontWeight: FONT_WEIGHT.medium,
    marginTop: 4,
  },
  labelActive: {
    color: '#4f5fe2',
    fontWeight: FONT_WEIGHT.bold,
  },
  indicator: {
    position: 'absolute',
    bottom: 0,
    left: '25%',
    right: '25%',
    height: 2.5,
    backgroundColor: '#6366f1',
    borderTopLeftRadius: 2,
    borderTopRightRadius: 2,
  },

  // More sheet
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.3)',
    justifyContent: 'flex-end',
  },
  moreSheet: {
    backgroundColor: '#e9eef8',
    borderTopLeftRadius: RADIUS['3xl'],
    borderTopRightRadius: RADIUS['3xl'],
    paddingTop: SPACING.md,
    paddingBottom: SPACING['4xl'],
    maxHeight: '70%',
    ...SHADOWS.xl,
  },
  sheetHandle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#e2e8f0',
    alignSelf: 'center',
    marginBottom: SPACING.lg,
  },
  sheetTitle: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
    paddingHorizontal: SPACING['2xl'],
    marginBottom: SPACING.lg,
  },
  moreList: {
    paddingHorizontal: SPACING.lg,
  },
  moreItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.lg,
    borderRadius: RADIUS.xl,
    marginBottom: SPACING.xs,
  },
  moreItemActive: {
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
  },
  moreIconContainer: {
    width: 40,
    height: 40,
    borderRadius: RADIUS.lg,
    backgroundColor: '#edf2fb',
    borderWidth: 1,
    borderColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: SPACING.md,
    ...SHADOWS.md,
  },
  moreIconContainerActive: {
    backgroundColor: '#eef1ff',
  },
  moreLabel: {
    fontSize: FONT_SIZE.md,
    fontWeight: FONT_WEIGHT.medium,
    color: '#334155',
    flex: 1,
  },
  moreLabelActive: {
    color: '#6366f1',
    fontWeight: FONT_WEIGHT.semibold,
  },
  moreActiveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#6366f1',
  },
});
