/**
 * ConversationDrawer — Neural Dark slide-in conversation history
 * Stitch ref: e0034294221049f79e6bc5e9ec4f868d
 */
import React, { useRef, useEffect, useState } from 'react';
import {
  View,
  Text,
  Modal,
  Animated,
  TouchableOpacity,
  TextInput,
  FlatList,
  StyleSheet,
  Dimensions,
  Pressable,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Button } from '../components/ui/Button';
import { NeuralPulse } from '../components/ui/NeuralPulse';
import { AppIcon } from '../components/ui/AppIcon';

const SCREEN_W = Dimensions.get('window').width;
const DRAWER_W = Math.min(SCREEN_W * 0.82, 320);

interface ConvSummary {
  id: string;
  title: string;
  timestamp: number;
}

interface ConversationDrawerProps {
  visible: boolean;
  onClose: () => void;
  conversations: ConvSummary[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  isOnline: boolean;
  onOpenSettings: () => void;
}

function groupByDate(conversations: ConvSummary[]): { label: string; items: ConvSummary[] }[] {
  const now = Date.now();
  const ONE_DAY = 86400000;
  const ONE_WEEK = 7 * ONE_DAY;

  const groups: Record<string, ConvSummary[]> = { Today: [], Yesterday: [], 'This Week': [], Older: [] };
  conversations.forEach((c) => {
    const diff = now - c.timestamp;
    if (diff < ONE_DAY) groups.Today.push(c);
    else if (diff < 2 * ONE_DAY) groups.Yesterday.push(c);
    else if (diff < ONE_WEEK) groups['This Week'].push(c);
    else groups.Older.push(c);
  });

  return Object.entries(groups)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }));
}

export function ConversationDrawer({
  visible,
  onClose,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  isOnline,
  onOpenSettings,
}: ConversationDrawerProps) {
  const slideAnim = useRef(new Animated.Value(-DRAWER_W)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.timing(slideAnim, { toValue: 0, duration: 250, useNativeDriver: true }),
        Animated.timing(opacityAnim, { toValue: 1, duration: 250, useNativeDriver: true }),
      ]).start();
    } else {
      Animated.parallel([
        Animated.timing(slideAnim, { toValue: -DRAWER_W, duration: 200, useNativeDriver: true }),
        Animated.timing(opacityAnim, { toValue: 0, duration: 200, useNativeDriver: true }),
      ]).start();
    }
  }, [visible]);

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(search.toLowerCase())
  );
  const groups = groupByDate(filtered);

  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={onClose}>
      {/* Backdrop */}
      <Animated.View style={[styles.backdrop, { opacity: opacityAnim }]}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
      </Animated.View>

      {/* Drawer */}
      <Animated.View style={[styles.drawer, { transform: [{ translateX: slideAnim }] }]}>
        {/* Logo header */}
        <LinearGradient
          colors={[NEURAL.surfaceContainerLow, NEURAL.surfaceContainer]}
          style={styles.drawerHeader}
        >
          <View style={styles.logoRow}>
            <AppIcon name="brain" size={18} color={NEURAL.primary} />
            <Text style={styles.logoText}>Cortex Lab</Text>
          </View>
          <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
            <AppIcon name="close" size={18} color={NEURAL.onSurfaceVariant} style={styles.closeIcon} />
          </TouchableOpacity>
        </LinearGradient>

        {/* New Chat */}
        <View style={styles.newChatWrap}>
          <Button
            label="+ New Chat"
            onPress={() => { onNewChat(); onClose(); }}
            fullWidth
          />
        </View>

        {/* Search */}
        <View style={styles.searchWrap}>
          <TextInput
            style={styles.searchInput}
            placeholder="Search conversations…"
            placeholderTextColor={NEURAL.outline}
            value={search}
            onChangeText={setSearch}
            selectionColor={NEURAL.primary}
          />
        </View>

        {/* Conversation groups */}
        <FlatList
          data={groups}
          keyExtractor={(g) => g.label}
          style={styles.list}
          showsVerticalScrollIndicator={false}
          renderItem={({ item: group }) => (
            <View>
              <Text style={styles.groupLabel}>{group.label}</Text>
              {group.items.map((conv) => {
                const isActive = conv.id === activeConversationId;
                return (
                  <TouchableOpacity
                    key={conv.id}
                    onPress={() => { onSelectConversation(conv.id); onClose(); }}
                    style={[styles.convRow, isActive && styles.convRowActive]}
                  >
                    {isActive && <View style={styles.activeIndicator} />}
                    <Text style={[styles.convTitle, isActive && styles.convTitleActive]} numberOfLines={2}>
                      {conv.title}
                    </Text>
                    <Text style={styles.convTime}>
                      {new Date(conv.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          )}
        />

        {/* Bottom: Settings + status */}
        <View style={styles.drawerFooter}>
          <TouchableOpacity
            onPress={() => { onOpenSettings(); onClose(); }}
            style={styles.settingsRow}
          >
            <AppIcon name="cog-outline" size={18} color={NEURAL.onSurfaceVariant} style={styles.settingsIcon} />
            <Text style={styles.settingsText}>Settings</Text>
          </TouchableOpacity>
          <View style={styles.serverStatus}>
            <NeuralPulse active={isOnline} size={5} color={isOnline ? NEURAL.tertiary : NEURAL.error} />
            <Text style={[styles.serverText, { color: isOnline ? NEURAL.tertiary : NEURAL.error }]}>
              {isOnline ? 'Server Connected' : 'Server Offline'}
            </Text>
          </View>
        </View>
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(6,14,32,0.75)',
  },
  drawer: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    width: DRAWER_W,
    backgroundColor: NEURAL.surfaceContainerLow,
    shadowColor: NEURAL.primary,
    shadowOpacity: 0.2,
    shadowRadius: 24,
    shadowOffset: { width: 8, height: 0 },
    elevation: 20,
  },
  drawerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING['4xl'],
    paddingBottom: SPACING.md,
  },
  logoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
  },
  logoText: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, letterSpacing: -0.5 },
  closeBtn: { padding: SPACING.sm },
  closeIcon: { marginVertical: 1 },

  newChatWrap: { paddingHorizontal: SPACING.lg, paddingVertical: SPACING.md },

  searchWrap: {
    paddingHorizontal: SPACING.lg,
    marginBottom: SPACING.md,
  },
  searchInput: {
    backgroundColor: NEURAL.surfaceContainerHighest,
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurface,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },

  list: { flex: 1, paddingHorizontal: SPACING.sm },
  groupLabel: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurfaceVariant,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    paddingHorizontal: SPACING.sm,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.xs,
  },
  convRow: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.lg,
    position: 'relative',
    marginBottom: 2,
  },
  convRowActive: {
    backgroundColor: NEURAL.surfaceContainerHigh,
    paddingLeft: SPACING.md + 6,
  },
  activeIndicator: {
    position: 'absolute',
    left: 8,
    top: '20%',
    bottom: '20%',
    width: 3,
    backgroundColor: NEURAL.primary,
    borderRadius: 2,
  },
  convTitle: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    fontWeight: FONT_WEIGHT.normal,
    marginBottom: 2,
  },
  convTitleActive: {
    color: NEURAL.onSurface,
    fontWeight: FONT_WEIGHT.semibold,
  },
  convTime: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.outline,
  },

  drawerFooter: {
    padding: SPACING.lg,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
    gap: SPACING.md,
  },
  settingsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
  },
  settingsIcon: { marginVertical: 1 },
  settingsText: { fontSize: FONT_SIZE.base, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.medium },
  serverStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  serverText: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
  },
});
