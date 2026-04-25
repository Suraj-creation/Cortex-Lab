/**
 * ConversationDrawer — Cortex Aurora slide-in conversation history
 */
import React from 'react';
import {
  View,
  Text,
  Modal,
  TouchableOpacity,
  Pressable,
  FlatList,
  StyleSheet,
  Platform,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';

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
  const formatTime = (ts: number) => {
    const d = new Date(ts);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) return 'Today';
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <Pressable style={styles.overlay} onPress={onClose}>
        <Pressable style={styles.drawer} onPress={(e) => e.stopPropagation()}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <View style={styles.logoContainer}>
                <View style={styles.logoDot} />
              </View>
              <View>
                <Text style={styles.headerTitle}>Cortex Lab</Text>
                <Text style={styles.headerSubtitle}>Conversation History</Text>
              </View>
            </View>
            <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
              <AppIcon name="close" size={18} color="#64748b" />
            </TouchableOpacity>
          </View>

          {/* New Chat button */}
          <View style={styles.newChatWrap}>
            <TouchableOpacity onPress={onNewChat} style={styles.newChatBtn} activeOpacity={0.7}>
              <AppIcon name="plus" size={16} color="#6366f1" />
              <Text style={styles.newChatText}>New Conversation</Text>
            </TouchableOpacity>
          </View>

          {/* Conversation list */}
          <FlatList
            data={conversations}
            keyExtractor={(c) => c.id}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => {
              const isActive = item.id === activeConversationId;
              return (
                <TouchableOpacity
                  onPress={() => { onSelectConversation(item.id); onClose(); }}
                  style={[styles.convItem, isActive && styles.convItemActive]}
                  activeOpacity={0.7}
                >
                  <AppIcon
                    name="chat-processing-outline"
                    size={16}
                    color={isActive ? '#6366f1' : '#94a3b8'}
                  />
                  <View style={styles.convContent}>
                    <Text
                      style={[styles.convTitle, isActive && styles.convTitleActive]}
                      numberOfLines={1}
                    >
                      {item.title}
                    </Text>
                    <Text style={styles.convTime}>{formatTime(item.timestamp)}</Text>
                  </View>
                  {isActive && (
                    <View style={styles.activeDot} />
                  )}
                </TouchableOpacity>
              );
            }}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <Text style={styles.emptyText}>No conversations yet</Text>
              </View>
            }
          />

          {/* Footer */}
          <View style={styles.footer}>
            <View style={styles.statusRow}>
              <View style={[styles.statusDot, { backgroundColor: isOnline ? '#10b981' : '#94a3b8' }]} />
              <Text style={styles.statusText}>
                {isOnline ? 'Backend Connected' : 'Offline'}
              </Text>
            </View>
            <TouchableOpacity onPress={() => { onOpenSettings(); onClose(); }} style={styles.settingsBtn}>
              <AppIcon name="cog-outline" size={16} color="#64748b" />
              <Text style={styles.settingsBtnText}>Settings</Text>
            </TouchableOpacity>
          </View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.3)',
    justifyContent: 'flex-end',
  },
  drawer: {
    backgroundColor: '#ffffff',
    borderTopLeftRadius: RADIUS['3xl'],
    borderTopRightRadius: RADIUS['3xl'],
    maxHeight: '85%',
    ...SHADOWS.xl,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING['2xl'],
    paddingTop: SPACING.xl,
    paddingBottom: SPACING.md,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
  },
  logoContainer: {
    width: 36,
    height: 36,
    borderRadius: RADIUS.lg,
    backgroundColor: '#6366f1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#ffffff',
  },
  headerTitle: {
    fontSize: FONT_SIZE.lg,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
  },
  headerSubtitle: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    marginTop: 1,
  },
  closeBtn: {
    padding: SPACING.sm,
    backgroundColor: '#f1f5f9',
    borderRadius: RADIUS.lg,
  },

  newChatWrap: {
    paddingHorizontal: SPACING.xl,
    paddingBottom: SPACING.md,
  },
  newChatBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACING.sm,
    backgroundColor: '#eef2ff',
    borderWidth: 1,
    borderColor: '#c7d2fe',
    borderRadius: RADIUS.xl,
    paddingVertical: SPACING.md,
  },
  newChatText: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#6366f1',
  },

  list: {
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.md,
  },
  convItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.lg,
    borderRadius: RADIUS.xl,
    marginBottom: 2,
    gap: SPACING.md,
  },
  convItemActive: {
    backgroundColor: '#eef2ff',
  },
  convContent: {
    flex: 1,
  },
  convTitle: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.medium,
    color: '#334155',
  },
  convTitleActive: {
    color: '#4338ca',
    fontWeight: FONT_WEIGHT.semibold,
  },
  convTime: {
    fontSize: 10,
    color: '#94a3b8',
    marginTop: 2,
  },
  activeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#6366f1',
  },

  emptyContainer: {
    paddingVertical: SPACING['4xl'],
    alignItems: 'center',
  },
  emptyText: {
    fontSize: FONT_SIZE.sm,
    color: '#94a3b8',
  },

  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING['2xl'],
    paddingVertical: SPACING.lg,
    borderTopWidth: 1,
    borderTopColor: '#f1f5f9',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.medium,
  },
  settingsBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.xs,
    backgroundColor: '#f1f5f9',
    borderRadius: RADIUS.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },
  settingsBtnText: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.medium,
  },
});
