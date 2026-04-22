/**
 * MemoryScreen — Neural Dark Memory Browser
 * Stitch ref: 9346356d0b4a49789a66f110c6968e73
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  FlatList,
} from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { ProgressBar } from '../components/ui/ProgressBar';
import { AppIcon } from '../components/ui/AppIcon';
import type { MemoryObject } from '../../shared/core/types';

const MEMORY_TYPES = ['All', 'Episodic', 'Procedural', 'Semantic', 'Belief'] as const;
type MemoryFilter = (typeof MEMORY_TYPES)[number];

const TYPE_BADGE: Record<string, 'primary' | 'secondary' | 'success' | 'warning' | 'tertiary'> = {
  episodic:   'primary',
  procedural: 'secondary',
  semantic:   'success',
  belief:     'warning',
  working:    'tertiary',
};

function formatDate(ts: string | number) {
  try {
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch { return ''; }
}

interface MemoryScreenProps {
  memories: MemoryObject[];
  memorySearch: string;
  setMemorySearch: (v: string) => void;
  memoryDraft: string;
  setMemoryDraft: (v: string) => void;
  memoryBusy: boolean;
  loadingView: boolean;
  onSearch: () => void;
  onAddMemory: () => void;
  onDeleteMemory: (id: string) => void;
  onLoadMore: () => void;
}

export function MemoryScreen({
  memories,
  memorySearch,
  setMemorySearch,
  memoryDraft,
  setMemoryDraft,
  memoryBusy,
  loadingView,
  onSearch,
  onAddMemory,
  onDeleteMemory,
  onLoadMore,
}: MemoryScreenProps) {
  const [filter, setFilter] = useState<MemoryFilter>('All');

  const filtered =
    filter === 'All'
      ? memories
      : memories.filter((m) => (m.memory_type || 'episodic').toLowerCase() === filter.toLowerCase());

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Memory Browser</Text>
          <Text style={styles.subtitle}>{memories.length} memories stored</Text>
        </View>

        {/* Search */}
        <TextInput
          placeholder="Semantic search memories…"
          value={memorySearch}
          onChangeText={setMemorySearch}
          onSubmitEditing={onSearch}
          pill
          returnKeyType="search"
          icon={<AppIcon name="magnify" size={14} color={NEURAL.onSurfaceVariant} />}
          style={styles.searchBar}
        />

        {/* Type filter chips */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterRow}>
          {MEMORY_TYPES.map((t) => (
            <TouchableOpacity
              key={t}
              onPress={() => setFilter(t)}
              style={[styles.filterChip, filter === t && styles.filterChipActive]}
            >
              <Text style={[styles.filterChipText, filter === t && styles.filterChipTextActive]}>
                {t}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Add Memory composer */}
        <Card variant="outlined" style={styles.composerCard}>
          <Text style={styles.composerTitle}>Add Memory</Text>
          <TextInput
            placeholder="Store an important thought, decision, or event…"
            value={memoryDraft}
            onChangeText={setMemoryDraft}
            multiline
            style={styles.composerInput}
          />
          <Button
            label={memoryBusy ? 'Saving…' : 'Save Memory'}
            onPress={onAddMemory}
            disabled={memoryBusy || !memoryDraft.trim()}
            fullWidth
            loading={memoryBusy}
          />
        </Card>

        {/* Memory list */}
        {loadingView ? (
          <ActivityIndicator color={NEURAL.primary} size="large" style={styles.loader} />
        ) : filtered.length === 0 ? (
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>No memories found</Text>
            <Text style={styles.emptyBody}>Start conversations to build your memory store.</Text>
          </View>
        ) : (
          <View style={styles.memList}>
            {filtered.map((mem) => (
              <MemoryCard key={mem.id} mem={mem} onDelete={onDeleteMemory} />
            ))}
          </View>
        )}

        {/* Pagination */}
        {filtered.length > 0 && (
          <TouchableOpacity style={styles.loadMoreBtn} onPress={onLoadMore} disabled={memoryBusy}>
            <Text style={styles.loadMoreText}>
              {memoryBusy ? 'Loading…' : 'Load 50 more'}
            </Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </View>
  );
}

function MemoryCard({ mem, onDelete }: { mem: MemoryObject; onDelete: (id: string) => void }) {
  const typeBadge = TYPE_BADGE[mem.memory_type?.toLowerCase() || 'episodic'] || 'primary';
  const imp = typeof mem.importance === 'number' ? mem.importance : 0;
  const entities = (mem.entities as string[]) || [];
  const topics = (mem.topics as string[]) || [];

  return (
    <Card variant="default" style={styles.memCard} leftAccent={true} leftAccentColor={NEURAL.primary}>
      {/* Top: type + emotion */}
      <View style={styles.memMeta}>
        <Badge label={mem.memory_type || 'episodic'} variant={typeBadge} small />
        {mem.emotion ? <Badge label={mem.emotion} variant="ghost" small /> : null}
        <Text style={styles.memSource}>
          {mem.source ? `via ${mem.source}` : ''}
        </Text>
      </View>

      {/* Importance progress */}
      <ProgressBar
        value={imp}
        label="Importance"
        style={styles.importanceBar}
      />

      {/* Content */}
      <Text style={styles.memText}>{mem.content}</Text>

      {/* Entity tags */}
      {entities.length > 0 && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.tagRow}>
          {entities.slice(0, 6).map((e, i) => (
            <View key={i} style={styles.entityTag}>
              <Text style={styles.entityTagText}>@{e}</Text>
            </View>
          ))}
        </ScrollView>
      )}

      {/* Topic pills */}
      {topics.length > 0 && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.tagRow}>
          {topics.slice(0, 5).map((t, i) => (
            <View key={i} style={styles.topicTag}>
              <Text style={styles.topicTagText}>#{ t}</Text>
            </View>
          ))}
        </ScrollView>
      )}

      {/* Footer */}
      <View style={styles.memFooter}>
        <Text style={styles.memDate}>{formatDate(mem.timestamp)}</Text>
        <TouchableOpacity onPress={() => onDelete(mem.id)} style={styles.deleteBtn}>
          <Text style={styles.deleteText}>Delete</Text>
        </TouchableOpacity>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  scrollContent: { paddingBottom: SPACING['5xl'] },
  header: { paddingHorizontal: SPACING.lg, paddingTop: SPACING.lg, paddingBottom: SPACING.md },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, letterSpacing: -0.5 },
  subtitle: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, marginTop: 2 },

  searchBar: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md },

  filterRow: { paddingHorizontal: SPACING.lg, gap: SPACING.sm, marginBottom: SPACING.md },
  filterChip: {
    paddingHorizontal: SPACING.md,
    paddingVertical: 5,
    borderRadius: RADIUS.full,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },
  filterChipActive: {
    backgroundColor: `${NEURAL.primary}26`,
    borderColor: `${NEURAL.primary}60`,
  },
  filterChipText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, fontWeight: FONT_WEIGHT.medium },
  filterChipTextActive: { color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },

  composerCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.lg, gap: SPACING.sm },
  composerTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface },
  composerInput: { marginBottom: 0 },

  loader: { marginTop: SPACING['4xl'] },
  empty: { alignItems: 'center', paddingVertical: SPACING['3xl'], paddingHorizontal: SPACING.lg },
  emptyTitle: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface, marginBottom: SPACING.sm },
  emptyBody: { fontSize: FONT_SIZE.base, color: NEURAL.onSurfaceVariant, textAlign: 'center' },

  memList: { gap: SPACING.sm, paddingHorizontal: SPACING.lg },

  memCard: { gap: SPACING.sm },
  memMeta: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm },
  memSource: { marginLeft: 'auto', fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },
  importanceBar: { marginVertical: SPACING.xs },
  memText: { fontSize: FONT_SIZE.base, color: NEURAL.onSurface, lineHeight: FONT_SIZE.base * 1.6 },

  tagRow: { gap: SPACING.sm },
  entityTag: {
    backgroundColor: `${NEURAL.secondary}22`,
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.sm,
    paddingVertical: 2,
    borderWidth: 1,
    borderColor: `${NEURAL.secondary}40`,
  },
  entityTagText: { fontSize: FONT_SIZE.xs, color: NEURAL.secondary, fontWeight: FONT_WEIGHT.medium },
  topicTag: {
    backgroundColor: `${NEURAL.primary}18`,
    borderRadius: RADIUS.full,
    paddingHorizontal: SPACING.sm,
    paddingVertical: 2,
    borderWidth: 1,
    borderColor: `${NEURAL.primary}30`,
  },
  topicTagText: { fontSize: FONT_SIZE.xs, color: NEURAL.primary, fontWeight: FONT_WEIGHT.medium },

  memFooter: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: SPACING.xs },
  memDate: { fontSize: FONT_SIZE.xs, color: NEURAL.outline },
  deleteBtn: {
    paddingHorizontal: SPACING.sm,
    paddingVertical: 3,
    borderRadius: RADIUS.full,
    backgroundColor: `${NEURAL.error}22`,
    borderWidth: 1,
    borderColor: `${NEURAL.error}40`,
  },
  deleteText: { fontSize: FONT_SIZE.xs, color: NEURAL.error, fontWeight: FONT_WEIGHT.semibold },

  loadMoreBtn: {
    marginHorizontal: SPACING.lg,
    marginTop: SPACING.xl,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderRadius: RADIUS.full,
    paddingVertical: SPACING.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },
  loadMoreText: { fontSize: FONT_SIZE.sm, color: NEURAL.primary, fontWeight: FONT_WEIGHT.semibold },
});
