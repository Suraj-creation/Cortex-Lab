/**
 * MemoryScreen — mobile memory intake and browser.
 */
import React from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Alert,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { SearchBar } from '../components/ui/SearchBar';
import { Badge } from '../components/ui/Badge';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { EmptyState } from '../components/ui/EmptyState';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { SectionHeader } from '../components/ui/SectionHeader';
import type {
  AmbientClientSessionInfo,
  AmbientRetentionTrace,
  LocalMemoryJournalEntry,
  MemoryObject,
} from '../../shared/core/types';

interface MemoryScreenProps {
  memories: MemoryObject[];
  memoryJournal: LocalMemoryJournalEntry[];
  memorySearch: string;
  setMemorySearch: (v: string) => void;
  memoryDraft: string;
  setMemoryDraft: (v: string) => void;
  memoryBusy: boolean;
  loadingView: boolean;
  lastRetentionTrace: AmbientRetentionTrace | null;
  lastSession: AmbientClientSessionInfo | null;
  lastUploadedDocument: string;
  onSearch: () => void;
  onAddMemory: () => void;
  onUploadDocument: () => void;
  onDeleteMemory: (id: string) => void;
  onLoadMore: () => void;
}

function formatJournalAge(createdAt: number) {
  const deltaMinutes = Math.max(1, Math.round((Date.now() - createdAt) / 60000));
  if (deltaMinutes < 60) return `${deltaMinutes}m ago`;
  const hours = Math.round(deltaMinutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function MemoryScreen({
  memories,
  memoryJournal,
  memorySearch,
  setMemorySearch,
  memoryDraft,
  setMemoryDraft,
  memoryBusy,
  loadingView,
  lastRetentionTrace,
  lastSession,
  lastUploadedDocument,
  onSearch,
  onAddMemory,
  onUploadDocument,
  onDeleteMemory,
  onLoadMore,
}: MemoryScreenProps) {
  const confirmDelete = (id: string) => {
    Alert.alert(
      'Delete Memory',
      'Are you sure you want to delete this memory?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: () => onDeleteMemory(id) },
      ],
    );
  };

  const formatDate = (ts: number | string) => {
    const d = new Date(typeof ts === 'string' ? ts : ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const journalPreview = memoryJournal.slice(0, 4);

  const header = (
    <View style={styles.headerStack}>
      <Card variant="accent" padding="lg" style={styles.heroCard}>
        <View style={styles.heroTopRow}>
          <View style={styles.heroIcon}>
            <AppIcon name="brain" size={22} color="#4f46e5" />
          </View>
          <View style={styles.heroCopy}>
            <Text style={styles.heroEyebrow}>Memory Studio</Text>
            <Text style={styles.heroTitle}>Capture thoughts, files, and structured recall</Text>
            <Text style={styles.heroBody}>
              Text and supported files are processed through the same retention pipeline, tagged,
              sessionized, and mirrored into a device-side journal so you can see what was kept.
            </Text>
          </View>
        </View>

        <View style={styles.heroBadges}>
          <Badge label={`${memories.length} remote memories`} variant="primary" size="sm" />
          <Badge label={`${memoryJournal.length} device receipts`} variant="violet" size="sm" />
        </View>
      </Card>

      <Card variant="default" padding="lg">
        <SectionHeader
          title="Search Memory Base"
          subtitle="Recall stored objects and retrieval-ready entries"
          icon={<AppIcon name="magnify" size={16} color="#4f46e5" />}
        />
        <View style={styles.searchWrap}>
          <SearchBar
            value={memorySearch}
            onChangeText={setMemorySearch}
            onSubmit={onSearch}
            placeholder="Search memories, tags, entities..."
          />
        </View>
      </Card>

      <Card variant="default" padding="lg" style={styles.intakeCard}>
        <SectionHeader
          title="Add Memory"
          subtitle="Paste a thought or upload a supported file"
          icon={<AppIcon name="database-plus-outline" size={16} color="#4f46e5" />}
        />

        <TextInput
          value={memoryDraft}
          onChangeText={setMemoryDraft}
          placeholder="Type a memory, insight, observation, or instruction you want Cortex to keep..."
          multiline
          style={styles.addInput}
        />

        <View style={styles.actionRow}>
          <Button
            label={memoryBusy ? 'Saving...' : 'Save Text Memory'}
            onPress={onAddMemory}
            disabled={!memoryDraft.trim() || memoryBusy}
            loading={memoryBusy}
            fullWidth
            icon={<AppIcon name="content-save-outline" size={16} color="#ffffff" />}
            style={styles.primaryAction}
          />
        </View>

        <View style={styles.actionRow}>
          <Button
            label="Upload PDF / TXT / MD / JSON / CSV"
            onPress={onUploadDocument}
            variant="secondary"
            disabled={memoryBusy}
            fullWidth
            icon={<AppIcon name="file-upload-outline" size={16} color="#475569" />}
          />
        </View>

        <Text style={styles.intakeHint}>
          PDFs are sent to PageIndex plus memory receipt. Text-like files are parsed on-device and
          stored directly through the memory pipeline.
        </Text>
      </Card>

      {lastRetentionTrace ? (
        <Card variant="outlined" padding="lg">
          <SectionHeader
            title="Latest Intake Result"
            subtitle={lastUploadedDocument || 'Most recent memory submission'}
            icon={<AppIcon name="check-decagram-outline" size={16} color="#10b981" />}
          />
          <View style={styles.resultHeader}>
            <Badge
              label={lastRetentionTrace.memory_decision || 'structured'}
              variant={lastRetentionTrace.memory_decision === 'priority' ? 'success' : 'primary'}
              size="sm"
            />
            {lastSession?.session_id ? (
              <Text style={styles.sessionId}>Session {lastSession.session_id.slice(0, 16)}...</Text>
            ) : null}
          </View>
          <Text style={styles.resultCopy}>
            {lastUploadedDocument
              ? `The document "${lastUploadedDocument}" was processed and mirrored into the intake journal.`
              : 'This memory was passed through the retention pipeline and stored with session context.'}
          </Text>
          <View style={styles.tagWrap}>
            {(lastRetentionTrace.tags || []).slice(0, 5).map((tag) => (
              <Badge key={tag} label={tag.replace(/_/g, ' ')} variant="default" size="sm" />
            ))}
          </View>
        </Card>
      ) : null}

      <Card variant="outlined" padding="lg">
        <SectionHeader
          title="Device Intake Journal"
          subtitle="Recent receipts stored locally on this phone"
          icon={<AppIcon name="cellphone-arrow-down" size={16} color="#8b5cf6" />}
        />
        {journalPreview.length > 0 ? (
          journalPreview.map((entry) => (
            <View key={entry.id} style={styles.journalRow}>
              <View style={styles.journalIcon}>
                <AppIcon
                  name={
                    entry.kind === 'document'
                      ? 'file-document-outline'
                      : entry.kind === 'ambient'
                        ? 'microphone-outline'
                        : 'text-box-edit-outline'
                  }
                  size={16}
                  color="#4f46e5"
                />
              </View>
              <View style={styles.journalBody}>
                <View style={styles.journalTitleRow}>
                  <Text style={styles.journalTitle} numberOfLines={1}>{entry.title}</Text>
                  <Badge label={entry.status} variant={entry.status === 'error' ? 'error' : 'info'} size="sm" />
                </View>
                <Text style={styles.journalPreview} numberOfLines={2}>{entry.preview}</Text>
                <View style={styles.journalMetaRow}>
                  <Text style={styles.journalMeta}>{entry.source}</Text>
                  <Text style={styles.journalMeta}>{formatJournalAge(entry.createdAt)}</Text>
                </View>
              </View>
            </View>
          ))
        ) : (
          <Text style={styles.emptyJournal}>
            Local receipts will appear here after you save a text memory or ingest a file.
          </Text>
        )}
      </Card>

      <SectionHeader
        title="Stored Memories"
        subtitle="Remote long-term memory objects"
        icon={<AppIcon name="database-outline" size={16} color="#4f46e5" />}
      />
    </View>
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 12 : 20}
    >
      {loadingView ? (
        <LoadingSpinner fullScreen message="Loading memories..." />
      ) : (
        <FlatList
          data={memories}
          keyExtractor={(m) => m.id}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          refreshControl={
            <RefreshControl
              refreshing={false}
              onRefresh={onLoadMore}
              tintColor="#6366f1"
              colors={['#6366f1']}
            />
          }
          ListHeaderComponent={header}
          renderItem={({ item }) => (
            <Card variant="default" padding="md" style={styles.memoryCard}>
              <View style={styles.memoryHeader}>
                <View style={styles.memoryMeta}>
                  <Badge
                    label={item.source || 'manual'}
                    variant={item.source === 'ambient' ? 'violet' : item.source === 'chat' ? 'primary' : 'default'}
                    size="sm"
                  />
                  {item.importance !== undefined && (
                    <Badge
                      label={`★ ${(item.importance * 100).toFixed(0)}%`}
                      variant={item.importance > 0.7 ? 'success' : 'default'}
                      size="sm"
                    />
                  )}
                </View>
                <TouchableOpacity onPress={() => confirmDelete(item.id)} style={styles.deleteBtn}>
                  <AppIcon name="delete-outline" size={16} color="#f43f5e" />
                </TouchableOpacity>
              </View>

              <Text style={styles.memoryContent} numberOfLines={4}>
                {item.content}
              </Text>

              {(item.topics || []).length > 0 ? (
                <View style={styles.tagWrap}>
                  {item.topics.slice(0, 4).map((topic) => (
                    <Badge key={`${item.id}-${topic}`} label={topic} variant="default" size="sm" />
                  ))}
                </View>
              ) : null}

              <View style={styles.memoryFooter}>
                <Text style={styles.memoryDate}>
                  {item.timestamp ? formatDate(item.timestamp) : ''}
                </Text>
                {item.id ? (
                  <Text style={styles.memoryId} numberOfLines={1}>
                    {item.id.slice(0, 12)}...
                  </Text>
                ) : null}
              </View>
            </Card>
          )}
          ListEmptyComponent={
            <EmptyState
              icon="brain"
              title="No memories yet"
              message="Start by saving a text memory or uploading a supported file above."
            />
          }
        />
      )}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#e9eef8',
  },
  list: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.md,
  },
  headerStack: {
    gap: SPACING.md,
    marginBottom: SPACING.md,
  },
  heroCard: {
    gap: SPACING.md,
  },
  heroTopRow: {
    flexDirection: 'row',
    gap: SPACING.md,
    alignItems: 'flex-start',
  },
  heroIcon: {
    width: 52,
    height: 52,
    borderRadius: 18,
    backgroundColor: '#f7faff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroCopy: {
    flex: 1,
    gap: SPACING.xs,
  },
  heroEyebrow: {
    fontSize: FONT_SIZE.xs,
    color: '#6366f1',
    fontWeight: FONT_WEIGHT.bold,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  heroTitle: {
    fontSize: FONT_SIZE.xl,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
  },
  heroBody: {
    fontSize: FONT_SIZE.sm,
    lineHeight: 19,
    color: '#475569',
  },
  heroBadges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.xs,
  },
  searchWrap: {
    marginTop: SPACING.sm,
  },
  intakeCard: {
    gap: SPACING.sm,
  },
  addInput: {
    marginTop: SPACING.sm,
  },
  actionRow: {
    marginTop: SPACING.sm,
  },
  primaryAction: {
    minHeight: 48,
  },
  intakeHint: {
    marginTop: SPACING.sm,
    fontSize: FONT_SIZE.xs,
    lineHeight: 17,
    color: '#64748b',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  sessionId: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
  },
  resultCopy: {
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    lineHeight: 19,
  },
  journalRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#edf2fb',
  },
  journalIcon: {
    width: 38,
    height: 38,
    borderRadius: 14,
    backgroundColor: '#f5f7fe',
    alignItems: 'center',
    justifyContent: 'center',
  },
  journalBody: {
    flex: 1,
    gap: 4,
  },
  journalTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  journalTitle: {
    flex: 1,
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },
  journalPreview: {
    fontSize: FONT_SIZE.sm,
    lineHeight: 18,
    color: '#475569',
  },
  journalMetaRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    flexWrap: 'wrap',
  },
  journalMeta: {
    fontSize: FONT_SIZE.xs,
    color: '#94a3b8',
  },
  emptyJournal: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
    lineHeight: 18,
  },
  memoryCard: {
    marginBottom: 0,
  },
  memoryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: SPACING.sm,
  },
  memoryMeta: {
    flexDirection: 'row',
    gap: 4,
    flexWrap: 'wrap',
    flex: 1,
  },
  deleteBtn: {
    padding: SPACING.xs,
    borderRadius: RADIUS.sm,
  },
  memoryContent: {
    fontSize: FONT_SIZE.base,
    color: '#1e293b',
    lineHeight: 20,
    marginBottom: SPACING.sm,
  },
  memoryFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: SPACING.sm,
  },
  memoryDate: {
    fontSize: 10,
    color: '#94a3b8',
    fontWeight: FONT_WEIGHT.medium,
  },
  memoryId: {
    fontSize: 10,
    color: '#cbd5e1',
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }),
  },
  tagWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.xs,
    marginTop: SPACING.sm,
  },
});
