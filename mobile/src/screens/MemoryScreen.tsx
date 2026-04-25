/**
 * MemoryScreen — Cortex Aurora Memory Browser
 * Light theme with search, add, delete, quality evaluation
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
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { SearchBar } from '../components/ui/SearchBar';
import { Badge } from '../components/ui/Badge';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { EmptyState } from '../components/ui/EmptyState';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import type {
  AmbientClientSessionInfo,
  AmbientRetentionTrace,
  MemoryObject,
} from '../../shared/core/types';

interface MemoryScreenProps {
  memories: MemoryObject[];
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

export function MemoryScreen({
  memories,
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

  return (
    <View style={styles.container}>
      {/* Search bar */}
      <View style={styles.searchSection}>
        <SearchBar
          value={memorySearch}
          onChangeText={setMemorySearch}
          onSubmit={onSearch}
          placeholder="Search memories..."
        />
        <Badge
          label={`${memories.length} memories`}
          variant="primary"
          size="sm"
        />
      </View>

      {/* Add memory input */}
      <View style={styles.addSection}>
        <TextInput
          value={memoryDraft}
          onChangeText={setMemoryDraft}
          placeholder="Add a new memory..."
          multiline
          style={styles.addInput}
        />
        <Button
          label={memoryBusy ? 'Adding...' : 'Add Memory'}
          onPress={onAddMemory}
          size="sm"
          disabled={!memoryDraft.trim() || memoryBusy}
          loading={memoryBusy}
          icon={<AppIcon name="plus" size={14} color="#ffffff" />}
        />
        <Button
          label="Upload PDF to Memory"
          onPress={onUploadDocument}
          size="sm"
          variant="outline"
          disabled={memoryBusy}
          icon={<AppIcon name="file-upload-outline" size={14} color="#6366f1" />}
        />
      </View>

      {lastRetentionTrace ? (
        <View style={styles.intakeSection}>
          <Card variant="accent" padding="md">
            <View style={styles.intakeHeader}>
              <Text style={styles.intakeTitle}>Latest intake</Text>
              <Badge
                label={lastRetentionTrace.memory_decision || 'structured'}
                variant={lastRetentionTrace.memory_decision === 'priority' ? 'success' : 'primary'}
                size="sm"
              />
            </View>
            <Text style={styles.intakeCopy}>
              {lastUploadedDocument
                ? `Document "${lastUploadedDocument}" was queued for PageIndex retrieval and memory refinement.`
                : 'This memory was sessionized, tagged, and stored through the same retention pipeline used by the companion flow.'}
            </Text>
            <View style={styles.intakeMeta}>
              {lastRetentionTrace.tags?.slice(0, 4).map((tag) => (
                <Badge key={tag} label={tag} variant="default" size="sm" />
              ))}
            </View>
            {lastSession?.session_id ? (
              <Text style={styles.intakeSession}>Session {lastSession.session_id.slice(0, 18)}…</Text>
            ) : null}
          </Card>
        </View>
      ) : null}

      {/* Memory list */}
      {loadingView ? (
        <LoadingSpinner fullScreen message="Loading memories..." />
      ) : (
        <FlatList
          data={memories}
          keyExtractor={(m) => m.id}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={false}
              onRefresh={onLoadMore}
              tintColor="#6366f1"
              colors={['#6366f1']}
            />
          }
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

              <View style={styles.memoryFooter}>
                <Text style={styles.memoryDate}>
                  {item.timestamp ? formatDate(item.timestamp) : ''}
                </Text>
                {item.id && (
                  <Text style={styles.memoryId} numberOfLines={1}>
                    {item.id.slice(0, 12)}...
                  </Text>
                )}
              </View>
            </Card>
          )}
          ListEmptyComponent={
            <EmptyState
              icon="brain"
              title="No memories yet"
              message="Add your first memory above, or ingest content through the chat."
            />
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  searchSection: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    gap: SPACING.sm,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  addSection: {
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
    gap: SPACING.sm,
  },
  intakeSection: {
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
  },
  intakeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  intakeTitle: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.bold,
    color: '#0f172a',
  },
  intakeCopy: {
    fontSize: FONT_SIZE.sm,
    color: '#334155',
    lineHeight: 18,
  },
  intakeMeta: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.xs,
    marginTop: SPACING.sm,
  },
  intakeSession: {
    marginTop: SPACING.sm,
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
  },
  addInput: {
    flex: 1,
  },
  list: {
    padding: SPACING.lg,
    gap: SPACING.md,
    paddingBottom: SPACING['5xl'],
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
});
