/**
 * DocumentsScreen — Neural Dark PageIndex Documents
 * Stitch ref: fe0b1a2ad2f44225b54dfb5339f9598c
 */
import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { ProgressBar } from '../components/ui/ProgressBar';
import { AppIcon } from '../components/ui/AppIcon';
import type { PageIndexDocument, PageIndexUsage } from '../../shared/core/api';

const DOC_STATUS_VARIANT: Record<string, 'success' | 'warning' | 'error' | 'primary'> = {
  ready:      'success',
  processing: 'warning',
  failed:     'error',
  pending:    'primary',
};

interface DocumentsScreenProps {
  documents: PageIndexDocument[];
  pageIndexUsage: PageIndexUsage | null;
  pageIndexEnabled: boolean | null;
  documentQuery: string;
  setDocumentQuery: (v: string) => void;
  documentAnswer: string;
  documentSections: { page: number; content: string; doc_id: string; score: number }[];
  documentTreeDocId: string | null;
  documentTreePreview: string[];
  documentsBusy: boolean;
  documentQueryBusy: boolean;
  loadingView: boolean;
  onUpload: () => void;
  onDeleteDocument: (id: string) => void;
  onToggleTree: (id: string) => void;
  onRunQuery: () => void;
  onClearAnswer: () => void;
  onRefresh: () => void;
}

export function DocumentsScreen({
  documents,
  pageIndexUsage,
  pageIndexEnabled,
  documentQuery,
  setDocumentQuery,
  documentAnswer,
  documentSections,
  documentTreeDocId,
  documentTreePreview,
  documentsBusy,
  documentQueryBusy,
  loadingView,
  onUpload,
  onDeleteDocument,
  onToggleTree,
  onRunQuery,
  onClearAnswer,
  onRefresh,
}: DocumentsScreenProps) {
  return (
    <View style={s.container}>
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={s.header}>
          <View style={s.headerLeft}>
            <Text style={s.title}>Documents</Text>
            {pageIndexEnabled != null && (
              <Badge label={pageIndexEnabled ? 'PageIndex Active' : 'PageIndex Off'} variant={pageIndexEnabled ? 'success' : 'ghost'} small />
            )}
          </View>
            <TouchableOpacity onPress={onRefresh} disabled={loadingView || documentsBusy} style={s.refreshButton}>
              <AppIcon name="refresh" size={22} color={NEURAL.primary} style={s.refreshIcon} />
            </TouchableOpacity>
        </View>

        {/* Usage Meter */}
        {pageIndexUsage && (
          <Card variant="outlined" style={s.usageCard}>
            <Text style={s.cardTitle}>PageIndex Usage — {pageIndexUsage.month}</Text>
            <ProgressBar
              value={pageIndexUsage.queries_used / Math.max(1, pageIndexUsage.queries_limit)}
              label="Queries"
              total={pageIndexUsage.queries_limit}
              style={s.progressBar}
            />
            <ProgressBar
              value={pageIndexUsage.pages_used / Math.max(1, pageIndexUsage.pages_limit)}
              label="Pages"
              total={pageIndexUsage.pages_limit}
              style={s.progressBar}
            />
          </Card>
        )}

        {/* Upload zone */}
        <TouchableOpacity
          onPress={onUpload}
          disabled={documentsBusy}
          style={[s.uploadZone, documentsBusy && s.uploadZoneBusy]}
          activeOpacity={0.8}
        >
          <AppIcon name="cloud-upload-outline" size={36} color={NEURAL.primary} style={s.uploadIcon} />
          <Text style={s.uploadTitle}>{documentsBusy ? 'Uploading…' : 'Tap to Upload PDF'}</Text>
          <Text style={s.uploadHint}>PDF documents • Max 50MB</Text>
        </TouchableOpacity>

        {/* Document Query */}
        <Card variant="outlined" style={s.queryCard}>
          <Text style={s.cardTitle}>Ask Your Documents</Text>
          <TextInput
            placeholder="Ask a question across indexed documents…"
            value={documentQuery}
            onChangeText={setDocumentQuery}
            multiline
            style={s.queryInput}
          />
          <View style={s.queryBtns}>
            <Button
              label={documentQueryBusy ? 'Querying…' : 'Run Query'}
              onPress={onRunQuery}
              disabled={documentQueryBusy || !documentQuery.trim()}
              loading={documentQueryBusy}
              size="sm"
            />
            <Button
              label="Clear"
              variant="secondary"
              size="sm"
              onPress={onClearAnswer}
              disabled={documentQueryBusy}
            />
          </View>
        </Card>

        {/* Query Answer */}
        {documentAnswer ? (
          <Card variant="elevated" style={s.answerCard} leftAccent leftAccentColor={NEURAL.tertiary}>
            <Text style={s.answerTitle}>Answer</Text>
            <Text style={s.answerText}>{documentAnswer}</Text>
            {documentSections.length > 0 && (
              <View style={s.sourcesList}>
                <Text style={s.sourcesLabel}>Sources</Text>
                {documentSections.slice(0, 5).map((sec, i) => (
                  <View key={`${sec.doc_id}-${sec.page}-${i}`} style={s.sourceRow}>
                    <Text style={s.sourceMeta}>p.{sec.page} · {(sec.score * 100).toFixed(0)}%</Text>
                    <Text style={s.sourceText} numberOfLines={3}>{sec.content}</Text>
                  </View>
                ))}
              </View>
            )}
          </Card>
        ) : null}

        {/* Document list */}
        {loadingView ? (
          <ActivityIndicator color={NEURAL.primary} size="large" style={s.loader} />
        ) : documents.length === 0 ? (
          <View style={s.empty}>
            <AppIcon name="file-document-outline" size={42} color={NEURAL.onSurfaceVariant} style={s.emptyIcon} />
            <Text style={s.emptyTitle}>No documents yet</Text>
            <Text style={s.emptyBody}>Upload a PDF to enable document retrieval.</Text>
          </View>
        ) : (
          <View style={s.docList}>
            {documents.map((doc) => (
              <DocumentCard
                key={doc.doc_id}
                doc={doc}
                isTreeOpen={documentTreeDocId === doc.doc_id}
                treePreview={documentTreeDocId === doc.doc_id ? documentTreePreview : []}
                onDelete={onDeleteDocument}
                onToggleTree={onToggleTree}
                busy={documentsBusy}
              />
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

function DocumentCard({
  doc,
  isTreeOpen,
  treePreview,
  onDelete,
  onToggleTree,
  busy,
}: {
  doc: PageIndexDocument;
  isTreeOpen: boolean;
  treePreview: string[];
  onDelete: (id: string) => void;
  onToggleTree: (id: string) => void;
  busy: boolean;
}) {
  const variant = DOC_STATUS_VARIANT[doc.status] || 'primary';
  return (
    <Card variant="default" style={s.docCard}>
      {/* Header row */}
      <View style={s.docHeader}>
        <AppIcon name="file-document-outline" size={28} color={NEURAL.onSurfaceVariant} style={s.docIcon} />
        <View style={s.docInfo}>
          <Text style={s.docName} numberOfLines={2}>{doc.filename}</Text>
          <View style={s.docMeta}>
            <Badge label={doc.status} variant={variant} small />
            <Badge label={`${doc.estimated_pages} pages`} variant="ghost" small />
          </View>
        </View>
      </View>

      {/* Tree toggle */}
      <TouchableOpacity
        onPress={() => onToggleTree(doc.doc_id)}
        disabled={busy}
        style={s.treeToggle}
      >
        <Text style={s.treeToggleText}>
          {isTreeOpen ? 'Hide Document Tree' : 'View Document Tree'}
        </Text>
      </TouchableOpacity>

      {/* Tree preview */}
      {isTreeOpen && treePreview.length > 0 && (
        <View style={s.treePreview}>
          {treePreview.slice(0, 8).map((line, i) => (
            <Text key={i} style={s.treePreviewLine} numberOfLines={2}>{line}</Text>
          ))}
        </View>
      )}

      {/* Actions */}
      <View style={s.docActions}>
        <Button
          label="Delete"
          size="xs"
          variant="error"
          onPress={() => onDelete(doc.doc_id)}
          disabled={busy}
        />
      </View>
    </Card>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  scroll: { paddingBottom: SPACING['5xl'] },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.md,
  },
  headerLeft: { gap: 6 },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  refreshButton: { padding: SPACING.xs },
  refreshIcon: { marginVertical: 1 },

  usageCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  cardTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  progressBar: { marginBottom: SPACING.xs },

  uploadZone: {
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.md,
    borderWidth: 2,
    borderColor: `${NEURAL.primary}60`,
    borderStyle: 'dashed',
    borderRadius: RADIUS.xl,
    backgroundColor: `${NEURAL.primary}10`,
    alignItems: 'center',
    paddingVertical: SPACING['3xl'],
    gap: SPACING.sm,
  },
  uploadZoneBusy: { opacity: 0.6 },
  uploadIcon: { marginBottom: 1 },
  uploadTitle: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface },
  uploadHint: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },

  queryCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  queryInput: { marginBottom: 0 },
  queryBtns: { flexDirection: 'row', gap: SPACING.sm },

  answerCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  answerTitle: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.bold, color: NEURAL.tertiary, textTransform: 'uppercase', letterSpacing: 0.5 },
  answerText: { fontSize: FONT_SIZE.base, color: NEURAL.onSurface, lineHeight: FONT_SIZE.base * 1.65 },
  sourcesList: { gap: SPACING.sm, marginTop: SPACING.sm },
  sourcesLabel: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, textTransform: 'uppercase', letterSpacing: 0.5 },
  sourceRow: {
    padding: SPACING.sm,
    backgroundColor: NEURAL.surfaceContainerLow,
    borderRadius: RADIUS.md,
    gap: 3,
  },
  sourceMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.primary },
  sourceText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface },

  loader: { marginVertical: SPACING['4xl'] },
  empty: { alignItems: 'center', paddingVertical: SPACING['3xl'] },
  emptyIcon: { marginBottom: SPACING.md },
  emptyTitle: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface, marginBottom: SPACING.sm },
  emptyBody: { fontSize: FONT_SIZE.base, color: NEURAL.onSurfaceVariant, textAlign: 'center', paddingHorizontal: SPACING.xl },

  docList: { paddingHorizontal: SPACING.lg, gap: SPACING.md },
  docCard: { gap: SPACING.sm },
  docHeader: { flexDirection: 'row', gap: SPACING.sm, alignItems: 'flex-start' },
  docIcon: { marginTop: 2 },
  docInfo: { flex: 1, gap: 4 },
  docName: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface },
  docMeta: { flexDirection: 'row', gap: SPACING.sm },
  treeToggle: { paddingVertical: SPACING.xs },
  treeToggleText: { fontSize: FONT_SIZE.sm, color: NEURAL.primary, fontWeight: FONT_WEIGHT.medium },
  treePreview: {
    backgroundColor: NEURAL.surfaceContainerLow,
    borderRadius: RADIUS.md,
    padding: SPACING.sm,
    gap: 4,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
  },
  treePreviewLine: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, lineHeight: FONT_SIZE.xs * 1.5 },
  docActions: { flexDirection: 'row', justifyContent: 'flex-end' },
});
