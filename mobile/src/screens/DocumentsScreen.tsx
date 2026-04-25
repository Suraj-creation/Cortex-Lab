/**
 * DocumentsScreen — Cortex Aurora PageIndex Documents
 * Upload, query, browse documents with light theme
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  FlatList,
  Alert,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { SearchBar } from '../components/ui/SearchBar';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Button } from '../components/ui/Button';
import { EmptyState } from '../components/ui/EmptyState';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import type { PageIndexDocument, PageIndexUsage } from '../../shared/core/api';

interface DocumentsScreenProps {
  documents: PageIndexDocument[];
  documentUsage: PageIndexUsage | null;
  onLoadDocuments: () => void;
  onDeleteDocument: (docId: string) => void;
  onUploadDocument: () => void;
  loadingView: boolean;
  api: any;
}

export function DocumentsScreen({
  documents,
  documentUsage,
  onLoadDocuments,
  onDeleteDocument,
  onUploadDocument,
  loadingView,
  api,
}: DocumentsScreenProps) {
  const [query, setQuery] = useState('');
  const [queryResults, setQueryResults] = useState<any[]>([]);
  const [querying, setQuerying] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<any>(null);

  const handleQuery = useCallback(async () => {
    if (!query.trim()) return;
    setQuerying(true);
    try {
      const res = await api.queryDocuments?.(query);
      setQueryResults(res?.sections || res?.results || []);
    } catch {
      Alert.alert('Error', 'Document query failed');
    }
    setQuerying(false);
  }, [api, query]);

  const loadDocDetail = useCallback(async (docId: string) => {
    try {
      const res = await api.getDocumentTree?.(docId);
      setSelectedDoc(res || { doc_id: docId });
    } catch {
      Alert.alert('Error', 'Failed to load document');
    }
  }, [api]);

  const confirmDelete = (docId: string) => {
    Alert.alert('Delete Document', 'This will permanently remove the document and its embeddings.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => onDeleteDocument(docId) },
    ]);
  };

  return (
    <View style={styles.container}>
      {/* Upload + stats area */}
      <View style={styles.topSection}>
        <View style={styles.topRow}>
          <Button
            label="Upload Document"
            onPress={onUploadDocument}
            size="sm"
            icon={<AppIcon name="cloud-upload-outline" size={14} color="#ffffff" />}
          />
          {documentUsage && (
            <View style={styles.usageRow}>
              <Text style={styles.usageText}>
                {documentUsage.pages_used ?? 0} pages · {documentUsage.queries_used ?? 0} queries
              </Text>
            </View>
          )}
        </View>
      </View>

      {/* Query section */}
      <View style={styles.querySection}>
        <SearchBar value={query} onChangeText={setQuery} onSubmit={handleQuery} placeholder="Query across all documents..." />
      </View>

      {/* Query Results */}
      {queryResults.length > 0 && (
        <View style={styles.queryResults}>
          <SectionHeader title="Query Results" subtitle={`${queryResults.length} matches`} />
          {queryResults.map((item, i) => (
            <Card key={`qr-${i}`} variant="accent" padding="md" style={styles.resultCard}>
              <Text style={styles.resultText} numberOfLines={4}>
                {item.content || item.text || JSON.stringify(item).slice(0, 200)}
              </Text>
              <View style={styles.resultMeta}>
                {item.page != null && <Badge label={`Page ${item.page}`} variant="primary" size="sm" />}
                {item.score != null && <Badge label={`${(item.score * 100).toFixed(0)}%`} variant="success" size="sm" />}
                {item.doc_id && <Text style={styles.resultDocName} numberOfLines={1}>{item.doc_id.slice(0, 12)}</Text>}
              </View>
            </Card>
          ))}
        </View>
      )}

      {/* Document detail overlay */}
      {selectedDoc && (
        <View style={styles.detailOverlay}>
          <View style={styles.detailSheet}>
            <View style={styles.detailHeader}>
              <Text style={styles.detailTitle} numberOfLines={1}>
                {selectedDoc.filename || selectedDoc.doc_id || 'Document'}
              </Text>
              <TouchableOpacity onPress={() => setSelectedDoc(null)} style={styles.closeBtn}>
                <AppIcon name="close" size={18} color="#64748b" />
              </TouchableOpacity>
            </View>
            <ScrollView style={styles.detailContent}>
              <View style={styles.detailMeta}>
                <Badge label={selectedDoc.status || 'ready'} variant="success" />
                {selectedDoc.estimated_pages != null && (
                  <Badge label={`${selectedDoc.estimated_pages} pages`} variant="primary" />
                )}
              </View>
              <Text style={styles.detailJson}>{JSON.stringify(selectedDoc, null, 2)}</Text>
            </ScrollView>
          </View>
        </View>
      )}

      {/* Document list */}
      <FlatList
        data={documents}
        keyExtractor={(d) => d.doc_id}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          loadingView ? <LoadingSpinner message="Loading documents..." /> :
          <EmptyState icon="file-document-outline" title="No Documents" message="Upload a PDF, Markdown, or text document to get started." />
        }
        renderItem={({ item }) => (
          <TouchableOpacity onPress={() => loadDocDetail(item.doc_id)} activeOpacity={0.7}>
            <Card variant="default" padding="md" style={styles.docCard}>
              <View style={styles.docHeader}>
                <AppIcon name="file-document-outline" size={18} color="#6366f1" />
                <Text style={styles.docTitle} numberOfLines={1}>{item.filename || 'Untitled'}</Text>
                <TouchableOpacity onPress={() => confirmDelete(item.doc_id)} style={styles.deleteBtn}>
                  <AppIcon name="delete-outline" size={16} color="#f43f5e" />
                </TouchableOpacity>
              </View>
              <View style={styles.docMeta}>
                <Badge label={item.status || 'ready'} variant={item.status === 'ready' ? 'success' : 'warning'} size="sm" />
                {item.estimated_pages > 0 && (
                  <Badge label={`${item.estimated_pages} pages`} variant="primary" size="sm" />
                )}
              </View>
              {item.uploaded_at && (
                <Text style={styles.docTime}>{new Date(item.uploaded_at).toLocaleDateString()}</Text>
              )}
            </Card>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },

  topSection: { paddingHorizontal: SPACING.lg, paddingVertical: SPACING.md, backgroundColor: '#ffffff', borderBottomWidth: 1, borderBottomColor: '#f1f5f9' },
  topRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  usageRow: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm },
  usageText: { fontSize: FONT_SIZE.xs, color: '#64748b', fontWeight: FONT_WEIGHT.medium },

  querySection: { paddingHorizontal: SPACING.lg, paddingVertical: SPACING.sm, backgroundColor: '#ffffff', borderBottomWidth: 1, borderBottomColor: '#f1f5f9' },

  queryResults: { paddingHorizontal: SPACING.lg, paddingVertical: SPACING.md, backgroundColor: '#fefce8', borderBottomWidth: 1, borderBottomColor: '#fef08a' },
  resultCard: { marginBottom: SPACING.sm },
  resultText: { fontSize: FONT_SIZE.sm, color: '#334155', lineHeight: 18 },
  resultMeta: { flexDirection: 'row', gap: SPACING.sm, marginTop: SPACING.sm, alignItems: 'center' },
  resultDocName: { fontSize: 10, color: '#94a3b8', flex: 1, fontFamily: 'monospace' },

  listContent: { padding: SPACING.lg, paddingBottom: SPACING['5xl'], gap: SPACING.md },

  docCard: { marginBottom: 0 },
  docHeader: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, marginBottom: SPACING.sm },
  docTitle: { flex: 1, fontSize: FONT_SIZE.md, fontWeight: FONT_WEIGHT.semibold, color: '#0f172a' },
  deleteBtn: { padding: SPACING.xs },
  docMeta: { flexDirection: 'row', gap: SPACING.sm, flexWrap: 'wrap', marginBottom: SPACING.xs },
  docTime: { fontSize: 10, color: '#94a3b8' },

  detailOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(15, 23, 42, 0.4)', justifyContent: 'flex-end', zIndex: 10 },
  detailSheet: { backgroundColor: '#ffffff', borderTopLeftRadius: RADIUS['3xl'], borderTopRightRadius: RADIUS['3xl'], maxHeight: '80%', ...SHADOWS.xl },
  detailHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: SPACING.xl, borderBottomWidth: 1, borderBottomColor: '#f1f5f9' },
  detailTitle: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.bold, color: '#0f172a', flex: 1 },
  closeBtn: { padding: SPACING.sm, backgroundColor: '#f1f5f9', borderRadius: RADIUS.lg },
  detailContent: { padding: SPACING.xl },
  detailMeta: { flexDirection: 'row', gap: SPACING.sm, marginBottom: SPACING.md },
  detailJson: { fontSize: 11, color: '#64748b', fontFamily: 'monospace', lineHeight: 16 },
});
