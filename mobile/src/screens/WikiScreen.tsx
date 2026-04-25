/**
 * WikiScreen — Cortex Aurora Personal Wiki Browser
 * Pages list, claims, search, rebuild, lint, compaction
 */
import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Alert,
  FlatList,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT, SHADOWS } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MetricCard } from '../components/ui/MetricCard';
import { SearchBar } from '../components/ui/SearchBar';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Button } from '../components/ui/Button';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { EmptyState } from '../components/ui/EmptyState';

type WikiTab = 'pages' | 'claims' | 'tools';

interface WikiScreenProps {
  api: any;
}

export function WikiScreen({ api }: WikiScreenProps) {
  const [activeTab, setActiveTab] = useState<WikiTab>('pages');
  const [pages, setPages] = useState<any[]>([]);
  const [claims, setClaims] = useState<any[]>([]);
  const [selectedPage, setSelectedPage] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [claimSearch, setClaimSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [lintStatus, setLintStatus] = useState<any>(null);
  const [compactionStatus, setCompactionStatus] = useState<any>(null);

  const loadPages = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getWikiPages?.();
      setPages(res?.pages || []);
    } catch {}
    setLoading(false);
  }, [api]);

  const searchWiki = useCallback(async () => {
    if (!searchQuery.trim()) { loadPages(); return; }
    setLoading(true);
    try {
      const res = await api.searchWiki?.(searchQuery);
      setPages(res?.results || res?.pages || []);
    } catch {}
    setLoading(false);
  }, [api, searchQuery, loadPages]);

  const loadPageDetail = useCallback(async (pageId: string) => {
    try {
      const res = await api.getWikiPage?.(pageId);
      setSelectedPage(res);
    } catch {
      Alert.alert('Error', 'Failed to load page details');
    }
  }, [api]);

  const searchClaims = useCallback(async () => {
    if (!claimSearch.trim()) return;
    setLoading(true);
    try {
      const res = await api.searchClaims?.(claimSearch);
      setClaims(res?.claims || res?.results || []);
    } catch {}
    setLoading(false);
  }, [api, claimSearch]);

  const loadClaims = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getClaimStats?.();
      setClaims(res?.claims || []);
    } catch {}
    setLoading(false);
  }, [api]);

  const triggerRebuild = useCallback(async () => {
    Alert.alert('Rebuild Wiki', 'This will rebuild the wiki from all memories. Continue?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Rebuild', onPress: async () => {
        setRebuilding(true);
        try {
          await api.rebuildWiki?.();
          Alert.alert('Success', 'Wiki rebuild initiated');
          loadPages();
        } catch { Alert.alert('Error', 'Failed to trigger rebuild'); }
        setRebuilding(false);
      }},
    ]);
  }, [api, loadPages]);

  const runLint = useCallback(async () => {
    try {
      await api.runWikiLint?.();
      const res = await api.getWikiLintLatest?.();
      setLintStatus(res);
      Alert.alert('Success', 'Wiki lint completed');
    } catch { Alert.alert('Error', 'Lint failed'); }
  }, [api]);

  const runCompaction = useCallback(async () => {
    try {
      await api.runWikiCompaction?.();
      const res = await api.getWikiCompactionLatest?.();
      setCompactionStatus(res);
      Alert.alert('Success', 'Wiki compaction completed');
    } catch { Alert.alert('Error', 'Compaction failed'); }
  }, [api]);

  useEffect(() => {
    loadPages();
  }, [loadPages]);

  useEffect(() => {
    if (activeTab === 'claims') loadClaims();
  }, [activeTab, loadClaims]);

  const tabs: { key: WikiTab; label: string; icon: string }[] = [
    { key: 'pages', label: 'Pages', icon: 'book-open-page-variant-outline' },
    { key: 'claims', label: 'Claims', icon: 'check-decagram-outline' },
    { key: 'tools', label: 'Tools', icon: 'wrench-outline' },
  ];

  return (
    <View style={styles.container}>
      {/* Tab bar */}
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.tab, activeTab === tab.key && styles.tabActive]}
            onPress={() => setActiveTab(tab.key)}
            activeOpacity={0.7}
          >
            <AppIcon name={tab.icon as any} size={16} color={activeTab === tab.key ? '#6366f1' : '#94a3b8'} />
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelActive]}>{tab.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* PAGES TAB */}
      {activeTab === 'pages' && !selectedPage && (
        <View style={styles.flex}>
          <View style={styles.searchArea}>
            <SearchBar value={searchQuery} onChangeText={setSearchQuery} onSubmit={searchWiki} placeholder="Search wiki pages..." />
          </View>
          <FlatList
            data={pages}
            keyExtractor={(p, i) => p.page_id || p.id || String(i)}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            refreshControl={<RefreshControl refreshing={loading} onRefresh={loadPages} tintColor="#6366f1" colors={['#6366f1']} />}
            renderItem={({ item }) => (
              <TouchableOpacity onPress={() => loadPageDetail(item.page_id || item.id)} activeOpacity={0.7}>
                <Card variant="default" padding="md" style={styles.pageCard}>
                  <Text style={styles.pageTitle}>{item.title || item.topic || 'Untitled'}</Text>
                  {item.summary && <Text style={styles.pageSummary} numberOfLines={2}>{item.summary}</Text>}
                  <View style={styles.pageFooter}>
                    {item.claim_count != null && <Badge label={`${item.claim_count} claims`} variant="primary" size="sm" />}
                    {item.topic && <Badge label={item.topic} variant="violet" size="sm" />}
                    {item.updated_at && <Text style={styles.pageTime}>{new Date(item.updated_at).toLocaleDateString()}</Text>}
                  </View>
                </Card>
              </TouchableOpacity>
            )}
            ListEmptyComponent={
              loading ? <LoadingSpinner message="Loading pages..." /> :
              <EmptyState icon="book-open-page-variant-outline" title="No Wiki Pages" message="Wiki pages are created when the Wiki Agent processes memories." />
            }
          />
        </View>
      )}

      {/* PAGE DETAIL */}
      {activeTab === 'pages' && selectedPage && (
        <ScrollView style={styles.flex} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <TouchableOpacity onPress={() => setSelectedPage(null)} style={styles.backBtn}>
            <AppIcon name="arrow-left" size={16} color="#6366f1" />
            <Text style={styles.backText}>Back to pages</Text>
          </TouchableOpacity>

          <Text style={styles.detailTitle}>{selectedPage.title || selectedPage.topic || 'Wiki Page'}</Text>
          {selectedPage.summary && <Text style={styles.detailSummary}>{selectedPage.summary}</Text>}
          {selectedPage.content && (
            <Card variant="outlined" padding="lg" style={styles.contentCard}>
              <Text style={styles.detailContent}>{selectedPage.content}</Text>
            </Card>
          )}
          {selectedPage.claims && selectedPage.claims.length > 0 && (
            <Card variant="outlined" padding="lg">
              <SectionHeader title="Linked Claims" subtitle={`${selectedPage.claims.length} claims`} />
              {selectedPage.claims.map((claim: any, i: number) => (
                <View key={i} style={styles.claimRow}>
                  <AppIcon name="check-circle-outline" size={14} color="#10b981" />
                  <Text style={styles.claimText}>{typeof claim === 'string' ? claim : claim.text || claim.content || JSON.stringify(claim)}</Text>
                </View>
              ))}
            </Card>
          )}
        </ScrollView>
      )}

      {/* CLAIMS TAB */}
      {activeTab === 'claims' && (
        <View style={styles.flex}>
          <View style={styles.searchArea}>
            <SearchBar value={claimSearch} onChangeText={setClaimSearch} onSubmit={searchClaims} placeholder="Search claims..." />
          </View>
          <FlatList
            data={claims}
            keyExtractor={(c, i) => c.id || String(i)}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => (
              <Card variant="default" padding="md" style={styles.claimCard}>
                <Text style={styles.claimContent}>{item.text || item.content || JSON.stringify(item)}</Text>
                <View style={styles.claimMeta}>
                  {item.source_page && <Badge label={item.source_page} variant="primary" size="sm" />}
                  {item.confidence && <Badge label={`${(item.confidence * 100).toFixed(0)}%`} variant="success" size="sm" />}
                </View>
              </Card>
            )}
            ListEmptyComponent={<EmptyState icon="check-decagram-outline" title="No Claims" message="Search for specific claims or they will populate as the wiki grows." />}
          />
        </View>
      )}

      {/* TOOLS TAB */}
      {activeTab === 'tools' && (
        <ScrollView style={styles.flex} contentContainerStyle={styles.scrollContent}>
          <SectionHeader title="Wiki Management Tools" icon={<AppIcon name="wrench-outline" size={16} color="#6366f1" />} />

          <Card variant="accent" padding="lg" style={styles.toolCard}>
            <View style={styles.toolHeader}>
              <AppIcon name="refresh" size={18} color="#6366f1" />
              <Text style={styles.toolTitle}>Rebuild Wiki</Text>
            </View>
            <Text style={styles.toolDesc}>Rebuild the entire wiki from all stored memories. This re-runs entity extraction, claim generation, and page synthesis.</Text>
            <Button label={rebuilding ? 'Rebuilding...' : 'Trigger Rebuild'} onPress={triggerRebuild} loading={rebuilding} size="sm" style={styles.toolBtn} />
          </Card>

          <Card variant="outlined" padding="lg" style={styles.toolCard}>
            <View style={styles.toolHeader}>
              <AppIcon name="text-search" size={18} color="#f59e0b" />
              <Text style={styles.toolTitle}>Wiki Lint</Text>
            </View>
            <Text style={styles.toolDesc}>Run quality checks on wiki pages: detect duplicates, stale claims, and inconsistencies.</Text>
            <Button label="Run Lint" onPress={runLint} variant="secondary" size="sm" style={styles.toolBtn} />
            {lintStatus && (
              <View style={styles.toolStatus}>
                <Badge label={lintStatus.status || 'completed'} variant="success" size="sm" />
                <Text style={styles.toolStatusText}>{lintStatus.issues_found ?? 0} issues found</Text>
              </View>
            )}
          </Card>

          <Card variant="outlined" padding="lg" style={styles.toolCard}>
            <View style={styles.toolHeader}>
              <AppIcon name="package-variant-closed" size={18} color="#8b5cf6" />
              <Text style={styles.toolTitle}>Wiki Compaction</Text>
            </View>
            <Text style={styles.toolDesc}>Merge overlapping pages, consolidate duplicate claims, and optimize page structure.</Text>
            <Button label="Run Compaction" onPress={runCompaction} variant="secondary" size="sm" style={styles.toolBtn} />
            {compactionStatus && (
              <View style={styles.toolStatus}>
                <Badge label={compactionStatus.status || 'completed'} variant="success" size="sm" />
                <Text style={styles.toolStatusText}>{compactionStatus.pages_merged ?? 0} pages merged</Text>
              </View>
            )}
          </Card>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },
  flex: { flex: 1 },

  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.sm,
    gap: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACING.xs,
    paddingVertical: SPACING.sm,
    borderRadius: RADIUS.lg,
  },
  tabActive: { backgroundColor: '#eef2ff' },
  tabLabel: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.medium, color: '#94a3b8' },
  tabLabelActive: { color: '#6366f1', fontWeight: FONT_WEIGHT.semibold },

  searchArea: {
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  listContent: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.md,
  },
  scrollContent: {
    padding: SPACING.lg,
    paddingBottom: SPACING['5xl'],
    gap: SPACING.md,
  },

  // Pages
  pageCard: { marginBottom: 0 },
  pageTitle: { fontSize: FONT_SIZE.md, fontWeight: FONT_WEIGHT.semibold, color: '#0f172a', marginBottom: SPACING.xs },
  pageSummary: { fontSize: FONT_SIZE.sm, color: '#64748b', lineHeight: 18, marginBottom: SPACING.sm },
  pageFooter: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, flexWrap: 'wrap' },
  pageTime: { fontSize: 10, color: '#94a3b8', marginLeft: 'auto' },

  // Page detail
  backBtn: { flexDirection: 'row', alignItems: 'center', gap: SPACING.xs, marginBottom: SPACING.md },
  backText: { fontSize: FONT_SIZE.sm, color: '#6366f1', fontWeight: FONT_WEIGHT.semibold },
  detailTitle: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: '#0f172a', marginBottom: SPACING.sm },
  detailSummary: { fontSize: FONT_SIZE.base, color: '#475569', lineHeight: 22, marginBottom: SPACING.lg },
  contentCard: { marginBottom: SPACING.md },
  detailContent: { fontSize: FONT_SIZE.base, color: '#334155', lineHeight: 22 },
  claimRow: { flexDirection: 'row', gap: SPACING.sm, paddingVertical: SPACING.sm, borderBottomWidth: 1, borderBottomColor: '#f8fafc' },
  claimText: { flex: 1, fontSize: FONT_SIZE.sm, color: '#475569', lineHeight: 18 },

  // Claims
  claimCard: { marginBottom: 0 },
  claimContent: { fontSize: FONT_SIZE.sm, color: '#334155', lineHeight: 18, marginBottom: SPACING.sm },
  claimMeta: { flexDirection: 'row', gap: SPACING.sm },

  // Tools
  toolCard: { marginBottom: 0 },
  toolHeader: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, marginBottom: SPACING.sm },
  toolTitle: { fontSize: FONT_SIZE.md, fontWeight: FONT_WEIGHT.semibold, color: '#0f172a' },
  toolDesc: { fontSize: FONT_SIZE.sm, color: '#64748b', lineHeight: 18, marginBottom: SPACING.md },
  toolBtn: { alignSelf: 'flex-start' },
  toolStatus: { flexDirection: 'row', alignItems: 'center', gap: SPACING.sm, marginTop: SPACING.sm },
  toolStatusText: { fontSize: FONT_SIZE.xs, color: '#64748b' },
});
