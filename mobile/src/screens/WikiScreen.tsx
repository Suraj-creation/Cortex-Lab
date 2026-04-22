/**
 * WikiScreen — Neural Dark personal wiki and claim intelligence surface.
 * Includes page search/detail, claim search, and lint/compaction controls.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
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
import { AppIcon } from '../components/ui/AppIcon';
import type { ApiClient } from '../../shared/core/api';
import type {
  WikiCompactionSummary,
  WikiLintSummary,
  WikiPageInfo,
} from '../../shared/core/types';

interface WikiScreenProps {
  api: ApiClient;
}

interface WikiStats {
  total_pages: number;
  total_topics: number;
  total_linked_claims: number;
}

interface ClaimStats {
  total: number;
  active: number;
  topics: number;
}

interface ClaimResult {
  id: string;
  text: string;
  confidence: number;
  source_ids: string[];
  topic: string;
}

function formatIso(iso: string | null): string {
  if (!iso) return 'never';
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return iso;
  const delta = Date.now() - ts;
  if (delta < 8000) return 'just now';
  if (delta < 60000) return `${Math.round(delta / 1000)}s ago`;
  if (delta < 3600000) return `${Math.round(delta / 60000)}m ago`;
  return `${Math.round(delta / 3600000)}h ago`;
}

function truncate(content: string, max: number): string {
  if (content.length <= max) return content;
  return `${content.slice(0, max)}...`;
}

export function WikiScreen({ api }: WikiScreenProps) {
  const [pages, setPages] = useState<WikiPageInfo[]>([]);
  const [searchResults, setSearchResults] = useState<WikiPageInfo[]>([]);
  const [selectedPage, setSelectedPage] = useState<WikiPageInfo | null>(null);
  const [wikiStats, setWikiStats] = useState<WikiStats | null>(null);

  const [claimStats, setClaimStats] = useState<ClaimStats | null>(null);
  const [claimResults, setClaimResults] = useState<ClaimResult[]>([]);

  const [lintSummary, setLintSummary] = useState<WikiLintSummary | null>(null);
  const [compactionSummary, setCompactionSummary] = useState<WikiCompactionSummary | null>(null);
  const [rebuildSummary, setRebuildSummary] = useState<{
    processed: number;
    pages_created: number;
    claims_linked: number;
  } | null>(null);

  const [searchDraft, setSearchDraft] = useState('');
  const [claimSearchDraft, setClaimSearchDraft] = useState('');

  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [maintenanceBusy, setMaintenanceBusy] = useState(false);
  const [error, setError] = useState('');

  const visiblePages = useMemo(
    () => (searchDraft.trim() ? searchResults : pages),
    [pages, searchDraft, searchResults],
  );

  const loadWikiOverview = useCallback(async () => {
    setLoading(true);
    try {
      const [pagesResult, claimStatsResult, lintResult, compactionResult] = await Promise.allSettled([
        api.listWikiPages(),
        api.getClaimStats(),
        api.getWikiLintLatest(),
        api.getWikiCompactionLatest(),
      ]);

      if (pagesResult.status === 'fulfilled') {
        setPages(pagesResult.value.pages || []);
        setSearchResults(pagesResult.value.pages || []);
        setWikiStats((pagesResult.value.stats as WikiStats) || null);
        if (pagesResult.value.error) {
          setError(String(pagesResult.value.error));
        }
      }

      if (claimStatsResult.status === 'fulfilled') {
        setClaimStats(claimStatsResult.value);
      }

      if (lintResult.status === 'fulfilled') {
        setLintSummary(lintResult.value);
      }

      if (compactionResult.status === 'fulfilled') {
        setCompactionSummary(compactionResult.value);
      }

      if (
        pagesResult.status === 'fulfilled'
        || claimStatsResult.status === 'fulfilled'
        || lintResult.status === 'fulfilled'
        || compactionResult.status === 'fulfilled'
      ) {
        setError('');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    void loadWikiOverview();
  }, [loadWikiOverview]);

  const runWikiSearch = useCallback(async () => {
    const query = searchDraft.trim();
    if (!query) {
      setSearchResults(pages);
      return;
    }

    setBusy(true);
    try {
      const result = await api.searchWiki(query, true);
      setSearchResults(result.results || []);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api, pages, searchDraft]);

  const openPage = useCallback(async (pageId: string) => {
    setBusy(true);
    try {
      const detail = await api.getWikiPage(pageId);
      setSelectedPage(detail);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api]);

  const runClaimSearch = useCallback(async () => {
    const query = claimSearchDraft.trim();
    if (!query) {
      setClaimResults([]);
      return;
    }

    setBusy(true);
    try {
      const result = await api.searchClaims(query, 0.5);
      setClaimResults(result.claims || []);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [api, claimSearchDraft]);

  const runLint = useCallback(async () => {
    setMaintenanceBusy(true);
    try {
      await api.runWikiLint();
      const latest = await api.getWikiLintLatest();
      setLintSummary(latest);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMaintenanceBusy(false);
    }
  }, [api]);

  const runCompaction = useCallback(async () => {
    setMaintenanceBusy(true);
    try {
      await api.runWikiCompaction();
      const latest = await api.getWikiCompactionLatest();
      setCompactionSummary(latest);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMaintenanceBusy(false);
    }
  }, [api]);

  const runRebuild = useCallback(async () => {
    setMaintenanceBusy(true);
    try {
      const summary = await api.rebuildWikiFromMemories(300, 8);
      setRebuildSummary({
        processed: summary.processed,
        pages_created: summary.pages_created,
        claims_linked: summary.claims_linked,
      });
      await loadWikiOverview();
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMaintenanceBusy(false);
    }
  }, [api, loadWikiOverview]);

  return (
    <View style={s.container}>
      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>
        <View style={s.header}>
          <View>
            <Text style={s.title}>Wiki Intelligence</Text>
            <Text style={s.subtitle}>Pages, claims, and semantic maintenance</Text>
          </View>
          <TouchableOpacity style={s.refreshButton} onPress={() => void loadWikiOverview()} disabled={busy || loading || maintenanceBusy}>
            <AppIcon name="refresh" size={20} color={NEURAL.primary} style={s.refreshIcon} />
          </TouchableOpacity>
        </View>

        <View style={s.badgeRowWrap}>
          <Badge label={`Pages ${wikiStats?.total_pages ?? 0}`} variant="info" small />
          <Badge label={`Topics ${wikiStats?.total_topics ?? 0}`} variant="secondary" small />
          <Badge label={`Claims ${wikiStats?.total_linked_claims ?? claimStats?.total ?? 0}`} variant="primary" small />
        </View>

        {error ? (
          <Card variant="outlined" style={s.errorCard}>
            <Text style={s.errorText}>{error}</Text>
          </Card>
        ) : null}

        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Page Search</Text>
          <TextInput
            placeholder="Search wiki pages or concepts..."
            value={searchDraft}
            onChangeText={setSearchDraft}
            multiline
            style={s.input}
          />
          <View style={s.rowBtns}>
            <Button
              label="Search"
              size="sm"
              onPress={() => void runWikiSearch()}
              disabled={busy}
            />
            <Button
              label="Clear"
              size="sm"
              variant="secondary"
              onPress={() => {
                setSearchDraft('');
                setSearchResults(pages);
              }}
              disabled={busy}
            />
          </View>
        </Card>

        <Card variant="outlined" style={s.card}>
          <View style={s.cardHeader}>
            <Text style={s.cardTitle}>Wiki Pages</Text>
            <Badge label={`${visiblePages.length} shown`} variant="info" small />
          </View>

          {loading ? (
            <ActivityIndicator color={NEURAL.primary} />
          ) : visiblePages.length > 0 ? (
            <View style={s.pageList}>
              {visiblePages.slice(0, 20).map((page) => (
                <TouchableOpacity
                  key={page.id}
                  style={[
                    s.pageRow,
                    selectedPage?.id === page.id && s.pageRowActive,
                  ]}
                  onPress={() => void openPage(page.id)}
                  disabled={busy}
                  activeOpacity={0.82}
                >
                  <View style={s.pageRowHead}>
                    <Text style={s.pageTitle} numberOfLines={1}>{page.title}</Text>
                    <Badge label={`v${page.version}`} variant="ghost" small />
                  </View>
                  <Text style={s.pagePreview} numberOfLines={2}>{truncate(page.content || '', 220)}</Text>
                  <View style={s.topicRow}>
                    {(page.topics || []).slice(0, 3).map((topic) => (
                      <Badge key={`${page.id}-${topic}`} label={topic} variant="tertiary" small />
                    ))}
                  </View>
                </TouchableOpacity>
              ))}
            </View>
          ) : (
            <Text style={s.empty}>No wiki pages found.</Text>
          )}
        </Card>

        {selectedPage ? (
          <Card variant="elevated" style={s.card} leftAccent leftAccentColor={NEURAL.tertiary}>
            <Text style={s.selectedTitle}>{selectedPage.title}</Text>
            <Text style={s.selectedMeta}>
              Updated {formatIso(selectedPage.updated_at)} · v{selectedPage.version}
            </Text>
            <Text style={s.selectedBody}>{selectedPage.content}</Text>
            <View style={s.topicRow}>
              {(selectedPage.topics || []).slice(0, 6).map((topic) => (
                <Badge key={`selected-${topic}`} label={topic} variant="tertiary" small />
              ))}
            </View>
          </Card>
        ) : null}

        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Claim Search</Text>
          <TextInput
            placeholder="Search atomic claims..."
            value={claimSearchDraft}
            onChangeText={setClaimSearchDraft}
            multiline
            style={s.input}
          />
          <View style={s.badgeRowWrap}>
            <Badge label={`Total ${claimStats?.total ?? 0}`} variant="info" small />
            <Badge label={`Active ${claimStats?.active ?? 0}`} variant="success" small />
            <Badge label={`Topics ${claimStats?.topics ?? 0}`} variant="secondary" small />
          </View>
          <View style={s.rowBtns}>
            <Button label="Search Claims" size="sm" onPress={() => void runClaimSearch()} disabled={busy} />
            <Button label="Reset" size="sm" variant="secondary" onPress={() => { setClaimSearchDraft(''); setClaimResults([]); }} disabled={busy} />
          </View>

          {claimResults.length > 0 ? (
            <View style={s.claimList}>
              {claimResults.slice(0, 8).map((claim) => (
                <View key={claim.id} style={s.claimRow}>
                  <View style={s.claimHead}>
                    <Badge label={claim.topic || 'unknown'} variant="primary" small />
                    <Text style={s.claimConfidence}>{Math.round(claim.confidence * 100)}%</Text>
                  </View>
                  <Text style={s.claimText}>{claim.text}</Text>
                </View>
              ))}
            </View>
          ) : null}
        </Card>

        <Card variant="outlined" style={s.card}>
          <Text style={s.cardTitle}>Maintenance</Text>
          <View style={s.maintenanceGrid}>
            <View style={s.maintenanceTile}>
              <Text style={s.tileLabel}>Lint</Text>
              <Text style={s.tileValue}>{lintSummary?.issue_total ?? 0} issues</Text>
              <Text style={s.tileMeta}>Checked {formatIso(lintSummary?.checked_at ?? null)}</Text>
            </View>
            <View style={s.maintenanceTile}>
              <Text style={s.tileLabel}>Compaction</Text>
              <Text style={s.tileValue}>{compactionSummary?.sections_compacted ?? 0} compacted</Text>
              <Text style={s.tileMeta}>Checked {formatIso(compactionSummary?.checked_at ?? null)}</Text>
            </View>
          </View>
          <View style={s.rowBtns}>
            <Button
              label={maintenanceBusy ? 'Running...' : 'Run Lint Sweep'}
              size="sm"
              variant="outline"
              onPress={() => void runLint()}
              disabled={maintenanceBusy}
              loading={maintenanceBusy}
            />
            <Button
              label={maintenanceBusy ? 'Running...' : 'Run Compaction'}
              size="sm"
              variant="secondary"
              onPress={() => void runCompaction()}
              disabled={maintenanceBusy}
            />
            <Button
              label={maintenanceBusy ? 'Running...' : 'Rebuild From Memories'}
              size="sm"
              variant="ghost"
              onPress={() => void runRebuild()}
              disabled={maintenanceBusy}
            />
          </View>
          {rebuildSummary ? (
            <Text style={s.tileMeta}>
              Rebuild: {rebuildSummary.processed} memories, {rebuildSummary.pages_created} pages created, {rebuildSummary.claims_linked} claims linked
            </Text>
          ) : null}
        </Card>
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: NEURAL.background },
  scroll: { paddingBottom: SPACING['5xl'] },

  header: {
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING.md,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: SPACING.sm,
  },
  title: { fontSize: FONT_SIZE['2xl'], fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  subtitle: { marginTop: 4, fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },

  refreshButton: { padding: SPACING.xs },
  refreshIcon: { marginTop: 1 },

  card: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md, gap: SPACING.sm },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: SPACING.sm },
  cardTitle: { fontSize: FONT_SIZE.base, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  input: { marginBottom: 2 },
  rowBtns: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.sm },

  badgeRowWrap: {
    marginHorizontal: SPACING.lg,
    marginBottom: SPACING.md,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.xs,
  },

  pageList: { gap: SPACING.sm },
  pageRow: {
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}70`,
    borderRadius: RADIUS.md,
    backgroundColor: NEURAL.surfaceContainerLow,
    padding: SPACING.sm,
    gap: SPACING.xs,
  },
  pageRowActive: {
    borderColor: `${NEURAL.primary}75`,
    backgroundColor: `${NEURAL.primary}14`,
  },
  pageRowHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: SPACING.sm },
  pageTitle: { flex: 1, fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurface },
  pagePreview: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, lineHeight: FONT_SIZE.xs * 1.5 },
  topicRow: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACING.xs },

  selectedTitle: { fontSize: FONT_SIZE.lg, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  selectedMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },
  selectedBody: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface, lineHeight: FONT_SIZE.sm * 1.6 },

  claimList: { marginTop: SPACING.xs, gap: SPACING.sm },
  claimRow: {
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
    paddingTop: SPACING.sm,
    gap: 4,
  },
  claimHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  claimConfidence: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },
  claimText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurface },

  maintenanceGrid: { flexDirection: 'row', gap: SPACING.sm, flexWrap: 'wrap' },
  maintenanceTile: {
    flexGrow: 1,
    width: '47%',
    borderRadius: RADIUS.md,
    backgroundColor: NEURAL.surfaceContainerLow,
    borderWidth: 1,
    borderColor: `${NEURAL.outlineVariant}70`,
    padding: SPACING.sm,
    gap: 2,
  },
  tileLabel: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant, textTransform: 'uppercase', letterSpacing: 0.4 },
  tileValue: { fontSize: FONT_SIZE.base, color: NEURAL.onSurface, fontWeight: FONT_WEIGHT.semibold },
  tileMeta: { fontSize: FONT_SIZE.xs, color: NEURAL.onSurfaceVariant },

  errorCard: { marginHorizontal: SPACING.lg, marginBottom: SPACING.md },
  errorText: { fontSize: FONT_SIZE.sm, color: NEURAL.error },
  empty: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant },
});
