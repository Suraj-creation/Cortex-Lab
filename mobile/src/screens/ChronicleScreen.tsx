/**
 * ChronicleScreen — Life Chronicle passive capture and moments timeline.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  TextInput,
  FlatList,
  RefreshControl,
} from 'react-native';
import { SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { AppIcon } from '../components/ui/AppIcon';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Button } from '../components/ui/Button';
import { EmptyState } from '../components/ui/EmptyState';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';

type ChronicleTab = 'capture' | 'moments' | 'status';

interface ChronicleScreenProps {
  api: any;
}

function parseCsv(input: string): string[] {
  return input
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

function compactText(value: unknown, maxLen: number = 220): string {
  if (typeof value === 'string') {
    return value.length > maxLen ? `${value.slice(0, maxLen)}...` : value;
  }
  try {
    const text = JSON.stringify(value);
    return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
  } catch {
    return 'No details';
  }
}

export function ChronicleScreen({ api }: ChronicleScreenProps) {
  const [activeTab, setActiveTab] = useState<ChronicleTab>('capture');

  const [status, setStatus] = useState<Record<string, any> | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);

  const [moments, setMoments] = useState<any[]>([]);
  const [momentsLoading, setMomentsLoading] = useState(false);
  const [selectedMoment, setSelectedMoment] = useState<any | null>(null);

  const [busyAction, setBusyAction] = useState(false);

  const [noteDraft, setNoteDraft] = useState('');
  const [locationDraft, setLocationDraft] = useState('');
  const [peopleDraft, setPeopleDraft] = useState('');
  const [tagsDraft, setTagsDraft] = useState('');
  const [emotionDraft, setEmotionDraft] = useState('');

  const [titleDraft, setTitleDraft] = useState('');
  const [windowSecondsDraft, setWindowSecondsDraft] = useState('180');
  const [domainDraft, setDomainDraft] = useState('everyday');
  const [retrievalHintDraft, setRetrievalHintDraft] = useState('');

  const loadStatus = useCallback(async () => {
    setStatusLoading(true);
    try {
      const res = await api.getChroniclePassiveStatus?.();
      setStatus((res || null) as Record<string, any> | null);
    } catch {
      setStatus(null);
    }
    setStatusLoading(false);
  }, [api]);

  const loadMoments = useCallback(async () => {
    setMomentsLoading(true);
    try {
      const res = await api.listChronicleMoments?.();
      setMoments(Array.isArray(res?.moments) ? res.moments : []);
    } catch {
      setMoments([]);
    }
    setMomentsLoading(false);
  }, [api]);

  useEffect(() => {
    void loadStatus();
    void loadMoments();
  }, [loadStatus, loadMoments]);

  const passiveEnabled = Boolean(status?.passive_mode_enabled);

  const enablePassive = useCallback(async () => {
    setBusyAction(true);
    try {
      await api.enableChroniclePassive?.(true, 'mobile-user');
      await loadStatus();
    } catch {}
    setBusyAction(false);
  }, [api, loadStatus]);

  const disablePassive = useCallback(async () => {
    setBusyAction(true);
    try {
      await api.disableChroniclePassive?.();
      await loadStatus();
    } catch {}
    setBusyAction(false);
  }, [api, loadStatus]);

  const addObservation = useCallback(async () => {
    if (!noteDraft.trim()) {
      return;
    }

    setBusyAction(true);
    try {
      await api.addChronicleObservation?.({
        note: noteDraft.trim(),
        location: locationDraft.trim() ? { name: locationDraft.trim() } : {},
        peoplePresent: parseCsv(peopleDraft),
        tags: parseCsv(tagsDraft),
        emotionHint: emotionDraft.trim(),
        source: 'mobile_manual',
      });
      setNoteDraft('');
      await loadStatus();
    } catch {}
    setBusyAction(false);
  }, [api, noteDraft, locationDraft, peopleDraft, tagsDraft, emotionDraft, loadStatus]);

  const saveWindow = useCallback(async () => {
    const seconds = Math.max(10, Math.min(1800, parseInt(windowSecondsDraft || '180', 10) || 180));

    setBusyAction(true);
    try {
      await api.saveChronicleWindow?.({
        title: titleDraft,
        windowSeconds: seconds,
        lifeDomain: domainDraft.trim() || 'everyday',
        retrievalHint: retrievalHintDraft.trim(),
      });
      await Promise.all([loadStatus(), loadMoments()]);
      setTitleDraft('');
      setRetrievalHintDraft('');
    } catch {}
    setBusyAction(false);
  }, [api, domainDraft, loadMoments, loadStatus, retrievalHintDraft, titleDraft, windowSecondsDraft]);

  const openMoment = useCallback(async (memoryId: string) => {
    try {
      const detail = await api.getChronicleMoment?.(memoryId);
      setSelectedMoment(detail || null);
    } catch {
      setSelectedMoment(null);
    }
  }, [api]);

  const tabs: { key: ChronicleTab; label: string; icon: string }[] = [
    { key: 'capture', label: 'Capture', icon: 'camera-wireless-outline' },
    { key: 'moments', label: 'Moments', icon: 'timeline-text-outline' },
    { key: 'status', label: 'Status', icon: 'heart-pulse' },
  ];

  const sortedMoments = useMemo(
    () => [...moments].sort((a, b) => String(b.timestamp || '').localeCompare(String(a.timestamp || ''))),
    [moments],
  );

  return (
    <View style={styles.container}>
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

      {activeTab === 'capture' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          <SectionHeader
            title="Passive Capture"
            subtitle="Record lightweight observations and save windows as moments"
            action={{ label: 'Refresh', onPress: () => void loadStatus() }}
          />

          <View style={styles.metricsGrid}>
            <MetricCard
              label="Mode"
              value={passiveEnabled ? 'Enabled' : 'Disabled'}
              tone={passiveEnabled ? 'emerald' : 'rose'}
              compact
              style={styles.metricHalf}
            />
            <MetricCard
              label="Buffer"
              value={String(status?.buffer_entries ?? 0)}
              tone="indigo"
              compact
              style={styles.metricHalf}
            />
          </View>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Passive Mode" icon={<AppIcon name="microphone-message" size={16} color="#6366f1" />} />
            <View style={styles.rowActions}>
              <Button
                label="Enable"
                variant="success"
                size="sm"
                onPress={() => void enablePassive()}
                loading={busyAction}
                disabled={passiveEnabled}
              />
              <Button
                label="Disable"
                variant="error"
                size="sm"
                onPress={() => void disablePassive()}
                loading={busyAction}
                disabled={!passiveEnabled}
              />
            </View>
            <Text style={styles.metaText}>
              Consent actor: {String(status?.consent_actor || 'n/a')} • Since: {String(status?.consent_granted_at || 'n/a')}
            </Text>
          </Card>

          <Card variant="default" padding="lg">
            <SectionHeader title="Add Observation" icon={<AppIcon name="notebook-plus-outline" size={16} color="#6366f1" />} />

            <Text style={styles.inputLabel}>What happened?</Text>
            <TextInput
              value={noteDraft}
              onChangeText={setNoteDraft}
              placeholder="Describe a moment, activity, thought, or interaction"
              placeholderTextColor="#94a3b8"
              multiline
              style={[styles.textInput, styles.multiline]}
            />

            <View style={styles.inputRow}>
              <View style={styles.inputCol}>
                <Text style={styles.inputLabel}>Location</Text>
                <TextInput
                  value={locationDraft}
                  onChangeText={setLocationDraft}
                  placeholder="Home, Office, Cafe"
                  placeholderTextColor="#94a3b8"
                  style={styles.textInput}
                />
              </View>
              <View style={styles.inputCol}>
                <Text style={styles.inputLabel}>Emotion Hint</Text>
                <TextInput
                  value={emotionDraft}
                  onChangeText={setEmotionDraft}
                  placeholder="calm, excited, focused"
                  placeholderTextColor="#94a3b8"
                  style={styles.textInput}
                />
              </View>
            </View>

            <Text style={styles.inputLabel}>People (comma separated)</Text>
            <TextInput
              value={peopleDraft}
              onChangeText={setPeopleDraft}
              placeholder="Alex, Priya"
              placeholderTextColor="#94a3b8"
              style={styles.textInput}
            />

            <Text style={styles.inputLabel}>Tags (comma separated)</Text>
            <TextInput
              value={tagsDraft}
              onChangeText={setTagsDraft}
              placeholder="work, planning, milestone"
              placeholderTextColor="#94a3b8"
              style={styles.textInput}
            />

            <Button
              label="Buffer Observation"
              onPress={() => void addObservation()}
              loading={busyAction}
              disabled={!passiveEnabled || !noteDraft.trim()}
              style={styles.submitBtn}
            />
          </Card>

          <Card variant="outlined" padding="lg">
            <SectionHeader title="Save Recent Window" icon={<AppIcon name="content-save-outline" size={16} color="#8b5cf6" />} />

            <Text style={styles.inputLabel}>Title (optional)</Text>
            <TextInput
              value={titleDraft}
              onChangeText={setTitleDraft}
              placeholder="Evening planning session"
              placeholderTextColor="#94a3b8"
              style={styles.textInput}
            />

            <View style={styles.inputRow}>
              <View style={styles.inputCol}>
                <Text style={styles.inputLabel}>Window Seconds</Text>
                <TextInput
                  value={windowSecondsDraft}
                  onChangeText={(value) => setWindowSecondsDraft(value.replace(/[^0-9]/g, ''))}
                  keyboardType="numeric"
                  style={styles.textInput}
                />
              </View>
              <View style={styles.inputCol}>
                <Text style={styles.inputLabel}>Life Domain</Text>
                <TextInput
                  value={domainDraft}
                  onChangeText={setDomainDraft}
                  placeholder="everyday"
                  placeholderTextColor="#94a3b8"
                  style={styles.textInput}
                />
              </View>
            </View>

            <Text style={styles.inputLabel}>Retrieval Hint</Text>
            <TextInput
              value={retrievalHintDraft}
              onChangeText={setRetrievalHintDraft}
              placeholder="Use when recalling planning context"
              placeholderTextColor="#94a3b8"
              style={styles.textInput}
            />

            <Button
              label="Save Window as Moment"
              variant="secondary"
              onPress={() => void saveWindow()}
              loading={busyAction}
              disabled={!passiveEnabled}
              style={styles.submitBtn}
            />
          </Card>
        </ScrollView>
      )}

      {activeTab === 'moments' && (
        <FlatList
          data={sortedMoments}
          keyExtractor={(item, idx) => String(item.memory_id || item.id || idx)}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={momentsLoading}
              onRefresh={() => void loadMoments()}
              tintColor="#6366f1"
              colors={['#6366f1']}
            />
          }
          ListHeaderComponent={
            <SectionHeader title="Captured Moments" subtitle={`${sortedMoments.length} moments`} action={{ label: 'Refresh', onPress: () => void loadMoments() }} />
          }
          renderItem={({ item }) => (
            <TouchableOpacity onPress={() => void openMoment(String(item.memory_id || item.id || ''))} activeOpacity={0.7}>
              <Card variant="default" padding="md" style={styles.momentCard}>
                <View style={styles.momentTop}>
                  <Text style={styles.momentTitle}>{String(item.title || item.memory_id || 'Moment')}</Text>
                  <Badge label={String(item.life_domain || 'everyday')} variant="primary" size="sm" />
                </View>
                <Text style={styles.momentNarrative} numberOfLines={2}>
                  {String(item.narrative || item.summary || 'No narrative available')}
                </Text>
                <View style={styles.momentMetaRow}>
                  <Text style={styles.momentMeta}>People: {Array.isArray(item.people_present) ? item.people_present.length : 0}</Text>
                  <Text style={styles.momentMeta}>Tags: {Array.isArray(item.tags) ? item.tags.length : 0}</Text>
                  <Text style={styles.momentMeta}>{String(item.timestamp || '')}</Text>
                </View>
              </Card>
            </TouchableOpacity>
          )}
          ListEmptyComponent={
            momentsLoading ? (
              <LoadingSpinner message="Loading moments..." />
            ) : (
              <EmptyState
                icon="timeline-text-outline"
                title="No Moments Yet"
                message="Buffer observations and save a window to create your first chronicle moment."
              />
            )
          }
          ListFooterComponent={
            selectedMoment ? (
              <Card variant="outlined" padding="lg" style={styles.selectedMomentCard}>
                <SectionHeader title="Selected Moment" />
                <Text style={styles.selectedMomentBody}>{compactText(selectedMoment, 1200)}</Text>
              </Card>
            ) : null
          }
        />
      )}

      {activeTab === 'status' && (
        <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
          {statusLoading && !status ? (
            <LoadingSpinner message="Loading status..." />
          ) : !status ? (
            <EmptyState icon="heart-pulse" title="No Status" message="Chronicle status is unavailable." />
          ) : (
            <>
              <SectionHeader title="Chronicle Service Status" action={{ label: 'Refresh', onPress: () => void loadStatus() }} />

              <View style={styles.metricsGrid}>
                <MetricCard
                  label="Passive Mode"
                  value={passiveEnabled ? 'Enabled' : 'Disabled'}
                  tone={passiveEnabled ? 'emerald' : 'rose'}
                  compact
                  style={styles.metricHalf}
                />
                <MetricCard
                  label="Saved Moments"
                  value={String(status.saved_moments ?? 0)}
                  tone="violet"
                  compact
                  style={styles.metricHalf}
                />
              </View>

              <Card variant="outlined" padding="lg">
                <SectionHeader title="Buffer Health" />
                <View style={styles.kvRow}>
                  <Text style={styles.kvKey}>Entries</Text>
                  <Text style={styles.kvValue}>{String(status.buffer_entries ?? 0)}</Text>
                </View>
                <View style={styles.kvRow}>
                  <Text style={styles.kvKey}>Window</Text>
                  <Text style={styles.kvValue}>{String(status.buffer_seconds ?? 0)}s</Text>
                </View>
                <View style={styles.kvRow}>
                  <Text style={styles.kvKey}>Oldest</Text>
                  <Text style={styles.kvValue}>{String(status.buffer_oldest || 'n/a')}</Text>
                </View>
                <View style={styles.kvRow}>
                  <Text style={styles.kvKey}>Newest</Text>
                  <Text style={styles.kvValue}>{String(status.buffer_newest || 'n/a')}</Text>
                </View>
                <View style={styles.kvRow}>
                  <Text style={styles.kvKey}>Updated</Text>
                  <Text style={styles.kvValue}>{String(status.updated_at || 'n/a')}</Text>
                </View>
              </Card>
            </>
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f8fafc' },
  scrollView: { flex: 1 },
  scrollContent: { padding: SPACING.lg, paddingBottom: SPACING['5xl'], gap: SPACING.md },

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
    backgroundColor: '#f8fafc',
  },
  tabActive: { backgroundColor: '#eef2ff' },
  tabLabel: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.medium, color: '#94a3b8' },
  tabLabelActive: { color: '#6366f1', fontWeight: FONT_WEIGHT.semibold },

  metricsGrid: { flexDirection: 'row', gap: SPACING.sm },
  metricHalf: { flex: 1 },

  rowActions: {
    flexDirection: 'row',
    gap: SPACING.sm,
    marginBottom: SPACING.sm,
  },
  metaText: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    lineHeight: 16,
  },

  inputLabel: {
    fontSize: FONT_SIZE.xs,
    color: '#64748b',
    fontWeight: FONT_WEIGHT.semibold,
    marginTop: SPACING.sm,
    marginBottom: 4,
  },
  textInput: {
    backgroundColor: '#f8fafc',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: RADIUS.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    fontSize: FONT_SIZE.sm,
    color: '#0f172a',
  },
  multiline: {
    minHeight: 86,
    textAlignVertical: 'top',
  },
  inputRow: { flexDirection: 'row', gap: SPACING.sm },
  inputCol: { flex: 1 },
  submitBtn: { marginTop: SPACING.md, alignSelf: 'flex-start' },

  momentCard: { marginBottom: 0 },
  momentTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: SPACING.xs,
    gap: SPACING.sm,
  },
  momentTitle: {
    flex: 1,
    fontSize: FONT_SIZE.md,
    fontWeight: FONT_WEIGHT.semibold,
    color: '#0f172a',
  },
  momentNarrative: {
    fontSize: FONT_SIZE.sm,
    color: '#475569',
    lineHeight: 18,
    marginBottom: SPACING.sm,
  },
  momentMetaRow: {
    flexDirection: 'row',
    gap: SPACING.md,
    flexWrap: 'wrap',
  },
  momentMeta: {
    fontSize: 10,
    color: '#94a3b8',
  },
  selectedMomentCard: { marginTop: SPACING.md },
  selectedMomentBody: {
    fontSize: FONT_SIZE.xs,
    color: '#334155',
    lineHeight: 17,
  },

  kvRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: '#f8fafc',
  },
  kvKey: {
    fontSize: FONT_SIZE.sm,
    color: '#64748b',
  },
  kvValue: {
    flex: 1,
    marginLeft: SPACING.sm,
    textAlign: 'right',
    fontSize: FONT_SIZE.sm,
    color: '#0f172a',
    fontWeight: FONT_WEIGHT.semibold,
  },
});
