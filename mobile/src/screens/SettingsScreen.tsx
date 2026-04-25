/**
 * SettingsScreen — Cortex Aurora Full-screen settings page
 * Light theme with clean form elements and indigo accents
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  KeyboardAvoidingView,
  Platform,
  Linking,
} from 'react-native';
import Slider from '@react-native-community/slider';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { AppIcon } from '../components/ui/AppIcon';
import { ModelDownloadManager } from '../components/modelpacks/ModelDownloadManager';
import { ModelRecommendationCard } from '../components/modelpacks/ModelRecommendationCard';
import { OfflineReadinessBadge } from '../components/modelpacks/OfflineReadinessBadge';

import type {
  ChatSettings,
  LLMProviderType,
  ModelpackEntry,
  ModelpackInstallState,
  ModelpackManifest,
} from '../../shared/core/types';

interface SettingsScreenProps {
  settings: ChatSettings;
  onUpdateSettings: (s: Partial<ChatSettings>) => void;
  onSelectLLMProvider: (provider: LLMProviderType) => void;
  onBack: () => void;
  onReconnect: () => void;
  reconnecting: boolean;
  connectionStatus: string;
  backendUrlLabel: string;
  localModelAvailable: boolean;
  modelpackManifest: ModelpackManifest | null;
  modelpackInstalls: Record<string, ModelpackInstallState>;
  modelpackCapabilityMessage: string;
  modelpackError: string;
  onRefreshModelpacks: () => void;
  onInstallModelpack: (pack: ModelpackEntry) => void;
}

const MODELPACK_DOCS_FALLBACK =
  'https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs%2Fapi%2Fkotlin%2Fgetting_started.md';

const FALLBACK_MODELPACKS: ModelpackEntry[] = [
  {
    id: 'gemma-4-e4b-it-litert-lm',
    display_name: 'Gemma 4 E4B IT (LiteRT-LM)',
    version: '2026.04.0',
    target: 'android-web',
    family: 'gemma-4',
    quantization: 'E4B',
    summary: 'Higher-quality Gemma 4 local model for capable devices.',
    availability: 'available',
    download_url: 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm',
    cta_label: 'Download to device',
    files: [
      { path: 'gemma-4-E4B-it.litertlm', size_bytes: 0, sha256: '' },
      { path: 'gemma-4-E4B-it-web.task', size_bytes: 0, sha256: '' },
    ],
  },
  {
    id: 'gemma-4-e2b-it-litert-lm',
    display_name: 'Gemma 4 E2B IT (LiteRT-LM)',
    version: '2026.04.0',
    target: 'android-web',
    family: 'gemma-4',
    quantization: 'E2B',
    summary: 'Lean Gemma 4 local model for faster installs and mid-range devices.',
    availability: 'available',
    download_url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm',
    cta_label: 'Download to device',
    files: [
      { path: 'gemma-4-E2B-it.litertlm', size_bytes: 0, sha256: '' },
      { path: 'gemma-4-E2B-it-web.task', size_bytes: 0, sha256: '' },
    ],
  },
  {
    id: 'gemma-3-5-ft-local',
    display_name: 'Gemma 3.5 Fine-Tuned (Planned)',
    version: 'planned',
    target: 'android-web',
    family: 'gemma-3.5',
    quantization: 'tbd',
    summary: 'Reserved slot for your upcoming fine-tuned local model integration.',
    availability: 'coming_soon',
    cta_label: 'Coming Soon',
    files: [],
  },
];

function normalizeModelpackManifest(input: ModelpackManifest | null): ModelpackManifest {
  const raw = input && typeof input === 'object' ? (input as unknown as Record<string, unknown>) : {};
  const rawPacks = Array.isArray(raw.packs) ? raw.packs : [];

  const packs: ModelpackEntry[] = rawPacks
    .filter((pack): pack is Record<string, unknown> => Boolean(pack && typeof pack === 'object'))
    .map((pack, idx) => {
      const rawFiles = Array.isArray(pack.files) ? pack.files : [];
      return {
        id: typeof pack.id === 'string' && pack.id.trim() ? pack.id.trim() : `pack-${idx + 1}`,
        display_name:
          typeof pack.display_name === 'string' && pack.display_name.trim()
            ? pack.display_name.trim()
            : `Model Pack ${idx + 1}`,
        version: typeof pack.version === 'string' && pack.version.trim() ? pack.version.trim() : 'unknown',
        target: typeof pack.target === 'string' ? pack.target : undefined,
        family: typeof pack.family === 'string' ? pack.family : undefined,
        quantization: typeof pack.quantization === 'string' ? pack.quantization : undefined,
        summary: typeof pack.summary === 'string' ? pack.summary : undefined,
        availability: pack.availability === 'coming_soon' ? 'coming_soon' : 'available',
        download_url:
          typeof pack.download_url === 'string' && pack.download_url.trim()
            ? pack.download_url.trim()
            : undefined,
        cta_label: typeof pack.cta_label === 'string' ? pack.cta_label : undefined,
        docs_url: typeof pack.docs_url === 'string' ? pack.docs_url : undefined,
        requires: Array.isArray(pack.requires)
          ? pack.requires.filter((item): item is string => typeof item === 'string')
          : undefined,
        files: rawFiles
          .filter((file): file is Record<string, unknown> => Boolean(file && typeof file === 'object'))
          .map((file) => ({
            path: typeof file.path === 'string' ? file.path : '',
            size_bytes: typeof file.size_bytes === 'number' ? file.size_bytes : 0,
            sha256: typeof file.sha256 === 'string' ? file.sha256 : '',
          })),
      };
    });

  return {
    schema_version: typeof raw.schema_version === 'string' ? raw.schema_version : '1.1',
    generated_at: typeof raw.generated_at === 'string' ? raw.generated_at : new Date().toISOString(),
    signature_required: raw.signature_required !== false,
    source: typeof raw.source === 'string' ? raw.source : 'mobile-fallback',
    docs_url: typeof raw.docs_url === 'string' ? raw.docs_url : MODELPACK_DOCS_FALLBACK,
    channels: Array.isArray(raw.channels)
      ? raw.channels.filter((item): item is string => typeof item === 'string')
      : undefined,
    packs: packs.length > 0 ? packs : FALLBACK_MODELPACKS,
  };
}

export function SettingsScreen({
  settings,
  onUpdateSettings,
  onSelectLLMProvider,
  onBack,
  onReconnect,
  reconnecting,
  connectionStatus,
  backendUrlLabel,
  localModelAvailable,
  modelpackManifest,
  modelpackInstalls,
  modelpackCapabilityMessage,
  modelpackError,
  onRefreshModelpacks,
  onInstallModelpack,
}: SettingsScreenProps) {
  const [maxTokensDraft, setMaxTokensDraft] = useState(String(settings.maxTokens ?? ''));
  const [modelpackLinkError, setModelpackLinkError] = useState('');

  useEffect(() => {
    setMaxTokensDraft(String(settings.maxTokens ?? ''));
  }, [settings.maxTokens]);

  const commitMaxTokensDraft = useCallback(() => {
    const parsed = parseInt(maxTokensDraft, 10);
    if (Number.isNaN(parsed)) {
      setMaxTokensDraft(String(settings.maxTokens ?? ''));
      return;
    }

    const bounded = Math.max(128, Math.min(65536, parsed));
    if (bounded !== settings.maxTokens) {
      onUpdateSettings({ maxTokens: bounded });
    }
    if (String(bounded) !== maxTokensDraft) {
      setMaxTokensDraft(String(bounded));
    }
  }, [maxTokensDraft, onUpdateSettings, settings.maxTokens]);

  const normalizedManifest = useMemo(
    () => normalizeModelpackManifest(modelpackManifest),
    [modelpackManifest],
  );

  const downloadableNow = useMemo(
    () =>
      normalizedManifest.packs.filter(
        (pack) => pack.availability !== 'coming_soon' && (pack.files.length > 0 || Boolean(pack.download_url)),
      ).length,
    [normalizedManifest.packs],
  );
  const installedNow = useMemo(
    () => Object.values(modelpackInstalls).filter((item) => item.status === 'installed').length,
    [modelpackInstalls],
  );

  const openExternalUrl = useCallback(async (url: string) => {
    try {
      setModelpackLinkError('');
      const supported = await Linking.canOpenURL(url);
      if (!supported) {
        throw new Error('This device cannot open the requested URL.');
      }
      await Linking.openURL(url);
    } catch (error) {
      setModelpackLinkError(error instanceof Error ? error.message : 'Failed to open URL.');
    }
  }, []);

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={styles.header}>
        <TouchableOpacity style={styles.headerButton} onPress={onBack}>
          <AppIcon name="arrow-left" size={20} color={NEURAL.onSurface} />
        </TouchableOpacity>
        <Text style={styles.title}>Settings</Text>
        <View style={styles.headerButton} />
      </View>

      <ScrollView
        style={styles.body}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        contentContainerStyle={styles.contentContainer}
      >
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>LLM Provider</Text>
          <View style={styles.segmentRow}>
            {(['local', 'gemma_local', 'gemini'] as const).map((p) => (
              <TouchableOpacity
                key={p}
                onPress={() => onSelectLLMProvider(p)}
                disabled={(p === 'local' || p === 'gemma_local') && !localModelAvailable}
                style={[
                  styles.segBtn,
                  settings.llmProvider === p && styles.segBtnActive,
                  (p === 'local' || p === 'gemma_local') && !localModelAvailable && styles.segBtnDisabled,
                ]}
              >
                <Text
                  style={[
                    styles.segText,
                    settings.llmProvider === p && styles.segTextActive,
                    (p === 'local' || p === 'gemma_local') && !localModelAvailable && styles.segTextDisabled,
                  ]}
                >
                  {p === 'gemini' ? 'Gemini' : p === 'gemma_local' ? 'Gemma Local' : 'Local'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          {!localModelAvailable ? (
            <Text style={styles.hintText}>
              Local inference is unavailable on the current deployed backend, so Gemini is the safe active option.
            </Text>
          ) : null}
        </View>

        <View style={styles.section}>
          <View style={styles.modelpackHeaderRow}>
            <View style={styles.modelpackHeaderText}>
              <Text style={styles.sectionLabel}>Local Model Packs</Text>
              <Text style={styles.hintText}>
                Download LiteRT Gemma artifacts inside the app with persistent install status.
              </Text>
            </View>
            <OfflineReadinessBadge
              ready={installedNow > 0}
              details={installedNow > 0 ? `${installedNow} installed` : `${downloadableNow} downloadable now`}
            />
          </View>

          <View style={styles.modelpackList}>
            {normalizedManifest.packs.map((pack) => {
              const available = pack.availability !== 'coming_soon';
              const installState = modelpackInstalls[pack.id];

              return (
                <View key={pack.id} style={styles.modelpackBlock}>
                  <ModelRecommendationCard
                    title={pack.display_name}
                    reason={pack.summary || 'Model pack prepared for local runtime.'}
                    recommended={available}
                  />
                  <ModelDownloadManager
                    packName={pack.display_name}
                    status={installState?.status || 'not_installed'}
                    progress={installState?.progress || 0}
                    detail={installState?.error || installState?.activeFile || undefined}
                    actionLabel={available ? pack.cta_label || 'Download to device' : pack.cta_label || 'Coming Soon'}
                    actionDisabled={!available}
                    onInstall={
                      available
                        ? () => {
                            onInstallModelpack(pack);
                          }
                        : undefined
                    }
                    onRetry={
                      available
                        ? () => {
                            onInstallModelpack(pack);
                          }
                        : undefined
                    }
                  />
                </View>
              );
            })}
          </View>

          <View style={styles.actionRow}>
            <Button
              label="LiteRT Kotlin Guide"
              variant="outline"
              onPress={() => void openExternalUrl(normalizedManifest.docs_url || MODELPACK_DOCS_FALLBACK)}
              size="sm"
            />
            <Button
              label="Refresh Catalog"
              variant="outline"
              onPress={onRefreshModelpacks}
              size="sm"
            />
          </View>

          {modelpackError ? (
            <Text style={[styles.connStatus, { color: NEURAL.error }]}>
              Modelpack catalog: {modelpackError}
            </Text>
          ) : null}
          {modelpackCapabilityMessage ? (
            <Text style={styles.hintText}>{modelpackCapabilityMessage}</Text>
          ) : null}
          {modelpackLinkError ? (
            <Text style={[styles.connStatus, { color: NEURAL.error }]}>
              {modelpackLinkError}
            </Text>
          ) : null}
        </View>

        <View style={styles.section}>
          <View style={styles.sliderHeader}>
            <Text style={styles.sectionLabel}>Temperature</Text>
            <Text style={styles.sliderValue}>{settings.temperature.toFixed(1)}</Text>
          </View>
          <Slider
            minimumValue={0}
            maximumValue={2}
            step={0.1}
            value={settings.temperature}
            onValueChange={(v: number) => onUpdateSettings({ temperature: Math.round(v * 10) / 10 })}
            minimumTrackTintColor={NEURAL.primary}
            maximumTrackTintColor={NEURAL.outlineVariant}
            thumbTintColor={NEURAL.primary}
            style={styles.slider}
          />
        </View>

        <View style={styles.section}>
          <View style={styles.sliderHeader}>
            <Text style={styles.sectionLabel}>Top-P</Text>
            <Text style={styles.sliderValue}>{settings.topP.toFixed(2)}</Text>
          </View>
          <Slider
            minimumValue={0}
            maximumValue={1}
            step={0.05}
            value={settings.topP}
            onValueChange={(v: number) => onUpdateSettings({ topP: Math.round(v * 20) / 20 })}
            minimumTrackTintColor={NEURAL.secondary}
            maximumTrackTintColor={NEURAL.outlineVariant}
            thumbTintColor={NEURAL.secondary}
            style={styles.slider}
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Max Tokens</Text>
          <TextInput
            placeholder="e.g. 4096"
            value={maxTokensDraft}
            onChangeText={(t) => setMaxTokensDraft(t.replace(/[^0-9]/g, ''))}
            onBlur={commitMaxTokensDraft}
            onSubmitEditing={commitMaxTokensDraft}
            keyboardType={Platform.OS === 'ios' ? 'number-pad' : 'numeric'}
            autoCorrect={false}
            autoCapitalize="none"
            returnKeyType="done"
          />
          <Text style={styles.hintText}>Allowed range: 128 to 65536</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Features</Text>
          {([
            { key: 'useRAG', label: 'RAG (Retrieval-Augmented Gen)', color: NEURAL.tertiary },
            { key: 'stream', label: 'Streaming Responses', color: NEURAL.primary },
          ] as const).map((t) => (
            <View key={t.key} style={styles.toggleRow}>
              <Text style={styles.toggleLabel}>{t.label}</Text>
              <Switch
                value={Boolean(settings[t.key as keyof ChatSettings])}
                onValueChange={(v) => onUpdateSettings({ [t.key]: v })}
                trackColor={{ false: NEURAL.outlineVariant, true: `${t.color}80` }}
                thumbColor={Boolean(settings[t.key as keyof ChatSettings]) ? t.color : NEURAL.onSurfaceVariant}
              />
            </View>
          ))}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionLabel}>Backend Connection</Text>
          <View style={styles.connectionCard}>
            <View style={styles.connectionHeader}>
              <View style={styles.connectionIconWrap}>
                <AppIcon name="cloud-check-outline" size={16} color={NEURAL.primary} />
              </View>
              <View style={styles.connectionMeta}>
                <Text style={styles.connectionTitle}>Automatic backend discovery</Text>
                <Text style={styles.connectionBody}>
                  Cortex Lab now connects to the deployed backend automatically on launch.
                </Text>
              </View>
            </View>
            <View style={styles.connectionEndpointRow}>
              <Text style={styles.connectionEndpointLabel}>Live endpoint</Text>
              <Text style={styles.connectionEndpointValue}>{backendUrlLabel}</Text>
            </View>
          </View>
          <View style={styles.actionRow}>
            <Button
              label={reconnecting ? 'Reconnecting…' : 'Re-check Backend'}
              variant="outline"
              onPress={onReconnect}
              disabled={reconnecting}
              loading={reconnecting}
              size="sm"
            />
          </View>
          {connectionStatus ? (
            <Text
              style={[
                styles.connStatus,
                { color: connectionStatus.startsWith('Connected') ? NEURAL.tertiary : NEURAL.onSurfaceVariant },
              ]}
            >
              {connectionStatus}
            </Text>
          ) : null}
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.md,
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  headerButton: {
    width: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: FONT_SIZE.xl,
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurface,
  },
  body: {
    flex: 1,
  },
  contentContainer: {
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.lg,
    paddingBottom: SPACING['5xl'],
  },
  section: { marginBottom: SPACING.xl },
  sectionLabel: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurfaceVariant,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: SPACING.sm,
  },
  hintText: {
    marginTop: SPACING.xs,
    fontSize: FONT_SIZE.xs,
    color: NEURAL.onSurfaceVariant,
  },
  modelpackHeaderRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: SPACING.sm,
  },
  modelpackHeaderText: {
    flex: 1,
  },
  modelpackList: {
    marginTop: SPACING.sm,
    gap: SPACING.md,
  },
  modelpackBlock: {
    gap: SPACING.sm,
  },
  segmentRow: { flexDirection: 'row', gap: SPACING.sm },
  segBtn: {
    flex: 1,
    paddingVertical: SPACING.sm + 2,
    borderRadius: RADIUS.lg,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    alignItems: 'center',
  },
  segBtnActive: { backgroundColor: `${NEURAL.primary}26`, borderColor: `${NEURAL.primary}60` },
  segBtnDisabled: { backgroundColor: `${NEURAL.outlineVariant}30`, borderColor: `${NEURAL.outlineVariant}60` },
  segText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, fontWeight: FONT_WEIGHT.medium },
  segTextActive: { color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },
  segTextDisabled: { color: `${NEURAL.onSurfaceVariant}99` },
  sliderHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: SPACING.sm,
  },
  sliderValue: { fontSize: FONT_SIZE.base, color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },
  slider: { height: 36 },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: `${NEURAL.outlineVariant}40`,
  },
  toggleLabel: { flex: 1, fontSize: FONT_SIZE.sm, color: NEURAL.onSurface },
  actionRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    marginTop: SPACING.md,
  },
  connectionCard: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#e2e8f0',
    borderRadius: RADIUS.xl,
    padding: SPACING.lg,
    gap: SPACING.md,
  },
  connectionHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: SPACING.md,
  },
  connectionIconWrap: {
    width: 36,
    height: 36,
    borderRadius: RADIUS.lg,
    backgroundColor: '#eef2ff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  connectionMeta: {
    flex: 1,
    gap: 2,
  },
  connectionTitle: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurface,
  },
  connectionBody: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurfaceVariant,
    lineHeight: 18,
  },
  connectionEndpointRow: {
    gap: 4,
    paddingTop: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: '#f1f5f9',
  },
  connectionEndpointLabel: {
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurfaceVariant,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  connectionEndpointValue: {
    fontSize: FONT_SIZE.sm,
    color: NEURAL.onSurface,
    fontWeight: FONT_WEIGHT.medium,
  },
  connStatus: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    marginTop: SPACING.sm,
    textAlign: 'center',
  },
});
