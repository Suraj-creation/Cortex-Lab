/**
 * SettingsSheet — Neural Dark bottom-sheet settings modal
 * Stitch ref: 7fe6d921c4054598bb1d3afad08837f5
 */
import React, { useRef, useEffect } from 'react';
import {
  View,
  Text,
  Modal,
  Animated,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Switch,
  Pressable,
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
import type { ChatSettings, ModelpackEntry, ModelpackManifest } from '../../shared/core/types';

interface SettingsSheetProps {
  visible: boolean;
  onClose: () => void;
  settings: ChatSettings;
  onUpdateSettings: (s: Partial<ChatSettings>) => void;
  onSave: (backendUrl: string) => void;
  onTestConnection: (backendUrl: string) => void;
  testingConnection: boolean;
  connectionStatus: string;
  backendUrl: string;
  modelpackManifest?: ModelpackManifest | null;
  modelpackError?: string;
  onRefreshModelpacks?: () => void;
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
    cta_label: 'Download from Hugging Face',
    files: [],
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
    cta_label: 'Download from Hugging Face',
    files: [],
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

function normalizeModelpackManifest(input: ModelpackManifest | null | undefined): ModelpackManifest {
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
    source: typeof raw.source === 'string' ? raw.source : 'mobile-sheet-fallback',
    docs_url: typeof raw.docs_url === 'string' ? raw.docs_url : MODELPACK_DOCS_FALLBACK,
    channels: Array.isArray(raw.channels)
      ? raw.channels.filter((item): item is string => typeof item === 'string')
      : undefined,
    packs: packs.length > 0 ? packs : FALLBACK_MODELPACKS,
  };
}

export function SettingsSheet({
  visible,
  onClose,
  settings,
  onUpdateSettings,
  onSave,
  onTestConnection,
  testingConnection,
  connectionStatus,
  backendUrl,
  modelpackManifest = null,
  modelpackError = '',
  onRefreshModelpacks,
}: SettingsSheetProps) {
  const [backendDraft, setBackendDraft] = React.useState(backendUrl);
  const [modelpackLinkError, setModelpackLinkError] = React.useState('');

  const normalizedManifest = React.useMemo(
    () => normalizeModelpackManifest(modelpackManifest),
    [modelpackManifest],
  );

  const downloadableNow = React.useMemo(
    () =>
      normalizedManifest.packs.filter(
        (pack) => pack.availability !== 'coming_soon' && Boolean(pack.download_url),
      ).length,
    [normalizedManifest.packs],
  );

  const openExternalUrl = React.useCallback(async (url: string) => {
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

  useEffect(() => {
    if (visible) {
      setBackendDraft(backendUrl);
      setModelpackLinkError('');
    }
  }, [visible, backendUrl]);
  const slideAnim = useRef(new Animated.Value(600)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.timing(slideAnim,   { toValue: 0,   duration: 280, useNativeDriver: true }),
        Animated.timing(opacityAnim, { toValue: 1,   duration: 280, useNativeDriver: true }),
      ]).start();
    } else {
      Animated.parallel([
        Animated.timing(slideAnim,   { toValue: 600, duration: 220, useNativeDriver: true }),
        Animated.timing(opacityAnim, { toValue: 0,   duration: 220, useNativeDriver: true }),
      ]).start();
    }
  }, [visible]);

  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={onClose} statusBarTranslucent>
      <View style={styles.modalRoot}>
        {/* Backdrop */}
        <Animated.View style={[styles.backdrop, { opacity: opacityAnim }]}> 
          <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
        </Animated.View>

        {/* Sheet */}
        <KeyboardAvoidingView
          style={styles.keyboardLayer}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          pointerEvents="box-none"
        >
          <Animated.View
            style={[styles.sheet, { transform: [{ translateY: slideAnim }] }]}
          >
            {/* Handle */}
            <View style={styles.handle} />

            <View style={styles.sheetHeader}>
              <Text style={styles.sheetTitle}>Settings</Text>
              <TouchableOpacity onPress={onClose}>
                <AppIcon name="close" size={18} color={NEURAL.onSurfaceVariant} style={styles.closeIcon} />
              </TouchableOpacity>
            </View>

            <ScrollView
              style={styles.body}
              showsVerticalScrollIndicator={false}
              keyboardShouldPersistTaps="always"
              keyboardDismissMode="none"
              contentContainerStyle={{ paddingBottom: SPACING['5xl'] }}
            >
          {/* LLM Provider */}
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>LLM Provider</Text>
            <View style={styles.segmentRow}>
              {(['local', 'gemma_local', 'gemini'] as const).map((p) => (
                <TouchableOpacity
                  key={p}
                  onPress={() => onUpdateSettings({ llmProvider: p })}
                  style={[styles.segBtn, settings.llmProvider === p && styles.segBtnActive]}
                >
                  <Text style={[styles.segText, settings.llmProvider === p && styles.segTextActive]}>
                    {p === 'gemini' ? 'Gemini' : p === 'gemma_local' ? 'Gemma Local' : 'Local'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Local model packs */}
          <View style={styles.section}>
            <View style={styles.modelpackHeaderRow}>
              <View style={styles.modelpackHeaderText}>
                <Text style={styles.sectionLabel}>Local Model Packs</Text>
                <Text style={styles.hintText}>
                  Direct links for LiteRT Gemma downloads used by Gemma Local mode.
                </Text>
              </View>
              <OfflineReadinessBadge ready={false} details={`${downloadableNow} downloadable now`} />
            </View>

            <View style={styles.modelpackList}>
              {normalizedManifest.packs.map((pack) => {
                const available = pack.availability !== 'coming_soon';
                const downloadUrl = pack.download_url;

                return (
                  <View key={pack.id} style={styles.modelpackBlock}>
                    <ModelRecommendationCard
                      title={pack.display_name}
                      reason={pack.summary || 'Model pack prepared for local runtime.'}
                      recommended={available}
                    />
                    <ModelDownloadManager
                      packName={pack.display_name}
                      status="not_installed"
                      actionLabel={available ? pack.cta_label || 'Download' : pack.cta_label || 'Coming Soon'}
                      actionDisabled={!available || !downloadUrl}
                      onInstall={
                        available && downloadUrl
                          ? () => {
                              void openExternalUrl(downloadUrl);
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
              {onRefreshModelpacks ? (
                <Button
                  label="Refresh Catalog"
                  variant="outline"
                  onPress={onRefreshModelpacks}
                  size="sm"
                />
              ) : null}
            </View>

            {modelpackError ? (
              <Text style={[styles.connStatus, { color: NEURAL.error }]}>
                Modelpack catalog: {modelpackError}
              </Text>
            ) : null}
            {modelpackLinkError ? (
              <Text style={[styles.connStatus, { color: NEURAL.error }]}>
                {modelpackLinkError}
              </Text>
            ) : null}
          </View>

          {/* Temperature */}
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

          {/* Top-P */}
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

          {/* Max Tokens (fixed for stable long-form responses) */}
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>Max Tokens</Text>
            <Text style={styles.fixedValue}>4096 (fixed)</Text>
          </View>

          {/* Toggles */}
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>Features</Text>
            {([
              { key: 'useRAG',  label: 'RAG (Retrieval-Augmented Gen)', color: NEURAL.tertiary },
              { key: 'stream',  label: 'Streaming Responses',            color: NEURAL.primary },
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

          {/* Backend URL */}
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>Backend URL</Text>
            <TextInput
              placeholder="http://192.168.1.x:8000"
              value={backendDraft}
              onChangeText={setBackendDraft}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              blurOnSubmit={false}
            />
          </View>

          {/* Connection test */}
          <View style={styles.section}>
            <Button
              label={testingConnection ? 'Testing…' : 'Test Connection'}
              variant="outline"
              onPress={() => onTestConnection(backendDraft)}
              disabled={testingConnection}
              loading={testingConnection}
              fullWidth
            />
            {connectionStatus ? (
              <Text style={[
                styles.connStatus,
                { color: connectionStatus.startsWith('Connected') ? NEURAL.tertiary : NEURAL.error }
              ]}>
                {connectionStatus}
              </Text>
            ) : null}
          </View>

          {/* Save */}
          <View style={styles.section}>
            <Button
              label="Save Settings"
              onPress={() => {
                onSave(backendDraft);
                onClose();
              }}
              fullWidth
            />
          </View>
            </ScrollView>
          </Animated.View>
        </KeyboardAvoidingView>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  modalRoot: {
    ...StyleSheet.absoluteFillObject,
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(6,14,32,0.8)',
    zIndex: 1,
  },
  keyboardLayer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
    zIndex: 2,
  },
  sheet: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: NEURAL.surfaceContainerLow,
    borderTopLeftRadius: RADIUS['3xl'],
    borderTopRightRadius: RADIUS['3xl'],
    maxHeight: '90%',
    shadowColor: NEURAL.primary,
    shadowOpacity: 0.15,
    shadowRadius: 20,
    elevation: 25,
  },
  handle: {
    width: 40,
    height: 4,
    backgroundColor: NEURAL.outlineVariant,
    borderRadius: 2,
    alignSelf: 'center',
    marginTop: SPACING.md,
    marginBottom: SPACING.sm,
  },
  sheetHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING.md,
  },
  sheetTitle: { fontSize: FONT_SIZE.xl, fontWeight: FONT_WEIGHT.bold, color: NEURAL.onSurface },
  closeIcon: { padding: SPACING.sm },
  body: { paddingHorizontal: SPACING.lg },

  section: { marginBottom: SPACING.xl },
  sectionLabel: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.semibold, color: NEURAL.onSurfaceVariant, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: SPACING.sm },

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
  segText: { fontSize: FONT_SIZE.sm, color: NEURAL.onSurfaceVariant, fontWeight: FONT_WEIGHT.medium },
  segTextActive: { color: NEURAL.primary, fontWeight: FONT_WEIGHT.bold },

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
    gap: SPACING.md,
  },
  modelpackBlock: {
    gap: SPACING.sm,
  },
  hintText: {
    fontSize: FONT_SIZE.xs,
    color: NEURAL.onSurfaceVariant,
    marginBottom: SPACING.sm,
  },
  actionRow: {
    flexDirection: 'row',
    gap: SPACING.sm,
    marginTop: SPACING.md,
  },

  sliderHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: SPACING.sm },
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

  fixedValue: {
    fontSize: FONT_SIZE.base,
    color: NEURAL.onSurface,
    fontWeight: FONT_WEIGHT.semibold,
    backgroundColor: NEURAL.surfaceContainerHigh,
    borderWidth: 1,
    borderColor: NEURAL.outlineVariant,
    borderRadius: RADIUS.lg,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
  },

  connStatus: { fontSize: FONT_SIZE.sm, fontWeight: FONT_WEIGHT.semibold, marginTop: SPACING.sm, textAlign: 'center' },
});
