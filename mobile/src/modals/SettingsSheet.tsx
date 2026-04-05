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
} from 'react-native';
import Slider from '@react-native-community/slider';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { AppIcon } from '../components/ui/AppIcon';
import type { ChatSettings } from '../../shared/core/types';

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
}: SettingsSheetProps) {
  const [backendDraft, setBackendDraft] = React.useState(backendUrl);

  useEffect(() => {
    if (visible) {
      setBackendDraft(backendUrl);
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
              {(['gemini', 'local'] as const).map((p) => (
                <TouchableOpacity
                  key={p}
                  onPress={() => onUpdateSettings({ llmProvider: p })}
                  style={[styles.segBtn, settings.llmProvider === p && styles.segBtnActive]}
                >
                  <Text style={[styles.segText, settings.llmProvider === p && styles.segTextActive]}>
                    {p === 'gemini' ? 'Gemini' : 'Local'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
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
