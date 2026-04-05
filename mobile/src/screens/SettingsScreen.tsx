/**
 * SettingsScreen — Full-screen settings page
 * Replaces modal sheet interaction for stable editing and navigation.
 */
import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import Slider from '@react-native-community/slider';
import { NEURAL, SPACING, RADIUS, FONT_SIZE, FONT_WEIGHT } from '../theme/colors';
import { Button } from '../components/ui/Button';
import { TextInput } from '../components/ui/TextInput';
import { AppIcon } from '../components/ui/AppIcon';
import type { ChatSettings } from '../../shared/core/types';

interface SettingsScreenProps {
  settings: ChatSettings;
  onUpdateSettings: (s: Partial<ChatSettings>) => void;
  onBack: () => void;
  onSave: (backendUrl: string) => void;
  onTestConnection: (backendUrl: string) => void;
  testingConnection: boolean;
  connectionStatus: string;
  backendUrl: string;
}

export function SettingsScreen({
  settings,
  onUpdateSettings,
  onBack,
  onSave,
  onTestConnection,
  testingConnection,
  connectionStatus,
  backendUrl,
}: SettingsScreenProps) {
  const [backendDraft, setBackendDraft] = useState(backendUrl);
  const [maxTokensDraft, setMaxTokensDraft] = useState(String(settings.maxTokens ?? ''));

  useEffect(() => {
    setBackendDraft(backendUrl);
  }, [backendUrl]);

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
          <View style={styles.actionRow}>
            <Button
              label={testingConnection ? 'Testing…' : 'Test Connection'}
              variant="outline"
              onPress={() => onTestConnection(backendDraft)}
              disabled={testingConnection}
              loading={testingConnection}
              size="sm"
            />
            <Button
              label="Save Settings"
              onPress={() => {
                commitMaxTokensDraft();
                onSave(backendDraft);
                onBack();
              }}
              size="sm"
            />
          </View>
          {connectionStatus ? (
            <Text
              style={[
                styles.connStatus,
                { color: connectionStatus.startsWith('Connected') ? NEURAL.tertiary : NEURAL.error },
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
    backgroundColor: NEURAL.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SPACING.lg,
    paddingTop: SPACING.md,
    paddingBottom: SPACING.md,
    backgroundColor: NEURAL.surfaceContainerLow,
    borderBottomWidth: 1,
    borderBottomColor: `${NEURAL.outlineVariant}50`,
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
  connStatus: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    marginTop: SPACING.sm,
    textAlign: 'center',
  },
});
