import React from 'react';
import {
  ActivityIndicator,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

import type { AuthStatus } from '../../shared/core/types';
import { AppIcon, type AppIconName } from '../components/ui/AppIcon';
import { Button } from '../components/ui/Button';
import { FONT_SIZE, FONT_WEIGHT, NEURAL, RADIUS, SHADOWS, SPACING } from '../theme/colors';

interface AuthLandingScreenProps {
  authStatus: AuthStatus | null;
  loading: boolean;
  error: string;
  onSignIn: () => void;
  onContinueLocal: () => void;
  onOpenSettings: () => void;
}

const FEATURES: Array<{ icon: AppIconName; title: string; body: string }> = [
  {
    icon: 'brain',
    title: 'Memory-first RAG',
    body: 'Save high-value turns, tag them, and retrieve them across chat, wiki, graph, and agents.',
  },
  {
    icon: 'microphone-outline',
    title: 'Eva voice companion',
    body: 'Use ambient capture and wake-word flows with your signed-in memory workspace.',
  },
  {
    icon: 'cloud-check-outline',
    title: 'Cloud backup',
    body: 'Keep device storage local-first, then back up through Supabase and Google Drive.',
  },
];

export function AuthLandingScreen({
  authStatus,
  loading,
  error,
  onSignIn,
  onContinueLocal,
  onOpenSettings,
}: AuthLandingScreenProps) {
  const googleReady = Boolean(authStatus?.enabled && authStatus.google.configured);
  const backupReady = Boolean(
    authStatus?.backup.supabase_postgres_configured
      || authStatus?.backup.supabase_storage_configured
      || authStatus?.backup.google_drive_configured,
  );

  return (
    <View style={styles.root}>
      <LinearGradient
        colors={['#fff7ed', '#eef7f1', '#f8fafc']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFill}
      />
      <View style={styles.glowOne} />
      <View style={styles.glowTwo} />

      <ScrollView
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.brandRow}>
          <View style={styles.brandIcon}>
            <AppIcon name="brain" size={24} color={NEURAL.tertiaryDim} />
          </View>
          <View>
            <Text style={styles.eyebrow}>Cortex Lab</Text>
            <Text style={styles.brandSub}>Eva memory operating system</Text>
          </View>
        </View>

        <View style={styles.heroCard}>
          <Text style={styles.pill}>Auth-first workspace</Text>
          <Text style={styles.title}>Sign in once. Let Eva remember the work.</Text>
          <Text style={styles.subtitle}>
            Connect Google OAuth to unlock cloud backup while keeping your phone as the local-first source of truth.
          </Text>

          <View style={styles.actions}>
            <Button
              label={loading ? 'Checking auth...' : 'Continue with Google'}
              onPress={onSignIn}
              loading={loading}
              disabled={!googleReady || loading}
              fullWidth
              size="lg"
            />
            <Button
              label="Use local-first mode"
              onPress={onContinueLocal}
              variant="outline"
              fullWidth
              size="lg"
            />
          </View>

          {!googleReady ? (
            <View style={styles.warningBox}>
              <Text style={styles.warningTitle}>Google OAuth is not ready on this backend.</Text>
              <Text style={styles.warningBody}>
                Add the backend auth environment variables in Render or local `.env`, then reconnect.
              </Text>
              <Button label="Open Settings" onPress={onOpenSettings} variant="ghost" size="sm" />
            </View>
          ) : null}

          {error ? <Text style={styles.errorText}>{error}</Text> : null}
        </View>

        <View style={styles.statusGrid}>
          <StatusTile icon="account-check-outline" label="Google OAuth" detail={googleReady ? 'Configured' : 'Missing env'} ready={googleReady} />
          <StatusTile icon="database-outline" label="Supabase backup" detail={backupReady ? 'Ready' : 'Waiting'} ready={backupReady} />
        </View>

        <View style={styles.featureList}>
          {FEATURES.map((feature) => (
            <View key={feature.title} style={styles.featureCard}>
              <View style={styles.featureIcon}>
                <AppIcon name={feature.icon} size={18} color={NEURAL.onPrimary} />
              </View>
              <View style={styles.featureText}>
                <Text style={styles.featureTitle}>{feature.title}</Text>
                <Text style={styles.featureBody}>{feature.body}</Text>
              </View>
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

function StatusTile({
  icon,
  label,
  detail,
  ready,
}: {
  icon: AppIconName;
  label: string;
  detail: string;
  ready: boolean;
}) {
  return (
    <View style={styles.statusTile}>
      <View style={[styles.statusIcon, ready ? styles.statusIconReady : styles.statusIconPending]}>
        {ready ? (
          <AppIcon name={icon} size={17} color={NEURAL.tertiaryDim} />
        ) : (
          <ActivityIndicator size="small" color={NEURAL.primary} />
        )}
      </View>
      <Text style={styles.statusLabel}>{label}</Text>
      <Text style={styles.statusDetail}>{detail}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: NEURAL.background,
  },
  glowOne: {
    position: 'absolute',
    top: -90,
    left: -70,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: 'rgba(251, 191, 36, 0.25)',
  },
  glowTwo: {
    position: 'absolute',
    right: -80,
    bottom: -80,
    width: 260,
    height: 260,
    borderRadius: 130,
    backgroundColor: 'rgba(16, 185, 129, 0.22)',
  },
  content: {
    minHeight: '100%',
    paddingTop: Platform.OS === 'ios' ? SPACING['4xl'] : SPACING['3xl'],
    paddingHorizontal: SPACING.lg,
    paddingBottom: SPACING['4xl'],
    gap: SPACING.lg,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: SPACING.md,
  },
  brandIcon: {
    width: 52,
    height: 52,
    borderRadius: RADIUS['2xl'],
    backgroundColor: 'rgba(255,255,255,0.72)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.9)',
    alignItems: 'center',
    justifyContent: 'center',
    ...SHADOWS.lg,
  },
  eyebrow: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.extrabold,
    color: NEURAL.onSurface,
    letterSpacing: 2.6,
    textTransform: 'uppercase',
  },
  brandSub: {
    marginTop: 3,
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurfaceVariant,
  },
  heroCard: {
    borderRadius: RADIUS['3xl'],
    backgroundColor: 'rgba(255,255,255,0.78)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.9)',
    padding: SPACING.xl,
    gap: SPACING.md,
    ...SHADOWS.xl,
  },
  pill: {
    alignSelf: 'flex-start',
    borderRadius: RADIUS.full,
    backgroundColor: '#fef3c7',
    color: '#92400e',
    overflow: 'hidden',
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.extrabold,
    letterSpacing: 1.4,
    textTransform: 'uppercase',
  },
  title: {
    fontSize: FONT_SIZE['4xl'],
    lineHeight: FONT_SIZE['4xl'] * 1.05,
    fontWeight: FONT_WEIGHT.extrabold,
    color: NEURAL.onSurface,
    letterSpacing: -1.4,
  },
  subtitle: {
    fontSize: FONT_SIZE.md,
    lineHeight: FONT_SIZE.md * 1.6,
    fontWeight: FONT_WEIGHT.medium,
    color: NEURAL.onSurfaceVariant,
  },
  actions: {
    marginTop: SPACING.sm,
    gap: SPACING.sm,
  },
  warningBox: {
    gap: SPACING.sm,
    borderRadius: RADIUS['2xl'],
    borderWidth: 1,
    borderColor: '#fcd34d',
    backgroundColor: 'rgba(255,251,235,0.86)',
    padding: SPACING.md,
  },
  warningTitle: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.bold,
    color: '#92400e',
  },
  warningBody: {
    fontSize: FONT_SIZE.sm,
    lineHeight: FONT_SIZE.sm * 1.5,
    color: '#92400e',
  },
  errorText: {
    borderRadius: RADIUS.xl,
    backgroundColor: NEURAL.errorContainer,
    color: NEURAL.errorDim,
    padding: SPACING.md,
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.semibold,
  },
  statusGrid: {
    flexDirection: 'row',
    gap: SPACING.md,
  },
  statusTile: {
    flex: 1,
    borderRadius: RADIUS['2xl'],
    backgroundColor: 'rgba(255,255,255,0.76)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.9)',
    padding: SPACING.md,
    ...SHADOWS.md,
  },
  statusIcon: {
    width: 34,
    height: 34,
    borderRadius: RADIUS.full,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: SPACING.sm,
  },
  statusIconReady: { backgroundColor: NEURAL.tertiaryContainer },
  statusIconPending: { backgroundColor: NEURAL.primaryContainer },
  statusLabel: {
    fontSize: FONT_SIZE.sm,
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurface,
  },
  statusDetail: {
    marginTop: 2,
    fontSize: FONT_SIZE.xs,
    fontWeight: FONT_WEIGHT.semibold,
    color: NEURAL.onSurfaceVariant,
  },
  featureList: {
    gap: SPACING.md,
  },
  featureCard: {
    flexDirection: 'row',
    gap: SPACING.md,
    borderRadius: RADIUS['2xl'],
    backgroundColor: 'rgba(255,255,255,0.75)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.9)',
    padding: SPACING.md,
    ...SHADOWS.md,
  },
  featureIcon: {
    width: 42,
    height: 42,
    borderRadius: RADIUS.xl,
    backgroundColor: NEURAL.onSurface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureText: { flex: 1 },
  featureTitle: {
    fontSize: FONT_SIZE.base,
    fontWeight: FONT_WEIGHT.bold,
    color: NEURAL.onSurface,
  },
  featureBody: {
    marginTop: 4,
    fontSize: FONT_SIZE.sm,
    lineHeight: FONT_SIZE.sm * 1.5,
    color: NEURAL.onSurfaceVariant,
  },
});
